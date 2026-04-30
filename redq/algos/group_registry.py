"""
GroupRegistry — per-experience-group state for Dynamic PIToD.

An "experience group" is the static-PIToD unit (default 5000 consecutive
transitions sharing a mask-generation event). The registry tracks dynamic
priority scores, consecutive-low-score strike counts, soft-eviction flags,
creation timestamps, and refresh bookkeeping.

Group ids are `buffer_index // experience_group_size`, modulo `num_groups`
so wrap-around of the underlying FIFO replay buffer recycles group slots.
"""

from typing import Dict, Optional

import numpy as np


class GroupRegistry:
    def __init__(
        self,
        buffer_capacity: int,
        experience_group_size: int,
        initial_priority: float = 1.0,
    ) -> None:
        if buffer_capacity % experience_group_size != 0:
            # not fatal — last group is just smaller — but warn-quiet via ceil
            num_groups = (buffer_capacity + experience_group_size - 1) // experience_group_size
        else:
            num_groups = buffer_capacity // experience_group_size
        self.num_groups = int(num_groups)
        self.experience_group_size = int(experience_group_size)
        self.buffer_capacity = int(buffer_capacity)

        self.scores = np.zeros(self.num_groups, dtype=np.float32)
        self.strikes = np.zeros(self.num_groups, dtype=np.int32)
        self.active = np.zeros(self.num_groups, dtype=bool)
        self.sealed = np.zeros(self.num_groups, dtype=bool)
        self.created_at = np.full(self.num_groups, -1, dtype=np.int64)
        self.last_refresh_at = np.full(self.num_groups, -1, dtype=np.int64)
        self.initial_score = np.zeros(self.num_groups, dtype=np.float32)

        self.max_priority_so_far = float(initial_priority)

    def group_id_for_buffer_index(self, buffer_idx: int) -> int:
        return (buffer_idx // self.experience_group_size) % self.num_groups

    def seal_group(self, group_id: int, env_step: int, init_score: float) -> None:
        """Called when a new experience group finishes filling.

        Resets all per-group state — so FIFO wrap-around naturally recycles the slot.
        """
        self.scores[group_id] = float(init_score)
        self.strikes[group_id] = 0
        self.active[group_id] = True
        self.sealed[group_id] = True
        self.created_at[group_id] = int(env_step)
        self.last_refresh_at[group_id] = int(env_step)
        self.initial_score[group_id] = float(init_score)
        if init_score > self.max_priority_so_far:
            self.max_priority_so_far = float(init_score)

    def update_score(
        self,
        group_id: int,
        new_score: float,
        epsilon: float,
        env_step: int,
        m_strikes: int,
        pruning_enabled: bool = True,
    ) -> bool:
        """Write back a refreshed score. Returns True iff the group became evicted this call."""
        new_score = float(new_score)
        self.scores[group_id] = new_score
        self.last_refresh_at[group_id] = int(env_step)
        if new_score > self.max_priority_so_far:
            self.max_priority_so_far = new_score

        newly_evicted = False
        if self.active[group_id]:
            if not pruning_enabled:
                self.strikes[group_id] = 0
                return False
            if new_score < epsilon:
                self.strikes[group_id] += 1
                if self.strikes[group_id] >= m_strikes:
                    self.active[group_id] = False
                    newly_evicted = True
            else:
                self.strikes[group_id] = 0
        return newly_evicted

    def to_transition_priority(self, group_id: int, alpha: float) -> float:
        """Map a group's score to the priority value all its transitions should carry."""
        if not self.active[group_id]:
            return 0.0
        s = max(float(self.scores[group_id]), 0.0)
        return s ** float(alpha) if s > 0.0 else 0.0

    def sample_refresh_targets(
        self,
        b_refresh: int,
        current_step: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """Stratified-ish sampling biased toward older active groups.

        Weight proportional to (current_step - created_at). Falls back to uniform
        over active groups if age weights are degenerate.
        """
        active_ids = np.where(self.active & self.sealed)[0]
        if active_ids.size == 0:
            return np.empty(0, dtype=np.int64)
        n = min(int(b_refresh), active_ids.size)

        ages = np.maximum(current_step - self.created_at[active_ids], 1).astype(np.float64)
        weight_sum = ages.sum()
        if weight_sum <= 0.0 or not np.isfinite(weight_sum):
            chosen = rng.choice(active_ids, size=n, replace=False)
        else:
            probs = ages / weight_sum
            chosen = rng.choice(active_ids, size=n, replace=False, p=probs)
        return chosen.astype(np.int64)

    def compute_epsilon(self, k: float = 1.0, abs_floor: float = 1e-4) -> float:
        """ε = mean(active_scores) - k*std(active_scores), with a small absolute floor."""
        mask = self.active & self.sealed
        if not mask.any():
            return abs_floor
        vals = self.scores[mask]
        eps = float(vals.mean() - k * vals.std())
        return max(eps, abs_floor)

    def num_active(self) -> int:
        return int((self.active & self.sealed).sum())

    def num_sealed(self) -> int:
        return int(self.sealed.sum())

    def snapshot(self, current_step: int) -> Dict[str, float]:
        mask = self.active & self.sealed
        if not mask.any():
            return dict(
                ScoreMean=0.0, ScoreMin=0.0, ScoreMax=0.0, ScoreStd=0.0,
                MeanStrikes=0.0, NumEvicted=int((~self.active & self.sealed).sum()),
                NumActive=0, BufferActiveFrac=0.0,
                GroupAgeMean=0.0, GroupAgeMax=0.0,
            )
        vals = self.scores[mask]
        ages = (current_step - self.created_at[mask]).astype(np.float64)
        return dict(
            ScoreMean=float(vals.mean()),
            ScoreMin=float(vals.min()),
            ScoreMax=float(vals.max()),
            ScoreStd=float(vals.std()),
            MeanStrikes=float(self.strikes[mask].mean()),
            NumEvicted=int((~self.active & self.sealed).sum()),
            NumActive=int(mask.sum()),
            BufferActiveFrac=float(mask.sum() / max(self.num_sealed(), 1)),
            GroupAgeMean=float(ages.mean()),
            GroupAgeMax=float(ages.max()),
        )

    def buffer_slot_range(self, group_id: int, buffer_size: int) -> Optional[np.ndarray]:
        """Return the array of buffer indices currently holding transitions for this group.

        Returns None if the group's slot range lies entirely beyond `buffer_size`
        (i.e. the FIFO hasn't filled that far yet).
        """
        start = group_id * self.experience_group_size
        end = min(start + self.experience_group_size, self.buffer_capacity)
        if start >= buffer_size:
            return None
        end = min(end, buffer_size)
        if end <= start:
            return None
        return np.arange(start, end, dtype=np.int64)
