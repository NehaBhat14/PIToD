"""
Dynamic PIToD: periodic group-level influence rescoring with prioritized replay.

This module implements the three-stage loop from the CS6955 proposal on top
of the static-PIToD (turn-over-dropout) infrastructure:

  1. Insertion  : when an experience group finishes filling, score it with
                  current theta using _evaluate_td_with_masks (flip - non_flip).
  2. Refresh    : every K env steps, re-score B_refresh older groups and
                  write the new priorities back into the SumTree.
  3. Prune      : if a group stays below a dynamically-scaled epsilon for M
                  consecutive refreshes, soft-evict it by zeroing its leaves.

The group <-> transition mapping is handled by GroupRegistry; this module
focuses on (a) computing dynamic scores, (b) broadcasting them into the
SumTree, and (c) orchestrating the whole loop via DynamicPIToDController.
"""

from __future__ import annotations

import bz2
import os
import pickle
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import Tensor

from redq.algos.group_registry import GroupRegistry
from redq.algos.sumtree import SumTree
from redq.utils.bias_utils import _evaluate_td_with_masks, _return_with_flip_and_non_flip_masks


# --------------------------------------------------------------------------- #
# Score computation                                                           #
# --------------------------------------------------------------------------- #

def _fetch_group_sample_tensors(
    agent,
    group_id: int,
    registry: GroupRegistry,
    n_samples_per_group: int,
    rng: np.random.RandomState,
):
    """Sample up to n_samples_per_group transition indices from the group's slot range."""
    slot_range = registry.buffer_slot_range(group_id, agent.replay_buffer.size)
    if slot_range is None or slot_range.size == 0:
        return None
    if slot_range.size > n_samples_per_group:
        chosen = rng.choice(slot_range, size=n_samples_per_group, replace=False)
    else:
        chosen = slot_range
    batch = agent.replay_buffer.sample_batch(batch_size=None, idxs=chosen)
    device = agent.device
    return dict(
        obs=Tensor(batch['obs1']).to(device),
        obs_next=Tensor(batch['obs2']).to(device),
        acts=Tensor(batch['acts']).to(device),
        rews=Tensor(batch['rews']).unsqueeze(1).to(device),
        done=Tensor(batch['done']).unsqueeze(1).to(device),
        masks=Tensor(batch['masks']).to(device),
    )


def compute_group_score_td(
    agent,
    group_id: int,
    registry: GroupRegistry,
    n_samples_per_group: int,
    rng: np.random.RandomState,
) -> Optional[float]:
    """TD-flip-delta influence signal for a single group.

    Returns (flip_td - non_flip_td) averaged over sampled transitions,
    or None if the group has no valid transitions in the buffer.
    """
    bundle = _fetch_group_sample_tensors(agent, group_id, registry, n_samples_per_group, rng)
    if bundle is None:
        return None
    non_flip_td, flip_td = _evaluate_td_with_masks(
        agent,
        bundle['obs'], bundle['acts'], bundle['obs_next'],
        bundle['rews'], bundle['done'], bundle['masks'],
    )
    # flip_td and non_flip_td are numpy arrays [n]. A positive mean delta
    # means "flipping this group's mask hurts TD" -> the group is informative.
    return float(np.mean(flip_td - non_flip_td))


def compute_group_scores_td_batch(
    agent,
    group_ids: np.ndarray,
    registry: GroupRegistry,
    n_samples_per_group: int,
    rng: np.random.RandomState,
) -> Dict[int, float]:
    """Score a batch of groups. Currently loops per group — the inner
    _evaluate_td_with_masks already batches across the n_samples_per_group
    transitions, which dominates cost. Truly cross-group batching would
    require flattening masks across groups; left as a future optimization.
    """
    out: Dict[int, float] = {}
    for gid in group_ids:
        gid_int = int(gid)
        s = compute_group_score_td(agent, gid_int, registry, n_samples_per_group, rng)
        if s is not None:
            out[gid_int] = s
    return out


# --------------------------------------------------------------------------- #
# SumTree broadcast                                                           #
# --------------------------------------------------------------------------- #

def broadcast_group_to_sumtree(
    sumtree: SumTree,
    registry: GroupRegistry,
    group_id: int,
    buffer_size: int,
    pitod_alpha: float,
) -> None:
    """Write the group's priority into every SumTree leaf covering its slot range."""
    slot_range = registry.buffer_slot_range(group_id, buffer_size)
    if slot_range is None or slot_range.size == 0:
        return
    priority = registry.to_transition_priority(group_id, pitod_alpha)
    # SumTree.update_batch iterates, but num_groups is small (~200) and each
    # group holds <= 5000 leaves — total writes per refresh bounded by B_refresh*G.
    for data_idx in slot_range:
        sumtree.update(int(data_idx), priority)
    # also bump sumtree.max_priority so newly-stored transitions use something
    # reasonable (optimistic init). Use the un-alpha'd score to stay in the
    # same unit as what freshly-sealed groups will see.
    if priority > sumtree.max_priority:
        sumtree.max_priority = priority


# --------------------------------------------------------------------------- #
# H2 tracking                                                                 #
# --------------------------------------------------------------------------- #

class H2Tracker:
    """Tags a small subset of early-training groups and logs their dynamic
    scores over the run — evidence for the H2 non-stationarity hypothesis.
    """

    def __init__(self, tag_step: int, tag_n_groups: int) -> None:
        self.tag_step = int(tag_step)
        self.tag_n_groups = int(tag_n_groups)
        self.tagged_group_ids: Optional[List[int]] = None
        self.records: List[Dict[str, Any]] = []

    def maybe_tag(self, env_step: int, registry: GroupRegistry) -> None:
        if self.tagged_group_ids is not None or env_step < self.tag_step:
            return
        sealed = np.where(registry.sealed)[0]
        if sealed.size == 0:
            return
        # Tag the first `tag_n_groups` sealed groups (earliest created_at).
        order = np.argsort(registry.created_at[sealed])
        chosen = sealed[order][: self.tag_n_groups]
        self.tagged_group_ids = [int(g) for g in chosen]

    def log(self, env_step: int, registry: GroupRegistry) -> None:
        if self.tagged_group_ids is None:
            return
        for gid in self.tagged_group_ids:
            self.records.append(dict(
                env_step=int(env_step),
                group_id=int(gid),
                score=float(registry.scores[gid]),
                active=bool(registry.active[gid]),
                strikes=int(registry.strikes[gid]),
            ))

    def dump(self, path: str) -> None:
        payload = dict(
            tag_step=self.tag_step,
            tag_n_groups=self.tag_n_groups,
            tagged_group_ids=self.tagged_group_ids,
            records=self.records,
        )
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        with bz2.BZ2File(path, "wb") as f:
            pickle.dump(payload, f)


# --------------------------------------------------------------------------- #
# Controller                                                                  #
# --------------------------------------------------------------------------- #

class DynamicPIToDController:
    def __init__(
        self,
        agent,
        registry: GroupRegistry,
        sumtree: SumTree,
        k_refresh: int,
        b_refresh: int,
        m_strikes: int,
        epsilon_k: float,
        pitod_alpha: float,
        n_samples_per_group: int,
        warmup_steps: int,
        early_phase_steps: int = 0,
        early_k_refresh: int = 0,
        early_b_refresh: int = 0,
        pruning_enabled: bool = True,
        prune_warmup_steps: int = 0,
        rng: Optional[np.random.RandomState] = None,
        h2_tracker: Optional[H2Tracker] = None,
    ) -> None:
        self.agent = agent
        self.registry = registry
        self.sumtree = sumtree
        self.k_refresh = int(k_refresh)
        self.b_refresh = int(b_refresh)
        self.m_strikes = int(m_strikes)
        self.epsilon_k = float(epsilon_k)
        self.pitod_alpha = float(pitod_alpha)
        self.n_samples_per_group = int(n_samples_per_group)
        self.warmup_steps = int(warmup_steps)
        self.early_phase_steps = int(early_phase_steps)
        self.early_k_refresh = int(early_k_refresh)
        self.early_b_refresh = int(early_b_refresh)
        self.pruning_enabled = bool(pruning_enabled)
        self.prune_warmup_steps = int(prune_warmup_steps)
        self.rng = rng if rng is not None else np.random.RandomState(0)
        self.h2_tracker = h2_tracker
        self.last_refresh_step = -1

    def _refresh_interval(self, env_step: int) -> int:
        if self.early_phase_steps > 0 and env_step < self.early_phase_steps and self.early_k_refresh > 0:
            return self.early_k_refresh
        return self.k_refresh

    def _refresh_batch_size(self, env_step: int) -> int:
        if self.early_phase_steps > 0 and env_step < self.early_phase_steps and self.early_b_refresh > 0:
            return self.early_b_refresh
        return self.b_refresh

    def _pruning_enabled(self, env_step: int) -> bool:
        if not self.pruning_enabled:
            return False
        return env_step >= self.prune_warmup_steps

    def should_refresh(self, env_step: int) -> bool:
        if env_step < self.warmup_steps:
            return False
        interval = self._refresh_interval(env_step)
        return (env_step - self.last_refresh_step) >= interval

    def snapshot_stats(self, env_step: int) -> Dict[str, float]:
        stats = dict(
            NumRefreshed=0.0,
            NewlyEvicted=0.0,
            Epsilon=float(self.registry.compute_epsilon(self.epsilon_k)),
            RefreshWallclock=0.0,
            ScheduleK=float(self._refresh_interval(env_step)),
            ScheduleB=float(self._refresh_batch_size(env_step)),
            PruningEnabled=float(self._pruning_enabled(env_step)),
        )
        stats.update(self.registry.snapshot(env_step))
        return stats

    def maybe_refresh(self, env_step: int) -> Optional[Dict[str, float]]:
        if not self.should_refresh(env_step):
            return None
        stats = self.refresh(env_step)
        self.last_refresh_step = int(env_step)
        return stats

    # --- stage 1 ----------------------------------------------------------- #

    def on_new_transition(self, env_step: int) -> None:
        """Called immediately after replay_buffer.store(...).

        If the write just filled a group (i.e. ptr % G == 0 after the store),
        that group is now "sealed" and eligible for scoring.
        """
        rb = self.agent.replay_buffer
        G = rb.experience_group_size
        if rb.ptr % G != 0:
            return
        if rb.size < G:
            return
        # The group that just finished is the one whose last slot was (ptr - 1) % max_size
        just_sealed_buffer_idx = (rb.ptr - 1) % rb.max_size
        gid = self.registry.group_id_for_buffer_index(just_sealed_buffer_idx)

        # If we haven't trained enough yet, use optimistic init so the new
        # group starts with a competitive sampling probability.
        if env_step < self.warmup_steps or env_step <= self.agent.delay_update_steps:
            init_score = self.registry.max_priority_so_far
        else:
            score = compute_group_score_td(
                self.agent, gid, self.registry, self.n_samples_per_group, self.rng,
            )
            init_score = float(score) if score is not None else self.registry.max_priority_so_far

        self.registry.seal_group(gid, env_step=env_step, init_score=init_score)
        broadcast_group_to_sumtree(
            self.sumtree, self.registry, gid, rb.size, self.pitod_alpha,
        )

        if self.h2_tracker is not None:
            self.h2_tracker.maybe_tag(env_step, self.registry)

    # --- stage 2 + 3 ------------------------------------------------------- #

    def refresh(self, env_step: int) -> Dict[str, float]:
        start = time.time()
        rb = self.agent.replay_buffer
        b_refresh = self._refresh_batch_size(env_step)
        pruning_enabled = self._pruning_enabled(env_step)
        gids = self.registry.sample_refresh_targets(b_refresh, env_step, self.rng)
        if gids.size == 0:
            return self.snapshot_stats(env_step)

        scores = compute_group_scores_td_batch(
            self.agent, gids, self.registry, self.n_samples_per_group, self.rng,
        )
        # Temporarily write new scores so epsilon reflects the refreshed state
        for gid, s in scores.items():
            self.registry.scores[gid] = s
        epsilon = self.registry.compute_epsilon(self.epsilon_k)

        num_newly_evicted = 0
        for gid, s in scores.items():
            evicted_now = self.registry.update_score(
                gid, s, epsilon, env_step, self.m_strikes,
                pruning_enabled=pruning_enabled,
            )
            if evicted_now:
                num_newly_evicted += 1
            # broadcast (this handles both active-with-new-score and just-evicted cases)
            broadcast_group_to_sumtree(
                self.sumtree, self.registry, gid, rb.size, self.pitod_alpha,
            )

        wallclock = time.time() - start
        stats = dict(
            NumRefreshed=len(scores),
            NewlyEvicted=num_newly_evicted,
            Epsilon=float(epsilon),
            RefreshWallclock=float(wallclock),
            ScheduleK=float(self._refresh_interval(env_step)),
            ScheduleB=float(b_refresh),
            PruningEnabled=float(pruning_enabled),
        )
        stats.update(self.registry.snapshot(env_step))

        if self.h2_tracker is not None:
            self.h2_tracker.log(env_step, self.registry)
        return stats
