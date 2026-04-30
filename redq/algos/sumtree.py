"""
SumTree — a binary-indexed tree for O(log N) proportional sampling by priority.

Shared by Prioritized Experience Replay (PER) and Dynamic PIToD. The leaves hold
per-transition priorities; internal nodes store the sum of their subtree.

Layout (flat numpy array of size 2 * capacity - 1):
    - indices [0, capacity - 1) are internal nodes
    - indices [capacity - 1, 2 * capacity - 1) are leaves
    - leaf i corresponds to transition index i (data_idx = tree_idx - capacity + 1)
    - parent(i) = (i - 1) // 2,  left(i) = 2 * i + 1,  right(i) = 2 * i + 2

All updates propagate to the root in O(log N). Sampling picks a value in
[0, total()) and walks the tree to the matching leaf.
"""

from typing import Tuple

import numpy as np


class SumTree:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"SumTree capacity must be positive, got {capacity}")
        self.capacity = int(capacity)
        self.tree = np.zeros(2 * self.capacity - 1, dtype=np.float64)
        self._leaf_offset = self.capacity - 1
        self.max_priority = 1.0  # used by PER to give fresh transitions optimistic priority

    def total(self) -> float:
        return float(self.tree[0])

    def update(self, data_idx: int, priority: float) -> None:
        if data_idx < 0 or data_idx >= self.capacity:
            raise IndexError(f"data_idx {data_idx} out of range [0, {self.capacity})")
        priority = float(max(priority, 0.0))
        tree_idx = data_idx + self._leaf_offset
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        # propagate delta up to the root
        parent = (tree_idx - 1) // 2
        while tree_idx != 0:
            self.tree[parent] += delta
            tree_idx = parent
            parent = (tree_idx - 1) // 2
        if priority > self.max_priority:
            self.max_priority = priority

    def update_batch(self, data_idxs: np.ndarray, priorities: np.ndarray) -> None:
        # straightforward loop; good enough at num_groups-scale (<= 200 calls per refresh)
        for data_idx, priority in zip(np.asarray(data_idxs).reshape(-1), np.asarray(priorities).reshape(-1)):
            self.update(int(data_idx), float(priority))

    def get(self, value: float) -> Tuple[int, float]:
        """Walk the tree to find the leaf whose cumulative priority covers `value`."""
        idx = 0
        while idx < self._leaf_offset:  # while still an internal node
            left = 2 * idx + 1
            right = left + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right
        data_idx = idx - self._leaf_offset
        return int(data_idx), float(self.tree[idx])

    def sample(self, batch_size: int, rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
        """Stratified sampling: split [0, total) into batch_size segments, draw one per segment."""
        total = self.total()
        if total <= 0.0:
            raise RuntimeError("SumTree.sample called with zero total priority")
        segment = total / batch_size
        idxs = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)
        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            value = rng.uniform(lo, hi)
            data_idx, priority = self.get(value)
            idxs[i] = data_idx
            priorities[i] = priority
        return idxs, priorities
