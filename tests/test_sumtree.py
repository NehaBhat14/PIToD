"""
Sampling-distribution unit test for redq.algos.sumtree.SumTree.

Run with:
    python -m tests.test_sumtree
or
    pytest tests/test_sumtree.py
"""

import os
import sys

import numpy as np

# Allow running as a script from the repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from redq.algos.sumtree import SumTree


def test_uniform_priorities_sample_uniformly() -> None:
    capacity = 100
    tree = SumTree(capacity)
    for i in range(capacity):
        tree.update(i, 1.0)

    rng = np.random.RandomState(0)
    n_samples = 20000
    idxs, _ = tree.sample(n_samples, rng)
    counts = np.bincount(idxs, minlength=capacity)
    expected = n_samples / capacity
    max_abs_dev = float(np.max(np.abs(counts - expected)))
    # very loose tolerance — stratified sampling already enforces near-uniform
    assert max_abs_dev < expected * 0.5, f"uniform test failed, max dev {max_abs_dev}"
    print(f"[PASS] uniform-priorities test: max dev = {max_abs_dev:.1f} (expected ~{expected})")


def test_spiked_priority_dominates_sampling() -> None:
    capacity = 100
    tree = SumTree(capacity)
    for i in range(capacity):
        tree.update(i, 1.0)
    tree.update(50, 100.0)  # index 50 is ~100x more likely

    rng = np.random.RandomState(1)
    n_samples = 20000
    idxs, _ = tree.sample(n_samples, rng)
    counts = np.bincount(idxs, minlength=capacity)
    # expected share for idx 50 ≈ 100 / (99 + 100) ≈ 0.5025
    expected_50 = n_samples * 100.0 / (99.0 + 100.0)
    diff = abs(counts[50] - expected_50)
    assert diff < expected_50 * 0.1, f"spike test failed: got {counts[50]}, expected ~{expected_50}"
    print(f"[PASS] spiked-priority test: idx 50 got {counts[50]} samples (expected ~{expected_50:.0f})")


def test_total_and_update_batch() -> None:
    tree = SumTree(8)
    tree.update_batch(np.arange(8), np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]))
    assert abs(tree.total() - 3.6) < 1e-9, f"total() = {tree.total()}"
    print(f"[PASS] update_batch + total: {tree.total():.4f}")


def test_zero_priority_not_sampled() -> None:
    tree = SumTree(10)
    for i in range(10):
        tree.update(i, 1.0)
    tree.update(3, 0.0)  # evict

    rng = np.random.RandomState(2)
    idxs, _ = tree.sample(5000, rng)
    assert (idxs == 3).sum() == 0, "evicted index should never be sampled"
    print("[PASS] zero-priority eviction: idx 3 never sampled")


if __name__ == "__main__":
    test_uniform_priorities_sample_uniformly()
    test_spiked_priority_dominates_sampling()
    test_total_and_update_batch()
    test_zero_priority_not_sampled()
    print("\nAll SumTree tests passed.")
