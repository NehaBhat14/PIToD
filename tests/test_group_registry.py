"""
Logic tests for GroupRegistry that do not require torch/gym/mujoco.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from redq.algos.group_registry import GroupRegistry


def test_construction_num_groups() -> None:
    reg = GroupRegistry(buffer_capacity=100, experience_group_size=10)
    assert reg.num_groups == 10
    assert not reg.active.any()
    assert not reg.sealed.any()
    print(f"[PASS] construction: num_groups = {reg.num_groups}")


def test_group_id_for_buffer_index_wraps() -> None:
    reg = GroupRegistry(buffer_capacity=100, experience_group_size=10)
    assert reg.group_id_for_buffer_index(0) == 0
    assert reg.group_id_for_buffer_index(9) == 0
    assert reg.group_id_for_buffer_index(10) == 1
    assert reg.group_id_for_buffer_index(99) == 9
    # wrap-around: buffer_idx 100 is slot 0 again
    assert reg.group_id_for_buffer_index(100) == 0
    print("[PASS] group_id_for_buffer_index wraps correctly")


def test_seal_and_snapshot() -> None:
    reg = GroupRegistry(buffer_capacity=50, experience_group_size=10)
    reg.seal_group(0, env_step=100, init_score=0.5)
    reg.seal_group(1, env_step=200, init_score=1.5)
    reg.seal_group(2, env_step=300, init_score=-0.1)
    assert reg.num_sealed() == 3
    assert reg.num_active() == 3
    snap = reg.snapshot(current_step=400)
    assert snap["NumActive"] == 3
    assert snap["NumEvicted"] == 0
    assert abs(snap["ScoreMean"] - (0.5 + 1.5 - 0.1) / 3) < 1e-6
    print(f"[PASS] seal + snapshot: mean={snap['ScoreMean']:.3f}")


def test_eviction_after_m_strikes() -> None:
    reg = GroupRegistry(buffer_capacity=50, experience_group_size=10)
    for g in range(5):
        reg.seal_group(g, env_step=100 * g, init_score=1.0)

    # Pick a permissive epsilon so group 0 starts getting strikes for a low score.
    # strike group 0 with a score well below eps for m=3 cycles
    evicted_flags = []
    for _ in range(3):
        evicted_flags.append(reg.update_score(
            group_id=0, new_score=-999.0, epsilon=0.5, env_step=1000, m_strikes=3,
        ))
    assert evicted_flags == [False, False, True]
    assert not reg.active[0]
    assert reg.strikes[0] == 3
    print("[PASS] eviction after 3 strikes")


def test_strikes_reset_on_good_score() -> None:
    reg = GroupRegistry(buffer_capacity=50, experience_group_size=10)
    reg.seal_group(0, env_step=0, init_score=1.0)
    reg.update_score(0, -1.0, epsilon=0.5, env_step=1, m_strikes=3)  # strike
    reg.update_score(0, -1.0, epsilon=0.5, env_step=2, m_strikes=3)  # strike
    assert reg.strikes[0] == 2
    reg.update_score(0, 1.0, epsilon=0.5, env_step=3, m_strikes=3)   # recover
    assert reg.strikes[0] == 0
    assert reg.active[0]
    print("[PASS] strikes reset on good score")


def test_to_transition_priority_evicted_zero() -> None:
    reg = GroupRegistry(buffer_capacity=50, experience_group_size=10)
    reg.seal_group(0, env_step=0, init_score=4.0)
    assert abs(reg.to_transition_priority(0, alpha=0.5) - 2.0) < 1e-6  # 4 ** 0.5
    reg.active[0] = False
    assert reg.to_transition_priority(0, alpha=0.5) == 0.0
    print("[PASS] to_transition_priority: active = score**alpha, evicted = 0")


def test_sample_refresh_targets_biases_old() -> None:
    reg = GroupRegistry(buffer_capacity=100, experience_group_size=10)
    # Group 0 created long ago (very old), group 9 just now
    reg.seal_group(0, env_step=0, init_score=1.0)
    for g in range(1, 10):
        reg.seal_group(g, env_step=990 + g, init_score=1.0)
    rng = np.random.RandomState(42)
    counts = np.zeros(10, dtype=np.int64)
    n_trials = 1000
    for _ in range(n_trials):
        chosen = reg.sample_refresh_targets(b_refresh=1, current_step=1000, rng=rng)
        counts[chosen[0]] += 1
    # group 0's age weight dominates (~1000 vs ~10 for others)
    assert counts[0] > counts[1:].sum(), f"oldest group not favored: counts = {counts}"
    print(f"[PASS] sample_refresh_targets biased to old groups: counts[0]={counts[0]}/{n_trials}")


def test_compute_epsilon_uses_active_only() -> None:
    reg = GroupRegistry(buffer_capacity=50, experience_group_size=10)
    for g in range(5):
        reg.seal_group(g, env_step=g, init_score=1.0)
    reg.scores[:] = [1.0, 1.0, 1.0, 1.0, 1.0]
    eps = reg.compute_epsilon(k=1.0)
    assert abs(eps - 1.0) < 1e-6  # std is 0, so mean - 0 = 1.0
    reg.scores[0] = 5.0  # outlier
    eps2 = reg.compute_epsilon(k=0.5)
    assert eps2 > 0
    print(f"[PASS] compute_epsilon: eps={eps:.3f}, eps2={eps2:.3f}")


def test_buffer_slot_range_bounds() -> None:
    reg = GroupRegistry(buffer_capacity=100, experience_group_size=10)
    # Before FIFO fills, group 5 (slots 50..60) is out of bounds at buffer_size=40
    assert reg.buffer_slot_range(5, buffer_size=40) is None
    # At buffer_size=55, group 5 has slots 50..55
    r = reg.buffer_slot_range(5, buffer_size=55)
    assert r is not None
    assert r.tolist() == [50, 51, 52, 53, 54]
    # Fully filled: all 10 slots
    r2 = reg.buffer_slot_range(5, buffer_size=100)
    assert r2.tolist() == list(range(50, 60))
    print("[PASS] buffer_slot_range clipping")


if __name__ == "__main__":
    test_construction_num_groups()
    test_group_id_for_buffer_index_wraps()
    test_seal_and_snapshot()
    test_eviction_after_m_strikes()
    test_strikes_reset_on_good_score()
    test_to_transition_priority_evicted_zero()
    test_sample_refresh_targets_biases_old()
    test_compute_epsilon_uses_active_only()
    test_buffer_slot_range_bounds()
    print("\nAll GroupRegistry tests passed.")
