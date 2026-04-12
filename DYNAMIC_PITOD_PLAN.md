# Dynamic PIToD — Implementation Plan

## Context

The repo at `c:\Users\Neha Bhat\PIToD` implements **static PIToD**: a REDQ-SAC agent with turn-over-dropout (a 20-sub-network ensemble where each transition is tagged at insertion with a 50%-zero binary mask). Influence per "experience group" (5000 transitions sharing a mask-generation event) = `return_with_flipped_mask − return_with_non_flipped_mask`, computed **post-hoc at end of training** in [bias_utils.py](redq/utils/bias_utils.py). The replay buffer samples uniformly.

The CS6955 project (`CS6955_Project_Proposal.pdf`) tests the hypothesis that experience utility is **non-stationary**: a transition's influence under early-training θ does not match its influence under late-training θ, so static priorities go stale. **Dynamic PIToD** periodically rescores older groups using the *current* network parameters, broadcasts the new scores into a prioritized sampling distribution, and prunes groups that stay uninformative across multiple cycles. Three baselines for comparison: Uniform Replay, PER, Static PIToD.

**User decisions already made:**
- **Option C (Hybrid):** group-level rescoring (reuses existing `bias_utils` machinery), transition-level SumTree sampling where every transition in a group inherits its group's priority.
- **Hopper-v2 only**, **REDQ-SAC** base, **drop MinAtar**, minimal **PER** baseline.

**Why this approach:** the static codebase's influence is fundamentally a group-level signal (the mask is shared across 5000 transitions), so per-transition priorities cannot be computed independently. The hybrid strategy exposes a transition-level SumTree externally (clean math, easy IS weights, matches the proposal's framing) while internally only writing `num_groups ≈ 200` distinct values.

---

## Files

### New (5)

| File | Purpose | LOC |
|---|---|---|
| [redq/algos/sumtree.py](redq/algos/sumtree.py) | Generic `SumTree` class shared by PER and Dynamic PIToD | ~150 |
| [redq/algos/group_registry.py](redq/algos/group_registry.py) | `GroupRegistry`: per-group scores, strikes, active flag, ages | ~180 |
| [redq/utils/dynamic_pitod_utils.py](redq/utils/dynamic_pitod_utils.py) | `DynamicPIToDController`, batched group rescoring, `H2Tracker` | ~240 |
| [dynamic-main-TH.py](dynamic-main-TH.py) | New entry script: copy of `main-TH.py` + new flags + controller wiring + SPS logging | ~300 |
| [tests/test_sumtree.py](tests/test_sumtree.py) | Sampling-distribution unit test for the SumTree | ~40 |

### Modified (2)

| File | Change | LOC |
|---|---|---|
| [redq/algos/core.py](redq/algos/core.py) | `ReplayBuffer.__init__` accepts SumTree/registry/replay_mode; `store` writes priorities; `sample_batch` branches; new `update_priorities` | ~80 |
| [redq/algos/redq_sac.py](redq/algos/redq_sac.py) | `REDQSACAgent` accepts replay-mode args; `sample_data` returns `is_weights, idxs`; `train` switches to per-sample TD loss × IS weights; PER mode writes back `|TD| + ε` | ~60 |

[bias_utils.py](redq/utils/bias_utils.py) is **untouched** — `_evaluate_td_with_masks` ([bias_utils.py:238](redq/utils/bias_utils.py#L238)) is already importable as-is and operates on raw tensors.

---

## Data structures

### `SumTree` ([redq/algos/sumtree.py](redq/algos/sumtree.py))
- Flat-array binary tree, capacity = `replay_buffer.max_size` (one leaf per transition).
- Methods: `update(idx, p)`, `update_batch(idxs, ps)`, `total()`, `get(s) → (idx, p)`, `sample(batch_size, rng) → (idxs, priorities)` via stratified `total/batch_size` segments.
- **Why transition-level, not group-level:** `sample_batch=256` from a 200-leaf group tree would need a second uniform draw inside each chosen group, breaking O(log N) sampling and complicating IS weights. Transition-level keeps the math clean; we just write the same priority into all leaves of a group.

### `GroupRegistry` ([redq/algos/group_registry.py](redq/algos/group_registry.py))
Sized `num_groups = max_size // experience_group_size` (= 200 for default 1M buffer / G=5000). All numpy arrays:
- `scores: float32[num_groups]` — current dynamic priority signal
- `strikes: int32[num_groups]` — consecutive cycles below ε
- `active: bool[num_groups]` — soft-eviction flag
- `created_at: int64[num_groups]` — env-step at sealing (for stratified refresh + age logging)
- `last_refresh_at: int64[num_groups]`
- `initial_score: float32[num_groups]` — snapshot at first scoring (for H2)

Methods: `seal_group(gid, env_step, init_score)`, `update_score(gid, new_score, epsilon, M)`, `to_transition_priority(gid, alpha)`, `sample_refresh_targets(B_refresh, current_step, rng)`, `compute_epsilon(k=1.0, abs_floor=1e-4)`, `snapshot()`.

### Soft eviction (chosen, not hard delete)
- The buffer is FIFO contiguous; physical deletion isn't possible.
- Eviction = `active[g]=False` → `to_transition_priority(g)=0` → SumTree leaves zeroed → sampling probability 0.
- When the FIFO wraps and overwrites a slot in that range, `seal_group` is called again at the next group boundary, resetting `active`, `strikes`, `scores`. Natural recovery.
- Eviction only blocks *initial* sampling; evicted slots can still appear as `obs2` of an earlier transition's TD target — this is correct, not a bug.

### Group ↔ transition indexing
For group `g`: transition slots are `[g*G, (g+1)*G)` (clipped to `replay_buffer.size`). `broadcast_group_to_sumtree(g)` calls `sumtree.update_batch(np.arange(g*G, min((g+1)*G, size)), [p_g] * count)` where `p_g = max(score_g, 0) ** alpha if active else 0`.

---

## Three-stage algorithm wiring

### Stage 1 — Insertion scoring
Inside `DynamicPIToDController.on_new_transition(t)`, called immediately after `agent.store_data(...)`:
```
if replay_buffer.ptr % G == 0 and replay_buffer.size >= G:
    just_sealed_gid = ((replay_buffer.ptr - 1) // G) % num_groups
    if agent_has_warmed_up:
        score = compute_group_score_td(agent, replay_buffer, just_sealed_gid, n_samples_per_group)
    else:
        score = registry.max_priority_so_far  # optimistic init
    registry.seal_group(just_sealed_gid, env_step=t, init_score=score)
    broadcast_group_to_sumtree(...)
```
**One scoring call per 5000 env-steps**, not per step. Cost: ~2 batched forward passes.

### Stage 2 — Periodic refresh
Every `K` env-steps (default `K=5000`, ablation grid `{1000, 5000, 10000}`), `controller.refresh(t)`:
1. `gids = registry.sample_refresh_targets(B_refresh, t, rng)` — stratified, biased toward older groups via `p ∝ (t − created_at)`.
2. `new_scores = compute_group_scores_td_batch(agent, replay_buffer, gids, n_samples_per_group)` — single batched forward pass over all sampled groups, then split per-group.
3. `epsilon = registry.compute_epsilon(k=epsilon_k)`
4. For each `(gid, s)`: `registry.update_score(gid, s, epsilon, M)` → `broadcast_group_to_sumtree(gid)`.
5. Return stats dict for logging.

### Stage 3 — Pruning
- ε per refresh = `mean(scores[active]) − epsilon_k * std(scores[active])`, floored at `1e-4` to avoid degenerate early training.
- `score < ε` → `strikes += 1`; else `strikes = 0`.
- `strikes ≥ M` (default `M=3`) → `active = False` → broadcast zero priorities.
- Reseal (FIFO wrap) clears all per-group state.

---

## Influence signal

**TD-loss-flip-delta** = `mean(flip_td − non_flip_td)` over `n_samples_per_group=64` transitions sampled from the group.

- Uses current θ (the whole point of "dynamic")
- ~2 forward passes per transition, no env rollout
- Matches the "Projected Influence on TD" interpretation
- Already implemented inside [`_evaluate_td_with_masks`](redq/utils/bias_utils.py#L238) — verified to take raw tensors, return per-sample numpy arrays. Reused via direct import in `dynamic_pitod_utils.py`.

**Reserve `_return_with_flip_and_non_flip_masks` for H2 logging only** (epoch boundaries, tagged subset) — too expensive for the hot loop.

---

## Sampling integration

**CLI flag:** `--replay_mode {uniform, per, static_pitod, dynamic_pitod}`, default `uniform`.

| Mode | SumTree? | Insertion priority | Update mechanism |
|---|---|---|---|
| `uniform` | no | — | — |
| `static_pitod` | no | — | (alias for uniform; preserves `log_evaluation` post-hoc analysis) |
| `per` | yes | `max_priority` | Training loop writes `|TD error| + 1e-6` per update |
| `dynamic_pitod` | yes | TD-flip-delta (or optimistic during warmup) | `DynamicPIToDController` rescores every K steps |

**`ReplayBuffer.sample_batch` changes:**
- Branch on `replay_mode`. Non-uniform branches call `sumtree.sample(batch_size, rng)`, compute IS weights `(size * p / total) ** -beta`, normalize by `max`.
- Returned dict gains `is_weights` (ones in uniform mode, for a uniform downstream code path).
- New `update_priorities(idxs, ps)` for PER's writeback.

**`REDQSACAgent.train` changes:**
- Replace `self.mse_criterion(q_prediction_cat, y_q) * self.num_Q` ([redq_sac.py:279](redq/algos/redq_sac.py#L279)) with per-sample squared error → multiply by `is_weights` → `.mean()`.
- In PER mode: compute `td_errors_per_sample = (q_prediction_cat − y_q).abs().mean(dim=1)`, call `replay_buffer.update_priorities(idxs, td_errors.detach().cpu().numpy() + 1e-6)`.
- In Dynamic PIToD mode: no per-step writeback; the controller owns priorities.

---

## New entry script — `dynamic-main-TH.py`

Copy of `main-TH.py` with these structural diffs:

**New CLI flags:**
- `--replay_mode` (str, default `uniform`)
- `--K_refresh` (int, default 5000)
- `--B_refresh` (int, default 32)
- `--M_strikes` (int, default 3)
- `--epsilon_k` (float, default 1.0)
- `--pitod_alpha` (float, default 0.6) — sampling exponent
- `--n_samples_per_group` (int, default 64)
- `--per_alpha` (0.6), `--per_beta_start` (0.4), `--per_beta_end` (1.0)
- `--h2_tag_n_groups` (int, default 2), `--h2_log` (bool)

**After agent construction**, build `SumTree`, `GroupRegistry`, `DynamicPIToDController`, `H2Tracker`, attach to agent, record `wallclock_start = time.time()`.

**Inside env loop**, right after `agent.train(logger)` at [main-TH.py:207](main-TH.py#L207):
```
if controller is not None:
    controller.on_new_transition(t)
    if (t + 1) % K_refresh == 0 and t >= dynamic_warmup_steps:
        stats = controller.refresh(t)
        logger.store(**{f"DynPIToD/{k}": v for k, v in stats.items()})
        if h2_tracker is not None:
            h2_tracker.log(t, registry)
```

**At epoch boundary**, log `SPS = t / (time.time() - wallclock_start)`, buffer-active fraction, group age stats.

**At end of training:** `h2_tracker.dump("h2_dynamic_scores.bz2")`.

---

## Logging additions

**Per-refresh** (stored via `logger.store`, dumped at next epoch):
- `DynPIToD/ScoreMean`, `ScoreMin`, `ScoreMax`, `ScoreStd`, `Epsilon`, `MeanStrikes`, `NumEvicted`, `RefreshWallclock`

**Per-epoch:**
- `SPS` (for H3)
- `BufferActiveFrac`, `GroupAgeMean`, `GroupAgeMax`

**H2 dataset** (`H2Tracker`):
- At `h2_tag_step` (default 10000), snapshot the group_ids spanning the first `h2_tag_n_groups * G` transitions.
- Each refresh: append `(env_step, group_id, score, active)` for tagged groups.
- On shutdown: pickle + bz2 compress, mirroring `_save_information_list_for_influences` ([bias_utils.py:301](redq/utils/bias_utils.py#L301)).

---

## Reused functions / modules

| Reused function | Location | Used by |
|---|---|---|
| `_evaluate_td_with_masks(agent, obs, acts, obs_next, rews, done, masks)` | [bias_utils.py:238](redq/utils/bias_utils.py#L238) | `compute_group_score_td` (TD-flip-delta signal) |
| `_return_with_flip_and_non_flip_masks(agent, masks, env, n_eval)` | [bias_utils.py:171](redq/utils/bias_utils.py#L171) | `H2Tracker` (epoch-boundary expensive eval) |
| `agent.get_redq_q_target_no_grad(...)` | [redq_sac.py](redq/algos/redq_sac.py) | Already used by `_evaluate_td_with_masks` |
| `agent.q_net_list[i](inputs, masks, flips)` | [redq_sac.py](redq/algos/redq_sac.py) | Same |
| `agent.replay_buffer.sample_batch(batch_size=None, idxs=indices)` | [core.py:72](redq/algos/core.py#L72) | Already supports explicit indices for batched group reads |
| `_save_information_list_for_influences` (pattern only, not the function itself) | [bias_utils.py:301](redq/utils/bias_utils.py#L301) | `H2Tracker.dump` mirrors its bz2 layout |

---

## Verification

1. **Smoke test** — `python dynamic-main-TH.py --env Hopper-v2 --replay_mode dynamic_pitod --epochs 2 --steps_per_epoch 1000 --K_refresh 500`. Asserts the controller ran ≥1 refresh without crash. SumTree priorities remain non-negative; `total() > 0`.
2. **SumTree unit test** ([tests/test_sumtree.py](tests/test_sumtree.py)) — uniform priorities → near-uniform sampling; spike one leaf → that leaf samples ~proportionally more often.
3. **Static-equivalence check** — `--replay_mode dynamic_pitod --K_refresh 100000000` (refresh effectively disabled, only insertion scoring runs) should track static-PIToD return curves within seed noise.
4. **PER sanity** — `--replay_mode per` short Hopper-v2 run; assert TD errors decrease, returns ≥ uniform after 100k steps.
5. **End-to-end** — one full 1M-step Hopper-v2 seed for each mode (`uniform`, `per`, `static_pitod`, `dynamic_pitod`) before launching the 5-seed grid.
6. **H1 evaluation** — 5 seeds × 4 modes × 1M steps; AUC and last-100-episodes mean return.
7. **H2 evaluation** — plot `h2_dynamic_scores.bz2` for `dynamic_pitod`; expect tagged early groups' scores to decay over the run.
8. **H3 evaluation** — compare `SPS` across modes; ablate `K ∈ {1000, 5000, 10000}` and `B_refresh ∈ {16, 32, 64}` on a single seed.

---

## Open design defaults (documented, not blocking)

- **Insertion-score warmup:** new groups get optimistic `max_priority` until `agent.__get_current_num_data() > delay_update_steps`. Avoids noisy scores from an undertrained agent.
- **IS weights in dynamic mode:** applied (same formula as PER, beta-annealed). Sampling is non-uniform → gradient bias is real. CLI knob `--no_is_weights_dynamic` available for ablation.
- **K = G = 5000** as the natural default; ablation grid is `{1000, 5000, 10000}` per the proposal.
- **ε floor** of `1e-4` to prevent degenerate pruning when scores cluster near zero in early training.
- **Mask wrap-around correctness:** when FIFO wraps and `store` writes a new transition into a slot belonging to an evicted group, the next group-boundary trigger reseals. Verified by code reading at [core.py:60-65](redq/algos/core.py#L60).

---

## Backups before modification

Before modifying any existing file, copy it alongside its original with a `.bak` suffix so static PIToD reproducibility is recoverable via a single `mv`:

- `redq/algos/core.py` → `redq/algos/core.py.bak`
- `redq/algos/redq_sac.py` → `redq/algos/redq_sac.py.bak`

Performed via a one-time `cp` (or `copy` on Windows) at the start of implementation; backups are not checked into git — add them to `.gitignore` if needed. This is separate from git history and provides a fast local rollback path.

## File-level checklist (dependency order)

| # | Action | File | LOC | Depends on |
|---|---|---|---|---|
| 0 | **BACKUP** existing files before any modification | `redq/algos/core.py.bak`, `redq/algos/redq_sac.py.bak` | — | — |
| 1 | **NEW** `SumTree` | [redq/algos/sumtree.py](redq/algos/sumtree.py) | ~150 | — |
| 2 | **NEW** `GroupRegistry` | [redq/algos/group_registry.py](redq/algos/group_registry.py) | ~180 | — |
| 3 | **NEW** SumTree unit test | [tests/test_sumtree.py](tests/test_sumtree.py) | ~40 | 1 |
| 4 | **MOD** `ReplayBuffer` (init/store/sample/update_priorities) | [redq/algos/core.py](redq/algos/core.py) | ~80 | 0, 1, 2 |
| 5 | **MOD** `REDQSACAgent` (init/sample_data/train + per-sample loss + IS weights) | [redq/algos/redq_sac.py](redq/algos/redq_sac.py) | ~60 | 0, 4 |
| 6 | **NEW** `DynamicPIToDController` + `H2Tracker` + scoring helpers | [redq/utils/dynamic_pitod_utils.py](redq/utils/dynamic_pitod_utils.py) | ~240 | 1, 2, 5 |
| 7 | **NEW** `dynamic-main-TH.py` (entry script) | [dynamic-main-TH.py](dynamic-main-TH.py) | ~300 | 1–6 |
| 8 | **VERIFY** smoke test (item 1 above) | — | — | 7 |
| 9 | **VERIFY** static-equivalence run (item 3 above) | — | — | 8 |
| 10 | **RUN** 5-seed × 4-mode × 1M-step Hopper-v2 grid | — | — | 9 |

**Totals:** ~870 LOC new, ~140 LOC modified. `bias_utils.py` and `main-TH.py` remain unchanged so static reproducibility is preserved.
