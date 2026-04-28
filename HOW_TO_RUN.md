# Dynamic PIToD — How to Run

Practical guide for the Dynamic PIToD implementation added on top of static PIToD. See [DYNAMIC_PITOD_PLAN.md](DYNAMIC_PITOD_PLAN.md) for the design rationale.

---

## 1. What was built

| File | Kind | What it does |
|---|---|---|
| [redq/algos/sumtree.py](redq/algos/sumtree.py) | new | `SumTree` — O(log N) proportional sampler shared by PER & Dynamic PIToD |
| [redq/algos/group_registry.py](redq/algos/group_registry.py) | new | `GroupRegistry` — per-experience-group scores, strikes, active flag, age |
| [redq/utils/dynamic_pitod_utils.py](redq/utils/dynamic_pitod_utils.py) | new | `DynamicPIToDController`, `H2Tracker`, TD-flip-delta scoring helpers |
| [dynamic-main-TH.py](dynamic-main-TH.py) | new | Entry script with four replay modes + SPS logging |
| [tests/test_sumtree.py](tests/test_sumtree.py) | new | 4 sampling-distribution tests (no torch needed) |
| [tests/test_group_registry.py](tests/test_group_registry.py) | new | 9 registry logic tests (no torch needed) |
| [redq/algos/core.py](redq/algos/core.py) | modified | `ReplayBuffer` gains prioritized sampling, `update_priorities`, optional SumTree writeback on `store` |
| [redq/algos/redq_sac.py](redq/algos/redq_sac.py) | modified | `REDQSACAgent.train` uses per-sample TD loss × IS-weights; PER mode writes `|TD|+ε` back |

Backups of the two modified files sit at `redq/algos/core.py.bak` and `redq/algos/redq_sac.py.bak`. To roll back either file: `mv redq/algos/core.py.bak redq/algos/core.py`.

[redq/utils/bias_utils.py](redq/utils/bias_utils.py) and [main-TH.py](main-TH.py) are **unchanged** — the original static-PIToD pipeline is fully preserved.

---

## 2. Environment setup

The code targets the existing static-PIToD environment (Python 3.8, torch 1.10+cu111, gym 0.17.3, mujoco-py 2.1.2.14). Dynamic PIToD adds **no new dependencies** beyond numpy (already present).

```bash
# activate the conda env that has torch + gym + mujoco-py
conda activate pitod   # or whatever yours is called

# sanity check
python -c "import torch, gym, mujoco_py; print('OK')"
```

On Colab, follow the existing [COLAB.md](COLAB.md) — no changes needed.

---

## 3. Four replay modes

A single flag selects the sampling strategy. The `static_pitod` mode is a label that reuses the uniform sampling path, so the static-PIToD post-hoc analysis still runs when `-evaluate_bias 1`.

| `--replay_mode` | Sampling | Priority source | Rescoring |
|---|---|---|---|
| `uniform` | uniform | — | — |
| `static_pitod` | uniform | — | post-hoc (via `-evaluate_bias 1`) |
| `per` | SumTree-prioritized | `\|TD error\|+ε` written by training loop | every update |
| `dynamic_pitod` | SumTree-prioritized | group TD-flip-delta from the controller | every `--k_refresh` env-steps |

---

## 4. CLI reference

### Legacy flags (overlap with `main-TH.py`; `-start_steps` is **dynamic-main-TH.py only**)
```
-env Hopper-v2          # env name (Hopper-v2 is the only one tested)
-seed 0
-epochs -1              # -1 = MBPO default (Hopper=125)
-steps_per_epoch 5000
-info DynPIToD          # subdir under ./runs/
-gpu_id 0
-num_q 2
-layer_norm 0           # 0/1
-layer_norm_policy 0
-hidden_sizes 128 128
-experience_group_size 5000
-start_steps 5000       # dynamic-main-TH.py only: random warmup; sets delay_update_steps when 'auto'. Lower for smoke.
-evaluate_bias 0        # 1 enables the (expensive) static-PIToD log_evaluation call
```

### Dynamic PIToD flags (new, all have defaults)
```
--replay_mode          {uniform,per,static_pitod,dynamic_pitod}   default uniform
--k_refresh            5000        env-steps between refresh cycles
--b_refresh            32          number of groups rescored per refresh
--m_strikes            3           consecutive low-score cycles before eviction
--epsilon_k            1.0         epsilon = mean - k*std (floored at 1e-4)
--pitod_alpha          0.6         priority exponent (score ** alpha)
--n_samples_per_group  64          transitions sampled per group for TD-flip-delta
--dynamic_warmup_steps 5000        skip rescoring until this many env-steps
--early_phase_steps    0           for env-steps < this, use early refresh settings if provided
--early_k_refresh      0           early-phase refresh interval; 0 keeps --k_refresh
--early_b_refresh      0           early-phase batch size; 0 keeps --b_refresh
--dynamic_pruning      1           {0,1} soft-evict low-score groups when strikes accumulate
--prune_warmup_steps   0           disable strikes / eviction until this env-step

--per_alpha            0.6
--per_beta_start       0.4
--per_beta_end         1.0
--per_beta_anneal_steps 1000000

--h2_log               1           {0,1}  log tagged-group scores for H2
--h2_tag_step          10000       env-step at which tagging happens
--h2_tag_n_groups      2           number of early groups to track (2 * 5000 = 10000 transitions)
```

---

## 5. Quick verification (no training required)

```bash
# should print "All SumTree tests passed."
python tests/test_sumtree.py

# should print "All GroupRegistry tests passed."
python tests/test_group_registry.py

# byte-compile every touched file
python -m py_compile redq/algos/sumtree.py redq/algos/group_registry.py \
    redq/algos/core.py redq/algos/redq_sac.py \
    redq/utils/dynamic_pitod_utils.py dynamic-main-TH.py analyze_dynamic_pitod_study.py

# shell syntax check for the focused screen runner
bash -n scripts/run_dynamic_pitod_screen.sh
```

---

## 6. Smoke test (short Hopper-v2 run, ~1 min on GPU)

Make sure the torch stack is active, then:

```bash
# dynamic_pitod: short run but enough steps for (1) Q updates and (2) at least one sealed group
# Default start_steps=5000 and experience_group_size=5000 exceed 2×1000 env steps — Loss* and DynPIToD/* stay 0.
python dynamic-main-TH.py -env Hopper-v2 -seed 0 -epochs 4 -steps_per_epoch 1000 \
    -info smoke --replay_mode dynamic_pitod -start_steps 500 -experience_group_size 1000 \
    --k_refresh 250 --b_refresh 8 --dynamic_warmup_steps 250 -gpu_id 0
```

Expected: `runs/smoke/.../progress.txt` shows **non-zero** `LossQ1` / `LossPi` after data exceeds `-start_steps`, and **non-zero** `DynPIToD/NumRefreshed` (or other `DynPIToD/*`) once refresh runs with sealed groups — plus `SPS` and no crash.

```bash
# per sanity
python dynamic-main-TH.py -env Hopper-v2 -seed 0 -epochs 4 -steps_per_epoch 1000 \
    -info smoke --replay_mode per -start_steps 500 -gpu_id 0

# uniform control
python dynamic-main-TH.py -env Hopper-v2 -seed 0 -epochs 4 -steps_per_epoch 1000 \
    -info smoke --replay_mode uniform -start_steps 500 -gpu_id 0
```

---

## 7. Static-equivalence check (sanity for refactor)

With refresh effectively disabled, Dynamic PIToD should reproduce uniform / static-PIToD return curves within seed noise (because all groups sit at optimistic `max_priority`):

```bash
python dynamic-main-TH.py -env Hopper-v2 -seed 0 -epochs 5 -steps_per_epoch 5000 \
    -info sanity --replay_mode dynamic_pitod --k_refresh 100000000
```

Compare `TestEpRet` to a matching `--replay_mode uniform` run.

---

## 8. Focused screen to make Dynamic PIToD stand out

The current repo now supports two light-weight algorithm changes that are useful for screening:

- **earlier / denser refresh** via `--early_phase_steps`, `--early_k_refresh`, `--early_b_refresh`
- **delayed or disabled pruning** via `--prune_warmup_steps` and `--dynamic_pruning`

Recommended Hopper-v2 screen (3 seeds first, then promote the best dynamic variant to 5 seeds):

```bash
for seed in 0 1 2; do
  python dynamic-main-TH.py -env Hopper-v2 -seed $seed -epochs 60 -steps_per_epoch 5000 \
      -info screen_uniform --replay_mode uniform

  python dynamic-main-TH.py -env Hopper-v2 -seed $seed -epochs 60 -steps_per_epoch 5000 \
      -info screen_static --replay_mode static_pitod -evaluate_bias 1

  python dynamic-main-TH.py -env Hopper-v2 -seed $seed -epochs 60 -steps_per_epoch 5000 \
      -info screen_dyn_base --replay_mode dynamic_pitod \
      --k_refresh 10000 --b_refresh 16 --dynamic_warmup_steps 10000 --m_strikes 5

  python dynamic-main-TH.py -env Hopper-v2 -seed $seed -epochs 60 -steps_per_epoch 5000 \
      -info screen_dyn_early --replay_mode dynamic_pitod \
      --k_refresh 10000 --b_refresh 16 --dynamic_warmup_steps 5000 \
      --early_phase_steps 50000 --early_k_refresh 5000 --early_b_refresh 16 \
      --m_strikes 5 --prune_warmup_steps 50000

  python dynamic-main-TH.py -env Hopper-v2 -seed $seed -epochs 60 -steps_per_epoch 5000 \
      -info screen_dyn_strong --replay_mode dynamic_pitod \
      --k_refresh 5000 --b_refresh 32 --dynamic_warmup_steps 5000 \
      --early_phase_steps 50000 --early_k_refresh 2500 --early_b_refresh 32 \
      --m_strikes 5 --prune_warmup_steps 50000

  python dynamic-main-TH.py -env Hopper-v2 -seed $seed -epochs 60 -steps_per_epoch 5000 \
      -info screen_dyn_noprune --replay_mode dynamic_pitod \
      --k_refresh 5000 --b_refresh 32 --dynamic_warmup_steps 5000 \
      --early_phase_steps 50000 --early_k_refresh 2500 --early_b_refresh 32 \
      --dynamic_pruning 0
done
```

Analyze the screen with:

```bash
python analyze_dynamic_pitod_study.py \
    --env Hopper-v2 \
    --spec uniform=screen_uniform:uniform \
    --spec static=screen_static:static_pitod \
    --spec dyn_base=screen_dyn_base:dynamic_pitod \
    --spec dyn_early=screen_dyn_early:dynamic_pitod \
    --spec dyn_strong=screen_dyn_strong:dynamic_pitod \
    --spec dyn_noprune=screen_dyn_noprune:dynamic_pitod \
    --analysis-seed 0 \
    --return-threshold 1000 \
    --output-dir figure/dynamic_screen
```

The script prints per-seed / aggregate AUC, final return, wall-clock, and return-threshold crossing stats, and saves:

- `learning_and_time.png`
- `return_vs_wallclock.png`
- `dynamic_diagnostics.png`
- `h2_<label>_seed<seed>.png` when H2 logs are present

---

## 9. Full experimental plan (from `CS6955_Project_Proposal.pdf`)

### H1 — Sample efficiency (5 seeds × 4 modes × 1M steps)

```bash
for seed in 0 1 2 3 4; do
  for mode in uniform per static_pitod dynamic_pitod; do
    python dynamic-main-TH.py -env Hopper-v2 -seed $seed \
        -epochs 200 -steps_per_epoch 5000 \
        -info H1_${mode} --replay_mode $mode
  done
done
```

**Analysis:** AUC of `TestEpRet` across seeds, plus mean over last 100 episodes. Use the same plotting helpers as `plot_main_results_pitod.py`.

### H2 — Non-stationarity of early-training influence

```bash
python dynamic-main-TH.py -env Hopper-v2 -seed 0 \
    -epochs 200 -steps_per_epoch 5000 \
    -info H2 --replay_mode dynamic_pitod \
    --h2_log 1 --h2_tag_step 10000 --h2_tag_n_groups 2
```

**Analysis:** at the end of the run, the script dumps `h2_dynamic_scores.bz2` inside the run directory. Each record is `(env_step, group_id, score, active, strikes)` for the tagged groups. Plot `score` vs `env_step` per `group_id` — H2 is supported if scores decay or turn negative as the agent learns.

```python
import bz2, pickle
with bz2.BZ2File("runs/H2/redq_sac_Hopper-v2_dynamic_pitod_s0/h2_dynamic_scores.bz2", "rb") as f:
    data = pickle.load(f)
# data["records"] is a list of dicts
```

### H3 — Wall-clock vs sample efficiency (ablation)

```bash
for k in 1000 5000 10000; do
  for b in 16 32 64; do
    python dynamic-main-TH.py -env Hopper-v2 -seed 0 \
        -epochs 200 -steps_per_epoch 5000 \
        -info H3_k${k}_b${b} --replay_mode dynamic_pitod \
        --k_refresh $k --b_refresh $b
  done
done
```

**Analysis:** compare the `SPS` column across runs. The `DynPIToD/RefreshWallclock` column isolates the cost of the refresh itself. H3 is supported if higher refresh frequencies reach a given return threshold in fewer env-steps even after accounting for wall-clock time.

---

## 10. Output layout

```
runs/
└── <info>/
    └── redq_sac_<env>_<replay_mode>/
        └── redq_sac_<env>_<replay_mode>_s<seed>/
            ├── config.json              # full hyperparameters
            ├── progress.txt             # per-epoch tabular log (read with pandas)
            └── h2_dynamic_scores.bz2    # only if --h2_log 1 + replay_mode=dynamic_pitod
```

Relevant columns in `progress.txt`:

- **Always:** `Epoch, TotalEnvInteracts, Time, SPS, EpRet*, TestEpRet*, LossQ1, LossPi`
- **Dynamic only:** `AverageDynPIToD/ScoreMean, AverageDynPIToD/ScoreStd, AverageDynPIToD/Epsilon, AverageDynPIToD/NumEvicted, AverageDynPIToD/NewlyEvicted, AverageDynPIToD/NumActive, AverageDynPIToD/NumRefreshed, AverageDynPIToD/BufferActiveFrac, AverageDynPIToD/GroupAgeMean, AverageDynPIToD/RefreshWallclock, AverageDynPIToD/ScheduleK, AverageDynPIToD/ScheduleB, AverageDynPIToD/PruningEnabled`
- **With `-evaluate_bias 1`:** `QBias*, NormQBias*, MCDisRet*` (static-PIToD post-hoc analysis)

---

## 11. Common pitfalls

- **`--dynamic_warmup_steps` vs `--k_refresh`** — refreshes now happen when enough steps have elapsed since the previous refresh, not only at fixed global multiples. If you want early refreshes, lower `--dynamic_warmup_steps` and/or use `--early_*`.
- **SumTree wall-clock** — the current `broadcast_group_to_sumtree` does `G` individual `sumtree.update` calls per group written; at `B_refresh=32, G=5000` that is 160k updates per refresh. If this shows up in `RefreshWallclock`, the optimization is to add a `SumTree.fill_range(lo, hi, priority)` method — not needed for correctness.
- **Static `main-TH.py` behavior** — entirely unchanged. If you need the published static-PIToD numbers, keep using `main-TH.py`, not `dynamic-main-TH.py`.
- **Dynamic logging** — `progress.txt` now gets Dynamic PIToD columns from epoch 0 onward, even before the first refresh. Older runs created before this change may not have the `AverageDynPIToD/*` columns.
- **PER IS-weights interact with REDQ's `utd_ratio`** — each env-step does `utd_ratio=4` updates, so PER writeback happens 4× more often than in a vanilla SAC implementation. This is intentional and consistent with the static REDQ-SAC loop.

---

## 11. Rollback

If anything goes wrong and you need to restore pristine static PIToD:

```bash
mv redq/algos/core.py.bak redq/algos/core.py
mv redq/algos/redq_sac.py.bak redq/algos/redq_sac.py
# all new files can be left in place; main-TH.py doesn't import them
```
