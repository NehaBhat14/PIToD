# Uniform vs PER Experiment Guide
**CS6955 Project — Dynamic PIToD | Hopper-v2**

This document is a complete, standalone guide to running and analysing the
Uniform Replay vs PER (Prioritized Experience Replay) comparison that forms
**Experiment 1 (H1)** of the project proposal.

---

## Table of Contents

1. [What We Are Testing](#1-what-we-are-testing)
2. [How the Code Is Structured](#2-how-the-code-is-structured)
3. [Files We Created](#3-files-we-created)
4. [Environment Setup](#4-environment-setup)
5. [Step-by-Step: Running the Experiment](#5-step-by-step-running-the-experiment)
6. [What Happens Inside Each Run](#6-what-happens-inside-each-run)
7. [Output Structure](#7-output-structure)
8. [Analysing and Plotting Results](#8-analysing-and-plotting-results)
9. [Reading the Figures](#9-reading-the-figures)
10. [Troubleshooting](#10-troubleshooting)
11. [Key Hyperparameters Reference](#11-key-hyperparameters-reference)

---

## 1. What We Are Testing

### The core question
Does **Prioritized Experience Replay (PER)** — which samples transitions
proportional to their TD error magnitude — lead to better sample efficiency
than **Uniform Replay** on the Hopper-v2 continuous control task?

### Why this matters for the proposal
This is **Hypothesis H1** from the project proposal:

> *"Dynamic influence curation will yield a more sample-efficient replay
> distribution, achieving higher cumulative rewards in fewer environment steps
> compared to Uniform Replay and PER."*

Before we can claim Dynamic PIToD beats PER, we need a clean Uniform vs PER
baseline to establish what PER itself achieves. These two runs are also the
cheapest of the four modes to compute (no group rescoring overhead).

### The two modes
| Mode | How it samples | Priority source | Extra cost |
|---|---|---|---|
| `uniform` | Uniformly at random from the buffer | None | None |
| `per` | Proportional to `(|TD error| + ε)^α` | Written back after every gradient step | SumTree lookup + writeback |

### Evaluation metrics (from the proposal)
- **AUC** — area under the learning curve (return vs env steps). Higher = more sample-efficient.
- **Final performance** — mean test return over the last 10% of training. Higher = better asymptotic policy.
- **SPS** (steps per second) — wall-clock throughput. Lower for PER due to the writeback overhead.

---

## 2. How the Code Is Structured

The entire codebase builds on **REDQ-SAC**, a Soft Actor-Critic variant with a
multi-Q ensemble and a high update-to-data ratio. Dynamic PIToD was added on
top of this base without modifying the original `main-TH.py`.

### The relevant files (you do not need to edit any of these)

```
PIToD/
├── dynamic-main-TH.py          ← entry script, supports all 4 replay modes
├── redq/
│   ├── algos/
│   │   ├── core.py             ← ReplayBuffer: stores transitions, samples batches
│   │   ├── redq_sac.py         ← REDQSACAgent: networks, training loop, PER writeback
│   │   └── sumtree.py          ← SumTree: O(log N) priority-proportional sampler
│   └── utils/
│       └── logx.py             ← EpochLogger: writes progress.txt
```

### How Uniform works (inside the code)
In `core.py`, `ReplayBuffer.sample_batch()` checks the mode:
```python
# uniform path — pure random
idxs = np.random.randint(0, self.size, size=batch_size)
is_weights = np.ones(len(idxs), dtype=np.float32)   # all weights = 1
```
The IS weights are all 1.0, so the Q-loss is a plain mean-squared-error.

### How PER works (inside the code)

**Step 1 — Sampling.** `sample_batch()` takes the prioritized path:
```python
idxs, priorities = self.sumtree.sample(batch_size, rng)
probs      = priorities / sumtree.total()
is_weights = (self.size * probs) ** (-beta)     # importance-sampling correction
is_weights = is_weights / is_weights.max()      # normalise so max weight = 1
```
Transitions with higher TD error are more likely to be sampled. IS weights
correct for the resulting bias so gradient updates remain unbiased in
expectation.

**Step 2 — Training.** In `redq_sac.py`, both Q-loss and policy loss are
multiplied by the IS weights before averaging:
```python
q_loss_all   = (per_sample_q_loss * is_weights_tensor).mean() * self.num_Q
policy_loss  = (per_sample_policy * is_weights_tensor).mean()
```

**Step 3 — Priority writeback.** After every gradient step:
```python
if self.replay_mode == "per" and self.sumtree is not None:
    td_abs = (q_prediction_cat - y_q).abs().mean(dim=1).detach().cpu().numpy()
    self.replay_buffer.update_priorities(batch_idxs, td_abs)
```
`update_priorities` applies the PER formula `(|td| + ε)^α` and writes the
result back into the SumTree leaf for each sampled transition. The next
sample will reflect these updated priorities immediately.

### The β annealing schedule
PER's IS-weight exponent β starts at 0.4 (weak correction, letting high-priority
samples dominate early) and anneals linearly to 1.0 (full correction) over 1M
steps. This is the standard Schaul et al. 2015 schedule:
```python
beta = beta_start + (steps_seen / anneal_steps) * (beta_end - beta_start)
     = 0.4        + (t / 1_000_000)             * 0.6
```

---

## 3. Files We Created

Two new files were added to the repo specifically for this experiment:

### `run_uniform_per.sh`
Shell script that runs **10 training jobs in sequence**:
- 5 seeds (0–4) of Uniform Replay
- 5 seeds (0–4) of PER

Each job trains for 200 epochs × 5000 steps = **1 million environment steps**.
All logs go to the `logs/` directory. Results go to `runs/H1/`.

### `compare_uniform_per.py`
Python analysis script that:
1. Finds all `progress.txt` files under `runs/H1/`
2. Parses mode and seed from the directory name
3. Produces three PDF figures in `figure/`
4. Prints a numeric summary table to the console

---

## 4. Environment Setup

```bash
# activate the conda environment that has torch + gym + mujoco-py
conda activate pitod    # replace with your actual env name

# confirm the stack is working
python -c "import torch, gym, mujoco_py; print('Environment OK')"
```

If you are on **Google Colab**, complete the setup in `COLAB.md` first, then
return here. No new dependencies are needed — Dynamic PIToD only uses numpy,
which is already installed.

---

## 5. Step-by-Step: Running the Experiment

### Step 1 — Sanity checks (no training, ~5 seconds)

```bash
cd /path/to/PIToD

# unit tests for the SumTree and GroupRegistry data structures
python -m tests.test_sumtree
python -m tests.test_group_registry

# syntax-check all relevant files
python -m py_compile \
    redq/algos/sumtree.py \
    redq/algos/group_registry.py \
    redq/algos/core.py \
    redq/algos/redq_sac.py \
    dynamic-main-TH.py \
    compare_uniform_per.py
```

Expected output:
```
[PASS] uniform-priorities test: max dev = ...
[PASS] spiked-priority test: ...
[PASS] update_batch + total: ...
[PASS] zero-priority eviction: ...
All SumTree tests passed.
All GroupRegistry tests passed.
```
If any test fails, stop here and check your installation.

---

### Step 2 — Smoke test (2 epochs each, ~2 minutes total)

This confirms both modes run end-to-end without crashing and write a
correctly-formatted `progress.txt`.

```bash
mkdir -p logs

# uniform smoke
python dynamic-main-TH.py \
    -env Hopper-v2 -seed 0 \
    -epochs 2 -steps_per_epoch 1000 \
    -info smoke --replay_mode uniform --h2_log 0

# per smoke
python dynamic-main-TH.py \
    -env Hopper-v2 -seed 0 \
    -epochs 2 -steps_per_epoch 1000 \
    -info smoke --replay_mode per --h2_log 0
```

Verify the files were created and have the right columns:
```bash
head -1 runs/smoke/redq_sac_Hopper-v2_uniform/redq_sac_Hopper-v2_uniform_s0/progress.txt
head -1 runs/smoke/redq_sac_Hopper-v2_per/redq_sac_Hopper-v2_per_s0/progress.txt
```

You should see a tab-separated header containing:
`Epoch`, `TotalEnvInteracts`, `Time`, `SPS`, `ReplayMode`,
`AverageTestEpRet`, `StdTestEpRet`, `MaxTestEpRet`, `MinTestEpRet`, ...

Run a quick Python check to confirm PER is different from Uniform:
```bash
python -c "
import pandas as pd
u = pd.read_csv('runs/smoke/redq_sac_Hopper-v2_uniform/redq_sac_Hopper-v2_uniform_s0/progress.txt', sep='\t')
p = pd.read_csv('runs/smoke/redq_sac_Hopper-v2_per/redq_sac_Hopper-v2_per_s0/progress.txt', sep='\t')
print('Uniform  mode:', u['ReplayMode'].iloc[0], '  SPS:', round(u['SPS'].mean(), 1))
print('PER      mode:', p['ReplayMode'].iloc[0], '  SPS:', round(p['SPS'].mean(), 1))
"
```
PER's SPS should be slightly lower than Uniform's (priority writeback overhead).

---

### Step 3 — Full experiment

```bash
chmod +x run_uniform_per.sh
./run_uniform_per.sh
```

The script runs all 10 jobs sequentially and saves live logs:

```
logs/uniform_s0.log  ...  logs/uniform_s4.log
logs/per_s0.log      ...  logs/per_s4.log
```

**To use a different GPU:**
```bash
GPU_ID=1 ./run_uniform_per.sh
```

**To run seeds in parallel** (faster if you have GPU memory to spare):
```bash
mkdir -p logs
for seed in 0 1 2 3 4; do
  nohup python dynamic-main-TH.py \
    -env Hopper-v2 -seed $seed -epochs 200 -info H1 \
    --replay_mode uniform --h2_log 0 \
    > logs/uniform_s${seed}.log 2>&1 &
done
wait   # wait for all uniform seeds to finish

for seed in 0 1 2 3 4; do
  nohup python dynamic-main-TH.py \
    -env Hopper-v2 -seed $seed -epochs 200 -info H1 \
    --replay_mode per --h2_log 0 \
    > logs/per_s${seed}.log 2>&1 &
done
wait
```

**Estimated wall-clock time:**
- ~500 SPS on a modern GPU → 1M steps ÷ 500 SPS ≈ **~2 hours per seed**
- 10 seeds sequential → **~20 hours total**
- 5 seeds parallel (both modes at once) → **~4 hours**

**Monitor progress while running:**
```bash
# watch the live log of one run
tail -f logs/uniform_s0.log

# check how many epochs have finished across all 10 runs
grep -c "" runs/H1/**/*/progress.txt 2>/dev/null

# see the latest test return for one seed
awk -F'\t' 'NR==1{for(i=1;i<=NF;i++) if($i=="AverageTestEpRet") col=i}
            NR>1{print NR-1, $col}' \
    runs/H1/redq_sac_Hopper-v2_uniform/redq_sac_Hopper-v2_uniform_s0/progress.txt \
    | tail -5
```

---

## 6. What Happens Inside Each Run

Understanding this helps when things go wrong.

### Phase 1: Random exploration (steps 0–5000)
```
delay_update_steps = start_steps = 5000   (default)
```
The agent takes random actions and stores transitions in the buffer. No
gradient updates happen. For PER, every stored transition gets the maximum
current priority so it will be sampled at least once.

### Phase 2: Training begins (step 5001 onwards)
At each environment step, `agent.train()` is called. Because `utd_ratio = 4`,
**4 gradient updates** happen per step:
1. Sample a batch of 256 transitions (uniform random, or SumTree-proportional for PER)
2. Compute Q-targets using the target networks
3. Compute per-sample TD error: `(q_pred - y_q)²`
4. Weight by IS weights, average, backprop → update Q networks
5. **[PER only]** Write `|TD error| + ε` back to the SumTree for each sampled transition
6. Every 20 updates: update the policy network and entropy coefficient α

### Phase 3: Epoch boundary (every 5000 steps)
- Run 10 deterministic test episodes in a separate environment
- Log `AverageTestEpRet` (the number that matters for H1)
- Write one row to `progress.txt`
- Print a summary table to stdout

---

## 7. Output Structure

After `./run_uniform_per.sh` finishes:

```
runs/H1/
├── redq_sac_Hopper-v2_uniform/
│   ├── redq_sac_Hopper-v2_uniform_s0/
│   │   ├── config.json      ← full hyperparameters (useful for reproducibility)
│   │   └── progress.txt     ← 200 rows, one per epoch
│   ├── redq_sac_Hopper-v2_uniform_s1/
│   ├── redq_sac_Hopper-v2_uniform_s2/
│   ├── redq_sac_Hopper-v2_uniform_s3/
│   └── redq_sac_Hopper-v2_uniform_s4/
└── redq_sac_Hopper-v2_per/
    ├── redq_sac_Hopper-v2_per_s0/
    │   ├── config.json
    │   └── progress.txt
    ├── redq_sac_Hopper-v2_per_s1/
    ...
```

### Key columns in `progress.txt`

| Column | Meaning |
|---|---|
| `Epoch` | Epoch number (0–199) |
| `TotalEnvInteracts` | Environment steps so far (x-axis for plots) |
| `AverageTestEpRet` | Mean return across 10 test episodes — **main metric** |
| `StdTestEpRet` | Std of those 10 test episodes |
| `SPS` | Steps per second — throughput metric for H3 |
| `ReplayMode` | The mode string (`uniform` or `per`) — sanity check |
| `AverageLossQ1` | Q-network loss — should decrease over training |
| `AverageAlpha` | SAC entropy coefficient — should stabilise |

---

## 8. Analysing and Plotting Results

Once all runs are done:

```bash
python compare_uniform_per.py
```

This will:
1. Scan `runs/H1/` for all `progress.txt` files
2. Parse the mode (`uniform` or `per`) and seed from each folder name
3. Print a table to the terminal
4. Save three PDF figures to `figure/`

**Custom paths (optional):**
```bash
python compare_uniform_per.py --data_dir runs/H1 --out_dir figure
```

**Console output will look like:**
```
Loading runs from: runs/H1

Runs found:
  mode=uniform  seed=0  epochs=200
  mode=uniform  seed=1  epochs=200
  ...
  mode=per      seed=4  epochs=200

Generating figures → figure/

  Saved: figure/H1_learning_curves.pdf
  Saved: figure/H1_auc_final.pdf
  Saved: figure/H1_sps.pdf

====================================================
  Mode          AUC mean   AUC std   Final mean  Final std
----------------------------------------------------
  Uniform          1842.3      312.1      2104.7      289.4
  PER              2015.6      198.2      2241.3      175.8
====================================================

  SPS (Steps per Second):
  Mode            Median        Std
  ----------------------------------
  Uniform          487.3       12.4
  PER              441.6       18.9
```

---

## 9. Reading the Figures

### `H1_learning_curves.pdf`
- **X-axis:** Environment steps (0 to 1M)
- **Y-axis:** `AverageTestEpRet` — mean return from 10 deterministic test episodes
- **Solid line:** Mean across 5 seeds
- **Shaded band:** ± 1 standard deviation across seeds
- **What to look for:** Does PER (red) rise faster than Uniform (blue)?
  Does it reach a higher asymptote? A higher, earlier-rising curve supports H1.

### `H1_auc_final.pdf`
Two bar charts side by side:
- **Left — AUC:** Total area under the return curve, normalised by step range.
  Captures both speed of learning and final level.
- **Right — Final Performance:** Mean `AverageTestEpRet` over the last 10%
  of training (last 20 epochs = last 100K steps). Purely asymptotic.
- Error bars are ± 1 std across seeds.

### `H1_sps.pdf`
- **Bar height:** Median SPS across all seeds and epochs for each mode
- PER should be ~5–15% slower than Uniform due to the SumTree priority
  writeback happening 4 times per environment step (utd_ratio=4)
- If bars are identical, PER's writeback is not firing — see Troubleshooting

---

## 10. Troubleshooting

### `KeyError: 'Hopper-v2'` immediately on startup
**Cause:** The agent looks up the SAC entropy target in a hardcoded dictionary
at `redq_sac.py:121`. Only `v2` names are in the dictionary.
```python
# This dict is what the code checks:
mbpo_target_entropy_dict = {
    'Hopper-v2': -1, 'HalfCheetah-v2': -3, 'Walker2d-v2': -3, ...
}
```
**Fix:** Make sure you are passing `-env Hopper-v2` (lowercase v, number 2).
Any typo — `Hopper-V2`, `Hopper-v3`, `hopper-v2` — will cause this crash.

---

### `AverageTestEpRet` column not found in `compare_uniform_per.py`
**Cause:** The logger in `logx.py` prepends `"Average"` to the key name when
`with_min_and_max=True` is used. So `log_tabular('TestEpRet', with_min_and_max=True)`
writes the column `AverageTestEpRet`, not `TestEpRet`.
If someone ran the old `main-TH.py` instead of `dynamic-main-TH.py`, the
column format may differ.
**What happens:** The compare script detects this and falls back to `TestEpRet`
automatically — you will see an `[INFO]` message and the script continues.
**Fix:** Always use `dynamic-main-TH.py` for both modes.

---

### Run crashes at or just after step 5000
**Cause:** The first 5000 steps collect random data only (`num_update = 0`).
At step 5001 training begins — `utd_ratio = 4` gradient updates fire at once,
including PER's SumTree writeback. If there is a shape mismatch, device error,
or SumTree issue it will surface here, not at startup.
**Fix:** Check the log file for the actual Python traceback:
```bash
grep -A 10 "Traceback" logs/per_s0.log
```
Common causes: wrong `gpu_id` (device mismatch), out-of-memory (reduce
`-hidden_sizes` or `batch_size`).

---

### PER and Uniform have identical SPS
**Cause:** Both conditions must be true for the PER writeback to fire:
```python
if self.replay_mode == "per" and self.sumtree is not None:
```
If either is false — for example because `--replay_mode` was not passed and
defaulted to `uniform` — no writeback happens and SPS will be the same.
**Fix:** Check the `ReplayMode` column in the `progress.txt` file:
```bash
awk -F'\t' 'NR==1{for(i=1;i<=NF;i++) if($i=="ReplayMode") col=i}
            NR==2{print "Mode actually used:", $col}' \
    runs/H1/redq_sac_Hopper-v2_per/redq_sac_Hopper-v2_per_s0/progress.txt
```
It must print `per`. If it prints `uniform`, the wrong flag was used.

---

### `No progress.txt files found` from compare script
**Cause:** The script scans `runs/H1/` by default. If runs were launched with a
different `-info` flag, they landed in a different folder.
**Fix:** Pass the correct path:
```bash
python compare_uniform_per.py --data_dir runs/<your_info_tag>
```

---

## 11. Key Hyperparameters Reference

These are the values used in both modes unless overridden. They match the
REDQ-SAC defaults from `dynamic-main-TH.py`.

| Parameter | Value | What it controls |
|---|---|---|
| `epochs` | 200 | Number of training epochs |
| `steps_per_epoch` | 5000 | Environment steps per epoch → 1M total |
| `batch_size` | 256 | Transitions sampled per gradient update |
| `utd_ratio` | 4 | Gradient updates per environment step |
| `num_Q` | 2 | Number of Q-networks in the ensemble |
| `num_min` | 2 | Q-networks used for target (min of all) |
| `lr` | 3e-4 | Learning rate for all networks |
| `gamma` | 0.99 | Discount factor |
| `start_steps` | 5000 | Random exploration before training begins |
| `replay_size` | 1.51M | Maximum buffer capacity |
| `experience_group_size` | 5000 | Transitions per ToD mask group |
| **PER only** | | |
| `per_alpha` | 0.6 | Priority exponent: `p = (|td| + ε)^α` |
| `per_beta_start` | 0.4 | IS-weight exponent at step 0 |
| `per_beta_end` | 1.0 | IS-weight exponent at step 1M (full correction) |
| `per_beta_anneal_steps` | 1,000,000 | Linear annealing horizon for β |

---

*Generated for CS6955 Advanced AI, Spring 2026 — PIToD project.*
