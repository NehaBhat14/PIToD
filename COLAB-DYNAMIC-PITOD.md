# Google Colab — A100 + Dynamic PIToD

Assumes **Colab with an A100 GPU** (Runtime → Change runtime type → **A100** where available). Setup matches [COLAB.md](COLAB.md) **Option A** (pip + MuJoCo 2.1 + `requirements-colab.txt`); only the **training command** uses [`dynamic-main-TH.py`](dynamic-main-TH.py) instead of `main-TH.py`. Flags and experiments: [HOW_TO_RUN.md](HOW_TO_RUN.md). Troubleshooting, Drive persistence, conda Option B, Python 3.12: [COLAB.md](COLAB.md).

**Blackwell `sm_120` GPUs** cannot use `requirements-colab.txt` as-is — see [requirements-colab-blackwell.txt](requirements-colab-blackwell.txt) and the header comments there.

---

### 1. System packages (compilers + headless GL)

```python
!apt-get update -qq && apt-get install -y -qq \
  build-essential python3-dev pkg-config \
  libosmesa6-dev patchelf libglew-dev libglfw3
```

### 2. Get the code

Use **HTTPS** clone (not `git@...` without SSH keys).

```bash
%%bash
set -e
cd /content
if [ ! -d PIToD/.git ]; then
  git clone https://github.com/NehaBhat14/PIToD.git PIToD
fi
cd PIToD
git pull
```

### 3a. Download MuJoCo 2.1.0

Separate cell first (no `pip`).

```bash
%%bash
set -euo pipefail
MJ="${HOME}/.mujoco"
ARCH="${MJ}/mujoco210-linux-x86_64.tar.gz"
mkdir -p "${MJ}"
if [ ! -f "${MJ}/mujoco210/bin/libmujoco210.so" ]; then
  echo "Downloading MuJoCo 2.1.0 ..."
  wget -nv -O "${ARCH}" \
    "https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz" \
    || curl -fL -o "${ARCH}" \
    "https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz"
  SZ="$(stat -c%s "${ARCH}" 2>/dev/null || wc -c < "${ARCH}")"
  if [ "${SZ}" -lt 1000000 ]; then
    echo "Download looks too small (${SZ} bytes)."
    exit 1
  fi
  tar -xzf "${ARCH}" -C "${MJ}"
fi
test -f "${MJ}/mujoco210/bin/libmujoco210.so"
echo "MuJoCo OK at ${MJ}/mujoco210"
```

### 3b. Install Python deps (without `mujoco-py` first)

```bash
%%bash
set -e
export MUJOCO_PY_MUJOCO_PATH="${HOME}/.mujoco/mujoco210"
export LD_LIBRARY_PATH="${MUJOCO_PY_MUJOCO_PATH}/bin:${LD_LIBRARY_PATH:-}"
pip install -r /content/PIToD/requirements-colab.txt
```

If Colab’s preinstalled packages conflict, append `--upgrade` to the `pip install` line. **Gym-only** (e.g. Hopper-v2): use [`requirements-colab-minimal.txt`](requirements-colab-minimal.txt) here instead, then **3c** unchanged.

### 3c. Install `mujoco-py` last (verbose)

```bash
%%bash
set -e
export MUJOCO_PY_MUJOCO_PATH="${HOME}/.mujoco/mujoco210"
export LD_LIBRARY_PATH="${MUJOCO_PY_MUJOCO_PATH}/bin:${LD_LIBRARY_PATH:-}"
pip install "mujoco-py==2.1.2.14" -v
```

If **3b** or **3c** fails, see [COLAB.md](COLAB.md) (“MuJoCo OK” but something still fails / setuptools / `--no-build-isolation`).

### 4. Run Dynamic PIToD

`-info` is required. Use **unbuffered** output on Colab (`python -u` or `PYTHONUNBUFFERED=1`).

```bash
%%bash
set -e
cd /content/PIToD
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
export LD_LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=osmesa
python -u dynamic-main-TH.py -env Hopper-v2 -info my_colab_run -seed 0 -gpu_id 0
```

Short smoke (optional — uses small `-start_steps` / `-experience_group_size` so Q-loss and Dynamic PIToD metrics are non-zero in a short run):

```bash
python -u dynamic-main-TH.py -env Hopper-v2 -seed 0 -epochs 4 -steps_per_epoch 1000 \
  -info colab_smoke --replay_mode dynamic_pitod -start_steps 500 -experience_group_size 1000 \
  --k_refresh 250 --b_refresh 8 --dynamic_warmup_steps 250 -gpu_id 0
```

Logs under `runs/<info>/.../progress.txt`. Do **not** use `%%capture` on the training cell. Optional: **`-evaluate_bias 1`**, Drive symlink for `runs/` — [COLAB.md](COLAB.md) §4–5 (same patterns as `main-TH.py`).

---

### 5. `PiToD.ipynb` — static vs dynamic smoke + Drive + live cell output

The notebook [PiToD.ipynb](PiToD.ipynb) is the recommended Colab path for a **paired** comparison:

1. **`static_pitod`** — uniform replay during training + post-hoc PIToD when `-evaluate_bias 1` (see [HOW_TO_RUN.md](HOW_TO_RUN.md) §3).
2. **`dynamic_pitod`** — SumTree sampling + periodic TD-flip-delta refresh (same matched flags otherwise).

After clone, it mounts Google Drive and **symlinks** `/content/PIToD/runs` and `/content/PIToD/figure` to `MyDrive/PIToD_runs/` so checkpoints and plots survive runtime disconnects. Training logs are also mirrored to `MyDrive/PIToD_runs/logs/<run>.log`.

**Why not `%%bash` for long training?** A `%%bash` cell often **buffers the whole subshell stdout** until the process exits; on Colab you may see an empty output area for hours, then only `Process is terminated` if the session ends early. That is separate from Python’s own stdout buffering.

**Live output in the notebook cell:** the notebook runs training from a **Python** cell using `subprocess.Popen(..., stdout=PIPE)` and a loop that does `print(line, end='', flush=True)` for every line read from the child. That attaches the trainer’s stdout to the **same cell’s output panel** as lines are produced. The child is still started with `python -u` and `PYTHONUNBUFFERED=1`.

For a one-shot train + plot workflow, use [run_dynamic_pitod_pipeline.py](run_dynamic_pitod_pipeline.py) locally or on Colab: it runs `dynamic-main-TH.py` then `plot_main_results_pitod.py` and **does not run git** unless you pass **`--with-git`** (optional **`--skip-push`** to commit without pushing). The [PiToD.ipynb](PiToD.ipynb) notebook calls `dynamic-main-TH.py` directly for finer control (e.g. paired static vs dynamic cells).
