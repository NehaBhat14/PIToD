# Google Colab — GPU + Dynamic PIToD

This guide runs **Dynamic PIToD** (and other replay modes) on Google Colab with a **normal GPU runtime** — e.g. **Tesla T4**, **L4**, **A100**, or **H100**. The install path is the same for all of them: match the PyTorch CUDA wheel in [`requirements-colab.txt`](requirements-colab.txt) to Colab’s driver (see [COLAB.md](COLAB.md) intro). For general Colab troubleshooting, Option B (conda), and Python 3.12 notes, see [COLAB.md](COLAB.md). For CLI flags and experiment grids, see [HOW_TO_RUN.md](HOW_TO_RUN.md).

---

## 1. Runtime: enable GPU and verify the driver

1. **Runtime → Change runtime type** → set **Hardware accelerator** to **GPU** (T4 on free tier; L4 / A100 / H100 on paid tiers when offered).
2. Confirm the GPU and CUDA driver:

```python
!nvidia-smi
```

You should see an NVIDIA GPU name (e.g. `Tesla T4`, `NVIDIA L4`, `A100`, `H100`) and a driver that supports **CUDA 12.x** on current Colab images.

**PyTorch wheel:** [requirements-colab.txt](requirements-colab.txt) uses `torch==2.5.1` with `--extra-index-url https://download.pytorch.org/whl/cu124`. If `pip` cannot install torch, open [PyTorch’s install matrix](https://pytorch.org/get-started/locally/) and adjust only the **first line** of `requirements-colab.txt` (e.g. `cu121`, `cu126`, `cu128`) — same as [COLAB.md](COLAB.md). The **CUDA Version** line in `nvidia-smi` (e.g. **13.0** on driver **580+**) is the **maximum** CUDA the driver supports; PyTorch still ships its **own** CUDA runtime inside the wheel (often 12.x), which is compatible with that driver.

### 1a. Blackwell `sm_120` (e.g. RTX PRO 6000 Blackwell Server Edition)

**Symptom:** PyTorch prints `CUDA capability sm_120 is not compatible with the current PyTorch installation` and lists only up to `sm_90`, then:

`RuntimeError: CUDA error: no kernel image is available for execution on the device`

**Cause:** [`requirements-colab.txt`](requirements-colab.txt) uses **`torch==2.5.1` + `cu124`**, which does **not** include Blackwell (`sm_120`) GPU kernels. You need **PyTorch 2.7+** with **CUDA 12.8** wheels ([stable install](https://pytorch.org/get-started/locally/) → Linux, Pip, **CUDA 12.8**).

**Fix (recommended):** reinstall torch/torchvision from the `cu128` index, then install the rest of the Colab stack from the Blackwell requirements file (same deps as Colab, newer torch pins):

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-colab-blackwell.txt
```

[`requirements-colab-blackwell.txt`](requirements-colab-blackwell.txt) is a copy of the Colab dependency set with **`torch==2.7.0`**, **`torchvision==0.22.0`**, and **`cu128`**. If `pip` ever resolves a CPU-only build, keep the **`pip install torch torchvision --index-url .../cu128`** line **before** `pip install -r` (see the comment at the top of that file).

**Verify:**

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("capability:", torch.cuda.get_device_capability(0))
x = torch.randn(4096, 4096, device="cuda")
print("ok:", (x @ x).sum().item())
PY
```

You want **`capability: (12, 0)`** and a successful matmul with **no** `sm_120` compatibility warning.

**Then** install `mujoco-py` as in §2.3c (unchanged).

---

## 2. Option A — pip install (same as COLAB.md; do not skip)

### 2.1 System packages (compilers + headless GL)

`mujoco-py` compiles C extensions and expects **MuJoCo 2.1** on disk (next step).

```python
!apt-get update -qq && apt-get install -y -qq \
  build-essential python3-dev pkg-config \
  libosmesa6-dev patchelf libglew-dev libglfw3
```

### 2.2 Get the code

Use an **HTTPS** clone URL. Do **not** use `git@github.com:...` unless you configure SSH keys on the runtime.

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

Use your fork URL if applicable, or upload/unzip the project under `/content/PIToD`.

### 2.3a Download MuJoCo 2.1.0 (required for `mujoco-py` / Gym MuJoCo)

Run in a **separate** cell first (no `pip` here — clearer errors if download fails).

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
    echo "Download looks too small (${SZ} bytes). Likely an HTML error page."
    exit 1
  fi
  tar -xzf "${ARCH}" -C "${MJ}"
fi
test -f "${MJ}/mujoco210/bin/libmujoco210.so"
echo "MuJoCo OK at ${MJ}/mujoco210"
```

### 2.3b Install Python deps (without `mujoco-py` first)

[`requirements-colab.txt`](requirements-colab.txt) omits `mujoco-py` so this step can finish with a visible log. Avoid `-q` while debugging.

```bash
%%bash
set -e
export MUJOCO_PY_MUJOCO_PATH="${HOME}/.mujoco/mujoco210"
export LD_LIBRARY_PATH="${MUJOCO_PY_MUJOCO_PATH}/bin:${LD_LIBRARY_PATH:-}"
pip install -r /content/PIToD/requirements-colab.txt
```

If Colab’s preinstalled packages conflict, append `--upgrade` to the `pip install` line.

**Gym-only (e.g. Hopper-v2):** you may use [`requirements-colab-minimal.txt`](requirements-colab-minimal.txt) instead of `requirements-colab.txt` for this step, then run **2.3c** unchanged.

### 2.3c Install `mujoco-py` last (verbose)

Keep the same env vars in `%%bash`:

```bash
%%bash
set -e
export MUJOCO_PY_MUJOCO_PATH="${HOME}/.mujoco/mujoco210"
export LD_LIBRARY_PATH="${MUJOCO_PY_MUJOCO_PATH}/bin:${LD_LIBRARY_PATH:-}"
pip install "mujoco-py==2.1.2.14" -v
```

If this fails, scroll up for the first `error:` from the compiler or `setup.py`. See [COLAB.md](COLAB.md) (“MuJoCo OK” but something still fails / setuptools pin / `--no-build-isolation`).

---

## 3. Post-install sanity (PyTorch + CUDA)

```bash
%%bash
cd /content/PIToD
export PYTHONPATH="$PWD"
python - <<'PY'
import torch
print("cuda_available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
x = torch.randn(2048, 2048, device="cuda")
y = torch.randn(2048, 2048, device="cuda")
z = (x @ y).sum().item()
print("matmul_ok:", z == z)
PY
```

Expect `cuda_available: True`, a **real GPU name** (T4 / L4 / A100 / H100 / …), and `matmul_ok: True`. Matmul size is reduced vs a huge square on **T4** (16 GB) to avoid OOM during the sanity check; increase if you have more VRAM.

---

## 4. Dynamic PIToD — run `dynamic-main-TH.py`

Static PIToD uses [`main-TH.py`](main-TH.py). For **uniform / PER / static_pitod / dynamic_pitod** replay modes, use [`dynamic-main-TH.py`](dynamic-main-TH.py).

**Headless MuJoCo + unbuffered logs** (recommended for Colab):

```bash
%%bash
set -e
cd /content/PIToD
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
export LD_LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=osmesa
python -u dynamic-main-TH.py -env Hopper-v2 -seed 0 -epochs 2 -steps_per_epoch 1000 \
  -info colab_smoke --replay_mode dynamic_pitod \
  --k_refresh 500 --b_refresh 8 --dynamic_warmup_steps 500 -gpu_id 0
```

On Colab there is a single visible GPU — use **`-gpu_id 0`**.

Logs land under `runs/colab_smoke/.../progress.txt` (see [HOW_TO_RUN.md](HOW_TO_RUN.md) §9).

### Optional: post-hoc static PIToD bias columns (expensive)

Add `-evaluate_bias 1` to enable `log_evaluation` and `QBias*` / related columns (see [HOW_TO_RUN.md](HOW_TO_RUN.md)).

### Optional: persist `runs/` on Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
%%bash
mkdir -p /content/drive/MyDrive/PIToD_runs
ln -sfn /content/drive/MyDrive/PIToD_runs /content/PIToD/runs
```

---

## 5. Performance notes (any GPU)

- Training is often limited by **Gym + mujoco-py + Python** stepping and **SumTree refresh** work, not only raw GPU FLOPs. Watch **`SPS`** and **`DynPIToD/RefreshWallclock`** in `progress.txt` ([HOW_TO_RUN.md](HOW_TO_RUN.md) §10).
- **T4 / smaller VRAM:** keep default smoke settings; for long runs, avoid huge batch sizes or extra processes on the same runtime.
- **No code changes** are required for a specific GPU SKU if `torch.cuda.is_available()` is true and the pinned wheel installs.

---

## 6. Checklist summary

| Step | Action |
|------|--------|
| Runtime | **GPU** runtime (T4 / L4 / A100 / H100, …); `nvidia-smi` |
| PyTorch | `pip install -r requirements-colab.txt`; adjust `--extra-index-url` only if driver/wheel mismatch |
| MuJoCo + mujoco-py | Sections **2.3a–2.3c** (same as [COLAB.md](COLAB.md) Option A) |
| Sanity | Section **3** — device name + CUDA matmul |
| Dynamic PIToD | Section **4** — `dynamic-main-TH.py` + env vars; full CLI in [HOW_TO_RUN.md](HOW_TO_RUN.md) |

---

## 7. Further reading

- [HOW_TO_RUN.md](HOW_TO_RUN.md) — replay modes, smoke tests, H1/H2/H3 grids, `progress.txt` columns
- [COLAB.md](COLAB.md) — Option B conda, verbose pip debugging, Python 3.12 notes

---

## 8. Optional: H100 / high-end GPU only

If your runtime is **H100** (Hopper), nothing extra is required beyond the same steps above — `cu124` PyTorch wheels are appropriate on current Colab drivers. Expect higher **`SPS`** than T4 for the same script; refresh wall-clock may still matter ([HOW_TO_RUN.md](HOW_TO_RUN.md) §10).
