# Google Colab setup

Use a **GPU** runtime: Runtime → Change runtime type → GPU.

For PyTorch’s CUDA index, run `!nvidia-smi` and match [`requirements-colab.txt`](requirements-colab.txt) (first line `--extra-index-url`) to [PyTorch’s install matrix](https://pytorch.org/get-started/locally/) if needed (e.g. `cu121`, `cu126`).

### Git on Colab

Use an **HTTPS** clone URL (e.g. `https://github.com/NehaBhat14/PIToD.git`). **Do not** use `git@github.com:...` unless you add SSH keys to the runtime; unauthenticated SSH clones fail with a non-zero exit status.

For a **private** repo, use HTTPS with a [personal access token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens) in the URL (avoid committing the token), or upload the project as a zip instead.

---

## Option A — pip only (no conda, recommended on Colab)

Colab already has Python; install deps with `pip` and use the repo’s requirements file (or paste the same installs into a cell).

### 1. System packages (compilers + headless GL)

`mujoco-py` compiles C extensions and expects **MuJoCo 2.1** on disk (next step). Without `build-essential` / `python3-dev`, pip often fails with **“Getting requirements to build wheel”** and no clear package name.

```python
!apt-get update -qq && apt-get install -y -qq \
  build-essential python3-dev pkg-config \
  libosmesa6-dev patchelf libglew-dev libglfw3
```

### 2. Get the code

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

Change the `git clone` URL if you use a fork. Or upload/unzip the project under `/content/PIToD`.

### 3a. Download MuJoCo 2.1.0 (required for `mujoco-py` / Gym MuJoCo envs)

Run this **first** (separate cell). It does **not** run `pip`, so if this fails you see a clear error (network, disk, bad archive).

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
  # GitHub error pages are tiny; a real tarball is ~4MB
  SZ="$(stat -c%s "${ARCH}" 2>/dev/null || wc -c < "${ARCH}")"
  if [ "${SZ}" -lt 1000000 ]; then
    echo "Download looks too small (${SZ} bytes). Open ${ARCH} in an editor or cat it — likely an HTML error page."
    exit 1
  fi
  tar -xzf "${ARCH}" -C "${MJ}"
fi
test -f "${MJ}/mujoco210/bin/libmujoco210.so"
echo "MuJoCo OK at ${MJ}/mujoco210"
```

### 3b. Install Python deps (**without** `mujoco-py` first)

[`requirements-colab.txt`](requirements-colab.txt) omits `mujoco-py` so this step can finish (or fail with a **visible** pip log). **Do not use `-q`** while debugging.

```bash
%%bash
set -e
export MUJOCO_PY_MUJOCO_PATH="${HOME}/.mujoco/mujoco210"
export LD_LIBRARY_PATH="${MUJOCO_PY_MUJOCO_PATH}/bin:${LD_LIBRARY_PATH:-}"
pip install -r /content/PIToD/requirements-colab.txt
```

If Colab’s preinstalled packages conflict, append `--upgrade` to that line.

If you **do not** have the repo cloned yet:

```bash
pip install -r https://raw.githubusercontent.com/<user>/PIToD/main/requirements-colab.txt
```

### 3c. Install `mujoco-py` last (verbose)

Still in `%%bash` so the env vars apply to the build:

```bash
%%bash
set -e
export MUJOCO_PY_MUJOCO_PATH="${HOME}/.mujoco/mujoco210"
export LD_LIBRARY_PATH="${MUJOCO_PY_MUJOCO_PATH}/bin:${LD_LIBRARY_PATH:-}"
pip install "mujoco-py==2.1.2.14" -v
```

If this alone errors, scroll up in the output for the **first** `error:` line from the compiler or `setup.py`.

Set the same `MUJOCO_PY_MUJOCO_PATH` and `LD_LIBRARY_PATH` when you run `python main-TH.py` (step 4).

To install **entirely from a notebook cell** without cloning, use `%%writefile` for [`requirements-colab.txt`](requirements-colab.txt), run steps **3b** and **3c** against that path, then run step 4.

### “MuJoCo OK” but something still fails

**“MuJoCo OK” only means step 3a worked.** The remaining failure is almost always **step 3b** (`pip install -r requirements-colab.txt`) or **step 3c** (`mujoco-py`). Capture the real message:

```bash
%%bash
set +e
export MUJOCO_PY_MUJOCO_PATH="${HOME}/.mujoco/mujoco210"
export LD_LIBRARY_PATH="${MUJOCO_PY_MUJOCO_PATH}/bin:${LD_LIBRARY_PATH:-}"
pip install -r /content/PIToD/requirements-colab.txt 2>&1 | tee /content/pip-3b.log
echo "Exit code: $?"
tail -80 /content/pip-3b.log
```

Open `/content/pip-3b.log` in Colab’s file browser and search for `error:` / `ERROR` / `Failed`.

**Try a smaller install** if you only run **Gym MuJoCo** envs (e.g. `Hopper-v2`, `AntTruncatedObs-v2`) — not DeepMind Control names like `cheetah-run`. Use [`requirements-colab-minimal.txt`](requirements-colab-minimal.txt) in place of step **3b**:

```bash
%%bash
set -e
export MUJOCO_PY_MUJOCO_PATH="${HOME}/.mujoco/mujoco210"
export LD_LIBRARY_PATH="${MUJOCO_PY_MUJOCO_PATH}/bin:${LD_LIBRARY_PATH:-}"
pip install -r /content/PIToD/requirements-colab-minimal.txt
```

Then run **3c** unchanged. (The code loads `dmc2gym` only when you use a DM Control `-env`.)

If **only 3c** fails, try:

```bash
pip install "setuptools>=64,<71" wheel
pip install "mujoco-py==2.1.2.14" --no-build-isolation -v
```

**Python 3.12 on Colab:** `pandas==2.0.3` has no prebuilt wheel, so pip tries to compile it and fails with “Getting requirements to build wheel”. [`requirements-colab.txt`](requirements-colab.txt) uses `pandas>=2.1.4,<3.0` instead; `git pull` your repo copy if you still see `2.0.3`.

### 4. Run training

`main-TH.py` requires **`-info`**. Set `PYTHONPATH` to the repo root so `redq` and `customenvs` import.

```bash
%%bash
set -e
cd /content/PIToD
export PYTHONPATH="$PWD"
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
export LD_LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=osmesa
python main-TH.py -env Hopper-v2 -info my_colab_run -seed 0 -gpu_id 0
```

### 5. Optional: persist logs on Google Drive

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

## Option B — conda + `environment-colab.yml`

Use this if you prefer matching the full conda stack (e.g. conda-forge `glew` / `mesalib`). Heavier on disk and time.

The file [`environment-colab.yml`](environment-colab.yml) omits conda `cudatoolkit` and uses PyTorch CUDA wheels (default **cu124**).

### 1. System packages (if `mujoco-py` / OpenGL fails)

```python
!apt-get update -qq && apt-get install -y -qq \
  build-essential python3-dev pkg-config \
  libosmesa6-dev patchelf libglew-dev libglfw3
```

### 2. Install Miniconda (once per new runtime)

```bash
%%bash
set -e
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
$HOME/miniconda/bin/conda init bash
```

Restart the runtime if prompted, then use `source` in each `%%bash` block below.

### 3. Clone the repo

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

### 4. Create the conda environment

```bash
%%bash
set -e
source "$HOME/miniconda/etc/profile.d/conda.sh"
cd /content/PIToD
conda env create -f environment-colab.yml
```

Update after YAML changes:

```bash
conda env update -f environment-colab.yml --prune
```

### 5. Run training

Download MuJoCo 2.1 once (same as Option A step 3) if `mujoco-py` was built against it, then:

```bash
%%bash
set -e
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate pitod-colab
cd /content/PIToD
export PYTHONPATH="$PWD"
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
export LD_LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=osmesa
python main-TH.py -env Hopper-v2 -info my_colab_run -seed 0 -gpu_id 0
```

---

## Notes

- [`requirements-colab.txt`](requirements-colab.txt) matches most of the pip section of [`environment-colab.yml`](environment-colab.yml); **`mujoco-py` is installed in step 3c** on Colab (conda YAML still installs it in one env). When bumping versions, update both files plus step **3c**.
- Original [`environment.yml`](environment.yml) is for local installs with conda `cudatoolkit=11.1.1`.
- Free Colab may run out of disk or time on large installs; pip-only is usually simpler than a full conda solve.

### If `pip` still fails while “building wheel”

1. **See which step failed**: **3a** prints `MuJoCo OK` → archives are fine. **3b** = a dependency in [`requirements-colab.txt`](requirements-colab.txt); **3c** = `mujoco-py` only. Re-run the failing step **without** `-q` and scroll up for the **first** `Error` / `error:` from the compiler or `setup.py`. Try [`requirements-colab-minimal.txt`](requirements-colab-minimal.txt) for Gym-only runs (see **“MuJoCo OK” but something still fails** above).

2. **Verbose bulk install**:

   ```bash
   %%bash
   pip install -r /content/PIToD/requirements-colab.txt -v 2>&1 | tail -120
   ```

3. **Python version**: run `!python --version`. If Colab gives **3.12+** and many packages try to compile from source, use **Option B** (conda env with `python=3.11`) or relax pins carefully.

4. **`mujoco-py` only** (step 3c):

   ```bash
   %%bash
   export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
   export LD_LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin:${LD_LIBRARY_PATH:-}"
   pip install mujoco-py==2.1.2.14 --no-build-isolation -v
   ```
