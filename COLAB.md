# Google Colab setup

Use a **GPU** runtime: Runtime → Change runtime type → GPU.

For PyTorch’s CUDA index, run `!nvidia-smi` and match [`requirements-colab.txt`](requirements-colab.txt) (first line `--extra-index-url`) to [PyTorch’s install matrix](https://pytorch.org/get-started/locally/) if needed (e.g. `cu121`, `cu126`).

### Git on Colab

Use an **HTTPS** clone URL (e.g. `https://github.com/NehaBhat14/PIToD.git`). **Do not** use `git@github.com:...` unless you add SSH keys to the runtime; unauthenticated SSH clones fail with a non-zero exit status.

For a **private** repo, use HTTPS with a [personal access token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens) in the URL (avoid committing the token), or upload the project as a zip instead.

---

## Option A — pip only (no conda, recommended on Colab)

Colab already has Python; install deps with `pip` and use the repo’s requirements file (or paste the same installs into a cell).

### 1. System packages (helps `mujoco-py` / headless GL)

```python
!apt-get update -qq && apt-get install -y -qq libosmesa6-dev patchelf libglew-dev
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

### 3. Install Python dependencies

After cloning, install from the file in the repo:

```python
!pip install -q -r /content/PIToD/requirements-colab.txt
```

If you **do not** have the repo yet, you can point `pip` at the raw file on GitHub (replace with your fork/branch path):

```python
!pip install -q -r https://raw.githubusercontent.com/<user>/PIToD/main/requirements-colab.txt
```

If Colab’s preinstalled packages conflict, add `--upgrade` to the `pip install` line.

To install **entirely from a notebook cell** without any file, paste the contents of [`requirements-colab.txt`](requirements-colab.txt) into a cell using `%%writefile`, then run `pip install -r`:

```python
%%writefile /content/requirements-colab.txt
# paste lines from requirements-colab.txt here (including the --extra-index-url line)
```

```python
!pip install -q -r /content/requirements-colab.txt
```

### 4. Run training

`main-TH.py` requires **`-info`**. Set `PYTHONPATH` to the repo root so `redq` and `customenvs` import.

```bash
%%bash
set -e
cd /content/PIToD
export PYTHONPATH="$PWD"
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
!apt-get update -qq && apt-get install -y -qq libosmesa6-dev patchelf libglew-dev
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

```bash
%%bash
set -e
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate pitod-colab
cd /content/PIToD
export PYTHONPATH="$PWD"
export MUJOCO_GL=osmesa
python main-TH.py -env Hopper-v2 -info my_colab_run -seed 0 -gpu_id 0
```

---

## Notes

- [`requirements-colab.txt`](requirements-colab.txt) mirrors the pip section of [`environment-colab.yml`](environment-colab.yml); keep them in sync when you change versions.
- Original [`environment.yml`](environment.yml) is for local installs with conda `cudatoolkit=11.1.1`.
- Free Colab may run out of disk or time on large installs; pip-only is usually simpler than a full conda solve.
