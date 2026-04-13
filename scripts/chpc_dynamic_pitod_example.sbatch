#!/bin/bash
#
# CHPC Slurm batch script for dynamic-main-TH.py on notchpeak-gpu (A100).
# Submit with:  sbatch scripts/chpc_dynamic_pitod_example.sbatch
# Monitor with: squeue -u $USER
#               tail -f slurm-<JOBID>.out
#
# Alternative accounts/partitions you also have access to:
#   soc-gpu-np      / soc-gpu-np      / gpu:a100:1   (SoC A100s)
#   notchpeak-gpu-guest / owner-gpu-guest / gpu:a100:1 (preemptible guest)
#   granite-gpu-guest   / cs4964    / gpu:l40s:1  (add --qos=granite-gpu-guest)
#
#SBATCH --job-name=pitod-dyn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=notchpeak-gpu
#SBATCH --account=notchpeak-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=u1522135@utah.edu

set -euo pipefail

# --- Repo location ---
JOB_ROOT="${HOME}/PIToD"
cd "$JOB_ROOT"

# --- MuJoCo 2.1 (already downloaded in step 5) ---
export MUJOCO_PY_MUJOCO_PATH="${HOME}/.mujoco/mujoco210"
export LD_LIBRARY_PATH="${MUJOCO_PY_MUJOCO_PATH}/bin:${LD_LIBRARY_PATH:-}"
# mujoco-py import requires NVIDIA *driver* libs on LD_LIBRARY_PATH (see builder.py get_nvidia_lib_dir)
for _nv in /usr/lib/nvidia /usr/lib64/nvidia; do
  if [[ -d "$_nv" ]]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${_nv}"
  fi
done
export MUJOCO_GL=osmesa

# --- Activate conda env ---
module load miniconda3
conda activate your-env

# --- Confirm GPU is allocated ---
echo "=== GPU check ==="
nvidia-smi -L
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "=================="

# --- Run training ---
export PYTHONPATH="$JOB_ROOT"
export PYTHONUNBUFFERED=1

python dynamic-main-TH.py \
  -env Hopper-v2 \
  -seed 0 \
  --replay_mode dynamic_pitod \
  -gpu_id 0 \
  -info CHPC_H1

echo "Job finished at $(date)"
