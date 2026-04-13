#!/bin/bash
# ============================================================
# H1 Experiment: Uniform Replay vs PER on Hopper-v2
# 5 seeds x 2 modes = 10 runs (sequential, single GPU safe)
#
# Output layout (from dynamic-main-TH.py + run_utils.py):
#   runs/H1/redq_sac_Hopper-v2_uniform/redq_sac_Hopper-v2_uniform_s{seed}/progress.txt
#   runs/H1/redq_sac_Hopper-v2_per/redq_sac_Hopper-v2_per_s{seed}/progress.txt
#
# Usage:
#   chmod +x run_uniform_per.sh
#   ./run_uniform_per.sh            # default GPU 0, all 5 seeds
#   GPU_ID=1 ./run_uniform_per.sh   # override GPU
# ============================================================

set -e  # exit on first error

ENV="Hopper-v2"
EPOCHS=200          # 200 * 5000 steps/epoch = 1M env steps
INFO="H1"
GPU_ID="${GPU_ID:-0}"
SEEDS=(0 1 2 3 4)

mkdir -p logs

echo "============================================================"
echo "  Uniform vs PER on $ENV"
echo "  epochs=$EPOCHS  steps_per_epoch=5000  total=1M"
echo "  seeds: ${SEEDS[*]}"
echo "  GPU: $GPU_ID"
echo "  results -> runs/$INFO/"
echo "============================================================"

# ----------------------------------------------------------
# UNIFORM REPLAY
# ----------------------------------------------------------
echo ""
echo ">>> replay_mode=uniform"

for seed in "${SEEDS[@]}"; do
    echo "    seed=$seed  ..."
    python dynamic-main-TH.py \
        -env "$ENV" \
        -seed "$seed" \
        -epochs "$EPOCHS" \
        -steps_per_epoch 5000 \
        -info "$INFO" \
        -gpu_id "$GPU_ID" \
        --replay_mode uniform \
        --h2_log 0 \
        2>&1 | tee "logs/uniform_s${seed}.log"
    echo "    seed=$seed done."
done

echo ">>> uniform complete."

# ----------------------------------------------------------
# PER  (Prioritized Experience Replay)
# Key hyperparameters follow Schaul et al. 2015 defaults:
#   alpha=0.6  beta anneals 0.4->1.0 over 1M steps
# ----------------------------------------------------------
echo ""
echo ">>> replay_mode=per"

for seed in "${SEEDS[@]}"; do
    echo "    seed=$seed  ..."
    python dynamic-main-TH.py \
        -env "$ENV" \
        -seed "$seed" \
        -epochs "$EPOCHS" \
        -steps_per_epoch 5000 \
        -info "$INFO" \
        -gpu_id "$GPU_ID" \
        --replay_mode per \
        --per_alpha 0.6 \
        --per_beta_start 0.4 \
        --per_beta_end 1.0 \
        --per_beta_anneal_steps 1000000 \
        --h2_log 0 \
        2>&1 | tee "logs/per_s${seed}.log"
    echo "    seed=$seed done."
done

echo ">>> per complete."

echo ""
echo "============================================================"
echo "  All runs finished. Results in runs/$INFO/"
echo "  Next: python compare_uniform_per.py"
echo "============================================================"
