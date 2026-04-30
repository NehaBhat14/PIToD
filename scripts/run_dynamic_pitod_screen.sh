#!/usr/bin/env bash
set -euo pipefail

# Focused Hopper-v2 screen for variants most likely to make Dynamic PIToD stand out.
# Override with env vars, e.g.:
#   SEEDS="0 1 2 3 4" EPOCHS=100 GPU_ID=1 bash scripts/run_dynamic_pitod_screen.sh

ENV_NAME="${ENV_NAME:-Hopper-v2}"
SEEDS="${SEEDS:-0 1 2}"
EPOCHS="${EPOCHS:-60}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-5000}"
GPU_ID="${GPU_ID:-0}"
BASE_FLAGS=(
  -env "${ENV_NAME}"
  -epochs "${EPOCHS}"
  -steps_per_epoch "${STEPS_PER_EPOCH}"
  -gpu_id "${GPU_ID}"
)

for seed in ${SEEDS}; do
  python dynamic-main-TH.py "${BASE_FLAGS[@]}" -seed "${seed}" \
    -info screen_uniform --replay_mode uniform

  python dynamic-main-TH.py "${BASE_FLAGS[@]}" -seed "${seed}" \
    -info screen_static --replay_mode static_pitod -evaluate_bias 1

  python dynamic-main-TH.py "${BASE_FLAGS[@]}" -seed "${seed}" \
    -info screen_dyn_base --replay_mode dynamic_pitod \
    --k_refresh 10000 --b_refresh 16 --dynamic_warmup_steps 10000 --m_strikes 5

  python dynamic-main-TH.py "${BASE_FLAGS[@]}" -seed "${seed}" \
    -info screen_dyn_early --replay_mode dynamic_pitod \
    --k_refresh 10000 --b_refresh 16 --dynamic_warmup_steps 5000 \
    --early_phase_steps 50000 --early_k_refresh 5000 --early_b_refresh 16 \
    --m_strikes 5 --prune_warmup_steps 50000

  python dynamic-main-TH.py "${BASE_FLAGS[@]}" -seed "${seed}" \
    -info screen_dyn_strong --replay_mode dynamic_pitod \
    --k_refresh 5000 --b_refresh 32 --dynamic_warmup_steps 5000 \
    --early_phase_steps 50000 --early_k_refresh 2500 --early_b_refresh 32 \
    --m_strikes 5 --prune_warmup_steps 50000

  python dynamic-main-TH.py "${BASE_FLAGS[@]}" -seed "${seed}" \
    -info screen_dyn_noprune --replay_mode dynamic_pitod \
    --k_refresh 5000 --b_refresh 32 --dynamic_warmup_steps 5000 \
    --early_phase_steps 50000 --early_k_refresh 2500 --early_b_refresh 32 \
    --dynamic_pruning 0
done
