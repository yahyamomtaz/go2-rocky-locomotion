#!/usr/bin/env bash
# Run from the project root after: uv sync --locked --extra train
set -euo pipefail
training_envs="${NUM_ENVS:-512}"
training_seed="${TRAINING_SEED:-11}"
training_prefix="${TRAINING_PREFIX:-climb_seed${training_seed}}"
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
trainer=(.venv/bin/python -u -m go2_rocky.train_ppo --envs "$training_envs" --seed "$training_seed")
"${trainer[@]}" --stage slope --max-slope 5 --iters "${GENTLE_ITERS:-250}" --warmup 25 \
    --name "${training_prefix}_gentle"
"${trainer[@]}" --stage slope --max-slope 15 --iters "${SLOPE_ITERS:-500}" --warmup 25 \
    --init-policy "runs/train/${training_prefix}_gentle/policy.pt" --name "${training_prefix}_steeper"
"${trainer[@]}" --stage rocky --max-slope 25 --roughness medium --surface-rockiness 1 \
    --rock-count 48 --iters "${ROCK_ITERS:-1000}" --warmup 25 \
    --init-policy "runs/train/${training_prefix}_steeper/policy.pt" --name "${training_prefix}_rocky"
