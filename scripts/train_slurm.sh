#!/usr/bin/env bash
# Example only: supply your site's --account and --partition when submitting.
#SBATCH --job-name=go2-climb
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=go2-train-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
# Pre-install using: uv sync --locked --extra train
# Pre-cache Go2 assets on a node with network access before compute-node use.
# Load your site's Python/CUDA modules or container before this script, as needed.
training_seed="${TRAINING_SEED:-11}"
training_name="${TRAINING_NAME:-hpc_rocky_seed${training_seed}_${SLURM_JOB_ID}}"
training_stage="${TRAINING_STAGE:-rocky}"
if [[ "$training_stage" == "rocky" ]]; then surface_default=1; else surface_default=0; fi
training_args=()
if [[ -n "${INIT_POLICY:-}" ]]; then
    training_args+=(--init-policy "$INIT_POLICY")
fi
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
    training_args+=(--resume "$RESUME_CHECKPOINT")
fi
# Forward Slurm's early warning to Python, which checkpoints at the next chunk.
training_pid=""
forward_signal() {
    if [[ -n "$training_pid" ]]; then kill -USR1 "$training_pid" 2>/dev/null || true; fi
}
trap forward_signal USR1 TERM INT
.venv/bin/python -u -m go2_rocky.train_ppo \
    --stage "$training_stage" --max-slope "${MAX_SLOPE:-15}" \
    --surface-rockiness "${SURFACE_ROCKINESS:-$surface_default}" \
    --roughness "${ROUGHNESS:-medium}" --envs "${NUM_ENVS:-512}" \
    --iters "${TRAINING_ITERS:-1500}" --seed "$training_seed" \
    --name "$training_name" "${training_args[@]}" &
training_pid=$!
# A trapped signal interrupts wait; continue waiting while the trainer saves.
while true; do
    if wait "$training_pid"; then exit 0; else training_status=$?; fi
    if ! kill -0 "$training_pid" 2>/dev/null; then exit "$training_status"; fi
done
