#!/usr/bin/env bash
# Run paired harness episodes, one fresh process (fresh agent + simulator) per episode.
#
# Usage:
#   scripts/run_harness_eval.sh <run_name> <mode> "<tasks>" "<seeds>" [extra launch args...]
# Example (Phase 1 dev set):
#   scripts/run_harness_eval.sh dev_baseline baseline "cube_lifting cube_stack spill_wipe" "0 1 2"
#
# Outputs: logs/<run_name>/<task>/seed_<k>/{episode_<task>_seed<k>_<mode>.json, console.log, trial_*/}
# The API key is read from $CAPX_API_KEY, else from $CAPX_CREDENTIAL_FILE (default ../api_key.md).
set -uo pipefail
cd "$(dirname "$0")/.."

RUN_NAME=$1
MODE=$2
TASKS=$3
SEEDS=$4
shift 4

if [[ -z "${CAPX_API_KEY:-}" ]]; then
  CRED_FILE=${CAPX_CREDENTIAL_FILE:-../api_key.md}
  CAPX_API_KEY=$(grep -oE 'sk-[A-Za-z0-9_-]+' "$CRED_FILE" | head -1)
  export CAPX_API_KEY
fi
export MUJOCO_GL=egl PYTHONUNBUFFERED=1

for task in $TASKS; do
  for seed in $SEEDS; do
    out="logs/${RUN_NAME}/${task}/seed_${seed}"
    if [[ -e "$out/episode_${task}_seed${seed}_${MODE}.json" ]]; then
      echo "[skip] $out already has an episode log (no reruns)"
      continue
    fi
    mkdir -p "$out"
    echo "[run] task=$task seed=$seed mode=$MODE -> $out"
    .venv/bin/python capx/envs/launch.py \
      --config-path "env_configs/harness_eval/${task}.yaml" \
      --harness-mode "$MODE" --seeds "$seed" --output-dir "$out" "$@" \
      > "$out/console.log" 2>&1
    echo "[done] task=$task seed=$seed exit=$?"
  done
done
