#!/usr/bin/env bash
# Paired evaluation: for every (task, seed) run all modes back-to-back from the same
# seeded initial state, one fresh process per episode. Mode order rotates with the seed
# so that slow drifts of the model gateway do not systematically favour one mode.
#
# Usage:
#   scripts/run_paired_eval.sh <run_name> "<tasks>" "<seeds>" ["<modes>"]
# Example (Phase 2 test set):
#   scripts/run_paired_eval.sh test_eval "cube_lifting cube_stack cube_restack" "$(seq -s ' ' 100 119)"
#
# Outputs: logs/<run_name>/<task>/<mode>/seed_<k>/{episode_<task>_seed<k>_<mode>.json, console.log, trial_*/}
# Existing episode logs are never overwritten or rerun.
set -uo pipefail
cd "$(dirname "$0")/.."

RUN_NAME=$1
TASKS=$2
SEEDS=$3
MODES=(${4:-baseline ours ablation})

if [[ -z "${CAPX_API_KEY:-}" ]]; then
  CRED_FILE=${CAPX_CREDENTIAL_FILE:-../api_key.md}
  CAPX_API_KEY=$(grep -oE 'sk-[A-Za-z0-9_-]+' "$CRED_FILE" | head -1)
  export CAPX_API_KEY
fi
export MUJOCO_GL=egl PYTHONUNBUFFERED=1

n=${#MODES[@]}
for task in $TASKS; do
  i=0
  for seed in $SEEDS; do
    for ((j = 0; j < n; j++)); do
      mode=${MODES[$(((i + j) % n))]}
      out="logs/${RUN_NAME}/${task}/${mode}/seed_${seed}"
      if [[ -e "$out/episode_${task}_seed${seed}_${mode}.json" ]]; then
        echo "[skip] $out already has an episode log (no reruns)"
        continue
      fi
      if ! PYTHONPATH=. .venv/bin/python scripts/check_services.py > "logs/${RUN_NAME}/health_check.log" 2>&1; then
        echo "[abort] service health check failed before task=$task seed=$seed mode=$mode:"
        grep "\[health\]" "logs/${RUN_NAME}/health_check.log"
        exit 2
      fi
      mkdir -p "$out"
      echo "[run] task=$task seed=$seed mode=$mode -> $out"
      .venv/bin/python capx/envs/launch.py \
        --config-path "env_configs/harness_eval/${task}.yaml" \
        --harness-mode "$mode" --seeds "$seed" --output-dir "$out" \
        > "$out/console.log" 2>&1
      echo "[done] task=$task seed=$seed mode=$mode exit=$?"
    done
    i=$((i + 1))
  done
done
