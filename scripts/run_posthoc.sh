#!/usr/bin/env bash
# Post-hoc root-cause verification (Phase 3). Runs post-hoc variants of `ours` on cube_lifting.
# Never writes into logs/test_eval/ (the frozen primary result).
#
# Usage:
#   scripts/run_posthoc.sh <regime> "<variants>" "<seeds>"
#     regime = matched : replay the frozen ours episode's turn-0 scene description and initial code
#                        (and, for gate_only, its turn-0 VDM verdict, whose input is unchanged)
#     regime = raw     : no replay; every call is sampled live
# Example:
#   scripts/run_posthoc.sh matched "ref_only gate_only ref_and_gate" "102 107 109 110 114 116"
#
# Outputs: logs/posthoc_diagnostic/<regime>/<variant>/cube_lifting/seed_<k>/
set -uo pipefail
cd "$(dirname "$0")/.."

REGIME=$1
VARIANTS=$2
SEEDS=$3
TASK=cube_lifting

if [[ -z "${CAPX_API_KEY:-}" ]]; then
  CRED_FILE=${CAPX_CREDENTIAL_FILE:-../api_key.md}
  CAPX_API_KEY=$(grep -oE 'sk-[A-Za-z0-9_-]+' "$CRED_FILE" | head -1)
  export CAPX_API_KEY
fi
export MUJOCO_GL=egl PYTHONUNBUFFERED=1
ROOT="logs/posthoc_diagnostic/${REGIME}"
mkdir -p "$ROOT"

for seed in $SEEDS; do
  for variant in $VARIANTS; do
    out="${ROOT}/${variant}/${TASK}/seed_${seed}"
    if [[ -e "$out/episode_${TASK}_seed${seed}_ours_${variant}.json" ]]; then
      echo "[skip] $out already has an episode log (no reruns)"
      continue
    fi
    extra=()
    if [[ "$REGIME" == "matched" ]]; then
      extra+=(--replay-turn0-from "logs/test_eval/${TASK}/ours/seed_${seed}/episode_${TASK}_seed${seed}_ours.json")
      [[ "$variant" == "gate_only" ]] && extra+=(--replay-turn0-vdm)
    elif [[ "$REGIME" != "raw" ]]; then
      echo "unknown regime $REGIME"; exit 1
    fi
    if ! PYTHONPATH=. .venv/bin/python scripts/check_services.py > "${ROOT}/health_check.log" 2>&1; then
      echo "[abort] service health check failed before seed=$seed variant=$variant"
      grep "\[health\]" "${ROOT}/health_check.log"
      exit 2
    fi
    mkdir -p "$out"
    echo "[run] regime=$REGIME variant=$variant seed=$seed -> $out"
    .venv/bin/python capx/envs/launch.py \
      --config-path "env_configs/harness_eval/${TASK}.yaml" \
      --harness-mode ours --posthoc-variant "$variant" --seeds "$seed" --output-dir "$out" \
      "${extra[@]}" > "$out/console.log" 2>&1
    echo "[done] regime=$REGIME variant=$variant seed=$seed exit=$?"
  done
done
