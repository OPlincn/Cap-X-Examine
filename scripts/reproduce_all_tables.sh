#!/usr/bin/env bash
# Regenerate every table used in the reports from raw episode logs.
# Offline: no simulator, no perception service, no model/network call. Needs only python3 (stdlib).
#
# Usage:  scripts/reproduce_all_tables.sh [--with-tests]
#   --with-tests  also run the offline unit tests (needs the project venv, still no network)
#
# Inputs:  logs/dev_baseline/, logs/dev_phase2/, logs/test_eval/, logs/posthoc_diagnostic/
# Outputs: reports/tables/*.md and *.json (logs/ is never modified)
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=reports/tables
mkdir -p "$OUT"

echo "[1/6] Phase 1 dev baseline (seeds 0-2, 4 tasks)"
python3 scripts/summarize_episodes.py logs/dev_baseline --expect-seeds 0,1,2 \
  --markdown "$OUT/phase1_dev_baseline.md" --json "$OUT/phase1_dev_baseline.json" > /dev/null
python3 scripts/analyze_failures.py logs/dev_baseline --json "$OUT/phase1_dev_failure_tags.json" \
  > "$OUT/phase1_dev_failure_tags.txt"

echo "[2/6] Phase 2 dev sanity check (final code, seeds 0-2; ours vs ablation)"
python3 scripts/paired_analysis.py logs/dev_phase2 --seeds 0-2 --modes ablation,ours \
  --markdown "$OUT/phase2_dev_sanity.md" --json "$OUT/phase2_dev_sanity.json" > /dev/null

echo "[3/6] Phase 2 frozen paired test evaluation (180 episodes): main tables"
python3 scripts/paired_analysis.py logs/test_eval --seeds 100-119 \
  --markdown "$OUT/phase2_main_results.md" --json "$OUT/phase2_main_results.json" > /dev/null

echo "[4/6] Phase 2 diagnostics (sign tests, root causes, gate isolation, lifting Δz)"
python3 scripts/phase2_diagnostics.py logs/test_eval --markdown "$OUT/phase2_diagnostics.md" > /dev/null

echo "[5/6] Phase 3 post-hoc 2x2 verification (cube_lifting seeds 102,107,109,110,114,116)"
python3 scripts/posthoc_analysis.py --seeds 102,107,109,110,114,116 \
  --markdown "$OUT/phase3_posthoc_2x2.md" --json "$OUT/phase3_posthoc_2x2.json" > /dev/null

echo "[6/6] Integrity summary"
python3 - <<'EOF'
import json
main = json.load(open("reports/tables/phase2_main_results.json"))
post = json.load(open("reports/tables/phase3_posthoc_2x2.json"))
print("  test_eval integrity problems:", main["integrity_problems"] or "none")
print("  posthoc missing episodes:   ", post["problems"] or "none")
for k in ("ALL/baseline", "ALL/ablation", "ALL/ours"):
    g = main["groups"][k]
    print(f"  {k:13} success {g['success']}/{g['n']}")
EOF

if [[ "${1:-}" == "--with-tests" ]]; then
  echo "[tests] offline unit tests"
  .venv/bin/python -m pytest -q tests/test_episode_telemetry.py tests/test_eg_vdm.py -p no:cacheprovider
fi
echo "Tables written to $OUT/"
