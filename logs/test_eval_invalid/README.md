# Incident: workstation crash during the Phase 2 test run (2026-10-24 ~22:07)

**What happened.** A GPU driver crash knocked the workstation offline while the paired test run
(`scripts/run_paired_eval.sh test_eval ...`, freeze commit `ed29cc7`, tag `phase2-freeze`) was in
progress. The machine came back on 2026-10-26 with a new kernel (6.8.0-40 → 6.8.0-138); the NVIDIA
driver is unchanged (535.154.05).

**State of `logs/test_eval/` after the crash (audited 2026-10-26):**

| Status | Count | Episodes |
|---|---|---|
| Complete, valid | 152 | all `cube_lifting` (60), all `cube_stack` except `ours/seed_119` (59), `cube_restack` seeds 100–110 all modes (33) |
| Complete, but hit an infrastructure fault | 1 | `cube_stack/ours/seed_119`: the GraspNet service returned HTTP 500 twice due to `CUDA out of memory. Tried to allocate 1.83 GiB` (episode reached official success). The OOM was transient: 92 later GraspNet requests succeeded |
| Interrupted by the crash | 1 | `cube_restack/ablation/seed_111`: console stops while loading the scene (22:07:15); no model call, no episode JSON |
| Never started | 26 | `cube_restack` seed 111 baseline/ours, seeds 112–119 all modes |

**Checks performed.**
- Code: `git diff phase2-freeze` shows no change outside the two vendor submodules. Their patches are the
  documented ones (`scripts/apply_vendor_patches.sh` → "already applied").
- Service errors in all 153 completed console logs: only the GraspNet OOM above. The IK HTTP 500s in
  `pyroki_8116.log` all come from a malformed manual probe made at service start, before the freeze; no
  episode contains them.
- Models (2026-10-26): `deepseek-v4.1-flash-1017` and `qwen3.8-flash` return the same model ids as all
  997 logged test calls, with reasoning still disabled.
- Services restarted with `--log-dir logs/services/run_20261026_*`; `scripts/check_services.py` passes.

**Planned handling (pre-registered rule in `docs/phase2/FROZEN_PROTOCOL.md`).**
1. Move `cube_stack/ours/seed_119` and `cube_restack/ablation/seed_111` to `logs/test_eval_invalid/`
   and run each once more, unchanged code. The replacement is made regardless of the original outcome.
2. Resume the same driver command; existing complete episodes are skipped (no reruns), so only the 28
   missing (task, mode, seed) keys run.
3. Report in the results: 2 infrastructure replacements, and that `cube_restack` seeds 111–119 (all
   three modes) and the `cube_stack/ours/seed_119` replacement ran after a two-day gap and a reboot.
   Restack seeds 111–119 keep within-seed pairing (all modes run together after the reboot). The single
   `cube_stack` seed 119 pair spans the gap.
