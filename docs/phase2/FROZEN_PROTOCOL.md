# Frozen protocol for the Phase 2 paired test evaluation

Frozen at the commit tagged `phase2-freeze`. After this commit, no code, prompt, config or budget changes
are made while `logs/test_eval/` is being produced. No test trajectory is used to change rules, prompts
or memory.

| Item | Value (identical for baseline / ours / ablation) |
|---|---|
| Tasks | `cube_lifting`, `cube_stack`, `cube_restack` (configs `env_configs/harness_eval/*.yaml`, official M4 prompts) |
| Seeds | 100–119 (20 per task); trial id = `env.reset(seed=...)` seed; initial-state hash logged and checked across modes |
| Modes | `baseline` (official M4 VDM) · `ours` (Execution-Grounded VDM + zero-motion gate) · `ablation` (same structured VDM + generic checklist, no record, no gate) |
| Coding model | `deepseek-v4.1-flash-1017`, temperature 0.2, thinking disabled, ≤ 8192 completion tokens/call |
| VDM model | `qwen3.8-flash`, temperature 0.2, thinking disabled |
| API / perception | S3 `FrankaControlApiReducedNoMolmo`; SAM3 :8114, Contact-GraspNet :8115, PyRoKi :8116 (launched with `--log-dir`) |
| Budgets per episode | 8 code executions · 16 model calls (coding + VDM) · 60,000 tokens (coding + VDM, in + out) · 1500 sim steps · 600 s wall time |
| Episode isolation | fresh process per episode; continuous state within the episode; no reruns (`max_trial_retries: 1`) |
| Order | `scripts/run_paired_eval.sh test_eval "cube_lifting cube_stack cube_restack" "100 … 119"`: seed-major, mode order rotated per seed |
| Health | `scripts/check_services.py` before every episode; a failure aborts the run (no episode is started) |
| Metrics | `python3 scripts/paired_analysis.py logs/test_eval --seeds 100-119 --markdown ... --json ...` |
| Evaluation-only signals | `turn_eval_success`, `turn_eval_reward`, `official_task_success`, `official_reward` are logged, never placed in a prompt |

**Infrastructure-fault rule (fixed in advance).** If a service outage is detected inside an episode
(e.g. `failed after N retries` from a perception/IK service), that episode is moved to
`logs/test_eval_invalid/` with a README. The same (task, mode, seed) is run once more after the service is
restored, and the analysis reports the number of such replacements. Outcome-based reruns are never allowed.
