# Training-Free Harness Improvement for CaP-X Coding Agents — Submission Package

**Repository:** `https://github.com/OPlincn/Cap-X-Examine`

**Method.** Execution-Grounded VDM (EG-VDM) with a Zero-Motion Gate: a training-free change to Step 3 (visual
differencing feedback) of the CaP-X M4 multi-turn loop. The paired evaluation uses 3 Robosuite tasks × 20 test
seeds × 3 conditions (180 episodes). The method did **not** improve task success (baseline 51/60, ablation 47/60,
ours 43/60). The post-hoc 2×2 verification isolates the two causes we identified.

| Package Component | Path |
|---|---|
| Offline table reproduction script & outputs | `scripts/reproduce_all_tables.sh` $\to$ `reports/tables/*.{md,json}` |
| Phase working documents & protocol | `docs/phase1/`, `docs/phase2/`, `docs/phase3/` |
| Human/AI refinement record | `logs/human_ai_refinements.md` |

## 1. Commits and tags

| Stage | Tag | Commit | Content |
|---|---|---|---|
| Official CaP-X base | — | `53e9966` | upstream `capgym/cap-x` |
| Uniform environment fixes E1–E5 | — | `903097d` | identical for every condition |
| Telemetry + budget enforcer | — | `e6c2512` | per-episode JSON logs, hard caps |
| Phase 1 (dev diagnosis + proposal) | `phase1` | `cce6865` | dev baseline logs, proposal |
| **Frozen code for the main result** | **`phase2-freeze`** | **`ed29cc7`** | code, prompts, budgets used for all 180 test episodes |
| Phase 2 results (analysis only) | `phase2-results` | `e3aa502` | `logs/test_eval/`, analysis scripts, report |
| Phase 3 post-hoc verification + deliverables | `phase3-final` | see `git rev-parse phase3-final` | post-hoc variants, `logs/posthoc_diagnostic/`, reports |

The main result is produced only by the code at `phase2-freeze`. The post-hoc variants added in Phase 3 are
disabled unless `--posthoc-variant` is passed. All 163 execution-record headers logged by the frozen `ours`
regenerate byte-identically from the Phase 3 code.

## 2. Environment

- Hardware: 1× RTX 3080 Laptop (16 GB), Ubuntu, NVIDIA driver 535.154.05.
- Python 3.10.14 in `.venv` (torch 2.3.0+cu121); exact package versions in `requirements_actual.txt`.
  Editable installs: `.`, `capx/third_party/{robosuite,sam3,contact_graspnet_pytorch}` (submodules at
  `robosuite@9729273`, `sam3@6fe87d6`, `contact_graspnet_pytorch@da3dcfb`).
- Vendor patches: `scripts/apply_vendor_patches.sh` (idempotent).
- SAM3 weights: `checkpoints/sam3/sam3.pt` (symlink; not in git).
- Models (campus OpenAI-compatible gateway): coding `deepseek-v4.1-flash-1017`, VDM `qwen3.8-flash`, both at
  temperature 0.2 with reasoning disabled (`capx/llm/client.py::MODEL_PAYLOAD_EXTRAS`). The key is read from
  `CAPX_API_KEY` or `CAPX_CREDENTIAL_FILE` (default `../api_key.md`) and never written to configs or logs.
- Services (SAM3 :8114, Contact-GraspNet :8115, PyRoKi IK :8116). Always launch with a log directory: without
  one, the server stdout pipe fills up and the IK server hangs.
  ```bash
  .venv/bin/python capx/serving/launch_servers.py --config-path env_configs/harness_eval/cube_lifting.yaml \
      --log-dir logs/services/run_$(date +%Y%m%d_%H%M%S)
  PYTHONPATH=. .venv/bin/python scripts/check_services.py      # health check (also run before every episode)
  ```

## 3. Uniform fixes to official CaP-X (`53e9966`) — identical for all conditions

| Id | Defect found in the official code | Fix (files) |
|---|---|---|
| E1 | `env.reset(seed)` did not seed Robosuite's own RNG, so a seed did not reproduce the scene | `_seed_robosuite_rng()` seeds the env RNG and placement samplers (`capx/envs/simulators/robosuite_base.py`, called in `robosuite_cube_lift.py`, `robosuite_cubes.py`, `robosuite_cubes_restack.py`, `robosuite_spill_wipe.py`); evaluation-only `get_initial_state_record()` (qpos hash) |
| E2 | Ground truth reachable by agent code: `cube_pos`, `object-state`, `gripper_to_cube*`, `proportion_wiped` in `obs`/`get_observation()`, plus the simulator handle `env` and `APIS` in the sandbox | `sanitize_agent_observation()` keeps camera and robot proprioception only; `env`/`APIS` removed for non-privileged configs (`capx/envs/tasks/base.py`, `capx/integrations/franka/control_reduced.py`) |
| E3 | Stale observations after motion: `get_observation()` returned the pre-move image and Cartesian pose | `_get_observations(force_update=True)` + `_refresh_gripper_link_pose()` (`robosuite_base.py`, task envs) |
| E4 | Molmo point-prompt tool documented but unservable on this GPU | `FrankaControlApiReducedNoMolmo` registry entry (`capx/integrations/__init__.py`, `control_reduced.py`) |
| E5 | GraspNet autograd memory growth (OOM); SAM3 gated HF download | `torch.no_grad()` + cache release; local checkpoint (`patches/*.patch`) |

Other shared infrastructure:
- LLM client usage and latency reporting, plus a retry on HTTP 429 (`capx/llm/client.py`).
- Per-episode telemetry and hard budgets (`capx/envs/telemetry.py`).
- Fresh process per episode with no reruns (`scripts/run_paired_eval.sh`).

## 4. Experimental conditions (Step 3 only; `capx/envs/eg_vdm.py`)

| | baseline | ablation | ours |
|---|---|---|---|
| VDM input | 2 images + task prompt (official M4) | same + generic verification checklist | same + execution & proprioception record (no code) |
| VDM output | free text | structured verdict (COMPLETE / NOT COMPLETE / UNCERTAIN) | structured verdict |
| Feedback slot | VDM text | `[Generic Verification Checklist]` + assessment | `[Execution & Proprioception Record]` + assessment |
| Zero-motion gate | — | — | no VDM call; deterministic verdict when no motion primitive ran |

Budgets (all conditions): 8 code executions, 16 model calls, 60,000 tokens, 1500 sim steps and 600 s per
episode (`env_configs/harness_eval/*.yaml`; protocol in `docs/phase2/FROZEN_PROTOCOL.md`).

## 5. Reproducing

**Recompute every table from raw logs (offline, stdlib Python, no model calls):**
```bash
scripts/reproduce_all_tables.sh                # writes reports/tables/*.md|json
scripts/reproduce_all_tables.sh --with-tests   # + offline unit tests (project venv)
```

**Re-run experiments (needs services and the gateway; creates new episodes and never overwrites existing ones):**
```bash
# Phase 1 dev baseline
scripts/run_harness_eval.sh dev_baseline baseline "cube_lifting cube_stack spill_wipe cube_restack" "0 1 2"
# Phase 2 frozen paired test (checkout phase2-freeze first)
git checkout phase2-freeze
scripts/run_paired_eval.sh test_eval "cube_lifting cube_stack cube_restack" "$(seq -s ' ' 100 119)"
# Phase 3 post-hoc verification (checkout phase3-final)
git checkout phase3-final
scripts/run_posthoc.sh matched "ref_only gate_only ref_and_gate" "102 107 109 110 114 116"
scripts/run_posthoc.sh raw     "ref_only gate_only ref_and_gate" "102 107 109 110 114 116"
```

## 6. Raw logs

| Path | Content |
|---|---|
| `logs/dev_baseline/` | Phase 1 dev baseline, 12 episodes |
| `logs/dev_phase2*/` | Phase 2 dev sanity checks (v1, v2 superseded; final `dev_phase2/`), invalidated service-outage episodes |
| `logs/test_eval/` | **frozen primary result**: 180 episode JSONs + console + per-turn images/videos, `RUN_INFO.txt`, incident note |
| `logs/test_eval_invalid/` | 2 episodes replaced under the pre-registered infrastructure-fault rule |
| `logs/posthoc_diagnostic/` | Phase 3 post-hoc root-cause verification (**not** part of the frozen evaluation) |
| `logs/engineering/` | smoke and budget-cap verification episodes |

Every episode JSON records:
- task, seed, mode and initial-state hash;
- per turn: code, stdout/stderr, VDM prompt/output, feedback slot, execution record, proprioception, timing,
  and the log-only `turn_eval_success`;
- per call: model and token usage;
- per episode: tool and perception counts, sim steps, wall time, termination reason and official evaluation.

## 7. Limitations of the package

- No OS-level sandbox: agent code runs in-process and only interface-level access is removed.
- The gateway VDM `qwen3.8-flash` is not a dated snapshot. Returned model ids are logged per call.
- Public GitHub repository: `https://github.com/OPlincn/Cap-X-Examine` (with tags `phase1`, `phase2-freeze`, `phase2-results`, and `phase3-final`).
