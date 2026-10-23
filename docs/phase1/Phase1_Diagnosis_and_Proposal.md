# Phase 1 — Environment Setup, Telemetry, Dev-Set Diagnosis, and Harness+VDM Proposal

*Date: 2026-10-23. Scope: Development set only (seeds 0, 1, 2). No test-set seed (100–119) has been run.*

---

## 1. Environment & Baseline Setup Summary

### 1.1 Repository and runtime

| Item | Value |
|---|---|
| Repo | `cap-task/cap-x`, branch `phase1`, based on official CaP-X commit `53e9966` |
| Our commits | `903097d` uniform environment fixes · `e6c2512` budget enforcer + telemetry · `4121a91` cube_restack config + triage script |
| Submodules | `robosuite@9729273`, `contact_graspnet_pytorch@da3dcfb`, `sam3@6fe87d6` freshly fetched (pristine), plus two vendor patches in `patches/` (applied by `scripts/apply_vendor_patches.sh`) |
| Python env | `cap-x/.venv` (Python 3.10.14, torch 2.3.0+cu121), copied from the working Test-wh env and re-pointed at the new repo; exact freeze in `requirements_actual.txt` |
| Weights | `checkpoints/sam3 → /home/oplin/PythonProjects/Test-wh/cap-x/checkpoints/sam3` (symlink, no copy) |
| Services | SAM3 :8114 (GPU), Contact-GraspNet :8115 (GPU), PyRoKi IK :8116 (CPU), launched with `capx/serving/launch_servers.py`; log in `logs/services/` |
| GPU | 1× RTX 3080 Laptop 16 GB |

### 1.2 Fixed models (from the campus gateway in `api_key.md`)

`api_key.md` contains only the endpoint and key, so the models were chosen from the gateway's `/models` list and **frozen in the configs**:

| Role | Model | Settings | Why |
|---|---|---|---|
| Coding agent | `deepseek-v4.1-flash-1017` (dated snapshot) | temperature 0.2, `thinking: disabled`, ≤ 8192 completion tokens | Pinned snapshot, keeps token cost predictable under a per-episode token cap |
| VDM | `qwen3.8-flash` | temperature 0.2, `enable_thinking: false` | Of the vision-capable models probed (qwen3.8-flash, qwen3.7-plus, kimi-k2.6), all described a real Robosuite render correctly; qwen3.8-flash was fastest and correctly reported the open gripper. `glm-5.3` ignored images; deepseek is text-only |

The key is read from `CAPX_API_KEY` (populated by the run script from `../api_key.md`); it never appears in configs, CLI args or logs.

### 1.3 M4 setting (guardrail check)

Configs: `cap-x/env_configs/harness_eval/{cube_lifting,cube_stack,cube_restack,spill_wipe}.yaml`. Each is a copy of the official `*_multiturn_vdm_reduced_api.yaml` with **unchanged task / multi-turn prompts**.

- S3 low-level API with in-context docstring examples (`FrankaControlApiReduced`).
- Multi-turn execution feedback (stdout/stderr) plus image VDM: an initial scene description and per-turn before/after differencing.
- Parallel ensembling is off (`use_parallel_ensemble: false`). There are no skill-library APIs and `evolve_skill_library` is off.
- No oracle code, no hand-written task solutions, no new high-level primitives.

### 1.4 Environment defects found in the official code (fixed identically for every condition)

| # | Defect (verified by test) | Impact if left | Fix |
|---|---|---|---|
| E1 | `env.reset(seed=s)` did **not** control the scene: seed 0 twice → different cube poses | Paired evaluation impossible | `_seed_robosuite_rng()` seeds robosuite's own RNG and placement sampler. Verified that a seed reproduces identical `qpos` **and** identical rendered image on all 4 tasks; dev-run initial-state hashes match the standalone test |
| E2 | **Ground-truth leak**: agent `obs`, `INPUTS` and the API's `get_observation()` contained `cube_pos`, `cubeA_pos`, `object-state`, `gripper_to_cube*`, and for spill wipe `proportion_wiped`, `wipe_centroid` and `wipe_radius`. The sandbox also exposed the simulator handle `env` (→ `env.task_completed()`) and `APIS` | Violates the information barrier | `sanitize_agent_observation()` keeps only camera data, robot proprioception and the prompt. `env` and `APIS` are removed from the non-privileged sandbox |
| E3 | **Stale observations**: after `move_to_joints`, `get_observation()` returned the *pre-move* RGB image and `robot_cartesian_pos` (mean pixel diff 0.0 after a large arm move) | Perception in turn *t+1* sees the world as it was before turn *t* | `_get_observations(force_update=True)` plus a refresh of the gripper link pose. After the fix, the image changes and matches `render()` exactly |
| E4 | Molmo point-prompt service has no server (it cannot fit next to SAM3 and GraspNet on 16 GB) | A documented tool that always fails | Registered `FrankaControlApiReducedNoMolmo` (S3 API minus `point_prompt_molmo`) |
| E5 | GraspNet accumulated autograd memory; SAM3 tried a gated HF download | OOM / startup failure | `torch.no_grad()` + cache release; local checkpoint (`patches/`) |

The interface-level barrier is not an OS sandbox: generated Python can still import modules. Robosuite's horizon also acts as a native simulator-step cap.

### 1.5 Budget enforcer and telemetry (verified)

`capx/envs/telemetry.py` (+ hooks in `trial.py`, `runner.py`, `tasks/base.py`):

| Cap (per episode, identical across conditions) | Value | Enforcement |
|---|---|---|
| `max_turns` (code executions) | 10 | Checked before each execution. After the 10th execution no VDM or decision call is made, because none could change the outcome |
| `max_model_calls` (coding + VDM) | 24 | Checked before each call |
| `max_total_tokens` (coding + VDM, in + out) | 200,000 | Pre-call reservation (estimated input + completion cap = remaining budget). Unknown usage is charged at the reservation, never zero. Output of an over-budget call is discarded, never executed |
| `max_sim_steps` | task horizon (1500; spill wipe 4000) | Guard on every motion primitive, raised as `BaseException` so `except Exception` in generated code cannot swallow it. May overshoot by ≤ 1 primitive (observed 107 for a cap of 100) |
| `max_wall_time_s` | 1200 (+120 s SIGALRM backstop) | Checked before every execution and model call |

Other protocol properties:
- Fresh process per episode (fresh agent, API clients and simulator); continuous state within an episode.
- `max_trial_retries: 1`, so there are no reruns. Timeouts and crashes are logged as failed episodes.

**Verification.**
- Five offline unit tests pass (`tests/test_episode_telemetry.py`).
- Four live budget tests each terminated with the correct reason (`logs/engineering/budgettest_*`): `max_sim_steps` (107 steps), `max_total_tokens`, `max_turns`, `max_model_calls`.

**Episode JSON** (`logs/dev_baseline/<task>/seed_<k>/episode_<task>_seed<k>_baseline.json`):
- **Header:** `task_name`, `seed`, `mode`, `initial_state` (qpos + hash).
- **Per-turn:** `generated_code`, `stdout`, `stderr`, `vdm_prompt` (text + saved image filenames), `vdm_output`, `code_exec_success`, `exec_wall_time`, `turn_wall_time`, `sim_steps_delta`, `tool_calls_delta`, `decision_after`, and links into `model_calls[]`, which holds per-call input/output tokens, latency, finish reason and returned model id.
- **Episode costs:** `coding_llm_calls/input/output_tokens`, `vdm_calls/input/output_tokens`, 10 counted tools, `extra_perception_calls`, `total_sim_steps`, `total_wall_time`, `timeout_or_budget_exceeded`, `termination_reason`.
- **Post-episode only:** `official_task_success`, `official_reward`, `episode_code_exec_success_rate`. These are computed after the agent loop ends and are never placed in a prompt.

**Reproduce the table:** `python3 scripts/summarize_episodes.py logs/dev_baseline --expect-seeds 0,1,2` (stdlib only; flags missing or duplicate episodes).

---

## 2. Dev Set Quantitative Summary (unmodified M4 baseline)

Three dev seeds per task. Cube Lift, Cube Stack and Spill Wipe were the planned tasks. They were solved 9/9, so **Cube Restack** (official M4 config, same API/protocol) was added to find genuine failures.

| Task | Success / 3 | Code exec success | Avg turns | Avg coding tok | Avg VDM tok | Avg total tok | Avg model calls | Avg perception calls | Avg sim steps | Avg wall time |
|---|---|---|---|---|---|---|---|---|---|---|
| cube_lifting | **3/3** | 82.4% (14/17) | 5.7 | 33,634 | 17,421 | 51,055 | 13.3 | 9.0 | 773 | 151 s |
| cube_stack | **3/3** | 66.7% (4/6) | 2.0 | 13,097 | 7,360 | 20,457 | 6.0 | 3.0 | 243 | 52 s |
| spill_wipe | **3/3** | 80.0% (4/5) | 1.7 | 9,963 | 7,102 | 17,065 | 5.3 | 1.3 | 390 | 51 s |
| cube_restack | **2/3** | 57.1% (4/7) | 2.3 | 16,765 | 8,284 | 25,049 | 6.7 | 3.3 | 360 | 81 s |
| **All** | **11/12** | 74.3% (26/35) | 2.9 | — | — | 28,406 | — | — | — | — |

No episode hit a budget cap; all 12 ended with the agent's FINISH. The largest episode used 73.6k tokens and 18 model calls, against caps of 200k and 24.

**Aggregate failure statistics (35 turns, 12 episodes).**
- **9 failed executions, all before any robot motion** (`sim_steps_delta = 0`):
  - 6 × `SyntaxError` in turn 0. They hit **6/12 episodes** and 3/3 restack episodes. The model returned two fenced blocks separated by prose; the official `_extract_code` keeps everything from the first ```` ```python ```` to the *last* ```` ``` ````, so the prose and the inner fences were executed.
  - 3 × `AssertionError` inside `plan_grasp` (GraspNet found no grasp for a cube already in the gripper). Two of them were consecutive repeats in the same episode.
- **9/9 of these zero-motion turns still triggered a VDM call** on two images of an unchanged scene: 23,185 tokens, which is 19% of all VDM tokens and 6.8% of all tokens.
- **In 1 of these 9 calls the VDM hallucinated task completion, and the agent finished on it.** This is the only dev failure.
- In **3/3 cube-lift episodes** the goal was reached early (turn 0 or 1). The VDM reported "cube still on the table", and the agent's next routine began with `open_gripper()` and dropped the cube. This explains why the easiest task has the most turns (5.7) and tokens (51k).

---

## 3. Concrete Failure Case Studies

Only one dev *episode* failed. The second and third studies are failed sub-trajectories inside successful episodes: turns that undid an achieved goal or repeated a crash. They show the same mechanism and they drive the cost numbers.

### Case 1 — `cube_restack`, seed 0 (episode FAILED): VDM declares an untouched scene "completed" after a zero-motion crash → premature FINISH (Patterns A + C)

The initial VDM description was **correct**:
> "A **green cube** is resting on top of a **red cube**. … Currently, the configuration is the opposite (green on top of red)."

Turn 0's extracted code contained the prose between the model's two fenced blocks:
```python
print("red center", red_center, "extent", red_extent)
```

Now I'll grasp the green cube (top one) and move it aside.

```python
# Plan grasp for green cube
```
`stderr`:
```
  File "<string>", line 48
    Now I'll grasp the green cube (top one) and move it aside.
SyntaxError: unterminated string literal (detected at line 48)
```
`sim_steps_delta = 0`; no tool was called, so the before and after images are pixel-identical (`visual_feedback_00/01.png`: green still on red).

`vdm_output` (turn 0):
> "The current state of the environment is identical to the previous state. **The red cube remains on top of the green cube**, and the gripper appears to be in an open position. **The task has been completed successfully.**"

Coding agent's decision: `FINISH`. Official result: `official_task_success = False`, reward 0.0012.

**Mechanism.**
- The VDM was asked to judge completion from two identical 2D images, with no knowledge that the code had crashed at line 48 before doing anything.
- It reversed its own correct initial description: a colour/stacking-order hallucination on a 2D view.
- The coding agent trusted the VDM over its own `stderr`, which proved nothing had executed.
- Nothing in the harness distinguishes "no robot action happened" from "something changed".

### Case 2 — `cube_lifting`, seed 0 (turns 1–7): VDM misses an achieved lift → agent re-runs a pick that drops the cube, three times (Pattern A, false-negative direction, + Pattern B)

Turn 1 code (excerpt) executed without error and printed `done`:
```python
joints_grasp = solve_ik(target_pos, quat); move_to_joints(joints_grasp); close_gripper()
lift_pos = target_pos + np.array([0,0,0.15])
joints_lift = solve_ik(lift_pos, quat); move_to_joints(joints_lift)
print("done")
```
`vdm_output` (turn 1):
> "The gripper fingers are now surrounding the sides of the cube, indicating that a grasp attempt is in progress or has just been made. The task of picking up and lifting the red cube is not yet completed, **as the cube remains on the table surface**."

The after-image (`visual_feedback_02.png`) shows the cube held between the fingers near the gripper. From this frontal camera a lifted cube and a resting cube differ only slightly in 2D.

Turn 2 began by re-perceiving the cube. Its **own stdout** shows the cube was in the air (z = +0.039; the table-level value in turn 1 was −0.095):
```
center [ 0.55964058 -0.05716019  0.03920575] extent [...]
```
…but it continued with the full pick routine, which opens the gripper first:
```python
move_to_joints(joints_pre)
open_gripper()          # ← drops the lifted cube
```
`vdm_output` (turn 2): "the red cube resting on the table surface … the robot has just dropped it."

The same cycle (lift → VDM "not lifted" → re-pick opens gripper → drop) repeated in turns 3–4 (stdout z = +0.080) and 5–6 (z = +0.042). The episode finally succeeded in turn 8 after **8 turns, 73.6k tokens, 200 s**, instead of one or two turns.

The same pattern occurred in **seed 1** (lifted in turn 0, measured z = +0.031 in turn 1, VDM "still resting on the table") and **seed 2** (lifted in turn 0; see Case 3).

**Mechanism.**
- The VDM's verdict is purely 2D and code-blind. It does not know the executed code ended with a 15 cm lift that completed without error.
- The coding agent receives an authoritative-sounding "not completed" with no evidence to weigh against its own execution trace.
- It has no habit of checking residual state (gripper holding object) before re-running a routine that begins with `open_gripper()`.

### Case 3 — `cube_lifting`, seed 2 (turns 1–2): identical crash repeated, VDM called twice on identical images (Pattern C)

Turns 1 and 2 both called `plan_grasp` on the cube already held in the gripper; GraspNet returned no grasps. `stderr` was identical in both turns:
```
  File ".../control_reduced.py", line 434, in plan_grasp
    vtf.SE3.from_matrix(self._env.grasp_sample) @ vtf.SE3.from_translation(np.array([0, 0, 0.12]))
AssertionError
```
`sim_steps_delta = 0` in both turns. The VDM was still called both times:
> T1: "The robot gripper is still holding the red cube … **The task of picking up the red cube has been completed, but the lifting action has not yet occurred** as the cube's height relative to the background has not increased."
> T2: "The current state is identical to the previous state. The red cube remains grasped by the gripper …"

The cube was actually already lifted. Turn 3 then re-grasped from a higher pose, which dropped it, and turn 4 re-lifted.

**Mechanism.**
- Two VDM calls (about 5k tokens) on pixel-identical images added no information.
- One of them asserted a false 3D judgement ("not lifted").
- Neither told the agent the most useful fact: the crash happened before any motion, and the robot is still holding the cube where the previous turn left it.

### Counter-examples where the VDM helped (kept for balance)

- **spill_wipe seed 2:** "a spot of the brown spill is still clearly visible" led to a second wipe and success.
- **cube_stack seed 1:** after a correct stack, "the gripper remains closed … opening the gripper has not been performed" prompted `open_gripper()` and FINISH.

Any modification must preserve these.

---

## 4. Proposed Single Core Modification (Harness + VDM Integration)

### 4.1 Hypothesis (one, testable)

> **H:** The M4 VDM produces wrong completion verdicts because it judges two 2D images without knowing what the turn's code did. It misses "nothing executed" turns, and 3D outcomes such as lift height are ambiguous in 2D. Giving the VDM the turn's own execution record turns its output into an execution-grounded verification:
> - the executed code;
> - exception status and line;
> - whether any motion primitive ran;
> - the agent's own stdout;
> - before/after proprioception (end-effector position, gripper opening).
>
> When no motion occurred, the harness replaces the VDM call with a deterministic "no physical change" report. Together these **reduce wrong VDM verdicts, and therefore premature FINISH and goal-undoing regenerations, increasing success at equal or lower model cost.**

Falsifiable predictions on the paired test set (see 4.4 for metrics):
1. VDM verdict accuracy, measured against per-turn evaluation-only success labels: `ours > baseline` and `ours > ablation`.
2. "Goal-undone" events (official success at the end of some turn but not at episode end) and premature FINISH: `ours < baseline`.
3. Task success: `ours ≥ baseline`. Headroom is mainly on cube_restack, since lift, stack and wipe are near ceiling.
4. Average VDM calls and tokens: `ours ≤ baseline`, because zero-motion turns skip the VDM.

If (1) holds but (3) does not, the report will say so: a better VDM that does not translate into success.

### 4.2 Exactly which step of CaP-Agent0 (M4) is modified

M4 loop per turn *t*:
1. The coding agent emits code.
2. Sandbox execution produces stdout/stderr.
3. **VDM step:** capture the after-image; the VDM compares the before/after images given only the task prompt; its text is appended to the decision prompt.
4. The coding agent decides `REGENERATE` + code, or `FINISH`.

**We modify only step 3**, the construction of the VDM input and whether the VDM is queried. Steps 1, 2 and 4 are unchanged: same prompts, parser, API, perception modules, budgets and stdout/stderr feedback. The coding agent's multi-turn prompt template is untouched; it simply receives a different VDM text in the existing slot. Reused mechanisms are the M4 VDM model, the image capture, the feedback slot and the decision parser. Our change is the *Execution-Grounded VDM with a no-motion gate*.

### 4.3 The three experimental groups

| Factor | `baseline` | `ours` (EG-VDM) | `ablation` (generic-verification VDM) |
|---|---|---|---|
| Coding model, VDM model, temperatures | fixed (§1.2) | same | same |
| Task prompts, multi-turn template, parser, API, perception, sim | official M4 | same | same |
| Budgets (§1.5), telemetry, seeds | same | same | same |
| VDM system/user instructions | official | structured: describe change → physical-state check (gripper holding? objects' relative 3D placement) → verdict `COMPLETE / NOT COMPLETE / UNCERTAIN` with evidence | **identical structured instructions to `ours`** |
| VDM sees execution record (code, exception + line, motion-primitive counts, stdout tail, EEF/gripper before/after) | no | **yes** | **no**; replaced by a length-matched generic checklist ("verify carefully; 2D views can be misleading about height and stacking") |
| Zero-motion gate (no motion primitive ran → no VDM call; deterministic report "No robot motion occurred this turn (code failed at line N: <exception>). The scene is physically unchanged; the previous visual assessment still applies: <previous verdict>") | no | **yes** (driven by the execution record) | **no**: VDM always called |

The ablation removes the core mechanism (execution grounding and the gate it enables) while keeping everything else in `ours`, including the structured verdict format and extra verification wording. If `ours ≈ ablation`, the gain comes from prompt restructuring, not grounding, and H is rejected.

Information-barrier check: every field in the execution record is produced by the agent's own code or by its own robot proprioception, which the agent can already query through `get_observation()["robot_cartesian_pos"]`. No reward, success flag, object pose or simulator state enters the VDM or the coding prompt. Motion counts are the harness's tally of the agent's own tool calls.

### 4.4 Telemetry additions for Phase 2 (logging only, identical in all three groups)

- `turn_eval_success` / `turn_eval_reward`: official check evaluated after each turn and written **only to the log**, never to a prompt. This enables the metrics VDM verdict accuracy, goal-undone events and premature FINISH.
- `proprio_before/after` per turn (non-privileged).
- `vdm_skipped_by_gate` flag; VDM verdict label parsed from the output.

### 4.5 Proposed code changes (pseudo-diff for review)

```python
# capx/envs/trial.py — in the turn loop (all modes): record the agent's own execution trace
proprio_before = sanitize_agent_observation(env.low_level_env.get_observation())["robot_cartesian_pos"]
obs_next, reward, terminated, truncated, info_step = env.step(code)
exec_record = {
    "code": code,
    "exec_ok": info_step["sandbox_rc"] == 0,
    "exception": last_exception_line(info_step["stderr"]),   # e.g. 'SyntaxError ... line 48'
    "motion_calls": {k: turn_tools.get(k, 0) for k in MOTION_TOOLS},
    "sim_steps_delta": tele.sim_steps() - steps_before,
    "stdout_tail": info_step["stdout"][-1500:],
    "proprio_before": proprio_before,                       # EEF xyz, gripper opening
    "proprio_after": sanitize_agent_observation(env.low_level_env.get_observation())["robot_cartesian_pos"],
}
turn_record["turn_eval_success"] = bool(env.low_level_env.task_completed())   # LOG ONLY

# capx/envs/trial.py — _handle_multi_turn_step, step 3 (VDM) only
if mode == "ours" and sum(exec_record["motion_calls"].values()) == 0:
    differencing_feedback = NO_MOTION_TEMPLATE.format(
        exception=exec_record["exception"] or "none", previous=last_vdm_verdict)
    vdm_skipped_by_gate = True                          # no VDM call, no tokens
else:
    differencing_feedback = _get_visual_differencing_feedback(
        visual_differencing_args, task_description, visual_feedback_base64_history,
        prompt_style="baseline" if mode == "baseline" else "structured",
        exec_record=exec_record if mode == "ours" else None,
        generic_checklist=(mode == "ablation"),         # length-matched placebo context
    )
# decision prompt construction, model query and parsing: unchanged for all modes
```

Estimated size: about 120 lines (one new helper module for templates and execution-record formatting, plus small edits in `trial.py`). No change to prompts in `env_configs`, the parser, APIs or budgets.

### 4.6 Considered and deliberately excluded

- **Fixing `_extract_code`** (the most frequent crash, 6/12 episodes). It is a different core mechanism (output parsing), and fixing it in only one group would confound the comparison. We keep the official parser in **all** groups; under `ours` such turns are handled by the gate.
- **A separate FINISH verification gate in the coding loop.** It would be a second mechanism, and with only one premature FINISH observed it is weakly motivated. EG-VDM addresses its cause.
- **Task-specific state classifiers or completion rules.** Excluded: they would amount to hand-written task knowledge.

### 4.7 Decisions needed before freezing (Phase 2)

1. **Test tasks** (20 paired seeds `100..119` × 3 groups each). Recommended: **cube_restack** (headroom, Pattern A/C failure) and **cube_lifting** (goal-undoing, cost), optionally cube_stack. Spill wipe showed no harmful VDM behaviour.
2. **Budgets stay as in §1.5.** The 200k token cap never bound on dev (max 73.6k); it is kept as a fixed safety cap. Lower it before freezing if a binding token budget is preferred.
3. **Small-sample caveat.** Dev evidence is 12 episodes. The failure mechanisms are consistent (3/3 lift goal-undoing, 9/9 zero-motion VDM calls), but frequencies at n = 20 per task will be noisy; we will report paired wins/losses with exact counts rather than claim significance.

---

## Appendix — Artifacts

| Path (under `cap-task/cap-x/`) | Content |
|---|---|
| `logs/dev_baseline/<task>/seed_<k>/episode_*.json` | Per-episode telemetry (12 files) |
| `logs/dev_baseline/<task>/seed_<k>/console.log`, `trial_*/` | Console, generated code, VDM before/after images, per-turn videos |
| `logs/dev_baseline/summary_table.md`, `summary.json` | Table in §2, recomputable with `scripts/summarize_episodes.py` |
| `logs/dev_baseline/failure_tags.{txt,json}` | Pattern triage from `scripts/analyze_failures.py` |
| `logs/engineering/` | Smoke and budget-cap verification episodes (not dev results) |
| `env_configs/harness_eval/*.yaml` | M4 configs used (prompts identical to official) |
| `git diff 53e9966..phase1` | All changes relative to the official code |
