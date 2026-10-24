# Human–AI Refinements Before the Phase 2 Code Freeze

This file records the "before vs. after" of the changes made between the Phase 1 proposal
(`docs/phase1/Phase1_Diagnosis_and_Proposal.md`, written by the AI agent) and the frozen Phase 2
implementation. Refinements 1 and 2 came from the human review (`Argon_Phase2_Task.md`). Refinements 3a/3b
were found by the AI agent on the development set while verifying Refinement 1. All three were made
**before** any test-set seed (100–119) was run.

---

## Refinement 1 — Step-3 mechanism and edge cases (human review)

### 1a. Turn-0 boundary bug in the zero-motion gate

**Before (Phase 1 §4.5 pseudo-code):**
```python
if mode == "ours" and sum(exec_record["motion_calls"].values()) == 0:
    differencing_feedback = NO_MOTION_TEMPLATE.format(
        exception=exec_record["exception"] or "none", previous=last_vdm_verdict)
```
Problem found by the reviewer: on turn 0, or when every turn so far had zero motion, `last_vdm_verdict`
is `None` or the *initial scene description*. That is exactly the `cube_restack` seed-0 situation
(SyntaxError on turn 0). The template could then forward an undefined or misleading "previous"
assessment and still leave room for a premature FINISH.

**After (`capx/envs/eg_vdm.py`):** an explicit per-episode state, initialised to a defined NOT COMPLETE
verdict, updated only after VDM calls on turns with robot motion. The gate always ends with an explicit
verdict line.
```python
INITIAL_MOTION_VERDICT = "NOT COMPLETE (No physical robot motion has been executed in this episode yet)"

@dataclass
class Step3State:
    last_valid_motion_verdict: str = INITIAL_MOTION_VERDICT
    last_valid_motion_label: str = "NOT COMPLETE"
    last_motion_turn: int | None = None

def gate_message(rec, state):
    ...
    if label == "COMPLETE":      # preserve, and warn against undoing it
        verdict = (f"Verdict: COMPLETE (preserved from turn {state.last_motion_turn}, the last turn with robot "
                   "motion) — Zero robot motion occurred this turn. Do NOT undo the completed state (do not "
                   "open the gripper, re-grasp or move the objects); output FINISH if the task is complete.")
    elif label == "UNCERTAIN":   # (small extension: an UNCERTAIN status is preserved as UNCERTAIN)
        verdict = (f"Verdict: UNCERTAIN (preserved from turn {state.last_motion_turn}) — Zero robot motion "
                   f"occurred this turn ({cause}). Verify the state with perception before deciding; ...")
    else:
        verdict = (f"Verdict: NOT COMPLETE — Zero robot motion occurred this turn ({cause}). Do NOT output "
                   "FINISH; fix the code error and execute the manipulation.")
```
Verified live on the dev set (`logs/dev_phase2*/cube_restack/ours/seed_1`, turn 0):
```
Code execution: FAILED at line 57: SyntaxError: unterminated string literal (detected at line 57)
Zero-Motion Gate: no robot motion primitive ... was executed this turn ...
Last assessment after robot motion: NOT COMPLETE (No physical robot motion has been executed in this episode yet)
Verdict: NOT COMPLETE — Zero robot motion occurred this turn (execution failed at line 57: ...). Do NOT output FINISH; ...
```
No VDM tokens were spent on that turn, and the agent fixed the code instead of finishing.

### 1b. Expose the execution & proprioception record in the Step-3 feedback slot

**Before (Phase 1 §4.5):** `exec_record` was passed *only into the VDM prompt*. The coding agent
(Step 4) saw just the VDM's text, so after an `UNCERTAIN` / `NOT COMPLETE` verdict it had no factual
gripper/EEF evidence and could still restart with `open_gripper()` and drop a held cube (Phase 1, Case 2).
```python
differencing_feedback = _get_visual_differencing_feedback(..., exec_record=exec_record if mode == "ours" else None, ...)
```

**After:** still only Step 3 changes, but the string returned into the existing VDM feedback slot has
two sections in `ours`, and a length/structure-matched pair of sections in `ablation`.
```python
# capx/envs/trial.py::_get_step3_feedback
if mode == "ours":
    header = eg_vdm.format_exec_header(exec_record)              # [Execution & Proprioception Record]
    if exec_record["num_motion_calls"] == 0:
        assessment, label = eg_vdm.gate_message(exec_record, step3_state)   # no VDM call
        return eg_vdm.compose_slot(mode, header, assessment)
    prompt = eg_vdm.structured_vdm_prompt(..., exec_record=exec_record, generic_checklist=False)
elif mode == "ablation":
    header = eg_vdm.GENERIC_SLOT_HEADER                          # [Generic Verification Checklist]
    prompt = eg_vdm.structured_vdm_prompt(..., exec_record=None, generic_checklist=True)  # always called
...
return eg_vdm.compose_slot(mode, header, assessment)
# compose_slot: f"{header}\n[Visual Differencing Assessment]\n{assessment}"
```
Example of what the coding agent now receives (dev set, `cube_restack` seed 0, `ours`, turn 0):
```
[Execution & Proprioception Record]
- Code execution: completed without exception
- Robot motion calls this turn: move_to_joints×5, open_gripper×2, close_gripper×1
- Perception/planning calls this turn: get_observation×1, segment_sam3_text_prompt×2, ...
- Simulator steps this turn: 211
- End-effector xyz (robot base frame, m): before [+0.438, +0.008, +0.185] -> after [+0.639, +0.010, +0.202] (Δ [+0.201, +0.002, +0.017])
- Gripper opening (1.0 = fully open, 0.0 = fully closed ...): before 0.21 -> after 0.99
[Visual Differencing Assessment]
Physical Change: ...
Gripper & Occlusion Check: ...
Verdict: NOT COMPLETE — ...
```
In `ablation` both the VDM instructions (`STRUCTURED_VDM_INSTRUCTIONS`) and the slot structure are
identical. The execution record is replaced by fixed generic text in both places
(`GENERIC_VDM_CHECKLIST`, ≈1.0k characters, in the VDM prompt; `GENERIC_SLOT_HEADER`, ≈0.4k characters,
in the slot), and there is no gate.

---

## Refinement 2 — Binding inference budget and task selection (human review)

**Before (Phase 1 §1.5, non-binding):** the largest dev episode used 73.6k of 200k tokens and 18 of 24 calls.
**After:** caps adjusted uniformly for all modes and frozen (`env_configs/harness_eval/*.yaml`):
```diff
 budget:
-  max_turns: 10            # code executions per episode
-  max_model_calls: 24      # coding + VDM calls per episode
-  max_total_tokens: 200000 # coding + VDM input+output tokens per episode
+  max_turns: 8             # code executions per episode (Phase 2, binding)
+  max_model_calls: 16      # coding + VDM calls per episode (Phase 2, binding)
+  max_total_tokens: 60000  # coding + VDM input+output tokens per episode (Phase 2, binding)
   max_sim_steps: 1500      # = robosuite horizon of this task
-  max_wall_time_s: 1200
+  max_wall_time_s: 600
   max_completion_tokens: 8192
```
The 60k cap is now binding: on the dev set an `ablation` episode stopped with
`max_total_tokens: used 59386, next call needs ~3769+256` after six unsuccessful grasp turns.

**Task selection — before (Phase 1 §4.7):** cube_restack + cube_lifting, cube_stack optional.
**After:** `cube_lifting`, `cube_stack`, `cube_restack` × seeds 100–119 × {baseline, ours, ablation}
= 180 paired episodes (`scripts/run_paired_eval.sh`, mode order rotated per seed).

---

## Refinement 3 — Code intent vs. physical outcome (found by the AI agent on the dev set)

### Refinement 3a — per-command gripper readings and an intent-vs-outcome instruction

**Observation (dev set, before the fix; preserved in `logs/dev_phase2_v1/cube_restack/ours/seed_2`):**
the grasp in turn 3 missed and the red cube stayed on the table, far from the green cube. The grounded VDM
nevertheless answered:
```
Physical Change: The red cube has moved from the left side of the workspace to a position on top of the green cube.
Gripper & Occlusion Check: The gripper is open and empty, located above the stacked cubes. ...
Verdict: COMPLETE — The red cube is successfully placed on top of the green cube, ...
```
The agent finished, and the episode failed. Diagnosis:
1. The VDM adopted the **intent** of the executed code (`# Place on top of green cube`) over the image.
2. The record only showed the gripper at turn start and end (1.00 → 0.99), hiding that the fingers had
   closed on nothing mid-turn.

A probe in the simulator measured gripper opening after `close_gripper()`: 0.015 when closing on nothing,
0.65 when closing on a cube (0.64 while lifted).

**Before:**
```python
# eg_vdm.format_exec_context_for_vdm
"Use this context to interpret the images (for example, a lift command that executed without error and a "
"closed gripper that stayed partially open suggest an object is held), but report what the images show."
# exec record: proprio only at turn start / end
```
**After:**
```python
# telemetry.wrap_tool: proprio read right after every open_gripper / close_gripper (logged in all modes)
if name in ("open_gripper", "close_gripper") and self.post_gripper_hook is not None:
    self.tool_events.append({"turn_idx": ..., "call": name, **self.post_gripper_hook()})

# eg_vdm.format_exec_header (shown in ours only)
"- Gripper opening right after each gripper command: close_gripper -> 0.02 (EEF z +0.010); open_gripper -> 0.99 (...)"

# eg_vdm.format_exec_context_for_vdm
"The code shows what the robot attempted, not what actually happened: a grasp can miss and an object can slip "
"or be placed beside its target. Judge the outcome from the images together with the proprioceptive readings, "
"never from the code's comments or intent. A gripper opening near 0.0 right after close_gripper means the "
"fingers closed fully with nothing between them; an opening that stays clearly above 0.0 after close_gripper "
"means something is between the fingers."
```
The gripper-scale legend in the slot became "(1.0 = fully open, 0.0 = fully closed with nothing between the
fingers)". These are generic gripper facts, not task rules. The pre-fix runs are kept in
`logs/dev_phase2_v1/`.

### Refinement 3b — withhold the code from the VDM, read images first (AI agent, dev set)

**Observation (after 3a; preserved in `logs/dev_phase2_v2/cube_restack/ours/seed_1`, turn 3):** after the
turn the cubes were no longer visible in the camera image at all. The agent's own stdout had already
reported implausible cube heights (`red center [... 0.094]`). The grounded VDM, despite the new warning,
answered:
```
Physical Change: The red cube has been moved and placed on top of the green cube.
Gripper & Occlusion Check: The gripper is open and empty, having retreated upwards. It partially occludes the stacked cubes below it. ...
Verdict: COMPLETE — The red cube is successfully stacked on top of the green cube as requested.
```
The agent finished, and the episode failed. This is the second dev case in which the VDM reported the
outcome stated in the code comments (`# Compute placement: red cube bottom should sit on green cube top`,
`# Open gripper to release`) instead of what the image showed. The warning text alone did not counter it.

**Before (3a):** VDM user content = task → execution context (**record + full executed code** + stdout) →
instructions → previous image → current image.
```python
f"Executed code:\n```python\n{code}\n```\n"
```
**After (3b):** VDM user content = task → previous image → current image → **execution record without the
code** (exception, motion/tool counts, simulator steps, EEF/gripper readings incl. per-command gripper
readings, stdout) → instructions.
```python
def format_exec_context_for_vdm(rec):
    """Execution record given to the VDM in ``ours``: measured facts and stdout, not the code."""
    return ("Execution record of the robot turn that happened between the two images "
            "(measured by the robot itself; it does not tell you whether the goal was achieved):\n"
            f"{format_exec_header(rec)}\n"
            f"Console stdout printed by the robot's code:\n```\n{stdout}\n```\n"
            "The record shows what the robot did, not what resulted: ... never from what the robot may have intended. ...")
```
The hypothesis is unchanged: the VDM is grounded in the turn's *execution record*, i.e. what physically
happened according to the robot's own measurements. The record no longer includes the program text, which
encodes intent. The coding agent's feedback slot is unchanged (it already sees its code). `ablation`
receives the same image-first ordering, so the structure stays matched. The dev sanity check was then
re-run from scratch (`logs/dev_phase2/`); the 3a runs are kept in `logs/dev_phase2_v2/`.

### Bug fix found during the dev check — last verdict line wins

`logs/dev_phase2/cube_restack/ablation/seed_0`, turn 1: the VDM wrote `Verdict: COMPLETE — ... Wait, looking
closer ...` and then `Verdict: NOT COMPLETE — ...`. `parse_verdict()` took the first match. It now takes the
last match (`_VERDICT_RE.findall(text)[-1]`), and `Step3State` stores the last verdict line. This affects the
logged label in `ours`/`ablation` and the gate's preserved status in `ours`. It is not a change to the method.

### Wording fix found during the dev check — query-only zero-motion turns

`logs/dev_phase2/cube_lifting/ours/seed_0`, turn 0 was a perception-only turn: the code ran without exception
and called no motion primitive. The gate still said "fix the code error". The NOT COMPLETE line now ends with
"Do NOT output FINISH; execute the manipulation." when there was no exception, and keeps "fix the code error
and execute the manipulation." when there was one.
