"""Step 3 of the M4 multi-turn loop: what goes into the VDM feedback slot.

Three harness modes share everything except this step:

- ``baseline``: official M4 VDM (two images + task prompt, free-text answer).
- ``ours`` (Execution-Grounded VDM): the VDM additionally receives the turn's own
  execution record (code, exception, tool-call counts, simulator steps, stdout,
  end-effector / gripper proprioception before -> after) and answers in a
  structured format with an explicit verdict. If no motion primitive ran, the VDM
  is not queried; a deterministic zero-motion report is returned instead. The
  feedback slot shows ``[Execution & Proprioception Record]`` followed by
  ``[Visual Differencing Assessment]``.
- ``ablation``: same structured VDM instructions and the same two-section slot,
  but the execution record is replaced by a generic verification checklist, and
  the VDM is always queried (no zero-motion gate).

Everything in the execution record comes from the agent's own code/output or from
robot proprioception that the agent can read through ``get_observation()``. No
reward, success flag or object state is used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

MOTION_TOOLS = ("move_to_joints", "open_gripper", "close_gripper")
PERCEPTION_TOOLS = (
    "get_observation",
    "segment_sam3_text_prompt",
    "segment_sam3_point_prompt",
    "plan_grasp",
    "get_oriented_bounding_box_from_3d_points",
    "solve_ik",
)

VERDICTS = ("NOT COMPLETE", "COMPLETE", "UNCERTAIN")
_VERDICT_RE = re.compile(r"Verdict\s*[:：]\s*\**\s*(NOT COMPLETE|COMPLETE|UNCERTAIN)", re.IGNORECASE)

INITIAL_MOTION_VERDICT = "NOT COMPLETE (No physical robot motion has been executed in this episode yet)"

_MAX_GRIPPER_EVENTS = 6

# Post-hoc diagnostic variants of ``ours`` (Phase 3). ``None`` is the frozen behaviour.
POSTHOC_VARIANTS = ("ref_only", "gate_only", "ref_and_gate")
HOLDING_OPENING_RANGE = (0.05, 0.85)


def uses_grasp_reference(variant: str | None) -> bool:
    return variant in ("ref_only", "ref_and_gate")


def uses_gate_expiry(variant: str | None) -> bool:
    return variant in ("gate_only", "ref_and_gate")
_STDOUT_CHARS = 1500

# ---------------------------------------------------------------------------
# Structured VDM instructions (identical for ``ours`` and ``ablation``)
# ---------------------------------------------------------------------------

STRUCTURED_VDM_SYSTEM = (
    "You are a visual verification module for a robot manipulation agent. You compare the "
    "previous and current camera images of the workspace, with the goal of the task in mind, "
    "and report objectively what physically changed and whether the task is complete. "
    "Do *NOT* write any code."
)

STRUCTURED_VDM_INSTRUCTIONS = (
    "Compare the previous state image with the current state image and answer in exactly this format:\n"
    "Physical Change: <what moved or changed between the two images; say 'none' if nothing changed>\n"
    "Gripper & Occlusion Check: <is the gripper open or closed, is it holding an object, is any object "
    "occluded by the arm, and what is the vertical/stacking relation between the task objects>\n"
    "Verdict: <COMPLETE | NOT COMPLETE | UNCERTAIN> — <one-sentence evidence for the verdict>\n"
    "Use UNCERTAIN when the images do not allow a confident judgement (for example height or "
    "stacking order that is ambiguous from a single 2D view)."
)

GENERIC_VDM_CHECKLIST = (
    "Generic verification checklist (apply before giving a verdict):\n"
    "1. Look at both images carefully and list every object that is relevant to the goal.\n"
    "2. A single 2D camera view can be misleading about height, depth and contact; objects that "
    "appear to overlap may be far apart, and an object near the gripper may or may not be held.\n"
    "3. Check the stacking order and vertical relation of the objects explicitly instead of "
    "assuming it from their screen positions.\n"
    "4. Check whether the gripper fingers are open or closed and whether an object is between them.\n"
    "5. Check whether the robot arm occludes any object that matters for the goal.\n"
    "6. Do not assume an action happened just because it would be the next logical step; describe "
    "only what is visible.\n"
    "7. If the evidence is ambiguous, prefer UNCERTAIN over guessing COMPLETE or NOT COMPLETE.\n"
    "8. Base the verdict only on what the two images show, and state the concrete visual evidence."
)

GENERIC_SLOT_HEADER = (
    "[Generic Verification Checklist]\n"
    "- The assessment below was produced by a visual verification module that compared the "
    "previous and current camera images.\n"
    "- A single 2D view can be ambiguous about height, contact and stacking order.\n"
    "- Check the gripper state and object relations before deciding on the next action.\n"
    "- Prefer verifying with perception over assuming the outcome of an action."
)


# ---------------------------------------------------------------------------
# Execution record
# ---------------------------------------------------------------------------

def last_exception_line(stderr: str) -> str | None:
    """Final ``SomeError: message`` line of a traceback (memory addresses normalized)."""
    lines = [ln.strip() for ln in (stderr or "").strip().splitlines() if ln.strip()]
    for ln in reversed(lines):
        if re.match(r"^[A-Za-z_][\w.]*(Error|Exception|Exceeded|Interrupt|Exit)\b", ln):
            return re.sub(r"0x[0-9a-f]+", "0x?", ln)
    return lines[-1] if lines else None


def user_code_error_line(stderr: str) -> int | None:
    """Line number of the failure inside the executed code block, if reported."""
    hits = re.findall(r'File "<string>", line (\d+)', stderr or "")
    return int(hits[-1]) if hits else None


def read_proprio(env: Any) -> dict[str, Any] | None:
    """Non-privileged robot state: end-effector xyz and gripper opening (robot base frame)."""
    try:
        cart = np.asarray(env.low_level_env.get_observation()["robot_cartesian_pos"], dtype=float)
    except Exception:
        return None
    return {
        "eef_xyz": [round(float(x), 4) for x in cart[:3]],
        "gripper_opening": round(float(cart[7]), 3) if cart.shape[0] > 7 else None,
    }


def build_exec_record(
    code: str,
    info_step: dict[str, Any],
    tool_calls_delta: dict[str, int],
    sim_steps_delta: int,
    proprio_before: dict[str, Any] | None,
    proprio_after: dict[str, Any] | None,
    gripper_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    exec_ok = info_step.get("sandbox_rc", 1) == 0
    stderr = info_step.get("stderr", "")
    return {
        "code": code,
        "exec_ok": exec_ok,
        "exception": None if exec_ok else last_exception_line(stderr),
        "error_line": None if exec_ok else user_code_error_line(stderr),
        "tool_calls": dict(tool_calls_delta),
        "motion_calls": {k: tool_calls_delta.get(k, 0) for k in MOTION_TOOLS},
        "num_motion_calls": sum(tool_calls_delta.get(k, 0) for k in MOTION_TOOLS),
        "sim_steps_delta": sim_steps_delta,
        "stdout_tail": (info_step.get("stdout", "") or "")[-_STDOUT_CHARS:],
        "proprio_before": proprio_before,
        "proprio_after": proprio_after,
        "gripper_events": list(gripper_events or []),
    }


def _fmt_vec(v: list[float] | None) -> str:
    return "unknown" if v is None else "[" + ", ".join(f"{x:+.3f}" for x in v) + "]"


def format_exec_header(rec: dict[str, Any], grasp_relative: bool = False) -> str:
    """Concise factual record shown in the feedback slot (and to the VDM in ``ours``).

    ``grasp_relative`` (post-hoc ``ref_*`` variants) replaces the turn-start -> turn-end Δ with the
    vertical displacement relative to the height at the last ``close_gripper`` of the turn.
    """
    if rec["exec_ok"]:
        status = "completed without exception"
    else:
        where = f" at line {rec['error_line']}" if rec["error_line"] is not None else ""
        status = f"FAILED{where}: {rec['exception']}"
    motion = ", ".join(f"{k}×{v}" for k, v in rec["motion_calls"].items() if v) or "none"
    other = ", ".join(
        f"{k}×{v}" for k, v in rec["tool_calls"].items() if v and k not in MOTION_TOOLS
    ) or "none"
    lines = [
        "[Execution & Proprioception Record]",
        f"- Code execution: {status}",
        f"- Robot motion calls this turn: {motion}",
        f"- Perception/planning calls this turn: {other}",
        f"- Simulator steps this turn: {rec['sim_steps_delta']}",
    ]
    pb, pa = rec["proprio_before"], rec["proprio_after"]
    if pb and pa and grasp_relative:
        lines.append(
            f"- End-effector xyz (robot base frame, m): at turn start {_fmt_vec(pb['eef_xyz'])} -> "
            f"at turn end {_fmt_vec(pa['eef_xyz'])} (the turn starts wherever the previous turn "
            f"ended, e.g. the high home pose, so this difference is not a lift height)"
        )
        closes = [e for e in rec.get("gripper_events", [])
                  if e.get("call") == "close_gripper" and e.get("eef_xyz") is not None]
        if closes:
            z_grasp, z_now = closes[-1]["eef_xyz"][2], pa["eef_xyz"][2]
            dz = z_now - z_grasp
            where = "above" if dz >= 0 else "below"
            lines.append(
                f"- Vertical displacement since the last close_gripper: grasp at EEF z = {z_grasp:+.3f} m "
                f"-> current EEF z = {z_now:+.3f} m (Δz_from_grasp = {dz:+.3f} m, i.e. {abs(dz):.3f} m "
                f"{where} the height where the gripper closed)"
            )
    elif pb and pa:
        delta = [round(a - b, 4) for a, b in zip(pa["eef_xyz"], pb["eef_xyz"])]
        lines.append(
            f"- End-effector xyz (robot base frame, m): before {_fmt_vec(pb['eef_xyz'])} -> "
            f"after {_fmt_vec(pa['eef_xyz'])} (Δ {_fmt_vec(delta)})"
        )
        if pb.get("gripper_opening") is not None and pa.get("gripper_opening") is not None:
            lines.append(
                f"- Gripper opening (1.0 = fully open, 0.0 = fully closed with nothing between the "
                f"fingers): before {pb['gripper_opening']:.2f} -> after {pa['gripper_opening']:.2f}"
            )
    events = [e for e in rec.get("gripper_events", []) if e.get("gripper_opening") is not None]
    if events:
        shown = events[-_MAX_GRIPPER_EVENTS:]
        prefix = "... " if len(events) > len(shown) else ""
        lines.append(
            "- Gripper opening right after each gripper command: " + prefix + "; ".join(
                f"{e['call']} -> {e['gripper_opening']:.2f} (EEF z {e['eef_xyz'][2]:+.3f})" for e in shown
            )
        )
    return "\n".join(lines)


def format_exec_context_for_vdm(rec: dict[str, Any], grasp_relative: bool = False) -> str:
    """Execution record given to the VDM in ``ours``: measured facts and stdout, not the code.

    The code itself is withheld from the VDM because its comments state the intended outcome
    (e.g. "place on top of the green cube"), which biased the VDM towards seeing that outcome.
    """
    stdout = rec["stdout_tail"].strip() or "(empty)"
    return (
        "Execution record of the robot turn that happened between the two images "
        "(measured by the robot itself; it does not tell you whether the goal was achieved):\n"
        f"{format_exec_header(rec, grasp_relative=grasp_relative)}\n"
        f"Console stdout printed by the robot's code:\n```\n{stdout}\n```\n"
        "The record shows what the robot did, not what resulted: a grasp can miss and "
        "an object can slip or be placed beside its target. Judge the outcome from the images together "
        "with the proprioceptive readings, never from what the robot may have intended. A gripper opening "
        "near 0.0 right after close_gripper means the fingers closed fully with nothing between them; "
        "an opening that stays clearly above 0.0 after close_gripper means something is between the "
        "fingers."
    )


# ---------------------------------------------------------------------------
# VDM prompt / verdicts / gate
# ---------------------------------------------------------------------------

def structured_vdm_prompt(
    task_description: str,
    prev_image: str,
    cur_image: str,
    *,
    exec_record: dict[str, Any] | None,
    generic_checklist: bool,
    grasp_relative: bool = False,
) -> list[dict[str, Any]]:
    # Images come before the execution record / checklist so the images are read first.
    content: list[dict[str, Any]] = [
        {"type": "text", "text": task_description},
        {"type": "text", "text": "Previous state (main camera):"},
        {"type": "image_url", "image_url": {"url": prev_image}},
        {"type": "text", "text": "Current state (main camera):"},
        {"type": "image_url", "image_url": {"url": cur_image}},
    ]
    if exec_record is not None:
        content.append({"type": "text", "text": format_exec_context_for_vdm(exec_record, grasp_relative)})
    elif generic_checklist:
        content.append({"type": "text", "text": GENERIC_VDM_CHECKLIST})
    content.append({"type": "text", "text": STRUCTURED_VDM_INSTRUCTIONS})
    return [
        {"role": "system", "content": STRUCTURED_VDM_SYSTEM},
        {"role": "user", "content": content},
    ]


def parse_verdict(text: str | None) -> str | None:
    if not text:
        return None
    # The last verdict wins: the VDM sometimes revises an initial verdict within the same answer.
    matches = _VERDICT_RE.findall(text)
    return matches[-1].upper() if matches else None


_CLAIM_DONE = re.compile(
    r"(task (has been|is|appears to (be|have been)) (successfully )?(completed|accomplished|achieved)"
    r"|successfully (lifted|stacked|placed|wiped|picked|completed)"
    r"|has been (lifted|stacked|placed|wiped)( successfully)?"
    r"|(is|now) (stacked|resting|placed) on top of"
    r"|task (has been )?completed)",
    re.IGNORECASE,
)
_NEGATION = re.compile(
    r"\b(not|no longer|has not|hasn't|isn't|incomplete|yet to|opposite)\b", re.IGNORECASE
)


def heuristic_verdict(text: str | None) -> str:
    """Verdict label for free-text (baseline) VDM output, used only for evaluation metrics."""
    if not text:
        return "UNCERTAIN"
    done = neg = False
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if _CLAIM_DONE.search(sent):
            if _NEGATION.search(sent):
                neg = True
            else:
                done = True
        elif re.search(r"(not (yet )?(been )?(completed|complete|achieved)|has not been)", sent, re.I):
            neg = True
    if done and not neg:
        return "COMPLETE"
    if neg:
        return "NOT COMPLETE"
    return "UNCERTAIN"


@dataclass
class Step3State:
    """Per-episode memory of the last verdict obtained after a turn with robot motion."""

    last_valid_motion_verdict: str = INITIAL_MOTION_VERDICT
    last_valid_motion_label: str = "NOT COMPLETE"
    last_motion_turn: int | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    # Post-hoc gate-expiry bookkeeping (unused by the frozen behaviour).
    consecutive_gated: int = 0
    expired: bool = False

    def update_after_vdm(self, turn_idx: int, vdm_text: str) -> str:
        label = parse_verdict(vdm_text) or "UNCERTAIN"
        line = next(
            (ln for ln in reversed(vdm_text.splitlines()) if ln.strip().lower().startswith("verdict")), ""
        )
        self.last_valid_motion_label = label
        self.last_valid_motion_verdict = f"{line.strip() or 'Verdict: ' + label} (VDM after turn {turn_idx})"
        self.last_motion_turn = turn_idx
        self.consecutive_gated = 0
        self.expired = False
        return label


def gate_message(rec: dict[str, Any], state: Step3State) -> tuple[str, str]:
    """Deterministic assessment for a turn in which no motion primitive ran.

    Returns (assessment_text, verdict_label).
    """
    if rec["exec_ok"]:
        cause = "the code ran without exception but called no motion primitive"
    else:
        where = f" at line {rec['error_line']}" if rec["error_line"] is not None else ""
        cause = f"execution failed{where}: {rec['exception']}"
    head = (
        "Zero-Motion Gate: no robot motion primitive (move_to_joints / open_gripper / close_gripper) "
        f"was executed this turn ({cause}). The scene is physically unchanged since the previous "
        "observation, so no new visual assessment was requested.\n"
        f"Last assessment after robot motion: {state.last_valid_motion_verdict}\n"
    )
    label = state.last_valid_motion_label
    if label == "COMPLETE":
        verdict = (
            f"Verdict: COMPLETE (preserved from turn {state.last_motion_turn}, the last turn with robot "
            "motion) — Zero robot motion occurred this turn. Do NOT undo the completed state (do not "
            "open the gripper, re-grasp or move the objects); output FINISH if the task is complete."
        )
    elif label == "UNCERTAIN":
        verdict = (
            f"Verdict: UNCERTAIN (preserved from turn {state.last_motion_turn}) — Zero robot motion "
            f"occurred this turn ({cause}). Verify the state with perception before deciding; do not "
            "undo progress that may already be complete."
        )
        label = "UNCERTAIN"
    else:
        action = "fix the code error and execute the manipulation" if not rec["exec_ok"] else "execute the manipulation"
        verdict = (
            f"Verdict: NOT COMPLETE — Zero robot motion occurred this turn ({cause}). Do NOT output "
            f"FINISH; {action}."
        )
        label = "NOT COMPLETE"
    return head + verdict, label


def gate_message_with_expiry(rec: dict[str, Any], state: Step3State) -> tuple[str, str]:
    """Post-hoc ``gate_only`` / ``ref_and_gate`` gate: a cached NOT COMPLETE claim expires.

    The cached claim is not repeated when the gripper is closed around an object after the
    zero-motion turn, or after one gated repeat. A state-aware diagnostic is returned instead.
    """
    state.consecutive_gated += 1
    pa = rec.get("proprio_after") or {}
    g = pa.get("gripper_opening")
    holding = g is not None and HOLDING_OPENING_RANGE[0] < g < HOLDING_OPENING_RANGE[1]
    stale_claim = state.last_motion_turn is not None and state.last_valid_motion_label == "NOT COMPLETE"
    if not stale_claim or not (holding or state.consecutive_gated >= 2 or state.expired):
        return gate_message(rec, state)

    state.expired = True
    if rec["exec_ok"]:
        cause = "the code ran without exception but called no motion primitive"
    else:
        where = f" at line {rec['error_line']}" if rec["error_line"] is not None else ""
        cause = f"execution failed{where}: {rec['exception']}"
    z = (pa.get("eef_xyz") or [None, None, None])[2]
    head = (
        "Zero-Motion Gate: no robot motion primitive (move_to_joints / open_gripper / close_gripper) "
        f"was executed this turn ({cause}). The scene is physically unchanged since the previous "
        "observation, so no new visual assessment was requested.\n"
        f"The earlier NOT COMPLETE verdict (VDM after turn {state.last_motion_turn}) has EXPIRED and is "
        "not repeated, because it may be stale.\n"
    )
    if holding:
        state_line = (
            f"Current robot state: the gripper is closed at opening {g:.2f} (an object is between the "
            f"fingers) with the end-effector at z = {z:+.3f} m. Zero motion occurred because {cause}; "
            "a grasp-planning failure can mean the object is already held in the gripper.\n"
        )
        verdict = (
            "Verdict: UNCERTAIN — Verify whether the goal is already met (e.g. inspect the current "
            "observation and proprioception) before re-planning a grasp; do not open the gripper or "
            "re-grasp an object that is already held."
        )
    else:
        state_line = ""
        verdict = (
            "Verdict: UNCERTAIN — Repeated zero-motion turn. Verify the current state with perception and "
            "fix the code before executing the manipulation."
        )
    return head + state_line + verdict, "UNCERTAIN"


def compose_slot(mode: str, header: str | None, assessment: str) -> str:
    """Text placed into the existing VDM feedback slot of the decision prompt."""
    if mode == "baseline":
        return assessment
    return f"{header}\n[Visual Differencing Assessment]\n{assessment}"
