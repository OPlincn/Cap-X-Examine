"""Offline checks for the Step-3 (EG-VDM) feedback logic."""

from capx.envs import eg_vdm


def _rec(**kw):
    base = dict(
        code="x = 1",
        info_step={"sandbox_rc": 0, "stdout": "done\n", "stderr": ""},
        tool_calls_delta={},
        sim_steps_delta=0,
        proprio_before={"eef_xyz": [0.5, 0.0, 0.3], "gripper_opening": 0.98},
        proprio_after={"eef_xyz": [0.5, 0.0, 0.3], "gripper_opening": 0.98},
    )
    base.update(kw)
    return eg_vdm.build_exec_record(**base)


SYNTAX_STDERR = (
    'Traceback (most recent call last):\n  File "base.py", line 196, in _exec_user_code\n'
    '  File "<string>", line 48\n    Now I\'ll grasp\nSyntaxError: unterminated string literal (detected at line 48)\n'
)


def test_turn0_zero_motion_gate_says_not_complete():
    rec = _rec(info_step={"sandbox_rc": 1, "stdout": "", "stderr": SYNTAX_STDERR})
    assert rec["num_motion_calls"] == 0 and rec["error_line"] == 48
    text, label = eg_vdm.gate_message(rec, eg_vdm.Step3State())
    assert label == "NOT COMPLETE"
    assert "No physical robot motion has been executed in this episode yet" in text
    assert "Do NOT output FINISH" in text and "line 48" in text


def test_gate_preserves_complete_and_warns_not_to_undo():
    st = eg_vdm.Step3State()
    st.update_after_vdm(2, "Physical Change: lifted\nGripper & Occlusion Check: closed\nVerdict: COMPLETE — cube held above table")
    text, label = eg_vdm.gate_message(_rec(), st)
    assert label == "COMPLETE" and "Do NOT undo" in text and "turn 2" in text


def test_header_shows_proprio_delta_and_motion():
    rec = _rec(
        tool_calls_delta={"move_to_joints": 3, "close_gripper": 1, "solve_ik": 3},
        sim_steps_delta=208,
        proprio_after={"eef_xyz": [0.5, 0.0, 0.45], "gripper_opening": 0.55},
    )
    h = eg_vdm.format_exec_header(rec)
    assert "[Execution & Proprioception Record]" in h
    assert "move_to_joints×3" in h and "solve_ik×3" in h
    assert "+0.150" in h and "before 0.98 -> after 0.55" in h


def test_verdict_parsing():
    assert eg_vdm.parse_verdict("Verdict: NOT COMPLETE — x") == "NOT COMPLETE"
    assert eg_vdm.parse_verdict("**Verdict:** COMPLETE — x") == "COMPLETE"
    assert eg_vdm.parse_verdict("no verdict here") is None
    assert eg_vdm.heuristic_verdict("The task has been completed.") == "COMPLETE"
    assert eg_vdm.heuristic_verdict("The task of lifting has not been completed yet.") == "NOT COMPLETE"
    assert eg_vdm.heuristic_verdict("The arm moved.") == "UNCERTAIN"


def test_slot_composition_and_ablation_has_no_exec_record():
    p_ours = eg_vdm.structured_vdm_prompt("task", "img0", "img1", exec_record=_rec(), generic_checklist=False)
    p_abl = eg_vdm.structured_vdm_prompt("task", "img0", "img1", exec_record=None, generic_checklist=True)
    ours_text = " ".join(c.get("text", "") for c in p_ours[1]["content"])
    abl_text = " ".join(c.get("text", "") for c in p_abl[1]["content"])
    assert "Execution record of the robot turn" in ours_text and "Execution record of the robot turn" not in abl_text
    assert "x = 1" not in ours_text  # the code itself is withheld from the VDM
    assert eg_vdm.STRUCTURED_VDM_INSTRUCTIONS in ours_text and eg_vdm.STRUCTURED_VDM_INSTRUCTIONS in abl_text
    assert eg_vdm.compose_slot("baseline", None, "a") == "a"
    assert eg_vdm.compose_slot("ablation", eg_vdm.GENERIC_SLOT_HEADER, "a").startswith("[Generic Verification Checklist]")


def test_header_lists_gripper_readings_after_commands():
    rec = _rec(
        tool_calls_delta={"move_to_joints": 4, "close_gripper": 1, "open_gripper": 1},
        sim_steps_delta=300,
        gripper_events=[
            {"call": "close_gripper", "gripper_opening": 0.015, "eef_xyz": [0.5, 0.0, 0.01]},
            {"call": "open_gripper", "gripper_opening": 0.99, "eef_xyz": [0.4, 0.1, 0.05]},
        ],
    )
    h = eg_vdm.format_exec_header(rec)
    assert "close_gripper -> 0.01 (EEF z +0.010)" in h or "close_gripper -> 0.02 (EEF z +0.010)" in h
    ctx = eg_vdm.format_exec_context_for_vdm(rec)
    assert "not what resulted" in ctx and "never from what the robot may have intended" in ctx


def test_last_verdict_wins_when_vdm_revises_itself():
    text = "Verdict: COMPLETE — first impression.\nRe-evaluating:\nVerdict: NOT COMPLETE — green is on top."
    assert eg_vdm.parse_verdict(text) == "NOT COMPLETE"
    st = eg_vdm.Step3State()
    assert st.update_after_vdm(1, text) == "NOT COMPLETE"
    assert "green is on top" in st.last_valid_motion_verdict


def test_gate_wording_for_query_only_turn():
    text, label = eg_vdm.gate_message(_rec(), eg_vdm.Step3State())
    assert label == "NOT COMPLETE" and "fix the code error" not in text and "execute the manipulation" in text


# --------------------------------------------------------------------------- #
# Post-hoc (Phase 3) variants
# --------------------------------------------------------------------------- #
def _lift_rec():
    return _rec(
        tool_calls_delta={"move_to_joints": 3, "open_gripper": 1, "close_gripper": 1},
        sim_steps_delta=208,
        proprio_before={"eef_xyz": [0.447, -0.002, 0.201], "gripper_opening": 0.21},
        proprio_after={"eef_xyz": [0.548, -0.058, 0.138], "gripper_opening": 0.43},
        gripper_events=[
            {"call": "open_gripper", "gripper_opening": 0.99, "eef_xyz": [0.45, 0.0, 0.214]},
            {"call": "close_gripper", "gripper_opening": 0.44, "eef_xyz": [0.548, -0.058, 0.0]},
        ],
    )


def test_grasp_relative_header_reports_lift_not_turn_delta():
    rec = _lift_rec()
    frozen = eg_vdm.format_exec_header(rec)
    fixed = eg_vdm.format_exec_header(rec, grasp_relative=True)
    assert "(Δ [+0.101, -0.056, -0.063])" in frozen
    assert "Δz_from_grasp = +0.138 m" in fixed and "above the height where the gripper closed" in fixed
    assert "(Δ [" not in fixed


def test_gate_expiry_when_holding_object():
    st = eg_vdm.Step3State()
    st.update_after_vdm(0, "Verdict: NOT COMPLETE — not lifted")
    crash = _rec(
        info_step={"sandbox_rc": 1, "stdout": "", "stderr": 'File "<string>", line 13\nAssertionError\n'},
        proprio_before={"eef_xyz": [0.548, -0.058, 0.138], "gripper_opening": 0.43},
        proprio_after={"eef_xyz": [0.548, -0.058, 0.138], "gripper_opening": 0.43},
    )
    frozen_text, frozen_label = eg_vdm.gate_message(crash, st)
    assert frozen_label == "NOT COMPLETE" and "Do NOT output FINISH" in frozen_text
    text, label = eg_vdm.gate_message_with_expiry(crash, st)
    assert label == "UNCERTAIN" and "EXPIRED" in text and "already held" in text
    assert "Do NOT output FINISH" not in text


def test_gate_expiry_not_holding_expires_after_one_repeat():
    st = eg_vdm.Step3State()
    st.update_after_vdm(0, "Verdict: NOT COMPLETE — cubes apart")
    crash = _rec(info_step={"sandbox_rc": 1, "stdout": "", "stderr": "AssertionError\n"})  # gripper open 0.98
    _, first = eg_vdm.gate_message_with_expiry(crash, st)
    _, second = eg_vdm.gate_message_with_expiry(crash, st)
    assert first == "NOT COMPLETE" and second == "UNCERTAIN"


def test_gate_expiry_keeps_turn0_initial_verdict():
    crash = _rec(info_step={"sandbox_rc": 1, "stdout": "", "stderr": SYNTAX_STDERR},
                 proprio_after={"eef_xyz": [0.5, 0.0, 0.3], "gripper_opening": 0.5})
    text, label = eg_vdm.gate_message_with_expiry(crash, eg_vdm.Step3State())
    assert label == "NOT COMPLETE" and "No physical robot motion has been executed" in text
