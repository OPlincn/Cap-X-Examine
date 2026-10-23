"""Offline checks for per-episode budget enforcement (no simulator or network)."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from capx.envs.telemetry import BudgetExceeded, EpisodeBudget, EpisodeTelemetry


@dataclass
class _Args:
    model: str = "fake"
    max_tokens: int = 8192


def _fake_query(usage):
    def fn(args, prompt):
        return {"content": "ok", "reasoning": None, "usage": usage, "max_tokens_seen": args.max_tokens}

    return fn


def _tele(tmp_path, **budget):
    return EpisodeTelemetry(
        task_name="t", seed=0, mode="baseline", budget=EpisodeBudget(**budget),
        coding_model="fake", vdm_model="fake-vlm", output_path=tmp_path / "ep.json",
    )


PROMPT = [{"role": "user", "content": [{"type": "text", "text": "x" * 300}]}]


def test_model_call_cap(tmp_path):
    tele = _tele(tmp_path, max_model_calls=2)
    q = _fake_query({"prompt_tokens": 10, "completion_tokens": 5})
    tele.query(q, _Args(), PROMPT, role="coding", purpose="p", turn_idx=0)
    tele.query(q, _Args(), PROMPT, role="vdm", purpose="p", turn_idx=0)
    with pytest.raises(BudgetExceeded) as exc:
        tele.query(q, _Args(), PROMPT, role="coding", purpose="p", turn_idx=1)
    assert exc.value.reason == "max_model_calls"
    assert tele.budget_exceeded == "max_model_calls"


def test_token_cap_reserves_and_charges(tmp_path):
    tele = _tele(tmp_path, max_total_tokens=1000)
    out = tele.query(_fake_query({"prompt_tokens": 100, "completion_tokens": 50}), _Args(), PROMPT,
                     role="coding", purpose="p", turn_idx=0)
    # completion cap = remaining - estimated input
    assert out["max_tokens_seen"] <= 1000 - 100
    assert tele.total_tokens == 150
    with pytest.raises(BudgetExceeded) as exc:
        tele.query(_fake_query({"prompt_tokens": 900, "completion_tokens": 50}), _Args(), PROMPT,
                   role="coding", purpose="p", turn_idx=1)
    assert exc.value.reason == "max_total_tokens"


def test_missing_usage_charged_at_reservation(tmp_path):
    tele = _tele(tmp_path, max_total_tokens=5000, max_completion_tokens=500)
    tele.query(_fake_query(None), _Args(), PROMPT, role="vdm", purpose="p", turn_idx=0)
    assert tele.total_tokens == tele.model_calls[0].estimated_input_tokens + 500


def test_tool_wrapper_counts_and_sim_guard(tmp_path):
    tele = _tele(tmp_path, max_sim_steps=10)
    steps = {"n": 0}
    tele.sim_steps_fn = lambda: steps["n"]

    def move(_q):
        steps["n"] += 6

    wrapped = tele.wrap_tool("move_to_joints", move)
    wrapped(None)
    wrapped(None)
    with pytest.raises(BudgetExceeded):
        wrapped(None)
    assert tele.tool_calls["move_to_joints"] == 2
    assert tele.wrap_tool("not_a_tool", move) is move


def test_turn_cap_and_json(tmp_path):
    tele = _tele(tmp_path, max_turns=1)
    tele.check_can_execute()
    tele.add_turn({"turn_idx": 0, "code_exec_success": True})
    with pytest.raises(BudgetExceeded):
        tele.check_can_execute()
    tele.set_termination("max_turns")
    tele.finalize({"official_task_success": False, "official_reward": 0.0})
    d = json.loads((tmp_path / "ep.json").read_text())
    assert d["timeout_or_budget_exceeded"] is True
    assert d["evaluation"]["episode_code_exec_success_rate"] == 1.0
    assert d["costs"]["num_code_executions"] == 1
