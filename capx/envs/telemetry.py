"""Per-episode budget enforcement and telemetry for paired harness evaluation.

One :class:`EpisodeTelemetry` is created per episode by ``capx.envs.trial``. It

- enforces hard per-episode caps on code executions, model calls (coding + VDM),
  total model tokens, simulator steps and wall time;
- counts low-level tool calls made inside the sandbox;
- records every model call, every turn and the termination reason;
- stores post-episode evaluation results separately from anything the agent saw.

Official reward / success flags are written only by :meth:`EpisodeTelemetry.finalize`,
after the agent loop has ended; nothing in this module feeds them back into prompts.
"""

from __future__ import annotations

import contextvars
import functools
import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

# Tools whose calls are counted. Perception tools are also tallied separately as
# "extra perception" service calls.
COUNTED_TOOLS = (
    "get_observation",
    "segment_sam3_text_prompt",
    "segment_sam3_point_prompt",
    "point_prompt_molmo",
    "plan_grasp",
    "solve_ik",
    "move_to_joints",
    "open_gripper",
    "close_gripper",
    "get_oriented_bounding_box_from_3d_points",
)
PERCEPTION_TOOLS = (
    "segment_sam3_text_prompt",
    "segment_sam3_point_prompt",
    "point_prompt_molmo",
    "plan_grasp",
)
MOTION_TOOLS = ("move_to_joints", "open_gripper", "close_gripper")
# Termination reasons that count as "timeout or budget exceeded".
BUDGET_TERMINATIONS = (
    "max_turns",
    "max_model_calls",
    "max_total_tokens",
    "max_sim_steps",
    "max_wall_time",
    "wall_timeout",
    "env_horizon",
)

# Rough character-per-token ratio used only for the pre-call budget reservation;
# the actual usage reported by the server is what gets charged.
_CHARS_PER_TOKEN = 3.0
_IMAGE_TOKEN_ESTIMATE = 400
_MIN_COMPLETION_TOKENS = 256


class BudgetExceeded(BaseException):
    """Raised when an episode budget is exhausted.

    Derives from ``BaseException`` so ``except Exception`` in generated code
    cannot swallow it.
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


@dataclass
class EpisodeBudget:
    """Hard per-episode caps, identical for every experimental condition."""

    max_turns: int = 10
    """Maximum number of code executions."""
    max_model_calls: int = 24
    """Maximum model calls, coding agent and VDM combined."""
    max_total_tokens: int = 200_000
    """Maximum input + output tokens, coding agent and VDM combined."""
    max_sim_steps: int = 1500
    """Maximum simulator control steps."""
    max_wall_time_s: int = 1200
    """Wall-clock limit for the agent loop (after scene reset)."""
    max_completion_tokens: int = 8192
    """Upper bound on completion tokens per model call."""


@dataclass
class ModelCallRecord:
    role: str  # "coding" or "vdm"
    purpose: str  # "initial_code", "multi_turn_decision", "vdm_initial", "vdm_diff"
    turn_idx: int | None
    model: str
    model_returned: str | None
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    max_tokens_requested: int
    estimated_input_tokens: int
    latency_s: float | None
    http_attempts: int | None
    finish_reason: str | None
    ok: bool
    error: str | None = None
    replayed: bool = False


_ACTIVE: contextvars.ContextVar[EpisodeTelemetry | None] = contextvars.ContextVar(
    "capx_episode_telemetry", default=None
)


def get_active_telemetry() -> EpisodeTelemetry | None:
    return _ACTIVE.get()


def _estimate_prompt_tokens(prompt: list[dict[str, Any]]) -> int:
    chars = 0
    images = 0
    for msg in prompt:
        content = msg.get("content")
        if isinstance(content, str):
            chars += len(content)
            continue
        for part in content or []:
            if part.get("type") == "text":
                chars += len(part.get("text", ""))
            else:
                images += 1
    return int(math.ceil(chars / _CHARS_PER_TOKEN)) + images * _IMAGE_TOKEN_ESTIMATE


class EpisodeTelemetry:
    def __init__(
        self,
        *,
        task_name: str,
        seed: int,
        mode: str,
        budget: EpisodeBudget,
        coding_model: str,
        vdm_model: str | None,
        output_path: str | Path,
    ) -> None:
        self.task_name = task_name
        self.seed = seed
        self.mode = mode
        self.budget = budget
        self.coding_model = coding_model
        self.vdm_model = vdm_model
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.start_time = time.time()
        self.end_time: float | None = None
        self.initial_state: dict[str, Any] | None = None
        self.model_calls: list[ModelCallRecord] = []
        self.turns: list[dict[str, Any]] = []
        self.tool_calls: Counter[str] = Counter()
        self.sim_steps_at_start = 0
        self.sim_steps_fn: Callable[[], int] | None = None
        self.termination_reason: str | None = None
        self.termination_detail: str = ""
        self.budget_exceeded: str | None = None
        self.evaluation: dict[str, Any] = {}
        self.reset_wall_time: float | None = None
        # Filled by the VDM query helpers in capx.envs.trial; read back into the turn record.
        self.last_vdm: dict[str, Any] | None = None
        # Optional ``() -> dict`` reading robot proprioception, called right after gripper
        # commands; readings are appended to ``tool_events`` (logged, used by EG-VDM).
        self.post_gripper_hook: Callable[[], dict[str, Any] | None] | None = None
        # Post-hoc turn-0 matching: {(purpose, turn_idx): {"content", "input_tokens", ...}}
        self.replay: dict[tuple[str, int], dict[str, Any]] = {}
        self.posthoc: dict[str, Any] | None = None
        self.tool_events: list[dict[str, Any]] = []
        self._token = None

    # ------------------------------------------------------------------ #
    # Context management
    # ------------------------------------------------------------------ #
    def activate(self) -> None:
        self._token = _ACTIVE.set(self)

    def deactivate(self) -> None:
        if self._token is not None:
            _ACTIVE.reset(self._token)
            self._token = None

    # ------------------------------------------------------------------ #
    # Aggregates
    # ------------------------------------------------------------------ #
    def _sum(self, role: str | None, attr: str) -> int:
        return sum(
            (getattr(c, attr) or 0) for c in self.model_calls if role is None or c.role == role
        )

    @property
    def total_tokens(self) -> int:
        return self._sum(None, "input_tokens") + self._sum(None, "output_tokens")

    @property
    def num_model_calls(self) -> int:
        return len(self.model_calls)

    @property
    def num_code_executions(self) -> int:
        return len(self.turns)

    def sim_steps(self) -> int:
        if self.sim_steps_fn is None:
            return 0
        return int(self.sim_steps_fn()) - self.sim_steps_at_start

    # ------------------------------------------------------------------ #
    # Budget checks
    # ------------------------------------------------------------------ #
    def _exceed(self, reason: str, detail: str) -> BudgetExceeded:
        if self.budget_exceeded is None:
            self.budget_exceeded = reason
            self.termination_detail = detail
        return BudgetExceeded(reason, detail)

    def check_can_execute(self) -> None:
        if self.num_code_executions >= self.budget.max_turns:
            raise self._exceed("max_turns", f"{self.num_code_executions} code executions")
        self.check_sim_steps()
        self.check_wall_time()

    def check_sim_steps(self) -> None:
        steps = self.sim_steps()
        if steps >= self.budget.max_sim_steps:
            raise self._exceed("max_sim_steps", f"{steps} simulator steps")

    def check_wall_time(self) -> None:
        elapsed = time.time() - self.start_time
        if elapsed >= self.budget.max_wall_time_s:
            raise self._exceed("max_wall_time", f"{elapsed:.0f} s")

    # ------------------------------------------------------------------ #
    # Model calls
    # ------------------------------------------------------------------ #
    def query(
        self,
        query_fn: Callable[[Any, list[dict]], dict[str, Any]],
        args: Any,
        prompt: list[dict[str, Any]],
        *,
        role: str,
        purpose: str,
        turn_idx: int | None,
    ) -> dict[str, Any]:
        """Budget-checked model call; charges actual usage to the episode."""
        if self.num_model_calls >= self.budget.max_model_calls:
            raise self._exceed("max_model_calls", f"{self.num_model_calls} calls")
        self.check_wall_time()

        est_in = _estimate_prompt_tokens(prompt)
        remaining = self.budget.max_total_tokens - self.total_tokens
        max_out = min(self.budget.max_completion_tokens, args.max_tokens, remaining - est_in)
        if max_out < _MIN_COMPLETION_TOKENS:
            raise self._exceed(
                "max_total_tokens",
                f"used {self.total_tokens}, next call needs ~{est_in}+{_MIN_COMPLETION_TOKENS}",
            )

        call_args = _with_max_tokens(args, max_out)
        try:
            out = query_fn(call_args, prompt)
        except BaseException as exc:
            self.model_calls.append(
                ModelCallRecord(
                    role=role, purpose=purpose, turn_idx=turn_idx, model=args.model,
                    model_returned=None, input_tokens=None, output_tokens=None,
                    reasoning_tokens=None, max_tokens_requested=max_out,
                    estimated_input_tokens=est_in, latency_s=None, http_attempts=None,
                    finish_reason=None, ok=False, error=repr(exc),
                )
            )
            raise

        usage = out.get("usage") or {}
        in_tok = usage.get("prompt_tokens")
        out_tok = usage.get("completion_tokens")
        if in_tok is None or out_tok is None:
            # Unknown usage is charged at the reserved upper bound, never zero.
            in_tok, out_tok = est_in, max_out
        details = usage.get("completion_tokens_details") or {}
        self.model_calls.append(
            ModelCallRecord(
                role=role, purpose=purpose, turn_idx=turn_idx, model=args.model,
                model_returned=out.get("model_returned"), input_tokens=in_tok,
                output_tokens=out_tok, reasoning_tokens=details.get("reasoning_tokens"),
                max_tokens_requested=max_out, estimated_input_tokens=est_in,
                latency_s=out.get("latency_s"), http_attempts=out.get("http_attempts"),
                finish_reason=out.get("finish_reason"), ok=True,
            )
        )
        if self.total_tokens > self.budget.max_total_tokens:
            # Output of an over-budget call is discarded, never executed.
            raise self._exceed("max_total_tokens", f"used {self.total_tokens}")
        out["_usage_charged"] = {"input_tokens": in_tok, "output_tokens": out_tok}
        return out

    def record_replayed(
        self,
        args: Any,
        prompt: list[dict[str, Any]],
        replay: dict[str, Any],
        *,
        role: str,
        purpose: str,
        turn_idx: int | None,
    ) -> dict[str, Any]:
        """Return a logged response from an earlier run instead of querying the model.

        Used only by post-hoc turn-0 matching. The logged token usage is charged against the
        budget exactly as the original call was, so budgets stay comparable.
        """
        if self.num_model_calls >= self.budget.max_model_calls:
            raise self._exceed("max_model_calls", f"{self.num_model_calls} calls")
        est_in = _estimate_prompt_tokens(prompt)
        in_tok, out_tok = replay.get("input_tokens"), replay.get("output_tokens")
        self.model_calls.append(
            ModelCallRecord(
                role=role, purpose=purpose, turn_idx=turn_idx, model=args.model,
                model_returned=replay.get("model_returned"), input_tokens=in_tok,
                output_tokens=out_tok, reasoning_tokens=None, max_tokens_requested=0,
                estimated_input_tokens=est_in, latency_s=0.0, http_attempts=0,
                finish_reason="replayed", ok=True, replayed=True,
            )
        )
        if self.total_tokens > self.budget.max_total_tokens:
            raise self._exceed("max_total_tokens", f"used {self.total_tokens}")
        return {
            "content": replay["content"],
            "reasoning": None,
            "usage": {"prompt_tokens": in_tok, "completion_tokens": out_tok},
            "model_returned": replay.get("model_returned"),
            "_usage_charged": {"input_tokens": in_tok, "output_tokens": out_tok},
        }

    # ------------------------------------------------------------------ #
    # Tool calls
    # ------------------------------------------------------------------ #
    def wrap_tool(self, name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        if name not in COUNTED_TOOLS:
            return fn

        @functools.wraps(fn)
        def wrapper(*a: Any, **kw: Any) -> Any:
            if name in MOTION_TOOLS:
                self.check_sim_steps()
            self.tool_calls[name] += 1
            result = fn(*a, **kw)
            if name in ("open_gripper", "close_gripper") and self.post_gripper_hook is not None:
                reading = self.post_gripper_hook()
                if reading is not None:
                    self.tool_events.append(
                        {"turn_idx": self.num_code_executions, "call": name, **reading}
                    )
            return result

        return wrapper

    # ------------------------------------------------------------------ #
    # Turns
    # ------------------------------------------------------------------ #
    def add_turn(self, record: dict[str, Any]) -> None:
        self.turns.append(record)
        self.save()

    def last_turn(self) -> dict[str, Any] | None:
        return self.turns[-1] if self.turns else None

    def set_termination(self, reason: str, detail: str = "") -> None:
        if self.termination_reason is None:
            self.termination_reason = reason
            if detail:
                self.termination_detail = detail

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def finalize(self, evaluation: dict[str, Any]) -> None:
        """Record post-episode evaluation (official reward/success) and save."""
        self.end_time = time.time()
        self.evaluation = evaluation
        self.save()

    def to_dict(self) -> dict[str, Any]:
        calls = [asdict(c) for c in self.model_calls]
        execs = [t for t in self.turns if t.get("executed", True)]
        ok_execs = sum(1 for t in execs if t.get("code_exec_success"))
        perception = sum(self.tool_calls.get(t, 0) for t in PERCEPTION_TOOLS)
        end = self.end_time or time.time()
        timeout_or_budget = self.budget_exceeded is not None or (
            self.termination_reason in BUDGET_TERMINATIONS
        )
        return {
            "task_name": self.task_name,
            "seed": self.seed,
            "mode": self.mode,
            "initial_state": self.initial_state,
            "coding_model": self.coding_model,
            "vdm_model": self.vdm_model,
            "budget": asdict(self.budget),
            "termination_reason": self.termination_reason,
            "termination_detail": self.termination_detail,
            "timeout_or_budget_exceeded": timeout_or_budget,
            "budget_exceeded": self.budget_exceeded,
            "posthoc": self.posthoc,
            "turns": self.turns,
            "model_calls": calls,
            "costs": {
                "num_code_executions": len(execs),
                "coding_llm_calls": sum(1 for c in calls if c["role"] == "coding"),
                "coding_input_tokens": self._sum("coding", "input_tokens"),
                "coding_output_tokens": self._sum("coding", "output_tokens"),
                "vdm_calls": sum(1 for c in calls if c["role"] == "vdm"),
                "vdm_input_tokens": self._sum("vdm", "input_tokens"),
                "vdm_output_tokens": self._sum("vdm", "output_tokens"),
                "vdm_gate_skips": sum(1 for t in self.turns if t.get("vdm_skipped_by_gate")),
                "total_model_calls": len(calls),
                "total_tokens": self.total_tokens,
                "tool_calls": {t: self.tool_calls.get(t, 0) for t in COUNTED_TOOLS},
                "extra_perception_calls": perception,
                "total_sim_steps": self.sim_steps(),
                "total_wall_time": end - self.start_time,
                "reset_wall_time": self.reset_wall_time,
            },
            "evaluation": {
                **self.evaluation,
                "episode_code_exec_success_rate": (ok_execs / len(execs)) if execs else None,
            },
        }

    def save(self) -> None:
        tmp = self.output_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, default=_json_default))
        tmp.replace(self.output_path)


def _with_max_tokens(args: Any, max_tokens: int) -> Any:
    import copy

    new_args = copy.copy(args)
    new_args.max_tokens = int(max_tokens)
    return new_args


def _json_default(o: Any) -> Any:
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    return repr(o)
