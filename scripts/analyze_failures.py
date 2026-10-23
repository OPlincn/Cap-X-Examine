#!/usr/bin/env python3
"""Tag recurring failure patterns in episode telemetry (standard library only).

Usage:
    python3 scripts/analyze_failures.py logs/dev_baseline [--json out.json]

Patterns (heuristic tags for triage; case studies are confirmed by reading the logs):
  A  premature FINISH: the agent answered FINISH but the official check failed;
     also flags VDM outputs that assert completion on a failed episode.
  B  residual-state hazards: a turn ends with an exception after motion started
     (sim_steps_delta > 0), so the next turn inherits an unknown arm/gripper state.
  C  wasted feedback: VDM called after a turn that crashed before any motion
     (sim_steps_delta == 0, before/after images identical in content), and
     consecutive turns failing with the same exception line.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

COMPLETION_CLAIMS = re.compile(
    r"(task (has been|is|appears to (be|have been)) (successfully )?(completed|accomplished|achieved)"
    r"|successfully (lifted|stacked|placed|wiped|picked|completed)"
    r"|has been (lifted|stacked|placed|wiped)( successfully)?"
    r"|(is|now) (stacked|resting) on top of"
    r"|spill (has been|is) (fully |completely )?(wiped|removed|cleaned))",
    re.IGNORECASE,
)
NEGATIONS = re.compile(r"\b(not|no longer|has not|hasn't|isn't|incomplete|yet to)\b", re.IGNORECASE)


def last_exception_line(stderr: str) -> str | None:
    lines = [ln.strip() for ln in stderr.strip().splitlines() if ln.strip()]
    for ln in reversed(lines):
        if re.match(r"^[A-Za-z_][\w.]*(Error|Exception|Exceeded|Interrupt)\b", ln):
            return re.sub(r"0x[0-9a-f]+", "0x?", ln)
    return lines[-1] if lines else None


def claims_completion(text: str | None) -> bool:
    if not text:
        return False
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if COMPLETION_CLAIMS.search(sent) and not NEGATIONS.search(sent):
            return True
    return False


def analyze(ep: dict) -> dict:
    success = bool(ep["evaluation"].get("official_task_success"))
    turns = ep["turns"]
    tags: list[str] = []
    per_turn = []
    prev_exc = None
    repeated = 0
    wasted_vdm_tokens = 0
    calls = ep["model_calls"]
    for t in turns:
        exc = last_exception_line(t["stderr"]) if t["stderr"] else None
        zero_motion_crash = (not t["code_exec_success"]) and t["sim_steps_delta"] == 0
        vdm_claim = claims_completion(t.get("vdm_output"))
        if zero_motion_crash and t.get("vdm_call_index") is not None:
            c = calls[t["vdm_call_index"]]
            wasted_vdm_tokens += (c["input_tokens"] or 0) + (c["output_tokens"] or 0)
        if exc and exc == prev_exc:
            repeated += 1
        prev_exc = exc if exc else None
        per_turn.append({
            "turn": t["turn_idx"],
            "exec_ok": t["code_exec_success"],
            "sim_steps": t["sim_steps_delta"],
            "exception": exc,
            "zero_motion_crash": zero_motion_crash,
            "motion_then_crash": (not t["code_exec_success"]) and t["sim_steps_delta"] > 0,
            "vdm_claims_completion": vdm_claim,
            "decision_after": t.get("decision_after"),
        })

    finished = ep["termination_reason"] == "finish"
    if finished and not success:
        tags.append("A:premature_finish")
    if not success and any(p["vdm_claims_completion"] for p in per_turn):
        tags.append("A:vdm_false_completion")
    if any(p["motion_then_crash"] for p in per_turn[:-1]):
        tags.append("B:motion_then_crash")
    n_zero = sum(1 for p in per_turn if p["zero_motion_crash"] and p["decision_after"] is not None)
    if n_zero:
        tags.append(f"C:vdm_on_zero_motion_crash x{n_zero}")
    if repeated:
        tags.append(f"C:repeated_exception x{repeated}")
    return {
        "task": ep["task_name"],
        "seed": ep["seed"],
        "mode": ep["mode"],
        "success": success,
        "termination": ep["termination_reason"],
        "turns": len(turns),
        "tags": tags,
        "wasted_vdm_tokens": wasted_vdm_tokens,
        "per_turn": per_turn,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    results = []
    for path in sorted(a.run_dir.rglob("episode_*.json")):
        ep = json.loads(path.read_text())
        if "official_task_success" not in ep.get("evaluation", {}):
            print(f"[in progress or unevaluated] {path}")
            continue
        results.append(analyze(ep))
    for r in results:
        status = "SUCCESS" if r["success"] else "FAIL"
        print(f"{r['task']:<13} seed={r['seed']:<3} {status:<7} turns={r['turns']:<2} "
              f"end={r['termination']:<16} tags={', '.join(r['tags']) or '-'}")
        for p in r["per_turn"]:
            print(f"    t{p['turn']}: ok={p['exec_ok']!s:<5} steps={p['sim_steps']:<4} "
                  f"vdm_done={p['vdm_claims_completion']!s:<5} next={p['decision_after']} exc={p['exception']}")
    if a.json:
        a.json.write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
