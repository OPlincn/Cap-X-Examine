#!/usr/bin/env python3
"""Post-hoc 2x2 root-cause verification table and generalization check (standard library only).

Compares, on the cube_lifting seeds where the frozen `ours` issued a false NOT COMPLETE after a
successful lift, the frozen `ours` (logs/test_eval, tag phase2-freeze) with the post-hoc variants
ref_only / gate_only / ref_and_gate (logs/posthoc_diagnostic/<regime>/...).

Usage:
    python3 scripts/posthoc_analysis.py [--seeds 102,107,109,110,114,116] [--markdown out.md] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

TASK = "cube_lifting"
VARIANTS = ("ref_only", "gate_only", "ref_and_gate")
LABEL = {
    "ours": "ours (frozen, `phase2-freeze`)",
    "ref_only": "ref_only (grasp-relative Δz, sticky gate)",
    "gate_only": "gate_only (original Δz, gate expiry)",
    "ref_and_gate": "ref_and_gate (both fixes)",
}
FACTORS = {"ours": ("original", "sticky"), "ref_only": ("grasp-relative", "sticky"),
           "gate_only": ("original", "expiry"), "ref_and_gate": ("grasp-relative", "expiry")}


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def episode_path(root: Path, condition: str, regime: str, seed: int) -> Path:
    if condition == "ours":
        return root / "test_eval" / TASK / "ours" / f"seed_{seed}" / f"episode_{TASK}_seed{seed}_ours.json"
    return (root / "posthoc_diagnostic" / regime / condition / TASK / f"seed_{seed}"
            / f"episode_{TASK}_seed{seed}_ours_{condition}.json")


def metrics(eps: list[dict]) -> dict:
    n = len(eps)
    t0_lifted = [e for e in eps if e["turns"] and e["turns"][0].get("turn_eval_success")]
    t0_fn = sum(1 for e in t0_lifted if e["turns"][0].get("vdm_verdict") == "NOT COMPLETE")
    execs = sum(len(e["turns"]) for e in eps)
    ok = sum(sum(1 for t in e["turns"] if t["code_exec_success"]) for e in eps)
    extra_after_goal = []
    for e in eps:
        first = next((i for i, t in enumerate(e["turns"]) if t.get("turn_eval_success")), None)
        if first is not None:
            extra_after_goal.append(len(e["turns"]) - 1 - first)
    return {
        "n": n,
        "turn0_lifted": len(t0_lifted),
        "turn0_false_negative": t0_fn,
        "avg_turns": statistics.fmean(len(e["turns"]) for e in eps) if eps else None,
        "avg_total_tokens": statistics.fmean(e["costs"]["total_tokens"] for e in eps) if eps else None,
        "avg_vdm_calls": statistics.fmean(e["costs"]["vdm_calls"] for e in eps) if eps else None,
        "avg_gate_skips": statistics.fmean(e["costs"].get("vdm_gate_skips", 0) for e in eps) if eps else None,
        "code_exec_rate": ok / execs if execs else None,
        "turn_cap_hits": sum(1 for e in eps if e["termination_reason"] == "max_turns"),
        "success": sum(1 for e in eps if e["evaluation"].get("official_task_success")),
        "avg_turns_after_goal": statistics.fmean(extra_after_goal) if extra_after_goal else None,
        "per_seed": {e["seed"]: {"turns": len(e["turns"]), "end": e["termination_reason"],
                                 "t0_verdict": e["turns"][0].get("vdm_verdict") if e["turns"] else None,
                                 "success": bool(e["evaluation"].get("official_task_success"))} for e in eps},
    }


def fmt(x, nd=1, pct=False):
    if x is None:
        return "-"
    return f"{100 * x:.{nd}f}%" if pct else f"{x:,.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", type=Path, default=Path("logs"))
    ap.add_argument("--seeds", default="102,107,109,110,114,116")
    ap.add_argument("--markdown", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]

    out: list[str] = []
    result: dict = {}
    problems: list[str] = []
    for regime in ("matched", "raw"):
        rows = {}
        for cond in ("ours",) + VARIANTS:
            eps = []
            for s in seeds:
                ep = load(episode_path(a.logs, cond, regime, s))
                if ep is None or "official_task_success" not in ep.get("evaluation", {}):
                    problems.append(f"{regime}/{cond}/seed {s}: missing or unevaluated")
                    continue
                eps.append(ep)
            rows[cond] = metrics(eps)
        result[regime] = rows
        title = ("Turn-0-matched (frozen turn-0 scene description and code replayed; gate_only also "
                 "replays the frozen turn-0 VDM verdict, whose input is unchanged)" if regime == "matched"
                 else "Raw (every call sampled live)")
        out.append(f"### {title}\n")
        out.append("| Condition | Reference frame | Gate | Turn-0 VDM false negative | Avg turns | Avg total tokens "
                   "| Avg VDM calls (gated) | Code exec success | 8-turn cap hits | Final success | Avg turns after goal reached |")
        out.append("|" + "---|" * 11)
        for cond, m in rows.items():
            ref, gate = FACTORS[cond]
            out.append(
                f"| {LABEL[cond]} | {ref} | {gate} | {m['turn0_false_negative']} / {m['turn0_lifted']} "
                f"| {fmt(m['avg_turns'])} | {fmt(m['avg_total_tokens'], 0)} | {fmt(m['avg_vdm_calls'])} ({fmt(m['avg_gate_skips'])}) "
                f"| {fmt(m['code_exec_rate'], 1, True)} | {m['turn_cap_hits']} / {m['n']} | {m['success']} / {m['n']} "
                f"| {fmt(m['avg_turns_after_goal'])} |"
            )
        out.append("")
        out.append("Per seed (turns, termination, turn-0 verdict):\n")
        out.append("| Seed | " + " | ".join(rows) + " |")
        out.append("|" + "---|" * (len(rows) + 1))
        for s in seeds:
            cells = []
            for cond, m in rows.items():
                p = m["per_seed"].get(s)
                cells.append("—" if p is None else
                             f"{'S' if p['success'] else 'F'} {p['turns']} {p['end']} / T0 {p['t0_verdict']}")
            out.append(f"| {s} | " + " | ".join(cells) + " |")
        out.append("")
    # Generalization check: ref_and_gate (raw) on the other cube_lifting test seeds.
    other = [s for s in range(100, 120) if s not in seeds]
    gen_rows = {}
    for cond in ("ours", "ref_and_gate"):
        eps = [ep for s in other
               if (ep := load(episode_path(a.logs, cond, "raw", s))) is not None
               and "official_task_success" in ep.get("evaluation", {})]
        if eps:
            gen_rows[cond] = metrics(eps)
    if len(gen_rows) == 2 and gen_rows["ref_and_gate"]["n"] > 0:
        result["generalization"] = {"seeds": other, **gen_rows}
        out.append(f"### Generalization: the other {len(other)} cube_lifting seeds (raw, live sampling)\n")
        out.append("| Condition | Episodes | Success | Avg turns | Avg total tokens | Avg VDM calls | 8-turn cap hits |")
        out.append("|---|---|---|---|---|---|---|")
        for cond, m in gen_rows.items():
            out.append(f"| {LABEL[cond]} | {m['n']} | {m['success']} / {m['n']} | {fmt(m['avg_turns'])} "
                       f"| {fmt(m['avg_total_tokens'], 0)} | {fmt(m['avg_vdm_calls'])} | {m['turn_cap_hits']} / {m['n']} |")
        out.append("")
    if problems:
        out.append("**Missing episodes:** " + "; ".join(problems))
    md = "\n".join(out)
    print(md)
    if a.markdown:
        a.markdown.write_text(md + "\n")
    if a.json:
        a.json.write_text(json.dumps({"seeds": seeds, "results": result, "problems": problems}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
