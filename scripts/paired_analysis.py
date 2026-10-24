#!/usr/bin/env python3
"""Recompute all Phase 2 tables from raw episode logs (standard library only).

Usage:
    python3 scripts/paired_analysis.py logs/test_eval --seeds 100-119 \
        [--markdown logs/test_eval/results.md] [--json logs/test_eval/results.json]

Layout expected: <run_dir>/<task>/<mode>/seed_<k>/episode_<task>_seed<k>_<mode>.json
Integrity: every (task, mode, seed) must exist exactly once and all modes of a
(task, seed) must share the same initial-state hash; violations are reported and
make the exit code non-zero (no episode is dropped or re-run).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

MODES = ("baseline", "ablation", "ours")


def parse_seeds(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return out


def load(run_dir: Path) -> dict[tuple[str, str, int], list[dict]]:
    eps: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for p in sorted(run_dir.rglob("episode_*.json")):
        d = json.loads(p.read_text())
        d["_path"] = str(p)
        eps[(d["task_name"], d["mode"], d["seed"])].append(d)
    return eps


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def f(x, nd=1, pct=False):
    if x is None:
        return "-"
    return f"{100 * x:.{nd}f}%" if pct else f"{x:,.{nd}f}"


# --------------------------------------------------------------------------- #
# Per-episode derived metrics
# --------------------------------------------------------------------------- #
def episode_metrics(e: dict) -> dict:
    turns = e["turns"]
    c = e["costs"]
    succ = bool(e["evaluation"].get("official_task_success"))
    evals = [t.get("turn_eval_success") for t in turns]
    goal_reached = any(v for v in evals if v is not None)
    first_true = next((i for i, v in enumerate(evals) if v), None)
    undone = first_true is not None and (
        any(v is False for v in evals[first_true + 1:]) or not succ
    )
    premature = sum(
        1 for t in turns if t.get("decision_after") == "finish" and t.get("turn_eval_success") is False
    )
    verdicts = []
    for t in turns:
        v, ev = t.get("vdm_verdict"), t.get("turn_eval_success")
        if v is None or ev is None:
            continue
        verdicts.append({
            "verdict": v,
            "truth": ev,
            "source": t.get("vdm_verdict_source"),
            "gate": bool(t.get("vdm_skipped_by_gate")),
        })
    return {
        "success": succ,
        "turns": c["num_code_executions"],
        "exec_ok": sum(1 for t in turns if t.get("code_exec_success")),
        "coding_in": c["coding_input_tokens"],
        "coding_out": c["coding_output_tokens"],
        "vdm_in": c["vdm_input_tokens"],
        "vdm_out": c["vdm_output_tokens"],
        "total_tokens": c["total_tokens"],
        "model_calls": c["total_model_calls"],
        "vdm_calls": c["vdm_calls"],
        "gate_skips": c.get("vdm_gate_skips", sum(1 for t in turns if t.get("vdm_skipped_by_gate"))),
        "tool_calls": sum(c["tool_calls"].values()),
        "perception_calls": c["extra_perception_calls"],
        "sim_steps": c["total_sim_steps"],
        "wall_time": c["total_wall_time"],
        "budget_or_timeout": bool(e["timeout_or_budget_exceeded"]),
        "termination": e["termination_reason"],
        "goal_reached": goal_reached,
        "goal_undone": undone,
        "premature_finish": premature,
        "verdicts": verdicts,
        "state_hash": (e.get("initial_state") or {}).get("qpos_sha256_16"),
    }


def aggregate(ms: list[dict]) -> dict:
    n = len(ms)
    execs = sum(m["turns"] for m in ms)
    v_all = [v for m in ms for v in m["verdicts"]]
    v_vdm = [v for v in v_all if not v["gate"]]

    def acc(vs):
        if not vs:
            return None, None, None, 0
        correct = sum(
            1 for v in vs
            if (v["verdict"] == "COMPLETE" and v["truth"]) or (v["verdict"] == "NOT COMPLETE" and not v["truth"])
        )
        false_complete = sum(1 for v in vs if v["verdict"] == "COMPLETE" and not v["truth"])
        false_incomplete = sum(1 for v in vs if v["verdict"] == "NOT COMPLETE" and v["truth"])
        return correct / len(vs), false_complete, false_incomplete, len(vs)

    a_all = acc(v_all)
    a_vdm = acc(v_vdm)
    vdm_slots = sum(m["vdm_calls"] for m in ms) + sum(m["gate_skips"] for m in ms)
    return {
        "n": n,
        "success": sum(m["success"] for m in ms),
        "code_exec_rate": sum(m["exec_ok"] for m in ms) / execs if execs else None,
        "avg": {k: mean([m[k] for m in ms]) for k in (
            "turns", "coding_in", "coding_out", "vdm_in", "vdm_out", "total_tokens", "model_calls",
            "vdm_calls", "gate_skips", "tool_calls", "perception_calls", "sim_steps", "wall_time")},
        "gate_skip_share": (sum(m["gate_skips"] for m in ms) / vdm_slots) if vdm_slots else 0.0,
        "budget_or_timeout": sum(m["budget_or_timeout"] for m in ms),
        "terminations": dict(sorted(
            {t: sum(1 for m in ms if m["termination"] == t) for t in {m["termination"] for m in ms}}.items())),
        "verdict_acc_all": a_all[0], "false_complete_all": a_all[1], "false_incomplete_all": a_all[2],
        "verdicts_all": a_all[3],
        "uncertain_all": sum(1 for v in v_all if v["verdict"] == "UNCERTAIN"),
        "verdict_acc_vdm": a_vdm[0], "verdicts_vdm": a_vdm[3],
        "goal_reached": sum(m["goal_reached"] for m in ms),
        "goal_undone": sum(m["goal_undone"] for m in ms),
        "goal_undone_final_fail": sum(1 for m in ms if m["goal_undone"] and not m["success"]),
        "premature_finish_eps": sum(1 for m in ms if m["premature_finish"]),
    }


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--seeds", default="100-119")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--markdown", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    seeds = parse_seeds(a.seeds)
    modes = [m for m in a.modes.split(",") if m]

    eps = load(a.run_dir)
    tasks = sorted({k[0] for k in eps})
    problems: list[str] = []
    M: dict[tuple[str, str, int], dict] = {}
    for task in tasks:
        for mode in modes:
            for s in seeds:
                lst = eps.get((task, mode, s), [])
                if len(lst) != 1:
                    problems.append(f"{task}/{mode}/seed {s}: {len(lst)} episode logs")
                    continue
                if "official_task_success" not in lst[0].get("evaluation", {}):
                    problems.append(f"{task}/{mode}/seed {s}: no official evaluation")
                    continue
                M[(task, mode, s)] = episode_metrics(lst[0])
        for s in seeds:
            hashes = {M[(task, m, s)]["state_hash"] for m in modes if (task, m, s) in M}
            if len(hashes) > 1:
                problems.append(f"{task}/seed {s}: initial-state hash mismatch {sorted(hashes)}")
    extra = sorted(k for k in eps if k[2] not in seeds or k[1] not in modes)
    if extra:
        problems.append(f"{len(extra)} episode keys outside the planned seeds/modes (ignored): {extra[:5]}")

    groups = {}
    for task in tasks + ["ALL"]:
        for mode in modes:
            ms = [M[k] for k in M if k[1] == mode and (task == "ALL" or k[0] == task)]
            if ms:
                groups[(task, mode)] = aggregate(ms)

    out: list[str] = []
    out.append("## Main results\n")
    out.append("| Task | Mode | Success | Code exec | Avg turns | Coding tok in/out | VDM tok in/out | Total tok "
               "| Model calls | VDM calls (gate-skipped %) | Tool calls | Sim steps | Wall (s) | Budget/timeout |")
    out.append("|" + "---|" * 14)
    for (task, mode), g in groups.items():
        av = g["avg"]
        out.append(
            f"| {task} | {mode} | {g['success']}/{g['n']} ({f(g['success'] / g['n'], 0, True)}) "
            f"| {f(g['code_exec_rate'], 1, True)} | {f(av['turns'])} | {f(av['coding_in'], 0)}/{f(av['coding_out'], 0)} "
            f"| {f(av['vdm_in'], 0)}/{f(av['vdm_out'], 0)} | {f(av['total_tokens'], 0)} | {f(av['model_calls'])} "
            f"| {f(av['vdm_calls'])} ({f(g['gate_skip_share'], 0, True)}) | {f(av['tool_calls'])} "
            f"| {f(av['sim_steps'], 0)} | {f(av['wall_time'], 0)} | {g['budget_or_timeout']} |"
        )
    out.append("\nTermination reasons: " + "; ".join(
        f"{t}/{m}: {g['terminations']}" for (t, m), g in groups.items() if t == "ALL"))

    out.append("\n## Diagnostics for hypothesis H\n")
    out.append("| Task | Mode | VDM verdict acc. (all slots) | false COMPLETE | false NOT COMPLETE | UNCERTAIN | "
               "verdict acc. (VDM calls only) | Episodes goal reached | Goal undone (of which final fail) | "
               "Episodes with premature FINISH |")
    out.append("|" + "---|" * 10)
    for (task, mode), g in groups.items():
        out.append(
            f"| {task} | {mode} | {f(g['verdict_acc_all'], 1, True)} (n={g['verdicts_all']}) | {g['false_complete_all']} "
            f"| {g['false_incomplete_all']} | {g['uncertain_all']} | {f(g['verdict_acc_vdm'], 1, True)} (n={g['verdicts_vdm']}) "
            f"| {g['goal_reached']} | {g['goal_undone']} ({g['goal_undone_final_fail']}) | {g['premature_finish_eps']} |"
        )

    out.append("\n## Paired comparison (same seed, same initial state)\n")
    paired = {}
    for task in tasks:
        for a_mode, b_mode in (("baseline", "ours"), ("ablation", "ours"), ("baseline", "ablation")):
            if a_mode not in modes or b_mode not in modes:
                continue
            wins, losses, ties_s, ties_f = [], [], [], []
            eff = []
            for s in seeds:
                if (task, a_mode, s) not in M or (task, b_mode, s) not in M:
                    continue
                x, y = M[(task, a_mode, s)], M[(task, b_mode, s)]
                if not x["success"] and y["success"]:
                    wins.append(s)
                elif x["success"] and not y["success"]:
                    losses.append(s)
                elif x["success"]:
                    ties_s.append(s)
                    eff.append((y["turns"] - x["turns"], y["total_tokens"] - x["total_tokens"]))
                else:
                    ties_f.append(s)
            paired[(task, a_mode, b_mode)] = {
                "wins": wins, "losses": losses, "ties_success": ties_s, "ties_fail": ties_f,
                "tied_success_delta_turns": mean([d[0] for d in eff]),
                "tied_success_delta_tokens": mean([d[1] for d in eff]),
                "tied_success_b_cheaper": sum(1 for d in eff if d[1] < 0),
            }
    out.append(f"| Task | {'A → B'} | B wins (A fail, B success) | B losses (A success, B fail) | Ties S/S | Ties F/F "
               "| On S/S ties: Δturns (B−A) | Δtokens (B−A) | B cheaper |")
    out.append("|" + "---|" * 9)
    for (task, am, bm), p in paired.items():
        out.append(
            f"| {task} | {am} → {bm} | {len(p['wins'])} {p['wins']} | {len(p['losses'])} {p['losses']} "
            f"| {len(p['ties_success'])} | {len(p['ties_fail'])} | {f(p['tied_success_delta_turns'], 2)} "
            f"| {f(p['tied_success_delta_tokens'], 0)} | {p['tied_success_b_cheaper']}/{len(p['ties_success'])} |"
        )

    out.append("\n## Per-seed outcomes (S = official success, number = code executions)\n")
    for task in tasks:
        out.append(f"\n**{task}**\n")
        out.append("| Seed | " + " | ".join(modes) + " | initial-state hash |")
        out.append("|" + "---|" * (len(modes) + 2))
        for s in seeds:
            cells = []
            for m in modes:
                k = (task, m, s)
                cells.append("—" if k not in M else f"{'S' if M[k]['success'] else 'F'} {M[k]['turns']}"
                             + (" (B)" if M[k]["budget_or_timeout"] else ""))
            h = next((M[(task, m, s)]["state_hash"] for m in modes if (task, m, s) in M), "—")
            out.append(f"| {s} | " + " | ".join(cells) + f" | `{h}` |")

    if problems:
        out.append("\n## INTEGRITY PROBLEMS\n")
        out.extend(f"- {p}" for p in problems)
    md = "\n".join(out)
    print(md)
    if a.markdown:
        a.markdown.write_text(md + "\n")
    if a.json:
        a.json.write_text(json.dumps({
            "groups": {f"{t}/{m}": g for (t, m), g in groups.items()},
            "paired": {f"{t}/{am}->{bm}": p for (t, am, bm), p in paired.items()},
            "per_episode": {f"{t}/{m}/{s}": {k: v for k, v in mm.items() if k != "verdicts"}
                            for (t, m, s), mm in M.items()},
            "integrity_problems": problems,
        }, indent=2))
    return 1 if any("hash mismatch" in p or "episode logs" in p or "no official" in p for p in problems) else 0


if __name__ == "__main__":
    sys.exit(main())
