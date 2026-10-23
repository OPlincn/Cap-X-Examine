#!/usr/bin/env python3
"""Recompute per-task results from episode telemetry JSON files (standard library only).

Usage:
    python3 scripts/summarize_episodes.py logs/dev_baseline [--markdown out.md] [--expect-seeds 0,1,2]

Every ``episode_*.json`` under the run directory is one episode. Episodes are never
dropped: a missing/duplicated (task, seed, mode) key is reported as an integrity error.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_episodes(run_dir: Path) -> list[dict]:
    episodes = []
    for path in sorted(run_dir.rglob("episode_*.json")):
        d = json.loads(path.read_text())
        d["_path"] = str(path)
        episodes.append(d)
    return episodes


def mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def fmt(x: float | None, nd: int = 1) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def summarize(episodes: list[dict]) -> dict[tuple[str, str], dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for e in episodes:
        groups[(e["task_name"], e["mode"])].append(e)
    rows = {}
    for key, eps in sorted(groups.items()):
        n = len(eps)
        succ = sum(1 for e in eps if e["evaluation"].get("official_task_success"))
        execs = sum(e["costs"]["num_code_executions"] for e in eps)
        ok_execs = sum(sum(1 for t in e["turns"] if t.get("code_exec_success")) for e in eps)
        rows[key] = {
            "n": n,
            "success": succ,
            "code_exec_success_rate": ok_execs / execs if execs else None,
            "avg_turns": mean([e["costs"]["num_code_executions"] for e in eps]),
            "avg_coding_tokens": mean(
                [e["costs"]["coding_input_tokens"] + e["costs"]["coding_output_tokens"] for e in eps]
            ),
            "avg_vdm_tokens": mean(
                [e["costs"]["vdm_input_tokens"] + e["costs"]["vdm_output_tokens"] for e in eps]
            ),
            "avg_total_tokens": mean([e["costs"]["total_tokens"] for e in eps]),
            "avg_model_calls": mean([e["costs"]["total_model_calls"] for e in eps]),
            "avg_vdm_calls": mean([e["costs"]["vdm_calls"] for e in eps]),
            "avg_perception_calls": mean([e["costs"]["extra_perception_calls"] for e in eps]),
            "avg_sim_steps": mean([e["costs"]["total_sim_steps"] for e in eps]),
            "avg_wall_time": mean([e["costs"]["total_wall_time"] for e in eps]),
            "avg_reward": mean([e["evaluation"].get("official_reward") for e in eps]),
            "terminations": dict(Counter(e["termination_reason"] for e in eps)),
            "budget_or_timeout": sum(1 for e in eps if e["timeout_or_budget_exceeded"]),
            "seeds": sorted(e["seed"] for e in eps),
        }
    return rows


def integrity(episodes: list[dict], expect_seeds: list[int] | None) -> list[str]:
    problems = []
    keys = Counter((e["task_name"], e["mode"], e["seed"]) for e in episodes)
    for k, c in keys.items():
        if c > 1:
            problems.append(f"duplicate episode {k} x{c}")
    if expect_seeds is not None:
        for task, mode in {(e["task_name"], e["mode"]) for e in episodes}:
            for s in expect_seeds:
                if (task, mode, s) not in keys:
                    problems.append(f"missing episode {(task, mode, s)}")
    for e in episodes:
        if "official_task_success" not in e.get("evaluation", {}):
            problems.append(f"no official evaluation in {e['_path']}")
    return problems


def to_markdown(rows: dict[tuple[str, str], dict]) -> str:
    lines = [
        "| Task | Mode | Success | Code exec success | Avg turns | Avg coding tok | Avg VDM tok "
        "| Avg total tok | Avg model calls | Avg perception calls | Avg sim steps | Avg wall time (s) "
        "| Avg reward | Terminations |",
        "|" + "---|" * 14,
    ]
    for (task, mode), r in rows.items():
        term = ", ".join(f"{k}:{v}" for k, v in sorted(r["terminations"].items()))
        lines.append(
            f"| {task} | {mode} | {r['success']}/{r['n']} | {fmt(r['code_exec_success_rate'] and 100 * r['code_exec_success_rate'])}% "
            f"| {fmt(r['avg_turns'])} | {fmt(r['avg_coding_tokens'], 0)} | {fmt(r['avg_vdm_tokens'], 0)} "
            f"| {fmt(r['avg_total_tokens'], 0)} | {fmt(r['avg_model_calls'])} | {fmt(r['avg_perception_calls'])} "
            f"| {fmt(r['avg_sim_steps'], 0)} | {fmt(r['avg_wall_time'], 0)} | {fmt(r['avg_reward'], 3)} | {term} |"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--markdown", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--expect-seeds", type=str, default=None)
    a = ap.parse_args()

    episodes = load_episodes(a.run_dir)
    if not episodes:
        print(f"no episode_*.json under {a.run_dir}", file=sys.stderr)
        return 1
    expect = [int(s) for s in a.expect_seeds.split(",")] if a.expect_seeds else None
    problems = integrity(episodes, expect)
    rows = summarize(episodes)
    md = to_markdown(rows)
    print(md)
    if problems:
        print("\nINTEGRITY PROBLEMS:\n" + "\n".join(problems))
    if a.markdown:
        a.markdown.write_text(md + "\n")
    if a.json:
        a.json.write_text(
            json.dumps(
                {"rows": {f"{t}/{m}": r for (t, m), r in rows.items()}, "integrity_problems": problems},
                indent=2,
            )
        )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
