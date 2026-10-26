#!/usr/bin/env python3
"""Secondary Phase 2 diagnostics recomputed from raw episode logs (standard library only).

Usage:
    python3 scripts/phase2_diagnostics.py logs/test_eval [--markdown logs/test_eval/diagnostics.md]

Sections:
  1. Exact two-sided sign tests on the paired win/loss counts (per task and pooled).
  2. Root cause of every failed episode (premature FINISH, sim-step horizon with/without
     saturated moves, token cap, turn cap).
  3. Turn-0 saturated-move programs per mode (turn 0 has an identical prompt in all modes).
  4. Zero-motion turns: VDM calls/tokens spent, false verdicts, FINISH decisions.
  5. cube_lifting/ours: VDM verdict vs. the record's end-effector Δz on goal-reached turns.
  6. Code-execution failures by exception type.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

MODES = ("baseline", "ablation", "ours")
MOTION = ("move_to_joints", "open_gripper", "close_gripper")


def sign_p(w: int, l: int) -> float:
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def fisher_two_sided(a: int, n1: int, b: int, n2: int) -> float:
    N, K = n1 + n2, a + b

    def pr(x: int) -> float:
        return math.comb(K, x) * math.comb(N - K, n1 - x) / math.comb(N, n1)

    p0 = pr(a)
    return min(1.0, sum(pr(x) for x in range(0, min(K, n1) + 1) if pr(x) <= p0 + 1e-12))


def saturated(t: dict) -> bool:
    mv = t["tool_calls_delta"].get("move_to_joints", 0)
    return mv >= 3 and t["sim_steps_delta"] >= 95 * mv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--markdown", type=Path)
    a = ap.parse_args()
    eps = [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(a.run_dir / "*/*/seed_*/episode_*.json")))]
    tasks = sorted({e["task_name"] for e in eps})
    out: list[str] = []

    # 1. sign tests
    succ = {(e["task_name"], e["mode"], e["seed"]): bool(e["evaluation"].get("official_task_success")) for e in eps}
    out.append("## 1. Paired sign tests (exact, two-sided)\n")
    out.append("| Task | A → B | B wins | B losses | p |")
    out.append("|---|---|---|---|---|")
    pooled = defaultdict(lambda: [0, 0])
    for task in tasks + ["ALL"]:
        for am, bm in (("baseline", "ours"), ("ablation", "ours"), ("baseline", "ablation")):
            if task == "ALL":
                w, l = pooled[(am, bm)]
            else:
                seeds = sorted({s for (t, m, s) in succ if t == task})
                w = sum(1 for s in seeds if not succ[(task, am, s)] and succ[(task, bm, s)])
                l = sum(1 for s in seeds if succ[(task, am, s)] and not succ[(task, bm, s)])
                pooled[(am, bm)][0] += w
                pooled[(am, bm)][1] += l
            out.append(f"| {task} | {am} → {bm} | {w} | {l} | {sign_p(w, l):.3f} |")

    # 2. root causes
    out.append("\n## 2. Root cause of failed episodes\n")
    out.append("| Task | Mode | Failures | Premature FINISH | Sim-step horizon (saturated moves) | Sim-step horizon (other) | Token cap | Turn cap |")
    out.append("|---|---|---|---|---|---|---|---|")
    for task in tasks:
        for mode in MODES:
            c = Counter()
            for e in eps:
                if e["task_name"] != task or e["mode"] != mode or e["evaluation"].get("official_task_success"):
                    continue
                term = e["termination_reason"]
                if term == "finish":
                    c["finish"] += 1
                elif term == "env_horizon":
                    c["hz_sat" if any(saturated(t) for t in e["turns"]) else "hz"] += 1
                else:
                    c[term] += 1
            out.append(f"| {task} | {mode} | {sum(c.values())} | {c['finish']} | {c['hz_sat']} | {c['hz']} "
                       f"| {c['max_total_tokens']} | {c['max_turns']} |")

    # 3. turn-0 saturation
    out.append("\n## 3. Turn-0 programs whose every move saturated (identical turn-0 prompt in all modes)\n")
    out.append("| Mode | Episodes with saturated turn 0 | of which failed | per task |")
    out.append("|---|---|---|---|")
    sat0 = {}
    for mode in MODES:
        sel = [e for e in eps if e["mode"] == mode and e["turns"] and saturated(e["turns"][0])]
        sat0[mode] = len(sel)
        failed = sum(1 for e in sel if not e["evaluation"].get("official_task_success"))
        out.append(f"| {mode} | {len(sel)} / {sum(1 for e in eps if e['mode'] == mode)} | {failed} "
                   f"| {dict(Counter(e['task_name'] for e in sel))} |")
    n_b = sum(1 for e in eps if e["mode"] == "baseline")
    n_o = sum(1 for e in eps if e["mode"] != "baseline")
    out.append(f"\nFisher exact (baseline vs ablation+ours): p = "
               f"{fisher_two_sided(sat0['baseline'], n_b, sat0['ablation'] + sat0['ours'], n_o):.3f}")

    # 4. zero-motion turns
    out.append("\n## 4. Zero-motion turns (no motion primitive ran) that received feedback\n")
    out.append("| Mode | Turns | VDM calls | VDM tokens | False COMPLETE | False NOT COMPLETE | FINISH after | Premature FINISH after |")
    out.append("|---|---|---|---|---|---|---|---|")
    for mode in MODES:
        c = Counter()
        for e in eps:
            if e["mode"] != mode:
                continue
            for t in e["turns"]:
                if any(t["tool_calls_delta"].get(k, 0) for k in MOTION) or t.get("decision_after") is None:
                    continue
                c["turns"] += 1
                if t.get("vdm_call_index") is not None:
                    call = e["model_calls"][t["vdm_call_index"]]
                    c["calls"] += 1
                    c["tokens"] += (call["input_tokens"] or 0) + (call["output_tokens"] or 0)
                ev = t.get("turn_eval_success")
                if t.get("vdm_verdict") == "COMPLETE" and not ev:
                    c["fc"] += 1
                if t.get("vdm_verdict") == "NOT COMPLETE" and ev:
                    c["fn"] += 1
                if t.get("decision_after") == "finish":
                    c["fin"] += 1
                    if not ev:
                        c["pfin"] += 1
        out.append(f"| {mode} | {c['turns']} | {c['calls']} | {c['tokens']:,} | {c['fc']} | {c['fn']} | {c['fin']} | {c['pfin']} |")

    # 5. lifting Δz
    out.append("\n## 5. cube_lifting / ours: structured VDM verdict on goal-reached turns vs. record Δz\n")
    out.append("| Seed | Turn | Verdict | Δz (after − before) | EEF z after | Gripper after close | Episode turns | Termination |")
    out.append("|---|---|---|---|---|---|---|---|")
    for e in eps:
        if e["task_name"] != "cube_lifting" or e["mode"] != "ours":
            continue
        for t in e["turns"]:
            if t.get("vdm_verdict_source") == "structured" and t.get("turn_eval_success"):
                pb, pa = t["proprio_before"], t["proprio_after"]
                closes = [round(x["gripper_opening"], 2) for x in t["exec_record"].get("gripper_events", [])
                          if x["call"] == "close_gripper"]
                out.append(f"| {e['seed']} | {t['turn_idx']} | {t['vdm_verdict']} | {pa['eef_xyz'][2] - pb['eef_xyz'][2]:+.3f} "
                           f"| {pa['eef_xyz'][2]:+.3f} | {closes} | {len(e['turns'])} | {e['termination_reason']} |")

    # 6. exceptions
    out.append("\n## 6. Code-execution failures by exception type\n")
    out.append("| Mode | Turns | Failed executions | Top exception types |")
    out.append("|---|---|---|---|")
    for mode in MODES:
        ex = Counter()
        n = 0
        for e in eps:
            if e["mode"] != mode:
                continue
            for t in e["turns"]:
                n += 1
                if not t["code_exec_success"]:
                    line = (t["stderr"].strip().splitlines() or ["?"])[-1]
                    ex[re.split(r"[:(]", line)[0][:40]] += 1
        out.append(f"| {mode} | {n} | {sum(ex.values())} | {', '.join(f'{k} {v}' for k, v in ex.most_common(4))} |")

    md = "\n".join(out)
    print(md)
    if a.markdown:
        a.markdown.write_text(md + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
