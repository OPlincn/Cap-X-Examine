# Phase 2 — Paired 20-Seed Test Evaluation: Results and Analysis

*Frozen code `ed29cc7` (tag `phase2-freeze`). Test seeds 100–119. 3 tasks × 20 seeds × 3 modes = 180 episodes, all complete.*
*Every number below is recomputed from `cap-x/logs/test_eval/` by:*
```bash
cd cap-x
python3 scripts/paired_analysis.py logs/test_eval --seeds 100-119 --markdown logs/test_eval/results.md --json logs/test_eval/results.json
python3 scripts/phase2_diagnostics.py logs/test_eval --markdown logs/test_eval/diagnostics.md
```

## 0. Summary

- **Hypothesis H is not supported on the primary metric.** Official task success is baseline **51/60 (85%)**, ablation **47/60 (78%)** and ours (Execution-Grounded VDM + zero-motion gate) **43/60 (72%)**.
  - Pooled paired comparison baseline → ours: 5 wins, 13 losses (exact sign test p = 0.096).
  - The only per-task comparison with p < 0.05 is a **regression**: cube_stack baseline → ours, 0 wins / 6 losses, p = 0.031.
- **The zero-motion gate did what it was designed for, at a cost.** On the 66 turns where no motion primitive ran, `ours` produced
  - 0 false COMPLETE verdicts and 0 premature FINISH decisions (baseline: 3 false COMPLETE and 3 premature FINISH; ablation: 2 and 1);
  - 0 VDM calls (baseline spent 143k VDM tokens on such turns over 60 episodes).
  - The cost: the gate re-issued a preserved **false NOT COMPLETE** 36 times.
- **The execution record caused a systematic false-negative on cube_lifting.** It reports start→end end-effector deltas. A pick-and-lift that starts from the high home pose therefore always shows a negative Δz. In 6/20 lifting episodes the grounded VDM read this as "not lifted", and the gate preserved that verdict until the 8-turn cap. All 6 still succeeded, but each cost 8 turns.
- **Part of the cube_stack regression is sampling noise.** 6 of the 11 structured-mode failures were decided by a turn-0 program in which every move saturated. Turn 0 has an identical prompt in all modes, so these losses cannot come from the Step-3 change (baseline 1/60 vs. ablation+ours 7/120 such turn-0 programs, Fisher p = 0.27).
- **Cost.** Total tokens per episode are essentially equal: baseline 27.7k, ablation 27.2k, ours 27.9k.
  - `ours` moves cost from the VDM (−20% VDM tokens, −28% VDM calls; 30% of VDM slots gated) to the coding agent (+11% coding tokens).
  - `ours` hits budget caps more often: 18 of 60 episodes vs. 9 for baseline.

---

## 1. Protocol and data integrity

| Item | Value (identical for all modes) |
|---|---|
| Models | coding `deepseek-v4.1-flash-1017` (T = 0.2, thinking off); VDM `qwen3.8-flash` (T = 0.2, thinking off). Returned model ids are identical in all logged calls |
| Budgets per episode | 8 code executions · 16 model calls (coding + VDM) · 60,000 tokens · 1500 sim steps · 600 s |
| Pairing | identical initial-state hash (`qpos_sha256_16`) for all three modes of every (task, seed), checked for all 60 groups (no mismatch) |
| Order | seed-major, mode order rotated per seed; fresh process per episode; no outcome-based reruns |

**Incident (details: `logs/test_eval/INCIDENT_2026-10-24_gpu_crash.md`).** A GPU driver crash stopped the run after 153 episodes. Applying the rule fixed before the test (`docs/phase2/FROZEN_PROTOCOL.md`), two episodes were moved to `logs/test_eval_invalid/` and run once more on unchanged code:
- `cube_stack/ours/seed_119`: GraspNet returned HTTP 500 because of a transient CUDA out-of-memory error.
- `cube_restack/ablation/seed_111`: interrupted while loading the scene, before any model call.

The 26 never-started episodes (cube_restack seeds 111–119) ran after a two-day gap and a reboot. Within-seed pairing is preserved because all three modes of each seed ran together. The single exception is the `cube_stack/ours/seed_119` replacement, whose two partner episodes ran before the gap.

---

## 2. Main results

### 2.1 Success and code execution

| Task | baseline | ablation | ours | Code-exec success (b / a / o) |
|---|---|---|---|---|
| cube_lifting | 19/20 (95%) | 19/20 (95%) | **20/20 (100%)** | 69.6% / 73.5% / 33.3% |
| cube_stack | **20/20 (100%)** | 15/20 (75%) | 14/20 (70%) | 65.2% / 66.7% / 72.1% |
| cube_restack | 12/20 (60%) | **13/20 (65%)** | 9/20 (45%) | 71.8% / 66.2% / 60.6% |
| **Overall** | **51/60 (85%)** | 47/60 (78%) | 43/60 (72%) | 69.3% / 68.0% / 53.7% |

The low code-exec rate of `ours` on cube_lifting is the gate loop of §4.2: 45 of its failed executions are the same `plan_grasp` AssertionError on an already-held cube.

### 2.2 Budget caps vs. actual average cost per episode (overall)

| Quantity | Cap | baseline | ablation | ours |
|---|---|---|---|---|
| Code executions (turns) | 8 | 2.7 | 2.5 | 3.0 |
| Model calls (coding + VDM) | 16 | 7.2 | 6.8 | 6.3 |
| — VDM calls (share of VDM slots skipped by gate) | — | 3.6 (0%) | 3.4 (0%) | 2.6 (30%) |
| Coding tokens in / out | — | 15,184 / 3,477 | 14,607 / 3,291 | 16,895 / 3,641 |
| VDM tokens in / out | — | 8,787 / 292 | 8,894 / 406 | 7,067 / 288 |
| Total tokens | 60,000 | 27,740 | 27,199 | 27,890 |
| Tool calls (sandbox) | — | 35.8 | 36.7 | 36.9 |
| Perception service calls | — | 5.7 | 5.5 | 6.8 |
| Sim steps | 1,500 | 438 | 505 | 526 |
| Wall time (s) | 600 | 76 | 73 | 77 |
| Episodes ending at a budget / horizon cap | — | 9 | 10 | 18 |

Termination reasons:
- baseline: finish 51, token cap 8, turn cap 1;
- ablation: finish 50, horizon 6, token cap 4;
- ours: finish 42, horizon 8, turn cap 6, token cap 4.

Per-task cost tables, including every per-seed outcome and hash, are in `cap-x/logs/test_eval/results.md`.

---

## 3. Diagnostics for hypothesis H

### 3.1 VDM verdict accuracy against the log-only per-turn official check

| Mode | Accuracy, all feedback slots | Accuracy, VDM calls only | False COMPLETE | False NOT COMPLETE | UNCERTAIN |
|---|---|---|---|---|---|
| baseline (free text, heuristic label) | 62.9% (n = 159) | 62.9% | 5 | 16 | 38 |
| ablation (structured, generic checklist) | **91.7%** (n = 145) | **91.7%** | 3 | 8 | 1 |
| ours (structured + execution record + gate) | 69.9% (n = 163) | 86.5% (n = 96) | 5 | 42 | 2 |

- **The structured verdict format is what improves verdict quality** (63% → 92%), not the execution record. On VDM calls, the grounded VDM in `ours` was *less* accurate than the generic one (86.5% vs. 91.7%).
- The baseline figure uses a keyword heuristic on free text. Its 38 UNCERTAIN labels mostly reflect free text without an explicit verdict, so it is not strictly comparable with the other two.
- Including gate slots, `ours` drops to 69.9% because the gate preserved wrong verdicts (36 false NOT COMPLETE on zero-motion turns; §4.2).

### 3.2 Goal-undone and premature FINISH

| Mode | Episodes reaching the goal at some turn | Goal undone (of which final failure) | Episodes with premature FINISH |
|---|---|---|---|
| baseline | 51 | 2 (0): lifting seeds 103, 110 | 5 (all cube_restack) |
| ablation | 47 | 0 | 3 |
| ours | 43 | 0 | 5 (4 cube_restack, 1 cube_stack) |

The Phase 1 failure that motivated H (goal reached, then undone by a re-grasp) was rare on the test set even for baseline: 2/60, both recovered. Premature FINISH was not reduced overall. All 5 premature FINISHes in `ours` (cube_restack 100, 103, 104, 112; cube_stack 103) follow a turn *with* robot motion in which the grounded VDM asserted that the red cube was on top of the green cube.
- In the two checked against the final image (cube_restack 104 and 112), the red cube sits on the table directly behind the green cube: the 2D overlap illusion.
- The other three were not inspected visually.

No premature FINISH in `ours` followed a zero-motion turn (§3.3).

### 3.3 The zero-motion gate in isolation

| Mode | Zero-motion turns | VDM calls / tokens on them | False COMPLETE | False NOT COMPLETE | Premature FINISH after |
|---|---|---|---|---|---|
| baseline | 56 | 56 / 143,270 | 3 | 4 | 3 |
| ablation | 43 | 43 / 123,093 | 2 | 1 | 1 |
| ours | 66 | 0 / 0 | **0** | 36 | **0** |

This is the part of H that holds: the gate removed every false COMPLETE and premature FINISH on no-op turns and saved about 2.2k VDM tokens per episode. It also turned single VDM mistakes into persistent ones, and `ours` had *more* zero-motion turns because of the loop in §4.2.

---

## 4. Paired win / loss breakdown

| Task | A → B | B wins (seeds) | B losses (seeds) | Ties S/S | Ties F/F | On S/S ties: Δturns, Δtokens (B − A); B cheaper |
|---|---|---|---|---|---|---|
| cube_lifting | baseline → ours | 1 [111] | 0 | 19 | 0 | +1.26, +7,036; 7/19 |
| cube_lifting | ablation → ours | 1 [118] | 0 | 19 | 0 | +1.74, +8,465; 8/19 |
| cube_stack | baseline → ours | 0 | 6 [103, 105, 115, 117, 118, 119] | 14 | 0 | −0.64, −5,324; 13/14 |
| cube_stack | ablation → ours | 4 [108, 110, 112, 116] | 5 [103, 105, 115, 118, 119] | 10 | 1 | −0.40, −4,836; 8/10 |
| cube_restack | baseline → ours | 4 [105, 106, 108, 118] | 7 [101, 102, 103, 107, 112, 114, 117] | 5 | 4 | −0.80, −5,689; 2/5 |
| cube_restack | ablation → ours | 4 [106, 109, 116, 119] | 8 [100, 103, 104, 107, 112, 113, 114, 117] | 5 | 3 | +0.40, +3,225; 3/5 |
| **pooled** | baseline → ours | **5** | **13** | — | — | p = 0.096 |
| **pooled** | ablation → ours | 9 | 13 | — | — | p = 0.523 |
| **pooled** | baseline → ablation | 7 | 11 | — | — | p = 0.481 |

**Efficiency.** On seeds where both succeed, `ours` is cheaper on cube_stack (13/14 seeds, −5.3k tokens on average). It is *more* expensive on cube_lifting (+7.0k tokens, +1.3 turns), entirely because of the 6 gate-loop episodes.

### 4.1 Root cause of every failed episode

| Task | Mode | Failures | Premature FINISH | Horizon, saturated moves | Horizon, other | Token cap | Turn cap |
|---|---|---|---|---|---|---|---|
| cube_lifting | baseline | 1 | 0 | 0 | 0 | 0 | 1 |
| cube_lifting | ablation | 1 | 0 | 1 | 0 | 0 | 0 |
| cube_stack | ablation | 5 | 1 | 3 | 0 | 1 | 0 |
| cube_stack | ours | 6 | 1 | 4 | 0 | 1 | 0 |
| cube_restack | baseline | 8 | 5 | 0 | 0 | 3 | 0 |
| cube_restack | ablation | 7 | 2 | 1 | 1 | 3 | 0 |
| cube_restack | ours | 11 | 4 | 2 | 2 | 3 | 0 |

A "saturated" turn is one where every `move_to_joints` hit its 100-step cap: the IK target was unreachable or the gripper orientation was wrong. Two such turns (≈660 steps each) exhaust the 1500-step horizon.

---

## 5. Trajectory case studies (test set)

### 5.1 Helpful: `cube_restack`, seed 106 — `ours` succeeds, baseline and ablation fail

- **baseline** (token cap after 6 turns, 54.3k tokens). The free-text VDM gave ambiguous progress reports:
  > T2: "the robot has successfully grasped the red cube (indicated by the gripper holding it) and moved it to be adjacent to the green cube. However, the red cube is placed next to the …"

  The agent wandered for four more turns.
- **ablation** (horizon after 5 turns). After turn 1 the structured VDM misread the stacking order:
  > T1: "Verdict: NOT COMPLETE — … the current state shows the green cube on top of the red cube (the reverse configuration)."

  A 723-step turn followed.
- **ours** (success in 3 turns, 32.4k tokens):
  ```
  T0  Code execution: FAILED at line 48: SyntaxError: unterminated string literal
      Verdict: NOT COMPLETE — Zero robot motion occurred this turn (...). Do NOT output FINISH; fix the code error and execute the manipulation.   [gate, 0 VDM tokens]
  T1  Gripper opening right after each gripper command: open_gripper -> 0.97 (EEF z +0.034); close_gripper -> -0.00 (EEF z -0.006); open_gripper -> 0.96 (EEF z +0.019)
      Verdict: NOT COMPLETE — ... both cubes are resting separately on the table surface.
  T2  Gripper opening right after each gripper command: open_gripper -> 0.99 (EEF z +0.089); close_gripper -> 0.41 (EEF z -0.000); open_gripper -> 0.97 (EEF z +0.029)
      Verdict: COMPLETE — The red cube is successfully placed on top of the green cube, and the gripper is open.   [turn_eval_success = True]
  ```
  The gate handled the turn-0 crash cheaply. The record exposed the empty turn-1 grasp (`close_gripper -> -0.00`), and every verdict was correct.
- **Honest caveat:** the turn-2 code comments do not mention the gripper reading. The success is *consistent with* the record helping, but the logs cannot show that the agent used it.

### 5.2 Harmful (mechanism): `cube_lifting`, seed 110 — the record misleads the VDM and the gate locks the error in

Turn 0 lifted the cube (`turn_eval_success = True`). The record and the grounded VDM's answer:
```
- End-effector xyz (robot base frame, m): before [+0.447, -0.002, +0.201] -> after [+0.548, -0.058, +0.138] (Δ [+0.101, -0.056, -0.063])
- Gripper opening right after each gripper command: open_gripper -> 0.99 (EEF z +0.214); close_gripper -> 0.44 (EEF z -0.000)
Gripper & Occlusion Check: The gripper is partially closed (opening 0.43), indicating it has grasped the red cube. ...
Verdict: NOT COMPLETE — The robot has successfully grasped the red cube, but the end-effector height (z=0.138) indicates the object has not yet been lifted off the table.
```
Root cause:
- The record has no table reference. Because the turn starts from the home pose (z ≈ 0.20), a correct lift (grasp at z ≈ 0.00, then up to 0.14) shows **Δz = −0.063**.
- The VDM used exactly that number to reject a lift that the gripper reading (0.44 = holding) and the image supported.

What followed:
- Turns 1–7 re-ran `plan_grasp` on the held cube, which raised `AssertionError` with no motion each time.
- The gate re-issued the preserved verdict every turn:
  > "Last assessment after robot motion: Verdict: NOT COMPLETE — … the end-effector height (z=0.138) indicates the object has not yet been lifted off the table. (VDM after turn 0) / Verdict: NOT COMPLETE — Zero robot motion occurred this turn (execution failed at line 13: AssertionError). Do NOT output FINISH; …"

The episode ended at the 8-turn cap (52.9k tokens) and counts as a success only because the cube stayed held. The same pattern occurred in **all 6** lifting episodes where the grounded VDM said NOT COMPLETE after a successful lift (seeds 102, 107, 109, 110, 114, 116; `logs/test_eval/diagnostics.md` §5). In every goal-reached lifting turn the record showed Δz between −0.02 and −0.07.

### 5.3 Regression: `cube_restack`, seed 117 — `ours` fails, baseline and ablation succeed

- **baseline:** turn 0 hit a SyntaxError; turn 1 restacked correctly; FINISH (2 turns, 21.1k tokens).
- **ours** (token cap after 5 turns, 60.7k tokens):
  - After the gated turn-0 crash, turn 1 grasped (`close_gripper -> 0.60`) but the VDM reported the reverse stacking order.
  - Turn 2 contains the VDM's own contradiction of the record:
    ```
    - Gripper opening right after each gripper command: ...; close_gripper -> 1.01 (EEF z -0.009); open_gripper -> 0.99 (EEF z +0.037)
    Verdict: NOT COMPLETE — The red cube is still grasped by the robot and has not been placed on top of the green cube.
    ```
    The gripper had just opened to 0.99, so "still grasped" contradicts the record.
  - Turn 3 moved only the gripper.
  - Turn 4 crashed (`AssertionError`); the gate kept NOT COMPLETE, and the token budget ran out.

Root cause: weaker manipulation by the coding agent in this sample (wrong-cube handling in turn 1), plus at least one VDM statement that contradicts the record it was given. The record did not prevent the error. The higher coding-token use of `ours` (§2.2) also brought the token cap closer.

### 5.4 Not caused by the method: `cube_stack`, seed 118 — turn-0 noise

`ours` used `grasp_quat = np.array([1.0, 0.0, 0.0, 0.0])` (gripper not pointing down). All six moves saturated, using 660 steps per turn, and two turns exhausted the 1500-step horizon. The record and verdicts were correct throughout (`close_gripper -> -0.05`, "Verdict: NOT COMPLETE"). Five more structured-mode cube_stack failures (ablation 112, 116, 117; ours 105, 119) were decided the same way in turn 0, whose input is identical across modes. Excluding them, the mechanism-attributable cube_stack failures are baseline 0, ablation 2, ours 3.

---

## 6. Interpretation, limitations, and what we would change

**What the evidence supports**
1. **Structured verdicts improve VDM accuracy** (63% → 92% vs. the free-text baseline), but more accurate verdicts did **not** translate into higher task success. Ablation is not better than baseline (7 wins vs. 11 losses).
2. **The zero-motion gate prevents false completion on no-op turns** (0 vs. 3 premature FINISH after zero-motion turns) and removes the VDM cost of those turns.
3. **The execution record as designed is harmful on lifting.** Start→end deltas without a reference frame mislead the VLM, and the gate turns a single wrong verdict into a loop. This is a design flaw found only on the test set; per protocol it was not fixed during the run.

**Not supported:** that EG-VDM raises success under a fixed budget. Pooled, `ours` loses 13 seeds and wins 5 against baseline (p = 0.096). The direction is negative, but n = 20 per task with a binary outcome cannot resolve differences of a few seeds. The cube_stack result (p = 0.031, uncorrected for 9 comparisons) is partly explained by turn-0 noise (§5.4).

**Limitations**
- **Sample size:** one coding model, one VDM, one camera view, and 20 seeds per task. Results should not be generalized beyond this setup.
- **Comparability of the baseline verdict label:** it is a keyword heuristic.
- **Metric definitions:** `turn_eval_success` measures the official success check after each turn, which is stricter than "visible progress".
- **Temporal gap:** cube_restack seeds 111–119 ran two days later after a reboot (§1). Within-seed pairing is intact.
- **Shared limitation:** none of the three modes can resolve the 2D "behind looks stacked" illusion on cube_restack. That needs depth-based geometry, which the record does not contain.

**Changes we would test next (not run; they would need a new freeze and new test seeds)**
- Report heights relative to the grasp height (e.g. "object lifted 0.14 m above the height where `close_gripper` held it") instead of start→end Δz.
- Let a gate-preserved NOT COMPLETE expire after one repeat so that the VDM is called again.
- Include the agent's own perception printouts (object centers) as a geometric check on stacking claims.
