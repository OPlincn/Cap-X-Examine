# Phase 3 — Post-Hoc Root-Cause Verification (not part of the frozen evaluation)

**Status.** The frozen primary result is `logs/test_eval/` under tag `phase2-freeze` (`ed29cc7`) and is
unchanged. Everything below was run *after* the frozen evaluation, on code tagged `phase3-final`. The runs
live in `logs/posthoc_diagnostic/`. They diagnose root causes; they are not a test of an improved method,
because the variants were designed after seeing the test trajectories.

## What was tested

The frozen `ours` issued a false `NOT COMPLETE` after a successful lift on 6 of 20 `cube_lifting` seeds:
102, 107, 109, 110, 114 and 116. Each of these episodes then ran to the 8-turn cap. Two suspected causes
were crossed in a 2×2 design:

| Variant | Δz reference frame in the execution record | Zero-motion gate |
|---|---|---|
| `ours` (frozen) | turn start → turn end (starts at the high home pose, so a lift shows Δz < 0) | sticky: repeats the cached verdict |
| `ref_only` | relative to the last `close_gripper` (`Δz_from_grasp`) | sticky |
| `gate_only` | turn start → turn end | expiry + state-aware diagnostic |
| `ref_and_gate` | relative to the last `close_gripper` | expiry + state-aware diagnostic |

Gate expiry works as follows. A cached `NOT COMPLETE` is not repeated: it expires immediately when the
gripper is closed around an object (0.05 < opening < 0.85), otherwise after one repeat. The agent instead
receives `Verdict: UNCERTAIN` with the current gripper state and a request to verify the goal before
re-grasping. The variants are implemented in `capx/envs/eg_vdm.py`
(`format_exec_header(..., grasp_relative=True)`, `gate_message_with_expiry`) and enabled only by
`--posthoc-variant`. Models, temperatures and budgets are unchanged.

## Two regimes

- **Turn-0 matched** (`logs/posthoc_diagnostic/matched/`). The frozen episode's turn-0 scene description
  and initial code are replayed, so every variant starts from the same first program. For `gate_only`,
  whose VDM input is unchanged, the frozen turn-0 verdict is also replayed. Replayed calls are charged
  their logged tokens.
  - Physical fidelity: 16 of 18 replayed turn-0 programs lifted the cube again. In the other two the grasp
    missed, because perception is not bit-deterministic; the false-negative denominators exclude them.
- **Raw** (`logs/posthoc_diagnostic/raw/`). Every call is sampled live.

## Results

```bash
python3 scripts/posthoc_analysis.py --markdown reports/tables/phase3_posthoc_2x2.md
```

| Regime | Condition | T0 false negative | Avg turns | Avg tokens | 8-turn cap | Success |
|---|---|---|---|---|---|---|
| — | ours (frozen) | 6/6 | 8.0 | 54.6k | 6/6 | 6/6 |
| matched | ref_only | 0/5 | 1.2 | 12.7k | 0/6 | 6/6 |
| matched | gate_only | 5/5 (verdict replayed) | 4.5 | 36.5k | 1/6 | 6/6 |
| matched | ref_and_gate | 0/6 | 1.0 | 11.4k | 0/6 | 6/6 |
| raw | ref_only | 0/6 | 1.0 | 10.9k | 0/6 | 6/6 |
| raw | gate_only | 2/6 | 2.8 | 23.7k | 1/6 | 6/6 |
| raw | ref_and_gate | 0/5 | 1.2 | 12.9k | 0/6 | 6/6 |

## Interpretation

1. **The reference frame is the primary cause.** With the grasp-relative record there were 0/11 turn-0
   false negatives in raw sampling, against 2/6 with the original record under the same resampling.
2. **Gate expiry alone mitigates but does not cure.** It breaks most lock-ins (cap hits 1/6 vs. 6/6). On
   seeds 110 (matched) and 114 (raw), the agent kept re-planning grasps despite the expiry diagnostic.
3. **The frozen 6/6 overstates the per-call error rate.** These seeds were selected because the VDM erred
   once (a selection effect); resampling with the original record gives 2/6.
4. **All conditions succeed 6/6.** The cube stays held, so the fixes recover efficiency (about −79% tokens),
   not success.

## Generalization check

`ref_and_gate` (raw) was also run on the other 14 `cube_lifting` seeds (100, 101, 103–106, 108, 111–113, 115,
117–119):

- **Success:** 14/14, the same as the frozen `ours` (14/14).
- **Turn cap and verdicts:** no 8-turn cap hits, and no false verdicts in 20 verdicts.
- **Cost:** 1.4 turns and 14.9k tokens on average, against 1.1 and 12.1k for frozen `ours`. The difference
  comes entirely from five episodes whose first program did not lift the cube (a zero-motion crash or a
  missed grasp). Each was correctly judged `NOT COMPLETE` and recovered in the next turn.
