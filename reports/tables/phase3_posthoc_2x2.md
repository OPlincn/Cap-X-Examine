### Turn-0-matched (frozen turn-0 scene description and code replayed; gate_only also replays the frozen turn-0 VDM verdict, whose input is unchanged)

| Condition | Reference frame | Gate | Turn-0 VDM false negative | Avg turns | Avg total tokens | Avg VDM calls (gated) | Code exec success | 8-turn cap hits | Final success | Avg turns after goal reached |
|---|---|---|---|---|---|---|---|---|---|---|
| ours (frozen, `phase2-freeze`) | original | sticky | 6 / 6 | 8.0 | 54,649 | 2.0 (6.0) | 12.5% | 6 / 6 | 6 / 6 | 7.0 |
| ref_only (grasp-relative Δz, sticky gate) | grasp-relative | sticky | 0 / 5 | 1.2 | 12,703 | 2.2 (0.0) | 100.0% | 0 / 6 | 6 / 6 | 0.0 |
| gate_only (original Δz, gate expiry) | original | expiry | 5 / 5 | 4.5 | 36,527 | 2.7 (2.7) | 40.7% | 1 / 6 | 6 / 6 | 3.3 |
| ref_and_gate (both fixes) | grasp-relative | expiry | 0 / 6 | 1.0 | 11,430 | 2.0 (0.0) | 100.0% | 0 / 6 | 6 / 6 | 0.0 |

Per seed (turns, termination, turn-0 verdict):

| Seed | ours | ref_only | gate_only | ref_and_gate |
|---|---|---|---|---|
| 102 | S 8 max_turns / T0 NOT COMPLETE | S 2 finish / T0 NOT COMPLETE | S 5 finish / T0 NOT COMPLETE | S 1 finish / T0 COMPLETE |
| 107 | S 8 max_turns / T0 NOT COMPLETE | S 1 finish / T0 COMPLETE | S 2 finish / T0 NOT COMPLETE | S 1 finish / T0 COMPLETE |
| 109 | S 8 max_turns / T0 NOT COMPLETE | S 1 finish / T0 COMPLETE | S 3 finish / T0 NOT COMPLETE | S 1 finish / T0 COMPLETE |
| 110 | S 8 max_turns / T0 NOT COMPLETE | S 1 finish / T0 COMPLETE | S 8 max_turns / T0 NOT COMPLETE | S 1 finish / T0 COMPLETE |
| 114 | S 8 max_turns / T0 NOT COMPLETE | S 1 finish / T0 COMPLETE | S 2 finish / T0 NOT COMPLETE | S 1 finish / T0 COMPLETE |
| 116 | S 8 max_turns / T0 NOT COMPLETE | S 1 finish / T0 COMPLETE | S 7 max_total_tokens / T0 NOT COMPLETE | S 1 finish / T0 COMPLETE |

### Raw (every call sampled live)

| Condition | Reference frame | Gate | Turn-0 VDM false negative | Avg turns | Avg total tokens | Avg VDM calls (gated) | Code exec success | 8-turn cap hits | Final success | Avg turns after goal reached |
|---|---|---|---|---|---|---|---|---|---|---|
| ours (frozen, `phase2-freeze`) | original | sticky | 6 / 6 | 8.0 | 54,649 | 2.0 (6.0) | 12.5% | 6 / 6 | 6 / 6 | 7.0 |
| ref_only (grasp-relative Δz, sticky gate) | grasp-relative | sticky | 0 / 6 | 1.0 | 10,863 | 2.0 (0.0) | 100.0% | 0 / 6 | 6 / 6 | 0.0 |
| gate_only (original Δz, gate expiry) | original | expiry | 2 / 6 | 2.8 | 23,664 | 2.0 (1.7) | 41.2% | 1 / 6 | 6 / 6 | 1.8 |
| ref_and_gate (both fixes) | grasp-relative | expiry | 0 / 5 | 1.2 | 12,903 | 2.0 (0.2) | 100.0% | 0 / 6 | 6 / 6 | 0.0 |

Per seed (turns, termination, turn-0 verdict):

| Seed | ours | ref_only | gate_only | ref_and_gate |
|---|---|---|---|---|
| 102 | S 8 max_turns / T0 NOT COMPLETE | S 1 finish / T0 COMPLETE | S 5 finish / T0 NOT COMPLETE | S 1 finish / T0 COMPLETE |
| 107 | S 8 max_turns / T0 NOT COMPLETE | S 1 finish / T0 COMPLETE | S 1 finish / T0 COMPLETE | S 2 finish / T0 NOT COMPLETE |
| 109 | S 8 max_turns / T0 NOT COMPLETE | S 1 finish / T0 COMPLETE | S 1 finish / T0 COMPLETE | S 1 finish / T0 COMPLETE |
| 110 | S 8 max_turns / T0 NOT COMPLETE | S 1 finish / T0 COMPLETE | S 1 finish / T0 COMPLETE | S 1 finish / T0 COMPLETE |
| 114 | S 8 max_turns / T0 NOT COMPLETE | S 1 finish / T0 COMPLETE | S 8 max_turns / T0 NOT COMPLETE | S 1 finish / T0 COMPLETE |
| 116 | S 8 max_turns / T0 NOT COMPLETE | S 1 finish / T0 COMPLETE | S 1 finish / T0 COMPLETE | S 1 finish / T0 COMPLETE |

### Generalization: the other 14 cube_lifting seeds (raw, live sampling)

| Condition | Episodes | Success | Avg turns | Avg total tokens | Avg VDM calls | 8-turn cap hits |
|---|---|---|---|---|---|---|
| ours (frozen, `phase2-freeze`) | 14 | 14 / 14 | 1.1 | 12,057 | 2.0 | 0 / 14 |
| ref_and_gate (both fixes) | 14 | 14 / 14 | 1.4 | 14,879 | 2.1 | 0 / 14 |

