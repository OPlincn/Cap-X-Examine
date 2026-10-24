## Main results

| Task | Mode | Success | Code exec | Avg turns | Coding tok in/out | VDM tok in/out | Total tok | Model calls | VDM calls (gate-skipped %) | Tool calls | Sim steps | Wall (s) | Budget/timeout |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cube_lifting | ablation | 3/3 (100%) | 75.0% | 1.3 | 7,645/1,753 | 5,905/228 | 15,532 | 4.7 | 2.3 (0%) | 11.0 | 187 | 44 | 0 |
| cube_lifting | ours | 3/3 (100%) | 36.4% | 3.7 | 21,453/2,655 | 5,275/221 | 29,604 | 6.3 | 2.0 (54%) | 18.7 | 178 | 83 | 1 |
| cube_restack | ablation | 3/3 (100%) | 75.0% | 4.0 | 27,044/6,056 | 13,274/746 | 47,120 | 10.0 | 5.0 (0%) | 71.3 | 899 | 115 | 0 |
| cube_restack | ours | 2/3 (67%) | 66.7% | 3.0 | 19,115/4,814 | 8,287/374 | 32,590 | 7.0 | 3.0 (25%) | 41.3 | 533 | 77 | 0 |
| ALL | ablation | 6/6 (100%) | 75.0% | 2.7 | 17,344/3,905 | 9,590/487 | 31,326 | 7.3 | 3.7 (0%) | 41.2 | 543 | 80 | 0 |
| ALL | ours | 5/6 (83%) | 50.0% | 3.3 | 20,284/3,735 | 6,781/298 | 31,097 | 6.7 | 2.5 (40%) | 30.0 | 356 | 80 | 1 |

Termination reasons: ALL/ablation: {'finish': 6}; ALL/ours: {'finish': 5, 'max_turns': 1}

## Diagnostics for hypothesis H

| Task | Mode | VDM verdict acc. (all slots) | false COMPLETE | false NOT COMPLETE | UNCERTAIN | verdict acc. (VDM calls only) | Episodes goal reached | Goal undone (of which final fail) | Episodes with premature FINISH |
|---|---|---|---|---|---|---|---|---|---|
| cube_lifting | ablation | 100.0% (n=4) | 0 | 0 | 0 | 100.0% (n=4) | 3 | 0 (0) | 0 |
| cube_lifting | ours | 30.0% (n=10) | 0 | 7 | 0 | 66.7% (n=3) | 3 | 0 (0) | 0 |
| cube_restack | ablation | 75.0% (n=12) | 1 | 1 | 1 | 75.0% (n=12) | 3 | 0 (0) | 0 |
| cube_restack | ours | 88.9% (n=9) | 1 | 0 | 0 | 83.3% (n=6) | 2 | 0 (0) | 1 |
| ALL | ablation | 81.2% (n=16) | 1 | 1 | 1 | 81.2% (n=16) | 6 | 0 (0) | 0 |
| ALL | ours | 57.9% (n=19) | 1 | 7 | 0 | 77.8% (n=9) | 5 | 0 (0) | 1 |

## Paired comparison (same seed, same initial state)

| Task | A → B | B wins (A fail, B success) | B losses (A success, B fail) | Ties S/S | Ties F/F | On S/S ties: Δturns (B−A) | Δtokens (B−A) | B cheaper |
|---|---|---|---|---|---|---|---|---|
| cube_lifting | ablation → ours | 0 [] | 0 [] | 3 | 0 | 2.33 | 14,072 | 1/3 |
| cube_restack | ablation → ours | 0 [] | 1 [2] | 2 | 0 | -1.50 | -21,232 | 2/2 |

## Per-seed outcomes (S = official success, number = code executions)


**cube_lifting**

| Seed | ablation | ours | initial-state hash |
|---|---|---|---|
| 0 | S 2 | S 2 | `ea9ed3c0eb08e50d` |
| 1 | S 1 | S 8 (B) | `d26adbe231020f0a` |
| 2 | S 1 | S 1 | `40c6f9e9c8eba23e` |

**cube_restack**

| Seed | ablation | ours | initial-state hash |
|---|---|---|---|
| 0 | S 4 | S 3 | `c232c2ef19039037` |
| 1 | S 5 | S 3 | `78d82e7d47831db1` |
| 2 | S 3 | F 3 | `d7870644e5b4eadc` |
