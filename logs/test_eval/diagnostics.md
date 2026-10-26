## 1. Paired sign tests (exact, two-sided)

| Task | A → B | B wins | B losses | p |
|---|---|---|---|---|
| cube_lifting | baseline → ours | 1 | 0 | 1.000 |
| cube_lifting | ablation → ours | 1 | 0 | 1.000 |
| cube_lifting | baseline → ablation | 1 | 1 | 1.000 |
| cube_restack | baseline → ours | 4 | 7 | 0.549 |
| cube_restack | ablation → ours | 4 | 8 | 0.388 |
| cube_restack | baseline → ablation | 6 | 5 | 1.000 |
| cube_stack | baseline → ours | 0 | 6 | 0.031 |
| cube_stack | ablation → ours | 4 | 5 | 1.000 |
| cube_stack | baseline → ablation | 0 | 5 | 0.062 |
| ALL | baseline → ours | 5 | 13 | 0.096 |
| ALL | ablation → ours | 9 | 13 | 0.523 |
| ALL | baseline → ablation | 7 | 11 | 0.481 |

## 2. Root cause of failed episodes

| Task | Mode | Failures | Premature FINISH | Sim-step horizon (saturated moves) | Sim-step horizon (other) | Token cap | Turn cap |
|---|---|---|---|---|---|---|---|
| cube_lifting | baseline | 1 | 0 | 0 | 0 | 0 | 1 |
| cube_lifting | ablation | 1 | 0 | 1 | 0 | 0 | 0 |
| cube_lifting | ours | 0 | 0 | 0 | 0 | 0 | 0 |
| cube_restack | baseline | 8 | 5 | 0 | 0 | 3 | 0 |
| cube_restack | ablation | 7 | 2 | 1 | 1 | 3 | 0 |
| cube_restack | ours | 11 | 4 | 2 | 2 | 3 | 0 |
| cube_stack | baseline | 0 | 0 | 0 | 0 | 0 | 0 |
| cube_stack | ablation | 5 | 1 | 3 | 0 | 1 | 0 |
| cube_stack | ours | 6 | 1 | 4 | 0 | 1 | 0 |

## 3. Turn-0 programs whose every move saturated (identical turn-0 prompt in all modes)

| Mode | Episodes with saturated turn 0 | of which failed | per task |
|---|---|---|---|
| baseline | 1 / 60 | 1 | {'cube_lifting': 1} |
| ablation | 4 / 60 | 4 | {'cube_lifting': 1, 'cube_stack': 3} |
| ours | 3 / 60 | 3 | {'cube_stack': 3} |

Fisher exact (baseline vs ablation+ours): p = 0.272

## 4. Zero-motion turns (no motion primitive ran) that received feedback

| Mode | Turns | VDM calls | VDM tokens | False COMPLETE | False NOT COMPLETE | FINISH after | Premature FINISH after |
|---|---|---|---|---|---|---|---|
| baseline | 56 | 56 | 143,270 | 3 | 4 | 6 | 3 |
| ablation | 43 | 43 | 123,093 | 2 | 1 | 6 | 1 |
| ours | 66 | 0 | 0 | 0 | 36 | 0 | 0 |

## 5. cube_lifting / ours: structured VDM verdict on goal-reached turns vs. record Δz

| Seed | Turn | Verdict | Δz (after − before) | EEF z after | Gripper after close | Episode turns | Termination |
|---|---|---|---|---|---|---|---|
| 100 | 0 | COMPLETE | -0.054 | +0.143 | [0.57] | 1 | finish |
| 101 | 0 | COMPLETE | -0.059 | +0.138 | [0.57] | 1 | finish |
| 102 | 0 | NOT COMPLETE | -0.040 | +0.145 | [0.79] | 8 | max_turns |
| 103 | 0 | COMPLETE | -0.047 | +0.119 | [0.54] | 1 | finish |
| 104 | 0 | COMPLETE | -0.050 | +0.145 | [0.34] | 1 | finish |
| 105 | 1 | COMPLETE | -0.051 | +0.138 | [0.44] | 2 | finish |
| 106 | 0 | COMPLETE | -0.056 | +0.130 | [0.45] | 1 | finish |
| 107 | 0 | NOT COMPLETE | -0.052 | +0.140 | [0.54] | 8 | max_turns |
| 108 | 0 | COMPLETE | -0.022 | +0.146 | [0.71] | 1 | finish |
| 109 | 0 | NOT COMPLETE | -0.041 | +0.134 | [0.64] | 8 | max_turns |
| 110 | 0 | NOT COMPLETE | -0.063 | +0.138 | [0.44] | 8 | max_turns |
| 111 | 0 | COMPLETE | -0.040 | +0.142 | [0.48] | 1 | finish |
| 112 | 0 | COMPLETE | -0.035 | +0.150 | [0.4] | 1 | finish |
| 113 | 0 | COMPLETE | -0.040 | +0.148 | [0.51] | 1 | finish |
| 114 | 0 | NOT COMPLETE | -0.035 | +0.135 | [0.49] | 8 | max_turns |
| 115 | 0 | COMPLETE | -0.036 | +0.145 | [0.4] | 1 | finish |
| 116 | 0 | NOT COMPLETE | -0.042 | +0.143 | [0.71] | 8 | max_turns |
| 117 | 0 | COMPLETE | -0.041 | +0.146 | [0.54] | 1 | finish |
| 118 | 0 | COMPLETE | -0.034 | +0.144 | [0.52] | 1 | finish |
| 119 | 0 | COMPLETE | -0.073 | +0.115 | [0.44] | 1 | finish |

## 6. Code-execution failures by exception type

| Mode | Turns | Failed executions | Top exception types |
|---|---|---|---|
| baseline | 163 | 50 | SyntaxError 38, AssertionError 12 |
| ablation | 153 | 49 | SyntaxError 36, ValueError 7, AssertionError 6 |
| ours | 177 | 82 | AssertionError 45, SyntaxError 29, ValueError 8 |
