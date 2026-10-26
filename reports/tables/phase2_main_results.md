## Main results

| Task | Mode | Success | Code exec | Avg turns | Coding tok in/out | VDM tok in/out | Total tok | Model calls | VDM calls (gate-skipped %) | Tool calls | Sim steps | Wall (s) | Budget/timeout |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cube_lifting | baseline | 19/20 (95%) | 69.6% | 2.3 | 10,381/1,817 | 7,814/230 | 20,242 | 6.5 | 3.2 (0%) | 16.6 | 214 | 61 | 2 |
| cube_lifting | ablation | 19/20 (95%) | 73.5% | 1.7 | 9,558/1,279 | 6,786/299 | 17,923 | 5.3 | 2.6 (0%) | 14.7 | 238 | 51 | 1 |
| cube_lifting | ours | 20/20 (100%) | 33.3% | 3.1 | 16,911/2,497 | 5,218/209 | 24,835 | 5.8 | 2.0 (48%) | 17.4 | 162 | 80 | 6 |
| cube_restack | baseline | 12/20 (60%) | 71.8% | 3.5 | 23,002/5,476 | 10,766/398 | 39,642 | 8.7 | 4.5 (0%) | 61.2 | 751 | 98 | 6 |
| cube_restack | ablation | 13/20 (65%) | 66.2% | 3.5 | 21,793/5,642 | 11,472/546 | 39,453 | 8.7 | 4.3 (0%) | 62.2 | 750 | 100 | 5 |
| cube_restack | ours | 9/20 (45%) | 60.6% | 3.5 | 22,579/5,758 | 9,127/377 | 37,840 | 7.5 | 3.2 (25%) | 60.4 | 842 | 91 | 7 |
| cube_stack | baseline | 20/20 (100%) | 65.2% | 2.3 | 12,168/3,140 | 7,782/246 | 23,336 | 6.5 | 3.2 (0%) | 29.5 | 348 | 68 | 1 |
| cube_stack | ablation | 15/20 (75%) | 66.7% | 2.4 | 12,469/2,954 | 8,423/373 | 24,220 | 6.5 | 3.2 (0%) | 33.1 | 527 | 69 | 4 |
| cube_stack | ours | 14/20 (70%) | 72.1% | 2.1 | 11,195/2,667 | 6,857/278 | 20,997 | 5.5 | 2.5 (14%) | 32.8 | 575 | 60 | 5 |
| ALL | baseline | 51/60 (85%) | 69.3% | 2.7 | 15,184/3,477 | 8,787/292 | 27,740 | 7.2 | 3.6 (0%) | 35.8 | 438 | 76 | 9 |
| ALL | ablation | 47/60 (78%) | 68.0% | 2.5 | 14,607/3,291 | 8,894/406 | 27,199 | 6.8 | 3.4 (0%) | 36.7 | 505 | 73 | 10 |
| ALL | ours | 43/60 (72%) | 53.7% | 3.0 | 16,895/3,641 | 7,067/288 | 27,890 | 6.3 | 2.6 (30%) | 36.9 | 526 | 77 | 18 |

Termination reasons: ALL/baseline: {'finish': 51, 'max_total_tokens': 8, 'max_turns': 1}; ALL/ablation: {'env_horizon': 6, 'finish': 50, 'max_total_tokens': 4}; ALL/ours: {'env_horizon': 8, 'finish': 42, 'max_total_tokens': 4, 'max_turns': 6}

## Diagnostics for hypothesis H

| Task | Mode | VDM verdict acc. (all slots) | false COMPLETE | false NOT COMPLETE | UNCERTAIN | verdict acc. (VDM calls only) | Episodes goal reached | Goal undone (of which final fail) | Episodes with premature FINISH |
|---|---|---|---|---|---|---|---|---|---|
| cube_lifting | baseline | 31.1% (n=45) | 0 | 12 | 19 | 31.1% (n=45) | 19 | 2 (0) | 0 |
| cube_lifting | ablation | 81.8% (n=33) | 0 | 6 | 0 | 81.8% (n=33) | 19 | 0 (0) | 0 |
| cube_lifting | ours | 26.3% (n=57) | 0 | 42 | 0 | 70.0% (n=20) | 20 | 0 (0) | 0 |
| cube_restack | baseline | 69.6% (n=69) | 5 | 3 | 13 | 69.6% (n=69) | 12 | 0 (0) | 5 |
| cube_restack | ablation | 95.5% (n=67) | 3 | 0 | 0 | 95.5% (n=67) | 13 | 0 (0) | 2 |
| cube_restack | ours | 94.0% (n=67) | 4 | 0 | 0 | 91.1% (n=45) | 9 | 0 (0) | 4 |
| cube_stack | baseline | 84.4% (n=45) | 0 | 1 | 6 | 84.4% (n=45) | 20 | 0 (0) | 0 |
| cube_stack | ablation | 93.3% (n=45) | 0 | 2 | 1 | 93.3% (n=45) | 15 | 0 (0) | 1 |
| cube_stack | ours | 92.3% (n=39) | 1 | 0 | 2 | 90.3% (n=31) | 14 | 0 (0) | 1 |
| ALL | baseline | 62.9% (n=159) | 5 | 16 | 38 | 62.9% (n=159) | 51 | 2 (0) | 5 |
| ALL | ablation | 91.7% (n=145) | 3 | 8 | 1 | 91.7% (n=145) | 47 | 0 (0) | 3 |
| ALL | ours | 69.9% (n=163) | 5 | 42 | 2 | 86.5% (n=96) | 43 | 0 (0) | 5 |

## Paired comparison (same seed, same initial state)

| Task | A → B | B wins (A fail, B success) | B losses (A success, B fail) | Ties S/S | Ties F/F | On S/S ties: Δturns (B−A) | Δtokens (B−A) | B cheaper |
|---|---|---|---|---|---|---|---|---|
| cube_lifting | baseline → ours | 1 [111] | 0 [] | 19 | 0 | 1.26 | 7,036 | 7/19 |
| cube_lifting | ablation → ours | 1 [118] | 0 [] | 19 | 0 | 1.74 | 8,465 | 8/19 |
| cube_lifting | baseline → ablation | 1 [111] | 1 [118] | 18 | 0 | -0.61 | -2,440 | 6/18 |
| cube_restack | baseline → ours | 4 [105, 106, 108, 118] | 7 [101, 102, 103, 107, 112, 114, 117] | 5 | 4 | -0.80 | -5,689 | 2/5 |
| cube_restack | ablation → ours | 4 [106, 109, 116, 119] | 8 [100, 103, 104, 107, 112, 113, 114, 117] | 5 | 3 | 0.40 | 3,225 | 3/5 |
| cube_restack | baseline → ablation | 6 [100, 104, 105, 108, 113, 118] | 5 [101, 102, 109, 116, 119] | 7 | 2 | -0.43 | -1,451 | 3/7 |
| cube_stack | baseline → ours | 0 [] | 6 [103, 105, 115, 117, 118, 119] | 14 | 0 | -0.64 | -5,324 | 13/14 |
| cube_stack | ablation → ours | 4 [108, 110, 112, 116] | 5 [103, 105, 115, 118, 119] | 10 | 1 | -0.40 | -4,836 | 8/10 |
| cube_stack | baseline → ablation | 0 [] | 5 [108, 110, 112, 116, 117] | 15 | 0 | -0.53 | -4,763 | 8/15 |

## Per-seed outcomes (S = official success, number = code executions)


**cube_lifting**

| Seed | baseline | ablation | ours | initial-state hash |
|---|---|---|---|---|
| 100 | S 2 | S 2 | S 1 | `8462bc0e96c43fbf` |
| 101 | S 2 | S 1 | S 1 | `8c840bccc9f97d75` |
| 102 | S 1 | S 1 | S 8 (B) | `9a0a2d7c6df94f88` |
| 103 | S 4 | S 2 | S 1 | `ea9d421188f4c6c5` |
| 104 | S 1 | S 1 | S 1 | `18e25c2b93fdbb3e` |
| 105 | S 1 | S 1 | S 2 | `47279f69d360ce4f` |
| 106 | S 5 | S 2 | S 1 | `70456b41391a4460` |
| 107 | S 6 (B) | S 2 | S 8 (B) | `c1aebb64488b6ef8` |
| 108 | S 1 | S 1 | S 1 | `b575188243d82aae` |
| 109 | S 1 | S 1 | S 8 (B) | `e053a0dfaaed4b98` |
| 110 | S 4 | S 1 | S 8 (B) | `9be087b7cb185384` |
| 111 | F 8 (B) | S 3 | S 1 | `cba8c91221a8f42b` |
| 112 | S 1 | S 2 | S 1 | `f6bd0a1372a48c56` |
| 113 | S 1 | S 2 | S 1 | `be12931eb790e0b2` |
| 114 | S 1 | S 1 | S 8 (B) | `6c8c7644590bf400` |
| 115 | S 1 | S 1 | S 1 | `62460e271318a803` |
| 116 | S 1 | S 1 | S 8 (B) | `c6a4ca671abbba01` |
| 117 | S 2 | S 2 | S 1 | `89e1112181e1c29d` |
| 118 | S 1 | F 5 (B) | S 1 | `8d647581f63bc400` |
| 119 | S 2 | S 2 | S 1 | `3629a9ab4d076fdc` |

**cube_restack**

| Seed | baseline | ablation | ours | initial-state hash |
|---|---|---|---|---|
| 100 | F 1 | S 4 | F 3 | `6031b92240d7cd43` |
| 101 | S 4 (B) | F 6 (B) | F 5 (B) | `08c6817505bda575` |
| 102 | S 3 | F 5 (B) | F 5 (B) | `da28394fa830e3cd` |
| 103 | S 6 (B) | S 4 | F 3 | `920b2fbb0831fe87` |
| 104 | F 5 (B) | S 3 | F 2 | `cb58914096c4c08f` |
| 105 | F 1 | S 2 | S 4 | `1108bb085df570ad` |
| 106 | F 6 (B) | F 5 (B) | S 3 | `43d25ade633559f1` |
| 107 | S 3 | S 3 | F 3 (B) | `370ea3f63541bb9e` |
| 108 | F 3 | S 3 | S 2 | `4905de14e81f2e68` |
| 109 | S 4 | F 5 (B) | S 4 | `32d56a94c6080aac` |
| 110 | S 3 | S 3 | S 3 | `52e113bce15bee7e` |
| 111 | F 4 | F 2 | F 6 (B) | `bc61a47ca4ca3c56` |
| 112 | S 2 | S 2 | F 2 | `9408da73fae96413` |
| 113 | F 1 | S 4 | F 3 (B) | `1e1cbedbc5243a0d` |
| 114 | S 4 | S 3 | F 5 (B) | `f3112c79a95092fc` |
| 115 | S 4 | S 3 | S 4 | `559d65e9b2aa612a` |
| 116 | S 4 | F 6 (B) | S 3 | `4f0c070551d8be83` |
| 117 | S 2 | S 3 | F 5 (B) | `6da20316bc56e908` |
| 118 | F 5 (B) | S 3 | S 3 | `a489b6d1a2dd1063` |
| 119 | S 6 (B) | F 2 | S 3 | `60711864bd97fe00` |

**cube_stack**

| Seed | baseline | ablation | ours | initial-state hash |
|---|---|---|---|---|
| 100 | S 2 | S 2 | S 2 | `52ed43cabf737da4` |
| 101 | S 2 | S 2 | S 2 | `d3e59081494515e0` |
| 102 | S 2 | S 2 | S 2 | `1bbc72f7579b3026` |
| 103 | S 2 | S 1 | F 2 | `19c2b84edb3d7b00` |
| 104 | S 3 | S 2 | S 1 | `5e90a3e5c5210684` |
| 105 | S 2 | S 2 | F 3 (B) | `d7115e8a31943e40` |
| 106 | S 2 | S 4 | S 2 | `dc39f33dbbbd5dee` |
| 107 | S 2 | S 1 | S 1 | `8491f9f846cc35a0` |
| 108 | S 2 | F 5 (B) | S 5 | `0a300622f4c9bb9c` |
| 109 | S 2 | S 1 | S 1 | `745c74a112778e5a` |
| 110 | S 3 | F 5 | S 1 | `c992b4e0438d2c53` |
| 111 | S 3 | S 1 | S 2 | `94b7e6df5b65ce13` |
| 112 | S 2 | F 3 (B) | S 1 | `e48da7eee0305d5e` |
| 113 | S 3 | S 2 | S 1 | `168024e7e9d47e80` |
| 114 | S 2 | S 2 | S 1 | `88103cc209f13e6d` |
| 115 | S 2 | S 2 | F 4 (B) | `ab3d7b2990276d19` |
| 116 | S 2 | F 4 (B) | S 1 | `fe430cb9428de71a` |
| 117 | S 1 | F 3 (B) | F 5 (B) | `da5d276fe82dc053` |
| 118 | S 6 (B) | S 2 | F 3 (B) | `8e36851d04f49fbf` |
| 119 | S 1 | S 2 | F 3 (B) | `34449d698f2480e2` |
