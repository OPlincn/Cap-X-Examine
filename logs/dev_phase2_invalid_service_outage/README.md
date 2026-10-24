Dev episodes invalidated by an infrastructure fault (not a method outcome).

The PyRoKi IK server was launched without `--log-dir`, so its stdout went to an undrained pipe. After
about 35 minutes of request logging the pipe buffer filled, the server blocked on a write, and every
`/ik` request timed out (`Read timed out (read timeout=15.0)`). Affected: cube_lifting ablation seeds 0
and 1 (all turns). Fix: services are now launched with `--log-dir logs/services/run_*` (applies to all
modes), and `scripts/run_paired_eval.sh` checks service health before every episode.
