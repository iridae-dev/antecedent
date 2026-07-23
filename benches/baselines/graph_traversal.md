# Graph traversal benchmark baseline

Workload: `dag_reach_chain_5k` — directed reachability on a 5,000-node chain
using a reusable [`GraphWorkspace`].

Established: 2026-07-21
Machine class: Apple M1 Max (arm64), 64 GB
Criterion sample size: 40
Regression budget: 20% wall-time on the same machine class.

## Accepted measurement

| Metric | Value |
|--------|-------|
| mean wall time | **26.26 µs** |
| CI (lower / upper) | 26.23 µs / 26.29 µs |

Gate: mean ≤ **31.51 µs** (20% over 26.26 µs).

```bash
cargo +1.85 bench -p causal-graph --bench traversal
```
