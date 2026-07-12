# Graph traversal benchmark baseline (Phase 0)

Workload: `dag_reach_chain_5k` — directed reachability on a 5,000-node chain
using a reusable [`GraphWorkspace`].

Established: 2026-07-21
Regression budget: 20% wall-time on the same machine class.

```bash
cargo +1.85 bench -p causal-graph --bench traversal
```
