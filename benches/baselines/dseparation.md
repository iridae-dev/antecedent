# d-separation benchmark baseline (Phase 1)

Workloads:

- `dsep_sparse_chain_200` — chain of 200 nodes, condition on middle
- `dsep_dense_n80` — denser DAG (edges to next 5 nodes), condition on two nodes

Regression budget: 20% wall-time on the same machine class.

```bash
cargo +1.85 bench -p causal-graph --bench dseparation
```
