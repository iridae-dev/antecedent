# d-separation benchmark baseline

Workloads:

- `dsep_sparse_chain_200` — chain of 200 nodes, condition on middle
- `dsep_dense_n80` — denser DAG (edges to next 5 nodes), condition on two nodes

Established: 2026-07-21
Machine class: Apple M1 Max (arm64), 64 GB
Criterion sample size: 40
Regression budget: 20% wall-time on the same machine class.

## Accepted measurements

| Workload | Mean | CI | Gate (≤ +20%) |
|----------|------|----|----------------|
| `dsep_sparse_chain_200` | **2.69 µs** | 2.63–2.77 µs | **3.23 µs** |
| `dsep_dense_n80` | **8.30 µs** | 8.28–8.31 µs | **9.96 µs** |

```bash
cargo +1.85 bench -p antecedent-graph --bench dseparation
```
