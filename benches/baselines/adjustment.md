# Adjustment search benchmark baseline (Phase 1)

Workload: `backdoor_minimal_n8_cov` — T→Y with 8 common causes; enumerate
minimal backdoor sets.

Regression budget: 20%.

```bash
cargo +1.85 bench -p causal-identify --bench adjustment
```
