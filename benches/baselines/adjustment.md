# Adjustment search benchmark baseline

Workload: `backdoor_minimal_n8_cov` — T→Y with 8 common causes; enumerate
minimal backdoor sets.

Established: 2026-07-21
Machine class: Apple M1 Max (arm64), 64 GB
Criterion sample size: 40
Regression budget: 20%.

## Accepted measurement

| Metric | Value |
|--------|-------|
| mean wall time | **188.7 µs** |
| CI (lower / upper) | 188.3 µs / 189.2 µs |

Gate: mean ≤ **226.4 µs** (20% over 188.7 µs).

```bash
cargo +1.85 bench -p causal-identify --bench adjustment
```
