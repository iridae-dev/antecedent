# Causal-response / interference benchmark baselines

Workloads (`antecedent-estimate` bench `response_interference`):

- `kennedy_curve_n4k_grid5` — cross-fitted Kennedy DR mean curve, n = 4 000,
  5-point grid, 1 confounder, default options (5 folds, Silverman bandwidth).
- `kennedy_curve_n4k_grid5_simultaneous` — same `curve_data(4000)` / 5-point
  `MeanCurve` fixture with `bandwidth = Some(0.35)` and
  `simultaneous_replicates = Some(100)` (explicit bandwidth required; 100 is
  the estimator minimum).
- `interference_cluster_n10k_2kdraws` — randomized exposure contrast under
  cluster randomization, n = 10 000 units / 1 000 clusters, degree-4 network,
  2 000 Monte Carlo exposure draws per level.

Established: 2026-08-18 (0.5.2 performance pass); simultaneous workload
2026-08-18 (0.6.0)
Machine class: Apple M1 Max (arm64), 64 GB
Criterion sample size: 100

## Accepted measurement

| Workload | mean wall time |
|----------|----------------|
| kennedy_curve_n4k_grid5 | **210 ms** |
| kennedy_curve_n4k_grid5_simultaneous | **216 ms** |
| interference_cluster_n10k_2kdraws | **167 ms** |

## Acceptance

Regressions exceeding **20%** wall-time vs the last accepted Criterion run on
the same machine class require an approved explanation and replacement
baseline. Additionally the bench asserts a hard soft-budget gate of
**1 s / iter** for each workload (including `kennedy_curve_n4k_grid5_simultaneous`)
on every invocation (including `--test` smokes) — ~5× headroom, sized so the
pre-0.5.2 failure modes cannot return:

- the O(n²) pseudo-outcome loop (≈4 s at n = 4 000; hours at n = 100 000);
- the O(draws × n × treated_clusters) cluster-membership scan.

## Scaling contract

Kennedy pseudo-outcomes are O(n) per fold in GAM predictions (the additive
covariate offset is hoisted per fold; `predict_row` is allocation-free). The
remaining O(n²/K) term is the exact marginal-density KDE sum, which is
definitional. Interference draws are O(n + clusters) each with reused
assignment/exposure buffers (`AssignmentSampler`), bit-identical to the
one-shot sampler (differential test
`monte_carlo_sampler_reuse_is_bit_identical_to_one_shot_reference`).

## How to refresh

```bash
cargo +1.85 bench -p antecedent-estimate --bench response_interference
```
