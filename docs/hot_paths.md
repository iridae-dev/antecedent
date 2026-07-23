# Designated hot paths

Index of designated hot paths with Criterion bench targets, baseline docs,
allocation/memory contracts, and owning crates.

| Hot path | Owner crate | Bench | Baseline | Allocation / memory contract |
|----------|-------------|-------|----------|------------------------------|
| Sample gather | `antecedent-kernels` | `gather` | [gather.md](../benches/baselines/gather.md) | Dispatch entry; no per-index heap |
| Kernel reductions | `antecedent-kernels` | `reductions` | [kernel_reductions.md](../benches/baselines/kernel_reductions.md) | Reuse out/table buffers; scalar↔portable differential |
| Graph reachability | `antecedent-graph` | `traversal` | [graph_traversal.md](../benches/baselines/graph_traversal.md) | Reusable `GraphWorkspace` |
| d-separation | `antecedent-graph` | `dseparation` | [dseparation.md](../benches/baselines/dseparation.md) | Batch / witness APIs; workspace reuse |
| Adjustment search | `antecedent-identify` | `adjustment` | [adjustment.md](../benches/baselines/adjustment.md) | Minimal-set enumeration budgets |
| Partial correlation batch | `antecedent-kernels` / `antecedent-stats` | `partial_correlation` | [partial_correlation.md](../benches/baselines/partial_correlation.md) | Reusable `ParCorrWorkspace` |
| PCMCI discovery | `antecedent-discovery` | `pcmci` | [pcmci.md](../benches/baselines/pcmci.md) | LaggedFrame + DiscoveryWorkspace; no per-CI plan rebuild |
| CI / orientation | `antecedent-stats` / `antecedent-discovery` | `ci_framework`, `orientation` | [ci_orientation.md](../benches/baselines/ci_orientation.md) | Batch CI; mask complete-case |
| Propensity bootstrap | `antecedent-estimate` | `propensity_bootstrap` | [propensity.md](../benches/baselines/propensity.md) | Workspace buffer reuse across replicates |
| Progressive estimate execute | `causal` | (conformance) `latency_tiers` | — | StageClock + ProgressSink + `StageResultSink` payloads; effort on `ExecutionPerformanceRecord` |
| Cancel mid-bootstrap | `antecedent-estimate` / `causal` | (conformance) `latency_tiers::cancel_mid_bootstrap` | — | Soft partial SE; `cancelled` flag; no silent full result |
| Adaptive bootstrap | `antecedent-estimate` / `causal` | (conformance) `latency_tiers::adaptive_bootstrap_pin` | — | SE relative early-stop; `early_stopped` + actual `bootstrap_replicates_ok` |
| Adaptive Bayesian draws | `antecedent-estimate` / `causal` | (conformance) `latency_tiers::adaptive_draws_pin` | — | Quantile-width early-stop; `early_stopped` + actual `n_draws` |
| Prepared re-estimate | `causal` | (conformance) `prepared_analysis` | — | Compile-once Ready plan; schema-gated refresh; 2nd shot skips compile |
| Discover-once / estimate-many | `causal` / Python | (conformance) `latency_tiers::interactive_refuses_inline_discovery`, `test_accepted_graph`, `test_discovery_interactive_guard` | — | Interactive refuses `Discover*`; `AcceptedGraph` version stable across estimate clicks |
| Shared estimate→refute workspace | `causal` | (conformance) `shared_workspace` | — | `StaticEstimateWorkspaces` for linear / propensity / AIPW across estimate→refute |
| Interactive graph×effect subsample | `antecedent-prob` / `causal` | (unit) `envelope::interactive_subsample_mass_accounting_honest` | — | Leftover identified mass → `unidentified_mass`; approximate diagnostic |
| Arrow CDI interactive estimate | Python / `antecedent-data` | (conformance) `test_arrow_interactive_smoke` | — | Prefer CDI borrow under `latency=interactive`; pandas correct but not latency default |
| Post-ID column projection | `antecedent-data` / `causal` | (conformance) `projection_wide` | — | Wide sheet → gather T/Y/Z only; ATE matches; `exec.project.columns` diagnostic |
| Batch multi-query | `causal` / Python | (conformance) `batch_analysis`, `test_analyze_many` | — | One table ingest, N AverageEffect queries; match solo ATE |
| Refute second click | `causal` / Python | (conformance) `refute_second_click`, `test_refute_second_click` | — | Prepared estimate then `refute(suite)`; ATE frozen; validation replaced |
| Matching index | `antecedent-stats` | `matching` | [matching.md](../benches/baselines/matching.md) | Exact path ≤ 10k; retain index on compatible fits |
| m-separation / PAG orient | `antecedent-graph` / `antecedent-discovery` | `mseparation`, `pag_orientation` | [pag.md](../benches/baselines/pag.md) | Sparse + stress fixtures |
| RPCMCI / temporal mediation | `antecedent-discovery` / `antecedent-estimate` | `rpcmci`, `temporal_mediation` | [regime_mediation.md](../benches/baselines/regime_mediation.md) | Multi-env plans must not clone sibling series |
| Shapley attribution | `antecedent-attribution` | `shapley` | [shapley.md](../benches/baselines/shapley.md) | Coalition cache; exact size gates |
| Design ranking / state append | `antecedent-design` / `antecedent-state` | `design_rank`, `state_append` | [design_state.md](../benches/baselines/design_state.md) | MonteCarloBudget; CacheBudget refuse |

## Smoke commands

Criterion smokes used by feature gates (`--test`):

```bash
cargo bench -p antecedent-kernels --bench gather -- --test
cargo bench -p antecedent-kernels --bench reductions -- --test
cargo bench -p antecedent-graph --bench traversal -- --test
cargo bench -p antecedent-graph --bench dseparation -- --test
cargo bench -p antecedent-identify --bench adjustment -- --test
cargo bench -p antecedent-kernels --bench partial_correlation -- --test
cargo bench -p antecedent-discovery --bench pcmci -- --test
cargo bench -p antecedent-attribution --bench shapley -- --test
cargo bench -p antecedent-design --bench design_rank -- --test
cargo bench -p antecedent-state --bench state_append -- --test
```

Absolute timings in baseline files are machine-class references (Apple M1).
Unexplained regressions beyond documented budgets block merge.
