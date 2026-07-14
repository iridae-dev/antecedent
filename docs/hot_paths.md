# Designated hot paths

Index of designated hot paths with Criterion bench targets, baseline docs,
allocation/memory contracts, and owning crates (DESIGN.md §28 / Phase 12).

| Hot path | Owner crate | Bench | Baseline | Allocation / memory contract |
|----------|-------------|-------|----------|------------------------------|
| Sample gather | `causal-kernels` | `gather` | [gather.md](../benches/baselines/gather.md) | Dispatch entry; no per-index heap |
| Graph reachability | `causal-graph` | `traversal` | [graph_traversal.md](../benches/baselines/graph_traversal.md) | Reusable `GraphWorkspace` |
| d-separation | `causal-graph` | `dseparation` | [dseparation.md](../benches/baselines/dseparation.md) | Batch / witness APIs; workspace reuse |
| Adjustment search | `causal-identify` | `adjustment` | [adjustment.md](../benches/baselines/adjustment.md) | Minimal-set enumeration budgets |
| Partial correlation batch | `causal-kernels` / `causal-stats` | `partial_correlation` | [partial_correlation.md](../benches/baselines/partial_correlation.md) | Reusable `ParCorrWorkspace` |
| PCMCI discovery | `causal-discovery` | `pcmci` | [pcmci.md](../benches/baselines/pcmci.md) | LaggedFrame + DiscoveryWorkspace; no per-CI plan rebuild |
| CI / orientation (P5) | `causal-stats` / `causal-discovery` | `ci_phase5`, `orientation` | [phase5_ci_orientation.md](../benches/baselines/phase5_ci_orientation.md) | Batch CI; mask complete-case |
| Propensity bootstrap | `causal-estimate` | `propensity_bootstrap` | [propensity.md](../benches/baselines/propensity.md) | Workspace buffer reuse across replicates |
| Matching index | `causal-stats` | `matching` | [matching.md](../benches/baselines/matching.md) | Exact path ≤ 10k; retain index on compatible fits |
| m-separation / PAG orient | `causal-graph` / `causal-discovery` | `mseparation`, `pag_orientation` | [phase8_pag.md](../benches/baselines/phase8_pag.md) | Sparse + stress fixtures |
| RPCMCI / temporal mediation | `causal-discovery` / `causal-estimate` | `rpcmci`, `temporal_mediation` | [phase9_regime_mediation.md](../benches/baselines/phase9_regime_mediation.md) | Multi-env plans must not clone sibling series |
| Shapley attribution | `causal-attribution` | `shapley` | [phase10_shapley.md](../benches/baselines/phase10_shapley.md) | Coalition cache; exact size gates |
| Design ranking / state append | `causal-design` / `causal-state` | `design_rank`, `state_append` | [phase11_design_state.md](../benches/baselines/phase11_design_state.md) | MonteCarloBudget; CacheBudget refuse |

## Smoke commands

Criterion smokes used by phase gates (`--test`):

```bash
cargo bench -p causal-kernels --bench gather -- --test
cargo bench -p causal-graph --bench traversal -- --test
cargo bench -p causal-graph --bench dseparation -- --test
cargo bench -p causal-identify --bench adjustment -- --test
cargo bench -p causal-kernels --bench partial_correlation -- --test
cargo bench -p causal-discovery --bench pcmci -- --test
cargo bench -p causal-attribution --bench shapley -- --test
cargo bench -p causal-design --bench design_rank -- --test
cargo bench -p causal-state --bench state_append -- --test
```

Absolute timings in baseline files are machine-class references (Apple M1).
Unexplained regressions beyond documented budgets block merge (DESIGN §28.9).
