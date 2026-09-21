# Designated hot paths

Index of designated hot paths with Criterion bench targets, baseline docs,
allocation/memory contracts, and owning crates.

| Hot path | Owner crate | Bench | Baseline | Allocation / memory contract |
|----------|-------------|-------|----------|------------------------------|
| Sample gather | `antecedent-kernels` | `gather` | [gather.md](https://github.com/iridae-dev/antecedent/blob/main/benches/baselines/gather.md) | Dispatch entry; no per-index heap |
| Kernel reductions | `antecedent-kernels` | `reductions` | [kernel_reductions.md](https://github.com/iridae-dev/antecedent/blob/main/benches/baselines/kernel_reductions.md) | Reuse out/table buffers; scalar↔portable differential |
| Graph reachability | `antecedent-graph` | `traversal` | [graph_traversal.md](https://github.com/iridae-dev/antecedent/blob/main/benches/baselines/graph_traversal.md) | Reusable `GraphWorkspace` |
| d-separation | `antecedent-graph` | `dseparation` | [dseparation.md](https://github.com/iridae-dev/antecedent/blob/main/benches/baselines/dseparation.md) | Batch / witness APIs; workspace reuse |
| Adjustment search | `antecedent-identify` | `adjustment` | [adjustment.md](https://github.com/iridae-dev/antecedent/blob/main/benches/baselines/adjustment.md) | Minimal-set enumeration budgets |
| Partial correlation batch | `antecedent-kernels` / `antecedent-stats` | `partial_correlation` | [partial_correlation.md](https://github.com/iridae-dev/antecedent/blob/main/benches/baselines/partial_correlation.md) | Reusable `ParCorrWorkspace` |
| PCMCI discovery | `antecedent-discovery` | `pcmci` | [pcmci.md](https://github.com/iridae-dev/antecedent/blob/main/benches/baselines/pcmci.md) | LaggedFrame + DiscoveryWorkspace; no per-CI plan rebuild |
| CI / orientation | `antecedent-stats` / `antecedent-discovery` | `ci_framework`, `orientation` | [ci_orientation.md](https://github.com/iridae-dev/antecedent/blob/main/benches/baselines/ci_orientation.md) | Batch CI; mask complete-case |
| Propensity bootstrap | `antecedent-estimate` | `propensity_bootstrap` | [propensity.md](https://github.com/iridae-dev/antecedent/blob/main/benches/baselines/propensity.md) | Workspace buffer reuse across replicates |
| Progressive estimate execute | `causal` | (conformance) `latency_tiers` | — | StageClock + ProgressSink + `StageResultSink` payloads; effort on `ExecutionPerformanceRecord` |
| Cancel mid-bootstrap | `antecedent-estimate` / `causal` | (conformance) `latency_tiers::cancel_mid_bootstrap_yields_partial_not_silent_full` | — | Soft partial SE; `cancelled` flag; no silent full result |
| Adaptive bootstrap | `antecedent-estimate` / `causal` | (conformance) `latency_tiers::adaptive_bootstrap_pin_stable_count_and_se`, `production_replicates` | — | Opt-in only (`ExecutionContext::production` and `for_tests` both evaluate the full request). When enabled, the stop is a Monte Carlo error bound, not a convergence heuristic: after `AdaptiveBootstrapBudget::required_replicates()` successes (`max(min_replicates, ⌈1 + 1/(2ε²)⌉)`, the count at which the relative MC SE of a bootstrap SE, `≈ 1/√(2(B−1))`, is `≤ ε`); `early_stopped` + actual `bootstrap_replicates_ok` |
| Adaptive Bayesian draws | `antecedent-estimate` / `causal` | (conformance) `latency_tiers::adaptive_draws_preserve_exact_nig_count_and_width` | — | Opt-in only. When enabled, Laplace MVN redraws stop once the effect-draw ESS reaches `ess_target` (the quantile-width heuristic was removed); `early_stopped` + actual `n_draws` |
| Prepared re-estimate | `causal` | (conformance) `prepared_analysis` | — | Compile-once `PreparedStudy`; identify-once cache (`exec.identify.cached`); temporal response caches `I(h)` per requested horizon; schema-gated refresh; 2nd shot skips compile and identification |
| Discover-once / estimate-many | `causal` / Python | (conformance) `latency_tiers::discovered_graph_builds_under_every_latency_tier`, `test_accepted_graph`, `test_discovery_interactive_guard` | — | Python `analyze` refuses inline `discovery=` under `latency="interactive"` (the Rust builder cannot express the combination); `AcceptedGraph` version stable across estimate clicks |
| Shared estimate→refute workspace | `causal` | (conformance) `shared_workspace` | — | `StaticEstimateWorkspaces` for linear / propensity / AIPW across estimate→refute |
| Interactive graph×effect subsample | `antecedent-estimate` / `causal` | (unit) `envelope::interactive_subsample_mass_accounting_honest`; (conformance) `response_facade::interactive_graph_posterior_*`, `response_facade::interactive_bayesian_*` | — | Identified atoms left out of the subsample are never fit (DBN-posterior envelopes fit first and leave them out of the mixture). Every subsampling path reports their mass as `subsampled_out_mass` (not unidentified, not unevaluable) and stays `GraphDependent`: the Frequentist graph-posterior ATE in its envelope diagnostic, graph-posterior responses on `StructuralResponseMixture`, and the Bayesian graph-posterior, DBN-posterior and CPDAG/PAG class envelopes on `CausalPosterior` (and its artifact). The `estimate.envelope.interactive_subsample` diagnostic lists the skipped atom keys and where their mass is reported |
| Arrow CDI interactive estimate | Python / `antecedent-data` | (conformance) `test_arrow_interactive_smoke` | — | PyArrow tables borrow via CDI whenever `try_as_arrow_c_columns` succeeds (no population kwargs); pandas/dict go through `as_columns`. `latency=interactive` is the intended consumer, not a gate on the ingest path. `performance.bytes_borrowed` is set on the Arrow analyze path |
| Post-ID column projection | `antecedent-data` / `causal` | (conformance) `projection_wide` | — | Wide sheet → gather T/Y/Z only; ATE matches; `exec.project.columns` diagnostic |
| Batch multi-query | `causal` / Python | (conformance) `batch_analysis`, `test_analyze_many` | — | One table ingest, N AverageEffect queries; match solo ATE |
| Refute second click | `causal` / Python | (conformance) `refute_second_click`, `test_prepared_refute_second_click` | — | Prepared estimate then `refute(suite)`; ATE frozen; validation replaced |
| Matching index | `antecedent-stats` | `matching` | [matching.md](https://github.com/iridae-dev/antecedent/blob/main/benches/baselines/matching.md) | Exact path ≤ 10k; retain index on compatible fits |
| m-separation / PAG orient | `antecedent-graph` / `antecedent-discovery` | `mseparation`, `pag_orientation` | [pag.md](https://github.com/iridae-dev/antecedent/blob/main/benches/baselines/pag.md) | Sparse + stress fixtures |
| RPCMCI / temporal mediation | `antecedent-discovery` / `antecedent-estimate` | `rpcmci`, `temporal_mediation` | [regime_mediation.md](https://github.com/iridae-dev/antecedent/blob/main/benches/baselines/regime_mediation.md) | Multi-env plans must not clone sibling series |
| Shapley attribution | `antecedent-attribution` | `shapley` | [shapley.md](https://github.com/iridae-dev/antecedent/blob/main/benches/baselines/shapley.md) | Coalition cache; exact size gates |
| Design ranking / state append | `antecedent-design` / `antecedent-state` | `design_rank`, `state_append` | [design_state.md](https://github.com/iridae-dev/antecedent/blob/main/benches/baselines/design_state.md) | MonteCarloBudget; CacheBudget refuse |
| Laplace GLM fit | `antecedent-prob` | `laplace_glm` | [laplace_glm.md](https://github.com/iridae-dev/antecedent/blob/main/benches/baselines/laplace_glm.md) | `LaplaceWorkspace` grow-only reuse (asserted in bench) |
| HMC GLM fit | `antecedent-prob` | `hmc` | [hmc.md](https://github.com/iridae-dev/antecedent/blob/main/benches/baselines/hmc.md) | `LaplaceWorkspace` grow-only reuse (asserted in bench); `--test` smoke is not a publication gate |
| MCMC diagnostics | `antecedent-prob` | `mcmc_stats` | [mcmc_stats.md](https://github.com/iridae-dev/antecedent/blob/main/benches/baselines/mcmc_stats.md) | Per-statistic Geyer / rank-normalized R̂ (no FFT) |
| GCM interventional sample | `antecedent-model` | `sample_overlay` | [sample_overlay.md](https://github.com/iridae-dev/antecedent/blob/main/benches/baselines/sample_overlay.md) | `MechanismWorkspace` grow-only reuse (asserted in bench) |
| Linear-Gaussian counterfactual | `antecedent-counterfactual` | `counterfactual_batch` | [counterfactual_batch.md](https://github.com/iridae-dev/antecedent/blob/main/benches/baselines/counterfactual_batch.md) | Full-column and `unit_rows` predict; streaming ≡ retained |
| Posterior functional eval | `antecedent-estimate` | `posterior_functional` | [posterior_functional.md](https://github.com/iridae-dev/antecedent/blob/main/benches/baselines/posterior_functional.md) | Eval workspace grow-only reuse (asserted in bench) |
| Kennedy response curve | `antecedent-estimate` | `response_interference` | [response_interference.md](https://github.com/iridae-dev/antecedent/blob/main/benches/baselines/response_interference.md) | O(n)/fold GAM predictions (additive offset hoist); allocation-free `predict_row`. Same fixture with opt-in simultaneous band (`kennedy_curve_n4k_grid5_simultaneous`: explicit bandwidth + 100 wild-multiplier replicates) |
| Temporal dose × horizon response | `antecedent-estimate` / `causal` | `temporal_response` | [temporal_response.md](https://github.com/iridae-dev/antecedent/blob/main/benches/baselines/temporal_response.md) | Prepare-once identification/indexer; multi-horizon estimate reuses fitted lag design across doses |
| Bayesian temporal response / sustained window | `antecedent-estimate` | `temporal_response` | [temporal_response.md](https://github.com/iridae-dev/antecedent/blob/main/benches/baselines/temporal_response.md) | Retain grid-sized response summaries or draw-sized window effects; shared moving-block resamples across mechanisms; prepared identification outside fit |
| Bayesian temporal mediation | `antecedent-estimate` | `temporal_mediation` | [regime_mediation.md](https://github.com/iridae-dev/antecedent/blob/main/benches/baselines/regime_mediation.md) | Two prepared designs and reusable posterior workspace; composed output retains four quantities per draw |
| Randomized interference MC | `antecedent-estimate` / `antecedent-stats` | `response_interference` | [response_interference.md](https://github.com/iridae-dev/antecedent/blob/main/benches/baselines/response_interference.md) | `AssignmentSampler` buffer reuse; O(n+clusters)/draw; validate network once |
| Temporal Pulse / Sustained without replicates | `antecedent-estimate` / `causal` | `temporal_zero_replicates` | [temporal_zero_replicates.md](https://github.com/iridae-dev/antecedent/blob/main/benches/baselines/temporal_zero_replicates.md) | Dependence-aware block length (score refits + Politis–White scans, O(n^1.5)) only when replicates are drawn; the rule length otherwise; refuters reuse the published length |
| Transport identify / catalog / eval / bootstrap | `antecedent` / `antecedent-identify` / `antecedent-expr` | `transport` | [transport.md](https://github.com/iridae-dev/antecedent/blob/main/benches/baselines/transport.md) | Identification by graph width and district count; catalog search by regime count; exact evaluation by joint cardinality; empirical-table bootstrap cost. Inspection stays off this path. |

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
cargo bench -p antecedent-estimate --bench response_interference -- --test
cargo bench -p antecedent-estimate --bench temporal_response -- --test
cargo bench -p antecedent-prob --bench laplace_glm -- --test
cargo bench -p antecedent-prob --bench hmc -- --test
cargo bench -p antecedent-prob --bench mcmc_stats -- --test
cargo bench -p antecedent-model --bench sample_overlay -- --test
cargo bench -p antecedent-counterfactual --bench counterfactual_batch -- --test
cargo bench -p antecedent --bench staged_handle -- --test
cargo bench -p antecedent --bench temporal_zero_replicates -- --test
cargo bench -p antecedent --bench transport -- --test
```

Absolute timings in baseline files are machine-class references (Apple M1).
Unexplained regressions beyond documented budgets block merge.

## 1.3 staged execution

Cached derivative estimation and refitted GCM unit counterfactual execution are
measured in `antecedent/benches/staged_handle.rs`.
[Baseline](https://github.com/iridae-dev/antecedent/blob/main/benches/baselines/staged_handle.md). Preparation is excluded from
these timings; supplied-data estimation remains inside each iteration.
The same target also smokes inspect, capability, and metadata-only artifact
reads (`inspect_n400`, `capability_n400`, `metadata_only_artifact_read`;
none published). Inspection must not clone datasets or rerun identification.
