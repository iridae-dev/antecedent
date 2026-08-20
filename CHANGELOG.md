# Changelog

All notable changes to Antecedent are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.7.0] — 2026-08-20

Time as a response, not a contrast: dose × horizon surfaces and temporal
intervention paths on `TemporalDag`. Artifact format **0.4**. Workspace and
Python package versions are **0.7.0**.

### Temporal Response

- Added temporal dose × horizon and intervention-response g-computation on
  `TemporalDag`, including prepared identification reuse and format 0.4
  artifact fields. Licensed `PulseEffect` / single-step `SustainedEffect`,
  which dispatch through the same `TemporalLinearAdjustment` machinery and
  agree numerically with that spine on the shared known-truth fixture.
  `InterventionResponse` licenses Soft(`constant`/`additive_shift`) and a
  single-step `Sequence`; multi-step and nested `Sequence` policies fail
  closed with a stable error. Temporal response pointwise bands now use the
  full homoskedastic OLS coefficient covariance (delta-method SE of the
  g-computed level), propagating intercept and adjustment-covariate
  uncertainty instead of scaling the treatment-coefficient SE alone; bands
  are pinned on `conformance/response/temporal_dose_horizon`. Empirical
  support is per `(dose, horizon)` cell against that horizon's lag-aligned
  treatment range; `SupportReport.status` summarizes the cell grid (fully
  supported / partially extrapolative / outside) instead of the union of
  horizon ranges. Two supporting fixtures add redundancy:
  `conformance/response/temporal_confounded_pulse` (adjustment set
  `Z@-1` at horizon 1, identified estimand `temporal.backdoor.unfolded`,
  the structural estimate under confounding, and horizon-dependent
  `I(h)` on `horizons=[1,2]`) and
  `conformance/response/temporal_horizon_support` (horizon-varying treatment
  support). Scalar ATE refuters skip on function-valued surfaces:
  Python exposes
  `refute.temporal_response.skipped` on `CausalResponseView.validation`, and
  Rust `Study` diagnostics carry the same code. Examples:
  `examples/python/temporal_response_curve.py`. Python query construction
  reads `TemporalResponseSpec::license()` (horizon cap, allowed policies,
  default treatment lag) instead of mirroring those values.

### Fixed

- **Single-step Sustained / Dynamic temporal identification uses the pulse
  backdoor set.** Licensed one-offset schedules previously went through
  general ID, which emits an empty adjustment set relabeled as
  `temporal.backdoor.unfolded`. Under confounding that was unadjusted OLS.
  Multi-step schedules still use sequential ID and remain estimator-refused.
- **Temporal response identifies once per requested horizon.** Identifying
  only at `max(horizons)` can drop a short-horizon confounder when it does
  not reach the longer outcome. Prepare caches `I(h)` for every requested
  horizon; estimate clicks still do not re-identify. A union of those sets
  is not used as one shared `Z`.
- **Conjugate Gram cache no longer restores stale `Xᵀy`.** The cache keyed on
  outcome pointer identity; SBC (and other shared-workspace refits) often
  recycle equal-length `y` allocations to the same address, so later
  replicates could rank against another replicate's likelihood. Cache only
  `XᵀX` and always recompute `Xᵀy` / `yᵀy`.

## [0.6.1] — 2026-08-20

Patch on the 0.6.0 contract cut. No public API rename. Licensed cells are
unchanged. Workspace and Python package versions are **0.6.1**.

### Fixed

- **Bayesian envelopes keep their prior anchor under Interactive subsample.**
  Stratified subsample could drop the atom that identification used as the
  prior anchor. Prepare now runs first; the anchor stays in original order.
  Temporal DBN posteriors also fit before subsample. Failures on the
  Interactive path demote rather than silently changing the identified set.
- **PC `BlockShuffle` parent RNG salt stays 0.** Parent selection under
  analytic CI had drifted the salt, so discovered parent sets were no longer
  bit-identical to 0.6.0.
- **Python ingest is one path.** `PreparedAnalysis.prepare` uses the same
  Arrow CDI ingest as `analyze`. NumPy and Arrow raise the same errors on
  the same bad input. `np.asarray` on a posterior artifact has a locked
  zero-copy / copy contract.

### Performance / quality

- Prepared estimate clicks no longer clone the `Study`.
- Kennedy mean curves reuse GAM workspace, row scratch, and local-quadratic
  weights across the grid.
- Closed-form Gaussian g-comp contrast; AIPW borrows design columns when
  nothing is trimmed; conjugate Gram stats cache across prior-scale refits;
  temporal linear adjustment prepares from column slices; intervention-response
  g-comp reuses scratch rows.
- PC parent CI tests batch per conditioning level. LPCMCI / PCMCI+ separating
  sets are `Arc`, not cloned.
- Python prefers Arrow CDI on prepared clicks and typed-graph ATE, and
  exposes posterior draws to NumPy without cloning the `Vec`.
- Execution finish paths share `AnalysisRoute` / identified-execute helpers
  (static ATE, RD, ADMG, PAG, Bayesian, temporal, front-door). Not a
  semantics change.

## [0.6.0] — 2026-08-19

### Changed

- **The matrix partition is total; unlicensed-and-unlisted now refuses.**
  Every cell is licensed, n/a, closed, **allowlisted** (running,
  unlicensed, named in `parity/support_allowlist.toml` with a reason and
  parent), or fails closed with the stable `refused:` id. All 167
  formerly silent default-refused cells were probed: 74 run and are
  allowlisted (PAG / bidirected-ADMG ATE, posterior ATE, Pulse on
  temporal graphs including non-collapsing accepted temporal
  CPDAGs/PAGs, frequentist TemporalMediation, validation copies of
  licensed families); 42 closed with proof — including two cells where
  a Bayesian label silently ran the frequentist estimator (ConditionalEffect,
  TemporalMediation: bit-identical output — a mislabeled number is not an
  honest number); 51 unreachable cells fail closed. ADR 0020 amended.
- **Fail-shut for the already-dead set.** ConditionalEffect /
  PathSpecificEffect / InterventionalDistribution / InterventionResponse
  off a Dag, TemporalMediation on TemporalCpdag/TemporalPag, and every
  query × graph-posterior × Frequentist now refuse at `Study::build` with
  stable ids — each proven unable to return a number before closing.
- **Sixteen licensed cells; two are real promotions.** `ConditionalEffect`
  × `Dag` × accepted (the analyze path was silently dropping the
  `AcceptedGraph` marker for conditional queries — fixed) and
  `InterventionResponse` × `Dag` × {explicit, accepted} on a new
  known-truth fixture (`conformance/response/intervention_response`,
  analytic `E[Y|do(T)]` under a zero-noise linear SCM, tolerance pinned
  at 3× the measured GAM regularization bias) with a
  `prepare_intervention_response` staged surface.
- **Honest graph-class collapses.** Cell classification collapses an ADMG
  with no bidirected edges and a (structurally always fully-oriented)
  CPDAG to the `Dag` cell under `AverageEffect`, and complete
  `TemporalCpdag`/`TemporalPag` to `TemporalDag` under temporal-effect
  queries — query-scoped to exactly where compile coerces to those paths.
  Static PAGs never collapse: circle marks are information.
- **Release notes cannot drift from the matrix.** The licensed-cell block
  in `docs/release-notes/v0.6.0.md` is generated between markers and
  enrolled in the release gate's regenerate-and-diff step; the prose
  package-version gate now scans README and the docs landing pages.
- **Thirteen licensed cells.** `ConditionalEffect`, `PathSpecificEffect`,
  and `InterventionalDistribution` × `Dag` × explicit × Frequentist × none
  are licensed on the staged handle: prepare caches their identification,
  clicks reuse it (`exec.identify.cached`), and the Python
  `PreparedAnalysis.prepare` accepts the three query types. Evidence:
  the conditional-effect, path-specific-natural, and ID/IDC hedge
  known-truth fixtures. `AverageEffect` × `Dag` × {explicit, accepted} ×
  Frequentist × full joins the matrix on the placebo + random-common-cause
  refuter fixture (`conformance/validate/refuters`), and × Bayesian × none
  joins on the shared-functional cross-check
  (`conformance/bayesian/shared_functional_ate`) now that the prepared
  handle caches Bayesian identification. Licensed rows carry `limitations`
  notes: the estimator is not a matrix axis (IPW / IV / front-door /
  rd.sharp ride the ATE cells on their own parity fixtures) and
  observation-mechanism curves ride the ResponseCurve cells.
- **Prepared Bayesian clicks no longer re-identify.** `execute_bayesian`
  identification is pure over (identifier, graph, query) and rd-blind, so
  `Study::prepare` caches it and clicks reuse it (`exec.identify.cached`),
  matching the ADR 0020 freeze contract. Prepared-vs-fresh bit equality is
  pinned in tests.
- **`graph_posterior` is refused on the prepared handle.** The placeholder
  graph shape previously slipped past `ensure_prepared_supported`, so a
  posterior-backed prepare "succeeded" and every click re-identified
  per-graph. Now a stable `refused:` error until such a cell is licensed.
- **`TemporalPolicy::Dynamic` classifies by schedule shape.** It previously
  bypassed the support matrix entirely (no axis name): one active step
  rides the `PulseEffect` cell, longer schedules hit the `SustainedEffect`
  closure — mirroring the estimator's multi-step refusal.
- **First enforced support-matrix refusals.** Cells listed in
  `parity/support_closed.toml` now fail closed with id `refused` at
  `Study::build` and the Python sidecars: Counterfactual, the derivative
  family, ResponseCurve on Pag/Cpdag/Admg, and PathSpecific /
  InterventionalDistribution with accepted or graph-posterior structure.
  Remaining default-refused cells still run until licensed or closed.
  `MediationEffect` and `SustainedEffect` are now in that closed set.
  `identify_only` graph-posterior / ADMG / non-DAG refusals use the same
  `refused:` id instead of a stringly `Unsupported`.
- **ResponseCurve on AcceptedGraph is licensed.**
  `PreparedAnalysis.prepare_response(accepted=True)` records
  `structure=accepted`. Same two-point Kennedy fixture as the explicit cell.
- **Stage APIs are not analyze cells.** `TransportQuery` and
  `InterferenceQuery` are n/a on temporal graphs and enforced-refused on
  static graphs. The implemented sID subset remains a stage API returning
  `NotCertified` outside it; it is not a licensed matrix cell.
- **Inventory docs cannot license a cell.** `scripts/gate_docs_support_matrix.sh`
  (unconditional in `gate_release.sh`) requires capabilities/comparison to
  link the matrix and fails unhedged PAG-response / Counterfactual support
  claims.

### Performance / quality

- **Simultaneous-band hot path.** Licensed `ResponseCurve` now has a Criterion
  workload (`kennedy_curve_n4k_grid5_simultaneous`: n = 4000, 5-point grid,
  explicit bandwidth, 100 wild-multiplier replicates) next to the Kennedy
  curve bench, with a 1 s/iter `--test` soft budget.

## [0.5.2] — 2026-08-18

Performance pass plus a localized correctness audit. Workspace and Python
package versions are **0.5.2**.

### Performance / quality

- **Prepared estimates identify once.** `Study::prepare` now runs static
  identification at prepare time and every `PreparedStudy` estimate click
  reuses it (exact: identification reads only prepare-frozen inputs), with an
  `exec.identify.cached` diagnostic making the reuse observable and tested.
  RD-sharp, Bayesian g-comp, posterior-graph, bidirected-ADMG, and PAG
  configurations keep their identify-per-run paths. The Python
  `PreparedAnalysis` handle holds its `PreparedStudy` behind an `Arc`, so
  per-click detach is a refcount bump instead of a deep clone.
- **`ComputeBudget.wall_ms` is documented as advisory.** The field was already
  rustdoc'd as not a hard stop; `StudyBuilder::compute_budget` and the
  architecture execution-model notes now say the same.
- **No direct `rayon`.** A workspace-invariant test refuses a direct `rayon`
  dependency so parallelism stays on `ExecutionContext` (transitive lockfile
  entries from faer / criterion are out of scope).
- **IO section lookup and skip scratch.** Seekable / mmap readers index sections
  by id instead of scanning; `read_selective` reuses one 64 KiB skip buffer
  across unread sections.
- **Laplace Newton reuses workspace buffers.** The per-iteration hessian/grad/
  `beta_old` `to_vec`s are gone; `solve_spd` can factor into caller storage
  instead of allocating a Cholesky and `y` on every call.
- **HMC leapfrog reuses `LaplaceWorkspace` q/p/grad.** Transition kernels no
  longer allocate position, momentum, and gradient vectors on every draw.
- **Coalition cache dense index + saturation flag.** Exact Shapley with
  `k ≤ 16` looks up payoffs by mask instead of hashing; a full byte budget
  sets `CacheStats.saturated` instead of refusing silently.
- **Local Markov / residual independence batch CI.** Pair enumeration is
  still quadratic, but each check now issues one partial-correlation batch
  instead of a `test_one` per pair (and local Markov no longer rebuilds
  parent columns inside the inner loop).
- **Coalition sampling reuses a `ValueBatch` buffer.** `sample_observational_into`
  writes into a caller slice; distribution-change and structure-change
  Shapley loops keep one buffer across masks instead of freezing a fresh
  `Arc` per coalition.
- **HMC / MCMC / GCM / CF benches.** `fit_hmc_glm` and `mcmc_summary` have
  Criterion smokes (workspace reuse; HMC `--test` is not a publication gate).
  Linear-Gaussian counterfactual predict now exercises `unit_rows`. GCM
  interventional sampling has a baseline doc.
- **Hot-path baseline metadata gate.** `scripts/gate_hot_path_baselines.sh`
  (via `gate_release.sh`) checks every `hot_paths.md` Baseline link exists
  and records a wall-time or an explicit waiver. `GATE_CRITERION_MEANS=1`
  optionally compares local Criterion means; CI does not enforce M1 numbers.
- **Gather allocation count.** An integration-test `#[global_allocator]`
  asserts `gather` into a pre-sized buffer allocates nothing after setup.
- **Arrow CDI `bytes_borrowed` on analysis results.** The Arrow analyze path
  keeps the ingest borrow count on `ExecutionPerformanceRecord` /
  `PerformanceView`. PyArrow tables still use CDI whenever the C exporter
  works; `latency=interactive` is not a gate on that ingest path.
- **Panel refute refits reuse per-unit column Arcs.** `PreparedRefutation::compile`
  builds a slice template from the stacked table; each replicate swaps only
  mutated or appended columns instead of copying every float column per unit.
- **Static-linear sensitivity grid is one Gram + per-point Cholesky.** The
  `[1, T, Z, u]` sufficient statistics replace a data pass at every partial-R²
  grid point for OLS (temporal / ridge / lasso / Huber keep the old path). A
  differential test pins Gram ATEs to the per-point QR path under
  `BackendSensitive` before the fast path is the default.

- **Sensitivity Gram uses the post-replace complete-case set.** Replacing T/Y
  marks those columns all-valid, so the QR data-pass can keep rows that were
  missing on the original T/Y. Gram now compiles on that same table.

### Correctness — localized audit

- **Nested CF refuses row-coupled mechanisms.** The per-unit fallback froze
  every node to unit `u`'s value across the full column; a state-space
  outcome then mixed other units' noise. That path now errors. Trajectory
  evaluation is unaffected (`unit_rows: None` over the real series).
- **Temporal CPDAG/PAG reject future→past arrowheads.** `TemporalDag` already
  refused source lag nearer the present than the target. `TemporalCpdag`
  `insert_directed` / `orient_undirected` and `TemporalPag` Circle→Arrow
  inserts / `set_marks` did not. A definite or partial arrowhead into an
  earlier lag now returns `GraphError::FutureToPast` (shared helper with
  `TemporalDag`).
- **Linear/GLM adjustment honor `TargetPopulation::Predicate`.** Prepare now
  intersects the complete-case mask with the predicate (named predicates
  need `with_population_registry`), matching the propensity path. The
  previous code accepted Predicate and estimated the full-sample ATE.
- **GLM ATT/ATC analytic SEs use the arm-law gradient.** Point estimates
  already averaged `μ(a,Z)−μ(c,Z)` over the target arm; the delta/sandwich
  gradient averaged over every row. Under a nonlinear link that is the ATE
  gradient, so CIs were for the wrong functional.
- **Trimmed IPW Hajek analytic SE uses retained n.** Point estimates already
  zeroed out-of-trim weights; the influence SE still divided by full-sample
  n, so trimming looked more precise than it was. Analytic SE now subsets
  to retained rows (matching AIPW).
- **Newey–West after trim/matching requires panel times.** Consecutive IF
  indices are not calendar time once rows are dropped or rematched.
  `influence_se_kind` refuses `NeweyWest` when a retained-row map is
  present without `panel_times`; with times, Bartlett products use calendar
  gaps rather than the retained index.
- **Soft CF refuses cross-family noise reuse.** Sampling already rejected a
  Discrete Uniform residual used as additive Gaussian U (and the reverse);
  abduction–prediction applied the soft slot to the abduced noise without
  that check.
- **Uncovered PD-path search flags `max_len` truncation.** Hitting the length
  cap on a path that had not reached the target (or a zero path budget)
  used to return `truncated = false`, so FCI/LPCMCI R9/R10 treated an
  incomplete search as done.
- **IV Wald reports parametric identification.** Graph relevance and exclusion
  checks were already sound; the status claimed nonparametric ATE. Wald
  identifies ATE under linearity (or LATE under monotonicity). AutoIdentifier
  still collects those Wald estimands; the envelope stays nonparametric when a
  backdoor/ID estimand is also present.
- **Matching gathers multiway cluster labels after trim.** `cluster_ids` and
  `panel_times` were already restricted to retained rows; `multiway_ids` stayed
  full-sample length, so a Multiway SE after trim either errored or paired
  the wrong units.
- **Front-door stacked SE refuses Homoskedastic.** Homoskedastic previously
  reused HC0 meat. The constructor default is already HC0; asking for
  classical Homoskedastic now errors instead of silently aliasing.
- **Discriminating-path search flags length/budget truncation.** `max_len < 4`
  or `max_paths == 0` previously returned `truncated = false`, and longer
  candidates skipped by the length cap were dropped silently, so R4 could
  miss a path and still claim a complete search.
- **PAG GAC forbids `Forb(T,Y)`, not all of `De(T)`.** Candidates were
  `An({T,Y}) \ De(T)`, which drops side-effect descendants of `T` that GAC
  allows (`Forb = De(cn)`, `cn = De(T) ∩ An(Y) \ {T}`).
- **Endpoint ∈ Z is not d/m-separated.** DAG d-sep treated `X ⊥ Y | X` as
  separated; PAG definite-status activity treated the same query as connected.
  All three now return not-separated when `Z ∩ {X,Y} ≠ ∅`.

### Round 4 — backlog completion

- **Expr evaluator**: `EmpiricalTableProvider::support` memoizes its cartesian
  products (hits are an `Arc` clone; invalidated on `set_domain`);
  probability lookups use a borrowed factor-key view instead of ~5 owned
  allocations per lookup; loop evaluators share one scoped mutable
  assignment (push/restore) instead of cloning per row; free variables are
  computed once at compile time; empty-set interning is cached; and the
  prepared functional-distribution arena is `Arc`-shared instead of
  deep-cloned per bootstrap replicate.
- **Attribution**: `structure_change` now performs exactly two GCM fits
  total (baseline and comparison hybrids, composed per mask by slot
  selection) instead of two full refits per coalition — pinned bit-for-bit
  against an inline reimplementation of the per-coalition path across every
  mask; robust payoffs precompute parent sources and reuse flat buffers
  across coalitions; `distribution_change` patches a persistent slot buffer
  incrementally between masks; and a documented planned-evaluation budget
  (`MAX_COALITION_SAMPLE_BUDGET`) refuses runaway `2^k × n_samples` plans up
  front instead of running for hours.
- **Validate/data**: bootstrap resampling rebuilds the table once per
  replicate via new `TabularData::with_replaced_floats` (was once per
  column, with per-column weight deep-copies); the warmed propensity
  workspace now reaches the OverlapRule and Riesz refuters (workspace reuse
  only — each still fits over its own mask); sensitivity residualization
  builds its adjustment design once for both targets; `complete_case_mask`
  gained a buffer-reusing form and its doubled call was deduplicated.
- **Model/counterfactual/state/io**: `TableView::float64_slice`/`float64_cow`
  provide borrowed column access, adopted across mechanism fitting,
  evaluation, do-sampling, and abduction (the old accessor copied the full
  column at every call site); `permutation_baseline` hoists loop-invariant
  gathers out of the permutation loop; `dense_of`/`gather_for` are O(1) via
  compile-time maps; `is_low_cardinality` early-exits instead of sorting a
  full copy; the particle filter reuses step scratch and swaps particle
  buffers (bit-identical trajectories); artifacts are zstd-encoded and
  BLAKE3-hashed once instead of twice (pack-time on-wire buffer streamed at
  write), `causal_artifact` stops double-cloning its payload, the mmap
  reader memoizes section verification, and the zero-copy counters
  (`mmap_views`, bytes-loaded bounds) are now asserted.
- **Masked temporal discovery**: the masked-frame CI path compacted **all**
  frame columns per CI test; it now compacts only the columns the query
  reads (x, y, Z, remapped) — per-test cost no longer scales with total
  column count. `LaggedFrame::column_index` is O(1) via a slot map (was a
  linear scan inside per-candidate MCI loops).
- `NetworkData` incoming adjacency is a true CSR (one edge array + offsets;
  was `n + 1` allocations with per-unit `Arc`s), same per-unit edge order.
- **GP mechanism family refuses above `GP_FAMILY_MAX_ROWS = 2_000`** — a
  documented behavior change: the 20-combination × O(n³) hyperparameter grid
  with an O(n²) Gram is unusable above a few thousand rows; the refusal is
  recorded in `failed_families` and selection falls back to other families.
- New baseline docs for `laplace_glm` and `posterior_functional` (measured
  on the reference machine), wired into `docs/hot_paths.md` and the release
  gate's required-baselines list alongside `response_interference`.

### Round 3 — backlog burn-down

- **Nested counterfactual is no longer quadratic in units**: the unit-wise
  fallback (taken whenever outer values vary across units — the normal case)
  ran a full-table predict per unit, O(n_units²·nodes). A column-frozen
  single pass evaluates all units at once for row-independent mechanisms —
  bit-identical to the historical loop (differential-pinned, including
  stochastic inner interventions via the shared RNG-stream argument);
  temporal mechanisms keep the exact per-unit fallback.
- `mcmc_stats`: autocovariances are computed lazily and stop at the Geyer
  truncation point instead of all n−1 lags (O(m·n²) → O(m·n·τ),
  bit-identical); `mcmc_summary` computes max R̂ + min bulk/tail ESS in one
  diagnostics pass where the publication gates previously paid 3×.
- GLM HMC carries the current log-posterior across iterations (the Gaussian
  path already did); Laplace reuses its Cholesky factor for the mode
  covariance instead of refactorizing inside `invert_spd`.
- `McmcDoSampler` carries the current state's KDE density across Metropolis
  iterations (exact 2× on the chain's dominant cost); rejection sampling
  builds its intervention overlay once, not per candidate row.
- `NonparametricSensitivity` fuses the treatment/outcome Nadaraya–Watson
  passes (identical kernel weights, computed once — exact 2× on an uncapped
  O(n²·dim) path); the posterior predictive check hoists per-draw
  coefficients out of the row loop.
- **Fixed (correctness): `local_markov_tests` paired topo-order positions
  with dense-id tables** — on any graph whose topological order is not the
  identity permutation of dense ids it tested the wrong variable pairs
  (including self-pairs). Comparison sets are now derived in dense-id space;
  p-value sets change on affected graphs, where the old values compared the
  wrong columns.
- Crate READMEs brought up to the 0.5 surface (the facade README advertised
  `antecedent = "0.1"` and the retired `CausalAnalysis::builder()` API);
  `gate_metadata_consistency.sh` now fails on stale README dependency
  versions and retired API names.

### Round 2 — expanded coverage (model, counterfactual, prob, attribution, expr, io, CI nulls)

All bit-identical or exact-identity rewrites; no statistical semantics change.

- **CI permutation nulls stop redoing loop-invariant work per replicate.**
  GPDC: the X-side centered distance matrix (n²) was recomputed, with two
  fresh n² buffers, for each of the 49 replicates — now prepared once per
  query with reused side buffers (`dcor_from_sides`, bit-pinned against the
  monolithic form). kNN: the null rebuilt a full `MatchingIndex` (plus
  feature/donor copies) per replicate while the workspace's cached index sat
  unused — now rewrites the Y column of a per-query feature buffer and
  computes k-th self-distances index-free (`MatchingIndex::kth_self_distances`,
  bit-pinned against the index path). G²: X codes and Z strata are invariant
  under a Y-only permutation and are now hoisted; strata are summed in
  sorted-key order, fixing a genuine determinism defect (the observed G²'s
  last bits depended on `HashMap` iteration order, which varies per process).
- **GCM model selection no longer fits the winning family twice**:
  `score_family` returned only a score and the winner was refit from scratch
  with identical inputs; the scoring fit is now retained — ~2× on every
  `assign_and_fit` (the deterministic refit was bit-identical by construction).
- **Ancestral sampling / abduction / counterfactual predict no longer copy the
  gathered parent matrix per node**: five `ws.parents…to_vec()` sites (a
  borrow-checker workaround) replaced with a grow-only gather buffer hoisted
  out of each node loop.
- **HMC leapfrog gradients no longer compute the full observed Hessian**:
  `accumulate_likelihood` unconditionally accumulated O(n·p²) curvature that
  the HMC path never reads — roughly (p+1)/2× of every gradient evaluation,
  ×(L+1) per draw per chain. Gated behind `want_hessian` (Laplace keeps it);
  gradients, and therefore draws, are bit-identical.
- Attribution: `unit_change` re-copied every parent column per row
  (O(rows·parents·n) memcpy for per-row scalars) — columns read once;
  `score_anomalies` hardcoded a test execution context that silently disabled
  the coalition cache (2^k·(1+k/2) instead of 2^k payoff evaluations per row
  at exact k=12 — a 7× penalty; the cache is numerically neutral for the
  deterministic payoff); the Shapley MC loop allocated two vectors per
  permutation (hoisted).
- Expr: interning built the `Arc` key before the cache lookup, so every hit
  paid two allocations — borrow-based lookup allocates only on miss.
- IO: the seek reader's uncompressed-section path made a full extra copy
  inside a function documented "decode without copying" — the owned buffer is
  now moved into the `Arc` (`decode_on_wire_arc_owned`).
- The `sample_overlay` bench constructed its `MechanismWorkspace` inside
  `b.iter()` — the bench guarding workspace reuse measured the cold case —
  and asserted nothing; the workspace is hoisted and a `grow_count` reuse
  gate now runs on every invocation.

### Round 1

Performance pass over the documented hot-path contracts (ADR 0011) and the
0.5.0 causal-response / interference surface. No statistical semantics change:
every rewritten loop is either bit-identical (differential-tested) or an exact
algebraic identity (parity-tested against the definitional form).

### Fixed

- **`ResponseCurve` / Kennedy DR was O(n²) in rows** — the cross-fitted
  pseudo-outcome loop re-predicted the additive outcome GAM for every
  (validation × training) pair. The additive structure makes the covariate
  part of that average fold-constant; it is now hoisted (exact identity,
  pinned by `pseudo_outcome_additive_hoist_matches_brute_force_double_loop`).
  Measured: n=20k analyze went from 100.3 s to 1.31 s (77×); n=10k from
  24.0 s to 0.35 s. The remaining super-linear term is the definitional
  marginal-density KDE sum, O(n²/K) cheap ops.
- `GamFit::predict_row` / `smooth_partial`: allocation-free span-local
  B-spline prediction (a cubic has 4 nonzero bases), bit-identical to
  `predict_gam` (pinned by two new tests). Replaces ~9 heap allocations per
  prediction on every response/ADE/Jacobian path.
- **Interference MC exposure probabilities**: the cluster design scanned a
  `Vec` per unit per draw — O(draws × n × treated_clusters) — and every draw
  re-sorted ids, re-validated the whole edge list, and allocated fresh
  assignment/exposure buffers. A precomputed `AssignmentSampler` now reuses
  all buffers, O(n + clusters) per draw, bit-identical to the old sampler
  (differential test
  `monte_carlo_sampler_reuse_is_bit_identical_to_one_shot_reference`).
- Propensity / AIPW bootstrap replicate loops honored their workspace
  contract in name only: `fit_propensity` sized `workspace.scores` then
  allocated a fresh vector anyway (making the bench's `scores_grow_count`
  assertion vacuous), and AIPW's default no-trim path cloned the full design
  matrix plus three O(n) vectors per replicate. New `fit_propensity_in_place`
  plus reused clip/weight/gather buffers; the existing bench gate is now
  meaningful.
- Adaptive Bayesian draws accumulated batches by re-merging and re-cloning the
  full coefficient matrix each batch (O(D²) copying); blocks are now
  concatenated once (`concat_coefficient_draws`). Draw counts and widths
  unchanged (pinned by `adaptive_draws_preserve_exact_nig_count_and_width`).
- Transport identification: the S-admissible subset scan enumerated all 2^k
  masks k+1 times (22M trips at the k=20 cap); it now enumerates each
  combination once via Gosper's hack in the identical visiting order, with
  loop-invariant selection nodes and buffers hoisted. Reachability probes
  reuse one `GraphWorkspace` instead of allocating per pair.
- `EfficientBackdoorIdentifier` truncated silently at `max_results`; it now
  emits a derivation note, matching `BackdoorIdentifier`.
- Python: `analyze(arrow_table, ..., on_stage=...)` raised `TypeError` — the
  Arrow C entry point lacked the parameter the dict path had (regression test
  added); `estimate_trial_transport` now releases the GIL like the
  interference path.

### Added

- `response_interference` Criterion bench with asserted 1 s soft-budget gates
  (`kennedy_curve_n4k_grid5`, `interference_cluster_n10k_2kdraws`) — the
  pre-0.5.2 quadratic loop fails it by ~4×. Wired into `gate_release.sh` and
  `docs/hot_paths.md` with a new baseline doc.

### Changed (gates and baselines)

- The Shapley 500 ms latency gate was dead code (`#[allow(dead_code)]` entry
  point never invoked) and the 200 ms exact-path budget had no assert; both
  now execute on every bench invocation. `design_rank` / `state_append` soft
  budgets (50 ms / 20 ms) are now asserted rather than prose. The matching
  bench's tautological "reuse gate" (a local counter incremented once) was
  removed in favor of the real estimate-side test it pretended to be.
- `benches/baselines/pcmci.md` re-established at 6.73 ms (gate 8.08 ms): the
  recorded 1.59 ms is not reproducible on the reference machine at any commit
  from 2026-07-19 to HEAD (bisect-benched at four points, 6.0–7.1 ms
  throughout); measured genuine drift is +11%, explained by the 0.4.0
  correctness fixes.
- `gate_release.sh` now runs the `partial_correlation` smoke (previously in no
  gate) and `gate_estimate_reuse.sh` (previously claimed as a PR gate by
  `benches/baselines/propensity.md` but wired to nothing).
- Stale names corrected: `docs/hot_paths.md` cited three test names that no
  longer exist and a Rust discovery-refusal guard that was retired in favor of
  the Python-side guard (`docs/artifacts.md` updated to say so);
  `benches/baselines/ci_orientation.md` named a nonexistent bench target
  (`ci_phase5` → `ci_framework`); `regime_mediation.md` budgets now state the
  asserted gate values (10 / 40 ms) instead of numbers 2× tighter than any
  gate enforced.

## [0.5.1] — 2026-08-18

Makes the claim→evidence chain machine-checkable, adds the gates that keep
it honest, and one new opt-in estimator feature. Workspace and Python
package versions are **0.5.1**.

### Added

- Opt-in per-row response diagnostics (`export_row_diagnostics`): retained-row
  indices, cross-fitted Kennedy pseudo-outcomes, and grid-major local-WLS
  influences, with a documented mathematical contract
  (`docs/causal-responses.md`, "Row-diagnostic export contract") covering
  centering, the exact SE/band reconstruction identities, layout, row-index
  semantics, cross-fitting, and stability. Non-finite influences are refused
  rather than exported.
- `scripts/gate_metadata_consistency.sh`: cross-file drift gate — package
  version, artifact format vs `STABLE_FORMAT`, retired library names,
  duplicate provenance ids, DOI syntax, and oracle-ledger/fixture agreement.
- `scripts/gate_evidence_reachability.sh`: every conformance fixture must be
  loaded by an executing test or declared in `conformance/UNEXERCISED.toml`
  (a two-way, shrink-only ledger); unexercised fixtures cannot back closed
  oracle rows or unmarked citations; new provenance records must carry
  `implementation_deviations` (139 legacy records frozen in a shrink-only
  backlog).
- Machine-readable `evidence_kind` on every `done` parity capability row
  (168 rows classified), with `external_oracle`, `known_truth_fixture`, and
  `limitations`; the schema gate enforces the fixture-authoritative rule for
  external claims.
- Consuming contract test for `conformance/estimate/uncertainty_routing`
  (routing table vs `SandwichKind`, both directions); the fixture was
  previously recorded but loaded by nothing.

### Fixed

- Python builds pin `profile = "release"` in `[tool.maturin]`. Previously a
  stale editable install (e.g. after a version bump) was silently rebuilt by
  the PEP 517/660 backend in Cargo's debug profile — bit-identical estimates
  at ~50× the wall time. `antecedent._native` now exports
  `__build_optimized__`, importing a debug-profile extension emits a
  `RuntimeWarning`, and the pytest suite hard-fails on an unoptimized
  extension (opt out with `ANTECEDENT_ALLOW_DEBUG_NATIVE=1`) instead of
  degrading silently.
- `LinearAdjustmentAte` with `fit_kind = huber` refused on unconverged IRLS
  instead of publishing the last iterate with an analytic SE.
- `artifact_migrate` conformance now asserts against the live `STABLE_FORMAT`
  instead of a second hardcoded copy that had drifted to 0.2.
- `parity/oracle_closure.toml` realigned to its fixtures' own oracle blocks:
  14 rows no longer name upstream packages the frozen fixtures never
  recorded, and 9 `pending-generation` pins were replaced with the real pins
  the fixtures carry.
- Two end-to-end tests now read the `path_specific_natural` and
  `interventional_distribution` fixtures they previously duplicated by hand;
  the reachability scan is word-boundary matched so identifier-name
  coincidences no longer count as fixture references.
- All 40 rustdoc warnings resolved workspace-wide; public `# Errors` docs no
  longer reference private items, and `EXTREME_WEIGHT_THRESHOLD` is public so
  the transport diagnostic doc links a real value.
- Citation metadata split by consumer: Zenodo ingests a new `.zenodo.json`
  (and ignores `CITATION.cff` when it is present), while `CITATION.cff`
  carries the CFF-1.2.0-schema-valid license list for GitHub's citation
  widget and `cffconvert`. Zenodo rejects both the list form and the SPDX
  OR-expression (zenodo/zenodo#2515), so no single spelling could satisfy
  every consumer; the metadata gate now checks both files stay present and
  consistent.

### Changed

- Estimate-manifest evidence honesty: rows whose conformance tests assert the
  fixture's synthetic `true_effect` (not the recorded DoWhy estimate) are now
  classified `internal_known_truth` with explicit limitations; only
  `estimate.linear_regression` (DoWhy, StableFloat) and `estimate.conditional`
  (statsmodels, atol 2e-9) claim `frozen_external_oracle` in that group.
- Provenance records for `estimate.aipw` and `estimate.response.kennedy_dr`
  now carry explicit `implementation_deviations` (no cross-fitting in AIPW;
  restricted nuisance families, Gaussian plug-in density, Silverman default
  bandwidth, and deterministic folds in the Kennedy curve).
- The three recorded general-ID oracles are declared recorded-but-unexercised
  everywhere they are cited; stale `causal`/`causal-library` identifiers and
  the "Format 0.1 frozen" / "CI does not run gates" statements corrected.

## [0.5.0] — 2026-08-15

Large causal-response release. Workspace and Python package versions are **0.5.0**.

### Added

- Function-valued continuous response queries, average/point/directional
  derivatives, elasticities, and low-dimensional Jacobians.
- Kennedy-style doubly robust response-curve estimation, Riesz average
  derivatives, explicit support diagnostics, and typed pointwise/simultaneous
  uncertainty semantics.
- Generic identified intervals/envelopes and sharp binary-IV Balke–Pearl
  ATE bounds via response-type enumeration (Rust + Python
  `antecedent.identify.binary_iv_bounds`). This is a contrast bound, not a
  continuous-response curve estimator.
- Explicit complete, censored, truncated, and selected observation mechanisms,
  with assumptions kept separate from the recorded-data process. Selected-outcome
  AIPW cross-fits both nuisance models over deterministic row-index folds
  (`crossfit_folds`, default 5), so no row's pseudo-value comes from a model that
  saw it; a fold that cannot support either model is refused rather than refit in
  sample. Selected-outcome IPW keeps its in-sample maximum-likelihood propensity,
  which is the published estimator for that correction. The AIPW method id is
  therefore `observation.selected.crossfit_logistic_aipw.v1`.
- Single-source selection diagrams, a sound certified subset of graphical
  transport identification, and trial-to-target IPW/AIPW estimation. Transported
  estimation requires the identification certificate: `trial_to_target_effect`
  and `transport_augmented_response_grid` take it as an argument and refuse a
  `NotCertified` result rather than returning a number the graph never licensed.
  Recursive-factorization certificates are also refused by those estimators
  (Dahabreh-style direct/standardize algebra only).
- Randomized interference queries decomposed into assignment design, exposure
  mapping, and exposure contrast, with exact/seeded-Monte-Carlo probabilities
  and Horvitz–Thompson/Hájek estimation.
- Response artifact format 0.3 with migrations from formats 0.1 and 0.2.
  Partially identified responses store a coordinate-free envelope, so the
  identified set of a scalar or vector functional is representable without a
  later format bump. An envelope is rejected under any other identification
  status, and Jacobian envelopes remain unrepresentable by design (see
  [ADR 0019](adr/0019-response-artifact-format.md)).
- Paper-level machine-readable provenance records and a 0.5 parity inventory.

### Fixed

- **Transport estimators refuse recursive-factorization certificates.**
  Dahabreh-style `trial_to_target_effect` / `transport_augmented_response_grid`
  previously ran for any `Transportable` result, including
  `RecursiveFactorization`. They now accept only `Direct` and `Standardize`.
- **Gaussian treatment residual scale uses GAM effective df.** Kennedy densities
  and Riesz scores no longer divide RSS by `n−1` when the treatment nuisance is
  a penalized GAM.
- **Unconverged additive GAM nuisances are refused** after an extended
  backfitting budget, matching the observation-path `require_ok` posture.
- **Bernoulli/Categorical intervention responses use exact finite mixtures**
  rather than Monte Carlo through a continuous spline.
- **Delayed-entry KM IPCW is `G(L−)/G(T−)`**, not `1/G(T−)`. The observation
  primitives fixture was regenerated for this correction.
- **Exposure HT means refuse any unit with `π_i = 0`** for the requested level,
  so impossible exposures cannot silently dilute the `/n` average.
- **Point derivatives require an explicit bandwidth**; Silverman's rule is
  refused for `m'`/`m''` (same undersmoothing discipline as simultaneous bands).
- **Finite-difference steps scale with `|a|`** so large treatment levels cannot
  collapse central differences to exact zero.

### Changed

- **`Identification.validate` and the module-level `validate` now default
  `refute` to `"cheap"` instead of `None`.** Previously an unset `refute` on
  `.validate(data)` resolved to a mode-dependent default suite chosen
  internally; it now always runs the explicit `"cheap"` suite. Pass
  `refute=False` for no refutation, or `refute="full"` / `"placebo"` for a
  different suite.
- PAG response envelopes emit an explicit warning that bounds are the
  pointwise min/max of completion-specific **point** curves (structural
  uncertainty only; not a confidence band).
- Transported estimation docs state that recursive factorization remains
  identify-only in 0.5.

### Compatibility

- Existing 0.4 query positional conventions are preserved: scalar response
  queries use `(treatment, outcome)`, with options keyword-only.
- New specialized observation, transport, and interference types live in stage
  namespaces. Only day-one response query types are added to the Python root.
- Cyclic/equilibrium models and multi-source meta-transport are not part of 0.5.

## [0.4.1] — 2026-07-30

Patch release. One correctness fix on the identify-only path; no API removals,
no behaviour change for existing callers.

### Fixed

- **`identify()` accepts an `Admg`, so latent confounding can fail honestly.**
  `identify()` previously took only a `Dag` or an edge list. A `Dag` has no way
  to express that a variable is unobservable, so a latent common cause flattened
  into one was treated as an ordinary adjustable node: on `T <- U -> Y` with `U`
  unmeasured, identification returned `NonparametricallyIdentified` with
  adjustment set `["U"]` — an effect reported identified by adjusting on
  something no study can measure. `analyze()` already accepted an `Admg` and got
  this right; only the identify-only path was missing it, so callers who
  identified before estimating got the wrong answer with no signal.

  `Study::identify_only` now mirrors `execute()`'s ADMG handling: a graph with
  bidirected edges routes through general ID (the only identifier that reasons
  about bidirected structure — the default is a backdoor strategy that would
  ignore it), and a graph without them is coerced to a DAG, which keeps an
  explicitly requested identifier meaningful rather than forcing general ID on a
  graph with no latent structure to reason about.

  Build `Dag.latent_project(observed)` to get the ADMG for a graph whose
  unobserved variables you want identification to respect.

### Added

- `identify_ate_admg` PyO3 binding and its typed stub.
- `identify()` and `Identification.graph` widened to `Dag | Admg | Sequence[...]`
  on both the staged and one-shot Python wrappers.

### Compatibility

`Dag` and edge-list inputs take exactly the same path as before. Code that was
passing a `Dag` containing a variable it treats as latent will now get a
different — and correct — answer once it switches to `latent_project`.

## [0.4.0] — 2026-07-30

API-surface freeze on both languages, plus an algorithmic correctness pass.

1. **Rust facade restructured**: `CausalAnalysis` → `Study`, discovery-to-estimation
   staged behind `AcceptedGraph`, `identify()` without data, estimators configured via
   `EstimatorSpec` and `with_*` setters, `IdentifierId`/`EstimatorId` closed and
   validated at parse time, one error hierarchy workspace-wide.
2. **Python facade frozen**: root namespace 206 names → 41 around three verbs, with
   everything else on stage modules.
3. **25 correctness defects fixed**, several changing returned values. See *Correctness*.

The migration policy on both surfaces is a **silent hard break** — no deprecated aliases,
no shims. Read the breaking-changes section before upgrading, and *Correctness* before
comparing numbers against 0.3.x: some differences are intended corrections.

A narrative version of this entry, aimed at readers deciding whether to adopt or
upgrade, is in [`docs/release-notes/v0.4.0.md`](docs/release-notes/v0.4.0.md).

### Why this release

The Python root namespace had grown to 206 names with no rule for what belonged
there — free `discover_*` functions, free `dag_from_*`/`dag_to_*` helpers, and
`target_*` population builders all lived flat at `import antecedent` alongside the
three verbs callers actually reach for every time. This release freezes the root at
41 names organized around **three verbs** (`analyze` / `identify` / `estimate`) and
moves everything else onto **stage modules** (`antecedent.discovery`,
`antecedent.priors`, `antecedent.attribution`, …) reached by module path. See
`docs/api_naming.md` for the full shape and the rule for where a name lives.

### Breaking changes

- **Root namespace frozen from 206 names to 41.** If a name you imported from
  `antecedent` directly is missing, it moved to a stage module. Read
  `docs/api_naming.md` for the complete list; the twelve stage modules are
  `antecedent.attribution`, `.data`, `.design`, `.discovery`, `.errors`,
  `.estimation`, `.extensibility`, `.gcm`, `.graph`, `.priors`, `.state`,
  `.validation`. A further five modules are reachable but intentionally outside
  `__all__`: `.counterfactual`, `.inference`, `.model`, `.population`, `.query`.

- **The 16 free `discover_*` functions are gone.** They are replaced by methods on
  the discovery config dataclasses in `antecedent.discovery`:

  ```python
  # before
  result = antecedent.discover_pc(data, alpha=0.05)

  # after
  result = antecedent.discovery.PC(alpha=0.05).run(data, seed=1)
  accepted = antecedent.discovery.PC(alpha=0.05).accept(data, seed=1)  # -> AcceptedGraph
  ```

  This covers `discover_pc`, `discover_ges`, `discover_lingam`, `discover_notears`,
  `discover_fci`, `discover_rfci`, `discover_pcmci`, `discover_pcmci_plus`,
  `discover_lpcmci`, `discover_jpcmci_plus`, `discover_rpcmci`, and the
  graph-posterior family (`discover_exact_dag_posterior`, `discover_order_mcmc`,
  `discover_structure_mcmc`, `discover_ci_screened_posterior`,
  `discover_dbn_posterior`) — now `ExactDagPosterior`, `OrderMcmc`, `StructureMcmc`,
  `CiScreenedPosterior`, `DbnPosterior`.

- **The 10 free `dag_from_*` / `dag_to_*` helpers are gone.** Use the class methods
  instead: `Dag.from_dot` / `Dag.to_dot` and the JSON / GML / NetworkX peers,
  likewise on `Cpdag` / `Pag` / `Admg`. These constructors now **preserve variable
  names** through a round-trip; earlier versions discarded them.

  ```python
  # before
  dag = antecedent.dag_from_dot(text)
  text = antecedent.dag_to_dot(dag)

  # after
  dag = antecedent.Dag.from_dot(text)
  text = dag.to_dot()
  ```

- **`target_*` helpers moved off the root** into `antecedent.population` as typed
  dataclasses (`AllRows`, `Treated`, `Untreated`, `Named`, `Rows`,
  `CustomDistribution`), with `target_all()` / `target_treated()` / … kept as
  constructor functions in that module (not at root) that return the new
  dataclass instances rather than a raw wire dict.

  ```python
  # before
  pop = antecedent.target_all()

  # after
  pop = antecedent.population.target_all()          # same shape, new location
  pop = antecedent.population.AllRows()              # or construct the dataclass directly
  ```

- **Queries are keyword-only after their identifier prefix**, and every query
  dataclass now uses `slots=True`. `AverageEffect("t", "y")` still works;
  `AverageEffect("t", "y", 0.0, 1.0)` (passing `control_level`/`active_level`
  positionally) is now a `TypeError`. Check each query class in
  `antecedent.query` for its own positional prefix (e.g. `MediationEffect` takes
  `treatment, outcome` positionally and requires `mediators=` by keyword).

- **`refute=True` now raises `TypeError`.** It never said *which* refutation suite
  to run, so it silently fell back to a mode-dependent default. `refute=False`
  still means no refutation, and leaving `refute` unset still runs the default
  suite exactly as before — only the literal `True` is now rejected. Spell it
  `refute="placebo"`, `"cheap"`, `"full"`, or a `Refute` enum member.

  ```python
  # before
  analyze(..., refute=True)

  # after
  analyze(..., refute="placebo")   # or "cheap" / "full" / Refute.PLACEBO
  analyze(...)                     # unset: same default suite as before
  ```

- **Module renames**: `prior_bank.py` → `priors.py` (`antecedent.prior_bank.*` is now
  `antecedent.priors.*` — `PriorCatalog`, `compose_external_priors`, `ComposedPrior`,
  `beta_from_moments` / `beta_from_mean_and_ess`, `gamma_from_moments` /
  `gamma_from_mean_and_ess`, all live there); `_analyze_handlers.py` → `_analyze.py`
  (private module, no import-path impact for callers using the public `analyze`).

- **`antecedent.identify` is the `identify()` function, not a module.**
  `__init__.py` rebinds the `identify` attribute to the function, so
  `import antecedent.identify as m` binds the function (per the language's
  `import a.b as c` ≡ `import a.b; c = a.b` rule), not the underlying module. Use
  `from antecedent.identify import Identification, validate, IdentifyResult` if you
  need something from that module `identify()` itself doesn't re-export.

- **`identify()` at the root now returns a staged `Identification`, not
  `IdentifyResult` directly.** `Identification` supports `.estimate(data)` /
  `.validate(data)` / `__bool__` (true when the estimand is identified) and can be
  downgraded via `.to_identify_result()`. The one-shot call that still returns
  `IdentifyResult` directly is `antecedent.estimation.identify(...)`, unchanged.

- **`CausalAnalysis` renamed to `Study`** (Rust; `2623654`). The facade struct,
  its builder, and its result type are now `Study` / `StudyBuilder` /
  `StudyResult` — construct via `Study::tabular(data)` (or `::series` /
  `::series_multi` / `::panel` / `::events`), not `CausalAnalysis::builder()`.
  Rust callers must update the type name at every reference. **Python is
  unaffected**: the PyO3-exposed class name (`PreparedAnalysis`) was
  deliberately left unchanged by this rename.

- **`IdentifierId` / `EstimatorId` closed** (`9b20f3e`). The `Other(Arc<str>)`
  escape hatch that let an unrecognized identifier/estimator name defer
  validation to first use is gone; an unrecognized name is now rejected at
  **parse time**, not later when the analysis runs. This reaches Python:
  `python/src/ate_api.rs` and `python/src/prepared_api.rs` construct these ids
  from caller-supplied strings, so a bad `identifier=` / `estimator=` string
  kwarg to `antecedent.analyze(...)` now raises immediately instead of
  surfacing as a deferred failure.

- **`AcceptedGraph` staging removals** (Rust; `2e41568`). The new
  `AcceptedGraph.asserted()` / `.accepted()` classmethods and `.review(...)`
  workflow (see Added, below) replace types and methods that are now
  **removed**: `GraphInput`, `CompiledAnalysis`, `DiscoveryAccept`, the
  `StudyBuilder::discover_*` methods, the setters they fed, and the
  `finish_*_review_and_run` continuations. The deprecated `AnalysisError`
  alias is also removed — use `CausalError`.

- **`EdgeEvidence.separating_sets` renamed to `separating_set`** (`0c44ed3`).
  A breaking change to a public field on a public struct; per the commit, no
  deprecated alias is added. **Rust-only** — this field is not touched by any
  Python binding file, so Python callers are unaffected.

- **`ReiszSensitivity` renamed to `RieszSensitivity`** — a spelling correction.
  Frigyes Riesz's name was misspelled throughout the public surface. This renames
  the re-exported type, the `antecedent_validate::reisz` module (now `riesz`), and
  the `ValidatorId::Reisz` variant (now `ValidatorId::Riesz`). Per this release's
  hard-break policy, no deprecated alias is added. **Rust-only** — the name was
  never exposed through any Python binding, so Python callers are unaffected.

  The **emitted refuter identifier also changes**, from `"sensitivity.reisz"` to
  `"sensitivity.riesz"`. This appears in the `refuter` field of refutation reports
  and therefore in serialized analysis artifacts, so it is a data-format change as
  well as an API one: code that matches on the old string literal, or that compares
  new artifacts against ones stored by 0.3.x, must be updated. Nothing inside the
  library matches on the value — it is written and round-tripped opaquely — so no
  migration is required for library behaviour itself. The conformance fixture
  directory `conformance/validate/reisz_sensitivity` and the
  `validate.reisz_sensitivity` parity id are renamed to match.

- **Unified error model** (`b9c232d`). Rust and Python errors are
  consolidated onto one hierarchy — `CausalError` (Rust) and `CausalError`
  plus typed subclasses (Python) — gathered in the new `errors.py` module
  (see Added, below) instead of being scattered across call sites. Code that
  matched on a previously scattered / ad hoc exception type should catch
  `CausalError`, or the specific typed subclass, instead.

### Added

- **`estimator_config=` dict kwarg on `analyze()`** for per-estimator tuning
  (standard-error kind, cluster/multiway/panel ids, GLM options, linear fit kind,
  caliper, `n_strata`, and the `rd.sharp` triple) without a bespoke Python kwarg per
  estimator. Unknown keys, and keys that belong to a *different* estimator than the
  one resolved for the call, are hard errors that name the offending key. See
  `python/src/estimator_config.rs` for the full estimator-id → valid-keys table and
  `python/tests/test_estimator_config.py` for worked examples.
- **`AcceptedGraph.asserted()` / `.accepted()`** classmethods (documented spellings
  for `.from_graph()` / `.from_discovery()`, which remain as thin aliases), a
  `.pending` tuple of unreviewed edges, `.review({edge: mark})` returning a
  version-bumped instance, and `__len__` / `__iter__` / `__contains__` / `__repr__`.
- **Richer `AnalysisResult` display**: `__repr__` on every result view, and
  `_repr_html_` for notebook display (compact verdict banner, effect summary,
  adjustment-set chips, refutation table, with an amber callout when
  `unidentified_mass > 0`). `ValidationView` supports `len()`, iteration,
  `[index]` / `["refuter_name"]`, `.failed`, and `.to_pandas()`. `PosteriorView`
  supports `np.asarray(result.posterior)` (requires
  `return_posterior_artifact=True`) and `.interval(level=0.95)`.
- **`ReviewRequired`** is a real exception class (subclassing the native
  `CausalReviewError`) carrying a structured `pending_edges: tuple[PendingEdge, ...]`
  list — not just a count — plus `kind`, `algorithm`, `pending_edge_count`, and
  `hint`. Exported at the root and as `antecedent.errors.ReviewRequired`.
- New modules: `errors.py` (the exception surface, previously scattered), `_coerce.py`
  (the five input-normalization functions every public entry point now funnels
  through), `identify.py` (the staged `identify()` / `Identification` described
  above), and the `results/` package (the view dataclasses plus `_repr_html_`
  rendering).

### Correctness

An audit of the algorithmic and numerical core fixed 25 confirmed defects. Each has a
regression test that was verified to fail against the previous code. Grouped by what a
caller would have observed.

**Results that were silently wrong**

- *Path-specific identification could be unearned.* `Dag::directed_paths` truncated at
  `max_paths` / `max_len` and returned `Ok` with no signal, but the recanting-witness
  criterion concludes identifiability from the *absence* of a witness — so a witness on a
  dropped path was invisible and the effect could be reported
  `NonparametricallyIdentified` against Avin–Shpitser–Pearl. Added
  `directed_paths_with_budget`; identification now fails closed on truncation.
- *Generalized adjustment over a PAG overclaimed.* `CompletionSampler` yields a
  deterministic low-mask prefix, not a sample. With the default cap the envelope examined
  32 of 729 valid MAG completions on a 7-node chain and reported `unidentified_weight = 0`
  with no truncation signal. It now reports the cap and downgrades
  `NonparametricallyIdentified` to `PartiallyIdentified` rather than asserting a class-wide
  property from a prefix.
- *A non-mixing graph-MCMC chain published as converged.* Identical edge traces yield
  R̂ = 1.0 and ESS = `n_chains·n_draws`; `all_chains_moved` is the only field that separates
  that from real convergence, and the gate computed it without reading it.
- *G² p-values drifted at large df.* `gamma_p_series` capped iterations at a fixed 500 with
  no convergence check, returning a partial sum: 0.739 against SciPy's 0.500 at df = 10⁶,
  with error appearing from ~10⁴. Now scales as `500 + 10√a`.
- *Binary and parentless variables were fit as point masses.* Mechanism selection scored
  families on conditional-mean fit, where `Constant` is unbeatable, so it won for every
  root and every binary column — sampling a binary treatment returned its mean. Selection
  now gates `Constant` on the claim it makes (that the target is deterministic).
- *`DbnPosterior` ignored lagged constraints.* `edge_forbidden` builds its `LaggedLink` with
  both endpoints pinned to lag 0, so a forbidden `X_{t-1} → Y_t` was a no-op and carried
  0.99999 posterior mass. Added lag-aware constraint checks.
- *RPCMCI regime reassignment fit nothing.* It used the raw parent value as its prediction —
  an implicit unit coefficient — so regimes sharing a link structure scored identically and
  every row collapsed into one. Now fits per-regime OLS.
- *LPCMCI could converge early*, before the X side had exhausted its conditioning pool.
- *PCMCI+ conditioning truncation evicted the mandatory MCI control set* in favour of
  optional contemporaneous candidates.
- *NOTEARS reported a nonzero gradient at a stationary point*: the L1 subgradient used
  `signum(0) = +1`, driving coordinates that belong at zero off it.
- *Distribution-change attribution collapsed players to a point mass*, which also made the
  mechanism swap dead code.
- *Sensitivity analysis was calibrated against the wrong variance.* The grid is documented
  as partial R² but was scaled by the marginal SD, so a nominal 0.2 realized as 0.556 when
  covariates predict the treatment.
- *Four CI tests accepted `BlockShuffle`'s `block_size` and permuted exchangeably*,
  under-dispersing the null for the autocorrelated data the parameter exists to protect. On
  AR(1) data at φ=0.7 that inflated Type I error to 0.20 against a nominal 0.05. Each now
  either honours the parameter or refuses it, and which one is no longer a matter of opinion:
  GPDC honours it because it residualizes on Z before permuting; `GSquared`, `KnnDependence`
  and `SymbolicCmi` honour it when the conditioning set is empty, where their strata collapse
  to a single time-ordered block. With a conditioning set those three refuse, because
  preserving `Y|Z` needs permutation within index sets scattered across time while preserving
  serial dependence needs contiguous runs — a structural limit, not a missing feature, and the
  error says so. Two scheduled calibration gates hold the honoured paths to nominal.

**Statistics that did not mean what they were called**

- *Anomaly scores* were a negative log density presented as an information-theoretic score.
  A density is unbounded, carries units, and moves under rescaling — multiplying the outcome
  by 100 shifted every score by exactly `ln(100)`. Now a tail probability.
- *Arrow strength* was `|β|`, ignoring parent variance, and ranked a parent that barely
  moves above one that dominates. Now `β²·Var(parent)` (Janzing et al. 2013), with variance
  propagated from the model in topological order.
- *Mechanism-change detection* had no multiple-testing correction across targets (~40%
  family-wise false-positive rate at ten targets). Benjamini–Hochberg is now applied by
  default, with raw and adjusted p-values reported separately.
- *ESS* was floored at 1 and clamped to N, unlike Stan/ArviZ, understating well-mixed
  chains; the publication gate used a flat 100 rather than the per-chain convention.
- *GAM `edf_approx`* double-counted the constant, one per smooth term.
- *`WeightingDoSampler`* ran self-normalized importance sampling with no ESS diagnostic.
- *`mcmc_graph_diagnostics`* reported a fabricated `mean_accept_prob = 1.0`.
- *`normal_ppf`*, *`regularized_incomplete_beta`*, and *`ln_gamma`* accuracy and domain
  handling at the tails and poles.

**Checks that existed but never ran**

- Simulation-based calibration and posterior interval coverage were implemented, exported,
  and had zero call sites. Both are now wired into `scripts/gate_calibration.sh`, alongside
  a new coverage test for the sharp-RD analytic standard error.
- SBC gated on the rank mean while ignoring its own uniformity statistic, which passes the
  U- and M-shaped rank distributions SBC exists to catch.
- Predictive checks reduced to the predictive mean, so a model with the right mean and
  5× wrong variance passed. A dispersion axis now participates in the verdict.
- Conditional sampling guarded only hard-set conditioning nodes, returning a biased estimate
  for soft/stochastic overrides instead of an error.

**Behaviour changes that follow from the above**

- Counterfactual abduction reports `PosteriorNoise` rather than `Invertible` on any model
  containing a discrete node, because discrete mechanisms are no longer mis-fit as
  `Constant`.
- The propensity caliper is interpreted on the **logit** scale by default, which is what the
  0.2 rule of thumb (Rosenbaum–Rubin 1985; Austin 2011) means. Pass
  `caliper_scale=CaliperScale::Raw` for the previous behaviour.
- Anomaly scores and arrow strengths are on new scales. Rankings are more trustworthy;
  absolute values are not comparable to 0.3.x.
- `MechanismChangeDetection` gains `adjusted_p_value`; `ArrowStrength` gains `coefficient`;
  `DoSampleResult` gains `ess`.

### Known limitations

- **Scope, not maturity.** Antecedent does not implement double machine learning, ML-based
  heterogeneous effect estimation, PAG-native full ID/IDC, or unsupervised regime discovery.
  These are documented boundaries — see [`docs/comparison.md`](docs/comparison.md) — not
  work in progress.
- The graph-MCMC publication gate uses a looser R̂ bound (1.2) and ESS floor than the HMC
  gate, because binary edge indicators mix more slowly than continuous parameters. Both now
  require every chain to have moved.
- The `gaussian-process` mechanism family remains behind a non-default cargo feature: its
  fit path uses a fixed hyperparameter grid with no gradient search, and it is documented as
  experimental rather than enabled.
- `JPCMCIPlus` and `RPCMCI` raise `CausalUnsupportedError` rather than producing an
  `AcceptedGraph`; use their per-regime results directly.
- No deprecation shims exist for any renamed or removed spelling in this release —
  see "Breaking changes" above for the complete migration list.
- CodeQL reports import cycles in the Python package. Investigated and confirmed a false
  positive: the hazard those rules describe needs a cycle among **module-scope** imports,
  and this package's module-scope graph is acyclic. The reported cycles appear only when
  function-body imports (which run after initialisation) and `TYPE_CHECKING` imports (which
  never run) are counted alongside module-scope ones. `python/tests/test_import_graph.py`
  now enforces that — it fails if the module-scope graph gains a cycle, and imports every
  module first in a fresh interpreter to prove no order-dependence — so the two rules are
  excluded against an enforced invariant rather than a comment. Rust runs the full
  security-and-quality suite with zero findings and no exclusions.

### Feedback we want

- Any Python facade name you relied on that this changelog's breaking-changes list
  doesn't cover.
- Places where `docs/api_naming.md`'s "rule for where a name lives" doesn't
  actually predict where something ended up.

File issues at
[github.com/iridae-dev/antecedent](https://github.com/iridae-dev/antecedent)
with the old import/call you used and what replaced it (or didn't).

## [0.3.0] — 2026-07-25

Second correctness cut after `0.2.0`, plus a maintainability pass on the
facade and Python bindings. Re-running analyses can still change discovery
graphs, structure-MCMC weights, GP fits, hierarchical GLM shrinkage, WLS /
Newey–West SEs, and whether Bayesian designs publish — treat the upgrade as
a behavior bump.

### Why this release

`0.2.0` closed sandwich / AIPW / HMC publication gaps. This release closes
the next ledger: **order-invariant GES**, **Tigramite-aligned LPCMCI**,
**Order MCMC topological weights**, GP / HierarchicalGlm / design math
edges, and a frozen **oracle fixture** suite so regressions fail in CI
instead of in production.

### Added

- Frozen **oracle / conformance fixtures** across stats, CI, graph /
  identification, discovery (including MCMC), estimation, GCM /
  counterfactuals, attribution / validation, and design / state / priors,
  with regenerated conformance docs.
- Local Python **ruff + mypy** gate (`scripts/gate_python_lint.sh`) wired
  into the binding split work.

### Fixed

#### Discovery

- **GES CPDAG search** is order-invariant (variable permutation no longer
  changes the returned equivalence class for the same score).
- **LPCMCI** aligned to the pinned Tigramite reference behavior (MM-009).
- **Order MCMC** weights by topological-order count (MM-011).
- **GPDC** residualization documented / contracted as centered Cholesky
  (MM-008).
- **ParCorr** and related CI / attribution math edges hardened against
  nonfinite and degenerate inputs.

#### Estimation / models

- **GP** hyperparameters selected with exact Cholesky NLML (MM-015);
  unused dense GP solver helper removed.
- **HierarchicalGlm** empirical-Bayes ridge applied on ordinary (unscaled)
  data (MM-014).
- **Lasso** uses an explicit intercept and standardization contract.
- **WLS** rejects negative and nonfinite weights.
- **GLM diagnostics** evaluated at the returned coefficients (not a stale
  working fit).
- Scalar **Newey–West** SEs share Bartlett helpers with panel / IF paths.
- **Front-door** SEs stack correctly; **GAM** roughness penalties applied
  as intended.

#### Bayesian / design / decision

- Invalid Bayesian **priors and designs fail closed** (no silent publish).
- **HMC** hardened further for the publication gate (clippy/fmt + sampler
  edges).
- Stable logistic coverage for extreme log Bayes factors.
- Design: keep **signed EIG** draws without pointwise clipping (MM-012);
  return `None` when no decision action is feasible (MM-013).

### Changed

- **`CausalAnalysis` execute path** split into modality modules
  (`static` / `bayesian` / `temporal` / `panel` / `pag` / `attribution`)
  instead of a single multi-thousand-line dispatcher — same public facade
  API, clearer ownership per path.
- Discovery **orientation rules** unified on a generic
  `OrientationRule<G: CpdagOps>` (static and temporal CPDAG ops share one
  trait shape).
- Graph JSON helpers in `antecedent-io` DRY’d via shared macros.
- **Python / PyO3 bindings** modularized (`ate_api`, `discovery_api`,
  `temporal_api`, `attribution_api`, `graph_build`, analyze handlers) with
  a shared graph-construction and exception bridge so Rust and Python
  raise the same error kinds for schema / name resolution.

### Known limitations

- Same experimental surfaces as `0.2.0` (graph MCMC gate looser than HMC;
  discovery / temporal / attribution still evolving).
- Artifact container format remains
  `FormatVersion { major: 0, minor: 2 }` (no container bump in this
  release).
- `0.3.x` may still introduce breaking API or behavior changes before a
  1.0 freeze.

### Feedback we want

- Discovery graphs that **flip under column permutation** after upgrade
  (should be rare now for GES; report if not).
- Structure-MCMC or LPCMCI outputs that disagree with a pinned Tigramite /
  reference notebook on the same seed and data.
- Bayesian / design runs that **used to publish and now error** — was the
  refusal correct?

File issues at
[github.com/iridae-dev/antecedent](https://github.com/iridae-dev/antecedent)
with a minimal repro.

## [0.2.0] — 2026-07-24

Antecedent is an identification-first causal engine for Rust and Python: it
runs discovery → identification → estimation → diagnostics under **structural
uncertainty**, instead of conditioning every number on a guessed DAG.

`0.2.0` is a **correctness cut**. The day-1 surface from `0.1.0` is still
there (`cargo add antecedent` / `pip install antecedent`), but several
estimators, sandwich/HAC paths, special functions, and Bayesian / MCMC
routes now match the formulas they claimed. Re-running `0.1.0` analyses can
change point estimates, SEs, CI widths, ESS, and whether a fit publishes at
all — treat the upgrade as a behavior bump, not a silent patch.

### Why this release

Causal pipelines fail quietly when the math under the API is wrong: an ATC
AIPW that is not doubly robust, a Student-t tail that is not one-sided, an
ESS that looks fine while chains disagree, or a cluster SE that returns
`NaN` and still looks like success. This release closes those gaps and
**fails closed** where degrees of freedom or diagnostics cannot support a
published number.

### Stable enough to build on

- Facade workflow: `CausalAnalysis` / Python `analyze`, identification before
  estimation, assumption tracking across the stack.
- Core frequentist estimation under explicit overlap policy (linear
  adjustment, IPW, AIPW, matching, common sandwich kinds).
- Artifact container format (`FormatVersion { major: 0, minor: 2 }`) and
  provenance-oriented IO.
- Rust + Python dual surface with shared semantics.

Semver is still `0.x`: the API can move, but the contracts above are what we
intend to keep honest.

### Experimental / evolving

- Graph-posterior / structure MCMC publication (intentionally looser gate
  than HMC).
- Broad discovery surface (PCMCI-family, score-based, continuous
  optimization) and temporal online `CausalState`.
- Attribution, experimental design, and some sensitivity / refute helpers.
- Public fields on many result types (prefer constructors; getters are
  unfinished API debt).

### Fixed

#### Special functions

- **Trigamma reflection** now uses
  `ψ₁(z) = π²/sin²(πz) − ψ₁(1−z)` (the previous branch added
  `ψ₁(1−z)`).
- **Student-t survival** is one-sided: for `t < 0` the incomplete-beta
  half-tail is reflected so `SF(−t) = 1 − SF(t)`.

#### Causal estimation (IPW / AIPW / overlap)

- **ATC AIPW influence function** no longer multiplies the control
  residual by a spurious `1/(1−e)`. Double robustness under correct
  propensity and misspecified `μ₀` is restored.
- **IPW clip-sensitivity** rebuilds estimand-specific observed-arm
  weights (ATE / ATT / ATC) instead of a generic `1/e + 1/(1−e)`
  diagnostic.
- Clip-sensitivity diagnostics use the **same clip bounds as production**
  `clamp_scores` (no artificial `[1e-6, 0.49]` remap of the applied
  clip) and multiply **`CustomDistribution` observation weights** into
  the grid so ESS matches published IPW weights.

#### Cluster / multiway / panel standard errors

- **Multiway CGM** uses full inclusion–exclusion with correct signs,
  **exact cluster-tuple interning** (no lossy packing collisions), and
  shared meat between influence-function and sandwich paths.
- **Panel cluster HAC** uses explicit `(cluster, time)` labels,
  within-unit Bartlett lags only, and **`lag = 0` equals one-way
  cluster / Arellano meat** (not White `Σ u²` with cluster DF).
- Series Newey–West Bartlett weights use
  `L_eff = min(lag, T−1)`, matching panel / IF helpers.
- **Fail closed** on fewer than two clusters, non-positive residual DF,
  missing panel times, and **materially negative multiway IE meat**
  (scalar and matrix). Invalid multiway IF inputs error instead of
  returning `Ok(NaN)`.

#### Bayesian inference / MCMC / Laplace

- Coefficient priors require **finite means and variances**; `PriorSet`
  rejects duplicate coefficient priors and simultaneous known-σ² /
  InvGamma residual specs.
- Laplace / HMC / conjugate designs reject nonfinite `X` / `y` /
  offsets, negative or nonfinite weights, non-binary Bernoulli
  outcomes, and negative Poisson outcomes. Zero observation weights
  are allowed (row dropped) when total weight mass is positive.
- **Gaussian HMC** targets a fixed prior-driven posterior: known `σ²`
  or joint `(β, log σ²)` under InvGamma — no state-dependent plug-in
  residual variance that broke detailed balance.
- **Gaussian / InvGamma Laplace** routes through the same prior-driven
  targets as HMC; InvGamma adaptive draws project to coefficients
  correctly.
- **MCMC publication gate** requires energy-error divergence tracking,
  chain movement, rank/folded R̂, and Geyer bulk **and** tail ESS
  (stuck or divergent chains no longer publish).
- **Geyer ESS** uses Stan/Vehtari multi-chain
  `ρ̂_t = 1 − (W − ā_t)/var̂⁺` and `τ̂ = −1 + 2 Σ P_t` (disagreeing
  chains no longer report ESS ≈ N).
- Shared **Poisson / probit** log-likelihood, score, and observed
  Hessian primitives for Laplace and HMC.
- **NIG Bayesian CI / Bayes factors / PPC** use the full
  Normal–inverse-gamma design marginal (not residual-vector shortcuts).

### Changed

- `OverlapReport::from_propensities` takes an additional
  `observation_weights: Option<&[f64]>` argument (pass `None` unless
  using `CustomDistribution`).
- `AnalyticSeKind::PanelClusterHac` requires `panel_times` on the
  estimator.
- Published MCMC / Laplace results that previously cleared a looser
  gate may now be rejected until diagnostics pass.

### Known limitations

- InvGamma Laplace is still a **local Gaussian** on `(β, log σ²)`, not
  the exact Student-t / NIG marginal.
- Graph MCMC diagnostics are **not** as strict as the HMC publication
  gate (by design, for now).
- Acklam `norm_inv` is duplicated in kernels and stats (interior math
  agrees; drift risk if only one copy is edited).
- Many result structs still expose public fields; cross-crate code should
  prefer constructors.
- `0.2.x` may still introduce breaking API or behavior changes before
  a 1.0 freeze.

### Feedback we want

If you upgrade from `0.1.0`, we especially want reports of:

- analyses that **used to publish and now error** (cluster DF, panel
  times, MCMC gate) — was the refusal correct for your data?
- **numerical deltas** on ATC AIPW, multiway / panel SEs, or Bayes CI /
  ESS that look wrong relative to a trusted reference
  (Stan, sandwich packages, textbook IF).
- missing fail-closed cases (NaN / zero SE / silent success) that still
  slip through.

File issues at
[github.com/iridae-dev/antecedent](https://github.com/iridae-dev/antecedent)
with a minimal repro (data shape, estimand, SE kind / sampler settings).
Correction reports beat feature requests for this release cycle.

## [0.1.0] — 2026-07-23

First crates.io-oriented release of the Rust library graph.

### Added

- Day-1 facade crate **`antecedent`** (`use antecedent::prelude::*`).
- Supporting crates published as **`antecedent-*`** (`antecedent-core`, …).
- Workspace publish metadata (repository, homepage, docs.rs, keywords, categories).
- `scripts/publish_crates.sh` and tag-driven `.github/workflows/publish-crates.yml`.
- `#[non_exhaustive]` on key public result / config types; sealed extension traits
  (`Identifier`, `Estimator`, `DiscoveryAlgorithm`, `Validator`).
- `FromStr` / `Display` for `IdentifierId` and `EstimatorId`.

### Notes

- **`0.1.x` may still introduce breaking changes.** Treat the release as a preview.
- Supporting libraries use **`antecedent-*`** names on crates.io and are **public
  dependencies** of `antecedent` (part of the semver surface). Day-1 usage is
  still only `cargo add antecedent`.
- The Python extension (`antecedent-py` / wheel `antecedent` on PyPI) is
  **not** published to crates.io (`publish = false`).
- `CustomEffectValidator` remains deliberately unsealed so host languages (PyO3)
  can implement the dyn-safe callback path.
- Known 0.1 API debt: many result structs still expose public fields rather than
  getters; prefer constructors (`::new` / `::from_parts`) for cross-crate builds.

[0.4.1]: https://github.com/iridae-dev/antecedent/releases/tag/v0.4.1
[0.4.0]: https://github.com/iridae-dev/antecedent/releases/tag/v0.4.0
[0.3.0]: https://github.com/iridae-dev/antecedent/releases/tag/v0.3.0
[0.2.0]: https://github.com/iridae-dev/antecedent/releases/tag/v0.2.0
[0.1.0]: https://github.com/iridae-dev/antecedent/releases/tag/v0.1.0
