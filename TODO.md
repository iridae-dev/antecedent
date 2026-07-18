# Remaining Work

Each numbered item is a **1.0 deliverable**. Check the parent box only when every required sub-item is complete, verified, and the matching `parity/*.toml` rows are `done`. Partial slices do not count.

Use `- [x]` / `- [ ]` for status. Do not remove or check off an item you just finished — leave notes until independently verified.

DESIGN.md is the roadmap; these are its unimplemented chapters. Parity inventories use `pending` / `in_progress` / `done` only. Unfinished chapters stay `pending` here and in `parity/*.toml` until the parent item is checked.

Out of scope for these DONE gates (per DESIGN): selection/transport ID as later modules; external Stan/PyMC adapters; unsupervised RPCMCI regime search; NOTEARS as optional extension.

Ordered foundations → dependents.

- [x] **1.** Thread `ExecutionContext.kernel_policy` through stats/CI call sites (2026-07-22) — G², distance correlation, weighted ParCorr, ParCorr mode / block-shuffle, `standardize_columns`, and data gather (`LaggedSamplePlan` / `SampleRequest` / `LaggedFrame` / pooled frames). Call sites with `ctx` pass `&ctx.kernel_policy`; free helpers take `policy: &KernelPolicy`. `PartialCorrelation` no longer stores a default policy. Escape hatch for §23.2 differential tests is reachable from CI/stats/data.

- [ ] **2. Deep identification** (DESIGN.md §10.1–10.3) — full semi-Markovian ID/IDC surface; highest scientific-correctness risk.
    - [ ] Missing `IdentificationStatus` variants: `IdentifiedUnderParametricRestrictions`, `IdentifiedUnderPriorRestrictions` (`crates/causal-core/src/identification.rs`).
    - [ ] ID algorithm for semi-Markovian models (Shpitser); memoized recursion over canonical subproblems; expression arena reuse (§10.5).
    - [ ] IDC for conditional interventional distributions.
    - [ ] Hedge certificates for non-identifiability.
    - [ ] `AutoIdentifier` that returns all valid estimands and selection rationale (no silent estimator choice); does not fork a second identifier for distribution queries.
    - [ ] Maximal (and remaining) adjustment-set search where defined beyond shipped backdoor / frontdoor / IV / RD / generalized adjustment (§10.4): cost-weighted selection, measurement-cost metadata, temporal history restrictions, positivity-aware ranking after a data check, and streaming enumeration (callbacks/iterators) so combinatorial sets need not be retained.
    - [ ] Sustained temporal-effect identification (g-formula / sequential ID) — `TemporalPolicy::Sustained` types, overlay, and wire already ship; `identify_temporal` is Pulse-only today (`temporal_backdoor` rejects Sustained).
    - [ ] Parity: `estimate.identify.general_id`, `pag.identify.full_id_idc` → `done`.

- [ ] **3. Distribution & path-specific pipelines** (DESIGN.md §8) — identify + estimate for shipped `CausalQuery::Distribution` / `PathSpecific`. Types, Unsupported plumbing, GCM sampling / path-contribution wrappers, and query wire exist; algorithms do not.
    - [ ] **Depends on item 2** (IDC / path-restricted ID in the same ID/IDC family — do not fork a second `AutoIdentifier`).
    - [ ] Interventional-distribution identification + estimation via IDC.
    - [ ] Path-restricted *natural effects* identification (path-restricted ID).
    - [ ] Nonparametric path-specific natural effects (`context.mediation.nonparametric` → `done`). Path *contribution* attribution already ships (`path_decompose` / `attribute_path_specific`).
    - [ ] Full `CausalQuery` CBOR model-bundle embedding for these queries via `causal-io` / `query_wire` where still incomplete.

- [ ] **4. Static (non-temporal) discovery** (DESIGN.md §13.3–13.6) — PCMCI family remains the temporal surface; static PC (4a–4b) now ships. Meek-rule and CI-test infrastructure already exist. Ship order below; parent stays unchecked until required sub-items are verified.
    - [x] **4a. Static `Cpdag` + Meek/collider** (2026-07-22) — Real static `Cpdag` (break `type Cpdag = TemporalCpdag` alias); `CpdagReview`; Meek R1–R4 + `OrientCollider` on static graphs. ContempMeek stays temporal-only. **Depends on:** shipped Meek on `TemporalCpdag`. Shipped: `causal_graph::Cpdag` / `CpdagReview`; `StaticOrientationRule` + `run_static_orientation_to_fixed_point`.
    - [x] **4b. Static PC** (2026-07-22) — Classic undirected skeleton + sepsets + collider/Meek orientation over `TabularData`; `Pc` algorithm type; Rust/Python `discover_pc`. Not PCMCI PC1. **Depends on:** 4a; shipped CI batch APIs. Soft input to item 6 (CI screening) and item 7 (`discovery=PC`). Shipped: `causal_discovery::Pc`, facade `discover_pc`, Python `PC` / `discover_pc`; parity `discovery.pc`. Parent item 4 stays unchecked until 4c–4g and 4i land (4h optional).
    - [ ] **4c. `Pag` FCI plumbing** — Public `Pag::remove_edge`; portability so FCI orientation rules target static `Pag` (today LPCMCI rules are `TemporalPag`-only). **Depends on:** shipped LPCMCI FCI-like rules.
    - [ ] **4d. Static FCI** — Possible-D-Sep adjacency phase; port R1–R4 / R8–R10 + discriminating / uncovered paths to `Pag`; classic FCI pipeline → static `Pag`. **Depends on:** 4b, 4c.
    - [ ] **4e. RFCI** — Early-stop / reduced Possible-D-Sep on top of FCI; `pag.discovery.fci_rfci` → `done`. **Depends on:** 4d.
    - [ ] **4f. GES / score-based DAG search** — Equivalence-class insert/delete/reverse operators using shipped Gaussian BIC `LocalScoreCache` (`causal-state`). Soft: 4b skeleton as screening. **Depends on:** shipped BIC score cache.
    - [ ] **4g. LiNGAM (DirectLiNGAM MVP)** — ICA / residual independence → causal order → `Dag`. Greenfield (no stubs). Independent of Meek/PC orientation stack.
    - [ ] **4h. NOTEARS (optional)** — Continuous acyclicity soft-constraint optimization; feature-gated. Not required for this item’s DONE gate (DESIGN §13.3).
    - [ ] **4i. CPDAG MEC / equivalence-class sampling** (DESIGN §6.5 item 15) — enumerate or sample DAG completions of a static `Cpdag` (today `try_into_dag` only when fully oriented). PAG `CompletionSampler` already ships. Soft dep of FCI/GES class-aware ID.
    - [ ] **4j. Pooled-frame time one-hot** — JPCMCI+ `DummyOptions` ships space one-hot + optional integer time-index dummy; high-dimensional one-hot of `T` remains deferred (`causal-data` `pooled_frame`).

- [ ] **5. Mechanism families and Bayesian inference gaps** (DESIGN.md §14.4, §15.4, §16, §18.4, §12, §23.2) — complete native Bayesian 1.0 beyond conjugate Gaussian + Laplace GLM. External Stan/PyMC adapters are **not** required (DESIGN §14.5).
    - [ ] Mechanism families: hierarchical linear/GLM, BVAR, linear Gaussian state-space, Gaussian-process mechanisms (`bayes.backend.hierarchical_bvar_gp` → `done`).
    - [ ] Native sampling backends needed for chain diagnostics (HMC and/or SMC) so ESS / R-hat / divergences are meaningful.
    - [ ] MCMC diagnostics: ESS, R-hat / convergence, divergence counts (`causal-prob` diagnostics; `bayes.validate.mcmc_diagnostics` → `done`).
    - [ ] Simulation-based calibration (SBC) and remaining §18.4 workflow diagnostics not already shipped: likelihood-family comparison and posterior calibration on synthetic SCMs (PPC / prior predictive / prior sensitivity already ship).
    - [ ] Bayes-factor CI, posterior dependence probability, and posterior-predictive CI diagnostics for supported conjugate models (`bayes.ci.tests` → `done`).
    - [ ] Counterfactual trajectories (§16) with shared-noise / batched evaluation (point/ITE/abduction paths that already exist stay; trajectories complete the subsystem).
    - [ ] Nested counterfactual expressions (§16) where identifiable under model assumptions (`allow_nested` exists; engine rejects true nested interventions today).
    - [ ] Conditional interventional sampling (§15.4) where supported (observational / hard / soft / stochastic / sequence / posterior-predictive sampling already ship).
    - [ ] Posterior draw reduction kernels (§23.2 deferred): shared scalar + portable reductions over posterior draw batches, wired through Bayesian estimation / PPC / SBC callers.

- [ ] **6. Bayesian graph discovery** (DESIGN.md §13.7) — additive to constraint-based discovery; graph-weighted effect envelopes over supplied `WeightedGraphSamples` already ship.
    - [ ] Wire documented `causal-discovery → causal-prob` dependency (absent today).
    - [ ] `GraphPosteriorEngine` trait + columnar/indexed `GraphPosterior` (weights, edge/orientation marginals, ESS, chain diagnostics, rejected invalid graphs).
    - [ ] Exact enumeration for very small DAGs.
    - [ ] Order MCMC and/or structure MCMC for discrete / small continuous models.
    - [ ] Candidate-edge posterior updates after CI screening (**uses** static PC from **item 4b** and Bayes-factor / posterior-dependence CI from item 5 as screening/proposal inputs).
    - [ ] Dynamic Bayesian network posterior search for bounded lag.
    - [ ] Parity: `bayes.discovery.dag_posterior` → `done`.

- [ ] **7. Python `analyze()` / bindings completeness** (DESIGN §25.3–25.4) — remaining gaps after PCMCI-family temporal `discovery=` and broader native exports.
    - [ ] **JPCMCI+ / RPCMCI one-shot**: wire `discover_jpcmci_plus` / `discover_rpcmci` through `analyze(discovery=…)` (multi-env columns / regime inputs on the OO path and `analyze_temporal_discover`). Rust facade builder methods already exist.
    - [ ] **Callback extensibility** (§25.4): explicit slow-path Python hooks for custom CI tests, mechanism wrappers, utility functions, and validators — GIL reacquire; plan marks callback regions as non-native-perf.
    - [ ] **Static `discovery=`**: end-to-end static discover→estimate on `analyze()` (`discovery=PC(…)` or equivalent). **Depends on item 4b** (static PC + `discover_pc`). Today `AverageEffect` still requires a supplied `graph=`.

- [ ] **8. Discovery validation** (DESIGN.md §18.3) — beyond shipped block-bootstrap link-frequency stability (`BlockBootstrapStability`). Effect refuters (§18.2) already ship.
    - [ ] Lag-window sensitivity.
    - [ ] Alpha-threshold sensitivity.
    - [ ] CI-test sensitivity.
    - [ ] Orientation stability.
    - [ ] Regime stability.
    - [ ] Environment holdout.
    - [ ] Synthetic-null calibration.
    - [ ] False-positive checks using permuted or phase-randomized data.

- [ ] **9. Query-model Planned variants** (DESIGN.md §8.1–8.2) — types/comments exist as roadmap; not the Distribution/PathSpecific algorithms (item 3).
    - [ ] `TemporalPolicy::Dynamic { rule }` for rule-driven temporal intervention schedules (Pulse / Sustained *policy types* already ship; Sustained *identification* is item 2).
    - [ ] `TargetPopulation::Predicate` and `CustomDistribution` (AllObserved / Treated / Untreated / Environment already ship).

- [ ] **10. Artifact mmap / stream / skip** (DESIGN.md §24.5) — container, CBOR wire, migration, and `model_bundle` already ship.
    - [ ] Memory-map or stream large Arrow sections without full buffer load.
    - [ ] Skip unread sections without materializing payloads.
    - [ ] Zero-copy write paths when Arrow buffers are already shared.

- [ ] **11. Structure-component attribution** (DESIGN.md §17.1–17.2) — `AttributionComponents::Structure` is in the type/wire model; `distribution_change` / `unit_change` reject it today. Inputs / Mechanisms / path / Shapley surfaces already ship (`parity/attribution.toml`).
    - [ ] Identify and estimate structure-change contributions between populations / units where defined.
    - [ ] Wire through attribution facade / GCM helpers without silently dropping the component.

- [ ] **12. Rolling mechanism diagnostics** (DESIGN.md §20) — incremental OLS / streaming cov / lag indexes / particle filter / BIC score cache already ship (`parity/design_state.toml`).
    - [ ] Rolling mechanism diagnostics maintained under `CausalState` events (bounded, versioned, reconstructible; eviction affects performance only).
