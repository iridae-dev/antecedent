# Remaining Work

Each numbered item is a **1.0 deliverable**. Check the parent box only when every required sub-item is complete, verified, and the matching `parity/*.toml` rows are `done`. Partial slices do not count.

Use `- [x]` / `- [ ]` for status. Do not remove or check off an item you just finished — leave notes until independently verified.

DESIGN.md is the roadmap; these are its unimplemented chapters. Parity inventories use `pending` / `in_progress` / `done` only. Unfinished chapters stay `pending` here and in `parity/*.toml` until the parent item is checked.

Out of scope for these DONE gates (per DESIGN): selection/transport ID as later modules; external Stan/PyMC adapters; unsupervised RPCMCI regime search; NOTEARS as optional extension.

**Verified done and removed (2026-07-22):** kernel-policy threading through stats/CI/data (§23.2 escape hatch); `IdentificationStatus` parametric/prior vocabulary + wire (estimation still nonparametric-only); static `Cpdag` + Meek/collider + classic PC (`discovery.pc`); `analyze(discovery=JPCMCIPlus|RPCMCI|PC)`; lag/alpha/CI discovery sensitivity validators; correctness follow-ups (FCI R10, front-door child-subset search, natural-mediation alias gate, multi-source path attribution, Wald/Hájek/matching SE, mechanism `MeanDiff` residuals, path-search budgets, latent-projection fail-closed). **Also removed after code verification:** Python `analyze()` / §25.4 callbacks (item 6); discovery validation OrientationStability / RegimeStability / EnvironmentHoldout / SyntheticNullCalibration / FalsePositiveCheck (item 7); query Planned variants `TemporalPolicy::Dynamic` + `TargetPopulation::{Predicate,CustomDistribution}` with fail-closed overlay/identify/estimate incl. temporal AllObserved gate (item 8); artifact mmap/stream/skip (item 9); structure-component attribution (item 10); rolling mechanism diagnostics (item 11). **Item 1 (deep identification) checked below (2026-07-22).**

Ordered foundations → dependents.

- [x] **1. Deep identification** (DESIGN.md §10.1–10.3) — full semi-Markovian ID/IDC surface; highest scientific-correctness risk. Status vocabulary for parametric/prior ID already ships; estimation still gates on nonparametric ID only (no fake producers).
    - [x] ID algorithm for semi-Markovian models (Shpitser); memoized recursion over canonical subproblems; expression arena reuse (§10.5).
    - [x] IDC for conditional interventional distributions.
    - [x] Hedge certificates for non-identifiability.
    - [x] `AutoIdentifier` that returns all valid estimands and selection rationale (no silent estimator choice); does not fork a second identifier for distribution queries.
    - [x] Maximal (and remaining) adjustment-set search where defined beyond shipped backdoor / frontdoor / IV / RD / generalized adjustment (§10.4): cost-weighted selection, measurement-cost metadata, temporal history restrictions, positivity-aware ranking after a data check, and streaming enumeration (callbacks/iterators) so combinatorial sets need not be retained. Front-door enumerates child subsets and bounded subsets of all intermediates `V\{T,Y}`.
    - [x] Sustained temporal-effect identification (g-formula / sequential ID) — Pulse uses unfolded backdoor; Sustained uses general ID on the unfolded multi-treatment window.
    - [x] Parity: `estimate.identify.general_id`, `pag.identify.full_id_idc` → `done`.

- [x] **2. Distribution & path-specific pipelines** (DESIGN.md §8) — identify + estimate for shipped `CausalQuery::Distribution` / `PathSpecific`. ID/IDC + discrete functional plug-in for Distribution; path-restricted NE ID + `functional.effect` for PathSpecific; CBOR model-bundle / analysis_wire complete.
    - [x] **Depends on item 1** (IDC / path-restricted ID in the same ID/IDC family — do not fork a second `AutoIdentifier`).
    - [x] Interventional-distribution identification + estimation via IDC.
    - [x] Path-restricted *natural effects* identification (path-restricted ID).
    - [x] Nonparametric path-specific natural effects (`context.mediation.nonparametric` → `done`). Path *contribution* attribution already ships (`path_decompose` / `attribute_path_specific`).
    - [x] Full `CausalQuery` CBOR model-bundle embedding for these queries via `causal-io` / `query_wire` where still incomplete.

- [ ] **3. Static (non-temporal) discovery** (DESIGN.md §13.3–13.6) — PCMCI family remains the temporal surface. **Shipped:** static `Cpdag` / `CpdagReview`, Meek R1–R4 + `OrientCollider`, classic PC (`Pc` / `discover_pc` / `discovery.pc`); static `Pag` / `PagReview`, Zhang FCI rules (`PagOps` / `FciOrientationRule`), classic FCI + RFCI (`Fci` / `Rfci` / `discover_fci` / `discover_rfci` / `pag.discovery.fci_rfci`); GES (`Ges` / `discover_ges` / `discovery.ges`); DirectLiNGAM (`DirectLingam` / `discover_lingam` / `discovery.lingam` / `DagReview`). ContempMeek stays temporal-only. Parent stays unchecked until required sub-items below are verified.
    - [x] **3a. `Pag` FCI plumbing** — Public `Pag::remove_edge` shipped (matches `TemporalPag`). FCI orientation rules target static `Pag` via `PagOps` / `FciOrientationRule` (Zhang R1–R4 / R8–R10 + collider + discriminating); LPCMCI APR/MMR remain temporal-only. **Depends on:** shipped LPCMCI FCI-like rules.
    - [x] **3b. Static FCI** — Possible-D-Sep adjacency phase; Zhang R1–R4 / R8–R10 + discriminating / uncovered paths on `Pag`; classic FCI pipeline → static `Pag` (`Fci` / `discover_fci` / `PagReview`). **Depends on:** shipped PC; 3a.
    - [x] **3c. RFCI** — Early-stop / reduced Possible-D-Sep on top of FCI; `pag.discovery.fci_rfci` → `done`. **Depends on:** 3b.
    - [x] **3d. GES / score-based DAG search** — Equivalence-class insert/delete/reverse operators using shipped Gaussian BIC `LocalScoreCache` (`causal-state`). Soft: PC skeleton as screening. **Depends on:** shipped BIC score cache.
    - [x] **3e. LiNGAM (DirectLiNGAM MVP)** — ICA / residual independence → causal order → `Dag`. Greenfield (no stubs). Independent of Meek/PC orientation stack.
    - [ ] **3f. NOTEARS (optional)** — Continuous acyclicity soft-constraint optimization; feature-gated. Not required for this item’s DONE gate (DESIGN §13.3).
    - [ ] **3g. CPDAG MEC / equivalence-class sampling** (DESIGN §6.5 item 15) — enumerate or sample DAG completions of a static `Cpdag` (today `try_into_dag` only when fully oriented). PAG `CompletionSampler` already ships. Soft dep of FCI/GES class-aware ID.
    - [ ] **3h. Pooled-frame time one-hot** — JPCMCI+ `DummyOptions` ships space one-hot + optional integer time-index dummy; high-dimensional one-hot of `T` remains deferred (`causal-data` `pooled_frame`).

- [x] **4. Mechanism families and Bayesian inference gaps** (DESIGN.md §14.4, §15.4, §16, §18.4, §12, §23.2) — complete native Bayesian 1.0 beyond conjugate Gaussian + Laplace GLM. External Stan/PyMC adapters are **not** required (DESIGN §14.5).
    - [x] Mechanism families: hierarchical linear/GLM, BVAR, linear Gaussian state-space, Gaussian-process mechanisms (`bayes.backend.hierarchical_bvar_gp` → `done`).
    - [x] Native sampling backends needed for chain diagnostics (HMC and/or SMC) so ESS / R-hat / divergences are meaningful.
    - [x] MCMC diagnostics: ESS, R-hat / convergence, divergence counts (`causal-prob` diagnostics; `bayes.validate.mcmc_diagnostics` → `done`).
    - [x] Simulation-based calibration (SBC) and remaining §18.4 workflow diagnostics not already shipped: likelihood-family comparison and posterior calibration on synthetic SCMs (PPC / prior predictive / prior sensitivity already ship).
    - [x] Bayes-factor CI, posterior dependence probability, and posterior-predictive CI diagnostics for supported conjugate models (`bayes.ci.tests` → `done`).
    - [x] Counterfactual trajectories (§16) with shared-noise / batched evaluation (point/ITE/abduction paths that already exist stay; trajectories complete the subsystem).
    - [x] Nested counterfactual expressions (§16) where identifiable under model assumptions (`allow_nested` exists; engine rejects true nested interventions today).
    - [x] Conditional interventional sampling (§15.4) where supported (observational / hard / soft / stochastic / sequence / posterior-predictive sampling already ship).
    - [x] Posterior draw reduction kernels (§23.2 deferred): shared scalar + portable reductions over posterior draw batches, wired through Bayesian estimation / PPC / SBC callers.

- [ ] **5. Bayesian graph discovery** (DESIGN.md §13.7) — additive to constraint-based discovery; graph-weighted effect envelopes over supplied `WeightedGraphSamples` already ship.
    - [ ] Wire documented `causal-discovery → causal-prob` dependency (absent today).
    - [ ] `GraphPosteriorEngine` trait + columnar/indexed `GraphPosterior` (weights, edge/orientation marginals, ESS, chain diagnostics, rejected invalid graphs).
    - [ ] Exact enumeration for very small DAGs.
    - [ ] Order MCMC and/or structure MCMC for discrete / small continuous models.
    - [ ] Candidate-edge posterior updates after CI screening (**uses** static PC from item 3 and Bayes-factor / posterior-dependence CI from item 4 as screening/proposal inputs).
    - [ ] Dynamic Bayesian network posterior search for bounded lag.
    - [ ] Parity: `bayes.discovery.dag_posterior` → `done`.
