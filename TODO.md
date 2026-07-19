# Remaining Work

Each numbered item is a **1.0 deliverable**. Check the parent box only when every required sub-item is complete, verified, and the matching `parity/*.toml` rows are `done`. Partial slices do not count.

Use `- [x]` / `- [ ]` for status. Do not remove or check off an item you just finished — leave notes until independently verified.

DESIGN.md is the roadmap; these are its unimplemented chapters. Parity inventories use `pending` / `in_progress` / `done` only. Unfinished chapters stay `pending` here and in `parity/*.toml` until the parent item is checked.

Out of scope for these DONE gates (per DESIGN): selection/transport ID as later modules; external Stan/PyMC adapters; unsupervised RPCMCI regime search.

**Verified done and removed (2026-07-22):** kernel-policy threading through stats/CI/data (§23.2 escape hatch); `IdentificationStatus` parametric/prior vocabulary + wire (estimation still nonparametric-only); static `Cpdag` + Meek/collider + classic PC (`discovery.pc`); `analyze(discovery=JPCMCIPlus|RPCMCI|PC)`; lag/alpha/CI discovery sensitivity validators; correctness follow-ups (FCI R10, front-door child-subset search, natural-mediation alias gate, multi-source path attribution, Wald/Hájek/matching SE, mechanism `MeanDiff` residuals, path-search budgets, latent-projection fail-closed). **Also removed after code verification:** Python `analyze()` / §25.4 callbacks (item 6); discovery validation OrientationStability / RegimeStability / EnvironmentHoldout / SyntheticNullCalibration / FalsePositiveCheck (item 7); query Planned variants `TemporalPolicy::Dynamic` + `TargetPopulation::{Predicate,CustomDistribution}` with fail-closed overlay/identify/estimate incl. temporal AllObserved gate (item 8); artifact mmap/stream/skip (item 9); structure-component attribution (item 10); rolling mechanism diagnostics (item 11). **Also removed after code verification:** Distribution & path-specific pipelines (item 2); static discovery 3a–3h (Pag FCI plumbing, FCI, RFCI, GES, DirectLiNGAM, CPDAG MEC, pooled-frame time one-hot) — parent 3 closed; NOTEARS remains item 3.5. **Deep identification (item 1)** — ID/IDC/hedge/AutoIdentifier/sustained temporal ID + parity + adjustment-set `max_history_lag` / `history_lags` filtering wired through static backdoor and unfolded temporal pulse. **Mechanism families & Bayesian gaps (item 4)** — EB hierarchical linear + HierarchicalGlm, Minnesota BVAR, LGSSM Kalman EM, GP hyperparam grid; HMC + MCMC diagnostics; real SBC (prior→simulate→refit→rank) + LOO family comparison + synthetic-SCM posterior calibration; Bayes CI; trajectories; nested hard CF with mediator freeze; conditional interventional sampling; posterior reduce kernels wired through PPC/SBC.

Ordered foundations → dependents.

- [ ] **3.5 NOTEARS / continuous SEM discovery** (DESIGN.md §13.3) — first-class 1.0 static discovery for the “driver graph” persona: continuous tabular data → single weighted DAG for intervene/simulate when GES leaves undirected marks and LiNGAM’s non-Gaussian assumption does not fit. **Not** a Cargo-feature “optional”; same DONE bar as GES/LiNGAM (algorithm + facade/Python + parity). Soft dep of item 3 graph/review surface (`Dag` / `DagReview`).
    - [ ] **3.5a. Core solver** — Linear SEM least-squares (or equivalent) loss + smooth exact acyclicity constraint (NOTEARS \(h(W)\)); L-BFGS-B / augmented Lagrangian (or documented native equivalent); fail-closed on non-finite / non-convergence.
    - [ ] **3.5b. Threshold → `Dag` + weights** — Post-hoc sparsity threshold to hard DAG; retain coefficient matrix for mechanism seeding; `DagReview` like DirectLiNGAM; scale/standardize policy documented (varsortability).
    - [ ] **3.5c. Facade / Python** — `Notears` / `discover_notears` / `GraphInput::DiscoverNotears` / analyze builder; parity `discovery.notears` → `done`.
    - [ ] **3.5d. Conformance** — Synthetic linear SEM fixture(s) + expected skeleton/orientation tolerance class; gate evidence.

- [ ] **5. Bayesian graph discovery** (DESIGN.md §13.7) — additive to constraint-based discovery; graph-weighted effect envelopes over supplied `WeightedGraphSamples` already ship.
    - [ ] Wire documented `causal-discovery → causal-prob` dependency (absent today).
    - [ ] `GraphPosteriorEngine` trait + columnar/indexed `GraphPosterior` (weights, edge/orientation marginals, ESS, chain diagnostics, rejected invalid graphs).
    - [ ] Exact enumeration for very small DAGs.
    - [ ] Order MCMC and/or structure MCMC for discrete / small continuous models.
    - [ ] Candidate-edge posterior updates after CI screening (**uses** static PC from item 3 and Bayes-factor / posterior-dependence CI from item 4 as screening/proposal inputs).
    - [ ] Dynamic Bayesian network posterior search for bounded lag.
    - [ ] Parity: `bayes.discovery.dag_posterior` → `done`.
