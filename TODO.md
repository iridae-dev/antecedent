# Remaining Work

Items are marked DONE with notes until independently verified and removed. Do not remove an item that you have just finished fixing. DESIGN.md is the roadmap; these are its unimplemented chapters, listed so nothing is lost.


1. **Data model** (DESIGN.md §5): `EventData` and `SampleRequest` as specified (neither type exists yet). Missing split strategies: random-IID, grouped/cluster, blocked-temporal, rolling-origin (only discovery/estimation-gap, environment-holdout, regime-holdout in `split.rs`). Arrow C Data Interface zero-copy (today Arrow enters via in-process `RecordBatch` and is copied — `arrow_adapter.rs`; the copy is diagnosed). Splits alone are lower risk; CDI and `EventData` raise safety/API risk.

2. **Performance infrastructure** (DESIGN.md §23.2 / feature inventory): Runtime-dispatched SIMD kernels (today: always-on portable autovectorized loops in `crates/causal-kernels/src/dispatch.rs`, scalar via `KernelPolicy`). Missing §21 kernels (covariance, standardization, pairwise distance, contingency, bootstrap weights). Documented feature-flag surface (`rayon`, `simd-runtime`, `blas`, `polars`, `serde-json`, `gaussian-process`, `hmc`, `smc`, `python`, `networkx-io`, `plot-data`) vs actual flags (`arrow` only; `faer` and portable kernels are required). Parallelism is hand-rolled `std::thread::scope` in discovery `engine.rs` — decide whether the roadmap keeps rayon or blesses the current approach. Wrong dispatch risks silent numerical drift.

3. **Static (non-temporal) discovery** (DESIGN.md §13.3–13.6): PC, FCI, RFCI, GES, LiNGAM, score-based search / NOTEARS. `causal-discovery` is temporal-only today (PCMCI family). Meek-rule and CI-test infrastructure already exists and is verified — PC is the natural first target. Full list is high surface; PC-only MVP is medium risk because CI/Meek reuse.

4. **Deep identification** (DESIGN.md §10.1–10.3): ID algorithm for semi-Markovian models, IDC, hedge certificates, `AutoIdentifier`, memoized recursion; maximal adjustment sets; missing `IdentificationStatus` variants (`IdentifiedUnderParametricRestrictions`, `IdentifiedUnderPriorRestrictions` — `crates/causal-core/src/identification.rs`). Backdoor / frontdoor / IV / RD and generalized adjustment shipped; not full ID/IDC. Highest scientific-correctness risk; status-variant plumbing alone is a smaller first slice. Softly helped by expr simplification (shipped in `causal-expr`).

5. **Bayesian graph discovery** (DESIGN.md §13.7): `GraphPosteriorEngine`, MCMC / enumeration / DBN structure search. Requires the documented `causal-discovery → causal-prob` dependency (absent today). High mixing/convergence and test burden; tiny-DAG enumeration is a medium-risk MVP. Softly helped by PC screening (item 3) and Bayes-factor / ESS work in item 6.

6. **Mechanism families and Bayesian gaps** (DESIGN.md §14.4, §16, §18.4, §12): Largest multi-backend bundle.
    - Mechanism families: BVAR, state-space, GP, hierarchical (only conjugate Gaussian + Laplace GLM exist today).
    - Counterfactual trajectories; simulation-based calibration (SBC).
    - ESS / R-hat diagnostics (`crates/causal-prob/src/diagnostics.rs` explicitly defers).
    - Bayes-factor CI and posterior dependence probability.
    ESS/SBC wait on HMC/SMC backends; overlaps optional feature flags in item 2.

7. **Distribution & path-specific pipelines** (DESIGN.md §8): Identify + estimate for shipped `CausalQuery::Distribution` / `PathSpecific` (types, Unsupported plumbing, GCM sampling / path-contribution wrappers, and minimal query wire are done). Interventional-distribution identification is **IDC** — implement only with/after deep identification (**item 4**); do not fork a second `AutoIdentifier`. Path-specific *natural effects* need path-restricted ID (same ID/IDC family); nonparametric path-specific remains waived under `parity/context.toml` `context.mediation.nonparametric`. Path *contribution* attribution already exists (`path_decompose` / `attribute_path_specific`). Full `CausalQuery` CBOR model-bundle embedding ships with `causal-io` model bundles / `query_wire`. Medium once IDC exists.
