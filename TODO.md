# TODO

Items are marked DONE with notes until independently verified and removed. Do not remove an item that you have just finished fixing.

---

## P5 — Roadmap features from DESIGN.md not yet built

DESIGN.md is the roadmap; these are its unimplemented chapters, listed so nothing is lost.
Ordered by estimated work and risk (lowest first): implementation surface × correctness/API
blast radius, not product priority.

1. **Python packaging** (DESIGN.md §25.5): Keep the wheel matrix green (CPython 3.11–3.14 ×
   manylinux / macOS / Windows). FFI panic isolation (`catch_ffi` / `detach_catch`) and
   `py.typed` + stubs already shipped; remaining work is verification/maintenance as public
   APIs grow, plus documenting any platform gaps. Lowest risk: CI already covers the matrix
   and parity marks it done.

2. **Core query model** (DESIGN.md §8): Reconcile roadmap with
   `crates/causal-core/src/query.rs`. Code has `MechanismChange` / `UnitChange` (document as
   built); DESIGN still lists planned `CausalQuery::Distribution` and `PathSpecific`. Either
   wire those variants through identify / estimate / attribution / IO / Python, or explicitly
   defer them in DESIGN. Doc-only reconcile is low risk; full pipeline support is medium.

3. **Statistical-layer follow-ups** (deferred from P5.4 / DESIGN §11.2): Bounded stats and
   design-API nits in `causal-stats` / `causal-estimate` / `causal-data`.
   - Rename `CompiledDesign.matrix` to DESIGN’s `PreparedDesignMatrix` and enrich the column
     map (mechanical rename with wide consumer blast radius).
   - Put `VariableId` on `ResamplingPlan::ClusterBootstrap` (cluster ids are fill-helper args
     today in `resample.rs`).
   - Score-exact GLM sandwich meat (production path uses working/Pearson residual approx in
     `glm_adjustment.rs`).
   - Lasso analytic SE policy (`LinearFitKind::Lasso` is NaN / bootstrap-only today — ship
     analytic SE or document permanent bootstrap-only).
   - Enable `ridge_on_separation` on `GlmOptions::default()` itself (estimate-layer defaults
     only today; flipping stats defaults is a behavior change for direct GLM callers).

4. **Graph algorithms** (DESIGN.md §6.5–6.7): `Dag::markov_blanket` shipped. Remaining:
   full ADMG blanket beyond adjacency-style parents/children/spouses
   (`Admg::markov_blanket` is not m-separation / inducing-path closure); intervention /
   mutilation via overlays instead of cloning (`Dag::mutilate` still returns a full new
   `Dag` — DESIGN marks overlays as planned). Theory-sensitive but localized to
   `causal-graph`.

5. **causal-expr completions** (DESIGN.md §9): Simplification (rewrite/simplify worklist +
   memoization) and compiled evaluators against empirical/posterior providers. LaTeX
   rendering already shipped (`CausalExprArena::latex`). Semantic-preservation risk;
   improves ID rewrite quality when deep identification lands.

6. **Serialization** (DESIGN.md §24): zstd section compression (fields always `None` in
   `crates/causal-io/src/container.rs`); real version migrations (only identity 0.1→0.1 in
   `migrate.rs`); GML and NetworkX-compatible exchange (intentional deviation for 1.0);
   model bundles. Careful but bounded IO + conformance work; status/query enum changes
   elsewhere force wire updates.

7. **Optional GAM interfaces** (DESIGN.md §11.2): Generalized additive interfaces as
   optional extensions — no spline / backfit / additive-smooth APIs yet. Medium numeric
   scope; benefits from the richer `PreparedDesignMatrix` / column map in item 3. Explicitly
   deferrable with low product risk.

8. **Data model** (DESIGN.md §5): `EventData` and `SampleRequest` as specified (neither type
   exists yet). Missing split strategies: random-IID, grouped/cluster, blocked-temporal,
   rolling-origin (only discovery/estimation-gap, environment-holdout, regime-holdout in
   `split.rs`). Arrow C Data Interface zero-copy (today Arrow enters via in-process
   `RecordBatch` and is copied — `arrow_adapter.rs`; the copy is diagnosed). Splits alone
   are lower risk; CDI and `EventData` raise safety/API risk.

9. **Performance infrastructure** (DESIGN.md §23.2 / feature inventory): Runtime-dispatched
   SIMD kernels (today: compile-time `cfg!` to autovectorized loops in
   `crates/causal-kernels/src/dispatch.rs`). Missing §21 kernels (covariance, standardization,
   pairwise distance, contingency, bootstrap weights). Documented feature-flag surface
   (`rayon`, `simd-runtime`, `blas`, `polars`, `serde-json`, `gaussian-process`, `hmc`,
   `smc`, `python`, `networkx-io`, `plot-data`) vs actual flags (`arrow`, `faer`,
   undocumented `portable-optimized`). Parallelism is hand-rolled `std::thread::scope` in
   discovery `engine.rs` — decide whether the roadmap keeps rayon or blesses the current
   approach. Wrong dispatch risks silent numerical drift.

10. **Static (non-temporal) discovery** (DESIGN.md §13.3–13.6): PC, FCI, RFCI, GES, LiNGAM,
    score-based search / NOTEARS. `causal-discovery` is temporal-only today (PCMCI family).
    Meek-rule and CI-test infrastructure already exists and is verified — PC is the natural
    first target. Full list is high surface; PC-only MVP is medium risk because CI/Meek
    reuse.

11. **Deep identification** (DESIGN.md §10.1–10.3): ID algorithm for semi-Markovian models,
    IDC, hedge certificates, `AutoIdentifier`, memoized recursion; maximal adjustment sets;
    missing `IdentificationStatus` variants (`IdentifiedUnderParametricRestrictions`,
    `IdentifiedUnderPriorRestrictions` — `crates/causal-core/src/identification.rs`).
    Backdoor / frontdoor / IV / RD and generalized adjustment shipped; not full ID/IDC.
    Highest scientific-correctness risk; status-variant plumbing alone is a smaller first
    slice. Softly helped by expr simplification (item 5).

12. **Bayesian graph discovery** (DESIGN.md §13.7): `GraphPosteriorEngine`, MCMC /
    enumeration / DBN structure search. Requires the documented
    `causal-discovery → causal-prob` dependency (absent today). High mixing/convergence and
    test burden; tiny-DAG enumeration is a medium-risk MVP. Softly helped by PC screening
    (item 10) and Bayes-factor / ESS work in item 13.

13. **Mechanism families and Bayesian gaps** (DESIGN.md §14.4, §16, §18.4, §12): Largest
    multi-backend bundle.
    - Mechanism families: BVAR, state-space, GP, hierarchical (only conjugate Gaussian +
      Laplace GLM exist today).
    - Counterfactual trajectories; simulation-based calibration (SBC).
    - ESS / R-hat diagnostics (`crates/causal-prob/src/diagnostics.rs` explicitly defers).
    - Bayes-factor CI and posterior dependence probability.
    ESS/SBC wait on HMC/SMC backends; overlaps optional feature flags in item 9.
