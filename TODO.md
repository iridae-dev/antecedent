# TODO

Items are marked DONE with notes until independently verified and removed. Do not remove an item that you have just finished fixing.

---

## P4 — Algorithm parity upgrades (bring implementations up to their published names)

### P4.4 J-PCMCI+ per Günther et al.
Scaffold is in place (`JpcmciPlus`, per-env pooled frames, four labeled phases,
`gunther_forbids` / `JpcmciNodeRole`, structure pin
`conformance/tigramite/jpcmci_plus_two_env`) but the skeleton is not yet faithful
to Günther Alg. 2 / tigramite `JPCMCIplus`. Remaining:

1. **Do not re-inject MCI-rejected lagged links.** After phases 2–4,
   `lagged_parents_as_scored` appends PC1 lagged parents with `p=0`, restoring
   edges that contemp MCI already removed. PCMCI+ correctly keeps only MCI
   survivors; J-PCMCI+ must do the same (drop this helper or emit only links
   still present in phase scored outputs).
2. **Prune `B̂^C` / `B̂^{CD}` for later phases.** Context (and dummy) parents
   rejected in phase 2 (resp. 3) must leave the conditioning parent sets used in
   phases 3–4. Today `merge_fixed_parents` only adds survivors; removed lagged
   context links from PC1 stay in `lagged_parents`, so later MCI over-conditions
   (tigramite keeps survivors in `observed_context_parents` / `dummy_parents`
   and strips context from `all_lagged_parents`).
3. **Tests that would have caught (1)–(2).** Unit / conformance coverage that
   asserts a lagged link removed by context or system MCI is absent from the
   CPDAG (and not used as a fixed conditioner downstream). The current two-env
   pin only checks `algorithm_id` + a weak true-edge subset.

### P4.4a J-PCMCI+ tigramite black-box edge-set pin
Full numerical edge-set equality vs pinned tigramite `JPCMCIplus` remains optional
(`parity/ci_deviations.md` §3). Add a black-box fixture when ready. Blocked on
P4.4 skeleton fixes above.

### P4.4b Multivariate single-node dummy CI
Production default is one-hot space-dummy columns + scalar ParCorr. Tigramite uses
one multivariate dummy node; wire `PairwiseMultivariateCi` / block ParCorr for a
single logical dummy when needed. Time dummy is still a scalar time-index column
(full T-way one-hot deferred in `pooled_frame.rs`).

---

## P5 — Roadmap features from DESIGN.md not yet built

DESIGN.md is the roadmap; these are its unimplemented chapters, listed so nothing is lost. Ordered
roughly by how much current claims/outputs depend on them.

1. **Static (non-temporal) discovery** (DESIGN.md:1211-1221): PC, FCI, RFCI, GES, LiNGAM,
   score-based search / NOTEARS. `causal-discovery` is temporal-only today. The Meek-rule and
   CI-test infrastructure already exists and is verified correct — PC is the natural first target,
   as DESIGN says.
2. **Bayesian graph discovery** (DESIGN.md:1281-1305): `GraphPosteriorEngine`, MCMC/enumeration/DBN
   structure search. Requires adding the documented `causal-discovery → causal-prob` dependency.
3. **Deep identification** (DESIGN.md:868-882, 903, 925): ID algorithm for semi-Markovian models,
   IDC, hedge certificates, `AutoIdentifier`, memoized recursion; maximal adjustment sets; the two
   missing `IdentificationStatus` variants (`IdentifiedUnderParametricRestrictions`,
   `IdentifiedUnderPriorRestrictions` — `crates/causal-core/src/identification.rs:11-20`).
4. **Statistical layer** (DESIGN.md §11.2–11.4): kernels shipped; DESIGN contracts incomplete.
   Present (verified): `fit_multinomial_logit`; NB2 IRLS + `NbAlphaPolicy`
   Fixed/MoM/`NestedMle` (digamma/trigamma); `BinomialProbit` + `GlmAdjustmentAte`;
   `fit_huber_m`; `fit_ridge`/`fit_lasso` + optional `ridge_on_separation`;
   `coefficient_covariance` HC0–HC3/cluster/multiway/NW/`PanelClusterHac`;
   `ResamplingPlan` (+ cluster/stationary/permutation via `fill_resample_indexes_grouped`);
   `AnalyticSeKind::{Homoskedastic,Hc1,Cluster}` on linear/IV/AIPW + matching AI hetero/cluster.
   Remaining: §11.2 `CompiledDesign` missing contrasts/standardization; fit results lack
   rank/condition/backend/allocation diagnostics (`GlmFit`/`MEstimateFit`/`LassoFit` are
   thin); ridge/lasso/Huber unused by any estimator (separation still hard-fails by default);
   NB rejected by `GlmAdjustmentAte`; sandwich on estimators stops at three kinds (no
   HC0/HC2/HC3/multiway/NW/panel; GLM adjustment still delta-method/bootstrap); optional
   GAM interfaces; §11.4 batch plan production under one `ExecutionContext` (today:
   single-plan `fill_*` + `CausalRng`).
5. **Mechanism families** (DESIGN.md:1422-1429): BVAR, state-space, GP, hierarchical (only
   conjugate Gaussian + Laplace GLM exist). Counterfactual trajectories (line 1637).
   Simulation-based calibration (line 1801). ESS/R-hat diagnostics
   (`crates/causal-prob/src/diagnostics.rs:3` explicitly defers). Bayes-factor CI and posterior
   dependence probability (DESIGN.md:1152-1157).
6. **Performance infrastructure** (DESIGN.md:983, 2112-2139, 2883-2903): runtime-dispatched SIMD
   kernels (nothing today: dispatch is compile-time `cfg!` to autovectorized loops,
   `crates/causal-kernels/src/dispatch.rs:21-26`); the missing kernels from the §21 list
   (covariance, standardization, pairwise distance, contingency, bootstrap weights); the documented
   feature-flag surface (`rayon`, `simd-runtime`, `blas`, `polars`, `serde-json`,
   `gaussian-process`, `hmc`, `smc`, `python`, `networkx-io`, `plot-data` — none exist; actual
   flags are `arrow`, `faer`, and undocumented `portable-optimized`). Note: `rayon` appears nowhere;
   parallelism is hand-rolled `std::thread::scope` (`engine.rs:412,499`) — decide whether the
   roadmap keeps rayon or blesses the current approach.
7. **Serialization** (DESIGN.md:185, 2273, 2289): zstd section compression (fields always `None`,
   `crates/causal-io/src/container.rs:163-165`); real version migrations (only identity 0.1→0.1
   exists, `migrate.rs:16-37`); GML and NetworkX-compatible exchange; model bundles.
8. **Data model** (DESIGN.md:310, 348, 458-508, 2348): `EventData`; `SampleRequest` as specified;
   the five missing split strategies (random-IID, grouped/cluster, blocked-temporal,
   rolling-origin — only discovery/estimation-gap, environment-holdout, regime-holdout exist,
   `split.rs:41,143,176`); Arrow C Data Interface zero-copy (today Arrow enters via in-process
   `RecordBatch` and is copied, `arrow_adapter.rs:31-35` — the copy is at least diagnosed).
9. **Graph algorithms** (DESIGN.md:623-641, 671, 705-707): DAG Markov blankets shipped
   (`Dag::markov_blanket`). Remaining: ADMG blanket beyond adjacency-style
   (`Admg::markov_blanket` is not full m-separation / inducing-path closure);
   intervention/mutilation via overlays instead of cloning (`Dag::mutilate` still
   returns a full new `Dag` — DESIGN marks overlays as planned).
10. **causal-expr** completions: simplification; compiled evaluators. LaTeX rendering
    shipped (`CausalExprArena::latex` — thin `ExprNode` walker).
11. **Core query model** (DESIGN.md:727-739): `CausalQuery::Distribution` and `PathSpecific`
    variants (code has undocumented `MechanismChange`/`UnitChange` instead — reconcile the roadmap
    with what emerged).
12. **Python packaging** (DESIGN.md:2321-2338 / §25.5): wheel matrix verification
    (CPython 3.11–3.14 × manylinux/macOS/Windows). FFI panic isolation shipped
    (`catch_ffi` / `detach_catch`); `py.typed` + stubs with P2.5.
