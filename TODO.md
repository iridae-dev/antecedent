# TODO

Items are marked DONE with notes until independently verified and removed. Do not remove an item that you have just finished fixing.

---

## P4 — Algorithm parity upgrades (bring implementations up to their published names)

### P4.4 J-PCMCI+ per Günther et al.
`crates/causal-discovery/src/jpcmci_plus.rs` still runs PCMCI independently per environment,
pools surviving links by intersection (`p = max`), merges per-env sepsets, and attaches
`context_variables` as decoration after pooling (`attach_context_nodes` — they never enter CI
tests). Module docs now match this scope. Sepset/plan interim bugs are fixed
(`merge_sepsets`; `MultiEnvSamplePlan` validates shared geometry). Remaining: the published
Günther et al. redesign — augment the variable set with observed context + dataset/time
dummies and run PCMCI+ once on pooled data under link assumptions.

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
4. **Statistical layer** (DESIGN.md:1000-1061): ~~multinomial logistic (unblocks P1.12)~~ **done**
   (`fit_multinomial_logit`); remaining: negative
   binomial, probit IRLS (unblocks P4.12), robust M-estimation, ridge/lasso (optional
   separation fallback; hard-fail already shipped); **robust covariance §11.3** — HC0–HC3, cluster, multiway, HAC/Newey-West (zero hits
   repo-wide today; SEs are homoskedastic-analytic or bootstrap); shared resampling engine §11.4
   additions — cluster and stationary-block bootstrap, permutation resampling.
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
9. **Graph algorithms** (DESIGN.md:623-641, 671): Markov blankets; intervention/mutilation via
   overlays instead of cloning (`mutilate` returns a full new Dag,
   `crates/causal-graph/src/ancestry.rs:73`).
10. **causal-expr** completions: simplification, LaTeX rendering, compiled evaluators.
11. **Core query model** (DESIGN.md:727-739): `CausalQuery::Distribution` and `PathSpecific`
    variants (code has undocumented `MechanismChange`/`UnitChange` instead — reconcile the roadmap
    with what emerged).
12. **Python packaging** (DESIGN.md:2321-2338): wheel matrix verification and explicit
    `catch_unwind` at the FFI boundary rather than relying on PyO3's PanicException
    (`py.typed` + stubs landed with P2.5).
