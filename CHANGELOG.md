# Changelog

All notable changes to Antecedent are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.2.0]: https://github.com/iridae-dev/antecedent/releases/tag/v0.2.0
[0.1.0]: https://github.com/iridae-dev/antecedent/releases/tag/v0.1.0
