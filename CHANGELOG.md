# Changelog

All notable changes to Antecedent are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
  under-dispersing the null for the autocorrelated data the parameter exists to protect.
  They now reject it rather than silently ignoring it.

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

[0.4.0]: https://github.com/iridae-dev/antecedent/releases/tag/v0.4.0
[0.3.0]: https://github.com/iridae-dev/antecedent/releases/tag/v0.3.0
[0.2.0]: https://github.com/iridae-dev/antecedent/releases/tag/v0.2.0
[0.1.0]: https://github.com/iridae-dev/antecedent/releases/tag/v0.1.0
