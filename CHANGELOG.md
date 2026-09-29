# Changelog

## 2.1.1

Corrections to 2.1.0. No new capability is licensed. Some estimates change on some inputs; some inputs that 2.1.0 accepted with a wrong or corrupted result are now refused; both are listed below.

### Changed estimates

These routes return different numbers from 2.1.0 on some inputs.

- Synthetic difference in differences fits its time weights to each donor's post-period mean, up to a common intercept. 2.1.0 fit them to the pre-period mean, which uniform weights reproduce exactly, so it computed a synthetic-control-weighted DiD with a plain pre-period average. Additive panels give the same effect; factor-model panels do not.
- Fuzzy regression discontinuity and regression kink count rows whose running variable equals the cutoff on the treated side, matching the `R >= cutoff` assignment rule. 2.1.0 dropped them from both sides.
- Bayesian IV mixes its seed before drawing and reports exchangeable-rank quantiles. Nearby seeds previously started nearly identical streams, so draws were under-dispersed or off-centre; draws for a given seed change.
- Negative binomial g-computation standard errors are no longer multiplied by the fitted dispersion α. The NB2 Fisher weight already carries α, so 2.1.0 scaled the standard error by √α (about three times too small at α = 0.1).
- A transported coefficient prior is widened by the variance of its reweighted mean. When the transport weights have a Kish effective sample size below 20, the source is dropped (α = 0) and the reason is recorded.
- A prior hydrated from a posterior that records no residual variance (a `PosteriorArtifact.from_moments` summary, a known-σ² or a GLM fit) keeps the source's absolute coefficient variances, and the target fit converts them with its own residual-variance estimate, recorded in a diagnostic note. 2.1.0 assumed σ² = 1, which mis-scaled the prior width by the target's σ². Such a prior cannot be composed with other priors.
- Conjugate Gaussian posteriors are solved after equilibrating the posterior precision, so badly scaled designs keep their digits; the reported condition number is that of the equilibrated system.
- `StudyResult::effect()` returns the queried mediation contrast (total, direct, or mediated), the scalar the contract and artifact already carried. 2.1.0 returned the total for every mediation query.

### Fixed

Artifacts and contracts

- Verification applies interventional-distribution atom replay and the checked AIPW lowering and row-binding requirements when the contract resolved that estimator without declaring it. In 2.1.0 such a study verified with no replay, so an edited artifact could still verify. Resolved-only IV and front-door contracts are checked against the resolved procedure.
- An interventional-distribution artifact is refused when it carries a scalar estimate its atoms do not define as a mean, or one that differs from that mean. A standard error without an estimate is refused, and the writer no longer emits one.
- Anomaly attribution queries re-encode to their original bytes, so 2.0.0 artifacts keep their digests and claim ids.
- Structural response masses outside [0, 1], trailing bytes after an artifact manifest, and manifests whose declared section sizes sum past 2 GiB are refused on load.
- Bayesian ADMG average effects advertise the resolved `functional.effect` estimator instead of the auto-bound `bayesian.gcomp` placeholder, and a Bayesian `ConditionalEffect` resolves `bayesian.conditional` in either builder call order.
- Graphless studies carry their `off_axis` support status on the live contract, and loaded panel DiD, policy-value and continuous-dose answers keep the unset support label the live answer publishes. A staggered event-study panel DiD publishes its own graphless coordinate, and staggered designs are labelled with their design and target.
- Bayesian bootstrap posteriors (robust average effect, anomaly and change attribution, the GCM counterfactual) no longer carry a Gaussian-likelihood disclosure.
- DBN posterior contrasts report failed estimation as unevaluable mass with a warning instead of as structural non-identification.

Prepared studies

- Refreshing a prepared linear-adjustment, GLM, RD, AIPW, IV, or front-door study with a different number of rows rebinds the retained fit. In 2.1.0 export failed after a shorter refresh and published the prepare-time row count after a longer one.
- Validators rebound on a prepared study run on the sealed Bayesian g-computation, Bayesian conditional, and temporal class effect routes.
- The sealed Bayesian conditional route fits and refutes on the identified columns, as the one-shot route does.

Transport

- Two-source z-transport no longer combines factors taken at different levels of a shared treatment. Intervention-factorized components are refused as `z_transport.components_not_independent` when they request a shared treatment at different levels or one side omits an ancestral treatment.
- When two catalog regimes supply the same factor, every binding path chooses by one documented rule: a family regime first, otherwise the lowest regime id.
- The z-transport planner can propose measuring the target population's observational joint when a recursive formula cites it. Transport evidence planning stops on cancellation; a candidate that exhausts its search limits is recorded as unresolved.
- `validate_z_experiment_family` refuses a controllable set too wide to enumerate instead of overflowing. Transport intervals refuse a zero coverage level. Functional-program limits on the wire path raise a resource error.

Statistics, discovery, and models

- G² and symbolic CMI treat non-integer category values as labels instead of rounding them. Bayesian CI tests report exact independence when the conditioning set determines X or Y.
- Discovery orientation rules (Meek R3, PAG R1 and R3) skip ambiguous triples, and a triple with no separating set is marked ambiguous. R9/R10 leave an edge unoriented instead of failing the run when the path search exhausts its budget. Discovery results record stationarity and linearity assumptions; FDR families finer than the test's p-value floor, DirectLiNGAM residuals whose order is not identified, and pairs a CI-screened posterior never proposed are reported. DirectLiNGAM fits collinear predecessors by minimum norm.
- Conditional interventional sampling grows its particle set for tail evidence and refuses only past 64 times the base count.
- Covariate-adjusted IPCW means flag a mean restricted by administrative censoring, and Bayesian random-intercept whitening discloses a fallback to iid.
- Truncated attribution path enumeration is a resource refusal. Sensitivity thresholds that fall between grid points are judged by the last cleared point; path subset-stability checks are no longer reported as informative; simulation-based and synthetic-SCM calibration checks stop on cancellation.
- The survival bootstrap refuses a known-censoring array that does not cover every subject, and `ridge_gram_inverse` returns an error instead of panicking on an empty design.

Python

- A design input named by a data column is validated like an inline array: a bool column accepts booleans or 0/1, an integer column finite whole numbers. In 2.1.0 `NaN` and `"False"` became `True` and 2.7 became 2.
- `refresh` and `estimate` accept the table used at `analyze` for a column-named design, including pyarrow tables; named design columns are dropped, not re-read.
- Checked transport programs and catalogs raise `CausalResourceError`, `CausalSerializationError`, `CausalUnsupportedError`, or `CausalValueError`; transport identification still raises `CausalIdentifyError`. A malformed transported view raises `CausalSerializationError`. Policy evaluation input checks raise `CausalValueError`.
- A transported contrast with nothing missing keeps the native per-point detail. Provider-supplied values for registry-owned provenance keys are dropped. `PolicyValue`, `MultiActionCatePoint`, `SurvivalDifferenceBand`, `InterferencePointwiseInterval`, `ExecutableProvider`, and `EconMLProviderAdapter` are in their modules' `__all__`; `dataclasses.replace` works on an inverse-probability `PolicyValue`. Prediction warns when a feature needs a lenient numeric parse.

### Now refused

Each of these returned a wrong or corrupted result in 2.1.0.

- kNN conditional-independence tests conditioned on two or more continuous variables: the permutation null did not hold its Type I error. PC and PCMCI runs using kNN stop at such conditioning sets.
- Weighted partial correlation with NaN, infinite, or negative weights, and static discovery with a weight vector that does not match the rows; these rows were silently dropped or misaligned.
- Clustered, multiway, and HAC standard errors for matching estimators, whose influence terms were charged to the wrong cluster.
- Conjugate Gaussian posteriors whose equilibrated precision is still numerically singular.
- Exact ratios conditioning on a stratum of a composite joint that the law never observed.
- GML edges to undeclared nodes and duplicate node ids; DOT edge attributes a reader cannot represent, including a circle mark in an ADMG and half-specified marks; non-finite shift interventions; response influence row ids past the u32 range.

### Compatibility

Rust

- `antecedent_io::CausalQueryWire::AnomalyAttribution.reference` is `Option<Option<(f64, f64)>>`: the outer `None` records a 2.0.0 body without the key. Code that builds or matches this variant must add the extra level.
- `antecedent_expr` `with_world_bound_leaves` takes the cited regimes. `antecedent_stats::ridge_gram_inverse` returns `Result<Option<_>>`.

Python

- `CausalResourceError` and `CausalSerializationError` do not subclass `ValueError`; code that caught `ValueError` around checked transport programs, catalogs, or transported-view loading must catch the typed classes.

### Known limitations

Recorded for 2.2 in the roadmap: sequential Bayesian updating hydrates a diagonal prior and can be overconfident against pooling (see [priors](docs/priors.md)); the design ranker's entropy score is not monotone in the design amount.

## 2.1.0

Delta from 2.0.0. Antecedent 2.1.0 keeps the `identify → estimate → inspect → refresh → export → consume` lifecycle and adds design-family studies, restricted-experiment transport, a fixed-DAG natural direct effect, and model-scoped Bayesian routes. Every licensed route executes from a checked operation retained at preparation.

### Added

#### Design-family studies

Retained `prepare` / `analyze` queries, on stage modules rather than the package root. On the experiment, policy, regime, and interference queries, row-aligned inputs may be data column names. Prepare resolves them and freezes the design. Refresh reuses that frozen design and accepts outcome-only data; it does not re-read design columns. Interval claims for these families are rows of [the graphless support matrix](docs/graphless-support-matrix.md), separate from the geometric matrix. A row is a support license for the named claim. It is not a coverage record. A coordinate the matrix does not name has no support license; a point may still be returned.

- `experiment`: Bernoulli, complete, stratified, cluster, fixed-cell 2×2 factorial, and independent multi-arm randomized effects; switchback sequences; Wald complier effects; one-sided treatment on the treated; fixed-coefficient CUPED; ANCOVA on pre-assignment covariates. The ANCOVA graphless row names one or two covariates; three or more stay point-only. `factorial.estimate` is a separate point utility for 2×2 Bernoulli cell means, main effects, interaction, and covariance-free variance bounds, and has no graphless row.
- `policy`: held-out value of a precomputed binary or multi-action policy under known randomized propensities, including doubly robust and cross-fitted scores, finite-class regret, uplift bins, and a kernel-smoothed group-to-dose policy under a declared known density. Binding global capacity or budget leaves the scalar value point-only.
- `regimes`: prespecified static or history-adaptive binary regimes. Graphless rows name a two-period g-formula under a caller-declared fixed known reward law, two- and three-period sequential doubly robust scores under caller-declared subject-excluded folds, and an additive marginal structural model with stabilized weights and subject-clustered CR1 interval claims. `evaluate_regime_value`, `evaluate_sequential_gformula`, and `evaluate_sequential_doubly_robust` are point utilities and do not verify that supplied predictions were fit out of fold.
- `quasi`: repeated-cross-section 2×2 difference in differences; balanced two-period panel and repeated-cross-section panel routes; staggered adoption against never-treated controls, with a graphless row for post-adoption event-time interval claims; synthetic control, ridge-augmented synthetic control, and synthetic difference-in-differences, including an exact uniform-assignment test of a stated constant effect and no effect-interval row; augmented panel difference in differences (point-only); fuzzy regression discontinuity and regression kink as local-polynomial ratios. Sharp regression discontinuity is the query `quasi.SharpRegressionDiscontinuity`.
- `survival`: individually randomized RMST and fixed-horizon survival contrasts, delayed entry, caller-supplied known censoring weights, and competing risks. Simultaneous full-grid survival and cumulative-incidence bands are separate graphless rows. No row names a competing-risk scalar incidence interval. Supplied censoring functions are held fixed.
- `InterferenceQuery` keeps the 2.0 geometric Bernoulli neighbor-count cell and adds graphless rows for cluster-randomized total exposure, two-stage saturation contrasts, and observational network exposure under caller-supplied known propensities. Externally fitted propensities stay point-only.

#### Restricted experiments (z-transport)

- Identifies a target effect from one source's experiments on a declared controllable set, for finite discrete diagrams of up to twelve observed and four controllable variables. A positive decision is a checked formula bound only to the joints it cites. A line-11 obstruction is certified from the selection diagram and rechecked on replay. Two sources are searched separately; complementary factors combine under disconnected outcome components or intervention-separated outcome groups, and a connected c-factor that would require a joint over both sources' interventions is refused as `z_transport.multi_source_combination_not_searched`.
- Outcome-kernel sensitivity reports an assumption range, witnesses, and a tipping fraction for a compatible fixed graph, including one declared treatment-level slice or one categorical non-treatment parent factor whose parents sit in the bound joint law. The range is not a sampling interval. Unsupported external-parent factors and joint deviations are refused.
- Hypothetical catalog deltas, a cost-ranked planner over checked sufficient additions, and arrival checks against the actual catalog. Cost is caller-supplied and is not a success probability. The versioned point artifact embeds the proof, catalog binding, laws, program, and premises digest; a consumer rechecks them and recomputes the point. Not-certified, budget, and cancellation outcomes carry structured inspection (reason kind, explored region or limits receipt) rather than a fabricated proof.
- Empirical cited tables publish a nominal percentile bootstrap and, under `empirical_support_bayesian_bootstrap` or `state_space_dirichlet`, a posterior equal-tail interval. Both are labelled `estimator_grid_not_measured` and have no coverage record. Exact laws stay point-only. The route prepares, exports, loads, inspects, and refreshes as a study. The [transport coverage matrix](parity/transport_coverage.md) is the scope table. Publishing the interval is not a calibration claim.

#### Nested counterfactuals

- `NestedCounterfactual` on an explicit three-node Markovian mediation DAG. Both treatment worlds replay one abduced exogenous table. A fitted non-separable outcome basis carries the mediator's abduced disturbance into the direct effect; a separable outcome does not; an underdetermined basis is a typed refusal. The effect is point-only. Other graphs, Bayesian inference, and refutation suites are refused. See [counterfactual coverage](parity/counterfactual_coverage.md).

#### Bayesian routes

Geometric-matrix additions relative to 2.0.0: Bayesian anomaly and change attribution; Bayesian trial transport; Bayesian finite-network interference; Bayesian intervention response on CPDAG and PAG class posteriors, including cheap and full validation. Conditional effects accept multiple declared modifiers and evaluate non-Gaussian levels on the outcome scale through the inverse link. Response curves can report a simultaneous band from the same joint posterior draws on a fixed grid.

Specialist procedures, scoped to their models: quadratic g-computation (`bayesian.basis.gcomp`), cross-fitted robust average effect (`bayesian.robust_ate`, interval labelled a modular bootstrap pushforward), bandwidth-conditional local-linear sharp RD (`rd.bayesian_local_linear`), and joint-Gaussian linear IV (`iv.bayesian_joint_linear`). They share the explicit-DAG Bayesian average-effect cell with `bayesian.gcomp`. That cell's coverage records name `bayesian.gcomp`. These four estimators have no coverage record of their own. Bayesian transport intervals draw each cited dataset from its own stream, and a cited law's probabilities must reconcile with its empirical counts before a point and its interval are published together.

#### External providers

- `extensibility.CausalProviderSpec`, `ProviderRegistry`, and `ProviderQuery`. Providers are registered explicitly or loaded from an `antecedent.providers` entry point; import does not scan plugins. `analyze` returns `ProviderAnalysisResult` with the provider output, uncertainty semantics, provenance, and trust boundary. `ProviderRegistry.verify` runs caller-supplied fixtures and records a digest-bearing report; only exact verified requests are `verified_extension`. Results pass a native envelope check. Plugin output is not a native license. `handoff.EconMLSpec.as_provider` adapts a caller-owned fit callback through the existing artifact encoder.

### Changed

- Preparation of a licensed route seals the identification proof, class envelope or posterior atoms, procedure, inference settings, and validation suite into the prepared plan. Estimate and refresh run from that plan after the builder is released. A refresh refuses a schema change. A one-shot run prepares and executes the same operation and reports the reused identification. This covers static CPDAG and PAG effects, graph-posterior effects, temporal class effects, temporal mediation, static and temporal response, Bayesian specialist procedures, trimmed and untrimmed AIPW, trial transport, and the public transport stages. Sealed static routes keep their projection and tier-replicate diagnostics.
- Each licensed cell in `parity/support_licensed.toml` cites builder-independent evidence for every estimator it licenses, and each of the 472 cells has a checked-execution citation. The geometric matrix is 472 licensed of 1475 meaningful cells (2.0.0: 463 of 1403). Of the 472, 424 cite coverage records in the registry, 44 are `estimator_grid_not_measured`, and 4 are `no_interval_reported`. The calibration pass for this cut has been run and the coverage surface is attested; a support license still is not a coverage record.

### Breaking changes from 2.0.0

Python

- `analyze` no longer accepts `running_variable`, `cutoff`, or `bandwidth`. Sharp regression discontinuity is `quasi.SharpRegressionDiscontinuity` passed as the query. Derivative and response routes that already took bandwidth through `estimator_config` are unchanged.
- Loading an exported result whose route executes from a sealed checked operation keeps the recorded answer and identities, but `loaded.acceptance.verified` is false and `acceptance.status` is `"sealed"`. `acceptance.unresolved` names the checked operation an independent consumer cannot replay. Code that treated `verified` as the only accepted state must accept `sealed`.
- Transport binding failures raise `CausalCancelledError`, `CausalResourceError`, `CausalUnsupportedError` (with a registered reason code), `CausalSerializationError`, or `CausalValueError`. In 2.0.0 several of these paths raised a bare `ValueError`. `CausalValueError` still subclasses `ValueError`; the other four do not.

Rust

- `IdentificationError` gains variants for cancellation, budgets, invalid input, unsupported input, invalid catalogs, missing evidence, and invalid derivations. The enum was already `#[non_exhaustive]`, so downstream matches keep compiling through their wildcard arm; code that routed every error through that arm now sees these cases there.
- `EstimationError::Refused` carries the transport reason codes. `IoError` (also `#[non_exhaustive]`) gains `Refused` and `ZTransport` variants.
- z-transport identification, verification, failure snapshots, the planner, and proposal replay take `SidLimits` and an `ExecutionContext`. Consumption takes `ZTransportConsumeLimits`. These entry points did not exist in 2.0.0.

## 2.0.0

Antecedent 2.0.0 keeps the established `identify → estimate → inspect → refresh → export → consume` lifecycle across discovery, graph uncertainty, identification, estimation, validation, temporal and response analysis, Bayesian inference, attribution, design, state, and artifacts.

### What's New?

#### Learners and heterogeneous effects

- Adds a Rust-native prediction layer with stable learner specifications, capability checks, zero-copy design views, row-indexed folds, and fold-local preprocessing. The causal estimators remain independent of a particular ML provider.
- Adds controlled linear, ridge, logistic, elastic-net, gradient-boosted-tree, random-forest, extra-tree, and CPU-native neural nuisance routes, plus restrained automatic nuisance selection where licensed. The standard Python wheel includes the neural route.
- Adds reusable out-of-fold predictions, fold assignments, nuisance diagnostics, overlap/trimming disclosure, and implementation provenance.
- Adds DML, cross-fitted AIPW, DR-Learner, and honest causal-forest paths. Forest leaf dispersion remains a diagnostic, not inferential uncertainty. The DR-Learner also supports pointwise CATE inference at prespecified profiles that exactly match retained covariate tuples in both arms and pass the propensity-overlap check; it uses cross-fitted DR scores and linear-final-stage HC0 covariance. Penalized final stages withhold inferential standard errors. The pointwise normal bounds are uncalibrated and are not simultaneous confidence bands; synthetic known-truth checks do not establish interval coverage.

#### Structural transport

- Adds explicit population, environment, evidence-regime, sampling, and dependence contracts. Separate experimental marginals never silently become a joint experiment, and a proposed intervention never becomes observed evidence.
- Adds population- and regime-aware functionals with checked derivation DAGs, theorem-scoped single-source identification, target-only identification, checked non-transportability witnesses, and bounded catalog search.
- Adds exact finite-discrete evaluation and prepared statistical transport with factor-level provider bindings, support diagnostics, joint bootstrap uncertainty, complementary-source synthesis, and target response grids.
- Adds durable transport artifacts and migrations that preserve evidence and source lineage across Rust, Python, and independent consumers.
- Distinguishes an unavailable joint/provider from invalid input, theorem-stage non-certification, structural non-transportability, support failure, numerical failure, and budget/cancellation.

#### Claim integrity and release boundaries

- Strengthens typed support, evidence, provenance, calibration, artifact, and refusal contracts across the expanded workflow.
- Makes calibration scope part of the result contract: `calibrated` requires a coverage record that applies to and attests the executing code.


### Breaking changes from 1.11

Python

- The theorem-stage transport names moved from `antecedent.transport` to
  `antecedent.transport.advanced` and are no longer re-exported: `DirectFormula`,
  `NonTransportableCertificate`, `PopulationFactor`, `RecursiveFactorizationFormula`,
  `SelectionDiagram`, `StandardizationFormula`, `TransportCertificate`,
  `TransportIdentification`, `TransportQuery`, `TrialTransportEstimate`,
  `estimate_trial_effect` and `identify`. `OverlapDiagnostic` and
  `TransportOverlapReport` are no longer exported anywhere; they are the type of
  `TrialTransportEstimate.overlap`. `antecedent.TransportQuery` is gone from the
  package root (use `antecedent.transport.advanced.TransportQuery`, or the
  ordinary-question wrapper `antecedent.transport.Transport`). Each old spelling
  raises an `AttributeError` naming its new home; see
  [the migration page](docs/migrations/2.0-transport-day1.md).
- `AcceptedGraph.asserted` and `AcceptedGraph.accepted` are removed: one spelling
  per object, `AcceptedGraph.from_graph` and `AcceptedGraph.from_discovery`.
- Graphs built from a discovery result carry every analysed variable, isolated
  ones included, taken from the new `PcmciDiscoveryResult.variable_names`; a
  result without names raises instead of defaulting to `x`/`y`.
- A refusal's `reason_code` is attached by the native layer; the `reason=<code>:`
  message prefix is no longer parsed into one.
- `PosteriorArtifact.draws` is a getter returning a float64 array.
- Without the native extension and without installed package metadata,
  `antecedent.__version__` is `"unknown"` (it used to be a stale literal).

Rust crates and features

- The Cargo feature `ml-gpu` is renamed `ml-neural` on `antecedent`,
  `antecedent-estimate` and `antecedent-learn`: it is a CPU-only Burn
  `NdArray<f32>` network and there is no GPU backend. The empty features
  `ml-faer` (`antecedent-learn`) and `hmc`, `smc` (`antecedent-prob`) are removed;
  the code they named is always compiled.
- `antecedent-identify::oracle_dot` (the frozen-oracle DOT parser, which panics on
  malformed input) is available only with the new `test-util` feature.
- `IdentityDomain` gains `LearnedTrial` and `TransportCertificate`
  (`IdentityDomain::ALL` is now 13 entries); learned-trial and transport-certificate
  artifact ids are derived in their own domains and differ from any 1.11 digest.
- The graph, posterior and analysis-result conversion functions re-exported from
  `antecedent::io` (`dag_from_dot`, `cpdag_to_json`, `encode_causal_posterior`, ...)
  return `IoError` instead of `CausalError`.

Wire formats and readers

- Readers reject duplicate section ids in a container, trailing bytes after a
  CBOR document, and unknown keys in query and response-query documents
  (`deny_unknown_fields`); a payload that carried a misspelled key used to load
  with the default and now fails.
- A posterior artifact's stored summaries must describe its embedded draws (the
  quantiles to 1e-9 relative, the mean and SD within eight standard errors); one
  whose summaries do not is rejected. External-estimate receipts report
  `evaluated_domain` as `"evaluated"` (it was `"unknown"`), and the analysis
  artifact gains an optional `HedgeCertificateWire`.
- Numerical behaviour changed where a result would have looked stronger than its
  evidence (for example Monte Carlo Shapley returns `Cancelled` unless every
  permutation finished, and `PriorSensitivity::evaluate` refuses when the estimator
  already carries a prior); the per-crate fixes are recorded in the commit history.
