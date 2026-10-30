# What the transport and uncertainty routes do not do

This page lists, for each delivered package, what it does not do, the request
most likely to be refused, and the remedy. It states limits only; the guides
state what each route does. Every route here is point-only, a structural
envelope, an assumption range, or derived-unmeasured. None licenses a
confidence interval, a posterior coverage statement or a calibration claim.

| Package | Does not do | Most likely refused request | Remedy |
| --- | --- | --- | --- |
| [Transported cohort effect](guides/transported-decisions-and-measurement.md) | Test the selection premise; estimate overlap or a density ratio; inherit the source model's standard error | An interval or standard error for a cohort contrast | Use the point; an interval needs a measured record at that coordinate |
| [Measurement transport](guides/transported-decisions-and-measurement.md) | Propagate the invariance premise's uncertainty through the declared-invariance route (only the pooled-differential route estimates the between-population variance); pool error for categorical, continuous or covariate-dependent models; give a calibrated interval | Sensitivity-and-specificity correction after declaring specificity only, or pooling with one validated source | Declare every parameter the route uses, validate a second source population, or validate the model in the target |
| [Mechanism transport](guides/mechanism-and-synthesis-transport.md) | Carry a mechanism behind a selection arrow; carry tables, non-Gaussian or externally bound mechanisms; average with a native refit | A mechanism the diagram marks as shifted | Refit it in the target; use finite transport for tables |
| [Transported synthesis](guides/mechanism-and-synthesis-transport.md) | Decide dependence between source and native estimates; give an interval or posterior coverage | A transport with no declared uncertainty, several claims through one transport without covariance, or contributions carried to different targets | Declare the transport variance once and the shared covariance |
| [Surrogate bridge transport](guides/mechanism-and-synthesis-transport.md) | Discharge the transport premise; substitute for an identified effect | A bridge whose terminal outcome has a selection arrow into it | Validate the surrogate relation in the target |
| [Transported policy, adaptive source, period shift](guides/transported-decisions-and-measurement.md) | Shift a transition, observation or utility mechanism; learn a drift model; give an interval | A shifted dynamics mechanism, a target law outside the source support, or an adaptive trial with no declared trial population | Evaluate with a model fitted in the target |
| [Discovered-structure transport](guides/discovered-structure-transport.md) | Pick a class member; weight members; certify non-transportability; handle PAGs or temporal structure | A request over a class too large to enumerate or with unresolved pairs | Add background knowledge or declare a graph |
| [Per-population discovery](guides/discovered-structure-transport.md) | Pool rows; test invariance; turn an edge difference into selection targets | Stacked populations or relabelled duplicates | Use the pooled algorithm directly with declared assumptions; supply disjoint rows |
| [Sample split](guides/discovered-structure-transport.md) | State that one dataset's discovered graph is the true graph or control its error; split dependent, weighted or masked rows; attest an effect interval outside the measured family | An interval on the same rows that selected the graph, or for an unresolved structure | Split exchangeable rows; use a fully orientable table or a declared graph |
| [Provider-sourced transport](guides/transported-decisions-and-measurement.md) | Execute a provider; prove a declared population; raise calibration | An attested-only provider where verification is required | Obtain a verification receipt covering the object |
| [Composed spread and exceedance](guides/composed-uncertainty.md) | Give coverage without a record at the composed coordinate | An interval for a composition | Use the spread, or measure the coordinate |
| [Whole-path uncertainty](guides/composed-uncertainty.md) | Propagate declared structure; handle fitted or sampled mechanisms; add the two spreads | A cross-world row without a unit-sampling declaration | Declare fixed units or a population sample |
| [Decision-reliability derived uncertainty](decision-reliability.md#derived-intervals-and-confidence-sets) | Cover a wrong law or unretained candidates; publish without a record | A published interval, a ranking probability from a point-only producer, or a ranking of independently drawn candidates | Keep per-draw utilities from one joint law; await measurement |
| [Adaptive procedure uncertainty](guides/composed-uncertainty.md#uncertainty-of-an-adaptive-procedure) | Handle more than the bounded design; claim coverage; cross paradigms | A posterior under value-dependent loss, or fixed-sample inference on an adaptive design | Model the loss; use the adaptive-weighting or anytime-valid paradigm |
| [Finite-SCM discretization](cross-world-execution.md#discretize-a-gaussian-composition-into-a-finite-model) | Bound variances, probabilities or counterfactuals; take latent loadings, non-Gaussian or bounded support | Levels left undeclared, or an interval on the discretized value | Declare levels; use the point and its bound |
| [Assumption range and revalidation](guides/assumption-range-and-revalidation.md) | Choose the bound; attach a probability; carry a record to another target | A range for a discovered structure, or calibration at a new target (it is withdrawn, never inherited) | Use the structural envelope; re-measure |
| [Inverse and decision reach](guides/inverse-outcome.md#inverse-queries-on-composed-cohort-and-transported-sources) | Supply a law from a point; supply an interval region; rank actions from scenario inversion; export a transported law | A quantile or probability on a point source, or a transported input to a consumer not declared to accept one (`cell_not_licensed`) | Supply a law (draws or a finite table), or accept the transported input explicitly with its receipt |
| [Interface-contract adapters and overlap](guides/build/scientific-interfaces.md#adapt-declarations-you-already-have) | Infer support, estimand or calibration; treat unknown overlap as disjoint | Independence declared where overlap is unknown | Register the sources in the evidence ledger |
| [Parquet, Arrow IPC and large tables](guides/large-tables.md) | Read decimal, binary or nested columns; cast a type lossily; give a calibrated interval above 100,000 rows; use Polars as a backend; hold a table resident beyond the admission pool | A resident read, or a full-design route, on a table larger than the pool | Stream the weighted Gaussian summary, lower `chunk_rows`, raise the execution's memory limit, or flatten and cast the column explicitly |
| [Sequentially randomized designs and regime survival](guides/temporal-designs.md) | Give a calibrated interval or posterior; identify a static switchback (assignment is a sequence of periods); infer an assignment probability or dose density; check sequential exchangeability for regime survival; pool error for a misclassified outcome that differs by arm or period; carry a band across a misclassification correction | A design option the route does not read (`bootstrap=` on an analytic interval), a local-quadratic grid with no curvature bound, or a realized probability below the floor | Omit the option; declare `curvature_bound` or use the smoothed curve; lower the floor or redesign the assignment |
| [Staleness and drift diagnostics](guides/staleness.md) | Schedule or alert; correct repeated assessments over time for multiplicity (each assessment controls only its own family); learn a drift model; assess positivity for a non-binary treatment or a non-numeric adjustment set; check the plan's own prior (the Bayesian check uses a reference conjugate model of the outcome regression); detect a change of variance across periods; assess a panel or multi-environment study as a whole | New data with a changed schema, or with fewer rows (or rows per arm) than the policy's floors (`refuse`) | Project the new data onto the prepared schema or re-prepare; collect the rows the policy requires; assess each unit or environment table |
| [Target-native validation and sensitivity ranges](guides/target-native-validation.md) | Measure the error rate of any refuter or pass threshold; let a scalar average-effect refutation certify another target (the claim refuses it); declare a simulated common cause in a graph posterior or a class with unresolved marks; bootstrap a series on a route with no weighted construction; protect placebo timing against a treatment that responds to the outcome within the shift; check an average or directional derivative or a Jacobian against a curve; replicate series in the posterior-predictive check; refute a partially identified envelope or a graph-dependent response | A refuter the study cannot construct (recorded in `refused` with the next attempt), or a claim carrying a refutation another program produced for a non-scalar target | Run the target-native suite on the claim's own target; declare a negative control when the graph supplies none; accept a DAG or temporal DAG; query point derivatives |
| [Missing data and multiple imputation](guides/missing-data.md) | Impute under missing not at random; run a sensitivity analysis for departures from missing at random; pool a graph-posterior mixture of a class, a latent-confounded graph, a series or a non-effect query, a route whose fit publishes no variance (a distribution, or a static mediation with no `bootstrap_replicates` on the imputation), or a multi-horizon Bayesian mediation (no joint draws); resample clusters on the static routes; publish a calibrated interval for the pooled estimate | A policy the route cell does not list in `parity/missing_policy_support.toml`, or `missing_not_at_random` (`missing_policy_not_applicable`) | Declare a policy the cell supports (the refusal names the next one); for missing not at random, a sensitivity analysis, which is not shipped |
| [Panel designs](guides/designs/panel-designs.md) | Give a calibrated interval (every rollout interval is unlicensed, calibration unmeasured); test parallel trends (pre-trend leads are diagnostics); run Sun–Abraham on an unbalanced panel; propagate supplied augmented-panel predictions' own error; give a conformal interval for several treated units or with fewer than 19 pre-periods; give a conformal guarantee under Bayesian inference (the Bayesian counterpart is a posterior predictive interval); reproduce the full rdrobust bandwidth selector | Sun–Abraham on an unbalanced panel, a never-treated comparison with no never-treated units, conformal intervals on a short pre-period, a resample under `dependent_time_series` | Use `callaway_santanna` or `imputation`, or declare `balanced_panel`; use `not_yet_treated`; use placebo variance; declare `clustered_panel` |
| [Unbalanced panel preparation](guides/missing-data.md) | Estimate a difference-in-differences effect; impute a missing unit-period; borrow a lag across a gap or a unit; relabel units or periods | A design that needs balance (two-period panel, synthetic control, balanced event study, augmented panel) on an unbalanced panel | Declare `balanced_panel`, `available_case` where the design's contrasts allow it, or restrict to a common window |
| [Continuous measurement error](measurement-models.md) | Correct error whose variance is not identified; correct differential error; correct additive error in a binary outcome; give a calibrated interval (logistic regression calibration is a first-order approximation and SIMEX retains extrapolation bias; both are approximate, uncalibrated corrections with a bootstrap standard error, and their bias across repeated datasets is assessed in the scheduled measurement phase) | A correction with no error variance, replicates or validation sample (`measurement_not_identified`) | Supply replicates or a validation sample; use the binary misclassification route for a binary outcome |
| [Bayesian intervention-level refuter state](support-matrix.md) | License the state or claim calibration (the cell reports `refused` while it executes); run under the default suite (an implicit request keeps the licensed validation none); refute a dose coefficient in place of the level (prior sensitivity refits the whole response and reads back the posterior-mean level); mix the predictive checks and prior sensitivity across graph-posterior atoms (they run on the primary identified atom, the plugin-level suite on the pooled level) | `refute="cheap"` on a temporal derivative off the identity scale expecting a finite-difference check (a diagnostic says it does not apply) | Use an identity-scale derivative for the consistency check |
| [Derivative refuter state](support-matrix.md) | License the state; apply ATE-shaped contrast refuters to a derivative; check finite-difference consistency for a directional derivative, a Jacobian, an order-two derivative off the identity scale, or a uniform or custom-weighted average derivative (a diagnostic says the check does not apply); resample a table whose columns cannot be read as floats | Expecting a mixed verdict across completions of a TemporalCpdag or TemporalPag derivative (each completion's Pulse-witness reports are kept) | Read the per-completion reports |
| [Temporal and class-aware intervention-level refuter state](support-matrix.md) | License the state; mix reports across horizons or across completions (each completion's and each horizon's reports are kept, and the completions' spread is reported); run on a Sequence class response | A Sequence (multi-step) intervention level on a TemporalCpdag or TemporalPag with cheap/full (it has no single Pulse witness, so the result carries no refutations) | Use a single-step policy, or validation none |
| [Frequentist intervals for attribution and counterfactuals](support-matrix.md) | License or calibrate the Frequentist mechanism-refit bootstrap interval (a requested replicate count runs it; the result's support status is `refused` and its calibration is unmeasured); resample clusters or a dependent series; re-select mechanism families per replicate (selection is held at the original fit, as on the Bayesian cell); form per-unit counterfactual intervals | A replicate count under a clustered or dependent-series sampling design (`option_not_applicable`) | Declare independent subjects, or use the point value; `inference=Bayesian` publishes the Bayesian-bootstrap posterior |
| [Temporal point-route uncertainty](support-matrix.md) | Claim coverage for the temporal cross-world, derivative, elasticity, Jacobian, conditional pulse and sustained, contemporaneous-projection ADMG or Bayesian lag-aligned attribution intervals (each is derived-unmeasured: no coverage record exists for the construction); resample a single series by rows (the frequentist interval refits every regression on shared circular blocks of series times, each row keeping its lag window); mix the posteriors of a class's completions (enumeration is not posterior mass, so several completions publish an interval for the identified set, never a mixed posterior); resample a contemporaneous-projection ADMG or a lag-aligned attribution by single rows, or by subjects or clusters the table does not label (a declared subject or cluster design is refused with the next attempt; both routes refit the whole procedure on circular blocks of consecutive rows, and the Bayesian side shares one Dirichlet weight per block); give a block length or short-series threshold measured for those row-block constructions (none has been measured, so every reading warns) | An interval with `bootstrap_replicates=0` (the point is published alone, with `estimate.temporal.point.uncertainty_withheld`), or one on a series whose regressions fit it exactly (zero replicate spread withholds the interval) | Request replicates (the default is 199, except a frequentist unit or nested counterfactual, which resamples only on an explicit `bootstrap_replicates`) or Bayesian draws; read the identified-set interval for a class with several completions |
| [Bayesian joint intervention on a tier closure](guides/tier-diagnostics.md) | License the Bayesian joint level (it executes unlicensed); estimate the saturated cell mean (the outcome regression is additive in the treatments, so the level is that model's joint level); run cheap or full on a tier closure | `refute="cheap"` on a CoDetermined joint level (`cell_not_applicable`) | Use validation none; use the Frequentist cell.aipw joint cell for the saturated cell mean |

## Reading a refusal

A refusal carries a registered reason code, a typed detail and a remedy. The
codes are listed in the refusal registry. A refusal is not a failure of the
data; it states which premise or declaration was not met.

## Not applicable versus refused

`not_applicable` is reserved for a cell whose estimand does not denote. A cell
whose route is not built is `refused`, and its reason names the next attempt.
Some refused cells execute and stay unlicensed (their closed rule says so): the
Bayesian Dag intervention-level refuter state, the static and temporal
derivative refuter states, the TemporalDag, TemporalCpdag, TemporalPag, Cpdag
and Pag intervention-level refuter states, and the Bayesian joint
InterventionResponse on a CoDetermined closure. The cells that stay not
applicable, with the reason each one does not denote:

- Pulse, sustained, temporal-conditional and temporal-mediation contrasts on a
  static graph class: they are indexed by a lag and a horizon, and a static
  class has no time index.
- Static ConditionalEffect and MediationEffect on TemporalDag, TemporalCpdag
  and TemporalPag, and TransportQuery and InterferenceQuery on those classes:
  the queries are typed against static graphs. An AverageEffect on one series
  under those graphs is the lag-0 contrast `E[Y_t | do(X_t)]`: it is compiled
  as the pulse at offset 0 read at horizon 1 (over a graph posterior too, where
  the per-atom proofs mix as for any temporal pulse) and executes unlicensed
  (support status `refused`, diagnostic
  `estimate.temporal_class.lag_zero_average_effect`).
- The Unknown and CoDetermined tier cells other than the ones they license: a
  tier background identifies through its closure (CoDetermined) or a
  two-scenario envelope (Unknown), and is only ever an explicit structure
  source; the ATE-shaped refuter suite has no state on either.
- ResponseCurve cheap and full: a function-valued curve has no ATE-shaped
  refuter state.

Pulse, sustained and temporal-conditional contrasts on one series under a TemporalAdmg are
identified on the unrolled window of the template (the complete ID algorithm, IDC for the
conditional contrast) and evaluated on the lag-aligned window table, with the diagnostic
`estimate.temporal_admg.window`, block resampling with a block never shorter than the window, a
series read as stationary over the template, and discrete data only. They are licensed when the
template is supplied, accepted or drawn from a graph posterior of lagged atoms, on the exact
enumerated truth of `conformance/estimate/temporal_window_truth`. Cheap validation refits the
plug-in on a random contiguous stretch of the windows and rotates the treatment columns together
by at least one window (the contrast must then be indistinguishable from zero); full validation
adds a 50% stretch and a split-half stationarity test, which a series whose effect vanishes
halfway fails. A graph posterior over contemporaneous TemporalAdmg atoms stays refused: it has no
lagged template to unroll. A window that holds a hedge is refused with the variables and
periods that carry it. Natural mediation on one series under a TemporalAdmg is identified on the
same window by the path-specific identifier (the recanting-district criterion applies to the
window ADMG) and is licensed with the total, natural direct and natural indirect effects
estimated on the same resampled windows, supplied, accepted or over a graph posterior of lagged
atoms. TemporalMediationEffect on a TemporalPag, supplied, accepted or over a graph posterior, is
licensed at every validation: the enumerated effects lie in the published identified set and the
mediation-native suite runs on every member. A ConditionalEffect on a TemporalAdmg read from tabular
rows is the conditional contrast of its contemporaneous projection, not the temporal conditional
contrast. Some contrasts denote but are not licensed in either inference mode. They are refused
with the reason rather than recorded as n/a. MediationEffect on an Admg, a Cpdag or a Pag
decomposes the same way over the structure (the class envelope of the path-specific executors)
and executes unlicensed at validation none. A TransportQuery reads a selection diagram. A Dag is the diagram with no latent confounding and runs the same sID identification and trial-to-target estimator (unlicensed); cheap and full validation run the transport-native suite (calibration of the supplied selection and treatment probabilities against the rows they describe, effective sample size of the weights, and a whole-procedure bootstrap range under full). A Cpdag or Pag names several diagrams and stays refused. InterferenceQuery is identified by its known assignment mechanism and exposure mapping, not by an edge, so it is stated under an Admg, TemporalAdmg (contemporaneous projection), Cpdag or Pag as well as a Dag and executes unlicensed; its cheap and full validation check the realized assignment against the declared design, report the smallest exposure probability and, under full, test by re-randomization that the contrast is unbiased around zero under the sharp null. The static switchback and
the static analogues of the dose-grid, sequential and experiment-bridge
designs are temporal by construction and have no static cell.

A posterior for a nearest-neighbour matching contrast is a principled typed
refusal (`route_not_supported`), not a missing route: a matching rule has no
likelihood for the outcomes or the assignment, so a prior has nothing to
update, and resampling the units is not a substitute. The refusal names the
next attempt: a Bayesian outcome model on the same covariates (the Bayesian
g-computation route), whose model-based effect on the treated is reported as
that different estimand.

## Labels that are never upgraded

- `point_only`: the value is one number under a declared premise.
- Structural envelope: a statement about which members of a class the rules
  certify, with no probability.
- Assumption range: the value at declared shifts of a premise, with no
  probability on the shifts.
- Derived-unmeasured: the method is derived and its law is exact under a stated
  model; no coverage has been measured.
- Transported: a claim about a target population carried with its transport
  identity; never presented as native target data.

## Sample size, series length and the measured range

A reported interval is `calibrated` only when its construction has a coverage
record whose measured range contains the execution's row count. The range is
the span of the record's sample-size grid points, and nothing beyond it is
inferred: three passing grid points license the span they bracket, not an
unlimited range, and a larger sample is `scope_not_assessed` with reason
`sample_size_outside_measured_range`. The ranges below are the designs the
calibration gate measures (`crates/antecedent/tests/practitioner_scale_calibration.rs`);
a design that fails at any grid point is recorded as a named boundary over its
whole range, and a result stays `scope_not_assessed` until a record exists.

| Construction | Measured range (design) | Beyond the range |
| --- | --- | --- |
| Linear adjustment, classical, HC1, cluster-robust (500 clusters) and 199-replicate bootstrap SE; cross-fitted AIPW, classical, HC1 and cluster-robust SE; twenty adjustment covariates | 1,000 to 100,000 rows | Not assessed. The point estimate and standard error are still returned; the interval carries no calibration claim |
| Default Bayesian (Laplace, 1,000 draws, prior scale 10) average effect | 1,000 to 100,000 rows | Not assessed above 100,000 rows. A request for fewer than 1,000 Laplace draws keeps its declared draws and summaries; its effect interval is the exchangeable-rank interval of those draws (expected posterior mass 0.95 at every count; the type-7 interval of 64 draws has 0.921), a separate construction (`rank_quantile_interval`) with its own records |
| CPDAG class graph posterior, joint influence-function SE | 1,000 to 100,000 rows | Not assessed |
| Frequentist temporal Pulse (horizons 1 and 2), single-step Sustained and Total, Direct and Mediated mediation effects (each contrast its own record), circular-block interval; default Bayesian Pulse and Bayesian mediation | 320 to 4,160 steps, so series of 1,024 and 4,096 steps with up to three lags are inside | Not assessed above 4,160 steps; the earlier records still cover 80 to 320 steps |
| Bernoulli ITT score, ANCOVA (one or two covariates) and fixed CUPED interval | No row count: 30 observed and 40 expected rows in each arm, probabilities 0.2 to 0.8 (ANCOVA: one common probability) | Refused with the arm rule named; 300 balanced rows, a 2:1 allocation and a 200-row trial at probability 0.2 are supported. Known-truth coverage is 2,000 allocations per cell from 80 to 600 rows ([graphless support matrix](graphless-support-matrix.md)) |
| Complete-randomization Neyman difference in means | 30 units per arm, unchanged | Its own construction; neither the arm rule above nor a row count applies |
| Small-arm Student interval of a complete randomization's difference in means and of the Bernoulli ANCOVA (one or two covariates): HC2 variance with Bell-McCaffrey degrees of freedom | A smaller arm of 20 of 300 participants, at the sample sizes 150 to 600 of its grid | Offered where the construction above is unsupported, from 100 rows with at least 2 in each arm and positive variance; it is its own construction (standard-error kind `hc2`), carries no calibration claim until a coverage record covers the execution, and is never graphless-licensed. Below 2 rows in an arm, below 100 rows or with zero variance the interval is withheld with the arm rule named |

Memory for the practitioner designs is bounded by the table: a 100,000-row,
22-column table is 17.6 MB, and the calibration replicates hold one table per
worker thread. A series is processed in place; the circular-block bootstrap
holds one resampled index vector per replicate.

Tables larger than these ranges, and larger than memory, are executable: Parquet
files, Parquet datasets and Arrow IPC files are read in bounded chunks under the
admission pool, row ids are checked 64-bit, and the weighted Gaussian / OLS
summary streams a file one chunk at a time ([large tables](guides/large-tables.md)).
Executing a larger table does not extend a measured range: its intervals remain
`scope_not_assessed` until a coverage record measures that size. A route that
needs the whole table resident is quoted before reading and refused, with the
quote and the next attempt, when the pool cannot hold it.
