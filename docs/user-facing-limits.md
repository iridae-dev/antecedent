# What the transport and uncertainty routes do not do

This page lists, for each delivered package, what it does not do, the request
most likely to be refused, and the remedy. It states limits only; the guides
state what each route does. Every route here is point-only, a structural
envelope, an assumption range, or derived-unmeasured. None licenses a
confidence interval, a posterior coverage statement or a calibration claim.

| Package | Does not do | Most likely refused request | Remedy |
| --- | --- | --- | --- |
| [Transported cohort effect](guides/transported-decisions-and-measurement.md) | Test the selection premise; estimate overlap or a density ratio; inherit the source model's standard error | An interval or standard error for a cohort contrast | Use the point; an interval needs a measured record at that coordinate |
| [Measurement transport](guides/transported-decisions-and-measurement.md) | Propagate the uncertainty of the invariance premise; pool differential error | Sensitivity-and-specificity correction after declaring specificity only | Declare every parameter the route uses, or validate the model in the target |
| [Mechanism transport](guides/mechanism-and-synthesis-transport.md) | Carry a mechanism behind a selection arrow; carry tables, non-Gaussian or externally bound mechanisms; average with a native refit | A mechanism the diagram marks as shifted | Refit it in the target; use finite transport for tables |
| [Transported synthesis](guides/mechanism-and-synthesis-transport.md) | Decide dependence between source and native estimates; give an interval or posterior coverage | A transport with no declared uncertainty, several claims through one transport without covariance, or contributions carried to different targets | Declare the transport variance once and the shared covariance |
| [Surrogate bridge transport](guides/mechanism-and-synthesis-transport.md) | Discharge the transport premise; substitute for an identified effect | A bridge whose terminal outcome has a selection arrow into it | Validate the surrogate relation in the target |
| [Transported policy, adaptive source, period shift](guides/transported-policy-and-adaptive-source.md) | Shift a transition, observation or utility mechanism; learn a drift model; give an interval | A shifted dynamics mechanism, a target law outside the source support, or an adaptive trial with no declared trial population | Evaluate with a model fitted in the target |
| [Discovered-structure transport](guides/discovered-structure-transport.md) | Pick a class member; weight members; certify non-transportability; handle PAGs or temporal structure | A request over a class too large to enumerate or with unresolved pairs | Add background knowledge or declare a graph |
| [Per-population discovery](guides/discovered-structure-transport.md) | Pool rows; test invariance; turn an edge difference into selection targets | Stacked populations or relabelled duplicates | Use the pooled algorithm directly with declared assumptions; supply disjoint rows |
| [Sample split](guides/discovered-structure-transport.md) | State that one dataset's discovered graph is the true graph or control its error; split dependent, weighted or masked rows; attest an effect interval outside the measured family | An interval on the same rows that selected the graph, or for an unresolved structure | Split exchangeable rows; use a fully orientable table or a declared graph |
| [Provider-sourced transport](guides/provider-sourced-transport.md) | Execute a provider; prove a declared population; raise calibration | An attested-only provider where verification is required | Obtain a verification receipt covering the object |
| [Composed spread and exceedance](guides/composed-uncertainty.md) | Give coverage without a record at the composed coordinate | An interval for a composition | Use the spread, or measure the coordinate |
| [Whole-path uncertainty](guides/composed-uncertainty.md) | Propagate declared structure; handle fitted or sampled mechanisms; add the two spreads | A cross-world row without a unit-sampling declaration | Declare fixed units or a population sample |
| [Decision-reliability derived uncertainty](guides/decision-reliability-uncertainty.md) | Cover a wrong law or unretained candidates; publish without a record | A published interval, a ranking probability from a point-only producer, or a ranking of independently drawn candidates | Keep per-draw utilities from one joint law; await measurement |
| [Adaptive procedure uncertainty](guides/adaptive-procedure-uncertainty.md) | Handle more than the bounded design; claim coverage; cross paradigms | A posterior under value-dependent loss, or fixed-sample inference on an adaptive design | Model the loss; use the adaptive-weighting or anytime-valid paradigm |
| [Finite-SCM discretization](guides/finite-scm-discretization.md) | Bound variances, probabilities or counterfactuals; take latent loadings, non-Gaussian or bounded support | Levels left undeclared, or an interval on the discretized value | Declare levels; use the point and its bound |
| [Assumption range and revalidation](guides/assumption-range-and-revalidation.md) | Choose the bound; attach a probability; carry a record to another target | A range for a discovered structure, or calibration at a new target (it is withdrawn, never inherited) | Use the structural envelope; re-measure |
| [Inverse and decision reach](guides/inverse-and-decision-reach.md) | Supply a law from a point; supply an interval region; rank actions from scenario inversion; export a transported law | A quantile or probability on a point source, or a transported input to a consumer not declared to accept one (`cell_not_licensed`) | Supply a law (draws or a finite table), or accept the transported input explicitly with its receipt |
| [Interface-contract adapters and overlap](guides/interface-contract-adapters.md) | Infer support, estimand or calibration; treat unknown overlap as disjoint | Independence declared where overlap is unknown | Register the sources in the evidence ledger |
| [Parquet, Arrow IPC and large tables](guides/large-tables.md) | Read decimal, binary or nested columns; cast a type lossily; give a calibrated interval above 100,000 rows; use Polars as a backend; hold a table resident beyond the admission pool | A resident read, or a full-design route, on a table larger than the pool | Stream the weighted Gaussian summary, lower `chunk_rows`, raise `memory_limit_bytes`, or flatten and cast the column explicitly |
| [Staleness and drift diagnostics](guides/staleness.md) | Schedule or alert; correct repeated assessments over time for multiplicity (each assessment controls only its own family); learn a drift model; assess positivity for a non-binary treatment or a non-numeric adjustment set; check the plan's own prior (the Bayesian check uses a reference conjugate model of the outcome regression); detect a change of variance across periods; assess a panel or multi-environment study as a whole | New data with a changed schema, or with fewer rows (or rows per arm) than the policy's floors (`refuse`) | Project the new data onto the prepared schema or re-prepare; collect the rows the policy requires; assess each unit or environment table |
| [Missing data and multiple imputation](guides/missing-data.md) | Impute on any route other than the average effect; impute under missing not at random; run a sensitivity analysis for departures from missing at random; pool a graph-posterior mixture, a matching or stratification estimator, or a route without a variance; resample clusters on the average-effect route; publish a calibrated interval for the pooled estimate | A policy the route cell does not list in `parity/missing_policy_support.toml`, or `missing_not_at_random` (`missing_policy_not_applicable`) | Declare a policy the cell supports (the refusal names the next one); for missing not at random, a sensitivity analysis, which is not shipped |
| [Unbalanced panel preparation](guides/missing-data.md) | Estimate a difference-in-differences effect; impute a missing unit-period; borrow a lag across a gap or a unit; relabel units or periods | A design that needs balance (two-period panel, synthetic control, balanced event study, augmented panel) on an unbalanced panel | Declare `balanced_panel`, `available_case` where the design's contrasts allow it, or restrict to a common window |
| [Continuous measurement error](measurement-models.md) | Correct error whose variance is not identified; correct differential error; correct additive error in a binary outcome; give a calibrated interval (SIMEX and logistic regression calibration are approximate corrections with a bootstrap standard error) | A correction with no error variance, replicates or validation sample (`measurement_not_identified`) | Supply replicates or a validation sample; use the binary misclassification route for a binary outcome |

## Reading a refusal

A refusal carries a registered reason code, a typed detail and a remedy. The
codes are listed in the refusal registry. A refusal is not a failure of the
data; it states which premise or declaration was not met.

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
