# Measurement models

A measured column is not the scientific quantity it is recorded for. A
measurement model states how they relate, for which population, regime and
arm, and what may be concluded. It is separate from the schema's
`MeasurementSpec` flag, so no schema digest changes.

## Kinds

Six kinds are distinct in type and in refusal: observed, latent scientific,
noisy or misclassified, probabilistic feature assignment, learned vector
representation, derived. A model relates one non-latent *indicator* to one
latent scientific *construct*. Offering a proxy, a learned representation or a
derived column as the construct is refused (`proxy_offered_as_construct`), as
is naming the measured column as the construct.

## Binding and structure

A model is bound to a population, a regime and an arm. Error that differs by
arm is an explicit per-arm declaration; error that differs by population or
time has no bounded route and is refused, never pooled with the
non-differential route (`differential_error_not_pooled`).

## Representations

A representation has a coordinate system and a version. A change needs an
explicit mapping; a mapping carries values only and carries intervention
meaning only when it declares the evidence for it. Coordinates of a learned
vector representation are not intervenable. (`representation_mapping_required`)

## Routes

| Route | Correction | Refused |
|---|---|---|
| classical continuous error | corrected normal equations (= regression calibration for a linear model), known or validated error variance; through `estimate_continuous`, logistic regression calibration and SIMEX | the `nonlinear_outcome_model` operation of `check_operation` (a logistic coefficient enters only through `estimate_continuous`) |
| repeated measurements | same, error variance and ICC from the sample's own replicates | normal-theory variance intervals; the `nonlinear_outcome_model` operation, as above |
| binary misclassification | Rogan-Gladen prevalence and risk difference, Youden index above the declared minimum | exposure misclassification |
| probabilistic categorical | confusion-matrix inversion, category proportions and outcome means | per-unit soft assignments |
| external latent posterior | posterior-mean substitution, or Rubin's rules for outcome-conditioned draws | any trust inferred from metadata |

Each route's assumptions, allowed operations and refused operations are in
`antecedent_contract::measurement::route_card`.

## Continuous classical error: calibration and SIMEX

`PreparedMeasurementModel::estimate_continuous` (Python
`PreparedMeasurementModel.estimate_continuous` with `ContinuousErrorData`)
corrects one treatment coefficient for classical additive error `W = X + U` in
a continuous column. It re-runs the prepare checks of the linear adjusted
effect, then:

| Choice | Options | What happens |
|---|---|---|
| error location | treatment | the error-prone column is the regressor; its coefficient is reported |
| | covariate | the error-prone column is the regressor and the treatment is an error-free covariate at a declared column; correcting the covariate removes the confounding it leaves in the treatment coefficient |
| | outcome, identity link | no correction: classical outcome error does not bias a slope and the sandwich variance already includes it; the record says `outcome_error_no_correction` |
| | outcome, logit link | refused (`route_not_supported`): a 0/1 outcome cannot carry additive continuous error; next attempt: the binary misclassification route |
| outcome link | identity, logit | linear or logistic outcome model |
| method | regression calibration | the outcome model on the best linear predictor of `X` given `W` and `Z`, from sample moments with `Var(W)` replaced by `Var(W) - sigma_u^2 / k` (per-row replicate counts allowed); refused when the implied reliability is at or below the model's floor. Exact for a linear model (it equals the corrected normal equations); a first-order approximation for a logistic model, good when the error or the coefficient is moderate |
| | SIMEX | added error `lambda sigma_u^2 / k_i` on the grid `0.5, 1, 1.5, 2` (plus the naive `lambda = 0`), `B` remeasurements per value, the averaged coefficients extrapolated to `lambda = -1` by a quadratic (default) or linear least-squares fit; the per-`lambda` means are returned. The extrapolant is approximate: at reliability 0.8 a linear slope of 2 extrapolates to about 1.94 |

The error variance comes from the model: declared with its standard error
(classical route) or estimated from the sample's replicates (repeated route).
Any other route, or a repeated route without replicated units, is refused with
`measurement_not_identified`; next attempt: supply replicates or a validation
sample. `antecedent_stats::continuous_error::error_variance_from_reliability`
converts a declared reliability ratio `lambda_r` and the observed `Var(W)` into
`sigma_u^2 = (1 - lambda_r) Var(W)` with a delta-method variance; that value is
data dependent (computed from the measurement it corrects) and its variance is
first order.

Uncertainty: linear regression calibration keeps the delta method above.
Logistic calibration and SIMEX (either link) bootstrap the whole procedure:
rows are resampled with their replicates, the error variance is redrawn from
`N(sigma^2, se^2)` truncated at `1e-3 sigma^2` or re-estimated from the
resampled replicates, and the correction is re-run, at least 50 times. Each
replicate draws from its own keyed stream of the request's seed, so the result
is bitwise reproducible at any thread count; buffers are funded from the
execution context's admission pool first, and a pool that cannot fund them
refuses before anything is allocated. More than a tenth of the replicates
failing refuses. The variance is split into the measurement-parameter part
(derivative in `sigma^2` by a central difference under common random numbers,
times the variance of `sigma^2`) and the remaining sampling part. The standard
error is uncalibrated, no interval is produced, and the result has no artifact
layout (`export` refuses).

## Uncertainty

Variances are first-order delta-method variances that include the uncertainty
of the estimated measurement parameters (error variance, sensitivity and
specificity, confusion columns, between-imputation variance). The record says
whether parameter uncertainty was propagated, declared exact, or not
representable (`ParameterUncertainty`). The standard error is an uncalibrated
estimate: no interval is produced and an interval request is refused
(`cell_not_licensed`) until the whole measurement-plus-estimation method has a
measured coverage record.

## Artifact

`measurement_model_v1` carries the model, its measurement,
population, regime, arm, representation and evidence identities and an optional corrected functional
with its sufficient inputs (moments, counts, class summaries or
per-imputation fits). A consumer holding the bytes and the retained model
identity recomputes the point and standard error with `antecedent-stats` only
and refuses any stored number that does not recompute.

## Facade and lifecycle

`StudyBuilder::measurement_model` prepares a corrected functional: the model,
the analysis binding (construct, population, regime, arm, representation, an
explicit mapping when the representation changed, the error structure the
analysis assumes) and the outcome-side data of one bounded route. Every check
runs before a number is formed, through `antecedent_contract::measurement`:
a well-formed model, the construct (a measured column is never offered in its
place), the binding, the declared structure, the operation the route enters
and, for the posterior route, the exact attested or verified provider object.
The data binding of the estimator is derived from the same fields. Estimation,
export and consumption follow the sections above; the result carries the model
identity, the `uncalibrated_standard_error` label and a closed interval.

What `refresh` does, per route:

| Route | `refresh` with new outcome-only data keeps | recomputes | refuses |
|---|---|---|---|
| classical continuous error | model, identities, binding, route, provider label | coefficients, covariance with the error-variance term, moments, data identity | data of another route family |
| repeated measurements | same | replicate-based error variance and ICC, coefficients, covariance | another route family; fewer replicated units than the model declares |
| binary misclassification | same | corrected prevalence or risk difference, variance parts | another functional or arm pair; Youden index at or below the minimum |
| probabilistic categorical | same | corrected proportions, outcome means, covariances | another functional or levels; a confusion matrix worse conditioned than declared |
| external latent posterior | same, including the attested or verified trust label | posterior-mean fit or per-imputation fits and their combination | another route family; fewer draws than declared |

Validation parameters are not data. `with_validation_parameters` is the only
way they change: it rebuilds the measurement and evidence identities (so the
model identity), re-runs every prepare check and recomputes. It refuses
parameters of another route and a different provider object. Both operations
return a new value, so a refused update leaves the prepared state untouched.

A relabeling mapping carries values across: the correction runs on the values
as recorded and the estimate reports the mapping identity. It never carries
intervention meaning without declared evidence, and the route itself refuses an
intervention on the recorded indicator.

## Python surface

`antecedent.measurement_models` states the model (`MeasurementModel`,
`MeasurementVariable`, the five parameter classes, `ParameterEvidence`,
`RepresentationVersion`, `RepresentationMapping`), the analysis
(`AnalysisBinding`) and the data (`ClassicalLinearData`, `RepeatedLinearData`,
`BinaryPrevalenceData`, `RiskDifferenceData`, `CategoricalData`,
`LatentPosteriorData`), and `prepare_measurement_model` runs every prepare
check. `PreparedMeasurementModel` offers `estimate`, `estimate_continuous`
(with `ContinuousErrorData`, returning `ContinuousMeasurementEstimate`), `refresh`,
`with_validation_parameters` and `check_operation`; each returns a new object
and a refusal leaves the prepared object untouched. `MeasurementEstimate`
carries the point, the naive point, the uncalibrated standard error and the
variance parts, exports with `export`, and `interval()` raises
`cell_not_licensed`. `consume_measurement_model_artifact` recomputes the stored
numbers against the retained model identity. A posterior provider is stated
with `ProviderObject` (an attestation, or a verification receipt the library
issued); a claim of native trust is refused by name. Refusals are
`MeasurementModelRefusal`.

## References

Carroll, Ruppert, Stefanski and Crainiceanu (2006), *Measurement Error in
Nonlinear Models*, doi:10.1201/9781420010138. Cook and Stefanski (1994),
Simulation-extrapolation estimation in parametric measurement error models,
JASA 89:1314-1328, doi:10.1080/01621459.1994.10476871. Rogan and Gladen (1978), Am J
Epidemiol 107:71-76, doi:10.1093/oxfordjournals.aje.a112510. Rubin (1987),
*Multiple Imputation for Nonresponse in Surveys*, doi:10.1002/9780470316696.
