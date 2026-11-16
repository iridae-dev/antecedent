# Measurement models

A measured column is not the scientific quantity it is recorded for. Use a
measurement model when a column records a quantity with error (a misclassified
diagnosis, a noisy dose, a provider's posterior for a latent trait) and the
analysis is about the quantity itself. The model states how column and quantity
relate, for which population, regime and arm, and what may be concluded; the
library then corrects one functional on one bounded route, or refuses. A
measurement model is separate from the schema's `MeasurementSpec` flag, so
declaring one does not change the schema digest.

## Minimal call

A prevalence measured by a test with declared sensitivity 0.9 and specificity
0.8, corrected by Rogan-Gladen:

```python
from antecedent import measurement_models as mm
from antecedent.outcomes import require_success

model = mm.MeasurementModel(
    indicator=mm.MeasurementVariable("schema:test_positive", "noisy_misclassified", "unit"),
    construct=mm.MeasurementVariable("construct:infected", "latent_scientific", "unit"),
    parameters=mm.BinaryParameters((mm.BinaryRates(0.9, 0.8, 1e-4, 4e-4),), min_youden=0.1),
    population_id="cohort",
    regime_id="observational",
    evidence=mm.ParameterEvidence.declared("lab"),        # who declared the rates
)
analysis = mm.AnalysisBinding(construct_id="construct:infected",
                              population_id="cohort", regime_id="observational")
prepared = mm.prepare_measurement_model(model, analysis, mm.BinaryPrevalenceData(410, 1000))
estimate = require_success(prepared.estimate())
estimate.point, estimate.naive_point   # 0.3 corrected, 0.41 as recorded
estimate.standard_error                # delta method, including the rates' variances
estimate.interval(0.95)                # Wald interval, calibration "not_measured"
```

`prepare_measurement_model` raises `MeasurementModelRefusal` for the first
failed check; `estimate` returns an `Outcome` whose refusal (for example a
Youden index at or below `min_youden`, `measurement_not_identified`) carries the
same code. Codes, details and remedies are listed under
[measurement model refusals](reference/refusal-codes.md#measurement-models).
The [transported measurement example](https://github.com/iridae-dev/antecedent/blob/v3.0.0/examples/python/transported_measurement.py)
carries a measurement model into another population.

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
| binary misclassification | Rogan-Gladen prevalence and risk difference, Youden index above the declared minimum | misclassification of an exposure (its own entry points, below) |
| probabilistic categorical | confusion-matrix inversion, category proportions and outcome means | per-unit soft assignments (their own entry point, below) |
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

Limits of the nonlinear corrections: logistic regression calibration is a
first-order approximation (it fits the logistic model on `E[X | W, Z]`, which is
exact only for a linear outcome model and accurate when the measurement error or
the coefficient is moderate), and SIMEX keeps extrapolation bias (the fitted
path in `lambda` is only an approximation of the true curve, so a quadratic
extrapolant under-corrects). Neither is shown to recover the generating
coefficient: on one simulated logistic dataset (generating slope 1, reliability
0.8) they return about 0.88 and 0.89, and that single-dataset value is a
regression pin, not accuracy evidence. Each implementation is checked against an
independent calculation on identical inputs (the same data, `lambda` grid,
replicates and remeasurement noise), which establishes that the code computes
the stated approximation, not that the approximation is unbiased. Their bias
across repeated datasets is not established: the route is approximate, makes no
coverage or accuracy claim, and its standard error is uncalibrated.

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
error is uncalibrated; `interval(level)` reports the Wald interval of the
corrected coefficient with calibration `not_measured`, and the result has no
artifact layout (`export` refuses).

## Uncertainty

Variances are first-order delta-method variances that include the uncertainty
of the estimated measurement parameters (error variance, sensitivity and
specificity, confusion columns, between-imputation variance). The record says
whether parameter uncertainty was propagated, declared exact, or not
representable (`ParameterUncertainty`). The standard error is an uncalibrated
estimate. The estimate's `interval(level)` is the Wald interval built from it,
reported with calibration `not_measured`; a route that derives no standard
error refuses the interval (`route_not_supported`).

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
identity, the `uncalibrated_standard_error` label and a Wald interval annotated
`not_measured`.

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

## Exposure errors

Four corrections have their own entry points in `antecedent.measurement_exposure`
(Rust `antecedent_estimate::measurement::exposure_routes`, numerics in
`antecedent_stats::measurement_error`); they are the numerical core of the composed
declarations described under [exposure measurement in the analysis](#exposure-measurement-in-the-analysis).
The entry points take their declarations as
arguments rather than through a prepared model, report a corrected value with
a variance split into named parts, an uncalibrated standard error and a Wald
interval with a `not_measured` calibration slot, and refuse what they cannot
identify.

| Correction | Corrects | Identified when | Refused |
|---|---|---|---|
| exposure misclassification | the exposure odds ratio and the exposure shares among cases and controls, for a sensitivity and specificity that may differ by outcome level | Youden index above the floor in each outcome level; rates declared or estimated from one validation sample per level (binomial variances propagated; one shared pair keeps its covariance) | a Youden index at or below the floor in either level (`measurement_not_identified`); a corrected share outside `(0, 1)` |
| per-unit soft exposure | the exposure coefficient of a linear outcome model on a per-unit probability that the exposure is present | posterior-mean substitution when the probabilities are calibrated given the covariates; multiple imputation when outcome-conditioned draws are supplied or generated | fewer than two imputations, a rank-deficient design |
| evidence-dependent detection | a prevalence recorded with detection probability `d(c)` that depends on a covariate (a count), optional false-positive rate | fully declared `d` (logistic, table): a point; partially declared monotone `d` with anchor bounds: sharp bounds `A / G` | mean detection net of false positives at or below the floor; contradictory anchors; bounds implying a prevalence above one |
| repeat codings as replicates | the prevalence of a binary or ordinal latent facet and each coding's response law, by a latent-class model | three or more conditionally independent codings (Kruskal's condition on levels, categories and codings); two codings of a binary facet with exchangeable codings and a declared law of the lowest level | any other count of codings; codings whose class-conditional laws are almost equal |

**Exposure misclassification.** With recorded exposure share `p_y` among units
with outcome `y`, `pi_y = (p_y + Sp_y - 1) / J_y`, `J_y = Se_y + Sp_y - 1`, and
the corrected odds ratio is `[pi_1 / (1 - pi_1)] / [pi_0 / (1 - pi_0)]`,
valid under case-control and cohort sampling. The delta-method variance has a
sampling part and a rates part; independent per-level rates add, one shared
pair adds its derivatives before squaring. Applying one pair to error that
differs by outcome misses the truth, which is why the structure is declared.

**Soft exposures.** With `X = q + eta`, `E[eta | q, Z] = 0` and
`Var(eta | q) = q (1 - q)`, least squares on `q` is consistent for the
coefficient of `X` and its HC0 sandwich contains the extra residual variance
`beta^2 q (1 - q)`; the report splits that term out as `exposure_uncertainty`
(`beta^2 mean(q (1 - q)) (M^-1)_11 / n`, at most the total). A single draw of
`X` from `q` alone is Berkson noise and attenuates the slope, so multiple
imputation draws the exposure given the outcome: a Gaussian outcome model with
the exposure as a binary regressor and `q` as its prior is fitted by EM, the
rows are bootstrapped for each imputation, the model is refitted and each
exposure drawn from its posterior; the fits are combined by Rubin's rules. The
draws are proper only if that outcome model is right. `soft_exposure_reliability`
compares the stated probabilities with repeat 0/1 codings in equal-width bins
(expected coding share `Se q + (1 - Sp)(1 - q)`, exact codings by default) and
reports gaps with standard errors and a chi-square summary; it never gates.

**Detection.** With presence probability `pi` independent of the covariate,
`P(D = 1) = pi d(c) + (1 - pi) f` and `pi = (sum D - n f) / (sum d - n f)`;
the variance adds Bernoulli sampling, the declared function's parameter
covariance and the false-positive rate's variance. A monotone envelope gives
each unit the tightest bounds the anchors allow; because `sum d` can reach any
value between the sums of the two bounds, `[A / G(hi), A / G(lo)]` is sharp
for the declaration. Standard errors of the bounds are sampling errors of each
endpoint; no interval over the identified set is claimed.

**Repeat codings.** The model is `P(c) = sum_j pi_j prod_k theta_{k j c_k}`
with conditional independence across codings given the level, an assumption
that is stated and not tested. It is fitted by EM from several starts (the
best likelihood is kept and the number of agreeing starts reported); levels
are ordered by expected coding score; the covariance is the inverse of the
outer product of per-unit scores in a softmax parametrization, carried to the
probabilities by the delta method and reported when invertible. The per-unit
posterior of a level can be passed to the soft-exposure correction as the
exposure probability.

## Exposure measurement in the analysis

The four corrections above are also measurement declarations on the exposure variable of an
effect query. `prepare` and `analyze` take `measurement=` and carry the declaration through the
whole estimate, with the uncertainty of the measurement propagated by multiple imputation:

```python
import antecedent as ant
from antecedent import exposure_measurement as em
from antecedent import measurement_exposure as mx

outcome = ant.analyze(
    data,                                    # "x" is the recorded, misclassified exposure
    graph=[("z", "x"), ("z", "y"), ("x", "y")],
    query=ant.AverageEffect(treatment="x", outcome="y"),
    measurement=em.MisclassifiedExposure(
        "x",
        mx.ExposureError.by_outcome(                   # rates may differ by outcome level
            cases=mx.ValidationCounts(90, 10, 80, 20),
            controls=mx.ValidationCounts(70, 30, 190, 10),
        ),
        covariates=("z",),
    ),
    measurement_draws=40,
)
result = outcome.value                       # a MeasuredAnalysis
result.effect, result.interval(0.95)         # pooled effect of the true exposure, Rubin interval
result.variance_components                   # within, between (exposure uncertainty), total
result.naive_effect                          # the recorded column taken at face value
result.exposure.export()                     # the draws, as a posterior artifact
```

| Declaration | Reads | Latent level | Parameter uncertainty redrawn per imputation |
|---|---|---|---|
| `MisclassifiedExposure(variable, error, ...)` | the recorded 0/1 column | present or absent | sensitivity and specificity, from validation counts (Beta posterior) or declared variances; one pair, or one per outcome level (a 0/1 outcome) |
| `SoftExposure(variable, probability, ...)` | a column of per-unit probabilities | present or absent | none beyond the imputation model; the probabilities are taken as calibrated given the covariates |
| `DetectedExposure(variable, detection_covariate, detection, ...)` | the recorded detection and the covariate that drives it | present or absent | the logistic function's parameters (with their covariance) and the false-positive rate; a partially declared monotone function identifies bounds only and is refused |
| `RepeatCodedExposure(variable, codings, ...)` | several coding columns | a binary or ordinal facet | the latent class response laws, refitted on a row bootstrap |

The declaration gives each unit a likelihood of what was recorded under every latent level. The
level is then imputed `measurement_draws` times from its posterior given that likelihood, the
outcome and the covariates: a softmax exposure model (in the covariates, or the supplied
probabilities) and an outcome model (Gaussian, or logistic for a 0/1 outcome) are fitted by EM on
a row bootstrap, so the draws carry the uncertainty of the imputation models as well as of the
measurement. Imputing from the recorded value alone would attenuate the effect; the outcome and
the covariates the analysis adjusts for are part of the imputation, and `covariates=` names them
(by default the other variables of the graph). One study is prepared per completed table with the
declared graph, estimator and inference, and `estimate` pools the per-draw effects by Rubin's
rules: the total variance is the mean within-draw variance plus `(1 + 1/m)` times the
between-draw variance, and the interval is a `t` interval with Rubin's degrees of freedom. The
interval carries the calibration status `not_measured`. The imputation outcome model must contain
the relations the analysis relies on; a Gaussian or logistic outcome model is the declared
assumption, and the draws are proper only when it is right.

`prepare` returns a `PreparedMeasuredAnalysis` (`rebind`, `reconfigure` and `estimate` work as
for a prepared study). The draws are an uncertain column: `ExposureDraws.export()` writes them as
a posterior artifact (one quantity per unit, one draw per imputation, the declaration and data
identities in its identification field), `consume_exposure_draws` reads them back, and
`ExposureDraws.completed_tables(data)` hands each draw to any stage that takes a table, such as
discovery, whose results are then combined as multiple-imputation results.

Refused: an error that differs by outcome level with a continuous outcome
(`route_not_supported`), a Youden index (or mean net detection) at or below its floor
(`measurement_not_identified`), a partially declared detection function, a declaration on a
variable other than the query's treatment, and a query that reports no scalar effect.

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
variance parts, exports with `export`, and `interval(level)` returns the Wald
interval with its calibration slot (`not_measured`). `consume_measurement_model_artifact` recomputes the stored
numbers against the retained model identity. A posterior provider is stated
with `ProviderObject` (an attestation, or a verification receipt the library
issued); a claim of native trust is refused by name. Refusals are
`MeasurementModelRefusal`; `estimate` returns an `Outcome` that carries the
refusal instead of raising it. Carrying a prepared model into another population
(`transport_measurement_model`, `verify_measurement_transport`,
`transport_pooled_differential`) is described in
[transported decisions and measurement](guides/transported-decisions-and-measurement.md).

`antecedent.measurement_exposure` offers `correct_exposure_misclassification`
(with `ExposureError.shared` or `ExposureError.by_outcome`, each pair declared
as `BinaryRates` or estimated from `ValidationCounts`), `soft_exposure_effect`
(`method="posterior_mean"`, `"multiple_imputation"` or `"supplied_draws"`),
`soft_exposure_reliability`, `detection_dependent_prevalence` (with
`LogisticDetection`, `TableDetection` or `MonotoneDetection`) and
`repeat_codings_latent_class`, all refusing with `MeasurementModelRefusal`.

## References

Carroll, Ruppert, Stefanski and Crainiceanu (2006), *Measurement Error in
Nonlinear Models*, doi:10.1201/9781420010138. Cook and Stefanski (1994),
Simulation-extrapolation estimation in parametric measurement error models,
JASA 89:1314-1328, doi:10.1080/01621459.1994.10476871. Rogan and Gladen (1978), Am J
Epidemiol 107:71-76, doi:10.1093/oxfordjournals.aje.a112510. Kruskal (1977),
Three-way arrays: rank and uniqueness of trilinear decompositions, Linear
Algebra Appl 18:95-138, doi:10.1016/0024-3795(77)90069-6. Rubin (1987),
*Multiple Imputation for Nonresponse in Surveys*, doi:10.1002/9780470316696.
