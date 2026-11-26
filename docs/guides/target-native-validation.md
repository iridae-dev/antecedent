# Refute the quantity a result reports and bound an untestable assumption

Use this page when you have a prepared study or an estimate. You can stress the quantity it
reports by refitting it under placebo, resampling and negative-control perturbations. You can
also state how strong a violation of an untestable assumption would have to be before that
quantity changes. For the refits, prepare the study with `refute="none"`. For the sensitivity
ranges, you need an estimate with its interval, coefficient statistics, matched differences
or posterior draws.

A refutation speaks to one quantity. The `Refute` suites (`cheap`, `placebo`, `full`) perturb
the data and re-estimate a scalar average effect. They say nothing about a conditional
effect, a response derivative, a temporal effect vector, a mediation decomposition, the mean
outcome under several interventions, or a joint contrast. The refuters here test the quantity
a result reports. The sensitivity analyses state how far an untestable assumption can be
relaxed before that quantity changes.

Every number here is a diagnostic. Pass thresholds are conventions with no measured error
rate. None of these checks supports an interval.

Runnable example: [`examples/python/refute_and_bound.py`](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/python/refute_and_bound.py)
refutes an adjusted estimate and then bounds the hidden confounding the refuters could not
detect.

## The target a refutation speaks to

A **refutation target** names the quantity. It names the family: scalar average effect,
conditional effect, response derivative, temporal effect, mediation, world-valued, joint
contrast, or attribution. It names the quadrant: static or temporal data, frequentist or
Bayesian inference. It names one label per coordinate. It names the identity of the analysis
that defines the quantity. Two targets are the same only when these identities agree. A
refutation computed under another analysis, for another family, or on other coordinates is
another target.

A scalar average-effect refutation does not certify any other target. A claim on a
derivative, a conditional effect, a mediation decomposition or a vector of world values that
carries one is refused. The next attempt is to run the target-native suite on the claim's
own target. Reports from the target-native suite are bound into a certificate. The
certificate's identity covers the target and every verdict. A report on another target is
refused.

## The catalogue

For every family and quadrant, `validation.target_refutation_plan(family, temporal=...,
bayesian=..., suite=...)` lists the refuters that apply:

| Refuter | What it does | Expected |
| --- | --- | --- |
| `target.placebo_treatment` | permute the treatment (a circular shift on a series) | the whole vector at zero (world values: no contrast between worlds) |
| `target.placebo_outcome` | permute the outcome (circular shift on a series) | zero |
| `target.random_common_cause` | add an independent common cause to the adjustment | unchanged |
| `target.subset_stability` | refit on random subsets (contiguous windows on a series) | unchanged |
| `target.bootstrap_stability` | refit the whole procedure on resamples | unchanged |
| `target.negative_control_exposure` / `_outcome` | swap in a declared negative control | zero |
| `target.pre_trend`, `target.anticipation` | temporal quadrants: treatment leads, pre-onset outcomes | zero |
| `target.sequential_ignorability` | mediation: range of mediator-outcome error correlation | break-even correlation above a bar |
| `target.finite_difference_consistency`, `target.local_linearity` | derivatives: the reported derivative against the fitted response | consistent; locally linear at the step |
| `target.prior_sensitivity`, `target.posterior_predictive` | Bayesian quadrants: alternative priors; replicated data | small posterior shift; observed target inside the predictive law |

Replicate refuters judge the **whole vector** with one joint discrepancy. That is the
squared Mahalanobis distance of the expected vector from the replicate mean, in units of
the replicate covariance. The unit is one replicate's spread. It is not the Monte Carlo
error of the mean. On one coordinate it is the squared standardised shift. On several it
keeps the correlation. It does not run marginal checks. The check passes when the
discrepancy stays within the tolerance set by `alpha`. Its tail probability under a
chi-square law on the target dimension is at least `alpha`. This is a fixed yardstick on a
stability or placebo diagnostic. It is **not a hypothesis test with a nominal rejection
rate**. For bootstrap and subset stability the replicate mean minus the original estimates
the resampling bias. With an unbiased procedure the discrepancy is about `k / replicates`
rather than chi-square with `k` degrees of freedom. A clean case rejects far below `alpha`.
Each replicate report also carries `standardised_bias`. That is the replicate mean minus
the expected vector, per coordinate, in replicate standard deviations. It also carries
`standardised_bias_norm`. That is the Mahalanobis length of the same shift, the square root
of the discrepancy, the effect size behind the verdict. Subset stability is inherently
weakly powered against one influential observation. Under circular windows a mean-type
statistic has a replicate mean equal to the original whatever the outlier. The replicate
spread that standardises the shift grows with it. Analyses that read neither row weights
nor an analysis mask refit lagged resamples on contiguous sub-windows, a third of the
series. They never use a block resample whose joins would pair values observed apart.
Attribution and weight-reading analyses keep the block resample. Join-crossing design rows
are hidden or carried as row weights. Its rejection rate on clean data and its power
against planted violations are measured separately. Nothing here is a calibration claim.
Each refit's random stream is keyed by refuter and replicate. A report does not depend on
the thread count. The replicate buffer is leased from the admission pool before it is
allocated.

## Refuting a prepared study's own target

```python
import antecedent as ant
from antecedent import validation

study = ant.prepare(data, graph=graph, query=query, refute="none")
checks = validation.TargetRefutationSpec(suite="full", replicates=30)
outcome = validation.refute_target(study, checks=checks)
report = outcome.value     # typed TargetRefutation; inspect outcome.refusal first
report.target.family      # e.g. "response_derivative"
report.reports            # one TargetCheck per refuter that ran
report.refused            # refuters that could not run, with reasons and next attempts
report.passed             # True, False, or None when no informative report ran

# Reuse the exact original native result rather than execute it again.
original = ant.estimate(study)
validated = validation.refute_target(original, checks=checks)
# Different observations are an explicit immutable branch.
updated = ant.rebind(study, data=new_data)
updated_validation = validation.refute_target(updated, checks=checks)
```

Each `TargetCheck` names its `refuter`, the `original` and `refuted` target vectors, the
`statistic` with its `value`, `df` and `p_value`, whether it `passed`, and whether it was
`informative`. Read `passed` with the caveats under [the catalogue](#the-catalogue). It is a
fixed yardstick. It is not a test with a nominal rejection rate.

Every refit runs the analysis again in full. Most refuters perturb the data and run the
study's own prepared analysis. Where a refuter changes the scientific problem, the study is
built again with that one change and run. Prepare with `refute="none"`. A scalar suite
configured on the study would run on every refit.

- **Random common cause.** A simulated cause is appended to the data and declared in the graph as a parent of the treatment and the outcome. On a temporal graph, that is at every lag the treatment appears at. The augmented study is prepared and run, so the analysis adjusts for it. With `common_cause_strength > 0` the cause also enters the outcome, and a non-binary treatment. The refuter checks that adjusting for a declared confounder recovers the target. This runs on a DAG or a temporal DAG. A graph posterior has no single structure to declare the cause in. A class that has not resolved to one graph has no single structure either.
- **Negative controls.** Declared ones are used as given. Otherwise the graph supplies them. An exposure candidate sits outside the adjustment set, is not a descendant of the treatment, and is d-separated from the outcome given the adjustment set. On a temporal graph, none of its nodes reaches the outcome. An outcome candidate is one the treatment cannot reach and is d-separated from. On series and panels the default negative-control outcome is the lagged placebo, the outcome moved back past the graph's longest lag. A refusal is recorded only when the graph supplies no candidate. Spread comes from the design's own resample: rows, circular blocks of a series, or whole units of a panel.
- **Series bootstrap.** A circular-block bootstrap with the dependence-aware block length, carried as row multiplicities. Each design row keeps its own lag window. No lag is read across a block join. A series that carries sampling weights keeps them. The multiplicities multiply each row's own weight. An analysis that reads no row weights is given the gathered circular-block series instead. Every row carries the weight, the missing values and the analysis mask of the row it was gathered from. A row whose lag window crosses a block join gets weight zero. Independent environments (`PreparedTargetData::MultiEnv`) are perturbed one environment at a time by the same rules and re-run together. A panel resamples whole units, each with its own weights.
- **Placebo timing and pre-period outcome on one series.** `target.pre_trend` puts the treatment `pre_trend_lead` periods later in the treatment's role. Its lags in the design are the actual treatment path. The coefficient is the effect of a future treatment on the present outcome given the past. `target.anticipation` puts the outcome from before every treatment period the target reads in the outcome's role. Both assume the treatment does not respond to the outcome within the shift. Rows without a shifted value get weight zero.
- **Derivative diagnostics.** The study's mean-curve query on the same data and graph is run at the five points the checks read. When the derivative's estimator also serves a curve, that estimator is used. Otherwise the analysis chooses the curve estimator. The reported derivative is checked against that curve. First derivatives are on the identity scale. An average derivative is read at seven nodes of its weighting law. A directional derivative or a Jacobian entry along a direction is read off the study's own intervention-response query at the shifted points.
- **Bayesian prior sensitivity.** The study is built under each alternative isotropic prior scale (`prior_scale_factors` times its own). The posterior of the whole target is read back. An explicit coefficient prior set has no single scale. Every Gaussian coefficient standard deviation of the set is multiplied by the factor around its own means. Variances are multiplied by the factor squared. The study is built under each scaled set. A prior hydrated from a posterior artifact, or composed from external sources, exists only once the design is bound. The same factor is applied where the analysis resolves it. That is `BayesianConfig::prior_variance_multiplier`, the squared factor on every Gaussian coefficient variance of the prior in force.
- **Posterior-predictive check.** Under the reference linear-Gaussian outcome model of the treatment and adjustment set, each posterior draw's noiseless outcome gives the target of the draw. Its noisy replicate gives a replicated target. Both go through the study's analysis. Temporal data uses the lagged reference model and preserves its series or unit structure. Unsupported reference models remain individual refuter refusals.

`validation.refute_target` uses the observations the preparation retained: independent
rows, one series, a panel, independent environments, or events already aligned to their
declared grid. It never flattens trajectories. It never converts a multidimensional target
into an average effect. Sampling weights, active-row masks and the declared dependence
structure follow each perturbation. An analysis that cannot preserve them returns an
individual refusal.

`TargetRefutationSpec` contains scientific settings. `Execution` contains the seed,
threads, resource limits, cache policy, cancellation and observers. The operation uses
the study's execution declaration unless one is supplied:

```python
checks = validation.TargetRefutationSpec(
    suite="full",
    replicates=30,
    pre_trend_lead=2,
    gathered_resamples=True,
    prior_scale_factors=(0.25, 4.0),
    prior_shift_tolerance=0.5,
    derivative_step=None,
    derivative_tolerance=0.1,
    linearity_tolerance=0.25,
)
outcome = validation.refute_target(
    original, checks=checks, execution=ant.Execution(seed=19, threads=2)
)
```

The other defaults are `alpha=0.05`, `subset_fraction=0.8`,
`mediation_rho_bar=0.1` and `common_cause_strength=0.0`. Negative-control
exposure, outcome and mediator names default to `None`. The graph supplies
compatible controls where possible. The default pre-trend lead is one period.
Gathered resampling is off. A class posterior's common-cause perturbation
requires an explicit `within_class="uniform_over_members"` assumption. Its
mass allocation is never silently chosen.

Successful outcomes contain a typed `TargetRefutation`. Its answer is partial
when some refuters refuse. Every reason and next attempt is retained. If no
refuter can run, the outcome is a scientific refusal with the individual causes
and next actions. Failed diagnostic checks remain reports. They are not execution
exceptions. `passed` judges the informative reports that ran. A `True` value
does not remove partial standing. It does not certify refused checks. It is `None` when
there are no informative reports. Bounds remain sets. A distribution must be queried
through an explicitly declared compatible response. It does not silently become a mean.

`original_executions` records original target executions performed by this
operation. That is one for a fresh preparation, and zero for the retained original result.
It does not count fits inside refuters. The analysis identity, the target identity
and the certificate remain available on the report. Inspection and notebook
representation read bounded retained metadata. They execute nothing.

An independently loaded artifact has no live study or observations. It cannot
be refitted (`route_not_supported`, `target_refutation.live_result_required`). Supply a
preparation or retained live outcome. An outcome that is not the retained result of its own
study is refused (`target_refutation.producer_mismatch`). A preparation with no
target-native refit is refused (`target_refutation.native_preparation_required`). When no
refuter applies the refusal is `target_refutation.no_applicable_refuters`. Each refuter that
could not run carries the code `refutation_not_applicable`. See
[refusal codes and remedies](../reference/refusal-codes.md#structured-refusals-and-their-remedies)
for the fields a refusal exposes. Resource failure, cancellation and callback failure
remain typed exceptions. Diagnostic operating characteristics remain not_measured.

## Sensitivity ranges

```python
from antecedent import validation

result = validation.evalue(1.8, lower=1.2, upper=2.7)   # a risk ratio and its interval
result.summary                    # the method's own numbers
band = result.range               # AssumptionRange
band.assumption, band.parameter   # what is relaxed, and the parameter that relaxes it
band.tested                       # RangePoint(parameter, target, test) per probed value
band.break_even                   # value at which the conclusion is explained away
band.break_even_significance      # value at which it loses significance, when reached
```

Each analysis returns a `SensitivityResult` whose `range` is an `AssumptionRange`. The
range names the assumption it relaxes, the range of the sensitivity parameter it probed,
the target or its test at each value, and the break-even value at which the conclusion is
explained away or loses significance. The range is an assumption range. It is not a
confidence interval. It states no sampling coverage. Its `calibration` field records the
calibration status. The target identity is bound into the range's identity. A range
computed for one target cannot be cited by a claim on another.

| Function | Assumption relaxed | Parameter | Break-even |
| --- | --- | --- | --- |
| `validation.evalue` | no unmeasured confounding, on the risk-ratio scale (VanderWeele and Ding 2017) | confounder risk ratio with treatment and outcome | the E-value of the estimate and of the confidence limit nearest the null |
| `validation.omitted_variable_bias`, `omitted_variable_bias_from_data` | no unmeasured confounding in a linear adjustment (Cinelli and Hazlett 2020) | partial R² with treatment and outcome | the robustness values `RV_q` and `RV_{q,α}`; benchmarks bound a confounder `k` times as strong as an observed covariate |
| `validation.rosenbaum_bounds`, `matched_sets_sensitivity` | random assignment within matched sets (Rosenbaum 2002) | hidden bias `Γ` | the largest `Γ` at which the upper-bound p-value stays at most `α` |
| `validation.mediation_sensitivity` | sequential ignorability of the mediator (Imai, Keele and Yamamoto 2010) | mediator-outcome error correlation `ρ` | the residual correlation, at which the indirect effect vanishes |

E-values convert odds ratios, hazard ratios and standardized mean differences
to the risk-ratio scale by the published approximations. An odds or hazard
ratio is itself the risk ratio for a rare outcome. A common outcome uses the
square root of the odds ratio or `(1 − 0.5^√HR) / (1 − 0.5^√(1/HR))`. A
standardized mean difference `d` with standard error `s` uses `exp(0.91·d)`
with limits `exp(0.91·d ± 1.78·s)`.

The signed-rank bound is computed exactly up to 300 non-zero pairs. Above that it uses
the normal approximation with the tie-corrected variance. Matched sets with
several controls use the separable approximation of Gastwirth, Krieger and
Rosenbaum (2000). Call `validation.amplify(Γ, Λ)`. You get the outcome effect `Δ` an
unobserved covariate with treatment effect `Λ` needs to produce `Γ`
(Rosenbaum and Silber 2009).

### Vector targets and posteriors

A claim about a conditional effect (one coordinate per stratum), a temporal
effect (one per horizon) or a joint contrast is explained away as soon as any
coordinate is. Its range is read at the **weakest coordinate**. The break-even
value is the smallest across coordinates. Each tested value reports the
weakest coordinate's adjusted effect.

| Function | Reads | Weakest-coordinate rule |
| --- | --- | --- |
| `validation.evalue_vector` | one estimate and interval per label | smallest E-value; tested target is the smallest adjusted risk ratio oriented away from the null |
| `validation.omitted_variable_bias_vector` | one coefficient, standard error and degrees of freedom per label | smallest `RV_q` and `RV_{q,α}`; tested target is the smallest share of an estimate left |
| `validation.rosenbaum_bounds_vector` | matched-pair differences per label | smallest `Γ*` (absent when a coordinate does not reject at `Γ = 1`); tested value is the largest upper-bound p-value |
| `validation.evalue_posterior` | posterior draws per label | each draw is bias-adjusted at every strength; `test` is the joint posterior probability that every coordinate keeps its reported side, `break_even` the smallest E-value of a posterior median |
| `validation.omitted_variable_bias_posterior` | coefficient draws and the per-draw bias scale `sd(Y|D,X)/sd(D|X)` | each draw moves toward the null by `scale * sqrt(R²yz R²dz / (1 - R²dz))`; the break-even is the equal partial R² at which the posterior median (and the credible limit) reaches zero |

The confounding structure stays an assumption of the analysis. The posterior
variants carry uncertainty about the effect through it. They do not make the
confounding assumption testable. A range is bound to its target. A scalar
range cannot be cited for a vector claim.

### Mediation components and marginal structural model parameters

| Function | Reads | Assumption relaxed |
| --- | --- | --- |
| `validation.mediation_sensitivity_vector` | one mediator and several outcomes (or horizons); `component="indirect"` or `"direct"` | sequential ignorability of the mediator: the error correlation `ρ` of Imai, Keele and Yamamoto (2010). The total effect does not depend on `ρ`, so the direct effect is the total minus the indirect effect at `ρ`; the indirect effect vanishes at the residual correlation `ρ̃` and the direct effect at `w / sqrt(1 + w²)`, `w = (ρ̃ - β₁/A) / sqrt(1 - ρ̃²)`, `A = β₂σ₁/σ₂`. The break-even is the correlation of smallest magnitude at which some outcome's component reaches zero |
| `validation.mediation_sensitivity_posterior` | coordinate-major draws of the mediator coefficient, the two residual standard deviations, their correlation and the total effect | the same, draw by draw; `test` is the joint posterior probability that every outcome keeps its side, `break_even` the correlation at which the weakest posterior median reaches zero and `break_even_significance` the same for the credible limit |
| `validation.msm_parameter_sensitivity` | an outcome, the exposure terms of a marginal structural model and its subject weights | sequential exchangeability (no unmeasured time-varying confounding): the model is fitted by weighted least squares with the weights held fixed, each exposure parameter has a robust (HC1) standard error and the effective degrees of freedom `(Σw)²/Σw² - p`, and the partial-R² range is read at the weakest parameter |
| `model="mediation"` or `model="marginal_structural"` on `evalue_vector`, `evalue_posterior`, `omitted_variable_bias_vector`, `omitted_variable_bias_posterior` | the E-value or partial-R² analysis of a mediation component or an MSM parameter | no unmeasured mediator-outcome confounding (treatment unconfounded), or sequential exchangeability; the range records that assumption in its identity |
| `validation.mediation_path_sensitivity` | one mediator and several outcomes; `path="treatment_mediator"`, `"mediator_outcome"`, `"treatment_outcome"` (the direct effect given the mediator) or `"indirect"` | no unmeasured confounding of that path (each bound to its own assumption); for `"indirect"` both paths are confounded at the tested strength, each coefficient moving toward zero by its own worst-case bias, the share left is the product of the two paths' shares and the break-even is the equal partial R² at which that product falls to `1 - q` (at `q = 1`, the smaller of the two robustness values) |
| `validation.msm_parameter_sensitivity_refit` | as `msm_parameter_sensitivity`, with the stabilized-weight models of each time point | as above, with every weight model refitted in each unit-bootstrap replicate, so the standard error carries the cost of estimating the weights |

Each single path is the Cinelli and Hazlett analysis of that regression
coefficient. `T → M` is in the mediator regression. `M → Y` and `T → Y` are in the
outcome regression. The weight-refit standard error is the spread of the
bootstrap replicates. The robust error that holds estimated weights fixed is
reported beside it (`fixed_weight_standard_error`). It is conservative for an
estimated denominator model. The plain `msm_parameter_sensitivity` is a
conservative analysis. The refit is the one that matches repeated sampling.

The marginal sensitivity model (`msm_sensitivity`) and the decision
sensitivity grid are the analyses for bounded inverse-probability weights and
for decisions. See [Probe sensitivity](../sensitivity-and-robustness.md).

## Related

- [Probe sensitivity to assumptions you cannot test](../sensitivity-and-robustness.md): decision-level assumption ranges, switch points and structural verdicts.
- [Preflight diagnostics](preflight-diagnostics.md) and [tier diagnostics](tier-diagnostics.md): checks before and after estimation.
