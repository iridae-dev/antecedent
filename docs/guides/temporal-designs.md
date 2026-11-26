# Estimate effects of repeated decisions and whole treatment regimes

Estimate what a decision at one period did to a later outcome. You can also estimate what a whole treatment path would have produced. The tables have one row per unit and period. You can analyse a micro-randomized trial or a sequentially randomized trial. That target is the excursion effect of one decision. You can analyse observational decision data by modelling the decision on measured history. You can analyse a static regime on a continuous outcome. You can analyse survival under always-treat versus never-treat.

Call `antecedent.temporal_designs.estimate_temporal_design`. `antecedent.analyze` takes the same designs as typed queries. In Rust, call `CausalQuery::TemporalDesign`. Where the experimenter recorded the assignment law, the recorded law identifies the target by design. Regime survival reads observational data under declared sequential exchangeability. The observational excursion does too. The observational regime contrast does too. No graph is involved. None of these calls has a cell in the graph-indexed [support matrix](../support-matrix.md). Support is stated per row of the graphless matrix (see [When a fit is supported](#when-a-fit-is-supported)).

## Typed queries

`antecedent.analyze(table, query=...)` and `antecedent.prepare(table, query=...)` take the same
designs as typed queries that name columns of the table:

```python
import antecedent
from antecedent.temporal_designs import TwoArmExcursion

result = antecedent.analyze(
    table, query=TwoArmExcursion("y", "unit", "period", assignment="a", probability="p", lag=1)
)
result.value.temporal_design  # TemporalDesignEstimate
```

These queries take `quantile_level=` where the design reads it. They take
`misclassification=OutcomeMisclassification(...)` where the design reads it. The queries are:

- `TwoArmExcursion`
- `MultiArmExcursion` (one probability column per arm)
- `FactorialExcursion`
- `InterferenceExcursion`
- `DoseExcursion`, with target `SmoothedDoseCurve`, `SmoothedDoseContrast`, or `LocalQuadraticDoseGrid`
- `RegimeSurvivalContrast`

The answer is structured. A fit its support row does not admit is
`unsupported_temporal_design_<uncertainty>`. A fit its support row admits is
`supported_temporal_design_<uncertainty>`. `load(result.export())` reads back an equal
`temporal_design`. `TemporalDesignEstimate.bootstrap_standard_error` is the whole-procedure
bootstrap standard deviation. It is set when the call resampled with `bootstrap=`.
`posterior_standard_deviation` is set for a Bayesian result. A rebound table must repeat the
units, the periods, and the recorded assignment law.

No example script in `examples/` covers these designs; the code blocks on this page are the
starting point.

### When a fit is supported

The sequentially randomized designs and regime survival each have a frequentist row and a
Bayesian row in the [graphless support matrix](../graphless-support-matrix.md). Those designs
are two-arm, multi-arm, categorical, factorial, and interference. They also include the three
dose targets and the median. A fit that meets its row's thresholds states the row. In Python,
read `TemporalDesignEstimate.supported`. In Rust, call `StudyResult::states_section_row()`.
The diagnostic is `estimate.temporal_design.graphless_support`. A row names the fewest
independent sequences and decisions it was measured at. It names the smallest assignment
probability it was measured at. A posterior or bootstrap row also needs the measured number of
draws or replicates. A row is stated for the settings that were measured. The quantile row is
the median, not another quantile level. The local-quadratic row is the level with a declared
curvature bound of zero. The survival row has a horizon of at most three periods. Each row's
coverage was measured on one named coordinate. The simultaneous band is published and is not
covered by the row. The Holm-adjusted p-values are published and are not covered by the row.
The other coordinates are published and are not covered by the row. Every other fit executes
as an **unsupported** structured answer. Both observational designs do too. Every interval and
posterior carries its calibration status in `TemporalDesignEstimate.calibration`.

## The excursion target

At each decision period `t` a unit's treatment is drawn from a law the experimenter records.
That law is the probability of the realized assignment given the unit's history. For a
continuous dose it is the density. The target pairs the decision with the same unit's outcome
`lag` periods later, `Y_{t+lag}`. It contrasts assigning one arm with another *at `t`*. Every
earlier and later assignment follows the design. This is the causal excursion effect of
micro-randomized trials (Boruvka et al. 2018, doi:10.1080/01621459.2017.1305274). Carryover is
inside the target. That is what separates it from the switchback direct effect. The switchback
call assumes no carryover. Decisions whose outcome period lies past the unit's follow-up are
left unpaired and counted. The target is identified because each assignment is drawn with a
recorded probability given the unit's history. That probability is at least `min_probability`
(default 0.05). Consistency is declared. No interference between units is declared, outside the
interference design's clusters.

| Design (`design=`) | Coordinates | Frequentist | Bayesian |
| --- | --- | --- | --- |
| `two_arm` | excursion effect, arm means | cluster-robust normal interval | Bayesian bootstrap over units |
| `multi_arm` | contrasts against the reference arm, arm means | plus a sup-t band and Holm family over the contrasts | same |
| `multi_arm` with `categorical` (and `ordered`) | reference contrasts as one Holm family, adjacent steps when ordered | same | same |
| `factorial` | both main effects (averaged over the other factor) and the interaction, cell means | plus a band and Holm family | same |
| `interference` | direct, spillover and total effects of own assignment and any treated peer of the interference cluster in the same period | clusters are the interference clusters | same |
| `dose` (`smoothed_contrast`) | kernel-smoothed contrast of two doses; the bandwidth is part of the target | inverse-density kernel scores | same |
| `dose` (`smoothed_curve`) | kernel-smoothed response on a grid | plus a sup-t band | same |
| `dose` (`local_quadratic`) | local-quadratic level or derivative of the dose response | bias-aware interval and band under a declared `curvature_bound` | posterior of the bandwidth projection; the bias bound is reported beside it |
| `two_arm` with `quantile_level` | quantile of the outcome law under the excursion, per arm, and its difference | whole-procedure cluster bootstrap (`bootstrap=` at least 99) | Bayesian bootstrap over units |
| `regime_survival` | survival at `horizon` under always-treat and never-treat, and the difference | parametric g-formula (pooled logistic hazard, Gaussian confounder transition); the subject bootstrap refits both models (`bootstrap=` at least 99) | joint-likelihood parametric g-formula |
| `observational_excursion` | the excursion effect and arm means of an observed decision with no recorded assignment law | pooled logistic assignment model on the `history` columns; the unit bootstrap refits it in every replicate (`bootstrap=` at least 99) | weighted-likelihood bootstrap: Dirichlet unit weights, the assignment model refitted per draw |
| `observational_regime` | mean end-of-follow-up outcome under an active and a reference whole treatment path, and their difference, under time-varying confounding | unit bootstrap refitting every per-period treatment model and outcome regression (`bootstrap=` at least 99) | weighted-likelihood bootstrap: Dirichlet unit weights, every model refitted per draw |

The arm and dose estimators are ratios of Horvitz-Thompson scores summed within independent
units. Serial dependence inside a unit is unrestricted. The Bayesian counterpart reweights
whole units with Dirichlet weights. That is a vanishing-concentration Dirichlet-process
posterior. It re-evaluates the same functional. It holds the recorded assignment law fixed.
The prior supplies no identifying premise.

### The temporal quantile target

`quantile_level=tau` asks for the `tau`-quantile of `Y_{t+lag}(a)` pooled over decisions. That
is the outcome law a randomly chosen decision would see if arm `a` were assigned at it and the
design were followed otherwise. It is estimated by inverting Hajek-weighted distribution
functions. It is not a quantile of unit-level effects. No randomized design identifies that
quantile.

### Dose grids and smoothing bias

A local-quadratic grid estimates `m(d)` with a smoothing bias that is not part of the target.
Declare a bound `M` on `|m'''|` with `curvature_bound`. The exact worst-case bias of the
linear smoother at each dose is `M/6 * sum |l_p| |D_p - d|^3`. That bias enters the interval.
The critical value is the folded-normal critical value of Armstrong and Kolesar,
doi:10.3982/QE1199. The same bias is added to the band half-width. Without a bound the
frequentist grid is refused (`required_option_missing`). The next attempt is the smoothed
curve, whose bandwidth is part of the estimand. The same construction is available for the
static randomized-dose grid. In Rust, call
`antecedent_estimate::dose_grid_functional::estimate_dose_grid_bias_aware`. The saved artifact
stores that result. In Python, call `antecedent.dose_grid.dose_functional(..., curvature_bound=M)`.

### Misclassified binary outcomes

`misclassification=(true_positive, false_negative, true_negative, false_positive)` corrects a
binary outcome recorded with non-differential error. It uses an independent validation sample.
Contrasts divide by the Youden index `se + sp - 1`. Arm means are shifted and divided. The
frequentist covariance adds the validation counts' delta-method contribution. The band is
dropped. The posterior draws `se` and `sp` from Beta posteriors per draw. A Youden index
below 0.1 is refused.

## The observational excursion

`ObservationalExcursion` keeps the excursion target of the sequentially randomized designs. It
changes only what identifies it. No assignment probability is recorded. The decision is modelled
on the measured decision-time `history` columns by a pooled logistic regression. Those columns
include lagged outcomes and earlier treatment. The contrast is the same Horvitz-Thompson ratio
with the fitted probability in place of the recorded one. Identification is sequential
exchangeability given `history`. It is positivity of both arms given `history`. A fitted
probability under `min_probability` is refused as `positivity_violated`. Consistency is
required. No interference between units is required. The history is declared. It is not
checked. A single-decision excursion with a known law does not substitute for it. This call is
the one that corrects confounding of the decision. The nuisance fit is inside every interval.
The interval carries the cost of estimating the assignment law. Read its calibration status
before relying on coverage.

## Whole static regimes on a continuous outcome

`ObservationalRegimeContrast` answers what the mean end-of-follow-up outcome would be had every
subject followed this whole treatment path rather than that one, when past treatment shapes later
confounders. Subjects are observed in the same periods. In each period `k` the `confounders`
`L_k` are measured, then the 0/1 `treatment` `A_k` is taken. `outcome` is read once, at each
subject's last period. `active` and `reference` are paths, one action per period. Any static
regime is expressible: always, never, switch on, switch off, and so on. A pooled regression of
the outcome on the final history does not identify the contrast. Earlier treatment moves the
later confounders. The estimators below do identify the contrast.

Identification requires sequential exchangeability: `Y^a` independent of `A_k` given
`L_0..L_k, A_0..A_{k-1}` in every period. That rules out an unmeasured time-varying confounder
of a later treatment and the outcome. It requires sequential positivity of the path actions.
Every fitted action probability is held to `min_probability`. A violation is refused as
`positivity_violated`. It requires consistency. It requires no interference between subjects.
It requires complete follow-up. These are declared. They are not checked. Loss to follow-up is
not modelled. Use regime survival for censored event times.

`method` selects the estimator. `g_formula` runs iterated conditional expectations. From the
last period back, the pseudo-outcome is regressed on the history, the period's treatment, and
its interaction with the period's confounders. The prediction with the period's action set to
the path becomes the next pseudo-outcome. `inverse_probability_weighting` is the saturated
marginal structural model of each path. Subjects who followed the path are weighted by the
inverse of the product of their fitted action probabilities. There is one logistic model per
period on the history. The mean is the weight-normalized outcome. `doubly_robust` is the
default. It is the recursive augmented estimator. It corrects each period's regression
prediction by the weighted residual of the next. It is consistent when either the outcome
regressions or the treatment models are right. Every nuisance model is inside the interval.
The frequentist interval refits all of them in each unit-bootstrap replicate. The Bayesian
posterior refits them under Dirichlet unit weights in each draw. The treatment models use a
Gaussian coefficient prior. The prior supplies no identifying premise.

## Regime survival

Rows per subject run `0, 1, 2, ...` while at risk. Confounders are recorded at the start of the
period, then treatment, then censoring during the period (`censored = 1`, event unobserved) or
the event. The target is `S_1(K) - S_0(K)`. Sequential exchangeability of treatment and
censoring given the measured history is declared. Positivity of both regimes is declared. The
floor is `min_probability`. Consistency is declared. These are not checked. The frequentist
estimate is the plug-in parametric g-formula. A pooled logistic hazard on the uncensored
person-periods is fitted by maximum likelihood. A Gaussian linear confounder transition is
fitted by maximum likelihood. Each subject's baseline confounders are carried forward under
each regime on fixed random paths shared across regimes. The fitted hazards are multiplied
out. The interval refits both models in every subject-bootstrap replicate. It takes the
percentile interval. The model fits are inside it. The treatment model is fitted only to
hold the regime probabilities to the floor. The Bayesian posterior factorizes the trajectory
likelihood into a conjugate Gaussian confounder transition and Laplace pooled-logistic laws
for treatment, censoring, and the hazard. It evaluates the parametric g-formula per draw with
common random numbers across regimes. The treatment and censoring posteriors report the share
of draws below the positivity floor.

## Check whether a mechanism changed, and roll fitted dynamics forward

Two standalone queries read the fitted dynamics of a series with lagged edges, without a typed design. Neither has a row in the support matrix. Read them as model-conditional diagnostics and projections. They live in `antecedent.temporal_model`. They take column names, the columns, and the lagged edges `(cause, lag, effect, 0)`.

`detect_mechanism_change` asks whether the mechanism of a target changed between two row ranges of one series. The simulated series below shifts the intercept of `y` by 5.0 halfway through, so the answer is known.

```python
import numpy as np
from antecedent.temporal_model import detect_mechanism_change, rollout_dynamics

EDGES = [("x", 1, "y", 0)]
n = 300
rng = np.random.default_rng(4)
x = np.zeros(n)
y = np.zeros(n)
for t in range(1, n):
    x[t] = 0.5 * x[t - 1] + rng.normal()
    y[t] = (1.0 if t < n // 2 else 6.0) + 2.0 * x[t - 1] + rng.normal()
names, cols = ["x", "y"], [x, y]

(found,) = detect_mechanism_change(
    names, cols, EDGES, ["y"], (0, 150), (150, 300), test="likelihood_ratio"
)
found.changed, found.p_value, found.method
```

The null law is a circular-block permutation of the residuals. The series' serial dependence stays inside a block. `found.method` names it. `test=` selects one of these:

- `"likelihood_ratio"`
- `"mean_difference"`
- `"kernel_two_sample"`
- `"classifier_two_sample"`
- `"change_point"`

For a probability instead of a p-value, pass `posterior_draws=` together with a `change_prior` you declare. The prior is never assumed. Omitting it is refused. The result then has `posterior_probability`. It has no `p_value`.

`rollout_dynamics` rolls the fitted dynamics forward from the last rows of the series under scheduled interventions and returns the per-step mean of every variable:

```python
hold = [{"variable": "x", "value": 1.0, "from": 0, "until": 3}]
rolled = rollout_dynamics(names, cols, EDGES, n_steps=4, interventions=hold)
rolled.mean(2, "y")   # close to 3.0: y = 1 + 2 * x at the previous step
uncertain = rollout_dynamics(
    names, cols, EDGES, n_steps=4, interventions=hold, posterior_draws=10
)
uncertain.uncertainty.names, uncertain.uncertainty.sd
```

`posterior_draws` adds `rolled.uncertainty`. That option is a block-weight posterior. `bootstrap_replicates` adds the same field. That option is a circular-block bootstrap. The field is the spread over refits of the dynamics for the step means. In the `path.step[..]` quantities it is one simulated path per refit. A path's summaries are the predictive law of a path. They add the fitted innovation noise to the refit uncertainty. They are much wider than the step-mean summaries. The rollout is the fitted model's projection under the interventions you schedule. It is not an identified effect of those interventions.

## Follow only the treatment links you chose

A treatment can change the first step, the second step, and the outcome directly. Give Antecedent the time-ordered graph. Name the treatment links you want to switch on. The other links keep their reference settings. The call fits the declared equations on source rows. It evaluates the selected contrast on target baseline rows.

In the synthetic example, treatment adds one unit to the first step. The second step doubles that unit. Activating only the first link therefore changes the outcome by two. Activating the remaining treatment links changes it by seven. The example checks both answers. It tests that an artifact for one link selection cannot be replayed as the other.

Run `cargo run -p antecedent --example transport_selected_path`.

[Read the complete Rust example](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/rust/transport_selected_path.rs).

## What is refused, and the remedy

Each code is described with its remedies in [refusal codes](../reference/refusal-codes.md).

| Refused request | Reason code | Remedy |
| --- | --- | --- |
| A realized assignment probability below `min_probability` | `positivity_violated` | A design whose arms keep the floor at every decision, or a lower floor |
| An arm, cell or exposure no paired decision realized | `arm_not_populated` | More units or periods, or probabilities away from the boundary |
| `bootstrap=` on a design whose interval is analytic, or under Bayesian inference | `option_not_applicable` | Omit it; only the quantile functional, frequentist regime survival, the frequentist observational excursion and the frequentist observational regime contrast resample |
| The quantile functional, regime survival, the observational excursion or the observational regime contrast without `bootstrap=` of at least 99 | `required_option_missing` | Set `bootstrap=99` or more |
| A regime contrast with identical paths, paths of different length or of a length other than the number of periods, a nonzero lag, a repeated column, or subjects observed in different periods | `invalid_argument` | Name two different paths with one action per period, keep `lag=0`, and give every subject the same periods |
| A regime whose actions the fitted history does not support (a fitted probability under `min_probability`, or no subject following a path) | `positivity_violated` / `arm_not_populated` | A path the data support, fewer or better-scaled confounders, or a lower floor |
| A local-quadratic grid without `curvature_bound` | `required_option_missing` | Declare the bound, or use `smoothed_curve` |
| A dose outside the observed range, or with local effective sample size below 10 | `not_supported` | A wider bandwidth or a dose inside the supported range |
| An interference decision with no other cluster member that period | `invalid_argument` | Record every member of the cluster each period |
| A Youden index below 0.1 | `route_not_supported` | A more accurate measurement or a larger validation sample |
| An observational excursion with no history column, a treatment not coded 0/1, or a history column equal to the outcome or treatment | `invalid_argument` | Name the decision-time covariates, code the treatment 0/1, and keep the columns distinct |
| A static switchback | n/a | Switchback assignment is a sequence of periods; there is no static switchback. Use the graphless switchback route or this design |

## Related

- [Evaluate a treatment regime over several periods](designs/sequential-regimes.md) when you supply your own conditional predictions and probabilities.
- [Estimate a treatment effect on time to an event](designs/survival-outcomes.md) for static randomized survival.
- [Analyze a randomized or factorial experiment](designs/randomized-experiments.md) for single-decision designs.
- [Plan power and sample size](power-and-sample-size.md) for switchback and sequential plans.
