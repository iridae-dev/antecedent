# Sequentially randomized designs and regime survival over time

`CausalQuery::TemporalDesign` (Rust) and `antecedent.temporal_designs.estimate_temporal_design`
(Python) read one table row per unit and period. The recorded assignment law identifies the
target by design; regime survival, the observational excursion and the observational regime
contrast read observational data under declared sequential exchangeability. Every route here executes and is **unlicensed**: no coverage record exists,
every interval and posterior carries `calibration = "unmeasured"`, and no support-matrix cell
is implied.

## Typed queries

`antecedent.analyze(table, query=...)` and `antecedent.prepare(table, query=...)` take the same
designs as typed queries that name columns of the table:

```python
from antecedent.temporal_designs import TwoArmExcursion

result = antecedent.analyze(
    table, query=TwoArmExcursion("y", "unit", "period", assignment="a", probability="p", lag=1)
)
result.temporal_design  # TemporalDesignEstimate
```

`TwoArmExcursion`, `MultiArmExcursion` (one probability column per arm), `FactorialExcursion`,
`InterferenceExcursion`, `DoseExcursion` (target `SmoothedDoseCurve`, `SmoothedDoseContrast` or
`LocalQuadraticDoseGrid`) and `RegimeSurvivalContrast` take `quantile_level=` and
`misclassification=OutcomeMisclassification(...)` where the design reads them. The answer is
structured (`unlicensed_temporal_design_<uncertainty>`), and `load(result.export())`
reads back an equal `temporal_design`. `TemporalDesignEstimate.bootstrap_standard_error` is the
whole-procedure bootstrap standard deviation (frequentist regime survival and the quantile
functional); `posterior_standard_deviation` is set for a Bayesian result. A refreshed table must
repeat the units, periods and recorded assignment law.

## The excursion target

At each decision period `t` a unit's treatment is drawn from a law the experimenter records:
the probability (or, for a continuous dose, the density) of the realized assignment given the
unit's history. The target pairs the decision with the same unit's outcome `lag` periods later,
`Y_{t+lag}`, and contrasts assigning one arm with another *at `t`*, every earlier and later
assignment following the design. This is the causal excursion effect of micro-randomized
trials (Boruvka et al. 2018, doi:10.1080/01621459.2017.1305274). Carryover is inside the
target, which is what separates it from the switchback direct effect: the switchback route
assumes no carryover. Decisions whose outcome period lies past the unit's follow-up are left
unpaired and counted.

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
| `regime_survival` | survival at `horizon` under always-treat and never-treat, and the difference | inverse-probability weighted hazards; the subject bootstrap refits the treatment and censoring models (`bootstrap=` at least 99) | joint-likelihood parametric g-formula |
| `observational_excursion` | the excursion effect and arm means of an observed decision with no recorded assignment law | pooled logistic assignment model on the `history` columns; the unit bootstrap refits it in every replicate (`bootstrap=` at least 99) | weighted-likelihood bootstrap: Dirichlet unit weights, the assignment model refitted per draw |
| `observational_regime` | mean end-of-follow-up outcome under an active and a reference whole treatment path, and their difference, under time-varying confounding | unit bootstrap refitting every per-period treatment model and outcome regression (`bootstrap=` at least 99) | weighted-likelihood bootstrap: Dirichlet unit weights, every model refitted per draw |

The arm and dose estimators are ratios of Horvitz-Thompson scores summed within independent
units, so serial dependence inside a unit is unrestricted. The Bayesian counterpart reweights
whole units with Dirichlet weights (a vanishing-concentration Dirichlet-process posterior)
and re-evaluates the same functional, holding the recorded assignment law fixed: the prior
supplies no identifying premise.

### The temporal quantile target

`quantile_level=tau` asks for the `tau`-quantile of `Y_{t+lag}(a)` pooled over decisions: the
outcome law a randomly chosen decision would see if arm `a` were assigned at it and the design
were followed otherwise. It is estimated by inverting Hajek-weighted distribution functions. It
is not a quantile of unit-level effects, which no randomized design identifies.

### Dose grids and smoothing bias

A local-quadratic grid estimates `m(d)` with a smoothing bias that is not part of the target.
Declare a bound `M` on `|m'''|` with `curvature_bound`: the exact worst-case bias of the
linear smoother at each dose, `M/6 * sum |l_p| |D_p - d|^3`, enters the interval (the
folded-normal critical value of Armstrong and Kolesar, doi:10.3982/QE1199) and is added to the
band half-width. Without a bound the frequentist grid is refused (`required_option_missing`)
and the smoothed curve, whose bandwidth is part of the estimand, is the next attempt. The same
construction is available for the static randomized-dose grid as
`antecedent_estimate::dose_grid_functional::estimate_dose_grid_bias_aware`, in its sealed artifact
and in Python as `antecedent.dose_grid.dose_functional(..., curvature_bound=M)`.

### Misclassified binary outcomes

`misclassification=(true_positive, false_negative, true_negative, false_positive)` corrects a
binary outcome recorded with non-differential error, using an independent validation sample.
Contrasts divide by the Youden index `se + sp - 1`; arm means are shifted and divided. The
frequentist covariance adds the validation counts' delta-method contribution (the band is
dropped); the posterior draws `se` and `sp` from Beta posteriors per draw. A Youden index
below 0.1 is refused.

## The observational excursion

`ObservationalExcursion` keeps the excursion target of the sequentially randomized designs and
changes only what identifies it. No assignment probability is recorded; the decision is modelled
on the measured decision-time `history` columns (including lagged outcomes and earlier
treatment) by a pooled logistic regression, and the contrast is the same Horvitz-Thompson ratio
with the fitted probability in place of the recorded one. Identification is sequential
exchangeability given `history`, positivity of both arms given it (a fitted probability under
`min_probability` is refused as `positivity_violated`), consistency and no interference between
units. The history is declared, not checked, and a single-decision excursion with a known law
does not substitute for it: this route is the one that corrects confounding of the decision.
The nuisance fit is inside every interval, so the interval carries the cost of estimating the
assignment law; coverage has not been measured.

## Whole static regimes on a continuous outcome

`ObservationalRegimeContrast` answers "what would the mean end-of-follow-up outcome be had every
subject followed this whole treatment path rather than that one" when past treatment shapes later
confounders. Subjects are observed in the same periods; in each period `k` the `confounders`
`L_k` are measured, then the 0/1 `treatment` `A_k` is taken, and `outcome` is read once, at each
subject's last period. `active` and `reference` are paths, one action per period, so any static
regime is expressible (always, never, switch on, switch off, and so on). A pooled regression of
the outcome on the final history does not identify the contrast, because the earlier treatment
moves the later confounders; the estimators below do.

Identification: sequential exchangeability (`Y^a` independent of `A_k` given
`L_0..L_k, A_0..A_{k-1}` in every period), which rules out an unmeasured time-varying confounder
of a later treatment and the outcome; sequential positivity of the path actions (every fitted
action probability is held to `min_probability` and a violation is refused as
`positivity_violated`); consistency; no interference between subjects; and complete follow-up.
These are declared, not checked. Loss to follow-up is not modelled: use regime survival for
censored event times.

`method` selects the estimator. `g_formula` runs iterated conditional expectations: from the
last period back, the pseudo-outcome is regressed on the history, the period's treatment and its
interaction with the period's confounders, and its prediction with the period's action set to the
path becomes the next pseudo-outcome. `inverse_probability_weighting` is the saturated marginal
structural model of each path: subjects who followed the path are weighted by the inverse of the
product of their fitted action probabilities (one logistic model per period on the history) and
the mean is the weight-normalized outcome. `doubly_robust` (the default) is the recursive
augmented estimator, which corrects each period's regression prediction by the weighted residual
of the next and is consistent when either the outcome regressions or the treatment models are
right. Every nuisance model is inside the interval: the frequentist interval refits all of them
in each unit-bootstrap replicate, and the Bayesian posterior refits them under Dirichlet unit
weights (a Gaussian coefficient prior on the treatment models) in each draw. The prior supplies
no identifying premise, and coverage has not been measured.

## Regime survival

Rows per subject run `0, 1, 2, ...` while at risk: confounders at the start of the period,
treatment, then censoring during the period (`censored = 1`, event unobserved) or the event.
The target is `S_1(K) - S_0(K)`. Sequential exchangeability of treatment and censoring given
the measured history, positivity of both regimes (the floor `min_probability`) and consistency
are declared, not checked. The frequentist interval refits the pooled logistic treatment and
censoring models in every subject-bootstrap replicate, so the nuisance fits are inside it;
coverage has not been measured. The Bayesian posterior factorizes the trajectory likelihood
into a conjugate Gaussian confounder transition and Laplace pooled-logistic treatment,
censoring and hazard laws, and evaluates the parametric g-formula per draw with common random
numbers across regimes; the treatment and censoring posteriors report the share of draws below
the positivity floor.

## What is refused, and the remedy

| Refused request | Reason code | Remedy |
| --- | --- | --- |
| A realized assignment probability below `min_probability` | `positivity_violated` | A design whose arms keep the floor at every decision, or a lower floor |
| An arm, cell or exposure no paired decision realized | `arm_not_populated` | More units or periods, or probabilities away from the boundary |
| `bootstrap=` on a design whose interval is analytic, or under Bayesian inference | `option_not_applicable` | Omit it; only the quantile functional, frequentist regime survival, the frequentist observational excursion and the frequentist observational regime contrast resample |
| The quantile functional, regime survival, the observational excursion or the observational regime contrast without `bootstrap=` of at least 99 | `required_option_missing` | Set `bootstrap=99` or more |
| A regime contrast with identical paths, paths of different length or of a length other than the number of periods, a nonzero lag, a repeated column, or subjects observed in different periods | `invalid_argument` | Name two different paths with one action per period, keep `lag=0`, and give every subject the same periods |
| A regime whose actions the fitted history does not support (a fitted probability under `min_probability`, or no subject following a path) | `positivity_violated` / `arm_not_populated` | A path the data support, fewer or better-scaled confounders, or a lower floor |
| A local-quadratic grid without `curvature_bound` | `required_option_missing` | Declare the bound, or use `smoothed_curve` |
| A dose outside the observed range, or with local effective sample size below 10 | `cell_not_licensed` | A wider bandwidth or a dose inside the supported range |
| An interference decision with no other cluster member that period | `invalid_argument` | Record every member of the cluster each period |
| A Youden index below 0.1 | `route_not_supported` | A more accurate measurement or a larger validation sample |
| An observational excursion with no history column, a treatment not coded 0/1, or a history column equal to the outcome or treatment | `invalid_argument` | Name the decision-time covariates, code the treatment 0/1, and keep the columns distinct |
| A static switchback | n/a | Switchback assignment is a sequence of periods; there is no static switchback. Use the graphless switchback route or this design |
