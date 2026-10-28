# Estimate an effect when treatment has several levels

The propensity estimators (`Aipw`, `PropensityWeighting`, `PropensityMatching`,
`PropensityStratification`, and the cross-fitted DML AIPW score) contrast two declared levels
of the treatment column. They do not need the column coded `0/1`.

## Declared levels

`AverageEffect(..., active_level=a, control_level=c)` (Rust:
`AverageEffectQuery::with_levels(treatment, outcome, c, a)`) names the two arms. Preparation
recodes `a` to the arm indicator `1` and `c` to `0`, so every downstream step is unchanged:

- `1/2` coding with `active_level=2, control_level=1` gives bit-identical estimates to the
  `0/1` coding of the same units.
- Swapping `active_level` and `control_level` on one column negates the contrast exactly: the
  cross-fit folds are dealt on the column's own level order, not on which level is active.
  Recoding the column itself the other way round gives the same contrast for the deterministic
  estimators and agrees up to fold-plan noise for the cross-fitted ones.
- The estimate is the difference between the two arms' mean outcomes, never scaled by the
  distance between the level labels.

The prepared problem and the result record the declared levels (Rust:
`PreparedPropensityProblem::treatment_levels`, `EffectEstimate::diagnostics.treatment_levels`):
the active and control level, the rows at each, and the rows excluded.

## Label-coded treatments

A string- or category-coded column is declared by its labels:
`AverageEffect("t", "y", control_level="a", active_level="b")`. Both levels must be labels (or
both numbers). The labels are mapped to the arm codes before preparation and the contrast runs
the same declared-level path as numeric levels, so `a/b`, `0/1`, `1/2` and `10/20` codings of the
same units give bit-identical estimates. The declaration is frozen with the study: `ant.rebind`
encodes a new table by the same labels, and a row at any other label is refused
(`treatment_level_undeclared`, listing the observed levels) rather than dropped.
`result.estimand_text` states the contrast with the declared levels, for example
`E[y | do(t='b')] - E[y | do(t='a')]`.

## Rows at neither level

A row whose treatment is at neither declared level is outside the contrast. Dropping it
changes the population the estimate speaks for, so it is never dropped silently:

- By default the fit refuses with `treatment_level_undeclared`
  (`treatment_levels.undeclared_level`), counting the rows, naming the first value and
  listing every observed treatment level with its row count, so the estimand is never reduced
  without saying which levels it would have left out.
- `other_levels = OtherLevels::ExcludeUndeclared` (Rust: `with_other_levels`; Python:
  `other_levels="exclude_undeclared"` on `Aipw`, `PropensityWeighting`, `PropensityMatching`,
  `PropensityStratification` and `DML`) declares the exclusion. The target population is then "rows at the two declared levels" and the
  Rust result reports how many rows were excluded. A declared exclusion executes through the
  ordinary dispatcher, not the checked all-observed routes, which refuse it.

## Regressions and instruments

`LinearAdjustment` and `GlmAdjustment` enter the treatment as one numeric column, so a discrete
column with levels beyond the contrasted pair (`0/1/2`) would be read as a dose: the coefficient
is a slope, not the contrast of the two declared levels. Such a column is refused as
`treatment_level_undeclared`, listing the observed levels, unless the rows at the other levels
are declared excluded (Rust: `with_other_levels(OtherLevels::ExcludeUndeclared)` on
`LinearAdjustmentAte` and `GlmAdjustmentAte`; the target population is then the rows at the two
levels and the result reports the excluded count). The Python `LinearAdjustment` and
`GlmAdjustment` have no `other_levels` argument, so from Python the refusal stands: recode the
treatment into the contrast of interest, or use a propensity estimator or `DML`, which do
declare it.

The instrumental estimators (`IvWald`, `Iv2Sls`) refuse a treatment with undeclared extra levels
(`treatment_level_undeclared`, `iv.undeclared_level`) and have no exclusion option: restricting
the rows to two treatment levels selects on the endogenous treatment and invalidates the
instrument. Recode the treatment into the binary contrast, or use a continuous dose.

## Treatments with more than two levels

In Python the two targets are `antecedent.multi_level_treatment.multi_level_effect(outcome,
treatment, active=..., control=..., target="full_population" | "pairwise_subpopulation",
levels=..., adjust={...})`; `target` has no default. The caller declares the adjustment columns:
no graph is identified or checked.

With three or more levels there is no single average effect of "the" treatment: a pair of
levels can be contrasted over different populations, and the two answers differ whenever the
effect varies with the covariates. `MultiLevelAipw` (Rust, `antecedent-estimate`) makes the
choice explicit; there is no default.

| Target | Population the contrast averages over | How |
|---|---|---|
| `MultiLevelAipw::pairwise_subpopulation()` | rows with `T` in `{active, control}` | Binary cross-fitted AIPW on those rows; rows at other levels are excluded and counted. |
| `MultiLevelAipw::full_population(levels)` | all complete-case rows | Generalized propensity `e_k(Z) = P(T = k given Z)` from a multinomial logit over the declared `levels`, a per-level outcome regression `m_k(Z)`, and the cross-fitted score `m_k(Z) + 1{T = k}(Y - m_k(Z)) / e_k(Z)` per level. |

The full-population target returns the standardized mean outcome of every declared level and
the contrast of the query's two levels over the same population, so any pair of levels is
comparable with any other. The variance of each mean and of the contrast comes from the shared
score table's influence-function summary. Every row must be at a declared level: a row at an
undeclared level refuses `treatment_level_undeclared`
(`treatment_levels.full_population_undeclared_level`); use the pairwise target to restrict the
population instead. A training fold with no row at a level, or too few rows at a level for its
outcome regression, refuses `arm_not_populated`. With no overlap clip, a generalized
propensity of exactly zero refuses; the overlap trim is refused because trimming one level
would redefine the common-support population of every contrast.

The result states which semantics produced it (`target`) and the population it speaks for
(`target_population`). The standard error is the plug-in iid score variance of cross-fitted
scores: it assumes consistent nuisances and positivity, and no coverage record measures it,
so the 95% interval of the full-population contrast is withheld as `scope_not_assessed`
(`interval_status`, with the matcher's code in `interval_reason`) until a coverage record
measured by `multilevel_generalized_propensity_interval` covers the configuration. The pairwise
target never publishes an interval.

## Compare every level with a reference

When the question is the adjusted effect of each declared level against one declared reference level, `antecedent.categorical_treatment.categorical_effects` dummy-codes the levels (the reference omitted) and fits them jointly, so every contrast uses the full covariance. A level with too few rows refuses by name; nothing is dropped or merged. The rows below are a hand-checkable fixture: the level means are `a` = 2, `b` = 5 and `c` = 9.

```python
from antecedent.categorical_treatment import categorical_effects

levels = ["c", "a", "b", "c", "b", "a", "c", "b", "c"]
outcome = [9.0, 1.0, 4.0, 7.0, 6.0, 3.0, 11.0, 5.0, 9.0]
fit = categorical_effects(
    outcome, levels, categories=["a", "b", "c"], reference="a", pairs=[("b", "c")]
)
fit.effect("b").estimate       # 3.0, effect(b) - effect(a)
fit.effect("c").estimate       # 7.0
fit.pair("b", "c").estimate    # 4.0, with a standard error that uses the off-diagonal covariance
fit.omnibus.p_value            # every level has the reference level's effect
fit.effect("c").p_holm         # Holm over the declared family of contrasts
```

Declare `ordered=True` with `categories` in order when the levels are ordered, and `monotonicity="non_decreasing"` (or `"non_increasing"`) to test that the adjusted level effects move in that direction; the order is then part of the estimand and a different order is a different question. The monotonicity p-value is a conservative union-intersection bound, and failing to reject does not prove monotonicity. `adjust={"age": age}` adds covariates. Standard errors, p-values and the monotonicity test are diagnostics with calibration unmeasured (`fit.caveats` says so). `fit.export()` and `consume_categorical_effects(..., expected_identity=fit.identity)` recompute the answer from the retained artifact.

## Estimate several treatments together

`antecedent.vector_treatment.joint_effects` estimates several treatments in one regression that shares one adjustment set and one row snapshot. It returns a named coefficient vector, its full covariance, declared contrasts whose standard errors use that covariance, a Holm adjustment over the declared contrasts and the joint Wald test that every treatment coefficient is zero. A different adjustment set, snapshot or row count, a constant or collinear treatment, or a rank-deficient block is refused.

```python
from antecedent.vector_treatment import Contrast, joint_effects

x1 = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0]
x2 = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
y = [3.0, 2.0, 5.0, 1.0, 0.0, 4.0]
fit = joint_effects(
    y, {"t1": x1, "t2": x2}, contrasts=[Contrast("diff", {"t1": 1.0, "t2": -1.0})]
)
fit.coefficient("t1").estimate    # 2.75
fit.coefficient("t2").estimate    # 0.75
fit.contrast("diff").estimate     # 2.0
fit.contrast("diff").standard_error, fit.contrast("diff").naive_independent_standard_error
fit.joint_wald.p_value
```

The contrast's standard error is larger than the naive one here because the two treatment coefficients are negatively correlated; ignoring the off-diagonal would understate the spread. The coefficients are conditional contrasts of one linear model, not a claim that each treatment is separately identified: that rests on the adjustment you declare. As with the categorical route, standard errors and p-values carry calibration `unmeasured`.
