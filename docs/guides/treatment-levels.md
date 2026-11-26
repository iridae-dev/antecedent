# Estimate an effect when treatment has several levels

Use this page when the treatment column is not coded `0/1`: it holds two arbitrary levels, labels,
or three or more levels. The route depends on the contrast you want.

| You want | Use | Adjustment |
|---|---|---|
| One contrast between two levels of a column, on the graph-identified route | [`AverageEffect(..., active_level=, control_level=)`](#declared-levels) | Identified from the graph |
| One contrast between two of several levels, with the averaging population declared | [`multi_level_treatment.multi_level_effect`](#treatments-with-more-than-two-levels) | Columns you declare; no graph is checked |
| Every level against one reference level, tested jointly | [`categorical_treatment.categorical_effects`](#compare-every-level-with-a-reference) | Columns you declare; no graph is checked |
| Several treatment columns in one regression | [`vector_treatment.joint_effects`](#estimate-several-treatments-together) | Columns you declare; no graph is checked |

A row at a level you did not declare is never dropped silently. The fit refuses with
`treatment_level_undeclared` (see [refusal codes](../reference/refusal-codes.md#other-registered-refusal-codes))
until you declare what happens to those rows.

These propensity estimators contrast two declared levels of the treatment column. They do not
need the column coded `0/1`.

- `Aipw`
- `PropensityWeighting`
- `PropensityMatching`
- `PropensityStratification`
- the cross-fitted DML AIPW score

## Declared levels

Call `AverageEffect` with `active_level=a` and `control_level=c`. In Rust, call
`AverageEffectQuery::with_levels(treatment, outcome, c, a)`. That names the two arms. The
defaults are `control_level=0.0` and `active_level=1.0`.

```python
import antecedent as ant

result = ant.analyze(
    data,
    graph=graph,
    query=ant.AverageEffect("t", "y", control_level=1, active_level=2),
    estimator="aipw",
)
```

The estimand is `E[Y | do(T=a)] - E[Y | do(T=c)]`. That is the difference between the mean
outcome with the treatment `T` set to the active level `a` and with it set to the control
level `c`. Antecedent recodes `a` to the arm indicator `1` and `c` to `0`. Every later step
is unchanged:

- `1/2` coding with `active_level=2, control_level=1` gives bit-identical estimates to the
  `0/1` coding of the same units.
- Swapping `active_level` and `control_level` on one column negates the contrast exactly: the
  cross-fit folds are dealt on the column's own level order, not on which level is active.
  Recoding the column itself the other way round gives the same contrast for the deterministic
  estimators and agrees up to fold-plan noise for the cross-fitted ones.
- The estimate is the difference between the two arms' mean outcomes, never scaled by the
  distance between the level labels.

The prepared problem and the result record the declared levels: the active and control level,
the rows at each, and the rows excluded. In Rust, read
`PreparedPropensityProblem::treatment_levels`. On the result, read
`EffectEstimate::diagnostics.treatment_levels`.

## Label-coded treatments

A string- or category-coded column is declared by its labels. Call
`AverageEffect("t", "y", control_level="a", active_level="b")`. Both levels must be labels, or
both numbers. The labels are mapped to the arm codes before the contrast runs. The contrast
uses the same declared-level analysis as numeric levels. Codings `a/b`, `0/1`, `1/2` and
`10/20` of the same units give bit-identical estimates. The declaration is frozen with the
study. `ant.rebind` encodes a new table by the same labels. A row at any other label is
refused (`treatment_level_undeclared`, listing the observed levels). It is not dropped.
`result.estimand_text` states the contrast with the declared levels, for example
`E[y | do(t='b')] - E[y | do(t='a')]`.

## Rows at neither level

A row whose treatment is at neither declared level is outside the contrast. Dropping it
changes the population the estimate speaks for, so it is never dropped silently:

- By default the fit refuses with `treatment_level_undeclared` (`treatment_levels.undeclared_level`). It counts the rows, names the first value and lists every observed treatment level with its row count. The estimand is never reduced without saying which levels it would have left out.
- In Rust, call `with_other_levels` with `OtherLevels::ExcludeUndeclared`. In Python, pass `other_levels="exclude_undeclared"` on `Aipw`, `PropensityWeighting`, `PropensityMatching`, `PropensityStratification` or `DML`. That declares the exclusion. The target population is then "rows at the two declared levels". The Rust result reports how many rows were excluded. The exclusion is applied on that estimate. The checked all-observed analyses refuse it.

## Regressions and instruments

`LinearAdjustment` and `GlmAdjustment` enter the treatment as one numeric column. A discrete
column with levels beyond the contrasted pair (`0/1/2`) would be read as a dose. The coefficient
is a slope. It is not the contrast of the two declared levels. Such a column is refused as
`treatment_level_undeclared`, listing the observed levels, unless the rows at the other levels
are declared excluded. In Rust, call `with_other_levels(OtherLevels::ExcludeUndeclared)` on
`LinearAdjustmentAte` or `GlmAdjustmentAte`. In Python, pass
`other_levels="exclude_undeclared"` on `LinearAdjustment` or `GlmAdjustment`. The target
population is then the rows at the two levels. The result reports the excluded count.

The instrumental estimators `IvWald` and `Iv2Sls` refuse a treatment with undeclared extra
levels (`treatment_level_undeclared`, `iv.undeclared_level`). They have no exclusion option.
Restricting the rows to two treatment levels selects on the endogenous treatment and
invalidates the instrument. Recode the treatment into the binary contrast, or use a continuous
dose.

## Treatments with more than two levels

With three or more levels there is no single average effect of "the" treatment. A pair of
levels can be contrasted over different populations. The two answers differ whenever the
effect varies with the covariates. Call `multi_level_treatment.multi_level_effect`. In Rust,
call `MultiLevelAipw` in `antecedent-estimate`. `target` has no default.

```python
from antecedent import multi_level_treatment

# Level 2 against level 0, each standardized over all complete-case rows.
full = multi_level_treatment.multi_level_effect(
    outcome, dose, active=2.0, control=0.0,
    target="full_population", levels=[0.0, 1.0, 2.0], adjust={"age": age},
)
full.effect, full.se_analytic, full.level_means

# Level 2 against level 0 on the rows at level 0 or 2 only.
pair = multi_level_treatment.multi_level_effect(
    outcome, dose, active=2.0, control=0.0,
    target="pairwise_subpopulation", adjust={"age": age},
)
pair.target_population, pair.excluded_rows
```

`outcome`, `dose` and `age` are numeric columns of equal length. A `NaN` marks a missing cell.
Its row leaves the complete cases. You declare the adjustment columns in `adjust`. No graph is
identified or checked. The result records `adjustment_basis == "declared"`. Identification
rests on your claim that the outcome under each level is independent of the received level
given those columns, together with positivity of every contrasted level, consistency and no
interference. `levels` is required for `"full_population"` and must contain `active` and
`control`. `folds` (default 5) sets the cross-fit fold count of the full-population fit.

| Target | Population the contrast averages over | How |
|---|---|---|
| `"pairwise_subpopulation"` (Rust: `MultiLevelAipw::pairwise_subpopulation()`) | rows with `T` in `{active, control}` | Binary cross-fitted AIPW on those rows; rows at other levels are excluded and counted. |
| `"full_population"` (Rust: `MultiLevelAipw::full_population(levels)`) | all complete-case rows | With `Z` the adjustment columns: generalized propensity `e_k(Z) = P(T = k given Z)` from a multinomial logit over the declared `levels`, a per-level outcome regression `m_k(Z)`, and the cross-fitted score `m_k(Z) + 1{T = k}(Y - m_k(Z)) / e_k(Z)` per level. |

The full-population target returns the standardized mean outcome of every declared level and
the contrast of the query's two levels over the same population. Any pair of levels is
comparable with any other. The variance of each mean and of the contrast comes from the shared
score table's influence-function summary. Every row must be at a declared level. A row at an
undeclared level refuses `treatment_level_undeclared`
(`treatment_levels.full_population_undeclared_level`). Use the pairwise target to restrict the
population instead. A training fold with no row at a level, or too few rows at a level for its
outcome regression, refuses `arm_not_populated`. With no overlap clip, a generalized
propensity of exactly zero refuses. The overlap trim is refused. Trimming one level would
redefine the common-support population of every contrast.

The result states which semantics produced it (`target`) and the population it speaks for
(`target_population`). The standard error is the plug-in iid score variance of cross-fitted
scores. It assumes consistent nuisances and positivity. The 95% interval carries its
calibration status in `interval_status`. The matcher's code is in `interval_reason`. The
full-population contrast reports `calibrated` only when a coverage record measured by
`multilevel_generalized_propensity_interval` covers the configuration. Otherwise it reports
`outside_measured_scope`, `measured_below_nominal` or `not_measured`. No coverage record
measures the pairwise target. Its interval always reports `not_measured`.

## Compare every level with a reference

When the question is the adjusted effect of each declared level against one declared reference level, call `antecedent.categorical_treatment.categorical_effects`. It dummy-codes the levels, with the reference omitted, and fits them jointly. Every contrast uses the full covariance. A level with too few rows refuses by name. Nothing is dropped or merged. The rows below are a hand-checkable fixture. The level means are `a` = 2, `b` = 5 and `c` = 9.

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

Declare `ordered=True` with `categories` in order when the levels are ordered. Pass `monotonicity="non_decreasing"` or `"non_increasing"` to test that the adjusted level effects move in that direction. The order is then part of the estimand. A different order is a different question. The monotonicity p-value is a conservative union-intersection bound. Failing to reject does not prove monotonicity. `adjust={"age": age}` adds covariates. Standard errors, p-values and the monotonicity test are diagnostics with calibration not_measured. `fit.caveats` says so. `fit.export()` recomputes the answer from the retained artifact. So does `consume_categorical_effects(..., expected_identity=fit.identity)`.

Rows are independent and equally weighted unless you declare otherwise. `covariance="hc0"` to `"hc3"` is the heteroskedasticity-robust sandwich. `sampling_weights=` takes one weight per row and fits the weighted regression. Rows of weight zero are deleted and counted. The rest are normalized to mean one. The variance is the sandwich of the weighted scores. Weights need a robust covariance. The model-based variance is refused for them. `covariance="cluster_robust"` with `cluster=` (one label per row) is the one-way cluster-robust variance. That is `G / (G - 1) * (n - 1) / (n - p)` times the outer products of the cluster score sums. Two clusters make the joint test singular and refuse. `covariance="newey_west"` with `serial_lag=` (rows in time order at unit spacing) is the Bartlett-weighted long-run variance of the score series. A declaration the covariance kind does not read is refused by name. A kind whose declaration is missing is refused by name. The details are `vector_treatment.cluster_unread`, `.cluster_ids_required`, `.lag_unread`, `.lag_required` and `.invalid_lag`. They are never dropped. `fit.declared_rows` records the rows used, the zero-weight rows, the Kish size of the weights, the cluster count and the lag. The weights, clusters and lag are bound by the artifact digests. The Gaussian `posterior=` is refused for declared rows (`categorical_treatment.posterior_declared_rows`). Its likelihood is independent homoskedastic errors.

## Estimate several treatments together

Call `antecedent.vector_treatment.joint_effects`. It estimates several treatments in one regression that shares one adjustment set and one row snapshot. You get a named coefficient vector and its full covariance. You get declared contrasts whose standard errors use that covariance, a Holm adjustment over the declared contrasts, and the joint Wald test that every treatment coefficient is zero. A different adjustment set, snapshot or row count is refused. A constant or collinear treatment is refused. A rank-deficient block is refused.

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

The contrast's standard error is larger than the naive one here because the two treatment coefficients are negatively correlated. Ignoring the off-diagonal would understate the spread. The coefficients are conditional contrasts of one linear model. They are not a claim that each treatment is separately identified. That rests on the adjustment you declare. As with the categorical analysis, standard errors and p-values carry calibration `not_measured`.

## Refusals you can meet

Each refusal carries a registered code and a `detail`; read both from the raised
`CausalUnsupportedError` (see [structured refusals](../reference/refusal-codes.md#structured-refusals-and-their-remedies)).

| Code or detail | When | What to change |
|---|---|---|
| `treatment_level_undeclared` (`treatment_levels.undeclared_level`) | Rows sit at a level that is neither the active nor the control level. | Declare `other_levels="exclude_undeclared"` on the estimator, or recode the column. |
| `treatment_level_undeclared` (`treatment_levels.full_population_undeclared_level`) | A row is outside `levels` on the full-population target. | Add the level to `levels`, or use the pairwise target. |
| `treatment_level_undeclared` (`iv.undeclared_level`) | An instrumental estimator meets extra treatment levels. | Recode the treatment to the binary contrast or use a continuous dose; there is no exclusion option. |
| `arm_not_populated` (`treatment_levels.level_not_populated`, `.training_fold_missing_level`, `.sparse_level`) | A declared level has no rows, or a training fold has none or too few for its outcome regression. | Declare only levels the data support, or collect rows at that level. |
| `arm_not_populated` (`categorical_treatment.sparse_level`, `.absent_level`) | A declared category has fewer than `min_level_rows` rows (default 2), or none. | Declare only levels the data support. |
| `design_rank_deficient` (`vector_treatment.collinear_treatments`) | Treatment columns are collinear. | Remove or combine the redundant treatment. |

`treatment_level_undeclared` and `arm_not_populated` are described under [other registered refusal codes](../reference/refusal-codes.md#other-registered-refusal-codes).

## Related

- [Heterogeneous effects and policies](heterogeneous-effects-and-policies.md) for the value of a rule that assigns one of several actions.
- [Responses, derivatives and distributions](../causal-responses.md) when the treatment is a continuous dose.
- [Sampling weights](sampling-weights.md) and [clusters](clustered-dml.md) for the row declarations the categorical route reads.
- [Reading a result](../start/reading-a-result.md) for calibration statuses such as `not_measured`.
