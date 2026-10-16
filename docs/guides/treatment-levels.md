# Treatment levels and multi-level treatments

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

## Rows at neither level

A row whose treatment is at neither declared level is outside the contrast. Dropping it
changes the population the estimate speaks for, so it is never dropped silently:

- By default the fit refuses with `treatment_level_undeclared`
  (`treatment_levels.undeclared_level`), counting the rows and naming the first value.
- `other_levels = OtherLevels::ExcludeUndeclared` (Rust: `with_other_levels`; Python:
  `other_levels="exclude_undeclared"` on `Aipw`, `PropensityWeighting`, `PropensityMatching`,
  `PropensityStratification` and `DML`) declares the exclusion. The target population is then "rows at the two declared levels" and the
  Rust result reports how many rows were excluded. A declared exclusion executes through the
  ordinary dispatcher, not the checked all-observed routes, which refuse it.

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
so interval claims remain closed.
