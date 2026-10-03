# Joint cells and derived treatments (2.2 E5)

A *joint cell* is the joint intervention `do(T_1 = a_1, .., T_k = a_k)` on up to three
binary components. This guide covers the two pieces added for it: an explicit
**derived-treatment declaration** (`antecedent.derived.DerivedTreatment`, Rust
`DerivedTreatmentDeclaration`) and a **factorized-propensity AIPW** over the cells
(`factorized_joint_cells`, Rust `fit_factorized_joint_cells`). The claim is `point_only`.
The record is `2.2E.E5.factorized_joint_cells` in `parity/promotion_2_2.toml`; no
calibration is run for it.

```python
from antecedent.derived import DerivedTreatment, SourceColumn, factorized_joint_cells

declaration = DerivedTreatment(
    name="a_and_b",
    sources=[
        SourceColumn("a", "treatment_construction", "at_treatment"),
        SourceColumn("b", "treatment_construction", "at_treatment"),
        SourceColumn("age", "admissible_pre_treatment_covariate", "pre_treatment"),
    ],
    legal_values=[0, 1, 2, 3],      # cell code: bit j is the j-th construction source
)
result = factorized_joint_cells(
    data, declaration, outcome="y", adjustment=["age"],
    contrasts=["interaction", "cell_minus_control:3"],
)
for cell in result.cells:           # one entry per cell; an unsupported cell says why
    print(cell.levels, cell.status, cell.estimate)
```

## The derived-treatment declaration

A declaration states the source columns, the transformation, the legal derived values, the
meaning of an intervention, and any exclusions:

- **Source roles.** `treatment_construction` (a component of the treatment itself),
  `admissible_pre_treatment_covariate` (used to build the treatment but measured before
  it, so it may stay in the adjustment set) and `forbidden_descendant` (measured after the
  treatment, never adjusted for). The declared measurement time must agree with the role:
  a post-treatment column can be neither a component nor an admissible covariate.
- **Transformation.** `joint_cell` (up to three components; injective, so setting the cell
  sets every component), `product` or `sum` (many-to-one). A many-to-one transformation with
  the intervention meaning `joint_components` is refused
  (`joint_cells.derived_intervention_not_defined`): the derived value does not fix its
  components. Only `joint_cell` runs through the factorized estimator.
- **Legal values.** At least two distinct finite values (integer cell codes below `2^k`
  for a joint cell). An observed derived value, or a joint-cell component that is not 0/1,
  outside the declaration is refused (`joint_cells.derived_illegal_value`).
- **Exclusions.** A column leaves the adjustment set only through a declared exclusion
  naming its causal rule (`constituent_of_treatment` for a construction source,
  `post_treatment_descendant` for a forbidden descendant) and a written justification. No
  rule excludes an admissible covariate; a stale or repeated exclusion is refused.

`check_derived_treatment` refuses, never repairs. Each refusal is
`derived_treatment_invalid` (or `design_rank_deficient`), carries the implicated columns on
`refusal_fields`, and drops nothing:

| Situation | Detail |
| --- | --- |
| A component in the adjustment set | `joint_cells.derived_constituent_in_adjustment` |
| A descendant in the adjustment set | `joint_cells.derived_descendant_in_adjustment` |
| Exactly duplicated components | `joint_cells.derived_duplicate_constituents` |
| An undeclared exact copy of a component in the adjustment set | `joint_cells.derived_constituent_copy_in_adjustment` |
| A covariate that tracks a component (linear `R^2 >= 0.999`) | `joint_cells.derived_treatment_tracked_by_covariates` |
| A numerically rank-deficient adjustment design | `joint_cells.derived_rank_deficient` (`design_rank_deficient`) |

The duplicate, tracking and rank findings are read from the preflight machinery
(`preflight_design`, see `preflight-diagnostics.md`). That machinery *detects*; it is never
used to choose a column to drop. Fixing a dependency is a declaration: name the column as a
source and exclude it under a causal rule, or remove it yourself.

## The factorized propensity

For an ordering `o` of the components, the cell propensity is the chain-rule product

    P(T = c | Z) = prod_j P(T_{o_j} = c_{o_j} | T_{o_1..o_{j-1}} = c_{o_1..o_{j-1}}, Z).

The product is an identity for any ordering; the ordering matters only through how each
conditional is *estimated*. Each conditional is a binary ridge-logistic model in `Z`
(`PropensityNuisance` with a `RidgeTuning` grid), fit separately inside each observed
prefix stratum, so the model is saturated in the earlier components and penalized in `Z`.
It differs from `CellSaturatedAipw`'s one multinomial logit linear in `Z`.

Folds are cross-fit from the same seeded, cell-stratified, unit-level plan as the other
cell routes. A conditional is fit on a fold's training rows, with the penalty chosen on those
rows only, and predicts that fold's rows. The cell outcome model is OLS on `[1 | Z]` per
cell, as in `CellSaturatedAipw`. The cell score is
`mu_hat_c + 1{T = c} (Y - mu_hat_c) / max(e_hat_c, clip)` and the cell estimate its mean.
The clip (default 0.01, must lie in `(0, 0.5)`) is the only floor; there is no unclipped
mode.

A prefix stratum whose training rows are all one class has a constant conditional
(probability 0 or 1). It is counted in `degenerate_conditionals` and is what lets a
populated cell stay supported when its sibling cell is empty.

## Declared learners

`nuisance="random_forest"` or `"gradient_boosted_trees"` (Rust
`FactorizedJointConfig::learner`, resolved by `declared_joint_nuisance`) replaces both the
ridge-logistic conditionals and the per-cell OLS outcome models with that `antecedent-learn`
learner, with its default hyperparameters. It runs under its own cross-fitting contract:

- **Out of fold only.** The same seeded, cell-stratified unit fold plan is used. A learner is
  fit on a fold's training rows (restricted to the prefix stratum for a conditional, to the cell
  for an outcome model) and predicts only that fold's rows, so no row trains and predicts the
  same fold. A recording learner in the unit tests checks this model by model.
- **No silent fallback.** A fit that fails, or a probability outside `[0, 1]`, makes the factor
  unsupported and refuses the cells that need it (`joint_cells.conditional_unsupported` or
  `joint_cells.outcome_model_unsupported`, naming the fold and the learner's error); the other
  cells keep their estimates. A learner that cannot be resolved in this build refuses the call
  (`route_not_supported`, `joint_cells.learner_unavailable`); `auto` is refused
  (`joint_cells.learner`) because it hides the learner actually fit. A prefix stratum whose
  training rows are one class is still a counted constant and does not call the learner.
- **Provenance and seeds.** The score table's provenance starts with
  `joint_cell.factorized.crossfit.learner_prefix` and carries `;nuisance=ml_joint_cell`, the
  learner identity (`outcome=...;propensity=...`), the distinct fitted implementations and
  `fold_seed=` (the config seed of the fold plan) and `ctx_seed=` (the execution context seed
  the learners draw from). The same declaration, seeds and data replay bit-identically.
- **Point only.** The result keeps the aligned `ScoreTable` and the family contrasts, but no
  interval or covariance: a flexible nuisance needs its own remainder-rate and coverage license
  and none is granted. A requested interval refuses `ml_nuisance_not_licensed`
  (`joint_cells.ml_interval_withheld`).

Other machine-learning names (`ml`, `neural_network`, `gradient_boosting`, ...) are not declared
learners and refuse `ml_nuisance_not_licensed` (`joint_cells.ml_learner_not_declared`). The
default ridge route is unchanged by the option. The tests recover a known effect with OLS and
logistic learners and, in a build with the forest provider, a random forest; no coverage or
calibration of the learner route is claimed.

## Per-cell support and refusals

A cell is evaluated or refused individually, with the others kept
(`joint_cell_unsupported`, or `arm_not_populated` for an empty cell). Under the declared
ordering a cell is refused when:

- it has no complete-case rows (`joint_cells.cell_empty`);
- its outcome model has no residual degrees of freedom in some training fold
  (`joint_cells.outcome_model_unsupported`);
- a conditional factor it needs has no training rows with the required prefix, or its fit
  failed (`joint_cells.conditional_unsupported`); or
- the Kish effective sample size of its in-cell weights is below `min_cell_ess` (default 10;
  `joint_cells.cell_ess_below_minimum`).

Every supported cell reports its weight ESS, the range of its propensity over all rows and
the share of rows below the clip. If no cell is supported the whole call refuses
(`joint_cells.no_supported_cell`) with every reason. A family contrast over a cell that is
not supported is withheld with `joint_cell_unsupported` rather than computed on a subset
(`joint_cells.contrast_cell_unsupported`).

## Checks

- **Normalization.** For every ordering the enumerated cell propensities are summed per row.
  A sum above one by more than `normalization_tolerance`, or (when every cell is
  enumerable) a deviation from one beyond it, refuses the fit
  (`joint_cells.normalization_failed`). With a cell that cannot be enumerated the sum is
  only bounded above by one and `max_abs_error` is absent. Because each conditional is a
  probability and its complement, the identity holds up to rounding; the check guards the
  indexing and the cross-fit assembly, not the model.
- **Ordering sensitivity.** The family is re-estimated under the declared ordering and, with
  `all_orderings`, every other permutation (at most six). A cell is flagged when an
  alternative ordering refuses it, or its estimates differ by more than
  `ordering_tolerance_sd` (default 0.05) outcome standard deviations. Orderings differ only
  through the conditional models' dependence on `Z`, so a flag points at that
  misspecification. It is a receipt about this table, never a verdict on an ordering.
- **Factorization** is tested against direct counting. On hand-counted 2- and 3-component laws
  with no covariate and cell counts divisible by the fold count, each prefix-stratum conditional
  is saturated and its training frequency equals the table's, so the chain-rule product under
  every ordering must equal the empirical joint cell frequency counted in the test, and the
  enumerated propensities must sum to one. This checks the chain-rule indexing and assembly on
  those laws (through the declared-learner route, whose unpenalized logistic is saturated in an
  intercept); it does not show the ridge conditionals fit a law with covariates, which the
  known-law recovery tests assess, and no comparison to the multinomial route is claimed.

## The score table and what is closed

The declared ordering's scores for the supported cells are retained in an aligned
`ScoreTable` (Rust `FactorizedJointFit::scores`) so a family contrast
(`cell_minus_control:<cell>`, and `interaction` for two components with all four cells
supported) has per-row scores. Point contrasts are the difference of the cell means. The
table's provenance carries the ridge tag (or the learner marker, see above), so the
established rule applies: **no interval or joint covariance is published** for a penalized
propensity (see `penalized-aipw.md`) or a learner-supplied nuisance. A requested interval
refuses `penalized_interval_not_licensed` (`joint_cells.interval_withheld`) on the ridge route
and `ml_nuisance_not_licensed` (`joint_cells.ml_interval_withheld`) with a declared learner. No
calibration record exists for this route.

Lasso is closed (`selection_inference_not_licensed`); a machine-learning name that is not a
declared learner refuses `ml_nuisance_not_licensed`; an unknown provider name is an invalid
argument.

## Limits

At most three binary components, so at most eight cells and at most six orderings. The
outcome model per cell is linear in `Z`; a misspecified outcome model weakens the double
robustness that the AIPW score relies on. The route does not take a causal graph: the
adjustment set is the caller's, checked only for the leakage and dependence above. A declared
learner uses its default hyperparameters (no tuning grid) and the Python wrapper accepts no
`penalties` with it.

Cancellation is observed once per cell, once per prefix-stratum conditional, once per penalty of
a ridge conditional and once per cell and fold of a declared learner's outcome models. A
cancelled fit is `cancelled_no_claim` (`joint_cells.cancelled`) with no fit and no score table.
`estimate_joint_cell_cost` states the planned conditional and outcome fits from the same
declaration (see the cost section of the preflight guide).
