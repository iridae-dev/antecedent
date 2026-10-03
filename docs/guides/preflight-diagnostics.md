# Preflight diagnostics, rank-drop plans and cost counts (2.2 E0/E1)

Before a nuisance fit refuses a table, `antecedent` can say what is wrong with it.
The surface adds no inferential claim, so it carries no promotion record; it is a
diagnostic and planning layer over existing cells. Two kinds of check are kept in
separate types so one is never read as the other.

| Check | Rust | Python | Fits a model? |
| --- | --- | --- | --- |
| Fit-free preflight | `PreparedStudy::diagnose`, `PreparedBatch::diagnose`, `preflight_design` | `prepared.diagnose()`, `antecedent.preflight.preflight(...)` | No |
| Propensity-fit diagnostics | `diagnose_fit`, `fit_diagnostics_design` | `prepared.diagnose_fit()`, `antecedent.preflight.fit_diagnostics(...)` | Yes |
| Opt-in rank drop | `plan_rank_drop` | `prepared.plan_rank_drop(priority)`, `antecedent.preflight.plan_rank_drop(...)` | No |
| Cost counts | `estimate_cost` | `prepared.estimate_cost()` | No |

`prepared.preflight()` (Python) is the older, structural-only inspection of the plan;
`diagnose` reads the data.

## Fit-free preflight

`diagnose` reads the retained table and the certified adjustment set only. It reports:

- complete-case rows, non-finite cells per column, and rows per arm (`control`/`active`)
  or per joint cell (unweighted, so an arm's count is also its effective sample size);
- exact duplicate columns (equal on the complete-case rows);
- the numerical rank of the `[1 | Z]` design, with every dependent column named and the
  exact linear relation behind it (`combo = 5*(intercept) + 3*a - 2*b`, in original
  units). Columns are scanned in adjustment-set order, so the rank is order-free but
  *which* column of a dependent pair is named is not;
- review flags (below).

Only a numerically rank-deficient design (`design_rank_deficient`) and an unpopulated arm
(`arm_not_populated`) are **blocking**, because regression-based nuisance fits refuse them.
`report.refusal()` turns the first blocking finding into the typed refusal below.

### Review flags

Raised for human review, with the measure and threshold they were compared with. They are
predictive facts about this table, not causal statements, and **never change the
adjustment set or the identification verdict**:

| Code | Meaning |
| --- | --- |
| `near_collinear_column` | a kept column leaves a residual under `1e-3` of its norm after the higher-priority columns (uncentered) |
| `adjustment_column_tracks_treatment` / `_outcome` | a single adjustment column has linear `R^2 >= 0.999` with the treatment or outcome (possible post-treatment variable or label leakage) |
| `treatment_near_determined_by_adjustment` / `outcome_near_determined_by_adjustment` | the whole adjustment design reaches `R^2 >= 0.999` |
| `column_separates_arms` | one adjustment column splits the arms completely or quasi-completely (a positivity warning) |
| `adjustment_duplicates_query_variable`, `duplicate_adjustment_columns`, `duplicate_query_columns` | exact aliases, by role |

Exact duplicates and deterministic dependencies (rank) are deliberately separate from
high correlation (review flags): the first is blocking arithmetic, the second is a
judgement for a person.

## Fit-requiring diagnostics

`diagnose_fit` fits a diagnostic propensity model (arm membership on `[1 | Z]`, **no**
separation ridge, so saturation is reported rather than hidden) and returns the score
min/max, nearest-rank quantiles (0.01 ... 0.99) and the Kish effective sample size of the
inverse-propensity weights in each arm. A fit that fails before any score exists comes
back as `status == "absent"` with the reason (and the numerical rank when that was the
cause); no fitted quantity is reconstructed. A fit that produced scores but is separated,
unconverged or saturated is returned with those flags set.

## Structured refusal fields

`EstimationError::RefusedWithFields` (additive; `Refused`, every existing code and message
are unchanged) carries `RefusalFields`: failing treatment or cell, stage, reason, per-arm
ESS, propensity extrema and quantiles, cluster count, numerical rank, design column count,
implicated columns and a remedy. Every entry is absent unless the failing step measured it.
Read it with `CausalError::refusal_fields()` (Rust) or `error.refusal_fields` (Python,
`None` on other errors). The preflight refusals use the new registered codes
`design_rank_deficient`, `arm_not_populated` and `rank_drop_not_licensed`. The existing
estimator and retarget refusal sites are not rewired to the container in this change.

## Opt-in rank-deficiency drop

`plan_rank_drop` declares a deterministic column priority first (`AdjustmentOrder`, or an
explicit list covering every adjustment column once, highest priority first) and returns a
plan: the dropped columns, the exact relation that makes each redundant, the kept
adjustment set, and a canonical design identity (`adjustment=[a,c];dropped=[b];priority=[a,b,c]`).
A dropped column is a numerically exact linear function of the kept ones on this table, so
it carries no information they do not.

It refuses with `rank_drop_not_licensed` (nothing dropped) when the priority does not cover
the adjustment set, when a dependent column is a treatment, outcome or effect modifier the
query needs, or when an adjustment column is an exact copy of a treatment or the outcome.

**The plan is a record, not an execution.** No estimator is re-run on the reduced design,
and a study prepared with the original adjustment set keeps it.

## Cost counts

`estimate_cost` counts what a prepared plan will do from its frozen configuration: nuisance
fits per pass (derived for cross-fitted AIPW, linear and GLM adjustment, and the propensity
estimators; absent otherwise, never zero), cross-fit folds, bootstrap replicates, passes
(`1 + replicates`, an upper bound that assumes each replicate refits every nuisance), the
`[1 | Z]` design bytes, and the **active inference default**, with a plain-language warning
when it resamples. A batch sums per-claim counts; the fit total is an upper bound because
identical cross-fitted AIPW nuisances are shared across queries.

It is a planning hint, not a runtime guarantee, and gives **counts only**: no seconds are
estimated, because the repository has no named local benchmark of these fits
(`benches/baselines` covers graph, counterfactual, kernel and Laplace workloads).

## Current behavior (E0 reproducers)

Recorded on branch 2.2.0 (crates 2.1.1) by the synthetic, bounded tests in
`crates/antecedent/tests/preflight_reproducers.rs` and `python/tests/test_preflight.py`,
from reading the code paths they exercise.

- **Rank 174 of 175, and duplicate columns.** The QR backend refuses a rank-deficient
  design with `StatsError::RankDeficient { rank, ncols }`: the rank and column count, not
  the column. The logistic IRLS behind the propensity fit goes through the same QR, so a
  duplicate covariate fails there too. Cross-fitted AIPW fits its scores inside `prepare`,
  so such a table never yields a prepared handle for `diagnose` to inspect; use
  `preflight_design` / `antecedent.preflight.preflight` on the declared design instead.
- **175 columns, near-separation.** With about as many columns as half the rows, random arm
  labels are linearly separable and the strict propensity fit refuses (it runs without the
  separation ridge). The fit-free preflight sees a full-rank design with no single
  separating column: the joint separation is visible only to the fit, which is why
  `diagnose_fit` keeps flagged scores. One column that separates the arms is visible
  without a fit (`column_separates_arms`).
- **Retargeted overlap refusal.** A row-weight retarget onto rows the treated arm never
  reaches refuses with `CausalError::Support`; it carries no structured fields today.
- **Configured estimator in a batch.** `BatchStudy::estimator` takes an `EstimatorId`
  (Python: a string id), so a configured estimator struct cannot be supplied to a batch
  entry point.
- **Not reproduced here:** joint-cell instability and repeated-entity or dyad dependence
  (no bounded public workload was added); those stay with their own cells.
