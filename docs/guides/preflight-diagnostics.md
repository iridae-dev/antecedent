# Preflight diagnostics, rank-drop plans and estimates, and cost counts (2.2 E0/E1)

Before a nuisance fit refuses a table, `antecedent` can say what is wrong with it.
The diagnostics and the plan add no inferential claim; they are a diagnostic and
planning layer over existing cells. The one entry point that estimates after a rank
drop (`estimate_with_rank_drop`) does change what is computed, so it carries its own
promotion record, `2.2E.E1.preflight_rank_drop_estimation`, `point_only`. Two kinds
of check are kept in separate types so one is never read as the other.

| Check | Rust | Python | Fits a model? |
| --- | --- | --- | --- |
| Fit-free preflight | `PreparedStudy::diagnose`, `PreparedBatch::diagnose`, `preflight_design` | `prepared.diagnose()`, `antecedent.preflight.preflight(...)` | No |
| Propensity-fit diagnostics | `diagnose_fit`, `fit_diagnostics_design` | `prepared.diagnose_fit()`, `antecedent.preflight.fit_diagnostics(...)` | Yes |
| Opt-in rank drop plan | `plan_rank_drop` | `prepared.plan_rank_drop(priority)`, `antecedent.preflight.plan_rank_drop(...)` | No |
| Estimate after a rank drop | `estimate_with_rank_drop` | `antecedent.preflight.estimate_with_rank_drop(...)` | Yes (the estimator) |
| Cost counts and, with a benchmark file, seconds | `estimate_cost` | `prepared.estimate_cost()` | No |

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

`RefusalFields` (Rust `CausalError::refusal_fields()`, Python `error.refusal_fields`, `None`
on errors that carry none) holds the failing treatment, cell or claim (`subject`), `stage`,
`reason`, per-arm ESS, propensity extrema and quantiles, cluster count and minimum, numerical
rank and design column count, implicated columns, a `remedy`, and, for a refused GLM fit, its
iterations, nearest fitted-probability margin and the count of rows within `1e-8` of 0 or 1.
Every entry is absent (`None`, or an empty list) unless the failing step measured it; a fit
that failed before it produced scores never has a propensity range reconstructed for it.

Fields ride beside the refusal and never change it: every existing reason code, message and
exception class is preserved (the tests pin the message strings). The rendered text of a
refusal never includes the fields.

Sites that populate the fields:

| Site | Stage | Reasons | Numbers carried |
| --- | --- | --- | --- |
| Preflight (`design_rank_deficient`, `arm_not_populated`, `rank_drop_not_licensed`), derived treatment, rank-drop estimate | `preflight`, `rank_drop`, ... | the finding | rank, columns, implicated columns, arm rows |
| GLM refused by an estimation path (`GlmFit::require_ok`: propensity, GLM adjustment, cross-fitted AIPW folds) | `glm_fit` | `non_converged`, `separated`, `boundary_saturated` | iterations, nearest margin, rows within `1e-8` |
| Rank-deficient design from the QR backend (the real AIPW refusal at rank 174 of 175) | `design_rank` | `rank_deficient` | numerical rank, design columns; subject and implicated columns from the facade (below) |
| Cluster-robust variance with fewer than two clusters (GLM cluster and multiway sandwiches, influence-function cluster, multiway and panel SEs) | `cluster_variance` | `too_few_clusters` | clusters found, minimum |
| Cluster-DML cluster, component and dyadic-giant-component refusals | `cluster_dml` | `too_few_clusters`, `too_few_components_per_fold`, `dyadic_giant_component` | found, declared minimum |
| Positivity refusal on fitted scores (propensity of exactly 0 or 1 with no clip: propensity estimators, cross-fitted AIPW, cell AIPW) | `positivity` | `propensity_not_interior` | extrema and nearest-rank quantiles of the finite scores |
| Weighted-overlap gate of a retarget and of a custom-weight AIPW | `retarget` | `arm_effective_sample_size_below_minimum`, `propensity_range_touches_zero_or_one`, `extreme_propensity_share_above_limit`, `propensity_range_unavailable`, `arm_support_not_computed` | per-arm Kish ESS, raw propensity range and nearest-rank quantiles of the finite raw propensities of rows the target weights |
| Retarget `depends_on` refusals (treatment or descendant, outside the adjustment set, no graph, undeclared nonconstant weights) | `retarget` | one per refusal | none: no score exists yet |
| Failed batch-retarget members and contrasts, and family declaration errors | `batch_retarget` | the stable `batch_retarget.*` detail | the overlap figures above for an overlap failure; the claim or contrast name as `subject` |
| Penalized-propensity fallback refusal | that of the failed fit | that of the failed fit | those of the failed fit |
| Builder: configured estimator plus an explicit `bootstrap_replicates` or `overlap_policy` (also reached through batch entry points) | `study_build` | `set_on_builder_and_configured_estimator` | none; the setting is the `subject` |

How they are attached, so a caller reads the right thing: a stats-layer refusal stays an
`EstimationError::Stats` (its text, reason-code absence and Python class
`CausalEstimateError` are unchanged) and its fields are derived from the record it carries;
an estimator refusal that must keep its variant is `EstimationError::UnsupportedWithFields` or
`OverlapWithFields`; a facade refusal that must keep its variant (the retarget support refusal,
the builder conflict, and a stats refusal that crossed `prepare`) is wrapped in
`CausalError::Diagnosed`, which renders and classifies as the refusal it wraps (match it
through `CausalError::peeled()`). A batch member reports its fields as
`MemberFailure::fields` (Python `RetargetMember.refusal_fields`).

The stats layer knows neither the treatment nor column names, so the facade adds them, with
the plan label preflight uses as `subject` (`effect(t -> y)`, `cell(a, b -> y)`). Where each
refusal crosses into the facade:

- the prepare-time score-table fit of a cross-fitted AIPW or cell-AIPW plan;
- every estimate of a prepared plan (`PreparedStudy::estimate`, which is where regression,
  GLM, IV and matching design fits run), against the table that estimate read;
- a one-shot `Study::run`, which has no frozen plan: after a stats refusal it prepares the
  study to obtain the identification, and names the refusal against that plan (the label
  alone when preparation cannot supply one);
- the retarget refusals (weighted overlap and the `depends_on` refusals), which take the plan
  label as `subject`; a batch member's subject stays the claim or contrast name.

For a rank refusal, `implicated_columns` is the columns the fit-free preflight scan names as
dependent on that table's `[1 | Z]` design. A refusal on a fold's subset may be a smaller rank
than the table's, and which of two identical columns is named depends on the scan
(adjustment-set) order. If the scan finds none, preflight cannot run, or the plan is outside
preflight's cells (static tabular average effects and discrete joint cells), the entries stay
absent. Stats refusals raised on routes the facade does not reach through those entry points
(for example temporal or transport estimators) carry the stage and numbers but no subject.

What the fields do not claim: the arm sizes and quantiles of a refused retarget are those the
overlap gate compared (Kish sizes under the target weights, raw held-out propensities of rows
the target weights), not a verdict on identification. A refusal raised before any score exists
has them absent.

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
and a study prepared with the original adjustment set keeps it. Executing a plan is the
separate, explicit entry point below.

## Estimating after a rank drop

`estimate_with_rank_drop(input, policy, estimator, ctx)` (Rust) and
`antecedent.preflight.estimate_with_rank_drop(data, treatment=, outcome=, adjustment=,
estimator=, priority=)` (Python) re-run a licensed estimator on the design with the plan's
dependent columns removed. It works on a declared binary-contrast design, the same input
as `preflight_design`, because a rank-deficient table cannot be prepared (the cross-fitted
AIPW fits its scores at prepare time, and a prepared handle with the original set keeps it).

It returns the **point estimate** together with the recorded drop (`plan`: dropped columns,
the exact relation behind each, the declared priority, the original and reduced adjustment
sets and the design identity), the numerical `span_check` and the estimator's invariance
label. It never drops silently and it refuses, with nothing estimated, when:

- the plan refuses (`rank_drop_not_licensed`: the priority does not cover the set, a
  dependent column is a protected effect modifier, or an adjustment column is an exact
  copy of the treatment or outcome, a treatment alias that a drop would not resolve);
- the independent span check fails (`rank_drop_not_licensed`, detail
  `rank_drop_estimate.span_not_preserved`): the retained `[1 | Z_kept]` must have full
  numerical rank equal to the original's, and every dropped column must lie in its span
  within the scan tolerance (ten times, for rounding);
- the estimator is not one of the two below, or the design is not a binary contrast of one
  treatment (`route_not_supported`);
- for cross-fitted AIPW, the diagnostic propensity fit on the reduced design separates,
  saturates, fails or does not converge (`rank_drop_not_licensed`, detail
  `rank_drop_estimate.propensity_separates`).

**Why a span-preserving drop cannot change the fitted nuisances.** A dropped column is an
exact linear combination of the intercept and the retained columns, so the reduced and the
original design have the same column space `V`. Ordinary least squares depends on the design
only through `V`: the fitted values are the projection of the outcome onto `V`, and the
treatment coefficient is the same linear functional of them. An unpenalized logistic maximum
likelihood depends on the design only through the set of linear predictors `{X b} = V`, so
its fitted probabilities coincide. Only the coefficients of the original design are not
identified, and the estimand never reads them. The reduced adjustment set is the original
set's projections, so no new identification claim is made: the returned sets are the
declared design's, not a graph search's. (The estimator itself is run through the ordinary
study path on the reduced table with a declared confounder graph over the retained columns;
that graph only carries the retained set into the estimator.)

Licensed estimators and their invariance label:

| Estimator | `projection_invariance` | Notes |
| --- | --- | --- |
| `linear.adjustment.ate` | `exact` | OLS; the reduced coefficient equals the OLS on the hand-built reduced design |
| `aipw` (default, unpenalized logistic) | `unpenalized_logistic_no_separation` | conditional: a fold-level separation that a ridge would regularize is not detected, only the full reduced-design fit is checked |

Penalized, lasso, matching and stratification nuisances are refused: a penalty depends on
the parametrization, not only the column space. The result is point only: the bootstrap is
forced to zero and no standard error or interval is returned or requestable.

Tests: the reduced estimate equals the estimate on a hand-built reduced design bit for bit,
and a NumPy / normal-equations OLS to 1e-8; a different declared priority drops a different
column and changes the design identity but not the estimate; a treatment alias, an
unlicensed estimator, a joint cell, a separating propensity and a span violation each
refuse.

## Cost counts and time estimates

`estimate_cost` counts what a prepared plan will do from its frozen configuration: nuisance
fits per pass (derived for cross-fitted AIPW, linear and GLM adjustment, and the propensity
estimators; absent otherwise, never zero), cross-fit folds, bootstrap replicates, passes
(`1 + replicates`, an upper bound that assumes each replicate refits every nuisance), the
`[1 | Z]` design bytes, and the **active inference default**, with a plain-language warning
when it resamples. A batch sums per-claim counts; the fit total is an upper bound because
identical cross-fitted AIPW nuisances are shared across queries.

### Counts for the 2.2 fit routes

Counts are read off the configuration the route executes (`CostEstimate.fit_route` names the
counted route), never from a second copy of its loop:

| Route | Propensity fits per pass | Outcome fits per pass |
|---|---|---|
| AIPW (unpenalized) | `folds` | `2 * folds` |
| AIPW, ridge or lasso propensity | `folds * penalties * (inner_folds + 1)`, an upper bound (a penalty is refit on the whole training set only when its inner loss improves) | `2 * folds` |
| AIPW, penalized GLM fallback declared | the above plus the destination's selection, an upper bound (it runs only when the GLM fit fails) | `4 * folds`, the outcome models run again |
| AIPW, cluster-DML declaration | `folds` (whole clusters, never rows, own the folds) | `2 * folds`; `cluster_labels` is the distinct cluster count |
| `dml` (AIPW score) | `folds` of the DML estimator | `2 * folds` |
| `dml` (partially linear score) | `folds` | `folds` |
| `dr_learner` | `folds` | `2 * folds + 1` (the final-stage regression) |

Refit-bootstrap replicates add passes (`1 + replicates`, each repeating the whole pipeline
including penalty selection), so a penalized plan's upper bound is `(per pass) * (1 +
replicates)`. The ML fallback and a full-sample lasso are closed and add no fit. Counts are
monotone in folds, replicates, the penalty grid and the inner folds (unit-tested), and the
penalized, cluster-DML and joint-cell counts equal the polls an instrumented run makes
(`cancellation_expansion_cells.rs`: one cancellation poll per penalty of every outer fold, per
whole-cluster fold, per cell, per conditional and per learner fold).

Outside a prepared plan:

- `BatchScores::estimate_retarget_cost(request)` counts a batch retarget: **zero nuisance
  fits** (the retained scores are reweighted), the claims, the contrasts and the snapshot rows,
  `claims * rows` weighted score reads and `claims * (claims + 1) / 2 * rows` covariance
  products. The unpublished max-t evaluator polls once per 1024 draws.
- `estimate_joint_cell_cost(config, components, orderings, rows)` counts a factorized joint
  cell from `FactorizedJointConfig::planned_fits`: per fold the distinct prefix-stratum
  conditionals over every ordering (shared between orderings), `folds * strata` conditional
  fits, each costing its ridge penalty selection (`penalties * (inner_folds + 1)`) or one fit
  under a declared learner, plus `cells * folds` per-cell outcome models. An invalid
  declaration is refused as the fit refuses it, never costed.
- `estimate_with_rank_drop` costs one fit-free rank scan, plus for AIPW one diagnostic
  propensity fit, plus the counts of the reduced study itself. Preflight, the inverse outcome,
  matched case-control, the descriptive comparison and the tier diagnostics fit no nuisance
  model.

It is a planning hint, not a runtime guarantee. **Seconds** are reported only when a named
local benchmark file backs them:

```
python3 scripts/bench_cost_model.py        # one command, writes parity/cost_model.toml
```

The script (run it on a quiet machine, on the python package built from the tree) times
`antecedent.analyze` for `linear.adjustment.ate` and `aipw` over a small `(rows, design
columns)` grid, with no bootstrap and with `--bootstrap` replicates (default 10), keeping the
fastest of `--repeats` runs. One sample is the best wall time divided by the nuisance-fit
count `estimate_cost` itself reports for that plan, so the cross-fit and bootstrap steps are
timed through the fits they add. Per estimator it fits the monotone model

```
seconds per counted fit = a + b * rows * columns^2        (a, b >= 0)
```

by least squares in `x = rows * columns^2`, clamped to non-negative coefficients, and writes
the coefficients, the grid, the benchmark name and the machine descriptor (platform, CPU
count) to `parity/cost_model.toml`. `estimate_cost` then reports
`seconds = nuisance_fits_upper_bound * (a + b * rows * columns^2)` with
`seconds_basis` set to `planning hint from named local benchmark <name>, machine <platform>
(<n> cpus)`. The model is end to end for the benchmarked plans: for an estimator whose
bootstrap replicates do not each refit every nuisance, the per-fit figure absorbs that and the
count stays an upper bound. The file is read from `ANTECEDENT_COST_MODEL`, else
`parity/cost_model.toml` relative to the working directory.

`seconds` stays `None`, with the reason in `seconds_basis`, when the file is missing or
unreadable, declares no machine or a negative coefficient, has no coefficients for the plan's
estimator, or the plan's rows, columns or fit counts are not derived. The benchmark times
plain `linear.adjustment.ate` and `aipw` fits only, so a penalized, lasso, fallback, `dml` or
`dr_learner` plan, a batch retarget and a joint cell report counts with `seconds = None` and
the reason; a cluster-DML declaration fits the same logistic and OLS models per fold as plain
AIPW and is timed with its coefficients. The prediction is
monotone in folds, replicates, rows and columns (unit-tested on a synthetic coefficients
struct, with hand arithmetic). Seconds describe the benchmark machine, not yours.

**How accurate the hint is.** On the committed grid (rows 400 to 1600, columns 4 to 16, one
run on the macOS arm64 development machine), the AIPW model predicts between 0.2 and 14 times
the measured time per counted fit: within 0.2 to 2.4 times without a bootstrap, and 2.2 to 14
times too high with 10 bootstrap replicates (a replicate does not cost a full fit, so the
count is an upper bound that the model then overstates). Per-fit time also jumps between 8
and 16 columns on that machine, which a quadratic term cannot follow. Treat `seconds` as an
order-of-magnitude upper bound for bootstrap plans, not a prediction, and regenerate the file
on the machine you plan for.

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
  reaches refuses with `CausalError::Support` (wrapped in `CausalError::Diagnosed`), and its
  fields carry the per-arm ESS and the propensity range under the target.
- **Configured estimator in a batch.** `BatchStudy::estimator_spec` now takes a configured
  estimator, so that limit is gone; combining it with an explicit `bootstrap_replicates` is
  refused when the study is built, with the setting named as the refusal's subject.
- **Not reproduced here:** joint-cell instability and repeated-entity or dyad dependence
  (no bounded public workload was added); those stay with their own cells.
