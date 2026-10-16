# Missing data

Missing values are never handled by accident. Each route says what it does with them, a
policy is declared with `missing=` (Rust: `StudyBuilder::missing`), and every result records
what happened to the rows.

```python
result = ant.analyze(data, query=query, missing="complete_case")
result.missing_data.policy         # "complete_case"
result.missing_data.assumption     # the assumption the policy needs
result.missing_data.accounting     # rows total / kept / dropped, per variable, per reason
```

## Defaults and policies

| Route | Without `missing=` | Policies it executes |
| --- | --- | --- |
| Average effect, conditional, response, distribution, static mediation | complete case, reported | `error`, `complete_case` |
| Randomized two-arm ITT (every design except CUPED, ANCOVA and receipt-adjusted) | refuses at the first missing outcome | `error`, `complete_case`, `inverse_probability` |
| Panel difference-in-differences | refuses at the first missing outcome | `error`, `balanced_panel`, `available_case` |
| Lagged temporal effect (single window) | refuses at the first missing time point | `error`, `complete_case` |
| NOTEARS discovery | refuses at the first missing cell | `error`, `complete_case`, `pairwise_available` |

A refusal is `missing_policy_required` and names the first missing cell (column and row) and
the policies the route can execute. A policy the route or design cannot execute is refused
when the study is built as `missing_policy_not_applicable`; it is never accepted and
ignored. Routes not listed read their rows in their own way and accept no declaration.

## Row accounting

Every result of the routes above carries `missing_data.accounting`:

- `rows_total`, `rows_kept` (rows whose analysed values entered the estimate),
  `rows_dropped` and `rows_in_denominator` (what the estimate's denominators count);
- `dropped_by_variable`: rows missing each analysed variable; `dropped_by_reason`:
  `missing_value`, `analysis_mask` or `unit_incomplete`;
- `dropped_rows`: zero-based positions in the submitted table, and `dropped_digest`, a BLAKE3
  digest of exactly those positions. With `participant=` the rows are also named in
  `dropped_labels` with `dropped_labels_digest`;
- `groups`: total, kept and fully observed rows per arm or treatment group;
  `units_total` / `units_dropped` for panels.

The policy enters the data-snapshot identity only when it differs from the route's default, so
a study that keeps the default has the identity it always had. The accounting is derived from
the data and the policy and is not part of the identity.

## What each policy assumes

**Average-effect routes, `complete_case`.** Rows with a missing treatment, outcome or
adjustment value are dropped and the estimate speaks for the retained rows. It speaks for the
full population when missingness is independent of the outcome given the treatment and the
adjustment covariates. `error` refuses the table instead.

**Randomized trials, `complete_case`.** The responders of each arm are averaged (difference
in responder means with Neyman variance over responders). This needs outcomes missing
completely at random within arm. Every assigned participant is still counted in the arm sizes
and the assignment vector is never narrowed.

**Randomized trials, `inverse_probability`.** Declare the baseline covariates the response may
depend on (`MissingPolicy.inverse_probability(["baseline"])`). Within each arm the probability
of responding is fitted by logistic regression on those covariates, responders are weighted by
its inverse, and the arm mean is the Hajek ratio. The assumption is outcomes missing at random
given arm and the declared covariates (the chance of responding depends on the outcome only
through them), positivity (fitted response probabilities above 0.02, else the fit refuses) and
a correct logistic model. Every assigned participant stays in its arm's denominator. The
variance is the arm-wise influence-function variance including the score of the response
model. Covariates must be complete. Bernoulli and complete randomization use the Hajek
contrast above. Cluster, stratified, factorial, multi-arm and switchback designs keep their own
estimator and variance (cluster totals, stratum weights, cell contrasts, sequence scores) and
run it on every assigned unit's inverse-probability pseudo-outcome (the weighted outcome, zero
for a non-responder), with the observation model fitted within each assignment group. Their
variance is computed on pseudo-outcomes that carry the influence of the fitted response model (stacked M-estimation), so estimating the response probabilities is accounted for; across strata or clusters the shared coefficients are treated as independent between blocks.
CUPED, ANCOVA and receipt-adjusted fits read further per-unit inputs and refuse.

**Panel difference-in-differences, `balanced_panel`.** Units with a missing outcome in any
period are dropped whole; single unit-periods are never deleted, so no unit is relabelled and
the panel stays balanced. It is valid when a unit's chance of being complete is independent of
its outcome changes within cohort. `complete_case` is refused for panels because it would
unbalance them.

**Panel difference-in-differences, `available_case`.** For staggered designs each event-time
or group-time contrast uses the units observed in its own period and the cohort baseline,
with its own unit and cluster counts; unit and period identities are preserved. For a
repeated cross-section, rows are independent subjects and missing rows are dropped. For a
two-period panel it equals `balanced_panel`. The augmented panel's row-aligned nuisance
columns cannot be restricted and refuse both.

**Temporal, `complete_case`.** Time stays on its original grid. A lag window that reads a
missing time point is dropped and every other window keeps its own time and its own lags, so
no lag crosses a gap and nothing is concatenated. The accounting counts windows, names the
dropped windows by series time and counts missing time points per variable. The point
estimate is the regression on intact windows; it is valid when missingness is independent of
the outcome innovation given the lagged regressors. The circular-block bootstrap
draws its blocks on the original time axis: two windows share a block only when they are
adjacent in time, a replicate keeps the intact windows its blocks cover, and no block joins
windows across a gap. Its Bartlett kernel-bias inflation is not applied (it is defined for
consecutive scores), and no interval is published. The refuters refit through contiguous
windows and are refused. The multi-step sustained, conditional, Bayesian, class and mediation
estimators jointly resample several lagged samples and still refuse a series with gaps.

**NOTEARS, `complete_case`.** Rows with a missing value in any discovery variable are dropped
and reported in `result.row_accounting`. The graph equals the one learned on the complete rows
and speaks for the full data only when missingness is completely at random.

**NOTEARS, `pairwise_available`.** The least-squares score depends on the data only through the
sample covariance, so each pair of variables uses the rows observed on both, the pairwise
covariance (correlation when standardizing) is repaired to positive semidefinite by clipping
negative eigenvalues, and the solver scores that matrix exactly. No row is deleted. The
accounting lists the rows behind every pair and the number of eigenvalues clipped. It needs
missingness completely at random and at least three rows on every pair.

## Claims

A claim over a result whose policy differs from the route's default carries the policy, the
assumption it needs and the digest of the dropped rows among its evidence references. Claims of
default policies keep exactly the references they had.

## Intervals

A handled missingness changes the variance construction, so no interval is published for it:
responder-weighted and complete-case randomized effects, restricted panels and temporal fits
on series with gaps report a point estimate and a standard error where one exists, and the
interval stays closed until a coverage record measures the construction.
