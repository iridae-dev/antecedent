# Descriptive raw-versus-adjusted comparison and reporting-scale transforms (2.2 E7)

`antecedent.descriptive` (Rust: `antecedent_estimate::descriptive_comparison`) holds two
small computations over estimates you already have. Both are `point_only` and
**descriptive**: they report a gap and re-express a pair of means, and neither identifies
or decomposes a causal effect. The record is `2.2E.E7.descriptive_comparison` in
`parity/promotion_2_2.toml`.

```python
from antecedent.descriptive import raw_vs_adjusted, transform_mean_pair

comparison = raw_vs_adjusted(
    outcome=y, treatment=t,            # treatment coded exactly 0/1
    adjusted_estimate=ate, adjusted_standard_error=se,
)
comparison.raw_difference, comparison.adjusted_estimate, comparison.gap

transform = transform_mean_pair(
    mean_active, mean_control,
    scales=["log_risk_ratio", "log_odds_ratio"],
    covariance=[[var_a, cov], [cov, var_c]],   # as the source estimate published it
)
transform.values, transform.covariance
```

## Raw versus adjusted

The raw contrast is `mean(Y | T=1) - mean(Y | T=0)` over the complete rows, with the Welch
standard error `sqrt(s1^2/n1 + s0^2/n0)`. It is set beside an adjusted estimate that was
computed under **the same estimand coding**: a difference of means, active level 1 versus
control level 0, on the all-observed population. Anything else is refused rather than
compared: another scale or level coding (`route_not_supported`,
`descriptive_comparison.estimand_coding_mismatch`), or a subpopulation, treated-only or
reweighted target (`population_not_estimable`,
`descriptive_comparison.population_not_all_observed`).

`gap = raw - adjusted`. Read it as "how far adjustment moved this number", not as bias,
not as confounding and not as a causal decomposition: it also contains the adjustment
model's functional form, any difference between the estimators, and sampling noise. Two
things are deliberately absent:

* **No interval or standard error for the gap.** The covariance between the raw and the
  adjusted estimate is not carried by either (they use the same rows) and is not assumed
  zero. The field `gap_interval_unavailable` carries the typed reason
  (`cell_not_licensed`, `descriptive_comparison.gap_interval_unavailable`).
* **No attribution to adjustment columns.** The adjusted estimate is not an additive
  function of the columns: a leave-one-out difference depends on the order, on the columns
  that remain and on the functional form. `attribute_gap_to_columns(...)` is a closed
  route and refuses with `effect_not_identified`
  (`descriptive_comparison.column_attribution_not_identified`).

`comparison.export()` writes `descriptive_comparison_v1` JSON (arm counts, means and sums of
squares, the adjusted estimate, the stored results and a SHA-256 digest);
`replay_descriptive_comparison` recomputes the raw difference, Welch standard error and gap
from the sufficient statistics in pure Python and compares them with the stored values.

## Reporting-scale transform

From a mean-outcome pair `(mu1, mu0)` (a licensed result's arm means) `transform_mean_pair`
reports one or more declared scales:

| scale | value | gradient `(d/dmu1, d/dmu0)` | domain |
| --- | --- | --- | --- |
| `mean_difference` | `mu1 - mu0` | `(1, -1)` | any finite pair |
| `risk_difference` | `mu1 - mu0` | `(1, -1)` | both in `[0, 1]` |
| `log_risk_ratio` | `ln(mu1 / mu0)` | `(1/mu1, -1/mu0)` | both `> 0` |
| `log_odds_ratio` | `logit(mu1) - logit(mu0)` | `(1/(mu1(1-mu1)), -1/(mu0(1-mu0)))` | both in `(0, 1)` |

Outside a scale's domain the call is refused (`invalid_argument`,
`descriptive_comparison.transform_domain`).

**Covariance, by the delta method.** For `theta = g(mu)` with `Sigma` the `2 x 2`
covariance of the pair and `J` the `k x 2` Jacobian of the requested family, the first-order
covariance of the family is `J Sigma J'` (a `k x k` matrix; its diagonal square roots are the
standard errors, `transform.standard_error(scale)`). It exists **only when the source
carries `Sigma`**. If you pass no covariance the transformed points are returned and
`covariance` is `None` with `covariance_unavailable` set to
`("required_option_missing", "descriptive_comparison.joint_covariance_unavailable")`.
`None` is never read as independence and never filled with zero: two arm means that share
data are generally correlated. A `Sigma` that is not finite, has a negative variance or is not
positive semidefinite is refused (`descriptive_comparison.invalid_covariance`).

The one covariance this module forms itself is the raw one: the two arms of a sample are
disjoint groups, so `raw_reporting_transform` uses `diag(s1^2/n1, s0^2/n0)`. An arm with
fewer than two rows has no sample variance and the covariance is then unavailable
(`descriptive_comparison.arm_too_small`).

**No interval.** A Wald interval from this covariance would rest on asymptotic normality of
the pair and on the linearization being adequate near the boundary of the scale (a risk
ratio of a rare event, an odds ratio near 0 or 1). No coverage record measures that, so
`level=...` is a typed refusal (`cell_not_licensed`,
`descriptive_comparison.interval_withheld`) and only the points and the covariance are
published.

## Limits

* A transform accepts at most 4 reporting scales in one request.
* Point only; no interval anywhere, no calibration run.
* The comparison is for the difference-of-means coding on the all-observed population; a
  ratio-scale or subpopulation comparison needs its own adjusted estimate and is refused here.
* The delta-method covariance is first order and describes the source `Sigma`'s own
  uncertainty. It adds nothing the source did not license: when the source interval is
  withheld, so is this.
* `raw_vs_adjusted` reads the data you pass it; it does not check that `adjusted_estimate`
  came from the same rows.
* The pass over the rows observes cancellation at the first row, every 4096 rows and once
  after the pass (Rust `raw_contrast` and `compare_raw_adjusted` take an `ExecutionContext`;
  Python `raw_vs_adjusted` and `raw_reporting_transform` take `cancel=`). A cancelled pass is
  `cancelled_no_claim` (`descriptive_comparison.cancelled`) with no contrast, never a verdict
  on the data.
