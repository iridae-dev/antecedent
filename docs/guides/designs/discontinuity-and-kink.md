# Estimate an effect at a threshold: regression discontinuity and kink

Estimate the effect of a treatment that switches on, or changes its dependence on a score, at a cutoff of a running variable: an eligibility score, an age, a date. Use a sharp design when crossing the cutoff sets treatment exactly. Use a fuzzy design when crossing it only shifts the probability of treatment. Use a kink design when what changes at the cutoff is the slope of the treatment with respect to the running variable.

You need a table with an outcome column, a treatment column, and a running-variable column. You need the cutoff `c`. You need a bandwidth `h`, the half-width of the window around `c` that the fit reads. The answer is local. It describes units near the cutoff. For a fuzzy design it describes the units whose treatment the cutoff changes (local compliers). Antecedent does not extend it to the rest of the range.

Runnable example: [`examples/python/discontinuity_and_iv.py`](https://github.com/iridae-dev/antecedent/blob/v3.1.0/examples/python/discontinuity_and_iv.py) simulates a sharp jump of 2, a fuzzy design with a treatment effect of 3 and a kink with a dose effect of 3, and checks each estimate against its known value.

## Sharp design

When the running variable is the only cause of treatment, the structure is fixed, so no graph is needed. Sharpness is checked, not assumed. Every complete row must satisfy `T = 1{R ≥ c}`. One row that breaks the rule refuses the design with [`rd_assignment_not_sharp`](../../reference/refusal-codes.md#other-registered-refusal-codes). The call does not report an intention-to-treat contrast under the treatment's name. Use the fuzzy design for imperfect compliance.

```python
import antecedent as ant

result = ant.analyze(data, query=ant.quasi.SharpRegressionDiscontinuity(
    "outcome", "treatment", "running", cutoff=0.0, bandwidth=1.5))
```

The estimand is the average effect for units at the cutoff, reported as a `local_at_cutoff` target. Under the continuity assumption below it equals the jump in the observed outcome regression:

```text
tau_c = E[Y(1) - Y(0) | R = c] = lim_{r↓c} E[Y | R = r] - lim_{r↑c} E[Y | R = r]
```

It is neither the population average effect nor the average over the bandwidth window. When the effect varies with `R` those are different numbers. Only `tau_c` is identified by the design. The estimate is the coefficient on `T` in an ordinary least-squares fit of `Y` on `[1, T, R - c, T·(R - c)]` over the rows with `|R - c| ≤ h`. Every row in the window is weighted equally (a uniform kernel). The sharp call has no bandwidth selector. It has no bias correction. A window wide enough to take in curvature biases the estimate without a diagnostic. Compare the estimate across several bandwidths.

`se` chooses the analytic standard error. `hc1` is the heteroskedasticity-robust default. You can also choose `hc0`, `hc2`, or `hc3`. You can also choose the classical `homoskedastic` formula. Each treats rows as independent. Each ignores smoothing bias. When the bandwidth window holds four rows or fewer the fit is saturated. The jump interpolates the rows exactly, with no residual degrees of freedom. The analytic standard error is withheld (NaN) instead of publishing a rounding-sized value. Widen the bandwidth.

## Fuzzy design and regression kink

`FuzzyRegressionDiscontinuity` reports the ratio of the outcome's intercept jump to the treatment's intercept jump at the cutoff. `RegressionKink` reports the ratio of the outcome's slope change to the treatment's slope change. `r↓c` and `r↑c` are the limits from above and below:

```text
fuzzy RD:  tau_F = (lim_{r↓c} E[Y | R = r] - lim_{r↑c} E[Y | R = r])
                 / (lim_{r↓c} E[T | R = r] - lim_{r↑c} E[T | R = r])
kink:      tau_K = (lim_{r↓c} dE[Y | R = r]/dr - lim_{r↑c} dE[Y | R = r]/dr)
                 / (lim_{r↓c} dE[T | R = r]/dr - lim_{r↑c} dE[T | R = r]/dr)
```

The numerator is the reduced form and the denominator the first stage.

```python
from antecedent.quasi import FuzzyRegressionDiscontinuity, RegressionKink

query = FuzzyRegressionDiscontinuity("outcome", "treatment", "running", cutoff=0.0, bandwidth=1.5)
result = ant.analyze(data, query=query)
```

Both fit local quadratics with triangular kernel weights on either side of the cutoff. The point estimate is bias-corrected from the same bandwidth. A jump uses an estimated cubic term. A kink uses a quartic term. The result's `local_polynomial_ratio` section reports the point estimate, the reduced form, and the first stage. It reports the cutoff, the bandwidth, and the contrast type. It reports local sample counts. It reports HC0 standard errors for the ratio and both components. It reports the ratio's interval (`ci_lower`, `ci_upper`).

Read it this way:

- The ratio's uncertainty is labelled `rbc_hc0_delta_normal_fixed_bandwidth`. That is a pointwise 95% normal interval from the HC0 delta-method standard error at the declared bandwidth. The interval is withheld when that standard error is not positive and finite. A saved artifact rejects interval fields or design metadata that disagree with the query.
- The local counts show sample size near the cutoff only. They do not establish continuity. They do not establish absence of manipulation. They do not establish adequate effective support.
- `support_status` is `supported` when the fit meets its row in the [graphless support matrix](../../graphless-support-matrix.md). The row requires enough local observations on each side and a published interval. There is no weak-instrument-robust inference. A weak first stage is refused instead (below).

### Refusals

The call refuses a singular local fit. It refuses too few positive-kernel observations on either side of the cutoff. It refuses a first-stage change that is numerically zero. It refuses a first-stage change whose 95% interval includes zero. A ratio of two changes is not meaningful when the denominator cannot be separated from zero. Widen the bandwidth, or use a design with a stronger first stage. Look up the code a refusal reports in [refusal codes](../../reference/refusal-codes.md).

### Options

| Option | What it does |
| --- | --- |
| `bandwidth_rule="mse_optimal"` | Selects a plug-in MSE-optimal window, capped at the declared `bandwidth`. It is not the full rdrobust selector. |
| `density_test=True` | Runs a McCrary-style density test of the running variable at the cutoff. A diagnostic only. |
| `balance_covariates=(...)` | Reports each covariate's jump at the cutoff. A diagnostic only. |
| `temporal_replicates=B` | Declares that the running variable is calendar time. Time is held fixed and the errors are redrawn: the errors about a cubic trend on each side are prewhitened by an AR(1) fit, their innovations are resampled in moving blocks within each side and recolored through the cutoff, and the whole ratio is refit per replicate. The interval is the estimate plus or minus a Student-t quantile times the interquartile bootstrap standard error. |

Any of these options returns the answer on `result.quasi_rollout`. Under Bayesian inference the local ratio reads a selected window. A calendar-time running variable uses a within-side block Bayesian bootstrap of the same prewhitened errors. The weights are flat-Dirichlet weights on consecutive blocks of innovations. Each prewhitening coefficient is drawn from its posterior. The whole ratio is refit per draw.

### When these results are supported

`result.quasi_rollout.supported` is true only when the fit meets its own row in the [graphless support matrix](../../graphless-support-matrix.md). A Bayesian fixed-bandwidth posterior states its row in the `estimate.design_posterior.graphless_support` diagnostic instead. Every other fit runs. It publishes its interval. It states no support. The rows cover what was measured:

| Design | Supported when |
| --- | --- |
| Fuzzy RD or kink, `bandwidth_rule="mse_optimal"` | At least 600 rows, at least 114 (jump) or 99 (kink) window rows on each side, and a first stage at least 4.8 (jump) or 4.46 (kink) of its standard errors at the selected bandwidth. |
| Fuzzy RD at the declared bandwidth with diagnostics | At least 150 window rows on each side and a first stage at least 5.5 of its standard errors. The kink at a declared bandwidth with diagnostics has no row. |
| Calendar time, declared bandwidth | At least 149 periods on each side and 199 replicates (500 draws under Bayesian inference). The jump's row is a sharp step: a treatment that changes without error at the cutoff. The kink's row needs a first stage at least 16.6 of its standard errors. |
| Bayesian fuzzy RD or kink, declared bandwidth | `prior_scale=100`, at least 80 window rows on each side, at least 2,000 draws, and a denominator at least 6.33 (jump) or 8.77 (kink) posterior standard deviations from zero. |

Three limits come with these rows.

- **First-stage strength.** The call refuses a first stage whose 95% interval includes zero. A first stage only just past that screen gives a ratio interval that under-covers. Support therefore needs the stronger first stage shown above. The statistic compared is the absolute first-stage estimate over its standard error. Frequentist and calendar-time fits use HC0. The Bayesian fixed-bandwidth posterior uses the posterior mean over the posterior standard deviation.
- **Serial dependence in calendar time.** The errors are modelled as a first-order autoregression. This is a premise you declare. The call does not check it. The rows were measured with AR(1) errors at coefficient 0.7. A strongly persistent higher-order error (an AR(2) at 0.5 and 0.3) under-covers.
- **The Bayesian fixed-bandwidth posterior** was measured on data that follow its own model. The responses are Gaussian. They are linear on each side of the cutoff inside the window. The coefficient prior scale is 100. Curvature inside the window is outside the row. Another prior scale is outside the row. A selected bandwidth is outside the row.

The interval at a selected bandwidth does not account for choosing the bandwidth on the same data. Each interval carries its calibration status (`result.inspect().calibration`).

## Assumptions you declare

- **Sharp RD:** the regression functions `E[Y(1) | R = r]` and `E[Y(0) | R = r]` are continuous in `r` at the cutoff. Continuity fails, for example, when units precisely manipulate the running variable. It also fails when another policy changes at the same cutoff. There is no interference.
- **Fuzzy RD:** the potential-outcome regressions are smooth through the cutoff. There is no precise manipulation of the running variable. Exclusion holds: the cutoff acts on the outcome only through treatment. Monotonicity holds for the local complier effect. There is no interference.
- **Kink:** the potential-outcome derivatives are smooth apart from the treatment-slope change at the cutoff. The same exclusion and monotonicity conditions hold.
- The density and balance diagnostics are descriptive. Neither is a test of the assumptions with a calibrated cutoff.

## Related

- [Estimate a policy effect from panel data](panel-designs.md) when units are followed over time rather than split at a threshold.
- [Reading a result](../../start/reading-a-result.md) for identification, support and calibration fields.
