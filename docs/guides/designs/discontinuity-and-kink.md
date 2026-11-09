# Estimate an effect at a threshold: regression discontinuity and kink

Estimate the effect of a treatment that switches on, or changes its dependence on a score, at a cutoff of a running variable: an eligibility score, an age, a date. Use a sharp design when crossing the cutoff sets treatment exactly, a fuzzy design when crossing it only shifts the probability of treatment, and a kink design when what changes at the cutoff is the slope of the treatment with respect to the running variable.

The answer is local. It describes units near the cutoff and, for a fuzzy design, the units whose treatment the cutoff changes (local compliers). Antecedent does not extend it to the rest of the range.

Runnable example: [`examples/python/discontinuity_and_iv.py`](https://github.com/iridae-dev/antecedent/blob/v3.0.0/examples/python/discontinuity_and_iv.py).

## Sharp design

When the running variable is the only cause of treatment, the structure is fixed, so no graph is needed. A treatment column that is not the threshold rule is refused rather than estimated as an intention-to-treat contrast; use the fuzzy design for imperfect compliance.

```python
import antecedent as ant

result = ant.analyze(data, query=ant.quasi.SharpRegressionDiscontinuity(
    "outcome", "treatment", "running", cutoff=0.0, bandwidth=1.5))
```

The effect is the local jump at the cutoff, reported as a `local_at_cutoff` target. `se` chooses the analytic standard error: heteroskedasticity-robust `hc1` (the default), `hc0`, `hc2` or `hc3`, or the classical `homoskedastic` formula. When the bandwidth window holds four rows or fewer the fit is saturated (the jump interpolates the rows exactly, with no residual degrees of freedom), so the analytic standard error is withheld (NaN) instead of publishing a rounding-sized value; widen the bandwidth.

## Fuzzy design and regression kink

`FuzzyRegressionDiscontinuity` reports the ratio of the outcome's intercept jump to the treatment's intercept jump at the cutoff. `RegressionKink` reports the ratio of the outcome's slope change to the treatment's slope change.

```python
from antecedent.quasi import FuzzyRegressionDiscontinuity, RegressionKink

query = FuzzyRegressionDiscontinuity("outcome", "treatment", "running", cutoff=0.0, bandwidth=1.5)
result = ant.analyze(data, query=query)
```

Both fit local quadratics with triangular kernel weights on either side of the cutoff. The point estimate is bias-corrected from the same bandwidth, with an estimated cubic term for a jump and a quartic term for a kink. The result's `local_polynomial_ratio` section reports the point estimate, the reduced form, the first stage, the cutoff, the bandwidth, the contrast type, local sample counts, HC0 standard errors for the ratio and both components, and the ratio's interval (`ci_lower`, `ci_upper`).

Read it this way:

- The ratio's uncertainty is labelled `rbc_hc0_delta_normal_fixed_bandwidth`: a pointwise 95% normal interval from the HC0 delta-method standard error at the declared bandwidth. The interval is withheld when that standard error is not positive and finite, and a saved artifact rejects interval fields or design metadata that disagree with the query.
- The local counts show sample size near the cutoff only. They do not establish continuity, absence of manipulation or adequate effective support.
- `support_status` is `supported` when the fit meets its row in the [graphless support matrix](../../graphless-support-matrix.md) (enough local observations on each side and a published interval). There is no weak-instrument-robust inference; a weak first stage is refused instead (below).

### Refusals

The route refuses singular local fits, too few positive-kernel observations on either side of the cutoff, and a first-stage change that is numerically zero or whose 95% interval includes zero. A ratio of two changes is not meaningful when the denominator cannot be separated from zero. Widen the bandwidth, or use a design with a stronger first stage.

### Options

| Option | What it does |
| --- | --- |
| `bandwidth_rule="mse_optimal"` | Selects a plug-in MSE-optimal window, capped at the declared `bandwidth`. It is not the full rdrobust selector. |
| `density_test=True` | Runs a McCrary-style density test of the running variable at the cutoff. A diagnostic only. |
| `balance_covariates=(...)` | Reports each covariate's jump at the cutoff. A diagnostic only. |
| `temporal_replicates=B` | Declares that the running variable is calendar time. The interval comes from a within-side moving-block bootstrap that refits the whole ratio. |

Any of these options returns the answer on `result.quasi_rollout`, outside the support matrix rows above: its interval is marked unsupported and carries its calibration status (`result.inspect().calibration`). Under Bayesian inference the local ratio reads a selected window, and a calendar-time running variable uses a within-side block Bayesian bootstrap: flat-Dirichlet weights on consecutive time blocks, with the whole ratio refit per draw.

## Assumptions you declare

- **Fuzzy RD:** smooth potential-outcome regressions through the cutoff, no precise manipulation of the running variable, exclusion (the cutoff acts on the outcome only through treatment), monotonicity for the local complier effect, and no interference.
- **Kink:** smooth potential-outcome derivatives apart from the treatment-slope change at the cutoff, with the same exclusion and monotonicity conditions.
- The density and balance diagnostics are descriptive. Neither is a test of the assumptions with a calibrated cutoff.

## Related

- [Estimate a policy effect from panel data](panel-designs.md) when units are followed over time rather than split at a threshold.
- [Reading a result](../../start/reading-a-result.md) for identification, support and calibration fields.
