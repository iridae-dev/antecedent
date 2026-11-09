# Interval for sampled recovery

When binary variables are missing by a mechanism you declare in a graph, `antecedent.transport.advanced.sampled_observation_recovery` recovers the effect from observed rows and returns an `inference.MeasuredInference` whose `recovered_effect` carries a BCa interval and its calibration status. The coverage record for this procedure measures level 0.95 with 2,000 whole-row bootstrap draws on 1000 to 4000 original rows. Other replicate or row counts are still reported, with a calibration status that says they lie outside the measured scope; they are never refused for that reason. The missingness assumptions in the graph and independent observations are declared premises, and the source recovery artifact and its numerical diagnostics stay `not_measured`.

## How the interval is built

The `bootstrap_bca` procedure applies the bias-corrected and accelerated transformation of [Efron (1987)](https://doi.org/10.1080/01621459.1987.10478410). Asymptotic regularity of BCa alone does not establish finite-sample coverage, which is why the interval's calibration status rests on a coverage record of the whole procedure.

- Each draw reruns the recovery formula and the downstream standardized recovered effect on a resampled table. A failed draw refuses the whole procedure; the interval is never conditioned on successful draws. The recovered law is never renormalized, and the 0.1 mass-defect tolerance of the original recovery stays in force.
- The acceleration uses the complete delete-one-row jackknife. Rows with identical observed patterns have identical delete-one effects, so one recovery runs per positive-count pattern (at most 729 under the six-binary-variable bound), weighted by multiplicity. Every delete-one table must retain the original recovery support.
- Bias correction uses the inverse-normal bootstrap rank with half weight for exact ties to the point estimate. Endpoints are linear (type-7) empirical quantiles.
- A boundary rank, zero jackknife variance, nonfinite arithmetic, transformation pole, or adjusted probability outside `[1/(2000+1), 2000/(2000+1)]` is refused: an endpoint beyond the resampling budget is never claimed.

## Replay

The version-3 artifact binds the method, tie, quantile and jackknife conventions, adjusted probabilities, every grouped delete-one record, the original rows and identities, and every attempted bootstrap draw. Its consumer rechecks identification and reruns the calculation under caller-retained identities and bounds, and accepts only this protocol.

## Related

- [Population, time and uncertainty routes](population-time-uncertainty.md)
- [Transport limits by route](guides/transport-scope.md#exact-binary-observation-recovery)
