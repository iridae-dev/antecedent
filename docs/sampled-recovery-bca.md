# Whole-row sampled recovery: BCa intervals

`antecedent.transport.advanced.sampled_observation_recovery` returns a checked
`inference.MeasuredInference` for the named `recovered_effect` at level 0.95,
with 2,000 whole-row bootstrap draws and sample counts 1000..4000. The original
recovery derivation and method must resolve to a current attesting coverage record.
Correct missingness assumptions and independent observations are declared premises.
Other sample sizes or protocols cannot inherit this calibration.

The `bootstrap_bca` procedure uses the bias-corrected and accelerated transformation
of [Efron (1987), *Better Bootstrap Confidence Intervals*](https://doi.org/10.1080/01621459.1987.10478410).
Its frozen repeated-sampling measurements, source replay and ordinary-wheel
producer/consumer checks support the measured adapter. Asymptotic BCa regularity
alone does not establish finite-sample coverage. The retained version-3 source
artifact remains `unmeasured`; the distinct measured envelope licenses only its
`recovered_effect` endpoints.

Each draw reruns the original licensed observation-recovery formula and downstream
standardized recovered effect. The procedure refuses any failed bootstrap draw;
it does not condition its interval on successful draws. The empirical recovered
law is never renormalized; the effect probabilities use their evaluated totals,
and the original 0.1 mass-defect tolerance remains in force.

The acceleration uses the complete delete-one-row jackknife. Rows with identical
observed patterns have identical delete-one effects, so the implementation runs
one recovery per positive-count pattern and weights the resulting centered second
and third moments by the original pattern multiplicity. There are at most 729
valid patterns under the six-binary-variable bound. Every delete-one table uses
its actual `n - 1` denominator and must retain the original recovery support.

Bias correction uses the inverse-normal bootstrap rank with half weight for exact
floating-point ties to the original point estimate. The two transformed
probabilities come from the BCa formula at 0.025 and 0.975. Endpoints use linear
(type-7) empirical quantiles. A boundary rank, zero jackknife variance, nonfinite
arithmetic, transformation pole, or adjusted probability outside
`[1 / (2000 + 1), 2000 / (2000 + 1)]` is refused. This resolution boundary prevents
claiming an endpoint beyond the frozen resampling budget; it is not a coverage
correction or an endpoint Monte Carlo precision guarantee.

The version-3 artifact binds this method, the exact tie/quantile/jackknife
convention, bias correction, acceleration, adjusted probabilities, every grouped
delete-one record, original rows and identities, and every attempted bootstrap
draw. Its consumer rechecks identification and reruns the complete calculation
under caller-retained identities and bounds. The consumer accepts only the version-3 BCa protocol.

The distinct ignored calibration fixture is
`binary_missingness_whole_row_recovery_bca_l95`, emitting
`cov.recovered_effect.m_graph.frequentist.bootstrap_bca.l95.binary_missingness_whole_row_recovery_bca_l95`.
It preserves the independently generated confounded binary missingness SCM,
structural truth 0.36, sample sizes 1,000/2,000/4,000, original seeds, shared
400/1,000/2,000 repetition protocol, failure denominator and coverage thresholds.
The interval protocol and resampling budget are bound to this record.
All estimator, support, numerical and cancellation failures remain denominator
failures. Ordinary reference tests check independent inverse-response algebra,
all individual delete-one effects, normal quadrature and interval endpoints;
they do not measure coverage.
