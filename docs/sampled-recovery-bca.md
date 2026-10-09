# Whole-row sampled recovery: BCa candidate protocol

The original 500-draw percentile procedure did not earn a coverage license: its
2,000-repetition recheck observed 0.937 coverage. The archived records and the
version-2 percentile artifact retain that method identity. They are not evidence
for the new procedure described here.

The new `bootstrap_bca` candidate uses 2,000 whole-row bootstrap draws and the
bias-corrected and accelerated transformation of
[Efron (1987), *Better Bootstrap Confidence Intervals*](https://doi.org/10.1080/01621459.1987.10478410).
Asymptotic BCa regularity does not establish finite-sample coverage for this
missingness estimator. Normal release production remains closed until the new
procedure independently passes its frozen calibration and lifecycle gates.

Each draw reruns the original licensed observation-recovery formula and downstream
standardized recovered effect. The candidate refuses any failed bootstrap draw;
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
under caller-retained identities and bounds. Version-2 percentile artifacts remain
replayable with their original digests and scientific meaning.

The distinct ignored calibration fixture is
`binary_missingness_whole_row_recovery_bca_l95`, emitting
`cov.recovered_effect.m_graph.frequentist.bootstrap_bca.l95.binary_missingness_whole_row_recovery_bca_l95`.
It preserves the independently generated confounded binary missingness SCM,
structural truth 0.36, sample sizes 1,000/2,000/4,000, original seeds, shared
400/1,000/2,000 repetition protocol, failure denominator and coverage thresholds.
Only the explicitly distinct interval method and its resampling budget change.
All estimator, support, numerical and cancellation failures remain denominator
failures. Ordinary reference tests check independent inverse-response algebra,
all individual delete-one effects, normal quadrature and interval endpoints;
they do not measure coverage.
