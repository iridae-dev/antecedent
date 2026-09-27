# Local-polynomial fuzzy RD and regression kink

`antecedent.quasi.FuzzyRegressionDiscontinuity` and
`RegressionKink` use native local-quadratic fits with triangular kernel
weights on either side of a declared cutoff. Fuzzy RD reports the ratio of
the outcome intercept jump to the treatment intercept jump. Regression kink
reports the ratio of the outcome slope change to the treatment slope change.
The functions disclose the chosen bandwidth, local row counts, reduced-form
change, and first-stage change, and refuse singular local fits, fewer than
four positive-weight rows on either side, and numerically zero first stages.

The point estimate uses a local-quadratic fit (order 2) corrected with the
estimated cubic term (order 3) from the same bandwidth. The variance uses the
local-cubic HC0 sandwich covariance, including outcome/treatment covariance,
and a delta method for the fuzzy ratio. A normal 95% interval is returned.
The implementation refuses fewer than five positive-kernel observations on
either side, singular local fits, and a first-stage change whose HC0 95%
interval includes zero. It does not provide weak-instrument-robust inference.

The result remains `unlicensed_point_utility`; its interval is labeled
`local_quadratic_rbc_hc0_delta_normal_unvalidated`. A seeded known-truth
simulation fixture checks nominal coverage for one smooth polynomial DGP. That
fixture does not calibrate arbitrary bandwidth choices, distributions,
dependence structures, or running-variable designs, so no general interval or
support-matrix claim is made. Counts indicate local sample size only; they do
not establish continuity, absence of manipulation, exclusion, monotonicity, or
adequate effective support.

The fuzzy RD interpretation assumes smooth potential outcome regressions,
no precise manipulation of the running variable, exclusion, monotonicity for
the local-complier effect, and no interference. The kink interpretation
requires smooth potential outcome derivatives apart from the threshold
induced treatment-slope change, with the corresponding exclusion and
monotonicity assumptions. Neither utility performs density or covariate
continuity tests.

`analyze(data, query=FuzzyRegressionDiscontinuity(...))` and the matching
`RegressionKink` query now use this same Rust kernel through retained
`PreparedAnalysis.prepare` / `estimate`. The prepared query fixes the cutoff,
bandwidth, variable names, and jump-versus-kink contrast. The result's
`local_polynomial_ratio` section reports the point estimate, reduced form,
first stage, local sample counts, and descriptive HC0 standard error. The
retained route does not publish a confidence interval: `ci_lower` and
`ci_upper` are `None`, and its artifact rejects fabricated interval fields.
The result remains `unlicensed_point_utility` and off the support-matrix axis.
