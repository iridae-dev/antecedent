# Checked Bayesian retention

The oracle contains closed-form algebra, independently calculated from the declared design matrices. It does not contain frozen native draws. Tests also reconstruct the algebra and compare sampled posterior summaries with Monte Carlo tolerances.

Gaussian: 64 repetitions of treatment 0/1 crossed with noise ±0.4 give 256 rows, `Y=1+2T+noise`. Coefficient precision is `X'X+0.01I`, coefficient prior mean is zero, and known noise variance is one. The effect is the treatment coefficient.

Quadratic basis: 20 repetitions of treatment 0/1, covariate −1/0/1, and noise ±0.2 give 240 rows, `Y=1+2T+0.5Z+0.4TZ+0.3Z²+noise`. The five columns are `[1,T,Z,TZ,Z²]`. The prior precision is `0.01I`; inverse-gamma shape and scale start at 0.001. Posterior coefficient covariance is `scale/(shape−1)` times the inverse precision. Balanced target covariates make the mean effect the treatment coefficient.

Joint internal candidate: two independent 128-row sources use the Gaussian design. The invariant treatment coefficient has prior precision 0.01; the varying baseline intercept has prior precision 0.25. The coefficient order is `[T,1]`, precision is `[[128.01,128],[128,256.25]]`, and moment is `[384,512]`. This is inspection evidence for the internal candidate, not calibrated public inference.

Prior transfer: the source structural effect is two and the independent target structural effect is four. The source envelope carries concrete data, graph, producing settings, and posterior bytes. Fresh consumption replays source production before target fitting. Tests refuse known identical full source/target snapshots and stale posterior bytes. Partial sample overlap is not inferred from distinct full snapshots.
