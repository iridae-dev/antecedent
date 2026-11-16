# conjugate_count_posteriors

**Suite path:** `conformance/bayesian/conjugate_count_posteriors`

Closed-form posteriors for two count queries within one cell, and the power-prior
discount chosen from how well a source predicts the target.

## Scenarios

- **beta_binomial_posterior**: `Beta(a, b)` prior and `k` successes in `n` trials
  give `Beta(a + k, b + n - k)`; the marginal likelihood of the count is
  `C(n, k) B(a + k, b + n - k) / B(a, b)`.
- **exceedance**: the exceedance probability `P(Y > threshold)` is a binomial
  proportion whose successes are the outcomes strictly above the threshold;
  an outcome equal to the threshold is not a success.
- **gamma_poisson_posterior**: `Gamma(shape a, rate b)` prior and counts `y_i` over
  unit exposures give `Gamma(a + Σy, b + n)` and the Poisson marginal likelihood.
- **empirical_bayes_beta**: the power prior at discount `α` is
  `Beta(a + α k0, b + α (n0 − k0))`; the score of `α` is the log marginal
  likelihood of the target count under it, and the empirical-Bayes discount is the
  grid point with the largest score (the smallest on a tie).

`PriorSpec::Beta` and `PriorSpec::Gamma` carry these priors. A Gaussian regression
fit refuses a prior set that carries one rather than ignoring it, and a prior never
upgrades identification.

Multiprecision quantile, CDF and predictive references, the selection-aware
interval and the linear-target marginal likelihood are checked in
`crates/antecedent-prob/tests/conjugate_counts_oracle.rs` against
`crates/antecedent-prob/tests/oracle_data/conjugate_counts_reference.py`.

## Expected summary

Top-level keys: `beta_binomial_posterior, empirical_bayes_beta, exceedance, gamma_poisson_posterior, tol` (5 fields).
