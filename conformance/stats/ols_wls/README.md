# OLS/WLS black-box reference

Run `uv run --frozen --script conformance/stats/ols_wls/reference.py --check`.
The retained PEP 723 script and lock isolate Python 3.12, statsmodels 0.14.4,
NumPy 2.1.3, SciPy 1.14.1 and pandas 2.2.3 from Antecedent's runtime. The
existing six frozen outputs reproduced on 27 October 2026. Run `--self-test`
instead of `--check` to verify altered coefficients, widened tolerance and
empty-case controls fail.

The same locked harness accepts `--fixture sandwich_covariance`, `--fixture
glm_irls` and `--fixture ols_wls_grid`. These check eleven sandwich covariances,
five GLM families and twelve continuous/categorical OLS/WLS scenarios. The new
grid includes unequal and zero-row weights, two sample sizes and unweighted
HC3 covariance comparisons. `--write-grid` regenerates that grid from the
declared deterministic designs, not from Antecedent output.

The oversized HAC scenario records both statsmodels' raw requested-lag result
and the result at Antecedent's effective lag `min(requested, n-1)`. The Rust
assertion checks the matched result and rejects equality with the unmatched
raw result. `--write-capped-lag` regenerates that additional comparison. This
documents an intentional bandwidth convention rather than changing either
implementation's answer to force agreement.

Inputs are the repository's frozen design matrices, responses and weights.
Only those inputs feed the public
[OLS](https://www.statsmodels.org/v0.14.4/generated/statsmodels.regression.linear_model.OLS.html)
and [WLS](https://www.statsmodels.org/v0.14.4/generated/statsmodels.regression.linear_model.WLS.html)
APIs. No intercept is added automatically. This harness imports no Antecedent
implementation or upstream source/tests; the shared comparison helper computes
no reference values. Outputs are numerical results of black-box execution on
repository inputs; no upstream implementation material is redistributed.

`least_squares_matches_frozen_statsmodels_outputs` in
`crates/antecedent-stats/tests/foundations_oracle.rs` exercises these cases.
Ordinary OLS/WLS, omitted intercept, rescaling/permutation and an ill-conditioned
design are checked. The ill-conditioned product check compares fitted values
at its stated tolerance; it does not promise matching unstable coefficients.
The rank-deficient comparator returns a pseudoinverse solution while Antecedent
refuses; this intentional difference is tested as a contract boundary.

The replaced generation command remains in `historical_reference`; retaining
it does not make the deleted historical harness reproducible. The replacement
reproduces the numerical values, not the original environment's metadata hashes.
These fixed-input comparisons establish no repeated-sampling coverage claim.
