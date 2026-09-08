# Independent posterior moment reference

**Suite path:** `conformance/bayesian/release_12_moments`

`generate.py` imports NumPy only. It freezes a 96-row Gaussian SCM and computes
Normal-inverse-gamma posterior means/covariances from matrix equations:
V = (X'X + I / scale²)^-1, m = V X'y, a = .001+n/2,
b = .001 + (y'y - m'V^-1m)/2, Cov(beta) = b V/(a-1).

The mediation reference uses independent-product moments, including the product
of variances. The sustained reference projects the *sum* of both lag coefficients
with their covariance. Both informative scale .1 and weak scale 10 are pinned.
This is an independent equation oracle, not external-package behavioral parity.
Tests compare actual retained posterior draws to these moments and round-trip
those draws through Rust/Python artifact encoders. Monte Carlo tolerances are
relative to posterior SD, not arbitrary tolerances around the point estimate.

Additional cases pin baseline-confounder adjustment in both mediation equations
and a repeated stationary mechanism: a single coefficient a multiplies b1+b2,
so its posterior draw is shared across both time copies. Each mechanism uses
its unique complete observed rows; overlapping windows do not duplicate data.

## Expected summary

Top-level keys: `data, posterior` (2 fields).
