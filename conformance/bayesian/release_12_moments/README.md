# Independent posterior moment reference

`generate.py` imports NumPy only. It freezes a 96-row Gaussian SCM and computes
Normal-inverse-gamma posterior means/covariances from matrix equations. With the
likelihood tempered by 1/κ (κ = 1 is the ordinary posterior; prior
β | σ² ~ N(0, σ² scale² I), σ² ~ InvGamma(.001, .001)):
P = X'X/κ + I / scale², V = P^-1, m = V X'y/κ, a = .001 + n/(2κ),
b = .001 + (y'y/κ - m'Pm)/2, Cov(beta) = b V/(a-1).
This is every row weighted 1/κ, an effective sample of n/κ rows; the prior keeps
its full weight.

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

## Serial-dependence tempering (`window`, `stationary_window`)

The Bayesian sustained contrasts on a `TemporalDag` are generalized posteriors:
each mechanism's likelihood on its time-ordered rows is tempered by its own κ̂,
computed for the contrast's direction c in that mechanism (`window`: c = (0, 1, 1),
the sum of both lags; `stationary_window`: c ∝ e_a in the m mechanism and
c ∝ (0, 1, 1) in the y mechanism, the gradient of a(b1 + b2); κ̂ is invariant to
the scale and sign of c). With OLS residuals ê and row weights w = X(X'X)^-1 c:

- ρ̂(x) = Σ x_t x_{t-1} / Σ x_{t-1}² + (1 + 3ρ̂)/n (Kendall bias correction),
  clamped to ±.97.
- κ̂_HAC: the score s = w ⊙ ê is prewhitened, u_t = s_t - ρ̂(s) s_{t-1}; its
  Bartlett long-run variance with bandwidth M = ⌊4(n/100)^{2/9}⌋ (weights
  1 - k/(M+1), uncentred autocovariances divided by n-1) is recoloured by
  1/(1 - ρ̂)² and divided by Σ s²/n.
- f_b = cv(b)/1.96 with cv(b) = 1.96 + 2.9694b + .4160b² - .5324b³ and
  b = (M + 1)/n (Kiefer–Vogelsang Bartlett fixed-b 95% critical value).
- κ̂_AR = w'Rw / w'w with R_ts = ρ̂(ê)^|t-s| (AR(1) residual variance ratio
  given the design).
- κ̂ = clamp(max(κ̂_HAC f_b², κ̂_AR), 1, max(n/(p+2), 1)).

`tempering` in `expected.json` pins κ̂ and its components per fitted design,
keyed by design rows; the test reads the reported κ̂ from the
`bayes.temporal.long_run_tempering` assumption of each fit. The stationary
reference keeps the independent-product moments of a and b1 + b2 from their
separately tempered posteriors.

## Modifier-mean uncertainty (`conditional`)

The design is (1, t, w, t(w - w̄)). Each ConditionalEffect draw is
b_t + b_tx (w̄_D - w̄), with w̄_D = Σ ω_i w_i and ω ~ Dirichlet(1, …, 1)
(Bayesian bootstrap of the modifier mean) drawn independently of β. So
E = E[b_t] and Var = Var(b_t) + (Var(b_tx) + E[b_tx]²) Var(w̄_D), with
Var(w̄_D) = Σ(w_i - w̄)² / (n(n + 1)); the cross term vanishes because
E[w̄_D - w̄] = 0.
