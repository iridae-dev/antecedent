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

## Serial-dependence tempering (`window`, `stationary_window`, `dependent_pulse`)

The Bayesian temporal contrasts on a `TemporalDag` are generalized posteriors:
each mechanism's likelihood on its time-ordered rows is tempered by its own κ̂,
computed for the contrast's direction c in that mechanism (`window`: c = (0, 1, 1),
the sum of both lags; `stationary_window`: c ∝ e_a in the m mechanism and
c ∝ (0, 1, 1) in the y mechanism, the gradient of a(b1 + b2); κ̂ is invariant to
the scale and sign of c). With OLS residuals ê, row weights w = X(X'X)^-1 c and
H = X(X'X)^-1 X':

- AR(q) residual model by REML. For partial autocorrelations r (|r| ≤ .97,
  coordinates z with r = .97 tanh z, AR coefficients by the Levinson step-up)
  and Γ the stationary autocovariance with unit innovations, the profile REML
  log-likelihood is ℓ_R = -½ log|Γ| - ½ log|X'Γ^-1 X| - ½ (n - p) log(RSS_GLS/(n - p)).
  Each order q = 1..4 is maximized by Nelder–Mead (reflect 1, expand 2,
  contract ½, shrink ½; initial simplex z_0 + .2 e_i warm-started from the
  previous order; stop when the objective range is ≤ 1e-12 and the simplex
  extent ≤ 1e-8) and the order minimizing -2ℓ_R + q ln(n - p) is kept (q = 0
  allowed). R is the correlation matrix of the fitted AR(q).
- κ̂_R = w'Rw / w'w: the variance ratio of c'β̂ given the realized design.
- Residual-scale factor n / max(n - tr(HR), p + 2): the OLS residual sum of
  squares loses tr(HR) rows of scale to the projection (p when iid), and the
  tempered posterior estimates σ² from it as if n/κ rows were independent.
- τ̂ = delta-method SD of log(κ̂_R · scale factor) in the z coordinates against the
  observed REML information (central differences, step 1e-3; symmetrized
  Hessian; 0 for q = 0 or a non-positive-definite information; capped at 1);
  the factor is multiplied by exp(τ̂²/2), the mean of a lognormal κ with that
  spread.
- Bound: three times the prewhitened Newey–West long-run-variance ratio of the
  score s = w ⊙ ê times the squared fixed-b factor. ρ̂(x) = Σ x_t x_{t-1} /
  Σ x_{t-1}² + (1 + 3ρ̂)/n (Kendall bias correction, clamped to ±.97);
  u_t = s_t - ρ̂(s) s_{t-1}; Bartlett long-run variance with bandwidth
  M = ⌊4(n/100)^{2/9}⌋ (weights 1 - k/(M+1), uncentred autocovariances divided
  by the series length), recoloured by 1/(1 - ρ̂)² and divided by Σ s²/n; when a
  Yule–Walker AR(q) (uncentred autocovariances, q ≤ 4 by BIC n ln σ̂²_q + q ln n)
  of the score has order q ≥ 2 the larger of that and the AR(q)-prewhitened ratio
  recoloured by 1/max(1 - Σφ̂, .03)² is used. f_b = cv(b)/1.96 with
  cv(b) = 1.96 + 2.9694b + .4160b² - .5324b³ and b = (M + 1)/n (Kiefer–Vogelsang
  Bartlett fixed-b 95% critical value).
- κ̂ = clamp(min(κ̂_R · n/(n - tr(HR)) · exp(τ̂²/2), 3 κ̂_HAC f_b²), 1, max(n/(p+2), 1)).
  An exact fit (RSS ≤ 1e-20 of the centred outcome sum of squares) keeps κ̂ = 1.

`tempering` in `expected.json` pins κ̂ and its components per fitted design,
keyed by design rows (`residual_ar_order` is the REML order, `df_loss` = tr(HR),
`kappa_log_sd` = τ̂, `score_ar_order` the prewhitening order of the bound,
`bounded` whether the bound held); the tests read the reported κ̂ from the
`bayes.temporal.long_run_tempering` assumption of each fit. On the `window`,
`stationary_window` and mediation fixtures the residuals are iid, so κ̂ is the
residual-scale factor n/(n - p) (one mechanism selects q = 1 with a small τ̂).
`dependent_pulse` is an AR(2)(0.3, 0.5) treatment and residual on the same 96
rows (`td`, `yd`; y = 0.8 t_{t-1} + e): REML selects q = 2, tr(HR) ≈ 21 of 95
rows and τ̂ ≈ 0.38, and κ̂ ≈ 5.8 against ≈ 2.6 for the kernel ratio; the Rust
test `v19_bayesian_temporal::release_12_dependent_pulse_moments_and_tempering_match_the_reference`
checks the Pulse posterior moments and κ̂ to 1e-5. The stationary reference keeps
the independent-product moments of a and b1 + b2 from their separately tempered
posteriors.

The temporal mediation references (`mediation`, `confounded_mediation`) temper
both mechanisms too: the mediator mechanism along its path slope a (c = e_1),
the outcome mechanism at the largest κ̂ over its direct c' (e_1), mediated b
(e_2) and total c' + â b (e_1 + â e_2) directions, with â the OLS mediator
slope. `tempering.mediation` / `tempering.confounded_mediation` list the two
mechanisms (`m`, `y`; both on the same rows). The Rust test
`v19_bayesian_temporal::release_12_mediation_moments_and_tempering_match_the_reference`
checks the moments and both κ̂.

## Modifier-mean uncertainty (`conditional`)

The design is (1, t, w, t(w - w̄)). Each ConditionalEffect draw is
b_t + b_tx (w̄_D - w̄), with w̄_D = Σ ω_i w_i and ω ~ Dirichlet(1, …, 1)
(Bayesian bootstrap of the modifier mean) drawn independently of β. So
E = E[b_t] and Var = Var(b_t) + (Var(b_tx) + E[b_tx]²) Var(w̄_D), with
Var(w̄_D) = Σ(w_i - w̄)² / (n(n + 1)); the cross term vanishes because
E[w̄_D - w̄] = 0.
