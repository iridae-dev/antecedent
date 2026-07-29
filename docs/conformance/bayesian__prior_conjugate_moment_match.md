# prior_conjugate_moment_match

**Suite path:** `conformance/bayesian/prior_conjugate_moment_match`

Two ways to build `Beta(alpha, beta)` or `Gamma(shape, rate)` conjugate
hyperparameters from a Gaussian-shaped summary — separate from
`prior_bank_ess` (which reports ESS for composed *Gaussian* coefficient
priors, not conjugate bounded/rate families).

## Why this exists

`PriorSpec` (`crates/antecedent-prob/src/prior.rs`) offers Gaussian
coefficient priors only. Nothing converts a Gaussian-shaped summary into the
hyperparameters a Beta (bounded proportion) or Gamma (non-negative rate)
prior would need. `BetaHyperparameters` / `GammaHyperparameters`
(`crates/antecedent-prob/src/conjugate_moment_match.rs`) fill that gap as
**standalone converters** — no inference backend in this crate consumes a
Beta or Gamma prior today (all four backends take a
`GaussianCoefficientPrior` design-matrix prior plus a residual-variance
model), so this is deliberately not a `PriorSpec` variant.

## Two contracts, not one

Each family exposes two constructors with distinct, honest contracts. An
earlier version of this module offered a single
`from_moments(mean, variance, target_ess)`: moment-match to `(mean,
variance)`, then discard that match and rescale to `target_ess` instead.
`variance` never affected the output under that signature, which both
misnamed the function (it built from `(mean, target_ess)`, not from
moments) and made the Beta variant reject satisfiable requests — e.g.
`(mean=0.5, variance=0.3, target_ess=10.0)` errored on the variance support
check even though the value actually returned, `Beta(6, 6)`, has no
relationship to `0.3`. The split below replaces it:

- **`from_moments(mean, variance)`** matches both moments exactly. Prior
  strength (`.ess()`) is whatever those moments imply — a derived
  consequence, not a request. It can be **negative** (a proper prior weaker
  than the flat/reference prior); that is a truthful report, not an error.
- **`from_mean_and_ess(mean, ess)`** matches the mean and a caller-declared
  prior-strength `ess` exactly. There is no `variance` parameter — `mean`
  and `ess` alone determine every other moment, so a `variance` argument
  would have nothing to do. Every `(mean, ess >= 0)` request is
  satisfiable: unlike `from_moments`, there is no support gate to violate.

## Scenarios

- **beta_moment_match**: an ordinary `(mean, variance)` pair inside the
  Beta support bound; the output's `mean()` / `variance()` must reproduce
  the exact input to tolerance.
- **beta_moment_match_negative_ess**: `(mean, variance)` still inside the
  support bound but weaker than the flat reference `Beta(1, 1)` (total
  concentration `< 2`); `alpha`/`beta` stay positive and proper while
  `ess()` is negative.
- **beta_mean_and_ess_zero**: `from_mean_and_ess(mean, 0.0)` must degrade
  to the same strength as the flat reference prior `Beta(1, 1)` at the
  requested mean — proper (`alpha > 0`, `beta > 0`), never vanishing.
- **beta_mean_and_ess_matches_any_request**: an ordinary `(mean, ess)` pair
  round-trips cleanly through `from_mean_and_ess`; unlike `from_moments`,
  there is no variance-derived support gate this constructor could ever
  fail against.
- **beta_moment_match_rejected_inputs**: `variance` at or above the support
  bound `mean*(1-mean)` (no epsilon slack — the bound is exact) and `mean`
  outside the open interval `(0, 1)` are each rejected rather than clamped.
- **beta_mean_and_ess_rejected_inputs**: `mean` outside `(0, 1)` and
  negative `ess` are each rejected.
- **gamma_moment_match** / **gamma_moment_match_negative_ess** /
  **gamma_mean_and_ess_zero** / **gamma_mean_and_ess_matches_any_request** /
  **gamma_moment_match_rejected_inputs** / **gamma_mean_and_ess_rejected_inputs**:
  the same six shapes for the Gamma family, whose only support requirement
  on `from_moments` is `mean > 0` and `variance > 0` (no upper bound on
  `variance` the way Beta has one). `from_mean_and_ess(mean, 0.0)` degrades
  to `Gamma(shape=1, ·)` — the reference exponential prior.

## ESS convention

`ess = alpha + beta - 2` for Beta and `ess = shape - 1` for Gamma — chosen
so each family's flat/reference prior (`Beta(1,1)`, `Gamma(shape=1,*)`) maps
to `ess = 0`. This is the same **prior-strength ESS** notion documented for
composed Gaussian priors in `docs/priors.md` and
`crates/antecedent-prob/src/external_prior.rs`, applied here to two
conjugate families instead of a Gaussian coefficient's precision. It is not
interchangeable with MCMC/autocorrelation ESS or with Kish
importance-weighting ESS, and it is not the only convention in the
literature (some sources report `alpha + beta` or `shape` directly, under
which the reference priors are `ess = 2` / `ess = 1` instead of `0`).

`from_moments` can report a negative `ess()`; `from_mean_and_ess` rejects a
negative `ess` *input* — a distinct check on a distinct role for the same
quantity (requested vs. read back).

## Expected summary

Top-level keys: `beta_mean_and_ess_matches_any_request, beta_mean_and_ess_rejected_inputs, beta_mean_and_ess_zero, beta_moment_match, beta_moment_match_negative_ess, beta_moment_match_rejected_inputs, gamma_mean_and_ess_matches_any_request, gamma_mean_and_ess_rejected_inputs, gamma_mean_and_ess_zero, gamma_moment_match, gamma_moment_match_negative_ess, gamma_moment_match_rejected_inputs, notes, tol` (14 fields).
