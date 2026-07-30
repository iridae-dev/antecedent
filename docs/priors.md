# External prior bank

Transfer previously fit posteriors into a new analysis as power / mixture
priors — without pretending priors create nonparametric identification.

Surveys are the usual motivating domain (product / context tags), but the
library speaks in **sources**, **targets**, and **designs**.

## Workflow

```text
catalog.filter(target) → rank(similarity)
  → map(effect|named params) → power-prior / mixture (α_k, w_k)
  → apply transport policy → PriorSet + assumptions
  → analyze(..., inference=Bayesian(prior_from=…))
  → prior PPC + prior sensitivity (α grid when banked)
```

1. Wrap posterior artifacts with `PriorSourceMeta` / `PriorSource`.
2. `PriorCatalog.compatible_with(...)` → accept / partial / reject reasons.
3. Callers supply **similarity scores**; the library ranks but does not invent
   domain similarity.
4. `compose_external_priors(...)` builds a `ComposedPrior` (optional
   `ConflictPolicy`, `TransportPolicy`).
5. Pass the composed prior as `Bayesian(prior_from=...)`.
6. With `refute="full"`, the ATE path attaches an **α-multiplier** sensitivity
   grid (not isotropic scales) when an external compose is present.

Python example: [`examples/python/prior_bank_surveys.py`](../examples/python/prior_bank_surveys.py)
(Rust: [`examples/rust/prior_bank_surveys.rs`](../examples/rust/prior_bank_surveys.rs)).

## Invariants

- Priors are recorded as `PriorRestriction` assumptions; they **never** upgrade
  `IdentificationStatus`.
- Heterogeneous designs transfer at the **effect-functional** (or explicitly
  mapped parameter) level by default — not silent `coef_i → coef_i`.
  When `Bayesian(prior_from=artifact)` leaves `mapping` unset, hydrate chooses
  identical subspace for matching designs and `EffectFunctional` when layouts
  differ and an effect quantity exists; otherwise an explicit mapping is required.
- Unmapped / incompatible mass goes to a weakly informative baseline; never
  silently renormalized.
- Conflict can only **shrink** external weight (`α → 0`), never invent
  identification or increase α.
- Population / environment shifts require an explicit `TransportPolicy`; missing
  policy → structured `transport_policy_required`. Pass `prior_sources=` (or
  `source_populations=`) with `target_population=` so compose reads
  `tags["population"]` via `populations_from_prior_sources` — callers need not
  thread population tags manually when catalog meta is available.

## What callers must supply

| Concern | Owner |
|---------|--------|
| Similarity / ranking scores | Caller |
| Product / context / taxonomy tags | Caller conventions on `tags` |
| Max-trust / governance of the bank | Caller |
| Compatibility, mapping, α/weights, transport assumptions, diagnostics | Library |

Exact tag keys on `compatible_with(tags=...)` are hard filters. Soft similarity
belongs in `catalog.rank(scores=...)`.

## Effective-sample-size accounting

`α` scales precision, and precision is proportional to sample size for a
Gaussian, so each source's contributed strength can be reported in ESS terms.
Set `ess` on an `ExternalPriorSourceSpec` (Python) / `ExternalPriorSource`
(Rust) to a caller-declared prior-strength sample size (e.g. the source
study's N) to get it back on `ComposedPrior`:

- `effective_ess` — per source, `α_applied · ess` (`None` when that source
  declared no `ess`). Forced to `0` for sources dropped from composition
  (`α == 0` on the power path; `α <= 0` or the mixture weight `<= 0` on the
  mixture path), so this always agrees with the arithmetic that actually ran.
- `composed_ess` — **path-dependent**:
  - **Power path**: precision genuinely adds (`Λ = Λ₀ + Σ αₖΛₖ`), so
    `Σ αₖ · essₖ` over contributing sources is sound — reported only when
    *every* contributing source declared an `ess` (`None` otherwise; a
    partial sum would misrepresent it as complete).
  - **Mixture path**: always `None`. The result is moment-matched and its
    variance includes between-component spread, so the composed prior is
    *weaker* than a precision-sum would imply — summing source ESS would
    overstate composed strength.
- `kish_ess` — Kish (1965) concentration-of-trust diagnostic
  (`(Σw)² / Σw²`) over the weight vector actually used: applied alphas on the
  power path, mixture weights on the mixture path (dropped sources zeroed).

**These three fields, plus MCMC/autocorrelation ESS reported elsewhere on
posterior diagnostics, are three distinct conventions and are never
interchangeable:**

| Convention | Measures | Where |
|------------|----------|-------|
| Prior-strength ESS | Sample size implied by contributed precision | `ComposedPrior.effective_ess` / `.composed_ess` |
| MCMC / autocorrelation ESS | Effectively independent draws in a chain | Posterior / inference diagnostics |
| Kish importance-weighting ESS | Concentration of a trust / importance weight vector | `ComposedPrior.kish_ess`, `TransportAdjustment.kish_ess()` |

## Conjugate moment-matching (Beta / Gamma)

Everything above stays in Gaussian-coefficient terms — `PriorSet` /
`PriorSpec` speak Gaussian coefficients only, and every inference backend in
`antecedent-prob` (conjugate, Laplace, HMC) consumes exactly that shape plus
a residual-variance model. A Gaussian summary cannot itself express a prior
over a **bounded proportion** or a **non-negative rate**, so
`BetaHyperparameters` / `GammaHyperparameters`
(`crates/antecedent-prob/src/conjugate_moment_match.rs`, Python
`antecedent.priors.beta_from_moments` /
`antecedent.priors.gamma_from_moments` and their `*_from_mean_and_ess`
siblings) convert a Gaussian-shaped summary into the matching conjugate
family. Each family exposes **two constructors with distinct contracts** —
reach for the one that matches what you actually know:

- **`from_moments(mean, variance)`** matches both moments exactly. Beta:
  total concentration `κ = mean·(1−mean)/var − 1` from the moments, then
  `α = mean·κ`, `β = (1−mean)·κ`. Gamma: `shape = mean²/var`, `rate =
  mean/var`. Prior strength (`.ess()`) is whatever those moments imply — a
  derived consequence, not something you request. Use this when you have a
  genuine `(mean, variance)` summary (e.g. from a composed prior's moments
  or a domain expert's elicited belief) and want the closest conjugate fit,
  prior strength included.
- **`from_mean_and_ess(mean, ess)`** matches the mean and a caller-declared
  prior-strength `ess` exactly. Beta: `α = mean·(ess+2)`, `β =
  (1−mean)·(ess+2)`. Gamma: `shape = ess+1`, `rate = shape/mean`. There is
  no `variance` parameter — `mean` and `ess` alone determine every other
  moment, so a `variance` argument would have nothing to do. Use this when
  what you actually know is a target mean and how much you want the prior
  to weigh (in sample-size terms), not a variance.

An earlier version of this module offered a single
`from_moments(mean, variance, target_ess)`: moment-match to `(mean,
variance)`, then discard that match and rescale to `target_ess` instead.
`variance` never affected the output under that signature — it was checked
for validity and then thrown away — which both misnamed the function (it
built from `(mean, target_ess)`, not from moments) and made the Beta variant
reject satisfiable requests, since the variance support check ran against a
value the rescale would immediately discard. The two-constructor split
above replaces that signature; there is no `target_ess` parameter anywhere
in the current API.

These are **standalone converters, not `PriorSpec` variants** — no backend
in this crate can consume a Beta or Gamma prior today (all four take a
Gaussian coefficient design-matrix prior), so adding a `PriorSpec::Beta` /
`::Gamma` would create a type the library accepts but nothing can use.
Callers get plain hyperparameter structs to hand to their own conjugate
update (Beta-Binomial, Gamma-Poisson) or record as an assumption.

Out-of-support input is **rejected, never silently clamped**: `from_moments`
requires `mean` strictly inside `(0, 1)` for Beta (`mean > 0` for Gamma),
and `var < mean·(1−mean)` for Beta (`var > 0` for Gamma, no upper bound —
unlike Beta, any positive variance is achievable at a given positive mean
via some shape); the variance comparison has no epsilon slack at the
boundary. `from_mean_and_ess` shares the same `mean` domain check but has
**no variance-derived gate to violate** — every `(mean, ess >= 0)` request
is satisfiable by construction — and rejects negative `ess`.
`from_mean_and_ess(mean, 0.0)` degrades to the reference-strength prior at
the requested mean — `Beta(1, 1)`-equivalent strength, or `Gamma(shape=1,
·)` — never something vanishing or improper.

**ESS convention**: `ess = α + β − 2` for Beta and `ess = shape − 1` for
Gamma — chosen so each family's flat/reference prior maps to `ess = 0`,
which is exactly why `from_mean_and_ess(mean, 0.0)` degrades to that
reference rather than to a degenerate prior. This is the same
**prior-strength ESS** notion in the table above (how much evidence a
prior's concentration is worth, in sample-size terms), applied to two
conjugate families instead of a Gaussian coefficient's precision — still
not interchangeable with MCMC ESS or Kish ESS. Some references instead
report `α + β` (total pseudo-count; `Beta(1,1)` would be `ess = 2`) or
`shape` directly (`Gamma(1, ·)` would be `ess = 1`); this module never uses
those conventions.

`from_moments` can report a **negative** `.ess()`: any `(mean, variance)`
match weaker than the flat/reference prior (Beta `κ < 2`; Gamma `shape <
1`) yields `α + β − 2 < 0` or `shape − 1 < 0` while `α`/`β` (or
`shape`/`rate`) stay positive and proper. That is a truthful report that
the supplied moments describe a prior weaker than the reference, not an
error — distinct from `from_mean_and_ess` rejecting a negative `ess`
*input*, since a caller cannot request negative prior strength even though
a moment match can honestly report it.

See `conformance/bayesian/prior_conjugate_moment_match/` for pinned analytic
scenarios covering both constructors (moment round trip, a moment match
with negative `ess`, `from_mean_and_ess(mean, 0.0)`, and each rejected
input).

## Supported transport policies

Documented in parity `bayes.prior_bank.transport`:

- `InvariantConditionalOutcome` — `P(Y|do(T),X)` stable across populations
- `InvariantEffectModifiers`
- `InvariantPropensity` (without transport weights → α forced to 0)

Unsupported environment / unidentified shifts are **not claimed**.

## Gates

```bash
bash scripts/gate_bayesian.sh
```

Runs `conformance/bayesian/prior_bank_*` plus `python/tests/test_prior_bank.py`.
