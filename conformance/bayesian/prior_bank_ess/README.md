# prior_bank_ess

Effective-sample-size (ESS) accounting on `compose_external_priors`, separate
from `prior_bank_power_mixture` (which pins the composed Gaussian moments
without any `ess` declared).

`α` scales precision, and precision is proportional to sample size for a
Gaussian, so a source's own contributed strength (`α · ess`) and — on the
power path only — the composed prior's total strength can be reported in
effective-sample-size terms when the caller declares each source's `ess`.

Four scenarios, each analytic:

- **power**: one power-path source declares `ess`. `effective_ess = α·ess`;
  with a single contributing source, `composed_ess` equals it exactly and
  `kish_ess = 1` (one active weight is maximally concentrated).
- **power_partial_coverage**: two power-path sources both contribute
  (`α > 0`), but one declares no `ess`. `composed_ess` must be `None` — a sum
  over only the source that *does* declare `ess` would misrepresent it as the
  composed total — even though that source's own `effective_ess` is still
  reported.
- **power_dropped_source**: a second source is dropped (`α = 0`) despite
  declaring a large `ess`. Its `effective_ess` must be `0`, and it must not
  block `composed_ess` for the surviving source (a dropped source never
  contributed evidence, so a missing/irrelevant `ess` on it cannot poison the
  sum).
- **mixture**: a single mixture-path source declares `ess`. `composed_ess`
  must always be `None` on this path — the moment-matched result folds in
  between-component spread (`second − μ²`), so it is *weaker* than a
  precision-sum would imply, and summing source ESS would overstate composed
  strength — while the source's own `effective_ess` is still reported.

`kish_ess` in every scenario is the Kish (1965) concentration-of-trust
diagnostic over the weight vector actually used in composition (applied
alphas on the power path, mixture weights on the mixture path) — a distinct
convention from the prior-strength ESS fields above, and from
MCMC/autocorrelation ESS reported elsewhere on posterior diagnostics.
