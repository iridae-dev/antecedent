# Laplace GLM baselines

Owner: `antecedent-prob` / `fit_laplace_glm`

## Criteria

- Repeated Laplace fits of the same shape must reuse `LaplaceWorkspace` scratch: after a
  warm `ws.prepare(n, p + 1, n_draws)` (joint `(β, λ)` state under the default InvGamma
  residual model), `ws.grow_count` must stay flat across every subsequent fit (asserted in
  the Criterion bench after the timed loop).
- Bench target: `laplace_gaussian_n500_p3` — n=500, p=3 (intercept + linear + quadratic
  column), `BayesLikelihood::GaussianIdentity`, isotropic Gaussian coefficient prior
  (variance 10), 256 posterior draws, `max_iter=40`, `fit_laplace_glm` end to end
  (MAP optimization + Laplace draws).

## Measured mean

- `laplace_gaussian_n500_p3`: **21.22 µs** (Criterion mean, CI 21.198–21.245 µs).
- Established: 2026-08-18. Machine class: Apple M1 Max.

## How to refresh

```bash
cargo bench -p antecedent-prob --bench laplace_glm
```

Record the Criterion mean and update this file with the run date and machine class. The
bench aborts (assert) if the workspace grows across fits — a refresh that trips the
assert is a reuse regression, not a new baseline.
