# Posterior functional evaluation baselines

Owner: `antecedent-estimate` / `GCompAteEvaluator::evaluate_batch`

## Criteria

- Batched posterior-functional evaluation must reuse `PosteriorEvalWorkspace` scratch:
  after a warm `ws.prepare(n_draws, ncols)`, `ws.grow_count` must stay flat across every
  subsequent `evaluate_batch` call over the same draw batch (asserted in the Criterion
  bench after the timed loop). `EffectBatch` output is likewise `prepare`d once and
  rewritten in place.
- Bench target: `posterior_gcomp_eval_n400_d512` — n=400 rows, 3 design columns
  (intercept, binary treatment, linear covariate), 512 posterior coefficient draws
  (`PosteriorDraws::from_column_major`), `GlmFamily::GaussianIdentity` g-computation ATE
  via the compiled evaluator (`GCompAteEvaluator::compile` once, then
  `evaluate_batch` per iteration over the full 512-draw batch).

## Measured mean

- `posterior_gcomp_eval_n400_d512`: **1.569 ms** (Criterion mean, CI 1.5664–1.5710 ms).
- Established: 2026-08-18. Machine class: Apple M1 Max.

## How to refresh

```bash
cargo bench -p antecedent-estimate --bench posterior_functional
```

Record the Criterion mean and update this file with the run date and machine class. The
bench aborts (assert) if the workspace grows across evaluations — a refresh that trips
the assert is a reuse regression, not a new baseline.
