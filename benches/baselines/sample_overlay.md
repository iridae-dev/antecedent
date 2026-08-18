# GCM interventional sampling baselines

Owner: `antecedent-model` / `sample_interventional`

## Criteria

- Repeated interventional draws of the same shape must reuse `MechanismWorkspace`
  scratch: after a warm sample, `ws.grow_count` must stay flat across further
  samples (asserted in the Criterion bench after the timed loop).
- Bench target: `sample_interventional_n1000_overlay` — fitted two-node linear
  Gaussian, hard `do(X := 1)`, n=1000 draws, workspace hoisted across Criterion
  iterations.

## Measured mean

- `sample_interventional_n1000_overlay`: **10.14 µs** (Criterion mean, CI 10.056–10.232 µs).
- Established: 2026-08-18. Machine class: Apple M1 Max.

## How to refresh

```bash
cargo bench -p antecedent-model --bench sample_overlay
```

Record the Criterion mean and update this file with the run date and machine class. The
bench aborts (assert) if the workspace grows across samples — a refresh that trips the
assert is a reuse regression, not a new baseline.

Gate: mean ≤ **12.16 µs** (20% over 10.14 µs).
