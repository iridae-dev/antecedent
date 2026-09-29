# Rank diagnostics numeric oracle

`diagnostics_oracle.json` freezes three shared four-chain, 128-draw arrays and
the rank R-hat, bulk ESS, and tail ESS computed with ArviZ 0.20.0, NumPy 2.1.3,
and SciPy 1.14.1. The `samples_chain_major` values are rounded to 12 decimal
places before writing; the Rust test consumes those exact values.

The arrays come from `numpy.random.default_rng(20210924).normal(size=(4, 128))`.
`shifted_chain` adds 1.25 to chain zero. `autocorrelated` applies
`x[c,0] = base[c,0]` and `x[c,d] = 0.8*x[c,d-1] + base[c,d]`.

The oracle values use `arviz.rhat(array, method="rank")`,
`arviz.ess(array, method="bulk")`, and `arviz.ess(array, method="tail")`.
`crates/antecedent-prob/tests/mcmc_arviz_oracle.rs` compares the implementation
with the frozen values on the same arrays. The separate `expected.json` is an
older threshold-only diagnostic routing contract and has no chain samples.
