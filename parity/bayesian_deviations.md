# Bayesian deviations

Intentional waivers relative to `parity/bayesian.toml`. The PCM/SCM mechanism
registry and do-samplers are `done` under gcm / bayesian inventory rows.

## 1. Bayesian DAG / DBN posterior search

DESIGN.md §13.7 graph posterior search is out of scope. The library ships
graph-weighted **effect envelopes** over supplied `WeightedGraphSamples` only.
Tracked as `intentional_deviation` on `bayes.discovery.dag_posterior`.

## 2. Hierarchical / BVAR / state-space / GP mechanisms

Listed in §14.4 as optional after the base backend is stable. Tracked as
`intentional_deviation` on `bayes.backend.hierarchical_bvar_gp`.

## 3. Stan / PyMC adapters

ADR 0006: native Laplace first; external adapters deferred. Tracked as
`intentional_deviation` on `bayes.backend.stan_pymc`.

## 4. MCMC chain diagnostics / SBC

§18.4 items that require multi-chain MCMC (ESS, divergences, SBC) wait for
HMC/SMC. PPC, prior sensitivity, and Laplace convergence/curvature diagnostics
are shipped. Tracked as `intentional_deviation` on
`bayes.validate.mcmc_diagnostics`.

## 5. Bayesian CI tests (§12)

Not a delivered Bayesian surface. Tracked as `intentional_deviation` on
`bayes.ci.tests`.

## Verification

Bayesian `done` rows are backed by conformance under `conformance/bayesian/`
and/or criterion benches mapped in `scripts/gate_bayesian.sh`.
