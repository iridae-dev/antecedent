# GCM / counterfactual deviations

Intentional waivers relative to `parity/gcm.toml`. Shapley / coalition
attribution is `done` under the attribution inventory.

## 1. Bayesian DAG posterior search

Graph-posterior **model collections** consume supplied `WeightedGraphSamples`.
Bayesian DAG/DBN search stays waived (see `bayesian_deviations.md`).

## Verification

GCM `done` rows are backed by conformance under `conformance/gcm/`, unit tests,
and/or criterion benches mapped in `scripts/gate_gcm.sh`.
