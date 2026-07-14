# Attribution deviations

Intentional limitations relative to `parity/attribution.toml` and
`parity/gcm.toml`.

## 1. Mechanism-change test proxies

Attribution surfaces required by DESIGN.md §17 / DoWhy-GCM are implemented.
Kernel two-sample / change-point mechanism tests beyond the shipped LR /
mean-diff / classifier proxies remain available as future extensions without
blocking current inventory rows.

## Verification

Attribution `done` rows are backed by unit tests, `conformance/attribution/`,
and/or criterion benches mapped in `scripts/gate_attribution.sh`.
