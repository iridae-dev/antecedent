# CI / discovery deviations

Intentional deviations relative to `parity/tigramite.toml` for the CI framework
and PCMCI family.

## 1. GPDC is native RBF-GP + distance correlation (no torch)

Numerical parity with Tigramite’s torch-backed GPDC is not required; deviations
are expected under the native backend. Tracked as `intentional_deviation` on
`tigramite.ci.gpdc`. Significance is a seeded permutation null on the residual
distance correlation (add-one p-value over 49 Y-residual shuffles), deterministic
under the run seed.

## 2. PCMCI+ conformance is clean-room Exact parents

`conformance/tigramite/pcmci_plus_lag0` pins a synthetic SCM with lagged +
contemporaneous parents. Black-box Tigramite PCMCI+ graph comparison is not
pinned in this fixture (`tigramite.available = false`). Tracked as
`intentional_deviation` on `tigramite.discovery.pcmci_plus`.

## Notes (not deviations)

- `weighted_parcorr` accepts observation weights via Python
 `discover_pcmci(_plus)(..., weights=...)`.
- Pairwise multivariate wrapper is registered as `pairwise_multivariate`.
- Multivariate ParCorr uses block residualization + first canonical correlation
 (scalar blocks reduce to ordinary ParCorr).
- PC1 + shifted MCI conditioning: the engine uses PC1 (`run_pc_stable`,
 `max_combinations = 1`) and MCI conditions on `pa(X_{t−τ})` over a
 `2·max_lag` frame. Boundary convention `p >= alpha` for independence (vs
 Tigramite's `p > alpha`) is retained.
