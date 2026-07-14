# Phase 5 deviations

Intentional deviations / notes for DESIGN.md §32 Phase 5 (PCMCI+ and full CI
framework) relative to `parity/tigramite.toml` (authoritative for `status`).

**Kept deferrals only** (everything else in Phase 5 is production depth with
calibration / conformance / benches as applicable):

## 1. GPDC is native RBF-GP + distance correlation (no torch)

Matches the Phase 5 locked decision. Numerical parity with Tigramite’s
torch-backed GPDC is not required; deviations are expected under the native
backend. Tracked as `intentional_deviation` on `tigramite.ci.gpdc`. Significance
is a seeded permutation null on the residual distance correlation (add-one
p-value over 49 Y-residual shuffles), deterministic under the run seed.

## 2. PCMCI+ conformance is clean-room Exact parents

`conformance/tigramite/pcmci_plus_lag0` pins a synthetic SCM with lagged +
contemporaneous parents. Black-box Tigramite PCMCI+ graph comparison is not
pinned in this fixture (`tigramite.available = false`). Tracked as
`intentional_deviation` on `tigramite.discovery.pcmci_plus`.

## 3. Circle endpoints / PAG / LPCMCI remain Phase 8

Temporal CPDAG rejects `Endpoint::Circle` with a clear error. PAG / LPCMCI stay
out of Phase 5 (`tigramite.discovery.lpcmci` pending → Phase 8).

## Notes (not deviations)

- `weighted_parcorr` accepts observation weights via Python
  `discover_pcmci(_plus)(..., weights=...)`.
- Pairwise multivariate wrapper is registered as `pairwise_multivariate`.
- Multivariate ParCorr uses block residualization + first canonical correlation
  (scalar blocks reduce to ordinary ParCorr).
- J-PCMCI+ / RPCMCI remain Phase 9 (pending in the inventory).

## 6. Correctness pass (2026-07-21): PC1 + shifted MCI conditioning

The engine's PC phase previously tested every `C(n-1, q)` conditioning subset per
candidate (exponential, and a different surviving-parent set than Tigramite); it now
implements PC1 (`run_pc_stable`, `max_combinations = 1`): candidates ranked by minimum
absolute statistic, one strongest-`q` conditioning set per level. The MCI phase
previously conditioned on `pa(X)` at unshifted lags; it now conditions on
`pa(X_{t−τ})` (parents shifted by the link lag) over a `2·max_lag` frame, matching
Tigramite's `2xtau_max` cut-off. `conformance/tigramite/pcmci_lag1` and
`pcmci_plus_lag0` pins hold. Boundary convention `p >= alpha` for independence (vs
Tigramite's `p > alpha`) is retained.
