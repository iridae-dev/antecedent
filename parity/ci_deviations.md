# CI / discovery deviations

Intentional deviations relative to `parity/tigramite.toml` for the CI framework
and PCMCI family.

## 1. GPDC is native RBF-GP + distance correlation (no torch)

Numerical parity with Tigramite’s torch-backed GPDC is not required; deviations
are expected under the native backend. Tracked as `intentional_deviation` on
`tigramite.ci.gpdc`. Significance is a seeded permutation null on the residual
distance correlation (add-one p-value over 49 Y-residual shuffles), deterministic
under the run seed.

## 2. PCMCI+ Meek R4 removed (aligned with tigramite)

PCMCI+ orientation applies Meek R1–R3 on contemporaneous links only
(`ContempMeekR1`–`ContempMeekR3`). Tigramite likewise applies R1–R3 on
contemporaneous links and does not run R4. The former intentional R4
deviation has been removed. Black-box edge-set equality is pinned in
`conformance/tigramite/pcmci_plus_lag0`.

## 3. LPCMCI / J-PCMCI+ / RPCMCI black-box equality

LPCMCI (P4.3) now runs Gerhardus & Runge Alg. 1 (middle marks, weakly-minimal
sepsets, interleaved ancestral/non-ancestral phases, R8–R10). J-PCMCI+ (P4.4)
runs Günther et al. pooled four-phase search (observed context + space/time
dummies under link assumptions; PCMCI+ majority + ContempMeek orientation).
Full tigramite edge-set numerical equality remains optional for LPCMCI /
J-PCMCI+ / RPCMCI; conformance pins structure / algorithm id. RPCMCI black-box
equality still deferred (P4.5).

## Notes (aligned with tigramite; not deviations)

- Alpha boundary: independence when `p > alpha`; retain links with `p <= alpha`
  (matches tigramite).
- FDR: BH/BY/Bonferroni/Holm via `FdrAdjustment`; `exclude_contemporaneous=true` by
  default (tigramite `get_corrected_pvalues`).
- ParCorr residualization: no intercept column; plain least-squares on Z only
  (matches tigramite `lstsq`). Analytic df remains `n − 2 − |Z|`.
- PC and MCI lagged frames both use depth `2 · max_lag` (tigramite
  `cut_off='2xtau_max'`).
- PCMCI MCI scores the full constrained candidate family; PC parents are
  conditioning-only (tigramite `run_mci`).
- PCMCI+ uses lagged-only PC1 then a contemporaneous MCI phase, lag-0
  symmetrization (both directions), majority collider, and Meek R1–R3 on
  contemporaneous links. Cross-variable edge sets match the
  `pcmci_plus_lag0` pin; autoregressive self-lags may still differ from
  tigramite and are allowed as extras in that fixture.
- `weighted_parcorr` accepts observation weights via Python
  `discover_pcmci(_plus)(..., weights=...)`.
- Pairwise multivariate wrapper is registered as `pairwise_multivariate`.
- Multivariate ParCorr uses block residualization + first canonical correlation
  (scalar blocks reduce to ordinary ParCorr).
- The engine uses PC1 (`run_pc_stable`, `max_combinations = 1`) and MCI
  conditions on `pa(X_{t−τ})` over the shared `2·max_lag` frame.
