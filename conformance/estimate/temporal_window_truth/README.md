# Window contrasts and mediation on a stationary template

Known truth for the analyses that read the unrolled window of a stationary template: the pulse,
sustained and conditional contrasts and the natural mediation decomposition on a supplied or
accepted `TemporalAdmg`, over a graph posterior of lagged `TemporalAdmg` atoms, and on a
`TemporalPag`. Every case is a binary stationary structural process whose noises are uniform on
equally likely bins, so the effects come from evaluating the structural equations of the unrolled
window under an intervention on every noise cell, not from an identification formula
(`reference.py`).

- `contrast` (columns `x`, `y`, `w`): `x` and `w` fair coins, `y(t)` follows `x(t)`, `w(t)` and
  `y(t-1)`. A window of three periods holds the history, the treated period and the outcome
  period. Pulse `1/8`, sustained `3/8`; the conditional contrasts (on `w` at the first treated
  period, which no treated period reaches) equal them because the modifier precedes the treatment.
- `planted_series`: the same process with the same support, both deterministic cells and the
  stationary law, in which the treatment has no effect (every contrast is exactly zero).
  Concatenated after `series` it is the planted break the split-half stationarity check must
  detect.
- `mediation` (columns `c`, `x`, `w`, `m`, `y`): `c(t-1) -> x`, `x -> m`, `w -> m`, `m -> y`,
  `x -> y`; total `5/16`, natural direct `5/16`, pure natural indirect `1/8`. The pure indirect
  effect differs from the total minus the direct effect.
- `series`: an Eulerian circuit through the multigraph of the observed windows, drawn in a fixed
  pseudo-random order, so its consecutive windows have exactly the law of the process (the check
  re-counts them). The contrast series repeats the exact window law sixteen times so that a
  contiguous stretch or a column rotation still holds every conditioning cell of the full table.
- `bow_witness`: two structural models with one latent cause behind the same observed law and
  different effects of the treatment, so a window that holds a bow arc has no identified value.

Run `uv run --frozen --script conformance/estimate/temporal_window_truth/reference.py --check`
(add `--self-test` to confirm the check rejects mutated fixtures).

Consumer: `crates/antecedent/tests/temporal_window_route_evidence.rs`.
