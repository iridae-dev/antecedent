# Transport theorem scope and limits

This page cites registries. It does not copy them.

Stage transport routes are licensed in `transport_stages.toml`; the analyze Cartesian cell remains trial-IPW `TransportQuery` on an explicit `Admg`.

## Theorem families

`TheoremScope` constructors live in
`crates/antecedent-core/src/query/transport_contract.rs`. Completeness is
per family:

| Family | Guarantee | Pin |
| --- | --- | --- |
| Classical single-source sID | complete in the paper experimental-information family | `antecedent.transport.advanced.identify_classical` → [arXiv:1312.7485v1](https://arxiv.org/abs/1312.7485v1) |
| Classical meta-transport | complete in the multi-source experimental family | `antecedent.transport.advanced.identify_meta` → [Bareinboim 2013](https://proceedings.mlr.press/v31/bareinboim13a.pdf) |
| Finite catalog search | sound and incomplete | closed as `transport.finite_catalog_search` |
| Limited / z-experiment | sound and incomplete within 12 observed and 4 controllable variables; positives use cited joints, line-11 obstructions use the complete source family | `antecedent.transport.advanced.identify_z_transport` → [arXiv:1309.6842](https://arxiv.org/abs/1309.6842) |

Classical sID completeness applies only to the paper experimental-information family, not to every catalog the API can represent. The static checker in
`scripts/check_transport_stages.py` refuses a completeness guarantee that
drops those pins.

## Support split

- Analyze Cartesian ([`docs/support-matrix.md`](../support-matrix.md), generated
  from `parity/support_licensed.toml`): licensed trial-IPW
  `TransportQuery` × `Admg` × explicit × Frequentist × none.
- Stage routes (`parity/transport_stages.toml`, default closed): identify,
  prepare, evaluate, uncertainty, consume for exact, statistical, grid,
  catalog, and learned-trial paths.

Do not add stage routes as fake analyze cells.

## Estimation assumptions

`EmpiricalTable` is the conservative default. `LearnedCategorical` and
`TrialAipw` change the assumption set and must be passed explicitly. Exact
laws make no sampling-coverage claim.

## Calibration scope

Bound records only:

- four `ClassicalTransport` percentile-bootstrap rows
- two `TransportQuery` / `transport.trial_ipw` analytic-SE rows

Learned-trial intervals stay uncalibrated (`calibration_reason`).
`transport.multi_source_calibrated_coverage` and
`transport.simultaneous_grid_bands` stay closed. T7/T8 full coverage
remeasurement is not claimed here.

## Migration and failures

- Day-1 verbs: [`migrations/2.0-transport-day1.md`](../migrations/2.0-transport-day1.md)
- Outcome map: [`transport-failure.md`](transport-failure.md)
- Architecture: [`architecture/transport-exact.md`](../architecture/transport-exact.md),
  [`architecture/transport-meta-grid.md`](../architecture/transport-meta-grid.md)
