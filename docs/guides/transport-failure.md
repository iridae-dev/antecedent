# Transport failure guide

These outcomes are different. Do not collapse them into one error. Each
section names a valid neighboring success.

A not-certified transport outcome is a conservative refusal, not a proof of impossibility.

| Outcome | Where to read it | Neighbor |
| --- | --- | --- |
| Not certified | `ident.certificate["identification_status"] == "not_certified"`, `ident.inspect().identification` | [Exact-law success](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/python/transport_exact.py) |
| Theorem-scoped impossibility | `ident.certificate["identification_status"] == "proven_non_transportable"` | Combined complementary sources in [the meta-grid example](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/python/transport_meta_grid.py) |
| Missing evidence | `ident.inspect().support`, `answer.kind="unavailable"` | Full vs partial catalog in the same meta-grid example |
| Local support / positivity | `result.support` | Fixture family `transport_support_local` |
| Uncalibrated / unavailable interval | `result.calibration`, `inspect().uncertainty` | Exact law (no interval) beside [statistical bootstrap](https://github.com/iridae-dev/antecedent/blob/v2.3.0/examples/rust/transport_statistical.rs) |
| Budget exhaustion | raised `transport.identification_budget`, or a stage whose `identification_status` is `budget_cancel` | Same query under default `SidLimits`; fixture `transport_budget_refusal` |

## Not certified

`identify` can return `not_certified` when the implemented search does not
decide the query. That is not an s-hedge. The neighboring success is the
same `X → Y` question in `examples/python/transport_exact.py`, which identifies
and evaluates.

```python
ident = ant.identify(graph=graph, query=query)
print(ident.certificate["identification_status"], ident.inspect().identification.summary)
```

`ident.status` is the general identification status (`NotIdentified` here, for
every refusal); the transport kind is `identification_status`.

## One identification-status vocabulary

Every transport binding reports `identification_status` with one of five
spellings, the Rust `TransportOutcomeKind` names: `identified`,
`proven_non_transportable`, `missing_evidence`, `not_certified` and
`budget_cancel`. In Rust it is `identification_status()` on each decision type
(`ClassicalTransportResult`, `CatalogTransportResult`, `ZTransportResult`,
`ZTransportDecision`, `TwoSourceZTransportDecision`, `MzTransportDecision`,
`MixedSourceDecision`, `ConditionalTransportDecision`, `ScenarioOutcome`,
`TemporalOutcome`); in Python it is the `identification_status` property of
each stage and identification, the `identification_status` key of each
`decision()` and scenario row, and `certificate["identification_status"]` on
`ant.identify`.

The spellings used before 2.2 are kept where they are serialized or public and
are deprecated:

| Earlier spelling | Where | Canonical |
| --- | --- | --- |
| `exhausted` | a stage's `outcome` (z, multi-source z, mixed-source, ADMG-conditional) | `budget_cancel` |
| `unevaluated` | a scenario row's `status` | `budget_cancel` |
| `stopped` | `TemporalOutcome::status()` | `budget_cancel` |
| `structurally_unidentified` | a scenario row's `status`, `TemporalOutcome::status()` | `proven_non_transportable` |
| `combined_identified` | a two-source z decision's `outcome` | `identified` |
| `named_route` | a mixed-source stage's `outcome` (another theorem-scoped route identifies it) | `identified` |

`antecedent.transport.advanced.identification_status(value)` reads any of these
(a string, a stage, or a decision mapping) as its canonical spelling, and
`TransportOutcomeKind::from_identification_status` does the same in Rust.
Reason codes (`transport_budget_cancel`, `transport_not_certified`, ...) are
unchanged.

## Theorem-scoped impossibility

A checked s-hedge is `proven_non_transportable`. Removing one complementary
source in `examples/python/transport_meta_grid.py` is the neighbor: both
sources identify; either source alone is a scoped obstruction
(`transport_negative_witness` / `transport_complementary_sources`).

## Missing evidence

The formula can be identified while a required joint is unbound. Then
`answer.kind == "unavailable"` and the support slot names the population.
The meta-grid script's partial catalog is that neighbor next to the full-law
success.

For a checked classical identification, `antecedent.transport.advanced`
exposes `inspect_proof_graph(identification, catalog)`. Its `steps` form the
checked rule graph; each step names the factor nodes it needs. Each `factors`
entry records the population, variables, conditioning and exact intervention
set, followed by the supplying regime ID or a binding failure. It verifies the
stored proof against the original diagram and query before reporting leaves.

`EvidenceCatalogDelta` holds proposed regimes separately from the supplied
catalog. `delta.preview(catalog)` makes a temporary structural preview for
re-identification; the source catalog and its prepared analyses retain their
original evidence status. A preview has no data snapshot, so actual results
must enter through the ordinary prepare and refresh path.

## Local support / positivity

A grid keeps unsupported coordinates. `result.support` locates the empty
cell; it does not delete the point. Each point's `support_status` (and the
transported curve's `support.point_status`) keeps the two failures apart: a
coordinate with no supplied law, provider or declared domain is
`missing_evidence`; an empty stratum or zero denominator in supplied evidence is
`outside_empirical_support`. The surface `support.status` is the weakest label
over the points, with `missing_evidence` weakest. The grid's support slot counts
them separately (`1 unavailable (1 missing evidence, 0 support failure)`);
artifacts written with the earlier undifferentiated count still load. See `transport_support_local` in
`parity/transport_stages.toml`. The neighboring success is a supported
assignment on the same grid.

## Uncalibrated or unavailable interval

Exact supplied laws report no sampling interval. The empirical-table
`StudyBuilder` path publishes a pointwise percentile bootstrap that is
nominal unless a coverage record is bound. Read `result.calibration` and
`inspect().uncertainty`. Do not infer coverage from a parent estimator name.

## Budget exhaustion

An exhausted identification step or depth budget is an error
(`transport.identification_budget`). It is never a negative witness. The
neighbor is the same query under default `SidLimits`
(`transport_budget_refusal`).
