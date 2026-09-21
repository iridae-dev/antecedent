# Transport failure guide

These outcomes are different. Do not collapse them into one error. Each
section names a valid neighboring success.

A not-certified transport outcome is a conservative refusal, not a proof of impossibility.

| Outcome | Where to read it | Neighbor |
| --- | --- | --- |
| Not certified | `ident.status`, `ident.inspect().identification` | [Exact-law success](../../examples/python/transport_exact.py) |
| Theorem-scoped impossibility | `ident.status == "proven_non_transportable"` | Combined complementary sources in [the meta-grid example](../../examples/python/transport_meta_grid.py) |
| Missing evidence | `ident.inspect().support`, `answer.kind="unavailable"` | Full vs partial catalog in the same meta-grid example |
| Local support / positivity | `result.support` | Fixture family `transport_support_local` |
| Uncalibrated / unavailable interval | `result.calibration`, `inspect().uncertainty` | Exact law (no interval) beside [statistical bootstrap](../../examples/rust/transport_statistical.rs) |
| Budget exhaustion | raised `transport.identification_budget` | Same query under default `SidLimits`; fixture `transport_budget_refusal` |

## Not certified

`identify` can return `not_certified` when the implemented search does not
decide the query. That is not an s-hedge. The neighboring success is the
same `X → Y` question in `examples/python/transport_exact.py`, which identifies
and evaluates.

```python
ident = ant.identify(graph=graph, query=query)
print(ident.status, ident.inspect().identification.summary)
```

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

## Local support / positivity

A grid keeps unsupported coordinates. `result.support` locates the empty
cell; it does not delete the point. See `transport_support_local` in
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
