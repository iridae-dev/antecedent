# When transport fails

A transport question can fail to produce a number for six different reasons, and each has a different remedy. Antecedent keeps them apart so you can tell whether to collect evidence, change the diagram, accept a weaker claim or give the search more room. A not-certified outcome is a conservative refusal, not a proof that transport is impossible.

| Outcome | Where to read it | A neighbouring success |
| --- | --- | --- |
| Not certified | `ident.certificate["identification_status"] == "not_certified"`, `ident.inspect().identification` | [Exact-law success](https://github.com/iridae-dev/antecedent/blob/v3.0.0/examples/python/transport_exact.py) |
| Theorem-scoped impossibility | `ident.certificate["identification_status"] == "proven_non_transportable"` | Complementary sources in [the meta-grid example](https://github.com/iridae-dev/antecedent/blob/v3.0.0/examples/python/transport_meta_grid.py) |
| Missing evidence | `ident.inspect().support`, `answer.kind == "unavailable"` | Full versus partial catalog in the same meta-grid example |
| Local support or positivity | `result.support` | A supported assignment on the same grid |
| Uncalibrated or unavailable interval | `result.inspect().calibration`, `result.inspect().uncertainty` | Exact law (no interval) beside the [statistical bootstrap](https://github.com/iridae-dev/antecedent/blob/v3.0.0/examples/rust/transport_statistical.rs) |
| Budget exhaustion | a raised `transport.identification_budget`, or a stage whose `identification_status` is `budget_cancel` | The same query under default `SidLimits` |

## One identification-status vocabulary

Every transport binding reports `identification_status` with one of five values: `identified`, `proven_non_transportable`, `missing_evidence`, `not_certified` and `budget_cancel`. In Python it is a property of each stage and identification, a key of each `decision()` and scenario row, and `certificate["identification_status"]` on `ant.identify`. In Rust it is `identification_status()` on each decision type. `ident.status` is the general identification status (`NotIdentified` for every refusal here); the transport kind is `identification_status`.

Some serialized and public fields still carry a narrower spelling. Each maps to one canonical value:

| Spelling in a field | Where | Canonical |
| --- | --- | --- |
| `exhausted` | a stage's `outcome` (z, multi-source z, mixed-source, ADMG-conditional) | `budget_cancel` |
| `unevaluated` | a scenario row's `status` | `budget_cancel` |
| `stopped` | a temporal outcome's status | `budget_cancel` |
| `structurally_unidentified` | a scenario row's or temporal outcome's status | `proven_non_transportable` |
| `combined_identified` | a two-source z decision's `outcome` | `identified` |
| `named_route` | a mixed-source stage's `outcome` (another theorem-scoped route identifies it) | `identified` |

`antecedent.transport.advanced.identification_status(value)` reads any of these, whether a string, a stage or a decision mapping, as its canonical value. Reason codes such as `transport_budget_cancel` and `transport_not_certified` are unchanged.

## Not certified

`identify` returns `not_certified` when the implemented search does not decide the query. That is not an s-hedge, and the effect may well be transportable by a formula the search does not cover. The neighbouring success is the same `X -> Y` question in `transport_exact.py`, which identifies and evaluates.

```python
ident = ant.identify(graph=graph, query=query)
print(ident.certificate["identification_status"], ident.inspect().identification.summary)
```

To go further, declare more structure (a smaller selection set, a source that can experiment on more variables) or use a route whose theorem family covers your question; [Transport limits by route](transport-scope.md) lists what each route decides and where it is incomplete.

## Theorem-scoped impossibility

A checked s-hedge makes the outcome `proven_non_transportable`: no formula exists from the declared evidence. The result is scoped to the declared experimental information, so adding evidence can change it. In `transport_meta_grid.py`, removing one complementary source leaves a scoped obstruction (`transport_negative_witness`, `transport_complementary_sources`), while both sources together identify the effect.

## Missing evidence

The formula can be identified while a required joint law is unbound. The answer is then `answer.kind == "unavailable"`, and the support slot names the population. The meta-grid script's partial catalog sits next to its full-law success.

For a checked classical identification, `antecedent.transport.advanced.inspect_proof_graph(identification, catalog)` lists the checked rule steps and, for each factor, the population, variables, conditioning, exact intervention set and the regime that supplies it or the binding failure. It verifies the stored proof against the diagram and query before reporting leaves.

`EvidenceCatalogDelta` holds proposed regimes apart from the supplied catalog. `delta.preview(catalog)` makes a temporary structural preview so you can ask whether a study you have not run would identify the query. A preview has no data snapshot: results must enter through the ordinary prepare and rebind path.

## Local support and positivity

A grid keeps unsupported coordinates. `result.support` locates the empty cell and does not delete the point. Each point's `support_status` keeps two failures apart: a coordinate with no supplied law, provider or declared domain is `missing_evidence`; an empty stratum or zero denominator in supplied evidence is `outside_empirical_support`. The surface `support.status` is the weakest label over the points, with `missing_evidence` weakest, and the grid's support slot counts them separately (for example `1 unavailable (1 missing evidence, 0 support failure)`). The remedy is evidence at the missing coordinate or a narrower target.

## Uncalibrated or unavailable interval

Exact supplied laws report no sampling interval. The empirical-table path publishes a pointwise percentile bootstrap that is nominal unless a coverage record is bound to that construction. Read `result.inspect().calibration` and `result.inspect().uncertainty`.

## Budget exhaustion

An exhausted identification step or depth budget raises `transport.identification_budget`. It is never a negative witness. Re-run the same query under larger limits or a simpler diagram; the outcome says the search stopped, not that the query cannot be answered.

## Related

- [Carry an effect to another population](transport/carry-an-effect.md)
- [Transport limits by route](transport-scope.md)
- [Refusal recipes](refusal-recipes.md)
- [Refusals](../concepts/refusals.md)
