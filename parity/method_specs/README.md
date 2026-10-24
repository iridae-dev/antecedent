# Method specifications

208 records in `specs.toml`, one per child in `parity/method_children/children.toml`. Each carries the identifying result, assumptions, sampling units, estimator or posterior construction, diagnostics and failure policy of its method, the route or method record that executes it, and an explicit review decision.

`python3 scripts/reconcile_method_specs.py` completes and checks the records; `--check` fails on a stale record, on a record whose gap differs from its child, on a review decision that contradicts the executing evidence, and on an admitted record that leaves a scientific field `unlisted`.

## Review decision

| Field | Meaning |
| --- | --- |
| `admission` | `licensed`, `executing_unlicensed`, `refused` or `not_admitted`. Derived from the counterpart inventory (`parity/counterparts.toml`), which is derived from the route manifest's method records and the verified fixtures. |
| `review_status` | `admitted` (advertised at the admission it names and no further), `specification_only` (a settled axis boundary), `blocked_research` (an unproved identification or posterior question) or `not_admitted`. An explicit `not_admitted` review, for example the reviewed Gaussian-panel slice, is kept and never overwritten. |
| `decision_basis` | The evidence behind the decision: the counterpart status, the executing fixture, the artifact consumer and the uncertainty procedure with its basis. |
| `provenance`, `api_review` | The clean-room provenance records that cover the implementing module (or the reason the module implements no algorithm) and whether its public surface is in the reviewed API snapshot. `scripts/gate_provenance_schema.py` fails an admitted method without them. |

A scientific field that a hand-written record leaves `unlisted` is filled from the implementing module's own header, so the check can find the sentence there. A record written under an earlier gap has its failure policy, diagnostics and estimator rewritten from the current evidence; its adjusted claim is the child's recorded `missing` text.

## Counts by review_status

| review_status | Specs |
| --- | --- |
| admitted | 178 |
| blocked_research | 6 |
| not_admitted | 8 |
| specification_only | 16 |

## Counts by admission

| admission | Specs |
| --- | --- |
| executing_unlicensed | 116 |
| licensed | 62 |
| not_admitted | 8 |
| refused | 22 |
