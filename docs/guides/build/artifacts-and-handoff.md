# Save, hand off and accept results

A result you export can be stored, sent to another service and read there without access to the data, the study or the process that produced it. The consumer gets the recorded claim with its identification, support and uncertainty statements. It does not get a live study: loading restores what was recorded, and recomputing means preparing again.

This guide follows one result through four steps: produce, store, consume and accept.

## What an artifact is

An artifact is a checksummed container of bytes. Loading it never runs a callback, never imports anything the bytes name and never recreates the study that produced it. A consumer that needs a different answer prepares a new study on its own data; the loaded result stays fixed.

An artifact has two parts that travel separately:

| Part | Where it lives | What it does |
| --- | --- | --- |
| Bytes | Wherever the result is shipped: a file, a queue, an object store. | Carry the recorded claim. |
| Expectation | A record the producer controls and a consumer trusts. | State the identity the consumer requires. |

An expectation read out of the received container cannot independently bind that container, because whoever can alter the bytes can alter the claim about them. Keep the expectation in a job registry, a signed manifest or any record the byte channel cannot rewrite.

## Produce and store

`result.export()` returns an immutable `ExportedArtifact`. Its `bytes(artifact)` (or `.data`) are the container, `.kind` names the scientific family, `.expectation` is the identity to retain, and `.metadata` holds family-specific records.

```python
artifact = result.export()
path.write_bytes(bytes(artifact))
expected = artifact.expectation  # store this where the bytes cannot rewrite it
```

Persistence does not take file paths implicitly: write the bytes yourself and read them back with `Path.read_bytes`. A result whose route cannot be reproduced from its recorded contract is not exported silently; the call refuses and names why. A refusal can also be exported, so a host can store "this question was declined, for this reason" under the same identity rules.

## Consume and accept

`ant.load(bytes, expected=expected)` rebuilds the recorded result and binds it to the identity you retained.

```python
loaded = ant.load(path.read_bytes(), expected=expected)
loaded.answer.value
loaded.acceptance
```

Read the acceptance, not only the absence of an error. It reports four things separately:

| Field | Meaning |
| --- | --- |
| `identity_bound` | The load was checked against an identity you supplied independently. |
| `answer_available` | The recorded answer is kept (the load was verified or sealed). |
| `replayable` | The recorded program can be re-executed from what the artifact holds. |
| `resumable` | A live session can be resumed from it. |

Receiving bytes, understanding their meaning, verifying their dependencies and being able to execute them are different states. A host that accepts a claim requires `identity_bound` and `answer_available`. An artifact whose family is recognised but not consumed can be inspected and forwarded with `export()`, and scientific consumption stays unavailable. A malformed or altered container raises a `serialization` error, and a different expected identity raises as well.

`ant.artifacts.inspect(bytes)` reads the container's metadata. It is inspection, not acceptance of a scientific claim.

## Across a process boundary

The producer writes the bytes and the identity it expects; the consumer reads both and also tries the two requests it must refuse. Run them as separate processes:

```bash
python examples/python/artifact_producer.py --output /tmp/antecedent-handoff
python examples/python/artifact_consumer.py --input /tmp/antecedent-handoff
```

In production, place `expected.json` in your trusted job registry or another separately controlled record store. The shared temporary directory here demonstrates the file format and the process boundary, not an independent trust boundary.

### Producer

<!-- example: examples/python/artifact_producer.py -->

### Consumer

<!-- example: examples/python/artifact_consumer.py -->

The consumer verifies a valid load and deliberately tries a different expected identity and altered bytes. Both must fail. Identity binding, replayability and resumability are reported separately, because accepting a result does not recreate its live study.

## Ship a fitted effect model as a small runtime object

When a service needs to evaluate a fitted effect model without the fitting code, export it with `antecedent.compact_export`. A `CompactExport` stores a coefficient vector, its full covariance, a finite basis (intercept, linear, power and one-hot terms of named quantities), the declared support of every input and a mask of query regions the model must refuse, all under one identity. It evaluates a point prediction `f(x) = b' phi(x)` inside the declared support and outside the mask, and nothing else.

```python
from antecedent.compact_export import CompactExport, Intercept, Linear, Numeric, Power, Quantity
from antecedent.outcomes import require_success

export = CompactExport.build(
    response=Quantity("y", units="mmHg", role="outcome"),
    inputs=[Numeric("x", 0.0, 10.0, units="mg", role="treatment")],
    terms=[Intercept(), Linear("x"), Power("x", 2)],
    coefficients=[1.0, 2.0, -0.5],
    covariance=[[0.04, 0.0, -0.002], [0.0, 0.01, 0.0], [-0.002, 0.0, 0.0004]],
)
at_two = export.evaluate({"x": 2.0})
at_two.point, at_two.standard_error     # 3.0 and 0.08 * sqrt(11)

kept = export.identity                  # retain this out of band
consumer = require_success(CompactExport.consume(bytes(export.export()), expected_identity=kept))
consumer.evaluate({"x": 2.0}) == at_two
```

The only uncertainty it reports is the model-based `sqrt(phi' V phi)`, with calibration unmeasured: it carries no extrapolation or misspecification uncertainty and is never an interval. A query outside the support, with a missing, unknown or non-finite quantity, or inside a mask region is refused with `cell_not_licensed` and a detail such as `compact_export.out_of_support` or `compact_export.masked_region`; `evaluate_many` returns refusals in place instead of raising. A consumer that passes a different `expected_identity` is refused with `compact_export.identity_unexpected`. Categorical inputs use `Categorical` and `OneHot` terms, and a `MaskRegion` names a region to refuse.

## Keep a result current

A stored artifact describes the data it was computed on. Whether new data make it stale is a question your application asks, not a timer in the library. Compare the new data with the stored result using the [staleness diagnostics](../staleness.md) and act on the typed recommendation: reuse, refresh, re-identify or refuse. To produce a refreshed result, `ant.rebind(study, data=...)` and `ant.estimate` run the same program on the new data as a new branch, and the earlier artifact remains valid for the data it describes. [Retained refresh and state publication](../retained-refresh.md) covers the full cycle.

## Where the format is specified

The container layout, section compression, the versioned durable format and the artifact kinds for each result family are in [artifact format and compatibility](../../artifacts.md). Artifacts written by an older supported format load and migrate; one that needs a newer reader than yours is refused rather than guessed. Record the library version a service was validated with, and treat a format change as a migration. [Embed Antecedent in an application or agent](embedding.md) lists the three version numbers a host should track.

To carry a result into a decision or composition, an artifact is checked against the consumer's [interface contract](scientific-interfaces.md) before anything runs.
