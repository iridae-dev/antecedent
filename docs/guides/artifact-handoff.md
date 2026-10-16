# Save a result and consume it independently

A consumer needs artifact bytes and an expected identity from a separately retained producer record. An expectation extracted from the received container does not independently bind that container.

Run the producer and consumer in separate processes:

```bash
python examples/python/artifact_producer.py --output /tmp/antecedent-handoff
python examples/python/artifact_consumer.py --input /tmp/antecedent-handoff
```

In production, place `expected.json` in your trusted job registry or another separately controlled record store. The shared temporary directory here demonstrates the file format and process boundary, not an independent trust boundary.

## Producer

<!-- example: examples/python/artifact_producer.py -->

## Consumer

<!-- example: examples/python/artifact_consumer.py -->

The consumer verifies a valid load and deliberately tries a different expected identity and altered bytes. Both must fail. It prints identity binding, replayability and resumability separately: accepting an execution does not recreate its live study.

Use [artifact format and compatibility](../artifacts.md) for specialized families and [the 2.4 migration](../migrations/2.4-python-ux.md) when replacing older export receipts.
