# Python analysis workflow (1.10)

These examples require the built `1.10.0` branch; see the
[Python environment setup](../examples/README.md#python-environment-110-branch).

## Start with one call

The existing notebook call still works. It now retains a prepared study on the
ordinary tabular and temporal routes supported by `PreparedAnalysis`:

```python
import antecedent as ant

result = ant.analyze(data, graph=graph, query=ant.AverageEffect("treatment", "outcome"))
result                         # notebook display
report = result.inspect()      # answer, four reasoning slots, calibration, diagnostics
study = result.study           # the preparation that produced this result
```

The paid-search and continuous-response notebooks need no extra preparation
steps. Their results can now be retained for repeated estimation. See the
[executable workflow example](../examples/python/analysis_workflow.py).

`result.answer` describes the answer shape. A partial answer does not expose an
unrestricted scalar there. Historical numerical fields (`effect`, `posterior`,
`response`, etc.) remain available for callers that explicitly handle their
scientific scope.

## Prepare explicitly when useful

```python
study = ant.prepare(data, graph=graph, query=query, seed=19)
report = study.inspect()        # includes cached identification; does not estimate
result = study.estimate()      # uses retained data and preparation's seed/threads
```

`prepare` uses the one-call workflow's scalar and response defaults. The older
`estimation.PreparedAnalysis.prepare` remains available with its historical
interactive defaults. `study.preflight()` is the explicit structural-only view;
`study.inspect()` reports everything already known.

| Operation | Data used | Changes the study's retained data? |
|---|---|---|
| `study.estimate()` | Retained data | No |
| `study.estimate(other_data)` | Supplied data | No |
| `study.refresh(new_data)` | Supplied data | Yes, after successful execution |
| `result.inspect()` | That result's execution | No |
| `result.export()` | That result's execution | No |

A failed refresh leaves the study usable. Refresh does not mutate an earlier
result or its exported bytes. A study is a live, in-process resource; retain it
while repeated estimation is useful and release references when done.

## Software and agent consumers

```python
report = result.inspect().to_dict()
# json.dumps(report, allow_nan=False) is supported.

try:
    next_result = study.refresh(new_data)
except ant.CausalError as error:
    refusal = error.report.to_dict()
    # Original exception type and descriptive message are preserved.
    # The failed study operation also retains error.study for recovery.
```

Refusals contain the operation, code, message, query and available review hint /
pending edges. Unknown scientific evidence stays unavailable. Existing Python
argument-validation `ValueError` and `TypeError` exceptions also receive reports;
their original exception types remain unchanged.

Reports retain identification mass, uncertainty sources and targets (including
omitted components), assumption obligations, and support evidence. A licensed
matrix cell does not establish empirical support or validate its assumptions.

Calibration is explicit on every result report and prepared inspection. Current
executions do not bind a calibration coverage artifact, so the default status is
`unavailable` with a reason. Supplied certificate calibration evidence is retained
with `scope_not_assessed`; neither refutation success nor a licensed matrix cell
is promoted into a calibration claim. This refactor does not add calibration
experiments or an MCP server.

## Portable executions

```python
encoded = result.export()
loaded = ant.load(encoded)
loaded.acceptance.verified
loaded.answer
loaded.inspect().to_dict()
assert loaded.export() == encoded
```

`load` runs the Rust semantic consumer. Malformed contracts fail closed;
missing or unrecognized contracts yield explicitly unavailable semantic views
that can still be forwarded. Acceptance verifies the artifact's semantic
contract, not the truth of causal assumptions. Loading does not reconstruct a
live study or recover the source dataset. The original decoded body is available
as `loaded.artifact.payload`.

For prior transfer, `Bayesian(prior_from=...)` still expects the posterior-only
payload from `result.study.export_artifact()`, taken before the study changes.
The full `result.export()` archive is intended for execution inspection and
forwarding. See [artifact examples](artifacts.md#exporting-prepared-results).

The legacy `claim_id` property names the compiled program identity; use the
artifact's `contract["claim"]["claim_id"]` for the execution's claim identity.

## Current boundaries

One-call retention covers ordinary prepared tabular / temporal scalar, class,
posterior-mixture and response routes. Legacy one-call paths with execution
callbacks, custom validators or estimator configuration, RD settings, panel /
event / multi-environment data, and non-posterior discovery configurations retain
their existing execution APIs. Accessing `.study` or `.export()` on a result that
has no retained execution gives a descriptive refusal; it never silently reruns
discovery. Derivatives can use explicit `prepare` on their licensed cells.

Retargeting remains available from frozen scores. A nonconstant-weight retarget
cannot yet export a contracted result: its target-weight identity is not encoded
in the portable contract. This is reported explicitly instead of labeling a new
population with the original target identity.
