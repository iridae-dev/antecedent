# Python analysis workflow (1.10)

These examples require the built `1.10.0` branch; see the
[Python environment setup](../examples/README.md#python-environment-110-branch).

## Start with one call

Every Antecedent analysis retains a reusable study and exports a contracted execution; custom validator results travel as caller-attested, not re-verifiable, evidence, and a row-weight retarget re-executes only on its own data snapshot.
Every reported interval states its calibration: calibrated when a coverage record matches the execution and the execution is inside that record's scope; scope_not_assessed when a record matches but the execution is outside its scope or the record is a boundary; unavailable with a reason code when no record exists.
Identities are distinct and stable: every IdentityDomain plus target_weights is domain-separated and registered in parity/identity.toml.

The existing notebook call still works. It retains a prepared study on the
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

`result.answer` is the safe consumption interface: `point`, `bounds`, `partial`,
or `unavailable`. A partial answer does not expose an unrestricted scalar there.
Historical fields (`effect`, `ate`, `posterior`, `response`) remain accessible
for existing callers. `.effect` / `.ate` emit a `UserWarning` when a point
display would misrepresent the claim; `ANTECEDENT_STRICT_ANSWER=1` raises
instead. That is a safe interface plus a warning, not misuse-proofing.

## Prepare explicitly when useful

```python
study = ant.prepare(data, graph=graph, query=query, seed=19)
report = study.inspect()        # includes cached identification; does not estimate
result = study.estimate()      # uses retained data and preparation's seed/threads
```

**Two `prepare` default regimes.** `ant.prepare(...)` uses one-call `analyze`
defaults (standard bootstrap and placebo refutation for scalar effects).
`antecedent.estimation.PreparedAnalysis.prepare(...)` keeps historical
interactive defaults (`refute=False`, `latency="interactive"`). Pass `refute`,
`bootstrap`, and `latency` explicitly if the two entry points must agree.

`study.preflight()` is the explicit structural-only view; `study.inspect()`
reports everything already known.

`identify(...)` returns an `Identification` whose `statement`, `verdict`,
`assumption_statements`, and `derivation_statements` are human-readable
state on the object (`identification.to_dict()` includes them). That is
readable identification data, not a notebook renderer.

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

`program_id` is the compiled program identity (`contract["program"]`).
`claim_id` is the execution claim identity (`contract["claim"]["claim_id"]`).
They are different layers. A prepared inspect has a program and no claim.
Earlier 1.10 drafts used `claim_id` for the program digest; that name now
means the execution claim, and `program_id` is the alias for the compiled
program.

## Current boundaries

**Composition evidence.** Every licensed support-matrix cell has a first-class
inspect/contract coordinate and completes inspect → preview → execute →
claim → consume on the Rust compiler path
(`compiler.e2e_licensed_cells`). That is composition-seam evidence on
synthetic licensed-coordinate fixtures (`internal_cross_check`). Parent
estimator evidence is not inherited.

**Reusable studies.** `analyze` is `prepare(...).estimate()`. Every licensed
Python product-matrix route retains a study and a contracted export. Custom
validator results travel as caller-attested evidence. A result with no
execution (prepared-only, cancelled, or a body-only load) refuses `.study` /
`.export()` with `not_executed` or `cancelled_no_claim`. It never silently
reruns discovery.

Retargeting remains available from frozen scores. A nonconstant-weight retarget
exports with a `target_weights` identity bound to its data snapshot. Re-execution
on a different snapshot raises `row_weights_bound_to_snapshot`.
