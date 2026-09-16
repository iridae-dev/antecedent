# Python analysis workflow (1.10)

```python
import antecedent as ant

result = ant.analyze(data, graph=graph, query=ant.AverageEffect("treatment", "outcome"))
study = result.study
updated = study.refresh(new_data)
report = result.inspect().to_dict()
loaded = ant.load(result.export())
```

These five lines are the Python API:

| Line | What it does |
|---|---|
| `ant.analyze(...)` | Identifies the query on the graph (or on a `discovery=` configuration), runs only a licensed path, and returns the result |
| `result.study` | The compiled study that produced the result, retained for reuse |
| `study.refresh(new_data)` | Re-executes the same program on same-schema data; `result` and its export stay tied to their own execution |
| `result.inspect().to_dict()` | Everything known about the execution as JSON-safe data: `answer`, the four reasoning slots, identities, `contract`, `calibration`, `diagnostics` |
| `ant.load(result.export())` | Exports the contracted execution and loads it through the Rust semantic consumer |

The same five verbs apply across query kinds and structures: swap in
`PulseEffect`, `Counterfactual`, `MediationEffect`, `ConditionalEffect` or
`ResponseCurve`, and a `Dag`, `Cpdag`, lagged edge list or `discovery=`
configuration. `python/tests/test_golden_path.py` runs exactly these five lines,
with no warnings, on each of those routes.

Every Antecedent analysis retains a reusable study and exports a contracted execution; custom validator results travel as caller-attested, not re-verifiable, evidence, and a row-weight retarget re-executes only on its own data snapshot.
Every reported interval states its calibration: calibrated when a coverage record matches the execution and the execution is inside that record's scope; scope_not_assessed when a record matches but the execution is outside its scope or the record is a boundary; unavailable with a reason code when no record exists.
Identities are distinct and stable: every IdentityDomain plus target_weights is domain-separated and registered in parity/identity.toml.

The paid-search and continuous-response notebooks use the same call; see the
[executable workflow example](../examples/python/analysis_workflow.py).

## Reading the answer

`result.answer` is the safe consumption interface. `answer.kind` is one of six
values, and a result loaded from `result.export()` gives the same kind as the
live result of that execution:

| `kind` | Meaning | Carries |
|---|---|---|
| `point` | A complete scalar claim | `value` |
| `bounds` | A set-identified scalar | `bounds`, the identified set `(lower, upper)` over identified completions |
| `partial` | Partial identification or leftover structural mass: no scalar, and for a function-valued claim no unrestricted curve (read `result.envelope`) | `detail` (the limitation id), and `bounds` whenever the execution computed a scalar identified set |
| `response` | A function-valued claim (response curve, intervention response, derivative, Jacobian) | read `result.response` / `result.estimate` |
| `structured` | An executed claim with no single scalar, such as a multi-horizon temporal mediation grid | read its structured fields (`result.mediation_grid`) |
| `unavailable` | No claim: not identified, refused, not executed, non-finite, or not semantically accepted | `detail` (why) |

`bounds` and `partial` answers never expose an unrestricted scalar. Historical
fields (`effect`, `ate`, `posterior`, `response`) remain accessible for existing
callers. `.effect` / `.ate` emit a `UserWarning` when a point display would
misrepresent the claim; `ANTECEDENT_STRICT_ANSWER=1` raises instead. Displays
follow the same rule: the notebook card, `repr(result)`, `repr(result.estimate)`
and `repr(result.posterior)` of a partial result name the identification verdict
and the limitation instead of a mean and interval.

Every display of identification uses one verdict table: identified (with its
restriction, for example "identified under parametric restrictions"), partially
identified, graph-dependent, or not identified. The calibration row shows the
status, the reason code and the coverage record id, omitting any that are absent.

## Prepare explicitly when useful

```python
study = ant.prepare(data, graph=graph, query=query, seed=19)
report = study.inspect()        # includes cached identification; does not estimate
result = study.estimate()      # uses retained data and preparation's seed/threads
```

`ant.analyze(...)`, `ant.prepare(...)` and
`antecedent.estimation.PreparedAnalysis.prepare(...)` share one table of omitted
defaults, owned by the Rust study builder and readable as
`antecedent._native.omitted_defaults()`. A budget you omit (`bootstrap`,
`refute`, `n_draws`) is resolved there, and `latency=` selects the tier for the
omitted budgets only; an explicit value is never changed by a tier.

A study has two views. `study.inspect()` reports everything already known
(cached identification, the `contract` identities and reasoning slots, and a
`calibration` that is `unavailable` / `not_executed` until an estimate runs).
`study.preflight()` is the cheap structural-only view taken before
identification. `study.preview_transform(intent)` previews a transformation
(`compatible_data_replace`, `retarget`, `filter_population`,
`new_conditional_query`, `change_graph`, `display_precision`, …) without
executing anything.

A query owns its target population:
`ant.AverageEffect("t", "y", target_population=Treated())`.

`identify(...)` returns an `Identification` whose `statement`, `verdict`,
`qualified_verdict`, `assumption_statements`, and `derivation_statements` are
human-readable state on the object (`identification.to_dict()` includes them).
`qualified_verdict` keeps the restriction an identified verdict holds under;
`statement` uses it. That is readable identification data, not a notebook
renderer.

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
`data_snapshot_id` is the identity of the data the execution ran on. They are
different layers. A prepared inspect has a program and no claim.

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

Retargeting remains available from frozen scores of the execution it follows:
after `estimate(new_data)`, `retarget` reweights the scores fitted on `new_data`.
A nonconstant-weight retarget exports under a `RowWeights` target whose
`target_weights` identity binds the exact weight bits, their row count, the data
snapshot, the score table (`score_reuse`) and `depends_on`. The weights travel in
the artifact, so a consumer re-derives the identity and confirms the population
the answer is about; `inspect().target_weights_id` shows it. Constant weights of
any scale keep the original target. The binding belongs to the retargeted
result, not to the plan: the plan is still an AllObserved study and re-estimates
on new data, while `reexecute_retarget(artifact)` re-runs the carried weights and
raises `row_weights_bound_to_snapshot` on a different snapshot.
