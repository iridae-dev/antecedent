# Reading a result

A result is a claim: an answer together with what it was asked, what it assumed, what evidence supported it and where it stops. This page shows where each part lives on the `Outcome` that `ant.estimate` and `ant.analyze` return. The examples continue [your first analysis](first-analysis.md), whose data are simulated with a known effect of 2.

Read in this order: the kind of answer, identification, uncertainty, calibration status, support status, diagnostics, then the limits. The condition under which each is true is a property of that part, not of the whole result.

## The answer

Start with `result.answer.kind` before extracting a number.

| Kind | What you have | Where to look |
|---|---|---|
| `point` | One scalar estimate | `answer.value` |
| `bounds` | A range of effects allowed by the identified structures | `answer.bounds` |
| `partial` | An answer limited by incomplete identification | `answer.detail`, any `answer.bounds`, and `result.envelope` |
| `response` | A curve, surface or derivative | `result.response` / `result.estimate` |
| `structured` | Several related results, such as temporal mediation | The query's fields, such as `result.mediation_grid` |
| `unavailable` | No usable claim | `answer.detail` |

`result` is an `Outcome`. The specialized result a query produces is `result.value`, and its family-specific fields are read there. `answer`, `inspect()`, `uncertainty`, `support`, `assumptions`, `diagnostics`, `provenance`, `identities`, `export()` and `study` are read on the outcome itself. Bounds describe what the assumptions determine; they are not a confidence interval. A `partial` answer does not supply an unrestricted scalar or curve.

## Identification

`result.inspect()` reports whether the declared graph, design and evidence determine the requested quantity, and under which assumptions:

```python
report = result.inspect().to_dict()
print(report["identification"]["summary"])
print(result.assumptions)
```

Identification depends on what you declared. More observations cannot repair a missing causal assumption, and a successful run does not show that the graph is true.

## Uncertainty

`result.uncertainty` states what kind of uncertainty is reported. Three different things can be called a range:

- a **sampling interval** describes variation across repeated samples under the method's assumptions;
- a **structural range** describes variation over graphs or other structural alternatives;
- a **sensitivity range** describes how far the answer could move under a stated departure from an assumption.

These are not interchangeable. Read how each interval was formed from the executed result's bindings, which report what the execution did rather than what was requested:

```python
for binding in result.value.interval_bindings:
    print(binding.method, binding.level, binding.dependence, binding.replicates_ok)
```

`method` names the construction (`analytic_se`, `bootstrap_se`, `circular_block_se`, `posterior_quantile`, `identified_set`, `anderson_rubin` and others, or `none`), and `dependence` names the dependence rule (`iid`, `cluster`, `serial_hac` and so on). `level` is `None` when no interval was formed. See [uncertainty](../concepts/uncertainty.md).

## Calibration status

`result.inspect().calibration.status` annotates the reported interval with whether a measured coverage record supports it. The interval is reported whatever the status says.

| Status | Meaning |
|---|---|
| `calibrated` | A matching coverage record still attests the current code and includes this execution's scope |
| `outside_measured_scope` | A record matches the construction, but this execution is outside its measured range, replicates, draws or unidentified mass, or the record no longer attests the current code |
| `measured_below_nominal` | The governing record is a boundary record: it measured coverage below nominal at this sample size (`observed_coverage` carries it) |
| `not_measured` | No coverage record measured this construction, level or identification, or no interval was reported; read the reason code |

An implemented interval, an identified effect or a passing diagnostic does not establish calibration. Calibration is recorded per method and scope, so when the status is not `calibrated`, read the reason code. [Assurance](../concepts/assurance.md) describes what a coverage record holds.

## Support status

`result.support` records the empirical support for the answer, for example overlap or sample size for the target population. A route is supported when a named combination of question, structure, evidence, inference and estimator has provenance and independent scientific checks. A route that is not supported refuses in release builds; a development build runs it and reports `would_refuse` beside the result. The calibration status of an interval (`calibrated`, `outside_measured_scope`, `measured_below_nominal` or `not_measured`) is an annotation, not a condition for running. Successful execution does not upgrade the scientific claim. Look a combination up in [support lookup](../support-lookup.md).

## Diagnostics

`result.diagnostics` assesses the data and the estimator's fit: balance, overlap, residual checks and similar. Diagnostics can show a problem. A passing diagnostic does not prove the causal model.

## Limits

A result also records what it does not establish. Read the assumptions that remain active, the intervals that are not calibrated, and any `partial` or `bounds` answer. An exported result preserves its contract; the loaded `acceptance` states which identity or replay checks were accepted, not whether the causal assumptions are true. See [results are claims](../concepts/results-are-claims.md).

## Refusals and raised errors

A valid question that the declared structure cannot answer returns an `Outcome` with a typed `refusal`: `result.accepted` is false and `result.refusal.code` names the reason. The refusal says what was asked, what is missing and what would resolve it. Missing evidence, nonidentification, an unavailable implementation, an exhausted search and an operational failure are different conditions with different remedies.

| Reason | What to inspect next |
|---|---|
| `effect_not_identified` | Review the graph and missing causal assumptions. More rows alone cannot fix identification. |
| `estimator_inference_mismatch` | Check whether the selected estimator supports the requested inference method. |
| `population_not_estimable` | Check the estimator's supported target populations. Changing the population changes the question. |
| `score_table_unavailable` | Retargeting needs prepared scores; check the supported retargeting workflow. |

Invalid arguments, cancellation, resource limits and callback failures raise `ant.CausalError` subclasses instead of returning an outcome. Opt in with `on_refusal="describe"` to keep the refusal as the primary outcome and read a separately labelled `associational` description and an `evidence_plan` beside it. See [refusals](../concepts/refusals.md) and [the fallback guide](../guides/fallback.md).
