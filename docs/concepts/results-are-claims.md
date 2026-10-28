# Results are claims

A result of `0.23` has almost no scientific meaning by itself. A claim says what was changed, for whom, relative to what, under which structure, how the quantity was identified and estimated, what data supported it, what uncertainty it represents and which assumptions remain active. Antecedent returns claims. Read the claim before projecting it to a number, a report or an API payload.

```python
result = ant.analyze(data, graph=graph, query=query)
report = result.inspect()
report.identification, report.support, report.uncertainty, report.assumptions, report.calibration
```

`result.answer` is the answer; `result.inspect()` is the claim around it. This sketch names the slots; the [first analysis](../start/first-analysis.md) shows them with output.

## The parts of a claim

| Part | What it says |
| --- | --- |
| Answer | The value, bounds, response or structured result, and its kind |
| Target | The estimand: the quantity the question asks for, for a stated population and time |
| Assumptions | What was taken as given |
| Evidence | The data, experiments or declared laws behind it, and their support for the requested coordinates |
| Uncertainty | Which sources are represented, how an interval was built and whether a calibration record supports it |
| Limits | What the claim does not establish |

## Answer shapes

| Kind | Meaning |
| --- | --- |
| `point` | A scalar claim under its recorded contract |
| `bounds` | A range determined by the declared structural knowledge |
| `partial` | A claim with unresolved identification or structural components |
| `response` | A curve, surface, derivative or intervention response |
| `structured` | Several linked results |
| `associational` | A noncausal description, labelled as such |
| `refused` | No claim; a typed refusal states why |
| `unavailable` | No usable claim, with a reason instead of fabricated precision |

Bounds are not automatically confidence intervals, and a confidence interval does not automatically represent graph uncertainty.

## Identified and model-conditional

An **identified** effect follows from the stated assumptions and evidence through an implemented identification route. A **model-conditional** response follows from a specified or fitted model: a cyclic equilibrium, a fitted joint response, a declared structural model. A model-conditional answer can be exact and useful; it does not say the design identifies the real-world effect. A fitted response, for example, does not establish identification of a whole intervention program that the data only assigned period by period.

## The standing vocabulary

These words name different checks. Each applies to a particular object, and none stands in for the others. Read them together in `result.inspect()`.

| Term | What it lets you say |
| --- | --- |
| Implemented | The capability exists in the build you are running |
| Executes | The declared request can run |
| Identified | Stated assumptions and evidence determine the target within the named identification scope |
| Model conditional | The result follows from the specified or fitted model |
| Licensed | The named combination of question, structure, evidence, inference and estimator has executing support evidence. The word is unrelated to a software license |
| Independently checked | A named calculation or proposition was compared with an appropriate independent reference |
| Calibrated | A matching measured construction and scope still applies to the executing code |
| Replayed | A specified calculation was reproduced under the consumer's procedure |
| Accepted artifact | Specified identity, interpretation or replay checks passed for a saved result |
| Current authority | A supported workflow checked the current standing of its sources for present use |
| Numerically certified | A named numerical proposition was established within its domain |

Two further distinctions come up constantly. A *frozen output* of Antecedent checks change against a recorded result; it is not an independent oracle. An *accepted artifact* means the result's contract and identities checked out; it does not mean the causal assumptions are true.

For a worked account of the evidence behind these distinctions, read [How Antecedent checks its scientific methods](../guides/scientific-verification.md).

## Successful execution does not upgrade the claim

A route can execute and still be outside licensed support, still model-conditional, still uncalibrated. In those cases the result keeps that status in its report and in exported artifacts. When you build on a result, read the status of the route that actually ran. The [support matrix](../support-matrix.md) and [support lookup](../support-lookup.md) say which combinations have executing evidence; [assurance](assurance.md) says what each kind of evidence establishes.

## What a claim is not

- Not a statement that the causal model describes reality. Antecedent aims not to make a stronger claim than the declared assumptions, evidence, implemented methods and data warrant.
- Not a number that survives serialization by itself. An exported result preserves its contract; loading it recreates the report, not the dataset or a live study.
- Not a diagnostic of your graph. A passing diagnostic and an identified effect do not establish calibration.
