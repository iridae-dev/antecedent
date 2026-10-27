# Refusals

Sometimes the correct causal result is not a number. A refusal is scientific information: it says what was asked, what is missing or incompatible, and, where one exists, the next meaningful option. Antecedent keeps the reasons distinct because they have different causes and different remedies.

## Categories

| Category | What it says | Next option |
| --- | --- | --- |
| Missing evidence | A formula may exist, but a required joint law, provider or declaration is absent | Supply the evidence the refusal names |
| Nonidentification | The declared structure and evidence do not determine the quantity | Different evidence, an experiment, or stronger assumptions you can defend. More rows do not help |
| Unavailable implementation | A meaningful request falls outside the implemented routes or the evidence-backed support surface | Choose a supported route, or a different question the available computation can answer |
| Exhausted search | A bounded search or enumeration stopped before finishing | A partial result, with the bound that stopped it and what was left unexamined; not a statement that no solution exists |
| Unlicensed execution status | A route can run, but without executing support evidence | The result runs and keeps that status; read it before relying on the answer |
| Operational failure | Cancellation, a resource limit, an invalid argument or a callback failure | Fix the call or the resources and rerun. This says nothing about the science |

A support failure is a further case: the effect may be identified while the actual data do not cover the requested intervention or population.

Information that is unavailable stays unavailable. It does not become an empty result, `NaN` or a generic error string.

## How a refusal reaches you

A valid question the declared structure cannot answer returns an `Outcome` carrying a typed refusal, not an exception:

```python
refused = ant.analyze(data, graph=confounded, query=query)
refused.accepted          # False
refused.answer.kind       # "refused"
refused.refusal.code      # for example "effect_not_identified"
refused.refusal.remedy    # what you can change, or None
```

The refusal's `code` is a registered reason; `detail` names the specific slot; `offending` names the quantity or identity it is about. Branch on these fields rather than parsing the message.

Operational failures raise subclasses of `ant.CausalError` instead. Invalid arguments, cancellation, resource limits and callback failures are not scientific outcomes, and they are never returned as one.

The [codes and remedies](../reference/refusal-codes.md) are listed in the reference.

## Describing what the data do say

A not-identified refusal is the answer to the causal question and is never replaced. If you also want to know what the data say and what evidence would change the answer, opt in:

```python
described = ant.analyze(data, graph=confounded, query=query, on_refusal="describe")
described.refusal              # the original refusal, unchanged
described.associational        # a separate outcome, labelled noncausal
described.evidence_plan        # evidence that would resolve the question
```

The associational result is not an answer to the causal question and ranks no action. Only identification-class refusals offer this; a search stopped by a budget, a cancellation or an invalid argument does not. See the [fallback guide](../guides/fallback.md).

## Reading limits

Each route states what it does not do, the request most likely to be refused, and the remedy. The [route limits page](../user-facing-limits.md) collects them for the transport and uncertainty routes. A refusal is not a failure of the data; it states which premise or declaration was not met. Do not add graph edges or assumptions only to make a refusal disappear.

## Selected routes that execute unlicensed

Some routes run without licensed evidence. They do not refuse, and they retain refused support status in the result. Successful execution does not upgrade the scientific claim: read support status in `result.inspect()` and, if your application accepts results, apply your own acceptance policy to it.
