# Is my analysis supported?

Support depends on the question, graph, how the graph was obtained, inference
method, and validation settings. In these docs, **licensed** means that this
combination has a supported execution path with recorded evidence and limits.
It does not establish that your graph is correct or your data are adequate.

Start with the closest worked example, then check its exact settings:

| Your question | Example or guide | Check before using it |
|---|---|---|
| What is the average effect? | [Python quickstart](python-workflow.md) | Graph assumptions and data support |
| How can I adjust for treatment selection? | [Weighting example](examples.md#python-scripts) | Propensity overlap and target population |
| What if edge directions remain uncertain? | [Partial-graph example](examples.md#python-scripts) | Whether the answer is bounds or partial |
| How does the outcome change with dose? | [Causal responses](causal-responses.md) | Support over the requested dose range |
| How does an intervention act over time? | [Temporal examples](examples.md#python-scripts) | Graph lags, requested horizons, and uncertainty |
| Can I reuse earlier evidence? | [Prior bank](priors.md) | Compatibility of the source and target analyses |
| Is the outcome censored or selected? | [Observation contract](observation-contract.md) | The mechanism and its separate identifying assumptions |
| Can I transfer effects or model interference? | [Transport and interference](transport-interference.md) | The required design assumptions |

These are starting points, not permission to combine arbitrary options.
The [full support matrix](support-matrix.md#licensed-cells) is the authoritative
list. Its [refusal reasons](support-matrix.md#refusal-reasons) explain excluded
combinations. `n/a` means the combination does not define a valid question.

For an analysis you have configured, `ant.prepare(...)` checks identification
without estimating. Use `study.inspect()` to read what is known before calling
`study.estimate()`. Preparation itself can refuse an unsupported request; see
[handling refusals](python-workflow.md#when-an-analysis-is-refused).
