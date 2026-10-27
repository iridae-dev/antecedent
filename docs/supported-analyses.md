# Choose a question and an estimator

Start from the question you want answered, then check that the combination of question, graph, inference method and validation setting is supported. Support depends on all four, and on how the graph was obtained.

**Licensed** means that a combination has a supported execution path with recorded evidence and limits. It does not establish that your graph is correct or your data are adequate, and it does not mean the cell's interval coverage was measured: the [support matrix](support-matrix.md) counts the licensed cells that have no coverage measurement for their estimator. Use [support lookup](support-lookup.md) to filter exact graph-based cells, or [the graphless matrix](graphless-support-matrix.md) for design-family studies.

| Your question | Start here | Check before using it |
|---|---|---|
| What is the average effect? | [Your first analysis](start/first-analysis.md) | Graph assumptions and data support |
| How can I adjust for treatment selection? | [Weighting example](examples.md) | Propensity overlap and target population |
| Treatment has more than two levels | [Estimate an effect when treatment has several levels](guides/treatment-levels.md) | Declared levels and the population each contrast averages over |
| What if edge directions remain uncertain? | [Partial-graph example](examples.md) | Whether the answer is bounds or partial |
| How does the outcome change with dose, or what is its distribution? | [Estimate responses, derivatives and outcome distributions](causal-responses.md) | Support over the requested dose range |
| Who benefits, and what would a treatment rule be worth? | [Heterogeneous effects and treatment policies](guides/heterogeneous-effects-and-policies.md) | Overlap, the policy's origin, and what the standard error conditions on |
| How does an intervention act over time? | [Temporal examples](examples.md) | Graph lags, requested horizons, and uncertainty |
| Can I reuse earlier evidence? | [Prior bank](priors.md) | Compatibility of the source and target analyses |
| Is the outcome censored, selected or missing? | [Observation contract](observation-contract.md) and [missing values](guides/missing-data.md) | The mechanism and its separate identifying assumptions |
| Do rows share clusters, or come from a weighted sample? | [Clusters](guides/clustered-dml.md) and [sampling weights](guides/sampling-weights.md) | The independence unit and the weighted population |
| Can I fit nuisance models with DML, DR-Learner, or a causal forest? | [Estimator options](python-options.md) and [penalized propensity](guides/penalized-aipw.md) | The estimator is licensed for that graph, inference, and validation cell; a forest does not add a CATE interval that was not estimated |
| How sensitive is the answer to an assumption I cannot test? | [Refute the reported quantity](guides/target-native-validation.md) and [probe sensitivity](sensitivity-and-robustness.md) | The range is an assumption range, not a confidence interval |
| The effect is not identified. What can I say? | [Describe the data and plan evidence](guides/fallback.md) | The description is not an answer to the causal question |
| Can I transfer effects or model interference? | [Carry an effect to another population](guides/transport/carry-an-effect.md) and [saturation and interference designs](guides/designs/interference-and-network-exposure.md) and `antecedent.transport.Transport` | Target, evidence, and the [support matrix](support-matrix.md) |
| Did I run a randomized or factorial experiment, evaluate a policy, or run a quasi-experimental, survival, or longitudinal-regime design? | [Capabilities](capabilities.md) and the family stage modules (`antecedent.experiment`, `antecedent.factorial`, `antecedent.policy`, `antecedent.quasi`, `antecedent.survival`, `antecedent.regimes`) | The [graphless support matrix](graphless-support-matrix.md) is the license for these claims; a matching row licenses that claim but is not a coverage record, and a design can still return a point when no row matches |

These are starting points, not permission to combine arbitrary options. The [full support matrix](support-matrix.md#licensed-cells) is the authoritative list. Its [refusal reasons](support-matrix.md#refusal-reasons) explain excluded combinations. `n/a` means the combination does not define a valid question.

For an analysis you have configured, `ant.prepare(...)` checks identification without estimating. Use `study.inspect()` to read what is known before calling `ant.estimate(study)`. Preparation itself can refuse an unsupported request; see [handling refusals](start/first-analysis.md#when-an-analysis-is-refused).

Specialized decision, ranking, recalculation and measured-inference adapters
have their own checked contracts; they do not add arbitrary combinations to the
analysis grid. Start with the [decision contract](decisions.md) and consult the
[measured interval scopes](population-time-uncertainty.md).
