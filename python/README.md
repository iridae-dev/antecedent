# Antecedent for Python

> Install from PyPI: `python -m pip install antecedent`. `calibrated` still requires a coverage record that attests the installed build.

Antecedent exposes the same causal contract as the Rust facade. The root namespace has an explicit exported-name contract; see [the naming dictionary](../docs/api_naming.md). Start with a query and graph, prepare and identify it, estimate on data, inspect the claim, and export the outcome or branch the study with `ant.rebind` / `ant.reconfigure` when appropriate.

```python
import antecedent as ant

query = ant.AverageEffect("treatment", "outcome")
study = ant.prepare(data, graph=graph, query=query, execution=ant.Execution(seed=19))
result = ant.estimate(study)  # an Outcome; ant.analyze(...) is the one-shot form
print(result.answer)
print(result.inspect().to_dict())

branch = ant.rebind(study, data=new_data)  # a new immutable study; nothing earlier changes
artifact = result.export()  # ant.load(bytes(artifact), expected=artifact.expectation)
```

Seed, threads, cancellation and resource limits belong to `ant.Execution`; the bootstrap count or posterior draws belong to `ant.Frequentist(...)` / `ant.Bayesian(...)`. A valid question the declared structure cannot answer returns an `Outcome` with a typed `refusal` (add `on_refusal="describe"` for a labelled associational description beside it); invalid arguments and operational failures raise. See the [migration guide](../docs/migration.md) for the removed spellings.

For large adjustment problems, Antecedent provides configured DML, DR-Learner, and causal-forest estimators. The standard wheel includes the CPU-native `NeuralNet` nuisance learner; no separate Python extra is needed. These estimators use held-out nuisance predictions and disclose their learner, folds, overlap, and uncertainty limitations; they do not make a CATE interval appear where none was estimated.

Structural transport uses `antecedent.transport.Transport` with explicit target and evidence information. Evidence availability is not inferred from a column name or a selection label.

Design-family studies live on their own stage modules — `antecedent.experiment`, `antecedent.policy`, `antecedent.quasi`, `antecedent.survival`, and `antecedent.regimes` — that prepare and analyze as retained studies. `antecedent.factorial.estimate` is a separate point utility, not a retained study. On the experiment, policy, regime, and interference queries, row-aligned inputs may be named by a data column; `prepare` resolves those names and freezes the design. Their interval claims are rows of the [graphless support matrix](../docs/graphless-support-matrix.md), separate from the geometric matrix.

Read the [first analysis](../docs/start/first-analysis.md), [supported analyses](../docs/supported-analyses.md) and [how to read a result](../docs/start/reading-a-result.md). The [support matrix](../docs/support-matrix.md) remains authoritative.
