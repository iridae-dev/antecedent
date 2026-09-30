# Antecedent for Python

> Install from PyPI: `python -m pip install antecedent`. `calibrated` still requires a coverage record that attests the installed build.

Antecedent exposes the same causal contract as the Rust facade. The root namespace contains 64 names; see [the naming dictionary](../docs/api_naming.md). Start with a query and graph, identify it, estimate on data, inspect the claim, and export or refresh the study when appropriate.

```python
import antecedent as ant

query = ant.AverageEffect("treatment", "outcome")
result = ant.analyze(data, graph=graph, query=query)
print(result.answer)
print(result.inspect().to_dict())
```

For large adjustment problems, Antecedent provides configured DML, DR-Learner, and causal-forest estimators. The standard wheel includes the CPU-native `NeuralNet` nuisance learner; no separate Python extra is needed. These estimators use held-out nuisance predictions and disclose their learner, folds, overlap, and uncertainty limitations; they do not make a CATE interval appear where none was estimated.

Structural transport uses `antecedent.transport.Transport` with explicit target and evidence information. Evidence availability is not inferred from a column name or a selection label.

2.1 adds design-family studies on their own stage modules — `antecedent.experiment`, `antecedent.policy`, `antecedent.quasi`, `antecedent.survival`, and `antecedent.regimes` — that prepare and analyze as retained studies. `antecedent.factorial.estimate` is a separate point utility, not a retained study. On the experiment, policy, regime, and interference queries, row-aligned inputs may be named by a data column; `prepare` resolves those names and freezes the design. Their interval claims are rows of the [graphless support matrix](../docs/graphless-support-matrix.md), separate from the geometric matrix.

Read the [Python workflow](../docs/python-workflow.md), [supported analyses](../docs/supported-analyses.md), and [2.1.0 release notes](../docs/release-notes/v2.1.0.md). The [support matrix](../docs/support-matrix.md) remains authoritative.
