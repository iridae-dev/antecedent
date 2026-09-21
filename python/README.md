# Antecedent for Python

> **2.0 release preparation:** installable package metadata remains 1.11.0
> until the release cut.

Antecedent exposes the same causal contract as the Rust facade. The root
namespace contains 56 names; see [the naming dictionary](../docs/api_naming.md).
Start with a query and graph, identify it, estimate on data, inspect the
claim, and export or refresh the study when appropriate.

```python
import antecedent as ant

query = ant.AverageEffect("treatment", "outcome")
result = ant.analyze(data, graph=graph, query=query)
print(result.answer)
print(result.inspect().to_dict())
```

For large adjustment problems, 2.0 adds configured DML, DR-Learner, and
causal-forest estimators. They use held-out nuisance predictions and disclose
their learner, folds, overlap, and uncertainty limitations; they do not make a
CATE interval appear where none was estimated.

Structural transport uses `antecedent.transport.Transport` with explicit
target and evidence information. Evidence availability is not inferred from a
column name or a selection label.

Read the [Python workflow](../docs/python-workflow.md), [supported analyses](../docs/supported-analyses.md), and [2.0 draft notes](../docs/release-notes/v2.0.0.md). The [support matrix](../docs/support-matrix.md) remains authoritative.
