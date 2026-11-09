# Antecedent for Python

Antecedent is a compiler and runtime for causal science. Declare a question, a model or study design, and the evidence; Antecedent checks what the question needs, compiles an analysis you can inspect, and executes a supported route. Every answer carries its identification, support, uncertainty, calibration status and the methods it rests on.

```bash
python -m pip install antecedent
```

Wheels cover CPython 3.11–3.14 on Linux (x86_64 and aarch64), macOS arm64 and Windows x86_64. The Python package exposes the same causal contract as the Rust facade, in Python's idiom; the root namespace has an explicit exported-name contract (see [the naming dictionary](https://antecedent.readthedocs.io/en/stable/api_naming/)).

Start with a query and graph, prepare and identify it, estimate on data, inspect the claim, and export the outcome or branch the study when the observations or declarations change. With a table `data`, a list of graph edges `graph` and later observations `new_data`:

```python
import antecedent as ant

query = ant.AverageEffect("treatment", "outcome")
study = ant.prepare(data, graph=graph, query=query, execution=ant.Execution(seed=19))
result = ant.estimate(study)  # an Outcome; ant.analyze(...) is the one-shot form
print(result.answer)
print(result.inspect().to_dict())
for citation in result.citations():  # the published methods behind this answer
    print(citation)

branch = ant.rebind(study, data=new_data)  # a new immutable study; nothing earlier changes
artifact = result.export()  # ant.load(bytes(artifact), expected=artifact.expectation)
```

Seed, threads, cancellation and resource limits belong to `ant.Execution`; the bootstrap count or posterior draws belong to `ant.Frequentist(...)` / `ant.Bayesian(...)`. A valid question the declared structure cannot answer returns an `Outcome` with a typed `refusal` (add `on_refusal="describe"` for a labelled associational description beside it); invalid arguments and operational failures raise.

A route runs when it has provenance and an independent scientific check; otherwise it returns a `not_supported` refusal that names the missing evidence. Every reported interval carries its calibration status (`calibrated`, `outside_measured_scope`, `measured_below_nominal` or `not_measured`), which annotates the interval and never decides whether a route runs. Every result names the provenance records behind the route it executed, so its citations travel with it: `result.citations()` for one result, `antecedent.citations.bibliography(...)` for a whole study.

For large adjustment problems, Antecedent provides configured DML, DR-Learner, and causal-forest estimators. The standard wheel includes the CPU-native `antecedent.learners.NeuralNet` nuisance learner; no separate Python extra is needed. These estimators use held-out nuisance predictions and disclose their learner, folds, overlap, and uncertainty limitations; they do not make a CATE interval appear where none was estimated.

Structural transport uses `antecedent.transport.Transport` with explicit target and evidence information. Evidence availability is not inferred from a column name or a selection label.

Design-family studies live on their own stage modules — `antecedent.experiment`, `antecedent.policy`, `antecedent.quasi`, `antecedent.survival`, and `antecedent.regimes` — that prepare and analyze as retained studies. `antecedent.factorial.estimate` is a separate point utility, not a retained study. On the experiment, policy, regime, and interference queries, row-aligned inputs may be named by a data column; `prepare` resolves those names and freezes the design. Their interval claims are rows of the [graphless support matrix](https://antecedent.readthedocs.io/en/stable/graphless-support-matrix/), separate from the geometric matrix.

Read the [first analysis](https://antecedent.readthedocs.io/en/stable/start/first-analysis/), [supported analyses](https://antecedent.readthedocs.io/en/stable/supported-analyses/), [how to read a result](https://antecedent.readthedocs.io/en/stable/start/reading-a-result/) and [how to cite the methods behind a study](https://antecedent.readthedocs.io/en/stable/guides/citing-a-study/). The [support matrix](https://antecedent.readthedocs.io/en/stable/support-matrix/) lists the supported units; [how Antecedent checks its scientific methods](https://antecedent.readthedocs.io/en/stable/guides/scientific-verification/#the-evidence-model) explains the model. Upgrading from 2.4? The [migration guide](https://antecedent.readthedocs.io/en/stable/migration/) gives every old and new form.
