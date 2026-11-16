# Antecedent for Python

Antecedent is a compiler and runtime for causal science. You declare a causal question, a causal graph or study design, and the data. Antecedent checks whether those declarations identify the quantity you asked for, compiles an analysis you can inspect before anything is fitted, and executes a supported route. Every answer carries its identification, support, uncertainty, calibration status and the methods it rests on. A question the declarations cannot answer returns a typed refusal that names what is missing.

This is the Python package. It exposes the same causal contract as the Rust crate [`antecedent`](https://crates.io/crates/antecedent), in Python's idiom. Documentation: <https://antecedent.readthedocs.io/>.

## Install

```bash
python -m pip install antecedent
```

Python 3.11 or later. Wheels cover CPython 3.11–3.14 on Linux (x86_64 and aarch64), macOS arm64 and Windows x86_64; other platforms need a [source build](https://antecedent.readthedocs.io/en/stable/start/installation/#build-from-source). NumPy and Pydantic are installed as dependencies.

## A first analysis

The example simulates a randomized treatment whose effect on the outcome is two units, so the estimate can be checked against a known truth. It prepares the analysis, inspects the plan, estimates the effect, repeats the analysis on changed data, and exports the result.

```python
import numpy as np
import antecedent as ant

rng = np.random.default_rng(17)
z = rng.normal(size=160)
treatment = rng.binomial(1, 0.5, size=160).astype(float)
data = {
    "treatment": treatment,
    "outcome": 2 * treatment + z + rng.normal(size=160),
    "z": z,
}
graph = [("treatment", "outcome"), ("z", "outcome")]
query = ant.AverageEffect("treatment", "outcome")

# Prepare: declare the question and check what it needs. Nothing is fitted yet.
study = ant.prepare(
    ant.data.table(data), graph=graph, query=query,
    inference=ant.Frequentist(bootstrap=25),
    execution=ant.Execution(seed=19),
)

# Inspect the compiled analysis before running it.
print(study.inspect())

# Execute, then read the answer with its uncertainty and identification.
result = ant.estimate(study)
print("Answer:", result.answer.kind, result.answer.value)
print("Uncertainty:", result.uncertainty.summary)
report = result.inspect().to_dict()
print("Identification:", report["identification"]["summary"])
for citation in result.citations():  # the published methods behind this answer
    print("Method:", citation.reference)
assert abs(result.answer.value - 2) < 0.5  # the simulated truth is 2

# Rebind: a new branch on changed observations. The earlier result is unchanged.
new_data = {**data, "outcome": data["outcome"] + treatment}
updated_study = ant.rebind(study, data=new_data)
updated = ant.estimate(updated_study)
print("Updated effect:", updated.answer.value)  # one unit larger

# Export the result and load it with an explicit expectation.
artifact = result.export()
loaded = ant.load(artifact, expected=artifact.expectation)
print("Artifact acceptance:", loaded.acceptance.status)
```

The estimate lands near two, and the updated branch is one unit larger. The bootstrap budget is small so the example runs quickly; read `result.inspect().calibration` before relying on interval coverage. [Your first analysis](https://antecedent.readthedocs.io/en/stable/start/first-analysis/) walks through each step, and [how to read a result](https://antecedent.readthedocs.io/en/stable/start/reading-a-result/) explains each field.

## How the pieces fit

- **Lifecycle.** `ant.prepare(...)` returns an immutable study; `ant.estimate(study)` returns an `Outcome`; `ant.analyze(...)` is the one-shot form. `ant.rebind(study, data=...)` and `ant.reconfigure(study, ...)` return new studies and change nothing earlier. `result.export()` produces an artifact that `ant.load(artifact, expected=artifact.expectation)` reads back.
- **Settings.** Seed, threads, cancellation and resource limits belong to `ant.Execution`; the bootstrap count or posterior draws belong to `ant.Frequentist(...)` / `ant.Bayesian(...)`.
- **Refusals.** A valid question the declared structure cannot answer returns an `Outcome` with a typed `refusal` (add `on_refusal="describe"` for a labelled associational description beside it); invalid arguments and operational failures raise.
- **Support.** A route runs when it has provenance and an independent scientific check; otherwise it returns a `not_supported` refusal that names the missing evidence. The [support matrix](https://antecedent.readthedocs.io/en/stable/support-matrix/) lists the supported units; [how Antecedent checks its scientific methods](https://antecedent.readthedocs.io/en/stable/guides/scientific-verification/#the-evidence-model) explains the model.
- **Calibration.** Every reported interval carries its calibration status (`calibrated`, `outside_measured_scope`, `measured_below_nominal` or `not_measured`), which annotates the interval and never decides whether a route runs.
- **Citations.** Every result names the provenance records behind the route it executed, so its citations travel with it: `result.citations()` for one result, `antecedent.citations.bibliography(...)` for a whole study. See [how to cite the methods behind a study](https://antecedent.readthedocs.io/en/stable/guides/citing-a-study/).

The root namespace has an explicit exported-name contract; see [the naming dictionary](https://antecedent.readthedocs.io/en/stable/api_naming/).

## Beyond one effect

| Task | Where |
| --- | --- |
| Large adjustment problems | Configured DML, DR-Learner and causal-forest estimators. The standard wheel includes the CPU-native `antecedent.learners.NeuralNet` nuisance learner; no separate Python extra is needed. These estimators use held-out nuisance predictions and disclose their learner, folds, overlap and uncertainty limitations; they do not make a CATE interval appear where none was estimated. |
| Transport to another population | `antecedent.transport.Transport`, with explicit target and evidence information. Evidence availability is not inferred from a column name or a selection label. |
| Design-family studies | `antecedent.experiment`, `antecedent.policy`, `antecedent.quasi`, `antecedent.survival` and `antecedent.regimes` prepare and analyze as retained studies. On the experiment, policy, regime and interference queries, row-aligned inputs may be named by a data column; `prepare` resolves those names and freezes the design. Their interval claims are rows of the [design-family support matrix](https://antecedent.readthedocs.io/en/stable/graphless-support-matrix/), separate from the geometric matrix. |
| Main effects and interaction of a 2×2 factorial | `antecedent.factorial.estimate`, a separate point utility, not a retained study. |

## Documentation

- [Your first analysis](https://antecedent.readthedocs.io/en/stable/start/first-analysis/) and [how to read a result](https://antecedent.readthedocs.io/en/stable/start/reading-a-result/)
- [Supported analyses](https://antecedent.readthedocs.io/en/stable/supported-analyses/): choose a question and an estimator
- [Python API](https://antecedent.readthedocs.io/en/stable/python-api/) and [Python workflow options](https://antecedent.readthedocs.io/en/stable/python-options/)
- [Examples](https://antecedent.readthedocs.io/en/stable/examples/) and the tutorials listed on the [documentation home](https://antecedent.readthedocs.io/en/stable/)
- Upgrading from 2.4? The [migration guide](https://antecedent.readthedocs.io/en/stable/migration/) gives every old and new form.

## License

MIT OR Apache-2.0.
