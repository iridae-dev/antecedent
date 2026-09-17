# Your first Python analysis

Estimate an effect, check the answer, and reuse the analysis on new data.
This guide uses Antecedent 1.10.

## Install

You need Python 3.11 or later. Install Antecedent from PyPI:

```bash
python -m pip install antecedent
```

NumPy is installed with Antecedent. If you already have an older version, use
`python -m pip install --upgrade antecedent`.

## Estimate a known effect

In this simulated experiment, treatment increases the outcome by 2 units.
A baseline measurement, `z`, also affects the outcome. Treatment is randomly
assigned, so `z` does not determine who receives it.

Copy this whole block into Python:

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

result = ant.analyze(data, graph=graph, query=query, seed=19, bootstrap=25)
print("Answer type:", result.answer.kind)
print("Estimated effect:", result.answer.value)
print("Calibration:", result.calibration.status)
```

Expect a `point` answer with an estimated effect close to **2**. Sampling noise
means it will not be exactly 2. The small bootstrap budget keeps this example
quick; it is not a recommended budget for a final analysis.

The graph states your causal assumptions. Antecedent checks whether those
assumptions allow the effect to be estimated before fitting it. A successful
run does not prove the graph is correct.

## Read the answer

Start with `result.answer.kind` before extracting a number:

| Kind | What you have | Where to look |
|---|---|---|
| `point` | One scalar estimate | `answer.value` |
| `bounds` | A range of effects allowed by the identified structures | `answer.bounds` |
| `partial` | An answer limited by incomplete identification | `answer.detail`, any `answer.bounds`, and `result.envelope` |
| `response` | A curve, surface, or derivative | `result.response` / `result.estimate` |
| `structured` | Several related results, such as temporal mediation | The query's fields, such as `result.mediation_grid` |
| `unavailable` | No usable claim | `answer.detail` |

Bounds describe what the assumptions determine; they are not a confidence
interval. A `partial` answer does not supply an unrestricted scalar or curve.
Use the answer interface rather than legacy `.effect` or `.ate` fields, which
can warn when a scalar would misrepresent the result.

For the reasoning and diagnostics behind the answer:

```python
report = result.inspect().to_dict()
print(result.inspect())
```

Read identification, data support, uncertainty, and assumptions together.
Calibration describes the evidence for the reported interval:

| Status | Meaning |
|---|---|
| `calibrated` | A matching coverage study includes this execution's scope |
| `scope_not_assessed` | A record matches, but does not assess this execution's scope, or records a boundary |
| `unavailable` | No coverage record is available; read the reason code |

An identified effect or a passing diagnostic does not establish calibration.

## Run the same analysis on new data

Keep `result.study` to reuse the graph, question, and settings. Here we simulate
an outcome whose treatment effect is one unit larger:

```python
new_data = {**data, "outcome": data["outcome"] + treatment}
study = result.study
updated = study.refresh(new_data)
print("Original:", result.answer.value)
print("Updated:", updated.answer.value)
```

The updated estimate should be one unit larger. The original result stays
unchanged. Refresh replaces the study's data only after a successful run;
new data must have the same schema.

## Save and reload a result

```python
encoded = result.export()
loaded = ant.load(encoded)
print("Accepted:", loaded.acceptance.verified)
print(loaded.answer)
```

The loaded result should be accepted and have the same answer. Acceptance checks
the saved result's contract, not whether its causal assumptions are true.
Loading restores the report, not the dataset or a live study for new estimates.

## When an analysis is refused

A refusal explains why the requested analysis cannot run. Keep the explanation
visible rather than replacing the answer with a default value:

```python
try:
    updated = study.refresh(new_data)
except ant.CausalError as error:
    print(error)
    print(error.report.to_dict())
```

| Reason | What to inspect next |
|---|---|
| `EffectNotIdentified` | Review the graph and missing causal assumptions. More rows alone cannot fix identification. |
| `estimator_inference_mismatch` | Check whether the selected estimator supports the requested inference method. |
| `population_not_estimable` | Check the estimator's supported target populations. Changing the population changes the question. |
| `score_table_unavailable` | Retargeting needs prepared scores; check the supported retargeting workflow. |

Do not add graph directions or assumptions solely to make a refusal disappear.
See [supported analyses](supported-analyses.md) and the
[workflow reference](python-options.md) for the relevant options.

## Next steps

- [Choose an example](examples.md) for weighting, discovery, or temporal effects.
- [Inspect before estimating](python-options.md#prepare-explicitly-when-useful).
- [Configure populations, likelihoods, and overlap](python-options.md).
- [Browse the Python API](python-api.md).
