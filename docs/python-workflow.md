# Your first Python analysis

Estimate an effect, check the answer, and reuse the analysis on new data. This guide is the 2.0.0 analysis: a graph, a query, `analyze`, then `answer.kind`, inspection, and calibration. Transport uses those same verbs. Callers moving names from the published 1.11 release should read the [transport migration](migrations/2.0-transport-day1.md).

## Install

You need Python 3.11 or later. Install Antecedent from PyPI:

```bash
python -m pip install antecedent
```

NumPy is installed with Antecedent. If you already have an older version, use `python -m pip install --upgrade antecedent`.

## Estimate a known effect

In this simulated experiment, treatment increases the outcome by 2 units. A baseline measurement, `z`, also affects the outcome. Treatment is randomly assigned, so `z` does not determine who receives it.

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

Expect a `point` answer with an estimated effect close to **2**. Sampling noise means it will not be exactly 2. The small bootstrap budget keeps this example quick; it is not a recommended budget for a final analysis.

The graph states your causal assumptions. Antecedent checks whether those assumptions allow the effect to be estimated before fitting it. A successful run does not prove the graph is correct.

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

Bounds describe what the assumptions determine; they are not a confidence interval. A `partial` answer does not supply an unrestricted scalar or curve. Use the answer interface rather than legacy `.effect` or `.ate` fields, which can warn when a scalar would misrepresent the result.

For the reasoning and diagnostics behind the answer:

```python
report = result.inspect().to_dict()
print(result.inspect())
```

Read identification, data support, uncertainty, and assumptions together. Calibration describes the evidence for the reported interval:

| Status | Meaning |
|---|---|
| `calibrated` | A matching coverage record still attests the current code and includes this execution's scope |
| `scope_not_assessed` | A record matches, but does not assess this execution's scope, records a boundary, or no longer attests the current code |
| `unavailable` | No coverage record is available; read the reason code |

An identified effect or a passing diagnostic does not establish calibration.

Every licensed analysis outside the transport day-1 views retains a reusable study and exports a contracted execution; custom validator results travel as caller-attested, not re-verifiable, evidence, and a row-weight retarget re-executes only on its own data snapshot.

Every reported interval states its calibration: `calibrated` only when a coverage record matches the execution, the execution is inside that record's scope, and the record still attests the current code; `scope_not_assessed` when a record matches but the execution is outside its scope, the record is a boundary, or the record is stale or non-attesting; `unavailable` with a reason code when no record exists.

Identities are distinct and stable: every `IdentityDomain` plus `target_weights` is domain-separated and registered in `parity/identity.toml`.

## Run the same analysis on new data

Keep `result.study` to reuse the graph, question, and settings. Here we simulate an outcome whose treatment effect is one unit larger:

```python
new_data = {**data, "outcome": data["outcome"] + treatment}
study = result.study
updated = study.refresh(new_data)
print("Original:", result.answer.value)
print("Updated:", updated.answer.value)
```

The updated estimate should be one unit larger. The original result stays unchanged. Refresh replaces the study's data only after a successful run; new data must have the same schema.

## Save and reload a result

```python
encoded = result.export()
loaded = ant.load(encoded)
print("Acceptance:", loaded.acceptance.status)
print(loaded.answer)
```

The loaded result keeps the same answer. `acceptance.status` is `verified` when the consumer replayed a verified program from the bytes alone, or `sealed` when the route ran through a sealed checked operation: loading then recognizes the artifact, verifies its contract and identities, keeps the recorded answer, and names the operation it cannot replay in `acceptance.unresolved`. `acceptance.verified` stays reserved for fully replayable programs. Acceptance checks the saved result's contract, not whether its causal assumptions are true. Loading restores the report, not the dataset or a live study for new estimates.

## When an analysis is refused

A refusal explains why the requested analysis cannot run. Keep the explanation visible rather than replacing the answer with a default value:

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

Do not add graph directions or assumptions solely to make a refusal disappear. See [supported analyses](supported-analyses.md) and the [workflow reference](python-options.md) for the relevant options.

## Transport the same way

Transport is the same `analyze` call. Its `data` and `graph` are not the quickstart's: `data` holds `price`, `sales`, and `preference` columns for the trial, and `graph` is an `Admg` over those three variables, for example `ant.Admg.from_edges(["price", "sales", "preference"], [("price", "sales"), ("preference", "sales")])`.

Wrap an ordinary question. Source identity, intervention regime, and sampling are scientific claims on `evidence`; the table only supplies columns and a snapshot digest.

```python
evidence = ant.transport.Evidence(
    source=ant.transport.Source(
        "trial", kind="experimental", interventions=["price"], sampling="independent",
    ),
    target_sampling="representative_sample",
)
query = ant.transport.Transport(
    ant.ResponseCurve("price", "sales", grid=[8, 9, 10, 11, 12]),
    target="new_market",
    evidence=evidence,
    selections=["preference"],
)
result = ant.analyze(data, graph=graph, query=query)
print(result.answer.kind, result.inspect().support.summary)
```

`EmpiricalTable` is the default provider. `LearnedCategorical` and `TrialAipw` change the assumption set and must be passed explicitly. If the formula is identified but a joint is unbound, `identify(...).inspect()` and `result.answer.detail` name the missing evidence.

Not-certified, missing evidence, local support failure, an uncalibrated interval, and a budget refusal are different outcomes. Do not treat them as one error.

A transport result exports through its own day-1 view, not the contracted execution the sections above describe. `ant.load(result.export())` rebuilds the result view from the verified identification and specialist artifacts and returns that view (`AnalysisResult` or `CausalResponseView`), not a `LoadedResult`: there is no `acceptance` slot to read, it is not a verified analysis program, and it does not restore a live study.

See the [2.0 transport migration](migrations/2.0-transport-day1.md), the [failure guide](guides/transport-failure.md), and [theorem scope](guides/transport-scope.md).

## Next steps

- [Choose an example](examples.md) for weighting, discovery, or temporal effects.
- [Inspect before estimating](python-options.md#prepare-explicitly-when-useful).
- [Configure populations, likelihoods, and overlap](python-options.md).
- [Browse the Python API](python-api.md).
