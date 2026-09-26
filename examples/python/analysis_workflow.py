"""Estimate an effect, reuse the analysis, and save the result.

The simulated treatment increases the outcome by 2. After estimating it,
we change the data so the effect is 3 and run the same study again.
See docs/python-workflow.md for installation and a step-by-step explanation."""

from __future__ import annotations

import json

import antecedent as ant
import numpy as np

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

new_data = {**data, "outcome": data["outcome"] + treatment}

# Estimate the effect, then keep the study so we can reuse it.
result = ant.analyze(data, graph=graph, query=query, seed=19, bootstrap=25)
study = result.study
updated = study.refresh(new_data)
report = result.inspect().to_dict()
loaded = ant.load(result.export())

print(result)
print("Calibration:", result.calibration.status, result.calibration.reason)
json.dumps(report, allow_nan=False)

# The loaded execution keeps the recorded answer. A verified load replayed the
# program; a sealed load verified the contract and identities and names the
# checked operation it cannot replay from bytes alone.
assert loaded.acceptance.verified or loaded.acceptance.sealed
assert loaded.answer == result.answer
assert loaded.export() == result.export()

# Refresh re-executed the same program on new data; the original result stays fixed.
assert np.isclose(updated.answer.value, result.answer.value + 1)
assert np.isclose(study.estimate().answer.value, updated.answer.value)
print("After refresh:", updated)

# To inspect identification before estimating, prepare the study explicitly.
prepared = ant.prepare(data, graph=graph, query=query, seed=19, bootstrap=25)
assert prepared.inspect().identification.available
assert np.isclose(prepared.estimate().answer.value, result.answer.value)
