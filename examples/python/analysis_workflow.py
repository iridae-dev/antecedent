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
result = ant.analyze(
    data,
    graph=graph,
    query=query,
    execution=ant.Execution(seed=19),
    inference=ant.Frequentist(bootstrap=25),
)
study = result.study
branch = ant.rebind(study, data=new_data)
updated = ant.estimate(branch)
report = result.inspect().to_dict()
artifact = result.export()
loaded = ant.load(artifact, expected=artifact.expectation)

print(result)
print("Calibration:", result.value.calibration.status, result.value.calibration.reason)
json.dumps(report, allow_nan=False)

# The loaded execution keeps the recorded answer. A verified load replayed the
# program; a sealed load verified the contract and identities and names the
# checked operation it cannot replay from bytes alone.
assert loaded.acceptance.verified or loaded.acceptance.sealed
assert loaded.answer.kind == result.answer.kind
assert loaded.answer.value == result.answer.value
assert bytes(loaded.export()) == bytes(artifact)

# The branch re-executed the same program on new data; the original result stays fixed.
assert np.isclose(updated.answer.value, result.answer.value + 1)
assert np.isclose(ant.estimate(study).answer.value, result.answer.value)
print("After rebinding:", updated)

# To inspect identification before estimating, prepare the study explicitly.
prepared = ant.prepare(
    data,
    graph=graph,
    query=query,
    execution=ant.Execution(seed=19),
    inference=ant.Frequentist(bootstrap=25),
)
assert prepared.inspect().identification.available
assert np.isclose(ant.estimate(prepared).answer.value, result.answer.value)
