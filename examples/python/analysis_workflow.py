"""One-call analysis, retained preparation, and portable execution reports (1.10)."""

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

# The five lines.
result = ant.analyze(data, graph=graph, query=query, seed=19, bootstrap=25)
study = result.study
updated = study.refresh(new_data)
report = result.inspect().to_dict()
loaded = ant.load(result.export())

print(result)
print("Calibration:", result.calibration.status, result.calibration.reason)
json.dumps(report, allow_nan=False)

# The loaded execution is verified and gives the same answer as the live one.
assert loaded.acceptance.verified
assert loaded.answer == result.answer
assert loaded.export() == result.export()

# Refresh re-executed the same program on new data; the original result stays fixed.
assert np.isclose(updated.answer.value, result.answer.value + 1)
assert np.isclose(study.estimate().answer.value, updated.answer.value)
print("After refresh:", updated)

# Explicit preparation is the same workflow when a caller wants to stop early.
prepared = ant.prepare(data, graph=graph, query=query, seed=19, bootstrap=25)
assert prepared.inspect().identification.available
assert np.isclose(prepared.estimate().answer.value, result.answer.value)
