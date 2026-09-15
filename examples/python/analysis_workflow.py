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

# The same single call used in the notebooks also gives us a reusable study.
result = ant.analyze(data, graph=graph, query=query, seed=19, bootstrap=25)
print(result)
study = result.study
print("Calibration:", result.calibration.status, result.calibration.reason)
json.dumps(result.inspect().to_dict(), allow_nan=False)

# An immutable execution can be inspected and forwarded independently.
encoded = result.export()
loaded = ant.load(encoded)
assert loaded.acceptance.verified
assert loaded.answer.value == result.answer.value
assert loaded.export() == encoded

# Refresh updates the study only after success; the original result stays fixed.
new_data = {**data, "outcome": data["outcome"] + treatment}
updated = study.refresh(new_data)
assert np.isclose(updated.effect, result.effect + 1)
assert result.export() == encoded
assert np.isclose(study.estimate().effect, updated.effect)
print("After refresh:", updated)

# Explicit preparation is the same workflow when a caller wants to stop early.
prepared = ant.prepare(data, graph=graph, query=query, seed=19, bootstrap=25)
assert prepared.inspect().identification.available
assert np.isclose(prepared.estimate().effect, result.effect)
