"""Keeping a fitted model's exogenous noise and evaluating it again under other settings.

Drawing returns the noise that produced the rows. Replaying that noise reproduces the draws bit
for bit, replaying it under an intervention gives that intervention's counterfactual for the very
same units, and noise edited by the caller (here one coordinate set to zero) is replayed as given.
The chain ``a -> b -> c`` is linear, so the variance of ``b`` once the noise of ``a`` is held at
zero is exactly the variance of ``b``'s own noise."""

from __future__ import annotations

import antecedent
import numpy as np

rng = np.random.default_rng(6)
a = 1.0 + rng.normal(size=3000)
b = 0.5 + 2.0 * a + 0.5 * rng.normal(size=3000)
c = -1.0 + 3.0 * b + 2.0 * rng.normal(size=3000)
model = antecedent.model.fit_gcm(["a", "b", "c"], [a, b, c], [("a", "b"), ("b", "c")])

# Draws come with their noise, one row per variable in the model's order.
draws, noise = model.sample_noise({}, 4000, seed=3)
assert noise.shape == (3, 4000)
np.testing.assert_array_equal(model.replay_noise(noise).draws, draws.draws)

# Replaying under a hard set changes that node and its descendants only.
fixed = model.replay_noise(noise, {"c": 10.0}).draws
assert (fixed[2] == 10.0).all()
np.testing.assert_array_equal(fixed[:2], draws.draws[:2])

# Under do(a = 2) each unit keeps its own noise for b, so the change in b is exactly linear in
# the unit's original a.
moved = model.replay_noise(noise, {"a": 2.0}).draws
assert (moved[0] == 2.0).all()
change_in_b = moved[1] - draws.draws[1]
assert np.isclose(np.corrcoef(change_in_b, 2.0 - draws.draws[0])[0, 1], 1.0)

# Noise the caller edits is replayed as given: with the noise of a held at zero every unit shares
# one a, so b varies only through its own noise.
held = noise.copy()
held[0] = 0.0
conditioned = model.replay_noise(held).draws
assert np.ptp(conditioned[0]) == 0.0
np.testing.assert_allclose(np.var(conditioned[1]), np.var(noise[1]), rtol=1e-9)

# Noise of another model's width is refused rather than reshaped.
try:
    model.replay_noise(noise[:2])
except Exception as refusal:  # noqa: BLE001 - the binding's refusal class is not the point
    assert "noise has 2 rows" in str(refusal), refusal
else:
    raise AssertionError("noise of the wrong width must be refused")
print(
    "replay is exact; do(a = 2) moves b by",
    round(float(change_in_b.mean()), 3),
    "on average",
)
