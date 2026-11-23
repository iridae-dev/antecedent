# SPDX-License-Identifier: MIT OR Apache-2.0
"""Compare an observation, a clamp and a shock on one feedback system.

The system is A = 0.4 B + U_A and B = 0.5 A + U_B with independent centered unit-variance
disturbances. No data are simulated: each answer is the Gaussian equilibrium law that the
stated equations imply, so the three numbers are fixture truths of this model and not claims
about a real system. A second system shows an intervention that has no unique solution and is
retained as failed mass.
"""

from __future__ import annotations

# %%
import json

from antecedent.cyclic import (
    CyclicRefusal,
    LinearCyclicModel,
    Set,
    Shift,
    World,
    consume_family_artifact,
)
from antecedent.outcomes import require_success

# %% [markdown]
# ## Declare the system
#
# Two variables feed each other: A = 0.4 B + U_A and B = 0.5 A + U_B, with independent
# centered unit-variance disturbances. The four worlds are the observed system, a clamp of A at
# 0, a clamp of A at 1 and a unit shock to A.

# %%
model = LinearCyclicModel.from_direct_effects(
    ["A", "B"],
    {("A", "B"): 0.4, ("B", "A"): 0.5},
    noise_covariance=[1.0, 0.0, 0.0, 1.0],
)
prepared = model.worlds(
    [
        World(0),
        World(1, Set("A", 0.0)),
        World(2, Set("A", 1.0)),
        World(3, Shift("A", 1.0)),
    ],
    component_id="feedback-pair",
    source_id="specification",
    certifying_evidence=["bench-derivation"],
)
assert prepared.inspect()["preparation_evaluations"] == 0
worlds = require_success(prepared.interventions())

# %% [markdown]
# ## The three questions and their answers
#
# The observed association of B with A, the effect of clamping A from 0 to 1, and the effect
# of a unit shock to A's disturbance are three different numbers on the same system.

# %%
observed = worlds[0]
association = observed.covariance("A", "B") / observed.covariance("A", "A")
clamp_effect = worlds[2].mean("B") - worlds[1].mean("B")
shock_effect = worlds[3].mean("B")
assert abs(association - 1.40625 / 1.8125) < 1e-10
assert abs(association - 0.776) < 5e-4
assert abs(clamp_effect - 0.5) < 1e-10
assert abs(shock_effect - 0.625) < 1e-10
assert abs(worlds[3].mean("A") - 1.25) < 1e-10

assert worlds[1].actions[0]["kind"] == "hard_clamp"
assert worlds[3].actions[0]["kind"] == "additive_shock"
assert worlds[1].solved_by["method"] == "acyclic_order"
assert worlds[3].solved_by["method"] == "strongly_connected_blocks"
assert all(world.certified and world.failed_mass == 0.0 for world in worlds.values())

# %% [markdown]
# ## Execute and replay
#
# Execute the family once, export the artifact and consume it again. The replayed result
# must equal the original.

# %%
result = require_success(prepared.execute())
artifact = result.export()
replayed = require_success(
    consume_family_artifact(bytes(artifact), expected_identity=result.identity)
)
assert replayed.to_dict() == result.to_dict()
three_questions = {
    "association_of_B_with_A": association,
    "clamp_A_from_0_to_1": clamp_effect,
    "unit_shock_to_U_A": shock_effect,
    "clamp_schedule": worlds[1].solved_by["method"],
    "shock_schedule": worlds[3].solved_by["method"],
}
print(json.dumps(three_questions, indent=2, default=str))

# %% [markdown]
# ## An intervention with no solution
#
# Two overlapping loops make a clamp on C unsolvable. The world is retained as failed mass, and
# asking it for a mean raises a refusal that names the next attempt.

# %%
model = LinearCyclicModel.from_direct_effects(
    ["A", "B", "C"],
    {("A", "B"): 1.0, ("A", "C"): 1.0, ("B", "A"): 1.0, ("C", "A"): 1.0},
)
worlds = require_success(
    model.worlds(
        [World(0), World(1, Set("C", 1.0))],
        component_id="overlapping-loops",
        source_id="specification",
        certifying_evidence=["bench-derivation"],
        certificate="report_uncertified",
    ).interventions()
)
assert list(worlds) == [0, 1]
assert not worlds[0].certified

clamped = worlds[1]
assert clamped.status == "no_solution"
assert clamped.failed_mass == 1.0
assert clamped.solved_by["method"] == "not_solved"
try:
    clamped.mean("B")
except CyclicRefusal as refusal:
    assert refusal.next_attempt
    next_attempt = refusal.next_attempt
else:
    raise AssertionError("a world with no solution must not report a mean")
unsolvable_clamp = {
    "status": clamped.status,
    "failed_mass": clamped.failed_mass,
    "next_attempt": next_attempt,
}
print(json.dumps(unsolvable_clamp, indent=2, default=str))
