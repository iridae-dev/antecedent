"""Transport a direct and an indirect effect to a target population, or learn why you cannot.

Two populations share one additive structural model: a covariate ``z`` whose law differs, a
treatment ``a`` that is only ever set, a mediator ``m`` and an outcome ``y``. The selection set
names the mechanisms that may differ. When it holds only covariates (or the treatment) the
target contrast is ``sum_z P_T(z) G(z)`` from the source equations alone; a selection node on the
outcome is outside the class and refuses with a two-model witness.

Structural model::

    M = 0.5 Z + (2 - 0.5 Z) A + U_M
    Y = (3 + 0.75 Z) A + 4 M + 1.5 Z + U_Y

with ``a1 = 2`` and ``a0 = 0.5``. Per unit, ``NDE(z) = 4.5 + 1.125 z`` and ``NIE(z) = 12 - 3 z``.
Target law ``{0: 0.25, 1: 0.75}`` (``E z = 0.75``): NDE 5.34375, NIE 9.75, total 15.09375."""

from __future__ import annotations

import math

from antecedent.transported_counterfactual import (
    AdditiveNoiseScm,
    Affine,
    CovariateLaw,
    EdgeAssignment,
    Mechanism,
    MechanismSelection,
    Premises,
    TransportedCounterfactualRefusal,
    TransportedPathSpecificEffect,
    transported_path_specific_effect,
)

A1, A0 = 2.0, 0.5

model = AdditiveNoiseScm(
    treatment="a",
    covariates=("z",),
    mechanisms=(
        Mechanism("m", {"z": 0.5, "a": Affine(2.0, {"z": -0.5})}),
        Mechanism("y", {"a": Affine(3.0, {"z": 0.75}), "m": 4.0, "z": 1.5}),
    ),
)
source = CovariateLaw.univariate("z", {0.0: 0.4, 1.0: 0.4, 2.0: 0.2})
target = CovariateLaw.univariate("z", {0.0: 0.25, 1.0: 0.75})
# The class's declarations default to undeclared; declaring them records a claim, it checks nothing.
premises = Premises(additive_noise=True, noise_laws_shared=True, cross_world_independence=True)


def contrast(assignment: EdgeAssignment, selection: MechanismSelection):
    return transported_path_specific_effect(
        model, source, target, selection, assignment, premises=premises
    )


covariate_shift = MechanismSelection.on("z")
nde = EdgeAssignment.natural_direct("y", treated_value=A1, control_value=A0)
nie = EdgeAssignment.natural_indirect("y", ["m"], treated_value=A1, control_value=A0)
tot = EdgeAssignment.total("y", ["m"], treated_value=A1, control_value=A0)
direct = contrast(nde, covariate_shift)
indirect = contrast(nie, covariate_shift)
total = contrast(tot, covariate_shift)
print(direct.explain())

assert math.isclose(direct.target_contrast, 5.34375, abs_tol=1e-12)
assert math.isclose(indirect.target_contrast, 9.75, abs_tol=1e-12)
assert math.isclose(total.target_contrast, direct.target_contrast + indirect.target_contrast)
# Mistaking the source law for the target's would answer 5.4, not 5.34375.
assert math.isclose(direct.source_contrast, 5.4, abs_tol=1e-12)
assert direct.inference_claim == "point_only"
assert "additive_noise" in direct.derivation.declared  # declared, never checked

# The exported artifact is replayed against the identity the consumer retained.
replayed = TransportedPathSpecificEffect.consume(direct.export(), expected_identity=direct.identity)
assert replayed.target_contrast == direct.target_contrast

# A selection node on the outcome's own mechanism is outside the class.
try:
    contrast(nde, MechanismSelection.on("y"))
except TransportedCounterfactualRefusal as refusal:
    assert refusal.reason_code == "transport_proven_non_transportable"
    witness = refusal.witness
    assert witness is not None and witness.selected_node == "y"
    # Two models agree on the source population yet differ on the target contrast.
    assert math.isclose(witness.target_contrast_b - witness.target_contrast_a, 1.5, abs_tol=1e-12)
    print("refused:", refusal.detail)
else:
    raise AssertionError("selection on the outcome must refuse")
