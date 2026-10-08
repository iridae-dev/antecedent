"""How much unmeasured confounding does a positive effect survive? (marginal sensitivity model)

A stratified average treatment effect (ATE) is identified at ``Lambda = 1`` (no unmeasured
confounding). ``Lambda`` bounds how far the odds of treatment may shift with a potential outcome;
``msm_ate_sensitivity`` returns the sharp ATE range at each ``Lambda`` and the tipping point where
the range first reaches a decision threshold. The range is an assumption range, not a confidence
interval: the strata are exact inputs, so the sampling interval is withheld.

Hand oracles for the two strata below: identified ATE = 0.5 * 0.4 + 0.5 * 0.2 = 0.3; sharp bounds
at Lambda = 2 are [0.04375, 0.4875]; the lower bound reaches 0 at (2 + sqrt(193)) / 7 = 2.27035."""

from __future__ import annotations

import math

from antecedent import msm_sensitivity, sensitivity_decision
from antecedent.joint_distribution import ScientificQuantity
from antecedent.msm_sensitivity import MsmStratum

# One row per stratum of the adjustment set: its population mass, the propensity of treatment,
# and the treated / control outcome rates (a column may also hold a finite OutcomeLaw).
strata = MsmStratum.table(
    {
        "mass": [0.5, 0.5],
        "propensity": [0.50, 0.25],
        "treated": [0.8, 0.5],
        "control": [0.4, 0.3],
    }
)

result = msm_sensitivity.msm_ate_sensitivity(strata, 4.0, decision_threshold=0.0)
print(result.explain())

assert math.isclose(result.identified, 0.3, abs_tol=1e-12)
lower, upper = msm_sensitivity.msm_ate_sensitivity(strata, 2.0, grid_points=2).assumption_range()
assert math.isclose(lower, 0.04375, abs_tol=1e-12)
assert math.isclose(upper, 0.4875, abs_tol=1e-12)
assert result.inference_claim == "assumption_range"

tipping = result.tipping
assert tipping is not None and tipping.bracketed and tipping.lambda_value is not None
assert math.isclose(tipping.lambda_value, (2.0 + math.sqrt(193.0)) / 7.0, abs_tol=1e-6)

# The same surface as a durable artifact a decision consumes: treat if the ATE is positive.
effect = ScientificQuantity(
    variable_id="ate",
    variable_name="ate",
    role="outcome",
    units="utils",
    population_id="target",
    regime_id="do(a=1)",
    horizon=0,
    functional_id="msm_ate",
)
surface = msm_sensitivity.msm_ate_sensitivity(strata, 2.0, grid_points=3)
artifact = surface.to_sensitivity_artifact(
    effect=effect,
    actions=[
        sensitivity_decision.SensitivityAction("treat", sensitivity_decision.quantity("ate")),
        sensitivity_decision.SensitivityAction("skip", sensitivity_decision.const(0.0)),
    ],
    causal_contract_id="example-contract",
)
decided = sensitivity_decision.decide(artifact.contract(), artifact)
print(decided.explain())
assert decided.kind == "invariant_action" and decided.invariant_action == "treat"
assert decided.sampling.status == "withheld"
