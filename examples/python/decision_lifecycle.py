"""From an analysis to a decision to the study worth running next.

The lifecycle, one stage per block:

1. analyze your own data (a response curve) and decide on it, with units stated, never inferred;
2. bind external evidence (a lab's curve) to an identified program, inspect what stands behind the
   claim, and decide on it;
3. turn the decision contract into a value-of-information problem and rank candidate studies by
   expected value of sample information, net of cost, then replay the ranking against identities
   retained independently of the bytes.

Hand oracles: ``E[Y | do(a)] = 2a`` on the grid ``-0.5, 0, 0.5`` (utility = the mean, so the
action reading ``do(a = 0.5)`` is worth about +1 and beats the one at ``-0.5``); the lab curve
``1 + 2a`` on ``0, 1, 2`` (utility ``2 * mean - 1``: wait 1, treat 9); and a binary-state guess
decision with prior 1/2 whose signals of accuracy 3/4 and 5/8 have EVSI 1/4 and 1/8 and, at cost
1/10 in utility units, net values 3/20 and 1/40."""

from __future__ import annotations

import numpy as np
from antecedent import ResponseCurve, analyze, decision, design, external, identify, program_claims
from antecedent.joint_distribution import ScientificQuantity

# -- 1. analyze, then decide on the result --------------------------------------------------
rng = np.random.default_rng(23)
a = rng.normal(size=400)
data = {"a": a, "y": 2.0 * a + rng.normal(scale=0.2, size=400)}
result = analyze(data, query=ResponseCurve("a", "y", grid=[-0.5, 0.0, 0.5]), graph=[("a", "y")])
print(result.claim())

low = ScientificQuantity.from_response_dose(result, -0.5, outcome_units="mmHg")
high = ScientificQuantity.from_response_dose(result, 0.5, outcome_units="mmHg")
dose_contract = decision.Contract(
    actions=(
        decision.Action("low", inputs=(low,), utility=decision.x(0)),
        decision.Action("high", inputs=(high,), utility=decision.x(0)),
    ),
    utility_units="mmHg",
    criterion=decision.Criterion.expected_utility(),
    target_population="target",
)
program = program_claims.ProgramBinding.from_response(result, outcome_units="mmHg", dose_units="mg")
own = dose_contract.evaluate(result, program=program)
print(own.explain())
assert own.selected == ("high",)
# A mean grid retains no outcome draws: no regret and no replayable export.
assert own.evpi is None

# -- 2. bind external evidence, inspect it, decide on it ------------------------------------
identification = identify(
    graph=[("x", "a"), ("x", "y"), ("a", "y")],
    names=["x", "a", "y"],
    query=ResponseCurve("a", "y", grid=[0.0, 1.0, 2.0]),
)
spec = external.response(
    identification,
    outcome_units="mmHg",
    population="target",
    require_evidence=("factor:z",),
    require_assumptions=("ignorability",),
)
claim = spec.bind(
    external.Response(
        provider=external.ProviderObject(
            provider_id="lab",
            object_id="curve",
            version="v3",
            snapshot="snap-9",
            request="req-1",
            meaning="interventional_predictive",
            capabilities=("mean",),
        ),
        values=[1.0, 3.0, 5.0],
        evidence=("factor:z",),
        assumptions=("ignorability",),
        attested_by="lab",
    )
)
inspection = claim.inspect()
print(inspection.explain())
# An attested claim is never reported as natively estimated.
assert inspection.native is False

wait, _, treat = claim.quantities
lab_contract = decision.Contract(
    actions=(
        decision.Action("wait", inputs=(wait,), utility=decision.x(0) * 2.0 - 1.0),
        decision.Action("treat", inputs=(treat,), utility=decision.x(0) * 2.0 - 1.0),
    ),
    utility_units="utility",
    criterion=decision.Criterion.expected_utility(),
    target_population="target",
)
lab = lab_contract.evaluate(claim)
assert lab.selected == ("treat",)
assert [round(action.expected_utility, 9) for action in lab.actions] == [1.0, 9.0]

# -- 3. rank the studies that could change the decision -------------------------------------
state = ScientificQuantity(
    variable_id="schema:state",
    variable_name="state",
    role="outcome",
    units="dimensionless",
    population_id="target",
    regime_id="observational",
    horizon=0,
    functional_id="state",
)
guess = decision.Contract(
    actions=(
        decision.Action("guess0", inputs=(state,), utility=1.0 - decision.x(0)),
        decision.Action("guess1", inputs=(state,), utility=decision.x(0)),
    ),
    utility_units="utility",
    criterion=decision.Criterion.expected_utility(),
    target_population="target",
)
declared = design.DesignDecision.from_contract(guess, prior=design.StatePrior.draws([0.0, 1.0]))


def study(label: str, accuracy: float) -> design.Candidate:
    law = design.ExternalLaw.posterior(
        states=[0.0, 1.0],
        statistics=[0.0, 1.0],
        predictive=[0.5, 0.5],
        posterior=[[accuracy, 1.0 - accuracy], [1.0 - accuracy, accuracy]],
    )
    signal = design.ExternalSignal("lab", f"signal-{label}", "v1", "snap", "lab-qa", law)
    return design.Candidate(label, 1, signal, cost=0.1, cost_unit="utility")


signal_spec = design.SignalSpec(
    prior_id="prior-1",
    state=state,
    observation=ScientificQuantity(
        variable_id="schema:signal",
        variable_name="signal",
        role="outcome",
        units="dimensionless",
        population_id="target",
        regime_id="observational",
        horizon=0,
        functional_id="state",
    ),
    evidence_lineage=("snapshot:example",),
    rng_seed=3,
)
cost_map = design.CostMap("utility", "utility", 1.0)
ranked = design.rank_designs(
    [study("sharp-test", 0.75), study("rough-test", 0.625)],
    decision=declared,
    signal=signal_spec,
    cost_map=cost_map,
)
print(ranked.explain())
assert [c.id for c in ranked.candidates] == ["sharp-test", "rough-test"]
sharp, rough = ranked.candidates
assert abs(sharp.evsi - 0.25) < 1e-12 and abs(sharp.net_value - 0.15) < 1e-12
assert abs(rough.evsi - 0.125) < 1e-12 and abs(rough.net_value - 0.025) < 1e-12
# The signal is attested by the lab: it is never reported as natively replayed.
assert sharp.provider_trust == "externally_attested" and not sharp.natively_replayed
assert ranked.decision_contract_identity == guess.identity

# A consumer recomputes the ranking against the identities it retained; a changed contract refuses.
replay = design.consume(ranked.export(), expected_identity=ranked.expectation())
assert [entry.id for entry in replay.entries] == ["sharp-test", "rough-test"]
try:
    design.consume(
        ranked.export(),
        expected_identity=design.Expectation(decision_contract_identity="another-contract"),
    )
except design.DesignRankingRefusal:
    pass
else:
    raise AssertionError("a ranking must not consume under another contract identity")
