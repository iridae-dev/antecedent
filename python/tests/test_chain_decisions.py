"""Decisions compiled together with a propagation chain, through the Python surface.

The fixtures are the Python spelling of ``crates/antecedent/src/analysis/
propagation_chain_decision_tests.rs`` and the robust-decision tests of
``propagation_chain_robust_decision.rs``, so every oracle is the hand value of four or three
declared rows:

* predictive draws ``1, 2, 3, 4``: ``promote`` (``x * x``) is worth ``(1 + 4 + 9 + 16) / 4 = 7.5``
  and ``hold`` (``x``) ``2.5``;
* paired action worlds ``(1, 2), (2, 2), (3, 2), (4, 2)``: ``promote`` is worth ``2.5`` with
  expected regret ``1 / 4``, ``hold`` ``2`` with expected regret ``(1 + 2) / 4``;
* an assumption envelope ``traffic + [0, 1]`` over draws ``1, 2, 3``: ``publish`` encloses
  ``[2, 3]`` and ``wait`` (``x - 10``) ``[-8, -7]``, so ``publish`` dominates.

Every compiler, number, identity and refusal is Rust's; the replayed decision must equal the
executed one.
"""

from __future__ import annotations

import numpy as np
import pytest
from antecedent import Execution, decision
from antecedent import propagation_chain as pc
from antecedent import transported_response as tr
from antecedent._artifact_protocol import ExportedArtifact
from antecedent.composition import SupportPolicy
from antecedent.errors import CausalTypeError, CausalValueError
from antecedent.joint_distribution import (
    DistributionIdentity,
    JointDistributionArtifact,
    ScientificQuantity,
)
from antecedent.outcomes import require_success

from _refusal import assert_registered_refusal

SEED = 13
TABLE = "joint:game-a"


def _quantity(
    variable: str = "steam:wishlists",
    name: str = "wishlists",
    regime: str = "declared-traffic",
    functional: str = "outcome",
) -> ScientificQuantity:
    return ScientificQuantity(
        variable_id=variable,
        variable_name=name,
        role="outcome",
        units="count",
        population_id="game-a",
        regime_id=regime,
        horizon=7,
        functional_id=functional,
    )


def _alternative() -> ScientificQuantity:
    return _quantity("steam:sales", "sales", "alternative-traffic")


def _contract(
    promote: decision.Expr,
    hold_input: ScientificQuantity,
    *,
    kind: decision.ActionKind = "external",
    criterion: decision.Criterion | None = None,
) -> decision.Contract:
    return decision.Contract(
        actions=(
            decision.Action("promote", inputs=(_quantity(),), utility=promote, kind=kind),
            decision.Action("hold", inputs=(hold_input,), utility=decision.x(0), kind=kind),
        ),
        utility_units="score",
        criterion=criterion or decision.Criterion.expected_utility(),
        target_population="game-a",
        horizon=7,
        structural_policy="require_invariant_best_action",
    )


def _square_contract() -> decision.Contract:
    return _contract(decision.x(0) * decision.x(0), _quantity())


def _chain(meaning: str = "predictive", reading: str = "predictive") -> pc.PropagationChain:
    incoming = pc.draws("incoming", [1.0, 2.0, 3.0, 4.0], meaning=meaning, alignment=TABLE)
    stage = pc.affine("wishlists", "derived_quantity", ["incoming"], scale=1.0, shift=0.0)
    return require_success(
        pc.propagate(
            [incoming], [stage], output="wishlists", reading=reading, output_quantity=_quantity()
        )
    )


def _world_chain(*, shared: bool = True) -> pc.PropagationChain:
    shock = [("disturbance", "traffic-world-shock")]
    traffic = pc.draws(
        "traffic", [1.0, 2.0, 3.0, 4.0], meaning="predictive", alignment=TABLE, factors=shock
    )
    conversion = pc.draws(
        "conversion",
        [2.0, 2.0, 2.0, 2.0],
        meaning="predictive",
        alignment=TABLE,
        factors=shock if shared else (),
    )
    stages = [
        pc.affine("wishlists", "derived_quantity", ["traffic"]),
        pc.affine("sales", "derived_quantity", ["conversion"]),
    ]
    return require_success(
        pc.propagate(
            [traffic, conversion],
            stages,
            output="wishlists",
            reading="predictive",
            output_quantity=_quantity(),
            joint_outputs=[("wishlists", _quantity()), ("sales", _alternative())],
        )
    )


def _world_contract() -> decision.Contract:
    return _contract(
        decision.x(0),
        _alternative(),
        kind="intervention",
        criterion=decision.Criterion.expected_regret(),
    )


def _utilities(result: pc.ChainDecision) -> dict[str, float]:
    return {action.id: action.expected_utility for action in result.actions}


def _exported_and_consumed(result: pc.ChainDecision) -> pc.ChainDecision:
    data = result.export()
    assert isinstance(data, ExportedArtifact)
    assert data.expectation is not None
    assert data.expectation.identity == result.program_identity
    replayed = require_success(
        pc.consume_chain_decision(bytes(data), expected_identity=result.program_identity)
    )
    assert replayed.replayed and not result.replayed
    assert replayed.actions == result.actions
    assert replayed.verdict == result.verdict
    assert replayed.chain.identity == result.chain.identity
    assert replayed.program_identity == result.program_identity
    assert bytes(replayed.export()) == bytes(data)
    with pytest.raises(pc.ChainDecisionRefusal) as wrong:
        require_success(pc.consume_chain_decision(bytes(data), expected_identity="0" * 64))
    assert wrong.value.code == "external_binding_mismatch"
    assert wrong.value.detail == "chain_decision.replay_refused"
    assert_registered_refusal(wrong.value)
    return replayed


def test_a_predictive_chain_and_its_decision_compile_into_one_program_and_replay() -> None:
    compiled = require_success(pc.compile_chain_decision(_chain(), _square_contract()))
    assert len(compiled.program_identity) == 64
    view = compiled.plan_view()
    assert view["identity"] == compiled.program_identity
    result = require_success(compiled.execute(execution=Execution(seed=SEED)))
    assert _utilities(result) == pytest.approx({"promote": 7.5, "hold": 2.5})
    assert result.selected == ("promote",)
    assert result.n_draws == 4
    assert result.program_identity == compiled.program_identity
    assert result.chain.program_identity == result.program_identity
    assert result.chain.output_law.draws.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert any("no measured coverage" in line for line in result.assumptions)
    replayed = _exported_and_consumed(result)
    assert result.method_provenance.records
    assert replayed.method_provenance == result.method_provenance


def test_posterior_parameter_draws_do_not_answer_an_outcome_risk_decision() -> None:
    with pytest.raises(pc.ChainDecisionRefusal) as refused:
        require_success(
            pc.compile_chain_decision(_chain("posterior", "posterior"), _square_contract())
        )
    assert refused.value.code == "decision_contract_unsatisfied"
    assert refused.value.detail == "chain_decision.refused"
    assert "outcome risk requires a predictive law" in refused.value.message
    assert_registered_refusal(refused.value)


def test_action_worlds_read_only_the_declared_joint_coupling() -> None:
    compiled = require_success(
        pc.compile_chain_decision(_world_chain(), _world_contract(), action_worlds=True)
    )
    result = require_success(compiled.execute(execution=Execution(seed=SEED)))
    assert result.n_draws == 4
    assert _utilities(result) == pytest.approx({"promote": 2.5, "hold": 2.0})
    regret = {action.id: action.expected_regret for action in result.actions}
    assert regret == pytest.approx({"promote": 0.25, "hold": 0.75})
    assert result.selected == ("promote",)
    _exported_and_consumed(result)
    with pytest.raises(pc.ChainDecisionRefusal) as unshared:
        require_success(
            pc.compile_chain_decision(
                _world_chain(shared=False), _world_contract(), action_worlds=True
            )
        )
    assert unshared.value.code == "decision_contract_unsatisfied"
    assert "shared" in unshared.value.message


def _transported_law() -> tr.TransportedResponse:
    identity = DistributionIdentity(
        semantic="interventional_predictive",
        quantities=(_quantity(),),
        alignment="joint",
        source_id="transported-wishlists",
        provider_id="declared-transport",
        rng_id="aligned-transport",
        snapshot_id="snapshot-1",
        causal_contract_id="transport-contract",
    )
    law = JointDistributionArtifact(
        identity, np.array([[1.0], [2.0], [3.0], [4.0]]), calibration="point_only"
    )
    declaration = tr.TransportDeclaration(
        route="declared",
        source_identity="source-trial",
        source_populations=("trial",),
        target_population="game-a",
        transport_identities=("selection-diagram",),
        assumptions=("transport mechanism declared",),
        provider_id="declared-transport",
        snapshot_id="snapshot-1",
        causal_contract_id="transport-contract",
    )
    return tr.TransportedResponse.declared_law(law, declaration)


def test_a_transported_law_decides_through_the_chain_compiler_under_its_support_policy() -> None:
    response = _transported_law()
    with pytest.raises(pc.ChainDecisionRefusal) as below:
        require_success(pc.compile_chain_decision(response, _square_contract()))
    assert below.value.code == "decision_contract_unsatisfied"
    assert "support is below the decision policy" in below.value.message
    compiled = require_success(
        pc.compile_chain_decision(
            response,
            _square_contract(),
            support_policy=SupportPolicy.require_all("extrapolative"),
        )
    )
    result = require_success(compiled.execute(execution=Execution(seed=17)))
    assert _utilities(result) == pytest.approx({"promote": 7.5, "hold": 2.5})
    assert any("transport receipt" in line for line in result.assumptions)
    _exported_and_consumed(result)


def test_declarations_that_do_not_apply_are_refused_before_the_native_layer() -> None:
    with pytest.raises(CausalValueError):
        require_success(
            pc.compile_chain_decision(
                _chain(), _square_contract(), support_policy=SupportPolicy.require_all()
            )
        )
    with pytest.raises(CausalValueError):
        require_success(
            pc.compile_chain_decision(_transported_law(), _square_contract(), action_worlds=True)
        )
    not_a_contract: object = "not a contract"
    with pytest.raises(CausalTypeError):
        require_success(pc.compile_chain_decision(_chain(), not_a_contract))
    not_a_source: object = [1.0, 2.0]
    with pytest.raises(CausalTypeError):
        require_success(pc.compile_chain_decision(not_a_source, _square_contract()))


def _envelope_quantity() -> ScientificQuantity:
    return _quantity("game:wishlists", functional="mean")


def _envelope_chain() -> pc.PropagationChain:
    traffic = pc.draws("traffic", [1.0, 2.0, 3.0], meaning="sampling", alignment="aligned")
    algorithm = pc.interval("algorithm", 0.0, 1.0)
    stage = pc.binary("wishlists", "derived_quantity", ["traffic", "algorithm"], "sum")
    return require_success(
        pc.propagate(
            [traffic, algorithm],
            [stage],
            output="wishlists",
            reading="assumption_envelope",
            output_quantity=_envelope_quantity(),
        )
    )


def _maximin_contract() -> decision.Contract:
    quantity = _envelope_quantity()
    return decision.Contract(
        actions=(
            decision.Action("publish", inputs=(quantity,), utility=decision.x(0), kind="external"),
            decision.Action(
                "wait", inputs=(quantity,), utility=decision.x(0) - 10.0, kind="external"
            ),
        ),
        utility_units="score",
        criterion=decision.Criterion.maximin_over_structures(),
        target_population="game-a",
        horizon=7,
        constraints=(
            decision.Constraint(
                "capacity", decision.x(0), bound=5.0, units="count", min_probability=1.0
            ),
        ),
        structural_policy="require_invariant_best_action",
    )


def test_a_robust_decision_dominates_over_the_assumption_envelope_and_replays() -> None:
    compiled = require_success(
        pc.compile_chain_robust_decision(_envelope_chain(), _maximin_contract())
    )
    assert compiled.plan_view()["identity"] == compiled.program_identity
    result = require_success(compiled.execute(execution=Execution(seed=19)))
    bounds = {a.id: (a.utility_lower, a.utility_upper) for a in result.actions}
    assert bounds["publish"] == pytest.approx((2.0, 3.0))
    assert bounds["wait"] == pytest.approx((-8.0, -7.0))
    assert all(a.certainly_admissible and not a.certainly_excluded for a in result.actions)
    capacity = result.actions[0].constraints[0]
    assert (capacity.id, capacity.required) == ("capacity", 1.0)
    assert capacity.probability_lower == pytest.approx(1.0)
    assert result.dominant_action == "publish"
    assert result.program_identity == compiled.program_identity

    data = result.export()
    replayed = require_success(
        pc.consume_chain_robust_decision(bytes(data), expected_identity=result.program_identity)
    )
    assert replayed.replayed
    assert replayed.actions == result.actions
    assert replayed.dominant_action == "publish"
    with pytest.raises(pc.ChainDecisionRefusal) as wrong:
        require_success(pc.consume_chain_robust_decision(bytes(data), expected_identity="1" * 64))
    assert wrong.value.code == "external_binding_mismatch"
    assert_registered_refusal(wrong.value)


def test_each_decision_compiler_refuses_the_other_reading() -> None:
    with pytest.raises(pc.ChainDecisionRefusal) as envelope:
        require_success(pc.compile_chain_decision(_envelope_chain(), _maximin_contract()))
    assert envelope.value.code == "decision_contract_unsatisfied"
    with pytest.raises(pc.ChainDecisionRefusal) as predictive:
        require_success(pc.compile_chain_robust_decision(_chain(), _square_contract()))
    assert predictive.value.code == "decision_contract_unsatisfied"
    assert "assumption envelope" in predictive.value.message
