"""Engineering ownership, refusal and replay for the explicit joint compiler."""

from __future__ import annotations

from dataclasses import replace

import antecedent as ant
import numpy as np
import pytest
from antecedent import decision
from antecedent import decision_robust as robust
from antecedent.artifacts import LoadContext
from antecedent.execution import ResourceLimits
from antecedent.joint_distribution import (
    DistributionIdentity,
    JointDistributionArtifact,
    ScientificQuantity,
)
from antecedent.outcomes import require_success


def fixture():
    data = {
        "t": np.tile([0.0, 1.0, 0.0, 1.0], 4),
        "z": np.tile([0.0, 0.0, 1.0, 1.0], 4),
        "y": np.arange(16, dtype=float),
    }
    study = ant.prepare(
        data,
        graph=[("z", "t"), ("z", "y"), ("t", "y")],
        query=ant.AverageEffect(treatment="t", outcome="y"),
        inference=ant.Frequentist(bootstrap=0),
        refute=False,
    )
    quantities = tuple(
        ScientificQuantity(
            variable_id=x,
            variable_name=x,
            role="outcome",
            units="units",
            population_id="target",
            regime_id="do(joint)",
            horizon=0,
            functional_id="outcome",
        )
        for x in ("a", "b")
    )
    identity = DistributionIdentity(
        semantic="interventional_predictive",
        quantities=quantities,
        alignment="joint",
        source_id="declared",
        provider_id="fixture",
        rng_id="none",
        snapshot_id="synthetic",
        causal_contract_id="conditional",
    )
    law = JointDistributionArtifact(
        identity, np.array([[0.0, 0.0], [1.0, 1.0]]), calibration="exact"
    )
    contract = robust.admissible_contract(
        decision.Contract(
            actions=(
                decision.Action("pair", inputs=quantities, utility=decision.x(0) * decision.x(1)),
                decision.Action("safe", inputs=(quantities[0],), utility=0.0 * decision.x(0) + 0.3),
            ),
            utility_units="units",
            criterion=decision.Criterion.expected_utility(),
            target_population="target",
            structural_policy="maximin",
        )
    )
    return study, contract, [robust.Claim.evaluated("s", law, support=robust.Support("supported"))]


def test_compiled_joint_source_replay_requires_independent_owner_and_keeps_scope():
    study, contract, claims = fixture()
    source = require_success(
        robust.joint_scenario_carrier(
            study, contract, claims, joint_structure="paired common states", exhaustive=True
        )
    )
    compiled = require_success(robust.compile_robust_decision(study, contract, source))
    result = require_success(compiled.evaluate())
    assert result.complete
    assert result.standing == "conditional_on_declared_joint_scenarios"
    assert result["native_verified"] is False
    assert result.to_dict()["standing"] == result.standing
    assert "_owner" not in result.to_dict()
    assert result.standing in repr(result)
    assert source.to_dict()["causal_target"] == source.causal_target
    assert compiled.to_dict()["standing"] == compiled.standing
    with pytest.raises(TypeError, match="does not support item assignment"):
        result.payload["complete"] = False
    loaded_source = require_success(ant.load(source.export(), expected=source.identity))
    independent = require_success(robust.compile_robust_decision(study, contract, loaded_source))
    assert independent.program_identity == compiled.program_identity
    sealed = compiled.export()
    context = LoadContext(robust_source=independent)
    loaded = require_success(ant.load(sealed, context=context))
    assert dict(loaded) == dict(result)
    with pytest.raises(ant.errors.CausalValueError, match="robust_source"):
        ant.load(sealed)
    assert ant.load(source.export(), expected="0" * 64).accepted is False
    with pytest.raises(ant.errors.CausalResourceError):
        compiled.evaluate(execution=ant.Execution(limits=ResourceLimits(memory_bytes=1)))
    assert require_success(compiled.evaluate()).complete


def test_incomplete_source_stays_incomplete_and_unsupported_policy_refuses():
    study, contract, claims = fixture()
    source = require_success(
        robust.joint_scenario_carrier(
            study,
            contract,
            claims,
            joint_structure="paired common states",
            exhaustive=False,
            search_stop="operations",
        )
    )
    compiled = require_success(robust.compile_robust_decision(study, contract, source))
    result = require_success(compiled.evaluate())
    assert not result.complete
    assert result["search_stop"] == "search.operations"
    assert result.verdict.kind == "insufficient_claims"
    assert (
        robust.joint_scenario_carrier(
            study, contract, claims, joint_structure="", exhaustive=True
        ).accepted
        is False
    )


def test_refusal_details_and_cancellation_remain_typed():
    study, contract, claims = fixture()
    source = require_success(
        robust.joint_scenario_carrier(
            study, contract, claims, joint_structure="paired", exhaustive=True
        )
    )
    bayes = robust.admissible_contract(
        replace(contract.contract, structural_policy="bayes_over_structures")
    )
    refused = robust.compile_robust_decision(study, bayes, source)
    assert not refused.accepted
    assert refused.refusal.fields["detail"] == ant._reason_details.decision_joint_scenarios.CARRIER
    compiled = require_success(robust.compile_robust_decision(study, contract, source))
    token = ant.CancellationToken()
    token.cancel()
    with pytest.raises(ant.errors.CausalCancelledError):
        compiled.evaluate(execution=ant.Execution(cancel=token))
    assert require_success(compiled.evaluate()).complete
