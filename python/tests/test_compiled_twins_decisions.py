"""The direct finite-scenario decision warns, and the compiled joint-scenario terminal reproduces it."""

from __future__ import annotations

import antecedent as ant
import numpy as np
import pytest
from antecedent import decision
from antecedent import decision_robust as robust
from antecedent._deprecation import AntecedentDeprecationWarning
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


def _compiled(study, contract, claims):
    source = require_success(
        robust.joint_scenario_carrier(
            study, contract, claims, joint_structure="paired common states", exhaustive=True
        )
    )
    return require_success(robust.compile_robust_decision(study, contract, source))


def test_finite_scenarios_warns_and_the_compiled_terminal_reaches_the_same_verdict():
    study, contract, claims = fixture()
    with pytest.warns(AntecedentDeprecationWarning, match="finite_scenarios"):
        # The released direct path returns an Outcome (it is an outcome operation).
        direct = require_success(robust.finite_scenarios(contract, claims))
    # The same call through the general entry warns once, naming the compiled replacement.
    with pytest.warns(AntecedentDeprecationWarning, match="compile_robust_decision"):
        robust.robust(contract, claims, kind="finite_scenarios")
    result = require_success(_compiled(study, contract, claims).evaluate())
    assert result.complete
    assert result.verdict.kind == direct.verdict.kind
    assert result.verdict.selected == direct.verdict.selected
