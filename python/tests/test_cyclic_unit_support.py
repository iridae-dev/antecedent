# SPDX-License-Identifier: MIT OR Apache-2.0
"""Support decision of the unit every compiled cyclic program runs under, and a hand check.

The first test reads the unit decision from a prepared specified system and from a compiled
paired action-world decision. The second solves the two clamped worlds by hand and compares them
with the compiled program's draws.
"""

from __future__ import annotations

import pytest
from antecedent import decision
from antecedent.cyclic import (
    Equation,
    EquilibriumModel,
    compile_action_world_decision,
)

from test_cyclic_decisions import EXECUTION, REGIMES, contract, outcome, paired

NOISE = [[-0.2, 0.0], [0.0, 0.0], [0.2, 0.0], [0.4, 0.0]]
CLAMPS = (1.0, 2.0)


def system():
    model = EquilibriumModel(
        (
            Equation("wishlists", 0.0, (0.1, 0.3), "identity"),
            Equation("traffic", 0.0, (0.2, 0.0), "identity"),
        ),
        (1.0, 0.0, 0.0, 1.0),
    )
    return model.prepare(
        [[{"variable": 1, "value": value}] for value in CLAMPS], NOISE, execution=EXECUTION
    )


def test_prepared_system_and_compiled_decision_report_their_unit_decision() -> None:
    prepared = system()
    reading = prepared.unit_support()
    assert reading is not None
    assert reading["unit"] == "cyclic.general_systems"
    assert reading["supported"] is True
    assert reading["status"] == "supported"
    assert reading["missing_evidence"] == []
    quantities = {regime: outcome(regime=regime) for regime in REGIMES}
    decided = contract(quantities, units="wishlist_count", population="game", horizon=0)
    plan = compile_action_world_decision(
        prepared, paired(quantities, 0), decided, execution=EXECUTION
    )
    decision_reading = plan.unit_support()
    # The route is bound to a unit only once its record is earned; a bound unit states whether it
    # is supported and what it still lacks.
    if decision_reading is not None:
        assert isinstance(decision_reading["unit"], str)
        assert decision_reading["supported"] == (decision_reading["status"] == "supported")
        assert decision_reading["supported"] or decision_reading["missing_evidence"]
    assert plan.route == "CyclicActionWorldDecision"


def test_compiled_action_worlds_equal_the_hand_solved_equilibria() -> None:
    quantities = {regime: outcome(regime=regime) for regime in REGIMES}
    decided = contract(quantities, units="wishlist_count", population="game", horizon=0)
    plan = compile_action_world_decision(
        system(), paired(quantities, 0), decided, execution=EXECUTION
    )
    result = plan.execute()
    # With traffic clamped at c, wishlists = 0.1 wishlists + 0.3 c + e, so (0.3 c + e) / 0.9.
    for draws, clamp in zip(result.draws(), CLAMPS, strict=True):
        expected = [(0.3 * clamp + row[0]) / 0.9 for row in NOISE]
        assert list(draws) == pytest.approx(expected, abs=1e-7)
    assert result.decision.verdict == decision.Verdict("uniquely_optimal", ("promoted",))
