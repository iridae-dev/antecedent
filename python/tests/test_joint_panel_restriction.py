"""A declared Gaussian restriction identifies the fitted panel only through general ID."""

from __future__ import annotations

import pytest
from antecedent import Execution, ResourceLimits, joint_panel, state
from antecedent.errors import (
    CausalCancelledError,
    CausalError,
    CausalResourceError,
    CausalValueError,
)
from test_joint_panel import declarations

RESTRICTION = "joint_adjustment.gaussian_restriction"
PERIODS = 24
GRAPH = {
    "periods": PERIODS,
    "channels": 3,
    "contexts": 1,
    "carryover_reach": 2,
    "outcome_persistence": True,
    "outcome_periods": [PERIODS - 1],
}


def prepared(*, restriction):
    groups, worlds = declarations()
    return (
        joint_panel.prepare(
            groups,
            channels=["a", "b", "c"],
            outcome="y",
            confounder="z",
            season="season",
            seasons=4,
            interval_ns=604_800_000_000_000,
            worlds=worlds,
            interactions={},
            options={
                "optimizer": {"max_iterations": 1, "gradient_tolerance": 1e-12},
                "inference": {"kind": "frequentist", "draws": 4},
                "initial": None,
            },
            gaussian_restriction=restriction,
        ),
        worlds,
    )


def attempt_ids(panel):
    return {attempt["id"]: attempt for attempt in panel.inspect()["attempts"]}


def test_declared_restriction_retains_identification_of_the_fit_as_a_separate_attempt():
    panel, _ = prepared(restriction=True)
    before = panel.inspect()
    assert before["gaussian_restriction_declared"] is True
    assert before["identified"] is False
    assert RESTRICTION not in attempt_ids(panel)
    bound = panel.with_unrolled_graph(**GRAPH)
    after = bound.inspect()
    assert after["identified"] is True
    assert after["fits_performed"] == 0
    attempt = attempt_ids(bound)[RESTRICTION]
    assert attempt["outcome"] == "Identified"
    assert "not nonparametrically" in attempt["detail"]
    assert "general ID identifies" in attempt_ids(bound)["joint_adjustment.observational"]["detail"]
    # The graph run is a new handle; the unbound one is untouched.
    assert panel.inspect()["identified"] is False


def test_the_verdict_alone_never_identifies_without_the_declaration():
    panel, _ = prepared(restriction=False)
    bound = panel.with_unrolled_graph(**GRAPH)
    inspection = bound.inspect()
    assert inspection["gaussian_restriction_declared"] is False
    assert inspection["identified"] is False
    assert RESTRICTION not in attempt_ids(bound)
    assert "general ID identifies" in attempt_ids(bound)["joint_adjustment.observational"]["detail"]


def test_a_declaration_does_not_identify_through_a_hedge():
    panel, _ = prepared(restriction=True)
    bow = [(("channel", 0, PERIODS - 1), ("outcome", PERIODS - 1))]
    bound = panel.with_unrolled_graph(**GRAPH, latent_confounding=bow)
    assert bound.inspect()["identified"] is False
    assert RESTRICTION not in attempt_ids(bound)
    assert "hedge" in attempt_ids(bound)["joint_adjustment.observational"]["detail"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"history_feedback": True}, "outcome history"),
        ({"carryover_reach": 1}, "carryover"),
        ({"outcome_persistence": False}, r"AR\(1\)"),
        ({"periods": 3, "outcome_periods": [2]}, "different channel count or period count"),
        ({"channels": 2}, "different channel count or period count"),
    ],
)
def test_a_graph_the_fitted_mechanism_does_not_realize_is_refused(change, message):
    panel, _ = prepared(restriction=True)
    with pytest.raises(CausalValueError, match=message):
        panel.with_unrolled_graph(**{**GRAPH, **change})
    assert panel.inspect()["identified"] is False


def test_malformed_graph_declarations_are_refused_before_any_run():
    panel, _ = prepared(restriction=True)
    for node in (("channel", 0), ("outcome",), ("sideways", 1), "outcome", ()):
        with pytest.raises(CausalValueError, match="unrolled graph node"):
            panel.with_unrolled_graph(**GRAPH, latent_confounding=[(node, ("outcome", 3))])
    with pytest.raises(CausalError):
        panel.with_unrolled_graph(**{**GRAPH, "periods": 0})
    # A latent edge naming a node outside the program is refused when the graph is built.
    with pytest.raises(CausalError):
        panel.with_unrolled_graph(
            **GRAPH, latent_confounding=[(("outcome", PERIODS + 5), ("outcome", 0))]
        )


def test_a_graph_too_large_to_fund_or_a_cancelled_run_is_an_operational_stop():
    panel, _ = prepared(restriction=True)
    # The size of the graph is bounded before anything is allocated from a few digits.
    with pytest.raises(CausalResourceError, match="exceeds"):
        panel.with_unrolled_graph(**{**GRAPH, "periods": 10**9})
    with pytest.raises(CausalResourceError, match="exceeds"):
        panel.with_unrolled_graph(**{**GRAPH, "channels": 2**63})
    tight = Execution(limits=ResourceLimits(memory_bytes=1))
    with pytest.raises(CausalResourceError):
        panel.with_unrolled_graph(**GRAPH, execution=tight)
    token = state.CancellationToken()
    token.cancel()
    with pytest.raises(CausalCancelledError):
        panel.with_unrolled_graph(**GRAPH, execution=Execution(cancel=token))


def test_the_declaration_survives_retargeting_and_is_not_inherited_by_a_fresh_panel():
    panel, worlds = prepared(restriction=True)
    retargeted = panel.retarget(worlds)
    assert retargeted.inspect()["gaussian_restriction_declared"] is True
    assert retargeted.with_unrolled_graph(**GRAPH).inspect()["identified"] is True
    other, _ = prepared(restriction=False)
    assert other.retarget(worlds).inspect()["gaussian_restriction_declared"] is False
