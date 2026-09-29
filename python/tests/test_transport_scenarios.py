"""Finite graph/selection scenario sets for one transport question.

Every scenario shares ``z -> x``, ``z -> y``, ``x -> y``, ``x <-> y`` and differs
only in which mechanisms may change: ``standardize`` (selection on ``z``) gives
``sum_z P_s(y | do(x), z) P*(z)``; ``direct`` (none) gives ``P_s(y | do(x))``;
``outcome_shift`` (selection on ``y``) is not transportable. Expected values are
computed here from the laws.
"""

import json

import pytest
from antecedent import Admg, Cpdag
from antecedent.errors import (
    CausalSerializationError,
    CausalUnsupportedError,
    CausalValueError,
)
from antecedent.transport import advanced as transport

NAMES = ["z", "x", "y"]
SOURCE = (0.32, 0.08, 0.12, 0.48)  # do(x=1) over (z, y)
TARGET = (0.3, 0.15, 0.2, 0.1, 0.05, 0.05, 0.05, 0.1)  # (z, x, y), P*(z=1) = 0.25
STANDARDIZED = 0.75 * 0.2 + 0.25 * 0.8
DIRECT = 0.08 + 0.48


def graph(names=NAMES):
    return Admg.from_edges(names, [("z", "x"), ("z", "y"), ("x", "y")], [("x", "y")])


def scenarios(weights=None, names=NAMES):
    w = weights or (None, None, None)
    return transport.TransportScenarioSet(
        [
            transport.TransportScenario("standardize", graph(names), ["z"], w[0]),
            transport.TransportScenario("direct", graph(names), [], w[1]),
            transport.TransportScenario("outcome_shift", graph(names), ["y"], w[2]),
        ]
    )


def catalog(with_target=True):
    regimes = [
        transport.EvidenceRegime(
            "trial", "source", kind="experimental", interventions=["x"], measured=["z", "y"]
        )
    ]
    if with_target:
        regimes.append(transport.EvidenceRegime("obs", "target", measured=["z", "x", "y"]))
    return transport.EvidenceCatalog(regimes=regimes)


def laws(source=SOURCE, with_target=True):
    out = [
        transport.ExactDiscreteLaw(
            "source",
            "trial",
            (("z", (0.0, 1.0)), ("y", (0.0, 1.0))),
            source,
            "trial",
            interventions=(("x", 1.0),),
        )
    ]
    if with_target:
        out.append(
            transport.ExactDiscreteLaw(
                "target",
                "obs",
                (("z", (0.0, 1.0)), ("x", (0.0, 1.0)), ("y", (0.0, 1.0))),
                TARGET,
                "target",
            )
        )
    return transport.ExactTransportData(tuple(out))


def prepare(sets=None, cat=None, data=None, **limits):
    return transport.prepare_transport_scenarios(
        sets or scenarios(),
        outcomes=["y"],
        treatments=["x"],
        source="source",
        target="target",
        catalog=cat or catalog(),
        laws=data or laws(),
        at={"x": 1.0},
        **limits,
    )


def by_name(report):
    return {s["name"]: s for s in report["scenarios"]}


def mean(scenario):
    return scenario["point"]["means"]["y"]


def test_identified_scenarios_disagree_and_the_envelope_spans_them():
    report = json.loads(prepare().estimate())
    assert report["scope"] == "finite_transport_scenarios_structural_envelope"
    s = by_name(report)
    assert s["standardize"]["status"] == s["direct"]["status"] == "identified"
    assert mean(s["standardize"]) == pytest.approx(STANDARDIZED, abs=1e-12)
    assert mean(s["direct"]) == pytest.approx(DIRECT, abs=1e-12)
    envelope = report["envelope"]["means"][0]
    assert (envelope["lower"], envelope["upper"]) == pytest.approx((STANDARDIZED, DIRECT))
    assert (envelope["lower_scenario"], envelope["upper_scenario"]) == ("standardize", "direct")
    assert "not_a_confidence_interval" in report["envelope"]["interpretation"]


def test_unidentified_scenarios_are_retained_not_filtered():
    report = json.loads(prepare().estimate())
    shift = by_name(report)["outcome_shift"]
    assert shift["status"] == "structurally_unidentified"
    assert shift["point"] is None
    counts = {m["status"]: m["count"] for m in report["masses"]}
    assert counts["identified"] == 2 and counts["structurally_unidentified"] == 1
    assert report["envelope"]["scenarios"] == ["direct", "standardize"]


def test_declared_weights_keep_unidentified_and_residual_mass():
    report = json.loads(prepare(scenarios((0.3, 0.2, 0.4))).estimate())
    assert report["residual_mass"] == pytest.approx(0.1)
    masses = {m["status"]: m["mass"] for m in report["masses"]}
    assert masses["structurally_unidentified"] == pytest.approx(0.4)
    weighted = report["weighted"]
    total = 0.3 * STANDARDIZED + 0.2 * DIRECT
    assert weighted["identified_mass"] == pytest.approx(0.5)
    assert weighted["unaccounted_mass"] == pytest.approx(0.5)
    assert weighted["identified_weighted_sums"]["y"] == pytest.approx(total)
    rng = weighted["ranges"][0]
    assert (rng["lower"], rng["upper"]) == pytest.approx((total, total + 0.5))
    with pytest.raises(CausalValueError):
        scenarios((0.3, None, 0.4))
    with pytest.raises(CausalValueError):
        scenarios((0.6, 0.3, 0.4))


def test_each_scenario_binds_its_own_evidence():
    report = json.loads(
        prepare(cat=catalog(with_target=False), data=laws(with_target=False)).estimate()
    )
    s = by_name(report)
    assert s["standardize"]["status"] == "missing_evidence"
    assert s["direct"]["status"] == "identified"
    unsupported = json.loads(prepare(data=laws(with_target=False)).estimate())
    assert by_name(unsupported)["standardize"]["status"] == "unsupported_provider"


def test_a_scenario_budget_reports_the_rest_unevaluated():
    report = json.loads(prepare(max_scenarios=2).estimate())
    assert by_name(report)["standardize"]["status"] == "unevaluated"
    assert report["receipt"]["stop"] == "search.operations"
    assert report["receipt"]["unevaluated"] == ["standardize"]


def test_scenario_and_variable_order_do_not_change_the_report():
    forward = prepare(scenarios((0.3, 0.2, 0.4)))
    shuffled = transport.TransportScenarioSet(list(reversed(scenarios((0.3, 0.2, 0.4)).scenarios)))
    backward = prepare(shuffled)
    assert json.loads(forward.estimate()) == json.loads(backward.estimate())


def test_refresh_keeps_decisions_and_moves_points():
    prepared = prepare()
    before = json.loads(prepared.estimate())
    prepared.refresh(laws(source=(0.1, 0.3, 0.3, 0.3)))
    after = json.loads(prepared.estimate())
    assert [s["status"] for s in after["scenarios"]] == [s["status"] for s in before["scenarios"]]
    assert mean(by_name(after)["direct"]) == pytest.approx(0.6)


def test_artifact_round_trip_keeps_failed_scenarios_and_refuses_tampering():
    prepared = prepare(scenarios((0.3, 0.2, 0.4)))
    live = json.loads(prepared.estimate())
    artifact = prepared.export()
    consumed = json.loads(transport.consume_transport_scenarios_artifact(artifact))
    assert consumed["scenarios"] == live["scenarios"]
    assert by_name(consumed)["outcome_shift"]["status"] == "structurally_unidentified"
    with pytest.raises(CausalSerializationError):
        transport.consume_transport_scenarios_artifact(artifact[:-4] + b"\x00\x00\x00\x00")


def test_cross_scenario_inference_and_equivalence_classes_are_refused():
    prepared = prepare()
    with pytest.raises(CausalUnsupportedError) as refused:
        prepared.aggregate_interval()
    assert refused.value.reason_code == "scenario_aggregate_not_licensed"
    cpdag = Cpdag.from_directed_undirected(NAMES, [("z", "x")], [("x", "y")])
    with pytest.raises(CausalUnsupportedError) as equivalence:
        transport.TransportScenario("class", cpdag, [])
    assert equivalence.value.reason_code == "route_not_supported"


# Checked execution: each licensed route runs from the retained, frozen plans with
# no builder alive, and the test inspects those retained plans.


def _prepared_without_builder(weights=None):
    builder = scenarios(weights)
    prepared = prepare(builder)
    del builder
    return prepared


def test_prepare_retains_checked_scenario_plans_after_builder_disposal():
    prepared = _prepared_without_builder()
    plan = dict(prepared.plan_summary())
    assert plan == {
        "direct": "compiled",
        "outcome_shift": "not_identified:structurally_unidentified",
        "standardize": "compiled",
    }
    report = json.loads(prepared.estimate())
    assert [s["status"] for s in report["scenarios"]] == [
        "identified",
        "structurally_unidentified",
        "identified",
    ]


def test_structural_envelope_executes_from_retained_plans_after_builder_disposal():
    prepared = _prepared_without_builder()
    plan = dict(prepared.plan_summary())
    assert plan["standardize"] == "compiled"
    envelope = json.loads(prepared.estimate())["envelope"]["means"][0]
    assert (envelope["lower"], envelope["upper"]) == pytest.approx((STANDARDIZED, DIRECT))


def test_weighted_report_executes_from_retained_plans_after_builder_disposal():
    prepared = _prepared_without_builder((0.3, 0.2, 0.4))
    plan = [kind for _, kind in prepared.plan_summary()]
    assert plan.count("compiled") == 2
    weighted = json.loads(prepared.estimate())["weighted"]
    assert weighted["unaccounted_mass"] == pytest.approx(0.5)
    assert weighted["identified_weighted_sums"]["y"] == pytest.approx(
        0.3 * STANDARDIZED + 0.2 * DIRECT
    )


def test_consume_rechecks_the_report_from_retained_plans_after_builder_disposal():
    prepared = _prepared_without_builder((0.3, 0.2, 0.4))
    plan = prepared.plan_summary()
    assert len(plan) == 3
    live = json.loads(prepared.estimate())
    consumed = json.loads(transport.consume_transport_scenarios_artifact(prepared.export()))
    assert consumed["scenarios"] == live["scenarios"]
    assert consumed["weighted"] == live["weighted"]
