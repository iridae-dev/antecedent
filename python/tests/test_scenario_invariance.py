"""2.3.0 A1: selection differences and invariances behind each scenario answer.

The fixture is the one of ``test_transport_scenarios``: every scenario shares ``z -> x``,
``z -> y``, ``x -> y`` and ``x <-> y``; the source holds the experiment ``do(x)`` over
``(z, y)`` and the target its observational joint over ``(z, x, y)``. By hand:

* selection on ``z``: no source law answers ``y`` directly, so the answer standardizes,
  ``sum_z P_s(y | do(x), z) P*(z)``: one source factor (``y`` given ``z`` under ``do(x)``,
  rule ``transport.pretreatment_standardize``) and one target factor (``z``);
* no selection: ``P_s(y | do(x))``, one source factor, rule ``transport.direct``, no target
  factor;
* selection on ``y``: the outcome's own mechanism differs, an s-hedge, so an obstruction and no
  invariance list.
"""

import json

import pytest
from antecedent import Admg
from antecedent.errors import CausalUnsupportedError
from antecedent.scenario_invariance import (
    InvarianceReport,
    ScenarioInvarianceRefusal,
    ScenarioSetInvarianceReport,
    invariance_report,
)
from antecedent.transport import advanced as transport

from _refusal import assert_registered_refusal

NAMES = ["z", "x", "y"]
SOURCE = (0.32, 0.08, 0.12, 0.48)  # do(x=1) over (z, y)
TARGET = (0.3, 0.15, 0.2, 0.1, 0.05, 0.05, 0.05, 0.1)  # (z, x, y)


def graph(edges=(("z", "x"), ("z", "y"), ("x", "y")), bidirected=(("x", "y"),)):
    return Admg.from_edges(NAMES, list(edges), list(bidirected))


def coordinates():
    return [transport.VariableCoordinate(name, "binary") for name in NAMES]


def scenarios(*rows):
    return transport.TransportScenarioSet(
        [transport.TransportScenario(name, graph(), list(sel)) for name, sel in rows],
        coordinates(),
    )


def catalog():
    return transport.EvidenceCatalog(
        regimes=[
            transport.EvidenceRegime(
                "trial", "source", kind="experimental", interventions=["x"], measured=["z", "y"]
            ),
            transport.EvidenceRegime("obs", "target", measured=["z", "x", "y"]),
        ]
    )


def laws(with_target=True):
    out = [
        transport.ExactDiscreteLaw(
            "source",
            "trial",
            (("z", (0.0, 1.0)), ("y", (0.0, 1.0))),
            SOURCE,
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


def prepare(sets, data=None):
    return transport.prepare_transport_scenarios(
        sets,
        outcomes=["y"],
        treatments=["x"],
        source="source",
        target="target",
        catalog=catalog(),
        laws=data or laws(),
        at={"x": 1.0},
    )


def estimated(sets=None, data=None):
    stage = prepare(
        sets or scenarios(("standardize", ["z"]), ("direct", []), ("outcome_shift", ["y"])), data
    )
    stage.estimate()
    return stage


def report_of(result, name):
    entry = result.scenario(name)
    assert entry is not None
    return entry.report


def test_a1_invariance_standardize_relies_on_the_source_conditional_and_target_marginal():
    report = report_of(invariance_report(estimated()), "standardize")
    assert isinstance(report, InvarianceReport)
    assert (report.status, report.kind) == ("identified", "identified")
    assert report.selection.targets == ("z",)
    assert report.selection.shared_mechanisms == ("x", "y")
    assert report.selection.directed_edges == (("z", "x"), ("z", "y"), ("x", "y"))
    assert report.selection.bidirected_edges == (("x", "y"),)
    assert report.rules == ("transport.pretreatment_standardize",)
    # y's mechanism is the invariance; z is only conditioned on because it may differ.
    assert report.invariances is not None and len(report.invariances) == 1
    (source,) = report.invariances
    assert source.population == "source"
    assert source.variables == ("y",)
    assert source.conditioned_on == ("z",)
    assert source.do_set == ("x",)
    assert source.invariant_mechanisms == ("y",)
    assert source.district_selection_targets == ()
    assert source.rule == "transport.pretreatment_standardize"
    # P*(z) is a target law read from the observational regime, not an invariance.
    assert report.target_factors is not None and len(report.target_factors) == 1
    (target,) = report.target_factors
    assert (target.variables, target.conditioned_on) == (("z",), ())
    assert target.regime is not None and target.regime != source.regime
    assert report.conditional is None and report.obstruction is None


def test_a1_invariance_no_selection_relies_on_the_direct_source_experiment_alone():
    report = report_of(invariance_report(estimated()), "direct")
    assert report.status == "identified"
    assert report.selection.targets == ()
    assert report.selection.shared_mechanisms == ("z", "x", "y")
    assert report.rules == ("transport.direct",)
    assert report.invariances is not None and len(report.invariances) == 1
    (source,) = report.invariances
    assert (source.variables, source.conditioned_on, source.do_set) == (("y",), (), ("x",))
    assert source.rule == "transport.direct"
    assert source.regime is not None
    assert report.target_factors == ()


def test_a1_invariance_selection_on_the_outcome_is_an_obstruction_without_an_invariance_list():
    report = report_of(invariance_report(estimated()), "outcome_shift")
    assert (report.status, report.kind) == ("structurally_unidentified", "obstructed")
    assert report.invariances is None and report.target_factors is None
    assert report.selection.targets == ("y",)
    witness = report.obstruction
    assert witness is not None and witness.kind == "s_hedge"
    # The hedge is rooted at the outcome, whose own mechanism is selected.
    assert "y" in witness.larger_nodes
    assert witness.selection_targets_in_larger == ("y",)


def test_a1_invariance_envelope_extremes_carry_the_report_of_the_scenario_behind_them():
    result = invariance_report(estimated())
    assert isinstance(result, ScenarioSetInvarianceReport)
    assert {s.name for s in result.scenarios} == {"direct", "outcome_shift", "standardize"}
    lower = result.extreme("y", "lower")
    upper = result.extreme("y", "upper")
    assert lower is not None and upper is not None
    assert (lower.scenario, upper.scenario) == ("standardize", "direct")
    assert lower.value < upper.value
    # 0.35 from the scenario selecting on z (it relies on P_s(y | do(x), z) and P*(z)),
    # and the larger value from the scenario with no selection (P_s(y | do(x))).
    assert lower.report == report_of(result, "standardize")
    assert upper.report == report_of(result, "direct")
    assert lower.report.target_factors and not upper.report.target_factors
    assert result.extreme("nope", "lower") is None


def test_a1_invariance_a_scenario_with_unsupported_laws_still_reports_its_formula():
    # The target regime is in the catalog but its law is not supplied: the evaluated result is
    # unsupported_provider, while the invariances are those of the identified formula.
    result = invariance_report(estimated(data=laws(with_target=False)))
    entry = result.scenario("standardize")
    assert entry is not None
    assert entry.result_status == "unsupported_provider"
    assert entry.report.status == "identified"
    assert entry.report.invariances is not None


def test_a1_invariance_identity_ignores_edge_order_and_scenario_name():
    first = invariance_report(estimated(scenarios(("a", ["z"]))))
    reordered = transport.TransportScenarioSet(
        [
            transport.TransportScenario(
                "b",
                graph(edges=(("x", "y"), ("z", "y"), ("z", "x")), bidirected=(("y", "x"),)),
                ["z"],
            )
        ],
        coordinates(),
    )
    second = invariance_report(estimated(reordered))
    one, two = report_of(first, "a"), report_of(second, "b")
    assert one == two
    assert one.identity == two.identity and one.identity.startswith("inv.v1.")
    # A different selection moves the identity.
    other = report_of(invariance_report(estimated(scenarios(("c", [])))), "c")
    assert other.identity != one.identity


def test_a1_invariance_is_derived_and_leaves_the_stage_and_artifact_unchanged():
    stage = estimated()
    before = stage.export()
    first = invariance_report(stage)
    assert stage.export() == before
    assert invariance_report(stage) == first
    # A refresh with the same decisions reports the same invariances.
    stage.refresh(laws())
    stage.estimate()
    assert invariance_report(stage) == first
    # The report is not part of the scenario report payload.
    assert "invariance" not in json.dumps(json.loads(stage.estimate()))


def test_a1_invariance_requires_an_estimated_scenario_stage():
    stage = prepare(scenarios(("direct", [])))
    with pytest.raises(ScenarioInvarianceRefusal) as raised:
        invariance_report(stage)
    assert raised.value.detail == "scenario_invariance.not_estimated"
    assert raised.value.reason_code == "not_executed"
    assert isinstance(raised.value, CausalUnsupportedError)
    assert_registered_refusal(raised.value)
    with pytest.raises(ScenarioInvarianceRefusal) as wrong:
        invariance_report(object())
    assert wrong.value.detail == "scenario_invariance.wrong_result_type"
    assert_registered_refusal(wrong.value)
