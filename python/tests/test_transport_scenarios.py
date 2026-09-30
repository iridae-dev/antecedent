"""Finite graph/selection scenario sets for one transport question.

Every scenario shares ``z -> x``, ``z -> y``, ``x -> y``, ``x <-> y`` and differs
only in which mechanisms may change: ``standardize`` (selection on ``z``) gives
``sum_z P_s(y | do(x), z) P*(z)``; ``direct`` (none) gives ``P_s(y | do(x))``;
``outcome_shift`` (selection on ``y``) is not transportable. Expected values are
computed here from the laws.
"""

import json
import struct

import antecedent
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


def coordinates(y="binary", cardinality=None, unit=None):
    return [
        transport.VariableCoordinate("z", "binary"),
        transport.VariableCoordinate("x", "binary"),
        transport.VariableCoordinate("y", y, unit=unit, cardinality=cardinality),
    ]


def scenarios(weights=None, names=NAMES, schema=None):
    w = weights or (None, None, None)
    return transport.TransportScenarioSet(
        [
            transport.TransportScenario("standardize", graph(names), ["z"], w[0]),
            transport.TransportScenario("direct", graph(names), [], w[1]),
            transport.TransportScenario("outcome_shift", graph(names), ["y"], w[2]),
        ],
        schema or coordinates(),
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


def _operations_to_decide(sets):
    for steps in range(1, 10_000):
        if json.loads(prepare(sets, max_steps=steps).estimate())["receipt"] is None:
            return steps
    raise AssertionError("small graphs decide within 10k operations")


def test_a_scenario_budget_reports_the_rest_unevaluated():
    direct = _operations_to_decide(
        transport.TransportScenarioSet(
            [transport.TransportScenario("direct", graph(), [])], coordinates()
        )
    )
    report = json.loads(prepare(max_steps=direct).estimate())
    s = by_name(report)
    assert s["direct"]["status"] == "identified"
    assert s["standardize"]["status"] == "unevaluated"
    assert s["standardize"]["detail"] == "scenarios.unevaluated_budget: search.operations"
    assert report["receipt"]["stop"] == "search.operations"
    assert report["receipt"]["explored"] == ["direct"]
    assert report["receipt"]["unevaluated"] == ["outcome_shift", "standardize"]
    # Enough for every scenario alone is not enough for the set.
    assert _operations_to_decide(scenarios()) > direct


def test_scenario_and_variable_order_do_not_change_the_report():
    forward = prepare(scenarios((0.3, 0.2, 0.4)))
    original = scenarios((0.3, 0.2, 0.4))
    shuffled = transport.TransportScenarioSet(
        list(reversed(original.scenarios)), list(reversed(original.coordinates))
    )
    backward = prepare(shuffled)
    assert json.loads(forward.estimate()) == json.loads(backward.estimate())
    # The same graphs declared with their variables in another order.
    reordered = prepare(scenarios((0.3, 0.2, 0.4), names=list(reversed(NAMES))))
    assert json.loads(reordered.estimate()) == json.loads(forward.estimate())


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


PREFIX = b"ANTECEDENT-TRANSPORT-SCENARIOS\x01"


def _head(buf, i):
    """One CBOR item head: (major type, argument, next offset)."""
    major, info = buf[i] >> 5, buf[i] & 31
    if info < 24:
        return major, info, i + 1
    width = {24: 1, 25: 2, 26: 4, 27: 8}[info]
    return major, int.from_bytes(buf[i + 1 : i + 1 + width], "big"), i + 1 + width


def _encode_head(major, n):
    if n < 24:
        return bytes([major << 5 | n])
    for info, width in ((24, 1), (25, 2), (26, 4), (27, 8)):
        if n < 1 << (8 * width):
            return bytes([major << 5 | info]) + n.to_bytes(width, "big")
    raise ValueError(n)


def _unframe(artifact):
    """The frame is a CBOR pair: the variable names and the artifact bytes."""
    body = artifact[len(PREFIX) :]
    _, _, i = _head(body, 0)
    _, count, i = _head(body, i)
    names = []
    for _ in range(count):
        _, length, i = _head(body, i)
        names.append(body[i : i + length].decode())
        i += length
    major, length, i = _head(body, i)
    if major == 2:
        return names, bytearray(body[i : i + length])
    inner = bytearray()
    for _ in range(length):
        _, value, i = _head(body, i)
        inner.append(value)
    return names, inner


def _frame(names, inner):
    out = bytearray(PREFIX) + _encode_head(4, 2) + _encode_head(4, len(names))
    for name in names:
        out += _encode_head(3, len(name.encode())) + name.encode()
    out += _encode_head(4, len(inner))
    for value in inner:
        out += _encode_head(0, value)
    return bytes(out)


def test_semantic_artifact_mutations_fail_consume_with_typed_errors():
    prepared = prepare(scenarios((0.3, 0.2, 0.4)))
    prepared.estimate()
    artifact = prepared.export()
    names, inner = _unframe(artifact)
    assert names == NAMES and _frame(names, inner) == artifact
    # Relabel the variables: z and y swap names. The names are bound into the
    # artifact's identity through its coordinate schema, so the frame cannot
    # silently relabel the replayed report.
    with pytest.raises(CausalSerializationError, match="coordinate schema"):
        transport.consume_transport_scenarios_artifact(_frame(["y", "x", "z"], inner))
    # Change one declared weight (0.3 -> 0.35) inside the stored scenario set.
    weight = b"\xfb" + struct.pack(">d", 0.3)
    reweighted = bytes(inner).replace(weight, b"\xfb" + struct.pack(">d", 0.35), 1)
    assert reweighted != bytes(inner)
    with pytest.raises(CausalSerializationError, match="premises digest"):
        transport.consume_transport_scenarios_artifact(_frame(names, reweighted))


def test_data_identity_binds_law_snapshots_and_the_provider():
    prepared = prepare(scenarios((0.3, 0.2, 0.4)))
    prepared.estimate()
    names, inner = _unframe(prepared.export())
    # A snapshot label no report shows: its edit is a data-identity failure, not
    # a scientific-premises one (the premises digest is untouched).
    relabelled = bytes(inner).replace(b"\x65trial", b"\x65other", 1)
    assert relabelled != bytes(inner)
    with pytest.raises(CausalSerializationError, match="data identity"):
        transport.consume_transport_scenarios_artifact(_frame(names, relabelled))
    provider = bytes(inner).replace(
        b"transport.exact_supplied_laws", b"transport.exact_supplied_lawz", 1
    )
    assert provider != bytes(inner)
    with pytest.raises(CausalSerializationError, match="data identity"):
        transport.consume_transport_scenarios_artifact(_frame(names, provider))


def test_cross_scenario_inference_and_equivalence_classes_are_refused():
    prepared = prepare()
    with pytest.raises(CausalUnsupportedError, match="scenarios.shared_data_aggregate") as refused:
        prepared.aggregate_interval()
    assert refused.value.reason_code == "scenario_aggregate_not_licensed"
    cpdag = Cpdag.from_directed_undirected(NAMES, [("z", "x")], [("x", "y")])
    with pytest.raises(
        CausalUnsupportedError, match="scenarios.equivalence_class_input"
    ) as equivalence:
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


def _schema_refusal(excinfo):
    assert excinfo.value.reason_code == "schema_mismatch"
    assert "scenarios.coordinate_mismatch" in str(excinfo.value)


@pytest.mark.parametrize(
    "override",
    [
        coordinates(y="categorical", cardinality=3),
        coordinates(y="continuous"),
        coordinates(unit="mmHg"),
    ],
    ids=["cardinality", "domain", "unit"],
)
def test_scenarios_disagreeing_on_the_schema_refuse_with_schema_mismatch(override):
    base = scenarios()
    changed = transport.TransportScenario("direct", graph(), [], coordinates=override)
    mixed = transport.TransportScenarioSet(
        [base.scenarios[0], changed, base.scenarios[2]], base.coordinates
    )
    with pytest.raises(CausalValueError) as refused:
        prepare(mixed)
    _schema_refusal(refused)


def test_scenarios_disagreeing_on_variable_names_refuse_with_schema_mismatch():
    renamed = Admg.from_edges(
        ["z", "x", "outcome"], [("z", "x"), ("z", "outcome"), ("x", "outcome")], []
    )
    base = scenarios()
    other = transport.TransportScenario("direct", renamed, [])
    with pytest.raises(CausalValueError) as refused:
        prepare(transport.TransportScenarioSet([base.scenarios[0], other], base.coordinates))
    _schema_refusal(refused)
    # A schema naming a variable no graph has.
    wrong = [*coordinates()[:2], transport.VariableCoordinate("outcome", "binary")]
    with pytest.raises(CausalValueError) as refused:
        prepare(scenarios(schema=wrong))
    _schema_refusal(refused)


def test_requests_outside_the_declared_domain_refuse_with_schema_mismatch():
    with pytest.raises(CausalValueError) as refused:
        transport.prepare_transport_scenarios(
            scenarios(),
            outcomes=["y"],
            treatments=["x"],
            source="source",
            target="target",
            catalog=catalog(),
            laws=laws(),
            at={"x": 2.0},
        )
    _schema_refusal(refused)


def test_weighted_ranges_use_the_declared_outcome_domain():
    # y is declared with three levels; no law realizes y = 2, so no atom has it,
    # yet the unaccounted half may still sit there.
    report = json.loads(
        prepare(
            scenarios((0.3, 0.2, 0.4), schema=coordinates(y="categorical", cardinality=3))
        ).estimate()
    )
    total = 0.3 * STANDARDIZED + 0.2 * DIRECT
    rng = report["weighted"]["ranges"][0]
    assert (rng["lower"], rng["upper"]) == pytest.approx((total, total + 0.5 * 2))


def test_a_support_failure_is_scenario_local():
    report = json.loads(prepare(data=laws(source=(0.4, 0.6, 0.0, 0.0))).estimate())
    s = by_name(report)
    assert s["standardize"]["status"] == "support_failure"
    assert s["direct"]["status"] == "identified"
    assert mean(s["direct"]) == pytest.approx(0.6)


def test_the_shared_budget_observes_memory_depth_and_cancellation():
    memory = json.loads(prepare(memory_bytes=1500).estimate(memory_bytes=None))
    assert memory["receipt"]["stop"] == "search.memory"
    # The first scenario entered stops mid-search: it is unevaluated, not explored.
    assert memory["receipt"]["explored"] == []
    assert memory["receipt"]["unevaluated"] == ["direct", "outcome_shift", "standardize"]
    depth = json.loads(prepare(max_depth=0).estimate())
    assert depth["receipt"]["stop"] == "search.depth"
    assert {s["status"] for s in depth["scenarios"]} == {"unevaluated"}
    token = antecedent.state.CancellationToken()
    token.cancel()
    cancelled = json.loads(prepare(cancel=token).estimate())
    assert cancelled["receipt"]["stop"] == "search.cancelled"
    assert cancelled["receipt"]["unevaluated"] == ["direct", "outcome_shift", "standardize"]
    assert {s["detail"] for s in cancelled["scenarios"]} == {
        "scenarios.unevaluated_budget: search.cancelled"
    }


def test_a_memory_truncated_report_replays_but_a_cancelled_one_is_not_exported():
    # A recorded memory bound is reproduced by the consumer, which replays the
    # identical prefix even though its own context sets no limit.
    prepared = prepare(memory_bytes=1500)
    live = json.loads(prepared.estimate())
    consumed = json.loads(transport.consume_transport_scenarios_artifact(prepared.export()))
    assert consumed["receipt"] == live["receipt"]
    assert consumed["scenarios"] == live["scenarios"]
    # An operation bound is recorded and replays too.
    prepared = prepare(max_steps=25)
    live = json.loads(prepared.estimate())
    assert live["receipt"]["stop"] == "search.operations"
    consumed = json.loads(transport.consume_transport_scenarios_artifact(prepared.export()))
    assert consumed["receipt"] == live["receipt"]
    # Cancellation falls where nothing recorded lets a consumer reproduce, so the
    # exporter refuses rather than write an artifact that can only fail replay.
    token = antecedent.state.CancellationToken()
    token.cancel()
    cancelled = prepare(cancel=token)
    cancelled.estimate()
    with pytest.raises(CausalSerializationError, match="cancellation"):
        cancelled.export()


def test_the_frozen_scenario_bound_refuses_65_scenarios():
    many = [transport.TransportScenario(f"s{i:02}", graph(), []) for i in range(65)]
    with pytest.raises(CausalUnsupportedError, match="scenarios.count") as refused:
        transport.TransportScenarioSet(many, coordinates())
    assert refused.value.reason_code == "route_not_supported"


def _environment(population):
    return transport.Environment(
        population, [transport.VariableCoordinate(n, "binary") for n in NAMES]
    )


def test_empirical_plug_in_points_per_scenario():
    cat = transport.EvidenceCatalog(
        environments=[_environment("source"), _environment("target")],
        regimes=catalog().regimes,
    )
    trial = transport.RegimeSample(
        "source",
        "trial",
        "trial",
        {"z": [0.0, 0.0, 1.0, 1.0, 1.0], "y": [0.0, 1.0, 1.0, 1.0, 0.0]},
        interventions=(("x", 1.0),),
    )
    target = transport.RegimeSample(
        "target",
        "obs",
        "target",
        {"z": [0.0, 0.0, 0.0, 1.0], "x": [0.0, 1.0, 0.0, 1.0], "y": [0.0, 1.0, 1.0, 0.0]},
    )
    data = transport.StatisticalTransportData(samples=(trial, target))
    prepared = prepare(cat=cat, data=data)
    report = json.loads(prepared.estimate())
    s = by_name(report)
    assert mean(s["direct"]) == pytest.approx(0.6)
    assert mean(s["standardize"]) == pytest.approx(0.75 * 0.5 + 0.25 * 2 / 3)
    assert s["outcome_shift"]["status"] == "structurally_unidentified"
    consumed = json.loads(transport.consume_transport_scenarios_artifact(prepared.export()))
    assert consumed["scenarios"] == report["scenarios"]
