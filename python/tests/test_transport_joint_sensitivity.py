"""2.2B X3: joint mechanism deviations of the registered surrogate z formula (Python surface)."""

import json

import pytest
from antecedent import Admg
from antecedent.errors import CausalSerializationError, CausalUnsupportedError, CausalValueError
from antecedent.transport import advanced as transport

P_W = {0: 0.65, 1: 0.35}
P_Y1 = {(0, 0): 0.2, (0, 1): 0.5, (1, 0): 0.3, (1, 1): 0.8}


def fixture():
    names = ["w", "z", "x", "y"]
    graph = Admg.from_edges(
        names,
        [("w", "z"), ("z", "x"), ("x", "y"), ("w", "y")],
        [("w", "y"), ("z", "y"), ("z", "x")],
    )
    query = transport.ZTransportQuery(
        transport.SelectionDiagram("source", "target", []),
        outcomes=["y"],
        treatments=["x"],
        controllable=["z"],
        experiment_assignment={"z": 0.0},
    )
    builder = transport.identify_z_transport(graph=graph, query=query)
    assert builder.outcome == "identified"
    coordinates = tuple(transport.VariableCoordinate(name, "binary") for name in names)
    catalog = transport.EvidenceCatalog(
        environments=(transport.Environment("source", coordinates),),
        regimes=(
            transport.EvidenceRegime(
                "do_z_0",
                "source",
                kind="experimental",
                interventions=["z"],
                intervention_values={"z": 0.0},
                measured=names,
            ),
        ),
        bindings=(
            transport.RegimeBinding(
                "do_z_0",
                "snapshot_z0",
                schema_names=names,
                sampling="independent",
                dependence="independent_studies",
            ),
        ),
    )
    probabilities = []
    for w in (0, 1):
        for x in (0, 1):
            for y in (0, 1):
                p_y = P_Y1[(w, x)] if y else 1.0 - P_Y1[(w, x)]
                probabilities.append(P_W[w] * (0.4 if x else 0.6) * p_y)
    law = transport.ExactDiscreteLaw(
        "source",
        "do_z_0",
        (("w", (0.0, 1.0)), ("x", (0.0, 1.0)), ("y", (0.0, 1.0))),
        tuple(probabilities),
        "snapshot_z0",
        interventions=(("z", 0.0),),
    )
    return graph, builder, catalog, (law,)


def closed_form(kernel, parent):
    """Independent closed form of the joint range for a binary outcome (spread 1)."""
    delta = {w: P_Y1[(w, 1)] - P_Y1[(w, 0)] for w in (0, 1)}
    up = {w: (1 - kernel) * delta[w] + kernel for w in (0, 1)}
    down = {w: (1 - kernel) * delta[w] - kernel for w in (0, 1)}
    upper = (1 - parent) * sum(P_W[w] * up[w] for w in (0, 1)) + parent * max(up.values())
    lower = (1 - parent) * sum(P_W[w] * down[w] for w in (0, 1)) + parent * min(down.values())
    return lower, upper


DEVIATION = transport.JointDeviation(
    {"shared_parent_marginal": 0.3, "outcome_kernel": 0.2},
    decision_threshold=0.5,
    frontier_points=9,
)


def prepared_stage():
    graph_builder, builder, catalog, laws = fixture()
    prepared = builder.prepare_exact(catalog, laws, {"x": 1.0})
    del builder, graph_builder
    return prepared


def test_joint_sensitivity_executes_the_retained_plan_after_builder_disposal():
    graph_builder, builder, catalog, laws = fixture()
    program = builder.inspect_proof(catalog)
    assert program["rules"] == ["ztr.surrogate_factorization"]
    prepared = builder.prepare_exact(catalog, laws, {"x": 1.0})
    del builder, graph_builder
    plan = json.loads(prepared.estimate())
    assert plan["status"] == "available"
    joint = transport.joint_mechanism_sensitivity(prepared, DEVIATION)
    lower, upper = closed_form(0.2, 0.3)
    assert joint["assumption_range"]["minimum"] == pytest.approx(lower, abs=1e-12)
    assert joint["assumption_range"]["maximum"] == pytest.approx(upper, abs=1e-12)
    assert joint["baseline"] == pytest.approx(0.65 * 0.3 + 0.35 * 0.5, abs=1e-12)
    assert joint["inference_claim"] == "assumption_range"
    assert "not a confidence interval" in joint["interpretation"]
    assert joint["uncertainty"]["status"] == "withheld"
    assert joint["uncertainty"]["reason_code"] == "cell_not_licensed"
    assert [f["factor"] for f in joint["factors"]] == ["outcome_kernel", "shared_parent_marginal"]
    assert len(joint["frontier"]) == 9
    assert joint["receipt"]["stop"] is None
    # The kernel axis is the 2.1 one-factor tipping point of the same stage.
    one_factor = prepared.mechanism_sensitivity(0.2, 0.45)
    kernel_only = transport.joint_mechanism_sensitivity(
        prepared, transport.JointDeviation({"outcome_kernel": 0.2}, decision_threshold=0.45)
    )
    assert kernel_only["assumption_range"] == one_factor["assumption_range"]
    assert kernel_only["axis_tipping"][0]["analytic"] == one_factor["tipping_fraction"]


def test_export_joint_sensitivity_after_builder_disposal():
    prepared = prepared_stage()
    builder = None
    with pytest.raises(Exception, match="estimate before exporting"):
        transport.export_joint_mechanism_sensitivity(prepared, DEVIATION)
    plan = json.loads(prepared.estimate())
    assert plan["status"] == "available" and builder is None
    artifact = transport.export_joint_mechanism_sensitivity(prepared, DEVIATION)
    assert isinstance(artifact, bytes) and len(artifact) > 100
    # The v2 one-factor consumer refuses the v3 artifact by version.
    with pytest.raises(CausalSerializationError):
        transport.consume_z_transport_sensitivity_artifact(artifact)


def test_consume_joint_sensitivity_artifact_replays_the_retained_plan():
    prepared = prepared_stage()
    builder = None
    plan = json.loads(prepared.estimate())
    assert plan["status"] == "available" and builder is None
    artifact = transport.export_joint_mechanism_sensitivity(prepared, DEVIATION)
    consumed = transport.consume_joint_mechanism_sensitivity_artifact(artifact)
    live = transport.joint_mechanism_sensitivity(prepared, DEVIATION)
    assert consumed["version"] == 3
    assert consumed["outcome"]["range"] == [
        live["assumption_range"]["minimum"],
        live["assumption_range"]["maximum"],
    ]
    assert consumed["inference_claim"] == "assumption_range"
    assert consumed["sampling"]["interval"] is None
    assert consumed["sampling"]["status"] == "withheld"
    assert consumed["receipt"]["operations_consumed"] == live["receipt"]["operations_consumed"]


def test_prepare_estimate_joint_sensitivity_export_and_independent_consume():
    prepared = prepared_stage()
    prepared.estimate()
    artifact = transport.export_joint_mechanism_sensitivity(prepared, DEVIATION)
    del prepared
    consumed = transport.consume_joint_mechanism_sensitivity_artifact(artifact)
    lower, upper = closed_form(0.2, 0.3)
    assert consumed["outcome"]["range"][0] == pytest.approx(lower, abs=1e-12)
    assert consumed["outcome"]["range"][1] == pytest.approx(upper, abs=1e-12)
    assert consumed["premises_digest"] and consumed["data_digest"]
    # Stored limits above the consumer's maxima refuse before any replay.
    with pytest.raises(CausalUnsupportedError) as caught:
        transport.consume_joint_mechanism_sensitivity_artifact(artifact, max_search_operations=10)
    assert caught.value.reason_code == "route_not_supported"
    assert "joint_sensitivity.consumer_limits" in str(caught.value)


def test_the_unmeasured_interval_route_refuses_with_cell_not_licensed():
    prepared = prepared_stage()
    with pytest.raises(CausalUnsupportedError) as caught:
        transport.joint_mechanism_sensitivity_interval(prepared, DEVIATION)
    assert caught.value.reason_code == "cell_not_licensed"
    assert "joint_sensitivity.interval_withheld" in str(caught.value)


def test_out_of_scope_requests_refuse_with_reason_codes():
    prepared = prepared_stage()
    cases = [
        (
            transport.JointDeviation({"outcome_kernel": 0.1}, total_budget=0.1),
            CausalUnsupportedError,
            "route_not_supported",
            "joint_sensitivity.budget_coupling",
        ),
        (
            transport.JointDeviation({"outcome_kernel": 0.1, "fixed_graph_parent": 0.1}),
            CausalUnsupportedError,
            "route_not_supported",
            "joint_sensitivity.fixed_graph_parent",
        ),
        (
            transport.JointDeviation({"fixed_graph_conditional": 0.1}),
            CausalUnsupportedError,
            "route_not_supported",
            "joint_sensitivity.fixed_graph_conditional",
        ),
        (
            transport.JointDeviation({"source_target_discrepancy": 0.1}),
            CausalUnsupportedError,
            "route_not_supported",
            "joint_sensitivity.source_target_discrepancy",
        ),
        (
            transport.JointDeviation({"treatment_mechanism": 0.1}),
            CausalValueError,
            "invalid_argument",
            "joint_sensitivity.treatment_factor",
        ),
        (
            transport.JointDeviation({"outcome_kernel": 1.5}),
            CausalValueError,
            "invalid_argument",
            "joint_sensitivity.invalid_fraction",
        ),
        (
            transport.JointDeviation({"latent_mechanism": 0.1}),
            CausalValueError,
            "invalid_argument",
            "joint_sensitivity.unknown_factor",
        ),
        (
            transport.JointDeviation({"outcome_kernel": 0.1}, frontier_points=34),
            CausalUnsupportedError,
            "route_not_supported",
            "joint_sensitivity.bounds_exceeded",
        ),
        (
            transport.JointDeviation({}),
            CausalUnsupportedError,
            "route_not_supported",
            "joint_sensitivity.factor_count",
        ),
    ]
    for deviation, error, code, detail in cases:
        with pytest.raises(error) as caught:
            transport.joint_mechanism_sensitivity(prepared, deviation)
        assert caught.value.reason_code == code, detail
        assert detail in str(caught.value)
    # The factor cap is two: a third declaration of any kind refuses by count.
    for third in ("treatment_mechanism", "fixed_graph_parent", "source_target_discrepancy"):
        deviation = transport.JointDeviation(
            {"outcome_kernel": 0.1, "shared_parent_marginal": 0.1, third: 0.1}
        )
        with pytest.raises(
            CausalUnsupportedError, match="joint_sensitivity.factor_count"
        ) as caught:
            transport.joint_mechanism_sensitivity(prepared, deviation)
        assert caught.value.reason_code == "route_not_supported"
    # A non-finite threshold refuses at construction with the same reason code and
    # detail as the Rust contract (invalid_argument / joint_sensitivity.invalid_threshold).
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(CausalValueError, match="joint_sensitivity.invalid_threshold") as caught:
            transport.JointDeviation({"outcome_kernel": 0.1}, decision_threshold=bad)
        assert caught.value.reason_code == "invalid_argument"


def test_tampered_joint_artifact_fails_with_typed_errors():
    prepared = prepared_stage()
    prepared.estimate()
    artifact = transport.export_joint_mechanism_sensitivity(prepared, DEVIATION)
    # A flipped byte inside the embedded baseline changes the data identity.
    marker = b"snapshot_z0"
    at = artifact.index(marker)
    tampered = artifact[:at] + b"snapshot_z1" + artifact[at + len(marker) :]
    with pytest.raises((CausalUnsupportedError, CausalSerializationError)) as caught:
        transport.consume_joint_mechanism_sensitivity_artifact(tampered)
    assert getattr(caught.value, "reason_code", None) == "transport_not_certified"
    assert "joint_sensitivity." in str(caught.value)
    # Truncation is a decoding failure.
    with pytest.raises(CausalSerializationError):
        transport.consume_joint_mechanism_sensitivity_artifact(artifact[:-7])


def test_a_budget_or_cancellation_stop_keeps_the_exact_range_and_its_receipt():
    from antecedent.errors import CausalError
    from antecedent.state import CancellationToken

    prepared = prepared_stage()
    prepared.estimate()
    full = transport.joint_mechanism_sensitivity(prepared, DEVIATION)
    tight = transport.JointDeviation(
        {"shared_parent_marginal": 0.3, "outcome_kernel": 0.2},
        decision_threshold=0.5,
        frontier_points=9,
        max_operations=6,
    )
    stopped = transport.joint_mechanism_sensitivity(prepared, tight)
    # The range is exact whatever the frontier budget; the frontier is not.
    assert stopped["assumption_range"] == full["assumption_range"]
    assert stopped["receipt"]["stop"] == "search.operations"
    assert stopped["unresolved_detail"] == "joint_sensitivity.budget"
    assert stopped["receipt"]["unevaluated"]
    assert any(point["status"] == "unevaluated" for point in stopped["frontier"])
    assert len(stopped["frontier"]) == len(full["frontier"])
    token = CancellationToken()
    token.cancel()
    with pytest.raises(CausalError, match="joint_sensitivity.budget") as cancelled:
        transport.joint_mechanism_sensitivity(prepared, DEVIATION, cancel=token)
    assert cancelled.value.reason_code == "transport_budget_cancel"
