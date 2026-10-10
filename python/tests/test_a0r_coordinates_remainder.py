"""A0 remainder: native response coordinates, one label per policy, scoped transport diagnostics.

Hand examples. A response curve over ``grid = [-0.5, 0, 0.5]`` has the coordinates
``do(a=-0.5)``, ``do(a=0)``, ``do(a=0.5)`` (the ``:g`` rendering Python derives), one
per dose, for outcome ``y`` in the declared units. A set or shifted policy answers
one scalar, so it has one coordinate and one label.
"""

from __future__ import annotations

import antecedent
import numpy as np
import pytest
from antecedent import Admg, ResponseCurve, analyze, intervention, transport
from antecedent.external import ExternalRefusal
from antecedent.results.coordinates import response_coordinates

GRID = [-0.5, 0.0, 0.5]


def _curve_data(seed: int = 23) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=400)
    return {"a": a, "y": 2.0 * a + rng.normal(scale=0.2, size=400)}


def _analyze_curve():
    return analyze(_curve_data(), query=ResponseCurve("a", "y", grid=GRID), graph=[("a", "y")])


def test_a0r_native_coordinates_equal_the_python_derivation_and_the_hand_values():
    result = _analyze_curve()
    native = result.response_coordinates(outcome_units="mmHg")
    derived = response_coordinates(ResponseCurve("a", "y", grid=GRID), outcome_units="mmHg")
    assert native == derived
    assert [q.regime_id for q in native] == ["do(a=-0.5)", "do(a=0)", "do(a=0.5)"]
    assert {(q.variable_id, q.units, q.population_id, q.horizon) for q in native} == {
        ("y", "mmHg", "target", 0)
    }
    assert {q.functional_id for q in native} == {"mean"}
    assert {q.transform_id for q in native} == {"identity"}
    # One coordinate per value and per support label, never positional.
    assert len(native) == len(result.support.point_status) == len(GRID)


def test_a0r_population_and_transform_are_declared_not_inferred():
    result = _analyze_curve()
    declared = result.response_coordinates(
        outcome_units="mmHg", population="source", transform="log"
    )
    assert {(q.population_id, q.transform_id) for q in declared} == {("source", "log")}
    query = ResponseCurve("a", "y", grid=GRID)
    assert declared == response_coordinates(
        query, outcome_units="mmHg", population="source", transform="log"
    )


@pytest.mark.parametrize("units", ["", "  "])
def test_a0r_units_are_required(units):
    with pytest.raises(ExternalRefusal) as refusal:
        _analyze_curve().response_coordinates(outcome_units=units)
    assert refusal.value.detail == "coordinate_support.units_required"
    assert refusal.value.reason_code == "invalid_argument"


def _policy_result(spec):
    rng = np.random.default_rng(1701)
    x = rng.normal(size=400)
    a = 0.7 * x + rng.normal(size=400)
    y = 1.0 + 2.0 * a + 0.8 * x + rng.normal(scale=0.1, size=400)
    return antecedent.analyze(
        {"x": x, "a": a, "y": y},
        query=antecedent.InterventionResponse("y", intervention=spec),
        graph=[("x", "a"), ("x", "y"), ("a", "y")],
    )


@pytest.mark.parametrize(
    ("spec", "regime"),
    [(intervention.Set("a", 0.25), "do(a=0.25)"), (intervention.Shift("a", 0.25), "shift(a=0.25)")],
)
def test_a0r_set_and_shifted_policies_have_one_coordinate_and_one_label(spec, regime):
    result = _policy_result(spec)
    assert tuple(result.support.point_status) == ("extrapolative",)
    assert result.support.status == "extrapolative"
    (coordinate,) = result.response_coordinates(outcome_units="mmHg")
    assert coordinate.regime_id == regime
    assert coordinate.functional_id == "mean"


def test_a0r_a_stochastic_policy_is_never_scalarized_into_a_regime():
    result = _policy_result(intervention.Gaussian("a", 0.25, 0.01))
    assert tuple(result.support.point_status) == ("extrapolative",)
    with pytest.raises(ExternalRefusal) as refusal:
        result.response_coordinates(outcome_units="mmHg")
    assert refusal.value.detail == "coordinate_support.regime_not_describable"
    assert refusal.value.reason_code == "route_not_supported"


def test_a0r_a_jacobian_refuses_with_a_typed_coordinate_support_detail():
    rng = np.random.default_rng(29)
    x = rng.normal(size=500)
    a = 0.5 * x + rng.normal(size=500)
    b = -0.2 * x + rng.normal(size=500)
    y = 8.0 + 1.5 * a - 0.75 * b + x + rng.normal(scale=0.15, size=500)
    result = antecedent.analyze(
        {"x": x, "a": a, "b": b, "y": y},
        query=antecedent.ResponseJacobian(["a", "b"], ["y"], at=[0.0, 0.0]),
        graph=[("x", "a"), ("x", "b"), ("x", "y"), ("a", "y"), ("b", "y")],
    )
    with pytest.raises(ExternalRefusal) as refusal:
        result.response_coordinates(outcome_units="mmHg")
    assert refusal.value.detail == "coordinate_support.functional_not_described"
    assert refusal.value.supplied == "jacobian"


def _transported_with_a_missing_coordinate():
    query = transport.Transport(
        ResponseCurve("x", "y", grid=[0.0, 1.0]),
        target="target",
        evidence=transport.Evidence(
            source=transport.Source(
                "source", kind="experimental", interventions=["x"], sampling="independent"
            ),
            target_sampling="representative_sample",
        ),
    )
    # No law under do(x = 0): coordinate 0 has no value.
    data = transport.ExactTransportData(
        (
            transport.ExactDiscreteLaw(
                "source",
                "source",
                (("y", (0.0, 1.0)),),
                (0.2, 0.8),
                "v1",
                interventions=(("x", 1.0),),
            ),
        )
    )
    return analyze(data, graph=Admg.from_edges(["x", "y"], [("x", "y")]), query=query)


def test_a0r_transport_grid_diagnostics_are_scoped_to_their_coordinate():
    view = _transported_with_a_missing_coordinate()
    labels = tuple(view.support.point_status)
    assert labels == ("missing_evidence", "supported")
    by_id = {d.id: d for d in view.support.diagnostics}
    # The failure record for requested coordinate 0 marks coordinate 0 and no other.
    located = by_id["grid:0"]
    assert located.scope == "per_coordinate"
    assert tuple(located.values) == (1.0, 0.0)
    assert "grid:1" not in by_id
    # No diagnostic that varies by coordinate is global, and every per-coordinate one
    # lines up with the requested coordinates.
    grid_records = [d for d in view.support.diagnostics if d.id.startswith("grid:")]
    assert grid_records
    assert all(d.scope == "per_coordinate" for d in grid_records)
    for diagnostic in view.support.diagnostics:
        if diagnostic.scope == "per_coordinate":
            assert len(diagnostic.values) == len(labels), diagnostic.id
    # Quantities this route does not compute per point are marked, not omitted.
    for name in ("transport.propensity_density", "transport.source_target_overlap"):
        assert by_id[name].scope == "inapplicable"
        assert tuple(by_id[name].values) == ()


def test_analyze_populates_native_scientific_coordinates_with_declared_units():
    result = analyze(
        _curve_data(),
        query=ResponseCurve("a", "y", grid=GRID),
        graph=[("a", "y")],
        outcome_units="mmHg",
        quantity_population="clinic",
        quantity_transform="identity",
    )
    assert result.quantities == result.response_coordinates(
        outcome_units="mmHg", population="clinic"
    )
    assert len(result.quantities) == len(GRID)


@pytest.mark.parametrize("units", ["", " "])
def test_analyze_coordinate_declarations_refuse_blank_units(units):
    from antecedent.errors import CausalValueError

    with pytest.raises(CausalValueError, match="outcome_units must be non-empty"):
        analyze(
            _curve_data(),
            query=ResponseCurve("a", "y", grid=GRID),
            graph=[("a", "y")],
            outcome_units=units,
        )


def test_declared_response_coordinates_survive_a_fresh_checked_artifact_consumer(tmp_path):
    import json
    import subprocess
    import sys

    from antecedent import artifacts
    from antecedent.errors import CausalSerializationError

    result = analyze(
        _curve_data(),
        query=ResponseCurve("a", "y", grid=GRID),
        graph=[("a", "y")],
        outcome_units="mmHg",
        quantity_population="clinic",
    )
    encoded = result.export()
    decoded = artifacts.loads(encoded)
    assert decoded.payload["response"]["coordinates"] == json.loads(
        json.dumps([quantity._wire() for quantity in result.quantities])
    )
    acceptance = artifacts.accept(encoded)
    assert result.claim_id == bytes(decoded.contract["claim"]["claim_id"]).hex()
    assert result.inspect().claim_id == result.claim_id
    artifact_path = tmp_path / "response.bin"
    artifact_path.write_bytes(encoded)
    consumer = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json, sys
from antecedent import artifacts
payload = open(sys.argv[1], 'rb').read()
print(json.dumps({'acceptance': artifacts.accept(payload),
                  'coordinates': artifacts.loads(payload).payload['response']['coordinates']}))
""",
            str(artifact_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    fresh = json.loads(consumer.stdout)
    assert fresh["acceptance"] == acceptance
    assert fresh["coordinates"] == decoded.payload["response"]["coordinates"]
    import copy

    changed_payload = copy.deepcopy(decoded.payload)
    changed_payload["response"]["coordinates"][0]["units"] = "kg"
    repacked = artifacts.dumps(
        decoded.payload_kind,
        decoded.payload,
        variable_names=decoded.variable_names,
        artifact_id=decoded.artifact_id,
        contract=decoded.contract,
    )
    assert artifacts.accept(repacked) == acceptance
    # The native encoder reuses independent contract verification; a valid
    # coordinate with changed units cannot retain the original body-bound claim.
    with pytest.raises(CausalSerializationError, match="claim.id"):
        artifacts.dumps(
            decoded.payload_kind,
            changed_payload,
            variable_names=decoded.variable_names,
            artifact_id=decoded.artifact_id,
            contract=decoded.contract,
        )
    response = artifacts.loads(
        artifacts.dumps(
            "response_result",
            decoded.payload["response"],
            variable_names=decoded.variable_names,
            artifact_id="coordinate-response",
        )
    )
    assert response.payload["coordinates"] == decoded.payload["response"]["coordinates"]


def test_response_export_refuses_semantic_relabeling_of_the_executed_response():
    from dataclasses import replace

    from antecedent.errors import CausalValueError

    result = analyze(
        _curve_data(),
        query=ResponseCurve("a", "y", grid=GRID),
        graph=[("a", "y")],
        outcome_units="mmHg",
    )
    changed = result.model_copy(
        update={"quantities": (replace(result.quantities[0], horizon=1), *result.quantities[1:])}
    )
    with pytest.raises(CausalValueError, match="executed native response"):
        changed.export()


@pytest.mark.parametrize(
    "metadata", [{"quantity_population": "clinic"}, {"quantity_transform": "log"}]
)
def test_response_metadata_without_declared_units_is_refused(metadata):
    from antecedent.errors import CausalValueError

    with pytest.raises(CausalValueError, match="require declared outcome_units"):
        analyze(
            _curve_data(), query=ResponseCurve("a", "y", grid=GRID), graph=[("a", "y")], **metadata
        )


def test_declared_native_response_coordinates_feed_public_decision_composition():
    from dataclasses import replace

    from antecedent import decision, program_claims

    query = ResponseCurve("a", "y", grid=GRID)
    result = analyze(
        _curve_data(),
        query=query,
        graph=[("a", "y")],
        bootstrap=0,
        outcome_units="kg",
        quantity_population="clinic",
        quantity_transform="log",
    )
    program = program_claims.ProgramBinding.from_response(
        result, outcome_units="kg", dose_units="mg", population="clinic", transform="log"
    )
    claim = program_claims.native_claim(result, program)
    assert claim.coordinates == result.quantities
    contract = decision.Contract(
        actions=(
            decision.Action("A", inputs=(claim.coordinates[0],), utility=decision.x(0)),
            decision.Action("B", inputs=(claim.coordinates[-1],), utility=decision.x(0)),
        ),
        utility_units="util",
        criterion=decision.Criterion.expected_utility(),
        target_population="clinic",
    )
    source = claim.as_decision_source(contract)
    assert source.coordinates == result.quantities
    evaluated = contract.evaluate(source.source)
    by_id = {action.id: action for action in evaluated.actions}
    assert by_id["B"].expected_utility == pytest.approx(result.response.values[-1][0])
    assert evaluated.selected == ("B",)
    for changed in (
        replace(program, outcome_units="lb"),
        replace(program, population_id="elsewhere"),
        replace(program, transform_id="identity"),
    ):
        with pytest.raises(ExternalRefusal) as caught:
            program_claims.native_claim(result, changed)
        assert caught.value.reason_code == "quantity_semantics_mismatch"
        assert caught.value.detail.startswith("program_binding.")
