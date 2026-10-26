"""Cross-world execution: one test per Python entry and for the native seal.

Response-type model: ``x = u_x`` with ``P(u_x = 1) = 0.4`` and ``y`` of type
never / complier / defier / always with probabilities ``(0.2, 0.5, 0.1, 0.2)``.
By hand:

* ``P(X=1, Y=0) = 0.4 (0.2 + 0.1) = 0.12`` and, among those units,
  ``P(Y_x0 = 1) = 0.1 / 0.3 = 1/3`` (only a defier answers ``y = 1`` at ``x = 0``);
* ``P(Y_x1 = 0, Y_x0 = 1) = P(defier) = 0.1``;
* ``P(Y_x1 = 1) - P(Y_x0 = 1) = (0.5 + 0.2) - (0.1 + 0.2) = 0.4``.
"""

from __future__ import annotations

import json

import pytest
from antecedent import _native
from antecedent import cross_world_execution as cw
from antecedent.outcomes import require_success

R = cw.CrossWorldExecutionRefusal
Q = (0.2, 0.5, 0.1, 0.2)
RESPONSE = (0, 0, 0, 1, 1, 0, 1, 1)  # never, complier, defier, always; cfg = x


def _scm() -> cw.FiniteSCM:
    return cw.FiniteSCM(
        population="pop-a",
        history="regime-1",
        exogenous=(cw.ExogenousTerm("u_x", (0.6, 0.4)), cw.ExogenousTerm("u_y", Q)),
        variables=(
            cw.FiniteVariable("x", 2, (), "u_x", (0, 1)),
            cw.FiniteVariable("y", 2, ("x",), "u_y", RESPONSE),
        ),
    )


def _do(name: str, level: int) -> cw.World:
    return cw.World(name, {"x": level})


def _request(units, quantity) -> cw.CrossWorldRequest:  # type: ignore[no-untyped-def]
    return cw.CrossWorldRequest(
        scm=_scm(),
        units=units,
        worlds=(_do("x0", 0), _do("x1", 1)),
        quantity=quantity,
    )


def _case() -> cw.CrossWorldRequest:
    return _request((cw.FactualUnit("case", {"x": 1, "y": 0}),), cw.probability("x0", "y", 1))


def test_execute_answers_a_counterfactual_for_the_units_matching_the_evidence() -> None:
    result = require_success(cw.CrossWorldExecutor().execute(_case()))
    assert result.value == pytest.approx(1.0 / 3.0, abs=1e-12)
    assert result.units[0].evidence_probability == pytest.approx(0.12, abs=1e-12)
    assert result.label == "point_only"
    assert result.work["abductions"] == 1


def test_seal_binds_a_coupled_joint_event_to_its_declarations() -> None:
    request = _request(
        (cw.FactualUnit("u"),), cw.joint_probability([("x1", "y", 0), ("x0", "y", 1)])
    )
    sealed = require_success(cw.seal(request, artifact_id="defier-event"))
    assert sealed.value == pytest.approx(0.1, abs=1e-12)
    assert sealed.result.label == "point_only"
    # The product of the marginals, 0.3 * 0.3, is not the coupled event.
    assert abs(sealed.value - 0.09) > 1e-3
    assert len(sealed.to_bytes()) > 0
    with pytest.raises(R) as error:
        sealed.interval()
    assert error.value.reason_code == "cell_not_licensed"


def test_consume_replays_with_a_fresh_executor_and_enforces_the_retained_identity() -> None:
    sealed = require_success(cw.seal(_case()))
    data = sealed.to_bytes()
    consumed = require_success(
        cw.consume_cross_world_execution_artifact(data, expected_identity=sealed.identity)
    )
    assert consumed.result.value == pytest.approx(1.0 / 3.0, abs=1e-12)
    assert consumed.inference_claim == "point_only"
    assert dict(consumed.identity) == dict(sealed.identity)
    other = require_success(
        cw.seal(
            _request((cw.FactualUnit("u"),), cw.joint_probability([("x1", "y", 0), ("x0", "y", 1)]))
        )
    )
    with pytest.raises(R) as error:
        require_success(
            cw.consume_cross_world_execution_artifact(data, expected_identity=other.identity)
        )
    assert error.value.detail == "cross_world_execution.artifact_changed"


def test_the_native_seal_entry_returns_the_sealed_bytes_and_a_replayable_summary() -> None:
    request = _request(
        (cw.FactualUnit("u"),), require_success(cw.mean_contrast(("x1", "y"), ("x0", "y")))
    )
    parts, refusal = _native.cross_world_seal(request.to_json(), "native-artifact", None)
    assert refusal is None and parts is not None
    data, summary = parts
    info = json.loads(summary)
    assert info["result"]["value"] == pytest.approx(0.4, abs=1e-12)
    assert info["inference_claim"] == "point_only"
    # The bytes replay through the consumer to the same value.
    consumed = require_success(cw.consume_cross_world_execution_artifact(bytes(data)))
    assert consumed.result.value == pytest.approx(0.4, abs=1e-12)
