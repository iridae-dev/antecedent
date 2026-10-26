"""Sealing and consuming a sequential evaluation, each against hand-solved values.

Horizon two, one unit that is ``sound`` or ``faulty`` (one half each); ``ship``
pays +10 / -10, ``repair`` pays +6 / -4, ``inspect`` costs 1/4 and raises an
alarm with probability 3/4 when faulty and 1/4 when sound. Hand values:
ship now ``0``, repair now ``1``; after an alarm ``P(faulty) = 3/4`` (repair
``-3/2``) and after no alarm ``P(faulty) = 1/4`` (ship ``5``), so inspecting then
choosing is worth ``-1/4 + 1/2 (-3/2) + 1/2 (5) = 3/2``.
"""

from __future__ import annotations

import pytest
from antecedent import measurement_models as mm
from antecedent import sequential_evaluation as se
from antecedent.errors import CausalTypeError, CausalValueError
from antecedent.outcomes import require_success

WAIT, INSPECT, SHIP, REPAIR = range(4)
NEGATIVE, ALARM = 0, 1
TOL = 1e-12
IDENTITY = ((1.0, 0.0), (0.0, 1.0))


def _alarm_panel() -> mm.MeasurementModel:
    return mm.MeasurementModel(
        indicator=mm.MeasurementVariable("alarm", "noisy_misclassified", "category"),
        construct=mm.MeasurementVariable("unit_condition", "latent_scientific", "category"),
        parameters=mm.BinaryParameters(
            rates=(mm.BinaryRates(sensitivity=0.75, specificity=0.75),), min_youden=0.1
        ),
        population_id="plant_a",
        regime_id="operations",
        evidence=mm.ParameterEvidence.declared("reliability_engineering"),
    )


def _problem() -> se.FiniteModel:
    return se.FiniteModel(
        population="plant_a",
        regime="operations",
        period="2026",
        state_variable="unit_condition",
        states=("sound", "faulty"),
        initial=(0.5, 0.5),
        horizon=2,
        actions=(
            se.Action("wait", "continue", transition=((0.75, 0.25), (0.0, 1.0))),
            se.Action(
                "inspect",
                "continue",
                transition=IDENTITY,
                utility_cost=0.25,
                channel=_alarm_panel(),
            ),
            se.Action("ship", "commit", transition=IDENTITY, immediate_utility=(10.0, -10.0)),
            se.Action("repair", "commit", transition=IDENTITY, immediate_utility=(6.0, -4.0)),
        ),
        terminal_utility=(0.0, 0.0),
        resource_budget=None,
        stopping=(),
    )


def _policy(version: str, entries: dict) -> se.Policy:
    return se.Policy.deterministic(version, entries)


def _inspect_then_choose() -> se.Policy:
    return _policy(
        "inspect_then_choose_v1",
        {(): INSPECT, ((INSPECT, ALARM),): REPAIR, ((INSPECT, NEGATIVE),): SHIP},
    )


def test_seal_evaluates_and_binds_the_policy_value_to_its_identities() -> None:
    artifact = require_success(se.seal(_problem(), _policy("repair_now_v1", {(): REPAIR})))
    assert artifact.value == pytest.approx(1.0, abs=TOL)
    assert artifact.summary["policy_version"] == "repair_now_v1"
    # A policy-only artifact carries neither estimate.
    assert artifact.summary["off_policy"] is None and artifact.summary["rollout"] is None
    assert set(artifact.identities) >= {"model", "policy", "evaluation"}
    # Another policy on the same model is another policy and evaluation identity.
    shipped = require_success(se.seal(_problem(), _policy("ship_now_v1", {(): SHIP})))
    assert shipped.value == pytest.approx(0.0, abs=TOL)
    assert shipped.identities["model"] == artifact.identities["model"]
    assert shipped.identities["policy"] != artifact.identities["policy"]
    assert shipped.identities["evaluation"] != artifact.identities["evaluation"]
    # Logged trajectories need the logging policy that generated them.
    with pytest.raises(CausalValueError):
        require_success(se.seal(_problem(), _policy("v", {(): SHIP}), trajectories=[object()]))  # type: ignore[list-item]


def test_consume_replays_the_stored_value_bit_for_bit_and_enforces_retained_identities() -> None:
    artifact = require_success(se.seal(_problem(), _inspect_then_choose()))
    assert artifact.value == pytest.approx(1.5, abs=TOL)
    data = bytes(artifact.export("inspect-then-choose"))
    identities = artifact.identities
    consumed = require_success(
        se.consume_artifact(
            data,
            model=identities["model"],
            policy=identities["policy"],
            evaluation=identities["evaluation"],
        )
    )
    assert consumed.value == artifact.value
    assert consumed.identities == identities
    assert len(consumed.histories) == 3 and len(consumed.decisions) == 3
    assert consumed.policy.version == "inspect_then_choose_v1"
    # A retained identity that is not the artifact's is refused.
    with pytest.raises(se.SequentialEvaluationRefusal) as wrong:
        require_success(se.consume_artifact(data, model="0" * 64))
    assert wrong.value.reason_code == "route_not_supported"
    assert wrong.value.detail == "sequential_evaluation.identity_mismatch"
    # Damaged and truncated bytes are refused, and bytes only.
    damaged = bytearray(data)
    damaged[len(damaged) // 2] ^= 0x55
    with pytest.raises(se.SequentialEvaluationRefusal):
        require_success(se.consume_artifact(bytes(damaged)))
    with pytest.raises(se.SequentialEvaluationRefusal):
        require_success(se.consume_artifact(data[:30]))
    with pytest.raises(CausalTypeError):
        require_success(se.consume_artifact("not bytes"))  # type: ignore[arg-type]
