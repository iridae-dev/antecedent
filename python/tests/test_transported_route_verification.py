"""Verification of the transported policy, period-shift and adaptive-trial evaluations.

Each ``verify_*`` re-derives the route from its request and compares the stored identity and
values bit for bit, so a stored value that was edited, or a request that differs from the one
that produced it, is refused with the route's own ``value_mismatch`` detail.
"""

from __future__ import annotations

import pytest
from antecedent import transported_constructors as tc
from antecedent.errors import CausalTypeError, CausalValueError
from antecedent.transported_response import TransportedResponseRefusal

from test_transported_constructors import (
    MECHANISMS,
    build_trial,
    diagram,
    model,
    policy,
    quantity,
    trial_design,
    trial_diagram,
)


def _policy_request(m, target_initial=(0.2, 0.8)):
    return dict(target_population="plant_b", target_initial=target_initial, diagram=diagram())


def _built_policy(m):
    return tc.from_policy(
        m,
        policy(m),
        coordinate=quantity("value", "plant_b"),
        provider_id="plant-transport",
        **_policy_request(m),
    )


def test_a_policy_evaluation_replays_and_an_edited_value_does_not() -> None:
    m = model()
    built = _built_policy(m)
    assert tc.verify_policy(m, policy(m), built, **_policy_request(m)) is None

    edited = dict(built.evaluation)
    edited["target_value"] = edited["target_value"] + 1.0
    with pytest.raises(TransportedResponseRefusal) as raised:
        tc.verify_policy(m, policy(m), edited, **_policy_request(m))
    assert raised.value.detail == "transported_policy.value_mismatch"


def test_a_policy_evaluation_does_not_replay_under_another_target_law() -> None:
    m = model()
    built = _built_policy(m)
    with pytest.raises(TransportedResponseRefusal) as raised:
        tc.verify_policy(m, policy(m), built, **_policy_request(m, target_initial=(0.3, 0.7)))
    assert raised.value.detail == "transported_policy.value_mismatch"


def test_a_period_shift_replays_and_an_edited_identity_does_not() -> None:
    m = model()
    target = tc.TargetPeriod.of_model(m, "2027")
    invariance = tc.InvarianceDeclaration("2026", "2027", 2, dict(MECHANISMS))
    in_2027 = tc.period_population_label("plant_a", "2027")
    built = tc.from_period_shift(
        m,
        policy(m),
        target=target,
        target_initial=(0.4, 0.6),
        invariance=invariance,
        coordinate=quantity("value", in_2027),
        provider_id="plant-transport",
    )
    request = dict(target=target, target_initial=(0.4, 0.6), invariance=invariance)
    assert tc.verify_period_shift(m, policy(m), built, **request) is None

    edited = dict(built.evaluation)
    edited["identity"] = "0" * 64
    with pytest.raises(TransportedResponseRefusal) as raised:
        tc.verify_period_shift(m, policy(m), edited, **request)
    assert raised.value.detail == "temporal_policy_transport.value_mismatch"


def test_an_adaptive_trial_estimate_replays_and_an_edited_mean_does_not() -> None:
    built = build_trial("contrast")
    request = dict(
        trial_population="trial_sites",
        unit_strata=[u % 2 for u in range(8)],
        stratum_labels=["mild", "severe"],
        target_stratum_law=(0.25, 0.75),
        diagram=trial_diagram(),
    )
    assert tc.verify_adaptive_trial(trial_design(), built, **request) is None

    edited = dict(built.evaluation)
    edited["treatment_mean"] = edited["treatment_mean"] + 0.01
    with pytest.raises(TransportedResponseRefusal) as raised:
        tc.verify_adaptive_trial(trial_design(), edited, **request)
    assert raised.value.detail == "adaptive_transport.value_mismatch"


def test_an_evaluation_of_another_route_or_type_is_refused_before_native_work() -> None:
    m = model()
    built = _built_policy(m)
    target = tc.TargetPeriod.of_model(m, "2027")
    invariance = tc.InvarianceDeclaration("2026", "2027", 2, dict(MECHANISMS))
    with pytest.raises(CausalValueError, match="not 'temporal_policy'"):
        tc.verify_period_shift(
            m, policy(m), built, target=target, target_initial=(0.4, 0.6), invariance=invariance
        )
    with pytest.raises(CausalTypeError, match="evaluation must be a TransportedConstruction"):
        tc.verify_policy(m, policy(m), "not an evaluation", **_policy_request(m))
