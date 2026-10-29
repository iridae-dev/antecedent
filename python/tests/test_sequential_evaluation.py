"""Sequential evaluation of a supplied contingent policy, against hand-solved values.

The decision problem is an inspection tree. A unit is ``sound`` or ``faulty`` (one half each).

====== ======== ======== =========================================================
action role     cost     effect
====== ======== ======== =========================================================
wait   continue 0        a sound unit degrades with probability 1/4
inspect continue 1/4     alarm w.p. 3/4 if faulty, 1/4 if sound; uses 1 resource
ship   commit   0        +10 if sound, -10 if faulty
repair commit   4        one decision time later: faulty is sound w.p. 1/2, then +10/-10
pilot  continue 1/4      an outcome [3, -1] two decision times later
====== ======== ======== =========================================================

Horizon 3, terminal utility [2, -2], budget of two inspections, boundaries on ``P(faulty)``: at
least 0.85 forces ``repair`` and at most 0.15 forces ``ship``.

The oracle in this file is written independently of the library: a forward joint-probability
backward induction over latent state and observed history, which never forms a conditional belief
or a conditional expectation of a delayed outcome. Hand values:

* ship now 0, repair now 1 (``0.5*6 + 0.5*(-4)``);
* inspect then repair after an alarm / ship after none: ``-1/4 + 1/2*(-3/2) + 1/2*5 = 3/2``;
* two inspections with forced stops: ``-1/2 + 0.3125*(-3) + 0.3125*8 + 0.375*1 = 1.4375``;
* pilot, inspect, then act: ``9/4``; pilot then ship ``3/4``; pilot then repair ``11/4``;
* wait to the horizon: ``2 - 4*0.7890625 = -1.15625``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace

import pytest
from antecedent import Execution, _native
from antecedent import measurement_models as mm
from antecedent import sequential_evaluation as se
from antecedent.errors import CausalTypeError, CausalValueError
from antecedent.outcomes import require_success

WAIT, INSPECT, SHIP, REPAIR, PILOT = range(5)
NEGATIVE, ALARM = 0, 1
TOL = 1e-12


# --------------------------------------------------------------------------- the problem


def alarm_panel(sensitivity: float = 0.75, specificity: float = 0.75, **overrides):
    fields = dict(
        indicator=mm.MeasurementVariable("alarm", "noisy_misclassified", "category"),
        construct=mm.MeasurementVariable("unit_condition", "latent_scientific", "category"),
        parameters=mm.BinaryParameters(
            rates=(mm.BinaryRates(sensitivity=sensitivity, specificity=specificity),),
            min_youden=0.1,
        ),
        population_id="plant_a",
        regime_id="operations",
        evidence=mm.ParameterEvidence.declared("reliability_engineering"),
    )
    fields.update(overrides)
    return mm.MeasurementModel(**fields)


IDENTITY = ((1.0, 0.0), (0.0, 1.0))


def actions(channel=None):
    return (
        se.Action("wait", "continue", transition=((0.75, 0.25), (0.0, 1.0))),
        se.Action(
            "inspect",
            "continue",
            transition=IDENTITY,
            utility_cost=0.25,
            resource_use=1.0,
            channel=alarm_panel() if channel is None else channel,
        ),
        se.Action("ship", "commit", transition=IDENTITY, immediate_utility=(10.0, -10.0)),
        se.Action(
            "repair",
            "commit",
            transition=((1.0, 0.0), (0.5, 0.5)),
            utility_cost=4.0,
            delayed=se.Delayed(1, (10.0, -10.0)),
        ),
        se.Action(
            "pilot",
            "continue",
            transition=IDENTITY,
            utility_cost=0.25,
            delayed=se.Delayed(2, (3.0, -1.0)),
        ),
    )


def model(boundaries: bool = True, **overrides) -> se.FiniteModel:
    stopping = (
        (
            se.StoppingBoundary("confident_faulty", (1,), REPAIR, upper=0.85),
            se.StoppingBoundary("confident_sound", (1,), SHIP, lower=0.15),
        )
        if boundaries
        else ()
    )
    fields = dict(
        population="plant_a",
        regime="operations",
        period="2026",
        state_variable="unit_condition",
        states=("sound", "faulty"),
        initial=(0.5, 0.5),
        horizon=3,
        actions=actions(),
        terminal_utility=(2.0, -2.0),
        resource_budget=2.0,
        stopping=stopping,
    )
    fields.update(overrides)
    return se.FiniteModel(**fields)


def table(**entries):
    return entries


def det(version: str, entries: dict) -> se.Policy:
    return se.Policy.deterministic(version, entries)


POLICIES: dict[str, tuple[dict, float]] = {
    "ship_now": ({(): SHIP}, 0.0),
    "repair_now": ({(): REPAIR}, 1.0),
    "inspect_then_act": (
        {(): INSPECT, ((INSPECT, ALARM),): REPAIR, ((INSPECT, NEGATIVE),): SHIP},
        1.5,
    ),
    "inspect_twice": (
        {
            (): INSPECT,
            ((INSPECT, ALARM),): INSPECT,
            ((INSPECT, NEGATIVE),): INSPECT,
            ((INSPECT, ALARM), (INSPECT, NEGATIVE)): REPAIR,
            ((INSPECT, NEGATIVE), (INSPECT, ALARM)): REPAIR,
        },
        1.4375,
    ),
    "pilot_inspect_act": (
        {
            (): PILOT,
            ((PILOT, 0),): INSPECT,
            ((PILOT, 0), (INSPECT, ALARM)): REPAIR,
            ((PILOT, 0), (INSPECT, NEGATIVE)): SHIP,
        },
        2.25,
    ),
    "pilot_then_ship": ({(): PILOT, ((PILOT, 0),): SHIP}, 0.75),
    "pilot_then_repair": ({(): PILOT, ((PILOT, 0),): REPAIR}, 2.75),
    "wait_three": (
        {(): WAIT, ((WAIT, 0),): WAIT, ((WAIT, 0), (WAIT, 0)): WAIT},
        -1.15625,
    ),
}


# --------------------------------------------------------------------------- the oracle

ACTS = {
    WAIT: dict(
        commit=False,
        cost=0.0,
        trans=((0.75, 0.25), (0.0, 1.0)),
        imm=(0.0, 0.0),
        delayed=None,
        lik=None,
    ),
    INSPECT: dict(
        commit=False,
        cost=0.25,
        trans=IDENTITY,
        imm=(0.0, 0.0),
        delayed=None,
        lik=((0.75, 0.25), (0.25, 0.75)),
    ),
    SHIP: dict(commit=True, cost=0.0, trans=IDENTITY, imm=(10.0, -10.0), delayed=None, lik=None),
    REPAIR: dict(
        commit=True,
        cost=4.0,
        trans=((1.0, 0.0), (0.5, 0.5)),
        imm=(0.0, 0.0),
        delayed=(1, (10.0, -10.0)),
        lik=None,
    ),
    PILOT: dict(
        commit=False, cost=0.25, trans=IDENTITY, imm=(0.0, 0.0), delayed=(2, (3.0, -1.0)), lik=None
    ),
}


def oracle(
    policy: dict,
    *,
    boundaries: bool = True,
    horizon: int = 3,
    forfeit: bool = False,
    init=(0.5, 0.5),
    terminal=(2.0, -2.0),
) -> float:
    """Value of a policy table by joint-probability backward induction (independent oracle)."""
    bounds = [(0.85, None, REPAIR), (None, 0.15, SHIP)] if boundaries else []

    def walk(history, joint, pending):
        t = len(history)
        mass = joint[0] + joint[1]
        total = 0.0
        for action, due in pending:
            if due == t:
                u = ACTS[action]["delayed"][1]
                total += joint[0] * u[0] + joint[1] * u[1]
        remaining = [(a, d) for a, d in pending if d != t]
        if t == horizon:
            assert not remaining or forfeit, "an outcome is censored by the horizon"
            return total + joint[0] * terminal[0] + joint[1] * terminal[1]
        faulty = joint[1] / mass
        distribution = None
        for upper, lower, stop in bounds:
            if (upper is not None and faulty >= upper) or (lower is not None and faulty <= lower):
                distribution = {stop: 1.0}
                break
        if distribution is None:
            action = policy[history]
            distribution = {action: 1.0} if isinstance(action, int) else action
        for action, pi in distribution.items():
            act = ACTS[action]
            branch = joint[0] * act["imm"][0] + joint[1] * act["imm"][1] - mass * act["cost"]
            if act["commit"]:
                items = list(remaining)
                if act["delayed"]:
                    items.append((action, t + act["delayed"][0]))
                last = max((d for _, d in items), default=t)
                state = list(joint)
                for step in range(t + 1, last + 1):
                    nxt = [0.0, 0.0]
                    for s in range(2):
                        for s2 in range(2):
                            nxt[s2] += state[s] * act["trans"][s][s2]
                    state = nxt
                    for a, due in items:
                        if due == step:
                            u = ACTS[a]["delayed"][1]
                            branch += state[0] * u[0] + state[1] * u[1]
            else:
                carry = list(remaining)
                if act["delayed"]:
                    carry.append((action, t + act["delayed"][0]))
                symbols = 2 if act["lik"] else 1
                for symbol in range(symbols):
                    nxt = [0.0, 0.0]
                    for s in range(2):
                        for s2 in range(2):
                            lik = act["lik"][symbol][s2] if act["lik"] else 1.0
                            nxt[s2] += joint[s] * act["trans"][s][s2] * lik
                    if nxt[0] + nxt[1] > 0.0:
                        branch += walk(history + ((action, symbol),), tuple(nxt), carry)
            total += pi * branch
        return total

    return walk((), tuple(init), [])


def refused(call, code: str, detail: str):
    with pytest.raises(se.SequentialEvaluationRefusal) as caught:
        call()
    assert caught.value.reason_code == code
    assert caught.value.detail == detail
    return caught.value


# --------------------------------------------------------------------------- exact values


def test_enumerated_values_match_the_hand_solution_and_the_independent_oracle():
    problem = model()
    for name, (entries, hand) in POLICIES.items():
        value = se.evaluate(problem, det(name, entries)).value
        assert value == pytest.approx(hand, abs=TOL), name
        assert oracle(entries) == pytest.approx(hand, abs=TOL), name
    # a coin between inspect-then-act (3/2) and repair now (1)
    coin = se.Policy.stochastic(
        "coin",
        {
            (): {INSPECT: 0.5, REPAIR: 0.5},
            ((INSPECT, NEGATIVE),): {SHIP: 1.0},
            ((INSPECT, ALARM),): {REPAIR: 1.0},
        },
    )
    assert se.evaluate(problem, coin).value == pytest.approx(1.25, abs=TOL)


def test_delayed_outcomes_and_the_censoring_rule():
    # pilot x3: the outcomes due at times 2 and 3 are realized (3*0.5 - 0.5 each), the one due at
    # time 4 is past the horizon. Terminal 0, cost 3/4: 1.25 when forfeited.
    entries = {(): PILOT, ((PILOT, 0),): PILOT, ((PILOT, 0), (PILOT, 0)): PILOT}
    refused(
        lambda: se.evaluate(model(), det("pilot3", entries)),
        "invalid_argument",
        "sequential_evaluation.delayed_outcome_beyond_horizon",
    )
    forfeiting = model(censoring="forfeit")
    evaluation = se.evaluate(forfeiting, det("pilot3", entries))
    assert evaluation.value == pytest.approx(1.25, abs=TOL)
    assert oracle(entries, forfeit=True) == pytest.approx(1.25, abs=TOL)
    assert evaluation.histories[-1].kind == "horizon"
    assert evaluation.histories[-1].forfeited_outcomes == 1


def test_receipts_account_for_the_value_and_name_the_forced_stops():
    evaluation = se.evaluate(model(), det("twice_v1", POLICIES["inspect_twice"][0]))
    assert evaluation.policy_version == "twice_v1"
    assert evaluation.inference == "point_only"
    assert len(evaluation.histories) == 7 and len(evaluation.decisions) == 7
    accounted = sum(h.reach_probability * h.realized_utility for h in evaluation.histories)
    assert accounted == pytest.approx(evaluation.value, abs=TOL)
    by_history = {h.history: h for h in evaluation.histories}
    aa = by_history[((INSPECT, ALARM), (INSPECT, ALARM))]
    assert aa.reach_probability == pytest.approx(0.3125, abs=1e-15)
    assert aa.belief == pytest.approx((0.1, 0.9), abs=1e-12)
    assert (aa.kind, aa.boundary) == ("forced_stop", 0)
    assert aa.continuation_value == pytest.approx(-3.0, abs=TOL)
    nn = by_history[((INSPECT, NEGATIVE), (INSPECT, NEGATIVE))]
    assert (nn.kind, nn.boundary) == ("forced_stop", 1)
    assert nn.continuation_value == pytest.approx(8.0, abs=TOL)
    mixed = by_history[((INSPECT, ALARM), (INSPECT, NEGATIVE))]
    assert mixed.kind == "decision"
    assert mixed.reach_probability == pytest.approx(0.1875, abs=1e-15)
    assert sorted(d.forced_by for d in evaluation.decisions if d.forced_by is not None) == [0, 1]
    assert evaluation.assumptions.keys() == {"exchangeability", "consistency", "positivity"}
    # the measurement model of the channel is bound into the receipt
    assert [i for i, _ in evaluation.measurement_identities] == [INSPECT]
    assert evaluation.model_identity == model().identity


def test_relabelling_actions_in_a_policy_by_label_and_the_model_identity():
    problem = model()
    by_label = se.Policy.deterministic(
        "labelled",
        {(): "inspect", ((INSPECT, ALARM),): "repair", ((INSPECT, NEGATIVE),): "ship"},
        problem,
    )
    assert se.evaluate(problem, by_label).value == pytest.approx(1.5, abs=TOL)
    # the display label is not part of the identity; the laws are
    assert replace(problem, label="another label").identity == problem.identity
    assert replace(problem, terminal_utility=(2.0, -2.5)).identity != problem.identity


# --------------------------------------------------------------------------- refusals


def test_a_policy_that_takes_an_unsupported_action_is_refused_with_its_history():
    problem = model(support_overrides=(se.SupportOverride((), (True, True, True, False, True)),))
    error = refused(
        lambda: se.evaluate(problem, det("v", POLICIES["repair_now"][0])),
        "history_support_violated",
        "sequential_evaluation.action_not_supported",
    )
    assert "repair" in error.message or "repair" in str(error)
    assert error.remedy
    # a supported action in the same history is evaluated
    assert se.evaluate(problem, det("v", POLICIES["ship_now"][0])).value == pytest.approx(0.0)


def test_incomplete_policy_and_over_budget_policy_are_refused():
    incomplete = dict(POLICIES["inspect_twice"][0])
    del incomplete[((INSPECT, ALARM), (INSPECT, NEGATIVE))]
    refused(
        lambda: se.evaluate(model(), det("v", incomplete)),
        "invalid_argument",
        "sequential_evaluation.policy_incomplete",
    )
    greedy = dict(POLICIES["inspect_twice"][0])
    greedy[((INSPECT, ALARM), (INSPECT, NEGATIVE))] = INSPECT
    greedy[((INSPECT, NEGATIVE), (INSPECT, ALARM))] = INSPECT
    refused(
        lambda: se.evaluate(model(), det("v", greedy)),
        "invalid_argument",
        "sequential_evaluation.resource_budget_exceeded",
    )


def test_a_static_score_table_and_policy_learning_are_not_sequential_evaluation():
    table_policy = se.ReweightedStaticScores(scores=(1.0, 0.0, 1.5), weights=(0.2, 0.3, 0.5))
    refused(
        lambda: se.evaluate(model(), table_policy),
        "route_not_supported",
        "sequential_evaluation.static_score_table",
    )
    learned = se.LearnedFromEvaluationData(rule="fitted_q_iteration")
    refused(
        lambda: se.resolve_policy(learned),
        "route_not_supported",
        "sequential_evaluation.policy_change_refused",
    )
    refused(
        lambda: se.compare(model(), [se.Comparison("c", "act_now", learned)]),
        "route_not_supported",
        "sequential_evaluation.policy_change_refused",
    )
    fixed = det("v", POLICIES["ship_now"][0])
    assert se.resolve_policy(fixed) == fixed


def test_another_population_or_period_and_any_interval_are_typed_refusals():
    problem = model()
    se.check_request(problem, target_population="plant_a", target_period="2026")
    error = refused(
        lambda: se.check_request(problem, target_population="plant_b"),
        "transport_missing_evidence",
        "sequential_evaluation.transported_policy_contract_missing",
    )
    assert "selection diagram" in error.remedy
    refused(
        lambda: se.check_request(problem, target_period="2027"),
        "transport_missing_evidence",
        "sequential_evaluation.temporal_transport_contract_missing",
    )
    refused(
        lambda: se.compare(
            problem,
            [se.Comparison("now", "act_now", det("v", POLICIES["ship_now"][0]))],
            target_population="plant_b",
        ),
        "transport_missing_evidence",
        "sequential_evaluation.transported_policy_contract_missing",
    )
    refused(se.request_interval, "cell_not_licensed", "sequential_evaluation.interval_not_licensed")


def test_observation_channels_pass_through_the_measurement_model_checks():
    inspect_only = det("v", POLICIES["inspect_twice"][0])

    def with_channel(channel):
        return model(actions=actions(channel))

    wrong_construct = alarm_panel(
        construct=mm.MeasurementVariable("something_else", "latent_scientific", "category")
    )
    refused(
        lambda: se.evaluate(with_channel(wrong_construct), inspect_only),
        "measurement_binding_mismatch",
        "sequential_evaluation.channel_construct_mismatch",
    )
    refused(
        lambda: se.evaluate(with_channel(alarm_panel(population_id="plant_b")), inspect_only),
        "measurement_binding_mismatch",
        "sequential_evaluation.channel_binding_mismatch",
    )
    proxy = alarm_panel(
        indicator=mm.MeasurementVariable("unit_condition", "noisy_misclassified", "category")
    )
    assert (
        pytest.raises(
            se.SequentialEvaluationRefusal, se.evaluate, with_channel(proxy), inspect_only
        ).value.reason_code
        == "proxy_offered_as_construct"
    )
    differential = alarm_panel(structure="differential_population")
    error = pytest.raises(
        se.SequentialEvaluationRefusal, se.evaluate, with_channel(differential), inspect_only
    ).value
    assert error.reason_code == "differential_error_not_pooled"
    assert error.detail == "measurement_model.structure_parameters"


def test_malformed_declarations_raise_before_any_native_call():
    with pytest.raises(CausalValueError):
        se.Action("x", "teleport", transition=IDENTITY)
    with pytest.raises(CausalTypeError):
        se.Action("x", "commit", transition=IDENTITY, channel="not a model")
    with pytest.raises(CausalValueError):
        model(censoring="never")
    with pytest.raises(CausalValueError):
        se.stream_uniform(1, 0, 0, "not_a_role")


# --------------------------------------------------------------------------- belief session


def test_a_belief_session_updates_by_bayes_and_records_each_observation_once():
    session = se.BeliefSession(model(), "unit_17")
    assert session.belief == pytest.approx((0.5, 0.5))
    receipt = session.observe(INSPECT, ALARM, "inspection_1")
    # P(alarm) = 0.5*0.25 + 0.5*0.75; P(faulty | alarm) = 0.375 / 0.5
    assert receipt.probability == pytest.approx(0.5, abs=1e-15)
    assert receipt.belief == pytest.approx((0.25, 0.75), abs=1e-15)
    assert receipt.measurement_identity is not None
    assert receipt.parameter_uncertainty_propagated is False
    assert session.history == ((INSPECT, ALARM),)
    state, recorded = session.ledger_state, session.recorded
    refused(
        lambda: session.observe(INSPECT, ALARM, "inspection_1"),
        "evidence_double_counted",
        "sequential_evaluation.record_already_counted",
    )
    assert (session.ledger_state, session.recorded, session.time) == (state, recorded, 1)
    assert session.observe(INSPECT, NEGATIVE, "inspection_2").belief == pytest.approx(
        (0.5, 0.5), abs=1e-15
    )
    refused(
        lambda: session.observe(INSPECT, ALARM, "inspection_3"),
        "invalid_argument",
        "sequential_evaluation.resource_budget_exceeded",
    )
    refused(
        lambda: session.observe(SHIP, 0, "ship_1"),
        "invalid_argument",
        "sequential_evaluation.commit_has_no_observation",
    )


# --------------------------------------------------------------------------- comparisons


def test_comparisons_and_the_value_of_collecting_evidence_through_the_design_ranker():
    problem = model(boundaries=False)
    comparisons = [
        se.Comparison("repair_now", "act_now", det("v_now", {(): REPAIR})),
        se.Comparison("wait_then_repair", "wait", det("v_wait", {(): WAIT, ((WAIT, 0),): REPAIR})),
        se.Comparison(
            "inspect_then_repair",
            "collect_evidence",
            det(
                "v_collect",
                {(): INSPECT, ((INSPECT, ALARM),): REPAIR, ((INSPECT, NEGATIVE),): REPAIR},
            ),
        ),
        se.Comparison(
            "inspect_then_choose",
            "choose_after_observation",
            det("v_choose", POLICIES["inspect_then_act"][0]),
        ),
    ]
    report = se.compare(problem, comparisons, collect_action=INSPECT, commit_actions=(SHIP, REPAIR))
    values = [row.value for row in report.rows]
    # independent backward induction on the same four policies
    expected = [
        oracle({(): REPAIR}, boundaries=False),
        oracle({(): WAIT, ((WAIT, 0),): REPAIR}, boundaries=False),
        oracle(
            {(): INSPECT, ((INSPECT, ALARM),): REPAIR, ((INSPECT, NEGATIVE),): REPAIR},
            boundaries=False,
        ),
        oracle(POLICIES["inspect_then_act"][0], boundaries=False),
    ]
    assert values == pytest.approx([1.0, -0.25, 0.75, 1.5], abs=TOL)
    assert values == pytest.approx(expected, abs=TOL)
    assert report.best == 3
    assert report.rows[3].difference_from_first == pytest.approx(0.5, abs=TOL)
    assert report.inference == "point_only"

    voi = report.value_of_information
    assert voi is not None
    # prior best commit value 1; posterior 0.5*(-3/2) + 0.5*5 = 7/4; EVSI 3/4; cost 1/4
    assert voi.prior_value == pytest.approx(1.0, abs=TOL)
    assert voi.expected_value_of_sample_information == pytest.approx(0.75, abs=TOL)
    assert voi.net_value_of_collecting == pytest.approx(0.5, abs=TOL)
    assert voi.enumerated_policy_value == pytest.approx(1.5, abs=TOL)
    assert voi.ranking_evaluation == "exact"
    assert voi.bayes_after_observation.decisions  # the policy whose value was enumerated
    direct = se.value_of_collecting_evidence(problem, INSPECT, (SHIP, REPAIR))
    assert direct.expected_value_of_sample_information == voi.expected_value_of_sample_information

    # a comparison must implement its declared kind
    refused(
        lambda: se.compare(
            problem, [se.Comparison("x", "act_now", det("v", POLICIES["inspect_then_act"][0]))]
        ),
        "invalid_argument",
        "sequential_evaluation.comparison_kind_mismatch",
    )
    # the cross-check is not offered for a model with a stopping boundary
    refused(
        lambda: se.value_of_collecting_evidence(model(), INSPECT, (SHIP, REPAIR)),
        "route_not_supported",
        "sequential_evaluation.voi_scope",
    )


# --------------------------------------------------------------------------- rollouts


def test_rollouts_on_common_random_streams_agree_with_the_enumeration():
    problem = model()
    for name in ("inspect_twice", "pilot_inspect_act"):
        entries, exact = POLICIES[name]
        policy = det(name, entries)
        estimate = se.rollout(problem, policy, seed=20_260_101, rollouts=5_000)
        assert estimate.monte_carlo_standard_error > 0.0
        assert abs(estimate.mean - exact) <= 5.0 * estimate.monte_carlo_standard_error
        assert estimate.error_label == "uncalibrated_diagnostic"
        assert estimate.inference == "point_only"
        again = se.rollout(problem, policy, seed=20_260_101, rollouts=5_000)
        assert again.mean == estimate.mean
    # a policy differing only at a history it never reaches draws bit-identical streams
    base = det("a", POLICIES["inspect_twice"][0])
    extended_entries = dict(POLICIES["inspect_twice"][0])
    extended_entries[((WAIT, 0), (WAIT, 0), (WAIT, 0))] = SHIP
    extended = det("b", extended_entries)
    first = se.rollout(problem, base, seed=5, rollouts=500)
    second = se.rollout(problem, extended, seed=5, rollouts=500)
    assert first.mean == second.mean
    assert first.monte_carlo_standard_error == second.monte_carlo_standard_error


def test_a_stream_uniform_is_a_pure_function_of_its_semantic_identity():
    u = se.stream_uniform(7, 3, 2, "transition")
    assert u == se.stream_uniform(7, 3, 2, "transition")
    assert 0.0 <= u < 1.0
    others = {
        se.stream_uniform(7, 3, 2, "observation"),
        se.stream_uniform(7, 4, 2, "transition"),
        se.stream_uniform(7, 3, 3, "transition"),
        se.stream_uniform(8, 3, 2, "transition"),
    }
    assert u not in others and len(others) == 4


def test_cancellation_and_bounds_are_observed():
    token = _native.CancellationToken()
    token.cancel()
    problem = model()
    policy = det("v", POLICIES["ship_now"][0])
    assert (
        pytest.raises(
            se.SequentialEvaluationRefusal,
            se.rollout,
            problem,
            policy,
            seed=1,
            rollouts=10,
            execution=Execution(cancel=token),
        ).value.reason_code
        == "cancelled_no_claim"
    )
    refused(
        lambda: se.rollout(problem, policy, seed=1, rollouts=0),
        "invalid_argument",
        "sequential_evaluation.rollouts",
    )
    refused(
        lambda: se.evaluate(problem, det("v", POLICIES["inspect_twice"][0]), max_nodes=3),
        "invalid_argument",
        "sequential_evaluation.limit_exceeded",
    )


# --------------------------------------------------------------------------- off-policy

LOG_SIZE = 256


def mu(history, inspections):
    if inspections >= 1:
        return {SHIP: 0.5, REPAIR: 0.5}
    return {INSPECT: 0.5, SHIP: 0.25, REPAIR: 0.25}


def small_problem():
    """Horizon two; ship +10/-10, repair +6/-4 with no delay; inspection costs 1/4."""
    return model(
        boundaries=False,
        horizon=2,
        resource_budget=None,
        terminal_utility=(0.0, 0.0),
        actions=(
            se.Action("wait", "continue", transition=((0.75, 0.25), (0.0, 1.0))),
            se.Action(
                "inspect", "continue", transition=IDENTITY, utility_cost=0.25, channel=alarm_panel()
            ),
            se.Action("ship", "commit", transition=IDENTITY, immediate_utility=(10.0, -10.0)),
            se.Action("repair", "commit", transition=IDENTITY, immediate_utility=(6.0, -4.0)),
        ),
    )


IMMEDIATE = {SHIP: (10.0, -10.0), REPAIR: (6.0, -4.0)}


def enumerated_log():
    """Every trajectory under the logging policy, repeated in proportion to its probability."""
    groups: dict[tuple, float] = {}
    lik = ((0.75, 0.25), (0.25, 0.75))
    for state in range(2):
        for first, p1 in mu((), 0).items():
            if first != INSPECT:
                key = (((first, None, p1),), IMMEDIATE[first][state])
                groups[key] = groups.get(key, 0.0) + 0.5 * p1
                continue
            for symbol in range(2):
                p_obs = lik[symbol][state]
                for second, p2 in mu(((INSPECT, symbol),), 1).items():
                    steps = ((INSPECT, symbol, p1), (second, None, p2))
                    key = (steps, IMMEDIATE[second][state])
                    groups[key] = groups.get(key, 0.0) + 0.5 * p1 * p_obs * p2
    log = []
    for index, ((steps, outcome), probability) in enumerate(sorted(groups.items())):
        count = probability * LOG_SIZE
        assert abs(count - round(count)) < 1e-9
        for copy in range(round(count)):
            log.append(
                se.LoggedTrajectory(
                    id=f"t{index:03d}-{copy:03d}",
                    steps=tuple(se.LoggedStep(a, o, p) for a, o, p in steps),
                    outcome_utility=outcome,
                )
            )
    return log


def logging_policy() -> se.Policy:
    entries = {(): mu((), 0)}
    for symbol in range(2):
        entries[((INSPECT, symbol),)] = mu(((INSPECT, symbol),), 1)
    return se.Policy.stochastic("logging_v1", entries)


def test_off_policy_weighting_recovers_the_enumerated_value():
    problem = small_problem()
    log = enumerated_log()
    assert len(log) == LOG_SIZE
    target = det("inspect_then_act_v1", POLICIES["inspect_then_act"][0] | {})
    estimate = se.evaluate_off_policy(problem, target, logging_policy(), log)
    # the log is the whole population, so the weighted mean is the g-formula value 3/2
    assert estimate.ordinary == pytest.approx(1.5, abs=1e-9)
    assert estimate.self_normalized == pytest.approx(1.5, abs=1e-9)
    assert estimate.ordinary == pytest.approx(
        oracle(
            {(): INSPECT, ((INSPECT, ALARM),): REPAIR, ((INSPECT, NEGATIVE),): SHIP},
            boundaries=False,
            horizon=2,
            terminal=(0.0, 0.0),
        ),
        abs=1e-9,
    )
    # weight 4 on a quarter of the trajectories: ESS = N / 4
    assert estimate.max_weight == pytest.approx(4.0)
    assert estimate.effective_sample_size == pytest.approx(LOG_SIZE / 4, abs=1e-6)
    assert estimate.inference == "point_only"
    assert estimate.uncalibrated_standard_error is not None
    assert estimate.trajectories == LOG_SIZE
    # permutation invariance: the same log in another order is the same estimate
    shuffled = se.evaluate_off_policy(problem, target, logging_policy(), list(reversed(log)))
    assert shuffled.ordinary == estimate.ordinary
    assert shuffled.ledger_state == estimate.ledger_state


def test_off_policy_refusals_positivity_duplicates_and_inconsistent_logs():
    problem = small_problem()
    log = enumerated_log()
    target = det("inspect_then_act_v1", POLICIES["inspect_then_act"][0])
    behaviour = logging_policy()
    # positivity: the target waits, which the logging policy never does
    waits = det("wait_first", {(): WAIT, ((WAIT, 0),): SHIP})
    refused(
        lambda: se.evaluate_off_policy(problem, waits, behaviour, log),
        "history_support_violated",
        "sequential_evaluation.behaviour_support_missing",
    )
    refused(
        lambda: se.evaluate_off_policy(problem, target, behaviour, [*log, log[0]]),
        "evidence_double_counted",
        "sequential_evaluation.trajectory_recorded_twice",
    )
    wrong = [
        replace(log[0], steps=(replace(log[0].steps[0], probability=0.125), *log[0].steps[1:])),
        *log[1:],
    ]
    refused(
        lambda: se.evaluate_off_policy(problem, target, behaviour, wrong),
        "invalid_argument",
        "sequential_evaluation.logged_trajectory_inconsistent",
    )
    ship_first = [t for t in log if t.steps[0].action == SHIP][:5]
    refused(
        lambda: se.evaluate_off_policy(problem, target, behaviour, ship_first),
        "history_support_violated",
        "sequential_evaluation.no_overlap",
    )


# --------------------------------------------------------------------------- the artifact


def sealed():
    problem = small_problem()
    log = [t for t in enumerated_log() if t.steps[0].action == INSPECT][::7]
    return problem, require_success(
        se.seal(
            problem,
            det("inspect_then_act_v1", POLICIES["inspect_then_act"][0]),
            behaviour=logging_policy(),
            trajectories=log,
            rollouts=(9, 300),
        )
    )


def test_the_artifact_replays_bit_for_bit_and_carries_receipts():
    _problem, artifact = sealed()
    assert artifact.value == pytest.approx(1.5, abs=TOL)
    data = bytes(artifact.export("sequential-evaluation-1"))
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
    assert consumed.summary["policy_version"] == "inspect_then_act_v1"
    assert consumed.summary["rollout"]["rollouts"] == 300
    assert consumed.summary["off_policy"]["records_counted"] > 0
    # the consumer can rebuild the model and policy it replayed
    assert consumed.policy.version == "inspect_then_act_v1"
    assert se.evaluate(consumed.model, consumed.policy).value == pytest.approx(1.5, abs=TOL)
    # a policy-only artifact carries neither estimate
    plain = require_success(se.seal(small_problem(), det("ship_v1", {(): SHIP})))
    assert plain.summary["off_policy"] is None and plain.summary["rollout"] is None


def test_a_resealed_other_evaluation_corruption_and_wrong_kinds_are_refused():
    problem, artifact = sealed()
    data = bytes(artifact.export("one"))
    other = require_success(se.seal(problem, det("ship_v1", {(): SHIP})))
    forged = bytes(other.export("two"))
    # internally consistent, but not the evaluation the consumer retained
    require_success(se.consume_artifact(forged))
    for wanted in (
        {"policy": artifact.identities["policy"]},
        {"evaluation": artifact.identities["evaluation"]},
        {"model": "0" * 64},
    ):
        refused(
            lambda wanted=wanted: require_success(se.consume_artifact(forged, **wanted)),
            "route_not_supported",
            "sequential_evaluation.identity_mismatch",
        )
    # a different law is a different model identity
    changed = require_success(
        se.seal(replace(problem, terminal_utility=(0.0, 1.0)), det("ship_v1", {(): SHIP}))
    )
    assert changed.identities["model"] != other.identities["model"]
    # damaged and truncated bytes, and bytes that are not an artifact
    damaged = bytearray(data)
    damaged[len(damaged) // 2] ^= 0x55
    for broken in (bytes(damaged), data[:30]):
        refused(
            lambda broken=broken: require_success(se.consume_artifact(broken)),
            "invalid_argument",
            "sequential_evaluation.malformed",
        )
    with pytest.raises(CausalTypeError):
        require_success(se.consume_artifact("not bytes"))  # type: ignore[arg-type]
    # logged trajectories need the logging policy that generated them
    with pytest.raises(CausalValueError):
        require_success(se.seal(problem, det("v", {(): SHIP}), trajectories=enumerated_log()[:1]))


def test_a_fresh_process_consumes_the_artifact_against_the_retained_identity(tmp_path):
    _problem, artifact = sealed()
    path = tmp_path / "artifact.bin"
    path.write_bytes(bytes(artifact.export("fresh")))
    script = (
        "import json, sys\n"
        "from antecedent import sequential_evaluation as se\n"
        "c = se.consume_artifact(open(sys.argv[1], 'rb').read(), evaluation=sys.argv[2])\n"
        "print(json.dumps({'value': c.value, 'identities': c.identities, "
        "'histories': len(c.histories)}))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script, str(path), artifact.identities["evaluation"]],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=True,
    )
    wire = json.loads(done.stdout)
    assert wire["value"] == pytest.approx(1.5, abs=TOL)
    assert wire["identities"] == artifact.identities
    assert wire["histories"] == 3
    refusal = subprocess.run(
        [sys.executable, "-c", script, str(path), "0" * 64],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert refusal.returncode != 0
    assert "identity_mismatch" in refusal.stderr


# --------------------------------------------------------------------------- declared bounds


def test_declared_bounds_are_refused_not_truncated():
    commit = se.Action("ship", "commit", transition=IDENTITY, immediate_utility=(1.0, 1.0))
    base = dict(
        population="plant_a",
        regime="operations",
        period="2026",
        state_variable="unit_condition",
        states=("sound", "faulty"),
        initial=(0.5, 0.5),
        horizon=1,
        actions=(commit,),
        terminal_utility=(0.0, 0.0),
    )
    ship_now = det("v", {(): 0})
    # the latent state space needs 2 to 4096 distinct named states; a state space beyond the
    # upper bound cannot be built here without a 4097 x 4097 transition table, so the declared
    # lower bound and the distinct-name rule stand for the structural state checks
    for bad_states, initial, terminal in (
        (("only",), (1.0,), (0.0,)),
        (("twin", "twin"), (0.5, 0.5), (0.0, 0.0)),
    ):
        flat = tuple(
            tuple(1.0 if i == j else 0.0 for j in range(len(bad_states)))
            for i in range(len(bad_states))
        )
        refused(
            lambda bad_states=bad_states, initial=initial, terminal=terminal, flat=flat: (
                se.evaluate(
                    se.FiniteModel(
                        **{
                            **base,
                            "states": bad_states,
                            "initial": initial,
                            "actions": (se.Action("ship", "commit", transition=flat),),
                            "terminal_utility": terminal,
                        }
                    ),
                    ship_now,
                )
            ),
            "invalid_argument",
            "sequential_evaluation.model_states",
        )
    # more than 1024 actions
    crowd = se.FiniteModel(
        **{
            **base,
            "actions": tuple(
                se.Action(f"a{i}", "commit", transition=IDENTITY, immediate_utility=(0.0, 0.0))
                for i in range(1025)
            ),
        }
    )
    refused(
        lambda: se.evaluate(crowd, ship_now),
        "invalid_argument",
        "sequential_evaluation.model_actions",
    )
    # a horizon beyond 1024 decision times, and a horizon of zero
    for horizon in (1025, 0):
        refused(
            lambda horizon=horizon: se.evaluate(
                se.FiniteModel(**{**base, "horizon": horizon}), ship_now
            ),
            "invalid_argument",
            "sequential_evaluation.model_horizon",
        )
    one = se.FiniteModel(**base)
    # more than 65536 policy entries
    big = se.Policy.deterministic("big", {((0, i),): 0 for i in range(65_537)})
    refused(
        lambda: se.evaluate(one, big), "invalid_argument", "sequential_evaluation.limit_exceeded"
    )
    # more than 10 million rollouts
    refused(
        lambda: se.rollout(one, ship_now, seed=1, rollouts=10_000_001),
        "invalid_argument",
        "sequential_evaluation.limit_exceeded",
    )
    # more than 150000 logged trajectories
    log = [
        se.LoggedTrajectory(f"t{i}", (se.LoggedStep(0, None, 1.0),), 0.0) for i in range(150_001)
    ]
    refused(
        lambda: se.evaluate_off_policy(one, ship_now, ship_now, log),
        "invalid_argument",
        "sequential_evaluation.limit_exceeded",
    )
