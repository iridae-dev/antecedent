"""Typed constructors of a transported response through the Python surface.

Each route runs in Rust and the response is built from the route's own result, so the receipt's
populations, identities and assumptions are copied, never typed. The oracles are the hand values
of ``crates/antecedent-design/tests/{transported_evaluation,transported_synthesis,
transported_bridge}.rs`` and ``crates/antecedent/tests/transported_constructors_facade.rs``.

* Inspection policy: states ``sound``/``faulty``, ``inspect`` raises an alarm with sensitivity
  and specificity 3/4, the policy ships on no alarm and holds on an alarm, ``ship`` pays +10 or
  -10, horizon two. The value is ``7.5 - 10 p`` with ``p = P(faulty)``: ``-0.5`` at ``p = 0.8``,
  ``1.5`` at ``0.6``, ``2.5`` at the source's ``0.5``.
* Adaptive trial (eight units, strata ``unit % 2``): target law ``(0.25, 0.75)`` gives treatment
  mean ``29/56``, control mean ``1/2``, contrast ``1/56``.
* Synthesis: native ``y = 1, v = 1`` and transported ``y = 2, v = 1`` with transport variance
  ``0.5`` pool to ``1.4`` with variance ``0.6``.
* Surrogate bridge: ``0.06 + 0.92 x`` at ``x = 0.8`` is ``0.796``.
"""

from __future__ import annotations

import pytest
from antecedent import adaptive_inference as ai
from antecedent import evidence_ledger as el
from antecedent import measurement_models as mm
from antecedent import scientific_bridge as sb
from antecedent import sequential_evaluation as se
from antecedent import transported_constructors as tc
from antecedent.joint_distribution import ScientificQuantity
from antecedent.transported_response import TransportedResponse, TransportedResponseRefusal

from _refusal import assert_registered_refusal

TOL = 1e-12
INSPECT, SHIP, HOLD = 0, 1, 2
IDENTITY = ((1.0, 0.0), (0.0, 1.0))
NOT_IN_TARGET = "transported_decision.coordinate_not_in_target"
NOT_TRANSPORTED = "transported_decision.not_transported"


def quantity(variable: str, population: str) -> ScientificQuantity:
    return ScientificQuantity(
        variable_id=variable,
        variable_name=variable,
        role="outcome",
        units="units",
        population_id=population,
        regime_id="do(treatment=1)",
        horizon=0,
        functional_id="average_effect",
    )


def is_digest(value: str) -> bool:
    return len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def refused(call) -> TransportedResponseRefusal:
    with pytest.raises(TransportedResponseRefusal) as caught:
        call()
    assert_registered_refusal(caught.value)
    assert caught.value.remedy or caught.value.detail == "transported_decision.invalid_value"
    return caught.value


# --------------------------------------------------------------------------- the inspection problem


def alarm_panel() -> mm.MeasurementModel:
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


def model() -> se.FiniteModel:
    return se.FiniteModel(
        population="plant_a",
        regime="operations",
        period="2026",
        state_variable="unit_condition",
        states=("sound", "faulty"),
        initial=(0.5, 0.5),
        horizon=2,
        actions=(
            se.Action("inspect", "continue", transition=IDENTITY, channel=alarm_panel()),
            se.Action("ship", "commit", transition=IDENTITY, immediate_utility=(10.0, -10.0)),
            se.Action("hold", "commit", transition=IDENTITY, immediate_utility=(0.0, 0.0)),
        ),
        terminal_utility=(0.0, 0.0),
    )


def policy(m: se.FiniteModel) -> se.Policy:
    return se.Policy.deterministic(
        "inspect_then_ship_on_quiet",
        {(): "inspect", ((INSPECT, 0),): "ship", ((INSPECT, 1),): "hold"},
        m,
    )


MECHANISMS = {
    "initial_state": "shifted",
    "utility": "invariant",
    "transition[0]": "invariant",
    "transition[1]": "invariant",
    "transition[2]": "invariant",
    "observation[0]": "invariant",
}


def diagram(**overrides) -> tc.SelectionDiagram:
    fields = dict(
        source_population="plant_a", target_population="plant_b", mechanisms=dict(MECHANISMS)
    )
    fields.update(overrides)
    return tc.SelectionDiagram(**fields)


# --------------------------------------------------------------------------- policy


def test_a_policy_value_is_carried_with_the_populations_and_digest_of_its_evaluation():
    m = model()
    built = tc.from_policy(
        m,
        policy(m),
        target_population="plant_b",
        target_initial=(0.2, 0.8),
        diagram=diagram(),
        coordinate=quantity("value", "plant_b"),
        provider_id="plant-transport",
    )
    response = built.response
    assert isinstance(response, TransportedResponse)
    assert response.means == pytest.approx((-0.5,), abs=TOL)
    assert built.evaluation["target_value"] == pytest.approx(-0.5, abs=TOL)
    assert built.evaluation["source_value"] == pytest.approx(2.5, abs=TOL)
    assert built.evaluation["inference"] == "point_only"
    assert not response.native_target_data

    receipt = response.receipt
    assert receipt.route == "policy"
    assert receipt.source_populations == ("plant_a",)
    assert receipt.target_population == "plant_b"
    assert receipt.support == "extrapolative"
    assert len(receipt.transport_identities) == 1 and is_digest(receipt.transport_identities[0])
    assert receipt.source_identity == built.evaluation["identity"]
    assert "shifted:initial_state" in receipt.assumptions

    # Same shape, changed target law: another value and another identity.
    other = tc.from_policy(
        m,
        policy(m),
        target_population="plant_b",
        target_initial=(0.3, 0.7),
        diagram=diagram(),
        coordinate=quantity("value", "plant_b"),
        provider_id="plant-transport",
    )
    assert other.response.means == pytest.approx((0.5,), abs=TOL)
    assert other.response.identity != response.identity


def test_a_policy_route_refusal_is_the_routes_own_and_a_source_coordinate_is_refused():
    m = model()

    def build(**overrides):
        fields = dict(
            target_population="plant_b",
            target_initial=(0.2, 0.8),
            diagram=diagram(),
            coordinate=quantity("value", "plant_b"),
            provider_id="plant-transport",
        )
        fields.update(overrides)
        return tc.from_policy(m, policy(m), **fields)

    error = refused(lambda: build(diagram=None))
    assert error.reason_code == "transport_missing_evidence"
    assert error.detail == "transported_policy.selection_diagram_missing"

    shifted = dict(MECHANISMS, **{"transition[0]": "shifted"})
    error = refused(lambda: build(diagram=diagram(mechanisms=shifted)))
    assert error.detail == "transported_policy.mechanism_not_invariant"

    partial = {k: v for k, v in MECHANISMS.items() if k != "observation[0]"}
    error = refused(lambda: build(diagram=diagram(mechanisms=partial)))
    assert error.detail == "transported_policy.mechanism_status_missing"

    error = refused(lambda: build(coordinate=quantity("value", "plant_a")))
    assert error.detail == NOT_IN_TARGET
    assert error.expected == "plant_b" and error.supplied == "plant_a"


# --------------------------------------------------------------------------- period shift


def test_a_period_shift_is_stated_in_period_qualified_populations():
    m = model()
    target = tc.TargetPeriod.of_model(m, "2027")
    assert target.horizon == 2 and target.alphabet[0][0] == "inspect"
    invariance = tc.InvarianceDeclaration("2026", "2027", 2, dict(MECHANISMS))
    in_2027 = tc.period_population_label("plant_a", "2027")
    assert in_2027 == "plant_a@2027"

    built = tc.from_period_shift(
        m,
        policy(m),
        target=target,
        target_initial=(0.4, 0.6),
        invariance=invariance,
        coordinate=quantity("value", in_2027),
        provider_id="plant-transport",
    )
    assert built.response.means == pytest.approx((1.5,), abs=TOL)
    assert built.evaluation["source_value"] == pytest.approx(2.5, abs=TOL)
    receipt = built.response.receipt
    assert receipt.source_populations == ("plant_a@2026",)
    assert receipt.target_population == "plant_a@2027"
    assert len(receipt.transport_identities) == 1 and is_digest(receipt.transport_identities[0])
    assert receipt.support == "extrapolative"
    assert "fixed_horizon:2" in receipt.assumptions
    assert receipt.route == "temporal_policy"
    assert "initial_state:shifted" in receipt.assumptions

    # The unqualified population is not the target period.
    error = refused(
        lambda: tc.from_period_shift(
            m,
            policy(m),
            target=target,
            target_initial=(0.4, 0.6),
            invariance=invariance,
            coordinate=quantity("value", "plant_a"),
            provider_id="plant-transport",
        )
    )
    assert error.detail == NOT_IN_TARGET


def test_a_time_varying_mechanism_and_a_missing_declaration_are_refused():
    m = model()
    target = tc.TargetPeriod.of_model(m, "2027")
    coordinate = quantity("value", "plant_a@2027")

    def build(invariance):
        return tc.from_period_shift(
            m,
            policy(m),
            target=target,
            target_initial=(0.4, 0.6),
            invariance=invariance,
            coordinate=coordinate,
            provider_id="plant-transport",
        )

    varying = tc.InvarianceDeclaration(
        "2026", "2027", 2, dict(MECHANISMS, **{"transition[0]": "shifted"})
    )
    error = refused(lambda: build(varying))
    assert error.reason_code == "transport_missing_evidence"
    assert error.detail == "temporal_policy_transport.mechanism_time_varying"

    error = refused(lambda: build(None))
    assert error.detail == "temporal_policy_transport.time_invariance_missing"


# --------------------------------------------------------------------------- adaptive trial


def trial_design(**changes) -> ai.AdaptiveDesign:
    outcomes = [[y] for y in (1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0)]
    arms = [0, 1, 0, 1, 1, 0, 1, 0]
    probabilities = [
        [0.2, 0.8] if u in (4, 5) else [0.4, 0.6] if u in (6, 7) else [0.5, 0.5] for u in range(8)
    ]
    fields = dict(
        label="trial",
        assignment="epsilon_greedy",
        outcome="single_outcome",
        monitoring="fixed_interim_looks",
        stopping="efficacy",
        arm_labels=("control", "treatment"),
        assigned_arm=arms,
        outcomes=outcomes,
        policy_history=[
            ai.PolicyEpoch(0, ai.policy_identity("policy/0")),
            ai.PolicyEpoch(4, ai.policy_identity("policy/1")),
        ],
        assignment_probabilities=probabilities,
        exposure=arms,
        availability=ai.Availability([0, 0, 0, 0, 1, 1, 1, 1], "immediate"),
        interim_analyses=[ai.InterimAnalysis(4, 0.5), ai.InterimAnalysis(8, 1.0)],
        stopping_rule=ai.StoppingRule.posterior_probability_best(0.95),
        probability_floor=0.05,
    )
    fields.update(changes)
    return ai.AdaptiveDesign(**fields)


def trial_diagram(**overrides) -> tc.AdaptiveTrialDiagram:
    fields = dict(
        source_population="trial_sites",
        target_population="clinic_network",
        mechanisms={"stratum_law": "shifted", "outcome_given_stratum_and_arm": "invariant"},
    )
    fields.update(overrides)
    return tc.AdaptiveTrialDiagram(**fields)


def build_trial(value="contrast", **overrides):
    fields = dict(
        trial_population="trial_sites",
        unit_strata=[u % 2 for u in range(8)],
        stratum_labels=["mild", "severe"],
        target_stratum_law=(0.25, 0.75),
        diagram=trial_diagram(),
        coordinate=quantity(value, "clinic_network"),
        provider_id="trial-transport",
        value=value,
    )
    fields.update(overrides)
    design = fields.pop("design", trial_design())
    return tc.from_adaptive_trial(design, **fields)


def test_an_adaptive_trial_mean_and_contrast_carry_the_trial_and_diagram_identities():
    treatment = build_trial("treatment_mean")
    control = build_trial("control_mean")
    contrast = build_trial("contrast")
    assert treatment.response.means == pytest.approx((29 / 56,), abs=TOL)
    assert control.response.means == pytest.approx((0.5,), abs=TOL)
    assert contrast.response.means == pytest.approx((1 / 56,), abs=TOL)
    assert treatment.evaluation["analysed_units"] == 8

    receipt = treatment.response.receipt
    assert receipt.source_populations == ("trial_sites",)
    assert receipt.target_population == "clinic_network"
    assert len(receipt.transport_identities) == 1 and is_digest(receipt.transport_identities[0])
    assert receipt.transport_identities[0] == treatment.evaluation["diagram_identity"]
    assert "stratum_law:shifted" in receipt.assumptions
    assert "outcome_given_stratum_and_arm:invariant" in receipt.assumptions
    assert receipt.support == "extrapolative"

    # A changed target law at unchanged shape is another estimate and another identity.
    mild = build_trial("treatment_mean", target_stratum_law=(1.0, 0.0))
    assert mild.response.means == pytest.approx((4 / 7,), abs=TOL)
    assert mild.response.identity != treatment.response.identity


def test_adaptive_refusals_are_the_routes_and_a_source_coordinate_is_refused():
    error = refused(lambda: build_trial(design=None))
    assert error.reason_code == "adaptive_contract_incomplete"
    assert error.detail == "adaptive_transport.assignment_monitoring_contract_missing"

    shifted = trial_diagram(
        mechanisms={
            "stratum_law": "shifted",
            "outcome_given_stratum_and_arm": "shifted",
        }
    )
    error = refused(lambda: build_trial(diagram=shifted))
    assert error.detail == "adaptive_transport.outcome_mechanism_not_invariant"

    error = refused(lambda: build_trial(target_stratum_law=None))
    assert error.detail == "adaptive_transport.target_stratum_law_missing"

    error = refused(lambda: build_trial(coordinate=quantity("contrast", "trial_sites")))
    assert error.detail == NOT_IN_TARGET


# --------------------------------------------------------------------------- synthesis

SELECTION = "01" * 32


def synthesis_ledger():
    ledger = el.EvidenceLedger()
    cohort = ledger.add_source(
        el.SourceDeclaration(
            "registry cohort", "observation_set", observations=el.rows("registry", range(0, 100))
        )
    )
    carried = ledger.add_source(
        el.SourceDeclaration(
            "trial a claim",
            "external_claim",
            observations=el.rows("registry", range(1000, 1100)),
            transport=el.TransportIdentity(
                source_populations=("trial_a",),
                target_population="registry",
                selection_diagram=SELECTION,
                assumptions=("outcome mechanism invariant",),
            ),
        )
    )
    return ledger, cohort, carried


def test_a_transported_synthesis_takes_its_populations_from_the_ledger_identities():
    ledger, cohort, carried = synthesis_ledger()
    before = ledger.state_digest
    built = tc.from_synthesis(
        ledger,
        "pooled effect",
        [
            tc.Contribution(cohort, 1.0, 1.0),
            tc.Contribution(carried, 2.0, 1.0, transport_variance=0.5),
        ],
        tc.Pool(),
        coordinate=quantity("pooled", "registry"),
        provider_id="synthesis-provider",
        native_population="registry",
    )
    assert built.response.means == pytest.approx((1.4,), abs=TOL)
    assert built.evaluation["variance"] == pytest.approx(0.6, abs=TOL)
    assert built.evaluation["transport_variance_added"] == pytest.approx(0.1, abs=TOL)
    assert built.evaluation["claim_class"] == "point_only"
    assert built.evaluation["distinct_observations"] == 200

    receipt = built.response.receipt
    assert receipt.route == "synthesis"
    assert receipt.source_populations == ("trial_a",)
    assert receipt.target_population == "registry"
    assert is_digest(receipt.transport_identities[0])
    assert receipt.source_identity == built.evaluation["result"]
    assert receipt.uncalibrated_standard_error == pytest.approx(0.6**0.5, abs=TOL)
    # The session's ledger is not changed by the constructor.
    assert ledger.state_digest == before


def test_synthesis_refusals_are_the_routes_and_carry_no_interval():
    ledger, cohort, carried = synthesis_ledger()

    def build(contributions, **overrides):
        fields = dict(
            coordinate=quantity("pooled", "registry"),
            provider_id="synthesis-provider",
        )
        fields.update(overrides)
        return tc.from_synthesis(ledger, "pooled", contributions, tc.Pool(), **fields)

    both = [tc.Contribution(cohort, 1.0, 1.0), tc.Contribution(carried, 2.0, 1.0, 0.5)]
    error = refused(lambda: build(both, coordinate=quantity("pooled", "trial_a")))
    assert error.detail == NOT_IN_TARGET

    second = ledger.add_source(
        el.SourceDeclaration(
            "second cohort", "observation_set", observations=el.rows("registry", range(5000, 5100))
        )
    )
    error = refused(
        lambda: build([tc.Contribution(cohort, 1.0, 1.0), tc.Contribution(second, 1.5, 1.0)])
    )
    assert error.detail == NOT_TRANSPORTED

    error = refused(
        lambda: build([tc.Contribution(cohort, 1.0, 1.0), tc.Contribution(carried, 2.0, 1.0)])
    )
    assert error.reason_code == "transport_missing_evidence"
    assert error.detail == "transported_synthesis.transport_variance_undeclared"

    error = refused(lambda: build(both, native_population="clinic"))
    assert error.detail == "transported_synthesis.target_population_mismatch"


# --------------------------------------------------------------------------- surrogate bridge


def bridge(claim_population: str, selection_targets=()) -> tc.SurrogateBridge:
    return tc.SurrogateBridge(
        surrogate=ScientificQuantity(
            "trial:biomarker",
            "trial:biomarker",
            "outcome",
            "mmol_per_l",
            claim_population,
            "do(treatment=1)",
            0,
            "average_effect",
        ),
        terminal=ScientificQuantity(
            "trial:survival",
            "trial:survival",
            "outcome",
            "log_hazard",
            claim_population,
            "do(treatment=1)",
            0,
            "average_effect",
        ),
        validation_trials=((0.0, 0.1), (0.5, 0.5), (1.0, 0.9), (1.5, 1.5)),
        validated_population="adults",
        validated_regime="do(treatment=1)",
        validation_evidence="meta-analysis:ckd-2024",
        diagram=tc.BridgeDiagram("adults", claim_population, tuple(selection_targets)),
        terminal_variable="survival",
        relation_modifiers=("age",),
    )


def source_claim(b: tc.SurrogateBridge, value: float = 0.8) -> sb.SourceClaim:
    return sb.SourceClaim.declared("biomarker-effect", value, b.surrogate)


def test_a_bridged_terminal_value_is_a_transported_point_in_the_claims_population():
    b = bridge("children", ["biomarker"])
    built = tc.from_surrogate_bridge(b, source_claim(b), provider_id="bridge-provider")
    assert built.response.means == pytest.approx((0.796,), abs=TOL)
    assert built.evaluation["value"] == pytest.approx(0.796, abs=TOL)
    assert built.evaluation["transported"] is True

    receipt = built.response.receipt
    assert receipt.source_populations == ("adults",)
    assert receipt.target_population == "children"
    assert len(receipt.transport_identities) == 1 and is_digest(receipt.transport_identities[0])
    assert receipt.support == "extrapolative"
    assert "relation_modifier:age" in receipt.assumptions
    assert "premise:surrogate_terminal_relation_invariant" in receipt.assumptions

    # Another selection diagram at the same shape is another transport identity.
    other_bridge = bridge("children", ["unrelated"])
    other = tc.from_surrogate_bridge(
        other_bridge, source_claim(other_bridge), provider_id="bridge-provider"
    )
    assert other.response.receipt.transport_identities != receipt.transport_identities
    assert other.response.identity != built.response.identity


def test_bridge_refusals_are_the_bridges_and_a_native_bridge_is_not_transported():
    contradicted = bridge("children", ["survival"])
    error = refused(
        lambda: tc.from_surrogate_bridge(
            contradicted, source_claim(contradicted), provider_id="bridge-provider"
        )
    )
    assert error.reason_code == "bridge_assumption_missing"
    assert error.detail == (
        "scientific_bridge.transport_premise_contradicted_by_selection_diagram"
    )

    native = bridge("adults", ["survival"])
    error = refused(
        lambda: tc.from_surrogate_bridge(native, source_claim(native), provider_id="bridge-provider")
    )
    assert error.detail == NOT_TRANSPORTED


def test_constructed_responses_are_never_native_target_data():
    m = model()
    built = tc.from_policy(
        m,
        policy(m),
        target_population="plant_b",
        target_initial=(0.2, 0.8),
        diagram=diagram(),
        coordinate=quantity("value", "plant_b"),
        provider_id="plant-transport",
    )
    assert built.response.native_target_data is False
    assert built.response.receipt.support != "supported"
    with pytest.raises(TransportedResponseRefusal):
        built.response.interval()
