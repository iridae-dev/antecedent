//! Assumption ranges for a violated transport premise.
//!
//! Hand-derived oracles: a point value `v` with an additive shift bound `b` on a grid of
//! three points `{0, b/2, b}` has range `[v - s, v + s]` at shift `s`; a proportional
//! shift has `[v - s |v|, v + s |v|]`. The decision against a reference `r` is
//! classified by the existing sensitivity surface:
//!
//! * `v = 3`, `r = 0`, `b = 2`: the range `[3 - s, 3 + s]` stays above 0 for `s <= 2`,
//!   so acting on the transported quantity is the invariant action;
//! * `v = 1`, `r = 0`, `b = 2`: at `s = 1` the lower end equals `r` (a tie in one
//!   scenario, a lead in the other) and at `s = 2` the lower end is below it, so the
//!   best action depends on where in the range the quantity lies at shifts 1 and 2.

use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_design::sensitivity_decision::{NoRobustReason, SensitivityOutcome};
use antecedent_design::transport_sensitivity::{
    REFERENCE_ACTION_ID, TRANSPORTED_ACTION_ID, TransportPremise, TransportRoute,
    TransportSensitivityRefusal, TransportSensitivitySpec, ViolationFamily,
    transport_assumption_range,
};

fn quantity(variable: &str, population: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: variable.into(),
        role: QuantityRole::Outcome,
        units: "mmHg".into(),
        population_id: population.into(),
        regime_id: "do(a=1)".into(),
        horizon: 0,
        functional_id: "mean".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn spec(route: TransportRoute, value: f64, family: ViolationFamily) -> TransportSensitivitySpec {
    TransportSensitivitySpec {
        route,
        quantity: quantity("effect", "target"),
        point_value: value,
        family,
        grid_points: 3,
        source_population: "source".into(),
        target_population: "target".into(),
        query_binding: "derivation:1".into(),
        provider_snapshot: "snapshot:1".into(),
        causal_contract_id: "contract:1".into(),
    }
}

#[test]
fn additive_range_is_the_closed_form_band_and_labelled_an_assumption_range() {
    let range = transport_assumption_range(&spec(
        TransportRoute::CohortDecision,
        3.0,
        ViolationFamily::AdditiveShift { bound: 2.0 },
    ))
    .unwrap();
    assert_eq!(range.grid(), &[0.0, 1.0, 2.0]);
    assert_eq!(range.lower(), &[3.0, 2.0, 1.0]);
    assert_eq!(range.upper(), &[3.0, 4.0, 5.0]);
    assert_eq!(range.range_at_bound(), (1.0, 5.0));
    assert_eq!(range.premise(), TransportPremise::SelectionDiagramInvariance);
    assert_eq!(range.coordinate().units, "mmHg");
    let statement = range.statement();
    assert!(statement.contains("not a probability"));
    assert!(statement.contains("not a confidence or credible interval"));
}

#[test]
fn proportional_range_scales_with_the_magnitude_of_a_negative_point_value() {
    let range = transport_assumption_range(&spec(
        TransportRoute::MeasurementTransport,
        -4.0,
        ViolationFamily::ProportionalShift { bound: 0.5 },
    ))
    .unwrap();
    assert_eq!(range.grid(), &[0.0, 0.25, 0.5]);
    assert_eq!(range.lower(), &[-4.0, -5.0, -6.0]);
    assert_eq!(range.upper(), &[-4.0, -3.0, -2.0]);
    assert_eq!(range.coordinate().units, "fraction");
    assert_eq!(range.premise(), TransportPremise::MeasurementInvariance);
}

#[test]
fn reference_is_reached_at_the_closed_form_shift_or_not_at_all() {
    let additive = transport_assumption_range(&spec(
        TransportRoute::MechanismTransport,
        3.0,
        ViolationFamily::AdditiveShift { bound: 2.0 },
    ))
    .unwrap();
    assert_eq!(additive.reference_reached_at(1.5), Some(1.5));
    assert_eq!(additive.reference_reached_at(5.0), Some(2.0));
    assert_eq!(additive.reference_reached_at(0.0), None);
    let proportional = transport_assumption_range(&spec(
        TransportRoute::MechanismTransport,
        4.0,
        ViolationFamily::ProportionalShift { bound: 0.5 },
    ))
    .unwrap();
    assert_eq!(proportional.reference_reached_at(3.0), Some(0.25));
    assert_eq!(proportional.reference_reached_at(1.0), None);
}

#[test]
fn artifact_keeps_the_acting_decision_when_the_range_never_reaches_the_reference() {
    let range = transport_assumption_range(&spec(
        TransportRoute::PolicyTransport,
        3.0,
        ViolationFamily::AdditiveShift { bound: 2.0 },
    ))
    .unwrap();
    let artifact = range.to_artifact(&quantity("status_quo", "target"), 0.0).unwrap();
    assert_eq!(
        artifact.outcome(),
        Some(&SensitivityOutcome::InvariantAction { action: TRANSPORTED_ACTION_ID.into() })
    );
    assert_eq!(artifact.grid(), &[0.0, 1.0, 2.0]);
    assert_eq!(artifact.coordinate().maximum, 2.0);
}

#[test]
fn artifact_reports_no_robust_action_where_the_range_straddles_the_reference() {
    let range = transport_assumption_range(&spec(
        TransportRoute::TemporalPolicyTransport,
        1.0,
        ViolationFamily::AdditiveShift { bound: 2.0 },
    ))
    .unwrap();
    let artifact = range.to_artifact(&quantity("status_quo", "target"), 0.0).unwrap();
    match artifact.outcome() {
        Some(SensitivityOutcome::NoRobustAction {
            reason: NoRobustReason::MixedWithinRange { coordinates },
            ..
        }) => assert_eq!(coordinates, &vec![1.0, 2.0]),
        other => panic!("unexpected outcome {other:?}"),
    }
    assert_ne!(TRANSPORTED_ACTION_ID, REFERENCE_ACTION_ID);
}

#[test]
fn identity_changes_with_value_bound_route_and_target_at_unchanged_shape() {
    let base =
        || spec(TransportRoute::CohortDecision, 3.0, ViolationFamily::AdditiveShift { bound: 2.0 });
    let id = transport_assumption_range(&base()).unwrap().identity();
    assert_eq!(id, transport_assumption_range(&base()).unwrap().identity());
    let mut value = base();
    value.point_value = 3.5;
    let mut bound = base();
    bound.family = ViolationFamily::AdditiveShift { bound: 2.5 };
    let mut route = base();
    route.route = TransportRoute::SynthesisTransport;
    let mut snapshot = base();
    snapshot.provider_snapshot = "snapshot:2".into();
    for changed in [value, bound, route, snapshot] {
        assert_ne!(transport_assumption_range(&changed).unwrap().identity(), id);
    }
}

#[test]
fn every_scalar_route_states_a_range_and_the_structural_route_refuses_with_a_remedy() {
    for route in TransportRoute::ALL {
        let result = transport_assumption_range(&spec(
            route,
            2.0,
            ViolationFamily::AdditiveShift { bound: 1.0 },
        ));
        if route == TransportRoute::DiscoveredStructure {
            let refusal = result.unwrap_err();
            assert_eq!(refusal, TransportSensitivityRefusal::NoScalarFamily { route });
            assert_eq!(refusal.reason_code(), antecedent_core::reason_code!("route_not_supported"));
            assert_eq!(refusal.detail(), "transport_sensitivity.no_scalar_family");
            assert!(refusal.remedy().contains("transported effect"));
        } else {
            let range = result.unwrap();
            assert_eq!(range.route(), route);
            assert_eq!(range.range_at_bound(), (1.0, 3.0));
        }
    }
}

#[test]
fn malformed_requests_are_refused_with_typed_details() {
    let base =
        || spec(TransportRoute::CohortDecision, 3.0, ViolationFamily::AdditiveShift { bound: 2.0 });
    let refused = |s: TransportSensitivitySpec| transport_assumption_range(&s).unwrap_err();

    let mut s = base();
    s.point_value = f64::NAN;
    assert_eq!(refused(s), TransportSensitivityRefusal::NonFinitePoint);

    for bound in [0.0, -1.0, f64::INFINITY] {
        let mut s = base();
        s.family = ViolationFamily::AdditiveShift { bound };
        assert_eq!(refused(s), TransportSensitivityRefusal::InvalidBound);
    }
    for points in [0, 1, 1_000_000] {
        let mut s = base();
        s.grid_points = points;
        assert_eq!(refused(s), TransportSensitivityRefusal::InvalidGrid { points });
    }
    let mut s = base();
    s.point_value = 0.0;
    s.family = ViolationFamily::ProportionalShift { bound: 0.1 };
    assert_eq!(refused(s), TransportSensitivityRefusal::ProportionalAtZero);

    let mut s = base();
    s.target_population = "source".into();
    s.quantity = quantity("effect", "source");
    assert_eq!(refused(s), TransportSensitivityRefusal::NotTransported);

    let mut s = base();
    s.quantity = quantity("effect", "elsewhere");
    let refusal = refused(s);
    assert_eq!(refusal.reason_code(), antecedent_core::reason_code!("quantity_semantics_mismatch"));
    assert_eq!(refusal.detail(), "transport_sensitivity.quantity_not_in_target");

    let mut s = base();
    s.query_binding = "  ".into();
    assert_eq!(refused(s), TransportSensitivityRefusal::BlankIdentity("query_binding"));
}

#[test]
fn artifact_refuses_a_reference_in_another_population_and_a_non_finite_reference() {
    let range = transport_assumption_range(&spec(
        TransportRoute::CohortDecision,
        3.0,
        ViolationFamily::AdditiveShift { bound: 2.0 },
    ))
    .unwrap();
    let refusal = range.to_artifact(&quantity("status_quo", "elsewhere"), 0.0).unwrap_err();
    assert!(matches!(refusal, TransportSensitivityRefusal::Surface { .. }));
    assert_eq!(refusal.detail(), "transport_sensitivity.surface_refused");
    assert_eq!(
        range.to_artifact(&quantity("status_quo", "target"), f64::NAN).unwrap_err(),
        TransportSensitivityRefusal::NonFinitePoint
    );
    // the reference may not reuse the transported quantity's id
    assert!(range.to_artifact(&quantity("effect", "target"), 0.0).is_err());
}
