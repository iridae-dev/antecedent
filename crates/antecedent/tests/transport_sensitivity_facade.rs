//! The assumption range of a transported response through the facade: built from the
//! response's own point value, quantity and receipt, sealed into the one sensitivity
//! artifact, and never inherited by another target.
//!
//! Oracle. The transported mean of `effect` in `clinic_b` is `2.0`. An additive shift
//! bound of `0.5` on three grid points is `0, 0.25, 0.5`, so the range is `[2, 2]`,
//! `[1.75, 2.25]`, `[1.5, 2.5]`. A proportional bound of `0.25` of `2.0` is a shift of
//! `0.5` at the bound: `[1.5, 2.5]`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::sensitivity_decision::{consume, export};
use antecedent::analysis::transport_sensitivity::{
    DETAIL_COORDINATE_RANGE, DETAIL_LAW_RESPONSE, DETAIL_ROUTE_UNDECLARED, TransportRoute,
    TransportSensitivityEntryRefusal, TransportSensitivityRefusal, TransportSensitivityRequest,
    ViolationFamily, route_of, transport_range, transport_sensitivity_artifact,
    transport_sensitivity_spec,
};
use antecedent::analysis::{TransportDeclaration, TransportedResponse, TransportedRoute};
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::SupportStatus;
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io::quantity_wire::DistributionMeaningWire;

fn quantity(variable: &str, population: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: variable.into(),
        role: QuantityRole::Outcome,
        units: "units".into(),
        population_id: population.into(),
        regime_id: "contrast:do(t=1)-do(t=0)".into(),
        horizon: 0,
        functional_id: "mean".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn declaration(target: &str) -> TransportDeclaration {
    TransportDeclaration {
        route: TransportedRoute::Declared,
        source_identity: "transport-1".into(),
        source_populations: vec!["trial".into()],
        target_population: target.into(),
        transport_identities: vec!["diagram-1".into()],
        assumptions: vec!["selection acts on age only".into()],
        support: SupportStatus::Extrapolative,
        uncalibrated_standard_error: None,
        provider_id: "transport-provider".into(),
        snapshot_id: "snapshot-1".into(),
        causal_contract_id: "contract-1".into(),
    }
}

fn point(target: &str, value: f64) -> TransportedResponse {
    TransportedResponse::declared_point(
        vec![quantity("effect", target)],
        vec![value],
        &declaration(target),
    )
    .unwrap()
}

fn request(family: ViolationFamily) -> TransportSensitivityRequest {
    TransportSensitivityRequest {
        coordinate: 0,
        route: Some(TransportRoute::CohortDecision),
        family,
        grid_points: 3,
    }
}

#[test]
fn the_range_is_built_from_the_responses_value_quantity_and_receipt() {
    let response = point("clinic_b", 2.0);
    let spec = transport_sensitivity_spec(
        &response,
        &request(ViolationFamily::AdditiveShift { bound: 0.5 }),
    )
    .unwrap();
    assert!((spec.point_value - 2.0).abs() < 1e-12);
    assert_eq!(spec.source_population, "trial");
    assert_eq!(spec.target_population, "clinic_b");
    assert_eq!(spec.query_binding, "transport-1");
    assert_eq!(spec.provider_snapshot, "snapshot-1");

    let range = transport_range(&response, &request(ViolationFamily::AdditiveShift { bound: 0.5 }))
        .unwrap();
    assert_eq!(range.grid(), &[0.0, 0.25, 0.5]);
    assert_eq!(range.lower(), &[2.0, 1.75, 1.5]);
    assert_eq!(range.upper(), &[2.0, 2.25, 2.5]);
    assert_eq!(range.range_at_bound(), (1.5, 2.5));

    let proportional =
        transport_range(&response, &request(ViolationFamily::ProportionalShift { bound: 0.25 }))
            .unwrap();
    assert_eq!(proportional.range_at_bound(), (1.5, 2.5));
}

#[test]
fn the_sealed_artifact_exports_and_a_consumer_recomputes_it() {
    let response = point("clinic_b", 2.0);
    let reference = quantity("keep", "clinic_b");
    let artifact = transport_sensitivity_artifact(
        &response,
        &request(ViolationFamily::AdditiveShift { bound: 0.5 }),
        &reference,
        2.2,
    )
    .unwrap();
    assert_eq!(artifact.grid(), &[0.0, 0.25, 0.5]);
    let bytes = export(&artifact, "transport-sensitivity").unwrap();
    let consumed = consume(&bytes, Some(artifact.identity())).unwrap();
    assert_eq!(consumed.identity(), artifact.identity());

    // The same family on a response about another target is another range with another
    // identity: an assumption range is never inherited by another target.
    let other = point("clinic_c", 2.0);
    let other_artifact = transport_sensitivity_artifact(
        &other,
        &request(ViolationFamily::AdditiveShift { bound: 0.5 }),
        &quantity("keep", "clinic_c"),
        2.2,
    )
    .unwrap();
    assert_ne!(other_artifact.identity(), artifact.identity());
    // And a different point value at the same declared family moves the range.
    let moved = transport_range(
        &point("clinic_b", 3.0),
        &request(ViolationFamily::AdditiveShift { bound: 0.5 }),
    )
    .unwrap();
    assert_eq!(moved.lower(), &[3.0, 2.75, 2.5]);
}

#[test]
fn a_response_that_cannot_be_shifted_is_refused_with_its_own_reason() {
    let response = point("clinic_b", 2.0);
    // A declared transport names no route of its own.
    assert_eq!(route_of(TransportedRoute::Declared), None);
    let mut undeclared = request(ViolationFamily::AdditiveShift { bound: 0.5 });
    undeclared.route = None;
    let error = transport_range(&response, &undeclared).unwrap_err();
    assert_eq!(error, TransportSensitivityEntryRefusal::RouteUndeclared);
    assert_eq!(error.detail(), DETAIL_ROUTE_UNDECLARED);
    assert_eq!(error.reason_code(), "route_not_supported");

    // A coordinate outside the response.
    let mut outside = request(ViolationFamily::AdditiveShift { bound: 0.5 });
    outside.coordinate = 3;
    let error = transport_range(&response, &outside).unwrap_err();
    assert_eq!(
        error,
        TransportSensitivityEntryRefusal::CoordinateOutOfRange { index: 3, available: 1 }
    );
    assert_eq!(error.detail(), DETAIL_COORDINATE_RANGE);

    // A structural premise has no scalar family.
    let mut structural = request(ViolationFamily::AdditiveShift { bound: 0.5 });
    structural.route = Some(TransportRoute::DiscoveredStructure);
    let error = transport_range(&response, &structural).unwrap_err();
    assert!(matches!(
        error,
        TransportSensitivityEntryRefusal::Range(TransportSensitivityRefusal::NoScalarFamily { .. })
    ));
    assert_eq!(error.detail(), "transport_sensitivity.no_scalar_family");

    // A proportional shift of zero is no range.
    let zero = point("clinic_b", 0.0);
    let error = transport_range(&zero, &request(ViolationFamily::ProportionalShift { bound: 0.5 }))
        .unwrap_err();
    assert_eq!(
        error,
        TransportSensitivityEntryRefusal::Range(TransportSensitivityRefusal::ProportionalAtZero)
    );
}

#[test]
fn a_transported_law_has_no_single_point_to_shift() {
    let outcome = quantity("y", "clinic_b");
    let identity = DistributionIdentity::new(
        DistributionMeaningWire::InterventionalPredictive,
        std::slice::from_ref(&outcome),
        DrawAlignment::Joint,
        DistributionProvenance {
            source_id: "transport".into(),
            provider_id: "transport-provider".into(),
            rng_id: "deterministic_exact".into(),
            snapshot_id: "snapshot-1".into(),
            causal_contract_id: "contract-1".into(),
        },
    )
    .unwrap();
    let law = DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".into(), "quantity".into()],
            shape: [4, 1],
            weights: None,
            supported: None,
            calibration: DistributionCalibration::PointOnly,
            trust: DistributionTrust::Unverified,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        vec![1.0, 1.0, 0.0, 0.0],
    )
    .unwrap();
    let response = TransportedResponse::declared_law(law, &declaration("clinic_b")).unwrap();
    let error = transport_range(&response, &request(ViolationFamily::AdditiveShift { bound: 0.5 }))
        .unwrap_err();
    assert_eq!(error, TransportSensitivityEntryRefusal::LawResponse);
    assert_eq!(error.detail(), DETAIL_LAW_RESPONSE);
    assert!(!error.remedy().is_empty());
}
