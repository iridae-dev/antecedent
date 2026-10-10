//! B3 mechanism discrepancy artifact: hand-derived acceptance numbers through the container, a
//! recomputing consumer, resealed-mutation refusals against a retained identity, unresealed
//! tampering, unknown versions, the never-certifies-invariance flag and the unmeasured
//! calibration coordinate.
//!
//! Frozen summaries (one parent `x = [0,1,2,3]`, n = 4): `X'X = [[4,6],[6,14]]`; the source has
//! `X'y = [10,19]`, `y'y = 30` (`b = [1.3, 0.8]`, `RSS = 1.8`, `sigma^2 = 0.9`); the target is the
//! source outcome plus `0.5 + 0.5 x`, `X'y = [15,29]`, `y'y = 66.5` (`b = [1.8, 1.3]`, the same
//! `RSS`). Both covariances are `[[0.63,-0.27],[-0.27,0.18]]`, the summed covariance is twice
//! that, the difference is `(0.5, 0.5)` and the Wald statistic is `25/6` with df 2, so
//! `p = exp(-25/12)`.
#![allow(clippy::float_cmp, reason = "bit-identical replay is the property under test")]

use antecedent_io::mechanism_discrepancy_artifact::{
    MECHANISM_DISCREPANCY_ARTIFACT_VERSION, MECHANISM_DISCREPANCY_CALIBRATION, MeasurementWire,
    MechanismDiscrepancyArtifact, MechanismDiscrepancyArtifactError, MechanismDiscrepancyMeta,
    MechanismDiscrepancyRequestWire, ParentWire, PopulationWire, decode_parts, encode_parts,
};

fn measurement(protocol: &str, parent_unit: &str) -> MeasurementWire {
    MeasurementWire {
        node: "V".into(),
        node_unit: "mg".into(),
        parents: vec![ParentWire { name: "x".into(), unit: parent_unit.into() }],
        protocol_id: protocol.into(),
    }
}

fn population(label: &str, xty: [f64; 2], yty: f64) -> PopulationWire {
    PopulationWire {
        label: label.into(),
        measurement: measurement("protocol-1", "cm"),
        n: 4,
        xtx: vec![4.0, 6.0, 6.0, 14.0],
        xty: xty.to_vec(),
        yty,
    }
}

fn request() -> MechanismDiscrepancyRequestWire {
    MechanismDiscrepancyRequestWire {
        source: population("source", [10.0, 19.0], 30.0),
        target: population("target", [15.0, 29.0], 66.5),
        compare_intercept: true,
        alpha: 0.05,
        power: 0.8,
        dependence: "independent".into(),
    }
}

fn close(actual: f64, expected: f64, what: &str) {
    assert!((actual - expected).abs() <= 1e-9, "{what}: {actual} vs {expected}");
}

fn export(request: &MechanismDiscrepancyRequestWire) -> (MechanismDiscrepancyArtifact, Vec<u8>) {
    let artifact = MechanismDiscrepancyArtifact::seal(request).unwrap();
    let bytes = artifact.to_bytes("b3-discrepancy-test").unwrap();
    (artifact, bytes)
}

#[test]
fn b3_discrepancy_artifact_acceptance_numbers_replay_through_the_container() {
    let (artifact, bytes) = export(&request());
    let consumed =
        MechanismDiscrepancyArtifact::from_bytes(&bytes, Some(artifact.identity())).unwrap();
    let report = consumed.report();
    close(report.result.statistic, 25.0 / 6.0, "W");
    assert_eq!(report.result.degrees_of_freedom, 2);
    close(report.result.p_value, (-25.0_f64 / 12.0).exp(), "p = exp(-25/12)");
    assert_eq!(report.result.conclusion, "not_rejected");
    assert_eq!(report.result.coefficient_names, ["(intercept)", "x"]);
    close(report.result.source.coefficients[0], 1.3, "source intercept");
    close(report.result.target.coefficients[1], 1.3, "target slope");
    close(report.result.coefficients[1].standard_error, 0.6, "se slope");
    // The six-decimal normal quantiles agree with the bisection values to well under 1e-6.
    let mdd = report.result.coefficients[1].minimal_detectable_difference;
    assert!((mdd - (1.959_964 + 0.841_621) * 0.6).abs() < 1e-6, "mdd {mdd}");
    assert_eq!(report.calibration, MECHANISM_DISCREPANCY_CALIBRATION);
    assert_eq!(report.calibration, "unmeasured");
    assert_eq!(report.version, MECHANISM_DISCREPANCY_ARTIFACT_VERSION);
    assert!(!report.result.non_rejection_certifies_invariance);
    assert_eq!(report.result.informs_selection_on, ["V"]);
    assert!(report.result.power_statement.contains("not detectable"));
    assert!(report.caveats[0].contains("does not certify"), "{:?}", report.caveats);
    assert!(report.alignment.contains("cannot be excluded"));
    assert_eq!(report, artifact.report());
}

#[test]
fn b3_discrepancy_artifact_incomparable_measurements_refuse_with_the_core_detail() {
    let mut units = request();
    units.target.measurement = measurement("protocol-1", "m");
    let (code, detail, _) =
        MechanismDiscrepancyArtifact::seal(&units).unwrap_err().refusal().expect("typed refusal");
    assert_eq!(code, "route_not_supported");
    assert_eq!(detail, "mechanism_discrepancy.incomparable_measurements");

    let mut protocol = request();
    protocol.target.measurement = measurement("protocol-2", "cm");
    let (_, detail, _) =
        MechanismDiscrepancyArtifact::seal(&protocol).unwrap_err().refusal().unwrap();
    assert_eq!(detail, "mechanism_discrepancy.incomparable_measurements");
}

#[test]
fn b3_discrepancy_artifact_dependence_must_be_declared_independent() {
    for kind in ["unknown", "shared_units"] {
        let mut dependent = request();
        dependent.dependence = kind.into();
        let (code, detail, _) =
            MechanismDiscrepancyArtifact::seal(&dependent).unwrap_err().refusal().unwrap();
        assert_eq!(code, "route_not_supported", "{kind}");
        assert_eq!(detail, "mechanism_discrepancy.dependence_unknown", "{kind}");
    }
    let mut bogus = request();
    bogus.dependence = "bogus".into();
    let (code, detail, _) =
        MechanismDiscrepancyArtifact::seal(&bogus).unwrap_err().refusal().unwrap();
    assert_eq!(code, "invalid_argument");
    assert_eq!(detail, "mechanism_discrepancy.invalid_request");
}

/// Seal `mutated` as a consumer-consistent artifact and consume it against the identity of
/// `original`: the mutation is resealed (its own digests are correct) but must still refuse.
fn resealed_mutation_refuses(
    original: &MechanismDiscrepancyRequestWire,
    mutated: &MechanismDiscrepancyRequestWire,
    field: &'static str,
) {
    let (artifact, _) = export(original);
    let (_, mutated_bytes) = export(mutated);
    // Alone, the mutated artifact is internally consistent ...
    assert!(MechanismDiscrepancyArtifact::from_bytes(&mutated_bytes, None).is_ok(), "{field}");
    // ... but against the identity the consumer retained it is refused.
    let error = MechanismDiscrepancyArtifact::from_bytes(&mutated_bytes, Some(artifact.identity()))
        .unwrap_err();
    assert_eq!(error, MechanismDiscrepancyArtifactError::IdentityMismatch { field }, "{field}");
    let (code, detail, _) = error.refusal().unwrap();
    assert_eq!(code, "route_not_supported");
    assert_eq!(detail, "mechanism_discrepancy.wrong_contract");
}

#[test]
fn b3_discrepancy_artifact_resealed_measurement_contract_is_refused() {
    let original = request();
    let mut protocol = original.clone();
    protocol.source.measurement = measurement("protocol-9", "cm");
    protocol.target.measurement = measurement("protocol-9", "cm");
    resealed_mutation_refuses(&original, &protocol, "measurement");
    let mut units = original.clone();
    units.source.measurement = measurement("protocol-1", "m");
    units.target.measurement = measurement("protocol-1", "m");
    resealed_mutation_refuses(&original, &units, "measurement");
}

#[test]
fn b3_discrepancy_artifact_resealed_evidence_is_refused() {
    let original = request();
    let mut source = original.clone();
    source.source.yty = 31.0;
    resealed_mutation_refuses(&original, &source, "source_evidence");
    let mut relabelled = original.clone();
    relabelled.source.label = "elsewhere".into();
    resealed_mutation_refuses(&original, &relabelled, "source_evidence");
    let mut target = original.clone();
    target.target.yty = 67.0;
    resealed_mutation_refuses(&original, &target, "target_evidence");
}

#[test]
fn b3_discrepancy_artifact_resealed_design_is_refused() {
    let original = request();
    let mut level = original.clone();
    level.alpha = 0.01;
    resealed_mutation_refuses(&original, &level, "design");
    let mut power = original.clone();
    power.power = 0.9;
    resealed_mutation_refuses(&original, &power, "design");
    let mut slope_only = original.clone();
    slope_only.compare_intercept = false;
    resealed_mutation_refuses(&original, &slope_only, "design");
}

#[test]
fn b3_discrepancy_artifact_unresealed_tampering_does_not_replay() {
    let (artifact, bytes) = export(&request());
    let (meta, numbers) = decode_parts(&bytes).unwrap();
    let reencode = |m: &MechanismDiscrepancyMeta, n: &[u8]| encode_parts(m, n, "forged").unwrap();

    // A stored statistic, p-value and detectability that the recomputation does not reproduce.
    let mut forged = meta.clone();
    forged.result.statistic += 1.0;
    assert!(matches!(
        MechanismDiscrepancyArtifact::from_bytes(&reencode(&forged, &numbers), None).unwrap_err(),
        MechanismDiscrepancyArtifactError::ResultMismatch(_)
    ));
    let mut forged = meta.clone();
    forged.result.coefficients[1].minimal_detectable_difference *= 0.5;
    assert!(matches!(
        MechanismDiscrepancyArtifact::from_bytes(&reencode(&forged, &numbers), None).unwrap_err(),
        MechanismDiscrepancyArtifactError::ResultMismatch(_)
    ));

    // A claim that non-rejection certifies invariance is refused outright.
    let mut forged = meta.clone();
    forged.result.non_rejection_certifies_invariance = true;
    assert_eq!(
        MechanismDiscrepancyArtifact::from_bytes(&reencode(&forged, &numbers), None).unwrap_err(),
        MechanismDiscrepancyArtifactError::UnsupportedSemantics(
            "non-rejection certifies invariance"
        )
    );

    // A changed source y'y (the seventh number) under the stored identity.
    let mut changed = numbers.clone();
    changed[48..56].copy_from_slice(&31.0_f64.to_le_bytes());
    assert_eq!(
        MechanismDiscrepancyArtifact::from_bytes(&reencode(&meta, &changed), None).unwrap_err(),
        MechanismDiscrepancyArtifactError::IdentityMismatch { field: "source_evidence" }
    );

    // A changed stored unit makes the measurements incomparable: the core refuses.
    let mut forged = meta.clone();
    forged.target.measurement.node_unit = "g".into();
    let error =
        MechanismDiscrepancyArtifact::from_bytes(&reencode(&forged, &numbers), None).unwrap_err();
    let (code, detail, _) = error.refusal().unwrap();
    assert_eq!(code, "route_not_supported");
    assert_eq!(detail, "mechanism_discrepancy.incomparable_measurements");

    // A changed stored row count disagrees with the intercept sum of squares.
    let mut forged = meta.clone();
    forged.source.n = 5;
    let (_, detail, _) =
        MechanismDiscrepancyArtifact::from_bytes(&reencode(&forged, &numbers), None)
            .unwrap_err()
            .refusal()
            .unwrap();
    assert_eq!(detail, "mechanism_discrepancy.inconsistent_summary");

    // A changed stored level.
    let mut forged = meta.clone();
    forged.alpha = 0.01;
    assert_eq!(
        MechanismDiscrepancyArtifact::from_bytes(&reencode(&forged, &numbers), None).unwrap_err(),
        MechanismDiscrepancyArtifactError::IdentityMismatch { field: "design" }
    );

    // A changed stored null, calibration coordinate, caveat or alignment.
    let mut forged = meta.clone();
    forged.null = "the mechanism is invariant".into();
    assert_eq!(
        MechanismDiscrepancyArtifact::from_bytes(&reencode(&forged, &numbers), None).unwrap_err(),
        MechanismDiscrepancyArtifactError::IdentityMismatch { field: "null" }
    );
    let mut forged = meta.clone();
    forged.calibration = "calibrated".into();
    assert_eq!(
        MechanismDiscrepancyArtifact::from_bytes(&reencode(&forged, &numbers), None).unwrap_err(),
        MechanismDiscrepancyArtifactError::UnsupportedSemantics("calibration")
    );
    let mut forged = meta.clone();
    forged.caveats.clear();
    assert_eq!(
        MechanismDiscrepancyArtifact::from_bytes(&reencode(&forged, &numbers), None).unwrap_err(),
        MechanismDiscrepancyArtifactError::UnsupportedSemantics("caveats")
    );
    let mut forged = meta.clone();
    forged.alignment = "non-rejection excludes a selection node".into();
    assert_eq!(
        MechanismDiscrepancyArtifact::from_bytes(&reencode(&forged, &numbers), None).unwrap_err(),
        MechanismDiscrepancyArtifactError::UnsupportedSemantics("alignment")
    );

    // A truncated numbers section.
    assert!(matches!(
        MechanismDiscrepancyArtifact::from_bytes(&reencode(&meta, &numbers[8..]), None)
            .unwrap_err(),
        MechanismDiscrepancyArtifactError::Malformed(_)
    ));
    // The untouched artifact still consumes.
    assert!(MechanismDiscrepancyArtifact::from_bytes(&bytes, Some(artifact.identity())).is_ok());
}

#[test]
fn b3_discrepancy_artifact_unknown_version_and_corruption_refuse() {
    let (_, bytes) = export(&request());
    let (meta, numbers) = decode_parts(&bytes).unwrap();
    let mut future = meta.clone();
    future.version = 2;
    let reencoded = encode_parts(&future, &numbers, "future").unwrap();
    assert_eq!(
        MechanismDiscrepancyArtifact::from_bytes(&reencoded, None).unwrap_err(),
        MechanismDiscrepancyArtifactError::UnsupportedVersion { version: 2 }
    );
    let mut feature = meta.clone();
    feature.feature = "mechanism_discrepancy_v2".into();
    let reencoded = encode_parts(&feature, &numbers, "future").unwrap();
    assert_eq!(
        MechanismDiscrepancyArtifact::from_bytes(&reencoded, None).unwrap_err(),
        MechanismDiscrepancyArtifactError::UnsupportedSemantics("feature marker")
    );
    assert!(MechanismDiscrepancyArtifact::from_bytes(&bytes[..bytes.len() / 2], None).is_err());
    assert!(MechanismDiscrepancyArtifact::from_bytes(&[], None).is_err());
}
