//! B4 joint vector-treatment artifact: hand-derived coefficients and full covariance through the
//! container, a recomputing consumer (summary replay for the model-based covariance, row replay
//! for a robust one), the row cap, resealed-mutation refusals against a retained identity,
//! unresealed tampering, unknown versions and the unmeasured calibration coordinate.
//!
//! Frozen dataset (n = 6, columns `1, t1, t2`): `t1 = [1,0,1,0,0,1]`, `t2 = [1,1,1,0,0,0]`,
//! `y = [3,2,5,1,0,4]`. `b = [2.75, 0.75]`, `RSS = 3.25`, `sigma^2 = 3.25 / 3`, and the treatment
//! block of the model-based covariance is `V = (3.25 / 36) [[9, -3], [-3, 9]]`. Calibration
//! (coverage, Type I error, power) is deliberately NOT measured here.
#![allow(clippy::float_cmp, reason = "bit-identical replay is the property under test")]

use antecedent_estimate::vector_treatment::{
    Contrast, NamedColumn, TreatmentColumn, VectorCovariance, VectorTreatmentInput,
    VectorTreatmentOptions, fit_vector_treatment,
};
use antecedent_io::vector_treatment_artifact::{
    AdjustmentColumnRequest, ContrastWire, TreatmentRequest, VECTOR_TREATMENT_ARTIFACT_VERSION,
    VECTOR_TREATMENT_CALIBRATION, VECTOR_TREATMENT_HC_ROW_CAP, VectorTreatmentArtifact,
    VectorTreatmentArtifactError, VectorTreatmentRequest, decode_parts, encode_parts,
};
use antecedent_kernels::erfc;

const X1: [f64; 6] = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0];
const X2: [f64; 6] = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
const W: [f64; 6] = [0.0, 1.0, 1.0, 0.0, 1.0, 0.0];
const Y: [f64; 6] = [3.0, 2.0, 5.0, 1.0, 0.0, 4.0];
const S: f64 = 3.25 / 36.0;

fn close(actual: f64, expected: f64, what: &str) {
    assert!((actual - expected).abs() <= 1e-9, "{what}: {actual} vs {expected}");
}

fn treatment(name: &str, values: &[f64], adjustment: &[&str]) -> TreatmentRequest {
    TreatmentRequest {
        name: name.into(),
        values: values.to_vec(),
        adjustment_set: adjustment.iter().map(|a| (*a).to_string()).collect(),
        row_snapshot: "snap".into(),
    }
}

fn contrasts() -> Vec<ContrastWire> {
    vec![
        ContrastWire {
            name: "diff".into(),
            weights: vec![("t1".into(), 1.0), ("t2".into(), -1.0)],
        },
        ContrastWire { name: "sum".into(), weights: vec![("t1".into(), 1.0), ("t2".into(), 1.0)] },
    ]
}

fn request() -> VectorTreatmentRequest {
    VectorTreatmentRequest {
        outcome: Y.to_vec(),
        row_snapshot: "snap".into(),
        adjustment: vec![],
        treatments: vec![treatment("t1", &X1, &[]), treatment("t2", &X2, &[])],
        covariance: "model_based".into(),
        contrasts: contrasts(),
    }
}

fn adjusted_request() -> VectorTreatmentRequest {
    VectorTreatmentRequest {
        adjustment: vec![AdjustmentColumnRequest { name: "w".into(), values: W.to_vec() }],
        treatments: vec![treatment("t1", &X1, &["w"]), treatment("t2", &X2, &["w"])],
        ..request()
    }
}

fn export(request: &VectorTreatmentRequest) -> (VectorTreatmentArtifact, Vec<u8>) {
    let artifact = VectorTreatmentArtifact::seal(request).unwrap();
    let bytes = artifact.to_bytes("b4-vector-test").unwrap();
    (artifact, bytes)
}

#[test]
fn b4_vector_artifact_round_trip_matches_hand_algebra_and_full_covariance() {
    let (artifact, bytes) = export(&request());
    let report =
        VectorTreatmentArtifact::from_bytes(&bytes, Some(artifact.identity())).unwrap().report();
    assert_eq!(report.treatments, ["t1", "t2"]);
    assert_eq!(report.replay, "summary");
    assert_eq!(report.covariance_kind, "model_based");
    let result = &report.result;
    assert_eq!(result.coefficients[0].name, "t1");
    close(result.coefficients[0].estimate, 2.75, "b1");
    close(result.coefficients[1].estimate, 0.75, "b2");
    // The FULL covariance, off-diagonals included.
    close(result.covariance[0], 9.0 * S, "V11");
    close(result.covariance[1], -3.0 * S, "V12");
    close(result.covariance[2], -3.0 * S, "V21");
    close(result.covariance[3], 9.0 * S, "V22");
    assert_eq!((result.n_rows, result.residual_df), (6, 3));
    close(result.residual_variance, 3.25 / 3.0, "sigma^2");
    // Contrast variance uses the off-diagonal: 24 s against the naive 18 s.
    let diff = &result.contrasts[0];
    close(diff.estimate, 2.0, "b1 - b2");
    close(diff.standard_error.powi(2), 24.0 * S, "diff variance");
    close(diff.naive_independent_standard_error.powi(2), 18.0 * S, "naive diff variance");
    let sum = &result.contrasts[1];
    close(sum.standard_error.powi(2), 12.0 * S, "sum variance");
}

#[test]
fn b4_vector_artifact_joint_wald_and_holm_match_closed_forms() {
    let report = export(&request()).0.report();
    // Wald = b' V^-1 b = 85.5 / (72 s) = 85.5 / 6.5; df 2 so p = exp(-stat / 2).
    let statistic = 85.5 / 6.5;
    close(report.result.joint_wald.statistic, statistic, "Wald");
    assert_eq!(report.result.joint_wald.degrees_of_freedom, 2);
    close(report.result.joint_wald.p_value, (-statistic / 2.0).exp(), "Wald p");
    assert_eq!(report.null, "all treatment coefficients are zero");
    // Holm over the declared two-contrast family.
    let raw: Vec<f64> = report
        .result
        .contrasts
        .iter()
        .map(|c| erfc((c.estimate / c.standard_error).abs() / 2.0_f64.sqrt()))
        .collect();
    for (c, p) in report.result.contrasts.iter().zip(&raw) {
        close(c.p_value, *p, "raw p");
    }
    let (low, high) = if raw[0] <= raw[1] { (0, 1) } else { (1, 0) };
    let holm_low = (2.0 * raw[low]).min(1.0);
    close(report.result.contrasts[low].p_holm, holm_low, "Holm of the smaller p");
    close(report.result.contrasts[high].p_holm, holm_low.max(raw[high]), "Holm of the larger p");
}

#[test]
fn b4_vector_artifact_calibration_is_unmeasured_and_says_so() {
    let report = export(&request()).0.report();
    assert_eq!(report.calibration, VECTOR_TREATMENT_CALIBRATION);
    assert_eq!(report.calibration, "unmeasured");
    assert_eq!(report.inference_claim, "asymptotic_wald_calibration_unmeasured");
    assert_eq!(report.version, VECTOR_TREATMENT_ARTIFACT_VERSION);
    assert!(report.caveats[0].contains("calibration is unmeasured"), "{:?}", report.caveats);
    assert!(report.caveats[0].contains("no interval"), "{:?}", report.caveats);
    assert!(report.caveats[1].contains("caller owns"), "{:?}", report.caveats);
}

#[test]
fn b4_vector_artifact_summary_replay_is_bit_identical_to_the_core_fit() {
    let artifact = VectorTreatmentArtifact::seal(&adjusted_request()).unwrap();
    let input = VectorTreatmentInput {
        outcome: Y.to_vec(),
        row_snapshot: "snap".into(),
        adjustment: vec![NamedColumn { name: "w".into(), values: W.to_vec() }],
        treatments: ["t1", "t2"]
            .iter()
            .zip([X1, X2])
            .map(|(name, values)| TreatmentColumn {
                name: (*name).into(),
                values: values.to_vec(),
                adjustment_set: vec!["w".into()],
                row_snapshot: "snap".into(),
            })
            .collect(),
    };
    let options = VectorTreatmentOptions {
        covariance: VectorCovariance::ModelBased,
        contrasts: vec![Contrast {
            name: "diff".into(),
            weights: vec![("t1".into(), 1.0), ("t2".into(), -1.0)],
        }],
    };
    let core = fit_vector_treatment(&input, &options).unwrap();
    let mut one = adjusted_request();
    one.contrasts.truncate(1);
    let sealed = VectorTreatmentArtifact::seal(&one).unwrap();
    let fit = sealed.fit();
    assert_eq!(fit.covariance, core.covariance);
    assert_eq!(fit.coefficients, core.coefficients);
    assert_eq!(fit.contrasts, core.contrasts);
    assert_eq!(fit.joint_wald, core.joint_wald);
    assert_eq!(fit.residual_variance, core.residual_variance);
    // The wider request carries the same adjustment declaration.
    assert_eq!(artifact.meta().adjustment, ["w"]);
}

#[test]
fn b4_vector_artifact_robust_covariance_replays_from_the_rows() {
    let mut hc = request();
    hc.covariance = "hc1".into();
    let (artifact, bytes) = export(&hc);
    assert_eq!(artifact.report().replay, "rows");
    let consumed = VectorTreatmentArtifact::from_bytes(&bytes, Some(artifact.identity())).unwrap();
    assert_eq!(consumed.report(), artifact.report());
    let input = VectorTreatmentInput {
        outcome: Y.to_vec(),
        row_snapshot: "snap".into(),
        adjustment: vec![],
        treatments: ["t1", "t2"]
            .iter()
            .zip([X1, X2])
            .map(|(name, values)| TreatmentColumn {
                name: (*name).into(),
                values: values.to_vec(),
                adjustment_set: vec![],
                row_snapshot: "snap".into(),
            })
            .collect(),
    };
    let options = VectorTreatmentOptions { covariance: VectorCovariance::Hc1, contrasts: vec![] };
    let core = fit_vector_treatment(&input, &options).unwrap();
    assert_eq!(consumed.fit().covariance, core.covariance);
    // HC1 differs from the model-based covariance on this design.
    assert_ne!(
        consumed.fit().covariance,
        VectorTreatmentArtifact::seal(&request()).unwrap().fit().covariance
    );
}

#[test]
fn b4_vector_artifact_robust_replay_above_the_row_cap_is_refused() {
    let n = VECTOR_TREATMENT_HC_ROW_CAP + 1;
    let column = |shift: usize| -> Vec<f64> { (0..n).map(|i| ((i + shift) % 7) as f64).collect() };
    let big = VectorTreatmentRequest {
        outcome: column(2),
        row_snapshot: "snap".into(),
        adjustment: vec![],
        treatments: vec![treatment("t1", &column(0), &[]), treatment("t2", &column(3), &[])],
        covariance: "hc0".into(),
        contrasts: vec![],
    };
    let (code, detail, _) = VectorTreatmentArtifact::seal(&big).unwrap_err().refusal().unwrap();
    assert_eq!(code, "route_not_supported");
    assert_eq!(detail, "vector_treatment.hc_replay_row_cap_exceeded");
    // The model-based covariance replays from a summary and has no row cap.
    let mut model = big;
    model.covariance = "model_based".into();
    let sealed = VectorTreatmentArtifact::seal(&model).unwrap();
    assert_eq!(sealed.report().replay, "summary");
    assert_eq!(sealed.meta().n_rows, n as u64);
}

#[test]
fn b4_vector_artifact_incompatible_treatments_refuse_with_the_core_detail() {
    let mut other_set = adjusted_request();
    other_set.treatments[1].adjustment_set = vec![];
    let (code, detail, _) =
        VectorTreatmentArtifact::seal(&other_set).unwrap_err().refusal().unwrap();
    assert_eq!(
        (code, detail.as_str()),
        ("route_not_supported", "vector_treatment.adjustment_set_mismatch")
    );

    let mut other_snapshot = request();
    other_snapshot.treatments[1].row_snapshot = "elsewhere".into();
    let (code, detail, _) =
        VectorTreatmentArtifact::seal(&other_snapshot).unwrap_err().refusal().unwrap();
    assert_eq!(
        (code, detail.as_str()),
        ("route_not_supported", "vector_treatment.row_snapshot_mismatch")
    );

    let mut short = request();
    short.treatments[1].values.pop();
    let (_, detail, _) = VectorTreatmentArtifact::seal(&short).unwrap_err().refusal().unwrap();
    assert_eq!(detail, "vector_treatment.row_count_mismatch");

    let mut constant = request();
    constant.treatments[1].values = vec![1.0; 6];
    let (code, detail, _) =
        VectorTreatmentArtifact::seal(&constant).unwrap_err().refusal().unwrap();
    assert_eq!(
        (code, detail.as_str()),
        ("design_rank_deficient", "vector_treatment.treatment_without_variation")
    );

    let mut collinear = request();
    collinear.treatments[1].values = X1.iter().map(|v| 2.0 * v).collect();
    let (_, detail, _) = VectorTreatmentArtifact::seal(&collinear).unwrap_err().refusal().unwrap();
    assert_eq!(detail, "vector_treatment.collinear_treatments");

    let mut single = request();
    single.treatments.truncate(1);
    let (_, detail, _) = VectorTreatmentArtifact::seal(&single).unwrap_err().refusal().unwrap();
    assert_eq!(detail, "vector_treatment.too_few_treatments");

    let mut unknown = request();
    unknown.contrasts[0].weights[0].0 = "zzz".into();
    let (_, detail, _) = VectorTreatmentArtifact::seal(&unknown).unwrap_err().refusal().unwrap();
    assert_eq!(detail, "vector_treatment.unknown_contrast_coefficient");

    let mut bad_kind = request();
    bad_kind.covariance = "cluster".into();
    let (_, detail, _) = VectorTreatmentArtifact::seal(&bad_kind).unwrap_err().refusal().unwrap();
    assert_eq!(detail, "vector_treatment.invalid_request");
}

/// Seal `mutated` as a consumer-consistent artifact and consume it against the identity of
/// `original`: the mutation is resealed (its own digests are correct) but must still refuse.
fn resealed_mutation_refuses(
    original: &VectorTreatmentRequest,
    mutated: &VectorTreatmentRequest,
    field: &'static str,
) {
    let (artifact, _) = export(original);
    let (_, mutated_bytes) = export(mutated);
    assert!(VectorTreatmentArtifact::from_bytes(&mutated_bytes, None).is_ok(), "{field}");
    let error =
        VectorTreatmentArtifact::from_bytes(&mutated_bytes, Some(artifact.identity())).unwrap_err();
    assert_eq!(error, VectorTreatmentArtifactError::IdentityMismatch { field }, "{field}");
    let (code, detail, _) = error.refusal().unwrap();
    assert_eq!(code, "route_not_supported");
    assert_eq!(detail, "vector_treatment.wrong_contract");
}

#[test]
fn b4_vector_artifact_resealed_identity_changes_are_refused() {
    let original = adjusted_request();
    let mut snapshot = original.clone();
    snapshot.row_snapshot = "snap-2".into();
    for t in &mut snapshot.treatments {
        t.row_snapshot = "snap-2".into();
    }
    resealed_mutation_refuses(&original, &snapshot, "row_snapshot");

    let mut renamed = original.clone();
    renamed.adjustment[0].name = "v".into();
    for t in &mut renamed.treatments {
        t.adjustment_set = vec!["v".into()];
    }
    resealed_mutation_refuses(&original, &renamed, "adjustment_set");

    let mut reordered = original.clone();
    reordered.treatments.reverse();
    reordered.contrasts.clear();
    let mut base = original.clone();
    base.contrasts.clear();
    resealed_mutation_refuses(&base, &reordered, "treatment_vector");

    let mut data = original.clone();
    data.outcome[0] = 3.5;
    resealed_mutation_refuses(&original, &data, "design");

    let mut declared = original.clone();
    declared.contrasts.truncate(1);
    resealed_mutation_refuses(&original, &declared, "contrasts");

    let mut kind = original.clone();
    kind.covariance = "hc0".into();
    resealed_mutation_refuses(&original, &kind, "design");
}

#[test]
fn b4_vector_artifact_coefficient_order_is_part_of_the_identity() {
    let forward = VectorTreatmentArtifact::seal(&request()).unwrap();
    let mut swapped = request();
    swapped.treatments.reverse();
    swapped.contrasts.clear();
    let reversed = VectorTreatmentArtifact::seal(&swapped).unwrap();
    assert_ne!(forward.identity().treatment_id, reversed.identity().treatment_id);
    assert_ne!(forward.identity().digest, reversed.identity().digest);
    let r = reversed.report();
    assert_eq!(r.treatments, ["t2", "t1"]);
    close(r.result.coefficients[0].estimate, 0.75, "b2 first");
    close(r.result.covariance[1], -3.0 * S, "V off-diagonal is symmetric under reordering");
}

#[test]
fn b4_vector_artifact_unresealed_tampering_does_not_replay() {
    let (artifact, bytes) = export(&request());
    let (meta, numbers) = decode_parts(&bytes).unwrap();

    let mut forged = meta.clone();
    forged.result.coefficients[0].estimate += 1.0;
    let reencoded = encode_parts(&forged, &numbers, "forged").unwrap();
    assert!(matches!(
        VectorTreatmentArtifact::from_bytes(&reencoded, None).unwrap_err(),
        VectorTreatmentArtifactError::ResultMismatch(_)
    ));

    let mut forged = meta.clone();
    forged.result.covariance[1] = 0.0;
    let reencoded = encode_parts(&forged, &numbers, "forged").unwrap();
    assert!(matches!(
        VectorTreatmentArtifact::from_bytes(&reencoded, None).unwrap_err(),
        VectorTreatmentArtifactError::ResultMismatch(_)
    ));

    let mut forged = meta.clone();
    forged.result.contrasts[0].p_holm *= 0.5;
    let reencoded = encode_parts(&forged, &numbers, "forged").unwrap();
    assert!(matches!(
        VectorTreatmentArtifact::from_bytes(&reencoded, None).unwrap_err(),
        VectorTreatmentArtifactError::ResultMismatch(_)
    ));

    // A changed residual sum of squares (the last embedded number) under the stored identity.
    let mut changed = numbers.clone();
    let last = changed.len() - 8;
    changed[last..].copy_from_slice(&9.0_f64.to_le_bytes());
    let reencoded = encode_parts(&meta, &changed, "forged").unwrap();
    assert_eq!(
        VectorTreatmentArtifact::from_bytes(&reencoded, None).unwrap_err(),
        VectorTreatmentArtifactError::IdentityMismatch { field: "design" }
    );
    // A truncated numbers section is malformed.
    let reencoded = encode_parts(&meta, &numbers[..numbers.len() - 8], "forged").unwrap();
    assert!(matches!(
        VectorTreatmentArtifact::from_bytes(&reencoded, None).unwrap_err(),
        VectorTreatmentArtifactError::Malformed(_)
    ));

    let mut forged = meta.clone();
    forged.null = "no treatment is non-zero".into();
    let reencoded = encode_parts(&forged, &numbers, "forged").unwrap();
    assert_eq!(
        VectorTreatmentArtifact::from_bytes(&reencoded, None).unwrap_err(),
        VectorTreatmentArtifactError::IdentityMismatch { field: "null" }
    );
    let mut forged = meta.clone();
    forged.calibration = "calibrated".into();
    let reencoded = encode_parts(&forged, &numbers, "forged").unwrap();
    assert_eq!(
        VectorTreatmentArtifact::from_bytes(&reencoded, None).unwrap_err(),
        VectorTreatmentArtifactError::UnsupportedSemantics("calibration")
    );
    let mut forged = meta.clone();
    forged.caveats.clear();
    let reencoded = encode_parts(&forged, &numbers, "forged").unwrap();
    assert_eq!(
        VectorTreatmentArtifact::from_bytes(&reencoded, None).unwrap_err(),
        VectorTreatmentArtifactError::UnsupportedSemantics("caveats")
    );
    // A changed declared contrast under the stored identity.
    let mut forged = meta.clone();
    forged.contrasts[0].weights[1].1 = 2.0;
    let reencoded = encode_parts(&forged, &numbers, "forged").unwrap();
    assert_eq!(
        VectorTreatmentArtifact::from_bytes(&reencoded, None).unwrap_err(),
        VectorTreatmentArtifactError::IdentityMismatch { field: "contrasts" }
    );
    assert!(VectorTreatmentArtifact::from_bytes(&bytes, Some(artifact.identity())).is_ok());
}

#[test]
fn b4_vector_artifact_unknown_major_version_and_corrupt_bytes_refuse() {
    let (_, bytes) = export(&request());
    let (meta, numbers) = decode_parts(&bytes).unwrap();
    for version in [0_u16, 2, 7] {
        let mut other = meta.clone();
        other.version = version;
        let reencoded = encode_parts(&other, &numbers, "other-version").unwrap();
        assert_eq!(
            VectorTreatmentArtifact::from_bytes(&reencoded, None).unwrap_err(),
            VectorTreatmentArtifactError::UnsupportedVersion { version }
        );
    }
    let mut other = meta;
    other.feature = "vector_treatment_v2".into();
    let reencoded = encode_parts(&other, &numbers, "other-feature").unwrap();
    assert_eq!(
        VectorTreatmentArtifact::from_bytes(&reencoded, None).unwrap_err(),
        VectorTreatmentArtifactError::UnsupportedSemantics("feature marker")
    );
    let mut corrupt = bytes.clone();
    let middle = corrupt.len() / 2;
    corrupt[middle] ^= 0xff;
    assert!(VectorTreatmentArtifact::from_bytes(&corrupt, None).is_err());
    assert!(VectorTreatmentArtifact::from_bytes(&bytes[..bytes.len() - 5], None).is_err());
    assert!(VectorTreatmentArtifact::from_bytes(&[], None).is_err());
}
