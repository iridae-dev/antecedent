//! B4 categorical-treatment artifact: group-mean closed forms through the container, a
//! recomputing consumer (summary replay for the model-based covariance, row replay for a robust
//! one), Holm multiplicity over the declared family, the declared monotonicity test and its null,
//! sparse/absent/undeclared-level refusals, resealed-mutation refusals against a retained
//! identity, unresealed tampering and the unmeasured calibration coordinate.
//!
//! Frozen dataset A (rows interleaved on purpose): `a = [1,3]`, `b = [4,6,5]`, `c = [9,7,11,9]`;
//! `RSS = 12`, `sigma^2 = 2`; with reference `a`, `b_b = 3`, `b_c = 7`, and
//! `V = [[5/3, 1], [1, 3/2]]`. Frozen dataset M (ordered `lo < mid < hi`, three rows each):
//! `lo = [3,2,1]`, `mid = [2,1,0]`, `hi = [1,0,-1]`; `sigma^2 = 1`, every adjacent-step variance
//! is `2/3` and every step is `-1`. Calibration (Type I error, coverage, power) is deliberately
//! NOT measured here.
#![allow(clippy::float_cmp, reason = "bit-identical replay is the property under test")]

use antecedent_io::categorical_treatment_artifact::{
    CATEGORICAL_HC_ROW_CAP, CATEGORICAL_TREATMENT_ARTIFACT_VERSION,
    CATEGORICAL_TREATMENT_CALIBRATION, CategoricalSpecWire, CategoricalTreatmentArtifact,
    CategoricalTreatmentArtifactError, CategoricalTreatmentRequest, decode_parts, encode_parts,
};
use antecedent_io::vector_treatment_artifact::AdjustmentColumnRequest;
use antecedent_kernels::erfc;

fn close(actual: f64, expected: f64, what: &str) {
    assert!((actual - expected).abs() <= 1e-9, "{what}: {actual} vs {expected}");
}

fn normal_p(z: f64) -> f64 {
    erfc(z.abs() / 2.0_f64.sqrt())
}

fn names(levels: &[&str]) -> Vec<String> {
    levels.iter().map(|l| (*l).to_string()).collect()
}

fn spec(levels: &[&str], ordered: bool, reference: &str) -> CategoricalSpecWire {
    CategoricalSpecWire {
        declared_levels: names(levels),
        scale: if ordered { "ordered" } else { "unordered" }.into(),
        reference: reference.into(),
        min_level_rows: 2,
        pairwise: vec![],
        monotonicity: None,
        covariance: "model_based".into(),
    }
}

fn abc_request() -> CategoricalTreatmentRequest {
    let rows = [
        ("c", 9.0),
        ("a", 1.0),
        ("b", 4.0),
        ("c", 7.0),
        ("b", 6.0),
        ("a", 3.0),
        ("c", 11.0),
        ("b", 5.0),
        ("c", 9.0),
    ];
    CategoricalTreatmentRequest {
        outcome: rows.iter().map(|r| r.1).collect(),
        row_snapshot: "snap".into(),
        adjustment: vec![],
        levels: rows.iter().map(|r| r.0.to_string()).collect(),
        spec: spec(&["a", "b", "c"], false, "a"),
    }
}

fn ordered_request() -> CategoricalTreatmentRequest {
    let mut rows: Vec<(&str, f64)> = Vec::new();
    for (level, values) in
        [("lo", [3.0, 2.0, 1.0]), ("mid", [2.0, 1.0, 0.0]), ("hi", [1.0, 0.0, -1.0])]
    {
        rows.extend(values.iter().map(|v| (level, *v)));
    }
    CategoricalTreatmentRequest {
        outcome: rows.iter().map(|r| r.1).collect(),
        row_snapshot: "snap".into(),
        adjustment: vec![],
        levels: rows.iter().map(|r| r.0.to_string()).collect(),
        spec: CategoricalSpecWire {
            monotonicity: Some("non_decreasing".into()),
            ..spec(&["lo", "mid", "hi"], true, "lo")
        },
    }
}

fn export(request: &CategoricalTreatmentRequest) -> (CategoricalTreatmentArtifact, Vec<u8>) {
    let artifact = CategoricalTreatmentArtifact::seal(request).unwrap();
    let bytes = artifact.to_bytes("b4-categorical-test").unwrap();
    (artifact, bytes)
}

#[test]
fn b4_categorical_artifact_group_mean_algebra_survives_the_container() {
    let (artifact, bytes) = export(&abc_request());
    let report = CategoricalTreatmentArtifact::from_bytes(&bytes, Some(artifact.identity()))
        .unwrap()
        .report();
    assert_eq!(report.replay, "summary");
    let result = &report.result;
    assert_eq!(result.level_order, ["a", "b", "c"]);
    assert_eq!(result.counts.iter().map(|c| c.rows).collect::<Vec<_>>(), [2, 3, 4]);
    assert_eq!(result.coefficients[0].name, "level:b");
    close(result.coefficients[0].estimate, 3.0, "b - a");
    close(result.coefficients[1].estimate, 7.0, "c - a");
    close(result.covariance[0], 5.0 / 3.0, "V_bb");
    close(result.covariance[1], 1.0, "V_bc");
    close(result.covariance[2], 1.0, "V_cb");
    close(result.covariance[3], 1.5, "V_cc");
    assert_eq!(result.level_contrasts[0].level, "b");
    assert_eq!(result.level_contrasts[0].reference, "a");
    close(result.level_contrasts[1].standard_error, 1.5_f64.sqrt(), "se c");
    assert_eq!(report.null, "every level has the reference level's effect");
    assert!(report.monotonicity_null.is_none());
    assert!(result.monotonicity.is_none());
}

#[test]
fn b4_categorical_artifact_omnibus_and_holm_family_match_closed_forms() {
    let mut request = abc_request();
    request.spec.pairwise = vec![("b".into(), "c".into())];
    let report = export(&request).0.report();
    let result = &report.result;
    let statistic = 319.0 / 9.0;
    close(result.omnibus.statistic, statistic, "omnibus");
    assert_eq!(result.omnibus.degrees_of_freedom, 2);
    close(result.omnibus.p_value, (-statistic / 2.0).exp(), "omnibus p");
    // Family: level b, level c and the requested pair b -> c (variance 7/6 off the covariance).
    assert_eq!(result.family_size, 3);
    let pair = &result.pairwise[0];
    close(pair.estimate, 4.0, "c - b");
    close(pair.standard_error.powi(2), 7.0 / 6.0, "pair variance");
    let p_b = normal_p(3.0 / (5.0_f64 / 3.0).sqrt());
    let p_c = normal_p(7.0 / 1.5_f64.sqrt());
    let p_pair = normal_p(4.0 / (7.0_f64 / 6.0).sqrt());
    // Holm step-down in ascending raw order: c, pair, b.
    let holm_c = (3.0 * p_c).min(1.0);
    let holm_pair = holm_c.max((2.0 * p_pair).min(1.0));
    let holm_b = holm_pair.max(p_b);
    close(result.level_contrasts[1].p_holm, holm_c, "Holm c");
    close(pair.p_holm, holm_pair, "Holm pair");
    close(result.level_contrasts[0].p_holm, holm_b, "Holm b");
}

#[test]
fn b4_categorical_artifact_unordered_level_permutation_gives_identical_artifacts() {
    let base = export(&abc_request());
    let mut permuted = abc_request();
    permuted.spec.declared_levels = names(&["c", "a", "b"]);
    let other = export(&permuted);
    assert_eq!(base.0.identity(), other.0.identity());
    assert_eq!(base.0.report().result, other.0.report().result);
    // The permuted artifact is consumable against the original's retained identity.
    assert!(CategoricalTreatmentArtifact::from_bytes(&other.1, Some(base.0.identity())).is_ok());
}

#[test]
fn b4_categorical_artifact_ordered_scale_is_part_of_the_estimand() {
    let ordered = {
        let mut request = abc_request();
        request.spec.scale = "ordered".into();
        CategoricalTreatmentArtifact::seal(&request).unwrap()
    };
    let unordered = CategoricalTreatmentArtifact::seal(&abc_request()).unwrap();
    assert_ne!(ordered.identity().level_scale_id, unordered.identity().level_scale_id);
    let mut reversed = abc_request();
    reversed.spec.scale = "ordered".into();
    reversed.spec.declared_levels = names(&["c", "b", "a"]);
    let reversed = CategoricalTreatmentArtifact::seal(&reversed).unwrap();
    assert_ne!(ordered.identity().level_scale_id, reversed.identity().level_scale_id);
    assert_eq!(reversed.report().result.level_order, ["c", "b", "a"]);
}

#[test]
fn b4_categorical_artifact_monotonicity_result_and_null_are_stored_and_replayed() {
    let (artifact, bytes) = export(&ordered_request());
    let report = CategoricalTreatmentArtifact::from_bytes(&bytes, Some(artifact.identity()))
        .unwrap()
        .report();
    let mono = report.result.monotonicity.as_ref().expect("a declared monotonicity test");
    assert_eq!(mono.direction, "non_decreasing");
    assert_eq!(
        mono.null,
        "every adjacent step of the adjusted level effects is non-negative (non-decreasing)"
    );
    assert_eq!(report.monotonicity_null.as_deref(), Some(mono.null.as_str()));
    assert_eq!(mono.steps.len(), 2);
    for step in &mono.steps {
        close(step.difference, -1.0, "adjacent step");
        close(step.standard_error.powi(2), 2.0 / 3.0, "step variance");
    }
    let z = -(1.5_f64).sqrt();
    close(mono.statistic, z, "oriented minimum z");
    // p = min(1, 2 Phi(z)) = erfc(sqrt(3/4)).
    close(mono.p_value, erfc(0.75_f64.sqrt()), "conservative union-intersection p");
    assert!(mono.conservative);
    assert_eq!(mono.calibration, "unmeasured");
    assert!(report.caveats[2].contains("failing to reject does not prove"), "{:?}", report.caveats);
    // The opposite direction has a different null and the reverse orientation (p = 1 here).
    let mut down = ordered_request();
    down.spec.monotonicity = Some("non_increasing".into());
    let flipped = export(&down).0.report();
    let flipped = flipped.result.monotonicity.unwrap();
    assert_eq!(flipped.direction, "non_increasing");
    assert_ne!(flipped.null, mono.null);
    close(flipped.statistic, 1.5_f64.sqrt(), "reverse orientation");
    close(flipped.p_value, 1.0, "no evidence against non-increasing");
}

#[test]
fn b4_categorical_artifact_calibration_is_unmeasured_and_says_so() {
    let report = export(&abc_request()).0.report();
    assert_eq!(report.calibration, CATEGORICAL_TREATMENT_CALIBRATION);
    assert_eq!(report.calibration, "unmeasured");
    assert_eq!(report.result.calibration, "unmeasured");
    assert_eq!(report.inference_claim, "asymptotic_wald_calibration_unmeasured");
    assert_eq!(report.version, CATEGORICAL_TREATMENT_ARTIFACT_VERSION);
    assert!(report.caveats[0].contains("calibration is unmeasured"), "{:?}", report.caveats);
}

#[test]
fn b4_categorical_artifact_sparse_absent_and_undeclared_levels_refuse_with_the_level_named() {
    let mut absent = abc_request();
    absent.spec.declared_levels = names(&["a", "b", "c", "d"]);
    let (code, detail, text) =
        CategoricalTreatmentArtifact::seal(&absent).unwrap_err().refusal().unwrap();
    assert_eq!(
        (code, detail.as_str()),
        ("arm_not_populated", "categorical_treatment.absent_level")
    );
    assert!(text.contains('d'), "{text}");

    let mut sparse = abc_request();
    sparse.spec.min_level_rows = 3;
    let (code, detail, text) =
        CategoricalTreatmentArtifact::seal(&sparse).unwrap_err().refusal().unwrap();
    assert_eq!(
        (code, detail.as_str()),
        ("arm_not_populated", "categorical_treatment.sparse_level")
    );
    assert!(text.contains("`a`"), "{text}");

    let mut undeclared = abc_request();
    undeclared.levels[0] = "zzz".into();
    let (code, detail, text) =
        CategoricalTreatmentArtifact::seal(&undeclared).unwrap_err().refusal().unwrap();
    assert_eq!(
        (code, detail.as_str()),
        ("invalid_argument", "categorical_treatment.undeclared_level")
    );
    assert!(text.contains("zzz"), "{text}");

    let mut reference = abc_request();
    reference.spec.reference = "q".into();
    let (_, detail, _) =
        CategoricalTreatmentArtifact::seal(&reference).unwrap_err().refusal().unwrap();
    assert_eq!(detail, "categorical_treatment.unknown_reference");

    let mut unordered_monotone = abc_request();
    unordered_monotone.spec.monotonicity = Some("non_decreasing".into());
    let (code, detail, _) =
        CategoricalTreatmentArtifact::seal(&unordered_monotone).unwrap_err().refusal().unwrap();
    assert_eq!(
        (code, detail.as_str()),
        ("route_not_supported", "categorical_treatment.monotonicity_requires_ordered")
    );

    let mut bad_kind = abc_request();
    bad_kind.spec.covariance = "cluster".into();
    let (_, detail, _) =
        CategoricalTreatmentArtifact::seal(&bad_kind).unwrap_err().refusal().unwrap();
    assert_eq!(detail, "categorical_treatment.invalid_request");
}

#[test]
fn b4_categorical_artifact_robust_covariance_replays_from_the_rows_and_has_a_cap() {
    let mut hc = abc_request();
    hc.spec.covariance = "hc1".into();
    let (artifact, bytes) = export(&hc);
    assert_eq!(artifact.report().replay, "rows");
    let consumed =
        CategoricalTreatmentArtifact::from_bytes(&bytes, Some(artifact.identity())).unwrap();
    assert_eq!(consumed.report(), artifact.report());
    assert_eq!(consumed.report().result.covariance_kind, "hc1");
    assert_ne!(
        consumed.report().result.covariance,
        CategoricalTreatmentArtifact::seal(&abc_request()).unwrap().report().result.covariance
    );

    let n = CATEGORICAL_HC_ROW_CAP + 1;
    let big = CategoricalTreatmentRequest {
        outcome: (0..n).map(|i| (i % 5) as f64).collect(),
        row_snapshot: "snap".into(),
        adjustment: vec![],
        levels: (0..n).map(|i| ["a", "b", "c"][i % 3].to_string()).collect(),
        spec: CategoricalSpecWire {
            covariance: "hc0".into(),
            ..spec(&["a", "b", "c"], false, "a")
        },
    };
    let (code, detail, _) =
        CategoricalTreatmentArtifact::seal(&big).unwrap_err().refusal().unwrap();
    assert_eq!(code, "route_not_supported");
    assert_eq!(detail, "categorical_treatment.hc_replay_row_cap_exceeded");
    let mut model = big;
    model.spec.covariance = "model_based".into();
    assert_eq!(CategoricalTreatmentArtifact::seal(&model).unwrap().report().replay, "summary");
}

#[test]
fn b4_categorical_artifact_adjustment_columns_enter_the_design_and_the_identity() {
    let mut adjusted = abc_request();
    adjusted.adjustment = vec![AdjustmentColumnRequest {
        name: "z".into(),
        values: vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
    }];
    let (artifact, bytes) = export(&adjusted);
    assert_eq!(artifact.meta().adjustment, ["z"]);
    assert!(CategoricalTreatmentArtifact::from_bytes(&bytes, Some(artifact.identity())).is_ok());
    let plain = CategoricalTreatmentArtifact::seal(&abc_request()).unwrap();
    assert_ne!(artifact.identity().adjustment_set_id, plain.identity().adjustment_set_id);
}

/// Seal `mutated` as a consumer-consistent artifact and consume it against the identity of
/// `original`: the mutation is resealed (its own digests are correct) but must still refuse.
fn resealed_mutation_refuses(
    original: &CategoricalTreatmentRequest,
    mutated: &CategoricalTreatmentRequest,
    field: &'static str,
) {
    let (artifact, _) = export(original);
    let (_, mutated_bytes) = export(mutated);
    assert!(CategoricalTreatmentArtifact::from_bytes(&mutated_bytes, None).is_ok(), "{field}");
    let error = CategoricalTreatmentArtifact::from_bytes(&mutated_bytes, Some(artifact.identity()))
        .unwrap_err();
    assert_eq!(error, CategoricalTreatmentArtifactError::IdentityMismatch { field }, "{field}");
    let (code, detail, _) = error.refusal().unwrap();
    assert_eq!(code, "route_not_supported");
    assert_eq!(detail, "categorical_treatment.wrong_contract");
}

#[test]
fn b4_categorical_artifact_resealed_identity_changes_are_refused() {
    let original = ordered_request();
    let mut snapshot = original.clone();
    snapshot.row_snapshot = "snap-2".into();
    resealed_mutation_refuses(&original, &snapshot, "row_snapshot");

    let mut reference = original.clone();
    reference.spec.reference = "mid".into();
    resealed_mutation_refuses(&original, &reference, "level_scale");

    let mut order = original.clone();
    order.spec.declared_levels = names(&["hi", "mid", "lo"]);
    order.spec.reference = "lo".into();
    resealed_mutation_refuses(&original, &order, "level_scale");

    let mut data = original.clone();
    data.outcome[0] = 3.5;
    resealed_mutation_refuses(&original, &data, "design");

    let mut kind = original.clone();
    kind.spec.covariance = "hc0".into();
    resealed_mutation_refuses(&original, &kind, "design");

    let mut pairs = original.clone();
    pairs.spec.pairwise = vec![("lo".into(), "hi".into())];
    resealed_mutation_refuses(&original, &pairs, "family");

    let mut direction = original.clone();
    direction.spec.monotonicity = Some("non_increasing".into());
    resealed_mutation_refuses(&original, &direction, "family");

    let mut minimum = original.clone();
    minimum.spec.min_level_rows = 3;
    resealed_mutation_refuses(&original, &minimum, "family");

    let mut dropped_test = original.clone();
    dropped_test.spec.monotonicity = None;
    resealed_mutation_refuses(&original, &dropped_test, "family");
}

#[test]
fn b4_categorical_artifact_unresealed_tampering_does_not_replay() {
    let (artifact, bytes) = export(&ordered_request());
    let (meta, numbers) = decode_parts(&bytes).unwrap();

    let mut forged = meta.clone();
    forged.result.level_contrasts[0].estimate += 1.0;
    let reencoded = encode_parts(&forged, &numbers, "forged").unwrap();
    assert!(matches!(
        CategoricalTreatmentArtifact::from_bytes(&reencoded, None).unwrap_err(),
        CategoricalTreatmentArtifactError::ResultMismatch(_)
    ));

    let mut forged = meta.clone();
    forged.result.monotonicity.as_mut().unwrap().p_value = 0.001;
    let reencoded = encode_parts(&forged, &numbers, "forged").unwrap();
    assert!(matches!(
        CategoricalTreatmentArtifact::from_bytes(&reencoded, None).unwrap_err(),
        CategoricalTreatmentArtifactError::ResultMismatch(_)
    ));

    let mut forged = meta.clone();
    forged.result.family_size += 1;
    let reencoded = encode_parts(&forged, &numbers, "forged").unwrap();
    assert!(matches!(
        CategoricalTreatmentArtifact::from_bytes(&reencoded, None).unwrap_err(),
        CategoricalTreatmentArtifactError::ResultMismatch(_)
    ));

    // A stored monotonicity null that is not the declared direction's.
    let mut forged = meta.clone();
    forged.monotonicity_null = Some("every adjacent step is zero".into());
    let reencoded = encode_parts(&forged, &numbers, "forged").unwrap();
    assert_eq!(
        CategoricalTreatmentArtifact::from_bytes(&reencoded, None).unwrap_err(),
        CategoricalTreatmentArtifactError::IdentityMismatch { field: "monotonicity_null" }
    );
    let mut forged = meta.clone();
    forged.null = "no level differs".into();
    let reencoded = encode_parts(&forged, &numbers, "forged").unwrap();
    assert_eq!(
        CategoricalTreatmentArtifact::from_bytes(&reencoded, None).unwrap_err(),
        CategoricalTreatmentArtifactError::IdentityMismatch { field: "null" }
    );

    // Level counts that disagree with the embedded Gram matrix.
    let mut forged = meta.clone();
    forged.counts[0].rows += 1;
    let reencoded = encode_parts(&forged, &numbers, "forged").unwrap();
    let error = CategoricalTreatmentArtifact::from_bytes(&reencoded, None).unwrap_err();
    assert_eq!(error.refusal().unwrap().1, "categorical_treatment.summary_inconsistent");

    // A changed residual sum of squares (the last embedded number) under the stored identity.
    let mut changed = numbers.clone();
    let last = changed.len() - 8;
    changed[last..].copy_from_slice(&9.0_f64.to_le_bytes());
    let reencoded = encode_parts(&meta, &changed, "forged").unwrap();
    assert_eq!(
        CategoricalTreatmentArtifact::from_bytes(&reencoded, None).unwrap_err(),
        CategoricalTreatmentArtifactError::IdentityMismatch { field: "design" }
    );

    let mut forged = meta.clone();
    forged.calibration = "calibrated".into();
    let reencoded = encode_parts(&forged, &numbers, "forged").unwrap();
    assert_eq!(
        CategoricalTreatmentArtifact::from_bytes(&reencoded, None).unwrap_err(),
        CategoricalTreatmentArtifactError::UnsupportedSemantics("calibration")
    );
    let mut forged = meta.clone();
    forged.caveats.pop();
    let reencoded = encode_parts(&forged, &numbers, "forged").unwrap();
    assert_eq!(
        CategoricalTreatmentArtifact::from_bytes(&reencoded, None).unwrap_err(),
        CategoricalTreatmentArtifactError::UnsupportedSemantics("caveats")
    );
    assert!(CategoricalTreatmentArtifact::from_bytes(&bytes, Some(artifact.identity())).is_ok());
}

#[test]
fn b4_categorical_artifact_unknown_major_version_and_corrupt_bytes_refuse() {
    let (_, bytes) = export(&abc_request());
    let (meta, numbers) = decode_parts(&bytes).unwrap();
    for version in [0_u16, 2, 7] {
        let mut other = meta.clone();
        other.version = version;
        let reencoded = encode_parts(&other, &numbers, "other-version").unwrap();
        assert_eq!(
            CategoricalTreatmentArtifact::from_bytes(&reencoded, None).unwrap_err(),
            CategoricalTreatmentArtifactError::UnsupportedVersion { version }
        );
    }
    let mut other = meta;
    other.feature = "categorical_treatment_v2".into();
    let reencoded = encode_parts(&other, &numbers, "other-feature").unwrap();
    assert_eq!(
        CategoricalTreatmentArtifact::from_bytes(&reencoded, None).unwrap_err(),
        CategoricalTreatmentArtifactError::UnsupportedSemantics("feature marker")
    );
    let mut corrupt = bytes.clone();
    let middle = corrupt.len() / 2;
    corrupt[middle] ^= 0xff;
    assert!(CategoricalTreatmentArtifact::from_bytes(&corrupt, None).is_err());
    assert!(CategoricalTreatmentArtifact::from_bytes(&bytes[..bytes.len() - 5], None).is_err());
    assert!(CategoricalTreatmentArtifact::from_bytes(&[], None).is_err());
}
