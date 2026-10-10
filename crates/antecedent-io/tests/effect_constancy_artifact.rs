//! F18 `EffectConstancy` artifact: the frozen acceptance numbers through the container, a
//! recomputing consumer, resealed-mutation refusals against a retained identity, unknown
//! versions and the unmeasured calibration coordinate.
#![allow(clippy::float_cmp, reason = "bit-identical replay is the property under test")]

use antecedent_io::effect_constancy_artifact::{
    DependenceWire, EFFECT_CONSTANCY_ARTIFACT_VERSION, EFFECT_CONSTANCY_CALIBRATION,
    EffectConstancyArtifact, EffectConstancyArtifactError, EffectConstancyRequestWire,
    EffectEstimandWire, FamilyWire, PartitionWire, decode_parts, encode_parts,
};

fn estimand() -> EffectEstimandWire {
    EffectEstimandWire {
        estimand: "ate_difference".into(),
        units: "outcome_units".into(),
        regime: "treat_vs_control".into(),
        population: "all_observed_h2".into(),
    }
}

fn part(label: &str, effect: f64, standard_error: f64) -> PartitionWire {
    PartitionWire {
        label: label.into(),
        coordinate: format!("period:{label}"),
        support: "supported".into(),
        estimand: estimand(),
        effect,
        standard_error,
    }
}

fn independent() -> DependenceWire {
    DependenceWire { kind: "independent".into(), covariance: vec![] }
}

fn all_pairs() -> FamilyWire {
    FamilyWire { kind: "all_pairs".into(), reference: None }
}

fn request(parts: Vec<PartitionWire>) -> EffectConstancyRequestWire {
    EffectConstancyRequestWire {
        partitions: parts,
        dependence: independent(),
        family: all_pairs(),
        alpha: 0.05,
    }
}

fn close(actual: f64, expected: f64, what: &str) {
    assert!((actual - expected).abs() <= 1e-9, "{what}: {actual} vs {expected}");
}

fn dependent_request() -> EffectConstancyRequestWire {
    EffectConstancyRequestWire {
        dependence: DependenceWire {
            kind: "covariance".into(),
            covariance: vec![0.25, 0.125, 0.125, 0.25],
        },
        ..request(vec![part("a", 1.0, 0.5), part("b", 2.0, 0.5)])
    }
}

fn export(request: &EffectConstancyRequestWire) -> (EffectConstancyArtifact, Vec<u8>) {
    let artifact = EffectConstancyArtifact::seal(request).unwrap();
    let bytes = artifact.to_bytes("f18-test").unwrap();
    (artifact, bytes)
}

#[test]
fn f18_artifact_frozen_acceptance_zero_difference_has_zero_statistic() {
    // Effects 1 and 1, variance 1/4 each, zero covariance.
    let (artifact, bytes) = export(&request(vec![part("p1", 1.0, 0.5), part("p2", 1.0, 0.5)]));
    let consumed = EffectConstancyArtifact::from_bytes(&bytes, Some(artifact.identity())).unwrap();
    let report = consumed.report();
    close(report.result.statistic, 0.0, "Q");
    close(report.result.p_value, 1.0, "p");
    assert_eq!(report.result.statistic_kind, "cochran_q");
    assert_eq!(report.result.degrees_of_freedom, 1);
    assert_eq!(report.result.conclusion, "not_rejected");
    assert_eq!(report.calibration, EFFECT_CONSTANCY_CALIBRATION);
    assert_eq!(report.calibration, "unmeasured");
    assert_eq!(report.inference_claim, "point_only");
    assert_eq!(report.version, EFFECT_CONSTANCY_ARTIFACT_VERSION);
    assert!(report.caveats[0].contains("does not prove"), "{:?}", report.caveats);
    close(report.result.pooled_effect.unwrap(), 1.0, "pooled");
}

#[test]
fn f18_artifact_frozen_acceptance_one_versus_two_has_statistic_two() {
    // Q = (2 - 1)^2 / (1/4 + 1/4) = 2; df 1: sf = erfc(1).
    let (artifact, bytes) = export(&request(vec![part("p1", 1.0, 0.5), part("p2", 2.0, 0.5)]));
    let report =
        EffectConstancyArtifact::from_bytes(&bytes, Some(artifact.identity())).unwrap().report();
    close(report.result.statistic, 2.0, "Q");
    close(report.result.p_value, 0.157_299_207_050_285_13, "p = erfc(1)");
    close(report.result.pooled_effect.unwrap(), 1.5, "pooled");
    assert_eq!(report.result.contrasts.len(), 1);
    close(report.result.contrasts[0].difference, -1.0, "difference");
    close(report.result.contrasts[0].p_holm, 0.157_299_207_050_285_13, "holm");
    assert_eq!(report.result.conclusion, "not_rejected");
    // Stored values are bit-identical to a fresh run.
    assert_eq!(report, artifact.report());
}

#[test]
fn f18_artifact_dependent_covariance_gives_wald_four() {
    // Var(e2 - e1) = 1/4 + 1/4 - 2/8 = 1/4, Wald = 1 / (1/4) = 4; sf = erfc(sqrt 2).
    let (artifact, bytes) = export(&dependent_request());
    let report =
        EffectConstancyArtifact::from_bytes(&bytes, Some(artifact.identity())).unwrap().report();
    assert_eq!(report.result.statistic_kind, "wald_chi_square");
    close(report.result.statistic, 4.0, "Wald");
    close(report.result.p_value, 0.045_500_263_896_358_42, "p");
    assert_eq!(report.result.conclusion, "rejected");
    assert!(report.result.pooled_effect.is_none());
    assert_eq!(report.dependence, "covariance");
}

#[test]
fn f18_artifact_three_partitions_keep_holm_values_and_canonical_order() {
    let (artifact, bytes) =
        export(&request(vec![part("c", 2.0, 1.0), part("a", 0.0, 1.0), part("b", 1.0, 1.0)]));
    let report =
        EffectConstancyArtifact::from_bytes(&bytes, Some(artifact.identity())).unwrap().report();
    assert_eq!(
        report.partitions.iter().map(|p| p.label.as_str()).collect::<Vec<_>>(),
        ["a", "b", "c"]
    );
    close(report.result.statistic, 2.0, "Q");
    close(report.result.p_value, (-1.0_f64).exp(), "p = exp(-1)");
    assert_eq!(report.result.degrees_of_freedom, 2);
    let c = &report.result.contrasts;
    close(c[0].p_holm, 0.959_000_244_373_907, "a-b holm");
    close(c[1].p_holm, 0.471_897_621_150_855_4, "a-c holm");
    close(c[2].p_holm, 0.959_000_244_373_907, "b-c holm");
}

#[test]
fn f18_artifact_identity_is_invariant_to_partition_order() {
    let parts = [part("a", 0.3, 0.4), part("b", 1.1, 0.6), part("c", -0.2, 0.5)];
    let forward = EffectConstancyArtifact::seal(&request(parts.to_vec())).unwrap();
    let reversed =
        EffectConstancyArtifact::seal(&request(parts.iter().rev().cloned().collect())).unwrap();
    assert_eq!(forward.identity(), reversed.identity());
    assert_eq!(forward.report(), reversed.report());

    let full = [[0.16, 0.05, -0.02], [0.05, 0.36, 0.03], [-0.02, 0.03, 0.25]];
    let build = |order: [usize; 3]| {
        let covariance =
            order.iter().flat_map(|&i| order.iter().map(move |&j| full[i][j])).collect();
        EffectConstancyRequestWire {
            dependence: DependenceWire { kind: "covariance".into(), covariance },
            ..request(order.iter().map(|&i| parts[i].clone()).collect())
        }
    };
    let base = EffectConstancyArtifact::seal(&build([0, 1, 2])).unwrap();
    let other = EffectConstancyArtifact::seal(&build([2, 0, 1])).unwrap();
    assert_eq!(base.identity(), other.identity());
    assert_eq!(base.report(), other.report());
}

#[test]
fn f18_artifact_incompatible_partitions_refuse_with_the_core_detail() {
    let mut changed = part("b", 1.0, 0.5);
    changed.estimand.regime = "other_regime".into();
    let error =
        EffectConstancyArtifact::seal(&request(vec![part("a", 1.0, 0.5), changed])).unwrap_err();
    let (code, detail, _) = error.refusal().expect("a typed refusal");
    assert_eq!(code, "route_not_supported");
    assert_eq!(detail, "effect_constancy.incompatible_partitions");

    let bad_covariance = EffectConstancyRequestWire {
        dependence: DependenceWire {
            kind: "covariance".into(),
            covariance: vec![0.25, 0.5, 0.5, 0.25],
        },
        ..request(vec![part("a", 1.0, 0.5), part("b", 2.0, 0.5)])
    };
    let (code, detail, _) =
        EffectConstancyArtifact::seal(&bad_covariance).unwrap_err().refusal().unwrap();
    assert_eq!(code, "invalid_argument");
    assert_eq!(detail, "effect_constancy.invalid_covariance");

    let unknown = EffectConstancyRequestWire {
        family: FamilyWire { kind: "against_reference".into(), reference: Some("zzz".into()) },
        ..request(vec![part("a", 1.0, 0.5), part("b", 2.0, 0.5)])
    };
    let (_, detail, _) = EffectConstancyArtifact::seal(&unknown).unwrap_err().refusal().unwrap();
    assert_eq!(detail, "effect_constancy.unknown_reference");

    let tag = EffectConstancyRequestWire {
        dependence: DependenceWire { kind: "bogus".into(), covariance: vec![] },
        ..request(vec![part("a", 1.0, 0.5), part("b", 2.0, 0.5)])
    };
    let (_, detail, _) = EffectConstancyArtifact::seal(&tag).unwrap_err().refusal().unwrap();
    assert_eq!(detail, "effect_constancy.invalid_request");
}

#[test]
fn f18_artifact_partition_count_is_bounded() {
    let many: Vec<PartitionWire> = (0..1025).map(|i| part(&format!("p{i:05}"), 1.0, 0.5)).collect();
    let error = EffectConstancyArtifact::seal(&request(many)).unwrap_err();
    assert_eq!(error, EffectConstancyArtifactError::LimitsExceeded("partitions"));
}

/// Seal `mutated` as a consumer-consistent artifact and consume it against the identity of
/// `original`: the mutation is resealed (its own digests are correct) but must still refuse.
fn resealed_mutation_refuses(
    original: &EffectConstancyRequestWire,
    mutated: &EffectConstancyRequestWire,
    field: &'static str,
) {
    let (artifact, _) = export(original);
    let (_, mutated_bytes) = export(mutated);
    // Alone, the mutated artifact is internally consistent ...
    assert!(EffectConstancyArtifact::from_bytes(&mutated_bytes, None).is_ok(), "{field}");
    // ... but against the identity the consumer retained it is refused.
    let error =
        EffectConstancyArtifact::from_bytes(&mutated_bytes, Some(artifact.identity())).unwrap_err();
    assert_eq!(error, EffectConstancyArtifactError::IdentityMismatch { field }, "{field}");
    let (code, detail, _) = error.refusal().unwrap();
    assert_eq!(code, "route_not_supported");
    assert_eq!(detail, "effect_constancy.wrong_contract");
}

#[test]
fn f18_artifact_resealed_partition_identity_is_refused() {
    let original = dependent_request();
    let mut mutated = original.clone();
    mutated.partitions[0].coordinate = "period:elsewhere".into();
    resealed_mutation_refuses(&original, &mutated, "partition_identity");
    let mut renamed = original.clone();
    renamed.partitions[1].label = "z".into();
    resealed_mutation_refuses(&original, &renamed, "partition_identity");
}

#[test]
fn f18_artifact_resealed_estimand_is_refused() {
    let original = dependent_request();
    let mut mutated = original.clone();
    for p in &mut mutated.partitions {
        p.estimand.units = "log_scale".into();
    }
    resealed_mutation_refuses(&original, &mutated, "estimand");
}

#[test]
fn f18_artifact_resealed_covariance_is_refused() {
    let original = dependent_request();
    let mut mutated = original.clone();
    mutated.dependence.covariance = vec![0.25, 0.0625, 0.0625, 0.25];
    resealed_mutation_refuses(&original, &mutated, "covariance");
    // Dropping the declared dependence is also a changed covariance identity.
    let mut independent_claim = original.clone();
    independent_claim.dependence = independent();
    resealed_mutation_refuses(&original, &independent_claim, "covariance");
}

#[test]
fn f18_artifact_resealed_multiplicity_family_is_refused() {
    let original = request(vec![part("a", 0.0, 1.0), part("b", 1.0, 1.0), part("c", 2.0, 1.0)]);
    let mut reference = original.clone();
    reference.family = FamilyWire { kind: "against_reference".into(), reference: Some("a".into()) };
    resealed_mutation_refuses(&original, &reference, "multiplicity_family");
    let mut other_level = original.clone();
    other_level.alpha = 0.01;
    resealed_mutation_refuses(&original, &other_level, "multiplicity_family");
}

#[test]
fn f18_artifact_resealed_evidence_is_refused() {
    let original = dependent_request();
    let mut mutated = original.clone();
    mutated.partitions[1].effect = 2.5;
    resealed_mutation_refuses(&original, &mutated, "evidence");
}

#[test]
fn f18_artifact_unresealed_tampering_does_not_replay() {
    let (artifact, bytes) = export(&dependent_request());
    let (meta, numbers) = decode_parts(&bytes).unwrap();

    // A stored statistic that the recomputation does not reproduce.
    let mut forged = meta.clone();
    forged.result.statistic += 1.0;
    let reencoded = encode_parts(&forged, &numbers, "forged").unwrap();
    let error = EffectConstancyArtifact::from_bytes(&reencoded, None).unwrap_err();
    assert!(matches!(error, EffectConstancyArtifactError::ResultMismatch(_)), "{error:?}");

    // A stored Holm value that the recomputation does not reproduce.
    let mut forged = meta.clone();
    forged.result.contrasts[0].p_holm *= 0.5;
    let reencoded = encode_parts(&forged, &numbers, "forged").unwrap();
    assert!(matches!(
        EffectConstancyArtifact::from_bytes(&reencoded, None).unwrap_err(),
        EffectConstancyArtifactError::ResultMismatch(_)
    ));

    // A changed number under the stored identity.
    let mut changed = numbers.clone();
    changed[0..8].copy_from_slice(&9.0_f64.to_le_bytes());
    let reencoded = encode_parts(&meta, &changed, "forged").unwrap();
    assert_eq!(
        EffectConstancyArtifact::from_bytes(&reencoded, None).unwrap_err(),
        EffectConstancyArtifactError::IdentityMismatch { field: "evidence" }
    );
    // A changed covariance entry (the last number) under the stored identity.
    let mut changed = numbers.clone();
    let last = changed.len() - 8;
    changed[last..].copy_from_slice(&0.5_f64.to_le_bytes());
    let reencoded = encode_parts(&meta, &changed, "forged").unwrap();
    assert!(EffectConstancyArtifact::from_bytes(&reencoded, None).is_err());

    // A changed stored multiplicity family.
    let mut forged = meta.clone();
    forged.family.kind = "against_reference".into();
    forged.family.reference = Some("a".into());
    let reencoded = encode_parts(&forged, &numbers, "forged").unwrap();
    assert_eq!(
        EffectConstancyArtifact::from_bytes(&reencoded, None).unwrap_err(),
        EffectConstancyArtifactError::IdentityMismatch { field: "multiplicity_family" }
    );

    // A changed stored null, claim, calibration coordinate or caveat.
    let mut forged = meta.clone();
    forged.null = "effect is constant".into();
    let reencoded = encode_parts(&forged, &numbers, "forged").unwrap();
    assert_eq!(
        EffectConstancyArtifact::from_bytes(&reencoded, None).unwrap_err(),
        EffectConstancyArtifactError::IdentityMismatch { field: "null" }
    );
    let mut forged = meta.clone();
    forged.calibration = "calibrated".into();
    let reencoded = encode_parts(&forged, &numbers, "forged").unwrap();
    assert_eq!(
        EffectConstancyArtifact::from_bytes(&reencoded, None).unwrap_err(),
        EffectConstancyArtifactError::UnsupportedSemantics("calibration")
    );
    let mut forged = meta.clone();
    forged.caveats.clear();
    let reencoded = encode_parts(&forged, &numbers, "forged").unwrap();
    assert_eq!(
        EffectConstancyArtifact::from_bytes(&reencoded, None).unwrap_err(),
        EffectConstancyArtifactError::UnsupportedSemantics("caveats")
    );
    // The untouched artifact still consumes.
    assert!(EffectConstancyArtifact::from_bytes(&bytes, Some(artifact.identity())).is_ok());
}

#[test]
fn f18_artifact_unknown_major_version_refuses() {
    let (_, bytes) = export(&dependent_request());
    let (meta, numbers) = decode_parts(&bytes).unwrap();
    for version in [0_u16, 2, 7] {
        let mut other = meta.clone();
        other.version = version;
        let reencoded = encode_parts(&other, &numbers, "other-version").unwrap();
        assert_eq!(
            EffectConstancyArtifact::from_bytes(&reencoded, None).unwrap_err(),
            EffectConstancyArtifactError::UnsupportedVersion { version }
        );
    }
    let mut other = meta.clone();
    other.feature = "effect_constancy_v2".into();
    let reencoded = encode_parts(&other, &numbers, "other-feature").unwrap();
    assert_eq!(
        EffectConstancyArtifact::from_bytes(&reencoded, None).unwrap_err(),
        EffectConstancyArtifactError::UnsupportedSemantics("feature marker")
    );
}

#[test]
fn f18_artifact_corrupt_and_truncated_bytes_refuse() {
    let (_, bytes) = export(&dependent_request());
    let mut corrupt = bytes.clone();
    let middle = corrupt.len() / 2;
    corrupt[middle] ^= 0xff;
    assert!(EffectConstancyArtifact::from_bytes(&corrupt, None).is_err());
    assert!(EffectConstancyArtifact::from_bytes(&bytes[..bytes.len() - 5], None).is_err());
    assert!(EffectConstancyArtifact::from_bytes(&[], None).is_err());
}
