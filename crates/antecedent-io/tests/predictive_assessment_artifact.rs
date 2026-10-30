//! Predictive assessment artifact: a fresh consumer re-scores every case from the
//! stored laws and observations, and refuses resealed edits of a law, an outcome, a
//! declaration, an identity or a stored value; unknown versions and claims; bounds.

use antecedent_core::ExecutionContext;
use antecedent_io::predictive_assessment_artifact::{
    PREDICTIVE_ASSESSMENT_ARTIFACT_VERSION, PredictiveAssessmentArtifact,
    PredictiveAssessmentArtifactError, PredictiveAssessmentConsumeLimits, decode_parts_for_test,
    encode_parts_for_test,
};
use antecedent_validate::predictive_assessment::{
    DependenceDeclaration, FitDeclaration, ForecastLaw, Horizon, LawSource, ObservationKey,
    Outcome, OverlapStatus, PredictiveAssessmentSpec, PredictiveTarget, PredictorLaws,
    ProviderIdentity, ProviderTrustLabel, RealizedObservations, ScoreKind, ValidationDesign,
};

type Error = PredictiveAssessmentArtifactError;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(3)
}

fn key(i: u64) -> ObservationKey {
    ObservationKey::from_key("artifact-test", &i.to_le_bytes())
}

fn target() -> PredictiveTarget {
    PredictiveTarget::new("outcome", Horizon::Contemporaneous, "cohort", "mmHg")
}

fn laws(n: usize, mean: f64, sd: f64) -> Vec<ForecastLaw> {
    (0..n).map(|i| ForecastLaw::gaussian(mean + 0.01 * i as f64, sd)).collect()
}

fn predictor(name: &str, laws: Vec<ForecastLaw>) -> PredictorLaws {
    PredictorLaws {
        name: name.to_owned(),
        source: LawSource::Native { model: name.to_owned() },
        target: target(),
        laws,
        fit: FitDeclaration::NotFitted,
    }
}

fn spec(n: usize) -> PredictiveAssessmentSpec {
    let outcomes: Vec<Outcome> =
        (0..n).map(|i| Outcome::Real(((i * 37 % 11) as f64 - 5.0) / 3.0)).collect();
    let mut spec = PredictiveAssessmentSpec::new(
        RealizedObservations { target: target(), ids: (0..n as u64).map(key).collect(), outcomes },
        vec![predictor("a", laws(n, 0.0, 1.0)), predictor("b", laws(n, 0.5, 1.5))],
        ValidationDesign::HeldOutSplit,
        vec![ScoreKind::Log, ScoreKind::Crps],
    );
    spec.dependence = Some(DependenceDeclaration::Iid);
    spec
}

fn sealed(spec: &PredictiveAssessmentSpec) -> (PredictiveAssessmentArtifact, Vec<u8>) {
    let artifact = PredictiveAssessmentArtifact::seal(spec, &ctx()).unwrap();
    let bytes = artifact.to_bytes("predictive-assessment-test").unwrap();
    (artifact, bytes)
}

fn first_f64(bytes: &[u8], at_word: usize) -> f64 {
    let mut word = [0_u8; 8];
    word.copy_from_slice(&bytes[at_word * 8..(at_word + 1) * 8]);
    f64::from_le_bytes(word)
}

fn put_f64(bytes: &mut [u8], at_word: usize, value: f64) {
    bytes[at_word * 8..(at_word + 1) * 8].copy_from_slice(&value.to_le_bytes());
}

#[test]
fn a_fresh_consumer_reproduces_every_score_and_the_identity() {
    let (artifact, bytes) = sealed(&spec(40));
    let consumed =
        PredictiveAssessmentArtifact::from_bytes(&bytes, Some(artifact.identity()), &ctx())
            .unwrap();
    assert_eq!(consumed.report(), artifact.report());
    assert_eq!(consumed.identity(), artifact.identity());
    // The artifact states the closed claims it carries.
    let meta = consumed.meta();
    assert_eq!(meta.inference_claim, "point_only");
    assert_eq!(meta.calibration, "unmeasured");
    assert_eq!(meta.causal_validity, "not_established");
    assert_eq!(meta.interval, "not_reported");
    // And the numbers are the ones the library computes from the laws.
    let report = consumed.report();
    assert_eq!(report.predictors.len(), 2);
    assert!(report.predictors[0].scores.iter().all(|s| s.mean.is_finite()));
}

#[test]
fn listing_the_cases_in_another_order_seals_the_same_assessment() {
    let base = spec(30);
    let mut shuffled = base.clone();
    shuffled.observations.ids.reverse();
    shuffled.observations.outcomes.reverse();
    for predictor in &mut shuffled.predictors {
        predictor.laws.reverse();
    }
    let (a, a_bytes) = sealed(&base);
    let (b, b_bytes) = sealed(&shuffled);
    assert_eq!(a.identity(), b.identity());
    assert_eq!(a_bytes, b_bytes);
    PredictiveAssessmentArtifact::from_bytes(&b_bytes, Some(a.identity()), &ctx()).unwrap();
}

#[test]
fn a_resealed_edit_of_a_stored_law_is_refused_by_recomputation() {
    let (artifact, bytes) = sealed(&spec(20));
    let (meta, mut numbers, keys) = decode_parts_for_test(&bytes).unwrap();
    // Words: 20 outcomes, then predictor "a" case 0: [code 0, mean, sd].
    let sd_word = 20 + 2;
    assert!((first_f64(&numbers, sd_word) - 1.0).abs() < 1e-12);
    put_f64(&mut numbers, sd_word, 2.0);
    let forged = encode_parts_for_test(&meta, &numbers, &keys, "forged").unwrap();
    // No retained identity needed: the stored identity itself no longer matches.
    let error = PredictiveAssessmentArtifact::from_bytes(&forged, None, &ctx()).unwrap_err();
    assert_eq!(error, Error::IdentityMismatch { field: "predictor_laws" });
    let (code, detail, _) = error.refusal().unwrap();
    assert_eq!(code, "route_not_supported");
    assert_eq!(detail, "predictive_assessment.wrong_contract");
    // The unforged bytes still consume.
    PredictiveAssessmentArtifact::from_bytes(&bytes, Some(artifact.identity()), &ctx()).unwrap();
}

#[test]
fn a_resealed_edit_of_a_realized_outcome_is_refused() {
    let (_, bytes) = sealed(&spec(20));
    let (meta, mut numbers, keys) = decode_parts_for_test(&bytes).unwrap();
    let before = first_f64(&numbers, 3);
    put_f64(&mut numbers, 3, before + 0.5);
    let forged = encode_parts_for_test(&meta, &numbers, &keys, "forged").unwrap();
    let error = PredictiveAssessmentArtifact::from_bytes(&forged, None, &ctx()).unwrap_err();
    assert_eq!(error, Error::IdentityMismatch { field: "observations" });
}

#[test]
fn a_resealed_edit_of_a_stored_score_is_refused() {
    let (artifact, bytes) = sealed(&spec(20));
    let (meta, mut numbers, keys) = decode_parts_for_test(&bytes).unwrap();
    // The stored result values are the tail of the numbers section; the first is the
    // first predictor's first score.
    let words = numbers.len() / 8;
    let first_result = words - meta.result_labels.len();
    assert!(meta.result_labels[0].starts_with("scores.a."));
    let stored = first_f64(&numbers, first_result);
    put_f64(&mut numbers, first_result, stored + 1e-6);
    let forged = encode_parts_for_test(&meta, &numbers, &keys, "forged").unwrap();
    let error =
        PredictiveAssessmentArtifact::from_bytes(&forged, Some(artifact.identity()), &ctx())
            .unwrap_err();
    assert_eq!(error, Error::ResultMismatch("scores"));
    let (code, detail, _) = error.refusal().unwrap();
    assert_eq!(code, "invalid_argument");
    assert_eq!(detail, "predictive_assessment.report_replay_mismatch");
}

#[test]
fn a_consistently_resealed_assessment_of_other_laws_fails_the_retained_identity() {
    let (original, _) = sealed(&spec(20));
    let mut other = spec(20);
    other.predictors[1].laws[4] = ForecastLaw::gaussian(0.0, 1.25);
    let (_, other_bytes) = sealed(&other);
    // Internally consistent, so it consumes without an expected identity ...
    PredictiveAssessmentArtifact::from_bytes(&other_bytes, None, &ctx()).unwrap();
    // ... and is refused by a consumer that kept the original's identity.
    let error =
        PredictiveAssessmentArtifact::from_bytes(&other_bytes, Some(original.identity()), &ctx())
            .unwrap_err();
    assert_eq!(error, Error::IdentityMismatch { field: "predictor_laws" });
}

#[test]
fn an_edited_stored_identity_is_refused() {
    let (_, bytes) = sealed(&spec(10));
    let (mut meta, numbers, keys) = decode_parts_for_test(&bytes).unwrap();
    meta.identity.configuration = "0".repeat(64);
    let forged = encode_parts_for_test(&meta, &numbers, &keys, "forged").unwrap();
    let error = PredictiveAssessmentArtifact::from_bytes(&forged, None, &ctx()).unwrap_err();
    assert_eq!(error, Error::IdentityMismatch { field: "configuration" });
}

#[test]
fn unknown_versions_and_unsupported_claims_are_refused_before_replay() {
    let (_, bytes) = sealed(&spec(10));
    let (meta, numbers, keys) = decode_parts_for_test(&bytes).unwrap();

    let mut future = meta.clone();
    future.version = PREDICTIVE_ASSESSMENT_ARTIFACT_VERSION + 1;
    let forged = encode_parts_for_test(&future, &numbers, &keys, "forged").unwrap();
    assert!(matches!(
        PredictiveAssessmentArtifact::from_bytes(&forged, None, &ctx()),
        Err(Error::UnsupportedVersion { version: 3 })
    ));

    for (edit, expected) in [
        ("calibration", "calibration"),
        ("inference_claim", "inference claim"),
        ("causal_validity", "causal validity"),
        ("interval", "interval"),
        ("caveats", "caveats"),
    ] {
        let mut claimed = meta.clone();
        match edit {
            "calibration" => claimed.calibration = "measured".into(),
            "inference_claim" => claimed.inference_claim = "interval".into(),
            "causal_validity" => claimed.causal_validity = "established".into(),
            "interval" => claimed.interval = "reported".into(),
            _ => claimed.caveats.clear(),
        }
        let forged = encode_parts_for_test(&claimed, &numbers, &keys, "forged").unwrap();
        assert_eq!(
            PredictiveAssessmentArtifact::from_bytes(&forged, None, &ctx()).unwrap_err(),
            Error::UnsupportedSemantics(expected)
        );
    }
}

#[test]
fn consumer_limits_and_corruption_are_refused_without_replay() {
    let (_, bytes) = sealed(&spec(25));
    let tight = PredictiveAssessmentConsumeLimits { max_cases: 10, ..Default::default() };
    assert_eq!(
        PredictiveAssessmentArtifact::from_bytes_with_limits(&bytes, None, tight, &ctx())
            .unwrap_err(),
        Error::LimitsExceeded("cases")
    );
    let few_predictors =
        PredictiveAssessmentConsumeLimits { max_predictors: 1, ..Default::default() };
    assert_eq!(
        PredictiveAssessmentArtifact::from_bytes_with_limits(&bytes, None, few_predictors, &ctx())
            .unwrap_err(),
        Error::LimitsExceeded("predictors")
    );
    let truncated = &bytes[..bytes.len() / 2];
    assert_eq!(
        PredictiveAssessmentArtifact::from_bytes(truncated, None, &ctx()).unwrap_err(),
        Error::Decode("io error: failed to fill whole buffer".into())
    );
    let mut flipped = bytes.clone();
    let middle = flipped.len() / 2;
    flipped[middle] ^= 0xff;
    // A flipped byte fails the container checksum of the section it lands in.
    let error = PredictiveAssessmentArtifact::from_bytes(&flipped, None, &ctx()).unwrap_err();
    assert!(
        format!("{error:?}")
            .contains("ChecksumMismatch { section: \"predictive_assessment_meta\" }"),
        "{error:?}"
    );
}

#[test]
fn the_declared_fit_overlap_and_a_provider_source_survive_the_round_trip() {
    let n = 12;
    let mut base = spec(n);
    base.predictors[0].fit = FitDeclaration::FittedOn((8..30).map(key).collect());
    base.predictors[1].source = LawSource::Provider {
        identity: ProviderIdentity {
            provider_id: "svc".into(),
            object_id: "law".into(),
            version_id: "3".into(),
            snapshot_id: "s".into(),
            request_id: "r".into(),
        },
        trust: ProviderTrustLabel::ExternallyAttested,
    };
    base.predictors[1].fit = FitDeclaration::Undeclared { reason: "no declaration".into() };
    let (artifact, bytes) = sealed(&base);
    let consumed =
        PredictiveAssessmentArtifact::from_bytes(&bytes, Some(artifact.identity()), &ctx())
            .unwrap();
    let report = consumed.report();
    assert_eq!(report.predictors[0].overlap, OverlapStatus::InSample { shared: 4, assessed: 12 });
    assert_eq!(report.predictors[1].overlap.label(), "unverified");
    assert_eq!(report.predictors[1].source, "provider");
}

#[test]
fn rolling_origin_with_serial_dependence_round_trips() {
    let n = 18;
    let mut base = spec(n);
    let one_step = PredictiveTarget::new("outcome", Horizon::Steps(1), "cohort", "mmHg");
    base.observations.target = one_step.clone();
    for predictor in &mut base.predictors {
        predictor.target = one_step.clone();
    }
    base.design = ValidationDesign::RollingOrigin {
        origins: (0..n as u64).map(|i| i / 3).collect(),
        horizon_steps: 1,
    };
    base.dependence = Some(DependenceDeclaration::Serial {
        time_index: (0..n as u64).collect(),
        truncation_lag: 2,
    });
    let (artifact, bytes) = sealed(&base);
    let consumed =
        PredictiveAssessmentArtifact::from_bytes(&bytes, Some(artifact.identity()), &ctx())
            .unwrap();
    assert_eq!(consumed.report().differences[0].dependence, "serial");
    // 324 case/lag units plus 124 name-byte units = 448.
    let limits = PredictiveAssessmentConsumeLimits::default();
    let at_limit = PredictiveAssessmentArtifact::from_bytes_with_replay_operations(
        &bytes,
        Some(artifact.identity()),
        limits,
        448,
        &ctx(),
    )
    .unwrap();
    assert_eq!(at_limit.report(), artifact.report());
    let error = PredictiveAssessmentArtifact::from_bytes_with_replay_operations(
        &bytes,
        None,
        limits,
        447,
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(error, Error::LimitsExceeded("comparison work"));
    // Case order is a time order here, so the PIT drift between the halves is reported.
    let drift = consumed.report().predictors[0].diagnostics.drift.as_ref().unwrap();
    assert_eq!((drift.reference_cases, drift.comparison_cases), (9, 9));
    assert_eq!(drift.label, "diagnostic");
    assert_eq!(consumed.report().predictors[0].scores[0].folds.len(), 6);
}

#[test]
fn a_refused_assessment_does_not_seal() {
    let mut callback = spec(5);
    callback.predictors[0].source = LawSource::Callback { description: "closure".into() };
    let error = PredictiveAssessmentArtifact::seal(&callback, &ctx()).unwrap_err();
    let (code, detail, _) = error.refusal().unwrap();
    assert_eq!(code, "route_not_supported");
    assert_eq!(detail, "predictive_assessment.law_callback_not_built");

    let mut iid = spec(6);
    iid.design = ValidationDesign::Blocked { blocks: vec![0, 0, 1, 1, 2, 2] };
    let error = PredictiveAssessmentArtifact::seal(&iid, &ctx()).unwrap_err();
    let (code, detail, _) = error.refusal().unwrap();
    assert_eq!(code, "sampling_dependence_unknown");
    assert_eq!(detail, "predictive_assessment.iid_not_licensed_for_dependent_design");
}

#[test]
fn serial_lag_amplification_is_refused_before_assessment() {
    let n = 8_000;
    let mut base = spec(n);
    base.dependence = Some(DependenceDeclaration::Serial {
        time_index: (0..n as u64).collect(),
        truncation_lag: 2,
    });
    let (_, bytes) = sealed(&base);
    let (mut meta, numbers, keys) = decode_parts_for_test(&bytes).unwrap();
    // A valid declared lag, still below the horizon, but requiring more than
    // the default 50 million serial terms across the two score differences.
    meta.dependence.truncation_lag = Some((n - 1) as u32);
    let forged = encode_parts_for_test(&meta, &numbers, &keys, "expensive-lag").unwrap();
    let error = PredictiveAssessmentArtifact::from_bytes(&forged, None, &ctx()).unwrap_err();
    assert_eq!(error, Error::LimitsExceeded("comparison work"));
}

#[test]
fn comparison_replay_is_bounded_and_duplicate_pairs_are_refused() {
    let (artifact, bytes) = sealed(&spec(20));
    let limits = PredictiveAssessmentConsumeLimits::default();
    // One implicit pair, two scores, seven linear passes per case, plus name bytes.
    let replayed = PredictiveAssessmentArtifact::from_bytes_with_replay_operations(
        &bytes, Some(artifact.identity()), limits, 404, &ctx()).unwrap();
    assert_eq!(replayed.report(), artifact.report());
    assert_eq!(PredictiveAssessmentArtifact::from_bytes_with_replay_operations(
        &bytes, None, limits, 403, &ctx()).unwrap_err(), Error::LimitsExceeded("comparison work"));
    let (mut meta, numbers, keys) = decode_parts_for_test(&bytes).unwrap();
    meta.comparisons = vec![("a".into(), "b".into()), ("a".into(), "b".into())];
    let forged = encode_parts_for_test(&meta, &numbers, &keys, "duplicates").unwrap();
    let error = PredictiveAssessmentArtifact::from_bytes(&forged, None, &ctx()).unwrap_err();
    assert_eq!(error.refusal().unwrap().1, "predictive_assessment.invalid_law");
    meta.comparisons = vec![("a".into(), "b".into()); 1000];
    let forged = encode_parts_for_test(&meta, &numbers, &keys, "amplification").unwrap();
    assert_eq!(PredictiveAssessmentArtifact::from_bytes(&forged, None, &ctx()).unwrap_err(),
        Error::LimitsExceeded("comparison work"));
    let mut ordered = spec(20);
    ordered.comparisons = vec![("b".into(), "a".into()), ("a".into(), "b".into())];
    let (artifact, bytes) = sealed(&ordered);
    let replayed = PredictiveAssessmentArtifact::from_bytes(&bytes, Some(artifact.identity()), &ctx()).unwrap();
    assert_eq!(replayed.report().differences.iter().map(|d| (d.candidate.as_str(), d.reference.as_str())).collect::<Vec<_>>(),
        vec![("b", "a"), ("b", "a"), ("a", "b"), ("a", "b")]);
}

#[test]
fn comparison_replay_also_charges_repeated_fit_key_sorts() {
    let mut assessment = spec(20);
    for predictor in &mut assessment.predictors {
        predictor.fit = FitDeclaration::FittedOn((1000..1100).map(key).collect());
    }
    let (artifact, bytes) = sealed(&assessment);
    let limits = PredictiveAssessmentConsumeLimits::default();
    // 280 paired-score units, 124 name-byte units, plus 1800 fit-key units.
    let accepted = PredictiveAssessmentArtifact::from_bytes_with_replay_operations(
        &bytes, Some(artifact.identity()), limits, 2204, &ctx()).unwrap();
    assert_eq!(accepted.report(), artifact.report());
    assert_eq!(PredictiveAssessmentArtifact::from_bytes_with_replay_operations(
        &bytes, None, limits, 2203, &ctx()).unwrap_err(), Error::LimitsExceeded("comparison work"));
}

#[test]
fn unselected_predictor_fit_keys_do_not_inflate_sparse_comparison_budget() {
    let mut assessment = spec(20);
    let mut unselected = predictor("unselected", laws(20, 1.0, 1.0));
    unselected.fit = FitDeclaration::FittedOn((1000..1100).map(key).collect());
    assessment.predictors.push(unselected);
    assessment.comparisons = vec![("a".into(), "b".into())];
    let (artifact, bytes) = sealed(&assessment);
    let limits = PredictiveAssessmentConsumeLimits::default();
    // 280 case units, six lookups and 778 name-byte units; no fit-overlap
    // work is charged for the unselected predictor.
    let accepted = PredictiveAssessmentArtifact::from_bytes_with_replay_operations(
        &bytes, Some(artifact.identity()), limits, 1064, &ctx()).unwrap();
    assert_eq!(accepted.report(), artifact.report());
    assert_eq!(PredictiveAssessmentArtifact::from_bytes_with_replay_operations(
        &bytes, None, limits, 1063, &ctx()).unwrap_err(), Error::LimitsExceeded("comparison work"));
}


#[test]
fn long_common_prefix_names_consume_byte_budget_before_comparison_replay() {
    let mut assessment = spec(20);
    let prefix = "q".repeat(16_384);
    let a = format!("{prefix}a");
    let b = format!("{prefix}b");
    assessment.predictors[0].name = a.clone();
    assessment.predictors[1].name = b.clone();
    assessment.comparisons = vec![(a, b)];
    let (artifact, bytes) = sealed(&assessment);
    let limits = PredictiveAssessmentConsumeLimits::default();
    assert_eq!(PredictiveAssessmentArtifact::from_bytes_with_replay_operations(
        &bytes, None, limits, 1_000_000, &ctx()).unwrap_err(), Error::LimitsExceeded("comparison work"));
    let replay = PredictiveAssessmentArtifact::from_bytes(&bytes, Some(artifact.identity()), &ctx()).unwrap();
    assert_eq!(replay.report(), artifact.report());
}


fn legacy_bytes(artifact: &PredictiveAssessmentArtifact, bytes: &[u8]) -> Vec<u8> {
    let (mut meta, numbers, keys) = decode_parts_for_test(bytes).unwrap();
    meta.version = 1;
    // The legacy identity did not bind provenance.
    let legacy = antecedent_validate::predictive_assessment::legacy_assessment_identity(artifact.spec()).unwrap();
    meta.identity.observations = legacy.observations;
    meta.identity.predictors = legacy.predictors;
    meta.identity.design = legacy.design;
    meta.identity.configuration = legacy.configuration;
    meta.identity.digest = legacy.digest;
    let keys: Vec<u8> = keys.chunks_exact(33).flat_map(|key| key[..32].iter().copied()).collect();
    encode_parts_for_test(&meta, &numbers, &keys, "legacy").unwrap()
}

#[test]
fn key_provenance_round_trips_and_is_bound_to_identity() {
    let mut base = spec(20);
    base.observations.ids[0] = ObservationKey::from_bytes(*base.observations.ids[0].as_bytes());
    let ledger_fit: Vec<_> = (100..104).map(|i| ObservationKey::from_bytes(*key(i).as_bytes())).collect();
    base.predictors[0].fit = FitDeclaration::FittedOn(ledger_fit.clone());
    let duplicate = ObservationKey::from_caller_minted_bytes(*ledger_fit[0].as_bytes());
    base.predictors[1].fit = FitDeclaration::FittedOn(vec![ledger_fit[0], duplicate]);
    let (artifact, bytes) = sealed(&base);
    let native = antecedent_validate::predictive_assessment::assess_predictive_laws(artifact.spec(), &ctx()).unwrap();
    assert_eq!(artifact.identity(), &native.identity);
    let replay = PredictiveAssessmentArtifact::from_bytes(&bytes, Some(artifact.identity()), &ctx()).unwrap();
    assert_eq!(replay.report(), artifact.report());
    assert!(replay.spec().observations.ids.iter().any(ObservationKey::is_caller_minted));
    assert!(replay.spec().observations.ids.iter().any(|key| !key.is_caller_minted()));
    match &replay.spec().predictors[1].fit {
        FitDeclaration::FittedOn(keys) => { assert_eq!(keys.len(), 1); assert!(keys[0].is_caller_minted()); }
        _ => panic!("expected fitted keys"),
    }
    assert!(matches!(replay.report().predictors[1].overlap, OverlapStatus::Unverified { .. }));
    // A coherently resealed claim with the same digest bytes but upgraded
    // provenance cannot match the independently retained identity.
    let mut upgraded = base.clone();
    upgraded.observations.ids = upgraded.observations.ids.iter().map(|key| ObservationKey::from_bytes(*key.as_bytes())).collect();
    for predictor in &mut upgraded.predictors {
        if let FitDeclaration::FittedOn(keys) = &mut predictor.fit {
            *keys = keys.iter().map(|key| ObservationKey::from_bytes(*key.as_bytes())).collect();
        }
    }
    let (upgraded, forged_bytes) = sealed(&upgraded);
    assert!(matches!(upgraded.report().predictors[1].overlap, OverlapStatus::HeldOutVerified { .. }));
    assert!(matches!(PredictiveAssessmentArtifact::from_bytes(&forged_bytes, Some(artifact.identity()), &ctx()), Err(Error::IdentityMismatch { .. })));
    let (meta, numbers, mut keys) = decode_parts_for_test(&bytes).unwrap();
    keys[32] ^= 1;
    let forged = encode_parts_for_test(&meta, &numbers, &keys, "changed-flag").unwrap();
    assert!(matches!(PredictiveAssessmentArtifact::from_bytes(&forged, Some(artifact.identity()), &ctx()), Err(Error::IdentityMismatch { .. })));
    keys[32] ^= 1;
    let first_fit_flag = 20 * 33 + 32;
    keys[first_fit_flag] ^= 1;
    let forged = encode_parts_for_test(&meta, &numbers, &keys, "changed-fit-flag").unwrap();
    assert!(matches!(PredictiveAssessmentArtifact::from_bytes(&forged, Some(artifact.identity()), &ctx()), Err(Error::IdentityMismatch { field: "predictor_laws" })));
    keys[first_fit_flag] ^= 1;
    keys[32] = 2;
    let forged = encode_parts_for_test(&meta, &numbers, &keys, "bad-flag").unwrap();
    assert!(matches!(PredictiveAssessmentArtifact::from_bytes(&forged, None, &ctx()), Err(Error::Malformed(_))));
}

#[test]
fn legacy_keys_are_untrusted_and_heldout_labels_are_refused() {
    let (unfitted, bytes) = sealed(&spec(20));
    let legacy = legacy_bytes(&unfitted, &bytes);
    let decoded = PredictiveAssessmentArtifact::from_bytes(&legacy, None, &ctx()).unwrap();
    assert!(decoded.spec().observations.ids.iter().all(ObservationKey::is_caller_minted));
    let reencoded = decoded.to_bytes("legacy-reexport").unwrap();
    assert_eq!(decode_parts_for_test(&reencoded).unwrap().0.version, 1);
    PredictiveAssessmentArtifact::from_bytes(&reencoded, None, &ctx()).unwrap();
    let mut insample = spec(20);
    insample.predictors[0].fit = FitDeclaration::FittedOn(vec![insample.observations.ids[0]]);
    let (insample, bytes) = sealed(&insample);
    let legacy = legacy_bytes(&insample, &bytes);
    let replay = PredictiveAssessmentArtifact::from_bytes(&legacy, None, &ctx()).unwrap();
    assert!(matches!(replay.report().predictors[0].overlap, OverlapStatus::InSample { .. }));

    let mut ledger = spec(20);
    ledger.observations.ids = ledger.observations.ids.iter().map(|key| ObservationKey::from_bytes(*key.as_bytes())).collect();
    ledger.predictors[0].fit = FitDeclaration::FittedOn((100..104).map(|i| ObservationKey::from_bytes(*key(i).as_bytes())).collect());
    let (heldout, bytes) = sealed(&ledger);
    assert!(matches!(heldout.report().predictors[0].overlap, OverlapStatus::HeldOutVerified { .. }));
    let legacy = legacy_bytes(&heldout, &bytes);
    assert_eq!(PredictiveAssessmentArtifact::from_bytes(&legacy, None, &ctx()).unwrap_err(), Error::UnsupportedSemantics("legacy held-out provenance"));
}


#[test]
fn artifact_version_must_match_the_key_section_layout() {
    let (_, bytes) = sealed(&spec(20));
    let (mut meta, numbers, v2_keys) = decode_parts_for_test(&bytes).unwrap();
    let legacy_keys: Vec<u8> = v2_keys.chunks_exact(33).flat_map(|key| key[..32].iter().copied()).collect();
    // Both versions are supported, but neither may interpret the other's stride.
    let bad_v2 = encode_parts_for_test(&meta, &numbers, &legacy_keys, "v2-legacy-keys").unwrap();
    assert!(matches!(PredictiveAssessmentArtifact::from_bytes(&bad_v2, None, &ctx()), Err(Error::Malformed(_))));
    meta.version = 1;
    let bad_v1 = encode_parts_for_test(&meta, &numbers, &v2_keys, "v1-v2-keys").unwrap();
    assert!(matches!(PredictiveAssessmentArtifact::from_bytes(&bad_v1, None, &ctx()), Err(Error::Malformed(_))));
}
