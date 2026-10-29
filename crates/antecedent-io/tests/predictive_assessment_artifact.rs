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
        Err(Error::UnsupportedVersion { version: 2 })
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
        format!("{error:?}").contains("ChecksumMismatch { section: \"predictive_assessment_meta\" }"),
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
