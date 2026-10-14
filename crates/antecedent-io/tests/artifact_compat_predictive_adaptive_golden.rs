//! Golden fixtures for the predictive assessment artifact (`predictive_assessment_v1`)
//! and the adaptive inference result artifact (`adaptive_inference_result_v1`).
//!
//! Each golden lives at `tests/fixtures/<family>_v1.bin`. A loader consumes it as a fresh
//! consumer (re-scoring every case, or replaying the monitoring contract and result,
//! rather than trusting the stored numbers) under the identity a producer derives today
//! from the same declaration, and requires the stored claims to stay closed. A missing
//! golden fails with the command that writes it. A changed golden is a compatibility
//! decision and needs a migration note.
//!
//! Generate or refresh both with
//! `cargo test -p antecedent-io --test artifact_compat_predictive_adaptive_golden -- --ignored write_`
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::path::PathBuf;

use antecedent_contract::adaptive_design::{
    AdaptiveDesignContract, AdaptiveDesignDeclaration, AssignmentFamily, DelayMechanism,
    InterimAnalysis, MonitoringStrategy, OutcomeAvailability, OutcomeStructure, PolicyEpoch,
    ProcedureCell, StoppingRuleDeclaration, StoppingRuleKind, StoppingSpec,
};
use antecedent_core::{ExecutionContext, SemanticDigest};
use antecedent_io::adaptive_inference_result_artifact::{
    ADAPTIVE_INFERENCE_RESULT_ARTIFACT_FEATURE, ADAPTIVE_INFERENCE_RESULT_ARTIFACT_VERSION,
    AdaptiveInferenceResultArtifact, AdaptiveInferenceResultArtifactError, AdaptiveResultSpec,
};
use antecedent_io::predictive_assessment_artifact::{
    PREDICTIVE_ASSESSMENT_ARTIFACT_VERSION, PredictiveAssessmentArtifact,
    PredictiveAssessmentArtifactError, decode_parts_for_test, encode_parts_for_test,
};
use antecedent_test_support::read_fixture_bytes;
use antecedent_validate::predictive_assessment::{
    DependenceDeclaration, FitDeclaration, ForecastLaw, Horizon, LawSource, ObservationKey,
    Outcome, PredictiveAssessmentSpec, PredictiveTarget, PredictorLaws, RealizedObservations,
    ScoreKind, ValidationDesign,
};

const GENERATOR: &str = "cargo test -p antecedent-io --test artifact_compat_predictive_adaptive_golden -- --ignored write_";
const PREDICTIVE_GOLDEN: &str = "predictive_assessment_v1.bin";
const ADAPTIVE_GOLDEN: &str = "adaptive_inference_result_v1.bin";

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn load(name: &str) -> Vec<u8> {
    read_fixture_bytes(&fixture_dir(), name, &format!("generate it with `{GENERATOR}`"))
}

fn write(name: &str, bytes: &[u8]) {
    let dir = fixture_dir();
    std::fs::create_dir_all(&dir).expect("create the fixture directory");
    std::fs::write(dir.join(name), bytes).expect("write the golden");
}

// -- predictive assessment ---------------------------------------------------------------------

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(3)
}

fn target() -> PredictiveTarget {
    PredictiveTarget::new("outcome", Horizon::Contemporaneous, "cohort", "mmHg")
}

fn predictor(name: &str, mean: f64, sd: f64, n: usize) -> PredictorLaws {
    PredictorLaws {
        name: name.to_owned(),
        source: LawSource::Native { model: name.to_owned() },
        target: target(),
        laws: (0..n).map(|i| ForecastLaw::gaussian(mean + 0.01 * i as f64, sd)).collect(),
        fit: FitDeclaration::NotFitted,
    }
}

fn assessment_spec() -> PredictiveAssessmentSpec {
    let n = 20_usize;
    let outcomes: Vec<Outcome> =
        (0..n).map(|i| Outcome::Real(((i * 37 % 11) as f64 - 5.0) / 3.0)).collect();
    let ids = (0..n as u64)
        .map(|i| ObservationKey::from_key("artifact-golden", &i.to_le_bytes()))
        .collect();
    let mut spec = PredictiveAssessmentSpec::new(
        RealizedObservations { target: target(), ids, outcomes },
        vec![predictor("a", 0.0, 1.0, n), predictor("b", 0.5, 1.5, n)],
        ValidationDesign::HeldOutSplit,
        vec![ScoreKind::Log, ScoreKind::Crps],
    );
    spec.dependence = Some(DependenceDeclaration::Iid);
    spec
}

#[test]
#[ignore = "writes tests/fixtures; run with `-- --ignored write_`"]
fn write_predictive_assessment_golden() {
    let artifact = PredictiveAssessmentArtifact::seal(&assessment_spec(), &ctx()).expect("seals");
    write(PREDICTIVE_GOLDEN, &artifact.to_bytes("golden-predictive-assessment").expect("bytes"));
}

#[test]
fn the_predictive_assessment_golden_is_rescored_and_gated() {
    let bytes = load(PREDICTIVE_GOLDEN);
    let fresh = PredictiveAssessmentArtifact::seal(&assessment_spec(), &ctx()).expect("seals");
    let consumed = PredictiveAssessmentArtifact::from_bytes(&bytes, Some(fresh.identity()), &ctx())
        .expect("a fresh consumer re-scores every case");
    assert_eq!(consumed.identity(), fresh.identity());
    assert_eq!(consumed.report(), fresh.report());
    let meta = consumed.meta();
    assert_eq!(meta.version, PREDICTIVE_ASSESSMENT_ARTIFACT_VERSION);
    assert_eq!(meta.inference_claim, "point_only");
    assert_eq!(meta.calibration, "unmeasured");
    assert_eq!(meta.interval, "not_reported");

    let (parts_meta, numbers, keys) = decode_parts_for_test(&bytes).expect("decodes");
    let mut newer = parts_meta;
    newer.version = PREDICTIVE_ASSESSMENT_ARTIFACT_VERSION + 1;
    let forged = encode_parts_for_test(&newer, &numbers, &keys, "newer").expect("encodes");
    assert!(matches!(
        PredictiveAssessmentArtifact::from_bytes(&forged, None, &ctx()),
        Err(PredictiveAssessmentArtifactError::UnsupportedVersion { version: 2 })
    ));
}

// -- adaptive inference result -----------------------------------------------------------------

fn adaptive_ctx() -> ExecutionContext {
    ExecutionContext::for_tests(11)
}

/// Twelve units alternating arms; arm 0 always fails, arm 1 always succeeds.
fn adaptive_contract() -> AdaptiveDesignContract {
    let assigned_arm: Vec<u32> = (0..12).map(|unit| unit % 2).collect();
    AdaptiveDesignContract::new(AdaptiveDesignDeclaration {
        label: "thompson_trial".to_owned(),
        cell: ProcedureCell {
            assignment: AssignmentFamily::ThompsonBetaBernoulli,
            outcome: OutcomeStructure::SingleOutcome,
            monitoring: MonitoringStrategy::FixedInterimLooks,
            stopping: StoppingRuleKind::Efficacy,
        },
        arm_labels: vec!["a".to_owned(), "b".to_owned()],
        assigned_arm: assigned_arm.clone(),
        policy_history: Some(vec![PolicyEpoch {
            first_unit: 0,
            policy_identity: SemanticDigest::from_bytes([3; 32]),
        }]),
        assignment_probabilities: Some(vec![vec![0.5, 0.5]; 12]),
        exposure: Some(assigned_arm),
        availability: Some(OutcomeAvailability {
            available_at_look: (0..12).map(|unit| Some(u32::from(unit >= 6))).collect(),
            delay: DelayMechanism::Immediate,
        }),
        outcomes: (0..12).map(|unit| vec![f64::from(unit % 2)]).collect(),
        interim_analyses: Some(vec![
            InterimAnalysis { units_enrolled: 6, information_fraction: 0.5 },
            InterimAnalysis { units_enrolled: 12, information_fraction: 1.0 },
        ]),
        stopping_rule: Some(StoppingRuleDeclaration {
            kind: StoppingRuleKind::Efficacy,
            spec: StoppingSpec::PosteriorProbabilityBest { threshold: 0.95 },
            ended_by_rule: false,
        }),
        probability_floor: 0.05,
    })
    .expect("complete contract")
}

fn posterior() -> AdaptiveResultSpec {
    AdaptiveResultSpec::BayesianPosterior { look: 1, prior_alpha: 1.0, prior_beta: 1.0 }
}

#[test]
#[ignore = "writes tests/fixtures; run with `-- --ignored write_`"]
fn write_adaptive_inference_result_golden() {
    let bytes = AdaptiveInferenceResultArtifact::to_bytes(
        &adaptive_contract(),
        &posterior(),
        "golden-adaptive-inference-result",
        &adaptive_ctx(),
    )
    .expect("bytes");
    write(ADAPTIVE_GOLDEN, &bytes);
}

#[test]
fn the_adaptive_inference_result_golden_is_replayed_and_gated() {
    type Artifact = AdaptiveInferenceResultArtifact;
    type Error = AdaptiveInferenceResultArtifactError;
    let bytes = load(ADAPTIVE_GOLDEN);
    let (fresh, _) =
        Artifact::seal(&adaptive_contract(), &posterior(), &adaptive_ctx()).expect("seals");
    let consumed = Artifact::from_bytes(&bytes, Some(fresh.identity()), &adaptive_ctx())
        .expect("a fresh consumer replays the stored result");
    assert_eq!(consumed.identity(), fresh.identity());
    let stored: Vec<u64> = consumed.result().values.iter().map(|v| v.to_bits()).collect();
    let current: Vec<u64> = fresh.result().values.iter().map(|v| v.to_bits()).collect();
    assert_eq!(stored, current);

    let meta = Artifact::decode_unchecked(&bytes).expect("inspectable");
    assert_eq!(meta.version, ADAPTIVE_INFERENCE_RESULT_ARTIFACT_VERSION);
    assert_eq!(meta.feature, ADAPTIVE_INFERENCE_RESULT_ARTIFACT_FEATURE);
    assert_eq!(meta.claim, "point_only");
    assert_eq!(meta.calibration, "unmeasured");

    let mut newer = meta;
    newer.version = ADAPTIVE_INFERENCE_RESULT_ARTIFACT_VERSION + 1;
    Artifact::reseal_identity(&mut newer);
    assert_eq!(
        Artifact::from_meta(&newer, None, &adaptive_ctx()).unwrap_err(),
        Error::UnsupportedVersion { version: 2 }
    );
}
