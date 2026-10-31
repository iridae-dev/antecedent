//! Golden fixtures for the predictive assessment artifact (`predictive_assessment_v1`)
//! and the adaptive inference result artifact (`adaptive_inference_result_v1`).
//!
//! Each golden lives at `tests/fixtures/<family>_v1.bin`. A loader consumes it as a fresh
//! consumer, re-scoring or replaying rather than trusting stored numbers, and requires
//! the stored claims to stay closed. The predictive v1 identity predates key-provenance
//! binding and is checked under its own identity; a fresh v2 artifact is checked separately. A missing
//! golden fails with the command that writes it. A changed golden is a compatibility
//! decision and needs a migration note.
//!
//! The predictive v1 golden is historical and must not be regenerated with a v2 producer.
//! The adaptive fixture generator remains available as `--ignored write_adaptive_`.
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

const GENERATOR: &str = "cargo test -p antecedent-io --test suite_01 -- --ignored artifact_compat_predictive_adaptive_golden::write_adaptive_";
const PREDICTIVE_GOLDEN: &str = "predictive_assessment_v1.bin";
const ADAPTIVE_GOLDEN: &str = "adaptive_inference_result_v1.bin";

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn load(name: &str) -> Vec<u8> {
    let missing = if name == PREDICTIVE_GOLDEN {
        "restore the historical predictive v1 fixture from version control".to_owned()
    } else {
        format!("generate it with `{GENERATOR}`")
    };
    read_fixture_bytes(&fixture_dir(), name, &missing)
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
fn the_predictive_assessment_golden_is_rescored_and_gated() {
    let bytes = load(PREDICTIVE_GOLDEN);
    let fresh = PredictiveAssessmentArtifact::seal(&assessment_spec(), &ctx()).expect("seals");
    // V1 retained only key digests; replay it under its legacy identity and then
    // require that exact identity. A v2 caller-minted identity is not v1's identity.
    let legacy = PredictiveAssessmentArtifact::from_bytes(&bytes, None, &ctx())
        .expect("the historical not-fitted fixture re-scores under its legacy identity");
    assert_eq!(legacy.meta().version, 1);
    let consumed =
        PredictiveAssessmentArtifact::from_bytes(&bytes, Some(legacy.identity()), &ctx())
            .expect("the historical identity gates the re-scored fixture");
    assert_eq!(consumed.identity(), legacy.identity());
    assert_eq!(consumed.report(), legacy.report());
    assert_eq!(consumed.report().predictors, fresh.report().predictors);
    assert!(matches!(
        PredictiveAssessmentArtifact::from_bytes(&bytes, Some(fresh.identity()), &ctx()),
        Err(PredictiveAssessmentArtifactError::IdentityMismatch { field: "observations" })
    ));
    let meta = consumed.meta();
    assert_eq!(meta.inference_claim, "point_only");
    assert_eq!(meta.calibration, "unmeasured");
    assert_eq!(meta.interval, "not_reported");

    // Current provenance-bound identity must also survive an independent encode
    // and replay with every score and diagnostic reproduced.
    let current_bytes = fresh.to_bytes("fresh-predictive-assessment").expect("bytes");
    let current =
        PredictiveAssessmentArtifact::from_bytes(&current_bytes, Some(fresh.identity()), &ctx())
            .expect("current artifact re-scores under the current identity");
    assert_eq!(current.meta().version, PREDICTIVE_ASSESSMENT_ARTIFACT_VERSION);
    assert_eq!(current.identity(), fresh.identity());
    assert_eq!(current.report(), fresh.report());

    let (parts_meta, numbers, keys) = decode_parts_for_test(&bytes).expect("decodes");
    let mut newer = parts_meta;
    newer.version = PREDICTIVE_ASSESSMENT_ARTIFACT_VERSION + 1;
    let forged = encode_parts_for_test(&newer, &numbers, &keys, "newer").expect("encodes");
    assert!(matches!(
        PredictiveAssessmentArtifact::from_bytes(&forged, None, &ctx()),
        Err(PredictiveAssessmentArtifactError::UnsupportedVersion { version })
            if version == PREDICTIVE_ASSESSMENT_ARTIFACT_VERSION + 1
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
    // The stored values and so the identity carry the producing platform's libm (the
    // posterior's probability of being best and expected loss go through exp, ln and the
    // beta functions), so the retained identity is taken from the golden itself: consume it
    // once ungated, then require the same bytes under the identity it reports. The fresh
    // values are compared within the replay tolerance rather than bit for bit.
    let first = Artifact::from_bytes(&bytes, None, &adaptive_ctx())
        .expect("a fresh consumer replays the stored result");
    let consumed = Artifact::from_bytes(&bytes, Some(first.identity()), &adaptive_ctx())
        .expect("the golden's own identity gates the same bytes");
    assert_eq!(consumed.identity(), first.identity());
    assert_eq!(consumed.result().values.len(), fresh.result().values.len());
    for (stored, current) in consumed.result().values.iter().zip(&fresh.result().values) {
        assert!((stored - current).abs() <= 1e-12 * 1.0_f64.max(stored.abs()));
    }

    // Closed form of the look-1 posterior (all twelve units: arm a failed six times, arm b
    // succeeded six times, Beta(1, 1) priors): Beta(1, 7) and Beta(7, 1), means 1/8 and 7/8,
    // variance 7 / (64 * 9), and P(b best) = 1 - 7 B(7, 8) = 1 - 1/3432.
    let variance = 7.0 / 576.0;
    let best_b = 3431.0 / 3432.0;
    let arm_a = [1.0, 7.0, 0.125, variance, 1.0 - best_b];
    let arm_b = [7.0, 1.0, 0.875, variance, best_b];
    for (stored, exact) in consumed.result().values[..5].iter().zip(arm_a) {
        assert!((stored - exact).abs() < 1e-12, "arm a: {stored} vs {exact}");
    }
    for (stored, exact) in consumed.result().values[6..11].iter().zip(arm_b) {
        assert!((stored - exact).abs() < 1e-12, "arm b: {stored} vs {exact}");
    }

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
