//! `adaptive_inference_result_v1`: a newer version, another feature marker and a stored
//! contract of another version are refused before any replay.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

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

type Artifact = AdaptiveInferenceResultArtifact;
type Error = AdaptiveInferenceResultArtifactError;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(11)
}

/// Twelve units alternating arms; arm 0 always fails, arm 1 always succeeds.
fn contract() -> AdaptiveDesignContract {
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
fn a_newer_version_and_another_feature_marker_are_refused() {
    let (_, meta) = Artifact::seal(&contract(), &posterior(), &ctx()).expect("seals");
    assert_eq!(meta.version, ADAPTIVE_INFERENCE_RESULT_ARTIFACT_VERSION);
    assert_eq!(meta.feature, ADAPTIVE_INFERENCE_RESULT_ARTIFACT_FEATURE);

    let mut newer = meta.clone();
    newer.version = ADAPTIVE_INFERENCE_RESULT_ARTIFACT_VERSION + 1;
    Artifact::reseal_identity(&mut newer);
    assert_eq!(
        Artifact::from_meta(&newer, None, &ctx()).unwrap_err(),
        Error::UnsupportedVersion { version: 2 }
    );

    let mut foreign = meta;
    foreign.feature = "adaptive_inference_result_v2".to_owned();
    Artifact::reseal_identity(&mut foreign);
    assert_eq!(
        Artifact::from_meta(&foreign, None, &ctx()).unwrap_err(),
        Error::UnsupportedSemantics("feature marker")
    );
}

#[test]
fn the_stored_metadata_of_a_sealed_result_is_readable_without_replay() {
    let bytes = Artifact::to_bytes(&contract(), &posterior(), "compat", &ctx()).expect("encodes");
    let meta = Artifact::decode_unchecked(&bytes).expect("decodes");
    assert_eq!(meta.version, ADAPTIVE_INFERENCE_RESULT_ARTIFACT_VERSION);
    assert_eq!(meta.claim, "point_only");
    assert_eq!(meta.calibration, "unmeasured");
}
