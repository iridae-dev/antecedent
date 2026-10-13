//! `adaptive_monitoring_contract_v1`: a fresh consumer sees the procedure, the
//! stopping rule and the paradigm, and refuses semantic mutation.
//!
//! Every edit below is *resealed*: the stored identity is recomputed after the
//! edit, so only the consumer's replay of the contract can refuse it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::adaptive_design::{
    AdaptiveDesignContract, AdaptiveDesignDeclaration, AssignmentFamily, BoundarySidedness,
    DelayMechanism, InferenceParadigm, InterimAnalysis, MonitoringStrategy, OutcomeAvailability,
    OutcomeStructure, PolicyEpoch, ProcedureCell, SpendingKind, StoppingRuleDeclaration,
    StoppingRuleKind, StoppingSpec,
};
use antecedent_core::SemanticDigest;
use antecedent_io_core::adaptive_monitoring_contract_artifact::{
    ADAPTIVE_MONITORING_CONTRACT_FEATURE, AdaptiveMonitoringContractArtifact,
    AdaptiveMonitoringContractError,
};

type Artifact = AdaptiveMonitoringContractArtifact;
type Error = AdaptiveMonitoringContractError;

fn group_sequential() -> AdaptiveDesignDeclaration {
    let cell = ProcedureCell {
        assignment: AssignmentFamily::FixedRandomization,
        outcome: OutcomeStructure::SingleOutcome,
        monitoring: MonitoringStrategy::GroupSequential,
        stopping: StoppingRuleKind::Efficacy,
    };
    let assigned_arm: Vec<u32> = (0..12).map(|unit| unit % 2).collect();
    AdaptiveDesignDeclaration {
        label: "group_sequential_trial".to_owned(),
        cell,
        arm_labels: vec!["control".to_owned(), "treatment".to_owned()],
        assigned_arm: assigned_arm.clone(),
        policy_history: Some(vec![PolicyEpoch {
            first_unit: 0,
            policy_identity: SemanticDigest::from_bytes([7; 32]),
        }]),
        assignment_probabilities: Some(vec![vec![0.5, 0.5]; 12]),
        exposure: Some(assigned_arm),
        availability: Some(OutcomeAvailability {
            available_at_look: (0..12).map(|unit| Some(u32::from(unit >= 6))).collect(),
            delay: DelayMechanism::Immediate,
        }),
        outcomes: (0..12).map(|unit| vec![f64::from(unit % 5) * 0.5]).collect(),
        interim_analyses: Some(vec![
            InterimAnalysis { units_enrolled: 6, information_fraction: 0.5 },
            InterimAnalysis { units_enrolled: 12, information_fraction: 1.0 },
        ]),
        stopping_rule: Some(StoppingRuleDeclaration {
            kind: StoppingRuleKind::Efficacy,
            spec: StoppingSpec::AlphaSpending {
                spending: SpendingKind::OBrienFlemingType,
                total_alpha: 0.05,
                sidedness: BoundarySidedness::TwoSidedSymmetric,
            },
            ended_by_rule: false,
        }),
        probability_floor: 0.05,
    }
}

fn thompson() -> AdaptiveDesignDeclaration {
    let mut declaration = group_sequential();
    declaration.cell = ProcedureCell {
        assignment: AssignmentFamily::ThompsonBetaBernoulli,
        outcome: OutcomeStructure::SingleOutcome,
        monitoring: MonitoringStrategy::FixedInterimLooks,
        stopping: StoppingRuleKind::Efficacy,
    };
    declaration.outcomes = (0..12).map(|unit| vec![f64::from(unit % 2)]).collect();
    declaration.stopping_rule = Some(StoppingRuleDeclaration {
        kind: StoppingRuleKind::Efficacy,
        spec: StoppingSpec::PosteriorProbabilityBest { threshold: 0.95 },
        ended_by_rule: true,
    });
    declaration
}

fn contract(declaration: AdaptiveDesignDeclaration) -> AdaptiveDesignContract {
    AdaptiveDesignContract::new(declaration).expect("complete contract")
}

fn bytes(contract: &AdaptiveDesignContract, paradigm: InferenceParadigm) -> Vec<u8> {
    Artifact::to_bytes(contract, paradigm, "adaptive-monitoring").expect("encodes")
}

#[test]
fn a_fresh_consumer_sees_the_procedure_the_rule_and_the_paradigm() {
    let source = contract(group_sequential());
    let encoded = bytes(&source, InferenceParadigm::FrequentistGroupSequential);
    let consumed = Artifact::from_bytes(&encoded, None).expect("consumes");
    assert_eq!(consumed.paradigm(), InferenceParadigm::FrequentistGroupSequential);
    assert_eq!(
        consumed.procedure().identifier(),
        "fixed_randomization/single_outcome/group_sequential/efficacy"
    );
    let rule = consumed.stopping_rule().expect("the rule travels with the contract");
    assert_eq!(rule.kind, StoppingRuleKind::Efficacy);
    assert_eq!(
        rule.spec,
        StoppingSpec::AlphaSpending {
            spending: SpendingKind::OBrienFlemingType,
            total_alpha: 0.05,
            sidedness: BoundarySidedness::TwoSidedSymmetric,
        }
    );
    assert_eq!(consumed.contract().identity(), source.identity());
    // Two looks at information 1/2 and 1: independent grid recursion gives
    // 2.771807648699362 and 1.979311378353442.
    let boundaries = consumed.boundaries().expect("alpha spending has boundaries");
    assert!((boundaries.critical[0] - 2.771_807_648_7).abs() < 5e-5);
    assert!((boundaries.critical[1] - 1.979_311_378_4).abs() < 5e-5);
    // The artifact is deterministic.
    assert_eq!(bytes(&source, InferenceParadigm::FrequentistGroupSequential), encoded);
}

#[test]
fn a_posterior_rule_has_no_boundaries_and_the_looks_are_accounted() {
    let source = contract(thompson());
    let (artifact, meta) =
        Artifact::seal(&source, InferenceParadigm::BayesianPosterior).expect("seals");
    assert!(artifact.boundaries().is_none());
    // Looks analyse 6 and 12 units: 18 slots, 12 distinct, 6 counted twice.
    assert_eq!(meta.evidence.look_sizes, vec![6, 12]);
    assert_eq!(meta.evidence.declared_slots, 18);
    assert_eq!(meta.evidence.distinct_observations, 12);
    assert_eq!(meta.evidence.repeated_slots, 6);
    assert_eq!(
        meta.evidence.independent_pooling_detail.as_deref(),
        Some("adaptive_inference.looks_share_observations")
    );
    assert_eq!(meta.feature, ADAPTIVE_MONITORING_CONTRACT_FEATURE);
}

fn edited(
    paradigm: InferenceParadigm,
    declaration: AdaptiveDesignDeclaration,
    edit: impl FnOnce(
        &mut antecedent_io_core::adaptive_monitoring_contract_artifact::MonitoringContractMeta,
    ),
) -> Result<Artifact, Error> {
    let encoded = bytes(&contract(declaration), paradigm);
    let mut meta = Artifact::decode_unchecked(&encoded).expect("decodes");
    edit(&mut meta);
    Artifact::reseal_identity(&mut meta);
    Artifact::from_meta(&meta, None)
}

#[test]
fn resealed_semantic_mutations_are_refused() {
    use InferenceParadigm as P;
    // Another stopping rule: the stored contract identity no longer matches.
    let outcome = edited(P::FrequentistGroupSequential, group_sequential(), |meta| {
        let rule = meta.declaration.stopping_rule.as_mut().unwrap();
        rule.parameter_bits = Some(0.01_f64.to_bits());
    });
    assert!(matches!(outcome, Err(Error::IdentityMismatch { field: "contract_identity" })));
    // Another realized assignment probability.
    let outcome = edited(P::FrequentistGroupSequential, group_sequential(), |meta| {
        meta.declaration.assignment_probability_bits.as_mut().unwrap()[3] =
            vec![0.45_f64.to_bits(), 0.55_f64.to_bits()];
    });
    assert!(matches!(outcome, Err(Error::Refused { .. }) | Err(Error::IdentityMismatch { .. })));
    // Another outcome.
    let outcome = edited(P::FrequentistGroupSequential, group_sequential(), |meta| {
        meta.declaration.outcome_bits[0] = vec![9.0_f64.to_bits()];
    });
    assert!(matches!(outcome, Err(Error::IdentityMismatch { field: "contract_identity" })));
    // A stored boundary that does not replay.
    let outcome = edited(P::FrequentistGroupSequential, group_sequential(), |meta| {
        meta.boundaries.as_mut().unwrap().critical_bits[0] = 2.5_f64.to_bits();
    });
    assert_eq!(outcome.unwrap_err(), Error::ReplayMismatch("boundaries"));
    // Boundaries claimed for a rule that spends none.
    let outcome = edited(P::BayesianPosterior, thompson(), |meta| {
        meta.boundaries =
            Some(antecedent_io_core::adaptive_monitoring_contract_artifact::BoundaryWire {
                critical_bits: vec![2.0_f64.to_bits()],
                spend_increment_bits: vec![0.05_f64.to_bits()],
            });
    });
    assert_eq!(outcome.unwrap_err(), Error::ReplayMismatch("boundaries"));
    // Accounting that double counts nothing.
    let outcome = edited(P::BayesianPosterior, thompson(), |meta| {
        meta.evidence.repeated_slots = 0;
        meta.evidence.declared_slots = 12;
    });
    assert_eq!(outcome.unwrap_err(), Error::ReplayMismatch("observation accounting"));
    // The pooling refusal dropped from the accounting.
    let outcome = edited(P::BayesianPosterior, thompson(), |meta| {
        meta.evidence.independent_pooling_detail = None;
    });
    assert_eq!(outcome.unwrap_err(), Error::ReplayMismatch("observation accounting"));
}

#[test]
fn a_paradigm_the_design_may_not_use_is_refused_on_consumption() {
    use InferenceParadigm as P;
    // Fixed-sample inference of an adaptive design.
    let outcome = edited(P::FrequentistGroupSequential, group_sequential(), |meta| {
        meta.paradigm = P::FrequentistFixedSample.as_str().to_owned();
    });
    match outcome.unwrap_err() {
        Error::Refused { code, detail, .. } => {
            assert_eq!(code, "adaptive_design_ignored");
            assert_eq!(detail, "adaptive_inference.fixed_sample_on_adaptive_data");
        }
        other => panic!("{other}"),
    }
    // A paradigm of another cell.
    let outcome = edited(P::BayesianPosterior, thompson(), |meta| {
        meta.paradigm = P::FrequentistAdaptiveWeighting.as_str().to_owned();
    });
    match outcome.unwrap_err() {
        Error::Refused { code, detail, .. } => {
            assert_eq!(code, "cell_not_licensed");
            assert_eq!(detail, "adaptive_inference.paradigm_not_licensed_for_cell");
        }
        other => panic!("{other}"),
    }
    let outcome = edited(P::BayesianPosterior, thompson(), |meta| {
        meta.paradigm = "posterior_of_something".to_owned();
    });
    assert!(matches!(outcome, Err(Error::UnsupportedSemantics(_))));
    // Sealing refuses the same request up front.
    let source = contract(group_sequential());
    assert!(Artifact::seal(&source, P::FrequentistFixedSample).is_err());
    assert!(Artifact::seal(&source, P::BayesianPosterior).is_err());
}

#[test]
fn a_removed_contract_element_is_refused_on_consumption() {
    let outcome = edited(InferenceParadigm::BayesianPosterior, thompson(), |meta| {
        meta.declaration.policy_history = None;
        meta.declaration.stopping_rule = None;
    });
    match outcome.unwrap_err() {
        Error::Refused { code, detail, message } => {
            assert_eq!(code, "adaptive_contract_incomplete");
            assert_eq!(detail, "adaptive_inference.policy_history_missing");
            assert!(message.contains("adaptive_inference.stopping_rule_missing"));
        }
        other => panic!("{other}"),
    }
}

#[test]
fn identity_version_corruption_and_retained_identity_are_checked() {
    let source = contract(thompson());
    let encoded = bytes(&source, InferenceParadigm::BayesianPosterior);
    let consumed = Artifact::from_bytes(&encoded, None).unwrap();
    // The retained identity of this artifact is accepted; another is not.
    assert!(Artifact::from_bytes(&encoded, Some(consumed.identity())).is_ok());
    let wrong = "0".repeat(64);
    assert!(matches!(
        Artifact::from_bytes(&encoded, Some(&wrong)),
        Err(Error::IdentityMismatch { field: "retained_identity" })
    ));
    // A consistently resealed contract of another experiment is refused.
    let mut other = thompson();
    other.label = "another_trial".to_owned();
    let other_bytes = bytes(&contract(other), InferenceParadigm::BayesianPosterior);
    assert!(matches!(
        Artifact::from_bytes(&other_bytes, Some(consumed.identity())),
        Err(Error::IdentityMismatch { field: "retained_identity" })
    ));
    // Another major version is refused before the payload is interpreted.
    let mut meta = Artifact::decode_unchecked(&encoded).unwrap();
    meta.version = 2;
    Artifact::reseal_identity(&mut meta);
    assert!(matches!(
        Artifact::from_meta(&meta, None),
        Err(Error::UnsupportedVersion { version: 2 })
    ));
    // A flipped byte inside the container does not decode as this artifact.
    let mut corrupt = encoded.clone();
    let middle = corrupt.len() / 2;
    corrupt[middle] ^= 0xff;
    assert!(Artifact::from_bytes(&corrupt, None).is_err());
    // Truncation too.
    assert!(Artifact::from_bytes(&encoded[..encoded.len() / 2], None).is_err());
}

#[test]
fn refusals_carry_registered_codes_and_namespaced_details() {
    let encoded =
        bytes(&contract(group_sequential()), InferenceParadigm::FrequentistGroupSequential);
    let mut meta = Artifact::decode_unchecked(&encoded).unwrap();
    meta.boundaries.as_mut().unwrap().critical_bits[1] = 3.0_f64.to_bits();
    Artifact::reseal_identity(&mut meta);
    let error = Artifact::from_meta(&meta, None).unwrap_err();
    let (code, detail, _) = error.refusal().expect("a replay mismatch is a registered refusal");
    assert!(antecedent_core::reason_code::is_runtime_refusal(code));
    assert_eq!(detail, "adaptive_inference.report_replay_mismatch");
    let wrong = Artifact::from_bytes(&encoded, Some(&"1".repeat(64))).unwrap_err();
    let (code, detail, _) = wrong.refusal().unwrap();
    assert_eq!(code, "route_not_supported");
    assert_eq!(detail, "adaptive_inference.wrong_contract");
}
