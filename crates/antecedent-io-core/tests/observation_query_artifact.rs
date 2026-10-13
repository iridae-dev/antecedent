//! Independent replay of the observation-query artifact, and the semantic
//! mutations a consumer must refuse.
//!
//! Oracle (hand computed; see the model-crate suite for the derivation):
//!   Z ~ Bernoulli(0.5); P(S=1 | Z) = 0.2, 0.8; P(Y=1 | S, Z) = 0.1, 0.5, 0.3, 0.9
//!   keyed (S=0,Z=0), (0,1), (1,0), (1,1)
//!   P(Y=1 | S=1) = 0.78, P(Y=1 | do(S=1)) = 0.6
//!   construct C (prevalence 0.1, P(Y=1|C) = 0.1, 0.7), sensitivity 0.9,
//!   specificity 0.8: P(C=1 | M=1) = 1/3, P(Y=1 | M=1) = 0.3

use antecedent_contract::InferenceBasis;
use antecedent_core::ExecutionContext;
use antecedent_core::measurement_model::{
    ArmBinding, BinaryArmRates, ErrorStructure, MeasurementBinding, MeasurementKind,
    MeasurementModel, MeasurementParameters, MeasurementRoute, MeasurementVariable,
    ParameterEvidence, RepresentationVersion,
};
use antecedent_io_core::measurement_model_wire::model_to_wire;
use antecedent_io_core::{
    MeasuredSlotWire, ObservationQueryArtifactError, ObservationQueryArtifactWire,
    ObservationQueryWire, ObservationResultWire, observation_query_identity,
};
use antecedent_model::compose::{
    ComposedModelSpec, Domain, MechanismContext, MechanismDescriptor, MechanismForm,
    MechanismOrigin, MechanismProvenance, ParentBinding, PosteriorDependence, TabularRow,
};

const TOL: f64 = 1e-12;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(5)
}

fn near(bits: u64, expected: f64) {
    let value = f64::from_bits(bits);
    assert!((value - expected).abs() <= TOL, "{value} vs {expected}");
}

fn level_bits(levels: &[(String, u64)], name: &str) -> u64 {
    levels.iter().find(|(l, _)| l == name).map(|(_, b)| *b).unwrap()
}

fn table(
    var: &str,
    parents: &[&str],
    rows: &[(&[&str], [f64; 2])],
    origin: MechanismOrigin,
) -> MechanismDescriptor {
    let levels: Vec<String> = vec!["0".into(), "1".into()];
    MechanismDescriptor {
        variable: var.into(),
        form: MechanismForm::Tabular {
            levels: levels.clone(),
            rows: rows
                .iter()
                .map(|(config, p)| TabularRow {
                    parent_levels: parents
                        .iter()
                        .zip(config.iter())
                        .map(|(parent, l)| ((*parent).to_owned(), (*l).to_owned()))
                        .collect(),
                    probabilities: vec![("0".to_owned(), p[0]), ("1".to_owned(), p[1])],
                })
                .collect(),
        },
        parents: parents
            .iter()
            .map(|p| ParentBinding {
                variable: (*p).to_owned(),
                units: "category".into(),
                transform: "identity".into(),
                domain: Domain::Levels(levels.clone()),
            })
            .collect(),
        conditioning_set: parents.iter().map(|p| (*p).to_owned()).collect(),
        output_units: "category".into(),
        output_transform: "identity".into(),
        output_support: Domain::Levels(levels),
        context: MechanismContext { population: "pop".into(), regime: "observational".into() },
        provenance: MechanismProvenance {
            origin,
            artifact_id: format!("fit:{var}"),
            evidence_id: format!("data:{var}"),
        },
        latents: Vec::new(),
        uncertainty: None,
    }
}

fn spec_of(mechanisms: Vec<MechanismDescriptor>) -> ComposedModelSpec {
    let mut edges = Vec::new();
    for m in &mechanisms {
        for p in &m.parents {
            edges.push((p.variable.clone(), m.variable.clone()));
        }
    }
    ComposedModelSpec {
        target: MechanismContext { population: "pop".into(), regime: "observational".into() },
        mechanisms,
        edges,
        invariances: Vec::new(),
        selection_diagrams: Vec::new(),
        dependence: PosteriorDependence::Unknown,
        admit_declared_mechanisms: true,
    }
}

fn confounded(origin: MechanismOrigin) -> ComposedModelSpec {
    spec_of(vec![
        table("Z", &[], &[(&[], [0.5, 0.5])], origin),
        table("S", &["Z"], &[(&["0"], [0.8, 0.2]), (&["1"], [0.2, 0.8])], origin),
        table(
            "Y",
            &["S", "Z"],
            &[
                (&["0", "0"], [0.9, 0.1]),
                (&["0", "1"], [0.5, 0.5]),
                (&["1", "0"], [0.7, 0.3]),
                (&["1", "1"], [0.1, 0.9]),
            ],
            origin,
        ),
    ])
}

fn construct_spec() -> ComposedModelSpec {
    spec_of(vec![
        table("C", &[], &[(&[], [0.9, 0.1])], MechanismOrigin::Declared),
        table(
            "Y",
            &["C"],
            &[(&["0"], [0.9, 0.1]), (&["1"], [0.3, 0.7])],
            MechanismOrigin::Declared,
        ),
    ])
}

fn assay(sensitivity: f64, construct: &str) -> MeasurementModel {
    MeasurementModel {
        model_id: "assay".into(),
        route: MeasurementRoute::BinaryMisclassification,
        indicator: MeasurementVariable {
            id: "test_result".into(),
            kind: MeasurementKind::NoisyMisclassified,
            units: "category".into(),
        },
        construct: MeasurementVariable {
            id: construct.into(),
            kind: MeasurementKind::LatentScientific,
            units: "category".into(),
        },
        binding: MeasurementBinding {
            population_id: "pop".into(),
            regime_id: "observational".into(),
            arm: ArmBinding::AllArms,
        },
        structure: ErrorStructure::NonDifferential,
        parameters: MeasurementParameters::Binary {
            rates: vec![BinaryArmRates {
                arm_id: None,
                sensitivity,
                specificity: 0.8,
                sensitivity_variance: 0.0,
                specificity_variance: 0.0,
            }],
            min_youden: 0.1,
        },
        representation: RepresentationVersion { coordinate_system_id: "assay".into(), version: 1 },
        evidence: ParameterEvidence::Declared { declarer: "lab".into() },
    }
}

fn measured_slot(model: &MeasurementModel) -> MeasuredSlotWire {
    MeasuredSlotWire {
        construct: "C".into(),
        indicator_id: "test_result".into(),
        indicator_kind: "noisy_misclassified".into(),
        recorded_level: "1".into(),
        positive_level: Some("1".into()),
        representation_system: "assay".into(),
        representation_version: 1,
        model: model_to_wire(model),
    }
}

fn pair(a: &str, b: &str) -> (String, String) {
    (a.to_owned(), b.to_owned())
}

fn paired_query() -> ObservationQueryWire {
    ObservationQueryWire::ConditioningVersusIntervention {
        outcome: "Y".into(),
        assignment: vec![pair("S", "1")],
    }
}

fn reseal(wire: &mut ObservationQueryArtifactWire) {
    wire.identity =
        observation_query_identity(wire.premises.as_ref(), &wire.query, &wire.inference_basis)
            .expect("digest");
}

#[test]
fn a_sealed_paired_artifact_replays_to_the_hand_values_and_round_trips() {
    let wire = ObservationQueryArtifactWire::seal(
        Some(&confounded(MechanismOrigin::Declared)),
        paired_query(),
        InferenceBasis::DeclaredModel,
        &ctx(),
    )
    .unwrap();
    let ObservationResultWire::Paired {
        conditioning_evidence_bits,
        conditioning,
        intervention,
        max_abs_difference_bits,
    } = &wire.result
    else {
        panic!("wrong result kind");
    };
    near(*conditioning_evidence_bits, 0.5);
    near(level_bits(conditioning, "1"), 0.78);
    near(level_bits(intervention, "1"), 0.6);
    near(*max_abs_difference_bits, 0.18);
    assert_eq!(wire.inference_basis, "exact_given_declared_model");

    let bytes = wire.export().unwrap();
    let decoded = ObservationQueryArtifactWire::decode(&bytes).unwrap();
    assert_eq!(decoded, wire);
    let consumed = decoded.consume(Some(&wire.identity), &ctx()).unwrap();
    assert_eq!(consumed.operation, "conditioning_and_intervention");
    assert_eq!(consumed.result, wire.result);
}

#[test]
fn the_operation_is_part_of_the_identity() {
    let spec = confounded(MechanismOrigin::Declared);
    let seal = |query| {
        ObservationQueryArtifactWire::seal(
            Some(&spec),
            query,
            InferenceBasis::DeclaredModel,
            &ctx(),
        )
        .unwrap()
    };
    let conditioning = seal(ObservationQueryWire::ConditionalOutcome {
        outcome: "Y".into(),
        observed: vec![pair("S", "1")],
        measured: None,
    });
    let intervention = seal(ObservationQueryWire::InterventionalOutcome {
        outcome: "Y".into(),
        interventions: vec![pair("S", "1")],
    });
    let paired = seal(paired_query());
    assert_eq!(conditioning.premises, intervention.premises);
    assert_ne!(conditioning.identity, intervention.identity);
    assert_ne!(conditioning.identity, paired.identity);
    assert_ne!(intervention.identity, paired.identity);
    let ObservationResultWire::Conditional { levels: c, operation, .. } = &conditioning.result
    else {
        panic!("wrong result kind");
    };
    assert_eq!(operation, "conditioning");
    near(level_bits(c, "1"), 0.78);
    let ObservationResultWire::Interventional { levels: i, operation, .. } = &intervention.result
    else {
        panic!("wrong result kind");
    };
    assert_eq!(operation, "intervention");
    near(level_bits(i, "1"), 0.6);
}

#[test]
fn listing_order_does_not_change_the_identity() {
    let forward = ObservationQueryArtifactWire::seal(
        Some(&confounded(MechanismOrigin::Declared)),
        paired_query(),
        InferenceBasis::DeclaredModel,
        &ctx(),
    )
    .unwrap();
    let mut shuffled = confounded(MechanismOrigin::Declared);
    shuffled.mechanisms.reverse();
    shuffled.edges.reverse();
    let backward = ObservationQueryArtifactWire::seal(
        Some(&shuffled),
        paired_query(),
        InferenceBasis::DeclaredModel,
        &ctx(),
    )
    .unwrap();
    assert_eq!(forward.identity, backward.identity);
    assert_eq!(forward.export().unwrap(), backward.export().unwrap());
}

#[test]
fn the_reading_follows_the_mechanisms_origin_and_cannot_be_edited_upward() {
    let fitted = ObservationQueryArtifactWire::seal(
        Some(&confounded(MechanismOrigin::Fitted)),
        paired_query(),
        InferenceBasis::DeclaredModel,
        &ctx(),
    )
    .unwrap();
    // The caller's declaration is ignored for a composed model.
    assert_eq!(fitted.inference_basis, "point_only_sample_fitted");
    assert!(fitted.consume(None, &ctx()).is_ok());

    let mut upgraded = fitted.clone();
    upgraded.inference_basis = "exact_given_declared_model".into();
    reseal(&mut upgraded);
    assert_eq!(
        upgraded.consume(None, &ctx()).unwrap_err(),
        ObservationQueryArtifactError::IdentityMismatch
    );
}

#[test]
fn semantic_mutations_are_refused() {
    let wire = ObservationQueryArtifactWire::seal(
        Some(&confounded(MechanismOrigin::Declared)),
        paired_query(),
        InferenceBasis::DeclaredModel,
        &ctx(),
    )
    .unwrap();

    // Not resealed: identity mismatch.
    let mut edited = wire.clone();
    edited.query = ObservationQueryWire::ConditioningVersusIntervention {
        outcome: "Y".into(),
        assignment: vec![pair("S", "0")],
    };
    assert_eq!(
        edited.consume(None, &ctx()).unwrap_err(),
        ObservationQueryArtifactError::IdentityMismatch
    );

    // Resealed with another assignment: the stored result no longer replays.
    reseal(&mut edited);
    assert_eq!(
        edited.consume(None, &ctx()).unwrap_err(),
        ObservationQueryArtifactError::ResultMismatch
    );

    // Resealed with another operation: the stored result has another shape.
    let mut swapped = wire.clone();
    swapped.query = ObservationQueryWire::InterventionalOutcome {
        outcome: "Y".into(),
        interventions: vec![pair("S", "1")],
    };
    reseal(&mut swapped);
    assert_eq!(
        swapped.consume(None, &ctx()).unwrap_err(),
        ObservationQueryArtifactError::ResultMismatch
    );

    // A stored value edited without touching the identity: result mismatch.
    let mut value = wire.clone();
    if let ObservationResultWire::Paired { max_abs_difference_bits, .. } = &mut value.result {
        *max_abs_difference_bits = 0.0_f64.to_bits();
    }
    assert_eq!(
        value.consume(None, &ctx()).unwrap_err(),
        ObservationQueryArtifactError::ResultMismatch
    );

    // A caller's retained identity that differs is refused even when sealed.
    assert_eq!(
        wire.consume(Some("not-the-identity"), &ctx()).unwrap_err(),
        ObservationQueryArtifactError::IdentityMismatch
    );

    // A premises edit (a table probability) changes the replayed answer.
    let mut premises = wire.clone();
    let mut changed = confounded(MechanismOrigin::Declared);
    if let MechanismForm::Tabular { rows, .. } = &mut changed.mechanisms[2].form {
        rows[3].probabilities = vec![("0".into(), 0.5), ("1".into(), 0.5)];
    }
    premises.premises =
        Some(antecedent_io_core::MechanismCompositionPremisesWire::from_spec(&changed));
    reseal(&mut premises);
    assert_eq!(
        premises.consume(None, &ctx()).unwrap_err(),
        ObservationQueryArtifactError::ResultMismatch
    );

    // Premises given to a declared-model query, and none to a composed one.
    let mut no_premises = wire.clone();
    no_premises.premises = None;
    reseal(&mut no_premises);
    assert!(matches!(
        no_premises.consume(None, &ctx()).unwrap_err(),
        ObservationQueryArtifactError::Malformed(_)
    ));
}

#[test]
fn foreign_versions_and_features_are_refused_before_the_payload_is_read() {
    let wire = ObservationQueryArtifactWire::seal(
        None,
        ObservationQueryWire::ConjugatePosterior {
            alpha_bits: 1.0_f64.to_bits(),
            beta_bits: 1.0_f64.to_bits(),
            successes: 7,
            failures: 3,
        },
        InferenceBasis::DeclaredModel,
        &ctx(),
    )
    .unwrap();
    let mut other_feature = wire.clone();
    other_feature.required_features = vec!["something_else".into()];
    assert!(matches!(
        other_feature.consume(None, &ctx()).unwrap_err(),
        ObservationQueryArtifactError::UnsupportedSemantics(_)
    ));
    let mut other_version = wire.clone();
    other_version.version = 2;
    assert!(matches!(
        other_version.consume(None, &ctx()).unwrap_err(),
        ObservationQueryArtifactError::UnsupportedVersion { version: 2 }
    ));
    let bytes = other_version.export().unwrap();
    assert!(ObservationQueryArtifactWire::decode(&bytes).is_err());
}

#[test]
fn a_measured_signal_is_inverted_through_its_model_and_a_swapped_model_changes_the_answer() {
    let spec = construct_spec();
    let query = |model: &MeasurementModel| ObservationQueryWire::ConditionalOutcome {
        outcome: "Y".into(),
        observed: Vec::new(),
        measured: Some(measured_slot(model)),
    };
    let wire = ObservationQueryArtifactWire::seal(
        Some(&spec),
        query(&assay(0.9, "C")),
        InferenceBasis::DeclaredModel,
        &ctx(),
    )
    .unwrap();
    let ObservationResultWire::Conditional {
        evidence_probability_bits,
        levels,
        construct_posterior: Some(posterior),
        ..
    } = &wire.result
    else {
        panic!("wrong result kind");
    };
    near(*evidence_probability_bits, 0.27);
    near(level_bits(levels, "1"), 0.3);
    near(level_bits(posterior, "1"), 1.0 / 3.0);
    assert!(wire.consume(None, &ctx()).is_ok());

    // A different sensitivity, swapped in and resealed, does not replay.
    let weaker = ObservationQueryArtifactWire::seal(
        Some(&spec),
        query(&assay(0.5, "C")),
        InferenceBasis::DeclaredModel,
        &ctx(),
    )
    .unwrap();
    assert_ne!(weaker.result, wire.result);
    let mut swapped = wire.clone();
    swapped.query = weaker.query.clone();
    reseal(&mut swapped);
    assert_eq!(
        swapped.consume(None, &ctx()).unwrap_err(),
        ObservationQueryArtifactError::ResultMismatch
    );

    // A model for another construct is refused at seal time, with its detail.
    let error = ObservationQueryArtifactWire::seal(
        Some(&spec),
        query(&assay(0.9, "not_the_construct")),
        InferenceBasis::DeclaredModel,
        &ctx(),
    )
    .unwrap_err();
    match error {
        ObservationQueryArtifactError::Refused { refusals } => {
            assert_eq!(refusals[0].code, "measurement_binding_mismatch");
            assert_eq!(refusals[0].detail, "observation_query.construct_mismatch");
        }
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn an_unreachable_level_cannot_be_sealed_and_states_the_supremum() {
    let error = ObservationQueryArtifactWire::seal(
        Some(&confounded(MechanismOrigin::Declared)),
        ObservationQueryWire::ThresholdCrossing {
            outcome: "Y".into(),
            outcome_level: "1".into(),
            signal: "S".into(),
            background: Vec::new(),
            threshold_bits: 0.9_f64.to_bits(),
            direction: "raise".into(),
        },
        InferenceBasis::DeclaredModel,
        &ctx(),
    )
    .unwrap_err();
    match error {
        ObservationQueryArtifactError::Refused { refusals } => {
            assert_eq!(refusals[0].code, "evidence_cannot_reach_level");
            assert_eq!(refusals[0].detail, "observation_query.level_unreachable");
            let supremum: f64 = refusals[0].supplied.as_deref().unwrap().parse().unwrap();
            assert!((supremum - 0.78).abs() <= TOL);
        }
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn declared_bayes_queries_replay_and_a_zero_probability_observation_is_refused() {
    let query = ObservationQueryWire::FinitePosterior {
        hypotheses: vec!["ill".into(), "well".into()],
        prior_bits: vec![0.01_f64.to_bits(), 0.99_f64.to_bits()],
        observation_levels: vec!["neg".into(), "pos".into()],
        likelihood_bits: [0.1_f64, 0.9, 0.95, 0.05].iter().map(|p| p.to_bits()).collect(),
        observations: vec!["pos".into()],
    };
    let wire = ObservationQueryArtifactWire::seal(
        None,
        query.clone(),
        InferenceBasis::SampleFittedPointOnly,
        &ctx(),
    )
    .unwrap();
    assert_eq!(wire.inference_basis, "point_only_sample_fitted");
    let ObservationResultWire::FinitePosterior { posterior, evidence_probability_bits, .. } =
        &wire.result
    else {
        panic!("wrong result kind");
    };
    near(level_bits(posterior, "ill"), 2.0 / 13.0);
    near(*evidence_probability_bits, 0.0585);
    assert!(wire.consume(None, &ctx()).is_ok());
    assert_eq!(wire.query.operation(), "posterior_update");

    // Evidence the model gives probability zero.
    let impossible = ObservationQueryWire::FinitePosterior {
        hypotheses: vec!["a".into(), "b".into()],
        prior_bits: vec![0.5_f64.to_bits(), 0.5_f64.to_bits()],
        observation_levels: vec!["x".into(), "y".into()],
        likelihood_bits: [1.0_f64, 0.0, 1.0, 0.0].iter().map(|p| p.to_bits()).collect(),
        observations: vec!["y".into()],
    };
    match ObservationQueryArtifactWire::seal(
        None,
        impossible,
        InferenceBasis::DeclaredModel,
        &ctx(),
    )
    .unwrap_err()
    {
        ObservationQueryArtifactError::Refused { refusals } => {
            assert_eq!(refusals[0].detail, "observation_query.zero_probability_evidence");
        }
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn evidence_requirements_replay_with_their_exact_counts() {
    let binary = ObservationQueryArtifactWire::seal(
        None,
        ObservationQueryWire::RequiredObservationsBinarySignal {
            prior_bits: 0.01_f64.to_bits(),
            level_bits: 0.5_f64.to_bits(),
            direction: "raise".into(),
            rate_given_hypothesis_bits: 0.9_f64.to_bits(),
            rate_given_complement_bits: 0.05_f64.to_bits(),
            max_observations: 100,
        },
        InferenceBasis::DeclaredModel,
        &ctx(),
    )
    .unwrap();
    let ObservationResultWire::RequiredObservations { observations, .. } = &binary.result else {
        panic!("wrong result kind");
    };
    assert_eq!(*observations, 2);
    assert!(binary.consume(None, &ctx()).is_ok());

    let conjugate = ObservationQueryArtifactWire::seal(
        None,
        ObservationQueryWire::RequiredObservationsConjugate {
            alpha_bits: 1.0_f64.to_bits(),
            beta_bits: 1.0_f64.to_bits(),
            level_bits: 0.9_f64.to_bits(),
            direction: "raise".into(),
            max_observations: 1000,
        },
        InferenceBasis::DeclaredModel,
        &ctx(),
    )
    .unwrap();
    let ObservationResultWire::RequiredObservations { observations, value_bits, .. } =
        &conjugate.result
    else {
        panic!("wrong result kind");
    };
    assert_eq!(*observations, 8);
    near(*value_bits, 0.9);

    let factor = ObservationQueryArtifactWire::seal(
        None,
        ObservationQueryWire::RequiredBayesFactor {
            prior_bits: 0.2_f64.to_bits(),
            level_bits: 0.8_f64.to_bits(),
            direction: "raise".into(),
        },
        InferenceBasis::DeclaredModel,
        &ctx(),
    )
    .unwrap();
    let ObservationResultWire::RequiredBayesFactor { factor_bits, already_satisfied } =
        &factor.result
    else {
        panic!("wrong result kind");
    };
    near(*factor_bits, 16.0);
    assert!(!already_satisfied);
}

#[test]
fn replay_observes_cancellation() {
    let wire = ObservationQueryArtifactWire::seal(
        None,
        ObservationQueryWire::ConjugatePosterior {
            alpha_bits: 1.0_f64.to_bits(),
            beta_bits: 1.0_f64.to_bits(),
            successes: 1,
            failures: 1,
        },
        InferenceBasis::DeclaredModel,
        &ctx(),
    )
    .unwrap();
    let cancelled = ExecutionContext::for_tests(5);
    cancelled.cancellation.cancel();
    assert_eq!(
        wire.consume(None, &cancelled).unwrap_err(),
        ObservationQueryArtifactError::Cancelled
    );
}
