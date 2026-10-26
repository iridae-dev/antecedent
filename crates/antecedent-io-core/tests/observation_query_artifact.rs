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
use antecedent_io_core::distribution_artifact::DrawAlignment;
use antecedent_io_core::measurement_model_wire::model_to_wire;
use antecedent_io_core::{
    MeasuredSlotWire, OBSERVATION_QUERY_GENERAL_ARTIFACT_VERSION,
    OBSERVATION_QUERY_GENERAL_FEATURE, ObservationQueryArtifactError, ObservationQueryArtifactWire,
    ObservationQueryWire, ObservationResultWire, RedrawPriorWire, RedrawTargetWire,
    RetainedMechanismWire, observation_query_identity,
};
use antecedent_model::compose::{
    ComposedModelSpec, Domain, MechanismContext, MechanismDescriptor, MechanismForm,
    MechanismOrigin, MechanismProvenance, ParentBinding, PosteriorDependence, TabularRow,
};
use antecedent_model::observation_uncertainty::MAX_REDRAWS;

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

// ------------------------------------------------------------ general queries

fn bits(value: f64) -> u64 {
    value.to_bits()
}

fn real_binding(var: &str) -> ParentBinding {
    ParentBinding {
        variable: var.into(),
        units: "u".into(),
        transform: "identity".into(),
        domain: Domain::Real,
    }
}

fn lg(var: &str, intercept: f64, sigma: f64, parents: &[(&str, f64)]) -> MechanismDescriptor {
    MechanismDescriptor {
        variable: var.into(),
        form: MechanismForm::LinearGaussian {
            intercept,
            sigma,
            coefficients: parents.iter().map(|(p, c)| ((*p).to_owned(), *c)).collect(),
        },
        parents: parents.iter().map(|(p, _)| real_binding(p)).collect(),
        conditioning_set: parents.iter().map(|(p, _)| (*p).to_owned()).collect(),
        output_units: "u".into(),
        output_transform: "identity".into(),
        output_support: Domain::Real,
        context: MechanismContext { population: "pop".into(), regime: "observational".into() },
        provenance: MechanismProvenance {
            origin: MechanismOrigin::Fitted,
            artifact_id: format!("fit:{var}"),
            evidence_id: format!("data:{var}"),
        },
        latents: Vec::new(),
        uncertainty: None,
    }
}

/// X ~ N(1, 4); M = 0.5 + 2 X + e; Y = -1 + 0.5 X + 3 M + 0.5 e.
fn gaussian_chain() -> ComposedModelSpec {
    spec_of(vec![
        lg("X", 1.0, 2.0, &[]),
        lg("M", 0.5, 1.0, &[("X", 2.0)]),
        lg("Y", -1.0, 0.5, &[("X", 0.5), ("M", 3.0)]),
    ])
}

fn gaussian_query(interventions: &[(&str, f64)], observed: &[(&str, f64)]) -> ObservationQueryWire {
    let pairs = |values: &[(&str, f64)]| {
        values.iter().map(|(n, v)| ((*n).to_owned(), bits(*v))).collect::<Vec<_>>()
    };
    ObservationQueryWire::GaussianOutcome {
        outcome: "Y".into(),
        interventions: pairs(interventions),
        observed: pairs(observed),
    }
}

#[test]
fn a_gaussian_claim_is_a_version_two_artifact_whose_bytes_a_version_one_reader_refuses() {
    let spec = gaussian_chain();
    let wire = ObservationQueryArtifactWire::seal(
        Some(&spec),
        gaussian_query(&[("X", 2.0)], &[("M", 5.0)]),
        InferenceBasis::DeclaredModel,
        &ctx(),
    )
    .unwrap();
    assert_eq!(wire.version, OBSERVATION_QUERY_GENERAL_ARTIFACT_VERSION);
    assert_eq!(wire.required_features, [OBSERVATION_QUERY_GENERAL_FEATURE]);
    let ObservationResultWire::GaussianConditional {
        operation,
        mean_bits,
        variance_bits,
        evidence_log_density_bits,
    } = &wire.result
    else {
        panic!("wrong result kind");
    };
    // do(X=2), M=5: the structural equation at X=2, M=5 -> N(15, 0.25).
    assert_eq!(operation, "conditional_intervention");
    near(*mean_bits, 15.0);
    near(*variance_bits, 0.25);
    // The density of M=5 under N(4.5, 1).
    let density = -0.5 * ((2.0 * std::f64::consts::PI).ln() + 0.25);
    near(evidence_log_density_bits.unwrap(), density);
    assert_eq!(wire.query.operation(), "conditional_intervention");

    let bytes = wire.export().unwrap();
    let decoded = ObservationQueryArtifactWire::decode(&bytes).unwrap();
    assert_eq!(decoded, wire);
    assert_eq!(decoded.consume(Some(&wire.identity), &ctx()).unwrap().result, wire.result);

    // A reader of the original format gates on the version before it interprets the payload:
    // the stored version is the general one, so it is refused as another version.
    assert_ne!(wire.version, antecedent_io_core::OBSERVATION_QUERY_ARTIFACT_VERSION);

    // Forcing the version back to 1, or the feature back to the original marker, is refused.
    let mut downgraded = wire.clone();
    downgraded.version = antecedent_io_core::OBSERVATION_QUERY_ARTIFACT_VERSION;
    assert!(matches!(
        downgraded.consume(None, &ctx()).unwrap_err(),
        ObservationQueryArtifactError::UnsupportedVersion { version: 1 }
    ));
    let mut relabelled = wire.clone();
    relabelled.required_features =
        vec![antecedent_io_core::OBSERVATION_QUERY_ARTIFACT_FEATURE.to_owned()];
    assert!(matches!(
        relabelled.consume(None, &ctx()).unwrap_err(),
        ObservationQueryArtifactError::UnsupportedSemantics(_)
    ));

    // A resealed edit of the observed value changes the replayed result.
    let mut edited = wire.clone();
    edited.query = gaussian_query(&[("X", 2.0)], &[("M", 6.0)]);
    reseal(&mut edited);
    assert_eq!(
        edited.consume(None, &ctx()).unwrap_err(),
        ObservationQueryArtifactError::ResultMismatch
    );
    // An edit that is not resealed is an identity mismatch.
    let mut stale = wire.clone();
    stale.query = gaussian_query(&[("X", 2.0)], &[("M", 6.0)]);
    assert_eq!(
        stale.consume(None, &ctx()).unwrap_err(),
        ObservationQueryArtifactError::IdentityMismatch
    );
}

#[test]
fn the_gaussian_operation_and_the_listing_order_are_part_of_the_identity() {
    let spec = gaussian_chain();
    let seal = |q: ObservationQueryWire| {
        ObservationQueryArtifactWire::seal(Some(&spec), q, InferenceBasis::DeclaredModel, &ctx())
            .unwrap()
    };
    let conditioning = seal(gaussian_query(&[], &[("M", 5.0)]));
    let intervention = seal(gaussian_query(&[("X", 2.0)], &[]));
    let both = seal(gaussian_query(&[("X", 2.0)], &[("M", 5.0)]));
    assert_eq!(conditioning.query.operation(), "conditioning");
    assert_eq!(intervention.query.operation(), "intervention");
    assert_ne!(conditioning.identity, intervention.identity);
    assert_ne!(conditioning.identity, both.identity);
    assert_ne!(intervention.identity, both.identity);
    let forward = seal(gaussian_query(&[], &[("M", 5.0), ("X", 1.0)]));
    let reversed = seal(gaussian_query(&[], &[("X", 1.0), ("M", 5.0)]));
    assert_eq!(forward.identity, reversed.identity);
    assert_eq!(forward.result, reversed.result);
    // A finite-model query keeps its own, original format.
    let finite = ObservationQueryArtifactWire::seal(
        Some(&confounded(MechanismOrigin::Declared)),
        paired_query(),
        InferenceBasis::DeclaredModel,
        &ctx(),
    )
    .unwrap();
    assert_eq!(finite.version, antecedent_io_core::OBSERVATION_QUERY_ARTIFACT_VERSION);
}

#[test]
fn a_gaussian_query_on_a_finite_model_and_a_set_and_seen_variable_are_refused() {
    let wrong_family = ObservationQueryArtifactWire::seal(
        Some(&confounded(MechanismOrigin::Declared)),
        gaussian_query(&[], &[("M", 1.0)]),
        InferenceBasis::DeclaredModel,
        &ctx(),
    )
    .unwrap_err();
    assert!(matches!(wrong_family, ObservationQueryArtifactError::Replay(_)));
    let both = ObservationQueryArtifactWire::seal(
        Some(&gaussian_chain()),
        gaussian_query(&[("X", 2.0)], &[("X", 2.0)]),
        InferenceBasis::DeclaredModel,
        &ctx(),
    )
    .unwrap_err();
    match both {
        ObservationQueryArtifactError::Refused { refusals } => {
            assert_eq!(refusals[0].detail, "observation_query.mixed_conditioning_and_intervention");
        }
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn observation_after_intervention_on_a_finite_model_replays_to_the_hand_value() {
    let spec = confounded(MechanismOrigin::Declared);
    let query = ObservationQueryWire::ConditionalInterventionalOutcome {
        outcome: "Y".into(),
        interventions: vec![pair("S", "1")],
        observed: vec![pair("Z", "1")],
        measured: None,
    };
    let wire = ObservationQueryArtifactWire::seal(
        Some(&spec),
        query,
        InferenceBasis::DeclaredModel,
        &ctx(),
    )
    .unwrap();
    let ObservationResultWire::Conditional {
        operation,
        evidence_probability_bits,
        levels,
        construct_posterior: None,
    } = &wire.result
    else {
        panic!("wrong result kind");
    };
    assert_eq!(operation, "conditional_intervention");
    near(*evidence_probability_bits, 0.5);
    near(level_bits(levels, "1"), 0.9);
    assert!(wire.consume(None, &ctx()).is_ok());
    assert_eq!(wire.version, OBSERVATION_QUERY_GENERAL_ARTIFACT_VERSION);

    // The same assignment as a pure intervention is a different claim.
    let pure = ObservationQueryArtifactWire::seal(
        Some(&spec),
        ObservationQueryWire::InterventionalOutcome {
            outcome: "Y".into(),
            interventions: vec![pair("S", "1")],
        },
        InferenceBasis::DeclaredModel,
        &ctx(),
    )
    .unwrap();
    assert_ne!(pure.identity, wire.identity);

    // A measured signal on the intervened construct is refused with the mixed detail.
    let measured = ObservationQueryArtifactWire::seal(
        Some(&construct_spec()),
        ObservationQueryWire::ConditionalInterventionalOutcome {
            outcome: "Y".into(),
            interventions: vec![pair("C", "1")],
            observed: Vec::new(),
            measured: Some(measured_slot(&assay(0.9, "C"))),
        },
        InferenceBasis::DeclaredModel,
        &ctx(),
    )
    .unwrap_err();
    match measured {
        ObservationQueryArtifactError::Refused { refusals } => {
            assert_eq!(refusals[0].detail, "observation_query.mixed_conditioning_and_intervention");
        }
        other => panic!("unexpected {other:?}"),
    }
}

fn redraw_query(seed: u64, draws: u32) -> ObservationQueryWire {
    let x: Vec<f64> = (0..40_u32).map(|i| f64::from(i) / 8.0).collect();
    let y: Vec<f64> =
        x.iter().enumerate().map(|(i, v)| 0.5 + 1.5 * v + 0.4 * (3.0 * i as f64).cos()).collect();
    let n = x.len() as f64;
    let sx: f64 = x.iter().sum();
    let sxx: f64 = x.iter().map(|v| v * v).sum();
    let sy: f64 = y.iter().sum();
    let sxy: f64 = x.iter().zip(&y).map(|(a, b)| a * b).sum();
    let syy: f64 = y.iter().map(|v| v * v).sum();
    ObservationQueryWire::ParameterRedraw {
        target: RedrawTargetWire::Gaussian {
            outcome: "Y".into(),
            interventions: vec![("X".into(), bits(2.0))],
            observed: Vec::new(),
        },
        retained: vec![RetainedMechanismWire::Gaussian {
            variable: "Y".into(),
            parents: vec!["X".into()],
            gram_bits: [n, sx, sx, sxx].iter().map(|v| bits(*v)).collect(),
            xty_bits: [sy, sxy].iter().map(|v| bits(*v)).collect(),
            yty_bits: bits(syy),
            rows: 40,
        }],
        prior: RedrawPriorWire {
            dirichlet_alpha_bits: bits(1.0),
            coefficient_precision_bits: bits(1e-6),
            variance_shape_bits: bits(1e-3),
            variance_scale_bits: bits(1e-3),
        },
        draws,
        seed,
    }
}

fn regression_spec() -> ComposedModelSpec {
    spec_of(vec![lg("X", 0.0, 1.0, &[]), lg("Y", 0.5, 0.4, &[("X", 1.5)])])
}

#[test]
fn redrawn_parameters_are_aligned_draws_with_the_distribution_alignment_identity() {
    let spec = regression_spec();
    let wire = ObservationQueryArtifactWire::seal(
        Some(&spec),
        redraw_query(9, 300),
        InferenceBasis::DeclaredModel,
        &ctx(),
    )
    .unwrap();
    assert_eq!(wire.query.operation(), "parameter_redraw");
    let ObservationResultWire::ParameterDraws {
        quantities,
        draws,
        point_bits,
        values_bits,
        alignment_identity,
        reading,
    } = &wire.result
    else {
        panic!("wrong result kind");
    };
    assert_eq!(quantities, &["conditional_mean", "conditional_variance"]);
    assert_eq!(*draws, 300);
    assert_eq!(values_bits.len(), 600);
    // The point value is the composition's own: 0.5 + 1.5 * 2 and sigma^2 = 0.16.
    near(point_bits[0], 3.5);
    near(point_bits[1], 0.16);
    // The draws are labelled a spread: no coverage is claimed.
    assert_eq!(reading, "uncalibrated_spread");

    // The repo's distribution artifact: joint alignment, unmeasured, the same identity.
    let distribution = wire.distribution().unwrap();
    assert_eq!(distribution.shape(), [300, 2]);
    assert_eq!(distribution.metadata().identity.alignment, DrawAlignment::Joint);
    assert_eq!(&distribution.alignment_identity().unwrap(), alignment_identity);
    assert_eq!(distribution.metadata().identity.quantities[0].regime_id, "do(X=2)");

    let consumed = ObservationQueryArtifactWire::decode(&wire.export().unwrap())
        .unwrap()
        .consume(Some(&wire.identity), &ctx())
        .unwrap();
    assert_eq!(consumed.result, wire.result);

    // A different seed is a different claim with different draws and a different alignment.
    let other = ObservationQueryArtifactWire::seal(
        Some(&spec),
        redraw_query(10, 300),
        InferenceBasis::DeclaredModel,
        &ctx(),
    )
    .unwrap();
    assert_ne!(other.identity, wire.identity);
    assert_ne!(other.result, wire.result);
    let ObservationResultWire::ParameterDraws { alignment_identity: other_alignment, .. } =
        &other.result
    else {
        panic!("wrong result kind");
    };
    assert_ne!(other_alignment, alignment_identity);

    // A forged draw is not the replay of the premises.
    let mut forged = wire.clone();
    if let ObservationResultWire::ParameterDraws { values_bits, .. } = &mut forged.result {
        values_bits[0] = bits(99.0);
    }
    assert_eq!(
        forged.consume(None, &ctx()).unwrap_err(),
        ObservationQueryArtifactError::ResultMismatch
    );
}

#[test]
fn a_redraw_refuses_a_draw_count_outside_its_range_and_oversized_data() {
    let spec = regression_spec();
    let error = ObservationQueryArtifactWire::seal(
        Some(&spec),
        redraw_query(1, 1),
        InferenceBasis::DeclaredModel,
        &ctx(),
    )
    .unwrap_err();
    assert!(matches!(error, ObservationQueryArtifactError::Refused { .. }));
    let error = ObservationQueryArtifactWire::seal(
        Some(&spec),
        redraw_query(1, MAX_REDRAWS + 1),
        InferenceBasis::DeclaredModel,
        &ctx(),
    )
    .unwrap_err();
    assert!(matches!(error, ObservationQueryArtifactError::LimitsExceeded(_)));
}
