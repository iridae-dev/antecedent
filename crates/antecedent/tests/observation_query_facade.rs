//! Facade integration of observation-conditioned queries: conditioning and
//! intervention computed from one declared causal model and kept distinct in
//! identity and value, the refusals a smuggled intervention, a measured value
//! used as the construct and an unreachable level owe, and an artifact an
//! independent consumer replays.
//!
//! Hand-computed oracle (all binary, levels "0" / "1"):
//!   Z ~ Bernoulli(0.5); P(S=1 | Z) = 0.2, 0.8; P(Y=1 | S, Z) = 0.1, 0.5, 0.3, 0.9
//!   keyed (S=0,Z=0), (0,1), (1,0), (1,1)
//!   P(Y=1 | S=1) = 0.78, P(Y=1 | do(S=1)) = 0.6, P(S=1) = 0.5
//!   P(Y=1 | S=0) = 0.18, P(Y=1 | do(S=0)) = 0.3
//! Measured construct (prevalence 0.1, P(Y=1 | C) = 0.1, 0.7, sensitivity 0.9,
//! specificity 0.8):
//!   P(C=1 | M=1) = 1/3, P(Y=1 | M=1) = 0.3, P(Y=1 | C=1) = 0.7.

use antecedent::analysis::observation_evidence::{
    required_observations_binary_signal, required_observations_conjugate,
};
use antecedent::analysis::observation_posterior::{
    FiniteBayesInput, conjugate_posterior, finite_posterior,
};
use antecedent::analysis::observation_query::{
    Direction, ObservationQueryRefusal, ObservationQueryWire, ObservationRequest,
    ObservationResultWire, consume_observation_query_artifact,
};
use antecedent::{
    DependenceInput, MechanismCompositionRequest, MechanismInput, MechanismRefit, MechanismSource,
    PreparedMechanismComposition, StudyBuilder,
};
use antecedent_contract::{
    InferenceBasis, MeasuredIndicator, MechanismDeclaration, MechanismOriginKind,
    ObservationOperation, ObservedSlot, QuantityRole, ScientificQuantity, SlotProvenance,
    SlotValue,
};
use antecedent_core::ExecutionContext;
use antecedent_core::measurement_model::{
    ArmBinding, BinaryArmRates, ErrorStructure, MeasurementBinding, MeasurementKind,
    MeasurementModel, MeasurementParameters, MeasurementRoute, MeasurementVariable,
    ParameterEvidence, RepresentationVersion,
};
use antecedent_io::observation_query_artifact::ObservationQueryArtifactWire;
use antecedent_model::compose::{Domain, TabularRow};

const TOL: f64 = 1e-12;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

fn near(a: f64, b: f64) {
    assert!((a - b).abs() <= TOL, "{a} vs {b}");
}

fn level(levels: &[(String, f64)], name: &str) -> f64 {
    levels.iter().find(|(l, _)| l == name).map(|(_, p)| *p).unwrap()
}

fn quantity(variable: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: format!("label of {variable}"),
        role: match variable {
            "S" => QuantityRole::Treatment,
            "Y" => QuantityRole::Outcome,
            _ => QuantityRole::Covariate,
        },
        units: "category".into(),
        population_id: "pop".into(),
        regime_id: "observational".into(),
        horizon: 0,
        functional_id: "realization".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

fn table_input(variable: &str, parents: &[&str], rows: &[(&[&str], [f64; 2])]) -> MechanismInput {
    let levels = vec!["0".to_owned(), "1".to_owned()];
    MechanismInput {
        declaration: MechanismDeclaration {
            output: quantity(variable),
            parents: parents.iter().map(|p| quantity(p)).collect(),
            conditioning_variable_ids: parents.iter().map(|p| (*p).to_owned()).collect(),
            origin: MechanismOriginKind::Declared,
            artifact_id: format!("declared:{variable}"),
            evidence_id: format!("assumed:{variable}"),
            latent_causes: Vec::new(),
        },
        source: MechanismSource::DeclaredTable {
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
        output_support: Domain::Levels(levels.clone()),
        parent_domains: parents
            .iter()
            .map(|p| ((*p).to_owned(), Domain::Levels(levels.clone())))
            .collect(),
        uncertainty: None,
        latent_loadings: Vec::new(),
    }
}

fn request_of(mechanisms: Vec<MechanismInput>) -> MechanismCompositionRequest {
    let mut edges = Vec::new();
    for m in &mechanisms {
        for p in &m.declaration.parents {
            edges.push((p.variable_id.clone(), m.declaration.output.variable_id.clone()));
        }
    }
    MechanismCompositionRequest {
        target_population_id: "pop".into(),
        target_regime_id: "observational".into(),
        mechanisms,
        edges,
        admissions: Vec::new(),
        dependence: DependenceInput::Unknown,
        admit_declared_mechanisms: true,
    }
}

fn confounded() -> PreparedMechanismComposition {
    StudyBuilder::mechanism_composition(
        request_of(vec![
            table_input("Z", &[], &[(&[], [0.5, 0.5])]),
            table_input("S", &["Z"], &[(&["0"], [0.8, 0.2]), (&["1"], [0.2, 0.8])]),
            table_input(
                "Y",
                &["S", "Z"],
                &[
                    (&["0", "0"], [0.9, 0.1]),
                    (&["0", "1"], [0.5, 0.5]),
                    (&["1", "0"], [0.7, 0.3]),
                    (&["1", "1"], [0.1, 0.9]),
                ],
            ),
        ]),
        &ctx(),
    )
    .unwrap()
}

fn construct_model() -> PreparedMechanismComposition {
    StudyBuilder::mechanism_composition(
        request_of(vec![
            table_input("C", &[], &[(&[], [0.9, 0.1])]),
            table_input("Y", &["C"], &[(&["0"], [0.9, 0.1]), (&["1"], [0.3, 0.7])]),
        ]),
        &ctx(),
    )
    .unwrap()
}

fn direct(variable: &str, level: &str) -> ObservedSlot {
    ObservedSlot {
        variable_id: variable.into(),
        level: level.into(),
        provenance: SlotProvenance::Recorded,
        value: SlotValue::Direct,
    }
}

fn representation() -> RepresentationVersion {
    RepresentationVersion { coordinate_system_id: "assay".into(), version: 1 }
}

fn assay_model(population: &str) -> MeasurementModel {
    MeasurementModel {
        model_id: "assay-validation".into(),
        route: MeasurementRoute::BinaryMisclassification,
        indicator: MeasurementVariable {
            id: "test_result".into(),
            kind: MeasurementKind::NoisyMisclassified,
            units: "category".into(),
        },
        construct: MeasurementVariable {
            id: "C".into(),
            kind: MeasurementKind::LatentScientific,
            units: "category".into(),
        },
        binding: MeasurementBinding {
            population_id: population.into(),
            regime_id: "observational".into(),
            arm: ArmBinding::AllArms,
        },
        structure: ErrorStructure::NonDifferential,
        parameters: MeasurementParameters::Binary {
            rates: vec![BinaryArmRates {
                arm_id: None,
                sensitivity: 0.9,
                specificity: 0.8,
                sensitivity_variance: 0.0,
                specificity_variance: 0.0,
            }],
            min_youden: 0.1,
        },
        representation: representation(),
        evidence: ParameterEvidence::Declared { declarer: "lab".into() },
    }
}

fn measured(model: Option<MeasurementModel>, recorded: &str) -> ObservedSlot {
    ObservedSlot {
        variable_id: "C".into(),
        level: recorded.into(),
        provenance: SlotProvenance::Recorded,
        value: SlotValue::Measured(MeasuredIndicator {
            indicator_id: "test_result".into(),
            kind: MeasurementKind::NoisyMisclassified,
            model,
            positive_level: Some("1".into()),
            representation: representation(),
        }),
    }
}

fn pair(variable: &str, level: &str) -> (String, String) {
    (variable.to_owned(), level.to_owned())
}

fn refusal_details(refusal: &ObservationQueryRefusal) -> Vec<&str> {
    refusal.details()
}

#[test]
fn conditioning_and_intervention_disagree_and_stay_distinct_in_identity_and_value() {
    let prepared = confounded();
    let conditioning = prepared.condition_on("Y", &[direct("S", "1")], &ctx()).unwrap();
    let intervention = prepared.intervene_on("Y", &[pair("S", "1")], &ctx()).unwrap();
    let paired = prepared.condition_versus_intervene("Y", &[pair("S", "1")], &ctx()).unwrap();

    assert_eq!(conditioning.operation(), "conditioning");
    assert_eq!(intervention.operation(), "intervention");
    assert_eq!(paired.operation(), "conditioning_and_intervention");
    near(level(&conditioning.conditional_levels().unwrap(), "1"), 0.78);
    near(conditioning.evidence_probability().unwrap(), 0.5);
    near(level(&intervention.interventional_levels().unwrap(), "1"), 0.6);
    let (c, i, difference) = paired.paired().unwrap();
    near(level(&c, "1"), 0.78);
    near(level(&i, "1"), 0.6);
    near(difference, 0.18);

    // Same variables, same levels, same model: three different claims.
    assert_ne!(conditioning.identity(), intervention.identity());
    assert_ne!(conditioning.identity(), paired.identity());
    assert_ne!(intervention.identity(), paired.identity());
    // A conditioning claim carries no interventional value and vice versa.
    assert!(conditioning.interventional_levels().is_none());
    assert!(intervention.conditional_levels().is_none());

    // Values from declared tables are exact given the declaration.
    assert_eq!(conditioning.inference_basis(), "exact_given_declared_model");
}

#[test]
fn an_intervention_smuggled_into_a_conditioning_slot_is_refused_with_its_remedy() {
    let prepared = confounded();
    let mut assigned = direct("S", "1");
    assigned.provenance = SlotProvenance::Assigned;
    let refusal = prepared.condition_on("Y", &[assigned], &ctx()).unwrap_err();
    assert_eq!(refusal.code(), "quantity_semantics_mismatch");
    assert_eq!(refusal_details(&refusal), ["observation_query.intervention_in_conditioning_slot"]);
    assert_eq!(refusal.refusals[0].offending.as_deref(), Some("S"));
    assert!(refusal.refusals[0].remedy.is_some());

    // The same value stated honestly, as an intervention, is answered.
    assert!(prepared.intervene_on("Y", &[pair("S", "1")], &ctx()).is_ok());

    // An intervention handed to a conditioning request is the same refusal.
    let smuggled = ObservationRequest {
        operation: ObservationOperation::Conditioning,
        outcome: "Y".into(),
        conditioning: vec![direct("Z", "1")],
        interventions: vec![pair("S", "1")],
    };
    let refusal = prepared.observe(&smuggled, &ctx()).unwrap_err();
    assert_eq!(refusal_details(&refusal), ["observation_query.intervention_in_conditioning_slot"]);

    // Observations and an intervention in one request have no bounded route.
    let mixed = ObservationRequest {
        operation: ObservationOperation::Intervention,
        outcome: "Y".into(),
        conditioning: vec![direct("Z", "1")],
        interventions: vec![pair("S", "1")],
    };
    let refusal = prepared.observe(&mixed, &ctx()).unwrap_err();
    assert_eq!(refusal.code(), "route_not_supported");
    assert_eq!(
        refusal_details(&refusal),
        ["observation_query.mixed_conditioning_and_intervention"]
    );
}

#[test]
fn a_measured_value_is_refused_as_the_construct_and_inverted_through_its_model() {
    let prepared = construct_model();
    // The recorded value used as the construct: refused.
    let refusal = prepared.condition_on("Y", &[measured(None, "1")], &ctx()).unwrap_err();
    assert_eq!(refusal.code(), "proxy_offered_as_construct");
    assert_eq!(refusal_details(&refusal), ["observation_query.measured_as_construct"]);

    // Through the declared confusion matrix: exact Bayes.
    let positive =
        prepared.condition_on("Y", &[measured(Some(assay_model("pop")), "1")], &ctx()).unwrap();
    near(level(&positive.conditional_levels().unwrap(), "1"), 0.3);
    near(level(&positive.construct_posterior().unwrap(), "1"), 1.0 / 3.0);
    near(positive.evidence_probability().unwrap(), 0.27);

    // Conditioning on the construct itself is a different claim and value.
    let as_construct = prepared.condition_on("Y", &[direct("C", "1")], &ctx()).unwrap();
    near(level(&as_construct.conditional_levels().unwrap(), "1"), 0.7);
    assert_ne!(positive.identity(), as_construct.identity());

    // A model established for another population is not applied.
    let refusal = prepared
        .condition_on("Y", &[measured(Some(assay_model("other_clinic")), "1")], &ctx())
        .unwrap_err();
    assert_eq!(refusal.code(), "measurement_binding_mismatch");
    assert_eq!(refusal_details(&refusal), ["observation_query.population_mismatch"]);
}

#[test]
fn a_threshold_no_observation_reaches_is_refused_with_the_supremum_reachable() {
    let prepared = confounded();
    let reached =
        prepared.threshold_crossing("Y", "1", "S", &[], 0.5, Direction::Raise, &ctx()).unwrap();
    let (rows, crossing) = reached.threshold_rows().unwrap();
    near(crossing, 0.5);
    assert!(rows.iter().any(|(l, _, c, crosses)| l == "1" && *crosses && (c - 0.78).abs() < TOL));

    let refusal =
        prepared.threshold_crossing("Y", "1", "S", &[], 0.9, Direction::Raise, &ctx()).unwrap_err();
    assert_eq!(refusal.code(), "evidence_cannot_reach_level");
    assert_eq!(refusal_details(&refusal), ["observation_query.level_unreachable"]);
    let supplied: f64 = refusal.refusals[0].supplied.as_deref().unwrap().parse().unwrap();
    near(supplied, 0.78);
    assert_eq!(refusal.refusals[0].expected.as_deref(), Some("0.9"));
}

#[test]
fn the_artifact_replays_and_every_semantic_mutation_is_refused() {
    let prepared = confounded();
    let claim = prepared.condition_versus_intervene("Y", &[pair("S", "1")], &ctx()).unwrap();
    let bytes = claim.export().unwrap();
    let consumed =
        consume_observation_query_artifact(&bytes, Some(claim.identity()), &ctx()).unwrap();
    assert_eq!(consumed.identity, claim.identity());
    assert_eq!(consumed.operation, "conditioning_and_intervention");
    assert_eq!(&consumed.result, claim.result());

    // A wrong retained identity is refused.
    let refusal = consume_observation_query_artifact(&bytes, Some("00"), &ctx()).unwrap_err();
    assert_eq!(refusal_details(&refusal), ["observation_query.artifact_refused"]);

    // Operation swapped without resealing: the identity no longer matches.
    let mut wire = ObservationQueryArtifactWire::decode(&bytes).unwrap();
    wire.query = ObservationQueryWire::InterventionalOutcome {
        outcome: "Y".into(),
        interventions: vec![pair("S", "1")],
    };
    let tampered = wire.export().unwrap();
    assert!(consume_observation_query_artifact(&tampered, None, &ctx()).is_err());

    // A stored value edited without resealing is a refused artifact.
    let mut wire = ObservationQueryArtifactWire::decode(&bytes).unwrap();
    if let ObservationResultWire::Paired { max_abs_difference_bits, .. } = &mut wire.result {
        *max_abs_difference_bits = 0.0_f64.to_bits();
    }
    let tampered = wire.export().unwrap();
    let refusal = consume_observation_query_artifact(&tampered, None, &ctx()).unwrap_err();
    assert_eq!(refusal_details(&refusal), ["observation_query.artifact_refused"]);

    // Intervention swapped for conditioning in a resealed artifact would
    // change the stored result, which no longer replays.
    let mut wire = ObservationQueryArtifactWire::decode(&bytes).unwrap();
    wire.result = ObservationResultWire::Interventional {
        operation: "intervention".into(),
        levels: vec![("0".into(), 0.4_f64.to_bits()), ("1".into(), 0.6_f64.to_bits())],
    };
    let tampered = wire.export().unwrap();
    assert!(consume_observation_query_artifact(&tampered, None, &ctx()).is_err());
}

#[test]
fn a_refreshed_composition_gives_a_new_claim_and_leaves_the_earlier_one_unchanged() {
    let prepared = confounded();
    let before = prepared.condition_on("Y", &[direct("S", "1")], &ctx()).unwrap();

    // Refit S on new data: P(S=1 | Z) = 0.5, 0.5 removes the confounding path.
    let refreshed = prepared
        .refresh_mechanisms(
            vec![MechanismRefit {
                variable_id: "S".into(),
                source: MechanismSource::DeclaredTable {
                    levels: vec!["0".into(), "1".into()],
                    rows: [("0", [0.5, 0.5]), ("1", [0.5, 0.5])]
                        .iter()
                        .map(|(z, p)| TabularRow {
                            parent_levels: vec![("Z".to_owned(), (*z).to_owned())],
                            probabilities: vec![("0".to_owned(), p[0]), ("1".to_owned(), p[1])],
                        })
                        .collect(),
                },
                artifact_id: "declared:S:v2".into(),
                evidence_id: "assumed:S:v2".into(),
                uncertainty: None,
            }],
            None,
            &ctx(),
        )
        .unwrap();
    let after = refreshed.condition_on("Y", &[direct("S", "1")], &ctx()).unwrap();
    near(level(&after.conditional_levels().unwrap(), "1"), 0.6);
    assert_ne!(before.identity(), after.identity());
    near(level(&before.conditional_levels().unwrap(), "1"), 0.78);
    // With the arrow into S removed, conditioning and intervention agree.
    let (c, i, difference) = refreshed
        .condition_versus_intervene("Y", &[pair("S", "1")], &ctx())
        .unwrap()
        .paired()
        .unwrap();
    near(level(&c, "1"), level(&i, "1"));
    near(difference, 0.0);
}

#[test]
fn closed_interval_and_calibration_routes_refuse() {
    let prepared = confounded();
    let claim = prepared.condition_on("Y", &[direct("S", "1")], &ctx()).unwrap();
    let interval = claim.interval().unwrap_err();
    assert_eq!(interval.code(), "cell_not_licensed");
    assert_eq!(refusal_details(&interval), ["observation_query.interval_route_closed"]);
    assert!(interval.refusals[0].remedy.is_some());
    let calibrated = claim.calibrated_probability().unwrap_err();
    assert_eq!(calibrated.code(), "cell_not_licensed");
    assert_eq!(refusal_details(&calibrated), ["observation_query.calibrated_probability_closed"]);

    // The same posterior read from a fitted model is a labelled point value and
    // is still refused an interval or calibration.
    let model = FiniteBayesInput {
        hypotheses: vec!["ill".into(), "well".into()],
        prior: vec![0.01, 0.99],
        observation_levels: vec!["neg".into(), "pos".into()],
        likelihood: vec![0.1, 0.9, 0.95, 0.05],
    };
    let fitted = finite_posterior(
        &model,
        &["pos".to_owned()],
        InferenceBasis::SampleFittedPointOnly,
        &ctx(),
    )
    .unwrap();
    assert_eq!(fitted.inference_basis(), "point_only_sample_fitted");
    assert_eq!(fitted.interval().unwrap_err().code(), "cell_not_licensed");
    assert_eq!(fitted.calibrated_probability().unwrap_err().code(), "cell_not_licensed");
}

#[test]
fn bayes_updates_through_the_facade_match_hand_values() {
    let model = FiniteBayesInput {
        hypotheses: vec!["ill".into(), "well".into()],
        prior: vec![0.01, 0.99],
        observation_levels: vec!["neg".into(), "pos".into()],
        likelihood: vec![0.1, 0.9, 0.95, 0.05],
    };
    let posterior =
        finite_posterior(&model, &["pos".to_owned()], InferenceBasis::DeclaredModel, &ctx())
            .unwrap();
    near(level(&posterior.posterior().unwrap(), "ill"), 2.0 / 13.0);
    near(posterior.evidence_probability().unwrap(), 0.0585);
    assert_eq!(posterior.operation(), "posterior_update");
    assert_eq!(posterior.inference_basis(), "exact_given_declared_model");

    // The same numbers read from a fitted model carry another identity.
    let fitted = finite_posterior(
        &model,
        &["pos".to_owned()],
        InferenceBasis::SampleFittedPointOnly,
        &ctx(),
    )
    .unwrap();
    assert_eq!(fitted.inference_basis(), "point_only_sample_fitted");
    assert_ne!(fitted.identity(), posterior.identity());
    assert_eq!(fitted.posterior(), posterior.posterior());

    let conjugate =
        conjugate_posterior(1.0, 1.0, 7, 3, InferenceBasis::DeclaredModel, &ctx()).unwrap();
    let (alpha, beta, predictive) = conjugate.conjugate_posterior().unwrap();
    near(alpha, 8.0);
    near(beta, 4.0);
    near(predictive, 2.0 / 3.0);

    // The claim exports and a fresh consumer replays it.
    let bytes = posterior.export().unwrap();
    let consumed =
        consume_observation_query_artifact(&bytes, Some(posterior.identity()), &ctx()).unwrap();
    assert_eq!(consumed.operation, "posterior_update");
    assert_eq!(&consumed.result, posterior.result());
}

#[test]
fn required_evidence_through_the_facade_matches_closed_forms() {
    let needed = required_observations_conjugate(
        1.0,
        1.0,
        0.9,
        Direction::Raise,
        1000,
        InferenceBasis::DeclaredModel,
        &ctx(),
    )
    .unwrap();
    let (count, value) = needed.required_observations().unwrap();
    assert_eq!(count, 8);
    near(value, 0.9);

    let readings = required_observations_binary_signal(
        0.01,
        0.5,
        Direction::Raise,
        0.9,
        0.05,
        100,
        InferenceBasis::DeclaredModel,
        &ctx(),
    )
    .unwrap();
    assert_eq!(readings.required_observations().unwrap().0, 2);
    assert_eq!(readings.operation(), "evidence_requirement");
}

#[test]
fn an_unreachable_evidence_level_is_refused_through_the_facade() {
    // A signal that does not move the probability: the prior is the supremum.
    let refusal = required_observations_binary_signal(
        0.3,
        0.9,
        Direction::Raise,
        0.5,
        0.5,
        100,
        InferenceBasis::DeclaredModel,
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(refusal.code(), "evidence_cannot_reach_level");
    assert_eq!(refusal_details(&refusal), ["observation_query.level_unreachable"]);
    let supplied: f64 = refusal.refusals[0].supplied.as_deref().unwrap().parse().unwrap();
    near(supplied, 0.3);
    assert_eq!(refusal.refusals[0].expected.as_deref(), Some("0.9"));

    // Only the observation budget is the obstacle: the value at the bound is stated.
    let budget = required_observations_conjugate(
        1.0,
        1.0,
        0.99,
        Direction::Raise,
        50,
        InferenceBasis::DeclaredModel,
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(budget.code(), "evidence_cannot_reach_level");
    let supplied: f64 = budget.refusals[0].supplied.as_deref().unwrap().parse().unwrap();
    near(supplied, 51.0 / 52.0);
}

#[test]
fn cancellation_is_not_a_verdict() {
    let prepared = confounded();
    let ctx = ExecutionContext::for_tests(1);
    ctx.cancellation.cancel();
    let refusal = prepared.condition_on("Y", &[direct("S", "1")], &ctx).unwrap_err();
    assert_eq!(refusal.code(), "cancelled_no_claim");
}
