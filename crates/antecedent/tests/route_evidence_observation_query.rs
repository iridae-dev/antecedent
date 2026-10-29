//! Facade entries for observation-conditioned queries and declared-model
//! evidence, one test per entry, each against its own hand-derived value.
//!
//! Confounded model (all binary): `Z ~ Bernoulli(0.5)`,
//! `P(S=1 | Z) = 0.9, 0.1`, `P(Y=1 | S, Z) = 0.1, 0.5, 0.3, 0.9` keyed
//! `(S=0,Z=0), (0,1), (1,0), (1,1)`:
//!   P(S=1) = 0.5, P(Y=1 | S=1) = 0.36, P(Y=1 | S=0) = 0.46,
//!   P(Y=1 | do(S=1)) = 0.6, P(Y=1 | do(S=0)) = 0.3,
//!   P(Y=1 | Z=1) = 0.54, P(Y=1 | S=1, Z=0) = 0.3 with P(S=1, Z=0) = 0.45.

use antecedent::analysis::observation_evidence::{
    required_bayes_factor, required_observations_binary_signal, required_observations_conjugate,
};
use antecedent::analysis::observation_posterior::{
    FiniteBayesInput, conjugate_posterior, finite_posterior,
};
use antecedent::analysis::observation_query::{
    ContinuousAssignments, Direction, ObservationRequest, consume_observation_query_artifact,
};
use antecedent::{
    DependenceInput, MechanismCompositionRequest, MechanismInput, MechanismSource,
    PreparedMechanismComposition, StudyBuilder,
};
use antecedent_contract::{
    InferenceBasis, MechanismDeclaration, MechanismOriginKind, ObservationOperation, ObservedSlot,
    QuantityRole, ScientificQuantity, SlotProvenance, SlotValue,
};
use antecedent_core::ExecutionContext;
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
            table_input("S", &["Z"], &[(&["0"], [0.1, 0.9]), (&["1"], [0.9, 0.1])]),
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

fn direct(variable: &str, level: &str) -> ObservedSlot {
    ObservedSlot {
        variable_id: variable.into(),
        level: level.into(),
        provenance: SlotProvenance::Recorded,
        value: SlotValue::Direct,
    }
}

fn pair(variable: &str, level: &str) -> (String, String) {
    (variable.to_owned(), level.to_owned())
}

fn screening_model() -> FiniteBayesInput {
    FiniteBayesInput {
        hypotheses: vec!["ill".into(), "well".into()],
        prior: vec![0.5, 0.5],
        observation_levels: vec!["neg".into(), "pos".into()],
        likelihood: vec![0.1, 0.9, 0.95, 0.05],
    }
}

#[test]
fn observe_dispatches_a_conditioning_or_an_intervention_request() {
    let prepared = confounded();
    let conditioning = ObservationRequest {
        operation: ObservationOperation::Conditioning,
        outcome: "Y".into(),
        conditioning: vec![direct("Z", "1")],
        interventions: vec![],
        continuous: ContinuousAssignments::default(),
    };
    let claim = prepared.observe(&conditioning, &ctx()).unwrap();
    assert_eq!(claim.operation(), "conditioning");
    near(level(&claim.conditional_levels().unwrap(), "1"), 0.54);
    near(claim.evidence_probability().unwrap(), 0.5);
    let intervention = ObservationRequest {
        operation: ObservationOperation::Intervention,
        outcome: "Y".into(),
        conditioning: vec![],
        interventions: vec![pair("S", "0")],
        continuous: ContinuousAssignments::default(),
    };
    let claim = prepared.observe(&intervention, &ctx()).unwrap();
    assert_eq!(claim.operation(), "intervention");
    near(level(&claim.interventional_levels().unwrap(), "1"), 0.3);
    // Recorded values of another variable after an intervention are the interventional
    // law conditioned on them; a variable that is both recorded and set is refused.
    let mixed = ObservationRequest {
        operation: ObservationOperation::Intervention,
        outcome: "Y".into(),
        conditioning: vec![direct("Z", "1")],
        interventions: vec![pair("S", "1")],
        continuous: ContinuousAssignments::default(),
    };
    let claim = prepared.observe(&mixed, &ctx()).unwrap();
    assert_eq!(claim.operation(), "conditional_intervention");
    near(level(&claim.conditional_levels().unwrap(), "1"), 0.9);
    let set_and_seen = ObservationRequest {
        operation: ObservationOperation::Intervention,
        outcome: "Y".into(),
        conditioning: vec![direct("S", "1")],
        interventions: vec![pair("S", "1")],
        continuous: ContinuousAssignments::default(),
    };
    let refusal = prepared.observe(&set_and_seen, &ctx()).unwrap_err();
    assert_eq!(refusal.code(), "route_not_supported");
    assert_eq!(refusal.details(), ["observation_query.mixed_conditioning_and_intervention"]);
}

#[test]
fn condition_on_answers_the_observational_conditional_with_its_evidence_probability() {
    let prepared = confounded();
    let claim = prepared.condition_on("Y", &[direct("S", "0")], &ctx()).unwrap();
    near(level(&claim.conditional_levels().unwrap(), "1"), 0.46);
    near(claim.evidence_probability().unwrap(), 0.5);
    assert_eq!(claim.inference_basis(), "exact_given_declared_model");
    // Two recorded values read the table cell; the order of the slots is not
    // part of the claim.
    let forward =
        prepared.condition_on("Y", &[direct("S", "1"), direct("Z", "0")], &ctx()).unwrap();
    let reversed =
        prepared.condition_on("Y", &[direct("Z", "0"), direct("S", "1")], &ctx()).unwrap();
    near(level(&forward.conditional_levels().unwrap(), "1"), 0.3);
    near(forward.evidence_probability().unwrap(), 0.45);
    assert_eq!(forward.identity(), reversed.identity());
    // An unknown variable is a typed refusal with a remedy.
    let refusal = prepared.condition_on("Y", &[direct("Q", "1")], &ctx()).unwrap_err();
    assert_eq!(refusal.code(), "unknown_variable");
    assert_eq!(refusal.details(), ["observation_query.unknown_variable"]);
    assert!(refusal.refusals[0].remedy.is_some());
}

#[test]
fn intervene_on_answers_the_truncated_factorization() {
    let prepared = confounded();
    let one = prepared.intervene_on("Y", &[pair("S", "1")], &ctx()).unwrap();
    near(level(&one.interventional_levels().unwrap(), "1"), 0.6);
    assert_eq!(one.operation(), "intervention");
    assert!(one.evidence_probability().is_none());
    let zero = prepared.intervene_on("Y", &[pair("S", "0")], &ctx()).unwrap();
    near(level(&zero.interventional_levels().unwrap(), "1"), 0.3);
    // Setting a root equals observing it: P(Y=1 | do(Z=1)) = P(Y=1 | Z=1).
    let root = prepared.intervene_on("Y", &[pair("Z", "1")], &ctx()).unwrap();
    near(level(&root.interventional_levels().unwrap(), "1"), 0.54);
    assert_ne!(one.identity(), zero.identity());
    let level = prepared.intervene_on("Y", &[pair("S", "7")], &ctx()).unwrap_err();
    assert_eq!(level.code(), "invalid_argument");
    assert_eq!(level.details(), ["observation_query.unknown_level"]);
    assert_eq!(level.refusals[0].offending.as_deref(), Some("S=7"));
    assert!(level.refusals[0].remedy.is_some());
    let unknown = prepared.intervene_on("Y", &[pair("Q", "1")], &ctx()).unwrap_err();
    assert_eq!(unknown.code(), "unknown_variable");
    assert_eq!(unknown.details(), ["observation_query.unknown_variable"]);
    assert!(unknown.refusals[0].remedy.is_some());
    let twice = prepared.intervene_on("Y", &[pair("S", "1"), pair("S", "0")], &ctx()).unwrap_err();
    assert_eq!(twice.code(), "invalid_argument");
    assert_eq!(twice.details(), ["observation_query.repeated_slot"]);
    assert!(twice.refusals[0].remedy.is_some());
}

#[test]
fn condition_versus_intervene_pairs_both_laws_and_states_their_gap() {
    let prepared = confounded();
    let claim = prepared.condition_versus_intervene("Y", &[pair("S", "1")], &ctx()).unwrap();
    let (conditioning, intervention, difference) = claim.paired().unwrap();
    near(level(&conditioning, "1"), 0.36);
    near(level(&intervention, "1"), 0.6);
    near(difference, 0.24);
    let zero = prepared.condition_versus_intervene("Y", &[pair("S", "0")], &ctx()).unwrap();
    let (c0, i0, d0) = zero.paired().unwrap();
    near(level(&c0, "1"), 0.46);
    near(level(&i0, "1"), 0.3);
    near(d0, 0.16);
    assert_eq!(claim.operation(), "conditioning_and_intervention");
    assert_ne!(claim.identity(), zero.identity());
}

#[test]
fn finite_posterior_applies_bayes_rule_to_the_declared_hypothesis_space() {
    // prior (1/2, 1/2), one negative reading: ill 0.5 * 0.1 = 0.05,
    // well 0.5 * 0.95 = 0.475, evidence 0.525, P(ill) = 2 / 21.
    let one = finite_posterior(
        &screening_model(),
        &["neg".to_owned()],
        InferenceBasis::DeclaredModel,
        &ctx(),
    )
    .unwrap();
    near(level(&one.posterior().unwrap(), "ill"), 2.0 / 21.0);
    near(one.evidence_probability().unwrap(), 0.525);
    // prior (0.01, 0.99), two positive readings:
    // ill 0.01 * 0.81 = 0.0081, well 0.99 * 0.0025 = 0.002475,
    // evidence 0.010575, P(ill) = 36 / 47.
    let rare = FiniteBayesInput { prior: vec![0.01, 0.99], ..screening_model() };
    let two = finite_posterior(
        &rare,
        &["pos".to_owned(), "pos".to_owned()],
        InferenceBasis::DeclaredModel,
        &ctx(),
    )
    .unwrap();
    near(level(&two.posterior().unwrap(), "ill"), 36.0 / 47.0);
    near(two.evidence_probability().unwrap(), 0.010_575);
    // An unknown observation level is refused, not skipped.
    let unknown = finite_posterior(
        &screening_model(),
        &["maybe".to_owned()],
        InferenceBasis::DeclaredModel,
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(unknown.code(), "invalid_argument");
    assert_eq!(unknown.details(), ["observation_query.unknown_level"]);
    assert_eq!(unknown.refusals[0].offending.as_deref(), Some("observation=maybe"));
}

#[test]
fn conjugate_posterior_adds_the_counts_to_the_beta_shapes() {
    // Beta(2, 3) after 4 successes and 6 failures: Beta(6, 9), predictive 6 / 15.
    let claim = conjugate_posterior(2.0, 3.0, 4, 6, InferenceBasis::DeclaredModel, &ctx()).unwrap();
    let (alpha, beta, predictive) = claim.conjugate_posterior().unwrap();
    near(alpha, 6.0);
    near(beta, 9.0);
    near(predictive, 0.4);
    // A non-positive shape is not a Beta prior.
    let bad = conjugate_posterior(0.0, 3.0, 4, 6, InferenceBasis::DeclaredModel, &ctx()).unwrap_err();
    assert_eq!(bad.code(), "invalid_argument");
    assert_eq!(bad.details(), ["observation_query.invalid_declaration"]);
}

#[test]
fn the_required_bayes_factor_is_the_ratio_of_target_to_prior_odds() {
    // prior 0.1 has odds 1/9, level 0.5 has odds 1: factor 9.
    let raise = required_bayes_factor(0.1, 0.5, Direction::Raise, &ctx()).unwrap();
    near(raise.required_factor().unwrap(), 9.0);
    // prior 0.8 has odds 4, level 0.5 has odds 1: factor 1/4 lowers it.
    let lower = required_bayes_factor(0.8, 0.5, Direction::Lower, &ctx()).unwrap();
    near(lower.required_factor().unwrap(), 0.25);
    // Probabilities on the boundary have no finite odds.
    let edge = required_bayes_factor(0.1, 1.0, Direction::Raise, &ctx()).unwrap_err();
    assert_eq!(edge.code(), "invalid_argument");
    assert_eq!(edge.details(), ["observation_query.invalid_declaration"]);
}

#[test]
fn readings_required_of_a_binary_signal_are_the_exact_minimum() {
    // prior 0.2 (odds 1/4), per-reading factor 0.8 / 0.2 = 4, level 0.9
    // (odds 9): 4^n / 4 >= 9 first holds at n = 3, giving odds 16, 16 / 17.
    let claim = required_observations_binary_signal(
        0.2,
        0.9,
        Direction::Raise,
        0.8,
        0.2,
        100,
        InferenceBasis::DeclaredModel,
        &ctx(),
    )
    .unwrap();
    let (count, value) = claim.required_observations().unwrap();
    assert_eq!(count, 3);
    near(value, 16.0 / 17.0);
    // With only two readings allowed the level is out of reach and the value
    // at the bound is stated: odds 4, probability 0.8.
    let refusal = required_observations_binary_signal(
        0.2,
        0.9,
        Direction::Raise,
        0.8,
        0.2,
        2,
        InferenceBasis::DeclaredModel,
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(refusal.code(), "evidence_cannot_reach_level");
    let supplied: f64 = refusal.refusals[0].supplied.as_deref().unwrap().parse().unwrap();
    near(supplied, 0.8);
}

#[test]
fn consecutive_outcomes_required_of_a_beta_model_match_the_predictive_closed_form() {
    // Beta(2, 2): (2 + n) / (4 + n) >= 0.7 first holds at n = 3, value 5 / 7.
    let raise = required_observations_conjugate(
        2.0,
        2.0,
        0.7,
        Direction::Raise,
        100,
        InferenceBasis::DeclaredModel,
        &ctx(),
    )
    .unwrap();
    let (count, value) = raise.required_observations().unwrap();
    assert_eq!(count, 3);
    near(value, 5.0 / 7.0);
    // Failures lower the predictive 2 / (4 + n) to at most 0.3 at n = 3: 2 / 7.
    let lower = required_observations_conjugate(
        2.0,
        2.0,
        0.3,
        Direction::Lower,
        100,
        InferenceBasis::DeclaredModel,
        &ctx(),
    )
    .unwrap();
    let (count, value) = lower.required_observations().unwrap();
    assert_eq!(count, 3);
    near(value, 2.0 / 7.0);
}

#[test]
fn an_exported_claim_decodes_to_a_canonical_artifact_of_its_own_identity() {
    let claim = finite_posterior(
        &screening_model(),
        &["neg".to_owned()],
        InferenceBasis::DeclaredModel,
        &ctx(),
    )
    .unwrap();
    let bytes = claim.export().unwrap();
    assert!(!bytes.is_empty());
    assert_eq!(claim.export().unwrap(), bytes, "export is a pure function of the claim");
    // The bytes are the canonical encoding of the wire they decode to.
    let wire = ObservationQueryArtifactWire::decode(&bytes).unwrap();
    assert_eq!(wire.export().unwrap(), bytes);
    // A claim of another observation is another artifact.
    let other = finite_posterior(
        &screening_model(),
        &["pos".to_owned()],
        InferenceBasis::DeclaredModel,
        &ctx(),
    )
    .unwrap();
    assert_ne!(other.export().unwrap(), bytes);
}

#[test]
fn a_fresh_consumer_replays_a_required_evidence_claim_and_refuses_edits() {
    let claim = required_bayes_factor(0.1, 0.5, Direction::Raise, &ctx()).unwrap();
    let bytes = claim.export().unwrap();
    let consumed =
        consume_observation_query_artifact(&bytes, Some(claim.identity()), &ctx()).unwrap();
    assert_eq!(consumed.identity, claim.identity());
    assert_eq!(&consumed.result, claim.result());
    // A retained identity that is not this claim's is refused.
    let refusal = consume_observation_query_artifact(&bytes, Some("00"), &ctx()).unwrap_err();
    assert_eq!(refusal.details(), ["observation_query.artifact_refused"]);
    // Truncated bytes are not an artifact.
    let cut = consume_observation_query_artifact(&bytes[..bytes.len() / 2], None, &ctx()).unwrap_err();
    assert_eq!(cut.code(), "invalid_argument");
    assert_eq!(cut.details(), ["observation_query.artifact_invalid"]);
    // A single-byte change either fails to replay or replays to this claim.
    for position in (0..bytes.len()).step_by((bytes.len() / 32).max(1)) {
        let mut mutated = bytes.clone();
        mutated[position] ^= 0x55;
        if let Ok(other) = consume_observation_query_artifact(&mutated, None, &ctx()) {
            assert_eq!(other.identity, claim.identity());
        }
    }
}
