//! Exact execution of identified counterfactual transport derivations against
//! independent finite-SCM enumeration.
//!
//! Every expected number is enumerated by the oracle from the target
//! population's own structural model; the derivation only ever sees the
//! supplied regime laws. The fixtures live with the identifier's tests and are
//! included here by path so one oracle serves both layers.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "../../antecedent-identify/tests/support/counterfactual_transport_adapter.rs"]
mod counterfactual_transport_adapter;
#[path = "../../antecedent-identify/tests/support/counterfactual_transport_cases.rs"]
mod counterfactual_transport_cases;
#[path = "../../antecedent-identify/tests/support/finite_scm_oracle.rs"]
mod finite_scm_oracle;

use antecedent_core::ExecutionContext;
use antecedent_estimate::{
    EstimationError, evaluate_counterfactual_transport, lower_counterfactual_transport_items,
};
use antecedent_expr::{ExactDiscreteLaw, ExactEvaluationLimits, ExactTransportData, ExprNode};
use antecedent_identify::{
    COUNTERFACTUAL_TRANSPORT_DEFAULT_LIMITS, CounterfactualTransportDecision,
    CounterfactualTransportDerivation, identify_counterfactual_transport,
};
use counterfactual_transport_adapter as adapter;
use counterfactual_transport_cases::{Case, ExpectedClass, Target, all_cases};

const TOLERANCE: f64 = 1e-12;

fn controllable_for(case: &Case) -> Vec<&'static str> {
    if case.name == "missing_interventional_evidence" {
        vec!["X"]
    } else {
        adapter::evidence_controllable(case)
    }
}

fn derive(case: &Case, target: &Target) -> CounterfactualTransportDerivation {
    let problem = adapter::problem(case);
    let catalog = adapter::catalog(case);
    let sources = adapter::sources(case, &controllable_for(case));
    let query = adapter::query(case, &target.query, sources);
    let ctx = ExecutionContext::for_tests(1);
    let decision = identify_counterfactual_transport(
        &problem,
        &query,
        &catalog.catalog,
        COUNTERFACTUAL_TRANSPORT_DEFAULT_LIMITS,
        &ctx,
    )
    .expect("a fixture is a valid request");
    match decision {
        CounterfactualTransportDecision::Identified(derivation) => *derivation,
        other => panic!("{} / {}: expected a derivation, got {other:?}", case.name, target.label),
    }
}

fn evaluate(
    derivation: &CounterfactualTransportDerivation,
    data: &ExactTransportData,
) -> Result<f64, EstimationError> {
    let ctx = ExecutionContext::for_tests(1);
    evaluate_counterfactual_transport(derivation, data, ExactEvaluationLimits::default(), &ctx)
        .map(|point| point.probability)
}

fn case_named(name: &str) -> Case {
    all_cases().into_iter().find(|c| c.name == name).expect("fixture exists")
}

fn refusal_code(error: &EstimationError) -> &'static str {
    match error {
        EstimationError::Refused { code, .. } => code,
        other => panic!("expected a registered refusal, got {other:?}"),
    }
}

fn refusal_message(error: &EstimationError) -> String {
    match error {
        EstimationError::Refused { message, .. } => message.clone(),
        other => panic!("expected a registered refusal, got {other:?}"),
    }
}

#[test]
fn identified_targets_equal_the_enumerated_truth() {
    let mut checked = 0;
    for case in all_cases() {
        let adapted = adapter::catalog(&case);
        let data = adapter::laws(&case, &adapted);
        for target in case.targets.iter().filter(|t| {
            t.class == ExpectedClass::Identified && t.query.population == case.target_population
        }) {
            let derivation = derive(&case, target);
            let point = evaluate_counterfactual_transport(
                &derivation,
                &data,
                ExactEvaluationLimits::default(),
                &ExecutionContext::for_tests(1),
            )
            .unwrap_or_else(|e| panic!("{} / {}: {e}", case.name, target.label));
            let truth = target.expected_value.expect("an identified target is defined");
            assert!(
                (point.probability - truth).abs() <= TOLERANCE,
                "{} / {}: derived {} against enumerated {truth}",
                case.name,
                target.label,
                point.probability
            );
            assert!(point.cells_read > 0, "{} / {}", case.name, target.label);
            assert!(point.conditioning_mass_checked);
            // The cited laws are the supplied ones, identified by population, regime
            // and snapshot.
            assert!(!point.evidence.is_empty());
            for evidence in &point.evidence {
                assert!(
                    derivation.cited.iter().any(|c| c.regime == evidence.regime
                        && c.population == evidence.population),
                    "{} / {}",
                    case.name,
                    target.label
                );
                assert!(evidence.snapshot_identity.starts_with(evidence.population.as_ref()));
            }
            checked += 1;
        }
    }
    assert!(checked >= 7, "the named positive queries were exercised ({checked})");
}

#[test]
fn natural_direct_and_indirect_effects_match_the_enumerated_contrasts() {
    let case = case_named("nonlinear_mediation_covariate_shift");
    let adapted = adapter::catalog(&case);
    let data = adapter::laws(&case, &adapted);
    let value = |label: &str| {
        let target = case.targets.iter().find(|t| t.label == label).unwrap();
        evaluate(&derive(&case, target), &data).unwrap()
    };
    let truth = |label: &str| {
        case.targets.iter().find(|t| t.label == label).unwrap().expected_value.unwrap()
    };
    let nde = value("Y_{x1,M_{x0}}") - value("Y_{x0,M_{x0}}");
    let nie = value("Y_{x0,M_{x1}}") - value("Y_{x0,M_{x0}}");
    let nde_truth = truth("Y_{x1,M_{x0}}") - truth("Y_{x0,M_{x0}}");
    let nie_truth = truth("Y_{x0,M_{x1}}") - truth("Y_{x0,M_{x0}}");
    assert!((nde - nde_truth).abs() <= TOLERANCE, "NDE {nde} vs {nde_truth}");
    assert!((nie - nie_truth).abs() <= TOLERANCE, "NIE {nie} vs {nie_truth}");
    // The fixture is not degenerate: effects are nonzero and the target law of Z
    // differs from the source's, so the shift matters.
    assert!(nde_truth.abs() > 1e-3 && nie_truth.abs() > 1e-3);
    let source_only = {
        let mut shifted = case.clone();
        shifted.target_population = "source";
        shifted
    };
    // Y_{x1,M_{x0}} is 1/2 in every covariate stratum, so it is the same in both
    // populations; Y_{x0,M_{x0}} depends on Z (1/2 at z=0, 5/16 at z=1).
    let source_value = shifted_truth(&source_only, "Y_{x0,M_{x0}}");
    assert!(
        (source_value - truth("Y_{x0,M_{x0}}")).abs() > 1e-3,
        "the source value differs from the target value"
    );
}

fn shifted_truth(case: &Case, label: &str) -> f64 {
    let target = case.targets.iter().find(|t| t.label == label).unwrap();
    let mut query = target.query.clone();
    case.target_population.clone_into(&mut query.population);
    query.value(&case.populations).unwrap()
}

#[test]
fn changed_evidence_moves_the_number_to_the_new_enumerated_truth() {
    let case = case_named("nonlinear_mediation_covariate_shift");
    // Y_{x0,M_{x0}} depends on Z (Y_{x1,M_{x0}} is 1/2 in every stratum).
    let target = &case.targets[1];
    assert_eq!(target.label, "Y_{x0,M_{x0}}");
    let derivation = derive(&case, target);
    // The target population's covariate law changes; the structure does not.
    let mut shifted = case.clone();
    let source = finite_scm_oracle::population(&case.populations, "source").clone();
    let new_target = finite_scm_oracle::population(&case.populations, "target")
        .with_exo_law("Uz", finite_scm_oracle::bern(0.5));
    shifted.populations = vec![
        finite_scm_oracle::Population::new("source", source),
        finite_scm_oracle::Population::new("target", new_target),
    ];
    let truth = target.query.value(&shifted.populations).unwrap();
    let before = target.expected_value.unwrap();
    assert!((truth - before).abs() > 1e-3, "the shift moves the target value");
    let adapted = adapter::catalog(&shifted);
    let data = adapter::laws(&shifted, &adapted);
    let value = evaluate(&derivation, &data).unwrap();
    assert!((value - truth).abs() <= TOLERANCE, "{value} vs {truth}");
}

#[test]
fn zero_support_conditioning_is_a_typed_refusal_not_a_number() {
    let case = case_named("zero_support_conditioning_event");
    let adapted = adapter::catalog(&case);
    let data = adapter::laws(&case, &adapted);
    let target = &case.targets[1];
    assert_eq!(target.class, ExpectedClass::UndefinedZeroSupport);
    assert!(target.expected_value.is_none());
    let derivation = derive(&case, target);
    let error = evaluate(&derivation, &data).unwrap_err();
    assert_eq!(refusal_code(&error), "transport_support_failure");
    assert!(
        refusal_message(&error).contains("counterfactual_transport.zero_conditioning_mass"),
        "{error:?}"
    );
    // The same query is defined, and enumerated, in the source population.
    let mut in_source = case.clone();
    in_source.target_population = "source";
    let source_target = &case.targets[0];
    let derivation = derive(&in_source, source_target);
    let adapted = adapter::catalog(&in_source);
    let data = adapter::laws(&in_source, &adapted);
    let value = evaluate(&derivation, &data).unwrap();
    let truth = source_target.expected_value.unwrap();
    assert!((truth - 0.25).abs() <= TOLERANCE);
    assert!((value - truth).abs() <= TOLERANCE, "{value} vs {truth}");
}

fn rebuilt(law: &ExactDiscreteLaw, population: &str) -> ExactDiscreteLaw {
    ExactDiscreteLaw::try_new(
        population,
        law.regime(),
        law.interventions().to_vec(),
        law.axes().to_vec(),
        law.probabilities().to_vec(),
        law.snapshot_identity(),
        law.tolerance(),
    )
    .unwrap()
}

#[test]
fn absent_mislabelled_and_counted_laws_are_refused_before_a_number() {
    let case = case_named("joint_counterfactual_two_source_regimes");
    let adapted = adapter::catalog(&case);
    let data = adapter::laws(&case, &adapted);
    let derivation = derive(&case, &case.targets[0]);
    evaluate(&derivation, &data).unwrap();

    // The source experiment at one level is withheld: its factor has no law.
    let kept: Vec<ExactDiscreteLaw> = data
        .laws()
        .iter()
        .filter(|law| {
            !(law.population() == "source"
                && law.interventions().iter().any(|a| a.value.as_f64() == Some(1.0)))
        })
        .cloned()
        .collect();
    assert!(kept.len() < data.laws().len());
    let thin = ExactTransportData::try_new(kept, 1 << 16).unwrap();
    let error = evaluate(&derivation, &thin).unwrap_err();
    assert_eq!(refusal_code(&error), "transport_missing_provider");

    // A source law relabelled as the target's: the catalog does not bind that
    // population to that regime.
    let mut laws: Vec<ExactDiscreteLaw> = data.laws().to_vec();
    let first_source = laws.iter().position(|l| l.population() == "source").unwrap();
    laws[first_source] = rebuilt(&laws[first_source], "target");
    let swapped = ExactTransportData::try_new(laws, 1 << 16).unwrap();
    let error = evaluate(&derivation, &swapped).unwrap_err();
    assert!(matches!(error, EstimationError::Refused { .. }), "{error:?}");

    // An empirical plug-in is not licensed: sampled inference is a separate route.
    let mut laws: Vec<ExactDiscreteLaw> = data.laws().to_vec();
    let law = &laws[0];
    laws[0] = ExactDiscreteLaw::try_empirical(
        law.population(),
        law.regime(),
        law.interventions().to_vec(),
        law.axes().to_vec(),
        law.probabilities().to_vec(),
        law.snapshot_identity(),
        law.tolerance(),
    )
    .unwrap();
    let empirical = ExactTransportData::try_new(laws, 1 << 16).unwrap();
    let error = evaluate(&derivation, &empirical).unwrap_err();
    assert_eq!(refusal_code(&error), "cell_not_licensed");
}

#[test]
fn evaluation_is_deterministic_and_invariant_to_provider_order() {
    let case = case_named("effect_of_treatment_on_treated_with_selection");
    let adapted = adapter::catalog(&case);
    let data = adapter::laws(&case, &adapted);
    let derivation = derive(&case, &case.targets[0]);
    let first = evaluate(&derivation, &data).unwrap();
    let again = evaluate(&derivation, &data).unwrap();
    assert_eq!(first.to_bits(), again.to_bits());
    let mut laws = data.laws().to_vec();
    laws.reverse();
    let reversed = ExactTransportData::try_new(laws, 1 << 16).unwrap();
    let permuted = evaluate(&derivation, &reversed).unwrap();
    assert_eq!(first.to_bits(), permuted.to_bits());
    let truth = case.targets[0].expected_value.unwrap();
    assert!((first - truth).abs() <= TOLERANCE, "{first} vs {truth}");
}

#[test]
fn lowering_tags_every_leaf_with_its_population_and_regime() {
    let case = case_named("transportable_latent_district");
    let derivation = derive(&case, &case.targets[0]);
    let lowered = lower_counterfactual_transport_items(&derivation);
    assert_eq!(lowered.len(), derivation.items.len());
    let mut leaves = 0;
    for item in &lowered {
        for index in 0..item.arena.len() {
            let id = antecedent_expr::ExprId::from_raw(u32::try_from(index).unwrap());
            if let ExprNode::Distribution { population, regime, .. } = item.arena.node(id) {
                leaves += 1;
                let name = item.arena.population(*population);
                let regime = regime.expect("a lowered leaf cites a regime");
                assert!(
                    derivation
                        .cited
                        .iter()
                        .any(|c| c.regime == regime && c.population.as_ref() == name),
                    "leaf of {name} regime {regime:?} is a cited regime"
                );
            }
        }
    }
    assert!(leaves >= 2, "the formula reads at least the source and target laws");
}
