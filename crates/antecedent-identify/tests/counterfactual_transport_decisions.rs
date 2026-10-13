//! Decisions of general finite counterfactual transport on the named
//! finite-SCM fixtures.
//!
//! Every expected class comes from the independent oracle's fixtures
//! (`counterfactual_transport_cases`), whose derivations are written out by
//! hand. Obstructions are additionally confirmed by the fixtures' own
//! alternative model families through the oracle, so a witness the identifier
//! reports is never the only evidence for non-transportability.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "support/counterfactual_transport_adapter.rs"]
mod counterfactual_transport_adapter;
#[path = "support/counterfactual_transport_cases.rs"]
mod counterfactual_transport_cases;
#[path = "support/finite_scm_oracle.rs"]
mod finite_scm_oracle;

use std::sync::Arc;

use antecedent_core::{
    CounterfactualEvent, CounterfactualEventQuery, ExecutionContext, ExogenousCoupling,
    SearchLimits, SearchStop, VariableId, WorldId, WorldSpec,
};
use antecedent_identify::{
    COUNTERFACTUAL_TRANSPORT_DEFAULT_LIMITS, CounterfactualFunctional,
    CounterfactualTransportDecision, CounterfactualTransportProblem, CounterfactualTransportQuery,
    CounterfactualTransportSource, counterfactual_transport_refusal,
    identify_counterfactual_transport,
};
use counterfactual_transport_adapter as adapter;
use counterfactual_transport_cases::{Case, ExpectedClass, Target, all_cases};

fn controllable_for(case: &Case) -> Vec<&'static str> {
    if case.name == "missing_interventional_evidence" {
        // The source can experiment on X; no such regime is supplied.
        vec!["X"]
    } else {
        adapter::evidence_controllable(case)
    }
}

fn decide_with(
    case: &Case,
    target: &Target,
    controllable: &[&str],
    limits: SearchLimits,
    ctx: &ExecutionContext,
) -> CounterfactualTransportDecision {
    let problem = adapter::problem(case);
    let catalog = adapter::catalog(case);
    let sources = adapter::sources(case, controllable);
    let query = adapter::query(case, &target.query, sources);
    identify_counterfactual_transport(&problem, &query, &catalog.catalog, limits, ctx)
        .expect("a fixture is a valid request")
}

fn decide(case: &Case, target: &Target) -> CounterfactualTransportDecision {
    decide_with(
        case,
        target,
        &controllable_for(case),
        COUNTERFACTUAL_TRANSPORT_DEFAULT_LIMITS,
        &ExecutionContext::for_tests(1),
    )
}

/// The targets a fixture asks about in the *target* population (the zero-support
/// fixture also states one query about the source).
fn target_queries(case: &Case) -> Vec<&Target> {
    case.targets.iter().filter(|t| t.query.population == case.target_population).collect()
}

#[test]
fn named_cases_decide_as_the_theory_says() {
    for case in all_cases() {
        for target in target_queries(&case) {
            let decision = decide(&case, target);
            let label = format!("{} / {}", case.name, target.label);
            match target.class {
                ExpectedClass::Identified | ExpectedClass::UndefinedZeroSupport => {
                    let derivation = decision.derivation().unwrap_or_else(|| {
                        panic!("{label}: expected a derivation, got {decision:?}")
                    });
                    derivation
                        .replay_checked(&ExecutionContext::for_tests(1))
                        .unwrap_or_else(|e| panic!("{label}: replay failed: {e}"));
                }
                ExpectedClass::NotIdentifiable | ExpectedClass::NotTransportable => {
                    let CounterfactualTransportDecision::ProvenNonTransportable(obstruction) =
                        &decision
                    else {
                        panic!("{label}: expected an obstruction, got {decision:?}");
                    };
                    let check = obstruction.verify().unwrap_or_else(|e| panic!("{label}: {e}"));
                    assert_ne!(check.first_value, check.second_value, "{label}");
                    assert_eq!(decision.reason_code(), Some("transport_proven_non_transportable"));
                }
                ExpectedClass::EvidenceInsufficient => {
                    let CounterfactualTransportDecision::MissingEvidence(missing) = &decision
                    else {
                        panic!("{label}: expected missing evidence, got {decision:?}");
                    };
                    assert_eq!(decision.reason_code(), Some("transport_missing_evidence"));
                    assert!(
                        missing.missing.iter().any(|m| m.population.as_ref() == "source"
                            && m.interventions == vec![adapter::variable_id(&case, "X")]),
                        "{label}: the missing regime is do(X) in the source: {missing:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn obstructions_agree_with_the_fixtures_own_alternative_models() {
    for case in all_cases() {
        let Some(alternative) = &case.alternative else { continue };
        let target = &case.targets[0];
        // The fixture's own pair of model families is a witness for the same query.
        let report = finite_scm_oracle::verify_non_identifiability_witness(
            &case.populations,
            alternative,
            &case.evidence,
            &target.query,
            1e-15,
            1e-6,
        )
        .unwrap_or_else(|e| panic!("{}: {e}", case.name));
        assert!(report.gap > 1e-6, "{}", case.name);
        // The identifier does not claim an obstruction the fixture does not have:
        // it either proves one or reports missing evidence.
        let decision = decide(&case, target);
        assert!(
            matches!(
                decision,
                CounterfactualTransportDecision::ProvenNonTransportable(_)
                    | CounterfactualTransportDecision::MissingEvidence(_)
            ),
            "{}: {decision:?}",
            case.name
        );
    }
}

#[test]
fn identified_derivations_cite_catalog_regimes_and_use_the_binary_complement() {
    let cases = all_cases();
    let case = cases
        .iter()
        .find(|c| c.name == "joint_counterfactual_two_source_regimes")
        .expect("fixture exists");
    let decision = decide(case, &case.targets[0]);
    let derivation = decision.derivation().expect("identified");
    // P(Y_1 = 1, X = 0) conflicts as one district (X is read at 0 and set to 1):
    // the binary consistency complement answers it.
    assert!(
        matches!(derivation.numerator, CounterfactualFunctional::Complement { .. }),
        "{:?}",
        derivation.numerator
    );
    let adapted = adapter::catalog(case);
    let source_experiment = adapted.regime_of
        [&adapter::RegimeKey { population: "source".to_owned(), intervened: vec!["X".to_owned()] }];
    let source_observational = adapted.regime_of
        [&adapter::RegimeKey { population: "source".to_owned(), intervened: vec![] }];
    let target_marginal = adapted.regime_of
        [&adapter::RegimeKey { population: "target".to_owned(), intervened: vec![] }];
    let cited: Vec<_> = derivation.cited.iter().map(|c| c.regime).collect();
    for regime in [source_experiment, source_observational, target_marginal] {
        assert!(cited.contains(&regime), "regime {regime:?} is cited: {cited:?}");
    }
    assert!(derivation.search.operations_consumed > 0);
    assert_eq!(
        derivation.search.stages.first().map(String::as_str),
        Some("stage:supplied_regimes")
    );
}

#[test]
fn conditional_reduction_moves_an_independent_conditioning_atom() {
    let cases = all_cases();
    let case = cases.iter().find(|c| c.name == "zero_support_conditioning_event").unwrap();
    let target = &case.targets[1];
    let decision = decide(case, target);
    let derivation = decision.derivation().expect("identified");
    // Y_{x=0} does not depend on the factual X: the conditioning moves into do(X=1).
    assert!(!derivation.moved.is_empty());
    // The original conditioning mass P(X = 1) is derivable from the target marginal
    // and stays a positivity guard.
    assert!(derivation.positivity_guard.is_some());
    assert!(derivation.conditioning_mass_checked);
}

#[test]
fn dependent_conditioning_is_not_reduced_when_the_event_depends_on_it() {
    let cases = all_cases();
    let case =
        cases.iter().find(|c| c.name == "effect_of_treatment_on_treated_with_selection").unwrap();
    let decision = decide(case, &case.targets[0]);
    let derivation = decision.derivation().expect("identified");
    // X <-> Y: the event Y_{x=1} is dependent on the factual X; no atom moves.
    assert!(derivation.moved.is_empty());
    assert!(derivation.positivity_guard.is_none());
    assert!(derivation.conditioning_mass_checked);
    assert!(
        !matches!(derivation.denominator, CounterfactualFunctional::Zero),
        "{:?}",
        derivation.denominator
    );
}

#[test]
fn decisions_are_invariant_to_declaration_order() {
    let cases = all_cases();
    let case = cases.iter().find(|c| c.name == "joint_counterfactual_two_source_regimes").unwrap();
    let target = &case.targets[0];
    let ctx = ExecutionContext::for_tests(1);
    let straight = decide(case, target);
    // Variables declared in reverse, regimes and sources in reverse.
    let mut variables = adapter::declared_levels(case);
    variables.reverse();
    let directed: Vec<(VariableId, VariableId)> = case
        .graph
        .directed
        .iter()
        .rev()
        .map(|(a, b)| (adapter::variable_id(case, a), adapter::variable_id(case, b)))
        .collect();
    let bidirected: Vec<(VariableId, VariableId)> = case
        .graph
        .bidirected
        .iter()
        .map(|(a, b)| (adapter::variable_id(case, b), adapter::variable_id(case, a)))
        .collect();
    let problem = CounterfactualTransportProblem::new(&variables, &directed, &bidirected).unwrap();
    let adapted = adapter::catalog(case);
    let mut regimes = adapted.catalog.regimes.to_vec();
    regimes.reverse();
    let catalog = antecedent_core::EvidenceCatalog::try_new(
        Vec::<antecedent_core::Environment>::new(),
        regimes,
        Vec::<antecedent_core::RegimeBinding>::new(),
        None,
    )
    .unwrap();
    let mut sources = adapter::sources(case, &controllable_for(case));
    sources.reverse();
    let query = adapter::query(case, &target.query, sources);
    let reordered = identify_counterfactual_transport(
        &problem,
        &query,
        &catalog,
        COUNTERFACTUAL_TRANSPORT_DEFAULT_LIMITS,
        &ctx,
    )
    .unwrap();
    let (a, b) = (straight.derivation().unwrap(), reordered.derivation().unwrap());
    assert_eq!(a.canonical_text(), b.canonical_text());
    assert_eq!(a.numerator, b.numerator);
}

#[test]
fn a_mutated_derivation_or_witness_does_not_replay() {
    let cases = all_cases();
    let ett =
        cases.iter().find(|c| c.name == "effect_of_treatment_on_treated_with_selection").unwrap();
    let decision = decide(ett, &ett.targets[0]);
    let derivation = decision.derivation().unwrap();
    let ctx = ExecutionContext::for_tests(1);
    derivation.replay_checked(&ctx).unwrap();
    let mut mutated = derivation.clone();
    mutated.numerator = CounterfactualFunctional::Zero;
    assert!(mutated.replay_checked(&ctx).is_err());
    let mut mutated = derivation.clone();
    mutated.populations = vec![Arc::from("other")];
    assert!(mutated.replay_checked(&ctx).is_err());

    let outcome = cases.iter().find(|c| c.name == "selection_on_outcome_mechanism").unwrap();
    let decision = decide(outcome, &outcome.targets[0]);
    let CounterfactualTransportDecision::ProvenNonTransportable(obstruction) = decision else {
        panic!("expected an obstruction");
    };
    obstruction.verify().unwrap();
    // A witness whose law is changed no longer agrees on the declared family.
    let mut tampered = (*obstruction).clone();
    tampered.witness.second = tampered.witness.first.clone();
    assert!(tampered.verify().is_err(), "two equal models do not separate the query");
    let mut tampered = (*obstruction).clone();
    tampered.witness.second_value = tampered.witness.first_value.clone();
    assert!(tampered.verify().is_err(), "equal recorded values do not separate the query");
    let mut tampered = (*obstruction).clone();
    tampered.witness.clique.reverse();
    if tampered.witness.clique.len() > 1 {
        assert!(tampered.verify().is_err());
    }
    let error = tampered_error_code(&obstruction);
    assert_eq!(error, ("transport_not_certified", "counterfactual_transport.invalid_witness"));
}

fn tampered_error_code(
    obstruction: &antecedent_identify::CounterfactualTransportObstruction,
) -> (&'static str, &'static str) {
    let mut tampered = obstruction.clone();
    "not_a_baseline".clone_into(&mut tampered.witness.baseline);
    let error = tampered.verify().unwrap_err();
    counterfactual_transport_refusal(&error).expect("an owned refusal")
}

#[test]
fn selection_missing_factors_and_wrong_labels_never_identify() {
    let cases = all_cases();
    let case = cases.iter().find(|c| c.name == "joint_counterfactual_two_source_regimes").unwrap();
    let target = &case.targets[0];
    let ctx = ExecutionContext::for_tests(1);
    let limits = COUNTERFACTUAL_TRANSPORT_DEFAULT_LIMITS;

    // Selection also on the outcome: the source experiment cannot be borrowed.
    let problem = adapter::problem(case);
    let adapted = adapter::catalog(case);
    let mut sources = adapter::sources(case, &controllable_for(case));
    sources[0] = CounterfactualTransportSource {
        selection_targets: vec![adapter::variable_id(case, "Z"), adapter::variable_id(case, "Y")]
            .into(),
        ..sources[0].clone()
    };
    let query = adapter::query(case, &target.query, sources);
    let decision =
        identify_counterfactual_transport(&problem, &query, &adapted.catalog, limits, &ctx)
            .unwrap();
    assert!(decision.derivation().is_none(), "{decision:?}");

    // The source observational law withheld: declared but absent.
    let specs: Vec<_> = case
        .evidence
        .iter()
        .filter(|s| !(s.population == "source" && s.regime.is_empty()))
        .cloned()
        .collect();
    let thin = adapter::catalog_of(case, &specs);
    let sources = adapter::sources(case, &controllable_for(case));
    let query = adapter::query(case, &target.query, sources);
    let decision =
        identify_counterfactual_transport(&problem, &query, &thin.catalog, limits, &ctx).unwrap();
    let CounterfactualTransportDecision::MissingEvidence(missing) = decision else {
        panic!("expected missing evidence, got {decision:?}");
    };
    // The declared observational law of either population licenses the factor
    // P(X, Y | Z); the target has no selection target on {X, Y}, so the first
    // derivation found cites the target's full observational law. Both are
    // sound; what must be named is an observational regime, never a do(.) one.
    assert!(!missing.missing.is_empty());
    assert!(missing.missing.iter().all(|m| m.interventions.is_empty()), "{missing:?}");

    // A population label that is neither the target nor a declared source.
    let sources = adapter::sources(case, &controllable_for(case));
    let mut query = adapter::query(case, &target.query, sources);
    query.target = Arc::from("elsewhere");
    let error = identify_counterfactual_transport(&problem, &query, &adapted.catalog, limits, &ctx)
        .unwrap_err();
    assert_eq!(
        counterfactual_transport_refusal(&error),
        Some(("invalid_argument", "counterfactual_transport.invalid_catalog"))
    );

    // A coupling this contract does not model is out of scope, not an error.
    let abduced = CounterfactualEventQuery::new(
        vec![WorldSpec::new([], []).unwrap()],
        ExogenousCoupling::SharedAbducedExogenous,
        [CounterfactualEvent::new(WorldId::new(0), adapter::variable_id(case, "Y"), 1.0).unwrap()],
        [],
    )
    .unwrap();
    let query = CounterfactualTransportQuery {
        event: abduced,
        target: Arc::from("target"),
        sources: adapter::sources(case, &controllable_for(case)).into(),
    };
    let decision =
        identify_counterfactual_transport(&problem, &query, &adapted.catalog, limits, &ctx)
            .unwrap();
    assert!(matches!(decision, CounterfactualTransportDecision::UnsupportedScope(_)));
    assert_eq!(decision.detail_code(), Some("counterfactual_transport.coupling_unsupported"));
}

#[test]
fn budget_cancellation_and_bounds_are_typed_stops_not_verdicts() {
    let cases = all_cases();
    let case = cases.iter().find(|c| c.name == "nonlinear_mediation_covariate_shift").unwrap();
    let target = &case.targets[0];
    let ctx = ExecutionContext::for_tests(1);
    // One operation cannot even close the evidence.
    let starved = decide_with(
        case,
        target,
        &controllable_for(case),
        SearchLimits { operations: 1, depth: 8 },
        &ctx,
    );
    let CounterfactualTransportDecision::Exhausted(receipt) = &starved else {
        panic!("expected an exhausted budget, got {starved:?}");
    };
    assert_eq!(receipt.stop, SearchStop::Operations);
    assert_eq!(starved.reason_code(), Some("transport_budget_cancel"));
    assert_eq!(starved.detail_code(), Some("counterfactual_transport.budget"));
    // Cancellation before the search is a stop with a receipt.
    let cancelled_ctx = ExecutionContext::for_tests(1);
    cancelled_ctx.cancellation.cancel();
    let cancelled = decide_with(
        case,
        target,
        &controllable_for(case),
        COUNTERFACTUAL_TRANSPORT_DEFAULT_LIMITS,
        &cancelled_ctx,
    );
    let CounterfactualTransportDecision::Exhausted(receipt) = &cancelled else {
        panic!("expected a cancelled search, got {cancelled:?}");
    };
    assert_eq!(receipt.stop, SearchStop::Cancelled);
    // Limits above the contract's maxima are refused by bound.
    let problem = adapter::problem(case);
    let adapted = adapter::catalog(case);
    let query = adapter::query(case, &target.query, adapter::sources(case, &[]));
    let error = identify_counterfactual_transport(
        &problem,
        &query,
        &adapted.catalog,
        SearchLimits { operations: usize::MAX, depth: 8 },
        &ctx,
    )
    .unwrap_err();
    assert_eq!(
        counterfactual_transport_refusal(&error),
        Some(("route_not_supported", "counterfactual_transport.bounds_exceeded"))
    );
    // Nine variables are outside the frozen contract.
    let nine: Vec<(VariableId, Vec<f64>)> =
        (0..9).map(|i| (VariableId::from_raw(i), vec![0.0, 1.0])).collect();
    let error = CounterfactualTransportProblem::new(&nine, &[], &[]).unwrap_err();
    assert_eq!(
        counterfactual_transport_refusal(&error),
        Some(("route_not_supported", "counterfactual_transport.bounds_exceeded"))
    );
    // A directed cycle is outside the acyclic contract.
    let two: Vec<(VariableId, Vec<f64>)> =
        (0..2).map(|i| (VariableId::from_raw(i), vec![0.0, 1.0])).collect();
    let (a, b) = (VariableId::from_raw(0), VariableId::from_raw(1));
    let error = CounterfactualTransportProblem::new(&two, &[(a, b), (b, a)], &[]).unwrap_err();
    assert_eq!(
        counterfactual_transport_refusal(&error),
        Some(("route_not_supported", "counterfactual_transport.graph_outside_contract"))
    );
}
