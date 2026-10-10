//! Structural policies on enumerated per-structure laws.

use antecedent_core::{QuantityRole, ScientificQuantity};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, HardConstraint,
    StructuralPolicy, UtilityExpr,
};
use antecedent_design::decision_structural::{
    AtomEvidence, AtomStatus, StructuralAtom, StructuralError, StructuralVerdict,
    evaluate_structural,
};
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io::quantity_wire::DistributionMeaningWire;

fn quantity(variable: &str, regime: &str, units: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: variable.into(),
        role: QuantityRole::Outcome,
        units: units.into(),
        population_id: "target".into(),
        regime_id: regime.into(),
        horizon: 0,
        functional_id: "outcome".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

/// One structure's exact law: action A and B payoffs, two equally likely rows.
fn law(a: f64, b: f64) -> Box<DistributionArtifact> {
    let columns = [quantity("a", "do(a=1)", "units"), quantity("b", "do(a=0)", "units")];
    let identity = DistributionIdentity::new(
        DistributionMeaningWire::InterventionalPredictive,
        &columns,
        DrawAlignment::Joint,
        DistributionProvenance {
            source_id: "structure".into(),
            provider_id: "exact-law".into(),
            rng_id: "deterministic_exact".into(),
            snapshot_id: "enumeration".into(),
            causal_contract_id: "checked".into(),
        },
    )
    .unwrap();
    Box::new(
        DistributionArtifact::new(
            DistributionMetadata {
                version: 1,
                identity,
                axes: ["draw".into(), "quantity".into()],
                shape: [2, 2],
                weights: None,
                supported: None,
                calibration: DistributionCalibration::Exact,
                trust: DistributionTrust::Unverified,
                legacy_posterior: None,
                legacy_bindings: None,
            },
            vec![a, b, a, b],
        )
        .unwrap(),
    )
}

fn atom(id: &str, p: Option<f64>, evidence: AtomEvidence) -> StructuralAtom {
    StructuralAtom { id: id.into(), probability: p, evidence }
}

fn evaluated(id: &str, p: Option<f64>, a: f64, b: f64) -> StructuralAtom {
    atom(id, p, AtomEvidence::Evaluated(law(a, b)))
}

fn contract(policy: StructuralPolicy) -> DecisionContract {
    let action = |id: &str, variable: &str, regime: &str| DecisionAction {
        id: id.into(),
        kind: ActionKind::Intervention,
        inputs: vec![quantity(variable, regime, "units")],
        utility: UtilityExpr::Input(0),
    };
    DecisionContract {
        actions: vec![action("A", "a", "do(a=1)"), action("B", "b", "do(a=0)")],
        utility_units: "units".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: policy,
    }
}

fn near(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-12
}

#[test]
fn invariant_best_action_requires_agreement_in_every_structure() {
    let c = contract(StructuralPolicy::RequireInvariantBestAction);
    let agree = [evaluated("s1", None, 5.0, 3.0), evaluated("s3", None, 6.0, 2.0)];
    let result = evaluate_structural(&c, &agree).unwrap();
    assert_eq!(result.verdict, StructuralVerdict::InvariantBest("A".into()));
    assert_eq!(result.actions[0].range, Some((5.0, 6.0)));

    let conflict = [
        evaluated("s1", None, 5.0, 3.0),
        evaluated("s2", None, 1.0, 4.0),
        evaluated("s3", None, 6.0, 2.0),
    ];
    let result = evaluate_structural(&c, &conflict).unwrap();
    assert_eq!(
        result.verdict,
        StructuralVerdict::NoInvariantBest(vec![
            ("s1".into(), vec!["A".into()]),
            ("s2".into(), vec!["B".into()]),
            ("s3".into(), vec!["A".into()]),
        ])
    );
    // The range keeps the structural spread rather than averaging it away.
    assert_eq!(result.actions[0].range, Some((1.0, 6.0)));
    assert!(result.actions[0].weighted_value.is_none());
}

#[test]
fn maximin_ranks_by_the_worst_structure() {
    let c = contract(StructuralPolicy::Maximin);
    let atoms = [
        evaluated("s1", None, 5.0, 3.0),
        evaluated("s2", None, 1.0, 4.0),
        evaluated("s3", None, 6.0, 2.0),
    ];
    // Worst case: A = 1, B = 2.
    let result = evaluate_structural(&c, &atoms).unwrap();
    assert_eq!(result.verdict, StructuralVerdict::WorstCaseChoice("B".into()));

    // The criteria form of the same idea ignores the declared policy.
    let mut criterion = contract(StructuralPolicy::ReportOnly);
    criterion.criterion = DecisionCriterion::MaximinOverStructures;
    let result = evaluate_structural(&criterion, &atoms).unwrap();
    assert_eq!(result.verdict, StructuralVerdict::WorstCaseChoice("B".into()));

    // Minimax over an identified set reads utilities as losses: worst loss
    // A = 6, B = 4, so B.
    let mut minimax = contract(StructuralPolicy::ReportOnly);
    minimax.criterion = DecisionCriterion::MinimaxOverIdentifiedSet;
    let result = evaluate_structural(&minimax, &atoms).unwrap();
    assert_eq!(result.verdict, StructuralVerdict::WorstCaseChoice("B".into()));
    // With A worse everywhere but one structure, the two criteria can differ.
    let skew = [evaluated("s1", None, 9.0, 3.0), evaluated("s2", None, 8.0, 7.0)];
    let loss_choice = evaluate_structural(&minimax, &skew).unwrap().verdict;
    let utility_choice = evaluate_structural(&criterion, &skew).unwrap().verdict;
    assert_eq!(loss_choice, StructuralVerdict::WorstCaseChoice("B".into()));
    assert_eq!(utility_choice, StructuralVerdict::WorstCaseChoice("A".into()));
}

#[test]
fn bayes_weights_by_genuine_probabilities_and_keeps_unresolved_mass() {
    let c = contract(StructuralPolicy::BayesOverStructures);
    let atoms = [
        evaluated("s1", Some(0.5), 5.0, 3.0),
        evaluated("s2", Some(0.3), 1.0, 4.0),
        evaluated("s3", Some(0.2), 6.0, 2.0),
    ];
    let result = evaluate_structural(&c, &atoms).unwrap();
    // A: 2.5 + 0.3 + 1.2 = 4.0; B: 1.5 + 1.2 + 0.4 = 3.1.
    assert!(near(result.actions[0].weighted_value.unwrap(), 4.0));
    assert!(near(result.actions[1].weighted_value.unwrap(), 3.1));
    assert!(near(result.actions[0].mass_where_best.unwrap(), 0.7));
    assert!(near(result.actions[1].mass_where_best.unwrap(), 0.3));
    assert!(near(result.evaluated_mass.unwrap(), 1.0));
    assert!(matches!(
        result.verdict,
        StructuralVerdict::BayesChoice { ref action, .. } if action == "A"
    ));

    // An unidentified structure keeps its mass; the others are not renormalized.
    let partial = [
        evaluated("s1", Some(0.5), 5.0, 3.0),
        evaluated("s2", Some(0.3), 1.0, 4.0),
        atom("s3", Some(0.2), AtomEvidence::Unidentified),
    ];
    let result = evaluate_structural(&c, &partial).unwrap();
    assert!(near(result.unidentified_mass.unwrap(), 0.2));
    assert!(near(result.evaluated_mass.unwrap(), 0.8));
    assert!(near(result.actions[0].weighted_value.unwrap(), 2.8));
    assert!(near(result.actions[1].weighted_value.unwrap(), 2.7));
    assert!(matches!(result.verdict, StructuralVerdict::InsufficientScience(_)));

    // Completion counts are not probabilities.
    let counts = [evaluated("s1", None, 5.0, 3.0), evaluated("s2", None, 1.0, 4.0)];
    let error = evaluate_structural(&c, &counts).unwrap_err();
    assert_eq!(error, StructuralError::ProbabilitiesRequired);
    assert_eq!(error.reason_code(), "decision_contract_unsatisfied");
}

#[test]
fn unresolved_structures_block_claims_that_need_them() {
    let atoms = [evaluated("s1", None, 5.0, 3.0), atom("s2", None, AtomEvidence::Unidentified)];
    for policy in [StructuralPolicy::RequireInvariantBestAction, StructuralPolicy::Maximin] {
        let result = evaluate_structural(&contract(policy), &atoms).unwrap();
        assert!(matches!(result.verdict, StructuralVerdict::InsufficientScience(_)));
    }
    // A structure whose evidence cannot answer the contract is unevaluated with
    // its registered reason, not silently dropped.
    let mut mismatched = contract(StructuralPolicy::ReportOnly);
    mismatched.actions[0].inputs[0].units = "kg".into();
    let result = evaluate_structural(&mismatched, &[evaluated("s1", None, 5.0, 3.0)]).unwrap();
    assert_eq!(
        result.atoms[0].status,
        AtomStatus::Unevaluated("quantity_semantics_mismatch".into())
    );
    let none = evaluate_structural(
        &contract(StructuralPolicy::Maximin),
        &[atom("s1", None, AtomEvidence::Unevaluated("budget".into()))],
    )
    .unwrap();
    assert!(matches!(none.verdict, StructuralVerdict::InsufficientScience(_)));
    let report = evaluate_structural(&contract(StructuralPolicy::ReportOnly), &atoms).unwrap();
    assert_eq!(report.verdict, StructuralVerdict::ReportOnly);
    assert_eq!(report.atoms[1].status, AtomStatus::Unidentified);
}

#[test]
fn a_constraint_failing_in_any_structure_excludes_the_action_there() {
    let mut c = contract(StructuralPolicy::Maximin);
    c.constraints.push(HardConstraint {
        id: "cap".into(),
        expr: UtilityExpr::Input(0),
        bound: 4.5,
        min_probability: 1.0,
        units: "units".into(),
        applies_to: vec!["A".into()],
    });
    // A pays 5 in s1 (violates the cap) and 1 in s2.
    let atoms = [evaluated("s1", None, 5.0, 3.0), evaluated("s2", None, 1.0, 4.0)];
    let result = evaluate_structural(&c, &atoms).unwrap();
    assert_eq!(result.actions[0].excluded_in, vec!["s1".to_owned()]);
    assert_eq!(result.verdict, StructuralVerdict::WorstCaseChoice("B".into()));

    c.constraints[0].applies_to.clear();
    c.constraints[0].bound = -1.0;
    let result = evaluate_structural(&c, &atoms).unwrap();
    assert_eq!(result.verdict, StructuralVerdict::NoAdmissibleAction);
}

#[test]
fn malformed_atoms_and_probabilities_refuse() {
    let c = contract(StructuralPolicy::ReportOnly);
    assert_eq!(evaluate_structural(&c, &[]).unwrap_err(), StructuralError::InvalidAtoms);
    let dup = [evaluated("s", None, 1.0, 2.0), evaluated("s", None, 1.0, 2.0)];
    assert_eq!(evaluate_structural(&c, &dup).unwrap_err(), StructuralError::InvalidAtoms);
    let some = [evaluated("a", Some(0.5), 1.0, 2.0), evaluated("b", None, 1.0, 2.0)];
    assert_eq!(evaluate_structural(&c, &some).unwrap_err(), StructuralError::InvalidProbabilities);
    let over = [evaluated("a", Some(0.7), 1.0, 2.0), evaluated("b", Some(0.7), 1.0, 2.0)];
    assert_eq!(evaluate_structural(&c, &over).unwrap_err(), StructuralError::InvalidProbabilities);
}

#[test]
fn missing_probability_mass_cannot_choose_an_action() {
    let c = contract(StructuralPolicy::BayesOverStructures);
    let partial = [evaluated("known", Some(0.8), 5.0, 3.0)];
    let result = evaluate_structural(&c, &partial).unwrap();
    assert!(near(result.unevaluated_mass.unwrap(), 0.2));
    assert!(matches!(result.verdict, StructuralVerdict::InsufficientScience(_)));
    let tiny = [
        evaluated("known", Some(1.0 - 1e-12), 5.0, 3.0),
        atom("unknown", Some(1e-12), AtomEvidence::Unevaluated("budget".into())),
    ];
    assert!(matches!(
        evaluate_structural(&c, &tiny).unwrap().verdict,
        StructuralVerdict::InsufficientScience(_)
    ));
    // Unknown payoffs on the missing mass can reverse the observed preference.
    let mut completed = partial.to_vec();
    completed.push(evaluated("missing", Some(0.2), 0.0, 100.0));
    assert!(
        matches!(evaluate_structural(&c, &completed).unwrap().verdict, StructuralVerdict::BayesChoice { action, .. } if action == "B")
    );
}

#[test]
fn bayes_never_averages_component_quantiles_or_maximum_regrets() {
    let atoms = [evaluated("s1", Some(0.5), 0.0, 3.0), evaluated("s2", Some(0.5), 10.0, 3.0)];
    for criterion in [DecisionCriterion::Quantile { p: 0.5 }, DecisionCriterion::Regret] {
        let mut c = contract(StructuralPolicy::BayesOverStructures);
        c.criterion = criterion;
        assert!(evaluate_structural(&c, &atoms).is_err());
    }
}

#[test]
fn zero_probability_structure_cannot_exclude_a_bayes_action() {
    let mut c = contract(StructuralPolicy::BayesOverStructures);
    c.constraints.push(HardConstraint {
        id: "cap".into(),
        applies_to: vec!["A".into()],
        expr: UtilityExpr::Input(0),
        bound: 10.0,
        min_probability: 1.0,
        units: "units".into(),
    });
    let atoms =
        [evaluated("supported", Some(1.0), 5.0, 3.0), evaluated("null", Some(0.0), 100.0, 0.0)];
    let r = evaluate_structural(&c, &atoms).unwrap();
    assert!(matches!(r.verdict, StructuralVerdict::BayesChoice { action, .. } if action == "A"));
}
