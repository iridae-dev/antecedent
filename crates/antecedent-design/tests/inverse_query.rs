//! Known-truth F7 inverse decision queries on hand-enumerated finite laws.
//!
//! Ordered actions `a0, a1, a2` have outcome laws Bernoulli(1/4, 1/2, 3/4),
//! realized as four equiprobable aligned rows:
//!
//! ```text
//! row   y(a0)  y(a1)  y(a2)
//!  0      1      1      1
//!  1      0      1      1
//!  2      0      0      1
//!  3      0      0      0
//! ```
//!
//! `E[Y]` and `P(Y >= 1)` are `1/4, 1/2, 3/4`. Under the engine's left-inverse
//! quantile the median `inf { x : F(x) >= 1/2 }` is `0, 0, 1`: `F_a1(0) = 1/2`
//! reaches the level exactly, so a1's median is 0, not 1/2 or 1.

use antecedent_core::{QuantityRole, ScientificQuantity};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, DecisionFunctional,
    StructuralPolicy, Tail, UtilityExpr,
};
use antecedent_design::decision_eval::{DecisionEvalError, MeanSource, evaluate_functional};
use antecedent_design::decision_structural::{AtomEvidence, StructuralAtom};
use antecedent_design::inverse_query::{
    Comparison, EnumeratedForwardValue, ExistenceClaim, FeasibilityStatus, ForwardClaim,
    ForwardEvidence, GridScope, IdentifiedSet, IntervalRegion, InverseConstraint, InverseQuery,
    InverseQueryError, SelectionOutcome, SelectionRule, evaluate_inverse_query,
    finite_enumeration_baseline,
};
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io::quantity_wire::DistributionMeaningWire;

use FeasibilityStatus::{
    Feasible, Infeasible, StructurallyAmbiguous, Unevaluated, Unidentified, Unsupported,
};

const REGIMES: [&str; 3] = ["do(a=0)", "do(a=1)", "do(a=2)"];
const IDS: [&str; 3] = ["a0", "a1", "a2"];
const A0: [f64; 4] = [1.0, 0.0, 0.0, 0.0];
const A1: [f64; 4] = [1.0, 1.0, 0.0, 0.0];
const A2: [f64; 4] = [1.0, 1.0, 1.0, 0.0];

fn near(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

fn quantity(variable: &str, regime: &str, functional: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: variable.into(),
        role: QuantityRole::Outcome,
        units: "units".into(),
        population_id: "target".into(),
        regime_id: regime.into(),
        horizon: 0,
        functional_id: functional.into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn artifact(
    columns: &[(ScientificQuantity, Vec<f64>)],
    alignment: DrawAlignment,
    supported: Option<Vec<bool>>,
) -> DistributionArtifact {
    let quantities: Vec<ScientificQuantity> = columns.iter().map(|(q, _)| q.clone()).collect();
    let identity = DistributionIdentity::new(
        DistributionMeaningWire::InterventionalPredictive,
        &quantities,
        alignment,
        DistributionProvenance {
            source_id: "enumerated".into(),
            provider_id: "exact-law".into(),
            rng_id: "deterministic_exact".into(),
            snapshot_id: "enumeration-1".into(),
            causal_contract_id: "checked-contract".into(),
        },
    )
    .unwrap();
    let rows = columns[0].1.len();
    let mut draws = Vec::with_capacity(rows * columns.len());
    for row in 0..rows {
        for (_, values) in columns {
            draws.push(values[row]);
        }
    }
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".into(), "quantity".into()],
            shape: [rows, columns.len()],
            weights: None,
            supported,
            calibration: DistributionCalibration::Exact,
            trust: DistributionTrust::Unverified,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        draws,
    )
    .unwrap()
}

fn law_with(a1: &[f64]) -> DistributionArtifact {
    let columns = [
        (quantity("y", REGIMES[0], "outcome"), A0.to_vec()),
        (quantity("y", REGIMES[1], "outcome"), a1.to_vec()),
        (quantity("y", REGIMES[2], "outcome"), A2.to_vec()),
    ];
    artifact(&columns, DrawAlignment::Joint, None)
}

fn bernoulli_law() -> DistributionArtifact {
    law_with(&A1)
}

fn contract_with(functional: &str) -> DecisionContract {
    DecisionContract {
        actions: IDS
            .iter()
            .zip(REGIMES)
            .map(|(id, regime)| DecisionAction {
                id: (*id).into(),
                kind: ActionKind::Intervention,
                inputs: vec![quantity("y", regime, functional)],
                utility: UtilityExpr::Input(0),
            })
            .collect(),
        utility_units: "units".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    }
}

fn query_on(
    contract: DecisionContract,
    grid: &[&str],
    constraints: Vec<InverseConstraint>,
) -> InverseQuery {
    InverseQuery {
        contract,
        grid_order: grid.iter().map(|id| (*id).to_owned()).collect(),
        grid_scope: GridScope::FiniteEnumeration,
        constraints,
        selection: SelectionRule::FirstInGridOrder,
        tolerance: 1e-12,
        max_evaluations: None,
    }
}

fn query(grid: &[&str], constraints: Vec<InverseConstraint>) -> InverseQuery {
    query_on(contract_with("outcome"), grid, constraints)
}

fn point(law: DistributionArtifact) -> ForwardEvidence {
    ForwardEvidence { point: Some(ForwardClaim::Law(Box::new(law))), ..ForwardEvidence::default() }
}

fn mean_at_least(target: f64) -> InverseConstraint {
    InverseConstraint::TargetMean { target, comparison: Comparison::AtLeast }
}

fn points(result: &antecedent_design::inverse_query::InverseResult) -> Vec<FeasibilityStatus> {
    result.actions.iter().map(|a| a.point.unwrap()).collect()
}

fn atom(id: &str, probability: Option<f64>, law: DistributionArtifact) -> StructuralAtom {
    StructuralAtom { id: id.into(), probability, evidence: AtomEvidence::Evaluated(Box::new(law)) }
}

fn scenarios(atoms: Vec<StructuralAtom>) -> ForwardEvidence {
    ForwardEvidence { scenarios: Some(atoms), ..ForwardEvidence::default() }
}

#[test]
fn f7_forward_inverse_round_trip() {
    // By hand: a1 realizes 1,1,0,0 with weight 1/4 each, so E[Y] = 2/4 = 1/2.
    let law = bernoulli_law();
    let forward = evaluate_functional(
        &contract_with("outcome"),
        "a1",
        DecisionFunctional::ExpectedUtility,
        &law,
    )
    .unwrap();
    assert!(near(forward.value, 0.5));
    // Invert: the only action whose mean is exactly the forward value.
    let constraints = vec![
        InverseConstraint::TargetMean { target: forward.value, comparison: Comparison::AtLeast },
        InverseConstraint::TargetMean { target: forward.value, comparison: Comparison::AtMost },
    ];
    let mut q = query(&IDS, constraints);
    q.selection = SelectionRule::RequireUnique;
    let result = evaluate_inverse_query(&q, &point(law)).unwrap();
    assert_eq!(result.feasible_actions, vec!["a1".to_owned()]);
    assert_eq!(result.selected.as_deref(), Some("a1"));
    assert_eq!(result.selection, SelectionOutcome::Selected);
    assert!(result.selection_certified, "every other action is definitely infeasible");
    assert_eq!(points(&result), vec![Infeasible, Feasible, Infeasible]);
    let values = &result.actions[1].point_values;
    assert!(near(values[0].value.unwrap(), 0.5));
    assert!(values[0].standard_error.is_some_and(|e| e.abs() < 1e-12));
    assert_eq!(result.point_source.as_ref().unwrap().snapshot_id, "enumeration-1");
    assert_eq!(result.contract_identity, q.contract.identity().unwrap());
}

#[test]
fn f7_target_mean_constraint_on_a_finite_law() {
    let result =
        evaluate_inverse_query(&query(&IDS, vec![mean_at_least(0.5)]), &point(bernoulli_law()))
            .unwrap();
    // Means by hand: 1/4, 2/4, 3/4.
    for (action, expected) in result.actions.iter().zip([0.25, 0.5, 0.75]) {
        assert!(near(action.point_values[0].value.unwrap(), expected));
    }
    assert_eq!(points(&result), vec![Infeasible, Feasible, Feasible]);
    assert_eq!(result.selected.as_deref(), Some("a1"), "least action reaching the mean");
    assert!(result.selection_certified);
}

#[test]
fn f7_target_quantile_uses_the_engines_left_inverse() {
    let at_least_one =
        InverseConstraint::TargetQuantile { p: 0.5, target: 1.0, comparison: Comparison::AtLeast };
    let result =
        evaluate_inverse_query(&query(&IDS, vec![at_least_one]), &point(bernoulli_law())).unwrap();
    // Medians by hand: F_a0(0) = 3/4, F_a1(0) = 1/2, F_a2(0) = 1/4 against level 1/2.
    let medians: Vec<f64> =
        result.actions.iter().map(|a| a.point_values[0].value.unwrap()).collect();
    assert_eq!(medians, vec![0.0, 0.0, 1.0]);
    assert_eq!(result.feasible_actions, vec!["a2".to_owned()]);
    assert_eq!(result.selected.as_deref(), Some("a2"));

    let at_most_zero =
        InverseConstraint::TargetQuantile { p: 0.5, target: 0.0, comparison: Comparison::AtMost };
    let result =
        evaluate_inverse_query(&query(&IDS, vec![at_most_zero]), &point(bernoulli_law())).unwrap();
    assert_eq!(points(&result), vec![Feasible, Feasible, Infeasible]);

    // A level that falls exactly on an atom boundary selects that atom: F_a0(0) = 3/4.
    let upper =
        InverseConstraint::TargetQuantile { p: 0.75, target: 1.0, comparison: Comparison::AtLeast };
    let result =
        evaluate_inverse_query(&query(&IDS, vec![upper]), &point(bernoulli_law())).unwrap();
    assert_eq!(points(&result), vec![Infeasible, Feasible, Feasible]);
}

#[test]
fn f7_probability_threshold_constraint_on_a_finite_law() {
    // P(Y >= 1) >= 1/2: probabilities by hand are 1/4, 2/4, 3/4.
    let upper = InverseConstraint::ProbabilityThreshold {
        outcome_threshold: 1.0,
        tail: Tail::Upper,
        probability: 0.5,
        comparison: Comparison::AtLeast,
    };
    let result =
        evaluate_inverse_query(&query(&IDS, vec![upper]), &point(bernoulli_law())).unwrap();
    for (action, expected) in result.actions.iter().zip([0.25, 0.5, 0.75]) {
        assert!(near(action.point_values[0].value.unwrap(), expected));
    }
    assert_eq!(result.feasible_actions, vec!["a1".to_owned(), "a2".to_owned()]);
    assert_eq!(result.selected.as_deref(), Some("a1"), "the frozen example: least action");

    // P(Y <= 0) >= 3/4 by hand: 3/4, 2/4, 1/4, so only a0.
    let lower = InverseConstraint::ProbabilityThreshold {
        outcome_threshold: 0.0,
        tail: Tail::Lower,
        probability: 0.75,
        comparison: Comparison::AtLeast,
    };
    let result =
        evaluate_inverse_query(&query(&IDS, vec![lower]), &point(bernoulli_law())).unwrap();
    assert_eq!(result.feasible_actions, vec!["a0".to_owned()]);
}

/// `risky` has utility `P * Q`; `safe` is the constant 3. On the joint rows
/// `P * Q = 4, 0, 4, 0` so `E = 2`, while the marginal means would give `1.5 * 3 = 4.5`.
fn risky_columns(alignment: DrawAlignment) -> DistributionArtifact {
    let columns = [
        (quantity("p", "do(a=1)", "outcome"), vec![1.0, 3.0, 2.0, 0.0]),
        (quantity("q", "do(a=1)", "outcome"), vec![4.0, 0.0, 2.0, 6.0]),
        (quantity("safe", "do(a=0)", "outcome"), vec![3.0; 4]),
    ];
    artifact(&columns, alignment, None)
}

fn risky_contract() -> DecisionContract {
    DecisionContract {
        actions: vec![
            DecisionAction {
                id: "risky".into(),
                kind: ActionKind::Intervention,
                inputs: vec![
                    quantity("p", "do(a=1)", "outcome"),
                    quantity("q", "do(a=1)", "outcome"),
                ],
                utility: UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Input(1)),
            },
            DecisionAction {
                id: "safe".into(),
                kind: ActionKind::Intervention,
                inputs: vec![quantity("safe", "do(a=0)", "outcome")],
                utility: UtilityExpr::Input(0),
            },
        ],
        utility_units: "units".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    }
}

#[test]
fn f7_nonlinear_utility_inverts_on_an_aligned_joint_law_and_refuses_marginals() {
    let grid = ["risky", "safe"];
    let q = query_on(risky_contract(), &grid, vec![mean_at_least(3.0)]);
    let result = evaluate_inverse_query(&q, &point(risky_columns(DrawAlignment::Joint))).unwrap();
    assert!(near(result.actions[0].point_values[0].value.unwrap(), 2.0), "E[P*Q] from joint rows");
    assert_eq!(points(&result), vec![Infeasible, Feasible]);
    assert_eq!(result.selected.as_deref(), Some("safe"));

    // Median of P*Q: sorted 0,0,4,4 with F(0) = 1/2 reaching the level, so 0.
    let median =
        InverseConstraint::TargetQuantile { p: 0.5, target: 0.0, comparison: Comparison::AtMost };
    let q = query_on(risky_contract(), &grid, vec![median]);
    let result = evaluate_inverse_query(&q, &point(risky_columns(DrawAlignment::Joint))).unwrap();
    assert_eq!(points(&result), vec![Feasible, Infeasible]);

    // The same numbers as independent marginals carry no pairing and refuse.
    let q = query_on(risky_contract(), &grid, vec![mean_at_least(3.0)]);
    let error =
        evaluate_inverse_query(&q, &point(risky_columns(DrawAlignment::IndependentMarginals)))
            .unwrap_err();
    assert!(matches!(error, InverseQueryError::Engine(DecisionEvalError::JointLawRequired { .. })));
    let refusal = error.to_refusal();
    assert_eq!(refusal.detail, "decision_evaluation.joint_law_required");
    assert_eq!(refusal.code, "joint_law_required");
}

fn mean_claim() -> ForwardClaim {
    ForwardClaim::Means(MeanSource {
        coordinates: REGIMES.iter().map(|r| quantity("y", r, "mean")).collect(),
        means: vec![0.25, 0.5, 0.75],
        provider_id: "lab".into(),
        snapshot_id: "snap-9".into(),
        causal_contract_id: "checked-contract".into(),
        rng_id: "none:mean_grid".into(),
    })
}

#[test]
fn f7_a_mean_never_answers_a_probability_or_quantile() {
    let evidence = ForwardEvidence { point: Some(mean_claim()), ..ForwardEvidence::default() };
    // A target mean on an affine utility is answerable from means.
    let q = query_on(contract_with("mean"), &IDS, vec![mean_at_least(0.5)]);
    let result = evaluate_inverse_query(&q, &evidence).unwrap();
    assert_eq!(points(&result), vec![Infeasible, Feasible, Feasible]);
    assert_eq!(result.actions[1].point_values[0].standard_error, None);

    let probability = InverseConstraint::ProbabilityThreshold {
        outcome_threshold: 1.0,
        tail: Tail::Upper,
        probability: 0.5,
        comparison: Comparison::AtLeast,
    };
    let quantile =
        InverseConstraint::TargetQuantile { p: 0.5, target: 1.0, comparison: Comparison::AtLeast };
    for constraint in [probability, quantile] {
        let q = query_on(contract_with("mean"), &IDS, vec![constraint]);
        let error = evaluate_inverse_query(&q, &evidence).unwrap_err();
        assert!(matches!(
            error,
            InverseQueryError::Engine(DecisionEvalError::MeanSourceInsufficient { .. })
        ));
        assert_eq!(error.to_refusal().detail, "decision_evaluation.mean_source_insufficient");
    }
}

#[test]
fn f7_unsupported_coordinate_is_one_actions_status() {
    let columns = [
        (quantity("y", REGIMES[0], "outcome"), A0.to_vec()),
        (quantity("y", REGIMES[1], "outcome"), A1.to_vec()),
        (quantity("y", REGIMES[2], "outcome"), A2.to_vec()),
    ];
    let masked = artifact(&columns, DrawAlignment::Joint, Some(vec![true, false, true]));
    let result =
        evaluate_inverse_query(&query(&IDS, vec![mean_at_least(0.5)]), &point(masked)).unwrap();
    assert_eq!(points(&result), vec![Infeasible, Unsupported, Feasible]);
    assert_eq!(result.selected.as_deref(), Some("a2"));
    assert!(!result.selection_certified, "the passed-over a1 is unsupported, not infeasible");
    assert!(!result.grid_fully_decided);
    assert_eq!(result.existence, ExistenceClaim::FoundFeasibleAction);
    assert!(result.actions[1].point_values.is_empty());
}

#[test]
fn f7_absent_coordinate_is_unevaluated_and_never_interpolated() {
    let columns = [
        (quantity("y", REGIMES[0], "outcome"), A0.to_vec()),
        (quantity("y", REGIMES[1], "outcome"), A1.to_vec()),
    ];
    let partial = artifact(&columns, DrawAlignment::Joint, None);
    let result =
        evaluate_inverse_query(&query(&IDS, vec![mean_at_least(0.5)]), &point(partial)).unwrap();
    assert_eq!(points(&result), vec![Infeasible, Feasible, Unevaluated]);
    assert!(!result.grid_fully_decided);
    assert!(!result.exhaustive_over_declared_set);
}

#[test]
fn f7_conflicting_scenarios_are_structurally_ambiguous() {
    // s2 differs from s1 only in a1, whose mean there is 1/4.
    let s1 = bernoulli_law();
    let s2 = law_with(&A0);
    let evidence = scenarios(vec![atom("s1", Some(0.6), s1), atom("s2", Some(0.4), s2)]);
    let q = query(&IDS, vec![mean_at_least(0.5)]);
    let result = evaluate_inverse_query(&q, &evidence).unwrap();
    let all: Vec<_> = result.actions.iter().map(|a| a.all_scenario.unwrap()).collect();
    assert_eq!(all, vec![Infeasible, StructurallyAmbiguous, Feasible]);
    let a1 = &result.actions[1];
    assert_eq!(
        a1.scenario_members.iter().map(|m| (m.id.as_str(), m.status)).collect::<Vec<_>>(),
        vec![("s1", Feasible), ("s2", Infeasible)]
    );
    let posterior = a1.posterior_probability.as_ref().unwrap();
    assert!(near(posterior.feasible_mass, 0.6));
    assert!(near(posterior.infeasible_mass, 0.4));
    assert!(near(posterior.unresolved_mass, 0.0));
    // No point claim: the other fields stay absent and nothing is selected.
    assert!(a1.point.is_none() && a1.interval_region.is_none() && a1.identified_set.is_none());
    assert_eq!(result.selection, SelectionOutcome::NoPointClaim);
    assert_eq!(result.selected, None);

    // Distinct fields: a feasible point claim does not hide the ambiguity.
    let both = ForwardEvidence {
        point: Some(ForwardClaim::Law(Box::new(bernoulli_law()))),
        scenarios: evidence.scenarios.clone(),
        ..ForwardEvidence::default()
    };
    let result = evaluate_inverse_query(&q, &both).unwrap();
    assert_eq!(result.actions[1].point, Some(Feasible));
    assert_eq!(result.actions[1].all_scenario, Some(StructurallyAmbiguous));
}

#[test]
fn f7_unidentified_atom_blocks_a_scenario_claim_and_keeps_its_mass() {
    let evidence = scenarios(vec![
        atom("s1", Some(0.7), bernoulli_law()),
        StructuralAtom {
            id: "s2".into(),
            probability: Some(0.3),
            evidence: AtomEvidence::Unidentified,
        },
    ]);
    let result = evaluate_inverse_query(&query(&IDS, vec![mean_at_least(0.5)]), &evidence).unwrap();
    let all: Vec<_> = result.actions.iter().map(|a| a.all_scenario.unwrap()).collect();
    assert_eq!(all, vec![Unidentified, Unidentified, Unidentified]);
    let posterior = result.actions[2].posterior_probability.as_ref().unwrap();
    assert!(near(posterior.feasible_mass, 0.7), "not renormalized to 1");
    assert!(near(posterior.unresolved_mass, 0.3));
}

#[test]
fn f7_unevaluated_atom_is_not_renormalized_away() {
    let evidence = scenarios(vec![
        atom("s1", Some(0.5), bernoulli_law()),
        StructuralAtom {
            id: "s2".into(),
            probability: Some(0.25),
            evidence: AtomEvidence::Unevaluated("budget".into()),
        },
    ]);
    let result = evaluate_inverse_query(&query(&IDS, vec![mean_at_least(0.5)]), &evidence).unwrap();
    let all: Vec<_> = result.actions.iter().map(|a| a.all_scenario.unwrap()).collect();
    assert_eq!(all, vec![Unevaluated, Unevaluated, Unevaluated]);
    let a2 = result.actions[2].posterior_probability.as_ref().unwrap();
    // 0.25 unevaluated plus 0.25 never assigned: 0.5 of the mass is unresolved.
    assert!(near(a2.feasible_mass, 0.5), "not 0.5 / 0.75");
    assert!(near(a2.infeasible_mass, 0.0));
    assert!(near(a2.unresolved_mass, 0.5));
    let a0 = result.actions[0].posterior_probability.as_ref().unwrap();
    assert!(near(a0.infeasible_mass, 0.5));
    let member = &result.actions[0].scenario_members[1];
    assert_eq!(member.reason.as_deref(), Some("budget"));
}

#[test]
fn f7_scenario_posterior_needs_genuine_probabilities() {
    let evidence =
        scenarios(vec![atom("s1", None, bernoulli_law()), atom("s2", None, bernoulli_law())]);
    let result = evaluate_inverse_query(&query(&IDS, vec![mean_at_least(0.5)]), &evidence).unwrap();
    assert!(result.actions.iter().all(|a| a.posterior_probability.is_none()));
    assert_eq!(result.actions[2].all_scenario, Some(Feasible));
    let partial =
        scenarios(vec![atom("s1", Some(0.5), bernoulli_law()), atom("s2", None, bernoulli_law())]);
    let error =
        evaluate_inverse_query(&query(&IDS, vec![mean_at_least(0.5)]), &partial).unwrap_err();
    assert_eq!(error, InverseQueryError::InvalidProbabilities);
    assert_eq!(error.to_refusal().detail, "inverse_query.invalid_probabilities");
}

#[test]
fn f7_identified_set_and_interval_region_are_separate_fields() {
    let members = vec![atom("m1", None, bernoulli_law()), atom("m2", None, law_with(&A2))];
    let evidence = ForwardEvidence {
        identified_set: Some(IdentifiedSet { members: members.clone(), exhaustive: true }),
        interval_region: Some(IntervalRegion {
            lower: ForwardClaim::Law(Box::new(law_with(&A0))),
            upper: ForwardClaim::Law(Box::new(law_with(&A2))),
            endpoints_bound_functional: true,
        }),
        ..ForwardEvidence::default()
    };
    let result = evaluate_inverse_query(&query(&IDS, vec![mean_at_least(0.5)]), &evidence).unwrap();
    // a1: m1 mean 1/2 and m2 mean 3/4 are both feasible; interval ends 1/4 and 3/4 conflict.
    let a1 = &result.actions[1];
    assert_eq!(a1.identified_set, Some(Feasible));
    assert_eq!(a1.interval_region, Some(StructurallyAmbiguous));
    assert!(a1.point.is_none() && a1.all_scenario.is_none());

    // The same members without a completeness declaration make no uniform claim.
    let open = ForwardEvidence {
        identified_set: Some(IdentifiedSet { members, exhaustive: false }),
        ..ForwardEvidence::default()
    };
    let result = evaluate_inverse_query(&query(&IDS, vec![mean_at_least(0.5)]), &open).unwrap();
    assert_eq!(result.actions[1].identified_set, Some(Unevaluated));
}

#[test]
fn f7_multiple_feasible_actions_follow_the_declared_rule() {
    let q = |grid: &[&str], rule: SelectionRule| {
        let mut built = query(grid, vec![mean_at_least(0.5)]);
        built.selection = rule;
        evaluate_inverse_query(&built, &point(bernoulli_law())).unwrap()
    };
    let first = q(&IDS, SelectionRule::FirstInGridOrder);
    assert_eq!(first.feasible_actions, vec!["a1".to_owned(), "a2".to_owned()]);
    assert_eq!(first.selected.as_deref(), Some("a1"));
    assert!(first.selection_certified);

    let last = q(&IDS, SelectionRule::LastInGridOrder);
    assert_eq!(last.selected.as_deref(), Some("a2"));

    let unique = q(&IDS, SelectionRule::RequireUnique);
    assert_eq!(unique.selection, SelectionOutcome::MultipleFeasible);
    assert_eq!(unique.selected, None);
    assert_eq!(unique.feasible_actions.len(), 2, "the full set is still reported");

    // The declared order, not the id order, is the tie rule.
    let reversed = q(&["a2", "a1", "a0"], SelectionRule::FirstInGridOrder);
    assert_eq!(reversed.selected.as_deref(), Some("a2"));
    assert_eq!(reversed.actions[0].position, 0);
}

#[test]
fn f7_one_found_continuous_grid_point_is_not_global_feasibility() {
    let mut sampled = query(&IDS, vec![mean_at_least(0.5)]);
    sampled.grid_scope = GridScope::ContinuousSample;
    let result = evaluate_inverse_query(&sampled, &point(bernoulli_law())).unwrap();
    assert_eq!(result.existence, ExistenceClaim::FoundFeasibleAction, "a witness point");
    assert!(result.grid_fully_decided);
    assert!(!result.exhaustive_over_declared_set, "never a global claim over a continuum");

    // No feasible sampled point does not show the continuum has none.
    let mut none_found = query(&IDS, vec![mean_at_least(0.9)]);
    none_found.grid_scope = GridScope::ContinuousSample;
    let result = evaluate_inverse_query(&none_found, &point(bernoulli_law())).unwrap();
    assert_eq!(result.existence, ExistenceClaim::Undetermined);
    assert!(!result.exhaustive_over_declared_set);

    // A finite enumeration that decides every action can say none is feasible.
    let finite = query(&IDS, vec![mean_at_least(0.9)]);
    let result = evaluate_inverse_query(&finite, &point(bernoulli_law())).unwrap();
    assert_eq!(result.existence, ExistenceClaim::NoFeasibleActionInDeclaredSet);
    assert!(result.exhaustive_over_declared_set);
}

#[test]
fn f7_finite_enumeration_baseline_agrees_with_the_engine() {
    let forward = [0.25, 0.5, 0.75];
    for comparison in [Comparison::AtLeast, Comparison::AtMost] {
        let baseline_points: Vec<EnumeratedForwardValue> = IDS
            .iter()
            .zip(forward)
            .map(|(id, value)| EnumeratedForwardValue {
                id: (*id).into(),
                value: Some(value),
                supported: true,
            })
            .collect();
        let baseline =
            finite_enumeration_baseline(&baseline_points, 0.5, comparison, 1e-12).unwrap();
        let constraint = InverseConstraint::TargetMean { target: 0.5, comparison };
        let result =
            evaluate_inverse_query(&query(&IDS, vec![constraint]), &point(bernoulli_law()))
                .unwrap();
        let engine: Vec<(String, FeasibilityStatus)> =
            result.actions.iter().map(|a| (a.id.clone(), a.point.unwrap())).collect();
        assert_eq!(baseline, engine, "{comparison:?}");
    }

    // An action off the evaluated grid is unevaluated in both.
    let partial_columns = [
        (quantity("y", REGIMES[0], "outcome"), A0.to_vec()),
        (quantity("y", REGIMES[1], "outcome"), A1.to_vec()),
    ];
    let partial = artifact(&partial_columns, DrawAlignment::Joint, None);
    let baseline = finite_enumeration_baseline(
        &[
            EnumeratedForwardValue { id: "a0".into(), value: Some(0.25), supported: true },
            EnumeratedForwardValue { id: "a1".into(), value: Some(0.5), supported: true },
            EnumeratedForwardValue { id: "a2".into(), value: None, supported: true },
        ],
        0.5,
        Comparison::AtLeast,
        1e-12,
    )
    .unwrap();
    let result =
        evaluate_inverse_query(&query(&IDS, vec![mean_at_least(0.5)]), &point(partial)).unwrap();
    let engine: Vec<FeasibilityStatus> = result.actions.iter().map(|a| a.point.unwrap()).collect();
    assert_eq!(baseline.iter().map(|(_, s)| *s).collect::<Vec<_>>(), engine);

    // A masked coordinate is unsupported in both.
    let columns = [
        (quantity("y", REGIMES[0], "outcome"), A0.to_vec()),
        (quantity("y", REGIMES[1], "outcome"), A1.to_vec()),
        (quantity("y", REGIMES[2], "outcome"), A2.to_vec()),
    ];
    let masked = artifact(&columns, DrawAlignment::Joint, Some(vec![true, true, false]));
    let baseline = finite_enumeration_baseline(
        &[
            EnumeratedForwardValue { id: "a0".into(), value: Some(0.25), supported: true },
            EnumeratedForwardValue { id: "a1".into(), value: Some(0.5), supported: true },
            EnumeratedForwardValue { id: "a2".into(), value: Some(0.75), supported: false },
        ],
        0.5,
        Comparison::AtLeast,
        1e-12,
    )
    .unwrap();
    let result =
        evaluate_inverse_query(&query(&IDS, vec![mean_at_least(0.5)]), &point(masked)).unwrap();
    let engine: Vec<FeasibilityStatus> = result.actions.iter().map(|a| a.point.unwrap()).collect();
    assert_eq!(baseline.iter().map(|(_, s)| *s).collect::<Vec<_>>(), engine);
}

#[test]
fn f7_evaluation_budget_leaves_candidates_unevaluated() {
    let mut limited = query(&IDS, vec![mean_at_least(0.5)]);
    limited.max_evaluations = Some(2);
    let result = evaluate_inverse_query(&limited, &point(bernoulli_law())).unwrap();
    assert_eq!(points(&result), vec![Infeasible, Feasible, Unevaluated]);
    assert!(result.budget_exhausted);
    assert_eq!(result.evaluations_used, 2);
    assert!(!result.grid_fully_decided);
    assert!(!result.exhaustive_over_declared_set);
}

#[test]
fn f7_malformed_queries_refuse_with_inverse_query_details() {
    let evidence = point(bernoulli_law());
    let detail = |q: &InverseQuery, e: &ForwardEvidence| {
        evaluate_inverse_query(q, e).unwrap_err().to_refusal().detail
    };
    let unknown = query(&["a0", "ghost"], vec![mean_at_least(0.5)]);
    assert_eq!(detail(&unknown, &evidence), "inverse_query.invalid_grid");
    let repeated = query(&["a0", "a0"], vec![mean_at_least(0.5)]);
    assert_eq!(detail(&repeated, &evidence), "inverse_query.invalid_grid");
    let none = query(&IDS, vec![]);
    assert_eq!(detail(&none, &evidence), "inverse_query.no_constraints");
    let bad_level = query(
        &IDS,
        vec![InverseConstraint::TargetQuantile {
            p: 1.0,
            target: 0.0,
            comparison: Comparison::AtLeast,
        }],
    );
    assert_eq!(detail(&bad_level, &evidence), "inverse_query.invalid_parameter");
    let mut negative = query(&IDS, vec![mean_at_least(0.5)]);
    negative.tolerance = -1.0;
    assert_eq!(detail(&negative, &evidence), "inverse_query.invalid_parameter");
    let ok = query(&IDS, vec![mean_at_least(0.5)]);
    assert_eq!(detail(&ok, &ForwardEvidence::default()), "inverse_query.no_forward_evidence");
    let blank = scenarios(vec![atom(" ", None, bernoulli_law())]);
    assert_eq!(detail(&ok, &blank), "inverse_query.invalid_scenarios");
}
