//! Known-truth decision evaluation on an enumerated aligned joint law.
//!
//! The risky action's utility is the product `P * Q`. With the enumerated
//! realizations below, `E[P] * E[Q] = 4.5` but `E[P * Q] = 2`; only genuine
//! joint rows give the second value, and only they choose the safe action.

use antecedent_core::{QuantityRole, ScientificQuantity};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, HardConstraint,
    StructuralPolicy, UtilityExpr,
};
use antecedent_design::decision_eval::{DecisionEvalError, Verdict, evaluate_contract};
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io::quantity_wire::DistributionMeaningWire;

fn quantity(variable: &str, regime: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: variable.into(),
        role: QuantityRole::Outcome,
        units: "units".into(),
        population_id: "target".into(),
        regime_id: regime.into(),
        horizon: 0,
        functional_id: "outcome".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn columns() -> Vec<ScientificQuantity> {
    vec![quantity("p", "do(a=1)"), quantity("q", "do(a=1)"), quantity("safe", "do(a=0)")]
}

fn artifact(
    p: [f64; 4],
    q: [f64; 4],
    alignment: DrawAlignment,
    meaning: DistributionMeaningWire,
    weights: Option<Vec<f64>>,
) -> DistributionArtifact {
    let identity = DistributionIdentity::new(
        meaning,
        &columns(),
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
    let mut draws = Vec::new();
    for i in 0..4 {
        draws.extend([p[i], q[i], 3.0]);
    }
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".into(), "quantity".into()],
            shape: [4, 3],
            weights,
            supported: None,
            calibration: DistributionCalibration::Exact,
            trust: DistributionTrust::Unverified,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        draws,
    )
    .unwrap()
}

fn joint(p: [f64; 4], q: [f64; 4]) -> DistributionArtifact {
    artifact(p, q, DrawAlignment::Joint, DistributionMeaningWire::InterventionalPredictive, None)
}

const P: [f64; 4] = [1.0, 3.0, 2.0, 0.0];
const Q: [f64; 4] = [4.0, 0.0, 2.0, 6.0];

fn contract(criterion: DecisionCriterion) -> DecisionContract {
    let cols = columns();
    DecisionContract {
        actions: vec![
            DecisionAction {
                id: "risky".into(),
                kind: ActionKind::Intervention,
                inputs: vec![cols[0].clone(), cols[1].clone()],
                utility: UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Input(1)),
            },
            DecisionAction {
                id: "safe".into(),
                kind: ActionKind::Intervention,
                inputs: vec![cols[2].clone()],
                utility: UtilityExpr::Input(0),
            },
        ],
        utility_units: "units".into(),
        criterion,
        constraints: vec![],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    }
}

fn near(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-12
}

#[test]
fn joint_rows_choose_the_safe_action_where_pairing_by_marginals_would_not() {
    let result =
        evaluate_contract(&contract(DecisionCriterion::PosteriorExpectedUtility), &joint(P, Q))
            .unwrap();
    let risky = &result.actions[0];
    assert!(near(risky.expected_utility, 2.0), "E[P*Q] from joint rows");
    assert!(near(result.actions[1].expected_utility, 3.0));
    assert_eq!(result.verdict, Verdict::UniquelyOptimal("safe".into()));
    // E[max(PQ, 3)] = (4 + 3 + 4 + 3) / 4 = 3.5, so EVPI = 0.5.
    assert!(near(result.evpi.unwrap(), 0.5));
    // Risky loses 3 in the two rows where PQ = 0; safe loses 1 where PQ = 4.
    assert!(near(risky.expected_regret.unwrap(), 1.5));
    assert!(near(result.actions[1].expected_regret.unwrap(), 0.5));
    assert_eq!(result.source.snapshot_id, "enumeration-1");
    assert_eq!(
        result.contract_identity,
        contract(DecisionCriterion::PosteriorExpectedUtility).identity().unwrap()
    );

    // Same marginals, different pairing: the choice flips.
    let flipped = joint(P, [0.0, 6.0, 2.0, 4.0]);
    let result =
        evaluate_contract(&contract(DecisionCriterion::PosteriorExpectedUtility), &flipped)
            .unwrap();
    assert!(near(result.actions[0].expected_utility, 5.5));
    assert_eq!(result.verdict, Verdict::UniquelyOptimal("risky".into()));
}

#[test]
fn independent_marginals_are_refused_not_paired_by_index() {
    let marginals = artifact(
        P,
        Q,
        DrawAlignment::IndependentMarginals,
        DistributionMeaningWire::InterventionalPredictive,
        None,
    );
    let error =
        evaluate_contract(&contract(DecisionCriterion::PosteriorExpectedUtility), &marginals)
            .unwrap_err();
    assert_eq!(
        error,
        DecisionEvalError::JointLawRequired {
            action: Some("risky".into()),
            supplied_alignment: "independent_marginals",
        }
    );
    assert_eq!(error.reason_code(), "joint_law_required");
}

#[test]
fn a_posterior_over_a_mean_cannot_answer_an_outcome_law_input() {
    let posterior = artifact(
        P,
        Q,
        DrawAlignment::Joint,
        DistributionMeaningWire::CausalFunctionalPosterior,
        None,
    );
    let error =
        evaluate_contract(&contract(DecisionCriterion::PosteriorExpectedUtility), &posterior)
            .unwrap_err();
    assert_eq!(error, DecisionEvalError::MeaningMismatch { action: "risky".into(), input: 0 });
    assert_eq!(error.reason_code(), "distribution_meaning_mismatch");
}

#[test]
fn hard_constraints_exclude_and_never_penalize() {
    let mut capped = contract(DecisionCriterion::PosteriorExpectedUtility);
    capped.constraints.push(HardConstraint {
        id: "q-cap".into(),
        expr: UtilityExpr::Input(1),
        bound: 5.0,
        min_probability: 1.0,
        units: "units".into(),
        applies_to: vec!["risky".into()],
    });
    // Q = 6 in one of four equally likely rows, so the cap holds with 0.75.
    let result =
        evaluate_contract(&capped, &joint([1.0, 3.0, 2.0, 0.0], [4.0, 0.0, 2.0, 6.0])).unwrap();
    let risky = &result.actions[0];
    assert!(!risky.admissible);
    assert!(near(risky.exclusions[0].probability, 0.75));
    assert_eq!(risky.exclusions[0].constraint_id, "q-cap");
    // The excluded action's utility is untouched: no penalty was subtracted.
    assert!(near(risky.expected_utility, 2.0));
    assert!(risky.expected_regret.is_none());
    assert_eq!(result.verdict, Verdict::UniquelyOptimal("safe".into()));
    assert!(near(result.evpi.unwrap(), 0.0));

    // A weaker probability requirement admits it again.
    capped.constraints[0].min_probability = 0.75;
    let result = evaluate_contract(&capped, &joint(P, Q)).unwrap();
    assert!(result.actions[0].admissible);

    // Every action excluded: no admissible action, no EVPI.
    capped.constraints[0].applies_to.clear();
    capped.constraints[0].expr = UtilityExpr::Input(0);
    capped.constraints[0].bound = -1.0;
    capped.constraints[0].min_probability = 1.0;
    let result = evaluate_contract(&capped, &joint(P, Q)).unwrap();
    assert_eq!(result.verdict, Verdict::NoAdmissibleAction);
    assert!(result.evpi.is_none());
    assert!(result.actions.iter().all(|a| !a.admissible));
}

#[test]
fn each_initial_criterion_is_scored_by_its_own_route() {
    let source = joint(P, Q);
    // P(utility >= 3): risky PQ = 4,0,4,0 -> 0.5; safe -> 1.
    let result = evaluate_contract(
        &contract(DecisionCriterion::ThresholdProbability { threshold: 3.0 }),
        &source,
    )
    .unwrap();
    assert!(near(result.actions[0].value, 0.5));
    assert!(near(result.actions[1].value, 1.0));
    assert_eq!(result.verdict, Verdict::UniquelyOptimal("safe".into()));

    // Lower quartile of utility: risky sorted 0,0,4,4 -> 0 at p = 0.25; safe -> 3.
    let result =
        evaluate_contract(&contract(DecisionCriterion::Quantile { p: 0.25 }), &source).unwrap();
    assert!(near(result.actions[0].value, 0.0));
    assert!(near(result.actions[1].value, 3.0));

    // Expected loss reads the same numbers as a quantity to minimize.
    let result =
        evaluate_contract(&contract(DecisionCriterion::PosteriorExpectedLoss), &source).unwrap();
    assert_eq!(result.verdict, Verdict::UniquelyOptimal("risky".into()));

    // Regret against the best action per draw: risky loses 3 when PQ = 0 and
    // nothing otherwise; safe loses 1 when PQ = 4.
    let result = evaluate_contract(&contract(DecisionCriterion::Regret), &source).unwrap();
    assert!(near(result.actions[0].value, 3.0));
    assert!(near(result.actions[1].value, 1.0));
    assert_eq!(result.verdict, Verdict::UniquelyOptimal("safe".into()));
    let result = evaluate_contract(&contract(DecisionCriterion::ExpectedRegret), &source).unwrap();
    assert!(near(result.actions[0].value, 1.5));
    assert!(near(result.actions[1].value, 0.5));
}

#[test]
fn weights_are_normalized_and_change_the_answer() {
    let weighted = artifact(
        P,
        Q,
        DrawAlignment::Joint,
        DistributionMeaningWire::InterventionalPredictive,
        // All mass on the two rows where PQ = 4.
        Some(vec![1.0, 0.0, 1.0, 0.0]),
    );
    let result =
        evaluate_contract(&contract(DecisionCriterion::PosteriorExpectedUtility), &weighted)
            .unwrap();
    assert!(near(result.actions[0].expected_utility, 4.0));
    assert_eq!(result.verdict, Verdict::UniquelyOptimal("risky".into()));
    assert!(near(result.effective_draws, 2.0));
}

#[test]
fn sampled_laws_report_point_values_without_unlicensed_monte_carlo_errors() {
    let exact = joint(P, Q);
    for calibration in [
        DistributionCalibration::Unmeasured,
        DistributionCalibration::PointOnly,
        DistributionCalibration::Measured,
    ] {
        let mut meta = exact.metadata().clone();
        meta.calibration = calibration;
        let sampled = DistributionArtifact::new(meta, exact.draws().to_vec()).unwrap();
        let result =
            evaluate_contract(&contract(DecisionCriterion::PosteriorExpectedUtility), &sampled)
                .unwrap();
        assert_eq!(result.verdict, Verdict::UniquelyOptimal("safe".into()));
        assert!(result.actions.iter().all(|a| a.standard_error.is_none()));
        assert!(result.assumptions.iter().any(|a| a.contains("point ranking")));
    }
}

#[test]
fn indistinguishable_actions_are_reported_as_such() {
    let cols = columns();
    let twin = |id: &str| DecisionAction {
        id: id.into(),
        kind: ActionKind::Policy,
        inputs: vec![cols[2].clone()],
        utility: UtilityExpr::Input(0),
    };
    let mut tied = contract(DecisionCriterion::PosteriorExpectedUtility);
    tied.actions = vec![twin("a"), twin("b")];
    let result = evaluate_contract(&tied, &joint(P, Q)).unwrap();
    assert_eq!(result.verdict, Verdict::Indistinguishable(vec!["a".into(), "b".into()]));
}

#[test]
fn structure_criteria_refuse_a_single_draw_source() {
    for criterion in
        [DecisionCriterion::MinimaxOverIdentifiedSet, DecisionCriterion::MaximinOverStructures]
    {
        let error = evaluate_contract(&contract(criterion), &joint(P, Q)).unwrap_err();
        assert!(matches!(error, DecisionEvalError::StructureInputsRequired(_)));
        assert_eq!(error.reason_code(), "route_not_supported");
    }
}

#[test]
fn a_missing_coordinate_is_a_semantics_mismatch() {
    let mut other = contract(DecisionCriterion::PosteriorExpectedUtility);
    other.actions[0].inputs[0].units = "kg".into();
    let error = evaluate_contract(&other, &joint(P, Q)).unwrap_err();
    assert_eq!(error, DecisionEvalError::QuantityNotFound { action: "risky".into(), input: 0 });
    assert_eq!(error.reason_code(), "quantity_semantics_mismatch");
}

#[test]
fn loss_regret_and_perfect_information_use_minimum_loss() {
    // Losses: risky [4,0,4,0], safe [3,3,3,3]. Perfect information
    // achieves mean loss 1.5; without it the minimum expected loss is 2.
    let result =
        evaluate_contract(&contract(DecisionCriterion::PosteriorExpectedLoss), &joint(P, Q))
            .unwrap();
    assert!(near(result.evpi.unwrap(), 0.5));
    assert!(near(result.actions[0].expected_regret.unwrap(), 0.5));
    assert!(near(result.actions[1].expected_regret.unwrap(), 1.5));
    assert!(near(result.actions[0].max_regret.unwrap(), 1.0));
    assert!(near(result.actions[1].max_regret.unwrap(), 3.0));
}

#[test]
fn zero_mass_atoms_do_not_change_worst_regret_or_tiny_quantiles() {
    use antecedent_design::decision_contract::DecisionFunctional;
    use antecedent_design::decision_eval::evaluate_functional;
    let source = artifact(
        P,
        Q,
        DrawAlignment::Joint,
        DistributionMeaningWire::InterventionalPredictive,
        Some(vec![1.0, 0.0, 1.0, 0.0]),
    );
    let c = contract(DecisionCriterion::Regret);
    let result = evaluate_contract(&c, &source).unwrap();
    assert!(near(result.actions[0].value, 0.0));
    assert!(near(result.actions[1].value, 1.0));
    assert_eq!(result.verdict, Verdict::UniquelyOptimal("risky".into()));
    let f = evaluate_functional(&c, "risky", DecisionFunctional::Quantile { p: 1e-15 }, &source)
        .unwrap();
    assert!(near(f.value, 4.0));
    let q =
        evaluate_contract(&contract(DecisionCriterion::Quantile { p: 1e-15 }), &source).unwrap();
    assert!(near(q.actions[0].value, f.value));
}

#[test]
fn quantile_functional_and_criterion_share_boundary_convention() {
    use antecedent_design::decision_contract::DecisionFunctional;
    use antecedent_design::decision_eval::evaluate_functional;
    let source = artifact(
        [0.0, 1.0, 2.0, 3.0],
        [1.0; 4],
        DrawAlignment::Joint,
        DistributionMeaningWire::InterventionalPredictive,
        Some(vec![0.1, 0.2, 0.3, 0.4]),
    );
    for p in [1e-15, 0.1, 0.3, 0.6, 0.9] {
        let c = contract(DecisionCriterion::Quantile { p });
        let direct =
            evaluate_functional(&c, "risky", DecisionFunctional::Quantile { p }, &source).unwrap();
        let ranked = evaluate_contract(&c, &source).unwrap();
        assert!(near(direct.value, ranked.actions[0].value));
    }
}

#[test]
fn nonfinite_aggregates_refuse_instead_of_ranking_nan_or_infinity() {
    use antecedent_design::decision_contract::DecisionFunctional;
    use antecedent_design::decision_eval::evaluate_functional;
    let mut c = contract(DecisionCriterion::PosteriorExpectedUtility);
    c.actions[0].utility = UtilityExpr::Input(0);
    let huge = joint([f64::MAX, -f64::MAX, f64::MAX, -f64::MAX], [1.0; 4]);
    assert!(matches!(
        evaluate_functional(&c, "risky", DecisionFunctional::Variance, &huge),
        Err(DecisionEvalError::NonFiniteUtility { .. })
    ));
    c.actions[1].utility =
        UtilityExpr::Add(Box::new(UtilityExpr::Input(0)), Box::new(UtilityExpr::Const(-f64::MAX)));
    assert!(matches!(
        evaluate_contract(&c, &huge),
        Err(DecisionEvalError::NonFiniteUtility { .. })
    ));
}

#[test]
fn hard_constraints_respect_tiny_positive_probability_mass() {
    let mut c = contract(DecisionCriterion::PosteriorExpectedUtility);
    c.constraints.push(HardConstraint {
        id: "cap".into(),
        expr: UtilityExpr::Input(1),
        bound: 5.0,
        min_probability: 1.0,
        units: "units".into(),
        applies_to: vec!["risky".into()],
    });
    let rare = artifact(
        P,
        Q,
        DrawAlignment::Joint,
        DistributionMeaningWire::InterventionalPredictive,
        Some(vec![1.0, 1.0, 1.0, 1e-15]),
    );
    assert!(!evaluate_contract(&c, &rare).unwrap().actions[0].admissible);
    // Zero success probability cannot meet even a tiny positive requirement.
    c.constraints[0].bound = -1.0;
    c.constraints[0].min_probability = 1e-15;
    assert!(!evaluate_contract(&c, &rare).unwrap().actions[0].admissible);
}
