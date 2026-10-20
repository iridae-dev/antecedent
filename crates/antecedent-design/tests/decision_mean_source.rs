//! A mean-only source answers only the expectation of an affine utility.

use antecedent_contract::ScientificQuantity;
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, HardConstraint,
    SourceRepresentation, StructuralPolicy, UtilityExpr,
};
use antecedent_design::decision_eval::{
    DecisionEvalError, MeanSource, Verdict, evaluate_contract_on_means,
};

mod common;

fn quantity(regime: &str, functional: &str) -> ScientificQuantity {
    ScientificQuantity {
        units: "mmHg".into(),
        regime_id: regime.into(),
        functional_id: functional.into(),
        ..common::quantity("y")
    }
}

/// `E[Y | do(a)] = 1 + 2a` on the grid `a = 0, 1, 2`.
fn source(functional: &str) -> MeanSource {
    MeanSource {
        coordinates: ["do(a=0)", "do(a=1)", "do(a=2)"]
            .iter()
            .map(|regime| quantity(regime, functional))
            .collect(),
        means: vec![1.0, 3.0, 5.0],
        provider_id: "lab".into(),
        snapshot_id: "snap-9".into(),
        causal_contract_id: "checked-contract".into(),
        rng_id: "none:mean_grid".into(),
    }
}

/// `2 * x0 - 1`.
fn affine() -> UtilityExpr {
    UtilityExpr::difference(
        UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Const(2.0)),
        UtilityExpr::Const(1.0),
    )
}

fn contract_with(functional: &str, utility: UtilityExpr) -> DecisionContract {
    DecisionContract {
        actions: vec![
            DecisionAction {
                id: "wait".into(),
                kind: ActionKind::Intervention,
                inputs: vec![quantity("do(a=0)", functional)],
                utility: affine(),
            },
            DecisionAction {
                id: "treat".into(),
                kind: ActionKind::Intervention,
                inputs: vec![quantity("do(a=2)", functional)],
                utility,
            },
        ],
        utility_units: "utility".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    }
}

fn contract() -> DecisionContract {
    contract_with("mean", affine())
}

#[test]
fn expected_utility_of_an_affine_utility_is_the_utility_of_the_means() {
    let result = evaluate_contract_on_means(&contract(), &source("mean")).unwrap();
    // 2 * 1 - 1 = 1 and 2 * 5 - 1 = 9, in declaration order.
    assert_eq!(result.actions[0].id, "wait");
    assert!((result.actions[0].expected_utility - 1.0).abs() < 1e-12);
    assert!((result.actions[1].expected_utility - 9.0).abs() < 1e-12);
    assert!((result.actions[1].value - 9.0).abs() < 1e-12);
    assert_eq!(result.verdict, Verdict::UniquelyOptimal("treat".into()));
    for outcome in &result.actions {
        assert!(outcome.admissible);
        assert_eq!(outcome.standard_error, None);
        assert_eq!(outcome.expected_regret, None);
        assert_eq!(outcome.max_regret, None);
    }
    assert_eq!(result.evpi, None);
    assert_eq!(result.n_draws, 0);
    assert!(result.effective_draws.abs() < f64::EPSILON);
    assert_eq!(result.source.provider_id, "lab");
    assert_eq!(result.source.snapshot_id, "snap-9");
    assert_eq!(result.source.causal_contract_id, "checked-contract");
    assert_eq!(result.contract_identity, contract().identity().unwrap());
    assert!(result.assumptions.iter().any(|a| a.contains("means only")));
}

#[test]
fn expected_loss_prefers_the_smaller_mean_and_exact_ties_are_indistinguishable() {
    let mut loss = contract();
    loss.criterion = DecisionCriterion::PosteriorExpectedLoss;
    let result = evaluate_contract_on_means(&loss, &source("mean")).unwrap();
    assert_eq!(result.verdict, Verdict::UniquelyOptimal("wait".into()));

    // Both actions read the same coordinate through the same utility.
    let mut tied = contract();
    tied.actions[1].inputs = tied.actions[0].inputs.clone();
    let result = evaluate_contract_on_means(&tied, &source("mean")).unwrap();
    assert_eq!(result.verdict, Verdict::Indistinguishable(vec!["wait".into(), "treat".into()]));
}

#[test]
fn a_nonlinear_utility_needs_more_than_a_mean() {
    let squared = UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Input(0));
    let error =
        evaluate_contract_on_means(&contract_with("mean", squared), &source("mean")).unwrap_err();
    let DecisionEvalError::MeanSourceInsufficient { needed } = &error else {
        panic!("expected MeanSourceInsufficient, got {error:?}");
    };
    assert!(needed.contains(&SourceRepresentation::JointDraws));
    let refusal = error.to_refusal();
    assert_eq!(refusal.code, "decision_contract_unsatisfied");
    assert_eq!(refusal.detail, "decision_evaluation.mean_source_insufficient");

    let clipped = UtilityExpr::maximum(UtilityExpr::Input(0), UtilityExpr::Const(0.0));
    assert!(matches!(
        evaluate_contract_on_means(&contract_with("mean", clipped), &source("mean")),
        Err(DecisionEvalError::MeanSourceInsufficient { .. })
    ));
}

#[test]
fn a_hard_constraint_and_other_criteria_are_refused() {
    let mut constrained = contract();
    constrained.constraints.push(HardConstraint {
        id: "cap".into(),
        expr: UtilityExpr::Input(0),
        bound: 10.0,
        min_probability: 0.9,
        units: "mmHg".into(),
        applies_to: vec![],
    });
    assert!(matches!(
        evaluate_contract_on_means(&constrained, &source("mean")),
        Err(DecisionEvalError::MeanSourceInsufficient { .. })
    ));

    for criterion in [
        DecisionCriterion::ThresholdProbability { threshold: 1.0 },
        DecisionCriterion::Quantile { p: 0.5 },
        DecisionCriterion::ExpectedRegret,
        DecisionCriterion::Regret,
    ] {
        let mut other = contract();
        other.criterion = criterion;
        assert!(matches!(
            evaluate_contract_on_means(&other, &source("mean")),
            Err(DecisionEvalError::MeanSourceInsufficient { .. })
        ));
    }
}

#[test]
fn a_mean_grid_is_not_an_outcome_law() {
    let error = evaluate_contract_on_means(&contract_with("outcome", affine()), &source("outcome"))
        .unwrap_err();
    assert_eq!(error, DecisionEvalError::MeaningMismatch { action: "wait".into(), input: 0 });
    assert_eq!(error.to_refusal().code, "distribution_meaning_mismatch");
}

#[test]
fn an_unknown_coordinate_or_a_malformed_source_refuses() {
    let mut missing = source("mean");
    missing.coordinates.truncate(2);
    missing.means.truncate(2);
    assert_eq!(
        evaluate_contract_on_means(&contract(), &missing).unwrap_err(),
        DecisionEvalError::QuantityNotFound { action: "treat".into(), input: 0 }
    );

    let mut ragged = source("mean");
    ragged.means.pop();
    assert!(matches!(
        evaluate_contract_on_means(&contract(), &ragged),
        Err(DecisionEvalError::Contract(_))
    ));
    let mut nan = source("mean");
    nan.means[0] = f64::NAN;
    assert!(matches!(
        evaluate_contract_on_means(&contract(), &nan),
        Err(DecisionEvalError::Contract(_))
    ));
}

#[test]
fn a_coordinate_named_twice_is_refused_instead_of_using_the_first() {
    let mut duplicated = source("mean");
    duplicated.coordinates[1] = duplicated.coordinates[0].clone();
    duplicated.means[1] = 99.0;
    let error = evaluate_contract_on_means(&contract(), &duplicated).unwrap_err();
    assert!(
        matches!(
            error,
            DecisionEvalError::Contract(
                antecedent_design::decision_contract::DecisionContractError::InvalidDeclaration(_)
            )
        ),
        "{error:?}"
    );
}

#[test]
fn negative_and_positive_zero_utilities_are_tied() {
    // Both actions read the same positive mean; one pays 1 * 0.0 = 0.0 and the other
    // 1 * -0.0 = -0.0, which are the same utility.
    let mut zero = contract();
    zero.actions[0].utility = UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Const(0.0));
    zero.actions[1].utility = UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Const(-0.0));
    zero.actions[1].inputs = zero.actions[0].inputs.clone();
    let result = evaluate_contract_on_means(&zero, &source("mean")).unwrap();
    assert_eq!(result.verdict, Verdict::Indistinguishable(vec!["wait".into(), "treat".into()]));
    assert!(result.assumptions.iter().any(|a| a.contains("does not assert a posterior")));
}
