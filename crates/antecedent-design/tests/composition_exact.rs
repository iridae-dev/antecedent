//! Composition-boundary support policy on mean grids whose affine-utility values are computed
//! by hand: the utility is 2x - 1 of the mean x of each action's coordinate.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::{ExternalTrustState, QuantityRole, ScientificQuantity};
use antecedent_core::SupportStatus;
use antecedent_design::composition_boundary::{
    ActionStatus, CoordinateIssue, DecisionInput, SupportPolicy, SupportedVerdict, TrustEvidence,
    TrustRequirement, UnsupportedReason, evaluate_with_support,
};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy, UtilityExpr,
};
use antecedent_design::decision_eval::{MeanEstimate, MeanSource, Verdict};

fn near(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12
}

fn q(regime: &str, functional: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "y".into(),
        variable_name: "y".into(),
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

fn attested() -> TrustEvidence {
    TrustEvidence::External(ExternalTrustState::ExternallyAttested { attestor: "lab".into() })
}

fn mean_input(id: &str, coordinates: &[(&str, f64)], statuses: &[SupportStatus]) -> DecisionInput {
    DecisionInput::from_mean_source(
        id,
        MeanSource {
            coordinates: coordinates.iter().map(|(regime, _)| q(regime, "mean")).collect(),
            means: coordinates.iter().map(|(_, mean)| *mean).collect(),
            provider_id: "lab".into(),
            snapshot_id: format!("snap-{id}"),
            causal_contract_id: "checked-contract".into(),
            rng_id: "none:mean_grid".into(),
            estimate: MeanEstimate::Frequentist,
        },
        statuses,
        &attested(),
        TrustRequirement::Unrestricted,
    )
    .unwrap()
}

/// `2 * x0 - 1`.
fn affine() -> UtilityExpr {
    UtilityExpr::difference(
        UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Const(2.0)),
        UtilityExpr::Const(1.0),
    )
}

fn contract() -> DecisionContract {
    let action = |id: &str, regime: &str| DecisionAction {
        id: id.into(),
        kind: ActionKind::Intervention,
        inputs: vec![q(regime, "mean")],
        utility: affine(),
    };
    DecisionContract {
        actions: vec![
            action("wait", "do(a=0)"),
            action("treat", "do(a=1)"),
            action("extend", "do(a=2)"),
        ],
        utility_units: "utility".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    }
}

const STATUSES: [SupportStatus; 3] =
    [SupportStatus::Supported, SupportStatus::WeakOverlap, SupportStatus::Supported];

/// Means 1, 3 and 4 give the utilities 1, 5 and 7. With weak overlap not admitted the middle
/// action is left out and the comparison is between 1 and 7; admitting weak overlap scores all
/// three. Either way the best action is extend, with the hand utility 7.
#[test]
fn the_support_policy_decides_which_hand_utilities_are_compared() {
    let grid =
        || mean_input("grid", &[("do(a=0)", 1.0), ("do(a=1)", 3.0), ("do(a=2)", 4.0)], &STATUSES);

    let strict =
        evaluate_with_support(&contract(), &[grid()], &SupportPolicy::compare_supported()).unwrap();
    assert_eq!(strict.outcomes.len(), 2);
    assert!(near(strict.outcomes[0].expected_utility, 1.0));
    assert!(near(strict.outcomes[1].expected_utility, 7.0));
    assert_eq!(
        strict.verdict,
        SupportedVerdict::Compared(Verdict::UniquelyOptimal("extend".into()))
    );
    assert!(matches!(strict.dispositions[1].status, ActionStatus::Unsupported { .. }));

    let mut policy = SupportPolicy::compare_supported();
    policy.weakest_support = SupportStatus::WeakOverlap;
    let lenient = evaluate_with_support(&contract(), &[grid()], &policy).unwrap();
    assert_eq!(lenient.outcomes.len(), 3);
    let utilities: Vec<f64> = lenient.outcomes.iter().map(|o| o.expected_utility).collect();
    for (got, want) in utilities.iter().zip([1.0, 5.0, 7.0]) {
        assert!(near(*got, want), "{got} vs {want}");
    }
    assert_eq!(
        lenient.verdict,
        SupportedVerdict::Compared(Verdict::UniquelyOptimal("extend".into()))
    );
}

/// Two grids answer different actions: the first has wait = 2 (utility 3) and extend = 0.5
/// (utility 0), the second treat = 1.5 (utility 2). Compared by value the best is wait at 3, and
/// each action is attributed to the grid that answered it; a grid that lacks a coordinate leaves
/// that action unsupported with the missing-coordinate reason.
#[test]
fn actions_answered_by_different_grids_are_compared_by_their_hand_values() {
    let first = mean_input(
        "first",
        &[("do(a=0)", 2.0), ("do(a=2)", 0.5)],
        &[SupportStatus::Supported, SupportStatus::Supported],
    );
    let second = mean_input("second", &[("do(a=1)", 1.5)], &[SupportStatus::Supported]);
    let decision = evaluate_with_support(
        &contract(),
        &[first.clone(), second],
        &SupportPolicy::compare_supported(),
    )
    .unwrap();
    assert_eq!(
        decision.verdict,
        SupportedVerdict::Compared(Verdict::UniquelyOptimal("wait".into()))
    );
    let utilities: Vec<f64> = decision.outcomes.iter().map(|o| o.expected_utility).collect();
    for (got, want) in utilities.iter().zip([3.0, 2.0, 0.0]) {
        assert!(near(*got, want), "{got} vs {want}");
    }
    assert_eq!(decision.dispositions[0].input_id.as_deref(), Some("first"));
    assert_eq!(decision.dispositions[1].input_id.as_deref(), Some("second"));
    assert_eq!(decision.dispositions[2].input_id.as_deref(), Some("first"));

    let alone =
        evaluate_with_support(&contract(), &[first], &SupportPolicy::compare_supported()).unwrap();
    assert_eq!(alone.outcomes.len(), 2);
    assert_eq!(
        alone.dispositions[1].status,
        ActionStatus::Unsupported {
            reasons: vec![UnsupportedReason {
                input_id: "first".into(),
                coordinate: 0,
                issue: CoordinateIssue::NotInSource,
            }],
        }
    );
    assert_eq!(alone.verdict, SupportedVerdict::Compared(Verdict::UniquelyOptimal("wait".into())));
}
