//! Structured refusals on a composition that mixes valid and invalid parts.
//!
//! One mean grid supplies means 1, 3 and 4 at do(a=0), do(a=1) and do(a=2) with supports
//! Supported, OutsideEmpiricalSupport and Supported. Five actions read it with the affine
//! utility 2x - 1 except `square` (x * x):
//!
//! - `wait` reads do(a=0): mean 1, utility 1.
//! - `treat` reads do(a=1): support below the allowed Supported, so it is left out.
//! - `extend` reads do(a=2): mean 4, utility 7.
//! - `ghost` reads do(a=9): the grid has no such coordinate.
//! - `square` reads do(a=2) (supported) but its utility is not affine, which a mean cannot answer.
//!
//! Hand-predicted partial result: exactly wait (1) and extend (7) are scored, extend is
//! uniquely optimal, and the refusal set is treat (below support), ghost (not in source) and
//! square (needs a joint law), in declaration order. Requiring every action instead refuses the
//! same three, named in that order.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::{ExternalTrustState, QuantityRole, ScientificQuantity};
use antecedent_core::SupportStatus;
use antecedent_design::composition_boundary::{
    ActionStatus, BoundaryError, CoordinateIssue, DecisionInput, SupportPolicy, SupportedVerdict,
    TrustEvidence, TrustRequirement, UnevaluatedReason, UnsupportedReason, evaluate_with_support,
};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy, UtilityExpr,
};
use antecedent_design::decision_eval::{MeanEstimate, MeanSource, Verdict};

fn q(regime: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "y".into(),
        variable_name: "y".into(),
        role: QuantityRole::Outcome,
        units: "units".into(),
        population_id: "target".into(),
        regime_id: regime.into(),
        horizon: 0,
        functional_id: "mean".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn grid() -> DecisionInput {
    let coordinates = [("do(a=0)", 1.0), ("do(a=1)", 3.0), ("do(a=2)", 4.0)];
    DecisionInput::from_mean_source(
        "grid",
        MeanSource {
            coordinates: coordinates.iter().map(|(r, _)| q(r)).collect(),
            means: coordinates.iter().map(|(_, m)| *m).collect(),
            provider_id: "lab".into(),
            snapshot_id: "snap-grid".into(),
            causal_contract_id: "checked-contract".into(),
            rng_id: "none:mean_grid".into(),
            estimate: MeanEstimate::Frequentist,
        },
        &[SupportStatus::Supported, SupportStatus::OutsideEmpiricalSupport, SupportStatus::Supported],
        &TrustEvidence::External(ExternalTrustState::ExternallyAttested {
            attestor: "lab".into(),
        }),
        TrustRequirement::Unrestricted,
    )
    .unwrap()
}

fn affine() -> UtilityExpr {
    UtilityExpr::difference(
        UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Const(2.0)),
        UtilityExpr::Const(1.0),
    )
}

fn contract() -> DecisionContract {
    let action = |id: &str, regime: &str, utility: UtilityExpr| DecisionAction {
        id: id.into(),
        kind: ActionKind::Intervention,
        inputs: vec![q(regime)],
        utility,
    };
    DecisionContract {
        actions: vec![
            action("wait", "do(a=0)", affine()),
            action("treat", "do(a=1)", affine()),
            action("extend", "do(a=2)", affine()),
            action("ghost", "do(a=9)", affine()),
            action(
                "square",
                "do(a=2)",
                UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Input(0)),
            ),
        ],
        utility_units: "utility".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    }
}

#[test]
fn a_mixed_composition_scores_exactly_the_valid_actions_and_reports_the_predicted_refusal_set() {
    let decision =
        evaluate_with_support(&contract(), &[grid()], &SupportPolicy::compare_supported()).unwrap();
    let scored: Vec<(&str, f64)> =
        decision.outcomes.iter().map(|o| (o.id.as_str(), o.expected_utility)).collect();
    assert_eq!(scored.len(), 2);
    assert_eq!(scored[0].0, "wait");
    assert!((scored[0].1 - 1.0).abs() < 1e-12);
    assert_eq!(scored[1].0, "extend");
    assert!((scored[1].1 - 7.0).abs() < 1e-12);
    assert_eq!(
        decision.verdict,
        SupportedVerdict::Compared(Verdict::UniquelyOptimal("extend".into()))
    );

    let statuses: Vec<(&str, &ActionStatus)> =
        decision.dispositions.iter().map(|d| (d.id.as_str(), &d.status)).collect();
    let reason = |coordinate, issue| ActionStatus::Unsupported {
        reasons: vec![UnsupportedReason { input_id: "grid".into(), coordinate, issue }],
    };
    assert_eq!(
        statuses,
        vec![
            ("wait", &ActionStatus::Evaluated),
            ("treat", &reason(0, CoordinateIssue::BelowSupport(SupportStatus::OutsideEmpiricalSupport))),
            ("extend", &ActionStatus::Evaluated),
            ("ghost", &reason(0, CoordinateIssue::NotInSource)),
            (
                "square",
                &ActionStatus::Unevaluated { reason: UnevaluatedReason::NonAffineUtilityNeedsJointLaw }
            ),
        ]
    );
    assert_eq!(decision.dispositions[0].input_id.as_deref(), Some("grid"));
    assert_eq!(decision.dispositions[1].input_id, None);
}

#[test]
fn requiring_every_action_refuses_with_exactly_the_three_unscored_actions_named() {
    let error =
        evaluate_with_support(&contract(), &[grid()], &SupportPolicy::require_all()).unwrap_err();
    assert_eq!(
        error,
        BoundaryError::UnsupportedActions {
            actions: vec!["treat".into(), "ghost".into(), "square".into()]
        }
    );
    let refusal = error.to_refusal();
    assert_eq!(refusal.stage, "compose");
    assert_eq!(refusal.code, "decision_contract_unsatisfied");
    assert_eq!(refusal.detail, "composition_boundary.unsupported_action_not_comparable");
    assert_eq!(refusal.offending.as_deref(), Some("treat,ghost,square"));
    assert!(refusal.remedy.is_some());
    assert_eq!(refusal.validate(), Ok(()));
}
