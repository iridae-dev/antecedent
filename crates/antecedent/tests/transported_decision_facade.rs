//! A transported response reaches the decision contract through the facade: bound to
//! the causal contract, under the contract's admissibility rules, with the transport
//! receipt carried into the result and a refusal for every request a transported
//! point cannot answer.
//!
//! Oracle. The transported mean of `effect` in `clinic_b` is `2.0`. The utility of
//! `treat` is `4 x - 1.5`, so its expected utility is `6.5`; `wait` is `0 * x = 0`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::float_cmp,
    clippy::too_many_lines,
    reason = "closed-form oracles; one scenario per test"
)]

use antecedent::Study;
use antecedent::analysis::{
    AdmissibilityRules, AdmissibleDecisionContract, BoundDecisionContract,
    DecisionDeclaredExclusion, SupportPolicy, TransportDeclaration, TransportedDemand,
    TransportedResponse, TransportedRoute, UncertaintyRequirement,
};
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::SupportStatus;
use antecedent_design::composition_boundary::{SupportedVerdict, UnsupportedActionPolicy};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy, UtilityExpr,
};
use antecedent_design::decision_eval::Verdict;
use antecedent_design::decision_transported_source::{
    DETAIL_LAW_NOT_SUPPLIED, DETAIL_SUPPORT_BELOW_POLICY,
};

#[path = "fixtures_common/mod.rs"]
mod common;

use common::fixtures::confounded_scm;

fn effect() -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "effect".into(),
        variable_name: "effect".into(),
        role: QuantityRole::Outcome,
        units: "units".into(),
        population_id: "clinic_b".into(),
        regime_id: "contrast:do(t=1)-do(t=0)".into(),
        horizon: 0,
        functional_id: "mean".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn declaration() -> TransportDeclaration {
    TransportDeclaration {
        route: TransportedRoute::Declared,
        source_identity: "transport-1".into(),
        source_populations: vec!["trial".into()],
        target_population: "clinic_b".into(),
        transport_identities: vec!["diagram-1".into()],
        assumptions: vec!["selection acts on age only".into()],
        support: SupportStatus::Extrapolative,
        uncalibrated_standard_error: Some(0.3),
        provider_id: "transport-provider".into(),
        snapshot_id: "snapshot-1".into(),
        causal_contract_id: "contract-1".into(),
    }
}

fn point(value: f64) -> TransportedResponse {
    TransportedResponse::declared_point(vec![effect()], vec![value], &declaration()).unwrap()
}

fn action(id: &str, utility: UtilityExpr) -> DecisionAction {
    DecisionAction {
        id: id.into(),
        kind: ActionKind::Intervention,
        inputs: vec![effect()],
        utility,
    }
}

/// `treat` earns `4 x - 1.5`; `wait` earns `0 x`; `hold` earns `x - 5`.
fn contract() -> DecisionContract {
    let treat = UtilityExpr::difference(
        UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Const(4.0)),
        UtilityExpr::Const(1.5),
    );
    let wait = UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Const(0.0));
    let hold = UtilityExpr::difference(UtilityExpr::Input(0), UtilityExpr::Const(5.0));
    DecisionContract {
        actions: vec![action("treat", treat), action("wait", wait), action("hold", hold)],
        utility_units: "utility".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "clinic_b".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    }
}

fn bound(rules: AdmissibilityRules) -> BoundDecisionContract {
    let (data, dag, query) = confounded_scm(64, 5);
    let causal = Study::tabular(data).query(query).graph(dag).inspect().unwrap();
    BoundDecisionContract::bind(&causal, AdmissibleDecisionContract { contract: contract(), rules })
        .unwrap()
}

fn accepting() -> SupportPolicy {
    SupportPolicy {
        unsupported: UnsupportedActionPolicy::RequireAllActions,
        weakest_support: SupportStatus::Extrapolative,
    }
}

#[test]
fn a_transported_point_decides_by_the_utility_of_the_mean_and_names_its_transport() {
    let bound = bound(AdmissibilityRules::default());
    let decided = bound
        .evaluate_transported(&point(2.0), TransportedDemand::ExpectedUtility, &accepting())
        .unwrap();
    let outcomes = &decided.decision().decision.outcomes;
    assert!((outcomes[0].expected_utility - 6.5).abs() < 1e-12);
    assert!(outcomes[1].expected_utility.abs() < 1e-12);
    assert_eq!(
        decided.decision().decision.verdict,
        SupportedVerdict::Compared(Verdict::UniquelyOptimal("treat".into()))
    );
    assert_eq!(decided.bound_identity(), bound.identity());
    assert_eq!(decided.causal_target(), bound.causal_target());
    assert_eq!(decided.receipt().target_population, "clinic_b");

    let inspection = decided.inspect();
    assert_eq!(inspection.inference, "point_only");
    assert!(!inspection.native_target_data);
    assert_eq!(inspection.target_population, "clinic_b");
    assert_eq!(inspection.source_populations, vec!["trial".to_owned()]);
    assert_eq!(inspection.support, "extrapolative");
    assert!((inspection.actions[0].1 - 6.5).abs() < 1e-12);
    assert!(inspection.assumptions.iter().any(|a| a.contains("not native target data")));

    // Repeating the request reproduces the identity; another value or another
    // transport at the same shape changes it.
    let again = bound
        .evaluate_transported(&point(2.0), TransportedDemand::ExpectedUtility, &accepting())
        .unwrap();
    assert_eq!(again.identity(), decided.identity());
    let mut other = declaration();
    other.transport_identities = vec!["diagram-2".into()];
    let moved = TransportedResponse::declared_point(vec![effect()], vec![2.0], &other).unwrap();
    let moved = bound
        .evaluate_transported(&moved, TransportedDemand::ExpectedUtility, &accepting())
        .unwrap();
    assert_ne!(moved.identity(), decided.identity());
}

#[test]
fn declared_exclusions_remove_actions_and_are_part_of_the_bound_decision() {
    let plain = bound(AdmissibilityRules::default());
    let exclude = |ids: &[&str]| {
        let mut rules = AdmissibilityRules::default();
        rules.declared_exclusions = ids
            .iter()
            .map(|id| DecisionDeclaredExclusion {
                action_id: (*id).into(),
                reason: "not offered".into(),
            })
            .collect();
        bound(rules)
    };
    let excluding = exclude(&["hold"]);
    assert_ne!(plain.identity(), excluding.identity());

    let full = plain
        .evaluate_transported(&point(2.0), TransportedDemand::ExpectedUtility, &accepting())
        .unwrap();
    assert_eq!(full.decision().decision.outcomes.len(), 3);
    // `hold` earns 2 - 5 = -3.
    assert!((full.decision().decision.outcomes[2].expected_utility + 3.0).abs() < 1e-12);

    let decided = excluding
        .evaluate_transported(&point(2.0), TransportedDemand::ExpectedUtility, &accepting())
        .unwrap();
    assert_eq!(decided.decision().decision.outcomes.len(), 2);
    assert_eq!(
        decided.decision().decision.verdict,
        SupportedVerdict::Compared(Verdict::UniquelyOptimal("treat".into()))
    );
    assert_ne!(decided.identity(), full.identity());

    // Exclusions that leave nothing to compare refuse with their own detail.
    let refusal = exclude(&["hold", "wait"])
        .evaluate_transported(&point(2.0), TransportedDemand::ExpectedUtility, &accepting())
        .unwrap_err();
    assert_eq!(refusal.code, "decision_contract_unsatisfied");
    assert_eq!(refusal.detail, "transported_decision.exclusions_leave_no_comparison");
}

#[test]
fn a_support_rule_stronger_than_the_transported_label_refuses() {
    let mut rules = AdmissibilityRules::default();
    rules.default_weakest_support = Some(SupportStatus::Supported);
    let refusal = bound(rules)
        .evaluate_transported(&point(2.0), TransportedDemand::ExpectedUtility, &accepting())
        .unwrap_err();
    assert_eq!(refusal.code, "transport_support_failure");
    assert_eq!(refusal.detail, "transported_decision.support_rule_unmet");
    assert_eq!(refusal.supplied.as_deref(), Some("extrapolative"));
    assert!(refusal.remedy.is_some());
}

#[test]
fn an_uncertainty_requirement_a_point_cannot_meet_refuses() {
    for requirement in
        [UncertaintyRequirement::Credible, UncertaintyRequirement::StructuralEnvelope]
    {
        let mut rules = AdmissibilityRules::default();
        rules.uncertainty = requirement;
        let refusal = bound(rules)
            .evaluate_transported(&point(2.0), TransportedDemand::ExpectedUtility, &accepting())
            .unwrap_err();
        assert_eq!(refusal.code, "decision_contract_unsatisfied");
        assert_eq!(refusal.detail, "transported_decision.uncertainty_requirement");
    }
    // A point requirement is met by a transported point.
    let mut rules = AdmissibilityRules::default();
    rules.uncertainty = UncertaintyRequirement::PointOnly;
    assert!(
        bound(rules)
            .evaluate_transported(&point(2.0), TransportedDemand::ExpectedUtility, &accepting())
            .is_ok()
    );
}

#[test]
fn a_point_refuses_laws_and_intervals_and_the_default_policy_refuses_the_label() {
    let bound = bound(AdmissibilityRules::default());
    let refusal =
        bound.evaluate_transported(&point(2.0), TransportedDemand::Law, &accepting()).unwrap_err();
    assert_eq!(refusal.code, "decision_contract_unsatisfied");
    assert_eq!(refusal.detail, DETAIL_LAW_NOT_SUPPLIED);
    assert!(refusal.remedy.is_some());

    let refusal = bound
        .evaluate_transported(&point(2.0), TransportedDemand::Interval, &accepting())
        .unwrap_err();
    assert_eq!(refusal.code, "cell_not_licensed");

    // The default policy admits only `supported` coordinates: accepting a
    // transported claim is the consumer's explicit choice.
    let refusal = bound
        .evaluate_transported(
            &point(2.0),
            TransportedDemand::ExpectedUtility,
            &SupportPolicy::compare_supported(),
        )
        .unwrap_err();
    assert_eq!(refusal.code, "cell_not_licensed");
    assert_eq!(refusal.detail, DETAIL_SUPPORT_BELOW_POLICY);
}
