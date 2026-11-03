//! Conditional study ranking from the actual combined decision, derived by hand.
//!
//! Native means of y over the doses 0, 1, 2 are 2.8, 4.0, 5.2 (P(m=1 | t) = 0.2, 0.5, 0.8 and
//! y = 2 + 4 m). A foreign provider returns intercept + gain * dose + 2 * mean(z) with
//! mean(z) = 0 and gain 0.5. The utilities are mean - cost with costs 0, 1, 3 for wait, treat,
//! extend. Support is declared so that wait is read from the native source, treat from
//! `external.0` and extend from `external.1`. With intercepts 3.0 (external.0) and 3.0
//! (external.1): wait 2.8, treat 3.5 - 1 = 2.5, extend 4.0 - 3 = 1.0, so wait is selected;
//! raising the intercept of external.0 to 4.5 gives treat 5.0 - 1 = 4.0 and treat is selected.
//! Structural ranking orders by declared sufficiency, then cost, then sample budget, then id.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
use antecedent::analysis::conditional_study_ranking::{
    ConditionalBranch, ConditionalCandidate, ConditionalStudyPolicy, ConditionalStudyRanking,
};
use antecedent_core::SupportStatus;
use antecedent::analysis::recalc_composite::*;
use antecedent::analysis::recalc_external::{CallbackPolicy, ExternalMeanProvider};
use antecedent::analysis::recalc_receipt::UtilitySpec;
use antecedent::analysis::recalc_static::{
    StaticResponseRequest, StaticResponseSession, execute_static_response_with_receipt,
};
use antecedent_contract::{ExternalProgramClaim, ProgramBinding};
use antecedent_core::recalc::Branch;
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_design::composition_boundary::SupportPolicy;
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy, UtilityExpr,
};
use antecedent_graph::{Admg, DenseNodeId};
use std::collections::BTreeMap;
#[path = "common/external_callback_fixture.rs"]
mod reference;
fn fixture() -> (CompositeSession, CompositeRequest, reference::Provider, reference::Provider) {
    let ctx = ExecutionContext::for_tests(7);
    let mut columns: Vec<(String, Vec<f64>)> =
        ["t", "m", "y"].iter().map(|name| ((*name).into(), Vec::new())).collect();
    for (t, ones) in [20_u32, 50, 80].into_iter().enumerate() {
        for row in 0..100_u32 {
            let m = f64::from(u32::from(row < ones));
            columns[0].1.push(f64::from(u32::try_from(t).unwrap()));
            columns[1].1.push(m);
            columns[2].1.push(2.0 + 4.0 * m);
        }
    }
    let mut graph = Admg::with_variables(3);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    graph.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    let request = StaticResponseRequest {
        columns: columns.into(),
        graph,
        treatment: VariableId::from_raw(0),
        outcome: VariableId::from_raw(2),
        support: vec![0.0, 1.0, 2.0],
        actions: vec![0.0, 1.0, 2.0],
        baseline: 0,
        active: 1,
        utility: UtilitySpec { benefit_per_unit: 1.0, cost: 0.0 },
    };
    let mut native = StaticResponseSession::new();
    let out = execute_static_response_with_receipt(&mut native, &request, &ctx).unwrap();
    for (actual, expected) in out.means.iter().zip([2.8, 4.0, 5.2]) {
        assert!((actual - expected).abs() < 1e-12);
    }
    assert!(native.producing_receipt().unwrap().totals().factor_builds > 0);
    let result = native.result().unwrap();
    let contract = result.executed_contract().unwrap();
    let graph_id = format!("graph:{}", contract.identities.identification);
    let binding = ProgramBinding {
        contract_id: antecedent_io::external_binding_wire::native_contract_identity(
            &graph_id,
            result.response().unwrap().identification_status.as_str(),
            None,
        )
        .unwrap(),
        graph_id,
        treatment_id: "t".into(),
        outcome_id: "y".into(),
        population_id: "target".into(),
        intervention_kind: "do".into(),
        horizon: 0,
        dose_grid: vec![0.0, 1.0, 2.0],
        dose_units: "dimensionless".into(),
        outcome_units: "dimensionless".into(),
        functional_id: "mean".into(),
        transform_id: "identity".into(),
    };
    let mut externals = Vec::new();
    for branch in [0, 1] {
        let mut e = reference::request(branch, CallbackPolicy::Deterministic);
        e.program = binding.clone();
        e.claim = ExternalProgramClaim::declared_by(&binding);
        e.contract.graph_id.clone_from(&binding.graph_id);
        e.contract.estimand = binding.expected_quantities();
        e.descriptor.identity.request_id = binding.identity();
        e.descriptor.identity.provider_id = format!("provider-{branch}");
        externals.push(e);
    }
    let p1 = reference::Provider::new(&externals[0]);
    let p2 = reference::Provider::new(&externals[1]);
    let decision_contract = decision_contract(&binding);
    let session = CompositeSession::from_native(native, &request, &ctx).unwrap();
    (
        session,
        CompositeRequest {
            native: request,
            native_program: binding,
            externals,
            decision_contract,
            support_policy: SupportPolicy::require_all(),
            input_order: vec!["native".into(), "external.0".into(), "external.1".into()],
        },
        p1,
        p2,
    )
}
fn decision_contract(binding: &ProgramBinding) -> DecisionContract {
    let costs = [0.0, 1.0, 3.0];
    let names = ["wait", "treat", "extend"];
    DecisionContract {
        actions: binding
            .expected_quantities()
            .into_iter()
            .enumerate()
            .map(|(i, q)| DecisionAction {
                id: names[i].into(),
                kind: ActionKind::Intervention,
                inputs: vec![q],
                utility: UtilityExpr::difference(
                    UtilityExpr::Input(0),
                    UtilityExpr::Const(costs[i]),
                ),
            })
            .collect(),
        utility_units: "net benefit".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::RequireInvariantBestAction,
    }
}


fn run(
    session: &mut CompositeSession,
    request: &CompositeRequest,
    p1: &mut reference::Provider,
    p2: &mut reference::Provider,
) {
    let mut providers: BTreeMap<Branch, &mut dyn ExternalMeanProvider> = BTreeMap::new();
    providers.insert(request.externals[0].branch, &mut *p1);
    providers.insert(request.externals[1].branch, &mut *p2);
    session.execute(request, &mut providers, &ExecutionContext::for_tests(7)).unwrap();
}

fn policy(candidates: &[(&str, bool, u64, u64)]) -> ConditionalStudyPolicy {
    ConditionalStudyPolicy {
        policy_id: "hand-policy".into(),
        branches: ["wait", "treat", "extend"]
            .into_iter()
            .map(|action| ConditionalBranch {
                action: action.into(),
                candidates: candidates
                    .iter()
                    .map(|(id, sufficient, cost, budget)| ConditionalCandidate {
                        semantic_id: format!("{action}-{id}"),
                        verified_sufficient: *sufficient,
                        cost_units: *cost,
                        sample_budget: *budget,
                    })
                    .collect(),
            })
            .collect(),
    }
}

/// Sufficiency first, then cost 5 < 10, then budget 200 < 300 and the id `a` < `z` within the
/// tie, and the insufficient study last although it is the cheapest.
const TABLE: [(&str, bool, u64, u64); 5] = [
    ("b", true, 10, 300),
    ("z", true, 10, 200),
    ("a", true, 10, 200),
    ("d", true, 5, 500),
    ("e", false, 0, 1),
];

fn supported_split(p1: &mut reference::Provider, p2: &mut reference::Provider) {
    p1.support = vec![
        SupportStatus::OutsideEmpiricalSupport,
        SupportStatus::Supported,
        SupportStatus::OutsideEmpiricalSupport,
    ];
    p2.support = vec![
        SupportStatus::OutsideEmpiricalSupport,
        SupportStatus::OutsideEmpiricalSupport,
        SupportStatus::Supported,
    ];
}

#[test]
fn the_selected_action_and_candidate_order_follow_the_hand_derived_utilities() {
    let (mut session, mut request, mut p1, mut p2) = fixture();
    request.input_order = vec!["external.0".into(), "external.1".into(), "native".into()];
    supported_split(&mut p1, &mut p2);
    run(&mut session, &request, &mut p1, &mut p2);
    // wait 2.8 > treat 2.5 > extend 1.0.
    let first = ConditionalStudyRanking::execute(&session, policy(&TABLE)).unwrap();
    assert_eq!(first.selected_action(), "wait");
    let ids: Vec<_> = first.entries().iter().map(|e| e.semantic_id.as_str()).collect();
    assert_eq!(ids, ["wait-d", "wait-a", "wait-z", "wait-b", "wait-e"]);
    let ranks: Vec<_> = first.entries().iter().map(|e| e.rank).collect();
    assert_eq!(ranks, [0, 1, 2, 3, 4]);

    // External.0 intercept 4.5: treat 5.0 - 1 = 4.0 beats wait 2.8.
    request.externals[0].model_parameters[0].1 = 4.5;
    run(&mut session, &request, &mut p1, &mut p2);
    let second = ConditionalStudyRanking::execute(&session, policy(&TABLE)).unwrap();
    assert_eq!(second.selected_action(), "treat");
    let ids: Vec<_> = second.entries().iter().map(|e| e.semantic_id.as_str()).collect();
    assert_eq!(ids, ["treat-d", "treat-a", "treat-z", "treat-b", "treat-e"]);

    // External.1 intercept 9.0: extend 9.0 + 1.0 - 3 = 7.0 beats treat 4.0.
    request.externals[1].model_parameters[0].1 = 9.0;
    run(&mut session, &request, &mut p1, &mut p2);
    let third = ConditionalStudyRanking::execute(&session, policy(&TABLE)).unwrap();
    assert_eq!(third.selected_action(), "extend");
    assert_eq!(third.entries()[0].semantic_id, "extend-d");
    assert_eq!(third.entries()[0].cost_units, 5);
}
