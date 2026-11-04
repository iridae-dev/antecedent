//! The composite recalculation (native finite engine plus two attested foreign mean callbacks)
//! against hand-derived expected utilities.
//!
//! The native engine runs on a table whose mediator share is 0.2, 0.5 and 0.8 at treatment
//! levels 0, 1, 2 and whose outcome is exactly `y = 2 + 4 m`; the front-door response is
//! therefore `2 + 4 P(m | t) = 2.8, 4.0, 5.2` by hand. The actions wait, treat and extend cost
//! 0, 1 and 3, so with the native input the expected utilities are `2.8 - 0, 4.0 - 1, 5.2 - 3 =
//! 2.8, 3.0, 2.2` (treat is uniquely best); making treat cost 2 gives `2.8, 2.0, 2.2` (wait is
//! uniquely best). The providers' own numbers differ from these, so a decision that read the
//! wrong source would not match.
//! SPDX-License-Identifier: MIT OR Apache-2.0
use antecedent::analysis::recalc_composite::*;
use antecedent::analysis::recalc_external::{CallbackPolicy, ExternalMeanProvider};
use antecedent::analysis::recalc_receipt::UtilitySpec;
use antecedent::analysis::recalc_static::{
    StaticResponseRequest, StaticResponseSession, execute_static_response_with_receipt,
};
use antecedent_contract::{ExternalProgramClaim, ProgramBinding};
use antecedent_core::recalc::Branch;
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_design::composition_boundary::{SupportPolicy, SupportedVerdict};
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

use antecedent_design::decision_eval::Verdict;

fn run(
    session: &mut CompositeSession,
    request: &CompositeRequest,
    p1: &mut reference::Provider,
    p2: &mut reference::Provider,
) -> CompositeOutcome {
    let mut providers: BTreeMap<Branch, &mut dyn ExternalMeanProvider> = BTreeMap::new();
    providers.insert(request.externals[0].branch, &mut *p1);
    providers.insert(request.externals[1].branch, &mut *p2);
    session.execute(request, &mut providers, &ExecutionContext::for_tests(7)).unwrap()
}

#[test]
fn composite_decision_equals_the_hand_utilities_of_the_analytic_front_door_response() {
    let (mut session, mut request, mut p1, mut p2) = fixture();
    let first = run(&mut session, &request, &mut p1, &mut p2);
    let utilities: Vec<f64> = first.decision.outcomes.iter().map(|o| o.expected_utility).collect();
    assert_eq!(utilities.len(), 3);
    for (got, want) in utilities.iter().zip([2.8, 3.0, 2.2]) {
        assert!((got - want).abs() < 1e-9, "{utilities:?}");
    }
    assert_eq!(
        first.decision.verdict,
        SupportedVerdict::Compared(Verdict::UniquelyOptimal("treat".into()))
    );
    // Treat costs 2 instead of 1: utilities (2.8, 2.0, 2.2), wait is uniquely best.
    request.decision_contract.actions[1].utility =
        UtilityExpr::difference(UtilityExpr::Input(0), UtilityExpr::Const(2.0));
    let second = run(&mut session, &request, &mut p1, &mut p2);
    let utilities: Vec<f64> = second.decision.outcomes.iter().map(|o| o.expected_utility).collect();
    for (got, want) in utilities.iter().zip([2.8, 2.0, 2.2]) {
        assert!((got - want).abs() < 1e-9, "{utilities:?}");
    }
    assert_eq!(
        second.decision.verdict,
        SupportedVerdict::Compared(Verdict::UniquelyOptimal("wait".into()))
    );
    // Shifting every native outcome by 1 shifts every mean by 1 to (3.8, 5.0, 6.2); with treat
    // still costing 2 the utilities are (3.8, 5.0 - 2, 6.2 - 3) = (3.8, 3.0, 3.2).
    for value in &mut request.native.columns[2].1 {
        *value += 1.0;
    }
    let third = run(&mut session, &request, &mut p1, &mut p2);
    let utilities: Vec<f64> = third.decision.outcomes.iter().map(|o| o.expected_utility).collect();
    for (got, want) in utilities.iter().zip([3.8, 3.0, 3.2]) {
        assert!((got - want).abs() < 1e-9, "{utilities:?}");
    }
    assert_eq!(
        third.decision.verdict,
        SupportedVerdict::Compared(Verdict::UniquelyOptimal("wait".into()))
    );
}
