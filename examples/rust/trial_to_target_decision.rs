// SPDX-License-Identifier: MIT OR Apache-2.0
//! Carry a trial effect to a target decision.
//!
//! The target contributes design information, while outcomes come from the synthetic trial.

use antecedent::analysis::trial_ipw_transport_decision::{
    TrialIpwRows, TrialIpwTransportDecisionRequest, compile_trial_ipw_transport_decision,
    consume_trial_ipw_transport_decision,
};
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::{
    ContinuousDomain, ExecutionContext, GridSpec, ResponseFunctional, ResponseQuery,
    TransportQuery, VariableId,
};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy, UtilityExpr,
};
use antecedent_graph::{Admg, SelectionDiagram};
use std::sync::Arc;

fn main() {
    let treatment = VariableId::from_raw(0);
    let outcome = VariableId::from_raw(2);
    let mut graph = Admg::with_variables(3);
    let source_node = graph.dense_of_static(treatment).unwrap();
    let outcome_node = graph.dense_of_static(outcome).unwrap();
    graph.insert_directed(source_node, outcome_node).unwrap();
    let diagram = SelectionDiagram::try_new(graph, [VariableId::from_raw(1)]).unwrap();
    let query = TransportQuery::new(
        ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome,
            treatment: ContinuousDomain::new(treatment, GridSpec::Values(Arc::from([0.0, 1.0]))),
        }),
        "trial",
        "target",
        [treatment],
    );
    let quantity = ScientificQuantity {
        variable_id: "2".into(),
        variable_name: "effect".into(),
        role: QuantityRole::Outcome,
        units: "score".into(),
        population_id: "target".into(),
        regime_id: "do(0=1)-do(0=0)".into(),
        horizon: 0,
        functional_id: "average_effect".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    };
    let decision = DecisionContract {
        actions: vec![
            DecisionAction {
                id: "treat".into(),
                kind: ActionKind::External,
                inputs: vec![quantity.clone()],
                utility: UtilityExpr::Input(0),
            },
            DecisionAction {
                id: "hold".into(),
                kind: ActionKind::External,
                inputs: vec![quantity.clone()],
                utility: UtilityExpr::Const(0.0),
            },
        ],
        utility_units: "score".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: Vec::new(),
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::RequireInvariantBestAction,
    };
    let rows = TrialIpwRows {
        unit_ids: (0..80).map(|i| format!("unit-{i}")).collect(),
        outcomes: (0..80).map(|i| if i < 40 && i % 2 == 0 { 1.0 } else { 0.0 }).collect(),
        treatment: (0..80).map(|i| i % 2 == 0).collect(),
        trial: (0..80).map(|i| i < 40).collect(),
        selection_probability: vec![0.5; 80],
        treatment_probability: vec![0.5; 80],
        draws: 100,
        seed: 31,
    };
    let request = TrialIpwTransportDecisionRequest { diagram, query, rows, quantity, decision };
    let plan = compile_trial_ipw_transport_decision(request).unwrap();
    assert_eq!(plan.plan_view().route().as_str(), "TrialIpwTransportDecision");
    let ctx = ExecutionContext::for_tests(31);
    let result = plan.execute(&ctx).unwrap();
    assert_eq!(result.effect_draws.len(), 100);
    assert_eq!(result.calibration, "not_measured");
    let bytes = plan.export(&result, "trial-ipw-decision", &ctx).unwrap();
    let replay =
        consume_trial_ipw_transport_decision(&bytes, result.lowered_program_identity, &ctx)
            .unwrap();
    assert_eq!(replay.effect_draws, result.effect_draws);
    assert_eq!(replay.decision.decision, result.decision.decision);
    println!("Target action comparison: {:?}", replay.decision.decision.verdict);
}
