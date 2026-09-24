//! Checked natural-direct-effect operation and detached numeric replay.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{RefuteSuite, Study};
use antecedent_core::{CausalQuery, ExecutionContext, NestedCounterfactualQuery, VariableId};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_io::{consume_analysis_result, verify_contract_against_body};

#[test]
fn natural_direct_effect_retains_worlds_and_replays_outcome_fit() {
    let index: Vec<f64> = (0..300).map(f64::from).collect();
    let x: Vec<f64> = index.iter().map(|i| (i * 0.37).sin()).collect();
    let m: Vec<f64> = index.iter().zip(&x).map(|(i, x)| 0.8 * x + (i * 0.91).cos()).collect();
    let y: Vec<f64> = index
        .iter()
        .zip(&x)
        .zip(&m)
        .map(|((i, x), m)| 1.7 * x + 4.0 * m + 0.01 * (i * 0.23).sin())
        .collect();
    let data = TabularData::from_f64_columns([
        ("x", x.as_slice()),
        ("m", m.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap();
    let mut graph = Dag::with_variables(3);
    for (source, target) in [(0, 1), (0, 2), (1, 2)] {
        graph
            .insert_directed(DenseNodeId::from_raw(source), DenseNodeId::from_raw(target))
            .unwrap();
    }
    let query = NestedCounterfactualQuery::with_levels(
        VariableId::from_raw(0),
        VariableId::from_raw(1),
        VariableId::from_raw(2),
        -1.0,
        2.0,
    )
    .unwrap();
    let builder = Study::tabular(data.clone())
        .graph(graph)
        .query(CausalQuery::NestedCounterfactual(query))
        .refute(RefuteSuite::None);
    let study = builder.clone().build().unwrap();
    let ctx = ExecutionContext::for_tests(417);
    let one_shot = study.run(&ctx).unwrap();
    let mut prepared = study.prepare(&ctx).unwrap();
    drop(builder);
    drop(study);
    let retained = prepared.checked_nested_counterfactual_operation().unwrap();
    assert_eq!(retained.query(), &query);
    assert_eq!(retained.graph().node_count(), 3);
    let first = prepared.estimate(&data, &ctx).unwrap();
    assert!((first.estimate.ate - 5.1).abs() < 0.001, "{}", first.estimate.ate);
    assert_eq!(one_shot.estimate.ate.to_bits(), first.estimate.ate.to_bits());
    assert_eq!(
        one_shot.executed_contract.as_ref().unwrap().identities.program,
        prepared.contract().unwrap().identities.program
    );

    let shifted_y: Vec<f64> = y.iter().map(|value| value + 0.7).collect();
    let shifted = TabularData::from_f64_columns([
        ("x", x.as_slice()),
        ("m", m.as_slice()),
        ("y", shifted_y.as_slice()),
    ])
    .unwrap();
    let refreshed = prepared.refresh(shifted, &ctx).unwrap();
    assert!((refreshed.estimate.ate - first.estimate.ate).abs() < 1e-8);
    let artifact = prepared.encode_contracted_result(&refreshed, "nested-checked", &ctx).unwrap();
    let consumed = consume_analysis_result(&artifact).unwrap();
    assert!(
        consumed.acceptance.accepts_as_verified_program(),
        "{:?}",
        consumed.acceptance.unresolved
    );
    let contract = consumed.contract.as_ref().unwrap();
    assert_eq!(
        contract.program.as_ref().unwrap().commitments.resolved_estimator.as_deref(),
        Some("mediation.linear"),
        "{:?}",
        contract.program.as_ref().unwrap().commitments
    );
    assert!(contract.program.as_ref().unwrap().checked_nested_counterfactual.is_some());
    assert!(contract.data_snapshot.as_ref().unwrap().nested_counterfactual_fit.is_some());

    let mut changed_body = consumed.body.clone();
    changed_body.estimate = Some(refreshed.estimate.ate + 1.0);
    let unresolved = verify_contract_against_body(&consumed.header, &changed_body, contract);
    assert!(
        unresolved.iter().any(|reason| reason.as_ref() == "body.nested_counterfactual_estimate"),
        "{unresolved:?}"
    );
    let mut changed_contract = contract.clone();
    changed_contract
        .data_snapshot
        .as_mut()
        .unwrap()
        .nested_counterfactual_fit
        .as_mut()
        .unwrap()
        .outcome_cross[1] += 10.0;
    let unresolved =
        verify_contract_against_body(&consumed.header, &consumed.body, &changed_contract);
    assert!(!unresolved.is_empty(), "tampered fit moments were accepted");

    let mut old_contract = contract.clone();
    old_contract.program.as_mut().unwrap().checked_nested_counterfactual = None;
    old_contract.data_snapshot.as_mut().unwrap().nested_counterfactual_fit = None;
    let unresolved = verify_contract_against_body(&consumed.header, &consumed.body, &old_contract);
    assert!(
        unresolved.iter().any(|reason| reason.as_ref() == "program.checked_nested_counterfactual"),
        "{unresolved:?}"
    );

    let mut changed_world = contract.clone();
    changed_world
        .program
        .as_mut()
        .unwrap()
        .checked_nested_counterfactual
        .as_mut()
        .unwrap()
        .active_bits = 4.0f64.to_bits();
    let unresolved = verify_contract_against_body(&consumed.header, &consumed.body, &changed_world);
    assert!(
        unresolved
            .iter()
            .any(|reason| reason.as_ref() == "program.checked_nested_counterfactual.binding"),
        "{unresolved:?}"
    );

    let mut changed_graph = contract.clone();
    if let antecedent_io::GraphIdentityWire::Dag(dag) =
        &mut changed_graph.identification.as_mut().unwrap().graph
    {
        dag.edges.pop();
    }
    let unresolved = verify_contract_against_body(&consumed.header, &consumed.body, &changed_graph);
    assert!(
        unresolved
            .iter()
            .any(|reason| reason.as_ref() == "program.checked_nested_counterfactual.graph"),
        "{unresolved:?}"
    );

    let mut singular_fit = contract.clone();
    singular_fit.data_snapshot.as_mut().unwrap().nested_counterfactual_fit.as_mut().unwrap().gram =
        [300.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let unresolved = verify_contract_against_body(&consumed.header, &consumed.body, &singular_fit);
    assert!(
        unresolved
            .iter()
            .any(|reason| reason.as_ref() == "dependencies.nested_counterfactual_fit_rank"),
        "{unresolved:?}"
    );
}
