use super::*;

fn collider_dag() -> antecedent_graph::Dag {
    let mut dag = antecedent_graph::Dag::with_variables(5);
    for (a, b) in [(1, 0), (1, 2), (3, 2), (3, 4), (0, 4)] {
        dag.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    dag
}

fn collider_query() -> AverageEffectQuery {
    AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(4))
        .with_effect_modifiers([VariableId::from_raw(2)])
}

#[test]
fn conditional_max_results_zero_is_invalid() {
    let query = collider_query();
    let config = crate::backdoor::AdjustmentSearchConfig { max_results: 0, ..Default::default() };
    let mut result = not_identified(CausalQuery::AverageEffect(query.clone()), "fixture");
    let outcome =
        validate_dag_conditional_adjustment(&collider_dag(), &query, &mut result, &config);
    assert!(matches!(outcome, Err(IdentificationError::InvalidInput { .. })), "{outcome:?}");
}

#[test]
fn conditional_truncation_emits_diagnostic() {
    let query = collider_query();
    let run = |max_results| {
        let config = crate::backdoor::AdjustmentSearchConfig { max_results, ..Default::default() };
        let mut result = not_identified(CausalQuery::AverageEffect(query.clone()), "fixture");
        validate_dag_conditional_adjustment(&collider_dag(), &query, &mut result, &config).unwrap();
        result
    };
    // {A} and {B} are both minimal adjustment sets; one result stops the enumeration.
    let truncated = run(1);
    assert_eq!(truncated.status, IdentificationStatus::NonparametricallyIdentified);
    let note = truncated
        .diagnostics
        .iter()
        .find(|d| d.code.as_ref() == CONDITIONAL_RESULTS_TRUNCATED_DIAGNOSTIC_CODE)
        .expect("truncation is reported");
    assert_eq!(note.kind, DiagnosticKind::Execution);
    assert!(!crate::envelope::search_truncated(&truncated));
    let full = run(64);
    assert!(
        !full
            .diagnostics
            .iter()
            .any(|d| d.code.as_ref() == CONDITIONAL_RESULTS_TRUNCATED_DIAGNOSTIC_CODE)
    );
}

#[test]
fn modifier_collider_requires_an_alternative_adjustment_set() {
    let mut dag = antecedent_graph::Dag::with_variables(5);
    for (a, b) in [(3, 0), (3, 2), (4, 2), (4, 1), (0, 1)] {
        dag.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
        .with_effect_modifiers([VariableId::from_raw(2)]);
    let mut identifier = BackdoorIdentifier::new();
    let prepared = identifier.prepare(&dag).unwrap();
    let result = identifier
        .identify(
            &prepared,
            &CausalQuery::AverageEffect(query.clone()),
            &mut IdentificationWorkspace::default(),
        )
        .unwrap();
    assert_eq!(result.status, IdentificationStatus::NonparametricallyIdentified);
    assert_eq!(result.estimands[0].adjustment_set.as_ref(), &[VariableId::from_raw(3)]);
    identifier.config.forbidden = Arc::from([VariableId::from_raw(3)]);
    let result = identifier
        .identify(
            &prepared,
            &CausalQuery::AverageEffect(query.clone()),
            &mut IdentificationWorkspace::default(),
        )
        .unwrap();
    assert_eq!(result.estimands[0].adjustment_set.as_ref(), &[VariableId::from_raw(4)]);
    identifier.config.forbidden = Arc::from([VariableId::from_raw(3), VariableId::from_raw(4)]);
    let result = identifier
        .identify(
            &prepared,
            &CausalQuery::AverageEffect(query),
            &mut IdentificationWorkspace::default(),
        )
        .unwrap();
    assert_eq!(result.status, IdentificationStatus::NotIdentified);
}
