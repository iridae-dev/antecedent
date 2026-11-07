use super::*;
use antecedent_core::execution::AdmissionPool;

fn assert_same_functional(before: &IdentificationResult, after: &IdentificationResult) {
    assert_eq!(before.query, after.query);
    assert_eq!(before.status, after.status);
    assert_eq!(before.performance, after.performance);
    assert_eq!(before.estimands[0].adjustment_set, after.estimands[0].adjustment_set);
    assert_eq!(
        before.arena.pretty(before.estimands[0].functional),
        after.arena.pretty(after.estimands[0].functional)
    );
}

#[test]
fn conditional_enumeration_memory_is_not_a_candidate_cap_or_scientific_result() {
    // Conditioning on W opens T <- A -> W <- B -> Y; A or B must be added.
    let mut dag = antecedent_graph::Dag::with_variables(5);
    for (a, b) in [(1, 0), (1, 2), (3, 2), (3, 4), (0, 4)] {
        dag.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(4))
        .with_effect_modifiers([VariableId::from_raw(2)]);
    let run = |context: Option<ExecutionContext>, max_candidates| {
        let config = crate::backdoor::AdjustmentSearchConfig {
            max_candidates,
            context,
            ..Default::default()
        };
        let mut result = not_identified(CausalQuery::AverageEffect(query.clone()), "fixture");
        validate_dag_conditional_adjustment(&dag, &query, &mut result, &config)?;
        Ok::<_, IdentificationError>(result)
    };
    let baseline = run(None, usize::MAX).unwrap();
    assert_eq!(baseline.status, IdentificationStatus::NonparametricallyIdentified);
    assert_eq!(baseline.estimands[0].adjustment_set.as_ref(), &[VariableId::from_raw(1)]);
    let pool = AdmissionPool::fixed(1, 1);
    let starved = ExecutionContext::for_tests(9).with_admission(pool.clone());
    assert!(matches!(
        run(Some(starved.clone()), usize::MAX),
        Err(IdentificationError::Budget { budget: crate::IdentificationBudget::Memory })
    ));
    assert_eq!(pool.available().bytes(), 1);
    // Over a declared ceiling the certified conditioning set is returned, with the
    // unfinished search reported as an execution diagnostic.
    let capped = run(Some(starved.clone()), 0).unwrap();
    assert_eq!(capped.status, IdentificationStatus::NonparametricallyIdentified);
    assert_eq!(
        capped.estimands[0].adjustment_set.as_ref(),
        &[VariableId::from_raw(1), VariableId::from_raw(3)]
    );
    assert!(
        capped
            .diagnostics
            .iter()
            .any(|d| d.code.as_ref() == CONDITIONAL_SEARCH_BOUNDED_DIAGNOSTIC_CODE
                && d.kind == DiagnosticKind::Execution)
    );
    starved.cancellation.cancel();
    assert!(matches!(run(Some(starved), usize::MAX), Err(IdentificationError::Cancelled)));
    assert_eq!(pool.available().bytes(), 1);
    let funded_pool = AdmissionPool::fixed(1, 1_000_000);
    let funded = ExecutionContext::for_tests(9).with_admission(funded_pool.clone());
    assert_same_functional(&baseline, &run(Some(funded), usize::MAX).unwrap());
    assert_eq!(funded_pool.available().bytes(), 1_000_000);
}

#[test]
fn mag_enumeration_memory_is_not_unidentified_completion_mass() {
    // Z confounds T -> Y; R -> T witnesses the causal edge's visibility.
    let mut mag = Pag::with_variables(4);
    for (a, b) in [(0, 1), (0, 2), (1, 2), (3, 1)] {
        mag.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    let run = |context, max_candidates| {
        identify_on_mag_completion(
            &mag,
            MagTarget {
                t: VariableId::from_raw(1),
                y: VariableId::from_raw(2),
                t_d: DenseNodeId::from_raw(1),
                y_d: DenseNodeId::from_raw(2),
            },
            Value::f64(1.0),
            Value::f64(0.0),
            AdjustmentBudget { max_candidates, max_examinations: 1_000_000, context },
        )
    };
    let baseline = run(None, usize::MAX).unwrap();
    assert_eq!(baseline.status, IdentificationStatus::NonparametricallyIdentified);
    assert_eq!(baseline.estimands[0].adjustment_set.as_ref(), &[VariableId::from_raw(0)]);
    let pool = AdmissionPool::fixed(1, 1);
    let starved = ExecutionContext::for_tests(9).with_admission(pool.clone());
    assert!(matches!(
        run(Some(&starved), usize::MAX),
        Err(IdentificationError::Budget { budget: crate::IdentificationBudget::Memory })
    ));
    assert_eq!(pool.available().bytes(), 1);
    // Over a declared ceiling the certified ancestral set is the answer, so no mass becomes
    // unidentified; the unfinished search is an execution diagnostic.
    let capped = run(Some(&starved), 0).unwrap();
    assert_eq!(capped.status, IdentificationStatus::NonparametricallyIdentified);
    assert!(capped.diagnostics.iter().any(|d| {
        d.code.as_ref() == MAG_SEARCH_BOUNDED_DIAGNOSTIC_CODE && d.kind == DiagnosticKind::Execution
    }));
    let envelope = IdentificationEnvelope::from_cases(vec![GraphIdentificationCase {
        graph: mag.clone(),
        result: capped,
        weight: ProbabilityMass(1.0),
    }]);
    assert_eq!(envelope.status, IdentificationStatus::NonparametricallyIdentified);
    starved.cancellation.cancel();
    assert!(matches!(run(Some(&starved), usize::MAX), Err(IdentificationError::Cancelled)));
    assert_eq!(pool.available().bytes(), 1);
    let funded_pool = AdmissionPool::fixed(1, 1_000_000);
    let funded = ExecutionContext::for_tests(9).with_admission(funded_pool.clone());
    assert_same_functional(&baseline, &run(Some(&funded), usize::MAX).unwrap());
    assert_eq!(funded_pool.available().bytes(), 1_000_000);
}
