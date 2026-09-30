//! Rebinding caller validators on a prepared handle must reach the sealed
//! operation that runs them, not only the retained study snapshot.

mod common;

use std::sync::Arc;

use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ConditionalEffectQuery, ExecutionContext, TemporalEffectQuery,
    TemporalPolicy, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_validate::{
    CustomEffectValidator, RefutationProblem, RefutationReport, ValidationError,
};
use common::fixtures::{confounded_cpdag, confounded_scm, confounded_series};

const NAME: &str = "caller.fixed";

struct FixedValidator {
    pass: bool,
}

impl CustomEffectValidator for FixedValidator {
    fn name(&self) -> &'static str {
        NAME
    }

    fn validate(
        &self,
        problem: &RefutationProblem<'_>,
        _ctx: &ExecutionContext,
    ) -> Result<RefutationReport, ValidationError> {
        Ok(RefutationReport::new(
            NAME,
            problem.original.ate,
            problem.original.ate,
            if self.pass { 1.0 } else { 0.0 },
            true,
            self.pass,
            (!self.pass).then(|| Arc::from("forced fail")),
            0,
        ))
    }
}

fn validator(pass: bool) -> Vec<Arc<dyn CustomEffectValidator>> {
    vec![Arc::new(FixedValidator { pass })]
}

fn verdict(result: &StudyResult) -> bool {
    result
        .refutations
        .iter()
        .find(|report| report.refuter.as_ref() == NAME)
        .expect("caller validator report")
        .passed
}

#[test]
fn rebound_validators_reach_the_sealed_bayesian_dag_effect() {
    let ctx = ExecutionContext::for_tests(41);
    let (data, dag, query) = confounded_scm(400, 41);
    let mut prepared = Study::tabular(data.clone())
        .graph(dag)
        .query(query)
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(32)))
        .refute(RefuteSuite::Cheap)
        .custom_validators(validator(true))
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    assert!(prepared.checked_bayesian_gcomp_operation().is_some());
    assert!(verdict(&prepared.estimate(&data, &ctx).unwrap()));
    prepared.rebind_custom_validators(validator(false)).unwrap();
    assert!(!verdict(&prepared.estimate(&data, &ctx).unwrap()));
}

#[test]
fn rebound_validators_reach_the_sealed_temporal_class_effect() {
    let ctx = ExecutionContext::for_tests(977);
    let data = confounded_series(640, 0.0, 0.0, 977);
    let query = TemporalEffectQuery::pulse(
        antecedent_core::VariableId::from_raw(0),
        antecedent_core::VariableId::from_raw(1),
        1.0,
    )
    .with_policy(TemporalPolicy::pulse(-1))
    .with_horizon_steps(1);
    let mut prepared = Study::series(data.clone())
        .graph(confounded_cpdag())
        .query(CausalQuery::TemporalEffect(query))
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .custom_validators(validator(true))
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    assert!(prepared.checked_temporal_class_effect_info().is_some());
    assert!(verdict(&prepared.estimate_series(&data, &ctx).unwrap()));
    prepared.rebind_custom_validators(validator(false)).unwrap();
    assert!(!verdict(&prepared.estimate_series(&data, &ctx).unwrap()));
}

#[test]
fn rebound_validators_reach_the_dispatched_bayesian_conditional_effect() {
    let ctx = ExecutionContext::for_tests(59);
    let n = 400usize;
    let mut columns = [const { Vec::new() }; 4];
    for row in 0..n {
        let z = (row % 8) as f64 - 3.5;
        let u = ((row * 17 % 31) as f64 - 15.0) / 15.0;
        let t = f64::from((row * 13 + row / 8) % 11 < 5);
        columns[0].push(t);
        columns[1].push(1.0 + 0.7 * u + 2.0 * t + 0.5 * t * z);
        columns[2].push(z);
        columns[3].push(u);
    }
    let data = TabularData::from_f64_columns([
        ("t", columns[0].as_slice()),
        ("y", columns[1].as_slice()),
        ("z", columns[2].as_slice()),
        ("u", columns[3].as_slice()),
    ])
    .unwrap();
    let mut graph = Dag::with_variables(4);
    for (from, to) in [(3, 0), (3, 1), (0, 1), (2, 1)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    let query = ConditionalEffectQuery::try_new(
        AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
            .with_effect_modifiers([VariableId::from_raw(2)]),
    )
    .unwrap();
    let mut prepared = Study::tabular(data.clone())
        .graph(graph)
        .query(CausalQuery::ConditionalEffect(query))
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(32)))
        .refute(RefuteSuite::Cheap)
        .custom_validators(validator(true))
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    // Caller validators keep the Bayesian conditional effect off the sealed
    // operation, so the rebound set must reach the dispatched route instead.
    assert!(prepared.checked_bayesian_conditional_operation().is_none());
    let first = prepared.estimate(&data, &ctx).unwrap();
    assert_eq!(first.logical_plan().estimator.as_deref(), Some("conditional.bayesian"));
    assert!(verdict(&first));
    prepared.rebind_custom_validators(validator(false)).unwrap();
    assert!(!verdict(&prepared.estimate(&data, &ctx).unwrap()));
}
