//! Bayesian quadratic-basis g-computation through the public prepared Study route.

use antecedent::{BayesianConfig, EstimatorId, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ConditionalEffectQuery, ExecutionContext, VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_graph::{Dag, DenseNodeId};

fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

fn d(i: u32) -> DenseNodeId {
    DenseNodeId::from_raw(i)
}

fn fixture() -> TabularData {
    let n = 400;
    let mut treatment = Vec::with_capacity(n);
    let mut outcome = Vec::with_capacity(n);
    let mut modifier = Vec::with_capacity(n);
    for i in 0..n {
        let z = if i % 2 == 0 { -1.0 } else { 1.0 };
        let t = f64::from((i / 2) % 2 == 1);
        let noise = ((i * 37 % 101) as f64 - 50.0) / 500.0;
        modifier.push(z);
        treatment.push(t);
        outcome.push(2.0 + 0.5 * t + 0.4 * z + 0.75 * t * z + noise);
    }
    TabularData::from_f64_columns([
        ("t", treatment.as_slice()),
        ("y", outcome.as_slice()),
        ("z", modifier.as_slice()),
    ])
    .unwrap()
}

fn dag() -> Dag {
    let mut graph = Dag::with_variables(3);
    for (from, to) in [(2, 0), (2, 1), (0, 1)] {
        graph.insert_directed(d(from), d(to)).unwrap();
    }
    graph
}

fn run(
    query: impl Into<CausalQuery>,
) -> (Study, antecedent::PreparedStudy, antecedent::StudyResult, TabularData) {
    let data = fixture();
    let study = Study::tabular(data.clone())
        .graph(dag())
        .query(query)
        .estimator(EstimatorId::BayesianBasisGcomp)
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(256).prior_scale(20.0),
        ))
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let ctx = ExecutionContext::for_tests(17);
    let prepared = study.prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    (study, prepared, result, data)
}

#[test]
fn staged_ate_and_cate_keep_shared_draws_and_round_trip_posterior_artifact() {
    let (_study, prepared, result, data) = run(AverageEffectQuery::binary_ate(v(0), v(1)));
    assert_eq!(result.logical_plan.estimator.as_deref(), Some("bayesian.basis.gcomp"));
    assert_eq!(result.support_status, Some(antecedent::CellStatus::Licensed));
    assert!((result.estimate.ate - 0.5).abs() < 0.08, "ATE={}", result.estimate.ate);
    let posterior = result.posterior.as_ref().expect("posterior");
    assert_eq!(posterior.draws.n_draws, 256);
    assert_eq!(posterior.draws.schema.quantities.len(), 401);
    assert!(posterior.summaries.q025[0] < posterior.summaries.q975[0]);

    let encoded = antecedent_io::encode_causal_posterior_bytes(posterior, "basis-stage").unwrap();
    let (wire, consumed) = antecedent_io::decode_causal_posterior_bytes(&encoded).unwrap();
    assert_eq!(wire.n_draws, 256);
    assert_eq!(consumed.as_slice(), posterior.draws.values.as_ref());

    let ctx = ExecutionContext::for_tests(19);
    let artifact = prepared.encode_contracted_result(&result, "basis-stage-result", &ctx);
    let artifact = artifact.expect("prepared result export");
    let (_, _, wire) = antecedent_io::decode_analysis_result_artifact(&artifact).unwrap();
    let exported_posterior = wire.posterior_artifact.expect("posterior payload");
    let (exported_meta, exported_draws) =
        antecedent_io::decode_causal_posterior_bytes(&exported_posterior).unwrap();
    assert_eq!(exported_meta.n_draws, 256);
    assert_eq!(exported_draws.as_slice(), posterior.draws.values.as_ref());

    let cate_query = ConditionalEffectQuery::try_new(
        AverageEffectQuery::binary_ate(v(0), v(1)).with_effect_modifiers([v(2)]),
    )
    .unwrap();
    let (_, _, cate, _) = run(cate_query);
    assert_eq!(cate.logical_plan.estimator.as_deref(), Some("bayesian.basis.gcomp"));
    let values = cate.estimate.cate.as_ref().expect("row CATE summary");
    assert_eq!(values.len(), data.row_count());
    assert!((values[0] - (0.5 - 0.75)).abs() < 0.12, "low-z CATE={}", values[0]);
    assert!((values[1] - (0.5 + 0.75)).abs() < 0.12, "high-z CATE={}", values[1]);
    let cate_posterior = cate.posterior.as_ref().unwrap();
    for row in 0..values.len() {
        assert_eq!(cate_posterior.draws.column(row + 1).unwrap().len(), 256);
    }
}
