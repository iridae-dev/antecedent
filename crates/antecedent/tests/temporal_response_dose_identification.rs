//! A single-step temporal pulse identifies at the dose it was asked for, not at do(T=1).
use antecedent::{RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ExecutionContext, Intervention, Lag, ResponseFunctional, ResponseQuery,
    TemporalPolicy, TemporalResponseSpec, Value, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_graph::{TemporalDag, ensure_lagged};
use std::sync::Arc;

#[allow(clippy::cast_precision_loss)]
fn identified_active_and_control(level: f64) -> (String, String) {
    let n = 120;
    let x = (0..n).map(|i| (i as f64 * 0.7).sin()).collect::<Vec<_>>();
    let y = std::iter::once(0.0).chain(x.iter().copied().take(n - 1)).collect::<Vec<_>>();
    let data =
        TimeSeriesData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())], 1).unwrap();
    let mut graph = TemporalDag::empty();
    let x_lag = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y_now = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(x_lag, y_now).unwrap();
    let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(1),
        interventions: Arc::from([Intervention::set(VariableId::from_raw(0), Value::f64(level))]),
    })
    .with_temporal(TemporalResponseSpec::new(vec![1], TemporalPolicy::pulse(-1), None).unwrap());
    let prepared = Study::series(data)
        .graph(graph)
        .query(CausalQuery::Response(query))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ExecutionContext::for_tests(3))
        .unwrap();
    let cache = prepared.temporal_identification().expect("temporal identification cached");
    let ate = cache.by_horizon[0].identification.average_effect().expect("average effect").clone();
    (format!("{:?}", ate.active), format!("{:?}", ate.control))
}

#[test]
fn pulse_identifies_at_requested_dose() {
    let (active, control) = identified_active_and_control(2.7);
    assert!(active.contains("2.7"), "active arm must carry the requested dose: {active}");
    assert!(control.contains('0') && !control.contains("2.7"), "control stays at 0: {control}");
}
