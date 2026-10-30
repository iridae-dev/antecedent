//! Diagnostic lifecycle invariance and prepared-context integration.
//! SPDX-License-Identifier: MIT OR Apache-2.0
use antecedent::{DataDiagnostics, EstimatorId, RefuteSuite, Study};
use antecedent_core::{AverageEffectQuery, ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};

fn study(diagnostics: Option<DataDiagnostics>) -> Study {
    let t: Vec<f64> = (0..200).map(|i| f64::from(i % 2)).collect();
    let y: Vec<f64> = (0..200).map(|i| 2. * t[i] + (i as f64).sin()).collect();
    let data = TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice())]).unwrap();
    let mut graph = Dag::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    Study::tabular(data)
        .graph(graph)
        .query(AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)))
        .estimator(EstimatorId::LinearAdjustmentAte)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .data_diagnostics(diagnostics)
        .build()
        .unwrap()
}

#[test]
fn automatic_and_explicit_reports_do_not_change_scientific_contracts_or_estimates() {
    let ctx = ExecutionContext::for_tests(7);
    let enabled = study(Some(DataDiagnostics::default())).prepare_deferred(&ctx).unwrap();
    let disabled = study(None).prepare_deferred(&ctx).unwrap();
    assert!(disabled.data_diagnostics().is_none());
    let report = enabled.data_diagnostics().unwrap();
    assert_eq!(report.context.roles["t"], "treatment");
    assert_eq!(report.context.roles["y"], "outcome");
    assert_eq!(report, &disabled.diagnose_data(&DataDiagnostics::default(), &ctx).unwrap());
    let a = enabled.prepared_inspection().unwrap().summary();
    let b = disabled.prepared_inspection().unwrap().summary();
    assert_eq!(a["identities"], b["identities"]);
    assert!(a["data_diagnostics"].is_object());
}

#[test]
fn rebound_data_rebuilds_evidence_and_relationship_limits_are_explicit() {
    let ctx = ExecutionContext::for_tests(7);
    let prepared =
        study(Some(DataDiagnostics { max_sample_rows: Some(100), ..DataDiagnostics::default() }))
            .prepare_deferred(&ctx)
            .unwrap();
    assert!(
        prepared
            .data_diagnostics()
            .unwrap()
            .checks
            .iter()
            .any(|c| c.code == "data.relationships" && c.status == "not_evaluated")
    );
    let data =
        TabularData::from_f64_columns([("t", &[0., 1., 0., 1.][..]), ("y", &[0., 2., 1., 3.][..])])
            .unwrap();
    let rebound = prepared.rebind_data(data, &ctx).unwrap();
    assert_ne!(
        prepared.data_diagnostics().unwrap().identity,
        rebound.data_diagnostics().unwrap().identity
    );
    assert_eq!(rebound.data_diagnostics().unwrap().rows_total, 4);
}

#[test]
fn series_refresh_replaces_evidence_atomically_and_cancelled_refresh_keeps_it() {
    use antecedent_core::{Lag, TemporalEffectQuery};
    use antecedent_data::TimeSeriesData;
    use antecedent_graph::{TemporalDag, ensure_lagged};

    let series = |shift: f64| {
        let t: Vec<f64> = (0..80).map(|i| (f64::from(i) * 0.37).sin()).collect();
        let y: Vec<f64> = (0..80)
            .map(|i| {
                shift + if i == 0 { 0.0 } else { 1.5 * t[i - 1] } + (i as f64 * 0.61).cos() * 0.01
            })
            .collect();
        TimeSeriesData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice())], 1).unwrap()
    };
    let mut graph = TemporalDag::empty();
    let t1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(t1, y0).unwrap();
    let ctx = ExecutionContext::for_tests(7);
    let mut prepared = Study::series(series(0.0))
        .graph(graph)
        .query(
            TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
                .with_horizon_steps(1),
        )
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let original = prepared.data_diagnostics().unwrap().clone();
    prepared.refresh_series(series(1.0), &ctx).unwrap();
    let refreshed = prepared.data_diagnostics().unwrap().clone();
    assert_ne!(original.identity, refreshed.identity);
    assert_ne!(original.snapshot_identity, refreshed.snapshot_identity);
    assert_eq!(original.rows_total, 80);
    ctx.cancellation.cancel();
    assert!(prepared.refresh_series(series(2.0), &ctx).is_err());
    assert_eq!(prepared.data_diagnostics().unwrap().identity, refreshed.identity);
}
