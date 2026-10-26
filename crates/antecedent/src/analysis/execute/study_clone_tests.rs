//! A `Study` copy is a reference-count bump: configuration, the heavy overlay
//! members, and every prepare-time cache are shared, never deep-copied.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    AverageEffectQuery, CausalQuery, ExecutionContext, Intervention, Lag, ResponseFunctional,
    ResponseQuery, TemporalEffectQuery, TemporalPolicy, TemporalResponseSpec, Value, VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_graph::{Dag, TemporalCpdag};

use crate::analysis::builder::RefuteSuite;
use crate::strategy_table::{EstimatorId, IdentifierId};
use crate::{EstimatorSpec, Study};

fn tabular() -> (TabularData, Study) {
    let n = 64;
    let t: Vec<f64> = (0..n).map(|i| f64::from(u8::from(i % 2 == 0))).collect();
    let z: Vec<f64> = (0..n).map(|i| ((i * 37 % 101) as f64 - 50.0) / 50.0).collect();
    let y: Vec<f64> = (0..n).map(|i| 2.0 * t[i] + z[i] + ((i * 13 % 47) as f64) / 100.0).collect();
    let data = TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
    ])
    .unwrap();
    let graph =
        Dag::from_named_edges(data.schema(), &[("z", "t"), ("z", "y"), ("t", "y")]).unwrap();
    let query = AverageEffectQuery::with_levels(
        data.schema().id_of("t").unwrap(),
        data.schema().id_of("y").unwrap(),
        0.0,
        1.0,
    );
    let study = Study::tabular(data.clone())
        .graph(graph)
        .query(query)
        .estimator(EstimatorSpec::Aipw(Box::new(antecedent_estimate::AipwAte::new())))
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    (data, study)
}

#[test]
fn study_clone_shares_configuration_graph_spec_and_identification() {
    let (_, study) = tabular();
    let prepared = study.prepare(&ExecutionContext::for_tests(5)).unwrap();
    let analysis = prepared.study();
    let cache = analysis.static_cache().expect("prepared static cache");
    let spec = analysis.estimator_spec.as_ref().expect("configured estimator");
    let (cache_refs, spec_refs, config_refs, graph_refs) = (
        Arc::strong_count(cache),
        Arc::strong_count(spec),
        Arc::strong_count(&analysis.config),
        Arc::strong_count(&analysis.graph),
    );

    let copy = analysis.clone();
    assert!(Arc::ptr_eq(&analysis.config, &copy.config));
    assert!(Arc::ptr_eq(&analysis.graph, &copy.graph));
    assert!(Arc::ptr_eq(spec, copy.estimator_spec.as_ref().unwrap()));
    assert!(Arc::ptr_eq(cache, copy.static_cache().unwrap()));
    assert_eq!(Arc::strong_count(cache), cache_refs + 1);
    assert_eq!(Arc::strong_count(spec), spec_refs + 1);
    assert_eq!(Arc::strong_count(&analysis.config), config_refs + 1);
    assert_eq!(Arc::strong_count(&analysis.graph), graph_refs + 1);
}

#[test]
fn rebinding_a_click_overlay_leaves_the_shared_configuration_alone() {
    let (_, study) = tabular();
    let prepared = study.prepare(&ExecutionContext::for_tests(5)).unwrap();
    let analysis = prepared.study();
    let mut click = analysis.clone();
    click.refute = RefuteSuite::PlaceboAndRcc;
    click.estimator = Some(EstimatorId::LinearAdjustmentAte);
    click.identifier = Some(IdentifierId::BackdoorAdjustment);
    click.estimator_spec = None;
    assert!(Arc::ptr_eq(&analysis.config, &click.config));
    assert!(Arc::ptr_eq(
        analysis.static_cache().unwrap(),
        click.static_cache().unwrap()
    ));
    assert_eq!(analysis.refute, RefuteSuite::None, "the source overlay is untouched");
}

#[test]
fn static_ate_operation_shares_the_prepared_identification() {
    let (data, _) = tabular();
    let graph =
        Dag::from_named_edges(data.schema(), &[("z", "t"), ("z", "y"), ("t", "y")]).unwrap();
    let query = AverageEffectQuery::with_levels(
        data.schema().id_of("t").unwrap(),
        data.schema().id_of("y").unwrap(),
        0.0,
        1.0,
    );
    let study =
        Study::tabular(data).graph(graph).query(query).refute(RefuteSuite::None).build().unwrap();
    let prepared = study.prepare(&ExecutionContext::for_tests(5)).unwrap();
    let cache = prepared.study().static_cache().expect("prepared static cache");
    let sealed = prepared
        .sealed_static_identification()
        .expect("static ATE seals an operation that retains its identification");
    assert!(Arc::ptr_eq(cache, sealed), "the operation reads the prepared allocation");

    let click = prepared.study().clone();
    assert!(Arc::ptr_eq(sealed, click.static_cache().unwrap()));
    // study + operation + the click copy.
    assert!(Arc::strong_count(cache) >= 3);
}

#[test]
fn temporal_class_lookup_returns_the_retained_envelope_without_copying() {
    let n = 60;
    let (mut t, mut y, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let phase = i as f64 * 0.37;
        z[i] = phase.sin();
        t[i] = f64::from(u8::from(phase.cos() > 0.0));
        if i > 0 {
            y[i] = 1.4 * t[i - 1] + 0.6 * z[i - 1] + 0.1 * (phase * 1.7).sin();
        }
    }
    let data = antecedent_data::TimeSeriesData::from_f64_columns(
        [("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())],
        1,
    )
    .unwrap();
    let mut graph = TemporalCpdag::empty();
    let t1 = graph.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = graph.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let z1 = graph.add_lagged(VariableId::from_raw(2), Lag::from_raw(1)).unwrap();
    graph.insert_directed(t1, y0).unwrap();
    graph.insert_directed(z1, y0).unwrap();
    let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(1),
        interventions: Arc::from([Intervention::set(VariableId::from_raw(0), Value::f64(1.0))]),
    })
    .with_temporal(TemporalResponseSpec::new([1], TemporalPolicy::pulse(-1), Some(1)).unwrap());
    let study = Study::series(data)
        .graph(graph)
        .query(CausalQuery::Response(query))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let ctx = ExecutionContext::for_tests(3);
    let prepared = std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(move || study.prepare(&ctx).unwrap())
        .unwrap()
        .join()
        .unwrap();
    let analysis = prepared.study();
    let cache = analysis.temporal_class_cache().expect("class cache");
    let (_, retained) = cache.by_horizon.first().expect("one certificate per horizon");

    let probe = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1)
        .with_max_history_lag(Some(1));
    let before = Arc::strong_count(retained);
    let bundle = analysis
        .identify_temporal_class(
            IdentifierId::GeneralizedAdjustment,
            &probe,
            &ExecutionContext::for_tests(91),
        )
        .expect("retained certificate");
    assert!(Arc::ptr_eq(&bundle.envelope, retained), "the envelope is shared, not copied");
    assert_eq!(Arc::strong_count(retained), before + 1);
}

#[test]
fn a_prepared_study_holds_one_identification_product() {
    let (_, study) = tabular();
    let prepared = study.prepare(&ExecutionContext::for_tests(5)).unwrap();
    let analysis = prepared.study();
    let cache = analysis.static_cache().expect("prepared static cache");
    assert!(analysis.pag_cache().is_none());
    assert!(analysis.cpdag_cache().is_none());
    assert!(analysis.temporal_cache().is_none());
    assert!(analysis.temporal_class_cache().is_none());
    assert!(analysis.graph_posterior_cache().is_none());
    assert!(analysis.dbn_posterior_cache().is_none());
    assert!(analysis.temporal_class_posterior_cache().is_none());
    assert!(analysis.transport_cache().is_none());
    let primary = analysis.identification.as_ref().and_then(|product| product.primary());
    assert!(std::ptr::eq(primary.expect("a static product answers for the whole study"), &cache.identification));

    let mut replaced = analysis.clone();
    replaced.install_identification(crate::analysis::IdentificationCache::Static(Arc::clone(cache)));
    assert!(Arc::ptr_eq(cache, replaced.static_cache().unwrap()));
    replaced.clear_identification_where(|product| product.pag_cache().is_some());
    assert!(replaced.static_cache().is_some(), "a product of another kind stays");
    replaced.clear_identification_where(|product| product.static_cache().is_some());
    assert!(replaced.identification.is_none());
    assert!(analysis.static_cache().is_some(), "the source study is untouched");
}
