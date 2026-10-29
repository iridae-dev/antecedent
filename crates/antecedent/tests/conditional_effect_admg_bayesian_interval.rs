//! The Bayesian `ConditionalEffect` on an ADMG and a `TemporalAdmg` publishes the
//! posterior-quantile interval of the Dirichlet row posterior of the stratified difference
//! `conformance/estimate/admg_conditional_strata` states (7/27), from the retained plan with the
//! builder dropped, under every validation depth.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ConditionalEffectQuery, ExecutionContext, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Admg, DenseNodeId, TemporalAdmg};

const FIXTURE: &str =
    include_str!("../../../conformance/estimate/admg_conditional_strata/expected.json");

fn table() -> TabularData {
    let document: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();
    let mut columns: Vec<Vec<f64>> = vec![Vec::new(); 3];
    for cell in document["table"].as_array().unwrap() {
        let count = usize::try_from(cell["count"].as_u64().unwrap()).unwrap();
        for (column, key) in columns.iter_mut().zip(["z", "x", "y"]) {
            column.extend(std::iter::repeat_n(cell[key].as_f64().unwrap(), count));
        }
    }
    TabularData::from_f64_columns(
        ["z", "x", "y"].iter().zip(&columns).map(|(name, column)| (*name, column.as_slice())),
    )
    .unwrap()
}

fn bow_graph() -> Admg {
    let mut graph = Admg::with_variables(3);
    for (from, to) in [(0, 1), (0, 2), (1, 2)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    graph.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    graph
}

fn conditional() -> CausalQuery {
    let var = VariableId::from_raw;
    CausalQuery::ConditionalEffect(
        ConditionalEffectQuery::try_new(
            AverageEffectQuery::binary_ate(var(1), var(2)).with_effect_modifiers(vec![var(0)]),
        )
        .unwrap(),
    )
}

/// `ConditionalEffect` x `Admg` and `TemporalAdmg` x explicit x Bayesian x none, cheap and full.
#[test]
fn conditional_effect_admg_bayesian_cells_execute_with_a_posterior_quantile_interval() {
    let document: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();
    let expected = document["expected_ate"].as_f64().unwrap();
    let tolerance = document["tolerance"]["bayesian_posterior_mean"].as_f64().unwrap();
    let data = table();
    let context = ExecutionContext::for_tests(44);
    let temporal = TemporalAdmg::from_static_contemporaneous(&bow_graph()).unwrap();
    for variant in 0..2 {
        for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
            let base = Study::tabular(data.clone())
                .query(conditional())
                .inference(InferenceMode::Bayesian(
                    BayesianConfig::conjugate().n_draws(256).prior_scale(1_000.0),
                ))
                .refute(suite)
                .bootstrap_replicates(0);
            let builder =
                if variant == 0 { base.graph(bow_graph()) } else { base.graph(temporal.clone()) };
            let study = builder.clone().build().unwrap();
            let prepared = study.prepare(&context).unwrap();
            drop(builder);
            drop(study);
            assert!(prepared.plan().logical.record.estimator.is_some());
            let click = prepared.estimate(&data, &context).unwrap();
            assert_eq!(
                click.interval().as_ref().map(|binding| binding.method),
                Some(antecedent_core::IntervalMethod::PosteriorQuantile),
                "variant {variant} {suite:?}"
            );
            assert!((click.estimate().as_effect().unwrap().ate - expected).abs() < tolerance);
        }
    }
}
