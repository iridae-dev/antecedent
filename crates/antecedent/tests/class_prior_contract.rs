//! Class-prior contract: not enumeration, not a graph posterior.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{BayesianConfig, ClassPrior, InferenceMode, RefuteSuite, Study};
use antecedent_core::{ExecutionContext, Lag, TemporalEffectQuery, VariableId};
use antecedent_data::TimeSeriesData;
use antecedent_discovery::GraphPosterior;
use antecedent_graph::{TemporalCpdag, TemporalDag, ensure_lagged};
use antecedent_prob::InferenceDiagnostics;

fn series() -> TimeSeriesData {
    TimeSeriesData::from_f64_columns(
        [("t", &[0.0, 1.0, 0.0, 1.0][..]), ("y", &[1.0, 2.0, 1.1, 2.2][..])],
        1,
    )
    .unwrap()
}

fn pulse() -> TemporalEffectQuery {
    TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
}

fn tiny_dag() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let t1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(t1, y0).unwrap();
    graph
}

#[test]
fn class_prior_refuses_frequentist() {
    let err = Study::series(series())
        .graph(tiny_dag())
        .query(pulse())
        .class_prior(ClassPrior::from_ordered([0.5, 0.5]).unwrap())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap_err();
    let message = err.to_string();
    assert!(message.contains("class_prior"), "{message}");
    assert!(message.contains("enumeration"), "{message}");
}

#[test]
fn class_prior_conflicts_with_graph_posterior() {
    let posterior = GraphPosterior::new(
        2,
        [1.0],
        [0_u64],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        1.0,
        InferenceDiagnostics::analytic("class_prior_contract"),
        0,
    )
    .unwrap();
    let err = Study::series(series())
        .graph_posterior(posterior)
        .query(pulse())
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(8)))
        .class_prior(ClassPrior::from_ordered([1.0]).unwrap())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap_err();
    assert!(err.to_string().contains("class_prior"), "{err}");
}

#[test]
fn temporal_cpdag_graph_posterior_stays_refused() {
    let mut graph = TemporalCpdag::empty();
    let t1 = graph.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = graph.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(t1, y0).unwrap();
    let posterior = GraphPosterior::new(
        2,
        [1.0],
        [0_u64],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        1.0,
        InferenceDiagnostics::analytic("class_prior_contract"),
        0,
    )
    .unwrap();
    // graph + graph_posterior is already a conflict; the matrix cell is also refused.
    let both = Study::series(series())
        .graph(antecedent::AcceptedGraph::from(graph))
        .graph_posterior(posterior.clone())
        .query(pulse())
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(8)))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build();
    assert!(both.is_err(), "graph and graph_posterior must stay exclusive");
    let err = Study::series(series())
        .graph_posterior(posterior)
        .query(pulse())
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(8)))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(1));
    // DBN mixer is TemporalDag-shaped; a stub posterior without lag masks refuses
    // rather than treating completions as posterior atoms.
    assert!(err.is_err(), "{err:?}");
}
