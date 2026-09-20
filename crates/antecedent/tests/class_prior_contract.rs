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
    // A stub posterior without lag masks still refuses; completions are not
    // posterior atoms.
    assert!(err.is_err(), "{err:?}");
}

#[test]
fn temporal_cpdag_graph_posterior_with_lag_masks_licenses_pulse() {
    use antecedent::CellStatus;
    use antecedent_discovery::{GraphPosteriorAtomKind, set_edge};

    let n = 3;
    let contemp = set_edge(set_edge(0, n, 0, 2, true), n, 2, 0, true);
    let lag = (1u64 << 1) | (1u64 << 7);
    let posterior = GraphPosterior::new(
        n,
        [1.0],
        [contemp],
        [0.0; 9],
        [0.0; 9],
        1.0,
        InferenceDiagnostics::analytic("class_prior_contract"),
        0,
    )
    .unwrap()
    .with_atom_kind(GraphPosteriorAtomKind::Cpdag)
    .with_lagged_marginals(1, vec![0.0; n * n])
    .unwrap()
    .with_lag_masks(vec![lag])
    .unwrap();
    let ctx = ExecutionContext::for_tests(1);
    let study = Study::series(class_series())
        .graph_posterior(posterior)
        .query(pulse())
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(8)))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    assert_eq!(study.support_status(), Some(CellStatus::Licensed));
    let result = study.run(&ctx).unwrap();
    assert!(result.structural_response.is_some());
}

/// Three-variable lag-1 series long enough for a class envelope.
fn class_series() -> TimeSeriesData {
    let n = 48usize;
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut z = vec![0.0; n];
    for i in 0..n {
        let phase = f64::from(u32::try_from(i).unwrap()) * 0.37;
        z[i] = phase.sin();
        t[i] = f64::from(u8::from(phase.cos() > 0.0));
        if i > 0 {
            y[i] = 1.4 * t[i - 1] + 0.6 * z[i - 1] + 0.1 * (phase * 1.7).sin();
        }
    }
    TimeSeriesData::from_f64_columns(
        [("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())],
        1,
    )
    .unwrap()
}

/// Shielded `z@1 — t@1` class: more than one completion.
fn class_cpdag() -> TemporalCpdag {
    let mut graph = TemporalCpdag::empty();
    let t1 = graph.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = graph.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let z1 = graph.add_lagged(VariableId::from_raw(2), Lag::from_raw(1)).unwrap();
    graph.insert_directed(z1, y0).unwrap();
    graph.insert_directed(t1, y0).unwrap();
    graph.insert_undirected(z1, t1).unwrap();
    graph
}

fn class_identities(
    prior: Option<ClassPrior>,
    max_completions: Option<usize>,
) -> antecedent_core::ContractIdentities {
    let mut builder = Study::series(class_series())
        .graph(class_cpdag())
        .query(pulse())
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(8)))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0);
    if let Some(prior) = prior {
        builder = builder.class_prior(prior);
    }
    if let Some(cap) = max_completions {
        builder = builder.max_completions(cap);
    }
    let ctx = ExecutionContext::for_tests(1);
    builder.build().unwrap().prepare(&ctx).unwrap().contract().unwrap().identities
}

#[test]
fn class_prior_is_an_identification_premise() {
    let none = class_identities(None, None);
    let even = class_identities(Some(ClassPrior::from_ordered([0.5, 0.5]).unwrap()), None);
    let skewed = class_identities(Some(ClassPrior::from_ordered([0.3, 0.7]).unwrap()), None);
    assert_ne!(
        none.identification, even.identification,
        "a caller-supplied class prior is a premise, not a display option"
    );
    assert_ne!(even.identification, skewed.identification, "the masses themselves are premises");
    assert_ne!(even.program, skewed.program);
    assert_eq!(
        even.inference_binding, skewed.inference_binding,
        "a structural class prior is not a numeric inference knob"
    );
    assert_eq!(even.data_snapshot, skewed.data_snapshot);
    assert_eq!(even.observation, skewed.observation);
    // Keyed masses are a set: the same masses in another key order are one prior.
    let pairs = |order: [(u64, f64); 2]| {
        class_identities(Some(ClassPrior::from_pairs(order).unwrap()), None).identification
    };
    assert_eq!(pairs([(7, 0.3), (9, 0.7)]), pairs([(9, 0.7), (7, 0.3)]));
    assert_ne!(pairs([(7, 0.3), (9, 0.7)]), pairs([(7, 0.7), (9, 0.3)]));
}

#[test]
fn completion_budget_is_a_program_commitment() {
    let uncapped = class_identities(None, None);
    let capped = class_identities(None, Some(1));
    let wider = class_identities(None, Some(4));
    assert_ne!(uncapped.program, capped.program, "a completion cap compiles a different program");
    assert_ne!(capped.program, wider.program);
    assert_eq!(
        capped.identification, wider.identification,
        "the search budget is not a structural premise"
    );
    assert_eq!(capped.inference_binding, wider.inference_binding);
}

#[test]
fn class_prior_and_completion_budget_reach_the_claim_id() {
    let ctx = ExecutionContext::for_tests(11);
    let data = class_series();
    let claim_id = |prior: Option<ClassPrior>, cap: Option<usize>| {
        let mut builder = Study::series(data.clone())
            .graph(class_cpdag())
            .query(pulse())
            .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(8)))
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0);
        if let Some(prior) = prior {
            builder = builder.class_prior(prior);
        }
        if let Some(cap) = cap {
            builder = builder.max_completions(cap);
        }
        let prepared = builder.build().unwrap().prepare(&ctx).unwrap();
        let result = prepared.estimate_series(&data, &ctx).unwrap();
        let claim = result.claim(&prepared.contract().unwrap(), &ctx).unwrap();
        (claim.claim_id, claim.identities)
    };
    let plain = claim_id(None, None);
    let prior = claim_id(Some(ClassPrior::from_ordered([0.3, 0.7]).unwrap()), None);
    let other_prior = claim_id(Some(ClassPrior::from_ordered([0.7, 0.3]).unwrap()), None);
    let capped = claim_id(None, Some(1));
    assert_ne!(plain.0, prior.0, "the class prior a claim was mixed under is part of the claim");
    assert_ne!(prior.0, other_prior.0);
    assert_ne!(plain.0, capped.0, "the completion budget is part of the claim");
    // The claim id moves because the layers it seals moved, not only because
    // the mixed number did.
    assert_ne!(plain.1.identification, prior.1.identification);
    assert_ne!(prior.1.identification, other_prior.1.identification);
    assert_ne!(plain.1.program, capped.1.program);
}
