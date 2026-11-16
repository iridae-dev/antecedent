//! Structure priors and imputation pooling through the compiled discovery query.
//!
//! A per-edge prior and soft tiers declared on a posterior search reach the compiled query, the
//! engine and the provenance of the accepted posterior; the compiled posterior equals the
//! engine's bit for bit and the hand-reweighted uniform posterior; a search without a prior is
//! the query it always was; and the posterior pooled across imputations is the hand mixture of
//! the per-imputation exact posteriors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    reason = "integration test with deterministic fixtures and bitwise comparisons"
)]

use std::collections::HashMap;
use std::sync::Arc;

use antecedent::discovery::{
    AdjacencyMask, BayesianDiscoverParams, EdgePriors, GraphPosterior, GraphPrior, SoftTiers,
};
use antecedent::{CompiledExecution, DiscoveredStructure, DiscoverySpec, Study};
use antecedent_core::{CausalRng, DiscoveryMethod, ExecutionContext, VariableId};
use antecedent_data::{SamplingRegularity, TabularData, TimeIndex, TimeSeriesData};

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn three() -> Vec<VariableId> {
    vec![v(0), v(1), v(2)]
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(0x5EED_0031)
}

fn on_large_stack(name: &str, body: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name(name.into())
        .stack_size(32 * 1024 * 1024)
        .spawn(body)
        .unwrap()
        .join()
        .unwrap();
}

/// `z -> t -> y`, `z -> y` with uniform noise.
fn chain_table(n: usize, seed: u64) -> TabularData {
    let mut rng = CausalRng::from_seed(seed);
    let mut uniform = || 2.0 * rng.next_f64() - 1.0;
    let (mut z, mut t, mut y) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..n {
        let zi = uniform();
        let ti = 0.8 * zi + uniform();
        let yi = 1.5 * ti + 0.6 * zi + uniform();
        z.push(zi);
        t.push(ti);
        y.push(yi);
    }
    TabularData::from_f64_columns([("z", z.as_slice()), ("t", t.as_slice()), ("y", y.as_slice())])
        .unwrap()
}

/// Per-edge probabilities on two pairs and a soft order that puts `z` before `t` and `y`.
fn declared() -> BayesianDiscoverParams {
    let tiers: Vec<Arc<[VariableId]>> = vec![Arc::from(vec![v(0)]), Arc::from(vec![v(1), v(2)])];
    let prior = GraphPrior::uniform()
        .with_edge_priors(EdgePriors::new(vec![(v(0), v(1), 0.2), (v(1), v(2), 0.7)]))
        .with_soft_tiers(SoftTiers::with_log_penalty(tiers, 1.5));
    BayesianDiscoverParams { prior, ..BayesianDiscoverParams::default() }
}

/// The prior mass of one DAG under [`declared`], by hand: each listed edge contributes its
/// probability when present and its complement when absent, and each present edge into `z` from
/// `t` or `y` (against the soft order) is charged `exp(-1.5)`. Unlisted edges are neutral.
fn hand_prior_factor(atom: &AdjacencyMask) -> f64 {
    let present = |from: usize, to: usize| atom.has_edge(3, from, to);
    let mut factor = 1.0;
    factor *= if present(0, 1) { 0.2 } else { 0.8 };
    factor *= if present(1, 2) { 0.7 } else { 0.3 };
    for from in [1, 2] {
        if present(from, 0) {
            factor *= (-1.5_f64).exp();
        }
    }
    factor
}

fn posterior_of(execution: &CompiledExecution) -> GraphPosterior {
    let Some(DiscoveredStructure::Posterior(posterior)) =
        execution.result().discovery().map(|payload| payload.structure.clone())
    else {
        panic!("the exact search returns a posterior");
    };
    posterior
}

#[test]
fn a_declared_prior_is_the_query_the_engine_and_the_hand_reweighting() {
    on_large_stack("structure-prior", || {
        let data = chain_table(300, 5);
        let params = declared();
        let spec = DiscoverySpec::exact_dag_posterior(params.clone());
        let plain = DiscoverySpec::exact_dag_posterior(BayesianDiscoverParams::default());

        // The compiled query states the prior; a search without one is the query it was.
        let query = spec.query(three()).unwrap();
        let declared_prior = query.method.structure_prior().expect("the query carries the prior");
        assert_eq!(declared_prior.edges.len(), 2);
        assert_eq!(declared_prior.soft_tier_log_penalty, Some(1.5));
        assert!(declared_prior.statement().contains("v0->v1:0.2"));
        let plain_query = plain.query(three()).unwrap();
        assert_eq!(plain_query.method, DiscoveryMethod::ExactDagPosterior { prior: None });
        assert_ne!(query.bounds.configuration, plain_query.bounds.configuration);

        // The compiled posterior equals the engine's bit for bit.
        let study = Study::tabular(data.clone()).discovery(spec, three()).build().unwrap();
        let execution = CompiledExecution::run(&study, &ctx()).unwrap();
        let compiled = posterior_of(&execution);
        let direct =
            antecedent::discovery::discover_exact_dag_posterior(&data, &three(), &params, &ctx())
                .unwrap();
        let bits = |weights: &[f64]| weights.iter().map(|w| w.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&compiled.weights()[..]), bits(&direct.weights()[..]));
        assert_eq!(compiled.atoms(), direct.atoms());

        // And it is the uniform posterior reweighted by the hand-computed prior factors.
        let uniform = antecedent::discovery::discover_exact_dag_posterior(
            &data,
            &three(),
            &BayesianDiscoverParams::default(),
            &ctx(),
        )
        .unwrap();
        assert_eq!(uniform.atoms(), compiled.atoms());
        let reweighted: Vec<f64> = uniform
            .weights()
            .iter()
            .zip(uniform.atoms().iter())
            .map(|(w, atom)| w * hand_prior_factor(atom))
            .collect();
        let total: f64 = reweighted.iter().sum();
        for (index, weight) in compiled.weights().iter().enumerate() {
            let expected = reweighted[index] / total;
            assert!(
                (weight - expected).abs() <= 1e-9 * expected.max(1e-12) + 1e-12,
                "atom {index}: compiled {weight} against hand-reweighted {expected}"
            );
        }

        // The accepted posterior states the prior, and a study on it carries the statement.
        let accepted = execution.accept_posterior(&ctx()).unwrap();
        let statement = accepted.provenance().structure_prior.clone().unwrap();
        assert_eq!(&*statement, declared_prior.statement());
        let no_prior = Study::tabular(data.clone()).discovery(plain, three()).build().unwrap();
        let plain_accepted =
            CompiledExecution::run(&no_prior, &ctx()).unwrap().accept_posterior(&ctx()).unwrap();
        assert!(plain_accepted.provenance().structure_prior.is_none());
        let on_discovered = Study::tabular(data)
            .accepted_graph_posterior(accepted)
            .query(antecedent::AverageEffectQuery::binary_ate(v(1), v(2)))
            .inference(antecedent::InferenceMode::Bayesian(antecedent::BayesianConfig::conjugate()))
            .refute(antecedent::RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap();
        let contract = on_discovered.inspect().unwrap();
        assert_eq!(
            contract.structure_provenance.as_deref().and_then(|p| p.structure_prior.as_deref()),
            Some(&*statement)
        );
    });
}

#[test]
fn the_search_variants_that_take_a_prior_each_carry_it() {
    let params = declared();
    let schedule = antecedent::discovery::GraphMcmcSchedule::default();
    let statement = |spec: &DiscoverySpec| {
        spec.query(three())
            .unwrap()
            .method
            .structure_prior()
            .map(antecedent_core::StructurePriorSpec::statement)
    };
    let expected = statement(&DiscoverySpec::exact_dag_posterior(params.clone()));
    assert!(expected.is_some());
    for spec in [
        DiscoverySpec::exact_dag_marginals(params.clone()),
        DiscoverySpec::order_mcmc(params.clone(), schedule, false),
        DiscoverySpec::structure_mcmc(params.clone(), schedule),
        DiscoverySpec::dbn_posterior(params.clone(), 1, false, schedule),
        DiscoverySpec::exact_dag_posterior(params.clone()).equivalence_classes(),
    ] {
        assert_eq!(statement(&spec), expected, "{}", spec.method().id());
    }
}

#[test]
fn a_pooled_posterior_is_the_hand_mixture_of_the_per_imputation_exact_posteriors() {
    on_large_stack("pooled-imputations", || {
        let params = BayesianDiscoverParams::default();
        let posteriors: Vec<GraphPosterior> = [11_u64, 12, 13]
            .iter()
            .map(|&seed| {
                antecedent::discovery::discover_exact_dag_posterior(
                    &chain_table(120, seed),
                    &three(),
                    &params,
                    &ctx(),
                )
                .unwrap()
            })
            .collect();
        let tables: Vec<TabularData> =
            [11_u64, 12, 13].iter().map(|&seed| chain_table(120, seed)).collect();
        let pooled = antecedent::discovery::discover_pooled_exact_dag_posterior(
            &tables,
            &three(),
            &params,
            &ctx(),
        )
        .unwrap();

        // The mixture, by hand: the mean over imputations of each graph's mass.
        let mut by_graph: HashMap<AdjacencyMask, f64> = HashMap::new();
        for posterior in &posteriors {
            for (atom, weight) in posterior.atoms().iter().zip(posterior.weights().iter()) {
                *by_graph.entry(atom.clone()).or_insert(0.0) += weight / 3.0;
            }
        }
        assert_eq!(pooled.n_graphs(), by_graph.len());
        for (atom, weight) in pooled.atoms().iter().zip(pooled.weights().iter()) {
            let expected = by_graph[atom];
            assert!((weight - expected).abs() <= 1e-12, "{weight} against hand mixture {expected}");
        }
        // The edge marginals are the means of the per-imputation marginals.
        for cell in 0..9 {
            let mean: f64 = posteriors.iter().map(|p| p.edge_marginals()[cell]).sum::<f64>() / 3.0;
            assert!((pooled.edge_marginals()[cell] - mean).abs() <= 1e-12, "cell {cell}");
        }
        assert_eq!(
            pooled.algorithm().map(|a| &**a),
            Some("pooled_imputations(exact_dag_posterior)")
        );

        // One imputation pools to itself; no posterior, or atoms of another class, are refused.
        let alone = GraphPosterior::pooled_over_imputations(&posteriors[..1]).unwrap();
        for (a, b) in alone.weights().iter().zip(posteriors[0].weights().iter()) {
            assert!((a - b).abs() <= 1e-12);
        }
        assert!(GraphPosterior::pooled_over_imputations(&[]).is_err());
    });
}

#[test]
fn pooled_exact_marginals_are_the_means_of_the_per_imputation_marginals() {
    on_large_stack("pooled-marginals", || {
        let params = BayesianDiscoverParams::default();
        let tables: Vec<TabularData> =
            [21_u64, 22].iter().map(|&seed| chain_table(150, seed)).collect();
        let parts: Vec<_> = tables
            .iter()
            .map(|table| {
                antecedent::discovery::discover_exact_dag_marginals(
                    table,
                    &three(),
                    &params,
                    &ctx(),
                )
                .unwrap()
            })
            .collect();
        let pooled = antecedent::discovery::discover_pooled_exact_dag_marginals(
            &tables,
            &three(),
            &params,
            &ctx(),
        )
        .unwrap();
        for cell in 0..9 {
            let mean = f64::midpoint(parts[0].edge_marginals[cell], parts[1].edge_marginals[cell]);
            assert!((pooled.edge_marginals[cell] - mean).abs() <= 1e-12, "cell {cell}");
        }
        let hand = f64::midpoint(parts[0].log_evidence.exp(), parts[1].log_evidence.exp()).ln();
        assert!((pooled.log_evidence - hand).abs() <= 1e-9 * hand.abs().max(1.0));
        assert!(
            antecedent::discovery::discover_pooled_exact_dag_marginals(
                &[],
                &three(),
                &params,
                &ctx()
            )
            .is_err()
        );
    });
}

/// The pooled posterior of a sampled search is the equal-weight mixture of the per-imputation
/// draw sets, graph by graph, and the per-imputation diagnostics are kept.
#[test]
fn pooled_sampled_posteriors_are_the_mixture_of_the_per_imputation_draw_sets() {
    on_large_stack("pooled-samplers", || {
        let params = BayesianDiscoverParams::default();
        let schedule = antecedent::discovery::GraphMcmcSchedule {
            n_chains: 4,
            n_warmup: 500,
            n_draws: 2_000,
            thin: 1,
        };
        let tables: Vec<TabularData> =
            [31_u64, 32, 33].iter().map(|&seed| chain_table(150, seed)).collect();
        let singles: Vec<GraphPosterior> = tables
            .iter()
            .map(|table| {
                antecedent::discovery::discover_structure_mcmc(
                    table,
                    &three(),
                    &params,
                    &schedule,
                    &ctx(),
                )
                .unwrap()
            })
            .collect();
        let pooled = antecedent::discovery::discover_pooled_structure_mcmc(
            &tables,
            &three(),
            &params,
            &schedule,
            &ctx(),
        )
        .unwrap();
        let mut by_graph: HashMap<AdjacencyMask, f64> = HashMap::new();
        for posterior in &singles {
            let total: f64 = posterior.weights().iter().sum();
            for (atom, weight) in posterior.atoms().iter().zip(posterior.weights().iter()) {
                *by_graph.entry(atom.clone()).or_insert(0.0) += weight / total / 3.0;
            }
        }
        assert_eq!(pooled.n_graphs(), by_graph.len());
        let mass: f64 = pooled.weights().iter().sum();
        for (atom, weight) in pooled.atoms().iter().zip(pooled.weights().iter()) {
            assert!((weight / mass - by_graph[atom]).abs() <= 1e-12);
        }
        assert!(pooled.algorithm().is_some_and(|a| a.starts_with("pooled_imputations")));
        let notes = &pooled.diagnostics().notes;
        assert_eq!(notes.iter().filter(|note| note.starts_with("imputation ")).count(), 3);
        assert!(notes.iter().any(|note| note.contains("equal_weight_mixture_of_per_imputation")));
    });
}

fn lag1_series(n: usize, seed: u64) -> TimeSeriesData {
    let mut rng = CausalRng::from_seed(seed);
    let mut normal = || {
        let u1 = rng.next_f64().max(1e-12);
        let u2 = rng.next_f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    };
    let mut x = vec![0.0; n];
    let mut y = vec![0.0; n];
    for t in 1..n {
        x[t] = normal();
        y[t] = 0.8 * x[t - 1] + 0.2 * normal();
    }
    let data = TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())]).unwrap();
    TimeSeriesData::try_new(
        data.storage().clone(),
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap()
}

/// A pooled template posterior is the hand mixture keyed by atom and lag mask, its lagged
/// marginals are the means of the per-imputation ones, and posteriors over different lag
/// windows (or a template beside a static posterior) are refused.
#[test]
fn a_pooled_dbn_posterior_is_the_hand_mixture_keyed_by_atom_and_lag_mask() {
    on_large_stack("pooled-dbn", || {
        let params = BayesianDiscoverParams::default();
        let schedule = antecedent::discovery::GraphMcmcSchedule::default();
        let vars = vec![v(0), v(1)];
        let series: Vec<TimeSeriesData> =
            [41_u64, 42, 43].iter().map(|&seed| lag1_series(200, seed)).collect();
        let singles: Vec<GraphPosterior> = series
            .iter()
            .map(|one| {
                antecedent::discovery::discover_dbn_posterior(
                    one,
                    &vars,
                    &params,
                    1,
                    false,
                    &schedule,
                    &ctx(),
                )
                .unwrap()
            })
            .collect();
        let pooled = antecedent::discovery::discover_pooled_dbn_posterior(
            &series,
            &vars,
            &params,
            1,
            false,
            &schedule,
            &ctx(),
        )
        .unwrap();

        let key = |posterior: &GraphPosterior, index: usize| {
            (
                posterior.atoms()[index].clone(),
                posterior.lag_masks().expect("a template posterior has lag masks")[index].clone(),
            )
        };
        let mut by_template: HashMap<(AdjacencyMask, AdjacencyMask), f64> = HashMap::new();
        for posterior in &singles {
            let total: f64 = posterior.weights().iter().sum();
            for (index, weight) in posterior.weights().iter().enumerate() {
                *by_template.entry(key(posterior, index)).or_insert(0.0) += weight / total / 3.0;
            }
        }
        assert_eq!(pooled.n_graphs(), by_template.len());
        let mass: f64 = pooled.weights().iter().sum();
        for (index, weight) in pooled.weights().iter().enumerate() {
            let expected = by_template[&key(&pooled, index)];
            assert!((weight / mass - expected).abs() <= 1e-12, "template {index}");
        }
        assert_eq!(pooled.max_lag(), Some(1));
        let lagged = pooled.lagged_edge_marginals().expect("lagged marginals pool");
        for (cell, value) in lagged.iter().enumerate() {
            let mean: f64 =
                singles.iter().map(|p| p.lagged_edge_marginals().unwrap()[cell]).sum::<f64>() / 3.0;
            assert!((value - mean).abs() <= 1e-12, "lagged cell {cell}");
        }

        // Different lag windows do not pool, and neither does a template with a static posterior.
        let wider = antecedent::discovery::discover_dbn_posterior(
            &series[0],
            &vars,
            &params,
            2,
            false,
            &schedule,
            &ctx(),
        )
        .unwrap();
        assert!(GraphPosterior::pooled_over_imputations(&[singles[0].clone(), wider]).is_err());
        let static_one = antecedent::discovery::discover_exact_dag_posterior(
            &chain_table(100, 3),
            &three()[..2],
            &params,
            &ctx(),
        )
        .unwrap();
        assert!(
            GraphPosterior::pooled_over_imputations(&[singles[0].clone(), static_one]).is_err()
        );
    });
}
