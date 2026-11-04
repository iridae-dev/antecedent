//! Known-truth checks for the ADMG front-door `AverageEffect` and `InterventionResponse`
//! under cheap and full validation (graph posterior, explicit and accepted structure).
//!
//! The expected values are the front-door formula evaluated on the fixture's enumerated
//! binary table, `conformance/estimate/admg_frontdoor_functional`:
//! `E[Y | do(t)] = sum_m P(m | t) sum_t' P(t') P(Y = 1 | t', m)`, whose contrast is the
//! fixture's `expected_ate` 0.282. The formula is evaluated here from the counts alone and
//! shares no code with the product. Each test executes every coordinate it cites.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    reason = "the fixture contains bounded nonnegative categorical values and one scenario per body"
)]

use std::collections::BTreeMap;
use std::sync::Arc;

use antecedent::{AcceptedGraph, BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ExecutionContext, Intervention, ResponseFunctional,
    ResponseIdentification, ResponseQuery, ResponseValue, Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_discovery::adjacency_mask::{adjacency_mask_from_admg, set_edge};
use antecedent_discovery::{
    AdjacencyMask, GraphPosterior, GraphPosteriorAtomKind, SamplingSummary,
};
use antecedent_graph::{Admg, DenseNodeId};
use antecedent_prob::InferenceDiagnostics;

const CHECKED: [RefuteSuite; 2] = [RefuteSuite::Cheap, RefuteSuite::Full];

fn pin() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../conformance/estimate/admg_frontdoor_functional/expected.json"
    ))
    .unwrap()
}

fn expand_contingency(pin: &serde_json::Value) -> TabularData {
    let columns: Vec<&str> =
        pin["columns"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    let mut values: Vec<Vec<f64>> = vec![Vec::new(); columns.len()];
    for cell in pin["contingency_table"].as_array().unwrap() {
        let count = usize::try_from(cell["count"].as_u64().unwrap()).unwrap();
        for (i, name) in columns.iter().enumerate() {
            values[i].extend(std::iter::repeat_n(cell[*name].as_f64().unwrap(), count));
        }
    }
    let pairs: Vec<(&str, &[f64])> =
        columns.iter().zip(values.iter()).map(|(name, col)| (*name, col.as_slice())).collect();
    TabularData::from_f64_columns(pairs).unwrap()
}

fn node(columns: &[&str], name: &str) -> DenseNodeId {
    DenseNodeId::from_raw(u32::try_from(columns.iter().position(|c| *c == name).unwrap()).unwrap())
}

fn admg_from_pin(pin: &serde_json::Value) -> Admg {
    let columns: Vec<&str> =
        pin["columns"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    let mut admg = Admg::with_variables(u32::try_from(columns.len()).unwrap());
    for edge in pin["graph"]["directed_edges"].as_array().unwrap() {
        admg.insert_directed(
            node(&columns, edge[0].as_str().unwrap()),
            node(&columns, edge[1].as_str().unwrap()),
        )
        .unwrap();
    }
    for edge in pin["graph"]["bidirected_edges"].as_array().unwrap() {
        admg.insert_bidirected(
            node(&columns, edge[0].as_str().unwrap()),
            node(&columns, edge[1].as_str().unwrap()),
        )
        .unwrap();
    }
    admg
}

fn query_from_pin(pin: &serde_json::Value) -> AverageEffectQuery {
    let columns: Vec<&str> =
        pin["columns"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    AverageEffectQuery::with_levels(
        VariableId::from_raw(node(&columns, pin["query"]["treatment"].as_str().unwrap()).raw()),
        VariableId::from_raw(node(&columns, pin["query"]["outcome"].as_str().unwrap()).raw()),
        pin["query"]["control_level"].as_f64().unwrap(),
        pin["query"]["active_level"].as_f64().unwrap(),
    )
}

/// A directed 3-cycle: not an ADMG, so its posterior mass stays unidentified.
fn cyclic_mask(n: usize) -> AdjacencyMask {
    let mut mask = set_edge(AdjacencyMask::EMPTY, n, 0, 1, true);
    mask = set_edge(mask, n, 1, 2, true);
    set_edge(mask, n, 2, 0, true)
}

fn admg_posterior(n: usize, weights: &[f64], masks: &[AdjacencyMask]) -> GraphPosterior {
    GraphPosterior::from_atoms(
        n,
        weights.to_vec(),
        masks.to_vec(),
        vec![0.0; n * n],
        vec![0.0; n * n],
        SamplingSummary {
            ess: 1.0,
            diagnostics: InferenceDiagnostics::analytic("admg_frontdoor_validation_known_truth"),
            rejected_invalid: 0,
        },
    )
    .unwrap()
    .with_atom_kind(GraphPosteriorAtomKind::Admg)
}

fn level_query(level: f64) -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(2),
        interventions: Arc::from([Intervention::set(VariableId::from_raw(0), Value::f64(level))]),
    })
}

fn response_values(result: &antecedent::StudyResult) -> Vec<f64> {
    match &result.response().expect("response payload").estimate {
        ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) => {
            mean.to_vec()
        }
        ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) => vec![*value],
        other => panic!("unexpected response estimate {other:?}"),
    }
}

/// `E[Y | do(t)] = sum_m P(m | t) sum_t' P(t') P(Y = 1 | t', m)` over the fixture counts.
fn frontdoor_interventional_mean(pin: &serde_json::Value, level: f64) -> f64 {
    let mut count = BTreeMap::<(u8, u8, u8), f64>::new();
    for cell in pin["contingency_table"].as_array().unwrap() {
        let key = |name: &str| {
            u8::try_from(
                cell[name].as_u64().unwrap_or_else(|| cell[name].as_f64().unwrap().round() as u64),
            )
            .unwrap()
        };
        *count.entry((key("t"), key("m"), key("y"))).or_default() +=
            cell["count"].as_u64().unwrap() as f64;
    }
    let c = |t: u8, m: u8, y: u8| count.get(&(t, m, y)).copied().unwrap_or(0.0);
    let total: f64 = count.values().sum();
    let t_total = |t: u8| {
        (0..2).flat_map(|m| (0..2).map(move |y| (m, y))).map(|(m, y)| c(t, m, y)).sum::<f64>()
    };
    let tm = |t: u8, m: u8| c(t, m, 0) + c(t, m, 1);
    let level = level as u8;
    (0..2u8)
        .map(|m| {
            let p_m = tm(level, m) / t_total(level);
            let inner: f64 = (0..2u8).map(|t| (t_total(t) / total) * (c(t, m, 1) / tm(t, m))).sum();
            p_m * inner
        })
        .sum()
}

const DRAWS: usize = 1_024;
const SEEDS: [u64; 3] = [1, 2, 3];

/// Monte-Carlo plus prior tolerance of a Bayesian posterior mean of a functional of binary
/// proportions on `N` table rows. Every functional here is a contrast or level of proportions
/// bounded in [0, 1]; a difference of two proportions from `N/2` rows each has sd at most
/// `1/sqrt(N)`, and the front-door composition at most doubles it, so `sd <= 2/sqrt(N)`.
/// The posterior mean of `DRAWS` independent draws then lies within `5 sd / sqrt(DRAWS)` of
/// the posterior mean, which differs from the plug-in truth by at most the nonlinearity bias
/// `sd^2` (the prior scale is made negligible with `prior_scale = 1e6`).
fn bayesian_tolerance(pin: &serde_json::Value) -> f64 {
    let rows: u64 = pin["contingency_table"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["count"].as_u64().unwrap())
        .sum();
    let sd = 2.0 / (rows as f64).sqrt();
    5.0 * sd / (DRAWS as f64).sqrt() + sd * sd
}

fn modes(pin: &serde_json::Value) -> [(&'static str, InferenceMode, f64); 2] {
    [
        ("frequentist", InferenceMode::Frequentist, 1e-9),
        (
            "bayesian",
            InferenceMode::Bayesian(
                BayesianConfig::conjugate().n_draws(DRAWS).prior_scale(1_000_000.0),
            ),
            bayesian_tolerance(pin),
        ),
    ]
}

/// `AverageEffect x Admg x graph_posterior x {Frequentist, Bayesian} x {cheap, full}`: the
/// identified front-door atom publishes the fixture's closed-form 0.282 and the cyclic atom's
/// 0.2 stays unidentified mass.
#[test]
fn admg_graph_posterior_average_effect_matches_the_frontdoor_formula_under_cheap_and_full() {
    let pin = pin();
    let data = expand_contingency(&pin);
    let query = query_from_pin(&pin);
    let admg = admg_from_pin(&pin);
    let n = admg.node_count();
    let truth = frontdoor_interventional_mean(&pin, 1.0) - frontdoor_interventional_mean(&pin, 0.0);
    assert!(
        (truth - pin["frequentist"]["expected_ate"].as_f64().unwrap()).abs() < 1e-12,
        "the hand formula reproduces the fixture's enumerated oracle"
    );
    let gp =
        admg_posterior(n, &[0.8, 0.2], &[adjacency_mask_from_admg(&admg).unwrap(), cyclic_mask(n)]);
    for (mode, inference, tolerance) in modes(&pin) {
        for suite in CHECKED {
            for seed in SEEDS {
                let label = format!("{mode} {suite:?} seed={seed}");
                let result = Study::tabular(data.clone())
                    .graph_posterior(gp.clone())
                    .query(query.clone())
                    .inference(inference.clone())
                    .refute(suite)
                    .bootstrap_replicates(0)
                    .build()
                    .unwrap()
                    .run(&ExecutionContext::for_tests(seed))
                    .unwrap();
                let ate = result.estimate().as_effect().unwrap().ate;
                assert!(
                    (ate - truth).abs() < tolerance,
                    "{label}: ate {ate} vs front-door {truth}"
                );
                let mixture = result.structural_response.as_ref().expect("ADMG posterior mixture");
                assert!(
                    (mixture.unidentified_mass - 0.2).abs() < 1e-12,
                    "{label}: unidentified mass"
                );
                assert!(!result.refutations.is_empty(), "{label}: refuters ran");
            }
        }
    }
}

/// `InterventionResponse x Admg x {graph_posterior, explicit, accepted} x {Frequentist,
/// Bayesian} x {cheap, full}`: the published level equals the front-door interventional
/// mean `E[Y | do(t = 1)]` computed from the counts.
#[test]
fn admg_intervention_response_matches_the_frontdoor_formula_under_cheap_and_full() {
    let pin = pin();
    let data = expand_contingency(&pin);
    let admg = admg_from_pin(&pin);
    let n = admg.node_count();
    let truth = frontdoor_interventional_mean(&pin, 1.0);
    let gp = admg_posterior(n, &[1.0], &[adjacency_mask_from_admg(&admg).unwrap()]);
    for (mode, inference, tolerance) in modes(&pin) {
        for suite in CHECKED {
            for seed in SEEDS {
                let label = format!("graph_posterior {mode} {suite:?} seed={seed}");
                let result = Study::tabular(data.clone())
                    .graph_posterior(gp.clone())
                    .query(CausalQuery::Response(level_query(1.0)))
                    .inference(inference.clone())
                    .refute(suite)
                    .bootstrap_replicates(0)
                    .build()
                    .unwrap()
                    .run(&ExecutionContext::for_tests(seed))
                    .unwrap();
                let got = response_values(&result)[0];
                assert!((got - truth).abs() < tolerance, "{label}: {got} vs front-door {truth}");
                assert!(!result.refutations.is_empty(), "{label}: refuters ran");

                for accepted in [false, true] {
                    let label = format!("accepted={accepted} {mode} {suite:?} seed={seed}");
                    let builder = Study::tabular(data.clone());
                    let builder = if accepted {
                        builder.graph(AcceptedGraph::from(admg.clone()))
                    } else {
                        builder.graph(admg.clone())
                    };
                    let result = builder
                        .query(CausalQuery::Response(level_query(1.0)))
                        .identifier(antecedent::IdentifierId::GeneralId)
                        .estimator(antecedent::EstimatorId::FunctionalEffect)
                        .inference(inference.clone())
                        .refute(suite)
                        .bootstrap_replicates(0)
                        .build()
                        .unwrap()
                        .run(&ExecutionContext::for_tests(seed))
                        .unwrap();
                    let got = response_values(&result)[0];
                    assert!(
                        (got - truth).abs() < tolerance,
                        "{label}: {got} vs front-door {truth}"
                    );
                    assert!(!result.refutations.is_empty(), "{label}: refuters ran");
                }
            }
        }
    }
}
