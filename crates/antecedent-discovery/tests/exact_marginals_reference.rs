//! Exact edge marginals of the Gaussian-BIC DAG posterior against brute-force enumeration.
//!
//! `oracle_data/exact_marginals_reference.py` (frozen in the JSON beside it) enumerates every
//! labelled DAG on four or five nodes, scores each node by a 60-digit least-squares fit and sums
//! `exp(score + log prior)` for the evidence and for each edge. The subset dynamic programme must
//! reproduce the log evidence and all `n^2` marginals, under the uniform prior, a Bernoulli edge
//! prior with a parent cap, and forbidden and required edges, including a near-collinear design
//! and scores of order a thousand nats.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{ExecutionContext, Lag, VariableId};
use antecedent_data::TabularData;
use antecedent_discovery::{
    exact_dag_marginals, DiscoveryConstraints, DiscoveryWorkspace, GraphPrior, LaggedLink,
    OrderMcmc, StructureMcmc,
};
use antecedent_state::GraphScoreFamily;
use serde_json::Value;

mod common;

fn link(vars: &[VariableId], pair: &Value) -> LaggedLink {
    let at = |k: usize| pair[k].as_u64().unwrap() as usize;
    LaggedLink {
        source: vars[at(0)],
        source_lag: Lag::CONTEMPORANEOUS,
        target: vars[at(1)],
        target_lag: Lag::CONTEMPORANEOUS,
    }
}

struct Problem {
    name: String,
    n: usize,
    data: TabularData,
    vars: Vec<VariableId>,
    prior: GraphPrior,
    marginals: Vec<f64>,
    log_evidence: f64,
}

fn problems() -> Vec<Problem> {
    let doc: Value =
        serde_json::from_str(include_str!("oracle_data/exact_marginals_reference.json")).unwrap();
    doc["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|case| {
            let n = case["n_vars"].as_u64().unwrap() as usize;
            let columns: Vec<Vec<f64>> = case["columns"]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| c.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect())
                .collect();
            let names: Vec<String> = (0..n).map(|i| format!("v{i}")).collect();
            let data = TabularData::new(common::continuous_storage(&names, columns));
            let vars: Vec<VariableId> = (0..n).map(|i| VariableId::from_raw(i as u32)).collect();
            let base = match case["p"].as_f64() {
                Some(p) => GraphPrior::bernoulli_edges(p).unwrap(),
                None => GraphPrior::uniform(),
            };
            let links = |key: &str| -> Arc<[LaggedLink]> {
                case[key].as_array().unwrap().iter().map(|pair| link(&vars, pair)).collect()
            };
            let constraints = DiscoveryConstraints {
                forbidden: links("forbidden"),
                required: links("required"),
                max_parents: case["max_parents"].as_u64().map(|m| m as usize),
                ..base.constraints.clone()
            };
            Problem {
                name: case["name"].as_str().unwrap().to_string(),
                n,
                data,
                vars,
                prior: base.with_constraints(constraints),
                marginals: case["marginals"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap())
                    .collect(),
                log_evidence: case["log_evidence"].as_f64().unwrap(),
            }
        })
        .collect()
}

#[test]
fn dynamic_programme_matches_brute_force_enumeration_of_every_dag() {
    for case in problems() {
        let got = exact_dag_marginals(
            &case.data,
            &case.vars,
            &case.prior,
            GraphScoreFamily::GaussianBic,
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
        let name = &case.name;
        assert!(
            (got.log_evidence - case.log_evidence).abs() <= 1e-9 * case.log_evidence.abs().max(1.0),
            "{name}: log evidence {} vs {}",
            got.log_evidence,
            case.log_evidence
        );
        assert_eq!(got.edge_marginals.len(), case.marginals.len());
        for (cell, (g, w)) in got.edge_marginals.iter().zip(&case.marginals).enumerate() {
            assert!(
                (g - w).abs() <= 1e-9,
                "{name}: P({} -> {}) = {g} vs {w}",
                cell / case.n,
                cell % case.n
            );
        }
    }
}

/// The two samplers target the same posterior, the Bernoulli prior, the parent cap and the
/// forbidden and required edges included: every cell within `0.03 + 5 x` its Monte-Carlo
/// standard error of the enumerated marginal.
#[test]
fn order_and_structure_samplers_match_the_enumerated_marginals_under_priors_and_constraints() {
    let structure =
        StructureMcmc::new().with_schedule(4, 1000, 8000, 1).with_diagnostics_gate(false);
    let order = OrderMcmc::new().with_schedule(4, 1000, 8000, 1).with_diagnostics_gate(false);
    for (index, case) in problems().into_iter().enumerate().filter(|(_, c)| c.n == 4) {
        let seed = 100 + index as u64;
        let posteriors = [
            (
                "structure",
                structure
                    .run(
                        &case.data,
                        &case.vars,
                        &case.prior,
                        GraphScoreFamily::GaussianBic,
                        &mut DiscoveryWorkspace::default(),
                        &ExecutionContext::for_tests(seed),
                    )
                    .unwrap(),
            ),
            (
                "order",
                order
                    .run(
                        &case.data,
                        &case.vars,
                        &case.prior,
                        GraphScoreFamily::GaussianBic,
                        &mut DiscoveryWorkspace::default(),
                        &ExecutionContext::for_tests(seed),
                    )
                    .unwrap(),
            ),
        ];
        for (label, post) in posteriors {
            let mcse = post.edge_mcse().expect("a sampled posterior carries edge MCSE");
            for (cell, (&got, &want)) in
                post.edge_marginals().iter().zip(&case.marginals).enumerate()
            {
                let band = 0.03 + 5.0 * mcse[cell];
                assert!(
                    (got - want).abs() <= band,
                    "{} {label}: P({} -> {}) is {got}, enumerated {want}, band {band}",
                    case.name,
                    cell / case.n,
                    cell % case.n
                );
            }
        }
    }
}
