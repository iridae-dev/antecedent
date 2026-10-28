//! CI-screened posterior against an exhaustive restricted-DAG enumeration.
//!
//! `oracle_data/ci_screened_posterior_reference.py` (frozen in the JSON beside it) screens the
//! pairs of four variables by Student-t partial-correlation tests over every conditioning set of
//! at most three other variables (asserting no decision is marginal), enumerates every DAG whose
//! directed edges lie on the surviving pairs, avoid the forbidden directed edges, contain the
//! required ones and respect the smaller of the two parent caps, and weights each by its 60-digit
//! Gaussian-BIC score and prior. The engine and the caller declare constraints separately; the
//! reference merges them as unions with the smaller cap. The product must retain exactly the
//! reference's pairs (with and without the default Benjamini-Hochberg adjustment), reproduce the
//! enumerated edge marginals within Monte-Carlo error, and floor a screened-out pair at exactly
//! `min(prior, alpha) / 2` on both cells while a forbidden pair keeps an exact zero.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{ExecutionContext, Lag, VariableId};
use antecedent_discovery::{
    CiScreenedPosterior, DiscoveryConstraints, DiscoveryWorkspace, GraphPrior, LaggedLink,
    StructureMcmc,
};
use antecedent_state::GraphScoreFamily;
use serde_json::Value;

mod common;

const N: usize = 4;

fn link(vars: &[VariableId], pair: &Value) -> LaggedLink {
    let at = |k: usize| pair[k].as_u64().unwrap() as usize;
    LaggedLink {
        source: vars[at(0)],
        source_lag: Lag::CONTEMPORANEOUS,
        target: vars[at(1)],
        target_lag: Lag::CONTEMPORANEOUS,
    }
}

fn constraints(
    side: &Value,
    vars: &[VariableId],
    base: DiscoveryConstraints,
) -> DiscoveryConstraints {
    let links = |key: &str| -> Arc<[LaggedLink]> {
        side.get(key)
            .and_then(Value::as_array)
            .map(|a| a.iter().map(|p| link(vars, p)).collect())
            .unwrap_or_else(|| Arc::from([]))
    };
    DiscoveryConstraints {
        forbidden: links("forbidden"),
        required: links("required"),
        max_parents: side.get("max_parents").and_then(Value::as_u64).map(|m| m as usize),
        ..base
    }
}

fn run(case: &Value, fdr: bool) -> antecedent_discovery::GraphPosterior {
    let names: Vec<String> = (0..N).map(|i| format!("v{i}")).collect();
    let columns: Vec<Vec<f64>> = case["columns"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect())
        .collect();
    let data = antecedent_data::TabularData::new(common::continuous_storage(&names, columns));
    let vars: Vec<VariableId> = (0..N).map(|i| VariableId::from_raw(i as u32)).collect();
    let alpha = case["alpha"].as_f64().unwrap();
    let engine_base = DiscoveryConstraints { alpha, max_cond_size: 3, ..Default::default() };
    let mut algorithm = CiScreenedPosterior::new()
        .with_constraints(constraints(&case["engine"], &vars, engine_base))
        .with_mcmc(
            StructureMcmc::new().with_schedule(4, 1000, 8000, 1).with_diagnostics_gate(false),
        );
    if !fdr {
        algorithm.fdr = None;
    }
    let base = match case["p"].as_f64() {
        Some(p) => GraphPrior::bernoulli_edges(p).unwrap(),
        None => GraphPrior::uniform(),
    };
    let caller = constraints(&case["caller"], &vars, base.constraints.clone());
    let prior = base.with_constraints(caller);
    algorithm
        .run(
            &data,
            &vars,
            &prior,
            GraphScoreFamily::GaussianBic,
            &mut DiscoveryWorkspace::default(),
            &ExecutionContext::for_tests(71),
        )
        .unwrap()
}

fn note_value<'a>(notes: &'a [Arc<str>], prefix: &str) -> Option<&'a str> {
    notes.iter().find_map(|n| n.strip_prefix(prefix))
}

#[test]
fn retained_pairs_marginals_and_floor_equal_the_restricted_enumeration() {
    let doc: Value =
        serde_json::from_str(include_str!("oracle_data/ci_screened_posterior_reference.json"))
            .unwrap();
    let cases = doc["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 11);
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let post = run(case, false);
        let notes = &post.diagnostics().notes;

        let pairs: Vec<(usize, usize)> = case["pairs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| (p[0].as_u64().unwrap() as usize, p[1].as_u64().unwrap() as usize))
            .collect();
        let kept = note_value(notes, "ci_screened_pairs=").expect("pair-count note");
        assert_eq!(kept.parse::<usize>().unwrap(), pairs.len(), "{name}: retained pair count");
        let out: Vec<String> = (0..N)
            .flat_map(|i| ((i + 1)..N).map(move |j| (i, j)))
            .filter(|p| !pairs.contains(p))
            .map(|(i, j)| format!("v{i}-v{j}"))
            .collect();
        match note_value(notes, "ci_screened_out pairs=[") {
            Some(rest) => {
                let listed = rest.split(']').next().unwrap();
                assert_eq!(listed, out.join(", "), "{name}: screened-out pairs");
            }
            None => assert!(out.is_empty(), "{name}: missing screened-out note for {out:?}"),
        }

        let want: Vec<f64> =
            case["marginals"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        let got = post.edge_marginals();
        assert_eq!(got.len(), N * N);
        let screened_out = |cell: usize| {
            let (i, j) = (cell / N, cell % N);
            i != j && !pairs.contains(&(i.min(j), i.max(j)))
        };
        for (cell, (g, w)) in got.iter().zip(&want).enumerate() {
            if screened_out(cell) {
                // Never proposed: exactly zero, or exactly the floor when a prior is stated.
                assert!((g - w).abs() <= 1e-12, "{name}: screened-out cell {cell}: {g} vs {w}");
            } else {
                assert!(
                    (g - w).abs() <= 0.03,
                    "{name}: P({} -> {}) = {g} vs enumerated {w}",
                    cell / N,
                    cell % N
                );
            }
        }
        assert_eq!(
            post.orientation_marginals().as_ref(),
            got.as_ref(),
            "{name}: orientation marginals are the floored edge marginals"
        );
    }
}

#[test]
fn the_default_benjamini_hochberg_screen_keeps_the_same_pairs_on_decisive_data() {
    let doc: Value =
        serde_json::from_str(include_str!("oracle_data/ci_screened_posterior_reference.json"))
            .unwrap();
    for case in doc["cases"].as_array().unwrap().iter().take(2) {
        let post = run(case, true);
        let kept = note_value(&post.diagnostics().notes, "ci_screened_pairs=").unwrap();
        assert_eq!(
            kept.parse::<usize>().unwrap(),
            case["pairs"].as_array().unwrap().len(),
            "{}",
            case["name"]
        );
    }
}
