//! The exact PAG-class posterior of Gaussian maximal ancestral graphs against brute force.
//!
//! `oracle_data/exact_pag_posterior_reference.py` (frozen in the JSON beside it) lists every
//! assignment of {absent, i->j, j->i, i<->j} to the pairs of three or four variables, keeps the
//! ancestral and maximal ones by definition (m-separation by enumerating simple paths), groups
//! them by their set of separation statements, marks each class's endpoints (a tail or arrowhead
//! only where every member agrees, otherwise a circle), fits every graph by minimising the
//! Gaussian likelihood directly with L-BFGS from several starts, and weights the classes by
//! exp(BIC) under a uniform prior over graphs. The product (enumeration, residual iterative
//! conditional fitting, BIC, class merging and PAG marks) must reproduce the number of graphs, the
//! set of classes with their PAGs and their weights, including on eight rows, on a nearly
//! collinear pair and on four variables with a latent confounder and a collider.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_discovery::exact_pag_posterior;
use serde_json::Value;

/// `a-b:mark_at_a,mark_at_b` for every adjacent pair, joined with `;`.
fn key(edges: &[(usize, usize, &str, &str)]) -> String {
    let mut parts: Vec<String> =
        edges.iter().map(|(a, b, ma, mb)| format!("{a}-{b}:{ma},{mb}")).collect();
    parts.sort();
    parts.join(";")
}

#[test]
fn classes_pags_and_weights_equal_the_brute_force_enumeration() {
    let cases: Vec<Value> =
        serde_json::from_str(include_str!("oracle_data/exact_pag_posterior_reference.json"))
            .unwrap();
    assert_eq!(cases.len(), 4);
    for case in &cases {
        let name = case["name"].as_str().unwrap();
        let columns: Vec<Vec<f64>> = case["columns"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect())
            .collect();
        let p = columns.len();
        let names: Vec<String> = (0..p).map(|i| format!("v{i}")).collect();
        let data = TabularData::from_f64_columns(
            names.iter().map(String::as_str).zip(columns.iter().map(Vec::as_slice)),
        )
        .unwrap();
        let variables: Vec<VariableId> = (0..p as u32).map(VariableId::from_raw).collect();
        let post = exact_pag_posterior(&data, &variables, &ExecutionContext::for_tests(1))
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        let note =
            post.diagnostics().notes.iter().find(|n| n.starts_with("exact_pag_posterior")).unwrap();
        assert!(
            note.contains(&format!("mags={} ", case["mags"].as_u64().unwrap())),
            "{name}: the number of maximal ancestral graphs: {note}"
        );
        assert!(note.contains("unscored=0"), "{name}: {note}");

        let mut got: BTreeMap<String, f64> = BTreeMap::new();
        let marks = post.mark_atoms().expect("a PAG posterior carries its circle marks");
        for ((atom, circles), weight) in
            post.atoms().iter().zip(marks.iter()).zip(post.weights().iter())
        {
            let mut edges = Vec::new();
            for i in 0..p {
                for j in (i + 1)..p {
                    let (fwd, back) = (atom.has_edge(p, i, j), atom.has_edge(p, j, i));
                    let (c_ij, c_ji) = (circles.has_edge(p, i, j), circles.has_edge(p, j, i));
                    let pair = match (fwd, back) {
                        (false, false) => continue,
                        (true, false) => (if c_ij { "circle" } else { "tail" }, "arrow"),
                        (false, true) => ("arrow", if c_ji { "circle" } else { "tail" }),
                        (true, true) if c_ij && c_ji => ("circle", "circle"),
                        (true, true) => ("arrow", "arrow"),
                    };
                    edges.push((i, j, pair.0, pair.1));
                }
            }
            *got.entry(key(&edges)).or_insert(0.0) += *weight;
        }

        let mut want: BTreeMap<String, f64> = BTreeMap::new();
        for class in case["classes"].as_array().unwrap() {
            let edges: Vec<(usize, usize, &str, &str)> = class["edges"]
                .as_array()
                .unwrap()
                .iter()
                .map(|e| {
                    (
                        e[0].as_u64().unwrap() as usize,
                        e[1].as_u64().unwrap() as usize,
                        e[2].as_str().unwrap(),
                        e[3].as_str().unwrap(),
                    )
                })
                .collect();
            *want.entry(key(&edges)).or_insert(0.0) += class["weight"].as_f64().unwrap();
        }
        // Distinct classes can share a PAG only if the product merged what the reference kept
        // apart: the number of classes is compared first.
        assert_eq!(post.n_graphs(), case["classes"].as_array().unwrap().len(), "{name}: classes");
        assert_eq!(
            got.keys().collect::<Vec<_>>(),
            want.keys().collect::<Vec<_>>(),
            "{name}: the PAGs of the classes"
        );
        let worst = got.iter().map(|(k, g)| (g - want[k]).abs()).fold(0.0_f64, f64::max);
        // The residual iterative conditional fitting stops at a relative likelihood change of
        // 1e-10; on a pair correlated to 1 - 1e-8 its fits are looser, by about 1e-4 in weight.
        let tolerance = if name.contains("near_collinear") { 2e-4 } else { 1e-5 };
        assert!(
            worst <= tolerance,
            "{name}: largest class-weight difference {worst}\n got {got:?}\nwant {want:?}"
        );
    }
}
