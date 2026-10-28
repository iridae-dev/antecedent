//! The count-fitted finite-table posterior against exact enumeration and the exact Beta
//! posterior (`oracle_data/count_fitted_posterior_reference.py`).
//!
//! * Point values equal the rational sum over the joint assignment space of forty random models
//!   (counted and declared nodes, up to five nodes and three levels, evidence on roots and children).
//! * Each posterior-quantile endpoint of a child's level probability lies between the exact
//!   Beta quantiles at five Monte Carlo standard deviations either side of its probability, for
//!   counts from empty to heavily skewed and prior weights from 0.001 to 20 (the shape-below-one
//!   gamma draws of a small prior underflow without a log-space normalisation).

use antecedent_estimate::count_fitted_posterior::{
    CountFittedModel, CountNode, NodeTable, PosteriorQuery,
};
use serde_json::Value;

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/count_fitted_posterior_reference.json"))
        .expect("frozen reference")
}

fn usizes(value: &Value) -> Vec<usize> {
    value.as_array().expect("array").iter().map(|v| v.as_u64().expect("index") as usize).collect()
}

fn model(nodes: &Value, rows: u64) -> CountFittedModel {
    let nodes = nodes
        .as_array()
        .expect("nodes")
        .iter()
        .map(|node| CountNode {
            cardinality: node["card"].as_u64().expect("card") as usize,
            parents: usizes(&node["parents"]),
            table: if node["kind"] == "declared" {
                NodeTable::Declared(
                    node["table"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect(),
                )
            } else {
                NodeTable::Counts(
                    node["table"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect(),
                )
            },
        })
        .collect();
    CountFittedModel::new(nodes, rows).expect("model")
}

#[test]
fn point_values_equal_the_exact_rational_enumeration() {
    let reference = reference();
    let cases = reference["point"].as_array().expect("cases");
    assert!(cases.len() >= 30);
    for case in cases {
        let model = model(&case["nodes"], 100);
        let evidence = case["evidence"]
            .as_array()
            .unwrap()
            .iter()
            .map(|pair| (pair[0].as_u64().unwrap() as usize, pair[1].as_u64().unwrap() as usize))
            .collect();
        let query = PosteriorQuery {
            target: case["target"].as_u64().unwrap() as usize,
            level: case["level"].as_u64().unwrap() as usize,
            evidence,
        };
        let got = model.point(&query, case["alpha"].as_f64().unwrap()).expect("point");
        let want = case["point"].as_f64().unwrap();
        assert!((got - want).abs() <= 1e-13, "{got} vs {want}: {case}");
    }
}

fn text(value: &Value) -> f64 {
    value.as_str().expect("text").parse().expect("number")
}

#[test]
fn interval_endpoints_sit_inside_the_exact_beta_quantile_brackets() {
    let reference = reference();
    for case in reference["quantile"].as_array().expect("cases") {
        let counts: Vec<u64> =
            case["counts"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
        let k = counts.len();
        let node_x =
            CountNode { cardinality: 2, parents: vec![], table: NodeTable::Counts(vec![1, 1]) };
        // Y has parent X; the row for X = 1 carries the counts, the row for X = 0 is unused.
        let mut table = vec![1; k];
        table.extend(&counts);
        let node_y =
            CountNode { cardinality: k, parents: vec![0], table: NodeTable::Counts(table) };
        let model =
            CountFittedModel::new(vec![node_x, node_y], counts.iter().sum()).expect("model");
        let query = PosteriorQuery {
            target: 1,
            level: case["target_level"].as_u64().unwrap() as usize,
            evidence: vec![(0, 1)],
        };
        let alpha = case["alpha"].as_f64().unwrap();
        let interval = model
            .posterior_interval(
                &query,
                case["level"].as_f64().unwrap(),
                alpha,
                case["draws"].as_u64().unwrap() as u32,
                17,
            )
            .expect("interval");
        assert!((interval.point - case["point"].as_f64().unwrap()).abs() <= 1e-13);
        for (name, got) in [("lower", interval.lower), ("upper", interval.upper)] {
            let (low, high) = (text(&case[name]["min"]), text(&case[name]["max"]));
            assert!(got.is_finite(), "{case}: {name} is not finite");
            assert!(
                got >= low - 1e-12 && got <= high + 1e-12,
                "{case}: {name} {got:e} outside [{low:e}, {high:e}]"
            );
        }
    }
}
