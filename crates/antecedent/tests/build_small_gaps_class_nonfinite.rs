// SPDX-License-Identifier: MIT OR Apache-2.0
//! A non-finite treatment value on a graph-class route is refused for what it is, a non-finite
//! value, not as a rank-deficient design.
use antecedent::{RefuteSuite, Study};
use antecedent_core::{AverageEffectQuery, ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_graph::{Admg, Cpdag, Dag, DenseNodeId};

const N: usize = 120;

fn columns(bad: f64) -> Vec<(&'static str, Vec<f64>)> {
    let at = |i: usize| i as f64;
    let z: Vec<f64> = (0..N).map(|i| (at(i) * 0.37).sin()).collect();
    let mut a: Vec<f64> = (0..N).map(|i| z[i] + (at(i) * 0.71).cos()).collect();
    let y: Vec<f64> = (0..N).map(|i| 3.0 * a[i] + z[i] + 0.1 * (at(i) * 0.31).sin()).collect();
    a[7] = bad;
    vec![("a", a), ("y", y), ("z", z)]
}

fn run(class: &str, bad: f64) -> Result<f64, String> {
    let data =
        TabularData::from_f64_columns(columns(bad).iter().map(|(name, c)| (*name, c.as_slice())))
            .unwrap();
    let edges = [(2_u32, 0_u32), (2, 1), (0, 1)];
    let builder = Study::tabular(data);
    let builder = if class == "dag" {
        let mut g = Dag::with_variables(3);
        for (f, t) in edges {
            g.insert_directed(DenseNodeId::from_raw(f), DenseNodeId::from_raw(t)).unwrap();
        }
        builder.graph(g)
    } else if class == "cpdag" {
        let mut g = Cpdag::with_variables(3);
        for (f, t) in edges {
            g.insert_directed(DenseNodeId::from_raw(f), DenseNodeId::from_raw(t)).unwrap();
        }
        builder.graph(g)
    } else {
        let mut g = Admg::with_variables(3);
        for (f, t) in edges {
            g.insert_directed(DenseNodeId::from_raw(f), DenseNodeId::from_raw(t)).unwrap();
        }
        builder.graph(g)
    };
    builder
        .query(AverageEffectQuery::with_levels(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            0.0,
            1.0,
        ))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .map_err(|e| e.to_string())
        .and_then(|s| s.run(&ExecutionContext::for_tests(5)).map_err(|e| e.to_string()))
        .map(|r| r.estimate().as_effect().unwrap().ate)
}

#[test]
fn build_small_gaps_an_infinite_treatment_value_on_a_class_route_is_a_non_finite_refusal() {
    for class in ["dag", "admg", "cpdag"] {
        assert!((run(class, 0.3).unwrap() - 3.0).abs() < 0.1, "{class}");
        for bad in [f64::INFINITY, f64::NEG_INFINITY] {
            let error = run(class, bad).expect_err("an infinite treatment value must refuse");
            assert!(!error.contains("rank deficient"), "{class}: {error}");
            assert!(error.contains("non-finite"), "{class}: {error}");
        }
    }
}
