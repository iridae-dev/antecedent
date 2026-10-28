//! The adjustment estimators on an equivalence-class graph and on a latent-variable graph.
//!
//! One saturated contingency table over a confounder `z`, a binary treatment `t` and a binary
//! outcome `y` (1000 rows) has closed forms that follow from its cell counts alone:
//!
//! * on an ADMG whose only edges are `z -> t`, `z -> y` and `t -> y` the effect is identified by
//!   adjusting for `z`, and the stratified difference in means is exactly 0.40;
//! * on the CPDAG that leaves `z - t` undirected there are two completions, `z -> t` (adjust for
//!   `z`, effect 0.40) and `t -> z` (`z` is a mediator, so the crude contrast 0.52 is the
//!   effect), and the equal-weight mixture is exactly 0.46.
//!
//! Every estimator that can run on the route is checked against those numbers: the estimators
//! that are exact on a saturated binary stratum (linear and logit adjustment, inverse
//! probability weighting, stratification) to rounding, and the cross-fitted AIPW to its sampling
//! tolerance. Matching on a propensity that takes two values is a tie-breaking procedure whose
//! point estimate is not a function of the cell counts, so it is not asserted here.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{EstimatorId, RefuteSuite, Study};
use antecedent_core::{AverageEffectQuery, ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_graph::{Admg, Cpdag, DenseNodeId};

/// `(z, t, y, count)` of the saturated table.
const CELLS: [(f64, f64, f64, usize); 8] = [
    (0.0, 0.0, 0.0, 360),
    (0.0, 0.0, 1.0, 40),
    (0.0, 1.0, 0.0, 50),
    (0.0, 1.0, 1.0, 50),
    (1.0, 0.0, 0.0, 70),
    (1.0, 0.0, 1.0, 30),
    (1.0, 1.0, 0.0, 120),
    (1.0, 1.0, 1.0, 280),
];

fn table() -> TabularData {
    let (mut z, mut t, mut y) = (vec![], vec![], vec![]);
    for (zz, tt, yy, count) in CELLS {
        for _ in 0..count {
            z.push(zz);
            t.push(tt);
            y.push(yy);
        }
    }
    TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())])
        .unwrap()
}

fn count(z: Option<f64>, t: f64, y: Option<f64>) -> f64 {
    CELLS
        .iter()
        .filter(|c| z.is_none_or(|z| c.0 == z) && c.1 == t && y.is_none_or(|y| c.2 == y))
        .map(|c| c.3 as f64)
        .sum()
}

/// The backdoor-adjusted contrast over `z`, from the cell counts.
fn adjusted_effect() -> f64 {
    let total: f64 = CELLS.iter().map(|c| c.3 as f64).sum();
    [0.0, 1.0]
        .iter()
        .map(|&z| {
            let mean = |t: f64| count(Some(z), t, Some(1.0)) / count(Some(z), t, None);
            (count(Some(z), 0.0, None) + count(Some(z), 1.0, None)) / total * (mean(1.0) - mean(0.0))
        })
        .sum()
}

/// The crude contrast, the effect when `z` is a mediator.
fn crude_effect() -> f64 {
    let mean = |t: f64| count(None, t, Some(1.0)) / count(None, t, None);
    mean(1.0) - mean(0.0)
}

fn node(raw: u32) -> DenseNodeId {
    DenseNodeId::from_raw(raw)
}

const T: u32 = 0;
const Y: u32 = 1;
const Z: u32 = 2;

fn admg() -> Admg {
    let mut graph = Admg::with_variables(3);
    for (from, to) in [(Z, Y), (T, Y), (Z, T)] {
        graph.insert_directed(node(from), node(to)).unwrap();
    }
    graph
}

fn cpdag() -> Cpdag {
    let mut graph = Cpdag::with_variables(3);
    for (from, to) in [(Z, Y), (T, Y)] {
        graph.insert_directed(node(from), node(to)).unwrap();
    }
    graph.insert_undirected(node(Z), node(T)).unwrap();
    graph
}

fn effect(estimator: EstimatorId, class: &str) -> f64 {
    let query = AverageEffectQuery::with_levels(
        VariableId::from_raw(T),
        VariableId::from_raw(Y),
        0.0,
        1.0,
    );
    let builder = Study::tabular(table());
    let builder = if class == "cpdag" { builder.graph(cpdag()) } else { builder.graph(admg()) };
    let result = builder
        .query(query)
        .estimator(estimator)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .and_then(|study| study.run(&ExecutionContext::for_tests(5)))
        .unwrap_or_else(|error| panic!("{estimator:?} on the {class}: {error}"));
    result.estimate().as_effect().unwrap().ate
}

#[test]
fn the_closed_forms_follow_from_the_cell_counts() {
    assert!((adjusted_effect() - 0.40).abs() < 1e-12, "{}", adjusted_effect());
    assert!((crude_effect() - 0.52).abs() < 1e-12, "{}", crude_effect());
}

#[test]
fn adjustment_estimators_recover_the_closed_form_on_the_latent_variable_and_equivalence_class_graphs()
{
    let completions = 0.5 * (adjusted_effect() + crude_effect());
    // (estimator, exact on a saturated binary stratum)
    let table = [
        (EstimatorId::LinearAdjustmentAte, true),
        (EstimatorId::GlmAdjustment, true),
        (EstimatorId::PropensityWeighting, true),
        (EstimatorId::PropensityStratification, true),
        (EstimatorId::Aipw, false),
    ];
    for (estimator, exact) in table {
        let tolerance = if exact { 1e-9 } else { 0.02 };
        let on_admg = effect(estimator, "admg");
        assert!(
            (on_admg - adjusted_effect()).abs() < tolerance,
            "{estimator:?} on the ADMG: {on_admg} vs {}",
            adjusted_effect()
        );
        let on_cpdag = effect(estimator, "cpdag");
        assert!(
            (on_cpdag - completions).abs() < tolerance,
            "{estimator:?} on the CPDAG: {on_cpdag} vs {completions}"
        );
    }
}
