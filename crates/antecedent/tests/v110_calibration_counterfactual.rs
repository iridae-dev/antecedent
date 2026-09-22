//! Repeated-sampling coverage of the counterfactual constructions that
//! heterogeneity-capable mechanism families introduced, at the level the facade
//! publishes.
//!
//! A Bayesian `Counterfactual` publishes two intervals from one Dirichlet
//! mechanism-refit posterior:
//!
//! * the **mean-ITE** credible interval (`posterior_quantile`): the average
//!   two-world contrast over the observed units; and
//! * the **per-unit** credible intervals (`unit_posterior_quantile`,
//!   `counterfactual.unit_effect_intervals`): the equal-tailed quantiles of
//!   each observed unit's contrast draws.
//!
//! With only linear-Gaussian outcome mechanisms both collapse onto one slope.
//! With an interaction or spline family selected they target different
//! things, so each is measured on the data-generating process whose unit
//! effects actually vary, with that process's own per-unit truth:
//!
//! * [`interaction_data`] — `y = 0.8a + 0.5b + 0.6ab + z + e`: a unit's contrast
//!   is `0.8 + 0.6·b`, selected family `LinearInteractions`;
//! * [`exp_modifier_data`] — `y = a·exp(z) + z + e`: a unit's contrast is
//!   `exp(z)`, selected family `LinearSpline`.
//!
//! Per-unit coverage is scored on **designated units** — in each replicate, a
//! fixed rule picks one unit per stratum (the first unit with `b = 0` and the
//! first with `b = 1`; the first two units for the spline design). Scoring
//! every unit of a replicate would record hundreds of intervals that share one
//! posterior, and the tally's Monte Carlo standard error assumes independent
//! Bernoulli draws; one unit per stratum per replicate keeps it honest. The
//! all-unit average coverage is printed beside each record as a diagnostic
//! only, never recorded.
//!
//! Every tally is keyed on the construction the runtime reports for the
//! interval it scores ([`bind`]), never declared here. The 0.90 mean-ITE
//! interval is read from the same draws (`common::reported::posterior_pair`);
//! the per-unit interval is published only at the reported level, so it is
//! recorded only there.
//!
//! Ignored tests run via `scripts/gate_calibration.sh` (release build).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::too_many_lines, clippy::doc_markdown)]
#![allow(
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    reason = "test scaffolding compares exact constants and indexes with small literals"
)]

mod common;

use std::sync::Arc;

use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    CausalQuery, CounterfactualQuery, ExecutionContext, Intervention, Value, VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_graph::{Dag, DenseNodeId};
use common::calibration::{CoverageTally, RecordKey, SampleGrid, gaussian, n_sim, stream_seed};
use common::calibration_bind::bind;
use common::reported::{GATE_LEVEL, REPORTED_LEVEL, gate_at, posterior_pair};
use common::static_dgp::{bernoulli, sigmoid, table, uniform};

// ---------------------------------------------------------------- designs

/// `z ~ N(0,1)`, `a ~ Bern(σ(0.5z))`, `b ~ Bern(0.5)`,
/// `y = 0.8a + 0.5b + 0.6ab + z + e` (columns `z, a, b, y`). The consumer's
/// interaction fixture. A unit's two-world contrast `Y(1) − Y(0)` is
/// `0.8 + 0.6·b` exactly: the disturbance cancels.
fn interaction_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let mut u = uniform(seed);
    let (mut z, mut a, mut b, mut y) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        z[i] = g();
        a[i] = bernoulli(&mut u, sigmoid(0.5 * z[i]));
        b[i] = bernoulli(&mut u, 0.5);
        y[i] = 0.8 * a[i] + 0.5 * b[i] + 0.6 * a[i] * b[i] + z[i] + g();
    }
    table(&[("z", &z), ("a", &a), ("b", &b), ("y", &y)])
}

/// `z ~ N(0, 0.6²)`, `a ~ Bern(σ(0.5z))`, `y = a·exp(z) + z + e` (columns
/// `z, a, y`). A unit's contrast is `exp(z)`: a nonlinear modifier only a
/// spline family can follow. `z` is scaled so the curve's range stays inside
/// the observed covariate support of both arms.
fn exp_modifier_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let mut u = uniform(seed);
    let (mut z, mut a, mut y) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        z[i] = 0.6 * g();
        a[i] = bernoulli(&mut u, sigmoid(0.5 * z[i]));
        y[i] = a[i] * z[i].exp() + z[i] + g();
    }
    table(&[("z", &z), ("a", &a), ("y", &y)])
}

fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

fn dag(n: u32, edges: &[(u32, u32)]) -> Dag {
    let mut g = Dag::with_variables(n);
    for &(s, t) in edges {
        g.insert_directed(DenseNodeId::from_raw(s), DenseNodeId::from_raw(t)).unwrap();
    }
    g
}

fn column(data: &TabularData, name: &str) -> Vec<f64> {
    data.float64_values(data.schema().id_of(name).unwrap()).unwrap().clone()
}

/// `do(treatment = 1)` against `do(treatment = 0)` on `outcome`.
fn query(treatment: u32, outcome: u32) -> CausalQuery {
    CausalQuery::Counterfactual(
        CounterfactualQuery::new(
            v(outcome),
            Arc::from([Intervention::set(v(treatment), Value::f64(1.0))]),
        )
        .with_control_level(0.0),
    )
}

/// The facade's default Bayesian configuration (Python `Bayesian()`).
fn run(
    data: TabularData,
    graph: Dag,
    query: CausalQuery,
    seed: u64,
) -> Option<(Study, StudyResult)> {
    let study = Study::tabular(data)
        .graph(graph)
        .query(query)
        .inference(InferenceMode::Bayesian(BayesianConfig::laplace()))
        .refute(RefuteSuite::None)
        .build()
        .ok()?;
    let result = study.run(&ExecutionContext::for_tests(seed)).ok()?;
    Some((study, result))
}

/// The outcome family the counterfactual path selected, from its audit diagnostic.
fn selected_outcome_family(result: &StudyResult, outcome: u32) -> String {
    let text = result
        .diagnostics
        .iter()
        .find(|d| d.code.as_ref() == "gcm.counterfactual.mechanisms")
        .map(|d| d.message.to_string())
        .unwrap_or_default();
    let marker = format!("variable: VariableId({outcome}),");
    text.split("MechanismAssignment {")
        .find(|chunk| chunk.contains(&marker))
        .and_then(|chunk| chunk.split("selected: ").nth(1))
        .and_then(|rest| rest.split(',').next())
        .unwrap_or("")
        .trim()
        .to_string()
}

// ---------------------------------------------------------------- emission

/// Mean-ITE interval method.
const MEAN_INTERVAL: &str = "posterior_quantile";
/// Per-unit interval method.
const UNIT_INTERVAL: &str = "unit_posterior_quantile";

/// This file's single coverage-record emission point.
fn keyed(
    test: &'static str,
    dgp: &'static str,
    interval: &'static str,
    level: f64,
    label: &str,
) -> CoverageTally {
    CoverageTally::for_record(RecordKey { test, dgp, interval }, level).labelled(label)
}

/// One replicate's per-unit intervals, read from the result and checked
/// against the published unit effects.
fn unit_intervals(result: &StudyResult) -> (Vec<f64>, Vec<(f64, f64)>) {
    let cf = result.counterfactual.as_ref().expect("counterfactual slot");
    let intervals = cf.unit_effect_intervals.as_ref().expect("Bayesian unit_effect_intervals");
    assert_eq!(
        intervals.level, REPORTED_LEVEL,
        "per-unit intervals are published at the reported level"
    );
    assert_eq!(intervals.method, UNIT_INTERVAL);
    let pairs: Vec<(f64, f64)> =
        intervals.lower.iter().copied().zip(intervals.upper.iter().copied()).collect();
    (cf.unit_effects.to_vec(), pairs)
}

/// Running all-unit coverage, printed beside the records as a diagnostic only.
#[derive(Default)]
struct AllUnits {
    covered: u64,
    scored: u64,
}

impl AllUnits {
    fn add(&mut self, intervals: &[(f64, f64)], truths: &[f64]) {
        for ((lo, hi), truth) in intervals.iter().zip(truths) {
            self.scored += 1;
            self.covered += u64::from(*lo <= *truth && *truth <= *hi);
        }
    }

    fn print(&self, test: &str) {
        eprintln!(
            "calibration-diagnostic {test}: all-unit average coverage {:.4} over {} unit intervals \
             (not a record: units within a replicate share one posterior)",
            self.covered as f64 / self.scored.max(1) as f64,
            self.scored
        );
    }
}

/// Measured coverage of each tally that is a named boundary, in tally order;
/// `None` gates the nominal band.
type Measured = [[Option<f64>; 3]; 4];

/// Score one design: per-unit intervals on two designated units and the
/// mean-ITE interval at the reported and gate levels.
#[allow(clippy::too_many_arguments)]
fn design(
    test: &'static str,
    dgp: &'static str,
    labels: [&str; 2],
    data_for: impl Fn(u64) -> TabularData,
    graph: &Dag,
    treatment: u32,
    outcome: u32,
    truth_for: impl Fn(&TabularData) -> Vec<f64>,
    designated: impl Fn(&TabularData) -> [usize; 2],
    family: &[&str],
    seed_tag: u64,
    measured: &Measured,
) {
    let mut tallies = [
        keyed(test, dgp, UNIT_INTERVAL, REPORTED_LEVEL, labels[0]),
        keyed(test, dgp, UNIT_INTERVAL, REPORTED_LEVEL, labels[1]),
        keyed(test, dgp, MEAN_INTERVAL, REPORTED_LEVEL, "mean_ite"),
        keyed(test, dgp, MEAN_INTERVAL, GATE_LEVEL, "mean_ite"),
    ];
    let mut all_units = AllUnits::default();
    let mut other_family = 0u32;
    for rep in 0..u64::from(n_sim()) {
        let seed = stream_seed(seed_tag, rep);
        let data = data_for(seed);
        let truths = truth_for(&data);
        let units = designated(&data);
        let Some((study, result)) = run(data, graph.clone(), query(treatment, outcome), seed)
        else {
            tallies.iter_mut().for_each(CoverageTally::skip);
            continue;
        };
        if rep == 0 {
            assert_eq!(result.logical_plan.estimator.as_deref(), Some("gcm.fit"));
        }
        let selected = selected_outcome_family(&result, outcome);
        other_family += u32::from(!family.contains(&selected.as_str()));
        let (effects, intervals) = unit_intervals(&result);
        assert_eq!(effects.len(), truths.len());
        all_units.add(&intervals, &truths);
        let mean_truth = truths.iter().sum::<f64>() / truths.len() as f64;
        let column = result
            .posterior
            .as_ref()
            .and_then(antecedent::CausalPosterior::effect_column)
            .expect("mean-ITE posterior column");
        let [reported, at_gate] = posterior_pair(&result, column);
        for tally in &mut tallies {
            bind(tally, &study, &result);
        }
        tallies[0].record(Some(intervals[units[0]]), truths[units[0]]);
        tallies[1].record(Some(intervals[units[1]]), truths[units[1]]);
        tallies[2].record(reported, mean_truth);
        tallies[3].record(at_gate, mean_truth);
    }
    all_units.print(test);
    eprintln!(
        "calibration-diagnostic {test}: outcome family outside {family:?} in {other_family} of {} replicates",
        n_sim()
    );
    gate_at(&tallies, measured);
}

// ---------------------------------------------------------------- tests

/// Interaction design at the consumer's `n = 2500`. The first `b = 0` and the
/// first `b = 1` unit of each replicate are the designated units.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn counterfactual_interaction_bayesian_unit_and_mean_ite_coverage() {
    let graph = dag(4, &[(0, 1), (0, 3), (1, 3), (2, 3)]);
    design(
        "counterfactual_interaction_bayesian_unit_and_mean_ite_coverage",
        "interaction_data",
        ["unit_b0", "unit_b1"],
        |seed| interaction_data(SampleGrid::HEAVY.n(2500), seed),
        &graph,
        1,
        3,
        |data| column(data, "b").iter().map(|b| 0.8 + 0.6 * b).collect(),
        |data| {
            let b = column(data, "b");
            [
                b.iter().position(|v| *v == 0.0).expect("a b = 0 unit"),
                b.iter().position(|v| *v == 1.0).expect("a b = 1 unit"),
            ]
        },
        &["LinearInteractions", "LinearSpline"],
        0x110_0CF1,
        &INTERACTION_MEASURED,
    );
}

/// Boundary readings of [`counterfactual_interaction_bayesian_unit_and_mean_ite_coverage`],
/// in tally order (`unit_b0`, `unit_b1`, mean ITE at 0.95, mean ITE at 0.90);
/// `None` gates the nominal band.
const INTERACTION_MEASURED: Measured =
    [[Some(0.868), None, None], [Some(0.882), None, None], [None, None, None], [None, None, None]];

/// Spline design at `n = 2500`. Units 0 and 1 of each replicate are the
/// designated units (each an independent draw of `z`).
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn counterfactual_exp_modifier_bayesian_unit_and_mean_ite_coverage() {
    let graph = dag(3, &[(0, 1), (0, 2), (1, 2)]);
    design(
        "counterfactual_exp_modifier_bayesian_unit_and_mean_ite_coverage",
        "exp_modifier_data",
        ["unit_0", "unit_1"],
        |seed| exp_modifier_data(SampleGrid::HEAVY.n(2500), seed),
        &graph,
        1,
        2,
        |data| column(data, "z").iter().map(|z| z.exp()).collect(),
        |_| [0, 1],
        &["LinearSpline"],
        0x110_0CF2,
        &EXP_MODIFIER_MEASURED,
    );
}

/// Boundary readings of [`counterfactual_exp_modifier_bayesian_unit_and_mean_ite_coverage`],
/// in tally order (`unit_0`, `unit_1`, mean ITE at 0.95, mean ITE at 0.90).
const EXP_MODIFIER_MEASURED: Measured = [
    [Some(0.933), None, Some(0.938)],
    [Some(0.918), None, None],
    [None, None, Some(0.936)],
    [None, None, Some(0.884)],
];
