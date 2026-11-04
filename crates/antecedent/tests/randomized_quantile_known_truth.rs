//! Known-truth evidence for the explicitly selected `randomized.quantile` estimator on a
//! supplied or accepted DAG: `AverageEffect x Dag x {explicit, accepted} x Frequentist x none`
//! with a quantile outcome functional.
//!
//! The design has forty rows: the control arm carries the outcomes 1, 2, ..., 20 and the treated
//! arm 3, 4, ..., 22 under the graph `t -> y` (no adjustment). The treated law is the control
//! law shifted by two, so the median effect is exactly 2. On the pooled-outcome grid (the
//! integers 3, ..., 20) the arm distribution functions are `F0(c) = c / 20` and
//! `F1(c) = (c - 2) / 20`, the piecewise-linear inversion at 1/2 gives the quantiles 10 and 12,
//! and every row influence of the difference is +-20, so the standard error is
//! `sqrt(40 * 400 / (39 * 40)) = 20 / sqrt(39)` and the interval is the estimate plus and minus
//! 1.959963984540054 times that. The full derivation, including the declared-grid case, is in
//! `crates/antecedent-estimate/tests/randomized_quantile_reference.rs`. The expected values here
//! are written out from that table, never read from the library, and compared to 1e-9.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{
    AcceptedGraph, CausalError, EstimatorId, RefuteSuite, StructureSource, Study, StudyResult,
};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ExecutionContext, OutcomeFunctional, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};

const NORMAL_975: f64 = 1.959_963_984_540_054;

fn vid(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn columns() -> (Vec<f64>, Vec<f64>) {
    let mut outcome: Vec<f64> = (1..=20).map(f64::from).collect();
    outcome.extend((3..=22).map(f64::from));
    let mut treatment = vec![0.0; 20];
    treatment.extend(vec![1.0; 20]);
    (treatment, outcome)
}

fn quantile_query(tau: f64) -> AverageEffectQuery {
    AverageEffectQuery::binary_ate(vid(0), vid(1))
        .with_outcome_functional(OutcomeFunctional::quantile(tau))
}

fn unconfounded() -> (TabularData, Dag) {
    let (treatment, outcome) = columns();
    let data =
        TabularData::from_f64_columns([("t", treatment.as_slice()), ("y", outcome.as_slice())])
            .unwrap();
    let mut graph = Dag::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    (data, graph)
}

fn verify_coordinate(coordinate: &str, accepted: bool) {
    let ctx = ExecutionContext::for_tests(2_204);
    let (data, graph) = unconfounded();
    let query = quantile_query(0.5);
    let base = Study::tabular(data.clone())
        .query(query.clone())
        .estimator(EstimatorId::RandomizedQuantile)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0);
    let builder = if accepted { base.graph(AcceptedGraph::from(graph)) } else { base.graph(graph) };
    let study = builder.clone().build().unwrap_or_else(|error| panic!("{coordinate}: {error}"));
    let prepared = study.prepare(&ctx).unwrap_or_else(|error| panic!("{coordinate}: {error}"));
    drop(builder);
    drop(study);

    assert_eq!(prepared.query(), &CausalQuery::AverageEffect(query));
    assert_eq!(
        prepared.structure_source(),
        if accepted { StructureSource::Accepted } else { StructureSource::Explicit },
        "{coordinate} structure source"
    );
    assert_eq!(prepared.support_status(), Some(antecedent::CellStatus::Supported), "{coordinate}");
    assert_eq!(
        prepared.plan().logical.record.estimator.as_deref(),
        Some("randomized.quantile"),
        "{coordinate} plan estimator"
    );

    let result = prepared.estimate(&data, &ctx).unwrap_or_else(|e| panic!("{coordinate}: {e}"));
    let effect = result.estimate().as_effect().unwrap();
    assert!((effect.ate - 2.0).abs() < 1e-9, "{coordinate}: estimate {} vs 2", effect.ate);
    let se = 20.0 / 39.0_f64.sqrt();
    assert!(
        (effect.uncertainty.analytic.se - se).abs() < 1e-9,
        "{coordinate}: se {} vs {se}",
        effect.uncertainty.analytic.se
    );
    // The reported 95% interval is the normal interval around the estimate.
    let (lower, upper) = (effect.ate - NORMAL_975 * se, effect.ate + NORMAL_975 * se);
    let radius = NORMAL_975 * effect.uncertainty.analytic.se;
    assert!(((effect.ate - radius) - lower).abs() < 1e-9, "{coordinate}: lower end");
    assert!(((effect.ate + radius) - upper).abs() < 1e-9, "{coordinate}: upper end");
    assert!(result.posterior.is_none(), "{coordinate} unexpected posterior");
    assert!(result.refutations.is_empty(), "{coordinate} refutations");
}

#[test]
fn dag_explicit_frequentist_none() {
    verify_coordinate("AverageEffect:Dag:explicit:Frequentist:none", false);
}

#[test]
fn dag_accepted_frequentist_none() {
    verify_coordinate("AverageEffect:Dag:accepted:Frequentist:none", true);
}

fn run(
    data: TabularData,
    graph: Dag,
    query: AverageEffectQuery,
) -> Result<StudyResult, CausalError> {
    Study::tabular(data)
        .query(query)
        .estimator(EstimatorId::RandomizedQuantile)
        .graph(graph)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .and_then(|study| study.run(&ExecutionContext::for_tests(2_205)))
}

#[test]
fn a_confounded_treatment_a_mean_functional_and_a_continuous_treatment_are_refused() {
    // z -> t and z -> y: the adjustment set is {z}, so the empirical arm laws are not the
    // potential-outcome laws.
    let (treatment, outcome) = columns();
    let z: Vec<f64> = (0..40).map(|row| f64::from(row % 2)).collect();
    let data = TabularData::from_f64_columns([
        ("t", treatment.as_slice()),
        ("y", outcome.as_slice()),
        ("z", z.as_slice()),
    ])
    .unwrap();
    let mut confounded = Dag::with_variables(3);
    for (from, to) in [(2_u32, 0_u32), (2, 1), (0, 1)] {
        confounded.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    let error = run(data, confounded, quantile_query(0.5)).unwrap_err();
    assert!(error.to_string().contains("adjustment set"), "{error}");

    // The estimator answers the quantile functional only.
    let (data, graph) = unconfounded();
    let mean = AverageEffectQuery::binary_ate(vid(0), vid(1));
    assert!(run(data, graph, mean).is_err());

    // Treatment coded outside 0/1.
    let (_, outcome) = columns();
    let coded: Vec<f64> = (0..40).map(|row| if row < 20 { 0.0 } else { 2.0 }).collect();
    let data = TabularData::from_f64_columns([("t", coded.as_slice()), ("y", outcome.as_slice())])
        .unwrap();
    let mut graph = Dag::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    assert!(run(data, graph, quantile_query(0.5)).is_err());
}

#[test]
fn a_level_the_grid_cannot_bracket_is_refused_and_a_small_arm_is_refused() {
    // Level 0.01 lies below the arm distribution functions at the first grid point (3/20 for
    // the control arm), so the grid cannot bracket it.
    let (data, graph) = unconfounded();
    assert!(run(data, graph, quantile_query(0.01)).is_err());

    let treatment: Vec<f64> = (0..40).map(|row| if row < 5 { 1.0 } else { 0.0 }).collect();
    let (_, outcome) = columns();
    let data =
        TabularData::from_f64_columns([("t", treatment.as_slice()), ("y", outcome.as_slice())])
            .unwrap();
    let mut graph = Dag::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    assert!(run(data, graph, quantile_query(0.5)).is_err());
}
