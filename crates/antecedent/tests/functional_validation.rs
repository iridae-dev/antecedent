//! Cheap and full validation of the functional queries: attribution, counterfactuals,
//! path-specific effects and interventional distributions refute the whole target vector of
//! their own result by refitting their own compiled program on perturbed data.
//!
//! Every number is checked against a reference the route never sees: the structural equations
//! that generated the table (path-specific effect, interventional law, counterfactual slope),
//! the unvalidated run of the same study (the validation never moves the answer) and the
//! Shapley efficiency identity (contributions add to the score).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "test fixtures: small exact indices, levels and enumerated probabilities"
)]

use std::sync::Arc;

use antecedent::{AcceptedGraph, CellStatus, RefuteSuite, Study};
use antecedent_core::{
    AnomalyAttributionQuery, CausalQuery, CounterfactualQuery, ExecutionContext, Intervention,
    InterventionalDistributionQuery, PathSpecificEffectQuery, Value, VariableId,
};
use antecedent_core::{Lag, NestedCounterfactualQuery, TemporalEffectQuery, TemporalPolicy};
use antecedent_data::{TableView, TabularData, TimeSeriesData};
use antecedent_graph::{Admg, Cpdag, Dag, DenseNodeId, TemporalCpdag, TemporalDag, ensure_lagged};

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn d(raw: u32) -> DenseNodeId {
    DenseNodeId::from_raw(raw)
}

fn bernoulli(p: f64, level: usize) -> f64 {
    if level == 1 { p } else { 1.0 - p }
}

/// Columns from the cells of a joint over four binary variables, `rows` rows in all.
fn table_from_joint(rows: f64, names: [&str; 4], joint: impl Fn([usize; 4]) -> f64) -> TabularData {
    let mut columns: [Vec<f64>; 4] = Default::default();
    for cell in 0..16_usize {
        let levels = [(cell >> 3) & 1, (cell >> 2) & 1, (cell >> 1) & 1, cell & 1];
        let count = (rows * joint(levels)).round() as usize;
        for (column, level) in columns.iter_mut().zip(levels) {
            column.extend(std::iter::repeat_n(level as f64, count));
        }
    }
    TabularData::from_f64_columns([
        (names[0], columns[0].as_slice()),
        (names[1], columns[1].as_slice()),
        (names[2], columns[2].as_slice()),
        (names[3], columns[3].as_slice()),
    ])
    .unwrap()
}

/// `T -> A`, `T -> B`, `A -> Y`, `B -> Y` with no latent cause: columns `t, a, b, y`.
mod routed {
    use super::bernoulli;

    pub fn p_t(t: usize) -> f64 {
        bernoulli(0.5, t)
    }
    pub fn p_a(a: usize, t: usize) -> f64 {
        bernoulli(0.15 + 0.4 * t as f64, a)
    }
    pub fn p_b(b: usize, t: usize) -> f64 {
        bernoulli(0.2 + 0.5 * t as f64, b)
    }
    pub fn p_y(y: usize, a: usize, b: usize) -> f64 {
        bernoulli(0.05 + 0.3 * a as f64 + 0.2 * b as f64 + 0.25 * (a * b) as f64, y)
    }

    /// Observational law of `(t, a, b, y)`.
    pub fn joint(levels: [usize; 4]) -> f64 {
        let [t, a, b, y] = levels;
        p_t(t) * p_a(a, t) * p_b(b, t) * p_y(y, a, b)
    }

    /// `E[Y(t = 1 into A, t = 0 into B)] - E[Y(t = 0)]`, from the equations.
    pub fn effect_through_a() -> f64 {
        let mut total = 0.0;
        for a in 0..2 {
            for b in 0..2 {
                total += p_b(b, 0) * (p_a(a, 1) - p_a(a, 0)) * p_y(1, a, b);
            }
        }
        total
    }

    /// `P(Y = 1 | do(T = 1))`, from the equations.
    pub fn y_under_treatment() -> f64 {
        let mut total = 0.0;
        for a in 0..2 {
            for b in 0..2 {
                total += p_a(a, 1) * p_b(b, 1) * p_y(1, a, b);
            }
        }
        total
    }
}

fn routed_data() -> TabularData {
    table_from_joint(8000.0, ["t", "a", "b", "y"], routed::joint)
}

fn routed_admg() -> Admg {
    let mut graph = Admg::with_variables(4);
    for (from, to) in [(0, 1), (0, 2), (1, 3), (2, 3)] {
        graph.insert_directed(d(from), d(to)).unwrap();
    }
    graph
}

fn report<'a>(
    result: &'a antecedent::StudyResult,
    refuter: &str,
) -> &'a antecedent_validate::RefutationReport {
    result
        .refutations
        .iter()
        .find(|report| report.refuter.as_ref() == refuter)
        .unwrap_or_else(|| panic!("no `{refuter}` in {:?}", result.refutations))
}

#[test]
fn a_path_specific_effect_on_an_admg_is_refuted_against_the_structural_effect() {
    let truth = routed::effect_through_a();
    assert!(truth.abs() > 0.05, "the reference effect must be material: {truth}");
    let query = CausalQuery::PathSpecific(
        PathSpecificEffectQuery::binary(v(0), v(3)).with_path_nodes([v(1)]).with_max_len(4),
    );
    let ctx = ExecutionContext::for_tests(21);
    let study = |suite| {
        Study::tabular(routed_data())
            .graph(routed_admg())
            .query(query.clone())
            .refute(suite)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
    };
    let validated = study(RefuteSuite::Cheap).run(&ctx).unwrap();
    let plain = study(RefuteSuite::None).run(&ctx).unwrap();
    // The validation never moves the answer.
    let ate = |r: &antecedent::StudyResult| r.estimate.as_effect().unwrap().ate;
    assert!((ate(&validated) - ate(&plain)).abs() < 1e-12);
    assert!((ate(&validated) - truth).abs() < 0.02, "{} vs {truth}", ate(&validated));
    assert_eq!(validated.support_status, Some(CellStatus::Refused));
    // The placebo treatment removes the effect the identified functional reads: the refitted
    // value is centred on zero, far from the structural effect.
    let placebo = report(&validated, "target.placebo_treatment");
    assert!((placebo.original_ate - truth).abs() < 0.02);
    assert!(placebo.refuted_ate.abs() < 0.25 * truth.abs(), "{}", placebo.refuted_ate);
    // Subset stability: the refit on random 80% subsets stays at the original value.
    let subset = report(&validated, "target.subset_stability");
    assert!((subset.refuted_ate - subset.original_ate).abs() < 0.02, "{}", subset.refuted_ate);
    assert!(subset.replicates >= 9);
    // The cross-world placebo permutes the mediator: the effect through it vanishes.
    let cross_world = report(&validated, "target.cross_world_placebo");
    assert!(cross_world.refuted_ate.abs() < 0.03, "{}", cross_world.refuted_ate);
    assert!((cross_world.original_ate - truth).abs() < 0.02);
    // The sensitivity range is bound to its assumption and reports a break-even correlation.
    let range = report(&validated, "target.cross_world_sensitivity");
    assert!((range.original_ate - truth).abs() < 0.02);
    assert!(range.comparison > 0.0);
    assert!(validated.diagnostics.iter().any(|d| {
        d.code.as_ref() == "refute.functional.cross_world_range"
            && d.message.contains("mediator_outcome_confounding")
    }));
    assert!(validated.refutations.iter().all(|r| r.produced_by.is_some()));
    assert!(validated.diagnostics.iter().any(|d| d.code.as_ref() == "refute.functional.state"));
}

#[test]
fn an_interventional_distribution_on_an_admg_is_refuted_by_its_atom_vector() {
    let truth = routed::y_under_treatment();
    let query = CausalQuery::Distribution(InterventionalDistributionQuery::new(
        v(3),
        [Intervention::set(v(0), Value::f64(1.0))],
    ));
    let ctx = ExecutionContext::for_tests(22);
    let result = Study::tabular(routed_data())
        .graph(routed_admg())
        .query(query)
        .refute(RefuteSuite::Full)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    // Two outcome levels: the vector is the probability of the first atom, and the atom the
    // route publishes first is one of the two levels of Y, so it is `truth` or `1 - truth`.
    let first = result.distribution().unwrap().atoms[0].probability;
    assert!(
        (first - truth).abs() < 0.02 || (first - (1.0 - truth)).abs() < 0.02,
        "{first} vs {truth}"
    );
    assert_eq!(result.support_status, Some(CellStatus::Refused));
    for refuter in ["target.subset_stability", "target.bootstrap_stability"] {
        let check = report(&result, refuter);
        assert!((check.original_ate - first).abs() < 1e-12);
        assert!((check.refuted_ate - first).abs() < 0.02, "{refuter}: {}", check.refuted_ate);
    }
    // An interventional law has no treatment-keyed null contrast: only the distribution placebo
    // ran, and it compares the permuted-treatment law with the observed marginal of Y.
    assert!(result.refutations.iter().all(|r| r.refuter.as_ref() != "target.placebo_treatment"));
    let observed_y1 = routed_data().float64_values(v(3)).unwrap().iter().sum::<f64>()
        / routed_data().row_count() as f64;
    let placebo = report(&result, "target.distribution_placebo");
    assert!(
        (placebo.refuted_ate - observed_y1).abs() < 0.02
            || (placebo.refuted_ate - (1.0 - observed_y1)).abs() < 0.02,
        "{} vs marginal {observed_y1}",
        placebo.refuted_ate
    );
    // The intervened law itself is far from that marginal, so the placebo is informative.
    assert!(
        (first - placebo.refuted_ate).abs() > 0.02 || (truth - observed_y1).abs() < 0.03,
        "{first} vs {}",
        placebo.refuted_ate
    );
}

/// `z -> t -> y`, `z -> y` over `(z, t, y)` with a deterministic, well-conditioned table.
fn linear_fixture(n: u32) -> (TabularData, Dag) {
    let z: Vec<f64> = (0..n).map(|i| (f64::from(i) * 1.37).sin()).collect();
    let t: Vec<f64> = (0..n).zip(&z).map(|(i, z)| 0.6 * z + (f64::from(i) * 0.91).cos()).collect();
    let y: Vec<f64> = (0..n)
        .zip(t.iter().zip(&z))
        .map(|(i, (t, z))| 1.0 + 2.0 * t + z + 0.4 * (f64::from(i) * 2.17).sin())
        .collect();
    let data = TabularData::from_f64_columns([
        ("z", z.as_slice()),
        ("t", t.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap();
    let mut graph = Dag::with_variables(3);
    for (source, target) in [(0, 1), (0, 2), (1, 2)] {
        graph.insert_directed(d(source), d(target)).unwrap();
    }
    (data, graph)
}

fn attribution_study(data: &TabularData, graph: &Dag, accepted: bool, suite: RefuteSuite) -> Study {
    let builder = Study::tabular(data.clone());
    let builder = if accepted {
        builder.graph(AcceptedGraph::from(graph.clone()))
    } else {
        builder.graph(graph.clone())
    };
    builder
        .query(CausalQuery::AnomalyAttribution(AnomalyAttributionQuery::new([v(2)], 100)))
        .refute(suite)
        .build()
        .unwrap()
}

#[test]
fn an_attribution_is_refitted_whole_and_its_contributions_keep_the_efficiency_identity() {
    let (data, graph) = linear_fixture(96);
    let ctx = ExecutionContext::for_tests(23);
    let validated = attribution_study(&data, &graph, false, RefuteSuite::Full).run(&ctx).unwrap();
    let plain = attribution_study(&data, &graph, false, RefuteSuite::None).run(&ctx).unwrap();
    let scores = &validated.anomaly().unwrap()[0];
    let reference = &plain.anomaly().unwrap()[0];
    // The validation never moves the scores or the contributions.
    assert_eq!(scores.scores.len(), reference.scores.len());
    for (a, b) in scores.scores.iter().zip(reference.scores.iter()) {
        assert!((a - b).abs() < 1e-12);
    }
    for (a, b) in scores.noise_contributions.iter().zip(reference.noise_contributions.iter()) {
        assert!((a - b).abs() < 1e-12);
    }
    // Shapley efficiency, computed from the published rows: the contributions of a row add to
    // its score minus the all-reference baseline, so the mean score is the mean baseline plus
    // the sum of the mean contributions the other coordinates of the refuted vector carry.
    let rows = scores.rows.len();
    let width = scores.noise_components.len();
    let mean_score = scores.scores.iter().sum::<f64>() / rows as f64;
    let mean_baseline = scores.baseline_scores.iter().sum::<f64>() / rows as f64;
    let mean_contributions: f64 = (0..rows)
        .map(|row| (0..width).map(|j| scores.noise_contributions[row * width + j]).sum::<f64>())
        .sum::<f64>()
        / rows as f64;
    assert!(
        (mean_score - mean_baseline - mean_contributions).abs() < 1e-6 * (1.0 + mean_score.abs())
    );
    assert_eq!(validated.support_status, Some(CellStatus::Refused));
    for refuter in ["target.subset_stability", "target.bootstrap_stability"] {
        let check = report(&validated, refuter);
        assert!((check.original_ate - mean_score).abs() < 1e-9, "{refuter}");
        assert!(check.replicates >= 9);
        // Random 80% subsets and bootstrap resamples of a well-specified linear model move the
        // mean score by a fraction of its size, not by its size.
        assert!(
            (check.refuted_ate - mean_score).abs() < 0.5 * mean_score.abs() + 0.1,
            "{refuter}: {} vs {mean_score}",
            check.refuted_ate
        );
    }
    // Permuting the abducted residuals of a block (and regenerating its descendants) destroys
    // the unit-level contribution of that block: the alignment with the original is low.
    let blocks = report(&validated, "target.noise_block_permutation");
    assert!(blocks.comparison < 0.5, "alignment {}", blocks.comparison);
    assert!(blocks.passed, "{:?}", blocks.failure_condition);
    // Cheap runs the held-out-population check only.
    let cheap = attribution_study(&data, &graph, false, RefuteSuite::Cheap).run(&ctx).unwrap();
    assert!(cheap.refutations.iter().any(|r| r.refuter.as_ref() == "target.subset_stability"));
    assert!(cheap.refutations.iter().all(|r| r.refuter.as_ref() != "target.bootstrap_stability"));
    // An accepted structure is attributed as a supplied one, at validation none and validated.
    let accepted = attribution_study(&data, &graph, true, RefuteSuite::Cheap).run(&ctx).unwrap();
    let accepted_scores = &accepted.anomaly().unwrap()[0];
    for (a, b) in accepted_scores.scores.iter().zip(reference.scores.iter()) {
        assert!((a - b).abs() < 1e-12);
    }
    assert_eq!(accepted.support_status, Some(CellStatus::Refused));
}

#[test]
fn a_counterfactual_is_refuted_against_the_structural_slope() {
    let (data, graph) = linear_fixture(240);
    let (control, active) = (0.0, 1.5);
    let query =
        CounterfactualQuery::new(v(2), Arc::from([Intervention::set(v(1), Value::f64(active))]))
            .with_control_level(control);
    let ctx = ExecutionContext::for_tests(24);
    let result = Study::tabular(data)
        .graph(graph)
        .query(CausalQuery::Counterfactual(query))
        .refute(RefuteSuite::Cheap)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    // y = 1 + 2 t + z + noise: moving t from `control` to `active` moves every unit's y by
    // 2 (active - control) = 3, whatever its noise.
    let truth = 2.0 * (active - control);
    let mean_ite = result.counterfactual().unwrap().mean_ite;
    assert!((mean_ite - truth).abs() < 0.1, "{mean_ite} vs {truth}");
    assert_eq!(result.support_status, Some(CellStatus::Refused));
    let placebo = report(&result, "target.placebo_treatment");
    assert!((placebo.original_ate - mean_ite).abs() < 1e-9);
    // With the treatment permuted the fitted y-mechanism carries no slope in t.
    assert!(placebo.refuted_ate.abs() < 0.5, "{}", placebo.refuted_ate);
    let subset = report(&result, "target.subset_stability");
    assert!((subset.refuted_ate - truth).abs() < 0.2, "{}", subset.refuted_ate);
}

/// Orthonormal centred columns (unit RMS), orthogonal to the centred `t` and to each other.
fn orthogonal_noise(n: usize, t: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    let center = |v: &mut Vec<f64>| {
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        v.iter_mut().for_each(|x| *x -= mean);
    };
    let mut tc = t.to_vec();
    center(&mut tc);
    let mut a: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();
    center(&mut a);
    let k = dot(&a, &tc) / dot(&tc, &tc);
    a.iter_mut().zip(&tc).for_each(|(x, y)| *x -= k * y);
    let mut b: Vec<f64> = (0..n).map(|i| (i as f64 * 1.13).cos()).collect();
    center(&mut b);
    let kt = dot(&b, &tc) / dot(&tc, &tc);
    let ka = dot(&b, &a) / dot(&a, &a);
    for i in 0..n {
        b[i] -= kt * tc[i] + ka * a[i];
    }
    let unit = |v: &mut Vec<f64>| {
        let rms = (dot(v, v) / v.len() as f64).sqrt();
        v.iter_mut().for_each(|x| *x /= rms);
    };
    unit(&mut a);
    unit(&mut b);
    (a, b)
}

#[test]
fn a_natural_direct_effect_is_bounded_by_its_cross_world_dependence_and_placebo() {
    // M = 0.8 T + 0.5 a, Y = 0.3 T + 1.2 M + u with u = 0.5 (0.3 a + sqrt(0.91) b): the errors of
    // the mediator and outcome equations correlate at exactly 0.3 in the sample. The structural
    // direct effect is 0.3; sequential ignorability would read 0.3 - 0.3 * 0.8 = 0.06.
    let n = 600;
    // The nested route fits every mechanism as linear Gaussian, which a two-level (low
    // cardinality) column is not a candidate for, so the treatment is a continuous dose with the
    // same two-level centre; the errors are orthogonal to it in the sample whatever its values.
    let t: Vec<f64> =
        (0..n).map(|i| (i % 2) as f64 + 0.3 * (i as f64 * 0.91).sin()).collect();
    let (a, b) = orthogonal_noise(n, &t);
    let m: Vec<f64> = t.iter().zip(&a).map(|(t, a)| 0.8 * t + 0.5 * a).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| 0.3 * t[i] + 1.2 * m[i] + 0.5 * (0.3 * a[i] + 0.91_f64.sqrt() * b[i]))
        .collect();
    let data = TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("m", m.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap();
    let mut graph = Dag::with_variables(3);
    for (from, to) in [(0, 1), (0, 2), (1, 2)] {
        graph.insert_directed(d(from), d(to)).unwrap();
    }
    let result = Study::tabular(data)
        .graph(graph)
        .query(CausalQuery::NestedCounterfactual(
            NestedCounterfactualQuery::new(v(0), v(1), v(2)).unwrap(),
        ))
        .refute(RefuteSuite::Cheap)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(25))
        .unwrap();
    let nde = result.estimate.as_effect().unwrap().ate;
    assert!((nde - 0.06).abs() < 0.02, "{nde}");
    assert_eq!(result.support_status, Some(CellStatus::Refused));
    // At rho = 0.3 the sensitivity curve returns the structural direct effect, 0.3.
    let range = result
        .diagnostics
        .iter()
        .find(|d| d.code.as_ref() == "refute.functional.cross_world_range")
        .expect("the cross-world range is published");
    let tail = range.message.split("+0.30: ").nth(1).expect("the grid carries rho = 0.3");
    let value: f64 = tail[..tail.find(',').unwrap()].trim().parse().unwrap();
    assert!((value - 0.3).abs() < 0.02, "{value}");
    let sensitivity = report(&result, "target.cross_world_sensitivity");
    assert!((sensitivity.original_ate - nde).abs() < 1e-9);
    assert!(sensitivity.comparison > 0.0);
    // With the mediator permuted the direct effect is the total effect, 0.3 + 1.2 * 0.8 = 1.26.
    let placebo = report(&result, "target.cross_world_placebo");
    assert!((placebo.refuted_ate - 1.26).abs() < 0.1, "{}", placebo.refuted_ate);
    // The treatment placebo removes the direct effect.
    let treatment = report(&result, "target.placebo_treatment");
    assert!(treatment.refuted_ate.abs() < 0.1, "{}", treatment.refuted_ate);
}

#[test]
fn a_cpdag_path_specific_effect_is_refuted_per_completion_against_the_structural_effect() {
    let truth = routed::effect_through_a();
    let mut graph = Cpdag::with_variables(4);
    for (from, to) in [(0, 1), (0, 2), (1, 3), (2, 3)] {
        graph.insert_directed(d(from), d(to)).unwrap();
    }
    let query = CausalQuery::PathSpecific(
        PathSpecificEffectQuery::binary(v(0), v(3)).with_path_nodes([v(1)]).with_max_len(4),
    );
    let result = Study::tabular(routed_data())
        .graph(graph)
        .query(query)
        .refute(RefuteSuite::Cheap)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(26))
        .unwrap();
    assert_eq!(result.support_status, Some(CellStatus::Refused));
    let ate = result.estimate.as_effect().unwrap().ate;
    assert!((ate - truth).abs() < 0.03, "{ate} vs {truth}");
    let placebo = report(&result, "target.placebo_treatment");
    assert!(placebo.refuted_ate.abs() < 0.25 * truth.abs(), "{}", placebo.refuted_ate);
    let subset = report(&result, "target.subset_stability");
    assert!((subset.refuted_ate - ate).abs() < 0.03, "{}", subset.refuted_ate);
}

fn conditional_series() -> TimeSeriesData {
    let n = 80;
    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];
    let mut y = vec![0.0; n];
    for t in 0..n {
        x[t] = ((t as f64) * 0.37).sin();
        w[t] = if t % 2 == 0 { 0.0 } else { 1.0 };
        if t > 0 {
            y[t] = 1.5 * x[t - 1] + 0.2 * w[t] + 0.8 * x[t - 1] * w[t];
        }
    }
    TimeSeriesData::from_f64_columns(
        [("x", x.as_slice()), ("y", y.as_slice()), ("w", w.as_slice())],
        1,
    )
    .unwrap()
}

fn conditional_dag() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let x1 = ensure_lagged(&mut graph, v(0), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut graph, v(1), Lag::CONTEMPORANEOUS).unwrap();
    let w0 = ensure_lagged(&mut graph, v(2), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(x1, y0).unwrap();
    graph.insert_directed(w0, y0).unwrap();
    graph
}

#[test]
fn a_temporal_conditional_pulse_is_refuted_on_the_series_for_a_dag_and_its_class() {
    // Y_t = 1.5 X_{t-1} + 0.2 W_t + 0.8 X_{t-1} W_t with W ~ Bernoulli(1/2): the marginal pulse
    // effect is 1.5 + 0.8 / 2 = 1.9 (a closed form independent of the estimator).
    let truth = 1.9;
    let query = TemporalEffectQuery::pulse(v(0), v(1), 1.0)
        .with_horizon_steps(1)
        .with_max_history_lag(Some(1))
        .with_effect_modifiers([v(2)])
        .with_policy(TemporalPolicy::pulse(-1));
    let dag = conditional_dag();
    let class = TemporalCpdag::from_temporal_dag(&dag);
    let run = |study: antecedent::analysis::StudyBuilder| {
        study
            .query(CausalQuery::TemporalEffect(query.clone()))
            .refute(RefuteSuite::Cheap)
            .build()
            .unwrap()
            .run(&ExecutionContext::for_tests(27))
            .unwrap()
    };
    for result in [
        run(Study::series(conditional_series()).graph(dag.clone())),
        run(Study::series(conditional_series()).graph(class)),
    ] {
        assert_eq!(result.support_status, Some(CellStatus::Refused));
        let ate = result.estimate.as_effect().unwrap().ate;
        assert!((ate - truth).abs() < 0.05, "{ate}");
        // The circular shift of the treatment removes the lagged dependence.
        let placebo = report(&result, "target.placebo_treatment");
        assert!(placebo.refuted_ate.abs() < 0.5 * truth, "{}", placebo.refuted_ate);
        // Contiguous windows keep the series' dependence and the effect.
        let subset = report(&result, "target.subset_stability");
        assert!((subset.refuted_ate - truth).abs() < 0.15, "{}", subset.refuted_ate);
    }
}
