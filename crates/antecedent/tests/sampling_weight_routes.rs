//! Sampling weights on the randomized-trial, panel difference-in-differences and temporal
//! routes, through the facade.
//!
//! The table's observation weights are the sampling weights of these routes. Each is checked
//! against an oracle computed here from the definition (Hajek arm means and their linearized
//! variance; weighted group means of change scores and the weighted cluster score sandwich;
//! weighted least squares on lag windows), then for the weight contract (rescaling, zero-weight
//! rows counted and excluded, refusal of constructions without a weighted contrast), and
//! against the truth of a sampled super-population.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::needless_range_loop,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    reason = "test fixtures index small literals; the oracles are written as the formulas read"
)]

use std::sync::Arc;

use antecedent::prelude::ExecutionContext;
use antecedent::{PanelDidQuery, RandomizedEffectQuery, Study};
use antecedent_core::{
    CausalQuery, Lag, RandomizationDesign, StreamDomain, TemporalEffectQuery, TemporalPolicy,
    VariableId,
};
use antecedent_data::{TabularData, TimeSeriesData};
use antecedent_graph::{TemporalDag, ensure_lagged};
use antecedent_kernels::{kish_ess, sigmoid, standard_normal};

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(77)
}

fn close(actual: f64, expected: f64, tolerance: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= tolerance * (1.0 + expected.abs()),
        "{what}: {actual} vs {expected}"
    );
}

// ---------------------------------------------------------------------------------------------
// Randomized trial
// ---------------------------------------------------------------------------------------------

struct Trial {
    assigned: Vec<bool>,
    probability: Vec<f64>,
    outcome: Vec<f64>,
    weight: Vec<f64>,
}

/// Bernoulli assignment with probability one half on units whose weight rises with a baseline
/// that also drives the outcome and the effect (`1 + x/2`), so the population effect is `1`.
fn trial(n: usize, seed: u64, informative: bool) -> Trial {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Test, 0x7101);
    let (mut assigned, mut outcome, mut weight) = (vec![], vec![], vec![]);
    while assigned.len() < n {
        let x = standard_normal(&mut rng);
        let inclusion = if informative { sigmoid(-1.0 + x) } else { 0.4 };
        if rng.next_f64() >= inclusion {
            continue;
        }
        let treated = rng.next_f64() < 0.5;
        outcome
            .push(x + if treated { 1.0 + 0.5 * x } else { 0.0 } + 0.7 * standard_normal(&mut rng));
        assigned.push(treated);
        weight.push(1.0 / inclusion);
    }
    Trial { assigned, probability: vec![0.5; n], outcome, weight }
}

fn units(n: usize) -> Vec<Arc<str>> {
    (0..n).map(|i| Arc::<str>::from(format!("unit-{i}"))).collect()
}

fn bernoulli_query(trial: &Trial) -> RandomizedEffectQuery {
    RandomizedEffectQuery::bernoulli_itt(
        VariableId::from_raw(0),
        trial.assigned.clone(),
        trial.probability.clone(),
        units(trial.assigned.len()),
        units(trial.assigned.len()),
        ("control", "treated"),
    )
}

fn weighted_table(outcome: &[f64], weight: &[f64]) -> TabularData {
    TabularData::from_f64_columns([("outcome", outcome)])
        .unwrap()
        .with_weights(Arc::from(weight.to_vec()))
        .unwrap()
}

/// Hajek arm means with row weight `s_i / P(arm_i)` and the linearized variance, from the
/// definition.
fn hajek_oracle(trial: &Trial, weight: &[f64]) -> (f64, f64, usize, usize) {
    let rows: Vec<usize> = (0..trial.outcome.len()).filter(|&i| weight[i] > 0.0).collect();
    let arm = |treated: bool| -> (f64, f64, usize) {
        let members: Vec<usize> =
            rows.iter().copied().filter(|&i| trial.assigned[i] == treated).collect();
        let a = |i: usize| {
            let p = if treated { trial.probability[i] } else { 1.0 - trial.probability[i] };
            weight[i] / p
        };
        let mass: f64 = members.iter().map(|&i| a(i)).sum();
        let mean = members.iter().map(|&i| a(i) * trial.outcome[i]).sum::<f64>() / mass;
        let n = members.len() as f64;
        let spread: f64 = members.iter().map(|&i| (a(i) * (trial.outcome[i] - mean)).powi(2)).sum();
        (mean, spread * n / (n - 1.0) / (mass * mass), members.len())
    };
    let (m1, v1, n1) = arm(true);
    let (m0, v0, n0) = arm(false);
    (m1 - m0, v1 + v0, n1, n0)
}

fn run_trial(
    query: RandomizedEffectQuery,
    outcome: &[f64],
    weight: &[f64],
) -> antecedent::StudyResult {
    let data = weighted_table(outcome, weight);
    let prepared =
        Study::tabular(data.clone()).query(query).build().unwrap().prepare(&ctx()).unwrap();
    prepared.estimate(&data, &ctx()).unwrap()
}

#[test]
fn a_weighted_bernoulli_trial_is_the_hajek_contrast_with_its_linearized_variance() {
    let trial = trial(240, 1, true);
    let result = run_trial(bernoulli_query(&trial), &trial.outcome, &trial.weight);
    let fit = result.randomized_effect().unwrap();
    let (effect, variance, treated, control) = hajek_oracle(&trial, &trial.weight);
    close(fit.effect, effect, 1e-10, "weighted ITT");
    close(fit.variance_upper_bound, variance, 1e-10, "linearized variance");
    assert_eq!((fit.treatment_units, fit.control_units), (treated, control));
    assert!(fit.interval_95.is_none() && fit.standard_error.is_none(), "no interval is published");
    assert_eq!(fit.uncertainty.as_ref(), "sampling_weighted_hajek_linearized_variance_no_interval");

    let record = result
        .estimate
        .as_effect()
        .unwrap()
        .diagnostics
        .sampling_weights
        .expect("a weighted trial reports its weight record");
    assert_eq!(record.rows_used, 240);
    assert_eq!(record.zero_weight_rows, 0);
    close(record.effective_sample_size, kish_ess(&trial.weight), 1e-12, "Kish size");
    assert!(
        result
            .diagnostics
            .iter()
            .any(|d| d.code.as_ref() == "estimate.randomized.sampling_weights")
    );

    // The unweighted trial speaks for the sampled units, not the population.
    let plain = {
        let data = TabularData::from_f64_columns([("outcome", trial.outcome.as_slice())]).unwrap();
        let prepared = Study::tabular(data.clone())
            .query(bernoulli_query(&trial))
            .build()
            .unwrap()
            .prepare(&ctx())
            .unwrap();
        prepared.estimate(&data, &ctx()).unwrap()
    };
    assert!(plain.estimate.as_effect().unwrap().diagnostics.sampling_weights.is_none());
}

#[test]
fn weights_rescale_and_zero_weight_rows_are_counted_for_a_trial() {
    let trial = trial(200, 2, true);
    let base = run_trial(bernoulli_query(&trial), &trial.outcome, &trial.weight);
    let base_fit = base.randomized_effect().unwrap();

    let scaled: Vec<f64> = trial.weight.iter().map(|w| w * 1.0e-3).collect();
    let rescaled = run_trial(bernoulli_query(&trial), &trial.outcome, &scaled);
    let rescaled_fit = rescaled.randomized_effect().unwrap();
    close(rescaled_fit.effect, base_fit.effect, 1e-10, "rescaled effect");
    close(
        rescaled_fit.variance_upper_bound,
        base_fit.variance_upper_bound,
        1e-10,
        "rescaled variance",
    );

    // Rows of weight zero with garbage outcomes: the same contrast, and the count.
    let mut padded = Trial {
        assigned: trial.assigned.clone(),
        probability: trial.probability.clone(),
        outcome: trial.outcome.clone(),
        weight: trial.weight.clone(),
    };
    for k in 0..6 {
        padded.assigned.push(k % 2 == 0);
        padded.probability.push(0.5);
        padded.outcome.push(1.0e7 * (f64::from(k) + 1.0));
        padded.weight.push(0.0);
    }
    let zero = run_trial(bernoulli_query(&padded), &padded.outcome, &padded.weight);
    let zero_fit = zero.randomized_effect().unwrap();
    close(zero_fit.effect, base_fit.effect, 1e-10, "effect with zero-weight rows");
    close(zero_fit.variance_upper_bound, base_fit.variance_upper_bound, 1e-10, "variance");
    let record = zero.estimate.as_effect().unwrap().diagnostics.sampling_weights.unwrap();
    assert_eq!((record.rows_used, record.zero_weight_rows), (200, 6));
}

#[test]
fn a_weighted_completely_randomized_trial_matches_the_oracle() {
    let mut trial = trial(200, 3, true);
    // Re-draw the assignment as a complete randomization of exactly 80 units.
    let n = trial.assigned.len();
    trial.assigned = (0..n).map(|i| i % 5 < 2).collect();
    trial.probability = vec![0.4; n];
    let treated_units = trial.assigned.iter().filter(|a| **a).count();
    let query = RandomizedEffectQuery::with_design(
        RandomizationDesign::Complete { treated_units },
        VariableId::from_raw(0),
        trial.assigned.clone(),
        trial.probability.clone(),
        units(n),
        units(n),
        ("control", "treated"),
    );
    let result = run_trial(query, &trial.outcome, &trial.weight);
    let fit = result.randomized_effect().unwrap();
    let (effect, variance, ..) = hajek_oracle(&trial, &trial.weight);
    close(fit.effect, effect, 1e-10, "complete-design weighted ITT");
    close(fit.variance_upper_bound, variance, 1e-10, "complete-design variance");
    assert_eq!(fit.assignment_design.as_ref(), "complete");
}

#[test]
fn designs_without_a_weighted_contrast_refuse_the_weights() {
    let trial = trial(120, 4, false);
    let data = weighted_table(&trial.outcome, &trial.weight);
    let n = trial.assigned.len();
    // Cluster randomization of 20 clusters of six rows.
    let clusters: Vec<Arc<str>> =
        (0..n).map(|i| Arc::<str>::from(format!("cluster-{}", i / 6))).collect();
    let assigned: Vec<bool> = (0..n).map(|i| (i / 6) % 2 == 0).collect();
    let query = RandomizedEffectQuery::with_design(
        RandomizationDesign::Cluster { treated_clusters: 10 },
        VariableId::from_raw(0),
        assigned,
        vec![0.5; n],
        clusters,
        units(n),
        ("control", "treated"),
    );
    let prepared =
        Study::tabular(data.clone()).query(query).build().unwrap().prepare(&ctx()).unwrap();
    let error = prepared.estimate(&data, &ctx()).unwrap_err().to_string();
    assert!(error.contains("sampling_weights.route_unsupported"), "{error}");
    assert!(error.starts_with("reason=option_not_applicable"), "{error}");
}

#[test]
fn invalid_trial_weights_are_refused_by_the_table_naming_the_first_row() {
    let outcome = vec![1.0, 2.0, 3.0, 4.0];
    let error = TabularData::from_f64_columns([("outcome", outcome.as_slice())])
        .unwrap()
        .with_weights(Arc::from(vec![1.0, -2.0, 1.0, 1.0]))
        .unwrap_err()
        .to_string();
    assert!(error.contains("row 1") && error.contains("negative"), "{error}");
}

/// The population effect is exactly `1`: weighted, the trial recovers it within four standard
/// errors of its own variance; unweighted, it estimates the effect among over-sampled units.
#[test]
fn a_weighted_trial_recovers_the_population_effect_the_unweighted_one_misses() {
    let trial = trial(6_000, 5, true);
    let result = run_trial(bernoulli_query(&trial), &trial.outcome, &trial.weight);
    let fit = result.randomized_effect().unwrap();
    assert!(
        (fit.effect - 1.0).abs() < 4.0 * fit.variance_upper_bound.sqrt(),
        "{} (se {})",
        fit.effect,
        fit.variance_upper_bound.sqrt()
    );
    let (unweighted, ..) = hajek_oracle(&trial, &vec![1.0; trial.assigned.len()]);
    assert!(unweighted > 1.15, "the sampled units' effect is larger: {unweighted}");
}

#[test]
fn the_weighted_trial_result_round_trips_through_its_artifact() {
    let trial = trial(120, 6, true);
    let data = weighted_table(&trial.outcome, &trial.weight);
    let prepared = Study::tabular(data.clone())
        .query(bernoulli_query(&trial))
        .build()
        .unwrap()
        .prepare(&ctx())
        .unwrap();
    let result = prepared.estimate(&data, &ctx()).unwrap();
    let bytes = prepared.encode_contracted_result(&result, "weighted-trial", &ctx()).unwrap();
    let (_, _, artifact) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    let encoded = artifact.randomized_effect.unwrap();
    assert_eq!(
        encoded.uncertainty.as_str(),
        "sampling_weighted_hajek_linearized_variance_no_interval"
    );
    assert!(encoded.interval_95.is_none());
}

// ---------------------------------------------------------------------------------------------
// Balanced two-period panel difference-in-differences
// ---------------------------------------------------------------------------------------------

struct Panel {
    outcome: Vec<f64>,
    weight: Vec<f64>,
    treated: Vec<bool>,
    post: Vec<bool>,
    subjects: Vec<Arc<str>>,
    clusters: Vec<Arc<str>>,
}

/// `clusters_per_arm` clusters of four sampled subjects per arm; the effect on a treated
/// subject is `1 + z/2`, so the population effect is `1`, and inclusion rises with `z`.
fn panel(clusters_per_arm: usize, seed: u64) -> Panel {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Test, 0x7102);
    let mut out = Panel {
        outcome: vec![],
        weight: vec![],
        treated: vec![],
        post: vec![],
        subjects: vec![],
        clusters: vec![],
    };
    let mut subject = 0;
    for cluster in 0..2 * clusters_per_arm {
        let active = cluster >= clusters_per_arm;
        let shock = 0.6 * standard_normal(&mut rng);
        let mut members = 0;
        while members < 4 {
            let z = standard_normal(&mut rng) + 0.5 * shock;
            let inclusion = sigmoid(-1.0 + z);
            if rng.next_f64() >= inclusion {
                continue;
            }
            let baseline = z + 0.5 * standard_normal(&mut rng);
            let effect = if active { 1.0 + 0.5 * z } else { 0.0 };
            let change = 0.3 + effect + shock + 0.5 * standard_normal(&mut rng);
            for (after, value) in [(false, baseline), (true, baseline + change)] {
                out.subjects.push(Arc::from(format!("s{subject}")));
                out.clusters.push(Arc::from(format!("c{cluster}")));
                out.treated.push(active);
                out.post.push(after);
                out.outcome.push(value);
                out.weight.push(1.0 / inclusion);
            }
            subject += 1;
            members += 1;
        }
    }
    out
}

/// Weighted group means of change scores and the weighted cluster score sandwich, from the
/// definition, over subjects of positive weight.
fn did_oracle(panel: &Panel) -> (f64, f64) {
    let subjects = panel.outcome.len() / 2;
    let rows: Vec<usize> = (0..subjects).filter(|&j| panel.weight[2 * j] > 0.0).collect();
    let total: f64 = rows.iter().map(|&j| panel.weight[2 * j]).sum();
    let scale = rows.len() as f64 / total;
    let change = |j: usize| panel.outcome[2 * j + 1] - panel.outcome[2 * j];
    let group = |treated: bool| -> Vec<usize> {
        rows.iter().copied().filter(|&j| panel.treated[2 * j] == treated).collect()
    };
    let mean = |members: &[usize]| -> (f64, f64) {
        let mass: f64 = members.iter().map(|&j| panel.weight[2 * j] * scale).sum();
        (members.iter().map(|&j| panel.weight[2 * j] * scale * change(j)).sum::<f64>() / mass, mass)
    };
    let (treated, control) = (group(true), group(false));
    let ((m1, mass1), (m0, mass0)) = (mean(&treated), mean(&control));
    let mut scores = std::collections::BTreeMap::<&str, f64>::new();
    for (members, m, mass, sign) in [(&treated, m1, mass1, 1.0), (&control, m0, mass0, -1.0)] {
        for &j in members {
            *scores.entry(panel.clusters[2 * j].as_ref()).or_default() +=
                sign * panel.weight[2 * j] * scale * (change(j) - m) / mass;
        }
    }
    let g = scores.len() as f64;
    (m1 - m0, (g / (g - 1.0) * scores.values().map(|v| v * v).sum::<f64>()).sqrt())
}

fn run_panel(panel: &Panel, weights: &[f64]) -> antecedent::StudyResult {
    let query = PanelDidQuery::new(
        VariableId::from_raw(0),
        panel.treated.clone(),
        panel.post.clone(),
        panel.subjects.clone(),
        panel.clusters.clone(),
    );
    let data = weighted_table(&panel.outcome, weights);
    let prepared =
        Study::tabular(data.clone()).query(query).build().unwrap().prepare(&ctx()).unwrap();
    prepared.estimate(&data, &ctx()).unwrap()
}

#[test]
fn a_weighted_panel_did_is_the_weighted_change_contrast_with_a_weighted_cluster_sandwich() {
    let panel = panel(25, 11);
    let result = run_panel(&panel, &panel.weight);
    let fit = result.panel_did().unwrap();
    let (effect, se) = did_oracle(&panel);
    close(fit.effect, effect, 1e-10, "weighted DiD");
    close(fit.standard_error, se, 1e-10, "weighted cluster score sandwich");
    assert!(fit.interval_95.is_none());
    assert_eq!(
        fit.uncertainty.as_ref(),
        "sampling_weighted_cluster_robust_standard_error_no_interval"
    );
    let record = result.estimate.as_effect().unwrap().diagnostics.sampling_weights.unwrap();
    assert_eq!(record.rows_used, 200);
    let per_subject: Vec<f64> = (0..200).map(|j| panel.weight[2 * j]).collect();
    close(record.effective_sample_size, kish_ess(&per_subject), 1e-12, "Kish size");
}

#[test]
fn panel_weights_rescale_and_zero_weight_subjects_are_counted() {
    let panel = panel(25, 12);
    let base = run_panel(&panel, &panel.weight);
    let base_fit = base.panel_did().unwrap();
    let scaled: Vec<f64> = panel.weight.iter().map(|w| w * 500.0).collect();
    let big = run_panel(&panel, &scaled);
    close(big.panel_did().unwrap().effect, base_fit.effect, 1e-10, "rescaled effect");
    close(big.panel_did().unwrap().standard_error, base_fit.standard_error, 1e-10, "rescaled SE");

    // Two extra subjects of weight zero, one per arm, in clusters that already exist.
    let mut padded = Panel {
        outcome: panel.outcome.clone(),
        weight: panel.weight.clone(),
        treated: panel.treated.clone(),
        post: panel.post.clone(),
        subjects: panel.subjects.clone(),
        clusters: panel.clusters.clone(),
    };
    for (k, (cluster, active)) in [("c0", false), ("c49", true)].into_iter().enumerate() {
        for after in [false, true] {
            padded.outcome.push(1.0e6 * (k as f64 + 1.0) + f64::from(u8::from(after)));
            padded.weight.push(0.0);
            padded.treated.push(active);
            padded.post.push(after);
            padded.subjects.push(Arc::from(format!("zero-{k}")));
            padded.clusters.push(Arc::from(cluster));
        }
    }
    let zero = run_panel(&padded, &padded.weight);
    close(zero.panel_did().unwrap().effect, base_fit.effect, 1e-10, "effect with zero weights");
    close(zero.panel_did().unwrap().standard_error, base_fit.standard_error, 1e-10, "SE");
    let record = zero.estimate.as_effect().unwrap().diagnostics.sampling_weights.unwrap();
    assert_eq!((record.rows_used, record.zero_weight_rows), (200, 2));
}

#[test]
fn a_subject_whose_two_rows_carry_different_weights_is_refused() {
    let panel = panel(8, 13);
    let mut weights = panel.weight.clone();
    weights[1] *= 2.0;
    let query = PanelDidQuery::new(
        VariableId::from_raw(0),
        panel.treated.clone(),
        panel.post.clone(),
        panel.subjects.clone(),
        panel.clusters.clone(),
    );
    let data = weighted_table(&panel.outcome, &weights);
    let prepared =
        Study::tabular(data.clone()).query(query).build().unwrap().prepare(&ctx()).unwrap();
    let error = prepared.estimate(&data, &ctx()).unwrap_err().to_string();
    assert!(error.contains("same sampling weight"), "{error}");
}

#[test]
fn a_weighted_panel_recovers_the_population_effect_the_unweighted_one_misses() {
    let panel = panel(150, 14);
    let weighted = run_panel(&panel, &panel.weight);
    let fit = weighted.panel_did().unwrap();
    assert!(
        (fit.effect - 1.0).abs() < 4.0 * fit.standard_error,
        "{} (se {})",
        fit.effect,
        fit.standard_error
    );
    let unit = vec![1.0; panel.weight.len()];
    let unweighted = run_panel(&panel, &unit);
    assert!(
        unweighted.panel_did().unwrap().effect > fit.effect + 0.1,
        "unweighted {} vs weighted {}",
        unweighted.panel_did().unwrap().effect,
        fit.effect
    );
}

#[test]
fn repeated_cross_sections_and_staggered_designs_refuse_weights() {
    let panel = panel(8, 15);
    let query = PanelDidQuery::repeated_cross_section(
        VariableId::from_raw(0),
        panel.treated.clone(),
        panel.post.clone(),
        panel.subjects.clone(),
        panel.clusters.clone(),
    );
    let data = weighted_table(&panel.outcome, &panel.weight);
    let prepared =
        Study::tabular(data.clone()).query(query).build().unwrap().prepare(&ctx()).unwrap();
    let error = prepared.estimate(&data, &ctx()).unwrap_err().to_string();
    assert!(error.contains("sampling_weights.route_unsupported"), "{error}");
}

// ---------------------------------------------------------------------------------------------
// Temporal single-window linear adjustment
// ---------------------------------------------------------------------------------------------

fn series(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Test, 0x7103);
    let t: Vec<f64> = (0..n).map(|_| standard_normal(&mut rng)).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            1.0 + 2.0 * i.checked_sub(1).map_or(0.0, |j| t[j])
                + 3.0 * i.checked_sub(2).map_or(0.0, |j| t[j])
                + 0.8 * standard_normal(&mut rng)
        })
        .collect();
    (t, y)
}

fn temporal_graph() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let t = VariableId::from_raw(0);
    let y = VariableId::from_raw(1);
    let t1 = ensure_lagged(&mut graph, t, Lag::from_raw(1)).unwrap();
    let t2 = ensure_lagged(&mut graph, t, Lag::from_raw(2)).unwrap();
    let y0 = ensure_lagged(&mut graph, y, Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(t1, y0).unwrap();
    graph.insert_directed(t2, y0).unwrap();
    graph
}

fn pulse() -> TemporalEffectQuery {
    TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
}

fn run_series(t: &[f64], y: &[f64], weights: &[f64]) -> Result<antecedent::StudyResult, String> {
    let data = TimeSeriesData::from_f64_columns([("t", t), ("y", y)], 1)
        .unwrap()
        .with_weights(Arc::from(weights.to_vec()))
        .unwrap();
    Study::series(data)
        .graph(temporal_graph())
        .query(CausalQuery::TemporalEffect(pulse()))
        .refute(antecedent::RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .map_err(|e| e.to_string())?
        .run(&ctx())
        .map_err(|e| e.to_string())
}

/// Weighted least squares of `y_i` on `[1, t_{i-1}]` for `i >= 1`, window `i` weighted by the
/// weight of the observation it ends at; the coefficient of `t_{i-1}`. The lag-1 pulse adjusts
/// for no other lag (the lag-2 treatment is independent of it), so the windows start at the
/// first row with a lag-1 value.
fn temporal_oracle(t: &[f64], y: &[f64], weights: &[f64]) -> f64 {
    let (mut sw, mut sx, mut sxx, mut sy, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for i in 1..t.len() {
        let (w, x) = (weights[i], t[i - 1]);
        sw += w;
        sx += w * x;
        sxx += w * x * x;
        sy += w * y[i];
        sxy += w * x * y[i];
    }
    (sw * sxy - sx * sy) / (sw * sxx - sx * sx)
}

#[test]
fn a_weighted_temporal_window_is_weighted_least_squares_on_lag_windows() {
    let n = 300;
    let (t, y) = series(n, 21);
    let mut rng = ExecutionContext::for_tests(22).rng.stream_for(StreamDomain::Test, 0x7104);
    let weights: Vec<f64> = (0..n).map(|_| 0.3 + 3.0 * rng.next_f64()).collect();
    let result = run_series(&t, &y, &weights).unwrap();
    let effect = result.estimate.as_effect().unwrap();
    close(effect.ate, temporal_oracle(&t, &y, &weights), 1e-9, "weighted window coefficient");
    let record = effect.diagnostics.sampling_weights.expect("record");
    assert_eq!(record.rows_used, (n - 1) as u64);
    close(record.effective_sample_size, kish_ess(&weights[1..]), 1e-12, "Kish size of the windows");

    // Rescaling the weights changes nothing.
    let scaled: Vec<f64> = weights.iter().map(|w| w * 40.0).collect();
    let again = run_series(&t, &y, &scaled).unwrap();
    close(again.estimate.as_effect().unwrap().ate, effect.ate, 1e-9, "rescaled coefficient");
}

#[test]
fn temporal_windows_of_weight_zero_stay_in_sequence_and_are_counted() {
    let n = 200;
    let (t, y) = series(n, 23);
    let mut weights = vec![1.0; n];
    for i in (10..n).step_by(17) {
        weights[i] = 0.0;
    }
    let zeros = weights[1..].iter().filter(|w| **w == 0.0).count() as u64;
    let result = run_series(&t, &y, &weights).unwrap();
    let effect = result.estimate.as_effect().unwrap();
    close(effect.ate, temporal_oracle(&t, &y, &weights), 1e-9, "coefficient with zero weights");
    let record = effect.diagnostics.sampling_weights.unwrap();
    assert_eq!(record.zero_weight_rows, zeros);
    assert_eq!(record.rows_used, (n - 1) as u64 - zeros);
}

#[test]
fn temporal_weights_refuse_constructions_without_a_weighted_fit() {
    let n = 200;
    let (t, y) = series(n, 24);
    let weights = vec![1.0; n];
    let data = TimeSeriesData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice())], 1)
        .unwrap()
        .with_weights(Arc::from(weights))
        .unwrap();
    let error = Study::series(data)
        .graph(temporal_graph())
        .query(CausalQuery::TemporalEffect(pulse()))
        .refute(antecedent::RefuteSuite::Cheap)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ctx())
        .unwrap_err()
        .to_string();
    assert!(error.contains("sampling_weights.route_unsupported"), "{error}");
}
