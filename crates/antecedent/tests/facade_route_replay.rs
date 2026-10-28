//! Replay and independent uncertainty references for the facade routes that execute outside the
//! support matrix: the declared white-innovation posterior, the sequential contrast, the
//! period-history match, the randomized quantile posterior and the Bayesian switchback effect.
//!
//! Each route is prepared once, executed twice from the retained plan, and read back through the
//! exported artifact. The replay must carry the same lowered program identity, reproduce the
//! published number bit for bit, and be recovered by the independent consumer. The estimate and
//! the reported uncertainty are compared with a computation that shares no code with the route:
//! least squares from the raw series, numpy and scipy pins, a brute-force donor search, and a
//! Dirichlet simulation with its own gamma sampler for the quantile posterior.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    reason = "integration test builds deterministic fixtures from small counts"
)]

use std::sync::Arc;

use antecedent::estimate::InnovationWhitenessGate;
use antecedent::prelude::ExecutionContext;
use antecedent::{
    BayesianConfig, DesignPosteriorModel, EstimatorId, InferenceMode, PreparedStudy,
    RandomizedEffectQuery, RefuteSuite, Study, StudyResult,
};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, HistoryMatchQuery, Lag, OutcomeFunctional,
    RandomizationDesign, SequentialContrastQuery, TemporalEffectQuery, TemporalPolicy, VariableId,
};
use antecedent_data::{TabularData, TimeSeriesData};
use antecedent_estimate::switchback_posterior::{
    CarryoverDeclaration, PeriodAdjustment, SequenceDependence, SwitchbackModel, SwitchbackPrior,
};
use antecedent_graph::{Dag, DenseNodeId, TemporalDag, ensure_lagged};

struct Lcg(u64);

impl Lcg {
    fn uniform(&mut self) -> f64 {
        self.0 =
            self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64 + 0.5) / (1_u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.uniform();
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Marsaglia-Tsang gamma sampler with unit scale.
    fn gamma(&mut self, shape: f64) -> f64 {
        if shape < 1.0 {
            let boost = self.uniform().powf(1.0 / shape);
            return self.gamma(shape + 1.0) * boost;
        }
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.normal();
            let v = (1.0 + c * x).powi(3);
            if v <= 0.0 {
                continue;
            }
            let u = self.uniform();
            if u.ln() < 0.5 * x * x + d - d * v + d * v.ln() {
                return d * v;
            }
        }
    }
}

fn close(actual: f64, expected: f64, tolerance: f64, what: &str) {
    assert!((actual - expected).abs() <= tolerance, "{what}: {actual} vs {expected}");
}

/// Execute the retained plan twice and return both results with the artifact bytes of the first.
/// The two executions agree on the lowered program and on the published scalar.
fn replay(
    prepared: &PreparedStudy,
    run: &dyn Fn(&PreparedStudy, &ExecutionContext) -> StudyResult,
    wire_id: &str,
) -> (StudyResult, antecedent_io::AnalysisResultConsumption) {
    let ctx = ExecutionContext::for_tests(0xA11);
    let first = run(prepared, &ctx);
    let again = run(prepared, &ctx);
    let identity = first.lowered_program_identity();
    assert!(identity.is_some(), "{wire_id}: the result carries no lowered program identity");
    assert_eq!(identity, again.lowered_program_identity(), "{wire_id}");
    assert_eq!(first.effect().to_bits(), again.effect().to_bits(), "{wire_id}: rerun differs");
    let bytes = prepared.encode_contracted_result(&first, wire_id, &ctx).unwrap();
    let consumed = antecedent_io::consume_analysis_result(&bytes).unwrap();
    (first, consumed)
}

// ---------------------------------------------------------------------------------------------
// Declared white-innovation posterior
// ---------------------------------------------------------------------------------------------

const N: usize = 400;
const DRAWS: usize = 4000;

fn white_series() -> (Vec<f64>, Vec<f64>) {
    let mut rng = Lcg(2);
    let (mut x, mut y) = (vec![0.0; N], vec![0.0; N]);
    let mut previous = 0.0_f64;
    for t in 0..N {
        previous = 0.5 * previous + rng.normal();
        x[t] = previous;
        let e = rng.normal();
        y[t] = if t > 0 { 0.8 * x[t - 1] } else { 0.0 } + e;
    }
    (x, y)
}

#[test]
fn the_white_innovation_posterior_replays_and_matches_least_squares() {
    let (x, y) = white_series();
    let data =
        TimeSeriesData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())], 1).unwrap();
    let mut graph = TemporalDag::empty();
    let x1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(x1, y0).unwrap();
    let query = CausalQuery::TemporalEffect(
        TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
            .with_policy(TemporalPolicy::pulse(-1))
            .with_horizon_steps(1),
    );
    let config = BayesianConfig::conjugate()
        .n_draws(DRAWS)
        .prior_scale(1.0e3)
        .white_innovation(InnovationWhitenessGate { max_lag: 6, level: 0.01 });
    let prepared = Study::series(data.clone())
        .graph(graph)
        .query(query)
        .inference(InferenceMode::Bayesian(config))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ExecutionContext::for_tests(11))
        .unwrap();
    let (result, consumed) =
        replay(&prepared, &|p, ctx| p.estimate_series(&data, ctx).unwrap(), "white-innovation");
    assert_eq!(consumed.body.estimate, Some(result.effect()));

    // Least squares of y_t on x_{t-1}, rows 1..N, from the raw series.
    let n = (N - 1) as f64;
    let (xs, ys) = (&x[..N - 1], &y[1..]);
    let (xm, ym) = (xs.iter().sum::<f64>() / n, ys.iter().sum::<f64>() / n);
    let sxx: f64 = xs.iter().map(|v| (v - xm).powi(2)).sum();
    let sxy: f64 = xs.iter().zip(ys).map(|(a, b)| (a - xm) * (b - ym)).sum();
    let beta = sxy / sxx;
    let rss: f64 = xs.iter().zip(ys).map(|(a, b)| (b - ym - beta * (a - xm)).powi(2)).sum();
    let se = (rss / (n - 2.0) / sxx).sqrt();
    let posterior = result.posterior.as_ref().expect("posterior");
    let column = posterior.effect_column().expect("effect column");
    close(
        posterior.summaries.mean[column],
        beta,
        3.0 * se / (DRAWS as f64).sqrt() + 2e-3,
        "posterior mean",
    );
    assert!((posterior.summaries.sd[column] / se - 1.0).abs() < 0.06);
}

// ---------------------------------------------------------------------------------------------
// Observational sequential contrast
// ---------------------------------------------------------------------------------------------

const SUBJECTS: usize = 80;

fn lcg_unit(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
    (*state >> 11) as f64 / 9_007_199_254_740_992.0
}

/// Columns l0, a0, l1, a1, y; the stream whose numpy reference values are pinned below.
fn sequential_panel() -> TabularData {
    let mut state = 12_345_u64;
    let (mut l0s, mut a0s, mut l1s, mut a1s, mut ys) = (vec![], vec![], vec![], vec![], vec![]);
    for _ in 0..SUBJECTS {
        let u: Vec<f64> = (0..5).map(|_| lcg_unit(&mut state)).collect();
        let l0 = 2.0 * u[0] - 1.0;
        let a0 = if u[1] < 0.5 + 0.2 * l0 { 1.0 } else { 0.0 };
        let l1 = 0.5 * l0 + 0.8 * a0 + (2.0 * u[2] - 1.0) * 0.5;
        let a1 = if u[3] < 0.5 + 0.1 * (l1 - 0.4) { 1.0 } else { 0.0 };
        ys.push(1.0 + 0.5 * l0 + 1.0 * a0 + 1.5 * a1 + 0.4 * l1 + (2.0 * u[4] - 1.0));
        l0s.push(l0);
        a0s.push(a0);
        l1s.push(l1);
        a1s.push(a1);
    }
    TabularData::from_f64_columns([
        ("l0", l0s.as_slice()),
        ("a0", a0s.as_slice()),
        ("l1", l1s.as_slice()),
        ("a1", a1s.as_slice()),
        ("y", ys.as_slice()),
    ])
    .unwrap()
}

#[test]
fn the_sequential_contrast_replays_and_matches_the_numpy_reference() {
    let data = sequential_panel();
    let query = SequentialContrastQuery {
        outcome: VariableId::from_raw(4),
        first_treatment: VariableId::from_raw(1),
        second_treatment: VariableId::from_raw(3),
        baseline: Arc::from([VariableId::from_raw(0)]),
        intermediate: Arc::from([VariableId::from_raw(2)]),
        active: [1, 1],
        control: [0, 0],
        fold_of_row: (0..SUBJECTS).map(|row| u32::from(matches!(row % 4, 1 | 2))).collect(),
        fold_count: 2,
        min_probability: 0.05,
    };
    let prepared = Study::tabular(data.clone())
        .query(CausalQuery::SequentialContrast(query))
        .build()
        .unwrap()
        .prepare(&ExecutionContext::for_tests(41))
        .unwrap();
    let (result, consumed) =
        replay(&prepared, &|p, ctx| p.estimate(&data, ctx).unwrap(), "sequential-wald");
    let fit = result.sequential_contrast().unwrap();
    close(fit.contrast, 2.230_853_305_140_346, 1e-4, "contrast");
    close(fit.standard_error.unwrap(), 0.689_242_100_473_076_1, 1e-4, "standard error");
    // The interval is the normal interval of that standard error.
    let [lower, upper] = fit.interval.unwrap();
    close(upper - lower, 2.0 * 1.959_963_984_540_054 * fit.standard_error.unwrap(), 1e-9, "width");
    // The independent consumer recovers the same contrast and standard error.
    let wire = consumed.body.sequential_contrast.as_ref().expect("sequential contrast section");
    assert_eq!(wire.contrast.to_bits(), fit.contrast.to_bits());
    assert_eq!(wire.standard_error, fit.standard_error);
}

// ---------------------------------------------------------------------------------------------
// Within-period history match
// ---------------------------------------------------------------------------------------------

#[test]
fn the_history_match_replays_and_matches_the_brute_force_donor_search() {
    type Row = (f64, [f64; 2], f64, f64, f64);
    let rows: [Row; 9] = [
        (1.0, [1.0, 3.0], 10.0, 1.0, 1.0),
        (1.0, [1.0, 1.0], 4.0, 0.0, 1.0),
        (1.0, [3.0, 3.0], 6.0, 0.0, 1.0),
        (1.0, [2.8, 3.0], 9.0, 1.0, 2.0),
        (2.0, [2.0, 2.0], 5.0, 0.0, 1.0),
        (2.0, [6.0, 6.0], 7.0, 0.0, 1.0),
        (2.0, [5.0, 6.0], 12.0, 1.0, 1.5),
        (2.0, [9.0, 9.0], 20.0, 1.0, 1.0),
        (3.0, [1.0, 1.0], 3.0, 1.0, 1.0),
    ];
    let col = |f: &dyn Fn(&Row) -> f64| -> Vec<f64> { rows.iter().map(f).collect() };
    let (h1, h2) = (col(&|r| r.1[0]), col(&|r| r.1[1]));
    let (y, treated, period, weight) = (col(&|r| r.2), col(&|r| r.3), col(&|r| r.0), col(&|r| r.4));
    let data = TabularData::from_f64_columns([
        ("h1", h1.as_slice()),
        ("h2", h2.as_slice()),
        ("y", y.as_slice()),
        ("treated", treated.as_slice()),
        ("period", period.as_slice()),
        ("weight", weight.as_slice()),
    ])
    .unwrap();
    let caliper = 2.0;
    let query = HistoryMatchQuery {
        outcome: VariableId::from_raw(2),
        treated: VariableId::from_raw(3),
        history: Arc::from([VariableId::from_raw(0), VariableId::from_raw(1)]),
        period: Some(VariableId::from_raw(4)),
        weight: Some(VariableId::from_raw(5)),
        caliper: Some(caliper),
    };
    let prepared = Study::tabular(data.clone())
        .query(CausalQuery::HistoryMatch(query))
        .build()
        .unwrap()
        .prepare(&ExecutionContext::for_tests(44))
        .unwrap();
    let (result, consumed) =
        replay(&prepared, &|p, ctx| p.estimate(&data, ctx).unwrap(), "history-match");
    let fit = result.history_match().unwrap();

    // Brute force: each treated row takes the nearest untreated donor of its period within the
    // caliper (Euclidean distance on the history); the contrast is the weighted mean difference.
    let (mut weighted, mut total, mut matched) = (0.0, 0.0, Vec::new());
    for (i, q) in rows.iter().enumerate().filter(|(_, r)| r.3 == 1.0) {
        let donor = rows
            .iter()
            .enumerate()
            .filter(|(_, d)| d.3 == 0.0 && d.0 == q.0)
            .map(|(j, d)| (j, ((q.1[0] - d.1[0]).powi(2) + (q.1[1] - d.1[1]).powi(2)).sqrt()))
            .filter(|(_, dist)| *dist <= caliper)
            .min_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((j, _)) = donor {
            weighted += q.4 * (q.2 - rows[j].2);
            total += q.4;
            matched.push((i, j));
        }
    }
    close(fit.contrast, weighted / total, 1e-12, "contrast");
    close(fit.weight_total, total, 1e-12, "weight");
    let queries: Vec<usize> = fit.matched_queries.iter().map(|&v| v as usize).collect();
    let donors: Vec<usize> = fit.matched_donors.iter().map(|&v| v as usize).collect();
    assert_eq!(matched.iter().map(|m| m.0).collect::<Vec<_>>(), queries);
    assert_eq!(matched.iter().map(|m| m.1).collect::<Vec<_>>(), donors);
    close(fit.standard_error.unwrap(), 2.096_440_251_568_134, 1e-12, "Abadie-Imbens se");
    let wire = consumed.body.history_match.as_ref().expect("history match section");
    assert_eq!(wire.contrast.to_bits(), fit.contrast.to_bits());
    assert_eq!(wire.standard_error, fit.standard_error);
}

// ---------------------------------------------------------------------------------------------
// Randomized quantile posterior
// ---------------------------------------------------------------------------------------------

#[test]
fn the_quantile_posterior_replays_and_matches_a_dirichlet_simulation() {
    let control = [0.4, 0.8, 1.1, 1.3, 1.7, 2.2, 2.4, 2.9, 3.3, 3.6, 4.1, 4.6];
    let active = [0.9, 1.4, 1.8, 2.1, 2.6, 2.8, 3.1, 3.4, 3.9, 4.3, 4.4, 4.8];
    let mut outcome = control.to_vec();
    outcome.extend(active);
    let mut treatment = vec![0.0; control.len()];
    treatment.extend(vec![1.0; active.len()]);
    let data =
        TabularData::from_f64_columns([("t", treatment.as_slice()), ("y", outcome.as_slice())])
            .unwrap();
    let concentration = 0.5;
    let thresholds = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let mut query =
        AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    query.outcome_functional = OutcomeFunctional::quantile(0.5);
    let mut dag = Dag::with_variables(2);
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let model = DesignPosteriorModel::QuantileGrid {
        thresholds: Arc::from(thresholds),
        bin_concentration: concentration,
    };
    let prepared = Study::tabular(data.clone())
        .graph(dag)
        .query(CausalQuery::AverageEffect(query))
        .estimator(EstimatorId::BayesianRandomizedQuantile)
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(40_000).design_model(model),
        ))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .prepare(&ExecutionContext::for_tests(9))
        .unwrap();
    let (result, consumed) =
        replay(&prepared, &|p, ctx| p.estimate(&data, ctx).unwrap(), "bayesian-quantile");
    assert!(consumed.body.posterior_row.is_some());
    let row = result.posterior_row().unwrap();
    let route_var = row.covariance.as_ref().unwrap()[8];

    // Independent simulation: Dirichlet bin masses, cumulative at the grid, piecewise-linear
    // inversion of the median.
    let bins = |sample: &[f64]| -> Vec<f64> {
        let mut counts = vec![0.0; thresholds.len() + 1];
        for v in sample {
            let k = thresholds.iter().position(|t| v <= t).unwrap_or(thresholds.len());
            counts[k] += 1.0;
        }
        counts
    };
    let (control_counts, active_counts) = (bins(&control), bins(&active));
    let mut rng = Lcg(77);
    let mut median = |counts: &[f64]| -> Option<f64> {
        let masses: Vec<f64> = counts.iter().map(|c| rng.gamma(c + concentration)).collect();
        let total: f64 = masses.iter().sum();
        let mut cumulative = 0.0;
        let mut previous = 0.0;
        for (k, mass) in masses.iter().enumerate().take(thresholds.len()) {
            cumulative += mass / total;
            if cumulative >= 0.5 {
                if k == 0 {
                    return None;
                }
                return Some(thresholds[k - 1] + (0.5 - previous) / (cumulative - previous));
            }
            previous = cumulative;
        }
        None
    };
    let draws = 40_000;
    let mut differences = Vec::with_capacity(draws);
    while differences.len() < draws {
        if let (Some(a), Some(c)) = (median(&active_counts), median(&control_counts)) {
            differences.push(a - c);
        }
    }
    let n = differences.len() as f64;
    let mean = differences.iter().sum::<f64>() / n;
    let var = differences.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let standard_error = (2.0 * var / draws as f64).sqrt();
    close(row.mean[2], mean, 6.0 * standard_error, "difference of medians");
    assert!((route_var / var - 1.0).abs() < 0.06, "variance {route_var} vs simulated {var}");
}

// ---------------------------------------------------------------------------------------------
// Bayesian switchback effect
// ---------------------------------------------------------------------------------------------

#[test]
fn the_switchback_posterior_replays_and_matches_the_scipy_student_t_reference() {
    const ASSIGNMENT: [[bool; 4]; 4] = [
        [true, false, true, false],
        [false, true, false, true],
        [true, true, false, false],
        [false, false, true, true],
    ];
    const OUTCOME: [[f64; 4]; 4] =
        [[2.3, 1.1, 2.9, 0.8], [0.7, 2.2, 1.4, 2.6], [2.8, 3.1, 0.9, 1.3], [1.2, 0.6, 2.4, 3.0]];
    let (mut y, mut a, mut sequences, mut periods) = (vec![], vec![], vec![], vec![]);
    for s in 0..4 {
        for t in 0..4 {
            y.push(OUTCOME[s][t]);
            a.push(ASSIGNMENT[s][t]);
            sequences.push(Arc::<str>::from(format!("s{s}")));
            periods.push(Arc::<str>::from(format!("p{t}")));
        }
    }
    let units: Vec<Arc<str>> = (0..16).map(|i| Arc::<str>::from(format!("row-{i}"))).collect();
    let query = RandomizedEffectQuery::with_design(
        RandomizationDesign::Switchback { periods: periods.into() },
        VariableId::from_raw(0),
        a,
        vec![0.5; 16],
        sequences,
        units,
        ("off", "on"),
    );
    let data = TabularData::from_f64_columns([("outcome", y.as_slice())]).unwrap();
    let model = DesignPosteriorModel::Switchback {
        model: SwitchbackModel {
            carryover: CarryoverDeclaration::None,
            dependence: SequenceDependence::CompoundSymmetric { rho: 0.4 },
            periods: PeriodAdjustment::None,
            prior: SwitchbackPrior {
                effect_precision: 0.5,
                nuisance_precision: 0.01,
                variance_shape: 2.0,
                variance_scale: 1.0,
            },
        },
    };
    let prepared = Study::tabular(data.clone())
        .query(query)
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(4000).design_model(model),
        ))
        .build()
        .unwrap()
        .prepare(&ExecutionContext::for_tests(0xB1A5))
        .unwrap();
    let (result, _consumed) =
        replay(&prepared, &|p, ctx| p.estimate(&data, ctx).unwrap(), "bayesian-switchback");
    assert_eq!(result.logical_plan.estimator.as_deref(), Some("bayesian.switchback_effect"));
    let summaries = &result.posterior.as_ref().expect("posterior").summaries;
    // Student-t with 20 degrees of freedom: location, scale and quantiles from scipy.stats.t.
    close(summaries.mean[0], 1.547_248_493, 5e-3, "mean");
    close(summaries.sd[0], 0.204_015_378_4, 1e-2, "sd");
    close(summaries.q025[0], 1.143_518_649, 1e-2, "2.5%");
    close(summaries.q975[0], 1.950_978_337, 1e-2, "97.5%");
}
