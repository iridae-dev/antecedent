//! Bayesian unit counterfactual on a linear `TemporalDag`.
//!
//! `Y_t = 1 + 2 X_{t-1} + e_t` with a noise that is recovered from the factual row and replayed
//! in the counterfactual world, so the unit contrast of `do(X = active)` against
//! `do(X = control)` is `slope * (active - control)` for every unit. The Bayesian twin must
//! publish the posterior of that product, not the ridge point. The expected centre and spread
//! come from an ordinary least squares fit computed here by hand; the long-run tempering
//! factor only widens the posterior, so the spread is bracketed by the iid standard error.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    reason = "test scaffolding computes exact least squares on small constant series"
)]

use std::sync::Arc;

use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, CounterfactualQuery, ExecutionContext, Intervention, Lag, Value, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_graph::{TemporalDag, ensure_lagged};

const N: usize = 400;

/// Deterministic uniform noise in `(-1.4, 1.4)` (sd about 0.81), independent across rows.
fn noise(len: usize) -> Vec<f64> {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = (state >> 11) as f64 / (1u64 << 53) as f64;
            (unit - 0.5) * 2.8
        })
        .collect()
}

fn series() -> (Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = noise(N).iter().map(|e| e * 1.1).collect();
    let e = noise(N + 7);
    let mut y = vec![0.0; N];
    for t in 1..N {
        y[t] = 1.0 + 2.0 * x[t - 1] + e[t + 3];
    }
    (x, y)
}

/// Intercept-slope least squares of `y[1..]` on `x[..N-1]`: `(slope, iid standard error)`.
fn ols(x: &[f64], y: &[f64]) -> (f64, f64) {
    let (xs, ys) = (&x[..N - 1], &y[1..]);
    let n = xs.len() as f64;
    let mean_x = xs.iter().sum::<f64>() / n;
    let mean_y = ys.iter().sum::<f64>() / n;
    let sxx: f64 = xs.iter().map(|v| (v - mean_x).powi(2)).sum();
    let sxy: f64 = xs.iter().zip(ys).map(|(a, b)| (a - mean_x) * (b - mean_y)).sum();
    let slope = sxy / sxx;
    let intercept = mean_y - slope * mean_x;
    let rss: f64 = xs.iter().zip(ys).map(|(a, b)| (b - intercept - slope * a).powi(2)).sum();
    (slope, (rss / (n - 2.0) / sxx).sqrt())
}

fn graph() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let x1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(x1, y0).unwrap();
    graph
}

fn run(x: &[f64], y: &[f64], control: f64, active: f64, bayesian: bool) -> antecedent::StudyResult {
    let data = TimeSeriesData::from_f64_columns([("x", x), ("y", y)], 1).unwrap();
    let query = CounterfactualQuery::new(
        VariableId::from_raw(1),
        Arc::from([Intervention::set(VariableId::from_raw(0), Value::f64(active))]),
    )
    .with_control_level(control);
    let mut builder = Study::series(data)
        .graph(graph())
        .query(CausalQuery::Counterfactual(query))
        .refute(RefuteSuite::None);
    if bayesian {
        builder = builder.inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(2_000).prior_scale(1_000.0),
        ));
    }
    let ctx = ExecutionContext::for_tests(2_024);
    builder.build().unwrap().prepare(&ctx).unwrap().estimate_retained(&ctx).unwrap()
}

#[test]
fn the_bayesian_temporal_counterfactual_is_the_posterior_of_the_slope_times_the_contrast() {
    let (x, y) = series();
    let (slope, se) = ols(&x, &y);
    assert!((slope - 2.0).abs() < 5.0 * se, "fixture slope {slope} se {se}");

    let frequentist = run(&x, &y, 0.0, 1.0, false);
    assert!(
        (frequentist.estimate.as_effect().unwrap().ate - slope).abs() < 1e-8,
        "frequentist point is the least squares slope"
    );
    assert!(frequentist.posterior.is_none());

    let bayesian = run(&x, &y, 0.0, 1.0, true);
    assert_eq!(bayesian.logical_plan.estimator.as_deref(), Some("bayesian.temporal.gcomp"));
    let posterior = bayesian.posterior.as_ref().expect("a posterior, not a ridge point");
    let column = posterior.effect_column().expect("effect draws");
    assert_eq!(posterior.draws.n_draws, 2_000);
    let mean = posterior.summaries.mean[column];
    let sd = posterior.summaries.sd[column];
    // Diffuse prior: the centre is the least squares slope; 2000 draws leave a Monte Carlo
    // error of about 0.02 standard errors.
    assert!((mean - slope).abs() < 0.25 * se, "posterior mean {mean} vs slope {slope} (se {se})");
    // The tempered posterior is never narrower than the iid one (up to Monte Carlo error)
    // and, for a serially independent series, stays within a small factor of it.
    assert!(sd > 0.85 * se && sd < 1.8 * se, "posterior sd {sd} vs iid se {se}");
    let mut draws = posterior.draws.column(column).unwrap().to_vec();
    draws.sort_by(f64::total_cmp);
    let (low, high) = (
        draws[(draws.len() as f64 * 0.025) as usize],
        draws[(draws.len() as f64 * 0.975) as usize],
    );
    assert!(low < slope && slope < high, "95% interval [{low}, {high}] around the slope {slope}");
    assert!(
        posterior.assumptions.entries.iter().any(|record| matches!(
            &record.assumption,
            antecedent_core::Assumption::ParametricRestriction(p)
                if p.id.as_ref() == "temporal.counterfactual.shared_additive_noise"
        )),
        "the shared-noise premise is stated on the posterior"
    );
    assert!(
        bayesian.diagnostics.iter().any(|d| d.code.as_ref() == "estimate.temporal.cross_world"
            && d.message.contains("parameter uncertainty only"))
    );
}

#[test]
fn the_posterior_scales_with_the_contrast() {
    let (x, y) = series();
    let unit = run(&x, &y, 0.0, 1.0, true);
    let wide = run(&x, &y, 0.5, 3.5, true);
    let column = |result: &antecedent::StudyResult| {
        let posterior = result.posterior.as_ref().unwrap();
        let column = posterior.effect_column().unwrap();
        (posterior.summaries.mean[column], posterior.summaries.sd[column])
    };
    let ((mean_unit, sd_unit), (mean_wide, sd_wide)) = (column(&unit), column(&wide));
    // The same seed draws the same slope draws; the contrast is the only difference, 3.0.
    assert!((mean_wide - 3.0 * mean_unit).abs() < 1e-9 * mean_unit.abs().max(1.0), "{mean_wide}");
    assert!((sd_wide - 3.0 * sd_unit).abs() < 1e-9 * sd_unit.max(1.0), "{sd_wide}");
}
