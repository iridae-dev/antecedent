//! Known-truth check of `bayesian.temporal.white_innovation_linear` on every fixed `TemporalDag`
//! coordinate it can estimate: pulse and one-step sustained effects, explicit and accepted
//! structure, and every validation suite.
//!
//! A declared white-innovation premise selects this estimator: the dispatch guard of the
//! temporal path refuses a bare `.estimator(..)` naming it, so the premise is the one way to
//! reach it, and each run asserts that the compiled plan names it. Under the premise the
//! posterior is the untempered full-likelihood posterior of one lag-aligned Gaussian regression
//! `y_t = a + b x_{t-1} + e_t`. With the stated diffuse prior (scale 1000) its centre is the
//! least-squares coefficient and its scale the least-squares standard error (a Student-t with
//! 397 degrees of freedom, whose standard deviation is that standard error times 1.0025), both
//! recomputed here in closed form from the raw series. The route publishes the mean of 20000
//! draws, so the stated Monte Carlo bound is five standard errors of that mean plus the
//! ridge shrinkage `|b| / (1e6 Sxx)` of the prior.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::cast_precision_loss,
    reason = "deterministic fixture indices are small and exact in f64"
)]

use antecedent::estimate::InnovationWhitenessGate;
use antecedent::{AcceptedGraph, BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ExecutionContext, Lag, TemporalEffectQuery, TemporalPolicy, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_graph::{TemporalDag, ensure_lagged};

const N: usize = 400;
const DRAWS: usize = 20_000;
const WIRE_ID: &str = "bayesian.temporal.white_innovation_linear";

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
}

/// `x_t = 0.5 x_{t-1} + noise`, `y_t = 0.8 x_{t-1} + e_t` with `e` iid normal.
fn series() -> (Vec<f64>, Vec<f64>) {
    let mut rng = Lcg(2);
    let (mut x, mut y) = (vec![0.0; N], vec![0.0; N]);
    let mut previous = 0.0_f64;
    for t in 0..N {
        previous = 0.5 * previous + rng.normal();
        x[t] = previous;
        y[t] = if t > 0 { 0.8 * x[t - 1] } else { 0.0 } + rng.normal();
    }
    (x, y)
}

fn data(x: &[f64], y: &[f64]) -> TimeSeriesData {
    TimeSeriesData::from_f64_columns([("x", x), ("y", y)], 1).unwrap()
}

/// `x_{t-1} -> y_t`.
fn graph() -> TemporalDag {
    let mut g = TemporalDag::empty();
    let x1 = ensure_lagged(&mut g, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut g, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_directed(x1, y0).unwrap();
    g
}

fn query(sustained: bool) -> CausalQuery {
    let base = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_horizon_steps(1);
    CausalQuery::TemporalEffect(if sustained {
        base.with_policy(TemporalPolicy::sustained(-1, -1))
    } else {
        base.with_policy(TemporalPolicy::pulse(-1))
    })
}

/// Least-squares coefficient on `x_{t-1}`, its standard error and `Sxx` for rows `t = 1..N`.
fn ols(x: &[f64], y: &[f64]) -> (f64, f64, f64) {
    let n = (N - 1) as f64;
    let (xs, ys) = (&x[..N - 1], &y[1..]);
    let (xm, ym) = (xs.iter().sum::<f64>() / n, ys.iter().sum::<f64>() / n);
    let sxx: f64 = xs.iter().map(|v| (v - xm).powi(2)).sum();
    let sxy: f64 = xs.iter().zip(ys).map(|(a, b)| (a - xm) * (b - ym)).sum();
    let beta = sxy / sxx;
    let rss: f64 = xs.iter().zip(ys).map(|(a, b)| (b - ym - beta * (a - xm)).powi(2)).sum();
    (beta, (rss / (n - 2.0) / sxx).sqrt(), sxx)
}

/// Pulse and one-step sustained, explicit and accepted structure, every validation suite on a
/// fixed `TemporalDag`: the declared white-innovation premise publishes the untempered
/// least-squares posterior of the lag-aligned regression.
#[test]
fn white_innovation_posterior_is_the_least_squares_posterior_on_every_fixed_dag_coordinate() {
    let (x, y) = series();
    let (beta, se, sxx) = ols(&x, &y);
    let mean_tolerance = 5.0 * se / (DRAWS as f64).sqrt() + beta.abs() / (1.0e6 * sxx) + 1e-9;
    let expected_sd = se * (397.0_f64 / 395.0).sqrt();
    let ctx = ExecutionContext::for_tests(11);
    for accepted in [false, true] {
        for sustained in [false, true] {
            for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
                let label = format!("accepted={accepted} sustained={sustained} {suite:?}");
                let base = Study::series(data(&x, &y));
                let base = if accepted {
                    base.graph(AcceptedGraph::temporal_dag(graph()))
                } else {
                    base.graph(graph())
                };
                let config = BayesianConfig::conjugate()
                    .n_draws(DRAWS)
                    .prior_scale(1.0e3)
                    .white_innovation(InnovationWhitenessGate { max_lag: 6, level: 0.01 });
                let study = base
                    .query(query(sustained))
                    .inference(InferenceMode::Bayesian(config))
                    .refute(suite)
                    .bootstrap_replicates(0)
                    .build()
                    .unwrap_or_else(|e| panic!("{label}: {e}"));
                let prepared = study.prepare(&ctx).unwrap_or_else(|e| panic!("{label}: {e}"));
                drop(study);
                let result = prepared
                    .estimate_series(&data(&x, &y), &ctx)
                    .unwrap_or_else(|e| panic!("{label}: {e}"));
                assert_eq!(result.logical_plan.estimator.as_deref(), Some(WIRE_ID), "{label}");
                let posterior = result.posterior.as_ref().expect("a Bayesian study keeps draws");
                let column = posterior.effect_column().unwrap();
                let mean = posterior.summaries.mean[column];
                let sd = posterior.summaries.sd[column];
                assert!(
                    (mean - beta).abs() < mean_tolerance,
                    "{label}: mean {mean} vs least squares {beta} (+/- {mean_tolerance})"
                );
                assert!(
                    (sd / expected_sd - 1.0).abs() < 0.05,
                    "{label}: sd {sd} vs closed form {expected_sd}"
                );
                assert!(
                    posterior.diagnostics.notes.iter().any(|n| n.starts_with("white_innovation")),
                    "{label}: the residual whiteness test must be recorded"
                );
                assert_eq!(posterior.draws.n_draws, DRAWS, "{label}");
            }
        }
    }
}
