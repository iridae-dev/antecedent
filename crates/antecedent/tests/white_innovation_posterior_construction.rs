//! Independent construction check of the untempered white-innovation posterior.
//!
//! Under the declared premise the lag-aligned rows `y_t = b0 + b1 x_{t-1} + e_t`, `e_t ~ N(0, s2)`,
//! are independent Gaussian terms, so with the isotropic coefficient prior `b | s2 ~ N(0, s2 c0^2 I)`
//! and the weakly informative `s2 ~ Inv-Gamma(1e-3, 1e-3)` the posterior is the exact
//! Normal-Inverse-Gamma one:
//!
//! ```text
//! P  = X'X + I / c0^2,   Vn = P^-1,   mn = Vn X'y,
//! an = a0 + n / 2,       bn = b0 + (y'y - mn' X'y) / 2,
//! ```
//!
//! and the marginal of `b1` is Student-t with `2 an` degrees of freedom, location `mn[1]` and
//! `Var(b1) = bn / (an - 1) * Vn[1][1]`. The reference below forms these from the raw series with
//! explicit 2 x 2 algebra; the prior scale is small enough (`c0 = 0.1`) that the posterior is visibly
//! shrunk away from least squares, so a flat or tempered construction cannot pass.
//!
//! The route publishes `DRAWS` independent exact draws, so the published summaries are Monte Carlo
//! estimates. Tolerances are Monte Carlo standard errors, not loosened constants: the mean is within
//! `5 sd / sqrt(DRAWS)` (the standard error of a mean of independent draws is `sd / sqrt(DRAWS)`)
//! and the standard deviation within `4 / sqrt(DRAWS)` relative (the relative standard error of a
//! near-Gaussian sample standard deviation is `1 / sqrt(2 DRAWS)`, so the bound is 5.6 standard
//! errors). The draws are seeded, so the outcome is deterministic.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    reason = "integration test builds deterministic fixtures from small counts"
)]

use antecedent::estimate::InnovationWhitenessGate;
use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    Assumption, CausalQuery, ExecutionContext, Lag, TemporalEffectQuery, TemporalPolicy, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_graph::{TemporalDag, ensure_lagged};

const N: usize = 400;
const DRAWS: usize = 200_000;
/// Isotropic coefficient prior scale: `V0 = PRIOR_SCALE^2 I`.
const PRIOR_SCALE: f64 = 0.1;
/// Inverse-Gamma shape and scale of the weakly informative residual-variance prior.
const A0: f64 = 1e-3;
const B0: f64 = 1e-3;
const ASSUMPTION: &str = "bayes.temporal.white_innovation_likelihood";

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

/// `x_t = 0.5 x_{t-1} + noise`, `y_t = 0.8 x_{t-1} + e_t` with `e` white.
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

fn lag_one_graph() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let x1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(x1, y0).unwrap();
    graph
}

/// Exact Normal-Inverse-Gamma posterior of the slope on `x_{t-1}` for rows `t = 1..N`, with the
/// least-squares slope of the same rows: `(mean, sd, least_squares)`.
fn exact_slope_posterior(x: &[f64], y: &[f64]) -> (f64, f64, f64) {
    let n = (N - 1) as f64;
    let (xs, ys) = (&x[..N - 1], &y[1..]);
    let sx: f64 = xs.iter().sum();
    let sxx: f64 = xs.iter().map(|v| v * v).sum();
    let sy: f64 = ys.iter().sum();
    let sxy: f64 = xs.iter().zip(ys).map(|(a, b)| a * b).sum();
    let syy: f64 = ys.iter().map(|v| v * v).sum();
    // P = X'X + I / c0^2 and its inverse.
    let ridge = 1.0 / (PRIOR_SCALE * PRIOR_SCALE);
    let (p00, p01, p11) = (n + ridge, sx, sxx + ridge);
    let det = p00 * p11 - p01 * p01;
    let (v00, v01, v11) = (p11 / det, -p01 / det, p00 / det);
    let m0 = v00 * sy + v01 * sxy;
    let m1 = v01 * sy + v11 * sxy;
    let a_n = A0 + n / 2.0;
    let b_n = B0 + (syy - (m0 * sy + m1 * sxy)) / 2.0;
    let variance = b_n / (a_n - 1.0) * v11;
    // Unpenalised slope from X'X b = X'y.
    let det_ols = n * sxx - sx * sx;
    let ols = (n * sxy - sx * sy) / det_ols;
    (m1, variance.sqrt(), ols)
}

#[test]
fn the_untempered_posterior_is_the_exact_normal_inverse_gamma_posterior_of_the_lag_window() {
    let (x, y) = series();
    let (ref_mean, ref_sd, ols) = exact_slope_posterior(&x, &y);

    let data =
        TimeSeriesData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())], 1).unwrap();
    let query = CausalQuery::TemporalEffect(
        TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
            .with_policy(TemporalPolicy::pulse(-1))
            .with_horizon_steps(1),
    );
    let config = BayesianConfig::conjugate()
        .n_draws(DRAWS)
        .prior_scale(PRIOR_SCALE)
        .white_innovation(InnovationWhitenessGate { max_lag: 6, level: 0.01 });
    let result = Study::series(data)
        .graph(lag_one_graph())
        .query(query)
        .inference(InferenceMode::Bayesian(config))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(11))
        .unwrap();
    let posterior = result.posterior.as_ref().expect("a Bayesian study retains its posterior");
    assert!(posterior.assumptions.entries.iter().any(|record| matches!(
        &record.assumption,
        Assumption::ParametricRestriction(p) if p.id.as_ref() == ASSUMPTION
    )));
    let column = posterior.effect_column().expect("effect column");
    let (mean, sd) = (posterior.summaries.mean[column], posterior.summaries.sd[column]);
    let root = (posterior.draws.n_draws as f64).sqrt();
    assert_eq!(posterior.draws.n_draws, DRAWS);

    // The prior is informative enough that the exact posterior is far from least squares in
    // Monte Carlo units, so neither a flat-prior nor a tempered fit can match the reference.
    assert!((ref_mean - ols).abs() > 20.0 * ref_sd / root, "shrinkage {ref_mean} vs {ols}");
    assert!((mean - ref_mean).abs() < 5.0 * ref_sd / root, "mean {mean} vs exact {ref_mean}");
    assert!((sd / ref_sd - 1.0).abs() < 4.0 / root, "sd {sd} vs exact {ref_sd}");
}
