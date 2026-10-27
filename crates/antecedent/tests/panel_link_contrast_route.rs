//! The temporal GLM link contrast through the public `Study` flow over a panel.
//!
//! Each unit contributes one decision time (its earliest row with a full lag window), so the rows
//! are independent subjects. The reference is computed here from the raw first rows: the Newton
//! logistic maximum-likelihood fit, the probability-scale contrast averaged over the observed
//! lag, its Fisher delta-method standard error, and for the Bayesian route the Newton mode and a
//! Laplace sample under the stated Gaussian prior. The routes are unlicensed estimators with
//! calibration unmeasured; a single dependent series is refused with its next attempt.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    clippy::many_single_char_names,
    reason = "integration test solves tiny dense systems by index from deterministic fixtures"
)]

use std::sync::Arc;

use antecedent::{BayesianConfig, EstimatorId, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ExecutionContext, Lag, TemporalEffectQuery, TemporalPolicy, VariableId,
};
use antecedent_data::{PanelData, PanelUnit, TimeSeriesData};
use antecedent_graph::{TemporalDag, ensure_lagged};
use antecedent_prob::BayesLikelihood;

const UNITS: usize = 300;
const STEPS: usize = 3;
const PRIOR_SD: f64 = 1.0;
const DRAWS: usize = 4000;

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

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Per unit `STEPS` rows of `(x, c, y)`: `c ~ N(0, 1)`, `x ~ Bernoulli(sigmoid(0.5 c))`,
/// `y_t ~ Bernoulli(sigmoid(-0.5 + x_{t-1} + 0.8 c_{t-1}))`.
fn units() -> Vec<[Vec<f64>; 3]> {
    let mut rng = Lcg(2024);
    (0..UNITS)
        .map(|_| {
            let (mut x, mut c, mut y) = (vec![], vec![], vec![]);
            for t in 0..STEPS {
                let ct = rng.normal();
                let xt = f64::from(rng.uniform() < sigmoid(0.5 * ct));
                let yt = if t == 0 {
                    0.0
                } else {
                    let eta = -0.5 + x[t - 1] + 0.8 * c[t - 1];
                    f64::from(rng.uniform() < sigmoid(eta))
                };
                x.push(xt);
                c.push(ct);
                y.push(yt);
            }
            [x, c, y]
        })
        .collect()
}

fn series(unit: &[Vec<f64>; 3]) -> TimeSeriesData {
    TimeSeriesData::from_f64_columns(
        [("x", unit[0].as_slice()), ("c", unit[1].as_slice()), ("y", unit[2].as_slice())],
        1,
    )
    .unwrap()
}

fn panel(units: &[[Vec<f64>; 3]]) -> PanelData {
    let members: Vec<PanelUnit> = units
        .iter()
        .enumerate()
        .map(|(i, unit)| PanelUnit { unit_id: u32::try_from(i).unwrap(), series: series(unit) })
        .collect();
    PanelData::try_new(Arc::from(members)).unwrap()
}

/// `c_{t-1} -> x_{t-1}`, `c_{t-1} -> y_t`, `x_{t-1} -> y_t`.
fn graph() -> TemporalDag {
    let mut g = TemporalDag::empty();
    let c1 = ensure_lagged(&mut g, VariableId::from_raw(1), Lag::from_raw(1)).unwrap();
    let x1 = ensure_lagged(&mut g, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut g, VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_directed(c1, x1).unwrap();
    g.insert_directed(c1, y0).unwrap();
    g.insert_directed(x1, y0).unwrap();
    g
}

fn query() -> TemporalEffectQuery {
    TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(2), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1)
        .with_max_history_lag(Some(1))
}

/// First aligned row of every unit: `(1, x_0, c_0)` and `y_1`.
fn first_rows(units: &[[Vec<f64>; 3]]) -> (Vec<[f64; 3]>, Vec<f64>) {
    (
        units.iter().map(|u| [1.0, u[0][0], u[1][0]]).collect(),
        units.iter().map(|u| u[2][1]).collect(),
    )
}

fn solve(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> [f64; 3] {
    for col in 0..3 {
        let pivot = (col..3).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs())).unwrap();
        a.swap(col, pivot);
        b.swap(col, pivot);
        for row in col + 1..3 {
            let f = a[row][col] / a[col][col];
            for k in col..3 {
                a[row][k] -= f * a[col][k];
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = [0.0; 3];
    for row in (0..3).rev() {
        let tail: f64 = (row + 1..3).map(|k| a[row][k] * x[k]).sum();
        x[row] = (b[row] - tail) / a[row][row];
    }
    x
}

fn invert(a: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut inv = [[0.0; 3]; 3];
    for k in 0..3 {
        let mut e = [0.0; 3];
        e[k] = 1.0;
        let col = solve(a, e);
        for r in 0..3 {
            inv[r][k] = col[r];
        }
    }
    inv
}

/// Newton mode and information matrix of a logistic model with prior precision `prec * I`.
fn newton(rows: &[[f64; 3]], y: &[f64], prec: f64) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut beta = [0.0; 3];
    let information = |beta: &[f64; 3]| {
        let mut h = [[0.0; 3]; 3];
        for (row, _) in rows.iter().zip(y) {
            let eta: f64 = (0..3).map(|k| row[k] * beta[k]).sum();
            let w = sigmoid(eta) * (1.0 - sigmoid(eta));
            for i in 0..3 {
                for j in 0..3 {
                    h[i][j] += w * row[i] * row[j];
                }
            }
        }
        for i in 0..3 {
            h[i][i] += prec;
        }
        h
    };
    for _ in 0..100 {
        let mut gradient = [0.0; 3];
        for (row, &yi) in rows.iter().zip(y) {
            let eta: f64 = (0..3).map(|k| row[k] * beta[k]).sum();
            for k in 0..3 {
                gradient[k] += row[k] * (yi - sigmoid(eta));
            }
        }
        for k in 0..3 {
            gradient[k] -= prec * beta[k];
        }
        let step = solve(information(&beta), gradient);
        for k in 0..3 {
            beta[k] += step[k];
        }
    }
    (beta, information(&beta))
}

fn contrast(rows: &[[f64; 3]], beta: &[f64; 3]) -> f64 {
    rows.iter()
        .map(|row| {
            let lag = beta[2] * row[2];
            sigmoid(beta[0] + beta[1] + lag) - sigmoid(beta[0] + lag)
        })
        .sum::<f64>()
        / rows.len() as f64
}

fn delta_se(rows: &[[f64; 3]], beta: &[f64; 3], information: &[[f64; 3]; 3]) -> f64 {
    let mut gradient = [0.0; 3];
    for row in rows {
        let lag = beta[2] * row[2];
        let (p1, p0) = (sigmoid(beta[0] + beta[1] + lag), sigmoid(beta[0] + lag));
        let (d1, d0) = (p1 * (1.0 - p1), p0 * (1.0 - p0));
        gradient[0] += d1 - d0;
        gradient[1] += d1;
        gradient[2] += (d1 - d0) * row[2];
    }
    for g in &mut gradient {
        *g /= rows.len() as f64;
    }
    let inv = invert(*information);
    let mut variance = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            variance += gradient[i] * inv[i][j] * gradient[j];
        }
    }
    variance.sqrt()
}

fn builder(units: &[[Vec<f64>; 3]]) -> antecedent::StudyBuilder {
    Study::panel(panel(units))
        .graph(graph())
        .query(CausalQuery::TemporalEffect(query()))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
}

#[test]
fn the_frequentist_contrast_matches_newton_and_the_delta_method() {
    let units = units();
    let (rows, y) = first_rows(&units);
    let (beta, information) = newton(&rows, &y, 0.0);
    let result = builder(&units)
        .estimator(EstimatorId::TemporalGlmLinkContrast)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(4))
        .unwrap();
    let estimate = result.estimate().as_effect().unwrap();
    assert!((estimate.ate - contrast(&rows, &beta)).abs() < 1e-8, "{}", estimate.ate);
    assert!(
        (estimate.se_analytic() - delta_se(&rows, &beta, &information)).abs() < 1e-8,
        "{}",
        estimate.se_analytic()
    );
    // The contrast is a probability difference, not the link-scale treatment coefficient.
    assert!((estimate.ate - beta[1]).abs() > 0.1);
    assert!(result.posterior.is_none());
    assert_ne!(result.support_status(), Some(antecedent::support::CellStatus::Licensed));
}

#[test]
fn the_bayesian_contrast_matches_the_mode_and_a_laplace_sample() {
    let units = units();
    let (rows, y) = first_rows(&units);
    let (mode, hessian) = newton(&rows, &y, 1.0 / (PRIOR_SD * PRIOR_SD));
    let config = BayesianConfig::laplace()
        .likelihood(BayesLikelihood::BernoulliLogit)
        .prior_scale(PRIOR_SD)
        .n_draws(DRAWS);
    let result = builder(&units)
        .estimator(EstimatorId::BayesianTemporalGlmLinkContrast)
        .inference(InferenceMode::Bayesian(config))
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(4))
        .unwrap();
    let posterior = result.posterior.as_ref().expect("a Bayesian study retains its posterior");
    let column = posterior.effect_column().unwrap();
    let mean = posterior.summaries.mean[column];
    // Laplace sample from N(mode, hessian^-1) by a Cholesky factor of the covariance.
    let cov = invert(hessian);
    let mut l = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..=i {
            let s: f64 = (0..j).map(|k| l[i][k] * l[j][k]).sum();
            l[i][j] = if i == j { (cov[i][i] - s).sqrt() } else { (cov[i][j] - s) / l[j][j] };
        }
    }
    let mut rng = Lcg(7);
    let samples = 100_000;
    let mut total = 0.0;
    for _ in 0..samples {
        let z = [rng.normal(), rng.normal(), rng.normal()];
        let mut beta = mode;
        for i in 0..3 {
            for j in 0..=i {
                beta[i] += l[i][j] * z[j];
            }
        }
        total += contrast(&rows, &beta);
    }
    let reference_mean = total / f64::from(samples);
    assert!((mean - reference_mean).abs() < 0.01, "{mean} vs {reference_mean}");
    // The posterior mean is not the contrast at the mode.
    let at_mode = contrast(&rows, &mode);
    assert!((at_mode - reference_mean).abs() < 0.02);
    assert_eq!(posterior.draws.n_draws, DRAWS);
}

#[test]
fn a_single_dependent_series_is_refused_with_its_next_attempt() {
    let units = units();
    let one = series(&units[0]);
    let error = Study::series(one)
        .graph(graph())
        .query(CausalQuery::TemporalEffect(query()))
        .estimator(EstimatorId::TemporalGlmLinkContrast)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap_err();
    let text = error.to_string();
    assert!(text.contains("single dependent series"), "{text}");
    assert!(text.contains("Next attempt"), "{text}");
    assert!(text.contains("route_not_supported"), "{text}");
}

#[test]
fn the_link_estimators_name_their_inference_mode() {
    let units = units();
    let error = builder(&units)
        .estimator(EstimatorId::BayesianTemporalGlmLinkContrast)
        .build()
        .unwrap_err();
    assert!(error.to_string().contains("estimator_inference_mismatch"));
    let error = builder(&units)
        .estimator(EstimatorId::TemporalGlmLinkContrast)
        .inference(InferenceMode::Bayesian(BayesianConfig::laplace()))
        .build()
        .unwrap_err();
    assert!(error.to_string().contains("estimator_inference_mismatch"), "{error}");
}

#[test]
fn both_link_contrast_estimators_publish_artifacts_a_consumer_recomputes_from() {
    let units = units();
    let panel_data = panel(&units);
    let ctx = ExecutionContext::for_tests(4);
    let bayesian = InferenceMode::Bayesian(
        BayesianConfig::laplace()
            .likelihood(BayesLikelihood::BernoulliLogit)
            .prior_scale(PRIOR_SD)
            .n_draws(400),
    );
    let cases = [
        (
            EstimatorId::TemporalGlmLinkContrast,
            InferenceMode::Frequentist,
            "temporal.glm_link_contrast",
        ),
        (
            EstimatorId::BayesianTemporalGlmLinkContrast,
            bayesian,
            "bayesian.temporal.glm_link_contrast",
        ),
    ];
    for (estimator, inference, wire_id) in cases {
        let prepared = builder(&units)
            .estimator(estimator)
            .inference(inference)
            .build()
            .unwrap()
            .prepare(&ctx)
            .unwrap();
        let result = prepared.estimate_panel(&panel_data, &ctx).unwrap();
        let bytes = prepared.encode_contracted_result(&result, wire_id, &ctx).unwrap();
        let consumed = antecedent_io::consume_analysis_result(&bytes).unwrap();
        assert_eq!(consumed.body.estimate, Some(result.effect()), "{wire_id}");
        let contract = consumed.contract.as_ref().expect("a contracted link-contrast artifact");
        assert_eq!(contract.estimator.as_deref(), Some(wire_id));
    }
}
