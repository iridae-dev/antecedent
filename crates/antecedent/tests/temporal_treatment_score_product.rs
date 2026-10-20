//! The temporal treatment-propensity nuisance product of a prepared panel study.
//!
//! Each unit contributes its earliest row with a full lag window, so rows are independent
//! subjects; the treatment of that row is modelled on the identified adjustment lag. References
//! are computed here from the raw first rows: the Newton ridge mode with the intercept
//! unpenalized, the Newton mode and Laplace sample under independent Gaussian priors, and the
//! scores at stated lag histories. A single dependent series is refused with its next attempt, and
//! the retained artifact is refit on load and refused when forged.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    reason = "integration test solves tiny dense systems by index from deterministic fixtures"
)]

use std::sync::Arc;

use antecedent::analysis::array_fits::{
    ArrayFit, ArrayFitArtifactWire, ArrayFitConsumeLimits, ArrayFitError, ArrayFitRequestWire,
    TemporalScoreModel, TemporalTreatmentScoreSpec,
};
use antecedent::{RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ExecutionContext, Lag, TemporalEffectQuery, TemporalPolicy, VariableId,
};
use antecedent_data::{PanelData, PanelUnit, TimeSeriesData};
use antecedent_graph::{TemporalDag, ensure_lagged};

const UNITS: usize = 300;
const STEPS: usize = 3;

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

/// Per unit `STEPS` rows of `(x, c, y)`: `c ~ N(0, 1)`, `x ~ Bernoulli(sigmoid(0.5 + 0.9 c))`,
/// `y` Bernoulli.
fn units() -> Vec<[Vec<f64>; 3]> {
    let mut rng = Lcg(77);
    (0..UNITS)
        .map(|_| {
            let (mut x, mut c, mut y) = (vec![], vec![], vec![]);
            for t in 0..STEPS {
                let ct = rng.normal();
                x.push(f64::from(rng.uniform() < sigmoid(0.5 + 0.9 * ct)));
                c.push(ct);
                y.push(if t == 0 {
                    0.0
                } else {
                    f64::from(rng.uniform() < sigmoid(-0.5 + x[t - 1] + 0.8 * c[t - 1]))
                });
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

fn query() -> CausalQuery {
    CausalQuery::TemporalEffect(
        TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(2), 1.0)
            .with_policy(TemporalPolicy::pulse(-1))
            .with_horizon_steps(1)
            .with_max_history_lag(Some(1)),
    )
}

fn prepared_panel(units: &[[Vec<f64>; 3]]) -> antecedent::PreparedStudy {
    let members: Vec<PanelUnit> = units
        .iter()
        .enumerate()
        .map(|(i, unit)| PanelUnit { unit_id: u32::try_from(i).unwrap(), series: series(unit) })
        .collect();
    Study::panel(PanelData::try_new(Arc::from(members)).unwrap())
        .graph(graph())
        .query(query())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ExecutionContext::for_tests(1))
        .unwrap()
}

/// First rows: `(1, c_0)` and the treatment `x_0`.
fn first_rows(units: &[[Vec<f64>; 3]]) -> (Vec<[f64; 2]>, Vec<f64>) {
    (units.iter().map(|u| [1.0, u[1][0]]).collect(), units.iter().map(|u| u[0][0]).collect())
}

fn solve2(a: [[f64; 2]; 2], b: [f64; 2]) -> [f64; 2] {
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    [(b[0] * a[1][1] - a[0][1] * b[1]) / det, (a[0][0] * b[1] - b[0] * a[1][0]) / det]
}

/// Newton mode and Hessian of a logistic model with a diagonal prior precision.
fn newton(rows: &[[f64; 2]], y: &[f64], prec: [f64; 2]) -> ([f64; 2], [[f64; 2]; 2]) {
    let hessian = |beta: &[f64; 2]| {
        let mut h = [[0.0; 2]; 2];
        for row in rows {
            let p = sigmoid(row[0] * beta[0] + row[1] * beta[1]);
            for i in 0..2 {
                for j in 0..2 {
                    h[i][j] += p * (1.0 - p) * row[i] * row[j];
                }
            }
        }
        h[0][0] += prec[0];
        h[1][1] += prec[1];
        h
    };
    let mut beta = [0.0; 2];
    for _ in 0..100 {
        let mut g = [0.0; 2];
        for (row, &yi) in rows.iter().zip(y) {
            let p = sigmoid(row[0] * beta[0] + row[1] * beta[1]);
            for k in 0..2 {
                g[k] += row[k] * (yi - p);
            }
        }
        for k in 0..2 {
            g[k] -= prec[k] * beta[k];
        }
        let step = solve2(hessian(&beta), g);
        beta = [beta[0] + step[0], beta[1] + step[1]];
    }
    (beta, hessian(&beta))
}

fn spec(model: TemporalScoreModel) -> TemporalTreatmentScoreSpec {
    TemporalTreatmentScoreSpec { model, histories: vec![vec![0.0], vec![1.5]] }
}

#[test]
fn the_penalized_score_is_the_ridge_mode_with_an_unpenalized_intercept() {
    let units = units();
    let (rows, y) = first_rows(&units);
    let prepared = prepared_panel(&units);
    let penalty = 2.0;
    let product = prepared
        .temporal_treatment_score(&spec(TemporalScoreModel::Penalized { penalty }))
        .unwrap();
    let (mode, _) = newton(&rows, &y, [0.0, penalty]);
    let value = |name: &str| product.value(name).unwrap();
    assert!((value("coefficient.0") - mode[0]).abs() < 1e-6);
    assert!((value("coefficient.1") - mode[1]).abs() < 1e-6);
    assert!((value("score0.at_mode") - sigmoid(mode[0])).abs() < 1e-6);
    assert!((value("score1.at_mode") - sigmoid(mode[0] + 1.5 * mode[1])).abs() < 1e-6);
    assert!(product.draws().is_empty());
    assert_eq!(product.result().uncertainty, "penalized_mode_no_uncertainty");
    assert_eq!(product.result().licensing, "unlicensed");
    assert_eq!(product.result().calibration, "unmeasured");
    assert_eq!(product.estimator_id(), "temporal.penalized_propensity");
}

#[test]
fn the_posterior_score_is_a_laplace_mean_not_the_mode_probability() {
    let units = units();
    let (rows, y) = first_rows(&units);
    let prepared = prepared_panel(&units);
    let (sd_i, sd_s) = (1.0, 0.5);
    let product = prepared
        .temporal_treatment_score(&spec(TemporalScoreModel::Posterior {
            intercept_sd: sd_i,
            slope_sd: sd_s,
            draws: 4000,
            seed: 3,
        }))
        .unwrap();
    let (mode, hessian) = newton(&rows, &y, [1.0 / (sd_i * sd_i), 1.0 / (sd_s * sd_s)]);
    let cov = {
        let det = hessian[0][0] * hessian[1][1] - hessian[0][1] * hessian[1][0];
        [[hessian[1][1] / det, -hessian[0][1] / det], [-hessian[1][0] / det, hessian[0][0] / det]]
    };
    let l00 = cov[0][0].sqrt();
    let l10 = cov[1][0] / l00;
    let l11 = (cov[1][1] - l10 * l10).sqrt();
    let mut rng = Lcg(5);
    let samples = 200_000;
    let (mut lag0, mut lag1) = (0.0, 0.0);
    for _ in 0..samples {
        let (z0, z1) = (rng.normal(), rng.normal());
        let beta = [mode[0] + l00 * z0, mode[1] + l10 * z0 + l11 * z1];
        lag0 += sigmoid(beta[0]);
        lag1 += sigmoid(beta[0] + 1.5 * beta[1]);
    }
    let value = |name: &str| product.value(name).unwrap();
    assert!((value("score0.at_mode") - sigmoid(mode[0])).abs() < 1e-6);
    assert!((value("score0.mean") - lag0 / f64::from(samples)).abs() < 0.01);
    assert!((value("score1.mean") - lag1 / f64::from(samples)).abs() < 0.01);
    assert!(product.draws().is_empty());
    assert!(product.result().uncertainty.contains("not_an_effect_posterior"));
    assert_eq!(product.estimator_id(), "bayesian.temporal.propensity_posterior");
}

#[test]
fn a_single_dependent_series_is_refused_with_its_next_attempt() {
    let units = units();
    let prepared = Study::series(series(&units[0]))
        .graph(graph())
        .query(query())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ExecutionContext::for_tests(1))
        .unwrap();
    let error = prepared
        .temporal_treatment_score(&spec(TemporalScoreModel::Penalized { penalty: 1.0 }))
        .unwrap_err();
    let text = error.to_string();
    assert!(text.contains("single dependent series"), "{text}");
    assert!(text.contains("Next attempt"), "{text}");
    assert!(text.contains("route_not_supported"), "{text}");
}

#[test]
fn the_retained_artifact_is_refit_on_load_and_refused_when_forged() {
    let units = units();
    let prepared = prepared_panel(&units);
    let product = prepared
        .temporal_treatment_score(&spec(TemporalScoreModel::Penalized { penalty: 2.0 }))
        .unwrap();
    let bytes = product.export().unwrap();
    let limits = ArrayFitConsumeLimits::default();
    assert_eq!(ArrayFit::consume(&bytes, limits).unwrap(), product);

    let mut wire = ArrayFitArtifactWire::decode(&bytes).unwrap();
    wire.result.values[1].value += 0.01;
    assert_eq!(
        ArrayFit::consume(&wire.export().unwrap(), limits).unwrap_err(),
        ArrayFitError::ResultMismatch
    );
    let mut wire = ArrayFitArtifactWire::decode(&bytes).unwrap();
    if let ArrayFitRequestWire::TemporalPropensity(request) = &mut wire.request {
        request.design.treatment[0] = 1.0 - request.design.treatment[0];
    }
    wire.request_digest = wire.expected_request_digest().unwrap();
    assert_eq!(
        ArrayFit::consume(&wire.export().unwrap(), limits).unwrap_err(),
        ArrayFitError::ResultMismatch
    );
}
