//! Known-truth check of `bayesian.temporal.glm_link_contrast` on every fixed `TemporalDag` panel
//! coordinate it can estimate: pulse and one-step sustained effects, explicit and accepted
//! structure, and every validation suite.
//!
//! The panel is a two-group design with a balanced +/-1 covariate `c` (split equally within each
//! group and within each group's responders, so the mode has a zero `c` coefficient and the
//! Hessian is block diagonal): `logit P(y = 1) = b0 + b1 x + bc c` over 100 untreated subjects
//! with 30 responders and 100 treated subjects with 60 responders. Under the stated independent
//! `N(0, 1)` prior on `(b0, b1)` the Laplace posterior is `N(mode, H^-1)`, with the mode and the
//! Hessian `H` solved here by Newton's method on the closed-form log posterior. The posterior of
//! the probability-scale contrast `sigmoid(b0 + b1) - sigmoid(b0)` is therefore the push-forward
//! of that Gaussian. Its mean is the average of a reference sample of 400000 draws taken here
//! from the Cholesky factor of `H^-1`; the route publishes the mean of its own 8000 draws, so
//! the stated Monte Carlo bound is six posterior standard errors of the route's mean plus five
//! of the reference mean.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    reason = "deterministic fixture counts are small and the 2x2 algebra uses textbook names"
)]

use std::sync::Arc;

use antecedent::{AcceptedGraph, BayesianConfig, EstimatorId, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ExecutionContext, Lag, TemporalEffectQuery, TemporalPolicy, VariableId,
};
use antecedent_data::{PanelData, PanelUnit, TimeSeriesData};
use antecedent_graph::{TemporalDag, ensure_lagged};
use antecedent_prob::BayesLikelihood;

const GROUP: usize = 100;
const UNTREATED_RESPONDERS: usize = 30;
const TREATED_RESPONDERS: usize = 60;
const STEPS: usize = 3;
const PRIOR_SD: f64 = 1.0;
const DRAWS: usize = 8000;
const REFERENCE_DRAWS: usize = 400_000;
const WIRE_ID: &str = "bayesian.temporal.glm_link_contrast";

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

/// Each unit: `x = [x0, 0, 0]`, `y = [0, y1, 0]`; the decision row is the second one.
fn panel() -> PanelData {
    let mut members = Vec::new();
    for (treated, responders) in [(0.0, UNTREATED_RESPONDERS), (1.0, TREATED_RESPONDERS)] {
        for index in 0..GROUP {
            let mut x = vec![0.0; STEPS];
            let mut c = vec![0.0; STEPS];
            let mut y = vec![0.0; STEPS];
            x[0] = treated;
            c[0] = if index % 2 == 0 { 1.0 } else { -1.0 };
            y[1] = f64::from(u8::from(index < responders));
            let series = TimeSeriesData::from_f64_columns(
                [("x", x.as_slice()), ("c", c.as_slice()), ("y", y.as_slice())],
                1,
            )
            .unwrap();
            members.push(PanelUnit { unit_id: u32::try_from(members.len()).unwrap(), series });
        }
    }
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

fn query(sustained: bool) -> CausalQuery {
    let base = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(2), 1.0)
        .with_horizon_steps(1)
        .with_max_history_lag(Some(1));
    CausalQuery::TemporalEffect(if sustained {
        base.with_policy(TemporalPolicy::sustained(-1, -1))
    } else {
        base.with_policy(TemporalPolicy::pulse(-1))
    })
}

/// Newton mode and Hessian of the logistic log posterior under the `N(0, PRIOR_SD^2)` prior.
fn mode_and_hessian() -> ([f64; 2], [[f64; 2]; 2]) {
    let n = GROUP as f64;
    let (s0, s1) = (UNTREATED_RESPONDERS as f64, TREATED_RESPONDERS as f64);
    let prec = 1.0 / (PRIOR_SD * PRIOR_SD);
    let hessian = |b: &[f64; 2]| {
        let (p0, p1) = (sigmoid(b[0]), sigmoid(b[0] + b[1]));
        let (w0, w1) = (p0 * (1.0 - p0), p1 * (1.0 - p1));
        [[n * w0 + n * w1 + prec, n * w1], [n * w1, n * w1 + prec]]
    };
    let mut b = [0.0_f64; 2];
    for _ in 0..100 {
        let (p0, p1) = (sigmoid(b[0]), sigmoid(b[0] + b[1]));
        let g = [(s0 - n * p0) + (s1 - n * p1) - prec * b[0], (s1 - n * p1) - prec * b[1]];
        let h = hessian(&b);
        let det = h[0][0] * h[1][1] - h[0][1] * h[1][0];
        b[0] += (h[1][1] * g[0] - h[0][1] * g[1]) / det;
        b[1] += (h[0][0] * g[1] - h[1][0] * g[0]) / det;
    }
    (b, hessian(&b))
}

/// Mean and standard deviation of the contrast under `N(mode, H^-1)` from a long reference
/// sample.
fn reference_posterior() -> (f64, f64) {
    let (mode, h) = mode_and_hessian();
    let det = h[0][0] * h[1][1] - h[0][1] * h[1][0];
    let cov = [[h[1][1] / det, -h[0][1] / det], [-h[1][0] / det, h[0][0] / det]];
    let l00 = cov[0][0].sqrt();
    let l10 = cov[1][0] / l00;
    let l11 = (cov[1][1] - l10 * l10).sqrt();
    // The covariate is balanced within each group and within responders, so its score at zero
    // vanishes, the mode has a zero coefficient, and its Hessian block is decoupled from the
    // others (the cross terms are sums of `w c` over balanced halves).
    let n = GROUP as f64;
    let (p0, p1) = (sigmoid(mode[0]), sigmoid(mode[0] + mode[1]));
    let c_sd =
        (n * p0 * (1.0 - p0) + n * p1 * (1.0 - p1) + 1.0 / (PRIOR_SD * PRIOR_SD)).recip().sqrt();
    let mut rng = Lcg(7);
    let (mut sum, mut sum_sq) = (0.0, 0.0);
    for _ in 0..REFERENCE_DRAWS {
        let (z0, z1) = (rng.normal(), rng.normal());
        let b0 = mode[0] + l00 * z0;
        let b1 = mode[1] + l10 * z0 + l11 * z1;
        let bc = c_sd * rng.normal();
        // The covariate is +1 on half the rows and -1 on the other half.
        let contrast = 0.5
            * ((sigmoid(b0 + b1 + bc) - sigmoid(b0 + bc))
                + (sigmoid(b0 + b1 - bc) - sigmoid(b0 - bc)));
        sum += contrast;
        sum_sq += contrast * contrast;
    }
    let n = REFERENCE_DRAWS as f64;
    let mean = sum / n;
    (mean, (sum_sq / n - mean * mean).sqrt())
}

/// Pulse and one-step sustained, explicit and accepted structure, every validation suite on a
/// fixed `TemporalDag` panel: the explicitly selected Bayesian link contrast publishes the
/// posterior of the probability-scale contrast of the saturated two-group Laplace model.
#[test]
fn bayesian_glm_link_contrast_matches_the_laplace_posterior_on_every_fixed_dag_coordinate() {
    let (reference_mean, reference_sd) = reference_posterior();
    // The N(0, 1) prior shrinks the logit coefficients, so the contrast at the mode lies below
    // the observed proportions' difference 0.3 and above the half-way point 0.2.
    let (mode, _) = mode_and_hessian();
    let at_mode = sigmoid(mode[0] + mode[1]) - sigmoid(mode[0]);
    assert!(at_mode > 0.2 && at_mode < 0.3, "{at_mode}");
    let mean_tolerance = 6.0 * reference_sd / (DRAWS as f64).sqrt()
        + 5.0 * reference_sd / (REFERENCE_DRAWS as f64).sqrt();
    let ctx = ExecutionContext::for_tests(4);
    for accepted in [false, true] {
        for sustained in [false, true] {
            for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
                let label = format!("accepted={accepted} sustained={sustained} {suite:?}");
                let base = Study::panel(panel());
                let base = if accepted {
                    base.graph(AcceptedGraph::temporal_dag(graph()))
                } else {
                    base.graph(graph())
                };
                let config = BayesianConfig::laplace()
                    .likelihood(BayesLikelihood::BernoulliLogit)
                    .prior_scale(PRIOR_SD)
                    .n_draws(DRAWS);
                let builder = base
                    .query(query(sustained))
                    .estimator(EstimatorId::BayesianTemporalGlmLinkContrast)
                    .inference(InferenceMode::Bayesian(config))
                    .refute(suite)
                    .bootstrap_replicates(0);
                let study = builder.clone().build().unwrap_or_else(|e| panic!("{label}: {e}"));
                drop(builder);
                let prepared = study.prepare(&ctx).unwrap_or_else(|e| panic!("{label}: {e}"));
                drop(study);
                let result = prepared
                    .estimate_panel(&panel(), &ctx)
                    .unwrap_or_else(|e| panic!("{label}: {e}"));
                assert_eq!(result.logical_plan.estimator.as_deref(), Some(WIRE_ID), "{label}");
                let posterior = result.posterior.as_ref().expect("a Bayesian study keeps draws");
                let column = posterior.effect_column().unwrap();
                let mean = posterior.summaries.mean[column];
                let sd = posterior.summaries.sd[column];
                assert!(
                    (mean - reference_mean).abs() < mean_tolerance,
                    "{label}: mean {mean} vs reference {reference_mean} (+/- {mean_tolerance})"
                );
                assert!(
                    (sd / reference_sd - 1.0).abs() < 0.1,
                    "{label}: sd {sd} vs reference {reference_sd}"
                );
                assert_eq!(posterior.draws.n_draws, DRAWS, "{label}");
            }
        }
    }
}
