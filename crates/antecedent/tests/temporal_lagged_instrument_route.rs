//! The lagged-instrument effect through the public `Study` flow.
//!
//! `z_{t-2} -> x_{t-1} -> y_t` with an unobserved-in-estimation confounder `u` of treatment and
//! outcome: back-door adjustment cannot identify the effect, a lagged cause of the treatment can.
//! The identifier certifies the instrument on the unfolded window graph (not from a time column);
//! the data checks relevance by the first-stage F. The reference is the closed-form Wald ratio of
//! covariances and its homoskedastic standard error on the lag-aligned rows, computed here from
//! the raw series. Both estimators are unlicensed with calibration unmeasured.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    reason = "integration test builds deterministic fixtures from small counts"
)]

use antecedent::{BayesianConfig, EstimatorId, IdentifierId, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ExecutionContext, Lag, TemporalEffectQuery, TemporalPolicy, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_graph::{TemporalDag, ensure_lagged};

const N: usize = 800;
const X: u32 = 0;
const Y: u32 = 1;
const Z: u32 = 2;
const U: u32 = 3;

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

/// `x_s = 0.9 z_{s-1} + 0.8 u_s + e`, `y_t = 2 x_{t-1} + u_{t-1} + e`; `z` and `u` iid.
fn series() -> [Vec<f64>; 4] {
    let mut rng = Lcg(31);
    let z: Vec<f64> = (0..N).map(|_| rng.normal()).collect();
    let u: Vec<f64> = (0..N).map(|_| rng.normal()).collect();
    let mut x = vec![0.0; N];
    let mut y = vec![0.0; N];
    for s in 0..N {
        let lagged = if s > 0 { z[s - 1] } else { 0.0 };
        x[s] = 0.9 * lagged + 0.8 * u[s] + 0.3 * rng.normal();
    }
    for t in 1..N {
        y[t] = 2.0 * x[t - 1] + u[t - 1] + 0.3 * rng.normal();
    }
    [x, y, z, u]
}

fn data(columns: &[Vec<f64>; 4]) -> TimeSeriesData {
    TimeSeriesData::from_f64_columns(
        [
            ("x", columns[0].as_slice()),
            ("y", columns[1].as_slice()),
            ("z", columns[2].as_slice()),
            ("u", columns[3].as_slice()),
        ],
        1,
    )
    .unwrap()
}

/// `z_{t-2} -> x_{t-1} -> y_t`, `u_{t-1} -> x_{t-1}`, `u_{t-1} -> y_t`; optionally the lagged
/// confounder path `u_{t-2} -> z_{t-2}`, `u_{t-2} -> y_t`.
fn graph(lagged_confounder: bool) -> TemporalDag {
    let mut g = TemporalDag::empty();
    let mut node = |variable: u32, lag: u32| {
        ensure_lagged(&mut g, VariableId::from_raw(variable), Lag::from_raw(lag)).unwrap()
    };
    let (z2, x1, y0, u1) = (node(Z, 2), node(X, 1), node(Y, 0), node(U, 1));
    let extra = lagged_confounder.then(|| node(U, 2));
    g.insert_directed(z2, x1).unwrap();
    g.insert_directed(x1, y0).unwrap();
    g.insert_directed(u1, x1).unwrap();
    g.insert_directed(u1, y0).unwrap();
    if let Some(u2) = extra {
        g.insert_directed(u2, z2).unwrap();
        g.insert_directed(u2, y0).unwrap();
    }
    g
}

fn query() -> CausalQuery {
    CausalQuery::TemporalEffect(
        TemporalEffectQuery::pulse(VariableId::from_raw(X), VariableId::from_raw(Y), 1.0)
            .with_policy(TemporalPolicy::pulse(-1))
            .with_horizon_steps(1),
    )
}

fn builder(columns: &[Vec<f64>; 4], lagged_confounder: bool) -> antecedent::StudyBuilder {
    Study::series(data(columns))
        .graph(graph(lagged_confounder))
        .query(query())
        .identifier(IdentifierId::TemporalIvUnfolded)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
}

/// Wald ratio, its homoskedastic standard error and the first-stage F over rows `i = 1..N - 1`
/// pairing `z_{i-1}`, `x_i` and `y_{i+1}`.
fn wald(columns: &[Vec<f64>; 4]) -> (f64, f64, f64) {
    let rows = N - 2;
    let pick =
        |c: &[f64], offset: usize| -> Vec<f64> { (0..rows).map(|k| c[k + offset]).collect() };
    let (zs, xs, ys) = (pick(&columns[2], 0), pick(&columns[0], 1), pick(&columns[1], 2));
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let (zm, xm, ym) = (mean(&zs), mean(&xs), mean(&ys));
    let szz: f64 = zs.iter().map(|z| (z - zm).powi(2)).sum();
    let szx: f64 = zs.iter().zip(&xs).map(|(z, x)| (z - zm) * (x - xm)).sum();
    let szy: f64 = zs.iter().zip(&ys).map(|(z, y)| (z - zm) * (y - ym)).sum();
    let effect = szy / szx;
    let slope = szx / szz;
    let df = (rows - 2) as f64;
    let first_stage_rss: f64 =
        zs.iter().zip(&xs).map(|(z, x)| (x - xm - slope * (z - zm)).powi(2)).sum();
    let structural_rss: f64 =
        xs.iter().zip(&ys).map(|(x, y)| (y - (ym - effect * xm) - effect * x).powi(2)).sum();
    let se = (structural_rss / df * szz / (szx * szx)).sqrt();
    let f = slope * slope * szz / (first_stage_rss / df);
    (effect, se, f)
}

#[test]
fn the_frequentist_route_matches_the_wald_ratio_and_certifies_the_instrument_from_the_graph() {
    let columns = series();
    let (effect, se, f) = wald(&columns);
    assert!(f > 10.0, "fixture first stage must be strong: F={f}");
    let result = builder(&columns, false)
        .estimator(EstimatorId::TemporalLaggedInstrument)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(2))
        .unwrap();
    let estimate = result.estimate.as_effect().unwrap();
    assert!((estimate.ate - effect).abs() < 1e-8, "{} vs {effect}", estimate.ate);
    assert!((estimate.se_analytic() - se).abs() < 1e-8, "{} vs {se}", estimate.se_analytic());
    // The structural effect is 2; the confounded regression coefficient is not.
    assert!((estimate.ate - 2.0).abs() < 3.0 * se, "{} vs 2", estimate.ate);
    assert!(result.posterior.is_none());
    assert_ne!(result.support_status, Some(antecedent::support::CellStatus::Licensed));
    // Exclusion is a recorded premise of the identification, taken from the graph.
    assert!(
        result.identification.required_assumptions.entries.iter().any(|a| matches!(
            a.assumption,
            antecedent_core::Assumption::ExclusionRestriction { .. }
        ))
    );
}

#[test]
fn the_bayesian_route_is_a_reduced_form_ratio_posterior_not_a_penalized_wald() {
    let columns = series();
    let (effect, se, _) = wald(&columns);
    let run = |prior_scale: f64| {
        builder(&columns, false)
            .estimator(EstimatorId::BayesianTemporalLaggedInstrument)
            .inference(InferenceMode::Bayesian(
                BayesianConfig::conjugate().prior_scale(prior_scale).n_draws(4000),
            ))
            .build()
            .unwrap()
            .run(&ExecutionContext::for_tests(2))
            .unwrap()
    };
    let diffuse = run(100.0);
    let posterior = diffuse.posterior.as_ref().expect("a Bayesian study retains its posterior");
    let column = posterior.effect_column().unwrap();
    let mean = posterior.summaries.mean[column];
    assert!((mean - effect).abs() < 3.0 * se, "{mean} vs {effect}");
    assert!(posterior.summaries.q025[column] < effect && effect < posterior.summaries.q975[column]);
    assert_eq!(posterior.draws.n_draws, 4000);
    let tight = run(0.05);
    let tight_posterior = tight.posterior.as_ref().unwrap();
    let tight_mean = tight_posterior.summaries.mean[tight_posterior.effect_column().unwrap()];
    assert!((tight_mean - effect).abs() > 0.5, "a tight prior must move the ratio: {tight_mean}");
}

#[test]
fn a_lagged_confounder_of_the_instrument_is_refused_with_its_next_attempt() {
    let columns = series();
    let error = builder(&columns, true)
        .estimator(EstimatorId::TemporalLaggedInstrument)
        .build()
        .and_then(|study| study.run(&ExecutionContext::for_tests(2)))
        .unwrap_err();
    let text = error.to_string();
    assert!(text.contains("effect_not_identified"), "{text}");
    assert!(text.contains("exogeneity"), "{text}");
    assert!(text.contains("Next attempt"), "{text}");
}

#[test]
fn the_identifier_and_the_estimators_go_together() {
    let columns = series();
    let without_identifier = Study::series(data(&columns))
        .graph(graph(false))
        .query(query())
        .estimator(EstimatorId::TemporalLaggedInstrument)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap_err();
    assert!(without_identifier.to_string().contains("option_not_applicable"));
    let frequentist_id_under_bayes = builder(&columns, false)
        .estimator(EstimatorId::TemporalLaggedInstrument)
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate()))
        .build()
        .unwrap_err();
    assert!(frequentist_id_under_bayes.to_string().contains("estimator_inference_mismatch"));
}
