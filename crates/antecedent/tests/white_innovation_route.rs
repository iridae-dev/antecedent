//! The declared white-innovation temporal posterior through the public `Study` flow.
//!
//! A single-step pulse on a `TemporalDag` has a tempered (generalized) posterior by default.
//! Declaring serially independent innovations on the Bayesian inference swaps in the untempered
//! full-likelihood posterior of the same effect, licensed only by that declaration and refutable
//! by a Ljung-Box test of the residuals. The same premise licenses the untempered posterior of a
//! response curve at the single horizon of one step ahead; any other horizon, and the default,
//! keep the tempered generalized posterior. The reference is the least-squares coefficient and its
//! standard error of the lag-aligned regression, computed here from the raw series. The route is
//! an unlicensed estimator with calibration unmeasured, and every result says so.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    reason = "integration test builds deterministic fixtures from small counts"
)]

use std::sync::Arc;

use antecedent::estimate::InnovationWhitenessGate;
use antecedent::prelude::{CausalSchemaBuilder, TabularData};
use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    Assumption, CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, Lag, ResponseFunctional,
    ResponseIdentification, ResponseQuery, ResponseValue, TemporalEffectQuery, TemporalPolicy,
    TemporalResponseSpec, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_graph::{TemporalDag, ensure_lagged};

const N: usize = 400;
const DRAWS: usize = 4000;
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

/// `x_t = rho_x x_{t-1} + noise`, `y_t = 0.8 x_{t-1} + e_t` with `e` AR(1) at `rho_e`.
fn series(rho_x: f64, rho_e: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = Lcg(seed);
    let (mut x, mut y) = (vec![0.0; N], vec![0.0; N]);
    let (mut previous, mut e) = (0.0_f64, 0.0_f64);
    for t in 0..N {
        previous = rho_x * previous + rng.normal();
        x[t] = previous;
        e = rho_e * e + rng.normal();
        y[t] = if t > 0 { 0.8 * x[t - 1] } else { 0.0 } + e;
    }
    (x, y)
}

fn data(x: &[f64], y: &[f64]) -> TimeSeriesData {
    TimeSeriesData::from_f64_columns([("x", x), ("y", y)], 1).unwrap()
}

fn lag_one_graph() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let x1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(x1, y0).unwrap();
    graph
}

fn pulse() -> CausalQuery {
    CausalQuery::TemporalEffect(
        TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
            .with_policy(TemporalPolicy::pulse(-1))
            .with_horizon_steps(1),
    )
}

fn gate() -> InnovationWhitenessGate {
    InnovationWhitenessGate { max_lag: 6, level: 0.01 }
}

fn study(
    x: &[f64],
    y: &[f64],
    config: BayesianConfig,
) -> Result<antecedent::StudyResult, antecedent::CausalError> {
    Study::series(data(x, y))
        .graph(lag_one_graph())
        .query(pulse())
        .inference(InferenceMode::Bayesian(config))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()?
        .run(&ExecutionContext::for_tests(11))
}

fn config() -> BayesianConfig {
    BayesianConfig::conjugate().n_draws(DRAWS).prior_scale(1.0e3)
}

/// Least-squares coefficient on `x_{t-1}` and its standard error for rows `t = 1..N`.
fn ols(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = (N - 1) as f64;
    let (xs, ys) = (&x[..N - 1], &y[1..]);
    let (xm, ym) = (xs.iter().sum::<f64>() / n, ys.iter().sum::<f64>() / n);
    let sxx: f64 = xs.iter().map(|v| (v - xm).powi(2)).sum();
    let sxy: f64 = xs.iter().zip(ys).map(|(a, b)| (a - xm) * (b - ym)).sum();
    let beta = sxy / sxx;
    let rss: f64 = xs.iter().zip(ys).map(|(a, b)| (b - ym - beta * (a - xm)).powi(2)).sum();
    (beta, (rss / (n - 2.0) / sxx).sqrt())
}

#[test]
fn a_declared_premise_gives_the_untempered_least_squares_posterior_with_its_premise_recorded() {
    let (x, y) = series(0.5, 0.0, 2);
    let (beta, se) = ols(&x, &y);
    let result = study(&x, &y, config().white_innovation(gate())).unwrap();
    let posterior = result.posterior.as_ref().expect("a Bayesian study retains its posterior");
    let column = posterior.effect_column().expect("effect column");
    let mean = posterior.summaries.mean[column];
    let sd = posterior.summaries.sd[column];
    assert!((mean - beta).abs() < 3.0 * se / (DRAWS as f64).sqrt() + 2e-3, "{mean} vs {beta}");
    assert!((sd / se - 1.0).abs() < 0.06, "posterior sd over OLS se {}", sd / se);
    assert!(posterior.assumptions.entries.iter().any(|record| matches!(
        &record.assumption,
        Assumption::ParametricRestriction(p) if p.id.as_ref() == ASSUMPTION
    )));
    assert!(
        posterior
            .diagnostics
            .notes
            .iter()
            .any(|note| note.starts_with("white_innovation.ljung_box"))
    );
}

#[test]
fn without_the_declaration_the_posterior_keeps_its_tempered_meaning() {
    let (x, y) = series(0.5, 0.0, 2);
    let result = study(&x, &y, config()).unwrap();
    let posterior = result.posterior.as_ref().unwrap();
    assert!(!posterior.assumptions.entries.iter().any(|record| matches!(
        &record.assumption,
        Assumption::ParametricRestriction(p) if p.id.as_ref() == ASSUMPTION
    )));
    // No white-innovation test was run, and the default posterior is centred on the same
    // least-squares coefficient (tempering rescales the likelihood, it does not move the mean).
    assert!(!posterior.diagnostics.notes.iter().any(|note| note.starts_with("white_innovation")));
    let (beta, se) = ols(&x, &y);
    let column = posterior.effect_column().expect("effect column");
    assert!((posterior.summaries.mean[column] - beta).abs() < 3.0 * se / (DRAWS as f64).sqrt() + 2e-3);
}

#[test]
fn autocorrelated_innovations_refuse_the_declaration_and_name_the_tempered_posterior() {
    let (x, y) = series(0.8, 0.8, 1);
    let error = study(&x, &y, config().white_innovation(gate())).unwrap_err();
    let text = error.to_string();
    assert!(text.contains("white_innovation.residual_autocorrelation"), "{text}");
    assert!(text.contains("tempered posterior"), "{text}");
    assert!(text.contains("Next attempt"), "{text}");
    // The same series without the premise still has its tempered posterior.
    assert!(study(&x, &y, config()).is_ok());
}

#[test]
fn a_route_that_does_not_read_the_premise_refuses_it() {
    let schema = CausalSchemaBuilder::new()
        .continuous("t")
        .treatment()
        .continuous("y")
        .outcome()
        .build()
        .unwrap();
    let values_t: Vec<f64> = (0..40).map(|i| f64::from(i % 2)).collect();
    let values_y: Vec<f64> =
        (0..40).map(|i| 1.0 + 2.0 * f64::from(i % 2) + 0.1 * f64::from(i % 7)).collect();
    let table =
        TabularData::from_f64_columns([("t", values_t.as_slice()), ("y", values_y.as_slice())])
            .unwrap();
    let dag = antecedent::Dag::from_named_edges(&schema, &[("t", "y")]).unwrap();
    let query = antecedent::AverageEffectQuery::binary_ate(
        schema.id_of("t").unwrap(),
        schema.id_of("y").unwrap(),
    );
    let error = Study::tabular(table)
        .graph(dag)
        .query(query)
        .inference(InferenceMode::Bayesian(config().white_innovation(gate())))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap_err();
    let text = error.to_string();
    assert!(text.contains("option_not_applicable"), "{text}");
    assert!(text.contains("Next attempt"), "{text}");
}

fn curve(horizons: Vec<u32>) -> CausalQuery {
    CausalQuery::Response(
        ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: VariableId::from_raw(1),
            treatment: ContinuousDomain::new(
                VariableId::from_raw(0),
                GridSpec::Values(Arc::from([0.0, 1.0, 2.0])),
            ),
        })
        .with_temporal(
            TemporalResponseSpec::new(horizons, TemporalPolicy::pulse(-1), None).unwrap(),
        ),
    )
}

fn curve_study(
    x: &[f64],
    y: &[f64],
    horizons: Vec<u32>,
    config: BayesianConfig,
) -> Result<antecedent::StudyResult, antecedent::CausalError> {
    Study::series(data(x, y))
        .graph(lag_one_graph())
        .query(curve(horizons))
        .inference(InferenceMode::Bayesian(config))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()?
        .run(&ExecutionContext::for_tests(12))
}

fn has_white_premise(assumptions: &antecedent_core::AssumptionSet) -> bool {
    assumptions.entries.iter().any(|record| {
        matches!(&record.assumption, Assumption::ParametricRestriction(p) if p.id.as_ref() == ASSUMPTION)
    })
}

#[test]
fn a_declared_premise_gives_the_untempered_curve_at_one_step_with_its_premise_recorded() {
    let (x, y) = series(0.5, 0.0, 2);
    let (beta, se) = ols(&x, &y);
    let result = curve_study(&x, &y, vec![1], config().white_innovation(gate())).unwrap();
    let response = result.response().expect("a temporal response payload");
    let ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) =
        &response.estimate
    else {
        panic!("expected a point-identified surface");
    };
    // The response is linear in the dose, so its slope is the lag-aligned coefficient and the
    // untempered posterior of a correctly specified model has the least-squares mean.
    let slope = mean[1] - mean[0];
    assert!((slope - beta).abs() < 3.0 * se / (DRAWS as f64).sqrt() + 5e-3, "{slope} vs {beta}");
    assert!(has_white_premise(&response.assumptions));
    let ids: Vec<&str> = response.support.diagnostics.iter().map(|d| d.id.as_ref()).collect();
    assert!(ids.contains(&"response.temporal_bayesian.white_innovation"), "{ids:?}");
    assert!(!ids.contains(&"response.temporal_bayesian.tempering"), "{ids:?}");
}

#[test]
fn the_curve_without_the_declaration_keeps_its_tempered_meaning() {
    let (x, y) = series(0.5, 0.0, 2);
    let result = curve_study(&x, &y, vec![1], config()).unwrap();
    let response = result.response().unwrap();
    assert!(!has_white_premise(&response.assumptions));
    let ids: Vec<&str> = response.support.diagnostics.iter().map(|d| d.id.as_ref()).collect();
    assert!(ids.contains(&"response.temporal_bayesian.tempering"), "{ids:?}");
    assert!(!ids.contains(&"response.temporal_bayesian.white_innovation"), "{ids:?}");
}

#[test]
fn a_curve_with_autocorrelated_innovations_refuses_the_premise_and_keeps_the_tempered_posterior() {
    let (x, y) = series(0.8, 0.8, 1);
    let error = curve_study(&x, &y, vec![1], config().white_innovation(gate())).unwrap_err();
    let text = error.to_string();
    assert!(text.contains("white_innovation.residual_autocorrelation"), "{text}");
    assert!(text.contains("tempered posterior"), "{text}");
    assert!(text.contains("Next attempt"), "{text}");
    assert!(curve_study(&x, &y, vec![1], config()).is_ok());
}

#[test]
fn a_multi_horizon_curve_refuses_the_premise_naming_the_next_attempt() {
    let (x, y) = series(0.5, 0.0, 2);
    let error = curve_study(&x, &y, vec![1, 2], config().white_innovation(gate())).unwrap_err();
    let text = error.to_string();
    assert!(text.contains("option_not_applicable"), "{text}");
    assert!(text.contains("Next attempt"), "{text}");
}
