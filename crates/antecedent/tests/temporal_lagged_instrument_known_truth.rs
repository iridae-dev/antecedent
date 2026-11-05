//! Known-truth check of `temporal.lagged_instrument` on every fixed `TemporalDag` coordinate it
//! can estimate: pulse and one-step sustained effects, explicit and accepted structure, and
//! every validation suite.
//!
//! The design is exactly balanced. The instrument `z` cycles `1, 0, -1, 0`; the confounder `u`
//! alternates `1, -1`; the treatment and outcome noises are the quarter-period sine `w`
//! (`0, 1, 0, -1`) shifted onto the aligned rows. The Wald rows number a whole number of
//! periods, so `z` is exactly uncorrelated with `u` and both noises, and the Wald ratio of
//! covariances equals the structural effect `2` to rounding error. The homoskedastic standard
//! error is recomputed here from the raw series by the closed form.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::cast_precision_loss,
    reason = "deterministic fixture indices are small and exact in f64"
)]

#[path = "retained_run/mod.rs"]
mod retained;

use retained::run_retained;

use antecedent::{AcceptedGraph, EstimatorId, IdentifierId, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ExecutionContext, Lag, TemporalEffectQuery, TemporalPolicy, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_graph::{TemporalDag, ensure_lagged};

/// Rows: `ROWS` aligned Wald rows (a multiple of the period 4) plus horizon 1 plus lag 1.
const ROWS: usize = 400;
const N: usize = ROWS + 2;
const X: u32 = 0;
const Y: u32 = 1;
const Z: u32 = 2;
const U: u32 = 3;
const STRUCTURAL: f64 = 2.0;

/// `x_s = 0.9 z_{s-1} + 0.8 u_s + 0.3 w_{s-1}`, `y_t = 2 x_{t-1} + u_{t-1} + 0.3 w_{t-2}`.
fn columns() -> [Vec<f64>; 4] {
    let cosine = [1.0, 0.0, -1.0, 0.0];
    let sine = [0.0, 1.0, 0.0, -1.0];
    let z: Vec<f64> = (0..N).map(|s| cosine[s % 4]).collect();
    let u: Vec<f64> = (0..N).map(|s| if s % 2 == 0 { 1.0 } else { -1.0 }).collect();
    let x: Vec<f64> =
        (0..N).map(|s| 0.9 * cosine[(s + 3) % 4] + 0.8 * u[s] + 0.3 * sine[(s + 3) % 4]).collect();
    let y: Vec<f64> = (0..N)
        .map(|t| if t == 0 { 0.0 } else { 2.0 * x[t - 1] + u[t - 1] + 0.3 * sine[(t + 2) % 4] })
        .collect();
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

/// `z_{t-2} -> x_{t-1} -> y_t`, `u_{t-1} -> x_{t-1}`, `u_{t-1} -> y_t`.
fn graph() -> TemporalDag {
    let mut g = TemporalDag::empty();
    let mut node = |variable: u32, lag: u32| {
        ensure_lagged(&mut g, VariableId::from_raw(variable), Lag::from_raw(lag)).unwrap()
    };
    let (z2, x1, y0, u1) = (node(Z, 2), node(X, 1), node(Y, 0), node(U, 1));
    g.insert_directed(z2, x1).unwrap();
    g.insert_directed(x1, y0).unwrap();
    g.insert_directed(u1, x1).unwrap();
    g.insert_directed(u1, y0).unwrap();
    g
}

fn query(sustained: bool) -> CausalQuery {
    let base = TemporalEffectQuery::pulse(VariableId::from_raw(X), VariableId::from_raw(Y), 1.0)
        .with_horizon_steps(1);
    CausalQuery::TemporalEffect(if sustained {
        base.with_policy(TemporalPolicy::sustained(-1, -1))
    } else {
        base.with_policy(TemporalPolicy::pulse(-1))
    })
}

/// Homoskedastic standard error of the Wald ratio over rows `k = 0..ROWS` pairing `z_k`,
/// `x_{k+1}` and `y_{k+2}`, from the closed form of covariances.
fn wald_standard_error(columns: &[Vec<f64>; 4]) -> (f64, f64) {
    let pick =
        |c: &[f64], offset: usize| -> Vec<f64> { (0..ROWS).map(|k| c[k + offset]).collect() };
    let (zs, xs, ys) = (pick(&columns[2], 0), pick(&columns[0], 1), pick(&columns[1], 2));
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let (zm, xm, ym) = (mean(&zs), mean(&xs), mean(&ys));
    let szz: f64 = zs.iter().map(|z| (z - zm).powi(2)).sum();
    let szx: f64 = zs.iter().zip(&xs).map(|(z, x)| (z - zm) * (x - xm)).sum();
    let szy: f64 = zs.iter().zip(&ys).map(|(z, y)| (z - zm) * (y - ym)).sum();
    let effect = szy / szx;
    let structural_rss: f64 =
        xs.iter().zip(&ys).map(|(x, y)| (y - (ym - effect * xm) - effect * x).powi(2)).sum();
    let se = (structural_rss / (ROWS - 2) as f64 * szz / (szx * szx)).sqrt();
    (effect, se)
}

fn run(
    columns: &[Vec<f64>; 4],
    accepted: bool,
    sustained: bool,
    suite: RefuteSuite,
) -> antecedent::StudyResult {
    let base = Study::series(data(columns));
    let base = if accepted {
        base.graph(AcceptedGraph::temporal_dag(graph()))
    } else {
        base.graph(graph())
    };
    run_retained(
        base.query(query(sustained))
            .identifier(IdentifierId::TemporalIvUnfolded)
            .estimator(EstimatorId::TemporalLaggedInstrument)
            .inference(InferenceMode::Frequentist)
            .refute(suite)
            .bootstrap_replicates(0),
        "retained run",
        &ExecutionContext::for_tests(2),
    )
}

/// Pulse and one-step sustained, explicit and accepted structure, every validation suite on a
/// fixed `TemporalDag`: the explicitly selected lagged-instrument estimator returns the Wald
/// ratio of the exactly balanced design, which is the structural effect.
#[test]
fn lagged_instrument_returns_the_structural_effect_on_every_fixed_dag_coordinate() {
    let columns = columns();
    let (wald, wald_se) = wald_standard_error(&columns);
    assert!((wald - STRUCTURAL).abs() < 1e-9, "the balanced design must be exact: {wald}");
    assert!(wald_se > 0.0);
    for accepted in [false, true] {
        for sustained in [false, true] {
            for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
                let label = format!("accepted={accepted} sustained={sustained} {suite:?}");
                let result = run(&columns, accepted, sustained, suite);
                let estimate = result.estimate().as_effect().unwrap();
                assert!((estimate.ate - STRUCTURAL).abs() < 1e-9, "{label}: {}", estimate.ate);
                assert!(
                    (estimate.se_analytic() - wald_se).abs() < 1e-9,
                    "{label}: se {} vs {wald_se}",
                    estimate.se_analytic()
                );
                assert!(result.posterior.is_none(), "{label}: frequentist route");
            }
        }
    }
}
