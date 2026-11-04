//! Independent construction check of the mechanism-refit bootstrap interval of a static
//! counterfactual.
//!
//! The route draws, for each replicate, multinomial row counts from one seeded stream, refits every
//! mechanism under those frequency weights on the original table, recomputes the mean unit
//! contrast, and publishes `estimate +/- z * SE` with `SE` the sample standard deviation of the
//! replicate values.
//!
//! The reference is a hand-written loop that shares nothing with the production fit: it replays the
//! same seeded count streams, then refits the one mechanism that carries the contrast (the outcome
//! `y ~ 1 + z + t`, linear Gaussian on this table) by weighted least squares from the normal
//! equations solved with Gaussian elimination. For a linear outcome mechanism the abduction-action-
//! prediction unit contrast is the coefficient on `t` for every unit, so the replicate value is that
//! coefficient and the point value is its unweighted least-squares value. Tolerances: the library
//! solves the same 3 x 3 weighted normal equations with an orthogonal factorisation; the design is
//! standardised-scale with condition number below 10^2, so the two solvers differ by about
//! `cond * eps = 10^-14` on the coefficient; 10^-9 absolute (relative for the standard error)
//! leaves five orders of margin without admitting any construction difference, which would move the
//! standard error at the 10^-3 relative level or more.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{CellStatus, InferenceMode, PublishedScalarUncertainty, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, CounterfactualQuery, ExecutionContext, Intervention, StreamDomain, Value,
    VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};

const ROWS: usize = 300;
const REPLICATES: u32 = 40;
/// Route key of the counterfactual bootstrap's count streams.
const ROUTE_KEY: u64 = 0x0CF0_B007;
/// Two-sided 95% normal critical value, `Phi^-1(0.975)`.
const Z_975: f64 = 1.959_963_984_540_054;

/// Deterministic uniform stream (xorshift64*).
struct Stream(u64);

impl Stream {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let bits = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11;
        (bits as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

/// Columns `z, t, y`: `t = 0.6 z + N(0, 1)`, `y = 1 + 2 t + z + N(0, 0.5^2)`.
fn columns() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut stream = Stream(3);
    let z: Vec<f64> = (0..ROWS).map(|_| stream.normal()).collect();
    let t: Vec<f64> = z.iter().map(|z| 0.6 * z + stream.normal()).collect();
    let y: Vec<f64> =
        t.iter().zip(&z).map(|(t, z)| 1.0 + 2.0 * t + z + 0.5 * stream.normal()).collect();
    (z, t, y)
}

fn table(z: &[f64], t: &[f64], y: &[f64]) -> TabularData {
    TabularData::from_f64_columns([("z", z), ("t", t), ("y", y)]).unwrap()
}

/// `z -> t`, `z -> y`, `t -> y` over `(z, t, y)`.
fn dag() -> Dag {
    let mut graph = Dag::with_variables(3);
    for (from, to) in [(0, 1), (0, 2), (1, 2)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    graph
}

fn study(data: TabularData, replicates: u32) -> Study {
    let query = CounterfactualQuery::new(
        VariableId::from_raw(2),
        Arc::from([Intervention::set(VariableId::from_raw(1), Value::f64(1.0))]),
    )
    .with_control_level(0.0);
    Study::tabular(data)
        .graph(dag())
        .query(CausalQuery::Counterfactual(query))
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(replicates)
        .build()
        .unwrap()
}

/// Coefficient on `t` of the weighted least squares of `y` on `(1, z, t)`: the 3 x 3 weighted normal
/// equations solved by Gaussian elimination with partial pivoting.
fn weighted_slope(z: &[f64], t: &[f64], y: &[f64], w: &[f64]) -> f64 {
    let mut a = [[0.0_f64; 4]; 3];
    for i in 0..z.len() {
        let x = [1.0, z[i], t[i]];
        for r in 0..3 {
            for c in 0..3 {
                a[r][c] += w[i] * x[r] * x[c];
            }
            a[r][3] += w[i] * x[r] * y[i];
        }
    }
    for col in 0..3 {
        let pivot = (col..3).max_by(|&p, &q| a[p][col].abs().total_cmp(&a[q][col].abs())).unwrap();
        a.swap(col, pivot);
        for r in col + 1..3 {
            let factor = a[r][col] / a[col][col];
            for c in col..4 {
                a[r][c] -= factor * a[col][c];
            }
        }
    }
    let mut beta = [0.0_f64; 3];
    for r in (0..3).rev() {
        let tail: f64 = (r + 1..3).map(|c| a[r][c] * beta[c]).sum();
        beta[r] = (a[r][3] - tail) / a[r][r];
    }
    beta[2]
}

/// The reference replicate values: the same seeded multinomial counts, refit by hand.
fn reference_replicates(
    ctx: &ExecutionContext,
    z: &[f64],
    t: &[f64],
    y: &[f64],
    replicates: u32,
) -> Vec<f64> {
    let n = z.len();
    (0..replicates)
        .map(|replicate| {
            let mut rng = ctx.rng.stream_for(
                StreamDomain::Attribution,
                ROUTE_KEY.wrapping_add(u64::from(replicate).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
            );
            let mut counts = vec![0.0_f64; n];
            for _ in 0..n {
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "the floor of a uniform draw scaled by the row count is a valid row index"
                )]
                let row = ((rng.next_f64() * n as f64) as usize).min(n - 1);
                counts[row] += 1.0;
            }
            weighted_slope(z, t, y, &counts)
        })
        .collect()
}

#[test]
fn the_refit_bootstrap_interval_equals_the_hand_refit_of_the_same_seeded_resamples() {
    let (z, t, y) = columns();
    let ctx = ExecutionContext::for_tests(3);
    let result = study(table(&z, &t, &y), REPLICATES).run(&ctx).unwrap();
    let effect = result.estimate().as_effect().expect("an effect estimate");

    // Point value: the unweighted least-squares slope of the outcome mechanism.
    let point = weighted_slope(&z, &t, &y, &vec![1.0; ROWS]);
    assert!((effect.ate - point).abs() < 1e-9, "point {} vs reference {point}", effect.ate);

    // Standard error: sample standard deviation of the hand-refit replicate values.
    let values = reference_replicates(&ctx, &z, &t, &y, REPLICATES);
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let se = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() as f64 - 1.0))
        .sqrt();
    assert!(se > 0.0 && se.is_finite());
    let published = effect.uncertainty.bootstrap.se.expect("bootstrap standard error");
    assert!(((published - se) / se).abs() < 1e-9, "published SE {published} vs hand-refit SE {se}");
    assert_eq!(effect.uncertainty.bootstrap.replicates_ok, Some(REPLICATES));
    assert_eq!(effect.uncertainty.bootstrap.replicates_failed, Some(0));

    // Interval: estimate +/- z SE with the 0.975 normal quantile.
    let (lower, upper) =
        PublishedScalarUncertainty::select(effect).interval(effect).expect("a published interval");
    assert!((lower - (point - Z_975 * se)).abs() < 1e-9, "lower {lower}");
    assert!((upper - (point + Z_975 * se)).abs() < 1e-9, "upper {upper}");
}

#[test]
fn the_bootstrap_route_is_supported_with_its_construction_check_and_stays_replicable() {
    let (z, t, y) = columns();
    let ctx = ExecutionContext::for_tests(3);
    let first = study(table(&z, &t, &y), REPLICATES).run(&ctx).unwrap();
    assert_eq!(first.support_status(), Some(CellStatus::Supported));
    // A different seed draws different resamples, so the reference differs from the first run's:
    // the check above is a statement about these streams, not about any stream.
    let other = ExecutionContext::for_tests(4);
    let values = reference_replicates(&other, &z, &t, &y, REPLICATES);
    let second = study(table(&z, &t, &y), REPLICATES).run(&other).unwrap();
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let se = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() as f64 - 1.0))
        .sqrt();
    let published = second.estimate().as_effect().unwrap().uncertainty.bootstrap.se.unwrap();
    assert!(((published - se) / se).abs() < 1e-9, "seed 4: published {published} vs {se}");
}
