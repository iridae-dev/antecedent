//! Independent construction check of the joint-horizon block-weight posterior.
//!
//! Under Bayesian inference a response surface that retains its joint horizon replicates replaces
//! the per-horizon conjugate fits with draws of the posterior over the surface given the empirical
//! law of the lag windows: for each draw, one Exp(1) weight per block of `L` consecutive series
//! times (the Dirichlet(1, ..., 1) block weights up to scale), shared by every horizon, and a
//! weighted least-squares refit of each horizon on those weights. The published draws are the
//! refit surface with its deviation from the full-sample surface scaled by a per-cell dispersion
//! factor `F_c > 0` (fixed-b ratio times kernel-bias factor, constant across draws).
//!
//! The posterior of the weights is exactly known, so the check replays it. The reference draws the
//! Exp(1) block weights itself from the same seeded stream (`-ln u`, no kernel call), refits the
//! one-horizon design `y_t ~ 1 + x_t` by weighted least squares from the 2 x 2 normal equations,
//! forms the two cells `b0` and `b0 + b1`, and requires that every retained draw's deviation from the
//! unweighted least-squares surface equal `F_c` times the reference deviation, with one `F_c` per
//! cell for all draws. That fixes the block partition, the weight law, the stream, the shared
//! weights and the weighted refit; a draw built any other way (equal weights, per-row weights, a
//! different block length, a conjugate fit) cannot be proportional to the reference over the draws.
//!
//! Tolerance: both sides are double precision on a 2 x 2 weighted normal system with condition below
//! 10^2 and weights in a ratio of at most about 10^2, so the coefficient difference is about
//! `cond * eps = 10^-14`; the retained draw adds two scalings at `eps * |value|`. 10^-9 absolute on
//! the deviation (themselves of order 10^-1) leaves five orders of margin, and any construction
//! difference moves a deviation at the 10^-2 level.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "integration test builds deterministic fixtures from small counts"
)]

use std::sync::Arc;

use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, CausalResponse, ContinuousDomain, ExecutionContext, GridSpec, Lag,
    ResponseFunctional, ResponseQuery, StreamDomain, TemporalPolicy, TemporalResponseSpec,
    VariableId, derive_stream,
};
use antecedent_data::{TableView, TimeSeriesData};
use antecedent_estimate::{
    JOINT_HORIZON_LAYOUT, JOINT_HORIZON_REPLICATES, TEMPORAL_RESPONSE_BLOCK_LENGTH,
};
use antecedent_graph::{TemporalDag, ensure_lagged};

const N: usize = 300;
const DRAWS: usize = 150;
/// Stream base of the block-weight draws.
const WEIGHT_STREAM: u64 = 0x7E51;

fn series() -> TimeSeriesData {
    let mut state = 0x1234_5678_9ABC_DEF1_u64;
    let mut normal = move || {
        let mut uniform = || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64 + 1e-12
        };
        let (u, v) = (uniform(), uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    };
    let mut x = vec![0.0; N];
    let mut y = vec![0.0; N];
    for t in 0..N {
        let previous = if t > 0 { x[t - 1] } else { 0.0 };
        x[t] = 0.3 + 0.5 * previous + 0.3 * normal();
        y[t] = 1.0 + 2.0 * x[t] + 0.5 * normal();
    }
    TimeSeriesData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())], 1).unwrap()
}

fn lag_one_graph() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let x0 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
    let y0 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(x0, y0).unwrap();
    graph
}

fn diagnostic(response: &CausalResponse, id: &str) -> Vec<f64> {
    response
        .support
        .diagnostics
        .iter()
        .find(|d| d.id.as_ref() == id)
        .unwrap_or_else(|| panic!("missing support diagnostic {id}"))
        .values
        .to_vec()
}

/// `(b0, b1)` of the weighted least squares of `y` on `(1, x)` from the 2 x 2 normal equations.
fn weighted_line(x: &[f64], y: &[f64], w: &[f64]) -> (f64, f64) {
    let (mut sw, mut swx, mut swxx, mut swy, mut swxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for i in 0..x.len() {
        sw += w[i];
        swx += w[i] * x[i];
        swxx += w[i] * x[i] * x[i];
        swy += w[i] * y[i];
        swxy += w[i] * x[i] * y[i];
    }
    let det = sw * swxx - swx * swx;
    ((swxx * swy - swx * swxy) / det, (sw * swxy - swx * swy) / det)
}

#[test]
fn the_block_weight_draws_are_the_exp1_block_weighted_refits_up_to_one_dispersion_factor_per_cell()
{
    std::thread::Builder::new()
        .name("block-weight-construction".into())
        .stack_size(32 * 1024 * 1024)
        .spawn(|| {
            let ctx = ExecutionContext::for_tests(8);
            let data = series();
            let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
                outcome: VariableId::from_raw(1),
                treatment: ContinuousDomain::new(
                    VariableId::from_raw(0),
                    GridSpec::Values(Arc::from([0.0, 1.0])),
                ),
            })
            .with_temporal(
                TemporalResponseSpec::new(vec![1], TemporalPolicy::pulse(0), None).unwrap(),
            );
            let result = Study::series(data.clone())
                .graph(lag_one_graph())
                .query(CausalQuery::Response(query))
                .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(DRAWS)))
                .refute(RefuteSuite::None)
                .bootstrap_replicates(0)
                .retain_joint_horizons(true)
                .build()
                .unwrap()
                .run(&ctx)
                .unwrap();
            let response = result.response().expect("a response");

            // The published layout: DRAWS replicate-major rows of the two dose cells.
            let layout = diagnostic(response, JOINT_HORIZON_LAYOUT);
            let (replicates, cells) = (layout[0] as usize, layout[1] as usize);
            assert_eq!((replicates, cells), (DRAWS, 2), "no draw may be dropped or reordered");
            let values = diagnostic(response, JOINT_HORIZON_REPLICATES);
            assert_eq!(values.len(), replicates * cells);

            // The block partition the route disclosed: block length and the rows it covers.
            let block_info = diagnostic(response, TEMPORAL_RESPONSE_BLOCK_LENGTH);
            let (block, rows) = (block_info[0] as usize, block_info[3] as usize);
            assert!(block >= 1 && rows > 2 * block && rows <= N);

            // Horizon 1 of a pulse(0) policy reads the treatment at the outcome's own step
            // (the treatment lag is horizon - 1), so there is no lost leading row: row i is series
            // time N - rows + i for both columns.
            let column = |id: u32| data.float64_slice(VariableId::from_raw(id)).unwrap().to_vec();
            let (x_all, y_all) = (column(0), column(1));
            let first = N - rows;
            let x: Vec<f64> = (0..rows).map(|i| x_all[first + i]).collect();
            let y: Vec<f64> = (0..rows).map(|i| y_all[first + i]).collect();

            // Full-sample surface: the centre the draws' deviations are taken from.
            let (a0, a1) = weighted_line(&x, &y, &vec![1.0; rows]);
            let point = [a0, a0 + a1];

            // Reference deviations: independently drawn Exp(1) block weights, hand refit.
            let mut reference = vec![[0.0_f64; 2]; replicates];
            for (replicate, out) in reference.iter_mut().enumerate() {
                let mut rng = ctx.rng.stream_for(
                    StreamDomain::TemporalBlock,
                    derive_stream(WEIGHT_STREAM, replicate as u64),
                );
                let mut weights = Vec::with_capacity(rows);
                while weights.len() < rows {
                    let weight = -rng.next_f64().max(f64::MIN_POSITIVE).ln();
                    let take = block.min(rows - weights.len());
                    weights.extend(std::iter::repeat_n(weight, take));
                }
                let (b0, b1) = weighted_line(&x, &y, &weights);
                *out = [b0 - point[0], b0 + b1 - point[1]];
            }

            for cell in 0..cells {
                let published: Vec<f64> =
                    (0..replicates).map(|r| values[r * cells + cell] - point[cell]).collect();
                let expected: Vec<f64> = reference.iter().map(|d| d[cell]).collect();
                let spread = expected.iter().map(|e| e * e).sum::<f64>();
                assert!(
                    (spread / replicates as f64).sqrt() > 1e-3,
                    "cell {cell}: the reference draws must vary"
                );
                // One positive dispersion factor per cell, shared by all draws.
                let factor =
                    published.iter().zip(&expected).map(|(d, e)| d * e).sum::<f64>() / spread;
                assert!(factor.is_finite() && factor > 0.0, "cell {cell}: factor {factor}");
                for (r, (d, e)) in published.iter().zip(&expected).enumerate() {
                    assert!(
                        (d - factor * e).abs() < 1e-9,
                        "cell {cell} draw {r}: published deviation {d} vs {factor} x {e}"
                    );
                }
            }
        })
        .unwrap()
        .join()
        .unwrap();
}
