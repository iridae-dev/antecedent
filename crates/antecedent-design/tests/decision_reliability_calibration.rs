//! Expectation checks of the decision-reliability selection correction against a known
//! optimum, over the candidate-set size, Monte Carlo budget and noise-coupling grid.
//!
//! Nothing here is measured at implementation time: the test is `#[ignore]`d and runs only
//! at the final measurement step. It states no coverage and issues no record. It checks two
//! expectation statements of the derived routes, both of the whole procedure (draw on a
//! selection stream, select, re-evaluate on an independent validation stream):
//!
//! * the winner's-curse bound is not below the mean in-sample bias of the selected
//!   candidate (Tippett's expected maximum of equal-mean Gaussians is the independent
//!   oracle: all true utilities are equal, which is the maximal selection bias);
//! * the held-out value of the selected candidate is unbiased for its true value.
//!
//! Design: candidate-set size {2, 8, 64}, draws per candidate {100, 1000}, noise coupling
//! {separate streams, shared stream with correlation 0, 0.5, 0.9}; at least 5000
//! replicates; every replicate names its streams by (run seed, purpose, replicate) and
//! selection and validation never share a coordinate. Boundary probes: two candidates
//! (the maximum is barely biased), 64 candidates at the smallest budget (the widest
//! selection), shared noise at 0.9 (the union bound must still hold).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "bounded simulation sizes and small indices"
)]

#[path = "../../antecedent/tests/common/calibration.rs"]
mod calibration;

use antecedent_core::ExecutionContext;
use antecedent_design::decision_reliability::{
    KnownOptimumDesign, NoiseCoupling, simulate_replicate,
};
use calibration::{grid_seed, map_replicates, n_sim, smoke};

const CANDIDATES: [usize; 3] = [2, 8, 64];
const BUDGETS: [usize; 2] = [100, 1000];
const COUPLINGS: [(&str, NoiseCoupling); 4] = [
    ("separate_streams", NoiseCoupling::SeparateStreams),
    ("shared_rho_0.0", NoiseCoupling::SharedStream { correlation: 0.0 }),
    ("shared_rho_0.5", NoiseCoupling::SharedStream { correlation: 0.5 }),
    ("shared_rho_0.9", NoiseCoupling::SharedStream { correlation: 0.9 }),
];

fn mean_and_se(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, (variance / n).sqrt())
}

#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn winners_curse_bound_holds_on_average_and_held_out_value_is_unbiased() {
    let replicates = if smoke() { n_sim() } else { n_sim().max(5000) };
    let mut cell = 0_u64;
    for &k in &CANDIDATES {
        for &budget in &BUDGETS {
            for &(label, coupling) in &COUPLINGS {
                cell += 1;
                let design = KnownOptimumDesign {
                    true_means: vec![0.0; k],
                    noise_sd: 1.0,
                    n_draws: budget,
                    coupling,
                };
                let run_seed = grid_seed(0x3d00_0000 + cell);
                let outcomes = map_replicates(replicates, |replicate| {
                    let ctx = ExecutionContext::for_tests(run_seed ^ replicate);
                    let outcome = simulate_replicate(&design, run_seed, replicate, &ctx)
                        .expect("known-optimum replicate");
                    (outcome.in_sample_bias, outcome.held_out_bias, outcome.winners_curse_bound)
                });
                let in_sample: Vec<f64> = outcomes.iter().map(|o| o.0).collect();
                let held_out: Vec<f64> = outcomes.iter().map(|o| o.1).collect();
                let bounds: Vec<f64> = outcomes.iter().map(|o| o.2).collect();
                let (bias, bias_se) = mean_and_se(&in_sample);
                let (held, held_se) = mean_and_se(&held_out);
                let (bound, _) = mean_and_se(&bounds);
                // A replicate without a reported bound is a failure, not a pass: the mean
                // is NaN and the comparison below is false.
                assert!(
                    bound + 3.0 * bias_se >= bias,
                    "k={k} budget={budget} {label}: bound {bound} below in-sample bias {bias} \
                     (se {bias_se})"
                );
                assert!(
                    held.abs() <= 4.0 * held_se,
                    "k={k} budget={budget} {label}: held-out bias {held} (se {held_se})"
                );
                println!(
                    "diagnostic-expectation winners_curse_bound k={k} budget={budget} \
                     coupling={label} replicates={replicates} in_sample_bias={bias:.5} \
                     bound={bound:.5} held_out_bias={held:.5} held_out_se={held_se:.5}; \
                     expectation_check_no_coverage_claim"
                );
            }
        }
    }
}
