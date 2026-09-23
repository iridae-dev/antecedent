//! Block-length sensitivity of the shared circular-block SE.
//!
//! Block length is not caller-configurable, so the check runs at the helper:
//! [`shared_circular_block_mixture_se_with_length`] at ×0.5, ×1 and ×2 of the
//! length production uses on each replicate — [`mixture_block_length`] over
//! the atom's influence, the (single-atom) mixture score and the regression's
//! normal-equation scores, at least the rule [`circular_block_length`]
//! (`max(span, ceil(m^(1/3)))`, capped at `m`) — with the construction the
//! temporal class / DBN atoms use: blocks of consecutive series times over the
//! lag-aligned rows, each row keeping its original lag window, and the
//! replicate SD scaled by the fixed-b factor. The atom is the Pulse regression
//! of `y[t] = B·x[t-1] + u[t]` (OLS slope with intercept) on its lag-aligned
//! `(x[t-1], y[t])` rows, `t = 1..n`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::{ColumnView, TableView, TimeSeriesData};
use antecedent_estimate::AlignedRows;

use super::{
    circular_block_length, mixture_block_length, shared_circular_block_mixture_se_with_length,
};

const B: f64 = 0.8;
const Z90: f64 = 1.644_853_626_951_472_2;
const REPLICATES: u32 = 199;

fn n_sim() -> u32 {
    std::env::var("ANTECEDENT_CALIBRATION_NSIM")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(400)
}

fn mix_seed(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn gaussian(seed: u64) -> impl FnMut() -> f64 {
    let mut state = mix_seed(seed) | 1;
    move || {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let u1 = ((state >> 33) as f64 / (1u64 << 31) as f64).clamp(1e-12, 1.0);
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let u2 = (state >> 33) as f64 / (1u64 << 31) as f64;
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn ar1(n: usize, rho: f64, sd: f64, gauss: &mut impl FnMut() -> f64) -> Vec<f64> {
    let innovation = sd * (1.0 - rho * rho).sqrt();
    let mut prev = sd * gauss();
    (0..n)
        .map(|_| {
            prev = rho * prev + innovation * gauss();
            prev
        })
        .collect()
}

/// `x` AR(1)(`rho`, SD 1), `u` AR(1)(`rho`, SD 0.5), `y[t] = B x[t-1] + u[t]`.
fn series(n: usize, rho: f64, seed: u64) -> TimeSeriesData {
    let mut gauss = gaussian(seed);
    let x = ar1(n, rho, 1.0, &mut gauss);
    let u = ar1(n, rho, 0.5, &mut gauss);
    let mut y = vec![0.0; n];
    for t in 1..n {
        y[t] = B * x[t - 1] + u[t];
    }
    TimeSeriesData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())], 1).unwrap()
}

fn column(data: &TimeSeriesData, raw: u32) -> Vec<f64> {
    let ColumnView::Float64(col) = data.column(VariableId::from_raw(raw)).unwrap() else {
        panic!("float column");
    };
    col.values.to_vec()
}

fn slope(pairs: impl Iterator<Item = (f64, f64)>) -> Option<f64> {
    let (mut n, mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for (x, y) in pairs {
        n += 1.0;
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }
    let den = n * sxx - sx * sx;
    (n > 2.0 && den.abs() > 1e-12).then(|| (n * sxy - sx * sy) / den)
}

/// Lag-aligned `(x[t-1], y[t])` rows, `t = 1..n` (design row `r` is time `r + 1`).
fn aligned_pairs(data: &TimeSeriesData) -> Vec<(f64, f64)> {
    let x = column(data, 0);
    let y = column(data, 1);
    (1..x.len()).map(|t| (x[t - 1], y[t])).collect()
}

/// The block length production uses for this atom on `pairs`:
/// [`mixture_block_length`] over the slope's Frisch–Waugh influence
/// (`(x − x̄)·e`, up to a scale the length does not depend on), the single-atom
/// mixture score (the same series) and the `[1, x]` regression's
/// normal-equation scores.
fn production_block_length(pairs: &[(f64, f64)], structural_span: usize) -> (usize, Vec<f64>) {
    let rows = pairs.len();
    let mut matrix = vec![1.0; rows];
    matrix.extend(pairs.iter().map(|p| p.0));
    let outcome: Vec<f64> = pairs.iter().map(|p| p.1).collect();
    let normal = antecedent_estimate::normal_equation_scores(&matrix, rows, 2, &outcome).unwrap();
    // normal[0] is the residual series (the intercept's score).
    let x_mean = pairs.iter().map(|p| p.0).sum::<f64>() / rows as f64;
    let influence: Vec<f64> =
        pairs.iter().zip(&normal[0]).map(|(p, e)| (p.0 - x_mean) * e).collect();
    let normal_refs: Vec<&[f64]> = normal.iter().map(Vec::as_slice).collect();
    let length =
        mixture_block_length(structural_span, rows, &[&influence], Some(&influence), &normal_refs);
    (length, influence)
}

struct Tally {
    covered: u32,
    scored: u32,
    with_interval: u32,
    length: f64,
}

impl Tally {
    const fn new() -> Self {
        Self { covered: 0, scored: 0, with_interval: 0, length: 0.0 }
    }

    fn record(&mut self, est: f64, se: f64) {
        self.scored += 1;
        if est.is_finite() && se.is_finite() && se > 0.0 {
            self.with_interval += 1;
            self.length += 2.0 * Z90 * se;
            if (est - B).abs() <= Z90 * se {
                self.covered += 1;
            }
        }
    }

    fn rate(&self) -> f64 {
        f64::from(self.covered) / f64::from(self.scored)
    }

    /// Mean interval length over replicates that produced an interval.
    fn mean_length(&self) -> f64 {
        if self.with_interval == 0 { f64::NAN } else { self.length / f64::from(self.with_interval) }
    }

    fn band(&self) -> (f64, f64) {
        let mcse = (0.9 * 0.1 / f64::from(self.scored)).sqrt();
        (0.9 - 3.0 * mcse, 0.9 + 3.0 * mcse)
    }
}

fn sensitivity(label: &str, rho: f64, n: usize, seed_base: u64) {
    // Pulse at lag 1, horizon 1: history + horizon = 2, as `temporal_class_block_span`,
    // over the n - 1 lag-aligned rows.
    let rows = n - 1;
    let rule = circular_block_length(2, rows);
    let factors = [0.5, 1.0, 2.0];
    let mut tallies = [Tally::new(), Tally::new(), Tally::new()];
    let mut production = Vec::new();
    let design = [AlignedRows { first_time: 1, rows }];
    for s in 0..n_sim() {
        // Fresh resampling stream per replicate, so bootstrap noise averages out.
        let ctx = ExecutionContext::for_tests(seed_base + u64::from(s));
        let data = series(n, rho, seed_base + u64::from(s));
        let pairs = aligned_pairs(&data);
        let est = slope(pairs.iter().copied()).unwrap();
        let (length, influence) = production_block_length(&pairs, 2);
        production.push(length);
        for (factor, tally) in factors.iter().zip(&mut tallies) {
            let scaled = ((length as f64 * factor).round() as usize).clamp(1, rows);
            // The production kernel-bias factor of the influence at the scaled length.
            let kernel_bias = antecedent_estimate::kernel_bias_scale(&[&influence], scaled);
            let block = shared_circular_block_mixture_se_with_length(
                &design,
                &[1.0],
                scaled,
                kernel_bias,
                REPLICATES,
                0xB10C_0000,
                &ctx,
                |_, rows| slope(rows.iter().map(|&r| pairs[r])),
            );
            tally.record(est, block.se);
        }
    }
    production.sort_unstable();
    let (min, median, max) =
        (production[0], production[production.len() / 2], production[production.len() - 1]);
    let (lo, hi) = tallies[1].band();
    let mut failures = Vec::new();
    for (factor, tally) in factors.iter().zip(&tallies) {
        let rate = tally.rate();
        eprintln!(
            "calibration {label} block x{factor} of the production length \
             (production L min/median/max = {min}/{median}/{max}, rule={rule}): nominal=0.90 \
             coverage={rate:.3} band=[{lo:.3}, {hi:.3}] mean_length={:.4} ({}/{} covered)",
            tally.mean_length(),
            tally.covered,
            tally.scored
        );
        if !(lo..=hi).contains(&rate) {
            failures.push(format!("x{factor}: {rate:.3}"));
        }
    }
    assert!(failures.is_empty(), "{label} block-length sensitivity outside band: {failures:?}");
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn shared_block_length_sensitivity_iid_n160() {
    sensitivity("shared circular block, iid n=160", 0.0, 160, 50_000);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn shared_block_length_sensitivity_ar1_rho05_n160() {
    sensitivity("shared circular block, AR(1) rho=0.5 n=160", 0.5, 160, 51_000);
}

#[test]
fn block_length_rule_matches_documented_formula() {
    assert_eq!(circular_block_length(2, 160), 6);
    assert_eq!(circular_block_length(2, 400), 8);
    assert_eq!(circular_block_length(2, 60), 4);
    assert_eq!(circular_block_length(9, 60), 9);
    assert_eq!(circular_block_length(2, 1), 1);
}
