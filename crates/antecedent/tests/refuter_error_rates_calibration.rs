//! Error rates of the target-native refuters' verdicts, on tabular and series data, frequentist
//! and Bayesian.
//!
//! A refuter publishes a verdict: passed, or rejected at the tolerance it states. Each design
//! below holds the assumption a refuter tests by construction (the known null) and counts how
//! often the verdict is a rejection; that false-rejection rate must be at most the stated
//! tolerance `alpha = 0.05`. It is scored the way the family-wise error of an adjusted p-value
//! is scored elsewhere: a replicate "covers" when the refuter does not reject, at level
//! `1 - alpha`, and the record is held to the one-sided floor (`assert_one_sided`), so a
//! rejection rate below `alpha` passes and one above it does not. A replicate whose refuter
//! publishes no verdict (the run refused or failed) is a skip, which counts as a miss.
//!
//! Each design also states a known violation of a stated size and reports the detection rate
//! (`calibration-power` lines). The detection rate is reported, not asserted: no power
//! requirement is stated for these verdicts, and none is invented here.
//!
//! What each verdict is, and why the null rate is bounded by `alpha` (see
//! `antecedent_validate::target_native::joint`): a replicate refuter passes when the Hotelling
//! tail of `D^2 = (e - m)' S^-1 (e - m)` is at least `alpha`, `e` the expected vector, `m` the
//! replicate mean and `S` the covariance of one replicate estimated from the replicates. The
//! tail has `1 / (1 / (R - 1) + 1 / (g - 1))` covariance degrees of freedom: `R` replicates of
//! a resample built from `g` independent units (rows, or blocks of a block resample), which the
//! refit declares.
//!
//! * Placebo refuters (permutation on rows, circular shift on series) have `e = 0` and replicates
//!   centred at zero when the procedure is sound, so `m` is a mean of `R` replicates and
//!   `R * D^2` is a Hotelling `T^2`; `D^2` is about `k / R` and the rejection rate is far below
//!   `alpha`.
//! * Stability refuters (random common cause, subset, bootstrap) have `e` the original estimate
//!   and replicates centred at it (unbiased procedure): the same bound. A subset window must
//!   keep that centring, so series windows are circular (every period equally often); a window
//!   confined to the series over-weights the middle and is not centred on the original.
//! * Negative-control and pre-trend refuters have `e = 0` and replicates that are resamples of
//!   the data. `m` is then the sample estimate of a truly zero effect and `S` the resampling
//!   spread of one replicate, so `D^2` is the squared standardised estimate on `k` degrees of
//!   freedom: its rejection rate is `alpha` up to the accuracy of the resampling standard
//!   error. These are the refuters whose rate the measurement can fail to hold.
//! * The posterior-predictive check rejects when the share of draws whose replicated
//!   discrepancy reaches the observed one is below `alpha`. When the observed target is what
//!   the model's own posterior centres on, that share concentrates at one half, so the null
//!   rejection rate is about zero. It has power only against an observed target the model's
//!   posterior does not reproduce.
//!
//! No sample-size grid point is tuned: designs scale with `grid_n`, the refits and draws are
//! fixed, and the violations are stated in absolute units and in posterior standard deviations.
//! Records are keyed on the design axes (structure `design`, validation `not_applicable`); the
//! refuters publish no runtime calibration key, so the construction is declared here from what
//! the verdict states (the refuter id is its interval-method field).
//!
//! Ignored tests run via `scripts/measure_calibration.sh`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "row indices, counts and window fractions convert exactly at these sizes"
)]

mod common;

use std::sync::Arc;

use antecedent::{RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, CausalRng, ExecutionContext, Lag, TemporalEffectQuery, TemporalPolicy, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_graph::{TemporalDag, ensure_lagged};
use antecedent_kernels::{shuffle, standard_normal, unbiased_index};
use antecedent_validate::ValidationError;
use antecedent_validate::target_native::{
    PosteriorPredictiveTarget, RefutationTarget, TargetFamily, TargetPerturbation, TargetQuadrant,
    TargetRefit, TargetRefitError, TargetRefutationConfig, TargetRefutationPlan, TargetRefuterKind,
    TargetSuite, posterior_predictive_target_check, run_target_plan,
};
use common::calibration::{
    Construction, CoverageSide, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, grid_n,
    grid_seed, map_replicates, n_sim,
};

/// Tolerance every refuter below is run at: it passes when its reference tail is at least this.
/// The records are at level `1 - ALPHA`.
const ALPHA: f64 = 0.05;

/// Refits of every replicate-based refuter.
const REFITS: u32 = 40;

/// Dependence block of the series refuters (placebo shifts, block bootstrap), in periods.
const BLOCK: usize = 10;

/// Autoregressive coefficient of the series errors (unit innovations, stationary start).
const NOISE_RHO: f64 = 0.5;

/// Effect of the treatment at lag 0 and lag 1 in the series and Gaussian designs.
const LAG0: f64 = 0.5;
const LAG1: f64 = 0.25;

/// Lead of the series pre-trend refuter. The target is the effect at lags 0 and 1, so a lead of
/// two puts the shifted treatment at lags -2 and -1, where the true effect is zero (the
/// treatment is serially independent, so no omitted lag leaks into the shifted regression).
const PRE_TREND_LEAD: u32 = 2;

/// Size of the spurious effect a defective procedure adds to every refit, in outcome units. The
/// replicate standard deviation of a coefficient is about `1 / sqrt(n)` for these designs, so
/// this is about `0.4 * sqrt(n)` standard deviations: 4 at the smallest grid point, 8 at the
/// largest.
const SPURIOUS_EFFECT: f64 = 0.4;

/// Size by which a defective procedure misreports its own target (the original differs from what
/// its refits reproduce), in the same units.
const MISREPORT: f64 = 0.4;

/// Posterior draws of the predictive check.
const POSTERIOR_DRAWS: usize = 100;

/// Misreport of the posterior-predictive violation, in posterior standard deviations of the
/// first coordinate.
const PREDICTIVE_MISREPORT_SDS: f64 = 4.0;

const NULL_TARGET: &str =
    "refuter rejects at most alpha = 0.05 of the time when the assumption it tests holds";

// ------------------------------------------------------------------ numerics

/// Solve `a x = b` (`p x p`, row-major) by Gaussian elimination with partial pivoting.
fn solve(mut a: Vec<f64>, mut b: Vec<f64>, p: usize) -> Option<Vec<f64>> {
    for col in 0..p {
        let pivot = (col..p)
            .max_by(|&i, &j| a[i * p + col].abs().total_cmp(&a[j * p + col].abs()))
            .unwrap_or(col);
        if a[pivot * p + col].abs() < 1e-12 {
            return None;
        }
        if pivot != col {
            for k in 0..p {
                a.swap(col * p + k, pivot * p + k);
            }
            b.swap(col, pivot);
        }
        for row in col + 1..p {
            let factor = a[row * p + col] / a[col * p + col];
            for k in col..p {
                a[row * p + k] -= factor * a[col * p + k];
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = vec![0.0; p];
    for row in (0..p).rev() {
        let tail: f64 = (row + 1..p).map(|k| a[row * p + k] * x[k]).sum();
        x[row] = (b[row] - tail) / a[row * p + row];
    }
    Some(x)
}

/// Least-squares coefficients of `y` on `columns` (each of the length of `y`).
fn coefficients(columns: &[Vec<f64>], y: &[f64]) -> Option<Vec<f64>> {
    let p = columns.len();
    let mut xtx = vec![0.0; p * p];
    let mut xty = vec![0.0; p];
    for i in 0..p {
        for j in 0..=i {
            let s: f64 = columns[i].iter().zip(&columns[j]).map(|(a, b)| a * b).sum();
            xtx[i * p + j] = s;
            xtx[j * p + i] = s;
        }
        xty[i] = columns[i].iter().zip(y).map(|(a, b)| a * b).sum();
    }
    solve(xtx, xty, p)
}

fn gather(values: &[f64], rows: &[usize]) -> Vec<f64> {
    rows.iter().map(|&i| values[i]).collect()
}

fn normals(m: usize, rng: &mut CausalRng) -> Vec<f64> {
    (0..m).map(|_| standard_normal(rng)).collect()
}

/// A stationary AR(1) series with unit innovations.
fn ar1(m: usize, rho: f64, rng: &mut CausalRng) -> Vec<f64> {
    let mut out = Vec::with_capacity(m);
    let mut previous = 0.0;
    for i in 0..m {
        let z = standard_normal(rng);
        previous = if i == 0 { z / (1.0 - rho * rho).sqrt() } else { rho * previous + z };
        out.push(previous);
    }
    out
}

fn resample_rows(n: usize, rng: &mut CausalRng) -> Vec<usize> {
    (0..n).map(|_| unbiased_index(rng, n)).collect()
}

/// A defective procedure: it adds `refit_shift` to every refit it returns and reports an
/// original that is `original_shift` away from what an honest fit gives. The sound procedure
/// has neither.
#[derive(Clone, Copy)]
struct Defect {
    refit_shift: f64,
    original_shift: f64,
}

const SOUND: Defect = Defect { refit_shift: 0.0, original_shift: 0.0 };
const SPURIOUS: Defect = Defect { refit_shift: SPURIOUS_EFFECT, original_shift: 0.0 };
const MISREPORTED: Defect = Defect { refit_shift: 0.0, original_shift: MISREPORT };

fn unsupported(reason: &str) -> TargetRefitError {
    TargetRefitError::Unsupported {
        reason: Arc::from(reason),
        next_attempt: Arc::from("declare the construction this design does not have"),
    }
}

fn singular() -> TargetRefitError {
    TargetRefitError::Failed(ValidationError::NotApplicable {
        message: "the adjustment regression is singular",
    })
}

// -------------------------------------------------- static frequentist procedure

/// Confounded tabular data with two outcomes, a negative-control exposure and two
/// negative-control outcomes. `X` confounds the treatment, the control exposure and every
/// outcome; the control exposure has no effect on anything and the treatment has no effect on the
/// control outcomes, so adjusting for `X` gives a zero coefficient for each control by
/// construction.
struct StaticData {
    x: Vec<f64>,
    treatment: Vec<f64>,
    control_exposure: Vec<f64>,
    outcome: [Vec<f64>; 2],
    control_outcome: [Vec<f64>; 2],
}

fn static_data(n: usize, seed: u64) -> StaticData {
    let mut rng = CausalRng::from_seed(seed);
    let mut data = StaticData {
        x: Vec::with_capacity(n),
        treatment: Vec::with_capacity(n),
        control_exposure: Vec::with_capacity(n),
        outcome: [Vec::with_capacity(n), Vec::with_capacity(n)],
        control_outcome: [Vec::with_capacity(n), Vec::with_capacity(n)],
    };
    for _ in 0..n {
        let x = standard_normal(&mut rng);
        let t = 0.5 * x + standard_normal(&mut rng);
        data.x.push(x);
        data.treatment.push(t);
        data.control_exposure.push(0.5 * x + standard_normal(&mut rng));
        data.outcome[0].push(0.5 * t + x + standard_normal(&mut rng));
        data.outcome[1].push(-0.3 * t + 0.5 * x + standard_normal(&mut rng));
        data.control_outcome[0].push(x + standard_normal(&mut rng));
        data.control_outcome[1].push(0.5 * x + standard_normal(&mut rng));
    }
    data
}

impl StaticData {
    /// Coefficient of `exposure` in `outcome ~ 1 + exposure + X (+ extra)` for each outcome, on
    /// `rows`.
    fn effects(
        &self,
        rows: &[usize],
        exposure: &[f64],
        outcomes: [&[f64]; 2],
        extra: Option<&[f64]>,
    ) -> Option<Vec<f64>> {
        let mut columns =
            vec![vec![1.0; rows.len()], gather(exposure, rows), gather(&self.x, rows)];
        if let Some(extra) = extra {
            columns.push(gather(extra, rows));
        }
        let mut out = Vec::with_capacity(2);
        for outcome in outcomes {
            out.push(coefficients(&columns, &gather(outcome, rows))?[1]);
        }
        Some(out)
    }
}

fn static_target() -> RefutationTarget {
    RefutationTarget::new(
        TargetFamily::JointContrast,
        TargetQuadrant::STATIC_FREQUENTIST,
        ["effect_on_first_outcome", "effect_on_second_outcome"],
        [1; 32],
    )
    .expect("two labelled coordinates")
}

struct StaticRefit {
    data: StaticData,
    target: RefutationTarget,
    original: Vec<f64>,
    defect: Defect,
}

impl StaticRefit {
    fn new(data: StaticData, defect: Defect) -> Option<Self> {
        let rows: Vec<usize> = (0..data.x.len()).collect();
        let mut original =
            data.effects(&rows, &data.treatment, [&data.outcome[0], &data.outcome[1]], None)?;
        for value in &mut original {
            *value += defect.original_shift;
        }
        Some(Self { data, target: static_target(), original, defect })
    }
}

impl TargetRefit for StaticRefit {
    fn target(&self) -> &RefutationTarget {
        &self.target
    }

    fn independent_resampling_units(&self) -> Option<usize> {
        // Rows resampled independently.
        Some(self.data.x.len())
    }

    fn original(&self) -> &[f64] {
        &self.original
    }

    fn refit(
        &self,
        perturbation: TargetPerturbation,
        rng: &mut CausalRng,
        _ctx: &ExecutionContext,
    ) -> Result<Vec<f64>, TargetRefitError> {
        let d = &self.data;
        let n = d.x.len();
        let all: Vec<usize> = (0..n).collect();
        let ys = [d.outcome[0].as_slice(), d.outcome[1].as_slice()];
        let fitted = match perturbation {
            TargetPerturbation::PlaceboTreatment { .. } => {
                let mut treatment = d.treatment.clone();
                shuffle(rng, &mut treatment);
                d.effects(&all, &treatment, ys, None)
            }
            TargetPerturbation::PlaceboOutcome { .. } => {
                let mut perm = all.clone();
                shuffle(rng, &mut perm);
                let first = gather(ys[0], &perm);
                let second = gather(ys[1], &perm);
                d.effects(&all, &d.treatment, [&first, &second], None)
            }
            TargetPerturbation::RandomCommonCause { .. } => {
                let cause = normals(n, rng);
                d.effects(&all, &d.treatment, ys, Some(&cause))
            }
            TargetPerturbation::Subset { fraction, .. } => {
                let mut rows = all.clone();
                shuffle(rng, &mut rows);
                rows.truncate((fraction * n as f64) as usize);
                d.effects(&rows, &d.treatment, ys, None)
            }
            TargetPerturbation::Bootstrap { .. } => {
                let rows = resample_rows(n, rng);
                d.effects(&rows, &d.treatment, ys, None)
            }
            TargetPerturbation::NegativeControlExposure { .. } => {
                let rows = resample_rows(n, rng);
                d.effects(&rows, &d.control_exposure, ys, None)
            }
            TargetPerturbation::NegativeControlOutcome { .. } => {
                let rows = resample_rows(n, rng);
                d.effects(&rows, &d.treatment, [&d.control_outcome[0], &d.control_outcome[1]], None)
            }
            _ => return Err(unsupported("the static design has no such refuter")),
        };
        let mut out = fitted.ok_or_else(singular)?;
        for value in &mut out {
            *value += self.defect.refit_shift;
        }
        Ok(out)
    }
}

// -------------------------------------------------- temporal frequentist procedure

/// A series with a serially independent treatment and autoregressive outcome errors:
/// `Y_t = LAG0 * T_t + LAG1 * T_{t-1} + e_t`. The target is the effect at lag 0 and lag 1.
struct SeriesData {
    treatment: Vec<f64>,
    outcome: Vec<f64>,
}

fn series_data(n: usize, seed: u64) -> SeriesData {
    let mut rng = CausalRng::from_seed(seed);
    let treatment = normals(n, &mut rng);
    let noise = ar1(n, NOISE_RHO, &mut rng);
    let outcome = (0..n)
        .map(|s| LAG0 * treatment[s] + if s > 0 { LAG1 * treatment[s - 1] } else { 0.0 } + noise[s])
        .collect();
    SeriesData { treatment, outcome }
}

fn circular_shift(values: &[f64], offset: usize) -> Vec<f64> {
    let n = values.len();
    (0..n).map(|i| values[(i + offset) % n]).collect()
}

/// Periods usable when the treatment is read `lead` periods ahead.
fn periods(n: usize, lead: usize) -> Vec<usize> {
    (1..n - lead).collect()
}

/// Circular block resample of `times` with blocks of [`BLOCK`] consecutive entries.
fn block_resample(times: &[usize], rng: &mut CausalRng) -> Vec<usize> {
    let m = times.len();
    let mut out = Vec::with_capacity(m);
    while out.len() < m {
        let start = unbiased_index(rng, m);
        for j in 0..BLOCK {
            if out.len() == m {
                break;
            }
            out.push(times[(start + j) % m]);
        }
    }
    out
}

impl SeriesData {
    fn len(&self) -> usize {
        self.outcome.len()
    }

    /// Coefficients of the treatment read `lead` periods ahead at lags 0 and 1 in
    /// `outcome ~ 1 + T_{t+lead} + T_{t+lead-1} (+ extra_t)`, over `times`.
    fn effects(
        &self,
        times: &[usize],
        lead: usize,
        treatment: &[f64],
        outcome: &[f64],
        extra: Option<&[f64]>,
    ) -> Option<Vec<f64>> {
        let mut columns = vec![
            vec![1.0; times.len()],
            times.iter().map(|&s| treatment[s + lead]).collect(),
            times.iter().map(|&s| treatment[s + lead - 1]).collect(),
        ];
        if let Some(extra) = extra {
            columns.push(gather(extra, times));
        }
        let coefficients = coefficients(&columns, &gather(outcome, times))?;
        Some(vec![coefficients[1], coefficients[2]])
    }

    fn placebo_offset(&self, rng: &mut CausalRng) -> usize {
        BLOCK + unbiased_index(rng, self.len() - 2 * BLOCK + 1)
    }
}

fn series_target() -> RefutationTarget {
    RefutationTarget::new(
        TargetFamily::TemporalEffect,
        TargetQuadrant::TEMPORAL_FREQUENTIST,
        ["effect_at_lag_0", "effect_at_lag_1"],
        [2; 32],
    )
    .expect("two labelled coordinates")
}

struct SeriesRefit {
    data: SeriesData,
    target: RefutationTarget,
    original: Vec<f64>,
    defect: Defect,
}

impl SeriesRefit {
    fn new(data: SeriesData, defect: Defect) -> Option<Self> {
        let times = periods(data.len(), 0);
        let mut original = data.effects(&times, 0, &data.treatment, &data.outcome, None)?;
        for value in &mut original {
            *value += defect.original_shift;
        }
        Some(Self { data, target: series_target(), original, defect })
    }
}

impl TargetRefit for SeriesRefit {
    fn target(&self) -> &RefutationTarget {
        &self.target
    }

    fn independent_resampling_units(&self) -> Option<usize> {
        // Circular blocks of `BLOCK` periods over the usable periods.
        Some(periods(self.data.len(), 0).len() / BLOCK)
    }

    fn original(&self) -> &[f64] {
        &self.original
    }

    fn refit(
        &self,
        perturbation: TargetPerturbation,
        rng: &mut CausalRng,
        _ctx: &ExecutionContext,
    ) -> Result<Vec<f64>, TargetRefitError> {
        let d = &self.data;
        let n = d.len();
        let times = periods(n, 0);
        let fitted = match perturbation {
            TargetPerturbation::PlaceboTreatment { .. } => {
                let shifted = circular_shift(&d.treatment, d.placebo_offset(rng));
                d.effects(&times, 0, &shifted, &d.outcome, None)
            }
            TargetPerturbation::PlaceboOutcome { .. } => {
                let shifted = circular_shift(&d.outcome, d.placebo_offset(rng));
                d.effects(&times, 0, &d.treatment, &shifted, None)
            }
            TargetPerturbation::RandomCommonCause { .. } => {
                let cause = normals(n, rng);
                d.effects(&times, 0, &d.treatment, &d.outcome, Some(&cause))
            }
            TargetPerturbation::Subset { fraction, .. } => {
                // A circular window of design rows, each keeping its own lag history: the
                // construction of a route that reads row weights (zero weight outside the
                // window). Every period lies in a window with the same probability, so the
                // window estimates centre on the full-series estimate; a window that must lie
                // inside the series over-weights the middle periods, and its replicate mean
                // then differs from the original by about one replicate spread.
                let m = times.len();
                let len = (fraction * m as f64) as usize;
                let start = unbiased_index(rng, m);
                let window: Vec<usize> = (0..len).map(|j| times[(start + j) % m]).collect();
                d.effects(&window, 0, &d.treatment, &d.outcome, None)
            }
            TargetPerturbation::Bootstrap { .. } => {
                let resampled = block_resample(&times, rng);
                d.effects(&resampled, 0, &d.treatment, &d.outcome, None)
            }
            TargetPerturbation::PreTrend { lead, .. } => {
                let lead = usize::try_from(lead).expect("lead fits usize");
                let resampled = block_resample(&periods(n, lead), rng);
                d.effects(&resampled, lead, &d.treatment, &d.outcome, None)
            }
            _ => return Err(unsupported("the series design declares no such construction")),
        };
        let mut out = fitted.ok_or_else(singular)?;
        for value in &mut out {
            *value += self.defect.refit_shift;
        }
        Ok(out)
    }
}

// ------------------------------------------------------- Gaussian posterior model

/// Linear model `y = a * c0 + b * c1 + e` with Gaussian errors of unit innovation variance
/// (independent rows when `rho = 0`, an AR(1) series otherwise), a flat prior and the error law
/// known. The posterior of the coefficients is exactly Gaussian,
/// `N(estimate(y), (W'W)^-1)` for the whitened design `W`, so the draws below are exact
/// posterior draws and data replicated from a draw is exactly the model's own predictive law.
struct GaussianModel {
    rho: f64,
    columns: [Vec<f64>; 2],
    whitened: [Vec<f64>; 2],
    inverse: [f64; 4],
}

/// Prais-Winsten whitening of a series: unit-variance independent errors for an AR(1) series.
fn whiten(values: &[f64], rho: f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(values.len());
    for (i, v) in values.iter().enumerate() {
        out.push(if i == 0 { (1.0 - rho * rho).sqrt() * v } else { v - rho * values[i - 1] });
    }
    out
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

impl GaussianModel {
    fn new(columns: [Vec<f64>; 2], rho: f64) -> Self {
        let whitened = [whiten(&columns[0], rho), whiten(&columns[1], rho)];
        let (a11, a12, a22) = (
            dot(&whitened[0], &whitened[0]),
            dot(&whitened[0], &whitened[1]),
            dot(&whitened[1], &whitened[1]),
        );
        let det = a11 * a22 - a12 * a12;
        let inverse = [a22 / det, -a12 / det, -a12 / det, a11 / det];
        Self { rho, columns, whitened, inverse }
    }

    /// Generalised-least-squares coefficients: the target the model reports.
    fn estimate(&self, y: &[f64]) -> [f64; 2] {
        let wy = whiten(y, self.rho);
        let r = [dot(&self.whitened[0], &wy), dot(&self.whitened[1], &wy)];
        [
            self.inverse[0] * r[0] + self.inverse[1] * r[1],
            self.inverse[2] * r[0] + self.inverse[3] * r[1],
        ]
    }

    /// Lower Cholesky factor `[l11, l21, l22]` of the posterior covariance.
    fn cholesky(&self) -> [f64; 3] {
        let l11 = self.inverse[0].sqrt();
        let l21 = self.inverse[2] / l11;
        [l11, l21, (self.inverse[3] - l21 * l21).sqrt()]
    }

    fn response(&self, coefficients: [f64; 2], noise: &[f64]) -> Vec<f64> {
        (0..noise.len())
            .map(|i| {
                coefficients[0] * self.columns[0][i]
                    + coefficients[1] * self.columns[1][i]
                    + noise[i]
            })
            .collect()
    }
}

fn gaussian_target(temporal: bool) -> RefutationTarget {
    let (family, quadrant, labels) = if temporal {
        (
            TargetFamily::TemporalEffect,
            TargetQuadrant::TEMPORAL_BAYESIAN,
            ["effect_at_lag_0", "effect_at_lag_1"],
        )
    } else {
        (
            TargetFamily::JointContrast,
            TargetQuadrant::STATIC_BAYESIAN,
            ["effect_of_first_regressor", "effect_of_second_regressor"],
        )
    };
    RefutationTarget::new(family, quadrant, labels, [if temporal { 4 } else { 3 }; 32])
        .expect("two labelled coordinates")
}

/// The posterior-predictive verdict of one dataset: data from the model at its true coefficients
/// (`LAG0`, `LAG1`), exact posterior draws, data replicated from each draw, and the target the
/// point procedure recomputes on each. `misreport_sds` moves the observed target's first
/// coordinate away from what the procedure computes on the data, in posterior standard
/// deviations of that coordinate; zero is the correct model.
fn predictive_verdict(temporal: bool, n: usize, seed: u64, misreport_sds: f64) -> Option<bool> {
    let mut rng = CausalRng::from_seed(seed);
    let (rho, columns) = if temporal {
        let treatment = normals(n + 1, &mut rng);
        (NOISE_RHO, [treatment[1..].to_vec(), treatment[..n].to_vec()])
    } else {
        (0.0, [normals(n, &mut rng), normals(n, &mut rng)])
    };
    let model = GaussianModel::new(columns, rho);
    let noise = ar1(n, rho, &mut rng);
    let y = model.response([LAG0, LAG1], &noise);
    let centre = model.estimate(&y);
    let l = model.cholesky();
    let mut draws = Vec::with_capacity(POSTERIOR_DRAWS * 2);
    let mut replicated = Vec::with_capacity(POSTERIOR_DRAWS * 2);
    for _ in 0..POSTERIOR_DRAWS {
        let (z0, z1) = (standard_normal(&mut rng), standard_normal(&mut rng));
        let theta = [centre[0] + l[0] * z0, centre[1] + l[1] * z0 + l[2] * z1];
        let replicate_noise = ar1(n, rho, &mut rng);
        let rep = model.estimate(&model.response(theta, &replicate_noise));
        draws.extend_from_slice(&theta);
        replicated.extend_from_slice(&rep);
    }
    let observed = [centre[0] + misreport_sds * l[0], centre[1]];
    let target = gaussian_target(temporal);
    posterior_predictive_target_check(&PosteriorPredictiveTarget {
        target: &target,
        observed: &observed,
        draws: &draws,
        replicated: &replicated,
        alpha: ALPHA,
    })
    .ok()
    .map(|report| report.passed)
}

// ------------------------------------------------------------------ scoring

/// Per-refuter verdicts of one replicate, in the order of the scenario's `kinds`: `None` for a
/// refuter that published no verdict. The outer `None` is a run that failed outright.
type Verdicts = Option<Vec<Option<bool>>>;

/// What a design measures: its axes, the refuters it scores and its scope facts.
struct Scenario {
    test: &'static str,
    dgp: &'static str,
    bayesian: bool,
    temporal: bool,
    kinds: &'static [TargetRefuterKind],
    refits: Option<u32>,
    draws: Option<u32>,
}

impl Scenario {
    fn construction(&self, kind: TargetRefuterKind) -> Construction {
        Construction {
            query: "TargetRefutation".into(),
            graph_class: "TargetNative".into(),
            structure: "design".into(),
            modality: if self.temporal { "time_series" } else { "tabular" }.into(),
            inference: if self.bayesian { "Bayesian" } else { "Frequentist" }.into(),
            estimator: if self.bayesian { "gaussian_conjugate" } else { "linear_adjustment" }
                .into(),
            interval_method: kind.as_str().into(),
            se_kind: String::new(),
            dependence: if self.temporal { "block" } else { "iid" }.into(),
            posterior: if self.bayesian { "gaussian_known_error_law" } else { "" }.into(),
            functional: if self.temporal { "temporal_effect" } else { "joint_contrast" }.into(),
            identification: "point".into(),
            reported_level: REPORTED_LEVEL,
        }
    }

    fn scope(&self, rows: usize) -> ScopeFacts {
        ScopeFacts {
            row_count: rows as u64,
            replicates_ok: self.refits,
            posterior_draws: self.draws,
            unidentified_mass: 0.0,
        }
    }

    /// One record per refuter: a replicate covers when the refuter does not reject. Asserted
    /// against the one-sided floor at level `1 - ALPHA`, every refuter reported together.
    fn false_rejection(&self, rows: usize, runs: &[Verdicts]) {
        assert!((REPORTED_LEVEL - (1.0 - ALPHA)).abs() < 1e-12, "records are at level 1 - alpha");
        let mut tallies: Vec<CoverageTally> = self
            .kinds
            .iter()
            .map(|kind| {
                CoverageTally::for_record(
                    RecordKey { test: self.test, dgp: self.dgp, interval: kind.as_str() },
                    REPORTED_LEVEL,
                )
                .labelled(kind.as_str())
            })
            .collect();
        for run in runs {
            for (i, tally) in tallies.iter_mut().enumerate() {
                match run.as_ref().and_then(|verdicts| verdicts[i]) {
                    None => tally.skip(),
                    Some(passed) => {
                        tally.bind_validation(
                            &self.construction(self.kinds[i]),
                            self.scope(rows),
                            "not_applicable",
                        );
                        tally.absorb(&passed);
                        tally.record(Some(if passed { (-1.0, 1.0) } else { (1.0, 2.0) }), 0.0);
                    }
                }
            }
        }
        let failures: Vec<String> = tallies
            .iter()
            .filter_map(|tally| {
                std::panic::catch_unwind(|| {
                    tally.assert_one_sided(CoverageSide::Upper, NULL_TARGET);
                })
                .err()
                .map(|error| {
                    error
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| error.downcast_ref::<&str>().map(|text| (*text).to_owned()))
                        .unwrap_or_else(|| "coverage failure".into())
                })
            })
            .collect();
        assert!(failures.is_empty(), "{}: {}", self.test, failures.join("; "));
    }

    /// Report the detection rate of each refuter against the stated violation; nothing is
    /// asserted. A replicate that published no verdict is not a detection. The tally keeps no
    /// record; it persists its state so an extended run counts every replicate.
    fn power(&self, rows: usize, violation: &str, runs: &[Verdicts]) {
        for (i, kind) in self.kinds.iter().enumerate() {
            let mut tally =
                CoverageTally::new(format!("{} [{}]", self.test, kind.as_str()), REPORTED_LEVEL);
            for run in runs {
                match run.as_ref().and_then(|verdicts| verdicts[i]) {
                    None => tally.skip(),
                    Some(passed) => {
                        tally.record(Some(if passed { (1.0, 2.0) } else { (-1.0, 1.0) }), 0.0);
                    }
                }
            }
            tally.persist();
            eprintln!(
                "calibration-power test={} refuter={} violation=\"{violation}\" rows={rows} \
                 detected={}/{} power={:.3}",
                self.test,
                kind.as_str(),
                tally.covered(),
                tally.attempts(),
                tally.rate()
            );
        }
    }
}

/// Run the full target-native plan on `refit` and pick `kinds`' verdicts.
fn plan_verdicts(
    refit: &dyn TargetRefit,
    kinds: &[TargetRefuterKind],
    config: &TargetRefutationConfig,
    seed: u64,
) -> Verdicts {
    let plan = TargetRefutationPlan::for_target(refit.target().clone(), TargetSuite::Full);
    let run = run_target_plan(&plan, refit, config, &ExecutionContext::for_tests(seed)).ok()?;
    Some(
        kinds
            .iter()
            .map(|kind| run.reports.iter().find(|r| r.kind == *kind).map(|r| r.passed))
            .collect(),
    )
}

fn static_verdicts(n: usize, seed: u64, defect: Defect, kinds: &[TargetRefuterKind]) -> Verdicts {
    let refit = StaticRefit::new(static_data(n, seed), defect)?;
    let config = TargetRefutationConfig { replicates: REFITS, alpha: ALPHA, ..Default::default() };
    plan_verdicts(&refit, kinds, &config, seed ^ 0x5EED_0001)
}

fn series_verdicts(n: usize, seed: u64, defect: Defect, kinds: &[TargetRefuterKind]) -> Verdicts {
    let refit = SeriesRefit::new(series_data(n, seed), defect)?;
    let config = TargetRefutationConfig {
        replicates: REFITS,
        alpha: ALPHA,
        pre_trend_lead: PRE_TREND_LEAD,
        ..Default::default()
    };
    plan_verdicts(&refit, kinds, &config, seed ^ 0x5EED_0002)
}

fn predictive_verdicts(temporal: bool, n: usize, seed: u64, misreport_sds: f64) -> Verdicts {
    Some(vec![predictive_verdict(temporal, n, seed, misreport_sds)])
}

use TargetRefuterKind as K;

/// Refuters whose expectation is a zero target: a spurious effect in a refit moves them.
const STATIC_NULL_KINDS: &[K] = &[
    K::PlaceboTreatment,
    K::PlaceboOutcome,
    K::NegativeControlExposure,
    K::NegativeControlOutcome,
];
/// Refuters that compare the refits with the original: a misreported original moves them.
const STABILITY_KINDS: &[K] = &[K::RandomCommonCause, K::SubsetStability, K::BootstrapStability];
const STATIC_ALL_KINDS: &[K] = &[
    K::PlaceboTreatment,
    K::PlaceboOutcome,
    K::RandomCommonCause,
    K::SubsetStability,
    K::BootstrapStability,
    K::NegativeControlExposure,
    K::NegativeControlOutcome,
];
const SERIES_NULL_KINDS: &[K] = &[K::PlaceboTreatment, K::PlaceboOutcome, K::PreTrend];
const SERIES_ALL_KINDS: &[K] = &[
    K::PlaceboTreatment,
    K::PlaceboOutcome,
    K::RandomCommonCause,
    K::SubsetStability,
    K::BootstrapStability,
    K::PreTrend,
];
const PREDICTIVE_KINDS: &[K] = &[K::PosteriorPredictive];

const STATIC_ROWS: usize = 200;
const SERIES_ROWS: usize = 240;

fn static_scenario(test: &'static str, kinds: &'static [K]) -> Scenario {
    Scenario {
        test,
        dgp: "static_data",
        bayesian: false,
        temporal: false,
        kinds,
        refits: Some(REFITS),
        draws: None,
    }
}

fn series_scenario(test: &'static str, kinds: &'static [K]) -> Scenario {
    Scenario {
        test,
        dgp: "series_data",
        bayesian: false,
        temporal: true,
        kinds,
        refits: Some(REFITS),
        draws: None,
    }
}

fn predictive_scenario(test: &'static str, temporal: bool) -> Scenario {
    Scenario {
        test,
        dgp: "predictive_verdict",
        bayesian: true,
        temporal,
        kinds: PREDICTIVE_KINDS,
        refits: None,
        draws: Some(POSTERIOR_DRAWS as u32),
    }
}

fn measure_static(scenario: &Scenario, seed_base: u64, defect: Defect) -> (usize, Vec<Verdicts>) {
    let n = grid_n(STATIC_ROWS);
    let kinds = scenario.kinds;
    (
        n,
        map_replicates(n_sim(), |rep| {
            static_verdicts(n, grid_seed(seed_base + rep), defect, kinds)
        }),
    )
}

fn measure_series(scenario: &Scenario, seed_base: u64, defect: Defect) -> (usize, Vec<Verdicts>) {
    let n = grid_n(SERIES_ROWS);
    let kinds = scenario.kinds;
    (
        n,
        map_replicates(n_sim(), |rep| {
            series_verdicts(n, grid_seed(seed_base + rep), defect, kinds)
        }),
    )
}

fn measure_predictive(
    scenario: &Scenario,
    seed_base: u64,
    misreport_sds: f64,
) -> (usize, Vec<Verdicts>) {
    let n = grid_n(if scenario.temporal { SERIES_ROWS } else { STATIC_ROWS });
    let temporal = scenario.temporal;
    (
        n,
        map_replicates(n_sim(), |rep| {
            predictive_verdicts(temporal, n, grid_seed(seed_base + rep), misreport_sds)
        }),
    )
}

// ------------------------------------------------------- static frequentist tests
//
// Known null. The procedure is sound: an adjusted regression of two outcomes on the treatment
// and the confounder, whose data satisfy every assumption each refuter tests (the permuted
// treatment or outcome is independent of the rest; a random cause added to the adjustment is
// independent; the control exposure and control outcomes have a true adjusted effect of zero).
// Level: `alpha = 0.05`, rate at most `alpha`.
//
// Known violations. A procedure that adds a spurious effect of 0.4 to every refit (what a
// refuter on zero targets must see), and one that reports an original 0.4 away from what its
// refits reproduce (what a stability refuter must see). Both are 4 to 8 replicate standard
// deviations over the grid.

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn static_frequentist_refuters_false_rejection_under_a_sound_procedure() {
    let scenario = static_scenario(
        "static_frequentist_refuters_false_rejection_under_a_sound_procedure",
        STATIC_ALL_KINDS,
    );
    let (n, runs) = measure_static(&scenario, 0x7F10_0000, SOUND);
    scenario.false_rejection(n, &runs);
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn static_frequentist_refuters_power_against_a_spurious_effect() {
    let scenario = static_scenario(
        "static_frequentist_refuters_power_against_a_spurious_effect",
        STATIC_NULL_KINDS,
    );
    let (n, runs) = measure_static(&scenario, 0x7F11_0000, SPURIOUS);
    scenario.power(n, "refits carry a spurious effect of 0.4", &runs);
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn static_frequentist_refuters_power_against_a_misreported_target() {
    let scenario = static_scenario(
        "static_frequentist_refuters_power_against_a_misreported_target",
        STABILITY_KINDS,
    );
    let (n, runs) = measure_static(&scenario, 0x7F12_0000, MISREPORTED);
    scenario.power(n, "the reported target is 0.4 from what the refits reproduce", &runs);
}

// ------------------------------------------------------- temporal frequentist tests
//
// Known null. A series whose treatment is serially independent and whose outcome errors are
// AR(1) with coefficient 0.5; the procedure is the distributed-lag regression at lags 0 and 1,
// which is unbiased. Circular shifts of at least one block (10 periods) leave the shifted
// series independent of the rest; a lead of two periods puts the pre-trend regression at lags
// -2 and -1 where the true effect is zero. Refits are circular block resamples (length 10) or
// circular windows of design rows. Anticipation needs a declared onset and the series has none, so it is
// refused and not measured.

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_frequentist_refuters_false_rejection_under_a_sound_procedure() {
    let scenario = series_scenario(
        "temporal_frequentist_refuters_false_rejection_under_a_sound_procedure",
        SERIES_ALL_KINDS,
    );
    let (n, runs) = measure_series(&scenario, 0x7F20_0000, SOUND);
    scenario.false_rejection(n, &runs);
}

// The same null design through the real facade: a `Study` on series data with a temporal
// pulse effect (`y_t = 0.5 x_{t-1} + AR(1) noise`, `x` serially independent), whose full suite
// refits the compiled program itself. Subset stability and pre-trend are scored as above, from
// the verdicts the study publishes; nothing here is a mock of the refit. The study's own
// default replicate count, block length and pre-trend lead apply. A refuter the route records
// as not run publishes no verdict and counts as a miss.

fn facade_series(n: usize, seed: u64) -> TimeSeriesData {
    let mut rng = CausalRng::from_seed(seed);
    let x = normals(n, &mut rng);
    let noise = ar1(n, NOISE_RHO, &mut rng);
    let y: Vec<f64> =
        (0..n).map(|t| noise[t] + if t > 0 { LAG0 * x[t - 1] } else { 0.0 }).collect();
    TimeSeriesData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())], 1)
        .expect("two float columns")
}

fn facade_verdicts(n: usize, seed: u64, kinds: &[TargetRefuterKind]) -> Verdicts {
    let mut graph = TemporalDag::empty();
    let x1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).ok()?;
    let y0 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).ok()?;
    graph.insert_directed(x1, y0).ok()?;
    let query = CausalQuery::TemporalEffect(
        TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
            .with_horizon_steps(1)
            .with_max_history_lag(Some(1))
            .with_policy(TemporalPolicy::pulse(-1)),
    );
    let result = Study::series(facade_series(n, seed))
        .graph(graph)
        .query(query)
        .refute(RefuteSuite::Full)
        .bootstrap_replicates(0)
        .build()
        .ok()?
        .run(&ExecutionContext::for_tests(seed ^ 0x5EED_0003))
        .ok()?;
    Some(
        kinds
            .iter()
            .map(|kind| {
                result
                    .refutations
                    .iter()
                    .find(|report| report.refuter.as_ref() == kind.as_str())
                    .map(|report| report.passed)
            })
            .collect(),
    )
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_facade_refuters_false_rejection_under_a_sound_procedure() {
    let scenario = Scenario {
        test: "temporal_facade_refuters_false_rejection_under_a_sound_procedure",
        dgp: "series_study_facade",
        bayesian: false,
        temporal: true,
        kinds: &[K::SubsetStability, K::PreTrend],
        refits: None,
        draws: None,
    };
    let n = grid_n(SERIES_ROWS);
    let kinds = scenario.kinds;
    let runs =
        map_replicates(n_sim(), |rep| facade_verdicts(n, grid_seed(0x7F2A_0000 + rep), kinds));
    scenario.false_rejection(n, &runs);
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_frequentist_refuters_power_against_a_spurious_effect() {
    let scenario = series_scenario(
        "temporal_frequentist_refuters_power_against_a_spurious_effect",
        SERIES_NULL_KINDS,
    );
    let (n, runs) = measure_series(&scenario, 0x7F21_0000, SPURIOUS);
    scenario.power(n, "refits carry a spurious effect of 0.4", &runs);
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_frequentist_refuters_power_against_a_misreported_target() {
    let scenario = series_scenario(
        "temporal_frequentist_refuters_power_against_a_misreported_target",
        STABILITY_KINDS,
    );
    let (n, runs) = measure_series(&scenario, 0x7F22_0000, MISREPORTED);
    scenario.power(n, "the reported target is 0.4 from what the refits reproduce", &runs);
}

// ------------------------------------------------------- Bayesian predictive-check tests
//
// Known null. The model is correct: Gaussian errors (independent rows, or AR(1) with the true
// coefficient) with a known error law and a flat prior, so the draws are exact posterior draws
// and the replicated data are the model's own predictive law. The observed target is what the
// point procedure computes on the data. Tolerance `alpha = 0.05` on the sampled tail fraction;
// the check is documented as a diagnostic whose tail is not a calibrated p-value, and its null
// rate is held to at most `alpha`.
//
// Known violation. The observed target is moved 4 posterior standard deviations (of its first
// coordinate) from what the point procedure computes, so the model's posterior does not
// reproduce the target it is asked about.

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn static_bayesian_predictive_check_false_rejection_under_a_correct_model() {
    let scenario = predictive_scenario(
        "static_bayesian_predictive_check_false_rejection_under_a_correct_model",
        false,
    );
    let (n, runs) = measure_predictive(&scenario, 0x7F30_0000, 0.0);
    scenario.false_rejection(n, &runs);
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn static_bayesian_predictive_check_power_against_a_misreported_target() {
    let scenario = predictive_scenario(
        "static_bayesian_predictive_check_power_against_a_misreported_target",
        false,
    );
    let (n, runs) = measure_predictive(&scenario, 0x7F31_0000, PREDICTIVE_MISREPORT_SDS);
    scenario.power(n, "observed target 4 posterior standard deviations off", &runs);
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_bayesian_predictive_check_false_rejection_under_a_correct_model() {
    let scenario = predictive_scenario(
        "temporal_bayesian_predictive_check_false_rejection_under_a_correct_model",
        true,
    );
    let (n, runs) = measure_predictive(&scenario, 0x7F40_0000, 0.0);
    scenario.false_rejection(n, &runs);
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_bayesian_predictive_check_power_against_a_misreported_target() {
    let scenario = predictive_scenario(
        "temporal_bayesian_predictive_check_power_against_a_misreported_target",
        true,
    );
    let (n, runs) = measure_predictive(&scenario, 0x7F41_0000, PREDICTIVE_MISREPORT_SDS);
    scenario.power(n, "observed target 4 posterior standard deviations off", &runs);
}
