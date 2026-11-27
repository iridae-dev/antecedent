//! Saturated two-decision selection-bias and final-outcome pattern-mixture model.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{ExecutionContext, execution::StreamDomain};
use antecedent_stats::{QuantileRule, quantile_sorted};
use serde::{Deserialize, Serialize};

use super::{JointSensitivityError, LongitudinalSensitivitySpec, LongitudinalSubject};

const BOOTSTRAP: StreamDomain = StreamDomain::external(0x4C4A_5345_4E53);

/// A fully specified sensitivity point, ordered by the two treatment departures then MNAR.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LongitudinalSensitivityPoint {
    /// Additive selection-bias contrasts at treatment decisions one and two.
    pub assignment: [f64; 2],
    /// Additive pattern-mixture shift at final outcome assessment.
    pub missingness: f64,
}

/// Contrast and simultaneous interval at a corner of the prespecified box.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LongitudinalSensitivityCorner {
    /// Coordinate of this corner.
    pub departures: LongitudinalSensitivityPoint,
    /// Mean of schedule `11` minus the mean of `00`.
    pub contrast: f64,
    /// Slice of the simultaneous sampling band at this corner.
    pub sampling_interval: [f64; 2],
}

/// The threshold decision established by a range of possible effects.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LongitudinalSensitivityChoice {
    /// The active schedule's advantage is strictly above the minimum required advantage.
    PreferActive,
    /// The control schedule is at least as valuable at every admissible effect.
    PreferControl,
    /// The range permits both decisions.
    Undetermined,
}

impl LongitudinalSensitivityChoice {
    fn from_range(range: [f64; 2], threshold: f64) -> Self {
        if range[0] > threshold {
            Self::PreferActive
        } else if range[1] <= threshold {
            Self::PreferControl
        } else {
            Self::Undetermined
        }
    }
}

/// The robust decision over the sensitivity region and its simultaneous sampling band.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LongitudinalSensitivityDecision {
    /// Active schedule must beat control by strictly more than this contrast.
    pub threshold: f64,
    /// Decision from the estimated range of effects.
    pub assumption_choice: LongitudinalSensitivityChoice,
    /// Decision after widening the entire range by the joint sampling radius.
    pub sampling_choice: LongitudinalSensitivityChoice,
}

/// Affine sensitivity surface and its explicitly simultaneous sampling reading.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LongitudinalSensitivityReport {
    /// The usual g-formula contrast under sequential ignorability and final MAR.
    pub reference_contrast: f64,
    /// Sensitivity corners with coordinate-wise slices of the simultaneous band.
    pub corners: Vec<LongitudinalSensitivityCorner>,
    /// Exact minimum and maximum fitted effect over the entire box.
    pub assumption_range: [f64; 2],
    /// Band projected onto the minimum and maximum, respectively.
    pub extrema_intervals: [[f64; 2]; 2],
    /// Confidence band radius common to every contrast in the box.
    pub sampling_radius: f64,
    /// Target coverage level of the subject bootstrap band.
    pub nominal_level: f64,
    /// The seed from which each complete-subject resample was derived.
    pub seed: u64,
    /// Number of bootstrap fits used.
    pub refits: u32,
    /// Robust threshold comparison after projecting the joint band.
    pub decision: Option<LongitudinalSensitivityDecision>,
}

#[derive(Clone, Copy, Default)]
struct Cell {
    count: u32,
    observed: u32,
    sum: f64,
}

#[derive(Default)]
struct Fit {
    // Row indexes correspond to the four-bit history W, A0, L1, A1.
    cells: [Cell; 16],
}

impl Fit {
    fn add(&mut self, row: &LongitudinalSubject) {
        let index = (((row.baseline as usize * 2 + row.treatment[0] as usize) * 2
            + row.intermediate as usize)
            * 2)
            + row.treatment[1] as usize;
        let cell = &mut self.cells[index];
        cell.count += 1;
        if let Some(outcome) = row.outcome {
            cell.observed += 1;
            cell.sum += outcome;
        }
    }

    fn check(&self) -> Result<(), JointSensitivityError> {
        if self.cells.iter().any(|cell| cell.count == 0)
            || self.cells.iter().enumerate().any(|(i, cell)| {
                let a0 = (i / 4) % 2;
                let a1 = i % 2;
                a0 == a1 && (cell.observed == 0 || !cell.sum.is_finite())
            })
        {
            return Err(JointSensitivityError::Positivity);
        }
        Ok(())
    }

    fn contrast(
        &self,
        point: LongitudinalSensitivityPoint,
        bounds: Option<[f64; 2]>,
    ) -> Result<f64, JointSensitivityError> {
        let means = [self.mean(0, point, bounds)?, self.mean(1, point, bounds)?];
        let contrast = means[1] - means[0];
        contrast.is_finite().then_some(contrast).ok_or(JointSensitivityError::Numeric)
    }

    fn mean(
        &self,
        chosen: usize,
        point: LongitudinalSensitivityPoint,
        bounds: Option<[f64; 2]>,
    ) -> Result<f64, JointSensitivityError> {
        let sign = if chosen == 0 { -1.0 } else { 1.0 };
        let mut mean = 0.0;
        let mut total = 0.0;
        for w in 0..2 {
            let base = 8 * w;
            let total_w: f64 =
                self.cells[base..base + 8].iter().map(|cell| f64::from(cell.count)).sum();
            let mut after_a0 = 0.0;
            let mut total_after_a0 = 0.0;
            for l1 in 0..2 {
                let index = base + 4 * chosen + 2 * l1;
                let follow = self.cells[index + chosen];
                let deviate = self.cells[index + 1 - chosen];
                let observed_mean = follow.sum / f64::from(follow.observed);
                let missing_mean = observed_mean - sign * point.missingness;
                // The pattern-mixture departure describes an existing missing stratum.
                if follow.observed != follow.count {
                    check_mean(missing_mean, bounds)?;
                }
                let follow_count = f64::from(follow.count);
                let deviate_count = f64::from(deviate.count);
                let mean_if_followed = observed_mean
                    - sign * point.missingness * f64::from(follow.count - follow.observed)
                        / follow_count;
                let mean_if_deviated = mean_if_followed - sign * point.assignment[1];
                check_mean(mean_if_deviated, bounds)?;
                after_a0 += follow_count * mean_if_followed + deviate_count * mean_if_deviated;
                total_after_a0 += follow_count + deviate_count;
            }
            let follow_mean = after_a0 / total_after_a0;
            let deviate_mean = follow_mean - sign * point.assignment[0];
            check_mean(deviate_mean, bounds)?;
            mean += total_after_a0 * follow_mean + (total_w - total_after_a0) * deviate_mean;
            total += total_w;
        }
        Ok(mean / total)
    }
}

fn check_mean(value: f64, bounds: Option<[f64; 2]>) -> Result<(), JointSensitivityError> {
    if !value.is_finite() {
        return Err(JointSensitivityError::Numeric);
    }
    if let Some([low, high]) = bounds {
        // Mixture arithmetic can move a boundary result by a few ulps.
        let tol = 16.0 * f64::EPSILON * low.abs().max(high.abs()).max(1.0);
        if value < low - tol || value > high + tol {
            return Err(JointSensitivityError::IncoherentBox);
        }
    }
    Ok(())
}

pub(super) fn execute(
    spec: &LongitudinalSensitivitySpec,
    rows: &[LongitudinalSubject],
    ctx: &ExecutionContext,
) -> Result<LongitudinalSensitivityReport, JointSensitivityError> {
    let mut fit = Fit::default();
    for row in rows {
        fit.add(row);
    }
    fit.check()?;
    let zero = LongitudinalSensitivityPoint { assignment: [0.0; 2], missingness: 0.0 };
    let reference_contrast = fit.contrast(zero, spec.outcome_support)?;
    let points = coordinates(spec);
    let values = points
        .iter()
        .map(|&point| fit.contrast(point, spec.outcome_support))
        .collect::<Result<Vec<_>, _>>()?;
    let mut deviations = Vec::with_capacity(spec.replicates as usize);
    for index in 0..spec.replicates {
        if ctx.cancellation.is_cancelled() {
            return Err(JointSensitivityError::Cancelled);
        }
        let mut rng = ctx.rng.stream_for(BOOTSTRAP, u64::from(index));
        let mut draw = Fit::default();
        for _ in rows {
            // Explicit unbiased choice on this row count, independent of host word size.
            let n = rows.len() as u64;
            let ceiling = u64::MAX - u64::MAX % n;
            let mut choice = rng.next_u64();
            while choice >= ceiling {
                choice = rng.next_u64();
            }
            draw.add(
                &rows[usize::try_from(choice % n).expect("the remainder is below the row count")],
            );
        }
        draw.check().map_err(|_| JointSensitivityError::RefitPositivity(index))?;
        let mut sup: f64 = 0.0;
        for (&point, &estimate) in points.iter().zip(&values) {
            // The bootstrap estimates the sampling variation of the unconstrained
            // affine plug-in. Feasibility identifies the scientific box at the
            // original fit and is not imposed as truncation of bootstrap variation.
            sup = sup.max((draw.contrast(point, None)? - estimate).abs());
        }
        deviations.push(sup);
    }
    deviations.sort_unstable_by(f64::total_cmp);
    let radius = quantile_sorted(&deviations, spec.level, QuantileRule::ExchangeableRank);
    let interval = |value: f64| [value - radius, value + radius];
    let assumption_range = [
        values.iter().copied().fold(f64::INFINITY, f64::min),
        values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    ];
    let extrema_intervals = assumption_range.map(interval);
    if extrema_intervals.iter().flatten().any(|x| !x.is_finite()) {
        return Err(JointSensitivityError::Numeric);
    }
    let decision = spec.decision_threshold.map(|threshold| LongitudinalSensitivityDecision {
        threshold,
        assumption_choice: LongitudinalSensitivityChoice::from_range(assumption_range, threshold),
        sampling_choice: LongitudinalSensitivityChoice::from_range(
            [extrema_intervals[0][0], extrema_intervals[1][1]],
            threshold,
        ),
    });
    Ok(LongitudinalSensitivityReport {
        reference_contrast,
        corners: points
            .into_iter()
            .zip(values)
            .map(|(departures, contrast)| LongitudinalSensitivityCorner {
                departures,
                contrast,
                sampling_interval: interval(contrast),
            })
            .collect(),
        assumption_range,
        extrema_intervals,
        sampling_radius: radius,
        nominal_level: spec.level,
        seed: ctx.rng.master_seed(),
        refits: spec.replicates,
        decision,
    })
}

fn coordinates(spec: &LongitudinalSensitivitySpec) -> Vec<LongitudinalSensitivityPoint> {
    let mut corners = Vec::with_capacity(8);
    for &a0 in &spec.assignment_limits[0] {
        for &a1 in &spec.assignment_limits[1] {
            for &mnar in &spec.mnar_limits {
                let point =
                    LongitudinalSensitivityPoint { assignment: [a0, a1], missingness: mnar };
                if !corners.contains(&point) {
                    corners.push(point);
                }
            }
        }
    }
    corners
}
