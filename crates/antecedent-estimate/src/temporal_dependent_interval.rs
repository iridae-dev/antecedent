//! Dependence-preserving sampling interval for the finite two-step sequence (2.3A, X5).
//!
//! Observations of one repeated unit share that unit's dependence, so the
//! interval resamples whole units: every complete history of a drawn unit moves
//! together, unit and time identities stay stable and are recorded, and the
//! entire prepared two-step estimator is re-run on each replicate (no row-IID
//! fallback). Replicate ids and unit-selection digests are deterministic
//! functions of the seed and the panel identity so a consumer can replay them.
//!
//! Calibration of this interval is **unmeasured**: it is implemented and
//! reproducible but makes no coverage claim, and the public route stays closed
//! (see [`route_frozen_refusal`]) until the whole-method grid has been measured.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use antecedent_core::{ExecutionContext, reason_code};

use crate::error::EstimationError;
use crate::splitmix::{GOLDEN_GAMMA, mix64, seed_mix, splitmix64};

/// Most replicates one interval may use.
pub const INTERVAL_MAX_REPLICATES: usize = 2000;
/// Fewest replicates an interval may use.
pub const INTERVAL_MIN_REPLICATES: usize = 20;
/// Most repeated units a panel may carry.
pub const INTERVAL_MAX_UNITS: usize = 100_000;
/// Fewest repeated units an interval is attempted with by default.
pub const DEFAULT_MIN_UNITS: usize = 20;
/// Inference claim of this engine: dependence is preserved, coverage is not measured.
pub const INTERVAL_CLAIM: &str = "dependence_preserving_calibration_unmeasured";
/// Calibration status of this engine.
pub const INTERVAL_CALIBRATION_STATUS: &str = "unmeasured";

/// Detail: the public interval route is closed.
pub const TEMPORAL_INTERVAL_ROUTE_FROZEN: &str = "temporal_interval.route_frozen";
/// Detail: unit ownership, time order or dependence is absent or incompatible.
pub const TEMPORAL_INTERVAL_UNKNOWN_UNITS: &str = "temporal_interval.unknown_units";
/// Detail: too few repeated units for a dependent interval.
pub const TEMPORAL_INTERVAL_TOO_FEW_UNITS: &str = "temporal_interval.too_few_units";
/// Detail: a history the estimator needs has no support.
pub const TEMPORAL_INTERVAL_UNSUPPORTED_HISTORY: &str = "temporal_interval.unsupported_history";
/// Detail: more replicates than the hard cap.
pub const TEMPORAL_INTERVAL_TOO_MANY_REPLICATES: &str = "temporal_interval.too_many_replicates";

/// The refusal of the public interval route while its calibration is unmeasured.
#[must_use]
pub fn route_frozen_refusal() -> EstimationError {
    EstimationError::refused(
        reason_code!("cell_not_licensed"),
        format!(
            "{TEMPORAL_INTERVAL_ROUTE_FROZEN}: a dependent-sampling interval is requested \
             before its whole-method grid and artifact gate pass"
        ),
    )
}

fn unknown_units(message: impl AsRef<str>) -> EstimationError {
    EstimationError::refused(
        reason_code!("route_not_supported"),
        format!("{TEMPORAL_INTERVAL_UNKNOWN_UNITS}: {}", message.as_ref()),
    )
}

fn unsupported_history(message: impl AsRef<str>) -> EstimationError {
    EstimationError::refused(
        reason_code!("route_not_supported"),
        format!("{TEMPORAL_INTERVAL_UNSUPPORTED_HISTORY}: {}", message.as_ref()),
    )
}

/// One complete two-step history of a unit: initial state, first action, the
/// step-2 covariate, second action and outcome, observed at `time_id`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SequenceHistory {
    /// Stable id of the period (window) the history starts at.
    pub time_id: u64,
    /// Pre-action state level.
    pub s0: u32,
    /// First action level.
    pub a1: u32,
    /// Step-2 covariate level.
    pub l2: u32,
    /// Second action level.
    pub a2: u32,
    /// Outcome after the last action.
    pub y: f64,
}

/// Every complete history of one repeated unit; they are resampled together.
#[derive(Clone, Debug, PartialEq)]
pub struct UnitHistories {
    /// Stable unit id.
    pub unit_id: u64,
    /// The unit's complete histories, strictly increasing in `time_id`.
    pub histories: Vec<SequenceHistory>,
}

pub(crate) fn fold(state: u64, word: u64) -> u64 {
    mix64(state ^ word).wrapping_add(GOLDEN_GAMMA)
}

pub(crate) fn fold_text(mut state: u64, text: &str) -> u64 {
    for byte in text.bytes() {
        state = fold(state, u64::from(byte));
    }
    fold(state, text.len() as u64)
}

/// A snapshot of repeated units with their complete histories.
#[derive(Clone, Debug)]
pub struct TemporalUnitPanel {
    snapshot_id: String,
    units: Vec<UnitHistories>,
    digest: u64,
}

impl TemporalUnitPanel {
    /// Declare a panel. `units` is `None` when no unit map was supplied.
    ///
    /// # Errors
    /// `route_not_supported` / `temporal_interval.unknown_units` for an absent unit
    /// map or snapshot id, duplicate unit ids, a unit without histories, a
    /// non-finite outcome or a unit whose time ids are not strictly increasing;
    /// `temporal_transport.bounds_exceeded` above [`INTERVAL_MAX_UNITS`].
    pub fn new(
        snapshot_id: impl Into<String>,
        units: Option<Vec<UnitHistories>>,
    ) -> Result<Self, EstimationError> {
        let snapshot_id = snapshot_id.into();
        let Some(units) = units else {
            return Err(unknown_units("no unit map was supplied; row-level independence is not assumed"));
        };
        if snapshot_id.is_empty() {
            return Err(unknown_units("the panel has no snapshot id"));
        }
        if units.len() > INTERVAL_MAX_UNITS {
            return Err(EstimationError::refused(
                reason_code!("route_not_supported"),
                format!(
                    "{}: {} units exceed the cap of {INTERVAL_MAX_UNITS}",
                    antecedent_identify::sid::temporal_sequence::TEMPORAL_BOUNDS_EXCEEDED,
                    units.len()
                ),
            ));
        }
        let mut seen = BTreeSet::new();
        let mut digest = fold_text(GOLDEN_GAMMA, &snapshot_id);
        for unit in &units {
            if !seen.insert(unit.unit_id) {
                return Err(unknown_units(format!("unit id {} appears twice", unit.unit_id)));
            }
            if unit.histories.is_empty() {
                return Err(unknown_units(format!("unit {} has no complete history", unit.unit_id)));
            }
            digest = fold(digest, unit.unit_id);
            let mut previous: Option<u64> = None;
            for history in &unit.histories {
                if previous.is_some_and(|p| history.time_id <= p) {
                    return Err(unknown_units(format!(
                        "unit {} has time ids that are not strictly ordered",
                        unit.unit_id
                    )));
                }
                if !history.y.is_finite() {
                    return Err(unknown_units(format!(
                        "unit {} has a non-finite outcome",
                        unit.unit_id
                    )));
                }
                previous = Some(history.time_id);
                for word in [
                    history.time_id,
                    u64::from(history.s0),
                    u64::from(history.a1),
                    u64::from(history.l2),
                    u64::from(history.a2),
                    history.y.to_bits(),
                ] {
                    digest = fold(digest, word);
                }
            }
        }
        Ok(Self { snapshot_id, units, digest })
    }

    /// Snapshot id.
    #[must_use]
    pub fn snapshot_id(&self) -> &str {
        &self.snapshot_id
    }

    /// The units, in declared order.
    #[must_use]
    pub fn units(&self) -> &[UnitHistories] {
        &self.units
    }

    /// Number of units.
    #[must_use]
    pub fn unit_count(&self) -> usize {
        self.units.len()
    }

    /// Number of complete histories over all units.
    #[must_use]
    pub fn history_count(&self) -> usize {
        self.units.iter().map(|u| u.histories.len()).sum()
    }

    /// Identity of the snapshot, every unit id, every time id and every value.
    #[must_use]
    pub const fn digest(&self) -> u64 {
        self.digest
    }
}

/// Why a conditional response could not be formed from histories.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ResponseGap {
    /// No history has this initial state with the first action.
    NoState,
    /// The cell of this step-2 covariate level has no history with the second action.
    NoCell { l2: u32 },
}

/// One pass of counts over histories for a fixed two-step sequence.
#[derive(Debug, Default)]
pub(crate) struct SequenceTallies {
    state_all: BTreeMap<u32, u64>,
    state_first: BTreeMap<u32, u64>,
    covariate: BTreeMap<(u32, u32), u64>,
    cell: BTreeMap<(u32, u32), (u64, f64)>,
    total: u64,
}

impl SequenceTallies {
    pub(crate) fn of(units: &[&UnitHistories], sequence: [u32; 2]) -> Self {
        let mut tallies = Self::default();
        for history in units.iter().flat_map(|u| u.histories.iter()) {
            tallies.total += 1;
            *tallies.state_all.entry(history.s0).or_insert(0) += 1;
            if history.a1 != sequence[0] {
                continue;
            }
            *tallies.state_first.entry(history.s0).or_insert(0) += 1;
            *tallies.covariate.entry((history.s0, history.l2)).or_insert(0) += 1;
            if history.a2 == sequence[1] {
                let entry = tallies.cell.entry((history.s0, history.l2)).or_insert((0, 0.0));
                entry.0 += 1;
                entry.1 += history.y;
            }
        }
        tallies
    }

    /// Share of all histories that start in `s0` (the observed initial-state law).
    #[allow(clippy::cast_precision_loss, reason = "counts are far below 2^53")]
    pub(crate) fn observed_state_mass(&self, s0: u32) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.state_all.get(&s0).copied().unwrap_or(0) as f64 / self.total as f64
    }

    /// Observed initial states, ascending.
    pub(crate) fn observed_states(&self) -> Vec<u32> {
        self.state_all.keys().copied().collect()
    }

    /// Response of the whole sequence given the initial state `s0`:
    /// `sum_l P(l | s0, a1) E[y | s0, a1, l, a2]`, read from the histories.
    #[allow(clippy::cast_precision_loss, reason = "counts are far below 2^53")]
    pub(crate) fn response(&self, s0: u32) -> Result<f64, ResponseGap> {
        let n_state = match self.state_first.get(&s0) {
            Some(n) if *n > 0 => *n,
            _ => return Err(ResponseGap::NoState),
        };
        let mut total = 0.0;
        for (&(_, l2), &n_cov) in self.covariate.range((s0, 0)..=(s0, u32::MAX)) {
            let (n_cell, sum) = match self.cell.get(&(s0, l2)) {
                Some(c) if c.0 > 0 => *c,
                _ => return Err(ResponseGap::NoCell { l2 }),
            };
            total += (n_cov as f64 / n_state as f64) * (sum / n_cell as f64);
        }
        Ok(total)
    }
}

/// One prepared whole estimator of a scalar two-step sequence quantity, re-run
/// unchanged on every unit-resampled replicate.
pub trait TemporalEstimator {
    /// Typed estimand label; an interval carries it so it cannot be relabeled.
    fn label(&self) -> &'static str;

    /// Estimate from `units` (a resample may repeat a unit).
    ///
    /// # Errors
    /// A refusal when a history the estimator needs has no support.
    fn estimate(&self, units: &[&UnitHistories]) -> Result<f64, EstimationError>;
}

/// The two-step response averaged over the panel's own observed initial-state law.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObservedStateSequence {
    /// Actions of the two steps.
    pub sequence: [u32; 2],
}

impl ObservedStateSequence {
    /// Estimand label.
    pub const LABEL: &'static str = "observed_initial_state";
}

impl TemporalEstimator for ObservedStateSequence {
    fn label(&self) -> &'static str {
        Self::LABEL
    }

    fn estimate(&self, units: &[&UnitHistories]) -> Result<f64, EstimationError> {
        let tallies = SequenceTallies::of(units, self.sequence);
        let mut total = 0.0;
        for s0 in tallies.observed_states() {
            let response = tallies.response(s0).map_err(|gap| {
                unsupported_history(format!("observed initial state {s0}: {gap:?}"))
            })?;
            total += tallies.observed_state_mass(s0) * response;
        }
        Ok(total)
    }
}

/// How the interval is read off the replicate points.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntervalMethod {
    /// Equal-tailed percentile interval of the replicate points.
    Percentile,
    /// Basic (reverse percentile) interval: `2 * point - upper, 2 * point - lower`.
    Basic,
}

/// Resampling design and bounds of one dependent interval.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DependentIntervalConfig {
    /// Replicates, within [`INTERVAL_MIN_REPLICATES`]..=[`INTERVAL_MAX_REPLICATES`].
    pub replicates: usize,
    /// Seed of the replicate stream.
    pub seed: u64,
    /// Two-sided level in (0, 1).
    pub level: f64,
    /// Interval construction.
    pub method: IntervalMethod,
    /// Fewest units an interval is attempted with.
    pub min_units: usize,
    /// Largest tolerated fraction of failed (dropped) replicates.
    pub max_failed_fraction: f64,
}

impl Default for DependentIntervalConfig {
    fn default() -> Self {
        Self {
            replicates: 500,
            seed: 0,
            level: 0.95,
            method: IntervalMethod::Percentile,
            min_units: DEFAULT_MIN_UNITS,
            max_failed_fraction: 0.05,
        }
    }
}

/// One replicate: stable id, the digest of the units it drew and its point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReplicateRecord {
    /// Replicate index.
    pub index: usize,
    /// Deterministic id from the seed, panel identity and index.
    pub replicate_id: u64,
    /// Digest of the drawn unit ids, in draw order.
    pub selection_digest: u64,
    /// The re-estimated point, or `None` when the replicate failed and was dropped.
    pub point: Option<f64>,
}

/// A dependence-preserving interval with its replay receipt.
#[derive(Clone, Debug, PartialEq)]
pub struct DependentInterval {
    /// Estimand label of the estimator.
    pub estimand: &'static str,
    /// Point estimate on the original panel.
    pub point: f64,
    /// Lower bound.
    pub lower: f64,
    /// Upper bound.
    pub upper: f64,
    /// Nominal level.
    pub level: f64,
    /// Construction.
    pub method: IntervalMethod,
    /// Every replicate, in order.
    pub replicates: Vec<ReplicateRecord>,
    /// Replicates dropped as failed.
    pub failed: usize,
    /// Units of the panel.
    pub units: usize,
    /// Snapshot id of the panel.
    pub snapshot_id: String,
    /// Identity of the panel (snapshot, unit ids, time ids, values).
    pub panel_digest: u64,
    /// Seed of the replicate stream.
    pub seed: u64,
    /// Always [`INTERVAL_CLAIM`].
    pub claim: &'static str,
    /// Always [`INTERVAL_CALIBRATION_STATUS`].
    pub calibration: &'static str,
}

impl DependentInterval {
    /// Upper minus lower.
    #[must_use]
    pub fn width(&self) -> f64 {
        self.upper - self.lower
    }

    /// Whether `value` lies within the closed interval.
    #[must_use]
    pub fn contains(&self, value: f64) -> bool {
        self.lower <= value && value <= self.upper
    }

    /// Digest over every replicate id, selection digest and point.
    #[must_use]
    pub fn replicate_digest(&self) -> u64 {
        let mut state = fold(GOLDEN_GAMMA, self.seed);
        state = fold(state, self.panel_digest);
        for record in &self.replicates {
            state = fold(state, record.replicate_id);
            state = fold(state, record.selection_digest);
            state = fold(state, record.point.map_or(u64::MAX, f64::to_bits));
        }
        state
    }
}

fn validate_config(config: &DependentIntervalConfig) -> Result<(), EstimationError> {
    if config.replicates > INTERVAL_MAX_REPLICATES {
        return Err(EstimationError::refused(
            reason_code!("route_not_supported"),
            format!(
                "{TEMPORAL_INTERVAL_TOO_MANY_REPLICATES}: {} replicates exceed the cap of \
                 {INTERVAL_MAX_REPLICATES}",
                config.replicates
            ),
        ));
    }
    let bad = |message: &str| EstimationError::refused(reason_code!("invalid_argument"), message);
    if config.replicates < INTERVAL_MIN_REPLICATES {
        return Err(bad("a dependent interval needs at least 20 replicates"));
    }
    if !(config.level > 0.0 && config.level < 1.0) {
        return Err(bad("the interval level must lie strictly between 0 and 1"));
    }
    if !(0.0..1.0).contains(&config.max_failed_fraction) {
        return Err(bad("the failed-replicate fraction must lie in [0, 1)"));
    }
    Ok(())
}

/// Linear-interpolated quantile (type 7) of an ascending non-empty slice.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "indices of a bounded replicate vector"
)]
fn quantile(sorted: &[f64], p: f64) -> f64 {
    let h = p.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let frac = h - lo as f64;
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

fn draw_units(panel: &TemporalUnitPanel, replicate_id: u64) -> (Vec<&UnitHistories>, u64) {
    let n = panel.units.len();
    let wide = u128::try_from(n).unwrap_or(1);
    let mut state = seed_mix(replicate_id);
    let mut selected = Vec::with_capacity(n);
    let mut digest = fold(GOLDEN_GAMMA, replicate_id);
    for _ in 0..n {
        let draw = u128::from(splitmix64(&mut state));
        let index = usize::try_from((draw * wide) >> 64).unwrap_or(0).min(n - 1);
        digest = fold(digest, panel.units[index].unit_id);
        selected.push(&panel.units[index]);
    }
    (selected, digest)
}

fn run_replicates<E: TemporalEstimator + ?Sized>(
    panel: &TemporalUnitPanel,
    estimator: &E,
    config: &DependentIntervalConfig,
    ctx: &ExecutionContext,
) -> Result<Vec<ReplicateRecord>, EstimationError> {
    let base = fold(fold(GOLDEN_GAMMA, config.seed), panel.digest);
    let mut records = Vec::with_capacity(config.replicates);
    for index in 0..config.replicates {
        crate::transport::refuse_cancelled(ctx, "temporal dependent interval replicate")?;
        let replicate_id = fold(base, index as u64);
        let (selected, selection_digest) = draw_units(panel, replicate_id);
        let point = estimator.estimate(&selected).ok().filter(|p| p.is_finite());
        records.push(ReplicateRecord { index, replicate_id, selection_digest, point });
    }
    Ok(records)
}

/// Resample whole units, re-run the whole estimator per replicate and read a
/// percentile (or basic) interval off the replicate points.
///
/// The interval carries no coverage claim: calibration is unmeasured and the
/// public route stays closed.
///
/// # Errors
/// `too_few_clusters` / `temporal_interval.too_few_units` below the minimum unit
/// count; `route_not_supported` with `temporal_interval.too_many_replicates` above
/// the cap and `temporal_interval.unsupported_history` when the original panel (or
/// too large a fraction of replicates) leaves a needed history unsupported;
/// `transport_budget_cancel` on cancellation.
pub fn dependent_unit_interval<E: TemporalEstimator + ?Sized>(
    panel: &TemporalUnitPanel,
    estimator: &E,
    config: &DependentIntervalConfig,
    ctx: &ExecutionContext,
) -> Result<DependentInterval, EstimationError> {
    validate_config(config)?;
    if panel.units.len() < config.min_units.max(2) {
        return Err(EstimationError::refused(
            reason_code!("too_few_clusters"),
            format!(
                "{TEMPORAL_INTERVAL_TOO_FEW_UNITS}: {} repeated units, at least {} required",
                panel.units.len(),
                config.min_units.max(2)
            ),
        ));
    }
    crate::transport::refuse_cancelled(ctx, "temporal dependent interval")?;
    let all = panel.units.iter().collect::<Vec<_>>();
    let point = estimator
        .estimate(&all)
        .map_err(|error| unsupported_history(format!("the original panel: {error}")))?;
    let replicates = run_replicates(panel, estimator, config, ctx)?;
    let mut points =
        replicates.iter().filter_map(|record| record.point).collect::<Vec<_>>();
    let failed = replicates.len() - points.len();
    #[allow(clippy::cast_precision_loss, reason = "replicate counts are at most 2000")]
    let fraction = failed as f64 / replicates.len() as f64;
    if fraction > config.max_failed_fraction || points.len() < 2 {
        return Err(unsupported_history(format!(
            "{failed} of {} unit-resampled replicates left a needed history unsupported, above \
             the allowed fraction {}",
            replicates.len(),
            config.max_failed_fraction
        )));
    }
    points.sort_by(f64::total_cmp);
    let tail = (1.0 - config.level) / 2.0;
    let (low, high) = (quantile(&points, tail), quantile(&points, 1.0 - tail));
    let (lower, upper) = match config.method {
        IntervalMethod::Percentile => (low, high),
        IntervalMethod::Basic => (2.0_f64.mul_add(point, -high), 2.0_f64.mul_add(point, -low)),
    };
    Ok(DependentInterval {
        estimand: estimator.label(),
        point,
        lower,
        upper,
        level: config.level,
        method: config.method,
        replicates,
        failed,
        units: panel.units.len(),
        snapshot_id: panel.snapshot_id.clone(),
        panel_digest: panel.digest,
        seed: config.seed,
        claim: INTERVAL_CLAIM,
        calibration: INTERVAL_CALIBRATION_STATUS,
    })
}
