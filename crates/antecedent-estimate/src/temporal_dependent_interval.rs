//! Dependence-preserving sampling interval for the finite two-step sequence.
//!
//! Observations of one repeated unit share that unit's dependence, so the
//! interval resamples whole units: every complete history of a drawn unit moves
//! together, unit and time identities stay stable and are recorded, and the
//! entire prepared two-step estimator is re-run on each replicate (no row-IID
//! fallback). Replicate ids and unit-selection digests are deterministic
//! functions of the seed and the panel identity so a consumer can replay them.
//!
//! The separately named `bootstrap_studentized` candidate is restricted to
//! original native response/effect estimators with an exact balanced-unit score
//! derivation: identical positive binary history multiplicities make all fitted
//! covariate weights and denominators invariant under unit resampling. Thus
//! `theta = mean(z_i)`, and `SE² = sum((z_i - mean(z))²)/(n(n-1))`. Every bootstrap
//! draw still refits the original estimator, checks its equality to the selected
//! unit-score mean, and uses its own selected-score SE. Equal-tailed quantiles of
//! `(theta* - theta)/SE*` are inverted as `theta - q_upper SE, theta - q_lower SE`.
//! This follows the bootstrap-t construction discussed by Hall (1988),
//! <https://doi.org/10.1214/aos/1176350933>. It is neither a symmetric-interval
//! higher-order claim nor a finite-sample guarantee; lattice, degenerate,
//! unbalanced and unsupported designs need their own methods/evidence.
//!
//! Calibration of this interval is **unmeasured**: it is implemented and
//! reproducible but makes no coverage claim, and the public route stays closed
//! (see [`route_frozen_refusal`]) until the whole-method grid has been measured.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use antecedent_core::{CanonicalWriter, ExecutionContext, reason_code};

use crate::error::EstimationError;
use crate::splitmix::{seed_mix, splitmix64};

/// Reference replicate count. Not enforced: the replicates are cancellable and bounded by the
/// execution budget.
pub const INTERVAL_MAX_REPLICATES: usize = 2000;
/// Fewest replicates an interval may use.
pub const INTERVAL_MIN_REPLICATES: usize = 20;
/// Most repeated units a panel may carry.
pub const INTERVAL_MAX_UNITS: usize = 100_000;
/// Fewest repeated units an interval is attempted with by default.
pub(crate) const DEFAULT_MIN_UNITS: usize = 20;
/// Units up to which the bootstrap-t candidate's exact balanced-unit scores are built; a larger
/// panel is read off the percentile route instead (its method is recorded on the interval).
pub(crate) const STUDENTIZED_MAX_UNITS: usize = 4096;
/// Inference claim of this engine: dependence is preserved, coverage is not measured.
pub const INTERVAL_CLAIM: &str = "dependence_preserving_calibration_unmeasured";
/// Calibration status of this engine.
pub const INTERVAL_CALIBRATION_STATUS: &str = "unmeasured";

/// Detail: the public interval route is closed.
pub(crate) const TEMPORAL_INTERVAL_ROUTE_FROZEN: &str = "temporal_interval.route_frozen";
/// Detail: unit ownership, time order or dependence is absent or incompatible.
pub(crate) const TEMPORAL_INTERVAL_UNKNOWN_UNITS: &str = "temporal_interval.unknown_units";
/// Detail: too few repeated units for a dependent interval.
pub const TEMPORAL_INTERVAL_TOO_FEW_UNITS: &str = "temporal_interval.too_few_units";
/// Detail: a history the estimator needs has no support.
pub const TEMPORAL_INTERVAL_UNSUPPORTED_HISTORY: &str = "temporal_interval.unsupported_history";
/// Detail: more replicates than the execution budget funds.
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

/// Domain tag of the panel identity.
const PANEL_DIGEST_TAG: &str = "antecedent.estimate.temporal_dependent_interval.panel";
/// Domain tag of the replicate id (the stream key of one resample).
const REPLICATE_ID_TAG: &str = "antecedent.estimate.temporal_dependent_interval.replicate_id";
/// Domain tag of the digest of one replicate's drawn units.
const SELECTION_DIGEST_TAG: &str = "antecedent.estimate.temporal_dependent_interval.selection";
/// Domain tag of the digest over every replicate of an interval.
const REPLICATE_SET_TAG: &str = "antecedent.estimate.temporal_dependent_interval.replicate_set";

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
            return Err(unknown_units(
                "no unit map was supplied; row-level independence is not assumed",
            ));
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
        let mut digest = CanonicalWriter::new(PANEL_DIGEST_TAG);
        digest.text(&snapshot_id);
        for unit in &units {
            if !seen.insert(unit.unit_id) {
                return Err(unknown_units(format!("unit id {} appears twice", unit.unit_id)));
            }
            if unit.histories.is_empty() {
                return Err(unknown_units(format!(
                    "unit {} has no complete history",
                    unit.unit_id
                )));
            }
            digest.word(unit.unit_id);
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
                    digest.word(word);
                }
            }
        }
        Ok(Self { snapshot_id, units, digest: crate::util::canonical_word(&digest) })
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
            tallies.observe(history, sequence);
        }
        tallies
    }

    fn observe(&mut self, history: &SequenceHistory, sequence: [u32; 2]) {
        self.total += 1;
        *self.state_all.entry(history.s0).or_insert(0) += 1;
        if history.a1 != sequence[0] {
            return;
        }
        *self.state_first.entry(history.s0).or_insert(0) += 1;
        *self.covariate.entry((history.s0, history.l2)).or_insert(0) += 1;
        if history.a2 == sequence[1] {
            let entry = self.cell.entry((history.s0, history.l2)).or_insert((0, 0.));
            entry.0 += 1;
            entry.1 += history.y;
        }
    }

    pub(crate) fn of_many(units: &[&UnitHistories], sequences: &[[u32; 2]]) -> Vec<Self> {
        if sequences.is_empty() {
            return Vec::new();
        }
        let mut tallies = sequences.iter().map(|_| Self::default()).collect::<Vec<_>>();
        for history in units.iter().flat_map(|u| &u.histories) {
            for (tally, &sequence) in tallies.iter_mut().zip(sequences) {
                tally.observe(history, sequence);
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

    /// Checked estimator-specific exact unit scores for a balanced binary panel.
    /// The default declines studentization. Advanced Rust implementations may
    /// override this hook, but the result remains unmeasured; an override does
    /// not grant the closed checked-adapter or replay-artifact scope.
    /// # Errors
    /// A certified implementation refuses incompatible balance/support.
    fn balanced_unit_scores(
        &self,
        _units: &[&UnitHistories],
        _ctx: &antecedent_core::ExecutionContext,
    ) -> Result<Option<Vec<f64>>, EstimationError> {
        Ok(None)
    }
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
    fn balanced_unit_scores(
        &self,
        units: &[&UnitHistories],
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<Option<Vec<f64>>, EstimationError> {
        balanced_linear_unit_scores(self, units, ctx)
    }

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

/// Closed native estimator family with an exact balanced-unit score derivation.
/// This wrapper accepts original checked query types, never arbitrary callbacks.
#[derive(Clone, Debug)]
pub enum BalancedTemporalEstimator {
    /// Original observed-state response.
    Observed(ObservedStateSequence),
    /// Original fixed-state response.
    Fixed(crate::temporal_initial_state::FixedStateQuery),
    /// Original fixed-target marginalized response.
    Marginalized(crate::temporal_initial_state::MarginalizedQuery),
}
impl TemporalEstimator for BalancedTemporalEstimator {
    fn label(&self) -> &'static str {
        match self {
            Self::Observed(q) => q.label(),
            Self::Fixed(q) => q.label(),
            Self::Marginalized(q) => q.label(),
        }
    }
    fn estimate(&self, units: &[&UnitHistories]) -> Result<f64, EstimationError> {
        match self {
            Self::Observed(q) => q.estimate(units),
            Self::Fixed(q) => q.estimate(units),
            Self::Marginalized(q) => q.estimate(units),
        }
    }
    fn balanced_unit_scores(
        &self,
        units: &[&UnitHistories],
        ctx: &ExecutionContext,
    ) -> Result<Option<Vec<f64>>, EstimationError> {
        balanced_linear_unit_scores(self, units, ctx)
    }
}

/// Exact linear scores for the original finite-history estimators only.
/// Every unit must contain the same positive multiplicity of all sixteen binary
/// histories. Then fixed target weights, covariate probabilities and denominators
/// are invariant to unit resampling, and the estimator is the mean of unit scores.
/// This public helper does not establish the score derivation for an arbitrary
/// callback. The checked adapter and artifact consumer use original native
/// estimators and independently replay their scores.
/// # Errors
/// Unequal/nonbinary/incomplete histories, or unsupported unit fits.
pub fn balanced_linear_unit_scores<E: TemporalEstimator + ?Sized>(
    estimator: &E,
    units: &[&UnitHistories],
    ctx: &ExecutionContext,
) -> Result<Option<Vec<f64>>, EstimationError> {
    let histogram = |unit: &UnitHistories| {
        let mut cells = BTreeMap::new();
        for h in &unit.histories {
            *cells.entry([h.s0, h.a1, h.l2, h.a2]).or_insert(0_usize) += 1;
        }
        cells
    };
    let Some(first) = units.first() else {
        return Err(unsupported_history("empty unit-score panel"));
    };
    let expected = histogram(first);
    if expected.len() != 16
        || expected.keys().flatten().any(|&level| level > 1)
        || units.iter().any(|unit| histogram(unit) != expected)
    {
        return Err(EstimationError::refused(
            reason_code!("route_not_supported"),
            "temporal_interval.studentized_unbalanced: exact unit scores require identical complete binary history multiplicities",
        ));
    }
    units
        .iter()
        .map(|unit| {
            crate::transport::refuse_cancelled(ctx, "temporal original unit score")?;
            estimator.estimate(&[*unit])
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

/// How the interval is read off the replicate points.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntervalMethod {
    /// Equal-tailed percentile interval of a checked paired effect only.
    Percentile,
    /// Checked paired-effect reverse percentile: `2 * point - upper, 2 * point - lower`.
    Basic,
    /// Equal-tailed bootstrap-t with exact balanced-unit scores and per-draw SE.
    /// This distinct candidate has no finite-sample or calibrated guarantee.
    Studentized,
}

/// Resampling design and bounds of one dependent interval.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DependentIntervalConfig {
    /// Replicates, at least [`INTERVAL_MIN_REPLICATES`].
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
            method: IntervalMethod::Studentized,
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

/// Complete studentization receipt, separate from original replicate points.
#[derive(Clone, Debug, PartialEq)]
pub struct StudentizationReceipt {
    /// Original whole-unit mean standard error, unbiased sample variance / n.
    pub standard_error: f64,
    /// Exact original unit scores, in original panel order.
    pub unit_scores: Vec<f64>,
    /// Each resample's own standard error; None for invalid/zero-variance draws.
    pub replicate_standard_errors: Vec<Option<f64>>,
    /// Each centered and studentized pivot, aligned with original draws.
    pub pivots: Vec<Option<f64>>,
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
    /// Present only for the distinct exact balanced-unit bootstrap-t method.
    pub studentization: Option<StudentizationReceipt>,
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

/// A count that fits keeps its value. A count that does not fit saturates.
fn bounded_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

/// A count that fits keeps its value. A count that does not fit saturates.
fn bounded_u32(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

impl DependentInterval {
    /// Actual construction and observed work of this internal measurement candidate.
    /// Units, successful replicates and the nominal level are those of the executed
    /// whole-unit procedure; this neither authenticates a sampling design nor activates it.
    #[must_use]
    pub fn calibration_basis(&self) -> antecedent_core::CalibrationBasis {
        use std::sync::Arc;
        let interval = match self.method {
            IntervalMethod::Percentile => "bootstrap_percentile",
            IntervalMethod::Basic => "bootstrap_basic",
            IntervalMethod::Studentized => "bootstrap_studentized",
        };
        antecedent_core::CalibrationBasis::new(
            [
                "TemporalTransport",
                "SelectionAdmg",
                "fixed",
                "panel",
                "Frequentist",
                "dependent_temporal_interval",
                interval,
                "",
                "whole_unit",
                "",
                self.estimand,
            ]
            .map(Arc::from),
            self.level,
            Arc::from("point"),
            bounded_u64(self.units),
            Some(bounded_u32(self.replicates.len() - self.failed)),
            None,
            0.,
        )
    }

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
        replicate_set_digest(self.seed, self.panel_digest, &self.replicates)
    }
}

/// The identity word over a seed, a panel identity and every replicate record.
fn replicate_set_digest(seed: u64, panel_digest: u64, replicates: &[ReplicateRecord]) -> u64 {
    let mut digest = CanonicalWriter::new(REPLICATE_SET_TAG);
    digest.word(seed).word(panel_digest);
    for record in replicates {
        digest.word(record.replicate_id).word(record.selection_digest);
        digest.word(record.point.map_or(u64::MAX, f64::to_bits));
    }
    crate::util::canonical_word(&digest)
}

fn validate_config(config: &DependentIntervalConfig) -> Result<(), EstimationError> {
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

/// Hyndman–Fan type-7 quantile of an ascending slice, the bootstrap-percentile rule.
fn quantile(sorted: &[f64], p: f64) -> f64 {
    antecedent_stats::quantile_sorted(sorted, p, antecedent_stats::QuantileRule::Interpolated)
}

/// Bias-corrected and accelerated percentile endpoints (Efron 1987): the bias constant
/// is the normal score of the replicate mass below the point estimate, and the
/// acceleration is the skewness of the complete delete-one-unit jackknife. `None` when
/// a leave-one-out estimate is unsupported or the constants are not finite, so the
/// plain percentile endpoints stand.
fn bias_corrected_accelerated<E: TemporalEstimator + ?Sized>(
    all: &[&UnitHistories],
    estimator: &E,
    point: f64,
    sorted: &[f64],
    tail: f64,
) -> Option<(f64, f64)> {
    use antecedent_kernels::{norm_cdf, norm_inv};
    #[allow(clippy::cast_precision_loss, reason = "replicate counts are at most 2000")]
    let total = sorted.len() as f64;
    let below = sorted.iter().filter(|v| **v < point).count() as f64;
    #[allow(clippy::float_cmp, reason = "exact ties with the point estimate are the bias count")]
    let equal = sorted.iter().filter(|v| **v == point).count() as f64;
    let fraction = (below + 0.5 * equal) / total;
    if !(fraction > 0.0 && fraction < 1.0) {
        return None;
    }
    let bias = norm_inv(fraction);
    let mut jackknife = Vec::with_capacity(all.len());
    for skip in 0..all.len() {
        let rest = all
            .iter()
            .enumerate()
            .filter_map(|(index, unit)| (index != skip).then_some(*unit))
            .collect::<Vec<_>>();
        jackknife.push(estimator.estimate(&rest).ok()?);
    }
    #[allow(clippy::cast_precision_loss, reason = "unit counts are small")]
    let mean = jackknife.iter().sum::<f64>() / jackknife.len() as f64;
    let cubes = jackknife.iter().map(|v| (mean - v).powi(3)).sum::<f64>();
    let squares = jackknife.iter().map(|v| (mean - v).powi(2)).sum::<f64>();
    if squares <= 0.0 || !squares.is_finite() {
        return None;
    }
    let acceleration = cubes / (6.0 * squares.powf(1.5));
    let adjust = |z: f64| {
        let shifted = bias + z;
        norm_cdf(bias + shifted / (1.0 - acceleration * shifted))
    };
    let (p_low, p_high) = (adjust(norm_inv(tail)), adjust(norm_inv(1.0 - tail)));
    if !(p_low.is_finite() && p_high.is_finite() && p_low < p_high) {
        return None;
    }
    Some((quantile(sorted, p_low.clamp(0.0, 1.0)), quantile(sorted, p_high.clamp(0.0, 1.0))))
}

fn draw_units(panel: &TemporalUnitPanel, replicate_id: u64) -> (Vec<&UnitHistories>, u64) {
    let n = panel.units.len();
    let wide = u128::try_from(n).unwrap_or(1);
    let mut state = seed_mix(replicate_id);
    let mut selected = Vec::with_capacity(n);
    let mut digest = CanonicalWriter::new(SELECTION_DIGEST_TAG);
    digest.word(replicate_id);
    for _ in 0..n {
        let draw = u128::from(splitmix64(&mut state));
        let index = usize::try_from((draw * wide) >> 64).unwrap_or(0).min(n - 1);
        digest.word(panel.units[index].unit_id);
        selected.push(&panel.units[index]);
    }
    (selected, crate::util::canonical_word(&digest))
}

fn run_replicates<E: TemporalEstimator + ?Sized>(
    panel: &TemporalUnitPanel,
    estimator: &E,
    config: &DependentIntervalConfig,
    ctx: &ExecutionContext,
) -> Result<Vec<ReplicateRecord>, EstimationError> {
    let mut base = CanonicalWriter::new(REPLICATE_ID_TAG);
    base.word(config.seed).word(panel.digest);
    let _admitted = crate::util::admit_array::<ReplicateRecord>(ctx, config.replicates, 1)?;
    let mut records = Vec::with_capacity(config.replicates);
    for index in 0..config.replicates {
        crate::transport::refuse_cancelled(ctx, "temporal dependent interval replicate")?;
        let mut keyed = base.clone();
        keyed.word(index as u64);
        let replicate_id = crate::util::canonical_word(&keyed);
        let (selected, selection_digest) = draw_units(panel, replicate_id);
        let point = estimator.estimate(&selected).ok().filter(|p| p.is_finite());
        records.push(ReplicateRecord { index, replicate_id, selection_digest, point });
    }
    Ok(records)
}

fn mean_se(values: &[f64]) -> Option<(f64, f64)> {
    // A studentization needs a strictly positive standard error: the owner
    // returns a zero SE as `Some`, so the refusal is this call site's policy.
    antecedent_stats::mean_and_se(values).filter(|&(_, se)| se > 0.0)
}

fn prepare_studentization<E: TemporalEstimator + ?Sized>(
    panel: &TemporalUnitPanel,
    estimator: &E,
    point: f64,
    ctx: &ExecutionContext,
) -> Result<StudentizationReceipt, EstimationError> {
    let all = panel.units.iter().collect::<Vec<_>>();
    let scores = estimator.balanced_unit_scores(&all, ctx)?.ok_or_else(|| {
        EstimationError::refused(
            reason_code!("route_not_supported"),
            "temporal_interval.studentized_estimator_not_certified",
        )
    })?;
    let (mean, standard_error) = mean_se(&scores).ok_or_else(|| {
        unsupported_history(
        "temporal_interval.studentized_zero_variance: unit scores need positive finite variance")
    })?;
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-10 * (1.0 + a.abs() + b.abs());
    if scores.len() != all.len() || !close(mean, point) {
        return Err(unsupported_history("temporal_interval.studentized_score_mismatch"));
    }
    Ok(StudentizationReceipt {
        standard_error,
        unit_scores: scores,
        replicate_standard_errors: Vec::new(),
        pivots: Vec::new(),
    })
}

fn studentize_replicates(
    panel: &TemporalUnitPanel,
    point: f64,
    records: &[ReplicateRecord],
    mut receipt: StudentizationReceipt,
    ctx: &ExecutionContext,
) -> Result<StudentizationReceipt, EstimationError> {
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-10 * (1.0 + a.abs() + b.abs());
    let lookup = panel
        .units
        .iter()
        .zip(&receipt.unit_scores)
        .map(|(unit, &score)| (unit.unit_id, score))
        .collect::<BTreeMap<_, _>>();
    let mut errors = Vec::with_capacity(records.len());
    let mut pivots = Vec::with_capacity(records.len());
    for record in records {
        crate::transport::refuse_cancelled(ctx, "temporal studentization replicate")?;
        let (selected, _) = draw_units(panel, record.replicate_id);
        let selected_scores = selected.iter().map(|unit| lookup[&unit.unit_id]).collect::<Vec<_>>();
        let variance = mean_se(&selected_scores);
        if let (Some(estimate), Some((mean, _))) = (record.point, variance) {
            if !close(mean, estimate) {
                return Err(unsupported_history("temporal_interval.studentized_score_mismatch"));
            }
        }
        let pair = record
            .point
            .zip(variance)
            .map(|(estimate, (_, se))| (se, (estimate - point) / se))
            .filter(|(_, pivot)| pivot.is_finite());
        errors.push(pair.map(|(se, _)| se));
        pivots.push(pair.map(|(_, pivot)| pivot));
    }
    receipt.replicate_standard_errors = errors;
    receipt.pivots = pivots;
    Ok(receipt)
}

/// Resample whole units and construct the balanced-unit studentized interval.
///
/// This receipt retains unmeasured standing. A separate source-bound measured
/// adapter resolves evidence for its exact validated protocol.
///
/// # Errors
/// `too_few_clusters` / `temporal_interval.too_few_units` below the minimum unit
/// count; `route_not_supported` with `temporal_interval.unsupported_history` when the original panel (or
/// too large a fraction of replicates) leaves a needed history unsupported;
/// `transport_budget_cancel` on cancellation.
pub fn dependent_unit_interval<E: TemporalEstimator + ?Sized>(
    panel: &TemporalUnitPanel,
    estimator: &E,
    config: &DependentIntervalConfig,
    ctx: &ExecutionContext,
) -> Result<DependentInterval, EstimationError> {
    validate_config(config)?;
    if config.method != IntervalMethod::Studentized {
        return Err(EstimationError::refused(
            reason_code!("route_not_supported"),
            "temporal_interval.retired_method: direct intervals require studentized",
        ));
    }
    if panel.units.len() > STUDENTIZED_MAX_UNITS {
        // The studentization needs a score per original unit and a per-draw SE; beyond the
        // budget the interval is the percentile route over the same unit resamples.
        let percentile = DependentIntervalConfig { method: IntervalMethod::Percentile, ..*config };
        return checked_paired_effect_interval(panel, estimator, &percentile, ctx);
    }
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
    let prepared = prepare_studentization(panel, estimator, point, ctx)?;
    let replicates = run_replicates(panel, estimator, config, ctx)?;
    let receipt = studentize_replicates(panel, point, &replicates, prepared, ctx)?;
    let se = receipt.standard_error;
    let mut points = receipt.pivots.iter().flatten().copied().collect::<Vec<_>>();
    let studentization = Some(receipt);
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
    let (lower, upper) = (point - high * se, point - low * se);
    Ok(DependentInterval {
        estimand: estimator.label(),
        point,
        lower,
        upper,
        level: config.level,
        method: config.method,
        studentization,
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

/// Percentile/basic resampling of a checked paired effect, sharing each unit draw.
/// This helper returns an unmeasured receipt; public authority additionally requires
/// the original checked source/proof and the exact calibrated effect protocol.
/// # Errors
/// Invalid configuration, unsupported resampled histories or cancellation.
#[doc(hidden)]
pub fn checked_paired_effect_interval<E: TemporalEstimator + ?Sized>(
    panel: &TemporalUnitPanel,
    estimator: &E,
    config: &DependentIntervalConfig,
    ctx: &ExecutionContext,
) -> Result<DependentInterval, EstimationError> {
    validate_config(config)?;
    if config.method == IntervalMethod::Studentized {
        return dependent_unit_interval(panel, estimator, config, ctx);
    }
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
    let mut points = replicates.iter().filter_map(|record| record.point).collect::<Vec<_>>();
    let failed = replicates.len() - points.len();
    #[allow(clippy::cast_precision_loss, reason = "replicate counts are at most 2000")]
    let fraction = failed as f64 / replicates.len() as f64;
    if fraction > config.max_failed_fraction || points.len() < 2 {
        return Err(unsupported_history(format!(
            "{failed} of {} unit-resampled replicates left a needed history unsupported, above the allowed fraction {}",
            replicates.len(),
            config.max_failed_fraction
        )));
    }
    points.sort_by(f64::total_cmp);
    let tail = (1.0 - config.level) / 2.0;
    let (low, high) = (quantile(&points, tail), quantile(&points, 1.0 - tail));
    let (lower, upper) = match config.method {
        IntervalMethod::Percentile => {
            bias_corrected_accelerated(&all, estimator, point, &points, tail).unwrap_or((low, high))
        }
        IntervalMethod::Basic => (2.0_f64.mul_add(point, -high), 2.0_f64.mul_add(point, -low)),
        IntervalMethod::Studentized => {
            return Err(EstimationError::unsupported(
                "studentized temporal intervals are built on the studentized path",
            ));
        }
    };
    Ok(DependentInterval {
        estimand: estimator.label(),
        point,
        lower,
        upper,
        level: config.level,
        method: config.method,
        studentization: None,
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

#[cfg(test)]
mod digest_tests {
    use super::*;

    /// The mean outcome of the drawn units' histories, in draw order.
    struct MeanOutcome;

    impl TemporalEstimator for MeanOutcome {
        fn label(&self) -> &'static str {
            "mean_outcome"
        }

        fn estimate(&self, units: &[&UnitHistories]) -> Result<f64, EstimationError> {
            let ys: Vec<f64> = units.iter().flat_map(|u| u.histories.iter().map(|h| h.y)).collect();
            Ok(ys.iter().sum::<f64>() / ys.len() as f64)
        }
    }

    fn history(time_id: u64, s0: u32, a1: u32, l2: u32, a2: u32, y: f64) -> SequenceHistory {
        SequenceHistory { time_id, s0, a1, l2, a2, y }
    }

    fn panel() -> TemporalUnitPanel {
        let units = vec![
            UnitHistories {
                unit_id: 5,
                histories: vec![history(0, 0, 1, 0, 1, 0.5), history(1, 1, 0, 1, 0, 1.0)],
            },
            UnitHistories { unit_id: 9, histories: vec![history(0, 0, 0, 0, 0, 2.0)] },
            UnitHistories { unit_id: 12, histories: vec![history(3, 1, 1, 1, 1, -1.5)] },
        ];
        TemporalUnitPanel::new("snap", Some(units)).unwrap()
    }

    /// Golden vectors: every value below was derived independently in Python from the
    /// documented framing (a length-prefixed tag, little-endian words, real bit patterns),
    /// a from-scratch BLAKE3, and the `SplitMix64` unit draws.
    #[test]
    fn panel_replicate_and_set_digests_match_the_independent_derivation() {
        let panel = panel();
        assert_eq!(panel.digest, 18_395_352_411_797_752_298);
        let config = DependentIntervalConfig { seed: 7, ..DependentIntervalConfig::default() };
        let ctx = ExecutionContext::for_tests(1);
        let mut config = config;
        config.replicates = 3;
        let records = run_replicates(&panel, &MeanOutcome, &config, &ctx).unwrap();
        let expected = [
            (13_596_569_364_991_760_287_u64, 9_528_841_699_978_479_609_u64, -1.0 / 3.0),
            (8_518_135_389_417_753_826, 18_080_053_928_592_104_426, -1.0 / 3.0),
            (1_008_529_447_291_554_322, 11_298_647_489_980_813_686, 2.0),
        ];
        assert_eq!(records.len(), 3);
        for (record, (id, selection, point)) in records.iter().zip(expected) {
            assert_eq!(record.replicate_id, id);
            assert_eq!(record.selection_digest, selection);
            assert!((record.point.unwrap() - point).abs() < 1e-15);
        }
        assert_eq!(replicate_set_digest(7, panel.digest, &records), 12_687_347_855_711_543_481);
        // A different seed or panel changes every replicate id.
        let other = DependentIntervalConfig { seed: 8, replicates: 3, ..config };
        let moved = run_replicates(&panel, &MeanOutcome, &other, &ctx).unwrap();
        assert!(moved.iter().zip(&records).all(|(a, b)| a.replicate_id != b.replicate_id));
    }

    #[test]
    fn calibration_basis_keeps_counts_that_fit() {
        let interval = DependentInterval {
            estimand: "mean",
            point: 1.0,
            lower: 0.0,
            upper: 2.0,
            level: 0.95,
            method: IntervalMethod::Percentile,
            studentization: None,
            replicates: vec![ReplicateRecord {
                index: 0,
                replicate_id: 1,
                selection_digest: 2,
                point: Some(1.0),
            }],
            failed: 0,
            units: 3,
            snapshot_id: "snap".to_owned(),
            panel_digest: 4,
            seed: 5,
            claim: INTERVAL_CLAIM,
            calibration: INTERVAL_CALIBRATION_STATUS,
        };
        let basis = interval.calibration_basis();
        assert_eq!(basis.row_count, 3);
        assert_eq!(basis.replicates_ok, Some(1));
    }
}
