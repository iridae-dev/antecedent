//! Semantic refresh of a prepared horizon-two temporal result onto a new observation period
//! (X5 `new_period_refresh`).
//!
//! # Contract
//!
//! A prepared two-step result is valid for one declared window: its time horizon, the lag
//! alignment of its coordinates, its intervention history, its selection targets and regimes,
//! its graph and proof, its observation period, the units observed, and its data snapshot
//! ([`TemporalWindowIdentity`]). A replacement snapshot with its own explicit period and unit
//! ids is checked against that identity by [`decide_refresh`]:
//!
//! * the proof is **reusable** only when the graph, horizon, lag alignment, intervention
//!   history, selection targets, regimes and proof id are all unchanged and both the period
//!   and the snapshot are new. The result is then re-evaluated on the new period (never
//!   copied) and a [`RefreshReceipt`] binds the old and new identities;
//! * anything else is a typed [`RefreshInvalidation`] with a registered refusal
//!   (`route_not_supported`, `temporal_refresh.*`): an appended third slice or changed horizon
//!   (`horizon_changed`), changed lag alignment (`lag_alignment_changed`), altered
//!   intervention history (`intervention_history_changed`), changed selection targets,
//!   regimes, graph or proof (`premises_changed`), and a replacement that is not a new
//!   snapshot of a new period (`stale_snapshot`).
//!
//! A stale proof, fit or interval never survives a refresh: the refreshed result carries no
//! interval, and the receipt's `interval_invalidated` flag records that one existed for the
//! old window. No new identification theorem is claimed; the claim stays point only.
//!
//! The digests here are non-cryptographic content fingerprints (two FNV-1a lanes over
//! length-prefixed fields); they detect accidental or unresealed change, not an adversary.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use antecedent_core::{ExecutionContext, reason_code};
use antecedent_expr::ExactTransportData;

use crate::error::EstimationError;
use crate::temporal_transport::PreparedTemporalSequence;

/// Inference claim of the route.
pub const TEMPORAL_REFRESH_INFERENCE_CLAIM: &str = "point_only";

/// The horizon every refreshable result has.
pub const TEMPORAL_REFRESH_HORIZON: usize = 2;

/// Half-open observation period `[start, end)` in an integer time index.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ObservationPeriod {
    /// First time index observed.
    pub start: i64,
    /// One past the last time index observed.
    pub end: i64,
}

/// Every input a prepared two-step result's validity rests on.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TemporalWindowIdentity {
    /// Identity of the (unrolled) graph the proof was derived on.
    pub graph_id: String,
    /// Time horizon; a refreshable result has [`TEMPORAL_REFRESH_HORIZON`].
    pub horizon: usize,
    /// Lag alignment: each coordinate name with the slice it is aligned to.
    pub lag_alignment: Vec<(String, u8)>,
    /// Ordered action of each step, as labels.
    pub intervention_history: Vec<String>,
    /// Time-indexed selection targets.
    pub selection_targets: BTreeSet<String>,
    /// Regimes of the evidence.
    pub regimes: BTreeSet<String>,
    /// Observation period of the snapshot.
    pub period: ObservationPeriod,
    /// Unit lineage: the repeated-unit ids observed.
    pub unit_ids: BTreeSet<String>,
    /// Identity of the data snapshot.
    pub snapshot_id: String,
    /// Identity of the proof the result was derived under.
    pub proof_id: String,
}

/// Two-lane FNV-1a over length-prefixed fields.
struct Digester(u64, u64);

impl Digester {
    const PRIME: u64 = 0x0100_0000_01b3;

    const fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325, 0x8422_2325_cbf2_9ce4)
    }

    fn byte(&mut self, b: u8) {
        self.0 = (self.0 ^ u64::from(b)).wrapping_mul(Self::PRIME);
        self.1 = (self.1 ^ u64::from(b ^ 0x5c)).wrapping_mul(Self::PRIME).rotate_left(5);
    }

    fn text(&mut self, value: &str) {
        for b in (value.len() as u64).to_le_bytes() {
            self.byte(b);
        }
        for b in value.bytes() {
            self.byte(b);
        }
    }

    fn list<'a>(&mut self, tag: &str, items: impl ExactSizeIterator<Item = &'a str>) {
        self.text(tag);
        self.text(&items.len().to_string());
        for item in items {
            self.text(item);
        }
    }

    fn finish(&self) -> String {
        format!("{:016x}{:016x}", self.0, self.1)
    }
}

impl TemporalWindowIdentity {
    /// Content fingerprint over every identity input.
    #[must_use]
    pub fn digest(&self) -> String {
        let mut d = Digester::new();
        d.text("temporal_window_identity_v1");
        d.text(&self.graph_id);
        d.text(&self.horizon.to_string());
        let lag: Vec<String> =
            self.lag_alignment.iter().map(|(name, slice)| format!("{name}@{slice}")).collect();
        d.list("lag_alignment", lag.iter().map(String::as_str));
        d.list("intervention_history", self.intervention_history.iter().map(String::as_str));
        d.list("selection_targets", self.selection_targets.iter().map(String::as_str));
        d.list("regimes", self.regimes.iter().map(String::as_str));
        d.text(&self.period.start.to_string());
        d.text(&self.period.end.to_string());
        d.list("unit_ids", self.unit_ids.iter().map(String::as_str));
        d.text(&self.snapshot_id);
        d.text(&self.proof_id);
        d.finish()
    }

    fn validate(&self) -> Result<(), EstimationError> {
        let invalid = |message: &str| {
            refuse(reason_code!("invalid_argument"), "temporal_refresh.invalid_replacement", message)
        };
        if self.period.end <= self.period.start {
            return Err(invalid("the replacement period is empty or reversed"));
        }
        if self.unit_ids.is_empty() {
            return Err(invalid("the replacement carries no unit ids"));
        }
        if self.snapshot_id.is_empty() || self.proof_id.is_empty() || self.horizon == 0 {
            return Err(invalid("the replacement lacks a snapshot id, proof id or horizon"));
        }
        Ok(())
    }
}

/// Why a refresh invalidates the old result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RefreshInvalidation {
    /// An appended slice or any other change of horizon.
    HorizonChanged,
    /// The lag alignment of the coordinates changed.
    LagAlignmentChanged,
    /// The intervention history was altered.
    InterventionHistoryChanged,
    /// Selection targets, regimes, graph or proof changed.
    PremisesChanged,
    /// The replacement is not a new snapshot of a new period.
    StaleSnapshot,
}

impl RefreshInvalidation {
    /// Namespaced detail literal.
    #[must_use]
    pub const fn detail(self) -> &'static str {
        match self {
            Self::HorizonChanged => "temporal_refresh.horizon_changed",
            Self::LagAlignmentChanged => "temporal_refresh.lag_alignment_changed",
            Self::InterventionHistoryChanged => "temporal_refresh.intervention_history_changed",
            Self::PremisesChanged => "temporal_refresh.premises_changed",
            Self::StaleSnapshot => "temporal_refresh.stale_snapshot",
        }
    }

    const fn message(self) -> &'static str {
        match self {
            Self::HorizonChanged => {
                "the replacement changes the horizon (an appended slice needs its own cell); the \
                 old proof, fit and interval are invalidated"
            }
            Self::LagAlignmentChanged => {
                "the lag alignment of the coordinates changed; the old proof, fit and interval \
                 are invalidated"
            }
            Self::InterventionHistoryChanged => {
                "the intervention history changed; the old proof, fit and interval are \
                 invalidated"
            }
            Self::PremisesChanged => {
                "the selection targets, regimes, graph or proof changed; the old proof, fit and \
                 interval are invalidated"
            }
            Self::StaleSnapshot => {
                "the replacement is not a new snapshot of a new observation period; nothing may \
                 be reused as fresh"
            }
        }
    }

    /// The registered refusal for this invalidation.
    #[must_use]
    pub fn into_error(self) -> EstimationError {
        refuse(reason_code!("route_not_supported"), self.detail(), self.message())
    }
}

/// Outcome of comparing an old identity with a replacement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RefreshDecision {
    /// The proof is reusable; the result must be re-evaluated on the new period.
    Reusable,
    /// The old result is invalidated.
    Invalidated(RefreshInvalidation),
}

/// Compare the old identity with a replacement. Checks run in a fixed order (horizon, lag
/// alignment, intervention history, premises, snapshot/period), each independently
/// sufficient to invalidate.
#[must_use]
pub fn decide_refresh(old: &TemporalWindowIdentity, new: &TemporalWindowIdentity) -> RefreshDecision {
    let invalidated = RefreshDecision::Invalidated;
    if old.horizon != new.horizon || new.horizon != TEMPORAL_REFRESH_HORIZON {
        return invalidated(RefreshInvalidation::HorizonChanged);
    }
    if old.lag_alignment != new.lag_alignment {
        return invalidated(RefreshInvalidation::LagAlignmentChanged);
    }
    if old.intervention_history != new.intervention_history {
        return invalidated(RefreshInvalidation::InterventionHistoryChanged);
    }
    if old.selection_targets != new.selection_targets
        || old.regimes != new.regimes
        || old.graph_id != new.graph_id
        || old.proof_id != new.proof_id
    {
        return invalidated(RefreshInvalidation::PremisesChanged);
    }
    if old.period == new.period || old.snapshot_id == new.snapshot_id {
        return invalidated(RefreshInvalidation::StaleSnapshot);
    }
    RefreshDecision::Reusable
}

/// Receipt binding the old and new period, snapshot and proof of a refresh.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RefreshReceipt {
    /// Digest of the old identity.
    pub old_identity_digest: String,
    /// Digest of the new identity.
    pub new_identity_digest: String,
    /// Old observation period.
    pub old_period: ObservationPeriod,
    /// New observation period.
    pub new_period: ObservationPeriod,
    /// Old snapshot id.
    pub old_snapshot_id: String,
    /// New snapshot id.
    pub new_snapshot_id: String,
    /// The proof id, unchanged by a reusable refresh.
    pub proof_id: String,
    /// Whether an interval or fit existed for the old window and was invalidated.
    pub interval_invalidated: bool,
    /// Always [`TEMPORAL_REFRESH_INFERENCE_CLAIM`].
    pub inference_claim: &'static str,
    /// Seal over every field above.
    pub digest: String,
}

impl RefreshReceipt {
    /// The seal this receipt's fields imply.
    #[must_use]
    pub fn compute_digest(&self) -> String {
        let mut d = Digester::new();
        d.text("temporal_refresh_receipt_v1");
        d.text(&self.old_identity_digest);
        d.text(&self.new_identity_digest);
        for period in [self.old_period, self.new_period] {
            d.text(&period.start.to_string());
            d.text(&period.end.to_string());
        }
        d.text(&self.old_snapshot_id);
        d.text(&self.new_snapshot_id);
        d.text(&self.proof_id);
        d.text(if self.interval_invalidated { "interval_invalidated" } else { "no_interval" });
        d.text(self.inference_claim);
        d.finish()
    }

    fn bind(old: &TemporalWindowIdentity, new: &TemporalWindowIdentity, interval: bool) -> Self {
        let mut receipt = Self {
            old_identity_digest: old.digest(),
            new_identity_digest: new.digest(),
            old_period: old.period,
            new_period: new.period,
            old_snapshot_id: old.snapshot_id.clone(),
            new_snapshot_id: new.snapshot_id.clone(),
            proof_id: new.proof_id.clone(),
            interval_invalidated: interval,
            inference_claim: TEMPORAL_REFRESH_INFERENCE_CLAIM,
            digest: String::new(),
        };
        receipt.digest = receipt.compute_digest();
        receipt
    }

    /// Check the seal against the fields.
    ///
    /// # Errors
    /// `route_not_supported` with `temporal_refresh.receipt_digest_mismatch` when a field was
    /// changed without resealing.
    pub fn verify(&self) -> Result<(), EstimationError> {
        if self.digest == self.compute_digest() {
            Ok(())
        } else {
            Err(refuse(
                reason_code!("route_not_supported"),
                "temporal_refresh.receipt_digest_mismatch",
                "the receipt's seal does not match its fields",
            ))
        }
    }
}

/// A prepared result held with the identity it is valid for.
#[derive(Clone, Debug, PartialEq)]
pub struct HeldTemporalResult<T> {
    /// What the result is valid for.
    pub identity: TemporalWindowIdentity,
    /// The evaluated value.
    pub value: T,
    /// Whether an interval or fit exists for this window.
    pub interval_present: bool,
}

fn refuse(code: &'static str, detail: &str, message: &str) -> EstimationError {
    EstimationError::refused(code, format!("{detail}: {message}"))
}

/// Refresh a held result onto a replacement identity by re-evaluating it there.
///
/// `evaluate` is called with the replacement identity only after the proof is judged
/// reusable; its value is a fresh evaluation, never the old value. The returned result
/// carries no interval.
///
/// # Errors
/// `invalid_argument` (`temporal_refresh.invalid_replacement`) for an empty period, no units
/// or a missing id; the typed [`RefreshInvalidation`] refusal (`route_not_supported`) when the
/// proof is not reusable; any error of `evaluate`.
pub fn refresh_held<T, F>(
    held: &HeldTemporalResult<T>,
    replacement: TemporalWindowIdentity,
    evaluate: F,
) -> Result<(HeldTemporalResult<T>, RefreshReceipt), EstimationError>
where
    F: FnOnce(&TemporalWindowIdentity) -> Result<T, EstimationError>,
{
    replacement.validate()?;
    if let RefreshDecision::Invalidated(reason) = decide_refresh(&held.identity, &replacement) {
        return Err(reason.into_error());
    }
    let value = evaluate(&replacement)?;
    let receipt = RefreshReceipt::bind(&held.identity, &replacement, held.interval_present);
    Ok((HeldTemporalResult { identity: replacement, value, interval_present: false }, receipt))
}

/// Accept a refreshed result only with a receipt that matches both the held result and the
/// refreshed one, so a resealed or foreign receipt is refused.
///
/// # Errors
/// `temporal_refresh.receipt_digest_mismatch` for an unsealed edit;
/// `temporal_refresh.receipt_mismatch` when the receipt's old identity is not the held result's,
/// its new identity, period or snapshot is not the refreshed result's, or its interval flag
/// disagrees with the held result; the typed invalidation when the pair is not a reusable
/// refresh; and `temporal_refresh.stale_interval` when the refreshed result still carries an
/// interval.
pub fn accept_refresh<T>(
    held: &HeldTemporalResult<T>,
    receipt: &RefreshReceipt,
    refreshed: HeldTemporalResult<T>,
) -> Result<HeldTemporalResult<T>, EstimationError> {
    receipt.verify()?;
    let mismatch = |message: &str| {
        refuse(reason_code!("route_not_supported"), "temporal_refresh.receipt_mismatch", message)
    };
    if receipt.old_identity_digest != held.identity.digest()
        || receipt.old_period != held.identity.period
        || receipt.old_snapshot_id != held.identity.snapshot_id
    {
        return Err(mismatch("the receipt's old identity is not the held result's"));
    }
    if receipt.new_identity_digest != refreshed.identity.digest()
        || receipt.new_period != refreshed.identity.period
        || receipt.new_snapshot_id != refreshed.identity.snapshot_id
        || receipt.proof_id != refreshed.identity.proof_id
    {
        return Err(mismatch("the receipt's new identity is not the refreshed result's"));
    }
    if receipt.interval_invalidated != held.interval_present {
        return Err(mismatch("the receipt's interval flag disagrees with the held result"));
    }
    if let RefreshDecision::Invalidated(reason) =
        decide_refresh(&held.identity, &refreshed.identity)
    {
        return Err(reason.into_error());
    }
    if refreshed.interval_present {
        return Err(refuse(
            reason_code!("route_not_supported"),
            "temporal_refresh.stale_interval",
            "a refreshed result may not carry an interval from the old window",
        ));
    }
    Ok(refreshed)
}

/// Refresh a [`PreparedTemporalSequence`] onto new exact laws of a new period.
///
/// `old` must describe the held prepared result (its horizon and intervention history length
/// are checked against the frozen decision); the identity decision runs before the prepared
/// result's own same-window refresh re-evaluates against `data`.
///
/// # Errors
/// As [`refresh_held`], `temporal_refresh.premises_changed` when `old` does not describe
/// `prepared`, and the refusals of [`PreparedTemporalSequence::refresh`].
pub fn refresh_prepared_sequence(
    prepared: &PreparedTemporalSequence,
    old: &TemporalWindowIdentity,
    new: &TemporalWindowIdentity,
    data: ExactTransportData,
    interval_existed: bool,
    ctx: &ExecutionContext,
) -> Result<(PreparedTemporalSequence, RefreshReceipt), EstimationError> {
    let decision = prepared.decision();
    if old.horizon != decision.spec.horizon()
        || old.intervention_history.len() != decision.sequence.len()
    {
        return Err(RefreshInvalidation::PremisesChanged.into_error());
    }
    new.validate()?;
    if let RefreshDecision::Invalidated(reason) = decide_refresh(old, new) {
        return Err(reason.into_error());
    }
    let refreshed = prepared.refresh(data, ctx)?;
    Ok((refreshed, RefreshReceipt::bind(old, new, interval_existed)))
}
