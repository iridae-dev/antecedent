//! Independent artifact for the semantic refresh of a prepared horizon-two result
//! onto a new observation period (2.3A, X5 `new_period_refresh`).
//!
//! Format version 1. The artifact records the old and the new window identity (graph,
//! horizon, lag alignment, intervention history, selection targets, regimes,
//! observation period, unit ids, snapshot and proof), a summary of the panel behind
//! each, the target initial-state law and action sequence the result is a function
//! of, the refresh decision (`reusable`, or `invalidated` with its typed reason), the
//! refresh receipt with its digest, whether an interval of the old window was
//! invalidated, and the old and re-evaluated points. A consumer trusts nothing: it
//! rebuilds both identities and checks each against its own panel summary (a changed
//! period, snapshot or unit set is refused), re-decides the refresh, re-evaluates both
//! points from the summaries and rebuilds the receipt, and accepts only what matches
//! bit for bit. A refreshed result never carries an interval; an artifact that does
//! is refused as `temporal_refresh.stale_interval`. The claim is point only.

use crate::IoError;
use crate::temporal_initial_state_artifact::{
    ContributionWire, InitialStateLawWire, PanelSummaryWire, TemporalInitialStateConsumeLimits,
    TemporalPremisesWire, hex64, mismatch,
};
use antecedent_core::{IdentityDomain, reason_code};
use antecedent_estimate::temporal_dependent_interval::TemporalUnitPanel;
use antecedent_estimate::temporal_initial_state::{
    InitialStateSpec, MarginalizedEffect, MarginalizedQuery,
};
use antecedent_estimate::temporal_refresh::{
    ObservationPeriod, RefreshDecision, RefreshReceipt, TEMPORAL_REFRESH_HORIZON,
    TEMPORAL_REFRESH_INFERENCE_CLAIM, TemporalWindowIdentity, decide_refresh,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// The artifact format this reader writes and accepts.
pub const TEMPORAL_REFRESH_ARTIFACT_VERSION: u32 = 1;
/// The feature marker of the accepted format.
pub const TEMPORAL_REFRESH_ARTIFACT_FEATURE: &str = "temporal_new_period_refresh_v1";
/// The artifact kind; a different kind is a different artifact.
pub const TEMPORAL_REFRESH_ARTIFACT_KIND: &str = "temporal_new_period_refresh";

const PREFIX: &str = "temporal_refresh_artifact";
const REUSABLE: &str = "reusable";
const INVALIDATED: &str = "invalidated";

fn same(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits()
}

fn refusal(code: &'static str, detail: &str, message: &str) -> IoError {
    IoError::Refused { code, message: format!("{detail}: {message}") }
}

fn invalid_replacement(message: &str) -> IoError {
    refusal(reason_code!("invalid_argument"), "temporal_refresh.invalid_replacement", message)
}

/// Strictly increasing strings as a set; anything else is not canonical.
fn canonical_set(items: &[String]) -> Result<BTreeSet<String>, IoError> {
    if items.windows(2).any(|w| w[0] >= w[1]) {
        return Err(mismatch(PREFIX, "identity_not_canonical"));
    }
    Ok(items.iter().cloned().collect())
}

/// One window identity on the wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct WindowIdentityWire {
    /// Graph identity.
    pub graph_id: String,
    /// Horizon.
    pub horizon: usize,
    /// Each coordinate with the slice it is aligned to.
    pub lag_alignment: Vec<(String, u8)>,
    /// Ordered action label of each step.
    pub intervention_history: Vec<String>,
    /// Selection targets, strictly increasing.
    pub selection_targets: Vec<String>,
    /// Regimes, strictly increasing.
    pub regimes: Vec<String>,
    /// First time index observed.
    pub period_start: i64,
    /// One past the last time index observed.
    pub period_end: i64,
    /// Unit ids, strictly increasing.
    pub unit_ids: Vec<String>,
    /// Snapshot identity.
    pub snapshot_id: String,
    /// Proof identity.
    pub proof_id: String,
}

impl WindowIdentityWire {
    /// Encode an identity.
    #[must_use]
    pub fn from_identity(identity: &TemporalWindowIdentity) -> Self {
        Self {
            graph_id: identity.graph_id.clone(),
            horizon: identity.horizon,
            lag_alignment: identity.lag_alignment.clone(),
            intervention_history: identity.intervention_history.clone(),
            selection_targets: identity.selection_targets.iter().cloned().collect(),
            regimes: identity.regimes.iter().cloned().collect(),
            period_start: identity.period.start,
            period_end: identity.period.end,
            unit_ids: identity.unit_ids.iter().cloned().collect(),
            snapshot_id: identity.snapshot_id.clone(),
            proof_id: identity.proof_id.clone(),
        }
    }

    /// Decode an identity, refusing a non-canonical set.
    ///
    /// # Errors
    /// An identity whose sets are not strictly increasing.
    pub fn to_identity(&self) -> Result<TemporalWindowIdentity, IoError> {
        Ok(TemporalWindowIdentity {
            graph_id: self.graph_id.clone(),
            horizon: self.horizon,
            lag_alignment: self.lag_alignment.clone(),
            intervention_history: self.intervention_history.clone(),
            selection_targets: canonical_set(&self.selection_targets)?,
            regimes: canonical_set(&self.regimes)?,
            period: ObservationPeriod { start: self.period_start, end: self.period_end },
            unit_ids: canonical_set(&self.unit_ids)?,
            snapshot_id: self.snapshot_id.clone(),
            proof_id: self.proof_id.clone(),
        })
    }

    /// Check this identity against the panel summary it is the window of: the same
    /// snapshot, exactly the panel's units, and a period that covers every time id.
    ///
    /// # Errors
    /// `invalid_argument` / `temporal_refresh.invalid_replacement`.
    pub fn check_against(&self, summary: &PanelSummaryWire) -> Result<(), IoError> {
        if self.period_end <= self.period_start {
            return Err(invalid_replacement("the period is empty or reversed"));
        }
        if self.snapshot_id != summary.snapshot_id {
            return Err(invalid_replacement("the window's snapshot is not the panel's"));
        }
        let units = summary.unit_ids.iter().map(ToString::to_string).collect::<BTreeSet<_>>();
        let named = self.unit_ids.iter().cloned().collect::<BTreeSet<_>>();
        if units.is_empty() || units != named || named.len() != self.unit_ids.len() {
            return Err(invalid_replacement("the window's units are not the panel's units"));
        }
        let covered = |time: u64| {
            i64::try_from(time).is_ok_and(|t| self.period_start <= t && t < self.period_end)
        };
        let all_inside = summary.time_min.absent_or_inside(covered)
            && summary.time_max.absent_or_inside(covered);
        if !all_inside {
            return Err(invalid_replacement("a history lies outside the window's period"));
        }
        Ok(())
    }
}

/// `Option<u64>` helper: absent counts as inside.
trait InsideExt {
    fn absent_or_inside(self, inside: impl Fn(u64) -> bool) -> bool;
}

impl InsideExt for Option<u64> {
    fn absent_or_inside(self, inside: impl Fn(u64) -> bool) -> bool {
        match self {
            Some(time) => inside(time),
            None => true,
        }
    }
}

/// A refresh receipt on the wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReceiptWire {
    /// Digest of the old identity.
    pub old_identity_digest: String,
    /// Digest of the new identity.
    pub new_identity_digest: String,
    /// Old period `(start, end)`.
    pub old_period: (i64, i64),
    /// New period `(start, end)`.
    pub new_period: (i64, i64),
    /// Old snapshot id.
    pub old_snapshot_id: String,
    /// New snapshot id.
    pub new_snapshot_id: String,
    /// The proof id.
    pub proof_id: String,
    /// Whether an interval or fit of the old window was invalidated.
    pub interval_invalidated: bool,
    /// Always `point_only`.
    pub inference_claim: String,
    /// The receipt's seal.
    pub digest: String,
}

impl ReceiptWire {
    /// Encode a receipt.
    #[must_use]
    pub fn from_receipt(receipt: &RefreshReceipt) -> Self {
        Self {
            old_identity_digest: receipt.old_identity_digest.clone(),
            new_identity_digest: receipt.new_identity_digest.clone(),
            old_period: (receipt.old_period.start, receipt.old_period.end),
            new_period: (receipt.new_period.start, receipt.new_period.end),
            old_snapshot_id: receipt.old_snapshot_id.clone(),
            new_snapshot_id: receipt.new_snapshot_id.clone(),
            proof_id: receipt.proof_id.clone(),
            interval_invalidated: receipt.interval_invalidated,
            inference_claim: receipt.inference_claim.into(),
            digest: receipt.digest.clone(),
        }
    }

    /// Decode a receipt, refusing any claim but the point-only one.
    ///
    /// # Errors
    /// A foreign inference claim.
    pub fn to_receipt(&self) -> Result<RefreshReceipt, IoError> {
        if self.inference_claim != TEMPORAL_REFRESH_INFERENCE_CLAIM {
            return Err(mismatch(PREFIX, "inference_claim"));
        }
        let period = |(start, end): (i64, i64)| ObservationPeriod { start, end };
        Ok(RefreshReceipt {
            old_identity_digest: self.old_identity_digest.clone(),
            new_identity_digest: self.new_identity_digest.clone(),
            old_period: period(self.old_period),
            new_period: period(self.new_period),
            old_snapshot_id: self.old_snapshot_id.clone(),
            new_snapshot_id: self.new_snapshot_id.clone(),
            proof_id: self.proof_id.clone(),
            interval_invalidated: self.interval_invalidated,
            inference_claim: TEMPORAL_REFRESH_INFERENCE_CLAIM,
            digest: self.digest.clone(),
        })
    }
}

/// The receipt a reusable refresh from `old` to `new` carries, rebuilt from the
/// identities alone.
#[must_use]
pub fn expected_receipt(
    old: &TemporalWindowIdentity,
    new: &TemporalWindowIdentity,
    interval_invalidated: bool,
) -> RefreshReceipt {
    let mut receipt = RefreshReceipt {
        old_identity_digest: old.digest(),
        new_identity_digest: new.digest(),
        old_period: old.period,
        new_period: new.period,
        old_snapshot_id: old.snapshot_id.clone(),
        new_snapshot_id: new.snapshot_id.clone(),
        proof_id: new.proof_id.clone(),
        interval_invalidated,
        inference_claim: TEMPORAL_REFRESH_INFERENCE_CLAIM,
        digest: String::new(),
    };
    receipt.digest = receipt.compute_digest();
    receipt
}

/// The re-evaluated point of a reusable refresh.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RefreshedResultWire {
    /// Always `marginalized_initial_state`.
    pub label: String,
    /// The re-evaluated value on the new period.
    pub value: f64,
    /// Contributions of every positive-mass state.
    pub contributions: Vec<ContributionWire>,
    /// A refreshed result never carries an interval; always `false`.
    pub interval_present: bool,
}

/// The replayed values of a consumed refresh artifact.
#[derive(Clone, Debug, PartialEq)]
pub struct RefreshReplay {
    /// The old point, re-evaluated.
    pub old_value: f64,
    /// The new point, re-evaluated; `None` for an invalidated refresh.
    pub new_value: Option<f64>,
    /// The re-decided refresh.
    pub decision: RefreshDecision,
}

/// The versioned new-period refresh artifact.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TemporalRefreshArtifactWire {
    /// Format version.
    pub version: u32,
    /// Required feature marker.
    pub required_features: Vec<String>,
    /// Artifact kind.
    pub kind: String,
    /// Initial-state variable, time order, regimes, graph and proof of the held result.
    pub premises: TemporalPremisesWire,
    /// The ordered action sequence.
    pub sequence: [u32; 2],
    /// The target initial-state law the result is a function of.
    pub law: InitialStateLawWire,
    /// The held result's window.
    pub old_window: WindowIdentityWire,
    /// The replacement window.
    pub new_window: WindowIdentityWire,
    /// The panel behind the old window.
    pub old_panel: PanelSummaryWire,
    /// The panel behind the new window.
    pub new_panel: PanelSummaryWire,
    /// The old point.
    pub old_value: f64,
    /// `reusable` or `invalidated`.
    pub decision: String,
    /// The typed invalidation detail, when the decision is `invalidated`.
    pub invalidation: Option<String>,
    /// Whether an interval or fit existed for the old window and was invalidated.
    pub interval_invalidated: bool,
    /// The re-evaluated result of a reusable refresh.
    pub refreshed: Option<RefreshedResultWire>,
    /// The receipt of a reusable refresh.
    pub receipt: Option<ReceiptWire>,
    /// Seal over every field above.
    pub seal: String,
}

#[derive(Deserialize)]
struct VersionPeek {
    version: u32,
}

/// The wire spelling of a decision.
fn decision_labels(decision: RefreshDecision) -> (String, Option<String>) {
    match decision {
        RefreshDecision::Reusable => (REUSABLE.into(), None),
        RefreshDecision::Invalidated(reason) => (INVALIDATED.into(), Some(reason.detail().into())),
    }
}

/// Everything a refresh artifact is built from.
#[derive(Clone, Copy, Debug)]
pub struct RefreshInputs<'a> {
    /// Premises of the held result.
    pub premises: &'a TemporalPremisesWire,
    /// Action sequence.
    pub sequence: [u32; 2],
    /// The target-marginal query.
    pub query: &'a MarginalizedQuery,
    /// Window of the held result.
    pub old_identity: &'a TemporalWindowIdentity,
    /// Panel of the held result.
    pub old_panel: &'a TemporalUnitPanel,
    /// The replacement window.
    pub new_identity: &'a TemporalWindowIdentity,
    /// The replacement panel.
    pub new_panel: &'a TemporalUnitPanel,
    /// Whether an interval or fit existed for the old window.
    pub interval_invalidated: bool,
}

impl TemporalRefreshArtifactWire {
    /// Build the artifact of a refresh, whichever way it was decided, and replay it
    /// once so a producer never writes an artifact its consumer would refuse.
    ///
    /// # Errors
    /// Invalid premises or windows, a state without support, or an encoding failure.
    pub fn checked(inputs: &RefreshInputs<'_>) -> Result<Self, IoError> {
        inputs.premises.validate()?;
        let decision = decide_refresh(inputs.old_identity, inputs.new_identity);
        let old_effect = inputs.query.effect(inputs.old_panel)?;
        let (label, invalidation) = decision_labels(decision);
        let (refreshed, receipt) = match decision {
            RefreshDecision::Reusable => {
                let effect = inputs.query.effect(inputs.new_panel)?;
                let receipt = expected_receipt(
                    inputs.old_identity,
                    inputs.new_identity,
                    inputs.interval_invalidated,
                );
                (Some(refreshed_wire(&effect)), Some(ReceiptWire::from_receipt(&receipt)))
            }
            RefreshDecision::Invalidated(_) => (None, None),
        };
        let mut wire = Self {
            version: TEMPORAL_REFRESH_ARTIFACT_VERSION,
            required_features: vec![TEMPORAL_REFRESH_ARTIFACT_FEATURE.into()],
            kind: TEMPORAL_REFRESH_ARTIFACT_KIND.into(),
            premises: inputs.premises.clone(),
            sequence: inputs.sequence,
            law: InitialStateLawWire::from_law(inputs.query.law()),
            old_window: WindowIdentityWire::from_identity(inputs.old_identity),
            new_window: WindowIdentityWire::from_identity(inputs.new_identity),
            old_panel: PanelSummaryWire::from_panel(inputs.old_panel, inputs.sequence),
            new_panel: PanelSummaryWire::from_panel(inputs.new_panel, inputs.sequence),
            old_value: old_effect.value,
            decision: label,
            invalidation,
            interval_invalidated: inputs.interval_invalidated,
            refreshed,
            receipt,
            seal: String::new(),
        };
        wire.seal = wire.compute_seal()?;
        wire.verify(&TemporalInitialStateConsumeLimits::default())?;
        Ok(wire)
    }

    /// The seal these fields imply. A consumer never trusts it.
    ///
    /// # Errors
    /// Encoding failure.
    pub fn compute_seal(&self) -> Result<String, IoError> {
        let mut body = self.clone();
        body.seal = String::new();
        Ok(crate::identity::digest_wire(
            IdentityDomain::TransportCertificate,
            &("temporal_refresh_seal_v1", body),
        )?
        .to_hex())
    }

    /// Encode as CBOR.
    ///
    /// # Errors
    /// Encoding failure.
    pub fn export(&self) -> Result<Vec<u8>, IoError> {
        crate::to_cbor(self)
    }

    /// Decode, refusing any other version, feature or kind first.
    ///
    /// # Errors
    /// An unsupported version, a decoding failure, or a foreign feature or kind.
    pub fn decode(bytes: &[u8]) -> Result<Self, IoError> {
        let peek: VersionPeek = crate::from_cbor(bytes)?;
        if peek.version != TEMPORAL_REFRESH_ARTIFACT_VERSION {
            return Err(IoError::UnsupportedVersion { version: peek.version });
        }
        let wire: Self = crate::from_cbor(bytes)?;
        if wire.required_features != [TEMPORAL_REFRESH_ARTIFACT_FEATURE]
            || wire.kind != TEMPORAL_REFRESH_ARTIFACT_KIND
        {
            return Err(mismatch(PREFIX, "unsupported_semantics"));
        }
        Ok(wire)
    }

    /// Re-decide the refresh and replay every stored value.
    ///
    /// # Errors
    /// A changed seal; a window that is not its panel's (`invalid_argument`); a
    /// stored `reusable` decision the identities no longer support (the typed
    /// `route_not_supported` invalidation); a source-labelled law; a stale interval;
    /// a receipt that is not the one the identities imply; or any stored value that
    /// the replay does not reproduce bit for bit.
    pub fn verify(
        &self,
        limits: &TemporalInitialStateConsumeLimits,
    ) -> Result<RefreshReplay, IoError> {
        if self.compute_seal()? != self.seal {
            return Err(mismatch(PREFIX, "seal"));
        }
        self.premises.validate()?;
        let query = MarginalizedQuery::new(
            self.sequence,
            InitialStateSpec::Law(self.law.to_law_unchecked()?),
        )?;
        self.law.check_digest(query.law())?;
        self.old_panel.check(PREFIX, limits)?;
        self.new_panel.check(PREFIX, limits)?;
        let (old, new) = (self.old_window.to_identity()?, self.new_window.to_identity()?);
        self.old_window.check_against(&self.old_panel)?;
        self.new_window.check_against(&self.new_panel)?;
        self.check_old_premises(&old)?;
        let decision = self.check_decision(&old, &new)?;
        let (old_value, _) = self.old_panel.marginalized(query.law())?;
        if !same(old_value, self.old_value) {
            return Err(mismatch(PREFIX, "old_value"));
        }
        let new_value = match decision {
            RefreshDecision::Reusable => Some(self.check_reusable(&query, &old, &new)?),
            RefreshDecision::Invalidated(_) => {
                if self.refreshed.is_some() || self.receipt.is_some() {
                    return Err(mismatch(PREFIX, "invalidated_carries_result"));
                }
                None
            }
        };
        Ok(RefreshReplay { old_value, new_value, decision })
    }

    /// The held result's window must be the premises' graph, proof and regimes.
    fn check_old_premises(&self, old: &TemporalWindowIdentity) -> Result<(), IoError> {
        let regimes = BTreeSet::from([
            self.premises.source_regime.clone(),
            self.premises.target_regime.clone(),
        ]);
        if old.graph_id != self.premises.graph_id
            || old.proof_id != self.premises.proof_id
            || old.regimes != regimes
            || old.horizon != TEMPORAL_REFRESH_HORIZON
        {
            return Err(mismatch(PREFIX, "premises"));
        }
        Ok(())
    }

    /// Re-decide; a stored `reusable` the identities no longer support is the
    /// typed invalidation, never a quiet pass.
    fn check_decision(
        &self,
        old: &TemporalWindowIdentity,
        new: &TemporalWindowIdentity,
    ) -> Result<RefreshDecision, IoError> {
        let decision = decide_refresh(old, new);
        let (label, invalidation) = decision_labels(decision);
        if self.decision == label && self.invalidation == invalidation {
            return Ok(decision);
        }
        if self.decision == REUSABLE {
            if let RefreshDecision::Invalidated(reason) = decision {
                return Err(reason.into_error().into());
            }
        }
        Err(mismatch(PREFIX, "decision"))
    }

    fn check_reusable(
        &self,
        query: &MarginalizedQuery,
        old: &TemporalWindowIdentity,
        new: &TemporalWindowIdentity,
    ) -> Result<f64, IoError> {
        let (Some(refreshed), Some(receipt)) = (&self.refreshed, &self.receipt) else {
            return Err(mismatch(PREFIX, "reusable_without_result"));
        };
        if refreshed.interval_present {
            return Err(refusal(
                reason_code!("route_not_supported"),
                "temporal_refresh.stale_interval",
                "a refreshed result may not carry an interval from the old window",
            ));
        }
        if refreshed.label != MarginalizedEffect::LABEL {
            return Err(mismatch(PREFIX, "label"));
        }
        let (value, contributions) = self.new_panel.marginalized(query.law())?;
        let rows_match = refreshed.contributions.len() == contributions.len()
            && refreshed.contributions.iter().zip(&contributions).all(|(a, b)| {
                a.state == b.state && same(a.mass, b.mass) && same(a.response, b.response)
            });
        if !same(refreshed.value, value) || !rows_match {
            return Err(mismatch(PREFIX, "new_value"));
        }
        let stored = receipt.to_receipt()?;
        stored.verify()?;
        if stored != expected_receipt(old, new, self.interval_invalidated) {
            return Err(refusal(
                reason_code!("route_not_supported"),
                "temporal_refresh.receipt_mismatch",
                "the receipt is not the one the old and new windows imply",
            ));
        }
        Ok(value)
    }

    /// Decode, re-decide and replay.
    ///
    /// # Errors
    /// As [`Self::decode`] and [`Self::verify`].
    pub fn consume(
        bytes: &[u8],
        limits: &TemporalInitialStateConsumeLimits,
    ) -> Result<(Self, RefreshReplay), IoError> {
        let wire = Self::decode(bytes)?;
        let replay = wire.verify(limits)?;
        Ok((wire, replay))
    }

    /// Digest of the new window's identity, as the receipt records it.
    ///
    /// # Errors
    /// A non-canonical identity.
    pub fn new_identity_digest(&self) -> Result<String, IoError> {
        Ok(self.new_window.to_identity()?.digest())
    }
}

fn refreshed_wire(effect: &MarginalizedEffect) -> RefreshedResultWire {
    RefreshedResultWire {
        label: effect.label().into(),
        value: effect.value,
        contributions: effect
            .contributions
            .iter()
            .map(|c| ContributionWire { state: c.s0, mass: c.mass, response: c.response })
            .collect(),
        interval_present: false,
    }
}

/// Lower-case hex of a 64-bit identity, for callers that tag a panel.
#[must_use]
pub fn panel_digest_hex(panel: &TemporalUnitPanel) -> String {
    hex64(panel.digest())
}
