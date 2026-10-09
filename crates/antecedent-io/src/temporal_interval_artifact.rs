//! Internal replay artifact for the dependence-preserving temporal interval
//! (2.3A, X5 `dependent_temporal_interval`).
//!
//! Format version 1. **Internal**: the public interval route is closed while the
//! interval's calibration is unmeasured, so no public producer writes this artifact;
//! it exists so the interval's whole resampling is reproducible and checkable. The
//! artifact embeds the unit panel (snapshot id, unit ids, time ids and values), the
//! estimator, the resampling design, and the interval with every replicate: its
//! whole-unit replicate id, its unit-selection digest and its re-estimated point.
//! The calibration is recorded as `unmeasured`; no coverage claim is made. A
//! consumer rebuilds the panel, re-runs the whole resampling and the whole
//! estimator on every replicate, and accepts only an interval identical to the
//! stored one, so a changed unit, time id, value or snapshot is refused even when
//! the artifact is resealed. An optional refresh receipt links the interval to the
//! refresh that produced its panel.

use crate::IoError;
use crate::temporal_initial_state_artifact::{InitialStateLawWire, hex64, mismatch};
use crate::temporal_refresh_artifact::ReceiptWire;
use antecedent_core::{ExecutionContext, IdentityDomain};
use antecedent_estimate::temporal_dependent_interval::{
    DependentInterval, DependentIntervalConfig, INTERVAL_CALIBRATION_STATUS, INTERVAL_CLAIM,
    IntervalMethod, ObservedStateSequence, SequenceHistory, TemporalEstimator, TemporalUnitPanel,
    UnitHistories, dependent_unit_interval,
};
use antecedent_estimate::temporal_initial_state::{
    FixedStateEffect, FixedStateQuery, InitialStateSpec, MarginalizedEffect, MarginalizedQuery,
};
use antecedent_estimate::temporal_refresh::RefreshReceipt;
use serde::{Deserialize, Serialize};

/// The artifact format this reader writes and accepts.
pub const TEMPORAL_INTERVAL_ARTIFACT_VERSION: u32 = 1;
/// The feature marker of the accepted format.
pub const TEMPORAL_INTERVAL_ARTIFACT_FEATURE: &str = "temporal_dependent_interval_v1";
/// The artifact kind; a different kind is a different artifact.
pub const TEMPORAL_INTERVAL_ARTIFACT_KIND: &str = "temporal_dependent_interval_internal";

const PREFIX: &str = "temporal_interval_artifact";

/// Consumer bounds. Nothing the artifact stores raises them.
#[derive(Clone, Copy, Debug)]
pub struct TemporalIntervalConsumeLimits {
    /// Most units a panel may carry.
    pub max_units: usize,
    /// Most complete histories over all units.
    pub max_histories: usize,
    /// Most replicates the replay will run.
    pub max_replicates: usize,
}

impl Default for TemporalIntervalConsumeLimits {
    fn default() -> Self {
        Self { max_units: 100_000, max_histories: 2_000_000, max_replicates: 2000 }
    }
}

/// One complete history on the wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct HistoryWire {
    /// Period id.
    pub time_id: u64,
    /// Pre-action state.
    pub s0: u32,
    /// First action.
    pub a1: u32,
    /// Step-2 covariate.
    pub l2: u32,
    /// Second action.
    pub a2: u32,
    /// Outcome.
    pub y: f64,
}

/// One repeated unit on the wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct UnitWire {
    /// Stable unit id.
    pub unit_id: u64,
    /// The unit's histories, strictly increasing in time.
    pub histories: Vec<HistoryWire>,
}

/// The unit panel on the wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct UnitPanelWire {
    /// Snapshot id.
    pub snapshot_id: String,
    /// The units in declared order.
    pub units: Vec<UnitWire>,
}

impl UnitPanelWire {
    /// Encode a panel.
    #[must_use]
    pub fn from_panel(panel: &TemporalUnitPanel) -> Self {
        Self {
            snapshot_id: panel.snapshot_id().into(),
            units: panel
                .units()
                .iter()
                .map(|unit| UnitWire {
                    unit_id: unit.unit_id,
                    histories: unit
                        .histories
                        .iter()
                        .map(|h| HistoryWire {
                            time_id: h.time_id,
                            s0: h.s0,
                            a1: h.a1,
                            l2: h.l2,
                            a2: h.a2,
                            y: h.y,
                        })
                        .collect(),
                })
                .collect(),
        }
    }

    /// Rebuild the panel through the core's own validation.
    ///
    /// # Errors
    /// The core's `temporal_interval.unknown_units` refusals.
    pub fn to_panel(&self) -> Result<TemporalUnitPanel, IoError> {
        let units = self
            .units
            .iter()
            .map(|unit| UnitHistories {
                unit_id: unit.unit_id,
                histories: unit
                    .histories
                    .iter()
                    .map(|h| SequenceHistory {
                        time_id: h.time_id,
                        s0: h.s0,
                        a1: h.a1,
                        l2: h.l2,
                        a2: h.a2,
                        y: h.y,
                    })
                    .collect(),
            })
            .collect();
        Ok(TemporalUnitPanel::new(self.snapshot_id.clone(), Some(units))?)
    }
}

/// The estimator the interval re-runs on every replicate.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct IntervalEstimatorWire {
    /// `observed_initial_state`, `fixed_initial_state` or `marginalized_initial_state`.
    pub kind: String,
    /// Actions of the two steps.
    pub sequence: [u32; 2],
    /// The fixed state (kind `fixed_initial_state` only).
    pub state: Option<u32>,
    /// The target law (kind `marginalized_initial_state` only).
    pub law: Option<InitialStateLawWire>,
}

impl IntervalEstimatorWire {
    /// The estimator over the panel's observed initial-state law.
    #[must_use]
    pub fn observed(sequence: [u32; 2]) -> Self {
        Self { kind: ObservedStateSequence::LABEL.into(), sequence, state: None, law: None }
    }

    /// The fixed-state estimator.
    #[must_use]
    pub fn fixed(query: &FixedStateQuery) -> Self {
        Self {
            kind: FixedStateEffect::LABEL.into(),
            sequence: query.sequence,
            state: Some(query.s0),
            law: None,
        }
    }

    /// The target-marginal estimator.
    #[must_use]
    pub fn marginalized(sequence: [u32; 2], query: &MarginalizedQuery) -> Self {
        Self {
            kind: MarginalizedEffect::LABEL.into(),
            sequence,
            state: None,
            law: Some(InitialStateLawWire::from_law(query.law())),
        }
    }

    /// Rebuild the estimator, refusing a kind with the wrong companion fields.
    ///
    /// # Errors
    /// A foreign kind, a missing or extra field, or a source-labelled law.
    pub fn to_estimator(&self) -> Result<Box<dyn TemporalEstimator>, IoError> {
        match (self.kind.as_str(), self.state, &self.law) {
            (ObservedStateSequence::LABEL, None, None) => {
                Ok(Box::new(antecedent_estimate::temporal_dependent_interval::BalancedTemporalEstimator::Observed(ObservedStateSequence { sequence: self.sequence })))
            }
            (FixedStateEffect::LABEL, Some(s0), None) => {
                Ok(Box::new(antecedent_estimate::temporal_dependent_interval::BalancedTemporalEstimator::Fixed(FixedStateQuery { sequence: self.sequence, s0 })))
            }
            (MarginalizedEffect::LABEL, None, Some(law)) => {
                let query = MarginalizedQuery::new(
                    self.sequence,
                    InitialStateSpec::Law(law.to_law_unchecked()?),
                )?;
                law.check_digest(query.law())?;
                Ok(Box::new(antecedent_estimate::temporal_dependent_interval::BalancedTemporalEstimator::Marginalized(query)))
            }
            _ => Err(mismatch(PREFIX, "estimator")),
        }
    }
}

/// The resampling design on the wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct IntervalConfigWire {
    /// Replicates.
    pub replicates: u64,
    /// Seed of the replicate stream.
    pub seed: u64,
    /// Two-sided level.
    pub level: f64,
    /// `percentile` or `basic`.
    pub method: String,
    /// Fewest units.
    pub min_units: u64,
    /// Largest tolerated fraction of failed replicates.
    pub max_failed_fraction: f64,
}

fn method_label(method: IntervalMethod) -> &'static str {
    match method {
        IntervalMethod::Percentile => "percentile",
        IntervalMethod::Basic => "basic",
        IntervalMethod::Studentized => "studentized",
    }
}

impl IntervalConfigWire {
    /// Encode a design.
    #[must_use]
    pub fn from_config(config: &DependentIntervalConfig) -> Self {
        Self {
            replicates: config.replicates as u64,
            seed: config.seed,
            level: config.level,
            method: method_label(config.method).into(),
            min_units: config.min_units as u64,
            max_failed_fraction: config.max_failed_fraction,
        }
    }

    /// Decode a design.
    ///
    /// # Errors
    /// A foreign method label or a count that does not fit.
    pub fn to_config(&self) -> Result<DependentIntervalConfig, IoError> {
        let method = match self.method.as_str() {
            "percentile" => IntervalMethod::Percentile,
            "basic" => IntervalMethod::Basic,
            "studentized" => IntervalMethod::Studentized,
            _ => return Err(mismatch(PREFIX, "config")),
        };
        let (Ok(replicates), Ok(min_units)) =
            (usize::try_from(self.replicates), usize::try_from(self.min_units))
        else {
            return Err(mismatch(PREFIX, "config"));
        };
        Ok(DependentIntervalConfig {
            replicates,
            seed: self.seed,
            level: self.level,
            method,
            min_units,
            max_failed_fraction: self.max_failed_fraction,
        })
    }
}

/// One replicate on the wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReplicateWire {
    /// Replicate index.
    pub index: u64,
    /// Whole-unit replicate id.
    pub replicate_id: u64,
    /// Digest of the drawn unit ids, in draw order.
    pub selection_digest: u64,
    /// The re-estimated point; `None` when the replicate failed.
    pub point: Option<f64>,
}

/// Original and per-resample variance receipt of the balanced bootstrap-t method.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct StudentizationWire {
    /// Original standard error.
    pub standard_error: f64,
    /// Original unit scores.
    pub unit_scores: Vec<f64>,
    /// Aligned resample standard errors.
    pub replicate_standard_errors: Vec<Option<f64>>,
    /// Aligned centered studentized pivots.
    pub pivots: Vec<Option<f64>>,
}

/// The stored interval.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct IntervalResultWire {
    /// Estimand label of the estimator.
    pub estimand: String,
    /// Point estimate on the original panel.
    pub point: f64,
    /// Lower bound.
    pub lower: f64,
    /// Upper bound.
    pub upper: f64,
    /// Nominal level.
    pub level: f64,
    /// Construction.
    pub method: String,
    /// Absent for historical percentile/basic artifacts, preserving their encoding.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub studentization: Option<StudentizationWire>,
    /// Every replicate, in order.
    pub replicates: Vec<ReplicateWire>,
    /// Replicates dropped as failed.
    pub failed: u64,
    /// Units of the panel.
    pub units: u64,
    /// Snapshot id of the panel.
    pub snapshot_id: String,
    /// Identity of the panel (hex).
    pub panel_digest: String,
    /// Seed.
    pub seed: u64,
    /// Always the engine's claim: dependence preserved, calibration unmeasured.
    pub claim: String,
    /// Always `unmeasured`.
    pub calibration: String,
    /// Digest over every replicate id, selection digest and point (hex).
    pub replicate_digest: String,
}

impl IntervalResultWire {
    /// Encode an interval.
    #[must_use]
    pub fn from_interval(interval: &DependentInterval) -> Self {
        Self {
            estimand: interval.estimand.into(),
            point: interval.point,
            lower: interval.lower,
            upper: interval.upper,
            level: interval.level,
            method: method_label(interval.method).into(),
            studentization: interval.studentization.as_ref().map(|receipt| StudentizationWire {
                standard_error: receipt.standard_error,
                unit_scores: receipt.unit_scores.clone(),
                replicate_standard_errors: receipt.replicate_standard_errors.clone(),
                pivots: receipt.pivots.clone(),
            }),
            replicates: interval
                .replicates
                .iter()
                .map(|r| ReplicateWire {
                    index: r.index as u64,
                    replicate_id: r.replicate_id,
                    selection_digest: r.selection_digest,
                    point: r.point,
                })
                .collect(),
            failed: interval.failed as u64,
            units: interval.units as u64,
            snapshot_id: interval.snapshot_id.clone(),
            panel_digest: hex64(interval.panel_digest),
            seed: interval.seed,
            claim: interval.claim.into(),
            calibration: interval.calibration.into(),
            replicate_digest: hex64(interval.replicate_digest()),
        }
    }
}

/// The versioned internal interval artifact.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TemporalIntervalArtifactWire {
    /// Format version.
    pub version: u32,
    /// Required feature marker.
    pub required_features: Vec<String>,
    /// Artifact kind.
    pub kind: String,
    /// The estimator.
    pub estimator: IntervalEstimatorWire,
    /// The unit panel.
    pub panel: UnitPanelWire,
    /// The resampling design.
    pub config: IntervalConfigWire,
    /// The interval and every replicate.
    pub result: IntervalResultWire,
    /// The refresh that produced the panel, when there was one.
    pub refresh_receipt: Option<ReceiptWire>,
    /// Seal over every field above.
    pub seal: String,
}

#[derive(Deserialize)]
struct VersionPeek {
    version: u32,
}

impl TemporalIntervalArtifactWire {
    /// Build the artifact of an interval computed on `panel`, and replay it once.
    ///
    /// # Errors
    /// An interval that is not this panel's, a bad link, or an encoding failure.
    pub fn checked(
        panel: &TemporalUnitPanel,
        estimator: IntervalEstimatorWire,
        config: &DependentIntervalConfig,
        interval: &DependentInterval,
        refresh_receipt: Option<&RefreshReceipt>,
        ctx: &ExecutionContext,
    ) -> Result<Self, IoError> {
        if interval.panel_digest != panel.digest() || interval.snapshot_id != panel.snapshot_id() {
            return Err(mismatch(PREFIX, "foreign_interval"));
        }
        let mut wire = Self {
            version: TEMPORAL_INTERVAL_ARTIFACT_VERSION,
            required_features: vec![TEMPORAL_INTERVAL_ARTIFACT_FEATURE.into()],
            kind: TEMPORAL_INTERVAL_ARTIFACT_KIND.into(),
            estimator,
            panel: UnitPanelWire::from_panel(panel),
            config: IntervalConfigWire::from_config(config),
            result: IntervalResultWire::from_interval(interval),
            refresh_receipt: refresh_receipt.map(ReceiptWire::from_receipt),
            seal: String::new(),
        };
        wire.seal = wire.compute_seal()?;
        wire.verify(&TemporalIntervalConsumeLimits::default(), ctx)?;
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
            &("temporal_interval_seal_v1", body),
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
        if peek.version != TEMPORAL_INTERVAL_ARTIFACT_VERSION {
            return Err(IoError::UnsupportedVersion { version: peek.version });
        }
        let wire: Self = crate::from_cbor(bytes)?;
        if wire.required_features != [TEMPORAL_INTERVAL_ARTIFACT_FEATURE]
            || wire.kind != TEMPORAL_INTERVAL_ARTIFACT_KIND
        {
            return Err(mismatch(PREFIX, "unsupported_semantics"));
        }
        Ok(wire)
    }

    fn check_limits(&self, limits: &TemporalIntervalConsumeLimits) -> Result<(), IoError> {
        let histories: usize = self.panel.units.iter().map(|u| u.histories.len()).sum();
        let replicates = usize::try_from(self.config.replicates).unwrap_or(usize::MAX);
        if self.panel.units.len() > limits.max_units
            || histories > limits.max_histories
            || replicates > limits.max_replicates
        {
            return Err(mismatch(PREFIX, "limits_exceeded"));
        }
        Ok(())
    }

    /// Re-run the whole resampling from the embedded panel and compare.
    ///
    /// # Errors
    /// A changed seal, a foreign calibration or claim, a refresh receipt that is not
    /// the panel's, the core's refusals on the rebuilt panel or design, or an interval
    /// the replay does not reproduce bit for bit.
    pub fn verify(
        &self,
        limits: &TemporalIntervalConsumeLimits,
        ctx: &ExecutionContext,
    ) -> Result<DependentInterval, IoError> {
        if self.compute_seal()? != self.seal {
            return Err(mismatch(PREFIX, "seal"));
        }
        self.check_limits(limits)?;
        if self.result.calibration != INTERVAL_CALIBRATION_STATUS
            || self.result.claim != INTERVAL_CLAIM
        {
            return Err(mismatch(PREFIX, "calibration"));
        }
        let panel = self.panel.to_panel()?;
        if let Some(receipt) = &self.refresh_receipt {
            let receipt = receipt.to_receipt()?;
            receipt.verify()?;
            if receipt.new_snapshot_id != panel.snapshot_id() {
                return Err(mismatch(PREFIX, "refresh_link"));
            }
        }
        let estimator = self.estimator.to_estimator()?;
        let config = self.config.to_config()?;
        let replayed = dependent_unit_interval(&panel, &*estimator, &config, ctx)?;
        if crate::to_cbor(&IntervalResultWire::from_interval(&replayed))?
            != crate::to_cbor(&self.result)?
        {
            return Err(mismatch(PREFIX, "interval_replay"));
        }
        Ok(replayed)
    }

    /// Decode and replay.
    ///
    /// # Errors
    /// As [`Self::decode`] and [`Self::verify`].
    pub fn consume(
        bytes: &[u8],
        limits: &TemporalIntervalConsumeLimits,
        ctx: &ExecutionContext,
    ) -> Result<(Self, DependentInterval), IoError> {
        let wire = Self::decode(bytes)?;
        let replayed = wire.verify(limits, ctx)?;
        Ok((wire, replayed))
    }
}
