//! Source-bound empirical two-step history recalculation artifact.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::expr_wire::ExprArenaWire;
use crate::{IoError, mz_transport_artifact::MzTransportPointWire};
use antecedent_core::IdentityDomain;
use serde::{Deserialize, Serialize};

/// Empirical history points, not supplied-exact laws or calibrated intervals.
pub const TEMPORAL_RECALC_FEATURE: &str = "temporal_observational_initial_shift_v1";
/// Published inference scope.
pub const TEMPORAL_RECALC_INFERENCE: &str = "empirical_point_only";

/// A complete history of one repeated unit.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TemporalHistoryWire {
    /// Start of this two-step window.
    pub time_id: u64,
    /// Binary pre-action state.
    pub s0: u32,
    /// Binary first action.
    pub a1: u32,
    /// Binary covariate before the second action.
    pub l2: u32,
    /// Binary second action.
    pub a2: u32,
    /// Binary outcome after the second action.
    pub y: f64,
}
/// Histories remain owned by their repeated unit, including in resampling.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TemporalUnitWire {
    /// Stable repeated-unit id.
    pub unit_id: u64,
    /// Complete histories, in ascending time order.
    pub histories: Vec<TemporalHistoryWire>,
}
/// The named functional; a mean is never relabeled as an effect.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum TemporalFunctionalWire {
    /// Mean response to one supported whole action sequence.
    Response {
        /// The two ordered actions.
        sequence: [u32; 2],
    },
    /// Difference of two supported sequence response means.
    Effect {
        /// Active ordered actions.
        active: [u32; 2],
        /// Control ordered actions.
        control: [u32; 2],
    },
}
impl TemporalFunctionalWire {
    /// Required whole-sequence means, in publication order.
    #[must_use]
    pub fn sequences(&self) -> Vec<[u32; 2]> {
        match *self {
            Self::Response { sequence } => vec![sequence],
            Self::Effect { active, control } => vec![active, control],
        }
    }
}
/// All actual data and premises required by a fresh checked consumer.
/// Coordinates are fixed: s0=0, a1=1, l2=2, a2=3, y=4.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TemporalRecalcRequestWire {
    /// Directed unrolled causal edges.
    pub edges: Vec<(u32, u32)>,
    /// Latent-confounding edges; identification must actually certify these.
    pub bidirected: Vec<(u32, u32)>,
    /// Only the root initial state may differ between populations.
    pub selection_targets: Vec<u32>,
    /// Only two is supported.
    pub horizon: usize,
    /// Coordinate-to-slice alignment, in variable-id order: 0,1,2,2,2.
    pub lag_alignment: [u8; 5],
    /// Declared half-open observation period containing every complete history window.
    pub period: (u64, u64),
    /// Raw complete histories with their stable repeated-unit ownership.
    pub units: Vec<TemporalUnitWire>,
    /// Identity of these actual source histories.
    pub snapshot_id: String,
    /// Caller-supplied target initial-state law: mass of states zero and one.
    pub initial_state: [f64; 2],
    /// Identity of the target initial-state law.
    pub initial_state_id: String,
    /// Explicit mean response or effect of whole sequences.
    pub functional: TemporalFunctionalWire,
    /// Benefit per response/effect unit.
    pub benefit_per_unit: f64,
    /// Decision cost.
    pub cost: f64,
}
/// Actual source DAG identification and the explicit initial-shift theorem projection.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemporalInitialShiftReportWire {
    /// Frozen DAG/root-selection sufficient-rule coordinate.
    pub rule: String,
    /// Ordered whole-sequence action assignment.
    pub sequence: [u32; 2],
    /// Actual checked SID joint-source proof, bound to measured observational data.
    pub source_proof: TemporalSourceIdProofWire,
    /// Executed joint source distribution over S0 and Y under both actions.
    pub source_joint: MzTransportPointWire,
    /// Positive-supported conditional means for source initial states zero and one.
    pub conditional_means: [f64; 2],
    /// Initial-law standardized target response.
    pub mean: f64,
}
/// Replayed native general-ID source program and its real observational provider binding.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemporalSourceIdProofWire {
    /// Actual source DAG on which this native observational ID program was checked.
    pub source_graph: crate::AdmgWire,
    /// Native source identification method, always `general_id`.
    pub method: String,
    /// Actual source population, with no auxiliary experimental population.
    pub population: String,
    /// Original source-ID expression, before provider names are bound.
    pub source_expression: ExprArenaWire,
    /// Original joint distribution root.
    pub source_root: u32,
    /// The same algebra bound to source regime0 observational providers.
    pub executable_expression: ExprArenaWire,
    /// Executable joint distribution root.
    pub executable_root: u32,
    /// Actual native general-ID rule trace, independently regenerated on consumption.
    pub derivation: Vec<(String, String)>,
}
/// A readable, source-bound artifact. Acceptance requires a fresh checked execution.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemporalRecalcArtifactWire {
    /// Format version.
    pub version: u32,
    /// Required empirical-point semantics.
    pub required_features: Vec<String>,
    /// Complete raw history ownership and declared causal/temporal premises.
    pub request: TemporalRecalcRequestWire,
    /// Producing master seed; dependent resampling uses whole-unit identities.
    pub seed: u64,
    /// Full checked whole-sequence proof, support and response for each requested mean.
    pub reports: Vec<TemporalInitialShiftReportWire>,
    /// Named response or effect point, as declared in the request.
    pub value: f64,
    /// Always empirical point only; no portable interval claim is manufactured.
    pub inference: String,
    /// Canonical graph/window/history-functional premise identity.
    pub premises_digest: String,
    /// Canonical complete source data and target-law identity.
    pub data_digest: String,
}
impl TemporalRecalcArtifactWire {
    /// Seal a checked producer's full reports and raw source inputs.
    /// # Errors
    /// Encoding failure or malformed bounded shape.
    pub fn seal(
        request: TemporalRecalcRequestWire,
        seed: u64,
        reports: Vec<TemporalInitialShiftReportWire>,
        value: f64,
    ) -> Result<Self, IoError> {
        let mut wire = Self {
            version: 1,
            required_features: vec![TEMPORAL_RECALC_FEATURE.into()],
            request,
            seed,
            reports,
            value,
            inference: TEMPORAL_RECALC_INFERENCE.into(),
            premises_digest: String::new(),
            data_digest: String::new(),
        };
        wire.premises_digest = wire.expected_premises_digest()?;
        wire.data_digest = wire.expected_data_digest()?;
        wire.validate()?;
        Ok(wire)
    }
    /// Rehash the declared structural and temporal premises.
    /// # Errors
    /// Encoding failure.
    pub fn expected_premises_digest(&self) -> Result<String, IoError> {
        let r = &self.request;
        let mut edges = r.edges.clone();
        edges.sort_unstable();
        edges.dedup();
        let mut bidirected = r.bidirected.clone();
        bidirected.sort_unstable();
        bidirected.dedup();
        Ok(crate::identity::digest_wire(
            IdentityDomain::TransportCertificate,
            &(
                TEMPORAL_RECALC_FEATURE,
                self.seed,
                edges,
                bidirected,
                &r.selection_targets,
                r.horizon,
                r.lag_alignment,
                r.period,
                &r.functional,
                r.benefit_per_unit.to_bits(),
                r.cost.to_bits(),
            ),
        )?
        .to_hex())
    }
    /// Rehash every history's actual unit/time ownership and the target law.
    /// # Errors
    /// Encoding failure.
    pub fn expected_data_digest(&self) -> Result<String, IoError> {
        let r = &self.request;
        Ok(crate::identity::digest_wire(
            IdentityDomain::DataSnapshot,
            &(&r.snapshot_id, &r.units, &r.initial_state_id, r.initial_state.map(f64::to_bits)),
        )?
        .to_hex())
    }
    fn validate(&self) -> Result<(), IoError> {
        let invalid_temporal_artifact = |s: &str| IoError::Refused {
            code: antecedent_core::reason_code!("invalid_argument"),
            message: format!("temporal_recalc.artifact_invalid: {s}"),
        };
        if self.version != 1 {
            return Err(IoError::UnsupportedVersion { version: self.version });
        }
        if self.required_features != [TEMPORAL_RECALC_FEATURE]
            || self.inference != TEMPORAL_RECALC_INFERENCE
            || !self.value.is_finite()
            || self.reports.len() != self.request.functional.sequences().len()
            || self.request.units.len() > 8192
            || self.request.units.iter().map(|u| u.histories.len()).sum::<usize>() > 131_072
            || self.request.edges.len() > 25
            || self.request.bidirected.len() > 25
        {
            return Err(invalid_temporal_artifact(
                "unsupported or unbounded empirical point shape",
            ));
        }
        if self.premises_digest != self.expected_premises_digest()?
            || self.data_digest != self.expected_data_digest()?
        {
            return Err(invalid_temporal_artifact("premise/data identity mismatch"));
        }
        Ok(())
    }
    /// Encode; this does not by itself grant scientific acceptance.
    /// # Errors
    /// Encoding failure.
    pub fn export(&self) -> Result<Vec<u8>, IoError> {
        self.validate()?;
        crate::to_cbor(self)
    }
    /// Decode and verify bounded shape and identities, before a real source-backed replay.
    /// # Errors
    /// Malformed/unsupported bytes or stale data/premise identity.
    pub fn decode(bytes: &[u8]) -> Result<Self, IoError> {
        if bytes.len() > 32 * 1024 * 1024 {
            return Err(IoError::Convert("temporal_recalc.artifact_limit".into()));
        }
        let wire: Self = crate::from_cbor(bytes)?;
        wire.validate()?;
        Ok(wire)
    }
    /// Full report equality through canonical wire encoding, including NaN semantics.
    /// # Errors
    /// Encoding failure.
    pub fn matches_reports(
        &self,
        reports: &[TemporalInitialShiftReportWire],
        value: f64,
    ) -> Result<bool, IoError> {
        Ok(crate::to_cbor(&self.reports)? == crate::to_cbor(&reports)?
            && self.value.to_bits() == value.to_bits())
    }
}
