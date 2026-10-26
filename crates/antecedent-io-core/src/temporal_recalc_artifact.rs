//! Source-bound empirical history recalculation artifact.
//!
//! The two-step layout is format version 1 and its bytes are the original ones. A
//! recalculation over any other number of steps is format version 2 under its own
//! feature marker; the two layouts share every type, and version 1 stays the only
//! accepted layout of a two-step request.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::IoError;
use crate::expr_wire::ExprArenaWire;
use crate::transport_point_wire::MzTransportPointWire;
use antecedent_core::IdentityDomain;
use serde::{Deserialize, Serialize};

/// Empirical history points, not supplied-exact laws or calibrated intervals.
pub const TEMPORAL_RECALC_FEATURE: &str = "temporal_observational_initial_shift_v1";
/// The feature marker of a recalculation over a number of steps other than two.
pub const TEMPORAL_RECALC_HORIZON_FEATURE: &str = "temporal_observational_initial_shift_v2";
/// The format version of a two-step recalculation.
pub const TEMPORAL_RECALC_VERSION: u32 = 1;
/// The format version of a recalculation over a number of steps other than two.
pub const TEMPORAL_RECALC_HORIZON_VERSION: u32 = 2;
/// The number of steps of the original layout.
const ORIGINAL_HORIZON: usize = 2;
/// Most steps an artifact may declare.
const MAX_HORIZON: usize = 64;
/// Published inference scope.
pub const TEMPORAL_RECALC_INFERENCE: &str = "empirical_point_only";

/// A complete history of one repeated unit.
///
/// A two-step history is written as the fields `s0, a1, l2, a2, y` (the original
/// layout); any other number of steps is written with the `actions` and
/// `covariates` lists.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(try_from = "HistoryLayout", into = "HistoryLayout")]
pub struct TemporalHistoryWire {
    /// Start of this window.
    pub time_id: u64,
    /// Binary pre-action state.
    pub s0: u32,
    /// Binary action of each step, in order.
    pub actions: Vec<u32>,
    /// Binary covariate before the action of each step after the first.
    pub covariates: Vec<u32>,
    /// Binary outcome after the last action.
    pub y: f64,
}

impl TemporalHistoryWire {
    /// A two-step history.
    #[must_use]
    pub fn two_step(time_id: u64, s0: u32, a1: u32, l2: u32, a2: u32, y: f64) -> Self {
        Self { time_id, s0, actions: vec![a1, a2], covariates: vec![l2], y }
    }

    /// Whether the history has one action per step and one covariate per later step.
    #[must_use]
    pub fn is_well_formed(&self, horizon: usize) -> bool {
        self.actions.len() == horizon && self.covariates.len() + 1 == horizon
    }
}

/// The serialized fields of a history: the original two-step fields, or the lists.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct HistoryLayout {
    time_id: u64,
    s0: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    a1: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    l2: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    a2: Option<u32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    actions: Vec<u32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    covariates: Vec<u32>,
    y: f64,
}

impl From<TemporalHistoryWire> for HistoryLayout {
    fn from(h: TemporalHistoryWire) -> Self {
        let original = h.actions.len() == ORIGINAL_HORIZON && h.covariates.len() == 1;
        let (a1, l2, a2) = if original {
            (Some(h.actions[0]), Some(h.covariates[0]), Some(h.actions[1]))
        } else {
            (None, None, None)
        };
        let (actions, covariates) =
            if original { (Vec::new(), Vec::new()) } else { (h.actions, h.covariates) };
        Self { time_id: h.time_id, s0: h.s0, a1, l2, a2, actions, covariates, y: h.y }
    }
}

impl TryFrom<HistoryLayout> for TemporalHistoryWire {
    type Error = String;

    fn try_from(l: HistoryLayout) -> Result<Self, String> {
        match (l.a1, l.l2, l.a2) {
            (Some(a1), Some(l2), Some(a2)) if l.actions.is_empty() && l.covariates.is_empty() => {
                Ok(Self::two_step(l.time_id, l.s0, a1, l2, a2, l.y))
            }
            (None, None, None) if !l.actions.is_empty() => {
                if l.actions.len() == ORIGINAL_HORIZON && l.covariates.len() == 1 {
                    return Err("a two-step history is written as the fields s0, a1, l2, a2".into());
                }
                Ok(Self {
                    time_id: l.time_id,
                    s0: l.s0,
                    actions: l.actions,
                    covariates: l.covariates,
                    y: l.y,
                })
            }
            _ => Err("a history is either the two-step fields or the actions and covariates".into()),
        }
    }
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
        /// The ordered actions, one per step.
        sequence: Vec<u32>,
    },
    /// Difference of two supported sequence response means.
    Effect {
        /// Active ordered actions.
        active: Vec<u32>,
        /// Control ordered actions.
        control: Vec<u32>,
    },
}
impl TemporalFunctionalWire {
    /// Required whole-sequence means, in publication order.
    #[must_use]
    pub fn sequences(&self) -> Vec<Vec<u32>> {
        match self {
            Self::Response { sequence } => vec![sequence.clone()],
            Self::Effect { active, control } => vec![active.clone(), control.clone()],
        }
    }
}
/// All actual data and premises required by a fresh checked consumer.
/// Coordinates are numbered in time order: s0=0, a1=1, then each later step's covariate
/// and action, and the outcome last (with two steps: l2=2, a2=3, y=4).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TemporalRecalcRequestWire {
    /// Directed unrolled causal edges.
    pub edges: Vec<(u32, u32)>,
    /// Latent-confounding edges; identification must actually certify these.
    pub bidirected: Vec<(u32, u32)>,
    /// Only the root initial state may differ between populations.
    pub selection_targets: Vec<u32>,
    /// The number of steps; two is format version 1, any other number version 2.
    pub horizon: usize,
    /// Coordinate-to-slice alignment, in variable-id order: with two steps 0,1,2,2,2.
    pub lag_alignment: Vec<u8>,
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
    pub sequence: Vec<u32>,
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
/// The format version and feature marker a request over `horizon` steps is written under.
fn layout(horizon: usize) -> (u32, &'static str) {
    if horizon == ORIGINAL_HORIZON {
        (TEMPORAL_RECALC_VERSION, TEMPORAL_RECALC_FEATURE)
    } else {
        (TEMPORAL_RECALC_HORIZON_VERSION, TEMPORAL_RECALC_HORIZON_FEATURE)
    }
}

/// Most directed or bidirected edges a request over `horizon` steps may carry.
fn max_edges(horizon: usize) -> usize {
    let variables = 2 * horizon + 1;
    25.max(variables.saturating_mul(variables - 1) / 2)
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
        let (version, feature) = layout(request.horizon);
        let mut wire = Self {
            version,
            required_features: vec![feature.into()],
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
        crate::digest_hex(
            IdentityDomain::TransportCertificate,
            &(
                layout(r.horizon).1,
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
        )
    }
    /// Rehash every history's actual unit/time ownership and the target law.
    /// # Errors
    /// Encoding failure.
    pub fn expected_data_digest(&self) -> Result<String, IoError> {
        let r = &self.request;
        crate::digest_hex(
            IdentityDomain::DataSnapshot,
            &(&r.snapshot_id, &r.units, &r.initial_state_id, r.initial_state.map(f64::to_bits)),
        )
    }
    fn validate(&self) -> Result<(), IoError> {
        let invalid_temporal_artifact = |s: &str| IoError::Refused {
            code: antecedent_core::reason_code!("invalid_argument"),
            message: format!("temporal_recalc.artifact_invalid: {s}"),
        };
        let horizon = self.request.horizon;
        if horizon == 0 || horizon > MAX_HORIZON {
            return Err(invalid_temporal_artifact("unsupported or unbounded empirical point shape"));
        }
        let (version, feature) = layout(horizon);
        if self.version != version {
            return Err(IoError::UnsupportedVersion { version: self.version });
        }
        if self.required_features != [feature]
            || self.inference != TEMPORAL_RECALC_INFERENCE
            || !self.value.is_finite()
            || self.reports.len() != self.request.functional.sequences().len()
            || self.request.units.len() > 8192
            || self.request.units.iter().map(|u| u.histories.len()).sum::<usize>() > 131_072
            || self.request.edges.len() > max_edges(horizon)
            || self.request.bidirected.len() > max_edges(horizon)
            || self.request.lag_alignment.len() != 2 * horizon + 1
            || self
                .request
                .units
                .iter()
                .flat_map(|u| &u.histories)
                .any(|h| !h.is_well_formed(horizon))
            || self
                .request
                .functional
                .sequences()
                .iter()
                .chain(self.reports.iter().map(|r| &r.sequence))
                .any(|sequence| sequence.len() != horizon)
        {
            return Err(invalid_temporal_artifact(
                "unsupported or unbounded empirical point shape",
            ));
        }
        crate::require_digest(
            &self.premises_digest,
            self.expected_premises_digest(),
            std::convert::identity,
            || invalid_temporal_artifact("premise/data identity mismatch"),
        )?;
        crate::require_digest(
            &self.data_digest,
            self.expected_data_digest(),
            std::convert::identity,
            || invalid_temporal_artifact("premise/data identity mismatch"),
        )?;
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
