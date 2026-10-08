//! Portable selective-recalculation receipt (`recalc_receipt_v1`, 2.3 C2).
//!
//! A bounded, checksummed container holding one receipt of a selective recalculation:
//!
//! * the previous and the requested per-stage own-input digests (the declarations the plan
//!   compared) and the capability and boundary declarations the plan was made under;
//! * per stage, the status (`reused`, `recomputed` or `refused`, with the dependency that
//!   determined it), the stage's effective identity in the requested workflow and the work
//!   counted for it (identifications, cross-fitted fold fits, score builds, reweights,
//!   decisions);
//! * the canonical plan identity, the canonical receipt identity and the work totals.
//!
//! A consumer trusts none of the stored table. It recomputes the plan from the stored previous
//! and requested digests and capabilities with the core planner, then requires the stored
//! per-stage status, determining dependency and identity, the per-stage counts (a reused or
//! refused stage did no work; a recomputed computation did its work), the totals and both
//! identities to match. A resealed edit of any of them is refused, and a consumer that
//! retained the receipt identity independently refuses a consistently resealed receipt of
//! another run as well.
//!
//! A loaded receipt never claims reuse in a fresh process: a derived stage stored as `reused`
//! under a fresh-process boundary is refused before the plan is compared, because a loaded
//! result recreates no prepared study, score table or fit. The artifact decides and records;
//! it computes nothing and carries no estimate.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;
use std::io::Cursor;
use std::sync::{Mutex, PoisonError};

use antecedent_core::recalc::{
    Boundary, RecalcCapabilities, RecalcPlan, RequestSupport, ResumeContext, RetargetSupport,
    Stage, StageIdentities, StageIdentity, StageStatus,
};
use serde::{Deserialize, Serialize};

use crate::container::{ArtifactManifest, EncodedArtifact, SectionBytes, section_descriptor};
use crate::convert::{from_cbor, to_cbor};
use crate::reader::ArtifactReader;
use crate::wire::{ArtifactKind, ProvenanceWire, SemanticVersion};

/// The artifact major version this reader writes and accepts.
pub const RECALC_RECEIPT_ARTIFACT_VERSION: u16 = 1;
/// The feature marker of the accepted format.
pub const RECALC_RECEIPT_ARTIFACT_FEATURE: &str = "recalc_receipt_v1";
/// Most bytes a receipt artifact may occupy, enforced on export and on consumption.
pub const MAX_RECALC_RECEIPT_ARTIFACT_BYTES: usize = 256 * 1024;
/// Most stage declarations or entries a receipt may carry.
pub const MAX_RECALC_RECEIPT_STAGES: usize = 64;

const ARTIFACT_KIND: &str = "recalc_receipt_v1";
const META_SECTION: &str = "recalc_receipt_meta";
const MAX_ROUTE_BYTES: usize = 96;
const MAX_INTERNED_ROUTES: usize = 64;

/// Why a receipt artifact was refused or could not be produced.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum RecalcReceiptArtifactError {
    /// The bytes do not decode as this format.
    #[error("recalc receipt artifact does not decode: {0}")]
    Decode(String),
    /// Another major version, refused before the payload is interpreted.
    #[error("unsupported recalc receipt artifact version {version}")]
    UnsupportedVersion {
        /// The stored version.
        version: u16,
    },
    /// A feature marker that is not this format's.
    #[error("unsupported recalc receipt semantics: {0}")]
    UnsupportedSemantics(&'static str),
    /// A collection exceeds the format's bound.
    #[error("recalc receipt consumer limit exceeded: {0}")]
    LimitsExceeded(&'static str),
    /// A stored declaration is not well formed (unknown stage, bad digest, duplicate).
    #[error("malformed recalc receipt artifact: {0}")]
    Malformed(String),
    /// A derived stage is stored as reused under a fresh-process boundary.
    #[error(
        "recalc_receipt.fresh_process_reuse: stage {stage} cannot be reused by a loaded receipt"
    )]
    FreshProcessReuse {
        /// The stage label.
        stage: String,
    },
    /// The stored table differs from the plan recomputed from the stored declarations.
    #[error("recalc_receipt.plan_mismatch: stage {stage} {field} differs from the recomputed plan")]
    PlanMismatch {
        /// The stage label (`*` when the stage sets differ).
        stage: String,
        /// Which field differs.
        field: &'static str,
    },
    /// A stage's counts contradict its status.
    #[error(
        "recalc_receipt.counts_inconsistent: stage {stage} counted work its status contradicts"
    )]
    CountsInconsistent {
        /// The stage label.
        stage: String,
    },
    /// The stored totals differ from the sum of the per-stage counts.
    #[error("recalc_receipt.totals_mismatch: stored totals differ from the per-stage counts")]
    TotalsMismatch,
    /// A stored or retained identity differs from the recomputed one.
    #[error("recalc_receipt.identity_mismatch: {field} changed")]
    IdentityMismatch {
        /// Which identity differs.
        field: &'static str,
    },
    /// The artifact could not be encoded.
    #[error("recalc receipt artifact does not encode: {0}")]
    Encode(String),
}

impl RecalcReceiptArtifactError {
    /// The registered refusal this error carries: `(code, detail, explanation)`.
    ///
    /// Present for a changed table, counts or identity and for a fresh-process reuse claim;
    /// absent for corruption, unsupported versions, malformed declarations and encoding
    /// failures.
    #[must_use]
    pub fn refusal(&self) -> Option<(&'static str, &'static str, String)> {
        let text = self.to_string();
        let explanation =
            text.split_once(": ").map_or_else(|| text.clone(), |(_, rest)| rest.to_owned());
        match self {
            Self::FreshProcessReuse { .. } => Some((
                antecedent_core::reason_code!("score_table_unavailable"),
                "recalc_receipt.fresh_process_reuse",
                explanation,
            )),
            Self::PlanMismatch { .. } => Some((
                antecedent_core::reason_code!("route_not_supported"),
                "recalc_receipt.plan_mismatch",
                explanation,
            )),
            Self::CountsInconsistent { .. } => Some((
                antecedent_core::reason_code!("route_not_supported"),
                "recalc_receipt.counts_inconsistent",
                explanation,
            )),
            Self::TotalsMismatch => Some((
                antecedent_core::reason_code!("route_not_supported"),
                "recalc_receipt.totals_mismatch",
                explanation,
            )),
            Self::IdentityMismatch { .. } => Some((
                antecedent_core::reason_code!("route_not_supported"),
                "recalc_receipt.identity_mismatch",
                explanation,
            )),
            _ => None,
        }
    }

    /// The stage a refusal names, when it names one.
    #[must_use]
    pub fn stage(&self) -> Option<&str> {
        match self {
            Self::FreshProcessReuse { stage }
            | Self::PlanMismatch { stage, .. }
            | Self::CountsInconsistent { stage } => Some(stage),
            _ => None,
        }
    }
}

impl From<crate::IoError> for RecalcReceiptArtifactError {
    fn from(error: crate::IoError) -> Self {
        Self::Decode(error.to_string())
    }
}

// -- wire types ---------------------------------------------------------------------------

/// Work counted for one stage (also the totals).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CountsWire {
    /// Identifications performed.
    pub identifications: u64,
    /// Cross-fitted nuisance fold fits performed.
    pub fold_fits: u64,
    /// Score tables built.
    pub score_computations: u64,
    /// Reweights of frozen scores performed.
    pub reweights: u64,
    /// Decision evaluations performed.
    pub decisions: u64,
}

impl CountsWire {
    /// The five counts in canonical order (identification, fold fit, score build, reweight,
    /// decision).
    #[must_use]
    pub const fn as_array(&self) -> [u64; 5] {
        [
            self.identifications,
            self.fold_fits,
            self.score_computations,
            self.reweights,
            self.decisions,
        ]
    }

    /// Sum over every kind of work.
    #[must_use]
    pub fn total(&self) -> u64 {
        self.as_array().iter().fold(0_u64, |sum, n| sum.saturating_add(*n))
    }

    fn merged(self, other: Self) -> Self {
        Self {
            identifications: self.identifications.saturating_add(other.identifications),
            fold_fits: self.fold_fits.saturating_add(other.fold_fits),
            score_computations: self.score_computations.saturating_add(other.score_computations),
            reweights: self.reweights.saturating_add(other.reweights),
            decisions: self.decisions.saturating_add(other.decisions),
        }
    }
}

/// One declared stage: its label and the hex digest of its own inputs.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeclaredStageWire {
    /// Stage label, for example `external_study.1`.
    pub stage: String,
    /// 64-character lowercase hex own-input digest.
    pub own: String,
}

/// What a fresh process was handed to resume with.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
// Independent availability flags mirroring `ResumeContext`.
#[allow(clippy::struct_excessive_bools)]
pub struct ResumeWire {
    /// A portable fitted predictor is available.
    #[serde(default)]
    pub portable_fit: bool,
    /// Portable frozen scores are available.
    #[serde(default)]
    pub portable_scores: bool,
    /// The portable scores bind the unchanged snapshot identity without raw data.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub scores_snapshot_bound: bool,
    /// A compatible data snapshot was supplied.
    #[serde(default)]
    pub supplied_data: bool,
    /// A compatible provider (callback) was supplied.
    #[serde(default)]
    pub supplied_provider: bool,
}

/// The capability and boundary declarations a plan is made under.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapabilitiesWire {
    /// `licensed`, `not_declared` or `incompatible`.
    pub retarget: String,
    /// `on_grid`, `off_grid` or `unsupported`.
    pub request: String,
    /// A separately licensed route that serves an off-grid or unsupported request.
    #[serde(default)]
    pub licensed_route: Option<String>,
    /// `in_process` or `fresh_process`.
    pub boundary: String,
    /// What the fresh process holds; present exactly under `fresh_process`.
    #[serde(default)]
    pub resume: Option<ResumeWire>,
}

/// One row of a plan on the wire.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanEntryWire {
    /// Stage label.
    pub stage: String,
    /// `reused`, `recomputed` or `refused`.
    pub tag: String,
    /// The determining dependency or reason inside the status parentheses.
    pub detail: String,
    /// The full canonical status, for example `recomputed(own:utility:modified)`.
    pub status: String,
    /// Effective identity of the stage in the requested workflow (hex).
    pub identity: String,
}

/// A plan on the wire.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanWire {
    /// Rows in topological order.
    pub entries: Vec<PlanEntryWire>,
    /// Canonical digest of the whole table (hex).
    pub identity: String,
    /// Whether the plan holds no refused stage.
    pub executable: bool,
}

/// One stage of a stored receipt.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReceiptEntryWire {
    /// Stage label.
    pub stage: String,
    /// `reused`, `recomputed` or `refused`.
    pub tag: String,
    /// The determining dependency or reason inside the status parentheses.
    pub detail: String,
    /// The full canonical status.
    pub status: String,
    /// Effective identity of the stage in the requested workflow (hex).
    pub identity: String,
    /// Work counted for the stage.
    pub counts: CountsWire,
}

/// Stored metadata of a receipt artifact.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecalcReceiptMeta {
    /// Metadata format version.
    pub version: u16,
    /// Feature marker.
    pub feature: String,
    /// Capability and boundary declarations.
    pub capabilities: CapabilitiesWire,
    /// Own-input digests of the previous workflow.
    pub previous: Vec<DeclaredStageWire>,
    /// Own-input digests of the requested workflow.
    pub requested: Vec<DeclaredStageWire>,
    /// Canonical plan identity (hex).
    pub plan_identity: String,
    /// Canonical receipt identity (hex).
    pub receipt_identity: String,
    /// Per-stage status, identity and counts, in topological order.
    pub entries: Vec<ReceiptEntryWire>,
    /// Sum of the per-stage counts.
    pub totals: CountsWire,
}

// -- conversions --------------------------------------------------------------------------

fn malformed(text: impl Into<String>) -> RecalcReceiptArtifactError {
    RecalcReceiptArtifactError::Malformed(text.into())
}

/// The stage named by `label`.
#[must_use]
pub fn stage_from_label(label: &str) -> Option<Stage> {
    Stage::all().into_iter().find(|stage| stage.label() == label)
}

static ROUTES: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());

/// A bounded process-wide table of licensed-route names, because the core carries them as
/// `&'static str`. At most [`MAX_INTERNED_ROUTES`] distinct names of [`MAX_ROUTE_BYTES`] bytes.
fn intern_route(route: &str) -> Result<&'static str, RecalcReceiptArtifactError> {
    let allowed = |b: u8| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-');
    if route.is_empty() || route.len() > MAX_ROUTE_BYTES || !route.bytes().all(allowed) {
        return Err(malformed("invalid licensed route name"));
    }
    let mut known = ROUTES.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(found) = known.iter().copied().find(|k| *k == route) {
        return Ok(found);
    }
    if known.len() >= MAX_INTERNED_ROUTES {
        return Err(RecalcReceiptArtifactError::LimitsExceeded("licensed routes"));
    }
    let leaked: &'static str = Box::leak(route.to_owned().into_boxed_str());
    known.push(leaked);
    Ok(leaked)
}

impl CapabilitiesWire {
    /// The wire form of `capabilities`.
    #[must_use]
    pub fn from_capabilities(capabilities: &RecalcCapabilities) -> Self {
        let retarget = match capabilities.retarget {
            RetargetSupport::Licensed => "licensed",
            RetargetSupport::NotDeclared => "not_declared",
            RetargetSupport::Incompatible => "incompatible",
        };
        let (request, licensed_route) = match capabilities.request {
            RequestSupport::OnGrid => ("on_grid", None),
            RequestSupport::OffGrid { licensed_route } => ("off_grid", licensed_route),
            RequestSupport::Unsupported { licensed_route } => ("unsupported", licensed_route),
        };
        let (boundary, resume) = match capabilities.boundary {
            Boundary::InProcess => ("in_process", None),
            Boundary::FreshProcess(resume) => (
                "fresh_process",
                Some(ResumeWire {
                    portable_fit: resume.portable_fit,
                    portable_scores: resume.portable_scores,
                    scores_snapshot_bound: resume.scores_snapshot_bound,
                    supplied_data: resume.supplied_data,
                    supplied_provider: resume.supplied_provider,
                }),
            ),
        };
        Self {
            retarget: retarget.to_owned(),
            request: request.to_owned(),
            licensed_route: licensed_route.map(str::to_owned),
            boundary: boundary.to_owned(),
            resume,
        }
    }

    /// The capabilities this wire form declares.
    ///
    /// # Errors
    /// An unknown name, a route on an on-grid request, a resume context without (or on) a
    /// fresh-process boundary, or an invalid or excess route name.
    pub fn to_capabilities(&self) -> Result<RecalcCapabilities, RecalcReceiptArtifactError> {
        let retarget = match self.retarget.as_str() {
            "licensed" => RetargetSupport::Licensed,
            "not_declared" => RetargetSupport::NotDeclared,
            "incompatible" => RetargetSupport::Incompatible,
            other => return Err(malformed(format!("unknown retarget support `{other}`"))),
        };
        let route = self.licensed_route.as_deref().map(intern_route).transpose()?;
        let request = match (self.request.as_str(), route) {
            ("on_grid", None) => RequestSupport::OnGrid,
            ("on_grid", Some(_)) => {
                return Err(malformed("an on-grid request names no licensed route"));
            }
            ("off_grid", licensed_route) => RequestSupport::OffGrid { licensed_route },
            ("unsupported", licensed_route) => RequestSupport::Unsupported { licensed_route },
            (other, _) => return Err(malformed(format!("unknown request support `{other}`"))),
        };
        let boundary = match (self.boundary.as_str(), self.resume) {
            ("in_process", None) => Boundary::InProcess,
            ("fresh_process", Some(resume)) => Boundary::FreshProcess(ResumeContext {
                portable_fit: resume.portable_fit,
                portable_scores: resume.portable_scores,
                scores_snapshot_bound: resume.scores_snapshot_bound,
                supplied_data: resume.supplied_data,
                supplied_provider: resume.supplied_provider,
            }),
            ("in_process", Some(_)) => {
                return Err(malformed("an in-process boundary names no resume context"));
            }
            ("fresh_process", None) => {
                return Err(malformed("a fresh-process boundary needs a resume context"));
            }
            (other, _) => return Err(malformed(format!("unknown boundary `{other}`"))),
        };
        Ok(RecalcCapabilities { retarget, request, boundary })
    }
}

/// The wire form of every declared stage, in topological order.
#[must_use]
pub fn identities_to_wire(identities: &StageIdentities) -> Vec<DeclaredStageWire> {
    Stage::all()
        .into_iter()
        .filter(|stage| identities.contains(*stage))
        .map(|stage| DeclaredStageWire {
            stage: stage.label(),
            own: identities.own(stage).to_hex(),
        })
        .collect()
}

/// The workflow a wire declaration list names, in any order.
///
/// # Errors
/// More than [`MAX_RECALC_RECEIPT_STAGES`] rows, an unknown stage, a duplicate stage or a
/// digest that is not 64 lowercase hex characters.
pub fn identities_from_wire(
    wire: &[DeclaredStageWire],
) -> Result<StageIdentities, RecalcReceiptArtifactError> {
    if wire.len() > MAX_RECALC_RECEIPT_STAGES {
        return Err(RecalcReceiptArtifactError::LimitsExceeded("stage declarations"));
    }
    let mut out = StageIdentities::new();
    for row in wire {
        let stage = stage_from_label(&row.stage)
            .ok_or_else(|| malformed(format!("unknown stage `{}`", row.stage)))?;
        if out.contains(stage) {
            return Err(malformed(format!("stage `{}` declared twice", row.stage)));
        }
        let own = StageIdentity::from_hex(&row.own).ok_or_else(|| {
            malformed(format!("stage `{}` digest is not 64 hex digits", row.stage))
        })?;
        out.set(stage, own);
    }
    Ok(out)
}

/// `(tag, detail)` of a status: `recomputed(own:utility:modified)` is
/// `("recomputed", "own:utility:modified")`.
fn status_parts(status: &StageStatus) -> (String, String) {
    let tag = status.tag();
    let text = status.to_string();
    let detail = text
        .strip_prefix(tag)
        .and_then(|rest| rest.strip_prefix('('))
        .and_then(|rest| rest.strip_suffix(')'))
        .unwrap_or_default()
        .to_owned();
    (tag.to_owned(), detail)
}

/// The wire form of a plan.
#[must_use]
pub fn plan_to_wire(plan: &RecalcPlan) -> PlanWire {
    PlanWire {
        entries: plan
            .entries()
            .iter()
            .map(|entry| {
                let (tag, detail) = status_parts(&entry.status);
                PlanEntryWire {
                    stage: entry.stage.label(),
                    tag,
                    detail,
                    status: entry.status.to_string(),
                    identity: entry.identity.to_hex(),
                }
            })
            .collect(),
        identity: plan.canonical_identity().to_hex(),
        executable: plan.is_executable(),
    }
}

/// Plan `requested` against `previous` under `capabilities`, from wire declarations.
///
/// # Errors
/// Malformed declarations or capabilities.
pub fn plan_from_wire(
    previous: &[DeclaredStageWire],
    requested: &[DeclaredStageWire],
    capabilities: &CapabilitiesWire,
) -> Result<RecalcPlan, RecalcReceiptArtifactError> {
    Ok(RecalcPlan::plan(
        &identities_from_wire(previous)?,
        &identities_from_wire(requested)?,
        &capabilities.to_capabilities()?,
    ))
}

// -- consistency --------------------------------------------------------------------------

/// Whether `counts` are what `stage` may have counted under its status: nothing unless it was
/// recomputed, only work it owns, and every kind of work a recomputed computation requires.
fn counts_consistent(stage: Stage, recomputed: bool, counts: &CountsWire) -> bool {
    let owned: [(u64, Stage); 5] = [
        (counts.identifications, Stage::Identification),
        (counts.fold_fits, Stage::ScoreArtifact),
        (counts.score_computations, Stage::ScoreArtifact),
        (counts.reweights, Stage::Law),
        (counts.decisions, Stage::Decision),
    ];
    if owned.iter().any(|(n, owner)| *n > 0 && !(recomputed && *owner == stage)) {
        return false;
    }
    if !recomputed {
        return true;
    }
    let required: &[u64] = match stage {
        Stage::Identification => &[counts.identifications],
        Stage::ScoreArtifact => &[counts.fold_fits, counts.score_computations],
        Stage::Law => &[counts.reweights],
        Stage::Decision => &[counts.decisions],
        _ => &[],
    };
    required.iter().all(|n| *n > 0)
}

fn compute_receipt_identity(
    entries: &[ReceiptEntryWire],
) -> Result<StageIdentity, RecalcReceiptArtifactError> {
    let mut parts: Vec<Vec<u8>> = Vec::with_capacity(entries.len() * 4);
    for entry in entries {
        let identity = StageIdentity::from_hex(&entry.identity)
            .ok_or_else(|| malformed("stage identity is not 64 hex digits"))?;
        let mut counts = Vec::with_capacity(40);
        for n in entry.counts.as_array() {
            counts.extend_from_slice(&n.to_le_bytes());
        }
        parts.push(entry.stage.clone().into_bytes());
        parts.push(entry.status.clone().into_bytes());
        parts.push(identity.as_bytes().to_vec());
        parts.push(counts);
    }
    let refs: Vec<&[u8]> = parts.iter().map(Vec::as_slice).collect();
    Ok(StageIdentity::of("recalc_receipt", &refs))
}

fn sorted_declared(identities: &StageIdentities) -> Vec<DeclaredStageWire> {
    identities_to_wire(identities)
}

fn refuse_fresh_reuse(
    meta: &RecalcReceiptMeta,
    capabilities: &RecalcCapabilities,
) -> Result<(), RecalcReceiptArtifactError> {
    if !matches!(capabilities.boundary, Boundary::FreshProcess(_)) {
        return Ok(());
    }
    for entry in &meta.entries {
        let stage = stage_from_label(&entry.stage)
            .ok_or_else(|| malformed(format!("unknown stage `{}`", entry.stage)))?;
        if !stage.is_input() && (entry.tag == "reused" || entry.status.starts_with("reused(")) {
            return Err(RecalcReceiptArtifactError::FreshProcessReuse {
                stage: entry.stage.clone(),
            });
        }
    }
    Ok(())
}

fn mismatch(stage: &str, field: &'static str) -> RecalcReceiptArtifactError {
    RecalcReceiptArtifactError::PlanMismatch { stage: stage.to_owned(), field }
}

/// Recompute everything from the stored declarations and require the stored table to match.
fn check(
    mut meta: RecalcReceiptMeta,
    expected_identity: Option<&str>,
) -> Result<RecalcReceiptArtifact, RecalcReceiptArtifactError> {
    if meta.version != RECALC_RECEIPT_ARTIFACT_VERSION {
        return Err(RecalcReceiptArtifactError::UnsupportedVersion { version: meta.version });
    }
    if meta.feature != RECALC_RECEIPT_ARTIFACT_FEATURE {
        return Err(RecalcReceiptArtifactError::UnsupportedSemantics("feature marker"));
    }
    if meta.entries.len() > MAX_RECALC_RECEIPT_STAGES {
        return Err(RecalcReceiptArtifactError::LimitsExceeded("receipt entries"));
    }
    let capabilities = meta.capabilities.to_capabilities()?;
    let previous = identities_from_wire(&meta.previous)?;
    let requested = identities_from_wire(&meta.requested)?;
    // A loaded receipt recreates no prepared study, score table or fit: it never claims reuse
    // of a derived stage in a fresh process, whatever else it says.
    refuse_fresh_reuse(&meta, &capabilities)?;
    let plan = RecalcPlan::plan(&previous, &requested, &capabilities);

    let mut by_stage: BTreeMap<Stage, &ReceiptEntryWire> = BTreeMap::new();
    for entry in &meta.entries {
        let stage = stage_from_label(&entry.stage)
            .ok_or_else(|| malformed(format!("unknown stage `{}`", entry.stage)))?;
        if by_stage.insert(stage, entry).is_some() {
            return Err(malformed(format!("stage `{}` listed twice", entry.stage)));
        }
    }
    if by_stage.len() != plan.entries().len() {
        return Err(mismatch("*", "stage set"));
    }
    let mut ordered = Vec::with_capacity(by_stage.len());
    let mut totals = CountsWire::default();
    for planned in plan.entries() {
        let label = planned.stage.label();
        let stored = by_stage.get(&planned.stage).ok_or_else(|| mismatch(&label, "stage set"))?;
        let (tag, detail) = status_parts(&planned.status);
        if stored.status != planned.status.to_string()
            || stored.tag != tag
            || stored.detail != detail
        {
            return Err(mismatch(&label, "status"));
        }
        if stored.identity != planned.identity.to_hex() {
            return Err(mismatch(&label, "identity"));
        }
        if !counts_consistent(planned.stage, tag == "recomputed", &stored.counts) {
            return Err(RecalcReceiptArtifactError::CountsInconsistent { stage: label });
        }
        totals = totals.merged(stored.counts);
        ordered.push((**stored).clone());
    }
    if meta.totals != totals {
        return Err(RecalcReceiptArtifactError::TotalsMismatch);
    }
    if meta.plan_identity != plan.canonical_identity().to_hex() {
        return Err(RecalcReceiptArtifactError::IdentityMismatch { field: "plan_identity" });
    }
    let receipt_identity = compute_receipt_identity(&ordered)?.to_hex();
    if meta.receipt_identity != receipt_identity {
        return Err(RecalcReceiptArtifactError::IdentityMismatch { field: "receipt_identity" });
    }
    if expected_identity.is_some_and(|expected| expected != receipt_identity) {
        return Err(RecalcReceiptArtifactError::IdentityMismatch {
            field: "retained_receipt_identity",
        });
    }
    // Canonical form: topological entries, declarations in topological order.
    meta.entries = ordered;
    meta.previous = sorted_declared(&previous);
    meta.requested = sorted_declared(&requested);
    Ok(RecalcReceiptArtifact { meta, plan })
}

// -- container ----------------------------------------------------------------------------

#[derive(Deserialize)]
struct VersionPeek {
    version: u16,
}

/// Encode a metadata section as a checksummed container.
///
/// Hidden: the producer path is [`RecalcReceiptArtifact::to_bytes`]; tests use this to build
/// deliberately inconsistent or resealed artifacts.
///
/// # Errors
/// An empty id, an encoding failure or an oversized payload.
#[doc(hidden)]
pub fn encode_parts(
    meta: &RecalcReceiptMeta,
    artifact_id: &str,
) -> Result<Vec<u8>, RecalcReceiptArtifactError> {
    let encode = |e: crate::IoError| RecalcReceiptArtifactError::Encode(e.to_string());
    if artifact_id.trim().is_empty() {
        return Err(RecalcReceiptArtifactError::Encode("missing artifact id".into()));
    }
    let meta_bytes = to_cbor(meta).map_err(encode)?;
    let encoded = EncodedArtifact {
        manifest: ArtifactManifest {
            format_version: crate::migrate::STABLE_FORMAT,
            minimum_reader_version: crate::migrate::STABLE_FORMAT,
            artifact_kind: ArtifactKind::Other(ARTIFACT_KIND.into()),
            library_version: SemanticVersion::from_crate_version(env!("CARGO_PKG_VERSION"))
                .map_err(encode)?,
            artifact_id: artifact_id.into(),
            sections: vec![section_descriptor(META_SECTION, "application/cbor", &meta_bytes)],
            provenance: ProvenanceWire { note: "recalc_receipt".into() },
        },
        sections: vec![SectionBytes::new(META_SECTION, meta_bytes)],
    };
    let mut bytes = Vec::new();
    encoded.write_to(&mut bytes).map_err(encode)?;
    if bytes.len() > MAX_RECALC_RECEIPT_ARTIFACT_BYTES {
        return Err(RecalcReceiptArtifactError::LimitsExceeded("artifact bytes"));
    }
    Ok(bytes)
}

/// Decode the metadata section of a container, refusing another version before the metadata
/// is interpreted.
///
/// Hidden: see [`encode_parts`].
///
/// # Errors
/// Oversized, truncated, corrupt, differently laid out or other-version artifacts.
#[doc(hidden)]
pub fn decode_parts(bytes: &[u8]) -> Result<RecalcReceiptMeta, RecalcReceiptArtifactError> {
    if bytes.len() > MAX_RECALC_RECEIPT_ARTIFACT_BYTES {
        return Err(RecalcReceiptArtifactError::LimitsExceeded("artifact bytes"));
    }
    let mut reader = ArtifactReader::open_seek(Cursor::new(bytes))?;
    let manifest = reader.manifest();
    if manifest.artifact_kind != ArtifactKind::Other(ARTIFACT_KIND.into())
        || manifest.sections.len() != 1
        || manifest.sections[0].id != META_SECTION
    {
        return Err(malformed("unsupported container layout"));
    }
    if manifest.sections[0].uncompressed_size > MAX_RECALC_RECEIPT_ARTIFACT_BYTES as u64 {
        return Err(RecalcReceiptArtifactError::LimitsExceeded("artifact bytes"));
    }
    let section = reader.load_section(META_SECTION)?;
    let peek: VersionPeek = from_cbor(section.as_bytes())?;
    if peek.version != RECALC_RECEIPT_ARTIFACT_VERSION {
        return Err(RecalcReceiptArtifactError::UnsupportedVersion { version: peek.version });
    }
    Ok(from_cbor(section.as_bytes())?)
}

/// A produced or consumed selective-recalculation receipt: the stored table and the plan
/// recomputed from its declarations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecalcReceiptArtifact {
    meta: RecalcReceiptMeta,
    plan: RecalcPlan,
}

impl RecalcReceiptArtifact {
    /// Seal the receipt of a run: plan `requested` against `previous` under `capabilities`
    /// and attach the work counted per stage (a stage missing from `counts` counted none).
    ///
    /// The result passes the same checks a consumer applies, so a table that contradicts its
    /// counts (a reused stage that did work, a recomputed computation that did none) or a
    /// fresh-process reuse claim cannot be sealed.
    ///
    /// # Errors
    /// The consumer's refusals for a contradictory table, and the format's bounds.
    pub fn seal(
        previous: &StageIdentities,
        requested: &StageIdentities,
        capabilities: &RecalcCapabilities,
        counts: &BTreeMap<Stage, CountsWire>,
    ) -> Result<Self, RecalcReceiptArtifactError> {
        let plan = RecalcPlan::plan(previous, requested, capabilities);
        let mut totals = CountsWire::default();
        let entries: Vec<ReceiptEntryWire> = plan
            .entries()
            .iter()
            .map(|entry| {
                let counts = counts.get(&entry.stage).copied().unwrap_or_default();
                totals = totals.merged(counts);
                let (tag, detail) = status_parts(&entry.status);
                ReceiptEntryWire {
                    stage: entry.stage.label(),
                    tag,
                    detail,
                    status: entry.status.to_string(),
                    identity: entry.identity.to_hex(),
                    counts,
                }
            })
            .collect();
        let receipt_identity = compute_receipt_identity(&entries)?.to_hex();
        let meta = RecalcReceiptMeta {
            version: RECALC_RECEIPT_ARTIFACT_VERSION,
            feature: RECALC_RECEIPT_ARTIFACT_FEATURE.to_owned(),
            capabilities: CapabilitiesWire::from_capabilities(capabilities),
            previous: identities_to_wire(previous),
            requested: identities_to_wire(requested),
            plan_identity: plan.canonical_identity().to_hex(),
            receipt_identity,
            entries,
            totals,
        };
        check(meta, None)
    }

    /// The stored metadata, in canonical order.
    #[must_use]
    pub const fn meta(&self) -> &RecalcReceiptMeta {
        &self.meta
    }

    /// The plan recomputed from the stored declarations.
    #[must_use]
    pub const fn plan(&self) -> &RecalcPlan {
        &self.plan
    }

    /// Canonical receipt identity (hex), independent of the order entries were supplied in.
    #[must_use]
    pub fn receipt_identity(&self) -> &str {
        &self.meta.receipt_identity
    }

    /// Canonical plan identity (hex).
    #[must_use]
    pub fn plan_identity(&self) -> &str {
        &self.meta.plan_identity
    }

    /// The capabilities the plan was made under.
    ///
    /// # Errors
    /// Never for a sealed or consumed artifact; the declarations were validated.
    pub fn capabilities(&self) -> Result<RecalcCapabilities, RecalcReceiptArtifactError> {
        self.meta.capabilities.to_capabilities()
    }

    /// Serialize through the checksummed sectioned container.
    ///
    /// # Errors
    /// An empty id, an encoding failure or an oversized payload.
    pub fn to_bytes(&self, artifact_id: &str) -> Result<Vec<u8>, RecalcReceiptArtifactError> {
        encode_parts(&self.meta, artifact_id)
    }

    /// Consume an artifact by recomputation.
    ///
    /// `expected_identity` is a receipt identity (hex) the consumer retained independently;
    /// when given, a consistently resealed receipt of another run is refused.
    ///
    /// # Errors
    /// Corruption, another major version, a derived stage reused in a fresh process, a table,
    /// counts, totals or identity that do not match the recomputed plan, or malformed
    /// declarations.
    pub fn from_bytes(
        bytes: &[u8],
        expected_identity: Option<&str>,
    ) -> Result<Self, RecalcReceiptArtifactError> {
        check(decode_parts(bytes)?, expected_identity)
    }

    /// Validate metadata as a consumer would, for hidden test reseals.
    ///
    /// # Errors
    /// The same refusals as [`Self::from_bytes`].
    #[doc(hidden)]
    pub fn from_meta(
        meta: RecalcReceiptMeta,
        expected_identity: Option<&str>,
    ) -> Result<Self, RecalcReceiptArtifactError> {
        check(meta, expected_identity)
    }
}
