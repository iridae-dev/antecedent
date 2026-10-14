//! Portable report of a transport revalidation (`transport_revalidation_v1`).
//!
//! A bounded, checksummed container holding the two transport scenes that were
//! compared (the earlier one, when there is one, and the current one), the
//! observed effects that were applied, and the classification the comparison
//! produced: for every claim and decision, `still_valid`, `recompute` or
//! `withdrawn`, the transport reasons in stable names, whether the earlier
//! numbers keep their authority, and the state of its uncertainty guarantee.
//!
//! # What a consumer does
//!
//! It trusts no stored classification. It rebuilds both scenes (every library
//! validation applies, and the stored wire must be in canonical order), runs
//! the same [`revalidate_transport`] the producer ran, and compares the whole
//! stored report and the classification identity with what that yields. A
//! resealed edit that moves a claim from `withdrawn` to `still_valid`, drops a
//! reason, or relabels calibration does not match and is refused. Nothing is
//! recomputed statistically: a scene holds identities, never numbers.
//!
//! # Refresh
//!
//! A report is a record of one comparison. When a scene changes again the
//! producer seals a new report against the scene this one held: *recompute*
//! every claim the classification names; *keep* the scene declarations that did
//! not change; a calibration record held at another target is never carried to
//! this one and is reported as withdrawn. The artifact never edits a scene.
//!
//! # What this is not
//!
//! It computes no statistic and declares no calibration record. A `still_valid`
//! claim is one whose declared inputs did not change; it is not a statement
//! that the claim is true, and the artifact carries no interval or posterior.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_contract::composition_verification::{
    GraphError, MAX_NODES, MAX_TEXT_BYTES, ObservedEffect, SupportStanding, TrustStanding,
};
use antecedent_contract::evidence_ledger::parse_digest_hex;
use antecedent_contract::transport_revalidation::{
    SceneCalibration, SceneClaim, SceneClaimKind, SceneSource, SceneTarget, TransportReason,
    TransportRevalidation, TransportScene, revalidate_transport,
};
use antecedent_core::{IdentityDomain, SemanticDigest};
use serde::{Deserialize, Serialize};

use crate::composition_verification_artifact::EffectWire;
use crate::convert::to_cbor;

/// The artifact major version this reader writes and accepts.
pub const TRANSPORT_REVALIDATION_ARTIFACT_VERSION: u16 = 1;
/// The feature marker of the accepted format.
pub const TRANSPORT_REVALIDATION_ARTIFACT_FEATURE: &str = "transport_revalidation_v1";
/// Most bytes a transport revalidation artifact may occupy.
pub const MAX_TRANSPORT_REVALIDATION_ARTIFACT_BYTES: usize = 16 * 1024 * 1024;
/// Most items one list of a scene or report may hold (sources, claims, edges
/// from a claim, effects).
pub const MAX_TRANSPORT_REVALIDATION_ITEMS: usize = MAX_NODES;

const ARTIFACT_KIND: &str = "transport_revalidation_v1";
const META_SECTION: &str = "transport_revalidation_meta";

crate::sectioned_artifact_enum! {
    /// Why a transport revalidation artifact was refused or could not be produced.
    #[derive(Clone, Debug, PartialEq, Eq)]
    TransportRevalidationArtifactError {
        noun: "transport revalidation",
        identity: "transport_revalidation.identity_mismatch",
        variants {
            /// A stored classification, reason, removal or report identity does
            /// not replay from the stored scenes.
            ReplayMismatch(&'static str),
        }
        display(f) {
            Self::ReplayMismatch(what) => {
                write!(f, "transport_revalidation.replay_mismatch: stored {what} does not replay")
            }
        }
    }
}

impl TransportRevalidationArtifactError {
    /// The registered refusal this error carries: `(code, detail, explanation)`.
    ///
    /// Present for a stored value that does not replay and for a changed
    /// identity; absent for corruption, unsupported versions or features,
    /// malformed values, limits and encoding failures.
    #[must_use]
    pub fn refusal(&self) -> Option<(&'static str, String, String)> {
        match self {
            Self::IdentityMismatch { field } => Some((
                antecedent_core::reason_code!("route_not_supported"),
                "transport_revalidation.identity_mismatch".to_owned(),
                format!("{field} changed"),
            )),
            Self::ReplayMismatch(what) => Some((
                antecedent_core::reason_code!("invalid_argument"),
                "transport_revalidation.replay_mismatch".to_owned(),
                format!("the stored {what} does not replay from the stored scenes"),
            )),
            _ => None,
        }
    }
}

type Failure = TransportRevalidationArtifactError;

fn malformed(text: impl Into<String>) -> Failure {
    Failure::Malformed(text.into())
}

fn graph_error(error: &GraphError) -> Failure {
    match error {
        GraphError::LimitExceeded(what) => Failure::LimitsExceeded(what),
        other => malformed(other.to_string()),
    }
}

// -- wire ---------------------------------------------------------------------------------------

/// One transport source on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SceneSourceWire {
    /// Source key.
    pub key: String,
    /// Content identity (hex).
    pub identity: String,
}

/// A claim's calibration declaration on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SceneCalibrationWire {
    /// `point_only`, `required`, `held` or `inherited`.
    pub kind: String,
    /// The claim a record is inherited from (`inherited`).
    pub from: Option<String>,
    /// Record identity (hex), for `held` and `inherited`.
    pub record: Option<String>,
    /// Identity (hex) of the target the record was established at.
    pub established_at_target: Option<String>,
}

/// One claim or decision of a scene on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SceneClaimWire {
    /// Claim key.
    pub key: String,
    /// `transported` or `decision`.
    pub kind: String,
    /// Content identity (hex).
    pub identity: String,
    /// Structural identity (hex), when it has one.
    pub structure: Option<String>,
    /// Quantity coordinate identity (hex).
    pub quantity: String,
    /// Source keys, sorted and distinct.
    pub sources: Vec<String>,
    /// Claim keys it rests on, sorted and distinct.
    pub rests_on: Vec<String>,
    /// Unsupported coordinates; `None` when supported.
    pub unsupported: Option<Vec<String>>,
    /// `TrustStanding` name.
    pub trust: String,
    /// Calibration declaration.
    pub calibration: SceneCalibrationWire,
}

/// A transport scene on the wire.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SceneWire {
    /// Target population identity (hex).
    pub target_identity: String,
    /// Unsupported target coordinates; `None` when supported.
    pub target_unsupported: Option<Vec<String>>,
    /// Selection diagram identity (hex).
    pub selection_diagram: String,
    /// Sources in key order.
    pub sources: Vec<SceneSourceWire>,
    /// Claims in key order.
    pub claims: Vec<SceneClaimWire>,
}

/// One claim's stored classification.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClaimReportWire {
    /// Claim key.
    pub key: String,
    /// `transported` or `decision`.
    pub kind: String,
    /// `still_valid`, `recompute` or `withdrawn`.
    pub class: String,
    /// Transport reasons, `name` or `name:detail`, sorted and distinct.
    pub reasons: Vec<String>,
    /// Whether the earlier numbers and uncertainty keep their authority.
    pub authority_retained: bool,
    /// `CalibrationState` name.
    pub calibration_state: String,
    /// Registered refusal code of a claim that is not usable as it stands.
    pub refusal_code: Option<String>,
    /// Refusal detail of a claim that is not usable as it stands.
    pub refusal_detail: Option<String>,
}

/// Stored metadata of a transport revalidation artifact.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransportRevalidationMeta {
    /// Metadata format version.
    pub version: u16,
    /// Feature marker.
    pub feature: String,
    /// Features a reader must understand. This version defines none; a reader
    /// refuses any entry.
    pub required_features: Vec<String>,
    /// The earlier scene; `None` for a first verification, which compares the
    /// current scene with itself.
    pub previous: Option<SceneWire>,
    /// The current scene.
    pub current: SceneWire,
    /// Observed effects, in key order.
    pub effects: Vec<EffectWire>,
    /// Classification of every claim of the current scene, in key order.
    pub claims: Vec<ClaimReportWire>,
    /// Claims of the earlier scene absent now, sorted.
    pub removed: Vec<String>,
    /// Identity (hex) of the classification, re-derived on consumption.
    pub revalidation: String,
    /// BLAKE3 identity (hex) of everything above.
    pub identity: String,
}

// -- conversions --------------------------------------------------------------------------------

fn hex(digest: SemanticDigest) -> String {
    digest.to_hex()
}

fn digest_of(text: &str, what: &'static str) -> Result<SemanticDigest, Failure> {
    parse_digest_hex(text)
        .filter(|digest| digest.to_hex() == text)
        .ok_or_else(|| malformed(format!("{what} is not a lowercase 64 hex digit digest")))
}

fn kind_name(kind: SceneClaimKind) -> &'static str {
    match kind {
        SceneClaimKind::Transported => "transported",
        SceneClaimKind::Decision => "decision",
    }
}

fn kind_from_name(name: &str) -> Result<SceneClaimKind, Failure> {
    match name {
        "transported" => Ok(SceneClaimKind::Transported),
        "decision" => Ok(SceneClaimKind::Decision),
        _ => Err(Failure::UnsupportedSemantics("scene claim kind")),
    }
}

fn sorted_distinct(values: &[String]) -> Vec<String> {
    let mut out = values.to_vec();
    out.sort_unstable();
    out.dedup();
    out
}

fn calibration_to_wire(calibration: &SceneCalibration) -> SceneCalibrationWire {
    match calibration {
        SceneCalibration::PointOnly => SceneCalibrationWire {
            kind: "point_only".to_owned(),
            from: None,
            record: None,
            established_at_target: None,
        },
        SceneCalibration::Required => SceneCalibrationWire {
            kind: "required".to_owned(),
            from: None,
            record: None,
            established_at_target: None,
        },
        SceneCalibration::Held { record, established_at_target } => SceneCalibrationWire {
            kind: "held".to_owned(),
            from: None,
            record: Some(hex(*record)),
            established_at_target: Some(hex(*established_at_target)),
        },
        SceneCalibration::Inherited { from, record, established_at_target } => {
            SceneCalibrationWire {
                kind: "inherited".to_owned(),
                from: Some(from.clone()),
                record: Some(hex(*record)),
                established_at_target: Some(hex(*established_at_target)),
            }
        }
        // `SceneCalibration` is not marked non-exhaustive; keep the match total.
    }
}

fn calibration_from_wire(wire: &SceneCalibrationWire) -> Result<SceneCalibration, Failure> {
    let pair = || -> Result<(SemanticDigest, SemanticDigest), Failure> {
        let record = wire.record.as_deref().ok_or_else(|| malformed("calibration record"))?;
        let target = wire
            .established_at_target
            .as_deref()
            .ok_or_else(|| malformed("calibration target"))?;
        Ok((digest_of(record, "calibration record")?, digest_of(target, "calibration target")?))
    };
    let bare = wire.from.is_none() && wire.record.is_none() && wire.established_at_target.is_none();
    Ok(match wire.kind.as_str() {
        "point_only" if bare => SceneCalibration::PointOnly,
        "required" if bare => SceneCalibration::Required,
        "held" if wire.from.is_none() => {
            let (record, established_at_target) = pair()?;
            SceneCalibration::Held { record, established_at_target }
        }
        "inherited" => {
            let from = wire.from.clone().ok_or_else(|| malformed("calibration source"))?;
            let (record, established_at_target) = pair()?;
            SceneCalibration::Inherited { from, record, established_at_target }
        }
        _ => return Err(Failure::UnsupportedSemantics("scene calibration")),
    })
}

fn unsupported_of(support: &SupportStanding) -> Option<Vec<String>> {
    match support {
        SupportStanding::Unsupported { coordinates } => Some(coordinates.clone()),
        _ => None,
    }
}

fn scene_to_wire(scene: &TransportScene) -> SceneWire {
    let mut sources: Vec<SceneSourceWire> = scene
        .sources
        .iter()
        .map(|s| SceneSourceWire { key: s.key.clone(), identity: hex(s.identity) })
        .collect();
    sources.sort_by(|a, b| a.key.cmp(&b.key));
    let mut claims: Vec<SceneClaimWire> = scene
        .claims
        .iter()
        .map(|c| SceneClaimWire {
            key: c.key.clone(),
            kind: kind_name(c.kind).to_owned(),
            identity: hex(c.identity),
            structure: c.structure.map(hex),
            quantity: hex(c.quantity),
            sources: sorted_distinct(&c.sources),
            rests_on: sorted_distinct(&c.rests_on),
            unsupported: unsupported_of(&c.support),
            trust: c.trust.as_str().to_owned(),
            calibration: calibration_to_wire(&c.calibration),
        })
        .collect();
    claims.sort_by(|a, b| a.key.cmp(&b.key));
    SceneWire {
        target_identity: hex(scene.target.identity),
        target_unsupported: unsupported_of(&scene.target.support),
        selection_diagram: hex(scene.selection_diagram),
        sources,
        claims,
    }
}

fn check_text(text: &str, what: &'static str) -> Result<(), Failure> {
    if text.trim().is_empty() || text.len() > MAX_TEXT_BYTES {
        return Err(malformed(format!("{what} is blank or too long")));
    }
    Ok(())
}

fn scene_from_wire(wire: &SceneWire) -> Result<TransportScene, Failure> {
    if wire.sources.len() > MAX_TRANSPORT_REVALIDATION_ITEMS
        || wire.claims.len() > MAX_TRANSPORT_REVALIDATION_ITEMS
    {
        return Err(Failure::LimitsExceeded("scene items"));
    }
    let support = |unsupported: &Option<Vec<String>>| match unsupported {
        None => SupportStanding::Supported,
        Some(coordinates) => SupportStanding::Unsupported { coordinates: coordinates.clone() },
    };
    let mut sources = Vec::with_capacity(wire.sources.len());
    for source in &wire.sources {
        check_text(&source.key, "source key")?;
        sources.push(SceneSource {
            key: source.key.clone(),
            identity: digest_of(&source.identity, "source identity")?,
        });
    }
    let mut claims = Vec::with_capacity(wire.claims.len());
    for claim in &wire.claims {
        check_text(&claim.key, "claim key")?;
        if claim.sources.len() > MAX_TRANSPORT_REVALIDATION_ITEMS
            || claim.rests_on.len() > MAX_TRANSPORT_REVALIDATION_ITEMS
        {
            return Err(Failure::LimitsExceeded("claim dependencies"));
        }
        let trust = TrustStanding::from_name(&claim.trust)
            .ok_or(Failure::UnsupportedSemantics("trust standing"))?;
        claims.push(SceneClaim {
            key: claim.key.clone(),
            kind: kind_from_name(&claim.kind)?,
            identity: digest_of(&claim.identity, "claim identity")?,
            structure: claim
                .structure
                .as_deref()
                .map(|t| digest_of(t, "structure identity"))
                .transpose()?,
            quantity: digest_of(&claim.quantity, "quantity identity")?,
            sources: claim.sources.clone(),
            rests_on: claim.rests_on.clone(),
            support: support(&claim.unsupported),
            trust,
            calibration: calibration_from_wire(&claim.calibration)?,
        });
    }
    let scene = TransportScene {
        target: SceneTarget {
            identity: digest_of(&wire.target_identity, "target identity")?,
            support: support(&wire.target_unsupported),
        },
        selection_diagram: digest_of(&wire.selection_diagram, "selection diagram identity")?,
        sources,
        claims,
    };
    // A stored scene is in canonical order, so a rebuilt one lists the same
    // wire. Anything else was edited or was not produced by this writer.
    if scene_to_wire(&scene) != *wire {
        return Err(malformed("scene is not in canonical form"));
    }
    scene.graph().map_err(|error| graph_error(&error))?;
    Ok(scene)
}

fn effects_to_wire(effects: &BTreeMap<String, ObservedEffect>) -> Result<Vec<EffectWire>, Failure> {
    let mut out = Vec::with_capacity(effects.len());
    for (key, effect) in effects {
        let (kind, text) = match effect {
            ObservedEffect::Recomputed { cause } => ("recomputed", cause),
            ObservedEffect::Withdrawn { cause } => ("withdrawn", cause),
            ObservedEffect::Refused { detail } => ("refused", detail),
            _ => return Err(Failure::UnsupportedSemantics("effect kind")),
        };
        out.push(EffectWire { key: key.clone(), kind: kind.to_owned(), text: text.clone() });
    }
    Ok(out)
}

fn effects_from_wire(wire: &[EffectWire]) -> Result<BTreeMap<String, ObservedEffect>, Failure> {
    if wire.len() > 2 * MAX_TRANSPORT_REVALIDATION_ITEMS + 2 {
        return Err(Failure::LimitsExceeded("effects"));
    }
    let mut out = BTreeMap::new();
    for effect in wire {
        check_text(&effect.text, "effect text")?;
        check_text(&effect.key, "effect key")?;
        let value = match effect.kind.as_str() {
            "recomputed" => ObservedEffect::Recomputed { cause: effect.text.clone() },
            "withdrawn" => ObservedEffect::Withdrawn { cause: effect.text.clone() },
            "refused" => ObservedEffect::Refused { detail: effect.text.clone() },
            _ => return Err(Failure::UnsupportedSemantics("effect kind")),
        };
        if out.insert(effect.key.clone(), value).is_some() {
            return Err(malformed("effect declared twice for one node"));
        }
    }
    Ok(out)
}

fn reason_text(reason: &TransportReason) -> String {
    let name = reason.as_str();
    match reason {
        TransportReason::SourceEvidenceChanged { source } => format!("{name}:{source}"),
        TransportReason::ClaimChanged { claim, cause } => {
            format!("{name}:{claim}:{}", cause.as_str())
        }
        TransportReason::ClaimAdded { claim }
        | TransportReason::ProviderTrustLost { claim } => format!("{name}:{claim}"),
        TransportReason::ObservedStale { node } => format!("{name}:{node}"),
        TransportReason::TargetUnsupported { coordinates } => format!("{name}:{coordinates}"),
        TransportReason::ClaimUnsupported { claim, coordinates } => {
            format!("{name}:{claim}:{coordinates}")
        }
        TransportReason::InputRetired { input } => format!("{name}:{input}"),
        _ => name.to_owned(),
    }
}

fn report_to_wire(report: &TransportRevalidation) -> Vec<ClaimReportWire> {
    report
        .claims()
        .map(|claim| {
            let refusal = claim.refusal();
            ClaimReportWire {
                key: claim.key.clone(),
                kind: kind_name(claim.kind).to_owned(),
                class: claim.class.as_str().to_owned(),
                reasons: claim.reasons.iter().map(reason_text).collect(),
                authority_retained: matches!(
                    claim.authority,
                    antecedent_contract::composition_verification::PriorAuthority::Retained
                ),
                calibration_state: claim.calibration.as_str().to_owned(),
                refusal_code: refusal.as_ref().map(|r| r.code.to_owned()),
                refusal_detail: refusal.map(|r| r.detail),
            }
        })
        .collect()
}

// -- identity and consumption ---------------------------------------------------------------------

fn compute_identity(meta: &TransportRevalidationMeta) -> Result<String, Failure> {
    crate::digest_hex(
        IdentityDomain::Program,
        &(
            "transport_revalidation_artifact_v1",
            meta.version,
            &meta.feature,
            &meta.required_features,
            &meta.previous,
            &meta.current,
            &meta.effects,
            &meta.claims,
            &meta.removed,
            &meta.revalidation,
        ),
    )
    .map_err(|error| Failure::Encode(error.to_string()))
}

fn derive(
    previous: Option<&TransportScene>,
    current: &TransportScene,
    effects: &BTreeMap<String, ObservedEffect>,
) -> Result<TransportRevalidation, Failure> {
    revalidate_transport(previous.unwrap_or(current), current, effects)
        .map_err(|error| graph_error(&error))
}

fn build_meta(
    previous: Option<&TransportScene>,
    current: &TransportScene,
    effects: &BTreeMap<String, ObservedEffect>,
    report: &TransportRevalidation,
) -> Result<TransportRevalidationMeta, Failure> {
    let mut meta = TransportRevalidationMeta {
        version: TRANSPORT_REVALIDATION_ARTIFACT_VERSION,
        feature: TRANSPORT_REVALIDATION_ARTIFACT_FEATURE.to_owned(),
        required_features: Vec::new(),
        previous: previous.map(scene_to_wire),
        current: scene_to_wire(current),
        effects: effects_to_wire(effects)?,
        claims: report_to_wire(report),
        removed: report.removed().to_vec(),
        revalidation: hex(report.identity()),
        identity: String::new(),
    };
    meta.identity = compute_identity(&meta)?;
    Ok(meta)
}

fn check(
    meta: &TransportRevalidationMeta,
    expected_identity: Option<&str>,
) -> Result<TransportRevalidationArtifact, Failure> {
    if meta.version != TRANSPORT_REVALIDATION_ARTIFACT_VERSION {
        return Err(Failure::UnsupportedVersion { version: meta.version });
    }
    if meta.feature != TRANSPORT_REVALIDATION_ARTIFACT_FEATURE {
        return Err(Failure::UnsupportedSemantics("feature marker"));
    }
    if !meta.required_features.is_empty() {
        return Err(Failure::UnsupportedSemantics("required feature"));
    }
    if meta.claims.len() > MAX_TRANSPORT_REVALIDATION_ITEMS
        || meta.removed.len() > MAX_TRANSPORT_REVALIDATION_ITEMS
    {
        return Err(Failure::LimitsExceeded("report items"));
    }
    let current = scene_from_wire(&meta.current)?;
    let previous = meta.previous.as_ref().map(scene_from_wire).transpose()?;
    let effects = effects_from_wire(&meta.effects)?;
    // The classification is derived again from the stored scenes; no stored
    // class, reason, authority or calibration state is believed.
    let report = derive(previous.as_ref(), &current, &effects)?;
    if meta.revalidation != hex(report.identity()) {
        return Err(Failure::ReplayMismatch("classification identity"));
    }
    if meta.claims != report_to_wire(&report) {
        return Err(Failure::ReplayMismatch("claim classification"));
    }
    if meta.removed != report.removed() {
        return Err(Failure::ReplayMismatch("removed claims"));
    }
    let identity = compute_identity(meta)?;
    if meta.identity != identity {
        return Err(Failure::IdentityMismatch { field: "identity" });
    }
    if expected_identity.is_some_and(|expected| expected != identity) {
        return Err(Failure::IdentityMismatch { field: "retained_identity" });
    }
    Ok(TransportRevalidationArtifact { previous, current, effects, report, identity })
}

fn encode_meta(meta: &TransportRevalidationMeta, artifact_id: &str) -> Result<Vec<u8>, Failure> {
    if artifact_id.trim().is_empty() {
        return Err(Failure::Encode("missing artifact id".into()));
    }
    let meta_bytes = to_cbor(meta).map_err(|error| Failure::Encode(error.to_string()))?;
    crate::encode_container(
        &crate::ArtifactContainer {
            kind: ARTIFACT_KIND,
            note: "transport_revalidation",
            max_bytes: MAX_TRANSPORT_REVALIDATION_ARTIFACT_BYTES,
            artifact_id,
            library_version: env!("CARGO_PKG_VERSION"),
        },
        &[crate::OutSection {
            id: META_SECTION,
            content_type: crate::CONTENT_CBOR,
            bytes: &meta_bytes,
        }],
    )
    .map_err(|error| crate::map_artifact_check!(error, TransportRevalidationArtifactError))
}

fn decode_meta(bytes: &[u8]) -> Result<TransportRevalidationMeta, Failure> {
    let parts = crate::decode_container(
        ARTIFACT_KIND,
        MAX_TRANSPORT_REVALIDATION_ARTIFACT_BYTES,
        &[META_SECTION],
        bytes,
    )
    .map_err(|error| crate::map_artifact_check!(error, TransportRevalidationArtifactError))?;
    let Ok([meta_bytes]) = <[Vec<u8>; 1]>::try_from(parts) else {
        return Err(Failure::Malformed("unsupported container layout".into()));
    };
    crate::decode_with_version(&meta_bytes, |version: u16| {
        if version == TRANSPORT_REVALIDATION_ARTIFACT_VERSION {
            Ok(())
        } else {
            Err(Failure::UnsupportedVersion { version })
        }
    })
}

// -- public type ----------------------------------------------------------------------------------

/// A validated transport revalidation report and the scenes it was derived from.
#[derive(Clone, Debug)]
pub struct TransportRevalidationArtifact {
    previous: Option<TransportScene>,
    current: TransportScene,
    effects: BTreeMap<String, ObservedEffect>,
    report: TransportRevalidation,
    identity: String,
}

impl TransportRevalidationArtifact {
    /// Compare `current` with `previous` (or with itself for a first
    /// verification), apply `effects`, and seal the classification with the
    /// scenes it came from.
    ///
    /// # Errors
    /// A scene whose graph does not build, a bound exceeded, or an effect kind
    /// the format does not carry.
    pub fn seal(
        previous: Option<&TransportScene>,
        current: &TransportScene,
        effects: &BTreeMap<String, ObservedEffect>,
    ) -> Result<(Self, TransportRevalidationMeta), Failure> {
        let report = derive(previous, current, effects)?;
        let meta = build_meta(previous, current, effects, &report)?;
        let artifact = check(&meta, None)?;
        Ok((artifact, meta))
    }

    /// Serialize a sealed report through the checksummed container.
    ///
    /// # Errors
    /// A report that does not seal, an encoding failure or an oversized payload.
    pub fn to_bytes(
        previous: Option<&TransportScene>,
        current: &TransportScene,
        effects: &BTreeMap<String, ObservedEffect>,
        artifact_id: &str,
    ) -> Result<Vec<u8>, Failure> {
        let (_, meta) = Self::seal(previous, current, effects)?;
        encode_meta(&meta, artifact_id)
    }

    /// Consume an artifact: rebuild both scenes, classify again and compare
    /// the whole stored report and every identity.
    ///
    /// `expected_identity` is an artifact identity (hex) the consumer retained
    /// independently; when given, a consistently resealed artifact of another
    /// comparison is refused.
    ///
    /// # Errors
    /// Corruption, another major version, an unknown or required feature, a
    /// bound exceeded, a scene the library refuses, a stored classification that
    /// does not replay, or an identity that does not match.
    pub fn from_bytes(bytes: &[u8], expected_identity: Option<&str>) -> Result<Self, Failure> {
        check(&decode_meta(bytes)?, expected_identity)
    }

    /// Read the stored metadata without replaying it.
    ///
    /// # Errors
    /// Corruption, another major version or a malformed container.
    pub fn inspect(bytes: &[u8]) -> Result<TransportRevalidationMeta, Failure> {
        decode_meta(bytes)
    }

    /// The re-derived classification.
    #[must_use]
    pub const fn revalidation(&self) -> &TransportRevalidation {
        &self.report
    }

    /// The current scene.
    #[must_use]
    pub const fn current(&self) -> &TransportScene {
        &self.current
    }

    /// The earlier scene, when the comparison had one.
    #[must_use]
    pub const fn previous(&self) -> Option<&TransportScene> {
        self.previous.as_ref()
    }

    /// The observed effects applied.
    #[must_use]
    pub const fn effects(&self) -> &BTreeMap<String, ObservedEffect> {
        &self.effects
    }

    /// The artifact identity (hex).
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// The classification identity (hex).
    #[must_use]
    pub fn revalidation_identity(&self) -> String {
        hex(self.report.identity())
    }

    /// Validate metadata as a consumer would, for tests that reseal edits.
    ///
    /// # Errors
    /// The same refusals as [`Self::from_bytes`].
    #[doc(hidden)]
    pub fn from_meta(
        meta: &TransportRevalidationMeta,
        expected_identity: Option<&str>,
    ) -> Result<Self, Failure> {
        check(meta, expected_identity)
    }

    /// Recompute the identity of `meta` after an edit, for tests that reseal.
    ///
    /// # Errors
    /// An encoding failure.
    #[doc(hidden)]
    pub fn reseal_identity(meta: &mut TransportRevalidationMeta) -> Result<(), Failure> {
        meta.identity = compute_identity(meta)?;
        Ok(())
    }

    /// Serialize metadata as it is, for tests that write an edited layout.
    ///
    /// # Errors
    /// An encoding failure.
    #[doc(hidden)]
    pub fn meta_to_bytes(
        meta: &TransportRevalidationMeta,
        artifact_id: &str,
    ) -> Result<Vec<u8>, Failure> {
        encode_meta(meta, artifact_id)
    }
}
