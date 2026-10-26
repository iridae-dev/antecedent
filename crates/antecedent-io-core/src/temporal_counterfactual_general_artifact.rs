//! Fixed-population temporal counterfactual artifact for any horizon and any number of worlds
//! (`temporal_counterfactual_general_v1`).
//!
//! The two-slice, two-world artifact of [`crate::temporal_counterfactual_artifact`] keeps its
//! stable layout. This artifact carries the same answer for the general executor
//! ([`antecedent_counterfactual::temporal_cross_world::general`]): one factual trajectory of
//! `2 * horizon + 1` nodes per unit, any number of named action histories, and a receipt with
//! one digest per history.
//!
//! * `temporal_counterfactual_general_meta` (CBOR): version, feature marker, mechanism class,
//!   the point-only claim, snapshot, horizon, mechanism fit, unit and history ids with their
//!   observation times, every named action history, the receipt, the identity digests and the
//!   stored world means;
//! * `temporal_counterfactual_general_values` (little-endian `f64`): for each unit in unit-id
//!   order, its `2 * horizon + 1` factual values and then its final outcome under each world.
//!
//! A consumer trusts none of the stored outcomes: it rebuilds the request, replays every world
//! with the general evaluator and refuses unless every per-unit outcome, every mean and the whole
//! receipt are bit-identical. The size of a request is bounded by the evaluator's work budget and
//! the container's byte budget, not by a count of slices or worlds.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{CanonicalWriter, ExecutionContext};
use antecedent_counterfactual::temporal_cross_world::general::{
    GeneralActionHistory, GeneralAbductionReceipt, GeneralFactualHistory, GeneralMechanismFit,
    GeneralNodeMechanism, GeneralTemporalResult, GeneralTemporalSpec, node_count,
    evaluate_general_temporal_counterfactual,
};
use antecedent_counterfactual::temporal_cross_world::{
    HistoryId, MECHANISM_CLASS, SnapshotId, TemporalCounterfactualError, UnitId,
};
use serde::{Deserialize, Serialize};

use crate::convert::to_cbor;
use crate::temporal_counterfactual_artifact::{RefusalWire, TemporalCounterfactualRefusalWire, UnitDrawWire, WitnessWire};

/// The artifact major version this reader writes and accepts.
pub const GENERAL_TEMPORAL_COUNTERFACTUAL_ARTIFACT_VERSION: u16 = 1;
/// The feature marker of the accepted format.
pub const GENERAL_TEMPORAL_COUNTERFACTUAL_ARTIFACT_FEATURE: &str =
    "temporal_counterfactual_general_v1";
/// The only claim of this route.
pub const GENERAL_TEMPORAL_COUNTERFACTUAL_INFERENCE_CLAIM: &str = "point_only";
/// Most bytes an artifact may occupy, enforced on export and on consumption.
pub const MAX_GENERAL_TEMPORAL_COUNTERFACTUAL_ARTIFACT_BYTES: usize = 128 * 1024 * 1024;

const ARTIFACT_KIND: &str = crate::artifact_kinds::GENERAL_TEMPORAL_COUNTERFACTUAL;
const META_SECTION: &str = "temporal_counterfactual_general_meta";
const VALUES_SECTION: &str = "temporal_counterfactual_general_values";

crate::sectioned_artifact_enum! {
    /// Why a general temporal counterfactual artifact was refused or could not be produced.
    #[derive(Clone, Debug, PartialEq)]
    GeneralTemporalCounterfactualArtifactError {
        noun: "general temporal counterfactual",
        identity: "temporal_counterfactual.artifact_changed",
        variants {
            /// A stored outcome, mean or receipt differs from the recomputed one.
            ResultMismatch(&'static str),
            /// Cooperative cancellation was observed.
            Cancelled,
            /// The core evaluator refused.
            Refused(Box<RefusalWire>),
        }
        display(f) {
            Self::ResultMismatch(what) => {
                write!(f, "stored general temporal counterfactual does not replay: {what}")
            }
            Self::Cancelled => {
                f.write_str("cancelled during general temporal counterfactual evaluation")
            }
            Self::Refused(wire) => {
                write!(f, "general temporal counterfactual refused: {}", wire.detail)
            }
        }
    }
}

type Error = GeneralTemporalCounterfactualArtifactError;

impl GeneralTemporalCounterfactualArtifactError {
    /// The structured refusal this error carries, for a core refusal and a changed identity.
    #[must_use]
    pub fn refusal(&self) -> Option<RefusalWire> {
        match self {
            Self::Refused(wire) => Some((**wire).clone()),
            Self::IdentityMismatch { field } => Some(TemporalCounterfactualRefusalWire {
                code: antecedent_core::reason_code!("route_not_supported").to_owned(),
                stage: "consume".to_owned(),
                detail: "temporal_counterfactual.artifact_changed".to_owned(),
                offending: Some((*field).to_owned()),
                remedy: Some(
                    "consume the artifact the producer exported, or retain the identity of the \
                     inputs you intend"
                        .to_owned(),
                ),
                witness: None,
            }),
            _ => None,
        }
    }
}

fn from_core(error: TemporalCounterfactualError) -> Error {
    match error {
        TemporalCounterfactualError::Cancelled => Error::Cancelled,
        TemporalCounterfactualError::Refused(refused) => {
            Error::Refused(Box::new(TemporalCounterfactualRefusalWire {
                code: refused.refusal.code.to_owned(),
                stage: refused.refusal.stage.to_owned(),
                detail: refused.refusal.detail.clone(),
                offending: refused.refusal.offending.clone(),
                remedy: refused.refusal.remedy.map(str::to_owned),
                witness: refused.witness.as_ref().map(|w| WitnessWire {
                    unit: w.unit.as_ref().map(|u| u.as_str().to_owned()),
                    history: w.history.as_ref().map(|h| h.as_str().to_owned()),
                    world: w.world.clone(),
                    node: w.node.map(|n| n.name().to_owned()),
                    residual: w.residual,
                    bound: w.bound,
                }),
            }))
        }
        other => Error::Malformed(other.to_string()),
    }
}

/// One node's linear additive-noise mechanism on the wire (nodes by index).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeneralMechanismWire {
    /// Node index it generates (`2 * slice` for a covariate, `2 * horizon` for the outcome).
    pub node: u32,
    /// Intercept.
    pub intercept: f64,
    /// Parent coefficients by node index.
    pub parent_coefficients: Vec<(u32, f64)>,
    /// Optional half-width of the noise support.
    pub noise_halfwidth: Option<f64>,
}

/// The mechanism fit on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeneralFitWire {
    /// Stable identity of the fit.
    pub fit_id: String,
    /// One mechanism per covariate and the outcome.
    pub mechanisms: Vec<GeneralMechanismWire>,
}

/// One unit's factual trajectory on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeneralFactualUnitWire {
    /// The unit.
    pub unit: String,
    /// The unit's factual history id.
    pub history: String,
    /// Observation time of each slice.
    pub times: Vec<u32>,
    /// Observed values of all `2 * horizon + 1` nodes, in node order.
    pub values: Vec<f64>,
}

/// A named action history on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeneralWorldWire {
    /// Name of the history (the world).
    pub name: String,
    /// Time of each slice's action.
    pub times: Vec<u32>,
    /// The action of each slice.
    pub actions: Vec<f64>,
    /// The units the world is evaluated for.
    pub units: Vec<String>,
}

/// A complete general temporal counterfactual request.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeneralTemporalRequestWire {
    /// Number of slices.
    pub horizon: u32,
    /// Whether the graph has latent confounding (refused).
    pub latent_confounding: bool,
    /// The mechanism fit.
    pub fit: GeneralFitWire,
    /// Snapshot the factual histories were read from.
    pub snapshot: String,
    /// Factual histories, one per unit.
    pub factual: Vec<GeneralFactualUnitWire>,
    /// The named action histories.
    pub worlds: Vec<GeneralWorldWire>,
}

/// The shared-abduction receipt on the wire.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeneralReceiptWire {
    /// Snapshot identity.
    pub snapshot: String,
    /// Digest of the horizon and mechanism fit.
    pub fit_digest: String,
    /// Digest of the factual histories.
    pub factual_digest: String,
    /// Digest of each action history, in request order.
    pub world_digests: Vec<String>,
    /// Per-unit exogenous-draw digests, sorted by unit id.
    pub unit_draws: Vec<UnitDrawWire>,
    /// Digest over all unit draws.
    pub exogenous_digest: String,
    /// Number of units.
    pub n_units: u64,
    /// Number of worlds.
    pub n_worlds: u64,
    /// Horizon.
    pub horizon: u64,
    /// Every world read the same draw for every unit.
    pub shared_by_all_worlds: bool,
    /// Overall receipt digest.
    pub digest: String,
}

impl GeneralReceiptWire {
    fn from_receipt(receipt: &GeneralAbductionReceipt) -> Self {
        Self {
            snapshot: receipt.snapshot.as_str().to_owned(),
            fit_digest: receipt.fit_digest.clone(),
            factual_digest: receipt.factual_digest.clone(),
            world_digests: receipt.world_digests.clone(),
            unit_draws: receipt
                .unit_draws
                .iter()
                .map(|d| UnitDrawWire {
                    unit: d.unit.as_str().to_owned(),
                    history: d.history.as_str().to_owned(),
                    digest: d.digest.clone(),
                })
                .collect(),
            exogenous_digest: receipt.exogenous_digest.clone(),
            n_units: receipt.n_units as u64,
            n_worlds: receipt.n_worlds as u64,
            horizon: receipt.horizon as u64,
            shared_by_all_worlds: receipt.shared_by_all_worlds,
            digest: receipt.digest.clone(),
        }
    }
}

/// Identity digests of one general temporal counterfactual.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeneralTemporalIdentity {
    /// Snapshot id.
    pub snapshot: String,
    /// Horizon and mechanism fit digest.
    pub fit_digest: String,
    /// Unit-history digest.
    pub factual_digest: String,
    /// Digest of each action history, in request order.
    pub world_digests: Vec<String>,
    /// Digest over every unit's exogenous draw.
    pub exogenous_digest: String,
    /// Overall receipt digest.
    pub receipt_digest: String,
    /// Digest over the canonical request.
    pub spec_id: String,
}

/// One unit's identifiers and observation times (the values are in the values section).
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeneralUnitMeta {
    /// The unit.
    pub unit: String,
    /// The unit's factual history id.
    pub history: String,
    /// Observation time of each slice.
    pub times: Vec<u32>,
}

/// The CBOR metadata section.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeneralTemporalCounterfactualMeta {
    /// Major version.
    pub version: u16,
    /// Feature marker.
    pub feature: String,
    /// Mechanism class of the route.
    pub mechanism_class: String,
    /// Always `point_only`.
    pub inference_claim: String,
    /// Number of slices.
    pub horizon: u32,
    /// Whether the graph declares latent confounding.
    pub latent_confounding: bool,
    /// The mechanism fit.
    pub fit: GeneralFitWire,
    /// Snapshot id.
    pub snapshot: String,
    /// Units in unit-id order.
    pub units: Vec<GeneralUnitMeta>,
    /// The named action histories.
    pub worlds: Vec<GeneralWorldWire>,
    /// The shared-abduction receipt.
    pub receipt: GeneralReceiptWire,
    /// Identity digests.
    pub identity: GeneralTemporalIdentity,
    /// Mean final outcome under each world, in request order.
    pub world_means: Vec<f64>,
}

/// One unit's counterfactual outcomes on the wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeneralUnitOutcomeWire {
    /// The unit.
    pub unit: String,
    /// The unit's factual history.
    pub history: String,
    /// The factual final outcome.
    pub factual_outcome: f64,
    /// Final outcome under each world, in request order.
    pub outcomes: Vec<f64>,
}

/// The full, self-describing report handed to host languages.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeneralTemporalReportWire {
    /// Major version.
    pub version: u16,
    /// Feature marker.
    pub feature: String,
    /// Mechanism class.
    pub mechanism_class: String,
    /// Always `point_only`.
    pub inference_claim: String,
    /// Snapshot id.
    pub snapshot: String,
    /// Number of slices.
    pub horizon: u32,
    /// World names, in request order.
    pub world_names: Vec<String>,
    /// Per-unit outcomes, sorted by unit id.
    pub units: Vec<GeneralUnitOutcomeWire>,
    /// Mean final outcome under each world.
    pub world_means: Vec<f64>,
    /// The shared-abduction receipt.
    pub receipt: GeneralReceiptWire,
    /// Identity digests.
    pub identity: GeneralTemporalIdentity,
}

fn spec_of(request: &GeneralTemporalRequestWire) -> GeneralTemporalSpec {
    let index = |n: u32| usize::try_from(n).unwrap_or(usize::MAX);
    GeneralTemporalSpec {
        horizon: index(request.horizon),
        latent_confounding: request.latent_confounding,
        fit: GeneralMechanismFit {
            fit_id: request.fit.fit_id.clone(),
            mechanisms: request
                .fit
                .mechanisms
                .iter()
                .map(|m| GeneralNodeMechanism {
                    node: index(m.node),
                    intercept: m.intercept,
                    parent_coefficients: m
                        .parent_coefficients
                        .iter()
                        .map(|&(p, c)| (index(p), c))
                        .collect(),
                    noise_halfwidth: m.noise_halfwidth,
                })
                .collect(),
        },
        snapshot: SnapshotId::new(request.snapshot.as_str()),
        factual: request
            .factual
            .iter()
            .map(|f| GeneralFactualHistory {
                unit: UnitId::new(f.unit.as_str()),
                history: HistoryId::new(f.history.as_str()),
                times: f.times.clone(),
                values: f.values.clone(),
            })
            .collect(),
        worlds: request
            .worlds
            .iter()
            .map(|w| GeneralActionHistory {
                name: w.name.clone(),
                times: w.times.clone(),
                actions: w.actions.clone(),
                units: w.units.iter().map(UnitId::new).collect(),
            })
            .collect(),
    }
}

fn spec_id(request: &GeneralTemporalRequestWire) -> String {
    let mut c = CanonicalWriter::new("temporal_counterfactual_general_v1.spec");
    c.text(&request.snapshot);
    c.word(u64::from(request.horizon));
    c.word(u64::from(request.latent_confounding));
    c.text(&request.fit.fit_id);
    let mut mechanisms: Vec<&GeneralMechanismWire> = request.fit.mechanisms.iter().collect();
    mechanisms.sort_by_key(|m| m.node);
    c.word(mechanisms.len() as u64);
    for m in mechanisms {
        c.word(u64::from(m.node));
        c.real(m.intercept);
        let mut coefs = m.parent_coefficients.clone();
        coefs.sort_by_key(|&(p, _)| p);
        c.word(coefs.len() as u64);
        for (p, coef) in &coefs {
            c.word(u64::from(*p));
            c.real(*coef);
        }
        match m.noise_halfwidth {
            Some(h) => {
                c.word(1);
                c.real(h);
            }
            None => {
                c.word(0);
            }
        }
    }
    let mut factual: Vec<&GeneralFactualUnitWire> = request.factual.iter().collect();
    factual.sort_by(|a, b| a.unit.cmp(&b.unit));
    c.word(factual.len() as u64);
    for f in factual {
        c.text(&f.unit);
        c.text(&f.history);
        c.word(f.times.len() as u64);
        for &t in &f.times {
            c.word(u64::from(t));
        }
        c.reals(&f.values);
    }
    c.word(request.worlds.len() as u64);
    for w in &request.worlds {
        c.text(&w.name);
        c.word(w.times.len() as u64);
        for &t in &w.times {
            c.word(u64::from(t));
        }
        c.reals(&w.actions);
        let mut units = w.units.clone();
        units.sort();
        c.word(units.len() as u64);
        for u in &units {
            c.text(u);
        }
    }
    c.finish_hex()
}

fn identity_of(
    request: &GeneralTemporalRequestWire,
    receipt: &GeneralAbductionReceipt,
) -> GeneralTemporalIdentity {
    GeneralTemporalIdentity {
        snapshot: receipt.snapshot.as_str().to_owned(),
        fit_digest: receipt.fit_digest.clone(),
        factual_digest: receipt.factual_digest.clone(),
        world_digests: receipt.world_digests.clone(),
        exogenous_digest: receipt.exogenous_digest.clone(),
        receipt_digest: receipt.digest.clone(),
        spec_id: spec_id(request),
    }
}

/// The first identity field that differs, in the order the contract names them.
fn identity_diff(
    stored: &GeneralTemporalIdentity,
    other: &GeneralTemporalIdentity,
) -> Option<&'static str> {
    if stored.snapshot != other.snapshot {
        Some("snapshot")
    } else if stored.fit_digest != other.fit_digest {
        Some("mechanism_fit")
    } else if stored.factual_digest != other.factual_digest {
        Some("unit_history")
    } else if stored.world_digests != other.world_digests {
        Some("action_history")
    } else if stored.exogenous_digest != other.exogenous_digest {
        Some("shared_abduction")
    } else if stored.receipt_digest != other.receipt_digest {
        Some("receipt")
    } else if stored.spec_id != other.spec_id {
        Some("request")
    } else {
        None
    }
}

/// Reals stored per unit: the factual nodes, then one outcome per world.
fn reals_per_unit(horizon: u32, n_worlds: usize) -> Option<usize> {
    node_count(usize::try_from(horizon).ok()?).checked_add(n_worlds)
}

fn values_of(request: &GeneralTemporalRequestWire, result: &GeneralTemporalResult) -> Vec<u8> {
    let mut reals = Vec::new();
    for (factual, outcome) in request.factual.iter().zip(&result.units) {
        reals.extend_from_slice(&factual.values);
        reals.extend_from_slice(&outcome.outcomes);
    }
    crate::artifact_codec::f64s_to_le_bytes(&reals)
}

fn encode_parts_impl(
    meta: &GeneralTemporalCounterfactualMeta,
    values: &[u8],
    artifact_id: &str,
) -> Result<Vec<u8>, Error> {
    if artifact_id.trim().is_empty() {
        return Err(Error::Encode("missing artifact id".into()));
    }
    let meta_bytes = to_cbor(meta).map_err(|error| Error::Encode(error.to_string()))?;
    crate::encode_container(
        &crate::ArtifactContainer {
            kind: ARTIFACT_KIND,
            note: "temporal_counterfactual_general_shared_abduction_point_only",
            max_bytes: MAX_GENERAL_TEMPORAL_COUNTERFACTUAL_ARTIFACT_BYTES,
            artifact_id,
            library_version: env!("CARGO_PKG_VERSION"),
        },
        &[
            crate::OutSection {
                id: META_SECTION,
                content_type: crate::CONTENT_CBOR,
                bytes: &meta_bytes,
            },
            crate::OutSection { id: VALUES_SECTION, content_type: crate::CONTENT_BYTES, bytes: values },
        ],
    )
    .map_err(|error| crate::map_artifact_check!(error, GeneralTemporalCounterfactualArtifactError))
}

/// Encode a metadata section and a values section as a checksummed container.
///
/// Hidden: the producer path is [`GeneralTemporalCounterfactualArtifact::to_bytes`]; tests use
/// this to build deliberately inconsistent or resealed artifacts.
///
/// # Errors
/// An empty id, an encoding failure or an oversized payload.
#[cfg(feature = "test-util")]
#[doc(hidden)]
pub fn encode_parts(
    meta: &GeneralTemporalCounterfactualMeta,
    values: &[u8],
    artifact_id: &str,
) -> Result<Vec<u8>, GeneralTemporalCounterfactualArtifactError> {
    encode_parts_impl(meta, values, artifact_id)
}

fn decode_parts_impl(
    bytes: &[u8],
) -> Result<(GeneralTemporalCounterfactualMeta, Vec<u8>), Error> {
    let parts = crate::decode_container(
        ARTIFACT_KIND,
        MAX_GENERAL_TEMPORAL_COUNTERFACTUAL_ARTIFACT_BYTES,
        &[META_SECTION, VALUES_SECTION],
        bytes,
    )
    .map_err(|error| crate::map_artifact_check!(error, GeneralTemporalCounterfactualArtifactError))?;
    let Ok([meta_bytes, payload]) = <[Vec<u8>; 2]>::try_from(parts) else {
        return Err(Error::Malformed("unsupported container layout".into()));
    };
    let meta = crate::decode_with_version(&meta_bytes, |version| {
        if version == GENERAL_TEMPORAL_COUNTERFACTUAL_ARTIFACT_VERSION {
            Ok(())
        } else {
            Err(Error::UnsupportedVersion { version })
        }
    })?;
    Ok((meta, payload))
}

/// Decode the two sections of a container, refusing another version before the metadata is
/// interpreted.
///
/// Hidden: see [`encode_parts`].
///
/// # Errors
/// Oversized, truncated, corrupt, differently laid out or other-version artifacts.
#[cfg(feature = "test-util")]
#[doc(hidden)]
pub fn decode_parts(
    bytes: &[u8],
) -> Result<
    (GeneralTemporalCounterfactualMeta, Vec<u8>),
    GeneralTemporalCounterfactualArtifactError,
> {
    decode_parts_impl(bytes)
}

fn request_of(
    meta: &GeneralTemporalCounterfactualMeta,
    values: &[u8],
) -> Result<GeneralTemporalRequestWire, Error> {
    let per_unit = reals_per_unit(meta.horizon, meta.worlds.len())
        .ok_or(Error::LimitsExceeded("horizon"))?;
    let expected = meta
        .units
        .len()
        .checked_mul(per_unit)
        .and_then(|n| n.checked_mul(8))
        .ok_or(Error::LimitsExceeded("units"))?;
    if values.len() != expected {
        return Err(Error::Malformed("values section length".into()));
    }
    let nodes = per_unit - meta.worlds.len();
    let reals = crate::artifact_codec::f64s_from_le_bytes(values);
    let factual = meta
        .units
        .iter()
        .zip(reals.chunks_exact(per_unit.max(1)))
        .map(|(u, row)| GeneralFactualUnitWire {
            unit: u.unit.clone(),
            history: u.history.clone(),
            times: u.times.clone(),
            values: row[..nodes].to_vec(),
        })
        .collect();
    Ok(GeneralTemporalRequestWire {
        horizon: meta.horizon,
        latent_confounding: meta.latent_confounding,
        fit: meta.fit.clone(),
        snapshot: meta.snapshot.clone(),
        factual,
        worlds: meta.worlds.clone(),
    })
}

/// A produced or consumed general temporal counterfactual artifact.
#[derive(Clone, Debug, PartialEq)]
pub struct GeneralTemporalCounterfactualArtifact {
    meta: GeneralTemporalCounterfactualMeta,
    request: GeneralTemporalRequestWire,
    result: GeneralTemporalResult,
}

impl GeneralTemporalCounterfactualArtifact {
    /// Evaluate `request` with the general evaluator and seal the answer with its receipt.
    ///
    /// The request is canonicalized to unit-id order.
    ///
    /// # Errors
    /// The evaluator's refusals, cancellation and the container's byte budget.
    pub fn seal(
        request: &GeneralTemporalRequestWire,
        ctx: &ExecutionContext,
    ) -> Result<Self, Error> {
        let mut request = request.clone();
        request.factual.sort_by(|a, b| a.unit.cmp(&b.unit));
        let result =
            evaluate_general_temporal_counterfactual(&spec_of(&request), ctx).map_err(from_core)?;
        let meta = GeneralTemporalCounterfactualMeta {
            version: GENERAL_TEMPORAL_COUNTERFACTUAL_ARTIFACT_VERSION,
            feature: GENERAL_TEMPORAL_COUNTERFACTUAL_ARTIFACT_FEATURE.to_owned(),
            mechanism_class: MECHANISM_CLASS.to_owned(),
            inference_claim: GENERAL_TEMPORAL_COUNTERFACTUAL_INFERENCE_CLAIM.to_owned(),
            horizon: request.horizon,
            latent_confounding: request.latent_confounding,
            fit: request.fit.clone(),
            snapshot: request.snapshot.clone(),
            units: request
                .factual
                .iter()
                .map(|f| GeneralUnitMeta {
                    unit: f.unit.clone(),
                    history: f.history.clone(),
                    times: f.times.clone(),
                })
                .collect(),
            worlds: request.worlds.clone(),
            receipt: GeneralReceiptWire::from_receipt(&result.receipt),
            identity: identity_of(&request, &result.receipt),
            world_means: result.world_means.clone(),
        };
        Ok(Self { meta, request, result })
    }

    /// The metadata section.
    #[must_use]
    pub fn meta(&self) -> &GeneralTemporalCounterfactualMeta {
        &self.meta
    }

    /// The canonical (unit-id ordered) request.
    #[must_use]
    pub fn request(&self) -> &GeneralTemporalRequestWire {
        &self.request
    }

    /// The core result.
    #[must_use]
    pub fn result(&self) -> &GeneralTemporalResult {
        &self.result
    }

    /// The identity digests.
    #[must_use]
    pub fn identity(&self) -> &GeneralTemporalIdentity {
        &self.meta.identity
    }

    /// The self-describing report for host languages.
    #[must_use]
    pub fn report(&self) -> GeneralTemporalReportWire {
        GeneralTemporalReportWire {
            version: self.meta.version,
            feature: self.meta.feature.clone(),
            mechanism_class: self.meta.mechanism_class.clone(),
            inference_claim: self.meta.inference_claim.clone(),
            snapshot: self.meta.snapshot.clone(),
            horizon: self.meta.horizon,
            world_names: self.result.world_names.clone(),
            units: self
                .result
                .units
                .iter()
                .map(|u| GeneralUnitOutcomeWire {
                    unit: u.unit.as_str().to_owned(),
                    history: u.history.as_str().to_owned(),
                    factual_outcome: u.factual_outcome,
                    outcomes: u.outcomes.clone(),
                })
                .collect(),
            world_means: self.meta.world_means.clone(),
            receipt: self.meta.receipt.clone(),
            identity: self.meta.identity.clone(),
        }
    }

    /// Serialize through the checksummed sectioned container.
    ///
    /// # Errors
    /// An empty id, an encoding failure or an oversized payload.
    pub fn to_bytes(&self, artifact_id: &str) -> Result<Vec<u8>, Error> {
        encode_parts_impl(&self.meta, &values_of(&self.request, &self.result), artifact_id)
    }

    /// Consume an artifact by recomputing every world.
    ///
    /// `expected` is an identity the consumer retained independently; when given, every field
    /// must match it, so a resealed change of action history, unit history, snapshot or
    /// mechanism fit is refused.
    ///
    /// # Errors
    /// Corruption, another major version, unsupported semantics, a changed identity, a stored
    /// outcome or receipt that does not replay, cancellation, or a core refusal.
    pub fn from_bytes(
        bytes: &[u8],
        expected: Option<&GeneralTemporalIdentity>,
        ctx: &ExecutionContext,
    ) -> Result<Self, Error> {
        let (meta, values) = decode_parts_impl(bytes)?;
        if meta.feature != GENERAL_TEMPORAL_COUNTERFACTUAL_ARTIFACT_FEATURE {
            return Err(Error::UnsupportedSemantics("feature marker"));
        }
        if meta.mechanism_class != MECHANISM_CLASS {
            return Err(Error::UnsupportedSemantics("mechanism class"));
        }
        if meta.inference_claim != GENERAL_TEMPORAL_COUNTERFACTUAL_INFERENCE_CLAIM {
            return Err(Error::UnsupportedSemantics("inference claim"));
        }
        let request = request_of(&meta, &values)?;
        let result =
            evaluate_general_temporal_counterfactual(&spec_of(&request), ctx).map_err(from_core)?;
        let recomputed = identity_of(&request, &result.receipt);
        if let Some(field) = identity_diff(&meta.identity, &recomputed) {
            return Err(Error::IdentityMismatch { field });
        }
        if meta.receipt != GeneralReceiptWire::from_receipt(&result.receipt) {
            return Err(Error::ResultMismatch("shared-abduction receipt"));
        }
        if result.units.len() != meta.units.len() {
            return Err(Error::ResultMismatch("per-unit outcomes"));
        }
        let per_unit = reals_per_unit(meta.horizon, meta.worlds.len())
            .ok_or(Error::LimitsExceeded("horizon"))?;
        let nodes = per_unit - meta.worlds.len();
        let stored = crate::artifact_codec::f64s_from_le_bytes(&values);
        let outcomes_replay = stored.chunks_exact(per_unit.max(1)).zip(&result.units).all(
            |(row, unit)| {
                row[nodes..].iter().map(|v| v.to_bits()).eq(unit.outcomes.iter().map(|v| v.to_bits()))
            },
        );
        if !outcomes_replay {
            return Err(Error::ResultMismatch("per-unit outcomes"));
        }
        let means_replay = meta.world_means.len() == result.world_means.len()
            && meta
                .world_means
                .iter()
                .zip(&result.world_means)
                .all(|(a, b)| a.to_bits() == b.to_bits());
        if !means_replay {
            return Err(Error::ResultMismatch("means"));
        }
        if let Some(expected) = expected {
            if let Some(field) = identity_diff(expected, &recomputed) {
                return Err(Error::IdentityMismatch { field });
            }
        }
        Ok(Self { meta, request, result })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(horizon: u32, n_worlds: usize) -> GeneralTemporalRequestWire {
        let nodes = 2 * horizon as usize + 1;
        let mut mechanisms = Vec::new();
        for slice in 0..horizon {
            let node = 2 * slice;
            mechanisms.push(GeneralMechanismWire {
                node,
                intercept: 1.0,
                parent_coefficients: if node == 0 { vec![] } else { vec![(node - 1, 0.5)] },
                noise_halfwidth: None,
            });
        }
        mechanisms.push(GeneralMechanismWire {
            node: 2 * horizon,
            intercept: 0.0,
            parent_coefficients: vec![(2 * horizon - 1, 2.0)],
            noise_halfwidth: None,
        });
        let units: Vec<String> = (0..3).map(|i| format!("u{i}")).collect();
        GeneralTemporalRequestWire {
            horizon,
            latent_confounding: false,
            fit: GeneralFitWire { fit_id: "fit".into(), mechanisms },
            snapshot: "snap".into(),
            factual: units
                .iter()
                .enumerate()
                .map(|(i, u)| GeneralFactualUnitWire {
                    unit: u.clone(),
                    history: format!("h{i}"),
                    times: (0..horizon).collect(),
                    values: (0..nodes).map(|k| ((i + 2) * (k + 1)) as f64 * 0.125).collect(),
                })
                .collect(),
            worlds: (0..n_worlds)
                .map(|w| GeneralWorldWire {
                    name: format!("w{w}"),
                    times: (0..horizon).collect(),
                    actions: (0..horizon).map(|t| ((w + t as usize) % 2) as f64).collect(),
                    units: units.clone(),
                })
                .collect(),
        }
    }

    #[test]
    fn a_long_horizon_with_many_worlds_seals_and_replays_bit_for_bit() {
        let ctx = ExecutionContext::for_tests(1);
        let sealed = GeneralTemporalCounterfactualArtifact::seal(&request(6, 5), &ctx).unwrap();
        assert_eq!(sealed.result().world_means.len(), 5);
        let bytes = sealed.to_bytes("general").unwrap();
        let consumed = GeneralTemporalCounterfactualArtifact::from_bytes(
            &bytes,
            Some(sealed.identity()),
            &ctx,
        )
        .unwrap();
        assert_eq!(consumed.report(), sealed.report());
        assert_eq!(consumed.identity(), sealed.identity());
        assert_eq!(consumed.meta().horizon, 6);
    }

    #[test]
    fn a_retained_identity_refuses_a_changed_world() {
        let ctx = ExecutionContext::for_tests(1);
        let a = GeneralTemporalCounterfactualArtifact::seal(&request(3, 2), &ctx).unwrap();
        let mut other = request(3, 2);
        other.worlds[1].actions[0] = 7.0;
        let b = GeneralTemporalCounterfactualArtifact::seal(&other, &ctx).unwrap();
        let bytes = b.to_bytes("general").unwrap();
        let error = GeneralTemporalCounterfactualArtifact::from_bytes(
            &bytes,
            Some(a.identity()),
            &ctx,
        )
        .unwrap_err();
        assert!(matches!(error, Error::IdentityMismatch { field: "action_history" }));
    }

    #[test]
    fn a_core_refusal_is_a_typed_refusal() {
        let mut bad = request(2, 2);
        bad.worlds[0].units.pop();
        let error = GeneralTemporalCounterfactualArtifact::seal(&bad, &ExecutionContext::for_tests(1))
            .unwrap_err();
        let refusal = error.refusal().expect("a structured refusal");
        assert_eq!(refusal.detail, "temporal_counterfactual.unpaired_histories");
    }
}
