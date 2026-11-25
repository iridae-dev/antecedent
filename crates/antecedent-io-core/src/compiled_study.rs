//! Format 0.6 compiled study (`compiled_study_v1`).
//!
//! The container is the existing CAUSAL frame: magic `CAUSAL\0\0`, a canonical
//! CBOR manifest, length-prefixed sections, BLAKE3 over the on-wire bytes, and
//! zstd for a logical section at or above 4 KiB. Format version is 0.6. The
//! sections are:
//!
//! * `program.invariant` — the writer output, stored byte for byte;
//! * `program.catalog` — one row per role: digest, length, and inline or external placement;
//! * `program.claim` — the sealed analysis result, when the study carries one;
//! * `program.body.<role>` — inline bytes for that role.
//!
//! An external row keeps the caller’s URI and has no body section. Opening
//! accepts bytes the caller’s resolver supplies for that URI. The expected
//! program identity is the retained digest of the invariant and is never read
//! out of the container. [`open_expecting`] binds that digest before [`open`]
//! reads the study; a digest of the container bytes does not bind. Each resolved
//! body must have the catalog’s length and
//! BLAKE3 digest. Source rows must also rebuild to the content digest the
//! catalog declares.
//!
//! When every required body matches, opening re-quotes layer 5 from the source
//! shape and recomputes the source-column mean. A required body the resolver
//! does not supply returns the sealed claim, loaded through
//! [`crate::load_analysis_result`], with resumability false. A mismatched
//! expectation or a body that fails its digest is refused.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_core::route_memory::RouteMemoryProfile;
use serde::{Deserialize, Serialize};

use crate::catalog_role::{self, RETAINED_SUMMARY, SOURCE_ROWS};
use crate::container::{ArtifactManifest, EncodedArtifact, SectionBytes, section_descriptor};
use crate::convert::{from_cbor, to_cbor};
use crate::error::IoError;
use crate::retained_artifact::artifact_identity;
use crate::source_witness::{SourceWitnessLimits, SourceWitnessWire};
use crate::wire::{ArtifactKind, FormatVersion, ProvenanceWire, SemanticVersion};

/// Artifact kind of a compiled study.
pub const KIND: &str = crate::artifact_kinds::COMPILED_STUDY;
/// Format of [`KIND`]. It is not migrated to 0.5.
pub const FORMAT: FormatVersion = FormatVersion { major: 0, minor: 6 };
/// Catalog schema version.
pub const CATALOG_VERSION: u16 = 1;
/// Exact writer output.
pub const INVARIANT_SECTION: &str = "program.invariant";
/// Role, digest, length, and placement.
pub const CATALOG_SECTION: &str = "program.catalog";
/// Sealed analysis-result claim, when the study carries one.
pub const CLAIM_SECTION: &str = "program.claim";

const ARTIFACT_ID: &str = "compiled-study";

/// Section id of an inline body.
#[must_use]
pub fn body_section(role: &str) -> String {
    format!("program.body.{role}")
}

/// Where a role’s bytes live.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Placement {
    /// Bytes are a `program.body.<role>` section.
    Inline,
    /// The caller’s URI. Opening takes the bytes from the resolver.
    External {
        /// Opaque URI, stored as the caller wrote it.
        uri: String,
    },
}

/// One continuation or producer payload.
#[derive(Clone, Debug)]
pub struct StudyBody {
    /// Catalog role. See [`crate::catalog_role`].
    pub role: String,
    /// Payload bytes. For [`catalog_role::SOURCE_ROWS`], a [`SourceWitnessWire`].
    pub bytes: Vec<u8>,
    /// `None` stores the bytes inline.
    pub external_uri: Option<String>,
    /// An unresolved required body blocks resume.
    pub required: bool,
    /// Summary-only retention. Refit stays refused.
    pub summary_only: bool,
    /// Variable names for [`catalog_role::SOURCE_ROWS`], in column order.
    pub names: Vec<String>,
}

/// What [`write`] stores.
pub struct StudyParts<'a> {
    /// Exact writer output. Opening returns these bytes unchanged.
    pub invariant: &'a [u8],
    /// Catalog roles.
    pub bodies: &'a [StudyBody],
    /// Analysis-result container, loaded as the sealed claim.
    pub claim: Option<&'a [u8]>,
}

/// Bytes of a compiled study and the program identity the caller retains apart from them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WrittenStudy {
    /// CAUSAL container.
    pub bytes: Vec<u8>,
    /// Digest of the invariant. Pass it to [`open`]; it is not taken from the container.
    pub expected_identity: String,
}

/// A required role the resolver did not supply.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnresolvedBody {
    /// Catalog role.
    pub role: String,
    /// External URI, when the row is not inline.
    pub uri: Option<String>,
}

/// A compiled study after acceptance.
#[derive(Clone, Debug)]
pub struct OpenedStudy {
    /// Every required body matched and the study is not summary-only.
    pub resumable: bool,
    /// Required roles with no bytes.
    pub unresolved: Vec<UnresolvedBody>,
    /// Layer 5 quote of the source shape, when every required body matched.
    pub layer5_bytes: Option<u64>,
    /// Mean of the first source column, recomputed from the resolved rows.
    pub estimate: Option<f64>,
    /// Sealed claim, when `program.claim` is present.
    pub claim: Option<crate::AnalysisResultLoad>,
    /// `Some` when a retained summary refuses refit.
    pub refit_refused: Option<String>,
    /// A callback role is in the catalog, so a live call needs the caller object.
    pub callback_requires_caller: bool,
    /// The caller supplied that object.
    pub callback_live: bool,
    /// `program.invariant`, unchanged.
    pub invariant: Vec<u8>,
    /// Resolved role bytes.
    pub bodies: BTreeMap<String, Vec<u8>>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "placement", rename_all = "snake_case", deny_unknown_fields)]
enum PlacementWire {
    Inline,
    External { uri: String },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct EntryWire {
    role: String,
    digest: String,
    length: u64,
    placement: PlacementWire,
    required: bool,
    summary_only: bool,
    content_digest: Option<String>,
    rows: Option<u64>,
    columns: Option<u64>,
    names: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct CatalogWire {
    version: u16,
    entries: Vec<EntryWire>,
    estimate_bits: Option<u64>,
}

fn refused(message: impl Into<String>) -> IoError {
    IoError::Refused {
        code: antecedent_core::reason_code!("invalid_argument"),
        message: message.into(),
    }
}

/// Layer 5 byte quote of a source with `rows` and `columns`.
///
/// # Errors
///
/// The shape is empty or the quote does not fit in `u64`.
pub fn layer5_quote(rows: u64, columns: u64) -> Result<u64, IoError> {
    if rows == 0 || columns == 0 {
        return Err(refused("layer 5 quote needs a source shape"));
    }
    RouteMemoryProfile::design(rows, columns)
        .estimate()
        .map(|estimate| estimate.total())
        .map_err(|error| refused(format!("layer 5 quote: {error}")))
}

/// Content digest of source-row bytes, which must be a [`SourceWitnessWire`].
///
/// # Errors
///
/// The bytes are not a witness, or they do not rebuild under `names`.
pub fn source_content_digest(bytes: &[u8], names: &[String]) -> Result<[u8; 32], IoError> {
    let witness: SourceWitnessWire = from_cbor(bytes)?;
    let storage = witness.to_storage(names, &SourceWitnessLimits::default())?;
    Ok(storage.content_digest())
}

/// Accept source bytes only when they rebuild to `declared`.
///
/// # Errors
///
/// The witness does not rebuild, or its content digest is not `declared`.
pub fn accept_source_body(
    bytes: &[u8],
    names: &[String],
    declared: &[u8; 32],
) -> Result<[u8; 32], IoError> {
    let got = source_content_digest(bytes, names)?;
    if got != *declared {
        return Err(refused("source content digest does not match the catalog"));
    }
    Ok(got)
}

fn source_mean(bytes: &[u8]) -> Result<(f64, u64, u64), IoError> {
    let witness: SourceWitnessWire = from_cbor(bytes)?;
    let Some(column) = witness.columns.first() else {
        return Err(refused("source rows have no column"));
    };
    if column.values.is_empty() {
        return Err(refused("source rows have no values"));
    }
    let sum: f64 = column.values.iter().copied().sum();
    let mean = sum / column.values.len() as f64;
    let columns = u64::try_from(witness.columns.len()).map_err(|_| IoError::TooLarge)?;
    Ok((mean, witness.row_count, columns))
}

fn section(id: &str, bytes: &[u8]) -> (crate::wire::SectionDescriptor, SectionBytes) {
    let descriptor = section_descriptor(id, "application/octet-stream", bytes);
    let stored = SectionBytes::new(id, bytes.to_vec());
    (descriptor, stored)
}

/// Write a compiled study.
///
/// # Errors
///
/// An empty invariant, an unknown or repeated role, a source that does not
/// rebuild, or a container encoding failure.
pub fn write(parts: StudyParts<'_>) -> Result<WrittenStudy, IoError> {
    if parts.invariant.is_empty() {
        return Err(refused("a compiled study needs its writer output"));
    }
    let mut entries = Vec::with_capacity(parts.bodies.len());
    let mut inline: Vec<(String, Vec<u8>)> = Vec::new();
    let mut seen = BTreeMap::<String, ()>::new();
    let mut estimate_bits = None;
    for body in parts.bodies {
        if !catalog_role::is_role(&body.role) {
            return Err(refused(format!("unknown catalog role `{}`", body.role)));
        }
        if seen.insert(body.role.clone(), ()).is_some() {
            return Err(refused(format!("catalog role `{}` is repeated", body.role)));
        }
        if body.summary_only && body.role != RETAINED_SUMMARY {
            return Err(refused("summary-only retention is the retained summary"));
        }
        let placement = match &body.external_uri {
            None => PlacementWire::Inline,
            Some(uri) if uri.is_empty() => {
                return Err(refused("an external placement needs its URI"));
            }
            Some(uri) => PlacementWire::External { uri: uri.clone() },
        };
        let (content_digest, rows, columns) = if body.role == SOURCE_ROWS {
            if body.names.is_empty() {
                return Err(refused("source rows need their variable names"));
            }
            let digest = source_content_digest(&body.bytes, &body.names)?;
            let (mean, row_count, column_count) = source_mean(&body.bytes)?;
            estimate_bits = Some(mean.to_bits());
            (Some(antecedent_core::to_hex(&digest)), Some(row_count), Some(column_count))
        } else if !body.names.is_empty() {
            return Err(refused("variable names belong to source rows"));
        } else {
            (None, None, None)
        };
        if matches!(placement, PlacementWire::Inline) {
            inline.push((body.role.clone(), body.bytes.clone()));
        }
        entries.push(EntryWire {
            role: body.role.clone(),
            digest: antecedent_core::blake3_hex(&body.bytes),
            length: u64::try_from(body.bytes.len()).map_err(|_| IoError::TooLarge)?,
            placement,
            required: body.required,
            summary_only: body.summary_only,
            content_digest,
            rows,
            columns,
            names: body.names.clone(),
        });
    }
    entries.sort_by(|left, right| left.role.cmp(&right.role));
    inline.sort_by(|left, right| left.0.cmp(&right.0));
    let catalog = to_cbor(&CatalogWire { version: CATALOG_VERSION, entries, estimate_bits })?;
    let mut descriptors = Vec::new();
    let mut sections = Vec::new();
    for (id, bytes) in [(INVARIANT_SECTION, parts.invariant), (CATALOG_SECTION, catalog.as_slice())]
    {
        let (descriptor, stored) = section(id, bytes);
        descriptors.push(descriptor);
        sections.push(stored);
    }
    if let Some(claim) = parts.claim {
        let (descriptor, stored) = section(CLAIM_SECTION, claim);
        descriptors.push(descriptor);
        sections.push(stored);
    }
    for (role, bytes) in &inline {
        let (descriptor, stored) = section(&body_section(role), bytes);
        descriptors.push(descriptor);
        sections.push(stored);
    }
    let artifact = EncodedArtifact {
        manifest: ArtifactManifest {
            format_version: FORMAT,
            minimum_reader_version: FORMAT,
            artifact_kind: ArtifactKind::Other(KIND.into()),
            library_version: SemanticVersion::from_crate_version(env!("CARGO_PKG_VERSION"))?,
            artifact_id: ARTIFACT_ID.into(),
            sections: descriptors,
            provenance: ProvenanceWire { note: KIND.into() },
        },
        sections,
    };
    let mut bytes = Vec::new();
    artifact.write_to(&mut bytes)?;
    Ok(WrittenStudy { bytes, expected_identity: artifact_identity(parts.invariant).to_hex() })
}

fn section_data<'a>(artifact: &'a EncodedArtifact, id: &str) -> Option<&'a [u8]> {
    artifact.sections.iter().find(|section| section.id == id).map(|section| section.data.as_ref())
}

fn entry_bytes(
    artifact: &EncodedArtifact,
    entry: &EntryWire,
    resolve: &dyn Fn(&str) -> Option<Vec<u8>>,
) -> Result<Result<Vec<u8>, UnresolvedBody>, IoError> {
    let inline = body_section(&entry.role);
    match &entry.placement {
        PlacementWire::Inline => match section_data(artifact, &inline) {
            Some(bytes) => Ok(Ok(bytes.to_vec())),
            None => Err(refused(format!("inline role `{}` has no body section", entry.role))),
        },
        PlacementWire::External { uri } => {
            if uri.is_empty() {
                return Err(refused("an external placement needs its URI"));
            }
            if artifact.manifest.sections.iter().any(|section| section.id == inline) {
                return Err(refused(format!(
                    "external role `{}` also has an inline body",
                    entry.role
                )));
            }
            match resolve(uri) {
                Some(bytes) => Ok(Ok(bytes)),
                None if entry.required => {
                    Ok(Err(UnresolvedBody { role: entry.role.clone(), uri: Some(uri.clone()) }))
                }
                None => Ok(Err(UnresolvedBody { role: String::new(), uri: None })),
            }
        }
    }
}

fn accept_bytes(entry: &EntryWire, bytes: &[u8]) -> Result<(), IoError> {
    let length = u64::try_from(bytes.len()).map_err(|_| IoError::TooLarge)?;
    if length != entry.length || antecedent_core::blake3_hex(bytes) != entry.digest {
        return Err(refused(format!("corrupt body for role `{}`", entry.role)));
    }
    if entry.role == SOURCE_ROWS {
        let Some(declared) = entry.content_digest.as_deref() else {
            return Err(refused("source rows declare no content digest"));
        };
        let Some(digest) = antecedent_core::digest_from_hex(declared) else {
            return Err(refused("source content digest is not 32 bytes"));
        };
        accept_source_body(bytes, &entry.names, &digest)?;
        let (_, rows, columns) = source_mean(bytes)?;
        if entry.rows != Some(rows) || entry.columns != Some(columns) {
            return Err(refused("source shape does not match the catalog"));
        }
    } else if entry.content_digest.is_some() {
        return Err(refused(format!("role `{}` declares a source content digest", entry.role)));
    }
    Ok(())
}

/// Admit `bytes` only when `retained` is the invariant's artifact identity.
///
/// The container's own digest is a different value. This binds the invariant
/// before [`open`] reads the study. Binding the container digest refuses.
///
/// # Errors
///
/// As [`open`]. A retained identity that is not the invariant is
/// `expected identity does not match the program invariant`.
pub fn open_expecting(
    bytes: &[u8],
    retained: &crate::retained_artifact::RetainedArtifactIdentity,
    resolve: &dyn Fn(&str) -> Option<Vec<u8>>,
    caller_supplied: bool,
) -> Result<OpenedStudy, IoError> {
    let artifact = EncodedArtifact::read_from(bytes)?;
    let invariant = section_data(&artifact, INVARIANT_SECTION)
        .ok_or_else(|| refused("compiled study has no invariant"))?;
    crate::retained_artifact::bind_retained_artifact(invariant, retained)
        .map_err(|_| refused("expected identity does not match the program invariant"))?;
    open(bytes, &retained.to_hex(), resolve, caller_supplied)
}

/// Open a compiled study.
///
/// `expected_identity` is the digest [`write`] returned: the invariant's
/// artifact identity, not the container's. `resolve` supplies bytes for an
/// external URI and returns `None` when it cannot. `caller_supplied` is the
/// live callback object; the study does not store one.
///
/// # Errors
///
/// A container that is not this kind, an expectation that is not the invariant’s
/// digest, or a resolved body whose length, digest, or source content digest fails.
pub fn open(
    bytes: &[u8],
    expected_identity: &str,
    resolve: &dyn Fn(&str) -> Option<Vec<u8>>,
    caller_supplied: bool,
) -> Result<OpenedStudy, IoError> {
    let artifact = EncodedArtifact::read_from(bytes)?;
    if artifact.manifest.artifact_kind != ArtifactKind::Other(KIND.into()) {
        return Err(IoError::Convert("not a compiled study".into()));
    }
    if artifact.manifest.format_version != FORMAT {
        return Err(IoError::UnsupportedFormat {
            major: artifact.manifest.format_version.major,
            minor: artifact.manifest.format_version.minor,
        });
    }
    let minimum = artifact.manifest.minimum_reader_version;
    if minimum.major > FORMAT.major
        || (minimum.major == FORMAT.major && minimum.minor > FORMAT.minor)
    {
        return Err(IoError::UnsupportedFormat { major: minimum.major, minor: minimum.minor });
    }
    let invariant = section_data(&artifact, INVARIANT_SECTION)
        .ok_or_else(|| refused("compiled study has no invariant"))?
        .to_vec();
    // The expected identity is the invariant's artifact digest, not the container's.
    let retained = crate::retained_artifact::RetainedArtifactIdentity::from_hex(expected_identity)
        .map_err(|_| refused("expected identity does not match the program invariant"))?;
    crate::retained_artifact::bind_retained_artifact(&invariant, &retained)
        .map_err(|_| refused("expected identity does not match the program invariant"))?;
    let catalog_bytes = section_data(&artifact, CATALOG_SECTION)
        .ok_or_else(|| refused("compiled study has no catalog"))?;
    let catalog: CatalogWire = from_cbor(catalog_bytes)?;
    if catalog.version != CATALOG_VERSION {
        return Err(IoError::UnsupportedVersion { version: u32::from(catalog.version) });
    }
    let claim = match section_data(&artifact, CLAIM_SECTION) {
        None => None,
        Some(claim) => Some(crate::load_analysis_result(claim)?),
    };
    let mut unresolved = Vec::new();
    let mut bodies = BTreeMap::new();
    let mut summary_only = false;
    let mut callback = false;
    for entry in &catalog.entries {
        if !catalog_role::is_role(&entry.role) {
            return Err(refused(format!("unknown catalog role `{}`", entry.role)));
        }
        if entry.summary_only {
            summary_only = true;
        }
        if entry.role == catalog_role::CALLBACK_DECLARATION
            || entry.role == catalog_role::CALLBACK_OUTPUT
        {
            callback = true;
        }
        match entry_bytes(&artifact, entry, resolve)? {
            Ok(resolved) => {
                accept_bytes(entry, &resolved)?;
                bodies.insert(entry.role.clone(), resolved);
            }
            Err(missing) if missing.role.is_empty() => {}
            Err(missing) => unresolved.push(missing),
        }
    }
    let refit_refused = summary_only.then(|| "summary-only".to_owned());
    let callback_live = callback && caller_supplied;
    if !unresolved.is_empty() {
        return Ok(OpenedStudy {
            resumable: false,
            unresolved,
            layer5_bytes: None,
            estimate: None,
            claim,
            refit_refused,
            callback_requires_caller: callback,
            callback_live,
            invariant,
            bodies,
        });
    }
    let (layer5_bytes, estimate) = match bodies.get(SOURCE_ROWS) {
        None => (None, None),
        Some(source) => {
            let (mean, rows, columns) = source_mean(source)?;
            let quote = layer5_quote(rows, columns)?;
            match catalog.estimate_bits {
                Some(bits) if bits == mean.to_bits() => {}
                _ => return Err(refused("the estimate does not reproduce from the source rows")),
            }
            (Some(quote), Some(mean))
        }
    };
    Ok(OpenedStudy {
        resumable: !summary_only,
        unresolved,
        layer5_bytes,
        estimate,
        claim,
        refit_refused,
        callback_requires_caller: callback,
        callback_live,
        invariant,
        bodies,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog_role::{
        CALLBACK_DECLARATION, CALLBACK_OUTPUT, DISCOVERY_STATISTICS, FROZEN_SCORES, KIND_INVENTORY,
        POSTERIOR_DRAWS, PROVIDER_PAYLOAD, SPILL,
    };
    use crate::policy_program_artifact::{POLICY_PROGRAM_MAGIC, decode_policy_program};
    use crate::provider_envelope::is_provider_envelope;
    use crate::staleness_artifact::{STALENESS_BASELINE_MAGIC, StalenessBaselineWire};
    use antecedent_data::column::{Float64Column, ValidityBitmap};
    use antecedent_data::{OwnedColumn, OwnedColumnarStorage};

    fn none(_: &str) -> Option<Vec<u8>> {
        None
    }

    fn witness(columns: &[&[f64]], names: &[&str]) -> (Vec<u8>, Vec<String>) {
        let rows = columns[0].len();
        let mut built = Vec::new();
        for (index, values) in columns.iter().enumerate() {
            let column = Float64Column::new(
                antecedent_core::VariableId::from_raw(u32::try_from(index).unwrap()),
                std::sync::Arc::<[f64]>::from(*values),
                ValidityBitmap::all_valid(rows),
            )
            .unwrap();
            built.push(OwnedColumn::Float64(column));
        }
        let mut schema = antecedent_core::CausalSchemaBuilder::new();
        for name in names {
            schema
                .add_variable(
                    std::sync::Arc::<str>::from(*name),
                    antecedent_core::ValueType::Continuous,
                    antecedent_core::SmallRoleSet::empty(),
                    None,
                    None,
                    antecedent_core::MeasurementSpec::default(),
                )
                .unwrap();
        }
        let storage =
            OwnedColumnarStorage::try_new(schema.build().unwrap(), built, None, None).unwrap();
        let wire = SourceWitnessWire::from_storage(&storage).unwrap();
        (to_cbor(&wire).unwrap(), names.iter().map(|name| (*name).to_owned()).collect())
    }

    fn claim() -> Vec<u8> {
        let query = serde_json::json!({"response": {
            "functional": {"average_derivative": {"outcome": 1, "treatment": 0, "weighting": "observed"}},
            "target_population": "all_observed", "observation": "complete", "observation_assumptions": []
        }});
        let result: crate::AnalysisResultWire = serde_json::from_value(serde_json::json!({
            "query": query,
            "identification": {"status": "nonparametrically_identified", "query": query,
                "estimands": [], "arena": {"var_sets": [], "interventions": [], "lists": [], "nodes": []},
                "derivation": [], "required_assumptions": [], "diagnostics": [], "candidates_examined": 0, "sets_returned": 0},
            "estimate": 1.0, "standard_error": 0.2, "assumptions": [], "diagnostics": [], "refutations": [],
            "response": null, "posterior_artifact": null, "mediation_grid": null, "structural_response": null
        }))
        .unwrap();
        let artifact =
            crate::encode_analysis_result_artifact(&result, vec!["t".into(), "y".into()], "claim")
                .unwrap();
        let mut bytes = Vec::new();
        artifact.write_to(&mut bytes).unwrap();
        bytes
    }

    fn source_body() -> StudyBody {
        let (bytes, names) = witness(&[&[1.0, 3.0], &[0.0, 2.0]], &["y", "t"]);
        StudyBody {
            role: SOURCE_ROWS.into(),
            bytes,
            external_uri: None,
            required: true,
            summary_only: false,
            names,
        }
    }

    #[test]
    fn matching_bodies_requote_layer_five_and_reproduce_the_estimate() {
        let body = source_body();
        let invariant = b"writer-output-\xff".to_vec();
        let sealed = claim();
        let written = write(StudyParts {
            invariant: &invariant,
            bodies: &[body.clone()],
            claim: Some(&sealed),
        })
        .unwrap();
        assert!(written.bytes.starts_with(b"CAUSAL\0\0"));
        let opened = open(&written.bytes, &written.expected_identity, &none, false).unwrap();
        assert!(opened.resumable);
        assert!(opened.unresolved.is_empty());
        assert_eq!(opened.invariant, invariant);
        assert_eq!(opened.estimate, Some(2.0));
        assert_eq!(opened.layer5_bytes, Some(layer5_quote(2, 2).unwrap()));
        assert_eq!(opened.bodies.get(SOURCE_ROWS), Some(&body.bytes));
        assert_eq!(opened.claim.as_ref().and_then(|claim| claim.body.estimate), Some(1.0));
    }

    #[test]
    fn an_unresolved_required_body_keeps_the_claim_and_is_not_resumable() {
        let mut body = source_body();
        body.external_uri = Some("memory://source".into());
        let invariant = b"invariant".to_vec();
        let sealed = claim();
        let written =
            write(StudyParts { invariant: &invariant, bodies: &[body], claim: Some(&sealed) })
                .unwrap();
        let opened = open(&written.bytes, &written.expected_identity, &none, false).unwrap();
        assert!(!opened.resumable);
        assert_eq!(opened.unresolved.len(), 1);
        assert_eq!(opened.unresolved[0].uri.as_deref(), Some("memory://source"));
        assert!(opened.estimate.is_none());
        assert!(opened.layer5_bytes.is_none());
        assert_eq!(opened.claim.as_ref().and_then(|claim| claim.body.estimate), Some(1.0));
        let resolve = |uri: &str| (uri == "memory://source").then(|| source_body().bytes);
        let resumed = open(&written.bytes, &written.expected_identity, &resolve, false).unwrap();
        assert!(resumed.resumable);
        assert_eq!(resumed.estimate, Some(2.0));
    }

    #[test]
    fn a_mismatched_expectation_or_a_corrupt_body_is_refused() {
        let body = source_body();
        let written =
            write(StudyParts { invariant: b"invariant", bodies: &[body.clone()], claim: None })
                .unwrap();
        let wrong = open(&written.bytes, &"00".repeat(32), &none, false).unwrap_err();
        assert!(wrong.to_string().contains("expected identity"), "{wrong}");
        let container = artifact_identity(&written.bytes).to_hex();
        assert_ne!(container, written.expected_identity);
        let as_container = open(&written.bytes, &container, &none, false).unwrap_err();
        assert!(as_container.to_string().contains("expected identity"), "{as_container}");
        let retained = crate::retained_artifact::RetainedArtifactIdentity::from_hex(
            &written.expected_identity,
        )
        .unwrap();
        let bound = open_expecting(&written.bytes, &retained, &none, false).unwrap();
        assert_eq!(bound.invariant, b"invariant");
        let container_id =
            crate::retained_artifact::RetainedArtifactIdentity::of_bytes(&written.bytes);
        let bound_container =
            open_expecting(&written.bytes, &container_id, &none, false).unwrap_err();
        assert!(bound_container.to_string().contains("expected identity"), "{bound_container}");
        let mut external = body;
        external.external_uri = Some("memory://source".into());
        let written =
            write(StudyParts { invariant: b"invariant", bodies: &[external], claim: None })
                .unwrap();
        let mut corrupt = source_body().bytes;
        let last = corrupt.len() - 1;
        corrupt[last] ^= 0x01;
        let resolve = |_: &str| Some(corrupt.clone());
        let error = open(&written.bytes, &written.expected_identity, &resolve, false).unwrap_err();
        assert!(error.to_string().contains("corrupt body"), "{error}");
    }

    #[test]
    fn a_source_content_digest_mismatch_is_refused() {
        let (bytes, names) = witness(&[&[1.0, 3.0]], &["y"]);
        let digest = source_content_digest(&bytes, &names).unwrap();
        let mut wrong = digest;
        wrong[0] ^= 0x01;
        let error = accept_source_body(&bytes, &names, &wrong).unwrap_err();
        assert!(error.to_string().contains("content digest"), "{error}");
    }

    #[test]
    fn summary_only_refuses_refit_and_a_callback_needs_its_caller() {
        let source = source_body();
        let summary = StudyBody {
            role: RETAINED_SUMMARY.into(),
            bytes: b"summary".to_vec(),
            external_uri: None,
            required: true,
            summary_only: true,
            names: Vec::new(),
        };
        let callback = StudyBody {
            role: CALLBACK_DECLARATION.into(),
            bytes: b"declaration".to_vec(),
            external_uri: None,
            required: true,
            summary_only: false,
            names: Vec::new(),
        };
        let output = StudyBody {
            role: CALLBACK_OUTPUT.into(),
            bytes: b"output-word".to_vec(),
            external_uri: None,
            required: true,
            summary_only: false,
            names: Vec::new(),
        };
        let written = write(StudyParts {
            invariant: b"invariant",
            bodies: &[source, summary, callback, output],
            claim: None,
        })
        .unwrap();
        let opened = open(&written.bytes, &written.expected_identity, &none, false).unwrap();
        assert!(!opened.resumable);
        assert_eq!(opened.refit_refused.as_deref(), Some("summary-only"));
        assert!(opened.callback_requires_caller);
        assert!(!opened.callback_live);
        assert_eq!(opened.estimate, Some(2.0));
        let live = open(&written.bytes, &written.expected_identity, &none, true).unwrap();
        assert!(live.callback_live);
        assert!(!live.resumable);
    }

    #[test]
    fn catalog_entries_decode_in_role_order() {
        let bodies = [
            StudyBody {
                role: SPILL.into(),
                bytes: b"spill-rows".to_vec(),
                external_uri: None,
                required: true,
                summary_only: false,
                names: Vec::new(),
            },
            StudyBody {
                role: FROZEN_SCORES.into(),
                bytes: b"frozen".to_vec(),
                external_uri: None,
                required: true,
                summary_only: false,
                names: Vec::new(),
            },
        ];
        let written =
            write(StudyParts { invariant: b"invariant", bodies: &bodies, claim: None }).unwrap();
        let artifact = EncodedArtifact::read_from(written.bytes.as_slice()).unwrap();
        let catalog: CatalogWire =
            from_cbor(section_data(&artifact, CATALOG_SECTION).unwrap()).unwrap();
        let roles: Vec<&str> = catalog.entries.iter().map(|entry| entry.role.as_str()).collect();
        assert_eq!(roles, vec![FROZEN_SCORES, SPILL]);
    }

    #[test]
    fn moved_continuation_roles_round_trip() {
        let roles = [
            FROZEN_SCORES,
            POSTERIOR_DRAWS,
            DISCOVERY_STATISTICS,
            SPILL,
            PROVIDER_PAYLOAD,
            CALLBACK_OUTPUT,
        ];
        let bodies: Vec<StudyBody> = roles
            .into_iter()
            .map(|role| StudyBody {
                role: role.into(),
                bytes: format!("payload-{role}").into_bytes(),
                external_uri: None,
                required: true,
                summary_only: false,
                names: Vec::new(),
            })
            .collect();
        let written =
            write(StudyParts { invariant: b"invariant", bodies: &bodies, claim: None }).unwrap();
        let opened = open(&written.bytes, &written.expected_identity, &none, true).unwrap();
        assert!(opened.resumable);
        for body in &bodies {
            assert_eq!(opened.bodies.get(&body.role), Some(&body.bytes));
        }
    }

    #[test]
    fn legacy_magics_are_not_a_compiled_study_and_still_identify() {
        assert!(open(b"ANTSPILL rest", "00", &none, false).is_err());
        assert!(crate::external_callback_artifact::ExternalCallbackArtifact::recognizes(
            b"ANTCALLBACK1\x00"
        ));
        assert!(is_provider_envelope(b"ANTEPROV2\0\0\0"));
        let staleness = StalenessBaselineWire::consume(STALENESS_BASELINE_MAGIC).unwrap_err();
        assert!(!staleness.to_string().contains("not a staleness baseline"), "{staleness}");
        assert_ne!(decode_policy_program(POLICY_PROGRAM_MAGIC), Err(IoError::BadMagic));
        assert!(KIND_INVENTORY.iter().any(|row| {
            row.kind == "frozen_scores_v1" && row.continuation == Some(FROZEN_SCORES)
        }));
        assert!(KIND_INVENTORY.iter().any(|row| {
            row.kind == "analysis_result" && row.continuation.is_none() && row.producer.is_none()
        }));
        assert!(
            KIND_INVENTORY
                .iter()
                .any(|row| row.kind == "ANTSPILL" && row.continuation == Some(SPILL))
        );
        assert!(
            KIND_INVENTORY
                .iter()
                .any(|row| { row.kind == "learned_trial" && row.continuation.is_none() })
        );
        let moved: Vec<_> =
            KIND_INVENTORY.iter().filter(|row| row.continuation.is_some()).collect();
        assert!(moved.len() >= 9, "{}", moved.len());
    }
}
