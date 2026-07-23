//! External prior bank: catalog metadata and compatibility filtering.
//!
//! Metadata wraps existing posterior artifacts — it is not a new draw format.
//! Compatibility never panics on mismatch.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::container::{EncodedArtifact, SectionBytes, section_descriptor};
use crate::convert::{from_cbor, to_cbor};
use crate::error::IoError;
use crate::posterior::{
    CausalPosteriorWire, PosteriorQuantityWire, decode_posterior_artifact,
    decode_posterior_meta_from_path,
};

/// Section id for prior-source metadata attached to a posterior artifact.
pub const PRIOR_SOURCE_META_SECTION: &str = "prior_source.meta";

/// Estimand fingerprint for catalog matching (query kind + treatment/outcome names).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct EstimandFingerprint {
    /// Query kind tag (e.g. `"ate"`, `"average_effect"`).
    pub query_kind: String,
    /// Treatment variable name.
    pub treatment: String,
    /// Outcome variable name.
    pub outcome: String,
}

impl EstimandFingerprint {
    /// Construct from owned string-likes.
    #[must_use]
    pub fn new(
        query_kind: impl Into<String>,
        treatment: impl Into<String>,
        outcome: impl Into<String>,
    ) -> Self {
        Self { query_kind: query_kind.into(), treatment: treatment.into(), outcome: outcome.into() }
    }
}

/// Role of a design variable in a prior-source summary.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum DesignVariableRole {
    /// Treatment.
    Treatment,
    /// Outcome.
    Outcome,
    /// Covariate / adjustment.
    Covariate,
    /// Other / uncategorized.
    Other,
}

/// One variable in a design schema summary.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct DesignVariableSummary {
    /// Variable name.
    pub name: String,
    /// Role in the source design.
    pub role: DesignVariableRole,
}

impl DesignVariableSummary {
    /// Construct a summary row.
    #[must_use]
    pub fn new(name: impl Into<String>, role: DesignVariableRole) -> Self {
        Self { name: name.into(), role }
    }
}

/// Declared bridge from a banked source into a target design (hydrate is P4B).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum PriorMapping {
    /// Identical coefficient subspace (P1-C sequential Bayes).
    IdenticalCoefficientSubspace,
    /// Effect-functional transfer via a named source quantity.
    EffectFunctional {
        /// Source effect / quantity name (e.g. `"ate"`).
        source_quantity: String,
    },
    /// Explicit source→target quantity name pairs.
    NamedParameters {
        /// `(source_name, target_name)` pairs.
        pairs: Vec<(String, String)>,
    },
}

/// Metadata for one prior-bank source (CBOR section or sidecar).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PriorSourceMeta {
    /// Artifact id (matches container manifest when attached).
    pub artifact_id: String,
    /// Estimand at fit time.
    pub estimand: EstimandFingerprint,
    /// Caller-chosen tags (product / context / population are conventions).
    #[serde(default)]
    pub tags: BTreeMap<String, String>,
    /// Design schema summary (variable names + roles).
    #[serde(default)]
    pub design: Vec<DesignVariableSummary>,
    /// Identification status string at fit time (same tags as posterior wire).
    pub identification: String,
    /// Optional contrast-coding summary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contrast: Option<String>,
    /// Optional free-form provenance map.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub provenance: BTreeMap<String, String>,
    /// Optional declared mapping for heterogeneous designs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub declared_mapping: Option<PriorMapping>,
}

impl PriorSourceMeta {
    /// Builder with required fields; tags / design / optional fields default empty.
    #[must_use]
    pub fn new(
        artifact_id: impl Into<String>,
        estimand: EstimandFingerprint,
        identification: impl Into<String>,
    ) -> Self {
        Self {
            artifact_id: artifact_id.into(),
            estimand,
            tags: BTreeMap::new(),
            design: Vec::new(),
            identification: identification.into(),
            contrast: None,
            provenance: BTreeMap::new(),
            declared_mapping: None,
        }
    }

    /// Attach caller tags.
    #[must_use]
    pub fn with_tags(mut self, tags: BTreeMap<String, String>) -> Self {
        self.tags = tags;
        self
    }

    /// Attach design summary.
    #[must_use]
    pub fn with_design(mut self, design: Vec<DesignVariableSummary>) -> Self {
        self.design = design;
        self
    }

    /// Attach declared mapping.
    #[must_use]
    pub fn with_mapping(mut self, mapping: PriorMapping) -> Self {
        self.declared_mapping = Some(mapping);
        self
    }
}

/// Where the posterior draws live for a catalog entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PriorArtifactBody {
    /// In-memory container bytes.
    Bytes(Arc<[u8]>),
    /// Path to a packed artifact on disk.
    Path(PathBuf),
}

/// One catalog entry: meta + posterior bytes or path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PriorSourceRef {
    /// Source metadata.
    pub meta: PriorSourceMeta,
    /// Posterior artifact body (optional when only meta is known).
    pub artifact: Option<PriorArtifactBody>,
}

impl PriorSourceRef {
    /// Meta-only entry (compatibility uses meta + optional quantity probe).
    #[must_use]
    pub fn from_meta(meta: PriorSourceMeta) -> Self {
        Self { meta, artifact: None }
    }

    /// Entry with in-memory artifact bytes.
    #[must_use]
    pub fn with_bytes(meta: PriorSourceMeta, bytes: impl Into<Arc<[u8]>>) -> Self {
        Self { meta, artifact: Some(PriorArtifactBody::Bytes(bytes.into())) }
    }

    /// Entry with a filesystem path.
    #[must_use]
    pub fn with_path(meta: PriorSourceMeta, path: impl Into<PathBuf>) -> Self {
        Self { meta, artifact: Some(PriorArtifactBody::Path(path.into())) }
    }
}

/// Target design for catalog filtering.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TargetDesign {
    /// Target estimand.
    pub estimand: EstimandFingerprint,
    /// Variable names required to be present on the source design (or mappable).
    pub variables: BTreeSet<String>,
    /// Optional tag filters: source must match all listed key→value pairs.
    pub tags: BTreeMap<String, String>,
    /// When true, unidentified source fits may be accepted (`AllowUnidentifiedAsPrior`).
    pub allow_unidentified: bool,
}

impl TargetDesign {
    /// Construct a target with estimand + required variables.
    #[must_use]
    pub fn new(
        estimand: EstimandFingerprint,
        variables: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            estimand,
            variables: variables.into_iter().map(Into::into).collect(),
            tags: BTreeMap::new(),
            allow_unidentified: false,
        }
    }

    /// Require tag equality filters.
    #[must_use]
    pub fn with_tags(mut self, tags: BTreeMap<String, String>) -> Self {
        self.tags = tags;
        self
    }

    /// Allow unidentified source fits as priors.
    #[must_use]
    pub fn allow_unidentified(mut self) -> Self {
        self.allow_unidentified = true;
        self
    }
}

/// Structured rejection reason (stable for conformance JSON).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case", tag = "code")]
pub enum CompatibilityRejectReason {
    /// Source fit was unidentified and caller did not opt in.
    UnidentifiedSource {
        /// Identification tag from meta.
        identification: String,
    },
    /// Estimand fingerprint mismatch with no declared mapping.
    EstimandMismatch {
        /// Source estimand.
        source: EstimandFingerprint,
        /// Target estimand.
        target: EstimandFingerprint,
    },
    /// Declared mapping references unknown source quantity names.
    MappingNamesUnknown {
        /// Missing source quantity names.
        missing: Vec<String>,
    },
    /// Required treatment/outcome absent from source design.
    MissingCoreVariables {
        /// Missing names.
        missing: Vec<String>,
    },
    /// Tag filter not satisfied.
    TagMismatch {
        /// Tag key that failed.
        key: String,
        /// Expected value.
        expected: String,
        /// Observed value (empty if absent).
        observed: String,
    },
    /// Posterior artifact unreadable when quantity probe was required.
    ArtifactUnreadable {
        /// Error message.
        message: String,
    },
}

/// Result of checking one source against a target design.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case", tag = "status")]
pub enum CompatibilityReport {
    /// Fully compatible for bank use under the declared rules.
    Compatible {
        /// Source artifact id.
        artifact_id: String,
    },
    /// Partially compatible: some gaps, but a bridge may exist.
    Partial {
        /// Source artifact id.
        artifact_id: String,
        /// Missing durable names / variables.
        missing: Vec<String>,
        /// Quantities that remain mappable (e.g. named effects).
        mappable: Vec<String>,
    },
    /// Rejected with a structured reason.
    Rejected {
        /// Source artifact id.
        artifact_id: String,
        /// Why.
        reason: CompatibilityRejectReason,
    },
}

impl CompatibilityReport {
    /// Artifact id for this report.
    #[must_use]
    pub fn artifact_id(&self) -> &str {
        match self {
            Self::Compatible { artifact_id }
            | Self::Partial { artifact_id, .. }
            | Self::Rejected { artifact_id, .. } => artifact_id,
        }
    }

    /// Whether this report is not a hard reject.
    #[must_use]
    pub const fn is_usable(&self) -> bool {
        matches!(self, Self::Compatible { .. } | Self::Partial { .. })
    }
}

/// Catalog of prior sources.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PriorCatalog {
    /// Sources in insertion order.
    pub sources: Vec<PriorSourceRef>,
}

impl PriorCatalog {
    /// Empty catalog.
    #[must_use]
    pub fn new() -> Self {
        Self { sources: Vec::new() }
    }

    /// Build from sources.
    #[must_use]
    pub fn from_sources(sources: Vec<PriorSourceRef>) -> Self {
        Self { sources }
    }

    /// Push a source.
    pub fn push(&mut self, source: PriorSourceRef) {
        self.sources.push(source);
    }

    /// Filter sources for compatibility with `target`.
    ///
    /// Returns one report per source in catalog order. Never panics on mismatch.
    #[must_use]
    pub fn filter_compatible(&self, target: &TargetDesign) -> Vec<CompatibilityReport> {
        self.sources.iter().map(|s| assess_compatibility(s, target)).collect()
    }

    /// Rank previously obtained reports by caller similarity scores.
    ///
    /// Stable sort: higher score first; unknown ids keep relative order at the end
    /// (score treated as −∞). Rejected reports are dropped.
    #[must_use]
    pub fn rank(
        &self,
        reports: &[CompatibilityReport],
        scores: &[(String, f64)],
    ) -> Vec<CompatibilityReport> {
        let score_map: BTreeMap<&str, f64> =
            scores.iter().map(|(id, s)| (id.as_str(), *s)).collect();
        let mut usable: Vec<(CompatibilityReport, f64, usize)> = reports
            .iter()
            .cloned()
            .enumerate()
            .filter(|(_, r)| r.is_usable())
            .map(|(i, r)| {
                let s = score_map.get(r.artifact_id()).copied().unwrap_or(f64::NEG_INFINITY);
                (r, s, i)
            })
            .collect();
        usable.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.2.cmp(&b.2))
        });
        usable.into_iter().map(|(r, _, _)| r).collect()
    }
}

/// Encode [`PriorSourceMeta`] as CBOR bytes.
///
/// # Errors
///
/// CBOR encode failure.
pub fn encode_prior_source_meta(meta: &PriorSourceMeta) -> Result<Vec<u8>, IoError> {
    to_cbor(meta)
}

/// Decode [`PriorSourceMeta`] from CBOR bytes.
///
/// # Errors
///
/// CBOR decode failure.
pub fn decode_prior_source_meta(bytes: &[u8]) -> Result<PriorSourceMeta, IoError> {
    from_cbor(bytes)
}

/// Attach (or replace) a `prior_source.meta` section on an encoded posterior artifact.
///
/// # Errors
///
/// CBOR encode failure, or artifact is not a `CausalPosterior`.
pub fn attach_prior_source_meta(
    mut artifact: EncodedArtifact,
    meta: &PriorSourceMeta,
) -> Result<EncodedArtifact, IoError> {
    if artifact.manifest.artifact_kind != crate::wire::ArtifactKind::CausalPosterior {
        return Err(IoError::Convert(format!(
            "expected CausalPosterior, got {:?}",
            artifact.manifest.artifact_kind
        )));
    }
    let bytes = encode_prior_source_meta(meta)?;
    let desc = section_descriptor(PRIOR_SOURCE_META_SECTION, "application/cbor", &bytes);
    if let Some(pos) = artifact.sections.iter().position(|s| s.id == PRIOR_SOURCE_META_SECTION) {
        artifact.sections[pos] = SectionBytes::new(PRIOR_SOURCE_META_SECTION, bytes);
        if let Some(mpos) =
            artifact.manifest.sections.iter().position(|s| s.id == PRIOR_SOURCE_META_SECTION)
        {
            artifact.manifest.sections[mpos] = desc;
        } else {
            artifact.manifest.sections.push(desc);
        }
    } else {
        artifact.sections.push(SectionBytes::new(PRIOR_SOURCE_META_SECTION, bytes));
        artifact.manifest.sections.push(desc);
    }
    if !meta.artifact_id.is_empty() {
        artifact.manifest.artifact_id.clone_from(&meta.artifact_id);
    }
    Ok(artifact)
}

/// Extract `prior_source.meta` if present.
///
/// # Errors
///
/// CBOR decode failure when the section exists but is malformed.
pub fn extract_prior_source_meta(
    artifact: &EncodedArtifact,
) -> Result<Option<PriorSourceMeta>, IoError> {
    let Some(sec) = artifact.sections.iter().find(|s| s.id == PRIOR_SOURCE_META_SECTION) else {
        return Ok(None);
    };
    Ok(Some(decode_prior_source_meta(&sec.data)?))
}

fn identification_ok_for_prior(identification: &str, allow_unidentified: bool) -> bool {
    if allow_unidentified {
        return true;
    }
    matches!(
        identification,
        "NonparametricallyIdentified"
            | "nonparametrically_identified"
            | "IdentifiedUnderParametricRestrictions"
            | "identified_under_parametric_restrictions"
            | "IdentifiedUnderPriorRestrictions"
            | "identified_under_prior_restrictions"
            | "PartiallyIdentified"
            | "partially_identified"
    )
}

fn estimands_match(a: &EstimandFingerprint, b: &EstimandFingerprint) -> bool {
    a.query_kind == b.query_kind && a.treatment == b.treatment && a.outcome == b.outcome
}

fn design_names(meta: &PriorSourceMeta) -> BTreeSet<&str> {
    meta.design.iter().map(|d| d.name.as_str()).collect()
}

fn load_posterior_wire(source: &PriorSourceRef) -> Result<Option<CausalPosteriorWire>, IoError> {
    let Some(body) = &source.artifact else {
        return Ok(None);
    };
    match body {
        PriorArtifactBody::Bytes(bytes) => {
            let art = EncodedArtifact::read_from(bytes.as_ref())?;
            let (meta, _) = decode_posterior_artifact(&art)?;
            Ok(Some(meta))
        }
        PriorArtifactBody::Path(path) => Ok(Some(decode_posterior_meta_from_path(path)?)),
    }
}

fn quantity_names(wire: &CausalPosteriorWire) -> (Vec<String>, bool, Vec<String>) {
    let mut effects = Vec::new();
    let mut named_coefs = Vec::new();
    let mut has_unnamed_coef = false;
    for q in &wire.quantities {
        match q {
            PosteriorQuantityWire::Coefficient { name, .. } => match name {
                Some(n) if !n.is_empty() => named_coefs.push(n.clone()),
                _ => has_unnamed_coef = true,
            },
            PosteriorQuantityWire::Effect { name } | PosteriorQuantityWire::Scalar { name } => {
                effects.push(name.clone());
            }
            PosteriorQuantityWire::ResidualVariance => {}
        }
    }
    let mut all = named_coefs;
    all.extend(effects.iter().cloned());
    (all, has_unnamed_coef, effects)
}

fn mapping_source_names(mapping: &PriorMapping) -> Vec<String> {
    match mapping {
        PriorMapping::IdenticalCoefficientSubspace => Vec::new(),
        PriorMapping::EffectFunctional { source_quantity } => {
            vec![source_quantity.clone()]
        }
        PriorMapping::NamedParameters { pairs } => pairs.iter().map(|(s, _)| s.clone()).collect(),
    }
}

fn check_tag_filters(
    source: &PriorSourceMeta,
    target: &TargetDesign,
    artifact_id: &str,
) -> Option<CompatibilityReport> {
    for (k, expected) in &target.tags {
        match source.tags.get(k) {
            Some(obs) if obs == expected => {}
            Some(obs) => {
                return Some(CompatibilityReport::Rejected {
                    artifact_id: artifact_id.into(),
                    reason: CompatibilityRejectReason::TagMismatch {
                        key: k.clone(),
                        expected: expected.clone(),
                        observed: obs.clone(),
                    },
                });
            }
            None => {
                return Some(CompatibilityReport::Rejected {
                    artifact_id: artifact_id.into(),
                    reason: CompatibilityRejectReason::TagMismatch {
                        key: k.clone(),
                        expected: expected.clone(),
                        observed: String::new(),
                    },
                });
            }
        }
    }
    None
}

fn missing_variables(
    source: &PriorSourceMeta,
    target: &TargetDesign,
) -> (Vec<String>, Vec<String>) {
    let source_vars = design_names(source);
    let core = [target.estimand.treatment.as_str(), target.estimand.outcome.as_str()];
    let missing_core: Vec<String> =
        core.iter().filter(|n| !source_vars.contains(**n)).map(|s| (*s).to_string()).collect();
    let mut missing: Vec<String> =
        target.variables.iter().filter(|v| !source_vars.contains(v.as_str())).cloned().collect();
    missing.extend(missing_core.iter().cloned());
    missing.sort();
    missing.dedup();
    (missing_core, missing)
}

/// `None` = mapping ok; `Some(report)` = early Partial or Rejected.
fn validate_declared_mapping(
    mapping: &PriorMapping,
    wire: Option<&CausalPosteriorWire>,
    qty_set: &BTreeSet<&str>,
    effect_names: &[String],
    artifact_id: &str,
) -> Option<CompatibilityReport> {
    let needed = mapping_source_names(mapping);
    if needed.is_empty() {
        return None;
    }
    if wire.is_none() {
        let mut mappable = effect_names.to_vec();
        if mappable.is_empty() {
            mappable.clone_from(&needed);
        }
        return Some(CompatibilityReport::Partial {
            artifact_id: artifact_id.into(),
            missing: needed,
            mappable,
        });
    }
    let unknown: Vec<String> =
        needed.into_iter().filter(|n| !qty_set.contains(n.as_str())).collect();
    if unknown.is_empty() {
        None
    } else {
        Some(CompatibilityReport::Rejected {
            artifact_id: artifact_id.into(),
            reason: CompatibilityRejectReason::MappingNamesUnknown { missing: unknown },
        })
    }
}

fn assess_compatibility(source: &PriorSourceRef, target: &TargetDesign) -> CompatibilityReport {
    let id = source.meta.artifact_id.clone();

    if let Some(reject) = check_tag_filters(&source.meta, target, &id) {
        return reject;
    }

    if !identification_ok_for_prior(source.meta.identification.as_str(), target.allow_unidentified)
    {
        return CompatibilityReport::Rejected {
            artifact_id: id,
            reason: CompatibilityRejectReason::UnidentifiedSource {
                identification: source.meta.identification.clone(),
            },
        };
    }

    let estimand_ok = estimands_match(&source.meta.estimand, &target.estimand);
    if !estimand_ok && source.meta.declared_mapping.is_none() {
        return CompatibilityReport::Rejected {
            artifact_id: id,
            reason: CompatibilityRejectReason::EstimandMismatch {
                source: source.meta.estimand.clone(),
                target: target.estimand.clone(),
            },
        };
    }

    let (missing_core, mut missing) = missing_variables(&source.meta, target);
    if missing_core.len() == 2 {
        let effect_only =
            matches!(source.meta.declared_mapping, Some(PriorMapping::EffectFunctional { .. }));
        if !effect_only {
            return CompatibilityReport::Rejected {
                artifact_id: id,
                reason: CompatibilityRejectReason::MissingCoreVariables { missing: missing_core },
            };
        }
    }

    let wire = match load_posterior_wire(source) {
        Ok(w) => w,
        Err(e) => {
            if source
                .meta
                .declared_mapping
                .as_ref()
                .is_some_and(|m| !matches!(m, PriorMapping::IdenticalCoefficientSubspace))
            {
                return CompatibilityReport::Rejected {
                    artifact_id: id,
                    reason: CompatibilityRejectReason::ArtifactUnreadable {
                        message: e.to_string(),
                    },
                };
            }
            None
        }
    };

    let (qty_names, has_unnamed_coef, effect_names) = match &wire {
        Some(w) => quantity_names(w),
        None => (Vec::new(), true, Vec::new()),
    };
    let qty_set: BTreeSet<&str> = qty_names.iter().map(String::as_str).collect();

    if let Some(mapping) = &source.meta.declared_mapping {
        if let Some(early) =
            validate_declared_mapping(mapping, wire.as_ref(), &qty_set, &effect_names, &id)
        {
            return early;
        }
    }

    let mut mappable = effect_names;
    if has_unnamed_coef && estimand_ok {
        if !missing.iter().any(|m| m == "durable_coef_names") {
            missing.push("durable_coef_names".into());
        }
        if missing.len() == 1 && missing[0] == "durable_coef_names" {
            return CompatibilityReport::Partial {
                artifact_id: id,
                missing: vec!["durable_coef_names".into()],
                mappable,
            };
        }
        return CompatibilityReport::Partial { artifact_id: id, missing, mappable };
    }

    if !missing.is_empty() {
        if mappable.is_empty() {
            if let Some(PriorMapping::EffectFunctional { source_quantity }) =
                &source.meta.declared_mapping
            {
                mappable.push(source_quantity.clone());
            }
        }
        return CompatibilityReport::Partial { artifact_id: id, missing, mappable };
    }

    CompatibilityReport::Compatible { artifact_id: id }
}

/// Whether a posterior wire has at least one named effect quantity.
#[must_use]
pub fn posterior_has_named_effect(wire: &CausalPosteriorWire) -> bool {
    wire.quantities
        .iter()
        .any(|q| matches!(q, PosteriorQuantityWire::Effect { name } if !name.is_empty()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::posterior::{PosteriorQuantityWire, encode_posterior_artifact};

    fn ate_estimand() -> EstimandFingerprint {
        EstimandFingerprint::new("ate", "t", "y")
    }

    fn design_tyz() -> Vec<DesignVariableSummary> {
        vec![
            DesignVariableSummary::new("t", DesignVariableRole::Treatment),
            DesignVariableSummary::new("y", DesignVariableRole::Outcome),
            DesignVariableSummary::new("z", DesignVariableRole::Covariate),
        ]
    }

    fn mini_posterior(artifact_id: &str, coef_names: Option<Vec<&str>>) -> EncodedArtifact {
        let mut quantities = Vec::new();
        if let Some(names) = coef_names {
            for (i, n) in names.into_iter().enumerate() {
                quantities.push(PosteriorQuantityWire::Coefficient {
                    index: u32::try_from(i).expect("coef index fits u32"),
                    name: Some(n.into()),
                });
            }
        } else {
            for i in 0..2u32 {
                quantities.push(PosteriorQuantityWire::Coefficient { index: i, name: None });
            }
        }
        quantities.push(PosteriorQuantityWire::Effect { name: "ate".into() });
        let n_q = quantities.len();
        let meta = CausalPosteriorWire {
            quantities,
            n_draws: 2,
            mean: vec![0.0; n_q],
            sd: vec![1.0; n_q],
            q025: vec![-1.0; n_q],
            q975: vec![1.0; n_q],
            identification: "NonparametricallyIdentified".into(),
            unidentified_mass: 0.0,
            backend_id: "laplace".into(),
            converged: true,
            hessian_condition: 1.0,
            draws_encoding: "f64_le_colmajor".into(),
        };
        let draws = vec![0.0f64; n_q * 2];
        encode_posterior_artifact(&meta, &draws, artifact_id, "0.1.0").unwrap()
    }

    #[test]
    fn prior_source_meta_cbor_round_trip() {
        let mut tags = BTreeMap::new();
        tags.insert("product".into(), "widget".into());
        let meta = PriorSourceMeta::new("src-a", ate_estimand(), "NonparametricallyIdentified")
            .with_tags(tags)
            .with_design(design_tyz())
            .with_mapping(PriorMapping::EffectFunctional { source_quantity: "ate".into() });
        let bytes = encode_prior_source_meta(&meta).unwrap();
        let back = decode_prior_source_meta(&bytes).unwrap();
        assert_eq!(back, meta);
    }

    #[test]
    fn attach_and_extract_prior_source_meta_section() {
        let art = mini_posterior("bank-1", Some(vec!["intercept", "coef_t"]));
        let meta = PriorSourceMeta::new("bank-1", ate_estimand(), "NonparametricallyIdentified")
            .with_design(design_tyz());
        let attached = attach_prior_source_meta(art, &meta).unwrap();
        let got = extract_prior_source_meta(&attached).unwrap().expect("meta section");
        assert_eq!(got.artifact_id, "bank-1");
        assert_eq!(got.design.len(), 3);
    }

    #[test]
    fn filter_matching_named_coefs_compatible() {
        let art = mini_posterior("match", Some(vec!["intercept", "coef_t", "coef_z"]));
        let mut buf = Vec::new();
        art.write_to(&mut buf).unwrap();
        let meta = PriorSourceMeta::new("match", ate_estimand(), "NonparametricallyIdentified")
            .with_design(design_tyz());
        let catalog = PriorCatalog::from_sources(vec![PriorSourceRef::with_bytes(meta, buf)]);
        let target = TargetDesign::new(ate_estimand(), ["t", "y", "z"]);
        let reports = catalog.filter_compatible(&target);
        assert!(matches!(
            &reports[0],
            CompatibilityReport::Compatible { artifact_id } if artifact_id == "match"
        ));
    }

    #[test]
    fn filter_wrong_estimand_rejected() {
        let meta = PriorSourceMeta::new(
            "wrong",
            EstimandFingerprint::new("ate", "t", "other_y"),
            "NonparametricallyIdentified",
        )
        .with_design(design_tyz());
        let catalog = PriorCatalog::from_sources(vec![PriorSourceRef::from_meta(meta)]);
        let target = TargetDesign::new(ate_estimand(), ["t", "y"]);
        let reports = catalog.filter_compatible(&target);
        assert!(matches!(
            &reports[0],
            CompatibilityReport::Rejected {
                reason: CompatibilityRejectReason::EstimandMismatch { .. },
                ..
            }
        ));
    }

    #[test]
    fn filter_unnamed_coefs_partial() {
        let art = mini_posterior("unnamed", None);
        let mut buf = Vec::new();
        art.write_to(&mut buf).unwrap();
        let meta = PriorSourceMeta::new("unnamed", ate_estimand(), "NonparametricallyIdentified")
            .with_design(design_tyz());
        let catalog = PriorCatalog::from_sources(vec![PriorSourceRef::with_bytes(meta, buf)]);
        let target = TargetDesign::new(ate_estimand(), ["t", "y", "z"]);
        let reports = catalog.filter_compatible(&target);
        match &reports[0] {
            CompatibilityReport::Partial { artifact_id, missing, mappable } => {
                assert_eq!(artifact_id, "unnamed");
                assert!(missing.iter().any(|m| m == "durable_coef_names"));
                assert!(mappable.iter().any(|m| m == "ate"));
            }
            other => panic!("expected Partial, got {other:?}"),
        }
    }

    #[test]
    fn filter_unidentified_rejected_unless_allowed() {
        let meta =
            PriorSourceMeta::new("uid", ate_estimand(), "NotIdentified").with_design(design_tyz());
        let catalog = PriorCatalog::from_sources(vec![PriorSourceRef::from_meta(meta)]);
        let target = TargetDesign::new(ate_estimand(), ["t", "y"]);
        let reports = catalog.filter_compatible(&target);
        assert!(matches!(
            &reports[0],
            CompatibilityReport::Rejected {
                reason: CompatibilityRejectReason::UnidentifiedSource { .. },
                ..
            }
        ));
        let allowed = target.allow_unidentified();
        let reports2 = catalog.filter_compatible(&allowed);
        assert!(reports2[0].is_usable());
    }

    #[test]
    fn rank_orders_by_score() {
        let reports = vec![
            CompatibilityReport::Compatible { artifact_id: "a".into() },
            CompatibilityReport::Partial {
                artifact_id: "b".into(),
                missing: vec![],
                mappable: vec!["ate".into()],
            },
            CompatibilityReport::Rejected {
                artifact_id: "c".into(),
                reason: CompatibilityRejectReason::EstimandMismatch {
                    source: ate_estimand(),
                    target: ate_estimand(),
                },
            },
        ];
        let catalog = PriorCatalog::new();
        let ranked = catalog.rank(&reports, &[("b".into(), 0.9), ("a".into(), 0.2)]);
        assert_eq!(ranked.len(), 2);
        assert_eq!(ranked[0].artifact_id(), "b");
        assert_eq!(ranked[1].artifact_id(), "a");
    }

    #[test]
    fn filter_table_driven_matrix() {
        let cases: Vec<(PriorSourceMeta, TargetDesign, &str)> = vec![
            (
                PriorSourceMeta::new("ok", ate_estimand(), "NonparametricallyIdentified")
                    .with_design(design_tyz())
                    .with_mapping(PriorMapping::IdenticalCoefficientSubspace),
                TargetDesign::new(ate_estimand(), ["t", "y"]),
                "compatible_or_partial",
            ),
            (
                PriorSourceMeta::new("bad_tags", ate_estimand(), "NonparametricallyIdentified")
                    .with_design(design_tyz()),
                TargetDesign::new(ate_estimand(), ["t", "y"])
                    .with_tags(BTreeMap::from([("product".into(), "gadget".into())])),
                "tag_reject",
            ),
        ];
        for (meta, target, kind) in cases {
            let catalog = PriorCatalog::from_sources(vec![PriorSourceRef::from_meta(meta)]);
            let r = &catalog.filter_compatible(&target)[0];
            match kind {
                "compatible_or_partial" => assert!(r.is_usable(), "{r:?}"),
                "tag_reject" => assert!(matches!(
                    r,
                    CompatibilityReport::Rejected {
                        reason: CompatibilityRejectReason::TagMismatch { .. },
                        ..
                    }
                )),
                _ => unreachable!(),
            }
        }
    }
}
