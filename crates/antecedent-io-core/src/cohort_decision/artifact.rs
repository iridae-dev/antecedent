//! Serialization and independent replay of cohort decision artifacts.

use super::{
    COHORT_DECISION_ARTIFACT_KIND, COHORT_DECISION_SECTION, COHORT_DECISION_VERSION, CohortBinding,
    CohortDecisionSource, CohortExpectation, CohortIntakeLimits, CohortOptions, CohortRefusal,
    ContrastDeclaration, CovariateShiftDeclaration, MAX_COHORT_DECISION_ARTIFACT_BYTES, ModelKind,
    POINT_SCOPE, ParentIdentity, RetainedEffectModel, SupportReceipt, invalid, io_refusal,
    semantics_mismatch,
};
use crate::compact_export::{CompactExport, ExportLimits};
use crate::container::{
    ArtifactManifest, CompressPolicy, EncodedArtifact, SectionBytes, section_descriptor_with_policy,
};
use crate::convert::{from_cbor, to_cbor};
use crate::quantity_wire::ScientificQuantityWire;
use crate::reader::ArtifactReader;
use crate::wire::{ArtifactKind, ProvenanceWire, SemanticVersion};
use antecedent_core::ExecutionContext;
use antecedent_learn::FittedEffect;
use serde::{Deserialize, Serialize};

// Artifact
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
#[allow(
    clippy::large_enum_variant,
    reason = "wire shape: boxing would change the serialized form and the public variant shape"
)]
pub(super) enum ModelWire {
    Portable {
        effect: FittedEffect,
        features: Vec<String>,
        population_id: String,
        contrast: ContrastDeclaration,
        parent: ParentIdentity,
    },
    Compact {
        export: Vec<u8>,
        identity: String,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SourceBody {
    pub(super) version: u16,
    pub(super) model: ModelWire,
    model_identity: String,
    pub(super) cohort: CohortBinding,
    cohort_digest: String,
    pub(super) outcome_units: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) shift: Option<CovariateShiftDeclaration>,
    pub(super) quantity: ScientificQuantityWire,
    pub(super) mean_contrast: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) standard_error: Option<f64>,
    support: SupportReceipt,
    pub(super) scope: String,
    identity: String,
}

impl CohortDecisionSource {
    fn to_body(&self) -> SourceBody {
        let model = match &self.model.kind {
            ModelKind::Portable { effect, features } => ModelWire::Portable {
                effect: effect.clone(),
                features: features.clone(),
                population_id: self.model.population_id.clone(),
                contrast: self.model.contrast.clone(),
                parent: self.model.parent.clone(),
            },
            ModelKind::Compact { export, bytes } => {
                ModelWire::Compact { export: bytes.clone(), identity: export.identity().to_owned() }
            }
        };
        SourceBody {
            version: COHORT_DECISION_VERSION,
            model,
            model_identity: self.model.identity.clone(),
            cohort: self.cohort.clone(),
            cohort_digest: self.cohort_digest.clone(),
            outcome_units: self.outcome_units.clone(),
            shift: self.shift.clone(),
            quantity: ScientificQuantityWire::from(&self.quantity),
            mean_contrast: self.mean_contrast,
            standard_error: self.standard_error,
            support: self.support.clone(),
            scope: POINT_SCOPE.to_owned(),
            identity: self.identity.clone(),
        }
    }

    /// Serialize through the checksummed sectioned container.
    ///
    /// # Errors
    /// A blank artifact id, an oversized payload or an encoding failure.
    pub fn to_bytes(&self, artifact_id: &str) -> Result<Vec<u8>, CohortRefusal> {
        self.to_bytes_within(artifact_id, MAX_COHORT_DECISION_ARTIFACT_BYTES)
    }

    /// [`Self::to_bytes`] under the caller's byte bound instead of
    /// [`MAX_COHORT_DECISION_ARTIFACT_BYTES`].
    ///
    /// # Errors
    /// As [`Self::to_bytes`]; a payload over `max_bytes` is `cohort_decision.oversized`.
    pub fn to_bytes_within(
        &self,
        artifact_id: &str,
        max_bytes: usize,
    ) -> Result<Vec<u8>, CohortRefusal> {
        if artifact_id.trim().is_empty() {
            return Err(invalid(
                antecedent_core::reason_details::cohort_decision::MISSING_ARTIFACT_ID,
                "artifact_id",
                "name the artifact",
            ));
        }
        let payload = to_cbor(&self.to_body()).map_err(|e| io_refusal(&e))?;
        let descriptor = section_descriptor_with_policy(
            COHORT_DECISION_SECTION,
            "application/cbor",
            &payload,
            CompressPolicy::Never,
        );
        let library_version = SemanticVersion::from_crate_version(env!("CARGO_PKG_VERSION"))
            .map_err(|e| io_refusal(&e))?;
        let encoded = EncodedArtifact {
            manifest: ArtifactManifest {
                format_version: crate::migrate::STABLE_FORMAT,
                minimum_reader_version: crate::migrate::STABLE_FORMAT,
                artifact_kind: ArtifactKind::Other(COHORT_DECISION_ARTIFACT_KIND.into()),
                library_version,
                artifact_id: artifact_id.into(),
                sections: vec![descriptor],
                provenance: ProvenanceWire { note: "cohort_decision_source".into() },
            },
            sections: vec![SectionBytes::new(COHORT_DECISION_SECTION, payload)],
        };
        let mut bytes = Vec::new();
        encoded.write_to(&mut bytes).map_err(|e| io_refusal(&e))?;
        if bytes.len() > max_bytes {
            return Err(invalid(
                antecedent_core::reason_details::cohort_decision::OVERSIZED,
                bytes.len().to_string(),
                "bind a smaller cohort",
            ));
        }
        Ok(bytes)
    }

    /// Independent consumer: decode with bounded sizes, rebuild the model and the cohort
    /// from the stored fields, recompute support, every prediction, the mean and the
    /// diagnostic, and require the retained expectations and identity.
    ///
    /// # Errors
    /// An oversized or malformed artifact, an unknown version, a changed field (even when
    /// the container was resealed), and any retained expectation that differs.
    pub fn consume(bytes: &[u8], expectation: &CohortExpectation) -> Result<Self, CohortRefusal> {
        Self::consume_with(bytes, expectation, &ExecutionContext::production_default(0))
    }

    /// As [`Self::consume`] under the caller's intake bounds instead of
    /// [`MAX_COHORT_DECISION_ARTIFACT_BYTES`] and [`super::MAX_COHORT_ROWS`].
    ///
    /// # Errors
    /// As [`Self::consume`].
    pub fn consume_within(
        bytes: &[u8],
        expectation: &CohortExpectation,
        limits: &CohortIntakeLimits,
    ) -> Result<Self, CohortRefusal> {
        Self::consume_with_limits(
            bytes,
            expectation,
            &ExecutionContext::production_default(0),
            limits,
        )
    }

    /// As [`Self::consume`] under a caller's execution context (cancellation, budgets).
    ///
    /// # Errors
    /// As [`Self::consume`].
    pub fn consume_with(
        bytes: &[u8],
        expectation: &CohortExpectation,
        ctx: &ExecutionContext,
    ) -> Result<Self, CohortRefusal> {
        Self::consume_with_limits(bytes, expectation, ctx, &CohortIntakeLimits::default())
    }

    /// As [`Self::consume_with`] under the caller's intake bounds.
    ///
    /// # Errors
    /// As [`Self::consume`].
    pub fn consume_with_limits(
        bytes: &[u8],
        expectation: &CohortExpectation,
        ctx: &ExecutionContext,
        limits: &CohortIntakeLimits,
    ) -> Result<Self, CohortRefusal> {
        let body = decode_body(bytes, limits.max_bytes)?;
        let model = rebuild_model(&body)?;
        let options = CohortOptions { shift: body.shift.clone(), expected_contrast_regime: None };
        let recomputed =
            Self::build_within(&model, &body.cohort, &body.outcome_units, &options, ctx, limits)?;
        check_expectation(&recomputed, expectation)?;
        if recomputed.to_body() != body
            || recomputed.mean_contrast.to_bits() != body.mean_contrast.to_bits()
        {
            return Err(invalid(
                antecedent_core::reason_details::cohort_decision::REPLAY_MISMATCH,
                "body",
                "the stored value, support or quantity differs from its recomputation; \
                 re-export from the producer",
            ));
        }
        if recomputed.identity != expectation.identity {
            return Err(invalid(
                antecedent_core::reason_details::cohort_decision::IDENTITY_UNEXPECTED,
                expectation.identity.clone(),
                "the artifact is not the source whose identity was retained",
            ));
        }
        Ok(recomputed)
    }
}

fn decode_body(bytes: &[u8], max_bytes: usize) -> Result<SourceBody, CohortRefusal> {
    if bytes.len() > max_bytes {
        return Err(invalid(
            antecedent_core::reason_details::cohort_decision::OVERSIZED,
            bytes.len().to_string(),
            "bind a smaller cohort",
        ));
    }
    let mut reader =
        ArtifactReader::open_seek(std::io::Cursor::new(bytes)).map_err(|e| io_refusal(&e))?;
    let (layout_ok, declared) = {
        let manifest = reader.manifest();
        let layout_ok = manifest.artifact_kind
            == ArtifactKind::Other(COHORT_DECISION_ARTIFACT_KIND.into())
            && manifest.sections.len() == 1
            && manifest.sections[0].id == COHORT_DECISION_SECTION;
        let declared = manifest
            .sections
            .iter()
            .try_fold(0u64, |total, section| total.checked_add(section.uncompressed_size));
        (layout_ok, declared)
    };
    if !layout_ok {
        return Err(invalid(
            antecedent_core::reason_details::cohort_decision::LAYOUT_INVALID,
            "manifest",
            "supply a cohort decision source",
        ));
    }
    if crate::error::declared_exceeds_usize_limit(declared, max_bytes) {
        return Err(invalid(
            antecedent_core::reason_details::cohort_decision::OVERSIZED,
            "declared_uncompressed_size",
            "bind a smaller cohort",
        ));
    }
    let section = reader.load_section(COHORT_DECISION_SECTION).map_err(|e| io_refusal(&e))?;
    if section.as_bytes().len() > max_bytes {
        return Err(invalid(
            antecedent_core::reason_details::cohort_decision::OVERSIZED,
            "section",
            "bind a smaller cohort",
        ));
    }
    let body: SourceBody = from_cbor(section.as_bytes()).map_err(|e| io_refusal(&e))?;
    if body.version != COHORT_DECISION_VERSION {
        return Err(invalid(
            antecedent_core::reason_details::cohort_decision::UNSUPPORTED_VERSION,
            body.version.to_string(),
            "supply an artifact written by this library version",
        ));
    }
    Ok(body)
}

fn rebuild_model(body: &SourceBody) -> Result<RetainedEffectModel, CohortRefusal> {
    let model = match &body.model {
        ModelWire::Portable { effect, features, population_id, contrast, parent } => {
            RetainedEffectModel::from_portable_parts(
                effect.clone(),
                features.clone(),
                population_id.clone(),
                contrast.clone(),
                parent.clone(),
            )?
        }
        ModelWire::Compact { export: original_bytes, identity } => {
            let export = CompactExport::consume(original_bytes, &ExportLimits::default(), identity)
                .map_err(|refusal| {
                    invalid(
                        antecedent_core::reason_details::cohort_decision::FITTED_MODEL_INVALID,
                        format!("{}: {}", refusal.detail, refusal.subject),
                        "re-export the model from the producer",
                    )
                })?;
            let mut model = RetainedEffectModel::from_compact_export(&export)?;
            // Replaying a verified model must retain its original container. Re-exporting
            // it with this reader's producer version changes nested bytes even though the
            // scientific model identity, predictions and support are identical.
            model.kind = ModelKind::Compact { export, bytes: original_bytes.clone() };
            model
        }
    };
    if model.identity != body.model_identity {
        return Err(invalid(
            antecedent_core::reason_details::cohort_decision::MODEL_IDENTITY_MISMATCH,
            body.model_identity.clone(),
            "the stored model identity differs from the model's recomputed identity",
        ));
    }
    Ok(model)
}

fn check_expectation(
    source: &CohortDecisionSource,
    expectation: &CohortExpectation,
) -> Result<(), CohortRefusal> {
    if expectation.parent_claim.as_deref().is_some_and(|c| c != source.model.parent.claim_id) {
        return Err(invalid(
            antecedent_core::reason_details::cohort_decision::WRONG_PARENT_IDENTITY,
            source.model.parent.claim_id.clone(),
            "the artifact was issued from a different parent claim than the one retained",
        ));
    }
    if expectation.population_id.as_deref().is_some_and(|p| p != source.cohort.population_id) {
        return Err(semantics_mismatch(
            antecedent_core::reason_details::cohort_decision::POPULATION_MISMATCH,
            source.cohort.population_id.clone(),
            "the cohort population differs from the retained population",
        ));
    }
    if expectation.outcome_units.as_deref().is_some_and(|u| u != source.outcome_units) {
        return Err(semantics_mismatch(
            antecedent_core::reason_details::cohort_decision::UNITS_MISMATCH,
            source.outcome_units.clone(),
            "the declared outcome units differ from the retained units; none are converted",
        ));
    }
    if expectation.contrast_regime.as_deref().is_some_and(|c| c != source.model.contrast.regime_id)
    {
        return Err(semantics_mismatch(
            antecedent_core::reason_details::cohort_decision::CONTRAST_MISMATCH,
            source.model.contrast.regime_id.clone(),
            "the model estimates a different contrast than the one retained",
        ));
    }
    if expectation.cohort_digest.as_deref().is_some_and(|d| d != source.cohort_digest) {
        return Err(invalid(
            antecedent_core::reason_details::cohort_decision::COHORT_CHANGED,
            source.cohort_digest.clone(),
            "the rows, weights or identities differ from the retained cohort snapshot",
        ));
    }
    Ok(())
}
