//! Original diagnostic evidence shared with native decision/inverse artifacts.
use antecedent_core::{ExecutionContext, ResponseCoordinateLabels, reason_code};
pub use antecedent_design::source_evidence::*;
use std::sync::Arc;

/// A read-only receipt issued only after actual original checked estimation and full comparison.
/// It restores no executable state, provider authority or calibration license.
#[derive(Clone, Debug)]
pub struct SourceResolution {
    source_digest: String,
    dependencies: Vec<Arc<str>>,
}
impl SourceResolution {
    /// Execute the supplied original checked preparation and compare the complete producing artifact.
    /// # Errors
    /// Unsupported source dependencies, original estimation/encoding refusal, or changed scientific evidence.
    pub fn execute(
        evidence: &SourceEvidence,
        prepared: &crate::PreparedStudy,
        ctx: &ExecutionContext,
    ) -> Result<Self, SourceEvidenceError> {
        let refuse =
            |detail| SourceEvidenceError { code: reason_code!("invalid_argument"), detail };
        if !evidence.is_native_source() {
            return Err(refuse("source_evidence.resolution_scope_unsupported"));
        }
        if evidence
            .unresolved()
            .iter()
            .any(|dependency| dependency.as_ref() != "dependencies.checked_response_grid_operation")
        {
            return Err(refuse("source_evidence.resolution_scope_unsupported"));
        }
        let super::builder::DataInput::Tabular(data) = &prepared.study().data else {
            return Err(refuse("source_evidence.resolution_scope_unsupported"));
        };
        let result = prepared
            .estimate(data, ctx)
            .map_err(|_| refuse("source_evidence.source_execution_failed"))?;
        let first = evidence
            .coordinates()
            .first()
            .ok_or_else(|| refuse("source_evidence.coordinates_unavailable"))?;
        let labels = ResponseCoordinateLabels {
            outcome_units: &first.units,
            population_id: &first.population_id,
            transform_id: &first.transform_id,
        };
        let bytes = prepared
            .encode_contracted_result_with_quantity_labels(
                &result,
                evidence.artifact_id(),
                ctx,
                Some(&labels),
            )
            .map_err(|_| refuse("source_evidence.source_encoding_failed"))?;
        evidence.require_same_source(&SourceEvidence::from_result(&bytes)?)?;
        Ok(Self {
            source_digest: evidence.summary()["source_artifact_digest"]
                .as_str()
                .ok_or_else(|| refuse("source_evidence.invalid_source_artifact"))?
                .to_owned(),
            dependencies: evidence.unresolved().to_vec(),
        })
    }
    /// Exact independently compared original source artifact digest.
    #[must_use]
    pub fn source_artifact_digest(&self) -> &str {
        &self.source_digest
    }
    /// Only the original checked-operation dependencies resolved by execution.
    #[must_use]
    pub fn resolved_dependencies(&self) -> &[Arc<str>] {
        &self.dependencies
    }
}
