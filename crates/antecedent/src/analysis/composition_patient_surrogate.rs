//! Verify retained patient evidence before a bundle exposes its compatible bounds.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_core::ExecutionContext;

use super::{
    ArtifactDescription, BundleStage, FACT_COORDINATES, FACT_LAW, FACT_REQUEST, NodeFailure,
    NodeKind, NodeVerifier, UpstreamNode, VerifiedNode,
};
use crate::analysis::kernel_programs::patient_surrogate::{
    PatientSurrogateError, PatientSurrogateProgramError, PatientSurrogateReading,
    StoredPatientSurrogate, consume_stored,
};

fn failed(reason: &(impl ToString + ?Sized)) -> NodeFailure {
    NodeFailure { stage: BundleStage::TamperedQuantity, reason: reason.to_string() }
}

pub(super) fn describe(
    bytes: &[u8],
    ctx: &ExecutionContext,
) -> Result<ArtifactDescription, NodeFailure> {
    let _lease = ctx.admit_scratch(1, (bytes.len() as u64).saturating_mul(4)).map_err(|error| {
        NodeFailure {
            stage: if ctx.cancellation.is_cancelled() {
                BundleStage::ExecutionCancelled
            } else {
                BundleStage::Oversized
            },
            reason: error.to_string(),
        }
    })?;
    let stored = StoredPatientSurrogate::decode(bytes).map_err(|error| failed(&error))?;
    let program = stored.program;
    let result = consume_stored(&stored, ctx).map_err(|error| {
        let mut failure = failed(&error);
        if let PatientSurrogateProgramError::Scientific(error) = error {
            failure.stage = match error {
                PatientSurrogateError::Cancelled => BundleStage::ExecutionCancelled,
                PatientSurrogateError::Resource => BundleStage::Oversized,
                _ => failure.stage,
            };
        }
        failure
    })?;
    let [lower, upper] = result.bounds().endpoints();
    let facts = BTreeMap::from([
        (FACT_REQUEST.into(), result.source_identity().to_hex()),
        (
            FACT_LAW.into(),
            match result.reading() {
                PatientSurrogateReading::PopulationIdentification => "identified_set",
                PatientSurrogateReading::EmpiricalPlugin => "empirical_plugin_bounds",
            }
            .into(),
        ),
        (
            FACT_COORDINATES.into(),
            antecedent_design::composition_verifiers::coordinate_id(&result.quantity().into()),
        ),
        ("reading".into(), result.reading().as_str().into()),
        ("program_identity".into(), program.to_hex()),
    ]);
    Ok(ArtifactDescription {
        kind: NodeKind::PatientSurrogateIdentification,
        identity: result.identity().to_hex(),
        facts,
        values: vec![
            ("lower".into(), lower),
            ("upper".into(), upper),
            ("surrogate_improver_probability".into(), result.bounds().improver_probability()),
        ],
    })
}

pub(super) struct PatientSurrogateVerifier(pub(super) ExecutionContext);

impl NodeVerifier for PatientSurrogateVerifier {
    fn kind(&self) -> NodeKind {
        NodeKind::PatientSurrogateIdentification
    }

    fn verify(
        &self,
        _id: &str,
        bytes: &[u8],
        upstream: &[UpstreamNode<'_>],
    ) -> Result<VerifiedNode, NodeFailure> {
        if !upstream.is_empty() {
            return Err(NodeFailure {
                stage: BundleStage::SwappedEvidence,
                reason: "patient surrogate identification retains its own joint endpoint evidence"
                    .into(),
            });
        }
        let value = describe(bytes, &self.0)?;
        Ok(VerifiedNode { identity: value.identity, facts: value.facts, values: value.values })
    }
}
