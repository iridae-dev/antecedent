//! Retained cohort means at the bundle boundary.

use antecedent_core::{ExecutionContext, reason_details::cohort_decision as details};
use antecedent_io_core::cohort_decision::{
    CohortDecisionSource, CohortIntakeLimits, CohortRefusal,
};

use super::{
    FACT_COORDINATES, FACT_EVIDENCE_DIGEST, FACT_LAW, FACT_MEAN_SOURCE, FACT_SNAPSHOT, FACT_TRUST,
    Inspected, LAW_MEAN_ONLY, TRUST_ATTESTED, TRUST_NATIVE, coordinate_id, failure, publish,
    push_value,
};
use crate::cohort_decision_input::cohort_mean_binding;
use crate::composition_bundle::{
    BundleStage, NodeFailure, NodeKind, NodeVerifier, UpstreamNode, VerifiedNode,
};

fn cohort_failure(error: &CohortRefusal) -> NodeFailure {
    let stage = match error.detail {
        details::OVERSIZED | details::COHORT_TOO_LARGE => BundleStage::Oversized,
        details::UNSUPPORTED_VERSION => BundleStage::IncompatibleVersion,
        details::CANCELLED => BundleStage::ExecutionCancelled,
        details::REPLAY_MISMATCH => BundleStage::TamperedQuantity,
        _ => BundleStage::SwappedEvidence,
    };
    failure(stage, error.to_string())
}

pub(super) fn inspect(bytes: &[u8], ctx: &ExecutionContext) -> Result<Inspected, NodeFailure> {
    let _lease = ctx.acquire_serial_numerical_admission(bytes.len() as u64).map_err(|error| {
        let stage = if ctx.cancellation.is_cancelled() {
            BundleStage::ExecutionCancelled
        } else {
            BundleStage::ResourceRefused
        };
        failure(stage, error.to_string())
    })?;
    let source =
        CohortDecisionSource::inspect_for_bundle(bytes, ctx, &CohortIntakeLimits::default())
            .map_err(|error| cohort_failure(&error))?;
    let model = source.model();
    let cohort = source.cohort();
    let mut inspected = Inspected { identity: source.identity().into(), ..Inspected::default() };
    let facts = &mut inspected.facts;
    publish(facts, FACT_COORDINATES, coordinate_id(&source.quantity().into()));
    publish(facts, FACT_LAW, LAW_MEAN_ONLY);
    publish(facts, FACT_MEAN_SOURCE, cohort_mean_binding(&source));
    publish(facts, FACT_SNAPSHOT, format!("{}|{}", cohort.digest(), model.parent().claim_id));
    publish(facts, FACT_EVIDENCE_DIGEST, model.identity());
    publish(facts, FACT_TRUST, if source.is_native() { TRUST_NATIVE } else { TRUST_ATTESTED });
    publish(facts, "cohort_digest", cohort.digest());
    publish(facts, "model_identity", model.identity());
    publish(facts, "population_id", cohort.population_id());
    publish(facts, "outcome_units", source.outcome_units());
    publish(facts, "support", source.support_status().as_str());
    publish(facts, "inference_scope", "point_only");
    publish(facts, "assumptions", source.assumptions().join("\n"));
    push_value(&mut inspected.values, "mean_contrast".into(), source.mean_contrast());
    Ok(inspected)
}

/// The cohort source verifier, using the caller's cancellation and admission pool.
pub struct CohortDecisionSourceVerifier(ExecutionContext);

impl CohortDecisionSourceVerifier {
    /// Verify retained cohort sources under this context.
    #[must_use]
    pub fn new(ctx: &ExecutionContext) -> Self {
        Self(ctx.clone())
    }
}

impl NodeVerifier for CohortDecisionSourceVerifier {
    fn kind(&self) -> NodeKind {
        NodeKind::CohortDecisionSource
    }

    fn verify(
        &self,
        _node_id: &str,
        bytes: &[u8],
        _upstream: &[UpstreamNode<'_>],
    ) -> Result<VerifiedNode, NodeFailure> {
        let inspected = inspect(bytes, &self.0)?;
        Ok(VerifiedNode {
            identity: inspected.identity,
            facts: inspected.facts,
            values: inspected.values,
        })
    }
}
