//! A validated ledger parent and an independently bound selected point pool.

use antecedent_core::ExecutionContext;
use antecedent_io_core::evidence_ledger_artifact::{
    EvidenceLedgerArtifact, EvidenceLedgerArtifactError,
};

use super::{
    FACT_COORDINATES, FACT_EVIDENCE_DIGEST, FACT_LAW, FACT_MEAN_SOURCE, FACT_SNAPSHOT, FACT_TRUST,
    Inspected, LAW_MEAN_ONLY, TRUST_ATTESTED, coordinate_id, failure, publish, push_value,
};
use crate::composition_bundle::{
    BundleStage, NodeFailure, NodeKind, NodeVerifier, UpstreamNode, VerifiedNode,
};
use crate::ledger_pool_selection::{LedgerPoolSelection, live_pool_binding};

fn ledger_failure(error: &EvidenceLedgerArtifactError) -> NodeFailure {
    let stage = match error {
        EvidenceLedgerArtifactError::UnsupportedVersion { .. } => BundleStage::IncompatibleVersion,
        EvidenceLedgerArtifactError::LimitsExceeded(_) => BundleStage::Oversized,
        EvidenceLedgerArtifactError::SynthesisMismatch(_) => BundleStage::TamperedQuantity,
        _ => BundleStage::SwappedEvidence,
    };
    failure(stage, error.to_string())
}

/// Consume a ledger using the normal artifact reader and the caller's admission budget.
///
/// # Errors
/// Invalid artifact, failure to reproduce its ordered evidence history, changed expected
/// identity, or the caller's size, admission or cancellation refusal.
#[doc(hidden)]
pub fn consume_ledger_for_bundle(
    bytes: &[u8],
    expected: Option<&str>,
    ctx: &ExecutionContext,
) -> Result<EvidenceLedgerArtifact, NodeFailure> {
    let _lease = ctx.acquire_serial_numerical_admission(bytes.len() as u64).map_err(|error| {
        let stage = if ctx.cancellation.is_cancelled() {
            BundleStage::ExecutionCancelled
        } else {
            BundleStage::ResourceRefused
        };
        failure(stage, error.to_string())
    })?;
    let ledger = EvidenceLedgerArtifact::from_bytes(bytes, expected)
        .map_err(|error| ledger_failure(&error))?;
    if ctx.cancellation.is_cancelled() {
        return Err(failure(BundleStage::ExecutionCancelled, "ledger replay was cancelled"));
    }
    Ok(ledger)
}

fn binding_key(result: &str) -> String {
    format!("ledger_pool.{result}")
}

pub(super) fn inspect_ledger(
    bytes: &[u8],
    ctx: &ExecutionContext,
) -> Result<Inspected, NodeFailure> {
    let artifact = consume_ledger_for_bundle(bytes, None, ctx)?;
    let ledger = artifact.ledger();
    let mut inspected = Inspected { identity: artifact.identity().into(), ..Inspected::default() };
    let facts = &mut inspected.facts;
    publish(facts, "ledger_head", ledger.head().to_hex());
    publish(facts, "ledger_state", ledger.state_digest().to_hex());
    for synthesis in artifact.syntheses() {
        if let Ok(binding) = live_pool_binding(&artifact, synthesis.result) {
            publish(facts, &binding_key(&synthesis.result.to_hex()), binding);
        }
    }
    Ok(inspected)
}

fn describe_selected(selection: &LedgerPoolSelection) -> Inspected {
    let mut inspected = Inspected { identity: selection.identity(), ..Inspected::default() };
    let facts = &mut inspected.facts;
    let mean = selection.mean_source();
    publish(facts, FACT_COORDINATES, coordinate_id(&selection.quantity().into()));
    publish(facts, FACT_LAW, LAW_MEAN_ONLY);
    publish(facts, FACT_MEAN_SOURCE, selection.identity());
    publish(facts, FACT_EVIDENCE_DIGEST, selection.evidence_binding());
    publish(facts, FACT_SNAPSHOT, format!("{}|{}", mean.snapshot_id, mean.causal_contract_id));
    publish(facts, FACT_TRUST, TRUST_ATTESTED);
    publish(facts, "ledger_result", selection.result_id());
    publish(facts, "pool_operation", selection.operation_id());
    publish(facts, "pool_method", selection.method());
    publish(facts, "inference_scope", "point_only");
    push_value(&mut inspected.values, "pooled_estimate".into(), selection.estimate());
    inspected
}

pub(super) fn inspect_selected(bytes: &[u8]) -> Result<Inspected, NodeFailure> {
    LedgerPoolSelection::inspect(bytes).map(|selection| describe_selected(&selection))
}

fn verified(inspected: Inspected) -> VerifiedNode {
    VerifiedNode { identity: inspected.identity, facts: inspected.facts, values: inspected.values }
}

/// A replayable full ledger under the caller's execution controls.
pub struct EvidenceLedgerVerifier(ExecutionContext);

impl EvidenceLedgerVerifier {
    /// Build a ledger node verifier under the caller's resource limits.
    #[must_use]
    pub fn new(ctx: &ExecutionContext) -> Self {
        Self(ctx.clone())
    }
}

impl NodeVerifier for EvidenceLedgerVerifier {
    fn kind(&self) -> NodeKind {
        NodeKind::EvidenceLedger
    }

    fn verify(
        &self,
        _node_id: &str,
        bytes: &[u8],
        _upstream: &[UpstreamNode<'_>],
    ) -> Result<VerifiedNode, NodeFailure> {
        inspect_ledger(bytes, &self.0).map(verified)
    }
}

/// A pool selection that depends on a live, verified ledger parent.
pub(super) struct LedgerPooledResultVerifier;

impl NodeVerifier for LedgerPooledResultVerifier {
    fn kind(&self) -> NodeKind {
        NodeKind::LedgerPooledResult
    }

    fn verify(
        &self,
        _node_id: &str,
        bytes: &[u8],
        upstream: &[UpstreamNode<'_>],
    ) -> Result<VerifiedNode, NodeFailure> {
        let selection = LedgerPoolSelection::inspect(bytes)?;
        let mut ledgers = upstream.iter().filter(|node| node.kind == NodeKind::EvidenceLedger);
        let ledger = ledgers.next().filter(|_| ledgers.next().is_none()).ok_or_else(|| {
            failure(
                BundleStage::CallbackUnavailable,
                "the pool selection requires one replayed ledger parent",
            )
        })?;
        let expected = selection.evidence_binding();
        let binding = ledger.facts.get(&binding_key(selection.result_id()));
        if binding != Some(&expected) {
            return Err(failure(
                BundleStage::SwappedEvidence,
                "the pool was revised, withdrawn, or differs from the active ledger result",
            ));
        }
        Ok(verified(describe_selected(&selection)))
    }
}
