//! A declared point quantity selected from a replayed ledger pool.

use antecedent_contract::ScientificQuantity;
use antecedent_contract::evidence_ledger::{
    ClaimClass, ResultId, SynthesisOperation, parse_digest_hex,
};
use antecedent_core::{CanonicalWriter, SemanticDigest};
use antecedent_io_core::convert::{from_cbor, to_cbor};
use antecedent_io_core::evidence_ledger_artifact::EvidenceLedgerArtifact;
use antecedent_io_core::quantity_wire::ScientificQuantityWire;
use serde::{Deserialize, Serialize};

use crate::composition_bundle::{BundleStage, NodeFailure};
use crate::decision_eval::{MeanEstimate, MeanSource};

const MAGIC: &[u8] = b"ANTE-LEDGER-POOL-SELECTION-1\0";
const MAX_BYTES: usize = 64 * 1024;

fn refusal(stage: BundleStage, why: impl Into<String>) -> NodeFailure {
    NodeFailure { stage, reason: why.into() }
}

/// A single point-estimate projection of the validated synthesis evidence.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PoolEvidence {
    result: String,
    operation: String,
    synthesis_digest: String,
    method: String,
    estimate: f64,
}

impl PoolEvidence {
    fn replay(ledger: &EvidenceLedgerArtifact, result: ResultId) -> Result<Self, NodeFailure> {
        let record = ledger
            .ledger()
            .require_valid(result)
            .map_err(|error| refusal(BundleStage::SwappedEvidence, error.to_string()))?;
        if record.claim_class != ClaimClass::PointOnly
            || !matches!(record.operation, SynthesisOperation::StatisticalPooling { .. })
        {
            return Err(refusal(
                BundleStage::UnsupportedLaw,
                "the selected ledger result must be a point pool",
            ));
        }
        let (method, synthesis_digest, estimate) =
            ledger.retained_synthesis_binding(result).ok_or_else(|| {
                refusal(
                    BundleStage::CallbackUnavailable,
                    "the selected pool needs one retained numerical synthesis",
                )
            })?;
        if !estimate.is_finite() {
            return Err(refusal(
                BundleStage::TamperedQuantity,
                "the selected estimate is nonfinite",
            ));
        }
        Ok(Self {
            result: result.to_hex(),
            operation: record.created_by.to_hex(),
            synthesis_digest: synthesis_digest.into(),
            method: method.into(),
            estimate,
        })
    }

    fn write(&self, canon: &mut CanonicalWriter) {
        canon
            .text(&self.result)
            .text(&self.operation)
            .text(&self.synthesis_digest)
            .text(&self.method)
            .real(self.estimate)
            .text("still_valid");
    }

    fn binding(&self) -> String {
        let mut canon = CanonicalWriter::new("antecedent.ledger_pool_selection.evidence.v1");
        self.write(&mut canon);
        canon.finish_hex()
    }
}

/// A bindable pooled result, eligible for affine point decisions while its evidence stands.
#[derive(Clone, Debug, PartialEq)]
pub struct LedgerPoolSelection {
    evidence: PoolEvidence,
    quantity: ScientificQuantity,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Wire {
    evidence: PoolEvidence,
    quantity: ScientificQuantityWire,
    identity: String,
}

impl LedgerPoolSelection {
    /// Select a stored point pool and declare its scientific quantity.
    ///
    /// # Errors
    /// The result is stale, absent or withdrawn, is an assumption range, or has no
    /// unambiguous numerical synthesis. The quantity must declare a point functional.
    pub fn select(
        ledger: &EvidenceLedgerArtifact,
        result: ResultId,
        quantity: ScientificQuantity,
    ) -> Result<Self, NodeFailure> {
        validate_quantity(&quantity)?;
        Ok(Self { evidence: PoolEvidence::replay(ledger, result)?, quantity })
    }

    /// Exact scalar quantity supplied by the caller.
    #[must_use]
    pub fn quantity(&self) -> &ScientificQuantity {
        &self.quantity
    }

    /// Value retained by the selected synthesis and independently reproduced by the ledger.
    #[must_use]
    pub fn estimate(&self) -> f64 {
        self.evidence.estimate
    }

    /// The selected ledger result id, in hexadecimal.
    #[must_use]
    pub fn result_id(&self) -> &str {
        &self.evidence.result
    }

    /// The operation that registered the pool.
    #[must_use]
    pub fn operation_id(&self) -> &str {
        &self.evidence.operation
    }

    /// The retained pooling procedure.
    #[must_use]
    pub fn method(&self) -> &str {
        &self.evidence.method
    }

    /// Scoped evidence-state identity. Unrelated operations leave it reusable.
    #[must_use]
    pub fn evidence_binding(&self) -> String {
        self.evidence.binding()
    }

    /// Bind the declared quantity and the complete selected pool.
    #[must_use]
    pub fn identity(&self) -> String {
        let mut canon = CanonicalWriter::new("antecedent.ledger_pool_selection.v1");
        self.evidence.write(&mut canon);
        self.quantity.write_canonical(&mut canon);
        canon.finish_hex()
    }

    /// Exact match to the current evidence for the selected result.
    ///
    /// # Errors
    /// The result has been changed or retired, or the numerical synthesis is absent.
    pub fn verify_against(&self, ledger: &EvidenceLedgerArtifact) -> Result<(), NodeFailure> {
        let result = digest(&self.evidence.result)?;
        let replay = PoolEvidence::replay(ledger, ResultId::from_digest(result))?;
        if replay.binding() != self.evidence_binding() {
            return Err(refusal(
                BundleStage::SwappedEvidence,
                "the selected pool differs from the current ledger evidence",
            ));
        }
        Ok(())
    }

    /// Export the selection. An independent consumer must verify its live ledger parent.
    ///
    /// # Errors
    /// Encoding failure or a declaration over the selection bound.
    pub fn to_bytes(&self) -> Result<Vec<u8>, NodeFailure> {
        let mut bytes = MAGIC.to_vec();
        bytes.extend(
            to_cbor(&Wire {
                evidence: self.evidence.clone(),
                quantity: (&self.quantity).into(),
                identity: self.identity(),
            })
            .map_err(|error| refusal(BundleStage::SwappedEvidence, error.to_string()))?,
        );
        if bytes.len() > MAX_BYTES {
            return Err(refusal(BundleStage::Oversized, "the ledger selection is too large"));
        }
        Ok(bytes)
    }

    /// Inspect the bounded declaration and check its internal identity.
    ///
    /// # Errors
    /// Invalid syntax, unsupported version or a changed identity. This semantic check must
    /// be followed by verification against an independently consumed ledger.
    pub fn inspect(bytes: &[u8]) -> Result<Self, NodeFailure> {
        if bytes.len() > MAX_BYTES {
            return Err(refusal(BundleStage::Oversized, "the ledger selection is too large"));
        }
        let body = bytes.strip_prefix(MAGIC).ok_or_else(|| {
            refusal(BundleStage::IncompatibleVersion, "unknown ledger selection version")
        })?;
        let wire: Wire = from_cbor(body)
            .map_err(|error| refusal(BundleStage::SwappedEvidence, error.to_string()))?;
        digest(&wire.evidence.result)?;
        digest(&wire.evidence.operation)?;
        digest(&wire.evidence.synthesis_digest)?;
        if !wire.evidence.estimate.is_finite() || wire.evidence.method.trim().is_empty() {
            return Err(refusal(BundleStage::TamperedQuantity, "invalid pooled estimate"));
        }
        let quantity = wire
            .quantity
            .try_into()
            .map_err(|error| refusal(BundleStage::TamperedQuantity, error))?;
        validate_quantity(&quantity)?;
        let selection = Self { evidence: wire.evidence, quantity };
        if selection.identity() != wire.identity {
            return Err(refusal(BundleStage::SwappedEvidence, "ledger selection identity differs"));
        }
        Ok(selection)
    }

    /// Whether these bytes declare a ledger-pool selection.
    #[must_use]
    pub fn recognizes(bytes: &[u8]) -> bool {
        bytes.starts_with(MAGIC)
    }

    /// Decision representation of the live, declared mean after ledger verification.
    #[must_use]
    pub fn mean_source(&self) -> MeanSource {
        MeanSource {
            coordinates: vec![self.quantity.clone()],
            means: vec![self.estimate()],
            provider_id: "ledger:pooled_point".into(),
            snapshot_id: self.evidence_binding(),
            causal_contract_id: format!("declared:ledger_pool:{}", self.result_id()),
            rng_id: "none:ledger_pool".into(),
            estimate: MeanEstimate::Frequentist,
        }
    }
}

pub(crate) fn live_pool_binding(
    ledger: &EvidenceLedgerArtifact,
    result: ResultId,
) -> Result<String, NodeFailure> {
    PoolEvidence::replay(ledger, result).map(|evidence| evidence.binding())
}

fn digest(id: &str) -> Result<SemanticDigest, NodeFailure> {
    parse_digest_hex(id)
        .ok_or_else(|| refusal(BundleStage::SwappedEvidence, "invalid ledger digest"))
}

fn validate_quantity(quantity: &ScientificQuantity) -> Result<(), NodeFailure> {
    if [
        &quantity.variable_id,
        &quantity.variable_name,
        &quantity.units,
        &quantity.population_id,
        &quantity.regime_id,
        &quantity.functional_id,
        &quantity.transform_id,
    ]
    .iter()
    .any(|field| field.trim().is_empty())
    {
        return Err(refusal(BundleStage::TamperedQuantity, "declare the full pooled quantity"));
    }
    if quantity.functional_id == "outcome" {
        return Err(refusal(
            BundleStage::UnsupportedLaw,
            "a pooled point cannot supply an outcome law",
        ));
    }
    Ok(())
}
