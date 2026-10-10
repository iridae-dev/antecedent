//! X6 per-proposal receipt linking identification repair and design ranking, with
//! re-identification on arrived data.
//!
//! A [`ProposalReceipt`] binds, for one candidate study, by identities and digests
//! and never by copy:
//!
//! * the frozen base failure (family, contract, the unresolved obligation ids, the
//!   repair premises and data digests);
//! * the hypothetical catalog delta (or hypothetical law) and the verified
//!   derivation the repair family produced on it, with the repair classification;
//! * the candidate's declared cost, sample budget and sample size;
//! * the source and snapshot lineage (evidence lineage and provider snapshot of the
//!   signal request, the source distribution digests);
//! * the ranking entry (signal request fingerprint, signal identity, EVSI and net
//!   value bits, rank) and a decision receipt (decision contract identity, utility
//!   unit, action set digest, ranking identity).
//!
//! A [`ProposalBundle`] holds every proposal in the canonical order of semantic id;
//! its identity is a digest over the two artifact digests, the contract identity and
//! the receipt identities in that order, so it never depends on the order the
//! proposals were supplied in. [`verify`] recomputes every receipt from the retained
//! [`RepairReportArtifact`] and [`DesignRankingArtifactWire`] and refuses a candidate
//! that appears in one artifact but not the other, a changed delta, a swapped signal,
//! a changed cost, lineage, value or decision identity, or a receipt whose identity
//! does not follow from its content. `verify` checks the seals of the two artifacts;
//! it does not replay them, so a caller consumes each artifact independently first.
//!
//! The delta and derivation a receipt retains are hypothetical: [`ProposalReceipt`]
//! always reports [`EvidenceState::Hypothetical`]. [`on_arrival`] never treats them
//! as evidence. It refuses a delivery that is itself not available evidence, and
//! otherwise re-runs the repair family's own identification on the arrived evidence
//! alone, answering [`ArrivalVerdict::Verified`], [`ArrivalVerdict::StillInsufficient`]
//! or [`ArrivalVerdict::Invalidated`] (a population or regime other than the
//! premises the derivation was verified under).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;
use std::sync::Arc;

use antecedent_core::{
    DistributionAvailability, EvidenceCatalog, EvidenceKind, EvidenceRegime, ExecutionContext,
    IdentityDomain, RegimeId, RegimeKind, VariableId,
};
use antecedent_io::transport_catalog_wire::{EvidenceCatalogWire, EvidenceRegimeWire};

use crate::design_ranking_artifact::{CandidateWire, DesignRankingArtifactWire};
use crate::repair::{FamilyVerdict, RepairCancelled, RepairFamily};
use crate::repair_artifact::{RepairFamilyRef, RepairOutcomeWire, RepairReportArtifact};
use crate::study_candidate::DurableStudyCandidate;

/// Receipt schema label hashed into every identity.
const RECEIPT_DOMAIN: &str = "antecedent.proposal_receipt.v1";
/// Bundle schema label hashed into every identity.
const BUNDLE_DOMAIN: &str = "antecedent.proposal_bundle.v1";

// ------------------------------------------------------------------- errors

/// A refused receipt, bundle or arrival: a registered reason code and a stable
/// `proposal_receipt.*` detail.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{detail}: {message}")]
pub struct ProposalReceiptError {
    /// Registered runtime-refusal reason code.
    pub code: &'static str,
    /// Stable `proposal_receipt.*` detail.
    pub detail: &'static str,
    /// Human-readable explanation.
    pub message: String,
}

impl ProposalReceiptError {
    /// A binding that does not agree with a retained artifact or with itself.
    fn mismatch(detail: &'static str, message: impl Into<String>) -> Self {
        Self {
            code: antecedent_core::reason_code!("design_signal_invalid"),
            detail,
            message: message.into(),
        }
    }

    /// A malformed request or input.
    fn invalid(detail: &'static str, message: impl Into<String>) -> Self {
        Self {
            code: antecedent_core::reason_code!("invalid_argument"),
            detail,
            message: message.into(),
        }
    }

    /// Hypothetical evidence presented as arrived evidence.
    fn not_evidence(message: impl Into<String>) -> Self {
        Self {
            code: antecedent_core::reason_code!("transport_missing_evidence"),
            detail: "proposal_receipt.hypothetical_not_evidence",
            message: message.into(),
        }
    }

    /// Cancellation observed while re-identifying.
    fn cancelled() -> Self {
        Self {
            code: antecedent_core::reason_code!("transport_budget_cancel"),
            detail: "proposal_receipt.arrival_cancelled",
            message: "the re-identification on arrived evidence was cancelled".to_owned(),
        }
    }
}

type Result<T> = std::result::Result<T, ProposalReceiptError>;

// ----------------------------------------------------------------- hashing

fn put_str(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn put_u64(hasher: &mut blake3::Hasher, value: u64) {
    hasher.update(&value.to_le_bytes());
}

fn put_list(hasher: &mut blake3::Hasher, values: &[String]) {
    put_u64(hasher, values.len() as u64);
    for value in values {
        put_str(hasher, value);
    }
}

fn put_opt_str(hasher: &mut blake3::Hasher, value: Option<&String>) {
    match value {
        Some(text) => {
            hasher.update(&[1]);
            put_str(hasher, text);
        }
        None => {
            hasher.update(&[0]);
        }
    }
}

fn put_opt_u64(hasher: &mut blake3::Hasher, value: Option<u64>) {
    match value {
        Some(number) => {
            hasher.update(&[1]);
            put_u64(hasher, number);
        }
        None => {
            hasher.update(&[0]);
        }
    }
}

fn hex(hasher: &blake3::Hasher) -> String {
    hasher.finalize().to_hex().to_string()
}

fn list_digest(label: &str, values: &[String]) -> String {
    let mut sorted = values.to_vec();
    sorted.sort();
    let mut hasher = blake3::Hasher::new();
    put_str(&mut hasher, label);
    put_list(&mut hasher, &sorted);
    hex(&hasher)
}

fn regimes_digest(regimes: &[EvidenceRegimeWire]) -> Result<String> {
    antecedent_io::identity::digest_wire(IdentityDomain::Identification, &regimes.to_vec())
        .map(|digest| digest.to_hex())
        .map_err(|error| {
            ProposalReceiptError::invalid("proposal_receipt.invalid_delta", error.to_string())
        })
}

// ---------------------------------------------------------------- receipts

/// Whether the evidence a receipt retains exists.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum EvidenceState {
    /// The delta and derivation are a hypothesis about a study not yet delivered.
    Hypothetical,
}

/// The frozen base failure a proposal repairs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BaseFailureBinding {
    /// Repair family (`transport` or `backdoor`).
    pub family: String,
    /// The failed contract's identity.
    pub contract: String,
    /// Unresolved obligation ids, sorted.
    pub obligation_ids: Vec<String>,
    /// Digest of the repair premises.
    pub premises_digest: String,
    /// Digest of the base data lineage.
    pub data_digest: String,
}

/// The hypothetical delta and the derivation checked on it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HypotheticalBinding {
    /// Repair classification of the candidate alone.
    pub classification: String,
    /// Digest of the hypothetical evidence delta.
    pub delta_digest: String,
    /// Number of hypothetical regimes in the delta.
    pub delta_regimes: u64,
    /// Digest of the checker, steps and verification flag of the derivation.
    pub derivation_digest: Option<String>,
    /// Whether the derivation was independently re-verified.
    pub derivation_verified: bool,
    /// Obligation ids the evidence passes the necessary screen for, sorted.
    pub addressed: Vec<String>,
    /// Obligation ids nothing addresses, sorted.
    pub unmet: Vec<String>,
}

/// The candidate's declared cost and size.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CostBinding {
    /// Declared cost units.
    pub units: u64,
    /// Cost unit label.
    pub unit_label: String,
    /// Declared sample budget.
    pub sample_budget: u64,
    /// Planned sample size.
    pub sample_size: u64,
}

/// Source and snapshot lineage of the proposal's valuation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LineageBinding {
    /// Evidence lineage of the signal request and the provider snapshot, sorted.
    pub snapshots: Vec<String>,
    /// Digest of the sorted source distribution digests of the ranking.
    pub source_digest: String,
}

/// The ranking entry the proposal was valued by.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValuationBinding {
    /// `evsi` or `net_value`.
    pub basis: String,
    /// Exact signal request fingerprint.
    pub signal_request_fingerprint: String,
    /// Identity of the signal receipt.
    pub signal_identity: String,
    /// IEEE bits of the candidate's EVSI.
    pub evsi_bits: u64,
    /// IEEE bits of the candidate's net value, when a cost mapping was used.
    pub net_value_bits: Option<u64>,
    /// Rank (0 best).
    pub rank: u64,
}

/// The decision receipt: which decision the value is a value of.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecisionReceipt {
    /// Decision contract identity.
    pub contract_identity: String,
    /// Utility unit.
    pub utility_unit: String,
    /// Digest of the sorted terminal action identities.
    pub action_ids_digest: String,
    /// Identity of the ranking value.
    pub ranking_identity: String,
}

/// One proposal's receipt.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProposalReceipt {
    /// Candidate semantic id.
    pub candidate_id: String,
    /// Frozen base failure.
    pub base_failure: BaseFailureBinding,
    /// Hypothetical delta and derivation.
    pub hypothetical: HypotheticalBinding,
    /// Cost and size.
    pub cost: CostBinding,
    /// Source and snapshot lineage.
    pub lineage: LineageBinding,
    /// Ranking entry.
    pub valuation: ValuationBinding,
    /// Decision receipt.
    pub decision: DecisionReceipt,
    /// Seal digest of the repair artifact the receipt was read from.
    pub repair_report_digest: String,
    /// Digest of the ranking artifact the receipt was read from.
    pub ranking_digest: String,
    /// Identity over every field above.
    pub identity: String,
}

impl ProposalReceipt {
    /// The state of the retained evidence: always hypothetical. A receipt never
    /// holds available evidence; arrival is answered by [`on_arrival`].
    #[must_use]
    pub const fn evidence_state(&self) -> EvidenceState {
        EvidenceState::Hypothetical
    }

    /// Whether the retained delta or derivation is available evidence: never.
    #[must_use]
    pub const fn is_available_evidence(&self) -> bool {
        false
    }

    /// Recompute the identity from the fields.
    #[must_use]
    pub fn compute_identity(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        put_str(&mut hasher, RECEIPT_DOMAIN);
        put_str(&mut hasher, &self.candidate_id);
        let base = &self.base_failure;
        put_str(&mut hasher, &base.family);
        put_str(&mut hasher, &base.contract);
        put_list(&mut hasher, &base.obligation_ids);
        put_str(&mut hasher, &base.premises_digest);
        put_str(&mut hasher, &base.data_digest);
        let hyp = &self.hypothetical;
        put_str(&mut hasher, &hyp.classification);
        put_str(&mut hasher, &hyp.delta_digest);
        put_u64(&mut hasher, hyp.delta_regimes);
        put_opt_str(&mut hasher, hyp.derivation_digest.as_ref());
        hasher.update(&[u8::from(hyp.derivation_verified)]);
        put_list(&mut hasher, &hyp.addressed);
        put_list(&mut hasher, &hyp.unmet);
        let cost = &self.cost;
        put_u64(&mut hasher, cost.units);
        put_str(&mut hasher, &cost.unit_label);
        put_u64(&mut hasher, cost.sample_budget);
        put_u64(&mut hasher, cost.sample_size);
        put_list(&mut hasher, &self.lineage.snapshots);
        put_str(&mut hasher, &self.lineage.source_digest);
        let value = &self.valuation;
        put_str(&mut hasher, &value.basis);
        put_str(&mut hasher, &value.signal_request_fingerprint);
        put_str(&mut hasher, &value.signal_identity);
        put_u64(&mut hasher, value.evsi_bits);
        put_opt_u64(&mut hasher, value.net_value_bits);
        put_u64(&mut hasher, value.rank);
        let decision = &self.decision;
        put_str(&mut hasher, &decision.contract_identity);
        put_str(&mut hasher, &decision.utility_unit);
        put_str(&mut hasher, &decision.action_ids_digest);
        put_str(&mut hasher, &decision.ranking_identity);
        put_str(&mut hasher, &self.repair_report_digest);
        put_str(&mut hasher, &self.ranking_digest);
        hex(&hasher)
    }

    /// Recompute and store the identity (after an edit, to re-seal).
    #[must_use]
    pub fn sealed(mut self) -> Self {
        self.identity = self.compute_identity();
        self
    }
}

// ----------------------------------------------------------------- derive

#[allow(clippy::cast_precision_loss, reason = "a declared cost is a small integer count")]
const fn amount_of(units: u64) -> f64 {
    units as f64
}

fn sorted(values: &[String]) -> Vec<String> {
    let mut out = values.to_vec();
    out.sort();
    out
}

fn singleton_outcome<'a>(
    repair: &'a RepairReportArtifact,
    id: &str,
) -> Result<&'a RepairOutcomeWire> {
    let singles = || {
        repair.report.outcomes.iter().filter(|o| o.candidates.len() == 1 && o.candidates[0] == id)
    };
    if let Some(found) = singles().find(|o| {
        matches!(
            o.classification.as_str(),
            "verified_sufficient" | "insufficient" | "not_certified"
        )
    }) {
        return Ok(found);
    }
    if singles().next().is_some() {
        return Err(ProposalReceiptError::mismatch(
            "proposal_receipt.candidate_not_proposed",
            format!("candidate {id} has no evaluated repair outcome (invalid or unevaluated)"),
        ));
    }
    Err(ProposalReceiptError::mismatch(
        "proposal_receipt.candidate_not_in_repair",
        format!("candidate {id} has no outcome in the repair artifact"),
    ))
}

fn find_scored<'a>(ranking: &'a DesignRankingArtifactWire, id: &str) -> Result<&'a CandidateWire> {
    ranking.candidates.iter().find(|c| c.semantic_id == id).ok_or_else(|| {
        ProposalReceiptError::mismatch(
            "proposal_receipt.candidate_not_in_ranking",
            format!("candidate {id} is not in the ranking artifact"),
        )
    })
}

fn check_declared(repair: &RepairReportArtifact, scored: &CandidateWire, id: &str) -> Result<()> {
    let declared =
        repair.premises.candidates.iter().find(|c| c.semantic_id == id).ok_or_else(|| {
            ProposalReceiptError::mismatch(
                "proposal_receipt.candidate_not_in_repair",
                format!("candidate {id} is not declared in the repair artifact"),
            )
        })?;
    if scored.cost_unit != declared.cost_unit_label
        || scored.cost_amount.to_bits() != amount_of(declared.cost_units).to_bits()
        || scored.sample_size != declared.sample_size
    {
        return Err(ProposalReceiptError::mismatch(
            "proposal_receipt.cost_mismatch",
            format!("candidate {id} is costed or sized differently in the two artifacts"),
        ));
    }
    if scored.request.candidate_id != id {
        return Err(ProposalReceiptError::mismatch(
            "proposal_receipt.signal_mismatch",
            format!("the signal request of candidate {id} names another candidate"),
        ));
    }
    Ok(())
}

fn derivation_digest(outcome: &RepairOutcomeWire) -> Option<String> {
    outcome.derivation.as_ref().map(|d| {
        let mut hasher = blake3::Hasher::new();
        put_str(&mut hasher, "antecedent.proposal_receipt.derivation");
        put_str(&mut hasher, &d.checker);
        put_list(&mut hasher, &d.steps);
        hasher.update(&[u8::from(d.verified)]);
        hex(&hasher)
    })
}

fn snapshots_of(scored: &CandidateWire) -> Vec<String> {
    let mut snapshots = scored.request.evidence_lineage.clone();
    if let Some(provider) = &scored.provider {
        snapshots.push(format!("provider:{}/{}", provider.provider_id, provider.snapshot_id));
    }
    snapshots.sort();
    snapshots
}

/// The receipt the two artifacts imply for `id`.
fn derive_receipt(
    repair: &RepairReportArtifact,
    ranking: &DesignRankingArtifactWire,
    id: &str,
) -> Result<ProposalReceipt> {
    let scored = find_scored(ranking, id)?;
    check_declared(repair, scored, id)?;
    let outcome = singleton_outcome(repair, id)?;
    let delta = outcome.delta.as_ref().ok_or_else(|| {
        ProposalReceiptError::mismatch(
            "proposal_receipt.candidate_not_proposed",
            format!("candidate {id} stores no hypothetical delta"),
        )
    })?;
    let declared = repair.premises.candidates.iter().find(|c| c.semantic_id == id);
    let (units, unit_label, sample_budget) = declared.map_or((0, String::new(), 0), |c| {
        (c.cost_units, c.cost_unit_label.clone(), c.sample_budget)
    });
    let mut obligation_ids: Vec<String> =
        repair.report.obligations.iter().map(|o| o.id.clone()).collect();
    obligation_ids.sort();
    let receipt = ProposalReceipt {
        candidate_id: id.to_owned(),
        base_failure: BaseFailureBinding {
            family: repair.premises.family.clone(),
            contract: repair.premises.contract.clone(),
            obligation_ids,
            premises_digest: repair.premises_digest.clone(),
            data_digest: repair.data_digest.clone(),
        },
        hypothetical: HypotheticalBinding {
            classification: outcome.classification.clone(),
            delta_digest: regimes_digest(delta)?,
            delta_regimes: delta.len() as u64,
            derivation_digest: derivation_digest(outcome),
            derivation_verified: outcome.derivation.as_ref().is_some_and(|d| d.verified),
            addressed: sorted(&outcome.addressed),
            unmet: sorted(&outcome.unmet),
        },
        cost: CostBinding { units, unit_label, sample_budget, sample_size: scored.sample_size },
        lineage: LineageBinding {
            snapshots: snapshots_of(scored),
            source_digest: list_digest(
                "antecedent.proposal_receipt.sources",
                &ranking.source_digests,
            ),
        },
        valuation: ValuationBinding {
            basis: ranking.basis.clone(),
            signal_request_fingerprint: scored.request_fingerprint.clone(),
            signal_identity: scored.signal_identity.clone(),
            evsi_bits: scored.evsi.to_bits(),
            net_value_bits: scored.net_value.map(f64::to_bits),
            rank: scored.rank,
        },
        decision: DecisionReceipt {
            contract_identity: ranking.decision.contract_identity.clone(),
            utility_unit: ranking.decision.utility_unit.clone(),
            action_ids_digest: list_digest(
                "antecedent.proposal_receipt.actions",
                &ranking.decision.action_ids,
            ),
            ranking_identity: ranking.ranking_identity.clone(),
        },
        repair_report_digest: repair.report_digest.clone(),
        ranking_digest: ranking.digest.clone(),
        identity: String::new(),
    };
    Ok(receipt.sealed())
}

/// Refuse an artifact whose stored seal does not match its content.
fn check_seals(repair: &RepairReportArtifact, ranking: &DesignRankingArtifactWire) -> Result<()> {
    let resealed = repair.clone().sealed().map_err(|error| {
        ProposalReceiptError::invalid("proposal_receipt.repair_digest_mismatch", error.to_string())
    })?;
    if resealed.premises_digest != repair.premises_digest
        || resealed.data_digest != repair.data_digest
        || resealed.report_digest != repair.report_digest
    {
        return Err(ProposalReceiptError::mismatch(
            "proposal_receipt.repair_digest_mismatch",
            "the repair artifact's digests do not match its content",
        ));
    }
    let ranking_digest = ranking.compute_digest().map_err(|error| {
        ProposalReceiptError::invalid("proposal_receipt.ranking_digest_mismatch", error.to_string())
    })?;
    if ranking_digest != ranking.digest {
        return Err(ProposalReceiptError::mismatch(
            "proposal_receipt.ranking_digest_mismatch",
            "the ranking artifact's digest does not match its content",
        ));
    }
    Ok(())
}

// ----------------------------------------------------------------- bundle

/// Every proposal of one repair report and one ranking, in canonical order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProposalBundle {
    /// Seal digest of the repair artifact.
    pub repair_report_digest: String,
    /// Digest of the ranking artifact.
    pub ranking_digest: String,
    /// Decision contract identity of the ranking.
    pub decision_contract_identity: String,
    /// Receipts in strictly increasing candidate semantic id.
    pub proposals: Vec<ProposalReceipt>,
    /// Identity over the digests above and every receipt identity, in canonical
    /// order; invariant to the order proposals were supplied in.
    pub identity: String,
}

impl ProposalBundle {
    /// Build the bundle of every candidate the ranking values, from the two
    /// retained artifacts.
    ///
    /// # Errors
    /// `proposal_receipt.repair_digest_mismatch` or
    /// `proposal_receipt.ranking_digest_mismatch` for an artifact whose seal does not
    /// match its content; `proposal_receipt.candidate_not_in_repair` for a ranked
    /// candidate the repair artifact does not declare or evaluate;
    /// `proposal_receipt.candidate_not_proposed` for one it only found invalid or
    /// left unevaluated; `proposal_receipt.cost_mismatch` and
    /// `proposal_receipt.signal_mismatch` for a candidate the two artifacts describe
    /// differently; `proposal_receipt.empty_bundle` for a ranking without candidates.
    pub fn build(
        repair: &RepairReportArtifact,
        ranking: &DesignRankingArtifactWire,
    ) -> Result<Self> {
        check_seals(repair, ranking)?;
        let receipts = ranking
            .candidates
            .iter()
            .map(|c| derive_receipt(repair, ranking, &c.semantic_id))
            .collect::<Result<Vec<_>>>()?;
        Self::from_receipts(receipts)
    }

    /// Assemble receipts into a bundle: sorted into canonical order and sealed.
    ///
    /// # Errors
    /// `proposal_receipt.empty_bundle` without receipts,
    /// `proposal_receipt.duplicate_candidate` for a repeated candidate and
    /// `proposal_receipt.bundle_inconsistent` for receipts read from different
    /// artifacts or decision contracts.
    pub fn from_receipts(mut receipts: Vec<ProposalReceipt>) -> Result<Self> {
        receipts.sort_by(|a, b| a.candidate_id.cmp(&b.candidate_id));
        let Some(first) = receipts.first() else {
            return Err(ProposalReceiptError::invalid(
                "proposal_receipt.empty_bundle",
                "a bundle holds at least one proposal",
            ));
        };
        if receipts.windows(2).any(|pair| pair[0].candidate_id == pair[1].candidate_id) {
            return Err(ProposalReceiptError::invalid(
                "proposal_receipt.duplicate_candidate",
                "a candidate has two receipts",
            ));
        }
        let (repair_report_digest, ranking_digest, contract) = (
            first.repair_report_digest.clone(),
            first.ranking_digest.clone(),
            first.decision.contract_identity.clone(),
        );
        if receipts.iter().any(|r| {
            r.repair_report_digest != repair_report_digest
                || r.ranking_digest != ranking_digest
                || r.decision.contract_identity != contract
        }) {
            return Err(ProposalReceiptError::mismatch(
                "proposal_receipt.bundle_inconsistent",
                "receipts name different repair artifacts, rankings or decision contracts",
            ));
        }
        Ok(Self {
            repair_report_digest,
            ranking_digest,
            decision_contract_identity: contract,
            proposals: receipts,
            identity: String::new(),
        }
        .sealed())
    }

    /// Recompute the identity from the digests and the receipts' recomputed
    /// identities, taken in candidate order whatever the stored order.
    #[must_use]
    pub fn compute_identity(&self) -> String {
        let mut identities: Vec<(&str, String)> = self
            .proposals
            .iter()
            .map(|p| (p.candidate_id.as_str(), p.compute_identity()))
            .collect();
        identities.sort();
        let mut hasher = blake3::Hasher::new();
        put_str(&mut hasher, BUNDLE_DOMAIN);
        put_str(&mut hasher, &self.repair_report_digest);
        put_str(&mut hasher, &self.ranking_digest);
        put_str(&mut hasher, &self.decision_contract_identity);
        put_u64(&mut hasher, identities.len() as u64);
        for (candidate, identity) in &identities {
            put_str(&mut hasher, candidate);
            put_str(&mut hasher, identity);
        }
        hex(&hasher)
    }

    /// Recompute and store the identity (after an edit, to re-seal).
    #[must_use]
    pub fn sealed(mut self) -> Self {
        self.identity = self.compute_identity();
        self
    }

    /// The receipt of `candidate_id`.
    #[must_use]
    pub fn proposal(&self, candidate_id: &str) -> Option<&ProposalReceipt> {
        self.proposals.iter().find(|p| p.candidate_id == candidate_id)
    }
}

fn check_receipt(stored: &ProposalReceipt, expected: &ProposalReceipt) -> Result<()> {
    let id = &stored.candidate_id;
    let refuse = |detail: &'static str, what: &str| {
        Err(ProposalReceiptError::mismatch(detail, format!("candidate {id}: {what} differs")))
    };
    if stored.base_failure.obligation_ids != expected.base_failure.obligation_ids {
        return refuse("proposal_receipt.obligation_mismatch", "the frozen obligation set");
    }
    if stored.base_failure != expected.base_failure {
        return refuse("proposal_receipt.base_failure_mismatch", "the frozen base failure");
    }
    if stored.hypothetical.delta_digest != expected.hypothetical.delta_digest {
        return refuse("proposal_receipt.delta_mismatch", "the hypothetical delta");
    }
    if stored.hypothetical != expected.hypothetical {
        return refuse("proposal_receipt.derivation_mismatch", "the verified derivation");
    }
    if stored.cost != expected.cost {
        return refuse("proposal_receipt.cost_mismatch", "the declared cost");
    }
    if stored.lineage.source_digest != expected.lineage.source_digest {
        return refuse("proposal_receipt.source_digest_mismatch", "the source digests");
    }
    if stored.lineage != expected.lineage {
        return refuse("proposal_receipt.lineage_mismatch", "the snapshot lineage");
    }
    if stored.valuation.signal_request_fingerprint != expected.valuation.signal_request_fingerprint
        || stored.valuation.signal_identity != expected.valuation.signal_identity
    {
        return refuse("proposal_receipt.signal_mismatch", "the signal request or identity");
    }
    if stored.valuation != expected.valuation {
        return refuse("proposal_receipt.value_mismatch", "the ranking value");
    }
    if stored.decision != expected.decision {
        return refuse("proposal_receipt.contract_mismatch", "the decision receipt");
    }
    if stored.repair_report_digest != expected.repair_report_digest {
        return refuse("proposal_receipt.repair_digest_mismatch", "the repair artifact digest");
    }
    if stored.ranking_digest != expected.ranking_digest {
        return refuse("proposal_receipt.ranking_digest_mismatch", "the ranking artifact digest");
    }
    if stored.identity != expected.identity || stored.identity != stored.compute_identity() {
        return refuse("proposal_receipt.identity_mismatch", "the receipt identity");
    }
    Ok(())
}

/// Cross-check a bundle against the two retained artifacts.
///
/// The seals of both artifacts are recomputed first. Every proposal's candidate must
/// then appear in both artifacts (refused otherwise), and every receipt field must
/// equal what the artifacts imply. The proposals must cover exactly the candidates
/// the ranking values, in canonical order, and the bundle identity must follow from
/// its content.
///
/// # Errors
/// A `proposal_receipt.*` refusal naming the first disagreement:
/// `repair_digest_mismatch`, `ranking_digest_mismatch`, `candidate_not_in_ranking`,
/// `candidate_not_in_repair`, `candidate_not_proposed`, `contract_mismatch`,
/// `obligation_mismatch`, `base_failure_mismatch`, `delta_mismatch`,
/// `derivation_mismatch`, `cost_mismatch`, `source_digest_mismatch`,
/// `lineage_mismatch`, `signal_mismatch`, `value_mismatch`, `identity_mismatch`,
/// `candidate_set_mismatch`, `non_canonical_order` or `bundle_identity_mismatch`.
pub fn verify(
    bundle: &ProposalBundle,
    repair: &RepairReportArtifact,
    ranking: &DesignRankingArtifactWire,
) -> Result<()> {
    check_seals(repair, ranking)?;
    if bundle.proposals.is_empty() {
        return Err(ProposalReceiptError::invalid(
            "proposal_receipt.empty_bundle",
            "a bundle holds at least one proposal",
        ));
    }
    let mut expected = Vec::with_capacity(bundle.proposals.len());
    for proposal in &bundle.proposals {
        expected.push(derive_receipt(repair, ranking, &proposal.candidate_id)?);
    }
    if bundle.repair_report_digest != repair.report_digest {
        return Err(ProposalReceiptError::mismatch(
            "proposal_receipt.repair_digest_mismatch",
            "the bundle names another repair artifact",
        ));
    }
    if bundle.ranking_digest != ranking.digest {
        return Err(ProposalReceiptError::mismatch(
            "proposal_receipt.ranking_digest_mismatch",
            "the bundle names another ranking artifact",
        ));
    }
    if bundle.decision_contract_identity != ranking.decision.contract_identity {
        return Err(ProposalReceiptError::mismatch(
            "proposal_receipt.contract_mismatch",
            "the bundle names another decision contract",
        ));
    }
    for (stored, wanted) in bundle.proposals.iter().zip(&expected) {
        check_receipt(stored, wanted)?;
    }
    check_coverage(bundle, ranking)?;
    if bundle.identity != bundle.compute_identity() {
        return Err(ProposalReceiptError::mismatch(
            "proposal_receipt.bundle_identity_mismatch",
            "the bundle identity does not follow from its content",
        ));
    }
    Ok(())
}

fn check_coverage(bundle: &ProposalBundle, ranking: &DesignRankingArtifactWire) -> Result<()> {
    if bundle.proposals.windows(2).any(|pair| pair[0].candidate_id >= pair[1].candidate_id) {
        return Err(ProposalReceiptError::mismatch(
            "proposal_receipt.non_canonical_order",
            "proposals are not in strictly increasing candidate order",
        ));
    }
    let held: BTreeSet<&str> = bundle.proposals.iter().map(|p| p.candidate_id.as_str()).collect();
    let ranked: BTreeSet<&str> =
        ranking.candidates.iter().map(|c| c.semantic_id.as_str()).collect();
    if held != ranked {
        return Err(ProposalReceiptError::mismatch(
            "proposal_receipt.candidate_set_mismatch",
            "the bundle's candidates are not exactly the ranking's candidates",
        ));
    }
    Ok(())
}

// ----------------------------------------------------------------- arrival

/// A joint or separate-marginal law of observed rows that arrived for a back-door
/// contract.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ObservedLaw {
    /// Population the rows were collected in.
    pub population: Arc<str>,
    /// Variables the law covers.
    pub measured: Vec<VariableId>,
    /// Whether the variables were observed jointly; `false` means separate marginals.
    pub joint: bool,
}

/// What arrived.
#[derive(Clone, Debug, PartialEq)]
pub enum ArrivedEvidence {
    /// Actual regimes (each [`EvidenceKind::Available`]) added to the base catalog.
    CatalogDelta(Vec<EvidenceRegime>),
    /// An observed law of a back-door contract's variables.
    ObservedLaw(ObservedLaw),
}

/// One delivery of a proposed study.
#[derive(Clone, Debug, PartialEq)]
pub struct Arrival {
    /// Identity of the delivered snapshot.
    pub snapshot_id: String,
    /// Rows delivered; must equal the candidate's planned sample size.
    pub sample_size: u64,
    /// The delivered evidence.
    pub evidence: ArrivedEvidence,
}

/// The premise the verified derivation rested on that the arrival changed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ChangedPremise {
    /// A regime of a population the proposed evidence is not in.
    Population {
        /// Populations the hypothetical evidence was in, sorted.
        proposed: Vec<String>,
        /// The population that arrived.
        arrived: String,
    },
    /// A regime of the right population under another intervention set.
    Regime {
        /// Population of the regime.
        population: String,
        /// Raw ids of the arrived intervention set, sorted.
        arrived_interventions: Vec<u32>,
    },
}

/// What re-identification on the arrived evidence found.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ArrivalVerdict {
    /// The family's identification now succeeds on the arrived evidence and its
    /// derivation was re-verified.
    Verified {
        /// Checker that decided.
        checker: String,
        /// Stable derivation facts.
        steps: Vec<String>,
        /// Whether the arrived laws equal the proposed laws exactly.
        exact: bool,
        /// Snapshot the evidence came from.
        snapshot_id: String,
    },
    /// The arrived evidence leaves the contract unmet or uncertified.
    StillInsufficient {
        /// Why, in the checker's words.
        reasons: Vec<String>,
    },
    /// The arrived evidence is not of the premises the derivation was verified under.
    Invalidated {
        /// The changed premise.
        changed: ChangedPremise,
    },
}

fn owner(family: RepairFamilyRef<'_>) -> &dyn RepairFamily {
    match family {
        RepairFamilyRef::Transport(f) => f,
        RepairFamilyRef::Backdoor(f) => f,
        RepairFamilyRef::ZTransport(f) => f,
    }
}

fn hypothetical_of(
    family: RepairFamilyRef<'_>,
    candidate: &DurableStudyCandidate,
) -> Result<Vec<EvidenceRegime>> {
    let regimes = match family {
        RepairFamilyRef::Transport(f) => f.hypothetical_regimes(&[candidate]),
        RepairFamilyRef::Backdoor(f) => f.hypothetical_regimes(&[candidate]),
        RepairFamilyRef::ZTransport(f) => f.hypothetical_regimes(&[candidate]),
    };
    regimes.map_err(|reasons| {
        ProposalReceiptError::invalid("proposal_receipt.invalid_delta", reasons.join("; "))
    })
}

fn regimes_wire_digest(regimes: &[EvidenceRegime]) -> Result<String> {
    let invalid =
        |message: String| ProposalReceiptError::invalid("proposal_receipt.invalid_delta", message);
    let catalog = EvidenceCatalog::try_new([], regimes.to_vec(), [], None)
        .map_err(|error| invalid(error.to_string()))?;
    regimes_digest(&EvidenceCatalogWire::from_catalog(&catalog).regimes)
}

/// The receipt must be the one this family and candidate produce.
fn check_receipt_matches(
    receipt: &ProposalReceipt,
    family: RepairFamilyRef<'_>,
    candidate: &DurableStudyCandidate,
) -> Result<Vec<EvidenceRegime>> {
    if candidate.semantic_id().as_ref() != receipt.candidate_id {
        return Err(ProposalReceiptError::mismatch(
            "proposal_receipt.candidate_mismatch",
            "the candidate is not the one the receipt binds",
        ));
    }
    let failed = owner(family);
    if failed.family_id() != receipt.base_failure.family
        || failed.contract_id() != receipt.base_failure.contract
    {
        return Err(ProposalReceiptError::mismatch(
            "proposal_receipt.base_failure_mismatch",
            "the family or contract is not the frozen base failure of the receipt",
        ));
    }
    let mut obligation_ids: Vec<String> =
        failed.unresolved_obligations().iter().map(|o| o.id.to_string()).collect();
    obligation_ids.sort();
    if obligation_ids != receipt.base_failure.obligation_ids {
        return Err(ProposalReceiptError::mismatch(
            "proposal_receipt.obligation_mismatch",
            "the family's unresolved obligations are not the frozen obligation set",
        ));
    }
    let proposed = hypothetical_of(family, candidate)?;
    if regimes_wire_digest(&proposed)? != receipt.hypothetical.delta_digest {
        return Err(ProposalReceiptError::mismatch(
            "proposal_receipt.delta_mismatch",
            "the candidate's hypothetical delta is not the one the receipt binds",
        ));
    }
    Ok(proposed)
}

fn observed_regime(law: &ObservedLaw) -> Result<EvidenceRegime> {
    let measured: Arc<[VariableId]> = law.measured.clone().into();
    let distribution = if law.joint {
        DistributionAvailability::Joint
    } else {
        DistributionAvailability::SeparateMarginals { variables: Arc::clone(&measured) }
    };
    EvidenceRegime::try_new(
        RegimeId::from_raw(0),
        RegimeKind::Observational,
        EvidenceKind::Available,
        Arc::<[VariableId]>::from([]),
        Arc::<[antecedent_core::InterventionAssignment]>::from([]),
        measured,
        Arc::clone(&law.population),
        distribution,
    )
    .map_err(|error| {
        ProposalReceiptError::invalid("proposal_receipt.arrival_invalid", error.to_string())
    })
}

/// The arrival as regimes, refused unless it is available evidence of the right
/// shape for the receipt.
fn arrived_regimes(
    arrival: &Arrival,
    family: RepairFamilyRef<'_>,
    candidate: &DurableStudyCandidate,
) -> Result<Vec<EvidenceRegime>> {
    if arrival.snapshot_id.trim().is_empty() {
        return Err(ProposalReceiptError::invalid(
            "proposal_receipt.arrival_invalid",
            "an arrival names the snapshot it delivers",
        ));
    }
    if arrival.sample_size != candidate.sample_size {
        return Err(ProposalReceiptError::mismatch(
            "proposal_receipt.arrival_sample_mismatch",
            format!(
                "{} rows arrived where the proposal planned {}",
                arrival.sample_size, candidate.sample_size
            ),
        ));
    }
    let regimes = match &arrival.evidence {
        ArrivedEvidence::CatalogDelta(regimes) => regimes.clone(),
        ArrivedEvidence::ObservedLaw(law) => {
            if !matches!(family, RepairFamilyRef::Backdoor(_)) {
                return Err(ProposalReceiptError::mismatch(
                    "proposal_receipt.arrival_family_mismatch",
                    "an observed law answers a back-door contract only",
                ));
            }
            vec![observed_regime(law)?]
        }
    };
    if regimes.is_empty() {
        return Err(ProposalReceiptError::invalid(
            "proposal_receipt.arrival_empty",
            "the arrival delivers no regime",
        ));
    }
    if regimes.iter().any(|r| r.evidence_kind != EvidenceKind::Available) {
        return Err(ProposalReceiptError::not_evidence(
            "a proposed or manipulable regime is a hypothesis; only available regimes arrive",
        ));
    }
    Ok(regimes)
}

fn ids_of(variables: &[VariableId]) -> Vec<u32> {
    let mut raw: Vec<u32> = variables.iter().map(|v| v.raw()).collect();
    raw.sort_unstable();
    raw
}

fn changed_premise(
    proposed: &[EvidenceRegime],
    arrived: &[EvidenceRegime],
) -> Option<ChangedPremise> {
    let populations: BTreeSet<String> = proposed.iter().map(|r| r.population.to_string()).collect();
    for regime in arrived {
        if !populations.contains(regime.population.as_ref()) {
            return Some(ChangedPremise::Population {
                proposed: populations.iter().cloned().collect(),
                arrived: regime.population.to_string(),
            });
        }
        let same_regime = proposed.iter().any(|p| {
            p.population == regime.population
                && ids_of(&p.interventions) == ids_of(&regime.interventions)
        });
        if !same_regime {
            return Some(ChangedPremise::Regime {
                population: regime.population.to_string(),
                arrived_interventions: ids_of(&regime.interventions),
            });
        }
    }
    None
}

fn signatures(regimes: &[EvidenceRegime]) -> Vec<String> {
    let mut out: Vec<String> = regimes
        .iter()
        .map(|r| {
            let mut levels: Vec<String> = r
                .intervention_values
                .iter()
                .map(|a| format!("{}={:?}", a.variable.raw(), a.value))
                .collect();
            levels.sort_unstable();
            let distribution = match &r.distribution {
                DistributionAvailability::Joint => "joint".to_owned(),
                DistributionAvailability::SeparateMarginals { variables } => {
                    format!("marginals{:?}", ids_of(variables))
                }
            };
            format!(
                "pop={};do={:?};levels={};cond={:?};vars={:?};dist={}",
                r.population,
                ids_of(&r.interventions),
                levels.join("|"),
                ids_of(&r.conditioned_on),
                ids_of(&r.measured),
                distribution
            )
        })
        .collect();
    out.sort();
    out
}

fn run_family(
    family: RepairFamilyRef<'_>,
    regimes: &[EvidenceRegime],
    ctx: &ExecutionContext,
) -> Result<FamilyVerdict> {
    let verdict = match family {
        RepairFamilyRef::Transport(f) => {
            let as_proposed: Vec<EvidenceRegime> = regimes
                .iter()
                .cloned()
                .map(|mut regime| {
                    regime.evidence_kind = EvidenceKind::Proposed;
                    regime
                })
                .collect();
            f.check_regimes(as_proposed, ctx)
        }
        RepairFamilyRef::Backdoor(f) => f.check_regimes(regimes, ctx),
        RepairFamilyRef::ZTransport(f) => f.check_regimes(
            regimes
                .iter()
                .cloned()
                .map(|mut regime| {
                    regime.evidence_kind = EvidenceKind::Proposed;
                    regime
                })
                .collect(),
            ctx,
        ),
    };
    verdict.map_err(|RepairCancelled| ProposalReceiptError::cancelled())
}

fn verdict_of(
    verdict: FamilyVerdict,
    blocked: bool,
    exact: bool,
    snapshot_id: &str,
) -> Result<ArrivalVerdict> {
    match verdict {
        FamilyVerdict::Sufficient(derivation) if derivation.verified && !blocked => {
            Ok(ArrivalVerdict::Verified {
                checker: derivation.checker.to_owned(),
                steps: derivation.steps,
                exact,
                snapshot_id: snapshot_id.to_owned(),
            })
        }
        FamilyVerdict::Sufficient(derivation) => Ok(ArrivalVerdict::StillInsufficient {
            reasons: vec![if blocked {
                "an unresolved assumption obligation no study can satisfy remains".to_owned()
            } else {
                format!("the {} derivation was not independently re-verified", derivation.checker)
            }],
        }),
        FamilyVerdict::Insufficient { reasons } | FamilyVerdict::NotCertified { reasons } => {
            Ok(ArrivalVerdict::StillInsufficient { reasons })
        }
        FamilyVerdict::Invalid { reasons } => Err(ProposalReceiptError::invalid(
            "proposal_receipt.arrival_invalid",
            reasons.join("; "),
        )),
    }
}

/// Re-run identification on evidence that actually arrived for a proposal.
///
/// The receipt must be the one `family` and `candidate` produce (same candidate,
/// base failure, obligations and hypothetical delta). The arrival is refused unless
/// it is available evidence of the planned sample size and fits the family. A
/// regime in a population, or under an intervention set, that the hypothetical
/// evidence was not in answers [`ArrivalVerdict::Invalidated`]; otherwise the
/// family's own identification (`check_regimes`) runs on the arrived regimes alone
/// and decides between [`ArrivalVerdict::Verified`] and
/// [`ArrivalVerdict::StillInsufficient`]. The hypothetical delta and derivation are
/// never consulted as evidence.
///
/// # Errors
/// `proposal_receipt.candidate_mismatch`, `base_failure_mismatch`,
/// `obligation_mismatch` or `delta_mismatch` for a receipt that is not this
/// family's and candidate's; `arrival_invalid` for a blank snapshot, an invalid
/// regime or a family that finds the arrival malformed; `arrival_sample_mismatch`
/// for another number of rows; `arrival_family_mismatch` for an observed law given
/// to a transport contract; `arrival_empty` for no regime;
/// `hypothetical_not_evidence` for a regime that is not available;
/// `arrival_cancelled` when cancellation stops the check.
pub fn on_arrival(
    receipt: &ProposalReceipt,
    family: RepairFamilyRef<'_>,
    candidate: &DurableStudyCandidate,
    arrival: &Arrival,
    ctx: &ExecutionContext,
) -> Result<ArrivalVerdict> {
    let proposed = check_receipt_matches(receipt, family, candidate)?;
    let arrived = arrived_regimes(arrival, family, candidate)?;
    if let Some(changed) = changed_premise(&proposed, &arrived) {
        return Ok(ArrivalVerdict::Invalidated { changed });
    }
    let verdict = run_family(family, &arrived, ctx)?;
    let blocked = owner(family).unresolved_obligations().iter().any(|o| !o.satisfiable_by_study());
    let exact = signatures(&proposed) == signatures(&arrived);
    verdict_of(verdict, blocked, exact, &arrival.snapshot_id)
}
