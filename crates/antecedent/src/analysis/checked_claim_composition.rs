//! In-process checked synthesis over independently trusted prepared-study claims.
use super::PreparedStudy;
use antecedent_contract::{
    CheckedWorldPointError, CheckedWorldPointLimits, ClaimDomains, ClaimEnvelope, ClaimKind,
    ClaimOperation, DomainStatus, HandoffReceipt, ProvenanceGraph, ProvenanceNode, ReasoningView,
    SlotAvailability, admit_composition_targets, composition_claim_premises_identity,
    quote_claim_premises,
};
use antecedent_core::{
    CanonicalWriter, ContractIdentities, ScientificTarget, SemanticDigest, TargetForm,
};
use std::sync::Arc;

/// An independently trusted claim paired with the prepared study that produced its program.
#[derive(Clone, Copy, Debug)]
pub struct PreparedClaimParent<'a> {
    /// Retained study used to verify the claim's program and target identities.
    pub prepared: &'a PreparedStudy,
    /// Caller-trusted claim envelope; its source authenticity is not verified here.
    pub claim: &'a ClaimEnvelope,
    /// Linear coefficient applied to its finite point value.
    pub coefficient: f64,
}

fn parent_target<'a>(
    parent: &PreparedClaimParent<'a>,
) -> Result<&'a ScientificTarget, CheckedWorldPointError> {
    use crate::planner::lowered::ProblemTarget;
    let lowered =
        parent.prepared.plan().lowered().ok_or(CheckedWorldPointError::WitnessMismatch)?;
    match lowered.scientific().target() {
        ProblemTarget::Target(target) => Ok(target),
        _ => Err(CheckedWorldPointError::WitnessMismatch),
    }
}

/// An in-process checked claim with its arithmetic proof and retained admission.
/// Its synthetic identity is not an analysis-result wire seal; portable export
/// and independent source authentication are not provided.
#[derive(Clone, Debug)]
pub struct CheckedStudyClaim {
    inner: Arc<CheckedStudyClaimInner>,
}
#[derive(Debug)]
struct CheckedStudyClaimInner {
    claim: ClaimEnvelope,
    parents: Arc<[SemanticDigest]>,
    provenance: ProvenanceGraph,
    receipt: HandoffReceipt,
    target: ScientificTarget,
    value: f64,
    proof: SemanticDigest,
    _lease: Arc<antecedent_core::execution::AdmissionLease>,
}
impl CheckedStudyClaim {
    /// The internally created conditional point claim.
    #[must_use]
    pub fn claim(&self) -> &ClaimEnvelope {
        &self.inner.claim
    }
    /// Actual parent claim IDs in canonical order.
    #[must_use]
    pub fn parents(&self) -> &[SemanticDigest] {
        &self.inner.parents
    }
    /// Parent and child ancestry.
    #[must_use]
    pub fn provenance(&self) -> &ProvenanceGraph {
        &self.inner.provenance
    }
    /// Explicit retention and loss report.
    #[must_use]
    pub fn receipt(&self) -> &HandoffReceipt {
        &self.inner.receipt
    }
    /// Exact declared output target.
    #[must_use]
    pub fn target(&self) -> &ScientificTarget {
        &self.inner.target
    }
    /// Derived finite point.
    #[must_use]
    pub fn point(&self) -> f64 {
        self.inner.value
    }
    /// Identity of the in-process arithmetic proof.
    #[must_use]
    pub fn proof_identity(&self) -> SemanticDigest {
        self.inner.proof
    }
}
fn outer_action_count(world: &antecedent_core::InterventionWorld) -> usize {
    let action = world.action();
    action.at_decision_interventions().map_or_else(
        || action.structural_actions().map_or(1, <[antecedent_core::StructuralAction]>::len),
        <[antecedent_core::Intervention]>::len,
    )
}
fn provenance_node(
    id: SemanticDigest,
    operation: &str,
    parents: &[Arc<str>],
    assumptions: &antecedent_core::assumption::AssumptionSet,
) -> ProvenanceNode {
    ProvenanceNode {
        artifact_id: Arc::from(id.to_hex()),
        operation: Arc::from(operation),
        parents: parents.iter().cloned().collect(),
        assumptions: assumptions.clone(),
        library_version: Arc::from(antecedent_core::VERSION),
        config_digest: None,
    }
}
/// Validate retained witnesses and resource limits before deriving the claim.
fn admit_checked_claim_composition(
    target: &ScientificTarget,
    parents: &[PreparedClaimParent<'_>],
    operation: ClaimOperation,
    limits: &CheckedWorldPointLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<Arc<antecedent_core::execution::AdmissionLease>, CheckedWorldPointError> {
    if ctx.cancellation.is_cancelled() {
        return Err(CheckedWorldPointError::Cancelled);
    }
    if !matches!(operation, ClaimOperation::Contrast | ClaimOperation::Aggregate) {
        return Err(CheckedWorldPointError::IncompatibleParents);
    }
    if parents.len() < 2 || parents.len() > limits.max_terms {
        return Err(CheckedWorldPointError::ParentCount);
    }
    // Check the exact retained producer identities before consulting a target.
    // The caller remains responsible for authenticity of the envelope itself.
    for parent in parents {
        if ctx.cancellation.is_cancelled() {
            return Err(CheckedWorldPointError::Cancelled);
        }
        let payloads =
            parent.prepared.program_cache().get().ok_or(CheckedWorldPointError::WitnessMismatch)?;
        if payloads.identities(parent.claim.identities.data_snapshot) != parent.claim.identities {
            return Err(CheckedWorldPointError::WitnessMismatch);
        }
        parent_target(parent)?;
    }
    let worlds = |t: &ScientificTarget| -> Result<usize, CheckedWorldPointError> {
        if t.history_indexed_law().is_some() {
            return Err(CheckedWorldPointError::HistoryLawNotSupported);
        }
        match t.form() {
            TargetForm::World(_) => Ok(1),
            TargetForm::Contrast(c) => Ok(c.worlds().len()),
            _ => Err(CheckedWorldPointError::WitnessMismatch),
        }
    };
    let mut total_worlds = worlds(target)?;
    let mut assumptions = target.assumptions().records().len();
    let count_actions = |t: &ScientificTarget| -> Result<usize, CheckedWorldPointError> {
        match t.form() {
            TargetForm::World(w) => Ok(outer_action_count(w)),
            TargetForm::Contrast(c) => c.worlds().worlds().iter().try_fold(0usize, |n, w| {
                if ctx.cancellation.is_cancelled() {
                    return Err(CheckedWorldPointError::Cancelled);
                }
                n.checked_add(outer_action_count(w)).ok_or(CheckedWorldPointError::TermLimit)
            }),
            _ => Err(CheckedWorldPointError::WitnessMismatch),
        }
    };
    for parent in parents {
        if ctx.cancellation.is_cancelled() {
            return Err(CheckedWorldPointError::Cancelled);
        }
        total_worlds = total_worlds
            .checked_add(worlds(parent_target(parent)?)?)
            .ok_or(CheckedWorldPointError::TermLimit)?;
        assumptions = assumptions
            .checked_add(parent_target(parent)?.assumptions().records().len())
            .ok_or(CheckedWorldPointError::TermLimit)?;
    }
    if total_worlds > limits.max_terms
        || assumptions > limits.max_assumption_records
        || total_worlds.checked_mul(parents.len()).is_none_or(|n| n > limits.max_comparisons)
    {
        return Err(CheckedWorldPointError::TermLimit);
    }
    let quote = quote_claim_premises(parents.iter().map(|p| p.claim), parents.len(), limits, ctx)?;
    let lease = admit_composition_targets(
        std::iter::once(target).chain(
            parents.iter().map(|p| parent_target(p).expect("prepared target checked above")),
        ),
        quote,
        limits,
        ctx,
    )?;
    let mut actions = count_actions(target)?;
    for parent in parents {
        actions = actions
            .checked_add(count_actions(parent_target(parent)?)?)
            .ok_or(CheckedWorldPointError::TermLimit)?;
    }
    if actions > limits.max_world_actions {
        return Err(CheckedWorldPointError::TermLimit);
    }
    Ok(lease)
}

/// Accumulate a target's world coefficients without changing their traversal order.
fn add_target_coefficients(
    map: &mut std::collections::BTreeMap<SemanticDigest, f64>,
    t: &ScientificTarget,
    coefficient: f64,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<(), CheckedWorldPointError> {
    let mut add_world = |world: &antecedent_core::InterventionWorld,
                         weight: f64|
     -> Result<(), CheckedWorldPointError> {
        if ctx.cancellation.is_cancelled() {
            return Err(CheckedWorldPointError::Cancelled);
        }
        let id = ScientificTarget::new(
            TargetForm::World(world.clone()),
            t.assumptions().clone(),
            t.inference(),
        )
        .identity();
        let next = map.get(&id).copied().unwrap_or(0.0) + weight;
        if !weight.is_finite() || !next.is_finite() {
            return Err(CheckedWorldPointError::NonFiniteArithmetic);
        }
        if next == 0.0 {
            map.remove(&id);
        } else {
            map.insert(id, next);
        }
        Ok(())
    };
    match t.form() {
        TargetForm::World(w) => add_world(w, coefficient),
        TargetForm::Contrast(c) => {
            for term in c.terms() {
                add_world(
                    c.worlds().get(term.world).ok_or(CheckedWorldPointError::WorldNotFound)?,
                    coefficient * term.coefficient,
                )?;
            }
            Ok(())
        }
        _ => Err(CheckedWorldPointError::WitnessMismatch),
    }
}

/// Derive a conditional point claim using retained prepared targets and independently trusted envelopes.
/// Parent program identities are matched to each prepared study; caller must authenticate the envelopes.
/// No new identification, support, uncertainty or calibration is asserted. Output is in-process only.
/// # Errors
/// Invalid parent bindings or premises, target arithmetic, limits, or cancellation.
#[allow(
    clippy::too_many_lines,
    reason = "the checked synthesis keeps its admission and proof construction in one operation"
)]
#[allow(
    clippy::missing_panics_doc,
    reason = "parent targets are immutably checked before the admitted traversal"
)]
pub fn compose_checked_study_claims(
    target: &ScientificTarget,
    parents: &[PreparedClaimParent<'_>],
    operation: ClaimOperation,
    limits: &CheckedWorldPointLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<CheckedStudyClaim, CheckedWorldPointError> {
    use antecedent_core::IdentificationStatus;
    use std::collections::BTreeMap;
    let lease = admit_checked_claim_composition(target, parents, operation, limits, ctx)?;
    let first = parents[0].claim;
    let mut ordered: Vec<_> = parents.iter().collect();
    ordered.sort_by_key(|p| p.claim.claim_id);
    if ordered.windows(2).any(|p| p[0].claim.claim_id == p[1].claim.claim_id) {
        return Err(CheckedWorldPointError::DuplicateParent);
    }
    let mut expected = BTreeMap::<SemanticDigest, f64>::new();
    let mut actual = BTreeMap::<SemanticDigest, f64>::new();
    add_target_coefficients(&mut expected, target, 1.0, ctx)?;
    let mut value = 0.0;
    let mut weight_sum = 0.0;
    let output_units = match target.form() {
        TargetForm::World(w) => w.binding().outcome().units().name(),
        TargetForm::Contrast(c) => c.worlds().worlds()[0].binding().outcome().units().name(),
        _ => return Err(CheckedWorldPointError::WitnessMismatch),
    };
    let mut writer = CanonicalWriter::new("antecedent.checked_claim_derivation.v1");
    writer
        .text(operation.as_str())
        .bytes(target.identity().as_bytes())
        .bytes(
            antecedent_core::scientific_target::composition_assumptions_identity(
                target.assumptions().records(),
            )
            .as_bytes(),
        )
        .word(parents.len() as u64);
    for parent in &ordered {
        if ctx.cancellation.is_cancelled() {
            return Err(CheckedWorldPointError::Cancelled);
        }
        let claim = parent.claim;
        let slot = claim
            .reasoning
            .identification
            .as_ref()
            .ok_or(CheckedWorldPointError::IncompatibleParents)?;
        if claim.kind != ClaimKind::Point
            || claim.value.is_none_or(|v| !v.is_finite())
            || !matches!(
                slot.status,
                IdentificationStatus::NonparametricallyIdentified
                    | IdentificationStatus::IdentifiedUnderParametricRestrictions
            )
            || !slot.masses_are_valid()
            || slot.identified_mass.to_bits() != 1.0_f64.to_bits()
            || slot.unidentified_mass != 0.0
            || slot.unevaluable_mass != 0.0
            || slot.incomplete_search_mass != 0.0
            || !slot.full_mass_scope
            || slot.search_capped
            || claim.identities.observation != first.identities.observation
            || claim.identities.data_snapshot != first.identities.data_snapshot
            || claim.reasoning.assumptions != first.reasoning.assumptions
            || parent_target(parent)?.assumptions() != target.assumptions()
            || !matches!(
                claim.domains.identification,
                DomainStatus::Identified | DomainStatus::Unknown
            )
            || !matches!(claim.domains.support, DomainStatus::Supported | DomainStatus::Unknown)
            || claim.domains.evaluated != DomainStatus::Evaluated
            || !parent.coefficient.is_finite()
            || parent.coefficient == 0.0
            || (operation == ClaimOperation::Aggregate && parent.coefficient < 0.0)
        {
            return Err(CheckedWorldPointError::IncompatibleParents);
        }
        let unit = match parent_target(parent)?.form() {
            TargetForm::World(w) => w.binding().outcome().units().name(),
            TargetForm::Contrast(c) => c.worlds().worlds()[0].binding().outcome().units().name(),
            _ => return Err(CheckedWorldPointError::WitnessMismatch),
        };
        if unit != output_units || claim.outcome_units.as_deref().is_some_and(|value| value != unit)
        {
            return Err(CheckedWorldPointError::IncompatibleParents);
        }
        add_target_coefficients(&mut actual, parent_target(parent)?, parent.coefficient, ctx)?;
        weight_sum += parent.coefficient;
        let operand = claim.value.ok_or(CheckedWorldPointError::IncompatibleParents)?;
        value += operand * parent.coefficient;
        if !weight_sum.is_finite() || !value.is_finite() {
            return Err(CheckedWorldPointError::NonFiniteArithmetic);
        }
        writer
            .bytes(claim.claim_id.as_bytes())
            .bytes(parent_target(parent)?.identity().as_bytes())
            .bytes(
                antecedent_core::scientific_target::composition_assumptions_identity(
                    parent_target(parent)?.assumptions().records(),
                )
                .as_bytes(),
            )
            .real(parent.coefficient)
            .real(operand)
            .bytes(composition_claim_premises_identity(claim).as_bytes());
        for identity in claim.identities.refs().iter() {
            writer.text(identity.domain.as_str()).bytes(identity.digest.as_bytes());
        }
        writer.word(claim.evidence.len() as u64);
        for item in claim.evidence.iter() {
            writer.text(item);
        }
    }
    if ctx.cancellation.is_cancelled() {
        return Err(CheckedWorldPointError::Cancelled);
    }
    if operation == ClaimOperation::Aggregate && weight_sum.to_bits() != 1.0_f64.to_bits() {
        return Err(CheckedWorldPointError::IncompatibleParents);
    }
    if expected != actual {
        return Err(CheckedWorldPointError::WitnessMismatch);
    }
    let proof = SemanticDigest::from_bytes(writer.finish());
    finish_checked_claim(target, first, &ordered, operation, value, proof, lease)
}

/// Construct the derived claim and retained handoff after arithmetic has been checked.
fn finish_checked_claim(
    target: &ScientificTarget,
    first: &ClaimEnvelope,
    ordered: &[&PreparedClaimParent<'_>],
    operation: ClaimOperation,
    value: f64,
    proof: SemanticDigest,
    lease: Arc<antecedent_core::execution::AdmissionLease>,
) -> Result<CheckedStudyClaim, CheckedWorldPointError> {
    let parent_ids: Arc<[SemanticDigest]> =
        ordered.iter().map(|p| p.claim.claim_id).collect::<Vec<_>>().into();
    // These in-process markers identify the derivation inputs; they do not
    // assert an identification product or an inference certificate.
    let mut identification =
        CanonicalWriter::new("antecedent.checked_claim_identification_premises.v1");
    identification.bytes(target.identity().as_bytes()).bytes(proof.as_bytes());
    let identification = SemanticDigest::from_bytes(identification.finish());
    let mut inference = CanonicalWriter::new("antecedent.checked_claim_inference_binding.v1");
    inference.bytes(target.identity().as_bytes()).bytes(proof.as_bytes());
    let inference = SemanticDigest::from_bytes(inference.finish());
    let mut claim_id = CanonicalWriter::new("antecedent.checked_claim_identity.v1");
    claim_id
        .bytes(proof.as_bytes())
        .bytes(identification.as_bytes())
        .bytes(inference.as_bytes())
        .real(value);
    let claim_id = SemanticDigest::from_bytes(claim_id.finish());
    let units = match target.form() {
        TargetForm::World(w) => w.binding().outcome().units().name(),
        TargetForm::Contrast(c) => c.worlds().worlds()[0].binding().outcome().units().name(),
        _ => return Err(CheckedWorldPointError::WitnessMismatch),
    };
    let reasoning = ReasoningView::new(
        SlotAvailability::unavailable("derived_identification_not_established"),
        SlotAvailability::unavailable("derived_support_not_established"),
        SlotAvailability::unavailable("derived_uncertainty_not_propagated"),
        first.reasoning.assumptions.clone(),
    );
    let parent_refs: Vec<Arc<str>> =
        parent_ids.iter().map(|id| Arc::from(format!("parent_claim:{}", id.to_hex()))).collect();
    let claim = ClaimEnvelope::new(
        claim_id,
        ContractIdentities::new(
            target.identity(),
            identification,
            None,
            None,
            inference,
            first.identities.observation,
            first.identities.data_snapshot,
        ),
        ClaimKind::Point,
        Some(value),
        Some(Arc::from(units)),
        reasoning,
        ClaimDomains::new(DomainStatus::Unknown, DomainStatus::Unknown, DomainStatus::Evaluated),
        None,
        parent_refs,
    );
    let mut provenance = ProvenanceGraph::new();
    for parent in ordered {
        provenance
            .try_push(provenance_node(
                parent.claim.claim_id,
                "parent_claim",
                &[],
                parent_target(parent)?.assumptions().records(),
            ))
            .map_err(|_| CheckedWorldPointError::WitnessMismatch)?;
    }
    let parent_hex: Vec<Arc<str>> = parent_ids.iter().map(|id| Arc::from(id.to_hex())).collect();
    provenance
        .try_push(provenance_node(
            claim_id,
            operation.as_str(),
            &parent_hex,
            target.assumptions().records(),
        ))
        .map_err(|_| CheckedWorldPointError::WitnessMismatch)?;
    let receipt = derived_handoff(parent_ids[0], claim_id, operation);
    Ok(CheckedStudyClaim {
        inner: Arc::new(CheckedStudyClaimInner {
            claim,
            parents: parent_ids,
            provenance,
            receipt,
            target: target.clone(),
            value,
            proof,
            _lease: lease,
        }),
    })
}

/// Describe the retained and withheld domains of a checked claim handoff.
fn derived_handoff(
    parent_id: SemanticDigest,
    claim_id: SemanticDigest,
    operation: ClaimOperation,
) -> HandoffReceipt {
    HandoffReceipt::new(
        parent_id,
        Some(claim_id),
        operation.as_str(),
        "checked_derive",
        [
            Arc::from("parent_claim_ids"),
            Arc::from("target"),
            Arc::from("point_value"),
            Arc::from("assumptions"),
        ],
        [
            Arc::from("identification"),
            Arc::from("support"),
            Arc::from("uncertainty"),
            Arc::from("calibration"),
            Arc::from("parent_evidence"),
        ],
        [],
        [Arc::from("portable_export")],
    )
}
