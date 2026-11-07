use super::{
    Arc, CanonicalWriter, ClaimEnvelope, ClaimKind, ScientificTarget, SemanticDigest, TargetForm,
};

/// One point-valued parent and the scientific target that its target digest names.
/// The caller must separately establish the authenticity of the parent claim.
#[derive(Clone, Copy, Debug)]
pub struct WorldPointParent<'a> {
    /// Input claim, normally independently accepted or verified.
    pub claim: &'a ClaimEnvelope,
    /// The single-world target whose identity is bound by the claim.
    pub target: &'a ScientificTarget,
}

/// Checked arithmetic over world point estimates. This is not a new claim envelope:
/// it does not establish identification, a program, uncertainty or calibration for
/// the resulting quantity. Its proof identity binds the target, operands and coefficients.
#[derive(Clone, Debug)]
pub struct CheckedWorldPoint {
    target: Arc<ScientificTarget>,
    value: f64,
    parent_ids: Arc<[SemanticDigest]>,
    evidence: Arc<[Arc<str>]>,
    proof_identity: SemanticDigest,
    _retained: Arc<antecedent_core::execution::AdmissionLease>,
}

impl CheckedWorldPoint {
    /// The exact declared linear contrast target.
    #[must_use]
    pub fn target(&self) -> &ScientificTarget {
        &self.target
    }
    /// Weighted sum of the parent point estimates.
    #[must_use]
    pub const fn value(&self) -> f64 {
        self.value
    }
    /// Parent claims ordered by their scientific world identity.
    #[must_use]
    pub fn parent_ids(&self) -> &[SemanticDigest] {
        &self.parent_ids
    }
    /// Union of the evidence references supplied by the parents.
    #[must_use]
    pub fn evidence(&self) -> &[Arc<str>] {
        &self.evidence
    }
    /// Scientific work still required before treating the result as a full claim.
    #[must_use]
    pub const fn limitations(&self) -> &'static [&'static str] {
        &[
            "identification_not_established",
            "uncertainty_not_propagated",
            "calibration_not_transferred",
        ]
    }
    /// Identity of this checked arithmetic, not a claim or contract identity.
    #[must_use]
    pub const fn proof_identity(&self) -> SemanticDigest {
        self.proof_identity
    }
}

/// Caller-selected work, target payload and retained-memory budgets.
/// Admission is acquired before the iterative quote and is held by the result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CheckedWorldPointLimits {
    /// Maximum combined target traversal work across all targets.
    pub max_target_work: usize,
    /// Maximum combined bytes in target payloads.
    pub max_target_payload_bytes: usize,
    /// Maximum action/expression nesting, additionally capped by the encoder's validated depth.
    pub max_target_depth: usize,
    /// Caller-authorized bytes reserved from the execution context throughout the result lifetime.
    pub max_memory_bytes: u64,
    /// Maximum contrast terms / parent world witnesses.
    pub max_terms: usize,
    /// Maximum worst-case term-to-witness comparisons.
    pub max_comparisons: usize,
    /// Maximum total top-level world actions over the target and witnesses.
    pub max_world_actions: usize,
    /// Maximum total assumption records over the target and witnesses.
    pub max_assumption_records: usize,
    /// Maximum evidence references across all parents.
    pub max_evidence_references: usize,
    /// Maximum combined UTF-8 bytes of evidence reference text.
    pub max_evidence_bytes: usize,
}

fn outer_action_count(world: &antecedent_core::InterventionWorld) -> usize {
    let action = world.action();
    action
        .at_decision_interventions()
        .map_or_else(|| action.structural_actions().map_or(1, <[_]>::len), <[_]>::len)
}

/// Why a checked world-point composition cannot be evaluated.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum CheckedWorldPointError {
    /// The declared target is not a linear world contrast.
    #[error("expected a linear world contrast target")]
    ExpectedContrast,
    /// The parent count does not match the contrast terms.
    #[error("parent count does not match the contrast worlds")]
    ParentCount,
    /// Execution was cancelled.
    #[error("world composition cancelled")]
    Cancelled,
    /// Memory admission was refused.
    #[error("world composition memory admission refused")]
    Admission,
    /// A caller-supplied structural work limit is exceeded.
    #[error("world composition work limit exceeded")]
    TermLimit,
    /// History-indexed laws need a separately admitted verifier.
    #[error("history-indexed target law is not supported by this point verifier")]
    HistoryLawNotSupported,
    /// The same claim identity was supplied twice.
    #[error("duplicate parent claim identity")]
    DuplicateParent,
    /// Parent kind, finite point, observation, data, units or assumption slots differ.
    #[error("incompatible world point parents")]
    IncompatibleParents,
    /// The evidence references exceed the bounded count or text budget.
    #[error("world evidence limit exceeded")]
    EvidenceLimit,
    /// A contrast term names no world in its declared target.
    #[error("contrast world not found")]
    WorldNotFound,
    /// Parent target evidence does not match exactly one term.
    #[error("world target witness mismatch")]
    WitnessMismatch,
    /// An arithmetic contribution or result is not finite.
    #[error("world arithmetic is not finite")]
    NonFiniteArithmetic,
}

/// Reserve a caller-authorized composition workspace and quote each borrowed target.
/// The returned lease must be retained while any output allocation remains live.
/// # Errors
/// Cancellation, admission, arithmetic overflow, target depth or work limits.
pub fn admit_composition_targets<'a>(
    targets: impl IntoIterator<Item = &'a ScientificTarget>,
    premise_quote: antecedent_core::scientific_target::CompositionTargetQuote,
    limits: &CheckedWorldPointLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<Arc<antecedent_core::execution::AdmissionLease>, CheckedWorldPointError> {
    if ctx.cancellation.is_cancelled() {
        return Err(CheckedWorldPointError::Cancelled);
    }
    // The quote's explicit stacks are charged before their first allocation.
    let stack_bytes = limits
        .max_target_work
        .checked_mul(128)
        .and_then(|v| v.checked_add(4096))
        .ok_or(CheckedWorldPointError::TermLimit)?;
    if u64::try_from(stack_bytes).map_err(|_| CheckedWorldPointError::TermLimit)?
        > limits.max_memory_bytes
    {
        return Err(CheckedWorldPointError::TermLimit);
    }
    let lease =
        Arc::new(ctx.acquire_admission(0, limits.max_memory_bytes).map_err(
            |error| match error {
                antecedent_core::execution::AdmissionError::Cancelled => {
                    CheckedWorldPointError::Cancelled
                }
                _ => CheckedWorldPointError::Admission,
            },
        )?);
    let mut work = 0usize;
    let mut payload = premise_quote.payload_bytes;
    for target in targets {
        // Apply the remaining global budgets to this traversal, including the
        // already-quoted parent premises, rather than charging after a full pass.
        let remaining_work = limits
            .max_target_work
            .checked_sub(premise_quote.work)
            .and_then(|v| v.checked_sub(work))
            .ok_or(CheckedWorldPointError::TermLimit)?;
        let remaining_payload = limits
            .max_target_payload_bytes
            .checked_sub(payload)
            .ok_or(CheckedWorldPointError::TermLimit)?;
        let qlimits = antecedent_core::scientific_target::CompositionQuoteLimits {
            max_work: remaining_work,
            max_payload_bytes: remaining_payload,
            max_depth: limits.max_target_depth,
        };
        let q = antecedent_core::scientific_target::quote_composition_target(target, &qlimits, ctx)
            .map_err(|_| {
                if ctx.cancellation.is_cancelled() {
                    CheckedWorldPointError::Cancelled
                } else {
                    CheckedWorldPointError::TermLimit
                }
            })?;
        work = work.checked_add(q.work).ok_or(CheckedWorldPointError::TermLimit)?;
        payload = payload.checked_add(q.payload_bytes).ok_or(CheckedWorldPointError::TermLimit)?;
        if work > limits.max_target_work || payload > limits.max_target_payload_bytes {
            return Err(CheckedWorldPointError::TermLimit);
        }
    }
    // Every target may be encoded repeatedly for each world witness. Charge
    // those passes and comparisons, not only the visitor's first traversal.
    let repetitions = limits.max_terms.checked_add(2).ok_or(CheckedWorldPointError::TermLimit)?;
    let total_work = work
        .checked_mul(repetitions)
        .and_then(|v| v.checked_add(premise_quote.work))
        .and_then(|v| v.checked_add(limits.max_comparisons))
        .and_then(|v| {
            limits
                .max_terms
                .checked_mul(limits.max_terms)
                .and_then(|n| n.checked_mul(4))
                .and_then(|n| v.checked_add(n))
        })
        .ok_or(CheckedWorldPointError::TermLimit)?;
    if total_work > limits.max_target_work || payload > limits.max_target_payload_bytes {
        return Err(CheckedWorldPointError::TermLimit);
    }
    // Covers visitor stacks, worst repeated world/assumption encoding, canonical
    // sorted buffers, result target clone, parent vectors, evidence tree and text.
    let target_bytes = work
        .checked_mul(128)
        .and_then(|v| payload.checked_mul(4).and_then(|p| v.checked_add(p)))
        .and_then(|v| v.checked_mul(repetitions))
        .ok_or(CheckedWorldPointError::TermLimit)?;
    let evidence_bytes = limits
        .max_evidence_references
        .checked_mul(256)
        .and_then(|v| v.checked_add(limits.max_evidence_bytes.checked_mul(2)?))
        .ok_or(CheckedWorldPointError::TermLimit)?;
    let total = target_bytes
        .checked_add(evidence_bytes)
        .and_then(|v| v.checked_add(4096))
        .ok_or(CheckedWorldPointError::TermLimit)?;
    if u64::try_from(total).map_err(|_| CheckedWorldPointError::TermLimit)?
        > limits.max_memory_bytes
    {
        return Err(CheckedWorldPointError::TermLimit);
    }
    Ok(lease)
}

/// Quote borrowed claim premises without attesting their scientific binding.
/// The count must match the iterator and remain within the caller's term limit.
/// # Errors
/// Cancellation or a work/evidence limit.
#[allow(
    clippy::too_many_lines,
    reason = "The bounded quote accounts for related inputs in one traversal."
)]
pub fn quote_claim_premises<'a>(
    claims: impl Iterator<Item = &'a ClaimEnvelope>,
    parent_count: usize,
    limits: &CheckedWorldPointLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<antecedent_core::scientific_target::CompositionTargetQuote, CheckedWorldPointError> {
    use crate::reasoning::SlotAvailability;
    use antecedent_core::assumption::{Assumption, AssumptionSource};
    if ctx.cancellation.is_cancelled() {
        return Err(CheckedWorldPointError::Cancelled);
    }
    if parent_count > limits.max_terms {
        return Err(CheckedWorldPointError::TermLimit);
    }
    let mut evidence_count = 0usize;
    let mut evidence_bytes = 0usize;
    let mut work = 0usize;
    let mut charge = |n: usize| -> Result<(), CheckedWorldPointError> {
        work = work.checked_add(n).ok_or(CheckedWorldPointError::TermLimit)?;
        if work > limits.max_target_work {
            return Err(CheckedWorldPointError::TermLimit);
        }
        Ok(())
    };
    let mut actual_count = 0usize;
    for claim in claims {
        actual_count = actual_count.checked_add(1).ok_or(CheckedWorldPointError::TermLimit)?;
        if actual_count > limits.max_terms || actual_count > parent_count {
            return Err(CheckedWorldPointError::TermLimit);
        }
        // Each parent has fixed identity/status fields, even with no text.
        charge(64)?;
        if ctx.cancellation.is_cancelled() {
            return Err(CheckedWorldPointError::Cancelled);
        }
        if let Some(units) = &claim.outcome_units {
            charge(units.len())?;
        }
        match &claim.reasoning.assumptions {
            SlotAvailability::Unavailable { reason } => charge(reason.len())?,
            SlotAvailability::Available(slot) => {
                charge(slot.obligations.len())?;
                for obligation in slot.obligations.iter() {
                    if ctx.cancellation.is_cancelled() {
                        return Err(CheckedWorldPointError::Cancelled);
                    }
                    charge(obligation.id.len())?;
                    charge(obligation.description.len())?;
                    if let Some(x) = &obligation.required_check {
                        charge(x.len())?;
                    }
                    if let Some(x) = &obligation.evidence {
                        charge(x.len())?;
                    }
                    match &obligation.source {
                        AssumptionSource::AlgorithmDefault { algorithm } => {
                            charge(algorithm.len())?
                        }
                        AssumptionSource::Derived { from } => charge(from.len())?,
                        _ => {}
                    }
                    match &obligation.assumption {
                        Some(Assumption::Custom { id, description }) => {
                            charge(id.len())?;
                            charge(description.len())?;
                        }
                        Some(Assumption::ParametricRestriction(p)) => {
                            charge(p.id.len())?;
                            charge(p.description.len())?;
                        }
                        Some(Assumption::PriorRestriction(p)) => {
                            charge(p.id.len())?;
                            charge(p.description.len())?;
                        }
                        _ => {}
                    }
                }
            }
        }
        evidence_count = evidence_count
            .checked_add(claim.evidence.len())
            .ok_or(CheckedWorldPointError::EvidenceLimit)?;
        if evidence_count > limits.max_evidence_references {
            return Err(CheckedWorldPointError::EvidenceLimit);
        }
        for item in claim.evidence.iter() {
            if ctx.cancellation.is_cancelled() {
                return Err(CheckedWorldPointError::Cancelled);
            }
            evidence_bytes = evidence_bytes
                .checked_add(item.len())
                .ok_or(CheckedWorldPointError::EvidenceLimit)?;
            if evidence_bytes > limits.max_evidence_bytes {
                return Err(CheckedWorldPointError::EvidenceLimit);
            }
        }
    }
    if actual_count != parent_count {
        return Err(CheckedWorldPointError::TermLimit);
    }
    // Upper bound repeated equality and lexicographic tree comparisons; no
    // unbounded evidence or obligation text is compared before this check.
    let base = work
        .checked_add(evidence_bytes)
        .and_then(|v| v.checked_add(evidence_count))
        .ok_or(CheckedWorldPointError::TermLimit)?;
    let passes = parent_count
        .checked_add(evidence_count)
        .and_then(|v| v.checked_add(2))
        .ok_or(CheckedWorldPointError::TermLimit)?;
    Ok(antecedent_core::scientific_target::CompositionTargetQuote {
        work: base.checked_mul(passes).ok_or(CheckedWorldPointError::TermLimit)?,
        payload_bytes: base,
    })
}

fn encode_parent_premises(writer: &mut CanonicalWriter, claim: &ClaimEnvelope) {
    use crate::reasoning::SlotAvailability;
    use antecedent_core::CanonicalEncode;
    use antecedent_core::assumption::AssumptionSource;
    writer.text(claim.outcome_units.as_deref().unwrap_or(""));
    match &claim.reasoning.assumptions {
        SlotAvailability::Unavailable { reason } => {
            writer.word(0).text(reason);
        }
        SlotAvailability::Available(slot) => {
            writer.word(1).word(slot.obligations.len() as u64);
            for obligation in slot.obligations.iter() {
                writer.text(&obligation.id).text(obligation.scope.as_str());
                if let crate::obligation::ObligationScope::Horizon { horizon } = obligation.scope {
                    writer.word(u64::from(horizon));
                }
                writer
                    .text(obligation.kind.as_str())
                    .text(obligation.status.as_str())
                    .text(&obligation.description);
                match &obligation.source {
                    AssumptionSource::UserDeclared => {
                        writer.word(1);
                    }
                    AssumptionSource::AlgorithmDefault { algorithm } => {
                        writer.word(2).text(algorithm);
                    }
                    AssumptionSource::Artifact => {
                        writer.word(3);
                    }
                    AssumptionSource::Derived { from } => {
                        writer.word(4).text(from);
                    }
                }
                for text in [&obligation.required_check, &obligation.evidence] {
                    match text {
                        Some(text) => {
                            writer.word(1).text(text);
                        }
                        None => {
                            writer.word(0);
                        }
                    }
                }
                match &obligation.assumption {
                    Some(assumption) => {
                        writer.word(1);
                        assumption.encode(writer);
                    }
                    None => {
                        writer.word(0);
                    }
                }
            }
        }
    }
}

/// Hash the claim premises consumed by checked composition after quotation and admission.
/// The identity binds units, assumption slots, identification mass and domain labels;
/// it does not authenticate the source of the supplied claim.
#[must_use]
pub fn composition_claim_premises_identity(claim: &ClaimEnvelope) -> SemanticDigest {
    use crate::reasoning::SlotAvailability;
    let mut writer = CanonicalWriter::new("antecedent.composition.claim_premises.v1");
    encode_parent_premises(&mut writer, claim);
    match &claim.reasoning.identification {
        SlotAvailability::Unavailable { .. } => {
            writer.word(0);
        }
        SlotAvailability::Available(slot) => {
            writer
                .word(1)
                .text(slot.status.as_str())
                .real(slot.identified_mass)
                .real(slot.unidentified_mass)
                .real(slot.unevaluable_mass)
                .real(slot.incomplete_search_mass)
                .word(u64::from(slot.full_mass_scope))
                .word(u64::from(slot.search_capped));
        }
    }
    writer
        .text(claim.domains.identification.as_str())
        .text(claim.domains.support.as_str())
        .text(claim.domains.evaluated.as_str());
    match claim.execution {
        Some(id) => {
            writer.word(1).bytes(id.as_bytes());
        }
        None => {
            writer.word(0);
        }
    }
    SemanticDigest::from_bytes(writer.finish())
}

/// Check and evaluate a declared linear combination of point-valued world claims.
///
/// Inputs are paired with contrast terms by their typed world identity, not by the
/// order supplied. The target's assumptions and any history-indexed law must match
/// each witnessed world target. History-indexed target laws are not supported by
/// this bounded verifier. Every parent must be a finite point over the same
/// observation, data snapshot, outcome units and assumptions slot. This operation
/// does not derive intervals or apply a coverage record. To propagate uncertainty,
/// use the typed [`crate::joint_law_rule::check_propagation`] and joint-law
/// propagation APIs with an appropriate dependence declaration. Callers must
/// verify parent claim authenticity separately.
///
/// # Errors
/// Returns a typed error when the target, witness, parent coordinates, reasoning,
/// numerical operands or result fail verification, or the caller's limits are
/// exceeded. See [`CheckedWorldPointLimits`] for caller-selected bounds. Cancellation
/// is checked between bounded stages; individual canonical hashes and sorts do not poll.
#[allow(
    clippy::too_many_lines,
    reason = "The bounded composition keeps its dependent validation steps together."
)]
pub fn compose_checked_world_points(
    target: &ScientificTarget,
    parents: &[WorldPointParent<'_>],
    limits: &CheckedWorldPointLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<CheckedWorldPoint, CheckedWorldPointError> {
    if ctx.cancellation.is_cancelled() {
        return Err(CheckedWorldPointError::Cancelled);
    }
    if parents.len() > limits.max_terms {
        return Err(CheckedWorldPointError::TermLimit);
    }
    let TargetForm::Contrast(contrast) = target.form() else {
        return Err(CheckedWorldPointError::ExpectedContrast);
    };
    if target.history_indexed_law().is_some() {
        return Err(CheckedWorldPointError::HistoryLawNotSupported);
    }
    let world_count = contrast.worlds().worlds().len();
    if world_count > limits.max_terms
        || world_count.checked_mul(parents.len()).is_none_or(|n| n > limits.max_comparisons)
    {
        return Err(CheckedWorldPointError::TermLimit);
    }
    if world_count < 2 || parents.len() != world_count {
        return Err(CheckedWorldPointError::ParentCount);
    }
    // These are borrowed metadata checks only. Reject unsupported parents before
    // quoting their contents, so the public error identifies the refusal.
    for parent in parents {
        if ctx.cancellation.is_cancelled() {
            return Err(CheckedWorldPointError::Cancelled);
        }
        if parent.target.history_indexed_law().is_some() {
            return Err(CheckedWorldPointError::HistoryLawNotSupported);
        }
        if !matches!(parent.target.form(), TargetForm::World(_)) {
            return Err(CheckedWorldPointError::WitnessMismatch);
        }
    }
    let premise_quote =
        quote_claim_premises(parents.iter().map(|p| p.claim), parents.len(), limits, ctx)?;
    let lease = admit_composition_targets(
        std::iter::once(target).chain(parents.iter().map(|p| p.target)),
        premise_quote,
        limits,
        ctx,
    )?;
    let mut action_count = 0usize;
    let mut assumption_count = target.assumptions().records().len();
    for world in contrast.worlds().worlds() {
        action_count = action_count
            .checked_add(outer_action_count(world))
            .ok_or(CheckedWorldPointError::TermLimit)?;
    }
    for parent in parents {
        if parent.target.history_indexed_law().is_some() {
            return Err(CheckedWorldPointError::HistoryLawNotSupported);
        }
        let TargetForm::World(world) = parent.target.form() else {
            return Err(CheckedWorldPointError::WitnessMismatch);
        };
        action_count = action_count
            .checked_add(outer_action_count(world))
            .ok_or(CheckedWorldPointError::TermLimit)?;
        assumption_count = assumption_count
            .checked_add(parent.target.assumptions().records().len())
            .ok_or(CheckedWorldPointError::TermLimit)?;
    }
    if action_count > limits.max_world_actions || assumption_count > limits.max_assumption_records {
        return Err(CheckedWorldPointError::TermLimit);
    }
    let terms = contrast.terms();
    if terms.len() != world_count {
        return Err(CheckedWorldPointError::ParentCount);
    }
    let first = parents.first().ok_or(CheckedWorldPointError::ParentCount)?.claim;
    let mut distinct_ids = std::collections::BTreeSet::new();
    if parents.iter().any(|p| !distinct_ids.insert(p.claim.claim_id)) {
        return Err(CheckedWorldPointError::DuplicateParent);
    }
    if parents.iter().any(|p| {
        p.claim.kind != ClaimKind::Point
            || p.claim.value.is_none_or(|value| !value.is_finite())
            || p.claim.identities.observation != first.identities.observation
            || p.claim.identities.data_snapshot != first.identities.data_snapshot
            || p.claim.outcome_units != first.outcome_units
            || p.claim.reasoning.assumptions != first.reasoning.assumptions
    }) {
        return Err(CheckedWorldPointError::IncompatibleParents);
    }
    for parent in parents {
        let TargetForm::World(world) = parent.target.form() else {
            return Err(CheckedWorldPointError::WitnessMismatch);
        };
        if parent.claim.outcome_units.as_deref() != Some(world.binding().outcome().units().name()) {
            return Err(CheckedWorldPointError::IncompatibleParents);
        }
    }
    let mut evidence_count = 0usize;
    let mut evidence_bytes = 0usize;
    for parent in parents {
        evidence_count = evidence_count
            .checked_add(parent.claim.evidence.len())
            .ok_or(CheckedWorldPointError::EvidenceLimit)?;
        if evidence_count > limits.max_evidence_references {
            return Err(CheckedWorldPointError::EvidenceLimit);
        }
        for item in parent.claim.evidence.iter() {
            evidence_bytes = evidence_bytes
                .checked_add(item.len())
                .ok_or(CheckedWorldPointError::EvidenceLimit)?;
            if evidence_bytes > limits.max_evidence_bytes {
                return Err(CheckedWorldPointError::EvidenceLimit);
            }
        }
    }
    let witnessed: Vec<_> = parents.iter().map(|p| p.target.identity()).collect();
    let mut ordered_terms = Vec::with_capacity(terms.len());
    for term in terms {
        let world =
            contrast.worlds().get(term.world).ok_or(CheckedWorldPointError::WorldNotFound)?;
        let expected = ScientificTarget::new(
            TargetForm::World(world.clone()),
            target.assumptions().clone(),
            target.inference(),
        )
        .identity();
        ordered_terms.push((expected, term));
    }
    ordered_terms.sort_by_key(|(identity, _)| *identity);
    let mut used = vec![false; parents.len()];
    let mut ids = Vec::with_capacity(ordered_terms.len());
    let mut evidence = std::collections::BTreeSet::new();
    let mut value = 0.0;
    let mut writer = CanonicalWriter::new("antecedent.checked_world_point.v1");
    writer.bytes(target.identity().as_bytes()).word(ordered_terms.len() as u64);
    for (expected, term) in ordered_terms {
        let candidates: Vec<_> = parents
            .iter()
            .enumerate()
            .filter(|(i, p)| {
                !used[*i] && witnessed[*i] == expected && p.claim.identities.target == witnessed[*i]
            })
            .collect();
        if candidates.len() != 1 {
            return Err(CheckedWorldPointError::WitnessMismatch);
        }
        let (index, parent) = candidates[0];
        used[index] = true;
        let operand = parent.claim.value.ok_or(CheckedWorldPointError::IncompatibleParents)?;
        let contribution = term.coefficient * operand;
        if !contribution.is_finite() {
            return Err(CheckedWorldPointError::NonFiniteArithmetic);
        }
        value += contribution;
        if !value.is_finite() {
            return Err(CheckedWorldPointError::NonFiniteArithmetic);
        }
        ids.push(parent.claim.claim_id);
        writer
            .bytes(parent.claim.claim_id.as_bytes())
            .bytes(witnessed[index].as_bytes())
            .real(term.coefficient)
            .real(operand);
        encode_parent_premises(&mut writer, parent.claim);
        let refs = parent.claim.identities.refs();
        writer.word(refs.len() as u64);
        for identity in refs.iter() {
            writer.text(identity.domain.as_str()).bytes(identity.digest.as_bytes());
        }
        writer.word(parent.claim.evidence.len() as u64);
        for item in parent.claim.evidence.iter() {
            writer.text(item);
            evidence.insert(Arc::clone(item));
        }
    }
    if ctx.cancellation.is_cancelled() {
        return Err(CheckedWorldPointError::Cancelled);
    }
    Ok(CheckedWorldPoint {
        target: Arc::new(target.clone()),
        value,
        parent_ids: ids.into(),
        evidence: evidence.into_iter().collect::<Vec<_>>().into(),
        proof_identity: SemanticDigest::from_bytes(writer.finish()),
        _retained: lease,
    })
}
