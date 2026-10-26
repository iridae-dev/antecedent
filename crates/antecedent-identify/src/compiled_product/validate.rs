//! Coherence checks and comparison helpers of a compiled product.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

pub(super) fn check_coherence(
    estimands: &[IdentifiedEstimand],
    assumptions: &AssumptionSet,
    parts: &CompiledProductParts,
) -> Result<(), ProductRefusal> {
    check_claim_class(parts.outcome, parts.claim_class)?;
    for world in &parts.world_obligations {
        check_claim_class(world.outcome, world.claim_class)?;
    }
    if let Some(assembled) = &parts.assembled {
        check_claim_class(assembled.outcome, assembled.claim_class)?;
    }
    if is_failure_outcome(parts.outcome) && !estimands.is_empty() {
        return Err(ProductRefusal::EstimandBesideNonIdentification { outcome: parts.outcome });
    }
    check_obligation_scope(parts)?;
    if parts.outcome == IdentificationOutcome::Identified {
        if estimands.is_empty() {
            return Err(ProductRefusal::IdentifiedWithoutEstimand);
        }
        if let Some(index) = assumptions
            .entries
            .iter()
            .position(|record| matches!(record.status, AssumptionStatus::Contradicted))
        {
            return Err(ProductRefusal::IdentifiedWithContradictedAssumption { index });
        }
        if let Some(index) = first_unresolved(&parts.unresolved_obligations) {
            return Err(ProductRefusal::IdentifiedWithUnresolvedObligation { index });
        }
    }
    for world in &parts.world_obligations {
        if world.outcome == IdentificationOutcome::Identified {
            if let Some(index) = first_unresolved(&world.unresolved) {
                return Err(ProductRefusal::IdentifiedWithUnresolvedObligation { index });
            }
        }
    }
    if let Some(assembled) = &parts.assembled {
        if assembled.outcome == IdentificationOutcome::Identified {
            if let Some(index) = first_unresolved(&assembled.unresolved) {
                return Err(ProductRefusal::IdentifiedWithUnresolvedObligation { index });
            }
        }
    }
    Ok(())
}

pub(super) fn first_unresolved(obligations: &[UnresolvedObligation]) -> Option<usize> {
    if obligations.is_empty() { None } else { Some(0) }
}

pub(super) fn check_obligation_scope(parts: &CompiledProductParts) -> Result<(), ProductRefusal> {
    for obligation in &parts.world_obligations {
        if obligation.world.is_none() {
            return Err(ProductRefusal::WorldObligationMissingWorld);
        }
    }
    if let Some(assembled) = &parts.assembled {
        if let Some(world) = assembled.world {
            return Err(ProductRefusal::AssembledObligationNamesWorld { world });
        }
    }
    Ok(())
}

pub(super) const fn states_identified_claim(
    outcome: IdentificationOutcome,
    class: IdentificationClaimClass,
) -> bool {
    matches!(outcome, IdentificationOutcome::Identified)
        && matches!(class, IdentificationClaimClass::IdentifiedClaim)
}

pub(super) const fn numerically_feasible(outcome: IdentificationOutcome, value: Option<f64>) -> bool {
    if matches!(outcome, IdentificationOutcome::NumericallyInfeasible) {
        return false;
    }
    match value {
        Some(number) => number.is_finite(),
        None => false,
    }
}

pub(super) const fn outcome_code(outcome: IdentificationOutcome) -> u64 {
    match outcome {
        IdentificationOutcome::Identified => 1,
        IdentificationOutcome::Unidentified => 2,
        IdentificationOutcome::Unsupported => 3,
        IdentificationOutcome::TimedOut => 4,
        IdentificationOutcome::NumericallyInfeasible => 5,
        IdentificationOutcome::Uncalibrated => 6,
    }
}

pub(super) const fn claim_code(class: IdentificationClaimClass) -> u64 {
    match class {
        IdentificationClaimClass::IdentifiedClaim => 1,
        IdentificationClaimClass::ModelConditional => 2,
        IdentificationClaimClass::UnsupportedExtrapolation => 3,
    }
}

pub(super) fn is_bounded_search(diagnostic: &Diagnostic) -> bool {
    let code = diagnostic.code.as_ref();
    code == crate::generalized::CAPPED_COMPLETION_DIAGNOSTIC_CODE
        || code.ends_with(".search_bounded")
        || code == crate::temporal_mag::HISTORY_CAPPED
}

pub(super) fn count(len: usize) -> u64 {
    u64::try_from(len).unwrap_or(u64::MAX)
}

pub(super) fn nth_raw(index: usize) -> u32 {
    u32::try_from(index).unwrap_or(u32::MAX)
}

pub(super) fn same_estimands(left: &[IdentifiedEstimand], right: &[IdentifiedEstimand]) -> bool {
    left.len() == right.len() && left.iter().zip(right).all(|(a, b)| same_estimand(a, b))
}

pub(super) fn same_estimand(left: &IdentifiedEstimand, right: &IdentifiedEstimand) -> bool {
    left.method == right.method
        && left.adjustment_set == right.adjustment_set
        && left.instruments == right.instruments
        && left.mediators == right.mediators
        && left.functional == right.functional
        && same_rd(left.rd_design, right.rd_design)
}

pub(super) const fn same_rd(left: Option<RdDesignParams>, right: Option<RdDesignParams>) -> bool {
    match (left, right) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            a.running_variable.raw() == b.running_variable.raw()
                && a.cutoff.to_bits() == b.cutoff.to_bits()
                && a.bandwidth.to_bits() == b.bandwidth.to_bits()
        }
        (None, Some(_)) | (Some(_), None) => false,
    }
}
