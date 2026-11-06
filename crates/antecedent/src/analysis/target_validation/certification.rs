//! Runtime ownership and certification of refutation reports.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, CausalError, RefutationReport, RefutationTarget, StudyResult, TargetCertificate,
    TargetCertificationError, TargetFamily, TargetRefutationOutcome, certify_target_refutations,
    refuse_scalar_reports_for_target, result_quadrant, result_refutation_target,
    result_target_family,
};

/// A pooled result is a new target, not the first completed-table result.
/// Do not carry its first-table checks forward as evidence for that target.
pub(crate) fn reset_pooled_refutations(result: &mut StudyResult) {
    if !result.refutations.is_empty() {
        result.view.refutations.clear();
        result.view.diagnostics.push(antecedent_core::Diagnostic::new(
            "refute.imputation.withheld",
            antecedent_core::DiagnosticKind::Scientific,
            antecedent_core::DiagnosticSeverity::Info,
            "completed-table refutations are withheld for the pooled estimate",
        ));
    }
    result.authority = crate::result::RuntimeRefutationAuthority::Fresh;
}

// This is an in-process owner state, never reconstructed from serialized fields.
// The result view invalidates it before giving a mutable reference to public
// target/identification/report state; no recursive caller-owned query is hashed.
#[allow(
    clippy::unnecessary_wraps,
    reason = "issuance callers use a shared fallible sealing interface"
)]
pub(in crate::analysis) fn seal_runtime_refutations(
    result: &mut StudyResult,
    origin: crate::result::RuntimeRefutationOrigin,
) -> Result<(), CausalError> {
    if result.lowered_program_identity().is_some() {
        result.authority =
            crate::result::RuntimeRefutationAuthority::Sealed { origin, owner: Arc::new(()) };
    }
    Ok(())
}

pub(in crate::analysis) fn validate_runtime_refutation_proof(
    result: &StudyResult,
) -> Result<(), CausalError> {
    match &result.authority {
        crate::result::RuntimeRefutationAuthority::Sealed { .. } => Ok(()),
        crate::result::RuntimeRefutationAuthority::Fresh if result.refutations.is_empty() => Ok(()),
        _ => Err(CausalError::Conflict {
            what: "refutations",
            detail: "checked result was modified or reports lack an issued binding",
        }),
    }
}

/// Refuse refutations `result` carries that its own program did not produce, when its target
/// is not the scalar average effect.
///
/// Provenance decides, not the report's name: a report the result's lowered program produced
/// during its own execution (stamped by [`super::stamp_route_refutations`]) speaks to that program's
/// target and passes, whatever suite produced it. Any other report (built outside a compiled
/// execution, or produced by another program) can only be a scalar average-effect refutation,
/// and it does not certify another target. A result no lowered program produced defines no
/// target to compare with and is left to the claim's own identity checks.
///
/// # Errors
///
/// [`TargetCertificationError::ScalarRefuterForOtherTarget`] for the first such report on a
/// result whose target is not the scalar average effect.
pub(crate) fn refuse_scalar_refutations_for_result(
    result: &StudyResult,
) -> Result<(), TargetCertificationError> {
    use crate::result::{RuntimeRefutationAuthority, RuntimeRefutationOrigin};
    match &result.authority {
        RuntimeRefutationAuthority::Invalidated => {
            return Err(TargetCertificationError::TargetMismatch {
                refuter: Arc::from("modified result"),
                claim_target: "executed result".to_owned(),
                report_target: "modified target or report vector".to_owned(),
                next_attempt: antecedent_validate::target_native::TARGET_NATIVE_NEXT_ATTEMPT,
            });
        }
        RuntimeRefutationAuthority::Fresh if !result.refutations.is_empty() => {
            return Err(TargetCertificationError::TargetMismatch {
                refuter: Arc::from("unissued refutation"),
                claim_target: "executed result".to_owned(),
                report_target: "caller-declared report".to_owned(),
                next_attempt: antecedent_validate::target_native::TARGET_NATIVE_NEXT_ATTEMPT,
            });
        }
        RuntimeRefutationAuthority::Sealed { origin: RuntimeRefutationOrigin::Route, .. }
            if result.response().is_some()
                && result.structural_response.is_some()
                && !result.refutations.is_empty() =>
        {
            return Err(TargetCertificationError::TargetMismatch {
                refuter: Arc::from("class response"),
                claim_target: "whole-result target".to_owned(),
                report_target: "per-completion route checks".to_owned(),
                next_attempt: antecedent_validate::target_native::TARGET_NATIVE_NEXT_ATTEMPT,
            });
        }
        _ => {}
    }
    // Scope metadata is descriptive until the checked-operation authority is
    // retained and verified. A same-program stamp cannot make it authoritative.
    if let Some(report) = result.refutations.iter().find(|report| report.scope.is_some()) {
        return Err(TargetCertificationError::TargetMismatch {
            refuter: Arc::clone(&report.refuter),
            claim_target: "whole-result target".to_owned(),
            report_target: "unverified descriptive scope".to_owned(),
            next_attempt: antecedent_validate::target_native::TARGET_NATIVE_NEXT_ATTEMPT,
        });
    }
    let family = result_target_family(result);
    if family == TargetFamily::ScalarAverageEffect {
        return Ok(());
    }
    let Some(program) = result.lowered_identity else {
        return Ok(());
    };
    let program = *program.as_bytes();
    let foreign: Vec<RefutationReport> =
        result.refutations.iter().filter(|report| !report.stamped_by(&program)).cloned().collect();
    // The family alone decides; the labels and source only name the claim in the error.
    let claim = RefutationTarget::new(family, result_quadrant(result), ["claim"], program)
        .map_err(|_| TargetCertificationError::ScalarRefuterForOtherTarget {
            refuter: Arc::from("unknown"),
            claim_family: family,
            next_attempt: antecedent_validate::target_native::TARGET_NATIVE_NEXT_ATTEMPT,
        })?;
    refuse_scalar_reports_for_target(&claim, &foreign)
}

/// Identify engine-issued structural route reports that may be exported only
/// as descriptive evidence with an unresolved replay dependency.
pub(in crate::analysis) fn descriptive_route_report_source(
    result: &StudyResult,
) -> Option<[u8; 32]> {
    use crate::result::{RuntimeRefutationAuthority, RuntimeRefutationOrigin};
    if !matches!(
        &result.authority,
        RuntimeRefutationAuthority::Sealed {
            origin: RuntimeRefutationOrigin::Route | RuntimeRefutationOrigin::ScopedCompletion,
            ..
        }
    ) || result.response().is_none()
        || result.structural_response.is_none()
        || result.refutations.is_empty()
    {
        return None;
    }
    let program = *result.lowered_identity?.as_bytes();
    result.refutations.iter().all(|report| report.stamped_by(&program)).then_some(program)
}

/// Bind target-native `reports` to `result`'s claim.
///
/// # Errors
///
/// [`TargetCertificationError::TargetMismatch`] for a report computed on another target
/// (another lowered program, family, quadrant or coordinate set);
/// [`TargetCertificationError::ScalarRefuterForOtherTarget`] when the result carries scalar
/// average-effect refutations for a target that is not that scalar.
pub fn certify_result_refutations(
    result: &StudyResult,
    outcome: &TargetRefutationOutcome,
) -> Result<TargetCertificate, TargetCertificationError> {
    refuse_scalar_refutations_for_result(result)?;
    let owner = match &result.authority {
        crate::result::RuntimeRefutationAuthority::Sealed { owner, .. } => Some(owner),
        _ => None,
    };
    if !owner.zip(outcome.owner.as_ref()).is_some_and(|(a, b)| Arc::ptr_eq(a, b)) {
        return Err(TargetCertificationError::TargetMismatch {
            refuter: Arc::from("target operation"),
            claim_target: "executed result".to_owned(),
            report_target: "unissued, modified, or foreign target outcome".to_owned(),
            next_attempt: antecedent_validate::target_native::TARGET_NATIVE_NEXT_ATTEMPT,
        });
    }
    let claim = result_refutation_target(result).map_err(|error| {
        TargetCertificationError::TargetMismatch {
            refuter: Arc::from("target operation"),
            claim_target: format!("undefined ({error})"),
            report_target: String::new(),
            next_attempt: antecedent_validate::target_native::TARGET_NATIVE_NEXT_ATTEMPT,
        }
    })?;
    certify_target_refutations(&claim, &outcome.run.reports)
}
