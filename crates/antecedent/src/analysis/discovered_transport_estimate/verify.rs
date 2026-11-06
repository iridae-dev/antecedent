//! Replay and verify exported discovered-transport estimates.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    ESTIMATE_FORMAT, EstimateWire, ExactEvaluationLimits, ExecutionContext, IoError,
    MemberEstimate, MemberOutcome, PreparedStudy, TransportAnswer, TransportGridState, VariableId,
    contrast_of, identity_of, summarise,
};

/// What replaying an exported estimate established.
#[derive(Clone, Debug, PartialEq)]
pub struct VerifiedEstimate {
    /// The answer recomputed from the consumed member artifacts.
    pub answer: TransportAnswer,
    /// Identity of the structural report the estimate was made over.
    pub report_identity: [u8; 32],
    /// Members that were consumed, by their index in the report's envelope.
    pub members: Vec<usize>,
}

/// Replay an exported estimate: consume every member's transport grid with its embedded proof
/// and tables, recompute the mean at every level, check the stored means, expression-free
/// summary and identity, and return the recomputed answer.
///
/// # Errors
///
/// A malformed artifact, a member grid that does not consume, a mean or an answer that does not
/// replay, or an identity that does not match.
pub fn verify_discovered_transport_estimate(
    bytes: &[u8],
    limits: ExactEvaluationLimits,
    ctx: &ExecutionContext,
) -> Result<VerifiedEstimate, IoError> {
    verify_estimate_with_member_count(bytes, limits, ctx, None)
}

pub(crate) fn verify_trial_member_estimate(
    bytes: &[u8],
    limits: ExactEvaluationLimits,
    ctx: &ExecutionContext,
    expected_members: usize,
) -> Result<VerifiedEstimate, IoError> {
    verify_estimate_with_member_count(bytes, limits, ctx, Some(expected_members))
}

fn verify_estimate_with_member_count(
    bytes: &[u8],
    limits: ExactEvaluationLimits,
    ctx: &ExecutionContext,
    expected_members: Option<usize>,
) -> Result<VerifiedEstimate, IoError> {
    let mismatch = |what: &str| IoError::Convert(format!("discovered transport estimate: {what}"));
    let wire: EstimateWire = antecedent_io::from_cbor(bytes)?;
    if wire.format != ESTIMATE_FORMAT {
        return Err(mismatch("unsupported format"));
    }
    if expected_members.is_some_and(|expected| wire.members.len() != expected) {
        return Err(mismatch("trial refinement member count differs before replay"));
    }
    let report_identity: [u8; 32] =
        wire.report_identity.as_slice().try_into().map_err(|_| mismatch("report identity"))?;
    let stored: [u8; 32] =
        wire.identity.as_slice().try_into().map_err(|_| mismatch("estimate identity"))?;
    let mut outcomes = Vec::with_capacity(wire.members.len());
    for member in &wire.members {
        let (_, result) =
            PreparedStudy::<TransportGridState>::consume(&member.artifact, limits, ctx)?;
        let outcome = VariableId::from_raw(member.outcome);
        let means = result
            .points()
            .iter()
            .map(|point| {
                point
                    .distribution()
                    .ok_or_else(|| mismatch("a member grid has an unavailable point"))?
                    .mean(outcome)
                    .map_err(|e| mismatch(&e.to_string()))
            })
            .collect::<Result<Vec<f64>, _>>()?;
        if means != member.means {
            return Err(mismatch("a member mean does not replay"));
        }
        outcomes.push(MemberOutcome::Estimated(MemberEstimate {
            expression: member.expression.clone(),
            contrast: contrast_of(&means),
            means,
            contrast_interval: None,
            artifact: Vec::new(),
            outcome,
        }));
    }
    let answer = summarise(&wire.levels, &outcomes, None)
        .map_err(|_| mismatch("the members do not summarise"))?;
    if answer.kind.label() != wire.kind {
        return Err(mismatch("the answer kind does not replay"));
    }
    let targets: Vec<VariableId> =
        wire.selection_targets.iter().copied().map(VariableId::from_raw).collect();
    let rebuilt = identity_of(&report_identity, &wire.levels, &targets, &wire, &answer);
    if rebuilt != stored {
        return Err(mismatch("the identity does not replay"));
    }
    Ok(VerifiedEstimate {
        answer,
        report_identity,
        members: wire.members.iter().map(|m| m.index as usize).collect(),
    })
}
