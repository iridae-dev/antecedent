//! F7: generalized finite inverse decision query.
//!
//! "Which of these declared actions satisfy these constraints?" over a finite,
//! ordered action grid. Target-mean, target-quantile and probability-threshold
//! constraints all run through the one typed functional engine on forward claims
//! that actually supply the law each needs: a quantile or a probability needs an
//! aligned joint law, a mean-only claim answers an affine target mean and nothing
//! else, and independent marginals refuse a nonlinear utility. The 2.2 finite
//! enumeration stays available as [`finite_enumeration_baseline`] and agrees with the
//! engine on target-mean cases.
//!
//! Feasibility is reported per action in distinct fields (point, interval region,
//! identified set, all scenarios, posterior probability); none is derived from
//! another. A continuous sample never claims global feasibility. Nothing here makes a
//! coverage or calibration claim.
//!
//! The 2.2 enumeration query of [`super::InverseQuery`] is a separate, unchanged
//! route; this module's names are reached by path
//! (`antecedent::analysis::inverse_query`).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_io::IoError;

pub use antecedent_design::decision_contract::{
    DecisionContract, DecisionFunctional, StructuralPolicy, Tail,
};
pub use antecedent_design::decision_eval::{MeanSource, evaluate_functional};
pub use antecedent_design::decision_structural::{AtomEvidence, StructuralAtom};
pub use antecedent_design::inverse_query::*;
pub use antecedent_design::inverse_query_artifact::{
    INVERSE_QUERY_ARTIFACT_FEATURE, INVERSE_QUERY_ARTIFACT_VERSION, INVERSE_QUERY_INFERENCE_CLAIM,
    InverseQueryArtifact, InverseQueryIdentity, InverseQueryWire, InverseResultWire,
    MAX_INVERSE_QUERY_ARTIFACT_BYTES, MAX_INVERSE_QUERY_CLAIMS,
};
pub use antecedent_io::distribution_artifact::DistributionArtifact;

/// Evaluate a query on forward evidence through the shared functional engine.
///
/// Equivalent to [`evaluate_inverse_query`].
///
/// # Errors
///
/// An invalid grid, constraint, tolerance or scenario declaration, no forward
/// evidence, and any engine refusal on a point or interval-endpoint claim
/// (`decision_evaluation.joint_law_required`,
/// `decision_evaluation.mean_source_insufficient`, ...).
pub fn evaluate(
    query: &InverseQuery,
    evidence: &ForwardEvidence,
) -> Result<InverseResult, InverseQueryError> {
    evaluate_inverse_query(query, evidence)
}

/// Evaluate and seal a query, its evidence and its result as a durable artifact.
///
/// # Errors
///
/// As [`evaluate`], plus more embedded claims than [`MAX_INVERSE_QUERY_CLAIMS`].
pub fn seal(
    query: InverseQuery,
    evidence: ForwardEvidence,
) -> Result<InverseQueryArtifact, InverseQueryError> {
    InverseQueryArtifact::new(query, evidence)
}

/// Serialize an artifact through the bounded sectioned container.
///
/// # Errors
///
/// An empty id, an encoding failure or an oversized payload.
pub fn export(artifact: &InverseQueryArtifact, artifact_id: &str) -> Result<Vec<u8>, IoError> {
    artifact.to_bytes(artifact_id)
}

/// Consume an artifact by re-evaluation through [`evaluate_inverse_query`]; `expected`
/// is an identity the consumer retained independently of the bytes.
///
/// # Errors
///
/// Corruption, another major version, a changed identity or result table, a global
/// claim on a continuous sample, or any evaluation refusal.
pub fn consume(
    bytes: &[u8],
    expected: Option<&InverseQueryIdentity>,
) -> Result<InverseQueryArtifact, IoError> {
    InverseQueryArtifact::from_bytes(bytes, expected)
}
