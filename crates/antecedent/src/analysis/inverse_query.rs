//! Generalized finite inverse decision query.
//!
//! "Which of these declared actions satisfy these constraints?" over a finite,
//! ordered action grid. Target-mean, target-quantile and probability-threshold
//! constraints all run through the one typed functional engine on forward claims
//! that actually supply the law each needs: a quantile or a probability needs an
//! aligned joint law, a mean-only claim answers an affine target mean and nothing
//! else, and independent marginals refuse a nonlinear utility. The finite
//! enumeration stays available as [`finite_enumeration_baseline`] and agrees with the
//! engine on target-mean cases.
//!
//! Feasibility is reported per action in distinct fields (point, interval region,
//! identified set, all scenarios, posterior probability); none is derived from
//! another. A continuous sample never claims global feasibility. Nothing here makes a
//! coverage or calibration claim.
//!
//! The enumeration query of [`super::InverseQuery`] is a separate, unchanged
//! route; this module's names are reached by path
//! (`antecedent::analysis::inverse_query`).
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_io::IoError;

pub use antecedent_design::decision_contract::{
    DecisionContract, DecisionFunctional, StructuralPolicy, Tail,
};
pub use antecedent_design::decision_eval::MeanSource;
pub use antecedent_design::decision_structural::{AtomEvidence, StructuralAtom};
pub use antecedent_design::inverse_query::{
    ActionReport, Comparison, ConstraintValue, EnumeratedForwardValue, ExistenceClaim,
    FeasibilityStatus, ForwardClaim, ForwardEvidence, GridScope, INVERSE_QUERY_SCOPE_NOTE,
    IdentifiedSet, IntervalRegion, InverseConstraint, InverseQuery, InverseQueryError,
    InverseResult, MemberStatus, PosteriorFeasibility, SelectionOutcome, SelectionRule,
};
pub use antecedent_design::inverse_query_artifact::{
    INVERSE_QUERY_ARTIFACT_FEATURE, INVERSE_QUERY_ARTIFACT_VERSION, INVERSE_QUERY_INFERENCE_CLAIM,
    InverseQueryArtifact, InverseQueryIdentity, InverseQueryWire, InverseResultWire,
    MAX_INVERSE_QUERY_ARTIFACT_BYTES, MAX_INVERSE_QUERY_CLAIMS,
};
pub use antecedent_io::distribution_artifact::DistributionArtifact;

/// The finite-enumeration baseline: classify each enumerated action by its
/// forward mean against `target`, within `tolerance`. An action off the
/// evaluated grid is `Unevaluated`, never interpolated; an unsupported one is
/// `Unsupported`. It uses no distribution engine, so it is the independent
/// reference the generalized query must agree with on target-mean cases.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_design::inverse_query::finite_enumeration_baseline`].
///
/// # Errors
/// A non-finite target, tolerance or forward value, or a negative tolerance.
#[deprecated(
    since = "3.1.0",
    note = "use the compiled InverseQueryDecision route, which calls this kernel function with the same inputs"
)]
pub fn finite_enumeration_baseline(
    points: &[antecedent_design::inverse_query::EnumeratedForwardValue],
    target: f64,
    comparison: antecedent_design::inverse_query::Comparison,
    tolerance: f64,
) -> Result<
    Vec<(String, antecedent_design::inverse_query::FeasibilityStatus)>,
    antecedent_design::inverse_query::InverseQueryError,
> {
    antecedent_design::inverse_query::finite_enumeration_baseline(
        points, target, comparison, tolerance,
    )
}

/// Evaluate an inverse query on forward evidence.
///
/// Each (action, claim, constraint) is answered by the shared functional
/// engine; the module only compares its value with the target. Per-action
/// refusals of unsupported or absent coordinates become that action's
/// `Unsupported` or `Unevaluated` status; a joint-law requirement the claim
/// cannot meet, a mean-only claim asked for a quantile or probability, an
/// outcome-law meaning mismatch and an invalid contract refuse the query.
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_design::inverse_query::evaluate_inverse_query`].
///
/// # Errors
/// Refuses an invalid grid, constraint, tolerance or scenario declaration, a
/// query with no forward evidence, and any engine refusal on a point or
/// interval-endpoint claim (see `InverseQueryError::Engine`).
#[deprecated(
    since = "3.1.0",
    note = "use the compiled InverseQueryDecision route, which calls this kernel function with the same inputs"
)]
pub fn evaluate_inverse_query(
    query: &antecedent_design::inverse_query::InverseQuery,
    evidence: &antecedent_design::inverse_query::ForwardEvidence,
) -> Result<
    antecedent_design::inverse_query::InverseResult,
    antecedent_design::inverse_query::InverseQueryError,
> {
    antecedent_design::inverse_query::evaluate_inverse_query(query, evidence)
}

/// Compute `functional` of the named action's utility on an aligned joint
/// distribution artifact.
///
/// The utility is evaluated per aligned draw exactly as in `evaluate_contract`
/// (joint alignment, coordinates located by `ScientificQuantity` coordinate
/// identity, distribution meaning checked, weights normalized to sum one). The
/// functional is then taken of the resulting finite weighted law:
///
/// - `Expectation` and `ExpectedUtility`: the weighted mean `sum w_i x_i`.
/// - `Variance`: the weighted population variance `sum w_i (x_i - mean)^2` of
///   the finite law, not an unbiased sample variance.
/// - `Probability { threshold, tail }`: `Lower` is `P(x <= threshold)` and
///   `Upper` is `P(x >= threshold)`; both include the atom at the threshold.
/// - `Quantile { p }`: the left inverse `F^-1(p) = inf { x : F(x) >= p }` of the
///   weighted empirical CDF, the smallest value whose cumulative weight reaches
///   `p`. Cumulative weights are compared to `p` with a relative roundoff tolerance of
///   eight machine epsilons so a level that falls exactly on an atom boundary selects that atom.
/// - `TailExpectation { p, tail }`: the fractional-boundary tail mean. Draws are
///   sorted ascending for `Lower` and descending for `Upper`; weights accumulate
///   until mass `p` is reached, taking only the needed fraction of the boundary
///   atom, and the weighted sum is divided by `p`. This is the explicitly
///   selected tail rule.
///
/// `standard_error` is `Some(0.0)` for an exact finite law; Monte Carlo
/// error is withheld for sampled laws (`None`).
///
/// Deprecated since 3.1.0 in favour of its compiled route, which the deprecation note
/// names. This forwards to [`antecedent_design::decision_eval::evaluate_functional`].
///
/// # Errors
/// Refuses an invalid contract, an unknown action
/// (`Contract(UnknownAction)`), a level `p` outside `(0, 1)` or a non-finite
/// threshold (`Contract(InvalidParameter("functional"))`), non-joint draws, a
/// missing or masked coordinate, a source meaning that cannot answer an
/// outcome-law input, and a non-finite utility.
#[deprecated(
    since = "3.1.0",
    note = "use the compiled DecisionFunctionalValue route, which calls this kernel function with the same inputs"
)]
pub fn evaluate_functional(
    contract: &antecedent_design::DecisionContract,
    action_id: &str,
    functional: antecedent_design::DecisionFunctional,
    source: &antecedent_io_core::distribution_artifact::DistributionArtifact,
) -> Result<
    antecedent_design::decision_eval::FunctionalValue,
    antecedent_design::decision_eval::DecisionEvalError,
> {
    antecedent_design::decision_eval::evaluate_functional(contract, action_id, functional, source)
}

/// Evaluate and seal a query, its evidence and its result as a durable artifact.
///
/// # Errors
///
/// As [`evaluate_inverse_query`], plus more embedded claims than [`MAX_INVERSE_QUERY_CLAIMS`].
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

/// Analytic conditional Gaussian law used by composed intervention sources.
pub use antecedent_design::gaussian_forward_law::GaussianForwardLaw;
