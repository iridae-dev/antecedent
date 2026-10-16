//! Typed refusals of the checked transport artifacts.
//!
//! The enums are plain data over identification errors, so they sit below the
//! estimators with [`IoError`](crate::IoError), which wraps each of them.
//! Every artifact module re-exports its own enum at its established path.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::TransportOutcomeKind;
use antecedent_identify::{IdentificationError, admg_conditional_refusal};

/// Why a z-transport artifact was refused by a consumer check. Callers match the
/// kind; they never parse the message.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ZTransportArtifactError {
    /// The feature marker, interval status or result shape is not this format's.
    #[error("unsupported semantics: {0}")]
    UnsupportedSemantics(&'static str),
    /// A limit the artifact recorded exceeds the consumer's, or a stored
    /// collection exceeds the consumer's bound.
    #[error("consumer limit exceeded: {0}")]
    LimitsExceeded(&'static str),
    /// The stored derivation does not verify against the stored graph and query.
    #[error("proof does not verify: {0}")]
    ProofMismatch(IdentificationError),
    /// The verified derivation does not bind to the stored catalog.
    #[error("catalog binding failed: {0}")]
    CatalogBinding(IdentificationError),
    /// A stored law is not a valid exact law or law set.
    #[error("invalid law: {0}")]
    LawInvalid(String),
    /// The stored program does not match the checked proof and binding.
    #[error("program mismatch: {0}")]
    ProgramMismatch(&'static str),
    /// The recomputed point differs from the stored point.
    #[error("point result does not replay")]
    PointMismatch,
    /// The premises digest does not match the stored premises.
    #[error("premises digest mismatch")]
    PremisesMismatch,
}

impl ZTransportArtifactError {
    /// The stable transport outcome kind this refusal maps to.
    #[must_use]
    pub const fn transport_outcome_kind(&self) -> TransportOutcomeKind {
        match self {
            Self::ProofMismatch(error) | Self::CatalogBinding(error) => {
                error.transport_outcome_kind()
            }
            Self::LimitsExceeded(_) => TransportOutcomeKind::BudgetCancel,
            Self::UnsupportedSemantics(_)
            | Self::LawInvalid(_)
            | Self::ProgramMismatch(_)
            | Self::PointMismatch
            | Self::PremisesMismatch => TransportOutcomeKind::InvalidInput,
        }
    }
}

/// Why an mz-transport artifact was refused. Callers match the kind.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum MzTransportArtifactError {
    /// The feature marker or result shape is not this format's.
    #[error("unsupported semantics: {0}")]
    UnsupportedSemantics(&'static str),
    /// A stored limit or collection exceeds the consumer's bound.
    #[error("consumer limit exceeded: {0}")]
    LimitsExceeded(&'static str),
    /// The re-run decision did not reproduce the stored derivation.
    #[error("proof does not verify: {0}")]
    ProofMismatch(IdentificationError),
    /// The verified derivation does not bind to the stored catalog.
    #[error("catalog binding failed: {0}")]
    CatalogBinding(IdentificationError),
    /// The stored source-specific bindings differ from the re-derived ones.
    #[error("source bindings do not match the checked binding")]
    BindingMismatch,
    /// A stored law is not a valid law or law set.
    #[error("invalid law: {0}")]
    LawInvalid(String),
    /// A recomputed point differs from the stored point.
    #[error("point result does not replay")]
    PointMismatch,
    /// A recomputed contrast differs from the stored contrast.
    #[error("contrast does not replay")]
    ContrastMismatch,
    /// The stored interval bookkeeping contradicts the licensing decision or
    /// does not recompute bit for bit from its seed and replicate spec.
    #[error("uncertainty bookkeeping does not check: {0}")]
    UncertaintyMismatch(&'static str),
    /// The premises digest does not match the stored premises.
    #[error("premises digest mismatch")]
    PremisesMismatch,
    /// The data-identity digest does not match the stored snapshot identities.
    #[error("data identity digest mismatch")]
    DataIdentityMismatch,
    /// The caller's variable names are not the names the artifact was built under.
    #[error("variable names do not match the verified name mapping")]
    NamesMismatch,
}

impl MzTransportArtifactError {
    /// The frozen `(reason code, mz_transport.* detail)` pair of this failure, when
    /// the reason-code registry declares one: a proof or binding failure carries the
    /// pair of the identification error it wraps. Other consumer checks are typed
    /// kinds without a record row.
    #[must_use]
    pub fn refusal(&self) -> Option<(&'static str, &'static str)> {
        match self {
            Self::ProofMismatch(inner) | Self::CatalogBinding(inner) => {
                antecedent_identify::mz_transport_refusal(inner)
            }
            _ => None,
        }
    }
}

/// Why a mixed-source artifact was refused. Callers match the kind.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum MixedSourceArtifactError {
    /// The feature marker, claim or result shape is not this format's.
    #[error("unsupported semantics: {0}")]
    UnsupportedSemantics(&'static str),
    /// A stored limit or collection exceeds the consumer's bound.
    #[error("consumer limit exceeded: {0}")]
    LimitsExceeded(&'static str),
    /// The re-run decision did not reproduce the stored derivation.
    #[error("proof does not verify: {0}")]
    ProofMismatch(IdentificationError),
    /// The verified derivation does not bind to the stored catalog.
    #[error("catalog binding failed: {0}")]
    CatalogBinding(IdentificationError),
    /// The stored source-named bindings differ from the re-derived ones.
    #[error("source bindings do not match the checked binding")]
    BindingMismatch,
    /// A stored law is not a valid law or law set.
    #[error("invalid law: {0}")]
    LawInvalid(String),
    /// A recomputed point differs from the stored point.
    #[error("point result does not replay")]
    PointMismatch,
    /// The premises digest does not match the stored premises.
    #[error("premises digest mismatch")]
    PremisesMismatch,
    /// The data-identity digest does not match the stored snapshot identities.
    #[error("data identity digest mismatch")]
    DataIdentityMismatch,
    /// The caller's variable names are not the names the artifact was built under.
    #[error("variable names do not match the verified name mapping")]
    NamesMismatch,
}

/// Why an ADMG conditional transport artifact was refused. Every kind carries
/// its registered `(reason code, admg_transport.* detail)` pair ([`Self::refusal`]).
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum AdmgConditionalArtifactError {
    /// The feature marker, result shape or name mapping is not this format's.
    #[error("admg_transport.invalid_artifact: unsupported semantics: {0}")]
    UnsupportedSemantics(&'static str),
    /// Another format version, refused before the payload is interpreted.
    #[error("admg_transport.invalid_artifact: unsupported artifact version {version}")]
    UnsupportedVersion {
        /// The stored version.
        version: u32,
    },
    /// The artifact claims an interval or carries counted laws.
    #[error("admg_transport.interval_withheld: the route publishes exact points only")]
    IntervalWithheld,
    /// A stored limit or collection exceeds the consumer's bound.
    #[error("admg_transport.consumer_limits: consumer limit exceeded: {0}")]
    LimitsExceeded(&'static str),
    /// The stored proof does not check, or the re-decision fails.
    #[error("proof does not verify: {0}")]
    ProofMismatch(IdentificationError),
    /// The re-decided derivation, re-bound leaves or a recomputed point differ.
    #[error("admg_transport.replay_mismatch: {0} does not replay")]
    ReplayMismatch(&'static str),
    /// A stored law is not a valid law or law set.
    #[error("admg_transport.invalid_artifact: invalid law: {0}")]
    LawInvalid(String),
    /// The premises digest does not match the stored premises.
    #[error("admg_transport.premises_mismatch: premises digest mismatch")]
    PremisesMismatch,
    /// The data-identity digest does not match the stored snapshot identities.
    #[error("admg_transport.data_identity_mismatch: data identity digest mismatch")]
    DataIdentityMismatch,
    /// The caller's variable names are not the verified name mapping.
    #[error("admg_transport.invalid_artifact: variable names do not match the verified mapping")]
    NamesMismatch,
}

impl AdmgConditionalArtifactError {
    /// The registered `(reason code, admg_transport.* detail)` pair of this failure.
    #[must_use]
    pub fn refusal(&self) -> (&'static str, &'static str) {
        use antecedent_core::reason_code;
        match self {
            Self::UnsupportedSemantics(_)
            | Self::UnsupportedVersion { .. }
            | Self::LawInvalid(_)
            | Self::NamesMismatch => {
                (reason_code!("invalid_argument"), "admg_transport.invalid_artifact")
            }
            Self::IntervalWithheld => {
                (reason_code!("cell_not_licensed"), "admg_transport.interval_withheld")
            }
            Self::LimitsExceeded(_) => {
                (reason_code!("route_not_supported"), "admg_transport.consumer_limits")
            }
            Self::ProofMismatch(inner) => admg_conditional_refusal(inner).unwrap_or((
                reason_code!("transport_not_certified"),
                "admg_transport.invalid_derivation",
            )),
            Self::ReplayMismatch(_) => {
                (reason_code!("transport_not_certified"), "admg_transport.replay_mismatch")
            }
            Self::PremisesMismatch => {
                (reason_code!("transport_not_certified"), "admg_transport.premises_mismatch")
            }
            Self::DataIdentityMismatch => {
                (reason_code!("transport_not_certified"), "admg_transport.data_identity_mismatch")
            }
        }
    }
}

/// Why a scenario artifact was refused.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum TransportScenarioArtifactError {
    /// The feature marker is not this format's.
    #[error("unsupported semantics: {0}")]
    UnsupportedSemantics(&'static str),
    /// A recorded limit or stored collection exceeds the consumer's bound.
    #[error("consumer limit exceeded: {0}")]
    LimitsExceeded(&'static str),
    /// The premises digest does not match the stored premises.
    #[error("premises digest mismatch")]
    PremisesMismatch,
    /// The recomputed report differs from the stored report.
    #[error("scenario report does not replay")]
    ReportMismatch,
    /// The stored laws disagree with the recorded provider or its samples.
    #[error("scenario provider mismatch: {0}")]
    ProviderMismatch(&'static str),
    /// The stored catalog, laws, provider or sample summaries do not match the
    /// data-identity digest.
    #[error("scenario data identity mismatch")]
    DataIdentityMismatch,
    /// The report was truncated by cancellation, which no consumer can
    /// reproduce: it is never exported.
    #[error("a report truncated by cancellation cannot be independently verified")]
    CancelledNotReplayable,
}

/// Why a temporal sequence artifact was refused.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum TemporalTransportArtifactError {
    /// The feature marker is not this format's.
    #[error("unsupported semantics: {0}")]
    UnsupportedSemantics(&'static str),
    /// A recorded limit or stored collection exceeds the consumer's bound.
    #[error("consumer limit exceeded: {0}")]
    LimitsExceeded(&'static str),
    /// The premises digest does not match the stored premises.
    #[error("premises digest mismatch")]
    PremisesMismatch,
    /// The recomputed report differs from the stored report.
    #[error("temporal sequence report does not replay")]
    ReportMismatch,
    /// The stored catalog or laws disagree with the recorded data digest.
    #[error("temporal sequence data identity mismatch")]
    DataIdentityMismatch,
    /// The report was not produced by the prepared sequence it is exported with.
    #[error("the report was not produced by this prepared sequence")]
    ForeignReport,
    /// The stored laws disagree with the recorded provider.
    #[error("temporal sequence provider mismatch: {0}")]
    ProviderMismatch(&'static str),
}

/// Why a learned-continuous artifact was refused. Callers match the kind.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum LearnedContinuousArtifactError {
    /// The feature marker or shape is not this format's.
    #[error("unsupported semantics: {0}")]
    UnsupportedSemantics(&'static str),
    /// A stored size or bound exceeds the consumer's or the frozen cell's bound.
    #[error("limit exceeded: {0}")]
    LimitsExceeded(&'static str),
    /// The request no longer validates against its stored graph, query and rows.
    #[error("request does not validate: {0}")]
    InvalidRequest(String),
    /// The certificate re-derived from the stored graph and query is not the stored one.
    #[error("certificate does not verify: {0}")]
    ProofMismatch(String),
    /// The stored fold assignment or scheme is not the recomputed one.
    #[error("fold provenance does not match the stored rows")]
    FoldMismatch,
    /// The stored provider provenance does not cover every fitted nuisance, names a
    /// learner other than the stored spec, or is inconsistent within a nuisance role.
    #[error("provider provenance is incomplete or contradicts the learner specs")]
    ProvenanceMismatch,
    /// The recomputed point differs from the stored point.
    #[error("point estimate does not replay")]
    PointMismatch,
    /// The analytic standard error or interval does not replay.
    #[error("analytic interval does not replay")]
    IntervalMismatch,
    /// The recomputed held-out diagnostics differ from the stored ones.
    #[error("nuisance diagnostics do not replay")]
    DiagnosticsMismatch,
    /// The recomputed membership or treatment overlap differs from the stored one.
    #[error("overlap report does not replay")]
    OverlapMismatch,
    /// The stored membership predictions leave the declared overlap threshold.
    #[error("membership overlap is below the declared threshold")]
    OverlapRefused,
    /// The stored interval status contradicts its format and request.
    #[error("uncertainty bookkeeping does not check: {0}")]
    UncertaintyMismatch(&'static str),
    /// The premises digest does not match the stored premises.
    #[error("premises digest mismatch")]
    PremisesMismatch,
    /// The data digest does not match the stored rows.
    #[error("data digest mismatch")]
    DataIdentityMismatch,
    /// The evidence digest does not match the stored result (point, nuisance predictions,
    /// provenance, folds, diagnostics, overlap, interval status).
    #[error("evidence digest mismatch")]
    EvidenceMismatch,
    /// The caller's variable names are not the names the artifact was built under.
    #[error("variable names do not match the verified name mapping")]
    NamesMismatch,
}

/// Why a smoothed-dose artifact was refused. Callers match the kind.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum SmoothedDoseArtifactError {
    /// The feature marker or shape is not this format's.
    #[error("unsupported semantics: {0}")]
    UnsupportedSemantics(&'static str),
    /// A stored size or bound exceeds the consumer's own limits or compiled bounds.
    #[error("limit exceeded: {0}")]
    LimitsExceeded(&'static str),
    /// The stored request no longer validates, or replay refused it; carries the
    /// registered reason code and the namespaced detail of the refusal.
    #[error("replay refused ({code}): {message}")]
    ReplayRefused {
        /// Registered runtime reason code.
        code: &'static str,
        /// The refusal message, starting with its namespaced detail.
        message: String,
    },
    /// The request is malformed (not a coded refusal).
    #[error("request does not validate: {0}")]
    InvalidRequest(String),
    /// The certificate re-derived from the stored graph and query is not the stored one.
    #[error("certificate does not verify: {0}")]
    ProofMismatch(String),
    /// The stored fold assignment or scheme is not the recomputed one.
    #[error("fold provenance does not match the stored rows")]
    FoldMismatch,
    /// The stored fold models or provenance do not match the specs, folds or design.
    #[error("fold models or provenance contradict the learner specs")]
    ProvenanceMismatch,
    /// A replayed grid estimate differs from the stored one.
    #[error("point estimate does not replay")]
    PointMismatch,
    /// A replayed quadrature record differs from the stored one.
    #[error("quadrature record does not replay")]
    QuadratureMismatch,
    /// A replayed smoothing-bias diagnostic differs from the stored one.
    #[error("smoothing-bias diagnostic does not replay")]
    SmoothingBiasMismatch,
    /// A replayed local-support record or influence diagnostic differs.
    #[error("local support does not replay")]
    SupportMismatch,
    /// The replayed held-out diagnostics differ from the stored ones.
    #[error("nuisance diagnostics do not replay")]
    DiagnosticsMismatch,
    /// The replayed membership overlap differs from the stored one.
    #[error("overlap report does not replay")]
    OverlapMismatch,
    /// The stored interval status contradicts the closed interval route.
    #[error("uncertainty bookkeeping does not check: {0}")]
    UncertaintyMismatch(&'static str),
    /// The premises digest does not match the stored premises.
    #[error("premises digest mismatch")]
    PremisesMismatch,
    /// The data digest does not match the stored rows.
    #[error("data digest mismatch")]
    DataIdentityMismatch,
    /// The evidence digest does not match the stored result.
    #[error("evidence digest mismatch")]
    EvidenceMismatch,
    /// The caller's variable names are not the names the artifact was built under.
    #[error("variable names do not match the verified name mapping")]
    NamesMismatch,
}

impl SmoothedDoseArtifactError {
    /// The registered reason code a replay refusal carries.
    #[must_use]
    pub const fn reason_code(&self) -> Option<&'static str> {
        match self {
            Self::ReplayRefused { code, .. } => Some(code),
            _ => None,
        }
    }
}
