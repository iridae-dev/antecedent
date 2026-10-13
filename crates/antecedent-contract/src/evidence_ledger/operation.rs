//! Ledger operations, derived results and their downstream effects.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{CanonicalWriter, SemanticDigest};

use super::ids::{ObservationId, OperationId, ResultId, SourceId, digest_word};
use super::overlap::MassAccount;
use super::source::SourceDeclaration;

/// The distinct ways evidence can be combined or re-used. None implies
/// another, and each has its own overlap policy in the ledger.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum SynthesisKind {
    /// Combine summaries of separate sources into one pooled summary.
    StatisticalPooling,
    /// Use prior information, or exchangeable sources, to inform an estimate.
    BayesianBorrowing,
    /// Carry a claim to a target population under a selection diagram.
    CausalTransport,
    /// Cite an existing result again; its observations are counted once.
    EvidenceReuse,
    /// Use the same observations in a later stage of one workflow.
    RepeatedObservationUse,
}

impl SynthesisKind {
    /// Every kind in a fixed order.
    pub const ALL: [Self; 5] = [
        Self::StatisticalPooling,
        Self::BayesianBorrowing,
        Self::CausalTransport,
        Self::EvidenceReuse,
        Self::RepeatedObservationUse,
    ];

    /// Stable `snake_case` name. The first four coincide with the names of
    /// the design-layer composition operations.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::StatisticalPooling => "statistical_pooling",
            Self::BayesianBorrowing => "bayesian_borrowing",
            Self::CausalTransport => "causal_transport",
            Self::EvidenceReuse => "evidence_reuse",
            Self::RepeatedObservationUse => "repeated_observation_use",
        }
    }

    /// Parse a wire name.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.as_str() == name)
    }
}

/// How the dependence between pooled sources is accounted for.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PoolingDependence {
    /// Sources are pairwise independent (known and disjoint observations).
    Independent,
    /// A full covariance of the summaries is supplied and preserved. The
    /// ledger binds its identity and whether it has cross terms.
    DeclaredCovariance {
        /// Identity of the covariance matrix.
        covariance: SemanticDigest,
        /// Whether the matrix has any nonzero off-diagonal entry.
        has_cross_covariance: bool,
    },
    /// Dependence is unknown; only a structural envelope over a common
    /// correlation range is licensed. Never a point, interval or posterior.
    AssumptionRange {
        /// Lower correlation bound.
        rho_lo: f64,
        /// Upper correlation bound.
        rho_hi: f64,
    },
}

/// One combination of evidence.
#[derive(Clone, Debug, PartialEq)]
pub enum SynthesisOperation {
    /// Pool summaries of two or more sources.
    StatisticalPooling {
        /// Contributing sources.
        sources: Vec<SourceId>,
        /// How their dependence is accounted for.
        dependence: PoolingDependence,
    },
    /// Borrow strength: priors (possibly data-derived) plus likelihood-bearing
    /// sources, or exchangeable studies with no prior source.
    BayesianBorrowing {
        /// Prior sources; each one's data origin must be disjoint from every
        /// likelihood source and from every other prior.
        priors: Vec<SourceId>,
        /// Likelihood-bearing sources, pairwise independent.
        likelihood: Vec<SourceId>,
    },
    /// Carry transported claims to the target alongside native target data.
    CausalTransport {
        /// Claims carrying a transport identity.
        transported: Vec<SourceId>,
        /// Native target-population evidence (no transport identity).
        native: Vec<SourceId>,
    },
    /// Cite a valid existing result again.
    EvidenceReuse {
        /// The result being cited.
        of: ResultId,
    },
    /// Declare that the same observations serve a later stage.
    RepeatedObservationUse {
        /// Sources whose observations are used again.
        sources: Vec<SourceId>,
        /// Named stage of the workflow that re-uses them.
        stage: String,
    },
}

fn sorted_digests(ids: &[SourceId]) -> Vec<SemanticDigest> {
    let mut out: Vec<SemanticDigest> = ids.iter().map(SourceId::digest).collect();
    out.sort_unstable();
    out
}

fn write_ids(writer: &mut CanonicalWriter, ids: &[SourceId]) {
    let sorted = sorted_digests(ids);
    writer.word(sorted.len() as u64);
    for digest in sorted {
        digest_word(writer, digest);
    }
}

impl SynthesisOperation {
    /// Which kind of combination this is.
    #[must_use]
    pub const fn kind(&self) -> SynthesisKind {
        match self {
            Self::StatisticalPooling { .. } => SynthesisKind::StatisticalPooling,
            Self::BayesianBorrowing { .. } => SynthesisKind::BayesianBorrowing,
            Self::CausalTransport { .. } => SynthesisKind::CausalTransport,
            Self::EvidenceReuse { .. } => SynthesisKind::EvidenceReuse,
            Self::RepeatedObservationUse { .. } => SynthesisKind::RepeatedObservationUse,
        }
    }

    /// Directly referenced sources, in declaration order (a reuse names a
    /// result, not sources).
    #[must_use]
    pub fn direct_sources(&self) -> Vec<SourceId> {
        match self {
            Self::StatisticalPooling { sources, .. }
            | Self::RepeatedObservationUse { sources, .. } => sources.clone(),
            Self::BayesianBorrowing { priors, likelihood } => {
                priors.iter().chain(likelihood).copied().collect()
            }
            Self::CausalTransport { transported, native } => {
                transported.iter().chain(native).copied().collect()
            }
            Self::EvidenceReuse { .. } => Vec::new(),
        }
    }

    /// Canonical encoding: source lists are order-insensitive.
    pub(crate) fn encode(&self, writer: &mut CanonicalWriter) {
        writer.text(self.kind().as_str());
        match self {
            Self::StatisticalPooling { sources, dependence } => {
                write_ids(writer, sources);
                match dependence {
                    PoolingDependence::Independent => {
                        writer.text("independent");
                    }
                    PoolingDependence::DeclaredCovariance { covariance, has_cross_covariance } => {
                        writer.text("declared_covariance");
                        digest_word(writer, *covariance);
                        writer.word(u64::from(*has_cross_covariance));
                    }
                    PoolingDependence::AssumptionRange { rho_lo, rho_hi } => {
                        writer.text("assumption_range");
                        writer.real(*rho_lo);
                        writer.real(*rho_hi);
                    }
                }
            }
            Self::BayesianBorrowing { priors, likelihood } => {
                write_ids(writer, priors);
                write_ids(writer, likelihood);
            }
            Self::CausalTransport { transported, native } => {
                write_ids(writer, transported);
                write_ids(writer, native);
            }
            Self::EvidenceReuse { of } => {
                digest_word(writer, of.digest());
            }
            Self::RepeatedObservationUse { sources, stage } => {
                write_ids(writer, sources);
                writer.text(stage);
            }
        }
    }
}

/// A request to register a combination as a ledger result.
#[derive(Clone, Debug, PartialEq)]
pub struct CombinationRequest {
    /// Human label; part of the result identity.
    pub label: String,
    /// The combination.
    pub operation: SynthesisOperation,
    /// A result this one recomputes; it becomes `Replaced`.
    pub replaces: Option<ResultId>,
}

/// What kind of statement a result licenses.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum ClaimClass {
    /// A point value. No interval or posterior is licensed by the ledger.
    PointOnly,
    /// A structural envelope over a declared dependence range.
    AssumptionRange,
}

impl ClaimClass {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::PointOnly => "point_only",
            Self::AssumptionRange => "assumption_range",
        }
    }
}

/// Downstream classification after evidence changes. Claim-agnostic: any
/// dependent artifact can be classified with it.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
#[non_exhaustive]
pub enum Revalidation {
    /// Nothing it rests on changed.
    StillValid,
    /// An input changed but a successor exists: recompute it.
    Recompute,
    /// An input it cannot do without is gone: it must not be used.
    Withdrawn,
}

impl Revalidation {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::StillValid => "still_valid",
            Self::Recompute => "recompute",
            Self::Withdrawn => "withdrawn",
        }
    }

    /// Parse a wire name.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        [Self::StillValid, Self::Recompute, Self::Withdrawn]
            .into_iter()
            .find(|class| class.as_str() == name)
    }
}

/// Standing of a result.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResultStatus {
    /// Usable.
    Valid,
    /// An input was superseded, replaced or updated: recompute it.
    NeedsRecompute {
        /// Operation that made it stale.
        cause: OperationId,
    },
    /// An input it cannot do without was withdrawn. Preserved, unusable.
    Withdrawn {
        /// Operation that withdrew its basis.
        cause: OperationId,
    },
    /// Recomputed by a newer result. Preserved, unusable.
    Replaced {
        /// The recomputed result.
        by: ResultId,
        /// Operation that registered it.
        operation: OperationId,
    },
}

impl ResultStatus {
    /// The downstream classification of this standing. A replaced result is
    /// retired and classed as withdrawn.
    #[must_use]
    pub const fn revalidation(&self) -> Revalidation {
        match self {
            Self::Valid => Revalidation::StillValid,
            Self::NeedsRecompute { .. } => Revalidation::Recompute,
            Self::Withdrawn { .. } | Self::Replaced { .. } => Revalidation::Withdrawn,
        }
    }

    pub(crate) fn encode(&self, writer: &mut CanonicalWriter) {
        match self {
            Self::Valid => {
                writer.text("valid");
            }
            Self::NeedsRecompute { cause } => {
                writer.text("needs_recompute");
                digest_word(writer, cause.digest());
            }
            Self::Withdrawn { cause } => {
                writer.text("withdrawn");
                digest_word(writer, cause.digest());
            }
            Self::Replaced { by, operation } => {
                writer.text("replaced");
                digest_word(writer, by.digest());
                digest_word(writer, operation.digest());
            }
        }
    }
}

/// A registered combination with its accounting.
#[derive(Clone, Debug, PartialEq)]
pub struct ResultRecord {
    /// Identity.
    pub id: ResultId,
    /// Human label.
    pub label: String,
    /// The combination.
    pub operation: SynthesisOperation,
    /// Observation mass, each observation counted once.
    pub mass: MassAccount,
    /// What the result licenses.
    pub claim_class: ClaimClass,
    /// Digest of the sorted distinct observations counted (absent when any
    /// contributing observation set is unknown).
    pub observations_digest: Option<SemanticDigest>,
    /// Transport identities carried by contributing sources, sorted.
    pub transport_identities: Vec<SemanticDigest>,
    /// Whether any observation is used again at a later stage. Such a result
    /// is not independent confirmation.
    pub data_reuse: bool,
    /// Overlap-accounted dependence note: `true` when a declared covariance
    /// or assumption range, not independence, licensed the combination.
    pub dependence_declared: bool,
    /// Result this one recomputes.
    pub replaces: Option<ResultId>,
    /// Operation that registered it.
    pub created_by: OperationId,
    /// Current standing.
    pub status: ResultStatus,
}

/// An operation on the ledger.
#[derive(Clone, Debug, PartialEq)]
pub enum LedgerOperation {
    /// Add a new source.
    Add {
        /// The source.
        source: SourceDeclaration,
    },
    /// The source grew: the replacement holds every earlier observation.
    Update {
        /// Source being updated.
        target: SourceId,
        /// Grown content.
        replacement: SourceDeclaration,
    },
    /// The source's content is swapped (a correction).
    Replace {
        /// Source being replaced.
        target: SourceId,
        /// Corrected content.
        replacement: SourceDeclaration,
    },
    /// Retract a source.
    Withdraw {
        /// Source being withdrawn.
        target: SourceId,
        /// Why.
        reason: String,
    },
    /// A newer model or claim formally replaces an older one.
    Supersede {
        /// Source being superseded.
        target: SourceId,
        /// The newer model or claim.
        replacement: SourceDeclaration,
    },
    /// Re-classify every result against the current source standings.
    RevalidateDownstream,
    /// Register a combination of evidence.
    Combine {
        /// The request.
        request: CombinationRequest,
    },
}

impl LedgerOperation {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Add { .. } => "add",
            Self::Update { .. } => "update",
            Self::Replace { .. } => "replace",
            Self::Withdraw { .. } => "withdraw",
            Self::Supersede { .. } => "supersede",
            Self::RevalidateDownstream => "revalidate_downstream",
            Self::Combine { .. } => "combine",
        }
    }
}

/// An overlap noticed when a source entered the ledger. Informational: the
/// refusal happens when overlapping sources are combined.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DetectedOverlap {
    /// The earlier source.
    pub existing: SourceId,
    /// Observations shared with it.
    pub shared: u64,
}

/// A result classification.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResultRevalidation {
    /// The result.
    pub result: ResultId,
    /// Its classification.
    pub class: Revalidation,
}

/// Downstream effect of one applied operation.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct OperationEffect {
    /// Source created by the operation.
    pub created_source: Option<SourceId>,
    /// Source retired (withdrawn, superseded, replaced or updated).
    pub retired_source: Option<SourceId>,
    /// Result created by the operation.
    pub created_result: Option<ResultId>,
    /// Result retired because the new one recomputes it.
    pub replaced_result: Option<ResultId>,
    /// Overlaps with earlier sources detected at entry.
    pub detected_overlaps: Vec<DetectedOverlap>,
    /// Classification of every result, sorted by id (set by operations that
    /// retire or re-check sources).
    pub revalidation: Vec<ResultRevalidation>,
}

/// An operation with its identity and effect.
#[derive(Clone, Debug, PartialEq)]
pub struct AppliedOperation {
    /// Identity: chained with the previous operation.
    pub id: OperationId,
    /// What was applied.
    pub operation: LedgerOperation,
    /// What it did downstream.
    pub effect: OperationEffect,
}

/// Rows shared by two sources, for composition-layer adapters.
pub type SharedObservations = Vec<ObservationId>;
