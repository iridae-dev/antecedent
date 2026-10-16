//! Evidence sources: what they are, which observations stand behind them and
//! how they were derived.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{CanonicalWriter, SemanticDigest};

use super::ids::{
    LEDGER_IDENTITY_VERSION, ObservationId, OperationId, SourceId, digest_word, finish,
};

/// What a ledger entry is.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
#[non_exhaustive]
pub enum SourceKind {
    /// A set of observations (a data snapshot, a cohort, a replicate set).
    ObservationSet,
    /// An experiment or trial summary.
    Experiment,
    /// A prior, including a power prior built from earlier data.
    Prior,
    /// A fitted model.
    FittedModel,
    /// A claim supplied by an external provider.
    ExternalClaim,
}

impl SourceKind {
    /// Every kind, in a fixed order.
    pub const ALL: [Self; 5] = [
        Self::ObservationSet,
        Self::Experiment,
        Self::Prior,
        Self::FittedModel,
        Self::ExternalClaim,
    ];

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ObservationSet => "observation_set",
            Self::Experiment => "experiment",
            Self::Prior => "prior",
            Self::FittedModel => "fitted_model",
            Self::ExternalClaim => "external_claim",
        }
    }

    /// Parse a wire name.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.as_str() == name)
    }

    /// Whether a model or claim: the kinds a supersession may replace.
    #[must_use]
    pub const fn is_model_or_claim(self) -> bool {
        matches!(self, Self::FittedModel | Self::ExternalClaim)
    }
}

/// The observations that stand behind a source.
///
/// `Undeclared` is a statement that the producer did not say, not that there
/// are none: an empty `Declared` set is how a source with no data origin (an
/// elicited prior) is spelled.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ObservationBasis {
    /// Row-level identities, in canonical sorted order once in the ledger.
    Declared(Vec<ObservationId>),
    /// The producer did not declare the observations.
    Undeclared,
}

/// How a source was derived from earlier sources.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Lineage {
    /// First-hand evidence; no earlier source in the ledger.
    Original,
    /// An exact copy of one earlier source.
    Duplicate {
        /// The copied source.
        of: SourceId,
    },
    /// A transformation of the same observations (rescaling, aggregation,
    /// re-encoding). Carries the transform identity.
    Transform {
        /// The transformed source.
        of: SourceId,
        /// Identity of the transformation.
        transform: SemanticDigest,
    },
    /// A subset of the observations of one earlier source.
    Subset {
        /// The parent source.
        of: SourceId,
    },
    /// A posterior formed from a prior and a likelihood source.
    PriorPlusLikelihood {
        /// The prior source (kind `Prior`).
        prior: SourceId,
        /// The likelihood-bearing source.
        likelihood: SourceId,
    },
    /// A model or claim built from data (or results) already used elsewhere.
    DerivedFromUsedData {
        /// The sources it was built from.
        inputs: Vec<SourceId>,
    },
}

impl Lineage {
    /// Earlier sources this lineage depends on, in declaration order.
    #[must_use]
    pub fn parents(&self) -> Vec<SourceId> {
        match self {
            Self::Original => Vec::new(),
            Self::Duplicate { of } | Self::Transform { of, .. } | Self::Subset { of } => vec![*of],
            Self::PriorPlusLikelihood { prior, likelihood } => vec![*prior, *likelihood],
            Self::DerivedFromUsedData { inputs } => inputs.clone(),
        }
    }

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Original => "original",
            Self::Duplicate { .. } => "duplicate",
            Self::Transform { .. } => "transform",
            Self::Subset { .. } => "subset",
            Self::PriorPlusLikelihood { .. } => "prior_plus_likelihood",
            Self::DerivedFromUsedData { .. } => "derived_from_used_data",
        }
    }

    pub(crate) fn encode(&self, writer: &mut CanonicalWriter) {
        writer.text(self.as_str());
        match self {
            Self::Original => {}
            Self::Duplicate { of } | Self::Subset { of } => digest_word(writer, of.digest()),
            Self::Transform { of, transform } => {
                digest_word(writer, of.digest());
                digest_word(writer, *transform);
            }
            Self::PriorPlusLikelihood { prior, likelihood } => {
                digest_word(writer, prior.digest());
                digest_word(writer, likelihood.digest());
            }
            Self::DerivedFromUsedData { inputs } => {
                let mut sorted: Vec<_> = inputs.iter().map(SourceId::digest).collect();
                sorted.sort_unstable();
                writer.word(sorted.len() as u64);
                for digest in sorted {
                    digest_word(writer, digest);
                }
            }
        }
    }
}

/// Where a transported claim came from and under which assumptions it was
/// carried to its target. A source carrying this is never native target data.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransportIdentity {
    /// Source populations or experiments the claim was learned in.
    pub source_populations: Vec<String>,
    /// Population the claim was carried to.
    pub target_population: String,
    /// Identity of the selection diagram.
    pub selection_diagram: SemanticDigest,
    /// Declared invariance and selection assumptions.
    pub assumptions: Vec<String>,
}

impl TransportIdentity {
    /// Canonical digest: populations and assumptions are order-insensitive.
    #[must_use]
    pub fn digest(&self) -> SemanticDigest {
        let mut writer = CanonicalWriter::new("antecedent.evidence_ledger.transport");
        writer.word(LEDGER_IDENTITY_VERSION);
        self.encode(&mut writer);
        finish(&writer)
    }

    pub(crate) fn encode(&self, writer: &mut CanonicalWriter) {
        let mut populations = self.source_populations.clone();
        populations.sort_unstable();
        writer.texts(&populations);
        writer.text(&self.target_population);
        digest_word(writer, self.selection_diagram);
        let mut assumptions = self.assumptions.clone();
        assumptions.sort_unstable();
        writer.texts(&assumptions);
    }
}

/// A source as a producer declares it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceDeclaration {
    /// Human label; part of the identity.
    pub label: String,
    /// What the source is.
    pub kind: SourceKind,
    /// The observations behind it.
    pub observations: ObservationBasis,
    /// How it was derived.
    pub lineage: Lineage,
    /// Transport provenance, when it is a transported claim.
    pub transport: Option<TransportIdentity>,
}

impl SourceDeclaration {
    /// A first-hand source with undeclared observations. Chain the `with_*`
    /// methods to say more.
    #[must_use]
    pub fn new(label: impl Into<String>, kind: SourceKind) -> Self {
        Self {
            label: label.into(),
            kind,
            observations: ObservationBasis::Undeclared,
            lineage: Lineage::Original,
            transport: None,
        }
    }

    /// Declare the row-level observations behind the source.
    #[must_use]
    pub fn with_observations(mut self, rows: impl IntoIterator<Item = ObservationId>) -> Self {
        self.observations = ObservationBasis::Declared(rows.into_iter().collect());
        self
    }

    /// Declare how the source was derived.
    #[must_use]
    pub fn with_lineage(mut self, lineage: Lineage) -> Self {
        self.lineage = lineage;
        self
    }

    /// Declare transport provenance.
    #[must_use]
    pub fn with_transport(mut self, transport: TransportIdentity) -> Self {
        self.transport = Some(transport);
        self
    }

    /// Content identity of a canonical declaration (rows sorted).
    pub(crate) fn identity(&self) -> SourceId {
        let mut writer = CanonicalWriter::new("antecedent.evidence_ledger.source");
        writer.word(LEDGER_IDENTITY_VERSION);
        writer.text(self.kind.as_str());
        writer.text(&self.label);
        match &self.observations {
            ObservationBasis::Declared(rows) => {
                writer.word(1);
                writer.word(rows.len() as u64);
                for row in rows {
                    digest_word(&mut writer, row.digest());
                }
            }
            ObservationBasis::Undeclared => {
                writer.word(0);
            }
        }
        self.lineage.encode(&mut writer);
        match &self.transport {
            Some(transport) => {
                writer.word(1);
                transport.encode(&mut writer);
            }
            None => {
                writer.word(0);
            }
        }
        SourceId::from_digest(finish(&writer))
    }
}

/// The observations a source is known to rest on after lineage is resolved.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EffectiveObservations {
    /// Sorted, distinct observation ids.
    Known(Arc<Vec<ObservationId>>),
    /// Not computable; stays unknown and is never promoted to independent.
    Unknown(&'static str),
}

impl EffectiveObservations {
    /// The known observations, when computable.
    #[must_use]
    pub fn known(&self) -> Option<&[ObservationId]> {
        match self {
            Self::Known(rows) => Some(rows.as_slice()),
            Self::Unknown(_) => None,
        }
    }
}

/// Where a source stands in the ledger.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SourceStatus {
    /// Available for new results.
    Active,
    /// Retracted. Original preserved; no new result may use it.
    Withdrawn {
        /// Why it was withdrawn.
        reason: String,
        /// Operation that withdrew it.
        by: OperationId,
    },
    /// A newer model or claim formally replaces it.
    Superseded {
        /// Successor source.
        by: SourceId,
        /// Operation that superseded it.
        operation: OperationId,
    },
    /// Its content was swapped for corrected content.
    Replaced {
        /// Successor source.
        by: SourceId,
        /// Operation that replaced it.
        operation: OperationId,
    },
    /// It grew: the successor holds every earlier observation and more.
    Updated {
        /// Successor source.
        by: SourceId,
        /// Operation that updated it.
        operation: OperationId,
    },
}

impl SourceStatus {
    /// Whether the source can enter new results.
    #[must_use]
    pub const fn is_active(&self) -> bool {
        matches!(self, Self::Active)
    }

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Withdrawn { .. } => "withdrawn",
            Self::Superseded { .. } => "superseded",
            Self::Replaced { .. } => "replaced",
            Self::Updated { .. } => "updated",
        }
    }

    pub(crate) fn encode(&self, writer: &mut CanonicalWriter) {
        writer.text(self.as_str());
        match self {
            Self::Active => {}
            Self::Withdrawn { reason, by } => {
                writer.text(reason);
                digest_word(writer, by.digest());
            }
            Self::Superseded { by, operation }
            | Self::Replaced { by, operation }
            | Self::Updated { by, operation } => {
                digest_word(writer, by.digest());
                digest_word(writer, operation.digest());
            }
        }
    }
}

/// One entry of the ledger.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvidenceSource {
    /// Content identity.
    pub id: SourceId,
    /// The canonical declaration.
    pub declaration: SourceDeclaration,
    /// Observations after lineage resolution.
    pub effective: EffectiveObservations,
    /// Current standing.
    pub status: SourceStatus,
    /// Operation that added it.
    pub added_by: OperationId,
}
