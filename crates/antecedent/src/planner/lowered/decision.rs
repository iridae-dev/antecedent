//! Layer four of the lowered program: evidence composition and decision demand.
//!
//! Which sources the program reads and how they combine, and, when compile starts
//! from a decision or a value-of-information request, what that decision needs the
//! statistical program to publish.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{CanonicalWriter, SemanticDigest};

const EVIDENCE_PROGRAM_TAG: &str = "antecedent.lowered_program.evidence";

/// What a source contributes.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum EvidenceRole {
    /// Observational rows of the study.
    Observations,
    /// Rows of a randomized experiment.
    Trial,
    /// A prior carried by an artifact.
    PriorArtifact,
    /// A source population that a transport reads.
    TransportSource,
}

impl EvidenceRole {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Observations => "observations",
            Self::Trial => "trial",
            Self::PriorArtifact => "prior_artifact",
            Self::TransportSource => "transport_source",
        }
    }
}

/// One source the program reads.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct EvidenceSourceRef {
    pub(crate) role: EvidenceRole,
    pub(crate) digest: Option<[u8; 32]>,
}

impl EvidenceSourceRef {
    /// What the source contributes.
    #[must_use]
    pub const fn role(&self) -> EvidenceRole {
        self.role
    }

    /// Content digest of the source, when the lowering bound one.
    #[must_use]
    pub const fn digest(&self) -> Option<[u8; 32]> {
        self.digest
    }
}

/// How the sources combine into one answer.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Composition {
    /// One source, one structure.
    Single,
    /// A weighted mixture over graph-posterior atoms.
    AtomMixture {
        /// Number of atoms.
        atoms: usize,
    },
    /// A source population transported to a target.
    Transported,
    /// Several sources pooled under a declared dependence.
    Pooled {
        /// Number of pooled sources.
        sources: usize,
    },
}

/// The evidence a decision needs the statistical program to publish.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum CarrierDemand {
    /// An affine utility of the mean; a point estimate carries it.
    Mean,
    /// Expected loss over a parameter; only a posterior carries it.
    ParameterPosterior,
    /// Risk or a threshold of an individual outcome; only an outcome law carries it.
    OutcomeLaw,
}

impl CarrierDemand {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Mean => "mean",
            Self::ParameterPosterior => "parameter_posterior",
            Self::OutcomeLaw => "outcome_law",
        }
    }
}

/// Preposterior value-of-information request the decision carries.
#[derive(Clone, Debug, PartialEq)]
pub struct EvsiProgram {
    pub(crate) candidates: Arc<[Arc<str>]>,
    pub(crate) prior_observation_ids: Arc<[Arc<str>]>,
    pub(crate) rng_seed: u64,
    pub(crate) mc_error_tolerance: f64,
    pub(crate) require_net_value: bool,
}

impl EvsiProgram {
    /// Semantic ids of the candidate designs.
    #[must_use]
    pub fn candidates(&self) -> &[Arc<str>] {
        &self.candidates
    }

    /// Observations already summarized by the prior.
    #[must_use]
    pub fn prior_observation_ids(&self) -> &[Arc<str>] {
        &self.prior_observation_ids
    }

    /// Seed of the Monte Carlo evaluation.
    #[must_use]
    pub const fn rng_seed(&self) -> u64 {
        self.rng_seed
    }

    /// Standard error at or below which an estimate counts as converged.
    #[must_use]
    pub const fn mc_error_tolerance(&self) -> f64 {
        self.mc_error_tolerance
    }

    /// Whether a missing cost-to-utility map refuses instead of reporting value and cost apart.
    #[must_use]
    pub const fn require_net_value(&self) -> bool {
        self.require_net_value
    }

    fn write(&self, writer: &mut CanonicalWriter) {
        writer
            .texts(&self.candidates[..])
            .texts(&self.prior_observation_ids[..])
            .word(self.rng_seed)
            .real(self.mc_error_tolerance)
            .word(u64::from(self.require_net_value));
    }
}

/// The decision compile started from.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionProgram {
    pub(crate) carrier: CarrierDemand,
    pub(crate) actions: usize,
    pub(crate) constraints: usize,
    pub(crate) chance_threshold: f64,
    pub(crate) utility_unit: Option<Arc<str>>,
    pub(crate) contract_identity: Option<Arc<str>>,
    pub(crate) information_value: Option<EvsiProgram>,
}

impl DecisionProgram {
    /// Evidence the decision needs.
    #[must_use]
    pub const fn carrier(&self) -> CarrierDemand {
        self.carrier
    }

    /// Number of candidate actions.
    #[must_use]
    pub const fn actions(&self) -> usize {
        self.actions
    }

    /// Number of constraints on admissible actions.
    #[must_use]
    pub const fn constraints(&self) -> usize {
        self.constraints
    }

    /// Minimum chance-constraint satisfaction.
    #[must_use]
    pub const fn chance_threshold(&self) -> f64 {
        self.chance_threshold
    }

    /// Unit of the utility, when declared.
    #[must_use]
    pub fn utility_unit(&self) -> Option<&str> {
        self.utility_unit.as_deref()
    }

    /// Identity of the decision contract, when declared.
    #[must_use]
    pub fn contract_identity(&self) -> Option<&str> {
        self.contract_identity.as_deref()
    }

    /// Value-of-information request, when compile started from one.
    #[must_use]
    pub const fn information_value(&self) -> Option<&EvsiProgram> {
        self.information_value.as_ref()
    }

    fn write(&self, writer: &mut CanonicalWriter) {
        writer
            .text(self.carrier.as_str())
            .word(self.actions as u64)
            .word(self.constraints as u64)
            .real(self.chance_threshold)
            .opt_text(self.utility_unit.as_deref())
            .opt_text(self.contract_identity.as_deref());
        match &self.information_value {
            Some(evsi) => {
                writer.word(1);
                evsi.write(writer);
            }
            None => {
                writer.word(0);
            }
        }
    }
}

/// Layer four: evidence composition and the decision demand.
#[derive(Clone, Debug, PartialEq)]
pub struct EvidenceDecisionProgram {
    pub(crate) sources: Arc<[EvidenceSourceRef]>,
    pub(crate) composition: Composition,
    pub(crate) decision: Option<DecisionProgram>,
}

impl EvidenceDecisionProgram {
    /// Sources the program reads.
    #[must_use]
    pub fn sources(&self) -> &[EvidenceSourceRef] {
        &self.sources
    }

    /// How the sources combine.
    #[must_use]
    pub const fn composition(&self) -> Composition {
        self.composition
    }

    /// The decision compile started from, when it started from one.
    #[must_use]
    pub const fn decision(&self) -> Option<&DecisionProgram> {
        self.decision.as_ref()
    }

    /// Identity of this layer.
    pub(crate) fn identity(&self) -> SemanticDigest {
        let mut writer = CanonicalWriter::new(EVIDENCE_PROGRAM_TAG);
        writer.word(self.sources.len() as u64);
        for source in &*self.sources {
            writer.text(source.role.as_str());
            match &source.digest {
                Some(digest) => writer.word(1).bytes(digest),
                None => writer.word(0),
            };
        }
        match self.composition {
            Composition::Single => writer.text("single"),
            Composition::AtomMixture { atoms } => writer.text("atom_mixture").word(atoms as u64),
            Composition::Transported => writer.text("transported"),
            Composition::Pooled { sources } => writer.text("pooled").word(sources as u64),
        };
        match &self.decision {
            Some(decision) => {
                writer.word(1);
                decision.write(&mut writer);
            }
            None => {
                writer.word(0);
            }
        }
        SemanticDigest::from_bytes(writer.finish())
    }
}
