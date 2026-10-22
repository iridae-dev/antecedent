//! Typed causal assumptions with source and scope.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::canonical_encode::{CanonicalEncode, encode_variables};
use crate::identity::CanonicalWriter;
use crate::ids::VariableId;

/// Collection of assumption records referenced by analysis artifacts.
#[derive(Clone, Debug, Default, Eq, PartialEq, Hash)]
pub struct AssumptionSet {
    /// Ordered assumption entries.
    pub entries: Vec<AssumptionRecord>,
}

impl AssumptionSet {
    /// Empty set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a record.
    pub fn push(&mut self, record: AssumptionRecord) {
        self.entries.push(record);
    }

    /// Append records that are not already present, preserving first-seen order.
    pub fn extend_unique<'a>(&mut self, records: impl IntoIterator<Item = &'a AssumptionRecord>) {
        let mut seen: std::collections::HashSet<AssumptionRecord> =
            self.entries.iter().cloned().collect();
        for record in records {
            if seen.insert(record.clone()) {
                self.entries.push(record.clone());
            }
        }
    }

    /// Number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether there are no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// One assumption with provenance of how it entered the analysis.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct AssumptionRecord {
    /// The assumption itself.
    pub assumption: Assumption,
    /// How the assumption was introduced.
    pub source: AssumptionSource,
    /// What part of the analysis it constrains.
    pub scope: AssumptionScope,
    /// Validation status (untestable is not validated).
    pub status: AssumptionStatus,
}

/// Typed causal / statistical assumption.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum Assumption {
    /// Causal Markov condition.
    CausalMarkov,
    /// Faithfulness.
    Faithfulness,
    /// Causal sufficiency (no latent confounders).
    CausalSufficiency,
    /// Consistency of potential outcomes.
    Consistency,
    /// Positivity / overlap.
    Positivity,
    /// No interference / SUTVA component.
    NoInterference,
    /// Temporal stationarity.
    Stationarity,
    /// Piecewise stationarity.
    PiecewiseStationarity,
    /// No selection bias.
    NoSelectionBias,
    /// Instrument exclusion restriction.
    ExclusionRestriction {
        /// Instrument variable.
        instrument: VariableId,
    },
    /// Monotonicity (e.g. IV).
    Monotonicity,
    /// Parametric modeling restriction.
    ParametricRestriction(ParametricAssumption),
    /// Prior / Bayesian modeling restriction (does not create identification).
    PriorRestriction(PriorAssumption),
    /// Extension point with stable id.
    Custom {
        /// Stable identifier.
        id: Arc<str>,
        /// Human-readable description.
        description: Arc<str>,
    },
}

impl CanonicalEncode for Assumption {
    fn encode(&self, writer: &mut CanonicalWriter) {
        match self {
            Self::CausalMarkov => {
                writer.word(1);
            }
            Self::Faithfulness => {
                writer.word(2);
            }
            Self::CausalSufficiency => {
                writer.word(3);
            }
            Self::Consistency => {
                writer.word(4);
            }
            Self::Positivity => {
                writer.word(5);
            }
            Self::NoInterference => {
                writer.word(6);
            }
            Self::Stationarity => {
                writer.word(7);
            }
            Self::PiecewiseStationarity => {
                writer.word(8);
            }
            Self::NoSelectionBias => {
                writer.word(9);
            }
            Self::ExclusionRestriction { instrument } => {
                writer.word(10);
                instrument.encode(writer);
            }
            Self::Monotonicity => {
                writer.word(11);
            }
            Self::ParametricRestriction(restriction) => {
                writer.word(12).text(&restriction.id).text(&restriction.description);
            }
            Self::PriorRestriction(restriction) => {
                writer.word(13).text(&restriction.id).text(&restriction.description);
            }
            Self::Custom { id, description } => {
                writer.word(14).text(id).text(description);
            }
        }
    }
}

/// Parametric restriction details.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct ParametricAssumption {
    /// Stable identifier for the restriction family.
    pub id: Arc<str>,
    /// Human-readable description.
    pub description: Arc<str>,
}

/// Prior restriction details (recorded as assumptions, not identification).
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct PriorAssumption {
    /// Stable identifier for the prior family.
    pub id: Arc<str>,
    /// Human-readable description.
    pub description: Arc<str>,
}

/// Origin of an assumption record.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum AssumptionSource {
    /// Declared by the caller.
    UserDeclared,
    /// Implied by algorithm defaults (must remain visible).
    AlgorithmDefault {
        /// Algorithm identifier.
        algorithm: Arc<str>,
    },
    /// Imported from an artifact.
    Artifact,
    /// Derived from another assumption or test.
    Derived {
        /// Parent record index or id.
        from: Arc<str>,
    },
}

/// Scope over which an assumption applies.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum AssumptionScope {
    /// Entire analysis.
    Global,
    /// Identification stage only.
    Identification,
    /// Estimation / inference stage only.
    Estimation,
    /// Discovery stage only.
    Discovery,
    /// Named subgraph or variable set.
    Variables {
        /// Variable IDs in scope.
        variables: Arc<[VariableId]>,
    },
}

/// A variable-set scope is a set: its variables are written sorted and unique, so
/// listing order and repeats do not change the encoding.
impl CanonicalEncode for AssumptionScope {
    fn encode(&self, writer: &mut CanonicalWriter) {
        match self {
            Self::Global => {
                writer.word(1);
            }
            Self::Identification => {
                writer.word(2);
            }
            Self::Estimation => {
                writer.word(3);
            }
            Self::Discovery => {
                writer.word(4);
            }
            Self::Variables { variables } => {
                let mut variables = variables.to_vec();
                variables.sort_unstable();
                variables.dedup();
                writer.word(5);
                encode_variables(writer, &variables);
            }
        }
    }
}

/// Whether an assumption has been tested.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum AssumptionStatus {
    /// Declared and not yet tested.
    Declared,
    /// Empirically supported within tolerance.
    Supported,
    /// Empirically contradicted.
    Contradicted,
    /// Not empirically testable from available data.
    Untestable,
}

impl AssumptionStatus {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Declared => "declared",
            Self::Supported => "supported",
            Self::Contradicted => "contradicted",
            Self::Untestable => "untestable",
        }
    }
}
