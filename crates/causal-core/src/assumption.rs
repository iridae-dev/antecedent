//! Typed causal assumptions with source and scope (DESIGN.md §7).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::ids::VariableId;

/// Collection of assumption records referenced by analysis artifacts.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
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
#[derive(Clone, Debug, Eq, PartialEq)]
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
#[derive(Clone, Debug, Eq, PartialEq)]
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

/// Parametric restriction details.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParametricAssumption {
    /// Stable identifier for the restriction family.
    pub id: Arc<str>,
    /// Human-readable description.
    pub description: Arc<str>,
}

/// Prior restriction details (recorded as assumptions, not identification).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PriorAssumption {
    /// Stable identifier for the prior family.
    pub id: Arc<str>,
    /// Human-readable description.
    pub description: Arc<str>,
}

/// Origin of an assumption record.
#[derive(Clone, Debug, Eq, PartialEq)]
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
#[derive(Clone, Debug, Eq, PartialEq)]
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
