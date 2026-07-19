//! Candidate experiment / measurement plans.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::{EnvironmentId, VariableId};

/// Cost / resource load of a candidate (library does not own currency units).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DesignCost {
    /// Scalar cost (caller-defined units).
    pub amount: f64,
    /// Optional sample-budget consumption.
    pub sample_budget: u64,
}

impl DesignCost {
    /// Zero cost.
    #[must_use]
    pub const fn zero() -> Self {
        Self { amount: 0.0, sample_budget: 0 }
    }
}

/// Measure additional variables (soft evidence on graph features involving them).
#[derive(Clone, Debug, PartialEq)]
pub struct MeasurementPlan {
    /// Variables to measure.
    pub variables: Arc<[VariableId]>,
    /// Cost.
    pub cost: DesignCost,
    /// Optional semantic tag for CRN streams.
    pub tag: u64,
}

/// Intervene on treatment-like targets (do-mutilation on graph draws).
#[derive(Clone, Debug, PartialEq)]
pub struct ExperimentPlan {
    /// Intervention targets.
    pub targets: Arc<[VariableId]>,
    /// Cost.
    pub cost: DesignCost,
    /// Optional semantic tag.
    pub tag: u64,
}

/// Observe / enlarge an environment partition.
#[derive(Clone, Debug, PartialEq)]
pub struct EnvironmentPlan {
    /// Environment to observe.
    pub environment: EnvironmentId,
    /// Additional rows expected.
    pub additional_rows: u64,
    /// Cost.
    pub cost: DesignCost,
    /// Optional semantic tag.
    pub tag: u64,
}

/// Increase sampling rate / sample size for an existing design matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct SamplingPlan {
    /// Additional samples (n increment).
    pub additional_samples: u64,
    /// Cost.
    pub cost: DesignCost,
    /// Optional semantic tag.
    pub tag: u64,
}

/// Candidate design plan variants.
#[derive(Clone, Debug, PartialEq)]
pub enum CandidateDesign {
    /// Measure variables.
    Measure(MeasurementPlan),
    /// Intervene.
    Intervene(ExperimentPlan),
    /// Observe environment.
    ObserveEnvironment(EnvironmentPlan),
    /// Increase sampling rate.
    IncreaseSamplingRate(SamplingPlan),
}

impl CandidateDesign {
    /// Declared cost of this candidate.
    #[must_use]
    pub fn cost(&self) -> DesignCost {
        match self {
            Self::Measure(p) => p.cost,
            Self::Intervene(p) => p.cost,
            Self::ObserveEnvironment(p) => p.cost,
            Self::IncreaseSamplingRate(p) => p.cost,
        }
    }

    /// Semantic CRN tag.
    #[must_use]
    pub fn tag(&self) -> u64 {
        match self {
            Self::Measure(p) => p.tag,
            Self::Intervene(p) => p.tag,
            Self::ObserveEnvironment(p) => p.tag,
            Self::IncreaseSamplingRate(p) => p.tag,
        }
    }
}
