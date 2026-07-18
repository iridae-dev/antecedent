//! Query submodule.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::ids::VariableId;
use crate::intervention::Intervention;
use crate::value::Value;

use super::TargetPopulation;
use super::error::QueryError;

#[derive(Clone, Debug, PartialEq)]
/// Interventional distribution query P(Y | do(...)) (DESIGN.md §8).
///
/// Distinct from [`ChangeAttributionQuery`] (population/period change attribution).
/// Identify/estimate algorithms for this query are deferred (IDC); GCM sampling is
/// the interim execution path.
pub struct InterventionalDistributionQuery {
    /// Outcome variable(s) whose interventional distribution is requested.
    pub outcomes: Arc<[VariableId]>,
    /// Interventions defining the `do(...)` world.
    pub interventions: Arc<[Intervention]>,
    /// Target population.
    pub target_population: TargetPopulation,
}

impl InterventionalDistributionQuery {
    /// Single-outcome interventional distribution under the given interventions.
    #[must_use]
    pub fn new(outcome: VariableId, interventions: impl Into<Arc<[Intervention]>>) -> Self {
        Self {
            outcomes: Arc::from([outcome]),
            interventions: interventions.into(),
            target_population: TargetPopulation::AllObserved,
        }
    }

    /// Multiple outcomes.
    #[must_use]
    pub fn with_outcomes(mut self, outcomes: impl Into<Arc<[VariableId]>>) -> Self {
        self.outcomes = outcomes.into();
        self
    }

    /// Set target population.
    #[must_use]
    pub fn with_target_population(mut self, population: TargetPopulation) -> Self {
        self.target_population = population;
        self
    }

    /// Validate outcomes and interventions.
    ///
    /// # Errors
    ///
    /// Empty outcomes or invalid interventions.
    pub fn validate(&self) -> Result<(), QueryError> {
        if self.outcomes.is_empty() {
            return Err(QueryError::EmptyDistributionOutcomes);
        }
        for iv in self.interventions.iter() {
            iv.validate().map_err(|e| QueryError::InvalidIntervention(e.to_string()))?;
        }
        Ok(())
    }
}

/// Path-specific effect / contribution query (DESIGN.md §8).
///
/// Prefer this over overloading [`MediationQuery`]. Path *contribution*
/// attribution is available via GCM `path_decompose`; path-restricted natural
/// effects (identify/estimate) are deferred.
#[derive(Clone, Debug, PartialEq)]
pub struct PathSpecificEffectQuery {
    /// Treatment / source variable.
    pub treatment: VariableId,
    /// Outcome variable.
    pub outcome: VariableId,
    /// Intermediate nodes constraining the path set (`empty` = all directed paths).
    pub path_nodes: Arc<[VariableId]>,
    /// Control intervention level.
    pub control: Intervention,
    /// Active intervention level.
    pub active: Intervention,
    /// Target population.
    pub target_population: TargetPopulation,
    /// Maximum number of paths to enumerate.
    pub max_paths: usize,
    /// Maximum path length (edges).
    pub max_len: usize,
}

impl PathSpecificEffectQuery {
    /// Binary 0/1 treatment contrast with all directed paths and default limits.
    #[must_use]
    pub fn binary(treatment: VariableId, outcome: VariableId) -> Self {
        Self {
            treatment,
            outcome,
            path_nodes: Arc::from([]),
            control: Intervention::set(treatment, Value::f64(0.0)),
            active: Intervention::set(treatment, Value::f64(1.0)),
            target_population: TargetPopulation::AllObserved,
            max_paths: 64,
            max_len: 16,
        }
    }

    /// Restrict to paths that visit these intermediate nodes (in any order).
    #[must_use]
    pub fn with_path_nodes(mut self, nodes: impl Into<Arc<[VariableId]>>) -> Self {
        self.path_nodes = nodes.into();
        self
    }

    /// Cap path enumeration.
    #[must_use]
    pub const fn with_max_paths(mut self, max_paths: usize) -> Self {
        self.max_paths = max_paths;
        self
    }

    /// Cap path length.
    #[must_use]
    pub const fn with_max_len(mut self, max_len: usize) -> Self {
        self.max_len = max_len;
        self
    }

    /// Set target population.
    #[must_use]
    pub fn with_target_population(mut self, population: TargetPopulation) -> Self {
        self.target_population = population;
        self
    }

    /// Validate ids, interventions, and limits.
    ///
    /// # Errors
    ///
    /// Treatment equals outcome, zero limits, path-node overlaps, or bad interventions.
    pub fn validate(&self) -> Result<(), QueryError> {
        if self.treatment == self.outcome {
            return Err(QueryError::TreatmentEqualsOutcome { id: self.treatment });
        }
        if self.max_paths == 0 {
            return Err(QueryError::NonPositivePathLimit);
        }
        if self.max_len == 0 {
            return Err(QueryError::NonPositivePathLimit);
        }
        if self.path_nodes.iter().any(|&n| n == self.treatment || n == self.outcome) {
            return Err(QueryError::PathNodeOverlapsTreatmentOrOutcome);
        }
        let control_var =
            self.control.primary_variable().ok_or(QueryError::AmbiguousInterventionTarget)?;
        if control_var != self.treatment {
            return Err(QueryError::InterventionVariableMismatch {
                expected: self.treatment,
                got: control_var,
            });
        }
        let active_var =
            self.active.primary_variable().ok_or(QueryError::AmbiguousInterventionTarget)?;
        if active_var != self.treatment {
            return Err(QueryError::InterventionVariableMismatch {
                expected: self.treatment,
                got: active_var,
            });
        }
        Ok(())
    }
}

