//! Query submodule.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::ids::VariableId;

use super::error::QueryError;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
/// Anomaly attribution query for observed units.
pub struct AnomalyAttributionQuery {
    /// Variables whose anomaly scores are requested.
    pub targets: Arc<[VariableId]>,
    /// Optional row indices into the factual table (`None` = all complete rows).
    pub unit_rows: Option<Arc<[usize]>>,
    /// Maximum number of units to score (hard size limit).
    pub max_units: usize,
}

impl AnomalyAttributionQuery {
    /// Score all complete rows for `targets`, capped at `max_units`.
    #[must_use]
    pub fn new(targets: impl Into<Arc<[VariableId]>>, max_units: usize) -> Self {
        Self { targets: targets.into(), unit_rows: None, max_units }
    }

    /// Restrict to explicit row indices.
    #[must_use]
    pub fn with_unit_rows(mut self, rows: impl Into<Arc<[usize]>>) -> Self {
        self.unit_rows = Some(rows.into());
        self
    }

    /// Validate targets and limits.
    ///
    /// # Errors
    ///
    /// Empty targets or zero `max_units`.
    pub fn validate(&self) -> Result<(), QueryError> {
        if self.targets.is_empty() {
            return Err(QueryError::EmptyAnomalyTargets);
        }
        if self.max_units == 0 {
            return Err(QueryError::NonPositiveAnomalyLimit);
        }
        Ok(())
    }
}

/// Population / period selector for change attribution.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PopulationSelector {
    /// All rows of the bound table.
    All,
    /// Explicit row indices into a tabular view.
    Rows(Arc<[usize]>),
    /// Multi-environment index (resolved against [`EnvironmentId`] maps at call sites).
    Environment {
        /// Dense environment index into a multi-env container.
        env_index: usize,
    },
    /// Inclusive-exclusive time/row range `[start, end)`.
    TimeRange {
        /// Start index (inclusive).
        start: usize,
        /// End index (exclusive).
        end: usize,
    },
}

impl PopulationSelector {
    /// Validate selector geometry.
    ///
    /// # Errors
    ///
    /// Empty row sets or inverted time ranges.
    pub fn validate(&self) -> Result<(), QueryError> {
        match self {
            Self::All | Self::Environment { .. } => Ok(()),
            Self::Rows(rows) => {
                if rows.is_empty() {
                    Err(QueryError::EmptyPopulationRows)
                } else {
                    Ok(())
                }
            }
            Self::TimeRange { start, end } => {
                if *end <= *start {
                    Err(QueryError::InvalidPopulationTimeRange { start: *start, end: *end })
                } else {
                    Ok(())
                }
            }
        }
    }
}

/// Which structural pieces participate in change decomposition.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum AttributionComponents {
    /// Input / exogenous / parent value changes only.
    Inputs,
    /// Mechanism (conditional) changes only.
    Mechanisms,
    /// Graph-structure changes only.
    Structure,
    /// Inputs and mechanisms jointly.
    InputsAndMechanisms,
    /// Full component set.
    All,
}

/// Shapley estimation mode.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum ShapleyMode {
    /// Exact enumeration of all coalitions (`2^n`).
    Exact,
    /// Monte Carlo coalition sampling.
    MonteCarlo {
        /// Number of coalition / permutation samples.
        n_samples: usize,
    },
    /// Random permutation sampling (classic Shapley estimator).
    Permutation {
        /// Number of random permutations.
        n_permutations: usize,
    },
}

/// Configuration for Shapley allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ShapleyConfig {
    /// Estimation mode.
    pub mode: ShapleyMode,
    /// Hard limit on exact combinatorial size (default 12).
    pub max_exact_components: usize,
    /// When true, Exact may exceed `max_exact_components` (explicit override).
    pub allow_exact_override: bool,
    /// RNG seed for approximate modes.
    pub seed: u64,
}

impl ShapleyConfig {
    /// Default exact config with size limit 12.
    #[must_use]
    pub const fn exact() -> Self {
        Self {
            mode: ShapleyMode::Exact,
            max_exact_components: 12,
            allow_exact_override: false,
            seed: 0,
        }
    }

    /// Monte Carlo Shapley with `n_samples` coalition evaluations.
    #[must_use]
    pub const fn monte_carlo(n_samples: usize) -> Self {
        Self {
            mode: ShapleyMode::MonteCarlo { n_samples },
            max_exact_components: 12,
            allow_exact_override: false,
            seed: 0,
        }
    }

    /// Permutation sampling with `n_permutations` random orders.
    #[must_use]
    pub const fn permutation(n_permutations: usize) -> Self {
        Self {
            mode: ShapleyMode::Permutation { n_permutations },
            max_exact_components: 12,
            allow_exact_override: false,
            seed: 0,
        }
    }

    /// Override the exact size limit.
    #[must_use]
    pub const fn with_max_exact_components(mut self, max: usize) -> Self {
        self.max_exact_components = max;
        self
    }

    /// Allow Exact above the configured limit (explicit opt-in).
    #[must_use]
    pub const fn with_exact_override(mut self, allow: bool) -> Self {
        self.allow_exact_override = allow;
        self
    }

    /// Set RNG seed for approximate modes.
    #[must_use]
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Validate configuration.
    ///
    /// # Errors
    ///
    /// Zero sample budgets or zero exact limit.
    pub fn validate(&self) -> Result<(), QueryError> {
        if self.max_exact_components == 0 {
            return Err(QueryError::NonPositiveShapleyLimit);
        }
        match self.mode {
            ShapleyMode::Exact => Ok(()),
            ShapleyMode::MonteCarlo { n_samples } => {
                if n_samples == 0 {
                    Err(QueryError::NonPositiveShapleySamples)
                } else {
                    Ok(())
                }
            }
            ShapleyMode::Permutation { n_permutations } => {
                if n_permutations == 0 {
                    Err(QueryError::NonPositiveShapleySamples)
                } else {
                    Ok(())
                }
            }
        }
    }
}

/// How to allocate total change across components.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AllocationMethod {
    /// Fixed sequential order (path-dependent; interactions explicit).
    Sequential {
        /// Component evaluation order.
        order: Arc<[crate::ids::ComponentId]>,
    },
    /// Shapley symmetrization (exact or approximate).
    Shapley {
        /// Approximation / size-limit config.
        approximation: ShapleyConfig,
    },
    /// Path-based dynamic-programming decomposition.
    PathBased,
}

impl AllocationMethod {
    /// Validate allocation settings.
    ///
    /// # Errors
    ///
    /// Empty sequential order or invalid Shapley config.
    pub fn validate(&self) -> Result<(), QueryError> {
        match self {
            Self::Sequential { order } if order.is_empty() => Err(QueryError::EmptyAllocationOrder),
            Self::Sequential { .. } | Self::PathBased => Ok(()),
            Self::Shapley { approximation } => approximation.validate(),
        }
    }
}

/// Distribution / population change attribution query.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ChangeAttributionQuery {
    /// Outcome whose marginal (or summary) change is attributed.
    pub outcome: VariableId,
    /// Baseline population / period.
    pub baseline: PopulationSelector,
    /// Comparison population / period.
    pub comparison: PopulationSelector,
    /// Which structural pieces participate.
    pub components: AttributionComponents,
    /// Allocation rule.
    pub allocation: AllocationMethod,
    /// Maximum number of attribution components (hard size guard for Exact).
    pub max_components: usize,
}

impl ChangeAttributionQuery {
    /// Construct with Shapley Monte Carlo allocation (common pinned baseline-GCM path).
    #[must_use]
    pub fn new(
        outcome: VariableId,
        baseline: PopulationSelector,
        comparison: PopulationSelector,
    ) -> Self {
        Self {
            outcome,
            baseline,
            comparison,
            components: AttributionComponents::Mechanisms,
            allocation: AllocationMethod::Shapley {
                approximation: ShapleyConfig::monte_carlo(2_000),
            },
            max_components: 64,
        }
    }

    /// Set component family.
    #[must_use]
    pub const fn with_components(mut self, components: AttributionComponents) -> Self {
        self.components = components;
        self
    }

    /// Set allocation method.
    #[must_use]
    pub fn with_allocation(mut self, allocation: AllocationMethod) -> Self {
        self.allocation = allocation;
        self
    }

    /// Cap the number of components considered.
    #[must_use]
    pub const fn with_max_components(mut self, max_components: usize) -> Self {
        self.max_components = max_components;
        self
    }

    /// Validate query.
    ///
    /// # Errors
    ///
    /// Invalid populations, allocation, or zero `max_components`.
    pub fn validate(&self) -> Result<(), QueryError> {
        if self.max_components == 0 {
            return Err(QueryError::NonPositiveComponentLimit);
        }
        self.baseline.validate()?;
        self.comparison.validate()?;
        self.allocation.validate()?;
        Ok(())
    }
}

/// Mechanism-change *detection* query — not attribution.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MechanismChangeQuery {
    /// Nodes whose mechanisms are tested for change.
    pub targets: Arc<[VariableId]>,
    /// Baseline population.
    pub baseline: PopulationSelector,
    /// Comparison population.
    pub comparison: PopulationSelector,
    /// Significance level for change tests.
    pub significance_level: OrderedFloatBits,
    /// Maximum targets to test.
    pub max_targets: usize,
}

/// Bit-pattern wrapper so [`MechanismChangeQuery`] stays `Eq`/`Hash` with an f64 level.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct OrderedFloatBits(u64);

impl OrderedFloatBits {
    /// From f64 (NaN → 0 payload).
    #[must_use]
    pub fn from_f64(v: f64) -> Self {
        Self(if v.is_nan() { 0 } else { v.to_bits() })
    }

    /// To f64.
    #[must_use]
    pub const fn to_f64(self) -> f64 {
        f64::from_bits(self.0)
    }
}

impl MechanismChangeQuery {
    /// Test all `targets` at the given significance level.
    #[must_use]
    pub fn new(
        targets: impl Into<Arc<[VariableId]>>,
        baseline: PopulationSelector,
        comparison: PopulationSelector,
        significance_level: f64,
        max_targets: usize,
    ) -> Self {
        Self {
            targets: targets.into(),
            baseline,
            comparison,
            significance_level: OrderedFloatBits::from_f64(significance_level),
            max_targets,
        }
    }

    /// Validate.
    ///
    /// # Errors
    ///
    /// Empty targets, invalid α, or bad populations.
    pub fn validate(&self) -> Result<(), QueryError> {
        if self.targets.is_empty() {
            return Err(QueryError::EmptyMechanismChangeTargets);
        }
        if self.max_targets == 0 {
            return Err(QueryError::NonPositiveComponentLimit);
        }
        let alpha = self.significance_level.to_f64();
        if !(alpha > 0.0 && alpha < 1.0) {
            return Err(QueryError::InvalidSignificanceLevel);
        }
        self.baseline.validate()?;
        self.comparison.validate()?;
        Ok(())
    }
}

/// Per-unit change attribution query.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct UnitChangeQuery {
    /// Outcome variable.
    pub outcome: VariableId,
    /// Optional factual row indices (`None` = all units up to `max_units`).
    pub unit_rows: Option<Arc<[usize]>>,
    /// Components to attribute (inputs / mechanisms / both).
    pub components: AttributionComponents,
    /// Allocation method.
    pub allocation: AllocationMethod,
    /// Hard unit count limit.
    pub max_units: usize,
}

impl UnitChangeQuery {
    /// Attribute change for `outcome` across units.
    #[must_use]
    pub fn new(outcome: VariableId, max_units: usize) -> Self {
        Self {
            outcome,
            unit_rows: None,
            components: AttributionComponents::Inputs,
            allocation: AllocationMethod::Shapley {
                approximation: ShapleyConfig::monte_carlo(500),
            },
            max_units,
        }
    }

    /// Restrict to explicit rows.
    #[must_use]
    pub fn with_unit_rows(mut self, rows: impl Into<Arc<[usize]>>) -> Self {
        self.unit_rows = Some(rows.into());
        self
    }

    /// Set components.
    #[must_use]
    pub const fn with_components(mut self, components: AttributionComponents) -> Self {
        self.components = components;
        self
    }

    /// Set allocation.
    #[must_use]
    pub fn with_allocation(mut self, allocation: AllocationMethod) -> Self {
        self.allocation = allocation;
        self
    }

    /// Validate.
    ///
    /// # Errors
    ///
    /// Zero `max_units` or invalid allocation.
    pub fn validate(&self) -> Result<(), QueryError> {
        if self.max_units == 0 {
            return Err(QueryError::NonPositiveAnomalyLimit);
        }
        self.allocation.validate()?;
        Ok(())
    }
}

