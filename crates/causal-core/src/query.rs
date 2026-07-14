//! Typed causal queries (DESIGN.md §8).
//!
//! Hot paths bind [`VariableId`]s; names are resolved only at API boundaries.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::ids::{EnvironmentId, VariableId};
use crate::intervention::Intervention;
use crate::value::Value;

pub use crate::intervention::TemporalPolicy;

/// Target population for an effect query (DESIGN.md §8.2).
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum TargetPopulation {
    /// All observed units.
    AllObserved,
    /// Treated units only.
    Treated,
    /// Untreated units only.
    Untreated,
    /// Environment-restricted population.
    Environment(EnvironmentId),
}

/// Average treatment effect (ATE / ATT-style) query.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct AverageEffectQuery {
    /// Treatment variable.
    pub treatment: VariableId,
    /// Outcome variable.
    pub outcome: VariableId,
    /// Optional effect modifiers .
    pub effect_modifiers: Arc<[VariableId]>,
    /// Control intervention level (typically treatment = 0).
    pub control: Intervention,
    /// Active intervention level (typically treatment = 1).
    pub active: Intervention,
    /// Target population.
    pub target_population: TargetPopulation,
}

impl AverageEffectQuery {
    /// ATE for binary treatment coded as 0/1 on `treatment`.
    #[must_use]
    pub fn binary_ate(treatment: VariableId, outcome: VariableId) -> Self {
        Self {
            treatment,
            outcome,
            effect_modifiers: Arc::from([]),
            control: Intervention::set(treatment, Value::f64(0.0)),
            active: Intervention::set(treatment, Value::f64(1.0)),
            target_population: TargetPopulation::AllObserved,
        }
    }

    /// ATE with explicit control/active float levels.
    #[must_use]
    pub fn with_levels(
        treatment: VariableId,
        outcome: VariableId,
        control_level: f64,
        active_level: f64,
    ) -> Self {
        Self {
            treatment,
            outcome,
            effect_modifiers: Arc::from([]),
            control: Intervention::set(treatment, Value::f64(control_level)),
            active: Intervention::set(treatment, Value::f64(active_level)),
            target_population: TargetPopulation::AllObserved,
        }
    }

    /// Attach effect modifiers (IDs already resolved).
    #[must_use]
    pub fn with_effect_modifiers(mut self, modifiers: impl Into<Arc<[VariableId]>>) -> Self {
        self.effect_modifiers = modifiers.into();
        self
    }

    /// Set target population.
    #[must_use]
    pub fn with_target_population(mut self, population: TargetPopulation) -> Self {
        self.target_population = population;
        self
    }

    /// Validate that interventions target the treatment variable.
    ///
    /// # Errors
    ///
    /// Returns [`QueryError`] when interventions are inconsistent.
    pub fn validate(&self) -> Result<(), QueryError> {
        if self.treatment == self.outcome {
            return Err(QueryError::TreatmentEqualsOutcome { id: self.treatment });
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
        if self.effect_modifiers.iter().any(|m| *m == self.treatment || *m == self.outcome) {
            return Err(QueryError::ModifierOverlapsTreatmentOrOutcome);
        }
        Ok(())
    }
}

/// Temporal effect query over a discrete horizon (DESIGN.md §8).
#[derive(Clone, Debug, PartialEq)]
pub struct TemporalEffectQuery {
    /// Treatment variable.
    pub treatment: VariableId,
    /// Outcome variable.
    pub outcome: VariableId,
    /// Temporal intervention policy.
    pub policy: TemporalPolicy,
    /// Control intervention level on the treatment variable.
    pub control: Intervention,
    /// Active intervention level on the treatment variable.
    pub active: Intervention,
    /// Outcome horizon in time steps after the policy origin (must be ≥ 1).
    pub horizon_steps: u32,
    /// Optional max history lag (steps) to retain when unfolding; `None` = planner default.
    pub max_history_lag: Option<u32>,
    /// Target population.
    pub target_population: TargetPopulation,
}

impl TemporalEffectQuery {
    /// Pulse intervention at step 0 with active float level; control is 0.0.
    #[must_use]
    pub fn pulse(treatment: VariableId, outcome: VariableId, active_level: f64) -> Self {
        Self {
            treatment,
            outcome,
            policy: TemporalPolicy::pulse(0),
            control: Intervention::set(treatment, Value::f64(0.0)),
            active: Intervention::set(treatment, Value::f64(active_level)),
            horizon_steps: 1,
            max_history_lag: None,
            target_population: TargetPopulation::AllObserved,
        }
    }

    /// Sustained intervention on `[0, until]` with active float level; control is 0.0.
    #[must_use]
    pub fn sustained(
        treatment: VariableId,
        outcome: VariableId,
        until: i32,
        active_level: f64,
    ) -> Self {
        Self {
            treatment,
            outcome,
            policy: TemporalPolicy::sustained(0, until),
            control: Intervention::set(treatment, Value::f64(0.0)),
            active: Intervention::set(treatment, Value::f64(active_level)),
            horizon_steps: 1,
            max_history_lag: None,
            target_population: TargetPopulation::AllObserved,
        }
    }

    /// Set outcome evaluation horizon in time steps.
    #[must_use]
    pub const fn with_horizon_steps(mut self, horizon_steps: u32) -> Self {
        self.horizon_steps = horizon_steps;
        self
    }

    /// Set optional max history lag for unfolding.
    #[must_use]
    pub const fn with_max_history_lag(mut self, max_history_lag: Option<u32>) -> Self {
        self.max_history_lag = max_history_lag;
        self
    }

    /// Replace the temporal policy.
    #[must_use]
    pub const fn with_policy(mut self, policy: TemporalPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Set target population.
    #[must_use]
    pub fn with_target_population(mut self, population: TargetPopulation) -> Self {
        self.target_population = population;
        self
    }

    /// Validate treatment/outcome, interventions, policy, and horizon.
    ///
    /// # Errors
    ///
    /// Returns [`QueryError`] on inconsistent configuration.
    pub fn validate(&self) -> Result<(), QueryError> {
        if self.treatment == self.outcome {
            return Err(QueryError::TreatmentEqualsOutcome { id: self.treatment });
        }
        if self.horizon_steps == 0 {
            return Err(QueryError::NonPositiveHorizon);
        }
        self.policy.validate().map_err(|e| match e {
            crate::intervention::InterventionError::InvalidTemporalWindow { from, until } => {
                QueryError::InvalidTemporalWindow { from, until }
            }
            other => QueryError::InvalidIntervention(other.to_string()),
        })?;
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

    /// Treatment time offset for the configured policy (Pulse `at` / Sustained `from`).
    #[must_use]
    pub fn treatment_offset(&self) -> i32 {
        #[allow(unreachable_patterns)] // `TemporalPolicy` is `#[non_exhaustive]`
        match self.policy {
            TemporalPolicy::Pulse { at } => at,
            TemporalPolicy::Sustained { from, .. } => from,
            _ => 0,
        }
    }

    /// Outcome evaluation offset: `horizon_steps - 1` (absolute from window origin).
    #[must_use]
    pub fn outcome_offset(&self) -> i32 {
        i32::try_from(self.horizon_steps.saturating_sub(1)).unwrap_or(i32::MAX)
    }
}

/// Which mediation contrast to identify / estimate (linear SEM path).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum MediationContrast {
    /// Total effect (direct + mediated).
    Total,
    /// Controlled / path-product direct effect (holding mediators fixed).
    Direct,
    /// Mediated / indirect effect (path through mediators).
    Mediated,
    /// Natural direct effect (linear SEM: coincides with controlled direct under linearity).
    NaturalDirect,
    /// Natural indirect effect (linear SEM: coincides with mediated under linearity).
    NaturalIndirect,
}

/// Mediation query: treatment → mediators → outcome (DESIGN.md §8).
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct MediationQuery {
    /// Treatment variable.
    pub treatment: VariableId,
    /// Outcome variable.
    pub outcome: VariableId,
    /// Mediator set (non-empty).
    pub mediators: Arc<[VariableId]>,
    /// Contrast of interest.
    pub contrast: MediationContrast,
    /// Control intervention level.
    pub control: Intervention,
    /// Active intervention level.
    pub active: Intervention,
    /// Target population.
    pub target_population: TargetPopulation,
}

impl MediationQuery {
    /// Linear mediation with binary 0/1 treatment levels.
    #[must_use]
    pub fn binary(
        treatment: VariableId,
        outcome: VariableId,
        mediators: impl Into<Arc<[VariableId]>>,
        contrast: MediationContrast,
    ) -> Self {
        Self {
            treatment,
            outcome,
            mediators: mediators.into(),
            contrast,
            control: Intervention::set(treatment, Value::f64(0.0)),
            active: Intervention::set(treatment, Value::f64(1.0)),
            target_population: TargetPopulation::AllObserved,
        }
    }

    /// Validate ids and interventions.
    ///
    /// # Errors
    ///
    /// Empty mediators, overlaps, or inconsistent interventions.
    pub fn validate(&self) -> Result<(), QueryError> {
        if self.treatment == self.outcome {
            return Err(QueryError::TreatmentEqualsOutcome { id: self.treatment });
        }
        if self.mediators.is_empty() {
            return Err(QueryError::EmptyMediators);
        }
        if self.mediators.iter().any(|&m| m == self.treatment || m == self.outcome) {
            return Err(QueryError::MediatorOverlapsTreatmentOrOutcome);
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

/// Conditional average effect given effect modifiers (DESIGN.md §8).
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct ConditionalEffectQuery {
    /// Inner ATE-style query; `effect_modifiers` must be non-empty.
    pub inner: AverageEffectQuery,
}

impl ConditionalEffectQuery {
    /// Wrap an ATE query that already carries modifiers.
    ///
    /// # Errors
    ///
    /// Empty effect modifiers.
    pub fn try_new(inner: AverageEffectQuery) -> Result<Self, QueryError> {
        if inner.effect_modifiers.is_empty() {
            return Err(QueryError::EmptyEffectModifiers);
        }
        inner.validate()?;
        Ok(Self { inner })
    }

    /// Validate.
    ///
    /// # Errors
    ///
    /// Empty modifiers or invalid inner query.
    pub fn validate(&self) -> Result<(), QueryError> {
        if self.inner.effect_modifiers.is_empty() {
            return Err(QueryError::EmptyEffectModifiers);
        }
        self.inner.validate()
    }
}

/// Top-level causal query enum.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum CausalQuery {
    /// Average / population effect (static).
    AverageEffect(AverageEffectQuery),
    /// Temporal effect over a discrete horizon.
    TemporalEffect(TemporalEffectQuery),
    /// Counterfactual / unit-level what-if query .
    Counterfactual(CounterfactualQuery),
    /// Anomaly attribution for one or more units .
    AnomalyAttribution(AnomalyAttributionQuery),
    /// Distribution / population change attribution .
    ChangeAttribution(ChangeAttributionQuery),
    /// Mechanism-change detection — not attribution .
    MechanismChange(MechanismChangeQuery),
    /// Per-unit change attribution .
    UnitChange(UnitChangeQuery),
    /// Mediation (direct / mediated / natural effects).
    Mediation(MediationQuery),
    /// Conditional average effect given modifiers.
    ConditionalEffect(ConditionalEffectQuery),
}

/// Counterfactual query over factual observations and interventions (DESIGN.md §16).
#[derive(Clone, Debug, PartialEq)]
pub struct CounterfactualQuery {
    /// Outcome variable(s) to predict under the counterfactual world.
    pub outcomes: Arc<[VariableId]>,
    /// Interventions defining the counterfactual world (applied after abduction).
    pub interventions: Arc<[Intervention]>,
    /// When true, allow nested counterfactual interventions under invertible SCMs.
    pub allow_nested: bool,
}

impl CounterfactualQuery {
    /// Construct a single-outcome counterfactual query.
    #[must_use]
    pub fn new(outcome: VariableId, interventions: impl Into<Arc<[Intervention]>>) -> Self {
        Self {
            outcomes: Arc::from([outcome]),
            interventions: interventions.into(),
            allow_nested: false,
        }
    }

    /// Enable nested interventions where the model supports them.
    #[must_use]
    pub const fn with_nested(mut self, allow_nested: bool) -> Self {
        self.allow_nested = allow_nested;
        self
    }

    /// Validate interventions.
    ///
    /// # Errors
    ///
    /// Empty outcomes or invalid interventions.
    pub fn validate(&self) -> Result<(), QueryError> {
        if self.outcomes.is_empty() {
            return Err(QueryError::EmptyCounterfactualOutcomes);
        }
        for iv in self.interventions.iter() {
            iv.validate().map_err(|e| QueryError::InvalidIntervention(e.to_string()))?;
        }
        Ok(())
    }
}

/// Anomaly attribution query for observed units (DESIGN.md §17 basic).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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

/// Population / period selector for change attribution (DESIGN.md §17.2).
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

/// Which structural pieces participate in change decomposition (DESIGN.md §17.2).
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

/// Shapley estimation mode (DESIGN.md §17.2 / §17.4).
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

/// Configuration for Shapley allocation (DESIGN.md §17.2 / §34.3).
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

/// How to allocate total change across components (DESIGN.md §17.2).
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

/// Distribution / population change attribution query (DESIGN.md §17.2).
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
    /// Construct with Shapley Monte Carlo allocation (common DoWhy-GCM path).
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

/// Mechanism-change *detection* query (DESIGN.md §17.3) — not attribution.
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

/// Per-unit change attribution query (DESIGN.md §17.1).
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
            components: AttributionComponents::InputsAndMechanisms,
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

impl CausalQuery {
    /// Construct an average-effect query.
    #[must_use]
    pub fn average_effect(query: AverageEffectQuery) -> Self {
        Self::AverageEffect(query)
    }

    /// Construct a temporal-effect query.
    #[must_use]
    pub fn temporal_effect(query: TemporalEffectQuery) -> Self {
        Self::TemporalEffect(query)
    }

    /// Construct a counterfactual query.
    #[must_use]
    pub fn counterfactual(query: CounterfactualQuery) -> Self {
        Self::Counterfactual(query)
    }

    /// Construct an anomaly attribution query.
    #[must_use]
    pub fn anomaly_attribution(query: AnomalyAttributionQuery) -> Self {
        Self::AnomalyAttribution(query)
    }

    /// Construct a change attribution query.
    #[must_use]
    pub fn change_attribution(query: ChangeAttributionQuery) -> Self {
        Self::ChangeAttribution(query)
    }

    /// Construct a mechanism-change detection query.
    #[must_use]
    pub fn mechanism_change(query: MechanismChangeQuery) -> Self {
        Self::MechanismChange(query)
    }

    /// Construct a unit-change attribution query.
    #[must_use]
    pub fn unit_change(query: UnitChangeQuery) -> Self {
        Self::UnitChange(query)
    }

    /// Construct a mediation query.
    #[must_use]
    pub fn mediation(query: MediationQuery) -> Self {
        Self::Mediation(query)
    }

    /// Construct a conditional-effect query.
    #[must_use]
    pub fn conditional_effect(query: ConditionalEffectQuery) -> Self {
        Self::ConditionalEffect(query)
    }

    /// Whether this query is the static ATE path.
    #[must_use]
    pub const fn is_static_ate(&self) -> bool {
        matches!(self, Self::AverageEffect(_))
    }

    /// Whether this query is a temporal effect.
    #[must_use]
    pub const fn is_temporal_effect(&self) -> bool {
        matches!(self, Self::TemporalEffect(_))
    }

    /// Whether this query is counterfactual.
    #[must_use]
    pub const fn is_counterfactual(&self) -> bool {
        matches!(self, Self::Counterfactual(_))
    }

    /// Whether this query is mediation.
    #[must_use]
    pub const fn is_mediation(&self) -> bool {
        matches!(self, Self::Mediation(_))
    }

    /// Whether this query is a conditional effect.
    #[must_use]
    pub const fn is_conditional_effect(&self) -> bool {
        matches!(self, Self::ConditionalEffect(_))
    }

    /// Validate the inner query.
    ///
    /// # Errors
    ///
    /// Propagates inner [`QueryError`].
    pub fn validate(&self) -> Result<(), QueryError> {
        match self {
            Self::AverageEffect(q) => q.validate(),
            Self::TemporalEffect(q) => q.validate(),
            Self::Counterfactual(q) => q.validate(),
            Self::AnomalyAttribution(q) => q.validate(),
            Self::ChangeAttribution(q) => q.validate(),
            Self::MechanismChange(q) => q.validate(),
            Self::UnitChange(q) => q.validate(),
            Self::Mediation(q) => q.validate(),
            Self::ConditionalEffect(q) => q.validate(),
        }
    }
}

/// Errors from query construction or validation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum QueryError {
    /// Treatment and outcome are the same variable.
    TreatmentEqualsOutcome {
        /// Shared id.
        id: VariableId,
    },
    /// Intervention does not target the declared treatment.
    InterventionVariableMismatch {
        /// Expected treatment id.
        expected: VariableId,
        /// Actual intervention target.
        got: VariableId,
    },
    /// Intervention sequence has no unique target variable.
    AmbiguousInterventionTarget,
    /// Effect modifier overlaps treatment or outcome.
    ModifierOverlapsTreatmentOrOutcome,
    /// Sustained window has `until < from`.
    InvalidTemporalWindow {
        /// Window start.
        from: i32,
        /// Window end.
        until: i32,
    },
    /// Horizon must be at least one time step.
    NonPositiveHorizon,
    /// Nested intervention failed validation.
    InvalidIntervention(String),
    /// Counterfactual query has no outcomes.
    EmptyCounterfactualOutcomes,
    /// Anomaly query has no targets.
    EmptyAnomalyTargets,
    /// Anomaly `max_units` must be ≥ 1.
    NonPositiveAnomalyLimit,
    /// Mediation query has no mediators.
    EmptyMediators,
    /// Mediator overlaps treatment or outcome.
    MediatorOverlapsTreatmentOrOutcome,
    /// Conditional effect requires non-empty modifiers.
    EmptyEffectModifiers,
    /// Population selector has no rows.
    EmptyPopulationRows,
    /// Time-range population has `end <= start`.
    InvalidPopulationTimeRange {
        /// Start.
        start: usize,
        /// End.
        end: usize,
    },
    /// Sequential allocation order is empty.
    EmptyAllocationOrder,
    /// Shapley exact component limit must be ≥ 1.
    NonPositiveShapleyLimit,
    /// Approximate Shapley sample / permutation count must be ≥ 1.
    NonPositiveShapleySamples,
    /// Change attribution `max_components` must be ≥ 1.
    NonPositiveComponentLimit,
    /// Mechanism-change query has no targets.
    EmptyMechanismChangeTargets,
    /// Significance level must be in (0, 1).
    InvalidSignificanceLevel,
}

impl core::fmt::Display for QueryError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::TreatmentEqualsOutcome { id } => {
                write!(f, "treatment and outcome are the same variable {id}")
            }
            Self::InterventionVariableMismatch { expected, got } => {
                write!(f, "intervention targets {got}, expected treatment {expected}")
            }
            Self::AmbiguousInterventionTarget => {
                write!(f, "intervention does not have a unique target variable")
            }
            Self::ModifierOverlapsTreatmentOrOutcome => {
                write!(f, "effect modifier overlaps treatment or outcome")
            }
            Self::InvalidTemporalWindow { from, until } => {
                write!(f, "invalid temporal window [{from}, {until}]")
            }
            Self::NonPositiveHorizon => write!(f, "horizon_steps must be >= 1"),
            Self::InvalidIntervention(msg) => write!(f, "invalid intervention: {msg}"),
            Self::EmptyCounterfactualOutcomes => {
                write!(f, "counterfactual query requires at least one outcome")
            }
            Self::EmptyAnomalyTargets => write!(f, "anomaly attribution requires targets"),
            Self::NonPositiveAnomalyLimit => write!(f, "anomaly max_units must be >= 1"),
            Self::EmptyMediators => write!(f, "mediation query requires mediators"),
            Self::MediatorOverlapsTreatmentOrOutcome => {
                write!(f, "mediator overlaps treatment or outcome")
            }
            Self::EmptyEffectModifiers => {
                write!(f, "conditional effect requires non-empty effect modifiers")
            }
            Self::EmptyPopulationRows => write!(f, "population selector has no rows"),
            Self::InvalidPopulationTimeRange { start, end } => {
                write!(f, "invalid population time range [{start}, {end})")
            }
            Self::EmptyAllocationOrder => write!(f, "sequential allocation order is empty"),
            Self::NonPositiveShapleyLimit => {
                write!(f, "Shapley max_exact_components must be >= 1")
            }
            Self::NonPositiveShapleySamples => {
                write!(f, "Shapley sample / permutation count must be >= 1")
            }
            Self::NonPositiveComponentLimit => {
                write!(f, "max_components / max_targets must be >= 1")
            }
            Self::EmptyMechanismChangeTargets => {
                write!(f, "mechanism-change detection requires targets")
            }
            Self::InvalidSignificanceLevel => {
                write!(f, "significance level must be in (0, 1)")
            }
        }
    }
}

impl std::error::Error for QueryError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_ate_binds_ids_not_names() {
        let t = VariableId::from_raw(0);
        let y = VariableId::from_raw(1);
        let q = AverageEffectQuery::binary_ate(t, y);
        q.validate().unwrap();
        assert_eq!(q.treatment, t);
        assert_eq!(q.outcome, y);
        assert_eq!(q.target_population, TargetPopulation::AllObserved);
        match &q.control {
            Intervention::Set { variable, value } => {
                assert_eq!(*variable, t);
                assert_eq!(*value, Value::f64(0.0));
            }
            other => panic!("expected Set, got {other:?}"),
        }
    }

    #[test]
    fn rejects_treatment_equals_outcome() {
        let id = VariableId::from_raw(0);
        let q = AverageEffectQuery::binary_ate(id, id);
        assert!(matches!(q.validate(), Err(QueryError::TreatmentEqualsOutcome { .. })));
    }

    #[test]
    fn causal_query_static_ate_flag() {
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ));
        assert!(q.is_static_ate());
        assert!(!q.is_temporal_effect());
    }

    #[test]
    fn temporal_pulse_query() {
        let t = VariableId::from_raw(0);
        let y = VariableId::from_raw(1);
        let q = TemporalEffectQuery::pulse(t, y, -0.03).with_horizon_steps(2);
        q.validate().unwrap();
        assert_eq!(q.policy, TemporalPolicy::Pulse { at: 0 });
        assert_eq!(q.horizon_steps, 2);
        let cq = CausalQuery::temporal_effect(q);
        assert!(cq.is_temporal_effect());
        cq.validate().unwrap();
    }

    #[test]
    fn rejects_inverted_sustained_window() {
        let t = VariableId::from_raw(0);
        let y = VariableId::from_raw(1);
        let q = TemporalEffectQuery::pulse(t, y, 1.0).with_policy(TemporalPolicy::sustained(5, 1));
        assert!(matches!(q.validate(), Err(QueryError::InvalidTemporalWindow { .. })));
    }

    #[test]
    fn rejects_zero_horizon() {
        let q = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
            .with_horizon_steps(0);
        assert!(matches!(q.validate(), Err(QueryError::NonPositiveHorizon)));
    }

    #[test]
    fn counterfactual_and_anomaly_queries() {
        let y = VariableId::from_raw(1);
        let t = VariableId::from_raw(0);
        let cf = CounterfactualQuery::new(y, [Intervention::set(t, Value::f64(1.0))]);
        cf.validate().unwrap();
        assert!(CausalQuery::counterfactual(cf).is_counterfactual());
        let an = AnomalyAttributionQuery::new([y], 100);
        an.validate().unwrap();
        CausalQuery::anomaly_attribution(an).validate().unwrap();
    }

    #[test]
    fn change_attribution_query_validates() {
        let y = VariableId::from_raw(2);
        let q = ChangeAttributionQuery::new(
            y,
            PopulationSelector::TimeRange { start: 0, end: 10 },
            PopulationSelector::TimeRange { start: 10, end: 20 },
        )
        .with_components(AttributionComponents::All)
        .with_allocation(AllocationMethod::Shapley {
            approximation: ShapleyConfig::monte_carlo(100).with_seed(1),
        });
        q.validate().unwrap();
        CausalQuery::change_attribution(q).validate().unwrap();

        let bad = ChangeAttributionQuery::new(
            y,
            PopulationSelector::TimeRange { start: 5, end: 5 },
            PopulationSelector::All,
        );
        assert!(matches!(bad.validate(), Err(QueryError::InvalidPopulationTimeRange { .. })));
    }

    #[test]
    fn shapley_exact_config_rejects_zero_limit() {
        let cfg = ShapleyConfig::exact().with_max_exact_components(0);
        assert!(matches!(cfg.validate(), Err(QueryError::NonPositiveShapleyLimit)));
    }
}
