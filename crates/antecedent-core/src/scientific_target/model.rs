//! Worlds, bindings, decision time, and assumptions of a scientific target.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::assumption::{AssumptionRecord, AssumptionSet};
use crate::callback::Portability;
use crate::canonical_encode::{CanonicalEncode, count, encode_variables};
use crate::identity::CanonicalWriter;
use crate::ids::VariableId;
use crate::intervention::{Intervention, TemporalPolicy};
use crate::query::{OutcomeFunctional, TargetPopulation};
use crate::text::is_blank;
use crate::value::Value;
use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;

use super::{ActionFault, ScientificTargetError};

/// Id of one intervention world inside a target.
///
/// This is not [`crate::query::WorldId`]. That index addresses an edge route in a
/// cross-world query. This id is the caller's label for a scientific world.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ScientificWorldId(u32);

impl ScientificWorldId {
    /// Id from its raw label.
    #[must_use]
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Raw label.
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

impl fmt::Display for ScientificWorldId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "W{}", self.0)
    }
}

/// Units of a scientific quantity. `1` is dimensionless.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct QuantityUnits {
    name: Arc<str>,
}

impl QuantityUnits {
    /// Units named `name` (`"1"`, `"USD"`, `"mg"`).
    ///
    /// # Errors
    ///
    /// [`ScientificTargetError::EmptyName`] when `name` is blank.
    pub fn new(name: impl Into<Arc<str>>) -> Result<Self, ScientificTargetError> {
        let name = name.into();
        if is_blank(&name) {
            return Err(ScientificTargetError::EmptyName { field: "units" });
        }
        Ok(Self { name })
    }

    /// Dimensionless quantity (`"1"`).
    #[must_use]
    pub fn dimensionless() -> Self {
        Self { name: Arc::from("1") }
    }

    /// Unit name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Observation unit the expectation is over (`"subject"`, `"cluster"`, `"series"`).
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct ObservationUnit {
    name: Arc<str>,
}

impl ObservationUnit {
    /// Unit named `name`.
    ///
    /// # Errors
    ///
    /// [`ScientificTargetError::EmptyName`] when `name` is blank.
    pub fn new(name: impl Into<Arc<str>>) -> Result<Self, ScientificTargetError> {
        let name = name.into();
        if is_blank(&name) {
            return Err(ScientificTargetError::EmptyName { field: "observation unit" });
        }
        Ok(Self { name })
    }

    /// Unit name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// Outcome quantity: variable, functional, and units.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct OutcomeQuantity {
    outcome: VariableId,
    functional: OutcomeFunctional,
    units: QuantityUnits,
}

impl OutcomeQuantity {
    /// Quantity with an explicit functional.
    ///
    /// # Errors
    ///
    /// [`ScientificTargetError::InvalidAction`] when the functional is not a valid
    /// scientific quantity.
    pub fn new(
        outcome: VariableId,
        functional: OutcomeFunctional,
        units: QuantityUnits,
    ) -> Result<Self, ScientificTargetError> {
        functional.validate().map_err(|_| ScientificTargetError::InvalidAction {
            reason: ActionFault::OutcomeFunctional,
        })?;
        Ok(Self { outcome, functional, units })
    }

    /// Mean of `outcome` in `units`.
    ///
    /// # Errors
    ///
    /// [`ScientificTargetError::InvalidAction`] when the mean functional fails validation.
    pub fn mean(outcome: VariableId, units: QuantityUnits) -> Result<Self, ScientificTargetError> {
        Self::new(outcome, OutcomeFunctional::Mean, units)
    }

    /// Outcome variable.
    #[must_use]
    pub const fn outcome(&self) -> VariableId {
        self.outcome
    }

    /// Functional of that outcome.
    #[must_use]
    pub const fn functional(&self) -> &OutcomeFunctional {
        &self.functional
    }

    /// Units of the functional.
    #[must_use]
    pub const fn units(&self) -> &QuantityUnits {
        &self.units
    }
}

/// Variables a target conditions on. Empty is unconditional.
///
/// The stored order is sorted and de-duplicated, so declaration order is not
/// part of the target.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct Conditioning {
    on: Arc<[VariableId]>,
}

impl Conditioning {
    /// Unconditional target.
    #[must_use]
    pub fn unconditional() -> Self {
        Self { on: Arc::from([]) }
    }

    /// Condition on these variables. Duplicates collapse.
    #[must_use]
    pub fn on(variables: impl IntoIterator<Item = VariableId>) -> Self {
        let mut on: Vec<VariableId> = variables.into_iter().collect();
        on.sort_unstable();
        on.dedup();
        Self { on: Arc::from(on) }
    }

    /// Conditioning variables, sorted.
    #[must_use]
    pub fn variables(&self) -> &[VariableId] {
        &self.on
    }
}

/// History retained for a temporal target.
///
/// [`Self::planner_default`] is a different scientific choice from an explicit
/// lag, including zero. An added time column does not fill this in.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct HistoryWindow {
    cap: HistoryCap,
}

/// How the retained history was chosen. A planner choice that has been resolved to a
/// number keeps that number, so the identity pins what the planner used.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
enum HistoryCap {
    /// The planner chooses and no cap has been resolved yet.
    PlannerDefault,
    /// The planner chose, and used this many lags.
    PlannerResolved(u32),
    /// The caller declared this many lags.
    Explicit(u32),
}

impl HistoryWindow {
    /// The planner's default history cap, not yet resolved. Not an explicit lag of zero.
    ///
    /// Call [`Self::resolve`] once the planner has fixed the cap, so the window
    /// records the number it used.
    #[must_use]
    pub const fn planner_default() -> Self {
        Self { cap: HistoryCap::PlannerDefault }
    }

    /// The planner's default, resolved to the `steps` lags it used.
    ///
    /// This differs from [`Self::max_lag`] (a declared cap) and from
    /// [`Self::planner_default`] (an unresolved choice).
    #[must_use]
    pub const fn planner_resolved(steps: u32) -> Self {
        Self { cap: HistoryCap::PlannerResolved(steps) }
    }

    /// Retain at most `steps` lags. Zero is an explicit empty history.
    #[must_use]
    pub const fn max_lag(steps: u32) -> Self {
        Self { cap: HistoryCap::Explicit(steps) }
    }

    /// Record the cap the planner used. A declared cap, or one already resolved,
    /// is returned unchanged.
    #[must_use]
    pub const fn resolve(self, steps: u32) -> Self {
        match self.cap {
            HistoryCap::PlannerDefault => Self::planner_resolved(steps),
            HistoryCap::PlannerResolved(_) | HistoryCap::Explicit(_) => self,
        }
    }

    /// Explicit cap, or [`None`] when the planner chose (resolved or not).
    #[must_use]
    pub const fn max_lag_steps(self) -> Option<u32> {
        match self.cap {
            HistoryCap::Explicit(steps) => Some(steps),
            HistoryCap::PlannerDefault | HistoryCap::PlannerResolved(_) => None,
        }
    }

    /// Cap in lags, declared or planner-resolved, or [`None`] while the planner
    /// default is unresolved.
    #[must_use]
    pub const fn resolved_cap(self) -> Option<u32> {
        match self.cap {
            HistoryCap::Explicit(steps) | HistoryCap::PlannerResolved(steps) => Some(steps),
            HistoryCap::PlannerDefault => None,
        }
    }

    /// Whether the planner, not the caller, chose the cap.
    #[must_use]
    pub const fn is_planner_default(self) -> bool {
        !matches!(self.cap, HistoryCap::Explicit(_))
    }
}

/// Law of the state at the analysis origin.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct InitialLaw {
    kind: InitialLawKind,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
enum InitialLawKind {
    ContemporaneousObserved,
    Named(Arc<str>),
}

impl InitialLaw {
    /// Contemporaneous observed law.
    #[must_use]
    pub const fn contemporaneous_observed() -> Self {
        Self { kind: InitialLawKind::ContemporaneousObserved }
    }

    /// Named initial-state law. The name is the identity; this type does not fit it.
    ///
    /// # Errors
    ///
    /// [`ScientificTargetError::EmptyName`] when `name` is blank.
    pub fn named(name: impl Into<Arc<str>>) -> Result<Self, ScientificTargetError> {
        let name = name.into();
        if is_blank(&name) {
            return Err(ScientificTargetError::EmptyName { field: "initial law" });
        }
        Ok(Self { kind: InitialLawKind::Named(name) })
    }

    /// Name when this is a named law.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        match &self.kind {
            InitialLawKind::ContemporaneousObserved => None,
            InitialLawKind::Named(name) => Some(name),
        }
    }
}

/// Decision time of a target.
///
/// [`Self::static_decision`] is one decision time: no horizon index, no history
/// window, and no separate initial-state law. A temporal frame with horizon 1 is
/// a different target.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct TimeFrame {
    pub(super) kind: TimeKind,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub(super) enum TimeKind {
    Static,
    Temporal {
        horizon_steps: u32,
        history: HistoryWindow,
        initial_law: InitialLaw,
    },
    /// Solution of simultaneous structural equations: no horizon, no history and
    /// no initial-state law.
    Equilibrium,
    /// Continuous time over `(0, horizon]` from a declared initial-state law. The
    /// horizon is stored by its bit pattern, with `-0.0` refused as non-positive.
    ContinuousTime {
        horizon_bits: u64,
        initial_law: InitialLaw,
    },
    /// Stationary law of a continuous-time process.
    Stationary,
}

impl TimeFrame {
    /// One decision time.
    #[must_use]
    pub const fn static_decision() -> Self {
        Self { kind: TimeKind::Static }
    }

    /// Equilibrium of simultaneous structural equations.
    ///
    /// This is not [`Self::static_decision`]: a static frame reads one decision
    /// time of an acyclic model, an equilibrium frame reads the solution of a
    /// feedback system. Neither asserts that the solution is physically stable.
    #[must_use]
    pub const fn equilibrium() -> Self {
        Self { kind: TimeKind::Equilibrium }
    }

    /// Continuous time over `(0, horizon]` from `initial_law`.
    ///
    /// The horizon is in the model's own time units. Setting derivatives to zero
    /// is not this frame; that is [`Self::stationary`] or [`Self::equilibrium`]
    /// under its own assumptions.
    ///
    /// # Errors
    ///
    /// [`ScientificTargetError::NonPositiveContinuousHorizon`] when `horizon` is not
    /// finite or not strictly positive.
    pub fn continuous_time(
        horizon: f64,
        initial_law: InitialLaw,
    ) -> Result<Self, ScientificTargetError> {
        if !horizon.is_finite() || horizon <= 0.0 {
            return Err(ScientificTargetError::NonPositiveContinuousHorizon);
        }
        Ok(Self { kind: TimeKind::ContinuousTime { horizon_bits: horizon.to_bits(), initial_law } })
    }

    /// Stationary law of a continuous-time process. It has no horizon and no
    /// initial-state law.
    #[must_use]
    pub const fn stationary() -> Self {
        Self { kind: TimeKind::Stationary }
    }

    /// Whether this is an equilibrium frame.
    #[must_use]
    pub const fn is_equilibrium(&self) -> bool {
        matches!(&self.kind, TimeKind::Equilibrium)
    }

    /// Whether this is a stationary frame.
    #[must_use]
    pub const fn is_stationary(&self) -> bool {
        matches!(&self.kind, TimeKind::Stationary)
    }

    /// Continuous-time horizon, or [`None`] for any other frame.
    #[must_use]
    pub fn continuous_horizon(&self) -> Option<f64> {
        match &self.kind {
            TimeKind::ContinuousTime { horizon_bits, .. } => Some(f64::from_bits(*horizon_bits)),
            TimeKind::Static
            | TimeKind::Temporal { .. }
            | TimeKind::Equilibrium
            | TimeKind::Stationary => None,
        }
    }

    /// Initial-state law of a continuous-time frame, or [`None`] for any other frame.
    #[must_use]
    pub const fn continuous_initial_law(&self) -> Option<&InitialLaw> {
        match &self.kind {
            TimeKind::ContinuousTime { initial_law, .. } => Some(initial_law),
            TimeKind::Static
            | TimeKind::Temporal { .. }
            | TimeKind::Equilibrium
            | TimeKind::Stationary => None,
        }
    }

    /// Temporal frame. `horizon_steps` counts steps after the policy origin.
    ///
    /// # Errors
    ///
    /// [`ScientificTargetError::NonPositiveHorizon`] when `horizon_steps` is zero.
    pub fn temporal(
        horizon_steps: u32,
        history: HistoryWindow,
        initial_law: InitialLaw,
    ) -> Result<Self, ScientificTargetError> {
        if horizon_steps == 0 {
            return Err(ScientificTargetError::NonPositiveHorizon);
        }
        Ok(Self { kind: TimeKind::Temporal { horizon_steps, history, initial_law } })
    }

    /// Whether this is the one-decision-time specialization.
    #[must_use]
    pub const fn is_static(&self) -> bool {
        matches!(&self.kind, TimeKind::Static)
    }

    /// Horizon in steps, or [`None`] for a frame that is not discrete temporal.
    #[must_use]
    pub const fn horizon_steps(&self) -> Option<u32> {
        match &self.kind {
            TimeKind::Temporal { horizon_steps, .. } => Some(*horizon_steps),
            TimeKind::Static
            | TimeKind::Equilibrium
            | TimeKind::ContinuousTime { .. }
            | TimeKind::Stationary => None,
        }
    }

    /// History window, or [`None`] for a frame that is not discrete temporal.
    #[must_use]
    pub const fn history(&self) -> Option<HistoryWindow> {
        match &self.kind {
            TimeKind::Temporal { history, .. } => Some(*history),
            TimeKind::Static
            | TimeKind::Equilibrium
            | TimeKind::ContinuousTime { .. }
            | TimeKind::Stationary => None,
        }
    }

    /// Initial-state law of a discrete temporal frame, or [`None`] otherwise.
    ///
    /// A continuous-time frame's law is [`Self::continuous_initial_law`].
    #[must_use]
    pub const fn initial_law(&self) -> Option<&InitialLaw> {
        match &self.kind {
            TimeKind::Temporal { initial_law, .. } => Some(initial_law),
            TimeKind::Static
            | TimeKind::Equilibrium
            | TimeKind::ContinuousTime { .. }
            | TimeKind::Stationary => None,
        }
    }
}

/// Graph family the target is stated under. This is an assumption, not a license to estimate.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum GraphFamily {
    /// No graph family declared.
    Unspecified,
    /// Directed acyclic graph.
    Dag,
    /// Acyclic directed mixed graph.
    Admg,
    /// Completed partially directed acyclic graph.
    Cpdag,
    /// Partial ancestral graph.
    Pag,
    /// Temporal directed acyclic graph.
    TemporalDag,
    /// Temporal completed partially directed acyclic graph.
    TemporalCpdag,
    /// Temporal partial ancestral graph.
    TemporalPag,
    /// Temporal acyclic directed mixed graph.
    TemporalAdmg,
    /// Co-determined variables.
    CoDetermined,
    /// Design-based target with no causal graph.
    Graphless,
    /// Directed graph whose mechanisms may be reciprocal, so it may contain directed
    /// cycles. `A → B` and `B → A` are two mechanisms; latent confounding stays a
    /// separate bidirected relation. Acyclic guarantees do not transfer to it.
    Cyclic,
}

impl GraphFamily {
    /// Canonical tag. Exhaustive, so a new family must be numbered here. Tags are
    /// append-only: an existing family keeps its number.
    const fn canonical_tag(self) -> u64 {
        match self {
            Self::Unspecified => 1,
            Self::Dag => 2,
            Self::Admg => 3,
            Self::Cpdag => 4,
            Self::Pag => 5,
            Self::TemporalDag => 6,
            Self::TemporalCpdag => 7,
            Self::TemporalPag => 8,
            Self::TemporalAdmg => 9,
            Self::CoDetermined => 10,
            Self::Graphless => 11,
            Self::Cyclic => 12,
        }
    }
}

/// Graph family plus the assumption records the target is conditional on.
///
/// The identity is the sorted, de-duplicated set of (assumption, scope) pairs. Record
/// order, repeats, source, and status (a diagnostic may move `Declared` to
/// `Supported`) are not part of it. A prior entered as an identifying
/// restriction changes the identity. Choosing a frequentist or Bayesian
/// realization does not: that is [`InferenceSelection`].
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct ModelAssumptions {
    graph: GraphFamily,
    records: AssumptionSet,
}

impl ModelAssumptions {
    /// Assumptions under `graph`.
    #[must_use]
    pub const fn new(graph: GraphFamily, records: AssumptionSet) -> Self {
        Self { graph, records }
    }

    /// Graph family.
    #[must_use]
    pub const fn graph(&self) -> GraphFamily {
        self.graph
    }

    /// Assumption records, in declaration order.
    #[must_use]
    pub const fn records(&self) -> &AssumptionSet {
        &self.records
    }
}

/// How a later layer may realize this target. Not part of scientific equality.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum InferenceSelection {
    /// No realization chosen.
    Unspecified,
    /// Frequentist model, score, or resampling realization.
    Frequentist,
    /// Bayesian posterior realization.
    Bayesian,
    /// Generalized posterior with its own learning rate and calibration.
    GeneralizedBayesian,
}

/// Population, unit, quantity, and conditioning shared by every world in a collection.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct WorldBinding {
    population: TargetPopulation,
    unit: ObservationUnit,
    outcome: OutcomeQuantity,
    conditioning: Conditioning,
}

impl WorldBinding {
    /// Binding. Worlds in one collection must compare equal on this value.
    ///
    /// # Errors
    ///
    /// [`ScientificTargetError::InvalidAction`] when the population is not usable, or
    /// [`ScientificTargetError::ConditioningIncludesOutcome`] when conditioning names
    /// the outcome.
    pub fn new(
        population: TargetPopulation,
        unit: ObservationUnit,
        outcome: OutcomeQuantity,
        conditioning: Conditioning,
    ) -> Result<Self, ScientificTargetError> {
        population.validate().map_err(|err| ScientificTargetError::InvalidAction {
            reason: ActionFault::Population(err),
        })?;
        if conditioning.variables().contains(&outcome.outcome()) {
            return Err(ScientificTargetError::ConditioningIncludesOutcome);
        }
        Ok(Self { population, unit, outcome, conditioning })
    }

    /// Target population.
    #[must_use]
    pub const fn population(&self) -> &TargetPopulation {
        &self.population
    }

    /// Observation unit.
    #[must_use]
    pub const fn unit(&self) -> &ObservationUnit {
        &self.unit
    }

    /// Outcome quantity.
    #[must_use]
    pub const fn outcome(&self) -> &OutcomeQuantity {
        &self.outcome
    }

    /// Conditioning.
    #[must_use]
    pub const fn conditioning(&self) -> &Conditioning {
        &self.conditioning
    }
}

/// Action in one world: simultaneous interventions, or one intervention under a schedule.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct WorldAction {
    pub(super) kind: ActionKind,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub(super) enum ActionKind {
    Interventions(Arc<[Intervention]>),
    Schedule {
        intervention: Intervention,
        policy: TemporalPolicy,
    },
    /// Structural actions on a feedback or continuous-time model, sorted by
    /// variable then action kind. Empty is the model as specified.
    Structural(Arc<[StructuralAction]>),
}

/// One structural action on a feedback or continuous-time model.
///
/// The four kinds are different scientific actions and none is rewritten into
/// another. A hard clamp deletes the variable's equation and holds its level. A
/// mechanism replacement swaps the deterministic mechanism and keeps the
/// disturbance. An additive shock perturbs the structural disturbance. An
/// initial-state change replaces the declared initial law of one state. Levels are
/// stored by bit pattern with `-0.0` folded to `0.0`.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct StructuralAction {
    kind: StructuralKind,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
enum StructuralKind {
    HardClamp { variable: VariableId, level_bits: u64 },
    ReplaceMechanism { variable: VariableId, family: Arc<str>, parameter_bits: Arc<[u64]> },
    AdditiveShock { variable: VariableId, delta_bits: u64 },
    InitialState { variable: VariableId, law: Arc<str>, level_bits: u64 },
}

fn finite_bits(value: f64, what: &'static str) -> Result<u64, ScientificTargetError> {
    if !value.is_finite() {
        return Err(ScientificTargetError::InvalidAction {
            reason: ActionFault::NonFiniteLevel { what },
        });
    }
    Ok(if value == 0.0 { 0.0_f64.to_bits() } else { value.to_bits() })
}

impl StructuralAction {
    /// Delete the equation of `variable` and hold it at `level`.
    ///
    /// # Errors
    ///
    /// [`ScientificTargetError::InvalidAction`] when `level` is not finite.
    pub fn hard_clamp(variable: VariableId, level: f64) -> Result<Self, ScientificTargetError> {
        let level_bits = finite_bits(level, "a hard-clamp level")?;
        Ok(Self { kind: StructuralKind::HardClamp { variable, level_bits } })
    }

    /// Replace the deterministic mechanism of `variable` by the named `family`
    /// with `parameters`, keeping its disturbance.
    ///
    /// # Errors
    ///
    /// [`ScientificTargetError::EmptyName`] for a blank family and
    /// [`ScientificTargetError::InvalidAction`] for a non-finite parameter.
    pub fn replace_mechanism(
        variable: VariableId,
        family: impl Into<Arc<str>>,
        parameters: &[f64],
    ) -> Result<Self, ScientificTargetError> {
        let family = family.into();
        if is_blank(&family) {
            return Err(ScientificTargetError::EmptyName { field: "mechanism family" });
        }
        let parameter_bits = parameters
            .iter()
            .map(|value| finite_bits(*value, "a mechanism parameter"))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            kind: StructuralKind::ReplaceMechanism {
                variable,
                family,
                parameter_bits: Arc::from(parameter_bits),
            },
        })
    }

    /// Add `delta` to the structural disturbance of `variable`.
    ///
    /// # Errors
    ///
    /// [`ScientificTargetError::InvalidAction`] when `delta` is not finite.
    pub fn additive_shock(variable: VariableId, delta: f64) -> Result<Self, ScientificTargetError> {
        let delta_bits = finite_bits(delta, "an additive shock")?;
        Ok(Self { kind: StructuralKind::AdditiveShock { variable, delta_bits } })
    }

    /// Replace the declared initial-state law `law` of state `variable` by `level`.
    ///
    /// # Errors
    ///
    /// [`ScientificTargetError::EmptyName`] for a blank law name and
    /// [`ScientificTargetError::InvalidAction`] for a non-finite level.
    pub fn initial_state(
        variable: VariableId,
        law: impl Into<Arc<str>>,
        level: f64,
    ) -> Result<Self, ScientificTargetError> {
        let law = law.into();
        if is_blank(&law) {
            return Err(ScientificTargetError::EmptyName { field: "initial law" });
        }
        let level_bits = finite_bits(level, "an initial-state level")?;
        Ok(Self { kind: StructuralKind::InitialState { variable, law, level_bits } })
    }

    /// Variable the action targets.
    #[must_use]
    pub const fn variable(&self) -> VariableId {
        match &self.kind {
            StructuralKind::HardClamp { variable, .. }
            | StructuralKind::ReplaceMechanism { variable, .. }
            | StructuralKind::AdditiveShock { variable, .. }
            | StructuralKind::InitialState { variable, .. } => *variable,
        }
    }

    /// Stable name of the action kind.
    #[must_use]
    pub const fn kind_name(&self) -> &'static str {
        match &self.kind {
            StructuralKind::HardClamp { .. } => "hard_clamp",
            StructuralKind::ReplaceMechanism { .. } => "replace_mechanism",
            StructuralKind::AdditiveShock { .. } => "additive_shock",
            StructuralKind::InitialState { .. } => "initial_state_change",
        }
    }

    /// Clamped level, when this is a hard clamp.
    #[must_use]
    pub fn hard_clamp_level(&self) -> Option<f64> {
        match &self.kind {
            StructuralKind::HardClamp { level_bits, .. } => Some(f64::from_bits(*level_bits)),
            _ => None,
        }
    }

    /// Replacement family and parameters, when this is a mechanism replacement.
    #[must_use]
    pub fn mechanism(&self) -> Option<(&str, Vec<f64>)> {
        match &self.kind {
            StructuralKind::ReplaceMechanism { family, parameter_bits, .. } => {
                let parameters = parameter_bits.iter().map(|bits| f64::from_bits(*bits)).collect();
                Some((&**family, parameters))
            }
            _ => None,
        }
    }

    /// Disturbance perturbation, when this is an additive shock.
    #[must_use]
    pub fn shock_delta(&self) -> Option<f64> {
        match &self.kind {
            StructuralKind::AdditiveShock { delta_bits, .. } => Some(f64::from_bits(*delta_bits)),
            _ => None,
        }
    }

    /// Replaced law and new level, when this is an initial-state change.
    #[must_use]
    pub fn initial_state_change(&self) -> Option<(&str, f64)> {
        match &self.kind {
            StructuralKind::InitialState { law, level_bits, .. } => {
                Some((&**law, f64::from_bits(*level_bits)))
            }
            _ => None,
        }
    }

    /// Canonical tag. Exhaustive and append-only.
    const fn canonical_tag(&self) -> u64 {
        match &self.kind {
            StructuralKind::HardClamp { .. } => 1,
            StructuralKind::ReplaceMechanism { .. } => 2,
            StructuralKind::AdditiveShock { .. } => 3,
            StructuralKind::InitialState { .. } => 4,
        }
    }

    const fn is_initial_state(&self) -> bool {
        matches!(&self.kind, StructuralKind::InitialState { .. })
    }
}

/// Sort structural actions by variable, then kind, and refuse conflicts.
///
/// One variable carries at most one action of each kind, and a hard clamp
/// excludes every other action on its variable.
fn canonical_structural(
    actions: impl IntoIterator<Item = StructuralAction>,
) -> Result<Arc<[StructuralAction]>, ScientificTargetError> {
    let mut actions: Vec<StructuralAction> = actions.into_iter().collect();
    actions.sort_by_key(|action| (action.variable(), action.canonical_tag()));
    for pair in actions.windows(2) {
        let (left, right) = (&pair[0], &pair[1]);
        if left.variable() != right.variable() {
            continue;
        }
        let clamp = |action: &StructuralAction| action.canonical_tag() == 1;
        if left.canonical_tag() == right.canonical_tag() || clamp(left) || clamp(right) {
            return Err(ScientificTargetError::DuplicateInterventionVariable {
                id: left.variable(),
            });
        }
    }
    Ok(Arc::from(actions))
}

impl WorldAction {
    /// Simultaneous interventions at one decision time.
    ///
    /// Pair with [`TimeFrame::static_decision`]. Variables are sorted by id so
    /// declaration order is not part of the world. A callback is refused.
    ///
    /// # Errors
    ///
    /// [`ScientificTargetError::InvalidAction`] when the list is empty or an
    /// intervention is invalid, [`ScientificTargetError::CallbackAction`] when an
    /// intervention carries a callback, and
    /// [`ScientificTargetError::DuplicateInterventionVariable`] when one variable
    /// is assigned twice.
    pub fn at_decision(
        interventions: impl IntoIterator<Item = Intervention>,
    ) -> Result<Self, ScientificTargetError> {
        Ok(Self { kind: ActionKind::Interventions(canonical_interventions(interventions)?) })
    }

    /// One intervention under `policy`.
    ///
    /// Pair with [`TimeFrame::temporal`]. Pulse, sustained, and dynamic schedules
    /// stay distinct, including a single-step sustained window and a pulse at the
    /// same offset.
    ///
    /// # Errors
    ///
    /// [`ScientificTargetError::InvalidAction`], [`ScientificTargetError::CallbackAction`],
    /// or [`ScientificTargetError::DuplicateInterventionVariable`] from the intervention
    /// or the policy.
    pub fn schedule(
        intervention: Intervention,
        policy: TemporalPolicy,
    ) -> Result<Self, ScientificTargetError> {
        let intervention = canonical_intervention(intervention);
        ensure_portable(&intervention)?;
        policy.validate().map_err(|err| ScientificTargetError::InvalidAction {
            reason: ActionFault::Intervention(err),
        })?;
        Ok(Self { kind: ActionKind::Schedule { intervention, policy } })
    }

    /// Simultaneous structural actions on a feedback or continuous-time model.
    ///
    /// Pair with [`TimeFrame::equilibrium`], [`TimeFrame::stationary`] or
    /// [`TimeFrame::continuous_time`]; an initial-state change needs the
    /// continuous-time frame. An empty list is the model as specified, which is
    /// a world of its own and not an intervention at level zero. Actions are
    /// sorted by variable, then kind, so declaration order is not part of the world.
    ///
    /// # Errors
    ///
    /// [`ScientificTargetError::DuplicateInterventionVariable`] when one variable
    /// carries two actions of one kind, or a hard clamp and any other action.
    pub fn structural(
        actions: impl IntoIterator<Item = StructuralAction>,
    ) -> Result<Self, ScientificTargetError> {
        Ok(Self { kind: ActionKind::Structural(canonical_structural(actions)?) })
    }

    /// Interventions when this is a one-decision-time action.
    #[must_use]
    pub fn at_decision_interventions(&self) -> Option<&[Intervention]> {
        match &self.kind {
            ActionKind::Interventions(interventions) => Some(interventions),
            ActionKind::Schedule { .. } | ActionKind::Structural(_) => None,
        }
    }

    /// Intervention and schedule when this action is temporal.
    #[must_use]
    pub fn schedule_parts(&self) -> Option<(&Intervention, &TemporalPolicy)> {
        match &self.kind {
            ActionKind::Schedule { intervention, policy } => Some((intervention, policy)),
            ActionKind::Interventions(_) | ActionKind::Structural(_) => None,
        }
    }

    /// Structural actions, in canonical order, when this is a structural action.
    #[must_use]
    pub fn structural_actions(&self) -> Option<&[StructuralAction]> {
        match &self.kind {
            ActionKind::Structural(actions) => Some(actions),
            ActionKind::Interventions(_) | ActionKind::Schedule { .. } => None,
        }
    }
}

fn canonical_interventions(
    interventions: impl IntoIterator<Item = Intervention>,
) -> Result<Arc<[Intervention]>, ScientificTargetError> {
    let mut interventions: Vec<Intervention> =
        interventions.into_iter().map(canonical_intervention).collect();
    if interventions.is_empty() {
        return Err(ScientificTargetError::InvalidAction { reason: ActionFault::NoIntervention });
    }
    let mut seen = Vec::new();
    for intervention in &interventions {
        ensure_portable(intervention)?;
        for variable in intervention.target_variables() {
            if seen.contains(&variable) {
                return Err(ScientificTargetError::DuplicateInterventionVariable { id: variable });
            }
            seen.push(variable);
        }
    }
    interventions.sort_by_key(min_target);
    Ok(Arc::from(interventions))
}

fn min_target(intervention: &Intervention) -> u32 {
    intervention.target_variables().into_iter().map(VariableId::raw).min().unwrap_or(u32::MAX)
}

pub(super) fn ensure_portable(intervention: &Intervention) -> Result<(), ScientificTargetError> {
    intervention.validate().map_err(|err| ScientificTargetError::InvalidAction {
        reason: ActionFault::Intervention(err),
    })?;
    ensure_concrete_levels(intervention)?;
    match intervention.portability() {
        Portability::Portable => Ok(()),
        Portability::Recoverable(_) | Portability::RuntimeBound { .. } => {
            Err(ScientificTargetError::CallbackAction)
        }
    }
}

/// A world names a level it can be evaluated at, never the symbolic `do(V)` marker.
fn ensure_concrete_levels(intervention: &Intervention) -> Result<(), ScientificTargetError> {
    match intervention {
        Intervention::Set { value: level, .. } | Intervention::Shift { delta: level, .. } => {
            level.validate_concrete_intervention_level().map_err(|err| {
                ScientificTargetError::InvalidAction { reason: ActionFault::Level(err) }
            })
        }
        Intervention::Sequence(sequence) => {
            sequence.steps.iter().try_for_each(|step| ensure_concrete_levels(&step.intervention))
        }
        Intervention::Stochastic { .. }
        | Intervention::Soft { .. }
        | Intervention::Transform { .. }
        | Intervention::Assign { .. } => Ok(()),
    }
}

/// `-0.0` levels are stored as `+0.0`, so the two spellings are one world.
pub(super) fn canonical_intervention(intervention: Intervention) -> Intervention {
    let zero = |value: Value| match value {
        Value::Float64(number) if number.to_bits() == (-0.0_f64).to_bits() => Value::Float64(0.0),
        other => other,
    };
    match intervention {
        Intervention::Set { variable, value } => Intervention::Set { variable, value: zero(value) },
        Intervention::Shift { variable, delta } => {
            Intervention::Shift { variable, delta: zero(delta) }
        }
        other => other,
    }
}

/// Offsets are relative to the policy origin and must lie in `[-horizon, +horizon]`.
///
/// A negative offset inside that interval is history the initial law supplies, and
/// a pulse there is a legitimate target. Horizon one admits `pulse(-1)` and refuses
/// `pulse(-2)`. A positive offset is admitted through `+horizon`.
pub(super) fn ensure_policy_within_horizon(
    policy: &TemporalPolicy,
    time: &TimeFrame,
) -> Result<(), ScientificTargetError> {
    let Some(horizon) = time.horizon_steps() else {
        return Ok(());
    };
    let bound = i64::from(horizon);
    let inside = |step: i32| (-bound..=bound).contains(&i64::from(step));
    let ok = match policy {
        TemporalPolicy::Pulse { at } => inside(*at),
        TemporalPolicy::Sustained { from, until } => inside(*from) && inside(*until),
        TemporalPolicy::Dynamic { active_at, .. } => active_at.iter().all(|step| inside(*step)),
    };
    if ok {
        Ok(())
    } else {
        Err(ScientificTargetError::PolicyOutsideHorizon {
            horizon_steps: horizon,
            policy: format!("{policy:?}"),
        })
    }
}

fn ensure_action_within_horizon(
    action: &WorldAction,
    time: &TimeFrame,
) -> Result<(), ScientificTargetError> {
    match &action.kind {
        ActionKind::Interventions(_) | ActionKind::Structural(_) => Ok(()),
        ActionKind::Schedule { intervention, policy } => {
            ensure_policy_within_horizon(policy, time)?;
            if let Intervention::Sequence(sequence) = intervention {
                for step in sequence.steps.iter() {
                    ensure_policy_within_horizon(&step.temporal, time)?;
                }
            }
            Ok(())
        }
    }
}

fn action_matches_time(action: &WorldAction, time: &TimeFrame) -> bool {
    match (&action.kind, &time.kind) {
        (ActionKind::Interventions(_), TimeKind::Static)
        | (ActionKind::Schedule { .. }, TimeKind::Temporal { .. })
        | (ActionKind::Structural(_), TimeKind::ContinuousTime { .. }) => true,
        (ActionKind::Structural(actions), TimeKind::Equilibrium | TimeKind::Stationary) => {
            !actions.iter().any(StructuralAction::is_initial_state)
        }
        (ActionKind::Interventions(_) | ActionKind::Schedule { .. }, _)
        | (ActionKind::Structural(_), TimeKind::Static | TimeKind::Temporal { .. }) => false,
    }
}

/// One intervention world.
///
/// `reference` marks an explicit reference intervention. It does not mean the
/// variable is untreated, and it is not implied by a level of zero.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct InterventionWorld {
    id: ScientificWorldId,
    action: WorldAction,
    reference: bool,
    binding: WorldBinding,
    time: TimeFrame,
}

impl InterventionWorld {
    /// World. A static frame takes [`WorldAction::at_decision`]; a temporal frame
    /// takes [`WorldAction::schedule`].
    ///
    /// # Errors
    ///
    /// [`ScientificTargetError::MismatchedActionTime`] when the action and the frame disagree,
    /// or [`ScientificTargetError::PolicyOutsideHorizon`] when a schedule reaches outside
    /// the frame's horizon.
    pub fn new(
        id: ScientificWorldId,
        action: WorldAction,
        reference: bool,
        binding: WorldBinding,
        time: TimeFrame,
    ) -> Result<Self, ScientificTargetError> {
        if !action_matches_time(&action, &time) {
            return Err(ScientificTargetError::MismatchedActionTime);
        }
        ensure_action_within_horizon(&action, &time)?;
        Ok(Self { id, action, reference, binding, time })
    }

    /// World id.
    #[must_use]
    pub const fn id(&self) -> ScientificWorldId {
        self.id
    }

    /// Intervention or schedule.
    #[must_use]
    pub const fn action(&self) -> &WorldAction {
        &self.action
    }

    /// Whether this world is the explicit reference intervention.
    #[must_use]
    pub const fn is_reference(&self) -> bool {
        self.reference
    }

    /// Shared scientific binding.
    #[must_use]
    pub const fn binding(&self) -> &WorldBinding {
        &self.binding
    }

    /// Decision time, horizon, history, and initial law.
    #[must_use]
    pub const fn time(&self) -> &TimeFrame {
        &self.time
    }
}

/// Worlds that bind one population, unit, outcome, units, quantity, and conditioning.
///
/// Horizon and history may differ. A contrast refuses that; a dose × time surface requires it.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct WorldCollection {
    worlds: Vec<InterventionWorld>,
}

impl WorldCollection {
    /// Collection sorted by world id.
    ///
    /// # Errors
    ///
    /// [`ScientificTargetError::EmptyWorlds`], [`ScientificTargetError::DuplicateWorldId`],
    /// or [`ScientificTargetError::IncompatibleBinding`].
    pub fn new(
        worlds: impl IntoIterator<Item = InterventionWorld>,
    ) -> Result<Self, ScientificTargetError> {
        let mut worlds: Vec<InterventionWorld> = worlds.into_iter().collect();
        if worlds.is_empty() {
            return Err(ScientificTargetError::EmptyWorlds);
        }
        worlds.sort_by_key(InterventionWorld::id);
        if let Some(pair) = worlds.windows(2).find(|pair| pair[0].id() == pair[1].id()) {
            return Err(ScientificTargetError::DuplicateWorldId { id: pair[0].id() });
        }
        let anchor = &worlds[0].binding;
        for world in &worlds[1..] {
            if let Some(field) = binding_mismatch(anchor, &world.binding) {
                return Err(ScientificTargetError::IncompatibleBinding { field });
            }
        }
        Ok(Self { worlds })
    }

    /// Worlds in id order.
    #[must_use]
    pub fn worlds(&self) -> &[InterventionWorld] {
        &self.worlds
    }

    /// World with this id.
    #[must_use]
    pub fn get(&self, id: ScientificWorldId) -> Option<&InterventionWorld> {
        let index = self.worlds.binary_search_by_key(&id, InterventionWorld::id).ok()?;
        self.worlds.get(index)
    }

    /// Number of worlds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.worlds.len()
    }

    /// Whether the collection is empty. A constructed collection is not empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.worlds.is_empty()
    }
}

fn binding_mismatch(left: &WorldBinding, right: &WorldBinding) -> Option<&'static str> {
    if left.population != right.population {
        return Some("population");
    }
    if left.unit != right.unit {
        return Some("unit");
    }
    if left.outcome.outcome != right.outcome.outcome {
        return Some("outcome");
    }
    if left.outcome.units != right.outcome.units {
        return Some("units");
    }
    if left.outcome.functional != right.outcome.functional {
        return Some("quantity");
    }
    if left.conditioning != right.conditioning {
        return Some("conditioning");
    }
    None
}

/// One history of a finite law over history-indexed experiments.
///
/// `levels` follows the coordinate order of the law. The mass is stored by its
/// bit pattern so `-0.0` and `0.0` are one value.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct HistoryIndexedExperiment {
    levels: Arc<[u32]>,
    mass_bits: u64,
}

impl HistoryIndexedExperiment {
    /// Level index of each coordinate.
    #[must_use = "the levels are the history this experiment names"]
    pub fn levels(&self) -> &[u32] {
        &self.levels
    }

    /// Probability of this history.
    #[must_use = "the mass is the probability of this history"]
    pub fn mass(&self) -> f64 {
        f64::from_bits(self.mass_bits)
    }
}

/// Observational law of a target over history-indexed experiments.
///
/// This is the law a two-step horizon reads. A unit history, a source
/// experiment, and an [`InitialLaw`] do not stand in for it.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct HistoryIndexedTargetLaw {
    snapshot: Arc<str>,
    coordinates: Arc<[VariableId]>,
    experiments: Arc<[HistoryIndexedExperiment]>,
}

impl HistoryIndexedTargetLaw {
    /// Declare the law. Experiments are stored in history order.
    ///
    /// # Errors
    /// A blank snapshot, no coordinates or experiments, a repeated coordinate
    /// or history, a width mismatch, a non-finite or negative mass, or masses
    /// that do not sum to one within `1e-8`.
    pub fn new(
        snapshot: impl Into<Arc<str>>,
        coordinates: impl IntoIterator<Item = VariableId>,
        experiments: impl IntoIterator<Item = (Vec<u32>, f64)>,
    ) -> Result<Self, ScientificTargetError> {
        let snapshot = snapshot.into();
        let mut seen = BTreeSet::new();
        let mut coordinates_vec = Vec::new();
        for id in coordinates {
            if !seen.insert(id) {
                return Err(ScientificTargetError::DuplicateHistoryCoordinate { id });
            }
            coordinates_vec.push(id);
        }
        if is_blank(&snapshot) || coordinates_vec.is_empty() {
            return Err(ScientificTargetError::EmptyHistoryIndexedLaw);
        }
        let width = coordinates_vec.len();
        let mut stored = Vec::new();
        for (index, (levels, mass)) in experiments.into_iter().enumerate() {
            if levels.len() != width {
                return Err(ScientificTargetError::HistoryWidth { index });
            }
            stored.push(HistoryIndexedExperiment {
                levels: Arc::from(levels),
                mass_bits: history_mass_bits(mass)?,
            });
        }
        if stored.is_empty() {
            return Err(ScientificTargetError::EmptyHistoryIndexedLaw);
        }
        stored.sort_by(|left, right| left.levels.as_ref().cmp(right.levels.as_ref()));
        if stored.windows(2).any(|pair| pair[0].levels == pair[1].levels) {
            return Err(ScientificTargetError::DuplicateHistory);
        }
        let mut total = 0.0;
        for experiment in &stored {
            total += experiment.mass();
        }
        if !total.is_finite() || (total - 1.0).abs() > 1e-8 {
            return Err(ScientificTargetError::UnnormalizedHistoryLaw);
        }
        Ok(Self {
            snapshot,
            coordinates: Arc::from(coordinates_vec),
            experiments: Arc::from(stored),
        })
    }

    /// Snapshot the masses were taken from.
    #[must_use = "the snapshot identifies the law the masses came from"]
    pub fn snapshot(&self) -> &str {
        &self.snapshot
    }

    /// Coordinates, in the order of each experiment's levels.
    #[must_use = "the coordinates order each experiment's levels"]
    pub fn coordinates(&self) -> &[VariableId] {
        &self.coordinates
    }

    /// Histories and their masses, in history order.
    #[must_use = "the experiments are the history-indexed law"]
    pub fn experiments(&self) -> &[HistoryIndexedExperiment] {
        &self.experiments
    }
}

fn history_mass_bits(mass: f64) -> Result<u64, ScientificTargetError> {
    if !mass.is_finite() || mass < 0.0 {
        return Err(ScientificTargetError::InvalidHistoryMass);
    }
    if mass == 0.0 { Ok(0.0f64.to_bits()) } else { Ok(mass.to_bits()) }
}

impl CanonicalEncode for OutcomeQuantity {
    fn encode(&self, writer: &mut CanonicalWriter) {
        self.outcome.encode(writer);
        writer.text(self.units.name());
        self.functional.encode(writer);
    }
}

impl CanonicalEncode for WorldBinding {
    fn encode(&self, writer: &mut CanonicalWriter) {
        self.population.encode(writer);
        writer.text(self.unit.name());
        self.outcome.encode(writer);
        encode_variables(writer, self.conditioning.variables());
    }
}

impl CanonicalEncode for HistoryWindow {
    fn encode(&self, writer: &mut CanonicalWriter) {
        match self.cap {
            HistoryCap::PlannerDefault => {
                writer.word(0);
            }
            HistoryCap::Explicit(steps) => {
                writer.word(1).word(u64::from(steps));
            }
            HistoryCap::PlannerResolved(steps) => {
                writer.word(2).word(u64::from(steps));
            }
        }
    }
}

impl CanonicalEncode for InitialLaw {
    fn encode(&self, writer: &mut CanonicalWriter) {
        match &self.kind {
            InitialLawKind::ContemporaneousObserved => {
                writer.word(1);
            }
            InitialLawKind::Named(name) => {
                writer.word(2).text(name);
            }
        }
    }
}

impl CanonicalEncode for TimeFrame {
    fn encode(&self, writer: &mut CanonicalWriter) {
        match &self.kind {
            TimeKind::Static => {
                writer.word(1);
            }
            TimeKind::Temporal { horizon_steps, history, initial_law } => {
                writer.word(2).word(u64::from(*horizon_steps));
                history.encode(writer);
                initial_law.encode(writer);
            }
            TimeKind::Equilibrium => {
                writer.word(3);
            }
            TimeKind::ContinuousTime { horizon_bits, initial_law } => {
                writer.word(4).word(*horizon_bits);
                initial_law.encode(writer);
            }
            TimeKind::Stationary => {
                writer.word(5);
            }
        }
    }
}

impl CanonicalEncode for StructuralAction {
    fn encode(&self, writer: &mut CanonicalWriter) {
        writer.word(self.canonical_tag());
        self.variable().encode(writer);
        match &self.kind {
            StructuralKind::HardClamp { level_bits, .. } => {
                writer.word(*level_bits);
            }
            StructuralKind::ReplaceMechanism { family, parameter_bits, .. } => {
                writer.text(family).word(count(parameter_bits.len()));
                for bits in parameter_bits.iter() {
                    writer.word(*bits);
                }
            }
            StructuralKind::AdditiveShock { delta_bits, .. } => {
                writer.word(*delta_bits);
            }
            StructuralKind::InitialState { law, level_bits, .. } => {
                writer.text(law).word(*level_bits);
            }
        }
    }
}

impl CanonicalEncode for WorldAction {
    fn encode(&self, writer: &mut CanonicalWriter) {
        match &self.kind {
            ActionKind::Interventions(interventions) => {
                writer.word(1).word(count(interventions.len()));
                for intervention in interventions.as_ref() {
                    intervention.encode(writer);
                }
            }
            ActionKind::Schedule { intervention, policy } => {
                writer.word(2);
                intervention.encode(writer);
                policy.encode(writer);
            }
            ActionKind::Structural(actions) => {
                writer.word(3).word(count(actions.len()));
                for action in actions.iter() {
                    action.encode(writer);
                }
            }
        }
    }
}

/// What a world says scientifically, without the caller's label or reference flag.
impl CanonicalEncode for InterventionWorld {
    fn encode(&self, writer: &mut CanonicalWriter) {
        self.action.encode(writer);
        self.binding.encode(writer);
        self.time.encode(writer);
    }
}

impl InterventionWorld {
    fn content_digest(&self) -> [u8; 32] {
        let mut writer = CanonicalWriter::new("antecedent.scientific_target.world");
        self.encode(&mut writer);
        writer.finish()
    }
}

/// A collection is the set of its worlds' contents, so world order and labels do not enter.
impl CanonicalEncode for WorldCollection {
    fn encode(&self, writer: &mut CanonicalWriter) {
        let mut digests: Vec<[u8; 32]> =
            self.worlds.iter().map(InterventionWorld::content_digest).collect();
        digests.sort_unstable();
        writer.word(count(digests.len()));
        for digest in &digests {
            writer.bytes(digest);
        }
    }
}

impl CanonicalEncode for ModelAssumptions {
    fn encode(&self, writer: &mut CanonicalWriter) {
        writer.word(self.graph.canonical_tag());
        // What is assumed, and where, is the target. How it entered, whether a diagnostic
        // has since supported it, record order, and repeats are not.
        let records: BTreeSet<[u8; 32]> = self.records.entries.iter().map(record_digest).collect();
        writer.word(count(records.len()));
        for digest in records {
            writer.bytes(&digest);
        }
    }
}

fn record_digest(record: &AssumptionRecord) -> [u8; 32] {
    let mut writer = CanonicalWriter::new("antecedent.scientific_target.assumption.v1");
    record.assumption.encode(&mut writer);
    record.scope.encode(&mut writer);
    writer.finish()
}

impl CanonicalEncode for HistoryIndexedTargetLaw {
    fn encode(&self, writer: &mut CanonicalWriter) {
        writer.word(1);
        writer.text(&self.snapshot);
        encode_variables(writer, &self.coordinates);
        writer.word(count(self.experiments.len()));
        for experiment in self.experiments.iter() {
            writer.word(count(experiment.levels.len()));
            for level in experiment.levels.iter() {
                writer.word(u64::from(*level));
            }
            writer.word(experiment.mass_bits);
        }
    }
}
