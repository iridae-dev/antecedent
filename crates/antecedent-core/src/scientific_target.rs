//! Inference-independent scientific target.
//!
//! A [`crate::CausalQuery`] is the question a route answers. This module is the
//! identity those questions are comparable under: population and observation
//! unit, quantity and units, intervention worlds, conditioning, decision time,
//! and graph or model assumptions. A static analysis is the one-decision-time
//! specialization. It is not implied by a horizon of one.
//!
//! A contrast assembles intervention worlds as one target. The reference
//! intervention is an explicit world, not the absence of treatment. A dose × time
//! surface is a declared grid of those worlds. Mediation is a separate target:
//! it carries cross-world premises, and a contrast does not construct it.
//!
//! Nothing here identifies, fits, or estimates. [`InferenceSelection`] is recorded
//! on a target and omitted from [`ScientificTarget::identity`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::assumption::{Assumption, AssumptionRecord, AssumptionScope, AssumptionSet};
use crate::callback::{CallbackImplementation, CallbackOutput, Portability};
use crate::identity::{CanonicalWriter, SemanticDigest};
use crate::ids::VariableId;
use crate::intervention::{Intervention, StochasticPolicy, TemporalPolicy};
use crate::query::{OutcomeFunctional, PredicateExpr, TargetPopulation};
use crate::text::is_blank;
use crate::value::Value;
use crate::value_expr::{ArithOp, CompareOp, LogicOp, PolicyReads, ValueExpr};

/// Domain tag for [`ScientificTarget::identity`].
pub const SCIENTIFIC_TARGET_IDENTITY_TAG: &str = "antecedent.scientific_target.v1";

/// Payload version written after the domain tag. Bump when the canonical fields change.
///
/// Format 3 identifies a world, a world collection, and a dose × time surface by
/// scientific content. A caller-chosen world id and a reference flag do not enter.
pub const SCIENTIFIC_TARGET_FORMAT: u16 = 3;

/// Refusal from a scientific-target constructor.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ScientificTargetError {
    /// A name, unit, or law label is empty or whitespace.
    #[error("{field} must be non-blank")]
    EmptyName {
        /// Which label was blank.
        field: &'static str,
    },
    /// An intervention, policy, functional, or population declaration is unusable.
    #[error("{reason}")]
    InvalidAction {
        /// Stable explanation.
        reason: String,
    },
    /// A callback is a program binding, not a declared scientific action.
    #[error("a scientific world cannot carry a callback intervention")]
    CallbackAction,
    /// A temporal frame needs a horizon of at least one step.
    #[error("horizon_steps must be >= 1")]
    NonPositiveHorizon,
    /// Conditioning names the outcome.
    #[error("conditioning includes the outcome")]
    ConditioningIncludesOutcome,
    /// Decision time and the action disagree about whether a schedule exists.
    #[error("static decision time has no schedule, and temporal decision time must name one")]
    MismatchedActionTime,
    /// Two interventions in one world assign the same variable.
    #[error("variable {id} is intervened more than once in one world")]
    DuplicateInterventionVariable {
        /// Variable assigned twice.
        id: VariableId,
    },
    /// A collection was built with no worlds.
    #[error("a world collection names at least one world")]
    EmptyWorlds,
    /// Two worlds share an id.
    #[error("duplicate world id {id}")]
    DuplicateWorldId {
        /// Repeated id.
        id: ScientificWorldId,
    },
    /// Worlds disagree on population, unit, outcome, units, quantity, or conditioning.
    #[error(
        "worlds do not bind the same {field}; a collection needs one population, unit, outcome, units, quantity, and conditioning"
    )]
    IncompatibleBinding {
        /// `population`, `unit`, `outcome`, `units`, `quantity`, or `conditioning`.
        field: &'static str,
    },
    /// A contrast names a world the collection does not contain.
    #[error("contrast names unknown world {id}")]
    UnknownWorld {
        /// Missing id.
        id: ScientificWorldId,
    },
    /// A contrast repeats a world.
    #[error("a contrast uses each world once")]
    WorldsNotDistinct,
    /// The collection is not exactly the worlds the contrast formula names.
    #[error("a contrast's worlds are exactly the worlds in its formula")]
    ContrastWorldSet,
    /// A general linear functional has missing, zero, repeated or nonfinite coefficients.
    #[error("invalid linear world functional: {reason}")]
    InvalidLinearContrast {
        /// Exact invalid declaration.
        reason: &'static str,
    },
    /// The reference id is not flagged as the explicit reference intervention.
    #[error("reference world {id} is not marked as the explicit reference intervention")]
    ReferenceNotMarked {
        /// Id that was supposed to be the reference.
        id: ScientificWorldId,
    },
    /// A difference's reference is not the subtracted world.
    #[error("reference world {id} is not the subtracted baseline of this difference")]
    ReferenceIsNotBaseline {
        /// Id passed as the reference.
        id: ScientificWorldId,
    },
    /// A world other than the contrast's reference is flagged.
    #[error("world {id} is marked as a reference but is not this contrast's reference world")]
    UnexpectedReference {
        /// Extra flagged world.
        id: ScientificWorldId,
    },
    /// Contrasted worlds do not share decision time, horizon, history, and initial law.
    #[error("contrasted worlds do not share a decision time, horizon, history, and initial law")]
    IncompatibleTime,
    /// The dose × time declaration is not a rectangular grid of temporal worlds.
    #[error(
        "dose-time surface is a declared grid of worlds, not a contrast of intervals: {reason}"
    )]
    NotADoseTimeGrid {
        /// What about the grid failed.
        reason: &'static str,
    },
    /// A cross-world mediation component was declared without its premises.
    #[error(
        "mediation requires explicit cross-world premises; algebra of interventional expectations does not supply them"
    )]
    MediationNeedsCrossWorldPremises,
    /// Mediation names no mediator.
    #[error("mediation requires at least one mediator")]
    EmptyMediators,
    /// A mediator is the treatment or the outcome.
    #[error("a mediator overlaps the treatment or the outcome")]
    MediatorOverlap,
    /// Treatment and outcome are one variable.
    #[error("treatment and outcome are the same variable {id}")]
    TreatmentEqualsOutcome {
        /// Shared id.
        id: VariableId,
    },
    /// A temporal policy reaches outside `[-horizon, +horizon]`.
    #[error("temporal policy {policy} lies outside the target horizon of {horizon_steps} steps")]
    PolicyOutsideHorizon {
        /// Horizon of the frame the policy was declared under.
        horizon_steps: u32,
        /// Short description of the offending policy.
        policy: String,
    },
    /// The worlds of a contrast do not realise the contrast's formula.
    #[error("contrast worlds do not realise the formula: {reason}")]
    IncoherentContrast {
        /// What about the four-world or two-world structure failed.
        reason: &'static str,
    },
    /// A mediation target compares an arm with itself.
    #[error("mediation control and active arms are the same intervention")]
    MediationArmsNotDistinct,
    /// A mediation arm does not assign the treatment.
    #[error("intervention targets {got}, expected treatment {expected}")]
    InterventionVariableMismatch {
        /// Treatment the arm declared.
        expected: VariableId,
        /// Variable the intervention assigns.
        got: VariableId,
    },
}

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
            reason: "outcome functional is not a valid scientific quantity".into(),
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
    max_lag: Option<u32>,
}

impl HistoryWindow {
    /// The planner's default history cap. Not an explicit lag of zero.
    #[must_use]
    pub const fn planner_default() -> Self {
        Self { max_lag: None }
    }

    /// Retain at most `steps` lags. Zero is an explicit empty history.
    #[must_use]
    pub const fn max_lag(steps: u32) -> Self {
        Self { max_lag: Some(steps) }
    }

    /// Explicit cap, or [`None`] for the planner default.
    #[must_use]
    pub const fn max_lag_steps(self) -> Option<u32> {
        self.max_lag
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
    kind: TimeKind,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
enum TimeKind {
    Static,
    Temporal { horizon_steps: u32, history: HistoryWindow, initial_law: InitialLaw },
}

impl TimeFrame {
    /// One decision time.
    #[must_use]
    pub const fn static_decision() -> Self {
        Self { kind: TimeKind::Static }
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

    /// Horizon in steps, or [`None`] for a static frame.
    #[must_use]
    pub const fn horizon_steps(&self) -> Option<u32> {
        match &self.kind {
            TimeKind::Static => None,
            TimeKind::Temporal { horizon_steps, .. } => Some(*horizon_steps),
        }
    }

    /// History window, or [`None`] for a static frame.
    #[must_use]
    pub const fn history(&self) -> Option<HistoryWindow> {
        match &self.kind {
            TimeKind::Static => None,
            TimeKind::Temporal { history, .. } => Some(*history),
        }
    }

    /// Initial-state law, or [`None`] for a static frame.
    #[must_use]
    pub const fn initial_law(&self) -> Option<&InitialLaw> {
        match &self.kind {
            TimeKind::Static => None,
            TimeKind::Temporal { initial_law, .. } => Some(initial_law),
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
        population
            .validate()
            .map_err(|err| ScientificTargetError::InvalidAction { reason: err.to_string() })?;
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
    kind: ActionKind,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
enum ActionKind {
    Interventions(Arc<[Intervention]>),
    Schedule { intervention: Intervention, policy: TemporalPolicy },
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
        policy
            .validate()
            .map_err(|err| ScientificTargetError::InvalidAction { reason: err.to_string() })?;
        Ok(Self { kind: ActionKind::Schedule { intervention, policy } })
    }

    /// Interventions when this is a one-decision-time action.
    #[must_use]
    pub fn at_decision_interventions(&self) -> Option<&[Intervention]> {
        match &self.kind {
            ActionKind::Interventions(interventions) => Some(interventions),
            ActionKind::Schedule { .. } => None,
        }
    }

    /// Intervention and schedule when this action is temporal.
    #[must_use]
    pub fn schedule_parts(&self) -> Option<(&Intervention, &TemporalPolicy)> {
        match &self.kind {
            ActionKind::Schedule { intervention, policy } => Some((intervention, policy)),
            ActionKind::Interventions(_) => None,
        }
    }
}

fn canonical_interventions(
    interventions: impl IntoIterator<Item = Intervention>,
) -> Result<Arc<[Intervention]>, ScientificTargetError> {
    let mut interventions: Vec<Intervention> =
        interventions.into_iter().map(canonical_intervention).collect();
    if interventions.is_empty() {
        return Err(ScientificTargetError::InvalidAction {
            reason: "a world needs at least one intervention".into(),
        });
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

fn ensure_portable(intervention: &Intervention) -> Result<(), ScientificTargetError> {
    intervention
        .validate()
        .map_err(|err| ScientificTargetError::InvalidAction { reason: err.to_string() })?;
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
        Intervention::Set { value: level, .. } | Intervention::Shift { delta: level, .. } => level
            .validate_concrete_intervention_level()
            .map_err(|err| ScientificTargetError::InvalidAction { reason: err.to_string() }),
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
fn canonical_intervention(intervention: Intervention) -> Intervention {
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
fn ensure_policy_within_horizon(
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
        ActionKind::Interventions(_) => Ok(()),
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
    matches!(
        (&action.kind, &time.kind),
        (ActionKind::Interventions(_), TimeKind::Static)
            | (ActionKind::Schedule { .. }, TimeKind::Temporal { .. })
    )
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

/// One finite nonzero term of an explicitly declared linear world functional.
#[derive(Clone, Copy, Debug)]
pub struct ContrastTerm {
    /// World the coefficient multiplies.
    pub world: ScientificWorldId,
    /// Finite nonzero coefficient; ordinary differences and interactions use `+1`/`-1`.
    pub coefficient: f64,
}

impl PartialEq for ContrastTerm {
    fn eq(&self, other: &Self) -> bool {
        self.world == other.world && self.coefficient.to_bits() == other.coefficient.to_bits()
    }
}
impl Eq for ContrastTerm {}
impl Hash for ContrastTerm {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.world.hash(state);
        self.coefficient.to_bits().hash(state);
    }
}

/// Shape of an [`InterventionContrast`]. Not a mediation and not a dose × time grid.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum ContrastForm {
    /// Explicit finite linear combination, including higher-order factorial contrasts.
    /// Per-world time frames remain declared; no exponential world expansion is implied.
    Linear,
    /// `E[Y(plus)] − E[Y(minus)]`, with `minus` the reference world.
    Difference {
        /// World with coefficient `+1`.
        plus: ScientificWorldId,
        /// World with coefficient `-1`. This is the reference world.
        minus: ScientificWorldId,
    },
    /// `E[Y(a,b)] − E[Y(a,0)] − E[Y(0,b)] + E[Y(0,0)]`.
    ///
    /// `0` is the contrast's reference world, not the absence of treatment.
    Interaction {
        /// World `Y(a,b)`, coefficient `+1`.
        joint: ScientificWorldId,
        /// World `Y(a,0)`, coefficient `-1`.
        a_at_reference: ScientificWorldId,
        /// World `Y(0,b)`, coefficient `-1`.
        b_at_reference: ScientificWorldId,
    },
}

/// A contrast of intervention worlds, including the two-intervention reference interaction.
///
/// The collection is exactly the worlds in the formula. Marginal intervals are not
/// subtracted here; the coefficients are the target.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct InterventionContrast {
    worlds: WorldCollection,
    form: ContrastForm,
    reference: ScientificWorldId,
    linear_terms: Arc<[ContrastTerm]>,
}

impl InterventionContrast {
    /// Declare an arbitrary finite linear combination of intervention-world means.
    ///
    /// Every supplied world appears exactly once with a finite nonzero coefficient.
    /// The reference is explicit. World quantities/populations/units/conditioning agree;
    /// each world keeps its own declared time frame and history. This supports explicit
    /// higher-order factorial contrasts and time-window functionals without generating
    /// undeclared worlds, identifying them, or subtracting marginal intervals.
    ///
    /// # Errors
    /// Missing/repeated worlds, an invalid coefficient, an empty functional, or a missing
    /// reference. Statistical and scientific support remain the compiler's responsibility.
    pub fn linear(
        worlds: WorldCollection,
        terms: impl IntoIterator<Item = (ScientificWorldId, f64)>,
        reference: ScientificWorldId,
    ) -> Result<Self, ScientificTargetError> {
        let mut terms: Vec<_> = terms
            .into_iter()
            .map(|(world, coefficient)| ContrastTerm { world, coefficient })
            .collect();
        if terms.is_empty()
            || terms.iter().any(|term| !term.coefficient.is_finite() || term.coefficient == 0.0)
        {
            return Err(ScientificTargetError::InvalidLinearContrast {
                reason: "coefficients must be finite and nonzero",
            });
        }
        terms.sort_by_key(|term| term.world);
        require_ids(&worlds, &terms.iter().map(|term| term.world).collect::<Vec<_>>())?;
        require_one_reference(&worlds, reference)?;
        Ok(Self { worlds, form: ContrastForm::Linear, reference, linear_terms: terms.into() })
    }

    /// Difference `E[Y(plus)] − E[Y(minus)]`.
    ///
    /// `reference` must be `minus`, and that world must be flagged. The two worlds
    /// share decision time, horizon, history, and initial law.
    ///
    /// # Errors
    ///
    /// See [`ScientificTargetError`] for an unknown, repeated, extra, or unflagged
    /// world, a reference that is not the baseline, worlds whose time frames differ, or
    /// [`ScientificTargetError::IncoherentContrast`] when the two worlds do not assign
    /// the same coordinates.
    pub fn difference(
        worlds: WorldCollection,
        plus: ScientificWorldId,
        minus: ScientificWorldId,
        reference: ScientificWorldId,
    ) -> Result<Self, ScientificTargetError> {
        require_ids(&worlds, &[plus, minus])?;
        if reference != minus {
            return Err(ScientificTargetError::ReferenceIsNotBaseline { id: reference });
        }
        require_one_reference(&worlds, reference)?;
        require_same_time(&worlds)?;
        require_coherent_difference(&worlds, plus, minus)?;
        Ok(Self {
            worlds,
            form: ContrastForm::Difference { plus, minus },
            reference,
            linear_terms: Arc::from([]),
        })
    }

    /// Interaction `E[Y(a,b)] − E[Y(a,0)] − E[Y(0,b)] + E[Y(0,0)]`.
    ///
    /// `reference` is the world `Y(0,0)`. The four worlds share one time frame and
    /// assign the same coordinates (a variable's level, or the schedule). Two disjoint
    /// factors `A` and `B` each move away from the reference: `a_at_reference` changes
    /// only `A`, `b_at_reference` only `B`, and `joint` takes both. The reference level
    /// is whatever the reference world assigns, not necessarily zero.
    /// There is no constructor that builds this from four separately estimated intervals.
    ///
    /// # Errors
    ///
    /// See [`ScientificTargetError`] for an unknown, repeated, extra, or wrongly
    /// flagged world, worlds whose time frames differ, or
    /// [`ScientificTargetError::IncoherentContrast`] when the four worlds are not the
    /// 2x2 cells `Y(a,b)`, `Y(a,0)`, `Y(0,b)`, `Y(0,0)` of two disjoint factors.
    pub fn interaction(
        worlds: WorldCollection,
        joint: ScientificWorldId,
        a_at_reference: ScientificWorldId,
        b_at_reference: ScientificWorldId,
        reference: ScientificWorldId,
    ) -> Result<Self, ScientificTargetError> {
        require_ids(&worlds, &[joint, a_at_reference, b_at_reference, reference])?;
        require_one_reference(&worlds, reference)?;
        require_same_time(&worlds)?;
        require_coherent_interaction(&worlds, joint, a_at_reference, b_at_reference, reference)?;
        Ok(Self {
            worlds,
            form: ContrastForm::Interaction { joint, a_at_reference, b_at_reference },
            reference,
            linear_terms: Arc::from([]),
        })
    }

    /// Worlds in the formula.
    #[must_use]
    pub const fn worlds(&self) -> &WorldCollection {
        &self.worlds
    }

    /// Difference or interaction.
    #[must_use]
    pub const fn form(&self) -> ContrastForm {
        self.form
    }

    /// Explicit reference world.
    #[must_use]
    pub const fn reference(&self) -> ScientificWorldId {
        self.reference
    }

    /// Coefficients in formula order.
    ///
    /// A difference is `(plus, +1)`, `(minus, -1)`. An interaction is
    /// `(joint, +1)`, `(a_at_reference, -1)`, `(b_at_reference, -1)`, `(reference, +1)`.
    #[must_use]
    pub fn terms(&self) -> Vec<ContrastTerm> {
        match self.form {
            ContrastForm::Linear => self.linear_terms.to_vec(),
            ContrastForm::Difference { plus, minus } => {
                vec![term(plus, 1), term(minus, -1)]
            }
            ContrastForm::Interaction { joint, a_at_reference, b_at_reference } => {
                vec![
                    term(joint, 1),
                    term(a_at_reference, -1),
                    term(b_at_reference, -1),
                    term(self.reference, 1),
                ]
            }
        }
    }
}

fn term(world: ScientificWorldId, coefficient: i8) -> ContrastTerm {
    ContrastTerm { world, coefficient: f64::from(coefficient) }
}

fn require_ids(
    worlds: &WorldCollection,
    ids: &[ScientificWorldId],
) -> Result<(), ScientificTargetError> {
    let mut sorted = ids.to_vec();
    sorted.sort_unstable();
    if sorted.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(ScientificTargetError::WorldsNotDistinct);
    }
    for id in ids {
        if worlds.get(*id).is_none() {
            return Err(ScientificTargetError::UnknownWorld { id: *id });
        }
    }
    let actual: Vec<ScientificWorldId> =
        worlds.worlds().iter().map(InterventionWorld::id).collect();
    if actual != sorted {
        return Err(ScientificTargetError::ContrastWorldSet);
    }
    Ok(())
}

fn require_one_reference(
    worlds: &WorldCollection,
    reference: ScientificWorldId,
) -> Result<(), ScientificTargetError> {
    let Some(chosen) = worlds.get(reference) else {
        return Err(ScientificTargetError::UnknownWorld { id: reference });
    };
    if !chosen.is_reference() {
        return Err(ScientificTargetError::ReferenceNotMarked { id: reference });
    }
    if let Some(other) =
        worlds.worlds().iter().find(|world| world.id() != reference && world.is_reference())
    {
        return Err(ScientificTargetError::UnexpectedReference { id: other.id() });
    }
    Ok(())
}

fn require_same_time(worlds: &WorldCollection) -> Result<(), ScientificTargetError> {
    let Some((first, rest)) = worlds.worlds().split_first() else {
        return Err(ScientificTargetError::EmptyWorlds);
    };
    if rest.iter().any(|world| world.time() != first.time()) {
        Err(ScientificTargetError::IncompatibleTime)
    } else {
        Ok(())
    }
}

/// What a world fixes: a variable's intervention, or the schedule it runs under.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum Coordinate {
    Variable(VariableId),
    Schedule,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum Assigned<'a> {
    Intervention(Intervention),
    Schedule(&'a TemporalPolicy),
}

fn coordinates(world: &InterventionWorld) -> BTreeMap<Coordinate, Assigned<'_>> {
    let mut map = BTreeMap::new();
    match &world.action.kind {
        ActionKind::Interventions(interventions) => {
            for intervention in interventions.iter() {
                for variable in intervention.target_variables() {
                    map.insert(
                        Coordinate::Variable(variable),
                        Assigned::Intervention(
                            projected_intervention(intervention, variable)
                                .expect("intervention declares this target"),
                        ),
                    );
                }
            }
        }
        ActionKind::Schedule { intervention, policy } => {
            for variable in intervention.target_variables() {
                map.insert(
                    Coordinate::Variable(variable),
                    Assigned::Intervention(
                        projected_intervention(intervention, variable)
                            .expect("intervention declares this target"),
                    ),
                );
            }
            map.insert(Coordinate::Schedule, Assigned::Schedule(policy));
        }
    }
    map
}

/// A sequence may set several channels at different times. Compare only the steps
/// that set this coordinate, otherwise changing A incorrectly appears to change B too.
fn projected_intervention(
    intervention: &Intervention,
    variable: VariableId,
) -> Option<Intervention> {
    match intervention {
        Intervention::Sequence(sequence) => {
            let steps = sequence
                .steps
                .iter()
                .filter_map(|step| {
                    projected_intervention(&step.intervention, variable).map(|intervention| {
                        crate::intervention::SequencedIntervention {
                            intervention,
                            temporal: step.temporal.clone(),
                        }
                    })
                })
                .collect::<Vec<_>>();
            (!steps.is_empty()).then(|| {
                Intervention::Sequence(crate::intervention::InterventionSequence::new(steps))
            })
        }
        _ => intervention.target_variables().contains(&variable).then(|| intervention.clone()),
    }
}

fn incoherent(reason: &'static str) -> ScientificTargetError {
    ScientificTargetError::IncoherentContrast { reason }
}

fn require_coherent_difference(
    worlds: &WorldCollection,
    plus: ScientificWorldId,
    minus: ScientificWorldId,
) -> Result<(), ScientificTargetError> {
    let (Some(plus), Some(minus)) = (worlds.get(plus), worlds.get(minus)) else {
        return Err(ScientificTargetError::EmptyWorlds);
    };
    let plus = coordinates(plus);
    let minus = coordinates(minus);
    if !plus.keys().eq(minus.keys()) {
        return Err(incoherent("the two worlds assign different variables or schedules"));
    }
    if plus == minus {
        return Err(incoherent("the two worlds are the same intervention"));
    }
    Ok(())
}

fn require_coherent_interaction(
    worlds: &WorldCollection,
    joint: ScientificWorldId,
    a_at_reference: ScientificWorldId,
    b_at_reference: ScientificWorldId,
    reference: ScientificWorldId,
) -> Result<(), ScientificTargetError> {
    let (Some(joint), Some(a_world), Some(b_world), Some(reference)) = (
        worlds.get(joint),
        worlds.get(a_at_reference),
        worlds.get(b_at_reference),
        worlds.get(reference),
    ) else {
        return Err(ScientificTargetError::EmptyWorlds);
    };
    let joint = coordinates(joint);
    let a_world = coordinates(a_world);
    let b_world = coordinates(b_world);
    let reference = coordinates(reference);
    for other in [&joint, &a_world, &b_world] {
        if !other.keys().eq(reference.keys()) {
            return Err(incoherent("the four worlds assign different variables or schedules"));
        }
    }
    let mut moves_a = false;
    let mut moves_b = false;
    for (coordinate, at_reference) in &reference {
        let a = &a_world[coordinate];
        let b = &b_world[coordinate];
        let both = &joint[coordinate];
        let a_moves = a != at_reference;
        let b_moves = b != at_reference;
        if a_moves && b_moves {
            return Err(incoherent("the two factors move the same variable or schedule"));
        }
        let expected = if a_moves {
            a
        } else if b_moves {
            b
        } else {
            at_reference
        };
        if both != expected {
            return Err(incoherent("the joint world is not the reference moved by both factors"));
        }
        moves_a |= a_moves;
        moves_b |= b_moves;
    }
    if !moves_a || !moves_b {
        return Err(incoherent("each factor must move away from the reference"));
    }
    Ok(())
}

/// Dose × time surface: one world for every dose on every horizon.
///
/// Cell order is dose-major, then horizon. The surface is not a difference of
/// corner intervals; worlds on the grid may have different horizons, which a
/// contrast refuses.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct DoseTimeSurface {
    worlds: WorldCollection,
    dose: VariableId,
    level_bits: Arc<[u64]>,
    horizons: Arc<[u32]>,
    cells: Arc<[ScientificWorldId]>,
}

impl DoseTimeSurface {
    /// Grid. `cells` has one id per dose and horizon, in dose-major order.
    ///
    /// Each cell is a temporal schedule that hard-sets `dose` to that level.
    /// Every cell shares the policy, history, and initial law. Dose levels are
    /// finite and strictly increasing, with `-0.0` stored as `+0.0`. Horizons
    /// are strictly increasing and at least 1.
    ///
    /// # Errors
    ///
    /// [`ScientificTargetError::NotADoseTimeGrid`] when the declaration is not
    /// that rectangle, or [`ScientificTargetError::UnknownWorld`] when a cell id
    /// is missing.
    pub fn new(
        worlds: WorldCollection,
        dose: VariableId,
        levels: impl IntoIterator<Item = f64>,
        horizons: impl IntoIterator<Item = u32>,
        cells: impl IntoIterator<Item = ScientificWorldId>,
    ) -> Result<Self, ScientificTargetError> {
        let level_bits = canonical_levels(levels)?;
        let horizons = canonical_horizons(horizons)?;
        let cells: Arc<[ScientificWorldId]> = cells.into_iter().collect();
        check_grid(&worlds, dose, &level_bits, &horizons, &cells)?;
        Ok(Self { worlds, dose, level_bits, horizons, cells })
    }

    /// Worlds of the grid.
    #[must_use]
    pub const fn worlds(&self) -> &WorldCollection {
        &self.worlds
    }

    /// Dose variable.
    #[must_use]
    pub const fn dose(&self) -> VariableId {
        self.dose
    }

    /// Dose levels as IEEE-754 bits. `-0.0` is stored as `+0.0`.
    #[must_use]
    pub fn level_bits(&self) -> &[u64] {
        &self.level_bits
    }

    /// Horizons in axis order.
    #[must_use]
    pub fn horizons(&self) -> &[u32] {
        &self.horizons
    }

    /// Cell world ids, dose-major then horizon.
    #[must_use]
    pub fn cells(&self) -> &[ScientificWorldId] {
        &self.cells
    }
}

fn canonical_levels(
    levels: impl IntoIterator<Item = f64>,
) -> Result<Arc<[u64]>, ScientificTargetError> {
    let mut bits = Vec::new();
    let mut previous: Option<f64> = None;
    for level in levels {
        if !level.is_finite() {
            return Err(grid("dose levels must be finite and strictly increasing"));
        }
        let level = if level.to_bits() == (-0.0_f64).to_bits() { 0.0 } else { level };
        if previous.is_some_and(|prev| level.total_cmp(&prev) != std::cmp::Ordering::Greater) {
            return Err(grid("dose levels must be finite and strictly increasing"));
        }
        previous = Some(level);
        bits.push(level.to_bits());
    }
    if bits.is_empty() {
        return Err(grid("a dose-time grid names at least one dose and one horizon"));
    }
    Ok(Arc::from(bits))
}

fn canonical_horizons(
    horizons: impl IntoIterator<Item = u32>,
) -> Result<Arc<[u32]>, ScientificTargetError> {
    let horizons: Vec<u32> = horizons.into_iter().collect();
    if horizons.is_empty() {
        return Err(grid("a dose-time grid names at least one dose and one horizon"));
    }
    let mut previous = 0_u32;
    for horizon in &horizons {
        if *horizon == 0 || *horizon <= previous {
            return Err(grid("horizons must be strictly increasing and at least 1"));
        }
        previous = *horizon;
    }
    Ok(Arc::from(horizons))
}

fn check_grid(
    worlds: &WorldCollection,
    dose: VariableId,
    levels: &[u64],
    horizons: &[u32],
    cells: &[ScientificWorldId],
) -> Result<(), ScientificTargetError> {
    let Some(expected) = levels.len().checked_mul(horizons.len()) else {
        return Err(grid("dose-time grid is too large"));
    };
    if cells.len() != expected || worlds.len() != cells.len() {
        return Err(grid("cells must be one world per dose and horizon, and no others"));
    }
    let mut seen = cells.to_vec();
    seen.sort_unstable();
    if seen.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(grid("a world is used for more than one cell"));
    }
    let mut anchor: Option<(TemporalPolicy, HistoryWindow, InitialLaw)> = None;
    for (index, id) in cells.iter().enumerate() {
        let Some(world) = worlds.get(*id) else {
            return Err(ScientificTargetError::UnknownWorld { id: *id });
        };
        if world.binding().outcome().outcome() == dose {
            return Err(grid("dose and outcome are the same variable"));
        }
        let dose_index = index / horizons.len();
        let horizon_index = index % horizons.len();
        let (policy, history, initial_law, horizon_steps, assigned) = cell_parts(world, dose)?;
        if horizon_steps != horizons[horizon_index] || assigned != levels[dose_index] {
            return Err(grid("a cell does not match its dose and horizon"));
        }
        if let Some((policy0, history0, law0)) = &anchor {
            if policy != policy0 || history != *history0 || initial_law != law0 {
                return Err(grid("cells must share a policy, history, and initial law"));
            }
        } else {
            anchor = Some((policy.clone(), history, initial_law.clone()));
        }
    }
    Ok(())
}

fn cell_parts(
    world: &InterventionWorld,
    dose: VariableId,
) -> Result<(&TemporalPolicy, HistoryWindow, &InitialLaw, u32, u64), ScientificTargetError> {
    let TimeKind::Temporal { horizon_steps, history, initial_law } = &world.time().kind else {
        return Err(grid("each cell is a temporal world"));
    };
    let ActionKind::Schedule { intervention, policy } = &world.action().kind else {
        return Err(grid("each cell is a scheduled dose assignment"));
    };
    let Intervention::Set { variable, value: Value::Float64(level) } = intervention else {
        return Err(grid("each cell hard-sets the dose"));
    };
    if *variable != dose {
        return Err(grid("each cell hard-sets the dose"));
    }
    let bits =
        if level.to_bits() == (-0.0_f64).to_bits() { 0.0_f64.to_bits() } else { level.to_bits() };
    Ok((policy, *history, initial_law, *horizon_steps, bits))
}

fn grid(reason: &'static str) -> ScientificTargetError {
    ScientificTargetError::NotADoseTimeGrid { reason }
}

/// Which mediation functional is the target.
///
/// These names match [`crate::query::MediationContrast`]. [`Self::Direct`] is the
/// natural direct path-sum, not a controlled direct effect licensed by algebra.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum MediationComponent {
    /// Total effect. Cross-world premises may be empty; mediators are still part of the target.
    Total,
    /// Natural direct path-sum that avoids the declared mediators.
    Direct,
    /// Effect along the declared mediators.
    Mediated,
    /// Natural direct effect.
    NaturalDirect,
    /// Natural indirect effect.
    NaturalIndirect,
}

impl MediationComponent {
    /// Whether algebra is not enough and cross-world premises are required.
    #[must_use]
    pub const fn requires_cross_world(self) -> bool {
        !matches!(self, Self::Total)
    }
}

/// Premise a mediation target must carry. A contrast does not have these.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[non_exhaustive]
pub enum CrossWorldPremise {
    /// Setting a variable to the value it would have taken does not further change descendants.
    Composition,
    /// The mediator's counterfactual under one treatment is independent of the other treatment world.
    CrossWorldIndependence,
    /// The functional is a nested path-specific effect.
    PathSpecificNested,
}

impl CrossWorldPremise {
    const fn tag(self) -> u64 {
        match self {
            Self::Composition => 1,
            Self::CrossWorldIndependence => 2,
            Self::PathSpecificNested => 3,
        }
    }
}

/// Treatment, mediators, and the two treatment interventions of a mediation target.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct MediationArms {
    treatment: VariableId,
    mediators: Arc<[VariableId]>,
    control: Intervention,
    active: Intervention,
    component: MediationComponent,
}

impl MediationArms {
    /// Arms. Mediators are sorted and de-duplicated.
    ///
    /// # Errors
    ///
    /// [`ScientificTargetError::EmptyMediators`], [`ScientificTargetError::MediatorOverlap`],
    /// [`ScientificTargetError::InterventionVariableMismatch`],
    /// [`ScientificTargetError::MediationArmsNotDistinct`],
    /// [`ScientificTargetError::InvalidAction`], or [`ScientificTargetError::CallbackAction`].
    pub fn new(
        treatment: VariableId,
        mediators: impl IntoIterator<Item = VariableId>,
        control: Intervention,
        active: Intervention,
        component: MediationComponent,
    ) -> Result<Self, ScientificTargetError> {
        let mut mediators: Vec<VariableId> = mediators.into_iter().collect();
        mediators.sort_unstable();
        mediators.dedup();
        if mediators.is_empty() {
            return Err(ScientificTargetError::EmptyMediators);
        }
        if mediators.contains(&treatment) {
            return Err(ScientificTargetError::MediatorOverlap);
        }
        let control = canonical_intervention(control);
        let active = canonical_intervention(active);
        expect_treatment(&control, treatment)?;
        expect_treatment(&active, treatment)?;
        if control == active {
            return Err(ScientificTargetError::MediationArmsNotDistinct);
        }
        Ok(Self { treatment, mediators: Arc::from(mediators), control, active, component })
    }

    /// Treatment variable.
    #[must_use]
    pub const fn treatment(&self) -> VariableId {
        self.treatment
    }

    /// Mediators, sorted.
    #[must_use]
    pub fn mediators(&self) -> &[VariableId] {
        &self.mediators
    }

    /// Reference treatment intervention.
    #[must_use]
    pub const fn control(&self) -> &Intervention {
        &self.control
    }

    /// Active treatment intervention.
    #[must_use]
    pub const fn active(&self) -> &Intervention {
        &self.active
    }

    /// Mediation component.
    #[must_use]
    pub const fn component(&self) -> MediationComponent {
        self.component
    }
}

fn expect_treatment(
    intervention: &Intervention,
    treatment: VariableId,
) -> Result<(), ScientificTargetError> {
    ensure_portable(intervention)?;
    match intervention.primary_variable() {
        Some(variable) if variable == treatment => Ok(()),
        Some(variable) => Err(ScientificTargetError::InterventionVariableMismatch {
            expected: treatment,
            got: variable,
        }),
        None => Err(ScientificTargetError::InvalidAction {
            reason: "mediation treatment intervention must target one variable".into(),
        }),
    }
}

/// Mediation target. Not constructible from [`InterventionContrast`].
///
/// [`MediationComponent::Direct`], [`MediationComponent::Mediated`],
/// [`MediationComponent::NaturalDirect`], and [`MediationComponent::NaturalIndirect`]
/// require [`CrossWorldPremise::CrossWorldIndependence`] and
/// [`CrossWorldPremise::PathSpecificNested`]. The two natural components are defined
/// through the nested counterfactual `Y(a, M(a'))`, so they also require
/// [`CrossWorldPremise::Composition`]. [`MediationComponent::Total`] may
/// record an empty premise set; the mediator set still keeps it distinct from a
/// treatment contrast.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct MediationTarget {
    binding: WorldBinding,
    time: TimeFrame,
    schedule: Option<TemporalPolicy>,
    arms: MediationArms,
    premises: BTreeSet<CrossWorldPremise>,
}

impl MediationTarget {
    /// Mediation under `time`. `schedule` is required exactly when `time` is temporal.
    ///
    /// # Errors
    ///
    /// [`ScientificTargetError::MediationNeedsCrossWorldPremises`] when a cross-world
    /// component lacks its premises, [`ScientificTargetError::MismatchedActionTime`]
    /// when the schedule and the frame disagree, [`ScientificTargetError::TreatmentEqualsOutcome`]
    /// or [`ScientificTargetError::MediatorOverlap`] when the arms collide with the outcome,
    /// [`ScientificTargetError::InvalidAction`] when a temporal policy is invalid, and
    /// [`ScientificTargetError::PolicyOutsideHorizon`] when it reaches outside the horizon.
    pub fn new(
        binding: WorldBinding,
        time: TimeFrame,
        schedule: Option<TemporalPolicy>,
        arms: MediationArms,
        premises: impl IntoIterator<Item = CrossWorldPremise>,
    ) -> Result<Self, ScientificTargetError> {
        if let Some(policy) = &schedule {
            policy
                .validate()
                .map_err(|err| ScientificTargetError::InvalidAction { reason: err.to_string() })?;
            ensure_policy_within_horizon(policy, &time)?;
        }
        let schedule_matches = matches!(
            (&time.kind, schedule.is_some()),
            (TimeKind::Static, false) | (TimeKind::Temporal { .. }, true)
        );
        if !schedule_matches {
            return Err(ScientificTargetError::MismatchedActionTime);
        }
        let outcome = binding.outcome().outcome();
        if arms.treatment == outcome {
            return Err(ScientificTargetError::TreatmentEqualsOutcome { id: outcome });
        }
        if arms.mediators.contains(&outcome) {
            return Err(ScientificTargetError::MediatorOverlap);
        }
        let premises: BTreeSet<CrossWorldPremise> = premises.into_iter().collect();
        let natural = matches!(
            arms.component,
            MediationComponent::NaturalDirect | MediationComponent::NaturalIndirect
        );
        if arms.component.requires_cross_world()
            && !(premises.contains(&CrossWorldPremise::CrossWorldIndependence)
                && premises.contains(&CrossWorldPremise::PathSpecificNested)
                && (!natural || premises.contains(&CrossWorldPremise::Composition)))
        {
            return Err(ScientificTargetError::MediationNeedsCrossWorldPremises);
        }
        Ok(Self { binding, time, schedule, arms, premises })
    }

    /// Population, unit, quantity, and conditioning.
    #[must_use]
    pub const fn binding(&self) -> &WorldBinding {
        &self.binding
    }

    /// Decision time.
    #[must_use]
    pub const fn time(&self) -> &TimeFrame {
        &self.time
    }

    /// Treatment schedule when the target is temporal.
    #[must_use]
    pub const fn schedule(&self) -> Option<&TemporalPolicy> {
        self.schedule.as_ref()
    }

    /// Treatment arms.
    #[must_use]
    pub const fn arms(&self) -> &MediationArms {
        &self.arms
    }

    /// Cross-world premises. Empty only for a total effect.
    #[must_use]
    pub const fn premises(&self) -> &BTreeSet<CrossWorldPremise> {
        &self.premises
    }
}

/// Which scientific object a [`ScientificTarget`] is.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum TargetForm {
    /// One world, `E[Y(a)]` under that world's action.
    World(InterventionWorld),
    /// Aligned worlds before a contrast or surface is chosen.
    Worlds(WorldCollection),
    /// Difference or reference interaction.
    Contrast(InterventionContrast),
    /// Declared dose × time grid.
    DoseTime(DoseTimeSurface),
    /// Mediation, including its cross-world premises.
    Mediation(MediationTarget),
}

/// Scientific target. Equality and [`Self::identity`] ignore [`InferenceSelection`].
#[derive(Clone, Debug)]
pub struct ScientificTarget {
    form: TargetForm,
    assumptions: ModelAssumptions,
    inference: InferenceSelection,
}

impl PartialEq for ScientificTarget {
    fn eq(&self, other: &Self) -> bool {
        self.form == other.form && self.assumptions == other.assumptions
    }
}

impl Eq for ScientificTarget {}

impl Hash for ScientificTarget {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.form.hash(state);
        self.assumptions.hash(state);
    }
}

impl ScientificTarget {
    /// Package a validated form. This does not identify or estimate it.
    #[must_use]
    pub const fn new(
        form: TargetForm,
        assumptions: ModelAssumptions,
        inference: InferenceSelection,
    ) -> Self {
        Self { form, assumptions, inference }
    }

    /// Scientific object.
    #[must_use]
    pub const fn form(&self) -> &TargetForm {
        &self.form
    }

    /// Graph and model assumptions.
    #[must_use]
    pub const fn assumptions(&self) -> &ModelAssumptions {
        &self.assumptions
    }

    /// Realization label. Excluded from [`Self::eq`] and [`Self::identity`].
    #[must_use]
    pub const fn inference(&self) -> InferenceSelection {
        self.inference
    }

    /// Same science under a different realization label.
    #[must_use]
    pub const fn with_inference(mut self, inference: InferenceSelection) -> Self {
        self.inference = inference;
        self
    }

    /// BLAKE3 digest of the scientific fields via [`CanonicalWriter`].
    ///
    /// Population, worlds, horizon, history, initial law, quantity, conditioning,
    /// graph, and the set of assumptions and scopes enter the digest. A world, a
    /// world collection, a dose × time surface, and a contrast are hashed by
    /// scientific content. A caller-chosen world id or a reference flag does not.
    /// A world collection is that set of contents. A dose × time surface hashes
    /// each cell's content in grid order. Inference selection, assumption source
    /// and status, and record order do not.
    #[must_use]
    pub fn identity(&self) -> SemanticDigest {
        let mut writer = CanonicalWriter::new(SCIENTIFIC_TARGET_IDENTITY_TAG);
        writer.word(u64::from(SCIENTIFIC_TARGET_FORMAT));
        write_form(&mut writer, &self.form);
        write_assumptions(&mut writer, &self.assumptions);
        SemanticDigest::from_bytes(writer.finish())
    }
}

fn count(len: usize) -> u64 {
    u64::try_from(len).unwrap_or(u64::MAX)
}

fn i32_bits(value: i32) -> u64 {
    u64::from(u32::from_le_bytes(value.to_le_bytes()))
}

fn i64_bits(value: i64) -> u64 {
    u64::from_le_bytes(value.to_le_bytes())
}

fn flag(value: bool) -> u64 {
    u64::from(u8::from(value))
}

fn write_var(writer: &mut CanonicalWriter, id: VariableId) {
    writer.word(u64::from(id.raw()));
}

fn write_vars(writer: &mut CanonicalWriter, variables: &[VariableId]) {
    writer.word(count(variables.len()));
    for variable in variables {
        write_var(writer, *variable);
    }
}

fn write_form(writer: &mut CanonicalWriter, form: &TargetForm) {
    match form {
        TargetForm::World(world) => {
            writer.word(1);
            write_world(writer, world);
        }
        TargetForm::Worlds(worlds) => {
            writer.word(2);
            write_worlds(writer, worlds);
        }
        TargetForm::Contrast(contrast) => {
            writer.word(3);
            write_contrast(writer, contrast);
        }
        TargetForm::DoseTime(surface) => {
            writer.word(4);
            write_surface(writer, surface);
        }
        TargetForm::Mediation(mediation) => {
            writer.word(5);
            write_mediation(writer, mediation);
        }
    }
}

fn write_world(writer: &mut CanonicalWriter, world: &InterventionWorld) {
    write_world_content(writer, world);
}

/// What a world says scientifically, without the caller's label or reference flag.
fn write_world_content(writer: &mut CanonicalWriter, world: &InterventionWorld) {
    write_action(writer, &world.action);
    write_binding(writer, &world.binding);
    write_time(writer, &world.time);
}

fn write_worlds(writer: &mut CanonicalWriter, worlds: &WorldCollection) {
    let mut digests: Vec<[u8; 32]> = worlds.worlds.iter().map(world_content_digest).collect();
    digests.sort_unstable();
    writer.word(count(digests.len()));
    for digest in &digests {
        writer.bytes(digest);
    }
}

fn world_content_digest(world: &InterventionWorld) -> [u8; 32] {
    let mut writer = CanonicalWriter::new("antecedent.scientific_target.world");
    write_world_content(&mut writer, world);
    writer.finish()
}

fn write_binding(writer: &mut CanonicalWriter, binding: &WorldBinding) {
    write_population(writer, &binding.population);
    writer.text(binding.unit.name());
    write_quantity(writer, &binding.outcome);
    write_vars(writer, binding.conditioning.variables());
}

fn write_quantity(writer: &mut CanonicalWriter, quantity: &OutcomeQuantity) {
    write_var(writer, quantity.outcome);
    writer.text(quantity.units.name());
    write_functional(writer, &quantity.functional);
}

fn write_functional(writer: &mut CanonicalWriter, functional: &OutcomeFunctional) {
    match functional {
        OutcomeFunctional::Mean => {
            writer.word(1);
        }
        OutcomeFunctional::Exceedance(threshold) => {
            writer.word(2).real(threshold.to_f64());
        }
        OutcomeFunctional::ExceedanceGrid(grid) => {
            writer.word(3).word(count(grid.len()));
            for threshold in grid.as_ref() {
                writer.real(threshold.to_f64());
            }
        }
        OutcomeFunctional::Quantile(tau) => {
            writer.word(4).real(tau.to_f64());
        }
    }
}

fn write_population(writer: &mut CanonicalWriter, population: &TargetPopulation) {
    match population {
        TargetPopulation::AllObserved => {
            writer.word(1);
        }
        TargetPopulation::Treated => {
            writer.word(2);
        }
        TargetPopulation::Untreated => {
            writer.word(3);
        }
        TargetPopulation::Environment(id) => {
            writer.word(4).word(u64::from(id.raw()));
        }
        TargetPopulation::Predicate(predicate) => {
            writer.word(5);
            write_predicate(writer, predicate);
        }
        TargetPopulation::CustomDistribution(id) => {
            writer.word(6).word(u64::from(id.raw()));
        }
        TargetPopulation::RowWeights { weights, depends_on } => {
            writer.word(7).bytes(weights);
            write_vars(writer, depends_on);
        }
        TargetPopulation::LocalAtCutoff { running, cutoff } => {
            writer.word(8).word(u64::from(running.raw())).real(cutoff.to_f64());
        }
    }
}

fn write_predicate(writer: &mut CanonicalWriter, predicate: &PredicateExpr) {
    match predicate {
        PredicateExpr::Named(name) => {
            writer.word(1).text(name);
        }
        PredicateExpr::Rows(rows) => {
            writer.word(2).word(count(rows.len()));
            for row in rows.as_ref() {
                writer.word(count(*row));
            }
        }
    }
}

fn write_time(writer: &mut CanonicalWriter, time: &TimeFrame) {
    match &time.kind {
        TimeKind::Static => {
            writer.word(1);
        }
        TimeKind::Temporal { horizon_steps, history, initial_law } => {
            writer.word(2).word(u64::from(*horizon_steps));
            write_history(writer, *history);
            write_initial_law(writer, initial_law);
        }
    }
}

fn write_history(writer: &mut CanonicalWriter, history: HistoryWindow) {
    match history.max_lag_steps() {
        None => {
            writer.word(0);
        }
        Some(steps) => {
            writer.word(1).word(u64::from(steps));
        }
    }
}

fn write_initial_law(writer: &mut CanonicalWriter, law: &InitialLaw) {
    match &law.kind {
        InitialLawKind::ContemporaneousObserved => {
            writer.word(1);
        }
        InitialLawKind::Named(name) => {
            writer.word(2).text(name);
        }
    }
}

fn write_action(writer: &mut CanonicalWriter, action: &WorldAction) {
    match &action.kind {
        ActionKind::Interventions(interventions) => {
            writer.word(1).word(count(interventions.len()));
            for intervention in interventions.as_ref() {
                write_intervention(writer, intervention);
            }
        }
        ActionKind::Schedule { intervention, policy } => {
            writer.word(2);
            write_intervention(writer, intervention);
            write_policy(writer, policy);
        }
    }
}

fn write_policy(writer: &mut CanonicalWriter, policy: &TemporalPolicy) {
    match policy {
        TemporalPolicy::Pulse { at } => {
            writer.word(1).word(i32_bits(*at));
        }
        TemporalPolicy::Sustained { from, until } => {
            writer.word(2).word(i32_bits(*from)).word(i32_bits(*until));
        }
        TemporalPolicy::Dynamic { rule, active_at } => {
            writer.word(3).word(u64::from(rule.raw())).word(count(active_at.len()));
            for step in active_at.as_ref() {
                writer.word(i32_bits(*step));
            }
        }
    }
}

fn write_intervention(writer: &mut CanonicalWriter, intervention: &Intervention) {
    match intervention {
        Intervention::Set { variable, value } => {
            writer.word(1);
            write_var(writer, *variable);
            write_value(writer, value);
        }
        Intervention::Shift { variable, delta } => {
            writer.word(2);
            write_var(writer, *variable);
            write_value(writer, delta);
        }
        Intervention::Stochastic { variable, policy } => {
            writer.word(3);
            write_var(writer, *variable);
            write_stochastic(writer, policy);
        }
        Intervention::Soft { variable, mechanism } => {
            writer.word(4);
            write_var(writer, *variable);
            writer.text(&mechanism.family_id);
            writer.reals(&mechanism.parameters);
        }
        Intervention::Transform { variable, expr, reads } => {
            writer.word(5);
            write_var(writer, *variable);
            write_expr(writer, expr);
            write_reads(writer, reads);
        }
        Intervention::Assign { variable, expr, reads } => {
            writer.word(6);
            write_var(writer, *variable);
            write_expr(writer, expr);
            write_reads(writer, reads);
        }
        Intervention::Sequence(sequence) => {
            writer.word(7).word(count(sequence.steps.len()));
            for step in sequence.steps.as_ref() {
                write_intervention(writer, &step.intervention);
                write_policy(writer, &step.temporal);
            }
        }
    }
}

fn write_value(writer: &mut CanonicalWriter, value: &Value) {
    match value {
        Value::Float64(number) => {
            writer.word(1).real(*number);
        }
        Value::Int64(number) => {
            writer.word(2).word(i64_bits(*number));
        }
        Value::Bool(bit) => {
            writer.word(3).word(flag(*bit));
        }
        Value::Category(code) => {
            writer.word(4).word(u64::from(*code));
        }
        Value::Label(label) => {
            writer.word(5).text(label);
        }
    }
}

fn write_stochastic(writer: &mut CanonicalWriter, policy: &StochasticPolicy) {
    match policy {
        StochasticPolicy::Bernoulli { p } => {
            writer.word(1).real(*p);
        }
        StochasticPolicy::Gaussian { mean, variance } => {
            writer.word(2).real(*mean).real(*variance);
        }
        StochasticPolicy::Categorical { probs } => {
            writer.word(3);
            writer.reals(probs);
        }
    }
}

fn write_reads(writer: &mut CanonicalWriter, reads: &PolicyReads) {
    write_vars(writer, &reads.current);
    writer.word(count(reads.history.len()));
    for &(variable, lag) in reads.history.as_ref() {
        write_var(writer, variable);
        writer.word(u64::from(lag));
    }
}

fn write_expr(writer: &mut CanonicalWriter, expr: &ValueExpr) {
    match expr {
        ValueExpr::Const(number) => {
            writer.word(1).real(number.get());
        }
        ValueExpr::Generated => {
            writer.word(2);
        }
        ValueExpr::Var(variable) => {
            writer.word(3);
            write_var(writer, *variable);
        }
        ValueExpr::Lag { variable, lag } => {
            writer.word(4);
            write_var(writer, *variable);
            writer.word(u64::from(*lag));
        }
        ValueExpr::Neg(inner) => {
            writer.word(5);
            write_expr(writer, inner);
        }
        ValueExpr::Arith { op, lhs, rhs } => {
            writer.word(6).word(arith_tag(*op));
            write_expr(writer, lhs);
            write_expr(writer, rhs);
        }
        ValueExpr::Clip { value, lower, upper } => {
            writer.word(7);
            write_expr(writer, value);
            write_expr(writer, lower);
            write_expr(writer, upper);
        }
        ValueExpr::Compare { op, lhs, rhs } => {
            writer.word(8).word(compare_tag(*op));
            write_expr(writer, lhs);
            write_expr(writer, rhs);
        }
        ValueExpr::Not(inner) => {
            writer.word(9);
            write_expr(writer, inner);
        }
        ValueExpr::Logic { op, lhs, rhs } => {
            writer.word(10).word(logic_tag(*op));
            write_expr(writer, lhs);
            write_expr(writer, rhs);
        }
        ValueExpr::If { condition, then_branch, else_branch } => {
            writer.word(11);
            write_expr(writer, condition);
            write_expr(writer, then_branch);
            write_expr(writer, else_branch);
        }
        ValueExpr::Indicator(inner) => {
            writer.word(12);
            write_expr(writer, inner);
        }
        ValueExpr::Callback(call) => {
            writer.word(13);
            write_callback(writer, call);
        }
    }
}

fn arith_tag(op: ArithOp) -> u64 {
    match op {
        ArithOp::Add => 1,
        ArithOp::Sub => 2,
        ArithOp::Mul => 3,
        ArithOp::Div => 4,
        ArithOp::Min => 5,
        ArithOp::Max => 6,
    }
}

fn compare_tag(op: CompareOp) -> u64 {
    match op {
        CompareOp::Lt => 1,
        CompareOp::Le => 2,
        CompareOp::Gt => 3,
        CompareOp::Ge => 4,
        CompareOp::Eq => 5,
        CompareOp::Ne => 6,
    }
}

fn logic_tag(op: LogicOp) -> u64 {
    match op {
        LogicOp::And => 1,
        LogicOp::Or => 2,
    }
}

fn write_callback(writer: &mut CanonicalWriter, call: &crate::callback::CallbackCall) {
    let spec = call.spec();
    writer.text(&spec.name);
    writer.texts(spec.inputs.as_ref());
    write_callback_output(writer, &spec.output);
    writer.word(count(spec.dependencies.len()));
    for dependency in spec.dependencies.as_ref() {
        writer.text(&dependency.name).text(&dependency.version);
    }
    match &spec.implementation {
        CallbackImplementation::Runtime => {
            writer.word(1);
        }
        CallbackImplementation::Recoverable { source_sha256, entry_point } => {
            writer.word(2).text(source_sha256).text(entry_point);
        }
    }
    writer.word(count(call.args().len()));
    for arg in call.args() {
        write_expr(writer, arg);
    }
}

fn write_callback_output(writer: &mut CanonicalWriter, output: &CallbackOutput) {
    match output {
        CallbackOutput::Real => {
            writer.word(1);
        }
        CallbackOutput::Range { lower, upper } => {
            writer.word(2).real(*lower).real(*upper);
        }
        CallbackOutput::Levels(levels) => {
            writer.word(3);
            writer.reals(levels);
        }
    }
}

/// A contrast is its form and its signed worlds in formula order. Caller-chosen world
/// labels and the reference flag are not written: the reference is the baseline term
/// of the formula, so relabelling the worlds is the same target.
fn write_contrast(writer: &mut CanonicalWriter, contrast: &InterventionContrast) {
    if contrast.form == ContrastForm::Linear {
        writer.word(3);
        let mut terms: Vec<_> = contrast
            .terms()
            .iter()
            .map(|term| {
                let mut item = CanonicalWriter::new("antecedent.linear_world_term.v1");
                item.real(term.coefficient);
                if let Some(world) = contrast.worlds.get(term.world) {
                    write_world_content(&mut item, world);
                }
                item.finish()
            })
            .collect();
        terms.sort_unstable();
        writer.word(count(terms.len()));
        for term in terms {
            writer.bytes(&term);
        }
        // Unlike the fixed difference/interaction forms, arbitrary coefficients do not
        // determine a unique reference. Its scientific content is part of this question.
        if let Some(reference) = contrast.worlds.get(contrast.reference) {
            write_world_content(writer, reference);
        }
        return;
    }
    writer.word(match contrast.form {
        ContrastForm::Difference { .. } => 1,
        ContrastForm::Interaction { .. } => 2,
        ContrastForm::Linear => unreachable!("linear form encoded above"),
    });
    let terms = contrast.terms();
    writer.word(count(terms.len()));
    for term in terms {
        // Keep previous fixed-form identities byte-for-byte compatible.
        writer.word(if term.coefficient.is_sign_negative() { u64::MAX } else { 1 });
        if let Some(world) = contrast.worlds.get(term.world) {
            write_world_content(writer, world);
        }
    }
}

fn write_surface(writer: &mut CanonicalWriter, surface: &DoseTimeSurface) {
    write_var(writer, surface.dose);
    writer.word(count(surface.level_bits.len()));
    for bits in surface.level_bits.as_ref() {
        writer.word(*bits);
    }
    writer.word(count(surface.horizons.len()));
    for horizon in surface.horizons.as_ref() {
        writer.word(u64::from(*horizon));
    }
    writer.word(count(surface.cells.len()));
    for id in surface.cells.as_ref() {
        match surface.worlds.get(*id) {
            Some(world) => {
                writer.word(1);
                write_world_content(writer, world);
            }
            None => {
                writer.word(0);
            }
        }
    }
}

fn write_mediation(writer: &mut CanonicalWriter, mediation: &MediationTarget) {
    write_binding(writer, &mediation.binding);
    write_time(writer, &mediation.time);
    match &mediation.schedule {
        None => {
            writer.word(0);
        }
        Some(policy) => {
            writer.word(1);
            write_policy(writer, policy);
        }
    }
    write_var(writer, mediation.arms.treatment);
    write_vars(writer, &mediation.arms.mediators);
    write_intervention(writer, &mediation.arms.control);
    write_intervention(writer, &mediation.arms.active);
    writer.word(component_tag(mediation.arms.component));
    let mut tags: Vec<u64> = mediation.premises.iter().map(|premise| premise.tag()).collect();
    tags.sort_unstable();
    writer.word(count(tags.len()));
    for tag in tags {
        writer.word(tag);
    }
}

fn component_tag(component: MediationComponent) -> u64 {
    match component {
        MediationComponent::Total => 1,
        MediationComponent::Direct => 2,
        MediationComponent::Mediated => 3,
        MediationComponent::NaturalDirect => 4,
        MediationComponent::NaturalIndirect => 5,
    }
}

fn write_assumptions(writer: &mut CanonicalWriter, assumptions: &ModelAssumptions) {
    writer.word(graph_tag(assumptions.graph));
    // What is assumed, and where, is the target. How it entered, whether a diagnostic
    // has since supported it, record order, and repeats are not.
    let records: BTreeSet<[u8; 32]> =
        assumptions.records.entries.iter().map(record_digest).collect();
    writer.word(count(records.len()));
    for digest in records {
        writer.bytes(&digest);
    }
}

fn record_digest(record: &AssumptionRecord) -> [u8; 32] {
    let mut writer = CanonicalWriter::new("antecedent.scientific_target.assumption.v1");
    write_assumption(&mut writer, &record.assumption);
    write_scope(&mut writer, &record.scope);
    writer.finish()
}

fn graph_tag(graph: GraphFamily) -> u64 {
    match graph {
        GraphFamily::Unspecified => 1,
        GraphFamily::Dag => 2,
        GraphFamily::Admg => 3,
        GraphFamily::Cpdag => 4,
        GraphFamily::Pag => 5,
        GraphFamily::TemporalDag => 6,
        GraphFamily::TemporalCpdag => 7,
        GraphFamily::TemporalPag => 8,
        GraphFamily::TemporalAdmg => 9,
        GraphFamily::CoDetermined => 10,
        GraphFamily::Graphless => 11,
    }
}

fn write_assumption(writer: &mut CanonicalWriter, assumption: &Assumption) {
    match assumption {
        Assumption::CausalMarkov => {
            writer.word(1);
        }
        Assumption::Faithfulness => {
            writer.word(2);
        }
        Assumption::CausalSufficiency => {
            writer.word(3);
        }
        Assumption::Consistency => {
            writer.word(4);
        }
        Assumption::Positivity => {
            writer.word(5);
        }
        Assumption::NoInterference => {
            writer.word(6);
        }
        Assumption::Stationarity => {
            writer.word(7);
        }
        Assumption::PiecewiseStationarity => {
            writer.word(8);
        }
        Assumption::NoSelectionBias => {
            writer.word(9);
        }
        Assumption::ExclusionRestriction { instrument } => {
            writer.word(10);
            write_var(writer, *instrument);
        }
        Assumption::Monotonicity => {
            writer.word(11);
        }
        Assumption::ParametricRestriction(restriction) => {
            writer.word(12).text(&restriction.id).text(&restriction.description);
        }
        Assumption::PriorRestriction(restriction) => {
            writer.word(13).text(&restriction.id).text(&restriction.description);
        }
        Assumption::Custom { id, description } => {
            writer.word(14).text(id).text(description);
        }
    }
}

fn write_scope(writer: &mut CanonicalWriter, scope: &AssumptionScope) {
    match scope {
        AssumptionScope::Global => {
            writer.word(1);
        }
        AssumptionScope::Identification => {
            writer.word(2);
        }
        AssumptionScope::Estimation => {
            writer.word(3);
        }
        AssumptionScope::Discovery => {
            writer.word(4);
        }
        AssumptionScope::Variables { variables } => {
            let mut variables = variables.to_vec();
            variables.sort_unstable();
            variables.dedup();
            writer.word(5);
            write_vars(writer, &variables);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::sync::Arc;

    use super::{
        Conditioning, ContrastForm, ContrastTerm, CrossWorldPremise, DoseTimeSurface, GraphFamily,
        HistoryWindow, InferenceSelection, InitialLaw, InterventionContrast, InterventionWorld,
        MediationArms, MediationComponent, MediationTarget, ModelAssumptions, ObservationUnit,
        OutcomeQuantity, QuantityUnits, ScientificTarget, ScientificTargetError, ScientificWorldId,
        TargetForm, TimeFrame, WorldAction, WorldBinding, WorldCollection,
    };
    use crate::VariableId;
    use crate::assumption::{
        Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
        AssumptionStatus, PriorAssumption,
    };
    use crate::intervention::{Intervention, TemporalPolicy};
    use crate::query::TargetPopulation;
    use crate::value::Value;

    fn vid(raw: u32) -> VariableId {
        VariableId::from_raw(raw)
    }

    fn world_id(raw: u32) -> ScientificWorldId {
        ScientificWorldId::from_raw(raw)
    }

    fn model(graph: GraphFamily) -> ModelAssumptions {
        ModelAssumptions::new(graph, AssumptionSet::new())
    }

    fn binding(population: TargetPopulation, conditioning: Conditioning) -> WorldBinding {
        WorldBinding::new(
            population,
            ObservationUnit::new("subject").unwrap(),
            OutcomeQuantity::mean(vid(1), QuantityUnits::dimensionless()).unwrap(),
            conditioning,
        )
        .unwrap()
    }

    fn binding_with(
        population: TargetPopulation,
        unit: &str,
        outcome: VariableId,
        units: &str,
        conditioning: Conditioning,
    ) -> WorldBinding {
        WorldBinding::new(
            population,
            ObservationUnit::new(unit).unwrap(),
            OutcomeQuantity::mean(outcome, QuantityUnits::new(units).unwrap()).unwrap(),
            conditioning,
        )
        .unwrap()
    }

    fn target(form: TargetForm, assumptions: ModelAssumptions) -> ScientificTarget {
        ScientificTarget::new(form, assumptions, InferenceSelection::Unspecified)
    }

    fn temporal_contrast(
        policy: TemporalPolicy,
        horizon: u32,
        history: HistoryWindow,
        population: TargetPopulation,
        conditioning: Conditioning,
        initial_law: InitialLaw,
    ) -> ScientificTarget {
        let time = TimeFrame::temporal(horizon, history, initial_law).unwrap();
        let shared = binding(population, conditioning);
        let treatment = vid(0);
        let reference = InterventionWorld::new(
            world_id(0),
            WorldAction::schedule(set_level(treatment, 0.0), policy.clone()).unwrap(),
            true,
            shared.clone(),
            time.clone(),
        )
        .unwrap();
        let active = InterventionWorld::new(
            world_id(1),
            WorldAction::schedule(set_level(treatment, 1.0), policy).unwrap(),
            false,
            shared,
            time,
        )
        .unwrap();
        let worlds = WorldCollection::new([reference, active]).unwrap();
        let contrast =
            InterventionContrast::difference(worlds, world_id(1), world_id(0), world_id(0))
                .unwrap();
        target(TargetForm::Contrast(contrast), model(GraphFamily::Dag))
    }

    fn pulse_contrast() -> ScientificTarget {
        temporal_contrast(
            TemporalPolicy::pulse(0),
            1,
            HistoryWindow::planner_default(),
            TargetPopulation::AllObserved,
            Conditioning::unconditional(),
            InitialLaw::named("observed_origin").unwrap(),
        )
    }

    fn static_contrast(
        population: TargetPopulation,
        conditioning: Conditioning,
    ) -> ScientificTarget {
        let time = TimeFrame::static_decision();
        let shared = binding(population, conditioning);
        let treatment = vid(0);
        let reference = InterventionWorld::new(
            world_id(0),
            WorldAction::at_decision([set_level(treatment, 0.0)]).unwrap(),
            true,
            shared.clone(),
            time.clone(),
        )
        .unwrap();
        let active = InterventionWorld::new(
            world_id(1),
            WorldAction::at_decision([set_level(treatment, 1.0)]).unwrap(),
            false,
            shared,
            time,
        )
        .unwrap();
        let worlds = WorldCollection::new([reference, active]).unwrap();
        let contrast =
            InterventionContrast::difference(worlds, world_id(1), world_id(0), world_id(0))
                .unwrap();
        target(TargetForm::Contrast(contrast), model(GraphFamily::Dag))
    }

    fn set_level(variable: VariableId, level: f64) -> Intervention {
        Intervention::set(variable, Value::f64(level))
    }

    fn digest_ne(left: &ScientificTarget, right: &ScientificTarget) {
        assert_ne!(left, right);
        assert_ne!(left.identity(), right.identity());
    }

    /// `PulseEffect` and `SustainedEffect` both use `temporal.linear.adjustment`.
    #[test]
    fn pulse_effect_differs_from_sustained_effect() {
        let pulse = pulse_contrast();
        let sustained = temporal_contrast(
            TemporalPolicy::sustained(0, 1),
            1,
            HistoryWindow::planner_default(),
            TargetPopulation::AllObserved,
            Conditioning::unconditional(),
            InitialLaw::named("observed_origin").unwrap(),
        );
        digest_ne(&pulse, &sustained);
    }

    /// `TemporalPolicy::is_single_step` is true for both a pulse and `Sustained { from: 0, until: 0 }`.
    #[test]
    fn single_step_sustained_differs_from_pulse() {
        let pulse = pulse_contrast();
        let sustained = temporal_contrast(
            TemporalPolicy::sustained(0, 0),
            1,
            HistoryWindow::planner_default(),
            TargetPopulation::AllObserved,
            Conditioning::unconditional(),
            InitialLaw::named("observed_origin").unwrap(),
        );
        digest_ne(&pulse, &sustained);
    }

    /// `TemporalConditionalPulse` and `ConditionalEffect` (`conditional.linear.adjustment`).
    #[test]
    fn conditional_pulse_differs_from_static_conditional_effect() {
        let modifiers = Conditioning::on([vid(2)]);
        let pulse = temporal_contrast(
            TemporalPolicy::pulse(0),
            1,
            HistoryWindow::planner_default(),
            TargetPopulation::AllObserved,
            modifiers.clone(),
            InitialLaw::named("observed_origin").unwrap(),
        );
        let static_conditional = static_contrast(TargetPopulation::AllObserved, modifiers);
        digest_ne(&pulse, &static_conditional);
    }

    /// Horizon 1 does not make a pulse into `AverageEffect`.
    #[test]
    fn horizon_one_pulse_differs_from_static_average_effect() {
        digest_ne(
            &pulse_contrast(),
            &static_contrast(TargetPopulation::AllObserved, Conditioning::unconditional()),
        );
    }

    #[test]
    fn inference_label_does_not_change_identity() {
        let frequentist = pulse_contrast().with_inference(InferenceSelection::Frequentist);
        let bayesian = pulse_contrast().with_inference(InferenceSelection::Bayesian);
        assert_eq!(frequentist, bayesian);
        assert_eq!(frequentist.identity(), bayesian.identity());
        assert_ne!(frequentist.inference(), bayesian.inference());
        let mut left = DefaultHasher::new();
        let mut right = DefaultHasher::new();
        frequentist.hash(&mut left);
        bayesian.hash(&mut right);
        assert_eq!(left.finish(), right.finish());
    }

    #[test]
    fn population_changes_digest() {
        let all = pulse_contrast();
        let treated = temporal_contrast(
            TemporalPolicy::pulse(0),
            1,
            HistoryWindow::planner_default(),
            TargetPopulation::Treated,
            Conditioning::unconditional(),
            InitialLaw::named("observed_origin").unwrap(),
        );
        digest_ne(&all, &treated);
    }

    #[test]
    fn horizon_changes_digest() {
        let one = pulse_contrast();
        let two = temporal_contrast(
            TemporalPolicy::pulse(0),
            2,
            HistoryWindow::planner_default(),
            TargetPopulation::AllObserved,
            Conditioning::unconditional(),
            InitialLaw::named("observed_origin").unwrap(),
        );
        digest_ne(&one, &two);
    }

    #[test]
    fn history_changes_digest() {
        let default_history = pulse_contrast();
        let explicit_zero = temporal_contrast(
            TemporalPolicy::pulse(0),
            1,
            HistoryWindow::max_lag(0),
            TargetPopulation::AllObserved,
            Conditioning::unconditional(),
            InitialLaw::named("observed_origin").unwrap(),
        );
        let one_lag = temporal_contrast(
            TemporalPolicy::pulse(0),
            1,
            HistoryWindow::max_lag(1),
            TargetPopulation::AllObserved,
            Conditioning::unconditional(),
            InitialLaw::named("observed_origin").unwrap(),
        );
        digest_ne(&default_history, &explicit_zero);
        digest_ne(&default_history, &one_lag);
        digest_ne(&explicit_zero, &one_lag);
    }

    #[test]
    fn initial_law_changes_digest() {
        let origin = pulse_contrast();
        let equilibrium = temporal_contrast(
            TemporalPolicy::pulse(0),
            1,
            HistoryWindow::planner_default(),
            TargetPopulation::AllObserved,
            Conditioning::unconditional(),
            InitialLaw::named("equilibrium").unwrap(),
        );
        let contemporaneous = temporal_contrast(
            TemporalPolicy::pulse(0),
            1,
            HistoryWindow::planner_default(),
            TargetPopulation::AllObserved,
            Conditioning::unconditional(),
            InitialLaw::contemporaneous_observed(),
        );
        digest_ne(&origin, &equilibrium);
        digest_ne(&origin, &contemporaneous);
    }

    #[test]
    fn worlds_must_share_population_outcome_units_and_conditioning() {
        let time = TimeFrame::static_decision();
        let action = |level: f64| WorldAction::at_decision([set_level(vid(0), level)]).unwrap();
        let pair = |left: WorldBinding, right: WorldBinding| {
            WorldCollection::new([
                InterventionWorld::new(world_id(0), action(0.0), true, left, time.clone()).unwrap(),
                InterventionWorld::new(world_id(1), action(1.0), false, right, time.clone())
                    .unwrap(),
            ])
        };
        let base = binding_with(
            TargetPopulation::AllObserved,
            "subject",
            vid(1),
            "1",
            Conditioning::unconditional(),
        );
        let cases = [
            (
                "population",
                binding_with(
                    TargetPopulation::Treated,
                    "subject",
                    vid(1),
                    "1",
                    Conditioning::unconditional(),
                ),
            ),
            (
                "unit",
                binding_with(
                    TargetPopulation::AllObserved,
                    "cluster",
                    vid(1),
                    "1",
                    Conditioning::unconditional(),
                ),
            ),
            (
                "outcome",
                binding_with(
                    TargetPopulation::AllObserved,
                    "subject",
                    vid(4),
                    "1",
                    Conditioning::unconditional(),
                ),
            ),
            (
                "units",
                binding_with(
                    TargetPopulation::AllObserved,
                    "subject",
                    vid(1),
                    "USD",
                    Conditioning::unconditional(),
                ),
            ),
            (
                "conditioning",
                binding_with(
                    TargetPopulation::AllObserved,
                    "subject",
                    vid(1),
                    "1",
                    Conditioning::on([vid(3)]),
                ),
            ),
        ];
        for (field, other) in cases {
            let err = pair(base.clone(), other).unwrap_err();
            assert_eq!(err, ScientificTargetError::IncompatibleBinding { field });
        }
    }

    fn joint_action(a_level: f64, b_level: f64) -> WorldAction {
        WorldAction::at_decision([set_level(vid(0), a_level), set_level(vid(2), b_level)]).unwrap()
    }

    fn interaction_world(
        id: u32,
        reference: bool,
        a_level: f64,
        b_level: f64,
    ) -> InterventionWorld {
        InterventionWorld::new(
            world_id(id),
            joint_action(a_level, b_level),
            reference,
            binding(TargetPopulation::AllObserved, Conditioning::unconditional()),
            TimeFrame::static_decision(),
        )
        .unwrap()
    }

    #[test]
    fn interaction_uses_explicit_reference_world() {
        let worlds = WorldCollection::new([
            interaction_world(0, true, 0.0, 0.0),
            interaction_world(1, false, 1.0, 0.0),
            interaction_world(2, false, 0.0, 1.0),
            interaction_world(3, false, 1.0, 1.0),
        ])
        .unwrap();
        let contrast = InterventionContrast::interaction(
            worlds,
            world_id(3),
            world_id(1),
            world_id(2),
            world_id(0),
        )
        .unwrap();
        assert_eq!(
            contrast.terms(),
            vec![
                ContrastTerm { world: world_id(3), coefficient: 1.0 },
                ContrastTerm { world: world_id(1), coefficient: -1.0 },
                ContrastTerm { world: world_id(2), coefficient: -1.0 },
                ContrastTerm { world: world_id(0), coefficient: 1.0 },
            ]
        );
        assert!(matches!(contrast.form(), ContrastForm::Interaction { .. }));
        let unmarked = WorldCollection::new([
            interaction_world(0, false, 0.0, 0.0),
            interaction_world(1, false, 1.0, 0.0),
            interaction_world(2, false, 0.0, 1.0),
            interaction_world(3, false, 1.0, 1.0),
        ])
        .unwrap();
        let err = InterventionContrast::interaction(
            unmarked,
            world_id(3),
            world_id(1),
            world_id(2),
            world_id(0),
        )
        .unwrap_err();
        assert_eq!(err, ScientificTargetError::ReferenceNotMarked { id: world_id(0) });

        let difference_worlds = WorldCollection::new([
            interaction_world(0, true, 0.0, 0.0),
            interaction_world(1, false, 1.0, 0.0),
        ])
        .unwrap();
        let err = InterventionContrast::difference(
            difference_worlds,
            world_id(1),
            world_id(0),
            world_id(1),
        )
        .unwrap_err();
        assert_eq!(err, ScientificTargetError::ReferenceIsNotBaseline { id: world_id(1) });
    }

    #[test]
    fn dose_time_surface_is_a_grid_not_a_contrast() {
        let shared = binding(TargetPopulation::AllObserved, Conditioning::unconditional());
        let law = InitialLaw::named("observed_origin").unwrap();
        let mut built = Vec::new();
        let mut cells = Vec::new();
        for (dose_index, level) in [0.0_f64, 1.0].into_iter().enumerate() {
            for (horizon_index, horizon) in [1_u32, 2].into_iter().enumerate() {
                let id = world_id(u32::try_from(dose_index * 2 + horizon_index).unwrap());
                let time =
                    TimeFrame::temporal(horizon, HistoryWindow::max_lag(1), law.clone()).unwrap();
                let action =
                    WorldAction::schedule(set_level(vid(0), level), TemporalPolicy::pulse(0))
                        .unwrap();
                built.push(
                    InterventionWorld::new(id, action, id.raw() == 0, shared.clone(), time)
                        .unwrap(),
                );
                cells.push(id);
            }
        }
        let worlds = WorldCollection::new(built.clone()).unwrap();
        let surface =
            DoseTimeSurface::new(worlds.clone(), vid(0), [0.0, 1.0], [1, 2], cells).unwrap();
        assert_eq!(surface.horizons(), &[1, 2]);
        assert_eq!(surface.cells().len(), 4);
        let surface_target = target(TargetForm::DoseTime(surface), model(GraphFamily::Dag));
        assert!(matches!(surface_target.form(), TargetForm::DoseTime(_)));
        let bundle = target(TargetForm::Worlds(worlds), model(GraphFamily::Dag));
        digest_ne(&surface_target, &bundle);

        let reference = InterventionWorld::new(
            built[0].id(),
            built[0].action().clone(),
            true,
            shared.clone(),
            built[0].time().clone(),
        )
        .unwrap();
        let corners = WorldCollection::new([reference, built[3].clone()]).unwrap();
        let err = InterventionContrast::difference(corners, world_id(3), world_id(0), world_id(0))
            .unwrap_err();
        assert_eq!(err, ScientificTargetError::IncompatibleTime);

        let short = WorldCollection::new(built[..3].to_vec()).unwrap();
        let err = DoseTimeSurface::new(
            short,
            vid(0),
            [0.0, 1.0],
            [1, 2],
            [world_id(0), world_id(1), world_id(2)],
        )
        .unwrap_err();
        assert!(matches!(err, ScientificTargetError::NotADoseTimeGrid { .. }));
    }

    fn mediation(
        time: TimeFrame,
        schedule: Option<TemporalPolicy>,
        component: MediationComponent,
        premises: &[CrossWorldPremise],
    ) -> Result<MediationTarget, ScientificTargetError> {
        let arms = MediationArms::new(
            vid(0),
            [vid(2)],
            set_level(vid(0), 0.0),
            set_level(vid(0), 1.0),
            component,
        )
        .unwrap();
        MediationTarget::new(
            binding(TargetPopulation::AllObserved, Conditioning::unconditional()),
            time,
            schedule,
            arms,
            premises.iter().copied(),
        )
    }

    #[test]
    fn mediation_requires_cross_world_premises_and_is_not_a_contrast() {
        let err =
            mediation(TimeFrame::static_decision(), None, MediationComponent::NaturalDirect, &[])
                .unwrap_err();
        assert_eq!(err, ScientificTargetError::MediationNeedsCrossWorldPremises);
        let mediated = mediation(
            TimeFrame::static_decision(),
            None,
            MediationComponent::NaturalDirect,
            &[
                CrossWorldPremise::Composition,
                CrossWorldPremise::CrossWorldIndependence,
                CrossWorldPremise::PathSpecificNested,
            ],
        )
        .unwrap();
        let contrast =
            static_contrast(TargetPopulation::AllObserved, Conditioning::unconditional());
        digest_ne(&target(TargetForm::Mediation(mediated), model(GraphFamily::Dag)), &contrast);
        assert!(matches!(contrast.form(), TargetForm::Contrast(_)));
    }

    /// `MediationEffect` and `TemporalMediationEffect`.
    #[test]
    fn temporal_mediation_differs_from_static_mediation() {
        let premises = [
            CrossWorldPremise::Composition,
            CrossWorldPremise::CrossWorldIndependence,
            CrossWorldPremise::PathSpecificNested,
        ];
        let static_mediation = mediation(
            TimeFrame::static_decision(),
            None,
            MediationComponent::NaturalDirect,
            &premises,
        )
        .unwrap();
        let temporal_mediation = mediation(
            TimeFrame::temporal(
                1,
                HistoryWindow::planner_default(),
                InitialLaw::named("observed_origin").unwrap(),
            )
            .unwrap(),
            Some(TemporalPolicy::pulse(0)),
            MediationComponent::NaturalDirect,
            &premises,
        )
        .unwrap();
        digest_ne(
            &target(TargetForm::Mediation(static_mediation), model(GraphFamily::Dag)),
            &target(TargetForm::Mediation(temporal_mediation), model(GraphFamily::Dag)),
        );
    }

    #[test]
    fn model_assumptions_change_digest() {
        let dag = pulse_contrast();
        let admg = ScientificTarget::new(
            dag.form().clone(),
            model(GraphFamily::Admg),
            InferenceSelection::Frequentist,
        );
        digest_ne(&dag, &admg);
        let mut records = AssumptionSet::new();
        records.push(AssumptionRecord {
            assumption: Assumption::PriorRestriction(PriorAssumption {
                id: Arc::from("half_normal"),
                description: Arc::from("shrinks an unidentified component"),
            }),
            source: AssumptionSource::UserDeclared,
            scope: AssumptionScope::Global,
            status: AssumptionStatus::Declared,
        });
        let with_prior = ScientificTarget::new(
            dag.form().clone(),
            ModelAssumptions::new(GraphFamily::Dag, records),
            InferenceSelection::Bayesian,
        );
        digest_ne(&dag, &with_prior);
    }

    fn interaction_of(
        reference: (f64, f64),
        a_at_reference: (f64, f64),
        b_at_reference: (f64, f64),
        joint: (f64, f64),
    ) -> Result<InterventionContrast, ScientificTargetError> {
        let worlds = WorldCollection::new([
            interaction_world(0, true, reference.0, reference.1),
            interaction_world(1, false, a_at_reference.0, a_at_reference.1),
            interaction_world(2, false, b_at_reference.0, b_at_reference.1),
            interaction_world(3, false, joint.0, joint.1),
        ])
        .unwrap();
        InterventionContrast::interaction(
            worlds,
            world_id(3),
            world_id(1),
            world_id(2),
            world_id(0),
        )
    }

    #[test]
    fn interaction_accepts_a_coherent_two_by_two_at_a_nonzero_reference() {
        // Y(a,b), Y(a,r_b), Y(r_a,b), Y(r_a,r_b) with a reference that is not zero.
        interaction_of((5.0, 5.0), (1.0, 5.0), (5.0, 2.0), (1.0, 2.0)).unwrap();
        interaction_of((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)).unwrap();
    }

    #[test]
    fn interaction_refuses_worlds_that_do_not_realise_the_formula() {
        let cases = [
            // The old shifted reference: both factors differ from (5,5) at once.
            ((5.0, 5.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)),
            // The joint world takes the wrong level of A.
            ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (2.0, 1.0)),
            // The joint world takes the wrong level of B.
            ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 3.0)),
            // a_at_reference moves B as well as A.
            ((0.0, 0.0), (1.0, 4.0), (0.0, 1.0), (1.0, 1.0)),
            // b_at_reference moves A as well as B.
            ((0.0, 0.0), (1.0, 0.0), (6.0, 1.0), (1.0, 1.0)),
            // Factor A never moves.
            ((0.0, 0.0), (0.0, 0.0), (0.0, 1.0), (0.0, 1.0)),
        ];
        for (reference, a, b, joint) in cases {
            let err = interaction_of(reference, a, b, joint).unwrap_err();
            assert!(
                matches!(err, ScientificTargetError::IncoherentContrast { .. }),
                "{reference:?} {a:?} {b:?} {joint:?}: {err:?}"
            );
        }
    }

    #[test]
    fn interaction_projects_multi_channel_sequences_onto_their_own_coordinates() {
        use crate::intervention::{InterventionSequence, SequencedIntervention};
        let shared = binding(TargetPopulation::AllObserved, Conditioning::unconditional());
        let time = TimeFrame::temporal(
            2,
            HistoryWindow::max_lag(2),
            InitialLaw::contemporaneous_observed(),
        )
        .unwrap();
        let cell = |id: u32, a: [f64; 2], b: [f64; 2]| {
            let steps = (0..2)
                .flat_map(|index| {
                    let at = i32::try_from(index).unwrap();
                    [
                        SequencedIntervention::new(
                            set_level(vid(0), a[index]),
                            TemporalPolicy::pulse(at),
                        ),
                        SequencedIntervention::new(
                            set_level(vid(2), b[index]),
                            TemporalPolicy::pulse(at),
                        ),
                    ]
                })
                .collect::<Vec<_>>();
            let action = WorldAction::schedule(
                Intervention::Sequence(InterventionSequence::new(steps)),
                TemporalPolicy::sustained(0, 1),
            )
            .unwrap();
            InterventionWorld::new(world_id(id), action, id == 0, shared.clone(), time.clone())
                .unwrap()
        };
        let build = |joint_b: [f64; 2], a_world_b: [f64; 2]| {
            let worlds = WorldCollection::new([
                cell(0, [0.0; 2], [0.0; 2]),
                cell(1, [1.0, 2.0], a_world_b),
                cell(2, [0.0; 2], [3.0, 4.0]),
                cell(3, [1.0, 2.0], joint_b),
            ])
            .unwrap();
            InterventionContrast::interaction(
                worlds,
                world_id(3),
                world_id(1),
                world_id(2),
                world_id(0),
            )
        };
        build([3.0, 4.0], [0.0; 2]).unwrap();
        assert!(matches!(
            build([3.0, 5.0], [0.0; 2]),
            Err(ScientificTargetError::IncoherentContrast { .. })
        ));
        assert!(matches!(
            build([3.0, 4.0], [0.0, 1.0]),
            Err(ScientificTargetError::IncoherentContrast { .. })
        ));
    }

    #[test]
    fn interaction_accepts_a_dose_and_schedule_factorisation() {
        let shared = binding(TargetPopulation::AllObserved, Conditioning::unconditional());
        let time = TimeFrame::temporal(
            2,
            HistoryWindow::planner_default(),
            InitialLaw::contemporaneous_observed(),
        )
        .unwrap();
        let cell = |id: u32, reference: bool, level: f64, policy: TemporalPolicy| {
            InterventionWorld::new(
                world_id(id),
                WorldAction::schedule(set_level(vid(0), level), policy).unwrap(),
                reference,
                shared.clone(),
                time.clone(),
            )
            .unwrap()
        };
        let pulse = TemporalPolicy::pulse(0);
        let held = TemporalPolicy::sustained(0, 1);
        let build = |joint_policy: TemporalPolicy| {
            let worlds = WorldCollection::new([
                cell(0, true, 0.0, pulse.clone()),
                cell(1, false, 1.0, pulse.clone()),
                cell(2, false, 0.0, held.clone()),
                cell(3, false, 1.0, joint_policy),
            ])
            .unwrap();
            InterventionContrast::interaction(
                worlds,
                world_id(3),
                world_id(1),
                world_id(2),
                world_id(0),
            )
        };
        build(held.clone()).unwrap();
        let err = build(pulse.clone()).unwrap_err();
        assert!(matches!(err, ScientificTargetError::IncoherentContrast { .. }));
    }

    #[test]
    fn difference_refuses_worlds_over_different_variables() {
        let one = WorldAction::at_decision([set_level(vid(0), 0.0)]).unwrap();
        let two =
            WorldAction::at_decision([set_level(vid(0), 1.0), set_level(vid(2), 1.0)]).unwrap();
        let shared = binding(TargetPopulation::AllObserved, Conditioning::unconditional());
        let time = TimeFrame::static_decision();
        let worlds = WorldCollection::new([
            InterventionWorld::new(world_id(0), one, true, shared.clone(), time.clone()).unwrap(),
            InterventionWorld::new(world_id(1), two, false, shared, time).unwrap(),
        ])
        .unwrap();
        let err = InterventionContrast::difference(worlds, world_id(1), world_id(0), world_id(0))
            .unwrap_err();
        assert!(matches!(err, ScientificTargetError::IncoherentContrast { .. }));
    }

    #[test]
    fn symbolic_marker_is_not_a_world_level() {
        let marker = Intervention::set(vid(0), Value::symbolic_intervention());
        assert!(matches!(
            WorldAction::at_decision([marker.clone()]),
            Err(ScientificTargetError::InvalidAction { .. })
        ));
        assert!(matches!(
            WorldAction::schedule(marker.clone(), TemporalPolicy::pulse(0)),
            Err(ScientificTargetError::InvalidAction { .. })
        ));
        let shift = Intervention::shift(vid(0), Value::symbolic_intervention());
        assert!(WorldAction::at_decision([shift]).is_err());
        let err = MediationArms::new(
            vid(0),
            [vid(2)],
            set_level(vid(0), 0.0),
            marker,
            MediationComponent::Total,
        )
        .unwrap_err();
        assert!(matches!(err, ScientificTargetError::InvalidAction { .. }));
    }

    #[test]
    fn mediation_arms_must_differ() {
        let err = MediationArms::new(
            vid(0),
            [vid(2)],
            set_level(vid(0), 1.0),
            set_level(vid(0), 1.0),
            MediationComponent::Total,
        )
        .unwrap_err();
        assert_eq!(err, ScientificTargetError::MediationArmsNotDistinct);
    }

    fn pulse_world(at: i32, horizon: u32) -> Result<InterventionWorld, ScientificTargetError> {
        InterventionWorld::new(
            world_id(0),
            WorldAction::schedule(set_level(vid(0), 1.0), TemporalPolicy::pulse(at)).unwrap(),
            false,
            binding(TargetPopulation::AllObserved, Conditioning::unconditional()),
            TimeFrame::temporal(
                horizon,
                HistoryWindow::planner_default(),
                InitialLaw::contemporaneous_observed(),
            )
            .unwrap(),
        )
    }

    #[test]
    fn schedule_must_lie_within_the_horizon() {
        assert!(matches!(
            pulse_world(-3, 1),
            Err(ScientificTargetError::PolicyOutsideHorizon { horizon_steps: 1, .. })
        ));
        assert!(matches!(
            pulse_world(7, 1),
            Err(ScientificTargetError::PolicyOutsideHorizon { horizon_steps: 1, .. })
        ));
        pulse_world(0, 1).unwrap();
        pulse_world(-1, 1).unwrap();

        let window = |until: i32| {
            InterventionWorld::new(
                world_id(0),
                WorldAction::schedule(set_level(vid(0), 1.0), TemporalPolicy::sustained(0, until))
                    .unwrap(),
                false,
                binding(TargetPopulation::AllObserved, Conditioning::unconditional()),
                TimeFrame::temporal(
                    2,
                    HistoryWindow::planner_default(),
                    InitialLaw::contemporaneous_observed(),
                )
                .unwrap(),
            )
        };
        window(2).unwrap();
        assert!(matches!(window(3), Err(ScientificTargetError::PolicyOutsideHorizon { .. })));

        let premises = [
            CrossWorldPremise::Composition,
            CrossWorldPremise::CrossWorldIndependence,
            CrossWorldPremise::PathSpecificNested,
        ];
        let temporal = |at: i32| {
            mediation(
                TimeFrame::temporal(
                    1,
                    HistoryWindow::planner_default(),
                    InitialLaw::contemporaneous_observed(),
                )
                .unwrap(),
                Some(TemporalPolicy::pulse(at)),
                MediationComponent::NaturalDirect,
                &premises,
            )
        };
        temporal(0).unwrap();
        assert!(matches!(temporal(7), Err(ScientificTargetError::PolicyOutsideHorizon { .. })));
        assert!(matches!(temporal(-3), Err(ScientificTargetError::PolicyOutsideHorizon { .. })));
    }

    #[test]
    fn a_negative_pulse_inside_the_horizon_is_a_legitimate_target() {
        pulse_world(-1, 1).unwrap();
        assert!(matches!(
            pulse_world(-2, 1),
            Err(ScientificTargetError::PolicyOutsideHorizon { horizon_steps: 1, .. })
        ));
    }

    #[test]
    fn natural_components_need_the_composition_premise() {
        for component in [MediationComponent::NaturalDirect, MediationComponent::NaturalIndirect] {
            let err = mediation(
                TimeFrame::static_decision(),
                None,
                component,
                &[CrossWorldPremise::CrossWorldIndependence, CrossWorldPremise::PathSpecificNested],
            )
            .unwrap_err();
            assert_eq!(err, ScientificTargetError::MediationNeedsCrossWorldPremises);
        }
        // Path-specific components are not defined through the nested counterfactual.
        mediation(
            TimeFrame::static_decision(),
            None,
            MediationComponent::Direct,
            &[CrossWorldPremise::CrossWorldIndependence, CrossWorldPremise::PathSpecificNested],
        )
        .unwrap();
    }

    fn record(
        assumption: Assumption,
        scope: AssumptionScope,
        status: AssumptionStatus,
    ) -> AssumptionRecord {
        AssumptionRecord { assumption, source: AssumptionSource::UserDeclared, scope, status }
    }

    fn with_records(records: Vec<AssumptionRecord>) -> ScientificTarget {
        let mut set = AssumptionSet::new();
        for entry in records {
            set.push(entry);
        }
        ScientificTarget::new(
            pulse_contrast().form().clone(),
            ModelAssumptions::new(GraphFamily::Dag, set),
            InferenceSelection::Unspecified,
        )
    }

    #[test]
    fn assumption_identity_is_a_canonical_set() {
        let markov = || {
            record(Assumption::CausalMarkov, AssumptionScope::Global, AssumptionStatus::Declared)
        };
        let positivity =
            || record(Assumption::Positivity, AssumptionScope::Global, AssumptionStatus::Declared);
        let once = with_records(vec![markov()]).identity();
        assert_eq!(with_records(vec![markov(), markov()]).identity(), once);
        assert_eq!(
            with_records(vec![markov(), positivity()]).identity(),
            with_records(vec![positivity(), markov(), markov()]).identity()
        );
        // A diagnostic moving Declared to Supported, or a different source, is the same science.
        let supported =
            record(Assumption::CausalMarkov, AssumptionScope::Global, AssumptionStatus::Supported);
        assert_eq!(with_records(vec![supported]).identity(), once);
        let derived = AssumptionRecord {
            source: AssumptionSource::Derived { from: Arc::from("discovery") },
            ..markov()
        };
        assert_eq!(with_records(vec![derived]).identity(), once);
        // What is assumed, and where, is the science.
        assert_ne!(with_records(vec![positivity()]).identity(), once);
        let scoped = record(
            Assumption::CausalMarkov,
            AssumptionScope::Estimation,
            AssumptionStatus::Declared,
        );
        assert_ne!(with_records(vec![scoped]).identity(), once);
        let custom = |description: &str| {
            record(
                Assumption::Custom { id: Arc::from("x"), description: Arc::from(description) },
                AssumptionScope::Global,
                AssumptionStatus::Declared,
            )
        };
        assert_ne!(
            with_records(vec![custom("a")]).identity(),
            with_records(vec![custom("b")]).identity()
        );
    }

    #[test]
    fn contrast_identity_ignores_world_labels() {
        let relabelled = {
            let time = TimeFrame::static_decision();
            let shared = binding(TargetPopulation::AllObserved, Conditioning::unconditional());
            let reference = InterventionWorld::new(
                world_id(40),
                WorldAction::at_decision([set_level(vid(0), 0.0)]).unwrap(),
                true,
                shared.clone(),
                time.clone(),
            )
            .unwrap();
            let active = InterventionWorld::new(
                world_id(9),
                WorldAction::at_decision([set_level(vid(0), 1.0)]).unwrap(),
                false,
                shared,
                time,
            )
            .unwrap();
            let worlds = WorldCollection::new([reference, active]).unwrap();
            let contrast =
                InterventionContrast::difference(worlds, world_id(9), world_id(40), world_id(40))
                    .unwrap();
            target(TargetForm::Contrast(contrast), model(GraphFamily::Dag))
        };
        let original =
            static_contrast(TargetPopulation::AllObserved, Conditioning::unconditional());
        assert_eq!(relabelled.identity(), original.identity());
    }

    fn static_world(id: u32, reference: bool, level: f64) -> InterventionWorld {
        InterventionWorld::new(
            world_id(id),
            WorldAction::at_decision([set_level(vid(0), level)]).unwrap(),
            reference,
            binding(TargetPopulation::AllObserved, Conditioning::unconditional()),
            TimeFrame::static_decision(),
        )
        .unwrap()
    }

    fn dose_surface(id_base: u32, high: f64, mark_first: bool) -> ScientificTarget {
        let shared = binding(TargetPopulation::AllObserved, Conditioning::unconditional());
        let law = InitialLaw::named("observed_origin").unwrap();
        let mut built = Vec::new();
        let mut cells = Vec::new();
        for (dose_index, level) in [0.0_f64, high].into_iter().enumerate() {
            for (horizon_index, horizon) in [1_u32, 2].into_iter().enumerate() {
                let slot = dose_index * 2 + horizon_index;
                let id = world_id(id_base + u32::try_from(slot).unwrap());
                let time =
                    TimeFrame::temporal(horizon, HistoryWindow::max_lag(1), law.clone()).unwrap();
                let action =
                    WorldAction::schedule(set_level(vid(0), level), TemporalPolicy::pulse(0))
                        .unwrap();
                let reference = mark_first && slot == 0;
                built.push(
                    InterventionWorld::new(id, action, reference, shared.clone(), time).unwrap(),
                );
                cells.push(id);
            }
        }
        let worlds = WorldCollection::new(built).unwrap();
        let surface = DoseTimeSurface::new(worlds, vid(0), [0.0, high], [1, 2], cells).unwrap();
        target(TargetForm::DoseTime(surface), model(GraphFamily::Dag))
    }

    #[test]
    fn world_id_and_reference_flag_do_not_change_content_identity() {
        let assumptions = model(GraphFamily::Dag);
        let world = |id: u32, reference: bool, level: f64| {
            target(TargetForm::World(static_world(id, reference, level)), assumptions.clone())
        };
        let baseline = world(0, true, 0.0);
        let relabelled = world(7, false, 0.0);
        assert_ne!(baseline, relabelled);
        assert_eq!(baseline.identity(), relabelled.identity());
        assert_ne!(baseline.identity(), world(0, true, 1.0).identity());

        let bundle = |ids: [u32; 2], flags: [bool; 2], high: f64| {
            let worlds = WorldCollection::new([
                static_world(ids[0], flags[0], 0.0),
                static_world(ids[1], flags[1], high),
            ])
            .unwrap();
            target(TargetForm::Worlds(worlds), assumptions.clone())
        };
        let ordered = bundle([0, 1], [true, false], 1.0);
        let swapped = bundle([9, 4], [false, true], 1.0);
        assert_ne!(ordered, swapped);
        assert_eq!(ordered.identity(), swapped.identity());
        assert_ne!(ordered.identity(), bundle([0, 1], [true, false], 2.0).identity());

        let surface = dose_surface(0, 1.0, true);
        let relabelled_surface = dose_surface(10, 1.0, false);
        assert_ne!(surface, relabelled_surface);
        assert_eq!(surface.identity(), relabelled_surface.identity());
        assert_ne!(surface.identity(), dose_surface(0, 2.0, true).identity());
    }

    #[test]
    fn negative_zero_level_is_the_same_world_as_zero() {
        let build = |level: f64| {
            let action = WorldAction::at_decision([set_level(vid(0), level)]).unwrap();
            InterventionWorld::new(
                world_id(0),
                action,
                true,
                binding(TargetPopulation::AllObserved, Conditioning::unconditional()),
                TimeFrame::static_decision(),
            )
            .unwrap()
        };
        let positive = target(TargetForm::World(build(0.0)), model(GraphFamily::Dag));
        let negative = target(TargetForm::World(build(-0.0)), model(GraphFamily::Dag));
        assert_eq!(positive, negative);
        assert_eq!(positive.identity(), negative.identity());
    }

    #[test]
    fn contrast_form_is_not_mediation() {
        let contrast =
            static_contrast(TargetPopulation::AllObserved, Conditioning::unconditional());
        let TargetForm::Contrast(contrast) = contrast.form() else {
            panic!("static helper builds a contrast");
        };
        let label = match contrast.form() {
            ContrastForm::Linear => "linear",
            ContrastForm::Difference { .. } => "difference",
            ContrastForm::Interaction { .. } => "interaction",
        };
        assert_eq!(label, "difference");
    }
    fn three_factor_linear(id_offset: u32, reverse: bool) -> ScientificTarget {
        let binding = WorldBinding::new(
            TargetPopulation::AllObserved,
            ObservationUnit::new("group").unwrap(),
            OutcomeQuantity::mean(vid(4), QuantityUnits::new("USD").unwrap()).unwrap(),
            Conditioning::unconditional(),
        )
        .unwrap();
        let worlds = WorldCollection::new((0_u32..8).map(|cell| {
            InterventionWorld::new(
                world_id(id_offset + cell),
                WorldAction::at_decision(
                    (0..3).map(|channel| set_level(vid(channel), f64::from((cell >> channel) & 1))),
                )
                .unwrap(),
                cell == 0,
                binding.clone(),
                TimeFrame::static_decision(),
            )
            .unwrap()
        }))
        .unwrap();
        let mut terms: Vec<_> = (0_u32..8)
            .map(|cell| {
                (
                    world_id(id_offset + cell),
                    if (3 - cell.count_ones()) % 2 == 0 { 1.0 } else { -1.0 },
                )
            })
            .collect();
        if reverse {
            terms.reverse();
        }
        target(
            TargetForm::Contrast(
                InterventionContrast::linear(worlds, terms, world_id(id_offset)).unwrap(),
            ),
            model(GraphFamily::Dag),
        )
    }
    #[test]
    fn arbitrary_linear_contrast_retains_higher_order_interaction_and_common_uncertainty_coefficients()
     {
        let target = three_factor_linear(0, false);
        let TargetForm::Contrast(contrast) = target.form() else { panic!("linear target") };
        assert_eq!(contrast.form(), ContrastForm::Linear);
        assert_eq!(contrast.terms().len(), 8);
        let value = contrast
            .terms()
            .iter()
            .map(|term| {
                let cell = term.world.raw();
                let a = f64::from(cell & 1);
                let b = f64::from((cell >> 1) & 1);
                let c = f64::from((cell >> 2) & 1);
                term.coefficient * (a + b + c + 2.0 * a * b + 3.0 * a * b * c)
            })
            .sum::<f64>();
        assert!((value - 3.0).abs() < 1e-12);
        assert_eq!(target.identity(), three_factor_linear(0, true).identity());
        assert_eq!(target.identity(), three_factor_linear(100, true).identity());
    }
    #[test]
    fn arbitrary_linear_contrast_refuses_missing_duplicated_or_nonfinite_terms() {
        let base = three_factor_linear(0, false);
        let TargetForm::Contrast(contrast) = base.form() else { panic!("linear target") };
        for invalid in [0.0, f64::NAN, f64::INFINITY] {
            let mut terms: Vec<_> =
                contrast.terms().iter().map(|term| (term.world, term.coefficient)).collect();
            terms[0].1 = invalid;
            assert!(
                InterventionContrast::linear(contrast.worlds().clone(), terms, world_id(0))
                    .is_err()
            );
        }
        let mut terms: Vec<_> =
            contrast.terms().iter().map(|term| (term.world, term.coefficient)).collect();
        terms.pop();
        assert!(matches!(
            InterventionContrast::linear(contrast.worlds().clone(), terms, world_id(0)),
            Err(ScientificTargetError::ContrastWorldSet)
        ));
        let mut terms: Vec<_> =
            contrast.terms().iter().map(|term| (term.world, term.coefficient)).collect();
        terms[1].0 = terms[0].0;
        assert!(matches!(
            InterventionContrast::linear(contrast.worlds().clone(), terms, world_id(0)),
            Err(ScientificTargetError::WorldsNotDistinct)
        ));
    }
    #[test]
    fn arbitrary_linear_contrast_supports_explicit_window_weights_and_identity_changes() {
        let binding = binding(TargetPopulation::AllObserved, Conditioning::unconditional());
        let worlds = WorldCollection::new((1..=3).map(|horizon| {
            InterventionWorld::new(
                world_id(horizon),
                WorldAction::schedule(set_level(vid(0), 1.0), TemporalPolicy::pulse(0)).unwrap(),
                horizon == 1,
                binding.clone(),
                TimeFrame::temporal(
                    horizon,
                    HistoryWindow::max_lag(2),
                    InitialLaw::named("same history law").unwrap(),
                )
                .unwrap(),
            )
            .unwrap()
        }))
        .unwrap();
        let mean = InterventionContrast::linear(
            worlds.clone(),
            [(world_id(1), 0.25), (world_id(2), 0.5), (world_id(3), 0.25)],
            world_id(1),
        )
        .unwrap();
        let changed = InterventionContrast::linear(
            worlds,
            [(world_id(1), 0.5), (world_id(2), 0.25), (world_id(3), 0.25)],
            world_id(1),
        )
        .unwrap();
        assert_ne!(
            target(TargetForm::Contrast(mean), model(GraphFamily::TemporalDag)).identity(),
            target(TargetForm::Contrast(changed), model(GraphFamily::TemporalDag)).identity()
        );
    }
}
