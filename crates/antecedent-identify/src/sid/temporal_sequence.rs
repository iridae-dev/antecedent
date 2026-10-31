//! One finite temporal transport sequence: exact and point-only.
//!
//! The whole sequence `do(A_1 = a_1, ..., A_H = a_H)` is one longitudinal
//! intervention on the explicit unrolled selection diagram of horizon `H`. It is decided by the
//! licensed classical catalog sID route in one call with both actions as the
//! treatment set; the time steps are never transported separately and
//! multiplied. The unrolled diagram fixes the finite discrete structure:
//! baseline covariates, the time-varying covariates observed before each action,
//! one action per step over one shared discrete alphabet, and one outcome after
//! the last action. A selection target on a coordinate is a time-indexed
//! mechanism difference between source and target at that coordinate's slice;
//! any coordinate without one is assumed invariant at its slice, and that
//! assumption is recorded per slice. Only the initial-state law (baseline and
//! step-1 covariates) is fixed per population, not uncertain: temporal sampling
//! intervals, initial-state uncertainty and new-period refresh are separate
//! routes that this decision does not carry.
//!
//! One [`antecedent_core::SearchBudget`] bounds the whole decision: the
//! catalog-aware search, the s-hedge check and every verification replay, then
//! the growth of the history lattice, charged per state with the live bytes
//! retained. A stop is a receipt; it is never a non-identification verdict.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;
use std::sync::Arc;

use antecedent_core::{
    EvidenceCatalog, ExecutionContext, NodeRef, RegimeId, SearchBudget, SearchLimits,
    SearchReceipt, SearchStop, Value, VariableDomain, VariableId, reason_code,
};
use antecedent_graph::{Admg, BitSet, DenseNodeId, SelectionDiagram, TemporalDag};

use super::scenarios::ScenarioCoordinate;
use super::{
    BoundTransportFunctional, CatalogTransportResult, ClassicalTransportQuery,
    ClassicalTransportResult, IdentificationError, SHedgeRecord, SearchCharge, SharedSearch,
    identify_catalog_transport_metered, identify_classical_transport_metered, sid_memory_bytes,
};

/// Most actions in the discrete alphabet under the earlier fixed ceiling.
#[deprecated(
    since = "2.4.1",
    note = "no longer enforced: the alphabet is bounded by the search budget"
)]
pub const TEMPORAL_MAX_ACTIONS: usize = 8;
/// Most coordinates the unrolled diagram had under the earlier fixed ceiling.
#[deprecated(
    since = "2.4.1",
    note = "no longer enforced: the number of coordinates is bounded by the search budget"
)]
pub const TEMPORAL_MAX_OBSERVED: usize = 12;
/// Most complete covariate histories under the earlier fixed ceiling.
#[deprecated(
    since = "2.4.1",
    note = "no longer enforced: every history state is charged to the search budget"
)]
pub const TEMPORAL_MAX_HISTORY_STATES: usize = 4096;

/// Detail: a required history has no certified support.
pub(crate) const TEMPORAL_HISTORY_OUTSIDE_SUPPORT: &str =
    "temporal_transport.history_outside_support";
/// Detail: a horizon the slots do not carry, a new period, or a changed window.
pub const TEMPORAL_HORIZON_DETAIL: &str = "temporal_transport.horizon";
/// Detail: the licensed rules certified nothing.
pub const TEMPORAL_SEARCH_INCOMPLETE: &str = "temporal_transport.search_incomplete";
/// Detail: the shared budget or cancellation stopped the decision.
pub const TEMPORAL_HISTORY_BUDGET: &str = "temporal_transport.history_budget";
/// Detail: an interval was requested.
pub(crate) const TEMPORAL_INTERVAL_REQUESTED: &str = "temporal_transport.interval_requested";
/// Detail: a hard cap was exceeded.
pub const TEMPORAL_BOUNDS_EXCEEDED: &str = "temporal_transport.bounds_exceeded";
/// Detail: the unrolled specification is malformed.
pub const TEMPORAL_INVALID_SPEC: &str = "temporal_transport.invalid_spec";
/// Detail: the action sequence is not a sequence over the alphabet.
pub const TEMPORAL_INVALID_SEQUENCE: &str = "temporal_transport.invalid_sequence";
/// Detail: a derivation exists but the catalog lacks a factor it needs.
pub const TEMPORAL_MISSING_EVIDENCE: &str = "temporal_transport.missing_evidence";
/// Detail: an independently verified s-hedge.
pub const TEMPORAL_CHECKED_OBSTRUCTION: &str = "temporal_transport.checked_obstruction";

/// A refused temporal transport request: a registered top-level reason code and
/// a stable `temporal_transport.*` detail.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{detail}: {message}")]
pub struct TemporalRefusal {
    /// Registered runtime-refusal reason code.
    pub code: &'static str,
    /// Stable `temporal_transport.*` detail.
    pub detail: &'static str,
    /// Human-readable explanation.
    pub message: String,
}

impl TemporalRefusal {
    /// A refusal of `detail` under `code`.
    #[must_use]
    pub fn new(code: &'static str, detail: &'static str, message: impl Into<String>) -> Self {
        Self { code, detail, message: message.into() }
    }

    fn invalid_spec(message: impl Into<String>) -> Self {
        Self::new(reason_code!("invalid_argument"), TEMPORAL_INVALID_SPEC, message)
    }

    fn invalid_sequence(message: impl Into<String>) -> Self {
        Self::new(reason_code!("invalid_argument"), TEMPORAL_INVALID_SEQUENCE, message)
    }

    /// A horizon the slots do not carry, a new period, or a changed measurement window.
    #[must_use]
    pub fn horizon(message: impl Into<String>) -> Self {
        Self::new(reason_code!("route_not_supported"), TEMPORAL_HORIZON_DETAIL, message)
    }

    /// A hard cap exceeded.
    #[must_use]
    pub fn bounds_exceeded(message: impl Into<String>) -> Self {
        Self::new(reason_code!("route_not_supported"), TEMPORAL_BOUNDS_EXCEEDED, message)
    }

    /// A history outside certified support.
    #[must_use]
    pub fn history_outside_support(message: impl Into<String>) -> Self {
        Self::new(
            reason_code!("transport_support_failure"),
            TEMPORAL_HISTORY_OUTSIDE_SUPPORT,
            message,
        )
    }

    /// An interval was requested: temporal sampling and initial-state
    /// uncertainty are not part of this route, which is point-only.
    #[must_use]
    pub fn interval_requested() -> Self {
        Self::new(
            reason_code!("estimator_inference_mismatch"),
            TEMPORAL_INTERVAL_REQUESTED,
            "the sequence route is exact and point-only; temporal sampling and \
             initial-state intervals are not offered here",
        )
    }
}

/// Why a temporal sequence could not be decided at all.
#[derive(Debug, thiserror::Error)]
pub enum TemporalSequenceError {
    /// A refused specification, sequence or bound.
    #[error(transparent)]
    Refused(#[from] TemporalRefusal),
    /// The identifier rejected its input.
    #[error(transparent)]
    Identification(#[from] IdentificationError),
}

impl From<TemporalRefusal> for IdentificationError {
    fn from(refusal: TemporalRefusal) -> Self {
        Self::invalid_input(refusal.to_string())
    }
}

/// Where each coordinate of the unrolled diagram sits in time.
///
/// Slice 0 is the baseline; slice `k` holds the step-`k` covariates and action;
/// the outcome belongs to the last slice. The order in time is `baseline <
/// covariates[0] < actions[0] < covariates[1] < actions[1] < ... < outcome`:
/// one covariate group and one action per step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TemporalSlots {
    /// Baseline covariates: the initial state before step 1.
    pub baseline: Vec<VariableId>,
    /// Time-varying covariates observed before each action, one group per step.
    pub covariates: Vec<Vec<VariableId>>,
    /// The action of each step.
    pub actions: Vec<VariableId>,
    /// The outcome observed after the last action.
    pub outcome: VariableId,
}

impl TemporalSlots {
    /// Number of steps: one action per step.
    #[must_use]
    pub fn horizon(&self) -> usize {
        self.actions.len()
    }

    /// Position in time order (baseline `0`, then each step's covariates and
    /// action, the outcome last), or `None` for a variable not in the slots.
    #[must_use]
    pub fn position(&self, variable: VariableId) -> Option<usize> {
        if self.baseline.contains(&variable) {
            return Some(0);
        }
        for (step, action) in self.actions.iter().enumerate() {
            if self.covariates.get(step).is_some_and(|group| group.contains(&variable)) {
                return Some(1 + 2 * step);
            }
            if variable == *action {
                return Some(2 + 2 * step);
            }
        }
        (variable == self.outcome).then(|| 1 + 2 * self.actions.len())
    }

    /// Time slice of a variable: 0 baseline, `k` for step `k`; the outcome is in
    /// the last slice.
    #[must_use]
    pub fn slice(&self, variable: VariableId) -> Option<u8> {
        let position = self.position(variable)?;
        let slice = if position == 1 + 2 * self.actions.len() {
            self.actions.len()
        } else {
            position.div_ceil(2)
        };
        u8::try_from(slice).ok()
    }

    /// Every coordinate, in time order.
    #[must_use]
    pub fn variables(&self) -> Vec<VariableId> {
        let mut all = self.baseline.clone();
        for (step, action) in self.actions.iter().enumerate() {
            if let Some(group) = self.covariates.get(step) {
                all.extend(group);
            }
            all.push(*action);
        }
        all.push(self.outcome);
        all
    }

    /// Covariates that make up the initial-state law: baseline and step-1 covariates.
    #[must_use]
    pub fn initial_state(&self) -> Vec<VariableId> {
        self.baseline.iter().chain(self.covariates.first().into_iter().flatten()).copied().collect()
    }

    /// Covariates of a complete history: the initial state and the covariates of
    /// every later step.
    #[must_use]
    pub fn history_coordinates(&self) -> Vec<VariableId> {
        self.initial_state()
            .into_iter()
            .chain(self.covariates.iter().skip(1).flatten().copied())
            .collect()
    }
}

/// The finite levels of a declared domain, or `None` for an unbounded one.
#[must_use]
fn domain_levels(domain: &VariableDomain) -> Option<Vec<Value>> {
    match domain {
        VariableDomain::Binary => Some(vec![Value::f64(0.0), Value::f64(1.0)]),
        VariableDomain::Categorical { cardinality } if *cardinality > 0 => {
            Some((0..*cardinality).map(|level| Value::f64(f64::from(level))).collect())
        }
        _ => None,
    }
}

/// A validated explicit unrolling: its horizon, discrete action alphabet, the
/// selection diagram over the unrolled coordinates, and each coordinate's
/// declared name, domain and unit.
#[derive(Clone, Debug)]
pub struct TemporalSequenceSpec {
    horizon: usize,
    slots: TemporalSlots,
    diagram: SelectionDiagram,
    schema: Arc<[ScenarioCoordinate]>,
}

impl TemporalSequenceSpec {
    /// Validate an explicit unrolling.
    ///
    /// The declared `horizon` must be the number of steps the slots carry (one
    /// action and one covariate group per step, at least one step). The slots
    /// must partition the diagram's coordinates; every coordinate is declared
    /// once with a unique name and a finite (binary or categorical) domain; the
    /// actions share one alphabet; no directed
    /// edge runs backward in time; and no selection target is an action, whose
    /// mechanism the sequence replaces. The number of coordinates, actions and
    /// complete covariate histories is bounded only by the search budget the
    /// decision runs under: every history state is charged before it is built.
    ///
    /// # Errors
    /// `route_not_supported` / `temporal_transport.horizon` for a horizon the
    /// slots do not carry,
    /// `route_not_supported` / `temporal_transport.bounds_exceeded` for an
    /// exceeded cap, and `invalid_argument` / `temporal_transport.invalid_spec`
    /// for anything malformed.
    pub fn try_new(
        horizon: usize,
        slots: TemporalSlots,
        diagram: SelectionDiagram,
        coordinates: impl Into<Arc<[ScenarioCoordinate]>>,
    ) -> Result<Self, TemporalRefusal> {
        if horizon == 0 || horizon != slots.actions.len() || horizon != slots.covariates.len() {
            return Err(TemporalRefusal::horizon(format!(
                "horizon {horizon} does not match the slots, which carry {} actions and {} \
                 covariate groups",
                slots.actions.len(),
                slots.covariates.len()
            )));
        }
        if u8::try_from(horizon).is_err() {
            return Err(TemporalRefusal::bounds_exceeded(format!(
                "horizon {horizon} exceeds the {} steps a time slice can index",
                u8::MAX
            )));
        }
        let mut schema = coordinates.into().to_vec();
        schema.sort_by_key(|c| c.variable);
        let graph = diagram.causal_graph();
        let nodes = graph.nodes().iter().copied().collect::<BTreeSet<NodeRef>>();
        let declared = schema.iter().map(|c| NodeRef::Static(c.variable)).collect::<BTreeSet<_>>();
        let names = schema.iter().map(|c| c.name.clone()).collect::<BTreeSet<_>>();
        let variables = slots.variables();
        let slotted = variables.iter().map(|v| NodeRef::Static(*v)).collect::<BTreeSet<_>>();
        if declared != nodes
            || declared.len() != schema.len()
            || names.len() != schema.len()
            || schema.iter().any(|c| c.name.trim().is_empty())
        {
            return Err(TemporalRefusal::invalid_spec(
                "every coordinate is declared exactly once with a unique non-empty name",
            ));
        }
        if slotted != nodes || slotted.len() != variables.len() {
            return Err(TemporalRefusal::invalid_spec(
                "the slots must place every coordinate of the unrolled diagram exactly once",
            ));
        }
        let spec = Self { horizon, slots, diagram, schema: schema.into() };
        spec.check_domains()?;
        spec.check_time_order()?;
        Ok(spec)
    }

    fn check_domains(&self) -> Result<(), TemporalRefusal> {
        let levels = |v: VariableId| {
            self.coordinate(v).and_then(|c| domain_levels(&c.domain)).ok_or_else(|| {
                TemporalRefusal::invalid_spec(format!(
                    "coordinate {} needs a finite binary or categorical domain",
                    v.raw()
                ))
            })
        };
        let first = levels(self.slots.actions[0])?;
        for action in &self.slots.actions[1..] {
            if levels(*action)? != first {
                return Err(TemporalRefusal::invalid_spec(
                    "every step uses one action alphabet: the action domains must be equal",
                ));
            }
        }
        for v in self.slots.history_coordinates() {
            levels(v)?;
        }
        levels(self.slots.outcome)?;
        Ok(())
    }

    fn check_time_order(&self) -> Result<(), TemporalRefusal> {
        let graph = self.diagram.causal_graph();
        for (index, node) in graph.nodes().iter().enumerate() {
            let NodeRef::Static(to) = node else {
                return Err(TemporalRefusal::invalid_spec("unrolled coordinates are static"));
            };
            let id = DenseNodeId::try_from_usize(index)
                .map_err(|e| TemporalRefusal::invalid_spec(e.to_string()))?;
            for parent in graph.parents(id) {
                let NodeRef::Static(from) = graph.nodes()[parent.as_usize()] else {
                    return Err(TemporalRefusal::invalid_spec("unrolled coordinates are static"));
                };
                if self.slots.position(from) > self.slots.position(*to) {
                    return Err(TemporalRefusal::invalid_spec(format!(
                        "edge {} -> {} runs backward in time",
                        from.raw(),
                        to.raw()
                    )));
                }
            }
        }
        if self.diagram.selection_targets().iter().any(|v| self.slots.actions.contains(v)) {
            return Err(TemporalRefusal::invalid_spec(
                "an action is set by the sequence; its mechanism cannot be a selection target",
            ));
        }
        Ok(())
    }

    /// The horizon: the number of steps.
    #[must_use]
    pub const fn horizon(&self) -> usize {
        self.horizon
    }

    /// The coordinates' places in time.
    #[must_use]
    pub const fn slots(&self) -> &TemporalSlots {
        &self.slots
    }

    /// The unrolled selection diagram.
    #[must_use]
    pub const fn diagram(&self) -> &SelectionDiagram {
        &self.diagram
    }

    /// The declared coordinates, in variable order.
    #[must_use]
    pub fn schema(&self) -> &[ScenarioCoordinate] {
        &self.schema
    }

    /// The declared coordinate of `variable`.
    #[must_use]
    pub fn coordinate(&self, variable: VariableId) -> Option<&ScenarioCoordinate> {
        self.schema.iter().find(|c| c.variable == variable)
    }

    /// The finite levels of `variable`.
    #[must_use]
    pub fn levels(&self, variable: VariableId) -> Vec<Value> {
        self.coordinate(variable).and_then(|c| domain_levels(&c.domain)).unwrap_or_default()
    }

    /// The action alphabet shared by every step.
    #[must_use]
    pub fn alphabet(&self) -> Vec<Value> {
        self.levels(self.slots.actions[0])
    }

    /// Time slice of each selection target: the time-indexed mechanism differences.
    #[must_use]
    pub fn selection_slices(&self) -> Vec<(VariableId, u8)> {
        self.diagram
            .selection_targets()
            .iter()
            .filter_map(|v| self.slots.slice(*v).map(|s| (*v, s)))
            .collect()
    }

    /// Validate `sequence` as one action per step over the alphabet.
    ///
    /// # Errors
    /// `temporal_transport.horizon` for a sequence longer than the horizon (a
    /// new period); `temporal_transport.invalid_sequence` for a shorter one or an
    /// action outside the alphabet.
    pub fn check_sequence(&self, sequence: &[Value]) -> Result<(), TemporalRefusal> {
        use std::cmp::Ordering;
        match sequence.len().cmp(&self.horizon) {
            Ordering::Greater => {
                return Err(TemporalRefusal::horizon(format!(
                    "a sequence of {} actions asks for a period beyond the horizon of {}",
                    sequence.len(),
                    self.horizon
                )));
            }
            Ordering::Less => {
                return Err(TemporalRefusal::invalid_sequence(format!(
                    "a sequence needs one action per step ({}), got {}",
                    self.horizon,
                    sequence.len()
                )));
            }
            Ordering::Equal => {}
        }
        let alphabet = self.alphabet();
        for (step, action) in sequence.iter().enumerate() {
            if !alphabet.iter().any(|a| a.as_f64() == action.as_f64()) {
                return Err(TemporalRefusal::invalid_sequence(format!(
                    "action {action:?} at step {} is outside the declared alphabet",
                    step + 1
                )));
            }
        }
        Ok(())
    }

    /// The whole sequence as one classical transport question: the outcome under
    /// the joint intervention on both actions.
    #[must_use]
    pub fn question(&self, source: &str, target: &str) -> ClassicalTransportQuery {
        ClassicalTransportQuery {
            outcomes: Arc::from([self.slots.outcome]),
            treatments: Arc::from(self.slots.actions.as_slice()),
            source: Arc::from(source),
            target: Arc::from(target),
        }
    }

    /// Check a catalog's coordinates against the unrolled window.
    ///
    /// # Errors
    /// `temporal_transport.invalid_spec` for a regime or environment naming a
    /// coordinate outside the window.
    pub fn check_catalog(&self, catalog: &EvidenceCatalog) -> Result<(), TemporalRefusal> {
        let known = |v: &VariableId| self.coordinate(*v).is_some();
        let bad = catalog.regimes.iter().any(|r| {
            !r.interventions.iter().all(known)
                || !r.measured.iter().all(known)
                || !r.conditioned_on.iter().all(known)
        }) || catalog
            .environments
            .iter()
            .any(|e| !e.variables.iter().all(|v| known(&v.variable)));
        if bad {
            return Err(TemporalRefusal::invalid_spec(
                "the catalog names a coordinate outside the unrolled window",
            ));
        }
        Ok(())
    }
}

/// Which template variable plays which role in a two-slice unrolling.
#[derive(Clone, Debug)]
pub struct TemplateRoles {
    /// Time-varying covariates, observed before the period's action.
    pub covariates: Vec<VariableId>,
    /// The action variable.
    pub action: VariableId,
    /// The outcome variable, observed after the period's action.
    pub outcome: VariableId,
}

/// An explicit finite unrolling of a temporal template.
#[derive(Clone, Debug)]
pub struct UnrolledSequence {
    /// The unrolled graph: directed edges from the template; add latent
    /// (bidirected) edges here before building the specification.
    pub admg: Admg,
    /// Where each unrolled coordinate sits in time.
    pub slots: TemporalSlots,
    variable_count: u32,
}

impl UnrolledSequence {
    /// The unrolled coordinate of template `variable` in `period` (1 or 2), or the baseline
    /// coordinate of baseline index `variable` for `period` 0. `None` when the period is not
    /// 0, 1 or 2, or the variable is not a template variable (baseline index) of the unrolling.
    #[must_use]
    pub fn coordinate(&self, variable: VariableId, period: usize) -> Option<VariableId> {
        match period {
            0 => self.slots.baseline.get(usize::try_from(variable.raw()).ok()?).copied(),
            1 | 2 => {
                if variable.raw() >= self.variable_count {
                    return None;
                }
                let offset = u32::try_from(period - 1).ok()?.checked_mul(self.variable_count)?;
                offset.checked_add(variable.raw()).map(VariableId::from_raw)
            }
            _ => None,
        }
    }
}

/// Unroll a lagged temporal template over two slices, reusing the
/// finite unfolding of `TemporalDag` (ADR 0021): template edges are replicated
/// per slice and parents before the first slice are dropped, so the initial
/// state is fixed rather than modelled. Template variables must be numbered
/// `0..n` and include every covariate, the action and the outcome.
///
/// The unrolled coordinate of template variable `v` in period `p` (1 or 2) is
/// `(p - 1) * n + v`, and baseline covariate `i` is `2 * n + i`. The outcome of
/// period 1 is an ordinary covariate of step 2. Each `(i, v)` of
/// `baseline_effects` adds the edge from baseline `i` to `v` in both periods.
///
/// # Errors
/// `temporal_transport.invalid_spec` for a template variable outside `0..n`, a
/// role missing from the template, or a baseline effect naming an unknown variable.
pub fn unroll_two_slice(
    template: &TemporalDag,
    roles: &TemplateRoles,
    baseline: usize,
    baseline_effects: &[(usize, VariableId)],
) -> Result<UnrolledSequence, TemporalRefusal> {
    let n = u32::try_from(roles.covariates.len() + 2)
        .map_err(|_| TemporalRefusal::bounds_exceeded("too many template variables"))?;
    let invalid = |message: &str| TemporalRefusal::invalid_spec(message.to_owned());
    let mut roled = roles.covariates.clone();
    roled.extend([roles.action, roles.outcome]);
    let unique = roled.iter().copied().collect::<BTreeSet<_>>();
    if unique.len() != roled.len() || roled.iter().any(|v| v.raw() >= n) {
        return Err(invalid("template roles must be distinct variables numbered 0..n"));
    }
    let indexer = antecedent_core::TemporalIndexer::new(n, 0, 2)
        .map_err(|e| TemporalRefusal::invalid_spec(e.to_string()))?;
    let unfolded = template.unfold(indexer).map_err(|error| match error {
        antecedent_graph::GraphError::TemporalMaterialization(
            antecedent_core::TemporalMaterializationError::Cancelled
            | antecedent_core::TemporalMaterializationError::Admission(
                antecedent_core::execution::AdmissionError::Cancelled,
            ),
        ) => TemporalRefusal::new(
            reason_code!("cancelled_no_claim"),
            TEMPORAL_BOUNDS_EXCEEDED,
            "temporal unfolding cancelled",
        ),
        antecedent_graph::GraphError::TemporalMaterialization(error) => TemporalRefusal::new(
            reason_code!("transport_budget_cancel"),
            TEMPORAL_BOUNDS_EXCEEDED,
            error.to_string(),
        ),
        other => TemporalRefusal::invalid_spec(other.to_string()),
    })?;
    let total = u32::try_from(baseline)
        .ok()
        .and_then(|b| b.checked_add(2 * n))
        .ok_or_else(|| TemporalRefusal::bounds_exceeded("too many unrolled coordinates"))?;
    let mut admg = Admg::with_variables(total);
    for edge in unfolded.dag.edges() {
        let (from, to) =
            edge.parent_child().ok_or_else(|| invalid("template edge is not directed"))?;
        admg.insert_directed(from, to).map_err(|e| TemporalRefusal::invalid_spec(e.to_string()))?;
    }
    for (i, effect) in baseline_effects {
        let (Ok(i), true) = (u32::try_from(*i), *i < baseline && effect.raw() < n) else {
            return Err(invalid("a baseline effect names an unknown baseline or variable"));
        };
        for period in 0..2 {
            admg.insert_directed(
                DenseNodeId::from_raw(2 * n + i),
                DenseNodeId::from_raw(period * n + effect.raw()),
            )
            .map_err(|e| TemporalRefusal::invalid_spec(e.to_string()))?;
        }
    }
    let at = |period: u32, v: VariableId| VariableId::from_raw(period * n + v.raw());
    let mut second = roles.covariates.iter().map(|v| at(1, *v)).collect::<Vec<_>>();
    second.push(at(0, roles.outcome));
    let slots = TemporalSlots {
        baseline: (0..total - 2 * n).map(|i| VariableId::from_raw(2 * n + i)).collect(),
        covariates: vec![roles.covariates.iter().map(|v| at(0, *v)).collect(), second],
        actions: vec![at(0, roles.action), at(1, roles.action)],
        outcome: at(1, roles.outcome),
    };
    Ok(UnrolledSequence { admg, slots, variable_count: n })
}

/// One coordinate's mechanism assumption at its time slice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MechanismAssumption {
    /// No selection target: the source and target mechanisms are assumed equal.
    Invariant,
    /// A selection target: the mechanism may differ, so it is never borrowed as invariant.
    DiffersBySelection,
}

impl MechanismAssumption {
    /// Stable name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Invariant => "invariant",
            Self::DiffersBySelection => "differs_by_selection",
        }
    }
}

/// The invariance assumed at one coordinate of one time slice, and whether the
/// derived formula relies on it (reads a source factor for that coordinate).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SliceInvariance {
    /// Time slice: 0 baseline, 1 first step, 2 second step.
    pub slice: u8,
    /// The coordinate.
    pub variable: VariableId,
    /// The assumption made about its mechanism.
    pub assumption: MechanismAssumption,
    /// Whether the derived formula reads a source factor for it.
    pub borrowed_from_source: bool,
}

/// One catalog regime placed at the time slices it touches.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SliceEvidence {
    /// Time slice.
    pub slice: u8,
    /// Population that supplied the regime.
    pub population: Arc<str>,
    /// Regime.
    pub regime: RegimeId,
    /// Coordinates the regime intervenes on in this slice.
    pub interventions: Vec<VariableId>,
    /// Coordinates the regime measures in this slice.
    pub measured: Vec<VariableId>,
    /// Whether the identified derivation cites this regime (one of its bound
    /// leaves reads it). A regime the catalog offers but no leaf reads is not
    /// evidence the sequence needs.
    pub cited_by_derivation: bool,
}

/// Complete covariate histories, grown step by step under the shared budget.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HistoryLattice {
    /// Initial states over baseline and step-1 covariates.
    pub initial: Vec<Vec<Value>>,
    /// Complete histories over baseline, step-1 and step-2 covariates.
    pub complete: Vec<Vec<Value>>,
}

/// How the sequence was decided. Only [`Self::StructurallyUnidentified`] is an
/// impossibility claim.
#[derive(Clone, Debug)]
pub enum TemporalOutcome {
    /// A checked derivation of the whole sequence whose leaves bind to the catalog.
    Identified(Box<BoundTransportFunctional>),
    /// An independently verified s-hedge on the unrolled diagram.
    StructurallyUnidentified(Box<SHedgeRecord>),
    /// A derivation exists but the catalog lacks a factor it needs.
    MissingEvidence {
        /// Unmet obligations, per strategy.
        obligations: Arc<[Arc<str>]>,
    },
    /// The bounded search certified nothing; not an impossibility claim.
    NotCertified {
        /// Scope notes, per strategy.
        obligations: Arc<[Arc<str>]>,
    },
    /// A search bound or cancellation stopped the decision.
    Stopped {
        /// The bound that stopped it.
        stop: SearchStop,
    },
}

impl TemporalOutcome {
    /// Identification status in the shared transport vocabulary
    /// ([`TransportOutcomeKind::IDENTIFICATION`](antecedent_core::TransportOutcomeKind::IDENTIFICATION)).
    ///
    /// [`Self::status`] keeps the spellings `structurally_unidentified` and `stopped`; they read as `proven_non_transportable` and `budget_cancel`.
    #[must_use]
    pub const fn identification_status(&self) -> antecedent_core::TransportOutcomeKind {
        match self {
            Self::Identified(_) => antecedent_core::TransportOutcomeKind::Identified,
            Self::StructurallyUnidentified(_) => {
                antecedent_core::TransportOutcomeKind::ProvenNonTransportable
            }
            Self::MissingEvidence { .. } => antecedent_core::TransportOutcomeKind::MissingEvidence,
            Self::NotCertified { .. } => antecedent_core::TransportOutcomeKind::NotCertified,
            Self::Stopped { .. } => antecedent_core::TransportOutcomeKind::BudgetCancel,
        }
    }

    /// Stable status name.
    #[must_use]
    pub const fn status(&self) -> &'static str {
        match self {
            Self::Identified(_) => "identified",
            Self::StructurallyUnidentified(_) => "structurally_unidentified",
            Self::MissingEvidence { .. } => "missing_evidence",
            Self::NotCertified { .. } => "not_certified",
            Self::Stopped { .. } => "stopped",
        }
    }
}

/// The limits a decision ran under, recorded so a consumer replays under exactly
/// the producer's limits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TemporalDecisionLimits {
    /// The one shared budget: identification, verification replays and history growth.
    pub budget: SearchLimits,
    /// The context's hard memory limit during the decision.
    pub memory_limit_bytes: Option<u64>,
}

/// The decision for one two-step sequence.
#[derive(Clone, Debug)]
pub struct TemporalSequenceDecision {
    /// The validated unrolling.
    pub spec: TemporalSequenceSpec,
    /// The action of each step, in order.
    pub sequence: Arc<[Value]>,
    /// The whole sequence as one classical question.
    pub query: ClassicalTransportQuery,
    /// How it was decided.
    pub outcome: TemporalOutcome,
    /// The history lattice (complete unless the decision stopped).
    pub histories: HistoryLattice,
    /// Every coordinate's mechanism assumption per time slice.
    pub invariances: Vec<SliceInvariance>,
    /// Time-varying confounders of the second action.
    pub time_varying_confounders: Vec<VariableId>,
    /// Which catalog evidence touches each time slice.
    pub evidence: Vec<SliceEvidence>,
    /// Present when the shared budget or cancellation stopped the decision.
    pub receipt: Option<SearchReceipt>,
    /// The limits in force.
    pub limits: TemporalDecisionLimits,
}

fn dense(diagram: &SelectionDiagram, variable: VariableId) -> Option<DenseNodeId> {
    diagram.causal_graph().dense_of_static(variable)
}

/// Whether a directed path from `from` reaches `to` without passing through
/// `avoid`: a cause of `to` that is not merely mediated by `avoid`.
fn reaches_avoiding(graph: &Admg, from: DenseNodeId, to: DenseNodeId, avoid: DenseNodeId) -> bool {
    let n = graph.node_count();
    let (mut seeds, mut cut) = (BitSet::with_len(n), BitSet::with_len(n));
    seeds.insert(from);
    cut.insert(avoid);
    let mut ws = antecedent_graph::GraphWorkspace::default();
    graph.descendants_within(&seeds, &BitSet::full(n), Some(&cut), &mut ws).contains(to)
}

/// Covariates of every step after the first that confound that step's action
/// and the outcome while lying downstream of the first action: the time-varying
/// confounders that a sequence-wise identification must handle jointly. A
/// covariate whose only route to the outcome runs through the action it
/// precedes mediates it and confounds nothing.
fn time_varying_confounders(spec: &TemporalSequenceSpec) -> Vec<VariableId> {
    let graph = spec.diagram().causal_graph();
    let slots = spec.slots();
    let (Some(first), Some(y)) =
        (dense(spec.diagram(), slots.actions[0]), dense(spec.diagram(), slots.outcome))
    else {
        return Vec::new();
    };
    let mut confounders = Vec::new();
    for (step, group) in slots.covariates.iter().enumerate().skip(1) {
        let Some(action) = dense(spec.diagram(), slots.actions[step]) else { continue };
        confounders.extend(group.iter().copied().filter(|l| {
            let Some(id) = dense(spec.diagram(), *l) else { return false };
            let after_first = graph.reaches(first, id);
            let affects_action = graph.parents(action).contains(&id)
                || graph.bidirected_neighbors(action).contains(&id);
            let affects_outcome = reaches_avoiding(graph, id, y, action)
                || graph.bidirected_neighbors(y).contains(&id);
            after_first && affects_action && affects_outcome
        }));
    }
    confounders
}

fn invariances(
    spec: &TemporalSequenceSpec,
    query: &ClassicalTransportQuery,
    outcome: &TemporalOutcome,
) -> Vec<SliceInvariance> {
    let borrowed = match outcome {
        TemporalOutcome::Identified(bound) => bound
            .leaf_factors()
            .into_iter()
            .filter(|(_, leaf)| leaf.binding.population == query.source)
            .flat_map(|(_, leaf)| leaf.variables.to_vec())
            .collect::<BTreeSet<_>>(),
        _ => BTreeSet::new(),
    };
    let slots = spec.slots();
    slots
        .variables()
        .into_iter()
        .filter(|v| !slots.actions.contains(v))
        .filter_map(|v| {
            Some(SliceInvariance {
                slice: slots.slice(v)?,
                variable: v,
                assumption: if spec.diagram().mechanism_may_differ(v) {
                    MechanismAssumption::DiffersBySelection
                } else {
                    MechanismAssumption::Invariant
                },
                borrowed_from_source: borrowed.contains(&v),
            })
        })
        .collect()
}

fn evidence_by_slice(
    spec: &TemporalSequenceSpec,
    catalog: &EvidenceCatalog,
    outcome: &TemporalOutcome,
) -> Vec<SliceEvidence> {
    let slots = spec.slots();
    // The regimes the derivation reads, from its own bound leaves.
    let cited = match outcome {
        TemporalOutcome::Identified(bound) => bound
            .leaf_factors()
            .into_iter()
            .filter_map(|(_, leaf)| leaf.binding.regime.map(|r| (leaf.binding.population, r)))
            .collect::<BTreeSet<_>>(),
        _ => BTreeSet::new(),
    };
    let mut rows = Vec::new();
    for regime in catalog.regimes.iter() {
        for slice in 0..=u8::try_from(slots.horizon()).unwrap_or(u8::MAX) {
            let in_slice = |v: &&VariableId| slots.slice(**v) == Some(slice);
            let interventions =
                regime.interventions.iter().filter(in_slice).copied().collect::<Vec<_>>();
            let measured = regime.measured.iter().filter(in_slice).copied().collect::<Vec<_>>();
            if !interventions.is_empty() || !measured.is_empty() {
                rows.push(SliceEvidence {
                    slice,
                    population: regime.population.clone(),
                    regime: regime.id,
                    interventions,
                    measured,
                    cited_by_derivation: cited.contains(&(regime.population.clone(), regime.id)),
                });
            }
        }
    }
    rows.sort_by(|a, b| {
        (a.slice, &a.population, a.regime).cmp(&(b.slice, &b.population, b.regime))
    });
    rows
}

/// Live bytes of `states` retained histories of `width` covariates.
fn lattice_bytes(base: u64, states: usize, width: usize) -> u64 {
    let per = width.saturating_mul(std::mem::size_of::<Value>()).saturating_add(24);
    base.saturating_add(u64::try_from(per.saturating_mul(states)).unwrap_or(u64::MAX))
}

/// Grow the history lattice one step at a time, charging the shared budget once
/// per state with the cumulative bytes retained. Step 1 enumerates the initial
/// states at depth one; step `k` extends each state with the step-`k`
/// covariates at depth `k`, up to the horizon. The error names the step that
/// stopped.
fn grow_histories(
    spec: &TemporalSequenceSpec,
    base: u64,
    search: &mut SharedSearch<'_>,
) -> Result<HistoryLattice, (SearchStop, usize)> {
    let slots = spec.slots();
    let extend = |states: Vec<Vec<Value>>,
                  coordinates: &[VariableId],
                  depth: usize,
                  step: usize,
                  retained: usize,
                  width: usize,
                  search: &mut SharedSearch<'_>|
     -> Result<Vec<Vec<Value>>, (SearchStop, usize)> {
        let mut grown = Vec::new();
        let levels = coordinates.iter().map(|v| spec.levels(*v)).collect::<Vec<_>>();
        if levels.iter().any(Vec::is_empty) {
            return Ok(grown);
        }
        for state in &states {
            // Mixed-radix walk over the suffixes: each state is charged before it is
            // built, so the product of levels is never materialized ahead of the budget.
            let mut digits = vec![0usize; levels.len()];
            loop {
                search
                    .charge(depth, lattice_bytes(base, retained + grown.len() + 1, width))
                    .map_err(|stop| (stop, step))?;
                let mut full = state.clone();
                full.extend(digits.iter().zip(&levels).map(|(d, l)| l[*d].clone()));
                grown.push(full);
                let mut place = digits.len();
                loop {
                    if place == 0 {
                        break;
                    }
                    place -= 1;
                    digits[place] += 1;
                    if digits[place] < levels[place].len() {
                        place = usize::MAX;
                        break;
                    }
                    digits[place] = 0;
                }
                if place != usize::MAX {
                    break;
                }
            }
        }
        Ok(grown)
    };
    let initial_coordinates = slots.initial_state();
    let initial =
        extend(vec![Vec::new()], &initial_coordinates, 1, 1, 0, initial_coordinates.len(), search)?;
    let mut complete = initial.clone();
    let mut width = initial_coordinates.len();
    for (index, group) in slots.covariates.iter().enumerate().skip(1) {
        let step = index + 1;
        width += group.len();
        let retained = complete.len();
        complete = extend(complete, group, step, step, retained, width, search)?;
    }
    Ok(HistoryLattice { initial, complete })
}

/// The catalog as decided: identification first, then history growth, on one
/// shared budget.
fn identify(
    spec: &TemporalSequenceSpec,
    query: &ClassicalTransportQuery,
    catalog: &EvidenceCatalog,
    search: &mut SharedSearch<'_>,
    ctx: &ExecutionContext,
) -> Result<TemporalOutcome, IdentificationError> {
    let diagram = spec.diagram();
    Ok(match identify_catalog_transport_metered(diagram, query, catalog, search.meter(), ctx)? {
        CatalogTransportResult::Identified(bound) => TemporalOutcome::Identified(bound),
        CatalogTransportResult::MissingEvidence { obligations, .. } => {
            TemporalOutcome::MissingEvidence { obligations }
        }
        CatalogTransportResult::NotCertified { obligations, .. } => {
            match identify_classical_transport_metered(diagram, query, search.meter(), ctx)? {
                ClassicalTransportResult::ProvenNonTransportable(hedge) => {
                    TemporalOutcome::StructurallyUnidentified(Box::new(hedge.to_record()))
                }
                _ => TemporalOutcome::NotCertified { obligations },
            }
        }
    })
}

/// Decide the whole sequence as one longitudinal intervention.
///
/// The treatment set is every action at once: the question asks for the outcome
/// under `do(A_1 = a_1, ..., A_H = a_H)` on the unrolled selection diagram, decided
/// by the catalog-aware classical route (and, only when it certifies nothing, the
/// classical identifier for an s-hedge). The steps are never transported
/// separately and multiplied. One [`SearchBudget`] bounds the decision: the
/// identification search, its verification replays, and then the growth of the
/// history lattice, whose every state is charged with the live bytes retained.
/// A stop leaves a receipt in the decision and outcome
/// [`TemporalOutcome::Stopped`]; it never claims non-identification.
///
/// # Errors
/// A refused sequence or catalog ([`TemporalSequenceError::Refused`]) or an
/// invalid query or catalog ([`TemporalSequenceError::Identification`]).
pub fn decide_temporal_transport_sequence(
    spec: &TemporalSequenceSpec,
    sequence: &[Value],
    source: &str,
    target: &str,
    catalog: &EvidenceCatalog,
    budget: SearchLimits,
    ctx: &ExecutionContext,
) -> Result<TemporalSequenceDecision, TemporalSequenceError> {
    decide_inner(
        spec,
        sequence,
        source,
        target,
        catalog,
        budget,
        ctx,
        #[cfg(test)]
        None,
    )
}

#[cfg_attr(
    test,
    allow(
        clippy::too_many_arguments,
        reason = "the test build adds the mid-search cancellation hook; the production signature has seven"
    )
)]
fn decide_inner(
    spec: &TemporalSequenceSpec,
    sequence: &[Value],
    source: &str,
    target: &str,
    catalog: &EvidenceCatalog,
    budget: SearchLimits,
    ctx: &ExecutionContext,
    #[cfg(test)] cancel_after: Option<(usize, antecedent_core::CancellationToken)>,
) -> Result<TemporalSequenceDecision, TemporalSequenceError> {
    spec.check_sequence(sequence)?;
    spec.check_catalog(catalog)?;
    catalog.validate().map_err(|e| IdentificationError::invalid_catalog(e.to_string()))?;
    let query = spec.question(source, target);
    let limits = TemporalDecisionLimits { budget, memory_limit_bytes: ctx.memory_limit_bytes() };
    let mut receipt = None;
    let mut histories = HistoryLattice::default();
    let outcome: TemporalOutcome;
    let stages = || -> Vec<String> {
        std::iter::once("identification".to_owned())
            .chain((1..=spec.horizon()).map(|step| format!("history_step_{step}")))
            .collect()
    };
    let pending = |from: usize| -> Vec<String> { stages()[from..].to_vec() };
    let finished = |upto: usize| -> Vec<String> { stages()[..upto].to_vec() };
    match SearchBudget::new(budget, ctx) {
        Err(mut stopped) => {
            stopped.unevaluated = pending(0);
            outcome = TemporalOutcome::Stopped { stop: stopped.stop };
            receipt = Some(stopped);
        }
        Ok(budget) => {
            let mut search = SharedSearch::new(budget);
            #[cfg(test)]
            {
                search.cancel_after = cancel_after;
            }
            match identify(spec, &query, catalog, &mut search, ctx) {
                Ok(found) => {
                    let base = sid_memory_bytes(spec.diagram(), 1).try_into().unwrap_or(u64::MAX);
                    match grow_histories(spec, base, &mut search) {
                        Ok(lattice) => {
                            histories = lattice;
                            outcome = found;
                        }
                        Err((stop, step)) => {
                            // Identification finished, and so did every history step
                            // before the one that stopped.
                            let from = step;
                            receipt = Some(search.receipt(stop, finished(from), pending(from)));
                            outcome = TemporalOutcome::Stopped { stop };
                        }
                    }
                }
                Err(error) if error.is_budget_or_cancel() => {
                    let stop = search.stop_of(&error);
                    receipt = Some(search.receipt(stop, Vec::new(), pending(0)));
                    outcome = TemporalOutcome::Stopped { stop };
                }
                Err(error) => return Err(error.into()),
            }
        }
    }
    Ok(TemporalSequenceDecision {
        spec: spec.clone(),
        sequence: sequence.into(),
        invariances: invariances(spec, &query, &outcome),
        time_varying_confounders: time_varying_confounders(spec),
        evidence: evidence_by_slice(spec, catalog, &outcome),
        query,
        outcome,
        histories,
        receipt,
        limits,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{
        DependenceGroup, DistributionAvailability, EvidenceKind, EvidenceRegime, RegimeBinding,
        RegimeKind, SamplingDesign,
    };

    use crate::test_support::v;

    /// `b -> l1 -> a1 -> l2 -> a2 -> y`, no selection: four initial states and
    /// eight complete histories.
    fn chain() -> (TemporalSequenceSpec, EvidenceCatalog) {
        let mut graph = Admg::with_variables(6);
        for i in 0..5 {
            graph.insert_directed(DenseNodeId::from_raw(i), DenseNodeId::from_raw(i + 1)).unwrap();
        }
        let slots = TemporalSlots {
            baseline: vec![v(0)],
            covariates: vec![vec![v(1)], vec![v(3)]],
            actions: vec![v(2), v(4)],
            outcome: v(5),
        };
        let coordinates = (0..6)
            .map(|i| ScenarioCoordinate {
                variable: v(i),
                name: Arc::from(format!("v{i}")),
                domain: VariableDomain::Binary,
                unit: None,
            })
            .collect::<Vec<_>>();
        let diagram = SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from([])).unwrap();
        let spec = TemporalSequenceSpec::try_new(2, slots, diagram, coordinates).unwrap();
        let regime = EvidenceRegime::try_new(
            RegimeId::from_raw(0),
            RegimeKind::Observational,
            EvidenceKind::Available,
            [],
            [],
            (0..6).map(v).collect::<Vec<_>>(),
            "target",
            DistributionAvailability::Joint,
        )
        .unwrap();
        let binding = RegimeBinding {
            dataset_identity: None,
            regime: regime.id,
            snapshot_identity: Arc::from("target-0"),
            schema_names: Arc::from([]),
            sampling: SamplingDesign::Independent,
            weights: None,
            dependence: DependenceGroup::IndependentStudies,
        };
        (spec, EvidenceCatalog::try_new([], vec![regime], vec![binding], None).unwrap())
    }

    fn decide_with_cancel(
        operations: usize,
        cancel_after: Option<usize>,
    ) -> TemporalSequenceDecision {
        let (spec, catalog) = chain();
        let ctx = ExecutionContext::for_tests(1);
        let hook = cancel_after.map(|after| (after, ctx.cancellation.clone()));
        decide_inner(
            &spec,
            &[Value::f64(1.0), Value::f64(0.0)],
            "source",
            "target",
            &catalog,
            SearchLimits { operations, depth: 256 },
            &ctx,
            hook,
        )
        .unwrap()
    }

    #[test]
    fn cancellation_stops_inside_the_history_steps_with_a_receipt_of_what_finished() {
        let complete = (1..5_000)
            .find(|ops| decide_with_cancel(*ops, None).receipt.is_none())
            .expect("the decision completes");
        // Four initial states and eight complete histories are charged after identification.
        let identified = complete - 12;
        let inside_step_1 = decide_with_cancel(1_000, Some(identified));
        assert!(matches!(
            inside_step_1.outcome,
            TemporalOutcome::Stopped { stop: SearchStop::Cancelled }
        ));
        let receipt = inside_step_1.receipt.expect("a stop leaves a receipt");
        assert_eq!(receipt.explored, ["identification"]);
        assert_eq!(receipt.unevaluated, ["history_step_1", "history_step_2"]);
        assert!(inside_step_1.histories.initial.is_empty());
        // Cancelled after step 1's four states: step 1 finished, step 2 did not.
        let inside_step_2 = decide_with_cancel(1_000, Some(identified + 4));
        assert!(matches!(
            inside_step_2.outcome,
            TemporalOutcome::Stopped { stop: SearchStop::Cancelled }
        ));
        let receipt = inside_step_2.receipt.expect("a stop leaves a receipt");
        assert_eq!(receipt.explored, ["identification", "history_step_1"]);
        assert_eq!(receipt.unevaluated, ["history_step_2"]);
        // A stop is never a verdict: nothing is claimed identified or refuted.
        assert!(inside_step_2.outcome.status() == "stopped");
    }
}
