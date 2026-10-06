//! One finite two-step temporal transport sequence (2.2A, X5): exact and point-only.
//!
//! The whole sequence `do(A_1 = a_1, A_2 = a_2)` is one longitudinal intervention
//! on the explicit two-slice unrolled selection diagram. It is decided by the
//! licensed classical catalog sID route in one call with both actions as the
//! treatment set; time step 1 and time step 2 are never transported separately
//! and multiplied. The unrolled diagram fixes the finite discrete structure:
//! baseline covariates, the time-varying covariates observed before each action,
//! one action per step over one shared discrete alphabet, and one outcome after
//! the last action. A selection target on a coordinate is a time-indexed
//! mechanism difference between source and target at that coordinate's slice;
//! any coordinate without one is assumed invariant at its slice, and that
//! assumption is recorded per slice. Only the initial-state law (baseline and
//! step-1 covariates) is fixed per population, not uncertain: temporal sampling
//! intervals, initial-state uncertainty and new-period refresh are 2.3A.
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
use antecedent_graph::{Admg, DenseNodeId, SelectionDiagram, TemporalDag};

use super::scenarios::ScenarioCoordinate;
use super::{
    BoundTransportFunctional, CatalogTransportResult, ClassicalTransportQuery,
    ClassicalTransportResult, IdentificationError, SHedgeRecord, SearchCharge, SharedSearch,
    identify_catalog_transport_metered, identify_classical_transport_metered, sid_memory_bytes,
};

/// The one licensed horizon: two steps.
pub const TEMPORAL_HORIZON: usize = 2;
/// Most actions in the discrete alphabet.
pub const TEMPORAL_MAX_ACTIONS: usize = 8;
/// Most coordinates in the unrolled diagram.
pub const TEMPORAL_MAX_OBSERVED: usize = 12;
/// Most complete covariate histories (baseline, step-1 and step-2 covariates).
pub const TEMPORAL_MAX_HISTORY_STATES: usize = 4096;

/// Detail: a required history has no certified support.
pub const TEMPORAL_HISTORY_OUTSIDE_SUPPORT: &str = "temporal_transport.history_outside_support";
/// Detail: a horizon other than two, a new period, or a changed window.
pub const TEMPORAL_HORIZON_DETAIL: &str = "temporal_transport.horizon";
/// Detail: the licensed rules certified nothing.
pub const TEMPORAL_SEARCH_INCOMPLETE: &str = "temporal_transport.search_incomplete";
/// Detail: the shared budget or cancellation stopped the decision.
pub const TEMPORAL_HISTORY_BUDGET: &str = "temporal_transport.history_budget";
/// Detail: an interval was requested.
pub const TEMPORAL_INTERVAL_REQUESTED: &str = "temporal_transport.interval_requested";
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

    /// A horizon other than two, a new period, or a changed measurement window.
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
    /// uncertainty are 2.3A, so this route is point-only.
    #[must_use]
    pub fn interval_requested() -> Self {
        Self::new(
            reason_code!("estimator_inference_mismatch"),
            TEMPORAL_INTERVAL_REQUESTED,
            "the two-step sequence route is exact and point-only; temporal sampling and \
             initial-state intervals are 2.3A",
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
/// Slice 0 is the baseline; slice 1 holds the step-1 covariates and action;
/// slice 2 holds the step-2 covariates, action and the outcome. The order in
/// time is `baseline < covariates[0] < actions[0] < covariates[1] < actions[1]
/// < outcome`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TemporalSlots {
    /// Baseline covariates: the initial state before step 1.
    pub baseline: Vec<VariableId>,
    /// Time-varying covariates observed before each action.
    pub covariates: [Vec<VariableId>; 2],
    /// The action of each step.
    pub actions: [VariableId; 2],
    /// The outcome observed after the last action.
    pub outcome: VariableId,
}

impl TemporalSlots {
    /// Position in time order, or `None` for a variable not in the slots.
    #[must_use]
    pub fn position(&self, variable: VariableId) -> Option<u8> {
        if self.baseline.contains(&variable) {
            Some(0)
        } else if self.covariates[0].contains(&variable) {
            Some(1)
        } else if variable == self.actions[0] {
            Some(2)
        } else if self.covariates[1].contains(&variable) {
            Some(3)
        } else if variable == self.actions[1] {
            Some(4)
        } else if variable == self.outcome {
            Some(5)
        } else {
            None
        }
    }

    /// Time slice of a variable: 0 baseline, 1 first step, 2 second step.
    #[must_use]
    pub fn slice(&self, variable: VariableId) -> Option<u8> {
        self.position(variable).map(|p| match p {
            0 => 0,
            1 | 2 => 1,
            _ => 2,
        })
    }

    /// Every coordinate, in time order.
    #[must_use]
    pub fn variables(&self) -> Vec<VariableId> {
        let mut all = self.baseline.clone();
        all.extend(&self.covariates[0]);
        all.push(self.actions[0]);
        all.extend(&self.covariates[1]);
        all.push(self.actions[1]);
        all.push(self.outcome);
        all
    }

    /// Covariates that make up the initial-state law: baseline and step-1 covariates.
    #[must_use]
    pub fn initial_state(&self) -> Vec<VariableId> {
        self.baseline.iter().chain(&self.covariates[0]).copied().collect()
    }

    /// Covariates of a complete history: the initial state and the step-2 covariates.
    #[must_use]
    pub fn history_coordinates(&self) -> Vec<VariableId> {
        self.initial_state().into_iter().chain(self.covariates[1].iter().copied()).collect()
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

/// A validated explicit unrolling: horizon two, discrete action alphabet, the
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
    /// Validate an explicit two-slice unrolling.
    ///
    /// The declared `horizon` must be two. The slots must partition the
    /// diagram's coordinates; every coordinate is declared once with a unique
    /// name and a finite (binary or categorical) domain; the two actions share
    /// one alphabet of at most [`TEMPORAL_MAX_ACTIONS`] actions; no directed
    /// edge runs backward in time; and no selection target is an action, whose
    /// mechanism the sequence replaces. The unrolled diagram has at most
    /// [`TEMPORAL_MAX_OBSERVED`] coordinates and the complete covariate
    /// histories number at most [`TEMPORAL_MAX_HISTORY_STATES`].
    ///
    /// # Errors
    /// `route_not_supported` / `temporal_transport.horizon` for another horizon,
    /// `route_not_supported` / `temporal_transport.bounds_exceeded` for an
    /// exceeded cap, and `invalid_argument` / `temporal_transport.invalid_spec`
    /// for anything malformed.
    pub fn try_new(
        horizon: usize,
        slots: TemporalSlots,
        diagram: SelectionDiagram,
        coordinates: impl Into<Arc<[ScenarioCoordinate]>>,
    ) -> Result<Self, TemporalRefusal> {
        if horizon != TEMPORAL_HORIZON {
            return Err(TemporalRefusal::horizon(format!(
                "horizon {horizon} is outside the licensed horizon of {TEMPORAL_HORIZON}"
            )));
        }
        let mut schema = coordinates.into().to_vec();
        schema.sort_by_key(|c| c.variable);
        let graph = diagram.causal_graph();
        if graph.node_count() > TEMPORAL_MAX_OBSERVED {
            return Err(TemporalRefusal::bounds_exceeded(format!(
                "{} coordinates exceed the bound of {TEMPORAL_MAX_OBSERVED}",
                graph.node_count()
            )));
        }
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
        let second = levels(self.slots.actions[1])?;
        if first != second {
            return Err(TemporalRefusal::invalid_spec(
                "both steps use one action alphabet: the action domains must be equal",
            ));
        }
        if first.len() > TEMPORAL_MAX_ACTIONS {
            return Err(TemporalRefusal::bounds_exceeded(format!(
                "{} actions exceed the alphabet bound of {TEMPORAL_MAX_ACTIONS}",
                first.len()
            )));
        }
        let mut states = 1usize;
        for v in self.slots.history_coordinates() {
            states = states.saturating_mul(levels(v)?.len());
        }
        levels(self.slots.outcome)?;
        if states > TEMPORAL_MAX_HISTORY_STATES {
            return Err(TemporalRefusal::bounds_exceeded(format!(
                "{states} complete covariate histories exceed the bound of \
                 {TEMPORAL_MAX_HISTORY_STATES}"
            )));
        }
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

    /// The horizon (always two).
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

    /// The action alphabet shared by both steps.
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
            treatments: Arc::from(self.slots.actions),
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
    /// The unrolled coordinate of template `variable` in `period` (1 or 2), or
    /// the baseline coordinate `index` for `period` 0.
    #[must_use]
    pub fn coordinate(&self, variable: VariableId, period: usize) -> VariableId {
        let period = u32::try_from(period.saturating_sub(1)).unwrap_or(0);
        VariableId::from_raw(period * self.variable_count + variable.raw())
    }
}

/// Unroll a lagged temporal template over the horizon of two, reusing the
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
    let unfolded =
        template.unfold(indexer).map_err(|e| TemporalRefusal::invalid_spec(e.to_string()))?;
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
        covariates: [roles.covariates.iter().map(|v| at(0, *v)).collect(), second],
        actions: [at(0, roles.action), at(1, roles.action)],
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
    diagram
        .causal_graph()
        .nodes()
        .iter()
        .position(|n| *n == NodeRef::Static(variable))
        .and_then(|i| DenseNodeId::try_from_usize(i).ok())
}

/// Whether a directed path from `from` reaches `to` without passing through
/// `avoid`: a cause of `to` that is not merely mediated by `avoid`.
fn reaches_avoiding(graph: &Admg, from: DenseNodeId, to: DenseNodeId, avoid: DenseNodeId) -> bool {
    let mut seen = vec![false; graph.node_count()];
    let mut stack = vec![from];
    while let Some(node) = stack.pop() {
        if node == to {
            return true;
        }
        if node == avoid || std::mem::replace(&mut seen[node.as_usize()], true) {
            continue;
        }
        stack.extend(graph.children(node).iter().copied());
    }
    false
}

/// Covariates of step 2 that confound the second action and the outcome while
/// lying downstream of the first action: the time-varying confounders that a
/// sequence-wise identification must handle jointly. A covariate whose only
/// route to the outcome runs through the second action mediates it and
/// confounds nothing.
fn time_varying_confounders(spec: &TemporalSequenceSpec) -> Vec<VariableId> {
    let graph = spec.diagram().causal_graph();
    let slots = spec.slots();
    let (Some(a1), Some(a2), Some(y)) = (
        dense(spec.diagram(), slots.actions[0]),
        dense(spec.diagram(), slots.actions[1]),
        dense(spec.diagram(), slots.outcome),
    ) else {
        return Vec::new();
    };
    slots.covariates[1]
        .iter()
        .copied()
        .filter(|l| {
            let Some(id) = dense(spec.diagram(), *l) else { return false };
            let after_first = graph.reaches(a1, id);
            let affects_second =
                graph.parents(a2).contains(&id) || graph.bidirected_neighbors(a2).contains(&id);
            let affects_outcome =
                reaches_avoiding(graph, id, y, a2) || graph.bidirected_neighbors(y).contains(&id);
            after_first && affects_second && affects_outcome
        })
        .collect()
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
        for slice in 0..=2u8 {
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
/// states at depth one; step 2 extends each with the step-2 covariates at depth
/// two, the horizon.
fn grow_histories(
    spec: &TemporalSequenceSpec,
    base: u64,
    search: &mut SharedSearch<'_>,
) -> Result<HistoryLattice, (SearchStop, &'static str)> {
    let slots = spec.slots();
    let extend = |states: Vec<Vec<Value>>,
                  coordinates: &[VariableId],
                  depth: usize,
                  step: &'static str,
                  retained: usize,
                  width: usize,
                  search: &mut SharedSearch<'_>|
     -> Result<Vec<Vec<Value>>, (SearchStop, &'static str)> {
        let mut grown = Vec::new();
        for state in &states {
            let mut suffixes: Vec<Vec<Value>> = vec![Vec::new()];
            for v in coordinates {
                let levels = spec.levels(*v);
                suffixes = suffixes
                    .into_iter()
                    .flat_map(|prefix| {
                        levels.iter().map(move |level| {
                            let mut next = prefix.clone();
                            next.push(level.clone());
                            next
                        })
                    })
                    .collect();
            }
            for suffix in suffixes {
                search
                    .charge(depth, lattice_bytes(base, retained + grown.len() + 1, width))
                    .map_err(|stop| (stop, step))?;
                let mut full = state.clone();
                full.extend(suffix);
                grown.push(full);
            }
        }
        Ok(grown)
    };
    let initial_coordinates = slots.initial_state();
    let initial = extend(
        vec![Vec::new()],
        &initial_coordinates,
        1,
        "history_step_1",
        0,
        initial_coordinates.len(),
        search,
    )?;
    let complete = extend(
        initial.clone(),
        &slots.covariates[1],
        TEMPORAL_HORIZON,
        "history_step_2",
        initial.len(),
        slots.history_coordinates().len(),
        search,
    )?;
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

/// Decide the whole two-step sequence as one longitudinal intervention.
///
/// The treatment set is both actions at once: the question asks for the outcome
/// under `do(A_1 = a_1, A_2 = a_2)` on the unrolled selection diagram, decided by
/// the catalog-aware classical route (and, only when it certifies nothing, the
/// classical identifier for an s-hedge). The two steps are never transported
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

#[allow(clippy::too_many_arguments)] // The public entry point's premises plus a test hook.
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
    let limits = TemporalDecisionLimits { budget, memory_limit_bytes: ctx.memory.hard_limit_bytes };
    let mut receipt = None;
    let mut histories = HistoryLattice::default();
    let outcome: TemporalOutcome;
    let pending = |from: usize| -> Vec<String> {
        ["identification", "history_step_1", "history_step_2"][from..]
            .iter()
            .map(|s| (*s).to_owned())
            .collect()
    };
    let finished = |upto: usize| -> Vec<String> {
        ["identification", "history_step_1", "history_step_2"][..upto]
            .iter()
            .map(|s| (*s).to_owned())
            .collect()
    };
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
                            let from = if step == "history_step_1" { 1 } else { 2 };
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

    fn v(i: u32) -> VariableId {
        VariableId::from_raw(i)
    }

    /// `b -> l1 -> a1 -> l2 -> a2 -> y`, no selection: four initial states and
    /// eight complete histories.
    fn chain() -> (TemporalSequenceSpec, EvidenceCatalog) {
        let mut graph = Admg::with_variables(6);
        for i in 0..5 {
            graph.insert_directed(DenseNodeId::from_raw(i), DenseNodeId::from_raw(i + 1)).unwrap();
        }
        let slots = TemporalSlots {
            baseline: vec![v(0)],
            covariates: [vec![v(1)], vec![v(3)]],
            actions: [v(2), v(4)],
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
