//! Exact point evaluation of one finite two-step temporal transport sequence.
//!
//! The provider is the finite discrete exact evaluator: the whole sequence's
//! identified functional is compiled once against supplied exact laws and
//! evaluated as one history-aware functional over the target initial-state law
//! and every complete history. Nothing is transported per step and multiplied.
//! Support is reported per history and per step, and a history outside certified
//! support refuses rather than being extrapolated. The claim is point-only: no
//! temporal sampling interval, no initial-state uncertainty and no new-period
//! refresh (2.3A).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use antecedent_core::{ExecutionContext, Value, VariableId, reason_code};
use antecedent_expr::{
    Assignment, ExactDiscreteLaw, ExactDistribution, ExactEvaluationLimits, ExactEvaluationPlan,
    ExactTransportData, InterventionAssignment,
};
use antecedent_identify::sid::temporal_sequence::{
    SliceEvidence, SliceInvariance, TEMPORAL_CHECKED_OBSTRUCTION, TEMPORAL_HISTORY_BUDGET,
    TEMPORAL_MISSING_EVIDENCE, TEMPORAL_SEARCH_INCOMPLETE, TemporalOutcome, TemporalRefusal,
    TemporalSequenceDecision,
};

use crate::error::EstimationError;
use crate::transport_scenarios::SCENARIO_EXACT_PROVIDER;

/// Provider identity of supplied exact laws.
pub const TEMPORAL_EXACT_PROVIDER: &str = SCENARIO_EXACT_PROVIDER;
/// Largest change of an initial-state mass a same-window refresh may carry.
pub const INITIAL_STATE_TOLERANCE: f64 = 1e-9;
/// The only inference claim of this route.
pub const TEMPORAL_INFERENCE_CLAIM: &str = "point_only";

/// One history's support at one step.
#[derive(Clone, Debug, PartialEq)]
pub struct HistorySupportRow {
    /// Step: 1 for an initial state, 2 for a complete history.
    pub step: u8,
    /// The covariate values, in the coordinates of the step.
    pub history: Vec<Value>,
    /// Mass of the history (with the first action for step 2) under the target's
    /// observational law; `None` when no supplied target law covers it.
    pub target_mass: Option<f64>,
    /// `supported`, `unreached` (served, but no target mass) or `outside_support`.
    pub status: &'static str,
}

/// Support of every history the sequence's functional can reach, per step.
#[derive(Clone, Debug, PartialEq)]
pub struct HistorySupportReport {
    /// Coordinates of a step-1 row: baseline and step-1 covariates.
    pub initial_coordinates: Vec<VariableId>,
    /// Coordinates of a step-2 row: the initial state and the step-2 covariates.
    pub history_coordinates: Vec<VariableId>,
    /// Whether a target observational law covered the reachability masses.
    pub target_law_used: bool,
    /// Every initial state, then every complete history, in lattice order.
    pub rows: Vec<HistorySupportRow>,
}

impl HistorySupportReport {
    /// Rows outside certified support.
    #[must_use]
    pub fn outside(&self) -> Vec<&HistorySupportRow> {
        self.rows.iter().filter(|r| r.status == "outside_support").collect()
    }
}

fn bits(value: &Value) -> Option<u64> {
    value.as_f64().map(f64::to_bits)
}

fn same(a: &Value, b: &Value) -> bool {
    a.as_f64().is_some() && a.as_f64() == b.as_f64()
}

/// Masses of the target's observational law by history projection, computed in
/// one pass over the cells.
struct TargetMasses {
    initial: HashMap<Vec<u64>, f64>,
    initial_first_action: HashMap<Vec<u64>, f64>,
    history_first_action: HashMap<Vec<u64>, f64>,
}

fn target_law<'a>(
    decision: &TemporalSequenceDecision,
    data: &'a ExactTransportData,
) -> Option<&'a ExactDiscreteLaw> {
    let slots = decision.spec.slots();
    let mut needed = slots.history_coordinates();
    needed.push(slots.actions[0]);
    data.laws().iter().find(|law| {
        law.population() == &*decision.query.target
            && law.interventions().is_empty()
            && needed.iter().all(|v| law.axes().iter().any(|a| a.variable == *v))
    })
}

fn target_masses(decision: &TemporalSequenceDecision, law: &ExactDiscreteLaw) -> TargetMasses {
    let slots = decision.spec.slots();
    let axes = law.axes();
    let position = |v: VariableId| axes.iter().position(|a| a.variable == v);
    let initial = slots.initial_state();
    let complete = slots.history_coordinates();
    let at = |v: &VariableId| position(*v).expect("the law covers every needed coordinate");
    let (initial_at, complete_at) =
        (initial.iter().map(at).collect::<Vec<_>>(), complete.iter().map(at).collect::<Vec<_>>());
    let first = at(&slots.actions[0]);
    let mut strides = vec![1usize; axes.len()];
    for i in (0..axes.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * axes[i + 1].values.len();
    }
    let mut masses = TargetMasses {
        initial: HashMap::new(),
        initial_first_action: HashMap::new(),
        history_first_action: HashMap::new(),
    };
    for (cell, mass) in law.probabilities().iter().enumerate() {
        let level =
            |axis: usize| &axes[axis].values[(cell / strides[axis]) % axes[axis].values.len()];
        let key = |coordinates: &[usize]| {
            coordinates.iter().map(|a| bits(level(*a)).unwrap_or(0)).collect::<Vec<_>>()
        };
        *masses.initial.entry(key(&initial_at)).or_insert(0.0) += mass;
        if same(level(first), &decision.sequence[0]) {
            *masses.initial_first_action.entry(key(&initial_at)).or_insert(0.0) += mass;
            *masses.history_first_action.entry(key(&complete_at)).or_insert(0.0) += mass;
        }
    }
    masses
}

/// What the evaluator reads from the exact laws, derived from the identified
/// proof: one signature per distribution leaf of the bound functional (the
/// population, the catalog regime, the intervention coordinates and the factor
/// variables it cites). Nothing here is assumed about the diagram's shape.
struct Requirements {
    leaves: Vec<antecedent_expr::LeafSignature>,
}

impl Requirements {
    fn of(decision: &TemporalSequenceDecision) -> Option<Self> {
        let TemporalOutcome::Identified(bound) = &decision.outcome else { return None };
        Some(Self { leaves: bound.leaf_factors().into_iter().map(|(_, leaf)| leaf).collect() })
    }
}

/// The world a leaf reads for `history` under the requested sequence: its
/// intervention assignments with every symbolic coordinate resolved from the
/// history or the sequence. `None` when a coordinate cannot be resolved (the
/// history does not determine it), which is never supported.
fn leaf_world(
    decision: &TemporalSequenceDecision,
    coordinates: &[VariableId],
    history: &[Value],
    leaf: &antecedent_expr::LeafSignature,
) -> Option<Vec<InterventionAssignment>> {
    let slots = decision.spec.slots();
    leaf.intervention
        .iter()
        .map(|i| {
            if !i.is_symbolic() {
                return Some(i.clone());
            }
            let value = if let Some(k) = coordinates.iter().position(|c| *c == i.variable) {
                history[k].clone()
            } else {
                let step = slots.actions.iter().position(|a| *a == i.variable)?;
                decision.sequence[step].clone()
            };
            Some(InterventionAssignment::concrete(i.variable, value))
        })
        .collect()
}

/// Whether the supplied laws contain, for every leaf the proof cites, a law the
/// evaluator will select for this history: the leaf's population and regime, the
/// same intervention coordinates at the history's (and the sequence's) values,
/// and axes covering the leaf's variables. A history is supported only when
/// every cited leaf is served, so the report is exactly as strict as the
/// evaluator and never more optimistic.
fn source_supported(
    decision: &TemporalSequenceDecision,
    requirements: &Requirements,
    data: &ExactTransportData,
    coordinates: &[VariableId],
    history: &[Value],
) -> bool {
    requirements.leaves.iter().all(|leaf| {
        let Some(world) = leaf_world(decision, coordinates, history, leaf) else { return false };
        data.laws().iter().any(|law| {
            law.population() == &*leaf.binding.population
                && leaf.binding.regime.is_none_or(|r| r == law.regime())
                && law.interventions().len() == world.len()
                && world.iter().all(|w| {
                    law.interventions()
                        .iter()
                        .any(|i| i.variable == w.variable && same(&i.value, &w.value))
                })
                && (leaf.domain == antecedent_expr::DomainRef::Observational)
                    == law.interventions().is_empty()
                && leaf
                    .variables
                    .iter()
                    .chain(leaf.conditioned_on.iter())
                    .all(|v| law.axes().iter().any(|a| a.variable == *v))
                && conditioning_has_mass(decision, coordinates, history, leaf, law)
        })
    })
}

/// Whether the event a conditional leaf conditions on (its conditioning
/// coordinates at the history's and the sequence's values) has positive mass in
/// `law`: a zero-mass conditioning event fails the evaluator's ratio.
fn conditioning_has_mass(
    decision: &TemporalSequenceDecision,
    coordinates: &[VariableId],
    history: &[Value],
    leaf: &antecedent_expr::LeafSignature,
    law: &ExactDiscreteLaw,
) -> bool {
    if leaf.conditioned_on.is_empty() {
        return true;
    }
    let slots = decision.spec.slots();
    let resolve = |v: VariableId| {
        coordinates.iter().position(|c| *c == v).map(|k| history[k].clone()).or_else(|| {
            slots.actions.iter().position(|a| *a == v).map(|s| decision.sequence[s].clone())
        })
    };
    let axes = law.axes();
    let mut strides = vec![1usize; axes.len()];
    for i in (0..axes.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * axes[i + 1].values.len();
    }
    let Some(events) = leaf
        .conditioned_on
        .iter()
        .map(|v| {
            let axis = axes.iter().position(|a| a.variable == *v)?;
            Some((axis, resolve(*v)?))
        })
        .collect::<Option<Vec<_>>>()
    else {
        return false;
    };
    law.probabilities().iter().enumerate().any(|(cell, mass)| {
        *mass > 0.0
            && events.iter().all(|(axis, value)| {
                same(&axes[*axis].values[(cell / strides[*axis]) % axes[*axis].values.len()], value)
            })
    })
}

/// The history/horizon-local support report of `data` for a decided sequence.
///
/// Step 1 rows are initial states: reached when the target's initial-state law
/// gives them mass, supported when the first action also has target mass there.
/// Step 2 rows are complete histories: outside support unless the supplied laws
/// serve every leaf the identified proof cites at the history and the sequence
/// (the evaluator reads them for every history of the lattice, reached or not),
/// `unreached` when served but the target law gives the history no mass, else
/// supported.
#[must_use]
#[doc(hidden)]
pub fn history_support(
    decision: &TemporalSequenceDecision,
    data: &ExactTransportData,
) -> HistorySupportReport {
    let slots = decision.spec.slots();
    let law = target_law(decision, data);
    let masses = law.map(|law| target_masses(decision, law));
    // Only an identified decision has a proof to read requirements from; any
    // other certifies nothing, so no history is supported.
    let requirements = Requirements::of(decision);
    let initial_coordinates = slots.initial_state();
    let history_coordinates = slots.history_coordinates();
    let key = |values: &[Value]| values.iter().map(|v| bits(v).unwrap_or(0)).collect::<Vec<_>>();
    let mut rows = Vec::new();
    let mut initial_ok = BTreeMap::new();
    for state in &decision.histories.initial {
        let (mass, first) = masses.as_ref().map_or((None, None), |m| {
            (m.initial.get(&key(state)).copied(), m.initial_first_action.get(&key(state)).copied())
        });
        let reached = masses.is_none() || mass.unwrap_or(0.0) > 0.0;
        let status = if !reached {
            "unreached"
        } else if masses.is_none() || first.unwrap_or(0.0) > 0.0 {
            "supported"
        } else {
            "outside_support"
        };
        initial_ok.insert(key(state), status == "supported");
        rows.push(HistorySupportRow {
            step: 1,
            history: state.clone(),
            target_mass: masses.as_ref().map(|_| mass.unwrap_or(0.0)),
            status,
        });
    }
    for history in &decision.histories.complete {
        let prefix = &history[..initial_coordinates.len()];
        let mass = masses
            .as_ref()
            .map(|m| m.history_first_action.get(&key(history)).copied().unwrap_or(0.0));
        // The exact evaluator enumerates the whole lattice: every complete history
        // reads the source factors the proof cites, reached or not, so source
        // support is required of all of them. Reachability only distinguishes
        // `supported` from `unreached` once the history is served.
        let reached =
            initial_ok.get(&key(prefix)).copied().unwrap_or(false) && mass.is_none_or(|m| m > 0.0);
        let served = requirements
            .as_ref()
            .is_some_and(|r| source_supported(decision, r, data, &history_coordinates, history));
        let status = if !served {
            "outside_support"
        } else if reached {
            "supported"
        } else {
            "unreached"
        };
        rows.push(HistorySupportRow {
            step: 2,
            history: history.clone(),
            target_mass: mass,
            status,
        });
    }
    HistorySupportReport {
        initial_coordinates,
        history_coordinates,
        target_law_used: law.is_some(),
        rows,
    }
}

/// The measurement window of a law set: which coordinates each supplied law
/// covers and intervenes on. A refresh keeps it; anything else re-prepares.
fn window(data: &ExactTransportData) -> BTreeSet<(String, u32, Vec<u32>, Vec<u32>)> {
    data.laws()
        .iter()
        .map(|law| {
            let mut axes = law.axes().iter().map(|a| a.variable.raw()).collect::<Vec<_>>();
            let mut interventions =
                law.interventions().iter().map(|i| i.variable.raw()).collect::<Vec<_>>();
            axes.sort_unstable();
            interventions.sort_unstable();
            (law.population().to_owned(), law.regime().raw(), axes, interventions)
        })
        .collect()
}

fn refuse(refusal: &TemporalRefusal) -> EstimationError {
    EstimationError::refused(refusal.code, format!("{}: {}", refusal.detail, refusal.message))
}

/// Refuse a decision that identified nothing, with the reason its outcome names.
fn require_identified(
    decision: &TemporalSequenceDecision,
) -> Result<&antecedent_identify::BoundTransportFunctional, EstimationError> {
    let joined = |obligations: &[Arc<str>]| {
        obligations.iter().map(ToString::to_string).collect::<Vec<_>>().join("; ")
    };
    match &decision.outcome {
        TemporalOutcome::Identified(bound) => Ok(bound),
        TemporalOutcome::StructurallyUnidentified(_) => Err(EstimationError::refused(
            reason_code!("transport_proven_non_transportable"),
            format!(
                "{TEMPORAL_CHECKED_OBSTRUCTION}: the sequence is not transportable on the \
                 unrolled diagram even with every source experiment (verified s-hedge)"
            ),
        )),
        TemporalOutcome::MissingEvidence { obligations } => Err(EstimationError::refused(
            reason_code!("transport_missing_evidence"),
            format!("{TEMPORAL_MISSING_EVIDENCE}: {}", joined(obligations)),
        )),
        TemporalOutcome::NotCertified { obligations } => Err(EstimationError::refused(
            reason_code!("transport_not_certified"),
            format!("{TEMPORAL_SEARCH_INCOMPLETE}: {}", joined(obligations)),
        )),
        TemporalOutcome::Stopped { stop } => Err(EstimationError::refused(
            reason_code!("transport_budget_cancel"),
            format!(
                "{TEMPORAL_HISTORY_BUDGET}: {}; the decision is a receipt, not a \
                 non-identification verdict",
                stop.code()
            ),
        )),
    }
}

/// Check every law against the unrolled window and the declared domains. A
/// coordinate outside the window is a period the two-step sequence does not have.
fn check_window(
    decision: &TemporalSequenceDecision,
    data: &ExactTransportData,
) -> Result<(), EstimationError> {
    let spec = &decision.spec;
    for law in data.laws() {
        let context = format!("the {} law of regime {}", law.population(), law.regime().raw());
        let unknown = |v: VariableId| {
            refuse(&TemporalRefusal::horizon(format!(
                "{context} covers coordinate {} outside the two-step window; a new period \
                 needs a new preparation",
                v.raw()
            )))
        };
        for axis in law.axes() {
            let coordinate =
                spec.coordinate(axis.variable).ok_or_else(|| unknown(axis.variable))?;
            if let Some(value) = axis.values.iter().find(|v| !coordinate.accepts(v)) {
                return Err(EstimationError::data_msg(format!(
                    "{context} gives {} the value {value:?} outside its declared domain",
                    coordinate.name
                )));
            }
        }
        for assignment in law.interventions() {
            let coordinate =
                spec.coordinate(assignment.variable).ok_or_else(|| unknown(assignment.variable))?;
            if !coordinate.accepts(&assignment.value) {
                return Err(EstimationError::data_msg(format!(
                    "{context} intervenes on {} outside its declared domain",
                    coordinate.name
                )));
            }
        }
    }
    Ok(())
}

fn outside_message(decision: &TemporalSequenceDecision, report: &HistorySupportReport) -> String {
    let name = |v: &VariableId| {
        decision.spec.coordinate(*v).map_or_else(|| v.raw().to_string(), |c| c.name.to_string())
    };
    let outside = report.outside();
    let shown = outside
        .iter()
        .take(3)
        .map(|row| {
            let coordinates = if row.step == 1 {
                &report.initial_coordinates
            } else {
                &report.history_coordinates
            };
            let cells = coordinates
                .iter()
                .zip(&row.history)
                .map(|(c, v)| format!("{}={}", name(c), v.as_f64().unwrap_or(f64::NAN)))
                .collect::<Vec<_>>()
                .join(", ");
            format!("step {} ({cells})", row.step)
        })
        .collect::<Vec<_>>()
        .join("; ");
    format!(
        "{} of {} histories are outside certified support: {shown}",
        outside.len(),
        report.rows.len()
    )
}

/// The one exact evaluation of a decided sequence, compiled against supplied laws.
#[derive(Clone, Debug)]
pub struct PreparedTemporalSequence {
    decision: TemporalSequenceDecision,
    data: ExactTransportData,
    request: Assignment,
    limits: ExactEvaluationLimits,
    plan: ExactEvaluationPlan,
    support: HistorySupportReport,
    id: u64,
}

/// Every preparation gets its own identity, so a report can be tied to the
/// preparation that evaluated it.
static NEXT_PREPARATION: AtomicU64 = AtomicU64::new(1);

/// The evaluated sequence: its point response and everything it rests on.
#[derive(Clone, Debug)]
pub struct TemporalSequenceReport {
    /// Always two.
    pub horizon: usize,
    /// The action of each step.
    pub sequence: Vec<Value>,
    /// Target distribution of the outcome under the whole sequence.
    pub distribution: ExactDistribution,
    /// Mean of the outcome.
    pub mean: f64,
    /// History/horizon-local support.
    pub support: HistorySupportReport,
    /// Every coordinate's mechanism assumption per time slice.
    pub invariances: Vec<SliceInvariance>,
    /// Time-varying confounders of the second action.
    pub time_varying_confounders: Vec<VariableId>,
    /// Which catalog evidence touches each time slice.
    pub evidence: Vec<SliceEvidence>,
    /// Always [`TEMPORAL_INFERENCE_CLAIM`].
    pub inference_claim: &'static str,
    origin: u64,
}

impl TemporalSequenceReport {
    /// Whether `prepared` is the preparation that evaluated this report. A
    /// report of another preparation (even of the same premises) is foreign to it.
    #[must_use]
    pub const fn is_from(&self, prepared: &PreparedTemporalSequence) -> bool {
        self.origin == prepared.id
    }
}

/// Compile a decided sequence once against supplied exact laws.
///
/// # Errors
/// A decision that identified nothing (its outcome names the refusal: a checked
/// obstruction, missing evidence, `temporal_transport.search_incomplete`, or the
/// `temporal_transport.history_budget` receipt); a law outside the two-step
/// window (`temporal_transport.horizon`) or the declared domains; a reached
/// history outside certified support
/// (`temporal_transport.history_outside_support`); or provider and resource
/// failures of the exact evaluator.
pub fn prepare_temporal_sequence(
    decision: TemporalSequenceDecision,
    data: ExactTransportData,
    limits: ExactEvaluationLimits,
    ctx: &ExecutionContext,
) -> Result<PreparedTemporalSequence, EstimationError> {
    let functional = require_identified(&decision)?.clone();
    check_window(&decision, &data)?;
    let support = history_support(&decision, &data);
    if !support.outside().is_empty() {
        return Err(refuse(&TemporalRefusal::history_outside_support(outside_message(
            &decision, &support,
        ))));
    }
    let slots = decision.spec.slots();
    let request = Assignment::from_pairs(
        slots.actions.iter().copied().zip(decision.sequence.iter().cloned()),
    );
    let plan = crate::transport::prepare_exact_transport(
        &functional,
        data.clone(),
        request.clone(),
        limits,
        ctx,
    )
    .map_err(|error| eval_refusal(&error))?;
    Ok(PreparedTemporalSequence {
        decision,
        data,
        request,
        limits,
        plan,
        support,
        id: NEXT_PREPARATION.fetch_add(1, Ordering::Relaxed),
    })
}

/// A support failure of the exact evaluator is a history outside certified
/// support; every other failure keeps its registered class.
fn eval_refusal(error: &antecedent_expr::EvalError) -> EstimationError {
    if crate::transport::is_support_failure(error) {
        refuse(&TemporalRefusal::history_outside_support(error.to_string()))
    } else {
        crate::transport::refuse_eval(error)
    }
}

impl PreparedTemporalSequence {
    /// The frozen decision: sequence, horizon, proof and invariances.
    #[must_use]
    pub const fn decision(&self) -> &TemporalSequenceDecision {
        &self.decision
    }

    /// The retained laws.
    #[must_use]
    pub const fn data(&self) -> &ExactTransportData {
        &self.data
    }

    /// Provider identity.
    #[must_use]
    pub const fn provider(&self) -> &'static str {
        TEMPORAL_EXACT_PROVIDER
    }

    /// The request that binds both actions to the sequence.
    #[must_use]
    pub const fn request(&self) -> &Assignment {
        &self.request
    }

    /// Evaluation limits.
    #[must_use]
    pub const fn limits(&self) -> ExactEvaluationLimits {
        self.limits
    }

    /// The history/horizon-local support of the retained laws.
    #[must_use]
    pub const fn support(&self) -> &HistorySupportReport {
        &self.support
    }

    /// Re-estimate against new laws of the same measurement window; the proof is
    /// kept. Laws covering another window (a new period, a longer horizon, other
    /// coordinates or regimes) refuse: they need a new preparation.
    ///
    /// # Errors
    /// `temporal_transport.horizon` for a changed window, and as
    /// [`prepare_temporal_sequence`] otherwise.
    pub fn refresh(
        &self,
        data: ExactTransportData,
        ctx: &ExecutionContext,
    ) -> Result<Self, EstimationError> {
        check_window(&self.decision, &data)?;
        if window(&data) != window(&self.data) {
            return Err(refuse(&TemporalRefusal::horizon(
                "the measurement window changed (other coordinates, regimes or interventions); \
                 a new window or horizon needs a new preparation",
            )));
        }
        let refreshed = prepare_temporal_sequence(self.decision.clone(), data, self.limits, ctx)?;
        // The initial-state law (baseline and step-1 covariates) is fixed per
        // population: a refresh that moves the target's law of the initial state
        // is initial-state uncertainty or a new period (2.3A), not fresh evidence
        // about the same sequence.
        let initial = |report: &HistorySupportReport| {
            report.rows.iter().filter(|r| r.step == 1).map(|r| r.target_mass).collect::<Vec<_>>()
        };
        let (before, after) = (initial(&self.support), initial(&refreshed.support));
        let moved = before.len() != after.len()
            || before.iter().zip(&after).any(|pair| match pair {
                (Some(a), Some(b)) => (a - b).abs() > INITIAL_STATE_TOLERANCE,
                (None, None) => false,
                _ => true,
            });
        if moved {
            return Err(refuse(&TemporalRefusal::horizon(
                "the target's initial-state law changed; the initial state is fixed per \
                 population, so a moved initial state is a new preparation (2.3A)",
            )));
        }
        Ok(refreshed)
    }

    /// Evaluate the whole sequence's functional.
    ///
    /// # Errors
    /// A history outside support surfaced by the evaluator, a numerical failure,
    /// or cancellation.
    pub fn evaluate(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<TemporalSequenceReport, EstimationError> {
        crate::transport::refuse_cancelled(ctx, "temporal transport evaluation")?;
        let distribution = self.plan.evaluate(ctx).map_err(|error| eval_refusal(&error))?;
        let outcome = self.decision.spec.slots().outcome;
        let mean =
            distribution.mean(outcome).map_err(|error| crate::transport::refuse_eval(&error))?;
        Ok(TemporalSequenceReport {
            horizon: self.decision.spec.horizon(),
            sequence: self.decision.sequence.to_vec(),
            distribution,
            mean,
            support: self.support.clone(),
            invariances: self.decision.invariances.clone(),
            time_varying_confounders: self.decision.time_varying_confounders.clone(),
            evidence: self.decision.evidence.clone(),
            inference_claim: TEMPORAL_INFERENCE_CLAIM,
            origin: self.id,
        })
    }
}
