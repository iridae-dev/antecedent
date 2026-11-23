//! The set of preference parameters compatible with recorded choices.
//!
//! Assumption: each recorded choice was at least as good as every rejected option, within a
//! declared tolerance, under the declared value model and the recorded expectations
//! ([`ASSUMPTION_CHOICES_OPTIMAL`]). With `F` the expected fixed part and `T` the expected
//! linear-form terms of each option ([`choice_features`]), a record with chosen option `c` and
//! rejected option `j` adds the relation
//!
//! `sum_k theta_k (T_ck - T_jk) >= -tolerance - (F_c - F_j)`
//!
//! with id `"choice.{record_id}.{rejected_option}"` and origin [`RelationOrigin::Choice`]. Only
//! available options are compared: an option listed as
//! [`OptionBelief::Unavailable`](crate::choice_record::OptionBelief::Unavailable) existed but
//! could not be chosen, so not choosing it teaches nothing about preferences and no relation is
//! formed against it. A record with a single available option forms no relation, varies no
//! parameter and changes nothing; it still counts as a record of the outcome
//! ([`uninformative_records`] names such records). A
//! difference smaller in magnitude than [`COEFFICIENT_RELATIVE_THRESHOLD`] times the larger of
//! its two operands is read as zero (it is rounding, not a recorded difference). A relation with
//! no remaining coefficient is not added; when it is violated the record contradicts the model
//! on its own and is reported as a single-record conflict.
//!
//! The learned set is the declared set (parameters with their declared ranges, declared and
//! shape-rule relations) with the choice relations appended in record order, then option order.
//! Each parameter's learned range comes from two linear programs over that set; the range it is
//! compared with to judge narrowing is the declared range restricted by the declared relations,
//! computed the same way. When the choice relations leave no admissible point, the outcome is a
//! conflict: a smallest set of records that cannot hold together with the declared set (an
//! exhaustive search up to a size), or an irreducible one from a deletion filter, together with
//! the smallest tolerance at which every record of the outcome becomes compatible.
//!
//! A stated comparison ([`BeliefSource::Stated`](crate::choice_record::BeliefSource::Stated)) is
//! what someone said they would choose among hypothetical bundles, not what they chose. It adds
//! relations exactly as a recorded choice does, with the same tolerance, and the learned set then
//! also stands on the statement's named assumption that it reflects the preferences the person
//! would act on (carried in [`LearnedPreferenceSet::assumptions`]).
//!
//! A learned set is an assumption range, not a probability statement.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use antecedent_contract::ExternalRefusal;
use antecedent_core::ExecutionContext;
use antecedent_stats::linear_program::{
    self, ConstraintSense, LinearConstraint, LinearProgram, LpError, LpOptions, LpOutcome,
    LpSolution,
};

use crate::choice_record::{
    ChoiceFeatures, ChoiceRecordError, ChoiceRecordSet, ChoiceSetFeatures, Pooling, choice_features,
};
use crate::preference_model::{
    LinearRelation, PreferenceModelError, PreferenceParameter, PreferenceRows, PreferenceSet,
    RelationOrigin, RelationSense, ValueModel,
};
use crate::preference_refusal::{LEARN, refusal};
use crate::signal::at_stage;

/// Identity of the assumption every learned set stands on: each recorded choice was at least as
/// good as every rejected option, within the declared tolerance, under the declared value model
/// and the recorded expectations.
pub const ASSUMPTION_CHOICES_OPTIMAL: &str = "assumption.recorded_choices_optimal_within_tolerance";

/// A difference `a - b` of two expected terms (or fixed parts) whose magnitude is at most this
/// fraction of `max(|a|, |b|)` is read as zero.
pub const COEFFICIENT_RELATIVE_THRESHOLD: f64 = 1e-12;

/// Largest number of record subsets the exhaustive smallest-conflict search may examine; above
/// it the conflict is found by the deletion filter alone and reported as
/// [`ConflictKind::Irreducible`].
pub const SMALLEST_CONFLICT_SUBSET_BUDGET: usize = 50_000;

/// Options of [`learn_compatible_set`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SetLearningOptions {
    /// How far below a rejected option's value a chosen option may fall, in utility units;
    /// finite and non-negative.
    pub tolerance: f64,
    /// Largest number of records the exhaustive smallest-conflict search tries together.
    pub conflict_search_size: usize,
    /// Tolerances and work limit of every linear program.
    pub lp: LpOptions,
}

impl Default for SetLearningOptions {
    fn default() -> Self {
        Self { tolerance: 0.0, conflict_search_size: 3, lp: LpOptions::default() }
    }
}

/// The range of one parameter before and after the recorded choices.
#[derive(Clone, Debug, PartialEq)]
pub struct ParameterRange {
    /// Parameter identity.
    pub parameter_id: String,
    /// The declared range `(lower, upper)`.
    pub declared: (f64, f64),
    /// Smallest and largest value of the parameter over the learned set.
    pub learned: (f64, f64),
    /// Some record of the outcome has two available options whose expected term for this
    /// parameter differs.
    pub varied_in_records: bool,
    /// The learned range lies strictly inside the declared range as restricted by the declared
    /// relations (beyond the linear-program feasibility tolerance) at either end.
    pub narrowed: bool,
}

/// A choice relation that is tight at an extreme of some parameter's learned range.
#[derive(Clone, Debug, Eq, PartialEq, PartialOrd, Ord, Hash)]
pub struct BindingChoice {
    /// The record.
    pub record_id: String,
    /// The rejected option whose relation is tight.
    pub rejected_option: String,
}

/// How a reported conflict was found.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConflictKind {
    /// The exhaustive search up to the configured size found it; no fewer records conflict.
    Smallest,
    /// The deletion filter found it; dropping any one of its records removes the conflict, but
    /// a conflict with fewer records may exist.
    Irreducible,
}

/// Records that cannot all hold together with the declared set.
#[derive(Clone, Debug, PartialEq)]
pub struct ChoiceConflict {
    /// Record identities, sorted.
    pub records: Vec<String>,
    /// How the conflict was found.
    pub kind: ConflictKind,
    /// The smallest total tolerance at which every record of the outcome becomes compatible
    /// with the declared set; it exceeds the tolerance the outcome was learned with.
    pub tolerance_needed: f64,
}

/// The declared set narrowed by recorded choices.
#[derive(Clone, Debug, PartialEq)]
pub struct LearnedPreferenceSet {
    /// Identity of the value model.
    pub model_identity: String,
    /// Digest of the record set the outcome was learned from: the decider's records alone for
    /// a per-decider set, every record otherwise.
    pub records_digest: String,
    /// The decider, or `None` for a set learned from every decider's records.
    pub decider: Option<String>,
    /// The declared parameters (declared ranges unchanged), the declared and shape-rule
    /// relations, then the choice relations in record order and option order.
    pub set: PreferenceSet,
    /// One entry per parameter, in parameter order.
    pub ranges: Vec<ParameterRange>,
    /// Choice relations tight at the minimiser or maximiser of some parameter, sorted by record
    /// then option, distinct.
    pub binding: Vec<BindingChoice>,
    /// The tolerance the choice relations were built with.
    pub tolerance: f64,
    /// Number of records the set was learned from, including records with a single available
    /// option (which form no relation).
    pub n_records: usize,
    /// Sorted, distinct: parameter assumptions, the record set's assumptions, and
    /// [`ASSUMPTION_CHOICES_OPTIMAL`].
    pub assumptions: Vec<String>,
}

/// What a group of records says about the preference parameters.
#[derive(Clone, Debug, PartialEq)]
pub enum SetOutcome {
    /// The records are compatible with the declared set.
    Learned(LearnedPreferenceSet),
    /// The records contradict the declared set.
    Contradictory(ChoiceConflict),
}

/// The outcome for one decider's records.
#[derive(Clone, Debug, PartialEq)]
pub struct DeciderSet {
    /// The decider.
    pub decider: String,
    /// Number of the decider's records.
    pub n_records: usize,
    /// What the decider's records say.
    pub outcome: SetOutcome,
}

/// The parameter values compatible with every decider at once.
#[derive(Clone, Debug, PartialEq)]
pub struct DeciderOverlap {
    /// The outcome of every decider's relations together.
    pub common: SetOutcome,
    /// When `common` is contradictory, the deciders of the conflict's records, sorted and
    /// distinct; empty otherwise.
    pub disagreeing: Vec<String>,
}

/// Every outcome of [`learn_compatible_set`].
#[derive(Clone, Debug, PartialEq)]
pub struct SetLearningResult {
    /// Every record read as one set of preferences; `Some` exactly under [`Pooling::Pooled`].
    pub pooled: Option<SetOutcome>,
    /// One outcome per decider, sorted by decider.
    pub per_decider: Vec<DeciderSet>,
    /// `Some` exactly under [`Pooling::PerDecider`] with more than one decider.
    pub overlap: Option<DeciderOverlap>,
}

/// Why the compatible set is refused. Contradictory records are an outcome, not an error.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum SetLearningError {
    /// The value model, or the learned set built from it, is invalid.
    #[error("invalid value model ({})", .0.detail())]
    Model(PreferenceModelError),
    /// The records are invalid or have no features under the model.
    #[error("invalid records: {0}")]
    Records(ChoiceRecordError),
    /// A linear program refused its input.
    #[error("linear program refused ({})", .0.detail())]
    Lp(LpError),
    /// The tolerance is negative.
    #[error("tolerance {tolerance} is negative")]
    ToleranceNegative {
        /// The tolerance given.
        tolerance: f64,
    },
    /// The tolerance is not finite.
    #[error("tolerance is not finite")]
    ToleranceNonFinite,
    /// The declared ranges and relations admit no parameter value before any record is added.
    #[error("the declared preference set is empty")]
    DeclaredSetEmpty,
    /// A choice relation id equals a declared relation id or another choice relation id.
    #[error("choice relation id {id} is not unique")]
    RelationIdCollision {
        /// The repeated id.
        id: String,
    },
    /// Two linear programs over the same region disagreed on whether it is empty.
    #[error("linear programs disagree on the region during {stage}")]
    ProgramInconsistent {
        /// Where the disagreement was found.
        stage: &'static str,
    },
    /// The execution context was cancelled between linear programs.
    #[error("learning the compatible set was cancelled")]
    Cancelled,
}

impl SetLearningError {
    /// Stable detail id, prefixed `preference_set_learning.`; a record refusal keeps its own
    /// `choice_record.` detail, the detail the returned refusal carries.
    #[must_use]
    pub const fn detail(&self) -> &'static str {
        use antecedent_core::reason_details;
        match self {
            Self::Model(_) => reason_details::preference_set_learning::MODEL_INVALID,
            Self::Records(inner) => inner.detail(),
            Self::Lp(_) => reason_details::preference_set_learning::LINEAR_PROGRAM_REFUSED,
            Self::ToleranceNegative { .. } => {
                reason_details::preference_set_learning::TOLERANCE_NEGATIVE
            }
            Self::ToleranceNonFinite => {
                reason_details::preference_set_learning::TOLERANCE_NON_FINITE
            }
            Self::DeclaredSetEmpty => reason_details::preference_set_learning::DECLARED_SET_EMPTY,
            Self::RelationIdCollision { .. } => {
                reason_details::preference_set_learning::RELATION_ID_COLLISION
            }
            Self::ProgramInconsistent { .. } => {
                reason_details::preference_set_learning::PROGRAM_INCONSISTENT
            }
            Self::Cancelled => reason_details::preference_set_learning::CANCELLED,
        }
    }
}

/// Learn the preference parameters compatible with recorded choices.
///
/// Per-decider sets use only that decider's records; the pooled set (only under
/// [`Pooling::Pooled`]) and the overlap (only under [`Pooling::PerDecider`] with more than one
/// decider) use every record.
///
/// # Errors
/// - [`SetLearningError::ToleranceNonFinite`], [`SetLearningError::ToleranceNegative`];
/// - [`SetLearningError::Model`] for an invalid model;
/// - [`SetLearningError::Records`] for invalid records or records without features under the
///   model (a value with no linear form in the free parameters, a belief that needs draws);
/// - [`SetLearningError::RelationIdCollision`] when a choice relation id is not unique;
/// - [`SetLearningError::DeclaredSetEmpty`] when the declared set admits no point;
/// - [`SetLearningError::Lp`] when a linear program refuses (its options, its work limit);
/// - [`SetLearningError::ProgramInconsistent`] when linear programs over one region disagree.
pub fn learn_compatible_set(
    model: &ValueModel,
    records: &ChoiceRecordSet,
    options: SetLearningOptions,
) -> Result<SetLearningResult, SetLearningError> {
    learn(model, records, options, None)
}

/// [`learn_compatible_set`] under an execution context: the context's cancellation token is read
/// before every linear program of the parameter ranges, the tolerance and the conflict search
/// (the smallest-conflict search can solve up to [`SMALLEST_CONFLICT_SUBSET_BUDGET`] programs).
///
/// # Errors
/// As [`learn_compatible_set`]; [`SetLearningError::Cancelled`] once the token is set.
pub fn learn_compatible_set_with_context(
    model: &ValueModel,
    records: &ChoiceRecordSet,
    options: SetLearningOptions,
    ctx: &ExecutionContext,
) -> Result<SetLearningResult, SetLearningError> {
    learn(model, records, options, Some(ctx))
}

pub(crate) fn learn(
    model: &ValueModel,
    records: &ChoiceRecordSet,
    options: SetLearningOptions,
    ctx: Option<&ExecutionContext>,
) -> Result<SetLearningResult, SetLearningError> {
    if !options.tolerance.is_finite() {
        return Err(SetLearningError::ToleranceNonFinite);
    }
    if options.tolerance < 0.0 {
        return Err(SetLearningError::ToleranceNegative { tolerance: options.tolerance });
    }
    model.validate().map_err(SetLearningError::Model)?;
    let features = choice_features(model, records).map_err(SetLearningError::Records)?;
    let model_identity = model.identity().map_err(SetLearningError::Model)?;
    let choices = record_choices(&features, options.tolerance);
    check_relation_ids(&model.preferences, &choices)?;

    let declared = &model.preferences;
    let declared_region = Region::of(declared);
    if declared.dimension() > 0 && !feasible(&declared_region, options.lp, ctx)? {
        return Err(SetLearningError::DeclaredSetEmpty);
    }
    let baseline = extremes(&declared_region, &declared.parameters, options.lp, ctx)?
        .ok_or(SetLearningError::DeclaredSetEmpty)?;
    let learner =
        Learner { model, model_identity, choices, baseline: baseline.ranges, options, ctx };

    let all: Vec<usize> = (0..learner.choices.len()).collect();
    let mut per_decider = Vec::new();
    for decider in records.deciders() {
        let members: Vec<usize> =
            all.iter().copied().filter(|&m| learner.choices[m].decider == decider).collect();
        let subset = ChoiceRecordSet {
            attributes: records.attributes.clone(),
            population: records.population.clone(),
            records: records.records.iter().filter(|r| r.decider == decider).cloned().collect(),
            pooling: Pooling::PerDecider,
        };
        let outcome = learner.outcome(&members, Some(decider.as_str()), &subset)?;
        per_decider.push(DeciderSet { decider, n_records: members.len(), outcome });
    }
    let pooled = match records.pooling {
        Pooling::Pooled { .. } => Some(learner.outcome(&all, None, records)?),
        Pooling::PerDecider => None,
    };
    let overlap = if matches!(records.pooling, Pooling::PerDecider) && per_decider.len() > 1 {
        let common = learner.outcome(&all, None, records)?;
        let disagreeing = match &common {
            SetOutcome::Learned(_) => Vec::new(),
            SetOutcome::Contradictory(conflict) => {
                let named: BTreeSet<&str> = learner
                    .choices
                    .iter()
                    .filter(|c| conflict.records.contains(&c.record_id))
                    .map(|c| c.decider.as_str())
                    .collect();
                named.into_iter().map(str::to_owned).collect()
            }
        };
        Some(DeciderOverlap { common, disagreeing })
    } else {
        None
    };
    Ok(SetLearningResult { pooled, per_decider, overlap })
}

/// Identities of the records, in record order, with fewer than two available options: they form
/// no choice relation and teach nothing about preferences.
#[must_use]
pub fn uninformative_records(features: &ChoiceFeatures) -> Vec<String> {
    features
        .sets
        .iter()
        .filter(|set| set.option_ids.len() < 2)
        .map(|set| set.record_id.clone())
        .collect()
}

/// Choice relations and slack rows of one record.
struct RecordChoices {
    record_id: String,
    decider: String,
    /// Relations with at least one nonzero coefficient, in option order.
    relations: Vec<LinearRelation>,
    /// For every rejected available option: the dense coefficients `T_c - T_j` and `-(F_c - F_j)`.
    slack_rows: Vec<(Vec<f64>, f64)>,
    /// Some relation has no coefficient and is violated.
    contradicts_alone: bool,
    /// Per parameter: two options' expected terms differ.
    varied: Vec<bool>,
}

/// `a - b`, or zero when the difference is within rounding of the operands.
fn difference(a: f64, b: f64) -> f64 {
    let d = a - b;
    if d.abs() <= COEFFICIENT_RELATIVE_THRESHOLD * a.abs().max(b.abs()) { 0.0 } else { d }
}

fn record_choices(features: &ChoiceFeatures, tolerance: f64) -> Vec<RecordChoices> {
    features.sets.iter().map(|set| one_record(set, features.n_parameters, tolerance)).collect()
}

fn one_record(set: &ChoiceSetFeatures, n_parameters: usize, tolerance: f64) -> RecordChoices {
    let term =
        |option: usize, j: usize| set.terms.get(option * n_parameters + j).copied().unwrap_or(0.0);
    let n_options = set.option_ids.len();
    let varied = (0..n_parameters)
        .map(|j| {
            (0..n_options).any(|first| {
                (first + 1..n_options)
                    .any(|second| difference(term(first, j), term(second, j)).abs() > 0.0)
            })
        })
        .collect();
    let chosen = set.chosen;
    let fixed = |option: usize| set.fixed.get(option).copied().unwrap_or(0.0);
    let mut relations = Vec::new();
    let mut slack_rows = Vec::new();
    let mut contradicts_alone = false;
    for (option, rejected) in set.option_ids.iter().enumerate() {
        if option == chosen {
            continue;
        }
        let coefficients: Vec<f64> =
            (0..n_parameters).map(|j| difference(term(chosen, j), term(option, j))).collect();
        let value_gap = difference(fixed(chosen), fixed(option));
        let bound = -tolerance - value_gap;
        let terms: Vec<(usize, f64)> = coefficients
            .iter()
            .enumerate()
            .filter(|(_, v)| v.abs() > 0.0)
            .map(|(j, v)| (j, *v))
            .collect();
        if terms.is_empty() {
            contradicts_alone |= bound > 0.0;
        } else {
            relations.push(LinearRelation {
                id: choice_relation_id(&set.record_id, rejected),
                terms,
                sense: RelationSense::AtLeast,
                bound,
                origin: RelationOrigin::Choice {
                    record_id: set.record_id.clone(),
                    rejected_option: rejected.clone(),
                },
            });
        }
        slack_rows.push((coefficients, -value_gap));
    }
    RecordChoices {
        record_id: set.record_id.clone(),
        decider: set.decider.clone(),
        relations,
        slack_rows,
        contradicts_alone,
        varied,
    }
}

fn choice_relation_id(record_id: &str, rejected_option: &str) -> String {
    format!("choice.{record_id}.{rejected_option}")
}

fn check_relation_ids(
    declared: &PreferenceSet,
    choices: &[RecordChoices],
) -> Result<(), SetLearningError> {
    let mut ids: BTreeSet<&str> = declared.relations.iter().map(|r| r.id.as_str()).collect();
    for relation in choices.iter().flat_map(|c| &c.relations) {
        if !ids.insert(relation.id.as_str()) {
            return Err(SetLearningError::RelationIdCollision { id: relation.id.clone() });
        }
    }
    Ok(())
}

/// A region `{theta : constraints, lower <= theta <= upper}`.
struct Region {
    constraints: Vec<LinearConstraint>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    /// Relation index of each constraint.
    row_relation: Vec<usize>,
}

impl Region {
    fn of(set: &PreferenceSet) -> Self {
        let PreferenceRows { coefficients, rhs, row_relation } = set.inequality_rows();
        Self {
            constraints: at_most(coefficients, rhs),
            lower: set.parameters.iter().map(|p| p.lower).collect(),
            upper: set.parameters.iter().map(|p| p.upper).collect(),
            row_relation,
        }
    }
}

fn at_most(coefficients: Vec<Vec<f64>>, rhs: Vec<f64>) -> Vec<LinearConstraint> {
    coefficients
        .into_iter()
        .zip(rhs)
        .map(|(coefficients, rhs)| LinearConstraint {
            coefficients,
            sense: ConstraintSense::AtMost,
            rhs,
        })
        .collect()
}

fn program(region: &Region, objective: Vec<f64>) -> LinearProgram {
    LinearProgram {
        objective,
        constraints: region.constraints.clone(),
        lower: region.lower.clone(),
        upper: region.upper.clone(),
    }
}

/// `Err(Cancelled)` when a context is given and its token is set.
fn observe(ctx: Option<&ExecutionContext>) -> Result<(), SetLearningError> {
    if ctx.is_some_and(|ctx| ctx.cancellation.is_cancelled()) {
        return Err(SetLearningError::Cancelled);
    }
    Ok(())
}

/// A linear-program refusal, with cancellation kept as cancellation.
fn lp_error(error: LpError) -> SetLearningError {
    match error {
        LpError::Cancelled => SetLearningError::Cancelled,
        other => SetLearningError::Lp(other),
    }
}

/// Whether the region has a point. Only for a region with at least one variable.
fn feasible(
    region: &Region,
    lp: LpOptions,
    ctx: Option<&ExecutionContext>,
) -> Result<bool, SetLearningError> {
    observe(ctx)?;
    let outcome = linear_program::solve(&program(region, vec![0.0; region.lower.len()]), lp)
        .map_err(SetLearningError::Lp)?;
    Ok(!matches!(outcome, LpOutcome::Infeasible(_)))
}

/// Minimiser or maximiser of parameter `k`, `None` when the region is empty.
fn extreme(
    region: &Region,
    k: usize,
    maximise: bool,
    lp: LpOptions,
    ctx: Option<&ExecutionContext>,
) -> Result<Option<LpSolution>, SetLearningError> {
    observe(ctx)?;
    let mut objective = vec![0.0; region.lower.len()];
    if let Some(slot) = objective.get_mut(k) {
        *slot = 1.0;
    }
    let program = program(region, objective);
    let outcome = if maximise {
        linear_program::maximize(&program, lp)
    } else {
        linear_program::solve(&program, lp)
    }
    .map_err(SetLearningError::Lp)?;
    match outcome {
        LpOutcome::Optimal(solution) => Ok(Some(solution)),
        LpOutcome::Infeasible(_) => Ok(None),
        LpOutcome::Unbounded { .. } => {
            Err(SetLearningError::ProgramInconsistent { stage: "parameter_range" })
        }
    }
}

/// Range of every parameter over a region, and the constraints tight at any extreme.
struct Extremes {
    ranges: Vec<(f64, f64)>,
    active: BTreeSet<usize>,
}

/// `None` when the region is empty. A parameter whose declared range is a single value needs
/// no program.
fn extremes(
    region: &Region,
    parameters: &[PreferenceParameter],
    lp: LpOptions,
    ctx: Option<&ExecutionContext>,
) -> Result<Option<Extremes>, SetLearningError> {
    let mut ranges = Vec::with_capacity(parameters.len());
    let mut active = BTreeSet::new();
    for (k, parameter) in parameters.iter().enumerate() {
        if parameter.lower.total_cmp(&parameter.upper).is_ge() {
            ranges.push((parameter.lower, parameter.upper));
            continue;
        }
        let (Some(low), Some(high)) =
            (extreme(region, k, false, lp, ctx)?, extreme(region, k, true, lp, ctx)?)
        else {
            return Ok(None);
        };
        active.extend(low.active.iter().copied());
        active.extend(high.active.iter().copied());
        let clamp = |v: f64| v.clamp(parameter.lower, parameter.upper);
        let (low, high) = (clamp(low.objective), clamp(high.objective));
        ranges.push((low, high.max(low)));
    }
    Ok(Some(Extremes { ranges, active }))
}

/// Number of subsets of sizes `1..=max_size` of `n` items exceeds `budget`.
fn subsets_exceed(n: usize, max_size: usize, budget: usize) -> bool {
    let mut binomial: u128 = 1;
    let mut total: u128 = 0;
    for size in 1..=max_size.min(n) {
        binomial = binomial * (n - size + 1) as u128 / size as u128;
        total += binomial;
        if total > budget as u128 {
            return true;
        }
    }
    false
}

/// Shared inputs of every outcome.
struct Learner<'a> {
    model: &'a ValueModel,
    model_identity: String,
    choices: Vec<RecordChoices>,
    /// Range of each parameter over the declared set.
    baseline: Vec<(f64, f64)>,
    options: SetLearningOptions,
    /// The caller's execution context, when the run is cancellable.
    ctx: Option<&'a ExecutionContext>,
}

impl Learner<'_> {
    fn declared(&self) -> &PreferenceSet {
        &self.model.preferences
    }

    /// The outcome of the records `members` (indices in record order); `subset` holds exactly
    /// those records.
    fn outcome(
        &self,
        members: &[usize],
        decider: Option<&str>,
        subset: &ChoiceRecordSet,
    ) -> Result<SetOutcome, SetLearningError> {
        let mut by_id: Vec<usize> = members.to_vec();
        by_id.sort_by(|a, b| self.choices[*a].record_id.cmp(&self.choices[*b].record_id));
        if let Some(&alone) = by_id.iter().find(|&&m| self.choices[m].contradicts_alone) {
            return Ok(SetOutcome::Contradictory(ChoiceConflict {
                records: vec![self.choices[alone].record_id.clone()],
                kind: ConflictKind::Smallest,
                tolerance_needed: self.tolerance_needed(members)?,
            }));
        }
        let declared = self.declared();
        let n_declared = declared.relations.len();
        let mut relations = declared.relations.clone();
        relations.extend(members.iter().flat_map(|&m| self.choices[m].relations.iter().cloned()));
        let set = PreferenceSet { parameters: declared.parameters.clone(), relations };
        set.validate().map_err(SetLearningError::Model)?;
        let region = Region::of(&set);
        if set.dimension() > 0 && !feasible(&region, self.options.lp, self.ctx)? {
            return Ok(SetOutcome::Contradictory(self.conflict(&by_id, members)?));
        }
        let found = extremes(&region, &set.parameters, self.options.lp, self.ctx)?
            .ok_or(SetLearningError::ProgramInconsistent { stage: "parameter_range" })?;
        let mut binding = BTreeSet::new();
        for row in &found.active {
            let Some(&relation) = region.row_relation.get(*row) else { continue };
            if relation < n_declared {
                continue;
            }
            if let Some(RelationOrigin::Choice { record_id, rejected_option }) =
                set.relations.get(relation).map(|r| &r.origin)
            {
                binding.insert(BindingChoice {
                    record_id: record_id.clone(),
                    rejected_option: rejected_option.clone(),
                });
            }
        }
        let ranges = self.ranges(members, &found.ranges);
        let mut assumptions: BTreeSet<String> =
            set.parameters.iter().map(|p| p.assumption_id.clone()).collect();
        assumptions.extend(subset.assumptions());
        assumptions.insert(ASSUMPTION_CHOICES_OPTIMAL.to_owned());
        Ok(SetOutcome::Learned(LearnedPreferenceSet {
            model_identity: self.model_identity.clone(),
            records_digest: subset.digest().map_err(SetLearningError::Records)?,
            decider: decider.map(str::to_owned),
            set,
            ranges,
            binding: binding.into_iter().collect(),
            tolerance: self.options.tolerance,
            n_records: members.len(),
            assumptions: assumptions.into_iter().collect(),
        }))
    }

    fn ranges(&self, members: &[usize], learned: &[(f64, f64)]) -> Vec<ParameterRange> {
        let slack = self.options.lp.feasibility_tolerance;
        self.declared()
            .parameters
            .iter()
            .enumerate()
            .map(|(k, parameter)| {
                let (base_low, base_high) =
                    self.baseline.get(k).copied().unwrap_or((parameter.lower, parameter.upper));
                let (low, high) = learned.get(k).copied().unwrap_or((base_low, base_high));
                let narrowed = low > base_low + slack * (1.0 + base_low.abs())
                    || high < base_high - slack * (1.0 + base_high.abs());
                ParameterRange {
                    parameter_id: parameter.id.clone(),
                    declared: (parameter.lower, parameter.upper),
                    learned: (low, high),
                    varied_in_records: members
                        .iter()
                        .any(|&m| self.choices[m].varied.get(k).copied().unwrap_or(false)),
                    narrowed,
                }
            })
            .collect()
    }

    /// The declared constraints in the parameters followed by `extra` zero columns.
    fn declared_constraints(&self, extra: usize) -> Vec<LinearConstraint> {
        let PreferenceRows { coefficients, rhs, .. } = self.declared().inequality_rows();
        let widened = coefficients
            .into_iter()
            .map(|mut row| {
                row.extend(std::iter::repeat_n(0.0, extra));
                row
            })
            .collect();
        at_most(widened, rhs)
    }

    /// A conflict among `by_id` (members sorted by record id), whose relations together with
    /// the declared set admit no point. Group `g` holds the declared constraints and the
    /// relations of the `g`th record by id, so every subset of records is tried together with
    /// the declared set.
    fn conflict(
        &self,
        by_id: &[usize],
        members: &[usize],
    ) -> Result<ChoiceConflict, SetLearningError> {
        let declared = self.declared_constraints(0);
        let dimension = self.declared().dimension();
        let mut constraints = Vec::new();
        let mut group = Vec::new();
        for (label, &m) in by_id.iter().enumerate() {
            for constraint in &declared {
                constraints.push(constraint.clone());
                group.push(label);
            }
            for relation in &self.choices[m].relations {
                let mut coefficients = vec![0.0; dimension];
                for (j, v) in &relation.terms {
                    if let Some(slot) = coefficients.get_mut(*j) {
                        *slot = *v;
                    }
                }
                constraints.push(LinearConstraint {
                    coefficients,
                    sense: ConstraintSense::AtLeast,
                    rhs: relation.bound,
                });
                group.push(label);
            }
        }
        let declared_set = self.declared();
        let lower: Vec<f64> = declared_set.parameters.iter().map(|p| p.lower).collect();
        let upper: Vec<f64> = declared_set.parameters.iter().map(|p| p.upper).collect();
        let lp = self.options.lp;
        let size = self.options.conflict_search_size;
        let (rows, labels, low, high) = (&constraints, &group, &lower, &upper);
        let smallest =
            if size == 0 || subsets_exceed(by_id.len(), size, SMALLEST_CONFLICT_SUBSET_BUDGET) {
                None
            } else {
                let searched = match self.ctx {
                    Some(ctx) => linear_program::smallest_infeasible_groups_with_context(
                        rows, labels, low, high, size, lp, ctx,
                    ),
                    None => linear_program::smallest_infeasible_groups(
                        rows, labels, low, high, size, lp,
                    ),
                };
                searched.map_err(lp_error)?
            };
        let (groups, kind) = if let Some(groups) = smallest {
            (groups, ConflictKind::Smallest)
        } else {
            let groups = match self.ctx {
                Some(ctx) => linear_program::irreducible_infeasible_groups_with_context(
                    rows, labels, low, high, lp, ctx,
                ),
                None => linear_program::irreducible_infeasible_groups(rows, labels, low, high, lp),
            }
            .map_err(lp_error)?
            .ok_or(SetLearningError::ProgramInconsistent { stage: "conflict_search" })?;
            (groups, ConflictKind::Irreducible)
        };
        let mut records: Vec<String> = groups
            .iter()
            .filter_map(|&g| by_id.get(g).map(|&m| self.choices[m].record_id.clone()))
            .collect();
        records.sort();
        records.dedup();
        Ok(ChoiceConflict { records, kind, tolerance_needed: self.tolerance_needed(members)? })
    }

    /// The smallest total tolerance at which every record of `members` is compatible with the
    /// declared set: the options' tolerance plus the least shared slack `s >= 0` with
    /// `sum_k theta_k (T_ck - T_jk) + s >= -tolerance - (F_c - F_j)` for every rejected option.
    fn tolerance_needed(&self, members: &[usize]) -> Result<f64, SetLearningError> {
        let tolerance = self.options.tolerance;
        let dimension = self.declared().dimension();
        let mut constraints = self.declared_constraints(1);
        for &m in members {
            for (coefficients, negative_gap) in &self.choices[m].slack_rows {
                let mut row = coefficients.clone();
                row.push(1.0);
                constraints.push(LinearConstraint {
                    coefficients: row,
                    sense: ConstraintSense::AtLeast,
                    rhs: -tolerance + negative_gap,
                });
            }
        }
        let mut lower: Vec<f64> = self.declared().parameters.iter().map(|p| p.lower).collect();
        let mut upper: Vec<f64> = self.declared().parameters.iter().map(|p| p.upper).collect();
        lower.push(0.0);
        upper.push(f64::INFINITY);
        let mut objective = vec![0.0; dimension];
        objective.push(1.0);
        let program = LinearProgram { objective, constraints, lower, upper };
        observe(self.ctx)?;
        match linear_program::solve(&program, self.options.lp).map_err(SetLearningError::Lp)? {
            LpOutcome::Optimal(solution) => Ok(tolerance + solution.objective.max(0.0)),
            LpOutcome::Infeasible(_) | LpOutcome::Unbounded { .. } => {
                Err(SetLearningError::ProgramInconsistent { stage: "tolerance_needed" })
            }
        }
    }
}

impl SetLearningError {
    /// Structured refusal for a failure to learn the preference set compatible with recorded
    /// choices.
    ///
    /// A value-model or choice-record refusal keeps its own refusal; a linear-program refusal
    /// is reported under `preference_set_learning.linear_program_refused` with the linear
    /// program's detail and message as the supplied text. Contradictory records are an
    /// outcome, not a refusal.
    #[must_use]
    pub fn to_refusal(&self) -> ExternalRefusal {
        let invalid = |remedy| {
            refusal(antecedent_core::reason_code!("invalid_argument"), LEARN, self.detail())
                .remedy(remedy)
        };
        let tolerance = "supply a finite, non-negative tolerance in utility units";
        match self {
            Self::Model(error) => at_stage(error.to_refusal(), LEARN),
            Self::Records(error) => at_stage(error.to_refusal(), LEARN),
            Self::Lp(error) => invalid(
                "relax SetLearningOptions::lp, or rescale the declared ranges and relations",
            )
            .supplied(format!("{}: {error}", error.detail()))
            .build(),
            Self::ToleranceNegative { tolerance: value } => {
                invalid(tolerance).supplied(value.to_string()).build()
            }
            Self::ToleranceNonFinite => invalid(tolerance).build(),
            Self::DeclaredSetEmpty => {
                invalid("widen the declared ranges or remove a declared relation").build()
            }
            Self::RelationIdCollision { id } => {
                invalid("rename the declared relation or the record whose choice relation collides")
                    .offending(id.clone())
                    .build()
            }
            Self::ProgramInconsistent { stage } => {
                invalid("tighten the feasibility tolerance in SetLearningOptions::lp")
                    .supplied(*stage)
                    .build()
            }
            Self::Cancelled => {
                refusal(antecedent_core::reason_code!("cancelled_no_claim"), LEARN, self.detail())
                    .build()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn differences_within_rounding_of_the_operands_are_zero() {
        assert!(difference(0.1 + 0.2, 0.3).abs() <= 0.0);
        assert!(difference(0.0, 0.0).abs() <= 0.0);
        assert!((difference(1.0, 0.5) - 0.5).abs() <= 0.0);
        assert!((difference(1e-20, 0.0) - 1e-20).abs() <= 0.0);
    }

    #[test]
    fn subset_counts_are_compared_with_the_budget() {
        // C(10,1) + C(10,2) + C(10,3) = 10 + 45 + 120 = 175.
        assert!(!subsets_exceed(10, 3, 175));
        assert!(subsets_exceed(10, 3, 174));
        assert!(!subsets_exceed(2, 5, 3));
        assert!(subsets_exceed(1000, 3, SMALLEST_CONFLICT_SUBSET_BUDGET));
    }

    #[test]
    fn choice_relation_ids_name_record_and_option() {
        assert_eq!(choice_relation_id("r7", "wait"), "choice.r7.wait");
    }
}
