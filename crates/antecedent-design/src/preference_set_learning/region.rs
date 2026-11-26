//! The compatible set under a value with shape parameters.
//!
//! The value is split into shape parameters `theta_S` and a part affine in the remaining free
//! parameters `theta_L` ([`crate::preference_model::ValueModel::shape_split`]):
//! `v = F(x; theta_S) + sum_k theta_k T_k(x; theta_S)`. A recorded choice of `c` over `j` is the
//! value relation `dF(theta_S) + sum_k theta_k dT_k(theta_S) >= -tolerance` with `dF` and `dT_k`
//! the differences of the expectations under the two options' recorded rows.
//!
//! Only the shape box is subdivided. On a shape box every `dF` and `dT_k` is enclosed with
//! outward-rounded interval arithmetic, `dF in [c, d]` and `dT_k in [a_k, b_k]`. Then
//! - if some `theta_S` of the box admits `theta_L`, `d + sum_k max(a_k theta_k, b_k theta_k) >=
//!   -tolerance` holds for that `theta_L` (the outer system);
//! - if `c + sum_k min(a_k theta_k, b_k theta_k) >= -tolerance` holds for some `theta_L`, that
//!   `theta_L` satisfies the relation for every `theta_S` of the box (the inner system).
//!
//! The inner system is linear with one auxiliary variable per non-degenerate `dT_k`
//! (`w <= a_k theta_k`, `w <= b_k theta_k`). The outer system is linear within each sign orthant
//! of `theta_L`: the orthants of up to [`ORTHANT_LIMIT`] linear parameters whose range crosses
//! zero are enumerated, and for any further such parameter `max(a theta, b theta)` is replaced
//! by its chord over the range, which lies above it, so the outer system stays a relaxation.
//! Declared linear relations take the shape and fixed parameters' part as an interval the same
//! way. A box is outside when the outer system is infeasible in every orthant, inside when the
//! inner system is feasible, and on the boundary otherwise; boundary boxes are split by the
//! certified rule (the widest shape parameter relative to its declared range, at its midpoint)
//! until every box is decided, the next boxes do not fit the execution or a caller box
//! budget, or every width is at most the minimum fraction of its declared range.
//!
//! What is proven: every shape value of an inside box admits linear parameters compatible with
//! every record; no shape value of a dropped box does. The records conflict exactly when every
//! box is dropped (a proof of emptiness); when no box is proven inside before the subdivision
//! stops, neither emptiness nor non-emptiness is certified and the outcome is undetermined. The
//! learned range of a shape parameter is the hull of the kept boxes; that of a linear parameter
//! is the hull of each kept box's outer linear-program ranges. Both are outer ranges.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet, BinaryHeap};

use antecedent_core::ExecutionContext;
use antecedent_stats::linear_program::{
    self, ConstraintSense, LinearConstraint, LinearProgram, LpOutcome,
};

use super::{
    ASSUMPTION_CHOICES_OPTIMAL, CellStatus, ChoiceConflict, ConflictKind, DeciderOverlap,
    DeciderSet, LearnedPreferenceSet, LearnedRegion, ParameterRange, RangeBasis, RegionCell,
    SetLearningError, SetLearningOptions, SetLearningResult, SetOutcome, UndeterminedSet,
    choice_relation_id, lp_error, observe, subsets_exceed,
};
use crate::choice_record::{ChoiceRecordSet, Pooling, ShapeData, shape_data};
use crate::preference_enclosure::{
    EncloseFailure, Interval, Queued, expectation, lexicographic, measure_box, split_box,
};
use crate::preference_model::{
    BeliefRows, PreferenceExpr, PreferenceSet, RelationOrigin, RelationSense, ValueModel,
    ValueRelation,
};

/// Largest number of linear parameters whose orthants are enumerated on one box; the outer
/// system of any further parameter whose range crosses zero uses the chord relaxation.
pub(super) const ORTHANT_LIMIT: usize = 6;

/// Largest number of record subsets the exhaustive smallest-conflict search classifies under a
/// value with shape parameters; above it the conflict is found by the deletion filter alone.
pub(super) const REGION_CONFLICT_SUBSET_BUDGET: usize = 256;

/// Largest number of shape points at which the tolerance that reconciles the records is sought.
const TOLERANCE_POINTS: usize = 64;

/// A value relation the learner works with: a recorded choice or a declared value relation.
struct Comparison {
    id: String,
    origin: RelationOrigin,
    chosen: BeliefRows,
    rejected: BeliefRows,
    tolerance: f64,
}

/// `dF` and `dT_k` (per linear position) of one comparison on a box, with its tolerance.
struct Gap {
    fixed: Interval,
    terms: Vec<Interval>,
    tolerance: f64,
}

/// Everything the linear programs of one box read.
struct BoxParts {
    gaps: Vec<Gap>,
    /// Per declared linear relation: the interval of its fixed and shape part.
    declared: Vec<Interval>,
}

/// What a box is.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Verdict {
    Inside,
    Boundary,
}

/// A kept box.
struct Cell {
    lower: Vec<f64>,
    upper: Vec<f64>,
    verdict: Verdict,
    /// `None` when an enclosure failed on the box.
    parts: Option<BoxParts>,
}

/// The kept boxes of one subdivision.
struct Subdivision {
    kept: Vec<Cell>,
    examined: usize,
    budget_exhausted: bool,
    /// Centres of examined boxes, in examination order, at most [`TOLERANCE_POINTS`].
    centres: Vec<Vec<f64>>,
}

/// Whether a group of comparisons admits a point.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Emptiness {
    Empty,
    NonEmpty,
    Undetermined,
}

impl Subdivision {
    fn emptiness(&self) -> Emptiness {
        if self.kept.is_empty() {
            Emptiness::Empty
        } else if self.kept.iter().any(|c| c.verdict == Verdict::Inside) {
            Emptiness::NonEmpty
        } else {
            Emptiness::Undetermined
        }
    }
}

struct RegionLearner<'a> {
    model: &'a ValueModel,
    data: ShapeData,
    options: SetLearningOptions,
    ctx: Option<&'a ExecutionContext>,
    /// Declared value relations, then every record's comparisons in record and option order.
    comparisons: Vec<Comparison>,
    /// Indices of the declared value relations in `comparisons`.
    declared_value: Vec<usize>,
    /// Comparisons of each record.
    per_record: Vec<Vec<usize>>,
    /// Declared `(lower, upper)` of each shape parameter, in shape order.
    shape_bounds: Vec<(f64, f64)>,
    /// Declared `(lower, upper)` of each linear parameter, in linear order.
    linear_bounds: Vec<(f64, f64)>,
}

/// Lease `boxes` live shape boxes. Each box stores two endpoints per parameter.
fn charge_boxes(
    quote: &mut antecedent_stats::execution_quote::WorkLease,
    ctx: Option<&ExecutionContext>,
    boxes: usize,
    dimension: usize,
) -> Result<(), antecedent_core::SearchReceipt> {
    let live = u64::try_from(boxes).unwrap_or(u64::MAX);
    let width = u64::try_from(dimension.max(1)).unwrap_or(u64::MAX);
    let state = live.saturating_mul(width).saturating_mul(16);
    quote.charge(ctx, live, state, "preference shape boxes")
}

/// Weighted rows of one available option, weights normalised to sum to one.
fn belief_rows(data: &ShapeData, record: usize, option: usize) -> BeliefRows {
    let rows = &data.records[record].options[option];
    BeliefRows { rows: rows.rows.concat(), weights: rows.normalised() }
}

/// Learn the compatible set of a model with shape parameters or value relations.
pub(super) fn learn(
    model: &ValueModel,
    records: &ChoiceRecordSet,
    options: SetLearningOptions,
    ctx: Option<&ExecutionContext>,
) -> Result<SetLearningResult, SetLearningError> {
    let data = shape_data(model, records).map_err(SetLearningError::Records)?;
    let model_identity = model.identity().map_err(SetLearningError::Model)?;
    let learner = RegionLearner::new(model, data, options, ctx)?;
    if learner.emptiness_of(&[])? == Emptiness::Empty {
        return Err(SetLearningError::DeclaredSetEmpty);
    }
    let all: Vec<usize> = (0..learner.data.records.len()).collect();
    let mut per_decider = Vec::new();
    for decider in records.deciders() {
        let members: Vec<usize> =
            all.iter().copied().filter(|&m| learner.data.records[m].decider == decider).collect();
        let subset = ChoiceRecordSet {
            attributes: records.attributes.clone(),
            population: records.population.clone(),
            records: records.records.iter().filter(|r| r.decider == decider).cloned().collect(),
            pooling: Pooling::PerDecider,
        };
        let outcome =
            learner.outcome(&model_identity, &members, Some(decider.as_str()), &subset)?;
        per_decider.push(DeciderSet { decider, n_records: members.len(), outcome });
    }
    let pooled = match records.pooling {
        Pooling::Pooled { .. } => Some(learner.outcome(&model_identity, &all, None, records)?),
        Pooling::PerDecider => None,
    };
    let overlap = if matches!(records.pooling, Pooling::PerDecider) && per_decider.len() > 1 {
        let common = learner.outcome(&model_identity, &all, None, records)?;
        let disagreeing = match &common {
            SetOutcome::Learned(_) | SetOutcome::Undetermined(_) => Vec::new(),
            SetOutcome::Contradictory(conflict) => {
                let named: BTreeSet<&str> = learner
                    .data
                    .records
                    .iter()
                    .filter(|r| conflict.records.contains(&r.record_id))
                    .map(|r| r.decider.as_str())
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

/// The chord of `max(a theta, b theta)` over `[lo, hi]` with `lo < 0 < hi` and `a < b`, as
/// `(slope, intercept)`; it lies on or above the maximum over the range.
fn chord(a: f64, b: f64, lo: f64, hi: f64) -> (f64, f64) {
    let at_lo = a * lo;
    let at_hi = b * hi;
    let slope = (at_hi - at_lo) / (hi - lo);
    (slope, at_lo - slope * lo)
}

/// Whether a constraint with no variable holds, within the linear-program tolerance.
fn constant_holds(constraint: &LinearConstraint, tolerance: f64) -> bool {
    let slack = tolerance * (1.0 + constraint.rhs.abs());
    match constraint.sense {
        ConstraintSense::AtLeast => 0.0 >= constraint.rhs - slack,
        ConstraintSense::AtMost => 0.0 <= constraint.rhs + slack,
        ConstraintSense::Equal => constraint.rhs.abs() <= slack,
    }
}

fn at_least(coefficients: Vec<f64>, rhs: f64) -> LinearConstraint {
    LinearConstraint { coefficients, sense: ConstraintSense::AtLeast, rhs }
}

fn at_most(coefficients: Vec<f64>, rhs: f64) -> LinearConstraint {
    LinearConstraint { coefficients, sense: ConstraintSense::AtMost, rhs }
}

/// Every `k`-subset of `0..n` in lexicographic order.
fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    if k == 0 || k > n {
        return out;
    }
    let mut current: Vec<usize> = (0..k).collect();
    loop {
        out.push(current.clone());
        let Some(i) = (0..k).rev().find(|&i| current[i] < n - k + i) else {
            return out;
        };
        current[i] += 1;
        for j in i + 1..k {
            current[j] = current[j - 1] + 1;
        }
    }
}

fn centre(lower: &[f64], upper: &[f64]) -> Vec<f64> {
    lower.iter().zip(upper).map(|(l, u)| 0.5 * l + 0.5 * u).collect()
}

impl<'a> RegionLearner<'a> {
    fn new(
        model: &'a ValueModel,
        data: ShapeData,
        options: SetLearningOptions,
        ctx: Option<&'a ExecutionContext>,
    ) -> Result<Self, SetLearningError> {
        let declared = &model.preferences;
        let mut ids: BTreeSet<String> = declared.relations.iter().map(|r| r.id.clone()).collect();
        let mut comparisons = Vec::new();
        let mut declared_value = Vec::new();
        for relation in &declared.value_relations {
            ids.insert(relation.id.clone());
            declared_value.push(comparisons.len());
            comparisons.push(Comparison {
                id: relation.id.clone(),
                origin: relation.origin.clone(),
                chosen: relation.chosen.clone(),
                rejected: relation.rejected.clone(),
                tolerance: relation.tolerance,
            });
        }
        let mut per_record = Vec::with_capacity(data.records.len());
        for (r, record) in data.records.iter().enumerate() {
            let mut own = Vec::new();
            for (o, option) in record.options.iter().enumerate() {
                if o == record.chosen {
                    continue;
                }
                let id = choice_relation_id(&record.record_id, &option.id);
                if !ids.insert(id.clone()) {
                    return Err(SetLearningError::RelationIdCollision { id });
                }
                own.push(comparisons.len());
                comparisons.push(Comparison {
                    id,
                    origin: RelationOrigin::Choice {
                        record_id: record.record_id.clone(),
                        rejected_option: option.id.clone(),
                    },
                    chosen: belief_rows(&data, r, record.chosen),
                    rejected: belief_rows(&data, r, o),
                    tolerance: options.tolerance,
                });
            }
            per_record.push(own);
        }
        let bounds = |indices: &[usize]| -> Vec<(f64, f64)> {
            indices
                .iter()
                .map(|j| (declared.parameters[*j].lower, declared.parameters[*j].upper))
                .collect()
        };
        let shape_bounds = bounds(&data.split.shape);
        let linear_bounds = bounds(&data.split.linear);
        Ok(Self {
            model,
            data,
            options,
            ctx,
            comparisons,
            declared_value,
            per_record,
            shape_bounds,
            linear_bounds,
        })
    }

    /// Comparisons of the declared value relations and of the records `members`.
    fn comparisons_of(&self, members: &[usize]) -> Vec<usize> {
        let mut out = self.declared_value.clone();
        for &m in members {
            out.extend(self.per_record[m].iter().copied());
        }
        out
    }

    /// Parameter intervals of a shape box: shape parameters over the box, fixed parameters at
    /// their value, linear parameters at zero (the parts never read them).
    fn box_parameters(&self, lower: &[f64], upper: &[f64]) -> Vec<Interval> {
        let mut params: Vec<Interval> = self
            .data
            .split
            .fixed
            .iter()
            .map(|value| Interval::point(value.unwrap_or(0.0)))
            .collect();
        for (position, j) in self.data.split.shape.iter().enumerate() {
            params[*j] = Interval { lo: lower[position], hi: upper[position] };
        }
        params
    }

    /// Enclosures of every gap and declared relation part on a box.
    fn box_parts(
        &self,
        comparisons: &[usize],
        lower: &[f64],
        upper: &[f64],
    ) -> Result<BoxParts, EncloseFailure> {
        let params = self.box_parameters(lower, upper);
        let form = &self.data.split.form;
        let difference =
            |part: &PreferenceExpr, c: &Comparison| -> Result<Interval, EncloseFailure> {
                let chosen = expectation(part, &c.chosen, &params)?;
                let rejected = expectation(part, &c.rejected, &params)?;
                if chosen.is_zero() && rejected.is_zero() {
                    Ok(Interval::point(0.0))
                } else {
                    chosen.minus(rejected)
                }
            };
        let mut gaps = Vec::with_capacity(comparisons.len());
        for &index in comparisons {
            let c = &self.comparisons[index];
            let fixed = difference(&form.fixed, c)?;
            let terms = self
                .data
                .split
                .linear
                .iter()
                .map(|j| difference(&form.terms[*j], c))
                .collect::<Result<Vec<_>, _>>()?;
            gaps.push(Gap { fixed, terms, tolerance: c.tolerance });
        }
        let mut declared = Vec::with_capacity(self.model.preferences.relations.len());
        for relation in &self.model.preferences.relations {
            let mut part = Interval::point(0.0);
            for (j, coefficient) in &relation.terms {
                if self.data.split.linear.contains(j) {
                    continue;
                }
                let parameter = params.get(*j).copied().ok_or(EncloseFailure::NonFinite)?;
                part = part.plus(parameter.scale(*coefficient)?)?;
            }
            declared.push(part);
        }
        Ok(BoxParts { gaps, declared })
    }

    /// Linear coefficients of a declared relation over the linear parameters.
    fn declared_coefficients(&self, relation: usize) -> Vec<f64> {
        let mut row = vec![0.0; self.data.split.linear.len()];
        for (j, coefficient) in &self.model.preferences.relations[relation].terms {
            if let Some(position) = self.data.split.linear.iter().position(|l| l == j) {
                row[position] += coefficient;
            }
        }
        row
    }

    /// Declared relations with their fixed and shape part at the end of its interval that makes
    /// them loosest (`outer`) or tightest (`!outer`), with `extra` zero columns appended.
    fn declared_constraints(
        &self,
        parts: &[Interval],
        outer: bool,
        extra: usize,
    ) -> Vec<LinearConstraint> {
        let mut constraints = Vec::new();
        for (r, relation) in self.model.preferences.relations.iter().enumerate() {
            let mut row = self.declared_coefficients(r);
            row.extend(std::iter::repeat_n(0.0, extra));
            let part = parts[r];
            let (loose_low, tight_low) = (relation.bound - part.hi, relation.bound - part.lo);
            let (loose_high, tight_high) = (relation.bound - part.lo, relation.bound - part.hi);
            let low = if outer { loose_low } else { tight_low };
            let high = if outer { loose_high } else { tight_high };
            match relation.sense {
                RelationSense::AtLeast => constraints.push(at_least(row, low)),
                RelationSense::AtMost => constraints.push(at_most(row, high)),
                RelationSense::Equal => {
                    constraints.push(at_least(row.clone(), low));
                    constraints.push(at_most(row, high));
                }
            }
        }
        constraints
    }

    /// The orthant bounds of the linear parameters on a box: up to [`ORTHANT_LIMIT`] parameters
    /// whose range crosses zero and whose term difference is not a single value are split at
    /// zero.
    fn orthants(&self, parts: &BoxParts) -> Vec<Vec<(f64, f64)>> {
        let crossing: Vec<usize> = (0..self.linear_bounds.len())
            .filter(|&k| {
                let (lo, hi) = self.linear_bounds[k];
                lo < 0.0
                    && hi > 0.0
                    && parts.gaps.iter().any(|g| g.terms[k].lo.total_cmp(&g.terms[k].hi).is_ne())
            })
            .take(ORTHANT_LIMIT)
            .collect();
        let count = 1_usize << crossing.len();
        (0..count)
            .map(|mask| {
                let mut bounds = self.linear_bounds.clone();
                for (bit, &k) in crossing.iter().enumerate() {
                    let (lo, hi) = bounds[k];
                    bounds[k] = if (mask >> bit) & 1 == 1 { (0.0, hi) } else { (lo, 0.0) };
                }
                bounds
            })
            .collect()
    }

    /// The outer system on one orthant: a relaxation of the set of linear parameters some shape
    /// value of the box admits.
    fn outer_constraints(&self, parts: &BoxParts, bounds: &[(f64, f64)]) -> Vec<LinearConstraint> {
        let mut constraints = self.declared_constraints(&parts.declared, true, 0);
        for gap in &parts.gaps {
            let mut row = Vec::with_capacity(gap.terms.len());
            let mut constant = gap.fixed.hi;
            for (term, (lo, hi)) in gap.terms.iter().zip(bounds) {
                let (a, b) = (term.lo, term.hi);
                if a.total_cmp(&b).is_eq() {
                    row.push(a);
                } else if *lo >= 0.0 {
                    row.push(b);
                } else if *hi <= 0.0 {
                    row.push(a);
                } else {
                    let (slope, intercept) = chord(a, b, *lo, *hi);
                    row.push(slope);
                    constant += intercept;
                }
            }
            constraints.push(at_least(row, -gap.tolerance - constant));
        }
        constraints
    }

    /// The inner system: the linear parameters, then one auxiliary variable per term difference
    /// that is not a single value. Feasible exactly when some linear parameters satisfy every
    /// relation for every shape value of the box.
    fn inner_system(&self, parts: &BoxParts) -> (Vec<LinearConstraint>, Vec<(f64, f64)>) {
        let n = self.linear_bounds.len();
        let auxiliary: Vec<(usize, usize)> = parts
            .gaps
            .iter()
            .enumerate()
            .flat_map(|(g, gap)| {
                gap.terms
                    .iter()
                    .enumerate()
                    .filter(|(_, t)| t.lo.total_cmp(&t.hi).is_ne())
                    .map(move |(k, _)| (g, k))
            })
            .collect();
        let width = n + auxiliary.len();
        let mut constraints = self.declared_constraints(&parts.declared, false, auxiliary.len());
        let mut bounds = self.linear_bounds.clone();
        for &(g, k) in &auxiliary {
            let (a, b) = (parts.gaps[g].terms[k].lo, parts.gaps[g].terms[k].hi);
            let (lo, hi) = self.linear_bounds[k];
            let corners = [a * lo, a * hi, b * lo, b * hi];
            let low = corners.iter().copied().fold(f64::INFINITY, f64::min);
            let high = corners.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            bounds.push((low, high));
        }
        for (g, gap) in parts.gaps.iter().enumerate() {
            let mut row = vec![0.0; width];
            for (k, term) in gap.terms.iter().enumerate() {
                match auxiliary.iter().position(|&(ag, ak)| ag == g && ak == k) {
                    Some(position) => row[n + position] = 1.0,
                    None => row[k] += term.lo,
                }
            }
            constraints.push(at_least(row, -gap.tolerance - gap.fixed.lo));
        }
        for (position, &(g, k)) in auxiliary.iter().enumerate() {
            let term = parts.gaps[g].terms[k];
            for slope in [term.lo, term.hi] {
                let mut row = vec![0.0; width];
                row[n + position] = 1.0;
                row[k] = -slope;
                constraints.push(at_most(row, 0.0));
            }
        }
        (constraints, bounds)
    }

    fn program(
        constraints: &[LinearConstraint],
        bounds: &[(f64, f64)],
        objective: Vec<f64>,
    ) -> LinearProgram {
        LinearProgram {
            objective,
            constraints: constraints.to_vec(),
            lower: bounds.iter().map(|b| b.0).collect(),
            upper: bounds.iter().map(|b| b.1).collect(),
        }
    }

    /// Whether the constraints and bounds admit a point.
    fn feasible(
        &self,
        constraints: &[LinearConstraint],
        bounds: &[(f64, f64)],
    ) -> Result<bool, SetLearningError> {
        let lp = self.options.lp;
        if bounds.is_empty() {
            return Ok(constraints.iter().all(|c| constant_holds(c, lp.feasibility_tolerance)));
        }
        observe(self.ctx)?;
        let program = Self::program(constraints, bounds, vec![0.0; bounds.len()]);
        let outcome = linear_program::solve(&program, lp).map_err(lp_error)?;
        Ok(!matches!(outcome, LpOutcome::Infeasible(_)))
    }

    /// Smallest and largest value of every variable, `None` when the system is empty.
    fn ranges(
        &self,
        constraints: &[LinearConstraint],
        bounds: &[(f64, f64)],
    ) -> Result<Option<Vec<(f64, f64)>>, SetLearningError> {
        if !self.feasible(constraints, bounds)? {
            return Ok(None);
        }
        let lp = self.options.lp;
        let mut out = Vec::with_capacity(bounds.len());
        for (k, (lo, hi)) in bounds.iter().enumerate() {
            let mut objective = vec![0.0; bounds.len()];
            objective[k] = 1.0;
            let program = Self::program(constraints, bounds, objective);
            let mut ends = [*lo, *hi];
            for (slot, maximise) in ends.iter_mut().zip([false, true]) {
                observe(self.ctx)?;
                let outcome = if maximise {
                    linear_program::maximize(&program, lp)
                } else {
                    linear_program::solve(&program, lp)
                }
                .map_err(lp_error)?;
                match outcome {
                    LpOutcome::Optimal(solution) => *slot = solution.objective.clamp(*lo, *hi),
                    LpOutcome::Infeasible(_) => return Ok(None),
                    LpOutcome::Unbounded { .. } => {
                        return Err(SetLearningError::ProgramInconsistent {
                            stage: "parameter_range",
                        });
                    }
                }
            }
            out.push((ends[0], ends[1].max(ends[0])));
        }
        Ok(Some(out))
    }

    /// The kept box for `lower..upper`, or `None` when the box is proven outside.
    fn examine(
        &self,
        comparisons: &[usize],
        lower: Vec<f64>,
        upper: Vec<f64>,
    ) -> Result<Option<Cell>, SetLearningError> {
        observe(self.ctx)?;
        let Ok(parts) = self.box_parts(comparisons, &lower, &upper) else {
            return Ok(Some(Cell { lower, upper, verdict: Verdict::Boundary, parts: None }));
        };
        let mut admits = false;
        for bounds in self.orthants(&parts) {
            if self.feasible(&self.outer_constraints(&parts, &bounds), &bounds)? {
                admits = true;
                break;
            }
        }
        if !admits {
            return Ok(None);
        }
        let (constraints, bounds) = self.inner_system(&parts);
        let verdict =
            if self.feasible(&constraints, &bounds)? { Verdict::Inside } else { Verdict::Boundary };
        Ok(Some(Cell { lower, upper, verdict, parts: Some(parts) }))
    }

    fn splittable(&self, cell: &Cell) -> bool {
        split_box(&self.shape_bounds, self.options.min_width_fraction, &cell.lower, &cell.upper)
            .is_some()
    }

    fn place(
        &self,
        cell: Cell,
        kept: &mut Vec<Cell>,
        slots: &mut Vec<Option<Cell>>,
        queue: &mut BinaryHeap<Queued>,
    ) {
        if cell.verdict == Verdict::Inside || !self.splittable(&cell) {
            kept.push(cell);
        } else {
            let (volume, _) = measure_box(&self.shape_bounds, &cell.lower, &cell.upper);
            queue.push(Queued { volume, lower: cell.lower.clone(), slot: slots.len() });
            slots.push(Some(cell));
        }
    }

    /// Cover the declared shape box for the comparisons `comparisons`.
    fn subdivide(&self, comparisons: &[usize]) -> Result<Subdivision, SetLearningError> {
        let lower: Vec<f64> = self.shape_bounds.iter().map(|b| b.0).collect();
        let upper: Vec<f64> = self.shape_bounds.iter().map(|b| b.1).collect();
        let mut kept = Vec::new();
        let mut slots: Vec<Option<Cell>> = Vec::new();
        let mut queue = BinaryHeap::new();
        let mut examined = 1;
        let mut budget_exhausted = false;
        let mut centres = vec![centre(&lower, &upper)];
        let mut quote = antecedent_stats::execution_quote::WorkLease::new();
        if let Some(root) = self.examine(comparisons, lower, upper)? {
            self.place(root, &mut kept, &mut slots, &mut queue);
        }
        while let Some(entry) = queue.pop() {
            observe(self.ctx)?;
            let Some(item) = slots[entry.slot].take() else {
                continue;
            };
            if examined + 2 > self.options.max_boxes {
                budget_exhausted = true;
                kept.push(item);
                continue;
            }
            if let Err(receipt) = charge_boxes(&mut quote, self.ctx, examined + 2, item.lower.len())
            {
                if receipt.stop == antecedent_core::SearchStop::Cancelled {
                    return Err(SetLearningError::Cancelled);
                }
                budget_exhausted = true;
                kept.push(item);
                continue;
            }
            let Some(halves) = split_box(
                &self.shape_bounds,
                self.options.min_width_fraction,
                &item.lower,
                &item.upper,
            ) else {
                kept.push(item);
                continue;
            };
            examined += 2;
            for (child_lower, child_upper) in halves {
                if centres.len() < TOLERANCE_POINTS {
                    centres.push(centre(&child_lower, &child_upper));
                }
                if let Some(child) = self.examine(comparisons, child_lower, child_upper)? {
                    self.place(child, &mut kept, &mut slots, &mut queue);
                }
            }
        }
        kept.sort_by(|a, b| lexicographic(&a.lower, &b.lower));
        Ok(Subdivision { kept, examined, budget_exhausted, centres })
    }

    /// The outer range of every linear parameter on a kept box: the hull over the orthants of
    /// the outer system's linear-program ranges; the declared ranges when an enclosure failed.
    fn linear_outer(&self, cell: &Cell) -> Result<Vec<(f64, f64)>, SetLearningError> {
        let Some(parts) = &cell.parts else {
            return Ok(self.linear_bounds.clone());
        };
        let mut hull: Option<Vec<(f64, f64)>> = None;
        for bounds in self.orthants(parts) {
            let constraints = self.outer_constraints(parts, &bounds);
            if let Some(found) = self.ranges(&constraints, &bounds)? {
                hull = Some(match hull {
                    None => found,
                    Some(so_far) => so_far
                        .iter()
                        .zip(&found)
                        .map(|(a, b)| (a.0.min(b.0), a.1.max(b.1)))
                        .collect(),
                });
            }
        }
        Ok(hull.unwrap_or_else(|| self.linear_bounds.clone()))
    }

    fn region(&self, subdivision: &Subdivision) -> Result<LearnedRegion, SetLearningError> {
        let mut cells = Vec::with_capacity(subdivision.kept.len());
        for cell in &subdivision.kept {
            cells.push(RegionCell {
                shape_lower: cell.lower.clone(),
                shape_upper: cell.upper.clone(),
                status: match cell.verdict {
                    Verdict::Inside => CellStatus::Inside,
                    Verdict::Boundary => CellStatus::Boundary,
                },
                linear_outer: self.linear_outer(cell)?,
            });
        }
        Ok(LearnedRegion {
            shape_parameters: self.data.split.shape.clone(),
            linear_parameters: self.data.split.linear.clone(),
            cells,
            boxes_used: subdivision.examined,
            min_width_fraction: self.options.min_width_fraction,
            budget_exhausted: subdivision.budget_exhausted,
        })
    }

    /// Whether some record of `members` varies parameter `j`.
    fn varied(&self, j: usize, members: &[usize]) -> bool {
        if self.data.split.shape.contains(&j) {
            self.data.varies(&self.data.attributes_moving_with(j), members)
        } else if self.data.split.linear.contains(&j) {
            self.data.varies(&self.data.term_attributes(j), members)
        } else {
            false
        }
    }

    fn ranges_of(&self, region: &LearnedRegion, members: &[usize]) -> Vec<ParameterRange> {
        let slack = self.options.lp.feasibility_tolerance;
        let hull = |pick: &dyn Fn(&RegionCell) -> (f64, f64)| -> Option<(f64, f64)> {
            region.cells.iter().map(pick).reduce(|a, b| (a.0.min(b.0), a.1.max(b.1)))
        };
        self.model
            .preferences
            .parameters
            .iter()
            .enumerate()
            .map(|(j, parameter)| {
                let declared = (parameter.lower, parameter.upper);
                let found = if let Some(p) = self.data.split.shape.iter().position(|s| *s == j) {
                    hull(&|c: &RegionCell| (c.shape_lower[p], c.shape_upper[p]))
                } else if let Some(p) = self.data.split.linear.iter().position(|l| *l == j) {
                    hull(&|c: &RegionCell| c.linear_outer[p])
                } else {
                    None
                };
                let learned = found.unwrap_or(declared);
                let narrowed = learned.0 > declared.0 + slack * (1.0 + declared.0.abs())
                    || learned.1 < declared.1 - slack * (1.0 + declared.1.abs());
                ParameterRange {
                    parameter_id: parameter.id.clone(),
                    declared,
                    learned,
                    varied_in_records: self.varied(j, members),
                    narrowed,
                    basis: RangeBasis::Enclosure,
                }
            })
            .collect()
    }

    /// The outcome of the records `members`; `subset` holds exactly those records.
    fn outcome(
        &self,
        model_identity: &str,
        members: &[usize],
        decider: Option<&str>,
        subset: &ChoiceRecordSet,
    ) -> Result<SetOutcome, SetLearningError> {
        let comparisons = self.comparisons_of(members);
        let subdivision = self.subdivide(&comparisons)?;
        match subdivision.emptiness() {
            Emptiness::Empty => {
                Ok(SetOutcome::Contradictory(self.conflict(members, &subdivision.centres)?))
            }
            Emptiness::Undetermined => Ok(SetOutcome::Undetermined(UndeterminedSet {
                model_identity: model_identity.to_owned(),
                records_digest: subset.digest().map_err(SetLearningError::Records)?,
                decider: decider.map(str::to_owned),
                n_records: members.len(),
                tolerance: self.options.tolerance,
                region: self.region(&subdivision)?,
            })),
            Emptiness::NonEmpty => {
                let region = self.region(&subdivision)?;
                let declared = &self.model.preferences;
                let mut value_relations = declared.value_relations.clone();
                for &m in members {
                    for &c in &self.per_record[m] {
                        let comparison = &self.comparisons[c];
                        value_relations.push(ValueRelation {
                            id: comparison.id.clone(),
                            origin: comparison.origin.clone(),
                            chosen: comparison.chosen.clone(),
                            rejected: comparison.rejected.clone(),
                            tolerance: comparison.tolerance,
                        });
                    }
                }
                let set = PreferenceSet {
                    parameters: declared.parameters.clone(),
                    relations: declared.relations.clone(),
                    value_relations,
                };
                set.validate().map_err(SetLearningError::Model)?;
                let ranges = self.ranges_of(&region, members);
                let mut assumptions: BTreeSet<String> =
                    set.parameters.iter().map(|p| p.assumption_id.clone()).collect();
                assumptions.extend(subset.assumptions());
                assumptions.insert(ASSUMPTION_CHOICES_OPTIMAL.to_owned());
                Ok(SetOutcome::Learned(LearnedPreferenceSet {
                    model_identity: model_identity.to_owned(),
                    records_digest: subset.digest().map_err(SetLearningError::Records)?,
                    decider: decider.map(str::to_owned),
                    set,
                    ranges,
                    binding: Vec::new(),
                    tolerance: self.options.tolerance,
                    n_records: members.len(),
                    assumptions: assumptions.into_iter().collect(),
                    region: Some(region),
                }))
            }
        }
    }

    fn emptiness_of(&self, records: &[usize]) -> Result<Emptiness, SetLearningError> {
        Ok(self.subdivide(&self.comparisons_of(records))?.emptiness())
    }

    /// A group of records proven to conflict with the declared set: the first group of the
    /// smallest size the exhaustive search finds, else the deletion filter's group. The kind is
    /// [`ConflictKind::Smallest`] or [`ConflictKind::Irreducible`] only when every smaller group,
    /// or every group with one record dropped, was proven to admit a point.
    fn conflict(
        &self,
        members: &[usize],
        centres: &[Vec<f64>],
    ) -> Result<ChoiceConflict, SetLearningError> {
        let mut by_id = members.to_vec();
        by_id.sort_by(|a, b| self.data.records[*a].record_id.cmp(&self.data.records[*b].record_id));
        let tolerance_needed = self.tolerance_needed(members, centres)?;
        let named = |group: &[usize]| -> Vec<String> {
            let mut ids: Vec<String> =
                group.iter().map(|m| self.data.records[*m].record_id.clone()).collect();
            ids.sort();
            ids.dedup();
            ids
        };
        let size = self.options.conflict_search_size;
        if size > 0 && !subsets_exceed(by_id.len(), size, REGION_CONFLICT_SUBSET_BUDGET) {
            let mut smaller_proven = true;
            for k in 1..=size.min(by_id.len()) {
                let mut all_admit = true;
                for combination in combinations(by_id.len(), k) {
                    let group: Vec<usize> = combination.iter().map(|&i| by_id[i]).collect();
                    match self.emptiness_of(&group)? {
                        Emptiness::Empty => {
                            let kind = if smaller_proven {
                                ConflictKind::Smallest
                            } else {
                                ConflictKind::NotMinimised
                            };
                            return Ok(ChoiceConflict {
                                records: named(&group),
                                kind,
                                tolerance_needed,
                            });
                        }
                        Emptiness::NonEmpty => {}
                        Emptiness::Undetermined => all_admit = false,
                    }
                }
                smaller_proven &= all_admit;
            }
        }
        let mut current = by_id.clone();
        let mut proven: BTreeMap<usize, bool> = BTreeMap::new();
        for &m in &by_id {
            let candidate: Vec<usize> = current.iter().copied().filter(|&x| x != m).collect();
            match self.emptiness_of(&candidate)? {
                Emptiness::Empty => current = candidate,
                Emptiness::NonEmpty => {
                    proven.insert(m, true);
                }
                Emptiness::Undetermined => {
                    proven.insert(m, false);
                }
            }
        }
        let kind = if current.iter().all(|m| proven.get(m).copied().unwrap_or(false)) {
            ConflictKind::Irreducible
        } else {
            ConflictKind::NotMinimised
        };
        Ok(ChoiceConflict { records: named(&current), kind, tolerance_needed })
    }

    /// `E_chosen[part] - E_rejected[part]` at a point, `None` where it cannot be evaluated.
    fn point_difference(
        part: &PreferenceExpr,
        comparison: &Comparison,
        point: &[f64],
    ) -> Option<f64> {
        let mean = |belief: &BeliefRows| -> Option<f64> {
            let width = belief.width();
            let mut total = 0.0;
            for (row, weight) in belief.rows.chunks_exact(width.max(1)).zip(&belief.weights) {
                total += weight * part.evaluate(row, point).ok()?;
            }
            Some(total)
        };
        let gap = mean(&comparison.chosen)? - mean(&comparison.rejected)?;
        gap.is_finite().then_some(gap)
    }

    /// A tolerance at which every record of `members` is compatible with the declared set: the
    /// options' tolerance plus the least shared slack, minimised over the linear parameters at
    /// each of the examined shape points and then over those points; infinite when no examined
    /// point admits the declared relations.
    fn tolerance_needed(
        &self,
        members: &[usize],
        centres: &[Vec<f64>],
    ) -> Result<f64, SetLearningError> {
        let comparisons = self.comparisons_of(members);
        let form = &self.data.split.form;
        let n = self.linear_bounds.len();
        let mut best: Option<f64> = None;
        for shape in centres {
            let mut point: Vec<f64> =
                self.data.split.fixed.iter().map(|value| value.unwrap_or(0.0)).collect();
            for (position, j) in self.data.split.shape.iter().enumerate() {
                point[*j] = shape[position];
            }
            let mut constraints = Vec::new();
            let mut evaluable = true;
            for &index in &comparisons {
                let comparison = &self.comparisons[index];
                let Some(fixed) = Self::point_difference(&form.fixed, comparison, &point) else {
                    evaluable = false;
                    break;
                };
                let mut row = Vec::with_capacity(n + 1);
                for j in &self.data.split.linear {
                    match Self::point_difference(&form.terms[*j], comparison, &point) {
                        Some(value) => row.push(value),
                        None => evaluable = false,
                    }
                }
                row.push(1.0);
                constraints.push(at_least(row, -comparison.tolerance - fixed));
            }
            if !evaluable {
                continue;
            }
            let parts: Vec<Interval> = self
                .model
                .preferences
                .relations
                .iter()
                .map(|relation| {
                    let value: f64 = relation
                        .terms
                        .iter()
                        .filter(|(j, _)| !self.data.split.linear.contains(j))
                        .map(|(j, coefficient)| coefficient * point.get(*j).copied().unwrap_or(0.0))
                        .sum();
                    Interval::point(value)
                })
                .collect();
            constraints.extend(self.declared_constraints(&parts, true, 1));
            let mut bounds = self.linear_bounds.clone();
            bounds.push((0.0, f64::INFINITY));
            let mut objective = vec![0.0; n];
            objective.push(1.0);
            observe(self.ctx)?;
            let program = Self::program(&constraints, &bounds, objective);
            if let LpOutcome::Optimal(solution) =
                linear_program::solve(&program, self.options.lp).map_err(lp_error)?
            {
                let slack = solution.objective.max(0.0);
                best = Some(best.map_or(slack, |b: f64| b.min(slack)));
            }
        }
        Ok(self.options.tolerance + best.unwrap_or(f64::INFINITY))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combinations_are_listed_in_lexicographic_order() {
        assert_eq!(
            combinations(4, 2),
            vec![vec![0, 1], vec![0, 2], vec![0, 3], vec![1, 2], vec![1, 3], vec![2, 3]]
        );
        assert!(combinations(2, 3).is_empty());
        assert_eq!(combinations(3, 3), vec![vec![0, 1, 2]]);
    }

    #[test]
    fn the_chord_lies_above_the_maximum_over_the_range() {
        let (a, b, lo, hi) = (-1.5, 2.0, -3.0, 4.0);
        let (slope, intercept) = chord(a, b, lo, hi);
        for i in 0..=70 {
            let theta = lo + (hi - lo) * f64::from(i) / 70.0;
            let maximum = (a * theta).max(b * theta);
            assert!(slope * theta + intercept >= maximum - 1e-12);
        }
        assert!((slope * lo + intercept - a * lo).abs() <= 1e-12);
        assert!((slope * hi + intercept - b * hi).abs() <= 1e-12);
    }
}
