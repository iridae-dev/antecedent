//! Enclosure scoring for the certified route: the criterion conventions, the per-draw
//! attributes and enclosed linear-form parts of each source, the hard-constraint exclusions, the
//! equality relations solved for one parameter, and the engine that encloses every admissible
//! action's criterion and every pairwise margin on a box of parameter values.

// SPDX-License-Identifier: MIT OR Apache-2.0

#[allow(clippy::wildcard_imports, reason = "a part of its parent module")]
use super::*;

// ---------------------------------------------------------------------------------------------
// Criterion conventions
// ---------------------------------------------------------------------------------------------

/// `+1` when larger criterion values are better, `-1` when smaller are.
pub(super) const fn orientation(criterion: DecisionCriterion) -> f64 {
    match criterion {
        DecisionCriterion::PosteriorExpectedLoss
        | DecisionCriterion::ExpectedRegret
        | DecisionCriterion::Regret => -1.0,
        _ => 1.0,
    }
}

/// Criteria ranked through expected-utility margins.
pub(super) const fn is_expectation(criterion: DecisionCriterion) -> bool {
    matches!(
        criterion,
        DecisionCriterion::PosteriorExpectedUtility
            | DecisionCriterion::PosteriorExpectedLoss
            | DecisionCriterion::ExpectedRegret
    )
}

/// An interval in the criterion's own orientation from one in the larger-is-better
/// orientation.
pub(super) fn oriented(sign: f64, score: (f64, f64)) -> (f64, f64) {
    if sign > 0.0 { score } else { (-score.1, -score.0) }
}

// ---------------------------------------------------------------------------------------------
// Sources
// ---------------------------------------------------------------------------------------------

/// Parameter-free parts of the linear form, enclosed per draw and in expectation.
pub(super) struct LinearData {
    /// `fixed[a][i]`.
    pub(super) fixed: Vec<Vec<Interval>>,
    /// `terms[a][i][j]`.
    pub(super) terms: Vec<Vec<Vec<Interval>>>,
    /// Expected fixed part per action.
    pub(super) mean_fixed: Vec<Interval>,
    /// Expected term per action and parameter.
    pub(super) mean_terms: Vec<Vec<Interval>>,
}

impl LinearData {
    pub(super) fn draw_value(
        &self,
        a: usize,
        i: usize,
        params: &[Interval],
    ) -> Result<Interval, EncloseFailure> {
        affine(self.fixed[a][i], &self.terms[a][i], params)
    }

    pub(super) fn draw_difference(
        &self,
        b: usize,
        a: usize,
        i: usize,
        params: &[Interval],
    ) -> Result<Interval, EncloseFailure> {
        let constant = self.fixed[b][i].minus(self.fixed[a][i])?;
        let coefficients = differences(&self.terms[b][i], &self.terms[a][i])?;
        affine(constant, &coefficients, params)
    }

    pub(super) fn mean(&self, a: usize, params: &[Interval]) -> Result<Interval, EncloseFailure> {
        affine(self.mean_fixed[a], &self.mean_terms[a], params)
    }

    pub(super) fn mean_difference(
        &self,
        a: usize,
        b: usize,
        params: &[Interval],
    ) -> Result<Interval, EncloseFailure> {
        let constant = self.mean_fixed[a].minus(self.mean_fixed[b])?;
        let coefficients = differences(&self.mean_terms[a], &self.mean_terms[b])?;
        affine(constant, &coefficients, params)
    }
}

pub(super) fn differences(
    left: &[Interval],
    right: &[Interval],
) -> Result<Vec<Interval>, EncloseFailure> {
    left.iter()
        .zip(right)
        .map(
            |(l, r)| {
                if l.is_zero() && r.is_zero() { Ok(Interval::point(0.0)) } else { l.minus(*r) }
            },
        )
        .collect()
}

/// One source's draws, read once.
pub(super) struct SourceData<'s> {
    pub(super) artifact: &'s DistributionArtifact,
    /// `attributes[a][i][k]`: attribute `k` of action `a` on draw `i`.
    pub(super) attributes: Vec<Vec<Vec<f64>>>,
    /// Normalized draw weights.
    pub(super) weights: Vec<f64>,
    /// Hard-constraint exclusions per action.
    pub(super) exclusions: Vec<Vec<ConstraintExclusion>>,
    pub(super) linear: Option<LinearData>,
    pub(super) receipt: SourceReceipt,
}

/// Read every action's attributes per draw, the constraint exclusions and, when the model has
/// a linear form, the enclosed parts.
pub(super) fn prepare_source<'s>(
    contract: &PreferenceContract,
    artifact: &'s DistributionArtifact,
) -> Result<SourceData<'s>, PreferenceCertifiedError> {
    use PreferenceCertifiedError as E;
    let probe = contract.probe_contract().map_err(E::Contract)?;
    let draws = utility_draws(&probe.contract, artifact).map_err(E::Eval)?;
    let n_draws = draws.weights.len();
    let mut attributes: Vec<Vec<Vec<f64>>> = Vec::with_capacity(contract.actions.len());
    for (a, action) in contract.actions.iter().enumerate() {
        let mut columns = Vec::with_capacity(action.inputs.len());
        for (k, input) in action.inputs.iter().enumerate() {
            let column = match (input, probe.probe_index[a][k]) {
                (ActionInput::Fixed { value, .. }, _) => vec![*value; n_draws],
                (ActionInput::Quantity(_), Some(p)) => draws.utilities[p].clone(),
                (ActionInput::Quantity(_), None) => {
                    return Err(E::Eval(DecisionEvalError::QuantityNotFound {
                        action: action.id.clone(),
                        input: k,
                    }));
                }
            };
            columns.push(column);
        }
        let rows: Vec<Vec<f64>> =
            (0..n_draws).map(|i| columns.iter().map(|c| c[i]).collect()).collect();
        attributes.push(rows);
    }
    let identity = &artifact.metadata().identity;
    let receipt = SourceReceipt {
        provider_id: identity.provider_id.clone(),
        snapshot_id: identity.snapshot_id.clone(),
        rng_id: identity.rng_id.clone(),
        causal_contract_id: identity.causal_contract_id.clone(),
    };
    let exclusions = read_exclusions(contract, artifact, &attributes, &draws.weights)?;
    let linear = match contract.model.linear_form() {
        Some(form) => Some(linear_data(contract, &form, &attributes, &draws.weights)?),
        None => None,
    };
    Ok(SourceData { artifact, attributes, weights: draws.weights, exclusions, linear, receipt })
}

/// Enclose the parts of the linear form on every draw and in expectation.
pub(super) fn linear_data(
    contract: &PreferenceContract,
    form: &LinearForm,
    attributes: &[Vec<Vec<f64>>],
    weights: &[f64],
) -> Result<LinearData, PreferenceCertifiedError> {
    let bounds = contract.model.preferences.bounds();
    let lower: Vec<f64> = bounds.iter().map(|b| b.0).collect();
    let upper: Vec<f64> = bounds.iter().map(|b| b.1).collect();
    let mut data = LinearData {
        fixed: Vec::new(),
        terms: Vec::new(),
        mean_fixed: Vec::new(),
        mean_terms: Vec::new(),
    };
    for (action, rows) in contract.actions.iter().zip(attributes) {
        let fail = |failure| box_refusal(failure, &action.id, &lower, &upper);
        let mut fixed = Vec::with_capacity(rows.len());
        let mut terms = Vec::with_capacity(rows.len());
        for row in rows {
            fixed.push(enclose(&form.fixed, row, &[]).map_err(fail)?);
            terms.push(
                form.terms
                    .iter()
                    .map(|term| enclose(term, row, &[]))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(fail)?,
            );
        }
        let mean_fixed =
            weighted_sum(weights.iter().copied().zip(fixed.iter().copied())).map_err(fail)?;
        let mean_terms = (0..form.terms.len())
            .map(|j| weighted_sum(weights.iter().copied().zip(terms.iter().map(|t| t[j]))))
            .collect::<Result<Vec<_>, _>>()
            .map_err(fail)?;
        data.fixed.push(fixed);
        data.terms.push(terms);
        data.mean_fixed.push(mean_fixed);
        data.mean_terms.push(mean_terms);
    }
    Ok(data)
}

pub(super) fn box_refusal(
    failure: EncloseFailure,
    subject: &str,
    lower: &[f64],
    upper: &[f64],
) -> PreferenceCertifiedError {
    match failure {
        EncloseFailure::LnDomain => PreferenceCertifiedError::LogarithmDomain {
            action: subject.to_owned(),
            lower: lower.to_vec(),
            upper: upper.to_vec(),
        },
        EncloseFailure::NonFinite => PreferenceCertifiedError::NonFiniteEnclosure {
            subject: subject.to_owned(),
            lower: lower.to_vec(),
            upper: upper.to_vec(),
        },
    }
}

/// Exclusions per action. Constraints never read parameters, so the ordinary contract bound at
/// the declared midpoint gives them when its value can be lowered; otherwise each constraint is
/// evaluated on the drawn attributes.
pub(super) fn read_exclusions(
    contract: &PreferenceContract,
    artifact: &DistributionArtifact,
    attributes: &[Vec<Vec<f64>>],
    weights: &[f64],
) -> Result<Vec<Vec<ConstraintExclusion>>, PreferenceCertifiedError> {
    use PreferenceCertifiedError as E;
    if contract.constraints.is_empty() {
        return Ok(vec![Vec::new(); contract.actions.len()]);
    }
    match contract.bind(&contract.model.preferences.midpoint()) {
        Ok(mut bound) => {
            bound.criterion = DecisionCriterion::PosteriorExpectedUtility;
            let result = evaluate_contract(&bound, artifact).map_err(E::Eval)?;
            contract
                .declared_exclusions(
                    result.actions.into_iter().map(|outcome| outcome.exclusions).collect(),
                )
                .map_err(E::Contract)
        }
        Err(PreferenceContractError::NotLowerable { .. }) => {
            own_exclusions(contract, attributes, weights)
        }
        Err(other) => Err(E::Contract(other)),
    }
}

/// Constraint exclusions evaluated on the drawn attributes, under the declared constraint ids.
pub(super) fn own_exclusions(
    contract: &PreferenceContract,
    attributes: &[Vec<Vec<f64>>],
    weights: &[f64],
) -> Result<Vec<Vec<ConstraintExclusion>>, PreferenceCertifiedError> {
    let mut exclusions = vec![Vec::new(); contract.actions.len()];
    for constraint in &contract.constraints {
        let applies: Vec<usize> = (0..contract.actions.len())
            .filter(|a| {
                constraint.applies_to.is_empty()
                    || constraint.applies_to.contains(&contract.actions[*a].id)
            })
            .collect();
        for a in applies {
            let action = &contract.actions[a];
            let mut held = 0.0;
            let mut positive_violation = false;
            for (row, weight) in attributes[a].iter().zip(weights) {
                let value = constraint.expr.evaluate(row).map_err(|error| {
                    PreferenceCertifiedError::Eval(DecisionEvalError::Contract(error))
                })?;
                if !value.is_finite() {
                    return Err(PreferenceCertifiedError::Eval(
                        DecisionEvalError::NonFiniteUtility { action: action.id.clone() },
                    ));
                }
                if value <= constraint.bound {
                    held += weight;
                } else if *weight > 0.0 {
                    positive_violation = true;
                }
            }
            if constraint_fails(constraint.min_probability, held, positive_violation) {
                exclusions[a].push(ConstraintExclusion {
                    constraint_id: constraint.id.clone(),
                    probability: held,
                    required: constraint.min_probability,
                });
            }
        }
    }
    Ok(exclusions)
}

// ---------------------------------------------------------------------------------------------
// Equality relations
// ---------------------------------------------------------------------------------------------

/// An equality relation solved for one parameter:
/// `theta_pivot = (bound - sum_k c_k theta_k) / coefficient`, every `k` a parameter that is
/// subdivided.
#[derive(Clone, Debug)]
pub(super) struct Elimination {
    pub(super) pivot: usize,
    pub(super) coefficient: f64,
    pub(super) others: Vec<(usize, f64)>,
    pub(super) bound: f64,
}

impl Elimination {
    /// Outward-rounded enclosure of the solved parameter over the intervals of the others.
    pub(super) fn derive(&self, params: &[Interval]) -> Result<Interval, EncloseFailure> {
        let mut sum = Interval::point(0.0);
        for (k, c) in &self.others {
            let value = params.get(*k).copied().ok_or(EncloseFailure::NonFinite)?;
            sum = sum.plus(value.scale(*c)?)?;
        }
        let reciprocal = 1.0 / self.coefficient;
        Interval::point(self.bound).minus(sum)?.times(Interval::rounded(reciprocal, reciprocal)?)
    }

    /// The solved parameter at a point of the others.
    pub(super) fn at(&self, point: &[f64]) -> f64 {
        let sum: f64 =
            self.others.iter().map(|(k, c)| c * point.get(*k).copied().unwrap_or(0.0)).sum();
        (self.bound - sum) / self.coefficient
    }
}

/// How the subdivision reads the relations of a set.
pub(super) struct RelationPlan {
    /// Relations checked on each box: every relation not solved, then the declared range of each
    /// solved parameter as two relations.
    pub(super) relations: Vec<LinearRelation>,
    /// Solved equalities, in the order they are applied.
    pub(super) eliminations: Vec<Elimination>,
    /// The subdivided box: the declared ranges, a solved parameter held at the lower end of its
    /// range with zero width (it contributes no dimension).
    pub(super) subdivided: Vec<(f64, f64)>,
}

/// Solve equality relations for one parameter each, so the subdivision runs over the set's own
/// dimensions and a box can lie wholly inside an equality. Equalities are taken in order of
/// their ids. One is solved when none of its parameters has been solved for and it has a
/// parameter of positive declared width that no solved equality reads; the pivot is the one with
/// the largest `|coefficient| * width` (the lowest index on ties). The pivot's declared range
/// becomes two relations on its solution. Any other equality stays a relation checked per box.
pub(super) fn relation_plan(set: &PreferenceSet) -> RelationPlan {
    let declared = set.bounds();
    let width = |j: usize| declared.get(j).map_or(0.0, |(l, u)| u - l);
    let mut order: Vec<usize> = (0..set.relations.len())
        .filter(|i| set.relations[*i].sense == RelationSense::Equal)
        .collect();
    order.sort_by(|a, b| set.relations[*a].id.cmp(&set.relations[*b].id));
    let mut solved = vec![false; set.relations.len()];
    let mut pivots: Vec<usize> = Vec::new();
    let mut read: Vec<usize> = Vec::new();
    let mut eliminations = Vec::new();
    for index in order {
        let relation = &set.relations[index];
        if relation.terms.iter().any(|(j, _)| pivots.contains(j)) {
            continue;
        }
        let mut pivot: Option<(usize, f64, f64)> = None;
        for &(j, c) in &relation.terms {
            let score = c.abs() * width(j);
            if score.is_nan() || score <= 0.0 || read.contains(&j) {
                continue;
            }
            let better = pivot.is_none_or(|(p, _, best)| {
                score > best || (score.total_cmp(&best).is_eq() && j < p)
            });
            if better {
                pivot = Some((j, c, score));
            }
        }
        let Some((p, coefficient, _)) = pivot else { continue };
        solved[index] = true;
        pivots.push(p);
        read.extend(relation.terms.iter().map(|(j, _)| *j));
        eliminations.push(Elimination {
            pivot: p,
            coefficient,
            others: relation.terms.iter().filter(|(j, _)| *j != p).copied().collect(),
            bound: relation.bound,
        });
    }
    let mut relations: Vec<LinearRelation> = set
        .relations
        .iter()
        .zip(&solved)
        .filter(|(_, solved)| !**solved)
        .map(|(relation, _)| relation.clone())
        .collect();
    let mut subdivided = declared.clone();
    for elimination in &eliminations {
        let p = elimination.pivot;
        let (lower, upper) = declared[p];
        let id = &set.parameters[p].id;
        for (sense, bound, end) in
            [(RelationSense::AtLeast, lower, "lower"), (RelationSense::AtMost, upper, "upper")]
        {
            relations.push(LinearRelation {
                id: format!("{id}.{end}"),
                terms: vec![(p, 1.0)],
                sense,
                bound,
                origin: RelationOrigin::Declared,
            });
        }
        subdivided[p] = (lower, lower);
    }
    RelationPlan { relations, eliminations, subdivided }
}

// ---------------------------------------------------------------------------------------------
// Box evaluation
// ---------------------------------------------------------------------------------------------

/// How per-structure enclosures combine.
pub(super) enum Combination {
    /// One source.
    Single,
    /// The worst over structures.
    Worst,
    /// The probability-weighted sum over structures, not renormalized.
    Bayes(Vec<f64>),
}

/// Score enclosures (larger is better) and pairwise margins on one box.
#[derive(Clone, Debug)]
pub(super) struct BoxEval {
    /// `scores[a]`; `None` for an excluded action.
    pub(super) scores: Vec<Option<Interval>>,
    /// `margins[a][b]` encloses `score_a - score_b`; `None` unless both are admissible and
    /// distinct.
    pub(super) margins: Vec<Vec<Option<Interval>>>,
}

/// One examined box that was kept.
pub(super) struct Examined {
    pub(super) lower: Vec<f64>,
    pub(super) upper: Vec<f64>,
    pub(super) boundary: bool,
    /// Some value relation is undecided on the box: its enclosure need not contain a point of
    /// the set, so a certified action does not stop the box from being split.
    pub(super) value_boundary: bool,
    /// Every parameter's interval on the box, solved parameters included.
    pub(super) params: Vec<Interval>,
    pub(super) eval: BoxEval,
    pub(super) best: Option<usize>,
    pub(super) candidates: Vec<usize>,
    pub(super) volume: f64,
}

pub(super) struct Engine<'a> {
    pub(super) contract: &'a PreferenceContract,
    pub(super) sources: &'a [SourceData<'a>],
    pub(super) combination: Combination,
    pub(super) admissible: Vec<bool>,
    pub(super) options: CertifiedOptions,
    /// `(lower, upper)` per parameter of the subdivided box: the declared range, or a zero-width
    /// range at its lower end for a parameter solved from an equality.
    pub(super) declared: Vec<(f64, f64)>,
    /// Declared `(lower, upper)` per parameter.
    pub(super) ranges: Vec<(f64, f64)>,
    /// Linear relations checked on each box.
    pub(super) relations: Vec<LinearRelation>,
    /// Equalities solved for one parameter each, applied in order.
    pub(super) eliminations: Vec<Elimination>,
    /// A point of the declared ranges and linear relations.
    pub(super) interior: &'a [f64],
    /// The caller's execution context, when the run is cancellable.
    pub(super) ctx: Option<&'a ExecutionContext>,
}

impl Engine<'_> {
    /// `Err(Cancelled)` once the caller's token is set.
    pub(super) fn observe(&self) -> Result<(), PreferenceCertifiedError> {
        observe(self.ctx)
    }

    pub(super) fn criterion(&self) -> DecisionCriterion {
        self.contract.criterion
    }

    pub(super) fn admissible_indices(&self) -> Vec<usize> {
        (0..self.admissible.len()).filter(|a| self.admissible[*a]).collect()
    }

    /// Where the box sits relative to the relations, and whether some value relation is
    /// undecided on it.
    pub(super) fn relation_fit(
        &self,
        params: &[Interval],
        lower: &[f64],
        upper: &[f64],
    ) -> Result<(Fit, bool), PreferenceCertifiedError> {
        let set = &self.contract.model.preferences;
        let linear =
            linear_relation_fit(&self.relations, params).map_err(|(failure, position)| {
                box_refusal(failure, &self.relations[position].id, lower, upper)
            })?;
        let boundary = match linear {
            Fit::Outside => return Ok((Fit::Outside, false)),
            Fit::Inside => false,
            Fit::Boundary => true,
        };
        let mut value_boundary = false;
        for relation in &set.value_relations {
            match value_relation_fit(&self.contract.model.value, relation, params)
                .map_err(|failure| box_refusal(failure, &relation.id, lower, upper))?
            {
                Fit::Outside => return Ok((Fit::Outside, false)),
                Fit::Inside => {}
                Fit::Boundary => value_boundary = true,
            }
        }
        let fit = if boundary || value_boundary { Fit::Boundary } else { Fit::Inside };
        Ok((fit, value_boundary))
    }

    /// Per-draw value enclosures of action `a` on the box.
    pub(super) fn draw_utilities(
        &self,
        source: &SourceData<'_>,
        a: usize,
        params: &[Interval],
    ) -> Result<Vec<Interval>, EncloseFailure> {
        (0..source.weights.len())
            .map(|i| match &source.linear {
                Some(linear) => linear.draw_value(a, i, params),
                None => enclose(&self.contract.model.value, &source.attributes[a][i], params),
            })
            .collect()
    }

    /// Enclosure of `U_b(i) - U_a(i)`.
    pub(super) fn draw_difference(
        source: &SourceData<'_>,
        utilities: &[Option<Vec<Interval>>],
        b: usize,
        a: usize,
        i: usize,
        params: &[Interval],
    ) -> Result<Interval, EncloseFailure> {
        match &source.linear {
            Some(linear) => linear.draw_difference(b, a, i, params),
            None => draws_of(utilities, b)?[i].minus(draws_of(utilities, a)?[i]),
        }
    }

    /// Per-draw regret `max(0, max_b U_b - U_a)` of action `a` over admissible rivals.
    pub(super) fn regrets(
        &self,
        source: &SourceData<'_>,
        utilities: &[Option<Vec<Interval>>],
        a: usize,
        params: &[Interval],
    ) -> Result<Vec<Interval>, EncloseFailure> {
        let rivals: Vec<usize> =
            self.admissible_indices().into_iter().filter(|b| *b != a).collect();
        (0..source.weights.len())
            .map(|i| {
                rivals.iter().try_fold(Interval::point(0.0), |regret, b| {
                    Ok(regret.max(Self::draw_difference(source, utilities, *b, a, i, params)?))
                })
            })
            .collect()
    }

    /// Score enclosure (larger is better) of action `a` on one source.
    pub(super) fn score(
        &self,
        source: &SourceData<'_>,
        utilities: &[Option<Vec<Interval>>],
        a: usize,
        params: &[Interval],
    ) -> Result<Interval, EncloseFailure> {
        let weights = &source.weights;
        match self.criterion() {
            DecisionCriterion::PosteriorExpectedUtility
            | DecisionCriterion::PosteriorExpectedLoss => {
                let mean = match &source.linear {
                    Some(linear) => linear.mean(a, params)?,
                    None => weighted_sum(
                        weights.iter().copied().zip(draws_of(utilities, a)?.iter().copied()),
                    )?,
                };
                Ok(if orientation(self.criterion()) > 0.0 { mean } else { mean.negated() })
            }
            DecisionCriterion::ThresholdProbability { threshold } => {
                let draws = draws_of(utilities, a)?;
                let mass = |reaches: &dyn Fn(&Interval) -> bool| {
                    weighted_sum(
                        weights
                            .iter()
                            .zip(draws)
                            .filter(|(_, value)| reaches(value))
                            .map(|(w, _)| (*w, Interval::point(1.0))),
                    )
                };
                let low = mass(&|value| value.lo >= threshold)?;
                let high = mass(&|value| value.hi >= threshold)?;
                Interval::checked(low.lo, high.hi)
            }
            DecisionCriterion::Quantile { p } => {
                let draws = draws_of(utilities, a)?;
                let lows: Vec<f64> = draws.iter().map(|value| value.lo).collect();
                let highs: Vec<f64> = draws.iter().map(|value| value.hi).collect();
                let low = left_inverse(&lows, weights, p).ok_or(EncloseFailure::NonFinite)?;
                let high = left_inverse(&highs, weights, p).ok_or(EncloseFailure::NonFinite)?;
                Interval::checked(low, high)
            }
            DecisionCriterion::Regret => {
                let regrets = self.regrets(source, utilities, a, params)?;
                let largest = regrets
                    .iter()
                    .zip(weights)
                    .filter(|(_, w)| **w > 0.0)
                    .fold(Interval::point(0.0), |top, (regret, _)| top.max(*regret));
                Ok(largest.negated())
            }
            DecisionCriterion::ExpectedRegret => {
                let regrets = self.regrets(source, utilities, a, params)?;
                Ok(weighted_sum(weights.iter().copied().zip(regrets))?.negated())
            }
            DecisionCriterion::MinimaxOverIdentifiedSet
            | DecisionCriterion::MaximinOverStructures => Err(EncloseFailure::NonFinite),
        }
    }

    /// Margin enclosure `score_a - score_b` on one source.
    pub(super) fn margin(
        &self,
        source: &SourceData<'_>,
        utilities: &[Option<Vec<Interval>>],
        scores: &[Option<Interval>],
        a: usize,
        b: usize,
        params: &[Interval],
    ) -> Result<Interval, EncloseFailure> {
        if is_expectation(self.criterion()) {
            let difference = if let Some(linear) = &source.linear {
                linear.mean_difference(a, b, params)?
            } else {
                let left = draws_of(utilities, a)?;
                let right = draws_of(utilities, b)?;
                let per_draw = left
                    .iter()
                    .zip(right)
                    .map(|(l, r)| l.minus(*r))
                    .collect::<Result<Vec<_>, _>>()?;
                weighted_sum(source.weights.iter().copied().zip(per_draw))?
            };
            let loss = self.criterion() == DecisionCriterion::PosteriorExpectedLoss;
            Ok(if loss { difference.negated() } else { difference })
        } else {
            let left = scores[a].ok_or(EncloseFailure::NonFinite)?;
            let right = scores[b].ok_or(EncloseFailure::NonFinite)?;
            left.minus(right)
        }
    }

    pub(super) fn source_box(
        &self,
        source: &SourceData<'_>,
        params: &[Interval],
        lower: &[f64],
        upper: &[f64],
    ) -> Result<BoxEval, PreferenceCertifiedError> {
        let n = self.contract.actions.len();
        let admissible = self.admissible_indices();
        let fail =
            |failure, a: usize| box_refusal(failure, &self.contract.actions[a].id, lower, upper);
        let means_only = source.linear.is_some()
            && matches!(
                self.criterion(),
                DecisionCriterion::PosteriorExpectedUtility
                    | DecisionCriterion::PosteriorExpectedLoss
            );
        let mut utilities: Vec<Option<Vec<Interval>>> = vec![None; n];
        if !means_only {
            for &a in &admissible {
                utilities[a] =
                    Some(self.draw_utilities(source, a, params).map_err(|f| fail(f, a))?);
            }
        }
        let mut scores = vec![None; n];
        for &a in &admissible {
            scores[a] = Some(self.score(source, &utilities, a, params).map_err(|f| fail(f, a))?);
        }
        let mut margins = vec![vec![None; n]; n];
        for &a in &admissible {
            for &b in admissible.iter().filter(|b| **b != a) {
                margins[a][b] = Some(
                    self.margin(source, &utilities, &scores, a, b, params)
                        .map_err(|f| fail(f, a))?,
                );
            }
        }
        Ok(BoxEval { scores, margins })
    }

    pub(super) fn evaluate_box(
        &self,
        params: &[Interval],
        lower: &[f64],
        upper: &[f64],
    ) -> Result<BoxEval, PreferenceCertifiedError> {
        let mut per_source = self
            .sources
            .iter()
            .map(|source| self.source_box(source, params, lower, upper))
            .collect::<Result<Vec<_>, _>>()?;
        let n = self.contract.actions.len();
        let fail =
            |failure, a: usize| box_refusal(failure, &self.contract.actions[a].id, lower, upper);
        match &self.combination {
            // A run always has at least one source.
            Combination::Single => per_source.pop().ok_or(PreferenceCertifiedError::InvalidAtoms),
            Combination::Worst => {
                let scores: Vec<Option<Interval>> = (0..n)
                    .map(|a| per_source.iter().filter_map(|e| e.scores[a]).reduce(Interval::min))
                    .collect();
                let mut margins = vec![vec![None; n]; n];
                for (a, row) in margins.iter_mut().enumerate() {
                    for (b, slot) in row.iter_mut().enumerate() {
                        if let (Some(left), Some(right)) = (scores[a], scores[b])
                            && a != b
                        {
                            *slot = Some(left.minus(right).map_err(|f| fail(f, a))?);
                        }
                    }
                }
                Ok(BoxEval { scores, margins })
            }
            Combination::Bayes(probabilities) => {
                let mix = |pick: &dyn Fn(&BoxEval) -> Option<Interval>, a: usize| {
                    let parts: Option<Vec<Interval>> = per_source.iter().map(pick).collect();
                    parts
                        .map(|parts| weighted_sum(probabilities.iter().copied().zip(parts)))
                        .transpose()
                        .map_err(|f| fail(f, a))
                };
                let mut scores = Vec::with_capacity(n);
                for a in 0..n {
                    scores.push(mix(&|e: &BoxEval| e.scores[a], a)?);
                }
                let mut margins = vec![vec![None; n]; n];
                for (a, row) in margins.iter_mut().enumerate() {
                    for (b, slot) in row.iter_mut().enumerate() {
                        *slot = mix(&|e: &BoxEval| e.margins[a][b], a)?;
                    }
                }
                Ok(BoxEval { scores, margins })
            }
        }
    }

    /// The action certified on the box, if any, and the actions not proven worse.
    pub(super) fn classify(&self, eval: &BoxEval) -> (Option<usize>, Vec<usize>) {
        let admissible = self.admissible_indices();
        let tie = self.options.tie_tolerance;
        let beats = |a: usize, b: usize| eval.margins[a][b].is_some_and(|m| m.lo > tie);
        let best =
            admissible.iter().copied().find(|&a| admissible.iter().all(|&b| b == a || beats(a, b)));
        let candidates = match best {
            Some(a) => vec![a],
            None => admissible
                .iter()
                .copied()
                .filter(|&a| !admissible.iter().any(|&b| b != a && beats(b, a)))
                .collect(),
        };
        (best, candidates)
    }

    /// Volume relative to the declared box and the largest relative width, over parameters
    /// of positive declared width.
    pub(super) fn measure_box(&self, lower: &[f64], upper: &[f64]) -> (f64, f64) {
        measure_box(&self.declared, lower, upper)
    }

    /// Every parameter's interval on the box `lower..upper` of subdivided coordinates: the box's
    /// own sides, and for a parameter solved from an equality the enclosure of its solution.
    pub(super) fn params(
        &self,
        lower: &[f64],
        upper: &[f64],
    ) -> Result<Vec<Interval>, PreferenceCertifiedError> {
        let mut params: Vec<Interval> =
            lower.iter().zip(upper).map(|(l, u)| Interval { lo: *l, hi: *u }).collect();
        for elimination in &self.eliminations {
            let subject = &self.contract.model.preferences.parameters[elimination.pivot].id;
            params[elimination.pivot] = elimination
                .derive(&params)
                .map_err(|failure| box_refusal(failure, subject, lower, upper))?;
        }
        Ok(params)
    }

    /// The point at the centre of the box, solved parameters at their solution.
    pub(super) fn centre(&self, lower: &[f64], upper: &[f64]) -> Vec<f64> {
        let mut point: Vec<f64> = lower.iter().zip(upper).map(|(l, u)| 0.5 * l + 0.5 * u).collect();
        for elimination in &self.eliminations {
            point[elimination.pivot] = elimination.at(&point);
        }
        point
    }

    /// Whether a point of the set is proven to exist at `point` (subdivided coordinates; a solved
    /// parameter is replaced by the enclosure of its solution): every linear relation holds with
    /// no tolerance and every value relation's gap enclosure is at least minus its tolerance. A
    /// failed enclosure is no proof.
    pub(super) fn point_witnessed(&self, point: &[f64]) -> bool {
        // Subdivided coordinates must lie in the subdivided box (a solved parameter's declared
        // range is checked as two relations).
        let in_box = self.declared.iter().enumerate().all(|(j, (lower, upper))| {
            self.eliminations.iter().any(|e| e.pivot == j)
                || point.get(j).is_some_and(|v| lower <= v && v <= upper)
        });
        if !in_box {
            return false;
        }
        let Ok(params) = self.params(point, point) else {
            return false;
        };
        if !linear_relations_hold(&self.relations, &params).unwrap_or(false) {
            return false;
        }
        let model = &self.contract.model;
        model.preferences.value_relations.iter().all(|relation| {
            matches!(value_relation_fit(&model.value, relation, &params), Ok(Fit::Inside))
        })
    }

    /// Score and margin enclosures at one full parameter point, every parameter (a solved one
    /// included) the degenerate interval of its coordinate.
    pub(super) fn point_eval(&self, point: &[f64]) -> Result<BoxEval, PreferenceCertifiedError> {
        let params: Vec<Interval> = point.iter().copied().map(Interval::point).collect();
        self.evaluate_box(&params, point, point)
    }

    /// The box as reported: subdivided coordinates as they are, a solved parameter's enclosure
    /// cut to its declared range.
    pub(super) fn reported(&self, item: &Examined) -> (Vec<f64>, Vec<f64>) {
        let mut lower = item.lower.clone();
        let mut upper = item.upper.clone();
        for elimination in &self.eliminations {
            let j = elimination.pivot;
            let (low, high) = self.ranges[j];
            lower[j] = item.params[j].lo.clamp(low, high);
            upper[j] = item.params[j].hi.clamp(low, high);
        }
        (lower, upper)
    }

    pub(super) fn examine(
        &self,
        lower: Vec<f64>,
        upper: Vec<f64>,
    ) -> Result<Option<Examined>, PreferenceCertifiedError> {
        let params = self.params(&lower, &upper)?;
        let (fit, value_boundary) = self.relation_fit(&params, &lower, &upper)?;
        let boundary = match fit {
            Fit::Outside => return Ok(None),
            Fit::Inside => false,
            Fit::Boundary => true,
        };
        let eval = self.evaluate_box(&params, &lower, &upper)?;
        let (best, candidates) = self.classify(&eval);
        let (volume, _) = self.measure_box(&lower, &upper);
        Ok(Some(Examined {
            lower,
            upper,
            boundary,
            value_boundary,
            params,
            eval,
            best,
            candidates,
            volume,
        }))
    }

    /// The two halves of a box, or `None` when it is not to be split.
    pub(super) fn split(&self, item: &Examined) -> Option<[(Vec<f64>, Vec<f64>); 2]> {
        split_box(&self.declared, self.options.min_width_fraction, &item.lower, &item.upper)
    }

    pub(super) fn place(
        &self,
        item: Examined,
        kept: &mut Vec<Examined>,
        slots: &mut Vec<Option<Examined>>,
        queue: &mut BinaryHeap<Queued>,
    ) {
        if (item.best.is_some() && !item.value_boundary) || self.split(&item).is_none() {
            kept.push(item);
        } else {
            queue.push(Queued {
                volume: item.volume,
                lower: item.lower.clone(),
                slot: slots.len(),
            });
            slots.push(Some(item));
        }
    }

    /// Cover the declared box; returns the kept boxes and the number examined.
    pub(super) fn subdivide(&self) -> Result<(Vec<Examined>, usize), PreferenceCertifiedError> {
        let lower: Vec<f64> = self.declared.iter().map(|b| b.0).collect();
        let upper: Vec<f64> = self.declared.iter().map(|b| b.1).collect();
        let mut kept = Vec::new();
        let mut slots = Vec::new();
        let mut queue = BinaryHeap::new();
        let mut examined = 1;
        self.observe()?;
        if let Some(root) = self.examine(lower, upper)? {
            self.place(root, &mut kept, &mut slots, &mut queue);
        }
        while let Some(entry) = queue.pop() {
            self.observe()?;
            let Some(item) = slots[entry.slot].take() else {
                continue;
            };
            let halves =
                if examined + 2 > self.options.max_boxes { None } else { self.split(&item) };
            let Some(halves) = halves else {
                kept.push(item);
                continue;
            };
            examined += 2;
            for (child_lower, child_upper) in halves {
                if let Some(child) = self.examine(child_lower, child_upper)? {
                    self.place(child, &mut kept, &mut slots, &mut queue);
                }
            }
        }
        if kept.is_empty() {
            return Err(PreferenceCertifiedError::EmptyPreferenceSet);
        }
        Ok((kept, examined))
    }

    /// Exact score (larger is better) of every admissible action at one parameter point.
    pub(super) fn point_scores(
        &self,
        point: &[f64],
    ) -> Result<Vec<Option<f64>>, PreferenceCertifiedError> {
        let per_source = self
            .sources
            .iter()
            .map(|source| self.source_point_scores(source, point))
            .collect::<Result<Vec<_>, _>>()?;
        let n = self.contract.actions.len();
        Ok((0..n)
            .map(|a| {
                let values: Option<Vec<f64>> = per_source.iter().map(|s| s[a]).collect();
                let values = values?;
                match &self.combination {
                    Combination::Single => values.first().copied(),
                    Combination::Worst => values.into_iter().reduce(f64::min),
                    Combination::Bayes(probabilities) => {
                        Some(probabilities.iter().zip(values).map(|(p, v)| p * v).sum())
                    }
                }
            })
            .collect())
    }

    /// Exact scores on one source: the bound ordinary contract evaluated by the decision layer
    /// when the value can be lowered and the source excludes the same actions as the combined
    /// analysis; otherwise the model evaluated on each draw with the same criterion rules.
    pub(super) fn source_point_scores(
        &self,
        source: &SourceData<'_>,
        point: &[f64],
    ) -> Result<Vec<Option<f64>>, PreferenceCertifiedError> {
        use PreferenceCertifiedError as E;
        let sign = orientation(self.criterion());
        let same_admissible =
            source.exclusions.iter().map(Vec::is_empty).eq(self.admissible.iter().copied());
        if same_admissible {
            match self.contract.bind(point) {
                Ok(bound) => {
                    let result = evaluate_contract(&bound, source.artifact).map_err(E::Eval)?;
                    return Ok(result
                        .actions
                        .iter()
                        .zip(&self.admissible)
                        .map(|(outcome, admissible)| admissible.then_some(sign * outcome.value))
                        .collect());
                }
                Err(PreferenceContractError::NotLowerable { .. }) => {}
                Err(other) => return Err(E::Contract(other)),
            }
        }
        self.direct_point_scores(source, point)
    }

    pub(super) fn direct_point_scores(
        &self,
        source: &SourceData<'_>,
        point: &[f64],
    ) -> Result<Vec<Option<f64>>, PreferenceCertifiedError> {
        let n = self.contract.actions.len();
        let admissible = self.admissible_indices();
        let weights = &source.weights;
        let mut utilities: Vec<Vec<f64>> = vec![Vec::new(); n];
        for &a in &admissible {
            utilities[a] = source.attributes[a]
                .iter()
                .map(|row| self.contract.model.value.evaluate(row, point))
                .collect::<Result<Vec<_>, _>>()
                .map_err(PreferenceCertifiedError::Model)?;
        }
        let best: Vec<f64> = (0..weights.len())
            .map(|i| admissible.iter().map(|a| utilities[*a][i]).fold(f64::NEG_INFINITY, f64::max))
            .collect();
        let criterion = self.criterion();
        let mut scores = vec![None; n];
        for &a in &admissible {
            let utility = &utilities[a];
            let regret: Vec<f64> = utility.iter().zip(&best).map(|(u, top)| top - u).collect();
            let value = criterion_value(criterion, utility, Some(&regret), weights)
                .map_err(PreferenceCertifiedError::Eval)?;
            let score = match (criterion, value) {
                (
                    DecisionCriterion::MinimaxOverIdentifiedSet
                    | DecisionCriterion::MaximinOverStructures,
                    _,
                ) => return Err(PreferenceCertifiedError::StructureInputsRequired),
                (_, Some(value)) => orientation(criterion) * value,
                (_, None) => {
                    return Err(PreferenceCertifiedError::Model(
                        PreferenceModelError::NonFiniteValue,
                    ));
                }
            };
            if !score.is_finite() {
                return Err(PreferenceCertifiedError::Model(PreferenceModelError::NonFiniteValue));
            }
            scores[a] = Some(score);
        }
        Ok(scores)
    }
}

pub(super) fn draws_of(
    utilities: &[Option<Vec<Interval>>],
    a: usize,
) -> Result<&[Interval], EncloseFailure> {
    utilities.get(a).and_then(Option::as_deref).ok_or(EncloseFailure::NonFinite)
}
