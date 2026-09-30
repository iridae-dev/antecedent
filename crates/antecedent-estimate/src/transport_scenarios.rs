//! Reports over the identified members of a finite transport scenario set.
//!
//! A structural envelope is the range of the target response across the
//! scenarios that identified it; it is not a confidence interval and not a sharp
//! causal bound, and it says nothing about scenarios that did not identify. A
//! weighted report uses only declared weights and never renormalizes over the
//! identified scenarios: the mass of every other scenario, and the undeclared
//! residual, is carried as unaccounted mass that may take any value in the
//! outcome's declared domain (from the set's shared coordinate schema, not from
//! the atoms the identified distributions happen to carry).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::VariableId;
use antecedent_expr::ExactDistribution;
use antecedent_identify::sid::scenarios::{
    SCENARIO_WEIGHT_TOLERANCE, ScenarioOutcome, ScenarioSetDecision, mass_sum, unaccounted_after,
};

use crate::error::EstimationError;

/// How a structural envelope may be read.
pub const STRUCTURAL_ENVELOPE_INTERPRETATION: &str =
    "range_over_identified_scenarios_not_a_confidence_interval_or_sharp_bound";
/// How a weighted scenario range may be read.
pub const WEIGHTED_RANGE_INTERPRETATION: &str = "declared_weight_range_with_unaccounted_mass_at_declared_domain_limits_not_a_confidence_interval";

/// Range of one outcome's mean across identified scenarios.
#[derive(Clone, Debug, PartialEq)]
pub struct MeanEnvelope {
    /// Outcome coordinate.
    pub outcome: VariableId,
    /// Smallest identified mean.
    pub lower: f64,
    /// Largest identified mean.
    pub upper: f64,
    /// Scenario attaining `lower` (first in canonical order on ties).
    pub lower_scenario: Arc<str>,
    /// Scenario attaining `upper` (first in canonical order on ties).
    pub upper_scenario: Arc<str>,
}

/// Structural envelope over the identified scenarios.
#[derive(Clone, Debug, PartialEq)]
pub struct StructuralEnvelope {
    /// Identified scenarios the envelope ranges over, in canonical order.
    pub scenarios: Vec<Arc<str>>,
    /// Per-outcome mean ranges.
    pub means: Vec<MeanEnvelope>,
    /// Atom-wise probability ranges when every identified scenario has the same
    /// atoms; `None` otherwise.
    pub atoms: Option<Vec<(f64, f64)>>,
    /// Always [`STRUCTURAL_ENVELOPE_INTERPRETATION`].
    pub interpretation: &'static str,
}

/// Declared-weight report over a weighted scenario set.
#[derive(Clone, Debug, PartialEq)]
pub struct WeightedScenarioReport {
    /// Declared weight of the identified scenarios.
    pub identified_mass: f64,
    /// Everything else: other scenarios' weight plus the undeclared residual.
    pub unaccounted_mass: f64,
    /// Per outcome, `Σ w_s · mean_s` over identified scenarios (not renormalized).
    pub identified_weighted_sums: Vec<(VariableId, f64)>,
    /// Per outcome, the range the mixture mean can take when unaccounted mass
    /// sits anywhere in the outcome's declared domain; `None` if no scenario
    /// identified, or if unaccounted mass remains and an outcome's declared
    /// domain is not finite (continuous, count or unspecified).
    pub ranges: Option<Vec<(VariableId, f64, f64)>>,
    /// Always [`WEIGHTED_RANGE_INTERPRETATION`].
    pub interpretation: &'static str,
}

fn means(distribution: &ExactDistribution) -> Result<Vec<f64>, EstimationError> {
    distribution
        .outcomes
        .iter()
        .map(|outcome| {
            distribution.mean(*outcome).map_err(|error| crate::transport::refuse_eval(&error))
        })
        .collect()
}

fn check_outcomes(points: &[(Arc<str>, &ExactDistribution)]) -> Result<(), EstimationError> {
    if points.windows(2).any(|pair| pair[0].1.outcomes != pair[1].1.outcomes) {
        return Err(EstimationError::data_msg("scenario distributions disagree on outcomes"));
    }
    Ok(())
}

/// Envelope of the identified scenarios' point distributions, given in
/// canonical order. `None` when no scenario identified.
///
/// # Errors
/// Distributions over different outcomes, or a non-numeric outcome.
pub(crate) fn structural_envelope(
    points: &[(Arc<str>, &ExactDistribution)],
) -> Result<Option<StructuralEnvelope>, EstimationError> {
    let Some((_, first)) = points.first() else { return Ok(None) };
    check_outcomes(points)?;
    let per_scenario = points.iter().map(|(_, d)| means(d)).collect::<Result<Vec<_>, _>>()?;
    let mut envelopes = Vec::with_capacity(first.outcomes.len());
    for (k, outcome) in first.outcomes.iter().enumerate() {
        let (mut lo, mut hi) = (0usize, 0usize);
        for (s, values) in per_scenario.iter().enumerate() {
            if values[k] < per_scenario[lo][k] {
                lo = s;
            }
            if values[k] > per_scenario[hi][k] {
                hi = s;
            }
        }
        envelopes.push(MeanEnvelope {
            outcome: *outcome,
            lower: per_scenario[lo][k],
            upper: per_scenario[hi][k],
            lower_scenario: Arc::clone(&points[lo].0),
            upper_scenario: Arc::clone(&points[hi].0),
        });
    }
    let atoms = points.iter().all(|(_, d)| d.atoms == first.atoms).then(|| {
        (0..first.probabilities.len())
            .map(|a| {
                points
                    .iter()
                    .map(|(_, d)| d.probabilities[a])
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), p| (lo.min(p), hi.max(p)))
            })
            .collect()
    });
    Ok(Some(StructuralEnvelope {
        scenarios: points.iter().map(|(name, _)| Arc::clone(name)).collect(),
        means: envelopes,
        atoms,
        interpretation: STRUCTURAL_ENVELOPE_INTERPRETATION,
    }))
}

/// Declared-weight report. `points` are the identified scenarios with their
/// weights; `outcomes` are the queried outcomes with the limits of their
/// declared domains (`None` for a domain that is not finite);
/// `unaccounted_mass` is one minus the identified weight (other scenarios plus
/// residual).
///
/// # Errors
/// Invalid masses, distributions over different outcomes, or a non-numeric outcome.
pub(crate) fn weighted_scenario_report(
    points: &[(Arc<str>, f64, &ExactDistribution)],
    outcomes: &[(VariableId, Option<(f64, f64)>)],
    unaccounted_mass: f64,
) -> Result<WeightedScenarioReport, EstimationError> {
    let identified_mass = mass_sum(points.iter().map(|(_, w, _)| *w));
    // Mass within the declared-weight tolerance of zero is zero, the same
    // tolerance `TransportScenarioSet::try_new` accepts weights under.
    let unaccounted_mass = if (0.0..=SCENARIO_WEIGHT_TOLERANCE).contains(&unaccounted_mass) {
        0.0
    } else {
        unaccounted_mass
    };
    if !unaccounted_mass.is_finite()
        || unaccounted_mass < 0.0
        || (identified_mass + unaccounted_mass - 1.0).abs() > 1e-9
    {
        return Err(EstimationError::data_msg("scenario masses must sum to one"));
    }
    let named = points.iter().map(|(n, _, d)| (Arc::clone(n), *d)).collect::<Vec<_>>();
    check_outcomes(&named)?;
    // Each sum is order independent, so renaming scenarios cannot move it.
    let mut terms = vec![Vec::with_capacity(points.len()); outcomes.len()];
    for (_, weight, distribution) in points {
        for (k, mean) in means(distribution)?.into_iter().enumerate() {
            terms[k].push(weight * mean);
        }
    }
    let sums =
        outcomes.iter().zip(terms).map(|((o, _), terms)| (*o, mass_sum(terms))).collect::<Vec<_>>();
    let ranges = if points.is_empty() {
        None
    } else if unaccounted_mass <= 0.0 {
        Some(sums.iter().map(|(o, sum)| (*o, *sum, *sum)).collect())
    } else {
        outcomes
            .iter()
            .zip(&sums)
            .map(|((outcome, limits), (_, sum))| {
                limits.map(|(lo, hi)| {
                    (*outcome, sum + unaccounted_mass * lo, sum + unaccounted_mass * hi)
                })
            })
            .collect::<Option<Vec<_>>>()
    };
    Ok(WeightedScenarioReport {
        identified_mass,
        unaccounted_mass,
        identified_weighted_sums: sums,
        ranges,
        interpretation: WEIGHTED_RANGE_INTERPRETATION,
    })
}

/// Stable status of a scenario after preparation and evaluation.
pub const SCENARIO_STATUSES: [&str; 7] = [
    "identified",
    "structurally_unidentified",
    "missing_evidence",
    "not_certified",
    "unsupported_provider",
    "support_failure",
    "unevaluated",
];

#[derive(Clone, Debug)]
enum ScenarioPlan {
    /// Identified and compiled against the supplied laws.
    Executable(antecedent_expr::ExactEvaluationPlan),
    /// Identified, but the supplied providers cannot evaluate it.
    Refused { status: &'static str, detail: String },
    /// Not identified; the decision says why.
    NotIdentified,
}

/// A decided scenario set with each identified scenario compiled once.
/// Evaluation never re-identifies; refresh recompiles against new laws only.
#[derive(Clone, Debug)]
pub struct PreparedScenarioSet {
    decision: ScenarioSetDecision,
    data: antecedent_expr::ExactTransportData,
    /// The finite samples the laws were fitted from, for the empirical plug-in.
    empirical: Option<Arc<crate::StatisticalTransportInput>>,
    request: antecedent_expr::Assignment,
    limits: antecedent_expr::ExactEvaluationLimits,
    plans: Vec<ScenarioPlan>,
}

/// One scenario's result.
#[derive(Clone, Debug)]
pub struct ScenarioResult {
    /// Scenario name.
    pub name: Arc<str>,
    /// Declared weight, if any.
    pub weight: Option<f64>,
    /// One of [`SCENARIO_STATUSES`].
    pub status: &'static str,
    /// Why the scenario did not produce a point, when it did not.
    pub detail: Option<String>,
    /// The point distribution of an identified, evaluated scenario.
    pub distribution: Option<ExactDistribution>,
}

/// Count and (for a weighted set) declared mass of one status.
#[derive(Clone, Debug, PartialEq)]
pub struct StatusMass {
    /// Status name.
    pub status: &'static str,
    /// Scenarios with this status.
    pub count: usize,
    /// Their declared weight; `None` for an unweighted set.
    pub mass: Option<f64>,
}

/// Every scenario, retained whatever its status, with the envelope and masses.
#[derive(Clone, Debug)]
pub struct ScenarioSetReport {
    /// Every scenario in canonical order, failed as well as successful.
    pub scenarios: Vec<ScenarioResult>,
    /// Count and mass per status, in [`SCENARIO_STATUSES`] order.
    pub masses: Vec<StatusMass>,
    /// Declared mass assigned to no scenario; `None` for an unweighted set.
    pub residual_mass: Option<f64>,
    /// Envelope over identified scenarios; `None` when none identified.
    pub envelope: Option<StructuralEnvelope>,
    /// Declared-weight report; `None` for an unweighted set.
    pub weighted: Option<WeightedScenarioReport>,
    /// Scenario-budget receipt when scenarios were left unevaluated.
    pub receipt: Option<antecedent_core::SearchReceipt>,
}

fn compile_scenario(
    functional: &antecedent_identify::BoundTransportFunctional,
    data: &antecedent_expr::ExactTransportData,
    request: &antecedent_expr::Assignment,
    limits: antecedent_expr::ExactEvaluationLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<ScenarioPlan, EstimationError> {
    if let Some(detail) = uncovered_regime(functional, data) {
        return Ok(ScenarioPlan::Refused { status: "unsupported_provider", detail });
    }
    match crate::transport::prepare_exact_transport(
        functional,
        data.clone(),
        request.clone(),
        limits,
        ctx,
    ) {
        Ok(plan) => Ok(ScenarioPlan::Executable(plan)),
        Err(error) => classify(&error)
            .map(|status| ScenarioPlan::Refused { status, detail: error.to_string() }),
    }
}

/// The first regime a bound leaf cites that no supplied law realizes. The
/// evaluator would otherwise report the gap as a missing table entry, which
/// reads as a support failure rather than a missing provider.
fn uncovered_regime(
    functional: &antecedent_identify::BoundTransportFunctional,
    data: &antecedent_expr::ExactTransportData,
) -> Option<String> {
    use antecedent_expr::ExprNode;
    let arena = functional.arena();
    let mut pending = vec![functional.root()];
    let mut seen = std::collections::BTreeSet::new();
    while let Some(id) = pending.pop() {
        if !seen.insert(id.raw()) {
            continue;
        }
        match arena.node(id) {
            ExprNode::Distribution { population, regime: Some(regime), .. } => {
                let population = arena.population(*population);
                if !data
                    .laws()
                    .iter()
                    .any(|law| law.regime() == *regime && law.population() == population)
                {
                    return Some(format!(
                        "no supplied law realizes {population} regime {}",
                        regime.raw()
                    ));
                }
            }
            ExprNode::Product(list) => pending.extend(arena.list(*list).iter().copied()),
            ExprNode::SumOut { expr, .. } => pending.push(*expr),
            ExprNode::Ratio { numerator, denominator } => {
                pending.extend([*numerator, *denominator])
            }
            _ => {}
        }
    }
    None
}

/// Provider and support failures are scenario-local statuses; anything else
/// (an invalid request, a numerical failure) is an error of the whole call.
fn classify(error: &antecedent_expr::EvalError) -> Result<&'static str, EstimationError> {
    use antecedent_core::TransportOutcomeKind as K;
    match crate::transport::transport_outcome_kind(error) {
        K::SupportFailure => Ok("support_failure"),
        K::MissingProvider | K::UnsupportedEvaluator => Ok("unsupported_provider"),
        _ => Err(crate::transport::refuse_eval(error)),
    }
}

fn compile_all(
    decision: &ScenarioSetDecision,
    data: &antecedent_expr::ExactTransportData,
    request: &antecedent_expr::Assignment,
    limits: antecedent_expr::ExactEvaluationLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<Vec<ScenarioPlan>, EstimationError> {
    decision
        .decisions
        .iter()
        .map(|d| match &d.outcome {
            ScenarioOutcome::Identified(functional) => {
                compile_scenario(functional, data, request, limits, ctx)
            }
            _ => Ok(ScenarioPlan::NotIdentified),
        })
        .collect()
}

/// Provider identity of supplied exact laws.
pub const SCENARIO_EXACT_PROVIDER: &str = "transport.exact_supplied_laws";

#[allow(clippy::needless_pass_by_value)] // A `map_err` adapter.
fn refuse_scenario(
    refusal: antecedent_identify::sid::scenarios::ScenarioSetRefusal,
) -> EstimationError {
    EstimationError::refused(refusal.code, format!("{}: {}", refusal.detail, refusal.message))
}

/// Check every sample row and intervention against the set's shared coordinate
/// schema: each column is a declared variable and each present value lies in its
/// declared domain (a missing entry is a missingness matter, not a value).
fn check_samples(
    decision: &ScenarioSetDecision,
    input: &crate::StatisticalTransportInput,
) -> Result<(), EstimationError> {
    let set = &decision.set;
    for sample in &input.samples {
        let context = format!("the {} sample of regime {}", sample.population, sample.regime.raw());
        for assignment in sample.interventions.iter() {
            set.check_value(assignment.variable, &assignment.value, &context)
                .map_err(refuse_scenario)?;
        }
        for (variable, column) in &sample.columns {
            if set.coordinate(*variable).is_none() {
                return Err(refuse_scenario(
                    antecedent_identify::sid::scenarios::ScenarioSetRefusal::coordinate_mismatch(
                        format!(
                            "{context} has a column for variable {} outside the shared schema",
                            variable.raw()
                        ),
                    ),
                ));
            }
            for value in column.iter().flatten() {
                set.check_value(*variable, &antecedent_core::Value::f64(*value), &context)
                    .map_err(refuse_scenario)?;
            }
        }
    }
    Ok(())
}

/// Check every law and the request against the set's shared coordinate schema.
fn check_schema(
    decision: &ScenarioSetDecision,
    data: &antecedent_expr::ExactTransportData,
    request: &antecedent_expr::Assignment,
) -> Result<(), EstimationError> {
    let set = &decision.set;
    let refuse = refuse_scenario;
    for law in data.laws() {
        let context = format!("the {} law of regime {}", law.population(), law.regime().raw());
        for axis in law.axes() {
            for value in axis.values.iter() {
                set.check_value(axis.variable, value, &context).map_err(refuse)?;
            }
        }
        for assignment in law.interventions() {
            set.check_value(assignment.variable, &assignment.value, &context).map_err(refuse)?;
        }
    }
    for (variable, value) in request.entries() {
        set.check_value(*variable, value, "the request").map_err(refuse)?;
    }
    Ok(())
}

/// Compile each identified scenario of a decided set once against supplied
/// exact laws.
///
/// # Errors
/// `schema_mismatch` for a law or request outside the shared coordinate
/// schema; a request or law set that fails for every scenario (not a
/// scenario-local provider or support failure); or cancellation.
pub fn prepare_transport_scenarios(
    decision: ScenarioSetDecision,
    data: antecedent_expr::ExactTransportData,
    request: antecedent_expr::Assignment,
    limits: antecedent_expr::ExactEvaluationLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<PreparedScenarioSet, EstimationError> {
    check_schema(&decision, &data, &request)?;
    let plans = compile_all(&decision, &data, &request, limits, ctx)?;
    Ok(PreparedScenarioSet { decision, data, empirical: None, request, limits, plans })
}

/// Compile each identified scenario against empirical plug-in frequency tables
/// fitted once from finite samples (with any supplied exact laws), through the
/// same plan as supplied laws. Points only: the bootstrap is fixed at zero
/// replicates, so no interval is computed, and the envelope stays a structural
/// range over identified scenarios. The tables are fitted once for the whole
/// set, not per scenario; each scenario then compiles against them, so
/// provider coverage is per scenario. Every sample row and intervention is
/// checked against the shared coordinate schema first.
///
/// # Errors
/// As [`prepare_transport_scenarios`], plus a sample outside the shared schema
/// (`schema_mismatch`) or one the plug-in cannot fit.
pub fn prepare_empirical_transport_scenarios(
    decision: ScenarioSetDecision,
    input: crate::StatisticalTransportInput,
    max_joint_cells: usize,
    request: antecedent_expr::Assignment,
    limits: antecedent_expr::ExactEvaluationLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<PreparedScenarioSet, EstimationError> {
    check_samples(&decision, &input)?;
    let options = crate::EmpiricalTableOptions {
        estimator: crate::EmpiricalTableEstimator::Plugin,
        bootstrap_replicates: 0,
        max_joint_cells,
        ..crate::EmpiricalTableOptions::default()
    };
    // Every scenario shares the catalog and question, so the fitted tables do
    // not depend on which identified scenario supplies the binding context;
    // per-scenario provider coverage is checked when each scenario compiles.
    let binding = decision.decisions.iter().find_map(|d| match &d.outcome {
        ScenarioOutcome::Identified(functional) => Some(functional),
        _ => None,
    });
    let data = match binding {
        Some(functional) => {
            crate::empirical_table::assemble_grid_point_laws(&input, functional, &options, ctx)?
        }
        None => antecedent_expr::ExactTransportData::try_new(
            input.supplied.clone(),
            max_joint_cells.max(1),
        )
        .map_err(|e| EstimationError::data_msg(e.to_string()))?,
    };
    let mut prepared = prepare_transport_scenarios(decision, data, request, limits, ctx)?;
    prepared.empirical = Some(Arc::new(input));
    Ok(prepared)
}

impl PreparedScenarioSet {
    /// The frozen decision.
    #[must_use]
    pub const fn decision(&self) -> &ScenarioSetDecision {
        &self.decision
    }
    /// Retained laws: supplied, or fitted by the empirical plug-in.
    #[must_use]
    pub const fn data(&self) -> &antecedent_expr::ExactTransportData {
        &self.data
    }
    /// The samples the empirical plug-in fitted, when it did.
    #[must_use]
    pub fn empirical_input(&self) -> Option<&crate::StatisticalTransportInput> {
        self.empirical.as_deref()
    }
    /// Provider identity: [`SCENARIO_EXACT_PROVIDER`] or the empirical plug-in's.
    #[must_use]
    pub const fn provider(&self) -> &'static str {
        if self.empirical.is_some() {
            crate::EMPIRICAL_TABLE_PLUGIN
        } else {
            SCENARIO_EXACT_PROVIDER
        }
    }
    /// Target request.
    #[must_use]
    pub const fn request(&self) -> &antecedent_expr::Assignment {
        &self.request
    }
    /// Evaluation limits.
    #[must_use]
    pub const fn limits(&self) -> antecedent_expr::ExactEvaluationLimits {
        self.limits
    }

    /// Each scenario's retained plan: `compiled`, the refusal status of an
    /// identified scenario the providers cannot run, or `not_identified:<status>`.
    #[must_use]
    pub fn plan_summary(&self) -> Vec<(Arc<str>, String)> {
        self.decision
            .decisions
            .iter()
            .zip(&self.plans)
            .map(|(d, plan)| {
                let kind = match plan {
                    ScenarioPlan::Executable(_) => "compiled".to_owned(),
                    ScenarioPlan::Refused { status, .. } => (*status).to_owned(),
                    ScenarioPlan::NotIdentified => format!("not_identified:{}", d.outcome.status()),
                };
                (Arc::clone(&d.scenario.name), kind)
            })
            .collect()
    }

    /// Refit from new samples and recompile; decisions are kept.
    ///
    /// # Errors
    /// As [`prepare_empirical_transport_scenarios`].
    pub fn refresh_empirical(
        &self,
        input: crate::StatisticalTransportInput,
        max_joint_cells: usize,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<Self, EstimationError> {
        prepare_empirical_transport_scenarios(
            self.decision.clone(),
            input,
            max_joint_cells,
            self.request.clone(),
            self.limits,
            ctx,
        )
    }

    /// Recompile every identified scenario against new supplied laws; decisions
    /// are kept.
    ///
    /// # Errors
    /// As [`prepare_transport_scenarios`].
    pub fn refresh(
        &self,
        data: antecedent_expr::ExactTransportData,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<Self, EstimationError> {
        prepare_transport_scenarios(
            self.decision.clone(),
            data,
            self.request.clone(),
            self.limits,
            ctx,
        )
    }

    /// Evaluate every compiled scenario and report all of them.
    ///
    /// # Errors
    /// A numerical failure, invalid outcome, or cancellation.
    pub fn evaluate(
        &self,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<ScenarioSetReport, EstimationError> {
        let mut scenarios = Vec::with_capacity(self.plans.len());
        for (decision, plan) in self.decision.decisions.iter().zip(&self.plans) {
            crate::transport::refuse_cancelled(ctx, "transport scenario evaluation")?;
            let scenario = &decision.scenario;
            let (status, detail, distribution) = match (plan, &decision.outcome) {
                (ScenarioPlan::Executable(plan), _) => match plan.evaluate(ctx) {
                    Ok(distribution) => ("identified", None, Some(distribution)),
                    Err(error) => (classify(&error)?, Some(error.to_string()), None),
                },
                (ScenarioPlan::Refused { status, detail }, _) => {
                    (*status, Some(detail.clone()), None)
                }
                (ScenarioPlan::NotIdentified, outcome) => {
                    (outcome.status(), Some(outcome_detail(outcome, &self.decision.set)), None)
                }
            };
            scenarios.push(ScenarioResult {
                name: Arc::clone(&scenario.name),
                weight: scenario.weight,
                status,
                detail,
                distribution,
            });
        }
        report(&self.decision, scenarios)
    }
}

fn outcome_detail(
    outcome: &ScenarioOutcome,
    set: &antecedent_identify::sid::scenarios::TransportScenarioSet,
) -> String {
    // Declared names in name order, so the detail does not depend on how the
    // caller numbered the variables.
    let names = |nodes: &[u32]| {
        let mut named = nodes
            .iter()
            .map(|n| {
                set.coordinate(VariableId::from_raw(*n))
                    .map_or_else(|| format!("VariableId({n})"), |c| c.name.to_string())
            })
            .collect::<Vec<_>>();
        named.sort();
        named.join(", ")
    };
    match outcome {
        ScenarioOutcome::Identified(_) => String::new(),
        ScenarioOutcome::StructurallyUnidentified(hedge) => {
            format!(
                "verified s-hedge: forest {{{}}} inside {{{}}}",
                names(&hedge.smaller.nodes),
                names(&hedge.larger.nodes)
            )
        }
        ScenarioOutcome::MissingEvidence { obligations }
        | ScenarioOutcome::NotCertified { obligations } => obligations.join("; "),
        ScenarioOutcome::Unevaluated { stop } => format!(
            "{}: {}",
            antecedent_identify::sid::scenarios::SCENARIO_UNEVALUATED_DETAIL,
            stop.code()
        ),
    }
}

fn report(
    decision: &ScenarioSetDecision,
    scenarios: Vec<ScenarioResult>,
) -> Result<ScenarioSetReport, EstimationError> {
    let weighted = decision.set.weighted();
    let masses = SCENARIO_STATUSES
        .iter()
        .map(|status| {
            let members = scenarios.iter().filter(|s| s.status == *status);
            StatusMass {
                status,
                count: members.clone().count(),
                mass: weighted.then(|| mass_sum(members.filter_map(|s| s.weight))),
            }
        })
        .collect::<Vec<_>>();
    let identified = scenarios
        .iter()
        .filter_map(|s| s.distribution.as_ref().map(|d| (Arc::clone(&s.name), s.weight, d)))
        .collect::<Vec<_>>();
    let envelope = structural_envelope(
        &identified.iter().map(|(n, _, d)| (Arc::clone(n), *d)).collect::<Vec<_>>(),
    )?;
    let weighted_report = if weighted {
        let points = identified
            .iter()
            .map(|(n, w, d)| (Arc::clone(n), w.unwrap_or(0.0), *d))
            .collect::<Vec<_>>();
        let identified_mass = mass_sum(points.iter().map(|(_, w, _)| *w));
        let outcomes = decision
            .decisions
            .iter()
            .find_map(|d| match &d.outcome {
                ScenarioOutcome::Identified(f) => Some(f.derivation().query().outcomes.to_vec()),
                _ => None,
            })
            .unwrap_or_default()
            .into_iter()
            .map(|o| {
                (
                    o,
                    decision.set.coordinate(o).and_then(
                        antecedent_identify::sid::scenarios::ScenarioCoordinate::support_limits,
                    ),
                )
            })
            .collect::<Vec<_>>();
        Some(weighted_scenario_report(&points, &outcomes, unaccounted_after(identified_mass))?)
    } else {
        None
    };
    Ok(ScenarioSetReport {
        scenarios,
        masses,
        residual_mass: decision.set.residual_mass(),
        envelope,
        weighted: weighted_report,
        receipt: decision.receipt.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::Value;

    fn binary(p_one: f64) -> ExactDistribution {
        ExactDistribution {
            outcomes: Arc::from([VariableId::from_raw(3)]),
            atoms: Arc::from([Arc::from([Value::f64(0.0)]), Arc::from([Value::f64(1.0)])]),
            probabilities: Arc::from([1.0 - p_one, p_one]),
            support: Arc::from([]),
        }
    }

    #[test]
    fn envelope_ranges_over_identified_scenarios_and_names_the_extremes() {
        let (a, b, c) = (binary(0.2), binary(0.7), binary(0.5));
        let points = vec![(Arc::from("a"), &a), (Arc::from("b"), &b), (Arc::from("c"), &c)];
        let envelope = structural_envelope(&points).unwrap().unwrap();
        let mean = &envelope.means[0];
        assert!((mean.lower - 0.2).abs() < 1e-12 && (mean.upper - 0.7).abs() < 1e-12);
        assert_eq!((&*mean.lower_scenario, &*mean.upper_scenario), ("a", "b"));
        assert_eq!(envelope.atoms.unwrap()[1], (0.2, 0.7));
        assert!(structural_envelope(&[]).unwrap().is_none());
    }

    #[test]
    fn weighted_report_never_renormalizes_over_identified_scenarios() {
        let (a, b) = (binary(0.2), binary(0.6));
        let points = vec![(Arc::from("a"), 0.3, &a), (Arc::from("b"), 0.2, &b)];
        let outcomes = [(VariableId::from_raw(3), Some((0.0, 1.0)))];
        let report = weighted_scenario_report(&points, &outcomes, 0.5).unwrap();
        let sum = report.identified_weighted_sums[0].1;
        assert!((sum - (0.3 * 0.2 + 0.2 * 0.6)).abs() < 1e-12);
        // The renormalized value (0.18 / 0.5 = 0.36) is not reported; the range
        // lets the unaccounted half sit anywhere in {0, 1}.
        let (_, lo, hi) = report.ranges.unwrap()[0];
        assert!((lo - 0.18).abs() < 1e-12 && (hi - 0.68).abs() < 1e-12);
        assert!(weighted_scenario_report(&points, &outcomes, 0.4).is_err());
        let none = weighted_scenario_report(&[], &outcomes, 1.0).unwrap();
        assert!(none.ranges.is_none());
    }

    #[test]
    fn weighted_ranges_use_the_declared_domain_not_the_observed_atoms() {
        // Both distributions carry atoms {0, 1}, but the outcome is declared
        // categorical with three levels: unaccounted mass may sit at 2.
        let (a, b) = (binary(0.2), binary(0.6));
        let points = vec![(Arc::from("a"), 0.3, &a), (Arc::from("b"), 0.2, &b)];
        let declared = [(VariableId::from_raw(3), Some((0.0, 2.0)))];
        let (_, lo, hi) =
            weighted_scenario_report(&points, &declared, 0.5).unwrap().ranges.unwrap()[0];
        assert!((lo - 0.18).abs() < 1e-12 && (hi - (0.18 + 0.5 * 2.0)).abs() < 1e-12);
        // An unbounded declared domain withholds the range while mass is unaccounted.
        let unbounded = [(VariableId::from_raw(3), None)];
        assert!(weighted_scenario_report(&points, &unbounded, 0.5).unwrap().ranges.is_none());
        let full = vec![(Arc::from("a"), 0.5, &a), (Arc::from("b"), 0.5, &b)];
        let (_, lo, hi) =
            weighted_scenario_report(&full, &unbounded, 0.0).unwrap().ranges.unwrap()[0];
        assert!((lo - 0.4).abs() < 1e-12 && (hi - 0.4).abs() < 1e-12);
    }
}
