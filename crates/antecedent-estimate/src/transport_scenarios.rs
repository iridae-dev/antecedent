//! Reports over the identified members of a finite transport scenario set.
//!
//! A structural envelope is the range of the target response across the
//! scenarios that identified it; it is not a confidence interval and not a sharp
//! causal bound, and it says nothing about scenarios that did not identify. A
//! weighted report uses only declared weights and never renormalizes over the
//! identified scenarios: the mass of every other scenario, and the undeclared
//! residual, is carried as unaccounted mass that may take any value in the
//! outcome's support.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::VariableId;
use antecedent_expr::ExactDistribution;
use antecedent_identify::sid::scenarios::{ScenarioOutcome, ScenarioSetDecision};

use crate::error::EstimationError;

/// How a structural envelope may be read.
pub const STRUCTURAL_ENVELOPE_INTERPRETATION: &str =
    "range_over_identified_scenarios_not_a_confidence_interval_or_sharp_bound";
/// How a weighted scenario range may be read.
pub const WEIGHTED_RANGE_INTERPRETATION: &str = "declared_weight_range_with_unaccounted_mass_at_outcome_support_limits_not_a_confidence_interval";

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
    /// sits anywhere in the outcome's support; `None` if no scenario identified.
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
pub fn structural_envelope(
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
/// weights; `outcomes` are the queried outcomes; `unaccounted_mass` is one minus
/// the identified weight (other scenarios plus residual).
///
/// # Errors
/// Invalid masses, distributions over different outcomes, or a non-numeric outcome.
pub fn weighted_scenario_report(
    points: &[(Arc<str>, f64, &ExactDistribution)],
    outcomes: &[VariableId],
    unaccounted_mass: f64,
) -> Result<WeightedScenarioReport, EstimationError> {
    let identified_mass: f64 = points.iter().map(|(_, w, _)| *w).sum();
    if !unaccounted_mass.is_finite()
        || unaccounted_mass < 0.0
        || (identified_mass + unaccounted_mass - 1.0).abs() > 1e-9
    {
        return Err(EstimationError::data_msg("scenario masses must sum to one"));
    }
    let named = points.iter().map(|(n, _, d)| (Arc::clone(n), *d)).collect::<Vec<_>>();
    check_outcomes(&named)?;
    let mut sums = outcomes.iter().map(|o| (*o, 0.0)).collect::<Vec<_>>();
    for (_, weight, distribution) in points {
        for (k, mean) in means(distribution)?.into_iter().enumerate() {
            sums[k].1 += weight * mean;
        }
    }
    let ranges = (!points.is_empty()).then(|| -> Result<_, EstimationError> {
        outcomes
            .iter()
            .enumerate()
            .map(|(k, outcome)| {
                let mut support = (f64::INFINITY, f64::NEG_INFINITY);
                for (_, _, d) in points {
                    for atom in d.atoms.iter() {
                        let value = atom[k].as_f64().ok_or_else(|| {
                            EstimationError::data_msg("weighted range needs a numeric outcome")
                        })?;
                        support = (support.0.min(value), support.1.max(value));
                    }
                }
                Ok((
                    *outcome,
                    sums[k].1 + unaccounted_mass * support.0,
                    sums[k].1 + unaccounted_mass * support.1,
                ))
            })
            .collect::<Result<Vec<_>, _>>()
    });
    Ok(WeightedScenarioReport {
        identified_mass,
        unaccounted_mass,
        identified_weighted_sums: sums,
        ranges: ranges.transpose()?,
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

/// Compile each identified scenario of a decided set once against `data`.
///
/// # Errors
/// A request or law set that fails for every scenario (not a scenario-local
/// provider or support failure), or cancellation.
pub fn prepare_transport_scenarios(
    decision: ScenarioSetDecision,
    data: antecedent_expr::ExactTransportData,
    request: antecedent_expr::Assignment,
    limits: antecedent_expr::ExactEvaluationLimits,
    ctx: &antecedent_core::ExecutionContext,
) -> Result<PreparedScenarioSet, EstimationError> {
    let plans = compile_all(&decision, &data, &request, limits, ctx)?;
    Ok(PreparedScenarioSet { decision, data, request, limits, plans })
}

impl PreparedScenarioSet {
    /// The frozen decision.
    #[must_use]
    pub const fn decision(&self) -> &ScenarioSetDecision {
        &self.decision
    }
    /// Retained laws.
    #[must_use]
    pub const fn data(&self) -> &antecedent_expr::ExactTransportData {
        &self.data
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

    /// Recompile every identified scenario against new laws; decisions are kept.
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
                    (outcome.status(), Some(outcome_detail(outcome)), None)
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

fn outcome_detail(outcome: &ScenarioOutcome) -> String {
    match outcome {
        ScenarioOutcome::Identified(_) => String::new(),
        ScenarioOutcome::StructurallyUnidentified(hedge) => {
            format!(
                "verified s-hedge: forest {:?} inside {:?}",
                hedge.smaller.nodes, hedge.larger.nodes
            )
        }
        ScenarioOutcome::MissingEvidence { obligations }
        | ScenarioOutcome::NotCertified { obligations } => obligations.join("; "),
        ScenarioOutcome::Unevaluated { reason } => (*reason).to_owned(),
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
                mass: weighted.then(|| members.filter_map(|s| s.weight).sum()),
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
        let identified_mass: f64 = points.iter().map(|(_, w, _)| *w).sum();
        let outcomes = decision
            .decisions
            .iter()
            .find_map(|d| match &d.outcome {
                ScenarioOutcome::Identified(f) => Some(f.derivation().query().outcomes.to_vec()),
                _ => None,
            })
            .unwrap_or_default();
        Some(weighted_scenario_report(&points, &outcomes, (1.0 - identified_mass).max(0.0))?)
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
        let outcomes = [VariableId::from_raw(3)];
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
}
