//! Python execution boundary for the finite-action inverse-outcome query.
//!
//! The forward response is evaluated by an existing response route in Python;
//! this boundary only hands its numbers (as published) and the caller's action
//! grid to the Rust classifier and returns the report as a dictionary.

use antecedent::{
    ActionConstraint, ActionSpec, ForwardEvaluation, ForwardInterval, IntervalScope,
    InverseOutcomeError, InverseOutcomeReport, InverseQuery, SupportBasis, TargetDirection,
    classify_inverse_outcome,
};
use antecedent_core::{IntervalInterpretation, SupportStatus, reason_code};
use pyo3::prelude::*;

use crate::transport_z_api::to_py_json;

/// `(scope, interpretation, level, lower, upper)` of a forward interval.
type IntervalTuple = (String, String, f64, Vec<f64>, Vec<f64>);

/// `(outcome, population, mean_response, point_identified, points, mean, support,
/// support_basis, interval, assumptions)` of a forward evaluation.
type ForwardTuple = (
    String,
    String,
    bool,
    bool,
    Vec<Vec<f64>>,
    Vec<f64>,
    Vec<String>,
    String,
    Option<IntervalTuple>,
    Vec<String>,
);

/// `(label, point, cost, constraints)` of one enumerated action.
type ActionTuple = (String, Vec<f64>, f64, Vec<(String, bool)>);

fn inverse_error(error: InverseOutcomeError) -> PyErr {
    crate::refusal(error.code, format!("{}: {}", error.detail, error.message))
}

fn invalid_wire(what: &str, name: &str) -> PyErr {
    crate::refusal(
        reason_code!("invalid_argument"),
        format!("inverse.invalid_forward: unknown {what} `{name}`"),
    )
}

fn parse_support(name: &str) -> Option<SupportStatus> {
    [
        SupportStatus::Supported,
        SupportStatus::WeakOverlap,
        SupportStatus::Extrapolative,
        SupportStatus::OutsideEmpiricalSupport,
        SupportStatus::MissingEvidence,
    ]
    .into_iter()
    .find(|status| status.as_str() == name)
}

fn parse_interpretation(name: &str) -> Option<IntervalInterpretation> {
    [IntervalInterpretation::Confidence, IntervalInterpretation::Credible]
        .into_iter()
        .find(|interpretation| interpretation.as_str() == name)
}

fn forward_from_tuple(tuple: ForwardTuple) -> PyResult<ForwardEvaluation> {
    let (
        outcome,
        population,
        mean_response,
        point_identified,
        points,
        mean,
        support,
        support_basis,
        interval,
        assumptions,
    ) = tuple;
    let support = support
        .iter()
        .map(|name| parse_support(name).ok_or_else(|| invalid_wire("support status", name)))
        .collect::<PyResult<Vec<_>>>()?;
    let support_basis = SupportBasis::parse(&support_basis)
        .ok_or_else(|| invalid_wire("support basis", &support_basis))?;
    let interval = interval
        .map(|(scope, interpretation, level, lower, upper)| {
            Ok::<_, PyErr>(ForwardInterval {
                scope: IntervalScope::parse(&scope)
                    .ok_or_else(|| invalid_wire("interval scope", &scope))?,
                interpretation: parse_interpretation(&interpretation)
                    .ok_or_else(|| invalid_wire("interval interpretation", &interpretation))?,
                level,
                lower,
                upper,
            })
        })
        .transpose()?;
    Ok(ForwardEvaluation {
        outcome,
        population,
        mean_response,
        point_identified,
        dimension: points.first().map_or(0, Vec::len),
        points,
        mean,
        support,
        support_basis,
        interval,
        assumptions,
    })
}

fn report_json(report: &InverseOutcomeReport) -> serde_json::Value {
    let actions = report
        .outcomes
        .iter()
        .map(|outcome| {
            serde_json::json!({
                "label": outcome.label,
                "point": outcome.point,
                "cost": outcome.cost,
                "status": outcome.status.as_str(),
                "reason": outcome.reason,
                "violated_constraints": outcome.violated_constraints,
                "estimate": outcome.estimate,
                "margin": outcome.margin,
                "within_tolerance": outcome.within_tolerance,
                "support_status": outcome.support_status.map(SupportStatus::as_str),
                "interval": outcome.interval.map(|(lower, upper)| [lower, upper]),
                "robustly_feasible": outcome.robustly_feasible,
            })
        })
        .collect::<Vec<_>>();
    serde_json::json!({
        "query": {
            "kind": "target_mean",
            "threshold": report.threshold,
            "direction": report.direction.as_str(),
        },
        "tolerance": report.tolerance,
        "budget": report.budget,
        "outcome": report.outcome,
        "population": report.population,
        "assumptions": report.assumptions,
        "support_basis": report.support_basis.as_str(),
        "interval": report.interval.map(|meta| serde_json::json!({
            "scope": meta.scope.as_str(),
            "interpretation": meta.interpretation.as_str(),
            "level": meta.level,
        })),
        "actions": actions,
        "feasible": report.feasible,
        "infeasible": report.infeasible,
        "unsupported": report.unsupported,
        "unevaluated": report.unevaluated,
        "cheapest_feasible": report.cheapest_feasible,
        "robustly_feasible": report.robustly_feasible,
        "enumerated": report.enumerated.as_str(),
        "inference_claim": "point_only",
        "scope_note": antecedent::INVERSE_SCOPE_NOTE,
        "identity": report.identity,
    })
}

/// Classify an enumerated action grid against a licensed forward response.
///
/// `query_kind` is `target_mean`, `probability_target`, `quantile_target` or
/// `observational_scenarios`; every kind but the first is a typed refusal.
#[pyfunction]
#[doc(hidden)]
#[pyo3(signature = (
    query_kind, threshold, direction, forward, actions, *, probability=None, level=None,
    budget=None, tolerance=1e-9, cancel=None
))]
#[allow(clippy::too_many_arguments)] // Mirrors the Python signature.
fn classify_inverse_outcome_stage(
    py: Python<'_>,
    query_kind: &str,
    threshold: f64,
    direction: &str,
    forward: ForwardTuple,
    actions: Vec<ActionTuple>,
    probability: Option<f64>,
    level: Option<f64>,
    budget: Option<f64>,
    tolerance: f64,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Py<PyAny>> {
    let query = match query_kind {
        "target_mean" => InverseQuery::TargetMean {
            threshold,
            direction: TargetDirection::parse(direction).ok_or_else(|| {
                crate::refusal(
                    reason_code!("invalid_argument"),
                    format!("inverse.invalid_target: unknown direction `{direction}`"),
                )
            })?,
        },
        "probability_target" => InverseQuery::ChanceConstraint {
            threshold,
            probability: probability.unwrap_or(f64::NAN),
        },
        "quantile_target" => InverseQuery::Quantile { level: level.unwrap_or(f64::NAN), threshold },
        "observational_scenarios" => InverseQuery::ObservationalScenarios,
        other => {
            return Err(crate::refusal(
                reason_code!("invalid_argument"),
                format!("inverse.invalid_target: unknown query kind `{other}`"),
            ));
        }
    };
    let forward = forward_from_tuple(forward)?;
    let actions = actions
        .into_iter()
        .map(|(label, point, cost, constraints)| ActionSpec {
            label,
            point,
            cost,
            constraints: constraints
                .into_iter()
                .map(|(name, satisfied)| ActionConstraint { name, satisfied })
                .collect(),
        })
        .collect::<Vec<_>>();
    let ctx = crate::py_execution_context_cancel(0, 1, cancel);
    let report =
        classify_inverse_outcome(&query, &forward, &actions, budget, tolerance, &ctx.cancellation)
            .map_err(inverse_error)?;
    to_py_json(py, &report_json(&report))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(classify_inverse_outcome_stage, module)?)?;
    Ok(())
}
