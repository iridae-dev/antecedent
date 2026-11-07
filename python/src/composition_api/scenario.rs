//! Decisions and utilities over scenario claims.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    AtomStatus, AtomSupport, BTreeMap, Bound, ClaimError, CpdagScenarioRun, Deserialize,
    DistributionCalibration, IdentifiedUtility, InputSupport, IntervalReading, OutcomeBinding,
    Pair, PreparedTransportScenariosStage, PyAny, PyAnyMethods, PyErr, PyModule, PyModuleMethods,
    PyRef, PyResult, Python, RobustVerdict, ScenarioClaimBinding, ScenarioDecision,
    ScenarioSetReport, ScientificQuantity, ScientificQuantityWire, StructuralDecisionResult,
    StructuralPolicy, StructuralVerdict, SupportStatus, Value, VariableId,
    admissible_contract_from_json_refusal, check_size, detach_catch, envelope_identified_utilities,
    evaluate_cpdag_decision, evaluate_scenario_decision, external_refusal_json,
    interval_identified_utilities, invalid_json, json, pyfunction, report_identified_utilities,
    weighted_identified_utilities, wrap_pyfunction,
};

// ---------------------------------------------------------------------------
// Decisions over real scenario and CPDAG-completion claims.
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct InputSupportWire {
    action_id: String,
    input: usize,
    status: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SupportWire {
    overall: String,
    #[serde(default)]
    per_input: Vec<InputSupportWire>,
}

fn status_of(name: &str) -> Result<SupportStatus, String> {
    SupportStatus::from_name(name)
        .ok_or_else(|| invalid_json("scenario_decision.unknown_support_status", Some(name.into())))
}

fn support_from(wire: SupportWire) -> Result<AtomSupport, String> {
    Ok(AtomSupport {
        overall: status_of(&wire.overall)?,
        per_input: wire
            .per_input
            .into_iter()
            .map(|p| {
                Ok(InputSupport {
                    action_id: p.action_id,
                    input: p.input,
                    status: status_of(&p.status)?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?,
    })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OutcomeWire {
    outcome: String,
    quantity: ScientificQuantityWire,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ScenarioSupportWire {
    scenario: String,
    support: SupportWire,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BindingWire {
    outcomes: Vec<OutcomeWire>,
    calibration: String,
    source_id: String,
    provider_id: String,
    causal_contract_id: String,
    premises_digest: String,
    data_digest: String,
    #[serde(default)]
    support: Vec<ScenarioSupportWire>,
    #[serde(default)]
    default_support: Option<SupportWire>,
}

fn binding_from(json: &str, names: &[String]) -> Result<ScenarioClaimBinding, String> {
    let wire: BindingWire = serde_json::from_str(json)
        .map_err(|e| invalid_json("scenario_decision.invalid_binding", Some(e.to_string())))?;
    let mut outcomes = Vec::with_capacity(wire.outcomes.len());
    for item in wire.outcomes {
        let unknown =
            || invalid_json("scenario_decision.unknown_outcome", Some(item.outcome.clone()));
        let position = names.iter().position(|name| *name == item.outcome).ok_or_else(unknown)?;
        let id = u32::try_from(position).map_err(|_| unknown())?;
        let quantity = ScientificQuantity::try_from(item.quantity)
            .map_err(|e| invalid_json("scenario_decision.invalid_quantity", Some(e.to_owned())))?;
        outcomes.push(OutcomeBinding { outcome: VariableId::from_raw(id), quantity });
    }
    let calibration = match wire.calibration.as_str() {
        "exact" => DistributionCalibration::Exact,
        "point_only" => DistributionCalibration::PointOnly,
        // A measured calibration cites a coordinate-specific record the binding cannot carry;
        // the laws are bound as exact, point-only or not_measured.
        "not_measured" => DistributionCalibration::NotMeasured,
        other => {
            return Err(invalid_json("scenario_decision.unknown_calibration", Some(other.into())));
        }
    };
    let mut binding = ScenarioClaimBinding::new(
        outcomes,
        &wire.source_id,
        &wire.provider_id,
        &wire.causal_contract_id,
        &wire.premises_digest,
        &wire.data_digest,
    );
    binding.calibration = calibration;
    if let Some(default) = wire.default_support {
        binding = binding.with_default_support(support_from(default)?);
    }
    for item in wire.support {
        binding = binding.with_support(&item.scenario, support_from(item.support)?);
    }
    Ok(binding)
}

fn policy_from(name: &str) -> Result<StructuralPolicy, String> {
    match name {
        "require_invariant_best_action" => Ok(StructuralPolicy::RequireInvariantBestAction),
        "maximin" => Ok(StructuralPolicy::Maximin),
        "bayes_over_structures" => Ok(StructuralPolicy::BayesOverStructures),
        "report_only" => Ok(StructuralPolicy::ReportOnly),
        other => Err(invalid_json("scenario_decision.unknown_policy", Some(other.into()))),
    }
}

const fn policy_name(policy: StructuralPolicy) -> &'static str {
    match policy {
        StructuralPolicy::RequireInvariantBestAction => "require_invariant_best_action",
        StructuralPolicy::Maximin => "maximin",
        StructuralPolicy::BayesOverStructures => "bayes_over_structures",
        StructuralPolicy::ReportOnly => "report_only",
    }
}

fn claim_refusal(error: &ClaimError) -> String {
    external_refusal_json(&error.to_refusal())
}

/// Declared weights must be the scenario set's own: all of them, equal to the
/// report's. A CPDAG completion has none to declare: counts are never probabilities.
fn check_declared_weights(
    scenarios: Option<&[(String, Option<f64>)]>,
    weights: Option<&BTreeMap<String, f64>>,
) -> Result<(), String> {
    let Some(weights) = weights else {
        return Ok(());
    };
    let Some(scenarios) = scenarios else {
        return Err(claim_refusal(&ClaimError::UnsupportedClaimShape(
            "completion counts are never probabilities; weights cannot be declared for CPDAG completions",
        )));
    };
    if scenarios.iter().all(|(_, weight)| weight.is_none()) {
        return Err(claim_refusal(&ClaimError::UnsupportedClaimShape(
            "scenario weights are declared only through the scenario set",
        )));
    }
    let same_ids = scenarios.len() == weights.len()
        && scenarios.iter().all(|(id, _)| weights.contains_key(id));
    let same_values = scenarios.iter().all(|(id, declared)| {
        matches!((declared, weights.get(id)), (Some(a), Some(b)) if (a - b).abs() <= 1e-12)
    });
    if same_ids && same_values {
        Ok(())
    } else {
        Err(claim_refusal(&ClaimError::IdentityMismatch("weights".to_owned())))
    }
}

fn structural_verdict_json(verdict: &StructuralVerdict) -> Value {
    match verdict {
        StructuralVerdict::InvariantBest(id) => json!({"kind": "invariant_best", "action": id}),
        StructuralVerdict::NoInvariantBest(leaders) => {
            json!({"kind": "no_invariant_best", "leaders": leaders})
        }
        StructuralVerdict::WorstCaseChoice(id) => {
            json!({"kind": "worst_case_choice", "action": id})
        }
        StructuralVerdict::Tied(actions) => json!({"kind": "tied", "actions": actions}),
        StructuralVerdict::BayesChoice { action, evaluated_mass } => {
            json!({"kind": "bayes_choice", "action": action, "evaluated_mass": evaluated_mass})
        }
        StructuralVerdict::ReportOnly => json!({"kind": "report_only"}),
        StructuralVerdict::InsufficientScience(why) => {
            json!({"kind": "insufficient_science", "reason": why})
        }
        StructuralVerdict::NoAdmissibleAction => json!({"kind": "no_admissible_action"}),
    }
}

fn structural_json(result: &StructuralDecisionResult) -> Value {
    let atoms: Vec<Value> = result
        .atoms
        .iter()
        .map(|atom| {
            let (status, reason, actions) = match &atom.status {
                AtomStatus::Evaluated(inner) => (
                    "evaluated",
                    None,
                    Some(
                        inner
                            .actions
                            .iter()
                            .map(|o| {
                                json!({"id": o.id, "admissible": o.admissible, "value": o.value})
                            })
                            .collect::<Vec<_>>(),
                    ),
                ),
                AtomStatus::Unidentified => ("unidentified", None, None),
                AtomStatus::Unevaluated(why) => ("unevaluated", Some(why.clone()), None),
            };
            json!({
                "id": atom.id,
                "probability": atom.probability,
                "status": status,
                "reason": reason,
                "actions": actions,
            })
        })
        .collect();
    let actions: Vec<Value> = result
        .actions
        .iter()
        .map(|a| {
            json!({
                "id": a.id,
                "per_atom": a.per_atom,
                "range": a.range,
                "weighted_value": a.weighted_value,
                "mass_where_best": a.mass_where_best,
                "excluded_in": a.excluded_in,
            })
        })
        .collect();
    json!({
        "contract_identity": result.contract_identity,
        "policy": policy_name(result.policy),
        "atoms": atoms,
        "actions": actions,
        "unidentified_mass": result.unidentified_mass,
        "unevaluated_mass": result.unevaluated_mass,
        "evaluated_mass": result.evaluated_mass,
        "verdict": structural_verdict_json(&result.verdict),
    })
}

fn robust_verdict_json(verdict: &RobustVerdict) -> Value {
    match verdict {
        RobustVerdict::StructurallyRobust(id) => {
            json!({"kind": "structurally_robust", "action": id})
        }
        RobustVerdict::SupportRobust(id) => json!({"kind": "support_robust", "action": id}),
        RobustVerdict::SupportDependent { supported_choice, unrestricted_choice } => json!({
            "kind": "support_dependent",
            "action": supported_choice,
            "unrestricted_choice": unrestricted_choice,
        }),
        RobustVerdict::GraphDependentChoice(leaders) => {
            json!({"kind": "graph_dependent_choice", "leaders": leaders})
        }
        RobustVerdict::UnsupportedExtrapolation => json!({"kind": "unsupported_extrapolation"}),
        RobustVerdict::InsufficientClaims(why) => {
            json!({"kind": "insufficient_claims", "reason": why})
        }
        RobustVerdict::NoAdmissibleAction => json!({"kind": "no_admissible_action"}),
        RobustVerdict::WorstCaseChoice(id) => json!({"kind": "worst_case_choice", "action": id}),
        RobustVerdict::Tied(actions) => json!({"kind": "tied", "actions": actions}),
        RobustVerdict::BayesChoice { action, evaluated_mass } => {
            json!({"kind": "bayes_choice", "action": action, "evaluated_mass": evaluated_mass})
        }
        RobustVerdict::ReportOnly => json!({"kind": "report_only"}),
    }
}

/// Decide over the claims of a real scenario or CPDAG-completion result.
///
/// `kind` is `transport` (a prepared scenario stage after `estimate()`) or
/// `cpdag` (a completion run). `weights_json`, when given, is the
/// `{scenario: weight}` map the caller expects the scenario set to have
/// declared; it is checked against the report and never applied to it.
/// Returns the atom table, masses, leaders and verdicts as JSON, or a refusal.
#[pyfunction]
#[pyo3(signature = (declaration_json, kind, result, binding_json, policy, weights_json=None, expected_causal=None))]
#[allow(clippy::too_many_lines)]
fn decide_scenario_claims(
    py: Python<'_>,
    declaration_json: &str,
    kind: &str,
    result: &Bound<'_, PyAny>,
    binding_json: &str,
    policy: &str,
    weights_json: Option<&str>,
    expected_causal: Option<&str>,
) -> PyResult<Pair> {
    check_size(declaration_json)?;
    check_size(binding_json)?;
    if let Some(text) = weights_json {
        check_size(text)?;
    }
    let run = || -> Result<String, DecisionStop> {
        let contract = admissible_contract_from_json_refusal(declaration_json)
            .map_err(|refusal| external_refusal_json(&refusal))?;
        let policy = policy_from(policy)?;
        let weights: Option<BTreeMap<String, f64>> = weights_json
            .map(serde_json::from_str)
            .transpose()
            .map_err(|e| invalid_json("scenario_decision.invalid_weights", Some(e.to_string())))?;
        let wrong =
            |slot: &str| invalid_json("scenario_decision.wrong_result_type", Some(slot.to_owned()));
        let decided = match kind {
            "transport" => {
                let stage = result
                    .extract::<PyRef<'_, PreparedTransportScenariosStage>>()
                    .map_err(|_| wrong("transport"))?;
                let (report, names) = stage.last_report().ok_or_else(|| {
                    invalid_json("scenario_decision.not_estimated", Some("estimate() first".into()))
                })?;
                let scenarios: Vec<(String, Option<f64>)> =
                    report.scenarios.iter().map(|s| (s.name.to_string(), s.weight)).collect();
                check_declared_weights(Some(scenarios.as_slice()), weights.as_ref())?;
                let binding = binding_from(binding_json, names)?;
                let expected = expected_causal.unwrap_or(&binding.causal_contract_id);
                detach_catch(py, || {
                    Ok(evaluate_scenario_decision(
                        &contract, expected, report, &binding, policy, None,
                    )
                    .map_err(|e| external_refusal_json(&e)))
                })
                .map_err(DecisionStop::Python)??
            }
            "cpdag" => {
                let completions =
                    result.extract::<PyRef<'_, CpdagScenarioRun>>().map_err(|_| wrong("cpdag"))?;
                let (report, names) = completions.report_and_names();
                check_declared_weights(None, weights.as_ref())?;
                let binding = binding_from(binding_json, names)?;
                let expected = expected_causal.unwrap_or(&binding.causal_contract_id);
                detach_catch(py, || {
                    Ok(evaluate_cpdag_decision(&contract, expected, report, &binding, policy, None)
                        .map_err(|e| external_refusal_json(&e)))
                })
                .map_err(DecisionStop::Python)??
            }
            other => {
                return Err(
                    invalid_json("scenario_decision.unknown_kind", Some(other.into())).into()
                );
            }
        };
        let ScenarioDecision { claims, decision, leaders } = decided;
        let atoms: Vec<Value> = claims
            .atoms
            .iter()
            .map(|a| {
                json!({
                    "id": a.id,
                    "status": a.status,
                    "weight": a.weight,
                    "support": a.support.as_str(),
                    "digest": a.digest,
                    "detail": a.detail,
                })
            })
            .collect();
        let masses: Vec<Value> = claims
            .masses
            .iter()
            .map(|m| json!({"status": m.status, "count": m.count, "mass": m.mass}))
            .collect();
        let scenarios: Vec<Value> = claims
            .source
            .scenarios
            .iter()
            .map(|s| json!({"id": s.id, "status": s.status, "digest": s.digest}))
            .collect();
        Ok(json!({
            "kind": claims.source.kind,
            "source": {
                "identity": claims.source.identity,
                "cpdag_identity": claims.source.cpdag_identity,
                "premises_digest": claims.source.premises_digest,
                "data_digest": claims.source.data_digest,
                "scenarios": scenarios,
            },
            "declared_weights": claims.declared_weights,
            "atoms": atoms,
            "masses": masses,
            "residual_mass": claims.residual_mass,
            "structural": structural_json(&decision.structural),
            "leaders": leaders,
            "robust": {
                "verdict": robust_verdict_json(&decision.robust.verdict),
                "unsupported_atoms": decision.robust.unsupported_atoms,
                "unsupported_mass": decision.robust.unsupported_mass,
            },
            "completion_counts": decision.completion_counts,
        })
        .to_string())
    };
    match run() {
        Ok(json) => Ok((Some(json), None)),
        Err(DecisionStop::Refusal(refusal)) => Ok((None, Some(refusal))),
        Err(DecisionStop::Python(error)) => Err(error),
    }
}

fn identified_utilities_json(utilities: &[IdentifiedUtility]) -> String {
    let items: Vec<Value> = utilities
        .iter()
        .map(|u| {
            json!({
                "action_id": u.action_id,
                "lower": u.utility.lower,
                "upper": u.utility.upper,
                "hard_exclusions": u.hard_exclusions,
                "support": {
                    "overall": u.support.overall.as_str(),
                    "per_input": u.support.per_input.iter().map(|p| json!({
                        "action_id": p.action_id,
                        "input": p.input,
                        "status": p.status.as_str(),
                    })).collect::<Vec<_>>(),
                },
            })
        })
        .collect();
    Value::Array(items).to_string()
}

fn support_of(json: &str) -> Result<AtomSupport, String> {
    let wire: SupportWire = serde_json::from_str(json)
        .map_err(|e| invalid_json("scenario_decision.invalid_support", Some(e.to_string())))?;
    support_from(wire)
}

/// Per-action utility intervals from the report of a scenario set or a CPDAG completion run.
///
/// `kind` is `transport` or `cpdag`; `route` is `report` (the weighted range when the
/// report carries one, else the envelope), `envelope` or `weighted`. The interval covers the
/// identified scenarios only. Returns the utilities as JSON, or a refusal.
#[pyfunction]
#[pyo3(signature = (declaration_json, kind, result, binding_json, support_json, route))]
fn scenario_identified_utilities(
    declaration_json: &str,
    kind: &str,
    result: &Bound<'_, PyAny>,
    binding_json: &str,
    support_json: &str,
    route: &str,
) -> PyResult<Pair> {
    check_size(declaration_json)?;
    check_size(binding_json)?;
    check_size(support_json)?;
    let run = || -> Result<String, String> {
        let contract = admissible_contract_from_json_refusal(declaration_json)
            .map_err(|refusal| external_refusal_json(&refusal))?;
        let wrong =
            |slot: &str| invalid_json("scenario_decision.wrong_result_type", Some(slot.to_owned()));
        let (report, names): (Option<ScenarioSetReport>, Vec<String>) = match kind {
            "transport" => {
                let stage = result
                    .extract::<PyRef<'_, PreparedTransportScenariosStage>>()
                    .map_err(|_| wrong("transport"))?;
                let (report, names) = stage.last_report().ok_or_else(|| {
                    invalid_json("scenario_decision.not_estimated", Some("estimate() first".into()))
                })?;
                (Some(report.clone()), names.to_vec())
            }
            "cpdag" => {
                let run =
                    result.extract::<PyRef<'_, CpdagScenarioRun>>().map_err(|_| wrong("cpdag"))?;
                let (report, names) = run.report_and_names();
                (report.report.clone(), names.to_vec())
            }
            other => {
                return Err(invalid_json("scenario_decision.unknown_kind", Some(other.into())));
            }
        };
        let binding = binding_from(binding_json, &names)?;
        let support = support_of(support_json)?;
        let unavailable = || claim_refusal(&ClaimError::RangeUnavailable);
        let report = report.ok_or_else(unavailable)?;
        let utilities = match route {
            "report" => report_identified_utilities(&contract, &report, &binding, &support),
            "envelope" => {
                let envelope = report.envelope.as_ref().ok_or_else(unavailable)?;
                envelope_identified_utilities(&contract, envelope, &binding, &support)
            }
            "weighted" => {
                let weighted = report.weighted.as_ref().ok_or_else(unavailable)?;
                weighted_identified_utilities(&contract, weighted, &binding, &support)
            }
            other => {
                return Err(invalid_json(
                    "scenario_decision.unknown_utility_route",
                    Some(other.into()),
                ));
            }
        }
        .map_err(|e| claim_refusal(&e))?;
        Ok(identified_utilities_json(&utilities))
    };
    Ok(match run() {
        Ok(json) => (Some(json), None),
        Err(refusal) => (None, Some(refusal)),
    })
}

#[derive(Deserialize)]
struct IntervalWire {
    quantity: ScientificQuantityWire,
    interval: antecedent::io::IdentifiedSetIntervalWire,
}

/// Per-action utility intervals from one identified-set interval per input quantity.
///
/// `intervals_json` is a list of `{quantity, interval}`; `reading` is `estimated_bounds` or
/// `coverage_interval`. Returns the utilities as JSON, or a refusal.
#[pyfunction]
#[pyo3(signature = (declaration_json, intervals_json, reading, support_json))]
fn interval_identified_action_utilities(
    declaration_json: &str,
    intervals_json: &str,
    reading: &str,
    support_json: &str,
) -> PyResult<Pair> {
    check_size(declaration_json)?;
    check_size(intervals_json)?;
    check_size(support_json)?;
    let run = || -> Result<String, String> {
        let contract = admissible_contract_from_json_refusal(declaration_json)
            .map_err(|refusal| external_refusal_json(&refusal))?;
        let reading = match reading {
            "estimated_bounds" => IntervalReading::EstimatedBounds,
            "coverage_interval" => IntervalReading::CoverageInterval,
            other => {
                return Err(invalid_json("scenario_decision.unknown_reading", Some(other.into())));
            }
        };
        let wires: Vec<IntervalWire> = serde_json::from_str(intervals_json)
            .map_err(|e| invalid_json("scenario_decision.invalid_interval", Some(e.to_string())))?;
        let mut intervals = Vec::with_capacity(wires.len());
        for wire in wires {
            let quantity = ScientificQuantity::try_from(wire.quantity).map_err(|e| {
                invalid_json("scenario_decision.invalid_quantity", Some(e.to_owned()))
            })?;
            let interval = antecedent::io::identified_set_interval_from_wire(&wire.interval)
                .map_err(|e| {
                    invalid_json("scenario_decision.invalid_interval", Some(e.to_string()))
                })?;
            intervals.push((quantity, interval));
        }
        let support = support_of(support_json)?;
        let utilities = interval_identified_utilities(&contract, &intervals, reading, &support)
            .map_err(|e| claim_refusal(&e))?;
        Ok(identified_utilities_json(&utilities))
    };
    Ok(match run() {
        Ok(json) => (Some(json), None),
        Err(refusal) => (None, Some(refusal)),
    })
}

/// Why a scenario decision stopped: a structured refusal, or an interrupt raised while the native
/// work had released the GIL.
enum DecisionStop {
    Refusal(String),
    Python(PyErr),
}

impl From<String> for DecisionStop {
    fn from(refusal: String) -> Self {
        Self::Refusal(refusal)
    }
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(decide_scenario_claims, m)?)?;
    m.add_function(wrap_pyfunction!(scenario_identified_utilities, m)?)?;
    m.add_function(wrap_pyfunction!(interval_identified_action_utilities, m)?)?;
    Ok(())
}
