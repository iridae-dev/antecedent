//! Python bridge for the 2.3.0 A1 selection-invariance report.
//!
//! The report is derived, never stored: it is a function of the decided scenario set
//! (read from each checked derivation) and the estimated report, so it is recomputed
//! from a prepared stage and is bound to no artifact byte. Every report, rule,
//! invariance and obstruction is Rust's; this module only names the variables and
//! hands the structure to Python as JSON.

use antecedent_core::{VariableId, reason_code};
use antecedent_estimate::EstimationError;
use antecedent_estimate::scenario_invariance_report::{EnvelopeSide, ScenarioSetInvarianceReport};
use antecedent_identify::sid::scenario_invariance::{
    InvarianceBody, InvarianceReport, ObstructionWitness,
};
use pyo3::prelude::*;
use serde_json::{Value, json};

use crate::transport_scenario_api::PreparedTransportScenariosStage;

/// `(report json, refusal json)`.
type Payload = (Option<String>, Option<String>);

fn name_of(names: &[String], variable: VariableId) -> String {
    names.get(variable.as_usize()).cloned().unwrap_or_else(|| format!("#{}", variable.raw()))
}

fn variables(names: &[String], values: &[VariableId]) -> Vec<String> {
    values.iter().map(|v| name_of(names, *v)).collect()
}

fn edges(names: &[String], values: &[(VariableId, VariableId)]) -> Vec<[String; 2]> {
    values.iter().map(|(a, b)| [name_of(names, *a), name_of(names, *b)]).collect()
}

fn witness_json(names: &[String], witness: &ObstructionWitness) -> Value {
    json!({
        "kind": witness.kind,
        "larger_nodes": variables(names, &witness.larger_nodes),
        "smaller_nodes": variables(names, &witness.smaller_nodes),
        "larger_directed": edges(names, &witness.larger_directed),
        "larger_bidirected": edges(names, &witness.larger_bidirected),
        "smaller_directed": edges(names, &witness.smaller_directed),
        "smaller_bidirected": edges(names, &witness.smaller_bidirected),
        "selection_targets_in_larger": variables(names, &witness.selection_targets_in_larger),
        "moves": variables(names, &witness.moves),
        "remaining": variables(names, &witness.remaining),
    })
}

fn body_json(names: &[String], body: &InvarianceBody) -> Value {
    match body {
        InvarianceBody::Identified { rules, invariances, target_factors, conditional } => json!({
            "kind": "identified",
            "rules": rules,
            "invariances": invariances.iter().map(|item| json!({
                "population": item.population.as_ref(),
                "variables": variables(names, &item.variables),
                "conditioned_on": variables(names, &item.conditioned_on),
                "do_set": variables(names, &item.do_set),
                "regime": item.regime,
                "invariant_mechanisms": variables(names, &item.invariant_mechanisms),
                "district_selection_targets": variables(names, &item.district_selection_targets),
                "rule": item.rule,
            })).collect::<Vec<_>>(),
            "target_factors": target_factors.iter().map(|item| json!({
                "variables": variables(names, &item.variables),
                "conditioned_on": variables(names, &item.conditioned_on),
                "regime": item.regime,
            })).collect::<Vec<_>>(),
            "conditional": conditional.as_ref().map(|reduction| json!({
                "moves": variables(names, &reduction.moves),
                "remaining": variables(names, &reduction.remaining),
            })),
        }),
        InvarianceBody::Obstructed { witness } => json!({
            "kind": "obstructed",
            "witness": witness_json(names, witness),
        }),
        InvarianceBody::Undecided { status, obligations, candidate } => json!({
            "kind": "undecided",
            "status": status,
            "obligations": obligations.iter().map(|o| &**o).collect::<Vec<&str>>(),
            "candidate": candidate.as_ref().map(|c| witness_json(names, c)),
        }),
    }
}

fn report_json(names: &[String], report: &InvarianceReport) -> Value {
    let selection = report.selection();
    json!({
        "status": report.status(),
        "identity": report.identity(),
        "canonical_text": report.canonical_text(),
        "selection": {
            "targets": variables(names, &selection.selection_targets),
            "shared_mechanisms": variables(names, &selection.shared_mechanisms),
            "directed_edges": edges(names, &selection.directed_edges),
            "bidirected_edges": edges(names, &selection.bidirected_edges),
        },
        "body": body_json(names, report.body()),
    })
}

fn set_json(names: &[String], set: &ScenarioSetInvarianceReport) -> Value {
    json!({
        "scenarios": set.scenarios.iter().map(|s| json!({
            "name": s.name.as_ref(),
            "result_status": s.result_status,
            "report": report_json(names, &s.report),
        })).collect::<Vec<_>>(),
        "extremes": set.extremes.iter().map(|e| json!({
            "outcome": name_of(names, e.outcome),
            "side": match e.side {
                EnvelopeSide::Lower => "lower",
                EnvelopeSide::Upper => "upper",
            },
            "value": e.value,
            "scenario": e.scenario.as_ref(),
            "report": report_json(names, &e.report),
        })).collect::<Vec<_>>(),
        "canonical_text": set.canonical_text(),
    })
}

fn refusal_json(code: &str, detail: &str, message: &str) -> String {
    json!({
        "code": code,
        "stage": "scenario_invariance",
        "detail": detail,
        "message": message,
        "offending": Value::Null,
        "remedy": Value::Null,
    })
    .to_string()
}

/// The invariance report of an estimated transport-scenario stage:
/// `(report json, refusal json)`. A stage that was not estimated, or an object that
/// is not a scenario stage, is a refusal value for Python to raise.
#[pyfunction]
fn scenario_invariance_report(stage: &Bound<'_, PyAny>) -> PyResult<Payload> {
    let Ok(stage) = stage.extract::<PyRef<'_, PreparedTransportScenariosStage>>() else {
        return Ok((
            None,
            Some(refusal_json(
                reason_code!("invalid_argument"),
                "scenario_invariance.wrong_result_type",
                "supply the stage returned by prepare_transport_scenarios",
            )),
        ));
    };
    let Some((prepared, report, names)) = stage.prepared_and_report() else {
        return Ok((
            None,
            Some(refusal_json(
                reason_code!("not_executed"),
                "scenario_invariance.not_estimated",
                "call estimate() on the scenario stage before asking for its invariance report",
            )),
        ));
    };
    match prepared.invariance_report(report) {
        Ok(set) => Ok((Some(set_json(names, &set).to_string()), None)),
        Err(EstimationError::Refused { code, message }) => {
            let (detail, text) = message
                .split_once(": ")
                .unwrap_or(("scenario_invariance.refused", message.as_str()));
            Ok((None, Some(refusal_json(code, detail, text))))
        }
        Err(other) => Err(crate::transport_common::error(other)),
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(scenario_invariance_report, m)?)
}
