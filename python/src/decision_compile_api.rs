//! Python bridge for compiling a study from a decision.
//!
//! Python declares the decision (terminal actions with an affine utility of the state, the
//! contract identity, the utility unit and the chance threshold), the carrier it demands of the
//! statistical program and the study (named columns, a DAG, a binary average effect, the
//! inference mode). Rust compiles the study with layer four of the lowered program filled from the
//! decision, checks that the route's statistical realization can publish the demanded carrier,
//! and refuses with a typed carrier error naming the next attempt. Nothing is fitted.

use std::sync::Arc;

use antecedent::analysis::DecisionCompileRequest;
use antecedent::planner::lowered::CarrierDemand;
use antecedent::{CausalError, CompileCause};
use antecedent_design::{AffineUtility, DecisionProblem};
use pyo3::prelude::*;

use crate::bridge::RefusalJson;
use crate::{detach_catch, value_err};

type Compiled = (Option<String>, Option<String>);

fn carrier_of(name: &str) -> PyResult<CarrierDemand> {
    match name {
        "mean" => Ok(CarrierDemand::Mean),
        "parameter_posterior" => Ok(CarrierDemand::ParameterPosterior),
        "outcome_law" => Ok(CarrierDemand::OutcomeLaw),
        other => Err(value_err(format!(
            "unknown carrier `{other}`; use mean, parameter_posterior or outcome_law"
        ))),
    }
}

/// The typed carrier refusal inside a compile error, as structured JSON.
fn carrier_refusal(error: &CausalError) -> Option<String> {
    let CausalError::Rejected(failure) = error.peeled() else { return None };
    let CompileCause::Decision(refusal) = failure.cause() else { return None };
    let refusal = **refusal;
    let port = refusal.port.map(|p| {
        serde_json::json!({
            "source_port": format!("{:?}", p.source_port),
            "legal_ports": p.legal_ports.iter().map(|l| format!("{l:?}")).collect::<Vec<_>>(),
        })
    });
    Some(
        RefusalJson::new(refusal.code, "decision_compile", refusal.detail, refusal.message)
            .offending(Some(refusal.realization))
            .remedy(Some(refusal.next_attempt.sentence()))
            .field("demanded", refusal.demanded.as_str())
            .field("realization", refusal.realization)
            .field("next_attempt", refusal.next_attempt.name())
            .field("missing_carrier", refusal.missing_carrier().map(|c| c.as_str()))
            .field("port", port.unwrap_or(serde_json::Value::Null))
            .json(),
    )
}

/// Compile the study for a terminal decision: `(program json, refusal json)`.
///
/// The program carries the lowered program identity, the demanded carrier, the actions,
/// constraints and chance threshold, the utility's per-action `(intercept, slope)` pairs, the
/// utility unit and the contract identity, and the identity the same study compiles to without a
/// decision (layer four differs, so the two never agree).
#[pyfunction]
#[pyo3(signature = (carrier, contract_identity, utility_unit, intercepts, slopes, names, columns, edges, treatment, outcome, *, chance_threshold=1.0, posterior=true, n_draws=256, prior_scale=10.0, memory_bytes=None, cancel=None))]
#[allow(
    clippy::too_many_arguments,
    reason = "one request: the decision, the carrier and the study"
)]
fn compile_decision_for_study(
    py: Python<'_>,
    carrier: &str,
    contract_identity: &str,
    utility_unit: &str,
    intercepts: Vec<f64>,
    slopes: Vec<f64>,
    names: Vec<String>,
    columns: Vec<Vec<f64>>,
    edges: Vec<(String, String)>,
    treatment: &str,
    outcome: &str,
    chance_threshold: f64,
    posterior: bool,
    n_draws: usize,
    prior_scale: f64,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Compiled> {
    let carrier = carrier_of(carrier)?;
    let utility = AffineUtility::new(intercepts, slopes)
        .map_err(|e| value_err(format!("the utility is not affine in the state: {e}")))?;
    let actions: Vec<usize> = (0..utility.intercepts.len()).collect();
    let mut problem: DecisionProblem<usize, f64> =
        DecisionProblem::new(actions, Arc::new(utility), vec![]);
    problem.chance_threshold = chance_threshold;
    let request = DecisionCompileRequest::new(carrier)
        .with_contract_identity(contract_identity)
        .with_utility_unit(utility_unit);
    let (treatment, outcome) = (treatment.to_owned(), outcome.to_owned());
    detach_catch(py, move || {
        let study = crate::design_ranking_api::evsi_study(
            &names,
            &columns,
            &edges,
            &treatment,
            &outcome,
            posterior,
            n_draws,
            prior_scale,
        )?;
        let ctx = crate::transport_common::execution_context(0, memory_bytes, cancel);
        let plan = match study.compile_for_decision(&problem, &request, &ctx) {
            Ok(plan) => plan,
            Err(error) => {
                return match carrier_refusal(&error) {
                    Some(refusal) => Ok((None, Some(refusal))),
                    None => Err(crate::py_err(error)),
                };
            }
        };
        let lowered = plan
            .lowered()
            .ok_or_else(|| value_err("the compiled plan carries no lowered program"))?;
        let decision = lowered
            .evidence()
            .decision()
            .ok_or_else(|| value_err("the compiled plan carries no decision program"))?;
        let estimate = study.compile(&ctx).map_err(crate::py_err)?;
        let estimate_identity = estimate
            .lowered()
            .ok_or_else(|| value_err("the estimate plan carries no lowered program"))?
            .identity()
            .to_hex();
        let body = serde_json::json!({
            "program_identity": lowered.identity().to_hex(),
            "estimate_program_identity": estimate_identity,
            "carrier": decision.carrier().as_str(),
            "actions": decision.actions(),
            "constraints": decision.constraints(),
            "chance_threshold": decision.chance_threshold(),
            "utility_unit": decision.utility_unit(),
            "contract_identity": decision.contract_identity(),
            "utility_coefficients": decision.utility().coefficients(),
        });
        Ok((Some(body.to_string()), None))
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(compile_decision_for_study, module)?)?;
    Ok(())
}
