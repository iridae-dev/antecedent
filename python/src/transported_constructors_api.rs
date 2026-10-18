//! Python bridge for the typed constructors of a transported response.
//!
//! Python declares the typed source of each route (a finite sequential model and policy with a
//! selection diagram or a time-invariance declaration, an adaptive trial's retained contract with
//! its strata and diagram, an evidence ledger with its contributions, a surrogate bridge with its
//! validation trials and diagram) as JSON; the route runs in Rust and the response is built from
//! the route's own result, so the populations, transport identities and assumptions are copied
//! from the producing result and never typed. A refusal comes back as structured JSON for the
//! Python layer to raise as a typed exception; a malformed declaration is an input error.
//!
//! Every response is a point labelled `extrapolative` at best; nothing here supplies an interval,
//! a law or a calibration claim. The synthesis constructor evaluates against a copy of the
//! session's ledger: the session itself is not changed.

use antecedent::analysis::adaptive_inference::{
    AdaptiveInferenceRefusal, AdaptiveInferenceSession,
};
use antecedent::analysis::sequential_evaluation as seq;
use antecedent::analysis::transported_constructors::{
    ActionAlphabet, ActionUtility, AdaptiveTransportRequest, AdaptiveTransportedValue,
    BridgeSelectionDiagram, HistoryAlphabet, MechanismStatus, ModelMechanism, SelectionDiagram,
    TargetUtility, TemporalPolicyRequest, TemporalTarget, TimeInvariance, TransportedContribution,
    TransportedPolicyRequest, TransportedSurrogateBridge, TransportedSynthesisMethod,
    TrialMechanism, TrialSelectionDiagram, UtilityProvenance, adaptive_trial_response,
    evaluate_temporal_policy, evaluate_transported_policy, surrogate_bridge_response,
    temporal_policy_response, transport_adaptive_trial, transported_policy_response,
    transported_synthesis_response, verify_adaptive_transport, verify_temporal_policy,
    verify_transported_policy,
};
use antecedent_contract::bridge::{BridgeScope, TrialEffectPair, TrialLevelValidation};
use antecedent_contract::evidence_ledger::SourceId;
use antecedent_core::SemanticDigest;
use antecedent_io::adaptive_monitoring_contract_artifact::{
    DeclarationWire, declaration_from_wire,
};
use antecedent_io::quantity_wire::ScientificQuantityWire;
use antecedent_stats::synthesis::{MeanPrior, TauPrior};
use pyo3::prelude::*;
use serde::Deserialize;
use serde_json::{Value as Json, json};

use crate::bridge::{check_declared_size, external_refusal_json};
use crate::evidence_ledger_api::EvidenceLedgerHandle;
use crate::transport_common::execution_context;
use crate::transported_response_api::{TransportedResponseHandle, quantity_of};

const MAX_JSON_BYTES: usize = crate::bridge::MAX_DECLARED_ARTIFACT_BYTES;

/// `(response, evaluation json, refusal json)`.
type Built = (Option<TransportedResponseHandle>, Option<String>, Option<String>);

fn refused(refusal: &antecedent_contract::ExternalRefusal) -> Built {
    (None, None, Some(external_refusal_json(refusal)))
}

fn declaration<T: serde::de::DeserializeOwned>(text: &str, what: &str) -> PyResult<T> {
    check_declared_size(text, MAX_JSON_BYTES, what)?;
    serde_json::from_str(text)
        .map_err(|error| crate::value_err(format!("invalid {what} declaration: {error}")))
}

// ---------------------------------------------------------------------------
// Declarations
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MechanismIn {
    mechanism: String,
    #[serde(default)]
    action: Option<usize>,
    status: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PopulationDiagramIn {
    source_population: String,
    target_population: String,
    mechanisms: Vec<MechanismIn>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct InvarianceIn {
    source_period: String,
    target_period: String,
    horizon: usize,
    mechanisms: Vec<MechanismIn>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ActionAlphabetIn {
    label: String,
    symbols: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TargetIn {
    period: String,
    horizon: usize,
    alphabet: Vec<ActionAlphabetIn>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DecisionIn {
    history: Vec<(usize, usize)>,
    actions: Vec<(usize, f64)>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PolicyIn {
    version: String,
    decisions: Vec<DecisionIn>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ActionUtilityIn {
    cost: f64,
    immediate: Vec<f64>,
    #[serde(default)]
    delayed: Option<Vec<f64>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TargetUtilityIn {
    source: String,
    digest: String,
    terminal: Vec<f64>,
    actions: Vec<ActionUtilityIn>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TrialMechanismIn {
    mechanism: String,
    status: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TrialDiagramIn {
    source_population: String,
    target_population: String,
    mechanisms: Vec<TrialMechanismIn>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ContributionIn {
    source: String,
    estimate: f64,
    sampling_variance: f64,
    #[serde(default)]
    transport_variance: Option<f64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MeanPriorIn {
    kind: String,
    #[serde(default)]
    mean: Option<f64>,
    #[serde(default)]
    variance: Option<f64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TauPriorIn {
    support: Vec<f64>,
    probabilities: Vec<f64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MethodIn {
    kind: String,
    #[serde(default)]
    sampling_covariance: Option<Vec<f64>>,
    #[serde(default)]
    mean_prior: Option<MeanPriorIn>,
    #[serde(default)]
    tau_prior: Option<TauPriorIn>,
    #[serde(default)]
    prior: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ScopeIn {
    population_id: String,
    regime_id: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ValidationIn {
    trials: Vec<(f64, f64)>,
    min_trials: u32,
    min_r_squared: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BridgeDiagramIn {
    source_population: String,
    target_population: String,
    selection_targets: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BridgeIn {
    surrogate: ScientificQuantityWire,
    terminal: ScientificQuantityWire,
    validation: ValidationIn,
    validated_in: ScopeIn,
    validation_evidence: String,
    diagram: BridgeDiagramIn,
    terminal_variable: String,
    relation_modifiers: Vec<String>,
}

fn status_of(name: &str) -> PyResult<MechanismStatus> {
    match name {
        "invariant" => Ok(MechanismStatus::Invariant),
        "shifted" => Ok(MechanismStatus::Shifted),
        other => Err(crate::value_err(format!(
            "mechanism status must be `invariant` or `shifted`, not `{other}`"
        ))),
    }
}

fn mechanisms_of(rows: Vec<MechanismIn>) -> PyResult<Vec<(ModelMechanism, MechanismStatus)>> {
    rows.into_iter()
        .map(|row| {
            let action = |name: &str| {
                row.action.ok_or_else(|| {
                    crate::value_err(format!("mechanism `{name}` names the action it belongs to"))
                })
            };
            let mechanism = match row.mechanism.as_str() {
                "initial_state" => ModelMechanism::InitialState,
                "utility" => ModelMechanism::Utility,
                "transition" => ModelMechanism::Transition { action: action("transition")? },
                "observation" => ModelMechanism::Observation { action: action("observation")? },
                other => {
                    return Err(crate::value_err(format!("unknown model mechanism `{other}`")));
                }
            };
            Ok((mechanism, status_of(&row.status)?))
        })
        .collect()
}

fn utility_of(text: Option<&str>) -> PyResult<Option<TargetUtility>> {
    text.map(|text| {
        let wire: TargetUtilityIn = declaration(text, "target utility")?;
        Ok(TargetUtility {
            provenance: UtilityProvenance { source: wire.source, digest: wire.digest },
            terminal: wire.terminal,
            actions: wire
                .actions
                .into_iter()
                .map(|a| ActionUtility { cost: a.cost, immediate: a.immediate, delayed: a.delayed })
                .collect(),
        })
    })
    .transpose()
}

fn trial_mechanisms_of(
    rows: Vec<TrialMechanismIn>,
) -> PyResult<Vec<(TrialMechanism, MechanismStatus)>> {
    rows.into_iter()
        .map(|row| {
            let mechanism = match row.mechanism.as_str() {
                "stratum_law" => TrialMechanism::StratumLaw,
                "outcome_given_stratum_and_arm" => TrialMechanism::OutcomeGivenStratumAndArm,
                other => {
                    return Err(crate::value_err(format!("unknown trial mechanism `{other}`")));
                }
            };
            Ok((mechanism, status_of(&row.status)?))
        })
        .collect()
}

fn policy_of(
    text: &str,
) -> PyResult<Result<seq::ContingentPolicy, seq::SequentialEvaluationError>> {
    let wire: PolicyIn = declaration(text, "sequential policy")?;
    let mut policy = seq::ContingentPolicy::new(wire.version);
    for decision in wire.decisions {
        let key = seq::HistoryKey::from_steps(
            decision
                .history
                .iter()
                .map(|&(action, observation)| seq::HistoryStep { action, observation }),
        );
        policy = match policy.decide_stochastic(key, decision.actions) {
            Ok(next) => next,
            Err(error) => return Ok(Err(error)),
        };
    }
    Ok(Ok(policy))
}

fn limits_of(max_nodes: Option<u64>) -> seq::EvaluationLimits {
    let mut limits = seq::EvaluationLimits::default();
    if let Some(nodes) = max_nodes {
        limits.max_nodes = nodes;
    }
    limits
}

fn source_id(value: &str) -> PyResult<SourceId> {
    SourceId::from_hex(value)
        .ok_or_else(|| crate::value_err(format!("`{value}` is not a source identity")))
}

// ---------------------------------------------------------------------------
// Results as JSON
// ---------------------------------------------------------------------------

fn premises_json(
    premises: &antecedent::analysis::transported_constructors::TransportPremises,
) -> Json {
    json!({
        "selection": premises.selection,
        "invariance": premises.invariance,
        "overlap": premises.overlap,
    })
}

// ---------------------------------------------------------------------------
// Functions
// ---------------------------------------------------------------------------

/// The history alphabet (actions and observation symbols) of a model, as JSON, for declaring
/// the target period's alphabet of a period shift.
#[pyfunction]
fn transported_history_alphabet(model_json: &str) -> PyResult<(Option<String>, Option<String>)> {
    let model = crate::sequential_evaluation_api::model_of(model_json)?;
    Ok(match HistoryAlphabet::of_model(&model) {
        Ok(alphabet) => {
            let actions: Vec<Json> = alphabet
                .actions
                .iter()
                .map(|a| json!({"label": a.label, "symbols": a.symbols}))
                .collect();
            (Some(Json::Array(actions).to_string()), None)
        }
        Err(error) => (None, Some(external_refusal_json(&error.to_refusal()))),
    })
}

/// A policy's value in another population, as a transported response.
#[pyfunction]
#[pyo3(signature = (model_json, policy_json, target_population, target_initial, diagram_json, coordinate_json, provider_id, max_nodes=None, target_utility_json=None, *, cancel=None))]
#[allow(clippy::too_many_arguments)] // Mirrors the Python signature.
fn transported_policy_construct(
    py: Python<'_>,
    model_json: &str,
    policy_json: &str,
    target_population: &str,
    target_initial: Option<Vec<f64>>,
    diagram_json: Option<&str>,
    coordinate_json: &str,
    provider_id: &str,
    max_nodes: Option<u64>,
    target_utility_json: Option<&str>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Built> {
    let model = crate::sequential_evaluation_api::model_of(model_json)?;
    let policy = match policy_of(policy_json)? {
        Ok(policy) => policy,
        Err(error) => return Ok(refused(&error.to_refusal())),
    };
    let diagram = match diagram_json {
        None => None,
        Some(text) => {
            let wire: PopulationDiagramIn = declaration(text, "selection diagram")?;
            Some(SelectionDiagram {
                source_population: wire.source_population,
                target_population: wire.target_population,
                mechanisms: mechanisms_of(wire.mechanisms)?,
            })
        }
    };
    let target_utility = utility_of(target_utility_json)?;
    let coordinate = quantity_of(coordinate_json, "transported coordinate")?;
    let target_population = target_population.to_owned();
    let provider = provider_id.to_owned();
    let outcome = crate::detach_catch(py, move || {
        let ctx = execution_context(0, None, cancel);
        let request = TransportedPolicyRequest {
            source: &model,
            policy: &policy,
            target_population: &target_population,
            target_initial: target_initial.as_deref(),
            diagram: diagram.as_ref(),
            target_utility: target_utility.as_ref(),
            limits: limits_of(max_nodes),
        };
        Ok(transported_policy_response(&request, coordinate, &provider, &ctx))
    })?;
    Ok(match outcome {
        Ok(built) => {
            let e = &built.evaluation;
            let evaluation = json!({
                "route": "policy",
                "target_value": e.target_value,
                "source_value": e.source_value,
                "source_population": e.source_population,
                "target_population": e.target_population,
                "shifted_mechanisms": e.shifted_mechanisms.iter().map(|m| m.name()).collect::<Vec<_>>(),
                "diagram_identity": e.diagram_identity,
                "target_initial": e.target_initial,
                "target_utility": e.target_utility.as_ref().map(|u| json!({
                    "source": u.provenance.source,
                    "digest": u.provenance.digest,
                    "identity": u.identity_hex(),
                })),
                "nodes_visited": e.nodes_visited,
                "premises": premises_json(&e.premises),
                "identity": e.identity,
                "inference": e.inference,
            });
            (
                Some(TransportedResponseHandle { inner: built.response }),
                Some(evaluation.to_string()),
                None,
            )
        }
        Err(refusal) => refused(&refusal),
    })
}

/// A policy's value in another period, as a transported response.
#[pyfunction]
#[pyo3(signature = (model_json, policy_json, target_json, target_initial, invariance_json, coordinate_json, provider_id, max_nodes=None, *, cancel=None))]
#[allow(clippy::too_many_arguments)] // Mirrors the Python signature.
fn temporal_policy_construct(
    py: Python<'_>,
    model_json: &str,
    policy_json: &str,
    target_json: &str,
    target_initial: Option<Vec<f64>>,
    invariance_json: Option<&str>,
    coordinate_json: &str,
    provider_id: &str,
    max_nodes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Built> {
    let model = crate::sequential_evaluation_api::model_of(model_json)?;
    let policy = match policy_of(policy_json)? {
        Ok(policy) => policy,
        Err(error) => return Ok(refused(&error.to_refusal())),
    };
    let wire: TargetIn = declaration(target_json, "target period")?;
    let target = TemporalTarget {
        period: wire.period,
        horizon: wire.horizon,
        alphabet: HistoryAlphabet {
            actions: wire
                .alphabet
                .into_iter()
                .map(|a| ActionAlphabet { label: a.label, symbols: a.symbols })
                .collect(),
        },
    };
    let invariance = match invariance_json {
        None => None,
        Some(text) => {
            let wire: InvarianceIn = declaration(text, "time invariance")?;
            Some(TimeInvariance {
                source_period: wire.source_period,
                target_period: wire.target_period,
                horizon: wire.horizon,
                mechanisms: mechanisms_of(wire.mechanisms)?,
            })
        }
    };
    let coordinate = quantity_of(coordinate_json, "transported coordinate")?;
    let provider = provider_id.to_owned();
    let outcome = crate::detach_catch(py, move || {
        let ctx = execution_context(0, None, cancel);
        let request = TemporalPolicyRequest {
            source: &model,
            policy: &policy,
            target: &target,
            target_initial: target_initial.as_deref(),
            invariance: invariance.as_ref(),
            limits: limits_of(max_nodes),
        };
        Ok(temporal_policy_response(&request, coordinate, &provider, &ctx))
    })?;
    Ok(match outcome {
        Ok(built) => {
            let e = &built.evaluation;
            let evaluation = json!({
                "route": "temporal_policy",
                "target_value": e.target_value,
                "source_value": e.source_value,
                "source_period": e.source_period,
                "target_period": e.target_period,
                "horizon": e.horizon,
                "invariance_identity": e.invariance_identity,
                "target_initial": e.target_initial,
                "nodes_visited": e.nodes_visited,
                "premises": premises_json(&e.premises),
                "identity": e.identity,
                "inference": e.inference,
            });
            (
                Some(TransportedResponseHandle { inner: built.response }),
                Some(evaluation.to_string()),
                None,
            )
        }
        Err(refusal) => refused(&refusal),
    })
}

fn adaptive_refusal(refusals: &[AdaptiveInferenceRefusal]) -> Built {
    let row = |r: &AdaptiveInferenceRefusal| {
        json!({
            "code": r.code,
            "stage": "adaptive_transport",
            "detail": r.detail,
            "message": r.message,
            "offending": Json::Null,
            "remedy": Json::Null,
        })
    };
    let mut first = refusals.first().map_or_else(|| json!({}), row);
    first["refusals"] = Json::Array(refusals.iter().map(row).collect());
    (None, None, Some(first.to_string()))
}

/// An adaptive trial's transported arm mean or contrast, as a transported response.
#[pyfunction]
#[pyo3(signature = (design_json, trial_population, unit_strata, stratum_labels, target_stratum_law, diagram_json, treatment_arm, control_arm, value, coordinate_json, provider_id))]
#[allow(clippy::too_many_arguments)] // Mirrors the Python signature.
fn adaptive_trial_construct(
    design_json: Option<&str>,
    trial_population: &str,
    unit_strata: Vec<usize>,
    stratum_labels: Vec<String>,
    target_stratum_law: Option<Vec<f64>>,
    diagram_json: Option<&str>,
    treatment_arm: usize,
    control_arm: usize,
    value: &str,
    coordinate_json: &str,
    provider_id: &str,
) -> PyResult<Built> {
    let value = match value {
        "treatment_mean" => AdaptiveTransportedValue::TreatmentMean,
        "control_mean" => AdaptiveTransportedValue::ControlMean,
        "contrast" => AdaptiveTransportedValue::Contrast,
        other => {
            return Err(crate::value_err(format!(
                "value must be `treatment_mean`, `control_mean` or `contrast`, not `{other}`"
            )));
        }
    };
    let session = match design_json {
        None => None,
        Some(text) => {
            check_declared_size(text, MAX_JSON_BYTES, "adaptive design")?;
            let json: Json = serde_json::from_str(text).map_err(|error| {
                crate::value_err(format!("invalid adaptive design declaration: {error}"))
            })?;
            let wire: DeclarationWire = serde_json::from_value(json).map_err(|error| {
                crate::value_err(format!("invalid adaptive design declaration: {error}"))
            })?;
            let declared = declaration_from_wire(&wire)
                .map_err(|error| crate::value_err(error.to_string()))?;
            match AdaptiveInferenceSession::prepare(declared) {
                Ok(session) => Some(session),
                Err(refusals) => return Ok(adaptive_refusal(&refusals)),
            }
        }
    };
    let diagram = match diagram_json {
        None => None,
        Some(text) => {
            let wire: TrialDiagramIn = declaration(text, "trial selection diagram")?;
            Some(TrialSelectionDiagram {
                source_population: wire.source_population,
                target_population: wire.target_population,
                mechanisms: trial_mechanisms_of(wire.mechanisms)?,
            })
        }
    };
    let coordinate = quantity_of(coordinate_json, "transported coordinate")?;
    let request = AdaptiveTransportRequest {
        contract: session.as_ref().map(AdaptiveInferenceSession::contract),
        trial_population,
        unit_strata: &unit_strata,
        stratum_labels: &stratum_labels,
        target_stratum_law: target_stratum_law.as_deref(),
        diagram: diagram.as_ref(),
        treatment_arm,
        control_arm,
    };
    Ok(match adaptive_trial_response(&request, value, coordinate, provider_id) {
        Ok(built) => {
            let e = &built.evaluation;
            let evaluation = json!({
                "route": "adaptive_trial",
                "value": value.as_str(),
                "treatment_mean": e.treatment_mean,
                "control_mean": e.control_mean,
                "contrast": e.contrast,
                "cells": e.cells.iter().map(|c| json!({
                    "stratum": c.stratum,
                    "arm": c.arm,
                    "units": c.units,
                    "weight_sum": c.weight_sum,
                    "mean": c.mean,
                })).collect::<Vec<_>>(),
                "analysed_units": e.analysed_units,
                "contract_identity": e.contract_identity,
                "diagram_identity": e.diagram_identity,
                "identity": e.identity,
                "inference": e.inference,
            });
            (
                Some(TransportedResponseHandle { inner: built.response }),
                Some(evaluation.to_string()),
                None,
            )
        }
        Err(refusal) => refused(&refusal),
    })
}

fn method_of(wire: MethodIn) -> PyResult<TransportedSynthesisMethod> {
    match wire.kind.as_str() {
        "pool" => {
            Ok(TransportedSynthesisMethod::Pool { sampling_covariance: wire.sampling_covariance })
        }
        "borrow" => {
            let mean = wire
                .mean_prior
                .ok_or_else(|| crate::value_err("a borrowing method declares `mean_prior`"))?;
            let tau = wire
                .tau_prior
                .ok_or_else(|| crate::value_err("a borrowing method declares `tau_prior`"))?;
            let mean_prior = match mean.kind.as_str() {
                "flat" => MeanPrior::Flat,
                "normal" => MeanPrior::Normal {
                    mean: mean
                        .mean
                        .ok_or_else(|| crate::value_err("a normal mean prior declares `mean`"))?,
                    variance: mean.variance.ok_or_else(|| {
                        crate::value_err("a normal mean prior declares `variance`")
                    })?,
                },
                other => return Err(crate::value_err(format!("unknown mean prior `{other}`"))),
            };
            Ok(TransportedSynthesisMethod::Borrow {
                mean_prior,
                tau_prior: TauPrior { support: tau.support, probabilities: tau.probabilities },
            })
        }
        "transported_prior" => {
            let prior = wire
                .prior
                .ok_or_else(|| crate::value_err("a transported prior declares `prior`"))?;
            Ok(TransportedSynthesisMethod::TransportedPrior { prior: source_id(&prior)? })
        }
        other => Err(crate::value_err(format!("unknown synthesis method `{other}`"))),
    }
}

/// A transported synthesis over an evidence ledger, as a transported response.
///
/// The synthesis is registered in a copy of the session's ledger; the session is unchanged.
#[pyfunction]
#[pyo3(signature = (ledger, label, contributions_json, method_json, native_population, coordinate_json, provider_id))]
fn transported_synthesis_construct(
    ledger: PyRef<'_, EvidenceLedgerHandle>,
    label: &str,
    contributions_json: &str,
    method_json: &str,
    native_population: Option<&str>,
    coordinate_json: &str,
    provider_id: &str,
) -> PyResult<Built> {
    let rows: Vec<ContributionIn> = declaration(contributions_json, "synthesis contributions")?;
    let contributions = rows
        .into_iter()
        .map(|row| {
            Ok(TransportedContribution {
                source: source_id(&row.source)?,
                estimate: row.estimate,
                sampling_variance: row.sampling_variance,
                transport_variance: row.transport_variance,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    let method = method_of(declaration(method_json, "synthesis method")?)?;
    let coordinate = quantity_of(coordinate_json, "transported coordinate")?;
    let mut staged = ledger.ledger().clone();
    Ok(
        match transported_synthesis_response(
            &mut staged,
            label,
            &contributions,
            &method,
            native_population,
            coordinate,
            provider_id,
        ) {
            Ok(built) => {
                let e = &built.evaluation;
                let hex = |ids: &[SourceId]| ids.iter().map(SourceId::to_hex).collect::<Vec<_>>();
                let evaluation = json!({
                    "route": "synthesis",
                    "result": e.result.to_hex(),
                    "method": e.method,
                    "estimate": e.estimate,
                    "variance": e.variance,
                    "transport_variance_added": e.transport_variance_added,
                    "native_sources": hex(&e.native_sources),
                    "transported_sources": hex(&e.transported_sources),
                    "transport_identities": e.transport_identities.iter().map(SemanticDigest::to_hex).collect::<Vec<_>>(),
                    "target_population": e.target_population,
                    "source_populations": e.source_populations,
                    "native_population": e.native_population,
                    "distinct_observations": e.mass.distinct_observations,
                    "repeated_slots": e.mass.repeated_slots,
                    "claim_class": e.claim_class.as_str(),
                });
                (
                    Some(TransportedResponseHandle { inner: built.response }),
                    Some(evaluation.to_string()),
                    None,
                )
            }
            Err(refusal) => refused(&refusal),
        },
    )
}

/// A surrogate bridge's terminal-outcome value in another population, as a transported
/// response. `claim_json` is the source claim in the scientific-bridge declaration form.
#[pyfunction]
fn surrogate_bridge_construct(
    bridge_json: &str,
    claim_json: &str,
    provider_id: &str,
) -> PyResult<Built> {
    let wire: BridgeIn = declaration(bridge_json, "surrogate bridge")?;
    let quantity = |wire: ScientificQuantityWire| {
        antecedent_contract::ScientificQuantity::try_from(wire).map_err(crate::value_err)
    };
    let bridge = TransportedSurrogateBridge {
        surrogate: quantity(wire.surrogate)?,
        terminal: quantity(wire.terminal)?,
        validation: TrialLevelValidation {
            trials: wire
                .validation
                .trials
                .into_iter()
                .map(|(surrogate_effect, terminal_effect)| TrialEffectPair {
                    surrogate_effect,
                    terminal_effect,
                })
                .collect(),
            min_trials: wire.validation.min_trials,
            min_r_squared: wire.validation.min_r_squared,
        },
        validated_in: BridgeScope {
            population_id: wire.validated_in.population_id,
            regime_id: wire.validated_in.regime_id,
        },
        validation_evidence: wire.validation_evidence,
        diagram: BridgeSelectionDiagram {
            source_population: wire.diagram.source_population,
            target_population: wire.diagram.target_population,
            selection_targets: wire.diagram.selection_targets,
        },
        terminal_variable: wire.terminal_variable,
        relation_modifiers: wire.relation_modifiers,
    };
    check_declared_size(claim_json, MAX_JSON_BYTES, "source claim")?;
    let claim_value: Json = serde_json::from_str(claim_json)
        .map_err(|error| crate::value_err(format!("invalid source claim declaration: {error}")))?;
    let claim_member = claim_value
        .get("claim")
        .ok_or_else(|| crate::value_err("a source claim declares `claim`"))?;
    let claim = crate::scientific_bridge_api::parse_claim(claim_member)?;
    Ok(match surrogate_bridge_response(&bridge, &claim, provider_id) {
        Ok(built) => {
            let e = &built.evaluation;
            let evaluation = json!({
                "route": "surrogate_bridge",
                "value": e.value,
                "source_value": e.source_value,
                "claim_id": e.claim.claim_id.to_hex(),
                "source_claim_id": e.source_claim_id.to_hex(),
                "contract_identity": e.contract_identity.to_hex(),
                "transported": e.transported,
            });
            (
                Some(TransportedResponseHandle { inner: built.response }),
                Some(evaluation.to_string()),
                None,
            )
        }
        Err(refusal) => refused(&refusal),
    })
}

// ---------------------------------------------------------------------------
// Verification: re-derive a stored evaluation from its request and compare
// ---------------------------------------------------------------------------

/// A stored number of an evaluation, refused by name when it is absent.
fn stored_f64(stored: &Json, key: &str) -> PyResult<f64> {
    stored.get(key).and_then(Json::as_f64).ok_or_else(|| {
        crate::value_err(format!("the stored evaluation has no numeric `{key}` to verify"))
    })
}

fn stored_text(stored: &Json, key: &str) -> PyResult<String> {
    stored.get(key).and_then(Json::as_str).map(str::to_owned).ok_or_else(|| {
        crate::value_err(format!("the stored evaluation has no `{key}` identity to verify"))
    })
}

fn stored_of(text: &str) -> PyResult<Json> {
    declaration(text, "stored evaluation")
}

/// Re-derive a transported policy evaluation from its model, policy, diagram and target law and
/// compare the stored identity and values bit for bit. Returns the refusal, or `None` when the
/// stored evaluation replays.
#[pyfunction]
#[pyo3(signature = (model_json, policy_json, target_population, target_initial, diagram_json, stored_json, max_nodes=None, target_utility_json=None, *, cancel=None))]
#[allow(clippy::too_many_arguments)] // Mirrors the Python signature.
fn transported_policy_verify(
    py: Python<'_>,
    model_json: &str,
    policy_json: &str,
    target_population: &str,
    target_initial: Option<Vec<f64>>,
    diagram_json: Option<&str>,
    stored_json: &str,
    max_nodes: Option<u64>,
    target_utility_json: Option<&str>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Option<String>> {
    let model = crate::sequential_evaluation_api::model_of(model_json)?;
    let policy = match policy_of(policy_json)? {
        Ok(policy) => policy,
        Err(error) => return Ok(Some(external_refusal_json(&error.to_refusal()))),
    };
    let diagram = match diagram_json {
        None => None,
        Some(text) => {
            let wire: PopulationDiagramIn = declaration(text, "selection diagram")?;
            Some(SelectionDiagram {
                source_population: wire.source_population,
                target_population: wire.target_population,
                mechanisms: mechanisms_of(wire.mechanisms)?,
            })
        }
    };
    let target_utility = utility_of(target_utility_json)?;
    let stored = stored_of(stored_json)?;
    let (identity, target_value, source_value) = (
        stored_text(&stored, "identity")?,
        stored_f64(&stored, "target_value")?,
        stored_f64(&stored, "source_value")?,
    );
    let target_population = target_population.to_owned();
    let outcome = crate::detach_catch(py, move || {
        let ctx = execution_context(0, None, cancel);
        let request = TransportedPolicyRequest {
            source: &model,
            policy: &policy,
            target_population: &target_population,
            target_initial: target_initial.as_deref(),
            diagram: diagram.as_ref(),
            target_utility: target_utility.as_ref(),
            limits: limits_of(max_nodes),
        };
        // The stored claim is the re-derived evaluation with the stored identity and values
        // put in place, so the facade's own verifier decides.
        let mut claimed = match evaluate_transported_policy(&request, &ctx) {
            Ok(fresh) => fresh,
            Err(error) => return Ok(Err(error)),
        };
        claimed.identity = identity;
        claimed.target_value = target_value;
        claimed.source_value = source_value;
        Ok(verify_transported_policy(&claimed, &request, &ctx))
    })?;
    Ok(outcome.err().map(|error| external_refusal_json(&error.to_refusal())))
}

/// Re-derive a temporal policy evaluation from its request and compare the stored identity and
/// values bit for bit.
#[pyfunction]
#[pyo3(signature = (model_json, policy_json, target_json, target_initial, invariance_json, stored_json, max_nodes=None, *, cancel=None))]
#[allow(clippy::too_many_arguments)] // Mirrors the Python signature.
fn temporal_policy_verify(
    py: Python<'_>,
    model_json: &str,
    policy_json: &str,
    target_json: &str,
    target_initial: Option<Vec<f64>>,
    invariance_json: Option<&str>,
    stored_json: &str,
    max_nodes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Option<String>> {
    let model = crate::sequential_evaluation_api::model_of(model_json)?;
    let policy = match policy_of(policy_json)? {
        Ok(policy) => policy,
        Err(error) => return Ok(Some(external_refusal_json(&error.to_refusal()))),
    };
    let wire: TargetIn = declaration(target_json, "target period")?;
    let target = TemporalTarget {
        period: wire.period,
        horizon: wire.horizon,
        alphabet: HistoryAlphabet {
            actions: wire
                .alphabet
                .into_iter()
                .map(|a| ActionAlphabet { label: a.label, symbols: a.symbols })
                .collect(),
        },
    };
    let invariance = match invariance_json {
        None => None,
        Some(text) => {
            let wire: InvarianceIn = declaration(text, "time invariance")?;
            Some(TimeInvariance {
                source_period: wire.source_period,
                target_period: wire.target_period,
                horizon: wire.horizon,
                mechanisms: mechanisms_of(wire.mechanisms)?,
            })
        }
    };
    let stored = stored_of(stored_json)?;
    let (identity, target_value, source_value) = (
        stored_text(&stored, "identity")?,
        stored_f64(&stored, "target_value")?,
        stored_f64(&stored, "source_value")?,
    );
    let outcome = crate::detach_catch(py, move || {
        let ctx = execution_context(0, None, cancel);
        let request = TemporalPolicyRequest {
            source: &model,
            policy: &policy,
            target: &target,
            target_initial: target_initial.as_deref(),
            invariance: invariance.as_ref(),
            limits: limits_of(max_nodes),
        };
        let mut claimed = match evaluate_temporal_policy(&request, &ctx) {
            Ok(fresh) => fresh,
            Err(error) => return Ok(Err(error)),
        };
        claimed.identity = identity;
        claimed.target_value = target_value;
        claimed.source_value = source_value;
        Ok(verify_temporal_policy(&claimed, &request, &ctx))
    })?;
    Ok(outcome.err().map(|error| external_refusal_json(&error.to_refusal())))
}

/// Re-derive an adaptive-trial transported estimate from its contract, strata, diagram and target
/// law and compare the stored identity and the three values bit for bit.
#[pyfunction]
#[pyo3(signature = (design_json, trial_population, unit_strata, stratum_labels, target_stratum_law, diagram_json, treatment_arm, control_arm, stored_json))]
#[allow(clippy::too_many_arguments)] // Mirrors the Python signature.
fn adaptive_trial_verify(
    design_json: Option<&str>,
    trial_population: &str,
    unit_strata: Vec<usize>,
    stratum_labels: Vec<String>,
    target_stratum_law: Option<Vec<f64>>,
    diagram_json: Option<&str>,
    treatment_arm: usize,
    control_arm: usize,
    stored_json: &str,
) -> PyResult<Option<String>> {
    let session = match design_json {
        None => None,
        Some(text) => {
            check_declared_size(text, MAX_JSON_BYTES, "adaptive design")?;
            let json: Json = serde_json::from_str(text).map_err(|error| {
                crate::value_err(format!("invalid adaptive design declaration: {error}"))
            })?;
            let wire: DeclarationWire = serde_json::from_value(json).map_err(|error| {
                crate::value_err(format!("invalid adaptive design declaration: {error}"))
            })?;
            let declared = declaration_from_wire(&wire)
                .map_err(|error| crate::value_err(error.to_string()))?;
            match AdaptiveInferenceSession::prepare(declared) {
                Ok(session) => Some(session),
                Err(refusals) => return Ok(adaptive_refusal(&refusals).2),
            }
        }
    };
    let diagram = match diagram_json {
        None => None,
        Some(text) => {
            let wire: TrialDiagramIn = declaration(text, "trial selection diagram")?;
            Some(TrialSelectionDiagram {
                source_population: wire.source_population,
                target_population: wire.target_population,
                mechanisms: trial_mechanisms_of(wire.mechanisms)?,
            })
        }
    };
    let stored = stored_of(stored_json)?;
    let request = AdaptiveTransportRequest {
        contract: session.as_ref().map(AdaptiveInferenceSession::contract),
        trial_population,
        unit_strata: &unit_strata,
        stratum_labels: &stratum_labels,
        target_stratum_law: target_stratum_law.as_deref(),
        diagram: diagram.as_ref(),
        treatment_arm,
        control_arm,
    };
    let mut claimed = match transport_adaptive_trial(&request) {
        Ok(fresh) => fresh,
        Err(error) => return Ok(Some(external_refusal_json(&error.to_refusal()))),
    };
    claimed.identity = stored_text(&stored, "identity")?;
    claimed.treatment_mean = stored_f64(&stored, "treatment_mean")?;
    claimed.control_mean = stored_f64(&stored, "control_mean")?;
    claimed.contrast = stored_f64(&stored, "contrast")?;
    Ok(verify_adaptive_transport(&claimed, &request)
        .err()
        .map(|error| external_refusal_json(&error.to_refusal())))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(transported_policy_verify, module)?)?;
    module.add_function(wrap_pyfunction!(temporal_policy_verify, module)?)?;
    module.add_function(wrap_pyfunction!(adaptive_trial_verify, module)?)?;
    module.add_function(wrap_pyfunction!(transported_history_alphabet, module)?)?;
    module.add_function(wrap_pyfunction!(transported_policy_construct, module)?)?;
    module.add_function(wrap_pyfunction!(temporal_policy_construct, module)?)?;
    module.add_function(wrap_pyfunction!(adaptive_trial_construct, module)?)?;
    module.add_function(wrap_pyfunction!(transported_synthesis_construct, module)?)?;
    module.add_function(wrap_pyfunction!(surrogate_bridge_construct, module)?)?;
    Ok(())
}
