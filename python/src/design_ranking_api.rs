//! Bounded Python bridge for candidate signals, EVSI with a cost mapping and the
//! durable design-ranking artifact.
//!
//! Python builds the declaration (decision, candidates, signal providers with attested values and
//! trust, optional cost mapping); the exact signal request fingerprint, provider binding,
//! preposterior integration, source-overlap and cost-unit checks, artifact identity, independent
//! recomputation and every refusal are Rust's. A refusal comes back as structured JSON for the
//! Python layer to raise as its own exception type; corruption, truncation and unknown versions
//! raise `CausalSerializationError`. Monte Carlo error coverage and rank guarantees are
//! `unmeasured`; an exact-integration EVSI is a point-only value.

use std::collections::BTreeMap;

use crate::bridge::{check_declared_size, external_refusal_json, serialization};
use antecedent::analysis::design_ranking::{
    ConsumeExpectation, CostMapDeclWire, DesignRankingError, DesignRankingRequestWire,
    StructuralCandidateDeclWire, compile_for_study, consume, evaluate, evaluate_for_study,
    rank_structural,
};
use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_data::TableView;
use pyo3::prelude::*;
use serde::Deserialize;

use crate::{detach_catch, value_err, with_reason_code};

/// Largest declaration or expectation JSON accepted.
const MAX_JSON_BYTES: usize = crate::bridge::MAX_DECLARED_ARTIFACT_BYTES;

type EvaluatePayload = (Option<String>, Option<Vec<u8>>, Option<String>);
type ConsumePayload = (Option<String>, Option<String>);
/// `(artifact body, artifact bytes, lowered program identity, refusal)`.
type StudyEvaluatePayload = (Option<String>, Option<Vec<u8>>, Option<String>, Option<String>);

/// Identities the consumer retained independently of the artifact bytes.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExpectationWire {
    #[serde(default)]
    artifact_identity: Option<String>,
    #[serde(default)]
    decision_contract_identity: Option<String>,
    #[serde(default)]
    signal_identities: Option<BTreeMap<String, String>>,
    #[serde(default)]
    source_digests: Option<Vec<String>>,
    #[serde(default)]
    cost_mapping: Option<CostMapDeclWire>,
    /// Assert that no cost mapping was used.
    #[serde(default)]
    assert_no_cost_mapping: bool,
}

impl From<ExpectationWire> for ConsumeExpectation {
    fn from(wire: ExpectationWire) -> Self {
        let cost_mapping = if wire.assert_no_cost_mapping {
            Some(None)
        } else {
            wire.cost_mapping.map(|m| {
                Some(antecedent_design::evsi::CostToUtilityMap {
                    cost_unit: m.cost_unit,
                    utility_unit: m.utility_unit,
                    utility_per_cost: m.utility_per_cost,
                })
            })
        };
        Self {
            artifact_identity: wire.artifact_identity,
            decision_contract_identity: wire.decision_contract_identity,
            signal_identities: wire.signal_identities,
            source_digests: wire.source_digests,
            cost_mapping,
        }
    }
}

fn check_size(text: &str) -> PyResult<()> {
    check_declared_size(text, MAX_JSON_BYTES, "design ranking")
}

fn invalid_declaration(error: &serde_json::Error) -> PyErr {
    with_reason_code(
        value_err(format!("design_ranking.invalid_request: {error}")),
        antecedent_core::reason_code!("invalid_argument"),
    )
}

/// A refusal becomes structured JSON; anything else is a serialization failure.
fn payload_error(error: &DesignRankingError) -> Result<String, PyErr> {
    error.refusal().map(|r| external_refusal_json(&r)).ok_or_else(|| serialization(error))
}

/// Evaluate every declared candidate's EVSI and seal the ranking:
/// `(artifact body json, artifact bytes, refusal json)`.
#[pyfunction]
#[pyo3(signature = (request_json, artifact_id, checked_prior=None))]
fn evaluate_design_ranking(
    py: Python<'_>,
    request_json: &str,
    artifact_id: &str,
    checked_prior: Option<PyRef<'_, crate::prior_signal_api::CheckedPriorSignal>>,
) -> PyResult<EvaluatePayload> {
    check_size(request_json)?;
    let mut request: DesignRankingRequestWire =
        serde_json::from_str(request_json).map_err(|e| invalid_declaration(&e))?;
    if let Some(prior) = checked_prior {
        if let Err(refusal) = prior.bind(&mut request) {
            return Ok((None, None, Some(refusal)));
        }
    }
    let artifact_id = artifact_id.to_owned();
    detach_catch(py, move || match evaluate(request) {
        Ok(done) => {
            let bytes = done.export(&artifact_id).map_err(serialization)?;
            let body = serde_json::to_string(done.artifact()).map_err(serialization)?;
            Ok((Some(body), Some(bytes), None))
        }
        Err(error) => Ok((None, None, Some(payload_error(&error)?))),
    })
}

fn consumed_json(done: &antecedent::analysis::design_ranking::ConsumedDesignRanking) -> String {
    let ranking = &done.ranking;
    let by_id: BTreeMap<&str, &antecedent::analysis::design_ranking::ConsumedCandidate> =
        done.candidates.iter().map(|c| (c.semantic_id.as_str(), c)).collect();
    let entries: Vec<serde_json::Value> = ranking
        .entries
        .iter()
        .map(|e| {
            let consumed = by_id.get(e.semantic_id.as_str());
            serde_json::json!({
                "semantic_id": e.semantic_id,
                "rank": e.rank,
                "evsi": e.evsi,
                "evpi": e.evpi,
                "net_value": e.net_value,
                "mc_stderr": e.mc_stderr,
                "replicates": e.replicates,
                "integration": e.integration.as_str(),
                "sample_size": e.sample_size,
                "study_cost_amount": e.study_cost_amount,
                "study_cost_unit": e.study_cost_unit,
                "rank_uncertain": e.rank_uncertain,
                "signal_identity": e.signal_identity,
                "request_fingerprint": e.request_fingerprint,
                "update_mode": e.update_mode.as_str(),
                "provider_trust": e.provider_trust.as_str(),
                "replay": consumed.map(|c| c.replay.as_str()),
                "natively_replayed": consumed.is_some_and(|c| c.natively_replayed()),
                "trust_limit": consumed.map(|c| c.trust_limit.clone()),
            })
        })
        .collect();
    serde_json::json!({
        "identity": done.identity,
        "ranking_identity": ranking.identity(),
        "calibration": done.calibration,
        "basis": ranking.basis.as_str(),
        "decision_contract_identity": ranking.decision_contract_identity,
        "utility_unit": ranking.utility_unit,
        "rng_seed": ranking.rng_seed,
        "source_digests": ranking.source_digests,
        "cost_mapping": ranking.cost_mapping.as_ref().map(|m| serde_json::json!({
            "cost_unit": m.cost_unit,
            "utility_unit": m.utility_unit,
            "utility_per_cost": m.utility_per_cost,
        })),
        "entries": entries,
        "lineage": done.provenance_chain.links().iter().map(|link| {
            serde_json::json!([link.id, link.stage.as_str(), link.parents])
        }).collect::<Vec<_>>(),
        "search": {
            "supplied": ranking.search.supplied,
            "evaluated": ranking.search.evaluated,
            "truncated": ranking.search.truncated,
            "unevaluated_ids": ranking.search.unevaluated_ids,
        },
    })
    .to_string()
}

/// Consume an artifact by independent recomputation: `(ranking json, refusal json)`.
///
/// `expectation_json` carries identities the caller retained independently of the bytes
/// (artifact digest, decision contract identity, per-candidate signal identities, source
/// digests, cost mapping); a changed one is refused against it.
#[pyfunction]
#[pyo3(signature = (artifact, expectation_json=None))]
fn consume_design_ranking(
    py: Python<'_>,
    artifact: Vec<u8>,
    expectation_json: Option<String>,
) -> PyResult<ConsumePayload> {
    let expectation: ConsumeExpectation = match expectation_json {
        Some(text) => {
            check_size(&text)?;
            serde_json::from_str::<ExpectationWire>(&text)
                .map_err(|e| invalid_declaration(&e))?
                .into()
        }
        None => ConsumeExpectation::default(),
    };
    detach_catch(py, move || match consume(&artifact, &expectation) {
        Ok(done) => Ok((Some(consumed_json(&done)), None)),
        Err(error) => {
            let error = DesignRankingError::Artifact(error);
            Ok((None, Some(payload_error(&error)?)))
        }
    })
}

/// The structural ordering when no probabilistic model is licensed:
/// `(ranking json, refusal json)`.
#[pyfunction]
fn rank_structural_designs(py: Python<'_>, candidates_json: &str) -> PyResult<ConsumePayload> {
    check_size(candidates_json)?;
    let candidates: Vec<StructuralCandidateDeclWire> =
        serde_json::from_str(candidates_json).map_err(|e| invalid_declaration(&e))?;
    detach_catch(py, move || match rank_structural(&candidates) {
        Ok(ranking) => {
            let entries: Vec<serde_json::Value> = ranking
                .structural
                .iter()
                .map(|e| {
                    serde_json::json!({
                        "semantic_id": e.semantic_id,
                        "rank": e.rank,
                        "verified_sufficient": e.verified_sufficient,
                        "cost_units": e.cost.units,
                        "sample_budget": e.cost.sample_budget,
                    })
                })
                .collect();
            Ok((
                Some(
                    serde_json::json!({
                        "basis": ranking.basis.as_str(),
                        "identity": ranking.identity(),
                        "entries": entries,
                    })
                    .to_string(),
                ),
                None,
            ))
        }
        Err(error) => Ok((None, Some(payload_error(&error)?))),
    })
}

/// The study a value-of-information request is compiled against: a binary average effect of
/// `treatment` on `outcome` over the named columns and DAG edges, with a Bayesian posterior
/// (`posterior`) or the default sampling realization.
#[allow(clippy::too_many_arguments, reason = "one study: columns, graph, query and inference")]
fn evsi_study(
    names: &[String],
    columns: &[Vec<f64>],
    edges: &[(String, String)],
    treatment: &str,
    outcome: &str,
    posterior: bool,
    n_draws: usize,
    prior_scale: f64,
) -> PyResult<Study> {
    if names.len() != columns.len() {
        return Err(value_err("names and columns must have the same length"));
    }
    let pairs: Vec<(&str, &[f64])> = names
        .iter()
        .zip(columns)
        .map(|(name, column)| (name.as_str(), column.as_slice()))
        .collect();
    let data = antecedent_data::TabularData::from_f64_columns(pairs).map_err(crate::py_err)?;
    let t_id = data.schema().id_of(treatment).map_err(crate::py_err)?;
    let y_id = data.schema().id_of(outcome).map_err(crate::py_err)?;
    let graph = crate::dag_from_named_edges(data.schema(), edges)?;
    let mut builder = Study::tabular(data)
        .graph(graph)
        .query(antecedent_core::AverageEffectQuery::binary_ate(t_id, y_id));
    if posterior {
        builder = builder.inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(n_draws).prior_scale(prior_scale),
        ));
    }
    builder.refute(RefuteSuite::None).bootstrap_replicates(0).build().map_err(crate::py_err)
}

/// A refusal as structured JSON; a compile failure that is not a refusal (a Monte Carlo store the
/// admission pool cannot hold) keeps its own error class.
fn study_refusal(error: DesignRankingError) -> PyResult<String> {
    if let Some(refusal) = error.refusal() {
        return Ok(external_refusal_json(&refusal));
    }
    match error {
        DesignRankingError::Compile(inner) => Err(crate::py_err(inner)),
        other => Err(serialization(&other)),
    }
}

fn parse_request(request_json: &str) -> PyResult<DesignRankingRequestWire> {
    check_size(request_json)?;
    serde_json::from_str(request_json).map_err(|e| invalid_declaration(&e))
}

/// Compile a value-of-information request against a study as the state belief of the
/// decision, without fitting or simulating anything: `(program json, refusal json)`.
///
/// The program json carries the lowered program identity, the demanded carrier, the decision
/// contract identity and utility unit, and the request the program holds (candidates in
/// canonical order, prior observations, seed, tolerance, ranking schedule and the quoted
/// Monte Carlo store).
#[pyfunction]
#[pyo3(signature = (request_json, names, columns, edges, treatment, outcome, *, posterior=true, n_draws=256, prior_scale=10.0, checked_prior=None, memory_bytes=None, cancel=None))]
#[allow(clippy::too_many_arguments, reason = "one request: declaration, study and bounds")]
fn compile_design_ranking_for_study(
    py: Python<'_>,
    request_json: &str,
    names: Vec<String>,
    columns: Vec<Vec<f64>>,
    edges: Vec<(String, String)>,
    treatment: &str,
    outcome: &str,
    posterior: bool,
    n_draws: usize,
    prior_scale: f64,
    checked_prior: Option<PyRef<'_, crate::prior_signal_api::CheckedPriorSignal>>,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<ConsumePayload> {
    let mut request = parse_request(request_json)?;
    if let Some(prior) = checked_prior {
        if let Err(refusal) = prior.bind(&mut request) {
            return Ok((None, Some(refusal)));
        }
    }
    let (treatment, outcome) = (treatment.to_owned(), outcome.to_owned());
    detach_catch(py, move || {
        let study = evsi_study(
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
        match compile_for_study(&study, request, &ctx) {
            Ok(plan) => {
                let lowered = plan.lowered().ok_or_else(|| {
                    value_err("the compiled plan carries no lowered program for the request")
                })?;
                let decision = lowered.evidence().decision().ok_or_else(|| {
                    value_err("the compiled plan carries no decision program for the request")
                })?;
                let evsi = decision.information_value().ok_or_else(|| {
                    value_err("the compiled plan carries no value-of-information program")
                })?;
                let schedule = evsi.rank();
                let store = evsi.monte_carlo();
                let body = serde_json::json!({
                    "program_identity": lowered.identity().to_hex(),
                    "carrier": decision.carrier().as_str(),
                    "contract_identity": decision.contract_identity(),
                    "utility_unit": decision.utility_unit(),
                    "candidates": evsi.candidates().iter().map(|id| &**id).collect::<Vec<_>>(),
                    "prior_observations": evsi
                        .prior_observation_ids()
                        .iter()
                        .map(|id| &**id)
                        .collect::<Vec<_>>(),
                    "rng_seed": evsi.rng_seed(),
                    "mc_error_tolerance": evsi.mc_error_tolerance(),
                    "schedule": {
                        "min_batches": schedule.min_batches(),
                        "max_batches": schedule.max_batches(),
                        "batch_size": schedule.batch_size(),
                        "draws_per_candidate": schedule.draws_per_candidate(),
                    },
                    "store": {
                        "candidates": store.candidates(),
                        "draws": store.draws(),
                        "tasks": store.tasks(),
                        "total_bytes": store.total_bytes(),
                    },
                });
                Ok((Some(body.to_string()), None))
            }
            Err(error) => Ok((None, Some(study_refusal(error)?))),
        }
    })
}

/// Evaluate a value-of-information request bound to the lowered program of a study compiled for
/// it: `(artifact body json, artifact bytes, lowered program identity, refusal json)`.
#[pyfunction]
#[pyo3(signature = (request_json, artifact_id, names, columns, edges, treatment, outcome, *, posterior=true, n_draws=256, prior_scale=10.0, checked_prior=None, memory_bytes=None, cancel=None))]
#[allow(clippy::too_many_arguments, reason = "one request: declaration, study and bounds")]
fn evaluate_design_ranking_for_study(
    py: Python<'_>,
    request_json: &str,
    artifact_id: &str,
    names: Vec<String>,
    columns: Vec<Vec<f64>>,
    edges: Vec<(String, String)>,
    treatment: &str,
    outcome: &str,
    posterior: bool,
    n_draws: usize,
    prior_scale: f64,
    checked_prior: Option<PyRef<'_, crate::prior_signal_api::CheckedPriorSignal>>,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<StudyEvaluatePayload> {
    let mut request = parse_request(request_json)?;
    if let Some(prior) = checked_prior {
        if let Err(refusal) = prior.bind(&mut request) {
            return Ok((None, None, None, Some(refusal)));
        }
    }
    let artifact_id = artifact_id.to_owned();
    let (treatment, outcome) = (treatment.to_owned(), outcome.to_owned());
    detach_catch(py, move || {
        let study = evsi_study(
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
        match evaluate_for_study(&study, request, &ctx) {
            Ok(done) => {
                let bytes = done.export(&artifact_id).map_err(serialization)?;
                let body = serde_json::to_string(done.artifact()).map_err(serialization)?;
                let program = done.lowered_program_identity().map(|digest| digest.to_hex());
                Ok((Some(body), Some(bytes), program, None))
            }
            Err(error) => Ok((None, None, None, Some(study_refusal(error)?))),
        }
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(evaluate_design_ranking, m)?)?;
    m.add_function(wrap_pyfunction!(consume_design_ranking, m)?)?;
    m.add_function(wrap_pyfunction!(compile_design_ranking_for_study, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_design_ranking_for_study, m)?)?;
    crate::rollout_api::register(m)?;
    m.add_function(wrap_pyfunction!(rank_structural_designs, m)?)
}
