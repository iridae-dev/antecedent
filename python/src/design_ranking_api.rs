//! Bounded Python bridge for F11/F12/F14: candidate signals, EVSI with a cost mapping and the
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

use antecedent::analysis::design_ranking::{
    ConsumeExpectation, CostMapDeclWire, DesignRankingError, DesignRankingRequestWire,
    StructuralCandidateDeclWire, consume, evaluate, rank_structural,
};
use antecedent_core::ExternalRefusal;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::Deserialize;

use crate::{CausalSerializationError, detach_catch, value_err, with_reason_code};

/// Largest declaration or expectation JSON accepted.
const MAX_JSON_BYTES: usize = 16 * 1024 * 1024;

type EvaluatePayload = (Option<String>, Option<Vec<u8>>, Option<String>);
type ConsumePayload = (Option<String>, Option<String>);

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

fn refusal_json(value: &ExternalRefusal) -> String {
    serde_json::json!({
        "code": value.code,
        "stage": value.stage,
        "detail": value.detail,
        "offending": value.offending,
        "expected": value.expected,
        "supplied": value.supplied,
        "remedy": value.remedy,
    })
    .to_string()
}

fn serialization(error: impl std::fmt::Display) -> PyErr {
    CausalSerializationError::new_err(error.to_string())
}

fn check_size(text: &str) -> PyResult<()> {
    if text.len() > MAX_JSON_BYTES {
        return Err(PyValueError::new_err("design ranking declaration is too large"));
    }
    Ok(())
}

fn invalid_declaration(error: &serde_json::Error) -> PyErr {
    with_reason_code(
        value_err(format!("design_ranking.invalid_request: {error}")),
        antecedent_core::reason_code!("invalid_argument"),
    )
}

/// A refusal becomes structured JSON; anything else is a serialization failure.
fn payload_error(error: &DesignRankingError) -> Result<String, PyErr> {
    error.refusal().map(|r| refusal_json(&r)).ok_or_else(|| serialization(error))
}

/// Evaluate every declared candidate's EVSI and seal the ranking:
/// `(artifact body json, artifact bytes, refusal json)`.
#[pyfunction]
fn evaluate_design_ranking(
    py: Python<'_>,
    request_json: &str,
    artifact_id: &str,
) -> PyResult<EvaluatePayload> {
    check_size(request_json)?;
    let request: DesignRankingRequestWire =
        serde_json::from_str(request_json).map_err(|e| invalid_declaration(&e))?;
    let artifact_id = artifact_id.to_owned();
    detach_catch(py, move || match evaluate(&request) {
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

/// The 2.2 structural ordering when no probabilistic model is licensed:
/// `(ranking json, refusal json)`.
#[pyfunction]
fn rank_structural_designs(candidates_json: &str) -> PyResult<ConsumePayload> {
    check_size(candidates_json)?;
    let candidates: Vec<StructuralCandidateDeclWire> =
        serde_json::from_str(candidates_json).map_err(|e| invalid_declaration(&e))?;
    match rank_structural(&candidates) {
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
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(evaluate_design_ranking, m)?)?;
    m.add_function(wrap_pyfunction!(consume_design_ranking, m)?)?;
    m.add_function(wrap_pyfunction!(rank_structural_designs, m)?)
}
