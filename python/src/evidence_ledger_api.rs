//! Python bridge for the evidence ledger session.
//!
//! Python declares sources (with row-level observation identities and
//! derivation lineage) and asks for combinations; the overlap check, the
//! numbers, the revalidation of stale results, the atomic refresh, the
//! artifact and every refusal are Rust's. A refusal comes back as structured
//! JSON for the Python layer to raise as a typed exception, and a refused
//! request leaves the session exactly as it was.
//!
//! Nothing here estimates anything about dependence: it is declared, or it
//! stays unknown, and no interval, posterior or coverage statement is made.

use antecedent::{
    BetweenStudy, BorrowingRequest, ConsumedEvidenceLedger, DependenceRangeRequest,
    EvidenceLedgerRefusal, EvidenceLedgerSession, LedgerRefresh, ObservationAccount,
    PoolingDeclaration, PoolingRequest, RevisionReport, SourceChange, SourceRevision, StudyBuilder,
    StudyEstimate, StudySummary, SynthesisOutcome, SynthesisValue,
    consume_evidence_ledger_artifact,
};
use antecedent_contract::evidence_ledger::{
    EffectiveObservations, EvidenceLedger, EvidenceSource, Lineage, MassAccount, ObservationBasis,
    ObservationId, Overlap, ResultId, ResultRecord, ResultRevalidation, ResultStatus,
    SourceDeclaration, SourceId, SourceKind, SourceStatus, TransportIdentity, parse_digest_hex,
};
use antecedent_stats::synthesis::{BetweenStudyMethod, MeanPrior, TauPrior};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde_json::{Map, Value as Json, json};

use crate::measurement_model_api::{
    list, member, object, opt_text, parse, present, real, reals, text, texts,
};
use crate::transport_common::execution_context;

// ---------------------------------------------------------------------------
// Refusals as data.
// ---------------------------------------------------------------------------

fn refusal_json(refusal: &EvidenceLedgerRefusal) -> String {
    json!({
        "code": refusal.code,
        "stage": "evidence_ledger",
        "detail": refusal.detail,
        "message": refusal.message,
        "offending": refusal.sources.first().map(SourceId::to_hex),
        "remedy": Json::Null,
        "sources": refusal.sources.iter().map(SourceId::to_hex).collect::<Vec<_>>(),
    })
    .to_string()
}

type Refused<T> = (Option<T>, Option<String>);

/// Exported bytes, identity, carried result ids and `(result id, reason)` of results not carried.
type LedgerExportParts = (Py<PyBytes>, String, Vec<String>, Vec<(String, String)>);

fn refused<T>(refusal: &EvidenceLedgerRefusal) -> Refused<T> {
    (None, Some(refusal_json(refusal)))
}

// ---------------------------------------------------------------------------
// Declarations.
// ---------------------------------------------------------------------------

fn source_id(value: &str) -> PyResult<SourceId> {
    SourceId::from_hex(value)
        .ok_or_else(|| crate::value_err(format!("`{value}` is not a source identity")))
}

fn result_id(value: &str) -> PyResult<ResultId> {
    ResultId::from_hex(value)
        .ok_or_else(|| crate::value_err(format!("`{value}` is not a result identity")))
}

fn source_ids(map: &Map<String, Json>, key: &str) -> PyResult<Vec<SourceId>> {
    texts(map, key)?.iter().map(|value| source_id(value)).collect()
}

fn optional_result(map: &Map<String, Json>, key: &str) -> PyResult<Option<ResultId>> {
    opt_text(map, key)?.map(|value| result_id(&value)).transpose()
}

fn digest(value: &str, what: &str) -> PyResult<antecedent_core::SemanticDigest> {
    parse_digest_hex(value)
        .ok_or_else(|| crate::value_err(format!("{what} must be 64 hexadecimal characters")))
}

fn unhex(value: &str) -> Option<Vec<u8>> {
    if value.len() % 2 != 0 {
        return None;
    }
    let nibble = |byte: u8| match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    };
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| Some((nibble(pair[0])? << 4) | nibble(pair[1])?))
        .collect()
}

fn parse_lineage(value: &Json) -> PyResult<Lineage> {
    let map = object(value, "lineage")?;
    Ok(match text(map, "kind")?.as_str() {
        "original" => Lineage::Original,
        "duplicate" => Lineage::Duplicate { of: source_id(&text(map, "of")?)? },
        "transform" => Lineage::Transform {
            of: source_id(&text(map, "of")?)?,
            transform: digest(&text(map, "transform")?, "a transform identity")?,
        },
        "subset" => Lineage::Subset { of: source_id(&text(map, "of")?)? },
        "prior_plus_likelihood" => Lineage::PriorPlusLikelihood {
            prior: source_id(&text(map, "prior")?)?,
            likelihood: source_id(&text(map, "likelihood")?)?,
        },
        "derived_from_used_data" => {
            Lineage::DerivedFromUsedData { inputs: source_ids(map, "inputs")? }
        }
        other => return Err(crate::value_err(format!("unknown lineage `{other}`"))),
    })
}

fn parse_transport(value: &Json) -> PyResult<TransportIdentity> {
    let map = object(value, "transport identity")?;
    Ok(TransportIdentity {
        source_populations: texts(map, "source_populations")?,
        target_population: text(map, "target_population")?,
        selection_diagram: digest(
            &text(map, "selection_diagram")?,
            "a selection diagram identity",
        )?,
        assumptions: texts(map, "assumptions")?,
    })
}

fn parse_declaration(value: &Json) -> PyResult<SourceDeclaration> {
    let map = object(value, "source declaration")?;
    let name = text(map, "kind")?;
    let kind = SourceKind::from_name(&name)
        .ok_or_else(|| crate::value_err(format!("unknown source kind `{name}`")))?;
    let mut declaration = SourceDeclaration::new(text(map, "label")?, kind);
    if present(map, "observations").is_some() && present(map, "observation_ids").is_some() {
        return Err(crate::value_err(
            "a source states its observations as keys or as identities, not both",
        ));
    }
    if let Some(ids) = present(map, "observation_ids") {
        let ids = ids
            .as_array()
            .ok_or_else(|| crate::value_err("`observation_ids` must be a list or null"))?
            .iter()
            .map(|value| {
                value.as_str().and_then(ObservationId::from_hex).ok_or_else(|| {
                    crate::value_err("an observation identity is 64 hexadecimal characters")
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        declaration = declaration.with_observations(ids);
    }
    if let Some(rows) = present(map, "observations") {
        let rows = rows
            .as_array()
            .ok_or_else(|| crate::value_err("`observations` must be a list or null"))?;
        let mut ids = Vec::with_capacity(rows.len());
        for row in rows {
            let pair = row
                .as_array()
                .filter(|pair| pair.len() == 2)
                .ok_or_else(|| crate::value_err("an observation is a [domain, key] pair"))?;
            let domain = pair[0]
                .as_str()
                .ok_or_else(|| crate::value_err("an observation domain must be a string"))?;
            let key = pair[1]
                .as_str()
                .and_then(unhex)
                .ok_or_else(|| crate::value_err("an observation key must be hexadecimal"))?;
            ids.push(ObservationId::from_key(domain, &key));
        }
        declaration = declaration.with_observations(ids);
    }
    if let Some(lineage) = present(map, "lineage") {
        declaration = declaration.with_lineage(parse_lineage(lineage)?);
    }
    if let Some(transport) = present(map, "transport") {
        declaration = declaration.with_transport(parse_transport(transport)?);
    }
    Ok(declaration)
}

fn parse_summaries(map: &Map<String, Json>, key: &str) -> PyResult<Vec<StudySummary>> {
    let mut out = Vec::new();
    for entry in list(map, key)? {
        let entry = object(entry, "study summary")?;
        out.push(StudySummary {
            source: source_id(&text(entry, "source")?)?,
            estimate: real(entry, "estimate")?,
            variance: real(entry, "variance")?,
        });
    }
    Ok(out)
}

fn parse_pooling(value: &Json) -> PyResult<PoolingRequest> {
    let map = object(value, "pooling request")?;
    Ok(PoolingRequest {
        label: text(map, "label")?,
        summaries: parse_summaries(map, "summaries")?,
        replaces: optional_result(map, "replaces")?,
    })
}

fn parse_pooling_dependence(value: &Json) -> PyResult<PoolingDeclaration> {
    let map = object(value, "pooling dependence")?;
    match text(map, "kind")?.as_str() {
        "independent" => Ok(PoolingDeclaration::Independent),
        "declared_covariance" => {
            Ok(PoolingDeclaration::DeclaredCovariance { covariance: reals(map, "covariance")? })
        }
        other => Err(crate::value_err(format!("unknown pooling dependence `{other}`"))),
    }
}

fn parse_borrowing(value: &Json) -> PyResult<BorrowingRequest> {
    let map = object(value, "borrowing request")?;
    let mean_prior = object(member(map, "mean_prior")?, "mean prior")?;
    let mean_prior = match text(mean_prior, "kind")?.as_str() {
        "flat" => MeanPrior::Flat,
        "normal" => MeanPrior::Normal {
            mean: real(mean_prior, "mean")?,
            variance: real(mean_prior, "variance")?,
        },
        other => return Err(crate::value_err(format!("unknown mean prior `{other}`"))),
    };
    let tau = object(member(map, "tau_prior")?, "tau prior")?;
    Ok(BorrowingRequest {
        label: text(map, "label")?,
        priors: source_ids(map, "priors")?,
        likelihood: parse_summaries(map, "likelihood")?,
        mean_prior,
        tau_prior: TauPrior {
            support: reals(tau, "support")?,
            probabilities: reals(tau, "probabilities")?,
        },
        replaces: optional_result(map, "replaces")?,
    })
}

fn parse_range(value: &Json) -> PyResult<DependenceRangeRequest> {
    let map = object(value, "dependence range request")?;
    let mut studies = Vec::new();
    for entry in list(map, "studies")? {
        let entry = object(entry, "study estimate")?;
        studies.push(StudyEstimate {
            source: source_id(&text(entry, "source")?)?,
            estimate: real(entry, "estimate")?,
            standard_error: real(entry, "standard_error")?,
        });
    }
    Ok(DependenceRangeRequest {
        label: text(map, "label")?,
        studies,
        rho_lo: real(map, "rho_lo")?,
        rho_hi: real(map, "rho_hi")?,
        replaces: optional_result(map, "replaces")?,
    })
}

fn parse_revision(value: &Json) -> PyResult<SourceRevision> {
    let map = object(value, "source revision")?;
    let target = source_id(&text(map, "target")?)?;
    let replacement = |map: &Map<String, Json>| parse_declaration(member(map, "replacement")?);
    match text(map, "kind")?.as_str() {
        "update" => Ok(SourceRevision::Update { target, replacement: replacement(map)? }),
        "replace" => Ok(SourceRevision::Replace { target, replacement: replacement(map)? }),
        "supersede" => Ok(SourceRevision::Supersede { target, replacement: replacement(map)? }),
        "withdraw" => Ok(SourceRevision::Withdraw { target, reason: text(map, "reason")? }),
        other => Err(crate::value_err(format!("unknown source revision `{other}`"))),
    }
}

fn parse_change(value: &Json) -> PyResult<SourceChange> {
    let map = object(value, "source change")?;
    match text(map, "kind")?.as_str() {
        "add" => Ok(SourceChange::Add(parse_declaration(member(map, "source")?)?)),
        "revise" => Ok(SourceChange::Revise(parse_revision(member(map, "revision")?)?)),
        other => Err(crate::value_err(format!("unknown source change `{other}`"))),
    }
}

// ---------------------------------------------------------------------------
// Results as data.
// ---------------------------------------------------------------------------

fn mass_json(mass: &MassAccount) -> Json {
    json!({
        "sources": mass.sources,
        "declared_slots": mass.declared_slots,
        "distinct_observations": mass.distinct_observations,
        "repeated_slots": mass.repeated_slots,
        "unknown_sources": mass.unknown_sources.iter().map(SourceId::to_hex).collect::<Vec<_>>(),
    })
}

fn overlap_json(overlap: Overlap) -> Json {
    match overlap {
        Overlap::Independent => json!({"kind": "independent"}),
        Overlap::SharedObservations { shared } => {
            json!({"kind": "shared_observations", "shared": shared})
        }
        Overlap::Unknown { reason } => json!({"kind": "unknown", "reason": reason}),
        _ => json!({"kind": "unrecognized"}),
    }
}

fn value_json(value: &SynthesisValue) -> Json {
    match value {
        SynthesisValue::FixedEffect(fit) => json!({
            "kind": "fixed_effect",
            "estimate": fit.estimate,
            "variance": fit.variance,
            "weights": fit.weights,
            "heterogeneity_q": fit.heterogeneity_q,
            "degrees_of_freedom": fit.degrees_of_freedom,
        }),
        SynthesisValue::RandomEffects(fit) => json!({
            "kind": "random_effects",
            "estimate": fit.estimate,
            "variance": fit.variance,
            "tau_squared": fit.tau_squared,
            "q": fit.q,
            "degrees_of_freedom": fit.degrees_of_freedom,
            "i_squared": fit.i_squared,
            "weights": fit.weights,
            "method": match fit.method {
                BetweenStudyMethod::DerSimonianLaird => "dersimonian_laird",
                BetweenStudyMethod::RestrictedMaximumLikelihood => "reml",
            },
        }),
        SynthesisValue::Hierarchical(fit) => json!({
            "kind": "hierarchical",
            "mean": fit.mean,
            "variance": fit.variance,
            "study_means": fit.study_means,
            "study_variances": fit.study_variances,
            "tau_posterior": fit.tau_posterior,
        }),
        SynthesisValue::DependenceEnvelope(envelope) => json!({
            "kind": "dependence_envelope",
            "studies": envelope.studies,
            "rho_lo": envelope.rho_lo,
            "rho_hi": envelope.rho_hi,
            "estimate_min": envelope.estimate_min,
            "rho_at_estimate_min": envelope.rho_at_estimate_min,
            "estimate_max": envelope.estimate_max,
            "rho_at_estimate_max": envelope.rho_at_estimate_max,
            "variance_min": envelope.variance_min,
            "rho_at_variance_min": envelope.rho_at_variance_min,
            "variance_max": envelope.variance_max,
            "rho_at_variance_max": envelope.rho_at_variance_max,
        }),
        _ => json!({"kind": "unrecognized"}),
    }
}

fn outcome_json(outcome: &SynthesisOutcome) -> Json {
    json!({
        "result": outcome.result.to_hex(),
        "value": value_json(&outcome.value),
        "mass": mass_json(&outcome.mass),
        "claim_class": outcome.claim_class.as_str(),
        "claim_label": outcome.value.claim_label(),
    })
}

fn status_json(status: &ResultStatus) -> Json {
    match status {
        ResultStatus::Valid => json!({"kind": "valid"}),
        ResultStatus::NeedsRecompute { cause } => {
            json!({"kind": "needs_recompute", "cause": cause.to_hex()})
        }
        ResultStatus::Withdrawn { cause } => json!({"kind": "withdrawn", "cause": cause.to_hex()}),
        ResultStatus::Replaced { by, operation } => json!({
            "kind": "replaced",
            "by": by.to_hex(),
            "operation": operation.to_hex(),
        }),
    }
}

fn record_json(record: &ResultRecord) -> Json {
    json!({
        "id": record.id.to_hex(),
        "label": record.label,
        "kind": record.operation.kind().as_str(),
        "sources": record.operation.direct_sources().iter().map(SourceId::to_hex).collect::<Vec<_>>(),
        "mass": mass_json(&record.mass),
        "claim_class": record.claim_class.as_str(),
        "observations_digest": record.observations_digest.map(|d| d.to_hex()),
        "transport_identities": record.transport_identities.iter().map(antecedent_core::SemanticDigest::to_hex).collect::<Vec<_>>(),
        "data_reuse": record.data_reuse,
        "dependence_declared": record.dependence_declared,
        "replaces": record.replaces.map(|id| id.to_hex()),
        "created_by": record.created_by.to_hex(),
        "status": status_json(&record.status),
        "standing": record.status.revalidation().as_str(),
    })
}

fn source_json(source: &EvidenceSource) -> Json {
    let (observations, unknown_reason) = match &source.effective {
        EffectiveObservations::Known(rows) => (Json::from(rows.len()), Json::Null),
        EffectiveObservations::Unknown(reason) => (Json::Null, Json::from(*reason)),
    };
    let status = match &source.status {
        SourceStatus::Active => json!({"kind": "active"}),
        SourceStatus::Withdrawn { reason, by } => {
            json!({"kind": "withdrawn", "reason": reason, "by": by.to_hex()})
        }
        SourceStatus::Superseded { by, operation } => {
            json!({"kind": "superseded", "by": by.to_hex(), "operation": operation.to_hex()})
        }
        SourceStatus::Replaced { by, operation } => {
            json!({"kind": "replaced", "by": by.to_hex(), "operation": operation.to_hex()})
        }
        SourceStatus::Updated { by, operation } => {
            json!({"kind": "updated", "by": by.to_hex(), "operation": operation.to_hex()})
        }
    };
    let declared = match &source.declaration.observations {
        ObservationBasis::Declared(rows) => Json::from(rows.len()),
        ObservationBasis::Undeclared => Json::Null,
    };
    json!({
        "id": source.id.to_hex(),
        "label": source.declaration.label,
        "kind": source.declaration.kind.as_str(),
        "lineage": source.declaration.lineage.as_str(),
        "lineage_parents": source.declaration.lineage.parents().iter().map(SourceId::to_hex).collect::<Vec<_>>(),
        "declared_observations": declared,
        "effective_observations": observations,
        "unknown_reason": unknown_reason,
        "status": status,
        "transported": source.declaration.transport.is_some(),
        "added_by": source.added_by.to_hex(),
    })
}

fn lineage_json(lineage: &Lineage) -> Json {
    match lineage {
        Lineage::Original => json!({"kind": "original"}),
        Lineage::Duplicate { of } => json!({"kind": "duplicate", "of": of.to_hex()}),
        Lineage::Transform { of, transform } => {
            json!({"kind": "transform", "of": of.to_hex(), "transform": transform.to_hex()})
        }
        Lineage::Subset { of } => json!({"kind": "subset", "of": of.to_hex()}),
        Lineage::PriorPlusLikelihood { prior, likelihood } => json!({
            "kind": "prior_plus_likelihood",
            "prior": prior.to_hex(),
            "likelihood": likelihood.to_hex(),
        }),
        Lineage::DerivedFromUsedData { inputs } => json!({
            "kind": "derived_from_used_data",
            "inputs": inputs.iter().map(SourceId::to_hex).collect::<Vec<_>>(),
        }),
    }
}

/// A declaration in the form [`parse_declaration`] reads back: observations
/// as identities, lineage and transport in full.
fn declaration_json(declaration: &SourceDeclaration) -> Json {
    json!({
        "label": declaration.label,
        "kind": declaration.kind.as_str(),
        "observation_ids": match &declaration.observations {
            ObservationBasis::Declared(rows) => {
                Json::Array(rows.iter().map(|r| Json::String(r.to_hex())).collect())
            }
            ObservationBasis::Undeclared => Json::Null,
        },
        "lineage": lineage_json(&declaration.lineage),
        "transport": declaration.transport.as_ref().map(|t| json!({
            "source_populations": t.source_populations,
            "target_population": t.target_population,
            "selection_diagram": t.selection_diagram.to_hex(),
            "assumptions": t.assumptions,
        })),
    })
}

fn classification_json(classification: &[ResultRevalidation]) -> Json {
    Json::Array(
        classification
            .iter()
            .map(|r| json!({"result": r.result.to_hex(), "class": r.class.as_str()}))
            .collect(),
    )
}

fn report_json(report: &RevisionReport) -> Json {
    json!({
        "operation": report.operation.to_hex(),
        "created_source": report.created_source.map(|id| id.to_hex()),
        "retired_source": report.retired_source.map(|id| id.to_hex()),
        "detected_overlaps": report.detected_overlaps.iter().map(|o| json!({
            "existing": o.existing.to_hex(),
            "shared": o.shared,
        })).collect::<Vec<_>>(),
        "revalidation": classification_json(&report.revalidation),
    })
}

fn ledger_json(ledger: &EvidenceLedger) -> Json {
    json!({
        "head": ledger.head().to_hex(),
        "state_digest": ledger.state_digest().to_hex(),
        "sources": ledger.sources().iter().map(source_json).collect::<Vec<_>>(),
        "results": ledger.results().iter().map(record_json).collect::<Vec<_>>(),
        "operations": ledger.operations().iter().map(|o| json!({
            "id": o.id.to_hex(),
            "name": o.operation.name(),
        })).collect::<Vec<_>>(),
    })
}

fn account_json(account: &ObservationAccount) -> Json {
    json!({
        "mass": mass_json(&account.mass),
        "pairs": account.pairs.iter().map(|pair| json!({
            "first": pair.first.to_hex(),
            "second": pair.second.to_hex(),
            "overlap": overlap_json(pair.overlap),
            "relation": pair.relation.as_str(),
        })).collect::<Vec<_>>(),
    })
}

// ---------------------------------------------------------------------------
// Handles.
// ---------------------------------------------------------------------------

/// The evidence ledger with the numbers computed over it. A refused request
/// leaves the session exactly as it was.
#[pyclass(name = "EvidenceLedgerHandle", skip_from_py_object)]
pub(crate) struct EvidenceLedgerHandle {
    inner: EvidenceLedgerSession,
}

/// A consumed artifact: the replayed ledger and the recomputed syntheses.
#[pyclass(name = "ConsumedEvidenceLedgerHandle", skip_from_py_object)]
pub(crate) struct ConsumedEvidenceLedgerHandle {
    inner: ConsumedEvidenceLedger,
}

#[pymethods]
impl EvidenceLedgerHandle {
    /// An empty session.
    #[new]
    fn new() -> Self {
        Self { inner: StudyBuilder::evidence_ledger() }
    }

    /// Identity of the last applied operation, as lowercase hex.
    fn head(&self) -> String {
        self.inner.head_hex()
    }

    /// Digest of the whole ledger state, as lowercase hex.
    fn state_digest(&self) -> String {
        self.inner.state_digest_hex()
    }

    /// Sources, results and operations of the ledger, as JSON.
    fn ledger_json(&self) -> String {
        ledger_json(self.inner.ledger()).to_string()
    }

    /// The ledger's record of one result, valid or not.
    fn result_json(&self, result: &str) -> PyResult<Option<String>> {
        Ok(self.inner.result(result_id(result)?).map(|r| record_json(r).to_string()))
    }

    /// The standing of one result: still_valid, recompute or withdrawn.
    fn standing(&self, result: &str) -> PyResult<Option<&'static str>> {
        Ok(self
            .inner
            .standing(result_id(result)?)
            .map(antecedent_contract::evidence_ledger::Revalidation::as_str))
    }

    /// Add a source; a refusal leaves the session unchanged.
    fn add_source(&mut self, declaration_json: &str) -> PyResult<Refused<String>> {
        let declaration = parse_declaration(&parse(declaration_json, "source declaration")?)?;
        Ok(match self.inner.add_source(declaration) {
            Ok(id) => (Some(id.to_hex()), None),
            Err(refusal) => refused(&refusal),
        })
    }

    /// A declaration for a fitted model or claim derived from a valid result.
    fn declare_from_result(
        &self,
        result: &str,
        label: &str,
        kind: &str,
    ) -> PyResult<Refused<String>> {
        let kind = SourceKind::from_name(kind)
            .ok_or_else(|| crate::value_err(format!("unknown source kind `{kind}`")))?;
        Ok(match self.inner.declare_from_result(result_id(result)?, label, kind) {
            Ok(declaration) => (Some(declaration_json(&declaration).to_string()), None),
            Err(refusal) => refused(&refusal),
        })
    }

    /// The mass the sources contribute, each observation counted once, and
    /// how every pair overlaps.
    fn account_observations(&self, sources_json: &str) -> PyResult<Refused<String>> {
        let parsed = parse(sources_json, "sources")?;
        let sources = parsed
            .as_array()
            .ok_or_else(|| crate::value_err("sources must be a list"))?
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .ok_or_else(|| crate::value_err("a source identity must be a string"))
                    .and_then(source_id)
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(match self.inner.account_observations(&sources) {
            Ok(account) => (Some(account_json(&account).to_string()), None),
            Err(refusal) => refused(&refusal),
        })
    }

    /// Pool the summaries of two or more studies.
    #[pyo3(signature = (request_json, dependence_json, *, cancel=None))]
    fn pool_summaries(
        &mut self,
        request_json: &str,
        dependence_json: &str,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Refused<String>> {
        let request = parse_pooling(&parse(request_json, "pooling request")?)?;
        let dependence = parse_pooling_dependence(&parse(dependence_json, "pooling dependence")?)?;
        let ctx = execution_context(0, None, cancel);
        Ok(match self.inner.pool_summaries(request, dependence, &ctx) {
            Ok(outcome) => (Some(outcome_json(&outcome).to_string()), None),
            Err(refusal) => refused(&refusal),
        })
    }

    /// Random-effects pooling of independent studies.
    #[pyo3(signature = (request_json, method, *, cancel=None))]
    fn random_effects(
        &mut self,
        request_json: &str,
        method: &str,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Refused<String>> {
        let request = parse_pooling(&parse(request_json, "pooling request")?)?;
        let method = match method {
            "dersimonian_laird" => BetweenStudy::DerSimonianLaird,
            "reml" => BetweenStudy::RestrictedMaximumLikelihood,
            other => {
                return Err(crate::value_err(format!(
                    "unknown between-study method `{other}`; use dersimonian_laird or reml"
                )));
            }
        };
        let ctx = execution_context(0, None, cancel);
        Ok(match self.inner.random_effects(request, method, &ctx) {
            Ok(outcome) => (Some(outcome_json(&outcome).to_string()), None),
            Err(refusal) => refused(&refusal),
        })
    }

    /// Hierarchical normal-normal borrowing with a declared finite prior on
    /// the between-study standard deviation.
    #[pyo3(signature = (request_json, *, cancel=None))]
    fn hierarchical_borrowing(
        &mut self,
        request_json: &str,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Refused<String>> {
        let request = parse_borrowing(&parse(request_json, "borrowing request")?)?;
        let ctx = execution_context(0, None, cancel);
        Ok(match self.inner.hierarchical_borrowing(request, &ctx) {
            Ok(outcome) => (Some(outcome_json(&outcome).to_string()), None),
            Err(refusal) => refused(&refusal),
        })
    }

    /// The pooled estimate and variance over a declared correlation range.
    #[pyo3(signature = (request_json, *, cancel=None))]
    fn dependence_range(
        &mut self,
        request_json: &str,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Refused<String>> {
        let request = parse_range(&parse(request_json, "dependence range request")?)?;
        let ctx = execution_context(0, None, cancel);
        Ok(match self.inner.dependence_range(request, &ctx) {
            Ok(outcome) => (Some(outcome_json(&outcome).to_string()), None),
            Err(refusal) => refused(&refusal),
        })
    }

    /// Carry transported claims to the target alongside native target data.
    #[pyo3(signature = (label, transported_json, native_json, replaces=None))]
    fn transport_claim(
        &mut self,
        label: &str,
        transported_json: &str,
        native_json: &str,
        replaces: Option<&str>,
    ) -> PyResult<Refused<String>> {
        let ids = |json: &str, what: &str| -> PyResult<Vec<SourceId>> {
            let parsed = parse(json, what)?;
            let map = json!({"ids": parsed});
            source_ids(map.as_object().ok_or_else(|| crate::value_err("internal"))?, "ids")
        };
        let transported = ids(transported_json, "transported sources")?;
        let native = ids(native_json, "native sources")?;
        let replaces = replaces.map(result_id).transpose()?;
        Ok(match self.inner.transport_claim(label, transported, native, replaces) {
            Ok(id) => (Some(id.to_hex()), None),
            Err(refusal) => refused(&refusal),
        })
    }

    /// Cite a valid existing result again.
    fn reuse_result(&mut self, label: &str, of: &str) -> PyResult<Refused<String>> {
        let of = result_id(of)?;
        Ok(match self.inner.reuse_result(label, of) {
            Ok(id) => (Some(id.to_hex()), None),
            Err(refusal) => refused(&refusal),
        })
    }

    /// Declare that the same observations serve a later stage.
    fn repeat_observations(
        &mut self,
        label: &str,
        sources_json: &str,
        stage: &str,
    ) -> PyResult<Refused<String>> {
        let parsed = parse(sources_json, "sources")?;
        let map = json!({"ids": parsed});
        let sources =
            source_ids(map.as_object().ok_or_else(|| crate::value_err("internal"))?, "ids")?;
        Ok(match self.inner.repeat_observations(label, sources, stage) {
            Ok(id) => (Some(id.to_hex()), None),
            Err(refusal) => refused(&refusal),
        })
    }

    /// Apply one revision; results that rested on the retired source are
    /// reclassified.
    fn revise_source(&mut self, revision_json: &str) -> PyResult<Refused<String>> {
        let revision = parse_revision(&parse(revision_json, "source revision")?)?;
        Ok(match self.inner.revise_source(revision) {
            Ok(report) => (Some(report_json(&report).to_string()), None),
            Err(refusal) => refused(&refusal),
        })
    }

    /// Reclassify every result against the current source standings.
    fn revalidate_downstream(&mut self) -> Refused<String> {
        match self.inner.revalidate_downstream() {
            Ok(classification) => (Some(classification_json(&classification).to_string()), None),
            Err(refusal) => refused(&refusal),
        }
    }

    /// Always the typed refusal: the ledger licenses no interval, posterior
    /// or coverage statement for pooled evidence.
    fn pooled_interval(&self, result: &str) -> PyResult<String> {
        let result = result_id(result)?;
        match self.inner.pooled_interval(result) {
            Err(refusal) => Ok(refusal_json(&refusal)),
            Ok(never) => match never {},
        }
    }

    /// The numbers of a result that is still valid. `(None, None)` is a valid
    /// result registered for accounting only.
    fn synthesis(&self, result: &str) -> PyResult<Refused<String>> {
        let result = result_id(result)?;
        Ok(match self.inner.synthesis(result) {
            Ok(Some(value)) => (Some(value_json(value).to_string()), None),
            Ok(None) => (None, None),
            Err(refusal) => refused(&refusal),
        })
    }

    /// Apply a batch of added and retired evidence to a copy of this
    /// session; all or nothing, and this session is unchanged either way.
    fn refresh(
        &self,
        changes_json: &str,
    ) -> PyResult<Refused<(EvidenceLedgerHandle, String, String)>> {
        let parsed = parse(changes_json, "source changes")?;
        let changes = parsed
            .as_array()
            .ok_or_else(|| crate::value_err("source changes must be a list"))?
            .iter()
            .map(parse_change)
            .collect::<PyResult<Vec<_>>>()?;
        Ok(match self.inner.refresh(changes) {
            Ok(LedgerRefresh { session, reports, classification }) => (
                Some((
                    EvidenceLedgerHandle { inner: session },
                    Json::Array(reports.iter().map(report_json).collect()).to_string(),
                    classification_json(&classification).to_string(),
                )),
                None,
            ),
            Err(refusal) => refused(&refusal),
        })
    }

    /// Write the ledger and the syntheses bound to its results.
    fn export(&self, py: Python<'_>, artifact_id: &str) -> Refused<LedgerExportParts> {
        match self.inner.export(artifact_id) {
            Ok(done) => (
                Some((
                    PyBytes::new(py, &done.bytes).unbind(),
                    done.identity,
                    done.carried.iter().map(ResultId::to_hex).collect(),
                    done.not_carried
                        .iter()
                        .map(|(id, why)| (id.to_hex(), (*why).to_owned()))
                        .collect(),
                )),
                None,
            ),
            Err(refusal) => refused(&refusal),
        }
    }
}

#[pymethods]
impl ConsumedEvidenceLedgerHandle {
    /// The replayed ledger and the recomputed syntheses, as JSON.
    fn summary_json(&self) -> String {
        let consumed = &self.inner;
        let mut summary = ledger_json(consumed.ledger());
        summary["identity"] = json!(consumed.identity());
        summary["syntheses"] = Json::Array(
            consumed
                .syntheses()
                .iter()
                .map(|s| {
                    json!({
                        "result": s.result.to_hex(),
                        "method": s.method,
                        "values": s.values,
                    })
                })
                .collect(),
        );
        summary.to_string()
    }
}

// ---------------------------------------------------------------------------
// Entry points.
// ---------------------------------------------------------------------------

/// Independently consume an evidence ledger artifact by replay and
/// recomputation, against an identity the consumer retained.
#[pyfunction]
#[pyo3(signature = (artifact, expected_identity=None))]
fn evidence_ledger_consume(
    artifact: &[u8],
    expected_identity: Option<&str>,
) -> Refused<ConsumedEvidenceLedgerHandle> {
    match consume_evidence_ledger_artifact(artifact, expected_identity) {
        Ok(inner) => (Some(ConsumedEvidenceLedgerHandle { inner }), None),
        Err(refusal) => refused(&refusal),
    }
}

/// Content identity of a source declaration is assigned by the ledger on
/// entry; the observation identity of a key is a pure function of its domain
/// and bytes.
#[pyfunction]
fn evidence_ledger_observation_id(domain: &str, key: &[u8]) -> String {
    ObservationId::from_key(domain, key).to_hex()
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<EvidenceLedgerHandle>()?;
    module.add_class::<ConsumedEvidenceLedgerHandle>()?;
    module.add_function(wrap_pyfunction!(evidence_ledger_consume, module)?)?;
    module.add_function(wrap_pyfunction!(evidence_ledger_observation_id, module)?)?;
    Ok(())
}
