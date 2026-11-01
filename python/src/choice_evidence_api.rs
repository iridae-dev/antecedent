//! Python bridge for the categorical-choice evidence-to-decision lifecycle.
//!
//! Python declares the frame, the retained sources, the observations, a sequential rule and a
//! ranking request as JSON; every check, posterior functional, EVSI, identity and refusal is
//! Rust's. A refusal comes back as structured JSON for the Python layer to raise as a typed
//! exception. Nothing here claims calibration, behavioural identification or transport.

use antecedent::choice_evidence::{
    ChoiceFrame, ChoiceMeaning, ChoiceObservations, ChoiceRefusal, ConflictPolicy, EvsiMethod,
    LifecycleRequest, LifecycleResult, LookOutcome, RankingRequest, RetainedChoiceEvidence,
    SequentialRule, SourceUse, StudyCandidate, evaluate_lifecycle,
};
use pyo3::prelude::*;
use serde_json::{Map, Value as Json, json};

use crate::measurement_model_api::{
    list, member, object, opt_text, parse, present, real, reals, text,
};

type Refused = (Option<String>, Option<String>);

fn refusal_json(refusal: &ChoiceRefusal) -> String {
    json!({
        "code": refusal.registered_code(),
        "stage": "choice_evidence",
        "detail": refusal.reason_code(),
        "message": refusal.to_string(),
        "offending": Json::Null,
        "remedy": refusal.next_attempt(),
    })
    .to_string()
}

fn strings(map: &Map<String, Json>, key: &str) -> PyResult<Vec<String>> {
    list(map, key)?
        .iter()
        .map(|v| {
            v.as_str()
                .map(str::to_owned)
                .ok_or_else(|| crate::value_err(format!("`{key}` must hold strings")))
        })
        .collect()
}

fn frame(value: &Json) -> PyResult<ChoiceFrame> {
    let map = object(value, "frame")?;
    let options = list(map, "options")?
        .iter()
        .map(|v| {
            v.as_str().map(str::to_owned).ok_or_else(|| crate::value_err("options must be strings"))
        })
        .collect::<PyResult<Vec<_>>>()?;
    let meaning = match text(map, "meaning")?.as_str() {
        "stated_preference" => ChoiceMeaning::StatedPreference,
        "observed_behavior" => ChoiceMeaning::ObservedBehavior,
        other => return Err(crate::value_err(format!("unknown choice meaning `{other}`"))),
    };
    Ok(ChoiceFrame { options, population: text(map, "population")?, meaning })
}

fn sources(map: &Map<String, Json>) -> PyResult<Vec<SourceUse>> {
    let Some(items) = present(map, "sources") else { return Ok(Vec::new()) };
    let items = items.as_array().ok_or_else(|| crate::value_err("`sources` must be a list"))?;
    items
        .iter()
        .map(|item| {
            let m = object(item, "source")?;
            Ok(SourceUse {
                evidence: RetainedChoiceEvidence {
                    label: text(m, "label")?,
                    frame: frame(member(m, "frame")?)?,
                    counts: reals(m, "counts")?,
                    observation_ids: strings(m, "observation_ids")?,
                },
                discount: real(m, "discount")?,
                transfer_rationale: opt_text(m, "transfer_rationale")?,
            })
        })
        .collect()
}

fn request(map: &Map<String, Json>) -> PyResult<LifecycleRequest> {
    let conflict_policy = match opt_text(map, "conflict_policy")?.as_deref() {
        None | Some("report") => ConflictPolicy::Report,
        Some("refuse") => ConflictPolicy::Refuse,
        Some(other) => return Err(crate::value_err(format!("unknown conflict policy `{other}`"))),
    };
    let observations = present(map, "observations")
        .map(|v| {
            let m = object(v, "observations")?;
            Ok::<_, PyErr>(ChoiceObservations {
                counts: reals(m, "counts")?,
                observation_ids: strings(m, "observation_ids")?,
            })
        })
        .transpose()?;
    let rule = present(map, "rule")
        .map(|v| {
            let m = object(v, "rule")?;
            Ok::<_, PyErr>(SequentialRule {
                min_probability_best: real(m, "min_probability_best")?,
                max_expected_loss: real(m, "max_expected_loss")?,
                min_observations: real(m, "min_observations")?,
                max_observations: real(m, "max_observations")?,
                allocation_floor: real(m, "allocation_floor")?,
            })
        })
        .transpose()?;
    let ranking = present(map, "ranking")
        .map(|v| {
            let m = object(v, "ranking")?;
            let candidates = list(m, "candidates")?
                .iter()
                .map(|c| {
                    let cm = object(c, "candidate")?;
                    let n = member(cm, "observations")?
                        .as_u64()
                        .and_then(|n| u32::try_from(n).ok())
                        .ok_or_else(|| crate::value_err("`observations` must be a count"))?;
                    Ok(StudyCandidate {
                        label: text(cm, "label")?,
                        observations: n,
                        cost: real(cm, "cost")?,
                    })
                })
                .collect::<PyResult<Vec<_>>>()?;
            let seed = member(m, "seed")?
                .as_u64()
                .ok_or_else(|| crate::value_err("`seed` must be a non-negative integer"))?;
            Ok::<_, PyErr>(RankingRequest { values: reals(m, "values")?, candidates, seed })
        })
        .transpose()?;
    Ok(LifecycleRequest {
        frame: frame(member(map, "frame")?)?,
        initial: reals(map, "initial")?,
        sources: sources(map)?,
        conflict_policy,
        observations,
        rule,
        ranking,
    })
}

fn result_json(out: &LifecycleResult) -> Json {
    let prior = &out.prior;
    let mut doc = Map::new();
    doc.insert(
        "prior".into(),
        json!({
            "options": prior.frame.options,
            "alpha": prior.alpha,
            "borrowed_observations": prior.borrowed_observations,
            "digest": prior.digest(),
            "has_conflict": prior.has_conflict(),
            "sources": prior.sources.iter().map(|s| json!({
                "label": s.label,
                "observation_ids": s.observation_ids,
                "discount": s.discount,
                "transferred": s.transferred,
                "conflict_log_bayes_factor": s.conflict_log_bayes_factor,
                "conflicting": s.conflicting,
            })).collect::<Vec<_>>(),
            "statements": prior.statements,
            "calibration": "unmeasured",
        }),
    );
    if let Some(i) = &out.inference {
        doc.insert(
            "inference".into(),
            json!({
                "alpha": i.posterior.alpha(),
                "mean": i.mean,
                "probability_best": i.probability_best,
                "expected_loss": i.expected_loss,
                "observations": i.observations,
                "lineage": i.lineage,
                "statements": i.statements,
                "calibration": "unmeasured",
            }),
        );
    }
    if let Some(look) = &out.look {
        let value = match look {
            LookOutcome::Select { option } => json!({"action": "select", "option": option}),
            LookOutcome::BudgetExhausted => json!({"action": "budget_exhausted"}),
            LookOutcome::Continue { allocation } => {
                json!({"action": "continue", "allocation": allocation})
            }
        };
        doc.insert("look".into(), value);
        doc.insert("stopping_statement".into(), json!(out.stopping_statement));
    }
    if let Some(rows) = &out.ranking {
        doc.insert(
            "ranking".into(),
            Json::Array(
                rows.iter()
                    .map(|r| {
                        json!({
                            "label": r.candidate.label,
                            "observations": r.candidate.observations,
                            "cost": r.candidate.cost,
                            "evsi": r.evsi,
                            "standard_error": r.standard_error,
                            "net_value": r.net_value,
                            "rank_uncertain": r.rank_uncertain,
                            "method": match r.method {
                                EvsiMethod::Exact { compositions } =>
                                    json!({"kind": "exact", "compositions": compositions}),
                                EvsiMethod::MonteCarlo { draws } =>
                                    json!({"kind": "monte_carlo", "draws": draws}),
                            },
                        })
                    })
                    .collect(),
            ),
        );
    }
    Json::Object(doc)
}

#[pyfunction]
fn choice_evidence_evaluate(payload_json: &str) -> PyResult<Refused> {
    let value = parse(payload_json, "choice evidence")?;
    let map = object(&value, "choice evidence")?;
    let req = request(map)?;
    Ok(match evaluate_lifecycle(&req, &|| false) {
        Ok(out) => (Some(result_json(&out).to_string()), None),
        Err(refusal) => (None, Some(refusal_json(&refusal))),
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(choice_evidence_evaluate, module)?)?;
    Ok(())
}
