//! Checked Python bridge for selective DML AIPW and DR-Learner execution.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::NumericColumn;
use std::collections::BTreeMap;

use antecedent::analysis::recalc_dr::{DrEstimator, DrRequest, DrSession, execute_dr_with_receipt};
use antecedent::analysis::recalc_receipt::{
    RecalcRequestCode, RecalcRunError, TargetWeights, UtilitySpec,
};
use antecedent::estimate::fit_counts::count_resolved_fits;
use antecedent::estimate::{DmlAte, DmlScore, DrLearner, LearnerSpec, OverlapPolicy};
use antecedent::execution::recalc::{RecalcCapabilities, ResumeContext, RetargetSupport};
use antecedent::io::recalc_receipt_artifact::{
    CapabilitiesWire, DeclaredStageWire, RecalcReceiptArtifact, ResumeWire, identities_from_wire,
    identities_to_wire, plan_to_wire,
};
use antecedent::prelude::{ExecutionContext, VariableId};
use pyo3::prelude::*;
use serde::Deserialize;

use crate::recalc_api::{
    RunPayload, artifact_invalid, bounded_context, context, counts_wire, invalid, parse_json,
    plan_refusal_json, refusal_json,
};
use crate::recalc_bounds::{bounded_prediction_rows, check_columns};
use crate::{detach_catch, py_msg};

type ScoreExportPayload = (Option<(String, Vec<u8>)>, Option<String>);
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OverlapWire {
    clip: Option<f64>,
    trim: Option<f64>,
}

#[derive(Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
struct ConfigWire {
    learner: Option<serde_json::Value>,
    outcome: Option<serde_json::Value>,
    treatment: Option<serde_json::Value>,
    final_learner: Option<serde_json::Value>,
    score: Option<String>,
    folds: Option<usize>,
    overlap: Option<OverlapWire>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RequestWire {
    edges: Vec<(String, String)>,
    treatment: String,
    outcome: String,
    kind: String,
    config: ConfigWire,
    benefit_per_unit: f64,
    cost: f64,
    target_weights: Option<Vec<f64>>,
    target_depends_on: Vec<String>,
}

fn learner(value: &serde_json::Value) -> PyResult<LearnerSpec> {
    let spec: LearnerSpec = match value {
        serde_json::Value::String(name) => LearnerSpec::parse(name).map_err(|e| {
            invalid(antecedent::reason::reason_details::recalc::DR_INVALID_LEARNER, e.to_string())
        })?,
        _ => serde_json::from_value(value.clone()).map_err(|e| {
            invalid(antecedent::reason::reason_details::recalc::DR_INVALID_LEARNER, e.to_string())
        })?,
    };
    spec.validate().map_err(|e| {
        invalid(antecedent::reason::reason_details::recalc::DR_INVALID_LEARNER, e.to_string())
    })?;
    Ok(spec)
}

fn estimator(kind: &str, config: ConfigWire) -> PyResult<DrEstimator> {
    // Any fold count of at least two is valid; the fold plan refuses a count above the number
    // of rows when the cross-fit is built.
    if config.folds.is_some_and(|n| n < 2) {
        return Err(invalid(
            antecedent::reason::reason_details::recalc::DR_INVALID_FOLDS,
            "fold count must be at least 2",
        ));
    }
    let overlap =
        config.overlap.map(|o| OverlapPolicy::RequireDiagnostics { clip: o.clip, trim: o.trim });
    if let Some(policy) = overlap {
        policy.validate().map_err(|e| {
            invalid(antecedent::reason::reason_details::recalc::DR_INVALID_OVERLAP, e.to_string())
        })?;
    }
    let shared = config.learner.as_ref().map(learner).transpose()?;
    let outcome = config.outcome.as_ref().map(learner).transpose()?;
    let treatment = config.treatment.as_ref().map(learner).transpose()?;
    match kind {
        "dml" => {
            if config.final_learner.is_some() {
                return Err(invalid(
                    antecedent::reason::reason_details::recalc::DR_INVALID_MODEL,
                    "DML has no final CATE learner",
                ));
            }
            let mut model = DmlAte::new();
            if let Some(value) = shared {
                model = model.with_learner(value);
            }
            if let Some(value) = outcome {
                model = model.with_outcome(value);
            }
            if let Some(value) = treatment {
                model = model.with_treatment(value);
            }
            if let Some(value) = config.folds {
                model = model.with_folds(value);
            }
            if let Some(value) = config.score {
                model = model.with_score(DmlScore::parse(&value).map_err(|e| {
                    invalid(
                        antecedent::reason::reason_details::recalc::DR_INVALID_SCORE,
                        e.to_string(),
                    )
                })?);
            }
            if let Some(value) = overlap {
                model = model.with_overlap(value);
            }
            Ok(DrEstimator::Dml(model))
        }
        "cate" => {
            if config.score.is_some() {
                return Err(invalid(
                    antecedent::reason::reason_details::recalc::DR_INVALID_MODEL,
                    "DR-Learner declares no selectable score",
                ));
            }
            let mut model = DrLearner::new();
            if let Some(value) = shared {
                model = model.with_learner(value);
            }
            if let Some(value) = outcome {
                model = model.with_outcome(value);
            }
            if let Some(value) = treatment {
                model = model.with_treatment(value);
            }
            if let Some(value) = config.final_learner {
                model = model.with_final_learner(learner(&value)?);
            }
            if let Some(value) = config.folds {
                model = model.with_folds(value);
            }
            if let Some(value) = overlap {
                model = model.with_overlap(value);
            }
            Ok(DrEstimator::Cate(model))
        }
        _ => Err(invalid(
            antecedent::reason::reason_details::recalc::DR_INVALID_MODEL,
            "unknown DR family model",
        )),
    }
}

fn build_request(
    names: Vec<String>,
    columns: &[NumericColumn<'_>],
    specification: &str,
    ctx: &ExecutionContext,
) -> PyResult<DrRequest> {
    check_columns(columns, ctx)?;
    let wire: RequestWire = parse_json(specification, "DR request")?;
    if names.is_empty() || names.len() != columns.len() {
        return Err(invalid(
            antecedent::reason::reason_details::recalc::INVALID_DATA,
            "one column per name required",
        ));
    }
    for (i, name) in names.iter().enumerate() {
        if name.trim().is_empty() || names[..i].contains(name) {
            return Err(invalid(
                antecedent::reason::reason_details::recalc::INVALID_DATA,
                "unique nonempty names required",
            ));
        }
    }
    let position = |name: &str| {
        names.iter().position(|n| n == name).and_then(|p| u32::try_from(p).ok()).ok_or_else(|| {
            invalid(
                antecedent::reason::reason_details::recalc::UNKNOWN_VARIABLE,
                format!("no column named `{name}`"),
            )
        })
    };
    let treatment = position(&wire.treatment)?;
    let outcome = position(&wire.outcome)?;
    let edges = wire
        .edges
        .iter()
        .map(|(a, b)| Ok((position(a)?, position(b)?)))
        .collect::<PyResult<Vec<_>>>()?;
    if wire.target_weights.is_none() && !wire.target_depends_on.is_empty() {
        return Err(invalid(
            antecedent::reason::reason_details::recalc::INVALID_REQUEST,
            "weight dependencies need weights",
        ));
    }
    let target = wire
        .target_weights
        .map(|weights| -> PyResult<TargetWeights> {
            Ok(TargetWeights {
                weights,
                depends_on: wire
                    .target_depends_on
                    .iter()
                    .map(|n| position(n).map(VariableId::from_raw))
                    .collect::<PyResult<Vec<_>>>()?,
            })
        })
        .transpose()?;
    let estimator = estimator(&wire.kind, wire.config)?;
    Ok(DrRequest {
        columns: crate::named_f64_columns(names, columns)?.into(),
        edges,
        treatment,
        outcome,
        estimator,
        target,
        utility: UtilitySpec { benefit_per_unit: wire.benefit_per_unit, cost: wire.cost },
    })
}

fn dr_error_refusal(error: &RecalcRunError) -> String {
    match error {
        RecalcRunError::Refused(plan) => plan_refusal_json(plan),
        RecalcRunError::NoLiveState(stage) => refusal_json(
            antecedent::reason::reason_code!("score_table_unavailable"),
            &stage.label(),
            antecedent::reason::reason_details::recalc::NO_LIVE_STATE,
            Some(&stage.label()),
            None,
            "no retained DR state",
            None,
        ),
        RecalcRunError::Request(detail) => {
            let detail = detail.as_str();
            let (code, stage) = match detail {
                antecedent::reason::reason_details::recalc::DR_PREDICTION_SCHEMA_MISMATCH
                | antecedent::reason::reason_details::recalc::DR_FIT_COUNTER_MISMATCH
                | antecedent::reason::reason_details::recalc::DR_EFFECT_UNAVAILABLE
                | antecedent::reason::reason_details::recalc::MEMORY_BUDGET_EXCEEDED => {
                    (antecedent::reason::reason_code!("invalid_argument"), "score_artifact")
                }
                antecedent::reason::reason_details::recalc::DR_PREDICTOR_UNAVAILABLE
                | antecedent::reason::reason_details::recalc::DR_SCORES_UNAVAILABLE
                | antecedent::reason::reason_details::recalc::DR_PREDICTION_OUT_OF_SUPPORT => {
                    (antecedent::reason::reason_code!("route_not_supported"), "score_artifact")
                }
                antecedent::reason::reason_details::recalc::INVALID_DATA => {
                    (antecedent::reason::reason_code!("invalid_argument"), "data_snapshot")
                }
                antecedent::reason::reason_details::recalc::INVALID_GRAPH => {
                    (antecedent::reason::reason_code!("invalid_argument"), "graph")
                }
                antecedent::reason::reason_details::recalc::INVALID_QUERY => {
                    (antecedent::reason::reason_code!("invalid_argument"), "query")
                }
                antecedent::reason::reason_details::recalc::CANCELLED => {
                    (antecedent::reason::reason_code!("cancelled_no_claim"), "score_artifact")
                }
                antecedent::reason::reason_details::recalc::INVALID_TARGET_WEIGHTS => {
                    (antecedent::reason::reason_code!("invalid_argument"), "target_population")
                }
                antecedent::reason::reason_details::recalc::DR_TARGET_INCOMPATIBLE => {
                    (antecedent::reason::reason_code!("route_not_supported"), "target_population")
                }
                _ => (antecedent::reason::reason_code!("invalid_argument"), "score_artifact"),
            };
            refusal_json(code, stage, detail, None, None, &error.to_string(), None)
        }
        RecalcRunError::Execution(cause) => {
            use antecedent::CausalError;
            use antecedent::estimate::EstimationError;
            use antecedent::inference::{GlmRefusalKind, StatsError};
            let (code, detail, stage) = match cause.peeled() {
                CausalError::Cancelled { .. } => (
                    antecedent::reason::reason_code!("cancelled_no_claim"),
                    antecedent::reason::reason_details::recalc::CANCELLED,
                    "score_artifact",
                ),
                CausalError::Conflict { what: "source", .. } => (
                    antecedent::reason::reason_code!("invalid_argument"),
                    antecedent::reason::reason_details::recalc::DR_SOURCE_MISMATCH,
                    "data_snapshot",
                ),
                CausalError::Estimate(
                    EstimationError::Refused { code, .. }
                    | EstimationError::RefusedWithFields { code, .. },
                ) if *code == "cancelled_no_claim" => (
                    antecedent::reason::reason_code!("cancelled_no_claim"),
                    antecedent::reason::reason_details::recalc::CANCELLED,
                    "score_artifact",
                ),
                CausalError::Estimate(
                    EstimationError::Refused { code, .. }
                    | EstimationError::RefusedWithFields { code, .. },
                ) if *code == "arm_not_populated" => (
                    antecedent::reason::reason_code!("arm_not_populated"),
                    antecedent::reason::reason_details::recalc::DR_ARM_NOT_POPULATED,
                    "score_artifact",
                ),
                CausalError::Estimate(
                    EstimationError::Refused { code, .. }
                    | EstimationError::RefusedWithFields { code, .. },
                ) if *code == "invalid_argument" => (
                    antecedent::reason::reason_code!("invalid_argument"),
                    antecedent::reason::reason_details::recalc::DR_MODEL_DATA_INVALID,
                    "data_snapshot",
                ),
                CausalError::Estimate(
                    EstimationError::Refused { code, .. }
                    | EstimationError::RefusedWithFields { code, .. },
                ) if *code == "design_rank_deficient" => (
                    antecedent::reason::reason_code!("design_rank_deficient"),
                    antecedent::reason::reason_details::recalc::DR_DESIGN_RANK_DEFICIENT,
                    "score_artifact",
                ),
                CausalError::NotIdentified { .. } => (
                    antecedent::reason::reason_code!("effect_not_identified"),
                    antecedent::reason::reason_details::recalc::DR_NOT_IDENTIFIED,
                    "identification",
                ),
                CausalError::Estimate(EstimationError::Data(_) | EstimationError::Query(_)) => (
                    antecedent::reason::reason_code!("invalid_argument"),
                    antecedent::reason::reason_details::recalc::DR_MODEL_DATA_INVALID,
                    "data_snapshot",
                ),
                CausalError::Estimate(EstimationError::Stats(StatsError::GlmRefused {
                    kind,
                    ..
                })) => (
                    antecedent::reason::reason_code!("route_not_supported"),
                    match kind {
                        GlmRefusalKind::NonConverged => {
                            antecedent::reason::reason_details::recalc::DR_GLM_NOT_CONVERGED
                        }
                        GlmRefusalKind::Separated => {
                            antecedent::reason::reason_details::recalc::DR_GLM_SEPARATED
                        }
                        GlmRefusalKind::BoundarySaturated => {
                            antecedent::reason::reason_details::recalc::DR_GLM_BOUNDARY_SATURATED
                        }
                    },
                    "score_artifact",
                ),
                CausalError::Estimate(EstimationError::Stats(StatsError::RankDeficient {
                    ..
                })) => (
                    antecedent::reason::reason_code!("design_rank_deficient"),
                    antecedent::reason::reason_details::recalc::DR_DESIGN_RANK_DEFICIENT,
                    "score_artifact",
                ),
                _ => (
                    antecedent::reason::reason_code!("route_not_supported"),
                    antecedent::reason::reason_details::recalc::DR_FIT_REFUSED,
                    "score_artifact",
                ),
            };
            refusal_json(code, stage, detail, None, None, &cause.to_string(), None)
        }
        RecalcRunError::Receipt(_) => refusal_json(
            antecedent::reason::reason_code!("invalid_argument"),
            "score_artifact",
            antecedent::reason::reason_details::recalc::DR_RECEIPT_INVALID,
            None,
            None,
            &error.to_string(),
            None,
        ),
    }
}

#[pyclass(name = "DrSessionHandle")]
pub struct PyDrSession {
    inner: DrSession,
}
impl PyDrSession {
    fn capabilities(&self) -> RecalcCapabilities {
        RecalcCapabilities {
            retarget: RetargetSupport::Supported,
            request: antecedent::execution::recalc::RequestSupport::OnGrid,
            boundary: self.inner.boundary(),
        }
    }
}
#[pymethods]
impl PyDrSession {
    #[new]
    fn new() -> Self {
        Self { inner: DrSession::new() }
    }
    #[staticmethod]
    fn resume(previous_json: &str, resume_json: &str) -> PyResult<Self> {
        let previous: Vec<DeclaredStageWire> = parse_json(previous_json, "previous")?;
        let previous = identities_from_wire(&previous).map_err(|e| artifact_invalid(&e))?;
        let wire: ResumeWire = parse_json(resume_json, "resume")?;
        let resume = ResumeContext {
            portable_fit: wire.portable_fit,
            portable_scores: wire.portable_scores,
            supplied_data: wire.supplied_data,
            supplied_provider: wire.supplied_provider,
            scores_snapshot_bound: wire.scores_snapshot_bound,
        };
        Ok(Self { inner: DrSession::resume(previous, resume) })
    }
    fn is_live(&self) -> bool {
        self.inner.is_live()
    }
    fn identities_json(&self) -> PyResult<String> {
        serde_json::to_string(&identities_to_wire(self.inner.identities())).map_err(py_msg)
    }
    fn capabilities_json(&self) -> PyResult<String> {
        serde_json::to_string(&CapabilitiesWire::from_capabilities(&self.capabilities()))
            .map_err(py_msg)
    }
    fn prediction_columns(&self) -> Option<Vec<String>> {
        self.inner.prediction_columns()
    }
    fn row_ids(&self) -> Option<Vec<u64>> {
        self.inner.score_table().map(|t| t.row_index.iter().map(|id| id.get()).collect())
    }
    fn score_contrast(&self) -> Option<Vec<f64>> {
        let t = self.inner.score_table()?;
        let c = t.columns.iter().position(|c| c.arm == 0 && c.threshold.is_none())?;
        let a = t.columns.iter().position(|c| c.arm == 1 && c.threshold.is_none())?;
        Some(t.column(a).ok()?.iter().zip(t.column(c).ok()?).map(|(a, c)| a - c).collect())
    }
    #[pyo3(signature=(names,columns,specification,*,seed=1,threads=None,memory_limit_bytes=None,cancel=None))]
    #[allow(clippy::too_many_arguments)]
    fn plan(
        &self,
        names: Vec<String>,
        columns: Vec<NumericColumn<'_>>,
        specification: &str,
        seed: u64,
        threads: Option<u32>,
        memory_limit_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<String> {
        let ctx = bounded_context(seed, threads, memory_limit_bytes, cancel);
        let request = build_request(names, &columns, specification, &ctx)?;
        serde_json::to_string(&plan_to_wire(&self.inner.plan(&request, &ctx))).map_err(py_msg)
    }
    #[pyo3(signature=(names,columns,specification,*,seed=1,threads=None,memory_limit_bytes=None,cancel=None))]
    #[allow(clippy::too_many_arguments)]
    fn execute(
        &mut self,
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<NumericColumn<'_>>,
        specification: &str,
        seed: u64,
        threads: Option<u32>,
        memory_limit_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<RunPayload> {
        let ctx = bounded_context(seed, threads, memory_limit_bytes, cancel);
        let request = build_request(names, &columns, specification, &ctx)?;
        drop(columns);
        let previous = self.inner.identities().clone();
        let caps = self.capabilities();
        let session = &mut self.inner;
        let ran = detach_catch(py, move || Ok(execute_dr_with_receipt(session, &request, &ctx)))?;
        match ran {
            Ok(outcome) => {
                let counts: BTreeMap<_, _> = outcome
                    .receipt
                    .entries()
                    .iter()
                    .map(|e| (e.stage, counts_wire(&e.counts)))
                    .collect();
                let artifact =
                    RecalcReceiptArtifact::seal(&previous, self.inner.identities(), &caps, &counts)
                        .map_err(py_msg)?;
                if artifact.plan() != &outcome.plan {
                    return Err(py_msg("DR receipt differs from executed plan"));
                }
                let bytes = artifact
                    .to_bytes(&format!("recalc-dr-{}", &artifact.receipt_identity()[..16]))
                    .map_err(py_msg)?;
                let result = serde_json::json!({"plan":plan_to_wire(&outcome.plan),"receipt":artifact.meta(),"law":{"ate":outcome.law.ate,"std_error":outcome.law.std_error},"decision":{"net_benefit":outcome.decision.net_benefit,"treat":outcome.decision.treat}});
                Ok((Some(result.to_string()), Some(bytes), None))
            }
            Err(error) => Ok((None, None, Some(dr_error_refusal(&error)))),
        }
    }
    #[pyo3(signature=(rows,columns,*,seed=1,threads=None,memory_limit_bytes=None))]
    fn predict(
        &self,
        py: Python<'_>,
        rows: &Bound<'_, PyAny>,
        columns: Vec<String>,
        seed: u64,
        threads: Option<u32>,
        memory_limit_bytes: Option<u64>,
    ) -> PyResult<(Option<String>, Option<String>)> {
        let ctx = bounded_context(seed, threads, memory_limit_bytes, None);
        let rows = bounded_prediction_rows(
            rows,
            antecedent::reason::reason_details::recalc::DR_PREDICTION_SCHEMA_MISMATCH,
            &ctx,
        )?;
        if !self.inner.is_live() {
            return Ok((
                None,
                Some(dr_error_refusal(&RecalcRunError::NoLiveState(
                    antecedent::execution::recalc::Stage::ScoreArtifact,
                ))),
            ));
        }
        if self.inner.fitted_effect().is_none() {
            return Ok((
                None,
                Some(dr_error_refusal(&RecalcRunError::Request(
                    RecalcRequestCode::DrPredictorUnavailable,
                ))),
            ));
        }
        if let Some(expected) = self.inner.prediction_columns()
            && expected != columns
        {
            return Ok((
                None,
                Some(refusal_json(
                    antecedent::reason::reason_code!("invalid_argument"),
                    "score_artifact",
                    antecedent::reason::reason_details::recalc::DR_PREDICTION_SCHEMA_MISMATCH,
                    None,
                    None,
                    "prediction feature names/order differ from retained CATE map",
                    None,
                )),
            ));
        }
        let features = self
            .inner
            .fitted_effect()
            .map(|m| m.features.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>())
            .unwrap_or_default();
        let (result, n) = detach_catch(py, move || {
            let result = count_resolved_fits(|| {
                if rows
                    .iter()
                    .any(|r| r.len() != features.len() || r.iter().any(|v| !v.is_finite()))
                {
                    return Err(RecalcRunError::Request(
                        RecalcRequestCode::DrPredictionSchemaMismatch,
                    ));
                }
                let data = (0..features.len())
                    .map(|c| rows.iter().map(|r| r[c]).collect::<Vec<_>>())
                    .collect::<Vec<_>>();
                let refs = data.iter().map(Vec::as_slice).collect::<Vec<_>>();
                self.inner.predict(&features, &refs, rows.len(), &ctx)
            });
            Ok(result)
        })?;
        match result {
            Ok(values) => {
                Ok((Some(serde_json::json!({"values":values,"model_fits":n}).to_string()), None))
            }
            Err(error) => Ok((None, Some(dr_error_refusal(&error)))),
        }
    }
    fn export_scores(&self, py: Python<'_>) -> PyResult<ScoreExportPayload> {
        detach_catch(py, || match self.inner.export_scores() {
            Ok(scores) => {
                let identity = scores.identity().to_owned();
                let bytes =
                    scores.to_bytes(&format!("dr-scores-{}", &identity[..16])).map_err(py_msg)?;
                Ok((Some((identity, bytes)), None))
            }
            Err(error) => Ok((None, Some(dr_error_refusal(&error)))),
        })
    }
    #[pyo3(signature=(*,seed=1,threads=None))]
    fn export_predictor(
        &self,
        py: Python<'_>,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<(Option<Vec<u8>>, Option<String>)> {
        let ctx = context(seed, threads);
        detach_catch(py, move || match self.inner.export_predictor_result(&ctx) {
            Ok(bytes) => Ok((Some(bytes), None)),
            Err(error) => Ok((None, Some(dr_error_refusal(&error)))),
        })
    }

    #[pyo3(signature=(names,columns,*,seed=1,threads=None,memory_limit_bytes=None,cancel=None))]
    #[allow(
        clippy::too_many_arguments,
        reason = "the Python bridge exposes the execution controls"
    )]
    fn export_predictor_with_source(
        &self,
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<NumericColumn<'_>>,
        seed: u64,
        threads: Option<u32>,
        memory_limit_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<(Option<Vec<u8>>, Option<String>)> {
        let ctx = bounded_context(seed, threads, memory_limit_bytes, cancel);
        check_columns(&columns, &ctx)?;
        let data = crate::tabular_from_numpy(&names, &columns)?;
        detach_catch(py, move || {
            match self.inner.export_predictor_result_with_source(&data, &ctx) {
                Ok(bytes) => Ok((Some(bytes), None)),
                Err(error) => Ok((None, Some(dr_error_refusal(&error)))),
            }
        })
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDrSession>()
}
