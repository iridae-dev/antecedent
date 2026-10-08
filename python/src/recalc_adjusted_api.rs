//! Python bridge for checked adjusted-regression selective execution.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent::analysis::recalc_adjusted::{
    AdjustedContrast, AdjustedModel, AdjustedRequest, AdjustedSession,
    execute_adjusted_with_receipt,
};
use antecedent::analysis::recalc_receipt::{RecalcRunError, TargetWeights, UtilitySpec};
use antecedent_core::VariableId;
use antecedent_core::recalc::{RecalcCapabilities, ResumeContext, RetargetSupport};
use antecedent_estimate::adjustment_resume::count_adjusted_model_fits;
use antecedent_estimate::categorical_treatment::{CategoricalTreatmentSpec, LevelScale};
use antecedent_estimate::vector_treatment::VectorCovariance;
use antecedent_io::recalc_receipt_artifact::{
    CapabilitiesWire, DeclaredStageWire, RecalcReceiptArtifact, ResumeWire, identities_from_wire,
    identities_to_wire, plan_to_wire,
};
use antecedent_stats::{GlmFamily, GlmOptions};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use serde::Deserialize;

use crate::recalc_api::{
    RunPayload, artifact_invalid, context, counts_wire, invalid, parse_json, plan_refusal_json,
    refusal_json,
};
use crate::recalc_bounds::{
    MAX_COLUMNS, MAX_ROWS, MAX_VALUES, bounded_prediction_rows, check_columns,
};
use crate::{detach_catch, py_msg};

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum ModelWire {
    Linear {
        covariance: String,
    },
    Glm {
        family: String,
        max_iter: u32,
        tolerance: f64,
    },
    Categorical {
        labels: Vec<String>,
        levels: Vec<String>,
        reference: String,
        ordered: bool,
        min_level_rows: usize,
        covariance: String,
    },
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum ContrastWire {
    Numeric { active: Vec<f64>, control: Vec<f64> },
    Categorical { from: String, to: String },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RequestWire {
    edges: Vec<(String, String)>,
    treatments: Vec<String>,
    outcome: String,
    adjustment: Vec<String>,
    model: ModelWire,
    contrast: ContrastWire,
    benefit_per_unit: f64,
    cost: f64,
    target_weights: Option<Vec<f64>>,
    target_depends_on: Vec<String>,
}

const MAX_GLM_ITERATIONS: u32 = 1_000;

fn covariance(name: &str) -> PyResult<VectorCovariance> {
    Ok(match name {
        "model_based" => VectorCovariance::ModelBased,
        "hc0" => VectorCovariance::Hc0,
        "hc1" => VectorCovariance::Hc1,
        "hc2" => VectorCovariance::Hc2,
        "hc3" => VectorCovariance::Hc3,
        _ => return Err(invalid("recalc.adjusted_invalid_model", "unknown covariance")),
    })
}

fn build_request(
    names: Vec<String>,
    columns: &[PyReadonlyArray1<'_, f64>],
    specification: &str,
) -> PyResult<AdjustedRequest> {
    let wire: RequestWire = parse_json(specification, "adjusted request")?;
    check_columns(&names, columns)?;
    if names.is_empty() || names.len() != columns.len() {
        return Err(invalid("recalc.invalid_data", "one named column per name is required"));
    }
    for (i, name) in names.iter().enumerate() {
        if name.trim().is_empty() || names[..i].contains(name) {
            return Err(invalid("recalc.invalid_data", "column names must be unique and nonempty"));
        }
    }
    let position =
        |name: &str| {
            names.iter().position(|n| n == name).and_then(|p| u32::try_from(p).ok()).ok_or_else(
                || invalid("recalc.unknown_variable", format!("no column named `{name}`")),
            )
        };
    let edges = wire
        .edges
        .iter()
        .map(|(a, b)| Ok((position(a)?, position(b)?)))
        .collect::<PyResult<Vec<_>>>()?;
    let treatments = wire.treatments.iter().map(|n| position(n)).collect::<PyResult<Vec<_>>>()?;
    let outcome = position(&wire.outcome)?;
    let adjustment = wire.adjustment.iter().map(|n| position(n)).collect::<PyResult<Vec<_>>>()?;
    if wire.target_weights.is_none() && !wire.target_depends_on.is_empty() {
        return Err(invalid("recalc.invalid_request", "weight dependencies need weights"));
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
    let model = match wire.model {
        ModelWire::Linear { covariance: name } => {
            AdjustedModel::Linear { covariance: covariance(&name)? }
        }
        ModelWire::Glm { family, max_iter, tolerance } => {
            let family = match family.as_str() {
                "binomial_logit" => GlmFamily::BinomialLogit,
                "binomial_probit" => GlmFamily::BinomialProbit,
                "gaussian_identity" => GlmFamily::GaussianIdentity,
                "poisson_log" => GlmFamily::PoissonLog,
                "negative_binomial" => GlmFamily::NegativeBinomial,
                _ => return Err(invalid("recalc.adjusted_invalid_model", "unknown GLM family")),
            };
            if max_iter == 0
                || max_iter > MAX_GLM_ITERATIONS
                || !tolerance.is_finite()
                || tolerance <= 0.0
            {
                return Err(invalid("recalc.adjusted_invalid_model", "invalid GLM fit options"));
            }
            AdjustedModel::Glm { family, options: GlmOptions::new(max_iter, tolerance) }
        }
        ModelWire::Categorical {
            labels,
            levels,
            reference,
            ordered,
            min_level_rows,
            covariance: name,
        } => AdjustedModel::Categorical {
            spec: CategoricalTreatmentSpec {
                declared_levels: levels,
                scale: if ordered { LevelScale::Ordered } else { LevelScale::Unordered },
                reference,
                min_level_rows,
                pairwise: Vec::new(),
                monotonicity: None,
                covariance: covariance(&name)?,
            },
            levels: labels,
        },
    };
    let contrast = match wire.contrast {
        ContrastWire::Numeric { active, control } => AdjustedContrast::Numeric { active, control },
        ContrastWire::Categorical { from, to } => AdjustedContrast::Categorical { from, to },
    };
    Ok(AdjustedRequest {
        columns: names.into_iter().zip(columns.iter().map(|c| c.as_array().to_vec())).collect(),
        edges,
        treatments,
        outcome,
        adjustment,
        model,
        contrast,
        target,
        utility: UtilitySpec { benefit_per_unit: wire.benefit_per_unit, cost: wire.cost },
    })
}

fn adjusted_error_refusal(error: &RecalcRunError) -> String {
    match error {
        RecalcRunError::Refused(plan) => plan_refusal_json(plan),
        RecalcRunError::Request(detail) => {
            let (code, stage) = match *detail {
                "recalc.adjustment_not_identified" => {
                    (antecedent_core::reason_code!("effect_not_identified"), "identification")
                }
                "recalc.invalid_graph" | "recalc.invalid_adjustment_set" => {
                    (antecedent_core::reason_code!("invalid_argument"), "identification")
                }
                "recalc.categorical_role_mismatch" => {
                    (antecedent_core::reason_code!("invalid_argument"), "query")
                }
                "recalc.cancelled" => {
                    (antecedent_core::reason_code!("cancelled_no_claim"), "score_artifact")
                }
                "recalc.memory_budget_exceeded" => {
                    (antecedent_core::reason_code!("invalid_argument"), "score_artifact")
                }
                "recalc.invalid_target_weights" => {
                    (antecedent_core::reason_code!("invalid_argument"), "target_population")
                }
                "recalc.invalid_data" => {
                    (antecedent_core::reason_code!("invalid_argument"), "data_snapshot")
                }
                "recalc.contrast_out_of_support" | "recalc.adjusted_prediction_out_of_support" => {
                    (antecedent_core::reason_code!("route_not_supported"), "treatment_grid")
                }
                "recalc.invalid_categorical_contrast"
                | "recalc.invalid_contrast"
                | "recalc.incompatible_contrast" => {
                    (antecedent_core::reason_code!("invalid_argument"), "law")
                }
                _ => (antecedent_core::reason_code!("invalid_argument"), "score_artifact"),
            };
            refusal_json(code, stage, detail, None, None, &error.to_string(), None)
        }
        RecalcRunError::NoLiveState(stage) => refusal_json(
            antecedent_core::reason_code!("score_table_unavailable"),
            &stage.label(),
            "recalc.no_live_state",
            Some(&stage.label()),
            None,
            "no retained adjusted fit",
            None,
        ),
        RecalcRunError::Execution(cause) => {
            use antecedent::CausalError;
            use antecedent_estimate::EstimationError;
            use antecedent_stats::{GlmRefusalKind, StatsError};
            let (code, detail, stage) = match cause.peeled() {
                CausalError::Estimate(EstimationError::Data(_) | EstimationError::Query(_)) => (
                    antecedent_core::reason_code!("invalid_argument"),
                    "recalc.adjusted_model_data_invalid",
                    "data_snapshot",
                ),
                CausalError::Estimate(
                    EstimationError::Refused { code, .. }
                    | EstimationError::RefusedWithFields { code, .. },
                ) if *code == "arm_not_populated" => (
                    antecedent_core::reason_code!("arm_not_populated"),
                    "recalc.adjusted_arm_not_populated",
                    "score_artifact",
                ),
                CausalError::Estimate(
                    EstimationError::Refused { code, .. }
                    | EstimationError::RefusedWithFields { code, .. },
                ) if *code == "invalid_argument" => (
                    antecedent_core::reason_code!("invalid_argument"),
                    "recalc.adjusted_model_data_invalid",
                    "data_snapshot",
                ),
                CausalError::Estimate(
                    EstimationError::Refused { code, .. }
                    | EstimationError::RefusedWithFields { code, .. },
                ) if *code == "design_rank_deficient" => (
                    antecedent_core::reason_code!("design_rank_deficient"),
                    "recalc.adjusted_design_rank_deficient",
                    "score_artifact",
                ),
                CausalError::Estimate(EstimationError::Stats(StatsError::GlmRefused {
                    kind,
                    ..
                })) => (
                    antecedent_core::reason_code!("route_not_supported"),
                    match kind {
                        GlmRefusalKind::NonConverged => "recalc.adjusted_glm_not_converged",
                        GlmRefusalKind::Separated => "recalc.adjusted_glm_separated",
                        GlmRefusalKind::BoundarySaturated => {
                            "recalc.adjusted_glm_boundary_saturated"
                        }
                    },
                    "score_artifact",
                ),
                CausalError::Estimate(EstimationError::Stats(StatsError::RankDeficient {
                    ..
                })) => (
                    antecedent_core::reason_code!("design_rank_deficient"),
                    "recalc.adjusted_design_rank_deficient",
                    "score_artifact",
                ),
                _ => (
                    antecedent_core::reason_code!("route_not_supported"),
                    "recalc.adjusted_fit_refused",
                    "score_artifact",
                ),
            };
            refusal_json(code, stage, detail, None, None, &cause.to_string(), None)
        }
        RecalcRunError::Receipt(_) => refusal_json(
            antecedent_core::reason_code!("invalid_argument"),
            "score_artifact",
            "recalc.adjusted_receipt_invalid",
            None,
            None,
            &error.to_string(),
            None,
        ),
    }
}

/// Native retained fit; declarations never supply executable model state.
#[pyclass(name = "AdjustedSessionHandle")]
pub struct PyAdjustedSession {
    inner: AdjustedSession,
}

impl PyAdjustedSession {
    fn capabilities(&self) -> RecalcCapabilities {
        RecalcCapabilities {
            retarget: RetargetSupport::Licensed,
            request: antecedent_core::recalc::RequestSupport::OnGrid,
            boundary: self.inner.boundary(),
        }
    }
}

#[pymethods]
impl PyAdjustedSession {
    #[new]
    fn new() -> Self {
        Self { inner: AdjustedSession::new() }
    }

    #[staticmethod]
    fn resume(previous_json: &str, resume_json: &str) -> PyResult<Self> {
        let previous: Vec<DeclaredStageWire> = parse_json(previous_json, "previous")?;
        let wire: ResumeWire = parse_json(resume_json, "resume")?;
        let previous = identities_from_wire(&previous).map_err(|e| artifact_invalid(&e))?;
        let resume = ResumeContext {
            portable_fit: wire.portable_fit,
            portable_scores: wire.portable_scores,
            supplied_data: wire.supplied_data,
            supplied_provider: wire.supplied_provider,
            scores_snapshot_bound: wire.scores_snapshot_bound,
        };
        Ok(Self { inner: AdjustedSession::resume(previous, resume) })
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
        self.inner.prediction_columns().map(<[String]>::to_vec)
    }

    fn predict(
        &self,
        py: Python<'_>,
        rows: &Bound<'_, PyAny>,
        columns: Vec<String>,
    ) -> PyResult<(Option<String>, Option<String>)> {
        let rows = bounded_prediction_rows(rows, "recalc.adjusted_prediction_schema_mismatch")?;
        if rows.len() > MAX_ROWS
            || columns.len() > MAX_COLUMNS
            || rows
                .iter()
                .try_fold(0usize, |n, row| n.checked_add(row.len()))
                .is_none_or(|n| n > MAX_VALUES)
        {
            return Err(invalid(
                "recalc.limits_exceeded",
                "adjusted predictions exceed row/column/value limits",
            ));
        }
        if let Some(expected) = self.inner.prediction_columns() {
            if expected != columns {
                return Ok((
                    None,
                    Some(refusal_json(
                        antecedent_core::reason_code!("invalid_argument"),
                        "score_artifact",
                        "recalc.adjusted_prediction_schema_mismatch",
                        None,
                        None,
                        "prediction feature names/order differ from the retained model",
                        None,
                    )),
                ));
            }
        }
        let (result, model_fits) =
            detach_catch(py, move || Ok(count_adjusted_model_fits(|| self.inner.predict(&rows))))?;
        match result {
            Ok(values) => Ok((
                Some(serde_json::json!({"values":values,"model_fits":model_fits}).to_string()),
                None,
            )),
            Err(error) => Ok((None, Some(adjusted_error_refusal(&error)))),
        }
    }

    #[pyo3(signature=(names,columns,specification,*,seed=1,threads=None))]
    fn plan(
        &self,
        names: Vec<String>,
        columns: Vec<PyReadonlyArray1<'_, f64>>,
        specification: &str,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<String> {
        let request = build_request(names, &columns, specification)?;
        serde_json::to_string(&plan_to_wire(&self.inner.plan(&request, &context(seed, threads))))
            .map_err(py_msg)
    }

    #[pyo3(signature=(names,columns,specification,*,seed=1,threads=None))]
    fn execute(
        &mut self,
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<PyReadonlyArray1<'_, f64>>,
        specification: &str,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<RunPayload> {
        let request = build_request(names, &columns, specification)?;
        drop(columns);
        let ctx = context(seed, threads);
        let previous = self.inner.identities().clone();
        let capabilities = self.capabilities();
        let planned = self.inner.plan(&request, &ctx);
        let mut session = std::mem::take(&mut self.inner);
        let (session, ran) = detach_catch(py, move || {
            let ran = execute_adjusted_with_receipt(&mut session, &request, &ctx);
            Ok((session, ran))
        })?;
        self.inner = session;
        match ran {
            Ok(outcome) => {
                let counts: BTreeMap<_, _> = outcome
                    .receipt
                    .entries()
                    .iter()
                    .map(|e| (e.stage, counts_wire(&e.counts)))
                    .collect();
                let artifact = RecalcReceiptArtifact::seal(
                    &previous,
                    self.inner.identities(),
                    &capabilities,
                    &counts,
                )
                .map_err(py_msg)?;
                if artifact.plan() != &outcome.plan {
                    return Err(py_msg("adjusted receipt differs from executed plan"));
                }
                let bytes = artifact
                    .to_bytes(&format!("recalc-adjusted-{}", &artifact.receipt_identity()[..16]))
                    .map_err(py_msg)?;
                let result = serde_json::json!({
                    "plan":plan_to_wire(&outcome.plan),"receipt":artifact.meta(),
                    "law":{"ate":outcome.law.ate,"std_error":outcome.law.std_error},
                    "decision":{"net_benefit":outcome.decision.net_benefit,"treat":outcome.decision.treat},
                });
                Ok((Some(result.to_string()), Some(bytes), None))
            }
            Err(RecalcRunError::Refused(plan)) => Ok((None, None, Some(plan_refusal_json(&plan)))),
            Err(RecalcRunError::NoLiveState(stage)) => Ok((
                None,
                None,
                Some(refusal_json(
                    antecedent_core::reason_code!("score_table_unavailable"),
                    &stage.label(),
                    "recalc.no_live_state",
                    Some(&stage.label()),
                    Some("supply raw data and refit in this process".to_owned()),
                    "no retained adjusted fit",
                    Some(&planned),
                )),
            )),
            Err(error @ (RecalcRunError::Request(_) | RecalcRunError::Execution(_))) => {
                Ok((None, None, Some(adjusted_error_refusal(&error))))
            }
            Err(RecalcRunError::Receipt(error)) => Err(py_msg(error)),
        }
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAdjustedSession>()
}
