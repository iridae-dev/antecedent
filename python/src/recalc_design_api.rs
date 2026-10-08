//! Checked public adapter for IV, sharp RD and linear front-door recalculation.
//! SPDX-License-Identifier: MIT OR Apache-2.0
use std::collections::BTreeMap;
use std::fmt::Write;

use antecedent::analysis::recalc_design::{
    DesignModel, DesignRequest, DesignRunError, DesignSession, consume_design_with_data,
    execute_design_with_receipt,
};
use antecedent::analysis::recalc_receipt::{RecalcOutcome, RecalcRunError, UtilitySpec};
use antecedent_core::recalc::{
    RecalcCapabilities, RequestSupport, ResumeContext, RetargetSupport, StageIdentities,
};
use antecedent_io::recalc_receipt_artifact::{
    CapabilitiesWire, DeclaredStageWire, RecalcReceiptArtifact, ResumeWire, identities_from_wire,
    identities_to_wire, plan_to_wire,
};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use serde::Deserialize;

use crate::recalc_api::{
    RunPayload, artifact_invalid, context, counts_wire, invalid, parse_json, plan_refusal_json,
    refusal_json,
};
use crate::recalc_bounds::check_columns;
use crate::{detach_catch, py_msg};

#[derive(Deserialize)]
#[serde(tag = "kind", deny_unknown_fields)]
enum ModelWire {
    #[serde(rename = "iv")]
    Iv { instrument: String },
    #[serde(rename = "rd")]
    Rd { running_variable: String, cutoff: f64, bandwidth: f64 },
    #[serde(rename = "frontdoor")]
    Frontdoor { mediator: String },
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RequestWire {
    edges: Vec<(String, String)>,
    treatment: String,
    outcome: String,
    model: ModelWire,
    benefit_per_unit: f64,
    cost: f64,
}
fn request(
    names: Vec<String>,
    columns: &[PyReadonlyArray1<'_, f64>],
    text: &str,
) -> PyResult<DesignRequest> {
    check_columns(&names, columns)?;
    let wire: RequestWire = parse_json(text, "design request")?;
    if names.is_empty()
        || names.len() != columns.len()
        || names.iter().enumerate().any(|(i, n)| n.trim().is_empty() || names[..i].contains(n))
    {
        return Err(invalid("recalc.invalid_data", "unique named columns required"));
    }
    let position =
        |name: &str| -> PyResult<u32> {
            names.iter().position(|n| n == name).and_then(|i| u32::try_from(i).ok()).ok_or_else(
                || invalid("recalc.unknown_variable", format!("no column named `{name}`")),
            )
        };
    let model = match wire.model {
        ModelWire::Iv { instrument } => DesignModel::Iv2Sls { instrument: position(&instrument)? },
        ModelWire::Rd { running_variable, cutoff, bandwidth } => {
            DesignModel::Rd { running_variable: position(&running_variable)?, cutoff, bandwidth }
        }
        ModelWire::Frontdoor { mediator } => {
            DesignModel::Frontdoor { mediator: position(&mediator)? }
        }
    };
    let treatment = position(&wire.treatment)?;
    let outcome = position(&wire.outcome)?;
    let edges = wire
        .edges
        .iter()
        .map(|(a, b)| Ok((position(a)?, position(b)?)))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(DesignRequest {
        columns: names.into_iter().zip(columns.iter().map(|c| c.as_array().to_vec())).collect(),
        edges,
        treatment,
        outcome,
        model,
        utility: UtilitySpec { benefit_per_unit: wire.benefit_per_unit, cost: wire.cost },
    })
}
fn diagnostic_number(value: f64) -> serde_json::Value {
    if value.is_finite() {
        serde_json::json!(value)
    } else if value.is_nan() {
        serde_json::json!("unavailable")
    } else if value.is_sign_positive() {
        serde_json::json!("positive_infinity")
    } else {
        serde_json::json!("negative_infinity")
    }
}
#[allow(clippy::too_many_lines)]
fn design_error(error: &DesignRunError) -> String {
    match error {
        DesignRunError::WeakInstrument { point, model_fits, diagnostics } => {
            let json = refusal_json(
                antecedent_core::reason_code!("route_not_supported"),
                "decision",
                "recalc.iv_decision_unavailable",
                None,
                None,
                &error.to_string(),
                None,
            );
            let mut wire: serde_json::Value =
                serde_json::from_str(&json).expect("internal refusal JSON");
            wire["fields"] = serde_json::json!({"point":point,"model_fits":model_fits,"diagnostics":{
                "f_statistic":diagnostic_number(diagnostics.f_statistic),"df1":diagnostics.df1,"df2":diagnostics.df2,"partial_r2":diagnostic_number(diagnostics.partial_r2),
                "anderson_rubin":diagnostics.anderson_rubin.map(|(lower,upper,level)|serde_json::json!({"lower":diagnostic_number(lower),"upper":diagnostic_number(upper),"nominal_level":level})),
                "uncertainty_withheld":diagnostics.uncertainty_withheld,
            }});
            wire.to_string()
        }
        DesignRunError::Recalc(RecalcRunError::Refused(plan)) => plan_refusal_json(plan),
        DesignRunError::Recalc(RecalcRunError::NoLiveState(stage)) => refusal_json(
            antecedent_core::reason_code!("score_table_unavailable"),
            &stage.label(),
            "recalc.no_live_state",
            Some(&stage.label()),
            None,
            &error.to_string(),
            None,
        ),
        DesignRunError::Recalc(RecalcRunError::Request(detail)) => {
            let (code, stage) = match *detail {
                "recalc.cancelled" => {
                    (antecedent_core::reason_code!("cancelled_no_claim"), "score_artifact")
                }
                "recalc.invalid_data" => {
                    (antecedent_core::reason_code!("invalid_argument"), "data_snapshot")
                }
                "recalc.invalid_graph" => {
                    (antecedent_core::reason_code!("invalid_argument"), "graph")
                }
                "recalc.invalid_design_roles" | "recalc.invalid_rd_window" => {
                    (antecedent_core::reason_code!("invalid_argument"), "query")
                }
                "recalc.design_not_checked" => {
                    (antecedent_core::reason_code!("effect_not_identified"), "identification")
                }
                "recalc.design_data_unavailable" => {
                    (antecedent_core::reason_code!("score_table_unavailable"), "data_snapshot")
                }
                "recalc.design_artifact_unverified" => {
                    (antecedent_core::reason_code!("route_not_supported"), "score_artifact")
                }
                "recalc.design_artifact_mismatch"
                | "recalc.invalid_design_artifact"
                | "recalc.design_point_unavailable"
                | "recalc.iv_diagnostics_unavailable"
                | "recalc.design_effect_unavailable"
                | "recalc.memory_budget_exceeded" => {
                    (antecedent_core::reason_code!("invalid_argument"), "score_artifact")
                }
                _ => (antecedent_core::reason_code!("invalid_argument"), "score_artifact"),
            };
            refusal_json(code, stage, detail, None, None, &error.to_string(), None)
        }
        DesignRunError::Recalc(RecalcRunError::Execution(cause)) => {
            use antecedent::CausalError;
            use antecedent_estimate::EstimationError;
            use antecedent_stats::StatsError;
            let (code, detail, stage) = match cause.peeled() {
                CausalError::NotIdentified { .. } => (
                    antecedent_core::reason_code!("effect_not_identified"),
                    "recalc.design_not_identified",
                    "identification",
                ),
                CausalError::Estimate(EstimationError::Data(_) | EstimationError::Query(_)) => (
                    antecedent_core::reason_code!("invalid_argument"),
                    "recalc.design_model_data_invalid",
                    "data_snapshot",
                ),
                CausalError::Estimate(EstimationError::Stats(StatsError::RankDeficient {
                    ..
                })) => (
                    antecedent_core::reason_code!("design_rank_deficient"),
                    "recalc.design_rank_deficient",
                    "score_artifact",
                ),
                CausalError::Estimate(
                    EstimationError::Refused { code, .. }
                    | EstimationError::RefusedWithFields { code, .. },
                ) if *code == "cancelled_no_claim" => (
                    antecedent_core::reason_code!("cancelled_no_claim"),
                    "recalc.cancelled",
                    "score_artifact",
                ),
                CausalError::Estimate(
                    EstimationError::Refused { code, .. }
                    | EstimationError::RefusedWithFields { code, .. },
                ) if *code == "invalid_argument" => (
                    antecedent_core::reason_code!("invalid_argument"),
                    "recalc.design_model_data_invalid",
                    "data_snapshot",
                ),
                _ => (
                    antecedent_core::reason_code!("route_not_supported"),
                    "recalc.design_fit_refused",
                    "score_artifact",
                ),
            };
            refusal_json(code, stage, detail, None, None, &cause.to_string(), None)
        }
        DesignRunError::Recalc(RecalcRunError::Receipt(_)) => refusal_json(
            antecedent_core::reason_code!("invalid_argument"),
            "score_artifact",
            "recalc.design_receipt_invalid",
            None,
            None,
            &error.to_string(),
            None,
        ),
    }
}
fn payload(
    previous: &StageIdentities,
    session: &DesignSession,
    caps: &RecalcCapabilities,
    outcome: &RecalcOutcome,
) -> PyResult<RunPayload> {
    let counts = outcome
        .receipt
        .entries()
        .iter()
        .map(|e| (e.stage, counts_wire(&e.counts)))
        .collect::<BTreeMap<_, _>>();
    let artifact = RecalcReceiptArtifact::seal(previous, session.identities(), caps, &counts)
        .map_err(py_msg)?;
    if artifact.plan() != &outcome.plan {
        return Err(py_msg("design receipt differs from executed plan"));
    }
    let bytes = artifact
        .to_bytes(&format!("recalc-design-{}", &artifact.receipt_identity()[..16]))
        .map_err(py_msg)?;
    let result = serde_json::json!({"plan":plan_to_wire(&outcome.plan),"receipt":artifact.meta(),"law":{"ate":outcome.law.ate,"std_error":outcome.law.std_error},"decision":{"net_benefit":outcome.decision.net_benefit,"treat":outcome.decision.treat}});
    Ok((Some(result.to_string()), Some(bytes), None))
}
type ReplayPayload =
    (Option<PyDesignSession>, Option<String>, Option<Vec<u8>>, Option<String>, Option<String>);

#[pyclass(name = "DesignSessionHandle")]
struct PyDesignSession {
    inner: DesignSession,
}
impl PyDesignSession {
    fn capabilities(&self) -> RecalcCapabilities {
        RecalcCapabilities {
            boundary: self.inner.boundary(),
            request: RequestSupport::OnGrid,
            retarget: RetargetSupport::NotDeclared,
        }
    }
}
#[pymethods]
impl PyDesignSession {
    #[new]
    fn new() -> Self {
        Self { inner: DesignSession::new() }
    }
    #[staticmethod]
    fn resume(previous_json: &str, resume_json: &str) -> PyResult<Self> {
        let previous: Vec<DeclaredStageWire> = parse_json(previous_json, "previous")?;
        let ids = identities_from_wire(&previous).map_err(|e| artifact_invalid(&e))?;
        let wire: ResumeWire = parse_json(resume_json, "resume")?;
        let context = ResumeContext {
            portable_fit: wire.portable_fit,
            portable_scores: wire.portable_scores,
            supplied_data: wire.supplied_data,
            supplied_provider: wire.supplied_provider,
            scores_snapshot_bound: wire.scores_snapshot_bound,
        };
        Ok(Self { inner: DesignSession::resume(ids, context) })
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
    #[pyo3(signature=(names,columns,specification,*,seed=1,threads=None))]
    fn plan(
        &self,
        names: Vec<String>,
        columns: Vec<PyReadonlyArray1<'_, f64>>,
        specification: &str,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<String> {
        let request = request(names, &columns, specification)?;
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
        let request = request(names, &columns, specification)?;
        drop(columns);
        let ctx = context(seed, threads);
        let previous = self.inner.identities().clone();
        let caps = self.capabilities();
        let mut session = std::mem::take(&mut self.inner);
        let (session, ran) = detach_catch(py, move || {
            let ran = execute_design_with_receipt(&mut session, &request, &ctx);
            Ok((session, ran))
        })?;
        self.inner = session;
        match ran {
            Ok(outcome) => payload(&previous, &self.inner, &caps, &outcome),
            Err(error) => Ok((None, None, Some(design_error(&error)))),
        }
    }
    #[staticmethod]
    #[pyo3(signature=(artifact,names=None,columns=None,specification=None,*,seed=1,threads=None))]
    #[allow(clippy::too_many_arguments)]
    fn consume(
        py: Python<'_>,
        artifact: &[u8],
        names: Option<Vec<String>>,
        columns: Option<Vec<PyReadonlyArray1<'_, f64>>>,
        specification: Option<&str>,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<ReplayPayload> {
        if artifact.len() > 16 * 1024 * 1024 {
            return Err(invalid("recalc.limits_exceeded", "design artifact exceeds 16MiB"));
        }
        let request = match (names, columns.as_ref(), specification) {
            (Some(names), Some(columns), Some(text)) => Some(request(names, columns, text)?),
            (None, None, None) => None,
            _ => {
                return Err(invalid(
                    "recalc.invalid_data",
                    "complete supplied design data required",
                ));
            }
        };
        drop(columns);
        let artifact = artifact.to_vec();
        let ctx = context(seed, threads);
        let replay = detach_catch(py, move || {
            Ok(consume_design_with_data(&artifact, request.as_ref(), &ctx))
        })?;
        match replay {
            Ok(replay) => {
                let session = Self { inner: replay.session };
                let result = payload(
                    &StageIdentities::default(),
                    &session.inner,
                    &session.capabilities(),
                    &replay.outcome,
                )?;
                let digest = replay.artifact_digest.iter().fold(
                    String::with_capacity(64),
                    |mut text, byte| {
                        write!(text, "{byte:02x}").expect("String formatting cannot fail");
                        text
                    },
                );
                Ok((Some(session), result.0, result.1, Some(digest), None))
            }
            Err(error) => Ok((None, None, None, None, Some(design_error(&error)))),
        }
    }
    fn export_result(&self, py: Python<'_>) -> PyResult<(Option<Vec<u8>>, Option<String>)> {
        detach_catch(py, || match self.inner.export_result() {
            Ok(bytes) => Ok((Some(bytes), None)),
            Err(error) => Ok((None, Some(design_error(&error)))),
        })
    }
}
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyDesignSession>()
}
