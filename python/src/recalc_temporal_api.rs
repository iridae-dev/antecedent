//! Checked empirical two-step history adapter; dependent intervals remain closed.
//! SPDX-License-Identifier: MIT OR Apache-2.0
use crate::recalc_api::{RunPayload, context, invalid, parse_json, refusal_json};
use crate::recalc_family_api::{caps, payload, resume, run_error};
use antecedent::analysis::recalc_receipt::RecalcRunError;
use antecedent::analysis::recalc_temporal::{
    TemporalOutcome, TemporalRequest, TemporalRunError, TemporalSession,
    consume_temporal_recalc_artifact, execute_temporal_with_receipt,
};
use antecedent_io::recalc_receipt_artifact::{CapabilitiesWire, identities_to_wire, plan_to_wire};
use pyo3::prelude::*;

#[cfg(feature = "calibration-internal")]
mod candidate;

fn request(text: &str) -> PyResult<TemporalRequest> {
    let request: TemporalRequest = parse_json(text, "temporal histories")?;
    if request.units.len() > 4096
        || request
            .units
            .iter()
            .try_fold(0usize, |n, u| n.checked_add(u.histories.len()))
            .is_none_or(|n| n > 100_000)
    {
        return Err(invalid("recalc.limits_exceeded", "temporal units/history rows exceed bounds"));
    }
    Ok(request)
}
fn error(error: &TemporalRunError) -> String {
    match error {
        TemporalRunError::Recalc(RecalcRunError::Execution(cause))
            if matches!(cause.peeled(), antecedent::CausalError::Estimate(
                antecedent_estimate::EstimationError::Refused {code,..}
                | antecedent_estimate::EstimationError::RefusedWithFields {code,..}
            ) if *code == "cell_not_licensed") =>
        {
            refusal_json(
                antecedent_core::reason_code!("cell_not_licensed"),
                "inference",
                "temporal_interval.route_frozen",
                None,
                None,
                &error.to_string(),
                None,
            )
        }
        TemporalRunError::Io(antecedent_io::IoError::Refused { code, .. })
            if *code == "cell_not_licensed" =>
        {
            refusal_json(
                antecedent_core::reason_code!("cell_not_licensed"),
                "inference",
                "temporal_interval.route_frozen",
                None,
                None,
                &error.to_string(),
                None,
            )
        }
        TemporalRunError::Recalc(error) => run_error(error, "recalc.temporal_fit_refused"),
        TemporalRunError::Io(antecedent_io::IoError::Refused { code, .. })
            if *code == "route_not_supported" =>
        {
            refusal_json(
                antecedent_core::reason_code!("route_not_supported"),
                "identification",
                "recalc.temporal_scope_unsupported",
                None,
                None,
                &error.to_string(),
                None,
            )
        }
        TemporalRunError::Io(antecedent_io::IoError::Refused { code, .. })
            if *code == "invalid_argument" =>
        {
            refusal_json(
                antecedent_core::reason_code!("invalid_argument"),
                "score_artifact",
                "recalc.temporal_artifact_invalid",
                None,
                None,
                &error.to_string(),
                None,
            )
        }
        TemporalRunError::Io(antecedent_io::IoError::Refused { code, .. })
            if *code == "transport_missing_evidence" =>
        {
            refusal_json(
                antecedent_core::reason_code!("transport_missing_evidence"),
                "identification",
                "recalc.temporal_missing_evidence",
                None,
                None,
                &error.to_string(),
                None,
            )
        }
        TemporalRunError::Io(antecedent_io::IoError::Refused { code, .. })
            if *code == "transport_budget_cancel" =>
        {
            refusal_json(
                antecedent_core::reason_code!("transport_budget_cancel"),
                "identification",
                "recalc.temporal_budget_cancel",
                None,
                None,
                &error.to_string(),
                None,
            )
        }
        TemporalRunError::Io(antecedent_io::IoError::Refused { code, .. })
            if *code == "transport_support_failure" =>
        {
            refusal_json(
                antecedent_core::reason_code!("transport_support_failure"),
                "law",
                "recalc.temporal_support_failure",
                None,
                None,
                &error.to_string(),
                None,
            )
        }
        TemporalRunError::Io(antecedent_io::IoError::Refused { code, .. })
            if *code == "transport_missing_provider" =>
        {
            refusal_json(
                antecedent_core::reason_code!("transport_missing_provider"),
                "score_artifact",
                "recalc.temporal_missing_provider",
                None,
                None,
                &error.to_string(),
                None,
            )
        }
        TemporalRunError::Io(antecedent_io::IoError::Refused { code, .. })
            if *code == "transport_unsupported_evaluator" =>
        {
            refusal_json(
                antecedent_core::reason_code!("transport_unsupported_evaluator"),
                "score_artifact",
                "recalc.temporal_evaluator_unsupported",
                None,
                None,
                &error.to_string(),
                None,
            )
        }
        TemporalRunError::Io(antecedent_io::IoError::Refused { code, .. })
            if *code == "transport_numerical_failure" =>
        {
            refusal_json(
                antecedent_core::reason_code!("transport_numerical_failure"),
                "score_artifact",
                "recalc.temporal_numerical_failure",
                None,
                None,
                &error.to_string(),
                None,
            )
        }
        TemporalRunError::Estimation(
            antecedent_estimate::EstimationError::Refused { code, .. }
            | antecedent_estimate::EstimationError::RefusedWithFields { code, .. },
        ) if *code == "route_not_supported" => refusal_json(
            antecedent_core::reason_code!("route_not_supported"),
            "data_snapshot",
            "recalc.temporal_history_unsupported",
            None,
            None,
            &error.to_string(),
            None,
        ),
        TemporalRunError::Estimation(
            antecedent_estimate::EstimationError::Refused { code, .. }
            | antecedent_estimate::EstimationError::RefusedWithFields { code, .. },
        ) if *code == "transport_support_failure" => refusal_json(
            antecedent_core::reason_code!("transport_support_failure"),
            "law",
            "recalc.temporal_support_failure",
            None,
            None,
            &error.to_string(),
            None,
        ),
        TemporalRunError::Estimation(
            antecedent_estimate::EstimationError::Refused { code, .. }
            | antecedent_estimate::EstimationError::RefusedWithFields { code, .. },
        ) if *code == "invalid_argument" => refusal_json(
            antecedent_core::reason_code!("invalid_argument"),
            "data_snapshot",
            "recalc.temporal_data_invalid",
            None,
            None,
            &error.to_string(),
            None,
        ),
        TemporalRunError::Estimation(
            antecedent_estimate::EstimationError::Refused { code, .. }
            | antecedent_estimate::EstimationError::RefusedWithFields { code, .. },
        ) if *code == "cancelled_no_claim" => refusal_json(
            antecedent_core::reason_code!("cancelled_no_claim"),
            "score_artifact",
            "recalc.cancelled",
            None,
            None,
            &error.to_string(),
            None,
        ),
        TemporalRunError::Estimation(
            antecedent_estimate::EstimationError::Refused { code, .. }
            | antecedent_estimate::EstimationError::RefusedWithFields { code, .. },
        ) if *code == "cell_not_licensed" => refusal_json(
            antecedent_core::reason_code!("cell_not_licensed"),
            "inference",
            "temporal_interval.route_frozen",
            None,
            None,
            &error.to_string(),
            None,
        ),
        TemporalRunError::Estimation(
            antecedent_estimate::EstimationError::Data(_)
            | antecedent_estimate::EstimationError::Query(_),
        ) => refusal_json(
            antecedent_core::reason_code!("invalid_argument"),
            "data_snapshot",
            "recalc.temporal_data_invalid",
            None,
            None,
            &error.to_string(),
            None,
        ),
        _ => refusal_json(
            antecedent_core::reason_code!("route_not_supported"),
            "score_artifact",
            "recalc.temporal_fit_refused",
            None,
            None,
            &error.to_string(),
            None,
        ),
    }
}
fn outcome(
    previous: &antecedent_core::recalc::StageIdentities,
    session: &TemporalSession,
    capabilities: &antecedent_core::recalc::RecalcCapabilities,
    out: &TemporalOutcome,
) -> PyResult<RunPayload> {
    payload(
        previous,
        session.identities(),
        capabilities,
        &out.recalc.plan,
        &out.recalc.receipt,
        serde_json::json!({
        "functional":out.functional,"means":out.means,"law":{"ate":out.recalc.law.ate,"std_error":out.recalc.law.std_error},
        "decision":{"net_benefit":out.recalc.decision.net_benefit,"treat":out.recalc.decision.treat}}),
    )
}
type ReplayPayload = (Option<PyTemporalSession>, Option<String>, Option<Vec<u8>>, Option<String>);
#[pyclass(name = "TemporalSessionHandle")]
struct PyTemporalSession {
    inner: TemporalSession,
}
#[pymethods]
impl PyTemporalSession {
    #[new]
    fn new() -> Self {
        Self { inner: TemporalSession::new() }
    }
    #[staticmethod]
    fn resume(previous_json: &str, resume_json: &str) -> PyResult<Self> {
        let (ids, ctx) = resume(previous_json, resume_json)?;
        Ok(Self { inner: TemporalSession::resume(ids, ctx) })
    }
    fn is_live(&self) -> bool {
        self.inner.is_live()
    }
    fn identities_json(&self) -> PyResult<String> {
        serde_json::to_string(&identities_to_wire(self.inner.identities())).map_err(crate::py_msg)
    }
    fn capabilities_json(&self) -> PyResult<String> {
        serde_json::to_string(&CapabilitiesWire::from_capabilities(&caps(self.inner.boundary())))
            .map_err(crate::py_msg)
    }
    #[pyo3(signature=(specification,*,seed=1,threads=None))]
    fn plan(&self, specification: &str, seed: u64, threads: Option<u32>) -> PyResult<String> {
        let request = request(specification)?;
        serde_json::to_string(&plan_to_wire(&self.inner.plan(&request, &context(seed, threads))))
            .map_err(crate::py_msg)
    }
    #[pyo3(signature=(specification,*,seed=1,threads=None))]
    fn execute(
        &mut self,
        py: Python<'_>,
        specification: &str,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<RunPayload> {
        let request = request(specification)?;
        let previous = self.inner.identities().clone();
        let capabilities = caps(self.inner.boundary());
        let mut session = std::mem::take(&mut self.inner);
        let ctx = context(seed, threads);
        let (session, ran) = crate::detach_catch(py, move || {
            let ran = execute_temporal_with_receipt(&mut session, &request, &ctx);
            Ok((session, ran))
        })?;
        self.inner = session;
        match ran {
            Ok(out) => outcome(&previous, &self.inner, &capabilities, &out),
            Err(cause) => Ok((None, None, Some(error(&cause)))),
        }
    }
    #[pyo3(signature=(*,seed=1,threads=None))]
    fn export_result(
        &self,
        py: Python<'_>,
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<(Option<Vec<u8>>, Option<String>)> {
        let ctx = context(seed, threads);
        crate::detach_catch(py, || {
            if ctx.cancellation.is_cancelled() {
                return Ok((
                    None,
                    Some(run_error(
                        &RecalcRunError::Request("recalc.cancelled"),
                        "recalc.temporal_fit_refused",
                    )),
                ));
            }
            match self.inner.export_result() {
                Ok(bytes) => Ok((Some(bytes), None)),
                Err(cause) => Ok((None, Some(error(&cause)))),
            }
        })
    }
    #[staticmethod]
    #[pyo3(signature=(artifact,*,seed=1,threads=None))]
    fn consume(
        py: Python<'_>,
        artifact: &[u8],
        seed: u64,
        threads: Option<u32>,
    ) -> PyResult<ReplayPayload> {
        if artifact.len() > 16 * 1024 * 1024 {
            return Err(invalid("recalc.limits_exceeded", "temporal artifact exceeds16MiB"));
        }
        let bytes = artifact.to_vec();
        let ctx = context(seed, threads);
        crate::detach_catch(py, move || match consume_temporal_recalc_artifact(&bytes, &ctx) {
            Ok((session, out)) => {
                let previous = antecedent_core::recalc::StageIdentities::new();
                let (result, receipt, refusal) =
                    outcome(&previous, &session, &caps(session.boundary()), &out)?;
                Ok((Some(Self { inner: session }), result, receipt, refusal))
            }
            Err(cause) => Ok((None, None, None, Some(error(&cause)))),
        })
    }
    #[cfg(feature = "calibration-internal")]
    #[pyo3(signature=(config_json,*,memory_limit_bytes=None,cancel=None))]
    fn interval_candidate(
        &self,
        py: Python<'_>,
        config_json: &str,
        memory_limit_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<(Option<candidate::NativeCheckedTemporalIntervalCandidate>, Option<String>)> {
        candidate::produce(self, py, config_json, memory_limit_bytes, cancel)
    }
    #[pyo3(signature=(config_json,*,memory_limit_bytes=None,cancel=None))]
    fn measured_interval(
        &self,
        py: Python<'_>,
        config_json: &str,
        memory_limit_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<(Option<crate::measured_inference_api::NativeMeasuredInference>, Option<String>)>
    {
        let config: antecedent_io::temporal_interval_artifact::IntervalConfigWire =
            parse_json(config_json, "checked temporal interval config")?;
        let config = config.to_config().map_err(crate::py_msg)?;
        crate::measured_inference_api::before_work(
            0,
            256 * 16 * 128,
            memory_limit_bytes,
            cancel.as_ref(),
        )?;
        let ctx = crate::measured_inference_api::context(0, memory_limit_bytes, cancel);
        crate::detach_catch(py, || {
            match antecedent::analysis::recalc_temporal_measured::CheckedMeasuredTemporal::produce(
                &self.inner,
                &config,
                &ctx,
            ) {
                Ok(measured) => {
                    crate::measured_inference_api::check_cancelled(&ctx)?;
                    Ok((
                        Some(crate::measured_inference_api::NativeMeasuredInference::from_checked(
                            measured,
                        )?),
                        None,
                    ))
                }
                Err(cause) => Ok((None, Some(error(&cause)))),
            }
        })
    }
    fn dependent_interval(&self) -> String {
        if self.inner.source_factor().is_none() {
            return error(&TemporalRunError::Recalc(
                antecedent::analysis::recalc_receipt::RecalcRunError::NoLiveState(
                    antecedent_core::recalc::Stage::ScoreArtifact,
                ),
            ));
        }
        error(&TemporalRunError::Io(antecedent_io::IoError::Refused {
            code: antecedent_core::reason_code!("cell_not_licensed"),
            message:
                "temporal_interval.route_frozen: use the measured original-source interval entry"
                    .into(),
        }))
    }
}
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    #[cfg(feature = "calibration-internal")]
    candidate::register(module)?;
    module.add_class::<PyTemporalSession>()
}
