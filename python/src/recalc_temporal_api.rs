//! Checked empirical two-step history adapter; dependent intervals remain closed.
//! SPDX-License-Identifier: MIT OR Apache-2.0
use crate::recalc_api::{RunPayload, bounded_context, context, invalid, parse_json, refusal_json};
use crate::recalc_bounds::check_values;
use crate::recalc_family_api::{caps, payload, resume, run_error};
use antecedent::analysis::recalc_receipt::RecalcRunError;
use antecedent::analysis::recalc_temporal::{
    TemporalOutcome, TemporalRequest, TemporalRunError, TemporalSession,
    consume_temporal_recalc_artifact, execute_temporal_with_receipt,
};
use antecedent_core::ExecutionContext;
use antecedent_io::recalc_receipt_artifact::{CapabilitiesWire, identities_to_wire, plan_to_wire};
use antecedent_io::temporal_recalc_artifact::{TemporalHistoryWire, TemporalUnitWire};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

#[cfg(feature = "calibration-internal")]
mod candidate;

/// Numbers carried by one history window: its time id and five binary states.
const HISTORY_VALUES: usize = 6;

/// Histories as columns: per unit its id and number of windows, then per window its time id, the
/// four binary states and the outcome. Units own consecutive windows in the order given.
type HistoryColumns<'py> = (
    PyReadonlyArray1<'py, u64>,
    PyReadonlyArray1<'py, u64>,
    PyReadonlyArray1<'py, u64>,
    PyReadonlyArray1<'py, u32>,
    PyReadonlyArray1<'py, u32>,
    PyReadonlyArray1<'py, u32>,
    PyReadonlyArray1<'py, u32>,
    PyReadonlyArray1<'py, f64>,
);

fn request(
    text: &str,
    histories: &HistoryColumns<'_>,
    ctx: &ExecutionContext,
) -> PyResult<TemporalRequest> {
    let (unit_ids, unit_lengths, time_ids, s0, a1, l2, a2, y) = histories;
    let mut request: TemporalRequest = parse_json(text, "temporal histories")?;
    if !request.units.is_empty() {
        return Err(invalid(
            "recalc.invalid_declaration",
            "temporal histories travel as columns, not in the declaration",
        ));
    }
    let (unit_ids, unit_lengths) = (unit_ids.as_array(), unit_lengths.as_array());
    let (time_ids, y) = (time_ids.as_array(), y.as_array());
    let states = [s0.as_array(), a1.as_array(), l2.as_array(), a2.as_array()];
    let rows = time_ids.len();
    let total = unit_lengths
        .iter()
        .try_fold(0usize, |n, &len| usize::try_from(len).ok().and_then(|len| n.checked_add(len)));
    if unit_ids.len() != unit_lengths.len()
        || y.len() != rows
        || states.iter().any(|column| column.len() != rows)
        || total != Some(rows)
    {
        return Err(invalid(
            "recalc.invalid_declaration",
            "history columns must agree with the per-unit window counts",
        ));
    }
    let values = rows
        .checked_mul(HISTORY_VALUES)
        .and_then(|n| n.checked_add(unit_ids.len().checked_mul(2)?));
    check_values(values, 0, ctx)?;
    let mut next = 0usize;
    request.units = unit_ids
        .iter()
        .zip(unit_lengths)
        .map(|(&unit_id, &len)| {
            let end = next + usize::try_from(len).unwrap_or(0);
            let histories = (next..end)
                .map(|i| TemporalHistoryWire {
                    time_id: time_ids[i],
                    s0: states[0][i],
                    a1: states[1][i],
                    l2: states[2][i],
                    a2: states[3][i],
                    y: y[i],
                })
                .collect();
            next = end;
            TemporalUnitWire { unit_id, histories }
        })
        .collect();
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
    #[pyo3(signature=(specification,histories,*,seed=1,threads=None,memory_limit_bytes=None,cancel=None))]
    fn plan(
        &self,
        specification: &str,
        histories: HistoryColumns<'_>,
        seed: u64,
        threads: Option<u32>,
        memory_limit_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<String> {
        let ctx = bounded_context(seed, threads, memory_limit_bytes, cancel);
        let request = request(specification, &histories, &ctx)?;
        serde_json::to_string(&plan_to_wire(&self.inner.plan(&request, &ctx)))
            .map_err(crate::py_msg)
    }
    #[pyo3(signature=(specification,histories,*,seed=1,threads=None,memory_limit_bytes=None,cancel=None))]
    fn execute(
        &mut self,
        py: Python<'_>,
        specification: &str,
        histories: HistoryColumns<'_>,
        seed: u64,
        threads: Option<u32>,
        memory_limit_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<RunPayload> {
        let ctx = bounded_context(seed, threads, memory_limit_bytes, cancel);
        let request = request(specification, &histories, &ctx)?;
        let previous = self.inner.identities().clone();
        let capabilities = caps(self.inner.boundary());
        let mut session = std::mem::take(&mut self.inner);
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
        if artifact.len() > crate::bridge::MAX_DECLARED_ARTIFACT_BYTES {
            return Err(invalid("recalc.limits_exceeded", "temporal artifact exceeds byte bound"));
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
