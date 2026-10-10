//! Factory-only measured scalar carrier and bounded fresh native consumer.
use antecedent_io::measured_inference::{
    MAX_MEASURED_ARTIFACT_BYTES, MeasuredExpectation, MeasuredInference,
};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::sync::{Arc, OnceLock};

enum NativeAuthority {
    Io(MeasuredInference),
    Checked(antecedent::analysis::recalc_temporal_measured::CheckedMeasuredTemporal),
}
impl NativeAuthority {
    fn report(&self) -> &antecedent_io::measured_inference::MeasuredInferenceReport {
        match self {
            Self::Io(value) => value.report(),
            Self::Checked(value) => value.report(),
        }
    }
    fn export(&self) -> &[u8] {
        match self {
            Self::Io(value) => value.export(),
            Self::Checked(value) => value.export(),
        }
    }
    fn source_artifact(&self) -> &[u8] {
        match self {
            Self::Io(value) => value.source_artifact(),
            Self::Checked(value) => value.source_artifact(),
        }
    }
}

/// Only successful original native production/independent replay can construct this class.
#[pyclass(module = "antecedent._native", frozen)]
pub(crate) struct NativeMeasuredInference {
    authority: NativeAuthority,
    payload: String,
    source_report: OnceLock<Arc<str>>,
}
impl NativeMeasuredInference {
    pub(crate) fn from_io(authority: MeasuredInference) -> PyResult<Self> {
        let payload = serde_json::to_string(authority.report()).map_err(crate::py_msg)?;
        Ok(Self {
            authority: NativeAuthority::Io(authority),
            payload,
            source_report: OnceLock::new(),
        })
    }
    pub(crate) fn from_checked(
        authority: antecedent::analysis::recalc_temporal_measured::CheckedMeasuredTemporal,
    ) -> PyResult<Self> {
        let payload = serde_json::to_string(authority.report()).map_err(crate::py_msg)?;
        Ok(Self {
            authority: NativeAuthority::Checked(authority),
            payload,
            source_report: OnceLock::new(),
        })
    }
}
#[pymethods]
impl NativeMeasuredInference {
    fn payload(&self) -> &str {
        &self.payload
    }
    fn export<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, self.authority.export())
    }
    fn source_artifact<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, self.authority.source_artifact())
    }
    fn source_report(&self, py: Python<'_>) -> PyResult<String> {
        if let Some(report) = self.source_report.get() {
            return Ok(report.to_string());
        }
        // Reading the unchanged original report is inspection only; it never fits,
        // licenses posterior quantities or constructs another native authority.
        let report = crate::detach_catch(py, || {
            if let NativeAuthority::Checked(value) = &self.authority {
                return serde_json::to_string(
                    &value.source_report().map_err(crate::transport_common::error)?,
                )
                .map_err(crate::py_msg);
            }
            let bytes = self.authority.source_artifact();
            match self.authority.report().route.as_str() {
                "joint_bayesian" => {
                    let wire = antecedent_io::joint_bayesian_transport_artifact::JointBayesianArtifactWire::decode(bytes).map_err(crate::transport_common::error)?;
                    serde_json::to_string(&wire).map_err(crate::py_msg)
                }
                "learned_joint" => {
                    let wire = antecedent_io::learned_joint_transport_artifact::LearnedJointArtifactWire::decode(bytes).map_err(crate::transport_common::error)?;
                    serde_json::to_string(&wire).map_err(crate::py_msg)
                }
                "nested_fisher" => {
                    let wire: antecedent_io::nested_markov_artifact::NestedFisherArtifact =
                        antecedent_io::from_cbor(bytes).map_err(crate::transport_common::error)?;
                    serde_json::to_string(&wire).map_err(crate::py_msg)
                }
                "nested_bayesian" => {
                    let wire: antecedent_io::nested_markov_bayesian_artifact::Artifact =
                        antecedent_io::from_cbor(bytes).map_err(crate::transport_common::error)?;
                    serde_json::to_string(&wire).map_err(crate::py_msg)
                }
                "sampled_recovery" => {
                    let wire: antecedent_io::sampled_recovery_artifact::SampledRecoveryArtifactWire = antecedent_io::from_cbor(bytes).map_err(crate::transport_common::error)?;
                    serde_json::to_string(&wire).map_err(crate::py_msg)
                }
                "temporal_interval" => {
                    let wire: antecedent_io::temporal_interval_artifact::TemporalIntervalArtifactWire = antecedent_io::from_cbor(bytes).map_err(crate::transport_common::error)?;
                    serde_json::to_string(&wire).map_err(crate::py_msg)
                }
                _ => Err(crate::value_err("measured_inference.source_report_not_supported")),
            }
        })?;
        let _ = self.source_report.set(Arc::from(report.as_str()));
        Ok(report)
    }
}

/// Conservative peak bound includes decode, actual fit/draw replay, original wire,
/// canonical envelope and explicit original source-report inspection.
pub(crate) fn before_work(
    bytes: usize,
    numerical_bytes: usize,
    memory: Option<u64>,
    cancel: Option<&crate::PyCancellationToken>,
) -> PyResult<()> {
    if cancel.is_some_and(|token| token.inner.is_cancelled()) {
        return Err(crate::with_reason_code(
            crate::CausalCancelledError::new_err("measured_inference.cancelled before native work"),
            antecedent_core::reason_code!("cancelled_no_claim"),
        ));
    }
    let planned = u64::try_from(bytes)
        .unwrap_or(u64::MAX)
        .saturating_mul(24)
        .saturating_add(u64::try_from(numerical_bytes).unwrap_or(u64::MAX).saturating_mul(16))
        .saturating_add(16 * 1024 * 1024);
    if memory.is_some_and(|limit| planned > limit) {
        return Err(crate::CausalResourceError::new_err(
            "measured_inference.memory_budget_exceeded: planned native peak exceeds caller limit",
        ));
    }
    Ok(())
}
pub(crate) fn check_cancelled(ctx: &antecedent_core::ExecutionContext) -> PyResult<()> {
    if ctx.cancellation.is_cancelled() {
        return Err(crate::with_reason_code(
            crate::CausalCancelledError::new_err(
                "measured_inference.cancelled before returning a claim",
            ),
            antecedent_core::reason_code!("cancelled_no_claim"),
        ));
    }
    Ok(())
}

pub(crate) fn context(
    seed: u64,
    memory: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> antecedent_core::ExecutionContext {
    let mut ctx = crate::py_execution_context_cancel(seed, 1, cancel);
    ctx.memory =
        antecedent_core::MemoryBudget { soft_limit_bytes: memory, hard_limit_bytes: memory };
    ctx
}

#[pyfunction]
#[pyo3(signature=(artifact,expected_identity_json,*,max_bytes=67_108_864,memory_limit_bytes=None,cancel=None))]
fn consume_measured_inference(
    py: Python<'_>,
    artifact: &[u8],
    expected_identity_json: &str,
    max_bytes: usize,
    memory_limit_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<NativeMeasuredInference> {
    if max_bytes > MAX_MEASURED_ARTIFACT_BYTES
        || artifact.len() > max_bytes
        || expected_identity_json.len() > 64 * 1024
    {
        return Err(crate::CausalResourceError::new_err(
            "measured_inference.consumer_bounds_exceeded",
        ));
    }
    before_work(artifact.len(), 0, memory_limit_bytes, cancel.as_ref())?;
    let expected: MeasuredExpectation = serde_json::from_str(expected_identity_json)
        .map_err(|e| crate::value_err(format!("measured_inference.invalid_expectation: {e}")))?;
    let digest = |text: &str| {
        text.len() == 64 && text.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    };
    if !digest(&expected.candidate_digest)
        || !digest(&expected.premises_digest)
        || !digest(&expected.data_digest)
        || !digest(&expected.seal)
        || expected.route.len() > 128
        || expected.scalars.is_empty()
        || expected.scalars.len() > 32
        || expected.scalars.iter().any(|s| s.len() > 128)
    {
        return Err(crate::value_err("measured_inference.invalid_expectation"));
    }
    let ctx = context(0, memory_limit_bytes, cancel);
    crate::detach_catch(py, || {
        if expected.route == "checked_temporal" {
            let value =
                antecedent::analysis::recalc_temporal_measured::CheckedMeasuredTemporal::consume(
                    artifact, &expected, &ctx,
                )
                .map_err(|error| {
                    use antecedent::analysis::recalc_temporal::TemporalRunError;
                    match error {
                        TemporalRunError::Io(error) => crate::transport_common::error(error),
                        other if ctx.cancellation.is_cancelled() => {
                            crate::CausalCancelledError::new_err(other.to_string())
                        }
                        other => crate::value_err(other.to_string()),
                    }
                })?;
            check_cancelled(&ctx)?;
            return NativeMeasuredInference::from_checked(value);
        }
        let verified = MeasuredInference::consume(artifact, &expected, &ctx)
            .map_err(crate::transport_common::error)?;
        check_cancelled(&ctx)?;
        NativeMeasuredInference::from_io(verified)
    })
}
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeMeasuredInference>()?;
    module.add_function(wrap_pyfunction!(consume_measured_inference, module)?)
}
