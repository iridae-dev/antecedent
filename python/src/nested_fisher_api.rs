//! Source-bound measured Fisher lifecycle and unchanged feature-only candidate lifecycle.
#[cfg(feature = "calibration-internal")]
use antecedent_core::ExecutionContext;
use antecedent_core::reason_code;
use antecedent_estimate::nested_markov_binary::{
    AdmgDeclaration, FitOptions, NestedMarkovInput, Regime, RegimeCounts,
};
use antecedent_io::{IoError, nested_markov_artifact::NestedFisherArtifact};
use pyo3::prelude::*;
#[cfg(feature = "calibration-internal")]
use pyo3::types::PyBytes;

#[cfg(feature = "calibration-internal")]
use antecedent_io::nested_markov_artifact::{NestedMarkovConsumeLimits, NestedMarkovExpectation};

type RegimeTuple = (Option<Vec<usize>>, Vec<usize>, Vec<f64>);
fn declared_input(
    variables: Vec<String>,
    directed: Vec<(usize, usize)>,
    bidirected: Vec<(usize, usize)>,
    regimes: Vec<RegimeTuple>,
) -> NestedMarkovInput {
    NestedMarkovInput {
        graph: AdmgDeclaration { variables, directed, bidirected },
        regimes: regimes
            .into_iter()
            .map(|(fixed, levels, cells)| RegimeCounts {
                regime: fixed.map_or(Regime::Observational, Regime::Interventional),
                levels,
                cells,
            })
            .collect(),
    }
}
fn error(e: IoError) -> PyErr {
    match e {
        IoError::Refused { code, message } if code == reason_code!("invalid_argument") => {
            crate::with_reason_code(crate::value_err(message), code)
        }
        IoError::Refused { code, message } => crate::refusal(code, message),
        other => crate::value_err(other.to_string()),
    }
}
/// Opaque original candidate; only actual native producers/replay can construct it.
#[cfg(feature = "calibration-internal")]
#[pyclass(name = "NativeNestedFisherCandidate", frozen)]
struct Candidate {
    artifact: NestedFisherArtifact,
    bytes: Vec<u8>,
    identity: String,
}
#[cfg(feature = "calibration-internal")]
impl Candidate {
    fn from_artifact(artifact: NestedFisherArtifact) -> PyResult<Self> {
        let bytes = artifact.export().map_err(error)?;
        let identity = blake3::hash(&bytes).to_hex().to_string();
        Ok(Self { artifact, bytes, identity })
    }
}
#[pymethods]
#[cfg(feature = "calibration-internal")]
impl Candidate {
    #[getter]
    fn identity(&self) -> &str {
        &self.identity
    }
    fn payload(&self) -> PyResult<String> {
        serde_json::to_string(&self.artifact).map_err(|e| crate::value_err(e.to_string()))
    }
    fn export<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.bytes)
    }
}
#[cfg(feature = "calibration-internal")]
#[pyfunction]
#[allow(
    clippy::too_many_arguments,
    reason = "existing graph/regime declaration converted at the binding boundary"
)]
fn nested_markov_fisher_candidate(
    py: Python<'_>,
    variables: Vec<String>,
    directed: Vec<(usize, usize)>,
    bidirected: Vec<(usize, usize)>,
    regimes: Vec<RegimeTuple>,
    nominal_level: f64,
    max_iterations: usize,
    tolerance: f64,
) -> PyResult<Candidate> {
    let input = declared_input(variables, directed, bidirected, regimes);
    let options = FitOptions { max_iterations, tolerance, ..FitOptions::default() };
    crate::detach_catch(py, move || {
        Candidate::from_artifact(
            NestedFisherArtifact::build(
                &input,
                &options,
                nominal_level,
                &ExecutionContext::for_tests(0),
            )
            .map_err(error)?,
        )
    })
}
#[cfg(feature = "calibration-internal")]
#[pyfunction]
fn consume_nested_markov_fisher_candidate(
    py: Python<'_>,
    bytes: &[u8],
    expected_identity: &str,
) -> PyResult<Candidate> {
    if bytes.len() > 256 * 1024 {
        return Err(crate::value_err("nested_markov.fisher_artifact_limits: candidate byte bound"));
    }
    if blake3::hash(bytes).to_hex().as_str() != expected_identity {
        return Err(crate::with_reason_code(
            crate::value_err(
                "nested_markov.fisher_identity_mismatch: candidate identity differs from expected",
            ),
            reason_code!("invalid_argument"),
        ));
    }
    crate::detach_catch(py, move || {
        Candidate::from_artifact(
            NestedFisherArtifact::consume(
                bytes,
                &NestedMarkovExpectation::default(),
                NestedMarkovConsumeLimits::default(),
                &ExecutionContext::for_tests(0),
            )
            .map_err(error)?,
        )
    })
}
#[pyfunction]
#[pyo3(signature=(variables,directed,bidirected,regimes,nominal_level,max_iterations,tolerance,*,memory_limit_bytes=None,cancel=None))]
#[allow(
    clippy::too_many_arguments,
    reason = "existing graph/regime declaration converted at the binding boundary"
)]
fn nested_markov_fisher_measured(
    py: Python<'_>,
    variables: Vec<String>,
    directed: Vec<(usize, usize)>,
    bidirected: Vec<(usize, usize)>,
    regimes: Vec<RegimeTuple>,
    nominal_level: f64,
    max_iterations: usize,
    tolerance: f64,
    memory_limit_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<crate::measured_inference_api::NativeMeasuredInference> {
    crate::measured_inference_api::before_work(0, 256 * 1024, memory_limit_bytes, cancel.as_ref())?;
    let ctx = crate::measured_inference_api::context(0, memory_limit_bytes, cancel);
    let input = declared_input(variables, directed, bidirected, regimes);
    let options = FitOptions { max_iterations, tolerance, ..FitOptions::default() };
    crate::detach_catch(py, move || {
        let (_, measured) =
            NestedFisherArtifact::build_measured(&input, &options, nominal_level, &ctx)
                .map_err(error)?;
        crate::measured_inference_api::NativeMeasuredInference::from_io(measured)
    })
}
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    #[cfg(feature = "calibration-internal")]
    module.add_class::<Candidate>()?;
    #[cfg(feature = "calibration-internal")]
    module.add_function(wrap_pyfunction!(nested_markov_fisher_candidate, module)?)?;
    #[cfg(feature = "calibration-internal")]
    module.add_function(wrap_pyfunction!(consume_nested_markov_fisher_candidate, module)?)?;
    module.add_function(wrap_pyfunction!(nested_markov_fisher_measured, module)?)?;
    Ok(())
}
