//! Feature-only intended nested Fisher producer/consumer lifecycle. Never registered
//! by normal builds; outputs remain typed unmeasured candidates, not licenses.
use antecedent_core::{ExecutionContext, reason_code};
use antecedent_estimate::nested_markov_binary::{
    AdmgDeclaration, FitOptions, NestedMarkovInput, Regime, RegimeCounts,
};
use antecedent_io::{
    IoError,
    nested_markov_artifact::{
        NestedFisherArtifact, NestedMarkovConsumeLimits, NestedMarkovExpectation,
    },
};
use pyo3::{prelude::*, types::PyBytes};

type RegimeTuple = (Option<Vec<usize>>, Vec<usize>, Vec<f64>);
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
#[pyclass(name = "NativeNestedFisherCandidate", frozen)]
struct Candidate {
    artifact: NestedFisherArtifact,
    bytes: Vec<u8>,
    identity: String,
}
impl Candidate {
    fn from_artifact(artifact: NestedFisherArtifact) -> PyResult<Self> {
        let bytes = artifact.export().map_err(error)?;
        let identity = blake3::hash(&bytes).to_hex().to_string();
        Ok(Self { artifact, bytes, identity })
    }
}
#[pymethods]
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
    let input = NestedMarkovInput {
        graph: AdmgDeclaration { variables, directed, bidirected },
        regimes: regimes
            .into_iter()
            .map(|(fixed, levels, cells)| RegimeCounts {
                regime: fixed.map_or(Regime::Observational, Regime::Interventional),
                levels,
                cells,
            })
            .collect(),
    };
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
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Candidate>()?;
    module.add_function(wrap_pyfunction!(nested_markov_fisher_candidate, module)?)?;
    module.add_function(wrap_pyfunction!(consume_nested_markov_fisher_candidate, module)?)?;
    Ok(())
}
