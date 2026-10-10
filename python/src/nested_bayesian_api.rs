//! Source-bound measured eleven-dimensional Bayesian lifecycle and unchanged internal candidates.
#[cfg(feature = "calibration-internal")]
use antecedent_core::ExecutionContext;
use antecedent_core::reason_code;
use antecedent_estimate::nested_markov_binary::{
    AdmgDeclaration, FitOptions, NestedMarkovInput, Regime, RegimeCounts,
};
use antecedent_io::{IoError, nested_markov_bayesian_artifact::Artifact};
use antecedent_learn::nested_markov_bayesian::{Options, Prior};
use pyo3::prelude::*;
#[cfg(feature = "calibration-internal")]
use pyo3::types::PyBytes;

#[cfg(feature = "calibration-internal")]
use antecedent_io::nested_markov_bayesian_artifact::{Expectation, Limits};

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
fn declared_prior(alpha: Vec<f64>, beta: Vec<f64>) -> PyResult<Prior> {
    let invalid = |name: &str| {
        crate::with_reason_code(
            crate::value_err(format!(
                "nested_markov.bayesian_invalid_prior: eleven {name} shapes required"
            )),
            reason_code!("invalid_argument"),
        )
    };
    Ok(Prior {
        alpha: alpha.try_into().map_err(|_| invalid("alpha"))?,
        beta: beta.try_into().map_err(|_| invalid("beta"))?,
    })
}
fn python_sampler(seed: u64) -> Options {
    Options {
        chains: 4,
        warmup: 2048,
        draws: 4096,
        max_proposals: 5_000_000,
        seed,
        credible_mass: 0.95,
    }
}
fn error(e: IoError) -> PyErr {
    match e {
        IoError::Refused { code, message } if code == reason_code!("invalid_argument") => {
            crate::with_reason_code(crate::value_err(message), code)
        }
        IoError::Refused { code, message } => crate::refusal(code, message),
        other => crate::with_reason_code(
            crate::value_err(other.to_string()),
            reason_code!("invalid_argument"),
        ),
    }
}
#[cfg(feature = "calibration-internal")]
#[pyclass(name = "NativeNestedMarkovPosteriorCandidate", frozen)]
struct Candidate {
    artifact: Artifact,
    bytes: Vec<u8>,
    identity: String,
}
#[cfg(feature = "calibration-internal")]
impl Candidate {
    fn from_artifact(artifact: Artifact) -> PyResult<Self> {
        let o = &artifact.options;
        if o.chains != 4
            || o.warmup != 2048
            || o.draws != 4096
            || o.max_proposals != 5_000_000
            || o.credible_mass.to_bits() != 0.95_f64.to_bits()
        {
            return Err(crate::refusal(
                reason_code!("route_not_supported"),
                "nested_markov.bayesian_sampler_scope: artifact is not the frozen Python pilot sampler",
            ));
        }
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
    reason = "typed graph/regime and named prior converted at the binding boundary"
)]
fn nested_markov_posterior_candidate(
    py: Python<'_>,
    variables: Vec<String>,
    directed: Vec<(usize, usize)>,
    bidirected: Vec<(usize, usize)>,
    regimes: Vec<RegimeTuple>,
    max_iterations: usize,
    tolerance: f64,
    alpha: Vec<f64>,
    beta: Vec<f64>,
    seed: u64,
) -> PyResult<Candidate> {
    let input = declared_input(variables, directed, bidirected, regimes);
    let options = FitOptions { max_iterations, tolerance, ..FitOptions::default() };
    let prior = declared_prior(alpha, beta)?;
    let sampler = python_sampler(seed);
    crate::detach_catch(py, move || {
        Candidate::from_artifact(
            Artifact::build(&input, &options, prior, sampler, &ExecutionContext::for_tests(0))
                .map_err(error)?,
        )
    })
}
#[cfg(feature = "calibration-internal")]
#[pyfunction]
fn consume_nested_markov_posterior_candidate(
    py: Python<'_>,
    bytes: &[u8],
    expected_identity: &str,
) -> PyResult<Candidate> {
    if bytes.len() > 16 * 1024 * 1024 {
        return Err(crate::with_reason_code(
            crate::value_err("nested_markov.bayesian_artifact_limits: candidate byte bound"),
            reason_code!("invalid_argument"),
        ));
    }
    if blake3::hash(bytes).to_hex().as_str() != expected_identity {
        return Err(crate::with_reason_code(
            crate::value_err(
                "nested_markov.bayesian_identity_mismatch: candidate identity differs from expected",
            ),
            reason_code!("invalid_argument"),
        ));
    }
    crate::detach_catch(py, move || {
        Candidate::from_artifact(
            Artifact::consume(
                bytes,
                &Expectation::default(),
                Limits::default(),
                &ExecutionContext::for_tests(0),
            )
            .map_err(error)?,
        )
    })
}
#[pyfunction]
#[pyo3(signature=(variables,directed,bidirected,regimes,max_iterations,tolerance,alpha,beta,seed,*,memory_limit_bytes=None,cancel=None))]
#[allow(
    clippy::too_many_arguments,
    reason = "typed graph/regime and named prior converted at the binding boundary"
)]
fn nested_markov_posterior_measured(
    py: Python<'_>,
    variables: Vec<String>,
    directed: Vec<(usize, usize)>,
    bidirected: Vec<(usize, usize)>,
    regimes: Vec<RegimeTuple>,
    max_iterations: usize,
    tolerance: f64,
    alpha: Vec<f64>,
    beta: Vec<f64>,
    seed: u64,
    memory_limit_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<crate::measured_inference_api::NativeMeasuredInference> {
    crate::measured_inference_api::before_work(
        0,
        16 * 1024 * 1024,
        memory_limit_bytes,
        cancel.as_ref(),
    )?;
    let ctx = crate::measured_inference_api::context(0, memory_limit_bytes, cancel);
    let input = declared_input(variables, directed, bidirected, regimes);
    let options = FitOptions { max_iterations, tolerance, ..FitOptions::default() };
    let prior = declared_prior(alpha, beta)?;
    let sampler = python_sampler(seed);
    crate::detach_catch(py, move || {
        let (_, measured) =
            Artifact::build_measured(&input, &options, prior, sampler, &ctx).map_err(error)?;
        crate::measured_inference_api::NativeMeasuredInference::from_io(measured)
    })
}
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    #[cfg(feature = "calibration-internal")]
    module.add_class::<Candidate>()?;
    #[cfg(feature = "calibration-internal")]
    module.add_function(wrap_pyfunction!(nested_markov_posterior_candidate, module)?)?;
    #[cfg(feature = "calibration-internal")]
    module.add_function(wrap_pyfunction!(consume_nested_markov_posterior_candidate, module)?)?;
    module.add_function(wrap_pyfunction!(nested_markov_posterior_measured, module)?)?;
    Ok(())
}
