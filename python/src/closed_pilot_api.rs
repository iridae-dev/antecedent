//! Python bindings of the three closed 2.3A calibrated-interval pilot routes: joint
//! Bayesian transport (A2), the binary nested-Markov pilot (A3) and sampled observation
//! recovery (A6). Each function validates its request against the pilot's scope through
//! the Rust core and always raises: the engine's scope refusal for a request outside the
//! pilot, otherwise `cell_not_licensed` with the route-frozen detail. Calibration is
//! measured only at the release cut, so no interval is published. Each backs one routed
//! wrapper in `antecedent.transport._closed_pilots` and is reached only through it.
use antecedent::analysis::closed_pilots::{
    ClosedPilotRefusal, ClosedRegimeCounts, JointBayesianRequest, NestedMarkovRequest,
    SampledRecoveryRequest, joint_bayesian_transport_refusal, nested_markov_refusal,
    sampled_recovery_refusal,
};
use antecedent_core::reason_code;
use pyo3::prelude::*;

/// A closed-route refusal as the Python exception a caller sees, always carrying
/// `reason_code`: an invalid argument is a value error, every other refusal the
/// registered unsupported class.
fn closed_err(refusal: &ClosedPilotRefusal) -> PyErr {
    if refusal.code == reason_code!("invalid_argument") {
        crate::with_reason_code(crate::value_err(refusal.text()), refusal.code)
    } else {
        crate::refusal(refusal.code, refusal.text())
    }
}

/// The closed joint Bayesian source-target transport route; always raises.
#[pyfunction]
#[pyo3(signature = (graph_class, dependence, varying, sharing, features, sources, has_target, draws))]
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
fn joint_bayesian_transport_closed(
    graph_class: &str,
    dependence: &str,
    varying: &str,
    sharing: &str,
    features: usize,
    sources: usize,
    has_target: bool,
    draws: usize,
) -> PyResult<()> {
    let request = JointBayesianRequest {
        graph_class,
        dependence,
        varying,
        sharing,
        features,
        sources,
        has_target,
        draws,
    };
    Err(closed_err(&joint_bayesian_transport_refusal(&request)))
}

/// One regime table of a nested-Markov request: `(fixed variables or None, levels, cells)`.
type RegimeTuple = (Option<Vec<usize>>, Vec<usize>, Vec<f64>);

/// The closed binary nested-Markov route; always raises.
#[pyfunction]
#[pyo3(signature = (variables, directed, bidirected, regimes))]
#[doc(hidden)]
fn binary_nested_markov_closed(
    variables: Vec<String>,
    directed: Vec<(usize, usize)>,
    bidirected: Vec<(usize, usize)>,
    regimes: Vec<RegimeTuple>,
) -> PyResult<()> {
    let request = NestedMarkovRequest {
        variables,
        directed,
        bidirected,
        regimes: regimes
            .into_iter()
            .map(|(fixed, levels, cells)| ClosedRegimeCounts { fixed, levels, cells })
            .collect(),
    };
    Err(closed_err(&nested_markov_refusal(&request)))
}

/// The closed sampled observation-recovery route; always raises.
#[pyfunction]
#[pyo3(signature = (graph_recoverable, partially_observed, fully_observed, replicates, rows))]
#[doc(hidden)]
fn sampled_observation_recovery_closed(
    graph_recoverable: bool,
    partially_observed: usize,
    fully_observed: usize,
    replicates: usize,
    rows: Vec<(u64, u8, u8, u8)>,
) -> PyResult<()> {
    let request = SampledRecoveryRequest {
        graph_recoverable,
        partially_observed,
        fully_observed,
        replicates,
        rows: &rows,
    };
    Err(closed_err(&sampled_recovery_refusal(&request)))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(joint_bayesian_transport_closed, module)?)?;
    module.add_function(wrap_pyfunction!(binary_nested_markov_closed, module)?)?;
    module.add_function(wrap_pyfunction!(sampled_observation_recovery_closed, module)?)?;
    Ok(())
}
