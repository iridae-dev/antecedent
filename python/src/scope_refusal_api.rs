//! Python bindings of the three scope-checked routes: joint
//! Bayesian transport, the binary nested-Markov route and sampled observation
//! recovery. Each function validates its request against the route's scope through
//! the Rust core: the engine's scope refusal for a request outside the route raises, and a
//! request inside it returns so the wrapper runs the route's own producer and reports the
//! interval with its calibration status. Each backs one routed
//! wrapper in `antecedent.transport._advanced_producers` and is reached only through it.
use antecedent::analysis::scope_refusals::{
    JointBayesianRequest, NestedMarkovRequest, RegimeCounts, SampledRecoveryRequest, ScopeRefusal,
    joint_bayesian_transport_scope_refusal, nested_markov_scope_refusal,
    sampled_recovery_scope_refusal,
};
use antecedent_core::reason_code;
use pyo3::prelude::*;

/// A scope refusal as the Python exception a caller sees, always carrying
/// `reason_code`: an invalid argument is a value error, every other refusal the
/// registered unsupported class.
fn scope_err(refusal: &ScopeRefusal) -> PyErr {
    if refusal.code == reason_code!("invalid_argument") {
        crate::with_reason_code(crate::value_err(refusal.text()), refusal.code)
    } else {
        crate::refusal(refusal.code, refusal.text())
    }
}

/// The joint Bayesian source-target transport route; raises the scope refusal, returns when in scope.
#[pyfunction]
#[pyo3(signature = (graph_class, dependence, varying, sharing, features, sources, has_target, draws))]
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
fn joint_bayesian_transport_scope(
    graph_class: &str,
    dependence: &str,
    varying: &str,
    sharing: &str,
    features: usize,
    sources: usize,
    has_target: bool,
    draws: usize,
) -> PyResult<()> {
    crate::catch_ffi(|| {
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
        match joint_bayesian_transport_scope_refusal(&request) {
            Some(refusal) => Err(scope_err(&refusal)),
            None => Ok(()),
        }
    })
}

/// One regime table of a nested-Markov request: `(fixed variables or None, levels, cells)`.
type RegimeTuple = (Option<Vec<usize>>, Vec<usize>, Vec<f64>);

/// The binary nested-Markov route; raises the scope refusal, returns when in scope.
#[pyfunction]
#[pyo3(signature = (variables, directed, bidirected, regimes))]
#[doc(hidden)]
fn binary_nested_markov_scope(
    variables: Vec<String>,
    directed: Vec<(usize, usize)>,
    bidirected: Vec<(usize, usize)>,
    regimes: Vec<RegimeTuple>,
) -> PyResult<()> {
    crate::catch_ffi(|| {
        let request = NestedMarkovRequest {
            variables,
            directed,
            bidirected,
            regimes: regimes
                .into_iter()
                .map(|(fixed, levels, cells)| RegimeCounts { fixed, levels, cells })
                .collect(),
        };
        match nested_markov_scope_refusal(&request) {
            Some(refusal) => Err(scope_err(&refusal)),
            None => Ok(()),
        }
    })
}

/// The sampled observation-recovery route; raises the scope refusal, returns when in scope.
#[pyfunction]
#[pyo3(signature = (graph_recoverable, partially_observed, fully_observed, replicates, rows))]
#[doc(hidden)]
fn sampled_observation_recovery_scope(
    graph_recoverable: bool,
    partially_observed: usize,
    fully_observed: usize,
    replicates: usize,
    rows: Vec<(u64, u8, u8, u8)>,
) -> PyResult<()> {
    crate::catch_ffi(|| {
        let request = SampledRecoveryRequest {
            graph_recoverable,
            partially_observed,
            fully_observed,
            replicates,
            rows: &rows,
        };
        match sampled_recovery_scope_refusal(&request) {
            Some(refusal) => Err(scope_err(&refusal)),
            None => Ok(()),
        }
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(joint_bayesian_transport_scope, module)?)?;
    module.add_function(wrap_pyfunction!(binary_nested_markov_scope, module)?)?;
    module.add_function(wrap_pyfunction!(sampled_observation_recovery_scope, module)?)?;
    Ok(())
}
