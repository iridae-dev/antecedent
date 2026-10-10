//! Python binding of the closed learned joint source-target transport route (2.3B B1 row
//! `learned_joint_transport`). The function validates its request against the row's scope
//! through the Rust core and always raises: the core's scope refusal for a request outside
//! the row, otherwise `cell_not_licensed` with `learned_joint_transport.route_frozen`.
//! Calibration is measured only at the release cut, so no interval is published. It backs
//! `antecedent.transport._learned_joint.learned_joint_transport` and is reached only
//! through it.
use antecedent::analysis::learned_joint_closed::{
    LearnedJointRefusal, LearnedJointRequest, learned_joint_transport_refusal,
};
use antecedent_core::reason_code;
use pyo3::prelude::*;

/// A closed-route refusal as the Python exception a caller sees, always carrying
/// `reason_code`: an invalid argument is a value error, every other refusal the registered
/// unsupported class.
fn closed_err(refusal: &LearnedJointRefusal) -> PyErr {
    if refusal.code == reason_code!("invalid_argument") {
        crate::with_reason_code(crate::value_err(refusal.text()), refusal.code)
    } else {
        crate::refusal(refusal.code, refusal.text())
    }
}

/// The closed learned joint source-target transport route; always raises.
#[pyfunction]
#[pyo3(signature = (
    graph_class, dependence, varying, sharing, features, basis_degree, sources, has_target, draws
))]
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
fn learned_joint_transport_closed(
    graph_class: &str,
    dependence: &str,
    varying: &str,
    sharing: &str,
    features: usize,
    basis_degree: usize,
    sources: usize,
    has_target: bool,
    draws: usize,
) -> PyResult<()> {
    let request = LearnedJointRequest {
        graph_class,
        dependence,
        varying,
        sharing,
        features,
        basis_degree,
        sources,
        has_target,
        draws,
    };
    Err(closed_err(&learned_joint_transport_refusal(&request)))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(learned_joint_transport_closed, module)?)?;
    Ok(())
}
