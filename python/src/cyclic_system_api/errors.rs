//! Cyclic execution and replay error classification for the Python boundary.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::prelude::ExecutionContext;
use pyo3::prelude::*;

pub(super) fn operation_error(
    error: impl std::error::Error + 'static,
    ctx: &ExecutionContext,
) -> PyErr {
    // A Python callback that raised inside the operation is found by its structured
    // failure, so the run boundary can restore the original exception as the cause.
    let failure = antecedent::execution::CallbackFailure::in_chain(&error).cloned();
    let converted = classified_operation_error(error, ctx);
    match &failure {
        Some(failure) => crate::interrupt::with_callback_failure(converted, failure),
        None => converted,
    }
}

fn classified_operation_error(
    error: impl std::error::Error + 'static,
    ctx: &ExecutionContext,
) -> PyErr {
    use antecedent::io::cyclic_system_artifact::CyclicArtifactError;
    if ctx.cancellation.is_cancelled() {
        return crate::CausalCancelledError::new_err(error.to_string());
    }
    let mut cause: &(dyn std::error::Error + 'static) = &error;
    loop {
        if let Some(CyclicArtifactError::RecoveryFit(failure)) =
            cause.downcast_ref::<CyclicArtifactError>()
        {
            use antecedent::estimate::experimental_fit::ExperimentalFitError;
            return match failure {
                ExperimentalFitError::Admission(refusal)
                    if antecedent::execution::AdmissionRefusal::from(*refusal).is_cancelled() =>
                {
                    crate::CausalCancelledError::new_err(failure.to_string())
                }
                ExperimentalFitError::Admission(_)
                | ExperimentalFitError::BudgetExceeded { .. }
                | ExperimentalFitError::ResourceOverflow => {
                    crate::CausalResourceError::new_err(failure.to_string())
                }
                _ => crate::value_err(failure.to_string()),
            };
        }
        if let Some(CyclicArtifactError::WorldRefusal(refusal)) =
            cause.downcast_ref::<CyclicArtifactError>()
        {
            return world_refusal_error(refusal);
        }
        if let Some(refusal) =
            cause.downcast_ref::<antecedent::gcm::cyclic_world::CyclicInterventionRefusal>()
        {
            return world_refusal_error(refusal);
        }
        if let Some(CyclicArtifactError::ScientificRefusal {
            code,
            detail,
            message,
            next_attempt,
        }) = cause.downcast_ref::<CyclicArtifactError>()
        {
            return scientific_error(code, detail, message, next_attempt);
        }
        if matches!(
            cause.downcast_ref::<CyclicArtifactError>(),
            Some(CyclicArtifactError::Cancelled)
        ) {
            return crate::CausalCancelledError::new_err(error.to_string());
        }
        if antecedent::execution::is_admission_refusal(cause)
            || matches!(
                cause.downcast_ref::<CyclicArtifactError>(),
                Some(
                    CyclicArtifactError::Admission(_)
                        | CyclicArtifactError::Execution(
                            antecedent::gcm::nonlinear_equilibrium::EquilibriumError::Resource(_)
                        )
                )
            )
            || matches!(
                cause.downcast_ref::<antecedent::gcm::nonlinear_equilibrium::EquilibriumError>(),
                Some(antecedent::gcm::nonlinear_equilibrium::EquilibriumError::Resource(_))
            )
        {
            return crate::CausalResourceError::new_err(error.to_string());
        }
        match cause.source() {
            Some(source) => cause = source,
            None => return crate::value_err(error.to_string()),
        }
    }
}

fn world_refusal_error(
    refusal: &antecedent::gcm::cyclic_world::CyclicInterventionRefusal,
) -> PyErr {
    use antecedent::gcm::cyclic_world::CyclicRefusalKind;
    match &refusal.kind {
        CyclicRefusalKind::Capability { .. } | CyclicRefusalKind::Evidence { .. } => {
            scientific_error(
                antecedent::reason::reason_code!("route_not_supported"),
                &format!("cyclic_world.{}", refusal.operation),
                &refusal.to_string(),
                refusal.next_attempt.sentence(),
            )
        }
        CyclicRefusalKind::Execution { .. } => {
            crate::CausalResourceError::new_err(refusal.to_string())
        }
        _ => crate::value_err(refusal.to_string()),
    }
}

pub(super) fn scientific_error(
    code: &str,
    detail: &str,
    message: &str,
    next_attempt: &str,
) -> PyErr {
    Python::attach(|py| {
        let build = || -> PyResult<PyErr> {
            let fields = pyo3::types::PyDict::new(py);
            fields.set_item("code", code)?;
            fields.set_item("detail", detail)?;
            fields.set_item("message", message)?;
            fields.set_item("remedy", next_attempt)?;
            fields.set_item("next_attempt", next_attempt)?;
            let class = pyo3::types::PyModule::import(py, "antecedent.errors")?
                .getattr("StructuredRefusal")?;
            Ok(PyErr::from_value(class.call1((fields,))?))
        };
        build().unwrap_or_else(|error| error)
    })
}

pub(super) fn replay_error(
    error: impl std::error::Error + 'static,
    ctx: &ExecutionContext,
) -> PyErr {
    let message = error.to_string();
    let classified = operation_error(error, ctx);
    Python::attach(|py| {
        if classified.is_instance_of::<crate::CausalResourceError>(py)
            || classified.is_instance_of::<crate::CausalCancelledError>(py)
        {
            classified
        } else {
            crate::serialization_err(message)
        }
    })
}
