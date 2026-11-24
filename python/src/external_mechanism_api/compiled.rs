//! The compiled external feedback fit.
//!
//! [`external_compiled_fit`] binds the same fit request `external_fit` reads into the
//! `ExternalFeedbackFit` program family (`Study::compile_external_feedback_fit`) and executes
//! it with the caller's trusted resolver, so the fit is the one `fit_external_feedback` returns
//! for the same request, together with the program's identity and route.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::Study;
use antecedent::analysis::compiled_external_feedback::{
    ExternalFeedbackFitError, ExternalFeedbackFitRequest,
};
use antecedent::gcm::external_mechanism::TrustedMechanismResolver;
use pyo3::prelude::*;

use super::{FitRequest, execution_context, input_admission, operation_error, resolver_of};

/// Compile and execute a fit request; `{"fit", "lowered_program_identity", "route", "claim"}`
/// as JSON.
#[pyfunction]
#[pyo3(signature = (request_json, resolver=None, *, max_bytes=None, cancel=None))]
pub(super) fn external_compiled_fit(
    py: Python<'_>,
    request_json: &str,
    resolver: Option<Py<PyAny>>,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<String> {
    let ctx = execution_context(max_bytes, cancel);
    let _input = input_admission(request_json, &ctx)?;
    let request: FitRequest =
        serde_json::from_str(request_json).map_err(|e| operation_error(e, &ctx))?;
    let resolver = resolver_of(resolver);
    crate::detach_catch(py, move || {
        let refused = |error: ExternalFeedbackFitError| match error {
            ExternalFeedbackFitError::Model(error) => operation_error(error, &ctx),
            ExternalFeedbackFitError::Compile(error) => crate::py_err(error),
            other => crate::value_err(other.to_string()),
        };
        let compiled = Study::compile_external_feedback_fit(ExternalFeedbackFitRequest::new(
            request.system,
            request.parameters,
            request.rows,
            request.policy,
        ))
        .map_err(refused)?;
        let run = compiled
            .execute(resolver.as_ref().map(|r| r as &dyn TrustedMechanismResolver), &ctx)
            .map_err(refused)?;
        let fit =
            serde_json::to_value(&run.fit).map_err(|e| crate::serialization_err(e.to_string()))?;
        let report = serde_json::json!({
            "fit": fit,
            "lowered_program_identity": run.lowered_program_identity.to_hex(),
            "route": compiled.plan_view().route().as_str(),
            "claim": run.claim,
        });
        serde_json::to_string(&report).map_err(|e| crate::serialization_err(e.to_string()))
    })
}
