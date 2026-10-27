//! Python exception hierarchy, error classes registered by `antecedent.errors`, and the conversion of native errors into them.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::*;

create_exception!(antecedent._native, CausalError, PyException);
create_exception!(antecedent._native, CausalIdentifyError, CausalError);
create_exception!(antecedent._native, CausalEstimateError, CausalError);
create_exception!(antecedent._native, CausalValidateError, CausalError);
create_exception!(antecedent._native, CausalDiscoveryError, CausalError);
create_exception!(antecedent._native, CausalModelError, CausalError);
create_exception!(antecedent._native, CausalCounterfactualError, CausalError);
create_exception!(antecedent._native, CausalAttributionError, CausalError);
create_exception!(antecedent._native, CausalDataError, CausalError);
create_exception!(antecedent._native, CausalGraphError, CausalError);
create_exception!(antecedent._native, CausalDesignError, CausalError);
create_exception!(antecedent._native, CausalStateError, CausalError);
create_exception!(antecedent._native, CausalSerializationError, CausalError);
create_exception!(antecedent._native, CausalCompileError, CausalError);
create_exception!(antecedent._native, CausalResourceError, CausalError);
create_exception!(antecedent._native, CausalReviewError, CausalError);
create_exception!(antecedent._native, CausalUnsupportedError, CausalError);
create_exception!(antecedent._native, CausalCancelledError, CausalError);

/// One entry in a raised `ReviewRequired`'s `pending_edges` sequence.
///
/// Read-only wire type mirroring `antecedent::error::PendingEdge`, exposed to Python
/// as plain attributes so `antecedent.errors.PendingEdge` (a frozen dataclass) can
/// normalize instances of this into its own type without depending on this class's
/// identity.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct CausalPendingEdge {
    #[pyo3(get)]
    source: String,
    #[pyo3(get)]
    target: String,
    #[pyo3(get)]
    at_source: String,
    #[pyo3(get)]
    at_target: String,
}

impl From<&FacadePendingEdge> for CausalPendingEdge {
    fn from(edge: &FacadePendingEdge) -> Self {
        Self {
            source: edge.source.clone(),
            target: edge.target.clone(),
            at_source: edge.at_source.clone(),
            at_target: edge.at_target.clone(),
        }
    }
}

/// The Python-defined `ReviewRequired` class, registered by `antecedent.errors` at
/// import time via [`set_review_error_class`]. `None` when the native module is used
/// standalone (e.g. `import antecedent._native` without the `antecedent` package),
/// in which case [`review_required_py_err`] falls back to bare `CausalReviewError`.
pub(crate) static REVIEW_ERROR_CLASS: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

/// Register the Python `ReviewRequired` class the error mapper should instantiate for
/// `CausalError::ReviewRequired`, collapsing what used to be ad hoc `setattr` calls at
/// each Python-facing construction site into this one registration.
///
/// Called once by `antecedent.errors` at import time. A second call is a silent
/// no-op: [`PyOnceLock`] only accepts the first `set`, and the intended caller is a
/// single top-level import.
#[pyfunction]
pub(crate) fn set_review_error_class(py: Python<'_>, cls: Py<PyAny>) {
    let _ = REVIEW_ERROR_CLASS.get_or_init(py, || cls);
}

/// Build the Python exception for `CausalError::ReviewRequired`.
///
/// Instantiates the registered `ReviewRequired` class (see [`set_review_error_class`])
/// when available, so callers get the real class in `type(err).__mro__` plus a
/// structured `pending_edges` attribute; falls back to bare `CausalReviewError` when
/// nothing is registered. Either way this is the single place that builds a
/// review-required Python exception — `kind` / `algorithm` / `pending_edge_count` /
/// `pending_edges` / `hint` / `message` are attached identically in both branches, and
/// `str(err)` stays byte-identical to `message`.
pub(crate) fn review_required_py_err(
    kind: String,
    algorithm: Option<String>,
    pending_edge_count: usize,
    pending_edges: Arc<[FacadePendingEdge]>,
    message: String,
    hint: String,
) -> PyErr {
    Python::attach(|py| {
        let edges: Vec<Py<CausalPendingEdge>> = pending_edges
            .iter()
            .filter_map(|e| Py::new(py, CausalPendingEdge::from(e)).ok())
            .collect();

        let err: PyErr = REVIEW_ERROR_CLASS
            .get(py)
            .and_then(|cls| cls.bind(py).call1((message.as_str(),)).ok())
            .map_or_else(|| CausalReviewError::new_err(message.clone()), PyErr::from_value);

        let inst = err.value(py);
        let _ = inst.setattr("kind", kind.as_str());
        let _ = inst.setattr("algorithm", algorithm.as_deref());
        let _ = inst.setattr("pending_edge_count", pending_edge_count);
        let _ = inst.setattr("pending_edges", edges);
        let _ = inst.setattr("hint", hint.as_str());
        let _ = inst.setattr("message", message.as_str());
        err
    })
}

/// The Python `CausalUnsupportedError` subclass, registered by `antecedent.errors`.
pub(crate) static UNSUPPORTED_ERROR_CLASS: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

/// Register the Python `CausalUnsupportedError` the error mapper instantiates.
///
/// Keeps one refusal class across the boundary: a refusal raised in Rust is the
/// same class, with the same `reason_code`, as one raised in Python.
#[pyfunction]
pub(crate) fn set_unsupported_error_class(py: Python<'_>, cls: Py<PyAny>) {
    let _ = UNSUPPORTED_ERROR_CLASS.get_or_init(py, || cls);
}

/// The Python `CausalValueError` subclass, registered by `antecedent.errors`.
pub(crate) static VALUE_ERROR_CLASS: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

/// Register the Python `CausalValueError` invalid-input failures are raised as.
#[pyfunction]
pub(crate) fn set_value_error_class(py: Python<'_>, cls: Py<PyAny>) {
    let _ = VALUE_ERROR_CLASS.get_or_init(py, || cls);
}

/// Invalid input at a native entry point, as the registered `CausalValueError`
/// (a `CausalValidateError` that is also a `ValueError`).
///
/// The message is unchanged. A registered `reason=<code>:` prefix already in
/// the message is copied onto `reason_code`.
pub(crate) fn value_err(message: impl AsRef<str>) -> PyErr {
    let message = message.as_ref();
    attach_reason_from_message(value_error_of(message), message)
}

/// [`value_err`] with an explicit registered reason code.
///
/// The code is an attribute. It is not written into the message.
pub(crate) fn value_err_code(code: &str, message: impl AsRef<str>) -> PyErr {
    with_reason_code(value_error_of(message.as_ref()), code)
}

pub(crate) fn value_error_of(message: &str) -> PyErr {
    Python::attach(|py| {
        VALUE_ERROR_CLASS
            .get(py)
            .and_then(|cls| cls.bind(py).call1((message,)).ok())
            .map_or_else(|| CausalValidateError::new_err(message.to_string()), PyErr::from_value)
    })
}

/// A serialization failure, as `CausalSerializationError`.
///
/// A registered `reason=<code>:` prefix already in the message is copied onto
/// `reason_code`. The message text is the argument unchanged.
#[allow(
    clippy::disallowed_methods,
    reason = "this is the constructor the serialization ban redirects callers to"
)]
pub(crate) fn serialization_err(message: impl AsRef<str>) -> PyErr {
    let message = message.as_ref();
    attach_reason_from_message(CausalSerializationError::new_err(message.to_string()), message)
}

/// A refusal, as the registered Python class, carrying its `reason=<code>:` prefix.
///
/// Every refusal raised in this crate goes through here, so a caller catches one
/// class and reads one `reason_code` whether Rust or Python refused.
pub(crate) fn refusal(code: &str, message: impl AsRef<str>) -> PyErr {
    let prefix = antecedent_core::reason_code::PREFIX;
    unsupported_py_err(format!("{prefix}{code}: {}", message.as_ref()))
}

/// A refusal message with no registered reason code, as the Python class.
pub(crate) fn unsupported_message(message: impl AsRef<str>) -> PyErr {
    unsupported_py_err(message.as_ref().to_owned())
}

/// Build the Python exception for a refusal, carrying its `reason=<code>:` prefix.
pub(crate) fn unsupported_py_err(message: String) -> PyErr {
    // A support-matrix refusal renders its verdict first: `refused: reason=<code>: …`.
    let reason = reason_code_in_message(&message).map(str::to_string);
    Python::attach(|py| {
        let err: PyErr = UNSUPPORTED_ERROR_CLASS
            .get(py)
            .and_then(|cls| cls.bind(py).call1((message.as_str(),)).ok())
            .map_or_else(|| CausalUnsupportedError::new_err(message.clone()), PyErr::from_value);
        if let Some(code) = reason {
            let _ = err.value(py).setattr("reason_code", code.as_str());
        }
        err
    })
}

/// Attach a registered runtime-refusal reason code to a raised exception.
///
/// Callers pass a code checked by `antecedent_core::reason_code!`, or one read
/// back from a reason-coded Rust message, so an unregistered code is never
/// attached.
pub(crate) fn with_reason_code(err: PyErr, code: &str) -> PyErr {
    Python::attach(|py| {
        let _ = err.value(py).setattr("reason_code", code);
        err
    })
}

/// Attach a refusal's structured `remedy` (what the caller can change to proceed).
///
/// Additive: set only when the refusal names one, so every other exception reads the
/// `remedy = None` class default `antecedent.errors` installs on `CausalError`.
pub(crate) fn with_remedy(err: PyErr, remedy: Option<&str>) -> PyErr {
    if let Some(remedy) = remedy {
        Python::attach(|py| {
            let _ = err.value(py).setattr("remedy", remedy);
        });
    }
    err
}

/// Attach the route-native diagnostics a refusal carries (`route_diagnostics`).
///
/// Each record is a dict with `route`, `outcome`, `failed`, and `unevaluable_mass`. An
/// unreported mass is `None`, never `0.0`. Additive: set only when the refusal carries
/// records, so every other exception reads the empty-tuple class default.
pub(crate) fn with_route_diagnostics(
    err: PyErr,
    records: &[antecedent_core::diagnostic::RouteNativeDiagnostic],
) -> PyErr {
    if records.is_empty() {
        return err;
    }
    Python::attach(|py| {
        let rows: Vec<serde_json::Value> = records
            .iter()
            .map(|record| {
                serde_json::json!({
                    "route": record.route_name(),
                    "outcome": record.outcome_name(),
                    "failed": record.failed(),
                    "unevaluable_mass": record.unevaluable_mass().get(),
                })
            })
            .collect();
        if let Ok(list) = py.import("json").and_then(|json| {
            json.call_method1("loads", (serde_json::Value::from(rows).to_string(),))
        }) {
            let _ = err.value(py).setattr("route_diagnostics", list);
        }
    });
    err
}

/// [`with_reason_code`] unless the exception already carries a finer code.
pub(crate) fn with_default_reason_code(err: PyErr, code: &str) -> PyErr {
    Python::attach(|py| {
        let value = err.value(py);
        let present = value.getattr("reason_code").ok().is_some_and(|existing| !existing.is_none());
        if !present {
            let _ = value.setattr("reason_code", code);
        }
        err
    })
}

/// The Python `EffectNotIdentified` class, registered by `antecedent.errors`.
pub(crate) static NOT_IDENTIFIED_ERROR_CLASS: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

/// Register the Python class a not-identified refusal is raised as.
#[pyfunction]
pub(crate) fn set_not_identified_error_class(py: Python<'_>, cls: Py<PyAny>) {
    let _ = NOT_IDENTIFIED_ERROR_CLASS.get_or_init(py, || cls);
}

/// A not-identified refusal carrying its identification outcome.
pub(crate) fn not_identified_py_err(status: &str, search_capped: bool, message: String) -> PyErr {
    Python::attach(|py| {
        let err: PyErr = NOT_IDENTIFIED_ERROR_CLASS
            .get(py)
            .and_then(|cls| cls.bind(py).call1((message.as_str(),)).ok())
            .map_or_else(|| CausalCompileError::new_err(message.clone()), PyErr::from_value);
        let value = err.value(py);
        let _ =
            value.setattr("reason_code", antecedent_core::reason_code!("effect_not_identified"));
        let _ = value.setattr("identification_status", status);
        let _ = value.setattr("search_capped", search_capped);
        let _ = value.setattr("search_complete", !search_capped);
        err
    })
}

pub(crate) trait IntoCausalPyErr {
    fn into_antecedent_py_err(self) -> PyErr;
}

pub(crate) fn py_err<E: IntoCausalPyErr>(e: E) -> PyErr {
    e.into_antecedent_py_err()
}

/// A `usize` index that has to cross the binding as a `u32` id.
///
/// Failure is an input error. It is not saturated and not zero.
pub(crate) fn fit_u32(index: usize) -> PyResult<u32> {
    u32::try_from(index).map_err(|_| value_err(format!("index {index} does not fit in u32")))
}

/// Fallback for domain errors not covered by [`IntoCausalPyErr`].
///
/// Stays a root `CausalError` (these are not invalid-argument failures). A
/// registered `reason=<code>:` prefix in the message is copied onto `reason_code`.
pub(crate) fn py_msg(e: impl ToString) -> PyErr {
    let message = e.to_string();
    attach_reason_from_message(domain_err(&message), &message)
}

#[allow(
    clippy::disallowed_methods,
    reason = "root CausalError is the domain fallback; invalid input uses value_err"
)]
pub(crate) fn domain_err(message: &str) -> PyErr {
    CausalError::new_err(message.to_string())
}

/// Convert a Rust panic payload into a typed Python error so panics never cross FFI.
pub(crate) fn catch_ffi<F, T>(f: F) -> PyResult<T>
where
    F: FnOnce() -> PyResult<T>,
{
    match panic::catch_unwind(AssertUnwindSafe(f)) {
        Ok(result) => result,
        Err(payload) => Err(CausalError::new_err(format!(
            "internal Rust panic: {}",
            panic_payload_msg(payload.as_ref())
        ))),
    }
}

pub(crate) fn panic_payload_msg(payload: &(dyn Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic payload".into()
    }
}

/// Release the GIL for native work, convert any panic into [`CausalError`], and let Ctrl-C in the
/// calling thread cancel the run (see [`interrupt`]).
pub(crate) fn detach_catch<F, T>(py: Python<'_>, f: F) -> PyResult<T>
where
    F: FnOnce() -> PyResult<T> + Send,
    T: Send,
{
    interrupt::run(py, f)
}

/// The registered reason code a converted [`RustCausalError`] carries, if it has one.
///
/// The one owner of which native refusal raises which code: [`IntoCausalPyErr`] attaches
/// exactly this, so a caller reads `reason_code` off the exception and never parses the
/// message. A code the error names in its own `reason=<code>:` message wins over the
/// variant's default; an unregistered code is never attached. An error that carries a
/// structured callback failure names the code that failure carries, and its text is not read.
pub(crate) fn reason_code_of(error: &RustCausalError) -> Option<String> {
    use antecedent_core::reason_code;
    let named = if let Some(failure) = callback_failure_of(error) {
        failure.code().filter(|code| reason_code::is_registered(code)).map(str::to_string)
    } else {
        let message = error.to_string();
        reason_code_in_message(&message)
            .filter(|code| reason_code::is_registered(code))
            .map(str::to_string)
    };
    let default = match error.peeled() {
        RustCausalError::Graph(e) => Some(if matches!(e, GraphError::UnknownVariableName { .. }) {
            reason_code!("unknown_variable")
        } else {
            reason_code!("graph_invalid")
        }),
        RustCausalError::Schema(
            SchemaError::UnknownVariableName { .. } | SchemaError::UnknownVariableId { .. },
        ) => Some(reason_code!("unknown_variable")),
        RustCausalError::SchemaMismatch { .. } => Some(reason_code!("schema_mismatch")),
        RustCausalError::NotIdentified { .. } => Some(reason_code!("effect_not_identified")),
        RustCausalError::Identify(antecedent_identify::IdentificationError::UnsupportedQuery {
            ..
        }) => Some(reason_code!("route_not_supported")),
        RustCausalError::Unsupported { .. } => Some(reason_code!("route_not_supported")),
        // A matrix refusal without a finer registered code carries the code of its cell verdict.
        RustCausalError::Support { id, .. } => Some(if id.as_str() == "not_applicable" {
            reason_code!("cell_not_applicable")
        } else {
            reason_code!("cell_not_licensed")
        }),
        RustCausalError::Missing { .. } => Some(reason_code!("required_option_missing")),
        RustCausalError::Conflict { .. } => Some(reason_code!("invalid_argument")),
        _ => None,
    };
    named.or_else(|| default.map(str::to_string))
}

/// The structured callback failure a converted native error carries, if any.
pub(crate) fn callback_failure_of(
    error: &RustCausalError,
) -> Option<&antecedent_core::CallbackFailure> {
    antecedent_core::CallbackFailure::in_chain(error.peeled())
}

/// The registered-shape `reason=<code>:` a message opens with, directly or after one
/// `<label>: ` (a matrix refusal renders its verdict first: `refused: reason=<code>: ...`).
pub(crate) fn reason_code_in_message(message: &str) -> Option<&str> {
    antecedent_core::reason_code::split_prefix(message)
        .or_else(|| {
            message
                .split_once(": ")
                .and_then(|(_, rest)| antecedent_core::reason_code::split_prefix(rest))
        })
        .map(|(code, _)| code)
}

/// Copy a registered reason code out of `message` onto `err`, leaving the message as it is.
pub(crate) fn attach_reason_from_message(err: PyErr, message: &str) -> PyErr {
    match reason_code_in_message(message)
        .filter(|code| antecedent_core::reason_code::is_registered(code))
    {
        Some(code) => with_reason_code(err, code),
        None => err,
    }
}

impl IntoCausalPyErr for RustCausalError {
    fn into_antecedent_py_err(self) -> PyErr {
        let code = reason_code_of(&self);
        let failure = callback_failure_of(&self).cloned();
        let remedy = self.remedy().map(str::to_owned);
        let fields = self.refusal_fields().map(std::borrow::Cow::into_owned);
        let routes = self.route_diagnostics();
        let err = with_remedy(uncoded_py_err(self), remedy.as_deref());
        let err = preflight_api::with_refusal_fields(err, fields.as_ref());
        let err = with_route_diagnostics(err, &routes);
        let err = match &failure {
            Some(failure) => interrupt::with_callback_failure(err, failure),
            None => err,
        };
        match code {
            Some(code) => with_default_reason_code(err, &code),
            None => err,
        }
    }
}

/// The Python exception of an error's class, before its reason code is attached (see
/// [`reason_code_of`]).
pub(crate) fn uncoded_py_err(error: RustCausalError) -> PyErr {
    match error {
        // A refusal with structured fields is the refusal it wraps.
        RustCausalError::Diagnosed { error, .. } => uncoded_py_err(*error),
        RustCausalError::Identify(
            e @ antecedent_identify::IdentificationError::UnsupportedQuery { .. },
        ) => unsupported_py_err(e.to_string()),
        RustCausalError::Identify(e) => CausalIdentifyError::new_err(e.to_string()),
        // A reason-coded estimator refusal is the one refusal class, as a Rust
        // `Unsupported` refusal is.
        RustCausalError::Estimate(
            e @ (antecedent_estimate::EstimationError::Refused { .. }
            | antecedent_estimate::EstimationError::RefusedWithFields { .. }
            | antecedent_estimate::EstimationError::TargetPopulation),
        ) => unsupported_py_err(e.to_string()),
        // An inference mode the route does not perform is the one refusal class, however the
        // route spelled it; a refusal with any other route record stays an estimate error.
        RustCausalError::Estimate(
            e @ antecedent_estimate::EstimationError::Prob(antecedent_prob::ProbError::Route {
                ..
            }),
        ) if {
            let records = RustCausalError::route_diagnostics(&RustCausalError::Estimate(e.clone()));
            !records.is_empty() && records.iter().all(|record| record.is_unsupported_inference())
        } =>
        {
            unsupported_py_err(e.to_string())
        }
        RustCausalError::Estimate(e) => CausalEstimateError::new_err(e.to_string()),
        RustCausalError::Validate(e) => CausalValidateError::new_err(e.to_string()),
        RustCausalError::Discovery(e) => CausalDiscoveryError::new_err(e.to_string()),
        // A model or attribution refusal that names its reason code is the one refusal
        // class too.
        RustCausalError::Model(e @ antecedent_model::ModelError::Unsupported { .. })
            if reason_code_in_message(&e.to_string()).is_some() =>
        {
            unsupported_py_err(e.to_string())
        }
        RustCausalError::Model(e) => CausalModelError::new_err(e.to_string()),
        RustCausalError::Counterfactual(e) => CausalCounterfactualError::new_err(e.to_string()),
        RustCausalError::Attribution(e @ antecedent_attribution::AttributionError::Message(_))
            if reason_code_in_message(&e.to_string()).is_some() =>
        {
            unsupported_py_err(e.to_string())
        }
        RustCausalError::Attribution(e) => CausalAttributionError::new_err(e.to_string()),
        RustCausalError::Serialization(e) => crate::serialization_err(e.to_string()),
        RustCausalError::Data(e) => CausalDataError::new_err(e.to_string()),
        RustCausalError::Graph(e) => CausalGraphError::new_err(e.to_string()),
        RustCausalError::Design(e) => CausalDesignError::new_err(e.to_string()),
        RustCausalError::Callback { name, message } => {
            CausalDesignError::new_err(format!("callback {name}: {message}"))
        }
        RustCausalError::State(e) => match &e {
            antecedent::state::StateError::CacheBudget { .. } => {
                CausalResourceError::new_err(e.to_string())
            }
            _ => CausalStateError::new_err(e.to_string()),
        },
        // A structure that does not describe the table is a data problem, and
        // callers should be able to catch it as one rather than the root class.
        RustCausalError::Schema(e) => CausalDataError::new_err(e.to_string()),
        RustCausalError::SchemaMismatch { detail } => CausalDataError::new_err(detail),
        RustCausalError::Compile { message } => CausalCompileError::new_err(message),
        RustCausalError::Rejected(failure) => {
            let err = CausalCompileError::new_err(failure.message().to_owned());
            // The typed cause family stays observable: `error.cause_kind` is `graph`, `data`...
            Python::attach(|py| {
                let _ = err.value(py).setattr("cause_kind", failure.cause().kind());
            });
            err
        }
        RustCausalError::NotIdentified { status, search_capped, message } => {
            not_identified_py_err(status.as_str(), search_capped, message)
        }
        RustCausalError::Resource { message } => CausalResourceError::new_err(message),
        RustCausalError::ReviewRequired {
            kind,
            algorithm,
            pending_edge_count,
            pending_edges,
            message,
            hint,
        } => review_required_py_err(
            kind.as_str().to_owned(),
            algorithm,
            pending_edge_count,
            pending_edges,
            message,
            hint,
        ),
        RustCausalError::Unsupported { message } => unsupported_py_err(message.to_string()),
        RustCausalError::Support { id, message } => unsupported_py_err(format!("{id}: {message}")),
        RustCausalError::Missing { field } => {
            CausalCompileError::new_err(format!("missing required field: {field}"))
        }
        RustCausalError::Cancelled { stage } => {
            CausalCancelledError::new_err(format!("cancelled during {stage}"))
        }
        // `CausalError` is `#[non_exhaustive]`: any variant added upstream maps to the
        // hierarchy root rather than failing the build. Give new variants an explicit
        // arm above when their Python-facing category is decided.
        ref other => CausalError::new_err(other.to_string()),
    }
}

impl IntoCausalPyErr for DataError {
    fn into_antecedent_py_err(self) -> PyErr {
        RustCausalError::from(self).into_antecedent_py_err()
    }
}

impl IntoCausalPyErr for GraphError {
    fn into_antecedent_py_err(self) -> PyErr {
        RustCausalError::from(self).into_antecedent_py_err()
    }
}

impl IntoCausalPyErr for IoError {
    fn into_antecedent_py_err(self) -> PyErr {
        RustCausalError::from(self).into_antecedent_py_err()
    }
}

impl IntoCausalPyErr for SchemaError {
    fn into_antecedent_py_err(self) -> PyErr {
        RustCausalError::from(self).into_antecedent_py_err()
    }
}

impl IntoCausalPyErr for arrow_schema::ArrowError {
    fn into_antecedent_py_err(self) -> PyErr {
        CausalDataError::new_err(self.to_string())
    }
}

impl IntoCausalPyErr for antecedent_validate::ValidationError {
    fn into_antecedent_py_err(self) -> PyErr {
        RustCausalError::from(self).into_antecedent_py_err()
    }
}

impl IntoCausalPyErr for antecedent::state::StateError {
    fn into_antecedent_py_err(self) -> PyErr {
        RustCausalError::from(self).into_antecedent_py_err()
    }
}

impl IntoCausalPyErr for antecedent::estimate::EstimationError {
    fn into_antecedent_py_err(self) -> PyErr {
        RustCausalError::from(self).into_antecedent_py_err()
    }
}

impl IntoCausalPyErr for antecedent::gcm::ModelError {
    fn into_antecedent_py_err(self) -> PyErr {
        RustCausalError::from(self).into_antecedent_py_err()
    }
}

impl IntoCausalPyErr for antecedent::design::DesignError {
    fn into_antecedent_py_err(self) -> PyErr {
        RustCausalError::from(self).into_antecedent_py_err()
    }
}
