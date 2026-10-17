//! Python batch callbacks for intervention programs.
//!
//! A callback is a Python function of numpy arrays called once per batch by the shared
//! program executor, with the GIL held only for the call. The binding is a runtime
//! binding: it executes in this process and has no durable form unless it was declared
//! from self-contained source (see `antecedent.intervention_callbacks`), in which case it
//! carries a recovery record the consumer re-verifies with
//! [`verify_callback_recovery`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    BatchCallback, CallbackError, CallbackImplementation, CallbackOutput, CallbackSpec,
    PinnedDependency, Portability, RecoveryRecord, ValueExpr, VariableId,
};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::interrupt::callback_failure;
use crate::{refusal, value_err};

/// A Python function behind the [`BatchCallback`] interface.
#[derive(Debug)]
struct PyBatchCallback {
    spec: CallbackSpec,
    function: Py<PyAny>,
}

impl BatchCallback for PyBatchCallback {
    fn spec(&self) -> &CallbackSpec {
        &self.spec
    }

    fn call(&self, inputs: &[&[f64]], out: &mut [f64]) -> Result<(), CallbackError> {
        Python::attach(|py| {
            let not_a_column = |found: String| CallbackError::NotRealValued { found };
            let columns: Vec<Bound<'_, PyAny>> =
                inputs.iter().map(|column| PyArray1::from_slice(py, column).into_any()).collect();
            let arguments = PyTuple::new(py, columns)
                .map_err(|error| CallbackError::Raised { message: error.to_string() })?;
            let returned =
                self.function.bind(py).call1(arguments).map_err(|error| CallbackError::Raised {
                    message: callback_failure(
                        &format!("callback {} raised", self.spec.name),
                        error,
                    ),
                })?;
            let array = py
                .import("numpy")
                .and_then(|numpy| numpy.call_method1("asarray", (returned,)))
                .map_err(|error| not_a_column(error.to_string()))?;
            let kind: String = array
                .getattr("dtype")
                .and_then(|dtype| dtype.getattr("kind"))
                .and_then(|kind| kind.extract())
                .map_err(|error| not_a_column(error.to_string()))?;
            if !matches!(kind.as_str(), "f" | "i" | "u" | "b") {
                let dtype = array.getattr("dtype").map_or_else(|_| kind.clone(), |d| d.to_string());
                return Err(not_a_column(format!("an array of dtype {dtype}")));
            }
            let ndim: usize = array
                .getattr("ndim")
                .and_then(|ndim| ndim.extract())
                .map_err(|error| not_a_column(error.to_string()))?;
            if ndim != 1 {
                let shape =
                    array.getattr("shape").map_or_else(|_| String::new(), |s| s.to_string());
                return Err(not_a_column(format!("an array of shape {shape}")));
            }
            let real = array
                .call_method1("astype", ("float64",))
                .map_err(|error| not_a_column(error.to_string()))?;
            let column = real
                .extract::<PyReadonlyArray1<'_, f64>>()
                .map_err(|error| not_a_column(error.to_string()))?;
            let values = column
                .as_slice()
                .map_err(|_| not_a_column("a non-contiguous array".to_string()))?;
            if values.len() != out.len() {
                return Err(CallbackError::WrongLength { got: values.len(), expected: out.len() });
            }
            out.copy_from_slice(values);
            Ok(())
        })
    }
}

fn output_of(kind: &str, params: &[f64]) -> PyResult<CallbackOutput> {
    match (kind, params) {
        ("real", []) => Ok(CallbackOutput::Real),
        ("range", [lower, upper]) => Ok(CallbackOutput::Range { lower: *lower, upper: *upper }),
        ("levels", levels) if !levels.is_empty() => {
            Ok(CallbackOutput::levels(levels.iter().copied()))
        }
        _ => Err(value_err(format!(
            "output {kind:?} with {} parameters is not a declaration; use real, range \
             (lower, upper) or levels (one or more)",
            params.len()
        ))),
    }
}

fn output_parts(output: &CallbackOutput) -> (&'static str, Vec<f64>) {
    match output {
        CallbackOutput::Range { lower, upper } => ("range", vec![*lower, *upper]),
        CallbackOutput::Levels(levels) => ("levels", levels.to_vec()),
        _ => ("real", Vec::new()),
    }
}

fn spec_of(
    name: String,
    inputs: Vec<String>,
    output_kind: &str,
    output_params: &[f64],
    dependencies: Vec<(String, String)>,
    source_sha256: Option<String>,
    entry_point: Option<String>,
) -> PyResult<CallbackSpec> {
    let implementation = match (source_sha256, entry_point) {
        (None, None) => CallbackImplementation::Runtime,
        (Some(source_sha256), Some(entry_point)) => {
            CallbackImplementation::Recoverable { source_sha256, entry_point }
        }
        _ => {
            return Err(value_err(
                "a recoverable callback needs both source_sha256 and entry_point",
            ));
        }
    };
    let spec = CallbackSpec {
        name,
        inputs: inputs.into(),
        output: output_of(output_kind, output_params)?,
        dependencies: dependencies
            .into_iter()
            .map(|(name, version)| PinnedDependency { name, version })
            .collect(),
        implementation,
    };
    if spec.is_well_formed() {
        Ok(spec)
    } else {
        Err(value_err(
            "callback declaration is malformed: it needs a name, distinct input names, a \
             well-formed output and, when recoverable, a 64-digit SHA-256 and an entry point",
        ))
    }
}

/// A Python function declared as a batch callback for intervention programs.
#[pyclass(name = "BatchCallbackHandle", frozen)]
pub struct PyBatchCallbackHandle {
    inner: Arc<PyBatchCallback>,
}

#[pymethods]
impl PyBatchCallbackHandle {
    #[new]
    #[pyo3(signature = (function, name, inputs, output_kind, output_params, dependencies, source_sha256=None, entry_point=None))]
    #[allow(clippy::too_many_arguments, reason = "the flat declaration the facade passes through")]
    fn new(
        function: Bound<'_, PyAny>,
        name: String,
        inputs: Vec<String>,
        output_kind: &str,
        output_params: Vec<f64>,
        dependencies: Vec<(String, String)>,
        source_sha256: Option<String>,
        entry_point: Option<String>,
    ) -> PyResult<Self> {
        if !function.is_callable() {
            return Err(value_err("a batch callback must be callable"));
        }
        let spec = spec_of(
            name,
            inputs,
            output_kind,
            &output_params,
            dependencies,
            source_sha256,
            entry_point,
        )?;
        Ok(Self { inner: Arc::new(PyBatchCallback { spec, function: function.unbind() }) })
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.spec.name.clone()
    }

    #[getter]
    fn inputs(&self) -> Vec<String> {
        self.inner.spec.inputs.to_vec()
    }

    /// Whether the implementation is recoverable (declared from self-contained source).
    #[getter]
    fn recoverable(&self) -> bool {
        self.inner.spec.recovery_record().is_some()
    }

    /// The recovery record fields, or `None` for a runtime binding.
    fn recovery_record<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(record) = self.inner.spec.recovery_record() else {
            return Ok(None);
        };
        let out = PyDict::new(py);
        out.set_item("name", &record.name)?;
        out.set_item("inputs", record.inputs.to_vec())?;
        let (kind, params) = output_parts(&record.output);
        out.set_item("output_kind", kind)?;
        out.set_item("output_params", params)?;
        out.set_item("source_sha256", &record.source_sha256)?;
        out.set_item("entry_point", &record.entry_point)?;
        let dependencies: Vec<(String, String)> =
            record.dependencies.iter().map(|d| (d.name.clone(), d.version.clone())).collect();
        out.set_item("dependencies", dependencies)?;
        Ok(Some(out))
    }
}

/// The refusal for exporting these callbacks, as `(reason_code, message)`, or `None` when
/// every one is recoverable.
#[pyfunction]
pub fn callback_export_refusal(
    callbacks: Vec<PyRef<'_, PyBatchCallbackHandle>>,
) -> Option<(String, String)> {
    let mut portability = Portability::Portable;
    for handle in &callbacks {
        // Portability depends on the callback's declaration, not on its arguments.
        let call = ValueExpr::callback(
            Arc::clone(&handle.inner) as Arc<dyn BatchCallback>,
            Vec::<ValueExpr>::new(),
        );
        portability = portability.and(call.portability());
    }
    portability.export_refusal().map(|r| (r.code.to_string(), r.message))
}

/// Verify a recovery record against the callback supplied for replay.
///
/// # Errors
///
/// A `policy_callback_unrecoverable` refusal naming the first differing field.
#[pyfunction]
#[pyo3(signature = (name, inputs, output_kind, output_params, source_sha256, entry_point, dependencies, supplied))]
#[allow(clippy::too_many_arguments, reason = "the flat record the facade passes through")]
pub fn verify_callback_recovery(
    name: String,
    inputs: Vec<String>,
    output_kind: &str,
    output_params: Vec<f64>,
    source_sha256: String,
    entry_point: String,
    dependencies: Vec<(String, String)>,
    supplied: PyRef<'_, PyBatchCallbackHandle>,
) -> PyResult<()> {
    let record = RecoveryRecord {
        name,
        inputs: inputs.into(),
        output: output_of(output_kind, &output_params)?,
        source_sha256,
        entry_point,
        dependencies: dependencies
            .into_iter()
            .map(|(name, version)| PinnedDependency { name, version })
            .collect(),
    };
    record.verify(&supplied.inner.spec).map_err(|mismatch| {
        let r = mismatch.refusal();
        refusal(r.code, r.message)
    })
}

/// Intervention for one callback program: `kind` is `"assign"` or `"transform"`, `args`
/// are the expressions passed to the callback, one per declared input.
pub(crate) fn program_intervention(
    kind: &str,
    variable: VariableId,
    callback: &PyBatchCallbackHandle,
    args: Vec<ValueExpr>,
) -> PyResult<antecedent_core::Intervention> {
    let expr = ValueExpr::callback(Arc::clone(&callback.inner) as Arc<dyn BatchCallback>, args);
    match kind {
        "assign" => Ok(antecedent_core::Intervention::assign(variable, expr)),
        "transform" => Ok(antecedent_core::Intervention::transform(variable, expr)),
        other => Err(value_err(format!("program mode {other:?} is not assign or transform"))),
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBatchCallbackHandle>()?;
    m.add_function(wrap_pyfunction!(callback_export_refusal, m)?)?;
    m.add_function(wrap_pyfunction!(verify_callback_recovery, m)?)
}
