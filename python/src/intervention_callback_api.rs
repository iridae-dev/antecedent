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
    Intervention, PinnedDependency, Portability, RecoveryRecord, ValueExpr, VariableId,
};
use antecedent_io::policy_program_artifact as artifact;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyTuple};

use crate::interrupt::callback_failure;
use crate::intervention_program_api::{index_resolver, io_err};
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

/// One program of a mixed list, as Python hands it over: a closed-form program (its
/// one-element program text) or a callback program `(mode, variable, callback, reads)` with
/// `reads` naming variables (`None` is the node's own generated value).
type ProgramItem =
    (Option<String>, Option<(String, String, Py<PyBatchCallbackHandle>, Vec<Option<String>>)>);

/// Encode a list of closed-form and callback programs, in order, as a versioned portable
/// artifact. Every callback must be recoverable: the artifact then carries its recovery record
/// and never its code, under an encoding version that only a program with callbacks uses.
///
/// # Errors
///
/// `policy_callback_not_portable` for a runtime-bound callback, or a malformed program.
#[pyfunction]
pub fn encode_callback_policy_program(
    py: Python<'_>,
    names: Vec<String>,
    items: Vec<ProgramItem>,
) -> PyResult<Bound<'_, PyBytes>> {
    let resolve = index_resolver(&names);
    let mut interventions: Vec<Intervention> = Vec::with_capacity(items.len());
    for (closed, callback) in &items {
        match (closed, callback) {
            (Some(text), None) => interventions
                .extend(artifact::interventions_from_named_json(text, &resolve).map_err(io_err)?),
            (None, Some((mode, variable, handle, reads))) => {
                let id = resolve(variable)
                    .map(VariableId::from_raw)
                    .ok_or_else(|| value_err(format!("unknown variable {variable:?}")))?;
                let args = reads
                    .iter()
                    .map(|read| match read {
                        Some(name) => resolve(name)
                            .map(|raw| ValueExpr::var(VariableId::from_raw(raw)))
                            .ok_or_else(|| value_err(format!("unknown variable {name:?}"))),
                        None => Ok(ValueExpr::generated()),
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                interventions.push(program_intervention(mode, id, handle.get(), args)?);
            }
            _ => return Err(value_err("a program item is either closed-form or a callback")),
        }
    }
    let bytes = artifact::encode_policy_program(&interventions).map_err(io_err)?;
    Ok(PyBytes::new(py, &bytes))
}

/// The recovery records an encoded program carries, in the order its callbacks run: what a
/// consumer must supply implementations for. The artifact's version, identity and validity are
/// checked first.
#[pyfunction]
pub fn policy_program_callback_records(py: Python<'_>, data: &[u8]) -> PyResult<Vec<Py<PyDict>>> {
    artifact::policy_program_callback_records(data)
        .map_err(io_err)?
        .iter()
        .map(|record| {
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
            Ok(out.unbind())
        })
        .collect()
}

/// Decode an encoded program that may carry callbacks, binding each recovery record to the
/// supplied callback of that name and re-verifying it against the record first.
///
/// Returns the programs in order as `("closed", program_text, None, None, None)` or
/// `("callback", mode, variable, callback_name, reads)`.
///
/// # Errors
///
/// The artifact's own refusals, and `policy_callback_unrecoverable` for a record with no
/// supplied callback or one that does not match.
#[pyfunction]
#[allow(clippy::type_complexity, reason = "the flat tuple the Python facade unpacks")]
pub fn decode_callback_policy_program(
    names: Vec<String>,
    data: &[u8],
    supplied: Vec<PyRef<'_, PyBatchCallbackHandle>>,
) -> PyResult<Vec<(String, String, Option<String>, Option<String>, Option<Vec<Option<String>>>)>> {
    let by_name: Vec<(&str, Arc<dyn BatchCallback>)> = supplied
        .iter()
        .map(|h| (h.inner.spec.name.as_str(), Arc::clone(&h.inner) as Arc<dyn BatchCallback>))
        .collect();
    let interventions = artifact::decode_policy_program_with(data, &|record| {
        by_name.iter().find(|(name, _)| *name == record.name).map(|(_, cb)| Arc::clone(cb))
    })
    .map_err(io_err)?;
    let spell = |id: VariableId| {
        names
            .get(id.raw() as usize)
            .cloned()
            .ok_or_else(|| value_err(format!("variable id {} has no name", id.raw())))
    };
    interventions
        .iter()
        .map(|iv| {
            let callback_program = match iv {
                Intervention::Transform { variable, expr: ValueExpr::Callback(call), .. } => {
                    Some(("transform", variable, call))
                }
                Intervention::Assign { variable, expr: ValueExpr::Callback(call), .. } => {
                    Some(("assign", variable, call))
                }
                _ => None,
            };
            if let Some((mode, variable, call)) = callback_program {
                let reads = call
                    .args()
                    .iter()
                    .map(|arg| match arg {
                        ValueExpr::Var(id) => spell(*id).map(Some),
                        ValueExpr::Generated => Ok(None),
                        _ => Err(value_err(format!(
                            "callback {} is applied to an expression; Python programs apply a \
                             callback to variables or the generated value only",
                            call.spec().name
                        ))),
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                return Ok((
                    "callback".to_string(),
                    mode.to_string(),
                    Some(spell(*variable)?),
                    Some(call.spec().name.clone()),
                    Some(reads),
                ));
            }
            if matches!(iv.portability(), Portability::Recoverable(_)) {
                return Err(value_err(
                    "a callback nested inside a larger expression has no Python program form",
                ));
            }
            let text = artifact::named_json_from_interventions(std::slice::from_ref(iv), &names)
                .map_err(io_err)?;
            Ok(("closed".to_string(), text, None, None, None))
        })
        .collect()
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBatchCallbackHandle>()?;
    m.add_function(wrap_pyfunction!(encode_callback_policy_program, m)?)?;
    m.add_function(wrap_pyfunction!(policy_program_callback_records, m)?)?;
    m.add_function(wrap_pyfunction!(decode_callback_policy_program, m)?)?;
    m.add_function(wrap_pyfunction!(callback_export_refusal, m)?)?;
    m.add_function(wrap_pyfunction!(verify_callback_recovery, m)?)
}
