//! Observe original synchronous native component work without issuing scientific authority.
use antecedent::analysis::recalc_attempt::{Operation, execute_with_attempt};
use pyo3::prelude::*;
use pyo3::types::PyAny;
use std::cell::Cell;

#[derive(Clone, Copy, Default)]
struct InputCopies {
    count: u64,
    bytes: u64,
}

thread_local! {
    static INPUT_COPIES: Cell<Option<InputCopies>> = const { Cell::new(None) };
}

// NumPy snapshots run synchronously while attached to Python, before their
// owning buffers can be handed to a native worker.
pub(crate) fn record_input_copy(bytes: u64) {
    INPUT_COPIES.with(|cell| {
        if let Some(totals) = cell.get() {
            cell.set(Some(InputCopies { count: totals.count + 1, bytes: totals.bytes + bytes }));
        }
    });
}

struct InputCopyScope(Option<InputCopies>);

impl InputCopyScope {
    fn enter() -> Self {
        Self(INPUT_COPIES.with(|cell| cell.replace(Some(InputCopies::default()))))
    }

    fn totals() -> InputCopies {
        INPUT_COPIES.with(|cell| cell.get().expect("input copy scope is active"))
    }
}

impl Drop for InputCopyScope {
    fn drop(&mut self) {
        let totals = Self::totals();
        INPUT_COPIES.with(|cell| {
            cell.set(self.0.map(|outer| InputCopies {
                count: outer.count + totals.count,
                bytes: outer.bytes + totals.bytes,
            }));
        });
    }
}

fn label(operation: Operation) -> &'static str {
    match operation {
        Operation::Identification => "identification",
        Operation::LeastSquaresSolve => "least_squares_solve",
        Operation::GlmFit => "glm_fit",
        Operation::AdjustedFit => "adjusted_fit",
        Operation::LearnerFit => "learner_fit",
        Operation::PosteriorFit => "posterior_fit",
        Operation::PosteriorDraw => "posterior_draw",
        Operation::ScoreConstruction => "score_construction",
        Operation::FactorConstruction => "factor_construction",
        Operation::ProgramCompilation => "program_compilation",
        Operation::ProviderBinding => "provider_binding",
        Operation::FactorEvaluation => "factor_evaluation",
        Operation::Integration => "integration",
        Operation::ProviderInvocation => "provider_invocation",
        Operation::LawSummary => "law_summary",
        Operation::Decision => "decision",
        Operation::CachedFailureRead => "cached_failure_read",
        Operation::PriorConstruction => "prior_construction",
    }
}
type ObservationPayload = (Option<Py<PyAny>>, Option<Py<PyAny>>, String);

/// The unchanged original value or exception and separate fixed-kind diagnostic observation.
#[pyfunction]
fn observe_native_attempts(
    py: Python<'_>,
    operation: &Bound<'_, PyAny>,
) -> PyResult<ObservationPayload> {
    crate::catch_ffi(|| {
        if !operation.is_callable() {
            return Err(crate::with_reason_code(
                crate::value_err(
                    "native_attempt.invalid_callable: a synchronous zero-argument callable is required",
                ),
                antecedent::reason::reason_code!("invalid_argument"),
            ));
        }
        // Execute once. Original callbacks, cancellation and transaction policies remain in force.
        let scope = InputCopyScope::enter();
        let execution = execute_with_attempt(|| operation.call0().map(Bound::unbind));
        let input = InputCopyScope::totals();
        drop(scope);
        let rows = execution
            .report
            .operations()
            .map(|(kind, counts)| {
                (
                    label(kind),
                    serde_json::json!({
                        "attempted":counts.attempted,"completed":counts.completed,
                        "failed":counts.failed,"unfinished":counts.unfinished(),
                    }),
                )
            })
            .collect::<std::collections::BTreeMap<_, _>>();
        let report = serde_json::json!({
            "scope":"synchronous_original_native_components",
            "operations":rows,
            "copies":antecedent::execution::copies(&execution.report),
            "copied_bytes":antecedent::execution::copied_bytes(&execution.report),
            "input_copies":input.count,
            "input_copied_bytes":input.bytes,
            "is_empty":execution.report.is_empty(),
            "is_complete":execution.report.is_complete(),
        })
        .to_string();
        let (value, error) = match execution.result {
            Ok(value) => (Some(value), None),
            Err(error) => (None, Some(error.into_value(py).into_any())),
        };
        Ok((value, error, report))
    })
}
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(observe_native_attempts, module)?)
}
