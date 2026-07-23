//! Physical-plan marking for Python / slow-path callbacks.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    Diagnostic, DiagnosticKind, DiagnosticSeverity, KernelSelection, ParallelTaskSpec,
    PhysicalExecutionPlanRecord,
};

/// Force serial execution and record a callback region on a physical plan.
#[must_use]
pub fn mark_python_callback_plan(
    mut record: PhysicalExecutionPlanRecord,
    region: &str,
) -> (PhysicalExecutionPlanRecord, Diagnostic) {
    record.worker_threads = 0;
    record.task_schedule =
        Arc::from([ParallelTaskSpec { dimension: Arc::from("serial"), units: 1 }]);
    record.expected_python_crossings = record.expected_python_crossings.saturating_add(1);
    let mut kernels = record.kernels.as_ref().to_vec();
    kernels.push((Arc::from(format!("python.callback.{region}")), KernelSelection::Scalar));
    record.kernels = Arc::from(kernels);
    let diagnostic = Diagnostic::new(
        "exec.python_callback_serial",
        DiagnosticKind::Execution,
        DiagnosticSeverity::Info,
        format!("Python callback region `{region}` forced serial execution"),
    );
    (record, diagnostic)
}
