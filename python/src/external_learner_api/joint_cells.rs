//! Python learners on the compiled factorized joint-cell fit.
//!
//! [`joint_cells_learner_fit`] declares two registered Python providers as the per-cell outcome
//! and conditional-propensity learners and runs the compiled `JointCellsLearnerFit` program.
//! Each provider fit returns its model as a portable prediction map.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::compiled_joint_cells_learners::JointCellsLearnersRequest;
use antecedent::Study;
use pyo3::prelude::*;
use serde_json::json;

use super::{callback_context, smoothed_dose::portable_factories, PythonReceipt};

/// Run the compiled factorized joint-cell fit with two registered Python providers.
///
/// The declaration, columns and orderings are what `factorized_joint_cells_json` takes. The
/// built-in nuisance is not used: `outcome` fits each cell's outcome regression and `treatment`
/// each binary conditional. The JSON holds the cell family, the route, the lowered program
/// identity, the claim and the calibration status.
#[pyfunction]
#[pyo3(signature = (
    names, columns, declaration, outcome, adjustment, outcome_learner, treatment, *,
    identity, ordering=None, all_orderings=true, folds=5, seed=1, clip=0.01, min_cell_ess=10.0,
    normalization_tolerance=1e-9, ordering_tolerance_sd=0.05, cancel=None, memory_limit_bytes=None
))]
#[allow(
    clippy::too_many_arguments,
    reason = "the joint query and its learners are independent inputs"
)]
pub(super) fn joint_cells_learner_fit(
    py: Python<'_>,
    names: Vec<String>,
    columns: Vec<Bound<'_, PyAny>>,
    declaration: &str,
    outcome: &str,
    adjustment: Vec<String>,
    outcome_learner: Py<PyAny>,
    treatment: Py<PyAny>,
    identity: String,
    ordering: Option<Vec<String>>,
    all_orderings: bool,
    folds: usize,
    seed: u64,
    clip: f64,
    min_cell_ess: f64,
    normalization_tolerance: f64,
    ordering_tolerance_sd: f64,
    cancel: Option<crate::PyCancellationToken>,
    memory_limit_bytes: Option<u64>,
) -> PyResult<(String, Vec<PythonReceipt>)> {
    let (learners, state) = portable_factories(py, outcome_learner, treatment, identity)?;
    let ctx = callback_context(seed, cancel, memory_limit_bytes)?;
    let parts = crate::derived_treatment_api::prepare_joint_caller(
        names,
        columns,
        declaration,
        outcome,
        adjustment,
        ordering,
        all_orderings,
        folds,
        seed,
        clip,
        min_cell_ess,
        normalization_tolerance,
        ordering_tolerance_sd,
        &ctx,
    )?;
    let treatment_names = parts.treatment_names.clone();
    let result = py.detach(|| {
        let compiled = Study::compile_joint_cells_learners(JointCellsLearnersRequest::new(
            parts.data,
            parts.treatments,
            parts.outcome,
            parts.adjustment,
            parts.orderings,
            parts.config,
            learners,
        ))
        .map_err(crate::transport_common::error)?;
        let run = compiled.execute(&ctx).map_err(crate::transport_common::error)?;
        let mut fit = crate::derived_treatment_api::joint_fit_json(&run.fit);
        fit["treatments"] = json!(treatment_names);
        fit["external_nuisances_unverified"] = json!(run.fit.external_nuisances_unverified);
        Ok(serde_json::json!({
            "fit": fit,
            "route": compiled.plan_view().route().as_str(),
            "program_identity": run.lowered_program_identity.to_hex(),
            "claim": run.claim,
            "calibration": run.calibration,
        })
        .to_string())
    });
    state.finish(result)
}
