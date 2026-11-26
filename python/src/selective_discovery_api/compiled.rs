//! Selective discovery resumed through the compiled structure-discovery route.
//!
//! The study searches with the contract's selective search, resumed from the artifact's portable
//! state (`DiscoverySpec::selective_static` / `selective_temporal`, then `resuming`), and runs as
//! the DiscoverStatic or DiscoverTemporal program of its algorithm. The adapter it runs is the
//! one the direct resume runs, so the outcome and the run report are the same.
//
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{CompiledExecution, DiscoverySpec, Study};
use pyo3::prelude::*;

use super::{
    artifact_refusal, contract_map, get_u64, kind_of, outcome_json, parse_json, report_json,
    static_contract, temporal_contract, NumericColumn, SelectiveDiscoveryArtifact,
};
use crate::{py_err, value_err};

/// `(outcome_json, report_json, program, refusal)` of a compiled resume.
type CompiledResume = (Option<String>, Option<String>, Option<String>, Option<String>);

/// Resume a selective search from artifact bytes through the compiler.
///
/// An unusable artifact is returned as a refusal, as the direct resume returns it; a refusal of
/// the search itself (another contract, other data, a diverging replay) is raised as the
/// compiler's error.
#[pyfunction]
#[pyo3(signature = (kind, contract_json, names, columns, artifact, expected_identity=None, *, threads=None, cancel=None))]
#[allow(clippy::too_many_arguments, reason = "one resume request")]
fn selective_discovery_compiled_resume(
    py: Python<'_>,
    kind: &str,
    contract_json: &str,
    names: Vec<String>,
    columns: Vec<NumericColumn<'_>>,
    artifact: &[u8],
    expected_identity: Option<&str>,
    threads: Option<u32>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<CompiledResume> {
    let kind = kind_of(kind)?;
    let parsed = parse_json(contract_json)?;
    let map = contract_map(&parsed, kind)?;
    let seed = get_u64(map, "seed")?;
    let state = match SelectiveDiscoveryArtifact::from_bytes(artifact, expected_identity) {
        Ok(a) => a.state().clone(),
        Err(error) => return Ok((None, None, None, Some(artifact_refusal(&error)))),
    };
    let n = names.len();
    let spec = if kind == "static" {
        DiscoverySpec::selective_static(static_contract(map, n)?)
    } else {
        DiscoverySpec::selective_temporal(temporal_contract(map, n)?)
    };
    let spec = spec.and_then(|spec| spec.resuming(state)).map_err(py_err)?;
    let batch = crate::numeric_columns_to_batch(&names, &columns)?;
    drop(columns);
    let ctx =
        crate::py_execution_context_cancel(seed, crate::resolve_user_threads(threads), cancel);
    let tabular = kind == "static";
    let (found, program) = crate::detach_catch(py, move || {
        let study = if tabular {
            let (table, variables) = crate::discovery_api::tabular_from_batch(&batch)?;
            Study::tabular(table).discovery(spec, variables).build().map_err(py_err)?
        } else {
            let (series, variables) = crate::discovery_api::series_from_batch(&batch)?;
            Study::series(series).discovery(spec, variables).build().map_err(py_err)?
        };
        let execution = CompiledExecution::run(&study, &ctx).map_err(py_err)?;
        let program = execution.lowered_identity().map(|identity| identity.to_hex());
        match execution.result().discovery().map(|payload| payload.structure.clone()) {
            Some(antecedent::DiscoveredStructure::Selective(found)) => Ok((*found, program)),
            _ => Err(value_err("the compiled search returned no selective outcome")),
        }
    })?;
    Ok((Some(outcome_json(&found.outcome)), Some(report_json(&found.report)), program, None))
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(selective_discovery_compiled_resume, module)?)?;
    Ok(())
}
