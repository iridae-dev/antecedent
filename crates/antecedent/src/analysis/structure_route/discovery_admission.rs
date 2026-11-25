//! Admit a finished discovery as the graph a downstream study reads.
//!
//! The admission calls [`accept_discovered`] and checks the recorded data
//! snapshot against the caller's table. The table is not stored on the graph.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::ClaimEnvelope;
use antecedent_core::ExecutionContext;

use super::CompiledExecution;
use super::acceptance::accept_discovered;
use crate::accepted::AcceptedGraph;
use crate::analysis::builder::DataInput;
use crate::error::CausalError;

/// Admit the structure `execution` already returned.
///
/// `caller` is the table the downstream study will read. Its snapshot is
/// computed the way the discovery study stamped its own, and compared with the
/// snapshot [`accept_discovered`] records. A different table is refused. The
/// returned graph carries that snapshot's digest in its provenance and does
/// not retain `caller`.
///
/// # Errors
///
/// The refusals of [`accept_discovered`], and a snapshot that is not the
/// caller's table.
pub(crate) fn admit_finished_discovery(
    execution: &CompiledExecution,
    caller: &DataInput,
    accept_marks: bool,
    ctx: &ExecutionContext,
) -> Result<AcceptedGraph, CausalError> {
    let (_, claim) = execution.contract_and_claim(ctx)?;
    admit_discovered_result(execution, &claim, caller, accept_marks)
}

/// [`accept_discovered`] on `execution`'s result, then the snapshot check.
fn admit_discovered_result(
    execution: &CompiledExecution,
    claim: &ClaimEnvelope,
    caller: &DataInput,
    accept_marks: bool,
) -> Result<AcceptedGraph, CausalError> {
    let accepted = accept_discovered(execution.result(), claim, accept_marks)?;
    let recorded = accepted.provenance().ok_or(CausalError::Conflict {
        what: "result",
        detail: "the accepted discovery carries no structure provenance",
    })?;
    let prepared = execution.prepared();
    let stamped = prepared.executed_contract(caller, prepared.study().settings.refute, None)?;
    let caller_snapshot = *stamped.identities.data_snapshot.as_bytes();
    if caller_snapshot != recorded.data_snapshot {
        return Err(CausalError::Conflict {
            what: "data_snapshot",
            detail: "the discovery read a different table than this study; pass the table the \
                     search ran on",
        });
    }
    Ok(accepted)
}
