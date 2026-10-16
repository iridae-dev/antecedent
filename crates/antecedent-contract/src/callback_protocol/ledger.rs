//! Callbacks as evidence-ledger sources.
//!
//! A callback is an external claim that rests on evidence: its own declared
//! sources and, for a nested callback, the callbacks it consumes. Registering
//! it makes that overlap computable by the one runtime ledger rather than
//! asserted: two callbacks that nest the same inner callback share it, and a
//! callback whose evidence is undeclared overlaps with unknown dependence,
//! never independence.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use super::declaration::{CallbackDeclaration, InputSupplier};
use super::refusal::{CallbackRefusal, request_mismatch};
use crate::evidence_ledger::{Lineage, SourceDeclaration, SourceId, SourceKind};
use crate::interface_contract::EvidenceBasis;

/// The ledger label of a callback source: its declaration identity.
#[must_use]
pub fn callback_source_label(declaration: &CallbackDeclaration) -> String {
    format!("callback:{}", declaration.canonical_identity().to_hex())
}

/// The ledger declaration of a callback.
///
/// `inner_sources` maps the declaration identity (hex) of each inner callback
/// to the ledger source it was registered as. The source is derived from the
/// callback's own evidence sources and its inner callbacks; a callback whose
/// own evidence is undeclared is registered as first-hand with undeclared
/// observations, so its overlap with anything is unknown.
///
/// # Errors
/// A nested callback input whose inner callback was not registered.
pub fn callback_source_declaration(
    declaration: &CallbackDeclaration,
    inner_sources: &BTreeMap<String, SourceId>,
) -> Result<SourceDeclaration, CallbackRefusal> {
    let label = callback_source_label(declaration);
    let source = SourceDeclaration::new(label, SourceKind::ExternalClaim);
    let EvidenceBasis::Declared(own) = &declaration.evidence else {
        return Ok(source);
    };
    let mut parents: BTreeSet<SourceId> = own.clone();
    for input in &declaration.inputs {
        if let InputSupplier::Callback { identity, .. } = &input.supplier {
            let Some(id) = inner_sources.get(identity) else {
                return Err(request_mismatch(&input.name));
            };
            parents.insert(*id);
        }
    }
    if parents.is_empty() {
        // No data and no inner callback: a law with nothing behind it overlaps
        // nothing.
        return Ok(source.with_observations(std::iter::empty()));
    }
    Ok(source.with_lineage(Lineage::DerivedFromUsedData { inputs: parents.into_iter().collect() }))
}
