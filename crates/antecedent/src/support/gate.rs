//! The facade's conversion from the shared `not_supported` refusal, and the gate an entry point
//! of a capability unit outside the support matrix passes through.
//!
//! A route listed under a capability record (`parity/capability_records.toml`) has no support
//! status of its own: it runs as an entry point of the unit that owns it. The unit's status is
//! derived from the support model by `scripts/generate_support_matrix_docs.py` into
//! `antecedent-io-core/data/unit_support_status.json`, which this build embeds.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::OnceLock;

use antecedent_core::{ExecutionContext, NotSupported};

use crate::error::CausalError;

use super::cell::SupportRefusal;

impl From<NotSupported> for CausalError {
    fn from(refusal: NotSupported) -> Self {
        Self::Support { id: SupportRefusal::NotSupported, message: refusal.message() }
    }
}

/// One capability unit's derived status.
#[derive(serde::Deserialize)]
struct CapabilityUnit {
    id: String,
    status: String,
    #[serde(default)]
    missing_evidence: Vec<String>,
}

#[derive(serde::Deserialize)]
struct UnitStatus {
    capabilities: Vec<CapabilityUnit>,
}

/// The embedded capability units, or `None` when the snapshot does not decode, so that every
/// gated route refuses rather than runs on a misread status.
fn capability_units() -> Option<&'static [CapabilityUnit]> {
    static UNITS: OnceLock<Option<Vec<CapabilityUnit>>> = OnceLock::new();
    UNITS
        .get_or_init(|| {
            serde_json::from_str::<UnitStatus>(antecedent_io_core::graphless_support::UNIT_STATUS)
                .ok()
                .map(|status| status.capabilities)
        })
        .as_deref()
}

/// Whether the capability unit `capability` is supported in this build's derived status.
#[must_use]
pub(crate) fn capability_supported(capability: &str) -> bool {
    capability_units().is_some_and(|units| {
        units.iter().any(|unit| unit.id == capability && unit.status == "supported")
    })
}

/// The derived status of the capability unit `capability`: whether it is supported and the
/// missing evidence it lists. `None` when the snapshot does not decode or names no such unit.
#[must_use]
pub(super) fn capability_status(capability: &str) -> Option<(bool, Vec<&'static str>)> {
    let unit = capability_units()?.iter().find(|unit| unit.id == capability)?;
    Some((unit.status == "supported", unit.missing_evidence.iter().map(String::as_str).collect()))
}

/// Refuse the entry point `route` of the capability unit `capability` when that unit is not
/// supported.
///
/// Release: [`NotSupported`] naming `route`, with the unit's missing evidence (or the unit
/// itself when it lists none). Development: records what a release build would refuse in the
/// log every clone of `ctx` shares, and the route runs.
///
/// # Errors
///
/// [`NotSupported`] in a release build when `capability` is not a supported unit.
pub(crate) fn require_capability_route(
    ctx: &ExecutionContext,
    capability: &'static str,
    route: &'static str,
) -> Result<(), NotSupported> {
    capability_route_gate(capability, route, |detail, missing| {
        antecedent_core::support_gate::require_supported(ctx, detail, missing)
    })
}

/// [`require_capability_route`] for an entry point that compiles without an
/// [`ExecutionContext`]. A development build records into the calling thread's log
/// ([`antecedent_core::support_gate::drain_unscoped`]).
///
/// # Errors
///
/// [`NotSupported`] in a release build when `capability` is not a supported unit.
pub(crate) fn require_capability_route_unscoped(
    capability: &'static str,
    route: &'static str,
) -> Result<(), NotSupported> {
    capability_route_gate(
        capability,
        route,
        antecedent_core::support_gate::require_supported_unscoped,
    )
}

fn capability_route_gate(
    capability: &'static str,
    route: &'static str,
    gate: impl FnOnce(&'static str, &[&str]) -> Result<(), NotSupported>,
) -> Result<(), NotSupported> {
    if capability_supported(capability) {
        return Ok(());
    }
    let listed = capability_units()
        .and_then(|units| units.iter().find(|unit| unit.id == capability))
        .map(|unit| unit.missing_evidence.iter().map(String::as_str).collect::<Vec<_>>())
        .filter(|missing| !missing.is_empty());
    gate(route, listed.as_deref().unwrap_or(&[capability]))
}
