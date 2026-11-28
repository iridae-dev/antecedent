//! The calibration state of an evidence unit, derived from its current coverage evidence
//! and declared scope (`parity/unit_index.json`). The support generator embeds these
//! annotations in `data/unit_support_status.json` ([`crate::graphless_support::UNIT_STATUS`]).
//! A unit without an embedded annotation is [`NOT_MEASURED`].
//!
//! A result states its unit's calibration through [`capability`] or [`unit`], never as a
//! literal, so that a unit's state follows its records.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::sync::OnceLock;

/// The default state of a unit without current coverage evidence.
pub const NOT_MEASURED: &str = "not_measured";

/// Fixed marker for an existing artifact format that carries no calibration evidence.
///
/// Retained artifacts replay independently of the reader's evidence snapshot. Live
/// calibration is queried separately for the unit that produced a result.
pub const LEGACY_WIRE_STATUS: &str = "not_measured";

/// Validate the vocabulary of a v1 field whose producer embedded a calibration annotation.
///
/// These labels describe the producer's historical metadata. A reader determines current
/// calibration separately from the unit or the interval's verified coverage record.
#[must_use]
pub fn is_v1_wire_status(status: &str) -> bool {
    matches!(
        status,
        LEGACY_WIRE_STATUS
            | "calibrated"
            | "measured_below_nominal"
            | "no_interval_reported"
            | "estimator_grid_not_measured"
    )
}

#[derive(serde::Deserialize)]
struct Snapshot {
    #[serde(default)]
    calibration: HashMap<String, String>,
}

/// `<kind>:<record id>` -> state, or an empty table when the snapshot does not decode, so that
/// no unit reads as measured on a misread snapshot.
fn states() -> &'static HashMap<String, &'static str> {
    static STATES: OnceLock<HashMap<String, &'static str>> = OnceLock::new();
    STATES.get_or_init(|| {
        let snapshot: Option<Snapshot> =
            serde_json::from_str(crate::graphless_support::UNIT_STATUS).ok();
        let mut interned: HashMap<String, &'static str> = HashMap::new();
        snapshot
            .map(|snapshot| snapshot.calibration)
            .unwrap_or_default()
            .into_iter()
            .map(|(unit, state)| {
                let state = *interned
                    .entry(state.clone())
                    .or_insert_with(|| Box::leak(state.into_boxed_str()));
                (unit, state)
            })
            .collect()
    })
}

/// The calibration state of the unit `id` of kind `kind` (`matrix`, `construction`,
/// `graphless` or `capability`, with its `parity/unit_index.json` record id).
#[must_use]
pub fn unit(kind: &str, id: &str) -> &'static str {
    states().get(&format!("{kind}:{id}")).copied().unwrap_or(NOT_MEASURED)
}

/// The calibration state of the capability unit `id` (`parity/capability_records.toml`).
#[must_use]
pub fn capability(id: &str) -> &'static str {
    unit("capability", id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unlisted_unit_is_not_measured() {
        assert_eq!(capability("no.such.unit"), NOT_MEASURED);
    }

    #[test]
    fn the_snapshot_omits_the_default_state() {
        for state in states().values() {
            assert_ne!(*state, NOT_MEASURED);
            assert!(!state.is_empty());
        }
    }
}
