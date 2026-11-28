//! The calibration state of an evidence unit: what the coverage records whose tests executed
//! it measured (`parity/unit_index.json`). `scripts/generate_support_matrix_docs.py` embeds the
//! states in `data/unit_support_status.json` ([`crate::graphless_support::UNIT_STATUS`]),
//! listing only the units some record measures; every other unit is [`NOT_MEASURED`].
//!
//! A result states its unit's calibration through [`capability`] or [`unit`], never as a
//! literal, so that a unit's state follows its records.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::sync::OnceLock;

/// The state of a unit no coverage record measures.
pub const NOT_MEASURED: &str = "not_measured";

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
    fn the_snapshot_lists_only_measured_units() {
        for state in states().values() {
            assert_ne!(*state, NOT_MEASURED);
            assert!(!state.is_empty());
        }
    }
}
