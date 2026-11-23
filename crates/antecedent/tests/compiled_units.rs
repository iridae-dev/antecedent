//! The compiled-route to support-unit table is current and agrees with the support data.
//!
//! `parity/compiled_units.toml` binds each compiled route outside the support matrix to the
//! unit whose status gates it; `src/support/units_generated.rs` is generated from it. This
//! reads both as text, and the capability records and the embedded unit status, so that a
//! unit renamed in one place and not in the others is named here.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read(relative: &str) -> String {
    fs::read_to_string(root().join(relative)).unwrap_or_else(|e| panic!("{relative}: {e}"))
}

/// The quoted value of every `key = "value"` line of `text`.
fn values(text: &str, key: &str) -> Vec<String> {
    let prefix = format!("{key} = \"");
    text.lines()
        .filter_map(|line| line.strip_prefix(prefix.as_str()))
        .filter_map(|rest| rest.split('"').next())
        .map(str::to_owned)
        .collect()
}

#[test]
fn the_generated_table_names_every_bound_route_and_unit() {
    let compiled = read("parity/compiled_units.toml");
    let generated = read("crates/antecedent/src/support/units_generated.rs");
    let routes = values(&compiled, "lowering");
    let units: BTreeSet<String> = values(&compiled, "unit").into_iter().collect();
    assert!(!routes.is_empty(), "no bound routes");
    assert_eq!(routes.len(), values(&compiled, "unit").len(), "a row lacks a unit");
    // The routes are the quoted patterns of `unit_of`, grouped by unit.
    let body = generated.split("fn unit_of").nth(1).expect("the generated file defines unit_of");
    let quoted: BTreeSet<&str> = body.split('"').skip(1).step_by(2).collect();
    for route in &routes {
        assert!(quoted.contains(route.as_str()), "{route} is not generated");
    }
    for unit in &units {
        assert!(generated.contains(&format!("UnitId::new(\"{unit}\")")), "{unit} is not generated");
    }
    assert_eq!(
        generated.matches("UnitId::new(").count(),
        units.len(),
        "the generated file holds a unit no row binds"
    );
    assert_eq!(quoted.len(), routes.len(), "the generated file binds a route no row lists");
}

#[test]
fn every_bound_unit_is_a_capability_record_with_an_embedded_status() {
    let compiled = read("parity/compiled_units.toml");
    let records = read("parity/capability_records.toml");
    let record_ids: BTreeSet<String> = values(&records, "id").into_iter().collect();
    let status: serde_json::Value =
        serde_json::from_str(antecedent_io_core::graphless_support::UNIT_STATUS)
            .expect("the embedded unit status decodes");
    let listed: BTreeSet<&str> = status["capabilities"]
        .as_array()
        .expect("capabilities list")
        .iter()
        .filter_map(|unit| unit["id"].as_str())
        .collect();
    for unit in values(&compiled, "unit") {
        assert!(record_ids.contains(&unit), "{unit} is not a capability record");
        assert!(listed.contains(unit.as_str()), "{unit} has no embedded status");
    }
}

#[test]
fn a_bound_module_holds_no_hand_written_unit_id() {
    let compiled = read("parity/compiled_units.toml");
    let modules = [
        "crates/antecedent/src/analysis/observational_policy.rs",
        "crates/antecedent/src/analysis/composition.rs",
        "crates/antecedent/src/analysis/composition_decision_terminal.rs",
        "crates/antecedent/src/analysis/composition_sequential_bundle.rs",
        "crates/antecedent/src/analysis/joint_panel/compiled.rs",
        "crates/antecedent/src/analysis/events/mod.rs",
    ];
    for module in modules {
        let text = read(module);
        for unit in values(&compiled, "unit") {
            assert!(
                !text.contains(&format!("\"{unit}\"")),
                "{module} spells the unit {unit}; use the generated constant"
            );
        }
    }
}
