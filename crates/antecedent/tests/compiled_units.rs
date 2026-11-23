//! The compiled-route to support-unit table is current and agrees with the support data.
//!
//! `parity/compiled_units.toml` binds each compiled route outside the support matrix to the
//! unit whose status gates it (its `[[route]]` rows) and lists the routes no record owns yet
//! (its `[[unbound]]` rows); `src/support/units_generated.rs` is generated from the bound rows.
//! This reads both as text, and the capability records and the embedded unit status, so that a
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

/// The lines of every `header` table (`[[route]]` or `[[unbound]]`) of `text`, in order.
fn tables(text: &str, header: &str) -> String {
    let mut inside = false;
    let mut kept = String::new();
    for line in text.lines() {
        if line.starts_with('[') {
            inside = line.trim() == header;
            continue;
        }
        if inside {
            kept.push_str(line);
            kept.push('\n');
        }
    }
    kept
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
    let compiled = tables(&read("parity/compiled_units.toml"), "[[route]]");
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
    let compiled = tables(&read("parity/compiled_units.toml"), "[[route]]");
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
    let compiled = tables(&read("parity/compiled_units.toml"), "[[route]]");
    let modules = [
        "crates/antecedent/src/analysis/observational_policy.rs",
        "crates/antecedent/src/analysis/composition.rs",
        "crates/antecedent/src/analysis/composition_decision_terminal.rs",
        "crates/antecedent/src/analysis/composition_sequential_bundle.rs",
        "crates/antecedent/src/analysis/joint_panel/compiled.rs",
        "crates/antecedent/src/analysis/events/mod.rs",
        "crates/antecedent/src/analysis/observation_query.rs",
        "crates/antecedent/src/analysis/sequential_policy_reliability.rs",
        "crates/antecedent/src/analysis/graph_imputation.rs",
        "crates/antecedent/src/analysis/categorical_posterior_candidate.rs",
        "crates/antecedent/src/analysis/cyclic.rs",
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

#[test]
fn an_unbound_route_is_never_bound_and_states_why() {
    let compiled = read("parity/compiled_units.toml");
    let bound: BTreeSet<String> =
        values(&tables(&compiled, "[[route]]"), "lowering").into_iter().collect();
    let unbound = tables(&compiled, "[[unbound]]");
    let listed = values(&unbound, "lowering");
    assert_eq!(listed.len(), values(&unbound, "reason").len(), "an unbound row lacks a reason");
    let generated = read("crates/antecedent/src/support/units_generated.rs");
    for route in &listed {
        assert!(!bound.contains(route), "{route} is bound and listed as unbound");
        assert!(
            !generated.contains(&format!("\"{route}\"")),
            "{route} is listed as unbound but generated"
        );
    }
}
