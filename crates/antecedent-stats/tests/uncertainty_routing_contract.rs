//! Uncertainty-routing contract conformance.
//!
//! Consumes `conformance/estimate/uncertainty_routing/expected.json` — the
//! frozen routing table mapping requested covariance kinds to their target
//! semantics and required metadata. Numeric agreement for each kind is owned
//! by `conformance/stats/sandwich_covariance` (exercised in
//! `foundations_oracle.rs`); this test owns the *routing contract*: the set of
//! kinds the library exposes, and what metadata each kind demands, must match
//! the frozen fixture in both directions.
//!
//! Until this test existed the fixture was recorded but loaded by nothing,
//! while `parity/oracle_closure.toml` carried the row as `closed` — evidence
//! cited but never executed. Do not delete this consumer without downgrading
//! that row.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use antecedent_stats::SandwichKind;
use serde_json::Value;

fn fixture() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../conformance/estimate/uncertainty_routing/expected.json");
    let raw = fs::read_to_string(path).expect("read uncertainty_routing fixture");
    serde_json::from_str(&raw).expect("parse uncertainty_routing fixture")
}

/// Fixture-facing name and required metadata for a covariance kind.
///
/// The match is exhaustive on purpose: adding a `SandwichKind` variant fails
/// compilation here until an arm is added, and the new arm then fails the
/// set-equality assertions below until the frozen fixture is regenerated to
/// describe the new route. That is the accountability loop this fixture exists
/// to provide.
fn route_of(kind: &SandwichKind) -> (&'static str, &'static [&'static str]) {
    match kind {
        SandwichKind::Homoskedastic => ("Homoskedastic", &[]),
        SandwichKind::Hc0 => ("Hc0", &[]),
        SandwichKind::Hc1 => ("Hc1", &[]),
        SandwichKind::Hc2 => ("Hc2", &[]),
        SandwichKind::Hc3 => ("Hc3", &[]),
        SandwichKind::Cluster { .. } => ("Cluster", &["cluster_ids"]),
        SandwichKind::Multiway { .. } => ("Multiway", &["multiway_ids"]),
        SandwichKind::NeweyWest { .. } => ("NeweyWest", &["lag"]),
        SandwichKind::PanelClusterHac { .. } => {
            ("PanelClusterHac", &["cluster_ids", "panel_times", "lag"])
        }
    }
}

#[test]
fn routing_table_matches_frozen_fixture_in_both_directions() {
    let groups = [0u32, 1];
    let dims: [&[u32]; 1] = [&groups];
    let time = [0i64, 1];
    let code_routes = [
        SandwichKind::Homoskedastic,
        SandwichKind::Hc0,
        SandwichKind::Hc1,
        SandwichKind::Hc2,
        SandwichKind::Hc3,
        SandwichKind::Cluster { groups: &groups },
        SandwichKind::Multiway { dimensions: &dims },
        SandwichKind::NeweyWest { lag: 1 },
        SandwichKind::PanelClusterHac { groups: &groups, time: &time, lag: 1 },
    ];

    let mut code: BTreeMap<&str, Vec<String>> = BTreeMap::new();
    for kind in &code_routes {
        let (name, meta) = route_of(kind);
        let dup = code.insert(name, meta.iter().map(ToString::to_string).collect());
        assert!(dup.is_none(), "duplicate code route {name}");
    }

    let v = fixture();
    let routes = v["routes"].as_array().expect("routes array");
    let mut frozen: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for r in routes {
        let kind = r["kind"].as_str().expect("route kind").to_owned();
        let meta: Vec<String> = r["required_metadata"]
            .as_array()
            .expect("required_metadata array")
            .iter()
            .map(|m| m.as_str().expect("metadata name").to_owned())
            .collect();
        let dup = frozen.insert(kind.clone(), meta);
        assert!(dup.is_none(), "duplicate fixture route {kind}");
    }

    // Both directions: no route the code supports is missing from the frozen
    // contract, and no frozen route has quietly vanished from the code.
    let code_kinds: Vec<&&str> = code.keys().collect();
    let frozen_kinds: Vec<&String> = frozen.keys().collect();
    assert_eq!(
        code_kinds.iter().map(ToString::to_string).collect::<Vec<_>>(),
        frozen_kinds.iter().map(ToString::to_string).collect::<Vec<_>>(),
        "covariance kinds diverge between SandwichKind and the frozen routing fixture"
    );
    for (kind, meta) in &code {
        assert_eq!(
            &frozen[*kind], meta,
            "required metadata for {kind} diverges from the frozen fixture"
        );
    }
}

#[test]
fn numeric_deferral_target_exists_and_is_the_exercised_oracle() {
    // The fixture defers all numeric tolerances to the sandwich_covariance
    // oracle. That deferral is only honest while the target exists.
    let v = fixture();
    let acceptance = v["acceptance"]["numeric_tolerances"].as_str().expect("numeric_tolerances");
    assert!(
        acceptance.contains("conformance/stats/sandwich_covariance"),
        "acceptance no longer defers to the sandwich_covariance oracle: {acceptance}"
    );
    let target = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../conformance/stats/sandwich_covariance/expected.json");
    assert!(target.exists(), "numeric deferral target missing: {}", target.display());
}
