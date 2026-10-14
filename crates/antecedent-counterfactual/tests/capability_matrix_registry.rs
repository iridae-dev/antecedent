//! The capability matrix in code equals the checked registry data.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_counterfactual::selective::rows;

const REGISTRY: &str = include_str!("../../../parity/cross_world_capability.toml");

/// Minimal reader of the registry's `[[row]]` tables: `key = "value"` and
/// `key = ["a", "b"]` lines, which is all the file contains.
fn parse() -> Vec<BTreeMap<String, Vec<String>>> {
    let mut rows: Vec<BTreeMap<String, Vec<String>>> = Vec::new();
    for line in REGISTRY.lines() {
        let line = line.trim();
        if line == "[[row]]" {
            rows.push(BTreeMap::new());
        } else if let (Some(row), Some((key, value))) = (rows.last_mut(), line.split_once(" = ")) {
            let items: Vec<String> = value
                .trim_matches(|c| c == '[' || c == ']')
                .split(", ")
                .map(|v| v.trim().trim_matches('"').to_owned())
                .filter(|v| !v.is_empty())
                .collect();
            let items = if value.starts_with('[') {
                items
            } else {
                vec![value.trim_matches('"').to_owned()]
            };
            row.insert(key.to_owned(), items);
        }
    }
    rows
}

#[test]
fn the_registry_rows_are_exactly_the_code_rows() {
    let registry = parse();
    let code = rows();
    assert_eq!(registry.len(), code.len());
    for (data, row) in registry.iter().zip(code) {
        let one = |key: &str| data.get(key).map(|v| v[0].as_str());
        let many = |key: &str| -> Vec<String> { data.get(key).cloned().unwrap_or_default() };
        assert_eq!(one("id"), Some(row.id));
        assert_eq!(one("quantity"), Some(row.quantity.name()), "{}", row.id);
        assert_eq!(one("model"), Some(row.model.name()), "{}", row.id);
        assert_eq!(one("population"), Some(row.population.name()), "{}", row.id);
        assert_eq!(one("coupling"), Some(row.coupling.name()), "{}", row.id);
        assert_eq!(one("status"), Some(row.status.name()), "{}", row.id);
        assert_eq!(one("route"), Some(row.route), "{}", row.id);
        assert_eq!(one("refusal_code"), row.refusal_code, "{}", row.id);
        assert_eq!(one("refusal_detail"), row.refusal_detail, "{}", row.id);
        let names = |items: Vec<&str>| items.into_iter().map(str::to_owned).collect::<Vec<_>>();
        assert_eq!(
            many("conditioning"),
            names(row.conditioning.iter().map(|c| c.name()).collect())
        );
        assert_eq!(many("history"), names(row.history.iter().map(|c| c.name()).collect()));
        assert_eq!(many("evidence"), names(row.evidence.iter().map(|c| c.name()).collect()));
        assert!(!data["theorem"][0].is_empty(), "{}", row.id);
        assert_eq!(!data["remedy"][0].is_empty(), row.refusal_code.is_some(), "{}", row.id);
    }
}
