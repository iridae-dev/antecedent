//! Independent consumers of the previous-release route inventory.
//!
//! The consumer here reads the route manifest and the support model
//! (`parity/support_cells.toml`, `parity/support_graphless.toml`,
//! `parity/capability_records.toml`) as plain text, with no use of the manifest parsers
//! elsewhere in the workspace. The manifest's `[[route]]` records are the previous-release
//! routes; the current methods are the support model's units: one per estimator and inference
//! of a support-matrix cell, and one per route that is an entry point of a supported estimator,
//! a graphless support row or a supported capability record. Every route is carried onto the
//! methods that carry it; each gets an explicit one-to-one, split, retired or kept-refusal
//! outcome, or an explicit error that names the remedy. A frozen previous-release route
//! artifact is read through the current reader.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use antecedent_io::release_migration::{
    CurrentMethod, ManifestRouteRecord, PreviousReleaseError, RouteAxis, RouteRecordError,
    RouteRecordOutcome, added_methods, consume_previous_route_records, migrate_route_record,
};

struct Route {
    id: String,
    status: String,
    design_time_model: String,
    inference: String,
    graph_class: String,
    notes: String,
    retained_route_test: Option<String>,
    capability_records: Vec<String>,
}

/// The first `key = "value"` string on a line.
fn field(line: &str, key: &str) -> Option<String> {
    let rest = line.strip_prefix(key)?.trim_start().strip_prefix('=')?.trim();
    let start = rest.find('"')? + 1;
    let end = rest.rfind('"')?;
    (end >= start).then(|| rest[start..end].to_owned())
}

fn read(relative: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").join(relative);
    std::fs::read_to_string(&path).unwrap_or_else(|_| panic!("{relative} is readable"))
}

/// The top-level lines of every `[[header]]` table; a nested table ends its parent.
fn tables(text: &str, header: &str) -> Vec<Vec<String>> {
    let mut found: Vec<Vec<String>> = Vec::new();
    let mut open = false;
    for line in text.lines() {
        if line.starts_with("[[") {
            open = line.trim() == header;
            if open {
                found.push(Vec::new());
            }
        } else if open {
            if let Some(table) = found.last_mut() {
                table.push(line.to_owned());
            }
        }
    }
    found
}

fn value(table: &[String], key: &str) -> Option<String> {
    table.iter().find_map(|line| field(line, key))
}

/// Every quoted string of the first `key = [...]` or `key = "..."` line.
fn strings(table: &[String], key: &str) -> Vec<String> {
    table
        .iter()
        .find_map(|line| {
            let rest = line.strip_prefix(key)?.trim_start().strip_prefix('=')?;
            Some(rest.split('"').skip(1).step_by(2).map(str::to_owned).collect())
        })
        .unwrap_or_default()
}

fn design_of(graph_class: &str, route_id: &str) -> &'static str {
    if graph_class.starts_with("Temporal") || route_id.contains("temporal") {
        "temporal"
    } else {
        "static"
    }
}

/// The inference realization a route record states.
fn inference_of(text: &str) -> &'static str {
    let first = text.split(';').next().unwrap_or("");
    if text.starts_with("Frequentist") && !first.contains("Bayesian") {
        "frequentist"
    } else if text.starts_with("Bayesian") {
        "bayesian"
    } else if text.contains("Bayesian") && text.contains("Frequentist") {
        "both"
    } else {
        "unlisted"
    }
}

fn routes_and_methods() -> (Vec<Route>, Vec<CurrentMethod>) {
    let routes: Vec<Route> = tables(&read("parity/capabilities/route_manifest.toml"), "[[route]]")
        .iter()
        .map(|t| Route {
            id: value(t, "id").unwrap_or_default(),
            status: value(t, "status").unwrap_or_default(),
            design_time_model: value(t, "design_time_model").unwrap_or_default(),
            inference: value(t, "inference").unwrap_or_default(),
            graph_class: value(t, "graph_class").unwrap_or_default(),
            notes: value(t, "notes").unwrap_or_default(),
            retained_route_test: value(t, "retained_route_test"),
            capability_records: strings(t, "capability_record"),
        })
        .collect();

    // Support-matrix cells: one method per (query, graph class, estimator, inference).
    let mut units: BTreeSet<(String, String, String, String)> = BTreeSet::new();
    let mut estimators: BTreeSet<String> = BTreeSet::new();
    for cell in tables(&read("parity/support_cells.toml"), "[[cell]]") {
        let query = value(&cell, "query").unwrap_or_default();
        let graph = value(&cell, "graph_class").unwrap_or_default();
        let inference = value(&cell, "inference").unwrap_or_default();
        for estimator in strings(&cell, "estimators") {
            estimators.insert(estimator.clone());
            units.insert((query.clone(), graph.clone(), estimator, inference.clone()));
        }
    }
    let mut methods: Vec<CurrentMethod> = units
        .iter()
        .map(|(query, graph, estimator, inference)| CurrentMethod {
            id: format!("method.{query}.{graph}.{estimator}.{}", inference.to_lowercase()),
            route: format!("supported.{query}.{graph}"),
            estimator: estimator.clone(),
            design: design_of(graph, "").to_owned(),
            inference: inference.to_lowercase(),
            status: "supported".to_owned(),
        })
        .collect();

    // Graphless support rows, by the route id they extend or the retained test they cite.
    let rows = tables(&read("parity/support_graphless.toml"), "[[support]]");
    let graphless_ids: Vec<String> = rows
        .iter()
        .map(|r| {
            let get = |key: &str| value(r, key).unwrap_or_default();
            format!("graphless.{}.{}.{}", get("family"), get("design"), get("method"))
        })
        .collect();
    let graphless_tests: BTreeSet<String> =
        rows.iter().filter_map(|r| value(r, "retained_route_test")).collect();
    // Capability records the support model supports.
    let supported_records: BTreeSet<String> =
        tables(&read("parity/capability_records.toml"), "[[record]]")
            .iter()
            .filter(|r| value(r, "status").as_deref() == Some("supported"))
            .filter_map(|r| value(r, "id"))
            .collect();

    // Every other route is an entry point of one support unit.
    for route in routes.iter().filter(|r| !r.id.starts_with("supported.")) {
        let mut estimator = "unlisted".to_owned();
        let carried = match route.status.as_str() {
            "supported" => {
                let named = estimators
                    .iter()
                    .find(|e| **e == route.id || route.notes.contains(&format!("stimator {e} ")));
                if let Some(wire_id) = named {
                    estimator.clone_from(wire_id);
                }
                named.is_some()
            }
            "graphless" => {
                route.retained_route_test.as_ref().is_some_and(|t| graphless_tests.contains(t))
                    || graphless_ids
                        .iter()
                        .any(|g| route.id == *g || route.id.starts_with(&format!("{g}.")))
            }
            "capability" => route.capability_records.iter().any(|r| supported_records.contains(r)),
            _ => false,
        };
        if carried {
            methods.push(CurrentMethod {
                id: format!("method.{}", route.id),
                route: route.id.clone(),
                estimator,
                design: design_of(&route.graph_class, &route.id).to_owned(),
                inference: inference_of(&route.inference).to_owned(),
                status: route.status.clone(),
            });
        }
    }
    (routes, methods)
}

fn migrate(
    route: &Route,
    methods: &[CurrentMethod],
) -> Result<RouteRecordOutcome, RouteRecordError> {
    migrate_route_record(
        &ManifestRouteRecord {
            id: &route.id,
            status: &route.status,
            design_time_model: &route.design_time_model,
            inference: &route.inference,
        },
        methods,
    )
}

fn fixture() -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/previous_release_routes_v1.json");
    std::fs::read(path).expect("previous-release route fixture is readable")
}

#[test]
fn every_previous_route_gets_an_explicit_outcome_and_none_is_a_break() {
    let (routes, methods) = routes_and_methods();
    assert!(routes.len() > 250 && methods.len() > 300, "{} {}", routes.len(), methods.len());
    let (mut kept, mut one, mut split, mut retired, mut narrowed) = (0, 0, 0, 0, 0);
    let mut occupied: BTreeMap<RouteAxis, usize> = BTreeMap::new();
    let mut carried: usize = 0;
    for route in &routes {
        assert!(!route.id.is_empty() && !route.status.is_empty(), "route {}", route.id);
        let outcome = migrate(route, &methods)
            .unwrap_or_else(|error| panic!("{}: genuine break: {error}", route.id));
        let assigned = match outcome {
            RouteRecordOutcome::RefusalKept => {
                assert_eq!(route.status, "not_supported", "{}", route.id);
                kept += 1;
                continue;
            }
            RouteRecordOutcome::Retired => {
                retired += 1;
                continue;
            }
            RouteRecordOutcome::OneToOne { method, .. } => {
                one += 1;
                vec![method]
            }
            RouteRecordOutcome::Split { methods, .. } => {
                assert!(methods.len() > 1, "{}", route.id);
                split += 1;
                methods
            }
        };
        for method in assigned {
            carried += 1;
            narrowed += usize::from(method.narrowed);
            for axis in method.axes {
                *occupied.entry(axis).or_default() += 1;
            }
        }
    }
    assert_eq!(kept + one + split + retired, routes.len());
    assert_eq!(retired, 0, "no previous route left the inventory");
    assert!(one > 0 && split > 0, "kept {kept}, one {one}, split {split}");
    // Every method that names a previous route is carried by exactly that route's outcome.
    let ids: Vec<&str> = routes.iter().map(|r| r.id.as_str()).collect();
    let added = added_methods(&methods, &ids);
    assert_eq!(carried + added.len(), methods.len());
    assert!(
        added.iter().all(|m| m.route == "unlisted"),
        "a method names a route not in the manifest"
    );
    assert_eq!(narrowed, 0, "current linked method statuses carry their route support kind");
    for axis in RouteAxis::ALL {
        assert!(
            occupied.get(&axis).copied().unwrap_or(0) > 0,
            "no migrated method occupies {axis}, so its migration is unverified"
        );
    }
}

#[test]
fn a_split_route_keeps_each_estimator_as_its_own_method() {
    let (routes, methods) = routes_and_methods();
    let route = routes.iter().find(|r| r.id == "supported.AverageEffect.Dag").unwrap();
    let RouteRecordOutcome::Split { methods: assigned, unserved_axes } =
        migrate(route, &methods).unwrap()
    else {
        panic!("the average-effect route splits across its estimators");
    };
    assert!(assigned.len() >= 10, "{}", assigned.len());
    assert!(unserved_axes.is_empty());
    let mut estimators: Vec<&str> = assigned.iter().map(|m| m.estimator.as_str()).collect();
    estimators.sort_unstable();
    estimators.dedup();
    assert!(estimators.len() > 5, "{estimators:?}");
    for axis in [RouteAxis::StaticFrequentist, RouteAxis::StaticBayesian] {
        assert!(assigned.iter().any(|m| m.axes.contains(&axis)), "{axis}");
    }
    assert!(assigned.iter().all(|m| !m.axes.contains(&RouteAxis::TemporalFrequentist)));
}

#[test]
fn the_current_method_decides_the_axis_a_previous_text_over_named() {
    let (routes, methods) = routes_and_methods();
    let route = routes.iter().find(|r| r.id == "gcm.simulate").unwrap();
    let RouteRecordOutcome::OneToOne { method, unserved_axes } = migrate(route, &methods).unwrap()
    else {
        panic!("one method carries the simulator");
    };
    assert_eq!(method.axes, [RouteAxis::StaticFrequentist]);
    // The previous text names temporal rollouts; the method records a static Dag.
    assert_eq!(unserved_axes, [RouteAxis::TemporalFrequentist]);
}

#[test]
fn a_frozen_previous_release_route_artifact_is_consumed_by_the_current_reader() {
    let (_, methods) = routes_and_methods();
    let migrated = consume_previous_route_records(&fixture(), &methods).unwrap();
    assert_eq!(migrated.len(), 10);
    let by_id: BTreeMap<&str, &RouteRecordOutcome> = migrated
        .iter()
        .map(|m| (m.id.as_str(), m.result.as_ref().expect("no fixture record is a break")))
        .collect();
    assert!(matches!(by_id["supported.AverageEffect.Dag"], RouteRecordOutcome::Split { .. }));
    // No current method carries this historical refused route, so its refusal remains.
    assert_eq!(by_id["refused.AnomalyAttribution.Admg"], &RouteRecordOutcome::RefusalKept);
    let RouteRecordOutcome::Split { methods: response, .. } =
        by_id["supported.InterventionResponse.CoDetermined"]
    else {
        panic!("the response route splits");
    };
    // Both current method records carry the supported status; this is a route-status
    // comparison, not an assertion that a calibration record attests coverage.
    assert_eq!(response.len(), 2);
    assert!(response.iter().all(|m| !m.narrowed && m.status == "supported"));
    assert!(response.iter().any(|m| m.estimator == "cell.aipw"));
    assert!(response.iter().any(|m| m.estimator == "response.bayesian"));
    let RouteRecordOutcome::Split { methods: temporal, .. } =
        by_id["supported.AnomalyAttribution.TemporalDag"]
    else {
        panic!("temporal attribution splits by realization");
    };
    let axes: Vec<RouteAxis> = temporal.iter().flat_map(|m| m.axes.clone()).collect();
    assert_eq!(axes, [RouteAxis::TemporalFrequentist, RouteAxis::TemporalBayesian]);
    let RouteRecordOutcome::OneToOne { method, unserved_axes } =
        by_id["gcm.attribute.temporal.bayesian"]
    else {
        panic!("one method carries the execution-only route");
    };
    assert_eq!(method.axes, [RouteAxis::TemporalBayesian]);
    assert_eq!(unserved_axes, &[RouteAxis::StaticBayesian]);
}

#[test]
fn a_fixture_with_a_removed_route_reports_each_break_beside_the_carried_records() {
    let (_, methods) = routes_and_methods();
    let artifact = br#"{"routes":[
        {"id":"supported.AverageEffect.Dag","status":"supported","design_time_model":"static","inference":"Frequentist"},
        {"id":"supported.RemovedQuery.Dag","status":"supported","design_time_model":"static","inference":"Frequentist"},
        {"id":"graphless.removed.design","status":"graphless","design_time_model":"static","inference":"unlisted"},
        {"id":"execution.removed.design","status":"execution_only","design_time_model":"static","inference":"unlisted"},
        {"id":"unknown.status.route","status":"withdrawn","design_time_model":"static","inference":"Frequentist"}
    ]}"#;
    let migrated = consume_previous_route_records(artifact, &methods).unwrap();
    assert_eq!(migrated.len(), 5);
    assert!(matches!(migrated[0].result, Ok(RouteRecordOutcome::Split { .. })));
    let error = migrated[1].result.clone().unwrap_err();
    assert_eq!(
        error,
        RouteRecordError::RouteWithoutCurrentMethod { id: "supported.RemovedQuery.Dag".into() }
    );
    assert!(error.to_string().contains("method that replaced it"));
    assert_eq!(
        migrated[2].result,
        Err(RouteRecordError::RouteWithoutCurrentMethod { id: "graphless.removed.design".into() })
    );
    assert_eq!(migrated[3].result, Ok(RouteRecordOutcome::Retired));
    assert!(matches!(migrated[4].result, Err(RouteRecordError::UnknownStatus { .. })));
}

#[test]
fn bytes_that_are_not_a_route_artifact_are_not_decoded() {
    let (_, methods) = routes_and_methods();
    assert_eq!(
        consume_previous_route_records(b"{\"routes\":[{\"id\":\"x\"}]}", &methods).unwrap_err(),
        PreviousReleaseError::ArtifactNotDecoded
    );
    let duplicate = br#"{"routes":[
        {"id":"repeated","status":"not_supported","design_time_model":"static","inference":"Frequentist"},
        {"id":"repeated","status":"not_supported","design_time_model":"static","inference":"Frequentist"}
    ]}"#;
    assert_eq!(
        consume_previous_route_records(duplicate, &methods).unwrap_err(),
        PreviousReleaseError::ArtifactNotDecoded
    );
    assert_eq!(
        consume_previous_route_records(b"[]", &methods).unwrap_err(),
        PreviousReleaseError::ArtifactNotDecoded
    );
}
