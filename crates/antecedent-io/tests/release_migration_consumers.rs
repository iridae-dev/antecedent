//! Independent consumers of the previous-release route inventory.
//!
//! The consumer here reads `parity/capabilities/route_manifest.toml` as plain text, with no
//! use of the manifest parsers elsewhere in the workspace. Its `[[route]]` records are the
//! previous-release routes and its `[[method]]` records the current methods. Every route is
//! carried onto the methods that carry it; each gets an explicit one-to-one, split, retired or
//! kept-refusal outcome, or an explicit error that names the remedy. A frozen previous-release
//! route artifact is read through the current reader.

use std::collections::BTreeMap;
use std::path::PathBuf;

use antecedent_io::release_migration::{
    added_methods, consume_previous_route_records, migrate_route_record, CurrentMethod,
    ManifestRouteRecord, PreviousReleaseError, RouteAxis, RouteRecordError, RouteRecordOutcome,
};

struct Route {
    id: String,
    status: String,
    design_time_model: String,
    inference: String,
}

/// The first `key = "value"` string on a line.
fn field(line: &str, key: &str) -> Option<String> {
    let rest = line.strip_prefix(key)?.trim_start().strip_prefix('=')?.trim();
    let start = rest.find('"')? + 1;
    let end = rest.rfind('"')?;
    (end >= start).then(|| rest[start..end].to_owned())
}

fn manifest_text() -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../parity/capabilities/route_manifest.toml");
    std::fs::read_to_string(&path).expect("route manifest is readable")
}

fn routes_and_methods() -> (Vec<Route>, Vec<CurrentMethod>) {
    let text = manifest_text();
    let mut routes: Vec<Route> = Vec::new();
    let mut methods: Vec<CurrentMethod> = Vec::new();
    let mut in_method = false;
    for line in text.lines() {
        match line.trim() {
            "[[route]]" => {
                in_method = false;
                routes.push(Route {
                    id: String::new(),
                    status: String::new(),
                    design_time_model: String::new(),
                    inference: String::new(),
                });
                continue;
            }
            "[[method]]" => {
                in_method = true;
                methods.push(CurrentMethod {
                    id: String::new(),
                    route: String::new(),
                    estimator: String::new(),
                    design: String::new(),
                    inference: String::new(),
                    status: String::new(),
                });
                continue;
            }
            _ => {}
        }
        if in_method {
            let Some(method) = methods.last_mut() else { continue };
            if let Some(value) = field(line, "id") {
                method.id = value;
            } else if let Some(value) = field(line, "route") {
                method.route = value;
            } else if let Some(value) = field(line, "estimator") {
                method.estimator = value;
            } else if let Some(value) = field(line, "design") {
                method.design = value;
            } else if let Some(value) = field(line, "inference") {
                method.inference = value;
            } else if let Some(value) = field(line, "status") {
                method.status = value;
            }
        } else {
            let Some(route) = routes.last_mut() else { continue };
            if let Some(value) = field(line, "id") {
                route.id = value;
            } else if let Some(value) = field(line, "status") {
                route.status = value;
            } else if let Some(value) = field(line, "design_time_model") {
                route.design_time_model = value;
            } else if let Some(value) = field(line, "inference") {
                route.inference = value;
            }
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
                assert_eq!(route.status, "closed", "{}", route.id);
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
    // Every previous refusal now executes, so the migration carries it as a method.
    assert!(one > 0 && split > 0, "kept {kept}, one {one}, split {split}");
    // Every method that names a previous route is carried by exactly that route's outcome.
    let ids: Vec<&str> = routes.iter().map(|r| r.id.as_str()).collect();
    let added = added_methods(&methods, &ids);
    assert_eq!(carried + added.len(), methods.len());
    assert!(
        added.iter().all(|m| m.route == "unlisted"),
        "a method names a route not in the manifest"
    );
    assert_eq!(narrowed, 1, "exactly one method keeps less than its route's licence");
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
    let route = routes.iter().find(|r| r.id == "licensed.AverageEffect.Dag").unwrap();
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
    assert!(matches!(by_id["licensed.AverageEffect.Dag"], RouteRecordOutcome::Split { .. }));
    // The previous refusal executes now: it migrates to the method that carries it.
    assert!(matches!(
        by_id["closed.AnomalyAttribution.Admg"],
        RouteRecordOutcome::OneToOne { .. } | RouteRecordOutcome::Split { .. }
    ));
    let RouteRecordOutcome::Split { methods: response, .. } =
        by_id["licensed.InterventionResponse.CoDetermined"]
    else {
        panic!("the response route splits");
    };
    assert_eq!(response.iter().filter(|m| m.narrowed).count(), 1);
    assert!(response.iter().any(|m| !m.narrowed && m.status == "licensed"));
    let RouteRecordOutcome::Split { methods: temporal, .. } =
        by_id["licensed.AnomalyAttribution.TemporalDag"]
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
        {"id":"licensed.AverageEffect.Dag","status":"licensed","design_time_model":"static","inference":"Frequentist"},
        {"id":"licensed.RemovedQuery.Dag","status":"licensed","design_time_model":"static","inference":"Frequentist"},
        {"id":"graphless.removed.design","status":"graphless","design_time_model":"static","inference":"unlisted"},
        {"id":"licensed.AverageEffect.Dag","status":"withdrawn","design_time_model":"static","inference":"Frequentist"}
    ]}"#;
    let migrated = consume_previous_route_records(artifact, &methods).unwrap();
    assert_eq!(migrated.len(), 4);
    assert!(matches!(migrated[0].result, Ok(RouteRecordOutcome::Split { .. })));
    let error = migrated[1].result.clone().unwrap_err();
    assert_eq!(
        error,
        RouteRecordError::RouteWithoutCurrentMethod { id: "licensed.RemovedQuery.Dag".into() }
    );
    assert!(error.to_string().contains("method that replaced it"));
    assert_eq!(migrated[2].result, Ok(RouteRecordOutcome::Retired));
    assert!(matches!(migrated[3].result, Err(RouteRecordError::UnknownStatus { .. })));
}

#[test]
fn bytes_that_are_not_a_route_artifact_are_not_decoded() {
    let (_, methods) = routes_and_methods();
    assert_eq!(
        consume_previous_route_records(b"{\"routes\":[{\"id\":\"x\"}]}", &methods).unwrap_err(),
        PreviousReleaseError::ArtifactNotDecoded
    );
    assert_eq!(
        consume_previous_route_records(b"[]", &methods).unwrap_err(),
        PreviousReleaseError::ArtifactNotDecoded
    );
}
