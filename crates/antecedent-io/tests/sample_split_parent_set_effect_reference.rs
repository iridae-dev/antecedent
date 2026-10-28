//! Acceptance of the treatment's parent set over every marked graph on four nodes.
//!
//! `oracle_data/sample_split_parent_set_effect_reference.py` (frozen in the JSON beside it)
//! lists, for every acyclic assignment of {absent, undirected, directed either way} to the six
//! node pairs and two treatment/outcome choices, what the definition gives: the parent set is
//! accepted only when every edge incident to the treatment is directed, undirected edges
//! elsewhere are listed and never completed, and an outcome that is a parent of the treatment is
//! refused after the ambiguity check; the fixed-graph effect route refuses any undirected edge
//! anywhere. The product's `treatment_parent_set` and `effect_on_structure` must agree on every
//! pattern, and the adjustment set of an accepted effect must be exactly the expected parents.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{NodeRef, VariableId};
use antecedent_data::TabularData;
use antecedent_discovery::selective::DiscoveredStructure;
use antecedent_graph::{Cpdag, DenseNodeId};
use antecedent_io::measured_inference::split_effect::effect_on_structure;
use antecedent_io::measured_inference::split_parent_set_effect::treatment_parent_set;
use serde_json::Value;

const PAIRS: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];

fn structure(pattern: &str) -> DiscoveredStructure {
    let mut graph = Cpdag::empty();
    let ids: Vec<DenseNodeId> = (0..4_u32)
        .map(|v| graph.add_node(NodeRef::Static(VariableId::from_raw(v))).unwrap())
        .collect();
    for (&(a, b), mark) in PAIRS.iter().zip(pattern.chars()) {
        match mark {
            '.' => {}
            '-' => graph.insert_undirected(ids[a], ids[b]).unwrap(),
            '>' => graph.insert_directed(ids[a], ids[b]).unwrap(),
            '<' => graph.insert_directed(ids[b], ids[a]).unwrap(),
            other => panic!("unknown mark {other}"),
        }
    }
    DiscoveredStructure::from_cpdag(&graph)
}

fn frame() -> TabularData {
    // Four deterministic, non-collinear columns of 24 rows.
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut next = || {
        state =
            state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((state >> 33) as f64) / f64::from(1_u32 << 31) - 0.5
    };
    let columns: Vec<Vec<f64>> = (0..4).map(|_| (0..24).map(|_| next()).collect()).collect();
    TabularData::from_f64_columns(
        ["v0", "v1", "v2", "v3"].into_iter().zip(columns.iter().map(Vec::as_slice)),
    )
    .unwrap()
}

fn ids(value: &Value) -> Vec<u32> {
    value.as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect()
}

fn pairs(value: &Value) -> Vec<(u32, u32)> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|p| (p[0].as_u64().unwrap() as u32, p[1].as_u64().unwrap() as u32))
        .collect()
}

#[test]
fn parent_set_and_fixed_graph_acceptance_agree_with_the_definition_on_every_pattern() {
    let rows: Vec<Value> = serde_json::from_str(include_str!(
        "oracle_data/sample_split_parent_set_effect_reference.json"
    ))
    .unwrap();
    assert!(rows.len() > 6_000, "{}", rows.len());
    let names: Vec<String> = (0..4).map(|i| format!("v{i}")).collect();
    let frame = frame();
    let (mut accepted, mut ambiguous, mut precedes) = (0, 0, 0);
    for row in &rows {
        let pattern = row["pattern"].as_str().unwrap();
        let (t, o) =
            (row["treatment"].as_u64().unwrap() as u32, row["outcome"].as_u64().unwrap() as u32);
        let s = structure(pattern);

        let got = treatment_parent_set(&s, &names, t, o);
        match row["parent_set"].as_str() {
            Some("ambiguous") => {
                ambiguous += 1;
                let text = format!("{:?}", got.expect_err(pattern));
                assert!(text.contains("treatment_parents_ambiguous"), "{pattern} {t}: {text}");
            }
            Some("outcome_precedes_treatment") => {
                precedes += 1;
                let text = format!("{:?}", got.expect_err(pattern));
                assert!(text.contains("outcome_precedes_treatment"), "{pattern} {t}: {text}");
            }
            Some(other) => panic!("unknown expectation {other}"),
            None => {
                accepted += 1;
                let want = &row["parent_set"];
                let n = got.unwrap_or_else(|e| panic!("{pattern} t={t}: {e:?}"));
                assert_eq!(n.parents, ids(&want["parents"]), "{pattern} t={t}: parents");
                assert_eq!(n.children, ids(&want["children"]), "{pattern} t={t}: children");
                assert_eq!(n.directed, pairs(&want["directed"]), "{pattern} t={t}: directed");
                let unresolved: Vec<(u32, u32)> = n.unresolved.iter().map(|u| (u.a, u.b)).collect();
                assert_eq!(unresolved, pairs(&want["unresolved"]), "{pattern} t={t}: unresolved");
                assert_eq!(n.edges, want["edges"].as_u64().unwrap() as usize);
            }
        }

        let effect = effect_on_structure(&s, &frame, t, o, 4, 0.95);
        match row["fixed"].as_str() {
            Some("unresolved") => {
                let text = format!("{:?}", effect.expect_err(pattern));
                assert!(text.contains("structure_unresolved"), "{pattern}: {text}");
            }
            Some("outcome_precedes_treatment") => {
                let text = format!("{:?}", effect.expect_err(pattern));
                assert!(text.contains("outcome_precedes_treatment"), "{pattern}: {text}");
            }
            Some(other) => panic!("unknown expectation {other}"),
            None => {
                let got = effect.unwrap_or_else(|e| panic!("{pattern} t={t}: {e:?}"));
                assert_eq!(got.adjustment_set, ids(&row["fixed"]["parents"]), "{pattern} t={t}");
            }
        }
    }
    assert!(
        accepted > 1_000 && ambiguous > 1_000 && precedes > 100,
        "{accepted} {ambiguous} {precedes}"
    );
}

/// The whole route on frozen ten-variable data, for both protocol versions and both declared
/// scales: the treatment's parents are the true parents, the coefficient and interval equal exact
/// least squares on the analysis rows of the split by definition, and the log-linear study
/// reports the exponential images (ordered, with the delta-method standard error) for a positive
/// and a negative shift.
#[test]
fn the_parent_set_route_equals_exact_least_squares_for_both_versions_and_scales() {
    use antecedent_core::ExecutionContext;
    use antecedent_io::measured_inference::split_parent_set_effect::{
        ParentSetEffectSource, Transformation, run,
    };

    let doc: Value =
        serde_json::from_str(include_str!("oracle_data/sample_split_pipeline_reference.json"))
            .unwrap();
    let case = &doc["parent_set"];
    let load = |key: &str| -> Vec<Vec<f64>> {
        case[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect())
            .collect()
    };
    let names: Vec<String> = (0..10).map(|i| format!("v{i}")).collect();
    let (treatment, outcome) =
        (case["treatment"].as_u64().unwrap() as u32, case["outcome"].as_u64().unwrap() as u32);
    let source = |columns: Vec<Vec<f64>>, version: u32, transformation, shift| {
        let mut s = ParentSetEffectSource::new(
            names.clone(),
            columns,
            treatment,
            outcome,
            case["split_seed"].as_u64().unwrap(),
            case["discovery_seed"].as_u64().unwrap(),
            transformation,
            case["max_degree"].as_u64().unwrap() as u32,
            shift,
        );
        s.version = version;
        s
    };
    let close = |got: f64, want: &Value, what: &str| {
        let want = want.as_f64().unwrap();
        assert!((got - want).abs() <= 1e-8 * want.abs().max(1e-3), "{what}: {got} vs {want}");
    };
    let ctx = ExecutionContext::for_tests(1);
    for study in case["studies"].as_array().unwrap() {
        let version = study["version"].as_u64().unwrap() as u32;
        let linear =
            run(&source(load("linear_columns"), version, Transformation::Linear, None), &ctx)
                .unwrap_or_else(|e| panic!("v{version} linear: {e:?}"));
        assert_eq!(linear.neighbourhood.parents, ids(&case["expected_parents"]), "v{version}");
        assert_eq!(linear.analysis_rows, study["analysis_rows"].as_u64().unwrap());
        assert_eq!(linear.discovery_rows, study["discovery_rows"].as_u64().unwrap());
        let reference = &study["linear"];
        assert_eq!(linear.coefficient.degrees_of_freedom, reference["df"].as_f64().unwrap());
        close(linear.coefficient.point, &reference["point"], "linear point");
        close(linear.coefficient.standard_error, &reference["se"], "linear se");
        close(linear.coefficient.lower, &reference["lower"], "linear lower");
        close(linear.coefficient.upper, &reference["upper"], "linear upper");
        close(linear.reported.point, &reference["point"], "reported linear point");

        for (shift_key, expected) in study["log_linear"]["shifts"].as_object().unwrap() {
            let shift: f64 = shift_key.parse().unwrap();
            let logged = run(
                &source(load("log_columns"), version, Transformation::LogLinear, Some(shift)),
                &ctx,
            )
            .unwrap_or_else(|e| panic!("v{version} log {shift}: {e:?}"));
            assert_eq!(logged.neighbourhood.parents, ids(&case["expected_parents"]));
            let coefficient = &study["log_linear"]["coefficient"];
            close(logged.coefficient.point, &coefficient["point"], "log coefficient");
            close(logged.coefficient.standard_error, &coefficient["se"], "log se");
            close(logged.reported.point, &expected["point"], &format!("ratio at {shift}"));
            close(logged.reported.lower, &expected["lower"], &format!("ratio lower at {shift}"));
            close(logged.reported.upper, &expected["upper"], &format!("ratio upper at {shift}"));
            close(logged.reported.standard_error, &expected["se"], &format!("ratio se at {shift}"));
            assert!(
                logged.reported.lower <= logged.reported.point
                    && logged.reported.point <= logged.reported.upper
            );
        }
    }
}
