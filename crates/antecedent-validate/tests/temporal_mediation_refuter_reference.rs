//! The temporal-mediation refuter suite against an exact-rational evaluation of its definitions
//! (`oracle_data/temporal_mediation_refuter_reference.py`).
//!
//! Series with missing time points (single gaps, multi-step restarts, gaps in the treatment, a
//! mediator, the outcome and the covariate, two mediators at treatment lag two, a scaled
//! contrast) are refitted by the script in `Fraction` arithmetic over the lag-aligned intact
//! windows. The perturbations are produced by the library RNG streams (the placebo-mediator
//! noise, the random-common-cause covariate that shares the first noise vector, and the first
//! uniform of the contiguous-window stream); the live test regenerates them, asserts they equal
//! the frozen ones, then runs `refute_temporal_mediation_with_gaps` and compares every replicate
//! mean, p-value and verdict plus the binary-treatment mediator-support diagnostic, which is
//! computed on the intact windows only.

use std::sync::Arc;

use antecedent_core::{
    CausalSchemaBuilder, ExecutionContext, Intervention, Lag, MeasurementSpec, MediationContrast,
    MediationQuery, RoleHint, SmallRoleSet, Value as CoreValue, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, LaggedColumn, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
    TimeSeriesData, ValidityBitmap,
};
use antecedent_estimate::TemporalMediationEstimator;
use antecedent_expr::{CausalExprArena, IdentifiedEstimand};
use antecedent_kernels::fill_standard_normal;
use antecedent_validate::RefutationReport;
use antecedent_validate::mediation::refute_temporal_mediation_with_gaps;
use serde_json::{Value, json};

// Historical streams of the suite: placebo mediator `PLACEBO + r` (shared with the random common
// cause), further mediators `PLACEBO_MORE + 100 r + j`, contiguous window `WINDOW + r`.
const PLACEBO: u64 = 0xF014_0000;
const PLACEBO_MORE: u64 = 0xF014_8000;
const WINDOW: u64 = 0xF015_0000;
const SEED: u64 = 17;
const REPS: u64 = 20;

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/temporal_mediation_refuter_reference.json"))
        .expect("frozen reference")
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().expect("array").iter().map(|x| x.as_f64().expect("f64")).collect()
}

fn indices(v: &Value) -> Vec<usize> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| usize::try_from(x.as_u64().unwrap()).unwrap())
        .collect()
}

fn contrast(name: &str) -> MediationContrast {
    match name {
        "Total" => MediationContrast::Total,
        "NaturalDirect" => MediationContrast::NaturalDirect,
        "Mediated" => MediationContrast::Mediated,
        other => panic!("{other}"),
    }
}

fn var(i: usize) -> VariableId {
    VariableId::from_raw(u32::try_from(i).unwrap())
}

/// Variables: 0 treatment, 1..=q mediators, q + 1 outcome, q + 2 covariate.
fn build(case: &Value) -> TimeSeriesData {
    let q = usize::try_from(case["q"].as_u64().unwrap()).unwrap();
    let n = case["t"].as_array().unwrap().len();
    let mut builder = CausalSchemaBuilder::new();
    for index in 0..(q + 3) {
        builder
            .add_variable(
                format!("v{index}"),
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
    }
    let schema = builder.build().unwrap();
    let mut series = vec![(floats(&case["t"]), indices(&case["miss_t"]))];
    for (j, m) in case["m"].as_array().unwrap().iter().enumerate() {
        series.push((floats(m), indices(&case["miss_m"][j])));
    }
    series.push((floats(&case["y"]), indices(&case["miss_y"])));
    series.push((floats(&case["z"]), indices(&case["miss_z"])));
    let cols: Vec<OwnedColumn> = series
        .into_iter()
        .enumerate()
        .map(|(i, (values, missing))| {
            let valid: Vec<bool> = (0..n).map(|t| !missing.contains(&t)).collect();
            OwnedColumn::Float64(
                Float64Column::new(var(i), Arc::from(values), ValidityBitmap::from_bools(&valid))
                    .unwrap(),
            )
        })
        .collect();
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap()
}

fn noise(stream: u64, n: usize) -> Vec<f64> {
    let ctx = ExecutionContext::for_tests(SEED);
    let mut rng = ctx.rng.stream(stream);
    let mut out = vec![0.0; n];
    fill_standard_normal(&mut rng, &mut out);
    out
}

fn perturbations(n: usize, q: usize) -> Value {
    let placebo: Vec<Vec<Vec<f64>>> = (0..REPS)
        .map(|r| {
            (0..q)
                .map(|j| {
                    let stream =
                        if j == 0 { PLACEBO + r } else { PLACEBO_MORE + r * 100 + j as u64 };
                    noise(stream, n)
                })
                .collect()
        })
        .collect();
    let window_u: Vec<f64> = (0..REPS)
        .map(|r| ExecutionContext::for_tests(SEED).rng.stream(WINDOW + r).next_f64())
        .collect();
    json!({ "noise": placebo, "window_u": window_u })
}

fn case_shape(case: &Value) -> (usize, usize) {
    (case["t"].as_array().unwrap().len(), usize::try_from(case["q"].as_u64().unwrap()).unwrap())
}

/// Writes the perturbations the reference script needs (the first stage of regeneration).
#[test]
#[ignore = "regeneration helper"]
fn emit() {
    let data: Value = serde_json::from_str(
        &std::fs::read_to_string(std::env::var("TEMPORAL_MEDIATION_REFUTER_DATA").unwrap())
            .unwrap(),
    )
    .unwrap();
    let mut out = serde_json::Map::new();
    for case in data.as_array().unwrap() {
        let (n, q) = case_shape(case);
        out.insert(case["name"].as_str().unwrap().to_string(), perturbations(n, q));
    }
    std::fs::write(
        std::env::var("TEMPORAL_MEDIATION_REFUTER_OUT").unwrap(),
        serde_json::to_string(&Value::Object(out)).unwrap(),
    )
    .unwrap();
}

fn same(a: &Value, b: &Value, what: &str) {
    match (a, b) {
        (Value::Array(x), Value::Array(y)) => {
            assert_eq!(x.len(), y.len(), "{what}");
            for (i, (p, q)) in x.iter().zip(y).enumerate() {
                same(p, q, &format!("{what}[{i}]"));
            }
        }
        (Value::Number(x), Value::Number(y)) => {
            let (x, y) = (x.as_f64().unwrap(), y.as_f64().unwrap());
            assert!((x - y).abs() <= 1e-12 * x.abs().max(1.0), "{what}: {x} vs {y}");
        }
        _ => assert_eq!(a, b, "{what}"),
    }
}

#[test]
fn temporal_mediation_refuter_frozen_perturbations_are_the_library_streams() {
    for entry in reference().as_array().unwrap() {
        let (n, q) = case_shape(&entry["case"]);
        same(
            &entry["perturbations"],
            &perturbations(n, q),
            entry["case"]["name"].as_str().unwrap(),
        );
    }
}

fn check(report: &RefutationReport, want: &Value, null: f64, label: &str) {
    let (mean, p, passed) =
        (want[0].as_f64().unwrap(), want[1].as_f64().unwrap(), want[2].as_bool().unwrap());
    assert!(
        (report.refuted_ate - mean).abs() <= 1e-7 * mean.abs().max(1.0),
        "{label} mean {} vs {mean}",
        report.refuted_ate
    );
    assert!((report.comparison - p).abs() <= 1e-5, "{label} p {} vs {p}", report.comparison);
    assert_eq!(report.passed, passed, "{label} verdict (p={p})");
    assert_eq!(report.replicates, 20, "{label}");
    assert!((report.original_ate - null).abs() <= 1e-12 * null.abs().max(1.0), "{label} null");
}

fn by_id(reports: &[RefutationReport], id: &str) -> Option<RefutationReport> {
    reports.iter().find(|r| r.refuter.as_ref() == id).cloned()
}

#[test]
fn temporal_mediation_refuter_suite_equals_the_exact_reference() {
    let ctx = || ExecutionContext::for_tests(SEED);
    let mut verdicts = Vec::new();
    for entry in reference().as_array().unwrap() {
        let case = &entry["case"];
        let name = case["name"].as_str().unwrap();
        let (_, q) = case_shape(case);
        let h = u32::try_from(case["h"].as_u64().unwrap()).unwrap();
        let delta = case["delta"].as_f64().unwrap();
        let gaps = case["gaps"].as_bool().unwrap();
        let data = build(case);
        let mediators: Vec<VariableId> = (1..=q).map(var).collect();
        let (treatment, outcome, covariate) = (var(0), var(q + 1), var(q + 2));
        let adjustment: Vec<LaggedColumn> = case["adj"]
            .as_array()
            .unwrap()
            .iter()
            .map(|l| LaggedColumn {
                variable: covariate,
                lag: Lag::from_raw(u32::try_from(l.as_u64().unwrap()).unwrap()),
            })
            .collect();
        let estimator = TemporalMediationEstimator::new()
            .with_allow_natural_controlled_alias(true)
            .with_allow_gaps(gaps);
        for run in entry["runs"].as_array().unwrap() {
            let cname = run["contrast"].as_str().unwrap();
            let mut query =
                MediationQuery::binary(treatment, outcome, mediators.clone(), contrast(cname))
                    .with_horizons([h])
                    .unwrap();
            query.active = Intervention::set(treatment, CoreValue::f64(delta));
            let mut arena = CausalExprArena::new();
            let functional = arena.temporal_mediation_ate(
                query.treatment,
                query.outcome,
                &query.mediators,
                CoreValue::f64(delta),
                CoreValue::f64(0.0),
            );
            let estimand = IdentifiedEstimand::temporal_mediation(
                "temporal_mediation.mediated",
                Arc::clone(&query.mediators),
                functional,
            );
            let mut original = estimator
                .estimate_with_adjustment(&data, &estimand, &query, &adjustment, &[], &ctx())
                .unwrap();
            let want_orig = run["original"].as_f64().unwrap();
            assert!(
                (original.effect.ate - want_orig).abs() <= 1e-8 * want_orig.abs().max(1.0),
                "{name} {cname}: original {} vs {want_orig}",
                original.effect.ate
            );
            let published = want_orig + run["shift"].as_f64().unwrap();
            original.effect.ate = published;
            let label = format!("{name} {cname} shift={}", run["shift"]);

            let cheap = refute_temporal_mediation_with_gaps(
                &data,
                &estimand,
                &query,
                &original,
                false,
                &adjustment,
                gaps,
                &ctx(),
            )
            .unwrap();
            let full = refute_temporal_mediation_with_gaps(
                &data,
                &estimand,
                &query,
                &original,
                true,
                &adjustment,
                gaps,
                &ctx(),
            )
            .unwrap();
            for reports in [&cheap, &full] {
                check(
                    &by_id(reports, "mediation.placebo_mediator").unwrap(),
                    &run["placebo"],
                    0.0,
                    &format!("{label} placebo"),
                );
                check(
                    &by_id(reports, "mediation.random_common_cause").unwrap(),
                    &run["rcc"],
                    published,
                    &format!("{label} rcc"),
                );
                let support = by_id(reports, "mediation.binary_mediator_support");
                if entry["support"].is_null() {
                    assert!(support.is_none(), "{label}: support reported for a non-binary arm");
                } else {
                    let want = &entry["support"];
                    let got = support.unwrap_or_else(|| panic!("{label}: no support report"));
                    let fraction = want["fraction"].as_f64().unwrap();
                    assert!((got.comparison - fraction).abs() <= 1e-12, "{label} fraction");
                    assert_eq!(got.passed, want["intersect"].as_bool().unwrap(), "{label} support");
                    assert_eq!(got.replicates, 1);
                    assert!(got.informative);
                    assert!(
                        (got.refuted_ate - published).abs() <= 1e-12 * published.abs().max(1.0)
                    );
                }
            }
            assert!(by_id(&cheap, "mediation.contiguous_window").is_none(), "{label}");
            check(
                &by_id(&full, "mediation.contiguous_window").unwrap(),
                &run["window"],
                published,
                &format!("{label} window"),
            );
            verdicts.extend(full.iter().filter(|r| r.replicates == 20).map(|r| r.passed));
        }
    }
    assert!(verdicts.iter().any(|&p| p) && verdicts.iter().any(|&p| !p));
}
