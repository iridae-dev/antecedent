//! The static-mediation refuter suite against an exact-rational evaluation of its definitions
//! (`oracle_data/static_mediation_reference.py`).
//!
//! The model is T -> M, T -> Y, M -> Y, Z -> M, Z -> Y with a treatment contrast of one, so the
//! refit estimates are products and sums of regression coefficients that the script evaluates in
//! `Fraction` arithmetic from the perturbations the library RNG streams produce (the placebo
//! mediator noise, the random-common-cause covariate and the Bernoulli 80% row masks). The live
//! test regenerates the perturbations from the streams and asserts they equal the frozen ones,
//! then runs `refute_static_mediation` for three requested contrasts and compares every report
//! (replicate mean, p-value, verdict) and the mediator-range overlap diagnostic, whose cases
//! include arm ranges that touch at exactly one value and ranges that miss by 1e-6.

use antecedent_core::{
    AssumptionSet, ExecutionContext, MediationContrast, MediationQuery, VariableId,
};
use antecedent_data::TabularData;
use antecedent_estimate::estimate_static_mediation;
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_kernels::fill_standard_normal;
use antecedent_validate::mediation::refute_static_mediation;
use serde_json::{Value, json};

// Historical streams of `refute_static_mediation`: placebo mediator `PLACEBO + 100 r + j`,
// random common cause `RCC + r`, row subset `SUBSET + r`, for replicate r in 0..20.
const PLACEBO: u64 = 0x1300_2000;
const RCC: u64 = 0x1300_4000;
const SUBSET: u64 = 0x1300_5000;
const SEED: u64 = 11;
const REPS: u64 = 20;

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/static_mediation_reference.json"))
        .expect("frozen reference")
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().expect("array").iter().map(|x| x.as_f64().expect("f64")).collect()
}

fn contrast(name: &str) -> MediationContrast {
    match name {
        "Total" => MediationContrast::Total,
        "NaturalDirect" => MediationContrast::NaturalDirect,
        "NaturalIndirect" => MediationContrast::NaturalIndirect,
        other => panic!("{other}"),
    }
}

/// ids: t = 0, m = 1, y = 2, z = 3.
fn build(case: &Value) -> (TabularData, Dag) {
    let (t, m, y, z) =
        (floats(&case["t"]), floats(&case["m"]), floats(&case["y"]), floats(&case["z"]));
    let data =
        TabularData::from_f64_columns([("t", &t[..]), ("m", &m[..]), ("y", &y[..]), ("z", &z[..])])
            .unwrap();
    let mut dag = Dag::with_variables(4);
    let [t, m, y, z] = [0, 1, 2, 3].map(DenseNodeId::from_raw);
    for (from, to) in [(t, m), (t, y), (m, y), (z, m), (z, y)] {
        dag.insert_directed(from, to).unwrap();
    }
    (data, dag)
}

fn query(name: &str) -> MediationQuery {
    MediationQuery::binary(
        VariableId::from_raw(0),
        VariableId::from_raw(2),
        [VariableId::from_raw(1)],
        contrast(name),
    )
}

fn noise(stream: u64, n: usize) -> Vec<f64> {
    let ctx = ExecutionContext::for_tests(SEED);
    let mut rng = ctx.rng.stream(stream);
    let mut out = vec![0.0; n];
    fill_standard_normal(&mut rng, &mut out);
    out
}

fn perturbations(n: usize) -> Value {
    let masks: Vec<Vec<bool>> = (0..REPS)
        .map(|r| {
            let ctx = ExecutionContext::for_tests(SEED);
            let mut rng = ctx.rng.stream(SUBSET + r);
            (0..n).map(|_| rng.next_f64() < 0.8).collect()
        })
        .collect();
    json!({
        "placebo_noise": (0..REPS).map(|r| noise(PLACEBO + r * 100, n)).collect::<Vec<_>>(),
        "rcc_noise": (0..REPS).map(|r| noise(RCC + r, n)).collect::<Vec<_>>(),
        "subset_mask": masks,
    })
}

/// Writes the perturbations the reference script needs (the first stage of regeneration).
#[test]
#[ignore = "regeneration helper"]
fn emit() {
    let data: Value = serde_json::from_str(
        &std::fs::read_to_string(std::env::var("STATIC_MEDIATION_REFERENCE_DATA").unwrap())
            .unwrap(),
    )
    .unwrap();
    let mut out = serde_json::Map::new();
    for case in data.as_array().unwrap() {
        let n = case["t"].as_array().unwrap().len();
        out.insert(case["name"].as_str().unwrap().to_string(), perturbations(n));
    }
    std::fs::write(
        std::env::var("STATIC_MEDIATION_REFERENCE_OUT").unwrap(),
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
fn frozen_perturbations_are_the_library_streams() {
    for entry in reference().as_array().unwrap() {
        let n = entry["case"]["t"].as_array().unwrap().len();
        same(&entry["perturbations"], &perturbations(n), entry["case"]["name"].as_str().unwrap());
    }
}

fn check(report: &antecedent_validate::RefutationReport, want: &Value, label: &str) {
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
}

#[test]
fn suite_equals_the_exact_reference() {
    let ctx = || ExecutionContext::for_tests(SEED);
    let mut verdicts = Vec::new();
    for entry in reference().as_array().unwrap() {
        let name = entry["case"]["name"].as_str().unwrap();
        let (data, dag) = build(&entry["case"]);
        for run in entry["runs"].as_array().unwrap() {
            let cname = run["contrast"].as_str().unwrap();
            let q = query(cname);
            let mut original =
                estimate_static_mediation(&data, &dag, &q, AssumptionSet::new(), 0, &[], &ctx())
                    .unwrap();
            let want_orig = run["original"].as_f64().unwrap();
            assert!(
                (original.effect.ate - want_orig).abs() <= 1e-8 * want_orig.abs().max(1.0),
                "{name} {cname}: original {} vs {want_orig}",
                original.effect.ate
            );
            original.effect.ate += run["shift"].as_f64().unwrap();
            let label = format!("{name} {cname} shift={}", run["shift"]);

            let cheap = refute_static_mediation(&data, &dag, &q, &original, false, &ctx()).unwrap();
            let full = refute_static_mediation(&data, &dag, &q, &original, true, &ctx());
            if run["subset"].is_null() {
                assert!(full.is_err(), "{label}: a singular subset must not yield a report");
                continue;
            }
            let full = full.unwrap();
            let by_id = |reports: &[antecedent_validate::RefutationReport], id: &str| {
                reports.iter().find(|r| r.refuter.as_ref() == id).cloned()
            };
            for reports in [&cheap, &full] {
                check(
                    &by_id(reports, "mediation.static.placebo_mediator").unwrap(),
                    &run["placebo"],
                    &format!("{label} placebo"),
                );
                check(
                    &by_id(reports, "mediation.static.random_common_cause").unwrap(),
                    &run["rcc"],
                    &format!("{label} rcc"),
                );
                let overlap = by_id(reports, "mediation.static.mediator_overlap").unwrap();
                assert_eq!(overlap.passed, entry["overlap"].as_bool().unwrap(), "{label} overlap");
                assert_eq!(overlap.comparison, f64::from(entry["overlap"].as_bool().unwrap()));
            }
            assert!(by_id(&cheap, "mediation.static.subset").is_none(), "{label}");
            check(
                &by_id(&full, "mediation.static.subset").unwrap(),
                &run["subset"],
                &format!("{label} subset"),
            );
            verdicts.extend(full.iter().map(|r| r.passed));
        }
    }
    assert!(verdicts.iter().any(|&p| p) && verdicts.iter().any(|&p| !p));
}
