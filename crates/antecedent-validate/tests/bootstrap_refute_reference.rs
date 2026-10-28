//! The iid row-bootstrap coverage refuter against an exact-rational evaluation of its
//! definition (`oracle_data/bootstrap_refute_reference.py`).
//!
//! The frozen file carries the data, the resample indices the library RNG produces for each
//! (case, replicate-count) and the expected replicate ATEs, interval and verdict computed in
//! `Fraction` arithmetic. The live test first regenerates the indices from the library RNG and
//! asserts they equal the frozen ones, so a change of the product's draws fails here instead of
//! silently de-synchronizing the reference; it then runs `BootstrapRefute` and compares the
//! report. Regimes: binary and continuous treatment, offsets of 1e5 and 1e6, a pair of
//! adjustment covariates correlated to about 1e-3, a seven-row sample whose resamples can be
//! singular, and percentile positions that fall exactly half-way between two order statistics.

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec,
    RoleHint, SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
};
use antecedent_estimate::{EstimationWorkspace, LinearAdjustmentAte};
use antecedent_expr::ExprId;
use antecedent_identify::IdentifiedEstimand;
use antecedent_kernels::unbiased_index;
use antecedent_validate::{BootstrapRefute, RefutationProblem};
use serde_json::Value;

/// The refuter's historical resample stream (`streams::BOOTSTRAP_REFUTE`).
const STREAM: u64 = 0xA7E0_0009_0000;
const SEED: u64 = 7;

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/bootstrap_refute_reference.json"))
        .expect("frozen reference")
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().expect("array").iter().map(|x| x.as_f64().expect("f64")).collect()
}

fn build(case: &Value) -> (TabularData, IdentifiedEstimand, usize) {
    let t = floats(&case["t"]);
    let y = floats(&case["y"]);
    let zs: Vec<Vec<f64>> = case["z"].as_array().expect("z").iter().map(floats).collect();
    let n = t.len();
    let mut b = CausalSchemaBuilder::new();
    let mut names = vec!["t".to_string(), "y".to_string()];
    names.extend((0..zs.len()).map(|k| format!("z{k}")));
    for (i, name) in names.iter().enumerate() {
        let role = match i {
            0 => RoleHint::TreatmentCandidate,
            1 => RoleHint::OutcomeCandidate,
            _ => RoleHint::Context,
        };
        b.add_variable(
            name.as_str(),
            ValueType::Continuous,
            SmallRoleSet::from_hint(role),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let mut data = vec![t, y];
    data.extend(zs.clone());
    let cols: Vec<OwnedColumn> = data
        .into_iter()
        .enumerate()
        .map(|(i, v)| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(i as u32),
                    Arc::from(v),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            )
        })
        .collect();
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let adjustment: Vec<VariableId> =
        (0..zs.len()).map(|k| VariableId::from_raw(2 + k as u32)).collect();
    let estimand = IdentifiedEstimand::backdoor(
        "backdoor.adjustment",
        Arc::from(adjustment),
        ExprId::from_raw(0),
    );
    (TabularData::new(storage), estimand, n)
}

fn indices(n: usize, replicates: usize) -> Vec<Vec<usize>> {
    let ctx = ExecutionContext::for_tests(SEED);
    let mut rng = ctx.rng.stream(STREAM);
    (0..replicates).map(|_| (0..n).map(|_| unbiased_index(&mut rng, n)).collect()).collect()
}

#[test]
fn frozen_resample_indices_are_the_library_stream() {
    for entry in reference().as_array().unwrap() {
        let n = entry["case"]["t"].as_array().unwrap().len();
        for run in entry["runs"].as_array().unwrap() {
            let reps = run["replicates"].as_u64().unwrap() as usize;
            let frozen: Vec<Vec<usize>> = run["indices"]
                .as_array()
                .unwrap()
                .iter()
                .map(|r| {
                    r.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect()
                })
                .collect();
            assert_eq!(frozen, indices(n, reps), "{}", entry["case"]["name"]);
        }
    }
}

/// Writes the indices the reference script needs (the first stage of regeneration).
#[test]
#[ignore = "regeneration helper"]
fn emit_indices() {
    let data: Value = serde_json::from_str(
        &std::fs::read_to_string(std::env::var("BOOTSTRAP_REFERENCE_DATA").unwrap()).unwrap(),
    )
    .unwrap();
    let settings = [
        ("plain", vec![41, 11, 21, 51]),
        ("offset", vec![41, 11]),
        ("near_collinear", vec![21]),
        ("continuous_treatment", vec![31]),
        ("tiny", vec![21]),
    ];
    let mut out = serde_json::Map::new();
    for case in data.as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let n = case["t"].as_array().unwrap().len();
        let mut per = serde_json::Map::new();
        for (nm, reps) in &settings {
            if *nm == name {
                for r in reps {
                    per.insert(r.to_string(), serde_json::json!(indices(n, *r)));
                }
            }
        }
        out.insert(name.to_string(), Value::Object(per));
    }
    std::fs::write(
        std::env::var("BOOTSTRAP_REFERENCE_OUT").unwrap(),
        serde_json::to_string(&Value::Object(out)).unwrap(),
    )
    .unwrap();
}

fn close(got: f64, want: f64, what: &str) {
    let tol = 1e-8 * want.abs().max(1.0);
    assert!((got - want).abs() <= tol, "{what}: {got} vs {want}");
}

#[test]
fn coverage_report_equals_the_exact_percentile_interval() {
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    for entry in reference().as_array().unwrap() {
        let name = entry["case"]["name"].as_str().unwrap().to_string();
        let (data, estimand, _n) = build(&entry["case"]);
        let est = LinearAdjustmentAte::new().with_bootstrap_replicates(0);
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = EstimationWorkspace::default();
        let ctx = ExecutionContext::for_tests(SEED);
        let original = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        close(original.ate, entry["full"].as_f64().unwrap(), &format!("{name} full-sample ATE"));
        let problem = RefutationProblem::new(
            &data,
            &estimand,
            &query,
            &original,
            Some("linear.adjustment.ate"),
            None,
        );
        for run in entry["runs"].as_array().unwrap() {
            let reps = run["replicates"].as_u64().unwrap() as u32;
            let level = run["level"].as_f64().unwrap();
            let refuter =
                BootstrapRefute { replicates: reps, ci_level: level, ..BootstrapRefute::new() };
            let ctx = ExecutionContext::for_tests(SEED);
            let got = refuter.refute(&problem, &mut ws, &ctx);
            let label = format!("{name} R={reps} level={level}");
            if run["singular"].as_bool().unwrap() {
                assert!(got.is_err(), "{label}: a singular resample must not yield a report");
                continue;
            }
            let report = got.unwrap_or_else(|e| panic!("{label}: {e}"));
            assert_eq!(report.replicates, reps, "{label}");
            close(report.refuted_ate, run["mean"].as_f64().unwrap(), &format!("{label} mean"));
            close(report.comparison, run["width"].as_f64().unwrap(), &format!("{label} width"));
            // The verdict flips only at the interval end points; skip it when the centre sits
            // within rounding of an end point.
            let (lo, hi, full) = (
                run["lo"].as_f64().unwrap(),
                run["hi"].as_f64().unwrap(),
                run["full"].as_f64().unwrap(),
            );
            if (full - lo).abs().min((full - hi).abs()) > 1e-9 * full.abs().max(1.0) {
                assert_eq!(report.passed, run["passed"].as_bool().unwrap(), "{label}");
            }
        }
    }
}

#[test]
fn verdicts_cover_both_outcomes() {
    let verdicts: Vec<bool> = reference()
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|e| e["runs"].as_array().unwrap().clone())
        .filter(|r| !r["singular"].as_bool().unwrap())
        .map(|r| r["passed"].as_bool().unwrap())
        .collect();
    assert!(verdicts.iter().any(|&p| p) && verdicts.iter().any(|&p| !p), "{verdicts:?}");
}
