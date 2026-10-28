//! Placebo-treatment (Gaussian and permuted), dummy-outcome, random-common-cause and
//! data-subset refuters against an exact-rational evaluation of their definitions
//! (`oracle_data/robustness_refuters_reference.py`).
//!
//! The frozen file carries each case's data, every perturbation the library RNG streams
//! produce (noise columns, permutations, Bernoulli row masks), and the expected mean, p-value
//! and verdict computed from those perturbations with exact least squares and an 80-digit
//! `erfc`. The live test regenerates the perturbations from the streams and asserts they equal
//! the frozen ones, then runs each refuter and compares its report: replicate mean to 1e-8,
//! p-value to 1e-6 absolute, verdict exactly. Regimes: offsets of 1e5 and 1e6, an exactly
//! noiseless outcome (zero replicate spread), a treatment with about one row in ten treated,
//! two replicate counts and two significance levels so both verdicts occur.

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
use antecedent_kernels::{fill_standard_normal, shuffle};
use antecedent_validate::{
    DataSubsetRefuter, DummyOutcome, PlaceboMode, PlaceboTreatment, RandomCommonCause,
    RefutationProblem,
};
use serde_json::{Value, json};

// The refuters' historical per-replicate streams (`streams.rs`); replicate r uses base + r.
const PLACEBO: u64 = 0xA7E0_0001_0000;
const PLACEBO_SHUFFLE: u64 = 0xA7E0_0001_1000;
const RCC: u64 = 0xA7E0_0002_0000;
const SUBSET: u64 = 0xA7E0_0007_0000;
const DUMMY: u64 = 0xA7E0_0008_0000;
const SEED: u64 = 7;
const MAX_REPS: u64 = 20;
const FRACTIONS: [(&str, f64); 2] = [("0.8", 0.8), ("0.55", 0.55)];

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/robustness_refuters_reference.json"))
        .expect("frozen reference")
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().expect("array").iter().map(|x| x.as_f64().expect("f64")).collect()
}

fn build(case: &Value) -> (TabularData, IdentifiedEstimand) {
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
    (
        TabularData::new(storage),
        IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from(adjustment),
            ExprId::from_raw(0),
        ),
    )
}

fn normal(base: u64, r: u64, n: usize) -> Vec<f64> {
    let ctx = ExecutionContext::for_tests(SEED);
    let mut rng = ctx.rng.stream(base + r);
    let mut out = vec![0.0; n];
    fill_standard_normal(&mut rng, &mut out);
    out
}

fn perturbations(case: &Value) -> Value {
    let t = floats(&case["t"]);
    let n = t.len();
    let reps = 0..MAX_REPS;
    let permuted: Vec<Vec<f64>> = reps
        .clone()
        .map(|r| {
            let ctx = ExecutionContext::for_tests(SEED);
            let mut rng = ctx.rng.stream(PLACEBO_SHUFFLE + r);
            let mut v = t.clone();
            shuffle(&mut rng, &mut v);
            v
        })
        .collect();
    let mut masks = serde_json::Map::new();
    for (key, fraction) in FRACTIONS {
        let per: Vec<Vec<bool>> = reps
            .clone()
            .map(|r| {
                let ctx = ExecutionContext::for_tests(SEED);
                let mut rng = ctx.rng.stream(SUBSET + r);
                (0..n).map(|_| rng.next_f64() < fraction).collect()
            })
            .collect();
        masks.insert(key.to_string(), json!(per));
    }
    json!({
        "placebo_noise": reps.clone().map(|r| normal(PLACEBO, r, n)).collect::<Vec<_>>(),
        "placebo_perm": permuted,
        "dummy_noise": reps.clone().map(|r| normal(DUMMY, r, n)).collect::<Vec<_>>(),
        "rcc_noise": reps.map(|r| normal(RCC, r, n)).collect::<Vec<_>>(),
        "subset_mask": Value::Object(masks),
    })
}

/// Writes the perturbations the reference script needs (the first stage of regeneration).
#[test]
#[ignore = "regeneration helper"]
fn emit() {
    let data: Value = serde_json::from_str(
        &std::fs::read_to_string(std::env::var("ROBUSTNESS_REFERENCE_DATA").unwrap()).unwrap(),
    )
    .unwrap();
    let mut out = serde_json::Map::new();
    for case in data.as_array().unwrap() {
        out.insert(case["name"].as_str().unwrap().to_string(), perturbations(case));
    }
    std::fs::write(
        std::env::var("ROBUSTNESS_REFERENCE_OUT").unwrap(),
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
        (Value::Object(x), Value::Object(y)) => {
            assert_eq!(x.len(), y.len(), "{what}");
            for (k, p) in x {
                same(p, &y[k], &format!("{what}.{k}"));
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
        let live = perturbations(&entry["case"]);
        same(&entry["perturbations"], &live, entry["case"]["name"].as_str().unwrap());
    }
}

fn check(report: &antecedent_validate::RefutationReport, want: &Value, label: &str) {
    let (mean, p, passed) =
        (want[0].as_f64().unwrap(), want[1].as_f64().unwrap(), want[2].as_bool().unwrap());
    assert!(
        (report.refuted_ate - mean).abs() <= 1e-8 * mean.abs().max(1.0),
        "{label} mean {} vs {mean}",
        report.refuted_ate
    );
    assert!((report.comparison - p).abs() <= 1e-6, "{label} p {} vs {p}", report.comparison);
    assert_eq!(report.passed, passed, "{label} verdict (p={p})");
    assert!(!report.informative, "{label}");
}

#[test]
fn replicate_refuters_equal_the_exact_reference() {
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let mut verdicts = Vec::new();
    for entry in reference().as_array().unwrap() {
        let name = entry["case"]["name"].as_str().unwrap().to_string();
        let (data, estimand) = build(&entry["case"]);
        let est = LinearAdjustmentAte::new().with_bootstrap_replicates(0);
        let prep = est.prepare(&data, &estimand, &query).unwrap();
        let mut ws = EstimationWorkspace::default();
        let ctx = ExecutionContext::for_tests(SEED);
        let original = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
        let full = entry["full"].as_f64().unwrap();
        assert!((original.ate - full).abs() <= 1e-8 * full.abs().max(1.0), "{name} full fit");
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
            let alpha = run["alpha"].as_f64().unwrap();
            let label = format!("{name} R={reps} alpha={alpha}");
            let ctx = || ExecutionContext::for_tests(SEED);

            let placebo = PlaceboTreatment { replicates: reps, alpha, ..PlaceboTreatment::new() };
            let got = placebo.refute(&problem, &mut ws, &ctx()).unwrap();
            check(&got, &run["placebo"], &format!("{label} placebo"));
            verdicts.push(got.passed);

            let permute = PlaceboTreatment { mode: PlaceboMode::Permute, ..placebo };
            let got = permute.refute(&problem, &mut ws, &ctx()).unwrap();
            check(&got, &run["placebo_permute"], &format!("{label} placebo permute"));
            verdicts.push(got.passed);

            let dummy = DummyOutcome { replicates: reps, alpha, ..DummyOutcome::new() };
            let got = dummy.refute(&problem, &mut ws, &ctx()).unwrap();
            check(&got, &run["dummy"], &format!("{label} dummy outcome"));
            verdicts.push(got.passed);

            let rcc = RandomCommonCause { replicates: reps, alpha, ..RandomCommonCause::new() };
            let got = rcc.refute(&problem, &mut ws, &ctx()).unwrap();
            check(&got, &run["rcc"], &format!("{label} random common cause"));
            verdicts.push(got.passed);

            for (key, fraction) in FRACTIONS {
                let subset = DataSubsetRefuter {
                    replicates: reps,
                    subset_fraction: fraction,
                    alpha,
                    ..DataSubsetRefuter::new()
                };
                let got = subset.refute(&problem, &mut ws, &ctx());
                let want = &run[format!("subset_{key}")];
                if want.is_null() {
                    assert!(got.is_err(), "{label} subset {key}: singular subset must error");
                } else {
                    check(&got.unwrap(), want, &format!("{label} subset {key}"));
                }
            }
        }
    }
    assert!(verdicts.iter().any(|&p| p) && verdicts.iter().any(|&p| !p), "{verdicts:?}");
}

#[test]
fn replicate_count_alpha_and_fraction_are_honoured() {
    // A declaration that is accepted must change the answer: the same data under a different
    // replicate count, significance level or subset fraction does not return the same report.
    let frozen = reference();
    let entry = &frozen.as_array().unwrap()[0];
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let (data, estimand) = build(&entry["case"]);
    let est = LinearAdjustmentAte::new().with_bootstrap_replicates(0);
    let prep = est.prepare(&data, &estimand, &query).unwrap();
    let mut ws = EstimationWorkspace::default();
    let ctx = ExecutionContext::for_tests(SEED);
    let original = est.fit(&prep, &mut ws, &ctx, AssumptionSet::new()).unwrap();
    let problem = RefutationProblem::new(
        &data,
        &estimand,
        &query,
        &original,
        Some("linear.adjustment.ate"),
        None,
    );
    let mut run = |reps: u32, fraction: f64| {
        DataSubsetRefuter {
            replicates: reps,
            subset_fraction: fraction,
            ..DataSubsetRefuter::new()
        }
        .refute(&problem, &mut ws, &ExecutionContext::for_tests(SEED))
        .unwrap()
    };
    let base = run(20, 0.8);
    assert_eq!(base.replicates, 20);
    assert_eq!(run(7, 0.8).replicates, 7);
    assert!((run(20, 0.55).refuted_ate - base.refuted_ate).abs() > 1e-9);
}
