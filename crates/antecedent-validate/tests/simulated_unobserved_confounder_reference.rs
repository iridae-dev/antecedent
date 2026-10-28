//! The simulated-unobserved-confounder refuter against a 60-digit evaluation of its definition
//! (`oracle_data/simulated_unobserved_confounder_reference.py`).
//!
//! The frozen file carries each case's data, the standard-normal confounder draws the library
//! RNG streams produce, and the expected mean refit, standardized shift and verdict computed
//! from those draws: residual SDs of T and Y given Z from exact least squares, T' = T + a sd_t u,
//! Y' = Y + b sd_y u, the refit with u left OUT of the adjustment set, and the floor
//! 1e-3 sd_y/sd_t on the centre. The live test regenerates the draws from the streams and
//! asserts they equal the frozen ones, then runs the refuter for five (a, b, threshold, R)
//! settings. Regimes: offsets of 1e5 and 1e6, an exactly null full-fit effect (the floor alone
//! sets the scale), strengths of zero (no confounder: the shift is exactly zero), strength
//! 2 (large shifts) and one covariate.

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
use antecedent_kernels::fill_standard_normal;
use antecedent_validate::{RefutationProblem, UnobservedCommonCause};
use serde_json::{Value, json};

/// The refuter's historical per-replicate stream base (`streams::UNOBSERVED_COMMON_CAUSE`).
const STREAM: u64 = 0xA7E0_0006_0000;
const SEED: u64 = 7;
const MAX_REPS: u64 = 20;

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/simulated_unobserved_confounder_reference.json"))
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

fn draws(n: usize) -> Vec<Vec<f64>> {
    (0..MAX_REPS)
        .map(|r| {
            let ctx = ExecutionContext::for_tests(SEED);
            let mut rng = ctx.rng.stream(STREAM + r);
            let mut u = vec![0.0; n];
            fill_standard_normal(&mut rng, &mut u);
            u
        })
        .collect()
}

/// Writes the confounder draws the reference script needs (the first stage of regeneration).
#[test]
#[ignore = "regeneration helper"]
fn emit() {
    let data: Value = serde_json::from_str(
        &std::fs::read_to_string(std::env::var("UC_REFERENCE_DATA").unwrap()).unwrap(),
    )
    .unwrap();
    let mut out = serde_json::Map::new();
    for case in data.as_array().unwrap() {
        let n = case["t"].as_array().unwrap().len();
        out.insert(case["name"].as_str().unwrap().to_string(), json!(draws(n)));
    }
    std::fs::write(
        std::env::var("UC_REFERENCE_OUT").unwrap(),
        serde_json::to_string(&Value::Object(out)).unwrap(),
    )
    .unwrap();
}

#[test]
fn frozen_draws_are_the_library_streams() {
    for entry in reference().as_array().unwrap() {
        let n = entry["case"]["t"].as_array().unwrap().len();
        let live = draws(n);
        for (r, row) in entry["draws"].as_array().unwrap().iter().enumerate() {
            for (a, b) in floats(row).iter().zip(&live[r]) {
                assert!(
                    (a - b).abs() <= 1e-12 * a.abs().max(1.0),
                    "{} r={r}",
                    entry["case"]["name"]
                );
            }
        }
    }
}

#[test]
fn refuter_equals_the_sixty_digit_reference() {
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
        let centre = entry["centre"].as_f64().unwrap();
        assert!((original.ate - centre).abs() <= 1e-8 * centre.abs().max(1.0), "{name} centre");
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
            let (a, b, thr) = (
                run["a"].as_f64().unwrap(),
                run["b"].as_f64().unwrap(),
                run["threshold"].as_f64().unwrap(),
            );
            let refuter = UnobservedCommonCause {
                replicates: reps,
                effect_on_treatment: a,
                effect_on_outcome: b,
                std_delta_threshold: thr,
                ..UnobservedCommonCause::new()
            };
            let label = format!("{name} R={reps} a={a} b={b} thr={thr}");
            let got = refuter
                .refute(&problem, &mut ws, &ExecutionContext::for_tests(SEED))
                .unwrap_or_else(|e| panic!("{label}: {e}"));
            let want_mean = run["mean_ate"].as_f64().unwrap();
            let want_delta = run["std_delta"].as_f64().unwrap();
            assert!(
                (got.refuted_ate - want_mean).abs() <= 1e-7 * want_mean.abs().max(1.0),
                "{label} mean {} vs {want_mean}",
                got.refuted_ate
            );
            // An exactly null centre gives a delta of order 1e-1 / 1e-3 floor: compare relatively.
            assert!(
                (got.comparison - want_delta).abs() <= 1e-6 * want_delta.abs().max(1e-3),
                "{label} std_delta {} vs {want_delta}",
                got.comparison
            );
            assert_eq!(got.passed, run["passed"].as_bool().unwrap(), "{label}");
            assert_eq!(got.replicates, reps, "{label}");
            verdicts.push(got.passed);
        }
    }
    assert!(verdicts.iter().any(|&p| p) && verdicts.iter().any(|&p| !p));
}

#[test]
fn invalid_strengths_are_refused() {
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
    for (a, b) in [(-0.1, 0.5), (0.5, f64::NAN), (f64::INFINITY, 0.5)] {
        let refuter = UnobservedCommonCause {
            effect_on_treatment: a,
            effect_on_outcome: b,
            ..UnobservedCommonCause::new()
        };
        assert!(refuter.refute(&problem, &mut ws, &ctx).is_err(), "{a} {b}");
    }
}

#[test]
fn an_outcome_determined_by_the_adjustment_set_is_refused() {
    // Y is an exact function of Z, so SD(Y | Z) is rounding noise (about 1e-16): a confounder
    // "of 0.5 residual SDs" would be a perturbation of that size, and the verdict would say
    // nothing. The refuter must refuse rather than report a vacuous pass.
    let n = 32usize;
    let z0: Vec<f64> = (0..n).map(|i| (i % 8) as f64 / 8.0).collect();
    let z1: Vec<f64> = (0..n).map(|i| ((i * 3 + 1) % 7) as f64 / 8.0).collect();
    let t: Vec<f64> = (0..n).map(|i| ((i * 5) % 11) as f64 / 11.0 + z0[i] / 2.0).collect();
    let y: Vec<f64> = (0..n).map(|i| 2.0 * z0[i] + 3.0 * z1[i]).collect();
    let case = json!({"name": "determined", "t": t, "y": y, "z": [z0, z1]});
    let (data, estimand) = build(&case);
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
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
    let got = UnobservedCommonCause::new().refute(&problem, &mut ws, &ctx);
    assert!(got.is_err(), "a rounding-noise residual SD must not scale a confounder: {got:?}");
}
