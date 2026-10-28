//! The leave-one-out adjustment-covariate check against an exact-rational evaluation of its
//! definition (`oracle_data/graph_refute_reference.py`).
//!
//! The reference refits the outcome regression with every covariate dropped in turn on the rows
//! complete for the full adjustment set and forms the relative change against the larger of the
//! full-fit magnitude and a thousandth of SD(Y)/SD(T). Regimes: a confounded treatment (large
//! change), an exactly null full-fit effect (the floor alone sets the denominator), covariates
//! with missing values in the last adjustment column (the rows must stay fixed across drops),
//! a single covariate (dropping leaves the empty set) and several thresholds so both verdicts
//! occur. The empty-adjustment-set short cut and the threshold declaration are checked too.

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
use antecedent_validate::{GraphRefuter, RefutationProblem};
use serde_json::Value;

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/graph_refute_reference.json"))
        .expect("frozen reference")
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().expect("array").iter().map(|x| x.as_f64().expect("f64")).collect()
}

fn build(case: &Value, adjustment_len: Option<usize>) -> (TabularData, IdentifiedEstimand) {
    let t = floats(&case["t"]);
    let y = floats(&case["y"]);
    let zs: Vec<Vec<f64>> = case["z"].as_array().expect("z").iter().map(floats).collect();
    let invalid: Vec<usize> = case["z_invalid_rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
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
    let last = data.len() - 1;
    let cols: Vec<OwnedColumn> = data
        .into_iter()
        .enumerate()
        .map(|(i, v)| {
            let validity = if i == last && !invalid.is_empty() {
                let mut bytes = vec![0u8; n.div_ceil(8)];
                for row in 0..n {
                    if !invalid.contains(&row) {
                        bytes[row / 8] |= 1 << (row % 8);
                    }
                }
                ValidityBitmap::from_bytes(bytes, n).unwrap()
            } else {
                ValidityBitmap::all_valid(n)
            };
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(i as u32), Arc::from(v), validity).unwrap(),
            )
        })
        .collect();
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let k = adjustment_len.unwrap_or(zs.len());
    let adjustment: Vec<VariableId> = (0..k).map(|j| VariableId::from_raw(2 + j as u32)).collect();
    (
        TabularData::new(storage),
        IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from(adjustment),
            ExprId::from_raw(0),
        ),
    )
}

fn close(got: f64, want: f64, what: &str) {
    assert!((got - want).abs() <= 1e-7 * want.abs().max(1.0), "{what}: {got} vs {want}");
}

fn original(
    data: &TabularData,
    estimand: &IdentifiedEstimand,
    query: &AverageEffectQuery,
) -> antecedent_estimate::EffectEstimate {
    let est = LinearAdjustmentAte::new().with_bootstrap_replicates(0);
    let prep = est.prepare(data, estimand, query).unwrap();
    let mut ws = EstimationWorkspace::default();
    est.fit(&prep, &mut ws, &ExecutionContext::for_tests(7), AssumptionSet::new()).unwrap()
}

#[test]
fn leave_one_out_equals_the_exact_reference() {
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let mut verdicts = Vec::new();
    for entry in reference().as_array().unwrap() {
        let name = entry["case"]["name"].as_str().unwrap();
        let (data, estimand) = build(&entry["case"], None);
        let original = original(&data, &estimand, &query);
        close(original.ate, entry["ate0"].as_f64().unwrap(), &format!("{name} full fit"));
        let problem = RefutationProblem::new(
            &data,
            &estimand,
            &query,
            &original,
            Some("linear.adjustment.ate"),
            None,
        );
        let mut ws = EstimationWorkspace::default();
        for run in entry["runs"].as_array().unwrap() {
            let thr = run["threshold"].as_f64().unwrap();
            let refuter = GraphRefuter { rel_delta_threshold: thr, ..GraphRefuter::new() };
            let got = refuter
                .refute(&problem, &mut ws, &ExecutionContext::for_tests(7))
                .unwrap_or_else(|e| panic!("{name} thr={thr}: {e}"));
            let label = format!("{name} thr={thr}");
            close(got.comparison, run["comparison"].as_f64().unwrap(), &format!("{label} delta"));
            close(got.refuted_ate, run["refuted"].as_f64().unwrap(), &format!("{label} ate"));
            assert_eq!(
                got.replicates as usize,
                entry["drops"].as_array().unwrap().len(),
                "{label}"
            );
            assert_eq!(got.passed, run["passed"].as_bool().unwrap(), "{label}");
            assert!(!got.informative, "{label}");
            verdicts.push(got.passed);
        }
    }
    assert!(verdicts.iter().any(|&p| p) && verdicts.iter().any(|&p| !p));
}

#[test]
fn empty_adjustment_set_has_nothing_to_drop() {
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let frozen = reference();
    let entry = &frozen.as_array().unwrap()[0];
    let (data, estimand) = build(&entry["case"], Some(0));
    let original = original(&data, &estimand, &query);
    let problem = RefutationProblem::new(
        &data,
        &estimand,
        &query,
        &original,
        Some("linear.adjustment.ate"),
        None,
    );
    let mut ws = EstimationWorkspace::default();
    let got =
        GraphRefuter::new().refute(&problem, &mut ws, &ExecutionContext::for_tests(7)).unwrap();
    assert!(got.passed);
    assert_eq!(got.replicates, 0);
    assert_eq!(got.comparison, 0.0);
    assert_eq!(got.refuted_ate, original.ate);
}
