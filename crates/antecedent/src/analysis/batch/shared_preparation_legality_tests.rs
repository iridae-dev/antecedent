use std::sync::Arc;

use antecedent_core::{AverageEffectQuery, VariableId};
use antecedent_data::TabularData;
use antecedent_expr::{ExprId, IdentifiedEstimand};

use super::SharedBatchDesign;

fn table(altered_row: Option<usize>) -> TabularData {
    let n = 60;
    let mut z: Vec<f64> = (0..n).map(|i| (i as f64) * 0.1).collect();
    if let Some(row) = altered_row {
        z[row] += 1.0;
    }
    let t: Vec<f64> = (0..n).map(|i| f64::from(u8::from(i % 2 == 0))).collect();
    let y: Vec<f64> = (0..n).map(|i| 2.0 * t[i] + 0.5 * (i as f64) * 0.1).collect();
    TabularData::from_f64_columns([("z", &z[..]), ("t", &t[..]), ("y", &y[..])]).expect("table")
}

#[test]
fn a_shared_design_must_carry_the_query_rows_and_values() {
    let z = VariableId::from_raw(0);
    let sets: Vec<Arc<[VariableId]>> = vec![Arc::from(vec![z])];
    let estimand = IdentifiedEstimand::backdoor(
        "backdoor.adjustment",
        Arc::from(vec![z]),
        ExprId::from_raw(0),
    );
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(1), VariableId::from_raw(2));
    let estimator = antecedent_estimate::AipwAte::new();
    let base = table(None);

    let mut honest = estimator.prepare(&base, &estimand, &query).expect("prepared");
    let shared = SharedBatchDesign::compile(&base, &sets, 5, 1).expect("shared design");
    assert!(shared.apply_to_propensity(&mut honest).expect("same rows and values"));

    let forged = SharedBatchDesign::compile(&table(Some(17)), &sets, 5, 1).expect("forged");
    let mut problem = estimator.prepare(&base, &estimand, &query).expect("prepared");
    let before = problem.design_matrix.clone();
    let refused = format!("{:?}", forged.apply_to_propensity(&mut problem).unwrap_err());
    assert!(refused.contains("rewrite_legality"), "{refused}");
    assert_eq!(problem.design_matrix, before, "a refused share leaves the design untouched");
    assert!(!problem.shared_design);
}
