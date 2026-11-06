use super::*;
use antecedent_core::{CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType};
use antecedent_data::{Float64Column, OwnedColumn, OwnedColumnarStorage};
use antecedent_graph::DenseNodeId;

fn one_result() -> StudyResult {
    let n = 20;
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 0..n {
        let ti = f64::from(u32::try_from(i % 2).unwrap());
        t[i] = ti;
        y[i] = 2.0 * ti;
    }
    let mut builder = CausalSchemaBuilder::new();
    for (name, role) in [("t", RoleHint::TreatmentCandidate), ("y", RoleHint::OutcomeCandidate)] {
        builder
            .add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(role),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
    }
    let schema = builder.build().unwrap();
    let cols: Vec<OwnedColumn> = [t, y]
        .into_iter()
        .enumerate()
        .map(|(i, values)| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(u32::try_from(i).unwrap()),
                    Arc::from(values),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            )
        })
        .collect();
    let data = TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
    let mut dag = Dag::with_variables(2);
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let ctx = ExecutionContext::for_tests(1);
    Study::tabular(data)
        .graph(dag)
        .query(query)
        .bootstrap_replicates(0)
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap()
}

#[test]
fn max_t_failure_leaves_a_diagnostic_instead_of_silent_none() {
    let base = one_result();
    let mut results = vec![base.clone(), base];
    // A zero-variance claim next to a real one: `max_t_critical` refuses
    // ("max-t requires positive finite SEs") on the degenerate diagonal entry.
    let cov =
        antecedent_estimate::JointCovariance { dim: 2, values: Arc::from([0.0, 0.0, 0.0, 4.0]) };
    assert!(antecedent_estimate::max_t_critical(&cov, 0.95, 4096, 1).is_err());

    let crit = resolve_max_t_critical(&mut results, &cov, "test interval");
    assert!(crit.is_none());
    for result in &results {
        assert!(
            result.diagnostics.iter().any(|d| &*d.code == "batch.joint_if.max_t_unavailable"
                && d.severity == antecedent_core::DiagnosticSeverity::Warning),
            "expected a max_t_unavailable warning, got {:?}",
            result.diagnostics
        );
    }
}
