//! Arrow null `Float64` cells are missing, never observed zeros.
//!
//! A null cell loaded from an Arrow record batch must leave the analysis
//! exactly as if the row had been dropped before loading.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{RefuteSuite, Study};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ContinuousDomain, ExecutionContext, GridSpec,
    ResponseFunctional, ResponseIdentification, ResponseQuery, ResponseValue, VariableId,
};
use antecedent_data::{TabularData, tabular_from_record_batch};
use antecedent_graph::{Dag, DenseNodeId};
use arrow_array::{Float64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};

const N: usize = 240;
const NULL_ROW: usize = 37;

/// `(treatment, outcome, confounder)` columns with `Y = 1 + 2T + 0.8Z + e`.
fn columns(binary: bool) -> [Vec<f64>; 3] {
    let z: Vec<f64> = (0..N).map(|i| (i as f64 / 17.0).sin()).collect();
    let t: Vec<f64> = (0..N)
        .map(|i| {
            let latent = z[i] + (i as f64 / 11.0).cos() + (i % 7) as f64 * 0.03;
            if binary { f64::from(u8::from(latent > 0.3)) } else { latent }
        })
        .collect();
    let y: Vec<f64> =
        (0..N).map(|i| 1.0 + 2.0 * t[i] + 0.8 * z[i] + (i as f64 / 13.0).sin() * 0.5).collect();
    [t, y, z]
}

/// Record batch where column `null_col` is null at [`NULL_ROW`], plus the
/// same data with that row removed.
fn with_null_and_dropped(binary: bool, null_col: usize) -> (TabularData, TabularData) {
    let cols = columns(binary);
    let names = ["treatment", "outcome", "confounder"];
    let schema = Arc::new(Schema::new(
        names.iter().map(|n| Field::new(*n, DataType::Float64, true)).collect::<Vec<_>>(),
    ));
    let arrays: Vec<Arc<dyn arrow_array::Array>> = cols
        .iter()
        .enumerate()
        .map(|(k, c)| {
            let values: Vec<Option<f64>> = c
                .iter()
                .enumerate()
                .map(|(i, &v)| (!(k == null_col && i == NULL_ROW)).then_some(v))
                .collect();
            Arc::new(Float64Array::from(values)) as Arc<dyn arrow_array::Array>
        })
        .collect();
    let batch = RecordBatch::try_new(schema, arrays).unwrap();
    let with_null = tabular_from_record_batch(&batch).unwrap().data;

    let dropped: Vec<Vec<f64>> = cols
        .iter()
        .map(|c| c.iter().enumerate().filter(|(i, _)| *i != NULL_ROW).map(|(_, &v)| v).collect())
        .collect();
    let dropped =
        TabularData::from_f64_columns(names.iter().zip(&dropped).map(|(n, c)| (*n, c.as_slice())))
            .unwrap();
    (with_null, dropped)
}

fn confounded_dag() -> Dag {
    let mut graph = Dag::with_variables(3);
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    graph
}

fn response_curve(data: TabularData) -> Vec<f64> {
    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(vec![-0.5, 0.0, 0.5].into()),
        ),
    });
    let result = Study::tabular(data)
        .graph(confounded_dag())
        .query(CausalQuery::Response(query))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(50))
        .unwrap();
    let response = result.response.as_ref().expect("response payload");
    let ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) =
        &response.estimate
    else {
        panic!("expected a point-identified response surface");
    };
    mean.to_vec()
}

fn ate(data: TabularData) -> f64 {
    Study::tabular(data)
        .graph(confounded_dag())
        .query(AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(50))
        .unwrap()
        .estimate
        .ate
}

fn assert_close(a: f64, b: f64, what: &str) {
    assert!((a - b).abs() <= 1e-9 * (1.0 + b.abs()), "{what}: with null {a} vs dropped {b}");
}

#[test]
fn arrow_null_cell_matches_dropped_row_for_response_curve() {
    for null_col in 0..3 {
        let (with_null, dropped) = with_null_and_dropped(false, null_col);
        let a = response_curve(with_null);
        let b = response_curve(dropped);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(&b) {
            assert_close(*x, *y, &format!("response curve, null in column {null_col}"));
        }
    }
}

#[test]
fn arrow_null_cell_matches_dropped_row_for_binary_ate() {
    for null_col in 0..3 {
        let (with_null, dropped) = with_null_and_dropped(true, null_col);
        assert_close(ate(with_null), ate(dropped), &format!("ATE, null in column {null_col}"));
    }
}
