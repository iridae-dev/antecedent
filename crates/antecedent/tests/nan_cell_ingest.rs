//! A `NaN` in a raw `f64` column is a missing cell, never an observation.
//!
//! Data built from plain slices (`from_f64_columns`) has no separate validity
//! input, so a `NaN` there must be marked invalid: the analysis must match the
//! one where that cell is missing through an explicit validity bitmap (and, for
//! tabular data, the one where the row was dropped before loading). Before,
//! such a cell stayed valid: a `NaN` treatment or confounder failed the ATE
//! with a rank-deficiency error and a `NaN` outcome returned a `NaN` effect.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ExecutionContext, Lag, TemporalEffectQuery, TemporalPolicy,
    VariableId,
};
use antecedent_data::storage::OwnedColumnarStorage;
use antecedent_data::{
    Float64Column, OwnedColumn, SamplingRegularity, TableView, TabularData, TimeIndex,
    TimeSeriesData, ValidityBitmap,
};
use antecedent_graph::{Dag, DenseNodeId, TemporalDag, ensure_lagged};

const N: usize = 240;
const NAN_ROW: usize = 37;
const NAMES: [&str; 3] = ["treatment", "outcome", "confounder"];

/// `(treatment, outcome, confounder)` columns with `Y = 1 + 2T + 0.8Z + e`.
fn tabular_columns() -> [Vec<f64>; 3] {
    let z: Vec<f64> = (0..N).map(|i| (i as f64 / 17.0).sin()).collect();
    let t: Vec<f64> = (0..N)
        .map(|i| {
            let latent = z[i] + (i as f64 / 11.0).cos() + (i % 7) as f64 * 0.03;
            f64::from(u8::from(latent > 0.3))
        })
        .collect();
    let y: Vec<f64> =
        (0..N).map(|i| 1.0 + 2.0 * t[i] + 0.8 * z[i] + (i as f64 / 13.0).sin() * 0.5).collect();
    [t, y, z]
}

/// Temporal columns `(x, y, z)` with `y_t = 0.7 x_{t-1} + 0.5 z_t + e_t`.
fn series_columns() -> [Vec<f64>; 3] {
    let z: Vec<f64> = (0..N).map(|i| (i as f64 / 9.0).sin()).collect();
    let x: Vec<f64> =
        (0..N).map(|i| (i as f64 * 0.21).cos() + 0.4 * z[i] + (i % 5) as f64 * 0.05).collect();
    let y: Vec<f64> = (0..N)
        .map(|i| {
            let lagged = if i == 0 { 0.0 } else { 0.7 * x[i - 1] };
            lagged + 0.5 * z[i] + (i as f64 / 7.0).sin() * 0.3
        })
        .collect();
    [x, y, z]
}

fn with_nan(cols: &[Vec<f64>; 3], col: usize) -> Vec<Vec<f64>> {
    let mut out = cols.to_vec();
    out[col][NAN_ROW] = f64::NAN;
    out
}

fn from_slices(cols: &[Vec<f64>]) -> TabularData {
    TabularData::from_f64_columns(NAMES.iter().zip(cols).map(|(n, c)| (*n, c.as_slice()))).unwrap()
}

/// The same table with `(col, NAN_ROW)` missing through an explicit bitmap.
fn with_invalid_bit(cols: &[Vec<f64>; 3], col: usize) -> OwnedColumnarStorage {
    let base = from_slices(cols);
    let mut bytes = vec![0xFFu8; N.div_ceil(8)];
    bytes[NAN_ROW / 8] &= !(1 << (NAN_ROW % 8));
    let columns: Vec<OwnedColumn> = cols
        .iter()
        .enumerate()
        .map(|(k, c)| {
            let validity = if k == col {
                ValidityBitmap::from_bytes(bytes.clone(), N).unwrap()
            } else {
                ValidityBitmap::all_valid(N)
            };
            let id = VariableId::from_raw(u32::try_from(k).unwrap());
            OwnedColumn::Float64(Float64Column::new(id, c.clone(), validity).unwrap())
        })
        .collect();
    OwnedColumnarStorage::try_new(base.schema().clone(), columns, None, None).unwrap()
}

fn confounded_dag() -> Dag {
    let mut graph = Dag::with_variables(3);
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    graph
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

fn pulse_dag() -> TemporalDag {
    let mut g = TemporalDag::empty();
    let x1 = ensure_lagged(&mut g, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut g, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let z0 = ensure_lagged(&mut g, VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    let z1 = ensure_lagged(&mut g, VariableId::from_raw(2), Lag::from_raw(1)).unwrap();
    g.insert_directed(x1, y0).unwrap();
    g.insert_directed(z0, y0).unwrap();
    g.insert_directed(z1, x1).unwrap();
    g.insert_directed(z1, y0).unwrap();
    g
}

fn pulse(series: TimeSeriesData) -> Result<f64, String> {
    let query = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1);
    Study::series(series)
        .graph(pulse_dag())
        .query(CausalQuery::TemporalEffect(query))
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(50))
        .map(|result| result.estimate.ate)
        .map_err(|e| format!("{e:?}"))
}

fn assert_close(a: f64, b: f64, what: &str) {
    assert!((a - b).abs() <= 1e-9 * (1.0 + b.abs()), "{what}: with NaN {a} vs reference {b}");
}

#[test]
fn from_f64_columns_marks_nan_cells_invalid() {
    let cols = with_nan(&tabular_columns(), 1);
    let data = from_slices(&cols);
    let outcome = data.column(VariableId::from_raw(1)).unwrap();
    assert!(!outcome.validity().is_valid(NAN_ROW));
    assert!(outcome.validity().is_valid(NAN_ROW + 1));
    let treatment = data.column(VariableId::from_raw(0)).unwrap();
    assert!(treatment.validity().is_all_valid());

    let series = TimeSeriesData::from_f64_columns(
        NAMES.iter().zip(&cols).map(|(n, c)| (*n, c.as_slice())),
        1,
    )
    .unwrap();
    assert!(!series.column(VariableId::from_raw(1)).unwrap().validity().is_valid(NAN_ROW));
}

#[test]
fn nan_cell_matches_dropped_row_for_binary_ate() {
    let cols = tabular_columns();
    let dropped: Vec<Vec<f64>> = cols
        .iter()
        .map(|c| c.iter().enumerate().filter(|(i, _)| *i != NAN_ROW).map(|(_, &v)| v).collect())
        .collect();
    let reference = ate(from_slices(&dropped));
    for col in 0..3 {
        let with_nan = ate(from_slices(&with_nan(&cols, col)));
        assert_close(with_nan, reference, &format!("ATE, NaN in column {col}"));
        let with_bit = ate(TabularData::new(with_invalid_bit(&cols, col)));
        assert_close(with_nan, with_bit, &format!("ATE, NaN vs invalid bit in column {col}"));
    }
}

/// Temporal effect paths require complete series: a `NaN` cell must fail
/// closed with the same typed error as a cell missing through the bitmap,
/// never a rank-deficiency error or a silent `NaN` effect.
#[test]
fn nan_cell_matches_invalid_bit_for_temporal_pulse() {
    let cols = series_columns();
    let series = |c: &[Vec<f64>]| {
        TimeSeriesData::from_f64_columns(NAMES.iter().zip(c).map(|(n, c)| (*n, c.as_slice())), 1)
            .unwrap()
    };
    let complete = pulse(series(&cols)).unwrap();
    assert!(complete.is_finite());
    for col in 0..3 {
        let with_nan = pulse(series(&with_nan(&cols, col)));
        let index =
            TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: N };
        let with_bit = pulse(TimeSeriesData::try_new(with_invalid_bit(&cols, col), index).unwrap());
        let err = with_nan.expect_err("a NaN cell must not yield an effect");
        assert!(
            err.contains("IncompleteSeries") && err.contains(&format!("VariableId({col})")),
            "Pulse, NaN in column {col}: {err}"
        );
        assert_eq!(Err(err), with_bit, "Pulse, NaN vs invalid bit in column {col}");
    }
}
