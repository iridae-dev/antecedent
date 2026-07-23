//! Surrogate null transforms for discovery false-positive checks.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::f64::consts::PI;
use std::sync::Arc;

use antecedent_core::CausalRng;
use antecedent_kernels::shuffle;
use realfft::RealFftPlanner;
use realfft::num_complex::Complex;

use crate::column::{ColumnView, Float64Column, OwnedColumn};
use crate::dataset::TimeSeriesData;
use crate::error::DataError;
use crate::storage::OwnedColumnarStorage;
use crate::table::TableView;

/// Independently permute each float64 column (destroy cross-dependence; keep marginals).
///
/// Validity bitmaps, analysis mask, weights, and time index are preserved in place.
///
/// # Errors
///
/// Non-float columns or reconstruction failures.
pub fn surrogate_permute_columns(
    data: &TimeSeriesData,
    rng: &mut CausalRng,
) -> Result<TimeSeriesData, DataError> {
    let schema = data.schema().clone();
    let mut cols = Vec::with_capacity(schema.len());
    for v in schema.variables() {
        let ColumnView::Float64(src) = data.column(v.id)? else {
            return Err(DataError::TypeMismatch { id: v.id, expected: "float64" });
        };
        let mut values = src.values.to_vec();
        shuffle(rng, &mut values);
        cols.push(OwnedColumn::Float64(Float64Column::new(
            v.id,
            Arc::from(values),
            src.validity.clone(),
        )?));
    }
    let analysis_mask = data.storage().analysis_mask().cloned();
    let weights = data.storage().weights().map(|w| Arc::from(w.to_vec()));
    let storage = OwnedColumnarStorage::try_new(schema, cols, analysis_mask, weights)?;
    TimeSeriesData::try_new(storage, data.time_index().clone())
}

/// Phase-randomize each float64 column (preserve power spectrum / autocorr).
///
/// DC and Nyquist (even `n`) bins stay real; other bins keep magnitude and get
/// independent uniform phases. Cross-series dependence is destroyed.
///
/// # Errors
///
/// Series shorter than 4, non-float columns, or FFT failures.
pub fn surrogate_phase_randomize(
    data: &TimeSeriesData,
    rng: &mut CausalRng,
) -> Result<TimeSeriesData, DataError> {
    let n = data.row_count();
    if n < 4 {
        return Err(DataError::InvalidArgument {
            message: "phase-randomize requires series length ≥ 4".into(),
        });
    }
    let schema = data.schema().clone();
    let mut planner = RealFftPlanner::<f64>::new();
    let r2c = planner.plan_fft_forward(n);
    let c2r = planner.plan_fft_inverse(n);
    let mut cols = Vec::with_capacity(schema.len());
    for v in schema.variables() {
        let ColumnView::Float64(src) = data.column(v.id)? else {
            return Err(DataError::TypeMismatch { id: v.id, expected: "float64" });
        };
        let values = phase_randomize_column(&src.values, n, &r2c, &c2r, rng)?;
        cols.push(OwnedColumn::Float64(Float64Column::new(
            v.id,
            Arc::from(values),
            src.validity.clone(),
        )?));
    }
    let analysis_mask = data.storage().analysis_mask().cloned();
    let weights = data.storage().weights().map(|w| Arc::from(w.to_vec()));
    let storage = OwnedColumnarStorage::try_new(schema, cols, analysis_mask, weights)?;
    TimeSeriesData::try_new(storage, data.time_index().clone())
}

fn phase_randomize_column(
    values: &[f64],
    n: usize,
    r2c: &std::sync::Arc<dyn realfft::RealToComplex<f64>>,
    c2r: &std::sync::Arc<dyn realfft::ComplexToReal<f64>>,
    rng: &mut CausalRng,
) -> Result<Vec<f64>, DataError> {
    let mut indata = values.to_vec();
    let mut spectrum = r2c.make_output_vec();
    r2c.process(&mut indata, &mut spectrum)
        .map_err(|e| DataError::InvalidArgument { message: format!("forward FFT failed: {e}") })?;

    // spectrum[0] = DC (real). spectrum[last] = Nyquist when n even (must stay real).
    let last = spectrum.len() - 1;
    let nyquist_real = n % 2 == 0;
    for (k, bin) in spectrum.iter_mut().enumerate().skip(1) {
        if nyquist_real && k == last {
            *bin = Complex::new(bin.re, 0.0);
            continue;
        }
        let mag = bin.norm();
        let phase = rng.next_f64() * 2.0 * PI;
        *bin = Complex::from_polar(mag, phase);
    }

    let mut out = c2r.make_output_vec();
    c2r.process(&mut spectrum, &mut out)
        .map_err(|e| DataError::InvalidArgument { message: format!("inverse FFT failed: {e}") })?;
    let scale = 1.0 / n as f64;
    for v in &mut out {
        *v *= scale;
    }
    Ok(out)
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use antecedent_core::{
        CausalRng, CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
        VariableId,
    };

    use super::*;
    use crate::column::ValidityBitmap;
    use crate::temporal::{SamplingRegularity, TimeIndex};

    fn two_col_ar1() -> TimeSeriesData {
        let n = 128usize;
        let mut x = vec![0.0; n];
        let mut y = vec![0.0; n];
        for t in 1..n {
            x[t] = 0.7 * x[t - 1] + (t as f64).sin() * 0.1;
            y[t] = 0.5 * x[t - 1] + 0.6 * y[t - 1] + (t as f64).cos() * 0.05;
        }
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "x",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(x),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(y),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        TimeSeriesData::try_new(
            storage,
            TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
        )
        .unwrap()
    }

    #[test]
    fn column_permute_preserves_marginal_sorted() {
        let data = two_col_ar1();
        let mut rng = CausalRng::from_seed(7);
        let out = surrogate_permute_columns(&data, &mut rng).unwrap();
        for id in [VariableId::from_raw(0), VariableId::from_raw(1)] {
            let ColumnView::Float64(a) = data.column(id).unwrap() else {
                panic!("float");
            };
            let ColumnView::Float64(b) = out.column(id).unwrap() else {
                panic!("float");
            };
            let mut sa = a.values.to_vec();
            let mut sb = b.values.to_vec();
            sa.sort_by(|x, y| x.partial_cmp(y).unwrap());
            sb.sort_by(|x, y| x.partial_cmp(y).unwrap());
            assert_eq!(sa, sb);
        }
    }

    #[test]
    fn phase_randomize_preserves_power_roughly() {
        let data = two_col_ar1();
        let mut rng = CausalRng::from_seed(11);
        let out = surrogate_phase_randomize(&data, &mut rng).unwrap();
        let ColumnView::Float64(a) = data.column(VariableId::from_raw(0)).unwrap() else {
            panic!("float");
        };
        let ColumnView::Float64(b) = out.column(VariableId::from_raw(0)).unwrap() else {
            panic!("float");
        };
        let var_a: f64 = {
            let m = a.values.iter().sum::<f64>() / a.values.len() as f64;
            a.values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / a.values.len() as f64
        };
        let var_b: f64 = {
            let m = b.values.iter().sum::<f64>() / b.values.len() as f64;
            b.values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / b.values.len() as f64
        };
        assert!((var_a - var_b).abs() / var_a.max(1e-12) < 0.05, "var {var_a} vs {var_b}");
    }

    #[test]
    fn phase_randomize_rejects_short() {
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "x",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let col = Float64Column::new(
            VariableId::from_raw(0),
            Arc::from([1.0, 2.0, 3.0]),
            ValidityBitmap::all_valid(3),
        )
        .unwrap();
        let storage =
            OwnedColumnarStorage::try_new(schema, vec![OwnedColumn::Float64(col)], None, None)
                .unwrap();
        let data = TimeSeriesData::try_new(
            storage,
            TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: 3 },
        )
        .unwrap();
        let mut rng = CausalRng::from_seed(1);
        assert!(surrogate_phase_randomize(&data, &mut rng).is_err());
    }
}
