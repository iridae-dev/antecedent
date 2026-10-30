//! Surrogate null transforms for discovery false-positive checks.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

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

fn require_uniform_weights(data: &TimeSeriesData) -> Result<(), DataError> {
    if let Some(weights) = data.storage().weights() {
        if let Some(&first) = weights.first() {
            if !first.is_finite() || weights.iter().any(|&w| !w.is_finite() || w != first) {
                return Err(DataError::invalid_value(
                    "surrogate observation weights",
                    "nonuniform weights do not support exchangeable surrogates",
                ));
            }
        }
    }
    Ok(())
}

/// Independently permute each float64 column (destroy cross-dependence; keep marginals).
///
/// Validity bitmaps, analysis mask, weights, and time index are preserved in place. Valid
/// values are permuted separately within included and excluded rows. Nonuniform weights are
/// refused because a row exchange would change the weighted empirical marginal.
///
/// # Errors
///
/// Non-float columns or reconstruction failures.
pub fn surrogate_permute_columns(
    data: &TimeSeriesData,
    rng: &mut CausalRng,
) -> Result<TimeSeriesData, DataError> {
    require_uniform_weights(data)?;
    let mask = data.storage().analysis_mask();
    let schema = data.schema().clone();
    let mut cols = Vec::with_capacity(schema.len());
    for v in schema.variables() {
        let ColumnView::Float64(src) = data.column(v.id)? else {
            return Err(DataError::TypeMismatch { id: v.id, expected: "float64" });
        };
        let mut values = src.values.to_vec();
        if src.validity.is_all_valid() && mask.is_none_or(|m| m.is_all_valid()) {
            shuffle(rng, &mut values);
        } else {
            for included in [true, false] {
                let eligible = |index: usize| {
                    src.validity.is_valid(index)
                        && mask.is_none_or(|m| m.is_valid(index)) == included
                };
                let mut observed: Vec<f64> = values
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| eligible(*index))
                    .map(|(_, &value)| value)
                    .collect();
                shuffle(rng, &mut observed);
                for ((_, value), shuffled) in values
                    .iter_mut()
                    .enumerate()
                    .filter(|(index, _)| eligible(*index))
                    .zip(observed)
                {
                    *value = shuffled;
                }
            }
        }
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
/// independent uniform phases. Cross-series dependence is destroyed. Nontrivial analysis
/// masks and nonuniform weights are refused because the FFT mixes all time points.
///
/// # Errors
///
/// Series shorter than 4, non-float columns, missing or nonfinite observations, or FFT failures.
pub fn surrogate_phase_randomize(
    data: &TimeSeriesData,
    rng: &mut CausalRng,
) -> Result<TimeSeriesData, DataError> {
    require_uniform_weights(data)?;
    if data.storage().analysis_mask().is_some_and(|mask| !mask.is_all_valid()) {
        return Err(DataError::invalid_value(
            "phase-randomize analysis mask",
            "excluded rows cannot be mixed into the analyzed series",
        ));
    }
    let n = data.row_count();
    if n < 4 {
        return Err(DataError::invalid_value(
            "phase-randomize series length",
            "must be at least 4",
        ));
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
        if !src.validity.is_all_valid() || !src.values.iter().all(|value| value.is_finite()) {
            return Err(DataError::invalid_value(
                "phase-randomize observations",
                "must be complete and finite",
            ));
        }
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
    r2c.process(&mut indata, &mut spectrum).map_err(|e| DataError::external("forward FFT", e))?;

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
    c2r.process(&mut spectrum, &mut out).map_err(|e| DataError::external("inverse FFT", e))?;
    let scale = 1.0 / n as f64;
    for v in &mut out {
        *v *= scale;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use antecedent_core::{
        CausalRng, CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
        VariableId,
    };

    use super::*;
    use crate::column::ValidityBitmap;
    use crate::temporal::{SamplingRegularity, TimeIndex};

    fn single_column(values: &[f64]) -> TimeSeriesData {
        let mut schema = CausalSchemaBuilder::new();
        schema
            .add_variable(
                "x",
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        let column = Float64Column::new(
            VariableId::from_raw(0),
            Arc::from(values),
            ValidityBitmap::all_valid(values.len()),
        )
        .unwrap();
        let storage = OwnedColumnarStorage::try_new(
            schema.build().unwrap(),
            vec![OwnedColumn::Float64(column)],
            None,
            None,
        )
        .unwrap();
        TimeSeriesData::try_new(
            storage,
            TimeIndex {
                regularity: SamplingRegularity::Regular { interval_ns: 1 },
                length: values.len(),
            },
        )
        .unwrap()
    }

    #[test]
    fn surrogate_masks_and_weighted_exchangeability_are_respected() {
        let base = single_column(&[1.0, 2.0, 3.0, 100.0, 200.0, 300.0]);
        let masked = base
            .with_analysis_mask(ValidityBitmap::from_bools(&[
                true, true, true, false, false, false,
            ]))
            .unwrap();
        let weighted = masked.with_weights(Arc::from([1.0; 6])).unwrap();
        for seed in 0..16 {
            let out =
                surrogate_permute_columns(&weighted, &mut CausalRng::from_seed(seed)).unwrap();
            let ColumnView::Float64(column) = out.column(VariableId::from_raw(0)).unwrap() else {
                panic!("float")
            };
            let mut included = column.values[..3].to_vec();
            let mut excluded = column.values[3..].to_vec();
            included.sort_by(f64::total_cmp);
            excluded.sort_by(f64::total_cmp);
            assert_eq!(included, [1.0, 2.0, 3.0]);
            assert_eq!(excluded, [100.0, 200.0, 300.0]);
        }
        assert!(surrogate_phase_randomize(&weighted, &mut CausalRng::from_seed(1)).is_err());
        let unequal = base.with_weights(Arc::from([1.0, 1.0, 1.0, 2.0, 1.0, 1.0])).unwrap();
        assert!(surrogate_permute_columns(&unequal, &mut CausalRng::from_seed(1)).is_err());
        assert!(surrogate_phase_randomize(&unequal, &mut CausalRng::from_seed(1)).is_err());
        let all = base
            .with_analysis_mask(ValidityBitmap::all_valid(6))
            .unwrap()
            .with_weights(Arc::from([1.0; 6]))
            .unwrap();
        assert!(surrogate_permute_columns(&all, &mut CausalRng::from_seed(1)).is_ok());
        assert!(surrogate_phase_randomize(&all, &mut CausalRng::from_seed(1)).is_ok());
    }

    #[test]
    fn permutation_preserves_observed_marginal_and_missing_positions() {
        let data = single_column(&[1.0, f64::NAN, 2.0, 3.0]);
        for seed in 0..16 {
            let out = surrogate_permute_columns(&data, &mut CausalRng::from_seed(seed)).unwrap();
            let ColumnView::Float64(column) = out.column(VariableId::from_raw(0)).unwrap() else {
                panic!("float");
            };
            assert!(!column.validity.is_valid(1));
            let mut observed: Vec<f64> = column
                .values
                .iter()
                .enumerate()
                .filter(|(index, _)| column.validity.is_valid(*index))
                .map(|(_, &value)| value)
                .collect();
            observed.sort_by(f64::total_cmp);
            assert_eq!(observed, vec![1.0, 2.0, 3.0]);
        }
    }

    #[test]
    fn phase_randomization_refuses_missing_or_infinite_observations() {
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let data = single_column(&[1.0, invalid, 2.0, 3.0]);
            assert!(surrogate_phase_randomize(&data, &mut CausalRng::from_seed(7)).is_err());
        }
    }

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
            sa.sort_by(f64::total_cmp);
            sb.sort_by(f64::total_cmp);
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
