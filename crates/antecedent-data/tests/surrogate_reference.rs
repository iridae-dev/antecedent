//! Surrogate transforms against their defining properties.
//!
//! Phase randomization (Theiler et al. 1992) must keep every Fourier magnitude and the mean
//! exactly and randomize only the phases. The reference is a direct O(n^2) discrete Fourier
//! transform that shares no code with the FFT the shipped transform uses, evaluated at odd
//! and even lengths (the Nyquist bin exists only for even n), on a series with a large mean,
//! and on a constant series. Phases are checked for uniformity bin by bin. The column
//! permutation is checked against the full permutation law of a short series, including a
//! gap whose position must not move.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::sync::Arc;

use antecedent_core::{
    CausalRng, CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::column::{ColumnView, Float64Column, OwnedColumn, ValidityBitmap};
use antecedent_data::table::TableView;
use antecedent_data::{
    OwnedColumnarStorage, SamplingRegularity, TimeIndex, TimeSeriesData, surrogate_permute_columns,
    surrogate_phase_randomize,
};

fn series(columns: &[Vec<f64>], validity: Option<ValidityBitmap>) -> TimeSeriesData {
    let n = columns[0].len();
    let mut builder = CausalSchemaBuilder::new();
    for i in 0..columns.len() {
        builder
            .add_variable(
                format!("v{i}"),
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
    }
    let cols = columns
        .iter()
        .enumerate()
        .map(|(i, values)| {
            let valid = validity.clone().unwrap_or_else(|| ValidityBitmap::all_valid(n));
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(u32::try_from(i).unwrap()),
                    Arc::from(values.clone()),
                    valid,
                )
                .unwrap(),
            )
        })
        .collect();
    let storage = OwnedColumnarStorage::try_new(builder.build().unwrap(), cols, None, None).unwrap();
    TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap()
}

fn column(data: &TimeSeriesData, i: u32) -> Vec<f64> {
    let ColumnView::Float64(c) = data.column(VariableId::from_raw(i)).unwrap() else {
        panic!("float64 column");
    };
    c.values.to_vec()
}

/// Direct discrete Fourier transform bin `k` of a real series.
fn dft(x: &[f64], k: usize) -> (f64, f64) {
    let n = x.len();
    let (mut re, mut im) = (0.0, 0.0);
    for (t, v) in x.iter().enumerate() {
        // Reduce the angle exactly in integers before the trigonometry.
        let angle = 2.0 * PI * ((k * t) % n) as f64 / n as f64;
        re += v * angle.cos();
        im -= v * angle.sin();
    }
    (re, im)
}

fn lcg_series(n: usize, seed: u64, mean: f64) -> Vec<f64> {
    let mut state = seed;
    let mut x = vec![0.0; n];
    for t in 0..n {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        let e = ((state >> 11) as f64 / (1u64 << 53) as f64) - 0.5;
        x[t] = mean + if t == 0 { e } else { 0.8 * (x[t - 1] - mean) + e } + (t as f64 * 0.37).sin();
    }
    x
}

#[test]
fn phase_randomization_keeps_every_fourier_magnitude_and_the_mean() {
    for n in [4usize, 5, 6, 7, 8, 9, 16, 31, 64, 101] {
        let x = lcg_series(n, 17 + n as u64, 50.0);
        let data = series(&[x.clone()], None);
        let out = column(&surrogate_phase_randomize(&data, &mut CausalRng::from_seed(9)).unwrap(), 0);
        assert_eq!(out.len(), n);
        let scale: f64 = x.iter().map(|v| v.abs()).sum();
        for k in 0..=n / 2 {
            let (a, b) = dft(&x, k);
            let (c, d) = dft(&out, k);
            let (ma, mb) = (a.hypot(b), c.hypot(d));
            assert!((ma - mb).abs() <= 1e-10 * scale, "n={n} bin {k}: {ma} vs {mb}");
        }
        // The DC bin is the sum: the mean is unchanged to rounding.
        let (sum_in, sum_out): (f64, f64) = (x.iter().sum(), out.iter().sum());
        assert!((sum_in - sum_out).abs() <= 1e-10 * scale);
        // The Nyquist bin of an even length stays real with its original sign.
        if n % 2 == 0 {
            let (a, _) = dft(&x, n / 2);
            let (c, _) = dft(&out, n / 2);
            assert!((a - c).abs() <= 1e-10 * scale, "n={n}: Nyquist {a} vs {c}");
        }
        // Not the identity: some phase moved.
        assert!(x.iter().zip(&out).any(|(p, q)| (p - q).abs() > 1e-6), "n={n}: unchanged");
    }
}

#[test]
fn circular_autocovariance_is_preserved_because_the_power_spectrum_is() {
    let n = 40usize;
    let x = lcg_series(n, 3, 0.0);
    let out = column(
        &surrogate_phase_randomize(&series(&[x.clone()], None), &mut CausalRng::from_seed(2)).unwrap(),
        0,
    );
    let circular = |s: &[f64], lag: usize| -> f64 { (0..n).map(|t| s[t] * s[(t + lag) % n]).sum() };
    for lag in 0..n {
        assert!((circular(&x, lag) - circular(&out, lag)).abs() < 1e-8 * circular(&x, 0).abs());
    }
}

#[test]
fn a_constant_series_is_a_fixed_point_and_columns_are_randomized_independently() {
    let constant = vec![3.25; 12];
    let out = column(
        &surrogate_phase_randomize(&series(&[constant.clone()], None), &mut CausalRng::from_seed(1))
            .unwrap(),
        0,
    );
    assert!(out.iter().all(|v| (v - 3.25).abs() < 1e-12));

    // Two identical columns receive different phases (cross-dependence is destroyed).
    let x = lcg_series(32, 5, 0.0);
    let both = surrogate_phase_randomize(&series(&[x.clone(), x.clone()], None), &mut CausalRng::from_seed(4))
        .unwrap();
    let (a, b) = (column(&both, 0), column(&both, 1));
    assert!(a.iter().zip(&b).any(|(p, q)| (p - q).abs() > 1e-6));
}

#[test]
fn surrogate_phases_are_uniform_on_the_circle() {
    // For a series with distinct magnitudes the phase of bin k is atan2(im, re) of the
    // surrogate; each is uniform on (-pi, pi]. Chi-square over eight equal arcs, per bin.
    let n = 12usize;
    let x = lcg_series(n, 8, 0.0);
    let data = series(&[x], None);
    let mut rng = CausalRng::from_seed(77);
    let draws = 8000usize;
    let bins = [1usize, 2, 3, 5];
    let mut counts = vec![[0usize; 8]; bins.len()];
    let mut joint = BTreeMap::new();
    for _ in 0..draws {
        let out = column(&surrogate_phase_randomize(&data, &mut rng).unwrap(), 0);
        let mut cells = Vec::new();
        for (slot, &k) in bins.iter().enumerate() {
            let (re, im) = dft(&out, k);
            let arc = (((im.atan2(re) + PI) / (2.0 * PI) * 8.0) as usize).min(7);
            counts[slot][arc] += 1;
            cells.push(arc / 4);
        }
        *joint.entry(cells).or_insert(0usize) += 1;
    }
    let expected = draws as f64 / 8.0;
    for (slot, row) in counts.iter().enumerate() {
        let chi: f64 = row.iter().map(|&c| (c as f64 - expected).powi(2) / expected).sum();
        // 7 degrees of freedom; 7 + 7 sqrt(14) + 30 is far beyond the 1e-9 tail.
        assert!(chi < 63.0, "bin {}: chi-square {chi}", bins[slot]);
    }
    // Phases of different bins are independent: the joint half-circle cells are uniform.
    let cell = draws as f64 / 16.0;
    assert_eq!(joint.len(), 16);
    let chi: f64 = joint.values().map(|&c| (c as f64 - cell).powi(2) / cell).sum();
    assert!(chi < 15.0 + 7.0 * 30.0_f64.sqrt() + 30.0, "joint chi-square {chi}");
}

#[test]
fn column_permutation_is_uniform_and_keeps_gaps_where_they_were() {
    let values = vec![1.0, 2.0, f64::NAN, 3.0, 4.0];
    let mut valid = vec![0b1111_1011u8];
    valid.truncate(1);
    let validity = ValidityBitmap::from_bytes(valid, 5).unwrap();
    let data = series(&[values.clone()], Some(validity));
    let mut rng = CausalRng::from_seed(31);
    let mut counts: BTreeMap<Vec<u64>, usize> = BTreeMap::new();
    let draws = 24 * 500;
    for _ in 0..draws {
        let out = surrogate_permute_columns(&data, &mut rng).unwrap();
        let ColumnView::Float64(c) = out.column(VariableId::from_raw(0)).unwrap() else {
            panic!("float64");
        };
        assert!(!c.validity.is_valid(2), "the gap moved");
        let observed: Vec<u64> = [0usize, 1, 3, 4].iter().map(|&i| c.values[i].to_bits()).collect();
        *counts.entry(observed).or_insert(0) += 1;
    }
    // All 4! orderings of the observed values appear, each with probability 1/24.
    assert_eq!(counts.len(), 24);
    let expected = draws as f64 / 24.0;
    let chi: f64 = counts.values().map(|&c| (c as f64 - expected).powi(2) / expected).sum();
    assert!(chi < 23.0 + 7.0 * 46.0_f64.sqrt() + 30.0, "chi-square {chi}");
}
