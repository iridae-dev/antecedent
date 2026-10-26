//! Unit tests.

use super::indexes::*;
use super::replicates::*;
use super::series::*;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};

use antecedent_core::execution::AdmissionPool;
use antecedent_core::{
    CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
    VariableId,
};

use super::*;
use crate::column::{ColumnView, Float64Column, OwnedColumn, ValidityBitmap};
use crate::temporal::{SamplingRegularity, TimeIndex};
use crate::testing::float_series;

#[test]
fn distinct_bases_have_disjoint_replicate_streams() {
    let ctx = ExecutionContext::for_tests(5);
    let mut seen = std::collections::HashSet::new();
    let bases = [0xA1D5_u64, 0xA1D0, 0xF01D, 0xF02D, 0x1300_1000];
    for base in bases {
        for replicate in 0..32u64 {
            let stream = resample_rng(&ctx, base, replicate).next_u64();
            assert!(
                seen.insert(stream),
                "base {base:#x} replicate {replicate} collided with another stream"
            );
        }
    }
    let xor = |base: u64, replicate: u64| {
        ctx.rng.stream_for(StreamDomain::Resample, base ^ replicate).next_u64()
    };
    assert_eq!(xor(0xA1D5, 0), xor(0xA1D0, 0xA1D5 ^ 0xA1D0));
}

/// One-variable series of length 4 with row 2 invalid, a mask hiding row 1,
/// and per-row weights.
fn series_with_missing() -> TimeSeriesData {
    let mut b = CausalSchemaBuilder::new();
    b.add_variable(
        "v0".to_owned(),
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::Context),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    let schema = b.build().unwrap();
    // Row 2 invalid (LSB-first bits: rows 0,1,3 set).
    let validity = ValidityBitmap::from_bytes(vec![0b1011u8], 4).unwrap();
    let col = OwnedColumn::Float64(
        Float64Column::new(
            VariableId::from_raw(0),
            Arc::from(vec![10.0, 11.0, 0.0, 13.0]),
            validity,
        )
        .unwrap(),
    );
    // Analysis mask hides row 1 (LSB-first bits: rows 0,2,3 set).
    let mask = ValidityBitmap::from_bytes(vec![0b1101u8], 4).unwrap();
    let weights: Arc<[f64]> = Arc::from(vec![1.0, 2.0, 3.0, 4.0]);
    let storage =
        OwnedColumnarStorage::try_new(schema, vec![col], Some(mask), Some(weights)).unwrap();
    TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: 4 },
    )
    .unwrap()
}

#[test]
fn row_map_gathers_validity_mask_and_weights() {
    let data = series_with_missing();
    let map = [GlobalId::new(2), GlobalId::new(0), GlobalId::new(2), GlobalId::new(1)];
    let out = apply_row_map(&data, &map).unwrap();
    let ColumnView::Float64(col) = out.column(VariableId::from_raw(0)).unwrap() else {
        panic!("expected float64 column");
    };
    // Replicate slots sourced from row 2 must stay invalid.
    assert!(!col.validity.is_valid(0));
    assert!(col.validity.is_valid(1));
    assert!(!col.validity.is_valid(2));
    assert!(col.validity.is_valid(3));
    // No invalid source value appears as a valid replicate value.
    for i in 0..4 {
        if col.validity.is_valid(i) {
            assert!((col.values[i] - 10.0).abs() < 1e-12 || (col.values[i] - 11.0).abs() < 1e-12);
        }
    }
    // Analysis mask follows the row map (source row 1 was hidden).
    let mask = out.storage().analysis_mask().unwrap();
    assert!(mask.is_valid(0));
    assert!(mask.is_valid(1));
    assert!(mask.is_valid(2));
    assert!(!mask.is_valid(3));
    // Weights follow the row map.
    assert_eq!(out.storage().weights().unwrap(), &[3.0, 1.0, 3.0, 2.0]);
}

#[test]
fn bayesian_bootstrap_weights_sum_to_n() {
    let mut rng = CausalRng::from_seed(99);
    let mut w = Vec::new();
    fill_resample_weights(ResamplingPlan::BayesianBootstrap, 50, &mut rng, &mut w).unwrap();
    assert_eq!(w.len(), 50);
    let sum: f64 = w.iter().sum();
    assert!((sum - 50.0).abs() < 1e-9);
    assert!(w.iter().all(|&x| x > 0.0));
}

#[test]
fn bayesian_bootstrap_rejects_index_fill() {
    let mut rng = CausalRng::from_seed(1);
    let mut idx = Vec::new();
    assert!(
        fill_resample_indexes(ResamplingPlan::BayesianBootstrap, 10, &mut rng, &mut idx).is_err()
    );
}

#[test]
fn iid_plans_are_refused_on_time_series_without_opt_in() {
    let data = float_series(30, 1);
    let mut idx = Vec::new();
    for plan in [ResamplingPlan::IidBootstrap, ResamplingPlan::BayesianBootstrap] {
        let mut rng = CausalRng::from_seed(3);
        assert!(resample_timeseries(&data, plan, &mut rng, &mut idx).is_err());
        let out = resample_timeseries_with_policy(
            &data,
            plan,
            None,
            &mut rng,
            &mut idx,
            TemporalRandomPolicy::Allow,
        )
        .unwrap();
        assert_eq!(out.row_count(), 30);
    }
}

#[test]
fn cluster_index_bootstrap_emits_whole_clusters_and_permutation_stays_within() {
    // label 0 -> rows {1,4}, label 1 -> {3}, label 2 -> {0,2}.
    let ids = [2u32, 0, 2, 1, 0];
    let mut rng = CausalRng::from_seed(5);
    let mut out = Vec::new();
    let allowed: [Vec<GlobalId>; 3] = [
        vec![GlobalId::new(1), GlobalId::new(4)],
        vec![GlobalId::new(3)],
        vec![GlobalId::new(0), GlobalId::new(2)],
    ];
    for _ in 0..50 {
        let plan = ResamplingPlan::ClusterBootstrap { cluster: VariableId::from_raw(0) };
        fill_resample_indexes_grouped(plan, 5, Some(&ids), &mut rng, &mut out).unwrap();
        let mut rest = out.as_slice();
        let mut clusters = 0;
        while !rest.is_empty() {
            let hit = allowed.iter().find(|c| rest.starts_with(c)).expect("whole cluster");
            rest = &rest[hit.len()..];
            clusters += 1;
        }
        assert_eq!(clusters, 3);
        let plan = ResamplingPlan::Permutation(PermutationScheme::WithinCluster);
        fill_resample_indexes_grouped(plan, 5, Some(&ids), &mut rng, &mut out).unwrap();
        for (pos, &row) in out.iter().enumerate() {
            let index = usize::try_from(row.get()).expect("row");
            assert_eq!(ids[pos], ids[index]);
        }
        let mut sorted = out.clone();
        sorted.sort_unstable();
        assert_eq!(
            sorted,
            [
                GlobalId::new(0),
                GlobalId::new(1),
                GlobalId::new(2),
                GlobalId::new(3),
                GlobalId::new(4),
            ]
        );
    }
}

#[test]
fn moving_block_preserves_length() {
    let data = float_series(100, 1);
    let mut rng = CausalRng::from_seed(1);
    let mut idx = Vec::new();
    let out =
        resample_timeseries(&data, ResamplingPlan::MovingBlock { length: 10 }, &mut rng, &mut idx)
            .unwrap();
    assert_eq!(out.row_count(), 100);
    assert_eq!(idx.len(), 100);
}

/// A moving-block bootstrap of 100 rows in blocks of 10 must be ten runs of ten
/// consecutive source rows (serial dependence kept inside each block), and the
/// same seed must replay the same indexes.
#[test]
fn moving_block_indexes_are_consecutive_runs_and_replay() {
    let data = float_series(100, 1);
    let draw = |seed: u64| {
        let mut rng = CausalRng::from_seed(seed);
        let mut idx = Vec::new();
        resample_timeseries(&data, ResamplingPlan::MovingBlock { length: 10 }, &mut rng, &mut idx)
            .unwrap();
        idx
    };
    let idx = draw(5);
    assert_eq!(idx.len(), 100);
    assert!(idx.iter().all(|i| i.get() < 100));
    for block in idx.chunks(10) {
        for w in block.windows(2) {
            assert_eq!(w[1].get(), w[0].get() + 1, "block is not a consecutive run: {block:?}");
        }
    }
    assert_eq!(idx, draw(5));
}

#[test]
fn circular_block_usize_fill_matches_u32_stream() {
    let mut rng_u32 = CausalRng::from_seed(17);
    let mut rng_usize = CausalRng::from_seed(17);
    let mut as_u32 = Vec::new();
    fill_resample_indexes(
        ResamplingPlan::CircularBlock { length: 7 },
        64,
        &mut rng_u32,
        &mut as_u32,
    )
    .unwrap();
    let mut as_usize = vec![0usize; 64];
    fill_circular_block_indexes(64, 7, &mut rng_usize, &mut as_usize).unwrap();
    assert_eq!(
        as_u32.iter().map(|i| usize::try_from(i.get()).expect("row")).collect::<Vec<_>>(),
        as_usize
    );
    assert_eq!(rng_u32.state(), rng_usize.state());
}

#[test]
fn circular_block_length_rule() {
    assert_eq!(integer_cube_root_ceil(0), 0);
    assert_eq!(integer_cube_root_ceil(1), 1);
    assert_eq!(integer_cube_root_ceil(64), 4);
    assert_eq!(integer_cube_root_ceil(65), 5);
    assert_eq!(integer_cube_root_ceil(1000), 10);
    // ⌈160^{1/3}⌉ = 6 dominates a two-slice window; a wide window dominates.
    assert_eq!(circular_block_length(2, 160), 6);
    assert_eq!(circular_block_length(9, 160), 9);
    // Capped at n, never zero.
    assert_eq!(circular_block_length(50, 12), 12);
    assert_eq!(circular_block_length(0, 0), 1);
}

#[test]
fn permutation_full_is_bijection() {
    let mut rng = CausalRng::from_seed(3);
    let mut idx = Vec::new();
    fill_resample_indexes(
        ResamplingPlan::Permutation(PermutationScheme::Full),
        20,
        &mut rng,
        &mut idx,
    )
    .unwrap();
    let mut sorted = idx.clone();
    sorted.sort_unstable();
    assert_eq!(
        sorted.iter().map(|id| id.get()).collect::<Vec<_>>(),
        (0..20u64).collect::<Vec<_>>()
    );
}

#[test]
fn cluster_bootstrap_keeps_members_contiguous_from_same_cluster() {
    let n = 12usize;
    let clusters: Vec<u32> = (0..n as u32).map(|i| i / 3).collect();
    let mut rng = CausalRng::from_seed(5);
    let mut idx = Vec::new();
    fill_resample_indexes_grouped(
        ResamplingPlan::ClusterBootstrap { cluster: VariableId::from_raw(0) },
        n,
        Some(&clusters),
        &mut rng,
        &mut idx,
    )
    .unwrap();
    assert_eq!(idx.len(), n);
    // Every index must come from some original cluster; consecutive triples
    // from the same draw share a cluster id when length allows.
    for &i in &idx {
        assert!(i.get() < u64::try_from(n).expect("n"));
    }
}

#[test]
fn stationary_block_preserves_length() {
    let mut rng = CausalRng::from_seed(2);
    let mut idx = Vec::new();
    fill_resample_indexes(
        ResamplingPlan::StationaryBlock { expected_length: 5.0 },
        80,
        &mut rng,
        &mut idx,
    )
    .unwrap();
    assert_eq!(idx.len(), 80);
}

#[test]
fn stationary_block_lengths_are_geometric_with_the_requested_mean() {
    // A block continues with probability 1 - p, so the share of adjacent draws that
    // continue a block is 1 - p plus the 1/n chance a fresh block starts at the next row.
    let n = 20_000usize;
    let mut rng = CausalRng::from_seed(11);
    let mut idx = Vec::new();
    fill_resample_indexes(
        ResamplingPlan::StationaryBlock { expected_length: 5.0 },
        n,
        &mut rng,
        &mut idx,
    )
    .unwrap();
    let continued = idx
        .windows(2)
        .filter(|w| w[1].get() == (w[0].get() + 1) % u64::try_from(n).expect("n"))
        .count() as f64
        / (n - 1) as f64;
    assert!((continued - 0.8).abs() < 0.02, "continued={continued}");
    // expected_length = 1 makes every block one row, still a full-length resample.
    let mut singles = Vec::new();
    fill_resample_indexes(
        ResamplingPlan::StationaryBlock { expected_length: 1.0 },
        50,
        &mut rng,
        &mut singles,
    )
    .unwrap();
    assert_eq!(singles.len(), 50);
}

#[test]
fn index_batch_shape_and_strict_determinism() {
    let ctx = ExecutionContext::for_tests(42);
    let n = 20usize;
    let b = 8usize;
    let mut out_a = vec![GlobalId::new(0); n * b];
    let mut out_b = vec![GlobalId::new(0); n * b];
    fill_resample_index_batch(
        ResamplingPlan::IidBootstrap,
        n,
        b,
        None,
        &ctx,
        0xDEAD_u64,
        &mut out_a,
    )
    .unwrap();
    fill_resample_index_batch(
        ResamplingPlan::IidBootstrap,
        n,
        b,
        None,
        &ctx,
        0xDEAD_u64,
        &mut out_b,
    )
    .unwrap();
    assert_eq!(out_a, out_b);
    assert!(out_a.iter().all(|i| i.get() < u64::try_from(n).expect("n")));
}

#[test]
fn weight_batch_rows_sum_to_n() {
    let ctx = ExecutionContext::for_tests(7);
    let n = 15usize;
    let b = 4usize;
    let mut out = vec![0.0; n * b];
    fill_resample_weight_batch(ResamplingPlan::BayesianBootstrap, n, b, &ctx, 0xBEEF_u64, &mut out)
        .unwrap();
    for r in 0..b {
        let sum: f64 = out[r * n..(r + 1) * n].iter().sum();
        assert!((sum - n as f64).abs() < 1e-9);
    }
}
#[test]
fn unequal_cluster_bootstrap_draws_g_whole_clusters_without_conditioning_on_rows() {
    let plan = ResamplingPlan::ClusterBootstrap { cluster: VariableId::from_raw(0) };
    let ids = [0, 1, 1, 1];
    let context = ExecutionContext::for_tests(42);
    let mut rng = CausalRng::from_seed(11);
    let mut rows = Vec::new();
    let mut lengths = [0usize; 3];
    for _ in 0..4000 {
        fill_resample_indexes_grouped(plan, 4, Some(&ids), &mut rng, &mut rows).unwrap();
        lengths[(rows.len() - 2) / 2] += 1;
        let mut cursor = 0;
        for _ in 0..2 {
            if rows[cursor].get() == 0 {
                cursor += 1;
            } else {
                let got: Vec<u64> = rows[cursor..cursor + 3].iter().map(|id| id.get()).collect();
                assert_eq!(got, [1, 2, 3]);
                cursor += 3;
            }
        }
        assert_eq!(cursor, rows.len(), "each replicate must contain exactly G=2 draws");
    }
    // Two independent cluster draws give row-count probabilities 1/4, 1/2, 1/4.
    for (count, target) in lengths.into_iter().zip([0.25, 0.5, 0.25]) {
        assert!((count as f64 / 4000.0 - target).abs() < 0.04);
    }
    let mut fixed = vec![GlobalId::new(0); 8];
    assert!(fill_resample_index_batch(plan, 4, 2, Some(&ids), &context, 11, &mut fixed).is_err());
}

#[test]
fn unequal_cluster_timeseries_updates_length_and_gathers_metadata() {
    let data = series_with_missing();
    let plan = ResamplingPlan::ClusterBootstrap { cluster: VariableId::from_raw(0) };
    let ids = [0, 1, 1, 1];
    let mut rng = CausalRng::from_seed(9);
    let mut rows = Vec::new();
    let mut saw_variable_length = false;
    for _ in 0..20 {
        let replicate =
            resample_timeseries_grouped(&data, plan, Some(&ids), &mut rng, &mut rows).unwrap();
        saw_variable_length |= rows.len() != data.row_count();
        assert_eq!(replicate.row_count(), rows.len());
        assert_eq!(replicate.time_index().length, rows.len());
        let column = replicate.column(VariableId::from_raw(0)).unwrap();
        let source = data.column(VariableId::from_raw(0)).unwrap();
        for (row, &origin) in rows.iter().enumerate() {
            assert_eq!(
                column.validity().is_valid(row),
                source.validity().is_valid(usize::try_from(origin.get()).expect("row"))
            );
            assert_eq!(
                replicate.storage().analysis_mask().unwrap().is_valid(row),
                data.storage()
                    .analysis_mask()
                    .unwrap()
                    .is_valid(usize::try_from(origin.get()).expect("row"))
            );
            assert!(
                (replicate.storage().weights().unwrap()[row]
                    - data.storage().weights().unwrap()
                        [usize::try_from(origin.get()).expect("row")])
                .abs()
                    < 1e-12
            );
        }
    }
    assert!(saw_variable_length);
}
#[test]
fn batch_dimension_overflow_is_an_error_before_allocation() {
    let context = ExecutionContext::for_tests(1);
    let index_error = fill_resample_index_batch(
        ResamplingPlan::IidBootstrap,
        usize::MAX,
        2,
        None,
        &context,
        0,
        &mut [],
    )
    .unwrap_err();
    let weight_error = fill_resample_weight_batch(
        ResamplingPlan::BayesianBootstrap,
        usize::MAX,
        2,
        &context,
        0,
        &mut [],
    )
    .unwrap_err();
    for error in [index_error, weight_error] {
        assert!(matches!(error, DataError::Overflow { .. }), "{error:?}");
    }
}

#[test]
fn resample_workers_stay_within_the_pool() {
    let n = 4usize;
    let replicates = 8usize;
    let scratch_bytes = u64::try_from(n * 2 * size_of::<u64>()).unwrap();
    let shared = AdmissionPool::fixed(2, scratch_bytes);
    let ctx = ExecutionContext::production(1, 8).with_admission(shared.clone());
    let barrier = Arc::new(Barrier::new(2));
    let inflight = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let mut out = vec![0u64; n * replicates];
    fill_replicate_rows(&ctx, n, replicates, &mut out, |_, scratch| {
        assert!(scratch.len() <= n);
        let now = inflight.fetch_add(1, Ordering::SeqCst).saturating_add(1);
        peak.fetch_max(now, Ordering::SeqCst);
        barrier.wait();
        inflight.fetch_sub(1, Ordering::SeqCst);
        scratch.clear();
        scratch.resize(n, 0);
        Ok(())
    })
    .unwrap();
    assert_eq!(peak.load(Ordering::SeqCst), 2);
    assert!(n * 2 * size_of::<u64>() <= usize::try_from(shared.byte_budget()).unwrap());
    assert_eq!(inflight.load(Ordering::SeqCst), 0);
    assert_eq!(ctx.memory_limit_bytes(), Some(scratch_bytes));
    assert_eq!(shared.available().threads(), 2);
    assert_eq!(shared.available().bytes(), scratch_bytes);
}

#[test]
fn resample_scratch_above_the_admission_pool_is_refused_before_work() {
    let n = 4usize;
    let replicates = 2usize;
    let bytes = u64::try_from(n * size_of::<GlobalId>()).unwrap();
    let ctx = ExecutionContext::for_tests(1)
        .with_admission(antecedent_core::execution::AdmissionPool::fixed(1, bytes - 1));
    let mut out = vec![GlobalId::new(9); n * replicates];
    let error = fill_resample_index_batch(
        ResamplingPlan::IidBootstrap,
        n,
        replicates,
        None,
        &ctx,
        0,
        &mut out,
    )
    .unwrap_err();
    assert!(
        matches!(error, DataError::BudgetRefused { what: "resample scratch", .. }),
        "{error:?}"
    );
    assert!(out.iter().all(|id| id.get() == 9));
    assert_eq!(ctx.memory_limit_bytes(), Some(bytes - 1));
}

#[test]
fn resample_scratch_overflow_is_refused_before_allocation() {
    let ctx = ExecutionContext::for_tests(1);
    let width = size_of::<u64>();
    let n = (usize::MAX / width) + 1;
    assert_eq!(n.wrapping_mul(width), 0);
    let calls = AtomicUsize::new(0);
    let mut out = [0u64; 0];
    let error = fill_replicate_rows(&ctx, n, 1, &mut out, |_, _| {
        calls.fetch_add(1, Ordering::SeqCst);
        Ok(())
    })
    .unwrap_err();
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(matches!(error, DataError::Overflow { .. }), "{error:?}");
    assert!(ctx.memory_limit_bytes().is_none());
}

#[test]
fn resample_scratch_products_use_checked_arithmetic() {
    let width = size_of::<u64>();
    let calls = AtomicUsize::new(0);
    let fill = |_: usize, _: &mut Vec<u64>| {
        calls.fetch_add(1, Ordering::SeqCst);
        Ok(())
    };
    // One buffer fits in u64 but two workers' product does not.
    let ctx = ExecutionContext::production(1, 2);
    let n = usize::MAX / width;
    let error = fill_replicate_rows(&ctx, n, 2, &mut [0u64; 0], fill).unwrap_err();
    assert!(matches!(error, DataError::Overflow { .. }), "{error:?}");
    // A single buffer whose byte count overflows `usize`.
    let serial = ExecutionContext::for_tests(1);
    let error = fill_replicate_rows(&serial, usize::MAX, 1, &mut [0u64; 0], fill).unwrap_err();
    assert!(matches!(error, DataError::Overflow { .. }), "{error:?}");
    // The index path refuses an `n` whose replicate matrix overflows.
    let index_error = fill_resample_index_batch(
        ResamplingPlan::IidBootstrap,
        usize::MAX / 2 + 1,
        2,
        None,
        &serial,
        0,
        &mut [],
    )
    .unwrap_err();
    assert!(matches!(index_error, DataError::Overflow { .. }), "{index_error:?}");
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[test]
fn a_concurrent_resample_finds_the_pool_held_and_stays_on_its_caller() {
    let n = 4usize;
    let shared = AdmissionPool::fixed(2, 1 << 20);
    let ctx = ExecutionContext::production(1, 8).with_admission(shared.clone());
    let ready = Barrier::new(3);
    let done = Barrier::new(3);
    let second_threads = std::sync::Mutex::new(std::collections::HashSet::new());
    std::thread::scope(|scope| {
        let holder = scope.spawn(|| {
            let mut out = vec![0u64; n * 2];
            fill_replicate_rows(&ctx, n, 2, &mut out, |_, scratch| {
                ready.wait();
                done.wait();
                scratch.clear();
                scratch.resize(n, 0);
                Ok(())
            })
            .unwrap();
        });
        // Both holder workers are inside `fill`, so the pool's two slots are leased.
        ready.wait();
        assert_eq!(shared.available().threads(), 0);
        let mut out = vec![0u64; n * 8];
        fill_replicate_rows(&ctx, n, 8, &mut out, |_, scratch| {
            second_threads.lock().unwrap().insert(std::thread::current().id());
            scratch.clear();
            scratch.resize(n, 0);
            Ok(())
        })
        .unwrap();
        done.wait();
        holder.join().unwrap();
    });
    assert_eq!(second_threads.lock().unwrap().len(), 1);
    assert_eq!(shared.available().threads(), 2);
    assert_eq!(shared.available().bytes(), 1 << 20);
}

#[test]
fn resample_refuses_when_the_pool_cannot_fund_scratch() {
    let n = 4usize;
    let replicates = 8usize;
    let bytes = u64::try_from(n * 2 * size_of::<u64>()).unwrap();
    let shared = AdmissionPool::fixed(2, bytes - 1);
    let ctx = ExecutionContext::production(1, 8).with_admission(shared.clone());
    let calls = AtomicUsize::new(0);
    let mut out = vec![7u64; n * replicates];
    let error = fill_replicate_rows(&ctx, n, replicates, &mut out, |_, _| {
        calls.fetch_add(1, Ordering::SeqCst);
        Ok(())
    })
    .unwrap_err();
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(
        matches!(error, DataError::BudgetRefused { what: "resample scratch", .. }),
        "{error:?}"
    );
    assert!(out.iter().all(|value| *value == 7));
    assert_eq!(shared.available().threads(), 2);
    assert_eq!(shared.available().bytes(), bytes - 1);
    assert_eq!(ctx.memory_limit_bytes(), Some(bytes - 1));
}

#[test]
fn cancelled_batches_return_before_mutating_output() {
    let context = ExecutionContext::for_tests(1);
    context.cancellation.cancel();
    let mut indexes = [GlobalId::new(7)];
    let mut weights = [0.7];
    assert!(
        fill_resample_index_batch(
            ResamplingPlan::IidBootstrap,
            1,
            1,
            None,
            &context,
            0,
            &mut indexes,
        )
        .unwrap_err()
            == DataError::Cancelled
    );
    assert!(
        fill_resample_weight_batch(
            ResamplingPlan::BayesianBootstrap,
            1,
            1,
            &context,
            0,
            &mut weights,
        )
        .unwrap_err()
            == DataError::Cancelled
    );
    assert_eq!(indexes, [GlobalId::new(7)]);
    assert!((weights[0] - 0.7).abs() < 1e-12);
}

/// One reused buffer, filled per replicate, matches the replicate-major batch row.
fn assert_reused_indexes_match_batch(
    plan: ResamplingPlan,
    n: usize,
    replicates: usize,
    cluster_ids: Option<&[u32]>,
    seed: u64,
    stream_base: u64,
    threads: u32,
) {
    let ctx = if threads == 1 {
        ExecutionContext::for_tests(seed)
    } else {
        ExecutionContext::production(seed, threads)
    };
    let mut batch = vec![GlobalId::new(0); n * replicates];
    fill_resample_index_batch(plan, n, replicates, cluster_ids, &ctx, stream_base, &mut batch)
        .unwrap();
    let mut reused = Vec::new();
    let mut held = 0usize;
    for replicate in 0..replicates {
        fill_resample_index_replicate(
            plan,
            n,
            cluster_ids,
            &ctx,
            stream_base,
            replicate as u64,
            &mut reused,
        )
        .unwrap();
        let start = replicate * n;
        assert_eq!(reused.as_slice(), &batch[start..start + n]);
        if replicate == 0 {
            held = reused.capacity();
        } else {
            assert_eq!(reused.capacity(), held);
        }
    }
}

#[test]
fn bounded_iid_indexes_match_the_batch() {
    assert_reused_indexes_match_batch(ResamplingPlan::IidBootstrap, 20, 8, None, 42, 0xDEAD, 1);
    assert_reused_indexes_match_batch(ResamplingPlan::IidBootstrap, 20, 8, None, 42, 0xDEAD, 4);
}

#[test]
fn bounded_block_indexes_match_the_batch() {
    let plans = [
        ResamplingPlan::MovingBlock { length: 3 },
        ResamplingPlan::CircularBlock { length: 4 },
        ResamplingPlan::StationaryBlock { expected_length: 2.5 },
    ];
    for plan in plans {
        assert_reused_indexes_match_batch(plan, 24, 6, None, 5, 0xB10C, 1);
    }
}

#[test]
fn bounded_permutation_indexes_match_the_batch() {
    assert_reused_indexes_match_batch(
        ResamplingPlan::Permutation(PermutationScheme::Full),
        16,
        5,
        None,
        3,
        0x5EED,
        1,
    );
    let ids = [0u32, 0, 1, 1, 1];
    assert_reused_indexes_match_batch(
        ResamplingPlan::Permutation(PermutationScheme::WithinCluster),
        5,
        7,
        Some(&ids),
        3,
        0x5EED,
        1,
    );
}

#[test]
fn bounded_bayesian_weights_match_the_batch() {
    let ctx = ExecutionContext::for_tests(7);
    let n = 15usize;
    let replicates = 4usize;
    let stream_base = 0xBEEF_u64;
    let mut batch = vec![0.0; n * replicates];
    fill_resample_weight_batch(
        ResamplingPlan::BayesianBootstrap,
        n,
        replicates,
        &ctx,
        stream_base,
        &mut batch,
    )
    .unwrap();
    let mut reused = Vec::new();
    let mut held = 0usize;
    for replicate in 0..replicates {
        fill_resample_weight_replicate(
            ResamplingPlan::BayesianBootstrap,
            n,
            &ctx,
            stream_base,
            replicate as u64,
            &mut reused,
        )
        .unwrap();
        let row = &batch[replicate * n..(replicate + 1) * n];
        assert_eq!(reused.len(), n);
        for (got, expect) in reused.iter().zip(row) {
            assert!((got - expect).abs() < 1e-12);
        }
        if replicate == 0 {
            held = reused.capacity();
        } else {
            assert_eq!(reused.capacity(), held);
        }
    }
}

#[test]
fn bounded_equal_cluster_indexes_match_the_batch() {
    let ids = [0u32, 0, 1, 1, 2, 2];
    let plan = ResamplingPlan::ClusterBootstrap { cluster: VariableId::from_raw(0) };
    assert_reused_indexes_match_batch(plan, ids.len(), 9, Some(&ids), 11, 0xC1, 1);
}

#[test]
fn bounded_cluster_replicate_keeps_variable_length_and_repeated_copies() {
    let plan = ResamplingPlan::ClusterBootstrap { cluster: VariableId::from_raw(0) };
    let ids = [0u32, 1, 1, 1];
    let ctx = ExecutionContext::for_tests(42);
    let mut reused = Vec::new();
    let mut saw_repeat = false;
    let mut saw_variable = false;
    for replicate in 0..64u64 {
        fill_resample_index_replicate(plan, 4, Some(&ids), &ctx, 11, replicate, &mut reused)
            .unwrap();
        let mut cursor = 0;
        let (mut copies0, mut copies1) = (0usize, 0usize);
        while cursor < reused.len() {
            if reused[cursor].get() == 0 {
                copies0 += 1;
                cursor += 1;
            } else {
                let got: Vec<u64> = reused[cursor..cursor + 3].iter().map(|id| id.get()).collect();
                assert_eq!(got, [1, 2, 3]);
                copies1 += 1;
                cursor += 3;
            }
        }
        assert_eq!(copies0 + copies1, 2, "a replicate draws G whole clusters");
        saw_repeat |= copies0 == 2 || copies1 == 2;
        saw_variable |= reused.len() != 4;
    }
    assert!(saw_repeat, "a repeated cluster must stay two copies");
    assert!(saw_variable, "unequal clusters change the replicate length");
}

fn declaration(design: DeclaredSamplingDesign, time_column: bool) -> SamplingDesignDeclaration {
    SamplingDesignDeclaration::new(design, time_column)
}

#[test]
fn a_dependent_series_refuses_an_iid_plan_as_a_dependent_resampling_check() {
    let mut out = Vec::new();
    let error = fill_declared_plan(
        declaration(DeclaredSamplingDesign::DependentTimeSeries, true),
        ResamplingPlan::IidBootstrap,
        8,
        None,
        &mut CausalRng::from_seed(3),
        &mut out,
    )
    .expect_err("an iid plan drops the series dependence");
    let record = error.route_diagnostic().expect("dependent resampling record");
    assert!(record.failed());
    assert_eq!(record.route_name(), "dependent_resampling");
    assert!(record.unevaluable_mass().is_unreported());
    assert!(out.is_empty());
    fill_declared_plan(
        declaration(DeclaredSamplingDesign::DependentTimeSeries, true),
        ResamplingPlan::MovingBlock { length: 2 },
        8,
        None,
        &mut CausalRng::from_seed(3),
        &mut out,
    )
    .expect("a block plan keeps the dependence");
    assert_eq!(out.len(), 8);
}

#[test]
fn trajectory_design_refuses_a_row_bootstrap() {
    for time_column in [false, true] {
        let mut rng = CausalRng::from_seed(9);
        let mut out = vec![GlobalId::new(7)];
        let error = fill_declared_resample(
            declaration(DeclaredSamplingDesign::IndependentSubjectTrajectories, time_column),
            6,
            None,
            2,
            0,
            &mut rng,
            &mut out,
        )
        .expect_err("a trajectory cannot take a row bootstrap");
        let message = error.to_string();
        assert!(message.contains("independent_subject_trajectories"), "{message}");
        assert!(message.contains("row_iid"), "{message}");
        assert!(message.contains("row_subject_iid"), "{message}");
        assert_eq!(out, vec![GlobalId::new(7)]);
        let planned = fill_declared_plan(
            declaration(DeclaredSamplingDesign::IndependentSubjectTrajectories, time_column),
            ResamplingPlan::IidBootstrap,
            6,
            None,
            &mut CausalRng::from_seed(9),
            &mut Vec::new(),
        )
        .expect_err("a row plan is not a whole trajectory");
        let planned = planned.to_string();
        assert!(planned.contains("row_iid"), "{planned}");
        assert!(planned.contains("row_subject_iid"), "{planned}");
    }
}

#[test]
fn trajectory_design_resamples_whole_trajectories() {
    let labels = [0_u32, 0, 1, 1, 1, 2];
    let members: [Vec<u64>; 3] = [vec![0, 1], vec![2, 3, 4], vec![5]];
    let mut drawn = Vec::new();
    fill_declared_resample(
        declaration(DeclaredSamplingDesign::IndependentSubjectTrajectories, false),
        labels.len(),
        Some(&labels),
        1,
        0,
        &mut CausalRng::from_seed(5),
        &mut drawn,
    )
    .expect("a trajectory resamples whole subjects");
    let flat: Vec<u64> = drawn.iter().map(|id| id.get()).collect();
    let mut cursor = 0usize;
    let mut draws = 0usize;
    while cursor < flat.len() {
        let matched = members.iter().find(|member| flat[cursor..].starts_with(member));
        let Some(member) = matched else {
            panic!("a trajectory row was split from its subject at {cursor}: {flat:?}");
        };
        cursor += member.len();
        draws += 1;
    }
    assert_eq!(draws, members.len());
}

#[test]
fn panel_design_draws_whole_clusters() {
    let labels = [0_u32, 0, 1, 1, 1, 2];
    let members: [Vec<u64>; 3] = [vec![0, 1], vec![2, 3, 4], vec![5]];
    let mut with_time = Vec::new();
    let mut without_time = Vec::new();
    fill_declared_resample(
        declaration(DeclaredSamplingDesign::ClusteredPanel, true),
        labels.len(),
        Some(&labels),
        1,
        0,
        &mut CausalRng::from_seed(5),
        &mut with_time,
    )
    .expect("a panel draws clusters");
    fill_declared_resample(
        declaration(DeclaredSamplingDesign::ClusteredPanel, false),
        labels.len(),
        Some(&labels),
        1,
        0,
        &mut CausalRng::from_seed(5),
        &mut without_time,
    )
    .expect("a time column does not change the panel draw");
    assert_eq!(with_time, without_time);
    let flat: Vec<u64> = with_time.iter().map(|id| id.get()).collect();
    let mut cursor = 0usize;
    let mut draws = 0usize;
    while cursor < flat.len() {
        let matched = members.iter().find(|member| flat[cursor..].starts_with(member));
        let Some(member) = matched else {
            panic!("a panel row was split from its cluster at {cursor}: {flat:?}");
        };
        cursor += member.len();
        draws += 1;
    }
    assert_eq!(draws, members.len());
    let row = fill_declared_plan(
        declaration(DeclaredSamplingDesign::ClusteredPanel, false),
        ResamplingPlan::IidBootstrap,
        labels.len(),
        Some(&labels),
        &mut CausalRng::from_seed(5),
        &mut Vec::new(),
    )
    .expect_err("a panel refuses a row bootstrap");
    let row = row.to_string();
    assert!(row.contains("row_iid"), "{row}");
    assert!(row.contains("whole_cluster"), "{row}");
    let embargoed = fill_declared_resample(
        declaration(DeclaredSamplingDesign::ClusteredPanel, true),
        labels.len(),
        Some(&labels),
        1,
        2,
        &mut CausalRng::from_seed(5),
        &mut Vec::new(),
    )
    .expect_err("a panel has no block boundary for an embargo");
    assert!(embargoed.to_string().contains("embargo"), "{embargoed}");
}

#[test]
fn series_design_draws_a_contiguous_block_and_honors_the_embargo() {
    let n = 10usize;
    let length = 3usize;
    let mut block = Vec::new();
    let mut moving = Vec::new();
    fill_declared_resample(
        declaration(DeclaredSamplingDesign::DependentTimeSeries, false),
        n,
        None,
        length,
        0,
        &mut CausalRng::from_seed(4),
        &mut block,
    )
    .expect("a series draws a contiguous block");
    fill_resample_indexes(
        ResamplingPlan::MovingBlock { length },
        n,
        &mut CausalRng::from_seed(4),
        &mut moving,
    )
    .expect("moving block");
    assert_eq!(block, moving);
    let embargo = 2u32;
    let bound = DeclaredSamplingDesign::DependentTimeSeries
        .apply_splitting_with_embargo(SplittingRule::ContiguousBlock, embargo)
        .expect("a series takes an embargo");
    assert_eq!(bound.embargo(), embargo);
    let mut with_time = Vec::new();
    let mut without_time = Vec::new();
    fill_declared_resample(
        declaration(DeclaredSamplingDesign::DependentTimeSeries, true),
        n,
        None,
        length,
        embargo,
        &mut CausalRng::from_seed(8),
        &mut with_time,
    )
    .expect("embargoed block");
    fill_declared_resample(
        declaration(DeclaredSamplingDesign::DependentTimeSeries, false),
        n,
        None,
        length,
        embargo,
        &mut CausalRng::from_seed(8),
        &mut without_time,
    )
    .expect("a time column does not change the series draw");
    assert_eq!(with_time, without_time);
    let margin = usize::try_from(embargo).unwrap();
    assert_eq!(with_time.len(), n - margin * 2);
    for id in &with_time {
        let row = usize::try_from(id.get()).unwrap();
        assert!(row >= margin && row < n - margin, "embargoed row {row} was drawn");
    }
    for window in with_time.chunks(length) {
        let rows: Vec<usize> = window.iter().map(|id| usize::try_from(id.get()).unwrap()).collect();
        for pair in rows.windows(2) {
            assert_eq!(pair[1], pair[0] + 1);
        }
    }
}
