use super::*;
use antecedent_core::{
    CausalSchemaBuilder, MeasurementSpec, PopulationSelector, RoleHint, SmallRoleSet, ValueType,
};
use antecedent_data::column::{Float64Column, ValidityBitmap};
use antecedent_data::{OwnedColumn, OwnedColumnarStorage};
use antecedent_graph::{Dag, DenseNodeId};

#[test]
fn detects_y_shift_not_necessarily_x() {
    let n = 80usize;
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
    let mut xv = Vec::new();
    let mut yv = Vec::new();
    for i in 0..n {
        let x = (i % 40) as f64 * 0.1;
        xv.push(x);
        yv.push(if i < 40 { 1.0 + 2.0 * x } else { 6.0 + 2.0 * x });
    }
    let validity = ValidityBitmap::all_valid(n);
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(xv), validity.clone()).unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(yv), validity).unwrap(),
        ),
    ];
    let data = TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
    let mut g = Dag::with_variables(2);
    g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let model = CompiledCausalModel::compile(g).unwrap();
    let q = MechanismChangeQuery::new(
        [VariableId::from_raw(0), VariableId::from_raw(1)],
        PopulationSelector::TimeRange { start: 0, end: 40 },
        PopulationSelector::TimeRange { start: 40, end: 80 },
        0.05,
        10,
    );
    let dets = detect_mechanism_changes(
        &model,
        &data,
        &q,
        MechanismChangeMethod::MeanDiff,
        &ExecutionContext::for_tests(1),
    )
    .unwrap();
    let y = dets.iter().find(|d| d.variable == VariableId::from_raw(1)).unwrap();
    assert!(y.changed, "y should be flagged changed: {y:?}");
}

fn two_period_data() -> (CompiledCausalModel, TabularData) {
    let n = 80usize;
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
    let mut xv = Vec::new();
    let mut yv = Vec::new();
    for i in 0..n {
        let x = (i % 40) as f64 * 0.1;
        xv.push(x);
        yv.push(if i < 40 { 1.0 + 2.0 * x } else { 6.0 + 2.0 * x });
    }
    let validity = ValidityBitmap::all_valid(n);
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(xv), validity.clone()).unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(yv), validity).unwrap(),
        ),
    ];
    let data = TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
    let mut g = Dag::with_variables(2);
    g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let model = CompiledCausalModel::compile(g).unwrap();
    (model, data)
}

#[test]
#[allow(clippy::float_cmp, reason = "the test checks exact boundary values")]
#[allow(
    clippy::single_range_in_vec_init,
    reason = "the fixture explicitly supplies one contiguous run"
)]
fn run_aware_block_stream_is_deterministic_and_keeps_intact_block_behavior() {
    let (model, data) = two_period_data();
    let q = MechanismChangeQuery::new(
        [VariableId::from_raw(0)],
        PopulationSelector::TimeRange { start: 0, end: 40 },
        PopulationSelector::TimeRange { start: 40, end: 80 },
        0.05,
        10,
    );
    let ctx = ExecutionContext::for_tests(71);
    let mut budget = antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
    let static_result = detect_mechanism_changes_block(
        &model,
        &data,
        &q,
        MechanismChangeMethod::MeanDiff,
        10,
        99,
        &ctx,
    )
    .unwrap();
    let intact = detect_mechanism_changes_block_with_runs(
        &model,
        &data,
        &q,
        MechanismChangeMethod::MeanDiff,
        10,
        99,
        &[0..40],
        &[0..40],
        &mut budget,
        &ctx,
    )
    .unwrap();
    assert_eq!(static_result[0].p_value, intact[0].p_value);
    let run = || {
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        detect_mechanism_changes_block_with_runs(
            &model,
            &data,
            &q,
            MechanismChangeMethod::MeanDiff,
            10,
            99,
            &[0..17, 17..40],
            &[0..13, 13..40],
            &mut budget,
            &ctx,
        )
        .unwrap()[0]
            .p_value
    };
    assert_eq!(run(), run());
    assert!(validate_runs(&[0..17, 18..40], 40, &ctx).is_err());
}

#[test]
fn partitioned_blocks_keep_every_row_without_crossing_a_run_join() {
    let segments = partition_permutation_blocks(
        &[0, 4, 8],
        4,
        12,
        6,
        &[0..3, 3..6],
        &[0..2, 2..6],
        &ExecutionContext::for_tests(1),
    )
    .unwrap();
    assert_eq!(
        segments,
        vec![(0, 0, 0, 3), (0, 1, 3, 4), (1, 1, 4, 6), (1, 2, 6, 8), (2, 3, 8, 12)]
    );
}

#[test]
#[allow(
    clippy::single_range_in_vec_init,
    reason = "the fixture explicitly supplies one contiguous run"
)]
fn run_validation_work_and_inner_lag_cancellation_are_refused() {
    let ctx = ExecutionContext::for_tests(1);
    let mut limits = ctx.temporal_materialization_limits();
    limits.max_work = 0;
    let mut budget = antecedent_core::TemporalBudget::new(limits);
    assert!(matches!(
        charge_run_validation((&[0..1], &[0..1]), &mut budget, &ctx),
        Err(AttributionError::Budget { .. })
    ));

    let mut cancelled = ExecutionContext::for_tests(1);
    cancelled.cancellation = antecedent_core::execution::CancellationToken::cancel_after_checks(2);
    let data = [1.0, 1.0, -1.0, -1.0];
    let runs = [0..2, 2..4];
    assert!(matches!(
        serial_inflation_with_runs(&data, &data, 4, Some((&runs, &runs)), Some(&cancelled)),
        Err(AttributionError::Cancelled)
    ));
}

#[test]
fn serial_inflation_does_not_pair_across_a_compaction_join() {
    let a = [1.0, 1.0, -1.0, -1.0];
    let b = a;
    let joined = serial_inflation(&a, &b, 2);
    let split = serial_inflation_with_runs(
        &a,
        &b,
        2,
        Some((&[0..2, 2..4], &[0..2, 2..4])),
        Some(&ExecutionContext::for_tests(1)),
    )
    .unwrap();
    assert!((joined - 1.25).abs() < 1e-12);
    assert!((split - 1.5).abs() < 1e-12);
}

#[test]
fn parent_shift_check_that_cannot_run_refuses() {
    let (model, data) = two_period_data();
    // Non-finite parent value in the comparison population: the parent mean-difference
    // test cannot be evaluated, so the covariate-shift guard must refuse, not pass.
    let mut xv = data.float64_values(VariableId::from_raw(0)).unwrap();
    xv[50] = f64::NAN;
    let yv = data.float64_values(VariableId::from_raw(1)).unwrap();
    let n = xv.len();
    let validity = ValidityBitmap::all_valid(n);
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(xv), validity.clone()).unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(yv), validity).unwrap(),
        ),
    ];
    let schema = data.schema().clone();
    let data = TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
    let q = MechanismChangeQuery::new(
        [VariableId::from_raw(1)],
        PopulationSelector::TimeRange { start: 0, end: 40 },
        PopulationSelector::TimeRange { start: 40, end: 80 },
        0.05,
        10,
    );
    let err = detect_mechanism_changes(
        &model,
        &data,
        &q,
        MechanismChangeMethod::MeanDiff,
        &ExecutionContext::for_tests(1),
    )
    .unwrap_err();
    assert!(
        matches!(&err, AttributionError::InvalidInput { message }
            if message.contains("parent-covariate shift check")),
        "{err:?}"
    );
}

#[test]
fn kernel_two_sample_flags_y_shift() {
    let (model, data) = two_period_data();
    let q = MechanismChangeQuery::new(
        [VariableId::from_raw(0), VariableId::from_raw(1)],
        PopulationSelector::TimeRange { start: 0, end: 40 },
        PopulationSelector::TimeRange { start: 40, end: 80 },
        0.05,
        10,
    );
    let dets = detect_mechanism_changes(
        &model,
        &data,
        &q,
        MechanismChangeMethod::KernelTwoSample,
        &ExecutionContext::for_tests(1),
    )
    .unwrap();
    let y = dets.iter().find(|d| d.variable == VariableId::from_raw(1)).unwrap();
    assert!(y.changed, "y should be flagged changed: {y:?}");
    assert_eq!(&*y.method, "kernel_two_sample");
    // The resolution of a permutation p-value is reported with it: 999 permutations
    // give a floor of exactly 1/1000, and a raw p-value cannot sit below it.
    assert_eq!(y.n_permutations, Some(999));
    assert!((y.p_value_floor.unwrap() - 0.001).abs() < 1e-15);
    assert!(y.p_value >= 0.001 - 1e-15);
}

/// The kernel null draws from the context's attribution stream: it follows the master
/// seed (reproducibly) and the verdict on a real shift is stable across seeds, with
/// every p-value inside the same resolution band.
#[test]
fn kernel_two_sample_follows_the_context_seed_and_agrees_across_seeds() {
    let (model, data) = two_period_data();
    let q = MechanismChangeQuery::new(
        [VariableId::from_raw(1)],
        PopulationSelector::TimeRange { start: 0, end: 40 },
        PopulationSelector::TimeRange { start: 40, end: 80 },
        0.05,
        10,
    );
    let run = |seed: u64| {
        detect_mechanism_changes(
            &model,
            &data,
            &q,
            MechanismChangeMethod::KernelTwoSample,
            &ExecutionContext::for_tests(seed),
        )
        .unwrap()
        .remove(0)
    };
    let again = run(1);
    assert_eq!(again.p_value.to_bits(), run(1).p_value.to_bits());
    for seed in 1..=6u64 {
        let d = run(seed);
        assert!(d.changed, "seed {seed}: {d:?}");
        assert!(d.p_value >= d.p_value_floor.unwrap() - 1e-15);
        assert!(d.p_value <= 0.01, "seed {seed}: p={}", d.p_value);
        assert_eq!(d.statistic.to_bits(), again.statistic.to_bits(), "statistic is seed-free");
    }
}

#[test]
fn change_point_flags_y_shift() {
    let (model, data) = two_period_data();
    let q = MechanismChangeQuery::new(
        [VariableId::from_raw(0), VariableId::from_raw(1)],
        PopulationSelector::TimeRange { start: 0, end: 40 },
        PopulationSelector::TimeRange { start: 40, end: 80 },
        0.05,
        10,
    );
    let dets = detect_mechanism_changes(
        &model,
        &data,
        &q,
        MechanismChangeMethod::ChangePoint,
        &ExecutionContext::for_tests(1),
    )
    .unwrap();
    let y = dets.iter().find(|d| d.variable == VariableId::from_raw(1)).unwrap();
    assert!(y.changed, "y should be flagged changed: {y:?}");
    assert_eq!(&*y.method, "change_point");
}

/// A 1-row comparison population has zero within-population variance by
/// construction. Before the fix, `mechanism_change` resolved populations directly
/// (bypassing `prep::resolve_population_pair`'s emptiness check and imposing no
/// minimum row count), and `LikelihoodRatio` reached `gaussian_segment_lr`'s
/// `v1 == 0.0` branch, which reported `changed: true, p_value: 0.0, statistic: inf`
/// — the strongest possible "changed" claim, manufactured from a single observation.
/// The fix refuses any population below a documented minimum before fitting anything.
#[test]
fn refuses_single_row_comparison_population() {
    let (model, data) = two_period_data();
    let q = MechanismChangeQuery::new(
        [VariableId::from_raw(1)],
        PopulationSelector::TimeRange { start: 0, end: 40 },
        PopulationSelector::Rows(std::sync::Arc::from([17usize])),
        0.05,
        10,
    );
    for method in [MechanismChangeMethod::LikelihoodRatio, MechanismChangeMethod::ChangePoint] {
        let result =
            detect_mechanism_changes(&model, &data, &q, method, &ExecutionContext::for_tests(1));
        assert!(
            result.is_err(),
            "a 1-row comparison population must be refused, not scored as a \
             definitive mechanism-change claim (method={method:?}): {result:?}"
        );
    }
}

/// Same defect, reached via a constant (rather than merely small) comparison
/// population: several rows, but every value bit-identical, so the segment still has
/// zero variance. This is caught by the same minimum-row gate here (below
/// `MIN_POPULATION_ROWS`); a comparison population at or above the minimum but still
/// exactly constant is caught downstream by `gaussian_segment_lr`'s own refusal
/// (`antecedent_stats::divergence::residual_lr_refuses_single_row_or_constant_segments`).
#[test]
fn refuses_undersized_comparison_population() {
    let (model, data) = two_period_data();
    let q = MechanismChangeQuery::new(
        [VariableId::from_raw(1)],
        PopulationSelector::TimeRange { start: 0, end: 40 },
        PopulationSelector::Rows(std::sync::Arc::from([10usize, 11, 12])),
        0.05,
        10,
    );
    let result = detect_mechanism_changes(
        &model,
        &data,
        &q,
        MechanismChangeMethod::LikelihoodRatio,
        &ExecutionContext::for_tests(1),
    );
    assert!(result.is_err(), "a 3-row comparison population must be refused: {result:?}");
}

/// `x`'s range genuinely differs by 5 units between the two populations here — exactly
/// the material parent-covariate shift `parent_covariate_shift_detected` now refuses
/// on, so this documents the current, honest behavior (refusal) rather than the
/// original claim (a same-mechanism verdict). The original pass was itself an artifact
/// of this test's near-deterministic 0.01-amplitude periodic "noise": with that little
/// noise, the baseline fit has almost no parameter uncertainty to be wrong about, so
/// the covariate shift never actually caused a false positive here even before the
/// refusal existed — see `mean_diff_covariate_shift_is_calibrated_or_refused` for the
/// case (realistic Gaussian noise) where it does.
#[test]
fn mean_diff_refuses_under_material_covariate_shift_even_with_same_mechanism() {
    let n = 80usize;
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
    let mut xv = Vec::new();
    let mut yv = Vec::new();
    for i in 0..n {
        // Same y = 1 + 2x + noise; x ranges differ → different marginal y mean, same mechanism.
        let x = if i < 40 { (i % 40) as f64 * 0.1 } else { 5.0 + (i % 40) as f64 * 0.1 };
        xv.push(x);
        yv.push(1.0 + 2.0 * x + 0.01 * ((i % 7) as f64 - 3.0));
    }
    let validity = ValidityBitmap::all_valid(n);
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(xv), validity.clone()).unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(yv), validity).unwrap(),
        ),
    ];
    let data = TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
    let mut g = Dag::with_variables(2);
    g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let model = CompiledCausalModel::compile(g).unwrap();
    let q = MechanismChangeQuery::new(
        [VariableId::from_raw(1)],
        PopulationSelector::TimeRange { start: 0, end: 40 },
        PopulationSelector::TimeRange { start: 40, end: 80 },
        0.05,
        10,
    );
    let result = detect_mechanism_changes(
        &model,
        &data,
        &q,
        MechanismChangeMethod::MeanDiff,
        &ExecutionContext::for_tests(1),
    );
    assert!(
        result.is_err(),
        "a materially shifted parent covariate should be refused, not scored: {result:?}"
    );
}

/// Null split of a homogeneous SCM: residual `MeanDiff` Type I smoke.
/// Not a full calibration gate — only checks flag rate stays near α.
#[test]
fn mean_diff_null_split_type_i_smoke() {
    let n = 200usize;
    let mid = n / 2;
    let mut flags = 0usize;
    let trials = 20usize;
    for trial in 0..trials {
        let mut rng = ExecutionContext::for_tests(0x4C01u64.wrapping_add(trial as u64))
            .rng
            .stream_for(StreamDomain::Attribution, 0x7E51);
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
        let mut xv = Vec::with_capacity(n);
        let mut yv = Vec::with_capacity(n);
        for _ in 0..n {
            let x = antecedent_kernels::standard_normal(&mut rng);
            let e = antecedent_kernels::standard_normal(&mut rng);
            xv.push(x);
            yv.push(1.0 + 2.0 * x + 0.25 * e);
        }
        let validity = ValidityBitmap::all_valid(n);
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(0), Arc::from(xv), validity.clone())
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(1), Arc::from(yv), validity).unwrap(),
            ),
        ];
        let data =
            TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
        let mut g = Dag::with_variables(2);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let model = CompiledCausalModel::compile(g).unwrap();
        let q = MechanismChangeQuery::new(
            [VariableId::from_raw(1)],
            PopulationSelector::TimeRange { start: 0, end: mid },
            PopulationSelector::TimeRange { start: mid, end: n },
            0.05,
            10,
        );
        let dets = detect_mechanism_changes(
            &model,
            &data,
            &q,
            MechanismChangeMethod::MeanDiff,
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
        let y = dets.iter().find(|d| d.variable == VariableId::from_raw(1)).unwrap();
        if y.changed {
            flags += 1;
        }
    }
    // α=0.05 → expect ~1 flag in 20; allow a few extras for smoke (not a calibration gate).
    assert!(
        flags <= 4,
        "null-split Type I smoke: MeanDiff flagged {flags}/{trials} (want ≤4 at α=0.05)"
    );
}

/// Family-wise Type I inflation across several truly-unchanged targets, and control via the
/// default Benjamini–Hochberg correction (Defect A).
///
/// `mean_diff_null_split_type_i_smoke` above only checks the *per-target* flag rate with a
/// single target, so it cannot see the family-wise effect: with `m` independent null targets
/// each judged against α=0.05 independently, `P(>=1 false flag) = 1 - 0.95^m`, which is
/// ≈40% at m=10. This test builds `m` independent, unrelated mechanisms (no true baseline →
/// comparison change in any of them) and compares the family-wise flag rate (>=1 of the `m`
/// targets flagged) with `correction: None` against the corrected default
/// `detect_mechanism_changes`.
#[test]
#[allow(
    clippy::too_many_lines,
    reason = "one fixture exercised end to end; later assertions read locals built above them"
)]
fn mean_diff_null_split_family_wise_type_i_rate() {
    let m = 10usize;
    let n = 120usize;
    let mid = n / 2;
    let trials = 150usize;
    let alpha = 0.05;

    let mut uncorrected_family_flags = 0usize;
    let mut corrected_family_flags = 0usize;

    for trial in 0..trials {
        let mut rng = ExecutionContext::for_tests(0x9A11u64.wrapping_add(trial as u64))
            .rng
            .stream_for(StreamDomain::Attribution, 0x51EE);
        let mut b = CausalSchemaBuilder::new();
        for k in 0..m {
            b.add_variable(
                format!("x{k}"),
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
            b.add_variable(
                format!("y{k}"),
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        }
        let schema = b.build().unwrap();
        let mut cols = Vec::with_capacity(2 * m);
        for k in 0..m {
            let mut xv = Vec::with_capacity(n);
            let mut yv = Vec::with_capacity(n);
            for _ in 0..n {
                let x = antecedent_kernels::standard_normal(&mut rng);
                let e = antecedent_kernels::standard_normal(&mut rng);
                xv.push(x);
                yv.push(1.0 + 2.0 * x + 0.25 * e);
            }
            let validity = ValidityBitmap::all_valid(n);
            cols.push(OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw((2 * k) as u32),
                    Arc::from(xv),
                    validity.clone(),
                )
                .unwrap(),
            ));
            cols.push(OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw((2 * k + 1) as u32),
                    Arc::from(yv),
                    validity,
                )
                .unwrap(),
            ));
        }
        let data =
            TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
        let mut g = Dag::with_variables((2 * m) as u32);
        for k in 0..m {
            g.insert_directed(
                DenseNodeId::from_raw((2 * k) as u32),
                DenseNodeId::from_raw((2 * k + 1) as u32),
            )
            .unwrap();
        }
        let model = CompiledCausalModel::compile(g).unwrap();
        let targets: Vec<VariableId> =
            (0..m).map(|k| VariableId::from_raw((2 * k + 1) as u32)).collect();
        let q = MechanismChangeQuery::new(
            targets,
            PopulationSelector::TimeRange { start: 0, end: mid },
            PopulationSelector::TimeRange { start: mid, end: n },
            alpha,
            m,
        );

        let uncorrected = detect_mechanism_changes_with_correction(
            &model,
            &data,
            &q,
            MechanismChangeMethod::MeanDiff,
            &ExecutionContext::for_tests(1),
            None,
        )
        .unwrap();
        if uncorrected.iter().any(|d| d.changed) {
            uncorrected_family_flags += 1;
        }

        let corrected = detect_mechanism_changes(
            &model,
            &data,
            &q,
            MechanismChangeMethod::MeanDiff,
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
        if corrected.iter().any(|d| d.changed) {
            corrected_family_flags += 1;
        }
    }

    let uncorrected_rate = uncorrected_family_flags as f64 / trials as f64;
    let corrected_rate = corrected_family_flags as f64 / trials as f64;
    // Expected uncorrected family-wise rate ≈ 1 - 0.95^10 ≈ 0.40; allow slop for MC noise.
    assert!(
        uncorrected_rate > 0.20,
        "expected inflated uncorrected family-wise rate (~0.40 expected), got \
         {uncorrected_rate} ({uncorrected_family_flags}/{trials})"
    );
    assert!(
        corrected_rate < uncorrected_rate - 0.10,
        "BH-corrected family-wise rate ({corrected_rate}, {corrected_family_flags}/{trials}) \
         should be well below the uncorrected rate ({uncorrected_rate})"
    );
    assert!(
        corrected_rate <= 0.15,
        "BH-corrected family-wise rate should stay near α=0.05 under the global null, got \
         {corrected_rate} ({corrected_family_flags}/{trials})"
    );
}

/// Build one draw of `y = 1 + 2x + N(0,1)` with the *same* mechanism in both
/// populations, but `x` shifted between them: `x_baseline ~ N(0,1)`,
/// `x_comparison ~ N(shift,1)`. Baseline rows come first.
fn covariate_shift_dataset(
    n_b: usize,
    n_c: usize,
    shift: f64,
    rng: &mut CausalRng,
) -> (CompiledCausalModel, TabularData) {
    let n = n_b + n_c;
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
    let mut xv = Vec::with_capacity(n);
    let mut yv = Vec::with_capacity(n);
    for i in 0..n {
        let mean_x = if i < n_b { 0.0 } else { shift };
        let x = mean_x + antecedent_kernels::standard_normal(rng);
        let e = antecedent_kernels::standard_normal(rng);
        xv.push(x);
        yv.push(1.0 + 2.0 * x + e);
    }
    let validity = ValidityBitmap::all_valid(n);
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(xv), validity.clone()).unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(yv), validity).unwrap(),
        ),
    ];
    let data = TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
    let mut g = Dag::with_variables(2);
    g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let model = CompiledCausalModel::compile(g).unwrap();
    (model, data)
}

/// Empirical Type-I (false-positive) rate of a residual two-sample method over
/// `trials` independent draws of [`covariate_shift_dataset`], at nominal `alpha`.
/// Trials on which the call is refused (material covariate shift detected) are
/// excluded from the denominator and counted separately, since a refusal is not a
/// false positive — it is the honest alternative to one.
fn covariate_shift_outcome_rates(
    method: MechanismChangeMethod,
    shift: f64,
    trials: usize,
    alpha: f64,
    seed_base: u64,
) -> (f64, f64) {
    let n_b = 60usize;
    let n_c = 60usize;
    let mut flags = 0usize;
    let mut refusals = 0usize;
    let mut scored = 0usize;
    for trial in 0..trials {
        let mut gen_rng =
            ExecutionContext::for_tests(seed_base.wrapping_add(trial as u64)).rng.stream(0xC0);
        let (model, data) = covariate_shift_dataset(n_b, n_c, shift, &mut gen_rng);
        let q = MechanismChangeQuery::new(
            [VariableId::from_raw(1)],
            PopulationSelector::TimeRange { start: 0, end: n_b },
            PopulationSelector::TimeRange { start: n_b, end: n_b + n_c },
            alpha,
            10,
        );
        match detect_mechanism_changes_with_correction(
            &model,
            &data,
            &q,
            method,
            &ExecutionContext::for_tests(1),
            None,
        ) {
            Ok(dets) => {
                scored += 1;
                let y = dets.iter().find(|d| d.variable == VariableId::from_raw(1)).unwrap();
                if y.changed {
                    flags += 1;
                }
            }
            Err(_) => refusals += 1,
        }
    }
    let flag_rate = if scored == 0 { 0.0 } else { flags as f64 / scored as f64 };
    let refusal_rate = refusals as f64 / trials as f64;
    (flag_rate, refusal_rate)
}

/// `attr-design-state-2`: residual two-sample tests compared in-sample baseline
/// residuals against out-of-sample comparison residuals. Under an *unchanged*
/// mechanism but a shifted parent distribution, the baseline model's own
/// extrapolation error inflates only the comparison side, so the null is rejected far
/// above the nominal rate — the reviewer's Python simulation found ≈12% at a 1σ
/// shift, ≈40% at 3σ, ≈61% at 5σ, against a nominal α=0.05.
///
/// This mirrors that simulation in-crate: `y = 1 + 2x + N(0,1)` with the identical
/// mechanism in both populations, only `x`'s mean shifted. At `shift=0` (no covariate
/// difference), the parent-shift guard essentially never fires and the cross-fitted
/// baseline keeps the flag rate near nominal. At `shift=3` and `shift=5`, no residual
/// reconstruction can make a same-distribution two-sample test calibrated against the
/// resulting shared bias (see the comment in
/// `detect_mechanism_changes_with_correction`), so the call is refused instead of
/// reporting an anti-conservative p-value: refusals should dominate, not false flags.
#[test]
fn mean_diff_covariate_shift_is_calibrated_or_refused() {
    let alpha = 0.05;
    let trials = 300usize;
    let (flag_rate, refusal_rate) = covariate_shift_outcome_rates(
        MechanismChangeMethod::MeanDiff,
        0.0,
        trials,
        alpha,
        0xC0_1B_A5_E0,
    );
    assert!(
        refusal_rate < 0.05,
        "shift=0 should almost never trigger the covariate-shift refusal, got {refusal_rate}"
    );
    assert!(
        flag_rate < 0.15,
        "MeanDiff Type-I rate at shift=0 should stay close to nominal α={alpha}, got {flag_rate}"
    );
    for shift in [3.0, 5.0] {
        let (flag_rate, refusal_rate) = covariate_shift_outcome_rates(
            MechanismChangeMethod::MeanDiff,
            shift,
            trials,
            alpha,
            0xC0_1B_A5_E0,
        );
        assert!(
            refusal_rate > 0.85,
            "MeanDiff at shift={shift} should be refused, not scored, on almost every \
             trial (a residual two-sample test cannot be calibrated against this \
             covariate shift): refusal_rate={refusal_rate}, flag_rate={flag_rate}"
        );
    }
}

/// Same reproduction as above, for `LikelihoodRatio` (the other test the finding
/// names explicitly).
#[test]
fn likelihood_ratio_covariate_shift_is_calibrated_or_refused() {
    let alpha = 0.05;
    let trials = 300usize;
    let (flag_rate, refusal_rate) = covariate_shift_outcome_rates(
        MechanismChangeMethod::LikelihoodRatio,
        0.0,
        trials,
        alpha,
        0x11_4E_11_D0,
    );
    assert!(
        refusal_rate < 0.05,
        "shift=0 should almost never trigger the covariate-shift refusal, got {refusal_rate}"
    );
    assert!(
        flag_rate < 0.15,
        "LikelihoodRatio Type-I rate at shift=0 should stay close to nominal α={alpha}, \
         got {flag_rate}"
    );
    for shift in [3.0, 5.0] {
        let (flag_rate, refusal_rate) = covariate_shift_outcome_rates(
            MechanismChangeMethod::LikelihoodRatio,
            shift,
            trials,
            alpha,
            0x11_4E_11_D0,
        );
        assert!(
            refusal_rate > 0.85,
            "LikelihoodRatio at shift={shift} should be refused, not scored, on almost \
             every trial: refusal_rate={refusal_rate}, flag_rate={flag_rate}"
        );
    }
}

/// Defect B: `residuals()` must thread a properly seeded RNG into posterior noise recovery
/// (`infer_noise_column_rng`) rather than resetting to a fixed `CausalRng::from_seed(0)` on
/// every call (the old `infer_noise_column` shim).
///
/// A `Discrete`-family target's noise-inference mode is always `Posterior` (see
/// `infer_noise_column_rng` in `antecedent-model/src/mechanism.rs`), so its residual draws
/// are genuinely a function of the RNG stream, not just of the data. This test forces
/// `MechanismRegistry::standard()` to fit `y` as `Discrete` (by giving it few distinct
/// values, since family eligibility is driven by data cardinality via `is_low_cardinality`,
/// not by the declared `ValueType`) and then runs `detect_mechanism_changes` twice under two
/// different `ExecutionContext` seeds. Post-fix, the seed is threaded through to the
/// posterior draws, so the resulting `MeanDiff` statistic differs between the two seeds.
/// Pre-fix, both calls silently reset to seed 0 regardless of `ExecutionContext`, so the
/// statistic is bit-identical across seeds — the RNG parameter is a no-op for any
/// `Posterior`-mode family.
///
/// Per the task brief: this does **not** attempt to show a flipped `changed` verdict (the
/// within-population draws stay i.i.d. conditioned on whichever seed is used, and the
/// two-sample tests treat samples as unordered sets, so a verdict flip could not be
/// constructed). It demonstrates the narrower, honest claim that the execution context's
/// seed is actually being honored after the fix and was being silently ignored before it.
#[test]
fn discrete_target_residuals_depend_on_execution_context_seed() {
    let n = 60usize;
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
    // x has 4 distinct categories, y has 2 — both well under the is_low_cardinality(_, 8)
    // threshold that routes the target to MechanismRegistry::standard().discrete =
    // [Discrete, Constant], and the x >= 2 split gives Discrete a real edge over Constant.
    let mut xv = Vec::with_capacity(n);
    let mut yv = Vec::with_capacity(n);
    for i in 0..n {
        let x = (i % 4) as f64;
        xv.push(x);
        yv.push(if x >= 2.0 { 1.0 } else { 0.0 });
    }
    let validity = ValidityBitmap::all_valid(n);
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(xv), validity.clone()).unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(yv), validity).unwrap(),
        ),
    ];
    let data = TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
    let mut g = Dag::with_variables(2);
    g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let model = CompiledCausalModel::compile(g).unwrap();
    let q = MechanismChangeQuery::new(
        [VariableId::from_raw(1)],
        PopulationSelector::TimeRange { start: 0, end: n },
        PopulationSelector::TimeRange { start: 0, end: n },
        0.05,
        10,
    );

    let d1 = detect_mechanism_changes(
        &model,
        &data,
        &q,
        MechanismChangeMethod::MeanDiff,
        &ExecutionContext::for_tests(1),
    )
    .unwrap();
    let d2 = detect_mechanism_changes(
        &model,
        &data,
        &q,
        MechanismChangeMethod::MeanDiff,
        &ExecutionContext::for_tests(2),
    )
    .unwrap();
    let s1 = d1.iter().find(|d| d.variable == VariableId::from_raw(1)).unwrap().statistic;
    let s2 = d2.iter().find(|d| d.variable == VariableId::from_raw(1)).unwrap().statistic;
    // Separated by a real margin, not just `!=`: posterior-sampled noise under two
    // different seeds should move the statistic materially, and an exact-inequality
    // check would also pass on a one-ULP difference.
    assert!(
        (s1 - s2).abs() > 1e-9,
        "Discrete-family residual statistic should depend on the ExecutionContext seed \
         (s1={s1}, s2={s2}); identical values mean the RNG is being ignored"
    );
}
