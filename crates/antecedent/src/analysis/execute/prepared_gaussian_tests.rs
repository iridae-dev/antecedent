// SPDX-License-Identifier: MIT OR Apache-2.0
//! Public prepared Gaussian execution regression tests.
use crate::{RefuteSuite, Study};
use antecedent_core::{AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, TargetForm};
use antecedent_data::TabularData;
use antecedent_graph::Dag;
use antecedent_stats::fit_counts::count_least_squares_solves;

fn contrast_study() -> Study {
    contrast_study_with_units(None)
}

fn contrast_study_with_units(unit: Option<&str>) -> Study {
    let outcome = CausalSchemaBuilder::new().continuous("t").treatment().continuous("y");
    let outcome = if let Some(unit) = unit { outcome.unit(unit) } else { outcome };
    let schema = outcome.outcome().build().unwrap();
    let treatment = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    let outcome = [1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0];
    let data = TabularData::try_from_schema_f64(
        schema.clone(),
        [("t", treatment.as_slice()), ("y", outcome.as_slice())],
    )
    .unwrap();
    let graph = Dag::from_named_edges(&schema, &[("t", "y")]).unwrap();
    let query =
        AverageEffectQuery::binary_ate(schema.id_of("t").unwrap(), schema.id_of("y").unwrap());
    Study::tabular(data)
        .graph(graph)
        .query(query)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
}

#[test]
fn prepared_gaussian_retains_the_compiled_program_after_builder_drop() {
    let ctx = ExecutionContext::for_tests(1);
    let study = contrast_study_with_units(Some("widgets"));
    let (compilation, preparation_solves) = count_least_squares_solves(|| {
        antecedent_core::execution_attempt::observe_execution(|| study.prepare(&ctx))
    });
    assert_eq!(preparation_solves, 0, "compilation must not fit");
    let public_compilation =
        antecedent_core::execution_attempt::observe_execution(|| study.compile(&ctx));
    assert_eq!(
        public_compilation
            .report
            .counts(antecedent_core::execution_attempt::Operation::Identification)
            .attempted,
        1
    );
    let plan = public_compilation.result.unwrap();
    assert!(plan.compiled_linear_program().is_some());
    assert_eq!(
        compilation
            .report
            .counts(antecedent_core::execution_attempt::Operation::Identification)
            .attempted,
        1,
        "preparation must not repeat identification"
    );
    let prepared = compilation.result.unwrap();
    assert_eq!(plan.record, prepared.plan().record);
    assert_eq!(
        plan.choice.selected(),
        antecedent_core::physical_choice::PhysicalAlternative::DirectPreparedNumericViews
    );
    assert_eq!(
        prepared.execution().checked_linear().unwrap().physical_backend(),
        prepared.plan().choice.selected()
    );
    let program = prepared.compiled_linear_program().unwrap();
    assert!(
        std::ptr::eq(program, prepared.plan().compiled_linear_program().unwrap()),
        "inspection and execution must retain the identical compiled product"
    );
    assert!(matches!(program.target().form(), TargetForm::Contrast(_)));
    let TargetForm::Contrast(contrast) = program.target().form() else { unreachable!() };
    for world in contrast.worlds().worlds() {
        assert_eq!(
            world.binding().outcome().units(),
            &antecedent_core::QuantityUnits::new("widgets").unwrap()
        );
    }
    assert_eq!(program.identification().world_obligations().len(), 2);
    assert_eq!(program.identification().target_identity(), Some(program.target().identity()));
    program.identification().validate_target(program.target()).unwrap();
    assert_eq!(
        program.identification().assumptions(),
        prepared.execution().checked_linear().unwrap().required_assumptions()
    );
    assert_eq!(program.operators().len(), 3);
    assert_eq!(program.physical_operators().len(), 3);
    drop(study);
    let (first, solves) = count_least_squares_solves(|| prepared.estimate_retained(&ctx).unwrap());
    assert!(solves > 0);
    let facts = first.performance.native_execution.as_ref().unwrap();
    assert!(facts.is_native());
    assert_eq!(facts.observed_model_fits(), Some(1));
    assert_eq!(
        facts.peak_admitted_bytes(),
        prepared.plan().record.estimated_peak_memory_bytes,
        "the receipt includes the design retained before observation began"
    );
    assert_eq!(first.performance.peak_rss_bytes, None);
    let (again, solves) = count_least_squares_solves(|| prepared.estimate_retained(&ctx).unwrap());
    assert_eq!(solves, 0);
    let facts = again.performance.native_execution.as_ref().unwrap();
    assert_eq!(facts.observed_model_fits(), Some(0));
    assert_eq!(facts.counts().cache_hits(), 1);
    assert_eq!(first.effect().to_bits(), again.effect().to_bits());
}

#[test]
fn prepared_gaussian_refuses_cancelled_and_insufficient_resource_execution() {
    let ctx = ExecutionContext::for_tests(1);
    let prepared = contrast_study().prepare(&ctx).unwrap();
    let mut bounded = ctx.clone();
    bounded.set_memory_limit(Some(1));
    let refused = prepared.estimate_retained_observed(&bounded);
    assert!(matches!(refused.result, Err(crate::CausalError::Resource { .. })));
    assert_eq!(
        refused.facts.unwrap().status(),
        antecedent_core::execution_facts::ExecutionFactStatus::Failed
    );
    ctx.cancellation.cancel();
    let cancelled = prepared.estimate_retained_observed(&ctx);
    assert!(matches!(cancelled.result, Err(crate::CausalError::Cancelled { .. })));
    assert_eq!(
        cancelled.facts.unwrap().status(),
        antecedent_core::execution_facts::ExecutionFactStatus::Cancelled
    );
}

#[test]
fn prepared_gaussian_reservations_follow_builder_cache_and_result_lifetimes() {
    use antecedent_core::execution::AdmissionPool;
    let pool = AdmissionPool::fixed(1, 1_000_000);
    let ctx = ExecutionContext::for_tests(1).with_admission(pool.clone());
    let prepared = contrast_study().prepare(&ctx).unwrap();
    let prepared_bytes = pool.byte_budget() - pool.available().bytes();
    assert!(prepared_bytes > 0);
    let first = prepared.estimate_retained(&ctx).unwrap();
    let held = pool.available().bytes();
    assert!(held < pool.byte_budget() - prepared_bytes);
    let again = prepared.estimate_retained(&ctx).unwrap();
    let held_again = pool.available().bytes();
    let first_output_charge = pool.byte_budget() - prepared_bytes - held;
    let second_output_charge = held - held_again;
    assert!(second_output_charge > 0, "each result has its own metadata");
    assert!(second_output_charge < first_output_charge, "the fitted buffer is shared");
    ctx.cancellation.cancel();
    assert_eq!(pool.available().bytes(), held_again, "live outputs survive cancellation");
    drop(prepared);
    assert!(pool.available().bytes() > held_again);
    assert!(pool.available().bytes() < pool.byte_budget());
    drop(first);
    assert!(pool.available().bytes() < pool.byte_budget());
    drop(again);
    assert_eq!(pool.available().bytes(), pool.byte_budget());
}

#[test]
fn prepared_gaussian_quotes_selected_columns_and_resampling_work() {
    let mut builder =
        CausalSchemaBuilder::new().continuous("t").treatment().continuous("y").outcome();
    for index in 0..512 {
        builder = builder.continuous(format!("unused{index}")).context();
    }
    let schema = builder.build().unwrap();
    let treatment: Vec<_> = (0..64).map(|i| f64::from(i % 2)).collect();
    let outcome: Vec<_> = treatment.iter().map(|t| 1.0 + 2.0 * t).collect();
    let mut columns = vec![("t".to_string(), treatment.clone()), ("y".to_string(), outcome)];
    for index in 0..512 {
        columns.push((format!("unused{index}"), treatment.clone()));
    }
    let data = TabularData::try_from_schema_f64(
        schema.clone(),
        columns.iter().map(|(name, values)| (name.as_str(), values.as_slice())),
    )
    .unwrap();
    let graph = Dag::from_named_edges(&schema, &[("t", "y")]).unwrap();
    let query =
        AverageEffectQuery::binary_ate(schema.id_of("t").unwrap(), schema.id_of("y").unwrap());
    let study = Study::tabular(data)
        .graph(graph)
        .query(query)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let mut ctx = ExecutionContext::for_tests(1);
    ctx.set_memory_limit(Some(250_000));
    let prepared = study.prepare(&ctx).unwrap();
    assert_eq!(prepared.plan().logical.design_columns_hint, 2);
    assert_eq!(prepared.plan().logical.row_count_hint, 64);
    assert!((prepared.estimate_retained(&ctx).unwrap().effect() - 2.0).abs() < 1e-12);
    let mut resampled = study;
    resampled.settings.bootstrap_replicates = 200;
    let mut parallel = ExecutionContext::production(1, 4);
    parallel.set_memory_limit(Some(250_000));
    assert!(matches!(resampled.compile(&parallel), Err(crate::CausalError::Resource { .. })));
}

#[test]
fn prepared_gaussian_materializes_the_actual_categorical_design() {
    use antecedent_core::physical_choice::PhysicalAlternative;
    use antecedent_core::{CategoryDomainId, MeasurementSpec, SmallRoleSet, ValueType};
    use antecedent_data::{
        CategoricalColumn, CategoryCode, CategoryDomain, CategoryLevel, Contrast, Float64Column,
        OwnedColumn, OwnedColumnarStorage, UnknownCategoryPolicy, ValidityBitmap,
    };
    use std::sync::Arc;
    let mut builder =
        CausalSchemaBuilder::new().continuous("t").treatment().continuous("y").outcome();
    builder
        .add_variable(
            "z",
            ValueType::Categorical,
            SmallRoleSet::empty(),
            None,
            Some(CategoryDomainId::from_raw(2)),
            MeasurementSpec::default(),
        )
        .unwrap();
    let schema = builder.build().unwrap();
    let t = schema.id_of("t").unwrap();
    let y = schema.id_of("y").unwrap();
    let z = schema.id_of("z").unwrap();
    let domain = Arc::new(
        CategoryDomain::try_new(
            CategoryDomainId::from_raw(2),
            (0..4)
                .map(|i| CategoryLevel { label: Arc::from(format!("level{i}")) })
                .collect::<Vec<_>>(),
            false,
            Some(CategoryCode::from_raw(0)),
            UnknownCategoryPolicy::Fail,
        )
        .unwrap(),
    );
    let treatment: Vec<_> = (0..64).map(|i| f64::from(i % 2)).collect();
    let category: Vec<_> = (0..64).map(|i| CategoryCode::from_raw((i / 2) % 4)).collect();
    let outcome: Vec<_> = (0..64)
        .map(|i| 2.0 * treatment[i] + f64::from(u32::try_from((i / 2) % 4).unwrap()))
        .collect();
    let columns = vec![
        OwnedColumn::Float64(
            Float64Column::new(t, Arc::<[f64]>::from(treatment), ValidityBitmap::all_valid(64))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(y, Arc::<[f64]>::from(outcome), ValidityBitmap::all_valid(64))
                .unwrap(),
        ),
        OwnedColumn::Categorical(
            CategoricalColumn::try_new(z, category, ValidityBitmap::all_valid(64), domain).unwrap(),
        ),
    ];
    let data = TabularData::new(
        OwnedColumnarStorage::try_new(schema.clone(), columns, None, None).unwrap(),
    );
    let graph = Dag::from_named_edges(&schema, &[("t", "y"), ("z", "t"), ("z", "y")]).unwrap();
    let fitter = antecedent_estimate::LinearAdjustmentAte::new()
        .with_bootstrap_replicates(0)
        .with_contrast(z, Contrast::Treatment { reference: CategoryCode::from_raw(0) });
    let ctx = ExecutionContext::for_tests(1);
    let prepared = Study::tabular(data)
        .graph(graph)
        .query(AverageEffectQuery::binary_ate(t, y))
        .estimator(fitter)
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    assert_eq!(prepared.plan().choice.selected(), PhysicalAlternative::MaterializedDesign);
    assert_eq!(prepared.plan().logical.design_columns_hint, 5);
    let checked = prepared.execution().checked_linear().unwrap();
    assert_eq!(checked.physical_backend(), prepared.plan().choice.selected());
    assert_eq!(checked.problem().design.ncols, 5);
    assert_eq!(checked.problem().preparation.categories()[0].n_columns, 3);
    let result = prepared.estimate_retained(&ctx).unwrap();
    assert!((result.effect() - 2.0).abs() < 1e-12);
    assert_eq!(result.performance.native_execution.unwrap().observed_model_fits(), Some(1));
}

#[test]
fn prepared_gaussian_refresh_replans_changed_rows_without_reidentifying() {
    let ctx = ExecutionContext::for_tests(1);
    let mut prepared = contrast_study().prepare(&ctx).unwrap();
    let original = prepared.estimate_retained(&ctx).unwrap();
    let original_source = prepared.compiled_linear_program().unwrap().source_digest();
    let schema = CausalSchemaBuilder::new()
        .continuous("t")
        .treatment()
        .continuous("y")
        .outcome()
        .build()
        .unwrap();
    let treatment: Vec<_> = (0..16).map(|i| f64::from(i % 2)).collect();
    let outcome: Vec<_> = treatment.iter().map(|t| 1.0 + 3.0 * t).collect();
    let data = TabularData::try_from_schema_f64(
        schema,
        [("t", treatment.as_slice()), ("y", outcome.as_slice())],
    )
    .unwrap();
    let (observed, solves) = count_least_squares_solves(|| {
        antecedent_core::execution_attempt::observe_execution(|| {
            crate::analysis::prepared::rebind_estimate(&mut prepared, data, &ctx)
        })
    });
    assert!(solves > 0);
    assert_eq!(
        observed
            .report
            .counts(antecedent_core::execution_attempt::Operation::Identification)
            .attempted,
        0
    );
    assert_eq!(prepared.plan().logical.row_count_hint, 16);
    assert_ne!(prepared.compiled_linear_program().unwrap().source_digest(), original_source);
    assert_eq!(prepared.plan().record.batch_size, Some(16));
    assert!((observed.result.unwrap().effect() - 3.0).abs() < 1e-12);
    assert!((original.effect() - 2.0).abs() < 1e-12);
    let (_, solves) = count_least_squares_solves(|| prepared.estimate_retained(&ctx).unwrap());
    assert_eq!(solves, 0);
}

#[test]
fn compile_observes_cancellation_and_a_refused_compile_holds_no_lease() {
    use antecedent_core::execution::AdmissionPool;
    let pool = AdmissionPool::fixed(1, 1_000_000);
    let ctx = ExecutionContext::for_tests(1).with_admission(pool.clone());
    let study = contrast_study();

    ctx.cancellation.cancel();
    let refused = study.compile(&ctx);
    assert!(matches!(refused, Err(crate::CausalError::Cancelled { .. })));
    assert_eq!(pool.available().bytes(), 1_000_000, "a cancelled compile reserves nothing");

    let live = ExecutionContext::for_tests(1).with_admission(pool.clone());
    let plan = study.compile(&live).unwrap();
    assert!(
        pool.available().bytes() < 1_000_000,
        "the compiled plan retains the bytes of the design it prepared"
    );
    drop(plan);
    assert_eq!(pool.available().bytes(), 1_000_000, "dropping the plan releases its retention");

    let mut starved = ExecutionContext::for_tests(1).with_admission(pool.clone());
    starved.set_memory_limit(Some(1));
    assert!(matches!(study.compile(&starved), Err(crate::CausalError::Resource { .. })));
    assert_eq!(pool.available().bytes(), 1_000_000, "a refused compile holds no lease");
}

#[test]
fn a_design_over_the_resident_budget_streams_through_the_closed_form_reduction() {
    let rows = 100_000_usize;
    let noise = |i: usize| ((i as f64 * 12.9898).sin() * 43758.5453).fract().abs() - 0.5;
    let z: Vec<f64> = (0..rows).map(|i| noise(i + 17) * 4.0).collect();
    let t: Vec<f64> = (0..rows).map(|i| f64::from(z[i] + 2.0 * noise(i + 31) > 0.0)).collect();
    let y: Vec<f64> =
        (0..rows).map(|i| 1.0 + 2.0 * t[i] + 0.5 * z[i] + 0.4 * noise(i + 104_729)).collect();
    let schema = CausalSchemaBuilder::new()
        .continuous("t")
        .treatment()
        .continuous("y")
        .outcome()
        .continuous("z")
        .context()
        .build()
        .unwrap();
    let data = TabularData::try_from_schema_f64(
        schema.clone(),
        [("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())],
    )
    .unwrap();
    let graph = Dag::from_named_edges(&schema, &[("t", "y"), ("z", "t"), ("z", "y")]).unwrap();
    let query =
        AverageEffectQuery::binary_ate(schema.id_of("t").unwrap(), schema.id_of("y").unwrap());
    let study = Study::tabular(data)
        .graph(graph)
        .query(query)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();

    let mut ctx = ExecutionContext::for_tests(1);
    ctx.set_memory_limit(Some(1_500_000));
    let plan = study.compile(&ctx).unwrap();
    assert!(plan.streams_design(), "the resident design cannot fit, so the plan streams");
    assert!(plan.compiled_linear_program().is_none(), "a streamed plan retains no design");
    assert!(plan.record.materializations.iter().any(|(name, how)| &**name == "design.matrix"
        && *how == antecedent_contract::BufferMaterialization::Chunked));
    // Layer five declares the scan the plan makes: one batch and the Gram of [1 | t | z]
    // held across chunks, not the three dense copies of a 100000-row design.
    let operators = plan.lowered().expect("compile attached a program").physical().operators();
    let scan = operators.last().expect("a scan operator");
    assert_eq!(scan.effect(), antecedent_core::PhysicalEffect::Scan);
    assert_eq!(scan.estimates().retained(), Some((9 + 6 + 2) * 8));
    assert_eq!(scan.estimates().scratch(), Some((3 + 2) * 8192 * 8));
    assert!(scan.estimates().scratch().unwrap() < 3 * rows as u64 * 3 * 8);

    let result = study.run(&ctx).unwrap();
    // The normal equations of [1 | t | z], solved by Cramer's rule.
    let n = rows as f64;
    let (st, sz) = (t.iter().sum::<f64>(), z.iter().sum::<f64>());
    let stt = t.iter().map(|v| v * v).sum::<f64>();
    let szz = z.iter().map(|v| v * v).sum::<f64>();
    let stz = t.iter().zip(&z).map(|(a, b)| a * b).sum::<f64>();
    let (sy, sty, szy) = (
        y.iter().sum::<f64>(),
        t.iter().zip(&y).map(|(a, b)| a * b).sum::<f64>(),
        z.iter().zip(&y).map(|(a, b)| a * b).sum::<f64>(),
    );
    let det = |m: [[f64; 3]; 3]| {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let gram = [[n, st, sz], [st, stt, stz], [sz, stz, szz]];
    let mut with_rhs = gram;
    for (row, value) in [sy, sty, szy].into_iter().enumerate() {
        with_rhs[row][1] = value;
    }
    let slope = det(with_rhs) / det(gram);
    assert!((result.effect() - slope).abs() < 1e-6 * slope.abs(), "{} vs {slope}", result.effect());

    // A cancelled run stops before it reads a batch.
    let cancelled = ExecutionContext::for_tests(1);
    cancelled.cancellation.cancel();
    assert!(study.run(&cancelled).is_err());
}
