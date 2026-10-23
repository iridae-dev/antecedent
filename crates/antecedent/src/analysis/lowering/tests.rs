//! Registry and guard checks for route lowering.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use antecedent_core::{AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, SemanticDigest};
use antecedent_data::TabularData;

use super::registry::REGISTERED;
use super::{GraphPosteriorCell, RouteCell};
use crate::planner::lowered::LoweringDepth;
use crate::{Dag, RefuteSuite, Study};

const FAMILIES: [&str; 5] =
    ["static_effect", "static_response", "structural", "temporal", "design_based"];

#[test]
fn every_route_cell_is_registered_once_to_a_family() {
    let mut names = BTreeSet::new();
    for (cell, family) in REGISTERED {
        assert!(FAMILIES.contains(family), "{} is registered to {family}", cell.as_str());
        assert!(names.insert(cell.as_str()), "{} is registered twice", cell.as_str());
    }
    assert_eq!(names.len(), 45, "route cell inventory drifted");
    for cell in GraphPosteriorCell::ALL {
        assert!(names.insert(RouteCell::GraphPosterior(cell).as_str()), "{}", cell.as_str());
    }
}

fn linear_study() -> Study {
    let schema = CausalSchemaBuilder::new()
        .continuous("t")
        .treatment()
        .continuous("y")
        .outcome()
        .build()
        .unwrap();
    let treatment = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    let outcome = [1.0, 3.1, 0.9, 2.9, 1.0, 3.0];
    let columns: [(&str, &[f64]); 2] = [("t", &treatment), ("y", &outcome)];
    let data = TabularData::try_from_schema_f64(schema.clone(), columns).unwrap();
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
fn compile_attaches_a_lowered_program_and_execution_stamps_it() {
    let ctx = ExecutionContext::for_tests(3);
    let study = linear_study();
    let plan = study.compile(&ctx).unwrap();
    let lowered = plan.lowered().expect("compile attaches a lowered program");
    assert_eq!(lowered.route(), "StaticAteDag");
    assert_eq!(lowered.depth(), LoweringDepth::Complete);
    let result = study.execute(&plan, &ctx).unwrap();
    assert_eq!(result.lowered_program_identity(), Some(lowered.identity()));
    let prepared = study.prepare(&ctx).unwrap();
    let estimated = prepared.estimate_retained(&ctx).unwrap();
    assert_eq!(estimated.lowered_program_identity(), prepared.plan().lowered_program_identity());
}

#[test]
fn dispatch_refuses_a_plan_without_a_lowered_program() {
    let ctx = ExecutionContext::for_tests(3);
    let study = linear_study();
    let mut plan = study.compile(&ctx).unwrap();
    plan.lowered = None;
    let refusal = study.execute(&plan, &ctx).unwrap_err();
    assert!(refusal.to_string().contains("route_not_lowered"), "{refusal}");
}

#[test]
fn compile_and_run_refuse_inside_a_lowered_dispatch() {
    let ctx =
        ExecutionContext::for_tests(3).enter_lowered_dispatch(SemanticDigest::from_bytes([7; 32]));
    let study = linear_study();
    let compiled = study.compile(&ctx).unwrap_err();
    assert!(compiled.to_string().contains("public_builder_reentry"), "{compiled}");
    assert!(study.run(&ctx).is_err(), "one-shot run inside a lowered dispatch must refuse");
}

#[test]
fn prepare_installs_the_identification_compile_retained() {
    use antecedent_core::execution_attempt::{Operation, observe_execution};
    let ctx = ExecutionContext::for_tests(3);
    let study = linear_study();
    let observed = observe_execution(|| study.prepare(&ctx));
    assert_eq!(
        observed.report.counts(Operation::Identification).attempted,
        1,
        "compile identifies once; preparation installs the retained products"
    );
    let prepared = observed.result.unwrap();
    let lowered = prepared.plan().lowered().unwrap();
    assert!(lowered.identification_retained());
    let program = prepared.plan().compiled_linear_program().unwrap();
    let crate::planner::lowered::ProblemTarget::Target(target) = lowered.scientific().target()
    else {
        panic!("the linear cell lowers its checked target");
    };
    assert_eq!(target.identity(), program.target().identity());
    assert_eq!(lowered.physical().operators().len(), program.physical_operators().len());
}

#[test]
fn quoted_design_width_counts_contrast_columns_the_schema_bound_misses() {
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
    let plan = prepared.plan();
    let design = plan.lowered().unwrap().physical().design();
    let schema_bound = plan.logical.record.query_variables.len() as u64;
    assert_eq!(schema_bound, 2, "the planner's bound counts the query's variables");
    assert_eq!(design.columns(), 5, "intercept, treatment and three contrast columns");
    assert_ne!(design.columns(), schema_bound);
    assert_eq!(design.source(), crate::planner::lowered::WidthSource::Quoted);
    assert!(!design.direct());
    assert_eq!(plan.choice.selected(), PhysicalAlternative::MaterializedDesign);
    let built = prepared.checked_linear_adjustment().unwrap().problem().design.ncols;
    assert_eq!(design.columns(), built as u64, "the quoted width is the built width");
}

#[test]
fn requote_replans_bytes_and_materialization_under_the_memory_limit() {
    use crate::planner::lowered::{DesignShape, WidthSource};
    use antecedent_contract::BufferMaterialization;
    let ctx = ExecutionContext::for_tests(3);
    let mut plan = linear_study().compile(&ctx).unwrap();
    let direct = DesignShape { rows: 10, columns: 3, source: WidthSource::Quoted, direct: true };
    plan.requote(direct, &ctx).unwrap();
    assert_eq!(plan.logical.design_columns_hint, 3);
    assert_eq!(plan.record.estimated_copy_bytes, Some(10 * 3 * 8));
    assert_eq!(plan.record.materializations[0].1, BufferMaterialization::Borrowed);
    let materialized = DesignShape { direct: false, ..direct };
    plan.requote(materialized, &ctx).unwrap();
    assert_eq!(plan.record.estimated_copy_bytes, Some(2 * 10 * 3 * 8));
    assert_eq!(plan.record.materializations[0].1, BufferMaterialization::CopiedContiguous);
    let mut bounded = ctx.clone();
    bounded.set_memory_limit(Some(1));
    assert!(matches!(plan.requote(direct, &bounded), Err(crate::CausalError::Resource { .. })));
}
