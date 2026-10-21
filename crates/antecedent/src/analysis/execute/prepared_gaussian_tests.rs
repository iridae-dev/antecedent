// SPDX-License-Identifier: MIT OR Apache-2.0
//! Public prepared Gaussian execution regression tests.
use crate::{RefuteSuite, Study};
use antecedent_core::{AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, TargetForm};
use antecedent_data::TabularData;
use antecedent_graph::Dag;
use antecedent_stats::fit_counts::count_least_squares_solves;

fn contrast_study() -> Study {
    let schema = CausalSchemaBuilder::new()
        .continuous("t")
        .treatment()
        .continuous("y")
        .outcome()
        .build()
        .unwrap();
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
    let study = contrast_study();
    let (compilation, preparation_solves) = count_least_squares_solves(|| {
        antecedent_core::execution_attempt::observe_execution(|| study.prepare(&ctx))
    });
    assert_eq!(preparation_solves, 0, "compilation must not fit");
    let plan = study.compile(&ctx).unwrap();
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
    let program = prepared.compiled_linear_program().unwrap();
    assert!(matches!(program.target().form(), TargetForm::Contrast(_)));
    assert_eq!(program.identification().world_obligations().len(), 2);
    assert_eq!(
        program.identification().assumptions(),
        prepared.checked_linear_adjustment().unwrap().required_assumptions()
    );
    assert_eq!(program.operators().len(), 3);
    assert_eq!(program.physical_operators().len(), 3);
    drop(study);
    let (first, solves) = count_least_squares_solves(|| prepared.estimate_retained(&ctx).unwrap());
    assert!(solves > 0);
    let facts = first.performance.native_execution.as_ref().unwrap();
    assert!(facts.is_native());
    assert_eq!(facts.observed_model_fits(), Some(1));
    assert!(facts.peak_admitted_bytes().unwrap() > 0);
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
    bounded.memory.hard_limit_bytes = Some(1);
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
