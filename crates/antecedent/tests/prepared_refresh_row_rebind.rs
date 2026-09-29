//! A refreshed prepared handle must publish the rows it actually executed on.
//!
//! Checked linear adjustment and checked AIPW retain a row-bound preparation.
//! Refreshing to a table of a different length has to rebind that preparation,
//! otherwise the exported contract either fails to rehash or silently reports
//! the prepare-time complete-case count.

use antecedent::{EstimatorId, RefuteSuite, Study, StudyBuilder};
use antecedent_core::ExecutionContext;
use antecedent_io::{AnalysisResultConsumption, consume_analysis_result};

mod common;
use common::fixtures::confounded_scm;

fn refresh_and_export(builder: StudyBuilder, rows: usize) -> AnalysisResultConsumption {
    let context = ExecutionContext::for_tests(73);
    let (replacement, _, _) = confounded_scm(rows, 91);
    let mut prepared = builder.build().unwrap().prepare(&context).unwrap();
    let refreshed = prepared.refresh(replacement, &context).unwrap();
    let artifact = prepared.encode_contracted_result(&refreshed, "refresh", &context).unwrap();
    let consumed = consume_analysis_result(&artifact).unwrap();
    assert!(
        consumed.acceptance.accepts_as_verified_program(),
        "rows={rows}: unresolved={:?}",
        consumed.acceptance.unresolved
    );
    consumed
}

#[test]
fn refreshed_linear_adjustment_publishes_the_refreshed_rows() {
    let (data, dag, query) = confounded_scm(512, 73);
    for rows in [300_usize, 900] {
        let builder = Study::tabular(data.clone())
            .graph(dag.clone())
            .query(query.clone())
            .estimator(EstimatorId::LinearAdjustmentAte)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0);
        let consumed = refresh_and_export(builder, rows);
        let published = consumed
            .contract
            .as_ref()
            .and_then(|contract| contract.program.as_ref())
            .and_then(|program| program.checked_linear_adjustment_lowering.as_ref())
            .map(|lowering| lowering.complete_case_rows);
        assert_eq!(published, Some(rows as u64));
    }
}

#[test]
fn refreshed_trimmed_aipw_exports_a_verified_row_binding() {
    let (data, dag, query) = confounded_scm(512, 73);
    let overlap = antecedent_estimate::OverlapPolicy::RequireDiagnostics {
        clip: Some(0.01),
        trim: Some(0.02),
    };
    for rows in [300_usize, 900] {
        let fitter =
            antecedent_estimate::AipwAte::new().with_bootstrap_replicates(0).with_overlap(overlap);
        let builder = Study::tabular(data.clone())
            .graph(dag.clone())
            .query(query.clone())
            .estimator(fitter)
            .refute(RefuteSuite::Cheap);
        refresh_and_export(builder, rows);
    }
}
