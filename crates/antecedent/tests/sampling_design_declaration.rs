//! A sampling design declared once on the study is held by every estimator that resamples.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{CausalError, EstimatorId, RefuteSuite, Study};
use antecedent_core::{DeclaredSamplingDesign, ExecutionContext};
use antecedent_data::{DataError, TableView};
use antecedent_estimate::LinearAdjustmentAte;

#[path = "fixtures_common/mod.rs"]
mod common;

use common::fixtures::confounded_scm;

const ROWS: usize = 240;

fn is_design_refusal(error: &CausalError) -> bool {
    matches!(
        error,
        CausalError::Data(DataError::SamplingDesign(_))
            | CausalError::Estimate(antecedent_estimate::EstimationError::Data(
                DataError::SamplingDesign(_)
            ))
    )
}

#[test]
fn a_row_bootstrap_is_refused_under_a_declared_trajectory_design() {
    let ctx = ExecutionContext::for_tests(11);
    let (data, dag, query) = confounded_scm(ROWS, 11);
    let outcome = Study::tabular(data)
        .graph(dag)
        .query(query)
        .estimator(EstimatorId::LinearAdjustmentAte)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(20)
        .sampling_design(DeclaredSamplingDesign::IndependentSubjectTrajectories)
        .build()
        .and_then(|study| study.run(&ctx));
    let error = outcome.expect_err("a row bootstrap must not run under a trajectory design");
    assert!(is_design_refusal(&error), "unexpected refusal: {error:?}");
    let message = error.to_string();
    assert!(message.contains("independent_subject_trajectories"), "{message}");
    assert!(message.contains("row_iid"), "{message}");
}

#[test]
fn a_request_without_resampling_is_accepted_under_the_declared_design() {
    let ctx = ExecutionContext::for_tests(11);
    let (data, dag, query) = confounded_scm(ROWS, 11);
    let result = Study::tabular(data)
        .graph(dag)
        .query(query)
        .estimator(EstimatorId::LinearAdjustmentAte)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .sampling_design(DeclaredSamplingDesign::IndependentSubjectTrajectories)
        .build()
        .and_then(|study| study.run(&ctx));
    assert!(result.is_ok(), "{:?}", result.err());
}

#[test]
fn a_whole_cluster_bootstrap_is_accepted_under_a_clustered_panel() {
    let ctx = ExecutionContext::for_tests(11);
    let (data, dag, query) = confounded_scm(ROWS, 11);
    let labels: Vec<u32> = (0..data.row_count()).map(|row| (row / 4) as u32).collect();
    let estimator =
        LinearAdjustmentAte::new().with_cluster_ids(labels).with_bootstrap_replicates(20);
    let result = Study::tabular(data)
        .graph(dag)
        .query(query)
        .estimator(estimator)
        .refute(RefuteSuite::None)
        .sampling_design(DeclaredSamplingDesign::ClusteredPanel)
        .build()
        .and_then(|study| study.run(&ctx));
    assert!(result.is_ok(), "{:?}", result.err());
}

#[test]
fn an_estimator_that_declares_another_design_conflicts_with_the_study() {
    let (data, dag, query) = confounded_scm(ROWS, 11);
    let estimator = LinearAdjustmentAte::new()
        .with_bootstrap_replicates(0)
        .with_sampling_design(DeclaredSamplingDesign::ClusteredPanel);
    let error = Study::tabular(data)
        .graph(dag)
        .query(query)
        .estimator(estimator)
        .refute(RefuteSuite::None)
        .sampling_design(DeclaredSamplingDesign::DependentTimeSeries)
        .build()
        .expect_err("two different designs for one study");
    assert!(
        matches!(error, CausalError::Conflict { what: "sampling_design", .. }),
        "unexpected error: {error:?}"
    );
}

fn run_with_labels(
    design: DeclaredSamplingDesign,
    labels: Option<Vec<u32>>,
) -> Result<(), CausalError> {
    let ctx = ExecutionContext::for_tests(11);
    let (data, dag, query) = confounded_scm(ROWS, 11);
    let mut estimator = LinearAdjustmentAte::new().with_bootstrap_replicates(20);
    if let Some(labels) = labels {
        estimator = estimator.with_cluster_ids(labels);
    }
    Study::tabular(data)
        .graph(dag)
        .query(query)
        .estimator(estimator)
        .refute(RefuteSuite::None)
        .sampling_design(design)
        .build()
        .and_then(|study| study.run(&ctx))
        .map(|_| ())
}

fn labels_of(size: usize) -> Vec<u32> {
    (0..ROWS).map(|row| (row / size) as u32).collect()
}

#[test]
fn subject_labels_let_a_trajectory_design_resample_whole_subjects() {
    let result =
        run_with_labels(DeclaredSamplingDesign::IndependentSubjectTrajectories, Some(labels_of(6)));
    assert!(result.is_ok(), "{:?}", result.err());
}

#[test]
fn a_panel_without_cluster_labels_refuses_the_row_bootstrap() {
    let error = run_with_labels(DeclaredSamplingDesign::ClusteredPanel, None)
        .expect_err("a panel needs its clusters to resample");
    assert!(is_design_refusal(&error), "unexpected refusal: {error:?}");
    let message = error.to_string();
    assert!(message.contains("clustered_panel") && message.contains("row_iid"), "{message}");
}

#[test]
fn a_dependent_series_refuses_a_whole_cluster_draw() {
    let error = run_with_labels(DeclaredSamplingDesign::DependentTimeSeries, Some(labels_of(4)))
        .expect_err("a series is resampled in contiguous blocks, not by cluster");
    assert!(is_design_refusal(&error), "unexpected refusal: {error:?}");
    let message = error.to_string();
    assert!(
        message.contains("dependent_time_series") && message.contains("whole_cluster"),
        "{message}"
    );
}

#[test]
fn the_same_design_on_the_estimator_and_the_study_is_not_a_conflict() {
    let (data, dag, query) = confounded_scm(ROWS, 11);
    let estimator = LinearAdjustmentAte::new()
        .with_bootstrap_replicates(0)
        .with_sampling_design(DeclaredSamplingDesign::ClusteredPanel);
    let built = Study::tabular(data)
        .graph(dag)
        .query(query)
        .estimator(estimator)
        .refute(RefuteSuite::None)
        .sampling_design(DeclaredSamplingDesign::ClusteredPanel)
        .build();
    assert!(built.is_ok(), "{:?}", built.err());
}
