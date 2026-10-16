//! Consumer side of a repeated-measures derived report: the decision
//! reliability carrier and its coverage design.
//!
//! A repeated-measures report is a plain derived statement (`derived_unmeasured`
//! with an exact closed form and a simulated law). No producer adapter maps it
//! to the decision-reliability carrier, whose producers are cohort-decision
//! and sequential-evaluation outputs, and no coverage route names it, so the
//! cell is REFUSED by absence of a route: this asserts both facts.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::adaptive_procedure_report::{
    RepeatedMeasuresProcedure, RepeatedMeasuresReportBody, RepeatedMeasuresReportRequest,
};
use antecedent::analysis::decision_reliability::decision_reliability_coverage_design;
use antecedent_core::ExecutionContext;
use antecedent_stats::adaptive_inference::{
    AnalysisEstimator, DERIVED_UNMEASURED, FinalAnalysis, OutcomeStructure,
    RepeatedMeasuresDeclaration, StoppingRule, UnitAssignment, UnitSummary, WithinUnitCorrelation,
};

#[test]
fn a_repeated_measures_report_has_no_decision_reliability_route() {
    let procedure = RepeatedMeasuresProcedure::declare(RepeatedMeasuresDeclaration {
        arms: 2,
        assignment: Some(UnitAssignment::Fixed { probabilities: vec![0.5, 0.5] }),
        enrolment: Some(vec![4]),
        lag_looks: 0,
        structure: Some(OutcomeStructure::RepeatedMeasures {
            measurements: 3,
            variance: 1.0,
            correlation: Some(WithinUnitCorrelation::CompoundSymmetric { rho: 0.3 }),
        }),
        analysis: Some(FinalAnalysis {
            estimator: AnalysisEstimator::ClusterRobust { summary: UnitSummary::Mean },
            plus: 1,
            minus: 0,
        }),
        stopping: Some(StoppingRule::NoStopping),
    })
    .expect("valid");
    let report = procedure
        .report(
            &RepeatedMeasuresReportRequest::ClosedForm { arm_means: vec![0.0, 0.2] },
            &ExecutionContext::for_tests(1),
        )
        .expect("closed form");
    assert_eq!(report.calibration, DERIVED_UNMEASURED);
    assert!(matches!(report.body, RepeatedMeasuresReportBody::ClosedForm { .. }));
    for cell in decision_reliability_coverage_design() {
        let text = format!("{} {}", cell.route, cell.claim).to_lowercase();
        assert!(
            !text.contains("repeated_measures") && !text.contains("repeated measures"),
            "no coverage route may name a repeated-measures report: {}",
            cell.route
        );
    }
}
