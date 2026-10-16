//! Facade integration of decision reliability: a hand-enumerated four-state law,
//! separate uncertainty components, closed publishing routes and an artifact a consumer
//! recomputes.
//!
//! Hand oracle (four equally likely states, per-state utilities):
//! * A = (3, 3, 2, 2), B = (2, 1, 3, 2), C = (1, 2, 1, 2): expected utilities 5/2, 2, 3/2.
//! * The last state is a three-way tie, so P(best) is A 7/12, B 1/3, C 1/12.
//! * A is selected; with three candidates considered the in-sample value is labelled.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::decision_reliability::{
    CandidateCoupling, CandidateDraws, CandidateOrigin, DecisionDraws,
    DecisionReliabilityExpectation, DrawSegment, PreparedDecisionReliability, PublishedRoute,
    RankingRequest, ReliabilityRequest, SelectionUncertainty, SourceParadigm, SourceUncertainty,
    StreamId, StreamPurpose, WithinStreamDependence, consume_decision_reliability_artifact,
};
use antecedent_core::ExecutionContext;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

fn four_state_law() -> DecisionDraws {
    let segment = DrawSegment::whole(StreamId::new(1, StreamPurpose::Selection, 0, "states"), 4);
    let candidate = |id: &str, utilities: [f64; 4]| CandidateDraws {
        id: id.to_owned(),
        utilities: utilities.to_vec(),
        segment: segment.clone(),
    };
    DecisionDraws {
        utility_unit: "utility".into(),
        producer: "enumerated_finite_law".into(),
        source: SourceParadigm::KnownLaw,
        coupling: CandidateCoupling::SharedSimulation,
        within_stream: WithinStreamDependence::Independent,
        weights: None,
        repeated_evaluations: 0,
        origin: CandidateOrigin::Enumerated,
        search_record: None,
        candidates: vec![
            candidate("A", [3.0, 3.0, 2.0, 2.0]),
            candidate("B", [2.0, 1.0, 3.0, 2.0]),
            candidate("C", [1.0, 2.0, 1.0, 2.0]),
        ],
    }
}

fn request() -> ReliabilityRequest {
    ReliabilityRequest {
        ranking: Some(RankingRequest::default()),
        ..ReliabilityRequest::expected_utility()
    }
}

#[test]
fn the_facade_reports_the_enumerated_truth_with_separate_components() {
    let prepared =
        PreparedDecisionReliability::assess(four_state_law(), None, request(), &ctx()).unwrap();
    let report = prepared.report();
    assert_eq!(report.decided_id, "A");
    // Component two: numerical error of the decision's value.
    assert!((report.monte_carlo_error.decision.value - 2.5).abs() < 1e-12);
    assert_eq!(report.monte_carlo_error.differences.len(), 2);
    // Component one: the law is declared known, so there is no source spread.
    assert_eq!(report.source_uncertainty, SourceUncertainty::KnownLaw);
    // Component three: a selection among three candidates, in-sample and labelled.
    let SelectionUncertainty::Selected(selection) = &report.selection_uncertainty else {
        panic!("a selection was made");
    };
    assert_eq!(selection.n_considered, 3);
    assert!((selection.in_sample.value - 2.5).abs() < 1e-12);
    let ranking = report.ranking.as_ref().unwrap();
    let best =
        |id: &str| ranking.statements.iter().find(|s| s.id == id).unwrap().probability_best.value;
    assert!((best("A") - 7.0 / 12.0).abs() < 1e-12);
    assert!((best("B") - 1.0 / 3.0).abs() < 1e-12);
    assert!((best("C") - 1.0 / 12.0).abs() < 1e-12);
    let inspection = prepared.inspect();
    assert_eq!(inspection.standing, "uncalibrated_point_diagnostic");
    assert_eq!(inspection.closed_routes.len(), PublishedRoute::ALL.len());
}

#[test]
fn publishing_a_probability_or_a_combined_number_is_refused_with_registered_codes() {
    let prepared =
        PreparedDecisionReliability::assess(four_state_law(), None, request(), &ctx()).unwrap();
    for route in PublishedRoute::ALL {
        let refusal = prepared.publish(route).unwrap_err();
        assert_eq!(refusal.code, "cell_not_licensed");
        assert_eq!(refusal.detail, "decision_reliability.cell_not_licensed");
        assert!(refusal.message.contains("coverage record"));
    }
    let refusal = prepared.combined_uncertainty().unwrap_err();
    assert_eq!(refusal.code, "uncertainty_components_not_combined");
    // An adaptive set without its search record refuses at the facade.
    let mut adaptive = four_state_law();
    adaptive.origin = CandidateOrigin::AdaptivelyGenerated;
    let refusal =
        PreparedDecisionReliability::assess(adaptive, None, request(), &ctx()).unwrap_err();
    assert_eq!(refusal.code, "candidate_search_record_required");
}

#[test]
fn the_exported_artifact_is_recomputed_and_mutations_are_refused() {
    let prepared =
        PreparedDecisionReliability::assess(four_state_law(), None, request(), &ctx()).unwrap();
    let bytes = prepared.export("decision-reliability-facade", &ctx()).unwrap();
    let expected = DecisionReliabilityExpectation {
        draws_digest: Some(four_state_law().digest_hex()),
        decided_id: Some("A".into()),
        ..DecisionReliabilityExpectation::default()
    };
    let consumed = consume_decision_reliability_artifact(&bytes, &expected, &ctx()).unwrap();
    assert_eq!(consumed.report(), prepared.report());
    // A consumer that retained another decision refuses.
    let wrong = DecisionReliabilityExpectation {
        decided_id: Some("B".into()),
        ..DecisionReliabilityExpectation::default()
    };
    assert!(consume_decision_reliability_artifact(&bytes, &wrong, &ctx()).is_err());
    // A damaged container is refused.
    let mut damaged = bytes;
    let middle = damaged.len() / 2;
    damaged[middle] ^= 0xFF;
    assert!(
        consume_decision_reliability_artifact(
            &damaged,
            &DecisionReliabilityExpectation::default(),
            &ctx()
        )
        .is_err()
    );
}
