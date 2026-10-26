//! Numerical workspace admission does not alter scientific results.
use antecedent_core::ExecutionContext;
use antecedent_design::decision_reliability::{
    CandidateCoupling, CandidateDraws, CandidateOrigin, DecisionDraws, DrawSegment, RankingRequest,
    ReliabilityError, ReliabilityRequest, SourceParadigm, StreamId, StreamPurpose, ValidationPlan,
    WithinStreamDependence, assess, rank,
};

fn carrier() -> DecisionDraws {
    let segment = DrawSegment::whole(StreamId::new(11, StreamPurpose::Selection, 0, "exact"), 256);
    DecisionDraws {
        utility_unit: "utility".into(),
        producer: "exact".into(),
        source: SourceParadigm::KnownLaw,
        coupling: CandidateCoupling::SharedSimulation,
        within_stream: WithinStreamDependence::Independent,
        weights: None,
        repeated_evaluations: 0,
        origin: CandidateOrigin::Enumerated,
        search_record: None,
        candidates: [("A", [3., 3., 2., 2.]), ("B", [2., 1., 3., 2.]), ("C", [1., 2., 1., 2.])]
            .into_iter()
            .map(|(id, values)| CandidateDraws {
                id: id.into(),
                utilities: values.repeat(64),
                segment: segment.clone(),
            })
            .collect(),
    }
}

#[test]
fn rank_table_refusal_releases_all_preceding_workspace_leases() {
    let data = carrier();
    let mut limited = ExecutionContext::for_tests(11);
    // Canonical order and borrowed-column headers fit, but the two
    // 3-by-256 u32 rank tables cannot fit this remaining pool.
    limited.set_memory_limit(Some(128));
    assert!(matches!(
        rank(&data, &RankingRequest::default(), &limited),
        Err(ReliabilityError::Admission { .. })
    ));
    assert_eq!(limited.admission.available().bytes(), 128);
    limited.set_memory_limit(Some(100_000));
    let funded = rank(&data, &RankingRequest::default(), &limited).unwrap();
    let reference =
        rank(&data, &RankingRequest::default(), &ExecutionContext::for_tests(11)).unwrap();
    assert_eq!(funded, reference);
    assert_eq!(limited.admission.available().bytes(), 100_000);
}

#[test]
fn split_assessment_admits_copied_rows_and_preserves_the_report_when_funded() {
    let data = carrier();
    let request = ReliabilityRequest {
        validation: ValidationPlan::SplitSelectionDraws { selection_fraction: 0.5, gap: 0 },
        ..ReliabilityRequest::expected_utility()
    };
    let mut ctx = ExecutionContext::for_tests(11);
    ctx.set_memory_limit(Some(0));
    assert!(matches!(assess(&data, None, &request, &ctx), Err(ReliabilityError::Admission { .. })));
    assert_eq!(ctx.admission.available().bytes(), 0);
    ctx.set_memory_limit(Some(1_000_000));
    let funded = assess(&data, None, &request, &ctx).unwrap();
    let reference = assess(&data, None, &request, &ExecutionContext::for_tests(11)).unwrap();
    assert_eq!(funded, reference);
    assert_eq!(ctx.admission.available().bytes(), 1_000_000);
}

#[test]
fn cancelled_reliability_work_does_not_allocate_or_publish_a_report() {
    let data = carrier();
    let mut ctx = ExecutionContext::for_tests(11);
    ctx.set_memory_limit(Some(100_000));
    ctx.cancellation.cancel();
    assert!(matches!(
        rank(&data, &RankingRequest::default(), &ctx),
        Err(ReliabilityError::Cancelled)
    ));
    assert!(matches!(
        assess(&data, None, &ReliabilityRequest::expected_utility(), &ctx),
        Err(ReliabilityError::Cancelled)
    ));
    assert_eq!(ctx.admission.available().bytes(), 100_000);
}
