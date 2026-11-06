use super::*;
use antecedent_core::{
    DistributionAvailability, Environment, EvidenceCatalog, EvidenceKind, EvidenceRegime, RegimeId,
    RegimeKind, Value, VariableCoordinate,
};
use antecedent_expr::{DiscreteAxis, ExactDiscreteLaw, InterventionAssignment, LawTolerance};
use antecedent_graph::{Admg, DenseNodeId};
use antecedent_identify::ClassicalTransportQuery;
fn v(n: u32) -> VariableId {
    VariableId::from_raw(n)
}
fn fixture() -> PreparedStudy<TransportGridState> {
    let mut graph = Admg::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    graph.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let diagram = SelectionDiagram::try_new(graph, []).unwrap();
    let query = ClassicalTransportQuery {
        outcomes: Arc::from([v(1)]),
        treatments: Arc::from([v(0)]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    };
    let ctx = ExecutionContext::for_tests(0);
    let antecedent_identify::ClassicalTransportResult::Identified(proof) =
        antecedent_identify::identify_classical_transport(
            &diagram,
            &query,
            SidLimits::default(),
            &ctx,
        )
        .unwrap()
    else {
        panic!()
    };
    let env = |name| {
        Environment::try_new(
            name,
            [0, 1].map(|n| VariableCoordinate {
                variable: v(n),
                domain: VariableDomain::Binary,
                unit: None,
            }),
            [],
        )
        .unwrap()
    };
    let regime = EvidenceRegime::try_new(
        RegimeId::from_raw(0),
        RegimeKind::Experimental,
        EvidenceKind::Available,
        [v(0)],
        [],
        [v(1)],
        "source",
        DistributionAvailability::Joint,
    )
    .unwrap();
    let catalog =
        EvidenceCatalog::try_new([env("source"), env("target")], [regime], [], None).unwrap();
    let law = ExactDiscreteLaw::try_new(
        "source",
        RegimeId::from_raw(0),
        [InterventionAssignment { variable: v(0), value: Value::Int64(1) }],
        [DiscreteAxis { variable: v(1), values: Arc::from([Value::Int64(0), Value::Int64(1)]) }],
        [0.3, 0.7],
        "snapshot",
        LawTolerance::default(),
    )
    .unwrap();
    let data = ExactTransportData::try_new([law], 1000).unwrap();
    StudyBuilder::transport_grid(
        TransportGridQuery {
            diagram,
            functional: proof.bind_catalog(&catalog).unwrap(),
            at: vec![
                Assignment::from_pairs([(v(0), Value::Int64(0))]),
                Assignment::from_pairs([(v(0), Value::Int64(1))]),
            ],
            limits: ExactEvaluationLimits::default(),
        },
        TransportGridData::Exact(data),
        &ctx,
    )
    .unwrap()
}
fn sampled_grid_input() -> TransportGridData {
    let raw = StatisticalTransportInput {
        supplied: vec![],
        samples: vec![antecedent_estimate::RegimeSample {
            population: Arc::from("source"),
            regime: RegimeId::from_raw(0),
            snapshot_identity: Arc::from("snapshot"),
            interventions: Arc::from([InterventionAssignment {
                variable: v(0),
                value: Value::Int64(1),
            }]),
            columns: std::collections::BTreeMap::from([(
                v(1),
                vec![
                    Some(0.0),
                    Some(0.0),
                    Some(0.0),
                    Some(1.0),
                    Some(1.0),
                    Some(1.0),
                    Some(1.0),
                    Some(1.0),
                    Some(1.0),
                    Some(1.0),
                ],
            )]),
        }],
    };
    let options = EmpiricalTableOptions { bootstrap_replicates: 0, ..Default::default() };
    TransportGridData::Statistical(raw, options)
}

#[test]
fn deferred_grid_preserves_ordered_failures_identity_and_artifact_bytes() {
    let mut ctx = ExecutionContext::for_tests(0);
    ctx.cache_policy = antecedent_core::CachePolicy::enabled(None);
    let eager = fixture();
    let deferred = DeferredTransportGrid::compile(
        eager.state.query.clone(),
        TransportGridData::Exact(eager.state.data.clone()),
        &ctx,
    )
    .unwrap();
    assert!(!deferred.inspect().support.is_available());
    let expected = eager.estimate(&ctx).unwrap();
    let materialized = deferred.materialize(&ctx).unwrap();
    let actual = materialized.estimate(&ctx).unwrap();
    assert_eq!(actual.identity(), expected.identity());
    assert_eq!(actual.export().unwrap(), expected.export().unwrap());
    assert!(matches!(actual.points()[0], TransportGridPoint::Unavailable(_)));
    let branch = deferred.rebind(vec![eager.state.query.at[1].clone()], &ctx).unwrap();
    assert_eq!(deferred.query().at.len(), 2);
    assert_eq!(branch.query().at.len(), 1);
    let next = branch.materialize(&ctx).unwrap();
    assert!(Arc::ptr_eq(&materialized.state.retention[1], &next.state.retention[1]));
    assert_eq!(next.estimate(&ctx).unwrap().points().len(), 1);
}
#[test]
fn deferred_statistical_grid_fits_only_on_execution_and_honors_caller_cache_pool() {
    let mut ctx = ExecutionContext::for_tests(0);
    ctx.cache_policy = antecedent_core::CachePolicy::enabled(None);
    let eager = fixture();
    let input = sampled_grid_input();
    let deferred =
        DeferredTransportGrid::compile(eager.state.query.clone(), input.clone(), &ctx).unwrap();
    let first = deferred.materialize(&ctx).unwrap();
    let second = deferred.materialize(&ctx).unwrap();
    assert!(Arc::ptr_eq(&first.state.retention[1], &second.state.retention[1]));
    let old = StudyBuilder::transport_grid(eager.state.query.clone(), input, &ctx)
        .unwrap()
        .estimate(&ctx)
        .unwrap();
    let actual = first.estimate(&ctx).unwrap();
    assert!(matches!(actual.points()[0], TransportGridPoint::Unavailable(_)));
    assert_eq!(actual.identity(), old.identity());
    assert_eq!(actual.export().unwrap(), old.export().unwrap());
    let mut disabled = ctx.clone();
    disabled.cache_policy.enabled = false;
    let uncached = deferred.materialize(&disabled).unwrap();
    assert!(!Arc::ptr_eq(&first.state.retention[1], &uncached.state.retention[1]));
    let mut bounded = ctx.clone();
    bounded.cache_policy.max_bytes = Some(0);
    let bounded = deferred.materialize(&bounded).unwrap();
    assert!(!Arc::ptr_eq(&first.state.retention[1], &bounded.state.retention[1]));
    let mut tiny = ctx.clone();
    tiny.set_memory_limit(Some(1));
    assert!(deferred.materialize(&tiny).is_err());
    assert_eq!(tiny.admission.available().bytes(), 1);
    let cancelled = ctx.clone();
    cancelled.cancellation.cancel();
    assert!(deferred.materialize(&cancelled).is_err());
}
#[test]
fn deferred_grid_shares_raw_and_fitted_storage_across_targets_with_one_admission() {
    let baseline = fixture();
    let input = sampled_grid_input();
    let TransportGridData::Statistical(raw, options) = &input else {
        panic!("statistical fixture");
    };
    let raw_bytes = super::super::statistical::retained_input_bytes(raw).unwrap();
    let law_bytes =
        super::super::statistical::declared_provider_bytes(raw, &baseline.state.query.functional)
            .unwrap();
    // Empirical fitting has a separate transient-workspace preflight. Allow
    // that actual quote for the initial fit, then occupy all spare capacity
    // before branching: retained rows and laws must never be admitted twice.
    let (_, workspace_bytes) =
        antecedent_estimate::statistical_transport::check_statistical_resources(
            raw,
            &baseline.state.query.functional,
            options,
            &ExecutionContext::for_tests(0),
        )
        .unwrap();
    let retained_bytes = raw_bytes + law_bytes;
    let capacity = retained_bytes.max(u64::try_from(workspace_bytes).unwrap());
    let pool = antecedent_core::execution::AdmissionPool::fixed(1, capacity);
    let mut ctx = ExecutionContext::for_tests(0).with_admission(pool.clone());
    ctx.cache_policy = antecedent_core::CachePolicy::enabled(None);
    let deferred =
        DeferredTransportGrid::compile(baseline.state.query.clone(), input, &ctx).unwrap();
    let first = deferred.materialize(&ctx).unwrap();
    assert_eq!(pool.available().bytes(), capacity - retained_bytes);
    let headroom = ctx.acquire_admission(0, capacity - retained_bytes).unwrap();
    assert_eq!(pool.available().bytes(), 0);
    let target = vec![baseline.state.query.at[1].clone()];
    let branch = deferred.rebind(target.clone(), &ctx).unwrap();
    let second = branch.materialize(&ctx).unwrap();
    let third = first.rebind_assignments(target, &ctx).unwrap();
    let rows = |state: &TransportGridState| match state.input.as_deref().unwrap() {
        RetainedGridData::Statistical(raw, _) => Arc::clone(raw),
        RetainedGridData::Exact(_) => panic!("statistical fixture"),
    };
    assert!(Arc::ptr_eq(&rows(&first.state), &rows(&second.state)));
    assert!(Arc::ptr_eq(&rows(&first.state), &rows(&third.state)));
    assert!(Arc::ptr_eq(&first.state.retention[1], &second.state.retention[1]));
    assert!(std::ptr::eq(
        first.state.data.laws()[0].probabilities().as_ptr(),
        second.state.data.laws()[0].probabilities().as_ptr()
    ));
    assert_eq!(pool.available().bytes(), 0);
    drop((deferred, branch, first, second, third, headroom));
    assert_eq!(pool.available().bytes(), capacity);
}
#[test]
fn portable_grid_reasoning_is_the_in_memory_view() {
    let ctx = ExecutionContext::for_tests(0);
    let result = fixture().estimate(&ctx).unwrap();
    assert_eq!(
        result.wire.reasoning,
        crate::analysis::contract::reasoning_section(&result.reasoning()),
        "one author for the in-memory and the exported slots"
    );
    let support = result.wire.reasoning.support.value.as_ref().expect("support slot");
    assert_eq!(support.matrix_status, "stage_contract");
    assert_eq!(support.matrix_coordinate.as_deref(), Some("transport.response_grid"));
    assert_eq!(
        support.empirical,
        "1 executable; 1 unavailable (1 missing evidence, 0 support failure); \
         population positivity assumed"
    );
}

#[test]
fn grid_support_distinguishes_missing_evidence_from_a_support_failure() {
    use antecedent_core::SupportStatus;
    let ctx = ExecutionContext::for_tests(0);
    let result = fixture().estimate(&ctx).unwrap();
    // No law is supplied under do(X = 0): that coordinate lacks evidence; it is not a
    // support verdict about the evidence that was supplied.
    let TransportGridPoint::Unavailable(missing) = &result.points()[0] else {
        panic!("the unbound coordinate is retained as unavailable");
    };
    assert_eq!(missing.kind, "missing_evidence");
    assert_eq!(result.points()[0].support_status(), SupportStatus::MissingEvidence);
    assert_eq!(result.points()[1].support_status(), SupportStatus::Supported);
    let mut failed = missing.clone();
    failed.kind = "support_failure".into();
    assert_eq!(
        TransportGridPoint::Unavailable(failed).support_status(),
        SupportStatus::OutsideEmpiricalSupport
    );
    let support = result.wire.reasoning.support.value.as_ref().expect("support slot");
    assert_eq!(
        support.empirical,
        "1 executable; 1 unavailable (1 missing evidence, 0 support failure); \
         population positivity assumed"
    );
}

#[test]
fn named_grid_identity_survives_export_and_names_cannot_be_rebound() {
    let ctx = ExecutionContext::for_tests(0);
    let original = fixture().estimate(&ctx).unwrap();
    let names: Vec<String> = (0..original.wire.nodes.len()).map(|i| format!("v{i}")).collect();
    let named = original.clone().with_variable_names(&names).unwrap();
    assert_ne!(original.identity(), named.identity());
    let bytes = named.export_named(&names).unwrap();
    let (_, consumed) = PreparedStudy::<TransportGridState>::consume(
        &bytes,
        ExactEvaluationLimits::default(),
        &ctx,
    )
    .unwrap();
    assert_eq!(consumed.identity(), named.identity());
    assert_eq!(consumed.variable_names(), names);
    let mut renamed = names;
    renamed[0].push_str("_other");
    assert!(consumed.clone().with_variable_names(&renamed).is_err());
    assert!(consumed.export_named(&renamed).is_err());
}

#[test]
fn grid_artifact_with_the_undifferentiated_support_summary_still_loads() {
    // Artifacts written before the support summary split its unavailable count keep
    // loading, and re-export byte for byte.
    let ctx = ExecutionContext::for_tests(0);
    let result = fixture().estimate(&ctx).unwrap();
    let artifact = antecedent_io::transport_certificate::read_bounded_transport_artifact(
        &result.export().unwrap(),
        &ctx,
    )
    .unwrap();
    let mut wire: GridWire = antecedent_io::from_cbor(&artifact.sections[0].data).unwrap();
    let support = wire.reasoning.support.value.as_mut().expect("support slot");
    support.empirical = "1 executable; 1 unavailable; population positivity assumed".into();
    let id = antecedent_io::transport_grid_wire::grid_identity(&wire).unwrap();
    let bytes = wire.export(&id).unwrap();
    let (_, loaded) = PreparedStudy::<TransportGridState>::consume(
        &bytes,
        ExactEvaluationLimits::default(),
        &ctx,
    )
    .unwrap();
    assert_eq!(loaded.identity(), id);
    assert_eq!(loaded.export().unwrap(), bytes);
    assert_eq!(
        loaded.points()[0].support_status(),
        antecedent_core::SupportStatus::MissingEvidence
    );
    // Any other summary is still a mismatch.
    let support = wire.reasoning.support.value.as_mut().expect("support slot");
    support.empirical = "2 executable; 0 unavailable; population positivity assumed".into();
    let id = antecedent_io::transport_grid_wire::grid_identity(&wire).unwrap();
    assert!(
        PreparedStudy::<TransportGridState>::consume(
            &wire.export(&id).unwrap(),
            ExactEvaluationLimits::default(),
            &ctx
        )
        .is_err()
    );
}

#[test]
fn prepared_grid_inspection_shares_the_executed_identification_and_assumptions() {
    let ctx = ExecutionContext::for_tests(0);
    let study = fixture();
    let before = study.inspect();
    let after = study.estimate(&ctx).unwrap().reasoning();
    assert_eq!(before.identification, after.identification);
    assert_eq!(before.assumptions, after.assumptions);
    assert!(!before.support.is_available() && after.support.is_available());
    assert!(!before.uncertainty.is_available());
}

#[test]
fn native_partial_grid_checks_semantics_beyond_container_checksums() {
    let ctx = ExecutionContext::for_tests(0);
    let study = fixture();
    let result = study.estimate(&ctx).unwrap();
    let (_, loaded) = PreparedStudy::<TransportGridState>::consume(
        &result.export().unwrap(),
        ExactEvaluationLimits::default(),
        &ctx,
    )
    .unwrap();
    assert_eq!(result.identity(), loaded.identity());
    assert!(matches!(loaded.points()[0], TransportGridPoint::Unavailable(_)));
    assert!(result.wire.points.iter().all(|point| match point {
        GridPointWire::Exact(bytes) | GridPointWire::Statistical(bytes) => bytes.is_empty(),
        GridPointWire::Unavailable(_) => true,
    }));
    let artifact = antecedent_io::transport_certificate::read_bounded_transport_artifact(
        &result.export().unwrap(),
        &ctx,
    )
    .unwrap();
    let encoded: GridWire = antecedent_io::from_cbor(&artifact.sections[0].data).unwrap();
    let mut legacy = encoded.clone();
    legacy.version = 1;
    legacy.point_evidence.clear();
    legacy.required_features =
        vec!["checked_transport_proof_v1".into(), "retained_transport_grid_v1".into()];
    let legacy_bytes = legacy.export(&digest(IdentityDomain::Execution, &legacy).unwrap()).unwrap();
    let (_, legacy_result) = PreparedStudy::<TransportGridState>::consume(
        &legacy_bytes,
        ExactEvaluationLimits::default(),
        &ctx,
    )
    .unwrap();
    assert_eq!(legacy_result.export().unwrap(), legacy_bytes);
    let reject = |mutated: GridWire| {
        let id = antecedent_io::transport_grid_wire::grid_identity(&mutated).unwrap();
        assert!(
            PreparedStudy::<TransportGridState>::consume(
                &mutated.export(&id).unwrap(),
                ExactEvaluationLimits::default(),
                &ctx
            )
            .is_err()
        );
    };
    let mut wire = encoded.clone();
    wire.required_features.push("future_required_feature".into());
    reject(wire);
    let mut wire = encoded.clone();
    wire.at[0][0].1 = ValueWire::from_value(&Value::Int64(1));
    reject(wire);
    let mut wire = encoded.clone();
    wire.points.swap(0, 1);
    reject(wire);
    let mut wire = encoded.clone();
    wire.point_evidence[1] = "changed".into();
    reject(wire);
    let mut wire = encoded.clone();
    wire.laws[0].axes[0].0 = 0;
    reject(wire);
    let mut wire = encoded.clone();
    wire.proof.proof.target = "substituted".into();
    reject(wire);
    let mut wire = encoded.clone();
    wire.catalog.regimes[0].population = "wrong_population".into();
    reject(wire);
    let mut wire = encoded;
    if let GridPointWire::Unavailable(failure) = &mut wire.points[0] {
        failure.kind = "support_failure".into();
    }
    reject(wire);
    let mut tiny = ctx.clone();
    tiny.set_memory_limit(Some(1));
    assert!(study.estimate(&tiny).is_err());
}
