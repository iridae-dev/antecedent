//! Public checked point composition: explicit world targets, no invented uncertainty.
use antecedent_contract::reasoning::{ReasoningView, SlotAvailability};
use antecedent_contract::{
    CheckedWorldPointLimits, ClaimDomains, ClaimEnvelope, ClaimKind, DomainStatus,
    WorldPointParent, compose_checked_world_points,
};
use antecedent_core::{
    AssumptionSet, Conditioning, ContractIdentities, ExecutionContext, GraphFamily,
    InferenceSelection, Intervention, InterventionContrast, InterventionWorld, ModelAssumptions,
    ObservationUnit, OutcomeQuantity, QuantityUnits, ScientificTarget, ScientificWorldId,
    SemanticDigest, TargetForm, TargetPopulation, TimeFrame, Value, VariableId, WorldAction,
    WorldBinding, WorldCollection,
};
use std::sync::Arc;

fn limits() -> CheckedWorldPointLimits {
    CheckedWorldPointLimits {
        max_target_work: 100_000,
        max_target_payload_bytes: 1_000_000,
        max_target_depth: 64,
        max_memory_bytes: 64_000_000,
        max_terms: 4,
        max_comparisons: 16,
        max_world_actions: 8,
        max_assumption_records: 8,
        max_evidence_references: 16,
        max_evidence_bytes: 1024,
    }
}

fn d(byte: u8) -> SemanticDigest {
    SemanticDigest::from_bytes([byte; 32])
}

fn fixture() -> (ScientificTarget, ScientificTarget, ScientificTarget, ClaimEnvelope, ClaimEnvelope)
{
    let v = VariableId::from_raw;
    let id = ScientificWorldId::from_raw;
    let binding = WorldBinding::new(
        TargetPopulation::AllObserved,
        ObservationUnit::new("subject").unwrap(),
        OutcomeQuantity::mean(v(1), QuantityUnits::dimensionless()).unwrap(),
        Conditioning::unconditional(),
    )
    .unwrap();
    let world = |i, level| {
        InterventionWorld::new(
            id(i),
            WorldAction::at_decision([Intervention::set(v(0), Value::f64(level))]).unwrap(),
            i == 0,
            binding.clone(),
            TimeFrame::static_decision(),
        )
        .unwrap()
    };
    let w0 = world(0, 0.0);
    let w1 = world(1, 1.0);
    let assumptions = ModelAssumptions::new(GraphFamily::Dag, AssumptionSet::new());
    let first = ScientificTarget::new(
        TargetForm::World(w0.clone()),
        assumptions.clone(),
        InferenceSelection::Unspecified,
    );
    let second = ScientificTarget::new(
        TargetForm::World(w1.clone()),
        assumptions.clone(),
        InferenceSelection::Unspecified,
    );
    let contrast = InterventionContrast::linear(
        WorldCollection::new([w0, w1]).unwrap(),
        [(id(0), -1.0), (id(1), 1.0)],
        id(0),
    )
    .unwrap();
    let target = ScientificTarget::new(
        TargetForm::Contrast(contrast),
        assumptions,
        InferenceSelection::Unspecified,
    );
    let claim = |id, target: &ScientificTarget, value| {
        ClaimEnvelope::new(
            d(id),
            ContractIdentities::new(
                target.identity(),
                d(2),
                Some(d(3)),
                Some(d(4)),
                d(5),
                d(6),
                d(7),
            ),
            ClaimKind::Point,
            Some(value),
            Some(Arc::from("1")),
            ReasoningView::new(
                SlotAvailability::unavailable("not_verified"),
                SlotAvailability::unavailable("not_verified"),
                SlotAvailability::unavailable("not_propagated"),
                SlotAvailability::unavailable("declared"),
            ),
            ClaimDomains::new(DomainStatus::Unknown, DomainStatus::Unknown, DomainStatus::Unknown),
            None,
            [Arc::from("alignment:joint_draws:snapshot"), Arc::from("data:study")],
        )
    };
    let p0 = claim(10, &first, 2.0);
    let p1 = claim(11, &second, 5.0);
    (target, first, second, p0, p1)
}

#[test]
fn checked_world_points_compute_the_declared_target_and_refuse_forgery() {
    let (target, first, second, p0, p1) = fixture();
    let inputs = [
        WorldPointParent { claim: &p1, target: &second },
        WorldPointParent { claim: &p0, target: &first },
    ];
    let result =
        compose_checked_world_points(&target, &inputs, &limits(), &ExecutionContext::for_tests(17))
            .unwrap();
    assert_eq!(result.value(), 3.0);
    assert_eq!(result.target().identity(), target.identity());
    assert_eq!(result.parent_ids().len(), 2);
    assert!(
        result.parent_ids().contains(&p0.claim_id) && result.parent_ids().contains(&p1.claim_id)
    );
    assert!(result.evidence().iter().any(|v| &**v == "data:study"));
    assert!(result.limitations().contains(&"uncertainty_not_propagated"));
    assert!(result.limitations().contains(&"calibration_not_transferred"));
    let mut too_small = limits();
    too_small.max_comparisons = 3;
    assert!(
        compose_checked_world_points(
            &target,
            &inputs,
            &too_small,
            &ExecutionContext::for_tests(17)
        )
        .is_err()
    );
    let mut no_evidence = limits();
    no_evidence.max_evidence_bytes = 0;
    assert!(
        compose_checked_world_points(
            &target,
            &inputs,
            &no_evidence,
            &ExecutionContext::for_tests(17)
        )
        .is_err()
    );
    let mut altered = p1.clone();
    altered.value = Some(6.0);
    let updated = compose_checked_world_points(
        &target,
        &[
            WorldPointParent { claim: &p0, target: &first },
            WorldPointParent { claim: &altered, target: &second },
        ],
        &limits(),
        &ExecutionContext::for_tests(17),
    )
    .unwrap();
    assert_eq!(updated.value(), 4.0);
    assert_ne!(updated.proof_identity(), result.proof_identity());
    let TargetForm::Contrast(form) = target.form() else { panic!("contrast") };
    let aggregate = InterventionContrast::linear(
        form.worlds().clone(),
        [(ScientificWorldId::from_raw(0), 0.25), (ScientificWorldId::from_raw(1), 0.75)],
        ScientificWorldId::from_raw(0),
    )
    .unwrap();
    let aggregate_target = ScientificTarget::new(
        TargetForm::Contrast(aggregate),
        target.assumptions().clone(),
        target.inference(),
    );
    let weighted = compose_checked_world_points(
        &aggregate_target,
        &inputs,
        &limits(),
        &ExecutionContext::for_tests(17),
    )
    .unwrap();
    assert_eq!(weighted.value(), 4.25);
    assert_ne!(weighted.proof_identity(), result.proof_identity());
    let mut changed0 = p0.clone();
    let mut changed1 = p1.clone();
    changed0.reasoning.assumptions = SlotAvailability::unavailable("different_premise");
    changed1.reasoning.assumptions = SlotAvailability::unavailable("different_premise");
    let changed = compose_checked_world_points(
        &target,
        &[
            WorldPointParent { claim: &changed0, target: &first },
            WorldPointParent { claim: &changed1, target: &second },
        ],
        &limits(),
        &ExecutionContext::for_tests(17),
    )
    .unwrap();
    assert_ne!(changed.proof_identity(), result.proof_identity());
    let mut changed_assumptions = p1.clone();
    changed_assumptions.reasoning.assumptions = SlotAvailability::unavailable("other_assumption");
    assert!(
        compose_checked_world_points(
            &target,
            &[
                WorldPointParent { claim: &p0, target: &first },
                WorldPointParent { claim: &changed_assumptions, target: &second },
            ],
            &limits(),
            &ExecutionContext::for_tests(17)
        )
        .is_err()
    );
    let mut wrong0 = p0.clone();
    wrong0.outcome_units = Some(Arc::from("wrong"));
    let mut wrong1 = p1.clone();
    wrong1.outcome_units = Some(Arc::from("wrong"));
    assert!(
        compose_checked_world_points(
            &target,
            &[
                WorldPointParent { claim: &wrong0, target: &first },
                WorldPointParent { claim: &wrong1, target: &second },
            ],
            &limits(),
            &ExecutionContext::for_tests(17)
        )
        .is_err()
    );
    let mut forged = p1.clone();
    forged.identities.target = first.identity();
    assert!(
        compose_checked_world_points(
            &target,
            &[
                WorldPointParent { claim: &p0, target: &first },
                WorldPointParent { claim: &forged, target: &second },
            ],
            &limits(),
            &ExecutionContext::for_tests(17)
        )
        .is_err()
    );
    let mut snapshot = p1.clone();
    snapshot.identities.data_snapshot = d(99);
    assert!(
        compose_checked_world_points(
            &target,
            &[
                WorldPointParent { claim: &p0, target: &first },
                WorldPointParent { claim: &snapshot, target: &second },
            ],
            &limits(),
            &ExecutionContext::for_tests(17)
        )
        .is_err()
    );
}

#[test]
fn target_quote_refuses_large_payload_and_retains_its_admission() {
    use antecedent_core::assumption::{
        Assumption, AssumptionRecord, AssumptionScope, AssumptionSource, AssumptionStatus,
    };
    use antecedent_core::execution::AdmissionPool;
    let (target, first, second, p0, p1) = fixture();
    let parents = [
        WorldPointParent { claim: &p0, target: &first },
        WorldPointParent { claim: &p1, target: &second },
    ];
    let pool = AdmissionPool::fixed(1, limits().max_memory_bytes);
    let ctx = ExecutionContext::for_tests(17).with_admission(pool.clone());
    let point = compose_checked_world_points(&target, &parents, &limits(), &ctx).unwrap();
    assert_eq!(pool.available().bytes(), 0);
    assert!(compose_checked_world_points(&target, &parents, &limits(), &ctx).is_err());
    let cloned = point.clone();
    assert!(std::ptr::eq(point.target(), cloned.target()));
    drop(point);
    assert_eq!(pool.available().bytes(), 0);
    drop(cloned);
    assert_eq!(pool.available().bytes(), limits().max_memory_bytes);
    let mut records = AssumptionSet::new();
    records.push(AssumptionRecord {
        assumption: Assumption::Custom {
            id: Arc::from("x".repeat(2048)),
            description: Arc::from("declared"),
        },
        source: AssumptionSource::UserDeclared,
        scope: AssumptionScope::Global,
        status: AssumptionStatus::Declared,
    });
    let oversized = ScientificTarget::new(
        target.form().clone(),
        ModelAssumptions::new(GraphFamily::Dag, records),
        target.inference(),
    );
    let mut small = limits();
    small.max_target_payload_bytes = 512;
    assert!(compose_checked_world_points(&oversized, &parents, &small, &ctx).is_err());
    assert_eq!(pool.available().bytes(), limits().max_memory_bytes);
    let mut insufficient = limits();
    insufficient.max_memory_bytes = 1;
    assert!(compose_checked_world_points(&target, &parents, &insufficient, &ctx).is_err());
}

#[test]
fn quotation_checks_nested_expression_depth_before_target_hashing() {
    use antecedent_core::scientific_target::{CompositionQuoteLimits, quote_composition_target};
    use antecedent_core::value_expr::{ExprNumber, PolicyReads, ValueExpr};
    let (_, first, _, _, _) = fixture();
    let TargetForm::World(world) = first.form() else { panic!("world") };
    let mut expr = ValueExpr::Const(ExprNumber::new(1.0));
    for _ in 0..12 {
        expr = ValueExpr::Neg(Box::new(expr));
    }
    let action = WorldAction::at_decision([Intervention::Assign {
        variable: VariableId::from_raw(0),
        expr,
        reads: PolicyReads::default(),
    }])
    .unwrap();
    let changed = InterventionWorld::new(
        world.id(),
        action,
        world.is_reference(),
        world.binding().clone(),
        world.time().clone(),
    )
    .unwrap();
    let nested = ScientificTarget::new(
        TargetForm::World(changed),
        first.assumptions().clone(),
        first.inference(),
    );
    let ctx = ExecutionContext::for_tests(17);
    let mut q =
        CompositionQuoteLimits { max_work: 10_000, max_payload_bytes: 1_000_000, max_depth: 5 };
    assert!(quote_composition_target(&nested, &q, &ctx).is_err());
    q.max_depth = 64;
    assert!(quote_composition_target(&nested, &q, &ctx).is_ok());
    ctx.cancellation.cancel();
    assert!(quote_composition_target(&nested, &q, &ctx).is_err());
}

#[test]
fn quotation_counts_sequence_and_expression_depth_together() {
    use antecedent_core::intervention::{
        InterventionSequence, SequencedIntervention, TemporalPolicy,
    };
    use antecedent_core::scientific_target::{CompositionQuoteLimits, quote_composition_target};
    use antecedent_core::value_expr::{ExprNumber, PolicyReads, ValueExpr};
    let (_, first, _, _, _) = fixture();
    let TargetForm::World(world) = first.form() else { panic!("world") };
    let mut expr = ValueExpr::Const(ExprNumber::new(1.0));
    for _ in 0..5 {
        expr = ValueExpr::Neg(Box::new(expr));
    }
    let mut intervention = Intervention::Assign {
        variable: VariableId::from_raw(0),
        expr,
        reads: PolicyReads::default(),
    };
    for _ in 0..5 {
        intervention = Intervention::Sequence(InterventionSequence::new(Arc::from([
            SequencedIntervention::new(intervention, TemporalPolicy::pulse(0)),
        ])));
    }
    let action = WorldAction::at_decision([intervention]).unwrap();
    let changed = InterventionWorld::new(
        world.id(),
        action,
        world.is_reference(),
        world.binding().clone(),
        world.time().clone(),
    )
    .unwrap();
    let nested = ScientificTarget::new(
        TargetForm::World(changed),
        first.assumptions().clone(),
        first.inference(),
    );
    let ctx = ExecutionContext::for_tests(17);
    let mut q =
        CompositionQuoteLimits { max_work: 10_000, max_payload_bytes: 1_000_000, max_depth: 8 };
    assert!(quote_composition_target(&nested, &q, &ctx).is_err());
    q.max_depth = 64;
    assert!(quote_composition_target(&nested, &q, &ctx).is_ok());
}

#[test]
fn work_budget_includes_repeated_world_encodes() {
    use antecedent_core::scientific_target::{CompositionQuoteLimits, quote_composition_target};
    let (target, first, second, p0, p1) = fixture();
    let parents = [
        WorldPointParent { claim: &p0, target: &first },
        WorldPointParent { claim: &p1, target: &second },
    ];
    let ctx = ExecutionContext::for_tests(17);
    let q =
        CompositionQuoteLimits { max_work: 100_000, max_payload_bytes: 1_000_000, max_depth: 64 };
    let once = [&target, &first, &second]
        .into_iter()
        .map(|t| quote_composition_target(t, &q, &ctx).unwrap().work)
        .sum::<usize>();
    let mut limited = limits();
    limited.max_target_work = once * 2;
    assert!(compose_checked_world_points(&target, &parents, &limited, &ctx).is_err());
    assert!(compose_checked_world_points(&target, &parents, &limits(), &ctx).is_ok());
}

#[test]
fn preflight_returns_specific_errors_before_admission() {
    use antecedent_contract::CheckedWorldPointError;
    use antecedent_core::scientific_target::HistoryIndexedTargetLaw;
    let (target, first, second, p0, p1) = fixture();
    let parents = [
        WorldPointParent { claim: &p0, target: &first },
        WorldPointParent { claim: &p1, target: &second },
    ];
    let mut no_memory = limits();
    no_memory.max_memory_bytes = 0;
    let ctx = ExecutionContext::for_tests(17);
    assert_eq!(
        compose_checked_world_points(&first, &parents, &no_memory, &ctx).unwrap_err(),
        CheckedWorldPointError::ExpectedContrast
    );
    let law = HistoryIndexedTargetLaw::new("snapshot", [VariableId::from_raw(0)], [(vec![0], 1.0)])
        .unwrap();
    let history = target.clone().with_history_indexed_law(law.clone());
    assert_eq!(
        compose_checked_world_points(&history, &parents, &no_memory, &ctx).unwrap_err(),
        CheckedWorldPointError::HistoryLawNotSupported
    );
    let history_parent = first.clone().with_history_indexed_law(law);
    let parents_with_history =
        [WorldPointParent { claim: &p0, target: &history_parent }, parents[1]];
    assert_eq!(
        compose_checked_world_points(&target, &parents_with_history, &no_memory, &ctx).unwrap_err(),
        CheckedWorldPointError::HistoryLawNotSupported
    );
    let wrong_kind = [WorldPointParent { claim: &p0, target: &target }, parents[1]];
    assert_eq!(
        compose_checked_world_points(&target, &wrong_kind, &no_memory, &ctx).unwrap_err(),
        CheckedWorldPointError::WitnessMismatch
    );
    let mut too_few = no_memory;
    too_few.max_terms = 1;
    assert_eq!(
        compose_checked_world_points(&target, &parents[..1], &too_few, &ctx).unwrap_err(),
        CheckedWorldPointError::TermLimit
    );
    ctx.cancellation.cancel();
    assert_eq!(
        compose_checked_world_points(&first, &parents, &no_memory, &ctx).unwrap_err(),
        CheckedWorldPointError::Cancelled
    );
}
