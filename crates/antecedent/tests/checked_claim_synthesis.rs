//! Checked claim synthesis uses actual prepared-study envelopes and identity domains.
use antecedent::analysis::PreparedStudy;
use antecedent::analysis::checked_claim_composition::{
    PreparedClaimParent, compose_checked_study_claims,
};
use antecedent::planner::lowered::ProblemTarget;
use antecedent::{Dag, DenseNodeId, RefuteSuite, Study};
use antecedent_contract::{CheckedWorldPointLimits, ClaimEnvelope, ClaimOperation, DomainStatus};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ExecutionContext, Intervention, InterventionContrast,
    InterventionWorld, ScientificTarget, ScientificWorldId, TargetForm, Value, VariableId,
    WorldCollection,
};
use antecedent_data::TabularData;

fn limits() -> CheckedWorldPointLimits {
    CheckedWorldPointLimits {
        max_target_work: 200_000,
        max_target_payload_bytes: 2_000_000,
        max_target_depth: 64,
        max_memory_bytes: 64_000_000,
        max_terms: 16,
        max_comparisons: 128,
        max_world_actions: 32,
        max_assumption_records: 128,
        max_evidence_references: 32,
        max_evidence_bytes: 4096,
    }
}

fn data() -> TabularData {
    // A continuous dose with observed 0, 1 and 2 supports both declared linear
    // contrasts over the same complete-case population.
    let treatment: Vec<f64> = (0..240).map(|i| (i as f64 - 120.0) / 40.0).collect();
    let outcome: Vec<f64> =
        treatment.iter().enumerate().map(|(i, t)| 3.0 * t + 0.1 * ((i % 5) as f64 - 2.0)).collect();
    TabularData::from_f64_columns([("t", treatment.as_slice()), ("y", outcome.as_slice())]).unwrap()
}

fn prepare(data: &TabularData, level: f64, ctx: &ExecutionContext) -> PreparedStudy {
    let mut graph = Dag::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let mut query =
        AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    query.active = Intervention::set(VariableId::from_raw(0), Value::f64(level));
    Study::tabular(data.clone())
        .graph(graph)
        .query(CausalQuery::AverageEffect(query))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(ctx)
        .unwrap()
}

fn target(study: &PreparedStudy) -> &ScientificTarget {
    let ProblemTarget::Target(target) = study.plan().lowered().unwrap().scientific().target()
    else {
        panic!("scalar scientific target")
    };
    target
}

fn combined(
    first: &ScientificTarget,
    second: &ScientificTarget,
    weights: [f64; 2],
) -> ScientificTarget {
    let TargetForm::Contrast(a) = first.form() else { panic!("first ATE") };
    let TargetForm::Contrast(b) = second.form() else { panic!("second ATE") };
    let world = |c: &InterventionContrast, coefficient: f64| {
        let term = c.terms().into_iter().find(|t| t.coefficient == coefficient).unwrap();
        c.worlds().get(term.world).unwrap().clone()
    };
    let raw = [
        (world(a, -1.0), -(weights[0] + weights[1])),
        (world(a, 1.0), weights[0]),
        (world(b, 1.0), weights[1]),
    ];
    let raw: Vec<_> = raw.into_iter().filter(|(_, coefficient)| *coefficient != 0.0).collect();
    let mut terms = Vec::new();
    let worlds: Vec<_> = raw
        .into_iter()
        .enumerate()
        .map(|(i, (w, coefficient))| {
            let id = ScientificWorldId::from_raw(i as u32);
            terms.push((id, coefficient));
            InterventionWorld::new(
                id,
                w.action().clone(),
                i == 0,
                w.binding().clone(),
                w.time().clone(),
            )
            .unwrap()
        })
        .collect();
    let contrast = InterventionContrast::linear(
        WorldCollection::new(worlds).unwrap(),
        terms,
        ScientificWorldId::from_raw(0),
    )
    .unwrap();
    ScientificTarget::new(
        TargetForm::Contrast(contrast),
        first.assumptions().clone(),
        first.inference(),
    )
}

fn claim(study: &PreparedStudy, data: &TabularData, ctx: &ExecutionContext) -> ClaimEnvelope {
    let result = study.estimate(data, ctx).unwrap();
    result.claim(&study.contract_for_result(&result).unwrap(), ctx).unwrap()
}

#[test]
fn prepared_claims_synthesize_checked_contrast_and_weighted_aggregate() {
    let execution = ExecutionContext::for_tests(17);
    let data = data();
    let first = prepare(&data, 1.0, &execution);
    let second = prepare(&data, 2.0, &execution);
    let c1 = claim(&first, &data, &execution);
    let c2 = claim(&second, &data, &execution);
    // The facade must prove this link; these are genuinely different identity domains.
    assert_ne!(c1.identities.target, target(&first).identity());
    assert_ne!(c2.identities.target, target(&second).identity());
    assert!(c1.outcome_units.is_none() && c2.outcome_units.is_none());
    assert_eq!(c1.domains.identification, DomainStatus::Identified);
    assert_eq!(c2.domains.identification, DomainStatus::Identified);
    let original_ids = [c1.claim_id, c2.claim_id];
    let composition = ExecutionContext::for_tests(99);
    for (weights, operation) in
        [([1.0, -1.0], ClaimOperation::Contrast), ([0.25, 0.75], ClaimOperation::Aggregate)]
    {
        let output = combined(target(&first), target(&second), weights);
        let parents = [
            PreparedClaimParent { prepared: &first, claim: &c1, coefficient: weights[0] },
            PreparedClaimParent { prepared: &second, claim: &c2, coefficient: weights[1] },
        ];
        let derived =
            compose_checked_study_claims(&output, &parents, operation, &limits(), &composition)
                .unwrap();
        let expected = weights[0] * c1.value.unwrap() + weights[1] * c2.value.unwrap();
        assert!((derived.claim().value.unwrap() - expected).abs() < 1e-12);
        assert!(
            derived.parents().contains(&original_ids[0])
                && derived.parents().contains(&original_ids[1])
        );
        assert!(derived.claim().identities.program.is_none());
        assert!(derived.claim().identities.identification_product.is_none());
        assert!(derived.claim().execution.is_none());
        assert_eq!(derived.claim().domains.identification, DomainStatus::Unknown);
        assert_eq!(derived.claim().calibration, Default::default());
        assert!(derived.provenance().validate().unwrap().is_empty());
        assert_eq!(derived.receipt().output, Some(derived.claim().claim_id));
        assert!(derived.receipt().unavailable_operations.iter().any(|s| &**s == "portable_export"));
        let same = compose_checked_study_claims(
            &output,
            &[parents[1], parents[0]],
            operation,
            &limits(),
            &ExecutionContext::for_tests(100),
        )
        .unwrap();
        assert_eq!(derived.proof_identity(), same.proof_identity());
        assert_eq!(derived.claim().claim_id, same.claim().claim_id);
    }
    assert_eq!([c1.claim_id, c2.claim_id], original_ids);
}

#[test]
fn mismatched_program_target_coefficients_and_support_are_refused() {
    let ctx = ExecutionContext::for_tests(17);
    let data = data();
    let first = prepare(&data, 1.0, &ctx);
    let second = prepare(&data, 2.0, &ctx);
    let c1 = claim(&first, &data, &ctx);
    let c2 = claim(&second, &data, &ctx);
    let output = combined(target(&first), target(&second), [1.0, -1.0]);
    let parents = [
        PreparedClaimParent { prepared: &first, claim: &c1, coefficient: 1.0 },
        PreparedClaimParent { prepared: &second, claim: &c2, coefficient: -1.0 },
    ];
    let wrong_binding =
        [PreparedClaimParent { prepared: &first, claim: &c2, coefficient: 1.0 }, parents[1]];
    assert!(
        compose_checked_study_claims(
            &output,
            &wrong_binding,
            ClaimOperation::Contrast,
            &limits(),
            &ctx
        )
        .is_err()
    );
    let wrong_coefficients = [parents[0], PreparedClaimParent { coefficient: 1.0, ..parents[1] }];
    assert!(
        compose_checked_study_claims(
            &output,
            &wrong_coefficients,
            ClaimOperation::Contrast,
            &limits(),
            &ctx
        )
        .is_err()
    );
    for status in [DomainStatus::OutsideScope, DomainStatus::Contradicted, DomainStatus::Limited] {
        let mut bad = c1.clone();
        bad.domains.support = status;
        let negative =
            [PreparedClaimParent { prepared: &first, claim: &bad, coefficient: 1.0 }, parents[1]];
        assert!(
            compose_checked_study_claims(
                &output,
                &negative,
                ClaimOperation::Contrast,
                &limits(),
                &ctx
            )
            .is_err()
        );
    }
    let wrong_output = combined(target(&first), target(&second), [0.25, 0.75]);
    assert!(
        compose_checked_study_claims(
            &wrong_output,
            &parents,
            ClaimOperation::Contrast,
            &limits(),
            &ctx
        )
        .is_err()
    );
    let mut failures = Vec::new();
    for kind in [
        antecedent_contract::ClaimKind::Bounds,
        antecedent_contract::ClaimKind::Associational,
        antecedent_contract::ClaimKind::Incomplete,
    ] {
        let mut bad = c1.clone();
        bad.kind = kind;
        failures.push(bad);
    }
    for status in [
        DomainStatus::OutsideScope,
        DomainStatus::Contradicted,
        DomainStatus::Limited,
        DomainStatus::NotClaimed,
    ] {
        let mut bad = c1.clone();
        bad.domains.identification = status;
        failures.push(bad);
    }
    for status in [DomainStatus::Unknown, DomainStatus::OutsideScope, DomainStatus::Limited] {
        let mut bad = c1.clone();
        bad.domains.evaluated = status;
        failures.push(bad);
    }
    let mut bad = c1.clone();
    bad.outcome_units = Some("conflicting-units".into());
    failures.push(bad);
    for change in 0..5 {
        let mut bad = c1.clone();
        let antecedent_contract::SlotAvailability::Available(slot) =
            &mut bad.reasoning.identification
        else {
            panic!("identified parent")
        };
        match change {
            0 => {
                slot.identified_mass = 0.5;
                slot.unidentified_mass = 0.5;
            }
            1 => {
                slot.identified_mass = f64::NAN;
            }
            2 => {
                slot.full_mass_scope = false;
            }
            3 => {
                slot.search_capped = true;
            }
            _ => {
                slot.status = antecedent_core::IdentificationStatus::NotIdentified;
            }
        }
        failures.push(bad);
    }
    for bad in failures {
        let invalid =
            [PreparedClaimParent { prepared: &first, claim: &bad, coefficient: 1.0 }, parents[1]];
        assert!(
            compose_checked_study_claims(
                &output,
                &invalid,
                ClaimOperation::Contrast,
                &limits(),
                &ctx
            )
            .is_err()
        );
    }
}

#[test]
fn checked_claim_clones_hold_admission_and_cancellation_remains_typed() {
    use antecedent_contract::CheckedWorldPointError;
    use antecedent_core::execution::AdmissionPool;
    let execution = ExecutionContext::for_tests(17);
    let data = data();
    let first = prepare(&data, 1.0, &execution);
    let second = prepare(&data, 2.0, &execution);
    let c1 = claim(&first, &data, &execution);
    let c2 = claim(&second, &data, &execution);
    let output = combined(target(&first), target(&second), [1.0, -1.0]);
    let parents = [
        PreparedClaimParent { prepared: &first, claim: &c1, coefficient: 1.0 },
        PreparedClaimParent { prepared: &second, claim: &c2, coefficient: -1.0 },
    ];
    let pool = AdmissionPool::fixed(1, limits().max_memory_bytes);
    let composition = ExecutionContext::for_tests(99).with_admission(pool.clone());
    let derived = compose_checked_study_claims(
        &output,
        &parents,
        ClaimOperation::Contrast,
        &limits(),
        &composition,
    )
    .unwrap();
    assert_eq!(pool.available().bytes(), 0);
    let clone = derived.clone();
    assert!(std::ptr::eq(derived.claim(), clone.claim()));
    drop(derived);
    assert_eq!(pool.available().bytes(), 0);
    drop(clone);
    assert_eq!(pool.available().bytes(), limits().max_memory_bytes);
    composition.cancellation.cancel();
    assert_eq!(
        compose_checked_study_claims(
            &output,
            &parents,
            ClaimOperation::Contrast,
            &limits(),
            &composition
        )
        .unwrap_err(),
        CheckedWorldPointError::Cancelled
    );
}
