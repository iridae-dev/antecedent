//! Engineering lifecycle contracts. The independent numerical oracle is separately owed.
use antecedent::{BoundDecisionContract, JointScenarioCarrier, PreparedStudy, RefuteSuite, Study};
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::execution_attempt::{Operation, observe_execution};
use antecedent_core::{AverageEffectQuery, ExecutionContext, SearchStop, VariableId};
use antecedent_data::TabularData;
use antecedent_design::decision_adapters::{
    ClaimProbability, SuppliedClaim, adapt_finite_scenarios,
};
use antecedent_design::decision_contract::{
    ActionKind, AdmissibilityRules, AdmissibleDecisionContract, DecisionAction, DecisionContract,
    DecisionCriterion, StructuralPolicy, UtilityExpr,
};
use antecedent_design::decision_robustness::{AtomSupport, RobustVerdict};
use antecedent_design::decision_structural::{AtomEvidence, StructuralVerdict};
use antecedent_estimate::LinearAdjustmentAte;
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io::quantity_wire::DistributionMeaningWire;

fn prepared(ctx: &ExecutionContext) -> PreparedStudy {
    let t = [0., 1., 0., 1., 0., 1., 0., 1.];
    let z = [0., 0., 1., 1., 0., 0., 1., 1.];
    let y = [0., 2., 1., 3., 1., 3., 2., 4.];
    let data = TabularData::from_f64_columns(vec![
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
    ])
    .unwrap();
    let mut graph = Dag::with_variables(3);
    for (a, b) in [(2, 0), (2, 1), (0, 1)] {
        graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    Study::tabular(data)
        .graph(graph)
        .query(AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)))
        .estimator(LinearAdjustmentAte::new().with_bootstrap_replicates(0))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .prepare(ctx)
        .unwrap()
}
fn q(name: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: name.into(),
        variable_name: name.into(),
        role: QuantityRole::Outcome,
        units: "units".into(),
        population_id: "target".into(),
        regime_id: "do(joint)".into(),
        horizon: 0,
        functional_id: "outcome".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}
fn law(positive: bool, alignment: DrawAlignment) -> Box<DistributionArtifact> {
    let identity = DistributionIdentity::new(
        DistributionMeaningWire::InterventionalPredictive,
        &[q("a"), q("b")],
        alignment,
        DistributionProvenance {
            source_id: "declared".into(),
            provider_id: "enumeration".into(),
            rng_id: "none".into(),
            snapshot_id: "fixture".into(),
            causal_contract_id: "conditional".into(),
        },
    )
    .unwrap();
    Box::new(
        DistributionArtifact::new(
            DistributionMetadata {
                version: 1,
                identity,
                axes: ["draw".into(), "quantity".into()],
                shape: [2, 2],
                weights: None,
                supported: None,
                calibration: DistributionCalibration::Exact,
                trust: DistributionTrust::Unverified,
                legacy_posterior: None,
                legacy_bindings: None,
            },
            if positive { vec![0., 0., 1., 1.] } else { vec![0., 1., 1., 0.] },
        )
        .unwrap(),
    )
}
fn contract(policy: StructuralPolicy) -> AdmissibleDecisionContract {
    AdmissibleDecisionContract {
        contract: DecisionContract {
            actions: vec![
                DecisionAction {
                    id: "pair".into(),
                    kind: ActionKind::Intervention,
                    inputs: vec![q("a"), q("b")],
                    utility: UtilityExpr::Mul(
                        Box::new(UtilityExpr::Input(0)),
                        Box::new(UtilityExpr::Input(1)),
                    ),
                },
                DecisionAction {
                    id: "safe".into(),
                    kind: ActionKind::Intervention,
                    inputs: vec![q("a")],
                    utility: UtilityExpr::sum(
                        UtilityExpr::product(UtilityExpr::Const(0.), UtilityExpr::Input(0)),
                        UtilityExpr::Const(0.3),
                    ),
                },
            ],
            utility_units: "units".into(),
            criterion: DecisionCriterion::PosteriorExpectedUtility,
            constraints: vec![],
            target_population: "target".into(),
            horizon: 0,
            structural_policy: policy,
        },
        rules: AdmissibilityRules::default(),
    }
}
fn evaluated(id: &str, positive: bool, p: ClaimProbability) -> SuppliedClaim {
    SuppliedClaim {
        id: id.into(),
        probability: p,
        evidence: AtomEvidence::Evaluated(law(positive, DrawAlignment::Joint)),
        support: AtomSupport::supported(),
    }
}
fn source(
    p: &PreparedStudy,
    claims: Vec<SuppliedClaim>,
    exhaustive: bool,
    stop: Option<SearchStop>,
    ctx: &ExecutionContext,
) -> JointScenarioCarrier {
    let c = p.contract().unwrap();
    JointScenarioCarrier::checked(
        &c.identities.target.to_hex(),
        &c.identities.identification.to_hex(),
        "common paired states across actions",
        exhaustive,
        stop,
        adapt_finite_scenarios(claims, StructuralPolicy::ReportOnly).unwrap(),
        ctx,
    )
    .unwrap()
}
fn compile(
    p: &PreparedStudy,
    s: JointScenarioCarrier,
    policy: StructuralPolicy,
    ctx: &ExecutionContext,
) -> antecedent::CompiledRobustDecision {
    let c = p.contract().unwrap();
    p.compile_for_robust_decision(
        BoundDecisionContract::bind(&c, contract(policy)).unwrap(),
        s,
        ctx,
    )
    .unwrap()
}
#[test]
fn compiler_binds_joint_source_without_scalar_fit_and_replays_under_independent_owner() {
    let ctx = ExecutionContext::for_tests(41);
    let p = prepared(&ctx);
    let s = source(
        &p,
        vec![
            evaluated("positive", true, ClaimProbability::CompletionCount(17)),
            evaluated("negative", false, ClaimProbability::CompletionCount(9)),
        ],
        true,
        None,
        &ctx,
    );
    let observed = observe_execution(|| {
        Ok::<_, antecedent::CausalError>(compile(&p, s.clone(), StructuralPolicy::Maximin, &ctx))
    });
    assert_eq!(observed.report.counts(Operation::LeastSquaresSolve).attempted, 0);
    assert_eq!(observed.report.counts(Operation::GlmFit).attempted, 0);
    let compiled = observed.result.unwrap();
    let metadata = compiled.plan().plan_view().unwrap().to_json();
    assert!(metadata.to_string().contains("identified_set"));
    assert!(metadata.to_string().contains("scenario_set"));
    assert!(metadata.to_string().contains(s.identity()));
    let readout = observe_execution(|| compiled.execute(&ctx));
    assert_eq!(readout.report.counts(Operation::LeastSquaresSolve).attempted, 0);
    let readout = readout.result.unwrap();
    assert!(readout.complete());
    assert_eq!(
        readout.decision().completion_counts,
        vec![("positive".into(), 17), ("negative".into(), 9)]
    );
    let bytes = s.export(&ctx).unwrap();
    assert!(JointScenarioCarrier::consume(&bytes, &"0".repeat(64), &ctx).is_err());
    let loaded = JointScenarioCarrier::consume(&bytes, s.identity(), &ctx).unwrap();
    let independent = compile(&prepared(&ctx), loaded, StructuralPolicy::Maximin, &ctx);
    assert_eq!(independent.program_identity(), compiled.program_identity());
    let sealed = compiled.export(&ctx).unwrap();
    assert_eq!(independent.consume(&sealed, &ctx).unwrap().decision(), readout.decision());
    let other = compile(&p, s, StructuralPolicy::ReportOnly, &ctx);
    assert!(other.consume(&sealed, &ctx).is_err());
    let mut corrupt = sealed.clone();
    let last = corrupt.len() - 1;
    corrupt[last] ^= 1;
    assert!(compiled.consume(&corrupt, &ctx).is_err());
    let mut wire: serde_json::Value =
        antecedent_io::from_cbor(&sealed[b"ANTE-COMPILED-ROBUST-1".len()..]).unwrap();
    wire["result"]["actions"][0]["range"] = serde_json::json!([91.0, 92.0]);
    let mut forged = b"ANTE-COMPILED-ROBUST-1".to_vec();
    forged.extend(antecedent_io::to_cbor(&wire).unwrap());
    assert!(compiled.consume(&forged, &ctx).is_err());
}
#[test]
fn incomplete_search_and_unidentified_mass_never_become_a_complete_choice() {
    let ctx = ExecutionContext::for_tests(41);
    let p = prepared(&ctx);
    let missing = SuppliedClaim {
        id: "unknown".into(),
        probability: ClaimProbability::Genuine(0.2),
        evidence: AtomEvidence::Unidentified,
        support: AtomSupport::supported(),
    };
    let s = source(
        &p,
        vec![
            evaluated("s1", true, ClaimProbability::Genuine(0.6)),
            evaluated("s2", false, ClaimProbability::Genuine(0.2)),
            missing,
        ],
        true,
        Some(SearchStop::Operations),
        &ctx,
    );
    let c = compile(&p, s, StructuralPolicy::Maximin, &ctx);
    let r = c.execute(&ctx).unwrap();
    assert!(!r.complete());
    assert_eq!(r.search_stop(), Some("search.operations"));
    assert_eq!(r.decision().robust.unidentified_mass, Some(0.2));
    assert_eq!(r.decision().robust.evaluated_mass, Some(0.8));
    assert!(matches!(r.decision().robust.verdict, RobustVerdict::InsufficientClaims(_)));
    assert!(matches!(r.decision().structural.verdict, StructuralVerdict::InsufficientScience(_)));
    let s =
        source(&p, vec![evaluated("s1", true, ClaimProbability::Unspecified)], false, None, &ctx);
    let r =
        compile(&p, s, StructuralPolicy::RequireInvariantBestAction, &ctx).execute(&ctx).unwrap();
    assert!(!r.complete());
    assert!(r.decision().robust.verdict.selected_action().is_none());
}
#[test]
fn joint_policy_identity_admission_and_cancellation_refusals_preserve_the_owner() {
    let ctx = ExecutionContext::for_tests(41);
    let p = prepared(&ctx);
    let causal = p.contract().unwrap();
    for (alignment, structure) in
        [(DrawAlignment::IndependentMarginals, "declared"), (DrawAlignment::Joint, "")]
    {
        let mut claim = evaluated("s", true, ClaimProbability::Unspecified);
        claim.evidence = AtomEvidence::Evaluated(law(true, alignment));
        assert!(
            JointScenarioCarrier::checked(
                &causal.identities.target.to_hex(),
                &causal.identities.identification.to_hex(),
                structure,
                true,
                None,
                adapt_finite_scenarios(vec![claim], StructuralPolicy::ReportOnly).unwrap(),
                &ctx
            )
            .is_err()
        );
    }
    let s = source(&p, vec![evaluated("s", true, ClaimProbability::Unspecified)], true, None, &ctx);
    let bound =
        BoundDecisionContract::bind(&causal, contract(StructuralPolicy::BayesOverStructures))
            .unwrap();
    assert!(p.compile_for_robust_decision(bound, s.clone(), &ctx).is_err());
    let c = compile(&p, s, StructuralPolicy::Maximin, &ctx);
    let tiny = ctx.clone().with_memory_limit(Some(1));
    assert!(c.execute(&tiny).is_err());
    assert!(c.export(&tiny).is_err());
    let cancelled = ExecutionContext::for_tests(41);
    cancelled.cancellation.cancel();
    assert!(c.execute(&cancelled).is_err());
    assert!(c.consume(&c.export(&ctx).unwrap(), &cancelled).is_err());
    assert!(c.execute(&ctx).is_ok());
}
#[test]
fn retained_cache_and_result_release_their_distinct_memory_leases() {
    let ctx = ExecutionContext::for_tests(41).with_memory_limit(Some(64 * 1024 * 1024));
    let p = prepared(&ctx);
    p.contract().unwrap();
    let baseline = ctx.admission().available().bytes();
    let s = source(&p, vec![evaluated("s", true, ClaimProbability::Unspecified)], true, None, &ctx);
    let c = compile(&p, s, StructuralPolicy::Maximin, &ctx);
    let r = c.execute(&ctx).unwrap();
    assert!(ctx.admission().available().bytes() < baseline);
    let clone = r.clone();
    drop(c);
    drop(r);
    assert!(ctx.admission().available().bytes() < baseline);
    drop(clone);
    assert_eq!(ctx.admission().available().bytes(), baseline);
}
/// Independent finite enumeration; deliberately owed and excluded from engineering.
#[test]
fn joint_dependence_and_maximin_match_independent_finite_enumeration() {
    let ctx = ExecutionContext::for_tests(41);
    let p = prepared(&ctx);
    // Identical Bernoulli(1/2) marginals: E[AB] is 1/2 under equal states,
    // zero under opposite states. Neither product is recoverable from marginals.
    for (positive, expected) in [(true, "pair"), (false, "safe")] {
        let s = source(
            &p,
            vec![evaluated("s", positive, ClaimProbability::Unspecified)],
            true,
            None,
            &ctx,
        );
        let r = compile(&p, s, StructuralPolicy::Maximin, &ctx).execute(&ctx).unwrap();
        assert_eq!(r.decision().robust.verdict.selected_action(), Some(expected));
        let pair = r.decision().structural.actions.iter().find(|a| a.id == "pair").unwrap();
        assert_eq!(pair.range, Some(if positive { (0.5, 0.5) } else { (0., 0.) }));
    }
    let s = source(
        &p,
        vec![
            evaluated("positive", true, ClaimProbability::Unspecified),
            evaluated("negative", false, ClaimProbability::Unspecified),
        ],
        true,
        None,
        &ctx,
    );
    assert_eq!(
        compile(&p, s.clone(), StructuralPolicy::Maximin, &ctx)
            .execute(&ctx)
            .unwrap()
            .decision()
            .robust
            .verdict
            .selected_action(),
        Some("safe")
    );
    assert!(
        compile(&p, s, StructuralPolicy::RequireInvariantBestAction, &ctx)
            .execute(&ctx)
            .unwrap()
            .decision()
            .robust
            .verdict
            .selected_action()
            .is_none()
    );
}

#[test]
fn bounded_requests_and_wrong_binding_refuse_and_failed_evaluation_stays_incomplete() {
    let ctx = ExecutionContext::for_tests(41);
    let p = prepared(&ctx);
    let causal = p.contract().unwrap();
    let too_many = (0..257)
        .map(|i| SuppliedClaim {
            id: format!("unknown-{i}"),
            probability: ClaimProbability::Unspecified,
            evidence: AtomEvidence::Unidentified,
            support: AtomSupport::supported(),
        })
        .collect();
    assert!(
        JointScenarioCarrier::checked(
            &causal.identities.target.to_hex(),
            &causal.identities.identification.to_hex(),
            "declared",
            true,
            None,
            adapt_finite_scenarios(too_many, StructuralPolicy::ReportOnly).unwrap(),
            &ctx
        )
        .is_err()
    );
    let wrong = JointScenarioCarrier::checked(
        &"0".repeat(64),
        &causal.identities.identification.to_hex(),
        "declared",
        true,
        None,
        adapt_finite_scenarios(
            vec![evaluated("s", true, ClaimProbability::Unspecified)],
            StructuralPolicy::ReportOnly,
        )
        .unwrap(),
        &ctx,
    )
    .unwrap();
    let bound = BoundDecisionContract::bind(&causal, contract(StructuralPolicy::Maximin)).unwrap();
    assert!(p.compile_for_robust_decision(bound, wrong, &ctx).is_err());
    let s = source(&p, vec![evaluated("s", true, ClaimProbability::Unspecified)], true, None, &ctx);
    for constraints in [false, true] {
        let mut c = contract(StructuralPolicy::Maximin);
        if constraints {
            c.contract.constraints = (0..257)
                .map(|i| antecedent_design::decision_contract::HardConstraint {
                    id: format!("c-{i}"),
                    expr: UtilityExpr::Input(0),
                    bound: 2.,
                    min_probability: 1.,
                    units: "units".into(),
                    applies_to: vec!["safe".into()],
                })
                .collect();
        } else {
            let action = c.contract.actions[0].clone();
            c.contract.actions = (0..257)
                .map(|i| DecisionAction { id: format!("a-{i}"), ..action.clone() })
                .collect();
        }
        assert!(s.evaluate(&c, &ctx).is_err());
    }
    let oversized = vec![0; antecedent::MAX_SCENARIO_CARRIER_BYTES + 1];
    assert!(
        JointScenarioCarrier::consume(
            &oversized,
            s.identity(),
            &ctx.clone().with_memory_limit(Some(1))
        )
        .is_err()
    );
    let mut c = contract(StructuralPolicy::Maximin);
    c.contract.actions[0].utility = UtilityExpr::product(
        UtilityExpr::Input(0),
        UtilityExpr::product(UtilityExpr::Const(f64::MAX), UtilityExpr::Const(f64::MAX)),
    );
    let failed = s.evaluate(&c, &ctx).unwrap();
    assert!(!failed.complete());
    assert!(failed.decision().robust.verdict.selected_action().is_none());
    assert!(matches!(
        failed.decision().structural.atoms[0].status,
        antecedent_design::decision_structural::AtomStatus::Unevaluated(_)
    ));
    let mut wire: serde_json::Value =
        antecedent_io::from_cbor(&s.export(&ctx).unwrap()[b"ANTE-JOINT-SCENARIOS-1".len()..])
            .unwrap();
    wire["version"] = serde_json::json!(2);
    let mut bytes = b"ANTE-JOINT-SCENARIOS-1".to_vec();
    bytes.extend(antecedent_io::to_cbor(&wire).unwrap());
    assert!(JointScenarioCarrier::consume(&bytes, s.identity(), &ctx).is_err());
}
