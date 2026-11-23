//! The compiled robust terminal and the direct `evaluate_adapted` / `evaluate_robust` calls
//! evaluate the same finite joint scenarios through the same kernel.
//!
//! The direct entries take claims a caller supplies with no link to a study. The compiled
//! terminal (`PreparedStudy::compile_for_robust_decision`) reads the same claims out of a
//! carrier bound to the study's target and identification identities, rebuilds them with
//! `adapt_finite_scenarios` and calls `evaluate_adapted`. On a complete, exhaustive scenario
//! set every number and verdict of the structural and robust results agree bit for bit; the
//! compiled terminal appends one assumption naming the declared joint structure, and an
//! incomplete set forces an `InsufficientClaims` verdict that the direct call does not.
//!
//! These decisions have no sampling interval, so there is no calibration basis to compare.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::float_cmp, reason = "the bits are compared exactly")]

use antecedent::{BoundDecisionContract, JointScenarioCarrier, PreparedStudy, RefuteSuite, Study};
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::{AverageEffectQuery, ExecutionContext, SearchStop, VariableId};
use antecedent_data::TabularData;
use antecedent_design::decision_adapters::{
    AdaptedDecision, ClaimProbability, SuppliedClaim, adapt_finite_scenarios, evaluate_adapted,
};
use antecedent_design::decision_contract::{
    ActionKind, AdmissibilityRules, AdmissibleDecisionContract, DecisionAction, DecisionContract,
    DecisionCriterion, StructuralPolicy, UtilityExpr,
};
use antecedent_design::decision_robustness::{AtomSupport, RobustVerdict, evaluate_robust};
use antecedent_design::decision_structural::AtomEvidence;
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

fn claims(positive_count: u64, negative_count: u64) -> Vec<SuppliedClaim> {
    vec![
        evaluated("positive", true, ClaimProbability::CompletionCount(positive_count)),
        evaluated("negative", false, ClaimProbability::CompletionCount(negative_count)),
    ]
}

fn direct(policy: StructuralPolicy, claims: Vec<SuppliedClaim>) -> AdaptedDecision {
    let adapted = adapt_finite_scenarios(claims, policy).unwrap();
    evaluate_adapted(&contract(policy), &adapted).unwrap()
}

fn assert_same_decision(compiled: &AdaptedDecision, direct: &AdaptedDecision) {
    assert_eq!(compiled.kind, direct.kind);
    assert_eq!(compiled.structural, direct.structural);
    assert_eq!(compiled.completion_counts, direct.completion_counts);
    for (c, d) in compiled.robust.actions.iter().zip(&direct.robust.actions) {
        assert_eq!(c.id, d.id);
        for (cr, dr) in [(c.range, d.range), (c.supported_range, d.supported_range)] {
            assert_eq!(
                cr.map(|(lo, hi)| (lo.to_bits(), hi.to_bits())),
                dr.map(|(lo, hi)| (lo.to_bits(), hi.to_bits()))
            );
        }
    }
    // The compiled terminal appends the declared joint structure to the assumptions; nothing
    // else of the robust result differs on a complete scenario set.
    let mut robust = compiled.robust.clone();
    let appended = robust.assumptions.pop().expect("the joint-structure assumption");
    assert!(appended.contains("conditional on declared joint scenarios"), "{appended}");
    assert_eq!(robust, direct.robust);
}

#[test]
fn compiled_robust_terminal_equals_direct_evaluate_adapted_on_joint_scenarios() {
    let ctx = ExecutionContext::for_tests(41);
    let p = prepared(&ctx);
    for policy in [StructuralPolicy::Maximin, StructuralPolicy::ReportOnly] {
        let s = source(&p, claims(17, 9), true, None, &ctx);
        let compiled = compile(&p, s, policy, &ctx).execute(&ctx).unwrap();
        assert!(compiled.complete());
        assert_same_decision(compiled.decision(), &direct(policy, claims(17, 9)));
    }
}

#[test]
fn evaluate_robust_on_the_adapted_atoms_is_the_same_kernel_call() {
    let adapted = adapt_finite_scenarios(claims(17, 9), StructuralPolicy::Maximin).unwrap();
    let (structural, robust) =
        evaluate_robust(&contract(StructuralPolicy::Maximin), &adapted.atoms, &adapted.profile)
            .unwrap();
    let decision = direct(StructuralPolicy::Maximin, claims(17, 9));
    assert_eq!(structural, decision.structural);
    assert_eq!(robust, decision.robust);
}

#[test]
fn an_exported_carrier_loaded_under_its_retained_identity_decides_as_the_direct_call_does() {
    let ctx = ExecutionContext::for_tests(41);
    let p = prepared(&ctx);
    let s = source(&p, claims(17, 9), true, None, &ctx);
    let bytes = s.export(&ctx).unwrap();
    // The carrier is the compiled replacement for supplied claims: it loads only under the
    // identity the consumer retained, then the study binds and decides it.
    assert!(JointScenarioCarrier::consume(&bytes, &"0".repeat(64), &ctx).is_err());
    let loaded = JointScenarioCarrier::consume(&bytes, s.identity(), &ctx).unwrap();
    assert_eq!(loaded.identity(), s.identity());
    let compiled = compile(&prepared(&ctx), loaded, StructuralPolicy::Maximin, &ctx);
    let readout = compiled.execute(&ctx).unwrap();
    assert_same_decision(readout.decision(), &direct(StructuralPolicy::Maximin, claims(17, 9)));
}

#[test]
fn an_incomplete_scenario_set_keeps_the_direct_numbers_and_withholds_the_verdict() {
    let ctx = ExecutionContext::for_tests(41);
    let p = prepared(&ctx);
    let unknown = || SuppliedClaim {
        id: "unknown".into(),
        probability: ClaimProbability::Genuine(0.2),
        evidence: AtomEvidence::Unidentified,
        support: AtomSupport::supported(),
    };
    let set = || {
        vec![
            evaluated("s1", true, ClaimProbability::Genuine(0.6)),
            evaluated("s2", false, ClaimProbability::Genuine(0.2)),
            unknown(),
        ]
    };
    let s = source(&p, set(), true, Some(SearchStop::Operations), &ctx);
    let compiled = compile(&p, s, StructuralPolicy::Maximin, &ctx).execute(&ctx).unwrap();
    let direct = direct(StructuralPolicy::Maximin, set());
    assert!(!compiled.complete());
    // The direct call has no declared search, so it reports its own verdict; the compiled
    // terminal withholds it. The per-action ranges and the masses are the same numbers.
    assert!(matches!(compiled.decision().robust.verdict, RobustVerdict::InsufficientClaims(_)));
    assert_eq!(compiled.decision().structural.actions, direct.structural.actions);
    for (c, d) in compiled.decision().robust.actions.iter().zip(&direct.robust.actions) {
        assert_eq!(
            c.range.map(|(lo, hi)| (lo.to_bits(), hi.to_bits())),
            d.range.map(|(lo, hi)| (lo.to_bits(), hi.to_bits()))
        );
    }
    assert_eq!(compiled.decision().robust.unidentified_mass, direct.robust.unidentified_mass);
    assert_eq!(compiled.decision().robust.evaluated_mass, direct.robust.evaluated_mass);
}

#[test]
fn a_posterior_policy_is_refused_by_the_compiled_terminal() {
    // The direct kernel accepts any policy the claims were adapted under; the compiled terminal
    // does not read a robust set as a posterior, and refuses before any evaluation.
    let ctx = ExecutionContext::for_tests(41);
    let p = prepared(&ctx);
    let s = source(&p, claims(17, 9), true, None, &ctx);
    let c = p.contract().unwrap();
    let bound =
        BoundDecisionContract::bind(&c, contract(StructuralPolicy::BayesOverStructures)).unwrap();
    assert!(p.compile_for_robust_decision(bound, s, &ctx).is_err());
}
