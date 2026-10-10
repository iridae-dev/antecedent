//! Exact finite SCM, two real attested providers and one measured shared workflow.
//! SPDX-License-Identifier: MIT OR Apache-2.0
use antecedent::analysis::recalc_composite::*;
use antecedent::analysis::recalc_external::{
    CallbackPolicy, ExternalMeanProvider, count_external_invocations,
};
use antecedent::analysis::recalc_receipt::UtilitySpec;
use antecedent::analysis::recalc_static::{
    StaticResponseRequest, StaticResponseSession, execute_static_response_with_receipt,
};
use antecedent_core::recalc::{Branch, Stage, StageStatus};
use antecedent_core::{ExecutionContext, ExternalProgramClaim, ProgramBinding, VariableId};
use antecedent_design::composition_boundary::{SupportPolicy, SupportedVerdict};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy, UtilityExpr,
};
use antecedent_graph::{Admg, DenseNodeId};
use antecedent_learn::fit_counts::observe_resolved_fits;
use std::collections::BTreeMap;
#[path = "common/external_callback_fixture.rs"]
mod reference;
fn fixture() -> (CompositeSession, CompositeRequest, reference::Provider, reference::Provider) {
    let ctx = ExecutionContext::for_tests(7);
    let mut columns: Vec<(String, Vec<f64>)> =
        ["t", "m", "y"].iter().map(|name| ((*name).into(), Vec::new())).collect();
    for (t, ones) in [20_u32, 50, 80].into_iter().enumerate() {
        for row in 0..100_u32 {
            let m = f64::from(u32::from(row < ones));
            columns[0].1.push(f64::from(u32::try_from(t).unwrap()));
            columns[1].1.push(m);
            columns[2].1.push(2.0 + 4.0 * m);
        }
    }
    let mut graph = Admg::with_variables(3);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    graph.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    let request = StaticResponseRequest {
        columns,
        graph,
        treatment: VariableId::from_raw(0),
        outcome: VariableId::from_raw(2),
        support: vec![0.0, 1.0, 2.0],
        actions: vec![0.0, 1.0, 2.0],
        baseline: 0,
        active: 1,
        utility: UtilitySpec { benefit_per_unit: 1.0, cost: 0.0 },
    };
    let mut native = StaticResponseSession::new();
    let out = execute_static_response_with_receipt(&mut native, &request, &ctx).unwrap();
    for (actual, expected) in out.means.iter().zip([2.8, 4.0, 5.2]) {
        assert!((actual - expected).abs() < 1e-12);
    }
    assert!(native.producing_receipt().unwrap().totals().factor_builds > 0);
    let result = native.result().unwrap();
    let contract = result.executed_contract.as_ref().unwrap();
    let graph_id = format!("graph:{}", contract.identities.identification);
    let binding = ProgramBinding {
        contract_id: antecedent_io::external_binding_wire::native_contract_identity(
            &graph_id,
            result.response.as_ref().unwrap().identification_status.as_str(),
            None,
        )
        .unwrap(),
        graph_id,
        treatment_id: "t".into(),
        outcome_id: "y".into(),
        population_id: "target".into(),
        intervention_kind: "do".into(),
        horizon: 0,
        dose_grid: vec![0.0, 1.0, 2.0],
        dose_units: "dimensionless".into(),
        outcome_units: "dimensionless".into(),
        functional_id: "mean".into(),
        transform_id: "identity".into(),
    };
    let mut externals = Vec::new();
    for branch in [0, 1] {
        let mut e = reference::request(branch, CallbackPolicy::Deterministic);
        e.program = binding.clone();
        e.claim = ExternalProgramClaim::declared_by(&binding);
        e.contract.graph_id.clone_from(&binding.graph_id);
        e.contract.estimand = binding.expected_quantities();
        e.descriptor.identity.request_id = binding.identity();
        e.descriptor.identity.provider_id = format!("provider-{branch}");
        externals.push(e);
    }
    let p1 = reference::Provider::new(&externals[0]);
    let p2 = reference::Provider::new(&externals[1]);
    let decision_contract = decision_contract(&binding);
    let session = CompositeSession::from_native(native, &request, &ctx).unwrap();
    (
        session,
        CompositeRequest {
            native: request,
            native_program: binding,
            externals,
            decision_contract,
            support_policy: SupportPolicy::require_all(),
            input_order: vec!["native".into(), "external.0".into(), "external.1".into()],
        },
        p1,
        p2,
    )
}
fn execute(
    session: &mut CompositeSession,
    request: &CompositeRequest,
    p1: &mut reference::Provider,
    p2: &mut reference::Provider,
) -> CompositeOutcome {
    let mut providers: BTreeMap<Branch, &mut dyn ExternalMeanProvider> = BTreeMap::new();
    providers.insert(request.externals[0].branch, p1);
    providers.insert(request.externals[1].branch, p2);
    let ((((out, fits), calls), work), ids) =
        antecedent_identify::execution_counts::count_checked_identifications(|| {
            antecedent_expr::execution_counts::count_static_work(|| {
                count_external_invocations(|| {
                    observe_resolved_fits(|| {
                        session
                            .execute(request, &mut providers, &ExecutionContext::for_tests(7))
                            .unwrap()
                    })
                })
            })
        });
    assert_eq!(fits, 0);
    assert_eq!(calls, out.receipt.totals().external_invocations);
    assert_eq!(work.factor_builds, out.receipt.totals().factor_builds);
    assert_eq!(work.integrations, out.receipt.totals().integrations);
    assert_eq!(ids, out.receipt.totals().identifications);
    let sealed = seal_receipt(&out);
    let identity = out.receipt.identity().to_hex();
    assert_eq!(sealed.receipt_identity(), identity);
    let bytes = sealed.to_bytes("composite-measured-history").unwrap();
    let consumed = antecedent_io::recalc_receipt_artifact::RecalcReceiptArtifact::from_bytes(
        &bytes,
        Some(&identity),
    )
    .unwrap();
    assert_eq!(consumed.receipt_identity(), identity);
    out
}
#[test]
fn native_two_provider_shared_receipt_and_selective_mutation() {
    let (mut session, mut request, mut p1, mut p2) = fixture();
    assert!(session.plan(&request).unwrap().is_executable());
    assert_eq!((p1.calls, p2.calls), (0, 0));
    let first = execute(&mut session, &request, &mut p1, &mut p2);
    assert_eq!(first.receipt.totals().external_invocations, 2);
    assert_eq!(first.receipt.totals().decisions, 1);
    assert_eq!(first.receipt.totals().factor_builds, 0);
    assert_eq!(
        first.decision.verdict,
        SupportedVerdict::Compared(antecedent_design::decision_eval::Verdict::UniquelyOptimal(
            "treat".into()
        ))
    );
    let native_pointer = std::ptr::from_ref(session.native().result().unwrap());
    let native_receipt = session.native().producing_receipt().unwrap().identity();
    let second = execute(&mut session, &request, &mut p1, &mut p2);
    assert_eq!(second.receipt.totals().total(), 0);
    request.decision_contract.actions[1].utility =
        UtilityExpr::difference(UtilityExpr::Input(0), UtilityExpr::Const(2.0));
    let utility = execute(&mut session, &request, &mut p1, &mut p2);
    assert_eq!(utility.receipt.totals().external_invocations, 0);
    assert_eq!(utility.receipt.totals().decisions, 1);
    assert_eq!(
        utility.decision.verdict,
        SupportedVerdict::Compared(antecedent_design::decision_eval::Verdict::UniquelyOptimal(
            "wait".into()
        ))
    );
    request.externals[0].columns[0].1 = vec![0.0, 2.0];
    let changed = execute(&mut session, &request, &mut p1, &mut p2);
    assert_eq!(changed.receipt.totals().external_invocations, 1);
    assert_eq!(changed.receipt.totals().decisions, 1);
    assert!(matches!(
        changed.plan.status(Stage::ProviderRequest(request.externals[1].branch)),
        Some(StageStatus::Reused { .. })
    ));
    assert_eq!((p1.calls, p2.calls), (2, 1));
    assert_eq!(native_pointer, std::ptr::from_ref(session.native().result().unwrap()));
    assert_eq!(native_receipt, session.native().producing_receipt().unwrap().identity());
}
#[test]
fn callback_failure_preserves_combined_native_and_foreign_success() {
    let (mut session, mut request, mut p1, mut p2) = fixture();
    execute(&mut session, &request, &mut p1, &mut p2);
    let old = request.clone();
    let native_pointer = std::ptr::from_ref(session.native().result().unwrap());
    let native_receipt = session.native().producing_receipt().unwrap().identity();
    request.externals[0].columns[0].1 = vec![0.0, 2.0];
    p1.fail = true;
    let mut providers: BTreeMap<Branch, &mut dyn ExternalMeanProvider> = BTreeMap::new();
    providers.insert(request.externals[0].branch, &mut p1);
    providers.insert(request.externals[1].branch, &mut p2);
    let (failure, calls) = count_external_invocations(|| {
        session.execute(&request, &mut providers, &ExecutionContext::for_tests(7))
    });
    assert_eq!(calls, 1);
    assert!(matches!(failure, Err(CompositeError::External { .. })));
    assert_eq!(native_pointer, std::ptr::from_ref(session.native().result().unwrap()));
    assert_eq!(native_receipt, session.native().producing_receipt().unwrap().identity());
    p1.fail = false;
    let retry = execute(&mut session, &old, &mut p1, &mut p2);
    assert_eq!(retry.receipt.totals().total(), 0);
}

#[test]
fn native_proof_substitution_refuses_purely_and_preserves_shared_state() {
    let (mut session, mut request, mut p1, mut p2) = fixture();
    execute(&mut session, &request, &mut p1, &mut p2);
    let old = request.clone();
    request.native_program.graph_id = "another-graph".into();
    for external in &mut request.externals {
        external.program = request.native_program.clone();
        external.claim = ExternalProgramClaim::declared_by(&external.program);
        external.contract.graph_id = external.program.graph_id.clone();
        external.descriptor.identity.request_id = external.program.identity();
    }
    assert!(matches!(
        session.plan(&request),
        Err(CompositeError::Request("composite_recalc.native_state_mismatch"))
    ));
    assert_eq!((p1.calls, p2.calls), (1, 1));
    let out = execute(&mut session, &old, &mut p1, &mut p2);
    assert_eq!(out.receipt.totals().total(), 0);
}

#[test]
fn native_data_refresh_counts_real_factors_and_retains_both_providers() {
    let (mut session, mut request, mut p1, mut p2) = fixture();
    execute(&mut session, &request, &mut p1, &mut p2);
    let old_snapshot = session
        .native()
        .result()
        .unwrap()
        .executed_contract
        .as_ref()
        .unwrap()
        .identities
        .data_snapshot;
    for value in &mut request.native.columns[2].1 {
        *value += 1.0;
    }
    let refreshed = execute(&mut session, &request, &mut p1, &mut p2);
    assert_eq!(refreshed.receipt.totals().external_invocations, 0);
    assert!(refreshed.receipt.totals().factor_builds > 0);
    assert!(refreshed.receipt.totals().integrations > 0);
    assert_eq!(refreshed.receipt.totals().decisions, 2);
    assert_eq!((p1.calls, p2.calls), (1, 1));
    assert_ne!(
        old_snapshot,
        session
            .native()
            .result()
            .unwrap()
            .executed_contract
            .as_ref()
            .unwrap()
            .identities
            .data_snapshot
    );
    let response = session.native().result().unwrap().response.as_ref().unwrap();
    let antecedent_core::ResponseIdentification::PointIdentified(
        antecedent_core::ResponseValue::Surface { mean: means, .. },
    ) = &response.estimate
    else {
        panic!("expected full actual finite response");
    };
    for (actual, expected) in means.iter().zip([3.8, 5.0, 6.2]) {
        assert!((actual - expected).abs() < 1e-12);
    }
}

fn decision_contract(binding: &ProgramBinding) -> DecisionContract {
    let costs = [0.0, 1.0, 3.0];
    let names = ["wait", "treat", "extend"];
    DecisionContract {
        actions: binding
            .expected_quantities()
            .into_iter()
            .enumerate()
            .map(|(i, q)| DecisionAction {
                id: names[i].into(),
                kind: ActionKind::Intervention,
                inputs: vec![q],
                utility: UtilityExpr::difference(
                    UtilityExpr::Input(0),
                    UtilityExpr::Const(costs[i]),
                ),
            })
            .collect(),
        utility_units: "net benefit".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::RequireInvariantBestAction,
    }
}

fn seal_receipt(
    out: &CompositeOutcome,
) -> antecedent_io::recalc_receipt_artifact::RecalcReceiptArtifact {
    let counts = out
        .receipt
        .entries()
        .iter()
        .map(|entry| {
            let c = entry.counts;
            (
                entry.stage,
                antecedent_io::recalc_receipt_artifact::CountsWire {
                    identifications: c.identifications,
                    fold_fits: c.fold_fits,
                    model_fits: c.model_fits,
                    score_computations: c.score_computations,
                    reweights: c.reweights,
                    decisions: c.decisions,
                    factor_builds: c.factor_builds,
                    program_compilations: c.program_compilations,
                    provider_bindings: c.provider_bindings,
                    factor_evaluations: c.factor_evaluations,
                    integrations: c.integrations,
                    provider_calls: c.provider_calls,
                    law_summaries: c.law_summaries,
                    posterior_draws: c.posterior_draws,
                    external_invocations: c.external_invocations,
                },
            )
        })
        .collect();
    antecedent_io::recalc_receipt_artifact::RecalcReceiptArtifact::seal(
        &out.previous,
        &out.requested,
        &out.capabilities,
        &counts,
    )
    .unwrap()
}

#[test]
fn declared_source_preference_selects_all_three_actual_sources_and_is_terminal_only() {
    use antecedent_core::SupportStatus;
    let (mut session, mut request, mut p1, mut p2) = fixture();
    p1.support = vec![
        SupportStatus::OutsideEmpiricalSupport,
        SupportStatus::Supported,
        SupportStatus::OutsideEmpiricalSupport,
    ];
    p2.support = vec![
        SupportStatus::OutsideEmpiricalSupport,
        SupportStatus::OutsideEmpiricalSupport,
        SupportStatus::Supported,
    ];
    request.input_order = vec!["external.0".into(), "external.1".into(), "native".into()];
    let initial = execute(&mut session, &request, &mut p1, &mut p2);
    let selected: Vec<_> =
        initial.decision.dispositions.iter().map(|x| x.input_id.as_deref()).collect();
    assert_eq!(selected, vec![Some("native"), Some("external.0"), Some("external.1")]);
    assert_eq!(
        initial.decision.verdict,
        SupportedVerdict::Compared(antecedent_design::decision_eval::Verdict::UniquelyOptimal(
            "wait".into()
        ))
    );
    request.input_order = vec!["native".into(), "external.0".into(), "external.1".into()];
    let order = execute(&mut session, &request, &mut p1, &mut p2);
    assert_eq!(order.receipt.totals().external_invocations, 0);
    assert_eq!(order.receipt.totals().factor_builds, 0);
    assert_eq!(order.receipt.totals().decisions, 1);
    request.input_order = vec!["external.0".into(), "external.1".into(), "native".into()];
    execute(&mut session, &request, &mut p1, &mut p2);
    let base = request.clone();
    for (branch, selected) in [(0, "treat"), (1, "extend")] {
        request.externals[branch].model_parameters[0].1 += 8.0;
        let out = execute(&mut session, &request, &mut p1, &mut p2);
        assert_eq!(out.receipt.totals().external_invocations, 1);
        assert_eq!(out.receipt.totals().factor_builds, 0);
        assert_eq!(
            out.decision.verdict,
            SupportedVerdict::Compared(antecedent_design::decision_eval::Verdict::UniquelyOptimal(
                selected.into()
            ))
        );
        let (mut fresh, _, _, _) = fixture();
        assert_eq!(execute(&mut fresh, &request, &mut p1, &mut p2).decision, out.decision);
        request = base.clone();
        execute(&mut session, &request, &mut p1, &mut p2);
    }
    for y in &mut request.native.columns[2].1 {
        *y -= 8.0;
    }
    let out = execute(&mut session, &request, &mut p1, &mut p2);
    assert_eq!(out.receipt.totals().external_invocations, 0);
    assert!(out.receipt.totals().factor_builds > 0);
    assert_eq!(
        out.decision.verdict,
        SupportedVerdict::Compared(antecedent_design::decision_eval::Verdict::UniquelyOptimal(
            "treat".into()
        ))
    );
    let published = request.clone();
    request.input_order = vec!["native".into(); 3];
    assert!(matches!(
        session.plan(&request),
        Err(CompositeError::Request("composite_recalc.request_unsupported"))
    ));
    let out = execute(&mut session, &published, &mut p1, &mut p2);
    assert_eq!(out.receipt.totals().total(), 0);
    // Actual sources lack this new quantity; do not synthesize support for it.
    request = published.clone();
    let mut absent = request.native_program.clone();
    absent.dose_grid = vec![99.0];
    request.decision_contract.actions[0].inputs = absent.expected_quantities();
    let mut providers: BTreeMap<Branch, &mut dyn ExternalMeanProvider> = BTreeMap::new();
    let (refused, calls) = count_external_invocations(|| {
        session.execute(&request, &mut providers, &ExecutionContext::for_tests(7))
    });
    assert!(matches!(refused, Err(CompositeError::Binding(_))));
    assert_eq!(calls, 0);
    assert_eq!(execute(&mut session, &published, &mut p1, &mut p2).receipt.totals().total(), 0);
}

fn conditional_policy() -> antecedent::analysis::conditional_study_ranking::ConditionalStudyPolicy {
    use antecedent::analysis::conditional_study_ranking::{
        ConditionalBranch, ConditionalCandidate, ConditionalStudyPolicy,
    };
    ConditionalStudyPolicy {
        policy_id: "conditional-study-v1".into(),
        branches: ["wait", "treat", "extend"]
            .into_iter()
            .map(|action| ConditionalBranch {
                action: action.into(),
                candidates: vec![
                    ConditionalCandidate {
                        semantic_id: format!("{action}-insufficient-cheap"),
                        verified_sufficient: false,
                        cost_units: 0,
                        sample_budget: 1,
                    },
                    ConditionalCandidate {
                        semantic_id: format!("{action}-sufficient-expensive"),
                        verified_sufficient: true,
                        cost_units: 20,
                        sample_budget: 100,
                    },
                    ConditionalCandidate {
                        semantic_id: format!("{action}-sufficient-cheap"),
                        verified_sufficient: true,
                        cost_units: 10,
                        sample_budget: 200,
                    },
                ],
            })
            .collect(),
    }
}
#[test]
fn actual_combined_decision_routes_original_structural_ranking_and_fresh_source_consumption() {
    use antecedent::analysis::conditional_study_ranking::ConditionalStudyRanking;
    use antecedent_core::SupportStatus;
    let (mut session, mut request, mut p1, mut p2) = fixture();
    request.input_order = vec!["external.0".into(), "external.1".into(), "native".into()];
    p1.support = vec![
        SupportStatus::OutsideEmpiricalSupport,
        SupportStatus::Supported,
        SupportStatus::OutsideEmpiricalSupport,
    ];
    p2.support = vec![
        SupportStatus::OutsideEmpiricalSupport,
        SupportStatus::OutsideEmpiricalSupport,
        SupportStatus::Supported,
    ];
    let no_source = ConditionalStudyRanking::execute(&session, conditional_policy());
    assert_eq!(no_source.err().unwrap().detail, "conditional_study_ranking.source_unavailable");
    execute(&mut session, &request, &mut p1, &mut p2);
    let before_calls = (p1.calls, p2.calls);
    let before = ConditionalStudyRanking::execute(&session, conditional_policy()).unwrap();
    assert_eq!(before.selected_action(), "wait");
    let ids: Vec<_> = before.entries().iter().map(|e| e.semantic_id.as_str()).collect();
    assert_eq!(
        ids,
        vec!["wait-sufficient-cheap", "wait-sufficient-expensive", "wait-insufficient-cheap"]
    );
    assert_eq!((p1.calls, p2.calls), before_calls);
    let bytes = before.to_bytes().unwrap();
    let inspected: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(inspected["source"]["input_order"][0], "external.0");
    assert_eq!(inspected["policy"]["branches"].as_array().unwrap().len(), 3);
    assert!(inspected["historical_receipt"].as_array().unwrap().len() > 100);
    let (mut fresh, _, _, _) = fixture();
    execute(&mut fresh, &request, &mut p1, &mut p2);
    let replay = ConditionalStudyRanking::consume(&bytes, &fresh, Some(before.identity())).unwrap();
    assert_eq!(replay.entries(), before.entries());
    // A real upstream provider update changes terminal action and the actual candidate table.
    request.externals[0].model_parameters[0].1 += 8.;
    let update = execute(&mut session, &request, &mut p1, &mut p2);
    assert_eq!(update.receipt.totals().external_invocations, 1);
    let after = ConditionalStudyRanking::execute(&session, conditional_policy()).unwrap();
    assert_eq!(after.selected_action(), "treat");
    assert_eq!(after.entries()[0].semantic_id, "treat-sufficient-cheap");
    assert_ne!(after.identity(), before.identity());
    assert_eq!(
        ConditionalStudyRanking::consume(&bytes, &session, None).err().unwrap().detail,
        "conditional_study_ranking.source_mismatch"
    );
    // Resealing altered scientific source fields still cannot match issued current state.
    let mut mutated: serde_json::Value =
        serde_json::from_slice(&after.to_bytes().unwrap()).unwrap();
    mutated["source"]["selected_action"] = serde_json::json!("extend");
    mutated["identity"] = serde_json::json!("");
    let digest = blake3::hash(&serde_json::to_vec(&mutated).unwrap()).to_hex().to_string();
    mutated["identity"] = serde_json::json!(digest);
    // Arbitrary JSON canonicalization also refuses; the independent consumer owns its canonical format.
    assert!(
        ConditionalStudyRanking::consume(&serde_json::to_vec(&mutated).unwrap(), &session, None)
            .is_err()
    );
    let mut incomplete = conditional_policy();
    incomplete.branches.pop();
    assert_eq!(
        ConditionalStudyRanking::execute(&session, incomplete).err().unwrap().detail,
        "conditional_study_ranking.invalid_policy"
    );
    let mut policy = conditional_policy();
    policy.policy_id = "conditional-study-v2".into();
    policy.branches[1].candidates[0].verified_sufficient = true;
    let rerank = ConditionalStudyRanking::execute(&session, policy).unwrap();
    assert_eq!(rerank.entries()[0].semantic_id, "treat-insufficient-cheap");
    assert_ne!(rerank.identity(), after.identity());
}
