//! The opt-in descriptive fallback: an identification-class refusal is returned unchanged beside
//! a compiled, labelled-noncausal association and the evidence that would resolve it.
//!
//! The fixtures are the two structural failures an observational study meets: a bow (an
//! unmeasured common cause of the treatment and the outcome, which no adjustment closes) and a
//! hedge (the effect passes through a mediator that shares an unmeasured cause with the
//! outcome). Both are refused by the complete identification algorithm. A successful causal
//! query never produces a fallback, a budget or cancellation is never converted, the
//! association never ranks actions, and a query kind with no association form says so.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    reason = "fixture indices convert exactly into the f64 columns"
)]

use std::sync::Arc;

use antecedent::analysis::evidence_plan::{EvidencePlanRequest, STUDY_POPULATION};
use antecedent::analysis::fallback::{
    FallbackOutcome, FallbackPart, FallbackRequest, RankedPlan, fallback_roles,
};
use antecedent::host::{HostStatus, fallback_response, host_schema};
use antecedent::report::{ReportFormat, render_fallback_report};
use antecedent::{
    CausalError, IDENTIFICATION_REFUSAL_CODES, InferenceMode, RefuteSuite, Study, StudyResult,
};
use antecedent_contract::{ClaimKind, DomainStatus};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, DistributionAvailability, ExecutionContext,
    InterventionalDistributionQuery, SearchLimits, VariableId,
};
use antecedent_data::TabularData;
use antecedent_design::{
    DurableStudyCandidate, ExpectedEvidence, RepairClassification, RepairLimits, RepairObjective,
    StudyCostDeclaration, StudyKind, UnitRules,
};
use antecedent_graph::{Admg, DenseNodeId};

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn node(raw: u32) -> DenseNodeId {
    DenseNodeId::from_raw(raw)
}

fn ctx(seed: u64) -> ExecutionContext {
    ExecutionContext::for_tests(seed)
}

/// Deterministic pseudo-noise in `[-0.5, 0.5)`.
fn noise(i: usize) -> f64 {
    ((i as f64 * 12.9898).sin() * 43758.5453).fract().abs() - 0.5
}

/// Three columns `z`, `x`, `y`: `x` is a 0/1 treatment that depends on `z`, `y` follows both.
fn zxy() -> TabularData {
    let n = 160;
    let z: Vec<f64> = (0..n).map(|i| f64::from(u8::from(noise(i) > 0.0))).collect();
    let x: Vec<f64> = (0..n).map(|i| f64::from(u8::from(z[i] + noise(i + 500) > 0.3))).collect();
    let y: Vec<f64> = (0..n).map(|i| 1.0 + 2.0 * x[i] + z[i] + noise(i + 900)).collect();
    TabularData::from_f64_columns([("z", &z[..]), ("x", &x[..]), ("y", &y[..])]).unwrap()
}

/// `z -> x`, `z -> y`, `x -> y` and `x <-> y`: the back door through `z` is closed by
/// adjustment, but the unmeasured common cause of `x` and `y` is not, so nothing identifies the
/// effect.
fn confounded() -> Admg {
    let mut graph = Admg::with_variables(3);
    graph.insert_directed(node(0), node(1)).unwrap();
    graph.insert_directed(node(0), node(2)).unwrap();
    graph.insert_directed(node(1), node(2)).unwrap();
    graph.insert_bidirected(node(1), node(2)).unwrap();
    graph
}

fn confounded_study(data: TabularData) -> Study {
    Study::tabular(data)
        .graph(confounded())
        .query(AverageEffectQuery::binary_ate(v(1), v(2)))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
}

/// Columns `t`, `w`, `y`: `t -> w -> y` with `t <-> y` and `w <-> y`, all observed: a hedge.
fn twy() -> TabularData {
    let n = 160;
    let t: Vec<f64> = (0..n).map(|i| f64::from(u8::from(noise(i) > 0.0))).collect();
    let w: Vec<f64> = (0..n).map(|i| f64::from(u8::from(t[i] + noise(i + 300) > 0.4))).collect();
    let y: Vec<f64> = (0..n).map(|i| 0.5 + 1.5 * w[i] + noise(i + 700)).collect();
    TabularData::from_f64_columns([("t", &t[..]), ("w", &w[..]), ("y", &y[..])]).unwrap()
}

fn hedge() -> Admg {
    let mut graph = Admg::with_variables(3);
    graph.insert_directed(node(0), node(1)).unwrap();
    graph.insert_directed(node(1), node(2)).unwrap();
    graph.insert_bidirected(node(0), node(2)).unwrap();
    graph.insert_bidirected(node(1), node(2)).unwrap();
    graph
}

fn hedge_study(data: TabularData) -> Study {
    Study::tabular(data)
        .graph(hedge())
        .query(AverageEffectQuery::binary_ate(v(0), v(2)))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
}

fn described(outcome: FallbackOutcome) -> antecedent::AnalysisWithFallback {
    match outcome {
        FallbackOutcome::Described(fallback) => *fallback,
        other => panic!("expected a described refusal, got {other:?}"),
    }
}

fn candidate(
    label: &str,
    kind: StudyKind,
    interventions: &[u32],
    measured: &[u32],
    cost: u64,
) -> DurableStudyCandidate {
    let ids = |raw: &[u32]| raw.iter().copied().map(v).collect::<Arc<[VariableId]>>();
    DurableStudyCandidate {
        label: Arc::from(label),
        kind,
        population: Arc::from(STUDY_POPULATION),
        interventions: ids(interventions),
        measured: ids(measured),
        joint_measurement: true,
        sample_size: 100,
        recruitment: Arc::from("consecutive recruitment"),
        timing: Arc::from("baseline"),
        unit_rules: UnitRules {
            unit: Arc::from("patient"),
            cluster: None,
            whole_cluster_sampling: false,
        },
        cost: StudyCostDeclaration {
            units: cost,
            unit_label: Arc::from("usd"),
            sample_budget: 100,
        },
        feasible: true,
        feasibility_notes: Arc::from([]),
        expected_evidence: Arc::from([ExpectedEvidence {
            population: Arc::from(STUDY_POPULATION),
            interventions: ids(interventions),
            intervention_values: Arc::from([]),
            conditioned_on: Arc::from([]),
            measured: ids(measured),
            distribution: DistributionAvailability::Joint,
        }]),
        external_provider: None,
    }
}

fn plan_request(candidates: Vec<DurableStudyCandidate>) -> EvidencePlanRequest {
    EvidencePlanRequest {
        candidates,
        objective: RepairObjective::MinimizeCost,
        limits: RepairLimits {
            search: SearchLimits { operations: 1000, depth: 2 },
            memory_limit_bytes: 16 * 1024 * 1024,
        },
        sensitivity: None,
        value: None,
    }
}

#[test]
fn an_unmeasured_confounder_returns_the_refusal_a_noncausal_description_and_the_evidence_needs() {
    let study = confounded_study(zxy());
    let direct = study.run(&ctx(1)).unwrap_err();
    let fallback =
        described(study.analyze_with_fallback(&FallbackRequest::default(), &ctx(1)).unwrap());

    // The refusal is the one the causal query raises, unchanged.
    assert_eq!(fallback.refusal.reason_code(), Some("effect_not_identified"));
    assert_eq!(fallback.refusal.to_string(), direct.to_string());
    assert!(fallback.refusal.is_identification_refusal());

    // The description is the compiled associational route: the outcome on the treatment,
    // holding the measured parent of the treatment fixed, with the refusal retained.
    let part = fallback.associational.as_ref().expect("an association form exists");
    assert_eq!(part.roles.treatment, v(1));
    assert_eq!(part.roles.outcome, v(2));
    assert_eq!(part.roles.adjustment, vec![v(0)]);
    assert_eq!(part.claim.kind, ClaimKind::Associational);
    assert!(!part.claim.kind.is_causal());
    assert_eq!(part.claim.domains.identification, DomainStatus::NotClaimed);
    let payload = part.result.association().expect("an association payload");
    assert_eq!(payload.refused_after.as_deref(), Some("effect_not_identified"));
    assert!(part.result.estimate().as_effect().is_none(), "an association is no effect estimate");
    assert_eq!(
        payload.estimate.terms.len(),
        2,
        "intercept and treatment; the held-fixed parent is not reported"
    );
    assert!(!fallback.ranks_actions());
    assert!(!fallback.answers_causal_question());

    // The evidence needs: verified obligations and evidence-gathering next actions.
    let needs = fallback.evidence.as_ref().expect("the graph is a directed or bidirected one");
    assert!(!needs.obligations.is_empty());
    assert!(needs.obligations.iter().all(|o| o.provenance.proof_step.is_some()));
    let actions: Vec<&str> = needs.next_actions.iter().map(|a| a.id.as_ref()).collect();
    assert!(actions.contains(&"run_experiment:x"), "{actions:?}");
    assert!(actions.contains(&"measure_variable:common_cause_of_x_y"), "{actions:?}");
    assert!(matches!(needs.ranking, RankedPlan::NotRequested));
    assert!(!needs.candidate_templates.templates.is_empty());

    // The text says plainly that the description is not an answer to the causal question.
    let report = render_fallback_report(&fallback, ReportFormat::Markdown);
    assert!(report.contains("not an answer to the causal question"), "{report}");
    // Markdown escapes the underscores of the reason code.
    assert!(report.contains(r"effect\_not\_identified"), "{report}");
    assert!(fallback.explain().contains("not an answer to the causal question"));
}

#[test]
fn a_hedge_returns_the_refusal_the_description_and_a_ranked_plan_over_declared_studies() {
    let study = hedge_study(twy());
    let trial = candidate("randomized trial", StudyKind::Experiment, &[0], &[2], 9);
    let survey = candidate("bigger survey", StudyKind::Observation, &[], &[0, 1, 2], 1);
    let request = FallbackRequest { evidence: Some(plan_request(vec![trial, survey])) };
    let fallback = described(study.analyze_with_fallback(&request, &ctx(2)).unwrap());

    assert_eq!(fallback.refusal.reason_code(), Some("effect_not_identified"));
    let part = fallback.associational.as_ref().unwrap();
    assert!(part.roles.adjustment.is_empty(), "the treatment has no measured parent");
    assert_eq!(part.claim.kind, ClaimKind::Associational);
    assert_eq!(part.result.association().unwrap().estimate.terms.len(), 2);

    let needs = fallback.evidence.as_ref().unwrap();
    let RankedPlan::Ranked(plan) = &needs.ranking else {
        panic!("candidates were declared, so the plan is ranked: {:?}", needs.ranking)
    };
    let best = &plan.plan.options[0];
    assert_eq!(best.labels[0].as_ref(), "randomized trial");
    assert_eq!(best.gate, RepairClassification::VerifiedSufficient);
    // The cheap survey cannot answer a hedge, whatever it costs.
    assert!(plan.plan.options.iter().any(|option| {
        option.labels[0].as_ref() == "bigger survey"
            && option.gate != RepairClassification::VerifiedSufficient
    }));
    let actions: Vec<&str> = needs.next_actions.iter().map(|a| a.id.as_ref()).collect();
    assert!(actions.iter().any(|id| id.starts_with("run_experiment:")), "{actions:?}");
    // A study supplies data, never an assumption: an assumption obligation is not satisfiable.
    assert!(
        needs
            .obligations
            .iter()
            .filter(|o| o.kind == antecedent_contract::EvidenceObligationKind::EstablishAssumption)
            .all(|o| !o.satisfiable_by_study())
    );
}

#[test]
fn a_successful_causal_query_returns_the_ordinary_result_and_never_a_fallback() {
    let x: Vec<f64> = (0..120).map(|i| f64::from(u8::from(noise(i) > 0.0))).collect();
    let y: Vec<f64> = (0..120).map(|i| 1.0 + 2.0 * x[i] + noise(i + 40)).collect();
    let data = TabularData::from_f64_columns([("x", &x[..]), ("y", &y[..])]).unwrap();
    let mut graph = Admg::with_variables(2);
    graph.insert_directed(node(0), node(1)).unwrap();
    let study = Study::tabular(data)
        .graph(graph)
        .query(AverageEffectQuery::binary_ate(v(0), v(1)))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let outcome = study.analyze_with_fallback(&FallbackRequest::default(), &ctx(3)).unwrap();
    let FallbackOutcome::Causal(result) = outcome else {
        panic!("an identified query must return its ordinary result");
    };
    let ordinary: StudyResult = study.run(&ctx(3)).unwrap();
    assert!(result.estimate().as_effect().is_some());
    assert!(result.association().is_none());
    assert_eq!(
        result.estimate().as_effect().unwrap().ate,
        ordinary.estimate().as_effect().unwrap().ate
    );
}

#[test]
fn only_identification_class_refusals_are_converted() {
    let status = antecedent_core::IdentificationStatus::NotIdentified;
    assert!(CausalError::not_identified(status, false, "bow").is_identification_refusal());
    // A search that stopped at a budget is not a proof of non-identification.
    assert!(!CausalError::not_identified(status, true, "capped").is_identification_refusal());
    assert!(!CausalError::Cancelled { stage: "identify" }.is_identification_refusal());
    assert!(!CausalError::Resource { message: "memory".into() }.is_identification_refusal());
    assert!(!CausalError::Unsupported { message: "unsupported" }.is_identification_refusal());
    assert!(IDENTIFICATION_REFUSAL_CODES.contains(&"effect_not_identified"));

    // A cancelled context is returned as the cancellation it is, never described.
    let study = confounded_study(zxy());
    let cancelled = ctx(4);
    cancelled.cancellation.cancel();
    let error = study
        .analyze_with_fallback(&FallbackRequest::default(), &cancelled)
        .expect_err("a cancellation is not a fallback");
    assert!(matches!(error.peeled(), CausalError::Cancelled { .. }), "{error}");
}

#[test]
fn a_query_with_no_association_form_says_so_in_a_typed_reason() {
    let study = Study::tabular(zxy())
        .graph(confounded())
        .query(CausalQuery::Distribution(InterventionalDistributionQuery::new(
            v(2),
            [antecedent_core::Intervention::set(v(1), antecedent_core::Value::f64(1.0))],
        )))
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let unavailable = fallback_roles(&study).unwrap_err();
    assert_eq!(unavailable.part, FallbackPart::Associational);
    assert_eq!(unavailable.code, "route_not_supported");
    assert_eq!(unavailable.detail, "fallback.no_association_form");
    assert!(unavailable.message.contains("Next attempt"), "{unavailable}");
}

#[test]
fn the_host_record_shows_the_causal_refusal_and_the_noncausal_status_side_by_side() {
    let study = confounded_study(zxy());
    let fallback =
        described(study.analyze_with_fallback(&FallbackRequest::default(), &ctx(5)).unwrap());
    let record = fallback_response(&fallback);
    assert_eq!(record["status"], "unsupported");
    assert_eq!(record["causal_status"], "unsupported");
    assert_eq!(record["associational_status"], HostStatus::Supported.as_str());
    assert_eq!(record["refusal"]["reason_code"], "effect_not_identified");
    assert_eq!(record["associational"]["claim"]["kind"], "associational");
    assert_eq!(record["associational"]["claim"]["causal"], false);
    assert_eq!(record["associational"]["ranks_actions"], false);
    assert!(record["evidence_plan"]["obligations"].as_array().is_some_and(|o| !o.is_empty()));
    let schema = host_schema().unwrap();
    for key in schema["$defs"]["fallback_response"]["required"].as_array().unwrap() {
        assert!(record.get(key.as_str().unwrap()).is_some(), "{key}");
    }
}

#[test]
fn a_bayesian_study_is_described_with_its_posterior_and_the_same_refusal() {
    let study = Study::tabular(zxy())
        .graph(confounded())
        .query(AverageEffectQuery::binary_ate(v(1), v(2)))
        .inference(InferenceMode::Bayesian(antecedent::BayesianConfig::conjugate().n_draws(200)))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let fallback =
        described(study.analyze_with_fallback(&FallbackRequest::default(), &ctx(6)).unwrap());
    assert_eq!(fallback.refusal.reason_code(), Some("effect_not_identified"));
    let payload = fallback.associational.as_ref().unwrap().result.association().unwrap();
    assert!(payload.estimate.posterior.is_some(), "the Bayesian reading carries its posterior");
}

#[test]
fn a_completed_search_refusal_points_at_the_option_and_a_capped_one_does_not() {
    let status = antecedent_core::IdentificationStatus::NotIdentified;
    let completed = CausalError::not_identified(status, false, "bow").to_string();
    assert!(completed.contains("analyze_with_fallback"), "{completed}");
    assert!(completed.contains("on_refusal=\"describe\""), "{completed}");
    assert!(completed.contains("not an answer to the causal question"), "{completed}");
    // A capped search is not converted, so its text offers nothing it will not deliver.
    let capped = CausalError::not_identified(status, true, "capped").to_string();
    assert!(!capped.contains("analyze_with_fallback"), "{capped}");

    // The refusal a real study raises says the same thing, with and without the option.
    let study = confounded_study(zxy());
    let direct = study.run(&ctx(5)).unwrap_err();
    assert!(direct.to_string().contains("analyze_with_fallback"), "{direct}");
    let fallback =
        described(study.analyze_with_fallback(&FallbackRequest::default(), &ctx(5)).unwrap());
    assert_eq!(fallback.refusal.to_string(), direct.to_string());
}

#[test]
fn without_the_option_a_refusal_and_a_success_are_reproduced_exactly() {
    // The plain path is the one the option wraps: two runs of the same refused study raise the
    // same bytes, and a successful one never gains a fallback or an association.
    let study = hedge_study(twy());
    let first = study.run(&ctx(6)).unwrap_err().to_string();
    let second = study.run(&ctx(6)).unwrap_err().to_string();
    assert_eq!(first, second);
    let wrapped =
        described(study.analyze_with_fallback(&FallbackRequest::default(), &ctx(6)).unwrap());
    assert_eq!(wrapped.refusal.to_string(), first);

    let x: Vec<f64> = (0..120).map(|i| f64::from(u8::from(noise(i) > 0.0))).collect();
    let y: Vec<f64> = (0..120).map(|i| 1.0 + 2.0 * x[i] + noise(i + 40)).collect();
    let data = TabularData::from_f64_columns([("x", &x[..]), ("y", &y[..])]).unwrap();
    let mut graph = Admg::with_variables(2);
    graph.insert_directed(node(0), node(1)).unwrap();
    let identified = Study::tabular(data)
        .graph(graph)
        .query(AverageEffectQuery::binary_ate(v(0), v(1)))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let plain = identified.run(&ctx(7)).unwrap();
    let FallbackOutcome::Causal(with_option) =
        identified.analyze_with_fallback(&FallbackRequest::default(), &ctx(7)).unwrap()
    else {
        panic!("an identified query is never described");
    };
    assert_eq!(
        plain.estimate().as_effect().unwrap().ate.to_bits(),
        with_option.estimate().as_effect().unwrap().ate.to_bits()
    );
    assert!(with_option.association().is_none());
}

#[test]
fn the_association_ranks_no_action_for_either_fixture() {
    let trial = candidate("randomized trial", StudyKind::Experiment, &[0], &[2], 9);
    let request = FallbackRequest { evidence: Some(plan_request(vec![trial])) };
    for study in [confounded_study(zxy()), hedge_study(twy())] {
        let fallback = described(study.analyze_with_fallback(&request, &ctx(8)).unwrap());
        assert!(!fallback.ranks_actions());
        assert!(!fallback.answers_causal_question());
        let part = fallback.associational.as_ref().expect("an association form exists");
        assert!(!part.claim.kind.is_causal());
        assert!(part.result.estimate().as_effect().is_none());
        // Only the evidence plan orders anything, and it orders studies, not actions on the
        // outcome; the host record states the association's own flag.
        let record = fallback_response(&fallback);
        assert_eq!(record["associational"]["ranks_actions"], false);
    }
}

#[test]
fn the_guide_and_the_manifest_rows_for_the_option_are_present() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let read = |path: &str| std::fs::read_to_string(root.join(path)).expect(path);
    let guide = read("docs/guides/fallback.md");
    assert!(guide.contains("on_refusal=\"describe\"") && guide.contains("analyze_with_fallback"));
    let surface = read("parity/python_facade_surface.toml");
    for row in [
        "antecedent::report::render_fallback_report",
        "antecedent::host::fallback_response",
        "antecedent::analysis::fallback::fallback_roles",
    ] {
        assert!(surface.contains(row), "{row}");
    }
    let products = read("parity/python_products.toml");
    assert!(products.contains("name = \"on_refusal\""));
    assert!(read("python/antecedent/__init__.py").contains("fallback"));
}
