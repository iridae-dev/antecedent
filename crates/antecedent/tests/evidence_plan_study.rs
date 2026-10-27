//! The evidence plan of a real study: a causal query the declared structure does not identify
//! is refused unchanged, and the not-identified product the compiled program retained is what
//! the evidence needs derive from.
//!
//! Preparation of such a study raises the refusal, so no prepared handle exists to read the
//! failure from. Compile does not raise it: the lowered program defers its identification and
//! keeps the not-identified product beside the reason. The fixtures are the structural failures
//! an observational study meets, each driven through `Study::tabular(..).graph(admg)
//! .query(AverageEffect..)`:
//!
//! * a bow with a measured parent (an unmeasured common cause of treatment and outcome, which a
//!   study that measures it closes by adjustment);
//! * a hedge whose mediator shares an unmeasured cause with the outcome (the front-door route
//!   is repaired by measuring that cause, the treatment's own cause stays unmeasured, and an
//!   experiment on the treatment answers it whatever is measured);
//! * an instrument graph (the instrument is measured, yet validity is an assumption no study
//!   establishes);
//! * an identified study, which owes no evidence at all.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    reason = "fixture indices convert exactly into the f64 columns"
)]

use std::sync::Arc;

use antecedent::analysis::evidence_plan::{
    EvidencePlanArtifact, EvidencePlanFacadeError, EvidencePlanRequest, EvidencePlanResult,
    PlanOutcome, STUDY_POPULATION,
};
use antecedent::{CausalError, RefuteSuite, Study};
use antecedent_contract::EvidenceObligationKind;
use antecedent_core::{
    AverageEffectQuery, DistributionAvailability, ExecutionContext, SearchLimits, VariableId,
};
use antecedent_data::TabularData;
use antecedent_design::{
    DurableStudyCandidate, ExpectedEvidence, RepairClassification, RepairConsumeLimits,
    RepairLimits, RepairObjective, StudyCostDeclaration, StudyKind, UnitRules, latent_expansion,
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

/// Columns `z`, `x`, `y`: `x` is a 0/1 treatment that depends on `z`, `y` follows both.
fn zxy() -> TabularData {
    let n = 160;
    let z: Vec<f64> = (0..n).map(|i| f64::from(u8::from(noise(i) > 0.0))).collect();
    let x: Vec<f64> = (0..n).map(|i| f64::from(u8::from(z[i] + noise(i + 500) > 0.3))).collect();
    let y: Vec<f64> = (0..n).map(|i| 1.0 + 2.0 * x[i] + z[i] + noise(i + 900)).collect();
    TabularData::from_f64_columns([("z", &z[..]), ("x", &x[..]), ("y", &y[..])]).unwrap()
}

/// Columns `t`, `w`, `y`: `t` binary, `w` follows it, `y` follows `w`.
fn twy() -> TabularData {
    let n = 160;
    let t: Vec<f64> = (0..n).map(|i| f64::from(u8::from(noise(i) > 0.0))).collect();
    let w: Vec<f64> = (0..n).map(|i| f64::from(u8::from(t[i] + noise(i + 300) > 0.4))).collect();
    let y: Vec<f64> = (0..n).map(|i| 0.5 + 1.5 * w[i] + noise(i + 700)).collect();
    TabularData::from_f64_columns([("t", &t[..]), ("w", &w[..]), ("y", &y[..])]).unwrap()
}

fn admg(nodes: u32, directed: &[(u32, u32)], bidirected: &[(u32, u32)]) -> Admg {
    let mut graph = Admg::with_variables(nodes);
    for (from, to) in directed {
        graph.insert_directed(node(*from), node(*to)).unwrap();
    }
    for (a, b) in bidirected {
        graph.insert_bidirected(node(*a), node(*b)).unwrap();
    }
    graph
}

/// `z -> x`, `z -> y`, `x -> y` and `x <-> y`: the back door through `z` is closed by
/// adjustment, the unmeasured common cause of `x` and `y` is not.
fn bow() -> Admg {
    admg(3, &[(0, 1), (0, 2), (1, 2)], &[(1, 2)])
}

/// `z -> x -> y` and `x <-> y`: an instrument `z` that is measured, an effect no
/// nonparametric argument identifies.
fn instrument() -> Admg {
    admg(3, &[(0, 1), (1, 2)], &[(1, 2)])
}

/// `t -> w -> y` with `t <-> y` and `w <-> y`.
fn hedge() -> Admg {
    admg(3, &[(0, 1), (1, 2)], &[(0, 2), (1, 2)])
}

fn study(data: TabularData, graph: Admg, treatment: u32, outcome: u32) -> Study {
    Study::tabular(data)
        .graph(graph)
        .query(AverageEffectQuery::binary_ate(v(treatment), v(outcome)))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
}

/// The id of the unmeasured common cause of the bidirected edge `a <-> b`.
fn common_cause(graph: &Admg, a: u32, b: u32) -> VariableId {
    let (_, latents) = latent_expansion(graph).unwrap();
    latents
        .iter()
        .find(|latent| latent.between == (v(a), v(b)) || latent.between == (v(b), v(a)))
        .expect("the bidirected edge has a latent cause")
        .variable
}

fn candidate(
    label: &str,
    kind: StudyKind,
    interventions: &[VariableId],
    measured: &[VariableId],
    cost: u64,
) -> DurableStudyCandidate {
    let ids = |raw: &[VariableId]| raw.iter().copied().collect::<Arc<[VariableId]>>();
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

fn request(candidates: Vec<DurableStudyCandidate>) -> EvidencePlanRequest {
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

fn action_ids(study: &Study, ctx: &ExecutionContext) -> Vec<String> {
    let (_, actions) = study.evidence_obligations(ctx).unwrap();
    actions.iter().map(|action| action.id.to_string()).collect()
}

/// The refusal is the one preparation and a run have always raised, and compile leaves the
/// identification deferred: retaining the not-identified product weakens nothing.
fn assert_refusal_unchanged(study: &Study) {
    let run = study.run(&ctx(1)).unwrap_err();
    assert_eq!(run.reason_code(), Some("effect_not_identified"));
    assert!(run.is_identification_refusal());
    let Err(prepare) = study.prepare(&ctx(1)) else {
        panic!("preparing a study whose effect is not identified must refuse");
    };
    assert_eq!(prepare.reason_code(), Some("effect_not_identified"));
    assert_eq!(prepare.to_string(), run.to_string());
    assert!(matches!(prepare.peeled(), CausalError::NotIdentified { .. }), "{prepare}");
    let plan = study.compile(&ctx(1)).unwrap();
    let lowered = plan.lowered().expect("compile lowers every route");
    assert!(!lowered.identification_retained(), "a refused identification is not an installed one");
}

/// Verified options come first, and inside that gate the declared cost orders them.
fn assert_gate_then_cost(result: &EvidencePlanResult) {
    let options = &result.plan.options;
    let first_unsure = options
        .iter()
        .position(|option| option.gate != RepairClassification::VerifiedSufficient)
        .unwrap_or(options.len());
    assert!(
        options[first_unsure..]
            .iter()
            .all(|option| option.gate != RepairClassification::VerifiedSufficient),
        "a verified option is ranked below an unverified one"
    );
    let costs: Vec<u64> = options[..first_unsure].iter().map(|option| option.cost.units).collect();
    assert!(costs.windows(2).all(|pair| pair[0] <= pair[1]), "{costs:?}");
    for (rank, option) in options.iter().enumerate() {
        assert_eq!(option.rank, rank);
    }
}

fn assert_artifact_round_trip(result: &EvidencePlanResult, ctx: &ExecutionContext) {
    let bytes = result.export("evidence-plan-study").unwrap();
    let artifact = EvidencePlanArtifact::from_bytes(&bytes).unwrap();
    let replayed = artifact.consume(RepairConsumeLimits::default(), ctx).unwrap();
    assert_eq!(replayed.identity(), result.plan.identity());
}

#[test]
fn an_unmeasured_confounder_is_closed_by_the_study_that_measures_it() {
    let study = study(zxy(), bow(), 1, 2);
    assert_refusal_unchanged(&study);

    // The evidence needs come from the retained product: every obligation keeps its proof step
    // and an assumption obligation is satisfiable by no study.
    let (obligations, _) = study.evidence_obligations(&ctx(2)).unwrap();
    assert!(!obligations.is_empty());
    assert!(obligations.iter().all(|o| o.provenance.proof_step.is_some()));
    assert!(
        obligations
            .iter()
            .filter(|o| o.kind == EvidenceObligationKind::EstablishAssumption)
            .all(|o| !o.satisfiable_by_study())
    );
    let actions = action_ids(&study, &ctx(2));
    assert!(actions.contains(&"measure_variable:common_cause_of_x_y".to_owned()), "{actions:?}");
    assert!(actions.contains(&"run_experiment:x".to_owned()), "{actions:?}");

    // The templates are unfinished studies: what to collect, never at what cost.
    let templates = study.candidate_templates(None, &ctx(2)).unwrap();
    assert!(!templates.templates.is_empty());
    assert!(templates.templates.iter().any(|t| t.kind == StudyKind::Experiment));
    assert!(templates.templates.iter().any(|t| t.kind == StudyKind::Observation));

    // The declared studies: a survey of the measured columns, an assay of the common cause and a
    // trial. Only the study that measures the cause closes the back door by adjustment.
    let graph = bow();
    let cause = common_cause(&graph, 1, 2);
    let survey = candidate("survey", StudyKind::Observation, &[], &[v(0), v(1), v(2)], 1);
    let assay = candidate("assay", StudyKind::Observation, &[], &[v(0), v(1), v(2), cause], 5);
    let trial = candidate("trial", StudyKind::Experiment, &[v(1)], &[v(2)], 20);
    let result = study.evidence_plan(&request(vec![survey, assay, trial]), &ctx(3)).unwrap();
    assert_eq!(result.name_of(cause), "common_cause_of_x_y");
    assert_eq!(result.plan.outcome, PlanOutcome::Closing);
    let best = &result.plan.options[0];
    assert_eq!(best.gate, RepairClassification::VerifiedSufficient);
    assert_eq!(best.labels[0].as_ref(), "assay");
    assert!(result.plan.options.iter().any(|option| {
        option.route == "backdoor"
            && option.labels[0].as_ref() == "assay"
            && option.gate == RepairClassification::VerifiedSufficient
    }));
    assert!(
        result
            .plan
            .options
            .iter()
            .filter(|option| option.labels[0].as_ref() == "survey")
            .all(|option| option.gate != RepairClassification::VerifiedSufficient),
        "measuring only what is measured today cannot close an unmeasured confounder"
    );
    assert_gate_then_cost(&result);
    assert_artifact_round_trip(&result, &ctx(3));
}

#[test]
fn a_front_door_route_is_repaired_by_measuring_the_cause_of_the_mediator() {
    let graph = hedge();
    let study = study(twy(), graph.clone(), 0, 2);
    assert_refusal_unchanged(&study);

    let cause_of_outcome_and_mediator = common_cause(&graph, 1, 2);
    let actions = action_ids(&study, &ctx(4));
    assert!(actions.contains(&"measure_variable:common_cause_of_w_y".to_owned()), "{actions:?}");

    // The mediator is measured today, the treatment's own unmeasured cause stays unmeasured:
    // the back door stays open, the front door closes once the mediator's cause is measured.
    let cohort = candidate(
        "mediator cohort",
        StudyKind::Observation,
        &[],
        &[v(0), v(1), v(2), cause_of_outcome_and_mediator],
        4,
    );
    let survey = candidate("survey", StudyKind::Observation, &[], &[v(0), v(1), v(2)], 1);
    let result = study.evidence_plan(&request(vec![survey, cohort]), &ctx(5)).unwrap();
    assert_eq!(result.name_of(cause_of_outcome_and_mediator), "common_cause_of_w_y");
    let front_door = result
        .plan
        .options
        .iter()
        .find(|option| {
            option.route == "frontdoor" && option.labels[0].as_ref() == "mediator cohort"
        })
        .expect("the front-door route screens the declared cohort");
    assert_eq!(front_door.gate, RepairClassification::VerifiedSufficient);
    assert!(
        result
            .plan
            .options
            .iter()
            .filter(|option| option.route == "backdoor")
            .all(|option| option.gate != RepairClassification::VerifiedSufficient),
        "the treatment's unmeasured cause keeps the back door open"
    );
    assert_eq!(result.plan.outcome, PlanOutcome::Closing);
    assert_gate_then_cost(&result);
    assert_artifact_round_trip(&result, &ctx(5));
}

#[test]
fn a_measured_instrument_is_proposed_but_its_validity_is_an_assumption_no_study_establishes() {
    let study = study(zxy(), instrument(), 1, 2);
    assert_refusal_unchanged(&study);

    let actions = action_ids(&study, &ctx(6));
    assert!(actions.contains(&"find_instrument".to_owned()), "{actions:?}");
    assert!(actions.contains(&"measure_variable:common_cause_of_x_y".to_owned()), "{actions:?}");

    let graph = instrument();
    let cause = common_cause(&graph, 1, 2);
    let iv_study =
        candidate("instrument cohort", StudyKind::Instrument, &[], &[v(0), v(1), v(2)], 2);
    let assay = candidate("assay", StudyKind::Observation, &[], &[v(0), v(1), v(2), cause], 6);
    let result = study.evidence_plan(&request(vec![iv_study, assay]), &ctx(7)).unwrap();

    // An instrument study supplies the joint law a Wald contrast is read from, never the
    // exclusion and independence assumptions: no instrument option is verified.
    assert!(
        result
            .plan
            .options
            .iter()
            .filter(|option| option.route == "instrument")
            .all(|option| option.gate != RepairClassification::VerifiedSufficient)
    );
    // The plan is closed by the study that measures the confounder instead.
    assert_eq!(result.plan.outcome, PlanOutcome::Closing);
    let best = &result.plan.options[0];
    assert_eq!(best.labels[0].as_ref(), "assay");
    assert_eq!(best.gate, RepairClassification::VerifiedSufficient);
    assert_gate_then_cost(&result);
    assert_artifact_round_trip(&result, &ctx(7));
}

#[test]
fn a_hedge_is_answered_by_an_experiment_and_a_cheaper_survey_cannot_answer_it() {
    let study = study(twy(), hedge(), 0, 2);
    assert_refusal_unchanged(&study);

    // Every route the failure admits names its action, in a deterministic order.
    assert_eq!(
        action_ids(&study, &ctx(8)),
        vec![
            "collect_panel",
            "find_instrument",
            "measure_variable:common_cause_of_t_y",
            "measure_variable:common_cause_of_w_y",
            "run_experiment:t",
            "run_experiment:w",
        ]
    );

    let trial = candidate("randomized trial", StudyKind::Experiment, &[v(0)], &[v(2)], 9);
    let survey = candidate("bigger survey", StudyKind::Observation, &[], &[v(0), v(1), v(2)], 1);
    let result = study.evidence_plan(&request(vec![trial, survey]), &ctx(9)).unwrap();
    let best = &result.plan.options[0];
    assert_eq!(best.labels[0].as_ref(), "randomized trial");
    assert_eq!(best.gate, RepairClassification::VerifiedSufficient);
    assert_eq!(result.plan.outcome, PlanOutcome::Closing);
    // The cheap survey is declared and screened, and is not verified whatever it costs.
    assert!(result.plan.options.iter().any(|option| {
        option.labels[0].as_ref() == "bigger survey"
            && option.gate != RepairClassification::VerifiedSufficient
    }));
    assert_gate_then_cost(&result);
    assert_artifact_round_trip(&result, &ctx(9));

    // Costs are compared in one unit only.
    let mut eur = candidate("survey in euros", StudyKind::Observation, &[], &[v(0), v(1), v(2)], 1);
    eur.cost.unit_label = Arc::from("eur");
    let trial = candidate("randomized trial", StudyKind::Experiment, &[v(0)], &[v(2)], 9);
    let refused = study.evidence_plan(&request(vec![trial, eur]), &ctx(10)).unwrap_err();
    assert_eq!(refused.code(), "design_cost_units_mismatch");
}

#[test]
fn a_refused_study_reports_its_evidence_needs_through_the_capability_report_and_host_record() {
    let ctx = ctx(12);
    let instrument_study = study(zxy(), instrument(), 1, 2);
    let report = instrument_study.failure_capability(&ctx).unwrap();
    let ids: Vec<&str> = report.next_actions.iter().map(|a| a.id.as_ref()).collect();
    assert!(ids.contains(&"find_instrument"), "{ids:?}");
    assert!(ids.contains(&"measure_variable:common_cause_of_x_y"), "{ids:?}");
    // The instrument's validity is owed and uncheckable: no study satisfies it.
    assert!(
        report.unmet_obligations.iter().any(|o| o.kind.as_str() == "uncheckable"
            && o.status.as_str() == "untestable"
            && o.required_check.is_some()),
        "{:?}",
        report.unmet_obligations
    );
    // Every obligation that a study could meet is an unrun check carrying its proof step.
    assert!(report.unmet_obligations.iter().any(|o| o.kind.as_str() == "check_not_run"));
    assert!(report.neighbors.iter().any(|n| n.coordinate.as_ref() == "identified_by:instrument"
        && n.required_action.as_ref() == "find_instrument"));

    let record = antecedent::host::operation_record(&report);
    let recorded: Vec<&str> = record["next_actions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| a["id"].as_str().unwrap())
        .collect();
    assert_eq!(recorded, ids);
    assert_eq!(
        record["unmet_obligations"].as_array().unwrap().len(),
        report.unmet_obligations.len()
    );
    assert_eq!(record["neighbors"].as_array().unwrap().len(), report.neighbors.len());

    // A hedge owes experiments on its nodes; the same report names them.
    let hedge_report = study(twy(), hedge(), 0, 2).failure_capability(&ctx).unwrap();
    let hedge_ids: Vec<&str> = hedge_report.next_actions.iter().map(|a| a.id.as_ref()).collect();
    assert!(hedge_ids.contains(&"run_experiment:t") && hedge_ids.contains(&"run_experiment:w"));
    assert!(hedge_report.neighbors.iter().any(|n| n.coordinate.as_ref() == "identified_by:hedge"));

    // An identified study has no failure to report.
    let identified = study(zxy(), admg(3, &[(0, 1), (1, 2)], &[]), 1, 2);
    assert!(matches!(identified.failure_capability(&ctx), Err(EvidencePlanFacadeError::NoFailure)));
}

#[test]
fn an_identified_study_owes_no_evidence() {
    // x -> y with nothing hidden, and a front-door structure whose mediator is measured: the
    // complete algorithm identifies both, so there is no structural gap to plan for.
    let simple = study(zxy(), admg(3, &[(0, 1), (1, 2)], &[]), 1, 2);
    let front_door = study(twy(), admg(3, &[(0, 1), (1, 2)], &[(0, 2)]), 0, 2);
    let declared =
        request(vec![candidate("survey", StudyKind::Observation, &[], &[v(0), v(1), v(2)], 1)]);
    for identified in [&simple, &front_door] {
        assert!(matches!(
            identified.evidence_obligations(&ctx(11)),
            Err(EvidencePlanFacadeError::NoFailure)
        ));
        assert!(matches!(
            identified.candidate_templates(None, &ctx(11)),
            Err(EvidencePlanFacadeError::NoFailure)
        ));
        let refused = identified.evidence_plan(&declared, &ctx(11)).unwrap_err();
        assert!(matches!(refused, EvidencePlanFacadeError::NoFailure), "{refused}");
        assert_eq!(refused.detail(), "evidence_plan.no_failure_to_plan");
        assert_eq!(refused.code(), "invalid_argument");
    }
}
