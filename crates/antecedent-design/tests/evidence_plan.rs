//! Evidence plans: verified options from several theorem routes, ordered by the
//! identification gate and then by value net of a declared cost map, the value of
//! measuring a confounder under a declared prior, budget versus scientific
//! impossibility, equal cost units, and the replayable `evidence_plan_v1` artifact.
//!
//! Independent oracles: the two-point binomial bet of the EVSI tests (prior
//! `theta in {1/4, 3/4}`, bet pays `theta - 1/2`): `EVSI(n = 2) = 1/16` and
//! `EVSI(n = 4) = 11/128`, written out by hand, never read from the code under test.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::{
    CancellationToken, DistributionAvailability, ExecutionContext, SearchLimits, VariableId,
};
use antecedent_design::candidate_generation::generate_candidates;
use antecedent_design::design_ranking_artifact::{SealInputs, seal};
use antecedent_design::evsi::{
    CostToUtilityMap, EvsiCandidate, EvsiReport, EvsiRequest, StudyCostSpec, evaluate_evsi,
};
use antecedent_design::ranking::RankingRecord;
use antecedent_design::signal::{
    NativeBinomialSignal, SignalLimits, SignalProvider, SignalRequest,
};
use antecedent_design::{
    AffineUtility, BackdoorRepairFamily, CandidateDesign, ConfounderPriorDeclaration,
    ConfounderValueFrame, DecisionPrior, DecisionProblem, DesignCost, DesignRankConfig,
    DurableStudyCandidate, EvidencePlan, EvidencePlanArtifact, ExpectedEvidence,
    FrontDoorRepairFamily, HedgeRepairFamily, PlanInputs, PlanOutcome, PlanValue,
    RepairClassification, RepairConsumeLimits, RepairFamily, RepairFamilyRef, RepairLimits,
    RepairObjective, RepairReport, RepairReportArtifact, SamplingPlan, SensitivityFinding,
    SensitivityMethod, SensitivityRepairFamily, StudyCostDeclaration, StudyKind, UnitRules,
    UnverifiedRoute, repair,
};
use antecedent_graph::{Admg, Dag, DenseNodeId};
use antecedent_test_support::within as near;

mod common;

const T: u32 = 0;
const Y: u32 = 1;
const Z: u32 = 2;

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn vars(raw: &[u32]) -> Arc<[VariableId]> {
    raw.iter().copied().map(v).collect::<Vec<_>>().into()
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

fn dag(nodes: u32, edges: &[(u32, u32)]) -> Dag {
    let mut graph = Dag::with_variables(nodes);
    for (a, b) in edges {
        graph.insert_directed(DenseNodeId::from_raw(*a), DenseNodeId::from_raw(*b)).unwrap();
    }
    graph
}

fn admg(nodes: u32, directed: &[(u32, u32)], bidirected: &[(u32, u32)]) -> Admg {
    let mut graph = Admg::with_variables(nodes);
    for (a, b) in directed {
        graph.insert_directed(DenseNodeId::from_raw(*a), DenseNodeId::from_raw(*b)).unwrap();
    }
    for (a, b) in bidirected {
        graph.insert_bidirected(DenseNodeId::from_raw(*a), DenseNodeId::from_raw(*b)).unwrap();
    }
    graph
}

/// A candidate with the sample size `n` (it feeds the binomial signal) and `cost` in `unit`.
fn study(
    label: &str,
    kind: StudyKind,
    interventions: &[u32],
    measured: &[u32],
    n: u64,
    cost: u64,
    unit: &str,
) -> DurableStudyCandidate {
    DurableStudyCandidate {
        label: Arc::from(label),
        kind,
        population: Arc::from("pop"),
        interventions: vars(interventions),
        measured: vars(measured),
        joint_measurement: true,
        sample_size: n,
        recruitment: Arc::from("consecutive recruitment"),
        timing: Arc::from("baseline"),
        unit_rules: UnitRules {
            unit: Arc::from("patient"),
            cluster: None,
            whole_cluster_sampling: false,
        },
        cost: StudyCostDeclaration { units: cost, unit_label: Arc::from(unit), sample_budget: n },
        feasible: true,
        feasibility_notes: Arc::from([]),
        expected_evidence: if kind == StudyKind::SampleIncrease {
            Arc::from([])
        } else {
            Arc::from([ExpectedEvidence {
                population: Arc::from("pop"),
                interventions: vars(interventions),
                intervention_values: Arc::from([]),
                conditioned_on: Arc::from([]),
                measured: vars(measured),
                distribution: DistributionAvailability::Joint,
            }])
        },
        external_provider: None,
    }
}

fn observation(label: &str, measured: &[u32], n: u64, cost: u64) -> DurableStudyCandidate {
    study(label, StudyKind::Observation, &[], measured, n, cost, "usd")
}

fn limits(operations: usize, depth: usize) -> RepairLimits {
    RepairLimits {
        search: SearchLimits { operations, depth },
        memory_limit_bytes: 16 * 1024 * 1024,
    }
}

fn run(
    family: &dyn RepairFamily,
    candidates: &[DurableStudyCandidate],
    limits: RepairLimits,
) -> RepairReport {
    repair(family, candidates, RepairObjective::MinimizeCost, limits, &ctx()).unwrap()
}

fn plan_of(
    reports: &[RepairReport],
    candidates: &[DurableStudyCandidate],
    value: Option<&PlanValue>,
) -> EvidencePlan {
    EvidencePlan::build(PlanInputs { reports, candidates, value }, Vec::new()).unwrap()
}

// ----------------------------------------------------------------- value fixtures

fn quantity(role: QuantityRole, id: &str) -> ScientificQuantity {
    ScientificQuantity {
        role,
        units: "dimensionless".into(),
        functional_id: "state".into(),
        ..common::quantity(id)
    }
}

fn evsi_candidate(study: &DurableStudyCandidate) -> EvsiCandidate {
    let id = study.semantic_id().to_string();
    let provider: Arc<dyn SignalProvider> = Arc::new(NativeBinomialSignal);
    EvsiCandidate {
        semantic_id: id.clone(),
        design: CandidateDesign::IncreaseSamplingRate(SamplingPlan {
            additional_samples: study.sample_size,
            cost: DesignCost::zero(),
            tag: 0,
        }),
        signal_request: SignalRequest {
            candidate_id: id,
            prior_id: "prior-1".into(),
            state_quantity: quantity(QuantityRole::Outcome, "schema:theta"),
            observation_quantity: quantity(QuantityRole::Outcome, "schema:signal"),
            sample_size: study.sample_size,
            rng_seed: 11,
            evidence_lineage: vec!["snapshot:a".into()],
            conditional_independence: "iid_given_state".into(),
            limits: SignalLimits::default(),
        },
        provider,
        cost: StudyCostSpec {
            amount: study.cost.units as f64,
            unit: study.cost.unit_label.to_string(),
        },
        reused_observation_ids: vec![],
    }
}

fn request(candidates: Vec<EvsiCandidate>, per_cost: f64) -> EvsiRequest {
    EvsiRequest {
        decision_contract_identity: "contract-1".into(),
        utility_unit: "utility".into(),
        action_ids: vec!["abstain".into(), "bet".into()],
        candidates,
        cost_map: Some(CostToUtilityMap {
            cost_unit: "usd".into(),
            utility_unit: "utility".into(),
            utility_per_cost: per_cost,
        }),
        require_net_value: false,
        prior_observation_ids: vec![],
        rank_config: DesignRankConfig {
            min_batches: 4,
            max_batches: 4,
            batch_size: 4,
            rank_uncertainty_threshold: 0.0,
        },
        rng_seed: 5,
        mc_error_tolerance: 1e-3,
        tie_tolerance: 1e-12,
        max_candidates: 16,
    }
}

fn bet() -> (DecisionProblem<usize, f64>, DecisionPrior<f64>) {
    let utility = AffineUtility::new(vec![0.0, -0.5], vec![0.0, 1.0]).unwrap();
    (
        DecisionProblem::new(vec![0, 1], Arc::new(utility), vec![]),
        DecisionPrior::Draws(vec![0.25, 0.75]),
    )
}

fn evaluate(studies: &[&DurableStudyCandidate], per_cost: f64) -> EvsiReport {
    let (problem, prior) = bet();
    let candidates = studies.iter().map(|s| evsi_candidate(s)).collect();
    evaluate_evsi(&problem, &prior, &request(candidates, per_cost), &CancellationToken::new())
        .unwrap()
}

fn value_of(report: &EvsiReport) -> PlanValue {
    let record = RankingRecord::from_evsi(report, &[]).unwrap();
    PlanValue::from_ranking(&record, "value-1").unwrap()
}

fn backdoor() -> BackdoorRepairFamily {
    BackdoorRepairFamily::try_new(
        dag(3, &[(T, Y), (Z, T), (Z, Y)]),
        v(T),
        v(Y),
        "pop",
        [v(T), v(Y)],
        &[],
    )
    .unwrap()
}

fn gate_of(
    plan: &EvidencePlan,
    route: &str,
    member: &DurableStudyCandidate,
) -> RepairClassification {
    plan.options
        .iter()
        .find(|o| o.route == route && o.members.len() == 1 && o.members[0] == member.semantic_id())
        .unwrap()
        .gate
}

// ------------------------------------------------------------------------ tests

#[test]
fn verified_options_come_first_and_are_ordered_by_net_value_under_the_declared_cost_map() {
    let family = backdoor();
    let cheap = observation("cheap cohort", &[T, Y, Z], 2, 1);
    let large = observation("large cohort", &[T, Y, Z], 4, 3);
    let blind = observation("blind registry", &[T, Y], 4, 1);
    let candidates = vec![cheap.clone(), large.clone(), blind.clone()];
    let report = run(&family, &candidates, limits(1000, 2));

    // 0.01 utility per usd: net(cheap) = 1/16 - 0.01, net(large) = 11/128 - 0.03.
    let cheap_price = 0.01;
    let value = value_of(&evaluate(&[&cheap, &large], cheap_price));
    let plan = plan_of(std::slice::from_ref(&report), &candidates, Some(&value));
    assert_eq!(plan.outcome, PlanOutcome::Closing);
    assert_eq!(plan.status(), None);
    let order: Vec<&str> = plan.options.iter().map(|o| o.labels[0].as_ref()).collect();
    assert_eq!(order, ["large cohort", "cheap cohort", "blind registry"]);
    let first = plan.options[0].value.as_ref().unwrap();
    assert!(near(first.evsi, 11.0 / 128.0, 1e-12), "{}", first.evsi);
    assert!(near(first.net_value.unwrap(), 11.0 / 128.0 - 0.03, 1e-12));
    assert!(plan.options[0].resolves.len() == 1 && plan.options[0].unresolved.is_empty());
    // The gate comes first: the study that cannot identify is last whatever it promises.
    assert_eq!(plan.options[2].gate, RepairClassification::Insufficient);
    assert_eq!(plan.options[2].unresolved.len(), 1);
    assert_eq!(plan.ranking_guarantee(), "point_only");
    assert_eq!(plan.calibration(), "unmeasured");

    // A dearer cost map reverses the verified pair, never the gate.
    let dear = value_of(&evaluate(&[&cheap, &large], 0.02));
    let plan = plan_of(std::slice::from_ref(&report), &candidates, Some(&dear));
    let order: Vec<&str> = plan.options.iter().map(|o| o.labels[0].as_ref()).collect();
    assert_eq!(order, ["cheap cohort", "large cohort", "blind registry"]);
}

#[test]
fn a_study_of_part_of_the_adjustment_set_does_not_identify() {
    // Z and U each confound; only a study measuring both identifies.
    let family = BackdoorRepairFamily::try_new(
        dag(4, &[(T, Y), (Z, T), (Z, Y), (3, T), (3, Y)]),
        v(T),
        v(Y),
        "pop",
        [v(T), v(Y)],
        &[],
    )
    .unwrap();
    let both = observation("both", &[T, Y, Z, 3], 2, 2);
    let only_z = observation("only z", &[T, Y, Z], 2, 1);
    let candidates = vec![both.clone(), only_z.clone()];
    let report = run(&family, &candidates, limits(1000, 2));
    let value = value_of(&evaluate(&[&both, &only_z], 0.01));
    let plan = plan_of(std::slice::from_ref(&report), &candidates, Some(&value));
    assert_eq!(gate_of(&plan, "backdoor", &both), RepairClassification::VerifiedSufficient);
    assert_eq!(gate_of(&plan, "backdoor", &only_z), RepairClassification::Insufficient);
    // The unvalued, unverified study is last whatever it would cost.
    assert_eq!(plan.options.last().unwrap().gate, RepairClassification::Insufficient);
}

#[test]
fn several_routes_are_compared_in_one_plan_each_stating_what_it_resolves() {
    // T -> M -> Y with U confounding T and Y; only T and Y are observed today.
    let graph = dag(4, &[(0, 1), (1, 2), (3, 0), (3, 2)]);
    let (t, m, y, u) = (0, 1, 2, 3);
    let back =
        BackdoorRepairFamily::try_new(graph.clone(), v(t), v(y), "pop", [v(t), v(y)], &[]).unwrap();
    let front = FrontDoorRepairFamily::try_new(graph, v(t), v(y), "pop", [v(t), v(y)]).unwrap();
    let mediator = observation("mediator cohort", &[t, m, y], 2, 4);
    let assay = observation("confounder assay", &[t, y, u], 2, 2);
    let candidates = vec![mediator.clone(), assay.clone()];
    let backdoor_report = run(&back, &candidates, limits(1000, 2));
    let frontdoor_report = run(&front, &candidates, limits(1000, 2));
    let plan = plan_of(&[backdoor_report, frontdoor_report], &candidates, None);

    assert_eq!(gate_of(&plan, "backdoor", &assay), RepairClassification::VerifiedSufficient);
    assert_eq!(gate_of(&plan, "frontdoor", &mediator), RepairClassification::VerifiedSufficient);
    assert_eq!(gate_of(&plan, "backdoor", &mediator), RepairClassification::Insufficient);
    assert_eq!(gate_of(&plan, "frontdoor", &assay), RepairClassification::Insufficient);
    // Without a value the order within the gate is the declared cost: the cheaper assay.
    assert_eq!(plan.options[0].labels[0].as_ref(), "confounder assay");
    assert_eq!(plan.options[1].labels[0].as_ref(), "mediator cohort");
    assert_eq!(plan.closing().count(), 2);
    for option in plan.closing() {
        let obligation = plan.obligation(&option.resolves[0]).unwrap();
        assert_eq!(obligation.provenance.family.as_ref(), option.route);
        assert!(option.derivation.as_ref().unwrap().verified);
    }
}

#[test]
fn a_hedge_is_answered_by_experiments_and_a_sampling_candidate_is_never_verified() {
    let family = HedgeRepairFamily::try_new(
        admg(3, &[(0, 1), (1, 2)], &[(0, 2), (1, 2)]),
        v(0),
        v(2),
        "pop",
        [v(0), v(1), v(2)],
    )
    .unwrap();
    let trial = study("randomized trial", StudyKind::Experiment, &[0], &[2], 2, 5, "usd");
    let more_rows = study("more rows", StudyKind::SampleIncrease, &[], &[0, 1, 2], 4, 1, "usd");
    let bigger_survey = observation("bigger survey", &[0, 1, 2], 4, 1);
    let candidates = vec![trial.clone(), more_rows.clone(), bigger_survey.clone()];
    let report = run(&family, &candidates, limits(1000, 2));
    let plan = plan_of(&[report], &candidates, None);
    assert_eq!(plan.options[0].labels[0].as_ref(), "randomized trial");
    assert_eq!(plan.options[0].gate, RepairClassification::VerifiedSufficient);
    for rejected in [&more_rows, &bigger_survey] {
        let gate = gate_of(&plan, "hedge", rejected);
        assert_ne!(gate, RepairClassification::VerifiedSufficient);
    }
    assert_eq!(plan.closing().count(), 1);
}

#[test]
fn an_instrument_is_not_certified_by_a_study_and_the_assumptions_remain() {
    let family = antecedent_design::InstrumentRepairFamily::try_new(
        dag(4, &[(0, 1), (1, 2), (3, 1), (3, 2)]),
        v(1),
        v(2),
        "pop",
        [v(1), v(2)],
    )
    .unwrap();
    let generation = generate_candidates(&family.unresolved_obligations());
    let template = generation
        .templates
        .iter()
        .find(|t| t.kind == StudyKind::Instrument)
        .expect("an instrument study is proposed");
    let candidate = template
        .complete(antecedent_design::StudyDeclarations {
            label: Some(Arc::from("encouragement trial")),
            sample_size: 2,
            recruitment: Arc::from("randomized encouragement"),
            timing: Arc::from("one wave"),
            unit_rules: UnitRules {
                unit: Arc::from("patient"),
                cluster: None,
                whole_cluster_sampling: false,
            },
            cost: StudyCostDeclaration { units: 3, unit_label: Arc::from("usd"), sample_budget: 2 },
            feasible: true,
            feasibility_notes: Arc::from([]),
            external_provider: None,
        })
        .unwrap();
    let report = run(&family, std::slice::from_ref(&candidate), limits(1000, 2));
    let plan = plan_of(&[report], std::slice::from_ref(&candidate), None);
    let option = &plan.options[0];
    assert_eq!(option.gate, RepairClassification::NotCertified);
    assert!(option.reasons[0].contains("does not establish assumption"));
    assert_eq!(option.assumptions_remaining.len(), 3);
    assert!(option.derivation.is_some());
    assert_eq!(plan.outcome, PlanOutcome::NoneCertified);
    assert_eq!(
        plan.status(),
        Some(("transport_missing_evidence", "evidence_obligations.wrong_contract"))
    );
}

#[test]
fn budget_exhaustion_is_reported_as_a_budget_and_never_as_impossibility() {
    let family = backdoor();
    let candidates = vec![observation("a", &[T, Y], 2, 1), observation("b", &[T, Y], 4, 1)];
    // A full search finds nothing certified within the declared studies.
    let full = run(&family, &candidates, limits(1000, 2));
    let none = plan_of(&[full], &candidates, None);
    assert_eq!(none.outcome, PlanOutcome::NoneCertified);
    assert_eq!(none.status(), Some(("transport_not_certified", "evidence_plan.none_certified")));

    // One operation stops the same search; the plan says the budget stopped it.
    let stopped = run(&family, &candidates, limits(1, 2));
    assert!(stopped.receipt.stop.is_some());
    let plan = plan_of(&[stopped], &candidates, None);
    assert_eq!(plan.outcome, PlanOutcome::BudgetExhausted);
    assert_eq!(plan.status(), Some(("transport_budget_cancel", "evidence_plan.budget")));
    assert!(plan.routes[0].budget.unevaluated_total > 0);
    assert!(plan.options.iter().any(|o| o.gate == RepairClassification::Unevaluated));
    assert!(plan.options.iter().all(|o| o.gate != RepairClassification::VerifiedSufficient));
}

#[test]
fn costs_are_compared_only_in_one_unit() {
    let family = backdoor();
    let usd = observation("usd cohort", &[T, Y, Z], 2, 1);
    let eur = study("eur cohort", StudyKind::Observation, &[], &[T, Y, Z], 4, 1, "eur");
    let candidates = vec![usd, eur];
    let report = repair(
        &family,
        &candidates,
        RepairObjective::MinimizeSampleBudget,
        limits(1000, 2),
        &ctx(),
    )
    .unwrap();
    let refused = EvidencePlan::build(
        PlanInputs { reports: &[report], candidates: &candidates, value: None },
        Vec::new(),
    )
    .unwrap_err();
    assert_eq!(refused.code, "design_cost_units_mismatch");
    assert_eq!(refused.detail, "evidence_plan.cost_units_mismatch");

    // A cost map from another unit than the studies' is refused too.
    let one = observation("one", &[T, Y, Z], 2, 1);
    let report = run(&family, std::slice::from_ref(&one), limits(1000, 2));
    let mut value = value_of(&evaluate(&[&one], 0.01));
    value.cost_map.as_mut().unwrap().cost_unit = "eur".into();
    let refused = EvidencePlan::build(
        PlanInputs {
            reports: &[report],
            candidates: std::slice::from_ref(&one),
            value: Some(&value),
        },
        Vec::new(),
    )
    .unwrap_err();
    assert_eq!(refused.detail, "evidence_plan.cost_units_mismatch");
}

#[test]
fn measuring_a_named_confounder_is_valued_under_the_declared_prior() {
    // The surface: abstain pays 0; bet pays theta - 1/2 at the two evaluated coordinates.
    let outcome = antecedent_design::sensitivity_decision::SensitivityOutcome::InvariantAction {
        action: "abstain".into(),
    };
    let frame = ConfounderValueFrame::from_table(
        vec![0.25, 0.75],
        vec!["abstain".into(), "bet".into()],
        vec![vec![0.0, 0.0], vec![-0.25, 0.25]],
        outcome,
        "sens-artifact".into(),
        ConfounderPriorDeclaration {
            declared_by: Arc::from("analyst"),
            basis: Arc::from("pilot assay, 2026"),
            draws_per_coordinate: vec![1, 1],
        },
    )
    .unwrap();
    let finding =
        SensitivityFinding { method: SensitivityMethod::EValue, strength: 1.4, robust: false };
    let family = SensitivityRepairFamily::try_new(
        dag(3, &[(T, Y), (Z, T), (Z, Y)]),
        v(T),
        v(Y),
        "pop",
        [v(T), v(Y), v(Z)],
        finding,
    )
    .unwrap();
    let assay = observation("confounder assay", &[T, Y, Z, family.confounder().raw()], 2, 1);
    let rows = observation("more rows", &[T, Y, Z], 4, 1);
    let candidates = vec![assay.clone(), rows.clone()];
    let report = run(&family, &candidates, limits(1000, 2));

    let (problem, prior) = (frame.problem().unwrap(), frame.decision_prior());
    let evsi = evaluate_evsi(
        &problem,
        &prior,
        &request(vec![evsi_candidate(&assay)], 0.01),
        &CancellationToken::new(),
    )
    .unwrap();
    // The grid utility reproduces the affine oracle: EVSI(n = 2) = 1/16.
    assert!(near(evsi.candidates[0].evsi, 1.0 / 16.0, 1e-12), "{}", evsi.candidates[0].evsi);
    let value = PlanValue::from_ranking(
        &RankingRecord::from_evsi(&evsi, &frame.source_digests()).unwrap(),
        "value-sens",
    )
    .unwrap();
    let plan = plan_of(&[report], &candidates, Some(&value));
    assert_eq!(plan.options[0].labels[0].as_ref(), "confounder assay");
    assert_eq!(plan.options[0].gate, RepairClassification::VerifiedSufficient);
    assert!(near(plan.options[0].value.as_ref().unwrap().evsi, 1.0 / 16.0, 1e-12));
    // The prior that valued it is recorded in the ranking's sources.
    let record = RankingRecord::from_evsi(&evsi, &frame.source_digests()).unwrap();
    assert!(record.source_digests.contains(&frame.prior.identity()));
    // More rows of the same measurements do not identify.
    assert_eq!(gate_of(&plan, "sensitivity", &rows), RepairClassification::Insufficient);
}

fn artifact_fixture() -> (
    EvidencePlan,
    Vec<RepairReportArtifact>,
    antecedent_design::design_ranking_artifact::DesignRankingArtifactWire,
) {
    let family = backdoor();
    let cheap = observation("cheap cohort", &[T, Y, Z], 2, 1);
    let large = observation("large cohort", &[T, Y, Z], 4, 3);
    let candidates = vec![cheap.clone(), large.clone()];
    let report = run(&family, &candidates, limits(1000, 2));
    let repair_artifact =
        RepairReportArtifact::build(RepairFamilyRef::Backdoor(&family), &candidates, &report)
            .unwrap();
    let (problem, prior) = bet();
    let req = request(vec![evsi_candidate(&cheap), evsi_candidate(&large)], 0.01);
    let evsi = evaluate_evsi(&problem, &prior, &req, &CancellationToken::new()).unwrap();
    let wire = seal(&SealInputs {
        problem: &problem,
        prior: &prior,
        request: &req,
        report: &evsi,
        source_digests: &[],
    })
    .unwrap();
    let record = RankingRecord::from_evsi(&evsi, &[]).unwrap();
    let value = PlanValue::from_ranking(&record, &wire.digest).unwrap();
    let plan = plan_of(&[report], &candidates, Some(&value));
    (plan, vec![repair_artifact], wire)
}

#[test]
fn the_plan_artifact_round_trips_and_replays_every_route_and_value() {
    let (plan, repairs, wire) = artifact_fixture();
    let artifact = EvidencePlanArtifact::build(&repairs, Some(&wire), &plan).unwrap();
    let bytes = artifact.to_bytes("evidence-plan").unwrap();
    let decoded = EvidencePlanArtifact::from_bytes(&bytes).unwrap();
    assert_eq!(decoded, artifact);
    let replayed = decoded.consume(RepairConsumeLimits::default(), &ctx()).unwrap();
    assert_eq!(replayed.identity(), plan.identity());
    assert_eq!(replayed.options, plan.options);
    assert_eq!(replayed.outcome, PlanOutcome::Closing);
}

#[test]
fn a_resealed_edit_of_an_option_a_value_or_the_outcome_is_refused() {
    let (plan, repairs, wire) = artifact_fixture();
    let artifact = EvidencePlanArtifact::build(&repairs, Some(&wire), &plan).unwrap();

    let mut cost = artifact.clone();
    cost.options[0].cost_units += 1;
    let refused = cost.clone().consume(RepairConsumeLimits::default(), &ctx()).unwrap_err();
    assert_eq!(refused.detail, "evidence_plan_artifact.invalid_artifact");
    let refused =
        cost.sealed().unwrap().consume(RepairConsumeLimits::default(), &ctx()).unwrap_err();
    assert_eq!(refused.detail, "evidence_plan_artifact.replay_mismatch");
    assert!(refused.message.starts_with("options"));

    let mut value = artifact.clone();
    value.options[0].value.as_mut().unwrap().evsi += 0.5;
    let refused =
        value.sealed().unwrap().consume(RepairConsumeLimits::default(), &ctx()).unwrap_err();
    assert_eq!(refused.detail, "evidence_plan_artifact.replay_mismatch");

    let mut gate = artifact.clone();
    gate.options[1].gate = "insufficient".into();
    let refused =
        gate.sealed().unwrap().consume(RepairConsumeLimits::default(), &ctx()).unwrap_err();
    assert_eq!(refused.detail, "evidence_plan_artifact.replay_mismatch");

    // An unreadable container and an unknown version are refused before any replay.
    assert!(EvidencePlanArtifact::from_bytes(b"not an artifact").is_err());
    let mut other = artifact;
    other.version = 2;
    assert_eq!(
        other.sealed().unwrap().consume(RepairConsumeLimits::default(), &ctx()).unwrap_err().detail,
        "evidence_plan_artifact.unsupported_version"
    );
}

#[test]
fn routes_without_a_theorem_checker_are_listed_and_not_scored() {
    let family = backdoor();
    let cohort = observation("cohort", &[T, Y, Z], 2, 1);
    let report = run(&family, std::slice::from_ref(&cohort), limits(1000, 2));
    let plan = EvidencePlan::build(
        PlanInputs { reports: &[report], candidates: std::slice::from_ref(&cohort), value: None },
        vec![UnverifiedRoute {
            route: Arc::from("panel"),
            obligations: vec![Arc::from("o1")],
            reason: Arc::from("no theorem checker decides a within-unit comparison"),
        }],
    )
    .unwrap();
    assert_eq!(plan.unverified.len(), 1);
    assert!(plan.options.iter().all(|o| o.route != "panel"));
}
