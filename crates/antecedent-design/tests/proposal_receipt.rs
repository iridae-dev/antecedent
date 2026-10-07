//! X6: the per-proposal receipt linking identification repair and design ranking, its
//! permutation-invariant bundle, the cross-artifact verifier and re-identification on
//! arrived data.
//!
//! Oracles (derived by hand, not read from the code under test):
//!
//! * Back-door contract: Z1 and Z2 each confound T -> Y, so the only admissible
//!   adjustment set is {Z1, Z2} and only one observational joint law over (T, Y, Z1, Z2)
//!   repairs it; a study of (T, Y, Z1) alone stays insufficient.
//! * Transport contract: X -> Y with a selection mechanism on X; the source experiment
//!   do(X) measuring Y repairs it, a source observation does not.
//! * Valuation: binary state, guess it for utility 1, prior 1/2. A signal of accuracy `a`
//!   has `EVSI = a - 1/2`: 3/4 gives 1/4 and 5/8 gives 1/8. With cost 0.01 utility per USD
//!   the study of 20 USD has net value `1/4 - 1/5 = 1/20` and the study of 5 USD has net
//!   value `1/8 - 1/20 = 3/40`, so the cheaper study ranks first.

use std::sync::Arc;

use antecedent_core::{
    CancellationToken, DistributionAvailability, Environment, EvidenceCatalog, EvidenceKind,
    ExecutionContext, ExternalCapability, ExternalScientificObject, ExternalTrustState,
    ProviderObjectIdentity, QuantityRole, ScientificQuantity, SignalProviderContract,
    VariableCoordinate, VariableDomain, VariableId,
};
use antecedent_design::design_ranking_artifact::{DesignRankingArtifactWire, SealInputs, seal};
use antecedent_design::evsi::{
    CostToUtilityMap, EvsiCandidate, EvsiRequest, StudyCostSpec, evaluate_evsi,
};
use antecedent_design::proposal_receipt::{
    Arrival, ArrivalVerdict, ArrivedEvidence, ChangedPremise, EvidenceState, ObservedLaw,
    ProposalBundle, ProposalReceipt, ProposalReceiptError, on_arrival, verify,
};
use antecedent_design::signal::{
    ExternalLaw, ExternalSignal, ExternalSignalBody, SignalLimits, SignalProvider, SignalRequest,
};
use antecedent_design::{
    AffineUtility, BackdoorRepairFamily, CandidateDesign, DecisionPrior, DecisionProblem,
    DesignCost, DesignRankConfig, DurableStudyCandidate, ExpectedEvidence, RepairFamilyRef,
    RepairLimits, RepairObjective, RepairReportArtifact, SamplingPlan, StudyCostDeclaration,
    StudyKind, TransportRepairFamily, UnitRules, repair,
};
use antecedent_graph::{Admg, Dag, DenseNodeId, SelectionDiagram};
use antecedent_identify::{ClassicalTransportQuery, SidLimits};

// ------------------------------------------------------------------- fixtures

const T: u32 = 0;
const Y: u32 = 1;
const Z1: u32 = 2;
const Z2: u32 = 3;
const X: u32 = 0;
const TY: u32 = 1;
const SAMPLES: u64 = 200;

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn vars(raw: &[u32]) -> Arc<[VariableId]> {
    raw.iter().copied().map(v).collect::<Vec<_>>().into()
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

fn dag4() -> Dag {
    let mut graph = Dag::with_variables(4);
    for (from, to) in [(Z1, T), (Z1, Y), (Z2, T), (Z2, Y), (T, Y)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    graph
}

fn backdoor_family() -> BackdoorRepairFamily {
    BackdoorRepairFamily::try_new(dag4(), v(T), v(Y), "pop", [v(T), v(Y)], &[]).unwrap()
}

fn study(
    label: &str,
    kind: StudyKind,
    population: &str,
    interventions: &[u32],
    measured: &[u32],
    units: u64,
) -> DurableStudyCandidate {
    DurableStudyCandidate {
        label: Arc::from(label),
        kind,
        population: Arc::from(population),
        interventions: vars(interventions),
        measured: vars(measured),
        joint_measurement: true,
        sample_size: SAMPLES,
        recruitment: Arc::from("consecutive recruitment"),
        timing: Arc::from("baseline"),
        unit_rules: UnitRules {
            unit: Arc::from("patient"),
            cluster: None,
            whole_cluster_sampling: false,
        },
        cost: StudyCostDeclaration { units, unit_label: Arc::from("USD"), sample_budget: SAMPLES },
        feasible: true,
        feasibility_notes: Arc::from([]),
        expected_evidence: Arc::from([ExpectedEvidence {
            population: Arc::from(population),
            interventions: vars(interventions),
            intervention_values: Arc::from([]),
            conditioned_on: Arc::from([]),
            measured: vars(measured),
            distribution: DistributionAvailability::Joint,
        }]),
        external_provider: None,
    }
}

fn observation(label: &str, measured: &[u32], units: u64) -> DurableStudyCandidate {
    study(label, StudyKind::Observation, "pop", &[], measured, units)
}

fn transport_catalog() -> EvidenceCatalog {
    let coordinate =
        |raw| VariableCoordinate { variable: v(raw), domain: VariableDomain::Binary, unit: None };
    let source = Environment::try_new("source", [coordinate(X), coordinate(TY)], [v(X)]).unwrap();
    let target = Environment::try_new("target", [coordinate(X), coordinate(TY)], []).unwrap();
    EvidenceCatalog::try_new([source, target], [], [], None).unwrap()
}

fn transport_family() -> TransportRepairFamily {
    let mut graph = Admg::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(X), DenseNodeId::from_raw(TY)).unwrap();
    let diagram = SelectionDiagram::try_new(graph, [v(X)]).unwrap();
    let query = ClassicalTransportQuery {
        outcomes: Arc::from([v(TY)]),
        treatments: Arc::from([v(X)]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    };
    TransportRepairFamily::try_new(
        diagram,
        query,
        transport_catalog(),
        SidLimits::default(),
        &ctx(),
    )
    .unwrap()
}

fn quantity(id: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: id.into(),
        variable_name: id.into(),
        role: QuantityRole::Outcome,
        units: "dimensionless".into(),
        population_id: "target".into(),
        regime_id: "observational".into(),
        horizon: 0,
        functional_id: "state".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn signal_request(candidate: &str, n: u64) -> SignalRequest {
    SignalRequest {
        candidate_id: candidate.into(),
        prior_id: "prior-1".into(),
        state_quantity: quantity("schema:state"),
        observation_quantity: quantity("schema:signal"),
        sample_size: n,
        rng_seed: 3,
        evidence_lineage: vec!["snapshot:a".into()],
        conditional_independence: "iid_given_state".into(),
        limits: SignalLimits::default(),
    }
}

fn external(req: &SignalRequest, law: ExternalLaw) -> Arc<dyn SignalProvider> {
    let object = ExternalScientificObject::Signal(SignalProviderContract {
        identity: ProviderObjectIdentity {
            provider_id: "lab".into(),
            object_id: format!("signal-{}", req.candidate_id),
            version_id: "v1".into(),
            snapshot_id: "snap".into(),
            request_id: req.fingerprint(),
        },
        candidate_id: req.candidate_id.clone(),
        prior_id: req.prior_id.clone(),
        observation: req.observation_quantity.clone(),
        capabilities: vec![ExternalCapability::Sample, ExternalCapability::Update],
    });
    let trust = ExternalTrustState::attest(&object, "lab-qa").unwrap();
    let body = ExternalSignalBody {
        sample_size: req.sample_size,
        state_quantity: req.state_quantity.clone(),
        observation_quantity: req.observation_quantity.clone(),
        law,
    };
    Arc::new(ExternalSignal::new(object, trust, body).unwrap())
}

fn accuracy_law(accuracy: f64) -> ExternalLaw {
    ExternalLaw::Posterior {
        states: vec![0.0, 1.0],
        statistics: vec![0.0, 1.0],
        predictive: vec![0.5, 0.5],
        posterior: vec![vec![accuracy, 1.0 - accuracy], vec![1.0 - accuracy, accuracy]],
    }
}

/// The ranked candidate of a repair candidate: same semantic id, cost and sample size.
fn ranked(study: &DurableStudyCandidate, accuracy: f64) -> EvsiCandidate {
    let id = study.semantic_id().to_string();
    let req = signal_request(&id, study.sample_size);
    EvsiCandidate {
        semantic_id: id,
        design: CandidateDesign::IncreaseSamplingRate(SamplingPlan {
            additional_samples: study.sample_size,
            cost: DesignCost::zero(),
            tag: 0,
        }),
        signal_request: req.clone(),
        provider: external(&req, accuracy_law(accuracy)),
        cost: StudyCostSpec {
            amount: f64::from(u32::try_from(study.cost.units).unwrap()),
            unit: study.cost.unit_label.to_string(),
        },
        reused_observation_ids: vec!["obs-future".into()],
    }
}

fn digests() -> Vec<String> {
    vec!["digest-b".into(), "digest-a".into()]
}

fn seal_ranking(candidates: Vec<EvsiCandidate>) -> DesignRankingArtifactWire {
    let utility = AffineUtility::new(vec![1.0, 0.0], vec![-1.0, 1.0]).unwrap();
    let problem: DecisionProblem<usize, f64> =
        DecisionProblem::new(vec![0, 1], Arc::new(utility), vec![]);
    let prior = DecisionPrior::Draws(vec![0.0, 1.0]);
    let req = EvsiRequest {
        decision_contract_identity: "contract-1".into(),
        utility_unit: "utility".into(),
        action_ids: vec!["guess0".into(), "guess1".into()],
        candidates,
        cost_map: Some(CostToUtilityMap {
            cost_unit: "USD".into(),
            utility_unit: "utility".into(),
            utility_per_cost: 0.01,
        }),
        require_net_value: false,
        prior_observation_ids: vec!["obs-prior".into()],
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
    };
    let report = evaluate_evsi(&problem, &prior, &req, &CancellationToken::new()).unwrap();
    seal(&SealInputs {
        problem: &problem,
        prior: &prior,
        request: &req,
        report: &report,
        source_digests: &digests(),
    })
    .unwrap()
}

struct Backdoor {
    family: BackdoorRepairFamily,
    cheap: DurableStudyCandidate,
    wide: DurableStudyCandidate,
    repair: RepairReportArtifact,
    ranking: DesignRankingArtifactWire,
    bundle: ProposalBundle,
}

/// `cheap` measures (T, Y, Z1) for 5 USD and stays insufficient; `wide` measures
/// (T, Y, Z1, Z2) jointly for 20 USD and repairs the contract.
fn backdoor() -> Backdoor {
    let family = backdoor_family();
    let cheap = observation("cheap", &[T, Y, Z1], 5);
    let wide = observation("wide", &[T, Y, Z1, Z2], 20);
    let candidates = [cheap.clone(), wide.clone()];
    let report = repair(
        &family,
        &candidates,
        RepairObjective::MinimizeCost,
        RepairLimits::default(),
        &ctx(),
    )
    .unwrap();
    let repair =
        RepairReportArtifact::build(RepairFamilyRef::Backdoor(&family), &candidates, &report)
            .unwrap();
    let ranking = seal_ranking(vec![ranked(&cheap, 0.625), ranked(&wide, 0.75)]);
    let bundle = ProposalBundle::build(&repair, &ranking).unwrap();
    Backdoor { family, cheap, wide, repair, ranking, bundle }
}

struct Transport {
    family: TransportRepairFamily,
    experiment: DurableStudyCandidate,
    bundle: ProposalBundle,
}

fn transport() -> Transport {
    let family = transport_family();
    let experiment = study("exp", StudyKind::Experiment, "source", &[X], &[TY], 3);
    let observed = study("obs", StudyKind::Observation, "source", &[], &[TY], 1);
    let candidates = [experiment.clone(), observed.clone()];
    let report = repair(
        &family,
        &candidates,
        RepairObjective::MinimizeCost,
        RepairLimits::default(),
        &ctx(),
    )
    .unwrap();
    let repair =
        RepairReportArtifact::build(RepairFamilyRef::Transport(&family), &candidates, &report)
            .unwrap();
    let ranking = seal_ranking(vec![ranked(&experiment, 0.75), ranked(&observed, 0.625)]);
    let bundle = ProposalBundle::build(&repair, &ranking).unwrap();
    Transport { family, experiment, bundle }
}

fn detail(error: &ProposalReceiptError) -> &'static str {
    error.detail
}

fn id_of(study: &DurableStudyCandidate) -> String {
    study.semantic_id().to_string()
}

fn receipt_of<'a>(
    bundle: &'a ProposalBundle,
    study: &DurableStudyCandidate,
) -> &'a ProposalReceipt {
    bundle.proposal(&id_of(study)).expect("the bundle holds the candidate")
}

fn law(measured: &[u32], joint: bool, population: &str) -> ObservedLaw {
    ObservedLaw {
        population: Arc::from(population),
        measured: measured.iter().copied().map(v).collect(),
        joint,
    }
}

fn arrive_law(law: ObservedLaw, sample_size: u64) -> Arrival {
    Arrival {
        snapshot_id: "snapshot:delivered".into(),
        sample_size,
        evidence: ArrivedEvidence::ObservedLaw(law),
    }
}

// ------------------------------------------------------------- receipt content

#[test]
fn x6_receipt_binds_failure_delta_derivation_cost_lineage_value_and_decision() {
    let fx = backdoor();
    let wide = receipt_of(&fx.bundle, &fx.wide);
    // Base failure: the one joint-law obligation of the frozen contract.
    assert_eq!(wide.base_failure.family, "backdoor");
    assert_eq!(wide.base_failure.contract, "backdoor:pop:treatment=0:outcome=1");
    assert_eq!(wide.base_failure.obligation_ids.len(), 1);
    assert_eq!(wide.base_failure.premises_digest, fx.repair.premises_digest);
    assert_eq!(wide.base_failure.data_digest, fx.repair.data_digest);
    // Hypothetical delta and its verified derivation.
    assert_eq!(wide.hypothetical.classification, "verified_sufficient");
    assert_eq!(wide.hypothetical.delta_regimes, 1);
    assert!(wide.hypothetical.derivation_digest.is_some() && wide.hypothetical.derivation_verified);
    assert_eq!(wide.hypothetical.addressed.len(), 1);
    // Cost, size and lineage.
    assert_eq!((wide.cost.units, wide.cost.unit_label.as_str()), (20, "USD"));
    assert_eq!((wide.cost.sample_budget, wide.cost.sample_size), (SAMPLES, SAMPLES));
    assert!(wide.lineage.snapshots.contains(&"snapshot:a".to_owned()));
    assert!(wide.lineage.snapshots.contains(&"provider:lab/snap".to_owned()));
    // Valuation: net 1/4 - 20 * 0.01 = 1/20, ranked second behind the cheaper study.
    let net = f64::from_bits(wide.valuation.net_value_bits.unwrap());
    assert!((net - 0.05).abs() < 1e-12, "{net}");
    assert!((f64::from_bits(wide.valuation.evsi_bits) - 0.25).abs() < 1e-12);
    assert_eq!(wide.valuation.rank, 1);
    let cheap = receipt_of(&fx.bundle, &fx.cheap);
    assert_eq!(cheap.valuation.rank, 0);
    assert!((f64::from_bits(cheap.valuation.net_value_bits.unwrap()) - 0.075).abs() < 1e-12);
    assert_eq!(cheap.hypothetical.classification, "insufficient");
    assert!(cheap.hypothetical.derivation_digest.is_none());
    assert_ne!(cheap.valuation.signal_identity, wide.valuation.signal_identity);
    // Decision receipt.
    assert_eq!(wide.decision.contract_identity, "contract-1");
    assert_eq!(wide.decision.utility_unit, "utility");
    assert_eq!(wide.decision.ranking_identity, fx.ranking.ranking_identity);
    assert_eq!(wide.ranking_digest, fx.ranking.digest);
    assert_eq!(wide.repair_report_digest, fx.repair.report_digest);
    assert_eq!(wide.identity, wide.compute_identity());
    verify(&fx.bundle, &fx.repair, &fx.ranking).unwrap();
}

#[test]
fn x6_receipt_never_marks_a_hypothetical_derivation_as_available() {
    let fx = transport();
    let receipt = receipt_of(&fx.bundle, &fx.experiment);
    // The derivation is verified, yet the receipt holds no available evidence.
    assert_eq!(receipt.hypothetical.classification, "verified_sufficient");
    assert!(receipt.hypothetical.derivation_verified);
    assert_eq!(receipt.evidence_state(), EvidenceState::Hypothetical);
    assert!(!receipt.is_available_evidence());
    // Presenting the hypothetical delta itself as an arrival is refused.
    let proposed = fx.family.hypothetical_regimes(&[&fx.experiment]).unwrap();
    assert!(proposed.iter().all(|r| r.evidence_kind == EvidenceKind::Proposed));
    let arrival = Arrival {
        snapshot_id: "snapshot:copy".into(),
        sample_size: SAMPLES,
        evidence: ArrivedEvidence::CatalogDelta(proposed),
    };
    let error = on_arrival(
        receipt,
        RepairFamilyRef::Transport(&fx.family),
        &fx.experiment,
        &arrival,
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(detail(&error), "proposal_receipt.hypothetical_not_evidence");
    assert_eq!(error.code, "transport_missing_evidence");
}

// ------------------------------------------------------------------- bundle

#[test]
fn x6_receipt_bundle_identity_is_invariant_to_proposal_order() {
    let fx = backdoor();
    assert_eq!(fx.bundle.proposals.len(), 2);
    assert!(fx.bundle.proposals[0].candidate_id < fx.bundle.proposals[1].candidate_id);
    let mut reversed = fx.bundle.proposals.clone();
    reversed.reverse();
    let rebuilt = ProposalBundle::from_receipts(reversed.clone()).unwrap();
    assert_eq!(rebuilt, fx.bundle);
    assert_eq!(rebuilt.identity, fx.bundle.identity);
    // Even a stored reorder does not move the identity (the verifier refuses it instead).
    let mut shuffled = fx.bundle.clone();
    shuffled.proposals = reversed;
    assert_eq!(shuffled.compute_identity(), fx.bundle.identity);
    let error = verify(&shuffled.sealed(), &fx.repair, &fx.ranking).unwrap_err();
    assert_eq!(detail(&error), "proposal_receipt.non_canonical_order");
    // Candidate order at the producers does not change the bundle either.
    let mut ranking = fx.ranking.clone();
    ranking.candidates.reverse();
    assert_eq!(ProposalBundle::build(&fx.repair, &ranking).unwrap().identity, fx.bundle.identity);
}

#[test]
fn x6_receipt_bundle_assembly_refuses_empty_duplicate_and_foreign_receipts() {
    let fx = backdoor();
    let error = ProposalBundle::from_receipts(vec![]).unwrap_err();
    assert_eq!(detail(&error), "proposal_receipt.empty_bundle");
    let one = fx.bundle.proposals[0].clone();
    let error = ProposalBundle::from_receipts(vec![one.clone(), one.clone()]).unwrap_err();
    assert_eq!(detail(&error), "proposal_receipt.duplicate_candidate");
    let mut foreign = fx.bundle.proposals[1].clone();
    foreign.ranking_digest = "another-ranking".into();
    let error = ProposalBundle::from_receipts(vec![one, foreign.sealed()]).unwrap_err();
    assert_eq!(detail(&error), "proposal_receipt.bundle_inconsistent");
}

type Mutation = (&'static str, fn(&mut ProposalReceipt), &'static str);

const MUTATIONS: &[Mutation] = &[
    (
        "obligation id",
        |r| r.base_failure.obligation_ids.push("extra".into()),
        "proposal_receipt.obligation_mismatch",
    ),
    (
        "family",
        |r| r.base_failure.family = "transport".into(),
        "proposal_receipt.base_failure_mismatch",
    ),
    (
        "contract",
        |r| r.base_failure.contract = "other".into(),
        "proposal_receipt.base_failure_mismatch",
    ),
    (
        "premises digest",
        |r| r.base_failure.premises_digest = "x".into(),
        "proposal_receipt.base_failure_mismatch",
    ),
    (
        "data digest",
        |r| r.base_failure.data_digest = "x".into(),
        "proposal_receipt.base_failure_mismatch",
    ),
    (
        "delta digest",
        |r| r.hypothetical.delta_digest = "x".into(),
        "proposal_receipt.delta_mismatch",
    ),
    (
        "derivation digest",
        |r| r.hypothetical.derivation_digest = Some("x".into()),
        "proposal_receipt.derivation_mismatch",
    ),
    (
        "classification",
        |r| r.hypothetical.classification = "not_certified".into(),
        "proposal_receipt.derivation_mismatch",
    ),
    ("delta size", |r| r.hypothetical.delta_regimes += 1, "proposal_receipt.derivation_mismatch"),
    ("cost units", |r| r.cost.units += 1, "proposal_receipt.cost_mismatch"),
    ("cost label", |r| r.cost.unit_label = "EUR".into(), "proposal_receipt.cost_mismatch"),
    ("sample size", |r| r.cost.sample_size += 1, "proposal_receipt.cost_mismatch"),
    (
        "source digest",
        |r| r.lineage.source_digest = "x".into(),
        "proposal_receipt.source_digest_mismatch",
    ),
    (
        "snapshot lineage",
        |r| r.lineage.snapshots.push("snapshot:forged".into()),
        "proposal_receipt.lineage_mismatch",
    ),
    (
        "request fingerprint",
        |r| r.valuation.signal_request_fingerprint = "x".into(),
        "proposal_receipt.signal_mismatch",
    ),
    (
        "signal identity",
        |r| r.valuation.signal_identity = "x".into(),
        "proposal_receipt.signal_mismatch",
    ),
    ("net value", |r| r.valuation.net_value_bits = Some(1), "proposal_receipt.value_mismatch"),
    ("evsi", |r| r.valuation.evsi_bits ^= 1, "proposal_receipt.value_mismatch"),
    ("rank", |r| r.valuation.rank += 1, "proposal_receipt.value_mismatch"),
    (
        "decision contract",
        |r| r.decision.contract_identity = "contract-2".into(),
        "proposal_receipt.contract_mismatch",
    ),
    (
        "action set",
        |r| r.decision.action_ids_digest = "x".into(),
        "proposal_receipt.contract_mismatch",
    ),
    (
        "ranking identity",
        |r| r.decision.ranking_identity = "x".into(),
        "proposal_receipt.contract_mismatch",
    ),
    (
        "repair digest",
        |r| r.repair_report_digest = "x".into(),
        "proposal_receipt.repair_digest_mismatch",
    ),
    (
        "ranking digest",
        |r| r.ranking_digest = "x".into(),
        "proposal_receipt.ranking_digest_mismatch",
    ),
];

#[test]
fn x6_receipt_every_bound_digest_refuses_when_mutated() {
    let fx = backdoor();
    let wide_id = id_of(&fx.wide);
    for &(name, mutate, expected) in MUTATIONS {
        // Unsealed: the mutated field disagrees with the artifacts.
        let mut bundle = fx.bundle.clone();
        let receipt = bundle.proposals.iter_mut().find(|p| p.candidate_id == wide_id).unwrap();
        mutate(receipt);
        let error = verify(&bundle, &fx.repair, &fx.ranking).unwrap_err();
        assert_eq!(detail(&error), expected, "unsealed {name}");
        // Resealed receipt and bundle: only the independent recomputation can refuse it.
        let receipt = bundle.proposals.iter_mut().find(|p| p.candidate_id == wide_id).unwrap();
        *receipt = receipt.clone().sealed();
        let error = verify(&bundle.sealed(), &fx.repair, &fx.ranking).unwrap_err();
        assert_eq!(detail(&error), expected, "resealed {name}");
    }
}

#[test]
fn x6_receipt_identity_and_bundle_digests_bind() {
    let fx = backdoor();
    let mut bundle = fx.bundle.clone();
    bundle.proposals[0].identity = "forged".into();
    let error = verify(&bundle, &fx.repair, &fx.ranking).unwrap_err();
    assert_eq!(detail(&error), "proposal_receipt.identity_mismatch");

    let mut bundle = fx.bundle.clone();
    bundle.identity = "forged".into();
    let error = verify(&bundle, &fx.repair, &fx.ranking).unwrap_err();
    assert_eq!(detail(&error), "proposal_receipt.bundle_identity_mismatch");

    let mut bundle = fx.bundle.clone();
    bundle.repair_report_digest = "x".into();
    let error = verify(&bundle.sealed(), &fx.repair, &fx.ranking).unwrap_err();
    assert_eq!(detail(&error), "proposal_receipt.repair_digest_mismatch");

    let mut bundle = fx.bundle.clone();
    bundle.ranking_digest = "x".into();
    let error = verify(&bundle.sealed(), &fx.repair, &fx.ranking).unwrap_err();
    assert_eq!(detail(&error), "proposal_receipt.ranking_digest_mismatch");

    let mut bundle = fx.bundle.clone();
    bundle.decision_contract_identity = "contract-2".into();
    let error = verify(&bundle.sealed(), &fx.repair, &fx.ranking).unwrap_err();
    assert_eq!(detail(&error), "proposal_receipt.contract_mismatch");

    // A bundle that drops a ranked candidate no longer covers the ranking.
    let mut bundle = fx.bundle.clone();
    bundle.proposals.pop();
    let error = verify(&bundle.sealed(), &fx.repair, &fx.ranking).unwrap_err();
    assert_eq!(detail(&error), "proposal_receipt.candidate_set_mismatch");

    // A proposal with an unknown candidate id.
    let mut bundle = fx.bundle.clone();
    bundle.proposals[0].candidate_id = "sc1:unknown".into();
    let error = verify(&bundle.sealed(), &fx.repair, &fx.ranking).unwrap_err();
    assert_eq!(detail(&error), "proposal_receipt.candidate_not_in_ranking");
}

// ------------------------------------------------------- cross-artifact checks

#[test]
fn x6_receipt_candidate_in_repair_but_not_in_ranking_is_refused() {
    let fx = backdoor();
    let mut ranking = fx.ranking.clone();
    ranking.candidates.retain(|c| c.semantic_id != id_of(&fx.wide));
    ranking.reseal().unwrap();
    // The ranking is honestly sealed, so only the cross-check can refuse the bundle.
    let error = verify(&fx.bundle, &fx.repair, &ranking).unwrap_err();
    assert_eq!(detail(&error), "proposal_receipt.candidate_not_in_ranking");
    // A bundle cannot be built over a ranking that values a candidate the repair never saw.
    let mut repair = fx.repair.clone();
    repair.premises.candidates.retain(|c| c.semantic_id != id_of(&fx.wide));
    let repair = repair.sealed().unwrap();
    let error = ProposalBundle::build(&repair, &fx.ranking).unwrap_err();
    assert_eq!(detail(&error), "proposal_receipt.candidate_not_in_repair");
}

#[test]
fn x6_receipt_candidate_in_ranking_but_not_in_repair_is_refused() {
    let fx = backdoor();
    let mut repair = fx.repair.clone();
    repair.premises.candidates.retain(|c| c.semantic_id != id_of(&fx.cheap));
    let repair = repair.sealed().unwrap();
    let error = verify(&fx.bundle, &repair, &fx.ranking).unwrap_err();
    assert_eq!(detail(&error), "proposal_receipt.candidate_not_in_repair");
}

#[test]
fn x6_receipt_a_candidate_the_repair_did_not_evaluate_cannot_be_proposed() {
    let fx = backdoor();
    let mut repair = fx.repair.clone();
    for outcome in &mut repair.report.outcomes {
        if outcome.candidates == [id_of(&fx.cheap)] {
            outcome.classification = "unevaluated".into();
            outcome.delta = None;
        }
    }
    let repair = repair.sealed().unwrap();
    let error = ProposalBundle::build(&repair, &fx.ranking).unwrap_err();
    assert_eq!(detail(&error), "proposal_receipt.candidate_not_proposed");
}

#[test]
fn x6_receipt_a_changed_delta_or_swapped_signal_in_an_artifact_breaks_the_binding() {
    let fx = backdoor();
    // Change the stored hypothetical delta of the sufficient candidate and re-seal: the
    // repair artifact is internally honest but is no longer the one the bundle binds.
    let mut repair = fx.repair.clone();
    for outcome in &mut repair.report.outcomes {
        if outcome.candidates == [id_of(&fx.wide)] {
            if let Some(delta) = &mut outcome.delta {
                delta.pop();
            }
        }
    }
    let repair = repair.sealed().unwrap();
    let error = verify(&fx.bundle, &repair, &fx.ranking).unwrap_err();
    assert_eq!(detail(&error), "proposal_receipt.repair_digest_mismatch");
    // Rebuilding over the changed artifact yields different delta digests.
    let rebuilt = ProposalBundle::build(&repair, &fx.ranking).unwrap();
    assert_ne!(
        receipt_of(&rebuilt, &fx.wide).hypothetical.delta_digest,
        receipt_of(&fx.bundle, &fx.wide).hypothetical.delta_digest
    );
    assert_ne!(rebuilt.identity, fx.bundle.identity);

    // Swap the two candidates' signals inside the ranking and re-seal.
    let mut ranking = fx.ranking.clone();
    let (first, second) = (ranking.candidates[0].clone(), ranking.candidates[1].clone());
    ranking.candidates[0].signal_identity = second.signal_identity;
    ranking.candidates[0].request_fingerprint = second.request_fingerprint;
    ranking.candidates[1].signal_identity = first.signal_identity;
    ranking.candidates[1].request_fingerprint = first.request_fingerprint;
    ranking.reseal().unwrap();
    let error = verify(&fx.bundle, &fx.repair, &ranking).unwrap_err();
    assert_eq!(detail(&error), "proposal_receipt.ranking_digest_mismatch");

    // Swap the signals between the two receipts of the bundle and re-seal it.
    let mut bundle = fx.bundle.clone();
    let (a, b) = (bundle.proposals[0].valuation.clone(), bundle.proposals[1].valuation.clone());
    bundle.proposals[0].valuation.signal_identity = b.signal_identity;
    bundle.proposals[0].valuation.signal_request_fingerprint = b.signal_request_fingerprint;
    bundle.proposals[1].valuation.signal_identity = a.signal_identity;
    bundle.proposals[1].valuation.signal_request_fingerprint = a.signal_request_fingerprint;
    bundle.proposals = bundle.proposals.iter().map(|p| p.clone().sealed()).collect();
    let error = verify(&bundle.sealed(), &fx.repair, &fx.ranking).unwrap_err();
    assert_eq!(detail(&error), "proposal_receipt.signal_mismatch");
}

#[test]
fn x6_receipt_unsealed_artifacts_are_refused_before_any_binding_is_trusted() {
    let fx = backdoor();
    let mut repair = fx.repair.clone();
    repair.report.obligations.clear();
    let error = verify(&fx.bundle, &repair, &fx.ranking).unwrap_err();
    assert_eq!(detail(&error), "proposal_receipt.repair_digest_mismatch");
    let mut ranking = fx.ranking.clone();
    ranking.candidates[0].evsi += 0.5;
    let error = verify(&fx.bundle, &fx.repair, &ranking).unwrap_err();
    assert_eq!(detail(&error), "proposal_receipt.ranking_digest_mismatch");
}

#[test]
fn x6_receipt_a_ranking_that_costs_a_candidate_differently_is_refused() {
    let fx = backdoor();
    let mut ranking = fx.ranking.clone();
    ranking.candidates[0].cost_amount += 1.0;
    ranking.reseal().unwrap();
    let error = ProposalBundle::build(&fx.repair, &ranking).unwrap_err();
    assert_eq!(detail(&error), "proposal_receipt.cost_mismatch");
}

// ------------------------------------------------------------------ on arrival

#[test]
fn x6_receipt_arrival_of_the_exact_proposed_law_is_verified_by_the_family_checker() {
    let fx = backdoor();
    let receipt = receipt_of(&fx.bundle, &fx.wide);
    let arrival = arrive_law(law(&[T, Y, Z1, Z2], true, "pop"), SAMPLES);
    let verdict =
        on_arrival(receipt, RepairFamilyRef::Backdoor(&fx.family), &fx.wide, &arrival, &ctx())
            .unwrap();
    match verdict {
        ArrivalVerdict::Verified { checker, steps, exact, snapshot_id } => {
            assert_eq!(checker, "backdoor.adjustment");
            assert_eq!(steps[1], "adjustment_set:[2,3]");
            assert!(exact);
            assert_eq!(snapshot_id, "snapshot:delivered");
        }
        other => panic!("expected Verified, got {other:?}"),
    }
}

#[test]
fn x6_receipt_arrival_missing_the_joint_law_is_still_insufficient() {
    let fx = backdoor();
    let receipt = receipt_of(&fx.bundle, &fx.wide);
    // The right variables in the right population, but as separate marginals.
    let marginals = arrive_law(law(&[T, Y, Z1, Z2], false, "pop"), SAMPLES);
    let verdict =
        on_arrival(receipt, RepairFamilyRef::Backdoor(&fx.family), &fx.wide, &marginals, &ctx())
            .unwrap();
    match verdict {
        ArrivalVerdict::StillInsufficient { reasons } => assert!(!reasons.is_empty()),
        other => panic!("expected StillInsufficient, got {other:?}"),
    }
    // A joint law that lacks Z2 leaves the second back-door path open.
    let narrow = arrive_law(law(&[T, Y, Z1], true, "pop"), SAMPLES);
    let verdict =
        on_arrival(receipt, RepairFamilyRef::Backdoor(&fx.family), &fx.wide, &narrow, &ctx())
            .unwrap();
    assert!(matches!(verdict, ArrivalVerdict::StillInsufficient { .. }), "{verdict:?}");
}

#[test]
fn x6_receipt_arrival_in_a_different_population_invalidates_the_derivation() {
    let fx = backdoor();
    let receipt = receipt_of(&fx.bundle, &fx.wide);
    let elsewhere = arrive_law(law(&[T, Y, Z1, Z2], true, "elsewhere"), SAMPLES);
    let verdict =
        on_arrival(receipt, RepairFamilyRef::Backdoor(&fx.family), &fx.wide, &elsewhere, &ctx())
            .unwrap();
    assert_eq!(
        verdict,
        ArrivalVerdict::Invalidated {
            changed: ChangedPremise::Population {
                proposed: vec!["pop".to_owned()],
                arrived: "elsewhere".to_owned(),
            }
        }
    );
}

#[test]
fn x6_receipt_arrival_that_differs_in_sample_or_shape_is_refused_with_a_typed_reason() {
    let fx = backdoor();
    let receipt = receipt_of(&fx.bundle, &fx.wide);
    let family = RepairFamilyRef::Backdoor(&fx.family);
    let run = |arrival: &Arrival| on_arrival(receipt, family, &fx.wide, arrival, &ctx());

    let short = arrive_law(law(&[T, Y, Z1, Z2], true, "pop"), SAMPLES - 1);
    assert_eq!(detail(&run(&short).unwrap_err()), "proposal_receipt.arrival_sample_mismatch");

    let mut blank = arrive_law(law(&[T, Y, Z1, Z2], true, "pop"), SAMPLES);
    blank.snapshot_id = "  ".into();
    assert_eq!(detail(&run(&blank).unwrap_err()), "proposal_receipt.arrival_invalid");

    let empty = Arrival {
        snapshot_id: "snapshot:none".into(),
        sample_size: SAMPLES,
        evidence: ArrivedEvidence::CatalogDelta(vec![]),
    };
    assert_eq!(detail(&run(&empty).unwrap_err()), "proposal_receipt.arrival_empty");

    // A receipt answers only the candidate and the family it was issued for.
    let arrival = arrive_law(law(&[T, Y, Z1, Z2], true, "pop"), SAMPLES);
    let error = on_arrival(receipt, family, &fx.cheap, &arrival, &ctx()).unwrap_err();
    assert_eq!(detail(&error), "proposal_receipt.candidate_mismatch");
    let other = receipt_of(&fx.bundle, &fx.cheap);
    let error = on_arrival(other, family, &fx.wide, &arrival, &ctx()).unwrap_err();
    assert_eq!(detail(&error), "proposal_receipt.candidate_mismatch");
    let mut forged = receipt.clone();
    forged.hypothetical.delta_digest = "x".into();
    let error = on_arrival(&forged.sealed(), family, &fx.wide, &arrival, &ctx()).unwrap_err();
    assert_eq!(detail(&error), "proposal_receipt.delta_mismatch");
    let mut forged = receipt.clone();
    forged.base_failure.obligation_ids.clear();
    let error = on_arrival(&forged.sealed(), family, &fx.wide, &arrival, &ctx()).unwrap_err();
    assert_eq!(detail(&error), "proposal_receipt.obligation_mismatch");
}

#[test]
fn x6_receipt_transport_arrival_of_the_exact_source_experiment_is_verified() {
    let fx = transport();
    let receipt = receipt_of(&fx.bundle, &fx.experiment);
    let mut regimes = fx.family.hypothetical_regimes(&[&fx.experiment]).unwrap();
    for regime in &mut regimes {
        regime.evidence_kind = EvidenceKind::Available;
    }
    let arrival = Arrival {
        snapshot_id: "snapshot:source-trial".into(),
        sample_size: SAMPLES,
        evidence: ArrivedEvidence::CatalogDelta(regimes),
    };
    let verdict = on_arrival(
        receipt,
        RepairFamilyRef::Transport(&fx.family),
        &fx.experiment,
        &arrival,
        &ctx(),
    )
    .unwrap();
    match verdict {
        ArrivalVerdict::Verified { checker, exact, .. } => {
            assert_eq!(checker, "classical_transport.catalog");
            assert!(exact);
        }
        other => panic!("expected Verified, got {other:?}"),
    }
}

#[test]
fn x6_receipt_transport_arrival_in_another_population_or_regime_is_invalidated() {
    let fx = transport();
    let receipt = receipt_of(&fx.bundle, &fx.experiment);
    let family = RepairFamilyRef::Transport(&fx.family);
    let delivered = |regimes| Arrival {
        snapshot_id: "snapshot:delivered".into(),
        sample_size: SAMPLES,
        evidence: ArrivedEvidence::CatalogDelta(regimes),
    };
    let available = || {
        let mut regimes = fx.family.hypothetical_regimes(&[&fx.experiment]).unwrap();
        for regime in &mut regimes {
            regime.evidence_kind = EvidenceKind::Available;
        }
        regimes
    };
    // The experiment was run in the target environment instead of the source.
    let mut moved = available();
    for regime in &mut moved {
        regime.population = Arc::from("target");
    }
    let verdict = on_arrival(receipt, family, &fx.experiment, &delivered(moved), &ctx()).unwrap();
    assert_eq!(
        verdict,
        ArrivalVerdict::Invalidated {
            changed: ChangedPremise::Population {
                proposed: vec!["source".to_owned()],
                arrived: "target".to_owned(),
            }
        }
    );
    // The right population under another intervention set: an observation, not do(X).
    let mut observed = available();
    for regime in &mut observed {
        regime.kind = antecedent_core::RegimeKind::Observational;
        regime.interventions = Arc::from([]);
    }
    let verdict =
        on_arrival(receipt, family, &fx.experiment, &delivered(observed), &ctx()).unwrap();
    assert_eq!(
        verdict,
        ArrivalVerdict::Invalidated {
            changed: ChangedPremise::Regime {
                population: "source".to_owned(),
                arrived_interventions: vec![],
            }
        }
    );
    // An observed law is a back-door delivery, not a transport one.
    let error = on_arrival(
        receipt,
        family,
        &fx.experiment,
        &arrive_law(law(&[X, TY], true, "source"), SAMPLES),
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(detail(&error), "proposal_receipt.arrival_family_mismatch");
}
