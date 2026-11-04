//! Proposal receipts on a graph with irrelevant nodes.
//!
//! Hand derivation. Nodes T=0, Y=1, Z1=2, Z2=3, I=4, D=5 with edges Z1->T, Z1->Y, Z2->T,
//! Z2->Y, T->Y, I->Z1, T->D. The only minimal back-door set for T -> Y is {Z1, Z2}: I only
//! causes Z1 (its path is already blocked by Z1) and D is a descendant of T and must never be
//! adjusted for. So the study of (T, Y, Z1, Z2) repairs the contract and the study of
//! (T, Y, I, D) does not, although it measures as many variables. Valuation: binary state,
//! prior 1/2, signal accuracy a gives EVSI a - 1/2; accuracy 3/4 at 20 USD gives
//! 1/4 - 1/5 = 1/20, accuracy 5/8 at 5 USD gives 1/8 - 1/20 = 3/40, so the useless study ranks
//! first on net value yet is insufficient, so the proposal rank puts the sufficient study first
//! (rank 0) and the insufficient one second (rank 1): sufficiency before net value.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::{
    ExternalCapability, ExternalScientificObject, ExternalTrustState, ProviderObjectIdentity,
    ScientificQuantity, SignalProviderContract,
};
use antecedent_core::{CancellationToken, DistributionAvailability, ExecutionContext, VariableId};
use antecedent_design::design_ranking_artifact::{DesignRankingArtifactWire, SealInputs, seal};
use antecedent_design::evsi::{
    CostToUtilityMap, EvsiCandidate, EvsiRequest, StudyCostSpec, evaluate_evsi,
};
use antecedent_design::proposal_receipt::{
    Arrival, ArrivalVerdict, ArrivedEvidence, ObservedLaw, ProposalBundle, ProposalReceipt,
    on_arrival,
};
use antecedent_design::signal::{
    ExternalLaw, ExternalSignal, ExternalSignalBody, SignalLimits, SignalProvider, SignalRequest,
};
use antecedent_design::{
    AffineUtility, BackdoorRepairFamily, CandidateDesign, DecisionPrior, DecisionProblem,
    DesignCost, DesignRankConfig, DurableStudyCandidate, ExpectedEvidence, RepairFamilyRef,
    RepairLimits, RepairObjective, RepairReportArtifact, SamplingPlan, StudyCostDeclaration,
    StudyKind, UnitRules, repair,
};
use antecedent_graph::{Dag, DenseNodeId};

// ------------------------------------------------------------------- fixtures

const T: u32 = 0;
const Y: u32 = 1;
const Z1: u32 = 2;
const Z2: u32 = 3;
const I: u32 = 4;
const D: u32 = 5;
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
    let mut graph = Dag::with_variables(6);
    for (from, to) in [(Z1, T), (Z1, Y), (Z2, T), (Z2, Y), (T, Y), (I, Z1), (T, D)] {
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

mod common;

fn quantity(id: &str) -> ScientificQuantity {
    ScientificQuantity {
        units: "dimensionless".into(),
        functional_id: "state".into(),
        ..common::quantity(id)
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

struct Fx {
    family: BackdoorRepairFamily,
    minimal: DurableStudyCandidate,
    irrelevant: DurableStudyCandidate,
    bundle: ProposalBundle,
}

fn fixture() -> Fx {
    let family = backdoor_family();
    let minimal = observation("minimal", &[T, Y, Z1, Z2], 20);
    let irrelevant = observation("irrelevant", &[T, Y, I, D], 5);
    let candidates = [minimal.clone(), irrelevant.clone()];
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
    let ranking = seal_ranking(vec![ranked(&minimal, 0.75), ranked(&irrelevant, 0.625)]);
    let bundle = ProposalBundle::build(&repair, &ranking).unwrap();
    Fx { family, minimal, irrelevant, bundle }
}

fn receipt<'a>(bundle: &'a ProposalBundle, s: &DurableStudyCandidate) -> &'a ProposalReceipt {
    bundle.proposal(s.semantic_id().as_ref()).expect("candidate in bundle")
}

fn law(measured: &[u32]) -> Arrival {
    Arrival {
        snapshot_id: "snapshot:delivered".into(),
        sample_size: SAMPLES,
        evidence: ArrivedEvidence::ObservedLaw(ObservedLaw {
            population: Arc::from("pop"),
            measured: measured.iter().copied().map(v).collect(),
            joint: true,
        }),
    }
}

#[test]
fn a_sufficient_study_ranks_before_an_insufficient_one_with_larger_net_value() {
    let fx = fixture();
    let min = receipt(&fx.bundle, &fx.minimal);
    let irr = receipt(&fx.bundle, &fx.irrelevant);
    assert_eq!(min.hypothetical.classification, "verified_sufficient");
    assert!(min.hypothetical.derivation_verified);
    assert_eq!(irr.hypothetical.classification, "insufficient");
    assert!(irr.hypothetical.derivation_digest.is_none());
    assert!(!irr.hypothetical.unmet.is_empty());
    // Net values 3/40 (irrelevant) and 1/20 (minimal); sufficiency decides the rank.
    let net = |r: &ProposalReceipt| f64::from_bits(r.valuation.net_value_bits.unwrap());
    assert!((net(irr) - 0.075).abs() < 1e-12);
    assert!((net(min) - 0.05).abs() < 1e-12);
    assert_eq!((min.valuation.rank, irr.valuation.rank), (0, 1));
}

#[test]
fn arrival_derives_the_adjustment_set_without_the_irrelevant_nodes() {
    let fx = fixture();
    let min = receipt(&fx.bundle, &fx.minimal);
    let family = RepairFamilyRef::Backdoor(&fx.family);
    match on_arrival(min, family, &fx.minimal, &law(&[T, Y, Z1, Z2]), &ctx()).unwrap() {
        ArrivalVerdict::Verified { steps, exact, .. } => {
            assert_eq!(steps[1], "adjustment_set:[2,3]");
            assert!(exact);
        }
        other => panic!("expected Verified, got {other:?}"),
    }
    // A law over the irrelevant nodes does not close the back-door paths.
    let verdict = on_arrival(min, family, &fx.minimal, &law(&[T, Y, I, D]), &ctx()).unwrap();
    assert!(matches!(verdict, ArrivalVerdict::StillInsufficient { .. }), "{verdict:?}");
    // Even with every variable delivered, the set adjusted for excludes I and D.
    if let ArrivalVerdict::Verified { steps, .. } =
        on_arrival(min, family, &fx.minimal, &law(&[T, Y, Z1, Z2, I, D]), &ctx()).unwrap()
    {
        assert_eq!(steps[1], "adjustment_set:[2,3]");
    }
}
