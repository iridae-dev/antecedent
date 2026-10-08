//! A0 remainders (Rust half): lineage of sensitivity inputs and the study-ranking
//! provider, the inverse/design refusal families under the F25 envelope validator,
//! evidence-obligation identity, and the cross-surface parity hooks.
//!
//! Every oracle is written by hand:
//!
//! * the root `decision:contract-1` link digest is BLAKE3 over the little-endian `u64`
//!   length and bytes of its id, then of its stage name `decision_contract`, with no
//!   parents (recomputed here, not read from the code under test);
//! * the guess decision of the frozen F14 record: binary state, prior 1/2, signals of
//!   accuracy 3/4 and 5/8 have `EVSI = 1/4` and `1/8`; with an identical utility-unit cost
//!   1/10 the net values are `3/20` and `1/40`;
//! * the invariant sensitivity surface: A is `5 - gamma` (5, 4, 3) against B = 1, so A
//!   leads at every grid point.
//!
//! The ignored tests are the cross-surface fixture hooks. They write or read files that do
//! not exist until the coordinator regenerates them:
//!
//! * Rust-built: `ANTECEDENT_WRITE_FIXTURES=1 cargo test -p antecedent-design --test
//!   a0r_parity -- --ignored a0r_regenerate_rust_fixtures`
//! * Python-built: `A0R_WRITE_FIXTURES=1 python -m pytest
//!   python/tests/test_a0r_lifecycle.py -k a0r_regenerate_python_fixtures`
//! * Rust consumer of the Python-built files: `cargo test -p antecedent-design --test
//!   a0r_parity -- --ignored a0r_consume_python_fixtures`
#![allow(clippy::float_cmp, reason = "exact hand-derived values")]

use std::collections::BTreeSet;
use std::sync::Arc;

use antecedent_core::{
    CancellationToken, CompositionStage, EvidenceObligation, EvidenceObligationError,
    EvidenceObligationKind, EvidenceObligationSpec, ExternalCapability, ExternalRefusal,
    ExternalScientificObject, ExternalTrustState, ObligationProvenance, ObligationRegime,
    ObligationScope, ProvenanceChain, ProvenanceChainError, ProviderObjectIdentity, QuantityRole,
    ScientificQuantity, SignalProviderContract, VariableId,
};
use antecedent_design::decision_contract::StructuralPolicy;
use antecedent_design::decision_eval::DecisionEvalError;
use antecedent_design::design_ranking_artifact::{
    CandidateWire, ConsumeExpectation, DESIGN_RANKING_CLAIM_LINK_ID, DesignRankingArtifactError,
    DesignRankingArtifactWire, SealInputs, consume, seal,
};
use antecedent_design::evsi::{
    CostToUtilityMap, EvsiCandidate, EvsiError, EvsiRequest, StudyCostSpec, evaluate_evsi,
};
use antecedent_design::inverse_query::InverseQueryError;
use antecedent_design::inverse_query_artifact::artifact_refusal;
use antecedent_design::ranking::RankingError;
use antecedent_design::sensitivity_decision::{
    ActionUtility, AssumptionCoordinate, AssumptionRangeStatement, PointSupport, SamplingStatus,
    SensitivityArtifact, SensitivityDecisionSpec, SensitivityParts, SensitivityProvenance,
    SurfaceQuantity, UncertaintyRelationship, UtilityTerm, contract_from_artifact,
    evaluate_sensitivity_decision,
};
use antecedent_design::signal::{
    ExternalLaw, ExternalSignal, ExternalSignalBody, SignalError, SignalLimits, SignalProvider,
    SignalRequest,
};
use antecedent_design::{
    AffineUtility, BackdoorRepairFamily, CandidateDesign, DecisionPrior, DecisionProblem,
    DesignCost, DesignRankConfig, RepairFamily, SamplingPlan,
};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_io::error::IoError;
use antecedent_io::external_claim_artifact::lineage_wire;
use antecedent_io::quantity_wire::ScientificQuantityWire;
use antecedent_io::sensitivity_artifact::SENSITIVITY_CLAIM_LINK_ID;

fn fixture_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../conformance/cross_surface")
}

// -- the frozen F14 guess decision (copied constants of design_ranking_artifact.rs) --------

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

fn guess_candidate(id: &str, accuracy: f64) -> EvsiCandidate {
    let req = signal_request(id, 1);
    EvsiCandidate {
        semantic_id: id.into(),
        design: CandidateDesign::IncreaseSamplingRate(SamplingPlan {
            additional_samples: 1,
            cost: DesignCost::zero(),
            tag: 0,
        }),
        provider: external(&req, accuracy_law(accuracy)),
        signal_request: req,
        cost: StudyCostSpec { amount: 0.1, unit: "utility".into() },
        reused_observation_ids: vec!["obs-future".into()],
    }
}

fn utility_map() -> CostToUtilityMap {
    CostToUtilityMap {
        cost_unit: "utility".into(),
        utility_unit: "utility".into(),
        utility_per_cost: 1.0,
    }
}

fn digests() -> Vec<String> {
    vec!["digest-b".into(), "digest-a".into()]
}

fn seal_guess() -> DesignRankingArtifactWire {
    let utility = AffineUtility::new(vec![1.0, 0.0], vec![-1.0, 1.0]).unwrap();
    let problem: DecisionProblem<usize, f64> =
        DecisionProblem::new(vec![0, 1], Arc::new(utility), vec![]);
    let prior = DecisionPrior::Draws(vec![0.0, 1.0]);
    let request = EvsiRequest {
        decision_contract_identity: "contract-1".into(),
        utility_unit: "utility".into(),
        action_ids: vec!["guess0".into(), "guess1".into()],
        candidates: vec![guess_candidate("cand-1", 0.75), guess_candidate("cand-2", 0.625)],
        cost_map: Some(utility_map()),
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
    let report = evaluate_evsi(&problem, &prior, &request, &CancellationToken::new()).unwrap();
    seal(&SealInputs {
        problem: &problem,
        prior: &prior,
        request: &request,
        report: &report,
        source_digests: &digests(),
    })
    .unwrap()
}

fn candidate_mut<'a>(wire: &'a mut DesignRankingArtifactWire, id: &str) -> &'a mut CandidateWire {
    wire.candidates.iter_mut().find(|c| c.semantic_id == id).unwrap()
}

fn signal_identity(wire: &DesignRankingArtifactWire, id: &str) -> String {
    wire.candidates.iter().find(|c| c.semantic_id == id).unwrap().signal_identity.clone()
}

fn digest_of(chain: &ProvenanceChain, id: &str) -> String {
    chain.digest_of(id).unwrap().to_owned()
}

/// Independent recomputation of a link digest from the documented byte layout.
fn expected_link_digest(id: &str, stage: &str, parent_digests: &[&str]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&u64::try_from(id.len()).unwrap().to_le_bytes());
    hasher.update(id.as_bytes());
    hasher.update(&u64::try_from(stage.len()).unwrap().to_le_bytes());
    hasher.update(stage.as_bytes());
    for parent in parent_digests {
        hasher.update(parent.as_bytes());
    }
    hasher.finalize().to_hex().to_string()
}

fn ranking_ids(wire: &DesignRankingArtifactWire) -> Vec<String> {
    vec![
        "decision:contract-1".into(),
        "distribution:digest-a".into(),
        "distribution:digest-b".into(),
        "provider:signal:lab/signal-cand-1@v1#snap".into(),
        format!("signal:cand-1:{}", signal_identity(wire, "cand-1")),
        "provider:signal:lab/signal-cand-2@v1#snap".into(),
        format!("signal:cand-2:{}", signal_identity(wire, "cand-2")),
        DESIGN_RANKING_CLAIM_LINK_ID.into(),
    ]
}

// -- (2) provenance: the study-ranking provider --------------------------------------------

#[test]
fn a0r_design_ranking_chain_has_the_literal_order_stages_and_an_independent_root_digest() {
    use CompositionStage as S;
    let wire = seal_guess();
    let chain = wire.provenance_chain().unwrap();

    let ids: Vec<&str> = chain.links().iter().map(|l| l.id.as_str()).collect();
    let expected = ranking_ids(&wire);
    assert_eq!(ids, expected.iter().map(String::as_str).collect::<Vec<_>>());
    let stages: Vec<S> = chain.links().iter().map(|l| l.stage).collect();
    assert_eq!(
        stages,
        [
            S::DecisionContract,
            S::DistributionArtifact,
            S::DistributionArtifact,
            S::ExternalProvider,
            S::StudyRankingProvider,
            S::ExternalProvider,
            S::StudyRankingProvider,
            S::Claim,
        ]
    );
    assert_eq!(DESIGN_RANKING_CLAIM_LINK_ID, "design_ranking_result");

    // The root digest is the documented BLAKE3 layout, recomputed here.
    assert_eq!(
        digest_of(&chain, "decision:contract-1"),
        expected_link_digest("decision:contract-1", "decision_contract", &[])
    );
    // A child digest chains its parent digests in declared order.
    let decision = digest_of(&chain, "decision:contract-1");
    let provider = digest_of(&chain, "provider:signal:lab/signal-cand-1@v1#snap");
    assert_eq!(
        digest_of(&chain, &expected[4]),
        expected_link_digest(
            &expected[4],
            "study_ranking_provider",
            &[decision.as_str(), provider.as_str()]
        )
    );

    // The reported ranking is derived from the contract, sources, providers and signals.
    assert_eq!(
        chain.require_stages(
            DESIGN_RANKING_CLAIM_LINK_ID,
            &[
                S::DecisionContract,
                S::DistributionArtifact,
                S::ExternalProvider,
                S::StudyRankingProvider
            ]
        ),
        Ok(())
    );
    assert_eq!(
        chain.require_stages(DESIGN_RANKING_CLAIM_LINK_ID, &[S::Data]),
        Err(ProvenanceChainError::MissingStage(S::Data))
    );
    // One candidate's signal stands on its own provider and the contract, not the other's.
    let behind: Vec<&str> =
        chain.lineage(&expected[4]).unwrap().into_iter().map(|l| l.id.as_str()).collect();
    assert_eq!(
        behind,
        ["decision:contract-1", "provider:signal:lab/signal-cand-1@v1#snap", expected[4].as_str()]
    );
}

#[test]
fn a0r_design_ranking_chain_digests_move_with_every_upstream_identity() {
    let wire = seal_guess();
    let base = wire.provenance_chain().unwrap();
    let ids = ranking_ids(&wire);
    let (decision, dist_a, signal_1, signal_2) = (&ids[0], &ids[1], &ids[4], &ids[6]);

    // Candidate order does not change the chain.
    let mut reordered = wire.clone();
    reordered.candidates.reverse();
    reordered.source_digests.reverse();
    assert_eq!(reordered.provenance_chain().unwrap(), base);

    // A changed decision contract identity moves the decision, every signal and the result.
    let mut changed = wire.clone();
    changed.decision.contract_identity = "contract-2".into();
    let chain = changed.provenance_chain().unwrap();
    assert!(chain.digest_of(decision).is_err(), "the old contract link is gone");
    assert_ne!(
        digest_of(&chain, "decision:contract-2"),
        digest_of(&base, decision),
        "a different contract is a different root"
    );
    assert_ne!(digest_of(&chain, signal_1), digest_of(&base, signal_1));
    assert_ne!(
        digest_of(&chain, DESIGN_RANKING_CLAIM_LINK_ID),
        digest_of(&base, DESIGN_RANKING_CLAIM_LINK_ID)
    );

    // A changed source digest moves the result only; signals do not read the source.
    let mut changed = wire.clone();
    changed.source_digests = vec!["digest-z".into(), "digest-b".into()];
    let chain = changed.provenance_chain().unwrap();
    assert!(chain.digest_of(dist_a).is_err());
    assert_eq!(digest_of(&chain, decision), digest_of(&base, decision));
    assert_eq!(digest_of(&chain, signal_1), digest_of(&base, signal_1));
    assert_ne!(
        digest_of(&chain, DESIGN_RANKING_CLAIM_LINK_ID),
        digest_of(&base, DESIGN_RANKING_CLAIM_LINK_ID)
    );

    // A changed signal identity of one candidate moves that signal and the result only.
    let mut changed = wire.clone();
    candidate_mut(&mut changed, "cand-2").signal_identity = "other-signal".into();
    let chain = changed.provenance_chain().unwrap();
    assert!(chain.digest_of(signal_2).is_err());
    assert_eq!(digest_of(&chain, signal_1), digest_of(&base, signal_1));
    assert_ne!(
        digest_of(&chain, DESIGN_RANKING_CLAIM_LINK_ID),
        digest_of(&base, DESIGN_RANKING_CLAIM_LINK_ID)
    );

    // A changed provider snapshot of one candidate moves its provider, its signal and the
    // result; the other candidate's signal is untouched.
    let mut changed = wire.clone();
    candidate_mut(&mut changed, "cand-1").provider.as_mut().unwrap().snapshot_id = "snap-2".into();
    let chain = changed.provenance_chain().unwrap();
    assert!(chain.digest_of("provider:signal:lab/signal-cand-1@v1#snap").is_err());
    assert!(chain.digest_of("provider:signal:lab/signal-cand-1@v1#snap-2").is_ok());
    assert_ne!(digest_of(&chain, signal_1), digest_of(&base, signal_1));
    assert_eq!(digest_of(&chain, signal_2), digest_of(&base, signal_2));
    assert_ne!(
        digest_of(&chain, DESIGN_RANKING_CLAIM_LINK_ID),
        digest_of(&base, DESIGN_RANKING_CLAIM_LINK_ID)
    );
}

#[test]
fn a0r_a_consumer_derives_the_same_chain_from_the_exported_bytes() {
    let wire = seal_guess();
    let chain = wire.provenance_chain().unwrap();
    let bytes = wire.to_bytes("a0r-ranking").unwrap();
    let expectation = ConsumeExpectation {
        artifact_identity: Some(wire.digest.clone()),
        decision_contract_identity: Some("contract-1".into()),
        source_digests: Some(digests()),
        cost_mapping: Some(Some(utility_map())),
        ..ConsumeExpectation::default()
    };
    let consumed = consume(&bytes, &expectation).unwrap();
    assert_eq!(consumed.identity, wire.digest);
    let loaded = DesignRankingArtifactWire::from_bytes(&bytes).unwrap();
    assert_eq!(loaded.provenance_chain().unwrap(), chain);
}

#[test]
fn a0r_python_composition_lineage_rows_reproduce_the_ranking_digests() {
    // Python builds these rows from the same constants and gets its digests from the same
    // native chain function; equal rows therefore mean equal digests on both surfaces.
    let wire = seal_guess();
    let chain = wire.provenance_chain().unwrap();
    let ids = ranking_ids(&wire);
    let s = |i: usize| ids[i].as_str();
    let rows: Vec<(&str, &str, Vec<&str>)> = vec![
        (s(0), "decision_contract", vec![]),
        (s(1), "distribution_artifact", vec![]),
        (s(2), "distribution_artifact", vec![]),
        (s(3), "external_provider", vec![]),
        (s(4), "study_ranking_provider", vec![s(0), s(3)]),
        (s(5), "external_provider", vec![]),
        (s(6), "study_ranking_provider", vec![s(0), s(5)]),
        (s(7), "claim", vec![s(0), s(1), s(2), s(4), s(6)]),
    ];
    let borrowed: Vec<(&str, &str, &[&str])> =
        rows.iter().map(|(id, stage, parents)| (*id, *stage, parents.as_slice())).collect();
    let computed = lineage_wire(&borrowed).unwrap();
    assert_eq!(computed.len(), chain.links().len());
    for row in &computed {
        assert_eq!(row.digest, digest_of(&chain, &row.id), "{}", row.id);
    }
}

// -- (2) provenance: sensitivity inputs ----------------------------------------------------

fn scientific(name: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: name.into(),
        variable_name: name.into(),
        role: QuantityRole::Outcome,
        units: "utils".into(),
        population_id: "target".into(),
        regime_id: "do(a=1)".into(),
        horizon: 0,
        functional_id: "sensitivity_surface".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn point(name: &str, values: &[f64]) -> SurfaceQuantity {
    SurfaceQuantity {
        quantity: ScientificQuantityWire::from(&scientific(name)),
        lower: values.to_vec(),
        upper: values.to_vec(),
    }
}

fn act(id: &str, quantity: &str) -> ActionUtility {
    ActionUtility { id: id.into(), utility: UtilityTerm::quantity(quantity) }
}

fn invariant_parts() -> SensitivityParts {
    SensitivityParts {
        coordinate: AssumptionCoordinate {
            id: "gamma".into(),
            scale: "sensitivity_parameter".into(),
            units: "dimensionless".into(),
            minimum: 0.0,
            maximum: 2.0,
        },
        grid: vec![0.0, 1.0, 2.0],
        support: vec![PointSupport::Supported; 3],
        quantities: vec![point("ua", &[5.0, 4.0, 3.0]), point("ub", &[1.0, 1.0, 1.0])],
        actions: vec![act("A", "ua"), act("B", "ub")],
        uncertainty: UncertaintyRelationship {
            assumption_range: AssumptionRangeStatement {
                kind: "assumption_range".into(),
                interpretation: "assumption range; not a probability".into(),
            },
            identified_bound: None,
            sampling: SamplingStatus::Withheld {
                reason_code: "cell_not_licensed".into(),
                detail: "joint_sensitivity.interval_withheld".into(),
            },
        },
        provenance: SensitivityProvenance {
            source_kind: "supplied_surface".into(),
            query_binding: "a0r-test".into(),
            provider_snapshot: "snapshot-1".into(),
            source_regime: "regime:1".into(),
            method: "hand-derived surface".into(),
            causal_contract_id: "checked-contract".into(),
            decision_threshold: None,
            source_tipping: vec![],
        },
    }
}

#[test]
fn a0r_sensitivity_chain_links_contract_snapshot_input_and_the_decision_identity() {
    use CompositionStage as S;
    let artifact = SensitivityArtifact::new(invariant_parts()).unwrap();
    let chain = artifact.provenance_chain().unwrap();
    let input = format!("sensitivity_input:{}", artifact.identity().digest);
    let ids: Vec<&str> = chain.links().iter().map(|l| l.id.as_str()).collect();
    assert_eq!(
        ids,
        [
            "causal_contract:checked-contract",
            "snapshot:snapshot-1",
            input.as_str(),
            "sensitivity_claim"
        ]
    );
    assert_eq!(SENSITIVITY_CLAIM_LINK_ID, "sensitivity_claim");
    let stages: Vec<S> = chain.links().iter().map(|l| l.stage).collect();
    assert_eq!(stages, [S::CausalContract, S::Data, S::SensitivityInput, S::Claim]);
    assert_eq!(
        chain.require_stages(
            "sensitivity_claim",
            &[S::CausalContract, S::Data, S::SensitivityInput]
        ),
        Ok(())
    );
    assert_eq!(
        digest_of(&chain, "causal_contract:checked-contract"),
        expected_link_digest("causal_contract:checked-contract", "causal_contract", &[])
    );

    // The decision made over the surface names the same input.
    let contract =
        contract_from_artifact(&artifact, StructuralPolicy::RequireInvariantBestAction).unwrap();
    let result =
        evaluate_sensitivity_decision(&contract, &artifact, &SensitivityDecisionSpec::default())
            .unwrap();
    assert_eq!(result.artifact_identity, artifact.identity().digest);
    assert!(chain.digest_of(&format!("sensitivity_input:{}", result.artifact_identity)).is_ok());
}

#[test]
fn a0r_sensitivity_chain_digests_move_with_the_contract_snapshot_and_numbers() {
    let base_artifact = SensitivityArtifact::new(invariant_parts()).unwrap();
    let base = base_artifact.provenance_chain().unwrap();
    let base_claim = digest_of(&base, "sensitivity_claim");

    let mut parts = invariant_parts();
    parts.provenance.causal_contract_id = "other-contract".into();
    let artifact = SensitivityArtifact::new(parts).unwrap();
    let chain = artifact.provenance_chain().unwrap();
    assert!(chain.digest_of("causal_contract:checked-contract").is_err());
    assert_ne!(digest_of(&chain, "sensitivity_claim"), base_claim);

    let mut parts = invariant_parts();
    parts.provenance.provider_snapshot = "snapshot-2".into();
    let artifact = SensitivityArtifact::new(parts).unwrap();
    let chain = artifact.provenance_chain().unwrap();
    assert!(chain.digest_of("snapshot:snapshot-1").is_err());
    assert_ne!(digest_of(&chain, "sensitivity_claim"), base_claim);
    assert_eq!(
        digest_of(&chain, "causal_contract:checked-contract"),
        digest_of(&base, "causal_contract:checked-contract"),
        "an unchanged upstream keeps its digest"
    );

    // A changed number changes the artifact identity, hence the input and the claim, while
    // the contract and snapshot links are unchanged.
    let mut parts = invariant_parts();
    parts.quantities[0] = point("ua", &[5.0, 4.0, 3.5]);
    let artifact = SensitivityArtifact::new(parts).unwrap();
    assert_ne!(artifact.identity().digest, base_artifact.identity().digest);
    let chain = artifact.provenance_chain().unwrap();
    assert_ne!(digest_of(&chain, "sensitivity_claim"), base_claim);
    assert_eq!(digest_of(&chain, "snapshot:snapshot-1"), digest_of(&base, "snapshot:snapshot-1"));
}

// -- (3) refusal families under the F25 envelope -------------------------------------------

struct Expect {
    code: &'static str,
    stage: &'static str,
    detail: &'static str,
}

fn check(refusal: &ExternalRefusal, expect: &Expect) {
    assert_eq!(refusal.validate(), Ok(()), "{}", refusal.detail);
    assert_eq!(refusal.code, expect.code, "{}", expect.detail);
    assert_eq!(refusal.stage, expect.stage, "{}", expect.detail);
    assert_eq!(refusal.detail, expect.detail);
}

#[test]
fn a0r_inverse_design_signal_update_evsi_and_cost_unit_refusals_validate_and_stay_distinct() {
    let joint = InverseQueryError::Engine(DecisionEvalError::JointLawRequired {
        action: Some("risky".into()),
        supplied_alignment: "independent_marginals",
    })
    .to_refusal();
    check(
        &joint,
        &Expect {
            code: "joint_law_required",
            stage: "inverse_query",
            detail: "decision_evaluation.joint_law_required",
        },
    );
    assert_eq!(joint.offending.as_deref(), Some("risky"));
    assert_eq!(joint.expected.as_deref(), Some("joint"));
    assert_eq!(joint.supplied.as_deref(), Some("independent_marginals"));
    assert!(joint.remedy.is_some());

    let grid = InverseQueryError::InvalidGrid("duplicate action").to_refusal();
    check(
        &grid,
        &Expect {
            code: "invalid_argument",
            stage: "inverse_query",
            detail: "inverse_query.invalid_grid",
        },
    );
    assert_eq!(grid.offending.as_deref(), Some("duplicate action"));

    let no_constraints = InverseQueryError::NoConstraints.to_refusal();
    check(
        &no_constraints,
        &Expect {
            code: "invalid_argument",
            stage: "inverse_query",
            detail: "inverse_query.no_constraints",
        },
    );

    let signal = SignalError::SampleSizeMismatch { declared: 3, requested: 1 }.to_refusal();
    check(
        &signal,
        &Expect {
            code: "design_signal_invalid",
            stage: "declare",
            detail: "signal_provider.sample_size_mismatch",
        },
    );
    assert_eq!(signal.expected.as_deref(), Some("1"));
    assert_eq!(signal.supplied.as_deref(), Some("3"));

    let capability = SignalError::CapabilityMissing(ExternalCapability::Update).to_refusal();
    check(
        &capability,
        &Expect {
            code: "external_capability_missing",
            stage: "declare",
            detail: "signal_provider.capability_missing",
        },
    );
    assert_eq!(capability.capability, Some(ExternalCapability::Update));

    let update = SignalError::PosteriorIncoherent.to_refusal();
    check(
        &update,
        &Expect {
            code: "design_signal_invalid",
            stage: "declare",
            detail: "signal_provider.posterior_incoherent",
        },
    );

    // An EVSI error that wraps a signal refusal keeps the signal family's own detail.
    let wrapped = EvsiError::Signal(SignalError::PosteriorIncoherent).to_refusal();
    check(
        &wrapped,
        &Expect {
            code: "design_signal_invalid",
            stage: "declare",
            detail: "signal_provider.posterior_incoherent",
        },
    );

    let (overlap, units, map, ranking) = evsi_and_cost_unit_refusals();

    // Each family keeps its own namespace and the details are all different.
    let all = [
        &joint,
        &grid,
        &no_constraints,
        &signal,
        &capability,
        &update,
        &wrapped,
        &overlap,
        &units,
        &map,
        &ranking,
    ];
    distinct_details_and_namespaces(&all);
}

/// The EVSI, cost-unit and ranking refusals, checked; returns the four that join the
/// distinctness comparison (overlap, units, map, ranking).
fn evsi_and_cost_unit_refusals()
-> (ExternalRefusal, ExternalRefusal, ExternalRefusal, ExternalRefusal) {
    let overlap = EvsiError::SourceOverlap(vec!["o1".into(), "o2".into()]).to_refusal();
    check(
        &overlap,
        &Expect { code: "design_signal_invalid", stage: "evaluate", detail: "evsi.source_overlap" },
    );
    assert_eq!(overlap.offending.as_deref(), Some("o1,o2"));
    assert!(overlap.remedy.is_some());

    let units = EvsiError::CostUnitsMismatch("candidate study cost unit usd".into()).to_refusal();
    check(
        &units,
        &Expect {
            code: "design_cost_units_mismatch",
            stage: "evaluate",
            detail: "evsi.cost_units_mismatch",
        },
    );
    assert_eq!(units.offending.as_deref(), Some("candidate study cost unit usd"));
    assert!(units.remedy.is_some());

    let map = EvsiError::CostMapRequired.to_refusal();
    check(
        &map,
        &Expect {
            code: "design_cost_units_mismatch",
            stage: "evaluate",
            detail: "evsi.cost_map_required",
        },
    );

    let ranking = RankingError::CostMappingMismatch.to_refusal();
    check(
        &ranking,
        &Expect {
            code: "design_cost_units_mismatch",
            stage: "consume",
            detail: "design_ranking.cost_units_mismatch",
        },
    );
    assert!(ranking.remedy.is_some());

    let digest = DesignRankingArtifactError::DigestMismatch.refusal().unwrap();
    check(
        &digest,
        &Expect {
            code: "design_signal_invalid",
            stage: "consume",
            detail: "design_ranking.digest_mismatch",
        },
    );

    (overlap, units, map, ranking)
}

/// Each family keeps its own namespace and the details are all different.
fn distinct_details_and_namespaces(all: &[&ExternalRefusal; 11]) {
    let details: BTreeSet<&str> = all.iter().map(|r| r.detail.as_str()).collect();
    assert_eq!(details.len(), 10, "the wrapped signal refusal repeats the update detail");
    let namespaces: BTreeSet<&str> =
        all.iter().map(|r| r.detail.split('.').next().unwrap()).collect();
    assert_eq!(
        namespaces,
        BTreeSet::from([
            "decision_evaluation",
            "design_ranking",
            "evsi",
            "inverse_query",
            "signal_provider"
        ])
    );
}

#[test]
fn a0r_inverse_functional_artifact_failures_convert_to_structured_refusals() {
    let wrong = artifact_refusal(&IoError::Refused {
        code: "inverse_functional_unsupported",
        message: "functional_inverse_query.wrong_contract: retained identity differs".into(),
    });
    check(
        &wrong,
        &Expect {
            code: "inverse_functional_unsupported",
            stage: "inverse_query",
            detail: "functional_inverse_query.wrong_contract",
        },
    );
    assert_eq!(wrong.offending.as_deref(), Some("retained identity differs"));

    let global = artifact_refusal(&IoError::Refused {
        code: "inverse_functional_unsupported",
        message: "functional_inverse_query.global_feasibility_claim: a continuous sample".into(),
    });
    check(
        &global,
        &Expect {
            code: "inverse_functional_unsupported",
            stage: "inverse_query",
            detail: "functional_inverse_query.global_feasibility_claim",
        },
    );

    // A message without a written detail falls back to the wrong-contract family.
    let bare = artifact_refusal(&IoError::Refused {
        code: "inverse_functional_unsupported",
        message: "no detail written".into(),
    });
    assert_eq!(bare.detail, "functional_inverse_query.wrong_contract");

    let other = artifact_refusal(&IoError::Convert("truncated payload".into()));
    check(
        &other,
        &Expect {
            code: "invalid_argument",
            stage: "inverse_query",
            detail: "functional_inverse_query.invalid_artifact",
        },
    );
    assert!(other.offending.as_deref().is_some_and(|text| text.contains("truncated payload")));

    // The artifact family is distinct from the query family.
    let query = InverseQueryError::NoConstraints.to_refusal();
    assert_ne!(wrong.detail.split('.').next(), query.detail.split('.').next());
}

// -- (4)/(5) evidence obligations ----------------------------------------------------------

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn backdoor_obligation() -> EvidenceObligation {
    // The Python hand example: t=0, y=1, z1=2, z2=3 with z1, z2 -> t, y and t -> y.
    let mut graph = Dag::with_variables(7);
    for (from, to) in [(2, 0), (2, 1), (3, 0), (3, 1), (0, 1)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    let family =
        BackdoorRepairFamily::try_new(graph, v(0), v(1), "clinic", [v(0), v(1)], &[]).unwrap();
    let mut obligations = family.unresolved_obligations();
    assert_eq!(obligations.len(), 1);
    obligations.remove(0)
}

#[test]
fn a0r_backdoor_obligation_has_the_derived_identity_and_the_python_visible_fields() {
    let obligation = backdoor_obligation();
    assert_eq!(obligation.kind, EvidenceObligationKind::ProvideJointLaw);
    assert_eq!(obligation.variables.as_ref(), [v(0), v(1), v(2), v(3)]);
    assert_eq!(obligation.population.as_deref(), Some("clinic"));
    assert!(obligation.regime.joint && obligation.regime.interventions.is_empty());
    assert_eq!(obligation.provenance.family.as_ref(), "backdoor");
    assert_eq!(
        obligation.provenance.proof_step.as_deref(),
        Some("backdoor.adjustment_set:0,1,2,3")
    );
    assert!(obligation.satisfiable_by_study());
    // The id is `eo1:<kind>:` and the first 32 hex digits of BLAKE3 over the canonical
    // content, recomputed here.
    let digest = blake3::hash(obligation.canonical().as_bytes()).to_hex().to_string();
    assert_eq!(obligation.id.as_ref(), format!("eo1:provide_joint_law:{}", &digest[..32]));
}

#[test]
fn a0r_obligation_refusals_use_the_registered_codes_and_details_python_reports() {
    let wrong = EvidenceObligationError::wrong_contract("a study never proves an assumption");
    assert_eq!(wrong.code, "transport_missing_evidence");
    assert_eq!(wrong.detail, "evidence_obligations.wrong_contract");
    let invalid = EvidenceObligation::try_new(EvidenceObligationSpec {
        quantities: std::collections::BTreeMap::new(),
        kind: EvidenceObligationKind::Measure,
        scope: ObligationScope::Factor,
        variables: Arc::from([v(0)]),
        population: Some(Arc::from("clinic")),
        regime: ObligationRegime::observational(false),
        reason: Arc::from(""),
        required_slots: Arc::from([Arc::from("factor:f")]),
        min_additional_samples: None,
        provenance: ObligationProvenance {
            family: Arc::from("backdoor"),
            source: Arc::from("contract"),
            proof_step: None,
        },
    })
    .unwrap_err();
    assert_eq!(invalid.code, "invalid_argument");
    assert_eq!(invalid.detail, "evidence_obligations.invalid_obligation");
}

// -- (4) cross-surface fixture hooks (ignored until regenerated) ---------------------------

fn obligation_fixture_json(obligation: &EvidenceObligation) -> String {
    format!(
        "{{\"id\":\"{}\",\"kind\":\"{}\",\"proof_step\":\"{}\",\"population\":\"{}\"}}\n",
        obligation.id,
        obligation.kind.as_str(),
        obligation.provenance.proof_step.as_deref().unwrap_or(""),
        obligation.population.as_deref().unwrap_or("")
    )
}

#[test]
#[ignore = "writes conformance/cross_surface/rust_a0r_*; run with ANTECEDENT_WRITE_FIXTURES=1"]
fn a0r_regenerate_rust_fixtures() {
    assert_eq!(
        std::env::var("ANTECEDENT_WRITE_FIXTURES").as_deref(),
        Ok("1"),
        "set ANTECEDENT_WRITE_FIXTURES=1 to write fixtures"
    );
    let dir = fixture_dir();
    let ranking = seal_guess();
    std::fs::write(
        dir.join("rust_a0r_design_ranking.bin"),
        ranking.to_bytes("a0r-ranking").unwrap(),
    )
    .unwrap();
    std::fs::write(
        dir.join("rust_a0r_obligation.identity.json"),
        obligation_fixture_json(&backdoor_obligation()),
    )
    .unwrap();
}

#[test]
#[ignore = "needs py_a0r_* from `A0R_WRITE_FIXTURES=1 python -m pytest python/tests/test_a0r_lifecycle.py -k a0r_regenerate_python_fixtures`"]
fn a0r_consume_python_fixtures() {
    let dir = fixture_dir();
    // The Python-built design ranking (same hand constants) is consumed by recomputation.
    let bytes = std::fs::read(dir.join("py_a0r_design_ranking.bin")).unwrap_or_else(|error| {
        panic!("missing py_a0r_design_ranking.bin ({error}); regenerate the Python fixtures")
    });
    let expectation = ConsumeExpectation {
        decision_contract_identity: Some("contract-1".into()),
        source_digests: Some(digests()),
        cost_mapping: Some(Some(utility_map())),
        ..ConsumeExpectation::default()
    };
    let consumed = consume(&bytes, &expectation).unwrap();
    assert_eq!(consumed.ranking.entries[0].semantic_id, "cand-1");
    assert!((consumed.ranking.entries[0].net_value.unwrap() - 0.15).abs() < 1e-12);
    assert!((consumed.ranking.entries[1].net_value.unwrap() - 0.025).abs() < 1e-12);
    assert_eq!(consumed.calibration, "unmeasured");
    // Python's chain over the same bytes equals Rust's over the same constants.
    let loaded = DesignRankingArtifactWire::from_bytes(&bytes).unwrap();
    assert_eq!(loaded.provenance_chain().unwrap().links().len(), 8);

    // The Python obligation has the identity Rust derives for the same contract.
    let text = std::fs::read_to_string(dir.join("py_a0r_obligation.identity.json"))
        .unwrap_or_else(|error| panic!("missing py_a0r_obligation.identity.json ({error})"));
    assert_eq!(text, obligation_fixture_json(&backdoor_obligation()));
}
