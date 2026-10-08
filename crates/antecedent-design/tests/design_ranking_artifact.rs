//! F14 / F12: the durable `design_ranking_v1` artifact and its independent consumer.
//!
//! Oracles are the constants of the core EVSI tests, written out by hand and not read from the
//! code under test:
//!
//! * Gaussian-mean decision, prior `N(1/2, 1)`, `n = 4`, noise variance 12:
//!   `EVSI = 0.04165773529384315` (unit normal linear loss `G(1) / 2`).
//! * Binomial bet with a two-point prior `{1/4, 3/4}` and `n = 2`: `EVSI = 1/16`, `EVPI = 1/8`.
//! * Frozen F14 record: binary state, guess it for utility 1, prior 1/2; signals of accuracy 3/4
//!   and 5/8 have `EVSI = 1/4` and `1/8`; with an identical utility-unit cost 1/10 the net values
//!   are `3/20` and `1/40`, in that order.
//!
//! Every "resealed" case recomputes the digest after the mutation, so only the independent
//! recomputation (or a retained expectation) can refuse it.

use std::collections::BTreeMap;
use std::sync::Arc;

use antecedent_core::{
    CancellationToken, ExternalCapability, ExternalScientificObject, ExternalTrustState,
    ProviderObjectIdentity, QuantityRole, ScientificQuantity, SignalProviderContract,
};
use antecedent_design::design_ranking_artifact::{
    CandidateWire, ConsumeExpectation, ConsumedCandidate, DESIGN_RANKING_CALIBRATION,
    DesignRankingArtifactError, DesignRankingArtifactWire, LawWire, MAX_RANKING_CANDIDATES,
    ReplayKind, SealInputs, consume, consume_wire, seal,
};
use antecedent_design::evsi::{
    CostToUtilityMap, EvsiCandidate, EvsiReport, EvsiRequest, StudyCostSpec, evaluate_evsi,
};
use antecedent_design::ranking::RankingError;
use antecedent_design::signal::{
    ExternalDecisionValues, ExternalLaw, ExternalSignal, ExternalSignalBody, NativeBinomialSignal,
    NativeGaussianMeanSignal, SignalLimits, SignalProvider, SignalRequest, SignalTrustLabel,
    SignalUpdateMode,
};
use antecedent_design::{
    AffineUtility, CandidateDesign, DecisionPrior, DecisionProblem, DesignCost, DesignRankConfig,
    SamplingPlan,
};

const EVSI_GAUSSIAN: f64 = 0.041_657_735_293_843_15;

fn near(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
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

fn external(
    req: &SignalRequest,
    capabilities: Vec<ExternalCapability>,
    law: ExternalLaw,
) -> Arc<dyn SignalProvider> {
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
        capabilities,
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

const UPDATE: [ExternalCapability; 2] = [ExternalCapability::Sample, ExternalCapability::Update];

fn candidate(
    id: &str,
    n: u64,
    provider: Arc<dyn SignalProvider>,
    cost: f64,
    unit: &str,
) -> EvsiCandidate {
    EvsiCandidate {
        semantic_id: id.into(),
        design: CandidateDesign::IncreaseSamplingRate(SamplingPlan {
            additional_samples: n,
            cost: DesignCost::zero(),
            tag: 0,
        }),
        signal_request: signal_request(id, n),
        provider,
        cost: StudyCostSpec { amount: cost, unit: unit.into() },
        reused_observation_ids: vec!["obs-future".into()],
    }
}

fn request(
    actions: &[&str],
    candidates: Vec<EvsiCandidate>,
    cost_map: Option<CostToUtilityMap>,
) -> EvsiRequest {
    EvsiRequest {
        decision_contract_identity: "contract-1".into(),
        utility_unit: "utility".into(),
        action_ids: actions.iter().map(|s| (*s).to_owned()).collect(),
        candidates,
        cost_map,
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

// -- the frozen F14 decision ----------------------------------------------------------------

const GUESS: [&str; 2] = ["guess0", "guess1"];

fn guess_problem() -> DecisionProblem<usize, f64> {
    // guess0 pays 1 - theta, guess1 pays theta.
    let utility = AffineUtility::new(vec![1.0, 0.0], vec![-1.0, 1.0]).unwrap();
    DecisionProblem::new(vec![0, 1], Arc::new(utility), vec![])
}

fn guess_prior() -> DecisionPrior<f64> {
    DecisionPrior::Draws(vec![0.0, 1.0])
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
    candidate(id, 1, external(&req, UPDATE.to_vec(), accuracy_law(accuracy)), 0.1, "utility")
}

fn frozen_candidates() -> Vec<EvsiCandidate> {
    vec![guess_candidate("cand-1", 0.75), guess_candidate("cand-2", 0.625)]
}

fn seal_guess(candidates: Vec<EvsiCandidate>) -> DesignRankingArtifactWire {
    let problem = guess_problem();
    let prior = guess_prior();
    let req = request(&GUESS, candidates, Some(utility_map()));
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

fn resealed(mut wire: DesignRankingArtifactWire) -> DesignRankingArtifactWire {
    wire.reseal().unwrap();
    wire
}

fn open() -> ConsumeExpectation {
    ConsumeExpectation::default()
}

fn retained(wire: &DesignRankingArtifactWire) -> ConsumeExpectation {
    ConsumeExpectation {
        artifact_identity: Some(wire.digest.clone()),
        decision_contract_identity: Some("contract-1".into()),
        signal_identities: Some(
            wire.candidates
                .iter()
                .map(|c| (c.semantic_id.clone(), c.signal_identity.clone()))
                .collect::<BTreeMap<_, _>>(),
        ),
        source_digests: Some(digests()),
        cost_mapping: Some(Some(utility_map())),
    }
}

fn refusal_detail(error: &DesignRankingArtifactError) -> String {
    error.refusal().expect("a refusal").detail
}

fn candidate_mut<'a>(wire: &'a mut DesignRankingArtifactWire, id: &str) -> &'a mut CandidateWire {
    wire.candidates.iter_mut().find(|c| c.semantic_id == id).unwrap()
}

#[test]
fn f14_round_trip_retains_net_values_three_twentieths_and_one_fortieth() {
    let wire = seal_guess(frozen_candidates());
    assert_eq!(wire.calibration, DESIGN_RANKING_CALIBRATION);
    assert_eq!(wire.calibration, "unmeasured");
    let bytes = wire.to_bytes("ranking-1").unwrap();
    let consumed = consume(&bytes, &retained(&wire)).unwrap();
    assert_eq!(consumed.identity, wire.digest);
    assert_eq!(consumed.calibration, "unmeasured");
    assert_eq!(consumed.provenance_chain, wire.provenance_chain().unwrap());
    let entries = &consumed.ranking.entries;
    assert_eq!(entries[0].semantic_id, "cand-1");
    assert_eq!(entries[1].semantic_id, "cand-2");
    assert!(near(entries[0].net_value.unwrap(), 0.15, 1e-12));
    assert!(near(entries[1].net_value.unwrap(), 0.025, 1e-12));
    assert!(near(entries[0].evsi, 0.25, 1e-12) && near(entries[1].evsi, 0.125, 1e-12));
    assert!(near(entries[0].evpi, 0.5, 1e-12));
    assert_eq!(consumed.ranking.rng_seed, Some(5));
    assert_eq!(consumed.ranking.source_digests, vec!["digest-a".to_owned(), "digest-b".to_owned()]);
    // The retained wire carries the references the receipt promises.
    let first = &wire.candidates.iter().find(|c| c.semantic_id == "cand-1").unwrap();
    assert_eq!(first.request.prior_id, "prior-1");
    assert_eq!(first.request_fingerprint, signal_request("cand-1", 1).fingerprint());
    assert_eq!(first.update_mode, "external_posterior");
    assert_eq!(first.trust, "externally_attested");
    assert_eq!(first.attestor.as_deref(), Some("lab-qa"));
    assert_eq!(first.provider.as_ref().unwrap().request_id, first.request_fingerprint);
    assert_eq!(wire.decision.contract_identity, "contract-1");
    assert!(near(wire.cost_map.as_ref().unwrap().utility_per_cost, 1.0, 0.0));
    // Bytes are deterministic and reproduce through a second consume.
    assert_eq!(consume(&bytes, &open()).unwrap().ranking, consumed.ranking);
    // The attested posterior is arithmetic-recomputed, never natively replayed.
    for item in &consumed.candidates {
        assert_eq!(item.replay, ReplayKind::AttestedLawArithmetic);
        assert!(!item.natively_replayed());
        assert_eq!(item.trust, SignalTrustLabel::ExternallyAttested);
        assert!(item.trust_limit.contains("not verified or natively replayed"));
    }
}

#[test]
fn f14_reordered_candidates_preserve_identity_and_bytes() {
    let forward = seal_guess(frozen_candidates());
    let mut reversed_input = frozen_candidates();
    reversed_input.reverse();
    let reversed = seal_guess(reversed_input);
    assert_eq!(forward.digest, reversed.digest);
    assert_eq!(forward.to_bytes("r").unwrap(), reversed.to_bytes("r").unwrap());
    assert_eq!(forward.ranking_identity, reversed.ranking_identity);
    // Reordering the stored list leaves the digest and the verdict unchanged.
    let mut shuffled = forward.clone();
    shuffled.candidates.reverse();
    assert_eq!(shuffled.compute_digest().unwrap(), forward.digest);
    assert!(consume_wire(&shuffled, &retained(&forward)).is_ok());
}

#[test]
fn f14_unsealed_mutation_corruption_and_unknown_version_refuse() {
    let wire = seal_guess(frozen_candidates());
    let mut edited = wire.clone();
    candidate_mut(&mut edited, "cand-1").evsi = 0.9;
    let error = consume_wire(&edited, &open()).unwrap_err();
    assert!(matches!(error, DesignRankingArtifactError::DigestMismatch), "{error:?}");
    assert_eq!(refusal_detail(&error), "design_ranking.digest_mismatch");

    // A flipped byte is a container failure, not a refusal.
    let mut bytes = wire.to_bytes("r").unwrap();
    let middle = bytes.len() / 2;
    bytes[middle] ^= 0x55;
    let error = consume(&bytes, &open()).unwrap_err();
    assert!(error.refusal().is_none(), "{error:?}");

    // Truncation and an unknown major version also refuse.
    let good = wire.to_bytes("r").unwrap();
    assert!(consume(&good[..good.len() / 2], &open()).is_err());
    let mut future = wire.clone();
    future.version = 2;
    let error = consume(&future.to_bytes("r").unwrap(), &open()).unwrap_err();
    assert!(matches!(error, DesignRankingArtifactError::UnsupportedVersion(2)), "{error:?}");
    assert_eq!(refusal_detail(&error), "design_ranking.unsupported_version");

    // A retained identity that differs refuses even for an honest artifact.
    let wrong = ConsumeExpectation { artifact_identity: Some("other".into()), ..open() };
    let error = consume_wire(&wire, &wrong).unwrap_err();
    assert!(matches!(error, DesignRankingArtifactError::ExpectedIdentityMismatch), "{error:?}");
}

#[test]
fn f14_altered_signal_or_update_is_refused_even_when_resealed() {
    type CandidateMutation = Box<dyn Fn(&mut CandidateWire)>;
    let wire = seal_guess(frozen_candidates());

    // Update mode relabelled to the other mode the law could claim.
    let mut edited = wire.clone();
    candidate_mut(&mut edited, "cand-1").update_mode = "native_update".into();
    let error = consume_wire(&resealed(edited), &open()).unwrap_err();
    assert!(
        matches!(
            &error,
            DesignRankingArtifactError::SignalInconsistent { slot: "update_mode", .. }
        ),
        "{error:?}"
    );
    assert_eq!(refusal_detail(&error), "design_ranking.signal_inconsistent");

    // An external law relabelled native trust.
    let mut edited = wire.clone();
    candidate_mut(&mut edited, "cand-1").trust = "native_licensed".into();
    let error = consume_wire(&resealed(edited), &open()).unwrap_err();
    assert!(
        matches!(&error, DesignRankingArtifactError::SignalInconsistent { slot: "trust", .. }),
        "{error:?}"
    );

    // The exact request changed (RNG seed, sample size): its fingerprint no longer matches.
    let mutations: Vec<CandidateMutation> = vec![
        Box::new(|c| c.request.rng_seed = 99),
        Box::new(|c| c.sample_size = 2),
        Box::new(|c| c.request.evidence_lineage.push("snapshot:forged".into())),
    ];
    for mutate in &mutations {
        let mut edited = wire.clone();
        mutate(candidate_mut(&mut edited, "cand-1"));
        let error = consume_wire(&resealed(edited), &open()).unwrap_err();
        assert!(
            matches!(
                &error,
                DesignRankingArtifactError::SignalInconsistent { slot: "request_fingerprint", .. }
            ),
            "{error:?}"
        );
    }

    // The attested law itself altered: the recomputed EVSI no longer matches the stored one.
    let mut edited = wire.clone();
    match &mut candidate_mut(&mut edited, "cand-1").law {
        LawWire::FinitePosterior { posterior, .. } => {
            *posterior = vec![vec![0.9, 0.1], vec![0.1, 0.9]];
        }
        other => panic!("expected an attested posterior, found {other:?}"),
    }
    let error = consume_wire(&resealed(edited), &open()).unwrap_err();
    assert!(
        matches!(&error, DesignRankingArtifactError::ReplayMismatch { quantity: "evsi", .. }),
        "{error:?}"
    );
    assert_eq!(refusal_detail(&error), "design_ranking.replay_mismatch");

    // A stored number edited and resealed.
    let mut edited = wire.clone();
    candidate_mut(&mut edited, "cand-2").evsi = 0.3;
    let error = consume_wire(&resealed(edited), &open()).unwrap_err();
    assert!(matches!(&error, DesignRankingArtifactError::ReplayMismatch { .. }), "{error:?}");

    // A consumer that retained the original signal identities refuses a swapped signal
    // whose numbers happen to agree (here: the same law answered for another request).
    let swapped = seal_guess(vec![
        guess_candidate("cand-1", 0.75),
        candidate(
            "cand-2",
            1,
            external(
                &signal_request("cand-2", 1),
                vec![
                    ExternalCapability::Sample,
                    ExternalCapability::Factor,
                    ExternalCapability::Update,
                ],
                ExternalLaw::Likelihood {
                    states: vec![0.0, 1.0],
                    statistics: vec![0.0, 1.0],
                    probabilities: vec![vec![0.625, 0.375], vec![0.375, 0.625]],
                },
            ),
            0.1,
            "utility",
        ),
    ]);
    assert!(near(
        swapped.candidates.iter().find(|c| c.semantic_id == "cand-2").unwrap().evsi,
        0.125,
        1e-12
    ));
    let error = consume_wire(&swapped, &retained(&wire)).unwrap_err();
    assert!(
        matches!(
            &error,
            DesignRankingArtifactError::ExpectedIdentityMismatch
                | DesignRankingArtifactError::Ranking(RankingError::SignalMismatch(_))
        ),
        "{error:?}"
    );
    let signals_only =
        ConsumeExpectation { signal_identities: retained(&wire).signal_identities, ..open() };
    let error = consume_wire(&swapped, &signals_only).unwrap_err();
    assert!(
        matches!(&error, DesignRankingArtifactError::Ranking(RankingError::SignalMismatch(id)) if id == "cand-2"),
        "{error:?}"
    );
    assert_eq!(refusal_detail(&error), "design_ranking.signal_mismatch");
}

#[test]
fn f14_altered_cost_mapping_amount_or_unit_is_refused_even_when_resealed() {
    let wire = seal_guess(frozen_candidates());

    let mut edited = wire.clone();
    candidate_mut(&mut edited, "cand-2").cost_unit = "usd".into();
    let error = consume_wire(&resealed(edited), &open()).unwrap_err();
    assert!(matches!(error, DesignRankingArtifactError::CostUnitsMismatch(_)), "{error:?}");
    let refusal = error.refusal().unwrap();
    assert_eq!(refusal.code, "design_cost_units_mismatch");
    assert_eq!(refusal.detail, "design_ranking.cost_units_mismatch");

    let mut edited = wire.clone();
    edited.cost_map.as_mut().unwrap().utility_unit = "qaly".into();
    let error = consume_wire(&resealed(edited), &open()).unwrap_err();
    assert!(matches!(error, DesignRankingArtifactError::CostUnitsMismatch(_)), "{error:?}");

    let mut edited = wire.clone();
    candidate_mut(&mut edited, "cand-1").cost_amount = 0.2;
    let error = consume_wire(&resealed(edited), &open()).unwrap_err();
    assert!(
        matches!(&error, DesignRankingArtifactError::ReplayMismatch { quantity: "net_value", .. }),
        "{error:?}"
    );

    let mut edited = wire.clone();
    edited.cost_map.as_mut().unwrap().utility_per_cost = 2.0;
    let error = consume_wire(&resealed(edited), &open()).unwrap_err();
    assert!(matches!(&error, DesignRankingArtifactError::ReplayMismatch { .. }), "{error:?}");
    // The consumer's retained mapping must equal the stored one.
    let other = CostToUtilityMap { utility_per_cost: 2.0, ..utility_map() };
    let expectation = ConsumeExpectation { cost_mapping: Some(Some(other)), ..open() };
    let error = consume_wire(&wire, &expectation).unwrap_err();
    assert!(
        matches!(&error, DesignRankingArtifactError::Ranking(RankingError::CostMappingMismatch)),
        "{error:?}"
    );
    assert_eq!(refusal_detail(&error), "design_ranking.cost_units_mismatch");
    let none = ConsumeExpectation { cost_mapping: Some(None), ..open() };
    assert!(matches!(
        consume_wire(&wire, &none).unwrap_err(),
        DesignRankingArtifactError::Ranking(RankingError::CostMappingMismatch)
    ));
}

#[test]
fn f14_source_overlap_is_refused_before_ranking_and_when_resealed() {
    // The evaluator refuses to produce the ranking at all.
    let mut overlapping = frozen_candidates();
    overlapping[0].reused_observation_ids = vec!["obs-prior".into()];
    let problem = guess_problem();
    let req = request(&GUESS, overlapping, Some(utility_map()));
    assert!(evaluate_evsi(&problem, &guess_prior(), &req, &CancellationToken::new()).is_err());

    // An artifact whose reuse list was edited to overlap and resealed refuses at consumption.
    let wire = seal_guess(frozen_candidates());
    let mut edited = wire.clone();
    candidate_mut(&mut edited, "cand-1").reused_observation_ids.push("obs-prior".into());
    let error = consume_wire(&resealed(edited), &open()).unwrap_err();
    match &error {
        DesignRankingArtifactError::SourceOverlap { candidate, ids } => {
            assert_eq!(candidate, "cand-1");
            assert_eq!(ids, &vec!["obs-prior".to_owned()]);
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(refusal_detail(&error), "design_ranking.source_overlap");

    // A consumer's retained source digests must equal the stored ones.
    let other_sources =
        ConsumeExpectation { source_digests: Some(vec!["digest-z".into()]), ..open() };
    let error = consume_wire(&wire, &other_sources).unwrap_err();
    assert!(
        matches!(&error, DesignRankingArtifactError::Ranking(RankingError::SourceDigestMismatch)),
        "{error:?}"
    );
    // Rewriting the stored digests and resealing breaks the ranking value's own identity.
    let mut hidden = wire.clone();
    hidden.source_digests = vec!["digest-z".into()];
    let error = consume_wire(&resealed(hidden), &open()).unwrap_err();
    assert!(
        matches!(&error, DesignRankingArtifactError::RankMismatch("ranking_identity")),
        "{error:?}"
    );
}

#[test]
fn f14_stored_order_and_search_receipt_must_follow_from_the_values() {
    let wire = seal_guess(frozen_candidates());
    let mut edited = wire.clone();
    candidate_mut(&mut edited, "cand-1").rank = 1;
    candidate_mut(&mut edited, "cand-2").rank = 0;
    let error = consume_wire(&resealed(edited), &open()).unwrap_err();
    assert!(matches!(&error, DesignRankingArtifactError::RankMismatch("rank")), "{error:?}");
    assert_eq!(refusal_detail(&error), "design_ranking.rank_mismatch");

    let mut edited = wire.clone();
    edited.search.truncated = true;
    let error = consume_wire(&resealed(edited), &open()).unwrap_err();
    assert!(
        matches!(&error, DesignRankingArtifactError::RankMismatch("search_receipt")),
        "{error:?}"
    );
}

#[test]
fn f14_bounds_are_checked_before_any_recomputation() {
    let wire = seal_guess(frozen_candidates());
    let mut huge = wire.clone();
    let template = huge.candidates[0].clone();
    for i in 0..=MAX_RANKING_CANDIDATES {
        let mut extra = template.clone();
        extra.semantic_id = format!("extra-{i}");
        huge.candidates.push(extra);
    }
    let error = consume_wire(&huge, &open()).unwrap_err();
    assert!(matches!(&error, DesignRankingArtifactError::Bounds("candidates")), "{error:?}");
    assert_eq!(refusal_detail(&error), "design_ranking.bounds_exceeded");
    // Oversized bytes refuse before decoding.
    let big = vec![0_u8; 9 * 1024 * 1024];
    assert!(consume(&big, &open()).is_err());
}

// -- native exact recomputation -------------------------------------------------------------

const BET: [&str; 2] = ["abstain", "bet"];

fn bet_problem() -> DecisionProblem<usize, f64> {
    let utility = AffineUtility::new(vec![0.0, -0.5], vec![0.0, 1.0]).unwrap();
    DecisionProblem::new(vec![0, 1], Arc::new(utility), vec![])
}

fn posterior_for_bet() -> ExternalLaw {
    ExternalLaw::Posterior {
        states: vec![0.25, 0.75],
        statistics: vec![0.0, 1.0, 2.0],
        predictive: vec![0.3125, 0.375, 0.3125],
        posterior: vec![vec![0.9, 0.1], vec![0.5, 0.5], vec![0.1, 0.9]],
    }
}

fn values_for_bet() -> ExternalLaw {
    ExternalLaw::DecisionValues(ExternalDecisionValues {
        branch_probabilities: vec![0.3125, 0.375, 0.3125],
        action_ids: vec!["abstain".into(), "bet".into()],
        values: vec![vec![0.0, -0.2], vec![0.0, 0.0], vec![0.0, 0.2]],
    })
}

fn seal_bet(candidates: Vec<EvsiCandidate>) -> (DesignRankingArtifactWire, EvsiReport) {
    let problem = bet_problem();
    let prior = DecisionPrior::Draws(vec![0.25, 0.75]);
    let req = request(&BET, candidates, None);
    let report = evaluate_evsi(&problem, &prior, &req, &CancellationToken::new()).unwrap();
    let wire = seal(&SealInputs {
        problem: &problem,
        prior: &prior,
        request: &req,
        report: &report,
        source_digests: &digests(),
    })
    .unwrap();
    (wire, report)
}

#[test]
fn f12_native_binomial_is_natively_replayed_and_equivalent_external_signals_are_not() {
    let native: Arc<dyn SignalProvider> = Arc::new(NativeBinomialSignal);
    let (wire, _) = seal_bet(vec![
        candidate("a_native", 2, native, 0.0, "utility"),
        candidate(
            "b_posterior",
            2,
            external(&signal_request("b_posterior", 2), UPDATE.to_vec(), posterior_for_bet()),
            0.0,
            "utility",
        ),
        candidate(
            "c_values",
            2,
            external(&signal_request("c_values", 2), UPDATE.to_vec(), values_for_bet()),
            0.0,
            "utility",
        ),
    ]);
    // Native and equivalent external signals agree on the fixed known-truth decision.
    for c in &wire.candidates {
        assert!(near(c.evsi, 1.0 / 16.0, 1e-12), "{}: {}", c.semantic_id, c.evsi);
        assert!(near(c.evpi, 0.125, 1e-12), "{}: {}", c.semantic_id, c.evpi);
    }
    assert_eq!(wire.basis, "evsi");
    let consumed = consume(&wire.to_bytes("bets").unwrap(), &open()).unwrap();
    let by_id: BTreeMap<&str, &ConsumedCandidate> =
        consumed.candidates.iter().map(|c| (c.semantic_id.as_str(), c)).collect();
    assert_eq!(by_id["a_native"].replay, ReplayKind::NativeExactRecomputed);
    assert!(by_id["a_native"].natively_replayed());
    assert_eq!(by_id["a_native"].trust, SignalTrustLabel::NativeLicensed);
    assert_eq!(by_id["b_posterior"].replay, ReplayKind::AttestedLawArithmetic);
    assert_eq!(by_id["b_posterior"].update_mode, SignalUpdateMode::ExternalPosterior);
    assert_eq!(by_id["c_values"].replay, ReplayKind::AttestedValuesCombined);
    assert_eq!(by_id["c_values"].update_mode, SignalUpdateMode::ExternalDecisionValues);
    for id in ["b_posterior", "c_values"] {
        assert!(!by_id[id].natively_replayed(), "{id}");
        assert_eq!(by_id[id].trust, SignalTrustLabel::ExternallyAttested, "{id}");
    }
    // The native law is retained as its family, not as a table; the attested values verbatim.
    let native_row = wire.candidates.iter().find(|c| c.semantic_id == "a_native").unwrap();
    assert!(matches!(native_row.law, LawWire::Binomial));
    let values = wire.candidates.iter().find(|c| c.semantic_id == "c_values").unwrap();
    match &values.law {
        LawWire::DecisionValues { branch_probabilities, values, .. } => {
            assert_eq!(branch_probabilities, &vec![0.3125, 0.375, 0.3125]);
            assert_eq!(values[2], vec![0.0, 0.2]);
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(values.request_fingerprint, signal_request("c_values", 2).fingerprint());
}

#[test]
fn f12_attested_decision_values_stay_attested_when_altered_or_relabelled() {
    let (wire, _) = seal_bet(vec![candidate(
        "values",
        2,
        external(&signal_request("values", 2), UPDATE.to_vec(), values_for_bet()),
        0.0,
        "utility",
    )]);
    // Values that no longer average to the prior value are incoherent even when resealed.
    let mut edited = wire.clone();
    match &mut candidate_mut(&mut edited, "values").law {
        LawWire::DecisionValues { values, .. } => values[2] = vec![0.0, 0.5],
        other => panic!("{other:?}"),
    }
    let error = consume_wire(&resealed(edited), &open()).unwrap_err();
    assert!(
        matches!(&error, DesignRankingArtifactError::SignalInconsistent { slot: "law", .. }),
        "{error:?}"
    );
    // Relabelling externally computed values as a native update contradicts the law.
    let mut edited = wire.clone();
    {
        let c = candidate_mut(&mut edited, "values");
        c.update_mode = "native_update".into();
        c.trust = "native_licensed".into();
        c.provider = None;
        c.attestor = None;
    }
    let error = consume_wire(&resealed(edited), &open()).unwrap_err();
    assert!(matches!(&error, DesignRankingArtifactError::SignalInconsistent { .. }), "{error:?}");
    // Dropping the provider identity of an external law is also refused.
    let mut edited = wire.clone();
    candidate_mut(&mut edited, "values").provider = None;
    let error = consume_wire(&resealed(edited), &open()).unwrap_err();
    assert!(
        matches!(&error, DesignRankingArtifactError::SignalInconsistent { slot: "provider", .. }),
        "{error:?}"
    );
}

#[test]
fn f12_native_gaussian_matches_the_closed_form_and_alterations_are_refused() {
    let utility = AffineUtility::new(vec![0.0, 0.0], vec![0.0, 1.0]).unwrap();
    let problem = DecisionProblem::new(vec![0_usize, 1], Arc::new(utility), vec![]);
    let prior = DecisionPrior::Normal { mean: 0.5, variance: 1.0 };
    let provider: Arc<dyn SignalProvider> = Arc::new(NativeGaussianMeanSignal::new(12.0).unwrap());
    let req =
        request(&["stay", "treat"], vec![candidate("gauss", 4, provider, 0.0, "utility")], None);
    let report = evaluate_evsi(&problem, &prior, &req, &CancellationToken::new()).unwrap();
    let wire = seal(&SealInputs {
        problem: &problem,
        prior: &prior,
        request: &req,
        report: &report,
        source_digests: &digests(),
    })
    .unwrap();
    assert!(near(wire.candidates[0].evsi, EVSI_GAUSSIAN, 1e-7));
    let consumed = consume(&wire.to_bytes("gauss").unwrap(), &open()).unwrap();
    assert_eq!(consumed.candidates[0].replay, ReplayKind::NativeExactRecomputed);
    assert!(consumed.candidates[0].natively_replayed());
    assert!(near(consumed.ranking.entries[0].evsi, EVSI_GAUSSIAN, 1e-7));

    // A different noise variance is a different signal: the law digest no longer matches.
    let mut edited = wire.clone();
    candidate_mut(&mut edited, "gauss").law = LawWire::GaussianMean { noise_variance: 1.0 };
    let error = consume_wire(&resealed(edited), &open()).unwrap_err();
    assert!(
        matches!(&error, DesignRankingArtifactError::SignalInconsistent { slot: "law_digest", .. }),
        "{error:?}"
    );
    // A stored value edited and resealed is caught by the closed-form recomputation.
    let mut edited = wire.clone();
    candidate_mut(&mut edited, "gauss").evsi = 0.05;
    let error = consume_wire(&resealed(edited), &open()).unwrap_err();
    assert!(
        matches!(&error, DesignRankingArtifactError::ReplayMismatch { quantity: "evsi", .. }),
        "{error:?}"
    );
    // A different prior in the decision table changes the decision baseline.
    let mut edited = wire.clone();
    edited.decision.prior =
        antecedent_design::design_ranking_artifact::PriorWire::Normal { mean: 0.6, variance: 1.0 };
    let error = consume_wire(&resealed(edited), &open()).unwrap_err();
    assert!(matches!(&error, DesignRankingArtifactError::ReplayMismatch { .. }), "{error:?}");
}

#[test]
fn f12_monte_carlo_value_is_bound_but_never_marked_replayed() {
    let utility = AffineUtility::new(vec![0.0, -0.45], vec![0.0, 1.0]).unwrap();
    let problem = DecisionProblem::new(vec![0_usize, 1], Arc::new(utility), vec![]);
    let prior = DecisionPrior::Draws(vec![0.2, 0.5, 0.8]);
    let provider: Arc<dyn SignalProvider> = Arc::new(NativeGaussianMeanSignal::new(0.25).unwrap());
    let mut req = request(&BET, vec![candidate("mc", 4, provider, 0.0, "utility")], None);
    req.rank_config = DesignRankConfig {
        min_batches: 16,
        max_batches: 16,
        batch_size: 64,
        rank_uncertainty_threshold: 0.0,
    };
    req.mc_error_tolerance = 1.0;
    let report = evaluate_evsi(&problem, &prior, &req, &CancellationToken::new()).unwrap();
    let wire = seal(&SealInputs {
        problem: &problem,
        prior: &prior,
        request: &req,
        report: &report,
        source_digests: &digests(),
    })
    .unwrap();
    let row = &wire.candidates[0];
    assert_eq!(row.integration, "monte_carlo");
    assert_eq!(row.replicates, 16 * 64);
    assert!(row.stderr > 0.0 && row.converged);
    assert!(near(row.evpi, 1.0 / 12.0, 1e-12));
    let consumed = consume(&wire.to_bytes("mc").unwrap(), &open()).unwrap();
    assert_eq!(consumed.candidates[0].replay, ReplayKind::MonteCarloNotReplayed);
    assert!(!consumed.candidates[0].natively_replayed());
    assert_eq!(consumed.calibration, "unmeasured");
    // Its error cannot be erased: a Monte Carlo value claiming zero error is refused.
    let mut edited = wire.clone();
    candidate_mut(&mut edited, "mc").stderr = 0.0;
    candidate_mut(&mut edited, "mc").replicates = 0;
    let error = consume_wire(&resealed(edited), &open()).unwrap_err();
    assert!(matches!(&error, DesignRankingArtifactError::ReplayMismatch { .. }), "{error:?}");
}
