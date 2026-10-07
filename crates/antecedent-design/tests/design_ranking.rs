//! F14 / X6: the in-memory `DesignRanking` value, its order-invariant identity, replay
//! checks, and the preserved 2.2 structural ordering.
//!
//! Oracles (frozen F14 record, enumerated by hand): binary state, guess it for utility 1 if
//! right, prior 1/2. A signal of accuracy `a` has `EVSI = a - 1/2`: accuracy 3/4 gives 1/4
//! and accuracy 5/8 gives 1/8. With an identical utility-unit cost of 1/10 the net values are
//! `1/4 - 1/10 = 3/20` and `1/8 - 1/10 = 1/40`, in that order. A noisy signal of accuracy
//! 0.51 has `EVSI = 0.01`.

use std::collections::BTreeMap;
use std::sync::Arc;

use antecedent_core::{
    CancellationToken, ExternalCapability, ExternalScientificObject, ExternalTrustState,
    ProviderObjectIdentity, QuantityRole, ScientificQuantity, SignalProviderContract,
};
use antecedent_design::evsi::{
    CostToUtilityMap, EvsiCandidate, EvsiError, EvsiReport, EvsiRequest, IntegrationMethod,
    StudyCostSpec, evaluate_evsi,
};
use antecedent_design::ranking::{
    DesignRanking, DesignRankingBasis, RankingError, StructuralCandidate,
};
use antecedent_design::signal::{
    ExternalLaw, ExternalSignal, ExternalSignalBody, SignalLimits, SignalProvider, SignalRequest,
};
use antecedent_design::{
    AffineUtility, CandidateDesign, DecisionPrior, DecisionProblem, DesignCost, DesignRankConfig,
    SamplingPlan, StudyCost,
};

fn near(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12
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

fn signal_request(candidate: &str) -> SignalRequest {
    SignalRequest {
        candidate_id: candidate.into(),
        prior_id: "prior-1".into(),
        state_quantity: quantity("schema:state"),
        observation_quantity: quantity("schema:signal"),
        sample_size: 1,
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

fn posterior_law(accuracy: f64) -> ExternalLaw {
    ExternalLaw::Posterior {
        states: vec![0.0, 1.0],
        statistics: vec![0.0, 1.0],
        predictive: vec![0.5, 0.5],
        posterior: vec![vec![accuracy, 1.0 - accuracy], vec![1.0 - accuracy, accuracy]],
    }
}

fn likelihood_law(accuracy: f64) -> ExternalLaw {
    ExternalLaw::Likelihood {
        states: vec![0.0, 1.0],
        statistics: vec![0.0, 1.0],
        probabilities: vec![vec![accuracy, 1.0 - accuracy], vec![1.0 - accuracy, accuracy]],
    }
}

const UPDATE: [ExternalCapability; 2] = [ExternalCapability::Sample, ExternalCapability::Update];
const FACTOR: [ExternalCapability; 3] =
    [ExternalCapability::Sample, ExternalCapability::Factor, ExternalCapability::Update];

fn candidate(id: &str, provider: Arc<dyn SignalProvider>, cost: f64, unit: &str) -> EvsiCandidate {
    EvsiCandidate {
        semantic_id: id.into(),
        design: CandidateDesign::IncreaseSamplingRate(SamplingPlan {
            additional_samples: 1,
            cost: DesignCost::zero(),
            tag: 0,
        }),
        signal_request: signal_request(id),
        provider,
        cost: StudyCostSpec { amount: cost, unit: unit.into() },
        reused_observation_ids: vec!["obs-future".into()],
    }
}

fn posterior_candidate(id: &str, accuracy: f64, cost: f64, unit: &str) -> EvsiCandidate {
    candidate(
        id,
        external(&signal_request(id), UPDATE.to_vec(), posterior_law(accuracy)),
        cost,
        unit,
    )
}

fn utility_map() -> CostToUtilityMap {
    CostToUtilityMap {
        cost_unit: "utility".into(),
        utility_unit: "utility".into(),
        utility_per_cost: 1.0,
    }
}

fn request(candidates: Vec<EvsiCandidate>, cost_map: Option<CostToUtilityMap>) -> EvsiRequest {
    EvsiRequest {
        decision_contract_identity: "contract-1".into(),
        utility_unit: "utility".into(),
        action_ids: vec!["guess0".into(), "guess1".into()],
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

fn problem() -> DecisionProblem<usize, f64> {
    // guess0 pays 1 - theta, guess1 pays theta.
    let utility = AffineUtility::new(vec![1.0, 0.0], vec![-1.0, 1.0]).unwrap();
    DecisionProblem::new(vec![0, 1], Arc::new(utility), vec![])
}

fn evaluate(req: &EvsiRequest) -> Result<EvsiReport, EvsiError> {
    evaluate_evsi(&problem(), &DecisionPrior::Draws(vec![0.0, 1.0]), req, &CancellationToken::new())
}

fn frozen_candidates() -> Vec<EvsiCandidate> {
    vec![
        posterior_candidate("cand-1", 0.75, 0.1, "utility"),
        posterior_candidate("cand-2", 0.625, 0.1, "utility"),
    ]
}

fn digests() -> Vec<String> {
    vec!["digest-b".into(), "digest-a".into()]
}

fn identities(report: &EvsiReport) -> BTreeMap<String, String> {
    report.candidates.iter().map(|c| (c.semantic_id.clone(), c.signal_receipt.identity())).collect()
}

#[test]
fn f14_net_values_are_three_twentieths_and_one_fortieth_in_any_input_order() {
    let forward = evaluate(&request(frozen_candidates(), Some(utility_map()))).unwrap();
    let mut reversed_candidates = frozen_candidates();
    reversed_candidates.reverse();
    let reversed = evaluate(&request(reversed_candidates, Some(utility_map()))).unwrap();

    let ranking = DesignRanking::from_evsi(&forward, &digests()).unwrap();
    let other = DesignRanking::from_evsi(&reversed, &digests()).unwrap();
    assert_eq!(ranking.basis, DesignRankingBasis::NetValue);
    let ids: Vec<&str> = ranking.entries.iter().map(|e| e.semantic_id.as_str()).collect();
    assert_eq!(ids, vec!["cand-1", "cand-2"]);
    assert!(near(ranking.entries[0].evsi, 0.25));
    assert!(near(ranking.entries[1].evsi, 0.125));
    assert!(near(ranking.entries[0].net_value.unwrap(), 0.15));
    assert!(near(ranking.entries[1].net_value.unwrap(), 0.025));
    // Permutation-invariant canonical order and identity.
    assert_eq!(ranking, other);
    assert_eq!(ranking.identity(), other.identity());
    assert_eq!(ranking.entries.iter().map(|e| e.rank).collect::<Vec<_>>(), vec![0, 1]);
}

#[test]
fn f14_independent_inspection_recovers_every_retained_field() {
    let report = evaluate(&request(frozen_candidates(), Some(utility_map()))).unwrap();
    let ranking = DesignRanking::from_evsi(&report, &digests()).unwrap();
    assert_eq!(ranking.decision_contract_identity.as_deref(), Some("contract-1"));
    assert_eq!(ranking.utility_unit.as_deref(), Some("utility"));
    assert_eq!(ranking.cost_mapping, Some(utility_map()));
    assert_eq!(ranking.rng_seed, Some(5));
    assert_eq!(ranking.source_digests, vec!["digest-a".to_owned(), "digest-b".to_owned()]);
    for entry in &ranking.entries {
        assert_eq!(entry.request_fingerprint, signal_request(&entry.semantic_id).fingerprint());
        assert_eq!(entry.provider_trust.as_str(), "externally_attested");
        assert_eq!(entry.update_mode.as_str(), "external_posterior");
        // The posterior is updated through the native likelihood path: exact enumeration.
        assert_eq!(entry.integration, IntegrationMethod::Exact);
        assert!(near(entry.evpi, 0.5));
        assert_eq!(entry.sample_size, 1);
        assert_eq!(entry.study_cost_unit, "utility");
    }
    assert!(!ranking.search.truncated);
    // A fresh build from an independent evaluation reproduces the identity.
    let again = evaluate(&request(frozen_candidates(), Some(utility_map()))).unwrap();
    assert_eq!(
        DesignRanking::from_evsi(&again, &digests()).unwrap().identity(),
        ranking.identity()
    );
}

#[test]
fn f14_altered_signal_update_source_cost_or_contract_changes_identity_and_refuses_replay() {
    let base = evaluate(&request(frozen_candidates(), Some(utility_map()))).unwrap();
    let ranking = DesignRanking::from_evsi(&base, &digests()).unwrap();
    let signals = identities(&base);
    assert_eq!(
        ranking.verify_replay("contract-1", &signals, &digests(), Some(&utility_map())),
        Ok(())
    );

    // Same law, same EVSI, but updated natively from a supplied likelihood instead of an
    // externally computed posterior: a different update mode, so a different identity.
    let altered_provider =
        external(&signal_request("cand-1"), FACTOR.to_vec(), likelihood_law(0.75));
    let altered = evaluate(&request(
        vec![
            candidate("cand-1", altered_provider, 0.1, "utility"),
            posterior_candidate("cand-2", 0.625, 0.1, "utility"),
        ],
        Some(utility_map()),
    ))
    .unwrap();
    assert!(near(altered.candidates[0].evsi, 0.25));
    let altered_ranking = DesignRanking::from_evsi(&altered, &digests()).unwrap();
    assert_ne!(altered_ranking.identity(), ranking.identity());
    assert_eq!(
        ranking.verify_replay(
            "contract-1",
            &identities(&altered),
            &digests(),
            Some(&utility_map())
        ),
        Err(RankingError::SignalMismatch("cand-1".into()))
    );

    let changed_digests = vec!["digest-a".to_owned(), "digest-z".to_owned()];
    assert_eq!(
        ranking.verify_replay("contract-1", &signals, &changed_digests, Some(&utility_map())),
        Err(RankingError::SourceDigestMismatch)
    );
    let other_map = CostToUtilityMap { utility_per_cost: 2.0, ..utility_map() };
    assert_eq!(
        ranking.verify_replay("contract-1", &signals, &digests(), Some(&other_map)),
        Err(RankingError::CostMappingMismatch)
    );
    assert_eq!(
        ranking.verify_replay("contract-1", &signals, &digests(), None),
        Err(RankingError::CostMappingMismatch)
    );
    assert_eq!(
        ranking.verify_replay("contract-2", &signals, &digests(), Some(&utility_map())),
        Err(RankingError::ContractMismatch)
    );
    let refusal = RankingError::CostMappingMismatch.to_refusal();
    assert_eq!(refusal.detail, "design_ranking.cost_units_mismatch");
    assert_eq!(refusal.code, "design_cost_units_mismatch");

    // A different cost mapping or source digest also changes the identity of the value.
    let rescaled = evaluate(&request(frozen_candidates(), Some(other_map))).unwrap();
    assert_ne!(
        DesignRanking::from_evsi(&rescaled, &digests()).unwrap().identity(),
        ranking.identity()
    );
    assert_ne!(
        DesignRanking::from_evsi(&base, &changed_digests).unwrap().identity(),
        ranking.identity()
    );
}

#[test]
fn f14_overlapping_source_data_and_incompatible_cost_units_refuse_before_ranking() {
    let mut overlapping = frozen_candidates();
    overlapping[0].reused_observation_ids = vec!["obs-prior".into()];
    let error = evaluate(&request(overlapping, Some(utility_map()))).unwrap_err();
    assert_eq!(error, EvsiError::SourceOverlap(vec!["obs-prior".into()]));
    assert_eq!(error.to_refusal().detail, "evsi.source_overlap");

    let mut currency = frozen_candidates();
    currency[1].cost.unit = "usd".into();
    let error = evaluate(&request(currency, Some(utility_map()))).unwrap_err();
    assert!(matches!(error, EvsiError::CostUnitsMismatch(_)));
    assert_eq!(error.to_refusal().code, "design_cost_units_mismatch");
}

#[test]
fn x6_noisy_candidate_regret_differs_from_its_structural_unlock_rank() {
    // Structurally "noisy" is verified sufficient and cheapest, so the 2.2 order puts it first.
    let structural = vec![
        StructuralCandidate {
            semantic_id: "precise".into(),
            verified_sufficient: true,
            cost: StudyCost { units: 5, sample_budget: 100 },
        },
        StructuralCandidate {
            semantic_id: "noisy".into(),
            verified_sufficient: true,
            cost: StudyCost { units: 1, sample_budget: 100 },
        },
    ];
    let structural_ranking = DesignRanking::with_fallback(None, &structural).unwrap();
    assert_eq!(structural_ranking.basis, DesignRankingBasis::StructuralSufficiencyCost);
    assert_eq!(structural_ranking.structural[0].semantic_id, "noisy");

    // Its signal is nearly uninformative: accuracy 0.51 gives EVSI 0.01 against 0.25 for the
    // accuracy-3/4 signal, so the decision-regret order is the reverse.
    let report = evaluate(&request(
        vec![
            posterior_candidate("noisy", 0.51, 0.0, "utility"),
            posterior_candidate("precise", 0.75, 0.0, "utility"),
        ],
        None,
    ))
    .unwrap();
    let source_digests = digests();
    let probabilistic =
        DesignRanking::with_fallback(Some((&report, source_digests.as_slice())), &structural)
            .unwrap();
    assert_eq!(probabilistic.basis, DesignRankingBasis::Evsi);
    let ids: Vec<&str> = probabilistic.entries.iter().map(|e| e.semantic_id.as_str()).collect();
    assert_eq!(ids, vec!["precise", "noisy"]);
    assert!(near(probabilistic.entries[0].evsi, 0.25));
    assert!(near(probabilistic.entries[1].evsi, 0.01));
    assert!(probabilistic.structural.is_empty() && structural_ranking.entries.is_empty());
}

#[test]
fn x6_structural_order_is_sufficiency_then_cost_then_budget_then_id() {
    let candidate = |id: &str, sufficient: bool, units: u64, budget: u64| StructuralCandidate {
        semantic_id: id.into(),
        verified_sufficient: sufficient,
        cost: StudyCost { units, sample_budget: budget },
    };
    let list = vec![
        candidate("tie_b", true, 3, 10),
        candidate("impossible", false, 1, 1),
        candidate("repair", true, 3, 10),
        candidate("cheap", true, 2, 99),
        candidate("tie_a", true, 3, 10),
        candidate("small_budget", true, 3, 5),
    ];
    let ranking = DesignRanking::from_structural(&list).unwrap();
    let ids: Vec<&str> = ranking.structural.iter().map(|e| e.semantic_id.as_str()).collect();
    assert_eq!(ids, vec!["cheap", "small_budget", "repair", "tie_a", "tie_b", "impossible"]);
    let mut shuffled = list.clone();
    shuffled.reverse();
    assert_eq!(DesignRanking::from_structural(&shuffled).unwrap().identity(), ranking.identity());

    assert_eq!(DesignRanking::from_structural(&[]), Err(RankingError::Empty));
    let duplicate = vec![candidate("a", true, 1, 1), candidate("a", false, 2, 2)];
    assert_eq!(
        DesignRanking::from_structural(&duplicate),
        Err(RankingError::DuplicateCandidate("a".into()))
    );
}

#[test]
fn x6_truncated_search_is_recorded_and_changes_the_identity() {
    let full = evaluate(&request(frozen_candidates(), None)).unwrap();
    let mut bounded = request(frozen_candidates(), None);
    bounded.max_candidates = 1;
    let cut = evaluate(&bounded).unwrap();
    let full_ranking = DesignRanking::from_evsi(&full, &digests()).unwrap();
    let cut_ranking = DesignRanking::from_evsi(&cut, &digests()).unwrap();
    assert!(!full_ranking.search.truncated);
    assert!(cut_ranking.search.truncated);
    assert_eq!(cut_ranking.search.unevaluated_ids, vec!["cand-2".to_owned()]);
    assert_eq!(cut_ranking.entries.len(), 1);
    assert_ne!(full_ranking.identity(), cut_ranking.identity());
}
