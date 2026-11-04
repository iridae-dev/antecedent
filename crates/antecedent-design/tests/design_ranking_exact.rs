//! Net-value design ranking against an independently enumerated expected value of sample
//! information, on a three-state prior and three signals that are not symmetric.
//!
//! Model: the state `theta` is uniform on `{1/5, 1/2, 4/5}`; the decision guesses 0 (utility
//! `1 - theta`) or 1 (utility `theta`). Prior expected utility is `1/2` and
//! `EVPI = E[max(theta, 1 - theta)] - 1/2 = (4/5 + 1/2 + 4/5)/3 - 1/2 = 1/5`. A one-observation
//! binary signal `y` with `P(y = 1 | theta) = f(theta)` has
//! `EVSI = sum_y max_a sum_k (1/3) P(y | theta_k) u_a(theta_k) - 1/2`. By hand:
//!
//! * A: `f = theta`; `P(y = 1) = 1/2`, guess 1 is worth `31/50` after y = 1 and guess 0 is worth
//!   `31/50` after y = 0, so `EVSI = 1/2 (31/50) + 1/2 (31/50) - 1/2 = 3/25`.
//! * B: `f = 1/10, 1/2, 9/10` (a sharper step signal), `EVSI = 4/25`.
//! * C: `f = 1/4 + theta/2` (a weak signal), `EVSI = 3/50`.
//!
//! The expected EVSI values are computed in this file by enumeration over the states and
//! observations (and equal those fractions); the library values come from `evaluate_evsi`
//! (which derives each signal's likelihood from the externally supplied posterior and predictive
//! law) and `RankingRecord::from_evsi`. With costs `1/50, 9/100, 1/100` in the same utility unit
//! the net values `1/10, 7/100, 1/20` order the candidates A, B, C, whereas the EVSI order is B,
//! A, C, so the ranking is by net value and not by EVSI. The input order does not change the
//! result. Tolerance 1e-12: a few float operations on numbers of order one.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::{
    ExternalCapability, ExternalScientificObject, ExternalTrustState, ProviderObjectIdentity,
    ScientificQuantity, SignalProviderContract,
};
use antecedent_core::CancellationToken;
use antecedent_design::evsi::{
    CostToUtilityMap, EvsiCandidate, EvsiReport, EvsiRequest, StudyCostSpec, evaluate_evsi,
};
use antecedent_design::ranking::{DesignRankingBasis, RankingRecord};
use antecedent_design::signal::{
    ExternalLaw, ExternalSignal, ExternalSignalBody, SignalLimits, SignalProvider, SignalRequest,
};
use antecedent_design::{
    AffineUtility, CandidateDesign, DecisionPrior, DecisionProblem, DesignCost, DesignRankConfig,
    SamplingPlan,
};

mod common;

const STATES: [f64; 3] = [0.2, 0.5, 0.8];
const TOLERANCE: f64 = 1e-12;

fn close(actual: f64, expected: f64, what: &str) {
    assert!((actual - expected).abs() <= TOLERANCE, "{what}: {actual} vs {expected}");
}

fn quantity(id: &str) -> ScientificQuantity {
    ScientificQuantity {
        units: "dimensionless".into(),
        functional_id: "state".into(),
        ..common::quantity(id)
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

/// Utility of guessing `action` (0 or 1) in state `theta`.
fn utility(action: usize, theta: f64) -> f64 {
    if action == 0 { 1.0 - theta } else { theta }
}

/// `EVSI` by enumeration over states and the two observations, for `P(y = 1 | theta) = f(theta)`.
fn enumerated_evsi(f: impl Fn(f64) -> f64) -> f64 {
    let prior_value = (0..2)
        .map(|a| STATES.iter().map(|t| utility(a, *t)).sum::<f64>() / 3.0)
        .fold(f64::MIN, f64::max);
    let posterior_value: f64 = [1_u8, 0]
        .into_iter()
        .map(|y| {
            let likelihood = |t: f64| if y == 1 { f(t) } else { 1.0 - f(t) };
            (0..2)
                .map(|a| STATES.iter().map(|t| likelihood(*t) * utility(a, *t)).sum::<f64>() / 3.0)
                .fold(f64::MIN, f64::max)
        })
        .sum();
    posterior_value - prior_value
}

/// The externally computed posterior for `f`, by Bayes over the three equally weighted states.
fn posterior_law(f: impl Fn(f64) -> f64) -> ExternalLaw {
    let predictive_one = STATES.iter().map(|t| f(*t)).sum::<f64>() / 3.0;
    let row = |y: u8| -> Vec<f64> {
        let likelihood = |t: f64| if y == 1 { f(t) } else { 1.0 - f(t) };
        let predictive = STATES.iter().map(|t| likelihood(*t)).sum::<f64>() / 3.0;
        STATES.iter().map(|t| likelihood(*t) / 3.0 / predictive).collect()
    };
    ExternalLaw::Posterior {
        states: STATES.to_vec(),
        statistics: vec![0.0, 1.0],
        predictive: vec![1.0 - predictive_one, predictive_one],
        posterior: vec![row(0), row(1)],
    }
}

fn candidate(id: &str, f: impl Fn(f64) -> f64, cost: f64) -> EvsiCandidate {
    let request = signal_request(id);
    let object = ExternalScientificObject::Signal(SignalProviderContract {
        identity: ProviderObjectIdentity {
            provider_id: "lab".into(),
            object_id: format!("signal-{id}"),
            version_id: "v1".into(),
            snapshot_id: "snap".into(),
            request_id: request.fingerprint(),
        },
        candidate_id: id.into(),
        prior_id: request.prior_id.clone(),
        observation: request.observation_quantity.clone(),
        capabilities: vec![ExternalCapability::Sample, ExternalCapability::Update],
    });
    let trust = ExternalTrustState::attest(&object, "lab-qa").unwrap();
    let body = ExternalSignalBody {
        sample_size: 1,
        state_quantity: request.state_quantity.clone(),
        observation_quantity: request.observation_quantity.clone(),
        law: posterior_law(f),
    };
    let provider: Arc<dyn SignalProvider> =
        Arc::new(ExternalSignal::new(object, trust, body).unwrap());
    EvsiCandidate {
        semantic_id: id.into(),
        design: CandidateDesign::IncreaseSamplingRate(SamplingPlan {
            additional_samples: 1,
            cost: DesignCost::zero(),
            tag: 0,
        }),
        signal_request: request,
        provider,
        cost: StudyCostSpec { amount: cost, unit: "utility".into() },
        reused_observation_ids: vec!["obs-future".into()],
    }
}

fn step(t: f64) -> f64 {
    if t < 0.4 {
        0.1
    } else if t > 0.6 {
        0.9
    } else {
        0.5
    }
}

fn candidates() -> Vec<EvsiCandidate> {
    vec![
        candidate("A", |t| t, 0.02),
        candidate("B", step, 0.09),
        candidate("C", |t| 0.25 + t / 2.0, 0.01),
    ]
}

fn evaluate(candidates: Vec<EvsiCandidate>) -> EvsiReport {
    let request = EvsiRequest {
        decision_contract_identity: "contract-1".into(),
        utility_unit: "utility".into(),
        action_ids: vec!["guess0".into(), "guess1".into()],
        candidates,
        cost_map: Some(CostToUtilityMap {
            cost_unit: "utility".into(),
            utility_unit: "utility".into(),
            utility_per_cost: 1.0,
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
    let utility = AffineUtility::new(vec![1.0, 0.0], vec![-1.0, 1.0]).unwrap();
    let problem = DecisionProblem::new(vec![0, 1], Arc::new(utility), vec![]);
    evaluate_evsi(
        &problem,
        &DecisionPrior::Draws(STATES.to_vec()),
        &request,
        &CancellationToken::new(),
    )
    .unwrap()
}

#[test]
fn net_value_ranking_matches_the_enumerated_evsi_and_cost() {
    // The enumeration agrees with the fractions worked out by hand.
    close(enumerated_evsi(|t| t), 3.0 / 25.0, "A by enumeration");
    close(enumerated_evsi(step), 4.0 / 25.0, "B by enumeration");
    close(enumerated_evsi(|t| 0.25 + t / 2.0), 3.0 / 50.0, "C by enumeration");

    let digests = vec!["digest-a".to_owned()];
    let forward = RankingRecord::from_evsi(&evaluate(candidates()), &digests).unwrap();
    let mut reversed_input = candidates();
    reversed_input.reverse();
    let reversed = RankingRecord::from_evsi(&evaluate(reversed_input), &digests).unwrap();

    assert_eq!(forward.basis, DesignRankingBasis::NetValue);
    let ids: Vec<&str> = forward.entries.iter().map(|e| e.semantic_id.as_str()).collect();
    assert_eq!(ids, vec!["A", "B", "C"], "ordered by net value, not by EVSI");
    let expected = [
        ("A", enumerated_evsi(|t| t), 0.02),
        ("B", enumerated_evsi(step), 0.09),
        ("C", enumerated_evsi(|t| 0.25 + t / 2.0), 0.01),
    ];
    for (entry, (id, evsi, cost)) in forward.entries.iter().zip(expected) {
        assert_eq!(entry.semantic_id, id);
        close(entry.evsi, evsi, id);
        close(entry.net_value.unwrap(), evsi - cost, id);
        close(entry.evpi, 0.2, "EVPI");
    }
    // By EVSI alone B would come first: the net-value order is a real consequence of the cost.
    assert!(forward.entries[1].evsi > forward.entries[0].evsi);
    assert_eq!(forward, reversed);
    assert_eq!(forward.identity(), reversed.identity());
}
