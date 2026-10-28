//! The numbers a `design_ranking_v1` artifact stores, and the independent consumer's native
//! replay of them, against the exact-rational and piecewise-quadrature reference of
//! `oracle_data/decision_exact_reference.py`.
//!
//! Binomial signals under a finite prior (the consumer rebuilds the binomial table from the
//! family parameters through logarithms) at 1 to 60 trials, with action ties, utility offsets
//! up to 1e6 and three or four actions; Gaussian sample-mean signals under a normal belief
//! (the consumer re-evaluates the expected maximum of affine functions) in seven regimes,
//! including a break-even point many prior standard deviations away, identical lines, a large
//! common offset and a prior spread of 1e-2. Each sealed value is compared with the
//! reference and each artifact must replay as natively recomputed.

use std::sync::Arc;

use antecedent_contract::ScientificQuantity;
use antecedent_core::CancellationToken;
use antecedent_design::design_ranking_artifact::{
    ConsumeExpectation, ReplayKind, SealInputs, consume, seal,
};
use antecedent_design::evsi::{EvsiCandidate, EvsiRequest, StudyCostSpec, evaluate_evsi};
use antecedent_design::signal::{
    NativeBinomialSignal, NativeGaussianMeanSignal, SignalLimits, SignalProvider, SignalRequest,
};
use antecedent_design::{
    AffineUtility, CandidateDesign, DecisionPrior, DecisionProblem, DesignCost, DesignRankConfig,
    SamplingPlan,
};
use serde_json::Value;

mod common;

fn quantity(id: &str) -> ScientificQuantity {
    ScientificQuantity {
        units: "dimensionless".into(),
        functional_id: "state".into(),
        ..common::quantity(id)
    }
}

fn candidate(id: &str, n: u64, provider: Arc<dyn SignalProvider>) -> EvsiCandidate {
    EvsiCandidate {
        semantic_id: id.into(),
        design: CandidateDesign::IncreaseSamplingRate(SamplingPlan {
            additional_samples: n,
            cost: DesignCost::zero(),
            tag: 0,
        }),
        signal_request: SignalRequest {
            candidate_id: id.into(),
            prior_id: "prior-1".into(),
            state_quantity: quantity("schema:state"),
            observation_quantity: quantity("schema:signal"),
            sample_size: n,
            rng_seed: 3,
            evidence_lineage: vec!["snapshot:a".into()],
            conditional_independence: "iid_given_state".into(),
            limits: SignalLimits::default(),
        },
        provider,
        cost: StudyCostSpec { amount: 0.0, unit: "utility".into() },
        reused_observation_ids: vec!["obs-future".into()],
    }
}

fn request(actions: &[String], candidates: Vec<EvsiCandidate>) -> EvsiRequest {
    EvsiRequest {
        decision_contract_identity: "contract-1".into(),
        utility_unit: "utility".into(),
        action_ids: actions.to_vec(),
        candidates,
        cost_map: None,
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

fn floats(value: &Value) -> Vec<f64> {
    value.as_array().expect("array").iter().map(|v| v.as_f64().expect("number")).collect()
}

fn close(got: f64, want: f64, what: &str) {
    assert!(
        (got - want).abs() <= 1e-9 + 1e-9 * want.abs(),
        "{what}: sealed {got}, reference {want}"
    );
}

fn seal_and_replay(
    problem_dbg: &[f64],
    problem: &DecisionProblem<usize, f64>,
    prior: &DecisionPrior<f64>,
    actions: &[String],
    candidates: Vec<EvsiCandidate>,
) -> antecedent_design::design_ranking_artifact::DesignRankingArtifactWire {
    let req = request(actions, candidates);
    let report = evaluate_evsi(problem, prior, &req, &CancellationToken::new()).expect("evsi");
    let wire = seal(&SealInputs {
        problem,
        prior,
        request: &req,
        report: &report,
        source_digests: &["digest-a".to_owned()],
    })
    .expect("seal");
    let consumed =
        consume(&wire.to_bytes("replay").expect("bytes"), &ConsumeExpectation::default())
            .unwrap_or_else(|e| {
                panic!("an honest artifact replays: {e:?}; intercepts {:?}", problem_dbg)
            });
    for entry in &consumed.candidates {
        assert_eq!(entry.replay, ReplayKind::NativeExactRecomputed, "{}", entry.semantic_id);
    }
    wire
}

#[test]
fn sealed_and_replayed_binomial_values_match_the_exact_rational_reference() {
    let root: Value =
        serde_json::from_str(include_str!("oracle_data/decision_exact_reference.json"))
            .expect("frozen reference");
    let mut used = 0;
    for (index, case) in root["discrete"].as_array().unwrap().iter().enumerate() {
        if !case["cutoffs"].is_null() || !case["weights"].is_null() || case["expect"].is_null() {
            continue;
        }
        used += 1;
        let intercepts = floats(&case["intercepts"]);
        let utility = AffineUtility::new(intercepts.clone(), floats(&case["slopes"])).unwrap();
        let actions: Vec<usize> = (0..intercepts.len()).collect();
        let problem = DecisionProblem::new(actions.clone(), Arc::new(utility), vec![]);
        let prior = DecisionPrior::Draws(floats(&case["thetas"]));
        let names: Vec<String> = actions.iter().map(|a| format!("a{a}")).collect();
        let ns: Vec<u64> =
            case["ns"].as_array().unwrap().iter().map(|n| n.as_u64().unwrap()).collect();
        let candidates = ns
            .iter()
            .map(|n| candidate(&format!("n{n:03}"), *n, Arc::new(NativeBinomialSignal)))
            .collect();
        let wire = seal_and_replay(&intercepts, &problem, &prior, &names, candidates);
        for row in &wire.candidates {
            let n = row.sample_size.to_string();
            close(
                row.evsi,
                case["expect"]["evsi"][&n].as_f64().unwrap(),
                &format!("case {index} evsi({n})"),
            );
            close(
                row.evpi,
                case["expect"]["evpi"].as_f64().unwrap(),
                &format!("case {index} evpi"),
            );
        }
    }
    assert!(used > 30);
}

#[test]
fn sealed_and_replayed_gaussian_values_match_the_quadrature_reference() {
    let root: Value =
        serde_json::from_str(include_str!("oracle_data/decision_exact_reference.json"))
            .expect("frozen reference");
    for (index, case) in root["gaussian"].as_array().unwrap().iter().enumerate() {
        let intercepts = floats(&case["intercepts"]);
        let utility = AffineUtility::new(intercepts.clone(), floats(&case["slopes"])).unwrap();
        let actions: Vec<usize> = (0..intercepts.len()).collect();
        let problem = DecisionProblem::new(actions.clone(), Arc::new(utility), vec![]);
        let prior = DecisionPrior::Normal {
            mean: case["mean"].as_f64().unwrap(),
            variance: case["variance"].as_f64().unwrap(),
        };
        let names: Vec<String> = actions.iter().map(|a| format!("a{a}")).collect();
        let noise = case["noise"].as_f64().unwrap();
        let candidates = case["ns"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| {
                let provider: Arc<dyn SignalProvider> =
                    Arc::new(NativeGaussianMeanSignal::new(noise).unwrap());
                candidate(&format!("n{:04}", n.as_u64().unwrap()), n.as_u64().unwrap(), provider)
            })
            .collect();
        let wire = seal_and_replay(&intercepts, &problem, &prior, &names, candidates);
        for row in &wire.candidates {
            let n = row.sample_size.to_string();
            close(
                row.evsi,
                case["evsi"][&n].as_f64().unwrap(),
                &format!("gaussian {index} evsi({n})"),
            );
            close(row.evpi, case["evpi"].as_f64().unwrap(), &format!("gaussian {index} evpi"));
        }
    }
}
