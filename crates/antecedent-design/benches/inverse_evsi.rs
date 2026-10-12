//! Criterion benches for the finite inverse query, EVSI evaluation and exact-objective
//! ranking: the three paths that rebuilt per-constraint, per-candidate or per-draw work.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    missing_docs,
    reason = "benchmark crate root: the criterion macros generate undocumented public items"
)]

use std::sync::Arc;

use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::{CancellationToken, EnvironmentId, ExecutionContext};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy, Tail,
    UtilityExpr,
};
use antecedent_design::decision_structural::{AtomEvidence, StructuralAtom};
use antecedent_design::evsi::{EvsiCandidate, EvsiRequest, StudyCostSpec, evaluate_evsi};
use antecedent_design::inverse_query::{
    Comparison, ForwardClaim, ForwardEvidence, GridScope, InverseConstraint, InverseQuery,
    SelectionRule, evaluate_inverse_query,
};
use antecedent_design::signal::{
    NativeGaussianMeanSignal, SignalLimits, SignalProvider, SignalRequest,
};
use antecedent_design::{
    AffineUtility, CandidateDesign, DecisionPrior, DecisionProblem, DecisionProblemId,
    DecisionRegistry, DesignCost, DesignEvaluationContext, DesignObjective, DesignRankConfig,
    DesignRanker, EffectWidthContext, EnvironmentGramSpec, EnvironmentPlan, GaussianMeanSignal,
    SamplingPlan,
};
use antecedent_io_core::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io_core::quantity_wire::DistributionMeaningWire;
use antecedent_prob::{GraphIdentFlag, WeightedGraphSamples};
use criterion::{Criterion, criterion_group, criterion_main};

const ACTIONS: usize = 48;
const ROWS: usize = 1500;
const MEMBERS: usize = 3;

fn quantity(variable: &str, regime: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: variable.into(),
        role: QuantityRole::Outcome,
        units: "units".into(),
        population_id: "target".into(),
        regime_id: regime.into(),
        horizon: 0,
        functional_id: "outcome".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn regime(i: usize) -> String {
    format!("do(a={i})")
}

fn law(seed: usize) -> DistributionArtifact {
    let quantities: Vec<ScientificQuantity> =
        (0..ACTIONS).map(|i| quantity("y", &regime(i))).collect();
    let identity = DistributionIdentity::new(
        DistributionMeaningWire::InterventionalPredictive,
        &quantities,
        DrawAlignment::Joint,
        DistributionProvenance {
            source_id: "bench".into(),
            provider_id: "exact-law".into(),
            rng_id: "deterministic_exact".into(),
            snapshot_id: format!("bench-{seed}"),
            causal_contract_id: "checked-contract".into(),
        },
    )
    .unwrap();
    let mut draws = Vec::with_capacity(ROWS * ACTIONS);
    for row in 0..ROWS {
        for action in 0..ACTIONS {
            let phase = ((row * 31 + action * 17 + seed * 7) % 1000) as f64 / 1000.0;
            draws.push(phase + action as f64 * 0.01);
        }
    }
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".into(), "quantity".into()],
            shape: [ROWS, ACTIONS],
            weights: None,
            supported: None,
            calibration: DistributionCalibration::Exact,
            trust: DistributionTrust::Unverified,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        draws,
    )
    .unwrap()
}

fn inverse_query() -> InverseQuery {
    InverseQuery {
        contract: DecisionContract {
            actions: (0..ACTIONS)
                .map(|i| DecisionAction {
                    id: format!("a{i}"),
                    kind: ActionKind::Intervention,
                    inputs: vec![quantity("y", &regime(i))],
                    utility: UtilityExpr::Input(0),
                })
                .collect(),
            utility_units: "units".into(),
            criterion: DecisionCriterion::PosteriorExpectedUtility,
            constraints: vec![],
            target_population: "target".into(),
            horizon: 0,
            structural_policy: StructuralPolicy::ReportOnly,
        },
        grid_order: (0..ACTIONS).map(|i| format!("a{i}")).collect(),
        grid_scope: GridScope::FiniteEnumeration,
        constraints: vec![
            InverseConstraint::TargetMean { target: 0.4, comparison: Comparison::AtLeast },
            InverseConstraint::TargetQuantile {
                p: 0.5,
                target: 0.4,
                comparison: Comparison::AtLeast,
            },
            InverseConstraint::ProbabilityThreshold {
                outcome_threshold: 0.3,
                tail: Tail::Upper,
                probability: 0.5,
                comparison: Comparison::AtLeast,
            },
        ],
        selection: SelectionRule::FirstInGridOrder,
        tolerance: 1e-12,
        max_evaluations: None,
    }
}

fn bench_inverse(c: &mut Criterion) {
    let query = inverse_query();
    let evidence = ForwardEvidence {
        point: Some(ForwardClaim::Law(Box::new(law(0)))),
        scenarios: Some(
            (0..MEMBERS)
                .map(|m| StructuralAtom {
                    id: format!("s{m}"),
                    probability: Some(1.0 / MEMBERS as f64),
                    evidence: AtomEvidence::Evaluated(Box::new(law(m + 1))),
                })
                .collect(),
        ),
        ..ForwardEvidence::default()
    };
    c.bench_function("inverse_query_48_actions_3_constraints_4_members", |b| {
        b.iter(|| evaluate_inverse_query(&query, &evidence).expect("query"));
    });
}

fn signal_request(id: &str, n: u64) -> SignalRequest {
    let q = |name: &str| ScientificQuantity {
        functional_id: "state".into(),
        regime_id: "observational".into(),
        ..quantity(name, "observational")
    };
    SignalRequest {
        candidate_id: id.into(),
        prior_id: "prior-1".into(),
        state_quantity: q("schema:theta"),
        observation_quantity: q("schema:signal"),
        sample_size: n,
        rng_seed: 11,
        evidence_lineage: vec!["snapshot:a".into()],
        conditional_independence: "iid_given_state".into(),
        limits: SignalLimits::default(),
    }
}

fn sampling(n: u64) -> CandidateDesign {
    CandidateDesign::IncreaseSamplingRate(SamplingPlan {
        additional_samples: n,
        cost: DesignCost::zero(),
        tag: 0,
    })
}

fn evsi_request(n_candidates: usize) -> EvsiRequest {
    let candidates = (0..n_candidates)
        .map(|i| {
            let n = 4 + 2 * i as u64;
            let provider: Arc<dyn SignalProvider> =
                Arc::new(NativeGaussianMeanSignal::new(0.5).unwrap());
            EvsiCandidate {
                semantic_id: format!("c{i:02}"),
                design: sampling(n),
                signal_request: signal_request(&format!("c{i:02}"), n),
                provider,
                cost: StudyCostSpec { amount: 0.0, unit: "utility".into() },
                reused_observation_ids: vec![],
            }
        })
        .collect();
    EvsiRequest {
        decision_contract_identity: "contract-1".into(),
        utility_unit: "utility".into(),
        action_ids: vec!["stay".into(), "treat".into()],
        candidates,
        cost_map: None,
        require_net_value: false,
        prior_observation_ids: vec![],
        rank_config: DesignRankConfig {
            min_batches: 4,
            max_batches: 4,
            batch_size: 8,
            rank_uncertainty_threshold: 0.0,
        },
        rng_seed: 5,
        mc_error_tolerance: 1.0,
        tie_tolerance: 1e-12,
        max_candidates: 64,
    }
}

fn bench_evsi(c: &mut Criterion) {
    let utility = AffineUtility::new(vec![0.0, 0.0], vec![0.0, 1.0]).unwrap();
    let problem = DecisionProblem::new(vec![0_usize, 1], Arc::new(utility), vec![]);
    let normal = DecisionPrior::Normal { mean: 0.5, variance: 1.0 };
    let draws = DecisionPrior::Draws((0..2000).map(|i| f64::from(i) / 1000.0 - 0.7).collect());
    let request = evsi_request(8);
    let cancel = CancellationToken::new();
    c.bench_function("evsi_8_candidates_exact_normal", |b| {
        b.iter(|| evaluate_evsi(&problem, &normal, &request, &cancel).expect("evsi"));
    });
    c.bench_function("evsi_8_candidates_mc_2000_draws", |b| {
        b.iter(|| evaluate_evsi(&problem, &draws, &request, &cancel).expect("evsi"));
    });
}

fn bench_exact_ranking(c: &mut Criterion) {
    let utility = AffineUtility::new(vec![0.0, 0.0], vec![0.0, 1.0]).unwrap();
    let registry = DecisionRegistry {
        problems: vec![Some(DecisionProblem::new(vec![0_usize, 1], Arc::new(utility), vec![]))],
        prior: DecisionPrior::Normal { mean: 0.5, variance: 1.0 },
        signal: Arc::new(GaussianMeanSignal::new(0.5).unwrap()),
    };
    let graphs =
        WeightedGraphSamples::new(vec![1.0], vec![GraphIdentFlag::Identified], vec![0]).unwrap();
    let eval = DesignEvaluationContext::<usize, f64> {
        graphs: &graphs,
        effect_width: None,
        model_loglik: None,
        decisions: Some(&registry),
        query_id_unlock: None,
        env_id_unlock: None,
        identified_under_intervention: None,
        graph_features: None,
    };
    let candidates: Vec<CandidateDesign> = (1..=16).map(|i| sampling(4 * i)).collect();
    let ranker = DesignRanker::new().with_config(DesignRankConfig {
        min_batches: 4,
        max_batches: 16,
        batch_size: 8,
        rank_uncertainty_threshold: 0.05,
    });
    let ctx = ExecutionContext::for_tests(3);
    let objective =
        DesignObjective::ReduceDecisionRegret { decision: DecisionProblemId::from_raw(0) };
    let once = ranker.rank(&objective, &candidates, &eval, &ctx).expect("rank");
    println!(
        "exact ranking: evaluations={} samples={} early_stopped={}",
        once.budget.evaluations, once.budget.samples, once.early_stopped
    );
    c.bench_function("rank_exact_decision_16_candidates", |b| {
        b.iter(|| ranker.rank(&objective, &candidates, &eval, &ctx).expect("rank"));
    });
}

fn bench_effect_width(c: &mut Criterion) {
    let p = 24usize;
    let gram = |scale: f64| -> Arc<[f64]> {
        let mut g = vec![0.5 * scale; p * p];
        for i in 0..p {
            g[i * p + i] = 10.0 * scale + i as f64;
        }
        Arc::from(g)
    };
    let ew = EffectWidthContext {
        xtx: gram(50.0),
        sigma2: 1.0,
        treatment_col: 1,
        n: 500,
        measure_columns: None,
        intervention_design: None,
        environment_grams: Some(
            (0..8)
                .map(|e| EnvironmentGramSpec {
                    environment: EnvironmentId::from_raw(e),
                    xtx: gram(5.0 + f64::from(e)),
                    n: 50,
                    sigma2: None,
                })
                .collect(),
        ),
    };
    let graphs =
        WeightedGraphSamples::new(vec![1.0], vec![GraphIdentFlag::Identified], vec![0]).unwrap();
    let eval = DesignEvaluationContext::<(), ()> {
        graphs: &graphs,
        effect_width: Some(&ew),
        model_loglik: None,
        decisions: None,
        query_id_unlock: None,
        env_id_unlock: None,
        identified_under_intervention: None,
        graph_features: None,
    };
    let candidates: Vec<CandidateDesign> = (0..8)
        .map(|e| {
            CandidateDesign::ObserveEnvironment(EnvironmentPlan {
                environment: EnvironmentId::from_raw(e),
                additional_rows: 50,
                cost: DesignCost::zero(),
                tag: u64::from(e),
            })
        })
        .collect();
    let ranker = DesignRanker::new().with_config(DesignRankConfig {
        min_batches: 4,
        max_batches: 16,
        batch_size: 8,
        rank_uncertainty_threshold: 0.05,
    });
    let ctx = ExecutionContext::for_tests(3);
    let objective = DesignObjective::ReduceEffectPosteriorWidth {
        query: antecedent_core::QueryId::from_raw(0),
    };
    let once = ranker.rank(&objective, &candidates, &eval, &ctx).expect("rank");
    println!(
        "effect width ranking: evaluations={} samples={} early_stopped={} first_samples={}",
        once.budget.evaluations,
        once.budget.samples,
        once.early_stopped,
        once.ranked[0].monte_carlo.samples
    );
    c.bench_function("rank_exact_effect_width_8_envs_p24", |b| {
        b.iter(|| ranker.rank(&objective, &candidates, &eval, &ctx).expect("rank"));
    });
}

criterion_group!(benches, bench_inverse, bench_evsi, bench_exact_ranking, bench_effect_width);
criterion_main!(benches);
