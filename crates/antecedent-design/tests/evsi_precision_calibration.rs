//! Final-only numerical diagnostics for original EVSI and paired-CRN ranking.
//! No sampling-coverage or native-calibration standing is issued by these tests.
//! For theta in {-1,+1}, equal weights, U(treat)=theta and U(stay)=0,
//! a Gaussian sample mean of variance v/n gives EVSI=Phi(sqrt(n/v))-0.5,
//! EVPI=0.5. This follows by integrating the two normal half-lines, independently
//! of the posterior-weight implementation. Exact binomial and conjugate oracles
//! are frozen rational/normal-integral values. Every seed runs original full scores.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(clippy::cast_precision_loss, reason = "bounded diagnostic grid dimensions")]
#[path = "../../antecedent/tests/common/calibration.rs"]
mod calibration;
use std::sync::Arc;

use antecedent_core::{CancellationToken, ExecutionContext, QuantityRole, ScientificQuantity};
use antecedent_design::design_ranking_artifact::{
    ConsumeExpectation, ReplayKind, SealInputs, consume, seal,
};
use antecedent_design::evsi::{
    CostToUtilityMap, EvsiCandidate, EvsiError, EvsiRequest, IntegrationMethod, StudyCostSpec,
    evaluate_evsi,
};
use antecedent_design::ranking::DesignRanking;
use antecedent_design::signal::{
    NativeBinomialSignal, NativeGaussianMeanSignal, SignalLimits, SignalProvider, SignalRequest,
};
use antecedent_design::{
    AffineUtility, CandidateDesign, DecisionPrior, DecisionProblem, DecisionProblemId,
    DecisionRegistry, DesignCost, DesignEvaluationContext, DesignObjective, DesignRankConfig,
    DesignRanker, GaussianMeanSignal, SamplingPlan, ScoreEvaluation,
};
use antecedent_prob::{GraphIdentFlag, WeightedGraphSamples};
use calibration::{grid_n, grid_seed, map_replicates, n_sim};

fn problem(intercept: f64) -> DecisionProblem<usize, f64> {
    DecisionProblem::new(
        vec![0, 1],
        Arc::new(AffineUtility::new(vec![0., intercept], vec![0., 1.]).unwrap()),
        vec![],
    )
}
fn quantity(name: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: format!("schema:{name}"),
        variable_name: name.into(),
        role: QuantityRole::Outcome,
        units: "dimensionless".into(),
        population_id: "declared_model".into(),
        regime_id: "observational".into(),
        horizon: 0,
        functional_id: "state".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}
fn sampling(n: u64) -> CandidateDesign {
    CandidateDesign::IncreaseSamplingRate(SamplingPlan {
        additional_samples: n,
        cost: DesignCost::zero(),
        tag: 0,
    })
}
fn candidate(id: &str, n: u64, provider: Arc<dyn SignalProvider>, cost: f64) -> EvsiCandidate {
    EvsiCandidate {
        semantic_id: id.into(),
        design: sampling(n),
        signal_request: SignalRequest {
            candidate_id: id.into(),
            prior_id: "independent_two_point_prior".into(),
            state_quantity: quantity("theta"),
            observation_quantity: quantity("sample_mean"),
            sample_size: n,
            rng_seed: 1,
            evidence_lineage: vec!["fixed_declared_prior".into()],
            conditional_independence: "iid_given_state".into(),
            limits: SignalLimits::default(),
        },
        provider,
        cost: StudyCostSpec { amount: cost, unit: "utility".into() },
        reused_observation_ids: vec![],
    }
}
fn config(draws: usize, adaptive: bool) -> DesignRankConfig {
    let batches = u32::try_from(draws / 64).unwrap();
    DesignRankConfig {
        min_batches: if adaptive { 4 } else { batches },
        max_batches: batches,
        batch_size: 64,
        rank_uncertainty_threshold: 0.,
    }
}
fn request(candidates: Vec<EvsiCandidate>, seed: u64, draws: usize) -> EvsiRequest {
    EvsiRequest {
        decision_contract_identity: "declared_binary_state_utility".into(),
        utility_unit: "utility".into(),
        action_ids: vec!["stay".into(), "treat".into()],
        candidates,
        cost_map: Some(CostToUtilityMap {
            cost_unit: "utility".into(),
            utility_unit: "utility".into(),
            utility_per_cost: 1.,
        }),
        require_net_value: true,
        prior_observation_ids: vec![],
        rank_config: config(draws, false),
        rng_seed: seed,
        mc_error_tolerance: 0.05,
        tie_tolerance: 1e-12,
        max_candidates: 8,
    }
}
fn normal_probability(x: f64) -> f64 {
    // Independent composite Simpson integral; x <= sqrt(5), 4096 even panels.
    let panels = 4096;
    let h = x / f64::from(panels);
    let density = |z: f64| (-z * z / 2.).exp() / std::f64::consts::TAU.sqrt();
    let mut sum = density(0.) + density(x);
    for i in 1..panels {
        sum += if i % 2 == 0 { 2. } else { 4. } * density(f64::from(i) * h);
    }
    0.5 + h * sum / 3.
}

#[test]
#[ignore = "calibration: final measurement only"]
fn evsi_finite_and_conjugate_exact_precision() {
    let cases = [
        (
            problem(-0.5),
            DecisionPrior::Draws(vec![0.25, 0.75]),
            candidate("binomial", 2, Arc::new(NativeBinomialSignal), 0.1),
            0.0625,
            0.125,
        ),
        (
            problem(0.),
            DecisionPrior::Normal { mean: 0.5, variance: 1. },
            candidate("conjugate", 4, Arc::new(NativeGaussianMeanSignal::new(12.).unwrap()), 0.1),
            0.041_657_735_293_843_15,
            0.197_796_557_401_306_07,
        ),
    ];
    for (model, prior, study, truth, evpi) in cases {
        for seed in [0, 17, u64::MAX] {
            let req = request(vec![study.clone()], seed, grid_n(1024));
            let report = evaluate_evsi(&model, &prior, &req, &CancellationToken::new()).unwrap();
            let row = &report.candidates[0];
            assert_eq!(row.integration.method, IntegrationMethod::Exact);
            assert_eq!(row.integration.replicates, 0);
            assert!(row.integration.stderr.abs() < f64::EPSILON);
            assert!((row.evsi - truth).abs() < 1e-7);
            assert!((row.evpi - evpi).abs() < 1e-7);
            assert!((row.net_value.unwrap() - (truth - 0.1)).abs() < 1e-7);
            let artifact = seal(&SealInputs {
                problem: &model,
                prior: &prior,
                request: &req,
                report: &report,
                source_digests: &["fixed_declared_prior".into()],
            })
            .unwrap();
            let replay = consume(
                &artifact.to_bytes("exact_precision").unwrap(),
                &ConsumeExpectation {
                    artifact_identity: Some(artifact.digest.clone()),
                    ..Default::default()
                },
            )
            .unwrap();
            assert!(replay.candidates[0].natively_replayed());
            assert_eq!(replay.calibration, "unmeasured");
        }
    }
    println!(
        "diagnostic-precision evsi_finite_and_conjugate_exact_precision absolute_error<1e-7 full_original_integration_and_artifact_replay; conditional_model_only_no_sampling_coverage"
    );
}

#[test]
#[ignore = "calibration: final measurement only"]
fn evsi_monte_carlo_score_error_and_net_value_precision() {
    let draws = grid_n(2048);
    for (n, variance, cost) in [(1, 1., 0.), (4, 1., 0.1), (5, 1., 0.6), (100, 100., 0.6)] {
        let truth = normal_probability((n as f64 / variance).sqrt()) - 0.5;
        let results = map_replicates(n_sim(), |rep| {
            let req = request(
                vec![candidate(
                    "gaussian_finite_prior",
                    n,
                    Arc::new(NativeGaussianMeanSignal::new(variance).unwrap()),
                    cost,
                )],
                grid_seed(0x51e5_0000 + rep),
                draws,
            );
            let prior = DecisionPrior::Draws(vec![-1., 1.]);
            let model = problem(0.);
            let report = evaluate_evsi(&model, &prior, &req, &CancellationToken::new()).unwrap();
            let row = &report.candidates[0];
            assert_eq!(row.integration.method, IntegrationMethod::MonteCarlo);
            assert_eq!(row.integration.replicates, draws as u64);
            assert!(row.integration.stderr.is_finite() && row.integration.stderr > 0.);
            assert!(row.integration.converged);
            assert!((row.evpi - 0.5).abs() < 1e-12);
            assert!((row.net_value.unwrap() - (row.evsi - cost)).abs() < 1e-12);
            let ranking =
                DesignRanking::from_evsi(&report, &["fixed_declared_prior".into()]).unwrap();
            assert!((ranking.entries[0].evsi - row.evsi).abs() < 1e-12);
            let artifact = seal(&SealInputs {
                problem: &model,
                prior: &prior,
                request: &req,
                report: &report,
                source_digests: &["fixed_declared_prior".into()],
            })
            .unwrap();
            let replay = consume(
                &artifact.to_bytes("mc_precision").unwrap(),
                &ConsumeExpectation {
                    artifact_identity: Some(artifact.digest.clone()),
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(replay.candidates[0].replay, ReplayKind::MonteCarloNotReplayed);
            assert_eq!(replay.calibration, "unmeasured");
            (row.evsi - truth, row.integration.stderr, row.net_value.unwrap())
        });
        let count = results.len() as f64;
        let bias = results.iter().map(|r| r.0).sum::<f64>() / count;
        let variance = results.iter().map(|r| r.1 * r.1).sum::<f64>() / count;
        let rmse = (results.iter().map(|r| r.0 * r.0).sum::<f64>() / count).sqrt();
        let contained = results.iter().filter(|r| r.0.abs() <= 1.96 * r.1).count() as f64 / count;
        assert!(bias.abs() <= 4.5 * (variance / count).sqrt(), "MC bias={bias}");
        assert!((0.6..1.4).contains(&(rmse / variance.sqrt())), "MC SE/RMSE mismatch");
        assert!(
            (contained - 0.95).abs() <= 3. * (0.95 * 0.05 / count).sqrt(),
            "MC error containment={contained}"
        );
        if cost > 0.5 {
            assert!(results.iter().all(|r| r.2 < 0.), "negative net value falsely positive");
        }
        println!(
            "diagnostic-precision evsi_monte_carlo_score_error_and_net_value_precision n={n} cost={cost} draws={draws} repetitions={count} bias={bias} rmse={rmse} reported_mc_se={} mc_95_error_containment={contained}; not_sampling_confidence_or_rank_guarantee",
            variance.sqrt()
        );
    }
}

#[test]
#[ignore = "calibration: final measurement only"]
fn paired_crn_adaptive_rank_selection_precision() {
    let draws = grid_n(4096);
    for (sizes, variance, scope) in
        [([1, 4], 1., "separated"), ([100, 101], 100., "near_tie"), ([4, 4], 1., "tie")]
    {
        let truths = sizes.map(|n| normal_probability((n as f64 / variance).sqrt()) - 0.5);
        assert!(scope == "tie" || truths[1] > truths[0]);
        let results = map_replicates(n_sim(), |rep| {
            let graphs =
                WeightedGraphSamples::new(vec![1.], vec![GraphIdentFlag::Identified], vec![0])
                    .unwrap();
            let registry = DecisionRegistry {
                problems: vec![Some(problem(0.))],
                prior: DecisionPrior::Draws(vec![-1., 1.]),
                signal: Arc::new(GaussianMeanSignal::new(variance).unwrap()),
            };
            let eval = DesignEvaluationContext {
                graphs: &graphs,
                decisions: Some(&registry),
                effect_width: None,
                model_loglik: None,
                query_id_unlock: None,
                env_id_unlock: None,
                identified_under_intervention: None,
                graph_features: None,
            };
            let candidates = sizes.map(sampling);
            let seed = grid_seed(0xc2a0_0000 + rep);
            let run = || {
                DesignRanker::new()
                    .with_config(config(draws, true))
                    .rank(
                        &DesignObjective::ReduceDecisionRegret {
                            decision: DecisionProblemId::from_raw(0),
                        },
                        &candidates,
                        &eval,
                        &ExecutionContext::for_tests(seed),
                    )
                    .unwrap()
            };
            let rank = run();
            let top = &rank.ranked[0];
            assert_eq!(top.evaluation, ScoreEvaluation::MonteCarlo);
            assert!((256..=draws as u64).contains(&top.monte_carlo.samples));
            assert_eq!(rank.budget.evaluations, 2 * rank.budget.samples);
            if scope == "tie" {
                assert!(top.rank_uncertain);
                assert!(
                    (top.score - rank.ranked[1].score).abs() < f64::EPSILON,
                    "identical candidates must share actual state/noise streams"
                );
            }
            if rep == 0 {
                let again = run();
                assert!((again.ranked[0].score - top.score).abs() < f64::EPSILON);
                assert_eq!(again.ranked[0].candidate_index, top.candidate_index);
                assert_eq!(again.budget.samples, rank.budget.samples);
            }
            (top.candidate_index == 1, top.rank_uncertain, rank.budget.samples)
        });
        let count = results.len() as f64;
        let selected = results.iter().filter(|r| r.0).count() as f64 / count;
        let wrong_confident = results.iter().filter(|r| !r.0 && !r.1).count() as f64 / count;
        if scope == "separated" {
            assert!(selected >= 0.95, "separated selection={selected}");
        }
        if scope == "near_tie" {
            assert!(
                wrong_confident <= 0.025 + 3. * (0.025 * 0.975 / count).sqrt(),
                "confident wrong selection={wrong_confident}"
            );
        }
        println!(
            "diagnostic-precision paired_crn_adaptive_rank_selection_precision scope={scope} sizes={sizes:?} true_scores={truths:?} noise={variance} max_draws={draws} repetitions={count} selected_high={selected} wrong_confident={wrong_confident} actual_draws={:?}; original_paired_sequential_rule_only",
            (results.iter().map(|r| r.2).min(), results.iter().map(|r| r.2).max())
        );
    }
    let token = CancellationToken::new();
    token.cancel();
    let req = request(
        vec![candidate("cancelled", 1, Arc::new(NativeGaussianMeanSignal::new(1.).unwrap()), 0.)],
        1,
        draws,
    );
    assert!(matches!(
        evaluate_evsi(&problem(0.), &DecisionPrior::Draws(vec![-1., 1.]), &req, &token),
        Err(EvsiError::Cancelled)
    ));
    let invalid = DecisionPrior::Draws(vec![]);
    assert!(evaluate_evsi(&problem(0.), &invalid, &req, &CancellationToken::new()).is_err());
}

#[test]
#[ignore = "calibration: final measurement only"]
fn evsi_fixed_rank_selection_and_tie_precision() {
    let draws = grid_n(2048);
    for (sizes, variance, cost, scope) in [
        ([1, 4], 1., 0., "separated"),
        ([100, 101], 100., 0., "near_tie"),
        ([4, 4], 1., 0., "tie"),
        ([1, 4], 1., 0.6, "negative_net"),
    ] {
        let truths = sizes.map(|n| normal_probability((n as f64 / variance).sqrt()) - 0.5 - cost);
        assert!(scope == "tie" || truths[1] > truths[0]);
        let results = map_replicates(n_sim(), |rep| {
            let provider: Arc<dyn SignalProvider> =
                Arc::new(NativeGaussianMeanSignal::new(variance).unwrap());
            let studies = vec![
                candidate("low", sizes[0], provider.clone(), cost),
                candidate("high", sizes[1], provider, cost),
            ];
            let seed = grid_seed(0xf12a_0000 + rep);
            let req = request(studies, seed, draws);
            let model = problem(0.);
            let prior = DecisionPrior::Draws(vec![-1., 1.]);
            let report = evaluate_evsi(&model, &prior, &req, &CancellationToken::new()).unwrap();
            assert!(report.candidates.iter().all(|r| r.integration.replicates == draws as u64));
            if scope == "tie" {
                assert!(report.candidates.iter().all(|r| r.rank_uncertain));
                assert!(
                    (report.candidates[0].evsi - report.candidates[1].evsi).abs() < f64::EPSILON
                );
            }
            if scope == "negative_net" {
                assert!(report.candidates.iter().all(|r| r.net_value.unwrap() < 0.));
            }
            if rep == 0 {
                let mut reversed = req.clone();
                reversed.candidates.reverse();
                let other =
                    evaluate_evsi(&model, &prior, &reversed, &CancellationToken::new()).unwrap();
                let a =
                    DesignRanking::from_evsi(&report, &["fixed_declared_prior".into()]).unwrap();
                let b = DesignRanking::from_evsi(&other, &["fixed_declared_prior".into()]).unwrap();
                assert_eq!(a.identity(), b.identity());
            }
            let top = &report.candidates[0];
            (top.semantic_id == "high", top.rank_uncertain)
        });
        let count = results.len() as f64;
        let selected = results.iter().filter(|r| r.0).count() as f64 / count;
        let wrong_confident = results.iter().filter(|r| !r.0 && !r.1).count() as f64 / count;
        if matches!(scope, "separated" | "negative_net") {
            assert!(selected >= 0.95, "fixed-rank separated selection={selected}");
        }
        if scope == "near_tie" {
            assert!(
                wrong_confident <= 0.025 + 3. * (0.025 * 0.975 / count).sqrt(),
                "fixed-rank confident wrong selection={wrong_confident}"
            );
        }
        println!(
            "diagnostic-precision evsi_fixed_rank_selection_and_tie_precision scope={scope} sizes={sizes:?} true_net_scores={truths:?} noise={variance} cost={cost} draws={draws} repetitions={count} selected_high={selected} wrong_confident={wrong_confident}; original_fixed_z_marginal_se_rule_no_sampling_coverage"
        );
    }
}
