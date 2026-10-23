//! Experiment planning through the compiler: the lowered program carries the
//! planning request, analytic power agrees with whole-procedure simulation of
//! each design within Monte Carlo error, the Bayesian closed forms agree with
//! simulation, the simulation is the same at every thread count, the admission
//! pool refuses a store it cannot hold, unplannable designs are refused with
//! the next attempt named, and a plan's sample size feeds design ranking.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::analysis::planning::compile_planning;
use antecedent::design::{
    ArmComparison, CandidateDesign, FactorialContrast, FamilyPower, Multiplicity, NormalPrior,
    PlannedDesign, PlanningAssumptions, PlanningCriterion, PlanningGoal, PlanningRequest,
    PlanningResult, SpendingFunction, StratumSpec, SuccessCriterion,
};
use antecedent::{BayesianConfig, CausalError, InferenceMode, RandomizedEffectQuery, Study};
use antecedent_core::{ExecutionContext, Lag, TemporalEffectQuery, TemporalPolicy, VariableId};
use antecedent_data::{SamplingRegularity, TabularData, TimeIndex, TimeSeriesData};
use antecedent_graph::{TemporalDag, ensure_lagged};

fn power(target: f64) -> PlanningCriterion {
    PlanningCriterion::Power { target }
}

fn evaluate(units: u64) -> PlanningGoal {
    PlanningGoal::Evaluate { units }
}

fn run(request: &PlanningRequest) -> PlanningResult {
    let ctx = ExecutionContext::for_tests(11);
    compile_planning(request, &ctx).unwrap().execute(&ctx).unwrap().result
}

/// Analytic and simulated criterion agree within four Monte Carlo standard
/// errors and `slack` for the normal approximations of the analytic law.
fn assert_agrees(result: &PlanningResult, slack: f64) {
    let simulated = result.simulation.as_ref().expect("simulation ran");
    let gap = (simulated.value - result.analytic.value).abs();
    assert!(
        gap <= 4.0 * simulated.monte_carlo_se + slack,
        "{} {}: analytic {} simulated {} ± {}",
        result.design,
        result.criterion,
        result.analytic.value,
        simulated.value,
        simulated.monte_carlo_se
    );
}

fn simulated(
    design: PlannedDesign,
    assumptions: PlanningAssumptions,
    units: u64,
    replicates: u32,
) -> PlanningResult {
    run(&PlanningRequest::new(design, assumptions, power(0.8), evaluate(units))
        .with_simulation(replicates, 0x5EED))
}

#[test]
fn compiled_plan_carries_every_assumption_in_its_identity() {
    let ctx = ExecutionContext::for_tests(1);
    let request = PlanningRequest::new(
        PlannedDesign::Cluster { mean_cluster_size: 12.0, icc: 0.05, cluster_size_cv: 0.0 },
        PlanningAssumptions::continuous(1.0).with_effect(0.3).with_attrition(0.1),
        power(0.8),
        PlanningGoal::SampleSize { max_units: 10_000 },
    );
    let plan = compile_planning(&request, &ctx).unwrap();
    let lowered = plan.lowered();
    assert_eq!(lowered.route(), "ExperimentPlanning");
    assert_eq!(lowered.planned_design(), Some("cluster"));
    let carried = lowered.evidence().planning().expect("layer four carries the request");
    assert_eq!(carried.request_identity(), request.identity());
    assert_eq!(compile_planning(&request, &ctx).unwrap().identity(), plan.identity());
    let changed = [
        PlanningRequest { assumptions: request.assumptions.with_attrition(0.2), ..request.clone() },
        PlanningRequest {
            design: PlannedDesign::Cluster {
                mean_cluster_size: 12.0,
                icc: 0.06,
                cluster_size_cv: 0.0,
            },
            ..request.clone()
        },
        PlanningRequest { criterion: power(0.9), ..request.clone() },
        request.clone().with_simulation(100, 3),
    ];
    for other in changed {
        assert_ne!(compile_planning(&other, &ctx).unwrap().identity(), plan.identity());
    }
    let executed = plan.execute(&ctx).unwrap();
    assert_eq!(executed.program_identity, plan.identity());
    assert!(executed.result.target_met);
    assert_eq!(executed.result.calibration, "unmeasured");
}

#[test]
fn two_arm_designs_agree_with_simulation() {
    let bernoulli = simulated(
        PlannedDesign::Bernoulli,
        PlanningAssumptions::continuous(1.0).with_effect(0.3).with_covariates(0.4, 1),
        200,
        2_000,
    );
    assert_eq!(bernoulli.analysis, "randomized_ancova_hc0");
    assert_agrees(&bernoulli, 0.025);
    let binary = simulated(
        PlannedDesign::Complete,
        PlanningAssumptions::binary(0.3).with_effect(0.15).with_attrition(0.05),
        300,
        2_000,
    );
    assert_agrees(&binary, 0.025);
    let stratified = simulated(
        PlannedDesign::Stratified {
            strata: vec![
                StratumSpec { share: 0.5, within: 1.0 },
                StratumSpec { share: 0.5, within: 2.0 },
            ],
        },
        PlanningAssumptions::continuous(1.5).with_effect(0.4),
        300,
        2_000,
    );
    assert_agrees(&stratified, 0.025);
    let factorial = simulated(
        PlannedDesign::Factorial { contrast: FactorialContrast::Interaction },
        PlanningAssumptions::continuous(1.0).with_effect(0.6),
        400,
        2_000,
    );
    assert_agrees(&factorial, 0.025);
}

#[test]
fn cluster_designs_agree_with_simulation() {
    let equal = simulated(
        PlannedDesign::Cluster { mean_cluster_size: 10.0, icc: 0.05, cluster_size_cv: 0.0 },
        PlanningAssumptions::continuous(1.0).with_effect(0.3),
        40,
        2_000,
    );
    assert!((equal.analytic.design_effect.unwrap() - 1.45).abs() < 1e-12);
    assert_agrees(&equal, 0.025);
    let unequal = simulated(
        PlannedDesign::Cluster { mean_cluster_size: 30.0, icc: 0.03, cluster_size_cv: 0.4 },
        PlanningAssumptions::continuous(1.0).with_effect(0.25).with_control_mean(1.0),
        40,
        1_500,
    );
    assert_agrees(&unequal, 0.03);
}

#[test]
fn multi_arm_families_agree_with_simulation() {
    for comparison in [
        ArmComparison::ManyToOne { multiplicity: Multiplicity::Dunnett, power: FamilyPower::Any },
        ArmComparison::ManyToOne {
            multiplicity: Multiplicity::Holm,
            power: FamilyPower::PerComparison,
        },
        ArmComparison::ManyToOne {
            multiplicity: Multiplicity::Bonferroni,
            power: FamilyPower::All,
        },
        ArmComparison::Omnibus,
    ] {
        let result = simulated(
            PlannedDesign::MultiArm { treatment_arms: 3, comparison },
            PlanningAssumptions::continuous(1.0).with_effect(0.45),
            240,
            2_000,
        );
        assert_agrees(&result, 0.025);
    }
}

#[test]
fn switchback_with_serial_dependence_and_washout_agrees_with_simulation() {
    let result = simulated(
        PlannedDesign::Switchback {
            sequences: 12,
            periods_per_block: 4,
            autocorrelation: 0.5,
            carryover: 0.3,
            washout_periods: 1,
        },
        PlanningAssumptions::continuous(1.0).with_effect(0.5).with_control_mean(2.0),
        40,
        1_500,
    );
    assert_agrees(&result, 0.03);
    let simulated = result.simulation.as_ref().unwrap();
    assert!((simulated.mean_estimate - 0.5).abs() < 4.0 * simulated.estimate_sd / 30.0);
}

#[test]
fn group_sequential_power_and_expected_size_agree_with_simulation() {
    let result = simulated(
        PlannedDesign::GroupSequential {
            information_fractions: vec![1.0 / 3.0, 2.0 / 3.0, 1.0],
            spending: SpendingFunction::OBrienFlemingType,
        },
        PlanningAssumptions::continuous(1.0).with_effect(0.3),
        400,
        2_000,
    );
    assert_agrees(&result, 0.025);
    let analytic = result.analytic.sequential.as_ref().unwrap();
    let simulated = result.simulation.as_ref().unwrap().sequential.as_ref().unwrap();
    assert!(
        (simulated.expected_units - analytic.expected_units_alternative).abs()
            <= 4.0 * simulated.expected_units_se + 8.0,
        "{} vs {}",
        simulated.expected_units,
        analytic.expected_units_alternative
    );
}

#[test]
fn bayesian_closed_forms_agree_with_simulation() {
    let design_prior = NormalPrior { mean: 0.3, sd: 0.15 };
    let analysis_prior = NormalPrior { mean: 0.0, sd: 1.0 };
    let assumptions = PlanningAssumptions::continuous(1.0);
    let criteria = [
        PlanningCriterion::Assurance {
            design_prior,
            success: SuccessCriterion::Posterior {
                analysis_prior,
                threshold: 0.0,
                probability: 0.95,
            },
            target: 0.6,
        },
        PlanningCriterion::Assurance {
            design_prior,
            success: SuccessCriterion::Significance,
            target: 0.6,
        },
        PlanningCriterion::CorrectDecision {
            design_prior,
            analysis_prior,
            threshold: 0.1,
            probability: 0.8,
            target: 0.6,
        },
    ];
    for criterion in criteria {
        let result = run(&PlanningRequest::new(
            PlannedDesign::Complete,
            assumptions,
            criterion,
            evaluate(200),
        )
        .with_simulation(4_000, 21));
        assert_agrees(&result, 0.02);
    }
    let precision = run(&PlanningRequest::new(
        PlannedDesign::Complete,
        assumptions,
        PlanningCriterion::PosteriorPrecision {
            analysis_prior: NormalPrior { mean: 0.0, sd: 0.5 },
            max_posterior_sd: 0.15,
        },
        evaluate(200),
    )
    .with_simulation(1_000, 21));
    let simulated = precision.simulation.as_ref().unwrap();
    assert!((simulated.value - precision.analytic.value).abs() < 0.02 * precision.analytic.value);
}

#[test]
fn simulation_is_the_same_at_every_thread_count() {
    let request = PlanningRequest::new(
        PlannedDesign::Complete,
        PlanningAssumptions::continuous(1.0).with_effect(0.3),
        power(0.8),
        evaluate(150),
    )
    .with_simulation(600, 99);
    let at = |threads: u32| {
        let ctx = ExecutionContext::production(5, threads);
        compile_planning(&request, &ctx).unwrap().execute(&ctx).unwrap().result.simulation
    };
    let serial = at(1);
    assert!(serial.is_some());
    assert_eq!(serial, at(4));
    assert_eq!(serial, at(3));
}

#[test]
fn a_tiny_pool_refuses_the_simulation_store() {
    let tiny = ExecutionContext::for_tests(1).with_memory_limit(Some(10_000));
    let compile_refusal = compile_planning(
        &PlanningRequest::new(
            PlannedDesign::Complete,
            PlanningAssumptions::continuous(1.0).with_effect(0.3),
            power(0.8),
            evaluate(200),
        )
        .with_simulation(1_000, 1),
        &tiny,
    )
    .unwrap_err();
    assert!(matches!(compile_refusal, CausalError::Resource { .. }), "{compile_refusal}");
    let plan = compile_planning(
        &PlanningRequest::new(
            PlannedDesign::Complete,
            PlanningAssumptions::continuous(1.0).with_effect(0.3),
            power(0.8),
            PlanningGoal::SampleSize { max_units: 100_000 },
        )
        .with_simulation(100, 1),
        &tiny,
    )
    .unwrap();
    let execute_refusal = plan.execute(&tiny).unwrap_err();
    assert!(matches!(execute_refusal, CausalError::Resource { .. }), "{execute_refusal}");
}

#[test]
fn unplannable_designs_are_refused_through_compile() {
    let ctx = ExecutionContext::for_tests(1);
    let assumptions = PlanningAssumptions::continuous(1.0).with_effect(0.3);
    let refusal = |design| {
        compile_planning(
            &PlanningRequest::new(design, assumptions, power(0.8), evaluate(100)),
            &ctx,
        )
        .unwrap_err()
        .to_string()
    };
    let interference = refusal(PlannedDesign::Interference { exposure: None });
    assert!(interference.contains("required_option_missing"), "{interference}");
    assert!(interference.contains("declare_exposure_model"), "{interference}");
    let adaptive = refusal(PlannedDesign::OutcomeAdaptive { rule: None });
    assert!(adaptive.contains("adaptive_contract_incomplete"), "{adaptive}");
    assert!(adaptive.contains("prespecify_stopping_rule"), "{adaptive}");
}

fn bernoulli_study(n: usize) -> Study {
    let y: Vec<f64> = (0..n).map(|i| (0.3 * i as f64).sin() + f64::from(i % 2 == 0)).collect();
    let data = TabularData::from_f64_columns([("outcome", y.as_slice())]).unwrap();
    let query = RandomizedEffectQuery::bernoulli_itt(
        VariableId::from_raw(0),
        (0..n).map(|i| i % 2 == 0).collect::<Vec<_>>(),
        vec![0.5; n],
        (0..n).map(|i| Arc::<str>::from(format!("unit-{i}"))).collect::<Vec<_>>(),
        (0..n).map(|i| Arc::<str>::from(format!("row-{i}"))).collect::<Vec<_>>(),
        ("control", "treated"),
    );
    Study::tabular(data).query(query).build().unwrap()
}

#[test]
fn a_randomized_study_compiles_for_planning_its_next_trial() {
    let ctx = ExecutionContext::for_tests(1);
    let study = bernoulli_study(120);
    let request = PlanningRequest::new(
        PlannedDesign::Bernoulli,
        PlanningAssumptions::continuous(1.0).with_effect(0.4),
        power(0.8),
        PlanningGoal::SampleSize { max_units: 10_000 },
    );
    let plan = study.compile_for_planning(&request, &ctx).unwrap();
    let estimate = study.compile(&ctx).unwrap();
    assert_ne!(Some(plan.identity()), estimate.lowered_program_identity());
    assert_eq!(
        plan.lowered().evidence().planning().unwrap().request_identity(),
        request.identity()
    );
    assert!(plan.execute(&ctx).unwrap().result.target_met);
    let mismatched = study
        .compile_for_planning(
            &PlanningRequest {
                design: PlannedDesign::Factorial { contrast: FactorialContrast::MainEffect },
                ..request
            },
            &ctx,
        )
        .unwrap_err()
        .to_string();
    assert!(mismatched.contains("standalone_planning"), "{mismatched}");
}

fn lag1_series() -> (TimeSeriesData, TemporalDag) {
    let n = 60usize;
    let mut x = vec![0.0; n];
    let mut y = vec![0.0; n];
    for t in 1..n {
        x[t] = (t as f64 * 0.05).sin();
        y[t] = 0.5 * x[t - 1];
    }
    let data = TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())]).unwrap();
    let series = TimeSeriesData::try_new(
        data.storage().clone(),
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap();
    let mut graph = TemporalDag::empty();
    let x1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(x1, y0).unwrap();
    (series, graph)
}

/// Two plans' sample sizes are the candidates' samples of a bound decision
/// ranking, and each scores the conjugate-normal value of sample information of
/// its planned size.
#[test]
fn planned_sample_sizes_feed_bound_design_ranking() {
    use antecedent::design::{
        AffineUtility, DecisionPrior, DecisionProblem, DecisionProblemId, DecisionRegistry,
        DesignObjective, DesignRankConfig, DesignRanker, GaussianMeanSignal,
    };

    let ctx = ExecutionContext::for_tests(1);
    let plans: Vec<PlanningResult> = [0.5, 0.25]
        .into_iter()
        .map(|effect| {
            let request = PlanningRequest::new(
                PlannedDesign::Complete,
                PlanningAssumptions::continuous(1.0).with_effect(effect),
                power(0.8),
                PlanningGoal::SampleSize { max_units: 100_000 },
            );
            compile_planning(&request, &ctx).unwrap().execute(&ctx).unwrap().result
        })
        .collect();
    let candidates: Vec<CandidateDesign> = plans
        .iter()
        .enumerate()
        .map(|(tag, plan)| plan.sampling_candidate(0.0, tag as u64))
        .collect();
    let (series, graph) = lag1_series();
    let prepared = Study::series(series)
        .graph(graph)
        .temporal_query(
            TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
                .with_policy(TemporalPolicy::pulse(-1))
                .with_horizon_steps(1),
        )
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(32)))
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let utility = AffineUtility::new(vec![1.0, -0.5], vec![0.2, 1.1]).unwrap();
    let registry = DecisionRegistry {
        problems: vec![Some(DecisionProblem::new(vec![0_usize, 1], Arc::new(utility), vec![]))],
        prior: DecisionPrior::Normal { mean: 1.0, variance: 2.0 },
        signal: Arc::new(GaussianMeanSignal::new(3.0).unwrap()),
    };
    let graphs = antecedent_prob::WeightedGraphSamples::new(
        vec![1.0],
        vec![antecedent_prob::GraphIdentFlag::Identified],
        vec![1],
    )
    .unwrap();
    let eval = antecedent::design::DesignEvaluationContext {
        graphs: &graphs,
        effect_width: None,
        model_loglik: None,
        decisions: Some(&registry),
        query_id_unlock: None,
        env_id_unlock: None,
        identified_under_intervention: None,
        graph_features: None,
    };
    let ranker = DesignRanker::new().with_config(DesignRankConfig {
        min_batches: 1,
        max_batches: 2,
        batch_size: 2,
        rank_uncertainty_threshold: 0.5,
    });
    let objective =
        DesignObjective::ReduceDecisionRegret { decision: DecisionProblemId::from_raw(0) };
    let ranking =
        prepared.rank_designs_bound(&ranker, &objective, &candidates, &eval, &ctx, None).unwrap();
    assert_eq!(ranking.ranked.len(), 2);
    for ranked in ranking.ranked.iter() {
        let n = plans[ranked.candidate_index].sample_size() as f64;
        let s = (4.0 * n / (2.0 * n + 3.0_f64)).sqrt();
        let u = (1.5_f64 / 0.9 - 1.0).abs() / s;
        let g = antecedent_kernels::norm_pdf(u) - u * antecedent_kernels::norm_sf(u);
        assert!((ranked.score - 0.9 * s * g).abs() < 1e-12, "{ranked:?}");
    }
    assert!(plans[1].sample_size() > plans[0].sample_size());
}
