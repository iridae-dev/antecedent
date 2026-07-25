//! Frozen exact-enumeration and Monte Carlo calibration fixtures for design methods.

use std::sync::Arc;

use antecedent_core::{EnvironmentId, ExecutionContext, VariableId};
use antecedent_design::{
    CandidateDesign, DecisionConstraint, DecisionProblem, DesignCost, DesignEvaluationContext,
    DesignObjective, DesignRankConfig, DesignRanker, EnvironmentPlan, ExperimentPlan,
    MeasurementPlan, SamplingPlan, Utility, evaluate_decision,
};
use antecedent_prob::{GraphIdentFlag, WeightedGraphSamples};
use serde_json::Value;

const EIG_FIXTURE: &str =
    include_str!("../../../conformance/design/expected_information_gain/expected.json");
const DECISION_FIXTURE: &str =
    include_str!("../../../conformance/design/decision_ranking/expected.json");
const RANKING_FIXTURE: &str =
    include_str!("../../../conformance/design/candidate_ranking/expected.json");

struct ProductUtility;

impl Utility<f64, f64> for ProductUtility {
    fn evaluate_batch(&self, actions: &[f64], outcomes: &[f64], out: &mut [f64]) {
        for (action_index, action) in actions.iter().enumerate() {
            for (outcome_index, outcome) in outcomes.iter().enumerate() {
                out[action_index * outcomes.len() + outcome_index] = action * outcome;
            }
        }
    }
}

struct FixedSatisfaction(Arc<[f64]>);

impl DecisionConstraint<f64, f64> for FixedSatisfaction {
    fn name(&self) -> &str {
        "fixture"
    }

    fn satisfaction_batch(&self, actions: &[f64], _outcomes: &[f64], out: &mut [f64]) {
        assert_eq!(actions.len(), self.0.len());
        out.copy_from_slice(&self.0);
    }
}

fn fixture_graphs() -> WeightedGraphSamples {
    WeightedGraphSamples::new(
        [0.5, 0.3, 0.2],
        [GraphIdentFlag::Unidentified, GraphIdentFlag::Unidentified, GraphIdentFlag::Unidentified],
        [10, 20, 30],
    )
    .expect("valid fixture posterior")
}

fn fixture_candidates() -> Vec<CandidateDesign> {
    vec![
        CandidateDesign::Measure(MeasurementPlan {
            variables: Arc::from([VariableId::from_raw(0)]),
            cost: DesignCost::zero(),
            tag: 1,
        }),
        CandidateDesign::Measure(MeasurementPlan {
            variables: Arc::from([VariableId::from_raw(0), VariableId::from_raw(1)]),
            cost: DesignCost::zero(),
            tag: 2,
        }),
        CandidateDesign::Intervene(ExperimentPlan {
            targets: Arc::from([VariableId::from_raw(0)]),
            cost: DesignCost::zero(),
            tag: 3,
        }),
        CandidateDesign::ObserveEnvironment(EnvironmentPlan {
            environment: EnvironmentId::from_raw(0),
            additional_rows: 50,
            cost: DesignCost::zero(),
            tag: 4,
        }),
        CandidateDesign::IncreaseSamplingRate(SamplingPlan {
            additional_samples: 1,
            cost: DesignCost::zero(),
            tag: 5,
        }),
        CandidateDesign::IncreaseSamplingRate(SamplingPlan {
            additional_samples: 100,
            cost: DesignCost::zero(),
            tag: 6,
        }),
    ]
}

fn candidate_names() -> [&'static str; 6] {
    [
        "measure_one",
        "measure_two",
        "intervene_one",
        "observe_fifty",
        "sample_one",
        "sample_one_hundred",
    ]
}

fn evaluation_context(graphs: &WeightedGraphSamples) -> DesignEvaluationContext<'_, (), ()> {
    DesignEvaluationContext {
        graphs,
        effect_width: None,
        model_loglik: None,
        decisions: None,
        query_id_unlock: None,
        env_id_unlock: None,
        identified_under_intervention: None,
        graph_features: Some(&[0, 1, 2]),
    }
}

#[test]
fn decision_enumeration_matches_exact_feasibility_contract() {
    let fixture: Value = serde_json::from_str(DECISION_FIXTURE).expect("decision fixture JSON");
    for case in fixture["cases"].as_array().expect("cases") {
        let name = case["name"].as_str().expect("case name");
        let actions: Vec<f64> = serde_json::from_value(case["actions"].clone()).expect("actions");
        let outcomes: Vec<f64> =
            serde_json::from_value(case["outcomes"].clone()).expect("outcomes");
        let satisfaction: Vec<f64> =
            serde_json::from_value(case["satisfaction"].clone()).expect("satisfaction");
        let mut problem = DecisionProblem::new(
            actions,
            Arc::new(ProductUtility),
            vec![Arc::new(FixedSatisfaction(Arc::from(satisfaction)))],
        );
        problem.chance_threshold = case["threshold"].as_f64().expect("threshold");
        let actual = evaluate_decision(&problem, &outcomes);

        assert_eq!(
            actual.chosen_action,
            case["chosen_action"].as_u64().map(|value| value as usize),
            "{name}"
        );
        assert!(
            (actual.expected_utility
                - case["expected_utility"].as_f64().expect("expected utility"))
            .abs()
                <= 1e-12,
            "{name}"
        );
        assert!(
            (actual.posterior_regret - case["regret"].as_f64().expect("regret")).abs() <= 1e-12,
            "{name}"
        );
    }
}

#[test]
fn fixed_budget_scores_match_exact_expected_information_gain() {
    let fixture: Value = serde_json::from_str(EIG_FIXTURE).expect("EIG fixture JSON");
    let samples = fixture["acceptance"]["fixed_budget_samples"].as_u64().expect("samples");
    let atol = fixture["acceptance"]["score_atol"].as_f64().expect("score_atol");
    let graphs = fixture_graphs();
    let candidates = fixture_candidates();
    let eval = evaluation_context(&graphs);
    let config = DesignRankConfig {
        min_batches: 2_000,
        max_batches: 2_000,
        rank_uncertainty_threshold: 0.0,
        batch_size: u32::try_from(samples / 2_000).expect("batch size"),
    };

    let ranking = DesignRanker::new()
        .with_config(config)
        .rank(
            &DesignObjective::ReduceGraphEntropy,
            &candidates,
            &eval,
            &ExecutionContext::for_tests(1_701),
        )
        .expect("fixed-budget rank");
    assert_eq!(ranking.budget.samples, samples);

    let cases = fixture["cases"].as_array().expect("cases");
    for ranked in ranking.ranked.iter() {
        let name = candidate_names()[ranked.candidate_index];
        let case =
            cases.iter().find(|case| case["name"].as_str() == Some(name)).expect("candidate case");
        let exact_target = case["expected_information_gain"].as_f64().expect("exact target");
        assert!((ranked.score - exact_target).abs() <= atol, "{name}: {ranked:?}");
    }
}

#[test]
fn candidate_ranking_replays_and_calibrates_across_seed_grid() {
    let fixture: Value = serde_json::from_str(RANKING_FIXTURE).expect("ranking fixture JSON");
    let acceptance = &fixture["acceptance"];
    let expected_top =
        acceptance["adaptive_top_candidate"].as_str().expect("adaptive top candidate");
    let seeds = acceptance["seeds"].as_array().expect("seeds");
    let graphs = fixture_graphs();
    let candidates = fixture_candidates();
    let eval = evaluation_context(&graphs);
    let config = DesignRankConfig {
        min_batches: acceptance["adaptive_min_batches"].as_u64().expect("min") as u32,
        max_batches: acceptance["adaptive_max_batches"].as_u64().expect("max") as u32,
        batch_size: acceptance["adaptive_batch_size"].as_u64().expect("batch") as u32,
        rank_uncertainty_threshold: acceptance["adaptive_rank_threshold"]
            .as_f64()
            .expect("threshold"),
    };

    for seed in seeds {
        let seed = seed.as_u64().expect("seed");
        let ranker = DesignRanker::new().with_config(config.clone());
        let first = ranker
            .rank(
                &DesignObjective::ReduceGraphEntropy,
                &candidates,
                &eval,
                &ExecutionContext::for_tests(seed),
            )
            .expect("first rank");
        let replay = ranker
            .rank(
                &DesignObjective::ReduceGraphEntropy,
                &candidates,
                &eval,
                &ExecutionContext::for_tests(seed),
            )
            .expect("replay rank");
        assert_eq!(first.ranked.len(), replay.ranked.len());
        assert_eq!(first.violations, replay.violations);
        assert_eq!(first.budget.samples, replay.budget.samples);
        assert_eq!(first.budget.evaluations, replay.budget.evaluations);
        assert_eq!(first.early_stopped, replay.early_stopped);
        for (left, right) in first.ranked.iter().zip(replay.ranked.iter()) {
            assert_eq!(left.candidate_index, right.candidate_index);
            assert_eq!(left.score.to_bits(), right.score.to_bits());
            assert_eq!(left.monte_carlo.stderr.to_bits(), right.monte_carlo.stderr.to_bits());
            assert_eq!(left.monte_carlo.samples, right.monte_carlo.samples);
            assert_eq!(left.rank, right.rank);
            assert_eq!(left.rank_uncertain, right.rank_uncertain);
        }
        assert_eq!(candidate_names()[first.ranked[0].candidate_index], expected_top, "seed {seed}");
    }
}
