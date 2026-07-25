#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::manual_map,
    clippy::match_wildcard_for_single_variants,
    clippy::doc_markdown,
    clippy::map_unwrap_or
)]
//! Rank candidate experimental designs by identification probability.
//!
//! See ADR 0016: ranking is advisory — it does not auto-rerun analyses.
//!
//! Run: `cargo run -p antecedent --example rank_designs`

use std::sync::Arc;

use antecedent::design::{
    CandidateDesign, DesignCost, DesignEvaluationContext, DesignObjective, DesignRankConfig,
    DesignRanker, EnvironmentPlan, ExperimentPlan, MeasurementPlan, SamplingPlan, rank_designs,
};
use antecedent::prelude::*;
use antecedent_core::{EnvironmentId, QueryId};
use antecedent_prob::{GraphIdentFlag, WeightedGraphSamples};

fn candidate_kind(c: &CandidateDesign) -> &'static str {
    match c {
        CandidateDesign::Measure(_) => "measure",
        CandidateDesign::Intervene(_) => "intervene",
        CandidateDesign::ObserveEnvironment(_) => "observe_environment",
        CandidateDesign::IncreaseSamplingRate(_) => "increase_sampling_rate",
    }
}

fn main() -> Result<(), CausalError> {
    let graphs = WeightedGraphSamples::new(
        vec![0.5, 0.3, 0.2],
        vec![
            GraphIdentFlag::Identified,
            GraphIdentFlag::Unidentified,
            GraphIdentFlag::Unidentified,
        ],
        vec![10, 20, 30],
    )
    .expect("weighted graph samples");

    let q = QueryId::from_raw(0);
    let candidates = vec![
        CandidateDesign::Measure(MeasurementPlan {
            variables: Arc::from([VariableId::from_raw(3)]),
            cost: DesignCost::zero(),
            tag: 1,
        }),
        CandidateDesign::ObserveEnvironment(EnvironmentPlan {
            environment: EnvironmentId::from_raw(7),
            additional_rows: 50,
            cost: DesignCost::zero(),
            tag: 2,
        }),
        CandidateDesign::IncreaseSamplingRate(SamplingPlan {
            additional_samples: 10,
            cost: DesignCost::zero(),
            tag: 3,
        }),
        CandidateDesign::Intervene(ExperimentPlan {
            targets: Arc::from([VariableId::from_raw(0)]),
            cost: DesignCost::zero(),
            tag: 4,
        }),
    ];

    let unlock_vars = [(q, Arc::from([VariableId::from_raw(3)]))];
    let unlock_envs = [(q, Arc::from([EnvironmentId::from_raw(7)]))];
    let ranker = DesignRanker::new().with_config(DesignRankConfig {
        min_batches: 2,
        max_batches: 4,
        batch_size: 4,
        rank_uncertainty_threshold: 1.0,
    });
    let ctx = ExecutionContext::for_tests(3);
    let eval = DesignEvaluationContext::<(), ()> {
        graphs: &graphs,
        effect_width: None,
        model_loglik: None,
        decisions: None,
        query_id_unlock: Some(&unlock_vars),
        env_id_unlock: Some(&unlock_envs),
        identified_under_intervention: None,
        graph_features: None,
    };

    let ranking = rank_designs(
        &ranker,
        &DesignObjective::IncreaseIdentificationProbability { query: q },
        &candidates,
        &eval,
        &ctx,
    )?;

    let best = ranking.ranked.first().map_or(0, |r| r.candidate_index);
    println!("best_index={best} mc_samples={}", ranking.budget.samples);
    for row in ranking.ranked.iter() {
        println!(
            "  candidate={} kind={} score={:.4}",
            row.candidate_index,
            candidate_kind(&row.candidate),
            row.score
        );
    }
    Ok(())
}
