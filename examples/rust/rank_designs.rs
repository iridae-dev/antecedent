#![allow(
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::manual_map,
    clippy::match_wildcard_for_single_variants,
    clippy::doc_markdown,
    clippy::map_unwrap_or
)]
#![allow(
    clippy::cast_possible_truncation,
    reason = "test scaffolding compares exact constants and indexes with small literals"
)]
//! Rank candidate experimental designs by identification probability.
//!
//! Ranking is advisory — it does not auto-rerun analyses. Ranking runs through a
//! prepared study's contract, which refuses an unlicensed study and reports the candidates the
//! objective cannot score instead of scoring them.
//!
//! Run: `cargo run -p antecedent --example rank_designs`

use std::sync::Arc;

use antecedent::design::{
    CandidateDesign, DesignCost, DesignEvaluationContext, DesignObjective, DesignRankConfig,
    DesignRanker, EnvironmentPlan, ExperimentPlan, MeasurementPlan, SamplingPlan,
};
use antecedent::prelude::*;
use antecedent_core::{EnvironmentId, QueryId};
use antecedent_data::TableView;
use antecedent_prob::{GraphIdentFlag, WeightedGraphSamples};

fn candidate_kind(c: &CandidateDesign) -> &'static str {
    match c {
        CandidateDesign::Measure(_) => "measure",
        CandidateDesign::Intervene(_) => "intervene",
        CandidateDesign::ObserveEnvironment(_) => "observe_environment",
        CandidateDesign::IncreaseSamplingRate(_) => "increase_sampling_rate",
    }
}

/// Runs the example end to end; `main` calls it and the example test suite runs it.
pub fn run() -> Result<(), CausalError> {
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

    // A small confounded study (z -> t, z -> y, t -> y) supplies the prepared contract.
    let n = 96_usize;
    let z: Vec<f64> = (0..n).map(|i| ((i * 37) % n) as f64 / n as f64 - 0.5).collect();
    let t: Vec<f64> = (0..n)
        .map(|i| if (i as f64 * 0.618_033_988).fract() + 0.5 * z[i] > 0.5 { 1.0 } else { 0.0 })
        .collect();
    let y: Vec<f64> = (0..n).map(|i| 2.0 * t[i] + z[i] + 0.1 * (i as f64 * 1.7).sin()).collect();
    let data = TabularData::from_f64_columns([("t", &t[..]), ("y", &y[..]), ("z", &z[..])])?;
    let dag = Dag::from_named_edges(data.schema(), &[("z", "t"), ("z", "y"), ("t", "y")])?;
    let (t_id, y_id) = (data.schema().id_of("t")?, data.schema().id_of("y")?);
    let query = AverageEffectQuery::binary_ate(t_id, y_id);
    let prepared = Study::tabular(data)
        .graph(dag)
        .query(query)
        .bootstrap_replicates(0)
        .build()?
        .prepare(&ctx)?;

    let ranking = prepared.rank_designs(
        &ranker,
        &DesignObjective::IncreaseIdentificationProbability { query: q },
        &candidates,
        &eval,
        &ctx,
    )?;

    let best = ranking.ranked.first().map_or(0, |r| r.candidate_index);
    println!("best_index={best} mc_samples={}", ranking.budget.samples);
    // More observational units cannot identify a structurally unidentified effect, so that
    // candidate is reported as a violation instead of being scored.
    assert_eq!(
        ranking.ranked.len() + ranking.violations.len(),
        candidates.len(),
        "every candidate is ranked or reported"
    );
    assert!(!ranking.violations.is_empty(), "the sampling-rate candidate is reported");
    assert!(best < candidates.len());
    assert!(ranking.budget.samples > 0);
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

fn main() -> Result<(), CausalError> {
    run()
}
