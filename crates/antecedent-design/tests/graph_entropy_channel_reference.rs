//! Graph-entropy design score against the mutual information of the declared channel.
//!
//! `oracle_data/graph_entropy_channel_reference.py` builds the joint law of a graph and a
//! symmetric-noise observation of its feature label at 60 digits and evaluates `I(G; Y)` by
//! the joint-law definition (not the posterior-entropy decomposition the product uses). The
//! priors have unnormalised weights, repeated feature labels, near-degenerate mass, a single
//! label (no information) and eight labels; designs span reliability zero (no rows), the
//! saturating clamps at 0.99 and 0.95, and small amounts where the gain is order 1e-4.

use std::sync::Arc;

use antecedent_core::{EnvironmentId, ExecutionContext, VariableId};
use antecedent_design::{
    CandidateDesign, DesignCost, DesignEvaluationContext, DesignObjective, DesignRankConfig,
    DesignRanker, EnvironmentPlan, ExperimentPlan, MeasurementPlan, SamplingPlan, ScoreEvaluation,
};
use antecedent_prob::{GraphIdentFlag, WeightedGraphSamples};
use serde_json::Value;

fn candidate(kind: &str, amount: u64, tag: u64) -> CandidateDesign {
    let vars = |n: u64| -> Arc<[VariableId]> {
        (0..u32::try_from(n).expect("small")).map(VariableId::from_raw).collect()
    };
    match kind {
        "measure" => CandidateDesign::Measure(MeasurementPlan {
            variables: vars(amount),
            cost: DesignCost::zero(),
            tag,
        }),
        "intervene" => CandidateDesign::Intervene(ExperimentPlan {
            targets: vars(amount),
            cost: DesignCost::zero(),
            tag,
        }),
        "observe" => CandidateDesign::ObserveEnvironment(EnvironmentPlan {
            environment: EnvironmentId::from_raw(0),
            additional_rows: amount,
            cost: DesignCost::zero(),
            tag,
        }),
        _ => CandidateDesign::IncreaseSamplingRate(SamplingPlan {
            additional_samples: amount,
            cost: DesignCost::zero(),
            tag,
        }),
    }
}

fn scores(graphs: &WeightedGraphSamples, features: Option<&[u32]>, cases: &[Value]) -> Vec<f64> {
    let candidates: Vec<CandidateDesign> = cases
        .iter()
        .enumerate()
        .map(|(i, c)| {
            candidate(
                c["kind"].as_str().expect("kind"),
                c["amount"].as_u64().expect("amount"),
                u64::try_from(i).expect("index"),
            )
        })
        .collect();
    let eval: DesignEvaluationContext<'_, (), ()> = DesignEvaluationContext {
        graphs,
        effect_width: None,
        model_loglik: None,
        decisions: None,
        query_id_unlock: None,
        env_id_unlock: None,
        identified_under_intervention: None,
        graph_features: features,
    };
    let config = DesignRankConfig {
        min_batches: 4,
        max_batches: 4,
        rank_uncertainty_threshold: 0.0,
        batch_size: 1,
    };
    let ranking = DesignRanker::new()
        .with_config(config)
        .rank(
            &DesignObjective::ReduceGraphEntropy,
            &candidates,
            &eval,
            &ExecutionContext::for_tests(7),
        )
        .expect("rank");
    let mut out = vec![f64::NAN; candidates.len()];
    for ranked in ranking.ranked.iter() {
        assert_eq!(ranked.evaluation, ScoreEvaluation::Exact);
        out[ranked.candidate_index] = ranked.score;
    }
    out
}

#[test]
fn entropy_scores_equal_mutual_information_of_the_declared_channel() {
    let text = include_str!("oracle_data/graph_entropy_channel_reference.json");
    let priors: Vec<Value> = serde_json::from_str(text).expect("reference");
    for prior in &priors {
        let name = prior["name"].as_str().expect("name");
        let weights: Vec<f64> =
            prior["weights"].as_array().expect("w").iter().map(|v| v.as_f64().expect("f")).collect();
        let features: Vec<u32> = prior["features"]
            .as_array()
            .expect("f")
            .iter()
            .map(|v| u32::try_from(v.as_u64().expect("u")).expect("u32"))
            .collect();
        let n = weights.len();
        let keys: Vec<u64> = features.iter().map(|&f| u64::from(f)).collect();
        let graphs =
            WeightedGraphSamples::new(weights, vec![GraphIdentFlag::Unidentified; n], keys)
                .expect("graphs");
        let cases = prior["cases"].as_array().expect("cases");
        // Features supplied explicitly, and read from the graph keys when omitted.
        for supplied in [Some(features.as_slice()), None] {
            let got = scores(&graphs, supplied, cases);
            for (case, score) in cases.iter().zip(&got) {
                let want = case["information_gain"].as_f64().expect("gain");
                let tol = 1e-9 + 1e-9 * want.abs();
                assert!(
                    (score - want).abs() <= tol,
                    "{name} {} {}: {score} vs {want} (explicit features: {})",
                    case["kind"],
                    case["amount"],
                    supplied.is_some()
                );
            }
        }
    }
}
