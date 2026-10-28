//! `DistinguishModels` against exact enumeration of its stated score.
//!
//! The score is the design's reliability times the mean absolute pairwise log-likelihood gap
//! over the requested model pairs, averaged over the equally likely draws of the table.
//! The reference enumerates the draws and pairs directly (the requested models are a subset of
//! the table rows, in a different order, so a row/column mix-up changes the answer), and the
//! ranker's Monte Carlo mean is held to the reported standard error. A context that cannot be
//! scored is refused rather than scored as zero.

use std::sync::Arc;

use antecedent_core::{ExecutionContext, ModelId, VariableId};
use antecedent_design::{
    CandidateDesign, DesignCost, DesignError, DesignEvaluationContext, DesignObjective,
    DesignRankConfig, DesignRanker, MeasurementPlan, ModelLoglikDraws,
};
use antecedent_prob::{GraphIdentFlag, WeightedGraphSamples};

const TABLE: [[f64; 5]; 4] = [
    [-10.0, -3.5, -7.25, -1.0, -20.0],
    [-12.5, -3.0, -9.0, -1.0, -2.0],
    [-1.0e4, -3.25, -7.0, -40.0, -21.5],
    [-9.75, -30.0, -7.125, -2.5, -20.0],
];

fn measure(variables: u32, tag: u64) -> CandidateDesign {
    CandidateDesign::Measure(MeasurementPlan {
        variables: (0..variables).map(VariableId::from_raw).collect(),
        cost: DesignCost::zero(),
        tag,
    })
}

fn graphs() -> WeightedGraphSamples {
    WeightedGraphSamples::new([1.0], [GraphIdentFlag::Unidentified], [0]).expect("graphs")
}

fn table(rows: &[[f64; 5]]) -> ModelLoglikDraws {
    ModelLoglikDraws {
        models: (0..u32::try_from(rows.len()).expect("rows")).map(ModelId::from_raw).collect(),
        loglik: rows.iter().flatten().copied().collect(),
        n_draws: 5,
    }
}

fn rank(
    ll: &ModelLoglikDraws,
    models: &[ModelId],
    candidates: &[CandidateDesign],
) -> Result<antecedent_design::ScoredDesignRanking, DesignError> {
    let graphs = graphs();
    let eval: DesignEvaluationContext<'_, (), ()> = DesignEvaluationContext {
        graphs: &graphs,
        effect_width: None,
        model_loglik: Some(ll),
        decisions: None,
        query_id_unlock: None,
        env_id_unlock: None,
        identified_under_intervention: None,
        graph_features: None,
    };
    DesignRanker::new()
        .with_config(DesignRankConfig {
            min_batches: 60,
            max_batches: 60,
            batch_size: 200,
            rank_uncertainty_threshold: 0.0,
        })
        .rank(
            &DesignObjective::DistinguishModels { models: Arc::from(models) },
            candidates,
            &eval,
            &ExecutionContext::for_tests(21),
        )
}

fn reliability_measure(k: f64) -> f64 {
    (1.0 - (-0.75 * k).exp()).clamp(0.0, 0.99)
}

#[test]
fn score_is_reliability_times_mean_pairwise_gap_over_the_table() {
    let ll = table(&TABLE);
    let requested = [ModelId::from_raw(3), ModelId::from_raw(0), ModelId::from_raw(2)];
    let pairs = [(3_usize, 0_usize), (3, 2), (0, 2)];
    let mean_gap: f64 = (0..5)
        .map(|d| pairs.iter().map(|&(i, j)| (TABLE[i][d] - TABLE[j][d]).abs()).sum::<f64>() / 3.0)
        .sum::<f64>()
        / 5.0;
    let candidates = [measure(1, 0), measure(2, 1), measure(6, 2)];
    let ranking = rank(&ll, &requested, &candidates).expect("rank");
    for ranked in ranking.ranked.iter() {
        let k = [1.0, 2.0, 6.0][ranked.candidate_index];
        let want = reliability_measure(k) * mean_gap;
        let band = 5.0 * ranked.monte_carlo.stderr + 1e-9;
        assert!(
            (ranked.score - want).abs() <= band,
            "candidate {}: {} vs {want} (band {band})",
            ranked.candidate_index,
            ranked.score
        );
    }
}

#[test]
fn contexts_that_cannot_be_scored_are_refused() {
    let ll = table(&TABLE[..3]);
    let candidates = [measure(1, 0)];
    // Requested model absent from the table.
    let missing = rank(&ll, &[ModelId::from_raw(0), ModelId::from_raw(3)], &candidates);
    assert!(matches!(missing, Err(DesignError::Config(_))), "{missing:?}");
    // The same model listed twice would contribute a zero-gap pair.
    let repeated = rank(&ll, &[ModelId::from_raw(0), ModelId::from_raw(0)], &candidates);
    assert!(matches!(repeated, Err(DesignError::Config(_))), "{repeated:?}");
    // A non-finite log-likelihood.
    let mut rows = TABLE;
    rows[1][2] = f64::NAN;
    let nan = rank(
        &table(&rows),
        &[ModelId::from_raw(0), ModelId::from_raw(1)],
        &candidates,
    );
    assert!(matches!(nan, Err(DesignError::Config(_))), "{nan:?}");
    // A valid request still ranks.
    assert!(rank(&ll, &[ModelId::from_raw(0), ModelId::from_raw(2)], &candidates).is_ok());
}
