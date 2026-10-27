//! Candidate studies are ranked with unlock lists derived from a finished repair search over
//! the obligations a failed identification owes: a study that merely names a variable (the
//! right column in another population, or more rows) is never credited, and the verified
//! study's own design action is what the ranker scores.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    reason = "fixture indices convert exactly into the f64 columns"
)]

use std::sync::Arc;

use antecedent::design::{
    DesignEvaluationContext, DesignObjective, DesignRankConfig, DesignRanker, StudyDeclarations,
    StudyKind, generate_candidates, rank_studies_bound,
};
use antecedent::{RefuteSuite, Study};
use antecedent_core::{AverageEffectQuery, ExecutionContext, QueryId, SearchLimits, VariableId};
use antecedent_data::TabularData;
use antecedent_design::{
    BackdoorRepairFamily, DurableStudyCandidate, RepairFamily, RepairLimits, RepairObjective,
    StudyCostDeclaration, UnitRules, repair,
};
use antecedent_graph::{Admg, Dag, DenseNodeId};
use antecedent_prob::{GraphIdentFlag, WeightedGraphSamples};

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn noise(i: usize) -> f64 {
    ((i as f64 * 12.9898).sin() * 43758.5453).fract().abs() - 0.5
}

/// An identified `x -> y` study: its prepared contract is licensed and carries the
/// identification product the ranker requires.
fn prepared_contract(ctx: &ExecutionContext) -> antecedent::analysis::CausalContract {
    let n = 160;
    let x: Vec<f64> = (0..n).map(|i| f64::from(u8::from(noise(i) > 0.0))).collect();
    let y: Vec<f64> = (0..n).map(|i| 1.0 + 2.0 * x[i] + noise(i + 900)).collect();
    let z: Vec<f64> = (0..n).map(|i| noise(i + 40)).collect();
    let data =
        TabularData::from_f64_columns([("z", &z[..]), ("x", &x[..]), ("y", &y[..])]).unwrap();
    let mut graph = Admg::with_variables(3);
    graph.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    Study::tabular(data)
        .graph(graph)
        .query(AverageEffectQuery::binary_ate(v(1), v(2)))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(ctx)
        .unwrap()
        .contract()
        .unwrap()
}

fn declarations(cost: u64) -> StudyDeclarations {
    StudyDeclarations {
        label: None,
        sample_size: 200,
        recruitment: Arc::from("census"),
        timing: Arc::from("one wave"),
        unit_rules: UnitRules {
            unit: Arc::from("patient"),
            cluster: None,
            whole_cluster_sampling: false,
        },
        cost: StudyCostDeclaration {
            units: cost,
            unit_label: Arc::from("usd"),
            sample_budget: 200,
        },
        feasible: true,
        feasibility_notes: Arc::from([]),
        external_provider: None,
    }
}

fn tiny_ranker() -> DesignRanker {
    DesignRanker::new().with_config(DesignRankConfig {
        min_batches: 1,
        max_batches: 2,
        batch_size: 2,
        rank_uncertainty_threshold: 0.5,
    })
}

/// `c -> t`, `c -> y`, `t -> y` with `c` unmeasured today: measuring `c` closes the back door.
fn family() -> BackdoorRepairFamily {
    let mut dag = Dag::with_variables(3);
    for (a, b) in [(0, 1), (2, 0), (2, 1)] {
        dag.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    BackdoorRepairFamily::try_new(dag, v(0), v(1), "pop", [v(0), v(1)], &[]).unwrap()
}

fn limits() -> RepairLimits {
    RepairLimits {
        search: SearchLimits { operations: 1000, depth: 2 },
        memory_limit_bytes: 16 * 1024 * 1024,
    }
}

/// Half the posterior mass has the confounder (identified once measured), half does not.
fn posterior() -> WeightedGraphSamples {
    WeightedGraphSamples::new(
        vec![0.5, 0.5],
        vec![GraphIdentFlag::Identified, GraphIdentFlag::Unidentified],
        vec![1, 2],
    )
    .unwrap()
}

#[test]
fn only_the_study_the_repair_search_verified_is_ranked_and_its_unlocks_are_derived() {
    let ctx = ExecutionContext::for_tests(3);
    let contract = prepared_contract(&ctx);
    let family = family();
    let generation = generate_candidates(&family.unresolved_obligations());
    let good: DurableStudyCandidate = generation.templates[0].complete(declarations(5)).unwrap();
    // The same column collected in another population: a name match, not a repair.
    let mut elsewhere = good.clone();
    elsewhere.label = Arc::from("other clinic");
    elsewhere.population = Arc::from("other");
    elsewhere.expected_evidence = elsewhere
        .expected_evidence
        .iter()
        .cloned()
        .map(|mut evidence| {
            evidence.population = Arc::from("other");
            evidence
        })
        .collect();
    let more_rows = DurableStudyCandidate {
        label: Arc::from("more rows"),
        kind: StudyKind::SampleIncrease,
        expected_evidence: Arc::from([]),
        ..good.clone()
    };
    let studies = vec![elsewhere, good, more_rows];
    let report = repair(&family, &studies, RepairObjective::MinimizeCost, limits(), &ctx).unwrap();

    let graphs = posterior();
    let eval = DesignEvaluationContext::<(), ()> {
        graphs: &graphs,
        effect_width: None,
        model_loglik: None,
        decisions: None,
        query_id_unlock: None,
        env_id_unlock: None,
        identified_under_intervention: None,
        graph_features: None,
    };
    let query = QueryId::from_raw(0);
    let ranked = rank_studies_bound(
        &contract,
        &tiny_ranker(),
        &DesignObjective::IncreaseIdentificationProbability { query },
        &studies,
        &report,
        &[v(0), v(1)],
        query,
        &eval,
        &ctx,
        None,
    )
    .unwrap();

    // Positions are in the declared slice: the verified study is the second one.
    assert_eq!(ranked.verified, vec![1]);
    assert_eq!(ranked.unlock, vec![v(2)]);
    let mut unverified: Vec<usize> = ranked.unverified.iter().map(|study| study.index).collect();
    unverified.sort_unstable();
    assert_eq!(unverified, vec![0, 2]);
    assert_eq!(ranked.ranking.ranked.len(), 1);
    assert_eq!(ranked.ranking.ranked[0].candidate_index, 1);
    assert!(ranked.ranking.ranked[0].score > 0.0, "{:?}", ranked.ranking.ranked[0]);
    let credited: Vec<usize> = ranked
        .ranking
        .violations
        .iter()
        .filter(|item| item.constraint.as_ref() == "unlicensed_candidate")
        .map(|item| item.candidate_index)
        .collect();
    assert!(credited.contains(&0) && credited.contains(&2), "{credited:?}");
}

#[test]
fn without_a_verified_study_nothing_is_ranked_and_no_unlock_is_invented() {
    let ctx = ExecutionContext::for_tests(3);
    let contract = prepared_contract(&ctx);
    let family = family();
    let generation = generate_candidates(&family.unresolved_obligations());
    let good = generation.templates[0].complete(declarations(5)).unwrap();
    // Measures the right column but not jointly with treatment and outcome.
    let marginal = DurableStudyCandidate {
        label: Arc::from("separate survey"),
        joint_measurement: false,
        expected_evidence: Arc::from([]),
        ..good
    };
    let more_rows = DurableStudyCandidate {
        label: Arc::from("more rows"),
        kind: StudyKind::SampleIncrease,
        ..marginal.clone()
    };
    let studies = vec![marginal, more_rows];
    let report = repair(&family, &studies, RepairObjective::MinimizeCost, limits(), &ctx).unwrap();
    let graphs = posterior();
    let eval = DesignEvaluationContext::<(), ()> {
        graphs: &graphs,
        effect_width: None,
        model_loglik: None,
        decisions: None,
        query_id_unlock: None,
        env_id_unlock: None,
        identified_under_intervention: None,
        graph_features: None,
    };
    let query = QueryId::from_raw(0);
    let ranked = rank_studies_bound(
        &contract,
        &tiny_ranker(),
        &DesignObjective::IncreaseIdentificationProbability { query },
        &studies,
        &report,
        &[v(0), v(1)],
        query,
        &eval,
        &ctx,
        None,
    )
    .unwrap();
    assert!(ranked.verified.is_empty());
    assert!(ranked.unlock.is_empty());
    assert!(ranked.ranking.ranked.is_empty());
    assert_eq!(ranked.unverified.len(), 2);
}
