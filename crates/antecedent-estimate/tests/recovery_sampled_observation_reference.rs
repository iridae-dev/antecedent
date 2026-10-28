//! Sampled observation recovery against an exactly enumerated finite structural model and an
//! independent BCa interval.
//!
//! `oracle_data/recovery_sampled_observation_reference.py` (frozen in the JSON beside it)
//! enumerates a binary structural model (confounder O, treatment X0, outcome X1, a response R0
//! that depends on O and a response R1 that depends on X0, every probability a multiple of 1/4)
//! with exact fractions. Its 18 observation patterns have probabilities that are multiples of
//! 1/512, so a table of 4096 rows holds each pattern an exact integer number of times; the
//! recovered effect of that table must be the exact backdoor effect 1/4 and its recovered law the
//! exact joint. For a sampled table of 1500 rows the reference recomputes, from the recovery
//! formula derived in the script, the point effect and every delete-one-row effect, and from the
//! product's bootstrap replicate effects (a seeded stream, an input of the reference) the BCa
//! bias correction, acceleration, adjusted tail probabilities and type-7 interval endpoints at 50
//! digits. This checks the recovery arithmetic and the BCa construction; the resampling stream
//! itself is checked by golden selection digests in `recovery_sampled.rs`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod support {
    pub mod recovery_scm;
}

use std::sync::Arc;

use antecedent_core::ExecutionContext;
use antecedent_estimate::{
    ObservationPattern, ObservationRow, SampledObservationInput, SampledRecoveryConfig,
    derive_sampled_recovery, estimate_sampled_recovery,
};
use antecedent_graph::{Admg, DenseNodeId, NodeRef};
use antecedent_identify::{RecoveredEffectQuery, RecoveryDerivation, RecoveryLimits};
use serde_json::Value;
use support::recovery_scm::{MModel, SNAPSHOT, v};

const EDGES: [(u32, u32); 3] = [(0, 1), (2, 0), (2, 1)];
const SEED: u64 = 99;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

fn model() -> MModel {
    let mut model = MModel::new(2, 1, &EDGES, &[vec![2], vec![0]], 5).unwrap();
    model.mechanisms.insert(2, vec![0.5]);
    model.mechanisms.insert(0, vec![0.25, 0.75]);
    model.mechanisms.insert(1, vec![0.25, 0.5, 0.5, 0.75]);
    model.mechanisms.insert(3, vec![0.5, 0.75]);
    model.mechanisms.insert(4, vec![0.5, 0.75]);
    model
}

fn derivation() -> Box<RecoveryDerivation> {
    let model = model();
    let mut graph = Admg::empty();
    for n in 0..model.k + model.m {
        graph.add_node(NodeRef::Static(v(n))).unwrap();
    }
    for (a, b) in EDGES {
        graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    let effect =
        RecoveredEffectQuery { graph, outcomes: Arc::from([v(1)]), treatments: Arc::from([v(0)]) };
    derive_sampled_recovery(
        &model.graph,
        &model.query(),
        &model.catalog(),
        &effect,
        RecoveryLimits::default(),
        &ctx(),
    )
    .unwrap()
}

fn config() -> SampledRecoveryConfig {
    SampledRecoveryConfig { normalization_tolerance: 0.25, ..SampledRecoveryConfig::bca(SEED) }
}

fn pattern(row: &Value) -> ObservationPattern {
    let get = |k: usize| u8::try_from(row[k].as_u64().unwrap()).unwrap();
    ObservationPattern { responses: get(0), proxies: get(1), fully: get(2) }
}

fn rows(table: &Value) -> SampledObservationInput {
    let mut rows = Vec::new();
    for entry in table.as_array().unwrap() {
        for _ in 0..entry[3].as_u64().unwrap() {
            let id = rows.len() as u64;
            rows.push(ObservationRow { id, pattern: pattern(entry) });
        }
    }
    SampledObservationInput { snapshot_id: SNAPSHOT.to_string(), rows }
}

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/recovery_sampled_observation_reference.json"))
        .expect("frozen reference parses")
}

/// Regenerates the frozen bootstrap replicate effects: reads the tables the reference script
/// prints with `inputs` from `RECOVERY_INPUTS` and writes the replicate effects to
/// `RECOVERY_STREAM`.
#[test]
#[ignore = "regenerates the frozen bootstrap stream of the reference; run by hand"]
fn dump_replicate_stream() {
    let path = std::env::var("RECOVERY_INPUTS").expect("RECOVERY_INPUTS");
    let out = std::env::var("RECOVERY_STREAM").expect("RECOVERY_STREAM");
    let inputs: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let result = estimate_sampled_recovery(
        &derivation(),
        &rows(&inputs["sampled_table"]),
        &config(),
        &ctx(),
    )
    .unwrap();
    let dumped = serde_json::json!({ "replicate_effects": result.replicate_effects });
    std::fs::write(out, serde_json::to_string(&dumped).unwrap()).unwrap();
}

#[test]
fn an_exact_count_table_recovers_the_exact_backdoor_effect_and_joint() {
    let data = reference();
    let result =
        estimate_sampled_recovery(&derivation(), &rows(&data["exact_table"]), &config(), &ctx())
            .unwrap();
    let truth = data["truth_effect"].as_f64().unwrap();
    assert_eq!(truth, 0.25);
    assert!((result.effect - truth).abs() < 1e-12, "{} vs {truth}", result.effect);
    assert_eq!(result.diagnostics.rows, 4096);
    let mut got = result.recovered_law.law().probabilities().to_vec();
    let mut want: Vec<f64> = data["exact_table_law"]
        .as_object()
        .unwrap()
        .values()
        .map(|v| v.as_f64().unwrap())
        .collect();
    got.sort_by(f64::total_cmp);
    want.sort_by(f64::total_cmp);
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(&want) {
        assert!((g - w).abs() < 1e-12, "{g} vs {w}");
    }
}

#[test]
fn point_delete_one_effects_and_the_bca_interval_equal_the_multiprecision_construction() {
    let data = reference();
    let result =
        estimate_sampled_recovery(&derivation(), &rows(&data["sampled_table"]), &config(), &ctx())
            .unwrap();
    // The replicate effects are the frozen stream (the input of the BCa part of the reference).
    let frozen: Vec<f64> =
        data["replicate_effects"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
    assert_eq!(result.replicate_effects.len(), frozen.len());
    for (g, w) in result.replicate_effects.iter().zip(&frozen) {
        assert!((g - w).abs() <= 1e-15 * w.abs().max(1.0));
    }
    let close = |got: f64, want: f64, what: &str| {
        assert!((got - want).abs() <= 1e-9 * want.abs().max(1e-3), "{what}: {got} vs {want}");
    };
    close(result.effect, data["point_effect"].as_f64().unwrap(), "point effect");
    let bca = result.receipt.bca.as_ref().expect("BCa receipt");
    let jackknife = data["jackknife"].as_array().unwrap();
    assert_eq!(bca.jackknife.len(), jackknife.len());
    for record in jackknife {
        let want = pattern(&record["pattern"]);
        let got = bca.jackknife.iter().find(|r| r.pattern == want).expect("pattern present");
        assert_eq!(got.multiplicity, record["multiplicity"].as_u64().unwrap());
        close(got.effect, record["effect"].as_f64().unwrap(), "delete-one effect");
    }
    close(bca.bias_correction, data["bias_correction"].as_f64().unwrap(), "z0");
    close(bca.acceleration, data["acceleration"].as_f64().unwrap(), "acceleration");
    let adjusted = data["adjusted_probabilities"].as_array().unwrap();
    for (g, w) in bca.adjusted_probabilities.iter().zip(adjusted) {
        close(*g, w.as_f64().unwrap(), "adjusted tail probability");
    }
    close(result.interval.lower, data["lower"].as_f64().unwrap(), "lower endpoint");
    close(result.interval.upper, data["upper"].as_f64().unwrap(), "upper endpoint");
    assert!(result.interval.lower < result.effect && result.effect < result.interval.upper);
}
