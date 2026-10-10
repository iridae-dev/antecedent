//! Frozen binary missingness-SCM design. The truth .36 is obtained directly
//! from the complete-data structural mechanisms, not the recovered sample.
//! Every replicate reidentifies/reconstructs the empirical observation law,
//! jointly recovers it and bootstraps whole observation rows. No calibration runs
//! in ordinary tests; failures count as misses under the shared cap and MCSE rules.
#[path = "../../antecedent/tests/common/calibration.rs"]
mod calibration;
#[path = "common/candidate_calibration.rs"]
mod candidate;
mod support {
    pub mod recovery_scm;
}
use antecedent_core::ExecutionContext;
use antecedent_estimate::{
    ObservationPattern, ObservationRow, SampledObservationInput, SampledRecoveryConfig,
    derive_sampled_recovery, estimate_sampled_recovery,
};
use antecedent_graph::{Admg, DenseNodeId};
use antecedent_identify::{RecoveredEffectQuery, RecoveryLimits};
use calibration::{CoverageTally, RecordKey, grid_n, grid_seed, map_replicates, n_sim};
use std::sync::Arc;
use support::recovery_scm::{MModel, SNAPSHOT, v};
const EDGES: [(u32, u32); 3] = [(0, 1), (2, 0), (2, 1)];
fn model() -> MModel {
    let mut m = MModel::new(2, 1, &EDGES, &[vec![2], vec![0]], 5).unwrap();
    m.mechanisms.insert(2, vec![0.4]);
    m.mechanisms.insert(0, vec![0.3, 0.7]);
    m.mechanisms.insert(1, vec![0.2, 0.5, 0.6, 0.8]);
    m.mechanisms.insert(3, vec![0.5, 0.8]);
    m.mechanisms.insert(4, vec![0.6, 0.9]);
    m
}
fn input(n: usize, seed: u64) -> SampledObservationInput {
    let mut rng = candidate::Generator::new(seed);
    let mut rows = Vec::with_capacity(n);
    for id in 0..n {
        let o = rng.binary(0.4);
        let t = rng.binary([0.3, 0.7][o]);
        let y = rng.binary([0.2, 0.5, 0.6, 0.8][2 * t + o]);
        let r0 = rng.binary([0.5, 0.8][o]);
        let r1 = rng.binary([0.6, 0.9][t]);
        let bit = |v: usize| u8::try_from(v).unwrap();
        rows.push(ObservationRow {
            id: u64::try_from(id).unwrap(),
            pattern: ObservationPattern {
                responses: bit(r0 | (r1 << 1)),
                proxies: bit((r0 & t) | ((r1 & y) << 1)),
                fully: bit(o),
            },
        });
    }
    SampledObservationInput { snapshot_id: SNAPSHOT.to_owned(), rows }
}

/// Ordinary identity proof using the exact original calibration construction;
/// no coverage tally, simulation grid or statistical measurement is performed.
#[test]
fn recovery_bca_original_measurement_identity_receipt() {
    let seed = 0x46ab_0000;
    let ctx = ExecutionContext::for_tests(seed);
    let source = model();
    let mut graph = Admg::with_variables(3);
    for (a, b) in EDGES {
        graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    let effect =
        RecoveredEffectQuery { graph, outcomes: Arc::from([v(1)]), treatments: Arc::from([v(0)]) };
    let catalog = source.catalog();
    let derivation = derive_sampled_recovery(
        &source.graph,
        &source.query(),
        &catalog,
        &effect,
        RecoveryLimits::default(),
        &ctx,
    )
    .unwrap();
    let sample = input(1000, seed);
    let config = SampledRecoveryConfig::bca(seed + 100_000);
    let result = estimate_sampled_recovery(&derivation, &sample, &config, &ctx).unwrap();
    assert!(result.receipt.verify_digest());
    assert_eq!(result.receipt.config, config);
    assert_eq!(result.failed_replicates, 0);
    let original_functional = format!(
        "recovered_effect:{}:treated={:016x}:control={:016x}",
        result.receipt.derivation_identity,
        config.treated_level.to_bits(),
        config.control_level.to_bits(),
    );
    // Parse the original emitted TOML string using its JSON-compatible quoted
    // string representation. This binds the native receipt to the authentic
    // registry row rather than a manually reconstructed expected identity.
    let original_row = include_str!("../../../parity/coverage_records.toml")
        .split("[[record]]")
        .find(|row| row.contains(
            "id = \"cov.recovered_effect.m_graph.frequentist.bootstrap_bca.l95.binary_missingness_whole_row_recovery_bca_l95\""
        )).unwrap();
    let functional_literal =
        original_row.lines().find_map(|line| line.strip_prefix("functional = ")).unwrap();
    let recorded_functional: String = serde_json::from_str(functional_literal).unwrap();
    assert_eq!(original_functional, recorded_functional);
    let canonical_functional = result.calibration_basis().functional.to_string();
    assert_ne!(original_functional, canonical_functional);
    let mut changed_graph = source.graph.clone();
    changed_graph.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(3)).unwrap();
    let mut changed_effect = effect.clone();
    changed_effect.outcomes = Arc::from([v(2)]);
    let mut changed_catalog = catalog.clone();
    Arc::make_mut(&mut changed_catalog.bindings)[0].sampling =
        antecedent_core::SamplingDesign::Clustered;
    let mutation_result = |graph: &Admg,
                           effect: &RecoveredEffectQuery,
                           catalog: &antecedent_core::EvidenceCatalog| {
        match derive_sampled_recovery(
            graph,
            &source.query(),
            catalog,
            effect,
            RecoveryLimits::default(),
            &ctx,
        ) {
            Err(_) => "refused",
            Ok(changed) => {
                assert_ne!(changed.scientific_identity(), derivation.scientific_identity());
                "different_scientific_identity"
            }
        }
    };
    let mutations = serde_json::json!({
        "graph": mutation_result(&changed_graph, &effect, &catalog),
        "effect": mutation_result(&source.graph, &changed_effect, &catalog),
        "sampling": mutation_result(&source.graph, &effect, &changed_catalog),
    });
    let replay = estimate_sampled_recovery(&derivation, &sample, &config, &ctx).unwrap();
    assert_eq!(result.receipt, replay.receipt);
    assert_eq!(result.replicate_effects, replay.replicate_effects);
    assert_eq!(result.recovered_cell_covariance, replay.recovered_cell_covariance);
    println!(
        "RECOVERY_IDENTITY_PROOF={}",
        serde_json::json!({
            "version": 1,
            "record_id": "cov.recovered_effect.m_graph.frequentist.bootstrap_bca.l95.binary_missingness_whole_row_recovery_bca_l95",
            "original_functional": original_functional,
            "canonical_functional": canonical_functional,
            "config": {
                "interval_method": config.interval_method.name(),
                "replicates": config.replicates, "seed": config.seed,
                "max_failed_fraction_bits": format!("{:016x}", config.max_failed_fraction.to_bits()),
                "normalization_tolerance_bits": format!("{:016x}", config.normalization_tolerance.to_bits()),
                "small_cell_count_bits": format!("{:016x}", config.small_cell_count.to_bits()),
                "treated_level_bits": format!("{:016x}", config.treated_level.to_bits()),
                "control_level_bits": format!("{:016x}", config.control_level.to_bits()),
            },
            "rows": result.receipt.rows, "input_digest": result.receipt.input_digest,
            "receipt_digest": result.receipt.receipt_digest,
            "scientific_derivation_identity": derivation.scientific_identity().as_ref(),
            "negative_mutations": mutations,
            "identical_repeated_native_receipt": true,
        })
    );
}

fn measure(test: &'static str, expected_id: &str) {
    let n = grid_n(2000);
    let mut tally = CoverageTally::for_record(
        RecordKey { test, dgp: "binary_confounded_missingness_scm", interval: "bootstrap_bca" },
        0.95,
    );
    let results = map_replicates(n_sim(), |rep| {
        let seed = grid_seed(0x46ab_0000 + rep);
        let ctx = ExecutionContext::for_tests(seed);
        let model = model();
        let mut graph = Admg::with_variables(3);
        for (a, b) in EDGES {
            graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        let effect = RecoveredEffectQuery {
            graph,
            outcomes: Arc::from([v(1)]),
            treatments: Arc::from([v(0)]),
        };
        let derivation = derive_sampled_recovery(
            &model.graph,
            &model.query(),
            &model.catalog(),
            &effect,
            RecoveryLimits::default(),
            &ctx,
        )?;
        estimate_sampled_recovery(
            &derivation,
            &input(n, seed),
            &SampledRecoveryConfig::bca(seed + 100_000),
            &ctx,
        )
    });
    // sum_o P(o)[P(Y=1|T=1,o)-P(Y=1|T=0,o)]=.6*.4+.4*.3=.36.
    for result in results {
        match result {
            Ok(result) => {
                candidate::bind(&mut tally, &result.calibration_basis());
                tally.record(Some((result.interval.lower, result.interval.upper)), 0.36);
            }
            Err(_) => tally.skip(),
        }
    }
    assert_eq!(tally.record_id().as_deref(), Some(expected_id));
    tally.assert();
}
// Distinct candidate: the failed B500 percentile evidence remains historical.
// Same independent DGP, truth, n grid, seeds, thresholds and failure denominator.
#[test]
#[ignore = "calibration: final measurement only"]
fn binary_missingness_whole_row_recovery_bca_l95() {
    measure(
        "binary_missingness_whole_row_recovery_bca_l95",
        "cov.recovered_effect.m_graph.frequentist.bootstrap_bca.l95.binary_missingness_whole_row_recovery_bca_l95",
    );
}
