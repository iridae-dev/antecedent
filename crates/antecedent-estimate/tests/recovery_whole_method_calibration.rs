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
fn measure(bca: bool, test: &'static str, expected_id: &str) {
    let n = grid_n(2000);
    let mut tally = CoverageTally::for_record(
        RecordKey {
            test,
            dgp: "binary_confounded_missingness_scm",
            interval: if bca { "bootstrap_bca" } else { "bootstrap_percentile" },
        },
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
            &if bca {
                SampledRecoveryConfig::bca(seed + 100_000)
            } else {
                SampledRecoveryConfig::new(500, seed + 100_000)
            },
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
#[test]
#[ignore = "calibration: final measurement only"]
fn binary_missingness_whole_row_recovery_l95() {
    measure(
        false,
        "binary_missingness_whole_row_recovery_l95",
        "cov.recovered_effect.m_graph.frequentist.bootstrap_percentile.l95.binary_missingness_whole_row_recovery_l95",
    );
}

// Distinct candidate: the failed B500 percentile evidence remains historical.
// Same independent DGP, truth, n grid, seeds, thresholds and failure denominator.
#[test]
#[ignore = "calibration: final measurement only"]
fn binary_missingness_whole_row_recovery_bca_l95() {
    measure(
        true,
        "binary_missingness_whole_row_recovery_bca_l95",
        "cov.recovered_effect.m_graph.frequentist.bootstrap_bca.l95.binary_missingness_whole_row_recovery_bca_l95",
    );
}
