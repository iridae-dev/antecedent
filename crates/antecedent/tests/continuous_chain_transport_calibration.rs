//! Prospective whole-contrast coverage under a randomized linear sequential DAG.
//! Independent source units and target baselines are regenerated each replicate;
//! the compiler refits both worlds on each paired bootstrap resample. This grid
//! is ignored and does not establish coverage until run and reviewed.

mod common;

use antecedent::analysis::temporal_path_transport::{
    ContinuousChainSourceRow, ContinuousChainTransportRequest, compile_continuous_chain_transport,
};
use antecedent_core::{CalibrationBasis, ExecutionContext};
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts,
    grid_n, grid_seed, map_replicates, n_sim, uniform,
};

const TEST: &str = "continuous_chain_transport_l95";
const DGP: &str = "randomized_linear_sequential_dag_independent_target";
const INTERVAL: &str = "paired_stratified_percentile_bootstrap_95";

fn replicate(seed: u64, n: usize) -> Option<((f64, f64), CalibrationBasis)> {
    let mut next = uniform(seed, 0xC017_1A01);
    let mut source = Vec::with_capacity(n);
    for i in 0..n {
        let treated = i >= n / 2;
        let a = if treated { 1.0 } else { 0.0 };
        let baseline = 2.0 * next() - 1.0;
        let m1 = baseline + a + (2.0 * next() - 1.0);
        let m2 = 2.0 * m1 + 3.0 * a + (2.0 * next() - 1.0);
        let outcome = 3.0 + baseline + m2 + 4.0 * a + (2.0 * next() - 1.0);
        source.push(ContinuousChainSourceRow { baseline, treated, mediators: vec![m1, m2], outcome });
    }
    // Target support is strictly inside the source support; finite-sample
    // overlap failures remain refusals and are counted by the tally.
    let target_baseline = (0..n).map(|_| next() - 0.5).collect();
    let request = ContinuousChainTransportRequest {
        source, target_baseline,
        mediator_parents: vec![vec![], vec![0], vec![0, 1]],
        treatment_edges: vec![true, true, true],
        active_treatment_edges: vec![true, false, false], bootstrap_replicates: 200,
    };
    let ctx = ExecutionContext::for_tests(seed ^ 0xC017_2001);
    let candidate = compile_continuous_chain_transport(request, &ctx).ok()?
        .calibration_candidate(&ctx).ok()?;
    assert_eq!(candidate.result.calibration, "not_measured");
    assert_eq!(candidate.basis.interval_method.as_ref(), INTERVAL);
    Some((candidate.result.interval, candidate.basis))
}

#[test]
#[ignore = "source-only prospective whole-path coverage; no record earned"]
fn continuous_chain_transport_l95() {
    let n = grid_n(240);
    let mut tally = CoverageTally::for_record(RecordKey {
        test: TEST, dgp: DGP, interval: INTERVAL,
    }, REPORTED_LEVEL);
    for result in map_replicates(n_sim(), |rep| replicate(grid_seed(0xC017_7001 + rep), n)) {
        if let Some((interval, basis)) = result {
            let construction = Construction {
                query: basis.query.to_string(), graph_class: basis.graph_class.to_string(),
                structure: basis.structure.to_string(), modality: basis.modality.to_string(),
                inference: basis.inference.to_string(), estimator: basis.estimator.to_string(),
                interval_method: basis.interval_method.to_string(), se_kind: basis.se_kind.to_string(),
                dependence: basis.dependence.to_string(), posterior: basis.posterior.to_string(),
                functional: basis.functional.to_string(), identification: basis.identification.to_string(),
                reported_level: basis.level,
            };
            tally.bind_validation(&construction, ScopeFacts {
                row_count: basis.row_count, replicates_ok: basis.replicates_ok,
                posterior_draws: basis.posterior_draws, unidentified_mass: basis.unidentified_mass,
            }, &basis.validation);
            // Activating only X -> M1 changes M1 by 1, M2 by 2, and Y by 2.
            tally.record(Some(interval), 2.0);
        } else {
            tally.skip();
        }
    }
    tally.assert();
}
