//! Whole-method candidate measurement; no coverage authority before review and attestation.
#![cfg(feature = "calibration-internal")]

#[path = "common/calibration.rs"]
mod calibration;
#[path = "../../antecedent-estimate/tests/common/candidate_calibration.rs"]
mod candidate;

use antecedent::analysis::graph_imputation::{
    GraphImputationSpec, GraphImputationSupport, graph_imputation_calibration_candidate,
};
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::{ImputationSpec, TabularData};
use calibration::{CoverageTally, RecordKey, grid_n, grid_seed, map_replicates, n_sim};
use std::sync::Arc;

/// T randomized/root, X independent of T, Y=0.8T+0.4X+N(0,1); missing Y depends only on T.
#[test]
#[ignore = "calibration: run through scripts/gate_calibration.sh"]
fn graph_imputation_mar_bic_nearest_rank_candidate() {
    let n = grid_n(200);
    let results = map_replicates(n_sim(), |replicate| {
        let seed = grid_seed(0x6a_3400 + replicate);
        let mut rng = candidate::Generator::new(seed);
        let mut treatment = Vec::with_capacity(n);
        let mut outcome = Vec::with_capacity(n);
        let mut covariate = Vec::with_capacity(n);
        for _ in 0..n {
            let t = rng.binary(0.5) as f64;
            let x = rng.normal();
            let y = 0.8 * t + 0.4 * x + rng.normal();
            treatment.push(t);
            covariate.push(x);
            outcome.push(if rng.uniform() < if t == 0.0 { 0.9 } else { 0.7 } {
                y
            } else {
                f64::NAN
            });
        }
        let source = TabularData::from_f64_columns([
            ("t", treatment.as_slice()),
            ("y", outcome.as_slice()),
            ("x", covariate.as_slice()),
        ])
        .ok()?;
        let spec = GraphImputationSpec {
            treatment: VariableId::from_raw(0),
            outcome: VariableId::from_raw(1),
            support: GraphImputationSupport::TreatmentRootDirectEffect,
            imputation: ImputationSpec::new()
                .with_imputations(8)
                .with_iterations(10)
                .with_seed(219),
            domains: Arc::from([1, 0, 0]),
            posterior_draws: 500,
            prior_scale: 1.0,
            execution_seed: 223,
        };
        graph_imputation_calibration_candidate(source, spec, &ExecutionContext::for_tests(seed))
            .ok()
    });
    let mut tally = CoverageTally::for_record(
        RecordKey {
            test: "graph_imputation_mar_bic_nearest_rank_candidate",
            dgp: "randomized_binary_covariate_linear_normal_mar_by_treatment",
            interval: "nearest_rank_95",
        },
        0.95,
    );
    for result in results {
        match result {
            Some((lower, upper, basis)) => {
                candidate::bind(&mut tally, &basis);
                tally.record(Some((lower, upper)), 0.8);
            }
            None => tally.skip(),
        }
    }
    tally.assert();
}
