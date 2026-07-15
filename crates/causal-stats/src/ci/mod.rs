//! Conditional independence tests (DESIGN.md §12).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod advanced;
mod analytic;
mod block_shuffle;
mod calibration;
mod factory;
mod gsquared;
mod pairwise_mv;
mod parcorr;
mod parcorr_variants;
mod types;

pub use advanced::{Gpdc, KnnCmi, MixedKnnCmi, OracleCi, SymbolicCmi};
pub use analytic::analytic_parcorr_ci;
pub use calibration::{
    CalibrationReport, calibrate_gsquared, calibrate_parcorr_like, type_i_within_two_se,
};
pub use factory::ci_from_name;
pub use gsquared::{GSquared, RegressionCi};
pub use pairwise_mv::{PairwiseMultivariateCi, pairwise_multivariate_test};
pub use parcorr::PartialCorrelation;
pub use parcorr_variants::{
    MultivariatePartialCorrelation, RobustPartialCorrelation, WeightedPartialCorrelation,
};
pub use types::{
    CiBatchRequest, CiBatchResult, CiPreparationPlan, CiQuery, CiResult, CiWorkspace,
    ConditionalIndependence, ConditionalIndependenceTest, ConfidenceMethod, KnnCmiWorkspace,
    PreparedCiTest, SignificanceMethod, analytic_confidence_level, nonparametric_permutation_count,
};

#[cfg(test)]
#[allow(clippy::cast_precision_loss, clippy::many_single_char_names)]
mod tests {
    use causal_core::ExecutionContext;

    use super::*;

    #[test]
    fn independent_noise_high_pvalue() {
        let n = 300usize;
        let x: Vec<f64> = (0..n).map(|i| (i % 7) as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| (i % 11) as f64).collect();
        let cols: [&[f64]; 2] = [&x, &y];
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 0 }];
        let req = CiBatchRequest {
            columns: &cols,
            queries: &queries,
            z_flat: &[],
            significance: SignificanceMethod::Analytic,
            confidence: ConfidenceMethod::default(),
        };
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let out = PartialCorrelation::new().test_batch_adhoc(&req, &mut ws, &ctx).unwrap();
        assert!(out.results[0].p_value > 0.01, "p={}", out.results[0].p_value);
    }

    #[test]
    fn dependent_low_pvalue() {
        let n = 200usize;
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| 2.0 * i as f64 + 0.01).collect();
        let cols: [&[f64]; 2] = [&x, &y];
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 0 }];
        let req = CiBatchRequest {
            columns: &cols,
            queries: &queries,
            z_flat: &[],
            significance: SignificanceMethod::Analytic,
            confidence: ConfidenceMethod::default(),
        };
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(2);
        let out = PartialCorrelation::new().test_batch_adhoc(&req, &mut ws, &ctx).unwrap();
        assert!(out.results[0].p_value < 1e-6);
        assert!(out.results[0].statistic > 0.99);
    }

    #[test]
    fn block_shuffle_runs() {
        let n = 120usize;
        let x: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let y: Vec<f64> = (0..n).map(|i| (i as f64).cos()).collect();
        let cols: [&[f64]; 2] = [&x, &y];
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 0 }];
        let req = CiBatchRequest {
            columns: &cols,
            queries: &queries,
            z_flat: &[],
            significance: SignificanceMethod::BlockShuffle { replicates: 50, block_size: 10 },
            confidence: ConfidenceMethod::default(),
        };
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(3);
        let out = PartialCorrelation::new().test_batch_adhoc(&req, &mut ws, &ctx).unwrap();
        assert!((0.0..=1.0).contains(&out.results[0].p_value));
        assert!(out.results[0].p_value > 0.0);
    }

    #[test]
    fn knn_reuses_permutation_plan_across_queries() {
        let n = 80usize;
        let x: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).cos()).collect();
        let z: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let cols: [&[f64]; 3] = [&x, &y, &z];
        let queries = [
            CiQuery { x: 0, y: 1, z_start: 0, z_len: 1 },
            CiQuery { x: 0, y: 1, z_start: 0, z_len: 1 },
            CiQuery { x: 0, y: 1, z_start: 0, z_len: 1 },
        ];
        let z_flat = [2usize];
        let req = CiBatchRequest {
            columns: &cols,
            queries: &queries,
            z_flat: &z_flat,
            significance: SignificanceMethod::Analytic,
            confidence: ConfidenceMethod::default(),
        };
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(9);
        let _ = KnnCmi::new(3).test_batch_adhoc(&req, &mut ws, &ctx).unwrap();
        let gen_after_first = ws.knn.index_generation;
        let builds_after_first = ws.knn.index_builds;
        let perm_ptr = ws.knn.perm.as_ptr();
        let _ = KnnCmi::new(3).test_batch_adhoc(&req, &mut ws, &ctx).unwrap();
        assert_eq!(ws.knn.index_generation, gen_after_first, "index must not rebuild per batch");
        assert_eq!(ws.knn.index_builds, builds_after_first, "MatchingIndex builds must stay flat");
        assert_eq!(ws.knn.perm.as_ptr(), perm_ptr, "permutation plan buffer must be reused");
        assert!(ws.knn.index.is_some());
    }
}
