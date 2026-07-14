//! Name → CI test factory for discovery / Python selection (Phase 5).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::ExecutionContext;

use super::advanced::{Gpdc, KnnCmi, MixedKnnCmi, OracleCi, SymbolicCmi};
use super::gsquared::{GSquared, RegressionCi};
use super::pairwise_mv::PairwiseMultivariateCi;
use super::parcorr::PartialCorrelation;
use super::parcorr_variants::{
    MultivariatePartialCorrelation, RobustPartialCorrelation, WeightedPartialCorrelation,
};
use super::types::{
    CiBatchRequest, CiBatchResult, CiWorkspace, ConditionalIndependence,
    ConditionalIndependenceTest,
};
use crate::error::StatsError;

/// Resolve a CI test by stable name string.
///
/// Recognized names (aliases listed):
/// - `parcorr` / `partial_corr` / `partial_correlation`
/// - `robust_parcorr` / `robust_partial_corr`
/// - `weighted_parcorr` / `weighted_partial_corr` (unit weights at query time)
/// - `multivariate_parcorr` / `multivariate_partial_corr`
/// - `pairwise_multivariate` / `pairwise_mv`
/// - `gsquared` / `g_squared`
/// - `regression`
/// - `cmi_knn` / `knn_cmi`
/// - `mixed_cmi_knn` / `mixed_knn_cmi`
/// - `symbolic_cmi`
/// - `gpdc`
/// - `oracle` (empty dependent set ⇒ all independent)
///
/// # Errors
///
/// Unknown name.
pub fn ci_from_name(
    name: &str,
) -> Result<Arc<dyn ConditionalIndependence + Send + Sync>, StatsError> {
    let key = name.trim().to_ascii_lowercase();
    let ci: Arc<dyn ConditionalIndependence + Send + Sync> = match key.as_str() {
        "parcorr" | "partial_corr" | "partial_correlation" => Arc::new(PartialCorrelation::new()),
        "robust_parcorr" | "robust_partial_corr" => Arc::new(RobustPartialCorrelation::new()),
        "weighted_parcorr" | "weighted_partial_corr" => Arc::new(UnitWeightedParCorr),
        "multivariate_parcorr" | "multivariate_partial_corr" => {
            Arc::new(MultivariatePartialCorrelation::new())
        }
        "pairwise_multivariate" | "pairwise_mv" => Arc::new(PairwiseMultivariateCi::new()),
        "gsquared" | "g_squared" => Arc::new(GSquared::new()),
        "regression" => Arc::new(RegressionCi::new()),
        "cmi_knn" | "knn_cmi" => Arc::new(KnnCmi::new(5)),
        "mixed_cmi_knn" | "mixed_knn_cmi" => Arc::new(MixedKnnCmi::new(5)),
        "symbolic_cmi" => Arc::new(SymbolicCmi::new()),
        "gpdc" => Arc::new(Gpdc::new()),
        "oracle" => Arc::new(OracleCi::new([])),
        _ => {
            return Err(StatsError::Backend(format!("unknown CI test name: {name}")));
        }
    };
    Ok(ci)
}

/// Weighted ParCorr with implicit unit weights sized to each batch.
#[derive(Clone, Copy, Debug, Default)]
struct UnitWeightedParCorr;

impl ConditionalIndependenceTest for UnitWeightedParCorr {
    fn test_batch(
        &self,
        request: &CiBatchRequest<'_>,
        workspace: &mut CiWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<CiBatchResult, StatsError> {
        let n = request.columns.first().map(|c| c.len()).unwrap_or(0);
        WeightedPartialCorrelation::new(vec![1.0; n]).test_batch(request, workspace, ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_parcorr_and_rejects_unknown() {
        assert!(ci_from_name("parcorr").is_ok());
        assert!(ci_from_name("gpdc").is_ok());
        assert!(ci_from_name("nope").is_err());
    }
}
