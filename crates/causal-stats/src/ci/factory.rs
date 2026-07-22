//! Name → CI test factory for discovery / Python selection .
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use super::advanced::{Gpdc, KnnDependence, MixedKnnDependence, OracleCi, SymbolicCmi};
use super::bayes::{BayesFactorCi, PosteriorDependenceCi, PosteriorPredictiveCi};
use super::gsquared::{GSquared, RegressionCi};
use super::pairwise_mv::PairwiseMultivariateCi;
use super::parcorr::PartialCorrelation;
use super::parcorr_variants::{MultivariatePartialCorrelation, RobustPartialCorrelation};
use super::types::ConditionalIndependence;
use crate::error::StatsError;

/// Resolve a CI test by stable name string.
///
/// Recognized names (aliases listed):
/// - `parcorr` / `partial_corr` / `partial_correlation`
/// - `robust_parcorr` / `robust_partial_corr`
/// - `weighted_parcorr` / `weighted_partial_corr` — **not** constructible here; requires
///   explicit weights via [`WeightedPartialCorrelation`](super::parcorr_variants::WeightedPartialCorrelation)
///   or `causal::resolve_ci(..., Some(weights))`
/// - `multivariate_parcorr` / `multivariate_partial_corr`
/// - `pairwise_multivariate` / `pairwise_mv`
/// - `gsquared` / `g_squared`
/// - `regression` (`ParCorr` alias)
/// - `knn_dependence` (kNN distance dependence proxy; **not** KSG CMI)
/// - `mixed_knn_dependence` (rank discrete-looking columns, then `knn_dependence`)
/// - `symbolic_cmi`
/// - `gpdc`
/// - `oracle` (empty dependent set ⇒ all independent)
/// - `bayes_factor` / `bayes_factor_ci`
/// - `posterior_dependence` / `posterior_dependence_ci`
/// - `posterior_predictive_ci` / `ppc_ci`
///
/// # Errors
///
/// Unknown name, or `weighted_parcorr` without a weight vector constructor.
pub fn ci_from_name(
    name: &str,
) -> Result<Arc<dyn ConditionalIndependence + Send + Sync>, StatsError> {
    let key = name.trim().to_ascii_lowercase();
    let ci: Arc<dyn ConditionalIndependence + Send + Sync> = match key.as_str() {
        "parcorr" | "partial_corr" | "partial_correlation" => Arc::new(PartialCorrelation::new()),
        "robust_parcorr" | "robust_partial_corr" => Arc::new(RobustPartialCorrelation::new()),
        "weighted_parcorr" | "weighted_partial_corr" => {
            return Err(StatsError::Backend(
                "weighted_parcorr requires observation weights; use \
                 WeightedPartialCorrelation::new(weights) or causal::resolve_ci(..., Some(weights))"
                    .into(),
            ));
        }
        "multivariate_parcorr" | "multivariate_partial_corr" => {
            Arc::new(MultivariatePartialCorrelation::new())
        }
        "pairwise_multivariate" | "pairwise_mv" => Arc::new(PairwiseMultivariateCi::new()),
        "gsquared" | "g_squared" => Arc::new(GSquared::new()),
        "regression" => Arc::new(RegressionCi::new()),
        "knn_dependence" => Arc::new(KnnDependence::new(5)),
        "mixed_knn_dependence" => Arc::new(MixedKnnDependence::new(5)),
        "symbolic_cmi" => Arc::new(SymbolicCmi::new()),
        "gpdc" => Arc::new(Gpdc::new()),
        "oracle" => Arc::new(OracleCi::new([])),
        "bayes_factor" | "bayes_factor_ci" => Arc::new(BayesFactorCi::new()),
        "posterior_dependence" | "posterior_dependence_ci" => {
            Arc::new(PosteriorDependenceCi::new())
        }
        "posterior_predictive_ci" | "ppc_ci" => Arc::new(PosteriorPredictiveCi::new(199)),
        _ => {
            return Err(StatsError::Backend(format!("unknown CI test name: {name}")));
        }
    };
    Ok(ci)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_parcorr_and_rejects_unknown() {
        assert!(ci_from_name("parcorr").is_ok());
        assert!(ci_from_name("gpdc").is_ok());
        assert!(ci_from_name("nope").is_err());
        assert!(ci_from_name("weighted_parcorr").is_err());
        assert!(ci_from_name("knn_dependence").is_ok());
        assert!(ci_from_name("mixed_knn_dependence").is_ok());
        // Legacy KSG-misleading names are not aliases.
        assert!(ci_from_name("cmi_knn").is_err());
        assert!(ci_from_name("knn_cmi").is_err());
    }
}
