//! Conditional-independence surface owned by discovery.
//!
//! Numeric kernels remain in `antecedent-stats`; this module re-exports the DESIGN
//! trait contract so discovery algorithms depend on a discovery-owned CI API.
//! Concrete test constructors live in `antecedent-stats` (use
//! [`ci_from_name`] or import the type from that crate).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::ensure_alpha_resolvable;

use crate::error::DiscoveryError;

/// Refuse a CI test / `alpha` / multiplicity combination whose decisions would not mean what
/// they say.
///
/// - A permutation test cannot report a p-value below `1/(replicates+1)`; with `alpha` at or
///   below that floor every test "fails to reject" whatever the data show, which reads as
///   evidence of independence (default 49 replicates: nothing rejects at `alpha <= 0.02`).
/// - A test whose `p_value` is a posterior probability (the Bayesian tests) is not a
///   frequentist p-value, so Benjamini–Hochberg / FDR adjustment of it is meaningless.
///
/// # Errors
///
/// [`DiscoveryError::Stats`] when `alpha` is unresolvable;
/// [`DiscoveryError::Unsupported`] when FDR is requested for a non-frequentist p-value.
pub fn ensure_ci_decisions_meaningful(
    ci: &dyn ConditionalIndependence,
    significance: SignificanceMethod,
    alpha: f64,
    fdr_requested: bool,
) -> Result<(), DiscoveryError> {
    ensure_alpha_resolvable(ci.min_attainable_p(significance), alpha)?;
    if fdr_requested && !ci.p_value_is_frequentist() {
        return Err(DiscoveryError::unsupported(
            "FDR adjustment requires frequentist p-values; this CI test reports a posterior \
             probability of independence in p_value",
        ));
    }
    Ok(())
}

pub use antecedent_stats::{
    CiBatchRequest, CiBatchResult, CiPreparationPlan, CiQuery, CiResult, CiWorkspace,
    ConditionalIndependence, ConditionalIndependenceTest, ConfidenceMethod, PartialCorrelation,
    PreparedCiTest, SignificanceMethod, ci_from_name,
};
