//! MCMC publication diagnostics for a fitted posterior.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_estimate::CausalPosterior;
use antecedent_prob::{HessianFactorization, InferenceDiagnostics};

use crate::common::RefutationReport;

/// MCMC chain diagnostics gate (ESS / R-hat / divergences).
///
/// Applicable only when the posterior was produced by an MCMC backend
/// (`InferenceDiagnostics::factorization == Mcmc`). The verdict is the library's own
/// publication predicate ([`antecedent_prob::InferenceDiagnostics::allows_posterior`]: split-Ř at
/// most 1.01, bulk and tail ESS of at least 100 per chain, no post-warmup divergences, movement
/// on every chain), so there are no thresholds to loosen here; a failure names each clause
/// that failed.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct McmcDiagnosticsCheck;

impl McmcDiagnosticsCheck {
    /// Construct.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Evaluate against a fitted posterior's diagnostics.
    ///
    /// Returns `None` when the posterior is not MCMC (caller should emit `NotApplicable`).
    #[must_use]
    pub fn check(&self, posterior: &CausalPosterior) -> Option<RefutationReport> {
        let ate = posterior
            .effect_column()
            .and_then(|c| posterior.summaries.mean.get(c).copied())
            .unwrap_or(f64::NAN);
        mcmc_report(&posterior.diagnostics, ate)
    }
}

/// Publication verdict of `d` as a report; `None` unless `d` is an MCMC diagnostic set.
pub(super) fn mcmc_report(d: &InferenceDiagnostics, ate: f64) -> Option<RefutationReport> {
    if d.factorization != HessianFactorization::Mcmc {
        return None;
    }
    let mut reasons: Vec<&'static str> = Vec::new();
    if !d.converged {
        reasons.push("the sampler did not report convergence");
    }
    reasons.extend(d.mcmc_publication_failures());
    let passed = reasons.is_empty();
    let rhat = d.rhat_max.unwrap_or(f64::INFINITY);
    Some(RefutationReport {
        refuter: Arc::from("mcmc_diagnostics"),
        original_ate: ate,
        refuted_ate: ate,
        comparison: rhat,
        informative: true,
        passed,
        failure_condition: if passed {
            None
        } else {
            Some(Arc::from(format!(
                "MCMC diagnostics failed: {} (rhat={rhat:.4} ess={:.1} tail_ess={:.1} \
                 divergences={})",
                reasons.join("; "),
                d.ess_bulk_min.unwrap_or(0.0),
                d.ess_tail_min.unwrap_or(0.0),
                d.n_postwarmup_divergences.map_or_else(|| "unknown".to_owned(), |n| n.to_string())
            )))
        },
        replicates: d.n_chains.unwrap_or(0),
    })
}
