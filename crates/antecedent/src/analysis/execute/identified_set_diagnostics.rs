// Imbens–Manski / product-posterior envelope diagnostics for class mixtures.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

/// Nominal level of the identified-set interval published on class structural mixtures.
pub const IDENTIFIED_SET_INTERVAL_LEVEL: f64 = 0.9;

/// Bayesian identified-set interval from identified completions' effect draws
/// (K-2: the no-`ClassPrior` path publishes per-completion posteriors only).
pub fn posterior_identified_set_interval<'a>(
    posteriors: impl Iterator<Item = &'a CausalPosterior>,
    rows: usize,
) -> Option<antecedent_estimate::IdentifiedSetInterval> {
    let mut means: Vec<f64> = Vec::new();
    let mut draws = Vec::new();
    for posterior in posteriors {
        let column = posterior.effect_column()?;
        let values = posterior.draws.column(column).ok()?;
        if values.is_empty() {
            return None;
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        // Completions with the same estimand fit the same posterior on the same
        // rows; their draws are independent copies of one distribution, and a
        // per-draw min / max over copies would widen the bounds spuriously.
        if means.iter().any(|m| (m - mean).abs() <= 1e-9 * mean.abs().max(1.0)) {
            continue;
        }
        means.push(mean);
        draws.push(values);
    }
    antecedent_estimate::imbens_manski_posterior_draws(
        &means,
        &draws,
        rows,
        IDENTIFIED_SET_INTERVAL_LEVEL,
    )
}

pub fn identified_set_interval_diagnostic(
    interval: &antecedent_estimate::IdentifiedSetInterval,
) -> Diagnostic {
    let (label, source) = match interval.method {
        antecedent_estimate::IdentifiedSetIntervalMethod::ProductPosteriorEnvelopeQuantile => (
            "product-posterior envelope quantile",
            "per-completion posterior draws paired by index (endpoint quantiles of the \
             per-draw min / max at Imbens-Manski tail probabilities; not a Frequentist IM interval)",
        ),
        _ => (
            "Imbens-Manski",
            "shared circular-block replicates refitting every completion on the same resample",
        ),
    };
    Diagnostic::new(
        "estimate.temporal_class.identified_set_interval",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "{label} {level:.0}% interval [{lo:.6}, {hi:.6}] for the identified set \
             [{bound_lo:.6}, {bound_hi:.6}] over {completions} identified completions; \
             bound SDs {se_lo:.6} / {se_hi:.6} from {source}; critical value {crit:.4} ({width}); \
             covers the true effect whenever it is one completion's effect; \
             conservative when completions nearly agree",
            level = interval.level * 100.0,
            lo = interval.lower,
            hi = interval.upper,
            bound_lo = interval.bound_lower,
            bound_hi = interval.bound_upper,
            completions = interval.completions,
            se_lo = interval.lower_se,
            se_hi = interval.upper_se,
            crit = interval.critical_value,
            width = if interval.width_retained {
                "set width retained"
            } else {
                "set width within noise: two-sided"
            },
        ),
    )
}

