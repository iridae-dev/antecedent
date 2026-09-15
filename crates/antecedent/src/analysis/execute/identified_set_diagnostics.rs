// Imbens–Manski / product-posterior envelope diagnostics for class mixtures.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

/// Nominal level of the identified-set interval published on class structural mixtures.
pub const IDENTIFIED_SET_INTERVAL_LEVEL: f64 = 0.9;

/// Seed for one completion's posterior fit: an independent stream per
/// completion key, so draw `k` of different completions shares no random numbers.
pub fn completion_fit_seed(ctx: &ExecutionContext, completion_key: u64) -> u64 {
    ctx.rng.stream(completion_key).next_u64()
}

/// Whether two prepared Bayesian problems fit the same model on the same rows
/// (same design, outcome, contrast, coefficient names and row dependence).
/// Completions whose fitted problems coincide share one posterior: they are one
/// model, and independent redraws of it would only add Monte Carlo spread to
/// the identified set.
pub fn same_fitted_problem(a: &PreparedBayesianProblem, b: &PreparedBayesianProblem) -> bool {
    a.design.nrows == b.design.nrows
        && a.design.ncols == b.design.ncols
        && a.design.matrix == b.design.matrix
        && a.design.outcome == b.design.outcome
        && a.active.to_bits() == b.active.to_bits()
        && a.control.to_bits() == b.control.to_bits()
        && a.coef_names == b.coef_names
        && a.serial_dependence == b.serial_dependence
}

/// Whether two sequential completions fitted the same stationary mechanisms
/// (as a set: each child's likelihood on the same design and rows).
pub fn same_fitted_mechanisms(
    a: &[antecedent_estimate::temporal_sequential::SequentialBayesianMechanism],
    b: &[antecedent_estimate::temporal_sequential::SequentialBayesianMechanism],
) -> bool {
    a.len() == b.len()
        && a.iter().all(|m| {
            b.iter()
                .any(|n| m.variable == n.variable && same_fitted_problem(&m.prepared, &n.prepared))
        })
}

/// Bayesian identified-set interval from identified completions' effect draws
/// (the no-`ClassPrior` path publishes per-completion posteriors only).
///
/// Every fitted completion enters, one draw column each. Completions are fitted
/// on their own seed streams ([`completion_fit_seed`]); completions with the
/// same fitted model carry the same posterior, which leaves the per-draw min /
/// max unchanged. `truncated` records a capped completion enumeration.
pub fn posterior_identified_set_interval<'a>(
    posteriors: impl Iterator<Item = &'a CausalPosterior>,
    rows: usize,
    truncated: bool,
) -> Option<antecedent_estimate::IdentifiedSetInterval> {
    let mut means: Vec<f64> = Vec::new();
    let mut draws = Vec::new();
    for posterior in posteriors {
        let column = posterior.effect_column()?;
        let values = posterior.draws.column(column).ok()?;
        if values.is_empty() {
            return None;
        }
        means.push(values.iter().sum::<f64>() / values.len() as f64);
        draws.push(values);
    }
    antecedent_estimate::imbens_manski_posterior_draws(
        &means,
        &draws,
        rows,
        IDENTIFIED_SET_INTERVAL_LEVEL,
    )
    .map(|interval| interval.with_truncated(truncated))
}

/// Provenance of a published identified-set interval, plus a warning when the
/// set spans a capped completion enumeration.
pub fn identified_set_interval_diagnostics(
    interval: &antecedent_estimate::IdentifiedSetInterval,
) -> Vec<Diagnostic> {
    use antecedent_estimate::IdentifiedSetIntervalMethod as Method;
    let level = interval.level * 100.0;
    let head = format!(
        "{level:.0}% interval [{lo:.6}, {hi:.6}] for the identified set [{bound_lo:.6}, \
         {bound_hi:.6}] over {completions} retained identified completions",
        lo = interval.lower,
        hi = interval.upper,
        bound_lo = interval.bound_lower,
        bound_hi = interval.bound_upper,
        completions = interval.completions,
    );
    let width = if interval.width_retained {
        "set width retained"
    } else {
        "set width within noise: two-sided"
    };
    let crit = interval.critical_value;
    let (se_lo, se_hi) = (interval.lower_se, interval.upper_se);
    let message = match interval.method {
        Method::ImbensManskiSharedBlock => format!(
            "Imbens-Manski {head}; per-completion endpoints min(estimate - c*SD) / \
             max(estimate + c*SD), each completion's SD from shared circular-block replicates \
             refitting every completion on the same resample (endpoint SDs {se_lo:.6} / \
             {se_hi:.6}); critical value {crit:.4} ({width}) on the largest completion SD; \
             covers the true effect with asymptotic probability at least {level:.0}% whenever \
             it is one retained identified completion's effect; conservative while completions \
             nearly agree relative to their SDs"
        ),
        Method::ProductPosteriorEnvelopeQuantile => format!(
            "product-posterior envelope quantile {head}; endpoints are quantiles of the \
             per-draw min / max of independently seeded completion posteriors at tail \
             probability {tail:.4} (critical value {crit:.4}, {width}, on the largest completion \
             posterior SD); every retained identified completion's posterior puts at most \
             {tail:.4} of its mass below the lower endpoint and at most {tail:.4} above the \
             upper one; not a Frequentist Imbens-Manski interval",
            tail = interval.tail_probability(),
        ),
        other => format!(
            "{other:?} {head}; endpoint SDs {se_lo:.6} / {se_hi:.6}; critical value {crit:.4} \
             ({width})"
        ),
    };
    let mut out = vec![Diagnostic::new(
        "estimate.temporal_class.identified_set_interval",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        message,
    )];
    if interval.truncated {
        out.push(Diagnostic::new(
            "estimate.temporal_class.identified_set_interval_truncated",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            format!(
                "completion enumeration or its equivalence audit was capped: the identified-set \
                 interval spans the {} retained identified completions only and says nothing \
                 about the effect under a completion that was not retained",
                interval.completions
            ),
        ));
    }
    out
}
