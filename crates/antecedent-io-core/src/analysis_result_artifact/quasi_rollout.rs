//! Wire section of a rollout design result: heterogeneity-robust or covariate-adjusted
//! difference in differences, conformal or multi-unit synthetic control, and the local ratio
//! with its design layer.
//!
//! The section is optional on the result body and absent on every other result, so their bodies
//! and digests are unchanged. A loader checks that the section belongs to its query, claims no
//! licence or calibration, publishes no scalar interval beside it, that each pointwise interval
//! is the normal interval of its standard error and each band contains it, and re-derives the
//! scalar summary from the reported effects wherever the design's aggregation defines it from
//! them alone.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::cmp::Ordering;

use serde::{Deserialize, Serialize};

use super::AnalysisResultWire;
use crate::{CausalQueryWire, IoError};

/// Normal 97.5% quantile of the pointwise intervals.
const NORMAL_Q975: f64 = 1.959_963_984_540_054;

/// One reported effect of a rollout design.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DesignEffectWire {
    /// `overall`, `group_time`, `event_time`, `group`, `calendar`, `pre_trend`, `period` or
    /// `unit`.
    pub kind: String,
    /// Adoption cohort.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group: Option<i64>,
    /// Calendar period.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub period: Option<i64>,
    /// Event time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub event_time: Option<i64>,
    /// Treated unit label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
    /// Point estimate.
    pub estimate: f64,
    /// Standard error of the uncertainty construction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub standard_error: Option<f64>,
    /// Pointwise 95% interval.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interval: Option<[f64; 2]>,
    /// Simultaneous 95% band.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub band: Option<[f64; 2]>,
    /// Treated units, rows or post periods behind the estimate.
    pub treated: usize,
    /// Comparison units or rows behind the estimate.
    pub comparison: usize,
}

/// A rollout design result with its whole-procedure uncertainty (unlicensed, calibration
/// unmeasured).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct QuasiRolloutWire {
    /// Design name.
    pub design: String,
    /// Estimator, comparison, adjustment and aggregation words.
    pub method: String,
    /// Reported effects; the first is the scalar summary.
    pub effects: Vec<DesignEffectWire>,
    /// How the uncertainty was computed.
    pub uncertainty: String,
    /// Always `unmeasured`.
    pub calibration: String,
    /// Requested replicates.
    pub replicates_requested: u32,
    /// Successful replicates.
    pub replicates_ok: u32,
    /// Failed replicates.
    pub replicates_failed: u32,
    /// Sup-t critical value of the simultaneous band.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub critical_value: Option<f64>,
    /// Named numeric diagnostics.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub diagnostics: Vec<(String, f64)>,
    /// Always `false`.
    pub licensed: bool,
}

impl QuasiRolloutWire {
    /// The named numeric diagnostic, when present.
    #[must_use]
    pub fn diagnostic(&self, name: &str) -> Option<f64> {
        self.diagnostics.iter().find(|(key, _)| key == name).map(|(_, value)| *value)
    }
}

/// The scalar summary a fresh consumer re-derives from the reported effects alone, when the
/// design's aggregation defines it from them: the mean of the post-adoption event-time effects
/// (dynamic), the mean of the calendar effects (calendar), the mean of the per-period effects
/// (conformal synthetic control) or the post-period-weighted mean of the per-unit effects
/// (multi-unit synthetic control). `None` for aggregations that need cohort sizes the section
/// does not carry.
#[must_use]
pub fn rederive_rollout_summary(wire: &QuasiRolloutWire) -> Option<f64> {
    fn kind_of<'a>(
        wire: &'a QuasiRolloutWire,
        kind: &'a str,
    ) -> impl Iterator<Item = &'a DesignEffectWire> + 'a {
        wire.effects.iter().filter(move |effect| effect.kind == kind)
    }
    let of = |kind: &'static str| kind_of(wire, kind);
    let mean = |values: Vec<f64>| {
        (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
    };
    match wire.design.as_str() {
        "staggered_did" if wire.method.contains(".dynamic") => mean(
            of("event_time")
                .filter(|effect| effect.event_time.is_some_and(|event| event >= 0))
                .map(|effect| effect.estimate)
                .collect(),
        ),
        "staggered_did" if wire.method.contains(".calendar") => {
            mean(of("calendar").map(|effect| effect.estimate).collect())
        }
        "conformal_synthetic_control" => mean(of("period").map(|effect| effect.estimate).collect()),
        "multi_unit_synthetic_control" | "multi_unit_synthetic_did" => {
            let units: Vec<&DesignEffectWire> = of("unit").collect();
            let total: usize = units.iter().map(|unit| unit.treated).sum();
            (total > 0).then(|| {
                units.iter().map(|unit| unit.estimate * unit.treated as f64).sum::<f64>()
                    / total as f64
            })
        }
        _ => None,
    }
}

fn design_matches(design: &str, query: &CausalQueryWire) -> bool {
    match query {
        CausalQueryWire::PanelDid(query) => {
            query.rollout.as_ref().is_some_and(|rollout| {
                (design == "covariate_adjusted_did")
                    == (rollout.design == "covariate_adjusted_two_period")
            }) && matches!(design, "staggered_did" | "covariate_adjusted_did")
        }
        CausalQueryWire::SyntheticControl(query) => {
            query.rollout.is_some()
                && matches!(
                    design,
                    "conformal_synthetic_control"
                        | "posterior_predictive_synthetic_control"
                        | "multi_unit_synthetic_control"
                        | "multi_unit_synthetic_did"
                )
        }
        CausalQueryWire::LocalPolynomialRatio(query) => {
            query.options.as_ref().is_some_and(|options| {
                let temporal = options.temporal_replicates.is_some();
                let label = if query.kink { "regression_kink" } else { "fuzzy_rd" };
                if temporal { design == format!("temporal_{label}") } else { design == label }
            })
        }
        _ => false,
    }
}

/// Check a rollout section against its query and re-derive what the section determines.
pub(in crate::analysis_result_artifact) fn validate_quasi_rollout(
    result: &AnalysisResultWire,
) -> Result<(), IoError> {
    let Some(rollout) = &result.quasi_rollout else {
        return Ok(());
    };
    let fail = |message: &str| Err(IoError::Convert(format!("quasi rollout section: {message}")));
    if !design_matches(&rollout.design, &result.query) {
        return fail("the design does not belong to the query");
    }
    // A Bayesian rollout section rides beside its posterior; a frequentist one has none and
    // publishes no scalar standard error.
    let bayesian = rollout.uncertainty.starts_with("bayesian_");
    if bayesian != result.posterior_artifact.is_some() {
        return fail("a Bayesian rollout section needs its posterior, a frequentist one none");
    }
    if rollout.licensed
        || rollout.calibration != "unmeasured"
        || (!bayesian && result.standard_error.is_some())
        || result.interval_lower.is_some()
        || result.interval_upper.is_some()
    {
        return fail("a rollout result claims no licence, calibration or scalar interval");
    }
    if bayesian && rollout.effects.iter().any(|effect| effect.standard_error.is_some()) {
        return fail("a credible interval carries no normal-interval standard error");
    }
    let Some(overall) = rollout.effects.first() else {
        return fail("the scalar summary is missing");
    };
    if overall.kind != "overall"
        || rollout.effects[1..].iter().any(|effect| effect.kind == "overall")
    {
        return fail("exactly the first effect is the scalar summary");
    }
    if result.estimate.is_some_and(|estimate| {
        (estimate - overall.estimate).abs() > 1e-9 * (1.0 + overall.estimate.abs())
    }) {
        return fail("the published estimate differs from the scalar summary");
    }
    if rollout.replicates_ok.checked_add(rollout.replicates_failed)
        != Some(rollout.replicates_requested)
    {
        return fail("replicate counts do not add up");
    }
    if rollout.critical_value.is_some_and(|critical| {
        critical.partial_cmp(&(NORMAL_Q975 - 1e-12)).is_none_or(Ordering::is_lt)
    }) {
        return fail("a simultaneous critical value is below the pointwise quantile");
    }
    let near = |a: f64, b: f64| (a - b).abs() <= 1e-9 * (1.0 + a.abs().max(b.abs()));
    for effect in &rollout.effects {
        if !effect.estimate.is_finite() {
            return fail("an effect estimate is not finite");
        }
        if let Some(interval) = effect.interval {
            if interval[0] > interval[1] {
                return fail("a pointwise interval is reversed");
            }
            // A conformal set is the hull of the accepted effects and need not contain the
            // pre-period point; a standard-error interval is centred on its estimate.
            if let Some(se) = effect.standard_error {
                if se.partial_cmp(&0.0) != Some(Ordering::Greater)
                    || !near(interval[0], effect.estimate - NORMAL_Q975 * se)
                    || !near(interval[1], effect.estimate + NORMAL_Q975 * se)
                {
                    return fail(
                        "a pointwise interval is not the normal interval of its standard error",
                    );
                }
            }
            if let Some(band) = effect.band {
                if band[0] > interval[0] + 1e-12 || band[1] < interval[1] - 1e-12 {
                    return fail("a simultaneous band does not contain its pointwise interval");
                }
            }
        } else if effect.band.is_some() {
            return fail("a band without its pointwise interval");
        }
    }
    if let Some(summary) = rederive_rollout_summary(rollout) {
        if !near(summary, overall.estimate) {
            return fail("the scalar summary does not re-derive from the reported effects");
        }
    }
    Ok(())
}
