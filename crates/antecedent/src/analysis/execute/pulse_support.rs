//! Support of a pulse's treatment levels in the observed treatment series.
//!
//! A pulse is a counterfactual dose. The fitted lag response is linear in the treatment, so a
//! level the series never visited is reached by extrapolation: the answer scales with the level
//! and says nothing about whether the response stays linear there. This module measures where
//! the control and the active level lie against the observed treatment interval and standard
//! deviation, names a level outside the interval in the support slot with those measures, and
//! refuses an active level further than a declared number of observed ranges beyond it.
//!
//! The control level is only reported: a control of 0 outside the observed interval is how
//! pulses are written, and it is never the refused quantity.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{
    Diagnostic, DiagnosticKind, DiagnosticSeverity, ExecutionContext, Intervention,
    TemporalEffectQuery, TemporalPolicy,
};
use antecedent_data::{TableView, TimeSeriesData};

use crate::CausalError;

/// Support diagnostic code of a pulse level outside the observed treatment interval.
pub(super) const PULSE_EXTRAPOLATION_CODE: &str = "estimate.temporal.pulse.level_extrapolation";

/// The observed treatment interval and spread a pulse is read against.
struct Observed {
    min: f64,
    max: f64,
    mean: f64,
    sd: f64,
}

impl Observed {
    fn of(data: &TimeSeriesData, query: &TemporalEffectQuery) -> Option<Self> {
        let values = data.float64_values(query.treatment).ok()?;
        let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
        if finite.is_empty() {
            return None;
        }
        let n = finite.len() as f64;
        let mean = finite.iter().sum::<f64>() / n;
        let sd = if finite.len() > 1 {
            (finite.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
        } else {
            0.0
        };
        let min = finite.iter().copied().fold(f64::INFINITY, f64::min);
        let max = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Some(Self { min, max, mean, sd })
    }

    fn range(&self) -> f64 {
        self.max - self.min
    }

    /// How far `level` lies outside `[min, max]`; zero inside.
    fn distance_outside(&self, level: f64) -> f64 {
        (self.min - level).max(level - self.max).max(0.0)
    }
}

fn hard_level(intervention: &Intervention) -> Option<f64> {
    match intervention {
        Intervention::Set { value, .. } => value.as_f64(),
        _ => None,
    }
}

/// The support of a pulse's levels: the diagnostics to publish, or a refusal when the active
/// level lies beyond the extrapolation bound of `ctx`.
///
/// Anything but a pulse on a numeric treatment reads no support and returns nothing.
///
/// # Errors
///
/// `positivity_violated` when the active level lies more than the bound's number of observed
/// treatment ranges outside the observed treatment interval.
pub(super) fn pulse_support(
    data: &TimeSeriesData,
    query: &TemporalEffectQuery,
    ctx: &ExecutionContext,
) -> Result<Vec<Diagnostic>, CausalError> {
    if !matches!(query.policy, TemporalPolicy::Pulse { .. }) {
        return Ok(Vec::new());
    }
    let (Some(observed), Some(active), Some(control)) =
        (Observed::of(data, query), hard_level(&query.active), hard_level(&query.control))
    else {
        return Ok(Vec::new());
    };
    let range = observed.range();
    let active_distance = observed.distance_outside(active);
    let control_distance = observed.distance_outside(control);
    if active_distance == 0.0 && control_distance == 0.0 {
        return Ok(Vec::new());
    }
    let ranges_of = |distance: f64| if range > 0.0 { distance / range } else { f64::INFINITY };
    let sd_of = |level: f64| {
        if observed.sd > 0.0 { (level - observed.mean).abs() / observed.sd } else { f64::INFINITY }
    };
    let bound = ctx.pulse_extrapolation_ranges();
    if range > 0.0 && ranges_of(active_distance) > bound {
        return Err(crate::compile_reason!(
            "positivity_violated",
            "the pulse's active treatment level {active} lies {:.4} observed treatment ranges \
             outside the observed treatment interval [{}, {}] ({:.2} standard deviations from \
             the observed mean), beyond the bound of {bound} ranges: a linear dose response \
             fitted on that interval says nothing about a dose this far from any observation, \
             so the effect is not computed. Next attempt: pulse a level inside the observed \
             interval, collect data that visits the level, or raise the bound with the \
             execution's pulse extrapolation bound if linearity out there is a declared \
             assumption.",
            ranges_of(active_distance),
            observed.min,
            observed.max,
            sd_of(active),
        ));
    }
    let outside: Vec<&str> = [("active", active_distance), ("control", control_distance)]
        .into_iter()
        .filter(|(_, distance)| *distance > 0.0)
        .map(|(role, _)| role)
        .collect();
    Ok(vec![
        Diagnostic::new(
            PULSE_EXTRAPOLATION_CODE,
            DiagnosticKind::Support,
            DiagnosticSeverity::Warning,
            format!(
                "the pulse's {} level lies outside the observed treatment interval [{}, {}]: \
                 its effect is the fitted linear lag response extrapolated to that dose, \
                 conditional on the response staying linear there (active level {active} is \
                 {:.4} observed ranges and {:.2} standard deviations from the observed mean; \
                 the extrapolation bound is {bound} ranges)",
                outside.join(" and "),
                observed.min,
                observed.max,
                ranges_of(active_distance),
                sd_of(active),
            ),
        )
        .with_fields([
            ("observed_min", observed.min.to_string()),
            ("observed_max", observed.max.to_string()),
            ("observed_mean", observed.mean.to_string()),
            ("observed_sd", observed.sd.to_string()),
            ("active_level", active.to_string()),
            ("control_level", control.to_string()),
            ("active_extrapolation_ranges", ranges_of(active_distance).to_string()),
            ("control_extrapolation_ranges", ranges_of(control_distance).to_string()),
            ("active_distance_sd", sd_of(active).to_string()),
            ("control_distance_sd", sd_of(control).to_string()),
            ("extrapolation_bound_ranges", bound.to_string()),
        ]),
    ])
}
