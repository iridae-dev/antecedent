//! Support classification of dose-horizon cells.

use std::sync::Arc;

use antecedent_core::{
    Diagnostic, DiagnosticKind, DiagnosticScope, DiagnosticSeverity, SupportDiagnostic,
    SupportRegion, SupportReport, SupportStatus, TemporalResponseSpec,
};

use super::surface::HorizonTreatmentRange;

fn cell_against_range(dose: f64, observed_min: f64, observed_max: f64) -> SupportStatus {
    if !observed_min.is_finite() || !observed_max.is_finite() {
        SupportStatus::Extrapolative
    } else if dose < observed_min || dose > observed_max {
        SupportStatus::OutsideEmpiricalSupport
    } else {
        SupportStatus::Supported
    }
}

/// Surface summary over the same geometry as the estimate.
///
/// All cells supported → [`SupportStatus::Supported`]. Mixed supported /
/// unsupported cells → [`SupportStatus::Extrapolative`] (partially
/// extrapolative). No cell supported → [`SupportStatus::OutsideEmpiricalSupport`],
/// unless every cell was unassessable (non-finite range), which stays
/// extrapolative.
fn summarize_surface_support(points: &[SupportStatus]) -> SupportStatus {
    let n = points.len();
    let n_supported = points.iter().filter(|status| **status == SupportStatus::Supported).count();
    if n == 0 {
        return SupportStatus::Extrapolative;
    }
    if n_supported == n {
        return SupportStatus::Supported;
    }
    if n_supported > 0 {
        return SupportStatus::Extrapolative;
    }
    if points.iter().any(|status| *status == SupportStatus::OutsideEmpiricalSupport) {
        SupportStatus::OutsideEmpiricalSupport
    } else {
        SupportStatus::Extrapolative
    }
}

pub(super) fn mean_curve_support(
    doses: &[f64],
    temporal: &TemporalResponseSpec,
    horizon_ranges: &[HorizonTreatmentRange],
) -> SupportReport {
    let mut point_status = Vec::with_capacity(doses.len().saturating_mul(horizon_ranges.len()));
    for &dose in doses {
        for &(lo, hi) in horizon_ranges {
            point_status.push(cell_against_range(dose, lo, hi));
        }
    }
    assemble_temporal_support(doses, temporal, horizon_ranges, point_status)
}

pub(super) fn intervention_support(
    eval_levels: &[f64],
    level: Option<f64>,
    shift: f64,
    temporal: &TemporalResponseSpec,
    horizon_ranges: &[HorizonTreatmentRange],
) -> SupportReport {
    let point_status: Vec<SupportStatus> = eval_levels
        .iter()
        .zip(horizon_ranges.iter())
        .map(|(&dose, &(lo, hi))| {
            if level.is_some() {
                cell_against_range(dose, lo, hi)
            } else {
                // A shift intervention evaluates the factual treatment law at
                // A + delta, not just at E[A] + delta. The latter can sit inside
                // [min(A), max(A)] while a large fraction of shifted rows are
                // outside it. Classify the whole shifted interval instead.
                shifted_range_against_range(shift, lo, hi)
            }
        })
        .collect();
    let shifted_extrapolation = level.is_none()
        && point_status.iter().any(|status| *status == SupportStatus::Extrapolative);
    let mut report = assemble_temporal_support(eval_levels, temporal, horizon_ranges, point_status);
    // InterventionResponse has one result cell per horizon. Reusing the mean
    // surface assembler used to claim an H x H dose-by-horizon layout even
    // though both the estimate and point_status contain only H cells.
    if let Some(layout) = report
        .diagnostics
        .iter_mut()
        .find(|diagnostic| diagnostic.id.as_ref() == "response.temporal.dose_horizon_layout")
    {
        layout.id = Arc::from("response.temporal.intervention_horizon_layout");
        layout.values = Arc::from([temporal.horizons.len() as f64]);
        layout.detail = Arc::from("one intervention-response cell per requested horizon");
    }
    if level.is_none() {
        let mut shifted_ranges = Vec::with_capacity(horizon_ranges.len().saturating_mul(2));
        for &(lo, hi) in horizon_ranges {
            shifted_ranges.push(lo + shift);
            shifted_ranges.push(hi + shift);
        }
        let shifted_min = shifted_ranges.iter().step_by(2).copied().fold(f64::INFINITY, f64::min);
        let shifted_max =
            shifted_ranges.iter().skip(1).step_by(2).copied().fold(f64::NEG_INFINITY, f64::max);
        report.query_region.minima =
            Arc::from([shifted_min, f64::from(temporal.horizons.first().copied().unwrap_or(1))]);
        report.query_region.maxima =
            Arc::from([shifted_max, f64::from(temporal.horizons.last().copied().unwrap_or(1))]);
        report.diagnostics.push(SupportDiagnostic {
            id: Arc::from("response.temporal.shifted_treatment_range"),
            values: Arc::from(shifted_ranges),
            detail: Arc::from(
                "per-horizon support requested by A + delta as [min_0+delta, max_0+delta, …]",
            ),
            scope: DiagnosticScope::Global,
        });
    }
    if shifted_extrapolation {
        report.warnings.push(Diagnostic::new(
            "response.temporal.shift_distribution_extrapolative",
            DiagnosticKind::Support,
            DiagnosticSeverity::Warning,
            "the shifted treatment distribution extends beyond at least one horizon's observed treatment range",
        ));
    }
    report
}

fn shifted_range_against_range(shift: f64, observed_min: f64, observed_max: f64) -> SupportStatus {
    if !shift.is_finite() || !observed_min.is_finite() || !observed_max.is_finite() {
        return SupportStatus::Extrapolative;
    }
    if shift == 0.0 {
        return SupportStatus::Supported;
    }
    let shifted_min = observed_min + shift;
    let shifted_max = observed_max + shift;
    if !shifted_min.is_finite() || !shifted_max.is_finite() {
        SupportStatus::Extrapolative
    } else if shifted_max < observed_min || shifted_min > observed_max {
        SupportStatus::OutsideEmpiricalSupport
    } else {
        SupportStatus::Extrapolative
    }
}

fn assemble_temporal_support(
    doses: &[f64],
    temporal: &TemporalResponseSpec,
    horizon_ranges: &[HorizonTreatmentRange],
    point_status: Vec<SupportStatus>,
) -> SupportReport {
    let status = summarize_surface_support(&point_status);
    let mixed = point_status.iter().any(|s| *s == SupportStatus::Supported)
        && point_status.iter().any(|s| *s != SupportStatus::Supported);
    let mut range_values = Vec::with_capacity(horizon_ranges.len().saturating_mul(2));
    for &(lo, hi) in horizon_ranges {
        range_values.push(lo);
        range_values.push(hi);
    }
    let mut warnings = Vec::new();
    if mixed {
        warnings.push(Diagnostic::new(
            "response.temporal.partial_horizon_support",
            DiagnosticKind::Support,
            DiagnosticSeverity::Warning,
            "some requested (dose, horizon) cells sit outside that horizon's lag-aligned \
             treatment range; inspect support.point_status",
        ));
    } else if status == SupportStatus::OutsideEmpiricalSupport {
        warnings.push(Diagnostic::new(
            "response.outside_empirical_support",
            DiagnosticKind::Support,
            DiagnosticSeverity::Warning,
            "no requested (dose, horizon) cell sits inside that horizon's lag-aligned \
             treatment range",
        ));
    }
    SupportReport {
        status,
        query_region: SupportRegion {
            minima: Arc::from(vec![
                doses.iter().copied().fold(f64::INFINITY, f64::min),
                f64::from(temporal.horizons.first().copied().unwrap_or(1)),
            ]),
            maxima: Arc::from(vec![
                doses.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                f64::from(temporal.horizons.last().copied().unwrap_or(1)),
            ]),
        },
        diagnostics: vec![
            SupportDiagnostic {
                id: Arc::from("response.temporal.dose_horizon_layout"),
                values: Arc::from(vec![doses.len() as f64, temporal.horizons.len() as f64]),
                detail: Arc::from(
                    "row-major dose × horizon surface: value[d * n_horizons + h]; \
                     grid stores [dose_d, horizon_h] pairs",
                ),
                scope: DiagnosticScope::Global,
            },
            SupportDiagnostic {
                id: Arc::from("response.temporal.horizon_treatment_range"),
                values: Arc::from(range_values),
                detail: Arc::from(
                    "per-horizon lag-aligned treatment range as [min_0, max_0, min_1, max_1, …]",
                ),
                scope: DiagnosticScope::Global,
            },
        ],
        warnings,
        point_status: Some(Arc::from(point_status)),
    }
}
