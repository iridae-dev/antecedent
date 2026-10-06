//! Support, overlap, and winsorization diagnostics for continuous responses.
//!
//! The support reports published on the scalar-curve and multivariate paths,
//! the outcome heavy-tail ratio for least-squares nuisances, and the
//! pseudo-outcome winsorization sensitivity probe.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    Diagnostic, DiagnosticKind, DiagnosticScope, DiagnosticSeverity, SupportDiagnostic,
    SupportRegion, SupportReport, SupportStatus,
};
use antecedent_stats::{
    LocalQuadraticWorkspace, QuantileRule, gaussian_local_quadratic_influence_prechecked,
    mad_sigma, median_sorted, quantile_sorted,
};

use super::{
    OUTCOME_TAIL_RATIO_BOUND, OUTCOME_TAIL_RATIO_UNSCALED, PSEUDO_OUTCOME_WINSOR_P,
    PSEUDO_OUTCOME_WINSOR_SHIFT_BOUND,
};
use crate::util::range;

fn sort_finite(values: &[f64]) -> Vec<f64> {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted
}

fn mad_scale(values: &[f64]) -> f64 {
    mad_sigma(values).unwrap_or(0.0)
}

pub(super) fn outcome_tail_ratio(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let sorted = sort_finite(values);
    let center = median_sorted(&sorted);
    let max_dev = values.iter().map(|v| (v - center).abs()).fold(0.0, f64::max);
    let scale = mad_scale(values);
    if scale <= 0.0 {
        if max_dev <= 0.0 { 0.0 } else { OUTCOME_TAIL_RATIO_UNSCALED }
    } else {
        max_dev / scale
    }
}

fn winsorize(values: &[f64], p: f64) -> Vec<f64> {
    let sorted = sort_finite(values);
    // An empty sample has no tail quantiles to clamp to; leave the values as they are.
    if sorted.is_empty() {
        return values.to_vec();
    }
    let lo = quantile_sorted(&sorted, p, QuantileRule::Interpolated);
    let hi = quantile_sorted(&sorted, 1.0 - p, QuantileRule::Interpolated);
    let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
    values.iter().map(|&v| v.clamp(lo, hi)).collect()
}

/// Outcome tail ratio for least-squares Kennedy / Riesz nuisances.
///
/// Does not change [`SupportStatus`]: overlap can be honest while the outcome
/// is outside the estimator's moment conditions. Matrix-cell licensing is
/// likewise untouched.
pub(super) fn push_outcome_tail_diagnostic(support: &mut SupportReport, outcome: &[f64]) {
    let ratio = outcome_tail_ratio(outcome);
    support.diagnostics.push(SupportDiagnostic {
        id: Arc::from("response.outcome_tail_ratio"),
        values: Arc::from([ratio, OUTCOME_TAIL_RATIO_BOUND]),
        detail: Arc::from(
            "max |Y - median| / (1.4826 MAD) of the retained outcome, then the warning bound",
        ),
        scope: DiagnosticScope::Global,
    });
    if ratio > OUTCOME_TAIL_RATIO_BOUND {
        support.warnings.push(Diagnostic::new(
            "response.heavy_tailed_outcome",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            "outcome tail ratio exceeds the bound the least-squares Kennedy nuisances can be trusted on; extreme rows can dominate the estimate",
        ));
    }
}

/// 1%/99% winsorization of φ, then a second local-quadratic pass.
///
/// The estimate itself is unchanged. A large shift means extreme pseudo-outcome
/// rows, not treatment-kernel overlap, are driving the published curve.
pub(super) fn push_pseudo_outcome_winsor_shift(
    support: &mut SupportReport,
    treatments: &[f64],
    pseudo: &[f64],
    grid: &[f64],
    mean: &[f64],
    bandwidth: f64,
    workspace: &mut LocalQuadraticWorkspace,
) {
    if grid.len() != mean.len() || treatments.len() != pseudo.len() || grid.is_empty() {
        return;
    }
    let clipped = winsorize(pseudo, PSEUDO_OUTCOME_WINSOR_P);
    let mut shifts = Vec::with_capacity(grid.len());
    for (&at, &raw) in grid.iter().zip(mean) {
        match gaussian_local_quadratic_influence_prechecked(
            workspace, treatments, &clipped, at, bandwidth,
        ) {
            Ok(fit) => shifts.push((raw - fit.point.value).abs()),
            Err(_) => return,
        }
    }
    if shifts.iter().any(|v| !v.is_finite()) {
        return;
    }
    let max_shift = shifts.iter().copied().fold(0.0, f64::max);
    support.diagnostics.push(SupportDiagnostic {
        id: Arc::from("response.pseudo_outcome_winsor_shift"),
        values: Arc::from(shifts),
        detail: Arc::from(
            "absolute shift of the fitted level after 1%/99% pseudo-outcome winsorization; one value per grid point",
        ),
        scope: DiagnosticScope::PerCoordinate,
    });
    let fitted_range = mean.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - mean.iter().copied().fold(f64::INFINITY, f64::min);
    let scale = fitted_range.abs().max(1e-12);
    if max_shift / scale > PSEUDO_OUTCOME_WINSOR_SHIFT_BOUND {
        support.warnings.push(Diagnostic::new(
            "response.pseudo_outcome_tail_sensitivity",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            "winsorizing the cross-fitted pseudo-outcome at the 1st and 99th percentiles moved the fitted curve; extreme pseudo-outcome rows are driving the estimate",
        ));
    }
}

pub(super) fn support_report(
    points: &[f64],
    observed: &[f64],
    ess: &[f64],
    density: Vec<f64>,
    minimum_ess: f64,
    density_floor_rows: usize,
) -> SupportReport {
    let (minimum, maximum) = range(observed);
    let outside = points.iter().any(|v| *v < minimum || *v > maximum);
    let weak = ess.iter().any(|v| *v < minimum_ess);
    // A clamped conditional density is a positivity failure on the nuisance side:
    // it does not move the requested coordinate outside the observed range, but the
    // curve is then driven by an inverse weight the data never supported.
    let clamped = density_floor_rows > 0;
    let status = if outside {
        SupportStatus::OutsideEmpiricalSupport
    } else if weak || clamped {
        SupportStatus::WeakOverlap
    } else {
        SupportStatus::Supported
    };
    let mut warnings = Vec::new();
    if outside {
        warnings.push(Diagnostic::new(
            "response.outside_empirical_support",
            DiagnosticKind::Support,
            DiagnosticSeverity::Warning,
            "at least one requested response coordinate is outside observed treatment support",
        ));
    } else if weak {
        warnings.push(Diagnostic::new(
            "response.weak_local_overlap",
            DiagnosticKind::Support,
            DiagnosticSeverity::Warning,
            "at least one requested response coordinate has low local effective sample size",
        ));
    }
    if clamped {
        warnings.push(Diagnostic::new(
            "response.conditional_density_floored",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            "at least one row hit the conditional treatment-density floor; the doubly robust weight for those rows is bounded by the floor, not estimated from data",
        ));
    }
    // One label per requested coordinate; `status` above is their worst value.
    let point_status = (points.len() == ess.len() && !points.is_empty()).then(|| {
        let labels: Vec<SupportStatus> = points
            .iter()
            .zip(ess)
            .map(|(point, local_ess)| {
                if *point < minimum || *point > maximum {
                    SupportStatus::OutsideEmpiricalSupport
                } else if *local_ess < minimum_ess || clamped {
                    SupportStatus::WeakOverlap
                } else {
                    SupportStatus::Supported
                }
            })
            .collect();
        Arc::from(labels)
    });
    SupportReport {
        status,
        query_region: SupportRegion {
            minima: Arc::from([points.iter().copied().fold(f64::INFINITY, f64::min)]),
            maxima: Arc::from([points.iter().copied().fold(f64::NEG_INFINITY, f64::max)]),
        },
        diagnostics: vec![
            SupportDiagnostic {
                id: Arc::from("response.local_ess"),
                values: Arc::from(ess.to_vec()),
                detail: Arc::from("Kish effective sample size of Gaussian local weights"),
                scope: DiagnosticScope::PerCoordinate,
            },
            SupportDiagnostic {
                id: Arc::from("response.local_density"),
                values: Arc::from(density),
                detail: Arc::from("Gaussian-kernel marginal treatment-density estimate"),
                scope: DiagnosticScope::PerCoordinate,
            },
            SupportDiagnostic {
                id: Arc::from("response.conditional_density_floor_rows"),
                values: Arc::from([density_floor_rows as f64]),
                detail: Arc::from(
                    "rows whose fitted conditional treatment density hit the positivity floor",
                ),
                scope: DiagnosticScope::Global,
            },
        ],
        warnings,
        point_status,
    }
}

pub(super) fn multivariate_support(
    at: &[f64],
    treatment_matrix: &[f64],
    dimensions: usize,
) -> SupportReport {
    let n = treatment_matrix.len() / dimensions;
    let mut minima = Vec::with_capacity(dimensions);
    let mut maxima = Vec::with_capacity(dimensions);
    let mut outside = false;
    for (j, &point) in at.iter().enumerate() {
        let (lo, hi) = range(&treatment_matrix[j * n..(j + 1) * n]);
        minima.push(lo);
        maxima.push(hi);
        outside |= point < lo || point > hi;
    }
    let status =
        if outside { SupportStatus::OutsideEmpiricalSupport } else { SupportStatus::Extrapolative };
    SupportReport {
        status,
        query_region: SupportRegion {
            minima: Arc::from(at.to_vec()),
            maxima: Arc::from(at.to_vec()),
        },
        diagnostics: vec![SupportDiagnostic {
            id: Arc::from("response.marginal_observed_bounds"),
            values: Arc::from(minima.into_iter().chain(maxima).collect::<Vec<_>>()),
            detail: Arc::from(
                "per-treatment minima followed by maxima; joint support is not established",
            ),
            scope: DiagnosticScope::Global,
        }],
        // One joint query point answers a vector-valued derivative; labels are
        // per response coordinate, so only the summary status applies here.
        point_status: None,
        warnings: {
            let mut warnings = vec![Diagnostic::new(
                "response.plugin_jacobian_model_dependent",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Warning,
                "multivariate derivative uses an additive GAM plug-in and marginal support checks",
            )];
            if outside {
                // The cubic B-spline basis is clamped at its boundary knots, so the
                // fitted surface is constant outside the fitted range and its plug-in
                // derivative is identically zero there. That zero is a property of the
                // basis, not evidence of a flat response, and must not be read as one.
                warnings.push(Diagnostic::new(
                    "response.clamped_basis_derivative",
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Warning,
                    "at least one coordinate is outside the fitted range, where the clamped spline basis makes the plug-in derivative exactly zero by construction",
                ));
            }
            warnings
        },
    }
}

#[cfg(test)]
mod point_status_tests {
    use super::*;

    #[test]
    fn static_curve_labels_each_coordinate_and_summarizes_to_the_worst() {
        let observed = [0.0, 1.0, 2.0, 3.0];
        let report =
            support_report(&[1.0, 2.0, 9.0], &observed, &[50.0, 2.0, 50.0], vec![0.1; 3], 10.0, 0);
        assert_eq!(
            report.point_status.as_deref(),
            Some(
                &[
                    SupportStatus::Supported,
                    SupportStatus::WeakOverlap,
                    SupportStatus::OutsideEmpiricalSupport
                ][..]
            )
        );
        let worst = report.point_status.as_ref().unwrap().iter().max_by_key(|s| s.severity());
        assert_eq!(worst.copied(), Some(report.status));
        // Only per-coordinate diagnostics align with the grid; a row count of
        // floored densities is global and must not read as local support.
        for diagnostic in &report.diagnostics {
            match diagnostic.scope {
                DiagnosticScope::PerCoordinate => {
                    assert_eq!(diagnostic.values.len(), 3, "{}", diagnostic.id);
                }
                DiagnosticScope::Global => {
                    assert_eq!(&*diagnostic.id, "response.conditional_density_floor_rows");
                }
                DiagnosticScope::Inapplicable => panic!("{}", diagnostic.id),
            }
        }
        let floored = support_report(&[1.0], &observed, &[50.0], vec![0.1], 10.0, 3);
        assert_eq!(floored.point_status.as_deref(), Some(&[SupportStatus::WeakOverlap][..]));
        assert_eq!(floored.status, SupportStatus::WeakOverlap);
    }
}
