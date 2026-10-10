//! 2.3 B1 (X4): one complete response-grid row, a named functional of a
//! randomized-dose response curve with per-dose support.
//!
//! # The row
//!
//! One statement per axis the B1 contract requires.
//!
//! * **Estimand.** For a continuous dose `D` and outcome `Y`, the dose response
//!   `m(d) = E[Y(d)]`. Exactly one *named functional* is requested per call:
//!   [`DoseFunctional::Level`] (`m(d)` at each grid dose), [`DoseFunctional::Derivative`]
//!   (`m'(d)` at each grid dose) or [`DoseFunctional::Contrast`] (`m(to) - m(from)`
//!   between two named doses). The fitted object is the Gaussian-kernel local
//!   quadratic smoother at the declared fixed bandwidth `h`; the functional is that
//!   smoother's level, first derivative or difference of levels. The smoothing bias
//!   `m_h - m` is not estimated and not included in any interval.
//!   [`estimate_dose_grid_quadratic_mean`] adds a caller-attested exact quadratic mean
//!   premise: polynomial reproduction derives zero smoothing bias at fixed bandwidth.
//!   Its sampling calibration remains unmeasured.
//! * **Graph certificate.** [`DoseDesign::RandomizedDose`]: the dose is assigned
//!   independently of the potential-outcome process (no parent of the dose is an
//!   ancestor of the outcome and no backdoor path is open), so the empty adjustment
//!   set is valid and `E[Y(d)] = E[Y | D = d]`. This is a caller attestation about
//!   the design; no graph is re-derived here. [`DoseDesign::Observational`] is
//!   refused (`dose_grid.graph_not_certified`): this row does not adjust.
//! * **Evidence.** Aligned finite `(dose, outcome)` rows. No external model or law.
//! * **Nuisance fits.** None beyond the fixed-bandwidth local quadratic itself. The
//!   bandwidth is declared by the caller and must lie in a caller-declared range; it
//!   is never data-selected here, so no selection variability is hidden.
//! * **Dependence.** Independent rows. Clustered, serial or linked rows are outside
//!   this row; the 2.2 clustered DML AIPW cell does not license them.
//! * **Support.** Per requested dose: `OutsideEmpiricalSupport` when the dose lies
//!   outside the observed dose range, `WeakOverlap` when the Kish effective sample
//!   size of the Gaussian local weights is below the declared minimum, else
//!   `Supported`. A dose that is not `Supported` is *refused*, never extrapolated
//!   ([`dose_support_report`] labels without refusing).
//! * **Numerical error.** [`NumericalReport`] reports the largest absolute sum of
//!   the linearized level influences over the fits. By the weighted normal
//!   equations that sum is exactly zero, so its magnitude is the round-off of the
//!   local solve. It is a measured residual, not a bound on estimation error.
//! * **Inference claim.** A pointwise normal interval (heteroskedasticity-robust
//!   sandwich standard error from the linearized influences) at one dose or for one
//!   contrast, with calibration [`IntervalCalibration::Unmeasured`]: no coverage
//!   claim is made and the public route stays closed until calibration is measured
//!   at the release cut. Level, derivative and contrast claims are separate typed
//!   fields, enabled separately through [`ClaimSet`]. A simultaneous band is a
//!   different claim: requesting it is refused (`dose_grid.simultaneous_band_closed`)
//!   and [`DoseGridRow::simultaneous_band`] is always `None`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    Diagnostic, DiagnosticKind, DiagnosticScope, DiagnosticSeverity, SupportDiagnostic,
    SupportRegion, SupportReport, SupportStatus,
};
use antecedent_stats::{
    LocalPolynomialInfluence, LocalQuadraticWorkspace, gaussian_local_quadratic_influence_with,
};

use crate::error::{EstimationError, RefusalFields};

/// Normal 97.5% quantile for the nominal 95% pointwise interval.
const Z_95: f64 = 1.959_963_984_540_054;
/// Fewest effective rows a local quadratic can be asked to stand on.
const MINIMUM_ESS_FLOOR: f64 = 3.0;

/// What licenses reading the fitted conditional mean as the dose response.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DoseDesign {
    /// The dose is randomized (independent of the potential-outcome process).
    RandomizedDose,
    /// The dose is observational; this row does not adjust and refuses it.
    Observational,
}

/// The one named functional requested per call.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DoseFunctional {
    /// The dose response `m(d)` at each grid dose.
    Level,
    /// The dose-response derivative `m'(d)` at each grid dose.
    Derivative,
    /// The difference `m(to) - m(from)` between two named doses.
    Contrast {
        /// Reference dose.
        from: f64,
        /// Comparison dose.
        to: f64,
    },
}

/// Which inference claims the caller declares. Each is separate; none implies another.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ClaimSet {
    /// A pointwise claim about `m(d)` (levels and contrasts).
    pub pointwise_level: bool,
    /// A pointwise claim about `m'(d)`.
    pub derivative: bool,
    /// A simultaneous band over the grid. Always refused: the route is closed.
    pub simultaneous_band: bool,
}

/// Calibration status of an interval. Only `Unmeasured` exists in 2.3 B1.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IntervalCalibration {
    /// Coverage has not been measured; no coverage claim is made.
    Unmeasured,
}

/// A pointwise normal interval for one functional value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointwiseInterval {
    /// Lower endpoint.
    pub lower: f64,
    /// Upper endpoint.
    pub upper: f64,
    /// Nominal level the normal quantile was taken at; not a coverage claim.
    pub nominal_level: f64,
    /// Calibration status (always unmeasured here).
    pub calibration: IntervalCalibration,
    /// Whether smoothing bias is accounted for (only under the exact-quadratic premise).
    pub smoothing_bias_included: bool,
}

/// A pointwise level `m(d)` at one dose.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DoseLevel {
    /// Grid dose.
    pub dose: f64,
    /// Support label of this dose.
    pub support: SupportStatus,
    /// Local-quadratic level.
    pub level: f64,
    /// Robust sandwich standard error.
    pub standard_error: f64,
    /// Kish effective sample size of the local weights.
    pub local_ess: f64,
    /// Pointwise interval.
    pub interval: PointwiseInterval,
}

/// A pointwise derivative `m'(d)` at one dose. Never filled by a level claim.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DoseDerivative {
    /// Grid dose.
    pub dose: f64,
    /// Support label of this dose.
    pub support: SupportStatus,
    /// Local-quadratic first derivative.
    pub derivative: f64,
    /// Robust sandwich standard error of the derivative.
    pub standard_error: f64,
    /// Kish effective sample size of the local weights.
    pub local_ess: f64,
    /// Pointwise interval.
    pub interval: PointwiseInterval,
}

/// A named contrast `m(to) - m(from)` with its own covariance-aware standard error.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DoseContrast {
    /// Reference dose.
    pub from: f64,
    /// Comparison dose.
    pub to: f64,
    /// Support label of the reference dose.
    pub from_support: SupportStatus,
    /// Support label of the comparison dose.
    pub to_support: SupportStatus,
    /// Contrast estimate.
    pub estimate: f64,
    /// Standard error from the difference of the two linearized influences, so the
    /// overlap of the two local windows is accounted for.
    pub standard_error: f64,
    /// Pointwise interval.
    pub interval: PointwiseInterval,
}

/// Measured numerical residual of the local solves.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NumericalReport {
    /// Local fits performed.
    pub fits: usize,
    /// Largest `|sum_i influence_i|`; zero in exact arithmetic.
    pub max_abs_influence_sum: f64,
}

/// A simultaneous band. Uninhabited in effect: the route is closed, so no value is built.
#[derive(Clone, Debug, PartialEq)]
pub struct SimultaneousBand {
    /// Lower envelope per grid dose.
    pub lower: Vec<f64>,
    /// Upper envelope per grid dose.
    pub upper: Vec<f64>,
}

/// Inputs of one row evaluation.
#[derive(Clone, Copy, Debug)]
pub struct DoseGridRequest<'a> {
    /// Observed doses.
    pub dose: &'a [f64],
    /// Observed outcomes, aligned with `dose`.
    pub outcome: &'a [f64],
    /// Grid doses for [`DoseFunctional::Level`] and [`DoseFunctional::Derivative`];
    /// ignored for a contrast, which names its own two doses.
    pub grid: &'a [f64],
    /// Fixed Gaussian bandwidth.
    pub bandwidth: f64,
    /// Declared inclusive range the bandwidth must lie in.
    pub bandwidth_range: (f64, f64),
    /// Smallest admissible Kish effective sample size at any requested dose.
    pub minimum_local_ess: f64,
    /// Named functional.
    pub functional: DoseFunctional,
    /// Declared claims.
    pub claims: ClaimSet,
    /// Design attestation.
    pub design: DoseDesign,
}

/// Result of one row evaluation. Exactly one of `levels`, `derivatives`, `contrast`
/// is populated, matching the requested functional.
#[derive(Clone, Debug, PartialEq)]
pub struct DoseGridRow {
    /// Requested functional.
    pub functional: DoseFunctional,
    /// Design attestation the result rests on.
    pub design: DoseDesign,
    /// Bandwidth used.
    pub bandwidth: f64,
    /// Rows used.
    pub n_rows: usize,
    /// Pointwise levels (level functional only).
    pub levels: Vec<DoseLevel>,
    /// Pointwise derivatives (derivative functional only).
    pub derivatives: Vec<DoseDerivative>,
    /// Contrast (contrast functional only).
    pub contrast: Option<DoseContrast>,
    /// Simultaneous band; always `None` (closed).
    pub simultaneous_band: Option<SimultaneousBand>,
    /// Per-dose support report.
    pub support: SupportReport,
    /// Numerical residual of the local solves.
    pub numerical: NumericalReport,
}

impl DoseGridRow {
    /// Bind one actual pointwise internal candidate under the exact-quadratic premise.
    /// Returns `None` for absent points or the unrestricted bias-uncorrected row.
    /// This records the declared model/design, not proof of randomization or coverage.
    #[cfg(feature = "calibration-internal")]
    #[must_use]
    pub fn quadratic_calibration_basis(
        &self,
        index: usize,
    ) -> Option<antecedent_core::CalibrationBasis> {
        use std::sync::Arc;
        let (interval, coordinate) = match self.functional {
            DoseFunctional::Level => {
                let point = self.levels.get(index)?;
                (point.interval, format!("level:{:016x}", point.dose.to_bits()))
            }
            DoseFunctional::Derivative => {
                let point = self.derivatives.get(index)?;
                (point.interval, format!("derivative:{:016x}", point.dose.to_bits()))
            }
            DoseFunctional::Contrast { from, to } if index == 0 => (
                self.contrast.as_ref()?.interval,
                format!("contrast:{:016x}:{:016x}", from.to_bits(), to.to_bits()),
            ),
            DoseFunctional::Contrast { .. } => return None,
        };
        if self.design != DoseDesign::RandomizedDose || !interval.smoothing_bias_included {
            return None;
        }
        let functional =
            format!("{coordinate}.bandwidth:{:016x}.exact_quadratic", self.bandwidth.to_bits());
        Some(antecedent_core::CalibrationBasis::new(
            [
                "DoseResponse",
                "Dag",
                "fixed",
                "tabular",
                "Frequentist",
                "dose_grid_quadratic",
                "analytic_se",
                "local_quadratic_HC0",
                "iid",
                "",
                &functional,
            ]
            .map(Arc::from),
            interval.nominal_level,
            Arc::from("point"),
            u64::try_from(self.n_rows).expect("bounded row count"),
            None,
            None,
            0.,
        ))
    }
}

fn refuse(code: &'static str, detail: &'static str, message: &str) -> EstimationError {
    EstimationError::refused_with_fields(
        code,
        format!("{detail}: {message}"),
        RefusalFields {
            stage: Some("dose_grid".to_owned()),
            reason: Some(detail.to_owned()),
            ..RefusalFields::default()
        },
    )
}

fn evaluation_points(request: &DoseGridRequest<'_>) -> Vec<f64> {
    match request.functional {
        DoseFunctional::Contrast { from, to } => vec![from, to],
        DoseFunctional::Level | DoseFunctional::Derivative => request.grid.to_vec(),
    }
}

fn validate_claims(request: &DoseGridRequest<'_>) -> Result<(), EstimationError> {
    if request.design == DoseDesign::Observational {
        return Err(refuse(
            antecedent_core::reason_code!("effect_not_identified"),
            "dose_grid.graph_not_certified",
            "this row does not adjust; it requires a randomized dose",
        ));
    }
    if request.claims.simultaneous_band {
        return Err(refuse(
            antecedent_core::reason_code!("route_not_supported"),
            "dose_grid.simultaneous_band_closed",
            "a simultaneous band is a separate claim whose calibration is unmeasured",
        ));
    }
    match request.functional {
        DoseFunctional::Derivative if !request.claims.derivative => Err(refuse(
            antecedent_core::reason_code!("option_not_applicable"),
            "dose_grid.derivative_without_claim",
            "a derivative was requested without declaring a derivative claim",
        )),
        DoseFunctional::Level | DoseFunctional::Contrast { .. }
            if !request.claims.pointwise_level =>
        {
            Err(refuse(
                antecedent_core::reason_code!("option_not_applicable"),
                "dose_grid.level_without_claim",
                "a level or contrast was requested without declaring a pointwise level claim",
            ))
        }
        _ => Ok(()),
    }
}

fn validate_inputs(request: &DoseGridRequest<'_>, points: &[f64]) -> Result<(), EstimationError> {
    let n = request.dose.len();
    let invalid = |message: &str| {
        refuse(
            antecedent_core::reason_code!("invalid_argument"),
            "dose_grid.invalid_request",
            message,
        )
    };
    if n < 3
        || request.outcome.len() != n
        || request.dose.iter().chain(request.outcome).any(|v| !v.is_finite())
    {
        return Err(invalid("rows must be aligned, finite and at least three"));
    }
    if points.is_empty() || points.iter().any(|v| !v.is_finite()) {
        return Err(invalid("the requested doses must be non-empty and finite"));
    }
    if !request.minimum_local_ess.is_finite() || request.minimum_local_ess < MINIMUM_ESS_FLOOR {
        return Err(invalid(
            "the minimum local effective sample size must be finite and at least 3",
        ));
    }
    let (low, high) = request.bandwidth_range;
    if !(low.is_finite() && high.is_finite() && low > 0.0 && low <= high) {
        return Err(invalid("the declared bandwidth range must be finite with 0 < low <= high"));
    }
    if !request.bandwidth.is_finite() || request.bandwidth < low || request.bandwidth > high {
        return Err(refuse(
            antecedent_core::reason_code!("invalid_argument"),
            "dose_grid.bandwidth_outside_range",
            "the bandwidth is not inside the declared bandwidth range",
        ));
    }
    Ok(())
}

fn kish_ess(dose: &[f64], at: f64, bandwidth: f64) -> f64 {
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    for &d in dose {
        let z = (d - at) / bandwidth;
        let w = (-0.5 * z * z).exp();
        sum += w;
        sum_sq += w * w;
    }
    if sum_sq > f64::EPSILON { sum * sum / sum_sq } else { 0.0 }
}

fn observed_range(dose: &[f64]) -> (f64, f64) {
    dose.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &d| (lo.min(d), hi.max(d)))
}

fn label_points(
    dose: &[f64],
    points: &[f64],
    bandwidth: f64,
    minimum_ess: f64,
) -> (Vec<SupportStatus>, Vec<f64>) {
    let (minimum, maximum) = observed_range(dose);
    let ess: Vec<f64> = points.iter().map(|&at| kish_ess(dose, at, bandwidth)).collect();
    let labels = points
        .iter()
        .zip(&ess)
        .map(|(&at, &local)| {
            if at < minimum || at > maximum {
                SupportStatus::OutsideEmpiricalSupport
            } else if local < minimum_ess {
                SupportStatus::WeakOverlap
            } else {
                SupportStatus::Supported
            }
        })
        .collect();
    (labels, ess)
}

fn build_report(
    dose: &[f64],
    points: &[f64],
    labels: &[SupportStatus],
    ess: Vec<f64>,
) -> SupportReport {
    let (minimum, maximum) = observed_range(dose);
    let status = labels
        .iter()
        .copied()
        .max_by_key(|label| label.severity())
        .unwrap_or(SupportStatus::MissingEvidence);
    let mut warnings = Vec::new();
    if labels.contains(&SupportStatus::OutsideEmpiricalSupport) {
        warnings.push(Diagnostic::new(
            "response.outside_empirical_support",
            DiagnosticKind::Support,
            DiagnosticSeverity::Warning,
            "at least one requested dose is outside observed dose support",
        ));
    }
    if labels.contains(&SupportStatus::WeakOverlap) {
        warnings.push(Diagnostic::new(
            "response.weak_local_overlap",
            DiagnosticKind::Support,
            DiagnosticSeverity::Warning,
            "at least one requested dose has low local effective sample size",
        ));
    }
    SupportReport {
        status,
        query_region: SupportRegion {
            minima: Arc::from([points.iter().copied().fold(f64::INFINITY, f64::min)]),
            maxima: Arc::from([points.iter().copied().fold(f64::NEG_INFINITY, f64::max)]),
        },
        diagnostics: vec![
            SupportDiagnostic {
                id: Arc::from("response.local_ess"),
                values: Arc::from(ess),
                detail: Arc::from("Kish effective sample size of Gaussian local weights"),
                scope: DiagnosticScope::PerCoordinate,
            },
            SupportDiagnostic {
                id: Arc::from("response.marginal_observed_bounds"),
                values: Arc::from([minimum, maximum]),
                detail: Arc::from("observed dose minimum then maximum"),
                scope: DiagnosticScope::Global,
            },
        ],
        warnings,
        point_status: Some(Arc::from(labels.to_vec())),
    }
}

/// Label every requested dose without refusing and return the support report.
///
/// One label per dose in `points`, in order; the report's `status` is their worst
/// label. This is the labelling half of the per-dose support contract; the row
/// itself refuses any dose that is not `Supported`.
///
/// # Errors
///
/// Fewer than three finite aligned rows, empty or non-finite `points`, or a
/// non-positive bandwidth, or a minimum local ESS below the row's floor of 3.
pub fn dose_support_report(
    dose: &[f64],
    points: &[f64],
    bandwidth: f64,
    minimum_local_ess: f64,
) -> Result<SupportReport, EstimationError> {
    let invalid = |message: &str| {
        refuse(
            antecedent_core::reason_code!("invalid_argument"),
            "dose_grid.invalid_request",
            message,
        )
    };
    if dose.len() < 3 || dose.iter().any(|v| !v.is_finite()) {
        return Err(invalid("doses must be finite and at least three"));
    }
    if points.is_empty() || points.iter().any(|v| !v.is_finite()) {
        return Err(invalid("the requested doses must be non-empty and finite"));
    }
    if !bandwidth.is_finite()
        || bandwidth <= 0.0
        || !minimum_local_ess.is_finite()
        || minimum_local_ess < MINIMUM_ESS_FLOOR
    {
        return Err(invalid(
            "bandwidth must be positive and minimum effective sample size at least 3",
        ));
    }
    let (labels, ess) = label_points(dose, points, bandwidth, minimum_local_ess);
    Ok(build_report(dose, points, &labels, ess))
}

fn refuse_unsupported(points: &[f64], labels: &[SupportStatus]) -> Result<(), EstimationError> {
    for (&at, &label) in points.iter().zip(labels) {
        match label {
            SupportStatus::Supported => {}
            SupportStatus::WeakOverlap => {
                return Err(refuse(
                    antecedent_core::reason_code!("cell_not_licensed"),
                    "dose_grid.insufficient_local_weight",
                    &format!(
                        "dose {at} has local effective sample size below the declared minimum"
                    ),
                ));
            }
            _ => {
                return Err(refuse(
                    antecedent_core::reason_code!("cell_not_licensed"),
                    "dose_grid.unsupported_dose",
                    &format!("dose {at} is outside the observed dose support"),
                ));
            }
        }
    }
    Ok(())
}

fn interval(estimate: f64, standard_error: f64) -> PointwiseInterval {
    PointwiseInterval {
        lower: estimate - Z_95 * standard_error,
        upper: estimate + Z_95 * standard_error,
        nominal_level: 0.95,
        calibration: IntervalCalibration::Unmeasured,
        smoothing_bias_included: false,
    }
}

fn fit_at(
    workspace: &mut LocalQuadraticWorkspace,
    request: &DoseGridRequest<'_>,
    at: f64,
) -> Result<LocalPolynomialInfluence, EstimationError> {
    gaussian_local_quadratic_influence_with(
        workspace,
        request.dose,
        request.outcome,
        at,
        request.bandwidth,
    )
    .map_err(|error| {
        refuse(
            antecedent_core::reason_code!("cell_not_licensed"),
            "dose_grid.insufficient_local_weight",
            &format!("the local quadratic could not be fitted at dose {at}: {error}"),
        )
    })
}

fn contrast_row(
    request: &DoseGridRequest<'_>,
    from: f64,
    to: f64,
    labels: &[SupportStatus],
    numerical: &mut NumericalReport,
) -> Result<DoseContrast, EstimationError> {
    let mut workspace = LocalQuadraticWorkspace::default();
    let fit_from = fit_at(&mut workspace, request, from)?;
    let fit_to = fit_at(&mut workspace, request, to)?;
    for fit in [&fit_from, &fit_to] {
        numerical.fits += 1;
        numerical.max_abs_influence_sum =
            numerical.max_abs_influence_sum.max(fit.influences.iter().sum::<f64>().abs());
    }
    let estimate = fit_to.point.value - fit_from.point.value;
    let standard_error = fit_to
        .influences
        .iter()
        .zip(&fit_from.influences)
        .map(|(upper, lower)| (upper - lower).powi(2))
        .sum::<f64>()
        .sqrt();
    Ok(DoseContrast {
        from,
        to,
        from_support: labels[0],
        to_support: labels[1],
        estimate,
        standard_error,
        interval: interval(estimate, standard_error),
    })
}

fn grid_rows(
    request: &DoseGridRequest<'_>,
    labels: &[SupportStatus],
    ess: &[f64],
    numerical: &mut NumericalReport,
) -> Result<(Vec<DoseLevel>, Vec<DoseDerivative>), EstimationError> {
    let mut workspace = LocalQuadraticWorkspace::default();
    let mut levels = Vec::new();
    let mut derivatives = Vec::new();
    for (index, &at) in request.grid.iter().enumerate() {
        let fit = fit_at(&mut workspace, request, at)?;
        numerical.fits += 1;
        numerical.max_abs_influence_sum =
            numerical.max_abs_influence_sum.max(fit.influences.iter().sum::<f64>().abs());
        if request.functional == DoseFunctional::Derivative {
            let standard_error = fit.robust_first_derivative_standard_error;
            derivatives.push(DoseDerivative {
                dose: at,
                support: labels[index],
                derivative: fit.point.first_derivative,
                standard_error,
                local_ess: ess[index],
                interval: interval(fit.point.first_derivative, standard_error),
            });
        } else {
            let standard_error = fit.robust_standard_error;
            levels.push(DoseLevel {
                dose: at,
                support: labels[index],
                level: fit.point.value,
                standard_error,
                local_ess: ess[index],
                interval: interval(fit.point.value, standard_error),
            });
        }
    }
    Ok((levels, derivatives))
}

/// Evaluate the requested named functional of the randomized-dose response curve.
///
/// See the module documentation for the estimand, certificate, nuisance, dependence,
/// support, numerical-error and inference statements. Order of refusal: design,
/// simultaneous band, claim declaration, input shape, bandwidth range, per-dose
/// support, then the local fits.
///
/// # Errors
///
/// A typed refusal whose detail is one of `dose_grid.graph_not_certified`,
/// `dose_grid.simultaneous_band_closed`, `dose_grid.derivative_without_claim`,
/// `dose_grid.level_without_claim`, `dose_grid.invalid_request`,
/// `dose_grid.bandwidth_outside_range`, `dose_grid.unsupported_dose` or
/// `dose_grid.insufficient_local_weight`.
pub fn estimate_dose_grid_functional(
    request: &DoseGridRequest<'_>,
) -> Result<DoseGridRow, EstimationError> {
    validate_claims(request)?;
    let points = evaluation_points(request);
    validate_inputs(request, &points)?;
    let (labels, ess) =
        label_points(request.dose, &points, request.bandwidth, request.minimum_local_ess);
    refuse_unsupported(&points, &labels)?;
    let support = build_report(request.dose, &points, &labels, ess.clone());
    let mut numerical = NumericalReport { fits: 0, max_abs_influence_sum: 0.0 };
    let (levels, derivatives, contrast) = match request.functional {
        DoseFunctional::Contrast { from, to } => {
            let row = contrast_row(request, from, to, &labels, &mut numerical)?;
            (Vec::new(), Vec::new(), Some(row))
        }
        DoseFunctional::Level | DoseFunctional::Derivative => {
            let (levels, derivatives) = grid_rows(request, &labels, &ess, &mut numerical)?;
            (levels, derivatives, None)
        }
    };
    Ok(DoseGridRow {
        functional: request.functional,
        design: request.design,
        bandwidth: request.bandwidth,
        n_rows: request.dose.len(),
        levels,
        derivatives,
        contrast,
        simultaneous_band: None,
        support,
        numerical,
    })
}

/// Evaluate the same local-quadratic row under a declared exact quadratic mean model.
///
/// For `E[Y|D=d] = beta0 + beta1*d + beta2*d²`, every nonsingular weighted
/// quadratic solve reproduces that conditional mean exactly, at any fixed bandwidth.
/// Its derivative and differences of levels also reproduce their targets. Conditional
/// on the observed doses the smoothing bias is therefore zero; the residual sandwich
/// still estimates sampling variance and its normal approximation awaits calibration.
/// The caller supplies the model premise: observed fit residuals cannot prove it.
///
/// # Errors
/// The same design, claims, support and numerical refusals as the unrestricted row.
pub fn estimate_dose_grid_quadratic_mean(
    request: &DoseGridRequest<'_>,
) -> Result<DoseGridRow, EstimationError> {
    let mut row = estimate_dose_grid_functional(request)?;
    for point in &mut row.levels {
        point.interval.smoothing_bias_included = true;
    }
    for point in &mut row.derivatives {
        point.interval.smoothing_bias_included = true;
    }
    if let Some(contrast) = &mut row.contrast {
        contrast.interval.smoothing_bias_included = true;
    }
    Ok(row)
}
