//! Fixed-basis plug-in gradient and sandwich uncertainty.

use std::sync::Arc;

#[cfg(feature = "calibration-internal")]
use antecedent_core::{DerivativeScale, ResponseFunctional, ResponseUncertainty, ResponseValue};
use antecedent_core::{
    Diagnostic, DiagnosticKind, DiagnosticSeverity, Intervention, MAX_NONPARAMETRIC_RESPONSE_DIM,
    VariableId,
};
use antecedent_data::TabularData;
#[cfg(feature = "calibration-internal")]
use antecedent_stats::normal_ppf;
use antecedent_stats::{DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace, invert_square};

use crate::EstimationError;

use super::{
    ContinuousResponseEstimator, DIRECTIONAL_TREATMENT_LIMIT_REMEDY, DiscreteAtom,
    JACOBIAN_TREATMENT_LIMIT_REMEDY, policy_support, sample::CompleteSample,
    sample::read_shared_complete_samples,
};

impl ContinuousResponseEstimator {
    /// The plug-in level and treatment gradient of every outcome at `at`, from one
    /// unpenalized-treatment additive-GAM target fit per outcome on the shared
    /// complete-case rows, with each gradient coordinate's coefficient-sandwich
    /// influence column when `with_influence`.
    pub(super) fn plugin_gradient_run(
        &self,
        data: &TabularData,
        outcomes: &[VariableId],
        treatments: &[VariableId],
        at: &[f64],
        limit: PluginGradientKind,
        with_influence: bool,
    ) -> Result<PluginGradientRun, EstimationError> {
        if treatments.len() > MAX_NONPARAMETRIC_RESPONSE_DIM {
            let (message, remedy) = limit.refusal();
            return Err(EstimationError::unsupported_with_remedy(message, remedy));
        }
        let samples =
            read_shared_complete_samples(data, outcomes, treatments, &self.adjustment_set)?;
        let all_treatments =
            samples.first().map(|sample| sample.treatment_matrix.clone()).unwrap_or_default();
        let mut run = PluginGradientRun {
            levels: Vec::with_capacity(samples.len()),
            gradients: Vec::with_capacity(samples.len()),
            influence: Vec::new(),
            all_treatments,
            n: samples.first().map_or(0, CompleteSample::len),
        };
        for sample in &samples {
            let design = self.outcome_target_design(sample)?;
            let fit = Self::fit_outcome_target_weighted(sample, &design, None)?;
            let (level, gradient) = Self::plugin_gradient_at_fit(&fit, sample, at, None)?;
            if with_influence {
                run.influence.push(plugin_gradient_influence(&fit, sample, at)?);
            }
            run.levels.push(level);
            run.gradients.push(gradient);
        }
        Ok(run)
    }

    /// Pointwise confidence band of the Frequentist additive-GAM plug-in gradient
    /// (a [`ResponseFunctional::Jacobian`] on the identity scale, or a
    /// [`ResponseFunctional::DirectionalDerivative`]): the closed interval route the
    /// calibration harness measures.
    ///
    /// Each coordinate is a linear functional `c'β̂` of the target fit's
    /// coefficients (`c` the treatment-basis derivatives at `at`), so its
    /// influence is the fixed-basis penalized least-squares sandwich already used
    /// for the intervention-response level, with no covariate-average term (the
    /// additive gradient does not depend on the adjustment covariates). The
    /// variance uses HC2 row-leverage residuals from the same penalized Gram,
    /// accounting for unequal leverage of the spline basis. The band conditions on the fixed knots and
    /// adjustment penalty and excludes sieve-approximation bias of a non-additive
    /// or rough outcome surface. The public route withholds this band
    /// (`response.derivative_interval_withheld`) until its coverage records are
    /// measured; the returned value is the published point, bit for bit.
    ///
    /// # Errors
    ///
    /// The public route's refusals, a transformed Jacobian scale (no interval is
    /// constructed for it), invalid options, or a design whose penalized Gram is
    /// rank deficient.
    #[cfg(feature = "calibration-internal")]
    #[doc(hidden)]
    pub fn plugin_gradient_interval_internal(
        &self,
        data: &TabularData,
        functional: &ResponseFunctional,
    ) -> Result<(ResponseValue, ResponseUncertainty), EstimationError> {
        let level = self.options.confidence_level;
        if !level.is_finite() || level <= 0.0 || level >= 1.0 {
            return Err(EstimationError::unsupported("invalid continuous-response options"));
        }
        let (outcomes, treatments, at, direction) = match functional {
            ResponseFunctional::Jacobian { outcomes, treatments, at, scale } => {
                if *scale != DerivativeScale::Identity {
                    return Err(EstimationError::unsupported(
                        "the plug-in gradient interval is constructed on the identity scale only",
                    ));
                }
                (outcomes, treatments, at, None)
            }
            ResponseFunctional::DirectionalDerivative { outcomes, treatments, at, direction } => {
                (outcomes, treatments, at, Some(direction))
            }
            _ => {
                return Err(EstimationError::unsupported(
                    "the plug-in gradient interval serves Jacobian and directional derivatives",
                ));
            }
        };
        let kind = if direction.is_some() {
            PluginGradientKind::Directional
        } else {
            PluginGradientKind::Jacobian
        };
        let run = self.plugin_gradient_run(data, outcomes, treatments, at, kind, true)?;
        let nf = run.n as f64;
        let z = normal_ppf(0.5 + level / 2.0);
        let mut values = Vec::new();
        let mut lower = Vec::new();
        let mut upper = Vec::new();
        for (gradient, influence) in run.gradients.iter().zip(&run.influence) {
            let coordinates: Vec<(f64, Vec<f64>)> = match direction {
                None => gradient.iter().copied().zip(influence.columns.iter().cloned()).collect(),
                Some(direction) => {
                    let value = gradient.iter().zip(direction.iter()).map(|(a, b)| a * b).sum();
                    let mut psi = vec![0.0; run.n];
                    for (column, &weight) in influence.columns.iter().zip(direction.iter()) {
                        for (out, v) in psi.iter_mut().zip(column) {
                            *out += weight * v;
                        }
                    }
                    vec![(value, psi)]
                }
            };
            for (value, psi) in coordinates {
                let se = plugin_sandwich_se(&psi, nf);
                values.push(value);
                lower.push(value - z * se);
                upper.push(value + z * se);
            }
        }
        let value = match direction {
            None => ResponseValue::Jacobian {
                outcomes: outcomes.len(),
                treatments: treatments.len(),
                values: Arc::from(values),
            },
            Some(_) => ResponseValue::Vector(Arc::from(values)),
        };
        Ok((
            value,
            ResponseUncertainty::PointwiseBand {
                level,
                lower: Arc::from(lower),
                upper: Arc::from(upper),
                interpretation: antecedent_core::IntervalInterpretation::Confidence,
                draws: None,
            },
        ))
    }

    /// Evaluate the plug-in level and treatment gradient at `at` on a fitted GAM.
    pub(super) fn plugin_gradient_at_fit(
        fit: &antecedent_stats::GamFit,
        sample: &CompleteSample,
        at: &[f64],
        weights: Option<&[f64]>,
    ) -> Result<(f64, Vec<f64>), EstimationError> {
        let mut treat_smooths = Vec::with_capacity(at.len());
        for j in 0..at.len() {
            treat_smooths.push(fit.smooth_for_raw_col(j).ok_or_else(|| {
                EstimationError::unsupported("outcome nuisance is missing a treatment smooth")
            })?);
        }
        // Empirical μ̂(at) = α + Σ_j f_j(at[j]) + mean_i Σ_k g_k(X_i[k]).
        // The covariate offset is recovered from the in-sample fitted values
        // so we do not re-evaluate every adjustment smooth.
        let n = sample.len() as f64;
        let mut observed_treat_partial = 0.0;
        for i in 0..sample.len() {
            for (j, &smooth) in treat_smooths.iter().enumerate() {
                let a_ij = sample.treatment_matrix[j * sample.len() + i];
                observed_treat_partial +=
                    weights.map_or(1.0, |w| w[i]) * fit.smooth_partial(smooth, a_ij)?;
            }
        }
        let fitted_mean = fit
            .fitted
            .iter()
            .enumerate()
            .map(|(i, v)| v * weights.map_or(1.0, |w| w[i]))
            .sum::<f64>()
            / n;
        let covariate_offset = fitted_mean - fit.intercept - observed_treat_partial / n;
        let mut treat_level = 0.0;
        let mut gradient = Vec::with_capacity(at.len());
        for (j, &smooth) in treat_smooths.iter().enumerate() {
            treat_level += fit.smooth_partial(smooth, at[j])?;
            gradient.push(fit.smooth_derivative(smooth, at[j])?);
        }
        Ok((fit.intercept + treat_level + covariate_offset, gradient))
    }
}

pub(super) fn predict_one(
    fit: &antecedent_stats::GamFit,
    raw_row: &[f64],
) -> Result<f64, EstimationError> {
    Ok(fit.predict_row(raw_row)?)
}

/// Fixed-basis penalized least-squares sandwich of an additive-GAM fit on its own
/// rows. Dropping the last basis in each smooth removes the partition-of-unity
/// alias with the intercept; the D2 penalty is invariant to the corresponding
/// constant coefficient shift, so the reduced design spans the fitted space and
/// penalizes it the same way.
struct PluginSandwich {
    n: usize,
    p: usize,
    /// Column-major `n × p` design `[1 | B₁ (less its last basis) | …]`.
    design: Vec<f64>,
    /// `X'X + S`.
    gram: Vec<f64>,
    /// First design column of each raw column's smooth.
    offsets: Vec<usize>,
}

impl PluginSandwich {
    fn new(
        fit: &antecedent_stats::GamFit,
        sample: &CompleteSample,
    ) -> Result<Self, EstimationError> {
        let n = sample.len();
        let p = 1 + fit.smooths.iter().map(|s| s.n_basis - 1).sum::<usize>();
        let mut design = vec![1.0; n];
        let mut gram = vec![0.0; p * p];
        let mut offsets = Vec::with_capacity(sample.raw_cols);
        let mut offset = 1;
        for raw_col in 0..sample.raw_cols {
            let smooth = plugin_smooth(fit, raw_col)?;
            let (basis, _) = antecedent_stats::expand_bspline(
                &sample_raw_column(sample, raw_col),
                smooth.n_basis,
                Some(&smooth.knots),
            )?;
            design.extend_from_slice(&basis[..n * (smooth.n_basis - 1)]);
            for r in 0..smooth.n_basis - 2 {
                for (a, da) in [1.0, -2.0, 1.0].iter().enumerate() {
                    for (b, db) in [1.0, -2.0, 1.0].iter().enumerate() {
                        if r + a < smooth.n_basis - 1 && r + b < smooth.n_basis - 1 {
                            gram[(offset + r + b) * p + offset + r + a] += smooth.lambda * da * db;
                        }
                    }
                }
            }
            offsets.push(offset);
            offset += smooth.n_basis - 1;
        }
        for j in 0..p {
            for k in 0..p {
                gram[j * p + k] +=
                    (0..n).map(|i| design[j * n + i] * design[k * n + i]).sum::<f64>();
            }
        }
        Ok(Self { n, p, design, gram, offsets })
    }

    /// Per-row influence `n e_i x_i'(X'X + S)⁻¹ c` of the linear functional `c'β̂`.
    fn coefficient_influence(
        &self,
        residuals: &[f64],
        gradient: &[f64],
    ) -> Result<Vec<f64>, EstimationError> {
        let (n, p) = (self.n, self.p);
        let solve = FaerBackend.least_squares(
            &self.gram,
            p,
            p,
            gradient,
            &mut LeastSquaresWorkspace::default(),
        )?;
        if solve.rank < p {
            return Err(EstimationError::unsupported(
                "response influence requires an identifiable penalized design",
            ));
        }
        let nf = n as f64;
        Ok((0..n)
            .map(|i| {
                nf * residuals[i]
                    * (0..p).map(|j| self.design[j * n + i] * solve.coefficients[j]).sum::<f64>()
            })
            .collect())
    }

    /// Diagonal of the fixed-basis penalized smoother `X(X'X + S)⁻¹X'`.
    /// HC2 divides each in-sample residual by `sqrt(1 - h_ii)` before forming the
    /// functional's coefficient influence.
    fn leverage(&self) -> Result<Vec<f64>, EstimationError> {
        let inverse = invert_square(&self.gram, self.p).ok_or_else(|| {
            EstimationError::unsupported(
                "response influence requires an invertible penalized design",
            )
        })?;
        let mut out = Vec::with_capacity(self.n);
        for i in 0..self.n {
            let h = (0..self.p)
                .map(|j| {
                    let xj = self.design[j * self.n + i];
                    xj * (0..self.p)
                        .map(|k| inverse[j * self.p + k] * self.design[k * self.n + i])
                        .sum::<f64>()
                })
                .sum::<f64>();
            if !(0.0..1.0).contains(&h) {
                return Err(EstimationError::unsupported(
                    "response influence needs finite row leverage below one",
                ));
            }
            out.push(h);
        }
        Ok(out)
    }
}

fn plugin_smooth(
    fit: &antecedent_stats::GamFit,
    raw_col: usize,
) -> Result<&antecedent_stats::RecordedSmooth, EstimationError> {
    fit.smooth_for_raw_col(raw_col)
        .and_then(|index| fit.smooths.get(index))
        .ok_or_else(|| EstimationError::unsupported("missing GAM smooth in response influence"))
}

/// Raw column `raw_col` of the sample: treatments first, then adjustment.
fn sample_raw_column(sample: &CompleteSample, raw_col: usize) -> Vec<f64> {
    let n = sample.len();
    if raw_col < sample.treatment_cols {
        sample.treatment_matrix[raw_col * n..(raw_col + 1) * n].to_vec()
    } else {
        let col = raw_col - sample.treatment_cols;
        sample.adjustment[col * n..(col + 1) * n].to_vec()
    }
}

fn center_in_place(psi: &mut [f64]) {
    let center = psi.iter().sum::<f64>() / psi.len() as f64;
    for v in psi {
        *v -= center;
    }
}

// Fixed-basis penalized g-computation sandwich ([`PluginSandwich`]) plus the
// covariate-average term of the level.
pub(super) fn intervention_plugin_influence(
    fit: &antecedent_stats::GamFit,
    sample: &CompleteSample,
    interventions: &[Intervention],
    row_means: &[f64],
) -> Result<Vec<f64>, EstimationError> {
    let n = sample.len();
    let nf = n as f64;
    // With no change to any treatment, the estimand is the observed outcome
    // mean. Its exact empirical influence is Y_i - mean(Y): fitting the GAM
    // cannot add nuisance-fit uncertainty to this identity functional. Applying
    // HC2 to the residual term alone would destroy its cancellation with the fit.
    if !interventions.is_empty()
        && interventions.iter().all(|intervention| {
            matches!(intervention, Intervention::Shift { delta, .. }
                if delta.as_f64().is_some_and(|value| value == 0.0))
        })
    {
        let mean = sample.outcome.iter().sum::<f64>() / nf;
        return Ok(sample.outcome.iter().map(|value| value - mean).collect());
    }
    let sandwich = PluginSandwich::new(fit, sample)?;
    let mut gradient = vec![1.0];
    for raw_col in 0..sample.raw_cols {
        let smooth = plugin_smooth(fit, raw_col)?;
        let observed = sample_raw_column(sample, raw_col);
        let (points, weights): (Vec<f64>, Vec<f64>) = if let Some(iv) = interventions.get(raw_col) {
            if let Intervention::Shift { delta, .. } = iv {
                let delta = delta.as_f64().filter(|d| d.is_finite()).ok_or_else(|| {
                    EstimationError::unsupported(
                        "intervention response requires finite numeric values",
                    )
                })?;
                (observed.iter().map(|&v| v + delta).collect(), vec![1.0 / nf; n])
            } else {
                policy_support(iv)?
                    .into_iter()
                    .map(|atom| match atom {
                        DiscreteAtom::Level { value, weight } => (value, weight),
                        DiscreteAtom::Shift { delta } => (delta, 1.0),
                    })
                    .unzip()
            }
        } else {
            (observed, vec![1.0 / nf; n])
        };
        let (counterfactual, _) =
            antecedent_stats::expand_bspline(&points, smooth.n_basis, Some(&smooth.knots))?;
        for j in 0..smooth.n_basis - 1 {
            gradient.push(
                counterfactual[j * points.len()..(j + 1) * points.len()]
                    .iter()
                    .zip(&weights)
                    .map(|(v, w)| v * w)
                    .sum(),
            );
        }
    }
    // The fitted residuals are shrunken by row leverage. Match the HC2
    // correction used by the same fit's plug-in gradient interval.
    let leverage = sandwich.leverage()?;
    let residuals: Vec<f64> = fit
        .residuals
        .iter()
        .zip(leverage)
        .map(|(residual, h)| residual / (1.0 - h).sqrt())
        .collect();
    let coefficient = sandwich.coefficient_influence(&residuals, &gradient)?;
    let mean = row_means.iter().sum::<f64>() / nf;
    let mut psi: Vec<_> =
        coefficient.iter().zip(row_means).map(|(c, row)| row - mean + c).collect();
    center_in_place(&mut psi);
    Ok(psi)
}

/// Which plug-in gradient query a run serves, for its refusal wording.
#[derive(Clone, Copy)]
pub(super) enum PluginGradientKind {
    Jacobian,
    Directional,
}

impl PluginGradientKind {
    fn refusal(self) -> (&'static str, &'static str) {
        match self {
            Self::Jacobian => (
                "plug-in response Jacobian supports at most two treatments",
                JACOBIAN_TREATMENT_LIMIT_REMEDY,
            ),
            Self::Directional => (
                "plug-in directional derivative supports at most two treatments",
                DIRECTIONAL_TREATMENT_LIMIT_REMEDY,
            ),
        }
    }
}

/// Plug-in levels and gradients of every outcome (and their influence on request).
pub(super) struct PluginGradientRun {
    pub(super) levels: Vec<f64>,
    pub(super) gradients: Vec<Vec<f64>>,
    /// One entry per outcome when influence was requested, else empty. Read only
    /// by the calibration-internal band.
    #[cfg_attr(
        not(feature = "calibration-internal"),
        allow(dead_code, reason = "read only by the calibration-internal band")
    )]
    influence: Vec<PluginGradientInfluence>,
    pub(super) all_treatments: Vec<f64>,
    /// Shared complete-case row count.
    #[cfg_attr(
        not(feature = "calibration-internal"),
        allow(dead_code, reason = "read only by the calibration-internal band")
    )]
    n: usize,
}

/// Coefficient-sandwich influence of one outcome's gradient coordinates.
#[cfg_attr(
    not(feature = "calibration-internal"),
    allow(dead_code, reason = "read only by the calibration-internal band")
)]
struct PluginGradientInfluence {
    /// One centred column per treatment coordinate.
    columns: Vec<Vec<f64>>,
}

/// Influence of each plug-in gradient coordinate `∂μ̂/∂a_j (at)`.
///
/// The additive gradient is `f_j'(at_j) = Σ_b B_b'(at_j) β_b`, a linear functional
/// of the treatment smooth's coefficients alone: it carries no covariate-average
/// term. In the reduced design the dropped last basis contributes nothing,
/// because the basis derivatives sum to zero (partition of unity).
fn plugin_gradient_influence(
    fit: &antecedent_stats::GamFit,
    sample: &CompleteSample,
    at: &[f64],
) -> Result<PluginGradientInfluence, EstimationError> {
    let sandwich = PluginSandwich::new(fit, sample)?;
    let leverage = sandwich.leverage()?;
    let hc2_residuals: Vec<f64> = fit
        .residuals
        .iter()
        .zip(&leverage)
        .map(|(residual, h)| residual / (1.0 - h).sqrt())
        .collect();
    let mut columns = Vec::with_capacity(at.len());
    for (j, &point) in at.iter().enumerate() {
        let index = fit.smooth_for_raw_col(j).ok_or_else(|| {
            EstimationError::unsupported("outcome nuisance is missing a treatment smooth")
        })?;
        let derivative = fit.smooth_basis_derivative(index, point)?;
        let mut gradient = vec![0.0; sandwich.p];
        let offset = sandwich.offsets[j];
        gradient[offset..offset + derivative.len() - 1]
            .copy_from_slice(&derivative[..derivative.len() - 1]);
        let mut psi = sandwich.coefficient_influence(&hc2_residuals, &gradient)?;
        center_in_place(&mut psi);
        columns.push(psi);
    }
    Ok(PluginGradientInfluence { columns })
}

/// HC2 standard error of a centred influence column: `sqrt(Σψ²) / n` after
/// dividing each fit residual by its own `sqrt(1 - h_ii)` in
/// [`plugin_gradient_influence`].
#[cfg(feature = "calibration-internal")]
fn plugin_sandwich_se(psi: &[f64], n: f64) -> f64 {
    if !(n.is_finite() && n > 0.0) {
        return f64::NAN;
    }
    (psi.iter().map(|x| x * x).sum::<f64>() / (n * n)).sqrt()
}

/// Why the Frequentist plug-in gradient publishes no interval.
pub(super) fn plugin_gradient_interval_withheld() -> Diagnostic {
    Diagnostic::new(
        "response.derivative_interval_withheld",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Warning,
        "no Frequentist interval is reported for this additive-GAM plug-in gradient: its coefficient-sandwich confidence band is wired for calibration but its repeated-sampling coverage is not yet measured, so the interval route stays closed and only the point value is published; Bayesian inference publishes a measured pointwise credible band for the same plug-in gradient",
    )
}

/// Model-dependence disclosure for the intervention-response g-computation,
/// worded for the outcome fit that actually ran. `target_fallback` carries the
/// reason the unpenalized treatment-spline target fit failed, if it did.
pub(super) fn intervention_plugin_warnings(target_fallback: Option<&str>) -> Vec<Diagnostic> {
    let Some(reason) = target_fallback else {
        return vec![Diagnostic::new(
            "response.intervention_plugin_model_dependent",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            "intervention response uses additive-GAM g-computation with unpenalized treatment splines (the target) and penalized adjustment splines (nuisances); SE includes fitted-coefficient and covariate-average influence conditional on the fitted spline knots and adjustment penalty; it excludes knot-selection, sieve-approximation bias of a non-additive or rough outcome surface, and policy-integration error",
        )];
    };
    let mut fallback = Diagnostic::new(
        "response.intervention_target_penalized_fallback",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Warning,
        "the unpenalized treatment-spline target fit failed (typically a binary or low-cardinality treatment that cannot support an unpenalized cubic basis), so the level and SE come from the penalized nuisance outcome fit; roughness shrinkage of the treatment smooths can bias the level toward the observed mean by a fraction of its SE, and the interval does not account for that bias",
    );
    fallback.fields = Arc::from(vec![(Arc::from("target_fit_error"), Arc::from(reason))]);
    vec![
        Diagnostic::new(
            "response.intervention_plugin_model_dependent",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            "intervention response uses additive-GAM g-computation with penalized treatment and adjustment splines (the unpenalized treatment-spline target fit failed); SE includes fitted-coefficient and covariate-average influence conditional on the fitted spline knots and penalties; it excludes penalty-induced shrinkage bias, knot-selection, sieve-approximation bias of a non-additive or rough outcome surface, and policy-integration error",
        ),
        fallback,
    ]
}
