//! Retained adjusted regression: the fitted coefficients and covariance remain live;
//! compatible contrasts and predictions perform no fitting.
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::float_cmp, reason = "binary and dummy variables require exact coded levels")]

use crate::EstimationError;
use crate::categorical_treatment::{
    CategoricalTreatmentInput, CategoricalTreatmentSpec, count_levels, dummy_columns, validate_spec,
};
use crate::vector_treatment::{VectorTreatmentInput, VectorTreatmentOptions, fit_joint_retained};
use antecedent_stats::{FaerBackend, GlmDesignRef, GlmFamily, GlmOptions, LeastSquaresWorkspace};
use std::cell::Cell;

thread_local! { static FITS: Cell<u64> = const { Cell::new(0) }; }
/// Measure successful model solves on the calling thread. Nested observers all see fits.
pub fn count_adjusted_model_fits<R>(work: impl FnOnce() -> R) -> (R, u64) {
    let before = FITS.with(Cell::get);
    let result = work();
    (result, FITS.with(Cell::get).saturating_sub(before))
}
pub(crate) fn record_model_fit() {
    FITS.with(|c| c.set(c.get().saturating_add(1)));
}

/// Retained model over a complete row snapshot. Columns: intercept, adjustment, treatments.
#[derive(Clone, Debug)]
pub struct AdjustedFit {
    coefficients: Vec<f64>,
    covariance: Vec<f64>,
    family: GlmFamily,
    matrix: Vec<f64>,
    nrows: usize,
    ncols: usize,
    first_treatment: usize,
    levels: Option<(Vec<String>, String)>,
    support: Vec<Vec<f64>>,
}
impl AdjustedFit {
    /// Fit the existing joint OLS engine; supports scalar or vector treatment roles.
    pub fn linear(
        input: &VectorTreatmentInput,
        options: &VectorTreatmentOptions,
    ) -> Result<Self, EstimationError> {
        let (_, coefficients, covariance) = fit_joint_retained(input, options, 1)?;
        Ok(Self::from_linear(input, coefficients, covariance, None))
    }
    fn from_linear(
        input: &VectorTreatmentInput,
        coefficients: Vec<f64>,
        covariance: Vec<f64>,
        levels: Option<(Vec<String>, String)>,
    ) -> Self {
        let nrows = input.outcome.len();
        let ncols = coefficients.len();
        let mut matrix = vec![1.; nrows];
        for z in &input.adjustment {
            matrix.extend(&z.values);
        }
        for t in &input.treatments {
            matrix.extend(&t.values);
        }
        Self {
            coefficients,
            covariance,
            family: GlmFamily::GaussianIdentity,
            matrix,
            nrows,
            ncols,
            first_treatment: 1 + input.adjustment.len(),
            support: input.treatments.iter().map(|t| support(&t.values)).collect(),
            levels,
        }
    }
    /// Validate and dummy-code the existing categorical regime, then retain its joint solve.
    pub fn categorical(
        input: &CategoricalTreatmentInput,
        spec: &CategoricalTreatmentSpec,
    ) -> Result<Self, EstimationError> {
        let order = validate_spec(spec)?;
        count_levels(input, spec, &order)?;
        let vector = VectorTreatmentInput {
            outcome: input.outcome.clone(),
            row_snapshot: input.row_snapshot.clone(),
            adjustment: input.adjustment.clone(),
            treatments: dummy_columns(input, &order, &spec.reference),
        };
        let (_, beta, cov) = fit_joint_retained(
            &vector,
            &VectorTreatmentOptions { covariance: spec.covariance, contrasts: vec![] },
            1,
        )?;
        Ok(Self::from_linear(&vector, beta, cov, Some((order, spec.reference.clone()))))
    }
    /// Fit an unpenalized GLM and retain its full Fisher covariance.
    pub fn glm(
        input: &VectorTreatmentInput,
        family: GlmFamily,
        options: &GlmOptions,
    ) -> Result<Self, EstimationError> {
        crate::vector_treatment::validate_input(input, 1)?;
        let n = input.outcome.len();
        if input.outcome.iter().any(|&y| match family {
            GlmFamily::BinomialLogit | GlmFamily::BinomialProbit => y != 0. && y != 1.,
            GlmFamily::PoissonLog | GlmFamily::NegativeBinomial => y < 0.,
            GlmFamily::GaussianIdentity => false,
        }) {
            return Err(EstimationError::data_msg("outcome incompatible with adjusted GLM family"));
        }
        let p = 1 + input.adjustment.len() + input.treatments.len();
        let mut x = vec![1.; n];
        for z in &input.adjustment {
            x.extend(&z.values);
        }
        for t in &input.treatments {
            x.extend(&t.values);
        }
        let mut ws = LeastSquaresWorkspace::default();
        let fit = crate::glm_adjustment::solve_adjustment_glm(
            family,
            GlmDesignRef { x_colmajor: &x, nrows: n, ncols: p, y: &input.outcome },
            FaerBackend,
            &mut ws,
            options,
        )?;
        let alpha = crate::glm_adjustment::resolve_nb_alpha(
            family,
            fit.nb_alpha,
            &x,
            n,
            p,
            &fit.coefficients,
            &input.outcome,
        );
        let covariance = crate::glm_adjustment::adjustment_glm_covariance(
            family,
            &x,
            n,
            p,
            &fit.coefficients,
            fit.deviance,
            alpha,
        )
        .ok_or_else(|| EstimationError::stats_msg("singular adjusted GLM information"))?;
        Ok(Self {
            coefficients: fit.coefficients,
            covariance,
            family,
            matrix: x,
            nrows: n,
            ncols: p,
            first_treatment: 1 + input.adjustment.len(),
            support: input.treatments.iter().map(|t| support(&t.values)).collect(),
            levels: None,
        })
    }
    /// Full coefficient covariance, in retained design order.
    #[must_use]
    pub fn covariance(&self) -> &[f64] {
        &self.covariance
    }
    /// Canonical numeric design dimension excluding intercept.
    #[must_use]
    pub fn feature_count(&self) -> usize {
        self.ncols - 1
    }
    /// Validate prediction feature shape and declared treatment support without fitting.
    #[must_use]
    pub fn prediction_refusal(&self, rows: &[Vec<f64>]) -> Option<&'static str> {
        for row in rows {
            if row.len() != self.ncols - 1 || row.iter().any(|x| !x.is_finite()) {
                return Some("recalc.adjusted_prediction_schema_mismatch");
            }
            let t = &row[self.first_treatment - 1..];
            if self.levels.is_some() {
                if t.iter().any(|x| *x != 0. && *x != 1.) || t.iter().sum::<f64>() > 1. {
                    return Some("recalc.adjusted_prediction_out_of_support");
                }
            } else if !t.iter().zip(&self.support).all(|(x, b)| in_support(*x, b)) {
                return Some("recalc.adjusted_prediction_out_of_support");
            }
        }
        None
    }
    /// Mean predictions for rows in adjustment-then-treatment design order. No fitting.
    pub fn predict(&self, rows: &[Vec<f64>]) -> Result<Vec<f64>, EstimationError> {
        rows.iter()
            .map(|row| {
                if row.len() != self.ncols - 1 || row.iter().any(|v| !v.is_finite()) {
                    return Err(EstimationError::data_msg(
                        "adjusted prediction feature schema mismatch",
                    ));
                }
                if self.levels.is_none()
                    && !row[self.first_treatment - 1..]
                        .iter()
                        .zip(&self.support)
                        .all(|(value, bounds)| in_support(*value, bounds))
                {
                    return Err(EstimationError::data_msg(
                        "adjusted prediction is outside observed treatment support",
                    ));
                }
                if self.levels.is_some() {
                    let dummies = &row[self.first_treatment - 1..];
                    if dummies.iter().any(|v| *v != 0. && *v != 1.)
                        || dummies.iter().sum::<f64>() > 1.
                    {
                        return Err(EstimationError::data_msg(
                            "categorical prediction is outside declared level support",
                        ));
                    }
                }
                let eta = self.coefficients[0]
                    + row.iter().zip(&self.coefficients[1..]).map(|(x, b)| x * b).sum::<f64>();
                let mean = mean(self.family, eta);
                if !mean.is_finite() {
                    return Err(EstimationError::stats_msg("nonfinite adjusted prediction"));
                }
                Ok(mean)
            })
            .collect()
    }
    /// Evaluate a probability/response scale contrast and delta SE conditional on row law.
    pub fn contrast(
        &self,
        active: &[f64],
        control: &[f64],
        weights: Option<&[f64]>,
    ) -> Result<(f64, f64), EstimationError> {
        let k = self.ncols - self.first_treatment;
        if active.len() != k
            || control.len() != k
            || active.iter().chain(control).any(|v| !v.is_finite())
        {
            return Err(EstimationError::data_msg("adjusted contrast dimension mismatch"));
        }
        if self.levels.is_none()
            && !active
                .iter()
                .chain(control)
                .zip(self.support.iter().chain(&self.support))
                .all(|(value, bounds)| in_support(*value, bounds))
        {
            return Err(EstimationError::data_msg(
                "adjusted contrast outside observed treatment support",
            ));
        }
        if let Some(w) = weights {
            if w.len() != self.nrows || w.iter().any(|v| !v.is_finite() || *v < 0.) {
                return Err(EstimationError::data_msg("invalid adjusted target weights"));
            }
        }
        let mass = weights.map_or(self.nrows as f64, |w| w.iter().sum());
        if !mass.is_finite() || mass <= 0. {
            return Err(EstimationError::data_msg("empty adjusted target law"));
        }
        let mut effect = 0.;
        let mut gradient = vec![0.; self.ncols];
        if self.family == GlmFamily::GaussianIdentity {
            // Identity-link contrasts cancel the intercept and adjustment block exactly.
            // Subtract coefficients directly rather than large predicted outcome levels.
            for (i, (a, c)) in active.iter().zip(control).enumerate() {
                let delta = a - c;
                gradient[self.first_treatment + i] = delta;
                effect += delta * self.coefficients[self.first_treatment + i];
            }
        } else {
            for r in 0..self.nrows {
                let weight = weights.map_or(1., |w| w[r]) / mass;
                let base = (0..self.first_treatment)
                    .map(|c| self.matrix[c * self.nrows + r] * self.coefficients[c])
                    .sum::<f64>();
                let eta_a = base
                    + active
                        .iter()
                        .zip(&self.coefficients[self.first_treatment..])
                        .map(|(x, b)| x * b)
                        .sum::<f64>();
                let eta_c = base
                    + control
                        .iter()
                        .zip(&self.coefficients[self.first_treatment..])
                        .map(|(x, b)| x * b)
                        .sum::<f64>();
                effect += weight * (mean(self.family, eta_a) - mean(self.family, eta_c));
                let da = crate::glm_adjustment::mean_derivative(self.family, eta_a);
                let dc = crate::glm_adjustment::mean_derivative(self.family, eta_c);
                for (c, g) in gradient.iter_mut().enumerate() {
                    let (a, b) = if c < self.first_treatment {
                        let x = self.matrix[c * self.nrows + r];
                        (x, x)
                    } else {
                        (active[c - self.first_treatment], control[c - self.first_treatment])
                    };
                    *g += weight * (da * a - dc * b);
                }
            }
        }
        let variance = (0..self.ncols)
            .flat_map(|i| (0..self.ncols).map(move |j| (i, j)))
            .map(|(i, j)| gradient[i] * self.covariance[i * self.ncols + j] * gradient[j])
            .sum::<f64>();
        if !effect.is_finite() || !variance.is_finite() || variance < -1e-12 {
            return Err(EstimationError::stats_msg("invalid adjusted contrast"));
        }
        Ok((effect, variance.max(0.).sqrt()))
    }
    /// Encode declared categorical levels, refusing unknown levels.
    pub fn categorical_arms(
        &self,
        from: &str,
        to: &str,
    ) -> Result<(Vec<f64>, Vec<f64>), EstimationError> {
        let (levels, reference) = self
            .levels
            .as_ref()
            .ok_or_else(|| EstimationError::data_msg("numeric adjusted model"))?;
        if !levels.iter().any(|l| l == from) || !levels.iter().any(|l| l == to) {
            return Err(EstimationError::data_msg("unknown adjusted categorical contrast level"));
        }
        let encode = |level: &str| {
            levels
                .iter()
                .filter(|l| *l != reference)
                .map(|l| if l == level { 1. } else { 0. })
                .collect()
        };
        Ok((encode(to), encode(from)))
    }
}
fn mean(family: GlmFamily, eta: f64) -> f64 {
    match family {
        GlmFamily::GaussianIdentity => eta,
        GlmFamily::BinomialLogit => {
            if eta >= 0. {
                1. / (1. + (-eta).exp())
            } else {
                let e = eta.exp();
                e / (1. + e)
            }
        }
        GlmFamily::BinomialProbit => {
            0.5 * antecedent_kernels::erfc(-eta / std::f64::consts::SQRT_2)
        }
        GlmFamily::PoissonLog | GlmFamily::NegativeBinomial => eta.exp(),
    }
}

fn support(values: &[f64]) -> Vec<f64> {
    if values.iter().all(|v| *v == 0. || *v == 1.) {
        vec![0., 1., 1.]
    } else {
        vec![
            values.iter().copied().fold(f64::INFINITY, f64::min),
            values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        ]
    }
}
fn in_support(value: f64, bounds: &[f64]) -> bool {
    if bounds.len() == 3 {
        value == 0. || value == 1.
    } else {
        value >= bounds[0] && value <= bounds[1]
    }
}
