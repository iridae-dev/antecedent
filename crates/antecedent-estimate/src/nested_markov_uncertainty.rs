//! Expected-Fisher uncertainty for the selected binary nested-Markov likelihood.
//!
//! This is an internal implementation for later calibration, not a licensed
//! interval. The IID multinomial likelihood has expected information
//! `N sum_i (dp_i/dtheta)(dp_i/dtheta)'/p_i`. Its inverse propagates the full
//! eleven-parameter covariance into both intervention means and their contrast.
//! Analytic cell derivatives preserve the equality constraints. Regular interior
//! fits only: sparse cells, a singular information matrix and fractional pseudo-counts
//! refuse. Correct model specification and IID sampling are declared assumptions;
//! a table alone cannot verify them. Empirical constraint residuals stay visible.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(feature = "calibration-internal")]
mod internal {
    use antecedent_core::ExecutionContext;
    use antecedent_stats::{chol_solve, cholesky_spd, special::normal_ppf};

    use crate::error::EstimationError;
    use crate::nested_markov_binary::{
        BinaryCells, FitOptions, NestedMarkovInput, NestedParameters, PilotReport, binary_cells,
        cell_index, evaluate_nested_markov_pilot,
    };

    /// Number of free parameters, ordered `a,c0,c1,q20,q21,q40,q41,g00,g01,g10,g11`.
    pub const PARAMETER_COUNT: usize = 11;

    /// A regular-model approximation awaiting whole-method repeated-sampling calibration.
    #[derive(Clone, Debug, PartialEq)]
    pub struct NestedMarkovUncertainty {
        /// Checked scope, constrained likelihood fit, and point contrasts.
        pub pilot: PilotReport,
        /// IID multinomial sample count (integer frequencies, not an effective sample size).
        pub sample_count: f64,
        /// Row-major inverse expected Fisher information for the eleven parameters.
        pub parameter_covariance: Vec<f64>,
        /// Row-major covariance of `[mean0, mean1, mean1-mean0]`.
        pub effect_covariance: [[f64; 3]; 3],
        /// Requested nominal two-sided level; no measured coverage claim.
        pub nominal_level: f64,
        /// Unmeasured normal delta-method interval candidates, in the same effect order.
        pub interval_candidates: [[f64; 2]; 3],
        /// Maximum absolute `I Cov - identity` residual; no regularization is applied.
        pub inverse_residual: f64,
    }

    /// One of the three actual candidate functionals, in covariance/interval order.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub enum NestedFisherFunctional {
        /// Mean of X4 under do(X2=0).
        Mean0,
        /// Mean of X4 under do(X2=1).
        Mean1,
        /// Difference of the two means.
        Contrast,
    }

    impl NestedMarkovUncertainty {
        /// Measurement binding for this actual candidate; never activates a public route.
        #[must_use]
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "successful candidate validated a bounded positive integer count total"
        )]
        pub fn calibration_basis(
            &self,
            functional: NestedFisherFunctional,
        ) -> antecedent_core::CalibrationBasis {
            use std::sync::Arc;
            let (query, target) = match functional {
                NestedFisherFunctional::Mean0 => ("InterventionResponse", "verma.do_x2_0.mean_x4"),
                NestedFisherFunctional::Mean1 => ("InterventionResponse", "verma.do_x2_1.mean_x4"),
                NestedFisherFunctional::Contrast => {
                    ("AverageEffect", "verma.do_x2_1_minus_0.mean_x4")
                }
            };
            antecedent_core::CalibrationBasis::new(
                [
                    query,
                    "Admg",
                    "fixed",
                    "tabular",
                    "Frequentist",
                    "nested_markov_fisher",
                    "analytic_se",
                    "expected_fisher_delta",
                    "iid_multinomial",
                    "",
                    target,
                ]
                .map(Arc::from),
                self.nominal_level,
                Arc::from("point"),
                self.sample_count as u64,
                None,
                None,
                0.,
            )
        }
    }

    fn refusal(detail: &str, message: &str) -> EstimationError {
        EstimationError::refused(
            antecedent_core::reason_code!("invalid_argument"),
            format!("{detail}: {message}"),
        )
    }

    fn cell_derivatives(p: &NestedParameters) -> [[f64; PARAMETER_COUNT]; 16] {
        let cells = p.cell_probabilities();
        let mut derivatives = [[0.; PARAMETER_COUNT]; 16];
        for x1 in 0..2 {
            for x2 in 0..2 {
                for x3 in 0..2 {
                    for x4 in 0..2 {
                        let index = cell_index(x1, x2, x3, x4);
                        let a = if x1 == 0 { p.a } else { 1. - p.a };
                        let c = if x3 == 0 { p.c[x2] } else { 1. - p.c[x2] };
                        let sign1 = if x1 == 0 { 1. } else { -1. };
                        let sign3 = if x3 == 0 { 1. } else { -1. };
                        let row = &mut derivatives[index];
                        row[0] = sign1 * cells[index] / a;
                        row[1 + x2] = sign3 * cells[index] / c;
                        if x4 == 1 {
                            row[3 + x1] = a * c * if x2 == 0 { 1. } else { -1. };
                        }
                        if x2 == 1 {
                            row[5 + x3] = a * c * if x4 == 0 { 1. } else { -1. };
                        }
                        row[7 + 2 * x1 + x3] = a * c * if x2 == x4 { 1. } else { -1. };
                    }
                }
            }
        }
        derivatives
    }

    fn information(p: &NestedParameters, n: f64) -> Vec<f64> {
        let cells = p.cell_probabilities();
        let derivatives = cell_derivatives(p);
        let mut matrix = vec![0.; PARAMETER_COUNT * PARAMETER_COUNT];
        for (cell, row) in cells.iter().zip(derivatives) {
            for i in 0..PARAMETER_COUNT {
                for j in 0..PARAMETER_COUNT {
                    matrix[i * PARAMETER_COUNT + j] += n * row[i] * row[j] / cell;
                }
            }
        }
        matrix
    }

    fn gradients(p: &NestedParameters) -> [[f64; PARAMETER_COUNT]; 3] {
        let mut gradient = [[0.; PARAMETER_COUNT]; 3];
        for (action, row) in gradient[..2].iter_mut().enumerate() {
            row[1 + action] = p.q4[1] - p.q4[0];
            row[5] = -p.c[action];
            row[6] = -(1. - p.c[action]);
        }
        for j in 0..PARAMETER_COUNT {
            gradient[2][j] = gradient[1][j] - gradient[0][j];
        }
        gradient
    }

    fn invert(information: &[f64]) -> Result<(Vec<f64>, f64), EstimationError> {
        let chol = cholesky_spd(information, PARAMETER_COUNT).ok_or_else(|| {
            refusal("nested_markov.fisher_singular_information", "expected information is not SPD")
        })?;
        let mut covariance = vec![0.; PARAMETER_COUNT * PARAMETER_COUNT];
        for j in 0..PARAMETER_COUNT {
            let mut column = [0.; PARAMETER_COUNT];
            column[j] = 1.;
            let solved = chol_solve(&chol, PARAMETER_COUNT, &column).ok_or_else(|| {
                refusal("nested_markov.fisher_singular_information", "information solve failed")
            })?;
            for i in 0..PARAMETER_COUNT {
                covariance[i * PARAMETER_COUNT + j] = solved[i];
            }
        }
        let mut residual = 0.0_f64;
        for i in 0..PARAMETER_COUNT {
            for j in 0..PARAMETER_COUNT {
                let product: f64 = (0..PARAMETER_COUNT)
                    .map(|k| {
                        information[i * PARAMETER_COUNT + k] * covariance[k * PARAMETER_COUNT + j]
                    })
                    .sum();
                residual = residual.max((product - if i == j { 1. } else { 0. }).abs());
            }
        }
        if !residual.is_finite() || residual > 1e-8 {
            return Err(refusal(
                "nested_markov.fisher_singular_information",
                "inverse failed residual check",
            ));
        }
        Ok((covariance, residual))
    }

    fn validate_counts(counts: &BinaryCells) -> Result<(), EstimationError> {
        if counts.total() > 9_007_199_254_740_992.
            || counts.counts().iter().any(|n| *n <= 0. || n.fract() != 0.)
        {
            return Err(refusal(
                "nested_markov.fisher_sampling_design",
                "requires positive integer IID multinomial counts",
            ));
        }
        Ok(())
    }

    /// Fit the selected graph and propagate the regular IID multinomial information.
    /// This feature-gated candidate does not activate the closed public interval route.
    ///
    /// # Errors
    /// Invalid level/counts, unsupported graph/regime, failed or boundary likelihood fit,
    /// cancellation or singular information. Correct specification remains an assumption.
    pub fn nested_markov_fisher_internal(
        input: &NestedMarkovInput,
        options: &FitOptions,
        nominal_level: f64,
        ctx: &ExecutionContext,
    ) -> Result<NestedMarkovUncertainty, EstimationError> {
        if !(nominal_level.is_finite() && 0. < nominal_level && nominal_level < 1.) {
            return Err(refusal(
                "nested_markov.fisher_invalid_level",
                "nominal level must lie strictly between zero and one",
            ));
        }
        let counts = binary_cells(input)?;
        validate_counts(&counts)?;
        let pilot = evaluate_nested_markov_pilot(input, options, ctx).map_err(|r| r.error)?;
        let (parameter_covariance, inverse_residual) =
            invert(&information(&pilot.fit.parameters, counts.total()))?;
        let gradients = gradients(&pilot.fit.parameters);
        let mut effect_covariance = [[0.; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for a in 0..PARAMETER_COUNT {
                    for b in 0..PARAMETER_COUNT {
                        effect_covariance[i][j] += gradients[i][a]
                            * parameter_covariance[a * PARAMETER_COUNT + b]
                            * gradients[j][b];
                    }
                }
            }
        }
        let z = normal_ppf(0.5 + 0.5 * nominal_level);
        let values = [
            pilot.comparison.model_means[0],
            pilot.comparison.model_means[1],
            pilot.comparison.model_contrast,
        ];
        let mut interval_candidates = [[0.; 2]; 3];
        for i in 0..3 {
            let variance = effect_covariance[i][i];
            if !variance.is_finite() || variance < 0. || !z.is_finite() {
                return Err(refusal(
                    "nested_markov.fisher_invalid_covariance",
                    "delta variance or quantile is invalid",
                ));
            }
            let radius = z * variance.sqrt();
            interval_candidates[i] = [values[i] - radius, values[i] + radius];
        }
        Ok(NestedMarkovUncertainty {
            pilot,
            sample_count: counts.total(),
            parameter_covariance,
            effect_covariance,
            nominal_level,
            interval_candidates,
            inverse_residual,
        })
    }
}

#[cfg(feature = "calibration-internal")]
pub use internal::{
    NestedFisherFunctional, NestedMarkovUncertainty, PARAMETER_COUNT, nested_markov_fisher_internal,
};
