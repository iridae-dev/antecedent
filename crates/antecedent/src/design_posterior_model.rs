//! Declared models of the randomized-design posterior routes.
//!
//! A posterior needs a stated likelihood and prior. The routes below take the likelihood from
//! the query's design and take the rest from the model declared here, never from the data:
//! the hazard partition and gamma prior of the piecewise-exponential survival posterior, and
//! the dose support, coefficient prior scales and residual prior of the polynomial dose-curve
//! posterior. A different declaration is a different posterior, so it is part of the study's
//! inference identity.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_estimate::dose_basis_regression::ResidualPrior;

/// The declared model of a randomized-design Bayesian estimator.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum DesignPosteriorModel {
    /// Hazards are constant on every interval `(cuts[k-1], cuts[k]]` and independent
    /// `Gamma(prior_shape, prior_rate)` a priori. The last cut is the survival horizon and must
    /// equal the query's `tau`.
    PiecewiseHazard {
        /// Strictly increasing positive interval right endpoints.
        cuts: Arc<[f64]>,
        /// Gamma prior shape: the prior event count.
        prior_shape: f64,
        /// Gamma prior rate: the prior time at risk.
        prior_rate: f64,
    },
    /// `E[Y | D = d, group]` is a polynomial in the dose rescaled to `[-1, 1]` on `support`,
    /// with coefficient `j` a priori `N(0, sigma^2 s_j^2)`.
    DosePolynomial {
        /// Declared dose support the polynomial is rescaled on.
        support: (f64, f64),
        /// Prior scale `s_j` of every polynomial coefficient.
        coefficient_scales: Arc<[f64]>,
        /// Prior of the residual variance.
        residual: ResidualPrior,
    },
    /// Arm outcomes are independent draws from the arm's law: Bernoulli with a Beta prior or
    /// Gaussian with a Normal-Inverse-Gamma prior, the same prior on every arm. The estimand
    /// (the trial's own units or the sampled population) is chosen by the estimator id.
    ArmOutcome {
        /// The declared arm outcome model and its prior.
        model: antecedent_estimate::randomized_posterior::ArmOutcomeModel,
    },
    /// Within each randomized arm the outcomes fall into the bins cut by the declared threshold
    /// grid, independently, with one multinomial probability vector per arm and a symmetric
    /// `Dirichlet(bin_concentration)` prior on it. The grid is part of the model: inference
    /// conditions on it and the within-bin law is the linear interpolation of the distribution
    /// function.
    QuantileGrid {
        /// Strictly increasing threshold grid of at least two cut points.
        thresholds: Arc<[f64]>,
        /// Symmetric Dirichlet concentration of every bin.
        bin_concentration: f64,
    },
    /// Switchback outcomes are Gaussian within independent sequences with the declared carryover,
    /// within-sequence dependence, period adjustment and diagonal prior.
    Switchback {
        /// The declared switchback model.
        model: antecedent_estimate::switchback_posterior::SwitchbackModel,
    },
}

impl DesignPosteriorModel {
    /// Canonical text of the declaration, bit for bit, bound into the inference identity.
    #[must_use]
    pub fn identity(&self) -> String {
        let bits = |values: &[f64]| {
            values.iter().map(|v| format!("{:016x}", v.to_bits())).collect::<Vec<_>>().join(",")
        };
        match self {
            Self::PiecewiseHazard { cuts, prior_shape, prior_rate } => format!(
                "piecewise_hazard;cuts={};shape={:016x};rate={:016x}",
                bits(cuts),
                prior_shape.to_bits(),
                prior_rate.to_bits()
            ),
            Self::ArmOutcome { model } => {
                use antecedent_estimate::randomized_posterior::ArmOutcomeModel;
                match model {
                    ArmOutcomeModel::Bernoulli(prior) => format!(
                        "arm_outcome;bernoulli;alpha={:016x};beta={:016x}",
                        prior.alpha.to_bits(),
                        prior.beta.to_bits()
                    ),
                    ArmOutcomeModel::Gaussian(prior) => format!(
                        "arm_outcome;gaussian;mean={:016x};pseudo_units={:016x};shape={:016x};scale={:016x}",
                        prior.mean.to_bits(),
                        prior.pseudo_units.to_bits(),
                        prior.variance_shape.to_bits(),
                        prior.variance_scale.to_bits()
                    ),
                }
            }
            Self::Switchback { model } => {
                use antecedent_estimate::switchback_posterior::{
                    CarryoverDeclaration, PeriodAdjustment, SequenceDependence,
                };
                let carryover = match model.carryover {
                    CarryoverDeclaration::None => "none",
                    CarryoverDeclaration::FirstOrder => "first_order",
                    CarryoverDeclaration::Unrestricted => "unrestricted",
                    _ => "other",
                };
                let dependence = match model.dependence {
                    SequenceDependence::Independent => "independent".to_owned(),
                    SequenceDependence::CompoundSymmetric { rho } => {
                        format!("compound_symmetric={:016x}", rho.to_bits())
                    }
                    SequenceDependence::Autoregressive { rho } => {
                        format!("autoregressive={:016x}", rho.to_bits())
                    }
                    SequenceDependence::Unspecified => "unspecified".to_owned(),
                    _ => "other".to_owned(),
                };
                let periods = match model.periods {
                    PeriodAdjustment::None => "none",
                    PeriodAdjustment::FixedEffects => "fixed_effects",
                };
                let prior = &model.prior;
                format!(
                    "switchback;carryover={carryover};dependence={dependence};periods={periods};effect={:016x};nuisance={:016x};shape={:016x};scale={:016x}",
                    prior.effect_precision.to_bits(),
                    prior.nuisance_precision.to_bits(),
                    prior.variance_shape.to_bits(),
                    prior.variance_scale.to_bits()
                )
            }
            Self::QuantileGrid { thresholds, bin_concentration } => format!(
                "quantile_grid;thresholds={};concentration={:016x}",
                bits(thresholds),
                bin_concentration.to_bits()
            ),
            Self::DosePolynomial { support, coefficient_scales, residual } => {
                let residual = match residual {
                    ResidualPrior::KnownVariance(variance) => {
                        format!("known={:016x}", variance.to_bits())
                    }
                    ResidualPrior::InverseGamma { shape, scale } => {
                        format!("inverse_gamma={:016x},{:016x}", shape.to_bits(), scale.to_bits())
                    }
                };
                format!(
                    "dose_polynomial;support={};scales={};residual={residual}",
                    bits(&[support.0, support.1]),
                    bits(coefficient_scales)
                )
            }
        }
    }
}
