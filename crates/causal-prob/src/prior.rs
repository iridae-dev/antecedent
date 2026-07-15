//! Prior specifications (DESIGN.md §14.4).
//!
//! Priors are recorded as assumptions; they do not create nonparametric
//! identification (DESIGN.md rule 4 / ADR 0006).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::{PriorAssumption, VariableId};

use crate::error::ProbError;

/// Contrast coding for categorical predictors (required for Bayesian GLMs).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ContrastCoding {
    /// Treatment (dummy) coding with a designated reference level.
    Treatment,
    /// Sum (deviation) coding.
    Sum,
}

/// Gaussian coefficient prior for conjugate / NIG linear models.
///
/// Under the conjugate Normal–Inv-Gamma (and known-σ² Normal) backends,
/// `variance[i]` is the diagonal entry of the *scale* matrix `V0` in
/// `β | σ² ~ N(mean, σ² · diag(V0))` — not an absolute prior variance of `β`.
/// Absolute prior variance of coefficient `i` is therefore `σ² · variance[i]`.
#[derive(Clone, Debug, PartialEq)]
pub struct GaussianCoefficientPrior {
    /// Prior mean per coefficient (length = p), or a single shared mean.
    pub mean: Arc<[f64]>,
    /// Diagonal of conjugate scale `V0` (length = p); see struct docs.
    pub variance: Arc<[f64]>,
}

impl GaussianCoefficientPrior {
    /// Isotropic weakly informative prior: mean 0, V0 diagonal `scale²`
    /// (absolute prior variance of β is `σ² · scale²` under conjugate models).
    #[must_use]
    pub fn isotropic(n_coef: usize, scale: f64) -> Self {
        let var = scale * scale;
        Self { mean: Arc::from(vec![0.0; n_coef]), variance: Arc::from(vec![var; n_coef]) }
    }

    /// Shared mean / V0-diagonal broadcast to `n_coef` coefficients.
    ///
    /// # Errors
    ///
    /// Non-positive variance or zero coefficients.
    pub fn shared(n_coef: usize, mean: f64, variance: f64) -> Result<Self, ProbError> {
        if n_coef == 0 {
            return Err(ProbError::InvalidPrior { message: "n_coef must be > 0" });
        }
        if !(variance > 0.0) {
            return Err(ProbError::InvalidPrior { message: "variance must be > 0" });
        }
        Ok(Self {
            mean: Arc::from(vec![mean; n_coef]),
            variance: Arc::from(vec![variance; n_coef]),
        })
    }

    /// Number of coefficients.
    #[must_use]
    pub fn len(&self) -> usize {
        self.mean.len()
    }

    /// Whether empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.mean.is_empty()
    }

    /// Precision (1/variance) vector.
    #[must_use]
    pub fn precision(&self) -> Vec<f64> {
        self.variance.iter().map(|&v| 1.0 / v).collect()
    }

    /// Validate lengths match.
    ///
    /// # Errors
    ///
    /// Length mismatch or non-positive variance.
    pub fn validate(&self) -> Result<(), ProbError> {
        if self.mean.len() != self.variance.len() {
            return Err(ProbError::InvalidPrior { message: "mean and variance length mismatch" });
        }
        if self.mean.is_empty() {
            return Err(ProbError::InvalidPrior { message: "empty coefficient prior" });
        }
        for &v in self.variance.iter() {
            if !(v > 0.0) && v.is_finite() {
                return Err(ProbError::InvalidPrior { message: "variance must be > 0" });
            }
            if !v.is_finite() {
                return Err(ProbError::InvalidPrior { message: "variance must be finite" });
            }
        }
        Ok(())
    }
}

/// Inv-Gamma prior on residual variance (conjugate Gaussian linear).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InvGammaPrior {
    /// Shape α > 0.
    pub shape: f64,
    /// Scale β > 0 (mean = β/(α−1) for α > 1).
    pub scale: f64,
}

impl InvGammaPrior {
    /// Weakly informative default.
    #[must_use]
    pub const fn weakly_informative() -> Self {
        Self { shape: 1e-3, scale: 1e-3 }
    }

    /// Validate.
    ///
    /// # Errors
    ///
    /// Non-positive shape or scale.
    pub fn validate(self) -> Result<(), ProbError> {
        if !(self.shape > 0.0) || !(self.scale > 0.0) {
            return Err(ProbError::InvalidPrior {
                message: "InvGamma shape and scale must be > 0",
            });
        }
        Ok(())
    }
}

/// A named prior specification entry.
#[derive(Clone, Debug, PartialEq)]
pub enum PriorSpec {
    /// Gaussian coefficient prior for a linear / GLM mechanism.
    GaussianCoefficients(GaussianCoefficientPrior),
    /// Residual variance prior (conjugate Gaussian).
    ResidualInvGamma(InvGammaPrior),
    /// Fixed residual variance (known σ²).
    KnownResidualVariance(f64),
}

impl PriorSpec {
    /// Convert to a [`PriorAssumption`] for the assumption record.
    #[must_use]
    pub fn as_assumption(&self) -> PriorAssumption {
        match self {
            Self::GaussianCoefficients(_) => PriorAssumption {
                id: Arc::from("gaussian_coefficients"),
                description: Arc::from("Gaussian prior on regression coefficients"),
            },
            Self::ResidualInvGamma(_) => PriorAssumption {
                id: Arc::from("residual_inv_gamma"),
                description: Arc::from("Inverse-Gamma prior on residual variance"),
            },
            Self::KnownResidualVariance(_) => PriorAssumption {
                id: Arc::from("known_residual_variance"),
                description: Arc::from("Known residual variance (no prior uncertainty)"),
            },
        }
    }

    /// Validate this prior.
    ///
    /// # Errors
    ///
    /// Invalid parameters.
    pub fn validate(&self) -> Result<(), ProbError> {
        match self {
            Self::GaussianCoefficients(p) => p.validate(),
            Self::ResidualInvGamma(p) => p.validate(),
            Self::KnownResidualVariance(v) => {
                if !(*v > 0.0) || !v.is_finite() {
                    return Err(ProbError::InvalidPrior {
                        message: "known residual variance must be finite and > 0",
                    });
                }
                Ok(())
            }
        }
    }
}

/// Collection of priors for an inference run.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PriorSet {
    /// Ordered prior entries.
    pub specs: Vec<PriorSpec>,
    /// Explicit contrast coding when categorical predictors are present.
    pub contrast: Option<ContrastCoding>,
    /// Variables that are categorical and require the declared contrast.
    pub categorical: Vec<VariableId>,
}

impl PriorSet {
    /// Empty set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Weakly informative Gaussian coefficient prior of width `scale` for `n_coef`.
    #[must_use]
    pub fn weakly_informative(n_coef: usize) -> Self {
        Self {
            specs: vec![
                PriorSpec::GaussianCoefficients(GaussianCoefficientPrior::isotropic(n_coef, 10.0)),
                PriorSpec::ResidualInvGamma(InvGammaPrior::weakly_informative()),
            ],
            contrast: None,
            categorical: Vec::new(),
        }
    }

    /// Push a prior spec.
    pub fn push(&mut self, spec: PriorSpec) {
        self.specs.push(spec);
    }

    /// Require an explicit contrast when categoricals are present.
    ///
    /// # Errors
    ///
    /// Categoricals listed without a contrast coding.
    pub fn validate_contrasts(&self) -> Result<(), ProbError> {
        if !self.categorical.is_empty() && self.contrast.is_none() {
            return Err(ProbError::InvalidPrior {
                message: "categorical predictors require explicit contrast coding",
            });
        }
        Ok(())
    }

    /// Validate all specs.
    ///
    /// # Errors
    ///
    /// Invalid specs or missing contrast.
    pub fn validate(&self) -> Result<(), ProbError> {
        for s in &self.specs {
            s.validate()?;
        }
        self.validate_contrasts()
    }

    /// First Gaussian coefficient prior, if any.
    #[must_use]
    pub fn gaussian_coefficients(&self) -> Option<&GaussianCoefficientPrior> {
        self.specs.iter().find_map(|s| match s {
            PriorSpec::GaussianCoefficients(p) => Some(p),
            _ => None,
        })
    }

    /// Residual Inv-Gamma prior, if any.
    #[must_use]
    pub fn residual_inv_gamma(&self) -> Option<InvGammaPrior> {
        self.specs.iter().find_map(|s| match s {
            PriorSpec::ResidualInvGamma(p) => Some(*p),
            _ => None,
        })
    }

    /// Known residual variance, if any.
    #[must_use]
    pub fn known_residual_variance(&self) -> Option<f64> {
        self.specs.iter().find_map(|s| match s {
            PriorSpec::KnownResidualVariance(v) => Some(*v),
            _ => None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weakly_informative_validates() {
        let p = PriorSet::weakly_informative(3);
        p.validate().unwrap();
        assert_eq!(p.gaussian_coefficients().unwrap().len(), 3);
    }

    #[test]
    fn categorical_requires_contrast() {
        let mut p = PriorSet::weakly_informative(2);
        p.categorical.push(VariableId::from_raw(0));
        assert!(p.validate().is_err());
        p.contrast = Some(ContrastCoding::Treatment);
        p.validate().unwrap();
    }
}
