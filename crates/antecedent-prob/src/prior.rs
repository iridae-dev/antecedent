//! Prior specifications.
//!
//! Priors are recorded as assumptions; they do not create nonparametric
//! identification.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{PriorAssumption, VariableId};

use crate::error::ProbError;

/// Floor applied when converting effect-draw SD into a prior scale.
const EFFECT_PRIOR_SD_FLOOR: f64 = 1e-12;

/// Contrast coding for categorical predictors (required for Bayesian GLMs).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ContrastCoding {
    /// Treatment (dummy) coding with a designated reference level.
    Treatment,
    /// Sum (deviation) coding.
    Sum,
}

/// Gaussian prior on a scalar effect functional (e.g. ATE).
///
/// Used for cross-design transfer: moments come from source effect draws or
/// stored summaries, then map onto a target coefficient (identity-link bridge).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EffectPrior {
    /// Prior mean of the effect functional.
    pub mean: f64,
    /// Prior SD of the effect functional (must be finite and > 0).
    pub sd: f64,
}

impl EffectPrior {
    /// Construct from mean / SD with validation.
    ///
    /// # Errors
    ///
    /// Non-finite mean or non-positive / non-finite SD.
    pub fn new(mean: f64, sd: f64) -> Result<Self, ProbError> {
        let p = Self { mean, sd };
        p.validate()?;
        Ok(p)
    }

    /// Sample mean / SD from effect draws (population SD with Bessel correction when `n > 1`).
    ///
    /// SD is floored at a tiny positive value so conjugate scale stays valid.
    ///
    /// # Errors
    ///
    /// Empty draws or non-finite moments.
    pub fn from_effect_draws(draws: &[f64]) -> Result<Self, ProbError> {
        if draws.is_empty() {
            return Err(ProbError::InvalidPrior { message: "from_effect_draws: empty draws" });
        }
        let n = draws.len() as f64;
        let mean = draws.iter().sum::<f64>() / n;
        if !mean.is_finite() {
            return Err(ProbError::InvalidPrior { message: "from_effect_draws: non-finite mean" });
        }
        let sd = if draws.len() == 1 {
            EFFECT_PRIOR_SD_FLOOR
        } else {
            let var = draws
                .iter()
                .map(|&x| {
                    let d = x - mean;
                    d * d
                })
                .sum::<f64>()
                / (n - 1.0);
            var.sqrt().max(EFFECT_PRIOR_SD_FLOOR)
        };
        if !sd.is_finite() {
            return Err(ProbError::InvalidPrior { message: "from_effect_draws: non-finite sd" });
        }
        Self::new(mean, sd)
    }

    /// Validate finite mean and positive finite SD.
    ///
    /// # Errors
    ///
    /// Invalid parameters.
    pub fn validate(self) -> Result<(), ProbError> {
        if !self.mean.is_finite() {
            return Err(ProbError::InvalidPrior { message: "effect prior mean must be finite" });
        }
        if !(self.sd > 0.0) || !self.sd.is_finite() {
            return Err(ProbError::InvalidPrior {
                message: "effect prior sd must be finite and > 0",
            });
        }
        Ok(())
    }
}

/// Gaussian coefficient prior for conjugate / NIG linear models.
///
/// Under the conjugate Normal–Inv-Gamma (and known-σ² Normal) backends,
/// `variance[i]` is the diagonal entry of the *scale* matrix `V0` in
/// `β | σ² ~ N(mean, σ² · V0)` — not an absolute prior variance of `β`.
/// Absolute prior variance of coefficient `i` is therefore `σ² · variance[i]`.
/// `V0` is diagonal unless the enclosing [`PriorSet`] carries a
/// [`PriorSpec::CoefficientCorrelation`], which makes it dense
/// (see [`PriorSet::coefficient_precision`]).
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
    /// Non-finite mean, non-positive / non-finite variance, or zero coefficients.
    pub fn shared(n_coef: usize, mean: f64, variance: f64) -> Result<Self, ProbError> {
        if n_coef == 0 {
            return Err(ProbError::InvalidPrior { message: "n_coef must be > 0" });
        }
        if !mean.is_finite() {
            return Err(ProbError::InvalidPrior { message: "mean must be finite" });
        }
        if !(variance > 0.0) || !variance.is_finite() {
            return Err(ProbError::InvalidPrior { message: "variance must be finite and > 0" });
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

    /// Conjugate-scale precision `V0^{-1}` (diagonal): `1 / variance[i]`.
    ///
    /// This is **not** the absolute prior precision of `β`. For absolute
    /// precision use [`Self::absolute_precision`].
    #[must_use]
    pub fn precision(&self) -> Vec<f64> {
        self.variance.iter().map(|&v| 1.0 / v).collect()
    }

    /// Absolute prior variance of each coefficient: `σ² · variance[i]`.
    ///
    /// GLM / non-Gaussian Laplace and HMC treat the likelihood as having no
    /// residual scale; pass `sigma2 = 1.0` so absolute variance equals `V0`.
    ///
    /// # Errors
    ///
    /// Non-finite or non-positive `sigma2`, or an invalid prior.
    pub fn absolute_variance(&self, sigma2: f64) -> Result<Vec<f64>, ProbError> {
        self.validate()?;
        validate_sigma2(sigma2)?;
        Ok(self.variance.iter().map(|&v| v * sigma2).collect())
    }

    /// Absolute prior precision of each coefficient: `1 / (σ² · variance[i])`.
    ///
    /// # Errors
    ///
    /// Non-finite or non-positive `sigma2`, or an invalid prior.
    pub fn absolute_precision(&self, sigma2: f64) -> Result<Vec<f64>, ProbError> {
        self.validate()?;
        validate_sigma2(sigma2)?;
        Ok(self.variance.iter().map(|&v| 1.0 / (v * sigma2)).collect())
    }

    /// Build a coefficient prior from **absolute** variances by converting to `V0`.
    ///
    /// Sets `variance[i] = absolute_variance[i] / σ²`. Callers that hold posterior
    /// SD² (or any absolute Var(β)) must go through this rather than writing into
    /// [`Self::variance`] directly.
    ///
    /// # Errors
    ///
    /// Length mismatch, non-finite / non-positive absolute variances or `sigma2`.
    pub fn from_absolute_variance(
        mean: impl Into<Arc<[f64]>>,
        absolute_variance: impl Into<Arc<[f64]>>,
        sigma2: f64,
    ) -> Result<Self, ProbError> {
        validate_sigma2(sigma2)?;
        let mean = mean.into();
        let absolute_variance = absolute_variance.into();
        if mean.len() != absolute_variance.len() {
            return Err(ProbError::InvalidPrior {
                message: "mean and absolute_variance length mismatch",
            });
        }
        if mean.is_empty() {
            return Err(ProbError::InvalidPrior { message: "empty coefficient prior" });
        }
        let mut variance = Vec::with_capacity(absolute_variance.len());
        for &av in absolute_variance.iter() {
            if !(av > 0.0) || !av.is_finite() {
                return Err(ProbError::InvalidPrior {
                    message: "absolute variance must be finite and > 0",
                });
            }
            let v0 = av / sigma2;
            if !(v0 > 0.0) || !v0.is_finite() {
                return Err(ProbError::InvalidPrior {
                    message: "absolute_variance / sigma2 must be finite and > 0",
                });
            }
            variance.push(v0);
        }
        let out = Self { mean, variance: Arc::from(variance) };
        out.validate()?;
        Ok(out)
    }

    /// Validate lengths match and all mean / variance entries are finite.
    ///
    /// # Errors
    ///
    /// Length mismatch, empty prior, non-finite mean, or non-positive / non-finite variance.
    pub fn validate(&self) -> Result<(), ProbError> {
        if self.mean.len() != self.variance.len() {
            return Err(ProbError::InvalidPrior { message: "mean and variance length mismatch" });
        }
        if self.mean.is_empty() {
            return Err(ProbError::InvalidPrior { message: "empty coefficient prior" });
        }
        for &m in self.mean.iter() {
            if !m.is_finite() {
                return Err(ProbError::InvalidPrior { message: "mean must be finite" });
            }
        }
        for &v in self.variance.iter() {
            if !(v > 0.0) || !v.is_finite() {
                return Err(ProbError::InvalidPrior { message: "variance must be finite and > 0" });
            }
        }
        Ok(())
    }
}

fn validate_sigma2(sigma2: f64) -> Result<(), ProbError> {
    if !(sigma2 > 0.0) || !sigma2.is_finite() {
        return Err(ProbError::InvalidPrior {
            message: "sigma2 must be finite and > 0 for absolute↔V0 conversion",
        });
    }
    Ok(())
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
    /// Non-positive or non-finite shape or scale.
    pub fn validate(self) -> Result<(), ProbError> {
        if !(self.shape > 0.0)
            || !(self.scale > 0.0)
            || !self.shape.is_finite()
            || !self.scale.is_finite()
        {
            return Err(ProbError::InvalidPrior {
                message: "InvGamma shape and scale must be finite and > 0",
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
    /// Correlation of the coefficient prior's conjugate scale `V0`.
    ///
    /// Pairs with the set's [`Self::GaussianCoefficients`] entry, whose
    /// `variance` stays the diagonal of `V0`: the dense scale is
    /// `V0 = D^{1/2} R D^{1/2}` with `D = diag(variance)` and `R` this matrix.
    /// Absent, `V0` is diagonal.
    CoefficientCorrelation(CoefficientCorrelation),
}

/// Symmetric positive-definite correlation matrix (row-major `p × p`, unit
/// diagonal) of a Gaussian coefficient prior's conjugate scale `V0`.
///
/// Stored as a correlation rather than a covariance so the per-coefficient
/// diagonal in [`GaussianCoefficientPrior::variance`] keeps its meaning (and its
/// absolute ↔ `V0` conversions) whether or not the prior is dense.
#[derive(Clone, Debug, PartialEq)]
pub struct CoefficientCorrelation {
    dim: usize,
    matrix: Arc<[f64]>,
}

/// Off-diagonal correlations at or below this magnitude are treated as zero.
const CORRELATION_ZERO: f64 = 1e-12;

impl CoefficientCorrelation {
    /// Validate and wrap a row-major `dim × dim` correlation matrix.
    ///
    /// # Errors
    ///
    /// Wrong length, non-finite entries, a diagonal other than 1, asymmetry,
    /// an entry outside `[-1, 1]`, or a matrix that is not positive definite.
    pub fn new(dim: usize, matrix: impl Into<Arc<[f64]>>) -> Result<Self, ProbError> {
        let matrix = matrix.into();
        if dim == 0 || matrix.len() != dim.saturating_mul(dim) {
            return Err(ProbError::InvalidPrior {
                message: "coefficient correlation must be a non-empty dim × dim matrix",
            });
        }
        for i in 0..dim {
            if (matrix[i * dim + i] - 1.0).abs() > 1e-12 {
                return Err(ProbError::InvalidPrior {
                    message: "coefficient correlation must have a unit diagonal",
                });
            }
            for j in 0..dim {
                let r = matrix[i * dim + j];
                if !r.is_finite() || r.abs() > 1.0 + 1e-12 {
                    return Err(ProbError::InvalidPrior {
                        message: "coefficient correlation entries must be finite and in [-1, 1]",
                    });
                }
                if (r - matrix[j * dim + i]).abs() > 1e-12 {
                    return Err(ProbError::InvalidPrior {
                        message: "coefficient correlation must be symmetric",
                    });
                }
            }
        }
        crate::linalg::cholesky_spd(&matrix, dim).map_err(|_| ProbError::InvalidPrior {
            message: "coefficient correlation must be positive definite",
        })?;
        Ok(Self { dim, matrix })
    }

    /// Correlation of a symmetric positive-definite covariance (row-major).
    ///
    /// Returns `None` when every off-diagonal correlation is zero (the
    /// covariance is diagonal), so a diagonal prior keeps its diagonal form.
    ///
    /// # Errors
    ///
    /// Wrong length, a non-positive or non-finite diagonal, or a covariance
    /// whose correlation is not a valid positive-definite correlation.
    pub fn from_covariance(covariance: &[f64], dim: usize) -> Result<Option<Self>, ProbError> {
        if dim == 0 || covariance.len() != dim.saturating_mul(dim) {
            return Err(ProbError::InvalidPrior {
                message: "coefficient covariance must be a non-empty dim × dim matrix",
            });
        }
        let mut sd = Vec::with_capacity(dim);
        for i in 0..dim {
            let v = covariance[i * dim + i];
            if !(v > 0.0) || !v.is_finite() {
                return Err(ProbError::InvalidPrior {
                    message: "coefficient covariance diagonal must be finite and > 0",
                });
            }
            sd.push(v.sqrt());
        }
        let mut matrix = vec![0.0; dim * dim];
        let mut dense = false;
        for i in 0..dim {
            for j in 0..dim {
                let r = if i == j {
                    1.0
                } else {
                    // Symmetrize so roundoff in the source cannot fail validation.
                    0.5 * (covariance[i * dim + j] + covariance[j * dim + i]) / (sd[i] * sd[j])
                };
                if i != j && r.abs() > CORRELATION_ZERO {
                    dense = true;
                }
                matrix[i * dim + j] = if i != j && r.abs() <= CORRELATION_ZERO { 0.0 } else { r };
            }
        }
        if !dense {
            return Ok(None);
        }
        Self::new(dim, matrix).map(Some)
    }

    /// Dimension `p`.
    #[must_use]
    pub const fn dim(&self) -> usize {
        self.dim
    }

    /// Row-major `p × p` correlation matrix.
    #[must_use]
    pub fn matrix(&self) -> &[f64] {
        &self.matrix
    }

    /// Lower-triangular Cholesky factor `L` of the correlation (`R = L L'`,
    /// row-major), for drawing correlated standard normals `L z`.
    ///
    /// # Errors
    ///
    /// A correlation that is not numerically positive definite.
    pub fn cholesky(&self) -> Result<Vec<f64>, ProbError> {
        crate::linalg::cholesky_spd(&self.matrix, self.dim)
    }

    /// Dense conjugate scale `V0 = D^{1/2} R D^{1/2}` for diagonal `variance`.
    ///
    /// # Errors
    ///
    /// `variance` length differs from [`Self::dim`].
    pub fn scale_matrix(&self, variance: &[f64]) -> Result<Vec<f64>, ProbError> {
        if variance.len() != self.dim {
            return Err(ProbError::InvalidPrior {
                message: "coefficient correlation dimension != coefficient prior length",
            });
        }
        let n = self.dim;
        let sd: Vec<f64> = variance.iter().map(|v| v.sqrt()).collect();
        let mut v0 = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                v0[i * n + j] =
                    if i == j { variance[i] } else { sd[i] * sd[j] * self.matrix[i * n + j] };
            }
        }
        Ok(v0)
    }
}

/// Conjugate-scale prior precision `V0⁻¹` of a coefficient prior.
///
/// Diagonal unless the prior set carries a [`PriorSpec::CoefficientCorrelation`].
/// The diagonal arm reproduces the per-coefficient arithmetic the backends have
/// always used, so diagonal priors fit bit-for-bit as before.
#[derive(Clone, Debug, PartialEq)]
pub enum CoefficientPrecision {
    /// `1 / variance[i]` per coefficient.
    Diagonal(Vec<f64>),
    /// Dense row-major `p × p` precision.
    Dense {
        /// Dimension `p`.
        dim: usize,
        /// Row-major `V0⁻¹`.
        matrix: Vec<f64>,
    },
}

impl From<Vec<f64>> for CoefficientPrecision {
    fn from(diag: Vec<f64>) -> Self {
        Self::Diagonal(diag)
    }
}

impl CoefficientPrecision {
    /// Dimension `p`.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Diagonal(d) => d.len(),
            Self::Dense { dim, .. } => *dim,
        }
    }

    /// Whether `p == 0`.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether the precision has off-diagonal entries.
    #[must_use]
    pub const fn is_dense(&self) -> bool {
        matches!(self, Self::Dense { .. })
    }

    /// Entry `(i, j)`.
    #[must_use]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        match self {
            Self::Diagonal(d) => {
                if i == j {
                    d[i]
                } else {
                    0.0
                }
            }
            Self::Dense { dim, matrix } => matrix[i * dim + j],
        }
    }

    /// `out = P x`.
    pub fn mul_into(&self, x: &[f64], out: &mut [f64]) {
        match self {
            Self::Diagonal(d) => {
                for i in 0..d.len() {
                    out[i] = d[i] * x[i];
                }
            }
            Self::Dense { dim, matrix } => {
                for i in 0..*dim {
                    let mut acc = 0.0;
                    for j in 0..*dim {
                        acc += matrix[i * dim + j] * x[j];
                    }
                    out[i] = acc;
                }
            }
        }
    }

    /// `grad −= P (β − m)`.
    pub fn sub_prior_gradient(&self, beta: &[f64], mean: &[f64], grad: &mut [f64]) {
        match self {
            Self::Diagonal(d) => {
                for i in 0..d.len() {
                    let diff = beta[i] - mean[i];
                    grad[i] -= d[i] * diff;
                }
            }
            Self::Dense { dim, matrix } => {
                let diff: Vec<f64> = (0..*dim).map(|j| beta[j] - mean[j]).collect();
                for i in 0..*dim {
                    let mut acc = 0.0;
                    for j in 0..*dim {
                        acc += matrix[i * dim + j] * diff[j];
                    }
                    grad[i] -= acc;
                }
            }
        }
    }

    /// `m += P / divisor` on a row-major `n × n` matrix (`divisor = 1` adds `P`).
    pub fn add_divided_to(&self, m: &mut [f64], n: usize, divisor: f64) {
        match self {
            Self::Diagonal(d) => {
                for i in 0..d.len() {
                    m[i * n + i] += d[i] / divisor;
                }
            }
            Self::Dense { dim, matrix } => {
                for i in 0..*dim {
                    for j in 0..*dim {
                        m[i * n + j] += matrix[i * dim + j] / divisor;
                    }
                }
            }
        }
    }

    /// Log prior kernel `−½ (β − m)' P (β − m)`.
    #[must_use]
    pub fn log_kernel(&self, beta: &[f64], mean: &[f64]) -> f64 {
        match self {
            Self::Diagonal(d) => {
                let mut lp = 0.0;
                for i in 0..d.len() {
                    let diff = beta[i] - mean[i];
                    lp -= 0.5 * d[i] * diff * diff;
                }
                lp
            }
            Self::Dense { .. } => -0.5 * self.quadratic(beta, mean),
        }
    }

    /// Quadratic form `(β − m)' P (β − m)`.
    #[must_use]
    pub fn quadratic(&self, beta: &[f64], mean: &[f64]) -> f64 {
        match self {
            Self::Diagonal(d) => {
                let mut q = 0.0;
                for i in 0..d.len() {
                    let diff = beta[i] - mean[i];
                    q += d[i] * diff * diff;
                }
                q
            }
            Self::Dense { dim, matrix } => {
                let diff: Vec<f64> = (0..*dim).map(|j| beta[j] - mean[j]).collect();
                let mut q = 0.0;
                for i in 0..*dim {
                    let mut acc = 0.0;
                    for j in 0..*dim {
                        acc += matrix[i * dim + j] * diff[j];
                    }
                    q += diff[i] * acc;
                }
                q
            }
        }
    }
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
            Self::CoefficientCorrelation(_) => PriorAssumption {
                id: Arc::from("coefficient_correlation"),
                description: Arc::from(
                    "Dense coefficient prior scale V0 (off-diagonal prior covariance)",
                ),
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
            Self::CoefficientCorrelation(c) => {
                CoefficientCorrelation::new(c.dim, c.matrix.clone()).map(|_| ())
            }
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

/// Residual-variance model for Gaussian linear targets (HMC / Laplace).
///
/// Resolved once from [`PriorSet`] before sampling or optimization. At most one
/// residual specification may appear in the prior set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GaussianVarianceModel {
    /// Fixed known residual variance for the entire run.
    Known {
        /// Residual variance σ² > 0.
        sigma2: f64,
    },
    /// Inverse-gamma prior on σ²; HMC state includes `λ = log(σ²)`.
    InvGamma {
        /// Shape α₀ > 0.
        shape: f64,
        /// Scale β₀ > 0.
        scale: f64,
    },
}

impl GaussianVarianceModel {
    /// Resolve the residual model from a validated prior set.
    ///
    /// `KnownResidualVariance` → [`Self::Known`]; `ResidualInvGamma` →
    /// [`Self::InvGamma`]; an omitted residual specification defaults to
    /// [`InvGammaPrior::weakly_informative`].
    ///
    /// # Errors
    ///
    /// More than one residual specification, or invalid known / InvGamma params.
    pub fn from_prior_set(prior: &PriorSet) -> Result<Self, ProbError> {
        let mut known: Option<f64> = None;
        let mut inv_gamma: Option<InvGammaPrior> = None;
        for spec in &prior.specs {
            match spec {
                PriorSpec::KnownResidualVariance(v) => {
                    if known.is_some() || inv_gamma.is_some() {
                        return Err(ProbError::InvalidPrior {
                            message: "PriorSet must contain at most one residual variance specification",
                        });
                    }
                    known = Some(*v);
                }
                PriorSpec::ResidualInvGamma(p) => {
                    if known.is_some() || inv_gamma.is_some() {
                        return Err(ProbError::InvalidPrior {
                            message: "PriorSet must contain at most one residual variance specification",
                        });
                    }
                    inv_gamma = Some(*p);
                }
                PriorSpec::GaussianCoefficients(_) | PriorSpec::CoefficientCorrelation(_) => {}
            }
        }
        if let Some(sigma2) = known {
            if !(sigma2 > 0.0) || !sigma2.is_finite() {
                return Err(ProbError::InvalidPrior {
                    message: "known residual variance must be finite and > 0",
                });
            }
            return Ok(Self::Known { sigma2 });
        }
        let ig = inv_gamma.unwrap_or_else(InvGammaPrior::weakly_informative);
        ig.validate()?;
        Ok(Self::InvGamma { shape: ig.shape, scale: ig.scale })
    }

    /// Unconstrained state dimension for coefficients of length `ncols`.
    #[must_use]
    pub const fn state_dim(self, ncols: usize) -> usize {
        match self {
            Self::Known { .. } => ncols,
            Self::InvGamma { .. } => ncols.saturating_add(1),
        }
    }

    /// Whether draws include a residual-variance column.
    #[must_use]
    pub const fn include_sigma2(self) -> bool {
        matches!(self, Self::InvGamma { .. })
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
    /// Extra prior-restriction assumptions (e.g. external bank mapping ids).
    pub restrictions: Vec<PriorAssumption>,
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
            restrictions: Vec::new(),
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
    /// Invalid specs, missing contrast, more than one coefficient prior, or more
    /// than one residual variance specification (known σ² and InvGamma cannot
    /// both appear).
    pub fn validate(&self) -> Result<(), ProbError> {
        for s in &self.specs {
            s.validate()?;
        }
        self.validate_contrasts()?;
        let mut n_coef = 0usize;
        let mut n_residual = 0usize;
        let mut n_corr = 0usize;
        for s in &self.specs {
            match s {
                PriorSpec::GaussianCoefficients(_) => {
                    n_coef = n_coef.saturating_add(1);
                }
                PriorSpec::ResidualInvGamma(_) | PriorSpec::KnownResidualVariance(_) => {
                    n_residual = n_residual.saturating_add(1);
                }
                PriorSpec::CoefficientCorrelation(_) => {
                    n_corr = n_corr.saturating_add(1);
                }
            }
        }
        if n_coef > 1 {
            return Err(ProbError::InvalidPrior {
                message: "PriorSet must contain at most one coefficient prior",
            });
        }
        if n_corr > 1 {
            return Err(ProbError::InvalidPrior {
                message: "PriorSet must contain at most one coefficient correlation",
            });
        }
        if let Some(corr) = self.coefficient_correlation() {
            match self.gaussian_coefficients() {
                Some(coef) if coef.len() == corr.dim() => {}
                Some(_) => {
                    return Err(ProbError::InvalidPrior {
                        message: "coefficient correlation dimension != coefficient prior length",
                    });
                }
                None => {
                    return Err(ProbError::InvalidPrior {
                        message: "coefficient correlation without a Gaussian coefficient prior",
                    });
                }
            }
        }
        if n_residual > 1 {
            return Err(ProbError::InvalidPrior {
                message: "PriorSet must contain at most one residual variance specification",
            });
        }
        Ok(())
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

    /// Correlation of the coefficient prior's conjugate scale `V0`, if dense.
    #[must_use]
    pub fn coefficient_correlation(&self) -> Option<&CoefficientCorrelation> {
        self.specs.iter().find_map(|s| match s {
            PriorSpec::CoefficientCorrelation(c) => Some(c),
            _ => None,
        })
    }

    /// Conjugate-scale precision `V0⁻¹` of `coef` under this set's correlation.
    ///
    /// `coef` is this set's coefficient prior (or the backend default when the
    /// set has none, in which case it carries no correlation). Diagonal when
    /// the set carries no [`PriorSpec::CoefficientCorrelation`].
    ///
    /// # Errors
    ///
    /// A correlation whose dimension differs from `coef`, or a dense `V0`
    /// that is not positive definite.
    pub fn coefficient_precision(
        &self,
        coef: &GaussianCoefficientPrior,
    ) -> Result<CoefficientPrecision, ProbError> {
        let Some(corr) = self.coefficient_correlation() else {
            return Ok(CoefficientPrecision::Diagonal(coef.precision()));
        };
        let v0 = corr.scale_matrix(&coef.variance)?;
        let n = corr.dim();
        let matrix = crate::linalg::invert_spd(&v0, n).map_err(|_| ProbError::InvalidPrior {
            message: "dense coefficient prior scale V0 is not positive definite",
        })?;
        Ok(CoefficientPrecision::Dense { dim: n, matrix })
    }

    /// Mark coefficient `indices` as holding **absolute** prior variances.
    ///
    /// A prior hydrated from a source posterior that records no residual
    /// variance cannot be converted to the conjugate `V0` scale at hydrate time.
    /// The marker is a [`PriorAssumption`] in [`Self::restrictions`] whose id is
    /// `absolute_coefficient_scale:<i>,<j>,…`; a Gaussian fit resolves it with
    /// the target's residual-variance estimate
    /// ([`Self::resolve_absolute_coefficient_scale`]); GLM fits read `V0` at
    /// `σ² ≡ 1`, so absolute and `V0` coincide there. Marking merges with an
    /// existing marker.
    pub fn mark_absolute_coefficient_scale(&mut self, indices: &[usize]) {
        let mut all = self.absolute_coefficient_scale().unwrap_or_default();
        all.extend_from_slice(indices);
        all.sort_unstable();
        all.dedup();
        self.restrictions.retain(|r| absolute_scale_indices(&r.id).is_none());
        if all.is_empty() {
            return;
        }
        let list = all.iter().map(ToString::to_string).collect::<Vec<_>>().join(",");
        self.restrictions.push(PriorAssumption {
            id: Arc::from(format!("{ABSOLUTE_COEFFICIENT_SCALE_ID}:{list}")),
            description: Arc::from(format!(
                "coefficient prior variances at [{list}] are absolute Var(beta) from a source \
                 posterior that records no residual variance; a Gaussian target converts them \
                 to its conjugate V0 with its own residual-variance estimate (plug-in), a GLM \
                 target uses them directly (sigma^2 = 1)"
            )),
        });
    }

    /// Coefficient indices whose prior variances are absolute, if marked.
    #[must_use]
    pub fn absolute_coefficient_scale(&self) -> Option<Vec<usize>> {
        let mut out: Option<Vec<usize>> = None;
        for r in &self.restrictions {
            if let Some(idx) = absolute_scale_indices(&r.id) {
                out.get_or_insert_with(Vec::new).extend(idx);
            }
        }
        out
    }

    /// Convert marked absolute coefficient variances to `V0 = absolute / σ²`.
    ///
    /// Returns `None` when the prior carries no marker (callers keep the prior
    /// as-is). The returned prior has the marker removed, so resolution is
    /// applied once.
    ///
    /// # Errors
    ///
    /// Non-finite / non-positive `sigma2`, a marked index outside the
    /// coefficient prior, or a marker without a coefficient prior.
    pub fn resolve_absolute_coefficient_scale(
        &self,
        sigma2: f64,
    ) -> Result<Option<Self>, ProbError> {
        let Some(indices) = self.absolute_coefficient_scale() else {
            return Ok(None);
        };
        validate_sigma2(sigma2)?;
        let mut out = self.clone();
        out.restrictions.retain(|r| absolute_scale_indices(&r.id).is_none());
        let coef = out.specs.iter_mut().find_map(|s| match s {
            PriorSpec::GaussianCoefficients(p) => Some(p),
            _ => None,
        });
        let Some(coef) = coef else {
            return Err(ProbError::InvalidPrior {
                message: "absolute coefficient scale marker without a coefficient prior",
            });
        };
        let mut variance = coef.variance.to_vec();
        for &i in &indices {
            let Some(v) = variance.get_mut(i) else {
                return Err(ProbError::InvalidPrior {
                    message: "absolute coefficient scale marker index out of range",
                });
            };
            *v /= sigma2;
        }
        coef.variance = Arc::from(variance);
        coef.validate()?;
        Ok(Some(out))
    }
}

/// Restriction-id prefix of the absolute-coefficient-scale marker
/// ([`PriorSet::mark_absolute_coefficient_scale`]).
pub const ABSOLUTE_COEFFICIENT_SCALE_ID: &str = "absolute_coefficient_scale";

fn absolute_scale_indices(id: &str) -> Option<Vec<usize>> {
    let list = id.strip_prefix(ABSOLUTE_COEFFICIENT_SCALE_ID)?.strip_prefix(':')?;
    list.split(',').map(|s| s.parse::<usize>().ok()).collect()
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

    #[test]
    fn effect_prior_from_draws_moments() {
        let draws = [1.0, 3.0, 5.0];
        let p = EffectPrior::from_effect_draws(&draws).unwrap();
        assert!((p.mean - 3.0).abs() < 1e-12);
        // sample sd of [1,3,5] = 2
        assert!((p.sd - 2.0).abs() < 1e-12);
    }

    #[test]
    fn effect_prior_rejects_empty_and_nonfinite() {
        assert!(EffectPrior::from_effect_draws(&[]).is_err());
        assert!(EffectPrior::new(f64::NAN, 1.0).is_err());
        assert!(EffectPrior::new(0.0, 0.0).is_err());
        assert!(EffectPrior::new(0.0, -1.0).is_err());
    }

    #[test]
    fn effect_prior_single_draw_floors_sd() {
        let p = EffectPrior::from_effect_draws(&[2.5]).unwrap();
        assert!((p.mean - 2.5).abs() < 1e-12);
        assert!(p.sd > 0.0);
    }

    #[test]
    fn residual_specs_must_be_unique() {
        let mut p = PriorSet::new();
        p.push(PriorSpec::GaussianCoefficients(GaussianCoefficientPrior::isotropic(1, 1.0)));
        p.push(PriorSpec::KnownResidualVariance(1.0));
        p.push(PriorSpec::ResidualInvGamma(InvGammaPrior::weakly_informative()));
        assert!(p.validate().is_err());
        assert!(GaussianVarianceModel::from_prior_set(&p).is_err());
    }

    #[test]
    fn coefficient_priors_must_be_unique() {
        let mut p = PriorSet::new();
        p.push(PriorSpec::GaussianCoefficients(GaussianCoefficientPrior::isotropic(1, 1.0)));
        p.push(PriorSpec::GaussianCoefficients(GaussianCoefficientPrior::isotropic(1, 2.0)));
        assert!(p.validate().is_err());
    }

    #[test]
    fn coefficient_prior_rejects_nonfinite_mean() {
        let prior = GaussianCoefficientPrior {
            mean: Arc::from(vec![f64::NAN]),
            variance: Arc::from(vec![1.0]),
        };
        assert!(prior.validate().is_err());
        assert!(GaussianCoefficientPrior::shared(1, f64::INFINITY, 1.0).is_err());
    }

    #[test]
    fn absolute_variance_round_trips_through_v0() {
        let prior = GaussianCoefficientPrior::from_absolute_variance(
            Arc::from(vec![1.0_f64, -0.5]),
            Arc::from(vec![0.25_f64, 1.0]),
            0.25,
        )
        .unwrap();
        // V0 = abs / σ² ⇒ 0.25/0.25 = 1, 1/0.25 = 4
        assert!((prior.variance[0] - 1.0).abs() < 1e-15);
        assert!((prior.variance[1] - 4.0).abs() < 1e-15);
        let abs = prior.absolute_variance(0.25).unwrap();
        assert!((abs[0] - 0.25).abs() < 1e-15);
        assert!((abs[1] - 1.0).abs() < 1e-15);
        let prec = prior.absolute_precision(0.25).unwrap();
        assert!((prec[0] - 4.0).abs() < 1e-15);
        assert!((prec[1] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn variance_model_defaults_to_weak_inv_gamma() {
        let mut p = PriorSet::new();
        p.push(PriorSpec::GaussianCoefficients(GaussianCoefficientPrior::isotropic(2, 1.0)));
        p.validate().unwrap();
        let model = GaussianVarianceModel::from_prior_set(&p).unwrap();
        let weak = InvGammaPrior::weakly_informative();
        assert_eq!(model, GaussianVarianceModel::InvGamma { shape: weak.shape, scale: weak.scale });
        assert_eq!(model.state_dim(2), 3);
        assert!(model.include_sigma2());
    }

    #[test]
    fn variance_model_known() {
        let mut p = PriorSet::new();
        p.push(PriorSpec::KnownResidualVariance(2.5));
        let model = GaussianVarianceModel::from_prior_set(&p).unwrap();
        assert_eq!(model, GaussianVarianceModel::Known { sigma2: 2.5 });
        assert_eq!(model.state_dim(4), 4);
        assert!(!model.include_sigma2());
    }

    fn dense_set() -> PriorSet {
        let mut p = PriorSet::new();
        p.push(PriorSpec::GaussianCoefficients(GaussianCoefficientPrior {
            mean: Arc::from(vec![0.5, -1.0]),
            variance: Arc::from(vec![4.0, 0.25]),
        }));
        p.push(PriorSpec::CoefficientCorrelation(
            CoefficientCorrelation::new(2, vec![1.0, 0.6, 0.6, 1.0]).unwrap(),
        ));
        p
    }

    #[test]
    fn coefficient_correlation_validates_and_pairs_with_the_coefficient_prior() {
        dense_set().validate().unwrap();
        assert!(CoefficientCorrelation::new(2, vec![1.0, 0.5, 0.4, 1.0]).is_err(), "asymmetric");
        assert!(CoefficientCorrelation::new(2, vec![2.0, 0.0, 0.0, 1.0]).is_err(), "diagonal");
        assert!(CoefficientCorrelation::new(2, vec![1.0, 1.0, 1.0, 1.0]).is_err(), "singular");
        assert!(CoefficientCorrelation::new(3, vec![1.0; 4]).is_err(), "shape");
        let mut orphan = PriorSet::new();
        orphan.push(dense_set().specs[1].clone());
        assert!(orphan.validate().is_err(), "correlation without coefficients");
        let mut mismatch = PriorSet::weakly_informative(3);
        mismatch.push(dense_set().specs[1].clone());
        assert!(mismatch.validate().is_err(), "dimension mismatch");
        assert!(
            CoefficientCorrelation::from_covariance(&[2.0, 0.0, 0.0, 3.0], 2).unwrap().is_none(),
            "a diagonal covariance keeps the diagonal form"
        );
    }

    #[test]
    fn dense_precision_matches_the_inverse_scale_matrix() {
        let set = dense_set();
        let coef = set.gaussian_coefficients().unwrap();
        let prec = set.coefficient_precision(coef).unwrap();
        assert!(prec.is_dense());
        // V0 = [[4, 0.6·2·0.5], [0.6, 0.25]] → V0⁻¹ = adj / det.
        let (a, b, d) = (4.0, 0.6, 0.25);
        let det = a * d - b * b;
        let expected = [d / det, -b / det, -b / det, a / det];
        for i in 0..2 {
            for j in 0..2 {
                assert!((prec.get(i, j) - expected[i * 2 + j]).abs() < 1e-12);
            }
        }
        let beta = [1.0, 0.0];
        let diff = [0.5, 1.0];
        let q = diff[0] * (expected[0] * diff[0] + expected[1] * diff[1])
            + diff[1] * (expected[2] * diff[0] + expected[3] * diff[1]);
        assert!((prec.quadratic(&beta, &coef.mean) - q).abs() < 1e-12);
        assert!((prec.log_kernel(&beta, &coef.mean) + 0.5 * q).abs() < 1e-12);
        let mut grad = [0.0, 0.0];
        prec.sub_prior_gradient(&beta, &coef.mean, &mut grad);
        assert!((grad[0] + expected[0] * diff[0] + expected[1] * diff[1]).abs() < 1e-12);
        assert!((grad[1] + expected[2] * diff[0] + expected[3] * diff[1]).abs() < 1e-12);
        // Without a correlation the precision is the per-coefficient diagonal.
        let diag = PriorSet::weakly_informative(2);
        let p = diag.coefficient_precision(diag.gaussian_coefficients().unwrap()).unwrap();
        assert_eq!(p, CoefficientPrecision::Diagonal(vec![0.01, 0.01]));
    }

    /// Absolute-scale resolution divides the diagonal and keeps the correlation,
    /// so the dense `V0` is the absolute covariance over `σ²`.
    #[test]
    fn absolute_scale_resolution_keeps_the_correlation() {
        let mut set = dense_set();
        set.mark_absolute_coefficient_scale(&[0, 1]);
        let resolved = set.resolve_absolute_coefficient_scale(4.0).unwrap().unwrap();
        let coef = resolved.gaussian_coefficients().unwrap();
        assert_eq!(coef.variance.as_ref(), &[1.0, 0.0625]);
        assert_eq!(resolved.coefficient_correlation(), set.coefficient_correlation());
    }
}
