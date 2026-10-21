//! Prior update of a Gaussian linear model: sequential equals batch, with no double use.
//!
//! The model is `y | x, β, σ² ~ N(xᵀβ, σ²)` with the conjugate normal-inverse-gamma prior
//! `β | σ² ~ N(m₀, σ² Λ₀⁻¹)`, `σ² ~ InvGamma(a₀, b₀)`. The posterior depends on the data only
//! through the additive terms `XᵀX`, `Xᵀy`, `yᵀy` and `n`, so a posterior kept as the natural
//! parameters `Λ = Λ₀ + XᵀX`, `η = Λ₀m₀ + Xᵀy`, `c = m₀ᵀΛ₀m₀ + yᵀy`, `a = a₀ + n/2` is updated
//! by adding a block's terms, and updating block by block gives the same posterior as one
//! update with every block pooled. The coefficient posterior mean is `Λ⁻¹η` and
//! `b = b₀ + (c − ηᵀΛ⁻¹η)/2`.
//!
//! Conjugate updating is only valid when each observation enters the likelihood once:
//!
//! * a block is identified by a caller label and the digest of its terms; a block whose label
//!   was already used is refused, whether its terms are the same (double use) or different
//!   (a relabelled source);
//! * a prior built from a posterior ([`ConjugateOlsPosterior::to_prior`]) remembers the blocks
//!   it already used, so updating it with one of them again is the same refusal;
//! * a posterior answers only under the prior it was built with. A different prior is a
//!   different posterior: [`ConjugateOlsPosterior::update`] and
//!   [`ConjugateOlsPosterior::summary`] refuse it, and [`ConjugateOlsPosterior::reprior`] is the
//!   rebuild from every own block under the new prior.
//!
//! The model is a stated likelihood and prior, not a distribution-free interval; the prior
//! alone identifies no causal target.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;
use std::sync::Arc;

use antecedent_core::CanonicalWriter;
use antecedent_stats::invert_square;

use crate::error::StateError;
use crate::suff_stats::LinearOlsSuffStats;

const PRIOR_TAG: &str = "antecedent.state.conjugate_refresh.prior";
const BLOCK_TAG: &str = "antecedent.state.conjugate_refresh.block";

/// Normal-inverse-gamma prior of a Gaussian linear model.
#[derive(Clone, Debug, PartialEq)]
pub struct ConjugateOlsPrior {
    mean: Vec<f64>,
    precision: Vec<f64>,
    shape: f64,
    rate: f64,
    used: BTreeMap<Arc<str>, String>,
}

impl ConjugateOlsPrior {
    /// A prior with mean `m₀`, row-major precision `Λ₀` (`β | σ² ~ N(m₀, σ² Λ₀⁻¹)`), and
    /// `σ² ~ InvGamma(shape, rate)`.
    ///
    /// # Errors
    /// A non-finite entry, a non-symmetric or singular precision, or a non-positive shape or
    /// rate.
    pub fn new(
        mean: Vec<f64>,
        precision: Vec<f64>,
        shape: f64,
        rate: f64,
    ) -> Result<Self, StateError> {
        let p = mean.len();
        if p == 0 || precision.len() != p * p {
            return Err(StateError::Shape("prior mean and precision disagree".into()));
        }
        let finite = mean.iter().chain(&precision).all(|value| value.is_finite());
        if !finite || !positive(shape) || !positive(rate) {
            return Err(StateError::InvalidEvent(
                "prior needs finite entries and positive shape and rate".into(),
            ));
        }
        for i in 0..p {
            for j in 0..i {
                let upper = precision.get(j * p + i).copied().unwrap_or(f64::NAN);
                let lower = precision.get(i * p + j).copied().unwrap_or(f64::NAN);
                if (upper - lower).abs() > 1e-12 * (1.0 + upper.abs()) {
                    return Err(StateError::InvalidEvent(
                        "prior precision is not symmetric".into(),
                    ));
                }
            }
        }
        if invert_square(&precision, p).is_none() {
            return Err(StateError::InvalidEvent("prior precision is singular".into()));
        }
        Ok(Self { mean, precision, shape, rate, used: BTreeMap::new() })
    }

    /// Independent coefficients with prior variance `variance` (in units of `σ²`) around zero.
    ///
    /// # Errors
    /// `variance` is not positive, or the errors of [`Self::new`].
    pub fn isotropic(p: usize, variance: f64, shape: f64, rate: f64) -> Result<Self, StateError> {
        if !positive(variance) {
            return Err(StateError::InvalidEvent("prior variance must be positive".into()));
        }
        let mut precision = vec![0.0; p * p];
        for i in 0..p {
            if let Some(slot) = precision.get_mut(i * p + i) {
                *slot = 1.0 / variance;
            }
        }
        Self::new(vec![0.0; p], precision, shape, rate)
    }

    /// Content digest of the prior, including the blocks it already used.
    #[must_use]
    pub fn digest(&self) -> String {
        let mut writer = CanonicalWriter::new(PRIOR_TAG);
        writer.reals(&self.mean).reals(&self.precision).real(self.shape).real(self.rate);
        writer.word(self.used.len() as u64);
        for (label, digest) in &self.used {
            writer.text(label).text(digest);
        }
        writer.finish_hex()
    }

    /// Labels of the blocks this prior already used (empty for a prior stated up front).
    pub fn used_labels(&self) -> impl Iterator<Item = &str> {
        self.used.keys().map(AsRef::as_ref)
    }

    /// Number of coefficients.
    #[must_use]
    pub fn ncols(&self) -> usize {
        self.mean.len()
    }

    /// Prior mean `m₀`.
    #[must_use]
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    /// Row-major prior precision `Λ₀`.
    #[must_use]
    pub fn precision(&self) -> &[f64] {
        &self.precision
    }

    /// Inverse-gamma shape of the innovation variance.
    #[must_use]
    pub const fn shape(&self) -> f64 {
        self.shape
    }

    /// Inverse-gamma rate of the innovation variance.
    #[must_use]
    pub const fn rate(&self) -> f64 {
        self.rate
    }
}

/// Posterior summary of the model.
#[derive(Clone, Debug, PartialEq)]
pub struct ConjugateOlsSummary {
    /// Posterior mean of the coefficients, `Λ⁻¹η`.
    pub mean: Vec<f64>,
    /// Row-major posterior precision `Λ` (`β | σ² ~ N(mean, σ² Λ⁻¹)`).
    pub precision: Vec<f64>,
    /// Row-major `Λ⁻¹`.
    pub scale: Vec<f64>,
    /// Posterior shape of `σ²`.
    pub shape: f64,
    /// Posterior rate of `σ²`.
    pub rate: f64,
    /// Observations used by this posterior since its prior.
    pub n: u64,
}

impl ConjugateOlsSummary {
    /// Posterior mean of `σ²`, `rate / (shape − 1)`, when `shape > 1`.
    #[must_use]
    pub fn residual_variance_mean(&self) -> Option<f64> {
        (self.shape > 1.0).then(|| self.rate / (self.shape - 1.0))
    }
}

/// Natural parameters of the posterior and the blocks that built it.
#[derive(Clone, Debug)]
pub struct ConjugateOlsPosterior {
    prior: ConjugateOlsPrior,
    prior_digest: String,
    precision: Vec<f64>,
    eta: Vec<f64>,
    sum_squares: f64,
    shape: f64,
    n: u64,
    used: BTreeMap<Arc<str>, String>,
}

fn positive(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

fn block_digest(block: &LinearOlsSuffStats) -> String {
    let mut writer = CanonicalWriter::new(BLOCK_TAG);
    writer.word(block.n()).reals(block.xtx()).reals(block.xty()).real(block.yty());
    writer.finish_hex()
}

fn quadratic(matrix: &[f64], p: usize, vector: &[f64]) -> f64 {
    let mut total = 0.0;
    for i in 0..p {
        for j in 0..p {
            let entry = matrix.get(i * p + j).copied().unwrap_or(0.0);
            let left = vector.get(i).copied().unwrap_or(0.0);
            let right = vector.get(j).copied().unwrap_or(0.0);
            total += left * entry * right;
        }
    }
    total
}

fn multiply(matrix: &[f64], p: usize, vector: &[f64]) -> Vec<f64> {
    (0..p)
        .map(|i| {
            (0..p)
                .map(|j| {
                    matrix.get(i * p + j).copied().unwrap_or(0.0)
                        * vector.get(j).copied().unwrap_or(0.0)
                })
                .sum()
        })
        .collect()
}

impl ConjugateOlsPosterior {
    /// The posterior of no observations: the prior itself.
    #[must_use]
    pub fn new(prior: &ConjugateOlsPrior) -> Self {
        let p = prior.ncols();
        let eta = multiply(&prior.precision, p, &prior.mean);
        let sum_squares = quadratic(&prior.precision, p, &prior.mean);
        Self {
            prior: prior.clone(),
            prior_digest: prior.digest(),
            precision: prior.precision.clone(),
            eta,
            sum_squares,
            shape: prior.shape,
            n: 0,
            used: prior.used.clone(),
        }
    }

    /// The prior this posterior answers under.
    #[must_use]
    pub const fn prior(&self) -> &ConjugateOlsPrior {
        &self.prior
    }

    fn check_prior(&self, prior: &ConjugateOlsPrior) -> Result<(), StateError> {
        if prior.digest() == self.prior_digest {
            return Ok(());
        }
        Err(StateError::StaleCache(
            "the posterior was built under another prior; a different prior is a different \
             posterior, so rebuild from every retained block under the new prior"
                .into(),
        ))
    }

    /// Add one block of observations under `prior`, which must be the prior this posterior
    /// was built with.
    ///
    /// # Errors
    /// A different prior, an invalidated or empty block, a column mismatch, or a block whose
    /// label was already used (by this posterior or by the prior it started from). Nothing
    /// changes.
    pub fn update(
        &mut self,
        prior: &ConjugateOlsPrior,
        label: &str,
        block: &LinearOlsSuffStats,
    ) -> Result<(), StateError> {
        self.check_prior(prior)?;
        let p = self.prior.ncols();
        if block.is_invalidated() {
            return Err(StateError::InvalidEvent(
                "block summary is invalidated; recompute it from the retained rows".into(),
            ));
        }
        if block.ncols() != p {
            return Err(StateError::Shape(format!("block ncols {} != {p}", block.ncols())));
        }
        if block.n() == 0 {
            return Err(StateError::InvalidEvent("block holds no observations".into()));
        }
        let digest = block_digest(block);
        if let Some(previous) = self.used.get(label) {
            let detail = if *previous == digest {
                "the same observations would enter the likelihood twice"
            } else {
                "the label was used for different observations"
            };
            return Err(StateError::InvalidEvent(format!(
                "block {label} was already used: {detail}; drop the duplicate block, or rebuild \
                 from the distinct blocks"
            )));
        }
        let mut precision = self.precision.clone();
        for (slot, add) in precision.iter_mut().zip(block.xtx()) {
            *slot += add;
        }
        let mut eta = self.eta.clone();
        for (slot, add) in eta.iter_mut().zip(block.xty()) {
            *slot += add;
        }
        self.precision = precision;
        self.eta = eta;
        self.sum_squares += block.yty();
        self.shape += block.n() as f64 / 2.0;
        self.n = self.n.saturating_add(block.n());
        self.used.insert(Arc::from(label), digest);
        Ok(())
    }

    /// The posterior, under `prior`, which must be the prior this posterior was built with.
    ///
    /// # Errors
    /// A different prior, or a singular posterior precision.
    pub fn summary(&self, prior: &ConjugateOlsPrior) -> Result<ConjugateOlsSummary, StateError> {
        self.check_prior(prior)?;
        let p = self.prior.ncols();
        let scale = invert_square(&self.precision, p)
            .ok_or_else(|| StateError::Numerical("singular posterior precision".into()))?;
        let mean = multiply(&scale, p, &self.eta);
        let explained: f64 = mean.iter().zip(&self.eta).map(|(m, e)| m * e).sum();
        let rate = self.prior.rate + 0.5 * (self.sum_squares - explained);
        if !positive(rate) {
            return Err(StateError::Numerical("posterior rate is not positive".into()));
        }
        Ok(ConjugateOlsSummary {
            mean,
            precision: self.precision.clone(),
            scale,
            shape: self.shape,
            rate,
            n: self.n,
        })
    }

    /// The posterior as the prior of the next, independent data.
    ///
    /// The prior remembers every block this posterior used, so a later update with one of them
    /// again is refused.
    ///
    /// # Errors
    /// The errors of [`Self::summary`].
    pub fn to_prior(&self) -> Result<ConjugateOlsPrior, StateError> {
        let summary = self.summary(&self.prior)?;
        Ok(ConjugateOlsPrior {
            mean: summary.mean,
            precision: summary.precision,
            shape: summary.shape,
            rate: summary.rate,
            used: self.used.clone(),
        })
    }

    /// Rebuild under `new_prior` from every block this posterior added itself.
    ///
    /// `blocks` must be exactly those blocks (same labels, same terms), and `new_prior` must
    /// not already have used any of them.
    ///
    /// # Errors
    /// A missing, extra or changed block, a block `new_prior` already used, or the errors of
    /// [`Self::update`].
    pub fn reprior(
        &self,
        new_prior: &ConjugateOlsPrior,
        blocks: &[(&str, &LinearOlsSuffStats)],
    ) -> Result<Self, StateError> {
        let own: BTreeMap<Arc<str>, String> = self
            .used
            .iter()
            .filter(|(label, _)| !self.prior.used.contains_key(*label))
            .map(|(label, digest)| (Arc::clone(label), digest.clone()))
            .collect();
        let complete = blocks.len() == own.len()
            && blocks.iter().all(|(label, block)| {
                own.get(*label).is_some_and(|digest| *digest == block_digest(block))
            });
        if !complete {
            return Err(StateError::InvalidEvent(
                "the blocks are not exactly the ones this posterior used; supply every retained \
                 block, or fit the new prior from the observations"
                    .into(),
            ));
        }
        let mut rebuilt = Self::new(new_prior);
        for (label, block) in blocks {
            rebuilt.update(new_prior, label, block)?;
        }
        Ok(rebuilt)
    }
}
