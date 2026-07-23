//! Columnar posterior storage.
//!
//! One heap object per draw is prohibited; draws are stored as structure-of-arrays.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::error::ProbError;

/// Parameter / quantity role in a posterior schema.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum PosteriorQuantityKind {
    /// Regression coefficient (indexed).
    Coefficient {
        /// Coefficient index in the design.
        index: usize,
        /// Optional stable name.
        name: Option<Arc<str>>,
    },
    /// Residual variance / dispersion.
    ResidualVariance,
    /// Scalar causal effect (e.g. ATE).
    Effect {
        /// Stable effect name (e.g. "ate").
        name: Arc<str>,
    },
    /// Generic named scalar.
    Scalar {
        /// Name.
        name: Arc<str>,
    },
}

/// Schema describing columnar posterior layout.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PosteriorSchema {
    /// Ordered quantities; column `i` in draws holds quantity `i`.
    pub quantities: Arc<[PosteriorQuantityKind]>,
}

impl PosteriorSchema {
    /// Number of quantities (columns).
    #[must_use]
    pub fn n_quantities(&self) -> usize {
        self.quantities.len()
    }

    /// Schema for `p` coefficients + optional residual variance + one effect.
    #[must_use]
    pub fn coefficients_and_effect(n_coef: usize, include_sigma2: bool, effect: &str) -> Self {
        let mut q = Vec::with_capacity(n_coef + 2);
        for i in 0..n_coef {
            q.push(PosteriorQuantityKind::Coefficient { index: i, name: None });
        }
        if include_sigma2 {
            q.push(PosteriorQuantityKind::ResidualVariance);
        }
        q.push(PosteriorQuantityKind::Effect { name: Arc::from(effect) });
        Self { quantities: Arc::from(q) }
    }

    /// Coefficient-only schema.
    #[must_use]
    pub fn coefficients(n_coef: usize) -> Self {
        let q: Vec<_> = (0..n_coef)
            .map(|i| PosteriorQuantityKind::Coefficient { index: i, name: None })
            .collect();
        Self { quantities: Arc::from(q) }
    }

    /// Coefficient schema with durable semantic names (e.g. `intercept`, `coef_t`).
    #[must_use]
    pub fn coefficients_named(names: impl IntoIterator<Item = impl Into<Arc<str>>>) -> Self {
        let q: Vec<_> = names
            .into_iter()
            .enumerate()
            .map(|(i, name)| PosteriorQuantityKind::Coefficient {
                index: i,
                name: Some(name.into()),
            })
            .collect();
        Self { quantities: Arc::from(q) }
    }

    /// Attach names onto coefficient quantities by index (leaves non-coefs unchanged).
    ///
    /// Names shorter than the coefficient count leave remaining coefs unnamed.
    #[must_use]
    pub fn with_coefficient_names(&self, names: &[Arc<str>]) -> Self {
        let mut quantities = self.quantities.to_vec();
        for q in &mut quantities {
            if let PosteriorQuantityKind::Coefficient { index, name } = q {
                if let Some(n) = names.get(*index) {
                    *name = Some(Arc::clone(n));
                }
            }
        }
        Self { quantities: Arc::from(quantities) }
    }
}

/// Columnar posterior draws: `values` is column-major `[n_draws × n_quantities]`.
///
/// Layout: quantity `q`, draw `d` is at `values[q * n_draws + d]`.
#[derive(Clone, Debug, PartialEq)]
pub struct PosteriorDraws {
    /// Schema.
    pub schema: PosteriorSchema,
    /// Number of draws.
    pub n_draws: usize,
    /// Column-major values.
    pub values: Arc<[f64]>,
}

impl PosteriorDraws {
    /// Construct from column-major values.
    ///
    /// # Errors
    ///
    /// Length mismatch.
    pub fn from_column_major(
        schema: PosteriorSchema,
        n_draws: usize,
        values: impl Into<Arc<[f64]>>,
    ) -> Result<Self, ProbError> {
        let values = values.into();
        let expected = n_draws.saturating_mul(schema.n_quantities());
        if values.len() != expected {
            return Err(ProbError::Shape {
                message: "posterior values length != n_draws * n_quantities",
            });
        }
        Ok(Self { schema, n_draws, values })
    }

    /// Number of quantities.
    #[must_use]
    pub fn n_quantities(&self) -> usize {
        self.schema.n_quantities()
    }

    /// Borrow column `q` (length `n_draws`).
    ///
    /// # Errors
    ///
    /// Out of range.
    pub fn column(&self, q: usize) -> Result<&[f64], ProbError> {
        if q >= self.n_quantities() {
            return Err(ProbError::Shape { message: "quantity index out of range" });
        }
        let start = q * self.n_draws;
        Ok(&self.values[start..start + self.n_draws])
    }

    /// Value at (draw, quantity).
    ///
    /// # Errors
    ///
    /// Out of range.
    pub fn get(&self, draw: usize, quantity: usize) -> Result<f64, ProbError> {
        if draw >= self.n_draws || quantity >= self.n_quantities() {
            return Err(ProbError::Shape { message: "draw/quantity out of range" });
        }
        Ok(self.values[quantity * self.n_draws + draw])
    }

    /// Contiguous batch view over draw range `[start, start+len)`.
    ///
    /// # Errors
    ///
    /// Out of range.
    pub fn batch(&self, start: usize, len: usize) -> Result<PosteriorBatch<'_>, ProbError> {
        if start.saturating_add(len) > self.n_draws {
            return Err(ProbError::Shape { message: "batch range out of draws" });
        }
        Ok(PosteriorBatch { draws: self, start, len })
    }

    /// Summary statistics per quantity.
    #[must_use]
    pub fn summarize(&self) -> PosteriorSummary {
        use antecedent_core::KernelPolicy;
        use antecedent_kernels::{PosteriorReduceOp, reduce_posterior_draws};

        let n_q = self.n_quantities();
        let mut mean = vec![0.0; n_q];
        let mut sd = vec![0.0; n_q];
        let mut q025 = vec![0.0; n_q];
        let mut q975 = vec![0.0; n_q];
        let policy = KernelPolicy::default_policy();
        for q in 0..n_q {
            let col = &self.values[q * self.n_draws..(q + 1) * self.n_draws];
            mean[q] = reduce_posterior_draws(col, PosteriorReduceOp::Mean, &policy).unwrap_or(0.0);
            sd[q] = reduce_posterior_draws(col, PosteriorReduceOp::Std, &policy).unwrap_or(0.0);
            let mut sorted = col.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            q025[q] = quantile_sorted(&sorted, 0.025);
            q975[q] = quantile_sorted(&sorted, 0.975);
        }
        PosteriorSummary {
            schema: self.schema.clone(),
            n_draws: self.n_draws,
            mean: Arc::from(mean),
            sd: Arc::from(sd),
            q025: Arc::from(q025),
            q975: Arc::from(q975),
        }
    }

    /// Empirical P(column `q` < threshold).
    ///
    /// # Errors
    ///
    /// Out of range.
    pub fn probability_below(&self, q: usize, threshold: f64) -> Result<f64, ProbError> {
        let col = self.column(q)?;
        if col.is_empty() {
            return Ok(0.0);
        }
        let count = col.iter().filter(|&&x| x < threshold).count();
        Ok(count as f64 / col.len() as f64)
    }
}

/// Borrowed batch of draws for batched functional evaluation.
#[derive(Clone, Copy, Debug)]
pub struct PosteriorBatch<'a> {
    /// Parent draws.
    pub draws: &'a PosteriorDraws,
    /// Start draw index.
    pub start: usize,
    /// Number of draws in this batch.
    pub len: usize,
}

impl<'a> PosteriorBatch<'a> {
    /// Slice of column `q` for this batch.
    ///
    /// # Errors
    ///
    /// Out of range.
    pub fn column(&self, q: usize) -> Result<&'a [f64], ProbError> {
        let full = self.draws.column(q)?;
        Ok(&full[self.start..self.start + self.len])
    }
}

/// Summary of a posterior (means, SDs, credible intervals).
#[derive(Clone, Debug, PartialEq)]
pub struct PosteriorSummary {
    /// Schema.
    pub schema: PosteriorSchema,
    /// Draw count.
    pub n_draws: usize,
    /// Per-quantity mean.
    pub mean: Arc<[f64]>,
    /// Per-quantity sample SD.
    pub sd: Arc<[f64]>,
    /// 2.5% quantile.
    pub q025: Arc<[f64]>,
    /// 97.5% quantile.
    pub q975: Arc<[f64]>,
}

fn quantile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let n = sorted.len();
    let idx = ((n as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(n - 1)]
}

/// Scratch for batched posterior functional evaluation.
#[derive(Clone, Debug, Default)]
pub struct PosteriorEvalWorkspace {
    /// Scratch buffer for intermediate per-draw effects.
    pub effects: Vec<f64>,
    /// Scratch for design / prediction rows.
    pub row: Vec<f64>,
    /// Grow count for reuse diagnostics.
    pub grow_count: u32,
}

impl PosteriorEvalWorkspace {
    /// Ensure capacity for `n_draws` effects and `ncols` row scratch.
    pub fn prepare(&mut self, n_draws: usize, ncols: usize) {
        let mut grew = false;
        if self.effects.len() < n_draws {
            self.effects.resize(n_draws, 0.0);
            grew = true;
        }
        if self.row.len() < ncols {
            self.row.resize(ncols, 0.0);
            grew = true;
        }
        if grew {
            self.grow_count = self.grow_count.saturating_add(1);
        }
    }
}

/// Batched effect output aligned with a [`PosteriorBatch`].
#[derive(Clone, Debug, Default)]
pub struct EffectBatch {
    /// Per-draw effect values.
    pub values: Vec<f64>,
}

impl EffectBatch {
    /// Ensure length.
    pub fn prepare(&mut self, n: usize) {
        self.values.resize(n, 0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn columnar_layout_no_object_per_draw() {
        let schema = PosteriorSchema::coefficients(2);
        let values: Arc<[f64]> = Arc::from(vec![
            1.0, 2.0, 3.0, // coef 0 draws
            4.0, 5.0, 6.0, // coef 1 draws
        ]);
        let draws = PosteriorDraws::from_column_major(schema, 3, values).unwrap();
        assert_eq!(draws.column(0).unwrap(), &[1.0, 2.0, 3.0]);
        assert_eq!(draws.get(1, 1).unwrap(), 5.0);
        let batch = draws.batch(1, 2).unwrap();
        assert_eq!(batch.column(0).unwrap(), &[2.0, 3.0]);
        // Storage is a single Arc<[f64]>, not Vec of draw objects.
        assert_eq!(draws.values.len(), 6);
    }

    #[test]
    fn summarize_and_probability() {
        let schema = PosteriorSchema::coefficients(1);
        let values: Arc<[f64]> = Arc::from(vec![-1.0, 0.0, 1.0, 2.0]);
        let draws = PosteriorDraws::from_column_major(schema, 4, values).unwrap();
        let s = draws.summarize();
        assert!((s.mean[0] - 0.5).abs() < 1e-12);
        assert!((draws.probability_below(0, 0.0).unwrap() - 0.25).abs() < 1e-12);
    }
}
