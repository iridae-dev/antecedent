//! Posterior predictive replicates as predictive laws.
//!
//! The Bayesian predictive checks in [`crate::bayesian_checks`] summarize
//! replicated outcomes by discrepancy statistics and tail probabilities. Where a
//! posterior predictive draw set also *is* a predictive law of each observed
//! outcome, it is assessed here, through the one scoring surface: each
//! observation's replicates are an equally weighted sample law scored by the
//! CRPS and checked by the PIT, exactly as any other sample law.
//! [`crate::PosteriorPredictiveCheck::replicates`] draws the replicates with the
//! same noise stream and parameter-draw selection as the check's own, so the two
//! views describe the same replicate set.
//!
//! The replicates are drawn at parameters fitted to the observed rows, so an
//! assessment of them is `in_sample` unless the observations scored are rows the
//! posterior was not conditioned on; the fit declaration says which.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::law::{ForecastLaw, LawSource, MAX_LAW_ATOMS, PredictiveTarget, PredictorLaws};
use super::overlap::FitDeclaration;
use super::refusal::{PredictiveAssessmentError, detail, invalid};

/// Replicated outcomes: `draws` replicate vectors over `observations` rows.
#[derive(Clone, Debug, PartialEq)]
pub struct PredictiveReplicates {
    draws: usize,
    observations: usize,
    /// Draw-major: replicate `d` of row `i` is `values[d * observations + i]`.
    values: Vec<f64>,
}

impl PredictiveReplicates {
    /// Wrap draw-major replicate values.
    ///
    /// # Errors
    /// No draws or rows, more draws than a sample law admits, a length that is
    /// not `draws * observations`, or a non-finite value.
    pub fn new(
        draws: usize,
        observations: usize,
        values: Vec<f64>,
    ) -> Result<Self, PredictiveAssessmentError> {
        if draws == 0
            || draws > MAX_LAW_ATOMS
            || observations == 0
            || values.len() != draws.saturating_mul(observations)
            || values.iter().any(|v| !v.is_finite())
        {
            return Err(invalid(
                detail::INVALID_LAW,
                format!(
                    "replicates need 1 to {MAX_LAW_ATOMS} draws of finite values over a positive \
                     number of rows, draw-major ({draws} x {observations} = {} values expected, \
                     {} supplied)",
                    draws.saturating_mul(observations),
                    values.len()
                ),
            ));
        }
        Ok(Self { draws, observations, values })
    }

    /// Number of replicate draws.
    #[must_use]
    pub const fn draws(&self) -> usize {
        self.draws
    }

    /// Number of observed rows.
    #[must_use]
    pub const fn observations(&self) -> usize {
        self.observations
    }

    /// One equally weighted sample law per observed row.
    #[must_use]
    pub fn laws(&self) -> Vec<ForecastLaw> {
        (0..self.observations)
            .map(|row| ForecastLaw::EmpiricalSample {
                draws: (0..self.draws).map(|d| self.values[d * self.observations + row]).collect(),
            })
            .collect()
    }

    /// The replicates as a named predictor of `target`.
    #[must_use]
    pub fn into_predictor(
        &self,
        name: impl Into<String>,
        model: impl Into<String>,
        target: PredictiveTarget,
        fit: FitDeclaration,
    ) -> PredictorLaws {
        PredictorLaws {
            name: name.into(),
            source: LawSource::Native { model: model.into() },
            target,
            laws: self.laws(),
            fit,
        }
    }
}
