//! Prior/posterior predictive checks and prior sensitivity (DESIGN.md §18.4 Phase 6 subset).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::needless_range_loop)]

use std::sync::Arc;

use causal_core::{CausalRng, ExecutionContext};
use causal_estimate::{
    BayesianGCompWorkspace, BayesianGComputationAte, CausalPosterior, PreparedBayesianProblem,
};
use causal_identify::IdentificationStatus;
use causal_prob::{PriorSensitivitySummary, PriorSet};

use crate::common::RefutationReport;
use crate::error::ValidationError;

/// Result of a prior or posterior predictive check.
#[derive(Clone, Debug)]
pub struct PredictiveCheckReport {
    /// Check kind.
    pub kind: PredictiveCheckKind,
    /// Observed summary statistic (e.g. outcome mean).
    pub observed: f64,
    /// Mean of the predictive summary across simulations.
    pub predictive_mean: f64,
    /// SD of the predictive summary.
    pub predictive_sd: f64,
    /// Two-sided tail probability of `observed` under the predictive distribution.
    pub p_value: f64,
    /// Number of predictive simulations.
    pub n_sims: u32,
}

/// Prior vs posterior predictive.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum PredictiveCheckKind {
    /// Simulate from the prior predictive.
    Prior,
    /// Simulate from the posterior predictive.
    Posterior,
}

/// Prior predictive check using coefficient draws from a weakly informative prior
/// (no data update) vs observed outcome mean.
#[derive(Clone, Debug)]
pub struct PriorPredictiveCheck {
    /// Simulations.
    pub n_sims: u32,
    /// RNG seed.
    pub seed: u64,
}

impl Default for PriorPredictiveCheck {
    fn default() -> Self {
        Self::new()
    }
}

impl PriorPredictiveCheck {
    /// Default 200 sims.
    #[must_use]
    pub fn new() -> Self {
        Self { n_sims: 200, seed: 0 }
    }

    /// Run against a prepared Bayesian design (uses prior draws only).
    ///
    /// # Errors
    ///
    /// Empty design.
    pub fn check(
        &self,
        problem: &PreparedBayesianProblem,
        _ctx: &ExecutionContext,
    ) -> Result<PredictiveCheckReport, ValidationError> {
        let n = problem.design.nrows;
        let p = problem.design.ncols;
        if n == 0 || p == 0 {
            return Err(ValidationError::Estimation("empty design for PPC".into()));
        }
        let observed = problem.design.outcome.iter().sum::<f64>() / n as f64;
        let mut rng = CausalRng::from_seed(self.seed);
        let prior = PriorSet::weakly_informative(p);
        let coef_prior = prior.gaussian_coefficients().ok_or_else(|| {
            ValidationError::Estimation("weakly informative prior missing coefficients".into())
        })?;
        let mut summaries = Vec::with_capacity(self.n_sims as usize);
        for _ in 0..self.n_sims {
            // Draw β ~ prior, then ŷ_i = x_i'β, summarize by mean.
            let mut mean_y = 0.0;
            for r in 0..n {
                let mut eta = 0.0;
                for c in 0..p {
                    let x = problem.design.matrix[c * n + r];
                    let b = coef_prior.mean[c]
                        + coef_prior.variance[c].sqrt() * standard_normal(&mut rng);
                    eta += x * b;
                }
                mean_y += eta;
            }
            summaries.push(mean_y / n as f64);
        }
        Ok(summarize_check(PredictiveCheckKind::Prior, observed, &summaries, self.n_sims))
    }
}

/// Posterior predictive check: resample outcome means from posterior coefficient draws.
#[derive(Clone, Debug)]
pub struct PosteriorPredictiveCheck {
    /// Number of posterior draws to use (capped by available).
    pub n_sims: u32,
}

impl Default for PosteriorPredictiveCheck {
    fn default() -> Self {
        Self::new()
    }
}

impl PosteriorPredictiveCheck {
    /// Default.
    #[must_use]
    pub fn new() -> Self {
        Self { n_sims: 200 }
    }

    /// Check using a fitted [`CausalPosterior`] that includes coefficient columns.
    ///
    /// # Errors
    ///
    /// Missing coefficients / empty draws.
    pub fn check(
        &self,
        problem: &PreparedBayesianProblem,
        posterior: &CausalPosterior,
    ) -> Result<PredictiveCheckReport, ValidationError> {
        let n = problem.design.nrows;
        let p = problem.design.ncols;
        let observed = problem.design.outcome.iter().sum::<f64>() / n as f64;
        let n_draws = posterior.draws.n_draws.min(self.n_sims as usize);
        if n_draws == 0 {
            return Err(ValidationError::Estimation("no posterior draws for PPC".into()));
        }
        let mut summaries = Vec::with_capacity(n_draws);
        for d in 0..n_draws {
            let mut mean_y = 0.0;
            for r in 0..n {
                let mut eta = 0.0;
                for c in 0..p {
                    let x = problem.design.matrix[c * n + r];
                    let b = posterior
                        .draws
                        .get(d, c)
                        .map_err(|e| ValidationError::Estimation(e.to_string()))?;
                    eta += x * b;
                }
                mean_y += eta;
            }
            summaries.push(mean_y / n as f64);
        }
        Ok(summarize_check(
            PredictiveCheckKind::Posterior,
            observed,
            &summaries,
            n_draws as u32,
        ))
    }
}

/// Prior sensitivity grid over isotropic coefficient prior scales.
#[derive(Clone, Debug)]
pub struct PriorSensitivity {
    /// Prior scales (σ of isotropic Gaussian coefficient prior).
    pub scales: Arc<[f64]>,
}

impl Default for PriorSensitivity {
    fn default() -> Self {
        Self::standard_grid()
    }
}

impl PriorSensitivity {
    /// Standard grid `{0.5, 1, 2, 5, 10, 20}`.
    #[must_use]
    pub fn standard_grid() -> Self {
        Self { scales: Arc::from(vec![0.5, 1.0, 2.0, 5.0, 10.0, 20.0]) }
    }

    /// Refit Bayesian g-comp at each prior scale; return sensitivity summary.
    ///
    /// # Errors
    ///
    /// Fit failures.
    pub fn evaluate(
        &self,
        estimator: &BayesianGComputationAte,
        problem: &PreparedBayesianProblem,
        identification: IdentificationStatus,
        workspace: &mut BayesianGCompWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<(PriorSensitivitySummary, Vec<CausalPosterior>), ValidationError> {
        let mut means = Vec::with_capacity(self.scales.len());
        let mut sds = Vec::with_capacity(self.scales.len());
        let mut posts = Vec::with_capacity(self.scales.len());
        for &scale in self.scales.iter() {
            let est = BayesianGComputationAte {
                prior_scale: scale,
                n_draws: estimator.n_draws.min(200),
                seed: estimator.seed,
                backend: estimator.backend,
                likelihood: estimator.likelihood,
                overlap: estimator.overlap,
            };
            let post = est.fit(problem, identification, workspace, ctx).map_err(|e| {
                ValidationError::Estimation(format!("prior sensitivity fit failed: {e}"))
            })?;
            let eq = post.effect_column().ok_or_else(|| {
                ValidationError::Estimation("missing effect column in sensitivity fit".into())
            })?;
            means.push(post.summaries.mean[eq]);
            sds.push(post.summaries.sd[eq]);
            posts.push(post);
        }
        Ok((
            PriorSensitivitySummary {
                prior_scales: Arc::clone(&self.scales),
                effect_means: Arc::from(means),
                effect_sds: Arc::from(sds),
            },
            posts,
        ))
    }

    /// Convert sensitivity range into a refutation-style report.
    #[must_use]
    pub fn to_report(&self, summary: &PriorSensitivitySummary, original_ate: f64) -> RefutationReport {
        let min = summary.effect_means.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = summary.effect_means.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max - min;
        RefutationReport {
            refuter: Arc::from("prior_sensitivity"),
            original_ate,
            refuted_ate: summary.effect_means.last().copied().unwrap_or(original_ate),
            comparison: range,
            informative: true,
            passed: range.is_finite(),
            failure_condition: None,
            replicates: self.scales.len() as u32,
        }
    }
}

fn summarize_check(
    kind: PredictiveCheckKind,
    observed: f64,
    summaries: &[f64],
    n_sims: u32,
) -> PredictiveCheckReport {
    let n = summaries.len() as f64;
    let mean = summaries.iter().sum::<f64>() / n.max(1.0);
    let var = if summaries.len() > 1 {
        summaries.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0)
    } else {
        0.0
    };
    let sd = var.sqrt();
    let below = summaries.iter().filter(|&&x| x <= observed).count() as f64;
    let p = (2.0 * (below / n.max(1.0)).min(1.0 - below / n.max(1.0))).min(1.0);
    PredictiveCheckReport {
        kind,
        observed,
        predictive_mean: mean,
        predictive_sd: sd,
        p_value: p,
        n_sims,
    }
}

fn standard_normal(rng: &mut CausalRng) -> f64 {
    let u1 = rng.next_f64().max(f64::EPSILON);
    let u2 = rng.next_f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Attach prior sensitivity onto a [`CausalPosterior`].
#[must_use]
pub fn with_prior_sensitivity(
    mut posterior: CausalPosterior,
    summary: PriorSensitivitySummary,
) -> CausalPosterior {
    posterior.prior_sensitivity = Some(summary);
    posterior
}

#[cfg(test)]
mod tests {
    use super::*;
    use causal_core::{
        AverageEffectQuery, CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet,
        ValueType, VariableId,
    };
    use causal_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
    };
    use causal_estimate::{BayesianBackendKind, BayesianGComputationAte};
    use causal_expr::{ExprId, IdentifiedEstimand};
    use causal_identify::IdentificationStatus;

    fn toy() -> (TabularData, IdentifiedEstimand, AverageEffectQuery) {
        let n = 60usize;
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "t",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "z",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let t: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
        let z: Vec<f64> = (0..n).map(|i| i as f64 * 0.05).collect();
        let y: Vec<f64> = (0..n).map(|i| 1.0 + 2.0 * t[i] + 0.3 * z[i]).collect();
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(0), Arc::from(t), ValidityBitmap::all_valid(n))
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(1), Arc::from(y), ValidityBitmap::all_valid(n))
                    .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(2), Arc::from(z), ValidityBitmap::all_valid(n))
                    .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let estimand = IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from([VariableId::from_raw(2)]),
            ExprId::from_raw(0),
        );
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        (TabularData::new(storage), estimand, query)
    }

    #[test]
    fn prior_and_posterior_ppc_run() {
        let (data, estimand, query) = toy();
        let bayes = BayesianGComputationAte {
            backend: BayesianBackendKind::ConjugateGaussian,
            n_draws: 100,
            seed: 2,
            prior_scale: 10.0,
            ..BayesianGComputationAte::new()
        };
        let prep = bayes.prepare(&data, &estimand, &query).unwrap();
        let ctx = ExecutionContext::for_tests(1);
        let prior_rep = PriorPredictiveCheck { n_sims: 50, seed: 3 }.check(&prep, &ctx).unwrap();
        assert_eq!(prior_rep.kind, PredictiveCheckKind::Prior);
        assert!(prior_rep.p_value.is_finite());

        let mut ws = BayesianGCompWorkspace::default();
        let post = bayes
            .fit(
                &prep,
                IdentificationStatus::NonparametricallyIdentified,
                &mut ws,
                &ctx,
            )
            .unwrap();
        let post_rep = PosteriorPredictiveCheck { n_sims: 50 }.check(&prep, &post).unwrap();
        assert_eq!(post_rep.kind, PredictiveCheckKind::Posterior);
    }

    #[test]
    fn prior_sensitivity_grid() {
        let (data, estimand, query) = toy();
        let bayes = BayesianGComputationAte {
            backend: BayesianBackendKind::ConjugateGaussian,
            n_draws: 80,
            seed: 4,
            ..BayesianGComputationAte::new()
        };
        let prep = bayes.prepare(&data, &estimand, &query).unwrap();
        let mut ws = BayesianGCompWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let sens = PriorSensitivity {
            scales: Arc::from(vec![1.0, 10.0, 50.0]),
        };
        let (summary, posts) = sens
            .evaluate(
                &bayes,
                &prep,
                IdentificationStatus::NonparametricallyIdentified,
                &mut ws,
                &ctx,
            )
            .unwrap();
        assert_eq!(summary.prior_scales.len(), 3);
        assert_eq!(posts.len(), 3);
        let rep = sens.to_report(&summary, posts[0].summaries.mean[posts[0].effect_column().unwrap()]);
        assert!(rep.passed);
    }
}
