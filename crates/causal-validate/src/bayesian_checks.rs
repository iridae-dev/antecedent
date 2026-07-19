//! Prior/posterior predictive checks and prior sensitivity (DESIGN.md §18.4 subset).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::needless_range_loop,
    clippy::too_many_lines,
    clippy::many_single_char_names
)]

use std::sync::Arc;

use causal_core::{CausalRng, ExecutionContext, KernelPolicy};
use causal_estimate::{
    BayesianGCompWorkspace, BayesianGComputationAte, CausalPosterior, PreparedBayesianProblem,
};
use causal_identify::IdentificationStatus;
use causal_kernels::{PosteriorReduceOp, reduce_posterior_draws, standard_normal};
use causal_prob::{PriorSensitivitySummary, PriorSet};
use causal_stats::GlmFamily;

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

impl PredictiveCheckReport {
    /// Convert to a suite [`RefutationReport`] using a two-sided α threshold on `p_value`.
    #[must_use]
    pub fn to_refutation_report(&self, original_ate: f64, alpha: f64) -> RefutationReport {
        let name = match self.kind {
            PredictiveCheckKind::Prior => "prior_predictive",
            PredictiveCheckKind::Posterior => "posterior_predictive",
        };
        let passed = self.p_value.is_finite() && self.p_value >= alpha;
        RefutationReport {
            refuter: Arc::from(name),
            original_ate,
            refuted_ate: self.predictive_mean,
            comparison: self.p_value,
            informative: true,
            passed,
            failure_condition: if passed {
                None
            } else {
                Some(Arc::from(format!(
                    "predictive check failed (p={} < alpha={alpha})",
                    self.p_value
                )))
            },
            replicates: self.n_sims,
        }
    }
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
    /// Mean family (inverse link applied to η before summarizing).
    pub family: GlmFamily,
}

impl Default for PriorPredictiveCheck {
    fn default() -> Self {
        Self::new()
    }
}

impl PriorPredictiveCheck {
    /// Default 200 sims, Gaussian identity.
    #[must_use]
    pub fn new() -> Self {
        Self { n_sims: 200, seed: 0, family: GlmFamily::GaussianIdentity }
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
            return Err(ValidationError::estimation_msg("empty design for PPC"));
        }
        let observed = problem.design.outcome.iter().sum::<f64>() / n as f64;
        let mut rng = CausalRng::from_seed(self.seed);
        let prior = PriorSet::weakly_informative(p);
        let coef_prior = prior.gaussian_coefficients().ok_or_else(|| {
            ValidationError::estimation_msg("weakly informative prior missing coefficients")
        })?;
        let mut summaries = Vec::with_capacity(self.n_sims as usize);
        let mut beta = vec![0.0; p];
        for _ in 0..self.n_sims {
            // Draw β ~ prior once per simulation, then μ_i = g^{-1}(x_i'β).
            for c in 0..p {
                beta[c] =
                    coef_prior.mean[c] + coef_prior.variance[c].sqrt() * standard_normal(&mut rng);
            }
            let mut mean_y = 0.0;
            for r in 0..n {
                let mut eta = 0.0;
                for c in 0..p {
                    eta += problem.design.matrix[c * n + r] * beta[c];
                }
                mean_y += self.family.mean_from_eta(eta);
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
    /// Mean family (inverse link applied to η before summarizing).
    pub family: GlmFamily,
}

impl Default for PosteriorPredictiveCheck {
    fn default() -> Self {
        Self::new()
    }
}

impl PosteriorPredictiveCheck {
    /// Default Gaussian identity.
    #[must_use]
    pub fn new() -> Self {
        Self { n_sims: 200, family: GlmFamily::GaussianIdentity }
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
            return Err(ValidationError::estimation_msg("no posterior draws for PPC"));
        }
        let mut summaries = Vec::with_capacity(n_draws);
        for d in 0..n_draws {
            let mut mean_y = 0.0;
            for r in 0..n {
                let mut eta = 0.0;
                for c in 0..p {
                    let x = problem.design.matrix[c * n + r];
                    let b = posterior.draws.get(d, c).map_err(ValidationError::from)?;
                    eta += x * b;
                }
                mean_y += self.family.mean_from_eta(eta);
            }
            summaries.push(mean_y / n as f64);
        }
        Ok(summarize_check(PredictiveCheckKind::Posterior, observed, &summaries, n_draws as u32))
    }
}

/// Default max relative range of effect means across the prior-sensitivity grid.
pub const DEFAULT_MAX_RELATIVE_PRIOR_RANGE: f64 = 0.5;

/// Prior sensitivity grid over isotropic coefficient prior scales.
#[derive(Clone, Debug)]
pub struct PriorSensitivity {
    /// Prior scales (σ of isotropic Gaussian coefficient prior).
    pub scales: Arc<[f64]>,
    /// Fail when `(max−min) / scale` exceeds this, where `scale` is
    /// `max(|means…|, |original_ate|, ε)`.
    pub max_relative_range: f64,
}

impl Default for PriorSensitivity {
    fn default() -> Self {
        Self::standard_grid()
    }
}

impl PriorSensitivity {
    /// Standard grid `{0.5, 1, 2, 5, 10, 20}` with [`DEFAULT_MAX_RELATIVE_PRIOR_RANGE`].
    #[must_use]
    pub fn standard_grid() -> Self {
        Self {
            scales: Arc::from(vec![0.5, 1.0, 2.0, 5.0, 10.0, 20.0]),
            max_relative_range: DEFAULT_MAX_RELATIVE_PRIOR_RANGE,
        }
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
                ValidationError::estimation_msg(format!("prior sensitivity fit failed: {e}"))
            })?;
            let eq = post.effect_column().ok_or_else(|| {
                ValidationError::estimation_msg("missing effect column in sensitivity fit")
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
    ///
    /// Passes when the relative range of effect means is finite and
    /// `≤ max_relative_range`.
    #[must_use]
    pub fn to_report(
        &self,
        summary: &PriorSensitivitySummary,
        original_ate: f64,
    ) -> RefutationReport {
        let min = summary.effect_means.iter().copied().fold(f64::INFINITY, f64::min);
        let max = summary.effect_means.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let range = max - min;
        let denom = summary
            .effect_means
            .iter()
            .copied()
            .map(f64::abs)
            .fold(original_ate.abs(), f64::max)
            .max(1e-8);
        let relative = range / denom;
        let passed = relative.is_finite() && relative <= self.max_relative_range;
        RefutationReport {
            refuter: Arc::from("prior_sensitivity"),
            original_ate,
            refuted_ate: summary.effect_means.last().copied().unwrap_or(original_ate),
            comparison: relative,
            informative: true,
            passed,
            failure_condition: if passed {
                None
            } else {
                Some(Arc::from(format!(
                    "prior sensitivity relative range {relative} exceeds max {}",
                    self.max_relative_range
                )))
            },
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
    let policy = KernelPolicy::default_policy();
    let mean = reduce_posterior_draws(summaries, PosteriorReduceOp::Mean, &policy).unwrap_or(0.0);
    let sd = reduce_posterior_draws(summaries, PosteriorReduceOp::Std, &policy).unwrap_or(0.0);
    let n = summaries.len() as f64;
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
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(t),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(y),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(2),
                    Arc::from(z),
                    ValidityBitmap::all_valid(n),
                )
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
        let prior_rep = PriorPredictiveCheck { n_sims: 50, seed: 3, ..PriorPredictiveCheck::new() }
            .check(&prep, &ctx)
            .unwrap();
        assert_eq!(prior_rep.kind, PredictiveCheckKind::Prior);
        assert!(prior_rep.p_value.is_finite());

        let mut ws = BayesianGCompWorkspace::default();
        let post = bayes
            .fit(&prep, IdentificationStatus::NonparametricallyIdentified, &mut ws, &ctx)
            .unwrap();
        let post_rep = PosteriorPredictiveCheck { n_sims: 50, ..PosteriorPredictiveCheck::new() }
            .check(&prep, &post)
            .unwrap();
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
            max_relative_range: DEFAULT_MAX_RELATIVE_PRIOR_RANGE,
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
        let rep =
            sens.to_report(&summary, posts[0].summaries.mean[posts[0].effect_column().unwrap()]);
        assert!(rep.passed);
    }
}

/// MCMC chain diagnostics gate (ESS / R-hat / divergences).
///
/// Applicable only when the posterior was produced by an MCMC backend
/// (`InferenceDiagnostics::factorization == Mcmc`).
#[derive(Clone, Copy, Debug)]
pub struct McmcDiagnosticsCheck {
    /// Maximum acceptable split-Ř.
    pub max_rhat: f64,
    /// Minimum acceptable bulk ESS.
    pub min_ess: f64,
    /// Maximum acceptable divergence count.
    pub max_divergences: u32,
}

impl Default for McmcDiagnosticsCheck {
    fn default() -> Self {
        Self { max_rhat: 1.05, min_ess: 10.0, max_divergences: u32::MAX / 4 }
    }
}

impl McmcDiagnosticsCheck {
    /// Construct with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Evaluate against a fitted posterior's diagnostics.
    ///
    /// Returns `None` when the posterior is not MCMC (caller should emit NotApplicable).
    pub fn check(&self, posterior: &CausalPosterior) -> Option<RefutationReport> {
        use causal_prob::HessianFactorization;
        let d = &posterior.diagnostics;
        if d.factorization != HessianFactorization::Mcmc {
            return None;
        }
        let rhat = d.rhat_max.unwrap_or(f64::INFINITY);
        let ess = d.ess_bulk_min.unwrap_or(0.0);
        let divs = d.n_divergences.unwrap_or(u32::MAX);
        let passed = rhat.is_finite()
            && rhat <= self.max_rhat
            && ess >= self.min_ess
            && divs <= self.max_divergences
            && d.allows_posterior();
        let ate = posterior
            .effect_column()
            .and_then(|c| posterior.summaries.mean.get(c).copied())
            .unwrap_or(f64::NAN);
        Some(RefutationReport {
            refuter: Arc::from("mcmc_diagnostics"),
            original_ate: ate,
            refuted_ate: ate,
            comparison: rhat,
            informative: true,
            passed,
            failure_condition: if passed {
                None
            } else {
                Some(Arc::from(format!(
                    "MCMC diagnostics failed: rhat={rhat:.4} ess={ess:.1} divergences={divs}"
                )))
            },
            replicates: d.n_chains.unwrap_or(0),
        })
    }
}

/// Simulation-based calibration ranks for a scalar posterior functional (DESIGN.md §18.4).
///
/// For each replicate: draw θ* from the prior predictive, simulate data, refit, and
/// record the rank of θ* among posterior draws of the primary effect.
#[derive(Clone, Debug)]
pub struct SimulationBasedCalibration {
    /// Number of SBC replicates.
    pub n_reps: u32,
    /// Draws per refit.
    pub n_draws: usize,
    /// RNG seed.
    pub seed: u64,
}

impl Default for SimulationBasedCalibration {
    fn default() -> Self {
        Self { n_reps: 50, n_draws: 100, seed: 0 }
    }
}

/// SBC report.
#[derive(Clone, Debug)]
pub struct SbcReport {
    /// Rank of the prior draw in each replicate (0..=n_draws).
    pub ranks: Arc<[u32]>,
    /// Mean rank / n_draws (≈ 0.5 when calibrated).
    pub mean_rank_frac: f64,
    /// Chi² uniformity diagnostic on coarse bins (lower is better).
    pub uniformity_stat: f64,
}

impl SimulationBasedCalibration {
    /// Construct.
    #[must_use]
    pub fn new(n_reps: u32) -> Self {
        Self { n_reps: n_reps.max(1), ..Self::default() }
    }

    /// Run SBC: draw θ from the prior, simulate `y` from the prior predictive under
    /// the fixed design matrix, refit the Bayesian g-computation estimator, and
    /// rank the true ATE among posterior effect draws.
    ///
    /// # Errors
    ///
    /// Fit failures.
    pub fn check(
        &self,
        estimator: &BayesianGComputationAte,
        problem: &PreparedBayesianProblem,
        identification: IdentificationStatus,
        workspace: &mut BayesianGCompWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<SbcReport, ValidationError> {
        let mut rng = CausalRng::from_seed(self.seed);
        let n = problem.design.nrows;
        let p = problem.design.ncols;
        let t_col = problem
            .design
            .treatment_column()
            .ok_or_else(|| ValidationError::estimation_msg("SBC: missing treatment column"))?;
        let mut ranks = Vec::with_capacity(self.n_reps as usize);
        let mut est = estimator.clone();
        est.n_draws = self.n_draws;
        let scale = estimator.prior_scale.max(1e-6);

        for rep in 0..self.n_reps {
            let mut beta = vec![0.0; p];
            for c in 0..p {
                beta[c] = scale * standard_normal(&mut rng);
            }
            let true_effect = (problem.active - problem.control) * beta[t_col];
            let mut y_rep = vec![0.0; n];
            for r in 0..n {
                let mut eta = 0.0;
                for c in 0..p {
                    eta += problem.design.matrix[c * n + r] * beta[c];
                }
                y_rep[r] = eta + standard_normal(&mut rng);
            }
            let mut sim_problem = problem.clone();
            let mut design = sim_problem.design.clone();
            design.outcome = Arc::from(y_rep);
            sim_problem.design = design;
            est.seed = self.seed ^ (u64::from(rep).wrapping_mul(0x9E37));
            let post = est.fit(&sim_problem, identification, workspace, ctx).map_err(|e| {
                ValidationError::estimation_msg(format!("SBC refit failed: {e}"))
            })?;
            let col = post
                .effect_column()
                .ok_or_else(|| ValidationError::estimation_msg("SBC: no effect column"))?;
            let draws = post.draws.column(col).map_err(|e| {
                ValidationError::estimation_msg(format!("SBC draws: {e}"))
            })?;
            let mut rank = 0u32;
            for &d in draws {
                if d < true_effect {
                    rank += 1;
                }
            }
            ranks.push(rank);
        }

        let n_d = self.n_draws.max(1) as f64;
        let fracs: Vec<f64> = ranks.iter().map(|&r| f64::from(r) / n_d).collect();
        let mean_rank_frac =
            reduce_posterior_draws(&fracs, PosteriorReduceOp::Mean, &ctx.kernel_policy).unwrap_or(0.5);
        let bins = 10usize;
        let mut counts = vec![0.0; bins];
        for &r in &ranks {
            let frac = f64::from(r) / n_d;
            let b = ((frac * bins as f64).floor() as usize).min(bins - 1);
            counts[b] += 1.0;
        }
        let expected = self.n_reps as f64 / bins as f64;
        let mut chi2 = 0.0;
        for c in counts {
            let d = c - expected;
            chi2 += d * d / expected.max(1.0);
        }
        Ok(SbcReport {
            ranks: Arc::from(ranks),
            mean_rank_frac,
            uniformity_stat: chi2,
        })
    }

    /// Convert to a refutation report (passes when mean rank fraction ∈ [0.35, 0.65]).
    #[must_use]
    pub fn to_report(&self, report: &SbcReport, original_ate: f64) -> RefutationReport {
        let passed = (0.35..=0.65).contains(&report.mean_rank_frac);
        RefutationReport {
            refuter: Arc::from("sbc"),
            original_ate,
            refuted_ate: report.mean_rank_frac,
            comparison: report.uniformity_stat,
            informative: true,
            passed,
            failure_condition: if passed {
                None
            } else {
                Some(Arc::from(format!(
                    "SBC mean rank frac {:.3} outside [0.35, 0.65]",
                    report.mean_rank_frac
                )))
            },
            replicates: self.n_reps,
        }
    }
}

/// Likelihood-family comparison via leave-one-out log predictive density gap.
#[derive(Clone, Copy, Debug)]
pub struct LikelihoodFamilyComparison {
    /// Reserved (API stability).
    pub n_placeholder: u8,
}

impl Default for LikelihoodFamilyComparison {
    fn default() -> Self {
        Self { n_placeholder: 0 }
    }
}

impl LikelihoodFamilyComparison {
    /// Compare Gaussian vs Bernoulli logit Laplace fits using a LOO predictive
    /// score (higher is better). Gap is best − second.
    ///
    /// # Errors
    ///
    /// Fit failures.
    pub fn compare(
        &self,
        problem: &PreparedBayesianProblem,
        ctx: &ExecutionContext,
    ) -> Result<(Arc<str>, f64), ValidationError> {
        let _ = self;
        use causal_prob::{
            BayesDesignRef, BayesFitOptions, BayesLikelihood, InferenceBackend, LaplaceGlmBackend,
            LaplaceWorkspace, PriorSet,
        };
        let design = BayesDesignRef {
            x_colmajor: &problem.design.matrix,
            nrows: problem.design.nrows,
            ncols: problem.design.ncols,
            y: &problem.design.outcome,
            weights: None,
            offsets: None,
        };
        let prior = PriorSet::weakly_informative(problem.design.ncols);
        let opts = BayesFitOptions { n_draws: 80, seed: 1, ..BayesFitOptions::default() };
        let mut ws = LaplaceWorkspace::default();
        let g = LaplaceGlmBackend
            .fit(BayesLikelihood::GaussianIdentity, design, &prior, &opts, &mut ws, ctx)
            .map_err(|e| ValidationError::estimation_msg(format!("Gaussian fit: {e}")))?;
        let g_score = loo_gaussian_lpd(
            &g.map,
            &problem.design.matrix,
            problem.design.nrows,
            problem.design.ncols,
            &problem.design.outcome,
        );

        let binary = problem.design.outcome.iter().all(|&y| y == 0.0 || y == 1.0);
        if !binary {
            return Ok((Arc::from("gaussian_identity"), 0.0));
        }
        let b = LaplaceGlmBackend
            .fit(BayesLikelihood::BernoulliLogit, design, &prior, &opts, &mut ws, ctx)
            .map_err(|e| ValidationError::estimation_msg(format!("Bernoulli fit: {e}")))?;
        let b_score = loo_bernoulli_lpd(
            &b.map,
            &problem.design.matrix,
            problem.design.nrows,
            problem.design.ncols,
            &problem.design.outcome,
        );
        if b_score >= g_score {
            Ok((Arc::from("bernoulli_logit"), b_score - g_score))
        } else {
            Ok((Arc::from("gaussian_identity"), g_score - b_score))
        }
    }
}

fn loo_gaussian_lpd(map: &[f64], x: &[f64], n: usize, p: usize, y: &[f64]) -> f64 {
    let mut resid = vec![0.0; n];
    let mut rss = 0.0;
    for r in 0..n {
        let mut eta = 0.0;
        for c in 0..p {
            eta += x[c * n + r] * map.get(c).copied().unwrap_or(0.0);
        }
        resid[r] = y[r] - eta;
        rss += resid[r] * resid[r];
    }
    let sigma2 = (rss / n.max(1) as f64).max(1e-8);
    let mut lpd = 0.0;
    for r in 0..n {
        let s2 = sigma2 * n as f64 / (n.saturating_sub(1)).max(1) as f64;
        lpd += -0.5
            * (s2.ln()
                + resid[r] * resid[r] / s2
                + std::f64::consts::LN_2
                + std::f64::consts::PI.ln());
    }
    lpd
}

fn loo_bernoulli_lpd(map: &[f64], x: &[f64], n: usize, p: usize, y: &[f64]) -> f64 {
    let mut lpd = 0.0;
    for r in 0..n {
        let mut eta = 0.0;
        for c in 0..p {
            eta += x[c * n + r] * map.get(c).copied().unwrap_or(0.0);
        }
        let prob = 1.0 / (1.0 + (-eta).exp());
        lpd += if y[r] > 0.5 {
            prob.max(1e-12).ln()
        } else {
            (1.0 - prob).max(1e-12).ln()
        };
    }
    lpd
}

/// Posterior calibration on synthetic SCMs: known-ATE credible-interval coverage.
#[derive(Clone, Debug)]
pub struct PosteriorCalibrationOnSyntheticScm {
    /// Monte Carlo replicates.
    pub n_reps: u32,
    /// Draws per fit.
    pub n_draws: usize,
    /// Nominal coverage level (e.g. 0.9).
    pub level: f64,
    /// RNG seed.
    pub seed: u64,
}

impl Default for PosteriorCalibrationOnSyntheticScm {
    fn default() -> Self {
        Self { n_reps: 40, n_draws: 100, level: 0.9, seed: 0 }
    }
}

/// Report for [`PosteriorCalibrationOnSyntheticScm`].
#[derive(Clone, Debug)]
pub struct PosteriorCalibrationReport {
    /// Empirical coverage of equal-tailed credible intervals.
    pub coverage: f64,
    /// Mean absolute error of posterior means vs true ATE.
    pub mean_abs_error: f64,
    /// Replicates.
    pub n_reps: u32,
}

impl PosteriorCalibrationOnSyntheticScm {
    /// Simulate known ATEs under the design, refit, and measure CI coverage.
    ///
    /// # Errors
    ///
    /// Fit failures.
    pub fn check(
        &self,
        estimator: &BayesianGComputationAte,
        problem: &PreparedBayesianProblem,
        identification: IdentificationStatus,
        workspace: &mut BayesianGCompWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<PosteriorCalibrationReport, ValidationError> {
        let mut rng = CausalRng::from_seed(self.seed);
        let n = problem.design.nrows;
        let p = problem.design.ncols;
        let t_col = problem
            .design
            .treatment_column()
            .ok_or_else(|| ValidationError::estimation_msg("calibration: missing treatment"))?;
        let mut covered = 0u32;
        let mut abs_err = 0.0;
        let mut est = estimator.clone();
        est.n_draws = self.n_draws;
        let alpha = ((1.0 - self.level) / 2.0).clamp(0.0, 0.5);

        for rep in 0..self.n_reps {
            let true_ate = standard_normal(&mut rng);
            let mut beta = vec![0.0; p];
            let diff = problem.active - problem.control;
            beta[t_col] = if diff.abs() > 1e-12 { true_ate / diff } else { true_ate };
            for c in 0..p {
                if c != t_col {
                    beta[c] = 0.5 * standard_normal(&mut rng);
                }
            }
            let mut y = vec![0.0; n];
            for r in 0..n {
                let mut eta = 0.0;
                for c in 0..p {
                    eta += problem.design.matrix[c * n + r] * beta[c];
                }
                y[r] = eta + standard_normal(&mut rng);
            }
            let mut sim = problem.clone();
            let mut design = sim.design.clone();
            design.outcome = Arc::from(y);
            sim.design = design;
            est.seed = self.seed ^ (u64::from(rep).wrapping_mul(0xC2B2));
            let post = est.fit(&sim, identification, workspace, ctx).map_err(|e| {
                ValidationError::estimation_msg(format!("calibration refit: {e}"))
            })?;
            let col = post
                .effect_column()
                .ok_or_else(|| ValidationError::estimation_msg("calibration: no effect"))?;
            let mut draws = post.draws.column(col).map_err(ValidationError::from)?.to_vec();
            draws.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let lo = quantile_sorted(&draws, alpha);
            let hi = quantile_sorted(&draws, 1.0 - alpha);
            let mean =
                reduce_posterior_draws(&draws, PosteriorReduceOp::Mean, &ctx.kernel_policy).unwrap_or(0.0);
            abs_err += (mean - true_ate).abs();
            if true_ate >= lo && true_ate <= hi {
                covered += 1;
            }
        }
        Ok(PosteriorCalibrationReport {
            coverage: f64::from(covered) / f64::from(self.n_reps.max(1)),
            mean_abs_error: abs_err / f64::from(self.n_reps.max(1)),
            n_reps: self.n_reps,
        })
    }
}

fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let q = q.clamp(0.0, 1.0);
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}
