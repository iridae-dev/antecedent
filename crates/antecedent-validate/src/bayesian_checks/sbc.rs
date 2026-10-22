//! Simulation-based calibration of a scalar posterior functional.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{CausalRng, ExecutionContext};
use antecedent_estimate::{
    BayesianGCompWorkspace, BayesianGComputationAte, PreparedBayesianProblem,
};
use antecedent_identify::IdentificationStatus;
use antecedent_kernels::{PosteriorReduceOp, reduce_posterior_draws, standard_normal};
use antecedent_prob::{
    GaussianCoefficientPrior, GaussianVarianceModel, InferenceDiagnostics, PriorSet,
    sample_inv_gamma,
};
use antecedent_stats::GlmFamily;

use crate::common::RefutationReport;
use crate::error::ValidationError;

use super::predictive::sample_poisson;

/// Simulation-based calibration ranks for a scalar posterior functional.
///
/// For each replicate: draw model parameters from the prior the estimator fits with, simulate data, refit,
/// and record the rank of the true effect among posterior draws of the primary effect. The rank
/// theorem (Talts et al. 2018) needs the generating prior to be *exactly* the fitted prior and the
/// posterior draws to be independent, so:
///
/// - For Gaussian outcomes, `σ²` is drawn from the estimator's residual-variance prior (fixed when known; a proper
///   inverse-gamma with finite mean otherwise) and `β | σ² ~ N(μ, σ² · diag(V0))` from its
///   coefficient prior, the conjugate convention of [`GaussianCoefficientPrior`]. A prior without
///   a finite-mean residual model (the weakly informative default) cannot be a generating prior
///   and is refused.
/// - For Bernoulli and Poisson GLMs, coefficients are drawn from the fitted
///   independent Gaussian prior and outcomes from the declared likelihood. The
///   ranked effect is the empirical-covariate average on the response scale.
/// - `n_draws + 1` must be a multiple of the 10 rank bins (e.g. 99), so every bin has equal prior
///   mass; at least 50 replicates are required so every bin expects at least five.
/// - For HMC, one post-warmup draw is retained from each independently seeded
///   chain per replicate; within-chain autocorrelation cannot enter the rank.
///   Other backends require independent draws.
#[derive(Clone, Debug)]
pub struct SimulationBasedCalibration {
    /// Number of SBC replicates (at least `5 ×` the rank bins, i.e. 50).
    pub n_reps: u32,
    /// Draws per refit (`n_draws + 1` a multiple of the 10 rank bins).
    pub n_draws: usize,
    /// RNG seed.
    pub seed: u64,
}

impl Default for SimulationBasedCalibration {
    fn default() -> Self {
        Self { n_reps: 50, n_draws: 99, seed: 0 }
    }
}

/// Refit seed for SBC replicate `rep`.
///
/// `0x9E37` is odd, so multiplication by it is a bijection on `u64`. Adjacent
/// replicates therefore get distinct seeds. They do share the high bits of
/// `seed`; that is the historical schedule, and widening the mix would move
/// every SBC draw.
pub(super) fn sbc_refit_seed(seed: u64, rep: u32) -> u64 {
    seed ^ u64::from(rep).wrapping_mul(0x9E37)
}

/// Rank-histogram bins of [`SimulationBasedCalibration`].
pub(super) const SBC_BINS: usize = 10;

/// Bin of `rank ∈ 0..=n_draws`; every bin holds `(n_draws + 1) / SBC_BINS` ranks when that divides.
pub(super) fn sbc_bin(rank: u32, n_draws: usize) -> usize {
    ((rank as usize) * SBC_BINS / (n_draws + 1)).min(SBC_BINS - 1)
}

/// Residual-variance prior a generating draw can sample from.
#[derive(Clone, Copy, Debug)]
pub(super) enum GeneratingVariance {
    Known(f64),
    InvGamma { shape: f64, scale: f64 },
}

/// Cholesky factor of the prior's coefficient correlation, when it is dense.
pub(super) fn prior_correlation_cholesky(
    prior: &PriorSet,
) -> Result<Option<Vec<f64>>, ValidationError> {
    prior
        .coefficient_correlation()
        .map(antecedent_prob::CoefficientCorrelation::cholesky)
        .transpose()
        .map_err(|e| ValidationError::estimation_msg(e.to_string()))
}

/// `p` standard normals with the prior's coefficient correlation: `L z` under
/// a dense prior, the iid draws themselves (same stream order) otherwise.
pub(super) fn prior_standard_normals(
    chol: Option<&[f64]>,
    p: usize,
    rng: &mut CausalRng,
) -> Vec<f64> {
    let z: Vec<f64> = (0..p).map(|_| standard_normal(rng)).collect();
    let Some(l) = chol else {
        return z;
    };
    (0..p).map(|i| (0..=i).map(|j| l[i * p + j] * z[j]).sum()).collect()
}

/// Draw `(σ², β)` from the fitted prior: `σ²` from its residual model, then
/// `β = μ + σ D^{1/2} L z` (the conjugate `β | σ² ~ N(μ, σ² V0)`, `V0` dense
/// under a coefficient correlation `R = L L'`, diagonal otherwise).
pub(super) fn draw_generating_parameters(
    coefficients: &GaussianCoefficientPrior,
    corr_chol: Option<&[f64]>,
    variance: GeneratingVariance,
    rng: &mut CausalRng,
) -> (f64, Vec<f64>) {
    let sigma2 = match variance {
        GeneratingVariance::Known(sigma2) => sigma2,
        GeneratingVariance::InvGamma { shape, scale } => sample_inv_gamma(shape, scale, rng),
    };
    let sigma = sigma2.sqrt();
    let z = prior_standard_normals(corr_chol, coefficients.mean.len(), rng);
    let beta = (0..coefficients.mean.len())
        .map(|c| coefficients.mean[c] + sigma * coefficients.variance[c].sqrt() * z[c])
        .collect();
    (sigma2, beta)
}

/// SBC report.
#[derive(Clone, Debug)]
pub struct SbcReport {
    /// Rank of the generating draw in each replicate (`0..=n_draws`).
    pub ranks: Arc<[u32]>,
    /// Mean rank / `n_draws` (≈ 0.5 when calibrated).
    pub mean_rank_frac: f64,
    /// Chi² uniformity diagnostic on coarse bins (lower is better).
    pub uniformity_stat: f64,
    /// Number of posterior draws compared with the generating effect in each rank.
    pub rank_draw_count: usize,
    /// Number of equal-probability rank bins in the uniformity diagnostic.
    pub rank_bins: usize,
}

/// Select draws for an SBC rank without treating serial HMC states as independent.
/// Native HMC stores its independent chains contiguously, so one draw from each
/// chain gives conditionally independent posterior draws after warmup. Diagnostics
/// have already passed the publication gate before this point.
pub(super) fn sbc_rank_draws(
    draws: &[f64],
    diagnostics: &InferenceDiagnostics,
) -> Result<Vec<f64>, ValidationError> {
    if let Some(chains) = diagnostics.n_chains {
        let chains = chains as usize;
        if chains != 4 || draws.len() % chains != 0 {
            return Err(ValidationError::NotApplicable {
                message: "HMC SBC requires four independent chains with equal retained lengths",
            });
        }
        let per_chain = draws.len() / chains;
        if per_chain == 0 {
            return Err(ValidationError::NotApplicable {
                message: "HMC SBC requires a retained draw in every chain",
            });
        }
        return Ok((0..chains).map(|chain| draws[(chain + 1) * per_chain - 1]).collect());
    }
    if let Some(ess) = diagnostics.ess_bulk_min {
        if !ess.is_finite() || ess < draws.len() as f64 {
            return Err(ValidationError::NotApplicable {
                message: "SBC needs independent draws or separate HMC chains; bulk ESS is below draw count",
            });
        }
    }
    Ok(draws.to_vec())
}

impl SimulationBasedCalibration {
    /// Construct.
    #[must_use]
    pub fn new(n_reps: u32) -> Self {
        Self { n_reps: n_reps.max(1), ..Self::default() }
    }

    /// Run SBC: draw parameters from the prior the estimator fits with, simulate `y` from the
    /// model under the fixed design matrix, refit the Bayesian g-computation estimator, and
    /// rank the true ATE among posterior effect draws.
    ///
    /// # Errors
    ///
    /// Fit failures; [`ValidationError::NotApplicable`] when the configuration or estimator
    /// cannot satisfy the rank theorem (see the type docs).
    #[allow(
        clippy::too_many_lines,
        reason = "one arm per posterior check kind over the same draws"
    )]
    pub fn check(
        &self,
        estimator: &BayesianGComputationAte,
        problem: &PreparedBayesianProblem,
        identification: IdentificationStatus,
        _workspace: &mut BayesianGCompWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<SbcReport, ValidationError> {
        if estimator.backend != antecedent_estimate::BayesianBackendKind::Hmc
            && (self.n_draws + 1) % SBC_BINS != 0
        {
            return Err(ValidationError::NotApplicable {
                message: "SBC requires n_draws + 1 to be a multiple of the 10 rank bins \
                          (e.g. 99), so every bin has equal prior mass",
            });
        }
        if (self.n_reps as usize) < 5 * SBC_BINS {
            return Err(ValidationError::NotApplicable {
                message: "SBC requires at least 50 replicates (five expected per rank bin)",
            });
        }
        let family = estimator.glm_family();
        if family == GlmFamily::NegativeBinomial {
            return Err(ValidationError::NotApplicable {
                message: "SBC does not have a declared negative-binomial generating law",
            });
        }
        let n = problem.design.nrows;
        let p = problem.design.ncols;
        let t_col = problem
            .design
            .treatment_column()
            .ok_or_else(|| ValidationError::estimation_msg("SBC: missing treatment column"))?;
        let prior = estimator.prior_in_force(p);
        let coefficients = prior.gaussian_coefficients().ok_or_else(|| {
            ValidationError::estimation_msg("SBC: the prior in force has no Gaussian coefficients")
        })?;
        if coefficients.mean.len() != p || coefficients.variance.len() != p {
            return Err(ValidationError::estimation_msg(
                "SBC: prior coefficient dimension mismatch",
            ));
        }
        let variance = if family == GlmFamily::GaussianIdentity {
            Some(
                match GaussianVarianceModel::from_prior_set(&prior)
                    .map_err(|e| ValidationError::estimation_msg(e.to_string()))?
                {
                    GaussianVarianceModel::Known { sigma2 } => GeneratingVariance::Known(sigma2),
                    GaussianVarianceModel::InvGamma { shape, scale } if shape > 1.0 => {
                        GeneratingVariance::InvGamma { shape, scale }
                    }
                    GaussianVarianceModel::InvGamma { .. } => {
                        return Err(ValidationError::NotApplicable {
                            message: "SBC needs a residual-variance prior with a finite mean (inverse-gamma \
                                  shape > 1) or a known residual variance to simulate from; the weakly \
                                  informative default has neither",
                        });
                    }
                },
            )
        } else {
            None
        };
        let mut rng = CausalRng::from_seed(self.seed);
        let mut est = estimator.clone();
        est.n_draws = self.n_draws;

        let corr_chol = prior_correlation_cholesky(&prior)?;
        // The generating draws share one RNG stream, so they are made in replicate order on this
        // thread; the refits (each seeded from its replicate index) run on the pool. A refit's
        // failure is carried as a value so a rank-count mismatch at an earlier replicate still
        // wins, as in a serial loop.
        let outcomes = crate::common::map_drawn_replicates(
            ctx,
            self.n_reps,
            |_| {
                let (sigma2, beta) = if let Some(variance) = variance {
                    draw_generating_parameters(
                        coefficients,
                        corr_chol.as_deref(),
                        variance,
                        &mut rng,
                    )
                } else {
                    let z = prior_standard_normals(corr_chol.as_deref(), p, &mut rng);
                    (
                        1.0,
                        coefficients
                            .mean
                            .iter()
                            .zip(coefficients.variance.iter())
                            .zip(&z)
                            .map(|((&mean, &variance), &z)| mean + variance.sqrt() * z)
                            .collect(),
                    )
                };
                let sigma = sigma2.sqrt();
                let mut true_effect = 0.0;
                let mut y_rep = vec![0.0; n];
                let mut eta_row = vec![0.0; n];
                crate::common::column_major_dot(&problem.design.matrix, n, &beta, &mut eta_row);
                for r in 0..n {
                    let eta = eta_row[r];
                    let observed_t = problem.design.matrix[t_col * n + r];
                    let eta_base = eta - observed_t * beta[t_col];
                    true_effect += family.mean_from_eta(eta_base + problem.active * beta[t_col])
                        - family.mean_from_eta(eta_base + problem.control * beta[t_col]);
                    let mean = family.mean_from_eta(eta);
                    if !mean.is_finite() || (family == GlmFamily::PoissonLog && mean > 1.0e4) {
                        return Err(ValidationError::NotApplicable {
                            message: "SBC generating prior produced an unsupported outcome mean",
                        });
                    }
                    y_rep[r] = match family {
                        GlmFamily::GaussianIdentity => mean + sigma * standard_normal(&mut rng),
                        GlmFamily::BinomialLogit | GlmFamily::BinomialProbit => {
                            f64::from(rng.next_f64() < mean)
                        }
                        GlmFamily::PoissonLog => sample_poisson(mean, &mut rng),
                        GlmFamily::NegativeBinomial => {
                            return Err(ValidationError::NotApplicable {
                                message: "SBC does not have a declared negative-binomial generating law",
                            });
                        }
                    };
                }
                Ok((y_rep, true_effect / n as f64))
            },
            BayesianGCompWorkspace::default,
            |workspace, ctx, rep, (y_rep, true_effect)| {
                let mut sim_problem = problem.clone();
                let mut design = sim_problem.design.clone();
                design.outcome = Arc::from(y_rep.as_slice());
                sim_problem.design = design;
                let mut est = est.clone();
                est.seed = sbc_refit_seed(self.seed, rep);
                Ok((|| {
                    let post =
                        est.fit(&sim_problem, identification, workspace, ctx).map_err(|e| {
                            ValidationError::estimation_msg(format!("SBC refit failed: {e}"))
                        })?;
                    let col = post
                        .effect_column()
                        .ok_or_else(|| ValidationError::estimation_msg("SBC: no effect column"))?;
                    let draws = post
                        .draws
                        .column(col)
                        .map_err(|e| ValidationError::estimation_msg(format!("SBC draws: {e}")))?;
                    let rank_draws = sbc_rank_draws(draws, &post.diagnostics)?;
                    let rank = rank_draws.iter().filter(|&&d| d < *true_effect).count();
                    Ok::<_, ValidationError>((
                        u32::try_from(rank).unwrap_or(u32::MAX),
                        rank_draws.len(),
                    ))
                })())
            },
        )?;
        let _admitted = crate::common::admit_cells(Some(ctx), self.n_reps as usize, 1)?;
        let mut ranks = Vec::with_capacity(self.n_reps as usize);
        let mut rank_draw_count = None;
        for outcome in outcomes {
            let (rank, draw_count) = outcome?;
            if rank_draw_count.is_some_and(|count| count != draw_count) {
                return Err(ValidationError::NotApplicable {
                    message: "SBC refits returned differing numbers of independent rank draws",
                });
            }
            rank_draw_count = Some(draw_count);
            ranks.push(rank);
        }

        let rank_draw_count = rank_draw_count.unwrap_or(self.n_draws);
        let rank_bins = if estimator.backend == antecedent_estimate::BayesianBackendKind::Hmc {
            rank_draw_count + 1
        } else {
            SBC_BINS
        };
        let n_d = rank_draw_count.max(1) as f64;
        let fracs: Vec<f64> = ranks.iter().map(|&r| f64::from(r) / n_d).collect();
        let mean_rank_frac =
            reduce_posterior_draws(&fracs, PosteriorReduceOp::Mean, &ctx.kernel_policy)
                .unwrap_or(0.5);
        let mut counts = vec![0.0; rank_bins];
        for &r in &ranks {
            let bin = if rank_bins == SBC_BINS {
                sbc_bin(r, rank_draw_count)
            } else {
                ((r as usize) * rank_bins / (rank_draw_count + 1)).min(rank_bins - 1)
            };
            counts[bin] += 1.0;
        }
        // At least 5 expected per bin (enforced above), so no floor on the denominator.
        let expected = f64::from(self.n_reps) / rank_bins as f64;
        let mut chi2 = 0.0;
        for c in counts {
            let d = c - expected;
            chi2 += d * d / expected;
        }
        Ok(SbcReport {
            ranks: Arc::from(ranks),
            mean_rank_frac,
            uniformity_stat: chi2,
            rank_draw_count,
            rank_bins,
        })
    }

    /// Convert to a refutation report.
    ///
    /// Passes only when **both** hold:
    /// - the mean rank fraction is in `[0.35, 0.65]` (catches gross location bias), and
    /// - the χ² uniformity statistic over 10 independent-draw bins or five
    ///   independent HMC-chain bins is below its 0.99 critical value (catches symmetric-about-0.5 U/M-shaped rank
    ///   distributions — overdispersed / underdispersed posteriors — that a mean-only
    ///   band cannot see because they average out to ≈0.5).
    #[must_use]
    pub fn to_report(&self, report: &SbcReport, original_ate: f64) -> RefutationReport {
        let mean_ok = (0.35..=0.65).contains(&report.mean_rank_frac);
        let critical = if report.rank_bins == 5 {
            SBC_CHI2_CRITICAL_4DF_P99
        } else {
            SBC_CHI2_CRITICAL_9DF_P99
        };
        let uniform_ok = report.uniformity_stat.is_finite() && report.uniformity_stat <= critical;
        let passed = mean_ok && uniform_ok;
        let failure_condition = if passed {
            None
        } else if !mean_ok && !uniform_ok {
            Some(Arc::from(format!(
                "SBC mean rank frac {:.3} outside [0.35, 0.65] and χ²={:.3} exceeds critical \
                     value {critical:.3} ({} df, p=0.99)",
                report.mean_rank_frac,
                report.uniformity_stat,
                report.rank_bins - 1
            )))
        } else if !mean_ok {
            Some(Arc::from(format!(
                "SBC mean rank frac {:.3} outside [0.35, 0.65]",
                report.mean_rank_frac
            )))
        } else {
            Some(Arc::from(format!(
                "SBC rank distribution non-uniform: χ²={:.3} exceeds critical value \
                     {critical:.3} ({} df, p=0.99); mean rank frac {:.3} looked \
                     fine but ranks are not uniformly distributed (U- or M-shaped)",
                report.uniformity_stat,
                report.rank_bins - 1,
                report.mean_rank_frac
            )))
        };
        RefutationReport::builder("sbc")
            .original_ate(original_ate)
            .refuted_ate(report.mean_rank_frac)
            .comparison(report.uniformity_stat)
            .informative(true)
            .passed(passed)
            .failure_condition(failure_condition)
            .replicates(self.n_reps)
            .build()
    }
}

/// χ² critical value at the 0.99 quantile with 9 degrees of freedom (10 rank bins − 1).
///
/// Used to gate [`SimulationBasedCalibration::to_report`]'s uniformity check: SBC ranks
/// should be uniform under a well-calibrated posterior, and this is the standard
/// one-sided χ² goodness-of-fit critical value (e.g. Talts et al. 2018 §3 use the
/// same chi-square uniformity diagnostic).
pub(super) const SBC_CHI2_CRITICAL_9DF_P99: f64 = 21.666;
/// Chi² 0.99 quantile with four degrees of freedom (five chain-rank bins).
const SBC_CHI2_CRITICAL_4DF_P99: f64 = 13.277;

#[cfg(test)]
mod seed_tests {
    use super::sbc_refit_seed;

    #[test]
    fn replicate_seeds_are_the_historical_bijection() {
        // Odd multiplier: distinct replicates cannot land on the same seed.
        assert_eq!(sbc_refit_seed(0x11, 0), 0x11);
        assert_eq!(sbc_refit_seed(0x11, 1), 0x11 ^ 0x9E37);
        let mut seen = std::collections::BTreeSet::new();
        for rep in 0..4096_u32 {
            assert!(seen.insert(sbc_refit_seed(0x11, rep)));
        }
    }
}
