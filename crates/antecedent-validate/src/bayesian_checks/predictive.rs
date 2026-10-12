//! Prior and posterior predictive checks.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{CausalRng, ExecutionContext, KernelPolicy};
use antecedent_estimate::{
    BayesianGComputationAte, CausalPosterior, PreparedBayesianProblem, SerialDependence,
    tempering_kappa_from_notes,
};
use antecedent_kernels::standard_normal;
use antecedent_prob::{GaussianVarianceModel, PriorSet, sample_gamma, sample_inv_gamma};
use antecedent_stats::GlmFamily;

use crate::common::RefutationReport;
use crate::error::ValidationError;

use super::calibration::{push_mean_and_dispersion, summarize_predictive_check};
use super::sbc::{prior_correlation_cholesky, prior_standard_normals};

/// Result of a prior or posterior predictive check.
///
/// Carries location and dispersion axes: (1) location, via the mean of `mean_y` over
/// simulations, and (2) dispersion, via the mean of the per-simulation
/// cross-observation SD of replicated outcomes. A model whose predictive mean is
/// unbiased but whose predictive spread is badly wrong (e.g. off by 5×) fails
/// on the dispersion axis even though the location axis looks fine.
///
/// On a *posterior* check those two axes have power only against a badly wrong prior or a
/// wrong noise level, not against misspecification: a regression with an intercept
/// reproduces the outcome mean by the normal equations, the Gaussian replicate SD is drawn
/// so as to match the observed residual SD, and a Bernoulli SD is a function of its mean.
/// The `residual` axes therefore add realized discrepancies the fit does not optimize
/// (residual skewness, excess kurtosis, curvature against the fitted mean, and a variance
/// trend against it; Gelman, Meng and Stern 1996), which respond to nonlinearity,
/// heteroskedasticity and heavy tails.
///
/// Replicates are draws from the observation model, `y_rep = g⁻¹(Xβ) + noise`
/// (Gaussian noise at a residual-variance draw, a Bernoulli draw for binary
/// likelihoods, a Poisson draw for counts), so their spread is comparable with
/// the observed outcome's, which carries the residual variance. See
/// [`PredictiveNoise`] for the one case that omits the noise term.
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
    /// Lower and upper inclusive Monte Carlo tails for the location statistic.
    /// Retained separately because taking the smaller tail does not commute
    /// with graph-mixture averaging.
    pub location_tails: [f64; 2],
    /// Observed dispersion statistic: sample SD of the outcome across rows.
    pub observed_dispersion: f64,
    /// Mean, across simulations, of the per-simulation cross-observation SD of
    /// predicted values (the dispersion test statistic).
    pub predictive_dispersion_mean: f64,
    /// Two-sided Monte Carlo p-value of `observed_dispersion` under the
    /// simulated dispersion-statistic distribution.
    pub dispersion_p_value: f64,
    /// Lower and upper inclusive Monte Carlo tails for dispersion.
    pub dispersion_tails: [f64; 2],
    /// Number of predictive simulations.
    pub n_sims: u32,
    /// Temporal discrepancy axis (lag-1 residual autocorrelation) on time-ordered
    /// designs; `None` for exchangeable rows and for prior checks.
    pub serial: Option<SerialDiscrepancy>,
    /// Realized residual-shape discrepancies of a posterior check (see the type docs), each
    /// with its own tails for information; the verdict uses [`Self::residual_p_value`]. Empty
    /// for prior checks, for families without a residual variance function (negative
    /// binomial), and for fewer than [`MIN_RESIDUAL_ROWS`] rows.
    pub residual: Vec<ResidualDiscrepancy>,
    /// Omnibus upper-tail p-value of the residual axes (`1` when there are none): at each draw
    /// the largest standardized departure of the four statistics from their replicate
    /// distribution, observed against replicated. One p-value for the whole family, so the
    /// verdict needs no multiplicity correction and keeps its Monte Carlo resolution `1/(n+1)`.
    pub residual_p_value: f64,
    /// How the replicates carry observation noise.
    pub noise: PredictiveNoise,
}

/// Fewest design rows for which the residual-shape axes are computed.
pub(crate) const MIN_RESIDUAL_ROWS: usize = 8;

/// A realized residual-shape statistic of a posterior predictive check.
///
/// Each statistic is computed on Pearson residuals `(y − μ) / sqrt(V(μ))` at a posterior draw,
/// once for the observed outcome and once for a replicate drawn at the same parameters.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ResidualStatistic {
    /// Third standardized moment of the residuals (asymmetry / skewed noise).
    Skewness,
    /// Fourth standardized moment minus three (heavy tails).
    ExcessKurtosis,
    /// Correlation of the residuals with `(μ − mean μ)²`: neglected curvature in the mean.
    Curvature,
    /// Correlation of the squared residuals with `μ`: variance that changes with the mean.
    VarianceTrend,
}

impl ResidualStatistic {
    /// Every statistic, in report order.
    pub const ALL: [Self; 4] =
        [Self::Skewness, Self::ExcessKurtosis, Self::Curvature, Self::VarianceTrend];

    /// Stable name for messages.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Skewness => "residual skewness",
            Self::ExcessKurtosis => "residual excess kurtosis",
            Self::Curvature => "curvature against the fitted mean",
            Self::VarianceTrend => "variance trend against the fitted mean",
        }
    }
}

/// One residual-shape axis of a [`PredictiveCheckReport`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResidualDiscrepancy {
    /// Statistic.
    pub statistic: ResidualStatistic,
    /// Mean, over posterior draws, of the statistic on the observed outcome.
    pub observed: f64,
    /// Mean, over posterior draws, of the statistic on the replicates.
    pub predictive_mean: f64,
    /// Two-sided posterior predictive p-value.
    pub p_value: f64,
    /// Lower / upper inclusive Monte Carlo tails `P(T_rep ≤ T_obs)`, `P(T_rep ≥ T_obs)`.
    pub tails: [f64; 2],
}

/// How predictive replicates carry the likelihood's observation noise.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum PredictiveNoise {
    /// Replicates are draws from the full observation model.
    #[default]
    Likelihood,
    /// Prior predictive under a residual-variance prior without a finite mean
    /// (e.g. the weakly informative `InvGamma(1e-3, 1e-3)` default), whose noise
    /// term would dominate any replicated statistic. Replicates are the fitted
    /// means `g⁻¹(Xβ)` only. The location axis then checks the observed mean
    /// against the prior on the average regression mean (it omits the `σ²/n`
    /// sampling noise of the observed mean). The dispersion axis keeps only its
    /// lower tail — the coefficient prior alone already spreads the replicates
    /// wider than the data, which noise could only widen further — and reports
    /// the upper tail as `1`, because unbounded noise reaches any larger spread.
    MeanOnlyImproperScale,
}

/// Posterior predictive check of serial dependence in the outcome residual.
///
/// Discrepancy `T(y, β) = Σ_t r_t r_{t-1} / Σ_t r_t²` with `r = y − Xβ` (lag-1
/// residual autocorrelation). For each posterior draw the realized value uses the
/// observed outcome; the replicated value uses a replicate drawn from the iid
/// Gaussian likelihood (`T` is scale-free, so the residual scale cancels). A small
/// p-value says the rows are not exchangeable under the fitted mean — the
/// misspecification the serial-dependence correction of temporal Bayesian
/// intervals exists for.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SerialDiscrepancy {
    /// Residual lag the statistic uses (always `1`).
    pub lag: u32,
    /// Mean realized lag-1 residual autocorrelation over posterior draws.
    pub observed: f64,
    /// Mean replicated lag-1 residual autocorrelation (≈ `−1/n` under iid rows).
    pub predictive_mean: f64,
    /// Two-sided posterior predictive p-value.
    pub p_value: f64,
    /// Lower / upper inclusive Monte Carlo tails `P(T_rep ≤ T_obs)`, `P(T_rep ≥ T_obs)`.
    pub tails: [f64; 2],
    /// Long-run-variance tempering factor `κ̂` recorded on the checked posterior
    /// (see `antecedent_estimate::serial_dependence`), when its likelihood was
    /// tempered for serial dependence. A failing serial axis on a tempered
    /// posterior records the dependence the tempering already widened the
    /// interval for; it does not refute the tempered interval.
    pub tempering_kappa: Option<f64>,
}

impl PredictiveCheckReport {
    /// Convert to a suite [`RefutationReport`] using a two-sided α threshold on both
    /// the location (`p_value`) and dispersion (`dispersion_p_value`) axes.
    ///
    /// A single mean-only statistic cannot distinguish a well-calibrated predictive
    /// distribution from one with the right mean but the wrong spread (U/M-shaped
    /// misspecification on variance), so both axes must clear the threshold.
    #[must_use]
    pub fn to_refutation_report(&self, original_ate: f64, alpha: f64) -> RefutationReport {
        let name = match self.kind {
            PredictiveCheckKind::Prior => "prior_predictive",
            PredictiveCheckKind::Posterior => "posterior_predictive",
        };
        let mean_ok = self.p_value.is_finite() && self.p_value >= alpha;
        let dispersion_ok = self.dispersion_p_value.is_finite() && self.dispersion_p_value >= alpha;
        let serial_ok = self.serial.is_none_or(|s| s.p_value.is_finite() && s.p_value >= alpha);
        // One omnibus p-value covers the whole residual family (see `residual_p_value`).
        let residual_ok = self.residual_p_value.is_finite() && self.residual_p_value >= alpha;
        let passed = mean_ok && dispersion_ok && serial_ok && residual_ok;
        let comparison = self
            .serial
            .map(|serial| serial.p_value)
            .into_iter()
            .chain((!self.residual.is_empty()).then_some(self.residual_p_value))
            .fold(self.p_value.min(self.dispersion_p_value), f64::min);
        RefutationReport {
            refuter: Arc::from(name),
            original_ate,
            refuted_ate: self.predictive_mean,
            comparison,
            informative: true,
            passed,
            failure_condition: if passed {
                None
            } else {
                let base = if mean_ok && dispersion_ok {
                    None
                } else if !mean_ok && !dispersion_ok {
                    Some(format!(
                        "predictive check failed on mean (p={} < alpha={alpha}) and dispersion \
                         (p={} < alpha={alpha})",
                        self.p_value, self.dispersion_p_value
                    ))
                } else if !mean_ok {
                    Some(format!("predictive check failed (p={} < alpha={alpha})", self.p_value))
                } else if self.kind == PredictiveCheckKind::Prior
                    && self.dispersion_tails[0] <= self.dispersion_tails[1]
                {
                    Some(format!(
                        "prior predictive dispersion check failed (p={} < alpha={alpha}): the \
                         prior predictive spread exceeds the observed spread, i.e. the prior is \
                         diffuse relative to the data (weakly informative defaults do this); a \
                         statement about the prior, not evidence against the likelihood",
                        self.dispersion_p_value
                    ))
                } else {
                    Some(format!(
                        "predictive dispersion check failed (p={} < alpha={alpha}); predictive \
                         spread does not match observed spread even though the mean matches",
                        self.dispersion_p_value
                    ))
                };
                let serial = self.serial.filter(|_| !serial_ok).map(|serial| {
                    let tempering = match serial.tempering_kappa {
                        Some(kappa) if kappa > 1.0 => format!(
                            "; the published interval is already tempered for this dependence \
                             (long-run-variance factor kappa={kappa:.4}, a generalized posterior), \
                             so this is an informative diagnostic of the dependence, not a \
                             refutation of the tempered interval"
                        ),
                        Some(_) => "; the likelihood was tempered but the long-run-variance \
                                    factor stayed at 1, so the interval carries no widening for \
                                    this dependence"
                            .to_owned(),
                        None => String::new(),
                    };
                    format!(
                        "posterior predictive serial-dependence check failed: lag-{} residual \
                         autocorrelation {:.4} vs replicated {:.4} (p={} < alpha={alpha}); rows \
                         are not exchangeable under the iid likelihood{tempering}",
                        serial.lag, serial.observed, serial.predictive_mean, serial.p_value
                    )
                });
                let residual = (!residual_ok).then(|| {
                    let axes = self
                        .residual
                        .iter()
                        .map(|r| {
                            format!(
                                "{} (observed {:.4} vs replicated {:.4})",
                                r.statistic.name(),
                                r.observed,
                                r.predictive_mean,
                            )
                        })
                        .collect::<Vec<_>>()
                        .join(", ");
                    format!(
                        "posterior predictive residual-shape check failed (omnibus p={} < \
                         alpha={alpha}): {axes}; the fitted model does not reproduce the shape \
                         of its own residuals (nonlinearity, heteroskedasticity or heavy tails)",
                        self.residual_p_value
                    )
                });
                Some(Arc::from(
                    base.into_iter().chain(serial).chain(residual).collect::<Vec<_>>().join("; "),
                ))
            },
            replicates: self.n_sims,
        }
    }

    /// Mixture-weighted aggregation of per-atom predictive checks of the same kind.
    ///
    /// Weights are graph-posterior mass. Location, dispersion statistics, and each
    /// one-sided tail are weighted means. Two-sided probabilities are computed
    /// from the mixed tails; `n_sims` is the maximum across atoms.
    ///
    /// `predictive_sd` is the SD of the *mixture* predictive distribution, so it uses
    /// the law of total variance — `sqrt(Σw σ² + Σw (μ − μ̄)²)`, the within-atom spread
    /// plus the spread between atom centers. A plain weighted mean of the per-atom SDs
    /// would drop the second term and report an envelope narrower than any completion
    /// disagreement justifies.
    ///
    /// Atoms with non-positive weight, and atoms whose `kind` differs from the first
    /// contributing atom, are dropped. Returns `None` when nothing contributes.
    #[must_use]
    #[allow(
        clippy::needless_range_loop,
        reason = "index-based loop over parallel arrays; the subscript also names the matrix coordinate"
    )]
    #[allow(
        clippy::too_many_lines,
        reason = "mixture-weights every field of the per-atom check summary, one statement each"
    )]
    pub fn mixture_weighted(items: &[(f64, &Self)]) -> Option<Self> {
        if items.iter().any(|(weight, _)| !weight.is_finite()) {
            return None;
        }
        let first = items.iter().find(|(w, _)| *w > 0.0)?.1;
        let kind = first.kind;
        let first_residual = first.residual.as_slice();
        let atoms: Vec<_> =
            items.iter().filter(|(w, report)| *w > 0.0 && report.kind == kind).collect();
        let max_weight = atoms.iter().map(|(w, _)| *w).fold(0.0_f64, f64::max);
        let total: f64 = atoms.iter().map(|(w, _)| w / max_weight).sum();
        let mut observed = 0.0;
        let mut predictive_mean = 0.0;
        let mut location_tails = [0.0; 2];
        let mut observed_dispersion = 0.0;
        let mut predictive_dispersion_mean = 0.0;
        let mut dispersion_tails = [0.0; 2];
        let mut n_sims = 0u32;
        for &&(w, report) in &atoms {
            if !report.observed.is_finite()
                || !report.predictive_mean.is_finite()
                || !report.predictive_sd.is_finite()
                || report.predictive_sd < 0.0
                || !report.observed_dispersion.is_finite()
                || !report.predictive_dispersion_mean.is_finite()
                || report
                    .location_tails
                    .iter()
                    .chain(&report.dispersion_tails)
                    .any(|tail| !tail.is_finite() || !(0.0..=1.0).contains(tail))
            {
                return None;
            }
            let weight = (w / max_weight) / total;
            observed += weight * report.observed;
            predictive_mean += weight * report.predictive_mean;
            observed_dispersion += weight * report.observed_dispersion;
            predictive_dispersion_mean += weight * report.predictive_dispersion_mean;
            for tail in 0..2 {
                location_tails[tail] += weight * report.location_tails[tail];
                dispersion_tails[tail] += weight * report.dispersion_tails[tail];
            }
            n_sims = n_sims.max(report.n_sims);
        }
        // The temporal axis mixes only when every contributing atom carries it.
        let serial = if atoms.iter().all(|(_, report)| report.serial.is_some()) {
            let mut observed_serial = 0.0;
            let mut predictive_serial = 0.0;
            let mut tails = [0.0; 2];
            // The mixture is tempered only as far as its least-tempered atom.
            let mut tempering_kappa = Some(f64::INFINITY);
            for &&(w, report) in &atoms {
                let serial = report.serial.expect("checked above");
                let weight = (w / max_weight) / total;
                observed_serial += weight * serial.observed;
                predictive_serial += weight * serial.predictive_mean;
                for tail in 0..2 {
                    tails[tail] += weight * serial.tails[tail];
                }
                tempering_kappa =
                    tempering_kappa.zip(serial.tempering_kappa).map(|(a, b)| a.min(b));
            }
            Some(SerialDiscrepancy {
                lag: 1,
                observed: observed_serial,
                predictive_mean: predictive_serial,
                p_value: (2.0 * tails[0].min(tails[1])).min(1.0),
                tails,
                tempering_kappa,
            })
        } else {
            None
        };
        // Residual axes mix only when every atom carries the same statistics in the same order.
        let residual = if atoms.iter().all(|(_, report)| {
            report.residual.len() == first_residual.len()
                && report
                    .residual
                    .iter()
                    .zip(first_residual)
                    .all(|(a, b)| a.statistic == b.statistic)
        }) {
            let mut mixed: Vec<ResidualDiscrepancy> = first_residual
                .iter()
                .map(|r| ResidualDiscrepancy {
                    statistic: r.statistic,
                    observed: 0.0,
                    predictive_mean: 0.0,
                    p_value: 1.0,
                    tails: [0.0; 2],
                })
                .collect();
            for &&(w, report) in &atoms {
                let weight = (w / max_weight) / total;
                for (out, r) in mixed.iter_mut().zip(&report.residual) {
                    out.observed += weight * r.observed;
                    out.predictive_mean += weight * r.predictive_mean;
                    for tail in 0..2 {
                        out.tails[tail] += weight * r.tails[tail];
                    }
                }
            }
            for out in &mut mixed {
                out.p_value = (2.0 * out.tails[0].min(out.tails[1])).min(1.0);
            }
            mixed
        } else {
            Vec::new()
        };
        let residual_p_value = if residual.is_empty() {
            1.0
        } else {
            atoms
                .iter()
                .map(|&&(w, report)| ((w / max_weight) / total) * report.residual_p_value)
                .sum::<f64>()
        };
        // Centered total variance retains small spread around large locations.
        // Hypot also avoids squaring away representable tiny/large SDs.
        let mut predictive_sd = 0.0_f64;
        for &&(w, report) in &atoms {
            let root_weight = ((w / max_weight) / total).sqrt();
            predictive_sd = predictive_sd
                .hypot(root_weight * report.predictive_sd)
                .hypot(root_weight * (report.predictive_mean - predictive_mean));
        }
        Some(Self {
            kind,
            observed,
            predictive_mean,
            predictive_sd,
            p_value: (2.0 * location_tails[0].min(location_tails[1])).min(1.0),
            location_tails,
            observed_dispersion,
            predictive_dispersion_mean,
            dispersion_p_value: (2.0 * dispersion_tails[0].min(dispersion_tails[1])).min(1.0),
            dispersion_tails,
            n_sims,
            serial,
            residual,
            residual_p_value,
            noise: if atoms.iter().any(|(_, r)| r.noise == PredictiveNoise::MeanOnlyImproperScale) {
                PredictiveNoise::MeanOnlyImproperScale
            } else {
                PredictiveNoise::Likelihood
            },
        })
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

/// Prior predictive check: coefficient (and residual-variance) draws from the
/// prior, no data update, replicated outcomes vs the observed outcome.
#[derive(Clone, Debug)]
pub struct PriorPredictiveCheck {
    /// Simulations.
    pub n_sims: u32,
    /// RNG seed.
    pub seed: u64,
    /// Observation family (inverse link applied to η, then the family's noise).
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

    /// 200 sims under the observation family `estimator` fits, seeded from `ctx`.
    #[must_use]
    pub fn for_estimator(estimator: &BayesianGComputationAte, ctx: &ExecutionContext) -> Self {
        Self { n_sims: 200, seed: ctx.rng.master_seed(), family: estimator.glm_family() }
    }

    /// Set the simulation count (a latency tier's predictive-check budget).
    #[must_use]
    pub const fn with_n_sims(mut self, n_sims: u32) -> Self {
        self.n_sims = n_sims;
        self
    }

    /// Run against a prepared Bayesian design with a weakly informative prior.
    ///
    /// Prefer [`Self::check_with_prior`] when an analysis / composed prior is known.
    ///
    /// # Errors
    ///
    /// Empty design.
    pub fn check(
        &self,
        problem: &PreparedBayesianProblem,
        ctx: &ExecutionContext,
    ) -> Result<PredictiveCheckReport, ValidationError> {
        let p = problem.design.ncols;
        let prior = PriorSet::weakly_informative(p);
        self.check_with_prior(problem, &prior, ctx)
    }

    /// Run prior predictive check under an explicit coefficient prior.
    ///
    /// Gaussian replicates add `N(0, σ²)` noise with `σ²` from the prior's residual
    /// model (fixed when known; one `InvGamma` draw per simulation otherwise). A
    /// residual prior without a finite mean yields a
    /// [`PredictiveNoise::MeanOnlyImproperScale`] report.
    ///
    /// # Errors
    ///
    /// Empty design, missing Gaussian coefficient prior, or an invalid residual
    /// variance specification.
    pub fn check_with_prior(
        &self,
        problem: &PreparedBayesianProblem,
        prior: &PriorSet,
        ctx: &ExecutionContext,
    ) -> Result<PredictiveCheckReport, ValidationError> {
        let n = problem.design.nrows;
        let p = problem.design.ncols;
        if n == 0 || p == 0 {
            return Err(ValidationError::estimation_msg("empty design for PPC"));
        }
        let coef_prior = prior.gaussian_coefficients().ok_or_else(|| {
            ValidationError::estimation_msg("prior missing Gaussian coefficients for PPC")
        })?;
        if coef_prior.len() != p {
            return Err(ValidationError::estimation_msg(
                "prior coefficient dimension mismatch for PPC",
            ));
        }
        let residual = if self.family == GlmFamily::GaussianIdentity {
            match GaussianVarianceModel::from_prior_set(prior)
                .map_err(|e| ValidationError::estimation_msg(e.to_string()))?
            {
                GaussianVarianceModel::Known { sigma2 } => PriorResidual::Known(sigma2),
                // InvGamma(a, b) has a finite mean only for a > 1.
                GaussianVarianceModel::InvGamma { shape, scale } if shape > 1.0 => {
                    PriorResidual::InvGamma { shape, scale }
                }
                GaussianVarianceModel::InvGamma { .. } => PriorResidual::Improper,
            }
        } else {
            PriorResidual::None
        };
        let mut rng = CausalRng::from_seed(self.seed);
        let mut noise_rng = CausalRng::from_seed(self.seed ^ REPLICATE_NOISE_STREAM);
        let mut mean_summaries = Vec::with_capacity(self.n_sims as usize);
        let mut disp_summaries = Vec::with_capacity(self.n_sims as usize);
        let mut beta = vec![0.0; p];
        let mut y_pred = vec![0.0; n];
        let corr_chol = prior_correlation_cholesky(prior)?;
        for _ in 0..self.n_sims {
            // Draw β ~ prior once per simulation, then μ_i = g^{-1}(x_i'β).
            let z = prior_standard_normals(corr_chol.as_deref(), p, &mut rng);
            for c in 0..p {
                beta[c] = coef_prior.mean[c] + coef_prior.variance[c].sqrt() * z[c];
            }
            fitted_means(problem, &beta, self.family, &mut y_pred);
            let sigma = match residual {
                PriorResidual::Known(sigma2) => sigma2.sqrt(),
                PriorResidual::InvGamma { shape, scale } => {
                    sample_inv_gamma(shape, scale, &mut noise_rng).sqrt()
                }
                PriorResidual::Improper | PriorResidual::None => 0.0,
            };
            if residual != PriorResidual::Improper {
                add_observation_noise(self.family, sigma, &mut y_pred, &mut noise_rng);
            }
            push_mean_and_dispersion(
                &y_pred,
                ctx.kernel_policy,
                &mut mean_summaries,
                &mut disp_summaries,
            );
        }
        let noise = if residual == PriorResidual::Improper {
            PredictiveNoise::MeanOnlyImproperScale
        } else {
            PredictiveNoise::Likelihood
        };
        Ok(summarize_predictive_check(
            PredictiveCheckKind::Prior,
            &problem.design.outcome,
            ctx.kernel_policy,
            &mean_summaries,
            &disp_summaries,
            self.n_sims,
            noise,
        ))
    }
}

/// Residual-variance source for Gaussian prior predictive replicates.
#[derive(Clone, Copy, Debug, PartialEq)]
enum PriorResidual {
    /// Non-Gaussian family: the family's own noise, no σ².
    None,
    /// Known residual variance.
    Known(f64),
    /// Proper inverse-gamma prior with a finite mean.
    InvGamma { shape: f64, scale: f64 },
    /// Inverse-gamma prior without a finite mean.
    Improper,
}

/// RNG stream offset for replicate observation noise.
const REPLICATE_NOISE_STREAM: u64 = crate::streams::REPLICATE_NOISE;

/// `out[r] = g⁻¹(x_r'β)`.
///
/// Accumulates column by column over the column-major design (contiguous
/// loads, vectorizable); each row still sums its terms in column order, so the
/// result is the row-by-row dot product bit for bit.
fn fitted_means(
    problem: &PreparedBayesianProblem,
    beta: &[f64],
    family: GlmFamily,
    out: &mut [f64],
) {
    crate::common::column_major_dot(&problem.design.matrix, problem.design.nrows, beta, out);
    if family != GlmFamily::GaussianIdentity {
        for slot in out.iter_mut() {
            *slot = family.mean_from_eta(*slot);
        }
    }
}

/// Replace the fitted means in `y` by draws from the observation model: Gaussian
/// noise of SD `sigma`, a Bernoulli draw for binary families, a Poisson draw for
/// count families. Negative binomial replicates use the Poisson draw (the
/// dispersion `α` is not carried here), a lower bound on their spread.
fn add_observation_noise(family: GlmFamily, sigma: f64, y: &mut [f64], rng: &mut CausalRng) {
    match family {
        GlmFamily::GaussianIdentity => {
            if sigma > 0.0 {
                for value in y.iter_mut() {
                    *value += sigma * standard_normal(rng);
                }
            }
        }
        GlmFamily::BinomialLogit | GlmFamily::BinomialProbit => {
            for value in y.iter_mut() {
                *value = if rng.next_f64() < *value { 1.0 } else { 0.0 };
            }
        }
        GlmFamily::PoissonLog | GlmFamily::NegativeBinomial => {
            for value in y.iter_mut() {
                *value = sample_poisson(*value, rng);
            }
        }
    }
}

/// One Poisson(`lambda`) draw: Knuth multiplication in chunks of at most 500
/// (exact), with a rounded normal approximation above `10⁴`.
pub(super) fn sample_poisson(lambda: f64, rng: &mut CausalRng) -> f64 {
    if !lambda.is_finite() {
        return lambda;
    }
    if lambda <= 0.0 {
        return 0.0;
    }
    if lambda > 1e4 {
        return (lambda + lambda.sqrt() * standard_normal(rng)).round().max(0.0);
    }
    let mut remaining = lambda;
    let mut count = 0.0;
    while remaining > 0.0 {
        let step = remaining.min(500.0);
        remaining -= step;
        let limit = (-step).exp();
        let mut product = rng.next_f64();
        while product > limit {
            count += 1.0;
            product *= rng.next_f64();
        }
    }
    count
}

/// Posterior predictive check: replicated outcomes from posterior coefficient draws.
#[derive(Clone, Debug)]
pub struct PosteriorPredictiveCheck {
    /// Number of posterior draws to use (capped by available).
    pub n_sims: u32,
    /// Observation family (inverse link applied to η, then the family's noise).
    pub family: GlmFamily,
    /// Seed of the replicate-noise stream.
    pub seed: u64,
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
        Self { n_sims: 200, family: GlmFamily::GaussianIdentity, seed: 0 }
    }

    /// 200 draws under the observation family `estimator` fits, seeded from `ctx`.
    #[must_use]
    pub fn for_estimator(estimator: &BayesianGComputationAte, ctx: &ExecutionContext) -> Self {
        Self { n_sims: 200, family: estimator.glm_family(), seed: ctx.rng.master_seed() }
    }

    /// Set the draw cap (a latency tier's predictive-check budget).
    #[must_use]
    pub const fn with_n_sims(mut self, n_sims: u32) -> Self {
        self.n_sims = n_sims;
        self
    }

    /// Check using a fitted [`CausalPosterior`] that includes coefficient columns.
    ///
    /// Each posterior draw `β_d` yields one replicate `y_rep ~ p(y | β_d, σ²_d)`.
    /// For the Gaussian family `σ²_d` is drawn from its conditional reference
    /// posterior given `β_d`, `σ²_d = SSR(β_d) / χ²_n` (the residual-variance
    /// column is not retained on effect posteriors), so `(β_d, σ²_d)` is a joint
    /// draw up to the prior's contribution to the residual-variance conditional.
    ///
    /// # Errors
    ///
    /// Missing coefficients / empty draws.
    pub fn check(
        &self,
        problem: &PreparedBayesianProblem,
        posterior: &CausalPosterior,
    ) -> Result<PredictiveCheckReport, ValidationError> {
        self.check_with_policy(problem, posterior, KernelPolicy::default_policy())
    }

    /// [`Self::check`] using the kernel policy on `ctx`.
    ///
    /// Replicate observation noise stays on `self.seed` mixed with
    /// [`crate::streams::REPLICATE_NOISE`]. Simulation-based calibration mixes
    /// its refit seeds with [`antecedent_core::mix64`] separately.
    ///
    /// # Errors
    ///
    /// Missing coefficients / empty draws.
    pub fn check_with_context(
        &self,
        problem: &PreparedBayesianProblem,
        posterior: &CausalPosterior,
        ctx: &ExecutionContext,
    ) -> Result<PredictiveCheckReport, ValidationError> {
        self.check_with_policy(problem, posterior, ctx.kernel_policy)
    }

    #[allow(
        clippy::cast_possible_truncation,
        reason = "the draw count is the number of posterior draws simulated, far below 2^32"
    )]
    fn check_with_policy(
        &self,
        problem: &PreparedBayesianProblem,
        posterior: &CausalPosterior,
        policy: KernelPolicy,
    ) -> Result<PredictiveCheckReport, ValidationError> {
        let n = problem.design.nrows;
        let p = problem.design.ncols;
        let total = posterior.draws.n_draws;
        let n_draws = total.min(self.n_sims as usize);
        if n_draws == 0 {
            return Err(ValidationError::estimation_msg("no posterior draws for PPC"));
        }
        let mut rng = CausalRng::from_seed(self.seed ^ REPLICATE_NOISE_STREAM);
        let mut mean_summaries = Vec::with_capacity(n_draws);
        let mut disp_summaries = Vec::with_capacity(n_draws);
        let mut y_pred = vec![0.0; n];
        let mut beta = vec![0.0; p];
        let mut residual = ResidualTracker::new(self.family, n);
        for k in 0..n_draws {
            // Spread the used draws over the whole store: chain-concatenated MCMC draws would
            // otherwise contribute only the start of the first chain.
            let d = strided_draw_index(k, n_draws, total);
            // Coefficients depend only on (d, c); hoist the fallible draw
            // accessor out of the row loop (n×p calls per draw → p).
            for (c, slot) in beta.iter_mut().enumerate() {
                *slot = posterior.draws.get(d, c).map_err(ValidationError::from)?;
            }
            fitted_means(problem, &beta, self.family, &mut y_pred);
            if let Some(tracker) = residual.as_mut() {
                tracker.set_means(&y_pred);
            }
            let sigma = if self.family == GlmFamily::GaussianIdentity {
                let ssr: f64 =
                    problem.design.outcome.iter().zip(&y_pred).map(|(y, m)| (y - m).powi(2)).sum();
                // χ²_n = Gamma(n/2, rate 1/2).
                let chi2 = sample_gamma(n as f64 / 2.0, 0.5, &mut rng);
                (ssr / chi2.max(f64::MIN_POSITIVE)).sqrt()
            } else {
                0.0
            };
            add_observation_noise(self.family, sigma, &mut y_pred, &mut rng);
            if let Some(tracker) = residual.as_mut() {
                tracker.push(&problem.design.outcome, &y_pred);
            }
            push_mean_and_dispersion(&y_pred, policy, &mut mean_summaries, &mut disp_summaries);
        }
        let mut report = summarize_predictive_check(
            PredictiveCheckKind::Posterior,
            &problem.design.outcome,
            policy,
            &mean_summaries,
            &disp_summaries,
            n_draws as u32,
            PredictiveNoise::Likelihood,
        );
        if let Some((axes, omnibus)) = residual.map(ResidualTracker::finish) {
            report.residual = axes;
            report.residual_p_value = omnibus;
        }
        Ok(report)
    }

    /// [`Self::check`] plus the temporal [`SerialDiscrepancy`] axis.
    ///
    /// Design rows must be in time order (lag-aligned temporal designs are).
    /// `seed` drives the replicated residuals of the serial axis. The tempering
    /// factor recorded on the posterior's inference notes, if any, is carried on
    /// [`SerialDiscrepancy::tempering_kappa`].
    ///
    /// # Errors
    ///
    /// Missing coefficients / empty draws, or fewer than three rows.
    pub fn check_temporal(
        &self,
        problem: &PreparedBayesianProblem,
        posterior: &CausalPosterior,
        seed: u64,
    ) -> Result<PredictiveCheckReport, ValidationError> {
        let mut report = self.check(problem, posterior)?;
        report.serial = Some(serial_discrepancy(problem, posterior, self.n_sims, seed)?);
        Ok(report)
    }
}

/// Index of the `k`-th of `take` draws spread evenly over `total` stored draws
/// (`k · total / take`); the identity when `take == total`.
pub(super) fn strided_draw_index(k: usize, take: usize, total: usize) -> usize {
    k * total / take
}

/// Standard deviation of the Pearson-residual scale `sqrt(V(μ))` for `family`, or `None` when the
/// family's variance function is not carried here (negative binomial).
fn pearson_scale(family: GlmFamily, mu: f64) -> Option<f64> {
    const FLOOR: f64 = 1e-12;
    match family {
        GlmFamily::GaussianIdentity => Some(1.0),
        GlmFamily::BinomialLogit | GlmFamily::BinomialProbit => {
            Some((mu * (1.0 - mu)).max(FLOOR).sqrt())
        }
        GlmFamily::PoissonLog => Some(mu.max(FLOOR).sqrt()),
        GlmFamily::NegativeBinomial => None,
    }
}

/// Sample correlation of two equally long slices (0 when either is constant).
fn correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        let (dx, dy) = (x - ma, y - mb);
        sab += dx * dy;
        saa += dx * dx;
        sbb += dy * dy;
    }
    if saa > 0.0 && sbb > 0.0 { sab / (saa * sbb).sqrt() } else { 0.0 }
}

/// The four [`ResidualStatistic`]s of Pearson residuals `r`, given the fitted means' curvature
/// basis `curvature = (μ − mean μ)²` and the fitted means `mu`. Scale-free, so the residual
/// scale (`σ` of a Gaussian) cancels.
fn residual_statistics(r: &[f64], curvature: &[f64], mu: &[f64], squared: &mut [f64]) -> [f64; 4] {
    let n = r.len() as f64;
    let mean = r.iter().sum::<f64>() / n;
    let (mut m2, mut m3, mut m4) = (0.0, 0.0, 0.0);
    for (slot, v) in squared.iter_mut().zip(r) {
        let d = v - mean;
        m2 += d * d;
        m3 += d * d * d;
        m4 += d * d * d * d;
        *slot = v * v;
    }
    let (m2, m3, m4) = (m2 / n, m3 / n, m4 / n);
    let (skewness, kurtosis) =
        if m2 > 0.0 { (m3 / m2.powf(1.5), m4 / (m2 * m2) - 3.0) } else { (0.0, 0.0) };
    [skewness, kurtosis, correlation(r, curvature), correlation(squared, mu)]
}

/// Accumulates the realized-discrepancy comparison of [`PosteriorPredictiveCheck`]: at every
/// posterior draw the statistics of the observed outcome against those of a replicate drawn at
/// the same parameters (Gelman, Meng and Stern 1996).
struct ResidualTracker {
    family: GlmFamily,
    mu: Vec<f64>,
    scale: Vec<f64>,
    curvature: Vec<f64>,
    residual: Vec<f64>,
    squared: Vec<f64>,
    /// Per-draw statistics of the observed outcome and of the replicate.
    observed: Vec<[f64; 4]>,
    replicate: Vec<[f64; 4]>,
}

impl ResidualTracker {
    /// `None` when the family has no residual variance function or the design is too small.
    fn new(family: GlmFamily, n: usize) -> Option<Self> {
        if n < MIN_RESIDUAL_ROWS || pearson_scale(family, 0.5).is_none() {
            return None;
        }
        Some(Self {
            family,
            mu: vec![0.0; n],
            scale: vec![1.0; n],
            curvature: vec![0.0; n],
            residual: vec![0.0; n],
            squared: vec![0.0; n],
            observed: Vec::new(),
            replicate: Vec::new(),
        })
    }

    /// Record the fitted means of the current draw (before observation noise is added).
    #[allow(
        clippy::needless_range_loop,
        reason = "index-based loop over parallel arrays; the subscript also names the matrix coordinate"
    )]
    fn set_means(&mut self, fitted: &[f64]) {
        self.mu.copy_from_slice(fitted);
        let mean = fitted.iter().sum::<f64>() / fitted.len() as f64;
        for i in 0..fitted.len() {
            // `new` verified the family has a variance function.
            self.scale[i] = pearson_scale(self.family, fitted[i]).unwrap_or(1.0);
            self.curvature[i] = (fitted[i] - mean) * (fitted[i] - mean);
        }
    }

    #[allow(
        clippy::needless_range_loop,
        reason = "index-based loop over parallel arrays; the subscript also names the matrix coordinate"
    )]
    fn statistics(&mut self, y: &[f64]) -> [f64; 4] {
        for i in 0..y.len() {
            self.residual[i] = (y[i] - self.mu[i]) / self.scale[i];
        }
        residual_statistics(&self.residual, &self.curvature, &self.mu, &mut self.squared)
    }

    /// Compare the observed outcome with the replicate at the current draw.
    fn push(&mut self, observed: &[f64], replicate: &[f64]) {
        let obs = self.statistics(observed);
        let rep = self.statistics(replicate);
        self.observed.push(obs);
        self.replicate.push(rep);
    }

    /// Per-statistic discrepancies and the omnibus upper-tail p-value.
    fn finish(self) -> (Vec<ResidualDiscrepancy>, f64) {
        let draws = self.observed.len() as f64;
        let mean = |rows: &[[f64; 4]], k: usize| rows.iter().map(|row| row[k]).sum::<f64>() / draws;
        // Replicate mean and SD of each statistic define the scale on which a departure is
        // compared; a statistic with no replicate spread cannot depart and is skipped.
        let centre: Vec<f64> = (0..4).map(|k| mean(&self.replicate, k)).collect();
        let spread: Vec<f64> = (0..4)
            .map(|k| {
                let var =
                    self.replicate.iter().map(|row| (row[k] - centre[k]).powi(2)).sum::<f64>()
                        / (draws - 1.0).max(1.0);
                var.sqrt()
            })
            .collect();
        let departure = |row: &[f64; 4]| {
            (0..4)
                .filter(|&k| spread[k] > 0.0)
                .map(|k| (row[k] - centre[k]).abs() / spread[k])
                .fold(0.0_f64, f64::max)
        };
        let exceed = self
            .observed
            .iter()
            .zip(&self.replicate)
            .filter(|(obs, rep)| departure(rep) >= departure(obs))
            .count() as f64;
        let omnibus = (1.0 + exceed) / (1.0 + draws);
        let axes = ResidualStatistic::ALL
            .iter()
            .enumerate()
            .map(|(k, &statistic)| {
                let below =
                    self.observed.iter().zip(&self.replicate).filter(|(o, r)| r[k] <= o[k]).count();
                let above =
                    self.observed.iter().zip(&self.replicate).filter(|(o, r)| r[k] >= o[k]).count();
                let tails =
                    [(1.0 + below as f64) / (1.0 + draws), (1.0 + above as f64) / (1.0 + draws)];
                ResidualDiscrepancy {
                    statistic,
                    observed: mean(&self.observed, k),
                    predictive_mean: centre[k],
                    p_value: (2.0 * tails[0].min(tails[1])).min(1.0),
                    tails,
                }
            })
            .collect();
        (axes, omnibus)
    }
}

/// Lag-1 autocorrelation `Σ r_t r_{t-1} / Σ r_t²` (0 for a zero vector).
fn lag1_autocorrelation(r: &[f64]) -> f64 {
    let den: f64 = r.iter().map(|v| v * v).sum();
    if den <= 0.0 || !den.is_finite() {
        return 0.0;
    }
    r.windows(2).map(|w| w[0] * w[1]).sum::<f64>() / den
}

fn serial_discrepancy(
    problem: &PreparedBayesianProblem,
    posterior: &CausalPosterior,
    n_sims: u32,
    seed: u64,
) -> Result<SerialDiscrepancy, ValidationError> {
    let n = problem.design.nrows;
    let p = problem.design.ncols;
    if n < 3 {
        return Err(ValidationError::estimation_msg("serial PPC needs at least three rows"));
    }
    let total = posterior.draws.n_draws;
    let n_draws = total.min(n_sims as usize);
    if n_draws == 0 {
        return Err(ValidationError::estimation_msg("no posterior draws for serial PPC"));
    }
    let mut rng = CausalRng::from_seed(seed ^ crate::streams::SERIAL_PPC);
    let mut beta = vec![0.0; p];
    let mut resid = vec![0.0; n];
    let mut replicate = vec![0.0; n];
    let (mut observed_sum, mut predictive_sum) = (0.0, 0.0);
    let (mut below, mut above) = (0usize, 0usize);
    for k in 0..n_draws {
        let d = strided_draw_index(k, n_draws, total);
        for (c, slot) in beta.iter_mut().enumerate() {
            *slot = posterior.draws.get(d, c).map_err(ValidationError::from)?;
        }
        crate::common::column_major_dot(&problem.design.matrix, n, &beta, &mut resid);
        for (out, &y) in resid.iter_mut().zip(problem.design.outcome.iter()) {
            *out = y - *out;
        }
        for value in &mut replicate {
            *value = standard_normal(&mut rng);
        }
        let t_obs = lag1_autocorrelation(&resid);
        let t_rep = lag1_autocorrelation(&replicate);
        observed_sum += t_obs;
        predictive_sum += t_rep;
        below += usize::from(t_rep <= t_obs);
        above += usize::from(t_rep >= t_obs);
    }
    let draws = n_draws as f64;
    let tails = [(1.0 + below as f64) / (1.0 + draws), (1.0 + above as f64) / (1.0 + draws)];
    let tempered = problem.serial_dependence != SerialDependence::Iid;
    Ok(SerialDiscrepancy {
        lag: 1,
        observed: observed_sum / draws,
        predictive_mean: predictive_sum / draws,
        p_value: (2.0 * tails[0].min(tails[1])).min(1.0),
        tails,
        tempering_kappa: tempering_kappa_from_notes(&posterior.diagnostics.notes)
            .or(tempered.then_some(1.0)),
    })
}
