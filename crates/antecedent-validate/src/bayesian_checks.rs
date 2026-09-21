//! Prior/posterior predictive checks and prior sensitivity.
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

use antecedent_core::{CausalRng, ExecutionContext, KernelPolicy};
use antecedent_estimate::{
    BayesianGCompWorkspace, BayesianGComputationAte, CausalPosterior, PreparedBayesianProblem,
    SerialDependence, tempering_kappa_from_notes,
};
use antecedent_identify::IdentificationStatus;
use antecedent_kernels::{
    PosteriorReduceOp, quantile_type7_sorted, reduce_posterior_draws, standard_normal,
};
use antecedent_prob::{
    ExternalPriorSource, GaussianVarianceModel, HessianFactorization, PriorSensitivityFamily,
    PriorSensitivitySummary, PriorSet, PriorSpec, compose_external_priors_with_alphas,
    sample_gamma, sample_inv_gamma,
};
use antecedent_stats::GlmFamily;

use crate::common::RefutationReport;
use crate::error::ValidationError;

/// Result of a prior or posterior predictive check.
///
/// Carries **two** discriminating axes so a model cannot pass merely by getting
/// the predictive mean right: (1) location, via the mean of `mean_y` over
/// simulations, and (2) dispersion, via the mean of the per-simulation
/// cross-observation SD of replicated outcomes. A model whose predictive mean is
/// unbiased but whose predictive spread is badly wrong (e.g. off by 5×) fails
/// on the dispersion axis even though the location axis looks fine.
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
    /// How the replicates carry observation noise.
    pub noise: PredictiveNoise,
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
        let passed = mean_ok && dispersion_ok && serial_ok;
        let comparison = self.serial.map_or(self.p_value.min(self.dispersion_p_value), |serial| {
            self.p_value.min(self.dispersion_p_value).min(serial.p_value)
        });
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
                Some(Arc::from(base.into_iter().chain(serial).collect::<Vec<_>>().join("; ")))
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
    pub fn mixture_weighted(items: &[(f64, &Self)]) -> Option<Self> {
        if items.iter().any(|(weight, _)| !weight.is_finite()) {
            return None;
        }
        let kind = items.iter().find(|(w, _)| *w > 0.0)?.1.kind;
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
        for _ in 0..self.n_sims {
            // Draw β ~ prior once per simulation, then μ_i = g^{-1}(x_i'β).
            for c in 0..p {
                beta[c] =
                    coef_prior.mean[c] + coef_prior.variance[c].sqrt() * standard_normal(&mut rng);
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
const REPLICATE_NOISE_STREAM: u64 = 0x0B5E_4A7E_0001_D00D;

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
    let n = problem.design.nrows;
    out.fill(0.0);
    for (c, &b) in beta.iter().enumerate() {
        let column = &problem.design.matrix[c * n..(c + 1) * n];
        for (slot, &x) in out.iter_mut().zip(column) {
            *slot += x * b;
        }
    }
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
fn sample_poisson(lambda: f64, rng: &mut CausalRng) -> f64 {
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
        let n = problem.design.nrows;
        let p = problem.design.ncols;
        let n_draws = posterior.draws.n_draws.min(self.n_sims as usize);
        if n_draws == 0 {
            return Err(ValidationError::estimation_msg("no posterior draws for PPC"));
        }
        let policy = KernelPolicy::default_policy();
        let mut rng = CausalRng::from_seed(self.seed ^ REPLICATE_NOISE_STREAM);
        let mut mean_summaries = Vec::with_capacity(n_draws);
        let mut disp_summaries = Vec::with_capacity(n_draws);
        let mut y_pred = vec![0.0; n];
        let mut beta = vec![0.0; p];
        for d in 0..n_draws {
            // Coefficients depend only on (d, c); hoist the fallible draw
            // accessor out of the row loop (n×p calls per draw → p).
            for (c, slot) in beta.iter_mut().enumerate() {
                *slot = posterior.draws.get(d, c).map_err(ValidationError::from)?;
            }
            fitted_means(problem, &beta, self.family, &mut y_pred);
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
            push_mean_and_dispersion(&y_pred, policy, &mut mean_summaries, &mut disp_summaries);
        }
        Ok(summarize_predictive_check(
            PredictiveCheckKind::Posterior,
            &problem.design.outcome,
            policy,
            &mean_summaries,
            &disp_summaries,
            n_draws as u32,
            PredictiveNoise::Likelihood,
        ))
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
    let n_draws = posterior.draws.n_draws.min(n_sims as usize);
    if n_draws == 0 {
        return Err(ValidationError::estimation_msg("no posterior draws for serial PPC"));
    }
    let mut rng = CausalRng::from_seed(seed ^ 0x5E71_A1D0_C0DE_0001);
    let mut beta = vec![0.0; p];
    let mut resid = vec![0.0; n];
    let mut replicate = vec![0.0; n];
    let (mut observed_sum, mut predictive_sum) = (0.0, 0.0);
    let (mut below, mut above) = (0usize, 0usize);
    for d in 0..n_draws {
        for (c, slot) in beta.iter_mut().enumerate() {
            *slot = posterior.draws.get(d, c).map_err(ValidationError::from)?;
        }
        for (r, out) in resid.iter_mut().enumerate() {
            let fitted: f64 =
                beta.iter().enumerate().map(|(c, b)| problem.design.matrix[c * n + r] * b).sum();
            *out = problem.design.outcome[r] - fitted;
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

/// Default max relative range of effect means across the prior-sensitivity grid.
pub const DEFAULT_MAX_RELATIVE_PRIOR_RANGE: f64 = 0.5;

/// Prior sensitivity grid: isotropic scales, external α multipliers, **or**
/// variance multipliers around the resolved prior in force.
///
/// Exactly one grid is non-empty. [`Self::evaluate`] (isotropic) is only the
/// right grid when no prior was supplied; a staged / transferred prior uses
/// [`Self::evaluate_resolved_prior`] and a prior-bank compose uses
/// [`Self::evaluate_external_alpha`].
#[derive(Clone, Debug)]
pub struct PriorSensitivity {
    /// Prior scales (σ of isotropic Gaussian coefficient prior). Empty in α mode.
    pub scales: Arc<[f64]>,
    /// Multipliers on post-conflict applied alphas. Empty in isotropic scale mode.
    pub alphas: Arc<[f64]>,
    /// Multipliers on the resolved prior's coefficient variances (means kept).
    pub variance_multipliers: Arc<[f64]>,
    /// Fail when `(max−min) / scale` exceeds this, where `scale` is
    /// `max(|means…|, |original_ate|, ε)`.
    pub max_relative_range: f64,
}

/// Inputs for external α-multiplier prior sensitivity.
#[derive(Clone, Copy, Debug)]
pub struct ExternalAlphaSensitivity<'a> {
    /// Hydrated external sources (same order as composition).
    pub sources: &'a [ExternalPriorSource],
    /// Post-conflict applied alphas (length must match `sources`).
    pub alphas_applied: &'a [f64],
}

impl Default for PriorSensitivity {
    fn default() -> Self {
        Self::standard_grid()
    }
}

impl PriorSensitivity {
    /// Standard isotropic grid `{0.5, 1, 2, 5, 10, 20}` with [`DEFAULT_MAX_RELATIVE_PRIOR_RANGE`].
    #[must_use]
    pub fn standard_grid() -> Self {
        Self {
            scales: Arc::from(vec![0.5, 1.0, 2.0, 5.0, 10.0, 20.0]),
            alphas: Arc::from([]),
            variance_multipliers: Arc::from([]),
            max_relative_range: DEFAULT_MAX_RELATIVE_PRIOR_RANGE,
        }
    }

    /// Standard resolved-prior grid: coefficient variances × `{0.25, 0.5, 1, 2, 4, 10}`.
    ///
    /// Multiplier `1` reproduces the prior in force; smaller multipliers tighten it
    /// around its own means, larger ones weaken it.
    #[must_use]
    pub fn standard_resolved_grid() -> Self {
        Self {
            scales: Arc::from([]),
            alphas: Arc::from([]),
            variance_multipliers: Arc::from(vec![0.25, 0.5, 1.0, 2.0, 4.0, 10.0]),
            max_relative_range: DEFAULT_MAX_RELATIVE_PRIOR_RANGE,
        }
    }

    /// Standard external-α multiplier grid `{0, 0.25, 0.5, 0.75, 1}`.
    ///
    /// Multiplier `0` is baseline-only; `1` uses full post-conflict applied alphas.
    #[must_use]
    pub fn standard_alpha_grid() -> Self {
        Self {
            scales: Arc::from([]),
            alphas: Arc::from(vec![0.0, 0.25, 0.5, 0.75, 1.0]),
            variance_multipliers: Arc::from([]),
            max_relative_range: DEFAULT_MAX_RELATIVE_PRIOR_RANGE,
        }
    }

    fn grid_len(&self) -> usize {
        if !self.variance_multipliers.is_empty() {
            self.variance_multipliers.len()
        } else if self.alphas.is_empty() {
            self.scales.len()
        } else {
            self.alphas.len()
        }
    }

    /// Refit at each variance multiplier around the resolved prior in force.
    ///
    /// `estimator.prior` must be the prior the reported posterior used. Coefficient
    /// means are kept; every coefficient variance is multiplied by the grid value;
    /// residual-variance and restriction specs are unchanged.
    ///
    /// # Errors
    ///
    /// Empty grid, no resolved prior on `estimator`, invalid multipliers, or fit failures.
    pub fn evaluate_resolved_prior(
        &self,
        estimator: &BayesianGComputationAte,
        problem: &PreparedBayesianProblem,
        identification: IdentificationStatus,
        workspace: &mut BayesianGCompWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<(PriorSensitivitySummary, Vec<CausalPosterior>), ValidationError> {
        if self.variance_multipliers.is_empty() {
            return Err(ValidationError::estimation_msg(
                "prior sensitivity variance-multiplier grid is empty",
            ));
        }
        let resolved = estimator.prior.as_ref().ok_or_else(|| {
            ValidationError::estimation_msg(
                "resolved-prior sensitivity requires the prior in force (estimator.prior is unset)",
            )
        })?;
        let mut means = Vec::with_capacity(self.variance_multipliers.len());
        let mut sds = Vec::with_capacity(self.variance_multipliers.len());
        let mut posts = Vec::with_capacity(self.variance_multipliers.len());
        for &mult in self.variance_multipliers.iter() {
            if !mult.is_finite() || mult <= 0.0 {
                return Err(ValidationError::estimation_msg(
                    "prior sensitivity variance multiplier must be finite and positive",
                ));
            }
            let est = BayesianGComputationAte {
                n_draws: estimator.n_draws.min(200),
                prior: Some(scale_coefficient_variances(resolved, mult)?),
                ..estimator.clone()
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
                family: PriorSensitivityFamily::ResolvedPriorVariance,
                prior_scales: Arc::from([]),
                alphas: Arc::from([]),
                variance_multipliers: Arc::clone(&self.variance_multipliers),
                effect_means: Arc::from(means),
                effect_sds: Arc::from(sds),
            },
            posts,
        ))
    }

    /// Refit Bayesian g-comp at each prior scale; return sensitivity summary.
    ///
    /// # Errors
    ///
    /// Fit failures or empty scale grid.
    pub fn evaluate(
        &self,
        estimator: &BayesianGComputationAte,
        problem: &PreparedBayesianProblem,
        identification: IdentificationStatus,
        workspace: &mut BayesianGCompWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<(PriorSensitivitySummary, Vec<CausalPosterior>), ValidationError> {
        if self.scales.is_empty() {
            return Err(ValidationError::estimation_msg(
                "prior sensitivity scale grid is empty (use evaluate_external_alpha for α mode)",
            ));
        }
        let mut means = Vec::with_capacity(self.scales.len());
        let mut sds = Vec::with_capacity(self.scales.len());
        let mut posts = Vec::with_capacity(self.scales.len());
        for &scale in self.scales.iter() {
            let est = BayesianGComputationAte {
                prior_scale: scale,
                n_draws: estimator.n_draws.min(200),
                prior: None,
                ..estimator.clone()
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
                family: PriorSensitivityFamily::IsotropicScale,
                prior_scales: Arc::clone(&self.scales),
                alphas: Arc::from([]),
                variance_multipliers: Arc::from([]),
                effect_means: Arc::from(means),
                effect_sds: Arc::from(sds),
            },
            posts,
        ))
    }

    /// Refit at each α-multiplier on post-conflict applied alphas (external prior bank).
    ///
    /// For multiplier `m`, composed alphas are `m * alphas_applied[k]` (clamped to `[0, 1]`).
    ///
    /// # Errors
    ///
    /// Empty α grid, length mismatch, compose failures, or fit failures.
    pub fn evaluate_external_alpha(
        &self,
        estimator: &BayesianGComputationAte,
        problem: &PreparedBayesianProblem,
        identification: IdentificationStatus,
        workspace: &mut BayesianGCompWorkspace,
        ctx: &ExecutionContext,
        external: ExternalAlphaSensitivity<'_>,
    ) -> Result<(PriorSensitivitySummary, Vec<CausalPosterior>), ValidationError> {
        if self.alphas.is_empty() {
            return Err(ValidationError::estimation_msg("prior sensitivity alpha grid is empty"));
        }
        if external.sources.len() != external.alphas_applied.len() {
            return Err(ValidationError::estimation_msg(
                "evaluate_external_alpha: sources / alphas_applied length mismatch",
            ));
        }
        let n_coef = problem.design.ncols;
        let baseline = PriorSet::weakly_informative(n_coef);
        let requested: Vec<f64> = external.sources.iter().map(|s| s.weight.alpha).collect();
        let mut means = Vec::with_capacity(self.alphas.len());
        let mut sds = Vec::with_capacity(self.alphas.len());
        let mut posts = Vec::with_capacity(self.alphas.len());
        for &mult in self.alphas.iter() {
            if !mult.is_finite() || !(0.0..=1.0).contains(&mult) {
                return Err(ValidationError::estimation_msg(
                    "prior sensitivity alpha multiplier must be finite and in [0, 1]",
                ));
            }
            let scaled: Vec<f64> =
                external.alphas_applied.iter().map(|&a| (a * mult).clamp(0.0, 1.0)).collect();
            let composed = compose_external_priors_with_alphas(
                external.sources,
                &requested,
                &scaled,
                &baseline,
            )
            .map_err(|e| {
                ValidationError::estimation_msg(format!("prior sensitivity compose failed: {e}"))
            })?;
            let est = BayesianGComputationAte {
                n_draws: estimator.n_draws.min(200),
                prior: Some(composed.prior),
                ..estimator.clone()
            };
            let post = est.fit(problem, identification, workspace, ctx).map_err(|e| {
                ValidationError::estimation_msg(format!("prior sensitivity α fit failed: {e}"))
            })?;
            let eq = post.effect_column().ok_or_else(|| {
                ValidationError::estimation_msg("missing effect column in α sensitivity fit")
            })?;
            means.push(post.summaries.mean[eq]);
            sds.push(post.summaries.sd[eq]);
            posts.push(post);
        }
        Ok((
            PriorSensitivitySummary {
                family: PriorSensitivityFamily::ExternalAlpha,
                prior_scales: Arc::from([]),
                alphas: Arc::clone(&self.alphas),
                variance_multipliers: Arc::from([]),
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
        let denom =
            summary.effect_means.iter().copied().map(f64::abs).fold(original_ate.abs(), f64::max);
        let informative = !summary.effect_means.is_empty()
            && original_ate.is_finite()
            && summary.effect_means.iter().all(|mean| mean.is_finite());
        let relative = if !informative {
            f64::NAN
        } else if denom == 0.0 {
            0.0
        } else {
            // Divide before subtraction to avoid overflow for opposite signs.
            max / denom - min / denom
        };
        let passed = relative.is_finite() && relative <= self.max_relative_range;
        let kind = match summary.family {
            PriorSensitivityFamily::IsotropicScale => "prior_sensitivity",
            PriorSensitivityFamily::ExternalAlpha => "prior_sensitivity_alpha",
            PriorSensitivityFamily::ResolvedPriorVariance => "prior_sensitivity_resolved_prior",
        };
        RefutationReport {
            refuter: Arc::from(kind),
            original_ate,
            refuted_ate: summary.effect_means.last().copied().unwrap_or(original_ate),
            comparison: relative,
            informative,
            passed,
            failure_condition: if passed {
                None
            } else {
                Some(Arc::from(format!(
                    "prior sensitivity relative range {relative} exceeds max {}",
                    self.max_relative_range
                )))
            },
            replicates: u32::try_from(self.grid_len()).unwrap_or(u32::MAX),
        }
    }
}

/// Copy of `prior` with every Gaussian coefficient variance multiplied by `mult`.
fn scale_coefficient_variances(prior: &PriorSet, mult: f64) -> Result<PriorSet, ValidationError> {
    let mut scaled = prior.clone();
    let mut found = false;
    for spec in &mut scaled.specs {
        if let PriorSpec::GaussianCoefficients(coef) = spec {
            coef.variance = coef.variance.iter().map(|v| v * mult).collect();
            found = true;
        }
    }
    if !found {
        return Err(ValidationError::estimation_msg(
            "resolved prior has no Gaussian coefficient block to perturb",
        ));
    }
    Ok(scaled)
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
    // (1 + count) / (1 + n) form (Davison & Hinkley): an exact-zero Monte Carlo
    // p-value is never valid evidence with a finite sample, so both tails are
    // bounded below by 1/(n+1) and the two-sided p-value by 2/(n+1).
    let below = summaries.iter().filter(|&&x| x <= observed).count() as f64;
    let above = summaries.iter().filter(|&&x| x >= observed).count() as f64;
    let p_lower = (1.0 + below) / (1.0 + n);
    let p_upper = (1.0 + above) / (1.0 + n);
    let p = (2.0 * p_lower.min(p_upper)).min(1.0);
    PredictiveCheckReport {
        kind,
        observed,
        predictive_mean: mean,
        predictive_sd: sd,
        p_value: p,
        location_tails: [p_lower, p_upper],
        // Populated by [`summarize_predictive_check`], the two-axis wrapper around this
        // function; callers of `summarize_check` directly (unit tests exercising the
        // Monte Carlo p-value formula) only care about the location axis above.
        observed_dispersion: 0.0,
        predictive_dispersion_mean: 0.0,
        dispersion_p_value: 1.0,
        dispersion_tails: [1.0, 1.0],
        n_sims,
        serial: None,
        noise: PredictiveNoise::Likelihood,
    }
}

/// Push the per-simulation location (mean) and dispersion (cross-observation SD)
/// summary statistics for one replicated outcome vector `y_pred`.
///
/// Dispersion is the sample SD of `y_pred` *across observations within a single
/// simulation* — how spread the replicated outcomes are over the design — which is
/// exactly the axis a mean-only PPC statistic is blind to (see D3 / defect C:
/// a model with an unbiased predictive mean but a badly wrong predictive spread
/// must be caught here, not just on the mean axis).
fn push_mean_and_dispersion(
    y_pred: &[f64],
    policy: KernelPolicy,
    mean_summaries: &mut Vec<f64>,
    disp_summaries: &mut Vec<f64>,
) {
    let n = y_pred.len().max(1) as f64;
    let mean_y = y_pred.iter().sum::<f64>() / n;
    let sd_y = reduce_posterior_draws(y_pred, PosteriorReduceOp::Std, &policy).unwrap_or(0.0);
    mean_summaries.push(mean_y);
    disp_summaries.push(sd_y);
}

/// Two-axis predictive check: combines a location (mean) Monte Carlo check with a
/// dispersion (cross-observation SD) Monte Carlo check via [`summarize_check`], and
/// merges both into a single [`PredictiveCheckReport`].
fn summarize_predictive_check(
    kind: PredictiveCheckKind,
    outcome: &[f64],
    policy: KernelPolicy,
    mean_summaries: &[f64],
    disp_summaries: &[f64],
    n_sims: u32,
    noise: PredictiveNoise,
) -> PredictiveCheckReport {
    let n = outcome.len().max(1) as f64;
    let observed_mean = outcome.iter().sum::<f64>() / n;
    let observed_dispersion =
        reduce_posterior_draws(outcome, PosteriorReduceOp::Std, &policy).unwrap_or(0.0);
    let mean_report = summarize_check(kind, observed_mean, mean_summaries, n_sims);
    let disp_report = summarize_check(kind, observed_dispersion, disp_summaries, n_sims);
    let (location_tails, dispersion_tails) = match noise {
        PredictiveNoise::Likelihood => (mean_report.location_tails, disp_report.location_tails),
        // Unbounded noise reaches any larger spread.
        PredictiveNoise::MeanOnlyImproperScale => {
            (mean_report.location_tails, [disp_report.location_tails[0], 1.0])
        }
    };
    let two_sided = |tails: [f64; 2]| (2.0 * tails[0].min(tails[1])).min(1.0);
    PredictiveCheckReport {
        kind,
        observed: mean_report.observed,
        predictive_mean: mean_report.predictive_mean,
        predictive_sd: mean_report.predictive_sd,
        p_value: two_sided(location_tails),
        location_tails,
        observed_dispersion,
        predictive_dispersion_mean: disp_report.predictive_mean,
        dispersion_p_value: two_sided(dispersion_tails),
        dispersion_tails,
        n_sims,
        serial: None,
        noise,
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
    #[test]
    fn review_prior_sensitivity_verdict_is_unit_invariant() {
        let sensitivity =
            PriorSensitivity { max_relative_range: 0.25, ..PriorSensitivity::default() };
        for scale in [1e-100, 1.0, 1e100] {
            let summary = PriorSensitivitySummary {
                prior_scales: Arc::from([1.0, 10.0]),
                alphas: Arc::from([]),
                effect_means: Arc::from([scale, 2.0 * scale]),
                effect_sds: Arc::from([scale, scale]),
                ..PriorSensitivitySummary::default()
            };
            let report = sensitivity.to_report(&summary, scale);
            assert!((report.comparison - 0.5).abs() < 1e-12);
            assert!(!report.passed);
        }
    }

    use super::*;
    use antecedent_core::{
        AverageEffectQuery, CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet,
        ValueType, VariableId,
    };
    use antecedent_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
    };
    use antecedent_estimate::{BayesianBackendKind, BayesianGComputationAte};
    use antecedent_expr::{ExprId, IdentifiedEstimand};
    use antecedent_identify::IdentificationStatus;
    use antecedent_prob::{ExternalPriorWeight, GaussianCoefficientPrior, PriorSpec};

    #[test]
    fn review_predictive_mixture_retains_tail_direction_and_small_variance() {
        let a = summarize_check(PredictiveCheckKind::Posterior, 0.0, &[-2.0; 99], 99);
        let b = summarize_check(PredictiveCheckKind::Posterior, 0.0, &[2.0; 99], 99);
        let mixed = PredictiveCheckReport::mixture_weighted(&[(0.5, &a), (0.5, &b)]).unwrap();
        assert!((mixed.p_value - 1.0).abs() < 1e-12);
        for weight in [1e-200, 1.0, 1e300] {
            let a = report(PredictiveCheckKind::Posterior, 1e12, 1.0);
            let b = report(PredictiveCheckKind::Posterior, 1e12 + 4.0, 1.0);
            let mixed =
                PredictiveCheckReport::mixture_weighted(&[(weight, &a), (weight, &b)]).unwrap();
            assert!((mixed.predictive_sd - 5.0f64.sqrt()).abs() < 1e-12);
        }
    }

    fn report(kind: PredictiveCheckKind, mean: f64, sd: f64) -> PredictiveCheckReport {
        PredictiveCheckReport {
            kind,
            observed: mean,
            predictive_mean: mean,
            predictive_sd: sd,
            p_value: 0.4,
            location_tails: [0.2, 0.8],
            observed_dispersion: 1.0,
            predictive_dispersion_mean: 1.0,
            dispersion_p_value: 0.4,
            dispersion_tails: [0.2, 0.8],
            n_sims: 200,
            serial: None,
            noise: PredictiveNoise::Likelihood,
        }
    }

    #[test]
    fn mixture_predictive_sd_includes_between_atom_spread() {
        // Two completions that disagree on location but are individually tight.
        // A weighted mean of the SDs would report 1.0 and hide the disagreement.
        let a = report(PredictiveCheckKind::Posterior, 0.0, 1.0);
        let b = report(PredictiveCheckKind::Posterior, 4.0, 1.0);
        let mixed = PredictiveCheckReport::mixture_weighted(&[(0.5, &a), (0.5, &b)]).unwrap();
        assert!((mixed.predictive_mean - 2.0).abs() < 1e-12, "mean={}", mixed.predictive_mean);
        // sqrt(1 + 2^2) = sqrt(5): within-atom variance plus between-atom variance.
        let expected = 5.0f64.sqrt();
        assert!(
            (mixed.predictive_sd - expected).abs() < 1e-12,
            "mixture sd must use total variance, got {} expected {expected}",
            mixed.predictive_sd
        );
        assert_eq!(mixed.n_sims, 200);
    }

    #[test]
    fn mixture_weighted_ignores_zero_weight_and_mismatched_kinds() {
        // A zero-weight leading atom must not decide the mixture's kind.
        let ignored = report(PredictiveCheckKind::Prior, 100.0, 50.0);
        let a = report(PredictiveCheckKind::Posterior, 1.0, 1.0);
        let other_kind = report(PredictiveCheckKind::Prior, 100.0, 50.0);
        let mixed = PredictiveCheckReport::mixture_weighted(&[
            (0.0, &ignored),
            (1.0, &a),
            (1.0, &other_kind),
        ])
        .unwrap();
        assert_eq!(mixed.kind, PredictiveCheckKind::Posterior);
        assert!((mixed.predictive_mean - 1.0).abs() < 1e-12, "mean={}", mixed.predictive_mean);
        assert!((mixed.predictive_sd - 1.0).abs() < 1e-12, "sd={}", mixed.predictive_sd);
    }

    #[test]
    fn mixture_weighted_is_none_without_positive_weight() {
        let a = report(PredictiveCheckKind::Posterior, 1.0, 1.0);
        assert!(PredictiveCheckReport::mixture_weighted(&[]).is_none());
        assert!(PredictiveCheckReport::mixture_weighted(&[(0.0, &a)]).is_none());
    }

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
    fn summarize_check_observed_outside_range_never_reports_zero() {
        // Simulated draws are tightly clustered; an observation far outside the
        // range on either side must not collapse the Monte Carlo p-value to
        // exactly 0 (D1: below/n or above/n hitting 0 or 1 exactly).
        let n = 200usize;
        let summaries: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect(); // [0, 1)
        let min_p = 2.0 / (n as f64 + 1.0);

        let low = summarize_check(PredictiveCheckKind::Posterior, -10.0, &summaries, n as u32);
        assert!(low.p_value > 0.0, "p_value must be strictly positive, got {}", low.p_value);
        assert!(low.p_value >= min_p, "p_value {} below the 2/(n+1) floor {min_p}", low.p_value);

        let high = summarize_check(PredictiveCheckKind::Posterior, 10.0, &summaries, n as u32);
        assert!(high.p_value > 0.0, "p_value must be strictly positive, got {}", high.p_value);
        assert!(high.p_value >= min_p, "p_value {} below the 2/(n+1) floor {min_p}", high.p_value);
    }

    #[test]
    fn summarize_check_observed_near_centre_gives_high_p_value() {
        // Sanity check that the corrected formula still behaves as expected in
        // the ordinary case: an observation near the middle of the simulated
        // distribution should give a p-value near 1, not just "not exactly 0".
        let n = 200usize;
        let summaries: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect(); // [0, 1)
        let centre = summarize_check(PredictiveCheckKind::Posterior, 0.5, &summaries, n as u32);
        assert!(centre.p_value > 0.9, "expected p_value near 1, got {}", centre.p_value);
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
            ..PriorSensitivity::standard_grid()
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
        assert!(summary.alphas.is_empty());
        assert_eq!(posts.len(), 3);
        let rep =
            sens.to_report(&summary, posts[0].summaries.mean[posts[0].effect_column().unwrap()]);
        assert!(rep.passed);
    }

    #[test]
    fn prior_sensitivity_external_alpha_pulls_toward_source() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../conformance/validate/bayesian_checks/expected.json"
        ))
        .unwrap();
        assert!(
            fixture["contracts"]["prior_sensitivity_full_trust_moves_toward_source"]
                .as_bool()
                .unwrap()
        );

        let (data, estimand, query) = toy();
        // Data ATE ≈ 2; bank a tight prior with treatment coef mean = 8.
        let bayes = BayesianGComputationAte {
            backend: BayesianBackendKind::ConjugateGaussian,
            n_draws: 120,
            seed: 7,
            ..BayesianGComputationAte::new()
        };
        let prep = bayes.prepare(&data, &estimand, &query).unwrap();
        let n = prep.design.ncols;
        let t_col = prep.design.treatment_column().expect("treatment column");
        let mut mean = vec![0.0; n];
        mean[t_col] = 8.0;
        let mut source_prior = PriorSet::new();
        source_prior.push(PriorSpec::GaussianCoefficients(GaussianCoefficientPrior {
            mean: Arc::from(mean),
            variance: Arc::from(vec![0.05; n]),
        }));
        let sources = [ExternalPriorSource {
            id: Arc::from("survey_a"),
            prior: source_prior,
            weight: ExternalPriorWeight::power(1.0).unwrap(),
            ess: None,
        }];
        let alphas_applied = [1.0_f64];
        let mut ws = BayesianGCompWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let sens = PriorSensitivity::standard_alpha_grid();
        let (summary, _) = sens
            .evaluate_external_alpha(
                &bayes,
                &prep,
                IdentificationStatus::NonparametricallyIdentified,
                &mut ws,
                &ctx,
                ExternalAlphaSensitivity { sources: &sources, alphas_applied: &alphas_applied },
            )
            .unwrap();
        assert_eq!(summary.alphas.len(), 5);
        assert!(summary.prior_scales.is_empty());
        assert!(summary.effect_means.iter().all(|m| m.is_finite()));
        let m0 = summary.effect_means[0];
        let m1 = *summary.effect_means.last().unwrap();
        // Full trust (m=1) should sit closer to the banked treatment mean than baseline (m=0).
        assert!(
            (m1 - 8.0).abs() < (m0 - 8.0).abs(),
            "m=1 mean {m1} should be closer to 8 than m=0 mean {m0}"
        );
        let rep = sens.to_report(&summary, m1);
        assert_eq!(rep.refuter.as_ref(), "prior_sensitivity_alpha");
        assert!(rep.informative);
        assert!(rep.comparison.is_finite() && rep.comparison > 0.0);
    }

    #[test]
    fn ppc_catches_variance_misspecification_mean_ok() {
        // Defect C regression: construct data whose outcome mean is exactly zero
        // but has large residual spread (SD ~5), so a model whose replicates have
        // a small cross-observation SD (tight coefficients, residual variance 1)
        // gets the *mean* right (mean-only PPC would pass) while badly
        // understating the *dispersion* of the outcome. The dispersion axis must
        // catch this even though the location axis does not.
        let n = 400usize;
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
        // t, z are both (near-)constant so the fitted/prior mean function has almost
        // no cross-observation spread; y has a large, fixed residual (+/- 5) on top of
        // a tiny signal, so E[y] is well predicted but SD(y) is not.
        let mut rng = CausalRng::from_seed(99);
        let t: Vec<f64> = (0..n).map(|_| 0.0).collect();
        let z: Vec<f64> = (0..n).map(|_| 0.0).collect();
        let mut y: Vec<f64> = (0..n).map(|_| 5.0 * standard_normal(&mut rng)).collect();
        let centre = y.iter().sum::<f64>() / n as f64;
        y.iter_mut().for_each(|v| *v -= centre);
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
        let data = TabularData::new(storage);

        let bayes = BayesianGComputationAte {
            backend: BayesianBackendKind::ConjugateGaussian,
            n_draws: 300,
            seed: 5,
            // Tight prior around 0: the mean function is confidently ~0 everywhere,
            // matching E[y]~0, but says nothing spreads out — the mean axis passes.
            prior_scale: 0.05,
            ..BayesianGComputationAte::new()
        };
        let prep = bayes.prepare(&data, &estimand, &query).unwrap();
        let ctx = ExecutionContext::for_tests(1);

        // The prior in force plus a residual variance of 1: replicates spread by ~1.
        let mut prior = bayes.prior_in_force(prep.design.ncols);
        prior.push(PriorSpec::KnownResidualVariance(1.0));
        let rep = PriorPredictiveCheck { n_sims: 300, seed: 6, ..PriorPredictiveCheck::new() }
            .check_with_prior(&prep, &prior, &ctx)
            .unwrap();
        assert_eq!(rep.noise, PredictiveNoise::Likelihood);

        assert!(
            rep.p_value >= 0.05,
            "expected the mean axis to look fine (unbiased predictive mean), got p={}",
            rep.p_value
        );
        assert!(
            rep.dispersion_p_value < 0.05,
            "expected the dispersion axis to catch the 5x variance mismatch, got \
             dispersion_p_value={}",
            rep.dispersion_p_value
        );
        let refuted = rep.to_refutation_report(0.0, 0.05);
        assert!(
            !refuted.passed,
            "predictive check must fail overall when dispersion is badly wrong even \
             though the mean matches"
        );
    }

    #[test]
    fn sbc_to_report_gates_on_uniformity_not_just_mean() {
        // Defect B regression: a U-shaped rank histogram (ranks piled at the two
        // extremes, overdispersed posterior) is symmetric about the middle bin so its
        // mean rank fraction lands squarely in [0.35, 0.65] — a mean-only gate would
        // pass it. The χ² uniformity statistic must catch it.
        let n_reps = 200u32;
        let n_draws = 100usize;
        // Every replicate's rank is pinned at one of the two extremes (bins 0 and 9),
        // alternating — a textbook U shape.
        let ranks: Vec<u32> =
            (0..n_reps).map(|i| if i % 2 == 0 { 0 } else { n_draws as u32 }).collect();
        let n_d = n_draws as f64;
        let fracs: Vec<f64> = ranks.iter().map(|&r| f64::from(r) / n_d).collect();
        let mean_rank_frac = fracs.iter().sum::<f64>() / fracs.len() as f64;
        assert!(
            (0.35..=0.65).contains(&mean_rank_frac),
            "fixture sanity: mean rank frac {mean_rank_frac} should look fine on its own"
        );
        let bins = 10usize;
        let mut counts = vec![0.0; bins];
        for &r in &ranks {
            let b = ((u64::from(r) * bins as u64) / (n_draws as u64)).min(bins as u64 - 1) as usize;
            counts[b] += 1.0;
        }
        let expected = f64::from(n_reps) / bins as f64;
        let mut chi2 = 0.0;
        for c in counts {
            let d = c - expected;
            chi2 += d * d / expected.max(1.0);
        }
        assert!(
            chi2 > SBC_CHI2_CRITICAL_9DF_P99,
            "fixture sanity: U-shaped ranks should trip the χ² statistic, got {chi2}"
        );
        let report = SbcReport { ranks: Arc::from(ranks), mean_rank_frac, uniformity_stat: chi2 };
        let sbc = SimulationBasedCalibration { n_reps, n_draws, seed: 0 };
        let rep = sbc.to_report(&report, 2.0);
        assert!(
            !rep.passed,
            "SBC must fail on a U-shaped (non-uniform) rank distribution even though \
             mean_rank_frac={mean_rank_frac:.3} is in [0.35, 0.65]"
        );
    }

    /// Calibration-gate tests for defect A: [`SimulationBasedCalibration`] and
    /// [`PosteriorCalibrationOnSyntheticScm`] are fully implemented but (prior to this
    /// change) had zero call sites outside the `lib.rs` re-export — the Bayesian
    /// posterior had no coverage check at all. `#[ignore]`d for the same reason as
    /// `antecedent-estimate/src/calibration_coverage.rs`: many refits per test, too
    /// slow for every `cargo test`. Run via `cargo test -p antecedent-validate
    /// <name> -- --ignored --nocapture`.
    mod calibration_gate {
        use super::*;

        fn coverage_band(n_reps: u32, level: f64) -> (f64, f64) {
            let se = (level * (1.0 - level) / f64::from(n_reps)).sqrt();
            let lo = (level - 4.0 * se).max(0.5);
            let hi = (level + 4.0 * se).min(1.0);
            (lo, hi)
        }

        #[test]
        #[ignore = "calibration: run via scripts/gate_calibration.sh"]
        fn sbc_conjugate_gaussian_ranks_are_uniform() {
            let (data, estimand, query) = toy();
            let bayes = BayesianGComputationAte {
                backend: BayesianBackendKind::ConjugateGaussian,
                n_draws: 300,
                seed: 11,
                prior_scale: 5.0,
                ..BayesianGComputationAte::new()
            };
            let prep = bayes.prepare(&data, &estimand, &query).unwrap();
            let mut ws = BayesianGCompWorkspace::default();
            let ctx = ExecutionContext::for_tests(1);
            let sbc = SimulationBasedCalibration { n_reps: 200, n_draws: 300, seed: 42 };
            let report = sbc
                .check(
                    &bayes,
                    &prep,
                    IdentificationStatus::NonparametricallyIdentified,
                    &mut ws,
                    &ctx,
                )
                .unwrap();
            let rep = sbc.to_report(&report, 2.0);
            assert!(
                rep.passed,
                "SBC should pass for a correctly specified conjugate Gaussian model: \
                 mean_rank_frac={:.3} chi2={:.3}",
                report.mean_rank_frac, report.uniformity_stat
            );
            assert!(
                (0.35..=0.65).contains(&report.mean_rank_frac),
                "mean_rank_frac={:.3} outside [0.35, 0.65]",
                report.mean_rank_frac
            );
            assert!(
                report.uniformity_stat < SBC_CHI2_CRITICAL_9DF_P99,
                "chi2={:.3} exceeds critical value {SBC_CHI2_CRITICAL_9DF_P99:.3}",
                report.uniformity_stat
            );
        }

        #[test]
        #[ignore = "calibration: run via scripts/gate_calibration.sh"]
        fn posterior_calibration_synthetic_scm_nominal_90_coverage() {
            let (data, estimand, query) = toy();
            let bayes = BayesianGComputationAte {
                backend: BayesianBackendKind::ConjugateGaussian,
                n_draws: 300,
                seed: 21,
                prior_scale: 5.0,
                ..BayesianGComputationAte::new()
            };
            let prep = bayes.prepare(&data, &estimand, &query).unwrap();
            let mut ws = BayesianGCompWorkspace::default();
            let ctx = ExecutionContext::for_tests(1);
            let calib = PosteriorCalibrationOnSyntheticScm {
                n_reps: 200,
                n_draws: 300,
                level: 0.9,
                seed: 77,
            };
            let report = calib
                .check(
                    &bayes,
                    &prep,
                    IdentificationStatus::NonparametricallyIdentified,
                    &mut ws,
                    &ctx,
                )
                .unwrap();
            let (lo, hi) = coverage_band(report.n_reps, 0.9);
            assert!(
                report.coverage >= lo && report.coverage <= hi,
                "nominal 90% credible-interval coverage={:.3} outside [{:.3}, {:.3}] \
                 ({} reps); mean_abs_error={:.3}",
                report.coverage,
                lo,
                hi,
                report.n_reps,
                report.mean_abs_error
            );
        }
    }

    /// Null calibration of the predictive checks: on correctly specified models the
    /// two-axis check fails at no more than its nominal rate.
    mod predictive_null_calibration {
        use super::*;
        use antecedent_estimate::OverlapPolicy;
        use antecedent_prob::{BayesLikelihood, InvGammaPrior};
        use antecedent_stats::CompiledDesign;

        const DATASETS: u64 = 200;
        const ALPHA: f64 = 0.05;

        fn problem(t: &[f64], x: &[f64], y: &[f64]) -> PreparedBayesianProblem {
            PreparedBayesianProblem {
                design: CompiledDesign::linear_adjustment(
                    t,
                    &[(VariableId::from_raw(2), x)],
                    y,
                    &[],
                )
                .unwrap(),
                method: Arc::from("backdoor.adjustment"),
                adjustment_set: Arc::from([VariableId::from_raw(2)]),
                active: 1.0,
                control: 0.0,
                overlap: OverlapPolicy::ExplicitOverride,
                coef_names: None,
                unit_ids: None,
                serial_dependence: SerialDependence::Iid,
            }
        }

        /// `t ~ Bernoulli(0.5)`, `x ~ N(0, 1)` and the linear predictor
        /// `1 + 0.5 t + 0.8 x` (times `scale`).
        fn covariates(n: usize, rng: &mut CausalRng) -> (Vec<f64>, Vec<f64>) {
            let t = (0..n).map(|_| if rng.next_f64() < 0.5 { 1.0 } else { 0.0 }).collect();
            let x = (0..n).map(|_| standard_normal(rng)).collect();
            (t, x)
        }

        /// Failure rate of the posterior predictive check over correctly specified
        /// datasets drawn by `outcome` and fitted by `estimator`.
        fn posterior_failure_rate(
            estimator: &BayesianGComputationAte,
            outcome: impl Fn(f64, f64, &mut CausalRng) -> f64,
        ) -> f64 {
            let mut failures = 0u32;
            for rep in 0..DATASETS {
                let mut rng = CausalRng::from_seed(0xCA11_B000 + rep);
                let (t, x) = covariates(150, &mut rng);
                let y: Vec<f64> =
                    t.iter().zip(&x).map(|(t, x)| outcome(*t, *x, &mut rng)).collect();
                let prep = problem(&t, &x, &y);
                let ctx = ExecutionContext::for_tests(rep);
                let post = estimator
                    .fit(
                        &prep,
                        IdentificationStatus::NonparametricallyIdentified,
                        &mut BayesianGCompWorkspace::default(),
                        &ctx,
                    )
                    .unwrap();
                let report = PosteriorPredictiveCheck {
                    n_sims: 100,
                    ..PosteriorPredictiveCheck::for_estimator(estimator, &ctx)
                }
                .check(&prep, &post)
                .unwrap();
                assert_eq!(report.noise, PredictiveNoise::Likelihood);
                failures += u32::from(!report.to_refutation_report(0.0, ALPHA).passed);
            }
            let rate = f64::from(failures) / DATASETS as f64;
            eprintln!("posterior predictive null failure rate {rate:.3} over {DATASETS} datasets");
            rate
        }

        #[test]
        fn posterior_predictive_gaussian_null_rate_is_at_most_nominal() {
            let estimator = BayesianGComputationAte {
                backend: BayesianBackendKind::ConjugateGaussian,
                n_draws: 100,
                seed: 3,
                ..BayesianGComputationAte::new()
            };
            let rate = posterior_failure_rate(&estimator, |t, x, rng| {
                1.0 + 0.5 * t + 0.8 * x + standard_normal(rng)
            });
            assert!(rate <= 2.0 * ALPHA, "correct Gaussian model failed the PPC at rate {rate}");
        }

        #[test]
        fn posterior_predictive_logit_null_rate_is_at_most_nominal() {
            let estimator = BayesianGComputationAte {
                backend: BayesianBackendKind::Laplace,
                n_draws: 100,
                seed: 4,
                ..BayesianGComputationAte::new()
            }
            .with_likelihood(BayesLikelihood::BernoulliLogit);
            assert_eq!(estimator.glm_family(), GlmFamily::BinomialLogit);
            let rate = posterior_failure_rate(&estimator, |t, x, rng| {
                let p = 1.0 / (1.0 + (-(-0.3 + 0.5 * t + 0.8 * x)).exp());
                if rng.next_f64() < p { 1.0 } else { 0.0 }
            });
            assert!(rate <= 2.0 * ALPHA, "correct logit model failed the PPC at rate {rate}");
        }

        /// Data drawn from a proper prior are a correctly specified prior predictive:
        /// each axis's p-value is (discretely) uniform, so each fails at ≈ α and the
        /// two-axis check at no more than ≈ 1 − (1 − α)².
        #[test]
        fn prior_predictive_proper_prior_null_rate_is_nominal() {
            let n = 100;
            let mut prior = PriorSet {
                specs: vec![PriorSpec::GaussianCoefficients(GaussianCoefficientPrior::isotropic(
                    3, 1.0,
                ))],
                contrast: None,
                categorical: Vec::new(),
                restrictions: Vec::new(),
            };
            let residual = InvGammaPrior { shape: 3.0, scale: 2.0 };
            prior.push(PriorSpec::ResidualInvGamma(residual));
            let (mut location, mut dispersion, mut either) = (0u32, 0u32, 0u32);
            for rep in 0..DATASETS {
                let mut rng = CausalRng::from_seed(0x9A10_0000 + rep);
                let (t, x) = covariates(n, &mut rng);
                let beta: Vec<f64> = (0..3).map(|_| standard_normal(&mut rng)).collect();
                let sigma = sample_inv_gamma(residual.shape, residual.scale, &mut rng).sqrt();
                let y: Vec<f64> = t
                    .iter()
                    .zip(&x)
                    .map(|(t, x)| {
                        beta[0] + beta[1] * t + beta[2] * x + sigma * standard_normal(&mut rng)
                    })
                    .collect();
                let report =
                    PriorPredictiveCheck { n_sims: 200, seed: rep, ..PriorPredictiveCheck::new() }
                        .check_with_prior(
                            &problem(&t, &x, &y),
                            &prior,
                            &ExecutionContext::for_tests(rep),
                        )
                        .unwrap();
                assert_eq!(report.noise, PredictiveNoise::Likelihood);
                location += u32::from(report.p_value < ALPHA);
                dispersion += u32::from(report.dispersion_p_value < ALPHA);
                either += u32::from(!report.to_refutation_report(0.0, ALPHA).passed);
            }
            let rate = |count: u32| f64::from(count) / DATASETS as f64;
            eprintln!(
                "prior predictive null failure rates: location {:.3}, dispersion {:.3}, either {:.3}",
                rate(location),
                rate(dispersion),
                rate(either)
            );
            // Nominal α = 0.05 per axis; 3 Monte Carlo SEs at 200 datasets ≈ 0.046.
            assert!(rate(location) <= 0.1, "location axis null rate {}", rate(location));
            assert!(rate(dispersion) <= 0.1, "dispersion axis null rate {}", rate(dispersion));
            assert!(rate(either) <= 0.16, "two-axis null rate {}", rate(either));
        }

        /// Under the weakly informative residual prior (no finite mean) the prior
        /// predictive scores only the direction the unbounded noise cannot reach.
        #[test]
        fn prior_predictive_improper_residual_prior_is_one_sided_on_dispersion() {
            let mut rng = CausalRng::from_seed(17);
            let (t, x) = covariates(120, &mut rng);
            // Observed spread far above the (tight) mean-only replicates.
            let y: Vec<f64> = t.iter().map(|_| 20.0 * standard_normal(&mut rng)).collect();
            let prep = problem(&t, &x, &y);
            let ctx = ExecutionContext::for_tests(1);
            let tight = PriorSet {
                specs: vec![
                    PriorSpec::GaussianCoefficients(GaussianCoefficientPrior::isotropic(3, 0.01)),
                    PriorSpec::ResidualInvGamma(InvGammaPrior::weakly_informative()),
                ],
                contrast: None,
                categorical: Vec::new(),
                restrictions: Vec::new(),
            };
            let report = PriorPredictiveCheck::new().check_with_prior(&prep, &tight, &ctx).unwrap();
            assert_eq!(report.noise, PredictiveNoise::MeanOnlyImproperScale);
            assert!((report.dispersion_tails[1] - 1.0).abs() < f64::EPSILON);
            assert!(report.dispersion_p_value > 0.5, "{report:?}");
            // A diffuse coefficient prior spreads even the means wider than the data.
            let calm: Vec<f64> = t.iter().map(|_| 0.01 * standard_normal(&mut rng)).collect();
            let diffuse = PriorPredictiveCheck::new().check(&problem(&t, &x, &calm), &ctx).unwrap();
            assert!(diffuse.dispersion_p_value < ALPHA, "{diffuse:?}");
            let verdict = diffuse.to_refutation_report(0.0, ALPHA);
            assert!(
                verdict.failure_condition.as_deref().is_some_and(|m| m.contains("diffuse")),
                "{verdict:?}"
            );
        }

        /// The serial axis on a tempered posterior says the interval is tempered.
        #[test]
        fn serial_failure_on_a_tempered_posterior_is_reported_as_informative() {
            let mut report = report(PredictiveCheckKind::Posterior, 0.0, 1.0);
            report.serial = Some(SerialDiscrepancy {
                lag: 1,
                observed: 0.6,
                predictive_mean: 0.0,
                p_value: 0.01,
                tails: [0.995, 0.005],
                tempering_kappa: Some(3.2),
            });
            let verdict = report.to_refutation_report(0.0, ALPHA);
            assert!(!verdict.passed);
            let message = verdict.failure_condition.unwrap();
            assert!(
                message.contains("already tempered") && message.contains("kappa=3.2"),
                "{message}"
            );
            let untempered = PredictiveCheckReport::mixture_weighted(&[
                (1.0, &report),
                (1.0, &{
                    let mut other = report.clone();
                    other.serial =
                        other.serial.map(|s| SerialDiscrepancy { tempering_kappa: None, ..s });
                    other
                }),
            ])
            .unwrap();
            assert_eq!(untempered.serial.unwrap().tempering_kappa, None);
        }
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
    /// Returns `None` when the posterior is not MCMC (caller should emit `NotApplicable`).
    #[must_use]
    pub fn check(&self, posterior: &CausalPosterior) -> Option<RefutationReport> {
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

/// Simulation-based calibration ranks for a scalar posterior functional.
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
    /// Rank of the prior draw in each replicate (`0..=n_draws`).
    pub ranks: Arc<[u32]>,
    /// Mean rank / `n_draws` (≈ 0.5 when calibrated).
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
            let post = est
                .fit(&sim_problem, identification, workspace, ctx)
                .map_err(|e| ValidationError::estimation_msg(format!("SBC refit failed: {e}")))?;
            let col = post
                .effect_column()
                .ok_or_else(|| ValidationError::estimation_msg("SBC: no effect column"))?;
            let draws = post
                .draws
                .column(col)
                .map_err(|e| ValidationError::estimation_msg(format!("SBC draws: {e}")))?;
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
            reduce_posterior_draws(&fracs, PosteriorReduceOp::Mean, &ctx.kernel_policy)
                .unwrap_or(0.5);
        let bins = 10usize;
        let mut counts = vec![0.0; bins];
        let n_draws_u = u64::try_from(self.n_draws.max(1)).unwrap_or(1);
        let bins_u = u64::try_from(bins).unwrap_or(1);
        for &r in &ranks {
            let b = usize::try_from(u64::from(r) * bins_u / n_draws_u).unwrap_or(0).min(bins - 1);
            counts[b] += 1.0;
        }
        let expected = f64::from(self.n_reps) / bins as f64;
        let mut chi2 = 0.0;
        for c in counts {
            let d = c - expected;
            chi2 += d * d / expected.max(1.0);
        }
        Ok(SbcReport { ranks: Arc::from(ranks), mean_rank_frac, uniformity_stat: chi2 })
    }

    /// Convert to a refutation report.
    ///
    /// Passes only when **both** hold:
    /// - the mean rank fraction is in `[0.35, 0.65]` (catches gross location bias), and
    /// - the χ² uniformity statistic over the 10 rank bins is below
    ///   `SBC_CHI2_CRITICAL_9DF_P99` (catches symmetric-about-0.5 U/M-shaped rank
    ///   distributions — overdispersed / underdispersed posteriors — that a mean-only
    ///   band cannot see because they average out to ≈0.5).
    #[must_use]
    pub fn to_report(&self, report: &SbcReport, original_ate: f64) -> RefutationReport {
        let mean_ok = (0.35..=0.65).contains(&report.mean_rank_frac);
        let uniform_ok = report.uniformity_stat.is_finite()
            && report.uniformity_stat <= SBC_CHI2_CRITICAL_9DF_P99;
        let passed = mean_ok && uniform_ok;
        RefutationReport {
            refuter: Arc::from("sbc"),
            original_ate,
            refuted_ate: report.mean_rank_frac,
            comparison: report.uniformity_stat,
            informative: true,
            passed,
            failure_condition: if passed {
                None
            } else if !mean_ok && !uniform_ok {
                Some(Arc::from(format!(
                    "SBC mean rank frac {:.3} outside [0.35, 0.65] and χ²={:.3} exceeds critical \
                     value {SBC_CHI2_CRITICAL_9DF_P99:.3} (9 df, p=0.99)",
                    report.mean_rank_frac, report.uniformity_stat
                )))
            } else if !mean_ok {
                Some(Arc::from(format!(
                    "SBC mean rank frac {:.3} outside [0.35, 0.65]",
                    report.mean_rank_frac
                )))
            } else {
                Some(Arc::from(format!(
                    "SBC rank distribution non-uniform: χ²={:.3} exceeds critical value \
                     {SBC_CHI2_CRITICAL_9DF_P99:.3} (9 df, p=0.99); mean rank frac {:.3} looked \
                     fine but ranks are not uniformly distributed (U- or M-shaped)",
                    report.uniformity_stat, report.mean_rank_frac
                )))
            },
            replicates: self.n_reps,
        }
    }
}

/// χ² critical value at the 0.99 quantile with 9 degrees of freedom (10 rank bins − 1).
///
/// Used to gate [`SimulationBasedCalibration::to_report`]'s uniformity check: SBC ranks
/// should be uniform under a well-calibrated posterior, and this is the standard
/// one-sided χ² goodness-of-fit critical value (e.g. Talts et al. 2018 §3 use the
/// same chi-square uniformity diagnostic).
const SBC_CHI2_CRITICAL_9DF_P99: f64 = 21.666;

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
            let post = est
                .fit(&sim, identification, workspace, ctx)
                .map_err(|e| ValidationError::estimation_msg(format!("calibration refit: {e}")))?;
            let col = post
                .effect_column()
                .ok_or_else(|| ValidationError::estimation_msg("calibration: no effect"))?;
            let mut draws = post.draws.column(col).map_err(ValidationError::from)?.to_vec();
            draws.sort_by(f64::total_cmp);
            let lo = quantile_type7_sorted(&draws, alpha);
            let hi = quantile_type7_sorted(&draws, 1.0 - alpha);
            let mean = reduce_posterior_draws(&draws, PosteriorReduceOp::Mean, &ctx.kernel_policy)
                .unwrap_or(0.0);
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
