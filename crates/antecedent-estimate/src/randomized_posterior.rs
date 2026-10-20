//! Bayesian counterpart of the randomized-trial arm contrasts: two arms, many arms and the 2x2
//! factorial.
//!
//! # Target and the two estimands
//!
//! The frequentist routes ([`crate::randomized_scores`], [`crate::multi_arm`],
//! [`crate::factorial_bernoulli`], [`crate::randomized_neyman`]) are design based: they estimate
//! the average of the unit-level potential outcomes over the `n` units of the trial (the
//! finite-population estimand) and their interval comes from the randomization distribution.
//! That interval is not a posterior. A posterior needs a likelihood, so this module states one
//! and answers **both** estimands under it:
//!
//! * [`RandomizedEstimand::Superpopulation`]: `theta_k = E[Y(k)]`, the mean of arm `k` in the
//!   population the units are sampled from. Observed units of arm `k` are exchangeable draws from
//!   the arm's outcome law, so the arm posteriors are independent Beta (binary outcomes) or
//!   Student-t (Gaussian outcomes with unknown variance) and a contrast `sum_k w_k theta_k` is a
//!   linear combination of independent posteriors.
//! * [`RandomizedEstimand::FinitePopulation`]: `ybar_k = n^-1 sum_i Y_i(k)`, the mean of arm `k`
//!   over the trial's own units. The `n_k` observed values are known; the `n - n_k` unobserved
//!   `Y_i(k)` are drawn from the arm's posterior predictive. Contrasts are linear in the arm
//!   means, so the unknown cross-arm dependence of `(Y_i(0), Y_i(1))` plays no role and the
//!   finite-population contrast has an exact posterior too (a Beta-Binomial convolution for
//!   binary outcomes; a Student-t predictive sum for Gaussian outcomes).
//!
//! The two laws answer different questions. The finite-population arm mean replaces the unobserved
//! share of each arm by its posterior predictive, so it is pulled toward the observed arm means
//! and each arm's parameter variance is scaled by `(1 - n_k / n)^2`: at fixed allocation fractions
//! the two contrast variances keep a constant ratio below one and never coincide, while their
//! centres converge to the same value as `n` grows. When every unit is observed under the arm the
//! finite-population arm mean is the observed mean and has no uncertainty at all. The report
//! always says which estimand it answers.
//!
//! # Randomization and ignorability
//!
//! Assignment probabilities are known and enter only through the row check: the posterior is a
//! function of the observed outcomes by arm, so it is the same under any assignment mechanism
//! that depends only on observed data and independent randomization (the likelihood principle).
//! The per-arm exchangeability used above requires the assignment probability of an arm to be
//! the same for every unit, so rows whose propensities differ are refused: a covariate-dependent
//! propensity makes `Y | A = k` a mixture over the covariate, and the arm posterior would then
//! target the confounded conditional mean. The next attempt is a stratified model with one
//! posterior per propensity stratum, or the Horvitz-Thompson route, which weights instead.
//!
//! # Numerics
//!
//! A contrast law is the convolution of the independent arm laws on a common lattice
//! (the private `posterior_lattice` module); the finite-population Beta-Binomial arm laws are carried
//! exactly and the continuous ones are discretized with a within-cell variance inflation of
//! about `1 / 3000` of the arm variance.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::reason_code;
use antecedent_stats::adaptive_inference::{BetaPosterior, regularized_incomplete_beta};
use antecedent_stats::{ln_gamma, student_t_ppf, student_t_sf};

use crate::error::EstimationError;
use crate::gaussian_linear_posterior::StudentPosterior;
use crate::posterior_lattice::{
    CELLS_PER_SCALE, LatticeTerm, convolve_terms, discretize, normalize,
};
use crate::util::refuse;

/// Likelihood label of the binary-outcome posterior, for evidence records.
pub const BERNOULLI_ARM_LIKELIHOOD: &str = "independent_bernoulli_arms_beta_prior";

/// Likelihood label of the Gaussian-outcome posterior, for evidence records.
pub const GAUSSIAN_ARM_LIKELIHOOD: &str = "independent_gaussian_arms_normal_inverse_gamma_prior";

/// Tail probability cut off each end of a continuous marginal's window.
const WINDOW_TAIL: f64 = 1e-8;

/// Which population the contrast is about.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RandomizedEstimand {
    /// Contrast of the arm means in the sampled population.
    Superpopulation,
    /// Contrast of the arm means over the trial's own units.
    FinitePopulation,
}

/// Beta prior of a Bernoulli arm; both shapes finite and positive, and each arm's posterior
/// shapes (prior plus counts) at least one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArmBetaPrior {
    /// First shape.
    pub alpha: f64,
    /// Second shape.
    pub beta: f64,
}

/// Normal-Inverse-Gamma prior of a Gaussian arm: `mu | s2 ~ N(mean, s2 / pseudo_units)`,
/// `s2 ~ InvGamma(shape, scale)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArmNormalPrior {
    /// Prior mean of the arm mean.
    pub mean: f64,
    /// Prior pseudo-observations `kappa0`; positive.
    pub pseudo_units: f64,
    /// Inverse-Gamma shape; positive.
    pub variance_shape: f64,
    /// Inverse-Gamma scale; positive.
    pub variance_scale: f64,
}

/// The outcome model of every arm.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ArmOutcomeModel {
    /// Binary outcomes with a Beta prior per arm.
    Bernoulli(ArmBetaPrior),
    /// Real outcomes with a Normal-Inverse-Gamma prior per arm.
    Gaussian(ArmNormalPrior),
}

/// Rows, model, contrasts and level of one posterior request.
#[derive(Clone, Debug)]
pub struct ArmContrastRequest<'a> {
    /// Outcome of each unit under its assigned arm.
    pub outcome: &'a [f64],
    /// Assigned arm of each unit.
    pub assignment: &'a [usize],
    /// Known assignment probabilities of each unit, one per arm, each row summing to one and
    /// identical across units.
    pub probabilities: &'a [Vec<f64>],
    /// The outcome model.
    pub model: ArmOutcomeModel,
    /// Contrast weights, one vector per contrast: length equal to the number of arms, summing to
    /// zero, every entry a multiple of one half.
    pub contrasts: &'a [Vec<f64>],
    /// Superpopulation or finite-population contrast.
    pub estimand: RandomizedEstimand,
    /// Credible level in `(0, 1)` of the equal-tailed interval.
    pub credible_level: f64,
}

/// Posterior of one arm's population mean (the superpopulation arm posterior).
#[derive(Clone, Debug, PartialEq)]
pub struct ArmPosteriorSummary {
    /// Units observed under the arm.
    pub observed_units: usize,
    /// Posterior mean of the arm mean.
    pub mean: f64,
    /// Posterior standard deviation of the arm mean; `NaN` when it is not finite.
    pub standard_deviation: f64,
}

/// Posterior of one contrast.
#[derive(Clone, Debug, PartialEq)]
pub struct ContrastSummary {
    /// The contrast weights.
    pub weights: Vec<f64>,
    /// The estimand the law answers.
    pub estimand: RandomizedEstimand,
    /// Posterior mean.
    pub mean: f64,
    /// Posterior standard deviation.
    pub standard_deviation: f64,
    /// Equal-tailed credible interval. For the exact discrete finite-population law the endpoints
    /// are the smallest support values whose CDF reaches each tail level, so the interval holds
    /// at least the stated mass.
    pub interval: [f64; 2],
    /// Posterior probability that the contrast is positive (strictly, for a discrete law).
    pub probability_positive: f64,
    /// Deterministic quantile-grid draws of the contrast law (the quantiles at `(i + 1/2) / m`),
    /// empty unless a draw count was requested with [`posterior_arm_contrasts_with_draws`].
    pub draws: Vec<f64>,
}

/// What the assignment mechanism contributed, kept for evidence synthesis.
#[derive(Clone, Debug, PartialEq)]
pub struct RandomizationEvidence {
    /// Units in the trial.
    pub units: usize,
    /// The constant known assignment probability of each arm.
    pub arm_probabilities: Vec<f64>,
    /// The premise under which the posterior ignores the assignment mechanism.
    pub ignorability: &'static str,
}

/// The posterior of a randomized trial's arm contrasts.
#[derive(Clone, Debug, PartialEq)]
pub struct RandomizedPosterior {
    /// Per-arm superpopulation posterior summaries.
    pub arms: Vec<ArmPosteriorSummary>,
    /// Per-contrast posterior, in request order.
    pub contrasts: Vec<ContrastSummary>,
    /// Credible level of every interval.
    pub credible_level: f64,
    /// Likelihood label.
    pub likelihood: &'static str,
    /// Randomization evidence.
    pub evidence: RandomizationEvidence,
}

/// Premise text recorded in [`RandomizationEvidence::ignorability`].
pub const RANDOMIZATION_IGNORABILITY: &str = "known unit-independent arm probabilities; \
    the assignment depends on no unobserved outcome or parameter, so the likelihood factorizes";

fn refused(code: &'static str, detail: &str, message: &str) -> EstimationError {
    refuse(code, detail, message)
}

fn invalid(detail: &str, message: &str) -> EstimationError {
    refused(reason_code!("invalid_argument"), detail, message)
}

struct ArmData {
    rows: usize,
    sum: f64,
    squares: f64,
}

/// One arm's law of the quantity the contrast weights act on.
enum ArmLaw {
    Beta(BetaPosterior),
    Student(StudentPosterior),
    FiniteBinary { observed: u64, missing: u64, alpha: f64, beta: f64 },
}

fn validate_rows(
    request: &ArmContrastRequest<'_>,
) -> Result<(Vec<ArmData>, Vec<f64>), EstimationError> {
    let n = request.outcome.len();
    let arms = request.probabilities.first().map_or(0, Vec::len);
    if n == 0 || arms < 2 || request.assignment.len() != n || request.probabilities.len() != n {
        return Err(invalid(
            "randomized_posterior.shape",
            "outcome, assignment and probability rows must align with at least two arms",
        ));
    }
    let reference = &request.probabilities[0];
    let total: f64 = reference.iter().sum();
    if reference.iter().any(|p| !p.is_finite() || *p <= 0.0 || *p > 1.0)
        || (total - 1.0).abs() > 1e-8
    {
        return Err(invalid(
            "randomized_posterior.invalid_probability",
            "each arm probability must be positive and the row must sum to one",
        ));
    }
    let constant = request.probabilities.iter().all(|row| {
        row.len() == arms && row.iter().zip(reference).all(|(a, b)| (a - b).abs() <= 1e-12)
    });
    if !constant {
        return Err(refused(
            reason_code!("route_not_supported"),
            "randomized_posterior.varying_propensity",
            "per-arm exchangeability needs the same assignment probabilities for every unit; \
             with unit-varying propensities Y | A is a covariate mixture. Next attempt: model \
             each propensity stratum separately, or use the Horvitz-Thompson route",
        ));
    }
    let mut data: Vec<ArmData> =
        (0..arms).map(|_| ArmData { rows: 0, sum: 0.0, squares: 0.0 }).collect();
    for (value, arm) in request.outcome.iter().zip(request.assignment) {
        let slot = data.get_mut(*arm).ok_or_else(|| {
            invalid("randomized_posterior.shape", "an assigned arm is outside the arm set")
        })?;
        if !value.is_finite() {
            return Err(invalid("randomized_posterior.invalid_outcome", "outcomes must be finite"));
        }
        let binary = value.to_bits() == 0 || value.to_bits() == 1.0_f64.to_bits();
        if matches!(request.model, ArmOutcomeModel::Bernoulli(_)) && !binary {
            return Err(refused(
                reason_code!("route_not_supported"),
                "randomized_posterior.non_binary_outcome",
                "the Bernoulli model needs 0/1 outcomes; use the Gaussian arm model for real \
                 outcomes or the Horvitz-Thompson route",
            ));
        }
        slot.rows += 1;
        slot.sum += value;
        slot.squares += value * value;
    }
    if let Some(empty) = data.iter().position(|d| d.rows == 0) {
        return Err(refused(
            reason_code!("arm_not_populated"),
            "randomized_posterior.arm_not_populated",
            &format!("arm {empty} has no observed unit"),
        ));
    }
    Ok((data, reference.clone()))
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "the doubled weights are validated near-integral before the cast"
)]
fn check_contrasts(contrasts: &[Vec<f64>], arms: usize) -> Result<Vec<Vec<i64>>, EstimationError> {
    let bad = |message: &str| invalid("randomized_posterior.invalid_contrast", message);
    if contrasts.is_empty() {
        return Err(bad("at least one contrast is required"));
    }
    contrasts
        .iter()
        .map(|weights| {
            let doubled: Vec<f64> = weights.iter().map(|w| 2.0 * w).collect();
            let integral = doubled.iter().all(|d| d.is_finite() && (d - d.round()).abs() <= 1e-9);
            if weights.len() != arms || !integral || weights.iter().sum::<f64>().abs() > 1e-9 {
                return Err(bad(
                    "a contrast needs one weight per arm, each a multiple of one half, summing to zero",
                ));
            }
            Ok(doubled.iter().map(|d| d.round() as i64).collect())
        })
        .collect()
}

fn arm_laws(
    request: &ArmContrastRequest<'_>,
    data: &[ArmData],
) -> Result<Vec<(ArmLaw, ArmPosteriorSummary)>, EstimationError> {
    let n = request.outcome.len();
    data.iter()
        .map(|arm| match request.model {
            ArmOutcomeModel::Bernoulli(prior) => bernoulli_arm(request, arm, n, prior),
            ArmOutcomeModel::Gaussian(prior) => gaussian_arm(request, arm, n, prior),
        })
        .collect()
}

fn bernoulli_arm(
    request: &ArmContrastRequest<'_>,
    arm: &ArmData,
    n: usize,
    prior: ArmBetaPrior,
) -> Result<(ArmLaw, ArmPosteriorSummary), EstimationError> {
    // Rows were validated as 0 or 1, so the arm sum is its success count.
    let successes_f = arm.sum.round();
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "checked integral and within [0, rows]"
    )]
    let successes = successes_f as u64;
    let failures = arm.rows as u64 - successes;
    let valid_prior =
        prior.alpha.is_finite() && prior.beta.is_finite() && prior.alpha > 0.0 && prior.beta > 0.0;
    let posterior = valid_prior
        .then(|| BetaPosterior::from_counts(prior.alpha, prior.beta, successes, failures).ok())
        .flatten()
        .ok_or_else(|| {
            invalid(
                "randomized_posterior.invalid_prior",
                "Beta prior shapes must be finite and positive with posterior shapes of at least one",
            )
        })?;
    let summary = ArmPosteriorSummary {
        observed_units: arm.rows,
        mean: posterior.mean(),
        standard_deviation: posterior.variance().sqrt(),
    };
    let law = match request.estimand {
        RandomizedEstimand::Superpopulation => ArmLaw::Beta(posterior),
        RandomizedEstimand::FinitePopulation => ArmLaw::FiniteBinary {
            observed: successes,
            missing: (n - arm.rows) as u64,
            alpha: posterior.alpha(),
            beta: posterior.beta(),
        },
    };
    Ok((law, summary))
}

fn gaussian_arm(
    request: &ArmContrastRequest<'_>,
    arm: &ArmData,
    n: usize,
    prior: ArmNormalPrior,
) -> Result<(ArmLaw, ArmPosteriorSummary), EstimationError> {
    let positive = |v: f64| v.is_finite() && v > 0.0;
    if !(prior.mean.is_finite()
        && positive(prior.pseudo_units)
        && positive(prior.variance_shape)
        && positive(prior.variance_scale))
    {
        return Err(invalid(
            "randomized_posterior.invalid_prior",
            "the Normal-Inverse-Gamma prior needs a finite mean and positive pseudo-units, shape and scale",
        ));
    }
    let rows = arm.rows as f64;
    let mean = arm.sum / rows;
    let deviations = (arm.squares - rows * mean * mean).max(0.0);
    let kappa = prior.pseudo_units + rows;
    let posterior_mean = (prior.pseudo_units * prior.mean + rows * mean) / kappa;
    let shape = prior.variance_shape + rows / 2.0;
    let scale = prior.variance_scale
        + 0.5 * deviations
        + 0.5 * prior.pseudo_units * rows * (mean - prior.mean).powi(2) / kappa;
    let df = 2.0 * shape;
    let marginal = StudentPosterior {
        location: posterior_mean,
        scale: (scale / (shape * kappa)).sqrt(),
        degrees_of_freedom: df,
    };
    let summary = ArmPosteriorSummary {
        observed_units: arm.rows,
        mean: posterior_mean,
        standard_deviation: marginal.standard_deviation(),
    };
    let law = match request.estimand {
        RandomizedEstimand::Superpopulation => ArmLaw::Student(marginal),
        RandomizedEstimand::FinitePopulation => {
            // Mean of the n units: observed sum plus the predictive sum of the n - n_k missing
            // values, a Student-t with scale^2 = (b / a)(N + N^2 / kappa).
            let n_f = n as f64;
            let missing = n_f - rows;
            let predictive = (scale / shape * (missing + missing * missing / kappa)).sqrt();
            ArmLaw::Student(StudentPosterior {
                location: (arm.sum + missing * posterior_mean) / n_f,
                scale: predictive / n_f,
                degrees_of_freedom: df,
            })
        }
    };
    Ok((law, summary))
}

/// Exact Beta-Binomial pmf of the number of successes among `missing` unobserved units.
fn beta_binomial_pmf(missing: u64, alpha: f64, beta: f64) -> Option<Vec<f64>> {
    let m = missing as f64;
    let ln_beta = |a: f64, b: f64| ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
    let base = ln_beta(alpha, beta);
    let mut pmf: Vec<f64> = (0..=missing)
        .map(|k| {
            let k = k as f64;
            let choose = ln_gamma(m + 1.0) - ln_gamma(k + 1.0) - ln_gamma(m - k + 1.0);
            (choose + ln_beta(k + alpha, m - k + beta) - base).exp()
        })
        .collect();
    normalize(&mut pmf)?;
    Some(pmf)
}

/// A scale that sets the lattice width: the arm's posterior standard deviation or scale.
fn law_scale(law: &ArmLaw) -> f64 {
    match law {
        ArmLaw::Beta(beta) => beta.variance().sqrt(),
        ArmLaw::Student(t) => t.scale,
        ArmLaw::FiniteBinary { .. } => f64::INFINITY,
    }
}

fn arm_term(law: &ArmLaw, h: f64) -> Result<LatticeTerm, EstimationError> {
    let numerical = || {
        refused(
            reason_code!("route_not_supported"),
            "posterior_lattice.numerical_failure",
            "an arm posterior could not be evaluated",
        )
    };
    match law {
        ArmLaw::Beta(beta) => {
            let cdf = |x: f64| {
                regularized_incomplete_beta(x.clamp(0.0, 1.0), beta.alpha(), beta.beta()).ok()
            };
            let sd = beta.variance().sqrt();
            let lo = (beta.mean() - 14.0 * sd).max(0.0);
            let hi = (beta.mean() + 14.0 * sd).min(1.0);
            discretize(&cdf, lo, hi, h)
        }
        ArmLaw::Student(t) => {
            let cdf =
                |x: f64| Some(1.0 - student_t_sf((x - t.location) / t.scale, t.degrees_of_freedom));
            let radius = -student_t_ppf(WINDOW_TAIL, t.degrees_of_freedom);
            if !radius.is_finite() {
                return Err(numerical());
            }
            discretize(&cdf, t.location - radius * t.scale, t.location + radius * t.scale, h)
        }
        ArmLaw::FiniteBinary { observed, missing, alpha, beta } => {
            let pmf = beta_binomial_pmf(*missing, *alpha, *beta).ok_or_else(numerical)?;
            // The lattice is the integer success count of the n units; `h` equals `1 / n`.
            Ok(LatticeTerm { start: i64::try_from(*observed).unwrap_or(i64::MAX), pmf })
        }
    }
}

fn summarize_contrast(
    law: &crate::posterior_lattice::LatticeLaw,
    weights: &[f64],
    estimand: RandomizedEstimand,
    level: f64,
    draws: usize,
) -> ContrastSummary {
    let tail = 0.5 * (1.0 - level);
    ContrastSummary {
        draws: if draws == 0 { Vec::new() } else { law.quantile_grid(draws) },
        weights: weights.to_vec(),
        estimand,
        mean: law.mean(),
        standard_deviation: law.standard_deviation(),
        interval: [law.quantile(tail), law.quantile(1.0 - tail)],
        probability_positive: law.probability_positive(),
    }
}

/// Posterior of the declared arm contrasts of a randomized trial.
///
/// # Errors
///
/// `invalid_argument` with `randomized_posterior.shape`, `.invalid_probability`,
/// `.invalid_outcome`, `.invalid_prior`, `.invalid_contrast` or `.invalid_level`;
/// `route_not_supported` with `randomized_posterior.varying_propensity` (the next attempt is
/// named in the message) or `randomized_posterior.non_binary_outcome`; `arm_not_populated` with
/// `randomized_posterior.arm_not_populated`; and the `posterior_lattice.*` limits for a
/// posterior too concentrated or a convolution beyond its work bound.
pub fn posterior_arm_contrasts(
    request: &ArmContrastRequest<'_>,
) -> Result<RandomizedPosterior, EstimationError> {
    posterior_arm_contrasts_with_draws(request, 0)
}

/// [`posterior_arm_contrasts`] that also returns `draws` quantile-grid draws of every contrast
/// law (none when `draws` is zero). The draws are the exact law's quantiles at `(i + 1/2) / draws`,
/// not Monte Carlo samples.
///
/// # Errors
///
/// As [`posterior_arm_contrasts`].
pub fn posterior_arm_contrasts_with_draws(
    request: &ArmContrastRequest<'_>,
    draws: usize,
) -> Result<RandomizedPosterior, EstimationError> {
    let level = request.credible_level;
    if !(level.is_finite() && level > 0.0 && level < 1.0) {
        return Err(invalid(
            "randomized_posterior.invalid_level",
            "the credible level must lie strictly between zero and one",
        ));
    }
    let (data, probabilities) = validate_rows(request)?;
    let coefficients = check_contrasts(request.contrasts, data.len())?;
    let laws = arm_laws(request, &data)?;
    let n = request.outcome.len();
    let mut contrasts = Vec::with_capacity(coefficients.len());
    for (weights, integers) in request.contrasts.iter().zip(&coefficients) {
        let used: Vec<usize> = (0..integers.len()).filter(|k| integers[*k] != 0).collect();
        let discrete = used.iter().all(|k| matches!(laws[*k].0, ArmLaw::FiniteBinary { .. }));
        let scale = used.iter().map(|k| law_scale(&laws[*k].0)).fold(f64::INFINITY, f64::min);
        // Exact discrete laws live on the lattice of width 1 / n; a mixed or continuous request
        // takes the width from the narrowest continuous marginal.
        let h = if discrete { 1.0 / n as f64 } else { scale / CELLS_PER_SCALE };
        let mut terms = Vec::with_capacity(used.len());
        for k in &used {
            terms.push((integers[*k], arm_term(&laws[*k].0, h)?));
        }
        let law = convolve_terms(&terms, 0.5 * h, !discrete)?;
        contrasts.push(summarize_contrast(&law, weights, request.estimand, level, draws));
    }
    Ok(RandomizedPosterior {
        arms: laws.into_iter().map(|(_, summary)| summary).collect(),
        contrasts,
        credible_level: level,
        likelihood: match request.model {
            ArmOutcomeModel::Bernoulli(_) => BERNOULLI_ARM_LIKELIHOOD,
            ArmOutcomeModel::Gaussian(_) => GAUSSIAN_ARM_LIKELIHOOD,
        },
        evidence: RandomizationEvidence {
            units: n,
            arm_probabilities: probabilities,
            ignorability: RANDOMIZATION_IGNORABILITY,
        },
    })
}

/// Model, estimand and credible level of a two-arm or factorial posterior.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArmPosteriorSpec {
    /// The outcome model.
    pub model: ArmOutcomeModel,
    /// Superpopulation or finite-population contrast.
    pub estimand: RandomizedEstimand,
    /// Credible level in `(0, 1)`.
    pub credible_level: f64,
}

/// Posterior of the treated-minus-control contrast of a two-arm trial with a constant known
/// treatment probability.
///
/// # Errors
///
/// As [`posterior_arm_contrasts`].
pub fn posterior_two_arm(
    outcome: &[f64],
    treated: &[bool],
    probability_treated: &[f64],
    spec: &ArmPosteriorSpec,
) -> Result<RandomizedPosterior, EstimationError> {
    let assignment: Vec<usize> = treated.iter().map(|t| usize::from(*t)).collect();
    let probabilities: Vec<Vec<f64>> =
        probability_treated.iter().map(|p| vec![1.0 - p, *p]).collect();
    posterior_arm_contrasts(&ArmContrastRequest {
        outcome,
        assignment: &assignment,
        probabilities: &probabilities,
        model: spec.model,
        contrasts: &[vec![-1.0, 1.0]],
        estimand: spec.estimand,
        credible_level: spec.credible_level,
    })
}

/// Posterior of the 2x2 factorial contrasts of independent Bernoulli factors with constant
/// probabilities: the main effect of A, the main effect of B and the interaction, in the cell
/// order and weights of [`crate::factorial_bernoulli::estimate_factorial_2x2`] (A and B main
/// effects with weights `+-1/2`, interaction `[1, -1, -1, 1]`). Cell order is `[a=0,b=0]`,
/// `[a=0,b=1]`, `[a=1,b=0]`, `[a=1,b=1]`; contrasts are returned in the order A, B, interaction.
///
/// # Errors
///
/// As [`posterior_arm_contrasts`]; a factor probability that varies across units is
/// `randomized_posterior.varying_propensity`.
pub fn posterior_factorial_2x2(
    outcome: &[f64],
    factor_a: &[bool],
    factor_b: &[bool],
    probability_a: &[f64],
    probability_b: &[f64],
    spec: &ArmPosteriorSpec,
) -> Result<RandomizedPosterior, EstimationError> {
    let n = outcome.len();
    if factor_a.len() != n
        || factor_b.len() != n
        || probability_a.len() != n
        || probability_b.len() != n
    {
        return Err(invalid(
            "randomized_posterior.shape",
            "outcome, factor assignments and probability vectors must have equal length",
        ));
    }
    let valid = |p: &f64| p.is_finite() && *p > 0.0 && *p < 1.0;
    if !probability_a.iter().chain(probability_b).all(valid) {
        return Err(invalid(
            "randomized_posterior.invalid_probability",
            "both factor probabilities must be strictly between zero and one",
        ));
    }
    let assignment: Vec<usize> =
        factor_a.iter().zip(factor_b).map(|(a, b)| 2 * usize::from(*a) + usize::from(*b)).collect();
    let probabilities: Vec<Vec<f64>> = probability_a
        .iter()
        .zip(probability_b)
        .map(|(pa, pb)| vec![(1.0 - pa) * (1.0 - pb), (1.0 - pa) * pb, pa * (1.0 - pb), pa * pb])
        .collect();
    posterior_arm_contrasts(&ArmContrastRequest {
        outcome,
        assignment: &assignment,
        probabilities: &probabilities,
        model: spec.model,
        contrasts: &[
            vec![-0.5, -0.5, 0.5, 0.5],
            vec![-0.5, 0.5, -0.5, 0.5],
            vec![1.0, -1.0, -1.0, 1.0],
        ],
        estimand: spec.estimand,
        credible_level: spec.credible_level,
    })
}
