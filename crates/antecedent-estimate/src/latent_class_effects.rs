//! B4 latent-class (finite mixture) regime effects.
//!
//! # The cell
//!
//! The population is a finite mixture of `K` latent regimes (classes), `2 <= K <= 4`. Within
//! class `k` the outcome follows a Gaussian linear model in the treatment `A` and the observed
//! covariates `X`:
//!
//! ```text
//! Y | A, X, class = k  ~  N(alpha_k + tau_k * A + gamma_k' X, sigma_k^2)
//! P(class = k)         =  pi_k            (constant class weights)
//! ```
//!
//! `tau_k` is the class-specific effect: the change in the class-`k` mean outcome per unit of
//! treatment (for a binary treatment it is the class-specific average treatment effect). The
//! covariates `X` enter every class's mean, so they inform class assignment through the
//! class-specific regression surfaces. A covariate-dependent (multinomial-logit) class prior is
//! deliberately **not** offered: the class weights are constants.
//!
//! # Estimands
//!
//! * the class-specific effects `tau_k`;
//! * the class weights `pi_k`;
//! * the mixture-average effect `sum_k pi_k * tau_k`;
//! * the per-unit posterior class responsibilities `P(class = k | y_i, a_i, x_i)`.
//!
//! # Estimation
//!
//! Maximum likelihood by EM. Each EM start is a deterministic seeded initialization (a
//! seeded `k-means++`-style choice of `K` outcome centers; restart 0 uses farthest-point
//! centers, later restarts use squared-distance sampling), or one caller-supplied hard
//! labelling ([`LatentClassConfig::initial_labels`]) which then replaces the restarts. The
//! M-step is a class-weighted least-squares fit; the E-step is computed in log space.
//! Convergence is `|ll_t - ll_{t-1}| <= tolerance * (1 + |ll_{t-1}|)` within
//! [`LatentClassConfig::max_iterations`]. The observed-data log-likelihood is checked to be
//! non-decreasing at every iteration (EM guarantees it; a violation is a numerical failure and
//! refuses). The restart with the highest converged log-likelihood is reported. Classes whose
//! effective size falls below the regression dimension plus one, whose design is rank
//! deficient, or whose residual variance falls to the declared floor are *degenerate*
//! (the unbounded-likelihood spurious solutions of Gaussian mixtures); such restarts are
//! discarded.
//!
//! # Label handling
//!
//! Mixture likelihoods are invariant under relabelling classes, so the raw EM labels carry no
//! meaning. Results are reported in a **canonical order**: ascending `tau_k`, ties broken by
//! ascending `alpha_k`, then by ascending `pi_k`. [`LatentClassResult::class_order`] retains the
//! mapping `canonical index -> raw EM component index` of the selected fit. Permuting the
//! initial labels or the seed therefore cannot change the reported result (beyond roundoff).
//! "Class 0" is the lowest-effect class, not a substantive name. If two classes share a
//! `tau` and an `alpha` the order is not informative about identity and the classes are not
//! separately identified in effect; that is reported through [`LatentClassResult::min_effect_gap`].
//!
//! # Sampling uncertainty
//!
//! Standard errors come from a **seeded nonparametric bootstrap with label alignment** (not
//! from the observed-information matrix, whose singularities near weak classes make it
//! unreliable). Each replicate resamples rows with replacement, warm-starts EM from the point
//! estimate, orders the fitted classes canonically and then aligns them to the point-estimate
//! classes by the permutation minimizing the squared distance in `(tau, alpha)`. Failed
//! replicates (degenerate, not converged) are counted and skipped; fewer than two successes
//! refuses. The reported quantity is the bootstrap standard deviation; **calibration is
//! [`CalibrationStatus::Unmeasured`]** and no coverage or interval claim is made.
//!
//! # Identification premises
//!
//! Declared by the caller (not checkable from the data):
//!
//! * treatment is as-if randomized *within class given `X`* (conditional ignorability inside
//!   each latent regime); the caller must set
//!   [`LatentClassConfig::assume_conditional_randomization`] or the fit refuses;
//! * the Gaussian-linear class model and constant class weights hold;
//! * `K` is the true number of classes (it is declared, not selected).
//!
//! Checked here:
//!
//! * the treatment varies, and the number of distinct `(A, X)` design points is at least
//!   `d + K - 1` with `d = 2 + dim(X)` (a necessary condition for mixture-of-regressions
//!   identifiability, not a sufficient one);
//! * every class weight is at least [`LatentClassConfig::min_class_weight`] and the
//!   *separation* (mean over units of the largest posterior responsibility) is at least
//!   [`LatentClassConfig::min_separation`], otherwise the fit refuses as a weak class;
//! * no class is degenerate; the EM converged; the likelihood was monotone.
//!
//! # Refusals
//!
//! Plain-string details, all prefixed `latent_class.`:
//!
//! * `latent_class.not_converged` (`mechanism_fit_not_converged`): no restart converged;
//!   `latent_class.likelihood_not_monotone` (same code): EM likelihood decreased;
//! * `latent_class.degenerate_class` (`population_not_estimable`): a single class requested
//!   (`K < 2`), or every restart collapsed a class;
//! * `latent_class.weak_class` (`population_not_estimable`): a class weight or the separation
//!   is below the declared minimum;
//! * `latent_class.randomization_not_declared` (`required_option_missing`);
//! * `latent_class.treatment_constant` (`effect_not_identified`);
//! * `latent_class.design_support_too_small` (`design_rank_deficient`);
//! * `latent_class.constant_outcome` (`population_not_estimable`);
//! * `latent_class.too_many_classes`, `latent_class.invalid_config`,
//!   `latent_class.length_mismatch`, `latent_class.non_finite_input`,
//!   `latent_class.too_few_rows`, `latent_class.too_many_covariates`
//!   (`invalid_argument`).
//!
//! # What a result does not say
//!
//! Point estimates and bootstrap standard errors only; the bootstrap's coverage is unmeasured.
//! Nothing here licenses a causal claim beyond the declared premises, and the class labels are
//! an ordering convenience, not a discovered meaning.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::needless_range_loop,
    reason = "dense linear-algebra loops index several parallel arrays by one position"
)]

use std::cmp::Ordering;
use std::f64::consts::TAU as TWO_PI;

use antecedent_core::reason_code;

use crate::effect_constancy::CalibrationStatus;
use crate::error::EstimationError;
use crate::splitmix::{mix64, splitmix64};

/// Most latent classes accepted.
pub const LATENT_CLASS_MAX_CLASSES: usize = 4;

/// Most restarts accepted.
pub const LATENT_CLASS_MAX_RESTARTS: usize = 64;

/// Most covariates accepted.
pub const LATENT_CLASS_MAX_COVARIATES: usize = 64;

/// Most bootstrap replicates accepted.
pub const LATENT_CLASS_MAX_BOOTSTRAP: usize = 5000;

/// Inference claim of the route.
pub const LATENT_CLASS_INFERENCE_CLAIM: &str = "point_with_bootstrap_se";

/// Statement attached to every result.
pub const LATENT_CLASS_CAVEAT: &str = "bootstrap standard errors are unmeasured for coverage; class \
                                       labels are an effect ordering, not a discovered meaning; \
                                       identification rests on the declared within-class \
                                       randomization and Gaussian-linear class models";

/// Relative tolerance of the log-likelihood monotonicity check.
const MONOTONE_TOLERANCE: f64 = 1e-8;

/// Whether a premise was verified from the data or only declared by the caller.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PremiseStatus {
    /// Verified here from the data and the fit.
    Checked,
    /// Asserted by the caller; not checkable from the data.
    Declared,
}

/// One identification premise and its status.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LatentClassPremise {
    /// Premise name.
    pub name: &'static str,
    /// Checked or declared.
    pub status: PremiseStatus,
}

/// Borrowed analysis data.
#[derive(Clone, Copy, Debug)]
pub struct LatentClassData<'a> {
    /// Outcome `Y`, one value per unit.
    pub outcome: &'a [f64],
    /// Treatment `A` (binary 0/1 or numeric), one value per unit.
    pub treatment: &'a [f64],
    /// Covariate columns `X`, each of length `n`.
    pub covariates: &'a [Vec<f64>],
}

/// Fit configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct LatentClassConfig {
    /// Number of latent classes `K`, `2..=4`.
    pub classes: usize,
    /// Seed of every deterministic stream (initialization, bootstrap).
    pub seed: u64,
    /// EM restarts (ignored when [`Self::initial_labels`] is set).
    pub restarts: usize,
    /// EM iteration bound per start.
    pub max_iterations: usize,
    /// Relative log-likelihood convergence tolerance.
    pub tolerance: f64,
    /// Smallest admissible class weight.
    pub min_class_weight: f64,
    /// Smallest admissible separation (mean largest posterior responsibility), in `[0, 1]`.
    pub min_separation: f64,
    /// Residual-variance floor as a fraction of the outcome variance.
    pub variance_floor: f64,
    /// Bootstrap replicates; `0` reports no standard errors.
    pub bootstrap_replicates: usize,
    /// One hard initial labelling (`labels[i] < classes`) used as the only start.
    pub initial_labels: Option<Vec<usize>>,
    /// The caller declares treatment as-if randomized within class given `X`.
    pub assume_conditional_randomization: bool,
}

impl LatentClassConfig {
    /// Defaults for `classes` classes and `seed`: 10 restarts, 500 iterations, tolerance
    /// `1e-10`, minimum weight `0.05`, minimum separation `0.7`, variance floor `1e-8`,
    /// 200 bootstrap replicates. Conditional randomization is **not** declared; set
    /// [`Self::assume_conditional_randomization`] to proceed.
    #[must_use]
    pub fn new(classes: usize, seed: u64) -> Self {
        Self {
            classes,
            seed,
            restarts: 10,
            max_iterations: 500,
            tolerance: 1e-10,
            min_class_weight: 0.05,
            min_separation: 0.7,
            variance_floor: 1e-8,
            bootstrap_replicates: 200,
            initial_labels: None,
            assume_conditional_randomization: false,
        }
    }

    /// Declare treatment as-if randomized within class given `X`.
    #[must_use]
    pub fn declare_conditional_randomization(mut self) -> Self {
        self.assume_conditional_randomization = true;
        self
    }
}

/// One class's report (canonical order).
#[derive(Clone, Debug, PartialEq)]
pub struct LatentClassEffect {
    /// Canonical index of the class.
    pub index: usize,
    /// Raw EM component index of the selected fit this class came from.
    pub raw_index: usize,
    /// Class weight `pi_k`.
    pub weight: f64,
    /// Bootstrap standard error of the weight.
    pub weight_se: Option<f64>,
    /// Class intercept `alpha_k`.
    pub intercept: f64,
    /// Class effect `tau_k`.
    pub effect: f64,
    /// Bootstrap standard error of the effect.
    pub effect_se: Option<f64>,
    /// Covariate coefficients `gamma_k`.
    pub covariate_coefficients: Vec<f64>,
    /// Residual variance `sigma_k^2` (maximum likelihood).
    pub residual_variance: f64,
    /// Effective class size: the sum of the class's responsibilities.
    pub effective_n: f64,
}

/// Bootstrap bookkeeping.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LatentClassBootstrap {
    /// Replicates requested.
    pub requested: usize,
    /// Replicates that converged to a non-degenerate fit.
    pub succeeded: usize,
    /// Replicates skipped.
    pub failed: usize,
}

/// Result of a latent-class effect fit.
#[derive(Clone, Debug, PartialEq)]
pub struct LatentClassResult {
    /// Classes in canonical order.
    pub classes: Vec<LatentClassEffect>,
    /// `class_order[c]` is the raw EM component index of canonical class `c`.
    pub class_order: Vec<usize>,
    /// Mixture-average effect `sum_k pi_k tau_k`.
    pub mixture_average_effect: f64,
    /// Bootstrap standard error of the mixture-average effect.
    pub mixture_average_se: Option<f64>,
    /// Posterior responsibilities, `n` rows by `K` canonical columns.
    pub responsibilities: Vec<Vec<f64>>,
    /// Most probable canonical class of each unit.
    pub hard_assignment: Vec<usize>,
    /// Observed-data log-likelihood of the selected fit.
    pub log_likelihood: f64,
    /// Log-likelihood at every EM iteration of the selected fit (non-decreasing).
    pub log_likelihood_trace: Vec<f64>,
    /// EM iterations of the selected fit.
    pub iterations: usize,
    /// Restarts attempted.
    pub restarts_attempted: usize,
    /// Restarts that converged to a non-degenerate fit.
    pub restarts_converged: usize,
    /// Bayesian information criterion of the selected fit.
    pub bic: f64,
    /// Mean over units of the largest posterior responsibility.
    pub separation: f64,
    /// Smallest gap between adjacent canonical effects (`0` means not separately identified).
    pub min_effect_gap: f64,
    /// Bootstrap bookkeeping, when requested.
    pub bootstrap: Option<LatentClassBootstrap>,
    /// Identification premises and their status.
    pub premises: Vec<LatentClassPremise>,
    /// Calibration state: unmeasured.
    pub calibration: CalibrationStatus,
    /// [`LATENT_CLASS_INFERENCE_CLAIM`].
    pub inference_claim: &'static str,
    /// [`LATENT_CLASS_CAVEAT`].
    pub caveat: &'static str,
}

fn refuse(code: &'static str, detail: &str, message: &str) -> EstimationError {
    EstimationError::refused(code, format!("{detail}: {message}"))
}

fn count(n: usize) -> f64 {
    f64::from(u32::try_from(n).unwrap_or(u32::MAX))
}

fn as_u64(n: usize) -> u64 {
    u64::try_from(n).unwrap_or(u64::MAX)
}

/// Uniform draw in `[0, 1)` from the top 53 bits.
#[allow(clippy::cast_precision_loss, reason = "a 53-bit integer is exact in f64")]
fn unit_draw(bits: u64) -> f64 {
    (bits >> 11) as f64 / (1_u64 << 53) as f64
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The prepared regression problem: rows `z_i = [1, a_i, x_i...]`.
struct Problem {
    n: usize,
    d: usize,
    z: Vec<f64>,
    y: Vec<f64>,
}

impl Problem {
    fn row(&self, i: usize) -> &[f64] {
        &self.z[i * self.d..(i + 1) * self.d]
    }

    fn resample(&self, indices: &[usize]) -> Self {
        let mut z = Vec::with_capacity(indices.len() * self.d);
        let mut y = Vec::with_capacity(indices.len());
        for &i in indices {
            z.extend_from_slice(self.row(i));
            y.push(self.y[i]);
        }
        Self { n: indices.len(), d: self.d, z, y }
    }
}

/// EM guards derived from the data and the configuration.
#[derive(Clone, Copy)]
struct Limits {
    min_n: f64,
    floor: f64,
}

#[derive(Clone)]
struct Params {
    weights: Vec<f64>,
    beta: Vec<Vec<f64>>,
    sigma2: Vec<f64>,
}

enum EmFailure {
    /// A class collapsed, was rank deficient or hit the variance floor.
    Degenerate,
    NotConverged,
    NotMonotone,
}

struct FitState {
    params: Params,
    resp: Vec<Vec<f64>>,
    ll: f64,
    trace: Vec<f64>,
}

fn validate_config(config: &LatentClassConfig) -> Result<(), EstimationError> {
    let invalid = reason_code!("invalid_argument");
    if config.classes < 2 {
        return Err(refuse(
            reason_code!("population_not_estimable"),
            "latent_class.degenerate_class",
            "a single class is the ordinary regression, not a mixture; declare at least two classes",
        ));
    }
    if config.classes > LATENT_CLASS_MAX_CLASSES {
        return Err(refuse(
            invalid,
            "latent_class.too_many_classes",
            "at most four latent classes are supported",
        ));
    }
    if !config.assume_conditional_randomization {
        return Err(refuse(
            reason_code!("required_option_missing"),
            "latent_class.randomization_not_declared",
            "declare that treatment is as-if randomized within class given the covariates",
        ));
    }
    let bad = config.restarts == 0
        || config.restarts > LATENT_CLASS_MAX_RESTARTS
        || config.max_iterations == 0
        || config.bootstrap_replicates > LATENT_CLASS_MAX_BOOTSTRAP
        || !(config.tolerance.is_finite() && config.tolerance > 0.0)
        || !(config.min_class_weight >= 0.0 && config.min_class_weight <= 0.5)
        || !(config.min_separation >= 0.0 && config.min_separation <= 1.0)
        || !(config.variance_floor.is_finite() && config.variance_floor > 0.0);
    if bad {
        return Err(refuse(
            invalid,
            "latent_class.invalid_config",
            "restarts, iterations, tolerance, weight, separation, floor or bootstrap count is out of range",
        ));
    }
    Ok(())
}

fn cmp_rows(a: &[f64], b: &[f64]) -> Ordering {
    a.iter().zip(b).map(|(x, y)| x.total_cmp(y)).find(|o| o.is_ne()).unwrap_or(Ordering::Equal)
}

fn distinct_design_points(problem: &Problem) -> usize {
    let mut rows: Vec<&[f64]> = (0..problem.n).map(|i| &problem.row(i)[1..]).collect();
    rows.sort_by(|a, b| cmp_rows(a, b));
    let mut distinct = 0;
    let mut previous: Option<&[f64]> = None;
    for row in rows {
        match previous {
            Some(q) if cmp_rows(q, row) == Ordering::Equal => {}
            _ => distinct += 1,
        }
        previous = Some(row);
    }
    distinct
}

fn validate_data(
    data: &LatentClassData<'_>,
    config: &LatentClassConfig,
) -> Result<(Problem, Limits), EstimationError> {
    let invalid = reason_code!("invalid_argument");
    let n = data.outcome.len();
    let p = data.covariates.len();
    if data.treatment.len() != n || data.covariates.iter().any(|c| c.len() != n) {
        return Err(refuse(
            invalid,
            "latent_class.length_mismatch",
            "outcome, treatment and covariate columns must share one length",
        ));
    }
    if p > LATENT_CLASS_MAX_COVARIATES {
        return Err(refuse(
            invalid,
            "latent_class.too_many_covariates",
            "the covariate count exceeds the supported bound",
        ));
    }
    let finite = data.outcome.iter().chain(data.treatment).all(|v| v.is_finite())
        && data.covariates.iter().all(|c| c.iter().all(|v| v.is_finite()));
    if !finite {
        return Err(refuse(invalid, "latent_class.non_finite_input", "inputs must be finite"));
    }
    let d = p + 2;
    if n < config.classes * (d + 1) {
        return Err(refuse(
            invalid,
            "latent_class.too_few_rows",
            "fewer rows than classes times regression dimension plus one",
        ));
    }
    if let Some(labels) = &config.initial_labels {
        if labels.len() != n || labels.iter().any(|&l| l >= config.classes) {
            return Err(refuse(
                invalid,
                "latent_class.invalid_config",
                "initial labels must give one label below the class count per row",
            ));
        }
    }
    let mut z = Vec::with_capacity(n * d);
    for i in 0..n {
        z.push(1.0);
        z.push(data.treatment[i]);
        for column in data.covariates {
            z.push(column[i]);
        }
    }
    let problem = Problem { n, d, z, y: data.outcome.to_vec() };
    let first = data.treatment[0];
    if data.treatment.iter().all(|v| v.total_cmp(&first) == Ordering::Equal) {
        return Err(refuse(
            reason_code!("effect_not_identified"),
            "latent_class.treatment_constant",
            "the treatment does not vary, so no class effect is identified",
        ));
    }
    if distinct_design_points(&problem) < d + config.classes - 1 {
        return Err(refuse(
            reason_code!("design_rank_deficient"),
            "latent_class.design_support_too_small",
            "too few distinct treatment and covariate points to identify the mixture",
        ));
    }
    let mean = problem.y.iter().sum::<f64>() / count(n);
    let variance = problem.y.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / count(n);
    if variance.is_nan() || variance <= 0.0 {
        return Err(refuse(
            reason_code!("population_not_estimable"),
            "latent_class.constant_outcome",
            "the outcome is constant, so no class structure exists",
        ));
    }
    let limits = Limits { min_n: count(d) + 1.0, floor: config.variance_floor * variance };
    Ok((problem, limits))
}

/// Solve `A x = b` for a symmetric positive-definite `A` (lower triangle read) by Cholesky;
/// `None` when a pivot is not safely positive.
fn solve_spd(matrix: &[f64], rhs: &[f64], dim: usize) -> Option<Vec<f64>> {
    let mut lower = vec![0.0; dim * dim];
    for i in 0..dim {
        for j in 0..=i {
            let mut acc = matrix[i * dim + j];
            for k in 0..j {
                acc -= lower[i * dim + k] * lower[j * dim + k];
            }
            if i == j {
                if acc.is_nan() || acc <= 1e-12 * matrix[i * dim + i] || !acc.is_finite() {
                    return None;
                }
                lower[i * dim + i] = acc.sqrt();
            } else {
                lower[i * dim + j] = acc / lower[j * dim + j];
            }
        }
    }
    let mut forward = vec![0.0; dim];
    for i in 0..dim {
        let mut acc = rhs[i];
        for k in 0..i {
            acc -= lower[i * dim + k] * forward[k];
        }
        forward[i] = acc / lower[i * dim + i];
    }
    let mut solution = vec![0.0; dim];
    for i in (0..dim).rev() {
        let mut acc = forward[i];
        for k in (i + 1)..dim {
            acc -= lower[k * dim + i] * solution[k];
        }
        solution[i] = acc / lower[i * dim + i];
    }
    Some(solution)
}

/// Weighted least squares of `y` on the problem's design.
fn wls(problem: &Problem, weights: &[f64]) -> Option<Vec<f64>> {
    let dim = problem.d;
    let mut gram = vec![0.0; dim * dim];
    let mut rhs = vec![0.0; dim];
    for (idx, &weight) in weights.iter().enumerate() {
        if weight <= 0.0 {
            continue;
        }
        let zi = problem.row(idx);
        for row in 0..dim {
            let wz = weight * zi[row];
            rhs[row] += wz * problem.y[idx];
            for col in 0..=row {
                gram[row * dim + col] += wz * zi[col];
            }
        }
    }
    solve_spd(&gram, &rhs, dim)
}

fn e_step(problem: &Problem, params: &Params, resp: &mut [Vec<f64>]) -> f64 {
    let k = params.weights.len();
    let mut logs = vec![0.0; k];
    let mut total = 0.0;
    for i in 0..problem.n {
        let zi = problem.row(i);
        let mut max = f64::NEG_INFINITY;
        for c in 0..k {
            let resid = problem.y[i] - dot(zi, &params.beta[c]);
            let s2 = params.sigma2[c];
            let term = params.weights[c].ln() - 0.5 * ((TWO_PI * s2).ln() + resid * resid / s2);
            logs[c] = term;
            if term > max {
                max = term;
            }
        }
        let sum: f64 = logs.iter().map(|term| (term - max).exp()).sum();
        let lse = max + sum.ln();
        for c in 0..k {
            resp[c][i] = (logs[c] - lse).exp();
        }
        total += lse;
    }
    total
}

fn m_step(problem: &Problem, resp: &[Vec<f64>], limits: Limits) -> Result<Params, EmFailure> {
    let mut params = Params { weights: Vec::new(), beta: Vec::new(), sigma2: Vec::new() };
    for column in resp {
        let n_c: f64 = column.iter().sum();
        if n_c.is_nan() || n_c < limits.min_n {
            return Err(EmFailure::Degenerate);
        }
        let beta = wls(problem, column).ok_or(EmFailure::Degenerate)?;
        let mut sse = 0.0;
        for (i, &w) in column.iter().enumerate() {
            let resid = problem.y[i] - dot(problem.row(i), &beta);
            sse += w * resid * resid;
        }
        let sigma2 = sse / n_c;
        if sigma2.is_nan() || sigma2 < limits.floor {
            return Err(EmFailure::Degenerate);
        }
        params.weights.push(n_c / count(problem.n));
        params.beta.push(beta);
        params.sigma2.push(sigma2);
    }
    Ok(params)
}

fn run_em(
    problem: &Problem,
    init: Vec<Vec<f64>>,
    config: &LatentClassConfig,
    limits: Limits,
) -> Result<FitState, EmFailure> {
    let mut resp = init;
    let mut params = m_step(problem, &resp, limits)?;
    let mut trace: Vec<f64> = Vec::new();
    for _ in 0..config.max_iterations {
        let ll = e_step(problem, &params, &mut resp);
        if !ll.is_finite() {
            return Err(EmFailure::NotConverged);
        }
        let previous = trace.last().copied();
        trace.push(ll);
        if let Some(prev) = previous {
            let scale = 1.0 + prev.abs();
            if ll < prev - MONOTONE_TOLERANCE * scale {
                return Err(EmFailure::NotMonotone);
            }
            if (ll - prev).abs() <= config.tolerance * scale {
                return Ok(FitState { params, resp, ll, trace });
            }
        }
        params = m_step(problem, &resp, limits)?;
    }
    Err(EmFailure::NotConverged)
}

fn hard_responsibilities(assign: &[usize], k: usize) -> Vec<Vec<f64>> {
    let mut resp = vec![vec![0.0; assign.len()]; k];
    for (i, &c) in assign.iter().enumerate() {
        resp[c][i] = 1.0;
    }
    resp
}

/// Seeded deterministic initialization from `k` outcome centers.
fn seeded_init(problem: &Problem, k: usize, restart: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut state = mix64(seed ^ mix64(as_u64(restart).wrapping_add(1)));
    let n_u64 = as_u64(problem.n).max(1);
    let first = usize::try_from(splitmix64(&mut state) % n_u64).unwrap_or(0);
    let mut centers = vec![problem.y[first]];
    while centers.len() < k {
        let dist: Vec<f64> = problem
            .y
            .iter()
            .map(|&v| centers.iter().map(|&c| (v - c) * (v - c)).fold(f64::INFINITY, f64::min))
            .collect();
        let total: f64 = dist.iter().sum();
        let mut pick = 0;
        if restart == 0 || total.is_nan() || total <= 0.0 {
            for (i, &v) in dist.iter().enumerate() {
                if v > dist[pick] {
                    pick = i;
                }
            }
        } else {
            let target = unit_draw(splitmix64(&mut state)) * total;
            let mut cumulative = 0.0;
            pick = dist.len() - 1;
            for (i, &v) in dist.iter().enumerate() {
                cumulative += v;
                if cumulative > target {
                    pick = i;
                    break;
                }
            }
        }
        centers.push(problem.y[pick]);
    }
    let assign: Vec<usize> = problem
        .y
        .iter()
        .map(|&v| {
            let mut best = 0;
            for (c, &center) in centers.iter().enumerate() {
                if (v - center).abs() < (v - centers[best]).abs() {
                    best = c;
                }
            }
            best
        })
        .collect();
    hard_responsibilities(&assign, k)
}

/// Run the restarts and keep the highest converged non-degenerate log-likelihood.
fn best_fit(
    problem: &Problem,
    config: &LatentClassConfig,
    limits: Limits,
) -> Result<(FitState, usize, usize), EstimationError> {
    let attempted = if config.initial_labels.is_some() { 1 } else { config.restarts };
    let mut best: Option<FitState> = None;
    let mut succeeded = 0;
    let mut degenerate = 0;
    let mut non_monotone = 0;
    for restart in 0..attempted {
        let init = match &config.initial_labels {
            Some(labels) => hard_responsibilities(labels, config.classes),
            None => seeded_init(problem, config.classes, restart, config.seed),
        };
        match run_em(problem, init, config, limits) {
            Ok(fit) => {
                succeeded += 1;
                let better = match &best {
                    Some(current) => fit.ll > current.ll,
                    None => true,
                };
                if better {
                    best = Some(fit);
                }
            }
            Err(EmFailure::Degenerate) => degenerate += 1,
            Err(EmFailure::NotMonotone) => non_monotone += 1,
            Err(EmFailure::NotConverged) => {}
        }
    }
    match best {
        Some(fit) => Ok((fit, attempted, succeeded)),
        None if degenerate == attempted => Err(refuse(
            reason_code!("population_not_estimable"),
            "latent_class.degenerate_class",
            "every start collapsed a class: the data show no separable class structure \
             (a single regime, a noise-free fit or a class too small to estimate)",
        )),
        None if non_monotone > 0 => Err(refuse(
            reason_code!("mechanism_fit_not_converged"),
            "latent_class.likelihood_not_monotone",
            "the EM log-likelihood decreased; the numerical fit is unreliable",
        )),
        None => Err(refuse(
            reason_code!("mechanism_fit_not_converged"),
            "latent_class.not_converged",
            "no start converged within the iteration bound; raise the bound or the tolerance",
        )),
    }
}

/// Canonical order: ascending effect, then intercept, then weight.
fn canonical_order(params: &Params) -> Vec<usize> {
    let mut order: Vec<usize> = (0..params.weights.len()).collect();
    order.sort_by(|&a, &b| {
        params.beta[a][1]
            .total_cmp(&params.beta[b][1])
            .then(params.beta[a][0].total_cmp(&params.beta[b][0]))
            .then(params.weights[a].total_cmp(&params.weights[b]))
            .then(a.cmp(&b))
    });
    order
}

fn reorder(params: &Params, order: &[usize]) -> Params {
    Params {
        weights: order.iter().map(|&c| params.weights[c]).collect(),
        beta: order.iter().map(|&c| params.beta[c].clone()).collect(),
        sigma2: order.iter().map(|&c| params.sigma2[c]).collect(),
    }
}

fn permute_into(k: usize, current: &mut Vec<usize>, used: &mut [bool], out: &mut Vec<Vec<usize>>) {
    if current.len() == k {
        out.push(current.clone());
        return;
    }
    for c in 0..k {
        if !used[c] {
            used[c] = true;
            current.push(c);
            permute_into(k, current, used, out);
            current.pop();
            used[c] = false;
        }
    }
}

fn permutations(k: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    permute_into(k, &mut Vec::new(), &mut vec![false; k], &mut out);
    out
}

fn sample_sd(values: &[f64]) -> f64 {
    let m = count(values.len());
    let mean = values.iter().sum::<f64>() / m;
    (values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (m - 1.0)).sqrt()
}

fn mixture_average(params: &Params) -> f64 {
    params.weights.iter().zip(&params.beta).map(|(w, b)| w * b[1]).sum()
}

/// One aligned bootstrap replicate: (effects, weights, average effect).
type Replicate = (Vec<f64>, Vec<f64>, f64);

fn bootstrap_replicate(
    problem: &Problem,
    reference: &Params,
    config: &LatentClassConfig,
    limits: Limits,
    b: usize,
) -> Option<Replicate> {
    let n_u64 = as_u64(problem.n).max(1);
    let mut state = mix64(config.seed.wrapping_add(0xB007_57A9)) ^ mix64(as_u64(b).wrapping_add(1));
    let indices: Vec<usize> = (0..problem.n)
        .map(|_| usize::try_from(splitmix64(&mut state) % n_u64).unwrap_or(0))
        .collect();
    let sample = problem.resample(&indices);
    let k = reference.weights.len();
    let mut resp = vec![vec![0.0; sample.n]; k];
    e_step(&sample, reference, &mut resp);
    let fit = run_em(&sample, resp, config, limits).ok()?;
    let canonical = reorder(&fit.params, &canonical_order(&fit.params));
    let mut best: Option<(f64, Vec<usize>)> = None;
    for perm in permutations(k) {
        let cost: f64 = (0..k)
            .map(|c| {
                let rep = &canonical.beta[perm[c]];
                let reference_beta = &reference.beta[c];
                (rep[1] - reference_beta[1]).powi(2) + (rep[0] - reference_beta[0]).powi(2)
            })
            .sum();
        let lower = match &best {
            Some((current, _)) => cost < *current,
            None => true,
        };
        if lower {
            best = Some((cost, perm));
        }
    }
    let perm = best?.1;
    let effects: Vec<f64> = perm.iter().map(|&c| canonical.beta[c][1]).collect();
    let weights: Vec<f64> = perm.iter().map(|&c| canonical.weights[c]).collect();
    let average = effects.iter().zip(&weights).map(|(t, w)| t * w).sum();
    Some((effects, weights, average))
}

/// Bootstrap standard errors with label alignment.
struct BootstrapSe {
    effects: Vec<f64>,
    weights: Vec<f64>,
    average: f64,
    summary: LatentClassBootstrap,
}

fn bootstrap_se(
    problem: &Problem,
    reference: &Params,
    config: &LatentClassConfig,
    limits: Limits,
) -> Result<Option<BootstrapSe>, EstimationError> {
    let requested = config.bootstrap_replicates;
    if requested == 0 {
        return Ok(None);
    }
    let k = reference.weights.len();
    let replicates: Vec<Replicate> = (0..requested)
        .filter_map(|b| bootstrap_replicate(problem, reference, config, limits, b))
        .collect();
    let succeeded = replicates.len();
    if succeeded < 2 {
        return Err(refuse(
            reason_code!("mechanism_fit_not_converged"),
            "latent_class.not_converged",
            "fewer than two bootstrap replicates converged to a non-degenerate fit",
        ));
    }
    let effects =
        (0..k).map(|c| sample_sd(&replicates.iter().map(|r| r.0[c]).collect::<Vec<_>>())).collect();
    let weights =
        (0..k).map(|c| sample_sd(&replicates.iter().map(|r| r.1[c]).collect::<Vec<_>>())).collect();
    let average = sample_sd(&replicates.iter().map(|r| r.2).collect::<Vec<_>>());
    let summary = LatentClassBootstrap { requested, succeeded, failed: requested - succeeded };
    Ok(Some(BootstrapSe { effects, weights, average, summary }))
}

fn premises() -> Vec<LatentClassPremise> {
    let declared = PremiseStatus::Declared;
    let checked = PremiseStatus::Checked;
    vec![
        LatentClassPremise { name: "conditional_randomization_within_class", status: declared },
        LatentClassPremise { name: "gaussian_linear_class_outcome", status: declared },
        LatentClassPremise { name: "constant_class_weights", status: declared },
        LatentClassPremise { name: "number_of_classes_declared", status: declared },
        LatentClassPremise { name: "treatment_varies_and_design_support", status: checked },
        LatentClassPremise { name: "class_weight_and_separation", status: checked },
        LatentClassPremise { name: "no_degenerate_class", status: checked },
        LatentClassPremise { name: "converged_monotone_likelihood", status: checked },
    ]
}

/// Fit the latent-class mixture of Gaussian linear outcome models and report class effects.
///
/// # Errors
///
/// Refuses with the details documented on the module: invalid configuration or data, an
/// undeclared randomization premise, a constant treatment, too little design support, a
/// degenerate or weak class, or an EM that did not converge or was not monotone.
pub fn fit_latent_class_effects(
    data: &LatentClassData<'_>,
    config: &LatentClassConfig,
) -> Result<LatentClassResult, EstimationError> {
    validate_config(config)?;
    let (problem, limits) = validate_data(data, config)?;
    let (fit, attempted, succeeded) = best_fit(&problem, config, limits)?;
    let order = canonical_order(&fit.params);
    let params = reorder(&fit.params, &order);
    let k = order.len();
    let resp: Vec<Vec<f64>> = order.iter().map(|&c| fit.resp[c].clone()).collect();
    let rows: Vec<Vec<f64>> =
        (0..problem.n).map(|i| (0..k).map(|c| resp[c][i]).collect()).collect();
    let hard_assignment: Vec<usize> = rows
        .iter()
        .map(|row| {
            let mut best = 0;
            for (c, &v) in row.iter().enumerate() {
                if v > row[best] {
                    best = c;
                }
            }
            best
        })
        .collect();
    let separation =
        rows.iter().map(|row| row.iter().copied().fold(0.0_f64, f64::max)).sum::<f64>()
            / count(problem.n);
    check_identification(&params, separation, config)?;
    let boot = bootstrap_se(&problem, &params, config, limits)?;
    let classes: Vec<LatentClassEffect> = (0..k)
        .map(|c| LatentClassEffect {
            index: c,
            raw_index: order[c],
            weight: params.weights[c],
            weight_se: boot.as_ref().map(|b| b.weights[c]),
            intercept: params.beta[c][0],
            effect: params.beta[c][1],
            effect_se: boot.as_ref().map(|b| b.effects[c]),
            covariate_coefficients: params.beta[c][2..].to_vec(),
            residual_variance: params.sigma2[c],
            effective_n: resp[c].iter().sum(),
        })
        .collect();
    let min_effect_gap =
        classes.windows(2).map(|w| w[1].effect - w[0].effect).fold(f64::INFINITY, f64::min);
    let free_parameters = count(k * (problem.d + 1) + k - 1);
    Ok(LatentClassResult {
        mixture_average_effect: mixture_average(&params),
        mixture_average_se: boot.as_ref().map(|b| b.average),
        classes,
        class_order: order,
        responsibilities: rows,
        hard_assignment,
        log_likelihood: fit.ll,
        iterations: fit.trace.len(),
        log_likelihood_trace: fit.trace,
        restarts_attempted: attempted,
        restarts_converged: succeeded,
        bic: -2.0 * fit.ll + free_parameters * count(problem.n).ln(),
        separation,
        min_effect_gap,
        bootstrap: boot.map(|b| b.summary),
        premises: premises(),
        calibration: CalibrationStatus::Unmeasured,
        inference_claim: LATENT_CLASS_INFERENCE_CLAIM,
        caveat: LATENT_CLASS_CAVEAT,
    })
}

fn check_identification(
    params: &Params,
    separation: f64,
    config: &LatentClassConfig,
) -> Result<(), EstimationError> {
    let weak = reason_code!("population_not_estimable");
    if params.weights.iter().any(|&w| w < config.min_class_weight) {
        return Err(refuse(
            weak,
            "latent_class.weak_class",
            "a class weight is below the declared minimum class weight",
        ));
    }
    if separation < config.min_separation {
        return Err(refuse(
            weak,
            "latent_class.weak_class",
            "the classes overlap: the mean largest posterior responsibility is below the \
             declared minimum separation",
        ));
    }
    Ok(())
}
