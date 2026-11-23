//! Exposure-measurement declarations and the per-unit imputation of the latent exposure.
//!
//! A measured exposure is declared on the variable it measures (a misclassified 0/1 record, a
//! per-unit probability, a covariate-dependent detection, repeat codings of a latent facet).
//! The declaration gives every unit a likelihood of what was recorded under each latent level
//! (`P(W | X = j, Y)` for misclassification, the detection function for detection, the
//! response laws for repeat codings, one for supplied probabilities). The level is then drawn
//! from its posterior
//!
//! ```text
//! P(X = j | record, y, z)  proportional to  prior_ij * L_ij * f(y | j, z)
//! ```
//!
//! where the exposure model (a softmax in the covariates, or the supplied probabilities) and
//! the outcome model (Gaussian, or logistic for a 0/1 outcome) are fitted by EM on a row
//! bootstrap of the units. The measurement parameters are redrawn for every draw: Beta draws
//! of declared or validated rates, normal draws of a logistic detection function's parameters
//! with their covariance, bootstrap refits of the latent-class model. The draws are proper
//! multiple imputations when the imputation models contain the relations the analysis relies
//! on; that is an assumption of the route and is part of its record.
//!
//! The draws are completed exposure columns. Each is analysed by the ordinary engine and the
//! per-draw estimates are pooled by [`pool_measured_effect`] with Rubin's rules.
//!
//! A partially declared monotone detection function identifies only bounds on the prevalence
//! and has no per-unit law, so it is not offered here: bounds on an effect would need the
//! effect at every detection function inside the envelope, and the effect of a unit-level
//! exposure is not monotone in the detection function in general.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_stats::measurement_error::{
    BinaryErrorRates, LatentClassConstraints, LatentClassData, LatentClassFit, LatentClassSettings,
    fit_latent_class,
};

use super::{MeasurementError, check_cancelled, digest, stats_error};
use crate::multiple_imputation::{RubinPooled, rubin_pool};

const TINY: f64 = 1e-12;
const EM_ITERATIONS: usize = 80;
const EM_TOLERANCE: f64 = 1e-7;

fn invalid(message: impl Into<String>) -> MeasurementError {
    MeasurementError::new(
        antecedent_core::reason_code!("invalid_argument"),
        antecedent_core::reason_details::measurement_model::INPUT_INVALID,
        message,
    )
}

// ---------------------------------------------------------------------------------------------
// Declarations
// ---------------------------------------------------------------------------------------------

/// A sensitivity and specificity, declared (with the variance of each estimate) or estimated
/// from a validation sample whose Beta posterior is redrawn for every imputation.
#[derive(Clone, Debug, PartialEq)]
pub struct RatePair {
    /// The point values and their declared variances.
    pub rates: BinaryErrorRates,
    /// `(true_positive, false_negative, true_negative, false_positive)` of the validation
    /// sample the rates were estimated from, when there was one.
    pub validation: Option<(u64, u64, u64, u64)>,
}

impl RatePair {
    /// Rates declared with the variances of their estimates (zero for exact).
    ///
    /// # Errors
    /// Rates outside `[0, 1]` or invalid variances.
    pub fn declared(
        sensitivity: f64,
        specificity: f64,
        sensitivity_variance: f64,
        specificity_variance: f64,
    ) -> Result<Self, MeasurementError> {
        let rates = BinaryErrorRates::new(
            sensitivity,
            specificity,
            sensitivity_variance,
            specificity_variance,
        )
        .map_err(|error| stats_error(&error))?;
        Ok(Self { rates, validation: None })
    }

    /// Rates estimated from a validation sample against a reference standard.
    ///
    /// # Errors
    /// A sample with an empty margin.
    pub fn validated(
        true_positive: u64,
        false_negative: u64,
        true_negative: u64,
        false_positive: u64,
    ) -> Result<Self, MeasurementError> {
        let rates = BinaryErrorRates::from_validation_counts(
            true_positive,
            false_negative,
            true_negative,
            false_positive,
        )
        .map_err(|error| stats_error(&error))?;
        Ok(Self {
            rates,
            validation: Some((true_positive, false_negative, true_negative, false_positive)),
        })
    }

    fn youden(&self) -> f64 {
        self.rates.sensitivity + self.rates.specificity - 1.0
    }

    fn draw(&self, stream: &mut Stream) -> (f64, f64) {
        if let Some((tp, fnn, tn, fp)) = self.validation {
            (
                stream.beta(tp as f64 + 0.5, fnn as f64 + 0.5),
                stream.beta(tn as f64 + 0.5, fp as f64 + 0.5),
            )
        } else {
            (
                stream.rate(self.rates.sensitivity, self.rates.sensitivity_variance),
                stream.rate(self.rates.specificity, self.rates.specificity_variance),
            )
        }
    }
}

/// How a misclassified exposure's error depends on the outcome.
#[derive(Clone, Debug, PartialEq)]
pub enum ExposureErrorDeclaration {
    /// One pair for both outcome levels.
    Shared(RatePair),
    /// A pair per outcome level; the outcome must be 0/1 (cases are the units with outcome 1).
    ByOutcome {
        /// Rates among units with the outcome.
        cases: RatePair,
        /// Rates among units without the outcome.
        controls: RatePair,
    },
}

/// A fully declared detection function of a covariate.
#[derive(Clone, Debug, PartialEq)]
pub enum DetectionFunctionDeclaration {
    /// `expit(intercept + slope * c)` with the parameter estimates' covariance.
    Logistic {
        /// Intercept.
        intercept: f64,
        /// Slope.
        slope: f64,
        /// Variance of the intercept estimate.
        intercept_variance: f64,
        /// Covariance of the two estimates.
        covariance: f64,
        /// Variance of the slope estimate.
        slope_variance: f64,
    },
    /// Linear interpolation between declared `(knot, value)` points.
    Table {
        /// Increasing covariate knots.
        knots: Vec<f64>,
        /// Detection probability at each knot.
        values: Vec<f64>,
    },
}

/// An exposure measurement declared on the exposure (or facet) variable.
#[derive(Clone, Debug, PartialEq)]
pub enum ExposureMeasurement {
    /// A recorded 0/1 exposure with a declared or validated error structure.
    Misclassified {
        /// The recorded exposure of every unit.
        recorded: Vec<bool>,
        /// Sensitivity and specificity, shared or by outcome level.
        error: ExposureErrorDeclaration,
        /// Smallest Youden index that identifies the exposure.
        min_youden: f64,
    },
    /// Per-unit probabilities that the exposure is present, calibrated given the covariates.
    Soft {
        /// The probability of every unit.
        probability: Vec<f64>,
    },
    /// A recorded 0/1 detection whose probability is a declared function of a covariate.
    Detected {
        /// The recorded detection of every unit.
        detected: Vec<bool>,
        /// The covariate that drives detection.
        covariate: Vec<f64>,
        /// The detection function.
        function: DetectionFunctionDeclaration,
        /// Probability of a detection when the exposure is absent.
        false_positive_rate: f64,
        /// Variance of the false-positive rate's estimate.
        false_positive_variance: f64,
        /// Smallest mean detection net of false positives that identifies the exposure.
        min_net_detection: f64,
    },
    /// Repeat codings of one latent binary or ordinal facet.
    RepeatCoded {
        /// Category index of every coding of every unit, row-major `units x coding_count`.
        codings: Vec<u8>,
        /// Codings per unit.
        coding_count: usize,
        /// Categories per coding.
        categories: usize,
        /// Latent levels.
        latent_levels: usize,
        /// Every coding has one response law given the level.
        exchangeable: bool,
        /// Declared response law of the lowest level.
        lowest_level: Option<Vec<f64>>,
        /// Starting points of the latent-class fit.
        starts: usize,
    },
}

impl ExposureMeasurement {
    /// Stable kind label.
    #[must_use]
    pub const fn kind(&self) -> &'static str {
        match self {
            Self::Misclassified { .. } => "misclassified_exposure",
            Self::Soft { .. } => "soft_exposure",
            Self::Detected { .. } => "detected_exposure",
            Self::RepeatCoded { .. } => "repeat_coded_exposure",
        }
    }

    /// Latent levels of the exposure.
    #[must_use]
    pub const fn levels(&self) -> usize {
        match self {
            Self::RepeatCoded { latent_levels, .. } => *latent_levels,
            _ => 2,
        }
    }

    /// Units the declaration covers.
    #[must_use]
    pub fn units(&self) -> usize {
        match self {
            Self::Misclassified { recorded, .. } => recorded.len(),
            Self::Soft { probability } => probability.len(),
            Self::Detected { detected, .. } => detected.len(),
            Self::RepeatCoded { codings, coding_count, .. } => {
                codings.len() / (*coding_count).max(1)
            }
        }
    }

    /// Digest of the declaration (its kind, error structure and every recorded value).
    #[must_use]
    #[allow(clippy::too_many_lines, reason = "one declaration resolved into per-unit likelihoods")]
    pub fn identity(&self) -> String {
        let flags = |v: &[bool]| v.iter().map(|&b| f64::from(u8::from(b))).collect::<Vec<f64>>();
        let pair = |p: &RatePair| {
            let mut out = vec![
                p.rates.sensitivity,
                p.rates.specificity,
                p.rates.sensitivity_variance,
                p.rates.specificity_variance,
            ];
            if let Some((a, b, c, d)) = p.validation {
                out.extend([a as f64, b as f64, c as f64, d as f64]);
            }
            out
        };
        let kind: Vec<f64> = self.kind().bytes().map(f64::from).collect();
        let digest = match self {
            Self::Misclassified { recorded, error, min_youden } => {
                let rates = match error {
                    ExposureErrorDeclaration::Shared(p) => pair(p),
                    ExposureErrorDeclaration::ByOutcome { cases, controls } => {
                        let mut v = pair(cases);
                        v.push(f64::NAN);
                        v.extend(pair(controls));
                        v
                    }
                };
                let rates: Vec<f64> = rates.into_iter().map(canonical).collect();
                digest(&[
                    ("kind", kind.as_slice()),
                    ("recorded", flags(recorded).as_slice()),
                    ("rates", rates.as_slice()),
                    ("min_youden", &[*min_youden]),
                ])
            }
            Self::Soft { probability } => {
                digest(&[("kind", kind.as_slice()), ("probability", probability.as_slice())])
            }
            Self::Detected {
                detected,
                covariate,
                function,
                false_positive_rate,
                false_positive_variance,
                min_net_detection,
            } => {
                let function = match function {
                    DetectionFunctionDeclaration::Logistic {
                        intercept,
                        slope,
                        intercept_variance,
                        covariance,
                        slope_variance,
                    } => vec![
                        1.0,
                        *intercept,
                        *slope,
                        *intercept_variance,
                        *covariance,
                        *slope_variance,
                    ],
                    DetectionFunctionDeclaration::Table { knots, values } => {
                        let mut v = vec![2.0];
                        v.extend(knots);
                        v.push(f64::INFINITY);
                        v.extend(values);
                        v
                    }
                };
                digest(&[
                    ("kind", kind.as_slice()),
                    ("detected", flags(detected).as_slice()),
                    ("covariate", covariate.as_slice()),
                    ("function", function.as_slice()),
                    (
                        "false_positive",
                        &[*false_positive_rate, *false_positive_variance, *min_net_detection],
                    ),
                ])
            }
            Self::RepeatCoded {
                codings,
                coding_count,
                categories,
                latent_levels,
                exchangeable,
                lowest_level,
                starts,
            } => {
                let codings: Vec<f64> = codings.iter().map(|&c| f64::from(c)).collect();
                let shape = [
                    *coding_count as f64,
                    *categories as f64,
                    *latent_levels as f64,
                    f64::from(u8::from(*exchangeable)),
                    *starts as f64,
                ];
                digest(&[
                    ("kind", kind.as_slice()),
                    ("codings", codings.as_slice()),
                    ("shape", shape.as_slice()),
                    ("lowest_level", lowest_level.as_deref().unwrap_or(&[])),
                ])
            }
        };
        digest.to_hex()
    }
}

fn canonical(v: f64) -> f64 {
    if v.is_nan() { -1.0 } else { v }
}

// ---------------------------------------------------------------------------------------------
// Random numbers
// ---------------------------------------------------------------------------------------------

/// `SplitMix64` stream with the draws the imputation needs.
struct Stream(u64);

impl Stream {
    fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (u, v) = (self.next().max(1e-300), self.next());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a uniform draw in [0, 1) scaled by n is a nonnegative index below n"
    )]
    fn below(&mut self, n: usize) -> usize {
        ((self.next() * n as f64) as usize).min(n.saturating_sub(1))
    }

    fn gamma(&mut self, shape: f64) -> f64 {
        if shape < 1.0 {
            let u = self.next().max(1e-300);
            return self.gamma(shape + 1.0) * u.powf(1.0 / shape);
        }
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.normal();
            let v = 1.0 + c * x;
            if v <= 0.0 {
                continue;
            }
            let v = v * v * v;
            let u = self.next().max(1e-300);
            if u.ln() < 0.5 * x * x + d - d * v + d * v.ln() {
                return d * v;
            }
        }
    }

    fn beta(&mut self, a: f64, b: f64) -> f64 {
        let x = self.gamma(a);
        let y = self.gamma(b);
        x / (x + y)
    }

    /// A rate with the stated mean and variance (the mean itself when the variance is zero or
    /// cannot be matched by a Beta law).
    fn rate(&mut self, mean: f64, variance: f64) -> f64 {
        let cap = mean * (1.0 - mean);
        if variance <= 0.0 || cap <= 0.0 || variance >= cap {
            return mean;
        }
        let k = cap / variance - 1.0;
        self.beta((mean * k).max(1e-3), ((1.0 - mean) * k).max(1e-3))
    }
}

// ---------------------------------------------------------------------------------------------
// Small linear algebra
// ---------------------------------------------------------------------------------------------

/// Solve `a x = b` for a small dense row-major `k x k` system by Gaussian elimination with
/// partial pivoting.
fn solve(a: &[f64], b: &[f64], k: usize) -> Option<Vec<f64>> {
    let mut m = a.to_vec();
    let mut x = b.to_vec();
    for col in 0..k {
        let mut pivot = col;
        for row in col + 1..k {
            if m[row * k + col].abs() > m[pivot * k + col].abs() {
                pivot = row;
            }
        }
        if m[pivot * k + col].abs() < 1e-14 {
            return None;
        }
        if pivot != col {
            for c in 0..k {
                m.swap(col * k + c, pivot * k + c);
            }
            x.swap(col, pivot);
        }
        for row in col + 1..k {
            let factor = m[row * k + col] / m[col * k + col];
            if factor != 0.0 {
                for c in col..k {
                    m[row * k + c] -= factor * m[col * k + c];
                }
                x[row] -= factor * x[col];
            }
        }
    }
    for col in (0..k).rev() {
        let mut s = x[col];
        for c in col + 1..k {
            s -= m[col * k + c] * x[c];
        }
        x[col] = s / m[col * k + col];
    }
    Some(x)
}

fn softmax_row(eta: &mut [f64]) {
    let max = eta.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0;
    for v in eta.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in eta.iter_mut() {
        *v /= sum;
    }
}

// ---------------------------------------------------------------------------------------------
// Imputation models
// ---------------------------------------------------------------------------------------------

/// Softmax regression of fractional level responsibilities on `x` (`n x p`, row-major); `phi`
/// is `p x (levels - 1)` row-major and `levels` the number of latent levels.
fn fit_exposure_model(
    x: &[f64],
    p: usize,
    r: &[f64],
    levels: usize,
    phi: &mut [f64],
    steps: usize,
) {
    let n = x.len() / p;
    let k = levels - 1;
    let dim = p * k;
    for _ in 0..steps {
        let mut gradient = vec![0.0; dim];
        let mut hessian = vec![0.0; dim * dim];
        let mut pi = vec![0.0; levels];
        for i in 0..n {
            let xi = &x[i * p..(i + 1) * p];
            pi[0] = 0.0;
            for l in 0..k {
                pi[l + 1] = (0..p).map(|a| xi[a] * phi[a * k + l]).sum();
            }
            softmax_row(&mut pi);
            for a in 0..p {
                for l in 0..k {
                    gradient[a * k + l] += xi[a] * (r[i * levels + l + 1] - pi[l + 1]);
                    for b in 0..p {
                        for m in 0..k {
                            let diagonal = if l == m { pi[l + 1] } else { 0.0 };
                            let w = diagonal - pi[l + 1] * pi[m + 1];
                            hessian[(a * k + l) * dim + b * k + m] += xi[a] * xi[b] * w;
                        }
                    }
                }
            }
        }
        for d in 0..dim {
            hessian[d * dim + d] += 1e-6;
        }
        if let Some(step) = solve(&hessian, &gradient, dim) {
            for (value, delta) in phi.iter_mut().zip(step) {
                *value += delta;
            }
        } else {
            return;
        }
    }
}

/// `E[y | level j, z] = a + b j + g z`: Gaussian, or logistic for a 0/1 outcome.
struct OutcomeModel {
    binary: bool,
    beta: Vec<f64>,
    variance: f64,
}

impl OutcomeModel {
    fn row(z: &[f64], level: usize) -> Vec<f64> {
        let mut row = Vec::with_capacity(2 + z.len());
        row.push(1.0);
        row.push(level as f64);
        row.extend_from_slice(z);
        row
    }

    fn fit(&mut self, y: &[f64], z: &[f64], pz: usize, r: &[f64], levels: usize, steps: usize) {
        let n = y.len();
        let q = 2 + pz;
        let zi = |i: usize| &z[i * pz..(i + 1) * pz];
        if self.binary {
            for _ in 0..steps {
                let mut h = vec![0.0; q * q];
                let mut g = vec![0.0; q];
                for i in 0..n {
                    for j in 0..levels {
                        let w = r[i * levels + j];
                        if w < 1e-12 {
                            continue;
                        }
                        let x = Self::row(zi(i), j);
                        let eta: f64 = x.iter().zip(&self.beta).map(|(a, b)| a * b).sum();
                        let mu = 1.0 / (1.0 + (-eta.clamp(-30.0, 30.0)).exp());
                        let c = w * mu * (1.0 - mu);
                        for a in 0..q {
                            g[a] += w * (y[i] - mu) * x[a];
                            for b in 0..q {
                                h[a * q + b] += c * x[a] * x[b];
                            }
                        }
                    }
                }
                for d in 0..q {
                    h[d * q + d] += 1e-6;
                }
                if let Some(step) = solve(&h, &g, q) {
                    for (b, s) in self.beta.iter_mut().zip(step) {
                        *b += s;
                    }
                } else {
                    return;
                }
            }
            return;
        }
        let mut gram = vec![0.0; q * q];
        let mut rhs = vec![0.0; q];
        for i in 0..n {
            for j in 0..levels {
                let w = r[i * levels + j];
                let x = Self::row(zi(i), j);
                for a in 0..q {
                    rhs[a] += w * x[a] * y[i];
                    for b in 0..q {
                        gram[a * q + b] += w * x[a] * x[b];
                    }
                }
            }
        }
        for d in 0..q {
            gram[d * q + d] += 1e-9;
        }
        let Some(beta) = solve(&gram, &rhs, q) else { return };
        self.beta = beta;
        let (mut num, mut den) = (0.0, 0.0);
        for i in 0..n {
            for j in 0..levels {
                let w = r[i * levels + j];
                let x = Self::row(zi(i), j);
                let mu: f64 = x.iter().zip(&self.beta).map(|(a, b)| a * b).sum();
                num += w * (y[i] - mu).powi(2);
                den += w;
            }
        }
        let mean = y.iter().sum::<f64>() / n as f64;
        let var_y = y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
        self.variance = (num / den.max(TINY)).max(1e-8 * var_y.max(1e-12));
    }

    fn log_density(&self, y: &[f64], z: &[f64], pz: usize, levels: usize) -> Vec<f64> {
        let n = y.len();
        let mut out = vec![0.0; n * levels];
        for i in 0..n {
            for j in 0..levels {
                let x = Self::row(&z[i * pz..(i + 1) * pz], j);
                let eta: f64 = x.iter().zip(&self.beta).map(|(a, b)| a * b).sum();
                out[i * levels + j] = if self.binary {
                    // y log(mu) + (1 - y) log(1 - mu), stably.
                    let softplus = |t: f64| t.max(0.0) + (-t.abs()).exp().ln_1p();
                    -y[i] * softplus(-eta) - (1.0 - y[i]) * softplus(eta)
                } else {
                    -0.5 * (y[i] - eta).powi(2) / self.variance - 0.5 * self.variance.ln()
                };
            }
        }
        out
    }
}

/// The columns an imputation conditions on besides the record: the outcome and the covariates
/// (column-major, `covariate_count` columns).
#[derive(Clone, Copy, Debug)]
pub struct ImputationData<'a> {
    /// The outcome of every unit.
    pub outcome: &'a [f64],
    /// Covariates, column-major (`covariate_count` columns of `units` values).
    pub covariates: &'a [f64],
    /// Number of covariate columns.
    pub covariate_count: usize,
}

fn standardised_rows(data: &ImputationData<'_>) -> Vec<f64> {
    let n = data.outcome.len();
    let p = data.covariate_count;
    let mut out = vec![0.0; n * p];
    for c in 0..p {
        let col = &data.covariates[c * n..(c + 1) * n];
        let mean = col.iter().sum::<f64>() / n as f64;
        let var = col.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
        let sd = if var.sqrt() < 1e-12 { 1.0 } else { var.sqrt() };
        for i in 0..n {
            out[i * p + c] = (col[i] - mean) / sd;
        }
    }
    out
}

/// Posterior of every unit's latent level, from models fitted by EM on `rows`.
#[allow(
    clippy::too_many_arguments,
    reason = "the likelihood, the conditioning columns, the fixed prior and the fit rows are the inputs of one posterior"
)]
#[allow(clippy::float_cmp, reason = "a 0/1 outcome is detected by exact comparison")]
fn posterior(
    likelihood: &[f64],
    y: &[f64],
    z: &[f64],
    pz: usize,
    levels: usize,
    fixed_prior: Option<&[f64]>,
    rows: &[usize],
) -> Vec<f64> {
    let n = y.len();
    let p = 1 + pz;
    let design = |i: usize| {
        let mut row = Vec::with_capacity(p);
        row.push(1.0);
        row.extend_from_slice(&z[i * pz..(i + 1) * pz]);
        row
    };
    let (mut xb, mut yb, mut zb, mut lb) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut fb = Vec::new();
    for &i in rows {
        xb.extend(design(i));
        yb.push(y[i]);
        zb.extend_from_slice(&z[i * pz..(i + 1) * pz]);
        lb.extend_from_slice(&likelihood[i * levels..(i + 1) * levels]);
        if let Some(prior) = fixed_prior {
            fb.extend_from_slice(&prior[i * levels..(i + 1) * levels]);
        }
    }
    let m = rows.len();
    let binary = y.iter().all(|&v| v == 0.0 || v == 1.0);
    let mut model = OutcomeModel { binary, beta: vec![0.0; 2 + pz], variance: 1.0 };
    let mut phi = vec![0.0; p * (levels - 1)];
    let mut r = vec![0.0; m * levels];
    for i in 0..m {
        let mut sum = 0.0;
        for j in 0..levels {
            let prior = if fixed_prior.is_some() { fb[i * levels + j] } else { 1.0 };
            r[i * levels + j] = lb[i * levels + j] * prior;
            sum += r[i * levels + j];
        }
        for j in 0..levels {
            r[i * levels + j] /= sum.max(TINY);
        }
    }
    let prior_of = |phi: &[f64], x: &[f64], count: usize| {
        let k = levels - 1;
        let mut out = vec![0.0; count * levels];
        for i in 0..count {
            let row = &mut out[i * levels..(i + 1) * levels];
            row[0] = 0.0;
            for l in 0..k {
                row[l + 1] = (0..p).map(|a| x[i * p + a] * phi[a * k + l]).sum();
            }
            softmax_row(row);
        }
        out
    };
    for _ in 0..EM_ITERATIONS {
        let prior_b = if fixed_prior.is_some() {
            fb.clone()
        } else {
            fit_exposure_model(&xb, p, &r, levels, &mut phi, 3);
            prior_of(&phi, &xb, m)
        };
        model.fit(&yb, &zb, pz, &r, levels, 3);
        let logf = model.log_density(&yb, &zb, pz, levels);
        let mut change = 0.0_f64;
        for i in 0..m {
            let row = &logf[i * levels..(i + 1) * levels];
            let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut sum = 0.0;
            let mut next = vec![0.0; levels];
            for j in 0..levels {
                next[j] = prior_b[i * levels + j] * lb[i * levels + j] * (row[j] - max).exp();
                sum += next[j];
            }
            for j in 0..levels {
                let value = next[j] / sum.max(TINY);
                change = change.max((value - r[i * levels + j]).abs());
                r[i * levels + j] = value;
            }
        }
        if change < EM_TOLERANCE {
            break;
        }
    }
    let mut x_all = Vec::with_capacity(n * p);
    for i in 0..n {
        x_all.extend(design(i));
    }
    let prior = fixed_prior.map_or_else(|| prior_of(&phi, &x_all, n), <[f64]>::to_vec);
    let logf = model.log_density(y, z, pz, levels);
    let mut out = vec![0.0; n * levels];
    for i in 0..n {
        let row = &logf[i * levels..(i + 1) * levels];
        let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0;
        for j in 0..levels {
            out[i * levels + j] =
                prior[i * levels + j] * likelihood[i * levels + j] * (row[j] - max).exp();
            sum += out[i * levels + j];
        }
        for j in 0..levels {
            out[i * levels + j] /= sum.max(TINY);
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Likelihood of the record under each level
// ---------------------------------------------------------------------------------------------

struct Prepared<'a> {
    declaration: &'a ExposureMeasurement,
    levels: usize,
    fixed_prior: Option<Vec<f64>>,
    point_fit: Option<LatentClassFit>,
}

fn detection_probability(
    function: &DetectionFunctionDeclaration,
    parameters: (f64, f64),
    c: f64,
) -> f64 {
    let d = match function {
        DetectionFunctionDeclaration::Logistic { .. } => {
            1.0 / (1.0 + (-(parameters.0 + parameters.1 * c)).exp())
        }
        DetectionFunctionDeclaration::Table { knots, values } => {
            let last = knots.len() - 1;
            if c <= knots[0] {
                values[0]
            } else if c >= knots[last] {
                values[last]
            } else {
                let hi = knots.partition_point(|&k| k <= c);
                let lo = hi - 1;
                let t = (c - knots[lo]) / (knots[hi] - knots[lo]);
                values[lo] + t * (values[hi] - values[lo])
            }
        }
    };
    d.clamp(1e-9, 1.0 - 1e-9)
}

fn latent_likelihood(fit: &LatentClassFit, codings: &[u8], coding_count: usize) -> Vec<f64> {
    let levels = fit.latent_levels;
    let n = codings.len() / coding_count;
    let mut out = vec![0.0; n * levels];
    for i in 0..n {
        let mut logs = vec![0.0; levels];
        for (j, log) in logs.iter_mut().enumerate() {
            for k in 0..coding_count {
                let category = usize::from(codings[i * coding_count + k]);
                *log += fit.response_of(k, j, category).max(TINY).ln();
            }
        }
        let max = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        for j in 0..levels {
            out[i * levels + j] = (logs[j] - max).exp();
        }
    }
    out
}

impl<'a> Prepared<'a> {
    #[allow(clippy::too_many_lines, reason = "one EM fit of the exposure and outcome models")]
    #[allow(clippy::float_cmp, reason = "a 0/1 outcome is detected by exact comparison")]
    fn new(declaration: &'a ExposureMeasurement, y: &[f64]) -> Result<Self, MeasurementError> {
        let n = y.len();
        if declaration.units() != n {
            return Err(invalid(
                "the declaration and the outcome cover different numbers of units",
            ));
        }
        let binary_outcome = y.iter().all(|&v| v == 0.0 || v == 1.0);
        let mut fixed_prior = None;
        let mut point_fit = None;
        match declaration {
            ExposureMeasurement::Misclassified { error, min_youden, .. } => {
                if !(min_youden.is_finite() && *min_youden > 0.0 && *min_youden < 1.0) {
                    return Err(invalid("min_youden must lie in (0, 1)"));
                }
                let pairs: Vec<&RatePair> = match error {
                    ExposureErrorDeclaration::Shared(p) => vec![p],
                    ExposureErrorDeclaration::ByOutcome { cases, controls } => {
                        if !binary_outcome {
                            return Err(MeasurementError::new(
                                antecedent_core::reason_code!("route_not_supported"),
                                antecedent_core::reason_details::measurement_model::BINARY_OUTCOME_CONTINUOUS_ERROR,
                                "an error declared by outcome level needs a 0/1 outcome; declare one shared error or analyse a 0/1 outcome",
                            ));
                        }
                        vec![cases, controls]
                    }
                };
                for pair in pairs {
                    if pair.youden() <= *min_youden {
                        return Err(MeasurementError::new(
                            antecedent_core::reason_code!("measurement_not_identified"),
                            antecedent_core::reason_details::measurement_model::YOUDEN_TOO_SMALL,
                            format!(
                                "Youden index {} is not above the declared minimum {min_youden}",
                                pair.youden()
                            ),
                        ));
                    }
                }
            }
            ExposureMeasurement::Soft { probability } => {
                if probability.iter().any(|q| !(q.is_finite() && (0.0..=1.0).contains(q))) {
                    return Err(invalid("exposure probabilities must lie in [0, 1]"));
                }
                let mut prior = Vec::with_capacity(2 * n);
                for q in probability {
                    let q = q.clamp(1e-9, 1.0 - 1e-9);
                    prior.extend([1.0 - q, q]);
                }
                fixed_prior = Some(prior);
            }
            ExposureMeasurement::Detected {
                covariate,
                function,
                false_positive_rate,
                false_positive_variance,
                min_net_detection,
                ..
            } => {
                if covariate.len() != n
                    || !(0.0..1.0).contains(false_positive_rate)
                    || !(false_positive_variance.is_finite() && *false_positive_variance >= 0.0)
                {
                    return Err(invalid(
                        "the detection covariate or false-positive rate is invalid",
                    ));
                }
                if let DetectionFunctionDeclaration::Table { knots, values } = function {
                    let ordered = knots.windows(2).all(|w| w[0] < w[1]);
                    if knots.len() < 2
                        || knots.len() != values.len()
                        || !ordered
                        || values.iter().any(|v| !(0.0..=1.0).contains(v))
                    {
                        return Err(invalid(
                            "a detection table needs increasing knots and probabilities in [0, 1]",
                        ));
                    }
                }
                let parameters = match function {
                    DetectionFunctionDeclaration::Logistic { intercept, slope, .. } => {
                        (*intercept, *slope)
                    }
                    DetectionFunctionDeclaration::Table { .. } => (0.0, 0.0),
                };
                let mean = covariate
                    .iter()
                    .map(|&c| detection_probability(function, parameters, c))
                    .sum::<f64>()
                    / n as f64;
                let net = mean - false_positive_rate;
                if net <= *min_net_detection {
                    return Err(MeasurementError::new(
                        antecedent_core::reason_code!("measurement_not_identified"),
                        antecedent_core::reason_details::measurement_model::YOUDEN_TOO_SMALL,
                        format!(
                            "mean detection net of false positives {net} is not above the declared minimum {min_net_detection}"
                        ),
                    ));
                }
            }
            ExposureMeasurement::RepeatCoded {
                codings,
                coding_count,
                categories,
                latent_levels,
                exchangeable,
                lowest_level,
                starts,
            } => {
                let fit = fit_latent(
                    codings,
                    *coding_count,
                    *categories,
                    *latent_levels,
                    *exchangeable,
                    lowest_level.as_deref(),
                    *starts,
                    1,
                )?;
                point_fit = Some(fit);
            }
        }
        Ok(Self { declaration, levels: declaration.levels(), fixed_prior, point_fit })
    }

    /// The likelihood of every unit's record under each level: at the point parameters when
    /// `draw` is `None`, otherwise at parameters redrawn from their uncertainty (`rows` are the
    /// bootstrap units a latent-class model is refitted on).
    #[allow(
        clippy::too_many_lines,
        reason = "one imputation draw: refit, redraw and impute in sequence"
    )]
    #[allow(
        clippy::float_cmp,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "0/1 outcomes compared exactly; a uniform draw scaled to a positive seed"
    )]
    fn likelihood(
        &self,
        y: &[f64],
        draw: Option<(&mut Stream, &[usize])>,
    ) -> Result<Vec<f64>, MeasurementError> {
        let n = y.len();
        match self.declaration {
            ExposureMeasurement::Misclassified { recorded, error, min_youden } => {
                let (mut stream, _) = match draw {
                    Some((s, r)) => (Some(s), r),
                    None => (None, &[][..]),
                };
                let mut pick = |pair: &RatePair| -> (f64, f64) {
                    match stream.as_deref_mut() {
                        None => (pair.rates.sensitivity, pair.rates.specificity),
                        Some(s) => {
                            for _ in 0..20 {
                                let (se, sp) = pair.draw(s);
                                if se + sp - 1.0 > *min_youden {
                                    return (se, sp);
                                }
                            }
                            (pair.rates.sensitivity, pair.rates.specificity)
                        }
                    }
                };
                let (case_rates, control_rates) = match error {
                    ExposureErrorDeclaration::Shared(pair) => {
                        let both = pick(pair);
                        (both, both)
                    }
                    ExposureErrorDeclaration::ByOutcome { cases, controls } => {
                        (pick(cases), pick(controls))
                    }
                };
                let mut out = vec![0.0; 2 * n];
                for i in 0..n {
                    let (se, sp) = if y[i] == 1.0 { case_rates } else { control_rates };
                    if recorded[i] {
                        out[2 * i] = 1.0 - sp;
                        out[2 * i + 1] = se;
                    } else {
                        out[2 * i] = sp;
                        out[2 * i + 1] = 1.0 - se;
                    }
                }
                Ok(out)
            }
            ExposureMeasurement::Soft { .. } => Ok(vec![1.0; 2 * n]),
            ExposureMeasurement::Detected {
                detected,
                covariate,
                function,
                false_positive_rate,
                false_positive_variance,
                ..
            } => {
                let (parameters, f) = match draw {
                    None => (
                        match function {
                            DetectionFunctionDeclaration::Logistic { intercept, slope, .. } => {
                                (*intercept, *slope)
                            }
                            DetectionFunctionDeclaration::Table { .. } => (0.0, 0.0),
                        },
                        *false_positive_rate,
                    ),
                    Some((stream, _)) => {
                        let parameters = match function {
                            DetectionFunctionDeclaration::Logistic {
                                intercept,
                                slope,
                                intercept_variance,
                                covariance,
                                slope_variance,
                            } => draw_logistic(
                                stream,
                                (*intercept, *slope),
                                (*intercept_variance, *covariance, *slope_variance),
                            ),
                            DetectionFunctionDeclaration::Table { .. } => (0.0, 0.0),
                        };
                        let f = if *false_positive_rate > 0.0 {
                            stream.rate(*false_positive_rate, *false_positive_variance)
                        } else {
                            0.0
                        };
                        (parameters, f)
                    }
                };
                let mut out = vec![0.0; 2 * n];
                for i in 0..n {
                    let d = detection_probability(function, parameters, covariate[i]);
                    if detected[i] {
                        out[2 * i] = f;
                        out[2 * i + 1] = d;
                    } else {
                        out[2 * i] = 1.0 - f;
                        out[2 * i + 1] = 1.0 - d;
                    }
                }
                Ok(out)
            }
            ExposureMeasurement::RepeatCoded {
                codings,
                coding_count,
                categories,
                latent_levels,
                exchangeable,
                lowest_level,
                starts,
            } => match draw {
                None => {
                    let fit = self.point_fit.as_ref().ok_or_else(|| invalid("no latent fit"))?;
                    Ok(latent_likelihood(fit, codings, *coding_count))
                }
                Some((stream, rows)) => {
                    let mut resampled = Vec::with_capacity(rows.len() * coding_count);
                    for &i in rows {
                        resampled
                            .extend_from_slice(&codings[i * coding_count..(i + 1) * coding_count]);
                    }
                    for attempt in 0..5_u64 {
                        let seed = (stream.next() * 4.0e9) as u64 + 1 + attempt;
                        if let Ok(fit) = fit_latent(
                            &resampled,
                            *coding_count,
                            *categories,
                            *latent_levels,
                            *exchangeable,
                            lowest_level.as_deref(),
                            *starts,
                            seed,
                        ) {
                            return Ok(latent_likelihood(&fit, codings, *coding_count));
                        }
                    }
                    Err(MeasurementError::new(
                        antecedent_core::reason_code!("measurement_not_identified"),
                        antecedent_core::reason_details::measurement_model::BOOTSTRAP_UNSTABLE,
                        "the latent-class model could not be refitted on resampled units",
                    ))
                }
            },
        }
    }
}

fn draw_logistic(stream: &mut Stream, mean: (f64, f64), cov: (f64, f64, f64)) -> (f64, f64) {
    let (vi, c, vs) = cov;
    if vi <= 0.0 && vs <= 0.0 {
        return mean;
    }
    // Cholesky of the 2x2 covariance, tolerating a semi-definite matrix.
    let l11 = vi.max(0.0).sqrt();
    let l21 = if l11 > 0.0 { c / l11 } else { 0.0 };
    let l22 = (vs - l21 * l21).max(0.0).sqrt();
    let (a, b) = (stream.normal(), stream.normal());
    (mean.0 + l11 * a, mean.1 + l21 * a + l22 * b)
}

#[allow(clippy::too_many_arguments, reason = "the latent-class declaration is one value per axis")]
fn fit_latent(
    codings: &[u8],
    coding_count: usize,
    categories: usize,
    latent_levels: usize,
    exchangeable: bool,
    lowest_level: Option<&[f64]>,
    starts: usize,
    seed: u64,
) -> Result<LatentClassFit, MeasurementError> {
    let units = codings.len() / coding_count.max(1);
    let data = LatentClassData { codings, units, coding_count, categories, latent_levels };
    let constraints =
        LatentClassConstraints { exchangeable, lowest_level: lowest_level.map(<[f64]>::to_vec) };
    let settings = LatentClassSettings { starts, seed, ..LatentClassSettings::default() };
    fit_latent_class(&data, &constraints, &settings).map_err(|error| stats_error(&error))
}

// ---------------------------------------------------------------------------------------------
// The draws
// ---------------------------------------------------------------------------------------------

/// A corrected exposure as an uncertain column.
#[derive(Clone, Debug, PartialEq)]
pub struct ExposureDraws {
    /// Kind of the declaration the draws come from.
    pub kind: &'static str,
    /// Latent levels.
    pub levels: usize,
    /// Units.
    pub rows: usize,
    /// Imputations.
    pub draw_count: usize,
    /// Posterior mean of every unit's level, scaled to `[0, 1]` (the probability the exposure
    /// is present for a binary exposure).
    pub probability: Vec<f64>,
    /// The imputed level of every unit in every draw, draw-major (`draw_count x rows`).
    pub draws: Vec<f64>,
    /// Digest of the declaration.
    pub declaration_identity: String,
    /// Digest of the outcome, the covariates and the declaration's data.
    pub data_identity: String,
    /// Seed of the imputations.
    pub seed: u64,
}

impl ExposureDraws {
    /// The completed exposure column of draw `k`.
    #[must_use]
    pub fn column(&self, k: usize) -> &[f64] {
        &self.draws[k * self.rows..(k + 1) * self.rows]
    }
}

/// Impute the latent exposure `draw_count` times.
///
/// The posterior of every unit is formed from models fitted on a row bootstrap and from
/// redrawn measurement parameters, so the draws carry the uncertainty of both; the point
/// posterior (`probability`) uses the full data and the point parameters.
///
/// # Errors
/// An unidentified declaration (`measurement_not_identified`), an outcome that cannot carry an
/// outcome-specific error (`route_not_supported`), malformed inputs (`invalid_argument`), a
/// latent-class model that cannot be refitted, or cancellation.
pub fn impute_exposure(
    declaration: &ExposureMeasurement,
    data: &ImputationData<'_>,
    draw_count: usize,
    seed: u64,
    ctx: &ExecutionContext,
) -> Result<ExposureDraws, MeasurementError> {
    check_cancelled(ctx)?;
    let n = data.outcome.len();
    if draw_count < 2 || n < 20 {
        return Err(invalid("an imputation needs at least two draws and twenty units"));
    }
    if data.outcome.iter().any(|v| !v.is_finite())
        || data.covariates.iter().any(|v| !v.is_finite())
        || data.covariates.len() != n * data.covariate_count
    {
        return Err(invalid("the outcome and covariates must be finite and rectangular"));
    }
    let prepared = Prepared::new(declaration, data.outcome)?;
    let levels = prepared.levels;
    let z = standardised_rows(data);
    let pz = data.covariate_count;
    let fixed = prepared.fixed_prior.as_deref();
    let everyone: Vec<usize> = (0..n).collect();
    let point_lik = prepared.likelihood(data.outcome, None)?;
    let point = posterior(&point_lik, data.outcome, &z, pz, levels, fixed, &everyone);
    let denominator = (levels - 1) as f64;
    let probability: Vec<f64> = (0..n)
        .map(|i| (0..levels).map(|j| j as f64 * point[i * levels + j]).sum::<f64>() / denominator)
        .collect();
    let mut stream = Stream::new(seed);
    let mut draws = vec![0.0; draw_count * n];
    for k in 0..draw_count {
        check_cancelled(ctx)?;
        let boot: Vec<usize> = (0..n).map(|_| stream.below(n)).collect();
        let likelihood = prepared.likelihood(data.outcome, Some((&mut stream, &boot)))?;
        let post = posterior(&likelihood, data.outcome, &z, pz, levels, fixed, &boot);
        for i in 0..n {
            let u = stream.next();
            let mut cumulative = 0.0;
            let mut level = levels - 1;
            for j in 0..levels - 1 {
                cumulative += post[i * levels + j];
                if u <= cumulative {
                    level = j;
                    break;
                }
            }
            draws[k * n + i] = level as f64;
        }
    }
    let data_identity = digest(&[
        ("outcome", data.outcome),
        ("covariates", data.covariates),
        ("declaration", [f64::from(u8::from(fixed.is_some()))].as_slice()),
    ])
    .to_hex();
    Ok(ExposureDraws {
        kind: declaration.kind(),
        levels,
        rows: n,
        draw_count,
        probability,
        draws,
        declaration_identity: declaration.identity(),
        data_identity,
        seed,
    })
}

// ---------------------------------------------------------------------------------------------
// Pooling
// ---------------------------------------------------------------------------------------------

/// The effect of a measured exposure pooled over the exposure draws.
#[derive(Clone, Debug, PartialEq)]
pub struct MeasuredEffect {
    /// Rubin's-rules pooling of the per-draw estimates (large-sample degrees of freedom).
    pub pooled: RubinPooled,
    /// The per-draw estimates.
    pub points: Vec<f64>,
    /// The per-draw variances.
    pub variances: Vec<f64>,
    /// The estimate that takes the recorded column at face value, when the exposure has one.
    pub naive: Option<f64>,
}

impl MeasuredEffect {
    /// Pooled standard error (uncalibrated).
    #[must_use]
    pub fn standard_error(&self) -> f64 {
        self.pooled.standard_error()
    }

    /// Pooled `t` interval at `level`; calibration is `not_measured`.
    #[must_use]
    pub fn interval(&self, level: f64) -> (f64, f64) {
        self.pooled.interval(level)
    }

    /// Named variance components: within-draw, between-draw (times `1 + 1/m`) and total.
    #[must_use]
    pub fn variance_components(&self) -> [(&'static str, f64); 3] {
        let m = self.pooled.imputations as f64;
        [
            ("within_imputation", self.pooled.within),
            ("between_imputation", (1.0 + 1.0 / m) * self.pooled.between),
            ("total", self.pooled.total),
        ]
    }
}

/// Pool per-draw estimates and their variances with Rubin's rules.
///
/// # Errors
/// Fewer than two draws, lengths that differ or non-finite values (`invalid_argument`).
pub fn pool_measured_effect(
    points: &[f64],
    variances: &[f64],
    naive: Option<f64>,
) -> Result<MeasuredEffect, MeasurementError> {
    let pooled = rubin_pool(points, variances, None).map_err(|error| invalid(error.to_string()))?;
    Ok(MeasuredEffect { pooled, points: points.to_vec(), variances: variances.to_vec(), naive })
}
