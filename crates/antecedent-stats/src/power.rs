//! Power numerics: the noncentral Student-t and F laws, test power from a
//! noncentrality, and the joint law of equicorrelated comparisons against one
//! shared control (Dunnett single-step, Bonferroni and Holm step-down).
//!
//! * [`noncentral_t_cdf`] is Lenth's algorithm AS 243 (1989): a Poisson-weighted
//!   series of incomplete beta functions with a bounded truncation error. Above
//!   a noncentrality of 37.5 the Poisson weights underflow and the CDF is the
//!   normal approximation of Johnson and Welch (1940), whose error there is far
//!   below the tolerance of any power statement.
//! * [`noncentral_f_sf`] is the Poisson mixture of central beta tails.
//! * [`EquicorrelatedComparisons`] holds `K` statistics `T_k = (Z_k + δ) / S`
//!   with `Corr(Z_j, Z_k) = ρ ≥ 0` and `S = sqrt(χ²_ν / ν)` (or `S = 1` for a
//!   normal reference). Writing `Z_k = sqrt(ρ) W + sqrt(1 − ρ) E_k`, the
//!   statistics are independent given `(W, S)`. A rejection count of a
//!   single-step or step-down procedure is then a function of conditionally
//!   independent, identically distributed exceedances, which a dynamic
//!   programme over the ordered critical values evaluates exactly; the outer
//!   integral over `(W, S)` is composite Gauss–Legendre quadrature. The Dunnett
//!   critical value solves the familywise null probability by bisection.
//!
//! Every function returns `NaN` (or `None`) for an input outside its domain;
//! callers validate and refuse before they get here.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_kernels::{norm_cdf, norm_sf};

use crate::adaptive_inference::regularized_incomplete_beta;
use crate::special::{gauss_legendre, ln_gamma, student_t_ppf, student_t_sf};

/// Most comparisons [`EquicorrelatedComparisons`] accepts.
pub const MAX_COMPARISONS: usize = 20;

/// Noncentrality above which [`noncentral_t_cdf`] uses the normal approximation.
const SERIES_NONCENTRALITY_LIMIT: f64 = 37.5;
/// Degrees of freedom above which a Student-t law is evaluated as normal.
const NORMAL_DF_LIMIT: f64 = 1.0e7;
const AS243_MAX_TERMS: usize = 5_000;
const AS243_ERROR: f64 = 1.0e-12;
const BISECTIONS: usize = 200;
const QUADRATURE_NODES: usize = 8;
const FACTOR_PANELS: usize = 16;
const FACTOR_HALF_RANGE: f64 = 8.5;
const SCALE_PANELS: usize = 12;
/// Wilson–Hilferty normal quantile bounding the chi-square scale integral.
const SCALE_TAIL_Z: f64 = 7.5;

/// Student-t CDF `P(T ≤ t)`; `df = None` is the standard normal.
#[must_use]
pub fn student_t_cdf(t: f64, df: Option<f64>) -> f64 {
    match df {
        Some(df) if df < NORMAL_DF_LIMIT => 1.0 - student_t_sf(t, df),
        _ => norm_cdf(t),
    }
}

/// Upper `1 − p` quantile of a Student-t law; `df = None` is the standard normal.
#[must_use]
pub fn student_t_upper_quantile(p: f64, df: Option<f64>) -> f64 {
    if !(p.is_finite() && p > 0.0 && p < 1.0) {
        return f64::NAN;
    }
    match df {
        Some(df) if df < NORMAL_DF_LIMIT => student_t_ppf(1.0 - p, df),
        _ => antecedent_kernels::norm_inv(1.0 - p),
    }
}

/// Noncentral Student-t CDF `P(T ≤ t)` with `df` degrees of freedom and
/// noncentrality `noncentrality` (Lenth 1989, algorithm AS 243).
///
/// Returns `NaN` for a non-positive or non-finite `df` or a non-finite argument.
#[must_use]
pub fn noncentral_t_cdf(t: f64, df: f64, noncentrality: f64) -> f64 {
    if !(df.is_finite() && df > 0.0) || t.is_nan() || !noncentrality.is_finite() {
        return f64::NAN;
    }
    if t == f64::INFINITY {
        return 1.0;
    }
    if t == f64::NEG_INFINITY {
        return 0.0;
    }
    if df >= NORMAL_DF_LIMIT {
        return norm_cdf(t - noncentrality);
    }
    if noncentrality.abs() > SERIES_NONCENTRALITY_LIMIT {
        // Johnson–Welch normal approximation.
        let z = (t * (1.0 - 1.0 / (4.0 * df)) - noncentrality) / (1.0 + t * t / (2.0 * df)).sqrt();
        return norm_cdf(z);
    }
    let (tt, del, negative) =
        if t < 0.0 { (-t, -noncentrality, true) } else { (t, noncentrality, false) };
    let mut tnc = 0.0;
    let x = tt * tt / (tt * tt + df);
    if x > 0.0 {
        let lambda = del * del;
        let mut p = 0.5 * (-0.5 * lambda).exp();
        let mut q = (2.0 / std::f64::consts::PI).sqrt() * p * del;
        let mut s = 0.5 - p;
        let mut a = 0.5;
        let b = 0.5 * df;
        let rxb = (1.0 - x).powf(b);
        let albeta = 0.5 * std::f64::consts::PI.ln() + ln_gamma(b) - ln_gamma(a + b);
        let mut xodd = regularized_incomplete_beta(x, a, b).unwrap_or(f64::NAN);
        let mut godd = 2.0 * rxb * (a * x.ln() - albeta).exp();
        let mut xeven = 1.0 - rxb;
        let mut geven = b * x * rxb;
        tnc = p * xodd + q * xeven;
        let mut en = 1.0;
        for _ in 0..AS243_MAX_TERMS {
            a += 1.0;
            xodd -= godd;
            xeven -= geven;
            godd *= x * (a + b - 1.0) / a;
            geven *= x * (a + b - 0.5) / (a + 0.5);
            p *= lambda / (2.0 * en);
            q *= lambda / (2.0 * en + 1.0);
            s -= p;
            en += 1.0;
            tnc += p * xodd + q * xeven;
            let bound = 2.0 * s * (xodd - godd);
            if bound <= AS243_ERROR {
                break;
            }
        }
    }
    tnc += norm_sf(del);
    let tnc = tnc.clamp(0.0, 1.0);
    if negative { 1.0 - tnc } else { tnc }
}

/// Critical value of a level-`alpha` test: the upper `alpha` quantile (one-sided)
/// or the upper `alpha / 2` quantile (two-sided) of a Student-t law, or of the
/// standard normal when `df` is `None`.
#[must_use]
pub fn critical_value(alpha: f64, df: Option<f64>, two_sided: bool) -> f64 {
    let tail = if two_sided { alpha / 2.0 } else { alpha };
    student_t_upper_quantile(tail, df)
}

/// Rejection probability of a level-`alpha` test whose statistic is noncentral
/// Student-t (`df`) or normal (`df = None`) with noncentrality `noncentrality`.
///
/// A one-sided test rejects for large statistics. A two-sided test rejects in
/// either tail and its power counts both tails.
#[must_use]
pub fn t_test_power(noncentrality: f64, df: Option<f64>, alpha: f64, two_sided: bool) -> f64 {
    if !(alpha.is_finite() && alpha > 0.0 && alpha < 1.0 && noncentrality.is_finite()) {
        return f64::NAN;
    }
    let critical = critical_value(alpha, df, two_sided);
    let upper = 1.0 - shifted_cdf(critical, df, noncentrality);
    let lower = if two_sided { shifted_cdf(-critical, df, noncentrality) } else { 0.0 };
    (upper + lower).clamp(0.0, 1.0)
}

fn shifted_cdf(t: f64, df: Option<f64>, noncentrality: f64) -> f64 {
    match df {
        Some(df) if df < NORMAL_DF_LIMIT => noncentral_t_cdf(t, df, noncentrality),
        _ => norm_cdf(t - noncentrality),
    }
}

/// Survival `P(F > x)` of a noncentral F law with `df1` and `df2` degrees of
/// freedom and noncentrality `lambda` (Poisson mixture of central beta tails).
#[must_use]
pub fn noncentral_f_sf(x: f64, df1: f64, df2: f64, lambda: f64) -> f64 {
    if !(df1.is_finite()
        && df1 > 0.0
        && df2.is_finite()
        && df2 > 0.0
        && lambda.is_finite()
        && lambda >= 0.0)
        || x.is_nan()
    {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 1.0;
    }
    if x == f64::INFINITY {
        return 0.0;
    }
    let y = df1 * x / (df1 * x + df2);
    let half = 0.5 * lambda;
    // Sum outward from the Poisson mode so that a large noncentrality loses no mass.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a finite, nonnegative mode"
    )]
    let mode = half.floor() as usize;
    let weight = |j: usize| -> f64 {
        let jf = j as f64;
        if half == 0.0 {
            return if j == 0 { 1.0 } else { 0.0 };
        }
        (-half + jf * half.ln() - ln_gamma(jf + 1.0)).exp()
    };
    let term = |j: usize| -> f64 {
        let cdf =
            regularized_incomplete_beta(y, 0.5 * df1 + j as f64, 0.5 * df2).unwrap_or(f64::NAN);
        weight(j) * (1.0 - cdf)
    };
    let mut total = 0.0;
    let mut j = mode;
    loop {
        let w = weight(j);
        total += term(j);
        if w < 1e-16 && j > mode + 10 {
            break;
        }
        j += 1;
        if j > mode + 100_000 {
            break;
        }
    }
    let mut j = mode;
    while j > 0 {
        j -= 1;
        let w = weight(j);
        total += term(j);
        if w < 1e-16 && j + 10 < mode {
            break;
        }
    }
    total.clamp(0.0, 1.0)
}

/// Upper `alpha` critical value of a central F law.
#[must_use]
pub fn f_critical_value(alpha: f64, df1: f64, df2: f64) -> f64 {
    if !(alpha.is_finite() && alpha > 0.0 && alpha < 1.0) {
        return f64::NAN;
    }
    let (mut low, mut high) = (0.0_f64, 1.0_f64);
    while noncentral_f_sf(high, df1, df2, 0.0) > alpha && high < 1e12 {
        high *= 2.0;
    }
    for _ in 0..BISECTIONS {
        let mid = 0.5 * (low + high);
        if noncentral_f_sf(mid, df1, df2, 0.0) > alpha {
            low = mid;
        } else {
            high = mid;
        }
        if high - low < 1e-12 * high.max(1.0) {
            break;
        }
    }
    0.5 * (low + high)
}

/// Rejection probability of the level-`alpha` F test with noncentrality `lambda`.
#[must_use]
pub fn f_test_power(lambda: f64, df1: f64, df2: f64, alpha: f64) -> f64 {
    let critical = f_critical_value(alpha, df1, df2);
    noncentral_f_sf(critical, df1, df2, lambda)
}

/// `K` comparison statistics `T_k = (Z_k + δ) / S` with equicorrelated
/// standard-normal `Z_k` and an independent chi scale `S`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EquicorrelatedComparisons {
    /// Number of comparisons `K`, between 1 and [`MAX_COMPARISONS`].
    pub comparisons: usize,
    /// Common correlation `ρ` in `[0, 1)`.
    pub correlation: f64,
    /// Degrees of freedom of the shared variance estimate; `None` is a known variance.
    pub df: Option<f64>,
    /// Whether a comparison rejects in either tail (`|T_k| ≥ c`) or the upper one.
    pub two_sided: bool,
}

/// One node of the outer quadrature: weight, factor `W` and scale `S`.
#[derive(Clone, Copy)]
struct Node {
    weight: f64,
    factor: f64,
    scale: f64,
}

impl EquicorrelatedComparisons {
    fn valid(&self) -> bool {
        (1..=MAX_COMPARISONS).contains(&self.comparisons)
            && self.correlation.is_finite()
            && (0.0..1.0).contains(&self.correlation)
            && self.df.is_none_or(|df| df.is_finite() && df > 0.0)
    }

    fn nodes(&self) -> Vec<Node> {
        let (unit_nodes, unit_weights) = gauss_legendre(QUADRATURE_NODES);
        let composite = |low: f64, high: f64, panels: usize| -> Vec<(f64, f64)> {
            let width = (high - low) / panels as f64;
            let mut out = Vec::with_capacity(panels * QUADRATURE_NODES);
            for panel in 0..panels {
                let start = low + width * panel as f64;
                for (u, w) in unit_nodes.iter().zip(&unit_weights) {
                    out.push((start + 0.5 * width * (u + 1.0), 0.5 * width * w));
                }
            }
            out
        };
        let factors = if self.correlation == 0.0 {
            vec![(0.0, 1.0)]
        } else {
            normalized(
                composite(-FACTOR_HALF_RANGE, FACTOR_HALF_RANGE, FACTOR_PANELS)
                    .into_iter()
                    .map(|(w, weight)| (w, weight * antecedent_kernels::norm_pdf(w)))
                    .collect(),
            )
        };
        let scales = match self.df {
            Some(df) if df < NORMAL_DF_LIMIT => {
                let bound = |z: f64| -> f64 {
                    let c = 2.0 / (9.0 * df);
                    let cube = (1.0 - c + z * c.sqrt()).max(0.0);
                    (cube * cube * cube).sqrt()
                };
                let (low, high) = (bound(-SCALE_TAIL_Z).max(1e-9), bound(SCALE_TAIL_Z));
                // Log density of S = sqrt(χ²_ν / ν) up to its constant, taken
                // relative to the mode so that no large terms cancel; the
                // weights are then normalized to one on the quadrature itself.
                let log_kernel = |s: f64| (df - 1.0) * s.ln() - 0.5 * df * s * s;
                let mode = ((df - 1.0).max(0.0) / df).sqrt().clamp(low, high);
                let peak = log_kernel(mode);
                normalized(
                    composite(low, high, SCALE_PANELS)
                        .into_iter()
                        .map(|(s, weight)| (s, weight * (log_kernel(s) - peak).exp()))
                        .collect(),
                )
            }
            _ => vec![(1.0, 1.0)],
        };
        let mut nodes = Vec::with_capacity(factors.len() * scales.len());
        for &(factor, wf) in &factors {
            for &(scale, ws) in &scales {
                nodes.push(Node { weight: wf * ws, factor, scale });
            }
        }
        nodes
    }

    /// Conditional probability that one statistic reaches `critical`, given the
    /// factor and the scale.
    fn exceedance(&self, node: Node, noncentrality: f64, critical: f64) -> f64 {
        let root = self.correlation.sqrt();
        let spread = (1.0 - self.correlation).sqrt();
        let centre = noncentrality + root * node.factor;
        let upper = norm_sf((critical * node.scale - centre) / spread);
        if self.two_sided {
            upper + norm_cdf((-critical * node.scale - centre) / spread)
        } else {
            upper
        }
    }

    /// Probability that no comparison reaches `critical` when every
    /// comparison has noncentrality `noncentrality`.
    #[must_use]
    pub fn none_reach(&self, noncentrality: f64, critical: f64) -> f64 {
        if !self.valid() || !critical.is_finite() || !noncentrality.is_finite() {
            return f64::NAN;
        }
        let exponent = i32::try_from(self.comparisons).unwrap_or(i32::MAX);
        self.nodes()
            .into_iter()
            .map(|node| {
                node.weight * (1.0 - self.exceedance(node, noncentrality, critical)).powi(exponent)
            })
            .sum::<f64>()
            .clamp(0.0, 1.0)
    }

    /// Dunnett's single-step critical value: the `c` at which the probability
    /// that any null comparison reaches `c` is `alpha`.
    #[must_use]
    pub fn dunnett_critical(&self, alpha: f64) -> Option<f64> {
        if !(self.valid() && alpha.is_finite() && alpha > 0.0 && alpha < 1.0) {
            return None;
        }
        let (mut low, mut high) = (0.0_f64, 1.0_f64);
        while 1.0 - self.none_reach(0.0, high) > alpha {
            high *= 2.0;
            if high > 1e3 {
                return None;
            }
        }
        for _ in 0..BISECTIONS {
            let mid = 0.5 * (low + high);
            if 1.0 - self.none_reach(0.0, mid) > alpha {
                low = mid;
            } else {
                high = mid;
            }
            if high - low < 1e-9 {
                break;
            }
        }
        Some(0.5 * (low + high))
    }

    /// Distribution of the number of rejections of a step-down procedure with
    /// non-increasing critical values `critical` (`c_1 ≥ … ≥ c_K`) when every
    /// comparison has noncentrality `noncentrality`.
    ///
    /// Entry `j − 1` is `P(R ≥ j)`: the procedure rejects at least `j`
    /// hypotheses, which happens when for every `l ≤ j` at least `l` statistics
    /// reach `c_l`. Constant critical values give a single-step procedure
    /// (Bonferroni, Dunnett). `P(R ≥ 1)` is disjunctive power, `P(R ≥ K)`
    /// conjunctive power and `E[R] / K` the power of each comparison.
    #[must_use]
    pub fn rejection_tail(&self, noncentrality: f64, critical: &[f64]) -> Option<Vec<f64>> {
        let k = self.comparisons;
        if !self.valid()
            || critical.len() != k
            || critical.iter().any(|c| !c.is_finite())
            || critical.windows(2).any(|pair| pair[1] > pair[0])
            || !noncentrality.is_finite()
        {
            return None;
        }
        let binomial = binomial_table(k);
        let mut tail = vec![0.0; k];
        let mut exceed = vec![0.0; k];
        let mut state = vec![0.0; k + 1];
        let mut next = vec![0.0; k + 1];
        for node in self.nodes() {
            for (slot, &c) in exceed.iter_mut().zip(critical) {
                *slot = self.exceedance(node, noncentrality, c);
            }
            state.iter_mut().for_each(|value| *value = 0.0);
            state[0] = 1.0;
            for i in 0..k {
                let previous = if i == 0 { 0.0 } else { exceed[i - 1] };
                let band = (exceed[i] - previous).max(0.0);
                let remaining = (1.0 - previous).max(0.0);
                let p = if remaining > 0.0 { (band / remaining).clamp(0.0, 1.0) } else { 0.0 };
                next.iter_mut().for_each(|value| *value = 0.0);
                for (m, &mass) in state.iter().enumerate() {
                    if mass == 0.0 {
                        continue;
                    }
                    let left = k - m;
                    for t in 0..=left {
                        let total = m + t;
                        if total < i + 1 {
                            continue;
                        }
                        let exponent_hit = i32::try_from(t).unwrap_or(i32::MAX);
                        let exponent_miss = i32::try_from(left - t).unwrap_or(i32::MAX);
                        next[total] += mass
                            * binomial[left][t]
                            * p.powi(exponent_hit)
                            * (1.0 - p).powi(exponent_miss);
                    }
                }
                std::mem::swap(&mut state, &mut next);
                tail[i] += node.weight * state.iter().sum::<f64>();
            }
        }
        Some(tail.into_iter().map(|value| value.clamp(0.0, 1.0)).collect())
    }
}

/// Rescale quadrature weights to sum to one.
///
/// Every probability the comparisons report is an average over the same nodes,
/// so the familywise level the Dunnett critical value is solved for
/// (`1 − none_reach`) and the rejection distribution read at that critical
/// value (`rejection_tail`) agree only when the weights carry unit mass; any
/// truncation or quadrature deficit would otherwise be counted as rejection by
/// one and not by the other.
fn normalized(mut nodes: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    let mass: f64 = nodes.iter().map(|(_, w)| w).sum();
    if mass > 0.0 && mass.is_finite() {
        for (_, w) in &mut nodes {
            *w /= mass;
        }
    }
    nodes
}

fn binomial_table(k: usize) -> Vec<Vec<f64>> {
    let mut table = vec![vec![0.0; k + 1]; k + 1];
    for n in 0..=k {
        table[n][0] = 1.0;
        for r in 1..=n {
            table[n][r] = table[n - 1][r - 1] + if r < n { table[n - 1][r] } else { 0.0 };
        }
    }
    table
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn central_t_matches_student() {
        for &(t, df) in &[(1.3, 7.0), (-0.4, 30.0), (2.5, 3.0)] {
            let reference = 1.0 - student_t_sf(t, df);
            assert!((noncentral_t_cdf(t, df, 0.0) - reference).abs() < 1e-10);
        }
    }

    #[test]
    fn two_sample_t_power_matches_the_published_value() {
        // d = 0.5, 64 per group, two-sided 0.05: 0.8015 (Cohen 1988; G*Power 0.80146).
        let delta = 0.5 * (64.0_f64 * 64.0 / 128.0).sqrt();
        let power = t_test_power(delta, Some(126.0), 0.05, true);
        assert!((power - 0.80146).abs() < 5e-4, "{power}");
    }

    #[test]
    fn f_with_one_numerator_df_is_a_squared_t() {
        let (delta, df) = (2.1, 40.0);
        let f = f_test_power(delta * delta, 1.0, df, 0.05);
        let t = t_test_power(delta, Some(df), 0.05, true);
        assert!((f - t).abs() < 1e-6, "{f} vs {t}");
    }

    #[test]
    fn dunnett_two_sided_critical_values_match_the_tables() {
        // Dunnett (1964), two-sided 0.05, equal allocation (ρ = 1/2).
        for &(k, df, want) in
            &[(2, None, 2.212), (3, None, 2.349), (2, Some(10.0), 2.57), (3, Some(10.0), 2.76)]
        {
            let comparisons =
                EquicorrelatedComparisons { comparisons: k, correlation: 0.5, df, two_sided: true };
            let got = comparisons.dunnett_critical(0.05).unwrap();
            assert!((got - want).abs() < 0.01, "k={k} df={df:?}: {got} vs {want}");
        }
    }

    #[test]
    fn single_step_tail_at_the_null_is_the_familywise_error() {
        let comparisons = EquicorrelatedComparisons {
            comparisons: 3,
            correlation: 0.5,
            df: Some(60.0),
            two_sided: true,
        };
        let c = comparisons.dunnett_critical(0.05).unwrap();
        let tail = comparisons.rejection_tail(0.0, &[c, c, c]).unwrap();
        assert!((tail[0] - 0.05).abs() < 1e-6, "{tail:?}");
        assert!(tail[0] >= tail[1] && tail[1] >= tail[2]);
    }
}
