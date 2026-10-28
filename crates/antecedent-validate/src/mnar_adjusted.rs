//! Covariate-adjusted estimators re-run under a missing-not-at-random departure.
//!
//! For each arm an outcome regression on the covariates is fitted to the arm's observed
//! outcomes. A missing outcome is imputed by that regression's prediction plus the arm's
//! departure: the pattern-mixture shift `loading * delta`, or, for the selection model, the
//! mean regression residual under the exponentially tilted weights `exp(loading * theta * y)`
//! of the observed outcomes (the observed residual law tilted toward the missing pattern).
//! Parameter zero imputes the fitted values, which leaves the least-squares fit unchanged:
//! the missing-at-random estimate. The declared estimator is then re-run on the completed
//! table: the arm-specific regression refitted and averaged over every unit (regression
//! adjustment), the Hajek inverse-propensity-weighted contrast with a logistic propensity
//! model, or the augmented inverse-propensity-weighted contrast combining the two.
//!
//! The sampling uncertainty at a parameter value is a seeded nonparametric bootstrap that
//! resamples units within arm and repeats the imputation fit, the departure and the estimator,
//! with the same resampling stream at every parameter value.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::mnar_sensitivity::{ArmLoading, MnarEstimator, MnarModel};

/// Rows, arm, possibly missing outcomes and covariate columns of one analysis.
pub(crate) struct AdjData {
    pub(crate) treated: Vec<bool>,
    pub(crate) outcome: Vec<Option<f64>>,
    pub(crate) covariates: Vec<Vec<f64>>,
}

impl AdjData {
    fn len(&self) -> usize {
        self.treated.len()
    }

    fn arm_rows(&self, treated: bool) -> Vec<usize> {
        (0..self.len()).filter(|&i| self.treated[i] == treated).collect()
    }

    fn gather(&self, rows: &[usize]) -> Self {
        Self {
            treated: rows.iter().map(|&i| self.treated[i]).collect(),
            outcome: rows.iter().map(|&i| self.outcome[i]).collect(),
            covariates: self
                .covariates
                .iter()
                .map(|column| rows.iter().map(|&i| column[i]).collect())
                .collect(),
        }
    }

    fn design(&self, row: usize, k: usize) -> f64 {
        if k == 0 {
            1.0
        } else {
            self.covariates[k - 1][row]
        }
    }

    fn width(&self) -> usize {
        self.covariates.len() + 1
    }
}

/// Solve `a x = b` (`a` is `q x q`, row-major) by elimination with partial pivoting; `None`
/// when a pivot is negligible next to the largest diagonal entry (a singular design).
fn solve(mut a: Vec<f64>, mut b: Vec<f64>, q: usize) -> Option<Vec<f64>> {
    let scale = (0..q).map(|i| a[i * q + i].abs()).fold(0.0, f64::max);
    if scale <= 0.0 || !scale.is_finite() {
        return None;
    }
    for col in 0..q {
        let mut pivot = col;
        for r in col + 1..q {
            if a[r * q + col].abs() > a[pivot * q + col].abs() {
                pivot = r;
            }
        }
        if a[pivot * q + col].abs() < 1e-12 * scale {
            return None;
        }
        if pivot != col {
            for c in 0..q {
                a.swap(col * q + c, pivot * q + c);
            }
            b.swap(col, pivot);
        }
        for r in col + 1..q {
            let factor = a[r * q + col] / a[col * q + col];
            for c in col..q {
                a[r * q + c] -= factor * a[col * q + c];
            }
            b[r] -= factor * b[col];
        }
    }
    let mut x = vec![0.0; q];
    for i in (0..q).rev() {
        let mut s = b[i];
        for c in i + 1..q {
            s -= a[i * q + c] * x[c];
        }
        x[i] = s / a[i * q + i];
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}

/// Least squares of `y` (indexed by global row) on the design over `rows`.
fn ols(data: &AdjData, rows: &[usize], y: &[f64]) -> Option<Vec<f64>> {
    let q = data.width();
    if rows.len() <= q {
        return None;
    }
    let mut xtx = vec![0.0; q * q];
    let mut xty = vec![0.0; q];
    for &i in rows {
        for r in 0..q {
            let xr = data.design(i, r);
            xty[r] += xr * y[i];
            for c in 0..q {
                xtx[r * q + c] += xr * data.design(i, c);
            }
        }
    }
    solve(xtx, xty, q)
}

fn predict(data: &AdjData, beta: &[f64], row: usize) -> f64 {
    (0..beta.len()).map(|k| beta[k] * data.design(row, k)).sum()
}

fn expit(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

/// Logistic propensity of the arm given the covariates, by iteratively reweighted least
/// squares; `None` when it does not converge or a propensity leaves `[1e-6, 1 - 1e-6]`.
pub(crate) fn propensity(data: &AdjData) -> Option<Vec<f64>> {
    let n = data.len();
    let q = data.width();
    let mut beta = vec![0.0; q];
    let mut converged = false;
    for _ in 0..50 {
        let mut xtwx = vec![0.0; q * q];
        let mut score = vec![0.0; q];
        for i in 0..n {
            let p = expit(predict(data, &beta, i));
            let w = p * (1.0 - p);
            let residual = f64::from(u8::from(data.treated[i])) - p;
            for r in 0..q {
                let xr = data.design(i, r);
                score[r] += xr * residual;
                for c in 0..q {
                    xtwx[r * q + c] += w * xr * data.design(i, c);
                }
            }
        }
        let step = solve(xtwx, score, q)?;
        let largest = step.iter().fold(0.0_f64, |m, s| m.max(s.abs()));
        for (b, s) in beta.iter_mut().zip(&step) {
            *b += s;
        }
        if largest < 1e-10 {
            converged = true;
            break;
        }
    }
    let ps: Vec<f64> = (0..n).map(|i| expit(predict(data, &beta, i))).collect();
    (converged && ps.iter().all(|p| (1e-6..=1.0 - 1e-6).contains(p))).then_some(ps)
}

/// The treated and control mean outcomes under the departure `param`, by `estimator`; `None`
/// for a singular design or an arm with too few observed outcomes.
pub(crate) fn arm_means(
    data: &AdjData,
    propensities: Option<&[f64]>,
    model: MnarModel,
    loading: ArmLoading,
    param: f64,
    estimator: MnarEstimator,
) -> Option<(f64, f64)> {
    let n = data.len();
    let arms = [data.arm_rows(true), data.arm_rows(false)];
    let observed_y: Vec<f64> = data.outcome.iter().map(|y| y.unwrap_or(f64::NAN)).collect();
    let mut completed = vec![0.0; n];
    for (k, arm) in arms.iter().enumerate() {
        let observed: Vec<usize> =
            arm.iter().copied().filter(|&i| data.outcome[i].is_some()).collect();
        let beta = ols(data, &observed, &observed_y)?;
        let load = if k == 0 { loading.treated } else { loading.control };
        let t = load * param;
        let shift = match model {
            MnarModel::DeltaPatternMixture => t,
            MnarModel::ExponentialTiltingSelection => {
                let ys = observed.iter().map(|&i| observed_y[i]);
                let anchor = if t >= 0.0 {
                    ys.fold(f64::NEG_INFINITY, f64::max)
                } else {
                    ys.fold(f64::INFINITY, f64::min)
                };
                let (mut total, mut weighted) = (0.0, 0.0);
                for &i in &observed {
                    let w = (t * (observed_y[i] - anchor)).exp();
                    total += w;
                    weighted += w * (observed_y[i] - predict(data, &beta, i));
                }
                weighted / total
            }
        };
        for &i in arm {
            completed[i] = match data.outcome[i] {
                Some(y) => y,
                None => predict(data, &beta, i) + shift,
            };
        }
    }
    let mut fitted: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
    for (k, arm) in arms.iter().enumerate() {
        let beta = ols(data, arm, &completed)?;
        fitted[k] = (0..n).map(|i| predict(data, &beta, i)).collect();
    }
    let nf = n as f64;
    let mean = |v: &[f64]| v.iter().sum::<f64>() / nf;
    match estimator {
        MnarEstimator::RegressionAdjusted => Some((mean(&fitted[0]), mean(&fitted[1]))),
        MnarEstimator::InverseProbabilityWeighting => {
            let e = propensities?;
            let (mut num1, mut den1, mut num0, mut den0) = (0.0, 0.0, 0.0, 0.0);
            for i in 0..n {
                if data.treated[i] {
                    num1 += completed[i] / e[i];
                    den1 += 1.0 / e[i];
                } else {
                    num0 += completed[i] / (1.0 - e[i]);
                    den0 += 1.0 / (1.0 - e[i]);
                }
            }
            Some((num1 / den1, num0 / den0))
        }
        MnarEstimator::AugmentedInverseProbabilityWeighting => {
            let e = propensities?;
            let (mut mu1, mut mu0) = (0.0, 0.0);
            for i in 0..n {
                mu1 += fitted[0][i];
                mu0 += fitted[1][i];
                if data.treated[i] {
                    mu1 += (completed[i] - fitted[0][i]) / e[i];
                } else {
                    mu0 += (completed[i] - fitted[1][i]) / (1.0 - e[i]);
                }
            }
            Some((mu1 / nf, mu0 / nf))
        }
        MnarEstimator::DifferenceInMeans => None,
    }
}

/// SplitMix64: a small deterministic stream for the bootstrap resamples.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Bootstrap standard error of the effect at `param`: units resampled within arm, the whole
/// pipeline repeated. The resampling stream depends only on `seed`, so the standard error is
/// a smooth function of the parameter. `None` when fewer than half the replicates fit.
pub(crate) fn bootstrap_se(
    data: &AdjData,
    model: MnarModel,
    loading: ArmLoading,
    param: f64,
    estimator: MnarEstimator,
    replicates: u32,
    seed: u64,
) -> Option<f64> {
    let arms = [data.arm_rows(true), data.arm_rows(false)];
    let mut rng = SplitMix(seed);
    let mut effects = Vec::with_capacity(replicates as usize);
    for _ in 0..replicates {
        let mut rows = Vec::with_capacity(data.len());
        for arm in &arms {
            for _ in 0..arm.len() {
                rows.push(arm[rng.below(arm.len())]);
            }
        }
        let sample = data.gather(&rows);
        let wants_propensity = matches!(
            estimator,
            MnarEstimator::InverseProbabilityWeighting
                | MnarEstimator::AugmentedInverseProbabilityWeighting
        );
        let ps = if wants_propensity {
            match propensity(&sample) {
                Some(ps) => Some(ps),
                None => continue,
            }
        } else {
            None
        };
        if let Some((mu1, mu0)) =
            arm_means(&sample, ps.as_deref(), model, loading, param, estimator)
        {
            effects.push(mu1 - mu0);
        }
    }
    let count = effects.len();
    if count < 10 || count < (replicates as usize) / 2 {
        return None;
    }
    let mean = effects.iter().sum::<f64>() / count as f64;
    let variance = effects.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / (count as f64 - 1.0);
    Some(variance.sqrt())
}

/// Build the analysis rows.
#[allow(clippy::float_cmp, reason = "the treatment is coded exactly 0 or 1")]
pub(crate) fn adjusted_data(
    treatment: &[f64],
    outcome: &[Option<f64>],
    covariates: &[&[f64]],
) -> AdjData {
    AdjData {
        treated: treatment.iter().map(|&t| t == 1.0).collect(),
        outcome: outcome.to_vec(),
        covariates: covariates.iter().map(|column| column.to_vec()).collect(),
    }
}
