//! Trend-corrected autoregressive error model of a local ratio in calendar time.
//!
//! The reported ratio is a fixed linear functional `a·e` of the window's errors `e`, and it
//! loads on the low-frequency part of `e`: the weights extrapolate a local polynomial to the
//! cutoff. The only error series the data show is the residual of an unweighted cubic trend
//! on each side, and that residual has the low-frequency part removed. Its autocovariances are
//! therefore not those of the error: for an error with autocovariances `γ`, the expected
//! lag-`k` product sum of the residuals is `tr(Γ M S_k M)`, with `M` the trend's residual
//! maker and `S_k` the symmetric lag-`k` shift. That is linear in `γ` with weights
//! `d[k][h] = Σ_{|i-j|=h} (M S_k M)[i][j]` that depend on the design alone.
//!
//! An AR(`p`) error has autocovariances fixed by its `p` partial autocorrelations, so the
//! `p` equations "expected residual autocorrelation at lag `k` equals the observed one" (`k`
//! from 1 to `p`) identify them without the bias of fitting the residual as if it were the
//! error, and the marginal variance is the residual sum of squares over its expectation.
//! For `p = 1` this is the lag-one correction. The order is the one minimizing
//! `N ln σ² + p ln N` with `σ²` the innovation variance of the corrected fit, between one and
//! [`MAX_ORDER`]. A first-order model understates the low-frequency variance of a
//! higher-order error, because its fit extrapolates the lag-one autocorrelation alone.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

/// Highest autoregressive order of the error model.
pub(super) const MAX_ORDER: usize = 4;

/// Largest partial autocorrelation in absolute value.
pub(super) const MAX_PARTIAL: f64 = 0.97;

/// Step of the numerical derivatives in a partial autocorrelation.
pub(super) const DERIVATIVE_STEP: f64 = 0.01;

/// Lags of the autocorrelation sum in Bartlett's covariance of the observed autocorrelations.
const BARTLETT_LAGS: usize = 300;

/// Expected residual lag-`k` product sums of one side's trend, as weights on the error's
/// autocovariances: `diagonals[k][h]` for `k` in `0..=MAX_ORDER`, `h` in `0..rows`.
#[derive(Clone)]
pub(super) struct LagMoments {
    diagonals: Arc<[Vec<f64>]>,
}

/// Designs kept by [`LagMoments::cached`] before the cache is emptied.
const MOMENT_CACHE_CAPACITY: usize = 128;

/// Exact bit pattern of a trend design: the row count, the basis and the inverse Gram matrix.
type DesignKey = Vec<u64>;

type MomentCache = Mutex<HashMap<DesignKey, LagMoments>>;

fn design_key(basis: &[Vec<f64>], inverse: &[Vec<f64>]) -> DesignKey {
    let size = |len: usize| u64::try_from(len).unwrap_or(u64::MAX);
    let mut key = vec![size(basis.len()), size(inverse.len())];
    key.extend(basis.iter().chain(inverse).flatten().map(|value| value.to_bits()));
    key
}

impl LagMoments {
    /// [`LagMoments::new`] memoized on the exact bits of `basis` and `inverse`.
    ///
    /// The moments are a pure function of the two matrices (no outcome enters), so a design
    /// that repeats across replicates, as a declared bandwidth over a fixed time index does,
    /// is built once. The key is the full bit pattern, so a hit is bit-identical to a fresh
    /// build and a changed design always misses.
    pub(super) fn cached(basis: &[Vec<f64>], inverse: &[Vec<f64>]) -> Self {
        static CACHE: OnceLock<MomentCache> = OnceLock::new();
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let key = design_key(basis, inverse);
        if let Some(found) = cache.lock().unwrap_or_else(PoisonError::into_inner).get(&key) {
            return found.clone();
        }
        let built = Self::new(basis, inverse);
        let mut guard = cache.lock().unwrap_or_else(PoisonError::into_inner);
        if guard.len() >= MOMENT_CACHE_CAPACITY {
            guard.clear();
        }
        guard.insert(key, built.clone());
        built
    }

    /// The moments of the trend with design rows `basis` and inverse Gram matrix `inverse`.
    pub(super) fn new(basis: &[Vec<f64>], inverse: &[Vec<f64>]) -> Self {
        let n = basis.len();
        let size = inverse.len();
        // P = X G, so the hat matrix is H = P Xᵀ.
        let projected: Vec<Vec<f64>> = basis
            .iter()
            .map(|row| {
                inverse
                    .iter()
                    .map(|inverse_row| inverse_row.iter().zip(row).map(|(a, b)| a * b).sum())
                    .collect()
            })
            .collect();
        let hat = |i: usize, j: usize| -> f64 {
            projected[i].iter().zip(&basis[j]).map(|(a, b)| a * b).sum()
        };
        let mut hat_matrix = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                hat_matrix[i * n + j] = hat(i, j);
            }
        }
        let h = |i: usize, j: usize| hat_matrix[i * n + j];
        let mut diagonals = Vec::with_capacity(MAX_ORDER + 1);
        for lag in 0..=MAX_ORDER {
            // S = identity for lag 0, else one half on the two diagonals at distance `lag`.
            let shift = |i: usize, j: usize| -> f64 {
                if lag == 0 {
                    f64::from(u8::from(i == j))
                } else if i.abs_diff(j) == lag {
                    0.5
                } else {
                    0.0
                }
            };
            // Xᵀ S X.
            let mut inner = vec![vec![0.0; size]; size];
            for i in 0..n {
                for j in i.saturating_sub(lag)..(i + lag + 1).min(n) {
                    let s = shift(i, j);
                    if s != 0.0 {
                        for (a, inner_row) in inner.iter_mut().enumerate() {
                            for (b, cell) in inner_row.iter_mut().enumerate() {
                                *cell += basis[i][a] * s * basis[j][b];
                            }
                        }
                    }
                }
            }
            // (H S)[i][j] and (S H)[i][j] by shifting H.
            let hat_shift = |i: usize, j: usize| -> f64 {
                if lag == 0 {
                    return h(i, j);
                }
                let mut total = 0.0;
                if j >= lag {
                    total += 0.5 * h(i, j - lag);
                }
                if j + lag < n {
                    total += 0.5 * h(i, j + lag);
                }
                total
            };
            let shift_hat = |i: usize, j: usize| -> f64 {
                if lag == 0 {
                    return h(i, j);
                }
                let mut total = 0.0;
                if i >= lag {
                    total += 0.5 * h(i - lag, j);
                }
                if i + lag < n {
                    total += 0.5 * h(i + lag, j);
                }
                total
            };
            let mut sums = vec![0.0; n];
            for i in 0..n {
                for j in i..n {
                    let mut low_rank = 0.0;
                    for (a, inner_row) in inner.iter().enumerate() {
                        for (b, value) in inner_row.iter().enumerate() {
                            low_rank += projected[i][a] * value * projected[j][b];
                        }
                    }
                    let entry = shift(i, j) - hat_shift(i, j) - shift_hat(i, j) + low_rank;
                    sums[j - i] += if i == j { entry } else { 2.0 * entry };
                }
            }
            diagonals.push(sums);
        }
        Self { diagonals: diagonals.into() }
    }

    /// Expected lag-`lag` product sum of the residuals of a unit-variance error with
    /// autocorrelations `rho`.
    fn expected(&self, lag: usize, rho: &[f64]) -> f64 {
        self.diagonals[lag].iter().zip(rho).map(|(d, r)| d * r).sum()
    }
}

/// The AR(`p`) model of one error series.
pub(super) struct ArModel {
    /// Autoregressive coefficients.
    pub(super) phi: Vec<f64>,
    /// Partial autocorrelations the coefficients follow from.
    pub(super) partial: Vec<f64>,
    /// Covariance of the partial autocorrelation estimates.
    pub(super) covariance: Vec<Vec<f64>>,
    /// Lower Cholesky factor of the covariance, for posterior draws.
    pub(super) factor: Vec<Vec<f64>>,
}

/// An AR model with its whitened innovations.
pub(super) struct FittedErrors {
    pub(super) model: ArModel,
    /// Centered innovations in time order, left side then right side, on the scale of the
    /// fitted model.
    pub(super) innovations: Vec<f64>,
}

/// Coefficients of the AR model with partial autocorrelations `partial` (Durbin-Levinson
/// step-up).
pub(super) fn coefficients_from_partial(partial: &[f64]) -> Vec<f64> {
    let mut phi: Vec<f64> = Vec::with_capacity(partial.len());
    for &pi in partial {
        let previous = phi.clone();
        phi = previous.iter().zip(previous.iter().rev()).map(|(a, b)| a - pi * b).collect();
        phi.push(pi);
    }
    phi
}

/// Autocorrelations `rho[0..len]` and coefficients of the AR model with partial
/// autocorrelations `partial`.
fn autocorrelations(partial: &[f64], len: usize) -> (Vec<f64>, Vec<f64>) {
    let p = partial.len();
    let len = len.max(p + 1);
    let mut rho = vec![0.0; len];
    rho[0] = 1.0;
    let mut phi: Vec<f64> = Vec::new();
    for (k, &pi) in partial.iter().enumerate() {
        let predicted: f64 = phi.iter().enumerate().map(|(j, c)| c * rho[k - j]).sum();
        let explained: f64 = phi.iter().enumerate().map(|(j, c)| c * rho[j + 1]).sum();
        rho[k + 1] = predicted + pi * (1.0 - explained);
        phi = coefficients_from_partial(&partial[..=k]);
    }
    for h in p + 1..len {
        rho[h] = phi.iter().enumerate().map(|(j, c)| c * rho[h - 1 - j]).sum();
    }
    (rho, phi)
}

/// Partial autocorrelations and the prediction-error filter of every order from the
/// autocorrelations `r[0..p]` at lags `1..=p` (Durbin-Levinson).
fn levinson(r: &[f64]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let mut partial = Vec::with_capacity(r.len());
    let mut filters: Vec<Vec<f64>> = Vec::with_capacity(r.len());
    let mut phi: Vec<f64> = Vec::new();
    for (k, &lag_k) in r.iter().enumerate() {
        let predicted: f64 = phi.iter().enumerate().map(|(j, c)| c * r[k - 1 - j]).sum();
        let explained: f64 = phi.iter().enumerate().map(|(j, c)| c * r[j]).sum();
        let pi = ((lag_k - predicted) / (1.0 - explained)).clamp(-MAX_PARTIAL, MAX_PARTIAL);
        partial.push(pi);
        phi = coefficients_from_partial(&partial);
        filters.push(phi.clone());
    }
    (partial, filters)
}

/// Expected residual product sums at lags `0..=p` of a unit-variance AR error, over both sides.
fn expected_moments(moments: &[LagMoments; 2], partial: &[f64]) -> Vec<f64> {
    let rows = moments[0].diagonals[0].len().max(moments[1].diagonals[0].len());
    let (rho, _) = autocorrelations(partial, rows);
    (0..=partial.len())
        .map(|lag| moments.iter().map(|side| side.expected(lag, &rho)).sum())
        .collect()
}

/// Expected residual autocorrelations at lags `1..=p`.
fn expected_ratios(moments: &[LagMoments; 2], partial: &[f64]) -> Vec<f64> {
    let expected = expected_moments(moments, partial);
    expected[1..].iter().map(|moment| moment / expected[0]).collect()
}

/// Jacobian of the expected residual autocorrelations in the partial autocorrelations.
fn ratio_jacobian(moments: &[LagMoments; 2], partial: &[f64]) -> Vec<Vec<f64>> {
    let p = partial.len();
    let mut jacobian = vec![vec![0.0; p]; p];
    for j in 0..p {
        let mut above = partial.to_vec();
        let mut below = partial.to_vec();
        above[j] = (above[j] + DERIVATIVE_STEP).min(MAX_PARTIAL);
        below[j] = (below[j] - DERIVATIVE_STEP).max(-MAX_PARTIAL);
        let high = expected_ratios(moments, &above);
        let low = expected_ratios(moments, &below);
        for k in 0..p {
            jacobian[k][j] = (high[k] - low[k]) / (above[j] - below[j]);
        }
    }
    jacobian
}

/// Partial autocorrelations whose expected residual autocorrelations equal `observed[0..p]`:
/// damped Newton steps from the partial autocorrelations of the observed ones.
fn solve_partial(moments: &[LagMoments; 2], observed: &[f64], p: usize) -> Option<Vec<f64>> {
    let (start, _) = levinson(&observed[..p]);
    let residual = |partial: &[f64]| -> Vec<f64> {
        expected_ratios(moments, partial).iter().zip(observed).map(|(e, o)| e - o).collect()
    };
    let size = |values: &[f64]| values.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let mut partial = start;
    for _ in 0..40 {
        let current = residual(&partial);
        if !size(&current).is_finite() {
            return None;
        }
        if size(&current) < 1e-10 {
            break;
        }
        let Some(inverse) = crate::util::invert_rows(&ratio_jacobian(moments, &partial)) else {
            break;
        };
        let step: Vec<f64> = inverse.iter().map(|row| super::dot(row, &current)).collect();
        let mut damping = 1.0;
        let mut next = partial.clone();
        while damping > 1e-3 {
            next = partial
                .iter()
                .zip(&step)
                .map(|(x, s)| (x - damping * s).clamp(-MAX_PARTIAL, MAX_PARTIAL))
                .collect();
            if size(&residual(&next)) < size(&current) {
                break;
            }
            damping *= 0.5;
        }
        partial = next;
    }
    partial.iter().all(|x| x.is_finite()).then_some(partial)
}

/// Lower Cholesky factor of a symmetric matrix, or `None` when it is not positive definite.
fn cholesky(matrix: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = matrix.len();
    let mut factor = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let sum: f64 = (0..j).map(|k| factor[i][k] * factor[j][k]).sum();
            if i == j {
                let value = matrix[i][i] - sum;
                if value <= 0.0 || !value.is_finite() {
                    return None;
                }
                factor[i][i] = value.sqrt();
            } else {
                factor[i][j] = (matrix[i][j] - sum) / factor[j][j];
            }
        }
    }
    Some(factor)
}

/// Large-sample covariance of the corrected partial autocorrelations: the observed residual
/// autocorrelations have Bartlett's covariance at the fitted autocorrelations, and the
/// partial autocorrelations are their inverse image under the correction (delta method).
/// For `p = 1` it is `(1 - r²) / pairs` over the squared slope of the expectation.
fn partial_covariance(
    moments: &[LagMoments; 2],
    partial: &[f64],
    observed_partial: &[f64],
    pairs: f64,
) -> Vec<Vec<f64>> {
    let p = partial.len();
    let diagonal = || -> Vec<Vec<f64>> {
        (0..p)
            .map(|i| {
                (0..p)
                    .map(|j| {
                        if i == j {
                            (1.0 - observed_partial[i].powi(2)).max(0.0) / pairs
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect()
    };
    let (rho, _) = autocorrelations(partial, 2 * BARTLETT_LAGS);
    // Lags are at most a few dozen, far inside `isize`.
    let signed = |value: usize| isize::try_from(value).unwrap_or(isize::MAX);
    let at = |lag: usize, other: isize| -> f64 {
        let index = (signed(lag) + other).unsigned_abs();
        rho.get(index).copied().unwrap_or(0.0)
    };
    let mut bartlett = vec![vec![0.0; p]; p];
    for (i, row) in bartlett.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            let (i, j) = (i + 1, j + 1);
            *cell = (1..BARTLETT_LAGS)
                .map(|h| {
                    let left = at(h, signed(i)) + at(h, -signed(i)) - 2.0 * rho[i] * rho[h];
                    let right = at(h, signed(j)) + at(h, -signed(j)) - 2.0 * rho[j] * rho[h];
                    left * right
                })
                .sum();
        }
    }
    let Some(inverse) = crate::util::invert_rows(&ratio_jacobian(moments, partial)) else {
        return diagonal();
    };
    let mut covariance = vec![vec![0.0; p]; p];
    for i in 0..p {
        for j in 0..p {
            let mut total = 0.0;
            for a in 0..p {
                for b in 0..p {
                    total += inverse[i][a] * bartlett[a][b] * inverse[j][b];
                }
            }
            covariance[i][j] = total / pairs;
        }
    }
    let finite = covariance.iter().flatten().all(|v| v.is_finite());
    if finite && cholesky(&covariance).is_some() { covariance } else { diagonal() }
}

/// The corrected AR model of the residuals of one series about its per-side trends, with
/// its whitened, centered and rescaled innovations. `None` when no model can be fit.
pub(super) fn fit_error_model(
    moments: &[LagMoments; 2],
    residuals: &[Vec<f64>; 2],
) -> Option<FittedErrors> {
    let rows: usize = residuals.iter().map(Vec::len).sum();
    let square: f64 = residuals.iter().flatten().map(|r| r * r).sum();
    if square <= 0.0 || !square.is_finite() {
        return None;
    }
    let observed: Vec<f64> = (1..=MAX_ORDER)
        .map(|lag| {
            residuals
                .iter()
                .filter(|side| side.len() > lag)
                .map(|side| side.iter().zip(&side[lag..]).map(|(a, b)| a * b).sum::<f64>())
                .sum::<f64>()
                / square
        })
        .collect();
    let count = rows as f64;
    let mut best: Option<(f64, Vec<f64>, f64)> = None;
    for p in 1..=MAX_ORDER {
        let Some(partial) = solve_partial(moments, &observed, p) else { continue };
        let expected = expected_moments(moments, &partial)[0];
        if expected <= 0.0 || !expected.is_finite() {
            continue;
        }
        let innovation_variance =
            square / expected * partial.iter().map(|pi| 1.0 - pi * pi).product::<f64>();
        let criterion = count * innovation_variance.ln() + p as f64 * count.ln();
        if criterion.is_finite() && best.as_ref().is_none_or(|(c, _, _)| criterion < *c) {
            best = Some((criterion, partial, innovation_variance));
        }
    }
    let (_, partial, innovation_variance) = best?;
    let p = partial.len();
    let (observed_partial, filters) = levinson(&observed);
    let pairs = (rows - 2).max(1) as f64;
    let covariance = partial_covariance(moments, &partial, &observed_partial, pairs);
    let factor = cholesky(&covariance)?;
    // Whiten the residuals with the uncorrected filter of order `p`, which leaves the
    // innovations empirically white, using the lower-order filters at a side's first rows.
    let variance_ratios: Vec<f64> = observed_partial
        .iter()
        .scan(1.0, |product, pi| {
            *product *= 1.0 - pi * pi;
            Some(*product)
        })
        .collect();
    let mut innovations = Vec::with_capacity(rows);
    for side in residuals {
        let start = innovations.len();
        for (t, value) in side.iter().enumerate() {
            let order = t.min(p);
            let predicted: f64 = if order == 0 {
                0.0
            } else {
                filters[order - 1].iter().enumerate().map(|(j, c)| c * side[t - 1 - j]).sum()
            };
            let relative = if order == 0 { 1.0 } else { variance_ratios[order - 1] };
            innovations.push((value - predicted) * (variance_ratios[p - 1] / relative).sqrt());
        }
        let mean = innovations[start..].iter().sum::<f64>() / side.len() as f64;
        for value in &mut innovations[start..] {
            *value -= mean;
        }
    }
    let variance = innovations.iter().map(|e| e * e).sum::<f64>() / innovations.len() as f64;
    let scale = if variance > 0.0 { (innovation_variance / variance).sqrt() } else { 0.0 };
    for value in &mut innovations {
        *value *= scale;
    }
    let phi = coefficients_from_partial(&partial);
    Some(FittedErrors { model: ArModel { phi, partial, covariance, factor }, innovations })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cubic monomial rows over `n` equally spaced points in `[-1, 1]` and the inverse of
    /// their Gram matrix, by Gauss-Jordan elimination.
    fn design(n: usize, scale: f64) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let basis: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let d = scale * (2.0 * i as f64 / (n as f64 - 1.0) - 1.0);
                vec![1.0, d, d * d, d * d * d]
            })
            .collect();
        let size = 4;
        let mut aug = vec![vec![0.0; 2 * size]; size];
        for (a, row) in aug.iter_mut().enumerate() {
            for b in 0..size {
                row[b] = basis.iter().map(|x| x[a] * x[b]).sum();
            }
            row[size + a] = 1.0;
        }
        for col in 0..size {
            let pivot = (col..size)
                .max_by(|&p, &q| aug[p][col].abs().total_cmp(&aug[q][col].abs()))
                .expect("pivot");
            aug.swap(col, pivot);
            let lead = aug[col][col];
            aug[col].iter_mut().for_each(|v| *v /= lead);
            for r in 0..size {
                if r != col {
                    let factor = aug[r][col];
                    let pivot_row = aug[col].clone();
                    aug[r].iter_mut().zip(&pivot_row).for_each(|(v, p)| *v -= factor * p);
                }
            }
        }
        (basis, aug.into_iter().map(|row| row[size..].to_vec()).collect())
    }

    fn bits(moments: &LagMoments) -> Vec<Vec<u64>> {
        moments.diagonals.iter().map(|d| d.iter().map(|v| v.to_bits()).collect()).collect()
    }

    #[test]
    fn cached_moments_equal_a_fresh_build_bit_for_bit() {
        let (basis, inverse) = design(17, 1.0);
        let fresh = LagMoments::new(&basis, &inverse);
        let first = LagMoments::cached(&basis, &inverse);
        let second = LagMoments::cached(&basis, &inverse);
        assert_eq!(bits(&fresh), bits(&first));
        assert_eq!(bits(&fresh), bits(&second));
    }

    #[test]
    fn a_changed_design_never_returns_a_stale_entry() {
        let (basis, inverse) = design(19, 1.0);
        let before = LagMoments::cached(&basis, &inverse);
        // Rescaled design, one more row, and a one-ulp change of one entry each miss.
        let (scaled_basis, scaled_inverse) = design(19, 0.5);
        let mut nudged = basis.clone();
        nudged[3][1] = f64::from_bits(nudged[3][1].to_bits() + 1);
        let (longer_basis, longer_inverse) = design(20, 1.0);
        for (b, g) in [
            (&scaled_basis, &scaled_inverse),
            (&nudged, &inverse),
            (&longer_basis, &longer_inverse),
        ] {
            let fresh = LagMoments::new(b, g);
            assert_eq!(bits(&LagMoments::cached(b, g)), bits(&fresh));
        }
        assert_eq!(bits(&LagMoments::cached(&basis, &inverse)), bits(&before));
    }
}
