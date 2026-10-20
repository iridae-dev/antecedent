//! Exact-to-grid law of an integer-weighted combination of independent scalar posteriors.
//!
//! A contrast `sum_k w_k x_k` of independent posterior quantities `x_k` (arm means, finite-
//! population arm means) has no closed-form law for Beta or Student-t marginals, but its law is
//! the convolution of the marginal laws. Each marginal is placed on a common lattice of cell
//! width `h`; the contrast weights are multiples of one half, `w_k = c_k / 2` with `c_k` an
//! integer and `sum c_k = 0`, so the contrast of the lattice indices is `h * sum c_k i_k / 2`
//! exactly. Because the weights sum to zero the half-cell offset of a continuous marginal (its
//! mass sits at the cell midpoint) cancels, and a marginal that is already discrete on the
//! lattice (the finite-population Beta-Binomial) is carried without any discretization.
//!
//! A continuous marginal is discretized by differences of its CDF over cells, so the only
//! numerical error is within-cell: a variance inflation of about `(1 / cells_per_scale)^2 / 12`
//! of the marginal variance, with `cells_per_scale = 16`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_wrap,
    reason = "lattice indices and counts are bounded by MAX_TERM_CELLS and the convolution work bound"
)]

use antecedent_core::reason_code;

use crate::error::EstimationError;
use crate::util::refuse;

/// Lattice cells per smallest marginal scale.
pub(crate) const CELLS_PER_SCALE: f64 = 16.0;
/// Most cells one marginal may occupy.
pub(crate) const MAX_TERM_CELLS: usize = 60_000;
/// Most multiply-adds one convolution chain may perform.
pub(crate) const MAX_CONVOLUTION_WORK: u64 = 400_000_000;

/// A marginal law on the lattice: mass `pmf[j]` at lattice index `start + j`.
pub(crate) struct LatticeTerm {
    pub(crate) start: i64,
    pub(crate) pmf: Vec<f64>,
}

/// The law of `sum_k c_k i_k`: mass `pmf[j]` at integer `offset + j`.
pub(crate) struct LatticeLaw {
    offset: i64,
    pmf: Vec<f64>,
    /// Contrast value of one unit of the integer, `h / 2`.
    unit: f64,
    /// Whether the marginals were discretized (mass spread within a cell) or exact.
    continuous: bool,
    /// Spacing of the integer lattice carrying mass: the greatest common divisor of the
    /// coefficients, so a discretized cell of one marginal is `spacing` integers wide.
    spacing: f64,
}

fn limit(detail: &str, message: &str) -> EstimationError {
    refuse(reason_code!("route_not_supported"), detail, message)
}

/// Discretize a continuous law with CDF `cdf` over `[lo, hi]` onto cells of width `h`.
///
/// `cdf` returns `None` on a numerical failure.
#[allow(
    clippy::cast_possible_truncation,
    reason = "cell indices are bounded by the finite window over the positive width h before the cast"
)]
pub(crate) fn discretize(
    cdf: &dyn Fn(f64) -> Option<f64>,
    lo: f64,
    hi: f64,
    h: f64,
) -> Result<LatticeTerm, EstimationError> {
    let numerical = || {
        limit(
            "posterior_lattice.numerical_failure",
            "a marginal posterior CDF could not be evaluated on the window",
        )
    };
    if !(lo.is_finite() && hi.is_finite() && h.is_finite() && h > 0.0 && hi >= lo) {
        return Err(numerical());
    }
    let first = (lo / h).floor() as i64;
    let last = ((hi / h).ceil() as i64).max(first + 1);
    let cells = usize::try_from(last - first).unwrap_or(usize::MAX);
    if cells > MAX_TERM_CELLS {
        return Err(limit(
            "posterior_lattice.too_concentrated",
            "the posteriors differ in scale by too much for the contrast lattice; \
             contrast the arms pairwise or use a coarser design",
        ));
    }
    let mut previous = cdf(first as f64 * h).ok_or_else(numerical)?;
    let mut pmf = Vec::with_capacity(cells);
    for index in first..last {
        let next = cdf((index + 1) as f64 * h).ok_or_else(numerical)?;
        pmf.push((next - previous).max(0.0));
        previous = next;
    }
    normalize(&mut pmf).ok_or_else(numerical)?;
    Ok(LatticeTerm { start: first, pmf })
}

/// Scale a mass vector to sum to one; `None` when it has no finite positive mass.
pub(crate) fn normalize(pmf: &mut [f64]) -> Option<()> {
    let total: f64 = pmf.iter().sum();
    if !(total.is_finite() && total > 0.0) {
        return None;
    }
    pmf.iter_mut().for_each(|m| *m /= total);
    Some(())
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Law of `sum_k c_k i_k` for independent lattice terms; `unit` is the contrast value of one unit.
pub(crate) fn convolve_terms(
    terms: &[(i64, LatticeTerm)],
    unit: f64,
    continuous: bool,
) -> Result<LatticeLaw, EstimationError> {
    let spacing = terms.iter().fold(0_u64, |g, (c, _)| gcd(g, c.unsigned_abs())).max(1);
    let mut offset = 0_i64;
    let mut pmf = vec![1.0];
    let mut work = 0_u64;
    for (coefficient, term) in terms {
        let span = i64::try_from(term.pmf.len() - 1).unwrap_or(i64::MAX);
        let low = (coefficient * span).min(0);
        let width = usize::try_from(coefficient.unsigned_abs())
            .unwrap_or(usize::MAX)
            .saturating_mul(term.pmf.len() - 1);
        work = work.saturating_add((pmf.len() as u64).saturating_mul(term.pmf.len() as u64));
        if work > MAX_CONVOLUTION_WORK {
            return Err(limit(
                "posterior_lattice.work_limit",
                "the contrast convolution exceeds its work bound; reduce the number of units or arms",
            ));
        }
        let mut next = vec![0.0; pmf.len() + width];
        for (i, mass) in term.pmf.iter().enumerate() {
            let shift = usize::try_from(coefficient * i as i64 - low).unwrap_or(0);
            for (j, acc) in pmf.iter().enumerate() {
                next[j + shift] += acc * mass;
            }
        }
        offset += coefficient * term.start + low;
        pmf = next;
    }
    Ok(LatticeLaw { offset, pmf, unit, continuous, spacing: spacing as f64 })
}

impl LatticeLaw {
    fn value(&self, index: usize) -> f64 {
        (self.offset + index as i64) as f64 * self.unit
    }

    /// Posterior mean.
    pub(crate) fn mean(&self) -> f64 {
        self.pmf.iter().enumerate().map(|(j, m)| m * self.value(j)).sum()
    }

    /// Posterior standard deviation (discretized marginals carry the within-cell inflation).
    pub(crate) fn standard_deviation(&self) -> f64 {
        let mean = self.mean();
        self.pmf
            .iter()
            .enumerate()
            .map(|(j, m)| m * (self.value(j) - mean).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Posterior probability that the contrast is positive. For a discretized law the mass on
    /// the zero cell counts half; for an exact discrete law only strictly positive values count.
    pub(crate) fn probability_positive(&self) -> f64 {
        let mut positive = 0.0;
        for (j, m) in self.pmf.iter().enumerate() {
            let index = self.offset + j as i64;
            if index > 0 {
                positive += m;
            } else if index == 0 && self.continuous {
                positive += 0.5 * m;
            }
        }
        positive.clamp(0.0, 1.0)
    }

    /// Quantile at `q` in `(0, 1)`: interpolated inside its cell for a discretized law, the
    /// smallest value with CDF at least `q` for an exact discrete law.
    pub(crate) fn quantile(&self, q: f64) -> f64 {
        let mut before = 0.0;
        for (j, m) in self.pmf.iter().enumerate() {
            if before + m >= q && *m > 0.0 {
                let centre = (self.offset + j as i64) as f64;
                let place = if self.continuous {
                    centre + self.spacing * ((q - before) / m - 0.5)
                } else {
                    centre
                };
                return place * self.unit;
            }
            before += m;
        }
        self.value(self.pmf.len() - 1)
    }

    /// The quantiles at `(i + 1/2) / count` for `i = 0..count`, ascending: a deterministic
    /// quantile grid of the law, in one pass over the mass.
    pub(crate) fn quantile_grid(&self, count: usize) -> Vec<f64> {
        let mut grid = Vec::with_capacity(count);
        let mut before = 0.0;
        let mut cell = 0_usize;
        for i in 0..count {
            let q = (i as f64 + 0.5) / count as f64;
            while cell < self.pmf.len()
                && !(before + self.pmf.get(cell).copied().unwrap_or(0.0) >= q
                    && self.pmf.get(cell).copied().unwrap_or(0.0) > 0.0)
            {
                before += self.pmf.get(cell).copied().unwrap_or(0.0);
                cell += 1;
            }
            if cell >= self.pmf.len() {
                grid.push(self.value(self.pmf.len().saturating_sub(1)));
                continue;
            }
            let m = self.pmf.get(cell).copied().unwrap_or(0.0);
            let centre = (self.offset + cell as i64) as f64;
            let place = if self.continuous {
                centre + self.spacing * ((q - before) / m - 0.5)
            } else {
                centre
            };
            grid.push(place * self.unit);
        }
        grid
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn difference_of_two_exact_discrete_laws_is_the_cross_correlation() {
        // x0 on {0, 1} with masses (0.25, 0.75); x1 on {0, 1} with (0.5, 0.5). Contrast x1 - x0
        // in half units: coefficients (-2, +2) would be the weights (-1, +1).
        let x0 = LatticeTerm { start: 0, pmf: vec![0.25, 0.75] };
        let x1 = LatticeTerm { start: 0, pmf: vec![0.5, 0.5] };
        let law = convolve_terms(&[(-2, x0), (2, x1)], 0.5, false).unwrap();
        // P(x1 - x0 = -1) = 0.5 * 0.75, 0 = 0.5 * 0.25 + 0.5 * 0.75, +1 = 0.5 * 0.25.
        let at = |value: i64| {
            let index = usize::try_from(2 * value - law.offset).unwrap();
            law.pmf[index]
        };
        assert!((at(-1) - 0.375).abs() < 1e-15);
        assert!((at(0) - 0.5).abs() < 1e-15);
        assert!((at(1) - 0.125).abs() < 1e-15);
        assert!((law.mean() - (0.5 - 0.75)).abs() < 1e-15);
        assert!((law.probability_positive() - 0.125).abs() < 1e-15);
    }
}
