//! Borrow-scoped analytic partial correlation for repeated discovery queries.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;

use super::analytic::{analytic_parcorr_ci, analytic_parcorr_pvalue};
use super::{CiQuery, CiResult, ConfidenceMethod};
use crate::gram::{form_xtx, invert_square};

/// Cross-products tied to an immutable, borrowed CI cohort.
///
/// A session cannot outlive its columns. Unlike a pointer-keyed workspace cache,
/// it cannot answer queries on replaced or mutated data. Unsupported or poorly
/// conditioned queries return `None` so the caller runs its original CI test.
pub struct CovarianceCiSession<'a> {
    columns: &'a [&'a [f64]],
    correlation: Vec<f64>,
    error: Vec<f64>,
}

impl<'a> CovarianceCiSession<'a> {
    pub(super) fn prepare(columns: &'a [&'a [f64]], ctx: &ExecutionContext) -> Option<Self> {
        let d = columns.len();
        let n = columns.first()?.len();
        // Preparation must amortize over a search, and fit both transient and
        // retained storage. Budgeted execution retains the existing route.
        let cells = n.checked_mul(d)?.checked_add(d.checked_mul(d)?)?;
        let bytes = cells.checked_add(d)?.checked_mul(size_of::<f64>())?;
        if n < 128
            || !(3..=64).contains(&d)
            || cells > 4_000_000
            || ctx.kernel_policy.force_scalar
            || !ctx.cache_policy.enabled
            || ctx.memory.hard_limit_bytes.is_some()
            || ctx.memory.soft_limit_bytes.is_some()
            || ctx.admission.byte_budget() != u64::MAX
            || ctx.cache_policy.max_bytes.is_some_and(|limit| bytes as u64 > limit)
        {
            return None;
        }
        let mut scaled = vec![0.0; n * d];
        let mut error = vec![0.0; d];
        for (c, column) in columns.iter().enumerate() {
            if ctx.cancellation.is_cancelled()
                || column.len() != n
                || column.iter().any(|v| !v.is_finite())
            {
                return None;
            }
            let mean = column.iter().sum::<f64>() / n as f64;
            let ss = column.iter().map(|v| (v - mean).powi(2)).sum::<f64>();
            let rms = (ss / n as f64).sqrt();
            let mean_abs = column.iter().map(|v| v.abs()).sum::<f64>() / n as f64;
            // Include summation, centering and scaling error. Very large
            // offsets, constants and over/underflow use residualization.
            error[c] = 64.0 * (n as f64 + 8.0) * f64::EPSILON * (1.0 + mean_abs / rms);
            if !ss.is_finite() || ss <= 0.0 || error[c] > 1e-6 {
                error[c] = f64::INFINITY;
                continue;
            }
            let norm = ss.sqrt();
            for (out, value) in scaled[c * n..(c + 1) * n].iter_mut().zip(*column) {
                *out = (value - mean) / norm;
            }
        }
        if ctx.cancellation.is_cancelled() {
            return None;
        }
        let mut correlation = vec![0.0; d * d];
        form_xtx(&scaled, n, d, &mut correlation);
        Some(Self { columns, correlation, error })
    }

    /// Answer an analytic query when its conditioning and decision margins are
    /// safe. `thresholds` must be sorted and contain every downstream rejection
    /// threshold, including possible multiplicity corrections. A rejected fast
    /// query must be evaluated by the original residual-based test.
    #[must_use]
    pub fn try_test(
        &self,
        query: CiQuery,
        z_flat: &[usize],
        confidence: ConfidenceMethod,
        thresholds: &[f64],
    ) -> Option<CiResult> {
        let d = self.columns.len();
        let n = self.columns[0].len();
        let z = z_flat.get(query.z_start..query.z_start.checked_add(query.z_len)?)?;
        let (x, y, k) = (query.x, query.y, z.len());
        if x >= d
            || y >= d
            || x == y
            || k > 12
            || n <= k + 2
            || z.iter().any(|&c| c >= d || c == x || c == y)
        {
            return None;
        }
        let delta = z.iter().fold(self.error[x].max(self.error[y]), |e, &c| e.max(self.error[c]));
        if !delta.is_finite() {
            return None;
        }
        let at = |i: usize, j: usize| self.correlation[i * d + j];
        let (mut vx, mut vy, mut cov) = (at(x, x), at(y, y), at(x, y));
        let mut bound = delta;
        if k > 0 {
            let a: Vec<_> = z.iter().flat_map(|&i| z.iter().map(move |&j| at(i, j))).collect();
            let inverse = invert_square(&a, k)?;
            let norm = inverse
                .chunks_exact(k)
                .map(|row| row.iter().map(|v| v.abs()).sum::<f64>())
                .fold(0.0_f64, f64::max);
            let mut defect = 0.0_f64;
            for i in 0..k {
                let mut row_defect = 0.0;
                for j in 0..k {
                    let product: f64 = (0..k).map(|h| a[i * k + h] * inverse[h * k + j]).sum();
                    row_defect += ((if i == j { 1.0 } else { 0.0 }) - product).abs();
                }
                defect = defect.max(row_defect);
            }
            let perturbation = defect + k as f64 * norm * (delta + 16.0 * k as f64 * f64::EPSILON);
            if !perturbation.is_finite() || perturbation >= 0.01 {
                return None;
            }
            bound = 8.0 * (k * k) as f64 * delta * (1.0 + norm).powi(2)
                + k as f64 * norm * perturbation / (1.0 - perturbation);
            for i in 0..k {
                for j in 0..k {
                    let inverse_ij = inverse[i * k + j];
                    vx -= at(x, z[i]) * inverse_ij * at(x, z[j]);
                    vy -= at(y, z[i]) * inverse_ij * at(y, z[j]);
                    cov -= at(x, z[i]) * inverse_ij * at(y, z[j]);
                }
            }
        }
        if !vx.is_finite() || !vy.is_finite() || vx.min(vy) <= (100.0 * bound).max(1e-8) {
            return None;
        }
        let scale = (vx * vy).sqrt();
        let r = cov / scale;
        let r_error = 4.0 * bound * (1.0 + r.abs()) / scale;
        if !r.is_finite() || r.abs() > 1.0 || r_error > 1e-6 {
            return None;
        }
        let df = (n - k - 2) as f64;
        let p_lo = analytic_parcorr_pvalue((r.abs() + r_error).min(1.0), df);
        let p_hi = analytic_parcorr_pvalue((r.abs() - r_error).max(0.0), df);
        let first = thresholds.partition_point(|&t| t < p_lo);
        if thresholds.get(first).is_some_and(|&t| t <= p_hi) {
            return None;
        }
        let ci = match confidence {
            ConfidenceMethod::None => None,
            ConfidenceMethod::Analytic { level } => Some(analytic_parcorr_ci(r, df, level)),
        };
        Some(CiResult { statistic: r, p_value: analytic_parcorr_pvalue(r, df), df, ci })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CiPreparationPlan, CiWorkspace, ConditionalIndependenceTest, PartialCorrelation};

    #[test]
    fn exhaustive_small_cohort_matches_residual_tests_and_guards_decisions() {
        let data: Vec<Vec<f64>> = (0..6)
            .map(|c| {
                (0..512)
                    .map(|r| {
                        let t = f64::from(r);
                        (t * (0.113 + f64::from(c) * 0.071)).sin()
                            + 0.3 * (t * 0.017).cos()
                            + 0.2 * (t * (0.231 + f64::from(c) * 0.03)).cos()
                    })
                    .collect()
            })
            .collect();
        let cols: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
        let ctx = ExecutionContext::production(19, 1);
        let session = CovarianceCiSession::prepare(&cols, &ctx).unwrap();
        let reference = ExecutionContext::for_tests(19);
        let test = PartialCorrelation;
        let prepared = test.prepare(&cols, &CiPreparationPlan::default(), &reference).unwrap();
        let mut ws = CiWorkspace::default();
        let mut fast = 0;
        for x in 0..6 {
            for y in x + 1..6 {
                for mask in 0u32..64 {
                    if mask.count_ones() > 3 || mask & ((1 << x) | (1 << y)) != 0 {
                        continue;
                    }
                    let z: Vec<_> = (0..6).filter(|&c| mask & (1 << c) != 0).collect();
                    let q = CiQuery { x, y, z_start: 0, z_len: z.len() };
                    let expected = test.test(&prepared, &cols, q, &z, &mut ws, &reference).unwrap();
                    if let Some(actual) =
                        session.try_test(q, &z, ConfidenceMethod::default(), &[0.05])
                    {
                        fast += 1;
                        assert!((actual.statistic - expected.statistic).abs() < 1e-10);
                        assert!((actual.p_value - expected.p_value).abs() < 1e-9);
                        let (lo, hi) = actual.ci.unwrap();
                        let (elo, ehi) = expected.ci.unwrap();
                        assert!((lo - elo).abs().max((hi - ehi).abs()) < 1e-9);
                        // An actual decision at this p-value must be recomputed.
                        assert!(
                            session
                                .try_test(q, &z, ConfidenceMethod::None, &[actual.p_value])
                                .is_none()
                        );
                    }
                }
            }
        }
        assert!(fast > 150, "preparation must actually accelerate the cohort: {fast}");
    }

    #[test]
    fn constants_collinearity_offsets_and_budgets_use_reference() {
        let x: Vec<_> = (0..256).map(|i| (f64::from(i) * 0.17).sin()).collect();
        let y: Vec<_> = (0..256).map(|i| (f64::from(i) * 0.23).cos()).collect();
        let offset: Vec<_> = x.iter().map(|v| v + 1e14).collect();
        let cols: [&[f64]; 4] = [&x, &y, &x, &offset];
        let mut ctx = ExecutionContext::production(19, 1);
        let session = CovarianceCiSession::prepare(&cols, &ctx).unwrap();
        let q = CiQuery { x: 0, y: 1, z_start: 0, z_len: 1 };
        assert!(session.try_test(q, &[2], ConfidenceMethod::None, &[0.05]).is_none());
        assert!(session.try_test(q, &[3], ConfidenceMethod::None, &[0.05]).is_none());
        ctx.cache_policy = antecedent_core::CachePolicy::enabled(Some(0));
        assert!(CovarianceCiSession::prepare(&cols, &ctx).is_none());
        ctx.cache_policy = antecedent_core::CachePolicy::enabled(None);
        ctx.memory.hard_limit_bytes = Some(1);
        assert!(CovarianceCiSession::prepare(&cols, &ctx).is_none());
    }
}
