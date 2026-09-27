//! Additive binary marginal structural model with subject-level stabilized IPTW.
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// Point coefficients and pointwise subject-clustered CR1 sandwich errors.
#[derive(Clone, Debug, PartialEq)]
pub struct MsmSummary {
    /// Weighted additive model intercept.
    pub intercept: f64,
    /// Subject-clustered CR1 standard error for the intercept.
    pub intercept_standard_error: f64,
    /// Weighted coefficients in treatment-period order.
    pub period_effects: Vec<f64>,
    /// Pointwise CR1 standard errors in treatment-period order.
    pub standard_errors: Vec<f64>,
    /// Kish effective sample size among observed terminal outcomes.
    pub effective_sample_size: f64,
    /// Largest observed stabilized trajectory weight.
    pub maximum_weight: f64,
    /// Number of observed terminal outcomes used in the fit.
    pub observed_subjects: usize,
}

/// Pointwise 95% intervals for the intercept and each period coefficient.
#[derive(Clone, Debug, PartialEq)]
pub struct MsmIntervals95 {
    /// Weighted intercept lower and upper endpoints.
    pub intercept: [f64; 2],
    /// Period coefficient lower and upper endpoints.
    pub period_effects: Vec<[f64; 2]>,
}

/// Form coefficient intervals only with enough independent subjects and
/// effective weighted support for the subject-clustered CR1 approximation.
pub fn pointwise_intervals_95(fit: &MsmSummary, subjects: usize) -> Option<MsmIntervals95> {
    if subjects < 300 || fit.observed_subjects < 200 || fit.effective_sample_size < 150.0
        || fit.intercept_standard_error <= 0.0
        || fit.standard_errors.iter().any(|&se| se <= 0.0 || !se.is_finite()) { return None; }
    let z = antecedent_stats::normal_ppf(0.975);
    let interval = |center: f64, se: f64| [center - z * se, center + z * se];
    let result = MsmIntervals95 {
        intercept: interval(fit.intercept, fit.intercept_standard_error),
        period_effects: fit.period_effects.iter().zip(&fit.standard_errors)
            .map(|(&center, &se)| interval(center, se)).collect(),
    };
    result.intercept.iter().chain(result.period_effects.iter().flatten()).all(|x| x.is_finite()).then_some(result)
}

/// Fit one terminal-outcome row per subject; all histories are subject-major.
pub fn fit_binary_msm(
    y: &[f64], a: &[bool], p: &[f64], numerator: &[f64], observed: &[bool],
    censor: &[f64], periods: usize, floor: f64,
) -> Result<MsmSummary, &'static str> {
    let n = y.len();
    let cells = n.checked_mul(periods).ok_or("MSM dimensions overflow")?;
    if n == 0 || periods == 0 || a.len() != cells || p.len() != cells
        || censor.len() != cells || observed.len() != n || numerator.len() != periods {
        return Err("MSM arrays must have matching subject and period dimensions");
    }
    if !floor.is_finite() || !(0.0 < floor && floor <= 0.5) {
        return Err("minimum probability must be finite and in (0, 0.5]");
    }
    if numerator.iter().any(|&v| !v.is_finite() || v < floor || v > 1.0 - floor) {
        return Err("stabilizing numerator probabilities violate the declared positivity floor");
    }
    let columns = periods + 1;
    let included = observed.iter().filter(|&&x| x).count();
    if included <= columns { return Err("MSM requires more observed subjects than coefficients"); }
    let mut design = vec![vec![0.0; columns]; n];
    let mut weights = vec![0.0; n];
    let mut max_weight: f64 = 0.0;
    let mut sum = 0.0;
    let mut sumsq = 0.0;
    for i in 0..n {
        design[i][0] = 1.0;
        if observed[i] && !y[i].is_finite() { return Err("observed terminal outcomes must be finite"); }
        let mut weight = 1.0;
        for t in 0..periods {
            let j = i * periods + t;
            if !p[j].is_finite() || p[j] < floor || p[j] > 1.0 - floor {
                return Err("sequential treatment positivity is violated");
            }
            if !censor[j].is_finite() || censor[j] < floor || censor[j] > 1.0 {
                return Err("sequential censoring positivity is violated");
            }
            design[i][t + 1] = f64::from(a[j]);
            if observed[i] {
                let top = if a[j] { numerator[t] } else { 1.0 - numerator[t] };
                let bottom = if a[j] { p[j] } else { 1.0 - p[j] };
                weight *= top / (bottom * censor[j]);
                if !weight.is_finite() { return Err("stabilized sequential weight overflowed"); }
            }
        }
        if observed[i] {
            weights[i] = weight;
            max_weight = max_weight.max(weight);
            sum += weight;
            sumsq += weight * weight;
            if !sum.is_finite() || !sumsq.is_finite() { return Err("stabilized weight diagnostics overflowed"); }
        }
    }
    let mut bread_input = vec![vec![0.0; columns]; columns];
    let mut rhs = vec![0.0; columns];
    for i in 0..n {
        if !observed[i] { continue; }
        for j in 0..columns {
            rhs[j] += weights[i] * design[i][j] * y[i];
            for k in 0..columns { bread_input[j][k] += weights[i] * design[i][j] * design[i][k]; }
        }
    }
    if rhs.iter().chain(bread_input.iter().flatten()).any(|v| !v.is_finite()) {
        return Err("weighted MSM normal equations overflowed");
    }
    let bread = invert(bread_input)?;
    let beta = matvec(&bread, &rhs);
    if beta.iter().any(|v| !v.is_finite()) { return Err("MSM coefficients are non-finite"); }
    let mut meat = vec![vec![0.0; columns]; columns];
    for i in 0..n {
        if !observed[i] { continue; }
        let residual = y[i] - dot(&design[i], &beta);
        for j in 0..columns {
            for k in 0..columns {
                meat[j][k] += weights[i] * weights[i] * design[i][j] * design[i][k] * residual * residual;
            }
        }
    }
    let mut covariance = matmul(&matmul(&bread, &meat), &bread);
    let correction = included as f64 / (included - columns) as f64;
    for row in &mut covariance { for v in row { *v *= correction; } }
    let intercept_standard_error = covariance[0][0].max(0.0).sqrt();
    let standard_errors = (1..columns).map(|j| covariance[j][j].max(0.0).sqrt()).collect::<Vec<_>>();
    let ess = sum * sum / sumsq;
    if covariance.iter().flatten().any(|v| !v.is_finite())
        || !intercept_standard_error.is_finite()
        || standard_errors.iter().any(|v| !v.is_finite()) || !ess.is_finite() {
        return Err("MSM covariance is non-finite");
    }
    Ok(MsmSummary { intercept: beta[0], intercept_standard_error, period_effects: beta[1..].to_vec(),
        standard_errors, effective_sample_size: ess, maximum_weight: max_weight,
        observed_subjects: included })
}

fn invert(mut a: Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>, &'static str> {
    let n = a.len();
    let mut inverse = vec![vec![0.0; n]; n];
    for i in 0..n { inverse[i][i] = 1.0; }
    for j in 0..n {
        let pivot = (j..n).max_by(|&x, &y| a[x][j].abs().total_cmp(&a[y][j].abs())).unwrap();
        let scale = a[pivot].iter().map(|v| v.abs()).fold(0.0, f64::max);
        if a[pivot][j].abs() <= 1e-12 * scale.max(1.0) { return Err("weighted MSM design is rank-deficient"); }
        a.swap(j, pivot);
        inverse.swap(j, pivot);
        let divisor = a[j][j];
        for k in 0..n { a[j][k] /= divisor; inverse[j][k] /= divisor; }
        for i in 0..n {
            if i == j { continue; }
            let factor = a[i][j];
            for k in 0..n { a[i][k] -= factor * a[j][k]; inverse[i][k] -= factor * inverse[j][k]; }
        }
    }
    Ok(inverse)
}

fn dot(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b).map(|(x, y)| x * y).sum() }
fn matvec(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> { a.iter().map(|row| dot(row, b)).collect() }
fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut out = vec![vec![0.0; n]; n];
    for i in 0..n { for j in 0..n { for k in 0..n { out[i][k] += a[i][j] * b[j][k]; } } }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_additive_truth_and_refusals() {
        let a = [false, false, false, true, true, false, true, true,
            false, false, false, true, true, false, true, true];
        let y = [1.0, 4.0, 3.0, 6.0, 1.0, 4.0, 3.0, 6.0];
        let result = fit_binary_msm(&y, &a, &[0.5; 16], &[0.5; 2], &[true; 8], &[1.0; 16], 2, 0.01).unwrap();
        assert!((result.intercept - 1.0).abs() < 1e-12);
        assert!((result.period_effects[0] - 2.0).abs() < 1e-12);
        assert!((result.period_effects[1] - 3.0).abs() < 1e-12);
        assert_eq!(result.observed_subjects, 8);
        assert!(result.standard_errors.iter().all(|v| v.is_finite()));
        assert!(pointwise_intervals_95(&result, 8).is_none());
        assert!(fit_binary_msm(&y, &a, &[0.0; 16], &[0.5; 2], &[true; 8], &[1.0; 16], 2, 0.01).unwrap_err().contains("positivity"));
        assert!(fit_binary_msm(&y, &a, &[0.5; 16], &[0.5; 2], &[true; 8], &[0.0; 16], 2, 0.01).unwrap_err().contains("censoring"));
    }

    #[test]
    fn subject_clustered_msm_intervals_cover_known_coefficients() {
        let n = 300;
        let simulations = 2_000;
        let mut state = 0x23A9_E381_7D45_BC06_u64;
        let mut uniform = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 11) as f64 / ((1_u64 << 53) as f64)
        };
        let probabilities = vec![0.5; n * 2];
        let censoring = vec![0.9; n * 2];
        let mut hits = [0_usize; 3];
        let mut skipped = 0;
        for _ in 0..simulations {
            let mut treatment = Vec::with_capacity(n * 2);
            let mut observed = Vec::with_capacity(n);
            let mut outcomes = Vec::with_capacity(n);
            for _ in 0..n {
                let a0 = uniform() < 0.5;
                let a1 = uniform() < 0.5;
                treatment.extend([a0, a1]);
                observed.push(uniform() < 0.9 && uniform() < 0.9);
                let noise = (uniform() - 0.5) * 2.0 * (1.0 + f64::from(a0));
                outcomes.push(1.0 + 2.0 * f64::from(a0) + 3.0 * f64::from(a1) + noise);
            }
            let fit = fit_binary_msm(&outcomes, &treatment, &probabilities, &[0.5; 2],
                &observed, &censoring, 2, 0.01).unwrap();
            if let Some(intervals) = pointwise_intervals_95(&fit, n) {
                for (j, (interval, truth)) in [intervals.intercept, intervals.period_effects[0], intervals.period_effects[1]]
                    .into_iter().zip([1.0, 2.0, 3.0]).enumerate() {
                    hits[j] += usize::from(interval[0] <= truth && truth <= interval[1]);
                }
            } else { skipped += 1; }
        }
        let mcse = (0.95_f64 * 0.05 / simulations as f64).sqrt();
        assert!(skipped <= simulations / 100, "weak-support skips {skipped}");
        for (j, count) in hits.into_iter().enumerate() {
            let coverage = count as f64 / simulations as f64;
            assert!((coverage - 0.95).abs() <= 3.0 * mcse, "MSM coefficient {j} coverage {coverage}");
        }
    }
}
