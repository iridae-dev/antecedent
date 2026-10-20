// SPDX-License-Identifier: MIT OR Apache-2.0
//! Exact sufficient-statistic likelihoods of known-regime affine Gaussian SCMs.
//! Aggregation retains each joint intervention regime and all cross moments;
//! it never substitutes local equilibrium regressions for the joint likelihood.
use crate::nonlinear_equilibrium::{
    Activation, Clamp, EquilibriumError, EquilibriumModel, MechanismLikelihoodScore, RegimeRow,
    cholesky, log_abs_determinant,
};
use antecedent_core::ExecutionContext;
use antecedent_kernels::gauss_jordan_solve;
use std::collections::BTreeMap;
#[derive(Clone, Debug)]
struct Group {
    count: u64,
    mean: Vec<f64>,
    scatter: Vec<f64>,
    clamps: Vec<Clamp>,
}
/// Checked aggregate of independent equilibrium rows with fixed known regimes.
#[derive(Clone, Debug)]
pub struct LinearRegimeMoments {
    order: usize,
    groups: Vec<Group>,
}
fn invalid(s: &str) -> EquilibriumError {
    EquilibriumError::Invalid(s.into())
}
fn inverse(mut matrix: Vec<f64>, n: usize) -> Result<Vec<f64>, EquilibriumError> {
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        out[i * n + i] = 1.0;
    }
    if !gauss_jordan_solve(&mut matrix, &mut out, n, n) {
        return Err(EquilibriumError::Singular);
    }
    Ok((0..n).flat_map(|i| (0..n).map(move |j| (i, j))).map(|(i, j)| out[j * n + i]).collect())
}
impl LinearRegimeMoments {
    /// Aggregate the actual observations by joint clamp target set. Actual assigned values retain their joint cross moments.
    /// Multivariate Welford accumulation preserves the centered cross moments.
    pub fn from_rows(
        rows: &[RegimeRow],
        order: usize,
        ctx: &ExecutionContext,
    ) -> Result<Self, EquilibriumError> {
        if rows.is_empty() || rows.len() > 10_000_000 || order == 0 || order > 128 {
            return Err(invalid("affine moment input limits"));
        }
        let mut groups: BTreeMap<Vec<usize>, Group> = BTreeMap::new();
        let mut leases = Vec::new();
        for row in rows {
            if ctx.cancellation.is_cancelled() {
                return Err(EquilibriumError::Resource("cancelled".into()));
            }
            if row.values.len() != order || row.values.iter().any(|v| !v.is_finite()) {
                return Err(invalid("affine observation shape"));
            }
            let mut clamps = row.clamps.clone();
            clamps.sort_by_key(|c| c.variable);
            for (i, clamp) in clamps.iter().enumerate() {
                #[allow(
                    clippy::float_cmp,
                    reason = "a hard clamp is an exact known structural value"
                )]
                let mismatch = clamp.variable >= order
                    || !clamp.value.is_finite()
                    || row.values[clamp.variable] != clamp.value
                    || clamps[..i].iter().any(|c| c.variable == clamp.variable);
                if mismatch {
                    return Err(invalid("affine observation clamp"));
                }
            }
            let key = clamps.iter().map(|c| c.variable).collect::<Vec<_>>();
            if !groups.contains_key(&key) {
                if groups.len() >= 4096 {
                    return Err(invalid("affine distinct regime limit"));
                }
                leases.push(
                    ctx.acquire_admission(0, ((order * order + 8 * order + 64) * 8) as u64)
                        .map_err(|e| EquilibriumError::Resource(e.to_string()))?,
                );
                groups.insert(
                    key.clone(),
                    Group {
                        count: 0,
                        mean: vec![0.0; order],
                        scatter: vec![0.0; order * order],
                        clamps,
                    },
                );
            }
            let group = groups.get_mut(&key).ok_or_else(|| invalid("missing affine regime"))?;
            group.count += 1;
            let count = group.count as f64;
            let delta: Vec<_> = row.values.iter().zip(&group.mean).map(|(x, m)| x - m).collect();
            for (mean, &d) in group.mean.iter_mut().zip(&delta) {
                *mean += d / count;
            }
            for (i, &di) in delta.iter().enumerate() {
                for j in 0..order {
                    group.scatter[i * order + j] += di * (row.values[j] - group.mean[j]);
                }
            }
        }
        if groups.values().any(|g| g.mean.iter().chain(&g.scatter).any(|v| !v.is_finite())) {
            return Err(invalid("nonfinite affine aggregates"));
        }
        Ok(Self { order, groups: groups.into_values().collect() })
    }
    /// Actual number of separately retained joint intervention regimes.
    #[must_use]
    pub fn regimes(&self) -> usize {
        self.groups.len()
    }
    /// Complete Gaussian likelihood and analytic score from joint sufficient statistics.
    /// Free affine systems need a verified bijection; contraction/dynamic stability
    /// are distinct. Full covariance, intercepts and clamped means are retained.
    pub fn likelihood_score(
        &self,
        model: &EquilibriumModel,
        ctx: &ExecutionContext,
    ) -> Result<(f64, Vec<MechanismLikelihoodScore>), EquilibriumError> {
        model.validate()?;
        let n = self.order;
        if model.order() != n
            || model
                .equations
                .iter()
                .any(|e| e.activation != Activation::Identity || e.expression.is_some())
        {
            return Err(invalid("affine moment model"));
        }
        let _lease = ctx
            .acquire_admission(1, ((64 * n * n + 256 * n) * 8) as u64)
            .map_err(|e| EquilibriumError::Resource(e.to_string()))?;
        let mut likelihood = 0.0;
        let mut scores = vec![
            MechanismLikelihoodScore {
                intercept: 0.0,
                weights: vec![0.0; n],
                parameters: vec![]
            };
            n
        ];
        for group in &self.groups {
            if ctx.cancellation.is_cancelled() {
                return Err(EquilibriumError::Resource("cancelled".into()));
            }
            let free = model.free_indices(&group.clamps)?;
            let k = free.len();
            if k == 0 {
                continue;
            }
            for block in model.blocks(&group.clamps)? {
                if !model.affine_certificate(&block) {
                    return Err(EquilibriumError::Uncertified { bound: f64::INFINITY });
                }
            }
            let count = group.count as f64;
            let covariance: Vec<_> = free
                .iter()
                .flat_map(|&i| free.iter().map(move |&j| model.noise_covariance[i * n + j]))
                .collect();
            let lower = cholesky(&covariance, k)?;
            let noise_logdet = 2.0 * (0..k).map(|i| lower[i * k + i].ln()).sum::<f64>();
            let precision = inverse(covariance, k)?;
            let jacobian = model.residual_jacobian(&group.mean, &free);
            let logdet = log_abs_determinant(jacobian.clone(), k)?;
            let jacobian_inverse = inverse(jacobian, k)?;
            let residual_mean = model.residual(&group.mean, &vec![0.0; n], &free);
            let a: Vec<_> = free
                .iter()
                .flat_map(|&i| {
                    (0..n).map(move |j| {
                        (if i == j { 1.0 } else { 0.0 }) - model.equations[i].weights[j]
                    })
                })
                .collect();
            let mut a_scatter = vec![0.0; k * n];
            for i in 0..k {
                for j in 0..n {
                    for r in 0..n {
                        a_scatter[i * n + j] += a[i * n + r] * group.scatter[r * n + j];
                    }
                }
            }
            let mut quadratic = 0.0;
            for i in 0..k {
                for r in 0..k {
                    let centered: f64 = (0..n).map(|j| a_scatter[i * n + j] * a[r * n + j]).sum();
                    quadratic += precision[i * k + r]
                        * (centered + count * residual_mean[i] * residual_mean[r]);
                }
            }
            likelihood += count * logdet
                - 0.5
                    * (quadratic
                        + count * (noise_logdet + k as f64 * (2.0 * std::f64::consts::PI).ln()));
            for (i, &equation) in free.iter().enumerate() {
                scores[equation].intercept +=
                    count * (0..k).map(|r| precision[i * k + r] * residual_mean[r]).sum::<f64>();
                for j in 0..n {
                    scores[equation].weights[j] += (0..k)
                        .map(|r| {
                            precision[i * k + r]
                                * (a_scatter[r * n + j] + count * residual_mean[r] * group.mean[j])
                        })
                        .sum::<f64>();
                    if let Some(local_j) = free.iter().position(|&v| v == j) {
                        scores[equation].weights[j] -= count * jacobian_inverse[local_j * k + i];
                    }
                }
            }
        }
        if !likelihood.is_finite()
            || scores
                .iter()
                .any(|s| !s.intercept.is_finite() || s.weights.iter().any(|v| !v.is_finite()))
        {
            return Err(invalid("nonfinite affine likelihood/score"));
        }
        Ok((likelihood, scores))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::nonlinear_equilibrium::Equation;
    #[test]
    fn joint_moments_match_row_likelihood_and_jacobian_scores_across_joint_interventions() {
        let ctx = ExecutionContext::for_tests(1);
        let model = EquilibriumModel {
            equations: vec![
                Equation {
                    id: "a".into(),
                    intercept: 0.1,
                    weights: vec![0.1, 0.3],
                    activation: Activation::Identity,
                    expression: None,
                    parameters: vec![],
                },
                Equation {
                    id: "b".into(),
                    intercept: -0.1,
                    weights: vec![-0.4, 0.1],
                    activation: Activation::Identity,
                    expression: None,
                    parameters: vec![],
                },
            ],
            noise_covariance: vec![1.0, 0.2, 0.2, 0.8],
        };
        let rows = vec![
            RegimeRow { values: vec![0.2, -0.4], clamps: vec![] },
            RegimeRow { values: vec![0.4, 0.7], clamps: vec![] },
            RegimeRow { values: vec![1.0, 0.3], clamps: vec![Clamp { variable: 0, value: 1.0 }] },
            RegimeRow { values: vec![1.0, 0.5], clamps: vec![Clamp { variable: 0, value: 1.0 }] },
            RegimeRow {
                values: vec![1.0, 2.0],
                clamps: vec![Clamp { variable: 0, value: 1.0 }, Clamp { variable: 1, value: 2.0 }],
            },
        ];
        let stats = LinearRegimeMoments::from_rows(&rows, 2, &ctx).unwrap();
        assert_eq!(stats.regimes(), 3);
        let (raw, raw_score) = model.likelihood_score(&rows, &ctx).unwrap();
        let (aggregated, score) = stats.likelihood_score(&model, &ctx).unwrap();
        assert!((raw - aggregated).abs() < 1e-12);
        for (a, b) in raw_score.iter().zip(&score) {
            assert!((a.intercept - b.intercept).abs() < 1e-12);
            for (x, y) in a.weights.iter().zip(&b.weights) {
                assert!((x - y).abs() < 1e-12);
            }
        }
    }
}
