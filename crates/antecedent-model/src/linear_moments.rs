// SPDX-License-Identifier: MIT OR Apache-2.0
//! Exact sufficient-statistic likelihoods of known-regime affine Gaussian SCMs.
//! Aggregation retains each joint intervention regime and all cross moments;
//! it never substitutes local equilibrium regressions for the joint likelihood.
use crate::nonlinear_equilibrium::{
    Activation, Clamp, EquilibriumError, EquilibriumModel, MAX_ORDER, MechanismLikelihoodScore,
    RegimeRow, cholesky, local_positions,
};
use crate::sparse_equilibrium::{FactorError, SolverBudgetCode, SparseLu};
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
/// Largest order admitted for exact affine sufficient-statistic scores. The per-regime
/// scatter and the score are dense `n x n` outputs and the disturbance-precision inverse is
/// `O(k^3)` for `k` free variables, so 1024 (about 8 MiB per regime, 1e9 flops per regime)
/// is the admitted bound; the Jacobian log-determinant, its inverse and the `A S` products
/// are sparse. Beyond it the refusal is `solver_model_order_budget`.
pub const MAX_AFFINE_MOMENT_ORDER: usize = 1024;

/// Most solve work one [`LinearRegimeMoments::likelihood_score`] may do, summed over every
/// retained regime, in the units of [`regime_work`] (about one floating-point operation each).
/// A single regime at [`MAX_AFFINE_MOMENT_ORDER`] costs about `1.3e10`, so this admits roughly
/// fifteen of them; the per-regime order bound alone would let 4096 regimes cost `5e13`. Beyond
/// it the refusal is `solver_iteration_budget`, raised before any regime is solved.
pub const MAX_AFFINE_MOMENT_WORK: u64 = 200_000_000_000;

/// Conservative operation count of one regime with `k` free variables in an order-`n` model:
/// the disturbance Cholesky and precision inverse, the sparse LU with its `k` unit solves, the
/// bijection certificate (another `k` solves and the interval residual) and the `A S` products.
fn regime_work(k: usize, n: usize) -> u128 {
    let (k, n) = (k as u128, n as u128);
    8 * k * k * k + 4 * k * n * n
}

/// Checked aggregate of independent equilibrium rows with fixed known regimes.
#[derive(Clone, Debug)]
pub struct LinearRegimeMoments {
    order: usize,
    groups: Vec<Group>,
}
use crate::nonlinear_equilibrium::invalid_equilibrium_input as invalid;
fn inverse(mut matrix: Vec<f64>, n: usize) -> Result<Vec<f64>, EquilibriumError> {
    if n.checked_mul(n) != Some(matrix.len()) || matrix.iter().any(|v| !v.is_finite()) {
        return Err(EquilibriumError::Singular);
    }
    let mut scales = Vec::with_capacity(n);
    for i in 0..n {
        let diagonal = matrix[i * n + i];
        if diagonal <= 0.0 {
            return Err(EquilibriumError::Singular);
        }
        scales.push(diagonal.sqrt());
    }
    for i in 0..n {
        for j in 0..n {
            let large = scales[i].max(scales[j]);
            let small = scales[i].min(scales[j]);
            matrix[i * n + j] = if i == j { 1.0 } else { (matrix[i * n + j] / large) / small };
        }
    }
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        out[i * n + i] = 1.0;
    }
    if !gauss_jordan_solve(&mut matrix, &mut out, n, n) {
        return Err(EquilibriumError::Singular);
    }
    let mut precision = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let large = scales[i].max(scales[j]);
            let small = scales[i].min(scales[j]);
            let value = (out[j * n + i] / large) / small;
            if !value.is_finite() {
                return Err(EquilibriumError::Singular);
            }
            precision[i * n + j] = value;
        }
    }
    Ok(precision)
}
#[allow(
    clippy::too_many_lines,
    reason = "each regime's factorization and score share its admitted scratch"
)]
fn score_regime(
    group: &Group,
    model: &EquilibriumModel,
    n: usize,
    ctx: &ExecutionContext,
    likelihood: &mut f64,
    scores: &mut [MechanismLikelihoodScore],
) -> Result<(), EquilibriumError> {
    if ctx.cancellation.is_cancelled() {
        return Err(EquilibriumError::Resource("cancelled".into()));
    }
    let free = model.free_indices(&group.clamps)?;
    let k = free.len();
    if k == 0 {
        return Ok(());
    }
    for block in model.blocks(&group.clamps)? {
        if !model.affine_certificate_checked(&block, ctx, usize::MAX, u64::MAX, &mut || {
            crate::nonlinear_equilibrium::check_cancel(ctx)
        })? {
            return Err(EquilibriumError::Uncertified { bound: f64::INFINITY });
        }
    }
    let count = group.count as f64;
    let covariance: Vec<_> = free
        .iter()
        .flat_map(|&i| free.iter().map(move |&j| model.noise_covariance[i * n + j]))
        .collect();
    crate::nonlinear_equilibrium::check_cancel(ctx)?;
    let lower = cholesky(&covariance, k)?;
    let noise_logdet = 2.0 * (0..k).map(|i| lower[i * k + i].ln()).sum::<f64>();
    let precision = inverse(covariance, k)?;
    crate::nonlinear_equilibrium::check_cancel(ctx)?;
    let reads = model.read_sets()?;
    let local = local_positions(&free);
    let jacobian = model.sparse_jacobian(&group.mean, &free, &reads, &local)?;
    // Fill is capped at the dense `k x k` worst case the lease above covers.
    let lu =
        SparseLu::factor(&jacobian, k * k, &mut || crate::nonlinear_equilibrium::check_cancel(ctx))
            .map_err(|error| match error {
                FactorError::FillBudget(required) => {
                    SolverBudgetCode::FactorFill.refusal(required as u64, (k * k) as u64)
                }
                FactorError::Interrupted(error) => error,
                FactorError::Singular | FactorError::NonFinite => EquilibriumError::Singular,
            })?;
    let logdet = lu.log_abs_determinant();
    let mut jacobian_inverse = vec![0.0; k * k];
    for c in 0..k {
        if c % 32 == 0 {
            crate::nonlinear_equilibrium::check_cancel(ctx)?;
        }
        let mut unit = vec![0.0; k];
        unit[c] = 1.0;
        for (r, v) in lu.solve(&unit).into_iter().enumerate() {
            jacobian_inverse[r * k + c] = v;
        }
    }
    let residual_mean = model.residual(&group.mean, &vec![0.0; n], &free);
    // Sparse rows of A = I - W over the free equations.
    let a_rows: Vec<Vec<(usize, f64)>> = free
        .iter()
        .map(|&i| {
            (0..n)
                .filter_map(|j| {
                    let v = (if i == j { 1.0 } else { 0.0 }) - model.equations[i].weights[j];
                    (v != 0.0).then_some((j, v))
                })
                .collect()
        })
        .collect();
    let mut a_scatter = vec![0.0; k * n];
    for (i, row) in a_rows.iter().enumerate() {
        for &(r, v) in row {
            for j in 0..n {
                a_scatter[i * n + j] += v * group.scatter[r * n + j];
            }
        }
    }
    let mut quadratic = 0.0;
    for i in 0..k {
        for r in 0..k {
            let centered: f64 = a_rows[r].iter().map(|&(j, v)| a_scatter[i * n + j] * v).sum();
            quadratic +=
                precision[i * k + r] * (centered + count * residual_mean[i] * residual_mean[r]);
        }
    }
    *likelihood += count * logdet
        - 0.5 * (quadratic + count * (noise_logdet + k as f64 * (2.0 * std::f64::consts::PI).ln()));
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

    Ok(())
}

impl LinearRegimeMoments {
    /// Aggregate the actual observations by joint clamp target set. Actual assigned values retain their joint cross moments.
    /// Multivariate Welford accumulation preserves the centered cross moments.
    pub fn from_rows(
        rows: &[RegimeRow],
        order: usize,
        ctx: &ExecutionContext,
    ) -> Result<Self, EquilibriumError> {
        if order > MAX_AFFINE_MOMENT_ORDER.min(MAX_ORDER) {
            return Err(SolverBudgetCode::ModelOrder
                .refusal(order as u64, MAX_AFFINE_MOMENT_ORDER.min(MAX_ORDER) as u64));
        }
        if rows.is_empty() || rows.len() > 10_000_000 || order == 0 {
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
    #[allow(
        clippy::too_many_lines,
        reason = "the scientific calculation and its checks form one bounded procedure"
    )]
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
        // Total solve work over every regime, charged before any is solved. Clamps are
        // distinct and in range (checked by `from_rows`), so a regime has `n - clamps` free
        // variables.
        let work: u128 =
            self.groups.iter().map(|g| regime_work(n.saturating_sub(g.clamps.len()), n)).sum();
        if work > u128::from(MAX_AFFINE_MOMENT_WORK) {
            return Err(SolverBudgetCode::IterationBudget
                .refusal(u64::try_from(work).unwrap_or(u64::MAX), MAX_AFFINE_MOMENT_WORK));
        }
        // Worst-case scratch of one regime, held for the whole score: the covariance, its
        // Cholesky factor, the scaled copy, the identity/inverse pair and the precision
        // (`inverse`), the dense Jacobian inverse, the sparse LU at its dense fill bound (capped
        // below at `k * k`), the sparse rows of `A`, `A S` and the certificate's own quote.
        let _lease = ctx
            .acquire_serial_numerical_admission(((64 * n * n + 256 * n) * 8) as u64)
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
            score_regime(group, model, n, ctx, &mut likelihood, &mut scores)?;
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
    #[allow(clippy::float_cmp, reason = "the test checks exact values and boundary behavior")]
    fn covariance_precision_and_affine_scores_are_unit_equilibrated() {
        let diagonal = inverse(vec![1e12, 0.0, 0.0, 1.0], 2).unwrap();
        assert!((diagonal[0] / 1e-12 - 1.0).abs() < 1e-14);
        assert_eq!(diagonal[3], 1.0);
        let correlated = inverse(vec![1e12, 5e5, 5e5, 1.0], 2).unwrap();
        let expected = [4e-12 / 3.0, -2e-6 / 3.0, -2e-6 / 3.0, 4.0 / 3.0];
        for (actual, reference) in correlated.iter().zip(expected) {
            assert!((actual / reference - 1.0).abs() < 1e-13);
        }
        assert!(inverse(vec![1.0, 1.0, 1.0, 1.0], 2).is_err());
        let model = EquilibriumModel {
            equations: (0..2)
                .map(|i| Equation {
                    id: format!("x{i}"),
                    intercept: 0.0,
                    weights: vec![0.0; 2],
                    activation: Activation::Identity,
                    expression: None,
                    parameters: vec![],
                })
                .collect(),
            noise_covariance: vec![1e12, 0.0, 0.0, 1.0],
        };
        let rows = vec![
            RegimeRow { values: vec![1e6, 1.0], clamps: vec![] },
            RegimeRow { values: vec![-1e6, -1.0], clamps: vec![] },
        ];
        let ctx = ExecutionContext::for_tests(1);
        let moments = LinearRegimeMoments::from_rows(&rows, 2, &ctx).unwrap();
        let (raw, _) = model.likelihood_score(&rows, &ctx).unwrap();
        let (aggregated, _) = moments.likelihood_score(&model, &ctx).unwrap();
        assert!((raw - aggregated).abs() < 1e-10);
    }

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

    fn identity_model(n: usize) -> EquilibriumModel {
        EquilibriumModel {
            equations: (0..n)
                .map(|i| Equation {
                    id: format!("x{i}"),
                    intercept: 0.0,
                    weights: vec![0.0; n],
                    activation: Activation::Identity,
                    expression: None,
                    parameters: vec![],
                })
                .collect(),
            noise_covariance: (0..n * n).map(|c| f64::from(u8::from(c / n == c % n))).collect(),
        }
    }

    #[test]
    fn the_score_is_refused_by_a_short_pool_or_a_cancelled_context() {
        let model = identity_model(2);
        let rows = vec![
            RegimeRow { values: vec![0.2, -0.4], clamps: vec![] },
            RegimeRow { values: vec![0.4, 0.7], clamps: vec![] },
        ];
        let ctx = ExecutionContext::for_tests(1);
        let stats = LinearRegimeMoments::from_rows(&rows, 2, &ctx).unwrap();
        assert!(stats.likelihood_score(&model, &ctx).is_ok());
        let tiny = ExecutionContext::for_tests(1).with_memory_limit(Some(64));
        assert!(matches!(
            stats.likelihood_score(&model, &tiny),
            Err(EquilibriumError::Resource(_))
        ));
        let cancelled = ExecutionContext::for_tests(1);
        cancelled.cancellation.cancel();
        assert!(matches!(
            stats.likelihood_score(&model, &cancelled),
            Err(EquilibriumError::Resource(_))
        ));
    }

    #[test]
    fn total_solve_work_across_regimes_is_bounded_before_any_regime_is_solved() {
        let n = 300;
        let per_regime = regime_work(n, n);
        let regimes = usize::try_from(u128::from(MAX_AFFINE_MOMENT_WORK) / per_regime).unwrap() + 1;
        // Aggregates are empty: the refusal precedes every use of them.
        let group = Group { count: 1, mean: Vec::new(), scatter: Vec::new(), clamps: Vec::new() };
        let over = LinearRegimeMoments { order: n, groups: vec![group.clone(); regimes] };
        let error = over.likelihood_score(&identity_model(n), &ExecutionContext::for_tests(1));
        assert_eq!(error.unwrap_err().budget_code(), Some(SolverBudgetCode::IterationBudget));
    }
}
