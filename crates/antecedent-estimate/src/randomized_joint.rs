//! Joint design-based contrasts of several outcomes under complete randomization of clusters.
//!
//! Several effects of one experiment are not several experiments. They come from the same
//! assigned units, so their sampling errors are correlated, and a quantity that combines them
//! (a product, a margin) inherits that correlation. This module estimates the per-row arm means
//! of `K` outcomes together and keeps the full covariance of those estimators, so a downstream
//! function of the outcomes can propagate the dependence instead of assuming it away.
//!
//! The estimator is the vector form of [`crate::randomized_neyman::complete_cluster_itt`]: the
//! arm mean of an outcome is the cluster total, averaged over the assigned clusters of the arm
//! and scaled to the average per outcome row, and the covariance is the conservative Neyman
//! bound `Cov(totals)/n` of each arm. Unit-level complete randomization is the case of one row
//! per cluster. The bound drops the unobservable effect covariance, a positive-semidefinite
//! term, so the reported covariance dominates the exact one for every linear combination.
//! Coverage of a function of the outcomes is a separate claim that this module does not make.
// SPDX-License-Identifier: MIT OR Apache-2.0

/// Minimum independently randomized units in each arm below which the covariance is still
/// reported but no interval over it is supported.
pub const MIN_UNITS_PER_ARM_FOR_INTERVAL: usize =
    crate::randomized_neyman::MIN_CLUSTERS_PER_ARM_FOR_INTERVAL;

/// Per-row arm means of `K` outcomes with the covariance of their estimators.
#[derive(Clone, Debug, PartialEq)]
pub struct JointClusterContrast {
    /// Number of outcomes `K`.
    pub outcomes: usize,
    /// Assigned clusters in the treated arm.
    pub treated_clusters: usize,
    /// Assigned clusters in the control arm.
    pub control_clusters: usize,
    /// Per-row mean of each outcome in the treated arm.
    pub treated_means: Vec<f64>,
    /// Per-row mean of each outcome in the control arm.
    pub control_means: Vec<f64>,
    /// `K x K` row-major covariance of the treated-arm mean estimators.
    pub treated_covariance: Vec<f64>,
    /// `K x K` row-major covariance of the control-arm mean estimators.
    pub control_covariance: Vec<f64>,
    /// Treated minus control mean of each outcome.
    pub effects: Vec<f64>,
    /// `K x K` row-major covariance of the effects: the sum of the two arm covariances.
    pub effect_covariance: Vec<f64>,
    /// Whether both arms hold enough independent clusters for a normal interval.
    pub interval_supported: bool,
}

/// Estimate the joint per-row arm means and their covariance.
///
/// `totals[k][c]` is outcome `k` summed over the rows of assigned cluster `c`, already
/// adjusted by any fixed pre-assignment coefficient; `assigned[c]` is the realized arm of
/// cluster `c`; `outcome_rows` is the total number of outcome rows. Returns `None` for no
/// outcomes, misaligned or non-finite totals, an arm with fewer than two clusters, or no rows.
#[must_use]
pub fn complete_cluster_joint_contrast(
    totals: &[&[f64]],
    assigned: &[bool],
    outcome_rows: usize,
) -> Option<JointClusterContrast> {
    let k = totals.len();
    let clusters = assigned.len();
    let treated_clusters = assigned.iter().filter(|a| **a).count();
    let control_clusters = clusters - treated_clusters;
    if k == 0
        || outcome_rows == 0
        || treated_clusters < 2
        || control_clusters < 2
        || totals.iter().any(|t| t.len() != clusters || t.iter().any(|v| !v.is_finite()))
    {
        return None;
    }
    let scale = clusters as f64 / outcome_rows as f64;
    let arm = |want: bool| -> (Vec<f64>, Vec<f64>) {
        let members: Vec<usize> = (0..clusters).filter(|c| assigned[*c] == want).collect();
        let n = members.len() as f64;
        let mean: Vec<f64> =
            totals.iter().map(|t| members.iter().map(|c| t[*c]).sum::<f64>() / n).collect();
        let mut covariance = vec![0.0; k * k];
        for a in 0..k {
            for b in a..k {
                let value = members
                    .iter()
                    .map(|c| (totals[a][*c] - mean[a]) * (totals[b][*c] - mean[b]))
                    .sum::<f64>()
                    / (n - 1.0)
                    / n
                    * scale
                    * scale;
                covariance[a * k + b] = value;
                covariance[b * k + a] = value;
            }
        }
        (mean.iter().map(|m| m * scale).collect(), covariance)
    };
    let (treated_means, treated_covariance) = arm(true);
    let (control_means, control_covariance) = arm(false);
    let effects: Vec<f64> = treated_means.iter().zip(&control_means).map(|(t, c)| t - c).collect();
    let effect_covariance: Vec<f64> =
        treated_covariance.iter().zip(&control_covariance).map(|(t, c)| t + c).collect();
    if effects.iter().chain(&effect_covariance).any(|v| !v.is_finite()) {
        return None;
    }
    let interval_supported = treated_clusters >= MIN_UNITS_PER_ARM_FOR_INTERVAL
        && control_clusters >= MIN_UNITS_PER_ARM_FOR_INTERVAL;
    Some(JointClusterContrast {
        outcomes: k,
        treated_clusters,
        control_clusters,
        treated_means,
        control_means,
        treated_covariance,
        control_covariance,
        effects,
        effect_covariance,
        interval_supported,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::randomized_neyman::complete_cluster_itt;

    fn uniform(mut state: u64) -> f64 {
        state ^= state >> 30;
        state = state.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        state ^= state >> 27;
        state = state.wrapping_mul(0x94D0_49BB_1331_11EB);
        state ^= state >> 31;
        (state >> 11) as f64 / (1_u64 << 53) as f64
    }

    /// Complete randomization of `treated` of `n` clusters, deterministic in `rep`.
    fn allocation(n: usize, treated: usize, rep: usize) -> Vec<bool> {
        let mut keyed = (0..n)
            .map(|i| {
                (
                    uniform(
                        (rep as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                            ^ (i as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03),
                    ),
                    i,
                )
            })
            .collect::<Vec<_>>();
        keyed.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut assigned = vec![false; n];
        for (_, i) in keyed.iter().take(treated) {
            assigned[*i] = true;
        }
        assigned
    }

    /// Two outcomes per cluster: a frequency-like total and a value-like total that moves with
    /// it, so their sampling errors are strongly correlated. Potential outcomes are fixed.
    fn population() -> (Vec<usize>, [Vec<f64>; 2], [Vec<f64>; 2]) {
        const CLUSTERS: usize = 60;
        let sizes = (0..CLUSTERS).map(|i| 2 + i % 3).collect::<Vec<_>>();
        let base = (0..CLUSTERS).map(|i| 1.0 + 0.7 * (i as f64 * 0.37).sin()).collect::<Vec<_>>();
        let control0 = (0..CLUSTERS).map(|i| sizes[i] as f64 * base[i]).collect::<Vec<_>>();
        let control1 = (0..CLUSTERS)
            .map(|i| sizes[i] as f64 * (5.0 + 3.0 * base[i] + 0.2 * (i as f64 * 0.91).cos()))
            .collect::<Vec<_>>();
        let treated0 = (0..CLUSTERS)
            .map(|i| control0[i] + sizes[i] as f64 * (0.4 + 0.1 * (i as f64 * 0.53).cos()))
            .collect::<Vec<_>>();
        let treated1 = (0..CLUSTERS)
            .map(|i| control1[i] - sizes[i] as f64 * (0.9 + 0.1 * (i as f64 * 0.29).sin()))
            .collect::<Vec<_>>();
        (sizes, [control0, control1], [treated0, treated1])
    }

    #[test]
    fn diagonal_reproduces_the_marginal_cluster_neyman_contrast() {
        let (sizes, control, treated) = population();
        let rows = sizes.iter().sum::<usize>();
        let assigned = allocation(sizes.len(), 30, 7);
        let observed = |k: usize| {
            (0..sizes.len())
                .map(|c| if assigned[c] { treated[k][c] } else { control[k][c] })
                .collect::<Vec<_>>()
        };
        let (o0, o1) = (observed(0), observed(1));
        let joint = complete_cluster_joint_contrast(&[&o0, &o1], &assigned, rows).unwrap();
        for (k, o) in [&o0, &o1].into_iter().enumerate() {
            let split = |want: bool| {
                (0..sizes.len()).filter(|c| assigned[*c] == want).map(|c| o[c]).collect::<Vec<_>>()
            };
            let marginal = complete_cluster_itt(&split(true), &split(false), rows).unwrap();
            assert!((joint.effects[k] - marginal.effect).abs() < 1e-12);
            assert!(
                (joint.effect_covariance[k * 2 + k] - marginal.variance_upper_bound).abs() < 1e-12
            );
        }
        assert_eq!(joint.effect_covariance[1], joint.effect_covariance[2]);
        assert!(joint.interval_supported);
    }

    #[test]
    fn cross_covariance_is_what_separate_marginals_would_lose() {
        // The two outcomes move together across clusters, so the sampling covariance is large and
        // positive; the variance of their difference is far below the sum of marginal variances.
        let (sizes, control, treated) = population();
        let rows = sizes.iter().sum::<usize>();
        let assigned = allocation(sizes.len(), 30, 11);
        let observed = |k: usize| {
            (0..sizes.len())
                .map(|c| if assigned[c] { treated[k][c] } else { control[k][c] })
                .collect::<Vec<_>>()
        };
        let (o0, o1) = (observed(0), observed(1));
        let joint = complete_cluster_joint_contrast(&[&o0, &o1], &assigned, rows).unwrap();
        let c = &joint.effect_covariance;
        let (v0, v1, cov) = (c[0], c[3], c[1]);
        assert!(cov.abs() > 0.0);
        let with_dependence = v0 + v1 - 2.0 * cov;
        let independence = v0 + v1;
        assert!(
            (with_dependence - independence).abs() > 0.1 * independence,
            "{with_dependence} vs {independence}: dependence must move the variance of a combination"
        );
    }

    #[test]
    fn joint_wald_region_covers_the_known_effect_vector_across_allocations() {
        // 2,000 complete allocations of 60 clusters, 30 per arm. The chi-square(2) Wald region
        // built from the joint covariance covers the known effect vector; the conservative
        // Neyman bound may over-cover.
        const REPLICATES: usize = 2_000;
        let (sizes, control, treated) = population();
        let rows = sizes.iter().sum::<usize>();
        let truth: Vec<f64> = (0..2)
            .map(|k| {
                (0..sizes.len()).map(|c| treated[k][c] - control[k][c]).sum::<f64>() / rows as f64
            })
            .collect();
        let chi2_2_95 = 5.991_464_547_107_98;
        let mut covered = 0;
        let mut marginal_covered = [0_usize; 2];
        for rep in 0..REPLICATES {
            let assigned = allocation(sizes.len(), 30, rep);
            let observed = |k: usize| {
                (0..sizes.len())
                    .map(|c| if assigned[c] { treated[k][c] } else { control[k][c] })
                    .collect::<Vec<_>>()
            };
            let (o0, o1) = (observed(0), observed(1));
            let joint = complete_cluster_joint_contrast(&[&o0, &o1], &assigned, rows).unwrap();
            let d = [joint.effects[0] - truth[0], joint.effects[1] - truth[1]];
            let c = &joint.effect_covariance;
            let det = c[0] * c[3] - c[1] * c[1];
            let quad = (c[3] * d[0] * d[0] - 2.0 * c[1] * d[0] * d[1] + c[0] * d[1] * d[1]) / det;
            covered += usize::from(quad <= chi2_2_95);
            for k in 0..2 {
                marginal_covered[k] +=
                    usize::from(d[k].abs() <= 1.959_963_984_540_054 * c[k * 3].sqrt());
            }
        }
        let rate = covered as f64 / REPLICATES as f64;
        eprintln!(
            "joint cluster Wald region: {covered}/{REPLICATES} = {rate:.4}; marginals {marginal_covered:?}"
        );
        assert!((0.93..=0.99).contains(&rate), "joint coverage {rate}");
        for m in marginal_covered {
            assert!((0.93..=0.99).contains(&(m as f64 / REPLICATES as f64)));
        }
    }

    #[test]
    fn refuses_misaligned_sparse_or_nonfinite_inputs() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let assigned = [true, true, false, false];
        assert!(complete_cluster_joint_contrast(&[&a], &assigned, 8).is_some());
        assert!(complete_cluster_joint_contrast(&[], &assigned, 8).is_none());
        assert!(complete_cluster_joint_contrast(&[&a[..3]], &assigned, 8).is_none());
        assert!(complete_cluster_joint_contrast(&[&a], &[true, false, false, false], 8).is_none());
        assert!(
            complete_cluster_joint_contrast(&[&[1.0, f64::NAN, 3.0, 4.0]], &assigned, 8).is_none()
        );
        let small = complete_cluster_joint_contrast(&[&a], &assigned, 8).unwrap();
        assert!(!small.interval_supported);
    }
}
