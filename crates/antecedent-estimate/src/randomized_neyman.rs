//! Design-based inference for complete randomization of independent clusters.
// SPDX-License-Identifier: MIT OR Apache-2.0

/// Minimum independently randomized clusters in each arm for a normal interval.
pub const MIN_CLUSTERS_PER_ARM_FOR_INTERVAL: usize = 30;
const NORMAL_95: f64 = 1.959_963_984_540_054;

/// A cluster-randomized intention-to-treat contrast over outcome totals.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClusterItt {
    /// Mean outcome contrast per outcome row.
    pub effect: f64,
    /// Neyman's conservative variance estimate, omitting the unobserved
    /// treatment-effect heterogeneity subtraction.
    pub variance_upper_bound: f64,
    /// Pointwise normal interval when both arms have adequate independent
    /// cluster support and the estimated variance is positive.
    pub interval_95: Option<[f64; 2]>,
}

/// Estimate an ITT from cluster outcome totals under complete randomization.
///
/// The caller must supply one total per independently assigned cluster and
/// the total number of observed outcome rows. Unequal cluster sizes are
/// allowed; the scaling targets the average effect per outcome row.
/// Returns `None` for an unestimable variance, nonfinite outcomes, or empty
/// arms. Sparse arms retain a point and variance but no interval.
#[must_use]
pub fn complete_cluster_itt(
    treated_totals: &[f64],
    control_totals: &[f64],
    outcome_rows: usize,
) -> Option<ClusterItt> {
    if treated_totals.len() < 2 || control_totals.len() < 2 || outcome_rows == 0
        || treated_totals.iter().chain(control_totals).any(|value| !value.is_finite())
    {
        return None;
    }
    let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
    let variance = |values: &[f64]| {
        let average = mean(values);
        values.iter().map(|value| (value - average).powi(2)).sum::<f64>()
            / (values.len() - 1) as f64
    };
    let clusters = treated_totals.len() + control_totals.len();
    let scale = clusters as f64 / outcome_rows as f64;
    let effect = scale * (mean(treated_totals) - mean(control_totals));
    let variance_upper_bound = scale.powi(2)
        * (variance(treated_totals) / treated_totals.len() as f64
            + variance(control_totals) / control_totals.len() as f64);
    if !effect.is_finite() || !variance_upper_bound.is_finite() || variance_upper_bound < 0.0 {
        return None;
    }
    let interval_95 = (treated_totals.len() >= MIN_CLUSTERS_PER_ARM_FOR_INTERVAL
        && control_totals.len() >= MIN_CLUSTERS_PER_ARM_FOR_INTERVAL
        && variance_upper_bound > 0.0)
        .then(|| {
            let radius = NORMAL_95 * variance_upper_bound.sqrt();
            [effect - radius, effect + radius]
        });
    Some(ClusterItt { effect, variance_upper_bound, interval_95 })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_cluster_arms_keep_variance_without_interval() {
        let fit = complete_cluster_itt(&[14.0, 4.0], &[4.0, 2.0], 6).unwrap();
        assert_eq!(fit.effect, 4.0);
        assert!(fit.variance_upper_bound > 0.0);
        assert_eq!(fit.interval_95, None);
    }

    #[test]
    fn calibrated_support_requires_independent_clusters_in_each_arm() {
        let treated = (0..30).map(|i| 3.0 + f64::from(i) / 10.0).collect::<Vec<_>>();
        let control = (0..30).map(|i| 1.0 + f64::from(i) / 10.0).collect::<Vec<_>>();
        assert!(complete_cluster_itt(&treated, &control, 60).unwrap().interval_95.is_some());
        assert!(complete_cluster_itt(&treated[..29], &control, 59).unwrap().interval_95.is_none());
        assert!(complete_cluster_itt(&treated, &control, 0).is_none());
    }

    #[test]
    fn fixed_population_cluster_randomization_covers_known_itt() {
        // Eighty independent clusters, unequal sizes, heterogeneous baseline
        // outcomes and effects. Only the allocation changes across studies.
        const CLUSTERS: usize = 80;
        const REPLICATES: usize = 2_000;
        let sizes = (0..CLUSTERS).map(|i| 2 + i % 3).collect::<Vec<_>>();
        let rows = sizes.iter().sum::<usize>();
        let control = (0..CLUSTERS).map(|i| {
            sizes[i] as f64 * (1.0 + 0.6 * (i as f64 * 0.37).sin())
        }).collect::<Vec<_>>();
        let treated = (0..CLUSTERS).map(|i| {
            control[i] + sizes[i] as f64 * (2.0 + 0.25 * (i as f64 * 0.53).cos())
        }).collect::<Vec<_>>();
        let truth = treated.iter().zip(&control).map(|(y1, y0)| y1 - y0).sum::<f64>()
            / rows as f64;
        let mut covered = 0;
        for rep in 0..REPLICATES {
            let mut state = rep as u64 + 0xD35A_7100_0000_0001;
            let mut order = (0..CLUSTERS).collect::<Vec<_>>();
            for i in (1..CLUSTERS).rev() {
                state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut x = state;
                x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                let j = ((x ^ (x >> 31)) % (i as u64 + 1)) as usize;
                order.swap(i, j);
            }
            let treatment = order[..CLUSTERS / 2].iter().map(|&i| treated[i]).collect::<Vec<_>>();
            let comparison = order[CLUSTERS / 2..].iter().map(|&i| control[i]).collect::<Vec<_>>();
            let fit = complete_cluster_itt(&treatment, &comparison, rows).unwrap();
            let [lower, upper] = fit.interval_95.expect("40 clusters per arm are supported");
            covered += usize::from(lower <= truth && truth <= upper);
        }
        // Binomial Monte Carlo uncertainty at nominal 95% is about 0.5 pp.
        // This gate allows finite-population conservatism without licensing
        // severe undercoverage.
        let rate = covered as f64 / REPLICATES as f64;
        eprintln!("cluster Neyman interval: {covered}/{REPLICATES} = {rate:.4}");
        assert!((0.93..=0.985).contains(&rate));
    }
}
