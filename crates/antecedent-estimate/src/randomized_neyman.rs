//! Design-based inference for complete randomization of independent clusters.
// SPDX-License-Identifier: MIT OR Apache-2.0

/// Minimum independently randomized clusters in each arm for a normal interval.
pub const MIN_CLUSTERS_PER_ARM_FOR_INTERVAL: usize = 30;
const NORMAL_95: f64 = 1.959_963_984_540_054;

/// Minimum independently assigned units in each arm for a complete-design interval.
pub const MIN_UNITS_PER_ARM_FOR_INTERVAL: usize = 30;

/// A completely randomized intention-to-treat contrast.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompleteItt {
    /// Difference between treated and control outcome means.
    pub effect: f64,
    /// Conservative Neyman variance, omitting the unobservable effect variance.
    pub variance_upper_bound: f64,
    /// Pointwise normal interval when both arms have adequate assigned units.
    pub interval_95: Option<[f64; 2]>,
}

/// Estimate a two-arm ITT under complete unit randomization.
///
/// Returns `None` when the arm means or variance cannot be estimated.
/// Sparse arms retain their point and variance without an interval.
#[must_use]
pub fn complete_unit_itt(treated: &[f64], control: &[f64]) -> Option<CompleteItt> {
    if treated.len() < 2 || control.len() < 2
        || treated.iter().chain(control).any(|value| !value.is_finite())
    {
        return None;
    }
    let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
    let variance = |values: &[f64]| {
        let average = mean(values);
        values.iter().map(|value| (value - average).powi(2)).sum::<f64>()
            / (values.len() - 1) as f64
    };
    let effect = mean(treated) - mean(control);
    let variance_upper_bound = variance(treated) / treated.len() as f64
        + variance(control) / control.len() as f64;
    if !effect.is_finite() || !variance_upper_bound.is_finite() || variance_upper_bound < 0.0 {
        return None;
    }
    let interval_95 = (treated.len() >= MIN_UNITS_PER_ARM_FOR_INTERVAL
        && control.len() >= MIN_UNITS_PER_ARM_FOR_INTERVAL
        && variance_upper_bound > 0.0)
        .then(|| {
            let radius = NORMAL_95 * variance_upper_bound.sqrt();
            [effect - radius, effect + radius]
        });
    Some(CompleteItt { effect, variance_upper_bound, interval_95 })
}

/// Minimum treated and control units within every block for a blocked interval.
pub const MIN_UNITS_PER_BLOCK_ARM_FOR_INTERVAL: usize = 15;

/// A blocked, completely randomized intention-to-treat contrast.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockedItt {
    /// Block-size weighted difference in outcome means.
    pub effect: f64,
    /// Sum of block-size squared weights times conservative Neyman variances.
    pub variance_upper_bound: f64,
    /// Pointwise normal interval under adequately supported independent blocks.
    pub interval_95: Option<[f64; 2]>,
}

/// Estimate an ITT under separate complete randomization within fixed blocks.
///
/// Every block must contain two observed units in each arm to estimate its
/// Neyman variance. The interval additionally requires four blocks, at least
/// 15 units per block arm, and at least 60 units per arm overall.
#[must_use]
pub fn blocked_unit_itt(blocks: &[(&[f64], &[f64])]) -> Option<BlockedItt> {
    let rows = blocks.iter().map(|(treated, control)| treated.len() + control.len())
        .sum::<usize>();
    if rows == 0 || blocks.is_empty() { return None; }
    let mut effect = 0.0;
    let mut variance_upper_bound = 0.0;
    let mut treated_total = 0;
    let mut control_total = 0;
    let mut supported = blocks.len() >= 4;
    for (treated, control) in blocks {
        let fit = complete_unit_itt(treated, control)?;
        let weight = (treated.len() + control.len()) as f64 / rows as f64;
        effect += weight * fit.effect;
        variance_upper_bound += weight.powi(2) * fit.variance_upper_bound;
        treated_total += treated.len();
        control_total += control.len();
        supported &= treated.len() >= MIN_UNITS_PER_BLOCK_ARM_FOR_INTERVAL
            && control.len() >= MIN_UNITS_PER_BLOCK_ARM_FOR_INTERVAL;
    }
    supported &= treated_total >= 60 && control_total >= 60 && variance_upper_bound > 0.0;
    if !effect.is_finite() || !variance_upper_bound.is_finite() { return None; }
    let interval_95 = supported.then(|| {
        let radius = NORMAL_95 * variance_upper_bound.sqrt();
        [effect - radius, effect + radius]
    });
    Some(BlockedItt { effect, variance_upper_bound, interval_95 })
}

/// Minimum observed units in each factorial cell for pointwise intervals.
pub const MIN_UNITS_PER_FACTORIAL_CELL_FOR_INTERVAL: usize = 30;

/// Main effects and interaction from a fixed-cell 2×2 factorial trial.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Factorial2x2 {
    /// Primary-factor main effect, averaging the two secondary-factor levels.
    pub primary: f64,
    /// Secondary-factor main effect, averaging the two primary-factor levels.
    pub secondary: f64,
    /// Difference in primary-factor effects between secondary-factor levels.
    pub interaction: f64,
    /// Conservative variance for either main effect.
    pub main_effect_variance_upper_bound: f64,
    /// Conservative variance for the interaction.
    pub interaction_variance_upper_bound: f64,
    /// Pointwise primary main-effect interval at adequately supported cells.
    pub primary_interval_95: Option<[f64; 2]>,
    /// Pointwise secondary main-effect interval at adequately supported cells.
    pub secondary_interval_95: Option<[f64; 2]>,
    /// Pointwise interaction interval at adequately supported cells.
    pub interaction_interval_95: Option<[f64; 2]>,
}

/// Estimate a fixed-cell 2×2 factorial trial from cells `[00, 10, 01, 11]`.
///
/// Each cell needs two finite observations to estimate its variance. The
/// intervals require at least 30 in every cell and positive variance. They
/// are pointwise; simultaneous familywise coverage is not claimed.
#[must_use]
pub fn factorial_2x2(cells: [&[f64]; 4]) -> Option<Factorial2x2> {
    if cells.iter().any(|cell| cell.len() < 2 || cell.iter().any(|value| !value.is_finite())) {
        return None;
    }
    let means = cells.map(|cell| cell.iter().sum::<f64>() / cell.len() as f64);
    let components = std::array::from_fn::<_, 4, _>(|i| {
        let mean = means[i];
        cells[i].iter().map(|value| (value - mean).powi(2)).sum::<f64>()
            / ((cells[i].len() - 1) * cells[i].len()) as f64
    });
    let primary = 0.5 * (means[1] - means[0] + means[3] - means[2]);
    let secondary = 0.5 * (means[2] - means[0] + means[3] - means[1]);
    let interaction = means[3] - means[2] - means[1] + means[0];
    let interaction_variance_upper_bound = components.iter().sum::<f64>();
    let main_effect_variance_upper_bound = interaction_variance_upper_bound / 4.0;
    if !primary.is_finite() || !secondary.is_finite() || !interaction.is_finite()
        || !interaction_variance_upper_bound.is_finite()
    {
        return None;
    }
    let supported = cells.iter().all(|cell| cell.len() >= MIN_UNITS_PER_FACTORIAL_CELL_FOR_INTERVAL)
        && interaction_variance_upper_bound > 0.0;
    let main_radius = NORMAL_95 * main_effect_variance_upper_bound.sqrt();
    let interaction_radius = NORMAL_95 * interaction_variance_upper_bound.sqrt();
    Some(Factorial2x2 {
        primary,
        secondary,
        interaction,
        main_effect_variance_upper_bound,
        interaction_variance_upper_bound,
        primary_interval_95: supported.then_some([primary - main_radius, primary + main_radius]),
        secondary_interval_95: supported.then_some([secondary - main_radius, secondary + main_radius]),
        interaction_interval_95: supported.then_some([interaction - interaction_radius, interaction + interaction_radius]),
    })
}

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

    fn allocation(size: usize, rep: usize) -> Vec<usize> {
        let mut state = rep as u64 + 0xD35A_7100_0000_0001;
        let mut order = (0..size).collect::<Vec<_>>();
        for i in (1..size).rev() {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = state;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            let j = ((x ^ (x >> 31)) % (i as u64 + 1)) as usize;
            order.swap(i, j);
        }
        order
    }

    #[test]
    fn complete_unit_randomization_covers_known_itt_and_withholds_sparse_interval() {
        const UNITS: usize = 80;
        const REPLICATES: usize = 2_000;
        let control = (0..UNITS).map(|i| 1.0 + 0.7 * (i as f64 * 0.41).sin()).collect::<Vec<_>>();
        let treated = (0..UNITS).map(|i| control[i] + 2.0 + 0.3 * (i as f64 * 0.29).cos())
            .collect::<Vec<_>>();
        let truth = treated.iter().zip(&control).map(|(y1, y0)| y1 - y0).sum::<f64>()
            / UNITS as f64;
        let mut covered = 0;
        for rep in 0..REPLICATES {
            let order = allocation(UNITS, rep);
            let treatment = order[..UNITS / 2].iter().map(|&i| treated[i]).collect::<Vec<_>>();
            let comparison = order[UNITS / 2..].iter().map(|&i| control[i]).collect::<Vec<_>>();
            let [lower, upper] = complete_unit_itt(&treatment, &comparison).unwrap().interval_95
                .expect("40 units per arm are supported");
            covered += usize::from(lower <= truth && truth <= upper);
        }
        let rate = covered as f64 / REPLICATES as f64;
        eprintln!("complete Neyman interval: {covered}/{REPLICATES} = {rate:.4}");
        assert!((0.93..=0.985).contains(&rate));
        assert!(complete_unit_itt(&treated[..29], &control[..30]).unwrap().interval_95.is_none());
    }

    #[test]
    fn blocked_randomization_covers_known_itt_and_withholds_sparse_blocks() {
        const BLOCKS: usize = 4;
        const UNITS_PER_BLOCK: usize = 30;
        const REPLICATES: usize = 2_000;
        let potential = (0..BLOCKS).map(|block| {
            (0..UNITS_PER_BLOCK).map(|i| {
                let baseline = block as f64 * 3.0 + 0.7 * (i as f64 * 0.41).sin();
                (baseline, baseline + 2.0 + 0.25 * (i as f64 * 0.27).cos())
            }).collect::<Vec<_>>()
        }).collect::<Vec<_>>();
        let truth = potential.iter().flatten().map(|(y0, y1)| y1 - y0).sum::<f64>()
            / (BLOCKS * UNITS_PER_BLOCK) as f64;
        let mut covered = 0;
        for rep in 0..REPLICATES {
            let observed = (0..BLOCKS).map(|block| {
                let order = allocation(UNITS_PER_BLOCK, rep * BLOCKS + block);
                let treated = order[..UNITS_PER_BLOCK / 2].iter()
                    .map(|&i| potential[block][i].1).collect::<Vec<_>>();
                let control = order[UNITS_PER_BLOCK / 2..].iter()
                    .map(|&i| potential[block][i].0).collect::<Vec<_>>();
                (treated, control)
            }).collect::<Vec<_>>();
            let refs = observed.iter().map(|(treated, control)|
                (treated.as_slice(), control.as_slice())).collect::<Vec<_>>();
            let [lower, upper] = blocked_unit_itt(&refs).unwrap().interval_95
                .expect("four blocks with 15 units per arm are supported");
            covered += usize::from(lower <= truth && truth <= upper);
        }
        let rate = covered as f64 / REPLICATES as f64;
        eprintln!("blocked Neyman interval: {covered}/{REPLICATES} = {rate:.4}");
        assert!((0.93..=0.985).contains(&rate));
        let sparse = [(&[2.0, 4.0][..], &[1.0, 3.0][..]); BLOCKS];
        assert!(blocked_unit_itt(&sparse).unwrap().interval_95.is_none());
    }

    #[test]
    fn factorial_cell_randomization_covers_three_known_pointwise_effects() {
        const UNITS: usize = 120;
        const CELL_SIZE: usize = 30;
        const REPLICATES: usize = 2_000;
        let potential = (0..UNITS).map(|i| {
            let x = i as f64;
            let baseline = 1.0 + 0.65 * (0.41 * x).sin();
            let primary = 2.0 + 0.2 * (0.23 * x).cos();
            let secondary = 1.0 + 0.15 * (0.19 * x).sin();
            let interaction = 0.5 + 0.1 * (0.31 * x).cos();
            [baseline, baseline + primary, baseline + secondary,
                baseline + primary + secondary + interaction]
        }).collect::<Vec<_>>();
        let average = |f: &dyn Fn(&[f64; 4]) -> f64| potential.iter().map(f).sum::<f64>()
            / UNITS as f64;
        let truths = [
            average(&|y| 0.5 * (y[1] - y[0] + y[3] - y[2])),
            average(&|y| 0.5 * (y[2] - y[0] + y[3] - y[1])),
            average(&|y| y[3] - y[2] - y[1] + y[0]),
        ];
        let mut covered = [0_usize; 3];
        for rep in 0..REPLICATES {
            let order = allocation(UNITS, rep);
            let cells = std::array::from_fn::<_, 4, _>(|cell| {
                order[cell * CELL_SIZE..(cell + 1) * CELL_SIZE].iter()
                    .map(|&i| potential[i][cell]).collect::<Vec<_>>()
            });
            let fit = factorial_2x2(cells.each_ref().map(Vec::as_slice)).unwrap();
            let intervals = [fit.primary_interval_95.unwrap(), fit.secondary_interval_95.unwrap(),
                fit.interaction_interval_95.unwrap()];
            for k in 0..3 {
                covered[k] += usize::from(intervals[k][0] <= truths[k] && truths[k] <= intervals[k][1]);
            }
        }
        eprintln!("factorial pointwise intervals: {covered:?}/{REPLICATES}");
        for count in covered {
            let rate = count as f64 / REPLICATES as f64;
            assert!((0.93..=0.985).contains(&rate));
        }
        let sparse = std::array::from_fn::<_, 4, _>(|_| vec![0.0, 1.0]);
        assert!(factorial_2x2(sparse.each_ref().map(Vec::as_slice)).unwrap()
            .interaction_interval_95.is_none());
    }

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
            let order = allocation(CLUSTERS, rep);
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
