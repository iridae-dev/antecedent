//! Reference posterior of the cluster total-effect contrast for arms of few clusters.
//!
//! Dirichlet weights on the `m` clusters of an arm give its weighted mean the variance
//! `s^2 (m - 1) / (m (m + 1))` against the `s^2 / m` of the mean, with near-normal tails, so the
//! Dirichlet-weight law of [`crate::estimate_cluster_interference_bayesian_bootstrap`] is too
//! narrow for a small arm. Below [`CLUSTER_DIRICHLET_MIN_ARM_CLUSTERS`] clusters in either arm
//! the posterior is the Student-t reference law of the same contrast.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::InterferenceQuery;
use antecedent_data::{NetworkData, TableView};
use std::collections::BTreeMap;

use crate::EstimationError;
use crate::interference::{InterferenceEstimate, estimate_cluster_interference_total};
use crate::score_posterior::{ScorePosterior, ScorePosteriorOptions, summarize};

/// Fewest clusters in each arm at which the Dirichlet-weight law is used: the smallest arm its
/// coverage was measured at (50, 100 and 200 clusters per arm in
/// `interference_pag_graph_posterior_cluster_bayesian_bootstrap_coverage`). The Student-t
/// reference below it was measured at 9, 12 and 24 clusters per arm.
///
/// With `m` clusters in each of two arms of equal variance the Dirichlet-weight 95% half-width
/// is `z sqrt((m - 1) / (m + 1)) / t` of the Student-t reference, `z` the normal and `t` the
/// Student-t 97.5% quantile at `2 (m - 1)` degrees of freedom: 0.827 at 9, 0.934 at 24 and
/// 0.968 at 50, so at fifty clusters the two laws differ by 3.2 percent of the half-width.
pub const CLUSTER_DIRICHLET_MIN_ARM_CLUSTERS: usize = 50;

/// The Student-t reference posterior of the cluster total-effect contrast when either arm has
/// fewer than [`CLUSTER_DIRICHLET_MIN_ARM_CLUSTERS`] clusters, else `None`.
///
/// The law is centred at the design-based contrast, scaled by the conservative cluster Neyman
/// standard error, with Welch-Satterthwaite degrees of freedom (the usual approximation to the
/// Behrens-Fisher posterior of two Gaussian arm means under the noninformative prior). Its
/// draws are the law's quantiles at `(i + 1/2) / draws`, not Monte Carlo samples.
///
/// # Errors
/// Every refusal of [`estimate_cluster_interference_total`], a design that is not a cluster
/// randomization, and options with fewer than two draws or a coverage outside (0, 1).
pub fn estimate_cluster_interference_small_arm_posterior(
    query: &InterferenceQuery,
    data: &NetworkData,
    assignment: &[bool],
    options: ScorePosteriorOptions,
) -> Result<Option<(InterferenceEstimate, ScorePosterior)>, EstimationError> {
    let estimate = estimate_cluster_interference_total(query, data, assignment)?;
    let antecedent_core::AssignmentDesign::ClusterRandomization { clusters, .. } =
        &query.assignment
    else {
        return Err(EstimationError::unsupported_coded(
            antecedent_core::reason_code!("route_not_supported"),
            "cluster total-effect posterior requires cluster randomization",
        ));
    };
    let antecedent_core::InterferenceFunctional::ExposureContrast { outcome, .. } =
        query.functional;
    let outcomes = data.units().float64_values(outcome)?;
    let mut by_cluster: BTreeMap<u32, (bool, f64)> = BTreeMap::new();
    for ((&cluster, &assigned), value) in clusters.iter().zip(assignment).zip(outcomes) {
        by_cluster.entry(cluster).or_insert((assigned, 0.0)).1 += value;
    }
    let scale = by_cluster.len() as f64 / assignment.len() as f64;
    let arm_totals = |arm: bool| {
        by_cluster
            .values()
            .filter(|entry| entry.0 == arm)
            .map(|entry| entry.1 * scale)
            .collect::<Vec<_>>()
    };
    let (treated_arm, control_arm) = (arm_totals(true), arm_totals(false));
    if treated_arm.len().min(control_arm.len()) >= CLUSTER_DIRICHLET_MIN_ARM_CLUSTERS {
        return Ok(None);
    }
    let posterior = cluster_student_reference(
        estimate.contrast.horvitz_thompson,
        &treated_arm,
        &control_arm,
        options,
    )?;
    Ok(Some((estimate, posterior)))
}

/// Student-t reference law of the cluster total-effect contrast for arms of few clusters: location
/// `centre`, scale the conservative cluster Neyman standard error of the scaled arm totals, and
/// Welch-Satterthwaite degrees of freedom. The draws are its quantiles at `(i + 1/2) / draws`.
fn cluster_student_reference(
    centre: f64,
    treated: &[f64],
    control: &[f64],
    options: ScorePosteriorOptions,
) -> Result<ScorePosterior, EstimationError> {
    if options.draws < 2
        || !options.coverage.is_finite()
        || options.coverage <= 0.0
        || options.coverage >= 1.0
    {
        return Err(EstimationError::data_msg(
            "score posterior requires two draws and a coverage in (0, 1)",
        ));
    }
    // Variance of an arm's mean and its degrees of freedom; every arm has at least two clusters.
    let component = |totals: &[f64]| {
        let clusters = totals.len() as f64;
        let mean = totals.iter().sum::<f64>() / clusters;
        let squares = totals.iter().map(|total| (total - mean).powi(2)).sum::<f64>();
        (squares / (clusters - 1.0) / clusters, clusters - 1.0)
    };
    let ((treated_variance, treated_df), (control_variance, control_df)) =
        (component(treated), component(control));
    let variance = treated_variance + control_variance;
    if !variance.is_finite() {
        return Err(EstimationError::stats_msg("cluster posterior produced a non-finite draw"));
    }
    let draws = if variance > 0.0 {
        let df = variance.powi(2)
            / (treated_variance.powi(2) / treated_df + control_variance.powi(2) / control_df);
        let scale = variance.sqrt();
        student_quantile_grid(options.draws, df)
            .into_iter()
            .map(|quantile| centre + scale * quantile)
            .collect::<Vec<_>>()
    } else {
        // Constant totals in both arms: the contrast is known.
        vec![centre; options.draws]
    };
    if draws.iter().any(|draw| !draw.is_finite()) {
        return Err(EstimationError::stats_msg("cluster posterior produced a non-finite draw"));
    }
    Ok(summarize(draws, centre, options.coverage))
}

/// Quantiles of the standard Student-t law with `df` degrees of freedom at the levels
/// `(i + 1/2) / count`, ascending and symmetric about zero.
fn student_quantile_grid(count: usize, df: f64) -> Vec<f64> {
    let log_norm = antecedent_stats::ln_gamma(0.5 * (df + 1.0))
        - antecedent_stats::ln_gamma(0.5 * df)
        - 0.5 * (df * std::f64::consts::PI).ln();
    let density = |t: f64| (log_norm - 0.5 * (df + 1.0) * (t * t / df).ln_1p()).exp();
    let mut grid = vec![0.0; count];
    let mut quantile = 0.0_f64;
    for index in count / 2..count {
        let upper_tail = ((count - index) as f64 - 0.5) / count as f64;
        // Newton from the previous quantile: the distribution function is concave above zero,
        // so the iterates rise to the root without overshooting it.
        for _ in 0..50 {
            let step =
                (antecedent_stats::student_t_sf(quantile, df) - upper_tail) / density(quantile);
            if !step.is_finite() {
                break;
            }
            quantile += step;
            if step.abs() <= 1e-12 * (1.0 + quantile.abs()) {
                break;
            }
        }
        grid[index] = quantile;
        grid[count - 1 - index] = -quantile;
    }
    grid
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{
        AssignmentDesign, ExposureLevel, ExposureMapping, InterferenceFunctional, VariableId,
    };
    use antecedent_data::TabularData;
    use std::sync::Arc;

    /// Two-unit clusters with alternating arms and a fixed outcome pattern, and the scaled
    /// cluster totals of each arm.
    fn paired_cluster_posterior(
        k: usize,
        draws: usize,
    ) -> (InterferenceEstimate, ScorePosterior, Vec<f64>, Vec<f64>) {
        let outcomes = (0..2 * k)
            .map(|unit| 4.0 + 3.0 * (unit as f64 * 0.7).sin() + f64::from(u8::from(unit % 4 < 2)))
            .collect::<Vec<_>>();
        let clusters = (0..k).flat_map(|c| [u32::try_from(c).unwrap(); 2]).collect::<Vec<_>>();
        let assignment = (0..2 * k).map(|unit| (unit / 2) % 2 == 0).collect::<Vec<_>>();
        let edges = (0..k as u64)
            .flat_map(|c| {
                [
                    antecedent_data::NetworkEdge::new(2 * c, 2 * c + 1, 1.0),
                    antecedent_data::NetworkEdge::new(2 * c + 1, 2 * c, 1.0),
                ]
            })
            .collect::<Vec<_>>();
        let table = TabularData::from_f64_columns([("y", &outcomes[..])]).unwrap();
        let network = NetworkData::try_new(table, edges).unwrap();
        let query = InterferenceQuery::new(
            AssignmentDesign::ClusterRandomization {
                clusters: Arc::from(clusters),
                treated_clusters: k / 2,
            },
            ExposureMapping::NeighborFraction,
            InterferenceFunctional::ExposureContrast {
                outcome: VariableId::from_raw(0),
                from: ExposureLevel { own: 0.0, neighbors: 0.0 },
                to: ExposureLevel { own: 1.0, neighbors: 1.0 },
            },
        );
        let options = ScorePosteriorOptions { draws, coverage: 0.95 };
        let (estimate, posterior) = estimate_cluster_interference_small_arm_posterior(
            &query,
            &network,
            &assignment,
            options,
        )
        .unwrap()
        .expect("an arm below the Dirichlet minimum takes the reference law");
        let scale = k as f64 / (2 * k) as f64;
        let totals = |treated: bool| {
            (0..k)
                .filter(|c| (c % 2 == 0) == treated)
                .map(|c| (outcomes[2 * c] + outcomes[2 * c + 1]) * scale)
                .collect::<Vec<_>>()
        };
        (estimate, posterior, totals(true), totals(false))
    }

    #[test]
    fn cluster_posterior_of_few_clusters_is_the_student_reference_law() {
        // Below fifty clusters per arm the posterior is the Student-t law at the design-based
        // contrast with the cluster Neyman standard error and Welch-Satterthwaite degrees of
        // freedom, so its central 95% interval is the design-based interval's.
        for arm in [2_usize, 4, 9, CLUSTER_DIRICHLET_MIN_ARM_CLUSTERS - 1] {
            let (estimate, posterior, treated, control) = paired_cluster_posterior(2 * arm, 4_000);
            let component = |v: &[f64]| {
                let m = v.iter().sum::<f64>() / v.len() as f64;
                v.iter().map(|x| (x - m).powi(2)).sum::<f64>()
                    / (v.len() - 1) as f64
                    / v.len() as f64
            };
            let (a, b) = (component(&treated), component(&control));
            let se = (a + b).sqrt();
            assert!((se * se - estimate.contrast.conservative_variance).abs() < 1e-12);
            let df = (a + b).powi(2) / ((a * a + b * b) / (arm - 1) as f64);
            let centre = estimate.contrast.horvitz_thompson;
            let critical = antecedent_stats::student_t_ppf(0.975, df);
            let half = critical * se;
            assert!((posterior.posterior_mean - centre).abs() < 1e-9 * (1.0 + half), "arm {arm}");
            assert!((posterior.interval.0 - (centre - half)).abs() < 0.01 * half, "arm {arm}");
            assert!((posterior.interval.1 - (centre + half)).abs() < 0.01 * half, "arm {arm}");
            // The draws are the law's quantile grid: ascending and symmetric about the centre.
            assert!(posterior.draws.windows(2).all(|pair| pair[0] <= pair[1]));
            let draw = posterior.draws[3_000];
            let level = 3_000.5 / 4_000.0;
            let expected = centre + se * antecedent_stats::student_t_ppf(level, df);
            assert!((draw - expected).abs() < 1e-8 * (1.0 + expected.abs()), "arm {arm}");
        }
    }

    #[test]
    fn an_arm_at_the_dirichlet_minimum_takes_no_reference_law() {
        let arm = CLUSTER_DIRICHLET_MIN_ARM_CLUSTERS;
        let outcomes = vec![1.0; 4 * arm];
        let table = TabularData::from_f64_columns([("y", &outcomes[..])]).unwrap();
        let clusters =
            (0..2 * arm).flat_map(|c| [u32::try_from(c).unwrap(); 2]).collect::<Vec<_>>();
        let edges = (0..2 * arm as u64)
            .flat_map(|c| {
                [
                    antecedent_data::NetworkEdge::new(2 * c, 2 * c + 1, 1.0),
                    antecedent_data::NetworkEdge::new(2 * c + 1, 2 * c, 1.0),
                ]
            })
            .collect::<Vec<_>>();
        let network = NetworkData::try_new(table, edges).unwrap();
        let assignment = (0..4 * arm).map(|unit| (unit / 2) % 2 == 0).collect::<Vec<_>>();
        let query = InterferenceQuery::new(
            AssignmentDesign::ClusterRandomization {
                clusters: Arc::from(clusters),
                treated_clusters: arm,
            },
            ExposureMapping::NeighborFraction,
            InterferenceFunctional::ExposureContrast {
                outcome: VariableId::from_raw(0),
                from: ExposureLevel { own: 0.0, neighbors: 0.0 },
                to: ExposureLevel { own: 1.0, neighbors: 1.0 },
            },
        );
        let options = ScorePosteriorOptions { draws: 100, coverage: 0.95 };
        let law = estimate_cluster_interference_small_arm_posterior(
            &query,
            &network,
            &assignment,
            options,
        )
        .unwrap();
        assert!(law.is_none());
    }
}
