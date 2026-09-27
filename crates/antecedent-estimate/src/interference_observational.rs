//! Supplied-propensity observational exposure contrasts on a fixed network.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use antecedent_core::{ExposureLevel, ExposureMapping, ExposurePropensityProvenance};
use antecedent_data::{NetworkData, TableView};
use antecedent_stats::exposures;

use crate::EstimationError;

/// Row-aligned observation design and requested contrast.
#[derive(Clone, Debug)]
pub struct ObservationalExposureSpec<'a> {
    /// Observed own-treatment assignment in unit-row order.
    pub assignment: &'a [bool],
    /// Partial-interference cluster id in unit-row order.
    pub clusters: &'a [u32],
    /// Built-in neighbor exposure mapping.
    pub exposure: &'a ExposureMapping,
    /// Baseline exposure level.
    pub from: ExposureLevel,
    /// Active exposure level.
    pub to: ExposureLevel,
    /// Marginal probability of the baseline exposure, one per row.
    pub propensity_from: &'a [f64],
    /// Marginal probability of the active exposure, one per row.
    pub propensity_to: &'a [f64],
    /// Provenance of both supplied probability vectors.
    pub propensity_provenance: ExposurePropensityProvenance,
}

/// Point result, descriptive cluster variance, and observed support.
#[derive(Clone, Debug, PartialEq)]
pub struct ObservationalExposureEstimate {
    /// Horvitz–Thompson contrast.
    pub horvitz_thompson: f64,
    /// Hájek contrast.
    pub hajek: f64,
    /// CR1 sandwich variance of the Horvitz–Thompson contrast, treating propensities as fixed.
    pub cluster_robust_variance: f64,
    /// Baseline-exposed units in the realized assignment.
    pub from_exposed_units: usize,
    /// Active-exposed units in the realized assignment.
    pub to_exposed_units: usize,
    /// Clusters containing a baseline-exposed unit.
    pub from_exposed_clusters: usize,
    /// Clusters containing an active-exposed unit.
    pub to_exposed_clusters: usize,
    /// Minimum supplied probability over both levels and all rows.
    pub minimum_exposure_probability: f64,
    /// Maximum supplied probability over both levels and all rows.
    pub maximum_exposure_probability: f64,
    /// Number of clusters in the partial-interference partition.
    pub clusters: usize,
    /// Source of supplied exposure probabilities.
    pub propensity_provenance: ExposurePropensityProvenance,
    /// Pointwise 95% cluster interval for HT when exposure probabilities are known.
    pub pointwise_interval: Option<ObservationalExposureInterval>,
    /// Why a pointwise interval was withheld, when one was not reported.
    pub interval_unavailable_reason: Option<&'static str>,
}

/// Pointwise uncertainty for the HT contrast across independent network clusters.
#[derive(Clone, Debug, PartialEq)]
pub struct ObservationalExposureInterval {
    /// Lower and upper pointwise 95% bounds.
    pub bounds: [f64; 2],
    /// Cluster score standard error of the Horvitz–Thompson contrast.
    pub standard_error: f64,
    /// Number of independent clusters minus one.
    pub degrees_of_freedom: f64,
}

/// Estimate an observational exposure contrast with supplied marginal probabilities.
/// Identification requires a caller-declared network exchangeability assumption;
/// this kernel checks topology and positivity but cannot verify exchangeability.
/// A pointwise cluster interval is returned only for known probabilities with
/// sufficient independent and exposed clusters. Externally fitted probabilities
/// retain a descriptive variance because their fitting uncertainty is unknown.
///
/// # Errors
///
/// Refuses invalid dimensions, exposure levels, propensities, topology, empirical
/// support, or fewer than two partial-interference clusters.
#[allow(clippy::too_many_lines)]
pub fn estimate_observational_exposure(
    data: &NetworkData,
    outcome: antecedent_core::VariableId,
    spec: &ObservationalExposureSpec<'_>,
) -> Result<ObservationalExposureEstimate, EstimationError> {
    let n = data.units().row_count();
    if n == 0
        || spec.assignment.len() != n
        || spec.clusters.len() != n
        || spec.propensity_from.len() != n
        || spec.propensity_to.len() != n
    {
        return Err(EstimationError::data_msg(
            "outcome, assignment, clusters, and propensity vectors must have equal non-zero length",
        ));
    }
    if !matches!(
        spec.exposure,
        ExposureMapping::NeighborCount
            | ExposureMapping::NeighborFraction
            | ExposureMapping::WeightedNeighborExposure
    ) {
        return Err(EstimationError::unsupported(
            "observational network exposure requires a built-in neighbor exposure mapping",
        ));
    }
    if [spec.from.own, spec.from.neighbors, spec.to.own, spec.to.neighbors]
        .iter()
        .any(|value| !value.is_finite())
        || ((spec.from.own - spec.to.own).abs() <= 1e-12
            && (spec.from.neighbors - spec.to.neighbors).abs() <= 1e-12)
    {
        return Err(EstimationError::data_msg(
            "exposure contrast levels must be finite and distinct",
        ));
    }
    if spec
        .propensity_from
        .iter()
        .chain(spec.propensity_to)
        .any(|p| !p.is_finite() || *p <= 0.0 || *p > 1.0)
    {
        return Err(EstimationError::data_msg(
            "observational exposure positivity failure: probabilities must lie in (0, 1]",
        ));
    }
    if spec.propensity_from.iter().zip(spec.propensity_to).any(|(from, to)| from + to > 1.0 + 1e-12) {
        return Err(EstimationError::data_msg(
            "distinct exposure levels cannot have marginal probabilities summing above one",
        ));
    }
    let cluster_ids = spec.clusters.iter().copied().collect::<BTreeSet<_>>();
    if cluster_ids.len() < 2 {
        return Err(EstimationError::unsupported(
            "cluster variance requires at least two partial-interference clusters",
        ));
    }
    let outcomes = data.units().float64_values(outcome)?;
    if outcomes.iter().any(|value| !value.is_finite()) {
        return Err(EstimationError::data_msg("observational network outcomes must be finite"));
    }
    let mut incoming = Vec::with_capacity(n);
    for unit in 0..n {
        let neighbors = data
            .incoming(unit)?
            .iter()
            .map(|edge| (edge.from as usize, edge.weight))
            .collect::<Vec<_>>();
        if neighbors.iter().any(|&(source, weight)| {
            spec.clusters[source] != spec.clusters[unit] || !weight.is_finite() || weight < 0.0
        }) {
            return Err(EstimationError::unsupported(
                "partial-interference assumption violated: network edge crosses cluster boundary",
            ));
        }
        incoming.push(neighbors);
    }
    let observed = exposures(spec.assignment, &incoming, spec.exposure)?;
    let mut from_sum = 0.0;
    let mut to_sum = 0.0;
    let mut from_weight = 0.0;
    let mut to_weight = 0.0;
    let mut from_count = 0;
    let mut to_count = 0;
    let mut from_clusters = BTreeSet::new();
    let mut to_clusters = BTreeSet::new();
    let mut influence_by_cluster = BTreeMap::<u32, f64>::new();
    let mut minimum_probability = f64::INFINITY;
    let mut maximum_probability: f64 = 0.0;
    for i in 0..n {
        let observed_level = observed[i];
        let is_from = (observed_level.own - spec.from.own).abs() <= 1e-12
            && (observed_level.neighbors - spec.from.neighbors).abs() <= 1e-12;
        let is_to = (observed_level.own - spec.to.own).abs() <= 1e-12
            && (observed_level.neighbors - spec.to.neighbors).abs() <= 1e-12;
        let from_score = if is_from { outcomes[i] / spec.propensity_from[i] } else { 0.0 };
        let to_score = if is_to { outcomes[i] / spec.propensity_to[i] } else { 0.0 };
        from_sum += from_score;
        to_sum += to_score;
        if is_from {
            from_weight += 1.0 / spec.propensity_from[i];
            from_count += 1;
            from_clusters.insert(spec.clusters[i]);
        }
        if is_to {
            to_weight += 1.0 / spec.propensity_to[i];
            to_count += 1;
            to_clusters.insert(spec.clusters[i]);
        }
        *influence_by_cluster.entry(spec.clusters[i]).or_default() +=
            (to_score - from_score) / n as f64;
        minimum_probability =
            minimum_probability.min(spec.propensity_from[i]).min(spec.propensity_to[i]);
        maximum_probability =
            maximum_probability.max(spec.propensity_from[i]).max(spec.propensity_to[i]);
    }
    if from_count == 0 || to_count == 0 || from_weight == 0.0 || to_weight == 0.0 {
        return Err(EstimationError::unsupported(
            "observational network contrast requires observed support at both requested exposure levels",
        ));
    }
    let horvitz_thompson = (to_sum - from_sum) / n as f64;
    let hajek = to_sum / to_weight - from_sum / from_weight;
    let cluster_count = influence_by_cluster.len();
    let mean_influence = influence_by_cluster.values().sum::<f64>() / cluster_count as f64;
    let cluster_robust_variance = cluster_count as f64 / (cluster_count - 1) as f64
        * influence_by_cluster.values().map(|value| (value - mean_influence).powi(2)).sum::<f64>();
    if !horvitz_thompson.is_finite() || !hajek.is_finite() || !cluster_robust_variance.is_finite() {
        return Err(EstimationError::data_msg(
            "observational network estimate overflowed finite precision",
        ));
    }
    let interval_unavailable_reason = if spec.propensity_provenance
        != ExposurePropensityProvenance::Known
    {
        Some("pointwise observational inference requires known, fixed exposure probabilities")
    } else if cluster_count < 30 || from_clusters.len() < 8 || to_clusters.len() < 8 {
        Some("pointwise observational inference requires 30 independent clusters and eight exposed clusters at each level")
    } else if cluster_robust_variance <= 0.0 {
        Some("pointwise observational inference requires positive between-cluster score variation")
    } else {
        None
    };
    let pointwise_interval = if interval_unavailable_reason.is_none() {
        let degrees_of_freedom = (cluster_count - 1) as f64;
        let standard_error = cluster_robust_variance.sqrt();
        let critical = antecedent_stats::student_t_ppf(0.975, degrees_of_freedom);
        if !critical.is_finite() {
            return Err(EstimationError::data_msg("pointwise observational critical value is unavailable"));
        }
        Some(ObservationalExposureInterval {
            bounds: [horvitz_thompson - critical * standard_error, horvitz_thompson + critical * standard_error],
            standard_error,
            degrees_of_freedom,
        })
    } else {
        None
    };
    Ok(ObservationalExposureEstimate {
        horvitz_thompson,
        hajek,
        cluster_robust_variance,
        from_exposed_units: from_count,
        to_exposed_units: to_count,
        from_exposed_clusters: from_clusters.len(),
        to_exposed_clusters: to_clusters.len(),
        minimum_exposure_probability: minimum_probability,
        maximum_exposure_probability: maximum_probability,
        clusters: cluster_count,
        propensity_provenance: spec.propensity_provenance,
        pointwise_interval,
        interval_unavailable_reason,
    })
}

#[cfg(test)]
mod tests {
    #![cfg_attr(test, allow(clippy::float_cmp, clippy::cast_possible_truncation, reason = "fixtures build small integer labels; tests assert exact deterministic values"))]
    use antecedent_core::{ExposureLevel, ExposureMapping, ExposurePropensityProvenance, VariableId};
    use antecedent_data::{NetworkData, NetworkEdge, TabularData};

    use super::{ObservationalExposureSpec, estimate_observational_exposure};

    struct XorShift(u64);

    impl XorShift {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn uniform(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1_u64 << 53) as f64
        }
    }

    #[test]
    fn observational_known_exposure_interval_covers_known_truth() {
        // Fixed finite population of unit potential outcomes; per-unit exposure is
        // re-randomized by independent Bernoulli(0.5) own-treatment within each
        // two-unit cluster, so the marginal probability of the (0,0) and (1,1)
        // levels is exactly 0.25. Known supplied propensities make the IPW
        // contrast design-unbiased and the CR1 cluster interval cover at the
        // nominal rate. Thirty independent clusters is the licensed support floor;
        // fifty keeps both requested levels comfortably above eight exposed
        // clusters. The (0,0) -> (1,1) contrast has known truth 3.5.
        const REPLICATES: usize = 2_000;
        const CLUSTERS: usize = 120;
        let truth = 3.5_f64;
        let labels = (0..CLUSTERS).flat_map(|cluster| [cluster as u32; 2]).collect::<Vec<_>>();
        let edges = (0..CLUSTERS)
            .flat_map(|cluster| {
                let first = (2 * cluster) as u32;
                [
                    NetworkEdge { from: first, to: first + 1, weight: 1.0 },
                    NetworkEdge { from: first + 1, to: first, weight: 1.0 },
                ]
            })
            .collect::<Vec<_>>();
        let base = (0..CLUSTERS * 2)
            .map(|unit| 0.5 * (unit / 2 % 5) as f64 + 0.1 * (unit % 2) as f64)
            .collect::<Vec<_>>();
        let probabilities = vec![0.25; CLUSTERS * 2];
        let mut rng = XorShift(0x39F1_77C4_2B8E_A013);
        let mut covered = 0_usize;
        let mut supported = 0_usize;
        for _ in 0..REPLICATES {
            let assignment =
                (0..CLUSTERS * 2).map(|_| rng.uniform() < 0.5).collect::<Vec<_>>();
            let outcomes = (0..CLUSTERS * 2)
                .map(|unit| {
                    let own = f64::from(assignment[unit]);
                    let neighbor = f64::from(assignment[unit ^ 1]);
                    base[unit] + 2.0 * own + 1.5 * neighbor
                })
                .collect::<Vec<_>>();
            let data = TabularData::from_f64_columns([("y", outcomes.as_slice())]).unwrap();
            let network = NetworkData::try_new(data, edges.clone()).unwrap();
            let spec = ObservationalExposureSpec {
                assignment: &assignment,
                clusters: &labels,
                exposure: &ExposureMapping::NeighborFraction,
                from: ExposureLevel { own: 0.0, neighbors: 0.0 },
                to: ExposureLevel { own: 1.0, neighbors: 1.0 },
                propensity_from: &probabilities,
                propensity_to: &probabilities,
                propensity_provenance: ExposurePropensityProvenance::Known,
            };
            let result =
                estimate_observational_exposure(&network, VariableId::from_raw(0), &spec).unwrap();
            if let Some(interval_95) = result.pointwise_interval {
                supported += 1;
                covered += usize::from(
                    interval_95.bounds[0] <= truth && truth <= interval_95.bounds[1],
                );
            }
        }
        let rate = covered as f64 / supported as f64;
        eprintln!("observational known: {covered}/{supported}, coverage={rate:.4}");
        assert!(supported >= 1_950, "too many sparse refusals: {supported}");
        assert!((0.93..=0.985).contains(&rate), "pointwise 95% coverage {rate}");
    }

    #[test]
    fn supplied_propensities_recover_known_network_contrast_and_cluster_variance() {
        let data = TabularData::from_f64_columns([("y", &[0.0, 0.0, 6.0, 6.0][..])]).unwrap();
        let edges = vec![
            NetworkEdge { from: 0, to: 1, weight: 1.0 },
            NetworkEdge { from: 1, to: 0, weight: 1.0 },
            NetworkEdge { from: 2, to: 3, weight: 1.0 },
            NetworkEdge { from: 3, to: 2, weight: 1.0 },
        ];
        let network = NetworkData::try_new(data, edges).unwrap();
        let spec = ObservationalExposureSpec {
            assignment: &[false, false, true, true],
            clusters: &[0, 0, 1, 1],
            exposure: &ExposureMapping::NeighborCount,
            from: ExposureLevel { own: 0.0, neighbors: 0.0 },
            to: ExposureLevel { own: 1.0, neighbors: 1.0 },
            propensity_from: &[0.5; 4],
            propensity_to: &[0.5; 4],
            propensity_provenance: ExposurePropensityProvenance::ExternallyEstimated,
        };
        let result =
            estimate_observational_exposure(&network, VariableId::from_raw(0), &spec).unwrap();
        assert!((result.horvitz_thompson - 6.0).abs() < 1e-12);
        assert!((result.hajek - 6.0).abs() < 1e-12);
        assert!((result.cluster_robust_variance - 36.0).abs() < 1e-12);
        assert_eq!((result.from_exposed_units, result.to_exposed_units), (2, 2));
        assert_eq!((result.from_exposed_clusters, result.to_exposed_clusters), (1, 1));
        assert_eq!(result.clusters, 2);
        assert_eq!(result.propensity_provenance, ExposurePropensityProvenance::ExternallyEstimated);
        assert!(result.pointwise_interval.is_none());
        assert!(result.interval_unavailable_reason.unwrap().contains("known, fixed"));
    }

    #[test]
    fn known_probabilities_publish_pointwise_interval_with_cluster_support() {
        let clusters = 60;
        let mut outcomes = Vec::new();
        let mut assignment = Vec::new();
        let mut labels = Vec::new();
        let mut edges = Vec::new();
        for cluster in 0..clusters {
            let treated = cluster % 2 == 1;
            let residual = (cluster % 7) as f64 / 10.0;
            for unit in 0..2 {
                outcomes.push(1.0 + 2.0 * f64::from(treated) + residual);
                assignment.push(treated);
                labels.push(cluster as u32);
                edges.push(NetworkEdge { from: (2 * cluster + 1 - unit) as u32, to: (2 * cluster + unit) as u32, weight: 1.0 });
            }
        }
        let data = TabularData::from_f64_columns([("y", outcomes.as_slice())]).unwrap();
        let network = NetworkData::try_new(data, edges).unwrap();
        let probabilities = vec![0.5; 2 * clusters];
        let spec = ObservationalExposureSpec {
            assignment: &assignment,
            clusters: &labels,
            exposure: &ExposureMapping::NeighborCount,
            from: ExposureLevel { own: 0.0, neighbors: 0.0 },
            to: ExposureLevel { own: 1.0, neighbors: 1.0 },
            propensity_from: &probabilities,
            propensity_to: &probabilities,
            propensity_provenance: ExposurePropensityProvenance::Known,
        };
        let result = estimate_observational_exposure(&network, VariableId::from_raw(0), &spec).unwrap();
        let interval = result.pointwise_interval.unwrap();
        assert!(interval.bounds[0] < 2.0 && 2.0 < interval.bounds[1]);
        assert_eq!(interval.degrees_of_freedom, 59.0);
        assert_eq!((result.from_exposed_clusters, result.to_exposed_clusters), (30, 30));
        assert!(result.interval_unavailable_reason.is_none());
    }

    #[test]
    fn impossible_marginal_probabilities_are_refused() {
        let data = TabularData::from_f64_columns([("y", &[0.0, 0.0, 6.0, 6.0][..])]).unwrap();
        let network = NetworkData::try_new(data, vec![]).unwrap();
        let spec = ObservationalExposureSpec {
            assignment: &[false, false, true, true],
            clusters: &[0, 0, 1, 1],
            exposure: &ExposureMapping::NeighborCount,
            from: ExposureLevel { own: 0.0, neighbors: 0.0 },
            to: ExposureLevel { own: 1.0, neighbors: 1.0 },
            propensity_from: &[0.7; 4],
            propensity_to: &[0.7; 4],
            propensity_provenance: ExposurePropensityProvenance::Known,
        };
        let error = estimate_observational_exposure(&network, VariableId::from_raw(0), &spec).unwrap_err();
        assert!(error.to_string().contains("summing above one"));
    }

    #[test]
    fn observed_exposure_refuses_cross_cluster_network() {
        let data = TabularData::from_f64_columns([("y", &[0.0, 0.0, 6.0, 6.0][..])]).unwrap();
        let network =
            NetworkData::try_new(data, vec![NetworkEdge { from: 0, to: 2, weight: 1.0 }]).unwrap();
        let spec = ObservationalExposureSpec {
            assignment: &[false, false, true, true],
            clusters: &[0, 0, 1, 1],
            exposure: &ExposureMapping::NeighborCount,
            from: ExposureLevel { own: 0.0, neighbors: 0.0 },
            to: ExposureLevel { own: 1.0, neighbors: 1.0 },
            propensity_from: &[0.5; 4],
            propensity_to: &[0.5; 4],
            propensity_provenance: ExposurePropensityProvenance::Known,
        };
        assert!(estimate_observational_exposure(&network, VariableId::from_raw(0), &spec).is_err());
    }
}
