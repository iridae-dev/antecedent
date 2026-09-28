//! Repeated-sampling evidence for cluster-level saturation intervals.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "calibration fixtures build small nonnegative cluster/unit counts and labels"
)]

use std::sync::Arc;

use antecedent_core::{
    AssignmentDesign, ExposureLevel, ExposureMapping, InterferenceFunctional, InterferenceQuery,
    VariableId,
};
use antecedent_data::{NetworkData, NetworkEdge, TabularData};
use antecedent_estimate::{
    estimate_cluster_interference_total_with_inference, estimate_saturation_interference,
};

const CLUSTERS: usize = 80;
const UNITS_PER_CLUSTER: usize = 3;

fn next_u64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut x = *state;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn uniform(state: &mut u64) -> f64 {
    (next_u64(state) >> 11) as f64 / (1_u64 << 53) as f64
}

fn centered_noise(state: &mut u64) -> f64 {
    (0..12).map(|_| uniform(state)).sum::<f64>() - 6.0
}

#[derive(Clone)]
struct Contrast {
    mapping: ExposureMapping,
    from: (f64, f64),
    to: (f64, f64),
}

#[test]
// length reflects the calibration protocol; refactor would change the simulation
#[allow(clippy::too_many_lines)]
fn saturation_pointwise_intervals_cover_effects_across_neighbor_mappings() {
    let contrasts = [
        Contrast { mapping: ExposureMapping::NeighborFraction, from: (0.0, 0.5), to: (1.0, 0.5) },
        Contrast { mapping: ExposureMapping::NeighborFraction, from: (0.0, 0.0), to: (0.0, 1.0) },
        Contrast { mapping: ExposureMapping::NeighborFraction, from: (0.0, 0.0), to: (1.0, 1.0) },
        Contrast {
            mapping: ExposureMapping::WeightedNeighborExposure,
            from: (0.0, 1.0 / 3.0),
            to: (1.0, 1.0 / 3.0),
        },
        Contrast { mapping: ExposureMapping::NeighborCount, from: (0.0, 1.0), to: (1.0, 1.0) },
    ];
    let edges = (0..CLUSTERS)
        .flat_map(|cluster| {
            (0..UNITS_PER_CLUSTER).flat_map(move |within| {
                let first = cluster * UNITS_PER_CLUSTER;
                [
                    NetworkEdge {
                        from: (first + (within + 1) % 3) as u32,
                        to: (first + within) as u32,
                        weight: 1.0,
                    },
                    NetworkEdge {
                        from: (first + (within + 2) % 3) as u32,
                        to: (first + within) as u32,
                        weight: 2.0,
                    },
                ]
            })
        })
        .collect::<Vec<_>>();
    let clusters = (0..CLUSTERS).flat_map(|id| [id as u32; UNITS_PER_CLUSTER]).collect::<Vec<_>>();
    let mut state = 0x8A43_F5D2_0977_141B;
    let mut accepted = [0_u32; 5];
    let mut covered = [0_u32; 5];
    for trial in 0..2_000 {
        let mut permutation = (0..CLUSTERS).collect::<Vec<_>>();
        for index in (1..CLUSTERS).rev() {
            permutation.swap(index, next_u64(&mut state) as usize % (index + 1));
        }
        let mut high = [false; CLUSTERS];
        for &cluster in &permutation[..CLUSTERS / 2] {
            high[cluster] = true;
        }
        let mut assignment = vec![false; CLUSTERS * UNITS_PER_CLUSTER];
        let mut realized = vec![0.0; assignment.len()];
        let mut noise = vec![0.0; assignment.len()];
        let mut cluster_noise = [0.0; CLUSTERS];
        let mut beta = [2.0; CLUSTERS];
        let mut gamma = [3.0; CLUSTERS];
        let mut interaction = [1.0; CLUSTERS];
        for cluster in 0..CLUSTERS {
            cluster_noise[cluster] = 0.5 * centered_noise(&mut state);
            if trial % 2 == 0 {
                beta[cluster] += 0.5 * centered_noise(&mut state);
                gamma[cluster] += 0.4 * centered_noise(&mut state);
                interaction[cluster] += 0.2 * centered_noise(&mut state);
            }
            for within in 0..UNITS_PER_CLUSTER {
                let index = cluster * UNITS_PER_CLUSTER + within;
                let p = if high[cluster] { 0.8 } else { 0.2 };
                realized[index] = p;
                assignment[index] = uniform(&mut state) < p;
                noise[index] = 0.25 * centered_noise(&mut state);
            }
        }
        for (contrast_index, contrast) in contrasts.iter().enumerate() {
            let outcomes = (0..assignment.len())
                .map(|index| {
                    let cluster = index / UNITS_PER_CLUSTER;
                    let within = index % UNITS_PER_CLUSTER;
                    let first = cluster * UNITS_PER_CLUSTER;
                    let a = f64::from(assignment[first + (within + 1) % 3]);
                    let b = f64::from(assignment[first + (within + 2) % 3]);
                    let neighbor = match contrast.mapping {
                        ExposureMapping::NeighborFraction => (a + b) / 2.0,
                        ExposureMapping::WeightedNeighborExposure => (a + 2.0 * b) / 3.0,
                        ExposureMapping::NeighborCount => a + b,
                        _ => unreachable!(),
                    };
                    let own = f64::from(assignment[index]);
                    5.0 + beta[cluster] * own
                        + gamma[cluster] * neighbor
                        + interaction[cluster] * own * neighbor
                        + cluster_noise[cluster]
                        + noise[index]
                })
                .collect::<Vec<_>>();
            let data = TabularData::from_f64_columns([("y", outcomes.as_slice())]).unwrap();
            let network = NetworkData::try_new(data, edges.clone()).unwrap();
            let query = InterferenceQuery::new(
                AssignmentDesign::TwoStageSaturation {
                    clusters: Arc::from(clusters.clone()),
                    low_probability: 0.2,
                    high_probability: 0.8,
                    high_clusters: CLUSTERS / 2,
                    realized_saturation: Arc::from(realized.clone()),
                },
                contrast.mapping.clone(),
                InterferenceFunctional::ExposureContrast {
                    outcome: VariableId::from_raw(0),
                    from: ExposureLevel { own: contrast.from.0, neighbors: contrast.from.1 },
                    to: ExposureLevel { own: contrast.to.0, neighbors: contrast.to.1 },
                },
            );
            let result = estimate_saturation_interference(&query, &network, &assignment).unwrap();
            if let Some(interval) = result.pointwise_interval {
                accepted[contrast_index] += 1;
                let truth = (0..CLUSTERS)
                    .map(|cluster| {
                        beta[cluster] * (contrast.to.0 - contrast.from.0)
                            + gamma[cluster] * (contrast.to.1 - contrast.from.1)
                            + interaction[cluster]
                                * (contrast.to.0 * contrast.to.1
                                    - contrast.from.0 * contrast.from.1)
                    })
                    .sum::<f64>()
                    / CLUSTERS as f64;
                covered[contrast_index] +=
                    u32::from(interval.bounds[0] <= truth && truth <= interval.bounds[1]);
            }
        }
    }
    for (index, contrast) in contrasts.iter().enumerate() {
        let rate = f64::from(covered[index]) / f64::from(accepted[index]);
        eprintln!(
            "saturation contrast {index} {:?}: {}/{}, coverage={rate:.4}",
            contrast.mapping, covered[index], accepted[index]
        );
        assert!(
            accepted[index] >= 1_800,
            "contrast {index} {:?}: interval support {}/2000",
            contrast.mapping,
            accepted[index]
        );
        assert!(
            (0.93..=0.97).contains(&rate),
            "contrast {index} {:?}: coverage {rate}",
            contrast.mapping
        );
    }
}

#[test]
fn complete_cluster_randomization_pointwise_total_interval_covers_truth() {
    let edges = (0..CLUSTERS)
        .flat_map(|cluster| {
            (0..UNITS_PER_CLUSTER).map(move |within| {
                let first = cluster * UNITS_PER_CLUSTER;
                NetworkEdge {
                    from: (first + (within + 1) % UNITS_PER_CLUSTER) as u32,
                    to: (first + within) as u32,
                    weight: 1.0,
                }
            })
        })
        .collect::<Vec<_>>();
    let clusters = (0..CLUSTERS).flat_map(|id| [id as u32; UNITS_PER_CLUSTER]).collect::<Vec<_>>();
    let query = InterferenceQuery::new(
        AssignmentDesign::ClusterRandomization {
            clusters: Arc::from(clusters),
            treated_clusters: CLUSTERS / 2,
        },
        ExposureMapping::NeighborFraction,
        InterferenceFunctional::ExposureContrast {
            outcome: VariableId::from_raw(0),
            from: ExposureLevel { own: 0.0, neighbors: 0.0 },
            to: ExposureLevel { own: 1.0, neighbors: 1.0 },
        },
    );
    let mut state = 0xBFD2_5126_F4C0_BA85;
    let mut covered = 0;
    for trial in 0..400 {
        let mut permutation = (0..CLUSTERS).collect::<Vec<_>>();
        for index in (1..CLUSTERS).rev() {
            permutation.swap(index, next_u64(&mut state) as usize % (index + 1));
        }
        let mut assigned_cluster = [false; CLUSTERS];
        for &cluster in &permutation[..CLUSTERS / 2] {
            assigned_cluster[cluster] = true;
        }
        let assignment = (0..CLUSTERS)
            .flat_map(|cluster| [assigned_cluster[cluster]; UNITS_PER_CLUSTER])
            .collect::<Vec<_>>();
        let mut outcomes = Vec::with_capacity(assignment.len());
        let mut total_effect = [6.0; CLUSTERS];
        for cluster in 0..CLUSTERS {
            let shock = 0.5 * centered_noise(&mut state);
            if trial % 2 == 0 {
                total_effect[cluster] += 0.7 * centered_noise(&mut state);
            }
            for _ in 0..UNITS_PER_CLUSTER {
                outcomes.push(
                    5.0 + total_effect[cluster] * f64::from(assigned_cluster[cluster])
                        + shock
                        + 0.25 * centered_noise(&mut state),
                );
            }
        }
        let data = TabularData::from_f64_columns([("y", outcomes.as_slice())]).unwrap();
        let network = NetworkData::try_new(data, edges.clone()).unwrap();
        let (estimate, interval) =
            estimate_cluster_interference_total_with_inference(&query, &network, &assignment)
                .unwrap();
        let interval = interval.expect("40 independent clusters per assignment arm");
        assert!(estimate.contrast.conservative_variance > 0.0);
        let truth = total_effect.iter().sum::<f64>() / CLUSTERS as f64;
        covered += u32::from(interval.bounds[0] <= truth && truth <= interval.bounds[1]);
    }
    assert!(covered >= 360, "cluster total coverage {covered}/400");
}

#[test]
fn complete_cluster_total_interval_calibrates_at_support_boundary() {
    const DRAWS: usize = 2_000;
    for cluster_count in [16, 80] {
        let edges = (0..cluster_count)
            .flat_map(|cluster| {
                let first = (3 * cluster) as u32;
                (0..3).map(move |within| NetworkEdge {
                    from: first + ((within + 1) % 3) as u32,
                    to: first + within as u32,
                    weight: 1.0,
                })
            })
            .collect::<Vec<_>>();
        let clusters = (0..cluster_count).flat_map(|id| [id as u32; 3]).collect::<Vec<_>>();
        let effects = (0..cluster_count)
            .map(|cluster| 2.0 + 0.4 * (0.41 * cluster as f64).cos())
            .collect::<Vec<_>>();
        let truth = effects.iter().sum::<f64>() / cluster_count as f64;
        let query = InterferenceQuery::new(
            AssignmentDesign::ClusterRandomization {
                clusters: Arc::from(clusters),
                treated_clusters: cluster_count / 2,
            },
            ExposureMapping::NeighborFraction,
            InterferenceFunctional::ExposureContrast {
                outcome: VariableId::from_raw(0),
                from: ExposureLevel { own: 0.0, neighbors: 0.0 },
                to: ExposureLevel { own: 1.0, neighbors: 1.0 },
            },
        );
        let mut state = 0x8b63_13ac_7f9d_2e51;
        let mut covered = 0_usize;
        for _ in 0..DRAWS {
            let mut order = (0..cluster_count).collect::<Vec<_>>();
            for i in (1..cluster_count).rev() {
                order.swap(i, next_u64(&mut state) as usize % (i + 1));
            }
            let mut treated = vec![false; cluster_count];
            for &cluster in &order[..cluster_count / 2] {
                treated[cluster] = true;
            }
            let assignment =
                (0..cluster_count).flat_map(|cluster| [treated[cluster]; 3]).collect::<Vec<_>>();
            let outcomes = (0..3 * cluster_count)
                .map(|unit| {
                    let cluster = unit / 3;
                    5.0 + 0.6 * (0.37 * cluster as f64).sin()
                        + 0.1 * (unit % 3) as f64
                        + effects[cluster] * f64::from(treated[cluster])
                })
                .collect::<Vec<_>>();
            let data = TabularData::from_f64_columns([("y", outcomes.as_slice())]).unwrap();
            let network = NetworkData::try_new(data, edges.clone()).unwrap();
            let (_, interval) =
                estimate_cluster_interference_total_with_inference(&query, &network, &assignment)
                    .unwrap();
            let interval = interval.expect("boundary includes eight randomized clusters per arm");
            assert_eq!(
                (interval.control_clusters, interval.treated_clusters),
                (cluster_count / 2, cluster_count / 2)
            );
            covered += usize::from(interval.bounds[0] <= truth && truth <= interval.bounds[1]);
        }
        let rate = covered as f64 / DRAWS as f64;
        eprintln!(
            "cluster-total {}+{}: {covered}/{DRAWS}, coverage={rate:.4}",
            cluster_count / 2,
            cluster_count / 2
        );
        assert!((0.93..=0.97).contains(&rate), "cluster-total 95% coverage {rate}");
    }
}
