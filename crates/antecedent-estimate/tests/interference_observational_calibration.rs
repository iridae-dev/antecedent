//! Repeated-sampling evidence for a known-propensity observational exposure interval.
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::cast_possible_truncation,
    reason = "calibration fixtures build small nonnegative cluster labels as u32"
)]

use antecedent_core::{ExposureLevel, ExposureMapping, ExposurePropensityProvenance, VariableId};
use antecedent_data::{NetworkData, NetworkEdge, TabularData};
use antecedent_estimate::{ObservationalExposureSpec, estimate_observational_exposure};

fn uniform(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state >> 11) as f64 / (1_u64 << 53) as f64
}

#[test]
fn known_exposure_probabilities_have_pointwise_cluster_coverage() {
    const CLUSTERS: usize = 60;
    const REPLICATES: usize = 400;
    let mut state = 0x5c0d_4e1a_093b_7f25_u64;
    let mut covered = 0;
    let mut published = 0;
    for _ in 0..REPLICATES {
        let mut outcomes = Vec::with_capacity(2 * CLUSTERS);
        let mut assignments = Vec::with_capacity(2 * CLUSTERS);
        let mut labels = Vec::with_capacity(2 * CLUSTERS);
        let mut edges = Vec::with_capacity(2 * CLUSTERS);
        for cluster in 0..CLUSTERS {
            let treated = uniform(&mut state) < 0.5;
            // Independent cluster baseline heterogeneity, shared by the two units.
            let baseline = 1.0 + (uniform(&mut state) - 0.5) * 2.0;
            for unit in 0..2 {
                outcomes.push(baseline + 2.0 * f64::from(treated));
                assignments.push(treated);
                labels.push(cluster as u32);
                edges.push(NetworkEdge {
                    from: (2 * cluster + 1 - unit) as u32,
                    to: (2 * cluster + unit) as u32,
                    weight: 1.0,
                });
            }
        }
        let data = TabularData::from_f64_columns([("y", outcomes.as_slice())]).unwrap();
        let network = NetworkData::try_new(data, edges).unwrap();
        let propensities = vec![0.5; 2 * CLUSTERS];
        let spec = ObservationalExposureSpec {
            assignment: &assignments,
            clusters: &labels,
            exposure: &ExposureMapping::NeighborCount,
            from: ExposureLevel { own: 0.0, neighbors: 0.0 },
            to: ExposureLevel { own: 1.0, neighbors: 1.0 },
            propensity_from: &propensities,
            propensity_to: &propensities,
            propensity_provenance: ExposurePropensityProvenance::Known,
        };
        let result = estimate_observational_exposure(&network, VariableId::from_raw(0), &spec).unwrap();
        if let Some(interval) = result.pointwise_interval {
            published += 1;
            covered += usize::from(interval.bounds[0] <= 2.0 && 2.0 <= interval.bounds[1]);
        }
    }
    eprintln!("observational known-propensity cluster interval: {covered}/{published} covered");
    assert!(published >= 390, "too many intervals withheld: {published}");
    assert!(covered >= 365, "coverage below calibrated floor: {covered}/{published}");
}
