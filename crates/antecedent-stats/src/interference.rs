//! Randomization-based kernels for known network assignment designs.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss)]

use antecedent_core::{
    AssignmentDesign, CausalRng, EXPOSURE_LEVEL_TOLERANCE, ExposureLevel, ExposureMapping,
};

use crate::StatsError;

/// How exposure probabilities were computed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExposureProbabilityMethod {
    /// All assignments in the design support were enumerated.
    Exact,
    /// Assignments were sampled from the design using a deterministic seed.
    MonteCarlo {
        /// Number of sampled assignments.
        draws: u32,
        /// Seed used for the deterministic assignment stream.
        seed: u64,
    },
}

/// Exposure probabilities for one requested level, in unit order.
#[derive(Clone, Debug, PartialEq)]
pub struct ExposureProbabilities {
    /// Probability for each unit.
    pub probabilities: Vec<f64>,
    /// Computation used.
    pub method: ExposureProbabilityMethod,
}

/// Randomization estimate of an exposure-specific mean.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RandomizationMean {
    /// Horvitz--Thompson mean.
    pub horvitz_thompson: f64,
    /// Self-normalized Hájek mean.
    pub hajek: f64,
    /// Conservative diagonal randomization variance for the HT mean.
    pub conservative_variance: f64,
    /// Units observed at this exposure.
    pub exposed_units: usize,
}

/// Randomization estimate of an exposure contrast (`to - from`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RandomizationContrast {
    /// HT contrast.
    pub horvitz_thompson: f64,
    /// Hájek contrast.
    pub hajek: f64,
    /// Conservative variance using a covariance-free Young bound.
    pub conservative_variance: f64,
}

/// Calculate the exposure of every unit from an assignment vector.
///
/// `incoming[i]` contains `(source_unit, weight)` pairs affecting unit `i`.
pub fn exposures(
    assignment: &[bool],
    incoming: &[Vec<(usize, f64)>],
    mapping: &ExposureMapping,
) -> Result<Vec<ExposureLevel>, StatsError> {
    validate_exposure_inputs(assignment.len(), incoming, mapping)?;
    let mut out = Vec::with_capacity(assignment.len());
    exposures_into(assignment, incoming, mapping, &mut out);
    Ok(out)
}

/// One-time validation for [`exposures`]; hoisted so per-draw loops skip it.
fn validate_exposure_inputs(
    n: usize,
    incoming: &[Vec<(usize, f64)>],
    mapping: &ExposureMapping,
) -> Result<(), StatsError> {
    if n != incoming.len() {
        return Err(StatsError::Backend("assignment/network length mismatch".into()));
    }
    if incoming
        .iter()
        .flatten()
        .any(|&(source, weight)| source >= n || !weight.is_finite() || weight < 0.0)
    {
        return Err(StatsError::Backend("invalid incoming network edge".into()));
    }
    if matches!(mapping, ExposureMapping::Custom(_)) {
        return Err(StatsError::Backend(
            "custom exposure mappings require a caller registry".into(),
        ));
    }
    Ok(())
}

/// Exposure computation into a reused buffer; inputs must be pre-validated.
fn exposures_into(
    assignment: &[bool],
    incoming: &[Vec<(usize, f64)>],
    mapping: &ExposureMapping,
    out: &mut Vec<ExposureLevel>,
) {
    out.clear();
    for (unit, edges) in incoming.iter().enumerate() {
        let own = f64::from(assignment[unit]);
        let neighbors = match mapping {
            ExposureMapping::OwnTreatment => 0.0,
            ExposureMapping::NeighborCount => edges
                .iter()
                .map(|&(source, _)| assignment.get(source).copied().map_or(0.0, f64::from))
                .sum(),
            ExposureMapping::NeighborFraction => {
                if edges.is_empty() {
                    0.0
                } else {
                    edges
                        .iter()
                        .map(|&(source, _)| assignment.get(source).copied().map_or(0.0, f64::from))
                        .sum::<f64>()
                        / edges.len() as f64
                }
            }
            ExposureMapping::WeightedNeighborExposure => {
                let weight: f64 = edges.iter().map(|edge| edge.1).sum();
                if weight == 0.0 {
                    0.0
                } else {
                    edges
                        .iter()
                        .map(|&(source, w)| {
                            w * assignment.get(source).copied().map_or(0.0, f64::from)
                        })
                        .sum::<f64>()
                        / weight
                }
            }
            ExposureMapping::Custom(_) => unreachable!(),
        };
        out.push(ExposureLevel { own, neighbors });
    }
}

/// Compute exposure probabilities exactly when the support is small enough, otherwise by Monte
/// Carlo. Exact enumeration is used for at most 20 Bernoulli units or 20 randomization units /
/// clusters.
pub fn exposure_probabilities(
    design: &AssignmentDesign,
    incoming: &[Vec<(usize, f64)>],
    mapping: &ExposureMapping,
    level: ExposureLevel,
    monte_carlo_draws: u32,
    seed: u64,
) -> Result<ExposureProbabilities, StatsError> {
    let n = incoming.len();
    validate_design(design, n)?;
    let support_dim = match design {
        AssignmentDesign::ClusterRandomization { clusters, .. } => {
            let mut ids = clusters.to_vec();
            ids.sort_unstable();
            ids.dedup();
            ids.len()
        }
        _ => n,
    };
    if support_dim <= 20 {
        let mut sums = vec![0.0; n];
        let mut mass = 0.0;
        // Capture (rather than swallow) the first real error from `exposures`: an invalid
        // exposure mapping or network is a caller bug that should surface its own message, not
        // get masked behind a misleading "empty support" error below.
        let mut first_error: Option<StatsError> = None;
        enumerate_assignments(design, n, |assignment, p| {
            match exposures(assignment, incoming, mapping) {
                Ok(levels) => {
                    for (i, actual) in levels.iter().enumerate() {
                        if same_exposure(*actual, level) {
                            sums[i] += p;
                        }
                    }
                    mass += p;
                }
                Err(err) => {
                    if first_error.is_none() {
                        first_error = Some(err);
                    }
                }
            }
        })?;
        if let Some(err) = first_error {
            return Err(err);
        }
        if mass <= 0.0 {
            return Err(StatsError::Backend("assignment design has empty support".into()));
        }
        for p in &mut sums {
            *p /= mass;
        }
        return Ok(ExposureProbabilities {
            probabilities: sums,
            method: ExposureProbabilityMethod::Exact,
        });
    }
    if monte_carlo_draws == 0 {
        return Err(StatsError::Backend("Monte Carlo exposure draws must be positive".into()));
    }
    validate_exposure_inputs(n, incoming, mapping)?;
    let mut rng = CausalRng::from_seed(seed);
    let mut counts = vec![0_u32; n];
    let mut sampler = AssignmentSampler::new(design, n);
    let mut assignment = vec![false; n];
    let mut levels = Vec::with_capacity(n);
    for _ in 0..monte_carlo_draws {
        sampler.sample_into(design, &mut assignment, &mut rng);
        exposures_into(&assignment, incoming, mapping, &mut levels);
        for (i, actual) in levels.iter().enumerate() {
            if same_exposure(*actual, level) {
                counts[i] = counts[i].saturating_add(1);
            }
        }
    }
    Ok(ExposureProbabilities {
        probabilities: counts
            .into_iter()
            .map(|count| f64::from(count) / f64::from(monte_carlo_draws))
            .collect(),
        method: ExposureProbabilityMethod::MonteCarlo { draws: monte_carlo_draws, seed },
    })
}

/// Estimate an exposure-specific mean with HT and Hájek weights.
///
/// Horvitz–Thompson averages over the full unit population (`/n`). Every unit must have
/// strictly positive exposure probability for the requested level: a zero probability means
/// that unit can never realize the exposure, so its potential outcome is undefined and
/// folding it into `/n` would silently attenuate the mean toward zero.
pub fn randomization_mean(
    outcomes: &[f64],
    observed: &[ExposureLevel],
    probabilities: &[f64],
    level: ExposureLevel,
) -> Result<RandomizationMean, StatsError> {
    let n = outcomes.len();
    if n == 0 || observed.len() != n || probabilities.len() != n {
        return Err(StatsError::Backend("outcome/exposure/probability length mismatch".into()));
    }
    let mut weighted_sum = 0.0;
    let mut weight_sum = 0.0;
    let mut diagonal_upper_terms = 0.0;
    let mut exposed_units = 0;
    for i in 0..n {
        if !outcomes[i].is_finite() || !probabilities[i].is_finite() {
            return Err(StatsError::Backend("non-finite outcome or exposure probability".into()));
        }
        // Population HT for an exposure-specific mean is only defined when every unit can
        // realize the exposure. Checking only observed exposures would let πᵢ=0 units dilute
        // the `/n` average with no contribution and no error.
        if probabilities[i] <= 0.0 {
            return Err(StatsError::Backend(
                "exposure positivity violation: every unit needs strictly positive probability for the requested exposure".into(),
            ));
        }
        if same_exposure(observed[i], level) {
            let weighted = outcomes[i] / probabilities[i];
            weighted_sum += weighted;
            weight_sum += 1.0 / probabilities[i];
            diagonal_upper_terms +=
                (1.0 - probabilities[i]) * outcomes[i].powi(2) / probabilities[i].powi(2);
            exposed_units += 1;
        }
    }
    if exposed_units == 0 || weight_sum == 0.0 {
        return Err(StatsError::Backend("requested exposure is absent from observed data".into()));
    }
    let ht = weighted_sum / n as f64;
    // Young's inequality bounds every unknown covariance by the corresponding diagonal
    // variances. Summing those pair bounds yields `sum_i Var(X_i) / n` for the mean; the term
    // below is its unbiased HT estimate. This is intentionally conservative and can be loose.
    let variance = diagonal_upper_terms / n as f64;
    Ok(RandomizationMean {
        horvitz_thompson: ht,
        hajek: weighted_sum / weight_sum,
        conservative_variance: variance,
        exposed_units,
    })
}

/// Estimate a `to - from` exposure contrast.
#[must_use]
pub fn randomization_contrast(
    from: RandomizationMean,
    to: RandomizationMean,
) -> RandomizationContrast {
    RandomizationContrast {
        horvitz_thompson: to.horvitz_thompson - from.horvitz_thompson,
        hajek: to.hajek - from.hajek,
        conservative_variance: 2.0 * (from.conservative_variance + to.conservative_variance),
    }
}

/// True when two [`ExposureLevel`] values name the same level.
///
/// This shares [`EXPOSURE_LEVEL_TOLERANCE`] with `InterferenceQuery::validate`, which rejects a
/// `from`/`to` pair this close together before it ever reaches this matcher. If the two
/// tolerances ever diverged, a pair validation calls "distinct" could still collapse onto the
/// same unit set here and produce a zero contrast with no warning.
fn same_exposure(a: ExposureLevel, b: ExposureLevel) -> bool {
    (a.own - b.own).abs() <= EXPOSURE_LEVEL_TOLERANCE
        && (a.neighbors - b.neighbors).abs() <= EXPOSURE_LEVEL_TOLERANCE
}

fn validate_design(design: &AssignmentDesign, n: usize) -> Result<(), StatsError> {
    match design {
        AssignmentDesign::Bernoulli { probabilities }
            if (probabilities.len() != 1 && probabilities.len() != n)
                || probabilities.iter().any(|p| !p.is_finite() || *p <= 0.0 || *p >= 1.0) =>
        {
            Err(StatsError::Backend(
                "Bernoulli design needs one probability or one per unit".into(),
            ))
        }
        AssignmentDesign::CompleteRandomization { treated } if *treated == 0 || *treated > n => {
            Err(StatsError::Backend("treated count must lie in 1..=n".into()))
        }
        AssignmentDesign::ClusterRandomization { clusters, treated_clusters } => {
            let mut ids = clusters.to_vec();
            ids.sort_unstable();
            ids.dedup();
            if clusters.len() != n
                || clusters.is_empty()
                || *treated_clusters == 0
                || *treated_clusters > ids.len()
            {
                Err(StatsError::Backend("invalid cluster assignment design".into()))
            } else {
                Ok(())
            }
        }
        _ => Ok(()),
    }
}

#[allow(clippy::unnecessary_wraps)]
fn enumerate_assignments(
    design: &AssignmentDesign,
    n: usize,
    mut visit: impl FnMut(&[bool], f64),
) -> Result<(), StatsError> {
    match design {
        AssignmentDesign::Bernoulli { probabilities } => {
            for mask in 0..(1_u64 << n) {
                let mut assignment = vec![false; n];
                let mut p = 1.0;
                for i in 0..n {
                    assignment[i] = mask & (1_u64 << i) != 0;
                    let pi = probabilities[if probabilities.len() == 1 { 0 } else { i }];
                    p *= if assignment[i] { pi } else { 1.0 - pi };
                }
                visit(&assignment, p);
            }
        }
        AssignmentDesign::CompleteRandomization { treated } => {
            for mask in 0..(1_u64 << n) {
                if mask.count_ones() as usize == *treated {
                    let assignment = (0..n).map(|i| mask & (1_u64 << i) != 0).collect::<Vec<_>>();
                    visit(&assignment, 1.0);
                }
            }
        }
        AssignmentDesign::ClusterRandomization { clusters, treated_clusters } => {
            let mut ids = clusters.to_vec();
            ids.sort_unstable();
            ids.dedup();
            for mask in 0..(1_u64 << ids.len()) {
                if mask.count_ones() as usize == *treated_clusters {
                    let assignment = clusters
                        .iter()
                        .map(|id| {
                            let j = ids.binary_search(id).expect("cluster id present");
                            mask & (1_u64 << j) != 0
                        })
                        .collect::<Vec<_>>();
                    visit(&assignment, 1.0);
                }
            }
        }
    }
    Ok(())
}

/// Reusable per-design scratch for repeated assignment draws.
///
/// Hoists the work that is invariant across Monte Carlo draws — the sorted
/// cluster-id table and each unit's cluster position — and reuses the key and
/// membership buffers, so a draw is O(n) (plus O(k) selection) with no
/// per-draw heap allocation. Draws consume the RNG stream in exactly the same
/// order as the historical one-shot sampler, so results are bit-identical.
struct AssignmentSampler {
    /// Sorted, deduplicated cluster ids (cluster designs only).
    cluster_ids: Vec<u32>,
    /// Each unit's position in `cluster_ids` (cluster designs only).
    unit_cluster_pos: Vec<usize>,
    /// Selection keys, reused across draws.
    keys: Vec<(u64, usize)>,
    /// Chosen-cluster membership by `cluster_ids` position, reused across draws.
    chosen: Vec<bool>,
}

impl AssignmentSampler {
    fn new(design: &AssignmentDesign, n: usize) -> Self {
        let (cluster_ids, unit_cluster_pos) = match design {
            AssignmentDesign::ClusterRandomization { clusters, .. } => {
                let mut ids = clusters.to_vec();
                ids.sort_unstable();
                ids.dedup();
                let pos = clusters
                    .iter()
                    .map(|id| ids.binary_search(id).expect("cluster id present"))
                    .collect();
                (ids, pos)
            }
            _ => (Vec::new(), Vec::new()),
        };
        let key_capacity = match design {
            AssignmentDesign::CompleteRandomization { .. } => n,
            AssignmentDesign::ClusterRandomization { .. } => cluster_ids.len(),
            AssignmentDesign::Bernoulli { .. } => 0,
        };
        let chosen = vec![false; cluster_ids.len()];
        Self { cluster_ids, unit_cluster_pos, keys: Vec::with_capacity(key_capacity), chosen }
    }

    fn sample_into(&mut self, design: &AssignmentDesign, out: &mut [bool], rng: &mut CausalRng) {
        let n = out.len();
        match design {
            AssignmentDesign::Bernoulli { probabilities } => {
                for (i, slot) in out.iter_mut().enumerate() {
                    let p = probabilities[if probabilities.len() == 1 { 0 } else { i }];
                    *slot = rng.next_f64() < p;
                }
            }
            AssignmentDesign::CompleteRandomization { treated } => {
                self.keys.clear();
                self.keys.extend((0..n).map(|i| (rng.next_u64(), i)));
                if *treated > 0 && *treated < n {
                    // Partial selection: the treated set is the `treated` smallest
                    // keys, identical to a full sort's prefix.
                    self.keys.select_nth_unstable(*treated - 1);
                }
                out.fill(false);
                for &(_, i) in self.keys.iter().take(*treated) {
                    out[i] = true;
                }
            }
            AssignmentDesign::ClusterRandomization { treated_clusters, .. } => {
                let k = self.cluster_ids.len();
                self.keys.clear();
                self.keys.extend((0..k).map(|pos| (rng.next_u64(), pos)));
                if *treated_clusters > 0 && *treated_clusters < k {
                    self.keys.select_nth_unstable(*treated_clusters - 1);
                }
                self.chosen.fill(false);
                for &(_, pos) in self.keys.iter().take(*treated_clusters) {
                    self.chosen[pos] = true;
                }
                for (slot, &pos) in out.iter_mut().zip(&self.unit_cluster_pos) {
                    *slot = self.chosen[pos];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[test]
    fn empty_network_collapses_to_own_treatment() {
        let incoming = vec![vec![], vec![]];
        let z = [false, true];
        let levels = exposures(&z, &incoming, &ExposureMapping::NeighborFraction).unwrap();
        assert_eq!(
            levels,
            vec![
                ExposureLevel { own: 0.0, neighbors: 0.0 },
                ExposureLevel { own: 1.0, neighbors: 0.0 }
            ]
        );
    }

    #[test]
    fn exact_bernoulli_probabilities_include_neighbors() {
        let incoming = vec![vec![(1, 1.0)], vec![(0, 1.0)]];
        let design = AssignmentDesign::Bernoulli { probabilities: Arc::from([0.5]) };
        let result = exposure_probabilities(
            &design,
            &incoming,
            &ExposureMapping::NeighborCount,
            ExposureLevel { own: 1.0, neighbors: 1.0 },
            10,
            7,
        )
        .unwrap();
        assert_eq!(result.method, ExposureProbabilityMethod::Exact);
        assert_eq!(result.probabilities, vec![0.25, 0.25]);
    }

    #[test]
    fn exact_path_propagates_invalid_exposure_mapping_error() {
        // A negative edge weight makes every call to `exposures` fail. Before the fix this
        // error was swallowed by `if let Ok(...)`, and the caller instead saw a misleading
        // "assignment design has empty support" error that hid the real cause.
        let incoming = vec![vec![(0_usize, -1.0)], vec![]];
        let design = AssignmentDesign::Bernoulli { probabilities: Arc::from([0.5]) };
        let err = exposure_probabilities(
            &design,
            &incoming,
            &ExposureMapping::NeighborCount,
            ExposureLevel { own: 1.0, neighbors: 0.0 },
            10,
            7,
        )
        .unwrap_err();
        assert!(err.to_string().contains("invalid incoming network edge"));
    }

    #[test]
    fn seeded_monte_carlo_is_deterministic() {
        let incoming = vec![vec![]; 21];
        let design = AssignmentDesign::Bernoulli { probabilities: Arc::from([0.4]) };
        let a = exposure_probabilities(
            &design,
            &incoming,
            &ExposureMapping::OwnTreatment,
            ExposureLevel { own: 1.0, neighbors: 0.0 },
            500,
            91,
        )
        .unwrap();
        let b = exposure_probabilities(
            &design,
            &incoming,
            &ExposureMapping::OwnTreatment,
            ExposureLevel { own: 1.0, neighbors: 0.0 },
            500,
            91,
        )
        .unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn monte_carlo_sampler_reuse_is_bit_identical_to_one_shot_reference() {
        // The buffer-reusing `AssignmentSampler` must replay the historical
        // one-shot sampler exactly: same RNG consumption order, same chosen
        // sets, same assignment vectors — so published MC probabilities do not
        // move. The reference below is the pre-0.5.2 per-draw implementation.
        fn reference_draw(design: &AssignmentDesign, n: usize, rng: &mut CausalRng) -> Vec<bool> {
            match design {
                AssignmentDesign::Bernoulli { probabilities } => (0..n)
                    .map(|i| {
                        rng.next_f64() < probabilities[if probabilities.len() == 1 { 0 } else { i }]
                    })
                    .collect(),
                AssignmentDesign::CompleteRandomization { treated } => {
                    let mut keys = (0..n).map(|i| (rng.next_u64(), i)).collect::<Vec<_>>();
                    keys.sort_unstable();
                    let mut assignment = vec![false; n];
                    for &(_, i) in keys.iter().take(*treated) {
                        assignment[i] = true;
                    }
                    assignment
                }
                AssignmentDesign::ClusterRandomization { clusters, treated_clusters } => {
                    let mut ids = clusters.to_vec();
                    ids.sort_unstable();
                    ids.dedup();
                    let mut keys = ids.iter().map(|&id| (rng.next_u64(), id)).collect::<Vec<_>>();
                    keys.sort_unstable();
                    let chosen =
                        keys.iter().take(*treated_clusters).map(|x| x.1).collect::<Vec<_>>();
                    clusters.iter().map(|id| chosen.contains(id)).collect()
                }
            }
        }

        let n = 60usize;
        let clusters: Arc<[u32]> = (0..n).map(|i| u32::try_from(i % 30).unwrap()).collect();
        let designs = [
            AssignmentDesign::Bernoulli { probabilities: Arc::from([0.3]) },
            AssignmentDesign::CompleteRandomization { treated: 25 },
            AssignmentDesign::ClusterRandomization { clusters, treated_clusters: 11 },
        ];
        for design in &designs {
            let mut rng_new = CausalRng::from_seed(4242);
            let mut rng_ref = CausalRng::from_seed(4242);
            let mut sampler = AssignmentSampler::new(design, n);
            let mut assignment = vec![false; n];
            for draw in 0..50 {
                sampler.sample_into(design, &mut assignment, &mut rng_new);
                let expected = reference_draw(design, n, &mut rng_ref);
                assert_eq!(assignment, expected, "design {design:?} draw {draw}");
            }
        }
    }

    #[test]
    fn ht_and_hajek_means_are_reported() {
        let observed = [
            ExposureLevel { own: 0.0, neighbors: 0.0 },
            ExposureLevel { own: 1.0, neighbors: 0.0 },
        ];
        let mean = randomization_mean(&[2.0, 4.0], &observed, &[0.5, 0.5], observed[1]).unwrap();
        assert!((mean.horvitz_thompson - 4.0).abs() < 1e-12);
        assert!((mean.hajek - 4.0).abs() < 1e-12);
    }

    #[test]
    fn ht_refuses_zero_probability_units_that_would_dilute_the_population_mean() {
        // Unit 0 can never realize the requested exposure (π=0). Including it in `/n`
        // would report HT = 2 instead of the undefined-population refusal.
        let observed = [
            ExposureLevel { own: 0.0, neighbors: 0.0 },
            ExposureLevel { own: 1.0, neighbors: 0.0 },
        ];
        let err = randomization_mean(&[2.0, 4.0], &observed, &[0.0, 0.5], observed[1]).unwrap_err();
        assert!(err.to_string().contains("positivity"));
    }
}
