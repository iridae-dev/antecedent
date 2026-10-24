// SPDX-License-Identifier: MIT OR Apache-2.0
//! Refuter state of a design-based exposure contrast.
//!
//! The ATE-shaped refuters do not apply to an exposure contrast: its identification is the
//! known assignment mechanism and the exposure mapping on a fixed network, so its `cheap` and
//! `full` suites check that mechanism against the realized assignment and the estimator
//! against the mechanism:
//!
//! * `interference.assignment_calibration` (cheap, full): the realized assignment against the
//!   declared design. Independent Bernoulli: the treated count against the sum of the supplied
//!   probabilities (a Poisson-binomial z test). Complete randomization: the treated count is
//!   the declared one. Cluster randomization: every cluster is wholly treated or wholly
//!   control and the treated-cluster count is the declared one. Designs with realized
//!   saturations or supplied exposure probabilities have no such check and report none.
//! * `interference.exposure_overlap` (cheap, full): the smallest probability of either
//!   requested exposure level; the Horvitz-Thompson weights are its reciprocals, so a
//!   probability under one in a hundred is flagged (descriptive, not a calibrated test).
//! * `interference.randomization_unbiasedness` (full; Bernoulli, complete and cluster
//!   randomization): with the outcomes held fixed the sharp null of no exposure effect makes
//!   every re-randomized contrast an unbiased draw around zero, whatever the outcomes are. The
//!   assignment is re-drawn from the declared design and the contrast recomputed; the mean of
//!   the re-randomized contrasts must be indistinguishable from zero. A mean away from zero
//!   means the exposure probabilities the estimator uses are not the ones the design implies.
//! * Two-stage saturation: the realized assignment is tested against the realized saturations
//!   (`interference.assignment_calibration`, a Poisson-binomial z test of the treated count),
//!   and `interference.saturation_allocation` checks that the saturations are the declared low
//!   and high values, constant within every cluster, with the declared number of high clusters.
//! * Supplied exposure probabilities (observed exposure): the realized exposure of every unit
//!   is computed from the assignment and the network, and the number of units at the baseline
//!   and at the active level is tested against the sum of the supplied probabilities
//!   (`interference.exposure_calibration_baseline` and `_active`), with the variance taken over
//!   clusters because exposures within a cluster are dependent; the smallest supplied
//!   probability is flagged as `interference.exposure_overlap`.
//!
//! The state executes under both inference modes and is unlicensed: its verdicts are checks,
//! not a calibration claim, and no coverage record names it.

use antecedent_core::{
    AssignmentDesign, ExecutionContext, ExposureLevel, InterferenceFunctional, InterferenceQuery,
    StreamDomain,
};
use antecedent_data::NetworkData;
use antecedent_estimate::{InterferenceEstimate, estimate_interference};
use antecedent_kernels::erfc;
use antecedent_validate::RefutationReport;

use super::{CausalError, RefuteSuite};

/// Two-sided level of the tests.
const SIGNIFICANCE: f64 = 0.05;
/// Smallest exposure probability that is not flagged.
const MIN_EXPOSURE_PROBABILITY: f64 = 0.01;
/// Re-randomizations when the study requested no replicates.
const DEFAULT_REDRAWS: u32 = 199;
/// Upper bound on the re-randomizations.
const MAX_REDRAWS: u32 = 999;

/// Borrowed inputs of one interference refutation.
pub(super) struct InterferenceRefutationInput<'a> {
    pub(super) query: &'a InterferenceQuery,
    pub(super) network: &'a NetworkData,
    pub(super) assignment: &'a [bool],
    /// The design-based estimate, absent under the model-based Bayesian route.
    pub(super) estimated: Option<&'a InterferenceEstimate>,
    pub(super) suite: RefuteSuite,
    /// The study's replicate count (zero: the default budget).
    pub(super) replicates: u32,
}

/// The reports of the requested suite; empty for [`RefuteSuite::None`].
///
/// # Errors
///
/// A cancelled run.
pub(super) fn refute_interference(
    input: &InterferenceRefutationInput<'_>,
    ctx: &ExecutionContext,
) -> Result<Vec<RefutationReport>, CausalError> {
    if input.suite == RefuteSuite::None {
        return Ok(Vec::new());
    }
    let mut reports = Vec::new();
    if let Some(report) = assignment_calibration(input.query, input.assignment) {
        reports.push(report);
    }
    if let Some(report) = saturation_allocation(input.query) {
        reports.push(report);
    }
    reports.extend(exposure_calibration(input));
    if let Some(minimum) = input
        .estimated
        .map(|estimated| estimated.minimum_exposure_probability)
        .or_else(|| supplied_minimum_probability(input.query))
    {
        reports.push(exposure_overlap(minimum));
    }
    if input.estimated.is_some()
        && input.suite == RefuteSuite::Full
        && matches!(
            input.query.assignment,
            AssignmentDesign::Bernoulli { .. }
                | AssignmentDesign::CompleteRandomization { .. }
                | AssignmentDesign::ClusterRandomization { .. }
        )
    {
        reports.push(randomization_unbiasedness(input, ctx)?);
    }
    Ok(reports)
}

fn z_p_value(total: f64, variance: f64) -> f64 {
    if variance > 0.0 && variance.is_finite() {
        erfc((total / variance.sqrt()).abs() / std::f64::consts::SQRT_2).clamp(0.0, 1.0)
    } else if total.abs() <= 1e-9 {
        1.0
    } else {
        0.0
    }
}

/// Probability of unit `unit` under a scalar-or-per-unit Bernoulli declaration.
fn bernoulli_probability(probabilities: &[f64], unit: usize) -> f64 {
    if probabilities.len() == 1 { probabilities[0] } else { probabilities[unit] }
}

fn assignment_calibration(
    query: &InterferenceQuery,
    assignment: &[bool],
) -> Option<RefutationReport> {
    let treated = assignment.iter().filter(|&&z| z).count();
    match &query.assignment {
        AssignmentDesign::Bernoulli { probabilities } => {
            if probabilities.len() != 1 && probabilities.len() != assignment.len() {
                return None;
            }
            let expected: f64 =
                (0..assignment.len()).map(|u| bernoulli_probability(probabilities, u)).sum();
            let variance: f64 = (0..assignment.len())
                .map(|u| {
                    let p = bernoulli_probability(probabilities, u);
                    p * (1.0 - p)
                })
                .sum();
            let p_value = z_p_value(treated as f64 - expected, variance);
            let passed = p_value >= SIGNIFICANCE;
            Some(RefutationReport::new(
                "interference.assignment_calibration",
                treated as f64,
                expected,
                p_value,
                true,
                passed,
                (!passed).then(|| {
                    std::sync::Arc::from(format!(
                        "{treated} treated units where the declared Bernoulli probabilities \
                         imply {expected:.2} (p = {p_value:.4}); the realized assignment does \
                         not follow the declared design"
                    ))
                }),
                1,
            ))
        }
        AssignmentDesign::CompleteRandomization { treated: declared } => {
            let passed = treated == *declared;
            Some(RefutationReport::new(
                "interference.assignment_calibration",
                treated as f64,
                *declared as f64,
                if passed { 1.0 } else { 0.0 },
                true,
                passed,
                (!passed).then(|| {
                    std::sync::Arc::from(format!(
                        "{treated} treated units where complete randomization assigns exactly \
                         {declared}"
                    ))
                }),
                1,
            ))
        }
        AssignmentDesign::ClusterRandomization { clusters, treated_clusters } => {
            if clusters.len() != assignment.len() {
                return None;
            }
            let mut by_cluster: std::collections::BTreeMap<u32, (usize, usize)> =
                std::collections::BTreeMap::new();
            for (cluster, &z) in clusters.iter().zip(assignment) {
                let entry = by_cluster.entry(*cluster).or_insert((0, 0));
                entry.0 += 1;
                entry.1 += usize::from(z);
            }
            let mixed = by_cluster.values().filter(|(n, t)| *t != 0 && t != n).count();
            let whole = by_cluster.values().filter(|(n, t)| t == n).count();
            let passed = mixed == 0 && whole == *treated_clusters;
            Some(RefutationReport::new(
                "interference.assignment_calibration",
                whole as f64,
                *treated_clusters as f64,
                if passed { 1.0 } else { 0.0 },
                true,
                passed,
                (!passed).then(|| {
                    std::sync::Arc::from(format!(
                        "{whole} wholly treated clusters and {mixed} partly treated clusters \
                         where cluster randomization treats exactly {treated_clusters} whole \
                         clusters"
                    ))
                }),
                1,
            ))
        }
        AssignmentDesign::TwoStageSaturation { realized_saturation, .. } => {
            if realized_saturation.len() != assignment.len() {
                return None;
            }
            let expected: f64 = realized_saturation.iter().sum();
            let variance: f64 = realized_saturation.iter().map(|p| p * (1.0 - p)).sum();
            let p_value = z_p_value(treated as f64 - expected, variance);
            let passed = p_value >= SIGNIFICANCE;
            Some(RefutationReport::new(
                "interference.assignment_calibration",
                treated as f64,
                expected,
                p_value,
                true,
                passed,
                (!passed).then(|| {
                    std::sync::Arc::from(format!(
                        "{treated} treated units where the realized saturations imply \
                         {expected:.2} (p = {p_value:.4}); the realized assignment does not \
                         follow the declared second stage"
                    ))
                }),
                1,
            ))
        }
        AssignmentDesign::ObservedExposure { .. } => None,
    }
}

/// Two-stage saturation: the realized saturation of every unit is the declared low or high
/// value, constant within its cluster, and exactly the declared number of clusters is high.
fn saturation_allocation(query: &InterferenceQuery) -> Option<RefutationReport> {
    let AssignmentDesign::TwoStageSaturation {
        clusters,
        low_probability,
        high_probability,
        high_clusters,
        realized_saturation,
    } = &query.assignment
    else {
        return None;
    };
    if clusters.len() != realized_saturation.len() {
        return None;
    }
    let near = |a: f64, b: f64| (a - b).abs() <= 1e-12;
    let mut by_cluster: std::collections::BTreeMap<u32, f64> = std::collections::BTreeMap::new();
    let mut constant = true;
    let mut declared = true;
    for (&cluster, &saturation) in clusters.iter().zip(realized_saturation.iter()) {
        declared &= near(saturation, *low_probability) || near(saturation, *high_probability);
        let first = *by_cluster.entry(cluster).or_insert(saturation);
        constant &= near(first, saturation);
    }
    let high =
        by_cluster.values().filter(|&&saturation| near(saturation, *high_probability)).count();
    let passed = constant && declared && high == *high_clusters;
    Some(RefutationReport::new(
        "interference.saturation_allocation",
        high as f64,
        *high_clusters as f64,
        if passed { 1.0 } else { 0.0 },
        true,
        passed,
        (!passed).then(|| {
            std::sync::Arc::from(format!(
                "{high} high-saturation clusters where {high_clusters} are declared; saturations \
                 constant within clusters: {constant}; every saturation is the declared low or \
                 high value: {declared}"
            ))
        }),
        1,
    ))
}

/// Smallest supplied exposure probability of an observed-exposure design.
fn supplied_minimum_probability(query: &InterferenceQuery) -> Option<f64> {
    let AssignmentDesign::ObservedExposure { propensity_from, propensity_to, .. } =
        &query.assignment
    else {
        return None;
    };
    Some(propensity_from.iter().chain(propensity_to.iter()).copied().fold(f64::INFINITY, f64::min))
}

/// Observed exposure: the units realized at the baseline and the active exposure level against
/// the sum of the supplied probabilities of that level, with the variance over clusters.
fn exposure_calibration(input: &InterferenceRefutationInput<'_>) -> Vec<RefutationReport> {
    let AssignmentDesign::ObservedExposure { clusters, propensity_from, propensity_to, .. } =
        &input.query.assignment
    else {
        return Vec::new();
    };
    let InterferenceFunctional::ExposureContrast { from, to, .. } = input.query.functional;
    let units = input.assignment.len();
    if clusters.len() != units || propensity_from.len() != units || propensity_to.len() != units {
        return Vec::new();
    }
    let incoming: Result<Vec<Vec<(usize, f64)>>, _> = (0..units)
        .map(|unit| {
            input.network.incoming(unit).map(|edges| {
                edges
                    .iter()
                    .filter_map(|edge| {
                        usize::try_from(edge.from.get()).ok().map(|source| (source, edge.weight))
                    })
                    .collect::<Vec<_>>()
            })
        })
        .collect();
    let Ok(incoming) = incoming else { return Vec::new() };
    let Ok(observed) =
        antecedent_stats::exposures(input.assignment, &incoming, &input.query.exposure)
    else {
        return Vec::new();
    };
    let at = |level: ExposureLevel, unit: usize| {
        (observed[unit].own - level.own).abs() <= 1e-9
            && (observed[unit].neighbors - level.neighbors).abs() <= 1e-9
    };
    [
        ("interference.exposure_calibration_baseline", from, propensity_from, "baseline"),
        ("interference.exposure_calibration_active", to, propensity_to, "active"),
    ]
    .into_iter()
    .map(|(id, level, probabilities, name)| {
        let mut by_cluster: std::collections::BTreeMap<u32, f64> =
            std::collections::BTreeMap::new();
        let (mut observed_units, mut expected) = (0.0, 0.0);
        for unit in 0..units {
            let exposed = f64::from(u8::from(at(level, unit)));
            observed_units += exposed;
            expected += probabilities[unit];
            *by_cluster.entry(clusters[unit]).or_insert(0.0) += exposed - probabilities[unit];
        }
        let variance: f64 = by_cluster.values().map(|total| total * total).sum();
        let p_value = z_p_value(observed_units - expected, variance);
        let passed = p_value >= SIGNIFICANCE;
        RefutationReport::new(
            id,
            observed_units,
            expected,
            p_value,
            true,
            passed,
            (!passed).then(|| {
                std::sync::Arc::from(format!(
                    "{observed_units} units realized the {name} exposure where the supplied \
                     probabilities imply {expected:.2} (p = {p_value:.4}); the supplied exposure \
                     probabilities do not describe the realized exposures"
                ))
            }),
            1,
        )
    })
    .collect()
}

/// One re-draw of the assignment from the declared design; `None` when the design is not one
/// the re-randomization test covers or its declaration is inconsistent with the units.
fn redraw_assignment(
    design: &AssignmentDesign,
    units: usize,
    rng: &mut antecedent_core::CausalRng,
) -> Option<Vec<bool>> {
    let mut below = |bound: usize| -> usize {
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a uniform in [0, 1) times the bound floors to an index below the bound"
        )]
        let index = ((rng.next_f64() * bound as f64) as usize).min(bound.saturating_sub(1));
        index
    };
    match design {
        AssignmentDesign::Bernoulli { .. } => None,
        AssignmentDesign::CompleteRandomization { treated } => {
            if *treated > units {
                return None;
            }
            let mut order: Vec<usize> = (0..units).collect();
            for position in 0..*treated {
                let other = position + below(units - position);
                order.swap(position, other);
            }
            let mut assignment = vec![false; units];
            for &unit in &order[..*treated] {
                assignment[unit] = true;
            }
            Some(assignment)
        }
        AssignmentDesign::ClusterRandomization { clusters, treated_clusters } => {
            if clusters.len() != units {
                return None;
            }
            let mut ids: Vec<u32> = clusters.to_vec();
            ids.sort_unstable();
            ids.dedup();
            if *treated_clusters > ids.len() {
                return None;
            }
            for position in 0..*treated_clusters {
                let other = position + below(ids.len() - position);
                ids.swap(position, other);
            }
            let chosen = &ids[..*treated_clusters];
            Some(clusters.iter().map(|cluster| chosen.contains(cluster)).collect())
        }
        AssignmentDesign::TwoStageSaturation { .. } | AssignmentDesign::ObservedExposure { .. } => {
            None
        }
    }
}

fn exposure_overlap(minimum: f64) -> RefutationReport {
    let passed = minimum.is_finite() && minimum >= MIN_EXPOSURE_PROBABILITY;
    RefutationReport::new(
        "interference.exposure_overlap",
        minimum,
        MIN_EXPOSURE_PROBABILITY,
        if minimum.is_finite() { minimum } else { 0.0 },
        true,
        passed,
        (!passed).then(|| {
            std::sync::Arc::from(format!(
                "the smallest probability of a requested exposure level is {minimum:.4}; the \
                 Horvitz-Thompson weights reach {:.0}",
                1.0 / minimum
            ))
        }),
        1,
    )
}

fn randomization_unbiasedness(
    input: &InterferenceRefutationInput<'_>,
    ctx: &ExecutionContext,
) -> Result<RefutationReport, CausalError> {
    let units = input.assignment.len();
    let design = &input.query.assignment;
    if let AssignmentDesign::Bernoulli { probabilities } = design {
        if probabilities.len() != 1 && probabilities.len() != units {
            return Ok(RefutationReport::new(
                "interference.randomization_unbiasedness",
                0.0,
                0.0,
                0.0,
                false,
                false,
                Some(std::sync::Arc::from(
                    "the declared Bernoulli probabilities are neither one value nor one per unit",
                )),
                0,
            ));
        }
    }
    let redraws = if input.replicates == 0 {
        DEFAULT_REDRAWS
    } else {
        input.replicates.clamp(2, MAX_REDRAWS)
    };
    let mut contrasts = Vec::with_capacity(redraws as usize);
    let mut failed = 0_u32;
    for draw in 0..redraws {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: crate::analysis::stage::STAGE_VALIDATE });
        }
        let mut rng = ctx
            .rng
            .stream_for(StreamDomain::Resample, 0x1F7E_5EED_u64.wrapping_add(u64::from(draw)));
        let assignment: Option<Vec<bool>> = match design {
            AssignmentDesign::Bernoulli { probabilities } => Some(
                (0..units)
                    .map(|unit| rng.next_f64() < bernoulli_probability(probabilities, unit))
                    .collect(),
            ),
            other => redraw_assignment(other, units, &mut rng),
        };
        let Some(assignment) = assignment else {
            failed += 1;
            continue;
        };
        let seed = rng.next_u64();
        match estimate_interference(input.query, input.network, &assignment, seed) {
            Ok(estimate) if estimate.contrast.horvitz_thompson.is_finite() => {
                contrasts.push(estimate.contrast.horvitz_thompson);
            }
            _ => failed += 1,
        }
    }
    let ok = u32::try_from(contrasts.len()).unwrap_or(u32::MAX);
    if contrasts.len() < 2 {
        return Ok(RefutationReport::new(
            "interference.randomization_unbiasedness",
            0.0,
            0.0,
            0.0,
            false,
            false,
            Some(std::sync::Arc::from(format!(
                "only {ok} of {redraws} re-randomizations produced a contrast ({failed} failed)"
            ))),
            ok,
        ));
    }
    let n = contrasts.len() as f64;
    let mean = contrasts.iter().sum::<f64>() / n;
    let variance = contrasts.iter().map(|c| (c - mean) * (c - mean)).sum::<f64>() / (n - 1.0);
    let p_value = z_p_value(mean, variance / n);
    let passed = p_value >= SIGNIFICANCE;
    Ok(RefutationReport::new(
        "interference.randomization_unbiasedness",
        mean,
        0.0,
        p_value,
        true,
        passed,
        (!passed).then(|| {
            std::sync::Arc::from(format!(
                "the mean of {ok} re-randomized contrasts is {mean:.4} (p = {p_value:.4}) where \
                 the sharp null makes it zero; the exposure probabilities the estimator uses do \
                 not follow the declared design ({failed} re-randomizations failed)"
            ))
        }),
        ok,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{ExposureLevel, ExposureMapping, InterferenceFunctional, VariableId};
    use std::sync::Arc;

    fn query(assignment: AssignmentDesign) -> InterferenceQuery {
        InterferenceQuery::new(
            assignment,
            ExposureMapping::NeighborCount,
            InterferenceFunctional::ExposureContrast {
                outcome: VariableId::from_raw(0),
                from: ExposureLevel { own: 0.0, neighbors: 1.0 },
                to: ExposureLevel { own: 1.0, neighbors: 0.0 },
            },
        )
    }

    #[test]
    fn a_bernoulli_treated_count_is_tested_against_the_sum_of_the_probabilities() {
        let design = AssignmentDesign::Bernoulli { probabilities: Arc::from([0.5]) };
        // Ten units: five treated is the expectation.
        let balanced = assignment_calibration(
            &query(design.clone()),
            &[true, true, true, true, true, false, false, false, false, false],
        )
        .unwrap();
        assert!(balanced.passed && (balanced.comparison - 1.0).abs() < 1e-12);
        // Nine treated: z = 4 / sqrt(2.5) = 2.5298, two-sided normal tail 0.01141.
        let skewed = assignment_calibration(
            &query(design),
            &[true, true, true, true, true, true, true, true, true, false],
        )
        .unwrap();
        assert!(!skewed.passed);
        assert!((skewed.comparison - 0.011_4).abs() < 1e-3, "{}", skewed.comparison);
    }

    #[test]
    fn complete_and_cluster_randomization_have_exact_checks() {
        let complete = query(AssignmentDesign::CompleteRandomization { treated: 2 });
        assert!(assignment_calibration(&complete, &[true, true, false, false]).unwrap().passed);
        assert!(!assignment_calibration(&complete, &[true, true, true, false]).unwrap().passed);
        let clustered = query(AssignmentDesign::ClusterRandomization {
            clusters: Arc::from([0, 0, 1, 1, 2, 2]),
            treated_clusters: 1,
        });
        // One wholly treated cluster of three: as declared.
        let ok = [true, true, false, false, false, false];
        assert!(assignment_calibration(&clustered, &ok).unwrap().passed);
        // A split cluster is not a cluster assignment.
        let split = [true, false, false, false, false, false];
        assert!(!assignment_calibration(&clustered, &split).unwrap().passed);
    }

    #[test]
    fn a_saturation_assignment_is_tested_against_its_realized_saturations_and_allocation() {
        // Two clusters of four: one at 0.25, one at 0.75.
        let saturations: Vec<f64> = [0.25; 4].into_iter().chain([0.75; 4]).collect();
        let design = |high_clusters, saturation: Vec<f64>| {
            query(AssignmentDesign::TwoStageSaturation {
                clusters: Arc::from([0, 0, 0, 0, 1, 1, 1, 1]),
                low_probability: 0.25,
                high_probability: 0.75,
                high_clusters,
                realized_saturation: Arc::from(saturation),
            })
        };
        let declared = design(1, saturations.clone());
        // Four treated of eight where the saturations sum to four.
        let balanced = [true, false, false, false, true, true, true, false];
        assert!(assignment_calibration(&declared, &balanced).unwrap().passed);
        // Everything treated: seven more than the sum, z = 4 / sqrt(1.5) beyond the level.
        assert!(!assignment_calibration(&declared, &[true; 8]).unwrap().passed);
        assert!(saturation_allocation(&declared).unwrap().passed);
        // Two clusters at the high value where one is declared.
        assert!(!saturation_allocation(&design(1, vec![0.75; 8])).unwrap().passed);
        // A saturation that is neither declared value, and one that varies in its cluster.
        let mut off = saturations.clone();
        off[0] = 0.5;
        assert!(!saturation_allocation(&design(1, off)).unwrap().passed);
        // Other designs have no allocation check.
        assert!(
            saturation_allocation(&query(AssignmentDesign::CompleteRandomization { treated: 1 }))
                .is_none()
        );
    }

    #[test]
    fn redrawn_complete_and_cluster_assignments_follow_their_design() {
        let ctx = ExecutionContext::for_tests(11);
        let complete = AssignmentDesign::CompleteRandomization { treated: 3 };
        let mut seen = std::collections::BTreeSet::new();
        for stream in 0..40 {
            let mut rng = ctx.rng.stream(stream);
            let draw = redraw_assignment(&complete, 8, &mut rng).unwrap();
            assert_eq!(draw.iter().filter(|&&z| z).count(), 3);
            seen.insert(draw);
        }
        // Forty draws of three out of eight are not one fixed subset.
        assert!(seen.len() > 10);
        let clustered = AssignmentDesign::ClusterRandomization {
            clusters: Arc::from([0, 0, 1, 1, 1, 2, 3, 3]),
            treated_clusters: 2,
        };
        for stream in 0..40 {
            let mut rng = ctx.rng.stream(100 + stream);
            let draw = redraw_assignment(&clustered, 8, &mut rng).unwrap();
            // Whole clusters only, exactly two of the four.
            let cluster_ids = [0_usize, 0, 1, 1, 1, 2, 3, 3];
            for cluster in 0..4 {
                let members: Vec<bool> = cluster_ids
                    .iter()
                    .zip(&draw)
                    .filter(|(id, _)| **id == cluster)
                    .map(|(_, z)| *z)
                    .collect();
                assert!(members.iter().all(|z| *z) || members.iter().all(|z| !*z));
            }
            let treated_clusters = (0..4)
                .filter(|&cluster| {
                    cluster_ids.iter().zip(&draw).any(|(id, z)| *id == cluster && *z)
                })
                .count();
            assert_eq!(treated_clusters, 2);
        }
        // A declaration that cannot be drawn is not a draw.
        let mut rng = ctx.rng.stream(7);
        assert!(
            redraw_assignment(&AssignmentDesign::CompleteRandomization { treated: 9 }, 8, &mut rng)
                .is_none()
        );
    }

    /// Sixty two-unit clusters, every other one treated; the pair of a treated cluster realizes
    /// the active level and the pair of an untreated cluster the baseline level.
    fn paired_clusters(
        propensity_from: f64,
        propensity_to: f64,
    ) -> (antecedent_data::NetworkData, Vec<bool>, InterferenceQuery) {
        let units = 120_usize;
        let assignment: Vec<bool> = (0..units).map(|unit| (unit / 2) % 2 == 1).collect();
        let clusters: Vec<u32> = (0..units).map(|unit| (unit / 2) as u32).collect();
        let edges: Vec<antecedent_data::NetworkEdge> = (0..units)
            .map(|unit| antecedent_data::NetworkEdge::new((unit ^ 1) as u64, unit as u64, 1.0))
            .collect();
        let outcomes = vec![1.0; units];
        let table =
            antecedent_data::TabularData::from_f64_columns([("y", outcomes.as_slice())]).unwrap();
        let query = InterferenceQuery::new(
            AssignmentDesign::ObservedExposure {
                clusters: Arc::from(clusters),
                propensity_from: Arc::from(vec![propensity_from; units]),
                propensity_to: Arc::from(vec![propensity_to; units]),
                provenance: antecedent_core::ExposurePropensityProvenance::Known,
                assume_network_exchangeability: true,
            },
            ExposureMapping::NeighborCount,
            InterferenceFunctional::ExposureContrast {
                outcome: VariableId::from_raw(0),
                from: ExposureLevel { own: 0.0, neighbors: 0.0 },
                to: ExposureLevel { own: 1.0, neighbors: 1.0 },
            },
        );
        (antecedent_data::NetworkData::try_new(table, edges).unwrap(), assignment, query)
    }

    #[test]
    fn supplied_exposure_probabilities_are_tested_against_the_realized_exposures() {
        let run = |from: f64, to: f64| {
            let (network, assignment, query) = paired_clusters(from, to);
            exposure_calibration(&InterferenceRefutationInput {
                query: &query,
                network: &network,
                assignment: &assignment,
                estimated: None,
                suite: RefuteSuite::Cheap,
                replicates: 0,
            })
        };
        // Sixty baseline-exposed and sixty active-exposed units against one half each.
        let honest = run(0.5, 0.5);
        assert_eq!(honest.len(), 2);
        for each in &honest {
            assert!((each.original_ate - 60.0).abs() < 1e-12);
            assert!((each.refuted_ate - 60.0).abs() < 1e-12);
            assert!(each.passed && (each.comparison - 1.0).abs() < 1e-12);
        }
        assert_eq!(honest[0].refuter.as_ref(), "interference.exposure_calibration_baseline");
        assert_eq!(honest[1].refuter.as_ref(), "interference.exposure_calibration_active");
        // Probabilities of 0.9 and 0.05 imply 108 baseline-exposed units; the cluster totals
        // are +0.2 in thirty clusters and -1.8 in thirty, a variance of 98.4, so
        // z = -48 / sqrt(98.4) = -4.84 and the normal tail is 1.3e-6.
        let dishonest = run(0.9, 0.05);
        assert!(!dishonest[0].passed);
        assert!((dishonest[0].refuted_ate - 108.0).abs() < 1e-9);
        assert!(dishonest[0].comparison < 1e-4, "{}", dishonest[0].comparison);
        // The active level implies six units against sixty realized.
        assert!(!dishonest[1].passed);
        assert!((dishonest[1].refuted_ate - 6.0).abs() < 1e-9);
        assert_eq!(supplied_minimum_probability(&paired_clusters(0.9, 0.05).2), Some(0.05));
        // Other designs have neither.
        let (network, assignment, _) = paired_clusters(0.5, 0.5);
        let bernoulli = query(AssignmentDesign::Bernoulli { probabilities: Arc::from([0.5]) });
        assert!(
            exposure_calibration(&InterferenceRefutationInput {
                query: &bernoulli,
                network: &network,
                assignment: &assignment,
                estimated: None,
                suite: RefuteSuite::Cheap,
                replicates: 0,
            })
            .is_empty()
        );
        assert!(supplied_minimum_probability(&bernoulli).is_none());
    }

    #[test]
    fn a_thin_exposure_probability_is_flagged() {
        assert!(exposure_overlap(0.2).passed);
        let thin = exposure_overlap(0.004);
        assert!(!thin.passed);
        assert!((thin.comparison - 0.004).abs() < 1e-12);
        assert!(!exposure_overlap(f64::NAN).passed);
    }
}
