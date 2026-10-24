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
//! * `interference.randomization_unbiasedness` (full, independent Bernoulli assignment): with
//!   the outcomes held fixed the sharp null of no exposure effect makes every re-randomized
//!   contrast an unbiased draw around zero, whatever the outcomes are. The assignment is
//!   re-drawn from the declared design and the contrast recomputed; the mean of the
//!   re-randomized contrasts must be indistinguishable from zero. A mean away from zero means
//!   the exposure probabilities the estimator uses are not the ones the design implies.
//!
//! The state executes under both inference modes and is unlicensed: its verdicts are checks,
//! not a calibration claim, and no coverage record names it.

use antecedent_core::{AssignmentDesign, ExecutionContext, InterferenceQuery, StreamDomain};
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
    if let Some(estimated) = input.estimated {
        reports.push(exposure_overlap(estimated.minimum_exposure_probability));
        if input.suite == RefuteSuite::Full {
            if let AssignmentDesign::Bernoulli { probabilities } = &input.query.assignment {
                reports.push(randomization_unbiasedness(input, probabilities, ctx)?);
            }
        }
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
    probabilities: &[f64],
    ctx: &ExecutionContext,
) -> Result<RefutationReport, CausalError> {
    let units = input.assignment.len();
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
        let mut rng = ctx.rng.stream_for(
            StreamDomain::Resample,
            0x1F7E_5EED_u64.wrapping_add(u64::from(draw)),
        );
        let assignment: Vec<bool> = (0..units)
            .map(|unit| rng.next_f64() < bernoulli_probability(probabilities, unit))
            .collect();
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
    fn a_thin_exposure_probability_is_flagged() {
        assert!(exposure_overlap(0.2).passed);
        let thin = exposure_overlap(0.004);
        assert!(!thin.passed);
        assert!((thin.comparison - 0.004).abs() < 1e-12);
        assert!(!exposure_overlap(f64::NAN).passed);
    }
}
