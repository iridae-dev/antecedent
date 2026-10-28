//! Conditional continuous-dose response using supplied dose densities.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

/// Value of a fixed baseline-group dose rule under a kernel-smoothed intervention.
/// The target is the average response over the prespecified kernel window,
/// inverse weighted by the known observed-dose density; it is not `do(D=d)`.
#[derive(Clone, Debug, PartialEq)]
pub struct DosePolicyValue {
    /// Frozen policy group-dose map in group order.
    pub policy_doses: Vec<(String, f64)>,
    /// Frozen reference group-dose map in group order.
    pub reference_doses: Vec<(String, f64)>,
    /// Kernel-smoothed value under the fixed policy rule.
    pub policy_value: f64,
    /// Kernel-smoothed value under the fixed reference rule.
    pub reference_value: f64,
    /// Paired policy-minus-reference value.
    pub incremental_value: f64,
    /// Independent-row variance of policy value.
    pub policy_variance: f64,
    /// Independent-row variance of reference value.
    pub reference_variance: f64,
    /// Paired independent-row variance of incremental value.
    pub incremental_variance: f64,
    /// Pointwise normal interval for policy value, when supported.
    pub policy_interval_95: Option<[f64; 2]>,
    /// Pointwise normal interval for reference value, when supported.
    pub reference_interval_95: Option<[f64; 2]>,
    /// Pointwise normal interval for incremental value, when supported.
    pub incremental_interval_95: Option<[f64; 2]>,
    /// Smallest number of observed rows in any requested local window.
    pub minimum_local_rows: usize,
    /// Smallest Kish effective sample size in any requested local window.
    pub minimum_effective_sample_size: f64,
    /// Largest normalized weight in any requested local window.
    pub maximum_normalized_weight: f64,
    /// Smallest supplied observed-dose density in any requested local window.
    pub minimum_dose_density: f64,
}

/// Observed rows of a fixed dose-policy evaluation, aligned by row.
#[derive(Clone, Copy, Debug)]
pub struct DoseRows<'a> {
    /// Outcome per row.
    pub outcome: &'a [f64],
    /// Observed dose per row.
    pub dose: &'a [f64],
    /// Baseline group label per row.
    pub groups: &'a [String],
    /// Supplied observed-dose density per row.
    pub density: &'a [f64],
}

/// Fixed group-to-dose rules for the policy and its reference, each naming every
/// observed group exactly once.
#[derive(Clone, Copy, Debug)]
pub struct DoseRules<'a> {
    /// Policy dose per group.
    pub policy: &'a [(String, f64)],
    /// Reference dose per group.
    pub reference: &'a [(String, f64)],
}

/// The prespecified kernel window of a fixed dose-policy evaluation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DoseWindow {
    /// Epanechnikov kernel half-width, in dose units.
    pub bandwidth: f64,
    /// Fewest rows any local window may hold (at least two).
    pub min_local_support: usize,
    /// Whether the supplied density is known rather than estimated; pointwise intervals
    /// require it.
    pub density_known: bool,
}

/// Evaluate fixed group-to-dose policy and reference rules on the same rows.
///
/// Group labels and both rules must be set without using these outcomes. The
/// group fractions are fixed at their observed values. Independent subjects,
/// conditional exchangeability within the supplied groups, a correct known
/// dose density, and local positivity are required for the pointwise intervals.
#[allow(
    clippy::too_many_lines,
    reason = "per-group local fit of both rules in one loop that accumulates the shared support and weight diagnostics, then the interval assembly"
)]
pub fn fixed_dose_policy_value(
    rows: &DoseRows<'_>,
    rules: &DoseRules<'_>,
    window: &DoseWindow,
) -> Result<DosePolicyValue, String> {
    let DoseRows { outcome, dose, groups, density } = *rows;
    let DoseRules { policy, reference } = *rules;
    let DoseWindow { bandwidth, min_local_support, density_known } = *window;
    let n = outcome.len();
    if n < 2
        || dose.len() != n
        || groups.len() != n
        || density.len() != n
        || !bandwidth.is_finite()
        || bandwidth <= 0.0
        || min_local_support < 2
        || outcome.iter().chain(dose).any(|value| !value.is_finite())
        || density.iter().any(|value| !value.is_finite() || *value <= 0.0)
        || groups.iter().any(String::is_empty)
    {
        return Err("fixed dose policy requires aligned finite rows, positive density and bandwidth, and local support of at least two".into());
    }
    let observed = groups.iter().map(String::as_str).collect::<BTreeSet<_>>();
    let mapping = |rule: &[(String, f64)]| -> Result<BTreeMap<String, f64>, String> {
        let mut doses = BTreeMap::new();
        for (group, target) in rule {
            if !target.is_finite()
                || !observed.contains(group.as_str())
                || doses.insert(group.clone(), *target).is_some()
            {
                return Err(
                    "dose rule must name every observed group exactly once with a finite target"
                        .into(),
                );
            }
        }
        if doses.len() != observed.len() {
            return Err(
                "dose rule must name every observed group exactly once with a finite target".into(),
            );
        }
        Ok(doses)
    };
    let policy = mapping(policy)?;
    let reference = mapping(reference)?;
    let mut policy_score = vec![0.0; n];
    let mut reference_score = vec![0.0; n];
    let mut policy_value = 0.0;
    let mut reference_value = 0.0;
    let mut minimum_local_rows = usize::MAX;
    let mut minimum_effective_sample_size = f64::INFINITY;
    let mut maximum_normalized_weight: f64 = 0.0;
    let mut minimum_density = f64::INFINITY;
    let mut minimum_group_rows = usize::MAX;
    for group in observed {
        let indices = groups
            .iter()
            .enumerate()
            .filter_map(|(i, value)| (value == group).then_some(i))
            .collect::<Vec<_>>();
        let group_rows = indices.len();
        minimum_group_rows = minimum_group_rows.min(group_rows);
        let group_fraction = group_rows as f64 / n as f64;
        for (target, value, score) in [
            (policy[group], &mut policy_value, &mut policy_score),
            (reference[group], &mut reference_value, &mut reference_score),
        ] {
            let mut weights = Vec::new();
            for &i in &indices {
                let scaled = (dose[i] - target) / bandwidth;
                if scaled.abs() >= 1.0 {
                    continue;
                }
                let weight = 0.75 * (1.0 - scaled * scaled) / density[i];
                if !weight.is_finite() || weight <= 0.0 {
                    return Err("dose policy weights overflowed or lost positivity".into());
                }
                weights.push((i, weight));
                minimum_density = minimum_density.min(density[i]);
            }
            if weights.len() < min_local_support {
                return Err(format!(
                    "dose policy local support failure: group {group:?} at dose {target} has {} rows, below min_local_support={min_local_support}",
                    weights.len()
                ));
            }
            let sum_weight = weights.iter().map(|(_, weight)| weight).sum::<f64>();
            let sum_squared = weights.iter().map(|(_, weight)| weight * weight).sum::<f64>();
            if !sum_weight.is_finite()
                || sum_weight <= 0.0
                || !sum_squared.is_finite()
                || sum_squared <= 0.0
            {
                return Err("dose policy weights overflowed finite precision".into());
            }
            // Summed about the first local outcome, so an outcome offset far above the spread
            // does not cost the mean (and the paired difference of two means) its digits.
            let pivot = outcome[weights[0].0];
            let shifted_mean =
                weights.iter().map(|(i, weight)| weight * (outcome[*i] - pivot)).sum::<f64>()
                    / sum_weight;
            let mean = pivot + shifted_mean;
            if !mean.is_finite() {
                return Err("dose policy value overflowed finite precision".into());
            }
            *value += group_fraction * mean;
            for (i, weight) in weights.iter().copied() {
                let normalized = weight / sum_weight;
                score[i] = group_fraction * normalized * ((outcome[i] - pivot) - shifted_mean);
                maximum_normalized_weight = maximum_normalized_weight.max(normalized);
            }
            minimum_local_rows = minimum_local_rows.min(weights.len());
            minimum_effective_sample_size =
                minimum_effective_sample_size.min(sum_weight * sum_weight / sum_squared);
        }
    }
    let variance = |scores: &[f64]| {
        scores.iter().map(|score| score * score).sum::<f64>() * minimum_group_rows as f64
            / (minimum_group_rows - 1) as f64
    };
    let policy_variance = variance(&policy_score);
    let reference_variance = variance(&reference_score);
    let differences = policy_score
        .iter()
        .zip(&reference_score)
        .map(|(policy, reference)| policy - reference)
        .collect::<Vec<_>>();
    let incremental_variance = variance(&differences);
    let incremental_value = policy_value - reference_value;
    if [
        policy_value,
        reference_value,
        incremental_value,
        policy_variance,
        reference_variance,
        incremental_variance,
        minimum_effective_sample_size,
        maximum_normalized_weight,
        minimum_density,
    ]
    .iter()
    .any(|value| !value.is_finite())
    {
        return Err("dose policy value or paired variance overflowed finite precision".into());
    }
    let interval_supported = density_known
        && n >= 600
        && minimum_group_rows >= 300
        && minimum_local_rows >= 80
        && minimum_effective_sample_size >= 50.0
        && maximum_normalized_weight <= 0.05
        && minimum_density >= 0.2
        && policy_variance > 0.0
        && reference_variance > 0.0
        && incremental_variance > 0.0;
    let interval = |value: f64, variance: f64| {
        interval_supported.then(|| {
            let radius = antecedent_stats::NORMAL_Q975 * variance.sqrt();
            [value - radius, value + radius]
        })
    };
    Ok(DosePolicyValue {
        policy_doses: policy.into_iter().collect(),
        reference_doses: reference.into_iter().collect(),
        policy_value,
        reference_value,
        incremental_value,
        policy_variance,
        reference_variance,
        incremental_variance,
        policy_interval_95: interval(policy_value, policy_variance),
        reference_interval_95: interval(reference_value, reference_variance),
        incremental_interval_95: interval(incremental_value, incremental_variance),
        minimum_local_rows,
        minimum_effective_sample_size,
        maximum_normalized_weight,
        minimum_dose_density: minimum_density,
    })
}

/// One baseline-group response at a prespecified target dose.
#[derive(Clone, Debug, PartialEq)]
pub struct DoseResponsePoint {
    /// Baseline stratum.
    pub baseline_group: String,
    /// Prespecified dose.
    pub target_dose: f64,
    /// Local inverse-density weighted response.
    pub response: f64,
    /// Number of rows inside the kernel window.
    pub local_rows: usize,
    /// Kish effective sample size of local weights.
    pub effective_sample_size: f64,
    /// Smallest supplied local dose density.
    pub minimum_dose_density: f64,
    /// Largest normalized local weight.
    pub maximum_normalized_weight: f64,
    /// Descriptive local outcome deviation; not an inferential standard error.
    pub local_outcome_sd: f64,
}

/// Evaluate a local Epanechnikov-kernel response for every group and target.
///
/// The density is supplied by the caller; this function does not fit or
/// authenticate it. Every group-target cell must have the requested support.
pub fn conditional_dose_response(
    outcome: &[f64],
    dose: &[f64],
    groups: &[String],
    density: &[f64],
    target_doses: &[f64],
    bandwidth: f64,
    min_local_support: usize,
) -> Result<Vec<DoseResponsePoint>, String> {
    let n = outcome.len();
    if n == 0 || dose.len() != n || groups.len() != n || density.len() != n {
        return Err(
            "outcome, dose, groups, and dose density must have equal non-zero length".into()
        );
    }
    if target_doses.is_empty() || target_doses.iter().any(|value| !value.is_finite()) {
        return Err("target_doses must be non-empty and finite".into());
    }
    if !bandwidth.is_finite() || bandwidth <= 0.0 || min_local_support < 2 {
        return Err("bandwidth must be positive and min_local_support must be at least two".into());
    }
    if outcome.iter().chain(dose).any(|value| !value.is_finite()) {
        return Err("outcomes and doses must be finite".into());
    }
    if density.iter().any(|value| !value.is_finite() || *value <= 0.0) {
        return Err("continuous-dose positivity failure: supplied dose densities must be finite and positive".into());
    }
    if groups.iter().any(String::is_empty) {
        return Err("baseline group labels must be non-empty strings".into());
    }
    let strata: BTreeSet<&String> = groups.iter().collect();
    let mut output = Vec::with_capacity(strata.len() * target_doses.len());
    for stratum in strata {
        for &target in target_doses {
            let mut weighted_outcomes = Vec::new();
            let mut weights = Vec::new();
            let mut local_densities = Vec::new();
            for i in 0..n {
                let scaled_distance = (dose[i] - target) / bandwidth;
                if groups[i] != *stratum || scaled_distance.abs() >= 1.0 {
                    continue;
                }
                let kernel = 0.75 * (1.0 - scaled_distance.powi(2));
                let weight = kernel / density[i];
                if !weight.is_finite() || weight <= 0.0 {
                    return Err("dose-response weights overflowed or lost positivity".into());
                }
                weighted_outcomes.push(outcome[i]);
                weights.push(weight);
                local_densities.push(density[i]);
            }
            if weights.len() < min_local_support {
                return Err(format!(
                    "dose-response support failure: stratum {stratum:?} at dose {target} has {} local rows, below min_local_support={min_local_support}",
                    weights.len()
                ));
            }
            let sum_weight = weights.iter().sum::<f64>();
            let sum_weight_squared = weights.iter().map(|weight| weight.powi(2)).sum::<f64>();
            // Summed about the first local outcome (see `fixed_dose_policy_value`).
            let pivot = weighted_outcomes[0];
            let response = pivot
                + weights
                    .iter()
                    .zip(&weighted_outcomes)
                    .map(|(weight, outcome)| weight * (outcome - pivot))
                    .sum::<f64>()
                    / sum_weight;
            let effective_n = sum_weight.powi(2) / sum_weight_squared;
            let local_sd = (weights
                .iter()
                .zip(&weighted_outcomes)
                .map(|(weight, outcome)| weight * (outcome - response).powi(2))
                .sum::<f64>()
                / sum_weight)
                .sqrt();
            let max_normalized_weight =
                weights.iter().map(|weight| weight / sum_weight).fold(0.0_f64, f64::max);
            let minimum_density = local_densities.iter().copied().fold(f64::INFINITY, f64::min);
            if !response.is_finite() || !effective_n.is_finite() || !local_sd.is_finite() {
                return Err("dose-response estimate overflowed finite precision".into());
            }
            output.push(DoseResponsePoint {
                baseline_group: stratum.clone(),
                target_dose: target,
                response,
                local_rows: weights.len(),
                effective_sample_size: effective_n,
                minimum_dose_density: minimum_density,
                maximum_normalized_weight: max_normalized_weight,
                local_outcome_sd: local_sd,
            });
        }
    }
    Ok(output)
}

#[cfg(test)]
mod policy_tests {
    #![cfg_attr(
        test,
        allow(
            clippy::float_cmp,
            reason = "tests assert exact reproducibility of deterministic estimates"
        )
    )]
    use super::*;

    #[test]
    fn fixed_group_dose_policy_paired_intervals_cover_kernel_smoothed_truth() {
        const N: usize = 800;
        const REPLICATES: usize = 2_000;
        let groups =
            (0..N).map(|i| if i < N / 2 { "a" } else { "b" }.to_string()).collect::<Vec<_>>();
        let density = vec![1.0; N];
        let policy = [("a".to_string(), 0.7), ("b".to_string(), 0.8)];
        let reference = [("a".to_string(), 0.3), ("b".to_string(), 0.4)];
        let baseline_mean = (0..N)
            .map(|i| 1.0 + f64::from(i >= N / 2) + 0.35 * (0.13 * i as f64).sin())
            .sum::<f64>()
            / N as f64;
        let truth_policy = baseline_mean + (1.5 * 0.7 + 2.0 * 0.8) / 2.0;
        let truth_reference = baseline_mean + (1.5 * 0.3 + 2.0 * 0.4) / 2.0;
        let truths = [truth_policy, truth_reference, truth_policy - truth_reference];
        let mut covered = [0_usize; 3];
        for replicate in 0..REPLICATES {
            let dose = (0..N)
                .map(|i| crate::splitmix::keyed_unit(replicate as u64, i as u64))
                .collect::<Vec<_>>();
            let outcome = (0..N)
                .map(|i| {
                    1.0 + f64::from(i >= N / 2)
                        + 0.35 * (0.13 * i as f64).sin()
                        + if i < N / 2 { 1.5 * dose[i] } else { 2.0 * dose[i] }
                })
                .collect::<Vec<_>>();
            let fit = fixed_dose_policy_value(
                &DoseRows { outcome: &outcome, dose: &dose, groups: &groups, density: &density },
                &DoseRules { policy: &policy, reference: &reference },
                &DoseWindow { bandwidth: 0.2, min_local_support: 80, density_known: true },
            )
            .unwrap();
            if replicate == 0 {
                let external = fixed_dose_policy_value(
                    &DoseRows {
                        outcome: &outcome,
                        dose: &dose,
                        groups: &groups,
                        density: &density,
                    },
                    &DoseRules { policy: &policy, reference: &reference },
                    &DoseWindow { bandwidth: 0.2, min_local_support: 80, density_known: false },
                )
                .unwrap();
                assert_eq!(external.incremental_value, fit.incremental_value);
                assert!(external.policy_interval_95.is_none());
                assert!(external.reference_interval_95.is_none());
                assert!(external.incremental_interval_95.is_none());
                let weak_density = vec![0.1; N];
                let weak = fixed_dose_policy_value(
                    &DoseRows {
                        outcome: &outcome,
                        dose: &dose,
                        groups: &groups,
                        density: &weak_density,
                    },
                    &DoseRules { policy: &policy, reference: &reference },
                    &DoseWindow { bandwidth: 0.2, min_local_support: 80, density_known: true },
                )
                .unwrap();
                assert!(weak.incremental_interval_95.is_none());
            }
            let intervals = [
                fit.policy_interval_95.unwrap(),
                fit.reference_interval_95.unwrap(),
                fit.incremental_interval_95.unwrap(),
            ];
            for (j, [lower, upper]) in intervals.into_iter().enumerate() {
                covered[j] += usize::from(lower <= truths[j] && truths[j] <= upper);
            }
        }
        for (name, count) in ["policy", "reference", "incremental"].into_iter().zip(covered) {
            let rate = count as f64 / REPLICATES as f64;
            eprintln!("fixed dose {name} known-truth coverage: {count}/{REPLICATES} = {rate:.4}");
            assert!((0.93..=0.985).contains(&rate));
        }
    }

    #[test]
    fn fixed_group_dose_policy_support_boundary_covers_known_truth() {
        const N: usize = 600;
        const REPLICATES: usize = 2_000;
        let groups =
            (0..N).map(|i| if i < N / 2 { "a" } else { "b" }.to_string()).collect::<Vec<_>>();
        let density = vec![1.0; N];
        let policy = [("a".to_string(), 0.7), ("b".to_string(), 0.8)];
        let reference = [("a".to_string(), 0.3), ("b".to_string(), 0.4)];
        let baseline_mean = (0..N)
            .map(|i| 1.0 + f64::from(i >= N / 2) + 0.35 * (0.13 * i as f64).sin())
            .sum::<f64>()
            / N as f64;
        let truth_policy = baseline_mean + (1.5 * 0.7 + 2.0 * 0.8) / 2.0;
        let truth_reference = baseline_mean + (1.5 * 0.3 + 2.0 * 0.4) / 2.0;
        let truth = [truth_policy, truth_reference, truth_policy - truth_reference];
        let mut covered = [0_usize; 3];
        for rep in 0..REPLICATES {
            let dose = (0..N)
                .map(|i| crate::splitmix::keyed_unit(rep as u64, i as u64))
                .collect::<Vec<_>>();
            let outcome = (0..N)
                .map(|i| {
                    1.0 + f64::from(i >= N / 2)
                        + 0.35 * (0.13 * i as f64).sin()
                        + if i < N / 2 { 1.5 * dose[i] } else { 2.0 * dose[i] }
                })
                .collect::<Vec<_>>();
            let fit = fixed_dose_policy_value(
                &DoseRows { outcome: &outcome, dose: &dose, groups: &groups, density: &density },
                &DoseRules { policy: &policy, reference: &reference },
                &DoseWindow { bandwidth: 0.2, min_local_support: 80, density_known: true },
            )
            .unwrap();
            let intervals =
                [fit.policy_interval_95, fit.reference_interval_95, fit.incremental_interval_95];
            for (j, interval) in intervals.into_iter().enumerate() {
                let [lower, upper] = interval.expect("support-boundary interval");
                covered[j] += usize::from(lower <= truth[j] && truth[j] <= upper);
            }
        }
        for (name, count) in ["policy", "reference", "incremental"].into_iter().zip(covered) {
            let rate = count as f64 / REPLICATES as f64;
            eprintln!(
                "fixed dose support-boundary {name} coverage: {count}/{REPLICATES} = {rate:.4}"
            );
            assert!((0.93..=0.985).contains(&rate));
        }
    }

    /// Known truth with a dose density that is not constant and a nonlinear response, so that
    /// the inverse-density weighting (not its reciprocal), the kernel normalisation and the
    /// group fractions are all exercised; the uniform-density tests above cannot distinguish
    /// `1 / f` from `f`. Groups: `a` has density `2d` and `b` has `2(1 - d)` on `[0, 1]`,
    /// `E[Y | d, g] = offset_g + sin(4d)`. The truths come from
    /// `tests/oracle_data/continuous_dose_policy_truth.py` (40-digit quadrature).
    const TRUTH: [f64; 3] = [2.093_884_582_331_400_7, 2.352_416_912_832_5, -0.258_532_330_501_099_4];

    fn policy_rules() -> ([(String, f64); 2], [(String, f64); 2]) {
        (
            [("a".to_string(), 0.7), ("b".to_string(), 0.3)],
            [("a".to_string(), 0.5), ("b".to_string(), 0.5)],
        )
    }

    fn group_response(group_a: bool, dose: f64) -> f64 {
        (if group_a { 1.0 } else { 2.0 }) + (4.0 * dose).sin()
    }

    #[test]
    fn nonuniform_density_policy_value_matches_quadrature_on_a_quantile_design() {
        // Doses at the midpoint quantiles of each known density: the estimator is then a
        // deterministic Riemann sum of the target integrals, with no sampling noise.
        const PER_GROUP: usize = 20_000;
        let (policy, reference) = policy_rules();
        let mut dose = Vec::new();
        let mut density = Vec::new();
        let mut outcome = Vec::new();
        let mut groups = Vec::new();
        for group_a in [true, false] {
            for i in 0..PER_GROUP {
                let u = (i as f64 + 0.5) / PER_GROUP as f64;
                let d = if group_a { u.sqrt() } else { 1.0 - u.sqrt() };
                dose.push(d);
                density.push(if group_a { 2.0 * d } else { 2.0 * (1.0 - d) });
                outcome.push(group_response(group_a, d));
                groups.push(if group_a { "a" } else { "b" }.to_string());
            }
        }
        let fit = fixed_dose_policy_value(
            &DoseRows { outcome: &outcome, dose: &dose, groups: &groups, density: &density },
            &DoseRules { policy: &policy, reference: &reference },
            &DoseWindow { bandwidth: 0.2, min_local_support: 80, density_known: true },
        )
        .unwrap();
        for (got, truth) in
            [fit.policy_value, fit.reference_value, fit.incremental_value].into_iter().zip(TRUTH)
        {
            assert!((got - truth).abs() < 2e-5, "{got} vs {truth}");
        }
        // The density enters as a divisor: using the density itself instead moves the value.
        let inverted = density.iter().map(|f| 1.0 / f).collect::<Vec<_>>();
        let wrong = fixed_dose_policy_value(
            &DoseRows { outcome: &outcome, dose: &dose, groups: &groups, density: &inverted },
            &DoseRules { policy: &policy, reference: &reference },
            &DoseWindow { bandwidth: 0.2, min_local_support: 80, density_known: true },
        )
        .unwrap();
        assert!((wrong.policy_value - TRUTH[0]).abs() > 0.01);
    }

    #[test]
    fn nonuniform_density_heteroscedastic_intervals_cover_quadrature_truth() {
        const N: usize = 800;
        const REPLICATES: usize = 1_500;
        let (policy, reference) = policy_rules();
        let groups =
            (0..N).map(|i| if i < N / 2 { "a" } else { "b" }.to_string()).collect::<Vec<_>>();
        let mut covered = [0_usize; 3];
        let mut intervals_seen = 0;
        for replicate in 0..REPLICATES {
            let mut dose = Vec::with_capacity(N);
            let mut density = Vec::with_capacity(N);
            let mut outcome = Vec::with_capacity(N);
            for i in 0..N {
                let group_a = i < N / 2;
                let u = crate::splitmix::keyed_unit(replicate as u64, i as u64);
                let d = if group_a { u.sqrt() } else { 1.0 - u.sqrt() };
                let (u1, u2) = (
                    crate::splitmix::keyed_unit_stream(replicate as u64, i as u64, 1),
                    crate::splitmix::keyed_unit_stream(replicate as u64, i as u64, 2),
                );
                let normal = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
                dose.push(d);
                density.push(if group_a { 2.0 * d } else { 2.0 * (1.0 - d) });
                outcome.push(group_response(group_a, d) + (0.3 + 0.6 * d) * normal);
            }
            let fit = fixed_dose_policy_value(
                &DoseRows { outcome: &outcome, dose: &dose, groups: &groups, density: &density },
                &DoseRules { policy: &policy, reference: &reference },
                &DoseWindow { bandwidth: 0.2, min_local_support: 80, density_known: true },
            )
            .unwrap();
            let intervals =
                [fit.policy_interval_95, fit.reference_interval_95, fit.incremental_interval_95];
            if intervals.iter().all(Option::is_some) {
                intervals_seen += 1;
                for (j, interval) in intervals.into_iter().enumerate() {
                    let [lower, upper] = interval.unwrap();
                    covered[j] += usize::from(lower <= TRUTH[j] && TRUTH[j] <= upper);
                }
            }
        }
        assert!(intervals_seen > REPLICATES * 9 / 10, "supported intervals: {intervals_seen}");
        for (name, count) in ["policy", "reference", "incremental"].into_iter().zip(covered) {
            let rate = count as f64 / intervals_seen as f64;
            eprintln!("nonuniform dose {name} coverage: {count}/{intervals_seen} = {rate:.4}");
            assert!((0.93..=0.985).contains(&rate), "{name} coverage {rate}");
        }
    }

    #[test]
    fn dose_policy_refuses_unmapped_and_unsupported_local_targets() {
        let outcome = (0..40).map(|i| f64::from(i) / 40.0).collect::<Vec<_>>();
        let dose = outcome.clone();
        let groups = vec!["a".to_string(); 40];
        let density = vec![1.0; 40];
        let policy = [("a".to_string(), 0.7)];
        let reference = [("a".to_string(), 0.3)];
        assert!(
            fixed_dose_policy_value(
                &DoseRows { outcome: &outcome, dose: &dose, groups: &groups, density: &density },
                &DoseRules { policy: &[], reference: &reference },
                &DoseWindow { bandwidth: 0.2, min_local_support: 2, density_known: true }
            )
            .is_err()
        );
        assert!(
            fixed_dose_policy_value(
                &DoseRows { outcome: &outcome, dose: &dose, groups: &groups, density: &density },
                &DoseRules { policy: &policy, reference: &reference },
                &DoseWindow { bandwidth: 0.05, min_local_support: 10, density_known: true }
            )
            .is_err()
        );
        let fit = fixed_dose_policy_value(
            &DoseRows { outcome: &outcome, dose: &dose, groups: &groups, density: &density },
            &DoseRules { policy: &policy, reference: &reference },
            &DoseWindow { bandwidth: 0.2, min_local_support: 2, density_known: true },
        )
        .unwrap();
        assert!(fit.incremental_interval_95.is_none());
        let fit = fixed_dose_policy_value(
            &DoseRows { outcome: &outcome, dose: &dose, groups: &groups, density: &density },
            &DoseRules { policy: &policy, reference: &reference },
            &DoseWindow { bandwidth: 0.2, min_local_support: 2, density_known: false },
        )
        .unwrap();
        assert!(fit.incremental_interval_95.is_none());
    }
}
