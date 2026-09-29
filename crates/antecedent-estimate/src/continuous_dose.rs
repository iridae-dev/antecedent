//! Conditional continuous-dose response using supplied dose densities.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

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

/// Evaluate fixed group-to-dose policy and reference rules on the same rows.
///
/// Group labels and both rules must be set without using these outcomes. The
/// group fractions are fixed at their observed values. Independent subjects,
/// conditional exchangeability within the supplied groups, a correct known
/// dose density, and local positivity are required for the pointwise intervals.
// arity and length mirror the estimator's fixed statistical contract; refactor would change behavior
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_lines)]
pub fn fixed_dose_policy_value(
    outcome: &[f64],
    dose: &[f64],
    groups: &[String],
    density: &[f64],
    policy: &[(String, f64)],
    reference: &[(String, f64)],
    bandwidth: f64,
    min_local_support: usize,
    density_known: bool,
) -> Result<DosePolicyValue, String> {
    use std::collections::BTreeMap;
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
            let mean =
                weights.iter().map(|(i, weight)| weight * outcome[*i]).sum::<f64>() / sum_weight;
            if !mean.is_finite() {
                return Err("dose policy value overflowed finite precision".into());
            }
            *value += group_fraction * mean;
            for (i, weight) in weights.iter().copied() {
                let normalized = weight / sum_weight;
                score[i] = group_fraction * normalized * (outcome[i] - mean);
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

/// Evaluate a local triangular-kernel response for every group and target.
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
            let response = weights
                .iter()
                .zip(&weighted_outcomes)
                .map(|(weight, outcome)| weight * outcome)
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

    fn uniform(mut state: u64) -> f64 {
        state ^= state >> 30;
        state = state.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        state ^= state >> 27;
        state = state.wrapping_mul(0x94D0_49BB_1331_11EB);
        ((state ^ (state >> 31)) >> 11) as f64 / (1_u64 << 53) as f64
    }

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
                .map(|i| {
                    uniform(
                        (replicate as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                            ^ (i as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03),
                    )
                })
                .collect::<Vec<_>>();
            let outcome = (0..N)
                .map(|i| {
                    1.0 + f64::from(i >= N / 2)
                        + 0.35 * (0.13 * i as f64).sin()
                        + if i < N / 2 { 1.5 * dose[i] } else { 2.0 * dose[i] }
                })
                .collect::<Vec<_>>();
            let fit = fixed_dose_policy_value(
                &outcome, &dose, &groups, &density, &policy, &reference, 0.2, 80, true,
            )
            .unwrap();
            if replicate == 0 {
                let external = fixed_dose_policy_value(
                    &outcome, &dose, &groups, &density, &policy, &reference, 0.2, 80, false,
                )
                .unwrap();
                assert_eq!(external.incremental_value, fit.incremental_value);
                assert!(external.policy_interval_95.is_none());
                assert!(external.reference_interval_95.is_none());
                assert!(external.incremental_interval_95.is_none());
                let weak_density = vec![0.1; N];
                let weak = fixed_dose_policy_value(
                    &outcome,
                    &dose,
                    &groups,
                    &weak_density,
                    &policy,
                    &reference,
                    0.2,
                    80,
                    true,
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
                .map(|i| {
                    uniform(
                        (rep as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                            ^ (i as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03),
                    )
                })
                .collect::<Vec<_>>();
            let outcome = (0..N)
                .map(|i| {
                    1.0 + f64::from(i >= N / 2)
                        + 0.35 * (0.13 * i as f64).sin()
                        + if i < N / 2 { 1.5 * dose[i] } else { 2.0 * dose[i] }
                })
                .collect::<Vec<_>>();
            let fit = fixed_dose_policy_value(
                &outcome, &dose, &groups, &density, &policy, &reference, 0.2, 80, true,
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
                &outcome,
                &dose,
                &groups,
                &density,
                &[],
                &reference,
                0.2,
                2,
                true
            )
            .is_err()
        );
        assert!(
            fixed_dose_policy_value(
                &outcome, &dose, &groups, &density, &policy, &reference, 0.05, 10, true
            )
            .is_err()
        );
        let fit = fixed_dose_policy_value(
            &outcome, &dose, &groups, &density, &policy, &reference, 0.2, 2, true,
        )
        .unwrap();
        assert!(fit.incremental_interval_95.is_none());
        let fit = fixed_dose_policy_value(
            &outcome, &dose, &groups, &density, &policy, &reference, 0.2, 2, false,
        )
        .unwrap();
        assert!(fit.incremental_interval_95.is_none());
    }
}
