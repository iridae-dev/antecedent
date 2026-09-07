//! Query-native stability checks for discrete identified functionals.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::too_many_arguments)]

use std::sync::Arc;

use antecedent_core::{ExecutionContext, InterventionalDistributionQuery, PathSpecificEffectQuery};
use antecedent_data::TabularData;
use antecedent_estimate::{
    FunctionalDistribution, FunctionalDistributionWorkspace, FunctionalEffect,
    InterventionalDistributionEstimate,
};
use antecedent_expr::IdentifiedEstimand;
use antecedent_identify::IdentificationResult;

use crate::ValidationError;
use crate::common::{RefutationReport, replicate_p_value, with_row_subset};

/// Refit the identified path functional on independent 80% row subsets.
/// Full validation also probes 50% subsets. The path assignment and natural
/// contrast remain frozen; these are never backdoor ATE refits.
pub fn refute_path(
    data: &TabularData,
    query: &PathSpecificEffectQuery,
    identification: &IdentificationResult,
    estimand: &IdentifiedEstimand,
    original: f64,
    full: bool,
    ctx: &ExecutionContext,
) -> Result<Vec<RefutationReport>, ValidationError> {
    let mut extra = vec![query.treatment, query.outcome];
    extra.extend(query.path_nodes.iter().copied());
    let estimator = FunctionalEffect::new();
    let mut reports = Vec::new();
    for &(fraction, id) in if full {
        &[(0.8, "path.subset"), (0.5, "path.half_sample")][..]
    } else {
        &[(0.8, "path.subset")][..]
    } {
        let mut values = Vec::new();
        for replicate in 0..20_u64 {
            if ctx.cancellation.is_cancelled() {
                return Err(ValidationError::Cancelled);
            }
            let subset = with_row_subset(data, fraction, ctx, 0xF012_0000 + replicate)?;
            let prepared = estimator.prepare(
                &subset,
                estimand,
                &identification.arena,
                identification.required_assumptions.clone(),
                &extra,
            )?;
            values.push(
                estimator
                    .estimate(&prepared, &mut FunctionalDistributionWorkspace::default(), ctx)?
                    .ate,
            );
        }
        let p = replicate_p_value(&values, original);
        reports.push(RefutationReport::new(
            id,
            original,
            values.iter().sum::<f64>() / values.len() as f64,
            p,
            true,
            p >= 0.05,
            (p < 0.05).then(|| Arc::from("subsampled natural path contrast is unstable")),
            20,
        ));
    }
    Ok(reports)
}

/// Check conditional probability normalization and stability of the entire
/// interventional probability table. Full validation also probes 50% subsets.
/// The comparison is maximum conditional total variation, including lost atoms;
/// outcome means (which may be undefined) do not enter the check.
pub fn refute_distribution(
    data: &TabularData,
    query: &InterventionalDistributionQuery,
    identification: &IdentificationResult,
    estimand: &IdentifiedEstimand,
    original: &InterventionalDistributionEstimate,
    full: bool,
    ctx: &ExecutionContext,
) -> Result<Vec<RefutationReport>, ValidationError> {
    let mut groups = Vec::new();
    for atom in original.atoms.iter() {
        if !groups.contains(&atom.conditioning) {
            groups.push(atom.conditioning.clone());
        }
    }
    let normalized = !groups.is_empty()
        && groups.iter().all(|group| {
            let atoms: Vec<_> =
                original.atoms.iter().filter(|a| &a.conditioning == group).collect();
            atoms.iter().all(|a| a.probability.is_finite() && (0.0..=1.0).contains(&a.probability))
                && (atoms.iter().map(|a| a.probability).sum::<f64>() - 1.0).abs() <= 1e-8
        });
    let mut reports = vec![RefutationReport::new(
        "distribution.normalization",
        1.0,
        f64::from(normalized),
        f64::from(normalized),
        true,
        normalized,
        (!normalized).then(|| Arc::from("invalid conditional probability table")),
        1,
    )];
    let estimator = FunctionalDistribution::new();
    for &(fraction, id) in if full {
        &[(0.8, "distribution.subset_tv"), (0.5, "distribution.half_sample_tv")][..]
    } else {
        &[(0.8, "distribution.subset_tv")][..]
    } {
        let mut distances = Vec::new();
        for replicate in 0..20_u64 {
            if ctx.cancellation.is_cancelled() {
                return Err(ValidationError::Cancelled);
            }
            let subset = with_row_subset(data, fraction, ctx, 0xF013_0000 + replicate)?;
            let prepared = estimator.prepare(
                &subset,
                query,
                estimand,
                &identification.arena,
                identification.required_assumptions.clone(),
            )?;
            let estimate = estimator.estimate(
                &prepared,
                &[],
                &mut FunctionalDistributionWorkspace::default(),
                ctx,
            )?;
            let distance = max_conditional_tv(&original.atoms, &estimate.atoms);
            distances.push(distance);
        }
        // Descriptive robustness threshold, not a calibrated hypothesis test.
        let mean = distances.iter().sum::<f64>() / distances.len() as f64;
        let passed = mean.is_finite() && mean <= 0.1;
        reports.push(RefutationReport::new(
            id,
            0.0,
            mean,
            mean,
            true,
            passed,
            (!passed).then(|| Arc::from("mean maximum conditional total variation exceeds 0.1")),
            20,
        ));
    }
    Ok(reports)
}

fn max_conditional_tv(
    original: &[antecedent_estimate::DistributionAtom],
    estimate: &[antecedent_estimate::DistributionAtom],
) -> f64 {
    let mut groups = Vec::new();
    for atom in original.iter().chain(estimate) {
        if !groups.contains(&atom.conditioning) {
            groups.push(atom.conditioning.clone());
        }
    }
    groups
        .iter()
        .map(|group| {
            if !original.iter().any(|a| &a.conditioning == group)
                || !estimate.iter().any(|a| &a.conditioning == group)
            {
                return 1.0;
            }
            let mut l1 = 0.0;
            for atom in original.iter().filter(|a| &a.conditioning == group) {
                let p = estimate
                    .iter()
                    .find(|a| a.conditioning == atom.conditioning && a.outcomes == atom.outcomes)
                    .map_or(0.0, |a| a.probability);
                l1 += (atom.probability - p).abs();
            }
            for atom in estimate.iter().filter(|a| &a.conditioning == group) {
                if !original
                    .iter()
                    .any(|a| a.conditioning == atom.conditioning && a.outcomes == atom.outcomes)
                {
                    l1 += atom.probability.abs();
                }
            }
            0.5 * l1
        })
        .fold(0.0_f64, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{Value, VariableId};
    use antecedent_estimate::DistributionAtom;

    #[test]
    fn full_table_distance_detects_equal_mean_distributions_and_lost_strata() {
        let atom = |value, probability| DistributionAtom {
            outcomes: Arc::from([(VariableId::from_raw(0), Value::f64(value))]),
            conditioning: Arc::from([]),
            probability,
        };
        // Both distributions have mean one; their TV is one, not zero.
        let extreme = [atom(0.0, 0.5), atom(2.0, 0.5)];
        let middle = [atom(1.0, 1.0)];
        assert!((max_conditional_tv(&extreme, &middle) - 1.0).abs() < 1e-12);
        assert!(max_conditional_tv(&extreme, &extreme).abs() < 1e-12);
        assert!((max_conditional_tv(&extreme, &[]) - 1.0).abs() < 1e-12);
    }
}
