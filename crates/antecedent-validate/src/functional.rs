//! Query-native stability checks for discrete identified functionals.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    ExecutionContext, InterventionalDistributionQuery, PathSpecificEffectQuery, VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_estimate::{
    FunctionalDistribution, FunctionalDistributionWorkspace, FunctionalEffect,
    InterventionalDistributionEstimate, RowResampling,
};
use antecedent_expr::IdentifiedEstimand;
use antecedent_identify::IdentificationResult;
use antecedent_kernels::erfc;

use crate::ValidationError;
use crate::common::{
    RefutationReport, STABILITY_ALPHA, STABILITY_REPLICATES, finish_p_report, map_replicates,
    with_contiguous_row_window, with_row_subset,
};

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
    for (slot, &(fraction, id)) in (if full {
        &[(0.8, "path.subset"), (0.5, "path.half_sample")][..]
    } else {
        &[(0.8, "path.subset")][..]
    })
    .iter()
    .enumerate()
    {
        let values = map_replicates(
            ctx,
            STABILITY_REPLICATES,
            || (),
            |(), ctx, replicate| {
                let replicate = u64::from(replicate);
                let subset = with_row_subset(
                    data,
                    fraction,
                    ctx,
                    subset_stream(crate::streams::FUNCTIONAL_SUBSET_80, slot, replicate),
                )?;
                let prepared = estimator.prepare(
                    &subset,
                    estimand,
                    &identification.arena,
                    identification.required_assumptions.clone(),
                    &extra,
                )?;
                Ok(estimator
                    .estimate(&prepared, &mut FunctionalDistributionWorkspace::default(), ctx)?
                    .ate)
            },
        )?;
        // Subsample refits are centred on the full-sample estimate by construction and
        // `original` is judged against single-replicate spread, so this cannot fail for a
        // wrong estimate: a pipeline stability check, not evidence for the effect.
        reports.push(finish_p_report(id, original, &values, original, STABILITY_ALPHA, |_, _| {
            "subsampled natural path contrast is unstable".to_owned()
        })?);
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
    let mut reports = vec![
        RefutationReport::builder("distribution.normalization")
            .original_ate(1.0)
            .refuted_ate(f64::from(normalized))
            .comparison(f64::from(normalized))
            .informative(true)
            .passed(normalized)
            .failure_condition(
                (!normalized).then(|| Arc::from("invalid conditional probability table")),
            )
            .replicates(1)
            .build(),
    ];
    let estimator = FunctionalDistribution::new();
    for (slot, &(fraction, id)) in (if full {
        &[(0.8, "distribution.subset_tv"), (0.5, "distribution.half_sample_tv")][..]
    } else {
        &[(0.8, "distribution.subset_tv")][..]
    })
    .iter()
    .enumerate()
    {
        let distances = map_replicates(
            ctx,
            STABILITY_REPLICATES,
            || (),
            |(), ctx, replicate| {
                let replicate = u64::from(replicate);
                let subset = with_row_subset(
                    data,
                    fraction,
                    ctx,
                    subset_stream(crate::streams::FUNCTIONAL_SUBSET_50, slot, replicate),
                )?;
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
                Ok(max_conditional_tv(&original.atoms, &estimate.atoms))
            },
        )?;
        // Descriptive robustness threshold, not a calibrated hypothesis test.
        let mean = distances.iter().sum::<f64>() / distances.len() as f64;
        let passed = mean.is_finite() && mean <= 0.1;
        reports.push(
            RefutationReport::builder(id)
                .original_ate(0.0)
                .refuted_ate(mean)
                .comparison(mean)
                .informative(true)
                .passed(passed)
                .failure_condition(
                    (!passed)
                        .then(|| Arc::from("mean maximum conditional total variation exceeds 0.1")),
                )
                .replicates(STABILITY_REPLICATES)
                .build(),
        );
    }
    Ok(reports)
}

/// Smallest bootstrap size of the split-half stationarity check's standard errors.
const SPLIT_HALF_REPLICATES: u32 = 40;

/// Check a contrast identified on the unrolled window of a temporal ADMG against the window
/// table it was evaluated on.
///
/// Every refit reads whole rows of the table (a row is one intact stretch of `span` consecutive
/// series times), so a row subset is a valid input of the same plug-in; what changes is which
/// stretches of the series enter it.
///
/// * `window.stretch_80` (and, under full validation, `window.stretch_50`): refit on a random
///   contiguous stretch of the windows. Like the other subsample refits it is centred on the
///   full-sample value, so it is a pipeline stability check and not evidence for the effect.
/// * `window.placebo_treatment`: rotate the treatment columns together by a random number of
///   windows no smaller than the window, which keeps the treatment law and its dependence
///   across treated periods and removes every link to the rest of the window, then refit. The
///   rotated contrast must be indistinguishable from zero.
/// * `window.split_half` (full validation only): the contrast on the first half of the windows
///   against the contrast on the second half, each with a circular-block bootstrap standard
///   error, tests the stationarity of the series that the window plug-in assumes.
///
/// `coordinates` are the window columns the functional reads, `treatment` the treated-period
/// columns among them, `span` the series times one row spans.
///
/// # Errors
///
/// A refit that fails (empty support in a subset, a non-finite value).
#[allow(
    clippy::too_many_arguments,
    reason = "the window table, its identified functional, the columns it reads, the treated columns and the span are one refit problem; a struct would only rename the arguments"
)]
pub fn refute_window_contrast(
    table: &TabularData,
    identification: &IdentificationResult,
    estimand: &IdentifiedEstimand,
    coordinates: &[VariableId],
    treatment: &[VariableId],
    span: u32,
    original: f64,
    full: bool,
    ctx: &ExecutionContext,
) -> Result<Vec<RefutationReport>, ValidationError> {
    let estimator = FunctionalEffect::new();
    let refit = |data: &TabularData, ctx: &ExecutionContext| -> Result<f64, ValidationError> {
        let prepared = estimator.prepare(
            data,
            estimand,
            &identification.arena,
            identification.required_assumptions.clone(),
            coordinates,
        )?;
        Ok(estimator.estimate(&prepared, &mut FunctionalDistributionWorkspace::default(), ctx)?.ate)
    };
    let stretches: &[(f64, &str, u64)] = if full {
        &[
            (0.8, "window.stretch_80", crate::streams::WINDOW_STRETCH_80),
            (0.5, "window.stretch_50", crate::streams::WINDOW_STRETCH_50),
        ]
    } else {
        &[(0.8, "window.stretch_80", crate::streams::WINDOW_STRETCH_80)]
    };
    let mut reports = Vec::new();
    for &(fraction, id, base) in stretches {
        let values = map_replicates(
            ctx,
            STABILITY_REPLICATES,
            || (),
            |(), ctx, replicate| {
                let stretch =
                    with_contiguous_row_window(table, fraction, ctx, base + u64::from(replicate))?;
                refit(&stretch, ctx)
            },
        )?;
        reports.push(finish_p_report(id, original, &values, original, STABILITY_ALPHA, |_, _| {
            "the contrast refitted on a contiguous stretch of windows is unstable".to_owned()
        })?);
    }

    let rows = table.row_count();
    let span_rows = span as usize;
    // Rotations by `span ..= rows - span` windows: the rotated treatment never overlaps the
    // stretch of the series its own window came from.
    let rotations = (rows + 1).saturating_sub(2 * span_rows);
    if rotations == 0 || treatment.is_empty() {
        reports.push(
            RefutationReport::builder("window.placebo_treatment")
                .original_ate(original)
                .refuted_ate(0.0)
                .comparison(0.0)
                .informative(false)
                .passed(false)
                .failure_condition(Some(Arc::from(
                    "the series is too short to rotate the treatment by a full window",
                )))
                .replicates(0)
                .build(),
        );
    } else {
        let values = map_replicates(
            ctx,
            STABILITY_REPLICATES,
            || (),
            |(), ctx, replicate| {
                let mut rng =
                    ctx.rng.stream(crate::streams::WINDOW_PLACEBO_SHIFT + u64::from(replicate));
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "next_f64 is in [0, 1), so the floored product is in [0, rotations) and is min-bounded to the last rotation"
                )]
                let offset = ((rng.next_f64() * rotations as f64) as usize).min(rotations - 1);
                let rotated = rotate_columns(table, treatment, span_rows + offset)?;
                refit(&rotated, ctx)
            },
        )?;
        reports.push(finish_p_report(
            "window.placebo_treatment",
            original,
            &values,
            0.0,
            STABILITY_ALPHA,
            |mean, p_value| {
                format!(
                    "the contrast of a rotated treatment is {mean:.4} (p = {p_value:.4}) where \
                     no effect is possible"
                )
            },
        )?);
    }

    if full {
        reports.push(split_half(table, identification, estimand, coordinates, span, ctx)?);
    }
    Ok(reports)
}

/// The contrast on the first half of the windows against the contrast on the second, each with
/// a circular-block bootstrap standard error.
fn split_half(
    table: &TabularData,
    identification: &IdentificationResult,
    estimand: &IdentifiedEstimand,
    coordinates: &[VariableId],
    span: u32,
    ctx: &ExecutionContext,
) -> Result<RefutationReport, ValidationError> {
    let estimator = FunctionalEffect::new()
        .with_bootstrap_replicates(SPLIT_HALF_REPLICATES)
        .with_row_resampling(
            RowResampling::WindowBlocks { span },
            antecedent_estimate::util::SamplingDeclaration::Undeclared,
        );
    let rows = table.row_count();
    let half = rows / 2;
    let mut halves = [(f64::NAN, f64::NAN); 2];
    for (slot, (start, end)) in [(0, half), (half, rows)].into_iter().enumerate() {
        let part = row_range(table, start, end)?;
        let prepared = estimator.prepare(
            &part,
            estimand,
            &identification.arena,
            identification.required_assumptions.clone(),
            coordinates,
        )?;
        let estimate =
            estimator.estimate(&prepared, &mut FunctionalDistributionWorkspace::default(), ctx)?;
        halves[slot] = (estimate.ate, estimate.se_bootstrap().unwrap_or(f64::NAN));
    }
    let [(first, first_se), (second, second_se)] = halves;
    // A half that one circular block covers has a single possible resample: its bootstrap
    // spread is exactly zero because nothing can vary, not because the contrast is precise, and
    // a z test on it would reject (or accept) stationarity on no evidence.
    let resamplable = [half, rows - half].iter().all(|&part| {
        RowResampling::WindowBlocks { span }.block_length(part).is_some_and(|block| block < part)
    });
    let usable = resamplable && [first_se, second_se].iter().all(|se| se.is_finite() && *se >= 0.0);
    if !usable {
        return Ok(RefutationReport::builder("window.split_half")
            .original_ate(first)
            .refuted_ate(second)
            .comparison(0.0)
            .informative(false)
            .passed(false)
            .failure_condition(Some(Arc::from(
                "a half of the windows gave no block-bootstrap standard error (none, or only one resampling block)",
            )))
            .replicates(SPLIT_HALF_REPLICATES)
            .build());
    }
    let variance = first_se * first_se + second_se * second_se;
    let p_value = if variance > 0.0 {
        let z = (first - second) / variance.sqrt();
        erfc(z.abs() / std::f64::consts::SQRT_2).clamp(0.0, 1.0)
    } else if (first - second).abs() <= 1e-12 {
        // Both halves are exactly the same constant under every resample.
        1.0
    } else {
        0.0
    };
    let passed = p_value >= STABILITY_ALPHA;
    Ok(RefutationReport::builder("window.split_half")
        .original_ate(first)
        .refuted_ate(second)
        .comparison(p_value)
        .informative(true)
        .passed(passed)
        .failure_condition((!passed).then(|| {
            Arc::from(format!(
                "the contrast is {first:.4} on the first half of the windows and {second:.4} on \
                 the second (p = {p_value:.4}); the series is not stationary over the template"
            ))
        }))
        .replicates(SPLIT_HALF_REPLICATES)
        .build())
}

/// The rows `start..end` of `table` as a table of its own; every column is read as float64
/// (a window table holds nothing else).
fn row_range(
    table: &TabularData,
    start: usize,
    end: usize,
) -> Result<TabularData, ValidationError> {
    let mut named = Vec::with_capacity(table.schema().len());
    for variable in table.schema().variables() {
        let values = table.float64_values(variable.id).map_err(ValidationError::from)?;
        named.push((Arc::clone(&variable.name), values[start..end].to_vec()));
    }
    TabularData::from_f64_columns(
        named.iter().map(|(name, values)| (Arc::clone(name), values.as_slice())),
    )
    .map_err(ValidationError::from)
}

/// `table` with each column of `columns` read `shift` rows further down, circularly; every
/// other column is untouched.
fn rotate_columns(
    table: &TabularData,
    columns: &[VariableId],
    shift: usize,
) -> Result<TabularData, ValidationError> {
    let mut named = Vec::with_capacity(table.schema().len());
    for variable in table.schema().variables() {
        let mut values = table.float64_values(variable.id).map_err(ValidationError::from)?;
        if columns.contains(&variable.id) && !values.is_empty() {
            let by = shift % values.len();
            values.rotate_left(by);
        }
        named.push((Arc::clone(&variable.name), values));
    }
    TabularData::from_f64_columns(
        named.iter().map(|(name, values)| (Arc::clone(name), values.as_slice())),
    )
    .map_err(ValidationError::from)
}

/// RNG stream of replicate `replicate` of subset-fraction `slot`. Each fraction gets its own
/// 2^16-stream block so the 50% subsets are drawn from uniforms independent of the 80% subsets
/// (sharing streams would nest `u < 0.5` inside `u < 0.8`). Slot 0 keeps the historical stream.
fn subset_stream(base: u64, slot: usize, replicate: u64) -> u64 {
    base + ((slot as u64) << 16) + replicate
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
    fn subset_fractions_draw_from_disjoint_streams() {
        // 20 replicates per fraction: no stream id is shared between the 80% and 50% slots.
        let a: Vec<u64> =
            (0..20).map(|r| subset_stream(crate::streams::FUNCTIONAL_SUBSET_80, 0, r)).collect();
        let b: Vec<u64> =
            (0..20).map(|r| subset_stream(crate::streams::FUNCTIONAL_SUBSET_80, 1, r)).collect();
        assert!(a.iter().all(|s| !b.contains(s)));
        // The historical 80% streams are unchanged.
        assert_eq!(a[3], 0xF012_0003);
    }

    fn counted_table() -> TabularData {
        let a = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
        TabularData::from_f64_columns([("a", a.as_slice()), ("b", b.as_slice())]).unwrap()
    }

    #[test]
    fn rotating_the_treatment_column_moves_only_that_column_by_the_shift() {
        let table = counted_table();
        let rotated = rotate_columns(&table, &[VariableId::from_raw(0)], 2).unwrap();
        assert_eq!(
            rotated.float64_values(VariableId::from_raw(0)).unwrap(),
            vec![2.0, 3.0, 4.0, 5.0, 0.0, 1.0]
        );
        assert_eq!(
            rotated.float64_values(VariableId::from_raw(1)).unwrap(),
            vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
        );
        // A shift of a whole length is the identity.
        let same = rotate_columns(&table, &[VariableId::from_raw(0)], 6).unwrap();
        assert_eq!(
            same.float64_values(VariableId::from_raw(0)).unwrap(),
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        );
    }

    #[test]
    fn a_row_range_keeps_every_column_aligned() {
        let part = row_range(&counted_table(), 2, 5).unwrap();
        assert_eq!(part.row_count(), 3);
        assert_eq!(part.float64_values(VariableId::from_raw(0)).unwrap(), vec![2.0, 3.0, 4.0]);
        assert_eq!(part.float64_values(VariableId::from_raw(1)).unwrap(), vec![12.0, 13.0, 14.0]);
    }

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
