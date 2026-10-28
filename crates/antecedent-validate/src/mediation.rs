//! Refuters for the linear temporal mediation contrast.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use antecedent_core::{ExecutionContext, MediationContrast, MediationQuery};
use antecedent_data::{LaggedColumn, TableView, TabularData, TimeIndex, TimeSeriesData};
use antecedent_estimate::{TemporalMediationEstimate, TemporalMediationEstimator};
use antecedent_expr::IdentifiedEstimand;

use crate::ValidationError;
use crate::common::{
    RefutationReport, STABILITY_ALPHA, STABILITY_REPLICATES, fill_gaussian, finish_p_report,
    map_replicates, with_contiguous_row_window, with_extra_float, with_replaced_float,
};

/// Query-native cheap/full dispatch. Temporal mediation uses the mediation
/// suite, not ATE [`crate::RefutationProblem`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QueryRefutationPlan {
    /// Placebo-mediator and RCC; full adds a contiguous-window stability check.
    TemporalMediation {
        /// Whether the full suite is requested.
        full: bool,
    },
}

impl QueryRefutationPlan {
    /// Plan for [`refute_temporal_mediation_adjusted`].
    #[must_use]
    pub const fn temporal_mediation(full: bool) -> Self {
        Self::TemporalMediation { full }
    }

    /// Run this plan on one identified atom against that atom's own contrast.
    ///
    /// # Errors
    ///
    /// Invalid refit data, regression failure, or cancellation.
    pub fn refute_temporal_atom(
        self,
        data: &TimeSeriesData,
        estimand: &IdentifiedEstimand,
        query: &MediationQuery,
        original: &TemporalMediationEstimate,
        adjustment: &[LaggedColumn],
        ctx: &ExecutionContext,
    ) -> Result<Vec<RefutationReport>, ValidationError> {
        match self {
            Self::TemporalMediation { full } => refute_temporal_mediation_adjusted(
                data, estimand, query, original, full, adjustment, ctx,
            ),
        }
    }

    /// [`Self::refute_temporal_atom`] on a series that may have missing time points when
    /// `gaps` is set (see [`refute_temporal_mediation_with_gaps`]).
    ///
    /// # Errors
    ///
    /// Invalid refit data, regression failure, or cancellation.
    #[allow(
        clippy::too_many_arguments,
        reason = "the atom's estimate inputs plus the gap declaration"
    )]
    pub fn refute_temporal_atom_with_gaps(
        self,
        data: &TimeSeriesData,
        estimand: &IdentifiedEstimand,
        query: &MediationQuery,
        original: &TemporalMediationEstimate,
        adjustment: &[LaggedColumn],
        gaps: bool,
        ctx: &ExecutionContext,
    ) -> Result<Vec<RefutationReport>, ValidationError> {
        match self {
            Self::TemporalMediation { full } => refute_temporal_mediation_with_gaps(
                data, estimand, query, original, full, adjustment, gaps, ctx,
            ),
        }
    }

    /// Mix per-atom reports of the same refuter by graph weight.
    ///
    /// The mixed check passes only if every positive-weight contributing atom
    /// passed ([`RefutationReport::mixture_weighted`]).
    #[must_use]
    pub fn mix_weighted(
        items: impl IntoIterator<Item = (f64, Vec<RefutationReport>)>,
    ) -> Vec<RefutationReport> {
        let mut order = Vec::new();
        let mut by_refuter: HashMap<Arc<str>, Vec<(f64, RefutationReport)>> = HashMap::new();
        for (weight, reports) in items {
            if weight <= 0.0 {
                continue;
            }
            for report in reports {
                let bucket = by_refuter.entry(Arc::clone(&report.refuter)).or_insert_with(|| {
                    order.push(Arc::clone(&report.refuter));
                    Vec::new()
                });
                bucket.push((weight, report));
            }
        }
        let mut mixed = Vec::with_capacity(order.len());
        for id in order {
            let Some(bucket) = by_refuter.get(&id) else {
                continue;
            };
            let borrowed: Vec<(f64, &RefutationReport)> =
                bucket.iter().map(|(w, r)| (*w, r)).collect();
            if let Some(report) = RefutationReport::mixture_weighted(&borrowed) {
                mixed.push(report);
            }
        }
        mixed
    }
}

/// Whether any column the mediation fit reads (treatment, outcome, mediators, adjustment
/// covariates) has a missing time point.
fn reads_a_gap(data: &TimeSeriesData, query: &MediationQuery, adjustment: &[LaggedColumn]) -> bool {
    let read = |id| {
        data.storage().columns().iter().any(|column| match column {
            antecedent_data::OwnedColumn::Float64(c) => c.id == id && !c.validity.is_all_valid(),
            _ => false,
        })
    };
    read(query.treatment)
        || read(query.outcome)
        || query.mediators.iter().any(|m| read(*m))
        || adjustment.iter().any(|a| read(a.variable))
}

/// Run the native suite with the same graph-derived adjustment as the estimate.
///
/// The series must be intact: this entry point declares no gaps, so a series with a missing
/// time point in a column the fit reads is refused (`NotApplicable`) rather than refitted as
/// if it were complete. Use [`refute_temporal_mediation_with_gaps`] to refute a gapped series.
///
/// # Errors
///
/// A gapped series, or as [`refute_temporal_mediation_with_gaps`].
pub fn refute_temporal_mediation_adjusted(
    data: &TimeSeriesData,
    estimand: &IdentifiedEstimand,
    query: &MediationQuery,
    original: &TemporalMediationEstimate,
    full: bool,
    adjustment: &[LaggedColumn],
    ctx: &ExecutionContext,
) -> Result<Vec<RefutationReport>, ValidationError> {
    if reads_a_gap(data, query, adjustment) {
        return Err(ValidationError::NotApplicable {
            message: "the series has missing time points but this mediation validation declares \
                      none; refute it with refute_temporal_mediation_with_gaps",
        });
    }
    refute_temporal_mediation_with_gaps(
        data, estimand, query, original, full, adjustment, false, ctx,
    )
}

/// [`refute_temporal_mediation_adjusted`] on a series that may have missing time points when
/// `gaps` is set: every refit drops the windows that read a gap, as the published estimate
/// does, and the perturbations (a noise mediator, an extra covariate, a contiguous row window)
/// keep the series' own missing values, so each refit sees the same intact windows.
///
/// # Errors
///
/// As [`refute_temporal_mediation_adjusted`].
#[allow(
    clippy::float_cmp,
    reason = "the treatment column is compared to its own deduplicated level values, which are copies of the same floats"
)]
#[allow(
    clippy::too_many_lines,
    clippy::too_many_arguments,
    reason = "one replicate pass feeds three refuters that share a noise draw, then the overlap diagnostic"
)]
pub fn refute_temporal_mediation_with_gaps(
    data: &TimeSeriesData,
    estimand: &IdentifiedEstimand,
    query: &MediationQuery,
    original: &TemporalMediationEstimate,
    full: bool,
    adjustment: &[LaggedColumn],
    gaps: bool,
    ctx: &ExecutionContext,
) -> Result<Vec<RefutationReport>, ValidationError> {
    let estimator = TemporalMediationEstimator::new()
        .with_allow_natural_controlled_alias(true)
        .with_allow_gaps(gaps);
    let tabular = TabularData::new(data.storage().clone());
    if estimand.mediators.is_empty() {
        return Err(ValidationError::NotApplicable {
            message: "mediation validation requires an identified mediator",
        });
    }
    let mediators = Arc::clone(&estimand.mediators);
    let mut placebo_query = query.clone();
    placebo_query.contrast = MediationContrast::Mediated;
    // The three refuters share one noise vector per replicate, so one pass drives all of them.
    let rows = tabular.row_count();
    let per_replicate = map_replicates(
        ctx,
        STABILITY_REPLICATES,
        || vec![0.0; rows],
        |noise, ctx, replicate| {
            let replicate = u64::from(replicate);
            fill_gaussian(noise, ctx, 0xF014_0000 + replicate);
            // Every mediator is replaced by noise; the first keeps the stream the single
            // mediator used, further ones draw their own.
            let mut replaced =
                with_replaced_float(&tabular, mediators[0], Arc::from(noise.clone()))?;
            for (j, &mediator) in mediators.iter().enumerate().skip(1) {
                let mut extra_noise = vec![0.0; rows];
                fill_gaussian(&mut extra_noise, ctx, 0xF014_8000 + replicate * 100 + j as u64);
                replaced = with_replaced_float(&replaced, mediator, Arc::from(extra_noise))?;
            }
            let series =
                TimeSeriesData::try_new(replaced.storage().clone(), data.time_index().clone())?;
            let placebo_ate = estimator
                .estimate_with_adjustment(&series, estimand, &placebo_query, adjustment, &[], ctx)?
                .effect
                .ate;
            let (augmented, id) =
                with_extra_float(&tabular, "__mediation_rcc", Arc::from(noise.clone()))?;
            let series =
                TimeSeriesData::try_new(augmented.storage().clone(), data.time_index().clone())?;
            let rcc_ate = estimator
                .estimate_with_adjustment(&series, estimand, query, adjustment, &[id], ctx)?
                .effect
                .ate;
            let subset_ate = if full {
                let window =
                    with_contiguous_row_window(&tabular, 0.8, ctx, 0xF015_0000 + replicate)?;
                let index = TimeIndex {
                    regularity: data.time_index().regularity.clone(),
                    length: window.row_count(),
                };
                let series = TimeSeriesData::try_new(window.storage().clone(), index)?;
                Some(
                    estimator
                        .estimate_with_adjustment(&series, estimand, query, adjustment, &[], ctx)?
                        .effect
                        .ate,
                )
            } else {
                None
            };
            Ok((placebo_ate, rcc_ate, subset_ate))
        },
    )?;
    let placebo: Vec<f64> = per_replicate.iter().map(|r| r.0).collect();
    let rcc: Vec<f64> = per_replicate.iter().map(|r| r.1).collect();
    let subset: Vec<f64> = per_replicate.iter().filter_map(|r| r.2).collect();
    let mut reports = vec![
        report("mediation.placebo_mediator", 0.0, &placebo)?,
        report("mediation.random_common_cause", original.effect.ate, &rcc)?,
    ];
    if full {
        reports.push(report("mediation.contiguous_window", original.effect.ate, &subset)?);
    }
    // Binary treatment permits a direct empirical mediator-support check. This
    // is a necessary range diagnostic, not a proof of conditional positivity.
    let treatment_lag = query.horizons.first().copied().filter(|&h| h >= 1).unwrap_or(1);
    let mut columns = vec![LaggedColumn {
        variable: query.treatment,
        lag: antecedent_core::Lag::from_raw(treatment_lag),
    }];
    columns.extend(
        mediators
            .iter()
            .map(|&variable| LaggedColumn { variable, lag: antecedent_core::Lag::CONTEMPORANEOUS }),
    );
    columns
        .push(LaggedColumn { variable: query.outcome, lag: antecedent_core::Lag::CONTEMPORANEOUS });
    columns.extend_from_slice(adjustment);
    let max_lag = columns.iter().map(|c| c.lag.raw()).max().unwrap_or(1);
    let mut scan = antecedent_estimate::HistoryScan::plan(data, max_lag, Arc::from(columns))?;
    let (sample, _) = scan.windows(data, &ctx.kernel_policy, gaps)?;
    let t = sample.column(0);
    let mut levels = t.to_vec();
    levels.sort_by(f64::total_cmp);
    levels.dedup();
    if levels.len() == 2 {
        // The shared fraction of each mediator's range; a set reports its least-shared mediator.
        let mut fraction = f64::INFINITY;
        // Closed ranges that touch at one value still intersect (the static suite's rule), so the
        // verdict is the intersection of every mediator's per-arm ranges, not a positive shared
        // fraction.
        let mut intersect = true;
        for j in 0..mediators.len() {
            let mut ranges = [(f64::INFINITY, f64::NEG_INFINITY); 2];
            for (&t, &m) in t.iter().zip(sample.column(1 + j)) {
                let group = usize::from(t == levels[1]);
                ranges[group].0 = ranges[group].0.min(m);
                ranges[group].1 = ranges[group].1.max(m);
            }
            let overlap = ranges[0].1.min(ranges[1].1) - ranges[0].0.max(ranges[1].0);
            intersect &= overlap >= 0.0;
            let shared = overlap.max(0.0);
            let width = ranges[0].1.max(ranges[1].1) - ranges[0].0.min(ranges[1].0);
            fraction = fraction.min(if width > 0.0 { shared / width } else { 0.0 });
        }
        // A range diagnostic, not a re-estimate: the effect columns carry the published
        // contrast, and `comparison` is the shared fraction of the mediator range.
        reports.push(
            RefutationReport::builder("mediation.binary_mediator_support")
                .original_ate(original.effect.ate)
                .refuted_ate(original.effect.ate)
                .comparison(fraction)
                .informative(true)
                .passed(intersect)
                .failure_condition((!intersect).then(|| {
                    Arc::from("binary treatment groups have disjoint empirical mediator ranges")
                }))
                .replicates(1)
                .build(),
        );
    }

    Ok(reports)
}

fn report(id: &str, target: f64, values: &[f64]) -> Result<RefutationReport, ValidationError> {
    // Linear mediation placebo/RCC/subset use the same OLS-orthogonal constructions
    // as the ATE refuters: they cannot falsify the claim, so they are not informative.
    finish_p_report(id, target, values, target, STABILITY_ALPHA, |_, _| {
        "mediation contrast is inconsistent with the refuter target".to_owned()
    })
}

/// Static mediation-native suite. Cheap tests a placebo mediator, an exogenous
/// random nuisance, and binary mediator range overlap. Full adds random subsets.
/// All numeric refits use the same static DAG estimator as the licensed cell.
///
/// # Errors
/// Invalid refit data, regression failure, or cancellation.
#[allow(clippy::float_cmp, reason = "Exact binary treatment levels define the two groups")]
pub fn refute_static_mediation(
    data: &TabularData,
    graph: &antecedent_graph::Dag,
    query: &MediationQuery,
    original: &TemporalMediationEstimate,
    full: bool,
    ctx: &ExecutionContext,
) -> Result<Vec<RefutationReport>, ValidationError> {
    use antecedent_estimate::estimate_static_mediation;
    let mut placebo_query = query.clone();
    placebo_query.contrast = MediationContrast::NaturalIndirect;
    let per_replicate = map_replicates(
        ctx,
        STABILITY_REPLICATES,
        || (),
        |(), ctx, rep| {
            let rep = u64::from(rep);
            let mut replaced = data.clone();
            let mut noise = vec![0.0; data.row_count()];
            for (j, &mediator) in query.mediators.iter().enumerate() {
                fill_gaussian(&mut noise, ctx, 0x1300_2000 + rep * 100 + j as u64);
                replaced = with_replaced_float(&replaced, mediator, Arc::from(noise.clone()))?;
            }
            let fit =
                |d: &TabularData, q: &MediationQuery, extra: &[antecedent_core::VariableId]| {
                    estimate_static_mediation(
                        d,
                        graph,
                        q,
                        original.effect.provenance.assumptions.clone(),
                        0,
                        extra,
                        ctx,
                    )
                };
            let placebo_ate = fit(&replaced, &placebo_query, &[])?.effect.ate;
            fill_gaussian(&mut noise, ctx, 0x1300_4000 + rep);
            let (augmented, id) = with_extra_float(data, "__mediation_rcc", Arc::from(noise))?;
            let rcc_ate = fit(&augmented, query, &[id])?.effect.ate;
            let subset_ate = if full {
                let sub = crate::common::with_row_subset(data, 0.8, ctx, 0x1300_5000 + rep)?;
                Some(fit(&sub, query, &[])?.effect.ate)
            } else {
                None
            };
            Ok((placebo_ate, rcc_ate, subset_ate))
        },
    )?;
    let placebo: Vec<f64> = per_replicate.iter().map(|r| r.0).collect();
    let rcc: Vec<f64> = per_replicate.iter().map(|r| r.1).collect();
    let subset: Vec<f64> = per_replicate.iter().filter_map(|r| r.2).collect();
    let mut reports = vec![
        report("mediation.static.placebo_mediator", 0.0, &placebo)?,
        report("mediation.static.random_common_cause", original.effect.ate, &rcc)?,
    ];
    if full {
        reports.push(report("mediation.static.subset", original.effect.ate, &subset)?);
    }
    let treatment = data.float64_values(query.treatment)?;
    let mut levels: Vec<_> = treatment.iter().copied().filter(|x| x.is_finite()).collect();
    levels.sort_by(f64::total_cmp);
    levels.dedup();
    if levels.len() == 2 {
        let mut overlap = true;
        for &m in query.mediators.iter() {
            let mediator = data.float64_values(m)?;
            let mut ranges = [(f64::INFINITY, f64::NEG_INFINITY); 2];
            for (&t, &m) in treatment.iter().zip(&mediator) {
                if !t.is_finite() || !m.is_finite() {
                    continue;
                }
                let g = usize::from(t == levels[1]);
                ranges[g].0 = ranges[g].0.min(m);
                ranges[g].1 = ranges[g].1.max(m);
            }
            overlap &= ranges[0].0.max(ranges[1].0) <= ranges[0].1.min(ranges[1].1);
        }
        // A range diagnostic, not a re-estimate: `comparison` is 1 when every mediator's
        // per-arm empirical ranges overlap and 0 otherwise.
        reports.push(
            RefutationReport::builder("mediation.static.mediator_overlap")
                .original_ate(original.effect.ate)
                .refuted_ate(original.effect.ate)
                .comparison(f64::from(overlap))
                .informative(false)
                .passed(overlap)
                .failure_condition((!overlap).then(|| {
                    Arc::from(
                        "the treatment arms have disjoint empirical mediator ranges (a necessary \
                         condition for conditional positivity fails)",
                    )
                }))
                .replicates(0)
                .build(),
        );
    }
    Ok(reports)
}
