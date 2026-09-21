//! Refuters for the linear temporal mediation contrast.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use antecedent_core::{ExecutionContext, MediationContrast, MediationQuery};
use antecedent_data::{
    LaggedColumn, LaggedSampleWorkspace, TableView, TabularData, TimeIndex, TimeSeriesData,
};
use antecedent_estimate::{TemporalMediationEstimate, TemporalMediationEstimator};
use antecedent_expr::IdentifiedEstimand;

use crate::ValidationError;
use crate::common::{
    RefutationReport, fill_gaussian, replicate_mean, replicate_p_value, with_contiguous_row_window,
    with_extra_float, with_replaced_float,
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

/// Run placebo-mediator and random-common-cause checks on the mediation model.
/// Full adds a contiguous-window contrast stability check, preserving lag adjacency.
/// Placebo always tests the indirect contrast against zero, including for a
/// direct/total query; RCC and window checks test the requested contrast.
#[allow(clippy::float_cmp)] // Exact equality distinguishes observed binary levels.
pub fn refute_temporal_mediation(
    data: &TimeSeriesData,
    estimand: &IdentifiedEstimand,
    query: &MediationQuery,
    original: &TemporalMediationEstimate,
    full: bool,
    ctx: &ExecutionContext,
) -> Result<Vec<RefutationReport>, ValidationError> {
    refute_temporal_mediation_adjusted(data, estimand, query, original, full, &[], ctx)
}

/// Run the native suite with the same graph-derived adjustment as the estimate.
#[allow(clippy::float_cmp, clippy::too_many_arguments)]
pub fn refute_temporal_mediation_adjusted(
    data: &TimeSeriesData,
    estimand: &IdentifiedEstimand,
    query: &MediationQuery,
    original: &TemporalMediationEstimate,
    full: bool,
    adjustment: &[LaggedColumn],
    ctx: &ExecutionContext,
) -> Result<Vec<RefutationReport>, ValidationError> {
    let estimator = TemporalMediationEstimator::new().with_allow_natural_controlled_alias(true);
    let tabular = TabularData::new(data.storage().clone());
    let mediator = *estimand.mediators.first().ok_or(ValidationError::NotApplicable {
        message: "mediation validation requires an identified mediator",
    })?;
    let mut placebo_query = query.clone();
    placebo_query.contrast = MediationContrast::Mediated;
    let mut noise = vec![0.0; tabular.row_count()];
    let mut placebo = Vec::new();
    let mut rcc = Vec::new();
    let mut subset = Vec::new();
    for replicate in 0..MEDIATION_REPLICATES {
        if ctx.cancellation.is_cancelled() {
            return Err(ValidationError::Cancelled);
        }
        fill_gaussian(&mut noise, ctx, 0xF014_0000 + replicate);
        let replaced = with_replaced_float(&tabular, mediator, Arc::from(noise.clone()))?;
        let series =
            TimeSeriesData::try_new(replaced.storage().clone(), data.time_index().clone())?;
        placebo.push(
            estimator
                .estimate_with_adjustment(&series, estimand, &placebo_query, adjustment, &[], ctx)?
                .effect
                .ate,
        );
        let (augmented, id) =
            with_extra_float(&tabular, "__mediation_rcc", Arc::from(noise.clone()))?;
        let series =
            TimeSeriesData::try_new(augmented.storage().clone(), data.time_index().clone())?;
        rcc.push(
            estimator
                .estimate_with_adjustment(&series, estimand, query, adjustment, &[id], ctx)?
                .effect
                .ate,
        );
        if full {
            let window = with_contiguous_row_window(&tabular, 0.8, ctx, 0xF015_0000 + replicate)?;
            let index = TimeIndex {
                regularity: data.time_index().regularity.clone(),
                length: window.row_count(),
            };
            let series = TimeSeriesData::try_new(window.storage().clone(), index)?;
            subset.push(
                estimator
                    .estimate_with_adjustment(&series, estimand, query, adjustment, &[], ctx)?
                    .effect
                    .ate,
            );
        }
    }
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
    let mut columns = vec![
        LaggedColumn {
            variable: query.treatment,
            lag: antecedent_core::Lag::from_raw(treatment_lag),
        },
        LaggedColumn { variable: mediator, lag: antecedent_core::Lag::CONTEMPORANEOUS },
        LaggedColumn { variable: query.outcome, lag: antecedent_core::Lag::CONTEMPORANEOUS },
    ];
    columns.extend_from_slice(adjustment);
    let max_lag = columns.iter().map(|c| c.lag.raw()).max().unwrap_or(1);
    let plan = data.plan_lagged_sample(max_lag, Arc::from(columns))?;
    let mut workspace = LaggedSampleWorkspace::default();
    let sample = plan.prepare(data, &mut workspace, &ctx.kernel_policy)?;
    let t = sample.column(0);
    let mut levels = t.to_vec();
    levels.sort_by(f64::total_cmp);
    levels.dedup();
    if levels.len() == 2 {
        let mut ranges = [(f64::INFINITY, f64::NEG_INFINITY); 2];
        for (&t, &m) in t.iter().zip(sample.column(1)) {
            let group = usize::from(t == levels[1]);
            ranges[group].0 = ranges[group].0.min(m);
            ranges[group].1 = ranges[group].1.max(m);
        }
        let shared = (ranges[0].1.min(ranges[1].1) - ranges[0].0.max(ranges[1].0)).max(0.0);
        let width = ranges[0].1.max(ranges[1].1) - ranges[0].0.min(ranges[1].0);
        let fraction = if width > 0.0 { shared / width } else { 0.0 };
        // A range diagnostic, not a re-estimate: the effect columns carry the published
        // contrast, and `comparison` is the shared fraction of the mediator range.
        reports.push(RefutationReport::new(
            "mediation.binary_mediator_support",
            original.effect.ate,
            original.effect.ate,
            fraction,
            true,
            fraction > 0.0,
            (fraction <= 0.0).then(|| {
                Arc::from("binary treatment groups have disjoint empirical mediator ranges")
            }),
            1,
        ));
    }

    Ok(reports)
}

/// Replicates per mediation refuter (placebo, random common cause, window / subset).
const MEDIATION_REPLICATES: u64 = 20;

fn report(id: &str, target: f64, values: &[f64]) -> Result<RefutationReport, ValidationError> {
    let p = replicate_p_value(values, target)?;
    // Linear mediation placebo/RCC/subset use the same OLS-orthogonal constructions
    // as the ATE refuters: they cannot falsify the claim, so they are not informative.
    Ok(RefutationReport::new(
        id,
        target,
        replicate_mean(values),
        p,
        false,
        p >= 0.05,
        (p < 0.05).then(|| Arc::from("mediation contrast is inconsistent with the refuter target")),
        u32::try_from(values.len()).unwrap_or(u32::MAX),
    ))
}

/// Static mediation-native suite. Cheap tests a placebo mediator, an exogenous
/// random nuisance, and binary mediator range overlap. Full adds random subsets.
/// All numeric refits use the same static DAG estimator as the licensed cell.
///
/// # Errors
/// Invalid refit data, regression failure, or cancellation.
#[allow(clippy::float_cmp)] // Exact binary treatment levels define the two groups.
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
    let mut placebo = Vec::new();
    let mut rcc = Vec::new();
    let mut subset = Vec::new();
    for rep in 0..MEDIATION_REPLICATES {
        if ctx.cancellation.is_cancelled() {
            return Err(ValidationError::Cancelled);
        }
        let mut replaced = data.clone();
        let mut noise = vec![0.0; data.row_count()];
        for (j, &mediator) in query.mediators.iter().enumerate() {
            fill_gaussian(&mut noise, ctx, 0x1300_2000 + rep * 100 + j as u64);
            replaced = with_replaced_float(&replaced, mediator, Arc::from(noise.clone()))?;
        }
        let fit = |d: &TabularData, q: &MediationQuery, extra: &[antecedent_core::VariableId]| {
            estimate_static_mediation(
                d,
                graph,
                q,
                original.effect.assumptions.clone(),
                0,
                extra,
                ctx,
            )
        };
        placebo.push(fit(&replaced, &placebo_query, &[])?.effect.ate);
        fill_gaussian(&mut noise, ctx, 0x1300_4000 + rep);
        let (augmented, id) = with_extra_float(data, "__mediation_rcc", Arc::from(noise))?;
        rcc.push(fit(&augmented, query, &[id])?.effect.ate);
        if full {
            let sub = crate::common::with_row_subset(data, 0.8, ctx, 0x1300_5000 + rep)?;
            subset.push(fit(&sub, query, &[])?.effect.ate);
        }
    }
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
        reports.push(RefutationReport::new(
            "mediation.static.mediator_overlap",
            original.effect.ate,
            original.effect.ate,
            f64::from(overlap),
            false,
            overlap,
            (!overlap).then(|| {
                Arc::from(
                    "the treatment arms have disjoint empirical mediator ranges (a necessary \
                     condition for conditional positivity fails)",
                )
            }),
            0,
        ));
    }
    Ok(reports)
}
