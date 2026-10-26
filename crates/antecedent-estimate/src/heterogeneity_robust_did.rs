//! Heterogeneity-robust difference in differences for staggered rollouts.
//!
//! **Estimand.** The group-time average effect on the treated `ATT(g, t)`: the mean effect at
//! calendar period `t` for the units first treated at period `g` (cohort `g`; cohort zero is
//! never treated), and its aggregations: the simple cohort-size-weighted average over
//! post-adoption cells, the event-time (dynamic) profile, the per-cohort average and the
//! per-calendar-period average (Callaway and Sant'Anna 2021).
//!
//! **Why not two-way fixed effects.** With effects that differ across cohorts or grow with
//! exposure, the static two-way fixed-effects coefficient is a weighted average of cell
//! effects with weights that can be negative, because already-treated units serve as
//! controls (Goodman-Bacon 2021; de Chaisemartin and D'Haultfœuille 2020). Every estimator
//! here compares a cohort only with units not yet treated (or never treated) at both periods
//! of its comparison, so no cell borrows an already-treated control. The static two-way
//! coefficient is still computed and reported as a diagnostic.
//!
//! **Estimators.**
//! - Callaway–Sant'Anna: each `ATT(g, t)` is a 2×2 comparison of cohort `g` with the
//!   declared comparison group (never treated, or not yet treated by `max(t, base) + δ`)
//!   between `t` and a base period. With baseline covariates the 2×2 uses outcome
//!   regression, inverse probability weighting (normalized odds weights of a logistic
//!   propensity), or the doubly robust combination (Sant'Anna and Zhao 2020).
//! - Sun–Abraham interaction weighting: cohort-by-event-time effects against the never-treated
//!   cohort (or the last-treated cohort before its adoption when no unit is never treated)
//!   relative to the period before adoption, aggregated with cohort shares. On a balanced
//!   panel the saturated interacted two-way regression gives exactly these 2×2 contrasts,
//!   which is how they are computed; an unbalanced panel is refused.
//! - Borusyak–Jaravel–Spiess imputation: unit and period effects are fitted by weighted
//!   alternating projections on untreated observations only (never-treated units and periods
//!   before adoption minus anticipation), the untreated outcome of every treated observation
//!   is imputed, and `ATT(g, t)` averages the imputed effects.
//!
//! **Anticipation** of `δ` periods moves the base period to `g − 1 − δ` and keeps the `δ`
//! periods before adoption out of the untreated sample and out of every aggregation.
//! Pre-adoption contrasts (event time below `−δ`) are reported as pre-trend falsification
//! diagnostics for every estimator; they are never part of an aggregation.
//!
//! **Unbalanced panels** are read as they are: each 2×2 uses the units observed at both of its
//! periods and the imputation uses every observed untreated row.
//!
//! **Uncertainty** is a whole-procedure resample ([`crate::rollout_resampling`]): whole
//! clusters are redrawn (or reweighted by exponential multipliers), every cell, nuisance fit
//! and aggregation is refit, and the interquartile bootstrap standard error gives pointwise
//! normal intervals and a sup-t simultaneous band within each block (aggregates, group-time
//! cells, pre-trend leads). The coverage of these intervals has not been measured.
// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
#[path = "heterogeneity_robust_did/tests.rs"]
mod tests;

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use antecedent_core::{
    DidAggregation, DidBasePeriod, DidComparisonGroup, DidNuisanceMethod, ExecutionContext,
    RolloutDidEstimator,
};

use crate::EstimationError;
use crate::util::{mean, sigmoid as logistic};
use crate::rollout_resampling::{
    DesignEffect, DesignEffectKind, ReplicateSummary, UnitWeighting, block_draws, resample_draws,
    summarize_posterior, summarize_replicates, unit_weights,
};

/// Semantic operation tag of the rollout difference-in-differences resample.
const RESAMPLE_OPERATION: u64 = 0x5244_4944_0001;

/// A control probability at or above this is an overlap failure of the inverse-probability
/// and doubly robust 2×2 comparisons.
const OVERLAP_CEILING: f64 = 0.995;

/// Iteration cap and tolerance of the weighted two-way fixed-effects projections.
const PROJECTION_ITERATIONS: usize = 20_000;
const PROJECTION_TOLERANCE: f64 = 1e-11;

/// The row-aligned long panel of a staggered rollout.
#[derive(Clone, Copy, Debug)]
pub struct RolloutPanel<'a> {
    /// Outcome of each row.
    pub outcome: &'a [f64],
    /// Rows the analysis may read (complete outcome and covariates, kept by the declared
    /// missing-data policy).
    pub kept: &'a [bool],
    /// Unit of each row.
    pub units: &'a [Arc<str>],
    /// Inference cluster of each row; constant within unit.
    pub clusters: &'a [Arc<str>],
    /// Calendar period of each row.
    pub periods: &'a [i64],
    /// First treated period of each row's unit; zero for never treated.
    pub cohorts: &'a [i64],
    /// Baseline covariate columns, each row-aligned; read at each unit's first kept row.
    pub covariates: &'a [Vec<f64>],
}

/// The declared estimator of a rollout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RolloutDidSpec {
    /// Estimator family.
    pub estimator: RolloutDidEstimator,
    /// Units the treated cohort is compared with.
    pub comparison: DidComparisonGroup,
    /// How baseline covariates enter a 2×2 comparison.
    pub nuisance: DidNuisanceMethod,
    /// Base period of pre-adoption contrasts.
    pub base_period: DidBasePeriod,
    /// Aggregation that defines the scalar summary.
    pub aggregation: DidAggregation,
    /// Periods of anticipation before adoption.
    pub anticipation: i64,
}

/// The whole-procedure resample to run with the estimate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RolloutResampling {
    /// How each replicate reweights the clusters.
    pub weighting: UnitWeighting,
    /// Replicates.
    pub replicates: u32,
}

/// A fitted rollout: the scalar summary, its aggregation, every group-time cell and the
/// pre-trend leads, each with the uncertainty of the resample when one was run.
#[derive(Clone, Debug, PartialEq)]
pub struct RolloutDidFit {
    /// Scalar summary of the declared aggregation.
    pub overall: DesignEffect,
    /// Components of the declared aggregation (event times, cohorts or calendar periods; empty
    /// for the simple aggregation).
    pub aggregated: Vec<DesignEffect>,
    /// Post-adoption and anticipation-window group-time effects.
    pub group_time: Vec<DesignEffect>,
    /// Pre-adoption event-time contrasts (falsification diagnostics).
    pub leads: Vec<DesignEffect>,
    /// Largest `|lead / se|` over the leads with a standard error.
    pub pretrend_max_abs_t: Option<f64>,
    /// Whether the simultaneous band of the leads covers zero at every lead.
    pub pretrend_band_covers_zero: Option<bool>,
    /// Static two-way fixed-effects coefficient on the post-adoption indicator (diagnostic).
    pub twfe_static: Option<f64>,
    /// Units read.
    pub units: usize,
    /// Units in a treated cohort.
    pub treated_units: usize,
    /// Never-treated units.
    pub never_treated_units: usize,
    /// Distinct inference clusters.
    pub clusters: usize,
    /// Rows read.
    pub rows: usize,
    /// Whether every unit is observed in every period.
    pub balanced: bool,
    /// Treated observations the imputation could not impute (unit or period effect not
    /// identified from untreated rows); zero for the other estimators.
    pub unimputed_treated_observations: usize,
    /// Cohorts with no estimable post-adoption cell (adopted before any base period).
    pub dropped_cohorts: Vec<i64>,
    /// Resample record, when one was run.
    pub resampling: Option<RolloutResampleRecord>,
}

/// Counts and critical value of the resample behind a fit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RolloutResampleRecord {
    /// Unit weighting of the replicates.
    pub weighting: UnitWeighting,
    /// Requested replicates.
    pub requested: u32,
    /// Replicates with a full finite aggregate vector.
    pub replicates_ok: u32,
    /// Replicates without one.
    pub replicates_failed: u32,
    /// Sup-t critical value of the aggregate band.
    pub critical_value: Option<f64>,
}

/// One unit of the prepared panel.
#[derive(Clone, Debug)]
struct PanelUnit {
    cohort: i64,
    cluster: usize,
    outcomes: BTreeMap<i64, f64>,
    covariates: Vec<f64>,
}

/// The panel read into units, with the period grid and the treated cohorts.
#[derive(Clone, Debug)]
pub struct PreparedRollout {
    units: Vec<PanelUnit>,
    periods: Vec<i64>,
    cohorts: Vec<i64>,
    clusters: usize,
    covariates: usize,
    balanced: bool,
    rows: usize,
}

/// Comparison set of one cell.
#[derive(Clone, Copy, Debug)]
enum Comparison {
    Never,
    NotYet,
    LastCohort(i64),
}

/// One estimated group-time cell.
#[derive(Clone, Copy, Debug)]
struct Cell {
    att: f64,
    treated: usize,
    comparison: usize,
}

/// Every quantity of one evaluation of the procedure.
#[derive(Clone, Debug, Default)]
struct Evaluation {
    overall: f64,
    aggregate: BTreeMap<i64, f64>,
    cells: BTreeMap<(i64, i64), Cell>,
    leads: BTreeMap<i64, f64>,
    unimputed: usize,
}

fn refused(message: impl Into<String>) -> EstimationError {
    EstimationError::refused("option_not_applicable", message)
}

fn invalid(message: impl Into<String>) -> EstimationError {
    EstimationError::refused("invalid_argument", message)
}

/// Read the kept rows of `panel` into units.
///
/// # Errors
///
/// Misaligned columns, a unit whose cohort or cluster changes, a repeated unit-period, a
/// negative cohort or non-positive period, a covariate missing at a unit's first kept row, no
/// treated cohort, or (for Sun–Abraham) an unbalanced panel.
#[allow(
    clippy::too_many_lines,
    reason = "one linear pass that validates the panel and gathers its units, each check refusing with its own message"
)]
pub fn prepare(
    panel: &RolloutPanel<'_>,
    spec: &RolloutDidSpec,
) -> Result<PreparedRollout, EstimationError> {
    let n = panel.outcome.len();
    let lengths = [
        panel.kept.len(),
        panel.units.len(),
        panel.clusters.len(),
        panel.periods.len(),
        panel.cohorts.len(),
    ];
    if lengths.iter().any(|len| *len != n)
        || panel.covariates.iter().any(|column| column.len() != n)
    {
        return Err(invalid("rollout DiD columns must be row-aligned"));
    }
    if spec.anticipation < 0 {
        return Err(invalid("anticipation must be a non-negative number of periods"));
    }
    if !panel.covariates.is_empty() && spec.estimator != RolloutDidEstimator::CallawaySantAnna {
        return Err(refused(
            "baseline covariates enter only the Callaway–Sant'Anna 2×2 comparisons (outcome \
             regression, inverse probability weighting or doubly robust); next attempt: the \
             callaway_santanna estimator with these covariates, or this estimator without them",
        ));
    }
    let mut cluster_ids: BTreeMap<&str, usize> = BTreeMap::new();
    for (cluster, kept) in panel.clusters.iter().zip(panel.kept) {
        if *kept {
            let next = cluster_ids.len();
            cluster_ids.entry(cluster.as_ref()).or_insert(next);
        }
    }
    // Clusters are numbered in label order so a resample does not depend on row order.
    for (index, value) in cluster_ids.values_mut().enumerate() {
        *value = index;
    }
    let mut units: BTreeMap<&str, PanelUnit> = BTreeMap::new();
    let mut first_row: BTreeMap<&str, (i64, usize)> = BTreeMap::new();
    let mut grid = BTreeSet::new();
    let mut rows = 0usize;
    for row in 0..n {
        if !panel.kept[row] {
            continue;
        }
        let (period, cohort) = (panel.periods[row], panel.cohorts[row]);
        if period <= 0 || cohort < 0 {
            return Err(invalid("periods must be positive and cohorts non-negative"));
        }
        let value = panel.outcome[row];
        if !value.is_finite() {
            return Err(invalid("a kept rollout row has a non-finite outcome"));
        }
        let label = panel.units[row].as_ref();
        let cluster = cluster_ids[panel.clusters[row].as_ref()];
        let unit = units.entry(label).or_insert_with(|| PanelUnit {
            cohort,
            cluster,
            outcomes: BTreeMap::new(),
            covariates: Vec::new(),
        });
        if unit.cohort != cohort || unit.cluster != cluster {
            return Err(invalid("adoption cohort and cluster must be stable within unit"));
        }
        if unit.outcomes.insert(period, value).is_some() {
            return Err(invalid("each unit has at most one row per period"));
        }
        let first = first_row.entry(label).or_insert((period, row));
        if period < first.0 {
            *first = (period, row);
        }
        grid.insert(period);
        rows += 1;
    }
    for (label, (_, row)) in &first_row {
        let unit = units.get_mut(label).ok_or_else(|| invalid("unit vanished"))?;
        unit.covariates = panel.covariates.iter().map(|column| column[*row]).collect();
        if unit.covariates.iter().any(|value| !value.is_finite()) {
            return Err(invalid(
                "baseline covariates must be finite at each unit's first kept row",
            ));
        }
    }
    let periods: Vec<i64> = grid.into_iter().collect();
    let units: Vec<PanelUnit> = units.into_values().collect();
    let cohorts: Vec<i64> = units
        .iter()
        .map(|unit| unit.cohort)
        .filter(|cohort| *cohort > 0)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    if cohorts.is_empty() {
        return Err(invalid("a rollout needs at least one treated cohort"));
    }
    if periods.len() < 2 {
        return Err(invalid("a rollout needs at least two periods"));
    }
    let balanced = units.iter().all(|unit| unit.outcomes.len() == periods.len());
    if spec.estimator == RolloutDidEstimator::SunAbraham && !balanced {
        return Err(refused(
            "the Sun–Abraham interaction-weighted estimator reads the saturated interacted \
             two-way regression, which equals its 2×2 contrasts only on a balanced panel; next \
             attempt: declare missing=balanced_panel to keep complete units, or the \
             callaway_santanna or imputation estimator, which read unbalanced panels",
        ));
    }
    if spec.estimator == RolloutDidEstimator::SunAbraham
        && spec.comparison == DidComparisonGroup::NotYetTreated
    {
        return Err(refused(
            "the Sun–Abraham estimator compares every cohort with one fixed control cohort (never \
             treated, or the last-treated cohort before its adoption); next attempt: comparison \
             never_treated with Sun–Abraham, or the callaway_santanna estimator with not-yet-treated \
             comparisons",
        ));
    }
    Ok(PreparedRollout {
        units,
        periods,
        cohorts,
        clusters: cluster_ids.len(),
        covariates: panel.covariates.len(),
        balanced,
        rows,
    })
}

impl PreparedRollout {
    /// Units read.
    #[must_use]
    pub fn unit_count(&self) -> usize {
        self.units.len()
    }

    /// Whether every unit is observed in every period.
    #[must_use]
    pub const fn is_balanced(&self) -> bool {
        self.balanced
    }

    /// The comparison set the estimator uses.
    fn comparison(&self, spec: &RolloutDidSpec) -> Result<Comparison, EstimationError> {
        let never = self.units.iter().any(|unit| unit.cohort == 0);
        match (spec.estimator, spec.comparison) {
            (RolloutDidEstimator::SunAbraham, _) if never => Ok(Comparison::Never),
            (RolloutDidEstimator::SunAbraham, _) => {
                let last = *self.cohorts.last().ok_or_else(|| invalid("no treated cohort"))?;
                if self.cohorts.len() < 2 {
                    return Err(refused(
                        "Sun–Abraham with no never-treated unit uses the last-treated cohort as \
                         control, which leaves no other cohort to estimate; next attempt: add \
                         never-treated units",
                    ));
                }
                Ok(Comparison::LastCohort(last))
            }
            (_, DidComparisonGroup::NeverTreated) if !never => Err(refused(
                "the never-treated comparison group is empty; next attempt: comparison \
                 not_yet_treated, which compares each cohort with units adopting later",
            )),
            (_, DidComparisonGroup::NeverTreated) => Ok(Comparison::Never),
            (_, DidComparisonGroup::NotYetTreated) => Ok(Comparison::NotYet),
        }
    }
}

/// Estimate the rollout and, when `resampling` is given, its whole-procedure uncertainty.
///
/// # Errors
///
/// A refused panel (see [`prepare`]), an empty comparison group, no estimable post-adoption
/// cell, a nuisance fit that fails on the full sample, or a refused resample (cancellation,
/// admission).
pub fn estimate_rollout_did(
    panel: &RolloutPanel<'_>,
    spec: &RolloutDidSpec,
    resampling: Option<RolloutResampling>,
    ctx: &ExecutionContext,
) -> Result<RolloutDidFit, EstimationError> {
    if ctx.cancellation.is_cancelled() {
        return Err(EstimationError::unsupported("execution cancelled before the rollout fit"));
    }
    let prepared = prepare(panel, spec)?;
    let comparison = prepared.comparison(spec)?;
    let ones = vec![1.0; prepared.clusters];
    let point = evaluate(&prepared, spec, comparison, &ones)
        .map_err(|reason| refused(format!("the rollout cannot be estimated: {reason}")))?;
    if point.cells.keys().all(|(group, period)| period < group) {
        return Err(refused(
            "no post-adoption group-time cell is estimable (each needs treated units and \
             comparison units observed at the period and at its base period); next attempt: \
             comparison not_yet_treated, fewer anticipation periods, or a panel with a period \
             before each adoption",
        ));
    }
    let dropped_cohorts: Vec<i64> = prepared
        .cohorts
        .iter()
        .copied()
        .filter(|cohort| {
            !point.cells.keys().any(|(group, period)| group == cohort && period >= group)
        })
        .collect();
    let mut fit = assemble(&prepared, spec, &point, dropped_cohorts);
    if let Some(resampling) = resampling {
        let layout = Layout::of(&point, spec.anticipation);
        let width = layout.width();
        let draws = resample_draws(ctx, RESAMPLE_OPERATION, resampling.replicates, width, |rng| {
            let weights = unit_weights(resampling.weighting, prepared.clusters, rng);
            evaluate(&prepared, spec, comparison, &weights)
                .ok()
                .map(|replicate| layout.vector(&replicate))
        })?;
        attach(&mut fit, &layout, &layout.vector(&point), &draws, resampling);
    }
    Ok(fit)
}

/// Semantic operation tag of the Bayesian-bootstrap posterior.
const POSTERIOR_OPERATION: u64 = 0x5244_4944_0002;

/// A Bayesian-bootstrap posterior of a rollout.
#[derive(Clone, Debug, PartialEq)]
pub struct RolloutDidPosterior {
    /// Every reported effect at its posterior mean, with equal-tailed credible intervals and
    /// simultaneous credible bands within each block (aggregates, group-time cells, leads).
    pub fit: RolloutDidFit,
    /// Named joint posterior columns, the overall summary first, in the order of the fit's
    /// overall, aggregated, group-time and lead effects.
    pub columns: Vec<(String, Vec<f64>)>,
    /// Requested draws.
    pub requested: u32,
    /// Draws whose refit produced every quantity.
    pub draws_ok: u32,
}

/// Bayesian-bootstrap posterior of a rollout: every draw puts flat-Dirichlet weights on the
/// independent sampling units (whole clusters, or whole subjects under independent subject
/// trajectories), refits every group-time cell, nuisance model and aggregation with those
/// weights, and keeps the whole vector, so aggregation uncertainty and the covariance across
/// cells come from the same joint draws. The posterior is the nonparametric posterior of the
/// estimand under a Dirichlet-process model of the unit distribution with a vanishing
/// concentration (Rubin 1981); the design's identifying assumptions stay declared.
///
/// # Errors
///
/// As [`estimate_rollout_did`], and fewer than 20 complete draws.
pub fn posterior_rollout_did(
    panel: &RolloutPanel<'_>,
    spec: &RolloutDidSpec,
    draws: u32,
    ctx: &ExecutionContext,
) -> Result<RolloutDidPosterior, EstimationError> {
    let mut fit = estimate_rollout_did(panel, spec, None, ctx)?;
    let prepared = prepare(panel, spec)?;
    let comparison = prepared.comparison(spec)?;
    let ones = vec![1.0; prepared.clusters];
    let point = evaluate(&prepared, spec, comparison, &ones)
        .map_err(|reason| refused(format!("the rollout cannot be estimated: {reason}")))?;
    let layout = Layout::of(&point, spec.anticipation);
    let width = layout.width();
    let raw = resample_draws(ctx, POSTERIOR_OPERATION, draws, width, |rng| {
        // Normalized exponential weights are flat-Dirichlet; every estimator here is
        // invariant to the weights' scale, so the normalization is implicit.
        let weights = unit_weights(UnitWeighting::Multiplier, prepared.clusters, rng);
        evaluate(&prepared, spec, comparison, &weights)
            .ok()
            .map(|replicate| layout.vector(&replicate))
    })?;
    let a = 1 + layout.aggregate.len();
    let c = a + layout.cells.len();
    let aggregate = summarize_posterior(&block_draws(&raw, 0..a), a, draws);
    if aggregate.draws_ok < 20 {
        return Err(refused(
            "fewer than 20 Bayesian-bootstrap draws produced every aggregate; next attempt: more \
             draws, or a comparison group populated in every cell",
        ));
    }
    fit.overall.attach_posterior(aggregate.quantities[0].as_ref());
    fit.overall.band = None;
    for (effect, quantity) in fit.aggregated.iter_mut().zip(&aggregate.quantities[1..]) {
        effect.attach_posterior(quantity.as_ref());
    }
    let cells = summarize_posterior(&block_draws(&raw, a..c), c - a, draws);
    for (effect, quantity) in fit.group_time.iter_mut().zip(&cells.quantities) {
        effect.attach_posterior(quantity.as_ref());
    }
    if c < width {
        let leads = summarize_posterior(&block_draws(&raw, c..width), width - c, draws);
        for (effect, quantity) in fit.leads.iter_mut().zip(&leads.quantities) {
            effect.attach_posterior(quantity.as_ref());
        }
        fit.pretrend_max_abs_t = leads
            .quantities
            .iter()
            .flatten()
            .filter(|quantity| quantity.sd > 0.0)
            .map(|quantity| (quantity.mean / quantity.sd).abs())
            .reduce(f64::max);
        fit.pretrend_band_covers_zero = leads.critical_value.map(|_| {
            fit.leads
                .iter()
                .all(|lead| lead.band.is_none_or(|band| band[0] <= 0.0 && band[1] >= 0.0))
        });
    }
    let complete: Vec<&Vec<f64>> =
        raw.iter().flatten().filter(|draw| draw.iter().all(|value| value.is_finite())).collect();
    let mut labels = vec!["att".to_string()];
    labels
        .extend(layout.aggregate.iter().map(|key| format!("{}_{key}", spec.aggregation.as_str())));
    labels.extend(layout.cells.iter().map(|(group, period)| format!("att_g{group}_t{period}")));
    labels.extend(layout.leads.iter().map(|event| format!("lead_e{event}")));
    let columns = labels
        .into_iter()
        .enumerate()
        .map(|(k, label)| {
            let column = complete.iter().map(|draw| draw[k]).collect();
            (label, column)
        })
        .collect();
    fit.resampling = Some(RolloutResampleRecord {
        weighting: UnitWeighting::Multiplier,
        requested: draws,
        replicates_ok: aggregate.draws_ok,
        replicates_failed: aggregate.draws_failed,
        critical_value: aggregate.critical_value,
    });
    Ok(RolloutDidPosterior { fit, columns, requested: draws, draws_ok: aggregate.draws_ok })
}

/// The fixed order of every quantity of the point evaluation in a replicate vector.
struct Layout {
    aggregate: Vec<i64>,
    cells: Vec<(i64, i64)>,
    leads: Vec<i64>,
}

impl Layout {
    /// The reported cells are the post-adoption and anticipation-window cells, in key order,
    /// exactly as [`assemble`] lists them.
    fn of(point: &Evaluation, anticipation: i64) -> Self {
        Self {
            aggregate: point.aggregate.keys().copied().collect(),
            cells: point
                .cells
                .keys()
                .copied()
                .filter(|(group, period)| period - group >= -anticipation)
                .collect(),
            leads: point.leads.keys().copied().collect(),
        }
    }

    fn width(&self) -> usize {
        1 + self.aggregate.len() + self.cells.len() + self.leads.len()
    }

    fn vector(&self, evaluation: &Evaluation) -> Vec<f64> {
        let mut values = Vec::with_capacity(self.width());
        values.push(evaluation.overall);
        values.extend(
            self.aggregate
                .iter()
                .map(|key| evaluation.aggregate.get(key).copied().unwrap_or(f64::NAN)),
        );
        values.extend(
            self.cells
                .iter()
                .map(|key| evaluation.cells.get(key).map_or(f64::NAN, |cell| cell.att)),
        );
        values.extend(
            self.leads.iter().map(|key| evaluation.leads.get(key).copied().unwrap_or(f64::NAN)),
        );
        values
    }
}

/// Attach the block summaries of a resample to the fit.
fn attach(
    fit: &mut RolloutDidFit,
    layout: &Layout,
    point: &[f64],
    draws: &[Option<Vec<f64>>],
    resampling: RolloutResampling,
) {
    let a = 1 + layout.aggregate.len();
    let c = a + layout.cells.len();
    let w = layout.width();
    // The scalar summary and its aggregation components share one band.
    let aggregate: ReplicateSummary =
        summarize_replicates(&point[0..a], &block_draws(draws, 0..a), resampling.replicates);
    fit.overall.attach(&aggregate.quantities[0]);
    for (effect, quantity) in fit.aggregated.iter_mut().zip(&aggregate.quantities[1..]) {
        effect.attach(quantity);
    }
    let cells =
        summarize_replicates(&point[a..c], &block_draws(draws, a..c), resampling.replicates);
    for (effect, quantity) in fit.group_time.iter_mut().zip(&cells.quantities) {
        effect.attach(quantity);
    }
    if c < w {
        let leads =
            summarize_replicates(&point[c..w], &block_draws(draws, c..w), resampling.replicates);
        for (effect, quantity) in fit.leads.iter_mut().zip(&leads.quantities) {
            effect.attach(quantity);
        }
        fit.pretrend_max_abs_t = fit
            .leads
            .iter()
            .filter_map(|lead| lead.standard_error.map(|se| (lead.estimate / se).abs()))
            .reduce(f64::max);
        fit.pretrend_band_covers_zero = leads.critical_value.map(|_| {
            fit.leads
                .iter()
                .all(|lead| lead.band.is_none_or(|band| band[0] <= 0.0 && band[1] >= 0.0))
        });
    }
    fit.resampling = Some(RolloutResampleRecord {
        weighting: resampling.weighting,
        requested: aggregate.requested,
        replicates_ok: aggregate.replicates_ok,
        replicates_failed: aggregate.replicates_failed,
        critical_value: aggregate.critical_value,
    });
}

/// The point fit as reported effects, before any uncertainty is attached.
fn assemble(
    prepared: &PreparedRollout,
    spec: &RolloutDidSpec,
    point: &Evaluation,
    dropped_cohorts: Vec<i64>,
) -> RolloutDidFit {
    let aggregate_kind = match spec.aggregation {
        DidAggregation::Simple => None,
        DidAggregation::Dynamic => Some(DesignEffectKind::EventTime),
        DidAggregation::Group => Some(DesignEffectKind::Group),
        DidAggregation::Calendar => Some(DesignEffectKind::Calendar),
    };
    let aggregated = point
        .aggregate
        .iter()
        .filter_map(|(key, value)| {
            let kind = aggregate_kind?;
            let mut effect = DesignEffect::point(kind, *value);
            match kind {
                DesignEffectKind::EventTime => effect.event_time = Some(*key),
                DesignEffectKind::Group => effect.group = Some(*key),
                _ => effect.period = Some(*key),
            }
            Some(effect)
        })
        .collect();
    let group_time = point
        .cells
        .iter()
        .filter(|((group, period), _)| period - group >= -spec.anticipation)
        .map(|((group, period), cell)| DesignEffect {
            group: Some(*group),
            period: Some(*period),
            event_time: Some(period - group),
            treated: cell.treated,
            comparison: cell.comparison,
            ..DesignEffect::point(DesignEffectKind::GroupTime, cell.att)
        })
        .collect();
    let leads = point
        .leads
        .iter()
        .map(|(event_time, value)| DesignEffect {
            event_time: Some(*event_time),
            ..DesignEffect::point(DesignEffectKind::PreTrend, *value)
        })
        .collect();
    let treated_units = prepared.units.iter().filter(|unit| unit.cohort > 0).count();
    RolloutDidFit {
        overall: DesignEffect {
            treated: treated_units,
            comparison: prepared.units.len() - treated_units,
            ..DesignEffect::point(DesignEffectKind::Overall, point.overall)
        },
        aggregated,
        group_time,
        leads,
        pretrend_max_abs_t: None,
        pretrend_band_covers_zero: None,
        twfe_static: twfe_static(prepared),
        units: prepared.units.len(),
        treated_units,
        never_treated_units: prepared.units.iter().filter(|unit| unit.cohort == 0).count(),
        clusters: prepared.clusters,
        rows: prepared.rows,
        balanced: prepared.balanced,
        unimputed_treated_observations: point.unimputed,
        dropped_cohorts,
        resampling: None,
    }
}

/// One evaluation of the whole procedure under cluster weights `cluster_weights`.
fn evaluate(
    prepared: &PreparedRollout,
    spec: &RolloutDidSpec,
    comparison: Comparison,
    cluster_weights: &[f64],
) -> Result<Evaluation, &'static str> {
    let weights: Vec<f64> =
        prepared.units.iter().map(|unit| cluster_weights[unit.cluster]).collect();
    let delta = spec.anticipation;
    let (cells, unimputed) = group_time_cells(prepared, spec, comparison, &weights)?;
    let cohort_size: BTreeMap<i64, f64> = prepared
        .cohorts
        .iter()
        .map(|group| {
            let size = prepared
                .units
                .iter()
                .zip(&weights)
                .filter(|(unit, _)| unit.cohort == *group)
                .map(|(_, weight)| *weight)
                .sum::<f64>();
            (*group, size)
        })
        .collect();
    let weighted = |entries: &[(i64, f64)]| -> f64 {
        let total: f64 = entries.iter().map(|(group, _)| cohort_size[group]).sum();
        if total > 0.0 {
            entries.iter().map(|(group, value)| cohort_size[group] * value).sum::<f64>() / total
        } else {
            f64::NAN
        }
    };
    let post_cells: Vec<((i64, i64), f64)> = cells
        .iter()
        .filter(|((group, period), _)| period >= group)
        .map(|(key, cell)| (*key, cell.att))
        .collect();
    let mut by_event: BTreeMap<i64, Vec<(i64, f64)>> = BTreeMap::new();
    let mut leads_by_event: BTreeMap<i64, Vec<(i64, f64)>> = BTreeMap::new();
    for ((group, period), cell) in &cells {
        let event_time = period - group;
        if event_time >= -delta {
            by_event.entry(event_time).or_default().push((*group, cell.att));
        } else {
            leads_by_event.entry(event_time).or_default().push((*group, cell.att));
        }
    }
    let leads: BTreeMap<i64, f64> =
        leads_by_event.iter().map(|(event, entries)| (*event, weighted(entries))).collect();
    let (overall, aggregate) = match spec.aggregation {
        DidAggregation::Simple => {
            let entries: Vec<(i64, f64)> =
                post_cells.iter().map(|((group, _), value)| (*group, *value)).collect();
            (weighted(&entries), BTreeMap::new())
        }
        DidAggregation::Dynamic => {
            let profile: BTreeMap<i64, f64> =
                by_event.iter().map(|(event, entries)| (*event, weighted(entries))).collect();
            let post: Vec<f64> =
                profile.iter().filter(|(event, _)| **event >= 0).map(|(_, value)| *value).collect();
            (mean(&post), profile)
        }
        DidAggregation::Group => {
            let mut per_group: BTreeMap<i64, Vec<f64>> = BTreeMap::new();
            for ((group, _), value) in &post_cells {
                per_group.entry(*group).or_default().push(*value);
            }
            let profile: BTreeMap<i64, f64> =
                per_group.iter().map(|(group, values)| (*group, mean(values))).collect();
            let entries: Vec<(i64, f64)> = profile.iter().map(|(g, v)| (*g, *v)).collect();
            (weighted(&entries), profile)
        }
        DidAggregation::Calendar => {
            let mut per_period: BTreeMap<i64, Vec<(i64, f64)>> = BTreeMap::new();
            for ((group, period), value) in &post_cells {
                per_period.entry(*period).or_default().push((*group, *value));
            }
            let profile: BTreeMap<i64, f64> =
                per_period.iter().map(|(period, entries)| (*period, weighted(entries))).collect();
            let values: Vec<f64> = profile.values().copied().collect();
            (mean(&values), profile)
        }
    };
    Ok(Evaluation { overall, aggregate, cells, leads, unimputed })
}

/// Whether `unit` belongs to the comparison set of cohort `group` between `period` and `base`.
fn compares(
    unit: &PanelUnit,
    comparison: Comparison,
    group: i64,
    period: i64,
    base: i64,
    delta: i64,
) -> bool {
    match comparison {
        Comparison::Never => unit.cohort == 0,
        Comparison::NotYet => {
            unit.cohort == 0 || (unit.cohort != group && unit.cohort > period.max(base) + delta)
        }
        Comparison::LastCohort(last) => unit.cohort == last && period.max(base) < last - delta,
    }
}

/// One 2×2 comparison of cohort `group` between `period` and `base`.
///
/// `Ok(None)` when the cell has no weighted treated or comparison unit; `Err` when a nuisance
/// fit on a populated cell fails.
#[allow(
    clippy::too_many_arguments,
    reason = "the cell coordinates and the procedure's state are distinct inputs"
)]
fn two_by_two(
    prepared: &PreparedRollout,
    spec: &RolloutDidSpec,
    comparison: Comparison,
    weights: &[f64],
    group: i64,
    period: i64,
    base: i64,
) -> Result<Option<Cell>, &'static str> {
    if let Comparison::LastCohort(last) = comparison {
        if period.max(base) >= last - spec.anticipation {
            return Ok(None);
        }
    }
    let mut treated: Vec<(f64, &[f64], f64)> = Vec::new();
    let mut controls: Vec<(f64, &[f64], f64)> = Vec::new();
    for (unit, weight) in prepared.units.iter().zip(weights) {
        if *weight <= 0.0 {
            continue;
        }
        let (Some(now), Some(then)) = (unit.outcomes.get(&period), unit.outcomes.get(&base)) else {
            continue;
        };
        let change = now - then;
        if unit.cohort == group {
            treated.push((change, unit.covariates.as_slice(), *weight));
        } else if compares(unit, comparison, group, period, base, spec.anticipation) {
            controls.push((change, unit.covariates.as_slice(), *weight));
        }
    }
    if treated.is_empty() || controls.is_empty() {
        return Ok(None);
    }
    let counts = (treated.len(), controls.len());
    let att = if prepared.covariates == 0 {
        weighted_mean(treated.iter().map(|&(y, _, w)| (y, w)))
            - weighted_mean(controls.iter().map(|&(y, _, w)| (y, w)))
    } else {
        covariate_two_by_two(&treated, &controls, spec.nuisance)?
    };
    if !att.is_finite() {
        return Err("a group-time comparison overflowed finite precision");
    }
    Ok(Some(Cell { att, treated: counts.0, comparison: counts.1 }))
}

fn weighted_mean(values: impl Iterator<Item = (f64, f64)>) -> f64 {
    let (mut sum, mut total) = (0.0, 0.0);
    for (value, weight) in values {
        sum += weight * value;
        total += weight;
    }
    if total > 0.0 { sum / total } else { f64::NAN }
}

/// The covariate-adjusted 2×2: outcome regression, normalized inverse probability weighting,
/// or the doubly robust combination of the two.
fn covariate_two_by_two(
    treated: &[(f64, &[f64], f64)],
    controls: &[(f64, &[f64], f64)],
    nuisance: DidNuisanceMethod,
) -> Result<f64, &'static str> {
    let regression = if matches!(
        nuisance,
        DidNuisanceMethod::OutcomeRegression | DidNuisanceMethod::DoublyRobust
    ) {
        let rows: Vec<(Vec<f64>, f64, f64)> =
            controls.iter().map(|&(y, x, w)| (with_intercept(x), y, w)).collect();
        Some(weighted_least_squares(&rows).ok_or(
            "the untreated-change outcome regression is singular in a group-time comparison; \
             next attempt: fewer covariates or the inverse probability weighting 2×2",
        )?)
    } else {
        None
    };
    let predict = |x: &[f64]| -> f64 { regression.as_ref().map_or(0.0, |beta| linear(beta, x)) };
    if nuisance == DidNuisanceMethod::OutcomeRegression {
        return Ok(weighted_mean(treated.iter().map(|&(y, x, w)| (y - predict(x), w))));
    }
    let mut rows: Vec<(Vec<f64>, f64, f64)> = Vec::with_capacity(treated.len() + controls.len());
    rows.extend(treated.iter().map(|&(_, x, w)| (with_intercept(x), 1.0, w)));
    rows.extend(controls.iter().map(|&(_, x, w)| (with_intercept(x), 0.0, w)));
    let gamma = weighted_logistic(&rows).ok_or(
        "the propensity of the treated cohort does not converge in a group-time comparison \
         (separation or too few units); next attempt: fewer covariates or the outcome \
         regression 2×2",
    )?;
    let mut odds = Vec::with_capacity(controls.len());
    for &(_, x, w) in controls {
        let p = logistic(linear(&gamma, x));
        if p >= OVERLAP_CEILING {
            return Err(
                "a comparison unit has propensity above the overlap ceiling in a group-time \
                 comparison; next attempt: the outcome regression 2×2 or coarser covariates",
            );
        }
        odds.push(w * p / (1.0 - p));
    }
    let treated_term = weighted_mean(treated.iter().map(|&(y, x, w)| (y - predict(x), w)));
    let control_term =
        weighted_mean(controls.iter().zip(&odds).map(|(&(y, x, _), &odds)| (y - predict(x), odds)));
    Ok(treated_term - control_term)
}

/// `[1, x…]`.
fn with_intercept(covariates: &[f64]) -> Vec<f64> {
    let mut row = Vec::with_capacity(covariates.len() + 1);
    row.push(1.0);
    row.extend_from_slice(covariates);
    row
}

/// `β₀ + Σ βⱼ xⱼ` for coefficients that lead with an intercept.
fn linear(coefficients: &[f64], covariates: &[f64]) -> f64 {
    coefficients[0] + coefficients[1..].iter().zip(covariates).map(|(b, x)| b * x).sum::<f64>()
}

/// Weighted least squares `argmin Σ w (y − xβ)²`; `None` when `XᵀWX` is not positive definite.
pub(crate) fn weighted_least_squares(rows: &[(Vec<f64>, f64, f64)]) -> Option<Vec<f64>> {
    let p = rows.first()?.0.len();
    let mut gram = vec![0.0; p * p];
    let mut rhs = vec![0.0; p];
    for (x, y, w) in rows {
        for i in 0..p {
            rhs[i] += w * x[i] * y;
            for j in 0..p {
                gram[i * p + j] += w * x[i] * x[j];
            }
        }
    }
    crate::util::solve_spd(&gram, &rhs, p).filter(|beta| beta.iter().all(|value| value.is_finite()))
}

/// Weighted logistic regression of the `0/1` responses through the shared GLM engine; `None`
/// without convergence or under separation.
fn weighted_logistic(rows: &[(Vec<f64>, f64, f64)]) -> Option<Vec<f64>> {
    let p = rows.first()?.0.len();
    let n = rows.len();
    let mut design = vec![0.0; n * p];
    for (r, (x, _, _)) in rows.iter().enumerate() {
        for (c, &value) in x.iter().enumerate() {
            design[c * n + r] = value;
        }
    }
    let y: Vec<f64> = rows.iter().map(|&(_, d, _)| d).collect();
    let w: Vec<f64> = rows.iter().map(|&(_, _, w)| w).collect();
    crate::util::fit_logistic_weighted(&design, n, p, &y, &w).ok()
}

/// The group-time cells of one evaluation under unit weights `weights`, with the count of
/// treated observations the imputation estimator cannot impute.
#[allow(
    clippy::type_complexity,
    reason = "the cells and the unimputed count are returned together"
)]
fn group_time_cells(
    prepared: &PreparedRollout,
    spec: &RolloutDidSpec,
    comparison: Comparison,
    weights: &[f64],
) -> Result<(BTreeMap<(i64, i64), Cell>, usize), &'static str> {
    let delta = spec.anticipation;
    let mut cells: BTreeMap<(i64, i64), Cell> = BTreeMap::new();
    let mut unimputed = 0usize;
    // Pre-adoption contrasts come from the 2×2 comparisons for every estimator.
    let pre_base = match spec.estimator {
        RolloutDidEstimator::SunAbraham => DidBasePeriod::Universal,
        _ => spec.base_period,
    };
    for &group in &prepared.cohorts {
        if let Comparison::LastCohort(last) = comparison {
            if group == last {
                continue;
            }
        }
        for &period in &prepared.periods {
            let event_time = period - group;
            let post = event_time >= -delta;
            let base = if post || pre_base == DidBasePeriod::Universal {
                group - 1 - delta
            } else {
                match prepared.periods.iter().rev().find(|candidate| **candidate < period) {
                    Some(previous) => *previous,
                    None => continue,
                }
            };
            if base == period || !prepared.periods.contains(&base) {
                continue;
            }
            let cell = if post && spec.estimator == RolloutDidEstimator::Imputation {
                None
            } else {
                two_by_two(prepared, spec, comparison, weights, group, period, base)?
            };
            if let Some(cell) = cell {
                cells.insert((group, period), cell);
            }
        }
    }
    if spec.estimator == RolloutDidEstimator::Imputation {
        let (imputed, missing) = imputation_cells(prepared, delta, weights)?;
        unimputed = missing;
        cells.extend(imputed);
    }
    Ok((cells, unimputed))
}

/// Unit and period effects fitted on the untreated observations by weighted alternating
/// projections, and the imputed group-time effects with the count of treated observations
/// that cannot be imputed.
#[allow(
    clippy::type_complexity,
    reason = "the cells and the unimputed count are returned together"
)]
fn imputation_cells(
    prepared: &PreparedRollout,
    delta: i64,
    weights: &[f64],
) -> Result<(BTreeMap<(i64, i64), Cell>, usize), &'static str> {
    let untreated =
        |unit: &PanelUnit, period: i64| unit.cohort == 0 || period < unit.cohort - delta;
    let index: BTreeMap<i64, usize> =
        prepared.periods.iter().enumerate().map(|(i, period)| (*period, i)).collect();
    let mut rows: Vec<(usize, usize, f64, f64)> = Vec::new();
    for (u, (unit, weight)) in prepared.units.iter().zip(weights).enumerate() {
        if *weight <= 0.0 {
            continue;
        }
        for (period, value) in &unit.outcomes {
            if untreated(unit, *period) {
                rows.push((u, index[period], *value, *weight));
            }
        }
    }
    let (alpha, lambda) = two_way_effects(&rows, prepared.units.len(), prepared.periods.len())
        .ok_or("the untreated two-way fixed effects did not converge")?;
    let mut sums: BTreeMap<(i64, i64), (f64, f64, usize)> = BTreeMap::new();
    let mut unimputed = 0usize;
    for (u, (unit, weight)) in prepared.units.iter().zip(weights).enumerate() {
        if *weight <= 0.0 || unit.cohort == 0 {
            continue;
        }
        for (period, value) in &unit.outcomes {
            if untreated(unit, *period) {
                continue;
            }
            match (alpha[u], lambda[index[period]]) {
                (Some(a), Some(l)) => {
                    let entry = sums.entry((unit.cohort, *period)).or_insert((0.0, 0.0, 0));
                    entry.0 += weight * (value - a - l);
                    entry.1 += weight;
                    entry.2 += 1;
                }
                _ => unimputed += 1,
            }
        }
    }
    let untreated_units = prepared
        .units
        .iter()
        .zip(weights)
        .filter(|(unit, weight)| **weight > 0.0 && unit.cohort == 0)
        .count();
    let cells = sums
        .into_iter()
        .filter(|(_, (_, total, _))| *total > 0.0)
        .map(|(key, (sum, total, count))| {
            (key, Cell { att: sum / total, treated: count, comparison: untreated_units })
        })
        .collect();
    Ok((cells, unimputed))
}

/// Weighted additive unit and period effects `y ≈ α_u + λ_t` on `rows` of
/// `(unit, period, y, weight)`; an effect with no row stays `None`. Normalized so the first
/// identified period effect is zero.
#[allow(clippy::type_complexity, reason = "unit and period effects are returned together")]
fn two_way_effects(
    rows: &[(usize, usize, f64, f64)],
    units: usize,
    periods: usize,
) -> Option<(Vec<Option<f64>>, Vec<Option<f64>>)> {
    let mut unit_weight = vec![0.0; units];
    let mut period_weight = vec![0.0; periods];
    for (u, t, _, w) in rows {
        unit_weight[*u] += w;
        period_weight[*t] += w;
    }
    let scale = 1.0 + rows.iter().fold(0.0_f64, |largest, (_, _, y, _)| largest.max(y.abs()));
    let mut alpha = vec![0.0; units];
    let mut lambda = vec![0.0; periods];
    let mut converged = false;
    for _ in 0..PROJECTION_ITERATIONS {
        let mut next_alpha = vec![0.0; units];
        for (u, t, y, w) in rows {
            next_alpha[*u] += w * (y - lambda[*t]);
        }
        for (value, total) in next_alpha.iter_mut().zip(&unit_weight) {
            if *total > 0.0 {
                *value /= total;
            }
        }
        let mut next_lambda = vec![0.0; periods];
        for (u, t, y, w) in rows {
            next_lambda[*t] += w * (y - next_alpha[*u]);
        }
        for (value, total) in next_lambda.iter_mut().zip(&period_weight) {
            if *total > 0.0 {
                *value /= total;
            }
        }
        let movement = alpha
            .iter()
            .zip(&next_alpha)
            .chain(lambda.iter().zip(&next_lambda))
            .fold(0.0_f64, |largest, (old, new)| largest.max((old - new).abs()));
        alpha = next_alpha;
        lambda = next_lambda;
        if movement < PROJECTION_TOLERANCE * scale {
            converged = true;
            break;
        }
    }
    if !converged {
        return None;
    }
    let shift = lambda.iter().zip(&period_weight).find(|(_, w)| **w > 0.0).map_or(0.0, |(l, _)| *l);
    Some((
        alpha.iter().zip(&unit_weight).map(|(a, w)| (*w > 0.0).then_some(a + shift)).collect(),
        lambda.iter().zip(&period_weight).map(|(l, w)| (*w > 0.0).then_some(l - shift)).collect(),
    ))
}

/// The static two-way fixed-effects coefficient on the post-adoption indicator, by
/// Frisch–Waugh–Lovell on every read row; a diagnostic, not an estimate of any `ATT`.
fn twfe_static(prepared: &PreparedRollout) -> Option<f64> {
    let index: BTreeMap<i64, usize> =
        prepared.periods.iter().enumerate().map(|(i, period)| (*period, i)).collect();
    let mut outcome_rows = Vec::new();
    let mut treatment_rows = Vec::new();
    for (u, unit) in prepared.units.iter().enumerate() {
        for (period, value) in &unit.outcomes {
            let treated = unit.cohort > 0 && *period >= unit.cohort;
            outcome_rows.push((u, index[period], *value, 1.0));
            treatment_rows.push((u, index[period], if treated { 1.0 } else { 0.0 }, 1.0));
        }
    }
    let residual = |rows: &[(usize, usize, f64, f64)]| -> Option<Vec<f64>> {
        let (alpha, lambda) = two_way_effects(rows, prepared.units.len(), prepared.periods.len())?;
        rows.iter().map(|(u, t, y, _)| Some(y - alpha[*u]? - lambda[*t]?)).collect()
    };
    let y = residual(&outcome_rows)?;
    let d = residual(&treatment_rows)?;
    let denominator: f64 = d.iter().map(|value| value * value).sum();
    (denominator > 1e-12).then(|| y.iter().zip(&d).map(|(a, b)| a * b).sum::<f64>() / denominator)
}
