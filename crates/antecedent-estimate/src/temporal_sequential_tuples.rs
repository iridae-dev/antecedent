//! Tuple-level replicate refits of the unfolded sequential g-computation level.
//!
//! [`crate::estimate_sequence_mechanisms`] fits every needed stationary mechanism of
//! an identified unfolded DAG on one lag-aligned sample and propagates the Sequence
//! overlays through the fitted means. An outer bootstrap that must refit a nuisance on
//! every replicate (the observation-adjusted pseudo-outcome) cannot reuse that entry:
//! reordering the raw series and rebuilding lags pairs values across block junctions.
//! [`PreparedSequenceLevel`] instead keeps the lag-aligned tuples of the full-sample fit
//! (one row per outcome-time anchor, every unfolded column of that anchor intact) and
//! refits every mechanism on a resampled set of anchors, with the outcome variable's
//! columns optionally replaced per replicate. Root-node factual means are recomputed on
//! the resampled rows, so the level's dependence on the empirical covariate (and, for
//! shifts, treatment) averages is part of every replicate.
//!
//! This is a separate entry point: the Frequentist point it reports is the same
//! estimate as `estimate_sequence_mechanisms`, but its resampling is driven entirely by
//! the caller's anchors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::Arc;

use antecedent_core::{ExecutionContext, IdentificationStatus, Lag, TemporalNodeKey, VariableId};
use antecedent_data::{LaggedColumn, LaggedSampleWorkspace, TemporalIndexer, TimeSeriesData};
use antecedent_expr::{EstimandMethod, IdentifiedEstimand};
use antecedent_graph::{DenseNodeId, TemporalDag};
use antecedent_stats::{DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace};

use crate::EstimationError;
use crate::temporal_sequential::SequentialMechanismOverlay;

/// Replacement values for one `(variable, lag)` column of a [`PreparedSequenceLevel`]
/// replicate, aligned with the replicate's anchors.
#[derive(Clone, Copy, Debug)]
pub struct SequenceColumnReplacement<'a> {
    /// Template variable whose lagged column is replaced.
    pub variable: VariableId,
    /// Lag of the column relative to the outcome-time anchor.
    pub lag: u32,
    /// One value per anchor, in anchor order.
    pub values: &'a [f64],
}

/// One unfolded Sequence level, fitted once on its lag-aligned tuples.
///
/// Sample row `i` is the tuple anchored at series row `first_anchor() + i`: every
/// unfolded column is read at `anchor − lag`. See
/// [`prepare_sequence_level`] and [`PreparedSequenceLevel::replicate`].
#[derive(Clone, Debug)]
pub struct PreparedSequenceLevel {
    /// Lag-aligned sample rows.
    n: usize,
    /// Series row of sample row 0 (the largest unfolded lag).
    base: usize,
    /// Rows of the source series.
    series_rows: usize,
    /// Sample columns in order.
    columns: Vec<LaggedColumn>,
    /// Column-major sample values (`columns.len()` blocks of `n`).
    values: Vec<f64>,
    /// Needed unfolded nodes in topological order.
    order: Vec<usize>,
    /// Sample column of each dense node.
    column_of: Vec<usize>,
    /// Sorted parents of each needed node.
    parents: Vec<Vec<usize>>,
    /// Whether a node's mechanism is fitted (not hard-intervened and has parents).
    fitted: Vec<bool>,
    /// Overlay applied at each dense node.
    overlay_at: Vec<Option<SequentialMechanismOverlay>>,
    /// Dense id of the outcome node.
    outcome: usize,
    /// Full-sample level.
    point: f64,
}

/// Prepare the Frequentist sequential g-computation level of `outcome` at
/// `outcome_offset` under `overlays` for tuple-level replicate refits.
///
/// Validation matches [`crate::estimate_sequence_mechanisms`] (Frequentist path), and
/// [`PreparedSequenceLevel::point`] equals that entry's point estimate.
///
/// # Errors
///
/// Empty, non-finite or duplicate overlays, an unidentified or non-unfolded estimand,
/// fewer than three aligned rows, or a singular full-sample fit.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn prepare_sequence_level(
    data: &TimeSeriesData,
    graph: &TemporalDag,
    indexer: &TemporalIndexer,
    estimand: &IdentifiedEstimand,
    outcome: VariableId,
    outcome_offset: i32,
    overlays: &[SequentialMechanismOverlay],
    status: IdentificationStatus,
    ctx: &ExecutionContext,
) -> Result<PreparedSequenceLevel, EstimationError> {
    if overlays.is_empty() {
        return Err(EstimationError::unsupported("sequential overlay requires at least one step"));
    }
    if overlays.iter().any(|overlay| {
        overlay.node.level.is_some_and(|level| !level.is_finite())
            || !overlay.node.shift.is_finite()
            || !overlay.multiplier.is_finite()
            || overlay.bounds.is_some_and(|(lo, hi)| !lo.is_finite() || !hi.is_finite() || lo > hi)
    }) {
        return Err(EstimationError::unsupported(
            "sequential overlays require finite levels and shifts",
        ));
    }
    if estimand.method_kind().ok() != Some(EstimandMethod::TemporalBackdoorUnfolded)
        || !matches!(
            status,
            IdentificationStatus::NonparametricallyIdentified
                | IdentificationStatus::IdentifiedUnderParametricRestrictions
        )
    {
        return Err(EstimationError::unsupported(
            "sequential overlay requires an identified unfolded design",
        ));
    }
    let unfolded =
        graph.unfold(indexer.clone()).map_err(|e| EstimationError::data_msg(e.to_string()))?;
    let dag = &unfolded.dag;
    let node_count = dag.node_count();
    let key_of = |dense: usize| {
        indexer
            .key_of(u32::try_from(dense).unwrap_or(u32::MAX))
            .map_err(|e| EstimationError::data_msg(e.to_string()))
    };
    let outcome = indexer
        .dense_id(TemporalNodeKey { variable: outcome, offset: outcome_offset })
        .map_err(|e| EstimationError::data_msg(e.to_string()))? as usize;
    let mut overlay_at = vec![None; node_count];
    for &overlay in overlays {
        let dense = indexer
            .dense_id(TemporalNodeKey {
                variable: overlay.node.variable,
                offset: overlay.node.offset,
            })
            .map_err(|e| EstimationError::data_msg(e.to_string()))? as usize;
        if overlay_at[dense].is_some() {
            return Err(EstimationError::unsupported(
                "Sequence assigns the same (variable, time) twice; refuse rather than collapse",
            ));
        }
        overlay_at[dense] = Some(overlay);
    }
    let hard = |i: usize| {
        overlay_at[i]
            .is_some_and(|overlay: SequentialMechanismOverlay| overlay.node.level.is_some())
    };
    // Needed nodes: ancestors of the outcome, cutting the incoming edges of hard Set /
    // constant overlays (an additive shift keeps its mechanism).
    let mut needed = vec![false; node_count];
    let mut pending = vec![outcome];
    while let Some(i) = pending.pop() {
        if needed[i] {
            continue;
        }
        needed[i] = true;
        if !hard(i) {
            pending.extend(
                dag.parents(DenseNodeId::from_raw(u32::try_from(i).unwrap_or(u32::MAX)))
                    .iter()
                    .map(|p| p.as_usize()),
            );
        }
    }
    let order: Vec<usize> = dag
        .topological_order()
        .ok_or_else(|| EstimationError::data_msg("unfolded graph is not acyclic"))?
        .into_iter()
        .map(DenseNodeId::as_usize)
        .filter(|&i| needed[i])
        .collect();
    let mut anchor = 0i32;
    for &i in &order {
        anchor = anchor.max(key_of(i)?.offset);
    }
    let mut columns = Vec::with_capacity(order.len());
    let mut column_of = vec![0; node_count];
    let mut max_lag = 0u32;
    for &i in &order {
        let key = key_of(i)?;
        let lag = u32::try_from(anchor - key.offset)
            .map_err(|_| EstimationError::unsupported("invalid unfolded time offset"))?;
        max_lag = max_lag.max(lag);
        column_of[i] = columns.len();
        columns.push(LaggedColumn { variable: key.variable, lag: Lag::from_raw(lag) });
    }
    let plan = data.plan_lagged_sample(max_lag, Arc::from(columns.clone()))?;
    let mut workspace = LaggedSampleWorkspace::default();
    let sample = plan.prepare(data, &mut workspace, &ctx.kernel_policy)?;
    let n = sample.n;
    if n < 3 {
        return Err(EstimationError::unsupported(
            "sequential overlay requires at least three complete aligned rows",
        ));
    }
    let values = sample.values.to_vec();
    let mut parents = vec![Vec::new(); node_count];
    let mut fitted = vec![false; node_count];
    for &i in &order {
        let child = key_of(i)?;
        let mut sorted: Vec<(usize, (u32, i32))> = Vec::new();
        for p in dag.parents(DenseNodeId::from_raw(u32::try_from(i).unwrap_or(u32::MAX))) {
            let parent = key_of(p.as_usize())?;
            sorted.push((p.as_usize(), (parent.variable.raw(), child.offset - parent.offset)));
        }
        // Stable coefficient positions across time copies, as in the sequential engine.
        sorted.sort_by_key(|&(_, key)| key);
        parents[i] = sorted.into_iter().map(|(p, _)| p).collect();
        fitted[i] = !hard(i) && !parents[i].is_empty();
    }
    let mut prepared = PreparedSequenceLevel {
        n,
        base: max_lag as usize,
        series_rows: antecedent_data::TableView::row_count(data),
        columns,
        values,
        order,
        column_of,
        parents,
        fitted,
        overlay_at,
        outcome,
        point: f64::NAN,
    };
    let rows: Vec<usize> = (0..n).collect();
    let point = prepared
        .level_on_rows(&rows, &[], &mut Vec::new(), &mut LeastSquaresWorkspace::default())
        .ok_or_else(|| EstimationError::stats_msg("singular sequential mechanism fit"))?;
    prepared.point = point;
    Ok(prepared)
}

impl PreparedSequenceLevel {
    /// Full-sample interventional level.
    #[must_use]
    pub const fn point(&self) -> f64 {
        self.point
    }

    /// Earliest outcome-time anchor with a complete lag-aligned tuple.
    #[must_use]
    pub const fn first_anchor(&self) -> usize {
        self.base
    }

    /// Largest coefficient count over the fitted mechanisms (intercept included).
    #[must_use]
    pub fn max_parameters(&self) -> usize {
        self.order
            .iter()
            .filter(|&&i| self.fitted[i])
            .map(|&i| self.parents[i].len() + 1)
            .max()
            .unwrap_or(0)
    }

    /// Estimating-equation scores of the full-sample level, for the response block
    /// length: the normal-equation scores of every fitted mechanism's regression (one
    /// series per coefficient, intercept included) and the centered column of every
    /// root node whose factual mean the level reads.
    #[must_use]
    pub fn estimating_scores(&self) -> Vec<Vec<f64>> {
        let n = self.n;
        let column = |c: usize| &self.values[c * n..(c + 1) * n];
        let mut scores = Vec::new();
        for &i in &self.order {
            if self.fitted[i] {
                let p = self.parents[i].len() + 1;
                let mut x = vec![1.0; n * p];
                for (k, &parent) in self.parents[i].iter().enumerate() {
                    x[(k + 1) * n..(k + 2) * n].copy_from_slice(column(self.column_of[parent]));
                }
                scores.extend(
                    crate::temporal_block::normal_equation_scores(
                        &x,
                        n,
                        p,
                        column(self.column_of[i]),
                    )
                    .unwrap_or_default(),
                );
            } else if self.overlay_at[i].is_none_or(|overlay| overlay.node.level.is_none()) {
                let values = column(self.column_of[i]);
                let mean = values.iter().sum::<f64>() / n as f64;
                scores.push(values.iter().map(|v| v - mean).collect());
            }
        }
        scores
    }

    /// Distinct lags (relative to the anchor) at which `variable` enters the tuple.
    #[must_use]
    pub fn lags_of(&self, variable: VariableId) -> Vec<u32> {
        let mut lags: Vec<u32> = self
            .columns
            .iter()
            .filter(|column| column.variable == variable)
            .map(|column| column.lag.raw())
            .collect();
        lags.sort_unstable();
        lags.dedup();
        lags
    }

    /// Refit every mechanism on the tuples anchored at `anchors` (outcome-time series
    /// rows, repeats allowed, each at least [`Self::first_anchor`]) and propagate the
    /// overlays. Columns named in `replacements` take the supplied per-anchor values
    /// instead of the stored ones. Returns `None` when a mechanism refit is singular or
    /// the level is not finite.
    ///
    /// # Errors
    ///
    /// Fewer than three anchors, an anchor outside the tuple range, or a replacement
    /// whose length differs from `anchors`.
    pub fn replicate(
        &self,
        anchors: &[usize],
        replacements: &[SequenceColumnReplacement<'_>],
    ) -> Result<Option<f64>, EstimationError> {
        self.replicate_into(
            anchors,
            replacements,
            &mut Vec::new(),
            &mut Vec::new(),
            &mut LeastSquaresWorkspace::default(),
        )
    }

    /// [`Self::replicate`] writing the row map and design gather into caller-owned buffers.
    ///
    /// # Errors
    ///
    /// Fewer than three anchors, an anchor outside the tuple range, or a replacement
    /// whose length differs from `anchors`.
    pub fn replicate_into(
        &self,
        anchors: &[usize],
        replacements: &[SequenceColumnReplacement<'_>],
        rows: &mut Vec<usize>,
        x_boot: &mut Vec<f64>,
        ls_ws: &mut LeastSquaresWorkspace,
    ) -> Result<Option<f64>, EstimationError> {
        if anchors.len() < 3
            || anchors
                .iter()
                .any(|&s| s < self.base || s >= self.series_rows || s - self.base >= self.n)
            || replacements.iter().any(|replacement| replacement.values.len() != anchors.len())
        {
            return Err(EstimationError::unsupported(
                "sequence replicate anchors must lie in the tuple range and align with replacements",
            ));
        }
        rows.clear();
        rows.extend(anchors.iter().map(|&s| s - self.base));
        Ok(self.level_on_rows(rows, replacements, x_boot, ls_ws))
    }

    /// Gather `rows` of every column (with replacements), refit, and propagate.
    fn level_on_rows(
        &self,
        rows: &[usize],
        replacements: &[SequenceColumnReplacement<'_>],
        x_boot: &mut Vec<f64>,
        ls_ws: &mut LeastSquaresWorkspace,
    ) -> Option<f64> {
        let m = rows.len();
        let gathered: Vec<Vec<f64>> = self
            .columns
            .iter()
            .enumerate()
            .map(|(c, column)| {
                replacements
                    .iter()
                    .find(|r| r.variable == column.variable && r.lag == column.lag.raw())
                    .map_or_else(
                        || {
                            let source = &self.values[c * self.n..(c + 1) * self.n];
                            rows.iter().map(|&row| source[row]).collect()
                        },
                        |replacement| replacement.values.to_vec(),
                    )
            })
            .collect();
        let mean = |c: usize| gathered[c].iter().sum::<f64>() / m as f64;
        let mut coefficients: Vec<Vec<f64>> = vec![Vec::new(); self.column_of.len()];
        for &i in &self.order {
            if !self.fitted[i] {
                continue;
            }
            let p = self.parents[i].len() + 1;
            x_boot.clear();
            x_boot.resize(m * p, 1.0);
            for (k, &parent) in self.parents[i].iter().enumerate() {
                x_boot[(k + 1) * m..(k + 2) * m].copy_from_slice(&gathered[self.column_of[parent]]);
            }
            let y = &gathered[self.column_of[i]];
            coefficients[i] = FaerBackend.least_squares(x_boot, m, p, y, ls_ws).ok()?.coefficients;
        }
        let mut level = vec![0.0; self.column_of.len()];
        for &i in &self.order {
            let natural = if coefficients[i].is_empty() {
                mean(self.column_of[i])
            } else {
                coefficients[i][0]
                    + self.parents[i]
                        .iter()
                        .enumerate()
                        .map(|(k, &node)| coefficients[i][k + 1] * level[node])
                        .sum::<f64>()
            };
            level[i] = self.overlay_at[i].map_or(natural, |overlay| overlay.assigned(natural));
        }
        let value = level[self.outcome];
        value.is_finite().then_some(value)
    }
}

#[cfg(test)]
mod tests {
    use antecedent_core::{
        AssumptionSet, ExecutionContext, IdentificationStatus, Lag, TargetPopulation, VariableId,
    };
    use antecedent_data::{TemporalIndexer, TimeSeriesData};
    use antecedent_graph::{TemporalDag, ensure_lagged};

    use super::*;
    use crate::temporal_sequential::{SequentialNodeOverlay, estimate_sequence_mechanisms};

    /// `Z_s ~ N`, `T_s = 0.5 Z_s + u`, `Y_s = 1 + 2 T_{s-1} + 0.8 T_{s-2} + 0.5 Z_{s-1} + e`.
    fn fixture() -> (TimeSeriesData, TemporalDag) {
        let n = 240usize;
        let ctx = ExecutionContext::for_tests(11);
        let mut rng = ctx.rng.stream(3);
        let mut draw = || 2.0 * rng.next_f64() - 1.0;
        let z: Vec<f64> = (0..n).map(|_| draw()).collect();
        let t: Vec<f64> = z.iter().map(|z| 0.5 * z + draw()).collect();
        let y: Vec<f64> = (0..n)
            .map(|s| {
                let at = |v: &[f64], lag: usize| s.checked_sub(lag).map_or(0.0, |i| v[i]);
                1.0 + 2.0 * at(&t, 1) + 0.8 * at(&t, 2) + 0.5 * at(&z, 1) + 0.3 * draw()
            })
            .collect();
        let data = TimeSeriesData::from_f64_columns(
            [("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())],
            1,
        )
        .unwrap();
        let mut graph = TemporalDag::empty();
        let id = VariableId::from_raw;
        let mut edge = |from: u32, from_lag: u32, to: u32, to_lag: u32| {
            let src = ensure_lagged(&mut graph, id(from), Lag::from_raw(from_lag)).unwrap();
            let dst = ensure_lagged(&mut graph, id(to), Lag::from_raw(to_lag)).unwrap();
            graph.insert_directed(src, dst).unwrap();
        };
        edge(2, 0, 0, 0);
        edge(0, 1, 1, 0);
        edge(0, 2, 1, 0);
        edge(2, 1, 1, 0);
        (data, graph)
    }

    fn identify(
        graph: &TemporalDag,
        overlays: &[SequentialMechanismOverlay],
    ) -> (IdentifiedEstimand, TemporalIndexer) {
        let schedule: Vec<(VariableId, i32)> =
            overlays.iter().map(|o| (o.node.variable, o.node.offset)).collect();
        let id_res = antecedent_identify::TemporalBackdoorIdentifier::new()
            .identify_temporal_schedule(
                graph,
                VariableId::from_raw(1),
                0,
                &schedule,
                None,
                TargetPopulation::AllObserved,
            )
            .unwrap();
        let estimand = id_res.result.estimands.first().cloned().expect("identified estimand");
        (estimand, id_res.indexer)
    }

    fn overlays() -> Vec<SequentialMechanismOverlay> {
        [(-2, Some(0.5), 0.0), (-1, None, 0.4)]
            .into_iter()
            .map(|(offset, level, shift)| {
                SequentialMechanismOverlay::from(SequentialNodeOverlay {
                    variable: VariableId::from_raw(0),
                    offset,
                    level,
                    shift,
                })
            })
            .collect()
    }

    #[test]
    fn prepared_point_matches_the_sequential_engine_and_identity_replicate() {
        let (data, graph) = fixture();
        let overlays = overlays();
        let (estimand, indexer) = identify(&graph, &overlays);
        let ctx = ExecutionContext::for_tests(5);
        let status = IdentificationStatus::NonparametricallyIdentified;
        let (effect, _) = estimate_sequence_mechanisms(
            &data,
            &graph,
            &indexer,
            &estimand,
            VariableId::from_raw(1),
            0,
            &overlays,
            status,
            AssumptionSet::new(),
            0,
            None,
            &ctx,
        )
        .unwrap();
        let prepared = prepare_sequence_level(
            &data,
            &graph,
            &indexer,
            &estimand,
            VariableId::from_raw(1),
            0,
            &overlays,
            status,
            &ctx,
        )
        .unwrap();
        assert!(
            (prepared.point() - effect.ate).abs() < 1e-10,
            "{} vs {}",
            prepared.point(),
            effect.ate
        );
        let anchors: Vec<usize> =
            (prepared.first_anchor()..antecedent_data::TableView::row_count(&data)).collect();
        let same = prepared.replicate(&anchors, &[]).unwrap().unwrap();
        assert!((same - prepared.point()).abs() < 1e-10);
        assert_eq!(prepared.lags_of(VariableId::from_raw(1)), vec![0]);
    }

    #[test]
    fn replicate_refits_on_anchors_and_uses_outcome_replacements() {
        let (data, graph) = fixture();
        let overlays = overlays();
        let (estimand, indexer) = identify(&graph, &overlays);
        let ctx = ExecutionContext::for_tests(5);
        let prepared = prepare_sequence_level(
            &data,
            &graph,
            &indexer,
            &estimand,
            VariableId::from_raw(1),
            0,
            &overlays,
            IdentificationStatus::NonparametricallyIdentified,
            &ctx,
        )
        .unwrap();
        let first = prepared.first_anchor();
        let anchors: Vec<usize> =
            (first..antecedent_data::TableView::row_count(&data)).step_by(2).collect();
        let half = prepared.replicate(&anchors, &[]).unwrap().unwrap();
        assert!((half - prepared.point()).abs() > 1e-8, "a different row set refits");
        // Adding a constant to the outcome column shifts the level by that constant.
        let y = match antecedent_data::TableView::column(&data, VariableId::from_raw(1)).unwrap() {
            antecedent_data::ColumnView::Float64(column) => column.values.to_vec(),
            _ => unreachable!(),
        };
        let shifted: Vec<f64> = anchors.iter().map(|&s| y[s] + 3.0).collect();
        let moved = prepared
            .replicate(
                &anchors,
                &[SequenceColumnReplacement {
                    variable: VariableId::from_raw(1),
                    lag: 0,
                    values: &shifted,
                }],
            )
            .unwrap()
            .unwrap();
        assert!((moved - half - 3.0).abs() < 1e-8, "{moved} vs {half}");
        assert!(prepared.replicate(&[first, first + 1], &[]).is_err());
        assert!(prepared.replicate(&[0, first, first + 1], &[]).is_err() || first == 0);
    }
}
