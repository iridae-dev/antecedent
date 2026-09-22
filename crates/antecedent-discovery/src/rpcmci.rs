//! RPCMCI: regime-PCMCI with typed assignments and per-regime graphs.
//!
//! Per-regime discovery keeps the full series lag alignment and retains only
//! effective samples whose entire lag window lies inside the regime (masked CI).
//! Optional alternating assignment refines labels by residual fit under each
//! regime's discovered lagged parents.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::needless_range_loop, clippy::too_many_lines)]
#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use antecedent_core::{ExecutionContext, RegimeId, VariableId};
use antecedent_data::{ColumnView, LaggedFrame, TableView, TimeSeriesData};
use antecedent_graph::TemporalCpdag;
use antecedent_stats::{
    ConditionalIndependence, DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace,
};

use crate::engine::DiscoveryWorkspace;
use crate::error::DiscoveryError;
use crate::pcmci_plus::PcmciPlus;
use crate::result::{AlgorithmRecord, CpdagDiscoveryResult, DiscoveryDiagnostic};

/// Retained `(source variable, lag)` parents, indexed by target variable.
type LaggedParents = Vec<Vec<(usize, usize)>>;

/// Columnar regime label per time index.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RegimeAssignment {
    /// `regimes[t]` is the regime id at time `t`.
    pub regimes: Arc<[RegimeId]>,
}

impl RegimeAssignment {
    /// Construct from a regime id per time step.
    ///
    /// # Errors
    ///
    /// Empty assignment.
    pub fn try_new(regimes: impl Into<Arc<[RegimeId]>>) -> Result<Self, DiscoveryError> {
        let regimes = regimes.into();
        if regimes.is_empty() {
            return Err(DiscoveryError::Unsupported {
                message: "regime assignment needs ≥1 time index",
            });
        }
        Ok(Self { regimes })
    }

    /// Length (series length).
    #[must_use]
    pub fn len(&self) -> usize {
        self.regimes.len()
    }

    /// Whether empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.regimes.is_empty()
    }

    /// Unique regime ids in ascending order.
    #[must_use]
    pub fn unique_regimes(&self) -> Vec<RegimeId> {
        let mut set = BTreeSet::new();
        for &r in self.regimes.iter() {
            set.insert(r);
        }
        set.into_iter().collect()
    }

    /// Row indexes belonging to `regime`.
    #[must_use]
    pub fn indexes_for(&self, regime: RegimeId) -> Vec<usize> {
        self.regimes.iter().enumerate().filter_map(|(i, &r)| (r == regime).then_some(i)).collect()
    }

    /// Regime at time `t`, if in range.
    #[must_use]
    pub fn at(&self, t: usize) -> Option<RegimeId> {
        self.regimes.get(t).copied()
    }
}

/// One temporal CPDAG (or equivalent) per regime — never collapsed to a single graph.
#[derive(Clone, Debug)]
pub struct RegimeGraphCollection {
    /// Graphs keyed by regime id.
    pub graphs: Arc<[(RegimeId, TemporalCpdag)]>,
}

impl RegimeGraphCollection {
    /// Lookup by regime.
    #[must_use]
    pub fn get(&self, regime: RegimeId) -> Option<&TemporalCpdag> {
        self.graphs.iter().find_map(|(r, g)| (*r == regime).then_some(g))
    }
}

/// RPCMCI discovery result.
#[derive(Clone, Debug)]
pub struct RpcmciDiscoveryResult {
    /// Final regime assignment used.
    pub assignments: RegimeAssignment,
    /// One CPDAG per retained regime.
    pub graphs: RegimeGraphCollection,
    /// Nested PCMCI+ results per regime (same order as `graphs`).
    pub per_regime: Arc<[CpdagDiscoveryResult]>,
    /// Algorithm record.
    pub algorithm: AlgorithmRecord,
    /// Diagnostics.
    pub diagnostics: Vec<DiscoveryDiagnostic>,
}

/// Default [`Rpcmci::switch_penalty`].
pub const DEFAULT_SWITCH_PENALTY: f64 = 1.0;

/// Regime-PCMCI discovery.
#[derive(Clone, Debug)]
pub struct Rpcmci {
    /// Nested PCMCI+.
    pub pcmci_plus: PcmciPlus,
    /// Minimum regime length (raw rows) to discover.
    pub min_regime_len: usize,
    /// Alternating assignment iterations (`0` = fixed labels only).
    pub alternating_iters: usize,
    /// Cost of one regime switch in the alternating refinement, in units of a variable's
    /// variance (see [`Self::with_switch_penalty`]).
    pub switch_penalty: f64,
    /// Optional regime assignment for [`crate::algorithm::DiscoveryAlgorithm`] dispatch.
    pub(crate) assignment: Option<RegimeAssignment>,
}

impl Default for Rpcmci {
    fn default() -> Self {
        Self::new()
    }
}

impl Rpcmci {
    /// Defaults: nested PCMCI+, min regime length 40, one alternating refinement pass.
    #[must_use]
    pub fn new() -> Self {
        Self {
            pcmci_plus: PcmciPlus::new(),
            min_regime_len: 40,
            alternating_iters: 1,
            switch_penalty: DEFAULT_SWITCH_PENALTY,
            assignment: None,
        }
    }

    /// Configure nested constraints via PCMCI+.
    #[must_use]
    pub fn with_pcmci_plus(mut self, pcmci_plus: PcmciPlus) -> Self {
        self.pcmci_plus = pcmci_plus;
        self
    }

    /// Minimum rows in a regime before discovery runs.
    #[must_use]
    pub fn with_min_regime_len(mut self, min_regime_len: usize) -> Self {
        self.min_regime_len = min_regime_len;
        self
    }

    /// Alternating assignment / discovery iterations after the initial labels.
    #[must_use]
    pub fn with_alternating_iters(mut self, alternating_iters: usize) -> Self {
        self.alternating_iters = alternating_iters;
        self
    }

    /// Per-switch penalty of the alternating refinement.
    ///
    /// Refined labels minimise the summed variance-standardised one-step squared error over
    /// all variables plus this penalty per regime change, so one unit is the cost of leaving a
    /// single variable's variance unexplained for one step. `0` allows a switch at every time
    /// point (which shatters the lag windows regime discovery needs).
    #[must_use]
    pub fn with_switch_penalty(mut self, switch_penalty: f64) -> Self {
        self.switch_penalty = switch_penalty;
        self
    }

    /// Store regime labels for [`crate::algorithm::DiscoveryAlgorithm::discover`].
    #[must_use]
    pub fn with_assignment(mut self, assignment: RegimeAssignment) -> Self {
        self.assignment = Some(assignment);
        self
    }

    /// Replace the CI test.
    #[must_use]
    pub fn with_ci(mut self, ci: Arc<dyn ConditionalIndependence + Send + Sync>) -> Self {
        self.pcmci_plus = self.pcmci_plus.with_ci(ci);
        self
    }

    /// Run RPCMCI with an explicit regime assignment (no silent single-graph collapse).
    ///
    /// # Errors
    ///
    /// Length mismatch, empty regimes, or nested discovery failures.
    pub fn run(
        &self,
        data: &TimeSeriesData,
        variables: &[VariableId],
        assignments: &RegimeAssignment,
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<RpcmciDiscoveryResult, DiscoveryError> {
        if assignments.len() != data.row_count() {
            return Err(DiscoveryError::data_msg(format!(
                "regime assignment length {} != series length {}",
                assignments.len(),
                data.row_count()
            )));
        }
        let mut assignment = assignments.clone();
        let mut diagnostics = Vec::new();
        let mut last = self.discover_regimes(data, variables, &assignment, workspace, ctx)?;

        for iter in 0..self.alternating_iters {
            let Some(updated) = reassign_by_lagged_residual(
                data,
                variables,
                &last.graphs,
                &assignment,
                self.switch_penalty,
            )?
            else {
                diagnostics.push(DiscoveryDiagnostic {
                    code: Arc::from("rpcmci.alternating_stop"),
                    message: Arc::from(format!(
                        "alternating assignment converged after {iter} refinement(s)"
                    )),
                });
                break;
            };
            if updated.regimes.as_ref() == assignment.regimes.as_ref() {
                diagnostics.push(DiscoveryDiagnostic {
                    code: Arc::from("rpcmci.alternating_stop"),
                    message: Arc::from(format!(
                        "alternating assignment unchanged after {iter} refinement(s)"
                    )),
                });
                break;
            }
            // Refinement is an optional improvement, so it must not be able to turn a run
            // that already succeeded into a hard failure. On data with no real regime
            // structure the fitted models agree, every row is a near-tie, and the labels
            // legitimately collapse toward one regime — leaving the others below
            // `min_regime_len`. Keep the last good result and say why, rather than
            // propagating "all regimes too short".
            match self.discover_regimes(data, variables, &updated, workspace, ctx) {
                Ok(next) => {
                    assignment = updated;
                    last = next;
                    diagnostics.push(DiscoveryDiagnostic {
                        code: Arc::from("rpcmci.alternating"),
                        message: Arc::from(format!(
                            "completed alternating refinement {}",
                            iter + 1
                        )),
                    });
                }
                Err(e) => {
                    diagnostics.push(DiscoveryDiagnostic {
                        code: Arc::from("rpcmci.alternating_rejected"),
                        message: Arc::from(format!(
                            "refinement {} discarded — re-discovery on the refined labels \
                             failed ({e}); keeping the previous assignment",
                            iter + 1
                        )),
                    });
                    break;
                }
            }
        }
        if assignment != *assignments {
            diagnostics.push(DiscoveryDiagnostic {
                code: Arc::from("rpcmci.post_selection"),
                message: Arc::from(
                    "regime labels were refined from the same data the per-regime tests use: \
                     reported p-values are conditional on data-driven labels and are not \
                     valid frequentist p-values for the refined regimes",
                ),
            });
        }
        diagnostics.extend(last.diagnostics);
        Ok(RpcmciDiscoveryResult {
            assignments: assignment,
            graphs: last.graphs,
            per_regime: last.per_regime,
            algorithm: last.algorithm,
            diagnostics,
        })
    }

    fn discover_regimes(
        &self,
        data: &TimeSeriesData,
        variables: &[VariableId],
        assignments: &RegimeAssignment,
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<RpcmciDiscoveryResult, DiscoveryError> {
        let regimes = assignments.unique_regimes();
        if regimes.is_empty() {
            return Err(DiscoveryError::Unsupported {
                message: "RPCMCI needs ≥1 distinct regime"
            });
        }

        let max_lag = self.pcmci_plus.engine().constraints.temporal.max_lag.raw();
        let frame_depth = 2 * max_lag;
        let full_frame = LaggedFrame::from_series(data, variables, frame_depth, &ctx.kernel_policy)
            .map_err(DiscoveryError::from)?;

        let mut graphs = Vec::with_capacity(regimes.len());
        let mut per_regime = Vec::with_capacity(regimes.len());
        let mut diagnostics = Vec::new();

        for regime in regimes {
            let idxs = assignments.indexes_for(regime);
            if idxs.len() < self.min_regime_len {
                diagnostics.push(DiscoveryDiagnostic {
                    code: Arc::from("rpcmci.skip_short"),
                    message: Arc::from(format!(
                        "regime {} has {} rows (< min {}); skipped",
                        regime.raw(),
                        idxs.len(),
                        self.min_regime_len
                    )),
                });
                continue;
            }
            let keep = regime_window_mask(assignments, regime, frame_depth, data.row_count());
            let retained = keep.iter().filter(|&&k| k).count();
            if retained < self.min_regime_len.saturating_sub(frame_depth as usize).max(8) {
                diagnostics.push(DiscoveryDiagnostic {
                    code: Arc::from("rpcmci.skip_short_windows"),
                    message: Arc::from(format!(
                        "regime {} has only {retained} valid lag windows; skipped",
                        regime.raw()
                    )),
                });
                continue;
            }
            let masked = full_frame
                .retain_effective(&keep)
                .map_err(|e| DiscoveryError::data_msg(format!("regime mask: {e}")))?;
            let result = self.pcmci_plus.run_on_frame(&masked, variables, workspace, ctx)?;
            diagnostics.push(DiscoveryDiagnostic {
                code: Arc::from("rpcmci.masked_ci"),
                message: Arc::from(format!(
                    "regime {}: retained {retained}/{} effective windows (no row-splicing)",
                    regime.raw(),
                    full_frame.n_effective()
                )),
            });
            graphs.push((regime, result.evidence.graph.clone()));
            per_regime.push(result);
        }

        if graphs.is_empty() {
            return Err(DiscoveryError::Unsupported {
                message: "RPCMCI produced no regime graphs (all regimes too short?)",
            });
        }

        let algorithm = crate::pipeline::algorithm_record(
            "rpcmci",
            format!(
                "regimes={},min_len={},nested={},alternating={}",
                graphs.len(),
                self.min_regime_len,
                self.pcmci_plus.engine().constraints.temporal.max_lag.raw(),
                self.alternating_iters
            ),
        );
        diagnostics.push(DiscoveryDiagnostic {
            code: Arc::from("rpcmci.graphs"),
            message: Arc::from(format!("produced {} per-regime temporal CPDAGs", graphs.len())),
        });

        Ok(RpcmciDiscoveryResult {
            assignments: assignments.clone(),
            graphs: RegimeGraphCollection { graphs: Arc::from(graphs) },
            per_regime: Arc::from(per_regime),
            algorithm,
            diagnostics,
        })
    }

    /// Infer a two-regime assignment by median split on `indicator`, then discover
    /// (with alternating refinement when configured).
    ///
    /// # Errors
    ///
    /// Missing / non-float indicator, or nested failures.
    pub fn run_median_split(
        &self,
        data: &TimeSeriesData,
        variables: &[VariableId],
        indicator: VariableId,
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<RpcmciDiscoveryResult, DiscoveryError> {
        let assignments = median_split_assignment(data, indicator)?;
        self.run(data, variables, &assignments, workspace, ctx)
    }
}

/// Effective-row mask: keep sample `i` (raw time `i + max_lag`) only when the full
/// lag window `t-max_lag..=t` lies in `regime`.
fn regime_window_mask(
    assignments: &RegimeAssignment,
    regime: RegimeId,
    max_lag: u32,
    series_len: usize,
) -> Vec<bool> {
    let ml = max_lag as usize;
    let n_eff = series_len.saturating_sub(ml);
    let mut keep = vec![false; n_eff];
    for i in 0..n_eff {
        let t = i + ml;
        keep[i] = (0..=ml).all(|l| assignments.at(t - l) == Some(regime));
    }
    keep
}

fn median_split_assignment(
    data: &TimeSeriesData,
    indicator: VariableId,
) -> Result<RegimeAssignment, DiscoveryError> {
    let ColumnView::Float64(col) = data
        .column(indicator)
        .map_err(|e| DiscoveryError::data_msg(format!("regime indicator: {e}")))?
    else {
        return Err(DiscoveryError::Unsupported { message: "regime indicator must be float64" });
    };
    // A missing indicator has no regime: refuse rather than let NaN fall on one side of the
    // split (`NaN <= median` is false, which would silently label it regime 1).
    if let Some(row) =
        (0..col.values.len()).find(|&i| !col.validity.is_valid(i) || !col.values[i].is_finite())
    {
        return Err(DiscoveryError::data_msg(format!(
            "regime indicator has a missing or non-finite value at row {row}; \
             a regime cannot be assigned"
        )));
    }
    let mut sorted: Vec<f64> = col.values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted[sorted.len() / 2];
    let regimes: Vec<RegimeId> = col
        .values
        .iter()
        .map(|&v| if v <= mid { RegimeId::from_raw(0) } else { RegimeId::from_raw(1) })
        .collect();
    RegimeAssignment::try_new(Arc::from(regimes))
}

/// One regime's fitted one-step equations, one per variable.
struct RegimeModel {
    regime: RegimeId,
    /// Indexed by variable position; a variable without retained lagged parents is an
    /// intercept-only equation (its regime-specific mean).
    equations: Vec<RegimeEquation>,
}

/// `variable ~ intercept + Σ β_k · parent_k(t − lag_k)`.
struct RegimeEquation {
    /// `(variable position, lag ≥ 1)` per parent.
    parents: Vec<(usize, usize)>,
    /// `[intercept, β₁, …]`, aligned with `parents`.
    coefficients: Vec<f64>,
}

/// Reassign time points to regimes by how well each regime's fitted lagged equations
/// predict the *whole* system one step ahead.
///
/// Every variable is scored under every regime (intercept-only when the regime retained no
/// lagged parent for it), each squared residual is divided by that variable's variance, and
/// the per-step costs are summed. Regimes are therefore compared on the same quantity: a
/// regime cannot win a time point merely because its only retained equation belongs to a
/// low-variance variable, and a variable a regime has no structure for pays its full
/// unexplained variance rather than dropping out of the score.
///
/// Labels are the minimum-cost path with a per-switch penalty of `switch_penalty`
/// (standardised-variance units), found by Viterbi, so an isolated near-tie cannot flip a
/// single time point and shatter the lag windows regime discovery needs. A regime is scored
/// only if every one of its equations is identified on the rows currently assigned to it;
/// returns `None` when fewer than two regimes qualify.
fn reassign_by_lagged_residual(
    data: &TimeSeriesData,
    variables: &[VariableId],
    graphs: &RegimeGraphCollection,
    current: &RegimeAssignment,
    switch_penalty: f64,
) -> Result<Option<RegimeAssignment>, DiscoveryError> {
    let n = data.row_count();
    if n < 2 || graphs.graphs.is_empty() {
        return Ok(None);
    }
    let mut cols: Vec<Vec<f64>> = Vec::with_capacity(variables.len());
    for &v in variables {
        let ColumnView::Float64(c) =
            data.column(v).map_err(|e| DiscoveryError::data_msg(format!("reassign col: {e}")))?
        else {
            return Err(DiscoveryError::Unsupported {
                message: "RPCMCI reassignment currently supports float64 columns only",
            });
        };
        cols.push(c.values.to_vec());
    }

    // Retained lagged parents per regime and target variable.
    let mut regime_parents: Vec<(RegimeId, LaggedParents)> = Vec::new();
    for (regime, g) in graphs.graphs.iter() {
        let mut by_target: LaggedParents = vec![Vec::new(); variables.len()];
        for (i, node) in g.nodes().iter().enumerate() {
            let antecedent_graph::NodeRef::Lagged { variable: tgt, lag: tlag } = node else {
                continue;
            };
            if !tlag.is_contemporaneous() {
                continue;
            }
            let Some(ti) = variables.iter().position(|v| v == tgt) else {
                continue;
            };
            let from = antecedent_graph::DenseNodeId::from_raw(crate::indexing::dense_u32(i));
            for p in g.parents(from) {
                if let Some(antecedent_graph::NodeRef::Lagged { variable: src, lag: slag }) =
                    g.nodes().get(p.as_usize())
                {
                    if slag.raw() >= 1 {
                        if let Some(si) = variables.iter().position(|v| v == src) {
                            let parent = (si, slag.raw() as usize);
                            if !by_target[ti].contains(&parent) {
                                by_target[ti].push(parent);
                            }
                        }
                    }
                }
            }
        }
        regime_parents.push((*regime, by_target));
    }
    if regime_parents.iter().all(|(_, by_target)| by_target.iter().all(Vec::is_empty)) {
        return Ok(None);
    }
    // Rows start after the deepest retained lag so every equation sees a full window.
    let first_row = regime_parents
        .iter()
        .flat_map(|(_, by_target)| by_target.iter().flatten())
        .map(|&(_, lag)| lag)
        .max()
        .unwrap_or(1);
    if n <= first_row + 1 {
        return Ok(None);
    }

    // Per-variable variance over the scored rows: the common yardstick for residuals.
    let scale: Vec<f64> = cols
        .iter()
        .map(|c| {
            let rows = &c[first_row..];
            let mean = rows.iter().sum::<f64>() / rows.len() as f64;
            rows.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / rows.len() as f64
        })
        .collect();

    // Fit each regime's equations on the rows currently assigned to it.
    let backend = FaerBackend;
    let mut models: Vec<RegimeModel> = Vec::with_capacity(regime_parents.len());
    'regimes: for (regime, by_target) in &regime_parents {
        let rows: Vec<usize> = (first_row..n).filter(|&t| current.regimes[t] == *regime).collect();
        let mut equations = Vec::with_capacity(variables.len());
        for (ti, parents) in by_target.iter().enumerate() {
            let ncols = parents.len() + 1;
            // Identification: more rows than free parameters.
            if rows.len() <= ncols {
                continue 'regimes;
            }
            let mut design = Vec::with_capacity(rows.len() * ncols);
            // Column-major: intercept column first, then one lagged parent per column.
            design.extend(std::iter::repeat_n(1.0, rows.len()));
            for &(si, lag) in parents {
                for &t in &rows {
                    design.push(cols[si][t - lag]);
                }
            }
            let y: Vec<f64> = rows.iter().map(|&t| cols[ti][t]).collect();
            let mut ws = LeastSquaresWorkspace::default();
            let Ok(fit) = backend.least_squares(&design, rows.len(), ncols, &y, &mut ws) else {
                continue 'regimes;
            };
            if fit.coefficients.iter().any(|c| !c.is_finite()) {
                continue 'regimes;
            }
            equations.push(RegimeEquation {
                parents: parents.clone(),
                coefficients: fit.coefficients.clone(),
            });
        }
        models.push(RegimeModel { regime: *regime, equations });
    }
    if models.len() < 2 {
        return Ok(None);
    }

    // Minimum-cost label path: cost of a step under a regime is the variance-standardised
    // one-step squared error summed over all variables; each switch costs `switch_penalty`.
    let penalty = switch_penalty.max(0.0);
    let k = models.len();
    let step_cost = |t: usize, m: &RegimeModel| -> f64 {
        let mut err = 0.0;
        for (ti, eq) in m.equations.iter().enumerate() {
            if scale[ti] <= f64::MIN_POSITIVE {
                continue; // constant column: nothing to explain
            }
            let mut pred = eq.coefficients[0];
            for (j, &(si, lag)) in eq.parents.iter().enumerate() {
                pred += eq.coefficients[j + 1] * cols[si][t - lag];
            }
            let resid = cols[ti][t] - pred;
            err += resid * resid / scale[ti];
        }
        err
    };
    let steps = n - first_row;
    let mut cost = vec![0.0f64; steps * k];
    let mut back = vec![0usize; steps * k];
    for (ri, m) in models.iter().enumerate() {
        cost[ri] = step_cost(first_row, m);
    }
    for s in 1..steps {
        let t = first_row + s;
        // Best predecessor for a switch; staying is preferred on ties.
        let (best_prev, best_prev_cost) = (0..k)
            .map(|r| (r, cost[(s - 1) * k + r]))
            .fold((0usize, f64::INFINITY), |acc, x| if x.1 < acc.1 { x } else { acc });
        for (ri, m) in models.iter().enumerate() {
            let stay = cost[(s - 1) * k + ri];
            let switch = best_prev_cost + penalty;
            let (prev, base) = if stay <= switch { (ri, stay) } else { (best_prev, switch) };
            cost[s * k + ri] = base + step_cost(t, m);
            back[s * k + ri] = prev;
        }
    }
    let mut state = (0..k)
        .map(|r| (r, cost[(steps - 1) * k + r]))
        .fold((0usize, f64::INFINITY), |acc, x| if x.1 < acc.1 { x } else { acc })
        .0;
    let mut out = current.regimes.to_vec();
    for s in (0..steps).rev() {
        out[first_row + s] = models[state].regime;
        state = back[s * k + state];
    }
    for t in 0..first_row {
        out[t] = out[first_row];
    }
    Ok(Some(RegimeAssignment { regimes: Arc::from(out) }))
}

/// Seed helper for regime discovery benches: build a two-regime assignment map.
#[must_use]
pub fn two_regime_half_split(series_len: usize) -> RegimeAssignment {
    let mid = series_len / 2;
    let regimes: Vec<RegimeId> = (0..series_len)
        .map(|t| if t < mid { RegimeId::from_raw(0) } else { RegimeId::from_raw(1) })
        .collect();
    RegimeAssignment { regimes: Arc::from(regimes) }
}

/// Count edges per regime (bench / conformance helper).
#[must_use]
pub fn regime_edge_counts(graphs: &RegimeGraphCollection) -> BTreeMap<u32, (usize, usize)> {
    let mut out = BTreeMap::new();
    for (rid, g) in graphs.graphs.iter() {
        out.insert(rid.raw(), (g.directed_edge_count(), g.undirected_edge_count()));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::{DiscoveryConstraints, TemporalConstraints};
    use antecedent_core::{
        CausalSchemaBuilder, Lag, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
    };
    use antecedent_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
        ValidityBitmap,
    };

    fn two_regime_series(n: usize) -> TimeSeriesData {
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "x",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let mut x = vec![0.0; n];
        let mut y = vec![0.0; n];
        let mid = n / 2;
        for t in 1..n {
            if t < mid {
                x[t] = 0.6 * x[t - 1] + 0.05 * (t as f64).sin();
                y[t] = 0.5 * x[t] + 0.1 * y[t - 1];
            } else {
                x[t] = 0.2 * x[t - 1] + 0.05 * (t as f64).cos();
                y[t] = -0.4 * x[t] + 0.1 * y[t - 1];
            }
        }
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(x),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(y),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        TimeSeriesData::try_new(
            storage,
            TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
        )
        .unwrap()
    }

    #[test]
    fn rpcmci_returns_one_graph_per_regime() {
        let data = two_regime_series(200);
        let vars = [VariableId::from_raw(0), VariableId::from_raw(1)];
        let assign = two_regime_half_split(200);
        let algo = Rpcmci::new().with_min_regime_len(40).with_alternating_iters(0).with_pcmci_plus(
            PcmciPlus::new().with_fdr(false).with_constraints(DiscoveryConstraints {
                temporal: TemporalConstraints {
                    max_lag: Lag::from_raw(1),
                    min_lag: Lag::CONTEMPORANEOUS,
                },
                alpha: 0.3,
                max_cond_size: 1,
                ..DiscoveryConstraints::default()
            }),
        );
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(3);
        let result = algo.run(&data, &vars, &assign, &mut ws, &ctx).unwrap();
        assert_eq!(result.algorithm.id.as_ref(), "rpcmci");
        assert!(result.graphs.get(RegimeId::from_raw(0)).is_some());
        assert!(result.graphs.get(RegimeId::from_raw(1)).is_some());
        assert!(result.diagnostics.iter().any(|d| d.code.as_ref() == "rpcmci.masked_ci"));
    }

    /// Two regimes that share a link structure but differ in coefficient magnitude.
    ///
    /// `x` follows the same process throughout, so the only thing distinguishing the halves
    /// is the strength of `x_{t-1} → y_t`.
    fn coefficient_shift_series(n: usize) -> TimeSeriesData {
        let mut b = CausalSchemaBuilder::new();
        for name in ["x", "y"] {
            b.add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        }
        let schema = b.build().unwrap();
        let mut x = vec![0.0; n];
        let mut y = vec![0.0; n];
        let mid = n / 2;
        for t in 1..n {
            x[t] = 0.5 * x[t - 1] + (t as f64 * 0.7).sin();
            // Same parent, very different strength: neither coefficient is near 1.
            y[t] = if t < mid { 0.2 * x[t - 1] } else { 4.0 * x[t - 1] };
        }
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(x),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(y),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        TimeSeriesData::try_new(
            storage,
            TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
        )
        .unwrap()
    }

    /// Alternating reassignment must distinguish regimes by fitted strength, not just by
    /// which links were retained.
    ///
    /// The refinement step used `pred = x_{t-1}` — the raw parent value, i.e. an implicit
    /// coefficient of exactly 1 and no intercept, with nothing ever fit. Two regimes that
    /// retain the *same* links then score every timestep identically, the strict `<`
    /// comparison always keeps the first, and every row collapses into one regime no matter
    /// what the data says. Fitting each regime's equation on its own rows is what makes the
    /// comparison mean anything.
    #[test]
    fn alternating_reassignment_separates_regimes_by_fitted_strength() {
        let n = 200;
        let data = coefficient_shift_series(n);
        let vars = [VariableId::from_raw(0), VariableId::from_raw(1)];
        let assign = two_regime_half_split(n);
        let algo = Rpcmci::new().with_min_regime_len(40).with_alternating_iters(1).with_pcmci_plus(
            PcmciPlus::new().with_fdr(false).with_constraints(DiscoveryConstraints {
                temporal: TemporalConstraints {
                    max_lag: Lag::from_raw(1),
                    min_lag: Lag::CONTEMPORANEOUS,
                },
                alpha: 0.3,
                max_cond_size: 1,
                ..DiscoveryConstraints::default()
            }),
        );
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(3);
        let result = algo.run(&data, &vars, &assign, &mut ws, &ctx).unwrap();

        assert_eq!(
            result.assignments.unique_regimes().len(),
            2,
            "refinement collapsed every row into one regime; assignment={:?}",
            result.assignments.regimes
        );
        // Refinement seeded from the true split should broadly agree with it.
        let agree = (1..n)
            .filter(|&t| {
                let truth = if t < n / 2 { RegimeId::from_raw(0) } else { RegimeId::from_raw(1) };
                result.assignments.regimes[t] == truth
            })
            .count();
        assert!(agree * 10 >= (n - 1) * 7, "only {agree}/{} rows kept their true regime", n - 1);
    }

    #[test]
    fn median_split_refuses_missing_indicator() {
        let mut x = vec![1.0; 20];
        x[7] = f64::NAN;
        let data = series_from(x, vec![0.0; 20]);
        let err = median_split_assignment(&data, VariableId::from_raw(0)).unwrap_err();
        assert!(err.to_string().contains("row 7"), "{err}");
    }

    #[test]
    fn regime_window_mask_rejects_boundary_crossing() {
        let assign = two_regime_half_split(10);
        let keep = regime_window_mask(&assign, RegimeId::from_raw(0), 2, 10);
        assert_eq!(keep.len(), 8);
        assert!(keep[0]);
        assert!(keep[2]);
        assert!(!keep[3]);
    }

    fn series_from(x: Vec<f64>, y: Vec<f64>) -> TimeSeriesData {
        let n = x.len();
        let mut b = CausalSchemaBuilder::new();
        for name in ["x", "y"] {
            b.add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        }
        let schema = b.build().unwrap();
        let cols = [x, y]
            .into_iter()
            .enumerate()
            .map(|(i, v)| {
                OwnedColumn::Float64(
                    Float64Column::new(
                        VariableId::from_raw(i as u32),
                        Arc::from(v),
                        ValidityBitmap::all_valid(n),
                    )
                    .unwrap(),
                )
            })
            .collect();
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        TimeSeriesData::try_new(
            storage,
            TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
        )
        .unwrap()
    }

    /// Regime 0 retains only `x_{t-1} → x_t` (Var x ≈ 100) and regime 1 only `y_{t-1} → y_t`
    /// (Var y ≈ 1). Scoring each regime on just its own retained equation on raw scales makes
    /// the low-variance regime win every step, whatever the data say. Scoring every variable
    /// under every regime, standardised, recovers the true labels.
    #[test]
    fn reassignment_compares_regimes_on_all_variables_standardised() {
        let n = 240usize;
        let mut state = 12345u64;
        let mut unif = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        // Uniform(-0.5, 0.5) has variance 1/12: scale to unit variance.
        let unit = 12.0f64.sqrt();
        let mut x = vec![0.0; n];
        let mut y = vec![0.0; n];
        for t in 1..n {
            let (ex, ey) = (unif() * unit, unif() * unit);
            if t < n / 2 {
                // Var x = 4.4² / (1 - 0.81) ≈ 100; y iid, Var y = 1.
                x[t] = 0.9 * x[t - 1] + 4.4 * ex;
                y[t] = ey;
            } else {
                // x iid, Var x = 100; Var y = 0.44² / (1 - 0.81) ≈ 1.
                x[t] = 10.0 * ex;
                y[t] = 0.9 * y[t - 1] + 0.44 * ey;
            }
        }
        let data = series_from(x, y);
        let vars = [VariableId::from_raw(0), VariableId::from_raw(1)];

        let lagged_self_parent = |v: u32| {
            let mut g = TemporalCpdag::empty();
            let past = g.add_lagged(VariableId::from_raw(v), Lag::from_raw(1)).unwrap();
            let now = g.add_lagged(VariableId::from_raw(v), Lag::CONTEMPORANEOUS).unwrap();
            g.insert_directed(past, now).unwrap();
            g
        };
        let graphs = RegimeGraphCollection {
            graphs: Arc::from(vec![
                (RegimeId::from_raw(0), lagged_self_parent(0)),
                (RegimeId::from_raw(1), lagged_self_parent(1)),
            ]),
        };
        let current = two_regime_half_split(n);
        let refined = reassign_by_lagged_residual(&data, &vars, &graphs, &current, 1.0)
            .unwrap()
            .expect("two identified regimes");
        let agree = (1..n).filter(|&t| refined.regimes[t] == current.regimes[t]).count();
        assert!(
            agree * 100 >= (n - 1) * 90,
            "only {agree}/{} rows kept their true regime; labels={:?}",
            n - 1,
            refined.regimes
        );
        let switches = refined.regimes.windows(2).filter(|w| w[0] != w[1]).count();
        assert!(switches <= 6, "switch penalty must keep labels contiguous, got {switches}");
    }

    /// Without a switch penalty labels chatter; with one they stay contiguous.
    #[test]
    fn switch_penalty_suppresses_chatter() {
        let n = 200usize;
        let mut x = vec![0.0; n];
        let mut y = vec![0.0; n];
        for t in 1..n {
            let tf = t as f64;
            x[t] = 0.5 * x[t - 1] + (tf * 0.7).sin();
            y[t] = if t < n / 2 { 0.2 * x[t - 1] } else { 4.0 * x[t - 1] } + 0.3 * (tf * 1.3).cos();
        }
        let data = series_from(x, y);
        let vars = [VariableId::from_raw(0), VariableId::from_raw(1)];
        let mk = |v_from: u32, v_to: u32| {
            let mut g = TemporalCpdag::empty();
            let past = g.add_lagged(VariableId::from_raw(v_from), Lag::from_raw(1)).unwrap();
            let now = g.add_lagged(VariableId::from_raw(v_to), Lag::CONTEMPORANEOUS).unwrap();
            g.insert_directed(past, now).unwrap();
            g
        };
        let graphs = RegimeGraphCollection {
            graphs: Arc::from(vec![
                (RegimeId::from_raw(0), mk(0, 1)),
                (RegimeId::from_raw(1), mk(0, 1)),
            ]),
        };
        let current = two_regime_half_split(n);
        let smooth = reassign_by_lagged_residual(&data, &vars, &graphs, &current, 3.0)
            .unwrap()
            .expect("two identified regimes");
        let free = reassign_by_lagged_residual(&data, &vars, &graphs, &current, 0.0)
            .unwrap()
            .expect("two identified regimes");
        let count = |a: &RegimeAssignment| a.regimes.windows(2).filter(|w| w[0] != w[1]).count();
        assert!(count(&smooth) <= count(&free));
        assert!(count(&smooth) <= 4, "penalised labels switched {} times", count(&smooth));
    }
}
