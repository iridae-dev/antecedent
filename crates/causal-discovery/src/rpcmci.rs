//! RPCMCI: regime-PCMCI with typed assignments and per-regime graphs.
//!
//! Per-regime discovery keeps the full series lag alignment and retains only
//! effective samples whose entire lag window lies inside the regime (masked CI).
//! Optional alternating assignment refines labels by residual fit under each
//! regime's discovered lagged parents.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::needless_range_loop
)]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use causal_core::{ExecutionContext, RegimeId, VariableId};
use causal_data::{ColumnView, LaggedFrame, TableView, TimeSeriesData};
use causal_graph::TemporalCpdag;
use causal_stats::ConditionalIndependence;

use crate::engine::DiscoveryWorkspace;
use crate::error::DiscoveryError;
use crate::pcmci_plus::PcmciPlus;
use crate::result::{AlgorithmRecord, CpdagDiscoveryResult, DiscoveryDiagnostic};

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

/// Regime-PCMCI discovery.
#[derive(Clone, Debug)]
pub struct Rpcmci {
    /// Nested PCMCI+.
    pub pcmci_plus: PcmciPlus,
    /// Minimum regime length (raw rows) to discover.
    pub min_regime_len: usize,
    /// Alternating assignment iterations (`0` = fixed labels only).
    pub alternating_iters: usize,
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
            let Some(updated) =
                reassign_by_lag1_residual(data, variables, &last.graphs, &assignment)?
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
            assignment = updated;
            last = self.discover_regimes(data, variables, &assignment, workspace, ctx)?;
            diagnostics.push(DiscoveryDiagnostic {
                code: Arc::from("rpcmci.alternating"),
                message: Arc::from(format!("completed alternating refinement {}", iter + 1)),
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

        let algorithm = AlgorithmRecord {
            id: Arc::from("rpcmci"),
            config: Arc::from(format!(
                "regimes={},min_len={},nested={},alternating={}",
                graphs.len(),
                self.min_regime_len,
                self.pcmci_plus.engine().constraints.temporal.max_lag.raw(),
                self.alternating_iters
            )),
        };
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

/// Reassign each time point to the regime whose retained lag-1 links best predict
/// contemporaneous values (sum of squared lag-1 residuals). Returns `None` if no
/// regime has usable links.
fn reassign_by_lag1_residual(
    data: &TimeSeriesData,
    variables: &[VariableId],
    graphs: &RegimeGraphCollection,
    current: &RegimeAssignment,
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

    let mut regime_links: Vec<(RegimeId, Vec<(usize, usize)>)> = Vec::new();
    for (regime, g) in graphs.graphs.iter() {
        let mut links = Vec::new();
        for (i, node) in g.nodes().iter().enumerate() {
            let causal_graph::NodeRef::Lagged { variable: tgt, lag: tlag } = node else {
                continue;
            };
            if !tlag.is_contemporaneous() {
                continue;
            }
            let Some(ti) = variables.iter().position(|v| v == tgt) else {
                continue;
            };
            let from = causal_graph::DenseNodeId::from_raw(i as u32);
            for p in g.parents(from) {
                if let Some(causal_graph::NodeRef::Lagged { variable: src, lag: slag }) =
                    g.nodes().get(p.as_usize())
                {
                    if slag.raw() == 1 {
                        if let Some(si) = variables.iter().position(|v| v == src) {
                            links.push((ti, si));
                        }
                    }
                }
            }
        }
        regime_links.push((*regime, links));
    }
    if regime_links.iter().all(|(_, l)| l.is_empty()) {
        return Ok(None);
    }

    let mut out = current.regimes.to_vec();
    for t in 1..n {
        let mut best_r = out[t];
        let mut best_err = f64::INFINITY;
        for (regime, links) in &regime_links {
            if links.is_empty() {
                continue;
            }
            let mut err = 0.0;
            for &(ti, si) in links {
                let pred = cols[si][t - 1];
                let resid = cols[ti][t] - pred;
                err += resid * resid;
            }
            err /= links.len() as f64;
            if err < best_err {
                best_err = err;
                best_r = *regime;
            }
        }
        out[t] = best_r;
    }
    out[0] = out[1];
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
    use causal_core::{
        CausalSchemaBuilder, Lag, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
    };
    use causal_data::{
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

    #[test]
    fn regime_window_mask_rejects_boundary_crossing() {
        let assign = two_regime_half_split(10);
        let keep = regime_window_mask(&assign, RegimeId::from_raw(0), 2, 10);
        assert_eq!(keep.len(), 8);
        assert!(keep[0]);
        assert!(keep[2]);
        assert!(!keep[3]);
    }
}
