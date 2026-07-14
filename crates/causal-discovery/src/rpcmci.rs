//! RPCMCI: regime-PCMCI with typed assignments and per-regime graphs (Phase 9).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use causal_core::{ExecutionContext, RegimeId, VariableId};
use causal_data::{
    ColumnView, Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TableView,
    TimeIndex, TimeSeriesData, ValidityBitmap,
};
use causal_graph::TemporalCpdag;
use causal_stats::ConditionalIndependence;

use crate::engine::DiscoveryWorkspace;
use crate::error::DiscoveryError;
use crate::pcmci_plus::PcmciPlus;
use crate::result::{AlgorithmRecord, CpdagDiscoveryResult, DiscoveryDiagnostic};

/// Columnar regime label per time index (DESIGN.md §13.5).
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
        self.regimes
            .iter()
            .enumerate()
            .filter_map(|(i, &r)| (r == regime).then_some(i))
            .collect()
    }
}

/// One temporal CPDAG (or equivalent) per regime — never collapsed to a single graph.
#[derive(Clone, Debug)]
pub struct RegimeGraphCollection {
    /// Ordered `(regime, graph)` pairs.
    pub graphs: Arc<[(RegimeId, TemporalCpdag)]>,
}

impl RegimeGraphCollection {
    /// Number of regimes with a graph.
    #[must_use]
    pub fn len(&self) -> usize {
        self.graphs.len()
    }

    /// Whether empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.graphs.is_empty()
    }

    /// Borrow graph for `regime`, if present.
    #[must_use]
    pub fn get(&self, regime: RegimeId) -> Option<&TemporalCpdag> {
        self.graphs.iter().find(|(r, _)| *r == regime).map(|(_, g)| g)
    }
}

/// Full RPCMCI result: assignments + per-regime discovery artifacts.
#[derive(Clone, Debug)]
pub struct RpcmciDiscoveryResult {
    /// Regime labels along the series.
    pub assignments: RegimeAssignment,
    /// One oriented CPDAG per regime.
    pub graphs: RegimeGraphCollection,
    /// Per-regime full discovery results (aligned with [`RegimeGraphCollection::graphs`]).
    pub per_regime: Arc<[CpdagDiscoveryResult]>,
    /// Algorithm metadata.
    pub algorithm: AlgorithmRecord,
    /// Diagnostics.
    pub diagnostics: Vec<DiscoveryDiagnostic>,
}

/// Regime-PCMCI discovery (own type; not a PCMCI flag).
#[derive(Clone, Debug)]
pub struct Rpcmci {
    /// Underlying PCMCI+ runner used per regime segment.
    pub pcmci_plus: PcmciPlus,
    /// Minimum rows required in a regime to attempt discovery.
    pub min_regime_len: usize,
}

impl Default for Rpcmci {
    fn default() -> Self {
        Self::new()
    }
}

impl Rpcmci {
    /// Default RPCMCI wrapping [`PcmciPlus::new`].
    #[must_use]
    pub fn new() -> Self {
        Self { pcmci_plus: PcmciPlus::new(), min_regime_len: 32 }
    }

    /// Replace the PCMCI+ configuration.
    #[must_use]
    pub fn with_pcmci_plus(mut self, pcmci_plus: PcmciPlus) -> Self {
        self.pcmci_plus = pcmci_plus;
        self
    }

    /// Minimum regime length.
    #[must_use]
    pub fn with_min_regime_len(mut self, min_regime_len: usize) -> Self {
        self.min_regime_len = min_regime_len;
        self
    }

    /// Replace CI on the nested engine.
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
        let regimes = assignments.unique_regimes();
        if regimes.is_empty() {
            return Err(DiscoveryError::Unsupported {
                message: "RPCMCI needs ≥1 distinct regime",
            });
        }

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
            let subset = subset_series(data, &idxs)?;
            let result = self.pcmci_plus.run(&subset, variables, workspace, ctx)?;
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
                "regimes={},min_len={},nested={}",
                graphs.len(),
                self.min_regime_len,
                self.pcmci_plus.engine.constraints.temporal.max_lag.raw()
            )),
        };
        diagnostics.push(DiscoveryDiagnostic {
            code: Arc::from("rpcmci.graphs"),
            message: Arc::from(format!(
                "produced {} per-regime temporal CPDAGs",
                graphs.len()
            )),
        });

        Ok(RpcmciDiscoveryResult {
            assignments: assignments.clone(),
            graphs: RegimeGraphCollection { graphs: Arc::from(graphs) },
            per_regime: Arc::from(per_regime),
            algorithm,
            diagnostics,
        })
    }

    /// Infer a two-regime assignment by median split on `indicator`, then discover.
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

fn median_split_assignment(
    data: &TimeSeriesData,
    indicator: VariableId,
) -> Result<RegimeAssignment, DiscoveryError> {
    let ColumnView::Float64(col) = data.column(indicator).map_err(|e| {
        DiscoveryError::data_msg(format!("regime indicator: {e}"))
    })? else {
        return Err(DiscoveryError::Unsupported {
            message: "regime indicator must be float64",
        });
    };
    let mut sorted: Vec<f64> = col.values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted[sorted.len() / 2];
    let regimes: Vec<RegimeId> = col
        .values
        .iter()
        .map(|&v| {
            if v <= mid {
                RegimeId::from_raw(0)
            } else {
                RegimeId::from_raw(1)
            }
        })
        .collect();
    RegimeAssignment::try_new(Arc::from(regimes))
}

fn subset_series(data: &TimeSeriesData, idxs: &[usize]) -> Result<TimeSeriesData, DiscoveryError> {
    let n = idxs.len();
    let schema = data.schema().clone();
    let mut cols = Vec::with_capacity(schema.len());
    for i in 0..schema.len() {
        let id = VariableId::from_raw(i as u32);
        let ColumnView::Float64(src) = data.column(id).map_err(|e| {
            DiscoveryError::data_msg(format!("subset column: {e}"))
        })? else {
            return Err(DiscoveryError::Unsupported {
                message: "RPCMCI subset currently supports float64 columns only",
            });
        };
        let values: Vec<f64> = idxs.iter().map(|&r| src.values[r]).collect();
        cols.push(OwnedColumn::Float64(
            Float64Column::new(id, Arc::from(values), ValidityBitmap::all_valid(n)).map_err(
                |e| DiscoveryError::data_msg(format!("subset float column: {e}")),
            )?,
        ));
    }
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None)
        .map_err(|e| DiscoveryError::data_msg(format!("subset storage: {e}")))?;
    TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .map_err(|e| DiscoveryError::data_msg(format!("subset series: {e}")))
}

/// Seed helper for regime discovery benches: build a two-regime assignment map.
#[must_use]
pub fn two_regime_half_split(series_len: usize) -> RegimeAssignment {
    let mid = series_len / 2;
    let regimes: Vec<RegimeId> = (0..series_len)
        .map(|t| {
            if t < mid {
                RegimeId::from_raw(0)
            } else {
                RegimeId::from_raw(1)
            }
        })
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
    use causal_core::{
        CausalSchemaBuilder, ExecutionContext, Lag, MeasurementSpec, RoleHint, SmallRoleSet,
        ValueType,
    };
    use causal_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
        TimeSeriesData, ValidityBitmap,
    };

    use super::*;
    use crate::constraints::{DiscoveryConstraints, TemporalConstraints};
    use crate::pcmci_plus::PcmciPlus;

    fn two_regime_series(n: usize) -> (TimeSeriesData, Vec<VariableId>, RegimeAssignment) {
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
            let a = if t < mid { 0.8 } else { 0.2 };
            x[t] = 0.4 * x[t - 1] + 0.1 * (t as f64).sin();
            y[t] = a * x[t] + 0.15 * y[t - 1] + 0.05 * (t as f64).cos();
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
        let data = TimeSeriesData::try_new(
            storage,
            TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
        )
        .unwrap();
        let assign = two_regime_half_split(n);
        (data, vec![VariableId::from_raw(0), VariableId::from_raw(1)], assign)
    }

    #[test]
    fn rpcmci_returns_one_graph_per_regime() {
        let (data, vars, assign) = two_regime_series(200);
        let algo = Rpcmci::new()
            .with_min_regime_len(40)
            .with_pcmci_plus(PcmciPlus::new().with_fdr(false).with_constraints(
                DiscoveryConstraints {
                    temporal: TemporalConstraints {
                        max_lag: Lag::from_raw(1),
                        min_lag: causal_core::Lag::CONTEMPORANEOUS,
                    },
                    alpha: 0.25,
                    max_cond_size: 2,
                    ..DiscoveryConstraints::default()
                },
            ));
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(3);
        let result = algo.run(&data, &vars, &assign, &mut ws, &ctx).unwrap();
        assert_eq!(result.algorithm.id.as_ref(), "rpcmci");
        assert_eq!(result.graphs.len(), 2);
        assert!(result.graphs.get(RegimeId::from_raw(0)).is_some());
        assert!(result.graphs.get(RegimeId::from_raw(1)).is_some());
        assert_eq!(result.per_regime.len(), 2);
    }
}
