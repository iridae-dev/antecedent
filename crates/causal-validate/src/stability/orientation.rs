//! Orientation stability via PCMCI+ block bootstrap.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::collections::BTreeMap;
use std::sync::Arc;

use causal_core::{ExecutionContext, Lag, VariableId};
use causal_data::{ResamplingPlan, TableView, TimeSeriesData, resample_timeseries};
use causal_discovery::{DiscoveryWorkspace, LaggedLink, PcmciPlus};
use causal_graph::{DenseNodeId, Endpoint, NodeRef};

use crate::error::ValidationError;

use super::pcmci_grid::{LinkStability, report_from_counts};

/// Undirected contemporaneous edge retention frequency.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UndirectedLinkStability {
    /// Endpoints with `a.raw() <= b.raw()`.
    pub a: VariableId,
    /// Other endpoint.
    pub b: VariableId,
    /// Fraction of replicates retaining an undirected contemp edge.
    pub frequency: f64,
}

/// Report from [`OrientationStability`].
#[derive(Clone, Debug)]
pub struct OrientationStabilityReport {
    /// Directed contemporaneous edge frequencies.
    pub directed: Arc<[LinkStability]>,
    /// Undirected contemporaneous edge frequencies.
    pub undirected: Arc<[UndirectedLinkStability]>,
    /// Fraction of replicates that produced ≥1 conflict edge among contemp pairs.
    pub conflict_rate: f64,
    /// Bootstrap replicates.
    pub replicates: u32,
    /// Moving-block length.
    pub block_size: usize,
}

/// Block-bootstrap orientation stability around a [`PcmciPlus`] configuration.
#[derive(Clone, Debug)]
pub struct OrientationStability {
    /// PCMCI+ configuration to re-run.
    pub pcmci_plus: PcmciPlus,
    /// Bootstrap replicates.
    pub replicates: u32,
    /// Block length.
    pub block_size: usize,
}

impl Default for OrientationStability {
    fn default() -> Self {
        Self::new()
    }
}

impl OrientationStability {
    /// Defaults: 20 replicates, block size 20, FDR off.
    #[must_use]
    pub fn new() -> Self {
        Self { pcmci_plus: PcmciPlus::new().with_fdr(false), replicates: 20, block_size: 20 }
    }

    /// Run orientation stability assessment.
    ///
    /// # Errors
    ///
    /// Data or discovery failures.
    pub fn run(
        &self,
        data: &TimeSeriesData,
        variables: &[VariableId],
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<OrientationStabilityReport, ValidationError> {
        if self.replicates == 0 || self.block_size == 0 {
            return Err(ValidationError::NotApplicable {
                message: "orientation stability requires positive replicates and block_size",
            });
        }
        if self.block_size > data.row_count() {
            return Err(ValidationError::NotApplicable {
                message: "block_size exceeds series length",
            });
        }
        let mut directed_counts: BTreeMap<LaggedLink, u32> = BTreeMap::new();
        let mut undirected_counts: BTreeMap<(VariableId, VariableId), u32> = BTreeMap::new();
        let mut conflict_reps = 0u32;
        let mut rng = ctx.rng.stream(0x0E1E_u64);
        let mut index_scratch = Vec::new();
        for _ in 0..self.replicates {
            let boot = resample_timeseries(
                data,
                ResamplingPlan::MovingBlock { length: self.block_size },
                &mut rng,
                &mut index_scratch,
            )
            .map_err(ValidationError::from)?;
            let result = self
                .pcmci_plus
                .run(&boot, variables, workspace, ctx)
                .map_err(ValidationError::from)?;
            let mut had_conflict = false;
            for edge in result.evidence.graph.edges() {
                let (Some(va), Some(vb)) = (
                    lagged_var_lag0(result.evidence.graph.nodes(), edge.a),
                    lagged_var_lag0(result.evidence.graph.nodes(), edge.b),
                ) else {
                    continue;
                };
                if edge.is_conflict() {
                    had_conflict = true;
                    continue;
                }
                if edge.is_undirected() {
                    let (a, b) = if va.raw() <= vb.raw() { (va, vb) } else { (vb, va) };
                    *undirected_counts.entry((a, b)).or_insert(0) += 1;
                    continue;
                }
                if edge.is_dag_directed() {
                    let (src, tgt) = match (edge.at_a, edge.at_b) {
                        (Endpoint::Tail, Endpoint::Arrow) => (va, vb),
                        (Endpoint::Arrow, Endpoint::Tail) => (vb, va),
                        _ => continue,
                    };
                    let link = LaggedLink {
                        source: src,
                        source_lag: Lag::CONTEMPORANEOUS,
                        target: tgt,
                        target_lag: Lag::CONTEMPORANEOUS,
                    };
                    *directed_counts.entry(link).or_insert(0) += 1;
                }
            }
            if had_conflict {
                conflict_reps += 1;
            }
        }
        let directed = report_from_counts(directed_counts, self.replicates, self.block_size);
        let mut undirected = Vec::with_capacity(undirected_counts.len());
        for ((a, b), c) in undirected_counts {
            undirected.push(UndirectedLinkStability {
                a,
                b,
                frequency: f64::from(c) / f64::from(self.replicates),
            });
        }
        undirected.sort_by(|x, y| {
            y.frequency.partial_cmp(&x.frequency).unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(OrientationStabilityReport {
            directed: directed.frequencies,
            undirected: Arc::from(undirected),
            conflict_rate: f64::from(conflict_reps) / f64::from(self.replicates),
            replicates: self.replicates,
            block_size: self.block_size,
        })
    }
}

fn lagged_var_lag0(nodes: &[NodeRef], id: DenseNodeId) -> Option<VariableId> {
    let node = nodes.get(id.as_usize())?;
    match node {
        NodeRef::Lagged { variable, lag } if lag.raw() == 0 => Some(*variable),
        _ => None,
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use std::sync::Arc;

    use causal_core::{
        CausalSchemaBuilder, ExecutionContext, Lag, MeasurementSpec, RoleHint, SmallRoleSet,
        ValueType, VariableId,
    };
    use causal_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
        TimeSeriesData, ValidityBitmap,
    };
    use causal_discovery::{DiscoveryConstraints, DiscoveryWorkspace, TemporalConstraints};

    use super::*;

    fn contemp_chain() -> (TimeSeriesData, Vec<VariableId>) {
        let n = 250usize;
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
        for t in 0..n {
            x[t] = (t as f64 * 0.03).sin();
            y[t] = 0.85 * x[t] + 0.05 * (t as f64 * 0.07).cos();
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
        (data, vec![VariableId::from_raw(0), VariableId::from_raw(1)])
    }

    #[test]
    fn contemp_dependence_appears_in_orientation_report() {
        let (data, vars) = contemp_chain();
        let constraints = DiscoveryConstraints {
            temporal: TemporalConstraints {
                max_lag: Lag::from_raw(1),
                min_lag: Lag::CONTEMPORANEOUS,
            },
            max_cond_size: 1,
            alpha: 0.1,
            ..Default::default()
        };
        let stab = OrientationStability {
            pcmci_plus: PcmciPlus::new().with_fdr(false).with_constraints(constraints),
            replicates: 6,
            block_size: 30,
        };
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(9);
        let report = stab.run(&data, &vars, &mut ws, &ctx).unwrap();
        assert_eq!(report.replicates, 6);
        let any = report.directed.iter().any(|l| l.frequency > 0.0)
            || report.undirected.iter().any(|l| l.frequency > 0.0);
        assert!(
            any,
            "expected contemp edge retention; directed={:?} undirected={:?}",
            report.directed, report.undirected
        );
    }
}
