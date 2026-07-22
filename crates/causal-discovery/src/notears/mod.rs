//! NOTEARS continuous SEM discovery → static [`Dag`] (Zheng et al. 2018).
//!
//! Least-squares linear SEM loss + smooth exact acyclicity \(h(W)=\operatorname{tr}(e^{W\circ W})-d\),
//! solved by augmented Lagrangian with native L-BFGS (documented equivalent of
//! L-BFGS-B with diagonal/forbidden entries fixed at zero by parameter packing).
//!
//! # Scale / varsortability
//!
//! NOTEARS is scale-sensitive (varsortability). By default [`Notears::standardize`]
//! is `true`: each continuous column is mean/sd standardized under
//! [`ExecutionContext::kernel_policy`] before optimization. Disable only when
//! columns are already on a comparable scale and you accept the varsortability risk.
//!
//! # Weight convention
//!
//! Dense soft weights are row-major \(d\times d\) with \(W_{ij}\) the weight of
//! directed edge \(i\to j\) (NOTEARS / Zheng et al.). Diagonal is always zero.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::too_many_lines
)]

mod acyclicity;
mod solver;

use std::sync::Arc;

use causal_core::{AssumptionSet, ExecutionContext, Lag, VariableId};
use causal_data::TabularData;
use causal_graph::{Dag, DagReview, DenseNodeId, NodeRef};
use causal_stats::standardize_columns;

use crate::constraints::DiscoveryConstraints;
use crate::engine::DiscoveryWorkspace;
use crate::error::DiscoveryError;
use crate::lingam::StaticDagDiscoveryResult;
use crate::pc::collect_float_columns;
use crate::result::{
    AlgorithmRecord, DiscoveryDiagnostic, DiscoveryIteration, DiscoveryPerformanceRecord,
    DiscoveryResult, EdgeEvidence, EvidenceSource, GraphEvidence, LaggedLink, ScoredLink,
};

use solver::{NotearsWorkspace, SolverConfig, solve_notears};

/// NOTEARS discovery result: hard DAG review plus soft weight matrix for mechanism seeding.
#[derive(Clone, Debug)]
pub struct NotearsDiscoveryResult {
    /// Thresholded DAG + review / edge evidence.
    pub discovery: StaticDagDiscoveryResult,
    /// Dense \(d\times d\) soft SEM weights before thresholding (row-major).
    ///
    /// \(W_{ij}\) = weight of edge \(i\to j\). Diagonal is zero.
    pub weights: Arc<[f64]>,
    /// Dimension \(d\) (`weights.len() == dim * dim`).
    pub dim: usize,
}

/// Continuous SEM discovery (NOTEARS) over tabular continuous data.
#[derive(Clone, Debug)]
pub struct Notears {
    /// Constraints / max parents / forbidden / required edges.
    pub constraints: DiscoveryConstraints,
    /// L1 penalty \(\lambda\) (default `0.1`).
    pub lambda: f64,
    /// Absolute soft-weight threshold for the hard DAG (default `0.3`).
    pub threshold: f64,
    /// Standardize columns before solving (default `true`; see module docs).
    pub standardize: bool,
    /// Maximum augmented-Lagrangian outer iterations (default `100`).
    pub max_iter: u32,
    /// Convergence tolerance on \(|h(W)|\) (default `1e-8`).
    pub h_tol: f64,
    /// Maximum AL penalty \(\rho\) (default `1e16`).
    pub rho_max: f64,
}

impl Default for Notears {
    fn default() -> Self {
        Self::new()
    }
}

impl Notears {
    /// Defaults matching common NOTEARS settings (`λ=0.1`, threshold `0.3`, standardize).
    #[must_use]
    pub fn new() -> Self {
        Self {
            constraints: DiscoveryConstraints {
                temporal: crate::constraints::TemporalConstraints {
                    max_lag: Lag::CONTEMPORANEOUS,
                    min_lag: Lag::CONTEMPORANEOUS,
                },
                ..DiscoveryConstraints::default()
            },
            lambda: 0.1,
            threshold: 0.3,
            standardize: true,
            max_iter: 100,
            h_tol: 1e-8,
            rho_max: 1e16,
        }
    }

    /// Configure constraints.
    #[must_use]
    pub fn with_constraints(mut self, constraints: DiscoveryConstraints) -> Self {
        self.constraints = constraints;
        self
    }

    /// L1 penalty.
    #[must_use]
    pub fn with_lambda(mut self, lambda: f64) -> Self {
        self.lambda = lambda;
        self
    }

    /// Hard-DAG absolute weight threshold.
    #[must_use]
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Whether to standardize columns (varsortability policy).
    #[must_use]
    pub fn with_standardize(mut self, standardize: bool) -> Self {
        self.standardize = standardize;
        self
    }

    /// AL outer iteration cap.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: u32) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Run NOTEARS.
    ///
    /// # Errors
    ///
    /// Data, numerical non-convergence / non-finite values, or graph failures.
    /// Fail-closed: never returns an empty soft failure DAG on solver error.
    pub fn run(
        &self,
        data: &TabularData,
        variables: &[VariableId],
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<NotearsDiscoveryResult, DiscoveryError> {
        let _ = workspace;
        self.constraints.validate()?;
        if variables.is_empty() {
            return Err(DiscoveryError::Unsupported {
                message: "NOTEARS requires at least one variable",
            });
        }
        if !self.lambda.is_finite() || self.lambda < 0.0 {
            return Err(DiscoveryError::unsupported("NOTEARS lambda must be finite and >= 0"));
        }
        if !self.threshold.is_finite() || self.threshold < 0.0 {
            return Err(DiscoveryError::unsupported("NOTEARS threshold must be finite and >= 0"));
        }

        let col_owned = collect_float_columns(data, variables)?;
        let n = col_owned[0].len();
        if n < 3 {
            return Err(DiscoveryError::stats_msg("insufficient rows for NOTEARS"));
        }
        for c in &col_owned {
            if c.len() != n {
                return Err(DiscoveryError::data_msg("column length mismatch"));
            }
        }

        let d = variables.len();
        // Column-major n×d design.
        let mut x = vec![0.0; n * d];
        for (j, col) in col_owned.iter().enumerate() {
            x[j * n..(j + 1) * n].copy_from_slice(col.as_ref());
        }
        if self.standardize {
            let cols: Vec<usize> = (0..d).collect();
            standardize_columns(&mut x, n, d, &cols, 1e-12, &ctx.kernel_policy)
                .map_err(DiscoveryError::from)?;
        }

        // Freeze diagonal + forbidden / tier-forbidden.
        let mut frozen = vec![false; d * d];
        for i in 0..d {
            frozen[i * d + i] = true;
        }
        for i in 0..d {
            for j in 0..d {
                if i == j {
                    continue;
                }
                if forbidden_edge(&self.constraints, variables, i, j) {
                    frozen[i * d + j] = true;
                }
            }
        }

        let cfg = SolverConfig {
            lambda: self.lambda,
            max_iter: self.max_iter,
            h_tol: self.h_tol,
            rho_max: self.rho_max,
            ..SolverConfig::default()
        };
        let mut solver_ws = NotearsWorkspace::default();
        let soft_w = solve_notears(&x, n, d, &frozen, &cfg, &mut solver_ws)
            .map_err(DiscoveryError::stats_msg)?;

        // Threshold → candidate edges (parent i → child j when |W_ij| >= threshold).
        let max_parents = self.constraints.max_parents.unwrap_or(d.saturating_sub(1));
        let mut by_child: Vec<Vec<(usize, f64)>> = vec![Vec::new(); d];
        for i in 0..d {
            for j in 0..d {
                if i == j || frozen[i * d + j] {
                    continue;
                }
                let w = soft_w[i * d + j];
                if w.abs() >= self.threshold {
                    by_child[j].push((i, w));
                }
            }
        }
        for parents in &mut by_child {
            parents.sort_by(|a, b| {
                b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal)
            });
            if parents.len() > max_parents {
                parents.truncate(max_parents);
            }
        }

        let mut edge_coefs: Vec<(usize, usize, f64)> = Vec::new();
        for (child, parents) in by_child.iter().enumerate() {
            for &(par, w) in parents {
                edge_coefs.push((par, child, w));
            }
        }

        // Seed required edges (contemporaneous) if both endpoints exist.
        for r in self.constraints.required.iter() {
            if r.source_lag != Lag::CONTEMPORANEOUS || r.target_lag != Lag::CONTEMPORANEOUS {
                continue;
            }
            let Some(si) = variables.iter().position(|v| *v == r.source) else {
                continue;
            };
            let Some(ti) = variables.iter().position(|v| *v == r.target) else {
                continue;
            };
            if !edge_coefs.iter().any(|&(a, b, _)| a == si && b == ti) {
                let w = soft_w[si * d + ti];
                edge_coefs.push((si, ti, if w.abs() > 0.0 { w } else { 1.0 }));
            }
        }

        // Insert by decreasing |weight| so stronger edges win if a residual cycle appears.
        edge_coefs
            .sort_by(|a, b| b.2.abs().partial_cmp(&a.2.abs()).unwrap_or(std::cmp::Ordering::Equal));

        let mut dag = Dag::empty();
        for &v in variables {
            dag.add_node(NodeRef::Static(v))?;
        }
        let mut kept: Vec<(usize, usize, f64)> = Vec::new();
        for &(par, child, w) in &edge_coefs {
            let from = DenseNodeId::from_raw(par as u32);
            let to = DenseNodeId::from_raw(child as u32);
            if dag.children(from).contains(&to) {
                continue;
            }
            match dag.insert_directed(from, to) {
                Ok(()) => kept.push((par, child, w)),
                Err(causal_graph::GraphError::Cycle { .. }) => {
                    return Err(DiscoveryError::stats_msg(
                        "NOTEARS thresholded weights induce a cycle (solver |h| tolerance vs threshold mismatch)",
                    ));
                }
                Err(e) => return Err(DiscoveryError::from(e)),
            }
        }

        let edge_evidence: Vec<EdgeEvidence> = kept
            .iter()
            .map(|&(par, child, w)| EdgeEvidence {
                link: LaggedLink {
                    source: variables[par],
                    source_lag: Lag::CONTEMPORANEOUS,
                    target: variables[child],
                    target_lag: Lag::CONTEMPORANEOUS,
                },
                statistic: Some(w),
                p_value: None,
                adjusted_p_value: None,
                interval: None,
                separating_sets: Arc::from([]),
                provenance: Arc::from([Arc::from("notears")]),
            })
            .collect();

        let links: Vec<ScoredLink> = edge_evidence
            .iter()
            .map(|e| ScoredLink {
                link: e.link,
                statistic: e.statistic.unwrap_or(0.0),
                p_value: 1.0,
                adjusted_p_value: None,
            })
            .collect();

        let edge_count = dag.edges().count();
        let review = DagReview::from_dag(dag.clone(), "notears");
        let evidence = GraphEvidence {
            graph: dag,
            edge_evidence: Arc::from(edge_evidence),
            links: Arc::from(links),
            source: EvidenceSource::Discovery { algorithm: Arc::from("notears") },
        };

        let discovery = DiscoveryResult {
            evidence,
            review,
            algorithm: AlgorithmRecord {
                id: Arc::from("notears"),
                config: Arc::from(format!(
                    "lambda={} threshold={} standardize={} max_iter={} h_tol={}",
                    self.lambda, self.threshold, self.standardize, self.max_iter, self.h_tol
                )),
            },
            assumptions: AssumptionSet::default(),
            iterations: Vec::<DiscoveryIteration>::new(),
            diagnostics: Vec::<DiscoveryDiagnostic>::new(),
            performance: DiscoveryPerformanceRecord {
                ci_tests: 0,
                links_retained: u64::try_from(edge_count).unwrap_or(u64::MAX),
                targets: u64::try_from(d).unwrap_or(u64::MAX),
                lagged_frame_bytes: 0,
                worker_threads: 1,
            },
            sepsets: crate::result::PcSepsets::default(),
        };

        Ok(NotearsDiscoveryResult { discovery, weights: Arc::from(soft_w), dim: d })
    }
}

fn forbidden_edge(
    constraints: &DiscoveryConstraints,
    variables: &[VariableId],
    par: usize,
    child: usize,
) -> bool {
    let Some(&src) = variables.get(par) else {
        return true;
    };
    let Some(&tgt) = variables.get(child) else {
        return true;
    };
    let link = LaggedLink {
        source: src,
        source_lag: Lag::CONTEMPORANEOUS,
        target: tgt,
        target_lag: Lag::CONTEMPORANEOUS,
    };
    constraints.is_forbidden(link) || constraints.tier_forbids(src, tgt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use causal_core::{
        CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType, VariableId,
    };
    use causal_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TableView, TabularData, ValidityBitmap,
    };

    /// Linear Gaussian SEM: X0 → X1 → X2.
    fn linear_chain(n: usize) -> (TabularData, Vec<VariableId>) {
        let mut b = CausalSchemaBuilder::new();
        for i in 0..3 {
            b.add_variable(
                format!("x{i}"),
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        }
        let schema = b.build().unwrap();
        let mut x0 = vec![0.0; n];
        let mut x1 = vec![0.0; n];
        let mut x2 = vec![0.0; n];
        for i in 0..n {
            // Deterministic pseudo-noise in (-0.5, 0.5).
            let e0 = ((i as f64 * 0.137) % 1.0) - 0.5;
            let e1 = ((i as f64 * 0.271) % 1.0) - 0.5;
            let e2 = ((i as f64 * 0.419) % 1.0) - 0.5;
            x0[i] = e0;
            x1[i] = 0.8 * x0[i] + e1;
            x2[i] = 0.8 * x1[i] + e2;
        }
        let owned = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(x0),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(x1),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(2),
                    Arc::from(x2),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, owned, None, None).unwrap();
        let data = TabularData::new(storage);
        let vars: Vec<_> = data.schema().variables().iter().map(|v| v.id).collect();
        (data, vars)
    }

    #[test]
    fn notears_recovers_chain_skeleton() {
        let (data, vars) = linear_chain(800);
        let alg = Notears::new().with_lambda(0.05).with_threshold(0.2);
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let result = alg.run(&data, &vars, &mut ws, &ctx).unwrap();
        assert_eq!(result.discovery.algorithm.id.as_ref(), "notears");
        assert_eq!(result.dim, 3);
        assert_eq!(result.weights.len(), 9);
        let g = &result.discovery.evidence.graph;
        let d = |i: u32| DenseNodeId::from_raw(i);
        assert!(
            g.children(d(0)).contains(&d(1)),
            "expected 0→1, edges={:?}",
            g.edges().collect::<Vec<_>>()
        );
        assert!(
            g.children(d(1)).contains(&d(2)),
            "expected 1→2, edges={:?}",
            g.edges().collect::<Vec<_>>()
        );
        // Soft weights for true edges should be non-trivial.
        assert!(result.weights[1].abs() > 0.1);
        assert!(result.weights[3 + 2].abs() > 0.1);
    }

    #[test]
    fn forbidden_edge_absent() {
        let (data, vars) = linear_chain(600);
        let mut constraints = DiscoveryConstraints {
            temporal: crate::constraints::TemporalConstraints {
                max_lag: Lag::CONTEMPORANEOUS,
                min_lag: Lag::CONTEMPORANEOUS,
            },
            ..DiscoveryConstraints::default()
        };
        constraints.forbidden = Arc::from([LaggedLink {
            source: vars[0],
            source_lag: Lag::CONTEMPORANEOUS,
            target: vars[1],
            target_lag: Lag::CONTEMPORANEOUS,
        }]);
        let alg =
            Notears::new().with_constraints(constraints).with_lambda(0.05).with_threshold(0.15);
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let result = alg.run(&data, &vars, &mut ws, &ctx).unwrap();
        let g = &result.discovery.evidence.graph;
        let d = |i: u32| DenseNodeId::from_raw(i);
        assert!(!g.children(d(0)).contains(&d(1)), "forbidden 0→1 must be absent");
        assert_eq!(result.weights[1], 0.0);
    }

    #[test]
    fn nan_columns_fail_closed() {
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "a",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "b",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let n = 20;
        let mut a = vec![0.1; n];
        a[0] = f64::NAN;
        let bcol = vec![0.2; n];
        let owned = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(a),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(bcol),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, owned, None, None).unwrap();
        let data = TabularData::new(storage);
        let vars: Vec<_> = data.schema().variables().iter().map(|v| v.id).collect();
        let alg = Notears::new();
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let err = alg.run(&data, &vars, &mut ws, &ctx);
        assert!(err.is_err(), "expected fail-closed on NaN");
    }
}
