//! DirectLiNGAM → static [`Dag`] (Shimizu et al. 2011).
//!
//! Causal-order search by residual–predictor independence (distance correlation),
//! then OLS coefficient pruning. Does not use ICA or the Meek/PC orientation stack.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::too_many_lines
)]

use std::sync::Arc;

use causal_core::{AssumptionSet, ExecutionContext, Lag, VariableId};
use causal_data::TabularData;
use causal_graph::{Dag, DagReview, DenseNodeId, NodeRef};
use causal_stats::{
    DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace,
};

use crate::constraints::DiscoveryConstraints;
use crate::engine::DiscoveryWorkspace;
use crate::error::DiscoveryError;
use crate::pc::collect_float_columns;
use crate::result::{
    AlgorithmRecord, DiscoveryDiagnostic, DiscoveryIteration, DiscoveryPerformanceRecord,
    DiscoveryResult, EdgeEvidence, EvidenceSource, GraphEvidence, LaggedLink, ScoredLink,
};

/// Static DirectLiNGAM discovery result (`Dag` evidence + review).
pub type StaticDagDiscoveryResult = DiscoveryResult<Dag, DagReview>;

/// DirectLiNGAM over tabular (non-temporal) continuous data.
#[derive(Clone, Debug)]
pub struct DirectLingam {
    /// Constraints / max parents / forbidden edges.
    pub constraints: DiscoveryConstraints,
    /// Absolute coefficient prune threshold after order search.
    pub prune_threshold: f64,
}

impl Default for DirectLingam {
    fn default() -> Self {
        Self::new()
    }
}

impl DirectLingam {
    /// Default DirectLiNGAM (`prune_threshold = 0.05`).
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
            prune_threshold: 0.05,
        }
    }

    /// Configure constraints.
    #[must_use]
    pub fn with_constraints(mut self, constraints: DiscoveryConstraints) -> Self {
        self.constraints = constraints;
        self
    }

    /// Absolute OLS coefficient prune threshold.
    #[must_use]
    pub fn with_prune_threshold(mut self, threshold: f64) -> Self {
        self.prune_threshold = threshold;
        self
    }

    /// Run DirectLiNGAM.
    ///
    /// # Errors
    ///
    /// Data, numerical, or graph failures.
    pub fn run(
        &self,
        data: &TabularData,
        variables: &[VariableId],
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<StaticDagDiscoveryResult, DiscoveryError> {
        let _ = (workspace, ctx);
        self.constraints.validate()?;
        if variables.is_empty() {
            return Err(DiscoveryError::Unsupported {
                message: "DirectLiNGAM requires at least one variable",
            });
        }

        let col_owned = collect_float_columns(data, variables)?;
        let n = col_owned[0].len();
        if n < 3 {
            return Err(DiscoveryError::stats_msg("insufficient rows for DirectLiNGAM"));
        }
        for c in &col_owned {
            if c.len() != n {
                return Err(DiscoveryError::data_msg("column length mismatch"));
            }
        }

        let p = variables.len();
        // Working residual matrix (column-major via Vec per variable), centered.
        let mut cols: Vec<Vec<f64>> = col_owned
            .iter()
            .map(|c| {
                let mut v = c.to_vec();
                center_inplace(&mut v);
                v
            })
            .collect();

        let mut remaining: Vec<usize> = (0..p).collect();
        let mut order: Vec<usize> = Vec::with_capacity(p);

        while remaining.len() > 1 {
            let mut best_j = remaining[0];
            let mut best_score = f64::INFINITY;
            for &j in &remaining {
                let mut score = 0.0;
                for &i in &remaining {
                    if i == j {
                        continue;
                    }
                    let resid = regress_residual(&cols[i], &cols[j]);
                    score += distance_correlation(&resid, &cols[j]);
                }
                if score < best_score {
                    best_score = score;
                    best_j = j;
                }
            }
            // Residualize remaining on chosen exogenous.
            for &i in &remaining {
                if i == best_j {
                    continue;
                }
                cols[i] = regress_residual(&cols[i], &cols[best_j]);
            }
            remaining.retain(|&i| i != best_j);
            order.push(best_j);
        }
        if let Some(last) = remaining.pop() {
            order.push(last);
        }

        // Rebuild original centered columns for pruning.
        let orig: Vec<Vec<f64>> = col_owned
            .iter()
            .map(|c| {
                let mut v = c.to_vec();
                center_inplace(&mut v);
                v
            })
            .collect();

        let max_parents = self.constraints.max_parents.unwrap_or(p.saturating_sub(1));
        let mut dag = Dag::empty();
        for &v in variables {
            dag.add_node(NodeRef::Static(v))?;
        }

        let mut edge_coefs: Vec<(usize, usize, f64)> = Vec::new();
        let backend = FaerBackend;
        let mut ls_ws = LeastSquaresWorkspace::default();

        for (pos, &child) in order.iter().enumerate() {
            if pos == 0 {
                continue;
            }
            let preds: Vec<usize> = order[..pos]
                .iter()
                .copied()
                .filter(|&par| {
                    !forbidden_edge(&self.constraints, variables, par, child)
                })
                .collect();
            if preds.is_empty() {
                continue;
            }
            // OLS child ~ preds (no intercept — already centered).
            let k = preds.len();
            let mut x = vec![0.0; n * k];
            for (c, &par) in preds.iter().enumerate() {
                for r in 0..n {
                    x[c * n + r] = orig[par][r];
                }
            }
            let fit = match backend.least_squares(&x, n, k, &orig[child], &mut ls_ws) {
                Ok(f) => f,
                Err(_) => {
                    // Fall back to pairwise prune on rank failure.
                    for &par in &preds {
                        let beta = simple_regression_coef(&orig[child], &orig[par]);
                        if beta.abs() >= self.prune_threshold
                            && !forbidden_edge(&self.constraints, variables, par, child)
                        {
                            edge_coefs.push((par, child, beta));
                        }
                    }
                    continue;
                }
            };
            // Keep largest |β| up to max_parents.
            let mut ranked: Vec<(usize, f64)> = preds
                .iter()
                .enumerate()
                .map(|(i, &par)| (par, fit.coefficients[i]))
                .filter(|(_, b)| b.abs() >= self.prune_threshold)
                .collect();
            ranked.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal));
            for (par, beta) in ranked.into_iter().take(max_parents) {
                edge_coefs.push((par, child, beta));
            }
        }

        // Seed required edges if both endpoints exist.
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
                edge_coefs.push((si, ti, 1.0));
            }
        }

        for &(par, child, _) in &edge_coefs {
            let from = DenseNodeId::from_raw(par as u32);
            let to = DenseNodeId::from_raw(child as u32);
            if dag.children(from).contains(&to) {
                continue;
            }
            // Respect causal order: parent must precede child.
            let po = order.iter().position(|&x| x == par);
            let co = order.iter().position(|&x| x == child);
            if let (Some(pi), Some(ci)) = (po, co) {
                if pi < ci {
                    let _ = dag.insert_directed(from, to);
                }
            }
        }

        let edge_evidence: Vec<EdgeEvidence> = edge_coefs
            .iter()
            .filter_map(|&(par, child, beta)| {
                let po = order.iter().position(|&x| x == par)?;
                let co = order.iter().position(|&x| x == child)?;
                if po >= co {
                    return None;
                }
                Some(EdgeEvidence {
                    link: LaggedLink {
                        source: variables[par],
                        source_lag: Lag::CONTEMPORANEOUS,
                        target: variables[child],
                        target_lag: Lag::CONTEMPORANEOUS,
                    },
                    statistic: Some(beta),
                    p_value: None,
                    adjusted_p_value: None,
                    interval: None,
                    separating_sets: Arc::from([]),
                    provenance: Arc::from([Arc::from("direct_lingam")]),
                })
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
        let review = DagReview::from_dag(dag.clone(), "direct_lingam");
        let evidence = GraphEvidence {
            graph: dag,
            edge_evidence: Arc::from(edge_evidence),
            links: Arc::from(links),
            source: EvidenceSource::Discovery {
                algorithm: Arc::from("direct_lingam"),
            },
        };

        Ok(DiscoveryResult {
            evidence,
            review,
            algorithm: AlgorithmRecord {
                id: Arc::from("direct_lingam"),
                config: Arc::from(format!(
                    "prune_threshold={} order={order:?}",
                    self.prune_threshold
                )),
            },
            assumptions: AssumptionSet::default(),
            iterations: Vec::<DiscoveryIteration>::new(),
            diagnostics: Vec::<DiscoveryDiagnostic>::new(),
            performance: DiscoveryPerformanceRecord {
                ci_tests: 0,
                links_retained: u64::try_from(edge_count).unwrap_or(u64::MAX),
                targets: u64::try_from(p).unwrap_or(u64::MAX),
                lagged_frame_bytes: 0,
                worker_threads: 1,
            },
            sepsets: crate::result::PcSepsets::default(),
        })
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

fn center_inplace(v: &mut [f64]) {
    if v.is_empty() {
        return;
    }
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    for x in v.iter_mut() {
        *x -= mean;
    }
}

/// Residual of `y` after simple OLS on centered `x` (no intercept).
fn regress_residual(y: &[f64], x: &[f64]) -> Vec<f64> {
    let beta = simple_regression_coef(y, x);
    y.iter().zip(x.iter()).map(|(yi, xi)| yi - beta * xi).collect()
}

fn simple_regression_coef(y: &[f64], x: &[f64]) -> f64 {
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    for (yi, xi) in y.iter().zip(x.iter()) {
        sxx += xi * xi;
        sxy += xi * yi;
    }
    if sxx <= 1e-15 {
        0.0
    } else {
        sxy / sxx
    }
}

/// Székely distance correlation (L1 pairwise distances).
fn distance_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 || y.len() != n {
        return 0.0;
    }
    let mut ax = vec![0.0; n * n];
    let mut ay = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            ax[i * n + j] = (x[i] - x[j]).abs();
            ay[i * n + j] = (y[i] - y[j]).abs();
        }
    }
    double_center_inplace(&mut ax, n);
    double_center_inplace(&mut ay, n);
    let mut dcov2 = 0.0;
    let mut dvarx = 0.0;
    let mut dvary = 0.0;
    for i in 0..n * n {
        dcov2 += ax[i] * ay[i];
        dvarx += ax[i] * ax[i];
        dvary += ay[i] * ay[i];
    }
    let nn = (n * n) as f64;
    dcov2 /= nn;
    dvarx /= nn;
    dvary /= nn;
    if dvarx <= 0.0 || dvary <= 0.0 {
        return 0.0;
    }
    (dcov2.max(0.0) / (dvarx * dvary).sqrt()).sqrt()
}

fn double_center_inplace(a: &mut [f64], n: usize) {
    let mut row = vec![0.0; n];
    let mut col = vec![0.0; n];
    let mut mean = 0.0;
    for i in 0..n {
        for j in 0..n {
            row[i] += a[i * n + j];
            col[j] += a[i * n + j];
            mean += a[i * n + j];
        }
    }
    for i in 0..n {
        row[i] /= n as f64;
        col[i] /= n as f64;
    }
    mean /= (n * n) as f64;
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = a[i * n + j] - row[i] - col[j] + mean;
        }
    }
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

    /// Non-Gaussian SEM: X0 → X1 → X2 with Laplace-like noise.
    fn lingam_chain(n: usize) -> (TabularData, Vec<VariableId>) {
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
            // Heavy-tailed noise via mixture of uniforms (non-Gaussian).
            let u = ((i as f64 * 0.137) % 1.0) - 0.5;
            let v = ((i as f64 * 0.271) % 1.0) - 0.5;
            let w = ((i as f64 * 0.419) % 1.0) - 0.5;
            let e0 = u * u * u * 4.0;
            let e1 = v * v * v * 4.0;
            let e2 = w * w * w * 4.0;
            x0[i] = e0;
            x1[i] = 0.9 * x0[i] + e1;
            x2[i] = 0.9 * x1[i] + e2;
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
    fn direct_lingam_recovers_chain_order_edges() {
        let (data, vars) = lingam_chain(500);
        let alg = DirectLingam::new().with_prune_threshold(0.2);
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let result = alg.run(&data, &vars, &mut ws, &ctx).unwrap();
        assert_eq!(result.algorithm.id.as_ref(), "direct_lingam");
        let g = &result.evidence.graph;
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
    }

    #[test]
    fn dcor_self_near_one() {
        let x: Vec<f64> = (0..40).map(|i| (i as f64) * 0.1).collect();
        let d = distance_correlation(&x, &x);
        assert!(d > 0.99, "dCor(x,x)={d}");
    }
}
