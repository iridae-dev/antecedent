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
//! directed edge \(i\to j\) (NOTEARS / Zheng et al. 2018). Diagonal is always zero.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg_attr(
    test,
    allow(
        clippy::float_cmp,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

mod acyclicity;
mod solver;

use std::sync::Arc;

use antecedent_core::{ExecutionContext, Lag, VariableId};
use antecedent_data::{
    MissingPolicy, RowAccounting, TableView, TabularData, complete_rows, first_missing_cell,
    policy_required_text,
};
use antecedent_graph::{Dag, DagReview, DenseNodeId, NodeRef};
use antecedent_stats::standardize_columns;

use crate::constraints::{DiscoveryConstraints, KnowledgeScope};
use crate::engine::DiscoveryWorkspace;
use crate::error::DiscoveryError;
use crate::lingam::StaticDagDiscoveryResult;
use crate::pc::collect_float_columns;
use crate::result::{
    AlgorithmRecord, DiscoveryDiagnostic, DiscoveryIteration, DiscoveryPerformanceRecord,
    DiscoveryResult, EdgeEvidence, EvidenceSource, GraphEvidence, LaggedLink, ScoredLink,
    discovery_assumptions,
};

use solver::{NotearsWorkspace, SolveFailure, SolverConfig, solve_notears};

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
    /// Rows used and dropped, with the missing count per variable and the dropped-row digest.
    ///
    /// Dropped rows are those with a missing value in any discovery variable; a table
    /// without missing values drops none.
    pub row_accounting: RowAccounting,
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
    /// Declared handling of missing values. `None` (the default) refuses a table with a
    /// missing value; [`MissingPolicy::CompleteCase`] drops every row with a missing value
    /// among the discovery variables and reports the rows dropped.
    pub missing: Option<MissingPolicy>,
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
            missing: None,
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

    /// Declare how missing values are handled; see [`Self::missing`].
    #[must_use]
    pub fn with_missing(mut self, policy: MissingPolicy) -> Self {
        self.missing = Some(policy);
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
    #[allow(
        clippy::too_many_lines,
        reason = "one algorithm pipeline over shared mutable graph state; the phases are marked in place"
    )]
    pub fn run(
        &self,
        data: &TabularData,
        variables: &[VariableId],
        _workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<NotearsDiscoveryResult, DiscoveryError> {
        self.constraints.validate()?;
        self.constraints
            .check_background_knowledge(KnowledgeScope::ScoreSearch, variables)
            .map_err(DiscoveryError::Knowledge)?;
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

        let mut col_owned = collect_float_columns(data, variables)?;
        let policy = self.missing.unwrap_or(MissingPolicy::ErrorOnMissing);
        if !matches!(
            policy,
            MissingPolicy::ErrorOnMissing
                | MissingPolicy::CompleteCase
                | MissingPolicy::PairwiseAvailable
        ) {
            return Err(DiscoveryError::Refused {
                code: antecedent_core::reason_code!("missing_policy_not_applicable"),
                message: format!(
                    "missing_data.policy_not_applicable: NOTEARS cannot execute missing-data \
                     policy `{}`; use one of: error, complete_case, pairwise_available",
                    policy.name()
                ),
            });
        }
        // The analysis mask is not consulted here (as before); only missing cells are.
        let keep = complete_rows(data, None, variables)?;
        let mut row_accounting =
            RowAccounting::from_selection(data, None, policy, variables, &keep)?;
        let mut pairwise = None;
        if row_accounting.rows_dropped() > 0 {
            if policy == MissingPolicy::ErrorOnMissing {
                let cell = first_missing_cell(data, variables)?.ok_or(
                    DiscoveryError::unsupported("NOTEARS missing values changed during the run"),
                )?;
                return Err(DiscoveryError::Refused {
                    code: antecedent_core::reason_code!("missing_policy_required"),
                    message: format!(
                        "missing_data.policy_required: {}",
                        policy_required_text(
                            &cell,
                            "NOTEARS",
                            &[MissingPolicy::CompleteCase, MissingPolicy::PairwiseAvailable]
                        )
                    ),
                });
            }
            if policy == MissingPolicy::PairwiseAvailable {
                let valid = variables
                    .iter()
                    .map(|&v| {
                        let validity = data.column(v)?.validity();
                        Ok((0..col_owned[0].len()).map(|r| validity.is_valid(r)).collect())
                    })
                    .collect::<Result<Vec<Vec<bool>>, DiscoveryError>>()?;
                let factor = pairwise_factor(&col_owned, &valid, self.standardize)?;
                row_accounting = pairwise_accounting(&row_accounting, variables, data, &factor);
                pairwise = Some(factor);
            } else {
                col_owned = col_owned
                    .iter()
                    .map(|column| {
                        column
                            .iter()
                            .zip(&keep)
                            .filter(|(_, kept)| **kept)
                            .map(|(value, _)| *value)
                            .collect::<Arc<[f64]>>()
                    })
                    .collect();
            }
        }
        let d = variables.len();
        let (x, n) = if let Some(factor) = pairwise {
            // The pairwise covariance enters the least-squares score through this d×d
            // factor, whose Gram matrix over d is the repaired covariance (already
            // standardized when requested).
            (factor.design, d)
        } else {
            let n = col_owned[0].len();
            if n < 3 {
                return Err(DiscoveryError::stats_msg("insufficient rows for NOTEARS"));
            }
            for c in &col_owned {
                if c.len() != n {
                    return Err(DiscoveryError::data_msg("column length mismatch"));
                }
            }
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
            (x, n)
        };

        // Freeze diagonal + forbidden / tier-forbidden.
        let mut frozen = vec![false; d * d];
        for i in 0..d {
            frozen[i * d + i] = true;
        }
        let index = crate::constraints::ConstraintIndex::compile(&self.constraints);
        for i in 0..d {
            for j in 0..d {
                if i == j {
                    continue;
                }
                if index.is_forbidden(
                    &self.constraints,
                    crate::constraints::contemp_link(variables[i], variables[j]),
                ) {
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
        let soft_w = solve_notears(&x, n, d, &frozen, &cfg, &mut solver_ws, &|| {
            ctx.cancellation.is_cancelled()
        })
        .map_err(|failure| match failure {
            SolveFailure::Cancelled => DiscoveryError::Cancelled,
            SolveFailure::Message(message) => DiscoveryError::stats_msg(message),
        })?;

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

        // Stronger edges are inserted first. A directed cycle refuses the whole
        // set before any edge is given to the DAG.
        edge_coefs
            .sort_by(|a, b| b.2.abs().partial_cmp(&a.2.abs()).unwrap_or(std::cmp::Ordering::Equal));
        let directed = dense_edge_ids(&edge_coefs);
        if let Err(refusal) = antecedent_graph::error::refuse_directed_cycle(
            antecedent_graph::error::DagOperation::Discovery,
            antecedent_graph::error::CyclicNextAttempt::Unlicensed {
                consumer: "notears",
                attempt: "thresholded weights that contain a directed cycle are not a DAG; read \
                          the feedback from CyclicDigraph::condensation. This search does not \
                          drop the weaker edges",
            },
            d,
            &directed,
        ) {
            return Err(DiscoveryError::stats_msg(refusal.to_string()));
        }

        let mut dag = Dag::empty();
        for &v in variables {
            dag.add_node(NodeRef::Static(v))?;
        }
        let mut kept: Vec<(usize, usize, f64)> = Vec::new();
        for &(par, child, w) in &edge_coefs {
            let from = DenseNodeId::from_raw(crate::indexing::dense_u32(par));
            let to = DenseNodeId::from_raw(crate::indexing::dense_u32(child));
            if dag.children(from).contains(&to) {
                continue;
            }
            match dag.insert_directed(from, to) {
                Ok(()) => kept.push((par, child, w)),
                Err(antecedent_graph::GraphError::Cycle { .. }) => {
                    return Err(DiscoveryError::stats_msg(
                        antecedent_graph::error::CyclicDagRefusal {
                            operation: antecedent_graph::error::DagOperation::Discovery,
                            next_attempt: antecedent_graph::error::CyclicNextAttempt::Unlicensed {
                                consumer: "notears",
                                attempt: "thresholded weights that contain a directed cycle are \
                                          not a DAG; read the feedback from \
                                          CyclicDigraph::condensation. This search does not drop \
                                          the weaker edges",
                            },
                        }
                        .to_string(),
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
                separating_set: None,
                provenance: Arc::from([Arc::from("notears")]),
            })
            .collect();

        let links: Vec<ScoredLink> = edge_evidence
            .iter()
            .map(|e| ScoredLink {
                link: e.link,
                statistic: e.statistic.unwrap_or(0.0),
                p_value: f64::NAN,
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
                    "lambda={} threshold={} standardize={} max_iter={} h_tol={}{}",
                    self.lambda,
                    self.threshold,
                    self.standardize,
                    self.max_iter,
                    self.h_tol,
                    self.constraints.knowledge_config()
                )),
            },
            assumptions: discovery_assumptions("notears", true),
            iterations: Vec::<DiscoveryIteration>::new(),
            diagnostics: vec![DiscoveryDiagnostic {
                code: Arc::from("notears.hard_threshold"),
                message: Arc::from(format!(
                    "NOTEARS continuous weights were hard-thresholded at |W| >= {}; \
                     the thresholded DAG is not claimed to match the unthresholded Markov class",
                    self.threshold
                )),
            }],
            performance: DiscoveryPerformanceRecord {
                ci_tests: 0,
                links_retained: u64::try_from(edge_count).unwrap_or(u64::MAX),
                targets: u64::try_from(d).unwrap_or(u64::MAX),
                lagged_frame_bytes: 0,
                worker_threads: 1,
            },
            sepsets: crate::result::PcSepsets::default(),
        };

        Ok(NotearsDiscoveryResult { discovery, weights: Arc::from(soft_w), dim: d, row_accounting })
    }
}

/// A pairwise-available covariance as a least-squares design.
struct PairwiseFactor {
    /// Column-major `d × d` design `X` with `XᵀX / d` equal to the repaired covariance.
    design: Vec<f64>,
    /// Rows observed on both variables of each pair `(i, j)`, `i < j`.
    overlaps: Vec<(usize, usize, usize)>,
    /// Eigenvalues clipped to zero by the positive-semidefinite repair.
    clipped: usize,
}

/// Pairwise-available covariance (or correlation when `standardize`), repaired to positive
/// semidefinite by clipping negative eigenvalues, factored so the NOTEARS least-squares score
/// `tr((I-W)ᵀ S (I-W)) / 2` is evaluated on it exactly.
/// Variable positions in a NOTEARS fit are dense indexes of the column list.
#[allow(clippy::cast_possible_truncation)]
fn dense_edge_ids(edges: &[(usize, usize, f64)]) -> Vec<(u32, u32)> {
    edges.iter().map(|&(parent, child, _)| (parent as u32, child as u32)).collect()
}

fn pairwise_factor(
    columns: &[Arc<[f64]>],
    valid: &[Vec<bool>],
    standardize: bool,
) -> Result<PairwiseFactor, DiscoveryError> {
    let d = columns.len();
    let n = columns[0].len();
    let mean: Vec<f64> = (0..d)
        .map(|i| {
            let seen: Vec<f64> = (0..n).filter(|&r| valid[i][r]).map(|r| columns[i][r]).collect();
            seen.iter().sum::<f64>() / seen.len().max(1) as f64
        })
        .collect();
    let mut cov = vec![0.0; d * d];
    let mut overlaps = Vec::new();
    for i in 0..d {
        for j in i..d {
            let rows: Vec<usize> = (0..n).filter(|&r| valid[i][r] && valid[j][r]).collect();
            if rows.len() < 3 {
                return Err(DiscoveryError::unsupported(
                    "pairwise-available NOTEARS needs at least three rows observed on every pair of variables",
                ));
            }
            let c = rows
                .iter()
                .map(|&r| (columns[i][r] - mean[i]) * (columns[j][r] - mean[j]))
                .sum::<f64>()
                / rows.len() as f64;
            cov[i * d + j] = c;
            cov[j * d + i] = c;
            if i < j {
                overlaps.push((i, j, rows.len()));
            }
        }
    }
    if standardize {
        let sd: Vec<f64> = (0..d).map(|i| cov[i * d + i].sqrt()).collect();
        if sd.iter().any(|v| !v.is_finite() || *v <= 1e-12) {
            return Err(DiscoveryError::unsupported(
                "pairwise-available NOTEARS found a variable with no variance",
            ));
        }
        for i in 0..d {
            for j in 0..d {
                cov[i * d + j] /= sd[i] * sd[j];
            }
        }
    }
    let (values, vectors) = antecedent_stats::symmetric_eigen(&cov, d)?;
    let clipped = values.iter().filter(|v| **v < 0.0).count();
    // Z = Λ^{1/2} Vᵀ has ZᵀZ = V Λ Vᵀ = S; scale by √d so that XᵀX / d = S.
    let scale = (d as f64).sqrt();
    let mut design = vec![0.0; d * d];
    for r in 0..d {
        let root = values[r].max(0.0).sqrt();
        for j in 0..d {
            design[j * d + r] = scale * root * vectors[j * d + r];
        }
    }
    Ok(PairwiseFactor { design, overlaps, clipped })
}

/// Accounting of a pairwise-available run: nothing is dropped; each pair reports the rows both
/// of its variables were observed on, and the repair is named.
fn pairwise_accounting(
    base: &RowAccounting,
    variables: &[VariableId],
    data: &TabularData,
    factor: &PairwiseFactor,
) -> RowAccounting {
    use antecedent_data::GroupAccounting;
    let name = |i: usize| {
        data.schema()
            .get(variables[i])
            .map_or_else(|_| Arc::<str>::from("?"), |v| Arc::clone(&v.name))
    };
    let total = base.rows_total;
    let mut pairs: Vec<GroupAccounting> = factor
        .overlaps
        .iter()
        .map(|&(i, j, rows)| GroupAccounting {
            label: Arc::from(format!("pair:{},{}", name(i), name(j))),
            total,
            kept: rows,
            observed: rows,
        })
        .collect();
    pairs.push(GroupAccounting {
        label: Arc::from(format!("psd_repair:eigenvalues_clipped={}", factor.clipped)),
        total: 0,
        kept: 0,
        observed: 0,
    });
    let mut out = RowAccounting::from_dropped(
        base.policy,
        total,
        Vec::new(),
        base.missing_by_variable.to_vec(),
    );
    out.groups = pairs.into();
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{
        CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType, VariableId,
    };
    use antecedent_data::{
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
