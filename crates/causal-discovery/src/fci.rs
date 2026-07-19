//! Classic static FCI over [`TabularData`] → [`Pag`].
//!
//! Phases (Spirtes et al.):
//! 1. PC-style adjacency skeleton
//! 2. Unshielded collider orientation
//! 3. Possible-D-Sep adjacency (further edge removals)
//! 4. Reset remaining edges to `o–o`
//! 5. Zhang FCI orientation ([`crate::rule_scheduling::default_fci_rules`])
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::too_many_lines
)]

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use causal_core::{AssumptionSet, ExecutionContext, Lag, VariableId};
use causal_data::TabularData;
use causal_graph::{DenseNodeId, Pag, PagReview};
use causal_stats::{
    CiBatchRequest, CiPreparationPlan, CiQuery, ConfidenceMethod, ConditionalIndependence,
    FdrAdjustment, PartialCorrelation, PreparedCiTest,
};

use crate::combinations::for_each_combination_vars;
use crate::constraints::DiscoveryConstraints;
use crate::engine::DiscoveryWorkspace;
use crate::error::DiscoveryError;
use crate::evidence::threshold_scored_links;
use crate::orientation::{OrientationError, OrientationState};
use crate::pc::{adjacent_vars, collect_float_columns, edge_key};
use crate::possible_d_sep::{possible_d_sep, PossibleDSepBudget};
use crate::result::{
    AlgorithmRecord, DiscoveryDiagnostic, DiscoveryIteration, DiscoveryPerformanceRecord,
    DiscoveryResult, EdgeEvidence, EvidenceSource, GraphEvidence, LaggedLink, PcSepsets, ScoredLink,
};
use crate::rule_scheduling::{
    default_fci_rules, run_fci_orientation_to_fixed_point, FciOrientationRule, LpcmciOrientCollider,
};

/// Static FCI discovery result (`Pag` evidence + review).
pub type StaticPagDiscoveryResult = DiscoveryResult<Pag, PagReview>;

/// Default Possible-D-Sep BFS expansion budget (nodes expanded).
const DEFAULT_PDS_MAX_NODES: usize = 10_000;

/// Classic FCI algorithm over tabular (non-temporal) data.
#[derive(Clone)]
pub struct Fci {
    /// Constraints / alpha / max conditioning size.
    pub constraints: DiscoveryConstraints,
    /// Pluggable CI test.
    pub ci: Arc<dyn ConditionalIndependence + Send + Sync>,
    /// Multiple-testing adjustment (`None` = off). Applied after PC skeleton.
    pub fdr: Option<FdrAdjustment>,
    /// Possible-D-Sep BFS expansion budget.
    pub pds_max_nodes: usize,
}

impl std::fmt::Debug for Fci {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Fci")
            .field("constraints", &self.constraints)
            .field("ci", &"<dyn ConditionalIndependence>")
            .field("fdr", &self.fdr)
            .field("pds_max_nodes", &self.pds_max_nodes)
            .finish()
    }
}

impl Default for Fci {
    fn default() -> Self {
        Self::new()
    }
}

impl Fci {
    /// Default FCI with ParCorr and BH FDR over skeleton tests.
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
            ci: Arc::new(PartialCorrelation::default()),
            fdr: Some(FdrAdjustment::bh().with_exclude_contemporaneous(false)),
            pds_max_nodes: DEFAULT_PDS_MAX_NODES,
        }
    }

    /// Configure constraints.
    #[must_use]
    pub fn with_constraints(mut self, constraints: DiscoveryConstraints) -> Self {
        self.constraints = constraints;
        self
    }

    /// Enable / disable BH FDR.
    #[must_use]
    pub fn with_fdr(mut self, fdr: bool) -> Self {
        self.fdr = fdr.then(|| FdrAdjustment::bh().with_exclude_contemporaneous(false));
        self
    }

    /// Full FDR configuration.
    #[must_use]
    pub fn with_fdr_adjustment(mut self, fdr: Option<FdrAdjustment>) -> Self {
        self.fdr = fdr;
        self
    }

    /// Replace the CI test.
    #[must_use]
    pub fn with_ci(mut self, ci: Arc<dyn ConditionalIndependence + Send + Sync>) -> Self {
        self.ci = ci;
        self
    }

    /// Bound Possible-D-Sep BFS expansions (fail-closed when exceeded).
    #[must_use]
    pub fn with_pds_max_nodes(mut self, max_nodes: usize) -> Self {
        self.pds_max_nodes = max_nodes;
        self
    }

    /// Run classic static FCI.
    ///
    /// # Errors
    ///
    /// Data, CI, Possible-D-Sep budget, or orientation failures.
    pub fn run(
        &self,
        data: &TabularData,
        variables: &[VariableId],
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<StaticPagDiscoveryResult, DiscoveryError> {
        self.constraints.validate()?;
        if variables.is_empty() {
            return Err(DiscoveryError::Unsupported {
                message: "FCI requires at least one variable",
            });
        }

        let col_owned = collect_float_columns(data, variables)?;
        let cols: Vec<&[f64]> = col_owned.iter().map(AsRef::as_ref).collect();
        let n = cols[0].len();
        if n < 3 {
            return Err(DiscoveryError::stats_msg("insufficient rows for FCI"));
        }
        for c in &cols {
            if c.len() != n {
                return Err(DiscoveryError::data_msg("column length mismatch"));
            }
        }

        let var_index: HashMap<VariableId, usize> =
            variables.iter().enumerate().map(|(i, v)| (*v, i)).collect();

        let plan = CiPreparationPlan {
            significance: self.constraints.significance,
            confidence: ConfidenceMethod::default(),
        };
        let prepared: PreparedCiTest =
            self.ci.prepare(&cols, &plan, ctx).map_err(DiscoveryError::from)?;
        workspace.prepared_ci = Some(prepared);

        let alpha = self.constraints.alpha;
        let max_cond = self.constraints.max_cond_size;
        let mut adj: HashMap<(u32, u32), ()> = HashMap::new();
        let mut edge_scores: HashMap<(u32, u32), ScoredLink> = HashMap::new();
        let mut sepsets: PcSepsets = PcSepsets::default();
        let mut ci_tests: u64 = 0;
        let mut iterations = Vec::new();

        // --- Phase 1: PC-style adjacency ---
        for i in 0..variables.len() {
            for j in (i + 1)..variables.len() {
                let a = variables[i];
                let b = variables[j];
                if self.static_forbidden(a, b) {
                    continue;
                }
                adj.insert(edge_key(a, b), ());
            }
        }

        let mut combo_scratch = Vec::new();
        let mut depth = 0usize;
        loop {
            let mut depth_tests = 0u64;
            let edges: Vec<(VariableId, VariableId)> = adj
                .keys()
                .map(|&(lo, hi)| (VariableId::from_raw(lo), VariableId::from_raw(hi)))
                .collect();

            for &(x, y) in &edges {
                if !adj.contains_key(&edge_key(x, y)) {
                    continue;
                }
                if self.static_required(x, y) {
                    continue;
                }
                let neighbors_x = adjacent_vars(x, &adj, variables);
                let neighbors_y = adjacent_vars(y, &adj, variables);
                let mut cand_sets: Vec<Vec<VariableId>> = Vec::new();
                let nx: Vec<VariableId> = neighbors_x.into_iter().filter(|&v| v != y).collect();
                let ny: Vec<VariableId> = neighbors_y.into_iter().filter(|&v| v != x).collect();
                if nx.len() >= depth {
                    for_each_combination_vars(&nx, depth, &mut combo_scratch, |c| {
                        cand_sets.push(c.to_vec());
                        true
                    });
                }
                if ny.len() >= depth {
                    for_each_combination_vars(&ny, depth, &mut combo_scratch, |c| {
                        cand_sets.push(c.to_vec());
                        true
                    });
                }
                cand_sets.sort_unstable();
                cand_sets.dedup();

                let mut independent = false;
                let mut best_stat = f64::NAN;
                let mut best_p = f64::NAN;
                let mut best_sep: Arc<[VariableId]> = Arc::from([]);

                for z in &cand_sets {
                    let (stat, p) =
                        self.ci_test(&cols, &var_index, x, y, z, workspace, ctx)?;
                    ci_tests += 1;
                    depth_tests += 1;
                    best_stat = stat;
                    best_p = p;
                    if p > alpha {
                        independent = true;
                        best_sep = Arc::from(z.as_slice());
                        break;
                    }
                }

                if depth == 0 && cand_sets.is_empty() {
                    let (stat, p) =
                        self.ci_test(&cols, &var_index, x, y, &[], workspace, ctx)?;
                    ci_tests += 1;
                    depth_tests += 1;
                    best_stat = stat;
                    best_p = p;
                    if p > alpha {
                        independent = true;
                        best_sep = Arc::from([]);
                    }
                }

                let key = edge_key(x, y);
                if independent {
                    adj.remove(&key);
                    record_sepset(&mut sepsets, x, y, &best_sep);
                } else if best_p.is_finite() {
                    let link = ScoredLink {
                        link: LaggedLink {
                            source: x,
                            source_lag: Lag::CONTEMPORANEOUS,
                            target: y,
                            target_lag: Lag::CONTEMPORANEOUS,
                        },
                        statistic: best_stat,
                        p_value: best_p,
                        adjusted_p_value: None,
                    };
                    edge_scores
                        .entry(key)
                        .and_modify(|s| {
                            if best_p < s.p_value {
                                *s = link;
                            }
                        })
                        .or_insert(link);
                }
            }

            iterations.push(DiscoveryIteration {
                label: Arc::from(format!("fci.pc.depth.{depth}")),
                ci_tests: depth_tests,
            });

            depth += 1;
            if depth > max_cond {
                break;
            }
            let max_deg = adj
                .keys()
                .map(|&(lo, hi)| {
                    let a = VariableId::from_raw(lo);
                    let b = VariableId::from_raw(hi);
                    let da = adjacent_vars(a, &adj, variables).len().saturating_sub(1);
                    let db = adjacent_vars(b, &adj, variables).len().saturating_sub(1);
                    da.max(db)
                })
                .max()
                .unwrap_or(0);
            if max_deg < depth {
                break;
            }
        }

        // Optional FDR on surviving skeleton edges.
        let mut scored: Vec<ScoredLink> = adj
            .keys()
            .map(|&(lo, hi)| {
                let x = VariableId::from_raw(lo);
                let y = VariableId::from_raw(hi);
                edge_scores.get(&(lo, hi)).copied().unwrap_or(ScoredLink {
                    link: LaggedLink {
                        source: x,
                        source_lag: Lag::CONTEMPORANEOUS,
                        target: y,
                        target_lag: Lag::CONTEMPORANEOUS,
                    },
                    statistic: 0.0,
                    p_value: 0.0,
                    adjusted_p_value: None,
                })
            })
            .collect();
        scored = threshold_scored_links(scored, self.fdr, alpha);
        let kept: HashSet<(u32, u32)> = scored
            .iter()
            .map(|s| edge_key(s.link.source, s.link.target))
            .collect();
        if self.fdr.is_some() {
            adj.retain(|k, _| kept.contains(k));
        }

        let dense_of = |v: VariableId| -> Result<DenseNodeId, DiscoveryError> {
            let idx = *var_index.get(&v).ok_or_else(|| {
                DiscoveryError::data_msg(format!("unknown variable {v:?}"))
            })?;
            Ok(DenseNodeId::from_raw(u32::try_from(idx).expect("fit")))
        };

        // Build circle–circle PAG skeleton.
        let mut pag = build_pag_circle_skeleton(variables, &var_index, &adj)?;

        // Orientation state from PC sepsets.
        let mut state = OrientationState::default();
        load_sepsets_into_state(&sepsets, &dense_of, &mut state)?;

        // --- Phase 2: unshielded colliders (needed for Possible-D-Sep) ---
        let collider_rules: [&dyn FciOrientationRule; 1] = [&LpcmciOrientCollider];
        let _ = run_fci_orientation_to_fixed_point(&mut pag, &collider_rules, &mut state)?;

        // --- Phase 3: Possible-D-Sep adjacency ---
        let mut pds_tests = 0u64;
        let edges_pds: Vec<(VariableId, VariableId)> = adj
            .keys()
            .map(|&(lo, hi)| (VariableId::from_raw(lo), VariableId::from_raw(hi)))
            .collect();
        for &(x, y) in &edges_pds {
            if !adj.contains_key(&edge_key(x, y)) {
                continue;
            }
            if self.static_required(x, y) {
                continue;
            }
            let xd = dense_of(x)?;
            let yd = dense_of(y)?;
            let pds_x = possible_d_sep(&pag, xd, yd, self.pds_max_nodes).map_err(pds_budget_err)?;
            let pds_y = possible_d_sep(&pag, yd, xd, self.pds_max_nodes).map_err(pds_budget_err)?;

            let mut cand_pool: Vec<VariableId> = Vec::new();
            for d in pds_x.iter().chain(pds_y.iter()) {
                let v = variables[d.as_usize()];
                if v != x && v != y {
                    cand_pool.push(v);
                }
            }
            cand_pool.sort_unstable();
            cand_pool.dedup();

            let mut independent = false;
            let mut best_sep: Arc<[VariableId]> = Arc::from([]);

            'pds_depth: for depth in 0..=max_cond.min(cand_pool.len()) {
                let mut depth_sets = Vec::new();
                for_each_combination_vars(&cand_pool, depth, &mut combo_scratch, |z| {
                    depth_sets.push(z.to_vec());
                    true
                });
                for z in &depth_sets {
                    let (_stat, p) =
                        self.ci_test(&cols, &var_index, x, y, z, workspace, ctx)?;
                    pds_tests += 1;
                    ci_tests += 1;
                    if p > alpha {
                        independent = true;
                        best_sep = Arc::from(z.as_slice());
                        break 'pds_depth;
                    }
                }
            }

            if independent {
                let key = edge_key(x, y);
                adj.remove(&key);
                record_sepset(&mut sepsets, x, y, &best_sep);
                let _ = pag.remove_edge(xd, yd);
                let dense_sep: Vec<DenseNodeId> =
                    best_sep.iter().filter_map(|v| dense_of(*v).ok()).collect();
                state.set_sepset(xd, yd, Arc::from(dense_sep));
            }
        }
        iterations.push(DiscoveryIteration {
            label: Arc::from("fci.possible_d_sep"),
            ci_tests: pds_tests,
        });

        // --- Phase 4: reset remaining edges to o–o ---
        pag = build_pag_circle_skeleton(variables, &var_index, &adj)?;
        state = OrientationState::default();
        load_sepsets_into_state(&sepsets, &dense_of, &mut state)?;

        // --- Phase 5: full Zhang FCI orientation ---
        let rules = default_fci_rules();
        let orient_delta = run_fci_orientation_to_fixed_point(&mut pag, &rules, &mut state)?;

        let mut diagnostics = Vec::new();
        if state.conflicts > 0 || orient_delta.conflicts > 0 {
            diagnostics.push(DiscoveryDiagnostic {
                code: Arc::from("fci.orientation_conflict"),
                message: Arc::from(format!(
                    "{} orientation conflict(s)",
                    state.conflicts.max(orient_delta.conflicts)
                )),
            });
        }

        // Refresh scored links to surviving edges.
        scored.retain(|s| adj.contains_key(&edge_key(s.link.source, s.link.target)));

        let edge_evidence: Vec<EdgeEvidence> = scored
            .iter()
            .map(|s| {
                let seps = sepsets
                    .get(&(
                        s.link.source,
                        Lag::CONTEMPORANEOUS,
                        s.link.target,
                        Lag::CONTEMPORANEOUS,
                    ))
                    .cloned()
                    .into_iter()
                    .collect::<Vec<_>>();
                EdgeEvidence {
                    link: s.link,
                    statistic: Some(s.statistic),
                    p_value: Some(s.p_value),
                    adjusted_p_value: s.adjusted_p_value,
                    interval: None,
                    separating_sets: Arc::from(seps),
                    provenance: Arc::from([Arc::from("fci")]),
                }
            })
            .collect();

        let evidence = GraphEvidence {
            graph: pag.clone(),
            edge_evidence: Arc::from(edge_evidence),
            links: Arc::from(scored),
            source: EvidenceSource::Discovery {
                algorithm: Arc::from("fci"),
            },
        };
        let review = PagReview::from_pag(pag, "fci");

        Ok(DiscoveryResult {
            evidence,
            review,
            algorithm: AlgorithmRecord {
                id: Arc::from("fci"),
                config: Arc::from(format!(
                    "alpha={},max_cond={},fdr={},pds_max={}",
                    alpha,
                    max_cond,
                    self.fdr.is_some(),
                    self.pds_max_nodes
                )),
            },
            assumptions: AssumptionSet::default(),
            iterations,
            diagnostics,
            performance: DiscoveryPerformanceRecord {
                ci_tests,
                links_retained: u64::try_from(adj.len()).unwrap_or(u64::MAX),
                targets: u64::try_from(variables.len()).unwrap_or(u64::MAX),
                lagged_frame_bytes: 0,
                worker_threads: 1,
            },
            sepsets,
        })
    }

    fn static_forbidden(&self, a: VariableId, b: VariableId) -> bool {
        let link_ab = LaggedLink {
            source: a,
            source_lag: Lag::CONTEMPORANEOUS,
            target: b,
            target_lag: Lag::CONTEMPORANEOUS,
        };
        let link_ba = LaggedLink {
            source: b,
            source_lag: Lag::CONTEMPORANEOUS,
            target: a,
            target_lag: Lag::CONTEMPORANEOUS,
        };
        self.constraints.is_forbidden(link_ab) && self.constraints.is_forbidden(link_ba)
    }

    fn static_required(&self, a: VariableId, b: VariableId) -> bool {
        let link_ab = LaggedLink {
            source: a,
            source_lag: Lag::CONTEMPORANEOUS,
            target: b,
            target_lag: Lag::CONTEMPORANEOUS,
        };
        let link_ba = LaggedLink {
            source: b,
            source_lag: Lag::CONTEMPORANEOUS,
            target: a,
            target_lag: Lag::CONTEMPORANEOUS,
        };
        self.constraints.is_required(link_ab) || self.constraints.is_required(link_ba)
    }

    fn ci_test(
        &self,
        cols: &[&[f64]],
        var_index: &HashMap<VariableId, usize>,
        x: VariableId,
        y: VariableId,
        z: &[VariableId],
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<(f64, f64), DiscoveryError> {
        let xi = *var_index.get(&x).ok_or_else(|| DiscoveryError::data_msg("missing x"))?;
        let yi = *var_index.get(&y).ok_or_else(|| DiscoveryError::data_msg("missing y"))?;
        workspace.z_flat.clear();
        for &v in z {
            let zi = *var_index.get(&v).ok_or_else(|| DiscoveryError::data_msg("missing z"))?;
            workspace.z_flat.push(zi);
        }
        let prepared = workspace.prepared_ci.as_ref().ok_or_else(|| {
            DiscoveryError::Unsupported {
                message: "CI test used before prepare()",
            }
        })?;
        let queries = [CiQuery {
            x: xi,
            y: yi,
            z_start: 0,
            z_len: workspace.z_flat.len(),
        }];
        let req = CiBatchRequest {
            columns: cols,
            queries: &queries,
            z_flat: &workspace.z_flat,
            significance: self.constraints.significance,
            confidence: ConfidenceMethod::default(),
        };
        let out = self
            .ci
            .test_batch(prepared, &req, &mut workspace.ci, ctx)
            .map_err(DiscoveryError::from)?;
        let result = out.results.into_iter().next().ok_or_else(|| {
            DiscoveryError::stats_msg("CI batch returned no results")
        })?;
        if !result.statistic.is_finite() || !result.p_value.is_finite() {
            return Err(DiscoveryError::stats_msg("non-finite CI statistic or p-value"));
        }
        Ok((result.statistic, result.p_value))
    }
}

fn pds_budget_err(b: PossibleDSepBudget) -> DiscoveryError {
    DiscoveryError::from(OrientationError::SearchBudgetExhausted {
        rule: "fci.possible_d_sep",
        max_paths: b.max_nodes,
        max_len: 0,
    })
}

pub(crate) fn record_sepset(sepsets: &mut PcSepsets, x: VariableId, y: VariableId, best_sep: &[VariableId]) {
    let sep_lagged: Arc<[(VariableId, Lag)]> = Arc::from(
        best_sep
            .iter()
            .map(|&v| (v, Lag::CONTEMPORANEOUS))
            .collect::<Vec<_>>(),
    );
    sepsets.insert(
        (x, Lag::CONTEMPORANEOUS, y, Lag::CONTEMPORANEOUS),
        Arc::clone(&sep_lagged),
    );
    sepsets.insert((y, Lag::CONTEMPORANEOUS, x, Lag::CONTEMPORANEOUS), sep_lagged);
}

pub(crate) fn load_sepsets_into_state(
    sepsets: &PcSepsets,
    dense_of: &dyn Fn(VariableId) -> Result<DenseNodeId, DiscoveryError>,
    state: &mut OrientationState,
) -> Result<(), DiscoveryError> {
    for ((sx, _, ty, _), sep) in sepsets {
        if sx.raw() > ty.raw() {
            continue;
        }
        let a = dense_of(*sx)?;
        let b = dense_of(*ty)?;
        let dense_sep: Vec<DenseNodeId> =
            sep.iter().filter_map(|(v, _)| dense_of(*v).ok()).collect();
        state.set_sepset(a, b, Arc::from(dense_sep));
    }
    Ok(())
}

pub(crate) fn build_pag_circle_skeleton(
    variables: &[VariableId],
    var_index: &HashMap<VariableId, usize>,
    adj: &HashMap<(u32, u32), ()>,
) -> Result<Pag, DiscoveryError> {
    let mut pag = Pag::with_variables(u32::try_from(variables.len()).unwrap_or(u32::MAX));
    if variables.iter().enumerate().any(|(i, v)| v.raw() as usize != i) {
        pag = Pag::empty();
        for &v in variables {
            pag.add_node(causal_graph::NodeRef::Static(v))
                .map_err(DiscoveryError::from)?;
        }
    }
    for &(lo, hi) in adj.keys() {
        let a_idx = *var_index
            .get(&VariableId::from_raw(lo))
            .ok_or_else(|| DiscoveryError::data_msg("unknown variable in skeleton"))?;
        let b_idx = *var_index
            .get(&VariableId::from_raw(hi))
            .ok_or_else(|| DiscoveryError::data_msg("unknown variable in skeleton"))?;
        let a = DenseNodeId::from_raw(u32::try_from(a_idx).expect("fit"));
        let b = DenseNodeId::from_raw(u32::try_from(b_idx).expect("fit"));
        pag.insert_circle_circle(a, b).map_err(DiscoveryError::from)?;
    }
    Ok(pag)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use causal_core::{
        CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
        VariableId,
    };
    use causal_data::{Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap};
    use causal_graph::Endpoint;
    use causal_stats::OracleCi;

    use super::*;

    fn tabular_n(ncols: usize, nrows: usize) -> TabularData {
        let mut b = CausalSchemaBuilder::new();
        for i in 0..ncols {
            b.add_variable(
                format!("v{i}"),
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        }
        let schema = b.build().unwrap();
        let owned: Vec<OwnedColumn> = (0..ncols)
            .map(|i| {
                OwnedColumn::Float64(
                    Float64Column::new(
                        VariableId::from_raw(i as u32),
                        Arc::from(vec![0.0; nrows]),
                        ValidityBitmap::all_valid(nrows),
                    )
                    .unwrap(),
                )
            })
            .collect();
        let storage = OwnedColumnarStorage::try_new(schema, owned, None, None).unwrap();
        TabularData::new(storage)
    }

    #[test]
    fn oracle_chain_recovers_skeleton() {
        let data = tabular_n(3, 50);
        let vars = [
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            VariableId::from_raw(2),
        ];
        let oracle = OracleCi::new([(0usize, 1usize), (1usize, 2usize)]);
        let fci = Fci::new().with_fdr(false).with_ci(Arc::new(oracle));
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let result = fci.run(&data, &vars, &mut ws, &ctx).unwrap();
        let g = &result.evidence.graph;
        assert!(g.has_edge(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)));
        assert!(g.has_edge(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)));
        assert!(!g.has_edge(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)));
        assert_eq!(result.algorithm.id.as_ref(), "fci");
    }

    #[test]
    fn oracle_collider_orients_into_middle() {
        let data = tabular_n(3, 40);
        let vars = [
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            VariableId::from_raw(2),
        ];
        let oracle = OracleCi::new([(0usize, 1usize), (1usize, 2usize)]);
        let fci = Fci::new().with_fdr(false).with_ci(Arc::new(oracle));
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(2);
        let result = fci.run(&data, &vars, &mut ws, &ctx).unwrap();
        let g = &result.evidence.graph;
        let e01 = g
            .edge_between(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1))
            .unwrap();
        let e21 = g
            .edge_between(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1))
            .unwrap();
        // Collider at 1: arrow into 1 on both edges.
        let at_1_from_0 = if e01.a.raw() == 1 { e01.at_a } else { e01.at_b };
        let at_1_from_2 = if e21.a.raw() == 1 { e21.at_a } else { e21.at_b };
        assert!(matches!(at_1_from_0, Endpoint::Arrow));
        assert!(matches!(at_1_from_2, Endpoint::Arrow));
    }

    #[test]
    fn review_tracks_remaining_circles() {
        let data = tabular_n(3, 40);
        let vars = [
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            VariableId::from_raw(2),
        ];
        // Fully connected dependent — skeleton retains all edges; circles remain.
        let oracle = OracleCi::new([(0usize, 1usize), (1usize, 2usize), (0usize, 2usize)]);
        let fci = Fci::new().with_fdr(false).with_ci(Arc::new(oracle));
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(3);
        let result = fci.run(&data, &vars, &mut ws, &ctx).unwrap();
        assert!(!result.review.pending_circles.is_empty() || result.review.is_complete());
        assert_eq!(result.review.algorithm.as_ref(), "fci");
    }
}
