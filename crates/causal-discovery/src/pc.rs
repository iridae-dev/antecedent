//! Static PC discovery over [`TabularData`].
//!
//! Classic undirected skeleton search + collider / Meek orientation → [`Cpdag`].
//! Distinct from PCMCI PC1 parent selection.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::field_reassign_with_default,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::unreadable_literal,
    clippy::zero_sized_map_values
)]

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use causal_core::{AssumptionSet, ExecutionContext, Lag, VariableId};
use causal_data::{TableView, TabularData};
use causal_graph::{Cpdag, CpdagReview, DenseNodeId};
use causal_stats::{
    CiBatchRequest, CiPreparationPlan, CiQuery, ConditionalIndependence, ConfidenceMethod,
    FdrAdjustment, PartialCorrelation, PreparedCiTest,
};

use crate::combinations::for_each_combination_vars;
use crate::constraints::DiscoveryConstraints;
use crate::engine::DiscoveryWorkspace;
use crate::error::DiscoveryError;
use crate::evidence::threshold_scored_links;
use crate::orientation::{
    MeekR1, MeekR2, MeekR3, MeekR4, OrientCollider, OrientationState, StaticOrientationRule,
    run_static_orientation_to_fixed_point,
};
use crate::result::{
    AlgorithmRecord, DiscoveryDiagnostic, DiscoveryIteration, DiscoveryPerformanceRecord,
    DiscoveryResult, EdgeEvidence, EvidenceSource, GraphEvidence, LaggedLink, PcSepsets,
    ScoredLink,
};

/// Static PC discovery result (`Cpdag` evidence + review).
pub type StaticCpdagDiscoveryResult = DiscoveryResult<Cpdag, CpdagReview>;

/// Classic PC algorithm over tabular (non-temporal) data.
#[derive(Clone)]
pub struct Pc {
    /// Constraints / alpha / max conditioning size.
    pub constraints: DiscoveryConstraints,
    /// Pluggable CI test.
    pub ci: Arc<dyn ConditionalIndependence + Send + Sync>,
    /// Multiple-testing adjustment (`None` = off). Static PC includes all edges in the family.
    pub fdr: Option<FdrAdjustment>,
}

impl std::fmt::Debug for Pc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pc")
            .field("constraints", &self.constraints)
            .field("ci", &"<dyn ConditionalIndependence>")
            .field("fdr", &self.fdr)
            .finish()
    }
}

impl Default for Pc {
    fn default() -> Self {
        Self::new()
    }
}

impl Pc {
    /// Default PC with `ParCorr` and BH FDR over all undirected tests.
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
            ci: Arc::new(PartialCorrelation),
            fdr: Some(FdrAdjustment::bh().with_exclude_contemporaneous(false)),
        }
    }

    /// Configure constraints.
    #[must_use]
    pub fn with_constraints(mut self, constraints: DiscoveryConstraints) -> Self {
        self.constraints = constraints;
        self
    }

    /// Enable / disable BH FDR (all static edges in family).
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

    /// Run static PC.
    ///
    /// # Errors
    ///
    /// Data, CI, or orientation failures.
    pub fn run(
        &self,
        data: &TabularData,
        variables: &[VariableId],
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<StaticCpdagDiscoveryResult, DiscoveryError> {
        self.constraints.validate()?;
        if variables.is_empty() {
            return Err(DiscoveryError::Unsupported {
                message: "PC requires at least one variable",
            });
        }

        let col_owned = collect_float_columns(data, variables)?;
        let cols: Vec<&[f64]> = col_owned.iter().map(AsRef::as_ref).collect();
        let n = cols[0].len();
        if n < 3 {
            return Err(DiscoveryError::stats_msg("insufficient rows for PC"));
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

        // Complete undirected skeleton minus forbidden edges.
        for i in 0..variables.len() {
            for j in (i + 1)..variables.len() {
                let a = variables[i];
                let b = variables[j];
                if self.static_forbidden(a, b) {
                    continue;
                }
                let key = edge_key(a, b);
                adj.insert(key, ());
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
                    let (stat, p) = self.ci_test(&cols, &var_index, x, y, z, workspace, ctx)?;
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
                    let (stat, p) = self.ci_test(&cols, &var_index, x, y, &[], workspace, ctx)?;
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
                    let sep_lagged: Arc<[(VariableId, Lag)]> = Arc::from(
                        best_sep.iter().map(|&v| (v, Lag::CONTEMPORANEOUS)).collect::<Vec<_>>(),
                    );
                    sepsets.insert(
                        (x, Lag::CONTEMPORANEOUS, y, Lag::CONTEMPORANEOUS),
                        Arc::clone(&sep_lagged),
                    );
                    sepsets.insert((y, Lag::CONTEMPORANEOUS, x, Lag::CONTEMPORANEOUS), sep_lagged);
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
                label: Arc::from(format!("pc.depth.{depth}")),
                ci_tests: depth_tests,
            });

            depth += 1;
            if depth > max_cond {
                break;
            }
            // Stop when no remaining edge has enough neighbors for larger cond sets.
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

        // Optional FDR on surviving undirected edges (re-test empty cond for a stable p).
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
        let kept: HashSet<(u32, u32)> =
            scored.iter().map(|s| edge_key(s.link.source, s.link.target)).collect();
        // If FDR ran, drop edges that failed; if FDR off, keep skeleton as-is.
        if self.fdr.is_some() {
            adj.retain(|k, ()| kept.contains(k));
        }

        // Build undirected CPDAG skeleton.
        let mut cpdag = Cpdag::with_variables(u32::try_from(variables.len()).unwrap_or(u32::MAX));
        // Map VariableId → dense: assume variables are 0..n contiguous for with_variables,
        // otherwise rebuild with explicit add_node order matching `variables`.
        if variables.iter().enumerate().any(|(i, v)| v.raw() as usize != i) {
            cpdag = Cpdag::empty();
            for &v in variables {
                cpdag.add_node(causal_graph::NodeRef::Static(v)).map_err(DiscoveryError::from)?;
            }
        }
        let dense_of = |v: VariableId| -> Result<DenseNodeId, DiscoveryError> {
            let idx = *var_index
                .get(&v)
                .ok_or_else(|| DiscoveryError::data_msg(format!("unknown variable {v:?}")))?;
            Ok(DenseNodeId::from_raw(u32::try_from(idx).expect("fit")))
        };
        for &(lo, hi) in adj.keys() {
            let a = dense_of(VariableId::from_raw(lo))?;
            let b = dense_of(VariableId::from_raw(hi))?;
            cpdag.insert_undirected(a, b).map_err(DiscoveryError::from)?;
        }

        // Orientation state from sepsets (dense ids).
        let mut state = OrientationState::default();
        for ((sx, _, ty, _), sep) in &sepsets {
            // Only store one direction with dense node ids.
            if sx.raw() > ty.raw() {
                continue;
            }
            let a = dense_of(*sx)?;
            let b = dense_of(*ty)?;
            let dense_sep: Vec<DenseNodeId> =
                sep.iter().filter_map(|(v, _)| dense_of(*v).ok()).collect();
            state.set_sepset(a, b, Arc::from(dense_sep));
        }

        let rules: [&dyn StaticOrientationRule; 5] =
            [&OrientCollider, &MeekR1, &MeekR2, &MeekR3, &MeekR4];
        let orient_delta = run_static_orientation_to_fixed_point(&mut cpdag, &rules, &mut state)?;

        let mut diagnostics = Vec::new();
        if state.conflicts > 0 || orient_delta.conflicts > 0 {
            diagnostics.push(DiscoveryDiagnostic {
                code: Arc::from("pc.orientation_conflict"),
                message: Arc::from(format!(
                    "{} orientation conflict(s)",
                    state.conflicts.max(orient_delta.conflicts)
                )),
            });
        }

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
                    provenance: Arc::from([Arc::from("pc")]),
                }
            })
            .collect();

        let evidence = GraphEvidence {
            graph: cpdag.clone(),
            edge_evidence: Arc::from(edge_evidence),
            links: Arc::from(scored),
            source: EvidenceSource::Discovery { algorithm: Arc::from("pc") },
        };
        let review = CpdagReview::from_cpdag(cpdag, "pc");

        Ok(DiscoveryResult {
            evidence,
            review,
            algorithm: AlgorithmRecord {
                id: Arc::from("pc"),
                config: Arc::from(format!(
                    "alpha={},max_cond={},fdr={}",
                    alpha,
                    max_cond,
                    self.fdr.is_some()
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
        let prepared = workspace.prepared_ci.as_ref().ok_or({
            DiscoveryError::Unsupported { message: "CI test used before prepare()" }
        })?;
        let queries = [CiQuery { x: xi, y: yi, z_start: 0, z_len: workspace.z_flat.len() }];
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
        let result = out
            .results
            .into_iter()
            .next()
            .ok_or_else(|| DiscoveryError::stats_msg("CI batch returned no results"))?;
        if !result.statistic.is_finite() || !result.p_value.is_finite() {
            return Err(DiscoveryError::stats_msg("non-finite CI statistic or p-value"));
        }
        Ok((result.statistic, result.p_value))
    }
}

pub(crate) fn edge_key(a: VariableId, b: VariableId) -> (u32, u32) {
    if a.raw() <= b.raw() { (a.raw(), b.raw()) } else { (b.raw(), a.raw()) }
}

pub(crate) fn adjacent_vars(
    v: VariableId,
    adj: &HashMap<(u32, u32), ()>,
    variables: &[VariableId],
) -> Vec<VariableId> {
    variables.iter().copied().filter(|&u| u != v && adj.contains_key(&edge_key(v, u))).collect()
}

pub(crate) fn collect_float_columns(
    data: &TabularData,
    variables: &[VariableId],
) -> Result<Vec<Arc<[f64]>>, DiscoveryError> {
    // Uses `TableView::float64_values`, which coerces Int64/Boolean to f64.
    let mut out = Vec::with_capacity(variables.len());
    for &v in variables {
        let vals = data.float64_values(v).map_err(DiscoveryError::from)?;
        out.push(Arc::from(vals));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use causal_core::{
        CausalSchemaBuilder, ExecutionContext, Lag, MeasurementSpec, RoleHint, SmallRoleSet,
        ValueType, VariableId,
    };
    use causal_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
    };
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
        // True: 0→1→2. Dependent pairs: (0,1), (1,2). Oracle drops (0,2) at depth 0
        // (no cond-set awareness), so orientation may treat it as a collider — skeleton only.
        let data = tabular_n(3, 50);
        let vars = [VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2)];
        let oracle = OracleCi::new([(0usize, 1usize), (1usize, 2usize)]);
        let pc = Pc::new().with_fdr(false).with_ci(Arc::new(oracle));
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let result = pc.run(&data, &vars, &mut ws, &ctx).unwrap();
        let g = &result.evidence.graph;
        assert!(g.has_edge(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)));
        assert!(g.has_edge(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)));
        assert!(!g.has_edge(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)));
    }

    #[test]
    fn pc_accepts_int64_columns_via_float_coerce() {
        let mut b = CausalSchemaBuilder::new();
        for i in 0..3 {
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
        let nrows = 40;
        let owned: Vec<OwnedColumn> = (0..3)
            .map(|i| {
                OwnedColumn::Int64(
                    causal_data::Int64Column::new(
                        VariableId::from_raw(i as u32),
                        Arc::from(vec![i64::from(i); nrows]),
                        ValidityBitmap::all_valid(nrows),
                    )
                    .unwrap(),
                )
            })
            .collect();
        let storage = OwnedColumnarStorage::try_new(schema, owned, None, None).unwrap();
        let data = TabularData::new(storage);
        let vars = [VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2)];
        let oracle = OracleCi::new([(0usize, 1usize), (1usize, 2usize)]);
        let pc = Pc::new().with_fdr(false).with_ci(Arc::new(oracle));
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(3);
        let result = pc.run(&data, &vars, &mut ws, &ctx).unwrap();
        assert!(result.evidence.graph.has_edge(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)));
    }

    #[test]
    fn oracle_collider_orients() {
        // True: 0→1←2. Dependent: (0,1), (1,2). (0,2) independent with empty sepset.
        let data = tabular_n(3, 40);
        let vars = [VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2)];
        let oracle = OracleCi::new([(0usize, 1usize), (1usize, 2usize)]);
        let pc = Pc::new().with_fdr(false).with_ci(Arc::new(oracle));
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(2);
        let result = pc.run(&data, &vars, &mut ws, &ctx).unwrap();
        let g = &result.evidence.graph;
        assert_eq!(
            g.edge_between(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1))
                .unwrap()
                .parent_child(),
            Some((DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)))
        );
        assert_eq!(
            g.edge_between(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1))
                .unwrap()
                .parent_child(),
            Some((DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)))
        );
    }

    fn independent_gaussians(ncols: usize, nrows: usize, seed: u64) -> TabularData {
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
        let mut state = seed;
        let mut next_gauss = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u1 = ((state >> 33) as f64 / f64::from(u32::MAX)).clamp(1e-12, 1.0 - 1e-12);
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u2 = ((state >> 33) as f64 / f64::from(u32::MAX)).clamp(1e-12, 1.0 - 1e-12);
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };
        let owned: Vec<OwnedColumn> = (0..ncols)
            .map(|i| {
                let vals: Vec<f64> = (0..nrows).map(|_| next_gauss()).collect();
                OwnedColumn::Float64(
                    Float64Column::new(
                        VariableId::from_raw(i as u32),
                        Arc::from(vals),
                        ValidityBitmap::all_valid(nrows),
                    )
                    .unwrap(),
                )
            })
            .collect();
        let storage = OwnedColumnarStorage::try_new(schema, owned, None, None).unwrap();
        TabularData::new(storage)
    }

    /// Under independent Gaussian noise, PC skeleton edge retention should track α.
    ///
    /// Scheduled via `scripts/gate_calibration.sh`. Loose band:
    /// with `N_SIM · C(p,2)` pair-trials the Monte Carlo SE near α=0.05 is small;
    /// we accept roughly ±4 SE plus a hard floor/ceiling for small budgets.
    /// Raised `N_SIM` (80) so total pair-trials ≈ 800 and MC SE(α) ≈ 0.0077.
    #[test]
    #[ignore = "calibration: run via scripts/gate_calibration.sh"]
    fn pc_null_fpr_near_alpha() {
        const N_VARS: usize = 5;
        const N_OBS: usize = 400;
        const N_SIM: u32 = 80;
        const ALPHA: f64 = 0.05;
        let n_pairs = (N_VARS * (N_VARS - 1)) / 2;
        let mut constraints = DiscoveryConstraints::default();
        constraints.alpha = ALPHA;
        constraints.max_cond_size = 2;
        constraints.temporal = crate::TemporalConstraints {
            max_lag: Lag::CONTEMPORANEOUS,
            min_lag: Lag::CONTEMPORANEOUS,
        };
        let pc = Pc::new().with_fdr(false).with_constraints(constraints);
        let vars: Vec<VariableId> = (0..N_VARS as u32).map(VariableId::from_raw).collect();
        let mut retained = 0u32;
        let mut total = 0u32;
        for s in 0..N_SIM {
            let data = independent_gaussians(N_VARS, N_OBS, 9000 + u64::from(s));
            let mut ws = DiscoveryWorkspace::default();
            let ctx = ExecutionContext::for_tests(100 + u64::from(s));
            let result = pc.run(&data, &vars, &mut ws, &ctx).unwrap();
            let g = &result.evidence.graph;
            let mut edges = 0u32;
            for i in 0..N_VARS {
                for j in (i + 1)..N_VARS {
                    if g.has_edge(DenseNodeId::from_raw(i as u32), DenseNodeId::from_raw(j as u32))
                    {
                        edges += 1;
                    }
                }
            }
            retained += edges;
            total += n_pairs as u32;
        }
        let rate = f64::from(retained) / f64::from(total);
        let se = (ALPHA * (1.0 - ALPHA) / f64::from(total)).sqrt();
        let lo = (ALPHA - 4.0 * se).max(0.01);
        let hi = (ALPHA + 4.0 * se).min(0.15);
        assert!(
            rate >= lo && rate <= hi,
            "PC null skeleton edge rate={rate:.3} outside [{lo:.3}, {hi:.3}] \
             ({retained}/{total}; α={ALPHA})"
        );
    }
}
