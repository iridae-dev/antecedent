//! Really Fast Causal Inference (RFCI) → static [`Pag`] (Colombo & Maathuis 2012).
//!
//! Skips FCI's Possible-D-Sep subset search. Instead:
//! 1. PC-style skeleton
//! 2. Lemma 3.1 unshielded-triple checks (local CI; may remove edges)
//! 3. Zhang R1–R3 / R8–R10 plus Lemma 3.2 discriminating-path CI checks
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::zero_sized_map_values
)]

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use causal_core::{AssumptionSet, ExecutionContext, Lag, VariableId};
use causal_data::TabularData;
use causal_graph::{DenseNodeId, Endpoint, Pag, PagReview};
use causal_stats::{
    CiBatchRequest, CiPreparationPlan, CiQuery, ConditionalIndependence, ConfidenceMethod,
    FdrAdjustment, PartialCorrelation, PreparedCiTest,
};

use crate::combinations::for_each_combination_vars;
use crate::constraints::DiscoveryConstraints;
use crate::discriminating_paths::{
    discriminating_implies_collider, find_discriminating_paths_with_budget,
};
use crate::engine::DiscoveryWorkspace;
use crate::error::DiscoveryError;
use crate::evidence::threshold_scored_links;
use crate::fci::{
    StaticPagDiscoveryResult, build_pag_circle_skeleton, load_sepsets_into_state, record_sepset,
};
use crate::orientation::{OrientationError, OrientationState, RuleDelta};
use crate::pc::{adjacent_vars, collect_float_columns, edge_key};
use crate::result::{
    AlgorithmRecord, DiscoveryDiagnostic, DiscoveryIteration, DiscoveryPerformanceRecord,
    DiscoveryResult, EdgeEvidence, EvidenceSource, GraphEvidence, LaggedLink, PcSepsets,
    ScoredLink,
};
use crate::rule_scheduling::{
    FciOrientationRule, LpcmciR1, LpcmciR2, LpcmciR3, LpcmciR8, LpcmciR9, LpcmciR10,
    run_fci_orientation_to_fixed_point,
};

/// Classic RFCI over tabular (non-temporal) data.
#[derive(Clone)]
pub struct Rfci {
    /// Constraints / alpha / max conditioning size.
    pub constraints: DiscoveryConstraints,
    /// Pluggable CI test.
    pub ci: Arc<dyn ConditionalIndependence + Send + Sync>,
    /// Multiple-testing adjustment (`None` = off).
    pub fdr: Option<FdrAdjustment>,
}

impl std::fmt::Debug for Rfci {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Rfci")
            .field("constraints", &self.constraints)
            .field("ci", &"<dyn ConditionalIndependence>")
            .field("fdr", &self.fdr)
            .finish()
    }
}

impl Default for Rfci {
    fn default() -> Self {
        Self::new()
    }
}

impl Rfci {
    /// Default RFCI with `ParCorr` and BH FDR.
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

    /// Run RFCI.
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
    ) -> Result<StaticPagDiscoveryResult, DiscoveryError> {
        self.constraints.validate()?;
        if variables.is_empty() {
            return Err(DiscoveryError::Unsupported {
                message: "RFCI requires at least one variable",
            });
        }

        let col_owned = collect_float_columns(data, variables)?;
        let cols: Vec<&[f64]> = col_owned.iter().map(AsRef::as_ref).collect();
        let n = cols[0].len();
        if n < 3 {
            return Err(DiscoveryError::stats_msg("insufficient rows for RFCI"));
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
        let mut combo_scratch = Vec::new();

        // --- Phase 1: PC-style adjacency (no Possible-D-Sep) ---
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
                label: Arc::from(format!("rfci.pc.depth.{depth}")),
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
        if self.fdr.is_some() {
            adj.retain(|k, ()| kept.contains(k));
        }

        let dense_of = |v: VariableId| -> Result<DenseNodeId, DiscoveryError> {
            let idx = *var_index
                .get(&v)
                .ok_or_else(|| DiscoveryError::data_msg(format!("unknown variable {v:?}")))?;
            Ok(DenseNodeId::from_raw(u32::try_from(idx).expect("fit")))
        };

        let mut pag = build_pag_circle_skeleton(variables, &var_index, &adj)?;

        // --- Phase 2: Lemma 3.1 unshielded triples (no Possible-D-Sep) ---
        let mut lemma_tests = 0u64;
        lemma_tests += self.rfci_unshielded_triples(
            &cols,
            &var_index,
            variables,
            &mut adj,
            &mut sepsets,
            &mut pag,
            &dense_of,
            workspace,
            ctx,
            alpha,
            &mut combo_scratch,
        )?;
        ci_tests += lemma_tests;
        iterations.push(DiscoveryIteration {
            label: Arc::from("rfci.lemma31_unshielded"),
            ci_tests: lemma_tests,
        });

        let mut state = OrientationState::default();
        load_sepsets_into_state(&sepsets, &dense_of, &mut state)?;

        // --- Phase 3: Zhang R1–R3 / R8–R10 + RFCI discriminating paths ---
        let zhang: [&dyn FciOrientationRule; 6] =
            [&LpcmciR1, &LpcmciR2, &LpcmciR3, &LpcmciR8, &LpcmciR9, &LpcmciR10];
        let mut orient_conflicts = 0u32;
        let mut rounds = 0u32;
        while rounds < 10_000 {
            rounds += 1;
            let d = run_fci_orientation_to_fixed_point(&mut pag, &zhang, &mut state)?;
            orient_conflicts = orient_conflicts.max(d.conflicts);
            let mut disc_changed = false;
            let disc = self.rfci_discriminating_paths(
                &cols,
                &var_index,
                variables,
                &mut adj,
                &mut sepsets,
                &mut pag,
                &mut state,
                &dense_of,
                workspace,
                ctx,
                alpha,
                &mut combo_scratch,
                &mut ci_tests,
            )?;
            if disc.edges_changed > 0 {
                disc_changed = true;
                orient_conflicts = orient_conflicts.max(disc.conflicts);
                // New edges removed ⇒ re-check Lemma 3.1 triples.
                let extra = self.rfci_unshielded_triples(
                    &cols,
                    &var_index,
                    variables,
                    &mut adj,
                    &mut sepsets,
                    &mut pag,
                    &dense_of,
                    workspace,
                    ctx,
                    alpha,
                    &mut combo_scratch,
                )?;
                ci_tests += extra;
                load_sepsets_into_state(&sepsets, &dense_of, &mut state)?;
            }
            if d.edges_changed == 0 && !disc_changed {
                break;
            }
        }

        let mut diagnostics = Vec::new();
        if state.conflicts > 0 || orient_conflicts > 0 {
            diagnostics.push(DiscoveryDiagnostic {
                code: Arc::from("rfci.orientation_conflict"),
                message: Arc::from(format!(
                    "{} orientation conflict(s)",
                    state.conflicts.max(orient_conflicts)
                )),
            });
        }

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
                    provenance: Arc::from([Arc::from("rfci")]),
                }
            })
            .collect();

        let evidence = GraphEvidence {
            graph: pag.clone(),
            edge_evidence: Arc::from(edge_evidence),
            links: Arc::from(scored),
            source: EvidenceSource::Discovery { algorithm: Arc::from("rfci") },
        };
        let review = PagReview::from_pag(pag, "rfci");

        Ok(DiscoveryResult {
            evidence,
            review,
            algorithm: AlgorithmRecord {
                id: Arc::from("rfci"),
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

    /// Colombo–Maathuis Lemma 3.1: local CI around unshielded triples.
    #[allow(clippy::too_many_arguments)]
    fn rfci_unshielded_triples(
        &self,
        cols: &[&[f64]],
        var_index: &HashMap<VariableId, usize>,
        variables: &[VariableId],
        adj: &mut HashMap<(u32, u32), ()>,
        sepsets: &mut PcSepsets,
        pag: &mut Pag,
        dense_of: &dyn Fn(VariableId) -> Result<DenseNodeId, DiscoveryError>,
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
        alpha: f64,
        combo_scratch: &mut Vec<VariableId>,
    ) -> Result<u64, DiscoveryError> {
        let mut tests = 0u64;
        let mut queue: VecDeque<(VariableId, VariableId, VariableId)> = VecDeque::new();
        let mut pending: HashSet<(u32, u32, u32)> = HashSet::new();

        let enqueue = |queue: &mut VecDeque<_>,
                       pending: &mut HashSet<_>,
                       a: VariableId,
                       b: VariableId,
                       c: VariableId| {
            let (lo, hi) = if a.raw() <= c.raw() { (a, c) } else { (c, a) };
            let key = (lo.raw(), b.raw(), hi.raw());
            if pending.insert(key) {
                queue.push_back((lo, b, hi));
            }
        };

        for &b in variables {
            let nbrs = adjacent_vars(b, adj, variables);
            for (i, &a) in nbrs.iter().enumerate() {
                for &c in &nbrs[i + 1..] {
                    if !adj.contains_key(&edge_key(a, c)) {
                        enqueue(&mut queue, &mut pending, a, b, c);
                    }
                }
            }
        }

        while let Some((a, b, c)) = queue.pop_front() {
            pending.remove(&(a.raw(), b.raw(), c.raw()));
            if !adj.contains_key(&edge_key(a, b)) || !adj.contains_key(&edge_key(b, c)) {
                continue;
            }
            if adj.contains_key(&edge_key(a, c)) {
                continue; // no longer unshielded
            }
            let Some(sep_ac) = sepset_vars(sepsets, a, c) else {
                continue;
            };
            let cond: Vec<VariableId> = sep_ac.into_iter().filter(|&z| z != b).collect();

            // a ⊥? b | sep(a,c)\{b}
            let (_stat_ab, p_ab) = self.ci_test(cols, var_index, a, b, &cond, workspace, ctx)?;
            tests += 1;
            if p_ab > alpha {
                let minimal = self.minimize_sepset(
                    cols, var_index, a, b, &cond, workspace, ctx, alpha, &mut tests,
                )?;
                self.remove_edge_update(
                    a,
                    b,
                    &minimal,
                    adj,
                    sepsets,
                    pag,
                    dense_of,
                    variables,
                    &mut queue,
                    &mut pending,
                )?;
                continue;
            }

            let (_stat_bc, p_bc) = self.ci_test(cols, var_index, b, c, &cond, workspace, ctx)?;
            tests += 1;
            if p_bc > alpha {
                let minimal = self.minimize_sepset(
                    cols, var_index, b, c, &cond, workspace, ctx, alpha, &mut tests,
                )?;
                self.remove_edge_update(
                    b,
                    c,
                    &minimal,
                    adj,
                    sepsets,
                    pag,
                    dense_of,
                    variables,
                    &mut queue,
                    &mut pending,
                )?;
                continue;
            }

            // Both dependent: orient v-structure iff b ∉ sep(a,c).
            let b_in_sep = sepset_vars(sepsets, a, c).is_some_and(|s| s.iter().any(|&z| z == b));
            if !b_in_sep {
                let ad = dense_of(a)?;
                let bd = dense_of(b)?;
                let cd = dense_of(c)?;
                // a *→ b ←* c (keep far marks; set arrow at b).
                orient_arrow_into(pag, ad, bd)?;
                orient_arrow_into(pag, cd, bd)?;
            }
            let _ = combo_scratch;
        }
        Ok(tests)
    }

    #[allow(clippy::too_many_arguments)]
    fn rfci_discriminating_paths(
        &self,
        cols: &[&[f64]],
        var_index: &HashMap<VariableId, usize>,
        variables: &[VariableId],
        adj: &mut HashMap<(u32, u32), ()>,
        sepsets: &mut PcSepsets,
        pag: &mut Pag,
        state: &mut OrientationState,
        dense_of: &dyn Fn(VariableId) -> Result<DenseNodeId, DiscoveryError>,
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
        alpha: f64,
        combo_scratch: &mut Vec<VariableId>,
        ci_tests: &mut u64,
    ) -> Result<RuleDelta, DiscoveryError> {
        let mut delta = RuleDelta::default();
        let (paths, truncated) = find_discriminating_paths_with_budget(pag, 64, 8);
        if truncated {
            return Err(DiscoveryError::from(OrientationError::SearchBudgetExhausted {
                rule: "rfci.discriminating_path",
                max_paths: 64,
                max_len: 8,
            }));
        }

        for path in paths {
            let a = path.a();
            let c = path.c();
            let b = path.b();
            let d_k = path.d_k();
            let a_v = variables[a.as_usize()];
            let b_v = variables[b.as_usize()];
            let Some(sep_ab) = sepset_vars(sepsets, a_v, b_v) else {
                continue;
            };

            // Lemma 3.2: consecutive pairs must stay dependent given all subsets of sep\{pair}.
            let mut remove: Option<(VariableId, VariableId, Vec<VariableId>)> = None;
            'pairs: for w in path.nodes.windows(2) {
                let u = variables[w[0].as_usize()];
                let v = variables[w[1].as_usize()];
                let pool: Vec<VariableId> =
                    sep_ab.iter().copied().filter(|&z| z != u && z != v).collect();
                for depth in 0..=pool.len() {
                    let mut sets = Vec::new();
                    for_each_combination_vars(&pool, depth, combo_scratch, |z| {
                        sets.push(z.to_vec());
                        true
                    });
                    for z in &sets {
                        let (_s, p) = self.ci_test(cols, var_index, u, v, z, workspace, ctx)?;
                        *ci_tests += 1;
                        if p > alpha {
                            let minimal = self.minimize_sepset(
                                cols, var_index, u, v, z, workspace, ctx, alpha, ci_tests,
                            )?;
                            remove = Some((u, v, minimal));
                            break 'pairs;
                        }
                    }
                }
            }

            if let Some((u, v, minimal)) = remove {
                if self.static_required(u, v) {
                    continue;
                }
                let key = edge_key(u, v);
                if adj.remove(&key).is_some() {
                    record_sepset(sepsets, u, v, &minimal);
                    let ud = dense_of(u)?;
                    let vd = dense_of(v)?;
                    let _ = pag.remove_edge(ud, vd);
                    state.set_sepset(
                        ud,
                        vd,
                        Arc::from(
                            minimal.iter().filter_map(|x| dense_of(*x).ok()).collect::<Vec<_>>(),
                        ),
                    );
                    delta.edges_changed += 1;
                }
                continue;
            }

            // Standard R4 orientation (Zhang) after Lemma 3.2 checks pass.
            let c_in_sep = sep_ab.iter().any(|&z| dense_of(z).ok().is_some_and(|d| d == c));
            let collider = discriminating_implies_collider(c_in_sep);
            let Some(e_cb) = pag.edge_between(c, b) else {
                continue;
            };
            let mark_at_c = if e_cb.a == c { e_cb.at_a } else { e_cb.at_b };
            if !matches!(mark_at_c, Endpoint::Circle) {
                continue;
            }
            if collider {
                if set_arrow_at(pag, state, &mut delta, c, d_k)? {
                    delta.edges_changed += 1;
                }
                if set_arrow_at(pag, state, &mut delta, c, b)? {
                    delta.edges_changed += 1;
                }
            } else if set_marks_oriented(
                pag,
                state,
                &mut delta,
                c,
                b,
                Endpoint::Tail,
                Endpoint::Arrow,
            )? {
                delta.edges_changed += 1;
            }
        }
        delta.fixed_point = delta.edges_changed == 0;
        Ok(delta)
    }

    #[allow(clippy::too_many_arguments)]
    fn remove_edge_update(
        &self,
        x: VariableId,
        y: VariableId,
        minimal: &[VariableId],
        adj: &mut HashMap<(u32, u32), ()>,
        sepsets: &mut PcSepsets,
        pag: &mut Pag,
        dense_of: &dyn Fn(VariableId) -> Result<DenseNodeId, DiscoveryError>,
        variables: &[VariableId],
        queue: &mut VecDeque<(VariableId, VariableId, VariableId)>,
        pending: &mut HashSet<(u32, u32, u32)>,
    ) -> Result<(), DiscoveryError> {
        if self.static_required(x, y) {
            return Ok(());
        }
        let key = edge_key(x, y);
        if adj.remove(&key).is_none() {
            return Ok(());
        }
        record_sepset(sepsets, x, y, minimal);
        let xd = dense_of(x)?;
        let yd = dense_of(y)?;
        let _ = pag.remove_edge(xd, yd);

        // New unshielded triples created by removing x–y: for each common neighbor.
        let nx = adjacent_vars(x, adj, variables);
        let ny = adjacent_vars(y, adj, variables);
        for &b in &nx {
            if b != y && adj.contains_key(&edge_key(y, b)) && !adj.contains_key(&edge_key(x, y)) {
                // triple x-b-y is unshielded if x–y gone
                let (lo, hi) = if x.raw() <= y.raw() { (x, y) } else { (y, x) };
                let key = (lo.raw(), b.raw(), hi.raw());
                if pending.insert(key) {
                    queue.push_back((lo, b, hi));
                }
            }
        }
        for &b in &ny {
            if b != x && adj.contains_key(&edge_key(x, b)) {
                let (lo, hi) = if x.raw() <= y.raw() { (x, y) } else { (y, x) };
                let key = (lo.raw(), b.raw(), hi.raw());
                if pending.insert(key) {
                    queue.push_back((lo, b, hi));
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn minimize_sepset(
        &self,
        cols: &[&[f64]],
        var_index: &HashMap<VariableId, usize>,
        x: VariableId,
        y: VariableId,
        z: &[VariableId],
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
        alpha: f64,
        tests: &mut u64,
    ) -> Result<Vec<VariableId>, DiscoveryError> {
        let mut s = z.to_vec();
        let mut i = 0;
        while i < s.len() {
            let mut trial = s.clone();
            trial.remove(i);
            let (_stat, p) = self.ci_test(cols, var_index, x, y, &trial, workspace, ctx)?;
            *tests += 1;
            if p > alpha {
                s = trial;
            } else {
                i += 1;
            }
        }
        Ok(s)
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
        let prepared = workspace
            .prepared_ci
            .as_ref()
            .ok_or(DiscoveryError::Unsupported { message: "CI test used before prepare()" })?;
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

fn sepset_vars(sepsets: &PcSepsets, a: VariableId, b: VariableId) -> Option<Vec<VariableId>> {
    sepsets
        .get(&(a, Lag::CONTEMPORANEOUS, b, Lag::CONTEMPORANEOUS))
        .map(|s| s.iter().map(|(v, _)| *v).collect())
}

fn orient_arrow_into(
    pag: &mut Pag,
    from: DenseNodeId,
    into: DenseNodeId,
) -> Result<(), DiscoveryError> {
    let Some(e) = pag.edge_between(from, into) else {
        return Ok(());
    };
    let at_from = if e.a == from { e.at_a } else { e.at_b };
    let at_into = if e.a == into { e.at_a } else { e.at_b };
    if matches!(at_into, Endpoint::Arrow) {
        return Ok(());
    }
    if e.a == from {
        pag.set_marks(from, into, at_from, Endpoint::Arrow).map_err(DiscoveryError::from)?;
    } else {
        pag.set_marks(into, from, Endpoint::Arrow, at_from).map_err(DiscoveryError::from)?;
    }
    Ok(())
}

fn set_marks_oriented(
    graph: &mut Pag,
    state: &mut OrientationState,
    delta: &mut RuleDelta,
    a: DenseNodeId,
    b: DenseNodeId,
    at_a: Endpoint,
    at_b: Endpoint,
) -> Result<bool, OrientationError> {
    let Some(e) = graph.edge_between(a, b) else {
        return Err(OrientationError::msg("missing edge in set_marks_oriented"));
    };
    if matches!(e.at_a, Endpoint::Conflict) || matches!(e.at_b, Endpoint::Conflict) {
        return Ok(false);
    }
    let result = if e.a == a {
        graph.set_marks(a, b, at_a, at_b)
    } else {
        graph.set_marks(b, a, at_b, at_a)
    };
    match result {
        Ok(()) => Ok(true),
        Err(causal_graph::GraphError::Cycle { .. }) => {
            state.record_conflict(delta, a, b, "cycle");
            if graph.mark_conflict(a, b).is_ok() {
                delta.edges_changed += 1;
                delta.fixed_point = false;
            }
            Ok(false)
        }
        Err(err) => Err(OrientationError::from(err)),
    }
}

fn set_arrow_at(
    graph: &mut Pag,
    state: &mut OrientationState,
    delta: &mut RuleDelta,
    at: DenseNodeId,
    other: DenseNodeId,
) -> Result<bool, OrientationError> {
    let e = graph
        .edge_between(at, other)
        .ok_or(OrientationError::Precondition { message: "discriminating path missing edge" })?;
    let at_other = if e.a == other { e.at_a } else { e.at_b };
    set_marks_oriented(graph, state, delta, at, other, Endpoint::Arrow, at_other)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use causal_core::{
        CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
        VariableId,
    };
    use causal_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
    };
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
        let vars = [VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2)];
        let oracle = OracleCi::new([(0usize, 1usize), (1usize, 2usize)]);
        let rfci = Rfci::new().with_fdr(false).with_ci(Arc::new(oracle));
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let result = rfci.run(&data, &vars, &mut ws, &ctx).unwrap();
        let g = &result.evidence.graph;
        assert!(g.has_edge(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)));
        assert!(g.has_edge(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)));
        assert!(!g.has_edge(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)));
        assert_eq!(result.algorithm.id.as_ref(), "rfci");
        // No Possible-D-Sep iteration label.
        assert!(result.iterations.iter().all(|i| !i.label.contains("possible_d_sep")));
        assert!(result.iterations.iter().any(|i| i.label.contains("lemma31")));
    }

    #[test]
    fn oracle_collider_orients_into_middle() {
        let data = tabular_n(3, 40);
        let vars = [VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2)];
        let oracle = OracleCi::new([(0usize, 1usize), (1usize, 2usize)]);
        let rfci = Rfci::new().with_fdr(false).with_ci(Arc::new(oracle));
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(2);
        let result = rfci.run(&data, &vars, &mut ws, &ctx).unwrap();
        let g = &result.evidence.graph;
        let e01 = g.edge_between(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let e21 = g.edge_between(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
        let at_1_from_0 = if e01.a.raw() == 1 { e01.at_a } else { e01.at_b };
        let at_1_from_2 = if e21.a.raw() == 1 { e21.at_a } else { e21.at_b };
        assert!(matches!(at_1_from_0, Endpoint::Arrow));
        assert!(matches!(at_1_from_2, Endpoint::Arrow));
    }
}
