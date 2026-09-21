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

use antecedent_core::{ExecutionContext, Lag, VariableId};
use antecedent_data::TabularData;
use antecedent_graph::{DenseNodeId, Endpoint, Pag, PagReview};
use antecedent_stats::{
    CiPreparationPlan, ConditionalIndependence, ConfidenceMethod, FdrAdjustment,
    PartialCorrelation, PreparedCiTest,
};

use crate::combinations::for_each_combination_vars;
use crate::constraints::DiscoveryConstraints;
use crate::discriminating_paths::DiscriminatingPathBudget;
use crate::discriminating_paths::{
    discriminating_implies_collider, find_discriminating_paths_for_edge,
};
use crate::engine::DiscoveryWorkspace;
use crate::error::DiscoveryError;
use crate::fci::{StaticPagDiscoveryResult, build_pag_circle_skeleton, load_sepsets_into_state};
use crate::orientation::{OrientationState, RuleDelta};
use crate::pc::{adjacent_vars, collect_float_columns, edge_key};
use crate::result::{
    DiscoveryDiagnostic, DiscoveryIteration, DiscoveryPerformanceRecord, DiscoveryResult,
    EvidenceSource, GraphEvidence, PcSepsets, discovery_assumptions,
};
use crate::rule_scheduling::{
    FciOrientationRule, LpcmciR1, LpcmciR2, LpcmciR3, LpcmciR8, LpcmciR9, LpcmciR10,
    orient_collider_leg, orient_discriminated_collider, run_fci_orientation_to_fixed_point,
    set_marks_oriented,
};
use crate::static_skeleton::{
    StaticSkeleton, StaticSkeletonInput, record_sepset, run_static_skeleton, skeleton_scored_links,
    static_ci_test, static_edge_evidence,
};

/// Classic RFCI over tabular (non-temporal) data.
#[derive(Clone)]
pub struct Rfci {
    /// Constraints / alpha / max conditioning size.
    pub constraints: DiscoveryConstraints,
    /// Pluggable CI test.
    pub ci: Arc<dyn ConditionalIndependence + Send + Sync>,
    /// Multiple-testing adjustment (`None` = off). Configured adjustments are refused until
    /// the complete adaptive CI family can be adjusted coherently.
    pub fdr: Option<FdrAdjustment>,
    /// Per-edge bounds of the discriminating-path search.
    pub discriminating_path_budget: DiscriminatingPathBudget,
}

impl std::fmt::Debug for Rfci {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Rfci")
            .field("constraints", &self.constraints)
            .field("ci", &"<dyn ConditionalIndependence>")
            .field("fdr", &self.fdr)
            .field("discriminating_path_budget", &self.discriminating_path_budget)
            .finish()
    }
}

impl Default for Rfci {
    fn default() -> Self {
        Self::new()
    }
}

impl Rfci {
    /// Default RFCI with `ParCorr` and no incomplete adaptive-family FDR adjustment.
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
            fdr: None,
            discriminating_path_budget: DiscriminatingPathBudget::default(),
        }
    }

    /// Per-edge bounds of the discriminating-path search. An edge that exhausts its budget
    /// keeps its circle mark and is reported in a diagnostic; the run does not fail.
    #[must_use]
    pub fn with_discriminating_path_budget(mut self, budget: DiscriminatingPathBudget) -> Self {
        self.discriminating_path_budget = budget;
        self
    }

    /// Configure constraints.
    #[must_use]
    pub fn with_constraints(mut self, constraints: DiscoveryConstraints) -> Self {
        self.constraints = constraints;
        self
    }

    /// Request / clear BH FDR. Enabling it is currently refused by [`Self::run`].
    #[must_use]
    pub fn with_fdr(mut self, fdr: bool) -> Self {
        self.fdr = fdr.then(|| FdrAdjustment::bh().with_exclude_contemporaneous(false));
        self
    }

    /// Full FDR configuration.
    ///
    /// RFCI currently refuses a configured adjustment because Lemma 3.1, discriminating-path,
    /// and sepset-minimization tests belong to the same adaptive CI family as adjacency tests.
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
    /// Data, CI, orientation failures, or an FDR configuration that cannot cover the complete
    /// adaptive CI family.
    pub fn run(
        &self,
        data: &TabularData,
        variables: &[VariableId],
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<StaticPagDiscoveryResult, DiscoveryError> {
        self.constraints.validate()?;
        if self.fdr.is_some() {
            return Err(DiscoveryError::unsupported(
                "RFCI FDR is refused until adjustment covers adjacency, Lemma 3.1, discriminating-path, and sepset-minimization CI tests",
            ));
        }
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
        let skel = run_static_skeleton(
            &StaticSkeletonInput {
                ci: &*self.ci,
                constraints: &self.constraints,
                cols: &cols,
                var_index: &var_index,
                variables,
                label: "rfci.pc.depth",
            },
            workspace,
            ctx,
        )?;
        let mut scored = skeleton_scored_links(&skel);
        let StaticSkeleton { mut adj, mut sepsets, mut ci_tests, mut iterations, .. } = skel;
        let mut combo_scratch = Vec::new();
        let dense_of = |v: VariableId| crate::pipeline::dense_of(&var_index, v);

        let mut pag = build_pag_circle_skeleton(variables, &var_index, &adj)?;

        let mut state = OrientationState {
            discriminating_budget: self.discriminating_path_budget,
            ..OrientationState::default()
        };

        // --- Phase 2: Lemma 3.1 unshielded triples (no Possible-D-Sep) ---
        let mut lemma_tests = 0u64;
        lemma_tests += self.rfci_unshielded_triples(
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
        )?;
        ci_tests += lemma_tests;
        iterations.push(DiscoveryIteration {
            label: Arc::from("rfci.lemma31_unshielded"),
            ci_tests: lemma_tests,
        });

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
                    &mut state,
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
        if !state.discriminating_skipped.is_empty() {
            diagnostics.push(DiscoveryDiagnostic {
                code: Arc::from("rfci.discriminating_path_budget"),
                message: Arc::from(format!(
                    "discriminating-path search exhausted its budget on {} edge(s); their circle \
                     marks were left unresolved (sound, possibly incomplete)",
                    state.discriminating_skipped.len()
                )),
            });
        }
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

        let edge_evidence = static_edge_evidence(&scored, &sepsets, "rfci");

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
            algorithm: crate::pipeline::algorithm_record(
                "rfci",
                format!("alpha={},max_cond={},fdr={}", alpha, max_cond, self.fdr.is_some()),
            ),
            assumptions: discovery_assumptions("rfci", false),
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
        state: &mut OrientationState,
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
            let (_stat_ab, p_ab) = static_ci_test(
                &*self.ci,
                &self.constraints,
                cols,
                var_index,
                a,
                b,
                &cond,
                workspace,
                ctx,
            )?;
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

            let (_stat_bc, p_bc) = static_ci_test(
                &*self.ci,
                &self.constraints,
                cols,
                var_index,
                b,
                c,
                &cond,
                workspace,
                ctx,
            )?;
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
                let mut delta = RuleDelta::default();
                orient_collider_leg(pag, state, &mut delta, ad, bd)?;
                orient_collider_leg(pag, state, &mut delta, cd, bd)?;
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
        // Per-edge budgets: an edge whose search runs out keeps its circle (sound) and is
        // recorded, instead of aborting the whole run.
        let budget = state.discriminating_budget;
        let mut paths = Vec::new();
        for i in 0..pag.node_count() {
            let b = DenseNodeId::from_raw(i as u32);
            for (c, _at_b, at_c) in pag.neighbors(b) {
                if !matches!(at_c, Endpoint::Circle) {
                    continue;
                }
                let (found, truncated) = find_discriminating_paths_for_edge(pag, b, c, budget);
                if truncated {
                    state.discriminating_skipped.insert((c.raw(), b.raw()));
                }
                paths.extend(found);
            }
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
                        let (_s, p) = static_ci_test(
                            &*self.ci,
                            &self.constraints,
                            cols,
                            var_index,
                            u,
                            v,
                            z,
                            workspace,
                            ctx,
                        )?;
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
                if self.constraints.static_required(u, v) {
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
                if orient_discriminated_collider(pag, state, &mut delta, d_k, c, b)? {
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
        if self.constraints.static_required(x, y) {
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
            let (_stat, p) = static_ci_test(
                &*self.ci,
                &self.constraints,
                cols,
                var_index,
                x,
                y,
                &trial,
                workspace,
                ctx,
            )?;
            *tests += 1;
            if p > alpha {
                s = trial;
            } else {
                i += 1;
            }
        }
        Ok(s)
    }
}

fn sepset_vars(sepsets: &PcSepsets, a: VariableId, b: VariableId) -> Option<Vec<VariableId>> {
    sepsets
        .get(&(a, Lag::CONTEMPORANEOUS, b, Lag::CONTEMPORANEOUS))
        .map(|s| s.iter().map(|(v, _)| *v).collect())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use antecedent_core::{
        CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
        VariableId,
    };
    use antecedent_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
    };
    use antecedent_graph::Endpoint;
    use antecedent_stats::OracleCi;

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
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../../../conformance/discovery/rfci/expected.json"))
                .unwrap();
        assert_eq!(fixture["cases"][0]["adjacencies"].as_array().unwrap().len(), 2);
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
    fn fdr_refuses_an_incomplete_adaptive_family() {
        let data = tabular_n(2, 20);
        let vars = [VariableId::from_raw(0), VariableId::from_raw(1)];
        let mut ws = DiscoveryWorkspace::default();
        let err = Rfci::new()
            .with_fdr(true)
            .run(&data, &vars, &mut ws, &ExecutionContext::for_tests(4))
            .unwrap_err();
        assert!(
            matches!(err, DiscoveryError::Unsupported { message } if message.contains("Lemma 3.1")),
            "{err}"
        );
    }

    #[test]
    fn oracle_collider_orients_into_middle() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../../../conformance/discovery/rfci/expected.json"))
                .unwrap();
        assert_eq!(fixture["cases"][1]["arrowheads_at"].as_array().unwrap().len(), 2);
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
