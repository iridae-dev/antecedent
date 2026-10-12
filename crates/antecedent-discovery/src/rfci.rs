//! Really Fast Causal Inference (RFCI) → static [`Pag`] (Colombo & Maathuis 2012).
//!
//! Skips FCI's Possible-D-Sep subset search. Instead:
//! 1. PC-style skeleton
//! 2. Lemma 3.1 unshielded-triple checks (local CI; may remove edges)
//! 3. Zhang R1–R3 / R8–R10 plus Lemma 3.2 discriminating-path CI checks
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

use std::collections::{HashSet, VecDeque};
use std::sync::Arc;

use antecedent_core::{ExecutionContext, Lag, VariableId};
use antecedent_data::TabularData;
use antecedent_graph::{DenseNodeId, Endpoint, Pag, PagReview};
use antecedent_stats::{ConditionalIndependence, FdrAdjustment, PartialCorrelation};

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
use crate::pc::{adjacent_vars, edge_key};
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
    StaticCiRun, StaticSkeleton, StaticSkeletonInput, record_sepset, run_static_skeleton,
    skeleton_scored_links, static_edge_evidence,
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

crate::debug_with_ci!(Rfci, discriminating_path_budget);

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
    #[allow(
        clippy::too_many_lines,
        reason = "one algorithm pipeline over shared mutable graph state; the phases are marked in place"
    )]
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
        let (col_owned, var_index) = crate::pc::prepare_static_columns(
            crate::pc::StaticColumnRequest {
                ci: &*self.ci,
                significance: self.constraints.significance,
                alpha: self.constraints.alpha,
                fdr: false,
                empty_variables: "RFCI requires at least one variable",
                insufficient_rows: "insufficient rows for RFCI",
            },
            data,
            variables,
            workspace,
            ctx,
        )?;
        let cols: Vec<&[f64]> = col_owned.iter().map(AsRef::as_ref).collect();

        let alpha = self.constraints.alpha;
        let max_cond = self.constraints.max_cond_size;
        let input = StaticSkeletonInput {
            ci: &*self.ci,
            constraints: &self.constraints,
            cols: &cols,
            var_index: &var_index,
            variables,
            label: "rfci.pc.depth",
        };
        let skel = run_static_skeleton(&input, workspace, ctx)?;
        let mut scored = skeleton_scored_links(&skel);
        let StaticSkeleton { mut adj, mut sepsets, mut ci_tests, mut iterations, .. } = skel;
        let mut combo_scratch = Vec::new();
        let dense_of = |v: VariableId| crate::pipeline::dense_of(&var_index, v);
        let mut ci = StaticCiRun::new(&input, workspace, ctx, alpha);

        let mut pag = build_pag_circle_skeleton(variables, &var_index, &adj)?;

        let mut state = OrientationState {
            discriminating_budget: self.discriminating_path_budget,
            ..OrientationState::default()
        };

        // --- Phase 2: Lemma 3.1 unshielded triples (no Possible-D-Sep) ---
        let tests_before = ci.tests;
        rfci_unshielded_triples(
            &mut ci,
            &mut RfciGraph {
                adj: &mut adj,
                sepsets: &mut sepsets,
                pag: &mut pag,
                state: &mut state,
                dense_of: &dense_of,
            },
        )?;
        let lemma_tests = ci.tests - tests_before;
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
            let tests_before = ci.tests;
            let disc = rfci_discriminating_paths(
                &mut ci,
                &mut RfciGraph {
                    adj: &mut adj,
                    sepsets: &mut sepsets,
                    pag: &mut pag,
                    state: &mut state,
                    dense_of: &dense_of,
                },
                &mut combo_scratch,
            )?;
            ci_tests += ci.tests - tests_before;
            if disc.edges_changed > 0 {
                disc_changed = true;
                orient_conflicts = orient_conflicts.max(disc.conflicts);
                // New edges removed ⇒ re-check Lemma 3.1 triples.
                let tests_before = ci.tests;
                rfci_unshielded_triples(
                    &mut ci,
                    &mut RfciGraph {
                        adj: &mut adj,
                        sepsets: &mut sepsets,
                        pag: &mut pag,
                        state: &mut state,
                        dense_of: &dense_of,
                    },
                )?;
                ci_tests += ci.tests - tests_before;
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
        diagnostics.extend(crate::rule_scheduling::pd_path_budget_diagnostic(&state, "rfci"));
        if state.conflicts > 0 || orient_conflicts > 0 {
            diagnostics.push(DiscoveryDiagnostic {
                code: Arc::from("rfci.orientation_conflict"),
                message: Arc::from(format!(
                    "{} orientation conflict(s)",
                    state.conflicts.max(orient_conflicts)
                )),
            });
        }

        scored.retain(|s| adj.contains(&edge_key(s.link.source, s.link.target)));

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
}

/// The mutable graph state the RFCI phases edit: skeleton, separating sets, PAG and marks.
struct RfciGraph<'a> {
    adj: &'a mut HashSet<(u32, u32)>,
    sepsets: &'a mut PcSepsets,
    pag: &'a mut Pag,
    state: &'a mut OrientationState,
    dense_of: &'a dyn Fn(VariableId) -> Result<DenseNodeId, DiscoveryError>,
}

/// Work queue of unshielded triples `(lo, middle, hi)`, each queued at most once.
#[derive(Default)]
struct UnshieldedTriples {
    queue: VecDeque<(VariableId, VariableId, VariableId)>,
    pending: HashSet<(u32, u32, u32)>,
}

impl UnshieldedTriples {
    fn push(&mut self, a: VariableId, b: VariableId, c: VariableId) {
        let (lo, hi) = if a.raw() <= c.raw() { (a, c) } else { (c, a) };
        if self.pending.insert((lo.raw(), b.raw(), hi.raw())) {
            self.queue.push_back((lo, b, hi));
        }
    }

    fn pop(&mut self) -> Option<(VariableId, VariableId, VariableId)> {
        let (a, b, c) = self.queue.pop_front()?;
        self.pending.remove(&(a.raw(), b.raw(), c.raw()));
        Some((a, b, c))
    }
}

/// Colombo–Maathuis Lemma 3.1: local CI around unshielded triples.
fn rfci_unshielded_triples(
    ci: &mut StaticCiRun<'_>,
    graph: &mut RfciGraph<'_>,
) -> Result<(), DiscoveryError> {
    let variables = ci.input.variables;
    let mut triples = UnshieldedTriples::default();

    for &b in variables {
        let nbrs = adjacent_vars(b, graph.adj, variables);
        for (i, &a) in nbrs.iter().enumerate() {
            for &c in &nbrs[i + 1..] {
                if !graph.adj.contains(&edge_key(a, c)) {
                    triples.push(a, b, c);
                }
            }
        }
    }

    while let Some((a, b, c)) = triples.pop() {
        if !graph.adj.contains(&edge_key(a, b)) || !graph.adj.contains(&edge_key(b, c)) {
            continue;
        }
        if graph.adj.contains(&edge_key(a, c)) {
            continue; // no longer unshielded
        }
        let Some(sep_ac) = sepset_vars(graph.sepsets, a, c) else {
            continue;
        };
        let cond: Vec<VariableId> = sep_ac.into_iter().filter(|&z| z != b).collect();

        // a ⊥? b | sep(a,c)\{b}
        if ci.p_value(a, b, &cond)? > ci.alpha {
            let minimal = ci.minimize_sepset(a, b, &cond)?;
            remove_edge_update(ci, graph, (a, b), &minimal, &mut triples)?;
            continue;
        }

        if ci.p_value(b, c, &cond)? > ci.alpha {
            let minimal = ci.minimize_sepset(b, c, &cond)?;
            remove_edge_update(ci, graph, (b, c), &minimal, &mut triples)?;
            continue;
        }

        // Both dependent: orient v-structure iff b ∉ sep(a,c).
        let b_in_sep = sepset_vars(graph.sepsets, a, c).is_some_and(|s| s.iter().any(|&z| z == b));
        if !b_in_sep {
            let ad = (graph.dense_of)(a)?;
            let bd = (graph.dense_of)(b)?;
            let cd = (graph.dense_of)(c)?;
            // a *→ b ←* c (keep far marks; set arrow at b).
            let mut delta = RuleDelta::default();
            orient_collider_leg(graph.pag, graph.state, &mut delta, ad, bd)?;
            orient_collider_leg(graph.pag, graph.state, &mut delta, cd, bd)?;
        }
    }
    Ok(())
}

/// Zhang R4 with Colombo–Maathuis Lemma 3.2: CI-check each discriminating path, then orient.
fn rfci_discriminating_paths(
    ci: &mut StaticCiRun<'_>,
    graph: &mut RfciGraph<'_>,
    combo_scratch: &mut Vec<VariableId>,
) -> Result<RuleDelta, DiscoveryError> {
    let (adj, sepsets, pag, state) =
        (&mut *graph.adj, &mut *graph.sepsets, &mut *graph.pag, &mut *graph.state);
    let dense_of = graph.dense_of;
    let variables = ci.input.variables;
    let mut delta = RuleDelta::default();
    // Per-edge budgets: an edge whose search runs out keeps its circle (sound) and is
    // recorded, instead of aborting the whole run.
    let budget = state.discriminating_budget;
    let mut paths = Vec::new();
    for i in 0..pag.node_count() {
        let b = DenseNodeId::from_raw(crate::indexing::dense_u32(i));
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
                    if ci.p_value(u, v, z)? > ci.alpha {
                        let minimal = ci.minimize_sepset(u, v, z)?;
                        remove = Some((u, v, minimal));
                        break 'pairs;
                    }
                }
            }
        }

        if let Some((u, v, minimal)) = remove {
            if ci.input.constraints.static_required(u, v) {
                continue;
            }
            let key = edge_key(u, v);
            if adj.remove(&key) {
                record_sepset(sepsets, u, v, &minimal);
                let ud = dense_of(u)?;
                let vd = dense_of(v)?;
                let _ = pag.remove_edge(ud, vd);
                let dense_sep: Vec<DenseNodeId> =
                    minimal.iter().map(|x| dense_of(*x)).collect::<Result<_, _>>()?;
                state.set_sepset(ud, vd, Arc::from(dense_sep));
                delta.edges_changed += 1;
            }
            continue;
        }

        // Standard R4 orientation (Zhang) after Lemma 3.2 checks pass.
        let c_in_sep =
            sep_ab.iter().map(|&z| dense_of(z)).collect::<Result<Vec<_>, _>>()?.contains(&c);
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
        } else if set_marks_oriented(pag, state, &mut delta, c, b, Endpoint::Tail, Endpoint::Arrow)?
        {
            delta.edges_changed += 1;
        }
    }
    delta.fixed_point = delta.edges_changed == 0;
    Ok(delta)
}

/// Remove `edge` as separated by `minimal` and queue the triples its removal unshields.
fn remove_edge_update(
    ci: &StaticCiRun<'_>,
    graph: &mut RfciGraph<'_>,
    edge: (VariableId, VariableId),
    minimal: &[VariableId],
    triples: &mut UnshieldedTriples,
) -> Result<(), DiscoveryError> {
    let (adj, sepsets, pag) = (&mut *graph.adj, &mut *graph.sepsets, &mut *graph.pag);
    let dense_of = graph.dense_of;
    let (x, y) = edge;
    let variables = ci.input.variables;
    if ci.input.constraints.static_required(x, y) {
        return Ok(());
    }
    let key = edge_key(x, y);
    if !adj.remove(&key) {
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
        if b != y && adj.contains(&edge_key(y, b)) && !adj.contains(&edge_key(x, y)) {
            // triple x-b-y is unshielded if x–y gone
            triples.push(x, b, y);
        }
    }
    for &b in &ny {
        if b != x && adj.contains(&edge_key(x, b)) {
            triples.push(x, b, y);
        }
    }
    Ok(())
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
