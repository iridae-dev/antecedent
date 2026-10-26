//! Classic static FCI over [`TabularData`] → [`Pag`].
//!
//! Phases (Spirtes, Meek & Richardson 1995):
//! 1. PC-style adjacency skeleton
//! 2. Unshielded collider orientation
//! 3. Possible-D-Sep adjacency (further edge removals)
//! 4. Reset remaining edges to `o–o`
//! 5. Zhang FCI orientation ([`crate::rule_scheduling::default_fci_rules`])
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

use std::collections::HashMap;
use std::sync::Arc;

use antecedent_core::{ExecutionContext, Lag, VariableId};
use antecedent_data::TabularData;
use antecedent_graph::{DenseNodeId, Endpoint, Pag, PagReview};
use antecedent_stats::{ConditionalIndependence, FdrAdjustment, PartialCorrelation};

use crate::combinations::for_each_combination_vars;
use crate::constraints::{DiscoveryConstraints, KnowledgeScope};
use crate::discriminating_paths::DiscriminatingPathBudget;
use crate::engine::DiscoveryWorkspace;
use crate::error::DiscoveryError;
use crate::orientation::{OrientationError, OrientationState};
use crate::pc::{edge_key, sorted_edge_pairs};
use crate::possible_d_sep::{PossibleDSepBudget, possible_d_sep};
use crate::result::{
    DiscoveryDiagnostic, DiscoveryIteration, DiscoveryPerformanceRecord, DiscoveryResult,
    EvidenceSource, GraphEvidence, PcSepsets, discovery_assumptions,
};
use crate::rule_scheduling::{
    FciOrientationRule, LpcmciOrientCollider, default_fci_rules, run_fci_orientation_to_fixed_point,
};
use crate::static_skeleton::{
    StaticSkeleton, StaticSkeletonInput, UndirectedAdj, record_sepset, run_static_skeleton,
    skeleton_scored_links, static_ci_test, static_edge_evidence,
};

/// Identity of this algorithm in its records.
const ALGORITHM: &str = crate::pipeline::StaticAlgorithm::Fci.as_str();

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
    /// Multiple-testing adjustment (`None` = off). Configured adjustments are refused until
    /// the complete multi-phase CI family can be adjusted coherently.
    pub fdr: Option<FdrAdjustment>,
    /// Possible-D-Sep BFS expansion budget.
    pub pds_max_nodes: usize,
    /// Per-edge bounds of the discriminating-path (R4) search.
    pub discriminating_path_budget: DiscriminatingPathBudget,
}

crate::debug_with_ci!(Fci, pds_max_nodes, discriminating_path_budget);

impl Default for Fci {
    fn default() -> Self {
        Self::new()
    }
}

impl Fci {
    /// Default FCI with `ParCorr` and no incomplete multi-phase FDR adjustment.
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
            pds_max_nodes: DEFAULT_PDS_MAX_NODES,
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
    /// FCI currently refuses a configured adjustment because its multi-phase CI family
    /// includes Possible-D-Sep tests; adjusting only the initial adjacency phase is invalid.
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

    /// Run classic static FCI.
    ///
    /// # Errors
    ///
    /// Data, CI, Possible-D-Sep budget, orientation failures, or an FDR configuration that
    /// cannot cover the complete multi-phase CI family.
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
        self.constraints
            .check_background_knowledge(KnowledgeScope::EquivalenceClass, variables)
            .map_err(DiscoveryError::Knowledge)?;
        if self.fdr.is_some() {
            return Err(DiscoveryError::unsupported(
                "FCI FDR is refused until adjustment covers both adjacency and Possible-D-Sep CI tests",
            ));
        }
        let (col_owned, var_index) = crate::pc::prepare_static_columns(
            crate::pc::StaticColumnRequest {
                ci: &*self.ci,
                significance: self.constraints.significance,
                alpha: self.constraints.alpha,
                fdr: false,
                empty_variables: "FCI requires at least one variable",
                insufficient_rows: "insufficient rows for FCI",
            },
            data,
            variables,
            workspace,
            ctx,
        )?;
        let cols: Vec<&[f64]> = col_owned.iter().map(AsRef::as_ref).collect();
        let covariance = crate::static_skeleton::CovarianceSearch::prepare(
            &*self.ci,
            &cols,
            &self.constraints,
            self.fdr,
            ctx,
        );

        let alpha = self.constraints.alpha;
        let max_cond = self.constraints.max_cond_size;
        let input = StaticSkeletonInput {
            covariance: covariance.as_ref(),
            ci: &*self.ci,
            constraints: &self.constraints,
            cols: &cols,
            var_index: &var_index,
            variables,
            label: "fci.pc.depth",
        };
        let skel = run_static_skeleton(&input, workspace, ctx)?;
        let mut scored = skeleton_scored_links(&skel);
        let StaticSkeleton {
            mut adj,
            mut sepsets,
            mut ci_tests,
            mut iterations,
            parent_limit_removed,
            ..
        } = skel;
        let mut combo_scratch = Vec::new();
        let dense_of = |v: VariableId| crate::pipeline::dense_of(&var_index, v);

        // Build circle–circle PAG skeleton.
        let mut pag = build_pag_circle_skeleton(variables, &var_index, &adj)?;
        apply_background_marks(&mut pag, &self.constraints, variables, &dense_of)?;

        // Orientation state from PC sepsets.
        let mut state = OrientationState::default();
        load_sepsets_into_state(&sepsets, &dense_of, &mut state)?;

        // --- Phase 2: unshielded colliders (needed for Possible-D-Sep) ---
        let collider_rules: [&dyn FciOrientationRule; 1] = [&LpcmciOrientCollider];
        let _ = run_fci_orientation_to_fixed_point(&mut pag, &collider_rules, &mut state)?;

        // --- Phase 3: Possible-D-Sep adjacency ---
        let mut pds_tests = 0u64;
        // Same reproducibility hazard as the Phase 1 skeleton loop (see `sorted_edge_pairs`
        // doc): this loop mutates `pag` (via `pag.remove_edge` below) and later pairs'
        // `possible_d_sep` traversals read that same `pag`, so which edge is processed
        // first affects the result.
        let edges_pds = sorted_edge_pairs(&adj);
        for &(x, y) in &edges_pds {
            if ctx.cancellation.is_cancelled() {
                return Err(DiscoveryError::Cancelled);
            }
            if !adj.contains(edge_key(x, y)) {
                continue;
            }
            if self.constraints.static_required(x, y) {
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
                    let outcome = static_ci_test(&input, x, y, z, workspace, ctx)?;
                    pds_tests += 1;
                    ci_tests += 1;
                    // An untestable query carries no information and never separates.
                    if outcome.is_some_and(|(_stat, p)| p > alpha) {
                        independent = true;
                        best_sep = Arc::from(z.as_slice());
                        break 'pds_depth;
                    }
                }
            }

            if independent {
                let key = edge_key(x, y);
                adj.remove(key);
                record_sepset(&mut sepsets, x, y, &best_sep);
                let _ = pag.remove_edge(xd, yd);
                let dense_sep: Vec<DenseNodeId> =
                    best_sep.iter().map(|v| dense_of(*v)).collect::<Result<_, _>>()?;
                state.set_sepset(xd, yd, Arc::from(dense_sep));
            }
        }
        iterations.push(DiscoveryIteration {
            label: Arc::from("fci.possible_d_sep"),
            ci_tests: pds_tests,
        });

        // --- Phase 4: reset remaining edges to o–o ---
        pag = build_pag_circle_skeleton(variables, &var_index, &adj)?;
        apply_background_marks(&mut pag, &self.constraints, variables, &dense_of)?;
        state = OrientationState {
            discriminating_budget: self.discriminating_path_budget,
            ..OrientationState::default()
        };
        load_sepsets_into_state(&sepsets, &dense_of, &mut state)?;

        // --- Phase 5: full Zhang FCI orientation ---
        let rules = default_fci_rules();
        let orient_delta = run_fci_orientation_to_fixed_point(&mut pag, &rules, &mut state)?;

        let mut diagnostics = Vec::new();
        diagnostics
            .extend(crate::static_skeleton::parent_limit_diagnostic(&parent_limit_removed, ALGORITHM));
        if !state.discriminating_skipped.is_empty() {
            diagnostics.push(DiscoveryDiagnostic {
                code: Arc::from("fci.discriminating_path_budget"),
                message: Arc::from(format!(
                    "discriminating-path search exhausted its budget on {} edge(s); their circle \
                     marks were left unresolved (sound, possibly incomplete)",
                    state.discriminating_skipped.len()
                )),
            });
        }
        diagnostics.extend(crate::rule_scheduling::pd_path_budget_diagnostic(&state, ALGORITHM));
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
        scored.retain(|s| adj.contains(edge_key(s.link.source, s.link.target)));

        let edge_evidence = static_edge_evidence(&scored, &sepsets, ALGORITHM);

        let evidence = GraphEvidence {
            graph: pag.clone(),
            edge_evidence: Arc::from(edge_evidence),
            links: Arc::from(scored),
            source: EvidenceSource::Discovery { algorithm: Arc::from(ALGORITHM) },
        };
        let review = PagReview::from_pag(pag, ALGORITHM);

        Ok(DiscoveryResult {
            evidence,
            review,
            algorithm: crate::pipeline::algorithm_record(
                ALGORITHM,
                format!(
                    "alpha={},max_cond={},fdr={},pds_max={}{}",
                    alpha,
                    max_cond,
                    self.fdr.is_some(),
                    self.pds_max_nodes,
                    self.constraints.knowledge_config()
                ),
            ),
            assumptions: discovery_assumptions(ALGORITHM, false),
            iterations,
            diagnostics,
            performance: DiscoveryPerformanceRecord {
                ci_tests,
                links_retained: crate::pipeline::count_u64(adj.len())?,
                targets: crate::pipeline::count_u64(variables.len())?,
                lagged_frame_bytes: 0,
                worker_threads: 1,
            },
            sepsets,
        })
    }
}

fn pds_budget_err(b: PossibleDSepBudget) -> DiscoveryError {
    DiscoveryError::from(OrientationError::SearchBudgetExhausted {
        rule: "fci.possible_d_sep",
        max_paths: b.max_nodes,
        max_len: 0,
    })
}

pub(crate) fn load_sepsets_into_state(
    sepsets: &PcSepsets,
    dense_of: &dyn Fn(VariableId) -> Result<DenseNodeId, DiscoveryError>,
    state: &mut OrientationState,
) -> Result<(), DiscoveryError> {
    crate::pc::load_sepsets_into_state(sepsets, dense_of, state)
}

/// Mark the edges of a circle skeleton whose direction background knowledge fixes.
///
/// A required link `a → b` is a direct cause among observed variables, so the edge is `a → b`
/// (tail at `a`, arrow at `b`). The reverse of a forbidden link `b → a`, and a tier that places
/// `b` after `a`, exclude only a tail at `b`: `a` and `b` may still share a latent common cause,
/// so the edge gets an arrowhead at `b` and the mark at `a` is left to the data. Pairs the search
/// found non-adjacent are left alone: knowledge never adds an edge.
///
/// # Errors
///
/// A variable outside the graph, or a graph mutation failure.
pub(crate) fn apply_background_marks(
    pag: &mut Pag,
    constraints: &DiscoveryConstraints,
    variables: &[VariableId],
    dense_of: &dyn Fn(VariableId) -> Result<DenseNodeId, DiscoveryError>,
) -> Result<(), DiscoveryError> {
    for (i, &a) in variables.iter().enumerate() {
        for &b in &variables[i + 1..] {
            let Some(fixed) = constraints.static_direction(a, b) else {
                continue;
            };
            let (from, to) = (dense_of(fixed.from)?, dense_of(fixed.to)?);
            if !pag.has_edge(from, to) {
                continue;
            }
            let at_from = if fixed.required { Endpoint::Tail } else { Endpoint::Circle };
            pag.set_marks(from, to, at_from, Endpoint::Arrow).map_err(DiscoveryError::from)?;
        }
    }
    Ok(())
}

pub(crate) fn build_pag_circle_skeleton(
    variables: &[VariableId],
    var_index: &HashMap<VariableId, usize>,
    adj: &UndirectedAdj,
) -> Result<Pag, DiscoveryError> {
    let mut pag = Pag::empty();
    for &v in variables {
        pag.add_node(antecedent_graph::NodeRef::Static(v)).map_err(DiscoveryError::from)?;
    }
    for (lo, hi) in adj.edge_pairs() {
        let a_idx = *var_index
            .get(&VariableId::from_raw(lo))
            .ok_or_else(|| DiscoveryError::data_msg("unknown variable in skeleton"))?;
        let b_idx = *var_index
            .get(&VariableId::from_raw(hi))
            .ok_or_else(|| DiscoveryError::data_msg("unknown variable in skeleton"))?;
        let a = DenseNodeId::from_raw(crate::indexing::dense_u32(a_idx));
        let b = DenseNodeId::from_raw(crate::indexing::dense_u32(b_idx));
        pag.insert_circle_circle(a, b).map_err(DiscoveryError::from)?;
    }
    Ok(pag)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use antecedent_core::{ExecutionContext, VariableId};
    use antecedent_graph::Endpoint;
    use antecedent_stats::OracleCi;

    use super::*;
    use crate::test_tables::tabular_n;

    #[test]
    fn oracle_chain_recovers_skeleton() {
        let data = tabular_n(3, 50);
        let vars = [VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2)];
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
    fn fdr_refuses_an_incomplete_multiphase_family() {
        let data = tabular_n(2, 20);
        let vars = [VariableId::from_raw(0), VariableId::from_raw(1)];
        let mut ws = DiscoveryWorkspace::default();
        let err = Fci::new()
            .with_fdr(true)
            .run(&data, &vars, &mut ws, &ExecutionContext::for_tests(4))
            .unwrap_err();
        assert!(
            matches!(err, DiscoveryError::Unsupported { message } if message.contains("Possible-D-Sep")),
            "{err}"
        );
    }

    #[test]
    fn oracle_collider_orients_into_middle() {
        let data = tabular_n(3, 40);
        let vars = [VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2)];
        let oracle = OracleCi::new([(0usize, 1usize), (1usize, 2usize)]);
        let fci = Fci::new().with_fdr(false).with_ci(Arc::new(oracle));
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(2);
        let result = fci.run(&data, &vars, &mut ws, &ctx).unwrap();
        let g = &result.evidence.graph;
        let e01 = g.edge_between(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let e21 = g.edge_between(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
        // Collider at 1: arrow into 1 on both edges.
        let at_1_from_0 = if e01.a.raw() == 1 { e01.at_a } else { e01.at_b };
        let at_1_from_2 = if e21.a.raw() == 1 { e21.at_a } else { e21.at_b };
        assert!(matches!(at_1_from_0, Endpoint::Arrow));
        assert!(matches!(at_1_from_2, Endpoint::Arrow));
    }

    #[test]
    fn review_tracks_remaining_circles() {
        let data = tabular_n(3, 40);
        let vars = [VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2)];
        // Fully connected dependent — skeleton retains all edges; circles remain.
        let oracle = OracleCi::new([(0usize, 1usize), (1usize, 2usize), (0usize, 2usize)]);
        let fci = Fci::new().with_fdr(false).with_ci(Arc::new(oracle));
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(3);
        let result = fci.run(&data, &vars, &mut ws, &ctx).unwrap();
        assert!(!result.review.pending_circles.is_empty() || result.review.is_complete());
        assert_eq!(result.review.algorithm.as_ref(), "fci");
    }

    /// Background knowledge on static FCI: a required link states a direct cause (tail at the
    /// cause, arrow at the effect); the reverse of a forbidden link and a later tier exclude only
    /// a tail at the later end, so they place an arrowhead there and nothing at the other end.
    mod knowledge {
        use std::sync::Arc;

        use antecedent_core::{ExecutionContext, VariableId};
        use antecedent_graph::{DenseNodeId, Endpoint, Pag};
        use antecedent_stats::OracleCi;

        use super::super::Fci;
        use crate::constraints::{DiscoveryConstraints, contemp_link};
        use crate::engine::DiscoveryWorkspace;
        use crate::error::DiscoveryError;
        use crate::test_tables::tabular_n;

        fn v(i: u32) -> VariableId {
            VariableId::from_raw(i)
        }

        fn run(constraints: DiscoveryConstraints) -> Result<Pag, DiscoveryError> {
            let data = tabular_n(2, 10);
            let fci = Fci::new()
                .with_fdr(false)
                .with_constraints(constraints)
                .with_ci(Arc::new(OracleCi::new([(0usize, 1usize)])));
            let mut ws = DiscoveryWorkspace::default();
            fci.run(&data, &[v(0), v(1)], &mut ws, &ExecutionContext::for_tests(1))
                .map(|r| r.evidence.graph)
        }

        /// Marks `(at 0, at 1)` of the edge between variables 0 and 1.
        fn marks(pag: &Pag) -> (Endpoint, Endpoint) {
            let e = pag.edge_between(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
            if e.a.raw() == 0 { (e.at_a, e.at_b) } else { (e.at_b, e.at_a) }
        }

        fn base() -> DiscoveryConstraints {
            Fci::new().constraints
        }

        #[test]
        fn without_knowledge_the_edge_keeps_circle_marks() {
            assert_eq!(marks(&run(base()).unwrap()), (Endpoint::Circle, Endpoint::Circle));
        }

        #[test]
        fn a_tier_or_a_forbidden_reverse_place_an_arrowhead_only() {
            let mut tiered = base();
            tiered.tiers = Arc::from([Arc::from([v(0)]), Arc::from([v(1)])]);
            assert_eq!(marks(&run(tiered).unwrap()), (Endpoint::Circle, Endpoint::Arrow));

            let mut forbidden = base();
            forbidden.forbidden = Arc::from([contemp_link(v(1), v(0))]);
            assert_eq!(marks(&run(forbidden).unwrap()), (Endpoint::Circle, Endpoint::Arrow));

            // The other way round: the arrowhead is at variable 0.
            let mut reversed = base();
            reversed.tiers = Arc::from([Arc::from([v(1)]), Arc::from([v(0)])]);
            assert_eq!(marks(&run(reversed).unwrap()), (Endpoint::Arrow, Endpoint::Circle));
        }

        #[test]
        fn a_required_link_is_adjacency_plus_tail_and_arrow() {
            let mut required = base();
            required.required = Arc::from([contemp_link(v(0), v(1))]);
            assert_eq!(marks(&run(required).unwrap()), (Endpoint::Tail, Endpoint::Arrow));
        }

        #[test]
        fn a_parent_limit_of_zero_leaves_no_adjacency() {
            let mut capped = base();
            capped.max_parents = Some(0);
            let data = tabular_n(2, 10);
            let result = Fci::new()
                .with_fdr(false)
                .with_constraints(capped)
                .with_ci(Arc::new(OracleCi::new([(0usize, 1usize)])))
                .run(
                    &data,
                    &[v(0), v(1)],
                    &mut DiscoveryWorkspace::default(),
                    &ExecutionContext::for_tests(1),
                )
                .unwrap();
            assert!(
                !result.evidence.graph.has_edge(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1))
            );
            assert!(
                result
                    .diagnostics
                    .iter()
                    .any(|d| &*d.code == "fci.parent_limit_removed_adjacencies")
            );
        }
    }
}
