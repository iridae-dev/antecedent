//! Static PC discovery over [`TabularData`].
//!
//! Classic undirected skeleton search + collider / Meek orientation → [`Cpdag`].
//! Distinct from PCMCI PC1 parent selection.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use antecedent_core::{ExecutionContext, Lag, VariableId};
use antecedent_data::{TableView, TabularData};
use antecedent_graph::{Cpdag, CpdagReview, DenseNodeId};
use antecedent_stats::{ConditionalIndependence, FdrAdjustment, PartialCorrelation};

use crate::constraints::{DiscoveryConstraints, KnowledgeScope};
use crate::engine::DiscoveryWorkspace;
use crate::error::DiscoveryError;
use crate::evidence::retain_after_family_fdr;
use crate::orientation::{
    MeekR1, MeekR2, MeekR3, MeekR4, OrientCollider, OrientationRule, OrientationState, RuleDelta,
    run_static_orientation_to_fixed_point, try_orient_undirected,
};
use crate::result::{
    DiscoveryDiagnostic, DiscoveryPerformanceRecord, DiscoveryResult, EvidenceSource,
    GraphEvidence, ScoredLink, discovery_assumptions,
};
use crate::static_skeleton::{
    StaticSkeleton, StaticSkeletonInput, UndirectedAdj, run_static_skeleton, skeleton_scored_links,
    static_edge_evidence,
};

/// Static PC discovery result (`Cpdag` evidence + review).
pub type StaticCpdagDiscoveryResult = DiscoveryResult<Cpdag, CpdagReview>;

/// PC algorithm over tabular (non-temporal) data.
///
/// Adjacency is removed in place while the search runs, so — as in the original algorithm
/// and unlike PC-stable — the skeleton depends on the order edges are visited. That order is
/// fixed (sorted variable ids), which makes a run reproducible but does not make the result
/// invariant to relabelling the variables.
#[derive(Clone)]
pub struct Pc {
    /// Constraints / alpha / max conditioning size.
    pub constraints: DiscoveryConstraints,
    /// Pluggable CI test.
    pub ci: Arc<dyn ConditionalIndependence + Send + Sync>,
    /// Multiple-testing adjustment (`None` = off). Static PC includes all edges in the family.
    pub fdr: Option<FdrAdjustment>,
}

crate::debug_with_ci!(Pc);

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
    ) -> Result<StaticCpdagDiscoveryResult, DiscoveryError> {
        self.constraints.validate()?;
        self.constraints
            .check_background_knowledge(KnowledgeScope::EquivalenceClass, variables)
            .map_err(DiscoveryError::Knowledge)?;
        let (col_owned, var_index) = prepare_static_columns(
            StaticColumnRequest {
                ci: &*self.ci,
                significance: self.constraints.significance,
                alpha: self.constraints.alpha,
                fdr: self.fdr.is_some(),
                empty_variables: "PC requires at least one variable",
                insufficient_rows: "insufficient rows for PC",
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
        let skel = run_static_skeleton(
            &StaticSkeletonInput {
                covariance: covariance.as_ref(),
                ci: &*self.ci,
                constraints: &self.constraints,
                cols: &cols,
                var_index: &var_index,
                variables,
                label: "pc.depth",
            },
            workspace,
            ctx,
        )?;
        let mut scored = skeleton_scored_links(&skel);
        let StaticSkeleton {
            mut adj,
            sepsets,
            ci_tests,
            iterations,
            family_p,
            family_edge,
            parent_limit_removed,
            ..
        } = skel;

        // Multiplicity correction (when configured) covers the edges that were actually
        // tested; constraint-required edges carry no test and pass through untouched.
        let (tested, untested): (Vec<ScoredLink>, Vec<ScoredLink>) =
            scored.drain(..).partition(|s| s.p_value.is_finite());
        // The family is one hypothesis per tested edge (see `retain_after_family_fdr`).
        let family_size = if family_p.len() == family_edge.len() && !family_edge.is_empty() {
            family_edge.iter().collect::<HashSet<_>>().len()
        } else {
            tested.len()
        };
        let fdr_diagnostic = self.fdr.and_then(|_| {
            crate::ci::fdr_resolution_diagnostic(
                &*self.ci,
                self.constraints.significance,
                alpha,
                family_size,
            )
        });
        scored = retain_after_family_fdr(tested, &family_p, &family_edge, self.fdr, alpha);
        scored.extend(untested);
        let kept: HashSet<(u32, u32)> =
            scored.iter().map(|s| edge_key(s.link.source, s.link.target)).collect();
        // If FDR ran, drop edges that failed; if FDR off, keep skeleton as-is.
        if self.fdr.is_some() {
            adj.retain(|k| kept.contains(k));
        }

        // Build undirected CPDAG skeleton.
        let mut cpdag = Cpdag::with_variables(u32::try_from(variables.len()).unwrap_or(u32::MAX));
        // Map VariableId → dense: assume variables are 0..n contiguous for with_variables,
        // otherwise rebuild with explicit add_node order matching `variables`.
        if variables.iter().enumerate().any(|(i, v)| v.raw() as usize != i) {
            cpdag = Cpdag::empty();
            for &v in variables {
                cpdag
                    .add_node(antecedent_graph::NodeRef::Static(v))
                    .map_err(DiscoveryError::from)?;
            }
        }
        let dense_of = |v: VariableId| crate::pipeline::dense_of(&var_index, v);
        for (lo, hi) in adj.edge_pairs() {
            let a = dense_of(VariableId::from_raw(lo))?;
            let b = dense_of(VariableId::from_raw(hi))?;
            cpdag.insert_undirected(a, b).map_err(DiscoveryError::from)?;
        }

        let mut state = OrientationState::default();
        load_sepsets_into_state(&sepsets, &dense_of, &mut state)?;
        orient_fixed_directions(&mut cpdag, &self.constraints, variables, &dense_of, &mut state)?;

        let rules: [&dyn OrientationRule<antecedent_graph::Cpdag>; 5] =
            [&OrientCollider, &MeekR1, &MeekR2, &MeekR3, &MeekR4];
        let orient_delta = run_static_orientation_to_fixed_point(&mut cpdag, &rules, &mut state)?;

        let mut diagnostics: Vec<_> = fdr_diagnostic.into_iter().collect();
        diagnostics
            .extend(crate::static_skeleton::parent_limit_diagnostic(&parent_limit_removed, "pc"));
        if state.conflicts > 0 || orient_delta.conflicts > 0 {
            diagnostics.push(DiscoveryDiagnostic {
                code: Arc::from("pc.orientation_conflict"),
                message: Arc::from(format!(
                    "{} orientation conflict(s)",
                    state.conflicts.max(orient_delta.conflicts)
                )),
            });
        }

        let edge_evidence = static_edge_evidence(&scored, &sepsets, "pc");

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
            algorithm: crate::pipeline::algorithm_record(
                "pc",
                format!(
                    "alpha={},max_cond={},fdr={}{}",
                    alpha,
                    max_cond,
                    self.fdr.is_some(),
                    self.constraints.knowledge_config()
                ),
            ),
            assumptions: discovery_assumptions("pc", true),
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

/// Orient the undirected edges of `cpdag` whose direction background knowledge fixes.
///
/// Knowledge is applied before colliders (Meek 1995), so a collider the data imply against a
/// fixed direction is recorded as an orientation conflict on that edge rather than silently
/// overriding the knowledge; knowledge that would close a directed cycle is recorded the same
/// way. Pairs the search found non-adjacent are left alone: knowledge never adds an edge.
///
/// # Errors
///
/// A variable outside the graph, or a graph mutation failure other than a cycle.
pub(crate) fn orient_fixed_directions(
    cpdag: &mut Cpdag,
    constraints: &DiscoveryConstraints,
    variables: &[VariableId],
    dense_of: &dyn Fn(VariableId) -> Result<DenseNodeId, DiscoveryError>,
    state: &mut OrientationState,
) -> Result<(), DiscoveryError> {
    let mut delta = RuleDelta::default();
    for (i, &a) in variables.iter().enumerate() {
        for &b in &variables[i + 1..] {
            let Some(fixed) = constraints.static_direction(a, b) else {
                continue;
            };
            let (from, to) = (dense_of(fixed.from)?, dense_of(fixed.to)?);
            let Some(edge) = cpdag.edge_between(from, to) else {
                continue;
            };
            if !edge.is_undirected() {
                // A score search can hand over an edge it has already directed. As stated there
                // is nothing to orient; against the statement the data compelled it, which is
                // recorded as a conflict like a collider that disagrees.
                if edge.parent_child().is_some_and(|(parent, _)| parent == to) {
                    state.record_conflict(&mut delta, from, to, "knowledge");
                    let _ = cpdag.mark_conflict(from, to);
                }
                continue;
            }
            let premise = format!("background knowledge: {}→{}", fixed.from.raw(), fixed.to.raw());
            try_orient_undirected(cpdag, state, &mut delta, from, to, premise)?;
        }
    }
    Ok(())
}

pub(crate) fn edge_key(a: VariableId, b: VariableId) -> (u32, u32) {
    if a.raw() <= b.raw() { (a.raw(), b.raw()) } else { (b.raw(), a.raw()) }
}

/// Deterministic traversal order over `adj`'s pairs.
///
/// `HashMap` iteration order is randomly seeded per-process (`RandomState`) and is not
/// stable across runs. The skeleton / Possible-D-Sep loops that consume this list mutate
/// `adj` (or the graph it backs) as they iterate, so which pair is visited first changes
/// `adjacent_vars` / reachability for pairs visited later in the *same* pass — an
/// unsorted traversal makes the resulting skeleton nondeterministic across runs on
/// bit-identical input. Sorting fixes a canonical order; it does not change the
/// algorithm's existing (documented) order-dependence, only makes it reproducible.
pub(crate) fn sorted_edge_pairs(adj: &UndirectedAdj) -> Vec<(VariableId, VariableId)> {
    let mut edges: Vec<(VariableId, VariableId)> = adj
        .edge_pairs()
        .into_iter()
        .map(|(lo, hi)| (VariableId::from_raw(lo), VariableId::from_raw(hi)))
        .collect();
    edges.sort_unstable();
    edges
}

pub(crate) fn adjacent_vars(
    v: VariableId,
    adj: &UndirectedAdj,
    variables: &[VariableId],
) -> Vec<VariableId> {
    variables.iter().copied().filter(|&u| u != v && adj.contains(edge_key(v, u))).collect()
}

/// Column vectors with the corresponding original-variable index.
pub(crate) type StaticColumns = (Vec<Arc<[f64]>>, HashMap<VariableId, usize>);

/// What [`prepare_static_columns`] needs from the algorithm besides the table.
#[derive(Clone, Copy)]
pub(crate) struct StaticColumnRequest<'a> {
    /// CI test to prepare.
    pub ci: &'a (dyn antecedent_stats::ConditionalIndependence + Sync),
    /// Significance method recorded on the prepared test.
    pub significance: antecedent_stats::SignificanceMethod,
    /// Level passed to the resolvability check.
    pub alpha: f64,
    /// Whether an FDR adjustment is configured.
    pub fdr: bool,
    /// Refusal when `variables` is empty.
    pub empty_variables: &'static str,
    /// Refusal when the table has fewer than three rows.
    pub insufficient_rows: &'static str,
}

/// Column vectors, variable index, and a prepared CI test for a static algorithm.
///
/// The refusal strings stay `&'static str` so each algorithm keeps the [`DiscoveryError`]
/// variant it already returned.
///
/// # Errors
///
/// Empty variable list, short or ragged columns, or a CI prepare failure.
pub(crate) fn prepare_static_columns(
    request: StaticColumnRequest<'_>,
    data: &TabularData,
    variables: &[VariableId],
    workspace: &mut DiscoveryWorkspace,
    ctx: &ExecutionContext,
) -> Result<StaticColumns, DiscoveryError> {
    let StaticColumnRequest { ci, significance, alpha, fdr, empty_variables, insufficient_rows } =
        request;
    crate::ci::ensure_ci_decisions_meaningful(ci, significance, alpha, fdr)?;
    if variables.is_empty() {
        return Err(DiscoveryError::unsupported(empty_variables));
    }
    let owned = collect_float_columns(data, variables)?;
    let n = owned.first().map_or(0, |c| c.len());
    crate::ci::ensure_ci_fits_rows(ci, n)?;
    if n < 3 {
        return Err(DiscoveryError::stats_msg(insufficient_rows));
    }
    if owned.iter().any(|c| c.len() != n) {
        return Err(DiscoveryError::data_msg("column length mismatch"));
    }
    let index = variables.iter().enumerate().map(|(i, v)| (*v, i)).collect();
    let cols: Vec<&[f64]> = owned.iter().map(AsRef::as_ref).collect();
    workspace.prepared_ci = Some(crate::ci::prepare_ci(ci, &cols, significance, ctx)?);
    Ok((owned, index))
}

/// Store one direction of each separating set as dense node ids.
/// A variable that is not in the graph is an error, not a silently shorter sepset.
///
/// # Errors
///
/// A separator variable that is not in the graph. Dropping it would orient from a shorter set.
pub(crate) fn load_sepsets_into_state(
    sepsets: &crate::result::PcSepsets,
    dense_of: &dyn Fn(VariableId) -> Result<DenseNodeId, DiscoveryError>,
    state: &mut OrientationState,
) -> Result<(), DiscoveryError> {
    for ((sx, _, ty, _), sep) in sepsets {
        if sx.raw() > ty.raw() {
            continue;
        }
        let a = dense_of(*sx)?;
        let b = dense_of(*ty)?;
        state.set_sepset(a, b, Arc::from(map_sepset(sep, dense_of)?));
    }
    Ok(())
}

pub(crate) fn map_sepset(
    sep: &[(VariableId, Lag)],
    dense_of: &dyn Fn(VariableId) -> Result<DenseNodeId, DiscoveryError>,
) -> Result<Vec<DenseNodeId>, DiscoveryError> {
    sep.iter().map(|(v, _)| dense_of(*v)).collect()
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

    use antecedent_core::{
        Assumption, CausalSchemaBuilder, ExecutionContext, Lag, MeasurementSpec, RoleHint,
        SmallRoleSet, ValueType, VariableId,
    };
    use antecedent_data::{OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap};
    use antecedent_stats::{
        CiBatchRequest, CiBatchResult, CiResult, CiWorkspace, OracleCi, PreparedCiTest, StatsError,
    };

    use std::collections::HashMap;

    use super::*;
    use crate::result::LaggedLink;
    use crate::test_tables::tabular_n;

    #[test]
    fn covariance_search_preserves_graphs_and_corrected_decisions() {
        use antecedent_stats::MultipleTestingMethod;
        let methods = [
            None,
            Some(MultipleTestingMethod::BenjaminiHochberg),
            Some(MultipleTestingMethod::BenjaminiYekutieli),
            Some(MultipleTestingMethod::Bonferroni),
            Some(MultipleTestingMethod::Holm),
        ];
        for seed in 0..12 {
            let mut rng = antecedent_core::CausalRng::from_seed(seed);
            let mut columns = vec![vec![0.0; 512]; 8];
            for row in 0..512 {
                for c in 0..8 {
                    columns[c][row] = antecedent_kernels::standard_normal(&mut rng)
                        + if c > 0 { 0.65 * columns[c - 1][row] } else { 0.0 };
                }
            }
            if seed % 4 == 1 {
                // Mix eligible queries with nearly collinear residual systems
                // that must use the original solver, including threaded batches.
                for row in 0..512 {
                    columns[7][row] = columns[0][row] + 1e-5 * columns[7][row];
                }
            }
            let data = crate::test_tables::continuous_table(
                &["a", "b", "c", "d", "e", "f", "g", "h"],
                columns,
            );
            let vars: Vec<_> = (0..8).map(VariableId::from_raw).collect();
            for method in methods {
                let pc = Pc::new()
                    .with_constraints(DiscoveryConstraints {
                        max_cond_size: 3,
                        max_parents: if seed % 3 == 0 { Some(2) } else { None },
                        ..DiscoveryConstraints::default()
                    })
                    .with_fdr_adjustment(
                        method
                            .map(|method| FdrAdjustment { method, exclude_contemporaneous: false }),
                    );
                let expected = pc
                    .run(
                        &data,
                        &vars,
                        &mut DiscoveryWorkspace::default(),
                        &ExecutionContext::for_tests(seed),
                    )
                    .unwrap();
                for threads in [1, 4] {
                    let actual = pc
                        .run(
                            &data,
                            &vars,
                            &mut DiscoveryWorkspace::default(),
                            &ExecutionContext::production(seed, threads),
                        )
                        .unwrap();
                    assert_eq!(actual.performance.ci_tests, expected.performance.ci_tests);
                    assert_eq!(
                        format!("{:?}", actual.evidence.graph),
                        format!("{:?}", expected.evidence.graph)
                    );
                    assert_eq!(actual.evidence.links.len(), expected.evidence.links.len());
                    for (a, b) in actual.evidence.links.iter().zip(expected.evidence.links.iter()) {
                        assert_eq!(a.link, b.link);
                        assert!((a.statistic - b.statistic).abs() < 1e-10);
                        assert!((a.p_value - b.p_value).abs() < 1e-9);
                        if let (Some(a), Some(b)) = (a.adjusted_p_value, b.adjusted_p_value) {
                            assert!((a - b).abs() < 1e-9);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn fixed_directions_leave_an_edge_the_search_already_directed_alone_or_mark_it_a_conflict() {
        // 0 → 1 and 2 → 3 are directed; 4 - 5 is undirected. Knowledge fixes 0 → 1 (as it is),
        // 3 → 2 (against the search) and 4 → 5 (open).
        let variables: Vec<VariableId> = (0..6).map(VariableId::from_raw).collect();
        let node = |raw: u32| DenseNodeId::from_raw(raw);
        let mut cpdag = Cpdag::with_variables(6);
        cpdag.insert_directed(node(0), node(1)).unwrap();
        cpdag.insert_directed(node(2), node(3)).unwrap();
        cpdag.insert_undirected(node(4), node(5)).unwrap();
        let constraints = DiscoveryConstraints {
            required: Arc::from([
                crate::constraints::contemp_link(variables[0], variables[1]),
                crate::constraints::contemp_link(variables[3], variables[2]),
                crate::constraints::contemp_link(variables[4], variables[5]),
            ]),
            ..DiscoveryConstraints::default()
        };
        let index: HashMap<VariableId, usize> =
            variables.iter().enumerate().map(|(i, v)| (*v, i)).collect();
        let dense = |v| crate::pipeline::dense_of(&index, v);
        let mut state = OrientationState::default();
        orient_fixed_directions(&mut cpdag, &constraints, &variables, &dense, &mut state).unwrap();
        let direction = |a: u32, b: u32| {
            cpdag
                .edge_between(node(a), node(b))
                .and_then(antecedent_graph::MarkedEdge::parent_child)
                .map(|(p, c)| (p.raw(), c.raw()))
        };
        assert_eq!(direction(0, 1), Some((0, 1)), "an edge already as stated is untouched");
        assert_eq!(direction(4, 5), Some((4, 5)), "an open edge is oriented as stated");
        assert_eq!(direction(2, 3), None, "an edge compelled against the statement is a conflict");
        assert_eq!(state.conflicts, 1);
    }

    #[test]
    fn map_sepset_refuses_an_unknown_variable() {
        let mut index = HashMap::new();
        index.insert(VariableId::from_raw(0), 0usize);
        let dense = |v| crate::pipeline::dense_of(&index, v);
        let err =
            map_sepset(&[(VariableId::from_raw(1), Lag::CONTEMPORANEOUS)], &dense).unwrap_err();
        assert!(err.to_string().contains("unknown variable"), "{err}");
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

    /// Static algorithms have no lag rows to drop: a weight vector must hold exactly one entry
    /// per row. A longer one used to be trimmed to its last `n` entries without a word.
    #[test]
    fn static_algorithms_refuse_weights_that_do_not_fit_the_rows() {
        let data = tabular_n(3, 40);
        let vars = [VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2)];
        let ctx = ExecutionContext::for_tests(1);
        let mut ws = DiscoveryWorkspace::default();
        for len in [39usize, 41, 80] {
            let ci: Arc<dyn ConditionalIndependence + Send + Sync> = Arc::new(
                antecedent_stats::WeightedPartialCorrelation::aligned_to_series_end(vec![1.0; len]),
            );
            let is_weight_refusal = |e: &DiscoveryError| e.to_string().contains("weights");
            let e =
                Pc::new().with_ci(Arc::clone(&ci)).run(&data, &vars, &mut ws, &ctx).unwrap_err();
            assert!(is_weight_refusal(&e), "pc {len}: {e}");
            let e = crate::Fci::new()
                .with_ci(Arc::clone(&ci))
                .run(&data, &vars, &mut ws, &ctx)
                .unwrap_err();
            assert!(is_weight_refusal(&e), "fci {len}: {e}");
            let e = crate::Rfci::new()
                .with_ci(Arc::clone(&ci))
                .run(&data, &vars, &mut ws, &ctx)
                .unwrap_err();
            assert!(is_weight_refusal(&e), "rfci {len}: {e}");
            let e = crate::Ges::new()
                .with_ci(Arc::clone(&ci))
                .with_pc_screening(true)
                .run(&data, &vars, &mut ws, &ctx)
                .unwrap_err();
            assert!(is_weight_refusal(&e), "ges {len}: {e}");
        }
    }

    #[test]
    fn pc_refuses_posterior_probability_tests_under_fdr() {
        let data = tabular_n(3, 40);
        let vars = [VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2)];
        let ctx = ExecutionContext::for_tests(1);
        let mut ws = DiscoveryWorkspace::default();
        for ci in [
            Arc::new(antecedent_stats::BayesFactorCi::new())
                as Arc<dyn ConditionalIndependence + Send + Sync>,
            Arc::new(antecedent_stats::PosteriorDependenceCi::new()),
        ] {
            let refused = Pc::new().with_fdr(true).with_ci(ci);
            let error = refused.run(&data, &vars, &mut ws, &ctx).unwrap_err().to_string();
            assert!(error.contains("FDR adjustment requires frequentist p-values"), "{error}");
        }
    }

    #[test]
    fn pc_success_records_nonempty_assumption_set() {
        let data = tabular_n(3, 40);
        let vars = [VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2)];
        let oracle = OracleCi::new([(0usize, 1usize), (1usize, 2usize)]);
        let pc = Pc::new().with_fdr(false).with_ci(Arc::new(oracle));
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let result = pc.run(&data, &vars, &mut ws, &ctx).unwrap();
        assert!(
            !result.assumptions.is_empty(),
            "successful PC must record algorithm assumptions (discovery-7)"
        );
        let kinds: Vec<_> = result.assumptions.entries.iter().map(|e| &e.assumption).collect();
        assert!(kinds.iter().any(|a| matches!(a, Assumption::Faithfulness)));
        assert!(kinds.iter().any(|a| matches!(a, Assumption::CausalMarkov)));
        assert!(kinds.iter().any(|a| matches!(a, Assumption::CausalSufficiency)));
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
                    antecedent_data::Int64Column::new(
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

    #[test]
    fn three_parent_collider_orients_all_legs() {
        // True: 0→1←2, 3→1. Dependent: (0,1), (1,2), (1,3). Parents 0, 2, 3 are mutually
        // non-adjacent, so node 1 has three legs converging on it in
        // `apply_orient_collider`'s `legs` pair loop -- regression for the stale-`legs`
        // snapshot bug where the second and third pairs sharing an already-oriented leg
        // re-called `orient_undirected` on it and hit
        // "orient_undirected requires an undirected Tail–Tail edge", aborting `Pc::run()`.
        let data = tabular_n(4, 40);
        let vars = [
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            VariableId::from_raw(2),
            VariableId::from_raw(3),
        ];
        let oracle = OracleCi::new([(0usize, 1usize), (1usize, 2usize), (1usize, 3usize)]);
        let pc = Pc::new().with_fdr(false).with_ci(Arc::new(oracle));
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(3);
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
        assert_eq!(
            g.edge_between(DenseNodeId::from_raw(3), DenseNodeId::from_raw(1))
                .unwrap()
                .parent_child(),
            Some((DenseNodeId::from_raw(3), DenseNodeId::from_raw(1)))
        );
    }

    fn independent_gaussians(ncols: usize, nrows: usize, seed: u64) -> TabularData {
        let mut rng = antecedent_core::CausalRng::from_seed(seed);
        let mut next_gauss = || antecedent_kernels::standard_normal(&mut rng);
        let columns: Vec<Vec<f64>> =
            (0..ncols).map(|_| (0..nrows).map(|_| next_gauss()).collect()).collect();
        let names: Vec<String> = (0..ncols).map(|i| format!("v{i}")).collect();
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        crate::test_tables::continuous_table(&refs, columns)
    }

    /// Under independent Gaussian noise, PC skeleton edge retention should track α.
    ///
    /// Measured locally via `scripts/gate_calibration.sh`. Loose band:
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
        let constraints = DiscoveryConstraints {
            alpha: ALPHA,
            max_cond_size: 2,
            temporal: crate::TemporalConstraints {
                max_lag: Lag::CONTEMPORANEOUS,
                min_lag: Lag::CONTEMPORANEOUS,
            },
            ..DiscoveryConstraints::default()
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

    /// Regression: `sorted_edge_pairs` must return a canonical ascending order, not
    /// raw `HashMap` iteration order.
    ///
    /// Builds the same 21-pair edge set (7-node complete graph) via two different
    /// insertion histories (ascending vs. descending). Pre-fix (`adj.iter().collect()`
    /// with no sort), `HashMap` iteration order depends on the randomly-seeded hasher
    /// and, for `hashbrown`, on insertion/removal history too — there is a vanishingly
    /// small chance (~1-in-21!) either raw order already happens to be canonically
    /// sorted, so comparing against the true sorted reference deterministically fails
    /// on the unsorted code and passes once `sort_unstable` is in place. This is the
    /// non-flaky counterpart to a same-process double-run comparison (see
    /// `full_run_reproducible_across_repeated_calls` below for why that alone isn't
    /// reliable pre-fix).
    #[test]
    fn sorted_edge_pairs_is_deterministic_regardless_of_insertion_order() {
        let n = 7u32;
        let mut expected: Vec<(VariableId, VariableId)> = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                expected.push((VariableId::from_raw(i), VariableId::from_raw(j)));
            }
        }
        debug_assert!(expected.is_sorted());

        let vars: Vec<VariableId> = (0..n).map(VariableId::from_raw).collect();
        let mut ascending = UndirectedAdj::from_variables(&vars);
        for &(a, b) in &expected {
            ascending.insert(edge_key(a, b));
        }
        let mut descending = UndirectedAdj::from_variables(&vars);
        for &(a, b) in expected.iter().rev() {
            descending.insert(edge_key(a, b));
        }

        assert_eq!(sorted_edge_pairs(&ascending), expected);
        assert_eq!(sorted_edge_pairs(&descending), expected);
    }

    /// End-to-end confirmation: two full `Pc::run()` calls on bit-identical input
    /// return a bit-identical skeleton.
    ///
    /// Caveat: this alone is not a reliable pre-fix regression test. Rust's default
    /// `HashMap` hasher keys are seeded per OS thread (`RandomState`), so two
    /// `Pc::run()` calls inside the *same* `#[test]` function may or may not observe
    /// different iteration order for the same insertion sequence depending on
    /// unspecified std/hashbrown implementation details -- relying on that would be
    /// flaky across Rust versions/platforms, which is exactly why
    /// `sorted_edge_pairs_is_deterministic_regardless_of_insertion_order` above exists
    /// as the genuine, always-failing-pre-fix check. This test instead confirms the
    /// fix's actual user-visible contract (repeatable results) end to end on a
    /// nontrivial graph (`max_cond_size = 2`, several degree->=2 nodes).
    #[test]
    fn full_run_reproducible_across_repeated_calls() {
        let data = tabular_n(9, 60);
        let vars: Vec<VariableId> = (0..9u32).map(VariableId::from_raw).collect();
        // Three node-disjoint (and thus edge-disjoint) 3-node chains: 0-1-2, 3-4-5,
        // 6-7-8. Nodes 1, 4, 7 have degree 2 (several degree>=2 nodes) while keeping
        // every unshielded triple isolated to its own component. A shared hub/chain
        // edge touched by two different unshielded triples used to hit a separate
        // collider-orientation bug (`orient_undirected requires an undirected
        // Tail–Tail edge`, from a stale `legs` snapshot in `apply_orient_collider`)
        // that is now fixed -- see `three_parent_collider_orients_all_legs` above --
        // but this test keeps the disjoint layout since it targets the
        // determinism fixes, not collider orientation.
        let oracle = OracleCi::new([(0usize, 1usize), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8)]);
        let constraints = DiscoveryConstraints {
            max_cond_size: 2,
            temporal: crate::TemporalConstraints {
                max_lag: Lag::CONTEMPORANEOUS,
                min_lag: Lag::CONTEMPORANEOUS,
            },
            ..DiscoveryConstraints::default()
        };
        let pc = Pc::new().with_fdr(false).with_ci(Arc::new(oracle)).with_constraints(constraints);

        let run_once = || {
            let mut ws = DiscoveryWorkspace::default();
            let ctx = ExecutionContext::for_tests(1);
            let result = pc.run(&data, &vars, &mut ws, &ctx).unwrap();
            let mut edges: Vec<(u32, u32)> = Vec::new();
            for i in 0..9u32 {
                for j in (i + 1)..9u32 {
                    if result
                        .evidence
                        .graph
                        .has_edge(DenseNodeId::from_raw(i), DenseNodeId::from_raw(j))
                    {
                        edges.push((i, j));
                    }
                }
            }
            edges
        };

        let first = run_once();
        let second = run_once();
        assert_eq!(first, second, "PC skeleton must be identical across repeated runs");
    }

    /// Test double whose p-value depends on which conditioning set was queried, so we
    /// can verify a retained edge's reported p-value is the MAXIMUM over every tested
    /// conditioning set at its winning depth, not whichever was tested last.
    struct ZDependentCi;

    impl ZDependentCi {
        fn p_for(query_x: usize, query_y: usize, cond: &[usize]) -> (f64, f64) {
            let (lo, hi) = if query_x <= query_y { (query_x, query_y) } else { (query_y, query_x) };
            if (lo, hi) == (0, 1) {
                let pval = match cond {
                    [2] => 0.40,
                    [3] => 0.10,
                    _ => 0.45, // cond == [] (depth 0).
                };
                (1.0 - pval, pval)
            } else {
                // Strongly dependent for every conditioning set: keeps the rest of the
                // K4 skeleton fully connected so (0, 1) always sees {2, 3} as candidate
                // conditioning variables at depth 1.
                (0.999, 0.001)
            }
        }
    }

    impl ConditionalIndependence for ZDependentCi {
        fn test_batch(
            &self,
            prepared: &PreparedCiTest,
            request: &CiBatchRequest<'_>,
            _workspace: &mut CiWorkspace,
            _ctx: &ExecutionContext,
        ) -> Result<CiBatchResult, StatsError> {
            prepared.ensure_compatible(request)?;
            let request = &prepared.bind_request(request);
            let results = request
                .queries
                .iter()
                .map(|query| {
                    let cond = &request.z_flat[query.z_start..query.z_start + query.z_len];
                    let (statistic, p_value) = Self::p_for(query.x, query.y, cond);
                    CiResult { statistic, p_value, df: 0.0, ci: None }
                })
                .collect();
            Ok(CiBatchResult { results })
        }
    }

    /// A retained edge reports the max p-value over every conditioning set it was tested
    /// with, across all depths: neither the last one tested nor the minimum.
    ///
    /// Pair (0, 1) is tested at depth 0 (`z = []`, p = 0.45) and depth 1 (`z = [2]`
    /// then `z = [3]`, p = 0.40 then 0.10). "Last tested" would report 0.10, a
    /// cross-depth minimum 0.10; the weakest evidence of dependence is 0.45.
    #[test]
    fn retained_edge_reports_max_p_not_last_tested() {
        let data = tabular_n(4, 30);
        let vars: Vec<VariableId> = (0..4u32).map(VariableId::from_raw).collect();

        let constraints = DiscoveryConstraints {
            alpha: 0.5,
            max_cond_size: 1,
            temporal: crate::TemporalConstraints {
                max_lag: Lag::CONTEMPORANEOUS,
                min_lag: Lag::CONTEMPORANEOUS,
            },
            ..DiscoveryConstraints::default()
        };
        let pc =
            Pc::new().with_fdr(false).with_ci(Arc::new(ZDependentCi)).with_constraints(constraints);
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let result = pc.run(&data, &vars, &mut ws, &ctx).unwrap();

        assert!(result.evidence.graph.has_edge(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)));
        let evidence = result
            .evidence
            .edge_evidence
            .iter()
            .find(|e| {
                let (a, b) = (e.link.source.raw(), e.link.target.raw());
                (a, b) == (0, 1) || (a, b) == (1, 0)
            })
            .expect("edge (0,1) evidence present");
        let p = evidence.p_value.expect("p-value recorded");
        assert!(
            (p - 0.45).abs() < 1e-12,
            "expected max-over-all-tests p=0.45 for retained edge (0,1), got {p}"
        );
    }

    /// A constraint-required edge is never tested, so it must not carry significance: no
    /// statistic or p-value, provenance `required`. Tested edges keep theirs.
    #[test]
    fn required_edge_reports_no_test_evidence() {
        let data = tabular_n(3, 30);
        let vars: Vec<VariableId> = (0..3u32).map(VariableId::from_raw).collect();
        let constraints = DiscoveryConstraints {
            required: Arc::from([LaggedLink {
                source: VariableId::from_raw(0),
                source_lag: Lag::CONTEMPORANEOUS,
                target: VariableId::from_raw(1),
                target_lag: Lag::CONTEMPORANEOUS,
            }]),
            temporal: crate::TemporalConstraints {
                max_lag: Lag::CONTEMPORANEOUS,
                min_lag: Lag::CONTEMPORANEOUS,
            },
            ..DiscoveryConstraints::default()
        };
        let oracle = OracleCi::new([(1usize, 2usize)]);
        let pc = Pc::new().with_fdr(false).with_ci(Arc::new(oracle)).with_constraints(constraints);
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let result = pc.run(&data, &vars, &mut ws, &ctx).unwrap();
        let find = |a: u32, b: u32| {
            result
                .evidence
                .edge_evidence
                .iter()
                .find(|e| (e.link.source.raw(), e.link.target.raw()) == (a, b))
                .unwrap_or_else(|| panic!("edge {a}-{b} present"))
        };
        let req = find(0, 1);
        assert_eq!(req.p_value, None);
        assert_eq!(req.statistic, None);
        assert!(req.provenance.iter().any(|p| &**p == "required"));
        let tested = find(1, 2);
        assert!(tested.p_value.is_some());
        assert!(tested.statistic.is_some());
    }

    /// Cancellation aborts with `Cancelled`; a half-searched skeleton is never returned.
    #[test]
    fn cancelled_pc_run_errors() {
        let data = tabular_n(3, 30);
        let vars: Vec<VariableId> = (0..3u32).map(VariableId::from_raw).collect();
        let pc = Pc::new().with_fdr(false).with_ci(Arc::new(OracleCi::new([(0usize, 1usize)])));
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        ctx.cancellation.cancel();
        let out = pc.run(&data, &vars, &mut ws, &ctx);
        assert!(matches!(out, Err(DiscoveryError::Cancelled)), "{out:?}");
    }

    /// Background knowledge on static PC. The dependent pairs of `OracleCi` are the adjacent
    /// pairs, and a non-adjacent pair is independent given the empty set (so every unshielded
    /// triple is a collider); the single-edge and triangle graphs below have none.
    mod knowledge {
        use std::sync::Arc;

        use antecedent_core::{ExecutionContext, VariableId};
        use antecedent_graph::{Cpdag, DenseNodeId};
        use antecedent_stats::OracleCi;

        use super::super::Pc;
        use crate::constraints::{DiscoveryConstraints, KnowledgeFault, contemp_link};
        use crate::engine::DiscoveryWorkspace;
        use crate::error::DiscoveryError;
        use crate::test_tables::tabular_n;

        fn v(i: u32) -> VariableId {
            VariableId::from_raw(i)
        }

        fn run(
            n: usize,
            oracle: &[(usize, usize)],
            constraints: DiscoveryConstraints,
        ) -> Result<crate::pc::StaticCpdagDiscoveryResult, DiscoveryError> {
            let data = tabular_n(n, 10);
            let vars: Vec<VariableId> = (0..n as u32).map(v).collect();
            let pc = Pc::new()
                .with_fdr(false)
                .with_constraints(constraints)
                .with_ci(Arc::new(OracleCi::new(oracle.to_vec())));
            let mut ws = DiscoveryWorkspace::default();
            pc.run(&data, &vars, &mut ws, &ExecutionContext::for_tests(1))
        }

        fn directed(graph: &Cpdag, a: u32, b: u32) -> Option<(u32, u32)> {
            graph
                .edge_between(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b))
                .and_then(antecedent_graph::MarkedEdge::parent_child)
                .map(|(p, c)| (p.raw(), c.raw()))
        }

        fn base() -> DiscoveryConstraints {
            Pc::new().constraints
        }

        #[test]
        fn without_knowledge_a_single_edge_stays_undirected() {
            let result = run(2, &[(0, 1)], base()).unwrap();
            let edge = result
                .evidence
                .graph
                .edge_between(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1))
                .unwrap();
            assert!(edge.is_undirected());
        }

        #[test]
        fn a_tier_a_forbidden_reverse_or_a_required_link_orient_the_edge() {
            let mut tiered = base();
            tiered.tiers = Arc::from([Arc::from([v(0)]), Arc::from([v(1)])]);
            let g = run(2, &[(0, 1)], tiered).unwrap().evidence.graph;
            assert_eq!(directed(&g, 0, 1), Some((0, 1)));

            // Forbidding 0 -> 1 leaves only 1 -> 0.
            let mut forbidden = base();
            forbidden.forbidden = Arc::from([contemp_link(v(0), v(1))]);
            let g = run(2, &[(0, 1)], forbidden).unwrap().evidence.graph;
            assert_eq!(directed(&g, 0, 1), Some((1, 0)));

            let mut required = base();
            required.required = Arc::from([contemp_link(v(1), v(0))]);
            let g = run(2, &[(0, 1)], required).unwrap().evidence.graph;
            assert_eq!(directed(&g, 0, 1), Some((1, 0)));
        }

        #[test]
        fn forbidding_both_directions_removes_the_adjacency() {
            let mut c = base();
            c.forbidden = Arc::from([contemp_link(v(0), v(1)), contemp_link(v(1), v(0))]);
            let g = run(2, &[(0, 1)], c).unwrap().evidence.graph;
            assert!(!g.has_edge(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)));
        }

        #[test]
        fn fixed_directions_propagate_by_meek_rule_two() {
            // Triangle, no unshielded triple. 0 -> 1 -> 2 are required, so 0 - 2 must be 0 -> 2
            // (the other orientation closes a directed cycle).
            let mut c = base();
            c.required = Arc::from([contemp_link(v(0), v(1)), contemp_link(v(1), v(2))]);
            let g = run(3, &[(0, 1), (1, 2), (0, 2)], c).unwrap().evidence.graph;
            assert_eq!(directed(&g, 0, 1), Some((0, 1)));
            assert_eq!(directed(&g, 1, 2), Some((1, 2)));
            assert_eq!(directed(&g, 0, 2), Some((0, 2)));
        }

        #[test]
        fn a_collider_against_a_fixed_direction_is_a_reported_conflict() {
            // 0 - 1 - 2 is an unshielded collider (0 and 2 are independent given nothing), but
            // 1 -> 0 is required: the data-implied arrow into 1 disagrees, and the edge is
            // marked as a conflict rather than silently taking either side.
            let mut c = base();
            c.required = Arc::from([contemp_link(v(1), v(0))]);
            let result = run(3, &[(0, 1), (1, 2)], c).unwrap();
            assert!(
                result.diagnostics.iter().any(|d| &*d.code == "pc.orientation_conflict"),
                "{:?}",
                result.diagnostics
            );
            let edge = result
                .evidence
                .graph
                .edge_between(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1))
                .unwrap();
            assert!(edge.is_conflict());
        }

        /// A star `0 - 1, 0 - 2, 0 - 3` (oracle: dependent pairs are the adjacencies).
        const STAR: [(usize, usize); 3] = [(0, 1), (0, 2), (0, 3)];

        fn degree(g: &Cpdag, node: u32) -> usize {
            (0..4)
                .filter(|&m| {
                    m != node && g.has_edge(DenseNodeId::from_raw(node), DenseNodeId::from_raw(m))
                })
                .count()
        }

        #[test]
        fn a_parent_limit_drops_the_weakest_adjacencies_and_reports_them() {
            let mut capped = base();
            capped.max_parents = Some(2);
            let result = run(4, &STAR, capped).unwrap();
            assert_eq!(degree(&result.evidence.graph, 0), 2);
            assert!(
                result
                    .diagnostics
                    .iter()
                    .any(|d| &*d.code == "pc.parent_limit_removed_adjacencies"),
                "{:?}",
                result.diagnostics
            );
            // A limit that never binds changes nothing and reports nothing.
            let mut loose = base();
            loose.max_parents = Some(3);
            let result = run(4, &STAR, loose).unwrap();
            assert_eq!(degree(&result.evidence.graph, 0), 3);
            assert!(result.diagnostics.iter().all(|d| !d.code.contains("parent_limit")));
        }

        #[test]
        fn adjacencies_known_to_be_children_do_not_count_toward_the_limit() {
            // 0 -> 1 is required, so 1 cannot be a parent of 0: only 2 and 3 count.
            let mut c = base();
            c.max_parents = Some(2);
            c.required = Arc::from([contemp_link(v(0), v(1))]);
            let result = run(4, &STAR, c).unwrap();
            assert_eq!(degree(&result.evidence.graph, 0), 3);
        }

        #[test]
        fn required_adjacencies_beyond_the_limit_cannot_all_hold() {
            let mut c = base();
            c.max_parents = Some(2);
            c.required = Arc::from([
                contemp_link(v(1), v(0)),
                contemp_link(v(2), v(0)),
                contemp_link(v(3), v(0)),
            ]);
            assert!(matches!(
                run(4, &STAR, c).unwrap_err(),
                DiscoveryError::Knowledge(KnowledgeFault::Contradictory(_))
            ));
        }

        #[test]
        fn unusable_knowledge_is_refused_before_any_test() {
            let fault = |c: DiscoveryConstraints| match run(2, &[(0, 1)], c).unwrap_err() {
                DiscoveryError::Knowledge(fault) => fault,
                other => panic!("expected a knowledge fault, got {other:?}"),
            };
            let mut contradictory = base();
            contradictory.required = Arc::from([contemp_link(v(0), v(1))]);
            contradictory.forbidden = Arc::from([contemp_link(v(0), v(1))]);
            assert!(matches!(fault(contradictory), KnowledgeFault::Contradictory(_)));

            let mut unknown = base();
            unknown.required = Arc::from([contemp_link(v(0), v(7))]);
            assert!(matches!(fault(unknown), KnowledgeFault::UnknownVariable(_)));
        }
    }
}
