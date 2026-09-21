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

use antecedent_core::{ExecutionContext, Lag, VariableId};
use antecedent_data::{TableView, TabularData};
use antecedent_graph::{Cpdag, CpdagReview, DenseNodeId};
use antecedent_stats::{
    CiPreparationPlan, ConditionalIndependence, ConfidenceMethod, FdrAdjustment,
    PartialCorrelation, PreparedCiTest,
};

use crate::constraints::DiscoveryConstraints;
use crate::engine::DiscoveryWorkspace;
use crate::error::DiscoveryError;
use crate::evidence::retain_after_family_fdr;
use crate::orientation::{
    MeekR1, MeekR2, MeekR3, MeekR4, OrientCollider, OrientationRule, OrientationState,
    run_static_orientation_to_fixed_point,
};
use crate::result::{
    DiscoveryDiagnostic, DiscoveryPerformanceRecord, DiscoveryResult, EvidenceSource,
    GraphEvidence, ScoredLink, discovery_assumptions,
};
use crate::static_skeleton::{
    StaticSkeleton, StaticSkeletonInput, run_static_skeleton, skeleton_scored_links,
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
        let skel = run_static_skeleton(
            &StaticSkeletonInput {
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
            mut adj, sepsets, ci_tests, iterations, family_p, family_edge, ..
        } = skel;

        // Multiplicity correction (when configured) covers the edges that were actually
        // tested; constraint-required edges carry no test and pass through untouched.
        let (tested, untested): (Vec<ScoredLink>, Vec<ScoredLink>) =
            scored.drain(..).partition(|s| s.p_value.is_finite());
        scored = retain_after_family_fdr(tested, &family_p, &family_edge, self.fdr, alpha);
        scored.extend(untested);
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
                cpdag
                    .add_node(antecedent_graph::NodeRef::Static(v))
                    .map_err(DiscoveryError::from)?;
            }
        }
        let dense_of = |v: VariableId| crate::pipeline::dense_of(&var_index, v);
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

        let rules: [&dyn OrientationRule<antecedent_graph::Cpdag>; 5] =
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
                format!("alpha={},max_cond={},fdr={}", alpha, max_cond, self.fdr.is_some()),
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
pub(crate) fn sorted_edge_pairs(adj: &HashMap<(u32, u32), ()>) -> Vec<(VariableId, VariableId)> {
    let mut edges: Vec<(VariableId, VariableId)> =
        adj.keys().map(|&(lo, hi)| (VariableId::from_raw(lo), VariableId::from_raw(hi))).collect();
    edges.sort_unstable();
    edges
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

    use antecedent_core::{
        Assumption, CausalSchemaBuilder, ExecutionContext, Lag, MeasurementSpec, RoleHint,
        SmallRoleSet, ValueType, VariableId,
    };
    use antecedent_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
    };
    use antecedent_stats::{
        CiBatchRequest, CiBatchResult, CiResult, CiWorkspace, OracleCi, StatsError,
    };

    use super::*;
    use crate::result::LaggedLink;

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

    /// C1 regression: `sorted_edge_pairs` must return a canonical ascending order, not
    /// raw `HashMap` iteration order.
    ///
    /// Builds the same 21-pair edge set (7-node complete graph) via two different
    /// insertion histories (ascending vs. descending). Pre-fix (`adj.keys().collect()`
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

        let mut ascending: HashMap<(u32, u32), ()> = HashMap::new();
        for &(a, b) in &expected {
            ascending.insert(edge_key(a, b), ());
        }
        let mut descending: HashMap<(u32, u32), ()> = HashMap::new();
        for &(a, b) in expected.iter().rev() {
            descending.insert(edge_key(a, b), ());
        }

        assert_eq!(sorted_edge_pairs(&ascending), expected);
        assert_eq!(sorted_edge_pairs(&descending), expected);
    }

    /// C1 end-to-end confirmation: two full `Pc::run()` calls on bit-identical input
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
        // but this test keeps the disjoint layout since it targets the C1/C2
        // determinism fixes, not collider orientation.
        let oracle = OracleCi::new([(0usize, 1usize), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8)]);
        let mut constraints = DiscoveryConstraints::default();
        constraints.max_cond_size = 2;
        constraints.temporal = crate::TemporalConstraints {
            max_lag: Lag::CONTEMPORANEOUS,
            min_lag: Lag::CONTEMPORANEOUS,
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
    /// conditioning set at its winning depth (C2), not whichever was tested last.
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

        let mut constraints = DiscoveryConstraints::default();
        constraints.alpha = 0.5;
        constraints.max_cond_size = 1;
        constraints.temporal = crate::TemporalConstraints {
            max_lag: Lag::CONTEMPORANEOUS,
            min_lag: Lag::CONTEMPORANEOUS,
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
        let mut constraints = DiscoveryConstraints::default();
        constraints.required = Arc::from([LaggedLink {
            source: VariableId::from_raw(0),
            source_lag: Lag::CONTEMPORANEOUS,
            target: VariableId::from_raw(1),
            target_lag: Lag::CONTEMPORANEOUS,
        }]);
        constraints.temporal = crate::TemporalConstraints {
            max_lag: Lag::CONTEMPORANEOUS,
            min_lag: Lag::CONTEMPORANEOUS,
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
}
