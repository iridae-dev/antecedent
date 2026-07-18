//! J-PCMCI+: multi-environment PCMCI+ with context and dummy nodes (Günther et al. UAI 2023).
//!
//! Pools environments into one lagged frame (no cross-env lag windows), synthesizes
//! space/time dummies, and runs the four-phase skeleton + PCMCI+ orientation under
//! Günther link assumptions. Observed context and dummies enter CI tests.
//!
//! Reference: Günther, Ninad, Runge — *Causal discovery for time series from multiple
//! datasets with latent contexts*, UAI 2023 (arXiv:2306.12896); pinned baseline `JPCMCIplus`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::too_many_lines
)]

use std::collections::HashMap;
use std::sync::Arc;

use causal_core::{AssumptionSet, ExecutionContext, Lag, VariableId};
use causal_data::{DummyOptions, MultiEnvironmentData, pool_multi_env_lagged_frame};
use causal_graph::{DenseNodeId, TemporalCpdag, TemporalCpdagReview};
use causal_stats::{
    ConfidenceMethod, ConditionalIndependence, FdrAdjustment, PairwiseMultivariateCi,
};

use crate::constraints::{
    ContextKind, DiscoveryConstraints, JpcmciNodeRole, MultiDatasetConstraints, SpaceDummyCiMode,
};
use crate::engine::{DiscoveryWorkspace, PcmciEngine};
use crate::error::DiscoveryError;
use crate::evidence::{
    cpdag_evidence_from_oriented, symmetrize_contemporaneous_links, threshold_scored_links,
};
use crate::orientation::{
    ContempMeekR1, ContempMeekR2, ContempMeekR3, OrientationRule, run_orientation_to_fixed_point,
    try_orient_undirected,
};
use crate::pcmci_plus::{contemp_mci_phase, lagged_pc1_parents, orient_majority_colliders};
use crate::pipeline::{
    algorithm_record, lagged_node_index, orientation_state_from_sepsets, push_diagnostic,
};
use crate::result::{
    CpdagDiscoveryResult, DiscoveryIteration, DiscoveryPerformanceRecord, LaggedLink,
    ScoredLink,
};

/// Alias for J-PCMCI+ discovery output (context-augmented temporal CPDAG).
pub type JpcmciPlusDiscoveryResult = CpdagDiscoveryResult;

/// J-PCMCI+ discovery over [`MultiEnvironmentData`].
///
/// Own type (not a PCMCI+ flag). Implements Günther et al. pooled four-phase search.
#[derive(Clone, Debug)]
pub struct JpcmciPlus {
    /// Shared engine (`min_lag` typically 0).
    pub engine: PcmciEngine,
    /// Multiple-testing adjustment on scored links (`None` = off).
    pub fdr: Option<FdrAdjustment>,
}

impl Default for JpcmciPlus {
    fn default() -> Self {
        Self::new()
    }
}

impl JpcmciPlus {
    /// Default J-PCMCI+ with `min_lag = 0` and space dummy enabled.
    #[must_use]
    pub fn new() -> Self {
        let mut constraints = DiscoveryConstraints::default();
        constraints.temporal.min_lag = Lag::CONTEMPORANEOUS;
        constraints.multi_dataset.include_space_dummy = true;
        constraints.multi_dataset.include_time_dummy = false;
        Self {
            engine: PcmciEngine::new().with_constraints(constraints),
            fdr: Some(FdrAdjustment::bh()),
        }
    }

    /// Configure constraints (caller should keep `min_lag = 0` for contemporaneous discovery).
    #[must_use]
    pub fn with_constraints(mut self, constraints: DiscoveryConstraints) -> Self {
        self.engine.constraints = constraints;
        self
    }

    /// Replace multi-dataset / context settings.
    #[must_use]
    pub fn with_multi_dataset(mut self, multi: MultiDatasetConstraints) -> Self {
        self.engine.constraints.multi_dataset = multi;
        self
    }

    /// Enable / disable BH FDR.
    #[must_use]
    pub fn with_fdr(mut self, fdr: bool) -> Self {
        self.fdr = fdr.then(FdrAdjustment::bh);
        self
    }

    /// Full FDR / FWER configuration.
    #[must_use]
    pub fn with_fdr_adjustment(mut self, fdr: Option<FdrAdjustment>) -> Self {
        self.fdr = fdr;
        self
    }

    /// Replace the CI test on the shared engine.
    #[must_use]
    pub fn with_ci(mut self, ci: Arc<dyn ConditionalIndependence + Send + Sync>) -> Self {
        self.engine = self.engine.with_ci(ci);
        self
    }

    /// Run J-PCMCI+ on multi-environment data.
    ///
    /// `variables` are **system** nodes. Observed context comes from
    /// [`MultiDatasetConstraints::context_variables`]; dummies are synthesized.
    ///
    /// # Errors
    ///
    /// Empty multi-env, pooling / engine / orientation failures.
    pub fn run(
        &self,
        data: &MultiEnvironmentData,
        variables: &[VariableId],
        workspace: &mut DiscoveryWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<JpcmciPlusDiscoveryResult, DiscoveryError> {
        if data.env_count() == 0 {
            return Err(DiscoveryError::Unsupported {
                message: "J-PCMCI+ needs ≥1 environment",
            });
        }
        if variables.is_empty() {
            return Err(DiscoveryError::Unsupported {
                message: "J-PCMCI+ needs ≥1 system variable",
            });
        }
        self.engine.constraints.validate()?;

        let md = &self.engine.constraints.multi_dataset;
        for &c in md.context_variables.iter() {
            if variables.contains(&c) {
                return Err(DiscoveryError::data_msg(format!(
                    "context variable {c} must not also appear in the system variable list"
                )));
            }
        }

        let max_lag = self.engine.constraints.temporal.max_lag.raw();
        let frame_depth = 2 * max_lag;
        let mut observed = variables.to_vec();
        observed.extend_from_slice(&md.context_variables);

        let dummies = DummyOptions {
            include_space_dummy: md.include_space_dummy && data.env_count() > 1,
            include_time_dummy: md.include_time_dummy,
        };
        let pooled =
            pool_multi_env_lagged_frame(data, &observed, frame_depth, dummies, &ctx.kernel_policy)
                .map_err(DiscoveryError::from)?;

        let space_ids_full = Arc::clone(&pooled.space_dummy_variables);
        let use_mv_space_dummy = md.space_dummy_ci == SpaceDummyCiMode::MultivariateBlock
            && !space_ids_full.is_empty();
        let logical_space_dummies: Arc<[VariableId]> = if use_mv_space_dummy {
            Arc::from([space_ids_full[0]])
        } else {
            Arc::clone(&space_ids_full)
        };

        let mut constraints = self.engine.constraints.clone();
        constraints.multi_dataset.space_dummy_variables = Arc::clone(&logical_space_dummies);
        constraints.multi_dataset.time_dummy_variables = pooled
            .time_dummy_variable
            .map(|t| Arc::from([t]) as Arc<[VariableId]>)
            .unwrap_or_else(|| Arc::from([]));
        constraints.multi_dataset.space_dummy_ci = md.space_dummy_ci;

        let system: Vec<VariableId> = variables.to_vec();
        let context: Vec<VariableId> = md.context_variables.to_vec();
        let time_context: Vec<VariableId> = context
            .iter()
            .copied()
            .filter(|&v| constraints.multi_dataset.context_kind(v) == ContextKind::Time)
            .collect();
        // Search graph uses the logical space-dummy id in MV mode (one-hot cols stay in frame).
        let mut all_vars = pooled.observed_variables.to_vec();
        all_vars.extend_from_slice(&logical_space_dummies);
        if let Some(t) = pooled.time_dummy_variable {
            all_vars.push(t);
        }
        let frame = &pooled.frame;

        if let Some(hard) = ctx.memory.hard_limit_bytes {
            if frame.values_bytes() > hard {
                return Err(DiscoveryError::Unsupported {
                    message: "pooled lagged frame exceeds ExecutionContext memory hard limit",
                });
            }
        }

        let ci: Arc<dyn ConditionalIndependence + Send + Sync> = if use_mv_space_dummy {
            let groups = causal_data::VectorVariableGroups::try_new(Arc::from([Arc::from(
                space_ids_full.to_vec(),
            )]))
            .map_err(DiscoveryError::from)?;
            let blocks = causal_data::column_blocks_for_frame(&groups, frame)
                .map_err(DiscoveryError::from)?;
            Arc::new(PairwiseMultivariateCi::with_column_blocks(blocks))
        } else {
            Arc::clone(&self.engine.ci)
        };

        let threads = ctx.parallelism.max_threads.get().max(1);
        {
            let cols: Vec<&[f64]> = (0..frame.ncols()).map(|i| frame.column(i)).collect();
            let plan = causal_stats::CiPreparationPlan {
                significance: constraints.significance,
                confidence: ConfidenceMethod::default(),
            };
            workspace.prepared_ci =
                Some(ci.prepare(&cols, &plan, ctx).map_err(DiscoveryError::from)?);
        }

        let engine = PcmciEngine {
            constraints: constraints.clone(),
            ci: Arc::clone(&ci),
            column_blocks: if use_mv_space_dummy {
                let groups = causal_data::VectorVariableGroups::try_new(Arc::from([Arc::from(
                    space_ids_full.to_vec(),
                )]))
                .map_err(DiscoveryError::from)?;
                causal_data::column_blocks_for_frame(&groups, frame).map_err(DiscoveryError::from)?
            } else {
                Arc::from([])
            },
        };

        let mut diagnostics = Vec::new();
        let space_diag = if use_mv_space_dummy {
            format!(
                "multivariate(k={})",
                space_ids_full.len()
            )
        } else {
            format!("{}", space_ids_full.len())
        };
        push_diagnostic(
            &mut diagnostics,
            "jpcmci_plus.pooled_frame",
            format!(
                "pooled {} envs → {} effective rows, {} observed + {} space-dummy + {} time-dummy cols",
                data.env_count(),
                frame.n_effective(),
                observed.len(),
                space_diag,
                usize::from(pooled.time_dummy_variable.is_some())
            ),
        );

        // --- Phase 1: PC1 lagged on system + time context ---
        let lagged_vars: Vec<VariableId> = system
            .iter()
            .chain(time_context.iter())
            .copied()
            .collect();
        let (mut lagged_parents, mut iterations, mut ci_tests, mut sepsets) =
            lagged_pc1_parents(&engine, frame, &lagged_vars, workspace, ctx, threads)?;
        // Ensure every system/context/dummy target has an entry.
        for &v in &all_vars {
            if !lagged_parents.iter().any(|(t, _)| *t == v) {
                lagged_parents.push((v, Vec::new()));
            }
        }
        iterations.push(DiscoveryIteration {
            label: Arc::from("jpcmci_plus.lagged_pc1"),
            ci_tests,
        });

        // --- Phase 2: MCI context–system ---
        let phase2_vars: Vec<VariableId> = system.iter().chain(context.iter()).copied().collect();
        let compiled2 = engine.constraints.compile(&phase2_vars)?;
        let search_context = |link: LaggedLink| {
            let sr = constraints.multi_dataset.role_of(link.source);
            let tr = constraints.multi_dataset.role_of(link.target);
            (sr.is_observed_context() && tr == JpcmciNodeRole::System)
                || (tr.is_observed_context() && sr == JpcmciNodeRole::System)
        };
        let (ctx_scored, ctx_sep, ctx_tests, trunc_a) = contemp_mci_phase(
            &engine,
            frame,
            &phase2_vars,
            &compiled2,
            &lagged_parents,
            workspace,
            ctx,
            Some(&search_context),
        )?;
        ci_tests += ctx_tests;
        iterations.push(DiscoveryIteration {
            label: Arc::from("jpcmci_plus.context_mci"),
            ci_tests: ctx_tests,
        });
        for (k, v) in ctx_sep {
            sepsets.insert(k, v);
        }

        let context_parents = exogenous_parents_from_scored(&ctx_scored, &constraints, true, false);
        // Strip rejected B̂^C from conditioners; keep only context MCI survivors.
        replace_exogenous_parents(
            &mut lagged_parents,
            &context_parents,
            &constraints,
            |r| r.is_observed_context(),
        );

        // --- Phase 3: MCI dummy–system (if any dummies) ---
        let mut dummy_scored = Vec::new();
        let mut trunc_b = 0u64;
        if !space_ids_full.is_empty() || pooled.time_dummy_variable.is_some() {
            let mut cons3 = constraints.clone();
            // Fix discovered context → system as required.
            let mut required = cons3.required.to_vec();
            required.extend(directed_exogenous_links(&context_parents));
            cons3.required = Arc::from(required);
            let engine3 = PcmciEngine {
                constraints: cons3.clone(),
                ci: Arc::clone(&engine.ci),
                column_blocks: Arc::clone(&engine.column_blocks),
            };
            let compiled3 = engine3.constraints.compile(&all_vars)?;
            let search_dummy = |link: LaggedLink| {
                let sr = constraints.multi_dataset.role_of(link.source);
                let tr = constraints.multi_dataset.role_of(link.target);
                (sr.is_dummy() && tr == JpcmciNodeRole::System)
                    || (tr.is_dummy() && sr == JpcmciNodeRole::System)
            };
            let (scored, dum_sep, dum_tests, t) = contemp_mci_phase(
                &engine3,
                frame,
                &all_vars,
                &compiled3,
                &lagged_parents,
                workspace,
                ctx,
                Some(&search_dummy),
            )?;
            trunc_b = t;
            ci_tests += dum_tests;
            iterations.push(DiscoveryIteration {
                label: Arc::from("jpcmci_plus.dummy_mci"),
                ci_tests: dum_tests,
            });
            for (k, v) in dum_sep {
                sepsets.insert(k, v);
            }
            dummy_scored = scored;
        }
        let dummy_parents =
            exogenous_parents_from_scored(&dummy_scored, &constraints, false, true);
        // Strip rejected B̂^{CD}; keep only dummy MCI survivors for phase 4.
        replace_exogenous_parents(
            &mut lagged_parents,
            &dummy_parents,
            &constraints,
            |r| r.is_dummy(),
        );

        // --- Phase 4: MCI system–system ---
        let mut cons4 = constraints.clone();
        let mut required4 = cons4.required.to_vec();
        required4.extend(directed_exogenous_links(&context_parents));
        required4.extend(directed_exogenous_links(&dummy_parents));
        cons4.required = Arc::from(required4);
        let engine4 = PcmciEngine {
            constraints: cons4.clone(),
            ci: Arc::clone(&engine.ci),
            column_blocks: Arc::clone(&engine.column_blocks),
        };
        let compiled4 = engine4.constraints.compile(&all_vars)?;
        let search_system = |link: LaggedLink| {
            constraints.multi_dataset.role_of(link.source) == JpcmciNodeRole::System
                && constraints.multi_dataset.role_of(link.target) == JpcmciNodeRole::System
        };
        let (sys_scored, sys_sep, sys_tests, trunc_c) = contemp_mci_phase(
            &engine4,
            frame,
            &all_vars,
            &compiled4,
            &lagged_parents,
            workspace,
            ctx,
            Some(&search_system),
        )?;
        ci_tests += sys_tests;
        iterations.push(DiscoveryIteration {
            label: Arc::from("jpcmci_plus.system_mci"),
            ci_tests: sys_tests,
        });
        for (k, v) in sys_sep {
            sepsets.insert(k, v);
        }

        let truncated = trunc_a + trunc_b + trunc_c;
        if truncated > 0 {
            push_diagnostic(
                &mut diagnostics,
                "mci.conditioning_truncated",
                format!(
                    "MCI conditioning sets dropped {truncated} weakest condition(s) at the column cap"
                ),
            );
        }

        // Merge phase scored survivors only (PCMCI+ style: no PC1 re-injection).
        let space_rep = logical_space_dummies.first().copied();
        let mut scored = Vec::new();
        scored.extend(ctx_scored);
        scored.extend(dummy_scored);
        scored.extend(sys_scored);
        scored = remap_space_dummy_links(scored, &space_ids_full, space_rep);
        scored = threshold_scored_links(scored, self.fdr, constraints.alpha);
        scored = symmetrize_contemporaneous_links(scored);
        // Exogenous → system: force directed (no undirected symmetrize residue).
        scored = orient_exogenous_links(scored, &constraints);

        let logical_exog = logical_exogenous_ids(&context, space_rep, pooled.time_dummy_variable);
        let mut cpdag = cpdag_from_jpcmci_links(&scored, &system, &logical_exog, max_lag)?;
        let node_ids = lagged_node_index(cpdag.nodes());
        let mut state = orientation_state_from_sepsets(&node_ids, &sepsets);

        // Force-direct exogenous → system edges before Meek.
        force_orient_exogenous(&mut cpdag, &mut state, &logical_exog, &system)?;

        let majority_delta = orient_majority_colliders(
            &engine4,
            frame,
            &lagged_parents,
            &mut cpdag,
            &mut state,
            workspace,
            ctx,
        )?;
        let rules: [&dyn OrientationRule; 3] = [&ContempMeekR1, &ContempMeekR2, &ContempMeekR3];
        let meek_delta = run_orientation_to_fixed_point(&mut cpdag, &rules, &mut state)?;

        let algorithm = algorithm_record(
            "jpcmci_plus",
            format!(
                "alpha={},max_lag={},fdr={:?},envs={},context={},space_dummy={},space_dummy_ci={:?},time_dummy={}",
                constraints.alpha,
                max_lag,
                self.fdr,
                data.env_count(),
                context.len(),
                space_ids_full.len(),
                md.space_dummy_ci,
                usize::from(pooled.time_dummy_variable.is_some())
            ),
        );
        let evidence = cpdag_evidence_from_oriented(cpdag.clone(), scored, &sepsets);
        let review = TemporalCpdagReview::from_cpdag(cpdag, algorithm.id.clone());
        let links_retained = evidence.links.len() as u64;
        push_diagnostic(
            &mut diagnostics,
            "jpcmci_plus.cpdag",
            format!(
                "Günther J-PCMCI+ CPDAG ({} nodes, {} envs)",
                evidence.graph.node_count(),
                data.env_count()
            ),
        );
        let conflicts = state.conflicts + majority_delta.conflicts + meek_delta.conflicts;
        if conflicts > 0 {
            push_diagnostic(
                &mut diagnostics,
                "orientation.conflicts",
                format!("{conflicts} orientation conflict(s) recorded"),
            );
        }

        Ok(CpdagDiscoveryResult {
            evidence,
            review,
            algorithm,
            assumptions: AssumptionSet::new(),
            iterations,
            diagnostics,
            performance: DiscoveryPerformanceRecord {
                ci_tests,
                links_retained,
                targets: system.len() as u64,
                lagged_frame_bytes: frame.values_bytes(),
                worker_threads: threads,
            },
            sepsets,
        })
    }
}

fn logical_exogenous_ids(
    context: &[VariableId],
    space_rep: Option<VariableId>,
    time_dummy: Option<VariableId>,
) -> Vec<VariableId> {
    let mut out = context.to_vec();
    if let Some(s) = space_rep {
        out.push(s);
    }
    if let Some(t) = time_dummy {
        out.push(t);
    }
    out
}

fn exogenous_parents_from_scored(
    scored: &[ScoredLink],
    constraints: &DiscoveryConstraints,
    want_context: bool,
    want_dummy: bool,
) -> HashMap<VariableId, Vec<(VariableId, Lag)>> {
    let mut map: HashMap<VariableId, Vec<(VariableId, Lag)>> = HashMap::new();
    for s in scored {
        let sr = constraints.multi_dataset.role_of(s.link.source);
        let tr = constraints.multi_dataset.role_of(s.link.target);
        let src_ok = (want_context && sr.is_observed_context()) || (want_dummy && sr.is_dummy());
        if src_ok && tr == JpcmciNodeRole::System {
            map.entry(s.link.target).or_default().push((s.link.source, s.link.source_lag));
        }
        // Contemporaneous tests may also list system → context; flip under exogeneity.
        let tgt_ok = (want_context && tr.is_observed_context()) || (want_dummy && tr.is_dummy());
        if tgt_ok && sr == JpcmciNodeRole::System {
            map.entry(s.link.source).or_default().push((s.link.target, Lag::CONTEMPORANEOUS));
        }
    }
    map
}

fn directed_exogenous_links(
    parents: &HashMap<VariableId, Vec<(VariableId, Lag)>>,
) -> Vec<LaggedLink> {
    let mut out = Vec::new();
    for (&target, list) in parents {
        for &(source, source_lag) in list {
            out.push(LaggedLink {
                source,
                source_lag,
                target,
                target_lag: Lag::CONTEMPORANEOUS,
            });
        }
    }
    out
}

/// Replace all parents of a given exogenous role class with MCI survivors.
///
/// Rejected lagged context/dummy links from PC1 must leave the conditioner set used
/// in later phases (pinned baseline `observed_context_parents` / `dummy_parents`).
fn replace_exogenous_parents(
    lagged_parents: &mut [(VariableId, Vec<(VariableId, Lag)>)],
    survivors: &HashMap<VariableId, Vec<(VariableId, Lag)>>,
    constraints: &DiscoveryConstraints,
    mut match_role: impl FnMut(JpcmciNodeRole) -> bool,
) {
    for (target, list) in lagged_parents.iter_mut() {
        list.retain(|&(src, _)| !match_role(constraints.multi_dataset.role_of(src)));
        if let Some(more) = survivors.get(target) {
            for &p in more {
                if !list.contains(&p) {
                    list.push(p);
                }
            }
        }
    }
}

fn remap_space_dummy_links(
    scored: Vec<ScoredLink>,
    space_dummies: &[VariableId],
    rep: Option<VariableId>,
) -> Vec<ScoredLink> {
    let Some(rep) = rep else {
        return scored;
    };
    if space_dummies.len() <= 1 {
        return scored;
    }
    scored
        .into_iter()
        .map(|mut s| {
            if space_dummies.contains(&s.link.source) {
                s.link.source = rep;
            }
            if space_dummies.contains(&s.link.target) {
                s.link.target = rep;
            }
            s
        })
        .collect()
}

fn orient_exogenous_links(
    scored: Vec<ScoredLink>,
    constraints: &DiscoveryConstraints,
) -> Vec<ScoredLink> {
    scored
        .into_iter()
        .map(|mut s| {
            let sr = constraints.multi_dataset.role_of(s.link.source);
            let tr = constraints.multi_dataset.role_of(s.link.target);
            if tr.is_exogenous() && sr == JpcmciNodeRole::System {
                // Flip to exogenous → system.
                std::mem::swap(&mut s.link.source, &mut s.link.target);
                s.link.source_lag = Lag::CONTEMPORANEOUS;
                s.link.target_lag = Lag::CONTEMPORANEOUS;
            }
            s
        })
        .collect()
}

fn cpdag_from_jpcmci_links(
    links: &[ScoredLink],
    system: &[VariableId],
    exogenous: &[VariableId],
    max_lag: u32,
) -> Result<TemporalCpdag, DiscoveryError> {
    let mut cpdag = TemporalCpdag::empty();
    let mut node_ids = HashMap::<(u32, u32), DenseNodeId>::new();
    for &v in system {
        for lag in 0..=max_lag {
            let id = cpdag.add_lagged(v, Lag::from_raw(lag)).map_err(DiscoveryError::from)?;
            node_ids.insert((v.raw(), lag), id);
        }
    }
    for &v in exogenous {
        let id = cpdag.add_context(v, None).map_err(DiscoveryError::from)?;
        node_ids.insert((v.raw(), 0), id);
    }
    for link in links {
        let Some(&src) = node_ids.get(&(link.link.source.raw(), link.link.source_lag.raw())) else {
            continue;
        };
        let Some(&tgt) = node_ids.get(&(link.link.target.raw(), link.link.target_lag.raw())) else {
            continue;
        };
        if cpdag.has_edge(src, tgt) {
            continue;
        }
        let src_exog = exogenous.contains(&link.link.source);
        let contemp = link.link.source_lag.is_contemporaneous()
            && link.link.target_lag.is_contemporaneous();
        let insert = if src_exog {
            cpdag.insert_directed(src, tgt)
        } else if contemp {
            cpdag.insert_undirected(src, tgt)
        } else {
            cpdag.insert_directed(src, tgt)
        };
        match insert {
            Ok(()) => {}
            Err(
                causal_graph::GraphError::Cycle { .. }
                | causal_graph::GraphError::DuplicateEdge { .. },
            ) => {}
            Err(e) => return Err(DiscoveryError::from(e)),
        }
    }
    Ok(cpdag)
}

fn force_orient_exogenous(
    cpdag: &mut TemporalCpdag,
    state: &mut crate::orientation::OrientationState,
    exogenous: &[VariableId],
    system: &[VariableId],
) -> Result<(), DiscoveryError> {
    let node_ids = lagged_node_index(cpdag.nodes());
    let mut delta = crate::orientation::RuleDelta::default();
    for &c in exogenous {
        let Some(&cid) = node_ids.get(&(c.raw(), 0)) else {
            continue;
        };
        for &x in system {
            let Some(&xid) = node_ids.get(&(x.raw(), 0)) else {
                continue;
            };
            if cpdag.has_edge(cid, xid)
                && cpdag.edge_between(cid, xid).is_some_and(|e| e.is_undirected())
            {
                let _ = try_orient_undirected(
                    cpdag,
                    state,
                    &mut delta,
                    cid,
                    xid,
                    format!("jpcmci.exogenous:{c}→{x}"),
                )?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::sync::Arc;

    use causal_core::{
        CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
    };
    use causal_data::{
        Float64Column, MultiEnvironmentData, OwnedColumn, OwnedColumnarStorage, SamplingRegularity,
        TimeIndex, TimeSeriesData, ValidityBitmap,
    };

    use super::*;
    use crate::constraints::{
        ContextKind, MultiDatasetConstraints, SpaceDummyCiMode, TemporalConstraints,
    };

    fn toy_env(n: usize, seed: f64) -> TimeSeriesData {
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "x",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let mut x = vec![0.0; n];
        let mut y = vec![0.0; n];
        for t in 1..n {
            x[t] = 0.5 * x[t - 1] + 0.1 * ((t as f64) + seed).sin();
            y[t] = 0.7 * x[t] + 0.2 * y[t - 1] + 0.05 * ((t as f64) + seed).cos();
        }
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(x),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(y),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        TimeSeriesData::try_new(
            storage,
            TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
        )
        .unwrap()
    }

    #[test]
    fn jpcmci_plus_two_env_toy() {
        let a = toy_env(120, 0.0);
        let b = toy_env(120, 1.0);
        let multi = MultiEnvironmentData::try_new(Arc::from([a, b])).unwrap();
        let vars = [VariableId::from_raw(0), VariableId::from_raw(1)];
        let algo = JpcmciPlus::new().with_fdr(false).with_constraints(DiscoveryConstraints {
            temporal: TemporalConstraints {
                max_lag: Lag::from_raw(1),
                min_lag: Lag::CONTEMPORANEOUS,
            },
            multi_dataset: MultiDatasetConstraints {
                include_space_dummy: true,
                ..MultiDatasetConstraints::default()
            },
            ..DiscoveryConstraints::default()
        });
        let mut ws = DiscoveryWorkspace::default();
        let ctx = ExecutionContext::for_tests(7);
        let result = algo.run(&multi, &vars, &mut ws, &ctx).unwrap();
        assert_eq!(result.algorithm.id.as_ref(), "jpcmci_plus");
        assert!(result.evidence.graph.node_count() >= 2);
        assert!(
            result.diagnostics.iter().any(|d| d.code.as_ref() == "jpcmci_plus.pooled_frame"),
            "expected pooled_frame diagnostic"
        );
    }

    #[test]
    fn gunther_forbids_wired_into_compile() {
        let sys = VariableId::from_raw(0);
        let ctx = VariableId::from_raw(1);
        let c = DiscoveryConstraints {
            multi_dataset: MultiDatasetConstraints {
                context_variables: Arc::from([ctx]),
                ..MultiDatasetConstraints::default()
            },
            temporal: TemporalConstraints {
                max_lag: Lag::from_raw(1),
                min_lag: Lag::CONTEMPORANEOUS,
            },
            ..DiscoveryConstraints::default()
        };
        let compiled = c.compile(&[sys, ctx]).unwrap();
        let into_ctx = LaggedLink {
            source: sys,
            source_lag: Lag::CONTEMPORANEOUS,
            target: ctx,
            target_lag: Lag::CONTEMPORANEOUS,
        };
        assert!(!compiled.allows(into_ctx));
    }

    fn toy_env_with_context(n: usize, seed: f64, c_level: f64) -> TimeSeriesData {
        let mut b = CausalSchemaBuilder::new();
        for name in ["x", "y", "c"] {
            b.add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        }
        let schema = b.build().unwrap();
        let mut x = vec![0.0; n];
        let mut y = vec![0.0; n];
        let ccol = vec![c_level; n];
        for t in 1..n {
            x[t] = 0.4 * x[t - 1] + 0.8 * c_level + 0.05 * ((t as f64) + seed).sin();
            y[t] = 0.5 * y[t - 1] + 0.6 * x[t] + 0.05 * ((t as f64) + seed).cos();
        }
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(x),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(y),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(2),
                    Arc::from(ccol),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        TimeSeriesData::try_new(
            storage,
            TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
        )
        .unwrap()
    }

    #[test]
    fn observed_context_enters_ci() {
        let a = toy_env_with_context(200, 0.0, -1.0);
        let b = toy_env_with_context(200, 1.0, 1.0);
        let multi = MultiEnvironmentData::try_new(Arc::from([a, b])).unwrap();
        let system = [VariableId::from_raw(0), VariableId::from_raw(1)];
        let ctx = VariableId::from_raw(2);
        let algo = JpcmciPlus::new().with_fdr(false).with_constraints(DiscoveryConstraints {
            temporal: TemporalConstraints {
                max_lag: Lag::from_raw(1),
                min_lag: Lag::CONTEMPORANEOUS,
            },
            alpha: 0.2,
            multi_dataset: MultiDatasetConstraints {
                context_variables: Arc::from([ctx]),
                include_space_dummy: true,
                ..MultiDatasetConstraints::default()
            },
            ..DiscoveryConstraints::default()
        });
        let mut ws = DiscoveryWorkspace::default();
        let result = algo.run(&multi, &system, &mut ws, &ExecutionContext::for_tests(11)).unwrap();
        assert!(
            result.iterations.iter().any(|i| i.label.as_ref() == "jpcmci_plus.context_mci"),
            "context MCI phase should run"
        );
        assert!(
            result.diagnostics.iter().any(|d| d.code.as_ref() == "jpcmci_plus.pooled_frame"),
            "pooled frame diagnostic expected"
        );
        // Context node present in CPDAG.
        assert!(
            result.evidence.graph.nodes().iter().any(|n| matches!(
                n,
                causal_graph::NodeRef::Context { variable, .. } if *variable == ctx
            )),
            "observed context should appear as Context node"
        );
    }

    #[test]
    fn replace_exogenous_parents_drops_rejected_context() {
        let sys = VariableId::from_raw(0);
        let ctx_v = VariableId::from_raw(1);
        let sys_parent = VariableId::from_raw(2);
        let constraints = DiscoveryConstraints {
            multi_dataset: MultiDatasetConstraints {
                context_variables: Arc::from([ctx_v]),
                context_kinds: Arc::from([(ctx_v, ContextKind::Time)]),
                ..MultiDatasetConstraints::default()
            },
            ..DiscoveryConstraints::default()
        };
        let mut lagged_parents = vec![(
            sys,
            vec![
                (ctx_v, Lag::from_raw(1)),
                (sys_parent, Lag::from_raw(1)),
                (ctx_v, Lag::CONTEMPORANEOUS),
            ],
        )];
        // MCI kept only contemporaneous context → system.
        let survivors = HashMap::from([(sys, vec![(ctx_v, Lag::CONTEMPORANEOUS)])]);
        replace_exogenous_parents(
            &mut lagged_parents,
            &survivors,
            &constraints,
            |r| r.is_observed_context(),
        );
        let parents = &lagged_parents[0].1;
        assert!(
            parents.contains(&(ctx_v, Lag::CONTEMPORANEOUS)),
            "survivor context parent must remain"
        );
        assert!(
            !parents.contains(&(ctx_v, Lag::from_raw(1))),
            "MCI-rejected lagged context must leave conditioner set"
        );
        assert!(
            parents.contains(&(sys_parent, Lag::from_raw(1))),
            "system lagged parents must be untouched"
        );
    }

    #[test]
    fn replace_exogenous_parents_drops_rejected_dummy() {
        let sys = VariableId::from_raw(0);
        let dummy = VariableId::from_raw(10);
        let sys_parent = VariableId::from_raw(1);
        let constraints = DiscoveryConstraints {
            multi_dataset: MultiDatasetConstraints {
                space_dummy_variables: Arc::from([dummy]),
                ..MultiDatasetConstraints::default()
            },
            ..DiscoveryConstraints::default()
        };
        let mut lagged_parents = vec![(
            sys,
            vec![
                (dummy, Lag::CONTEMPORANEOUS),
                (sys_parent, Lag::from_raw(1)),
            ],
        )];
        // Dummy link rejected by MCI → empty survivors.
        let survivors = HashMap::new();
        replace_exogenous_parents(
            &mut lagged_parents,
            &survivors,
            &constraints,
            |r| r.is_dummy(),
        );
        let parents = &lagged_parents[0].1;
        assert!(
            !parents.iter().any(|&(s, _)| s == dummy),
            "rejected dummy parent must be stripped"
        );
        assert!(
            parents.contains(&(sys_parent, Lag::from_raw(1))),
            "system parents must remain"
        );
    }

    /// Smooth time context drives X only contemporaneously; lagged C→X is PC1-plausible
    /// (autocorrelated C) but MCI-rejected given C_t. Must not reappear in the CPDAG.
    fn toy_env_time_context_contemp_only(n: usize, seed: f64, env_shift: f64) -> TimeSeriesData {
        let mut b = CausalSchemaBuilder::new();
        for name in ["x", "y", "c"] {
            b.add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        }
        let schema = b.build().unwrap();
        let mut x = vec![0.0; n];
        let mut y = vec![0.0; n];
        let mut c = vec![0.0; n];
        for t in 0..n {
            let tf = t as f64;
            c[t] = env_shift + (0.3 * tf + seed).sin() + 0.15 * (0.7 * tf + seed).cos();
            let eps_x = 0.05 * ((tf + seed * 1.1).sin());
            let eps_y = 0.05 * ((tf + seed * 1.3).cos());
            x[t] = 0.85 * c[t] + eps_x;
            y[t] = 0.6 * x[t] + eps_y;
        }
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(x),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(y),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(2),
                    Arc::from(c),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        TimeSeriesData::try_new(
            storage,
            TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
        )
        .unwrap()
    }

    #[test]
    fn mci_rejected_lagged_context_absent_from_cpdag() {
        let a = toy_env_time_context_contemp_only(400, 0.0, -1.0);
        let b = toy_env_time_context_contemp_only(400, 1.0, 1.0);
        let multi = MultiEnvironmentData::try_new(Arc::from([a, b])).unwrap();
        let system = [VariableId::from_raw(0), VariableId::from_raw(1)];
        let ctx_v = VariableId::from_raw(2);
        let algo = JpcmciPlus::new().with_fdr(false).with_constraints(DiscoveryConstraints {
            temporal: TemporalConstraints {
                max_lag: Lag::from_raw(1),
                min_lag: Lag::CONTEMPORANEOUS,
            },
            alpha: 0.05,
            multi_dataset: MultiDatasetConstraints {
                context_variables: Arc::from([ctx_v]),
                context_kinds: Arc::from([(ctx_v, ContextKind::Time)]),
                include_space_dummy: true,
                ..MultiDatasetConstraints::default()
            },
            ..DiscoveryConstraints::default()
        });
        let mut ws = DiscoveryWorkspace::default();
        let result = algo.run(&multi, &system, &mut ws, &ExecutionContext::for_tests(23)).unwrap();

        let lagged_ctx_to_system = result.evidence.links.iter().any(|s| {
            s.link.source == ctx_v
                && !s.link.source_lag.is_contemporaneous()
                && (s.link.target == system[0] || s.link.target == system[1])
        });
        assert!(
            !lagged_ctx_to_system,
            "lagged context→system removed by context MCI must not re-enter the CPDAG \
             (old lagged_parents_as_scored reinjection); links={:?}",
            result
                .evidence
                .links
                .iter()
                .map(|s| (s.link.source.raw(), s.link.source_lag.raw(), s.link.target.raw(), s.p_value))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn multivariate_space_dummy_uses_single_logical_node() {
        // M=3 → two one-hot columns; MV mode exposes one logical SpaceDummy in the CPDAG.
        let envs = [
            toy_env(180, 0.0),
            toy_env(180, 1.0),
            toy_env(180, 2.0),
        ];
        let multi = MultiEnvironmentData::try_new(Arc::from(envs)).unwrap();
        let vars = [VariableId::from_raw(0), VariableId::from_raw(1)];
        let algo = JpcmciPlus::new().with_fdr(false).with_constraints(DiscoveryConstraints {
            temporal: TemporalConstraints {
                max_lag: Lag::from_raw(1),
                min_lag: Lag::CONTEMPORANEOUS,
            },
            alpha: 0.1,
            multi_dataset: MultiDatasetConstraints {
                include_space_dummy: true,
                space_dummy_ci: SpaceDummyCiMode::MultivariateBlock,
                ..MultiDatasetConstraints::default()
            },
            ..DiscoveryConstraints::default()
        });
        let mut ws = DiscoveryWorkspace::default();
        let result = algo.run(&multi, &vars, &mut ws, &ExecutionContext::for_tests(31)).unwrap();
        assert!(
            result
                .diagnostics
                .iter()
                .any(|d| d.message.contains("multivariate(k=2)")),
            "expected multivariate(k=2) diagnostic; got {:?}",
            result.diagnostics
        );
        let space_ids: BTreeSet<u32> = result
            .evidence
            .links
            .iter()
            .flat_map(|s| [s.link.source.raw(), s.link.target.raw()])
            .filter(|&id| id >= 2)
            .collect();
        assert!(
            space_ids.len() <= 1,
            "MV mode must collapse to ≤1 logical space-dummy id; got {space_ids:?}"
        );
        assert!(
            result.algorithm.config.as_ref().contains("space_dummy_ci=MultivariateBlock"),
            "config={}",
            result.algorithm.config
        );
    }
}