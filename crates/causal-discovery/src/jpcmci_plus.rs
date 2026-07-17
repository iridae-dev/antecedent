//! J-PCMCI+: multi-environment PCMCI+ with context and dummy nodes (Günther et al. UAI 2023).
//!
//! Pools environments into one lagged frame (no cross-env lag windows), synthesizes
//! space/time dummies, and runs the four-phase skeleton + PCMCI+ orientation under
//! Günther link assumptions. Observed context and dummies enter CI tests.
//!
//! Reference: Günther, Ninad, Runge — *Causal discovery for time series from multiple
//! datasets with latent contexts*, UAI 2023 (arXiv:2306.12896); tigramite `JPCMCIplus`.
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
use causal_stats::{ConfidenceMethod, ConditionalIndependence, FdrAdjustment};

use crate::constraints::{
    ContextKind, DiscoveryConstraints, JpcmciNodeRole, MultiDatasetConstraints,
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
        let pooled = pool_multi_env_lagged_frame(data, &observed, frame_depth, dummies)
            .map_err(DiscoveryError::from)?;

        let mut constraints = self.engine.constraints.clone();
        constraints.multi_dataset.space_dummy_variables =
            Arc::clone(&pooled.space_dummy_variables);
        constraints.multi_dataset.time_dummy_variables = pooled
            .time_dummy_variable
            .map(|t| Arc::from([t]) as Arc<[VariableId]>)
            .unwrap_or_else(|| Arc::from([]));

        let system: Vec<VariableId> = variables.to_vec();
        let context: Vec<VariableId> = md.context_variables.to_vec();
        let time_context: Vec<VariableId> = context
            .iter()
            .copied()
            .filter(|&v| constraints.multi_dataset.context_kind(v) == ContextKind::Time)
            .collect();
        let all_vars = pooled.all_variables();
        let frame = &pooled.frame;

        if let Some(hard) = ctx.memory.hard_limit_bytes {
            if frame.values_bytes() > hard {
                return Err(DiscoveryError::Unsupported {
                    message: "pooled lagged frame exceeds ExecutionContext memory hard limit",
                });
            }
        }

        let threads = ctx.parallelism.max_threads.get().max(1);
        {
            let cols: Vec<&[f64]> = (0..frame.ncols()).map(|i| frame.column(i)).collect();
            let plan = causal_stats::CiPreparationPlan {
                significance: constraints.significance,
                confidence: ConfidenceMethod::default(),
            };
            workspace.prepared_ci =
                Some(self.engine.ci.prepare(&cols, &plan, ctx).map_err(DiscoveryError::from)?);
        }

        let engine = PcmciEngine {
            constraints: constraints.clone(),
            ci: Arc::clone(&self.engine.ci),
        };

        let mut diagnostics = Vec::new();
        push_diagnostic(
            &mut diagnostics,
            "jpcmci_plus.pooled_frame",
            format!(
                "pooled {} envs → {} effective rows, {} observed + {} space-dummy + {} time-dummy cols",
                data.env_count(),
                frame.n_effective(),
                observed.len(),
                pooled.space_dummy_variables.len(),
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
        // Fold contemporaneous context parents into lagged_parents for later MCI.
        merge_fixed_parents(&mut lagged_parents, &context_parents);

        // --- Phase 3: MCI dummy–system (if any dummies) ---
        let mut dummy_scored = Vec::new();
        let mut trunc_b = 0u64;
        if !pooled.space_dummy_variables.is_empty() || pooled.time_dummy_variable.is_some() {
            let mut cons3 = constraints.clone();
            // Fix discovered context → system as required.
            let mut required = cons3.required.to_vec();
            required.extend(directed_exogenous_links(&context_parents));
            cons3.required = Arc::from(required);
            let engine3 = PcmciEngine { constraints: cons3.clone(), ci: Arc::clone(&engine.ci) };
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
        merge_fixed_parents(&mut lagged_parents, &dummy_parents);

        // --- Phase 4: MCI system–system ---
        let mut cons4 = constraints.clone();
        let mut required4 = cons4.required.to_vec();
        required4.extend(directed_exogenous_links(&context_parents));
        required4.extend(directed_exogenous_links(&dummy_parents));
        cons4.required = Arc::from(required4);
        let engine4 = PcmciEngine { constraints: cons4.clone(), ci: Arc::clone(&engine.ci) };
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

        // Merge scored links; remap one-hot space dummies → first space-dummy id.
        let space_rep = pooled.space_dummy_variables.first().copied();
        let mut scored = Vec::new();
        scored.extend(ctx_scored);
        scored.extend(dummy_scored);
        scored.extend(sys_scored);
        // Include lagged PC1 system/time-context parents as scored survivors.
        scored.extend(lagged_parents_as_scored(&lagged_parents, &constraints));
        scored = remap_space_dummy_links(scored, &pooled.space_dummy_variables, space_rep);
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
                "alpha={},max_lag={},fdr={:?},envs={},context={},space_dummy={},time_dummy={}",
                constraints.alpha,
                max_lag,
                self.fdr,
                data.env_count(),
                context.len(),
                pooled.space_dummy_variables.len(),
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

fn merge_fixed_parents(
    lagged_parents: &mut [(VariableId, Vec<(VariableId, Lag)>)],
    extra: &HashMap<VariableId, Vec<(VariableId, Lag)>>,
) {
    for (target, list) in lagged_parents.iter_mut() {
        if let Some(more) = extra.get(target) {
            for &p in more {
                if !list.contains(&p) {
                    list.push(p);
                }
            }
        }
    }
}

fn lagged_parents_as_scored(
    lagged_parents: &[(VariableId, Vec<(VariableId, Lag)>)],
    constraints: &DiscoveryConstraints,
) -> Vec<ScoredLink> {
    let mut out = Vec::new();
    for &(target, ref parents) in lagged_parents {
        for &(src, slag) in parents {
            if slag.is_contemporaneous() {
                continue;
            }
            // Skip exogenous→exogenous / into exogenous (already forbidden).
            if constraints.multi_dataset.gunther_forbids(LaggedLink {
                source: src,
                source_lag: slag,
                target,
                target_lag: Lag::CONTEMPORANEOUS,
            }) {
                continue;
            }
            out.push(ScoredLink {
                link: LaggedLink {
                    source: src,
                    source_lag: slag,
                    target,
                    target_lag: Lag::CONTEMPORANEOUS,
                },
                statistic: 1.0,
                p_value: 0.0,
                adjusted_p_value: None,
            });
        }
    }
    out
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
    use std::sync::Arc;

    use causal_core::{
        CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
    };
    use causal_data::{
        Float64Column, MultiEnvironmentData, OwnedColumn, OwnedColumnarStorage, SamplingRegularity,
        TimeIndex, TimeSeriesData, ValidityBitmap,
    };

    use super::*;
    use crate::constraints::{MultiDatasetConstraints, TemporalConstraints};

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
}
