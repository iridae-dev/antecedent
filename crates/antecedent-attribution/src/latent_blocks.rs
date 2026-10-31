//! Attribution on a mixed graph: noise blocks, and when they are identified.
//!
//! A latent common cause ties the noise terms of the mechanisms it reaches together, so on a
//! mixed graph the noise terms of one district (a c-component of the bidirected edges) cannot
//! be set to the reference, or swapped between populations, one at a time: they are one
//! player. This module decides when such an attribution is identified from the observed law,
//! and runs the two attributions with the blocks as players.
//!
//! # Identification
//!
//! The attribution reconstructs the target from the noise of its ancestors, so it needs the
//! structural function `f_i` of every ancestor `i` and the noise vector of every unit. Under
//! additive noise `f_i(pa) = E[x_i | do(pa)]` and the unit's noise is `x_i - f_i(pa)`. The
//! interventional law of `x_i` given its parents is the district factor `Q[{i}]` of the
//! ancestral sub-graph (the induced graph on the ancestors of the targets is the latent
//! projection onto them). Tian and Pearl (2002) identify `Q[S]` from the observed law when `S`
//! is an ancestral set of the district `D` that holds it, and Shpitser and Pearl (2006) show
//! that otherwise a hedge is present. For one node `i` that is: no other node of its district
//! reaches `i` through directed edges that stay inside the district. When that holds for every
//! ancestor, each mechanism is fitted by regressing the node on its parents, the residual
//! vector of each unit is its noise, and the blocks are the districts.
//!
//! When it fails for some ancestor the attribution is not identified, and the refusal cites the
//! hedge ([`MechanismHedge`]) and the two-SCM witness the tests construct
//! ([`HEDGE_WITNESS_ID`]): two additive-noise SCMs that agree on the observed law and give the
//! target different values at the reference noise, so different attributions to the blocks.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{ChangeAttributionQuery, ComponentId, ExecutionContext, VariableId};
use antecedent_counterfactual::{AbductionMissingPolicy, CounterfactualEngine};
use antecedent_data::TabularData;
use antecedent_graph::{Admg, BitSet, Dag, DenseNodeId, GraphWorkspace};
use antecedent_model::{
    CompiledCausalModel, CompiledMechanismStore, MechanismRegistry, MechanismSlot,
    MechanismWorkspace, NoiseBatchMut, ParentGatherPlan, SelectionPolicy, ValueBatchMut,
    evaluate_batch_topo,
};

use crate::change_common::{
    CachedOutcomeLawPayoff, ChangeAllocation, DISTRIBUTION_STREAM, DifferenceMeasure,
    attribution_stream, run_change_allocation, total_change,
};
use crate::coalition::{Coalition, full_coalition_mask};
use crate::distribution_change::{
    DistributionChangeOptions, PopulationRefit, joint_players, refit_populations,
    validate_distribution_change_query,
};
use crate::error::AttributionError;
use crate::prep::{resolve_change_populations, resolve_outcome_dense};
use crate::result::ChangeAttributionResult;
use crate::shapley::CoalitionPayoff;

/// Identifier of the two-SCM witness that a hedged mechanism is not identified. The tests that
/// construct it are in [`HEDGE_WITNESS_TEST`].
pub const HEDGE_WITNESS_ID: &str = "latent_district_ancestor_hedge";

/// The test file that builds the witness pair and checks that its two SCMs agree on the
/// observed law and disagree on the attribution.
pub const HEDGE_WITNESS_TEST: &str = "crates/antecedent-attribution/tests/latent_block_witness.rs";

/// A mechanism the observed law does not identify: `ancestor` shares a district with `node` and
/// reaches it through directed edges that stay inside the district.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MechanismHedge {
    /// The ancestor of the target whose mechanism is not identified.
    pub node: VariableId,
    /// The district member that reaches `node` inside the district.
    pub ancestor: VariableId,
    /// The directed path from `ancestor` to `node`, both ends included, inside the district.
    pub path: Vec<VariableId>,
}

impl MechanismHedge {
    /// The refusal text: the hedge, and the witness that shows it is not an artefact of the
    /// estimator.
    #[must_use]
    pub fn describe(&self) -> String {
        let path: Vec<String> = self.path.iter().map(|v| format!("v{}", v.raw())).collect();
        format!(
            "the mechanism of v{node} is not identified: v{ancestor} shares its district and \
             reaches it through directed edges inside that district ({path}), so the structural \
             function of v{node} is confounded with its noise (a hedge, Shpitser and Pearl 2006). \
             Two additive-noise SCMs on this graph agree on the observed law and differ in the \
             target at the reference noise, hence in the attribution (witness \
             {HEDGE_WITNESS_ID}, built and checked in {HEDGE_WITNESS_TEST})",
            node = self.node.raw(),
            ancestor = self.ancestor.raw(),
            path = path.join(" -> "),
        )
    }
}

/// The noise blocks of an identified attribution, and the graph its mechanisms are fitted on.
#[derive(Clone, Debug)]
pub struct NoiseBlockPlan {
    /// The directed part of the mixed graph over the same nodes: the graph the mechanisms are
    /// fitted on, each node regressed on its parents.
    pub directed: Dag,
    /// The districts of the ancestral sub-graph: a partition of the ancestors of the targets
    /// (the targets included), each block ascending by variable id and the blocks ordered by
    /// their smallest member.
    pub blocks: Vec<Vec<VariableId>>,
}

/// Whether the attribution to `targets` is identified on a mixed graph.
#[derive(Clone, Debug)]
pub enum NoiseBlockIdentification {
    /// Every ancestor's mechanism is identified: the plan to run.
    Identified(NoiseBlockPlan),
    /// Some ancestor's mechanism is not identified.
    Hedged(MechanismHedge),
}

/// Decide whether the attribution to `targets` is identified on `admg`, and plan it.
///
/// # Errors
///
/// A target that is not a node of the graph, a non-static node, or a graph error.
pub fn plan_noise_blocks(
    admg: &Admg,
    targets: &[VariableId],
) -> Result<NoiseBlockIdentification, AttributionError> {
    let n = admg.node_count();
    let mut seeds = Vec::with_capacity(targets.len());
    for &target in targets {
        seeds.push(
            admg.dense_of_static(target)
                .ok_or_else(|| AttributionError::missing_var("target", target))?,
        );
    }
    let mut ancestral = BitSet::with_len(n);
    let mut workspace = GraphWorkspace::default();
    admg.ancestors_of(&seeds, &mut ancestral, &mut workspace);
    for &seed in &seeds {
        ancestral.insert(seed);
    }
    let variable_of = |node: DenseNodeId| {
        admg.static_variable(node).ok_or(AttributionError::unsupported(
            "attribution on a mixed graph needs static variables",
        ))
    };

    // The districts of the ancestral sub-graph: the latent projection onto the ancestors.
    let components = admg.district_components_within(&ancestral);
    let mut component_of = vec![usize::MAX; n];
    for (index, component) in components.iter().enumerate() {
        for node in component.iter() {
            component_of[node.as_usize()] = index;
        }
    }

    for node in ancestral.iter() {
        let district = component_of[node.as_usize()];
        // The district members that reach `node` through edges that stay in the district.
        let mut toward = vec![None::<DenseNodeId>; n];
        let mut seen = vec![false; n];
        seen[node.as_usize()] = true;
        let mut stack = vec![node];
        while let Some(current) = stack.pop() {
            for &parent in admg.parents(current) {
                let index = parent.as_usize();
                if !seen[index] && component_of[index] == district {
                    seen[index] = true;
                    toward[index] = Some(current);
                    stack.push(parent);
                }
            }
        }
        let witness = (0..n).find(|&index| index != node.as_usize() && seen[index]);
        if let Some(start) = witness {
            let mut path = Vec::new();
            let mut at = Some(DenseNodeId::from_index(start));
            while let Some(current) = at {
                path.push(variable_of(current)?);
                at = toward[current.as_usize()];
            }
            return Ok(NoiseBlockIdentification::Hedged(MechanismHedge {
                node: variable_of(node)?,
                ancestor: variable_of(DenseNodeId::from_index(start))?,
                path,
            }));
        }
    }

    let mut blocks = Vec::with_capacity(components.len());
    for component in &components {
        let mut members = Vec::new();
        for node in component.iter() {
            members.push(variable_of(node)?);
        }
        members.sort_by_key(|variable| variable.raw());
        blocks.push(members);
    }
    Ok(NoiseBlockIdentification::Identified(NoiseBlockPlan {
        directed: admg.directed_part()?,
        blocks,
    }))
}

/// Group the player nodes of an attribution into the noise blocks that hold them.
///
/// Returns the component of each block (its member with the smallest variable id) and, for
/// each entry of `players_dense`, the position of its block: the coalition bit that switches
/// the entry.
///
/// # Errors
///
/// An ancestor that no block names, or that two blocks name.
pub(crate) fn group_players(
    model: &CompiledCausalModel,
    players_dense: &[DenseNodeId],
    blocks: &[Vec<VariableId>],
) -> Result<(Vec<ComponentId>, Vec<usize>), AttributionError> {
    let mut group_of: Vec<Option<usize>> = vec![None; players_dense.len()];
    let mut components: Vec<ComponentId> = Vec::new();
    for block in blocks {
        let group = components.len();
        let mut representative: Option<VariableId> = None;
        for &variable in block {
            let Some(dense) = model.dense_of(variable) else { continue };
            // Both slices have one entry per player. Borrow the matching slot
            // directly so its player association never becomes an unchecked index.
            let Some((_, slot)) =
                players_dense.iter().zip(&mut group_of).find(|(player, _)| **player == dense)
            else {
                continue;
            };
            match *slot {
                Some(held) if held == group => continue,
                Some(_) => {
                    return Err(AttributionError::invalid_input(
                        "noise blocks overlap: a mechanism belongs to two blocks",
                    ));
                }
                None => *slot = Some(group),
            }
            representative = Some(match representative {
                Some(current) if current.raw() <= variable.raw() => current,
                _ => variable,
            });
        }
        if let Some(variable) = representative {
            components.push(ComponentId::from_variable(variable));
        }
    }
    let mut group_bit = Vec::with_capacity(group_of.len());
    for held in group_of {
        group_bit.push(held.ok_or(AttributionError::invalid_input(
            "an ancestor of the target belongs to no noise block",
        ))?);
    }
    Ok((components, group_bit))
}

/// Distribution-change attribution whose Shapley players are noise blocks.
///
/// Mechanisms are fitted on each population (`refit` reweights the rows under which they are
/// fitted, with the families held at the ones selected on the unweighted populations), and each
/// population's noise is abducted row by row under its own fitted mechanisms. A coalition
/// takes, for each block it holds, the comparison population's mechanisms and its noise
/// vectors, and the baseline's for every other block. A noise vector is drawn jointly for all
/// the members of a block, from one row of the source population, and independently across
/// blocks, so the dependence a latent common cause puts between the noise terms of a block is
/// kept and the independence between blocks is the one the graph asserts. Every coalition uses
/// the same row draws (common random numbers).
///
/// `blocks` must come from an identified [`plan_noise_blocks`]: without identification the
/// residuals are not the noise.
///
/// # Errors
///
/// As [`crate::distribution_change`]; an ancestor of the outcome that no block holds; allocation
/// by path products, which needs linear edges of a single graph.
#[allow(
    clippy::too_many_lines,
    reason = "one pass: fit both populations, abduct their noise, draw the rows once, allocate"
)]
pub fn distribution_change_blocks(
    graph_model: &CompiledCausalModel,
    data: &TabularData,
    query: &ChangeAttributionQuery,
    options: &DistributionChangeOptions,
    blocks: &[Vec<VariableId>],
    refit: Option<&PopulationRefit<'_>>,
    ctx: &ExecutionContext,
) -> Result<ChangeAttributionResult, AttributionError> {
    crate::change_common::validate_change_sample_count(options.measure, options.n_samples)?;
    validate_distribution_change_query(query)?;
    let (baseline_data, comparison_data) = resolve_change_populations(data, query)?;
    let (baseline_store, comparison_store) = if let Some(refit) = refit {
        refit_populations(graph_model, &baseline_data, &comparison_data, refit, ctx)?
    } else {
        let registry = MechanismRegistry::standard();
        let (baseline, _) =
            registry.assign_and_fit(graph_model, &baseline_data, SelectionPolicy::BestScore)?;
        let (comparison, _) =
            registry.assign_and_fit(graph_model, &comparison_data, SelectionPolicy::BestScore)?;
        (baseline, comparison)
    };
    let abduct = |store: &CompiledMechanismStore,
                  population: &TabularData|
     -> Result<(Arc<[f64]>, usize), AttributionError> {
        let model = graph_model
            .clone()
            .with_mechanisms(CompiledMechanismStore { slots: Arc::clone(&store.slots) });
        let exogenous = CounterfactualEngine::from_ref(&model).abduct(
            population,
            AbductionMissingPolicy::Error,
            ctx,
        )?;
        Ok((Arc::clone(&exogenous.noise), exogenous.n_units))
    };
    let (baseline_noise, baseline_units) = abduct(&baseline_store, &baseline_data)?;
    let (comparison_noise, comparison_units) = abduct(&comparison_store, &comparison_data)?;

    let outcome = resolve_outcome_dense(graph_model, query.outcome)?;
    let (mechanism_players, _) =
        joint_players(graph_model, outcome, query.max_components, query.components)?;
    let players_dense: Vec<DenseNodeId> =
        mechanism_players.iter().filter_map(|c| graph_model.dense_of(c.variable())).collect();
    if players_dense.is_empty() {
        return Err(AttributionError::invalid_input("no components to attribute"));
    }
    let (components, group_bit) = group_players(graph_model, &players_dense, blocks)?;
    crate::shapley::check_coalition_sample_budget(
        components.len(),
        &query.allocation,
        options.n_samples,
    )?;

    let n_nodes = graph_model.n_nodes();
    let n_samples = options.n_samples;
    let mut is_player = vec![false; n_nodes];
    for node in &players_dense {
        is_player[node.as_usize()] = true;
    }
    let (eval_order, eval_gathers): (Vec<DenseNodeId>, Vec<ParentGatherPlan>) = graph_model
        .node_order
        .iter()
        .zip(graph_model.parent_gathers.iter())
        .filter(|(node, _)| is_player[node.as_usize()])
        .map(|(&node, gather)| {
            (node, ParentGatherPlan { child: gather.child, parents: Arc::clone(&gather.parents) })
        })
        .unzip();

    // One row of each population per block per draw, shared by every coalition. Under a refit the
    // rows are drawn in proportion to the weights the mechanisms were fitted under (the Bayesian
    // bootstrap weights, or the counts of a resample), so the sampled population is the weighted
    // one and a row of weight zero is never drawn; unweighted draws would reproduce the
    // unweighted population whatever the weights and collapse the spread of a posterior.
    let mut rng = attribution_stream(ctx, DISTRIBUTION_STREAM, options.seed);
    let mut draw_rows = |units: usize, weights: Option<&[f64]>| -> Vec<usize> {
        let cumulative: Option<Vec<f64>> = weights.map(|weights| {
            let mut total = 0.0;
            weights
                .iter()
                .map(|weight| {
                    total += weight.max(0.0);
                    total
                })
                .collect()
        });
        (0..components.len() * n_samples)
            .map(|_| {
                let u = rng.next_f64();
                match cumulative.as_deref() {
                    Some(cumulative) if cumulative.last().is_some_and(|total| *total > 0.0) => {
                        let target = u * cumulative[cumulative.len() - 1];
                        cumulative.partition_point(|held| *held <= target).min(units - 1)
                    }
                    _ => row_below(u, units),
                }
            })
            .collect()
    };
    let baseline_rows = draw_rows(baseline_units, refit.map(|held| held.baseline()));
    let comparison_rows = draw_rows(comparison_units, refit.map(|held| held.comparison()));

    let mut payoff = BlockSwapPayoff {
        slots: baseline_store.slots.to_vec(),
        baseline: baseline_store,
        comparison: comparison_store,
        scratch: Coalition::empty(),
        players_dense,
        group_bit,
        baseline_noise,
        baseline_units,
        comparison_noise,
        comparison_units,
        baseline_rows,
        comparison_rows,
        eval_order,
        eval_gathers,
        outcome,
        n_nodes,
        n_samples,
        measure: options.measure,
        baseline_law: None,
        law_memo: crate::change_common::LawMemo::default(),
        noise_buf: vec![0.0; n_nodes * n_samples],
        values_buf: vec![0.0; n_nodes * n_samples],
        ws: MechanismWorkspace::default(),
    };
    let v0 = payoff.value(&Coalition::empty())?;
    let v_full = payoff.value(&full_coalition_mask(components.len())?)?;
    let total = total_change(options.measure, v0, v_full);
    let request = ChangeAllocation {
        outcome: query.outcome,
        players: &components,
        allocation: &query.allocation,
        model_for_paths: None,
    };
    run_change_allocation(&request, &mut payoff, total, Arc::from([]), ctx)
}

/// The row a uniform draw `u` in `[0, 1)` selects among `units` rows.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "a uniform in [0, 1) times a row count floors to a row index below that count"
)]
fn row_below(u: f64, units: usize) -> usize {
    ((u * units as f64) as usize).min(units.saturating_sub(1))
}

/// The coalition payoff of a block change attribution: the outcome summary of the hybrid
/// system that takes the comparison mechanisms and noise rows for the blocks of the coalition.
struct BlockSwapPayoff {
    /// Mechanism slots of the hybrid at `scratch`.
    slots: Vec<MechanismSlot>,
    baseline: CompiledMechanismStore,
    comparison: CompiledMechanismStore,
    scratch: Coalition,
    players_dense: Vec<DenseNodeId>,
    group_bit: Vec<usize>,
    /// Abducted noise of each population, `noise[node * units + unit]`.
    baseline_noise: Arc<[f64]>,
    baseline_units: usize,
    comparison_noise: Arc<[f64]>,
    comparison_units: usize,
    /// The row each draw takes from each population for each block: `[block * n_samples + draw]`.
    baseline_rows: Vec<usize>,
    comparison_rows: Vec<usize>,
    eval_order: Vec<DenseNodeId>,
    eval_gathers: Vec<ParentGatherPlan>,
    outcome: DenseNodeId,
    n_nodes: usize,
    n_samples: usize,
    measure: DifferenceMeasure,
    baseline_law: Option<(f64, f64)>,
    /// Outcome laws computed so far, by coalition.
    law_memo: crate::change_common::LawMemo,
    noise_buf: Vec<f64>,
    values_buf: Vec<f64>,
    ws: MechanismWorkspace,
}

impl CachedOutcomeLawPayoff for BlockSwapPayoff {
    fn law_memo(&mut self) -> &mut crate::change_common::LawMemo {
        &mut self.law_memo
    }

    fn measure(&self) -> DifferenceMeasure {
        self.measure
    }

    fn baseline_law(&self) -> Option<(f64, f64)> {
        self.baseline_law
    }

    fn set_baseline_law(&mut self, law: (f64, f64)) {
        self.baseline_law = Some(law);
    }

    fn law_at(&mut self, coalition: &Coalition) -> Result<(f64, f64), AttributionError> {
        let n = self.n_samples;
        for (position, &node) in self.players_dense.iter().enumerate() {
            let group = self.group_bit[position];
            let from_comparison = coalition.contains(group);
            let index = node.as_usize();
            if from_comparison != self.scratch.contains(group) {
                let source = if from_comparison { &self.comparison } else { &self.baseline };
                self.slots[index] = source.slots[index].clone();
            }
            let (noise, units, rows) = if from_comparison {
                (&self.comparison_noise, self.comparison_units, &self.comparison_rows)
            } else {
                (&self.baseline_noise, self.baseline_units, &self.baseline_rows)
            };
            for draw in 0..n {
                let row = rows[group * n + draw];
                self.noise_buf[index * n + draw] = noise[index * units + row];
            }
        }
        self.scratch = coalition.clone();
        self.values_buf.fill(0.0);
        let noise = NoiseBatchMut::new(n, self.n_nodes, &mut self.noise_buf)?;
        let mut values = ValueBatchMut::new(n, self.n_nodes, &mut self.values_buf)?;
        evaluate_batch_topo(
            &self.eval_order,
            &self.eval_gathers,
            &self.slots,
            &noise,
            &mut values,
            &mut self.ws,
        )?;
        let start = self.outcome.as_usize() * n;
        let (mean, variance) = antecedent_stats::mean_var(&self.values_buf[start..start + n]);
        Ok((mean, variance))
    }
}

impl CoalitionPayoff for BlockSwapPayoff {
    fn value(&mut self, coalition: &Coalition) -> Result<f64, AttributionError> {
        self.cached_payoff_value(coalition)
    }

    fn fork<'a>(&'a self) -> Option<Box<dyn CoalitionPayoff + Send + 'a>> {
        // The outcome law is a pure function of the coalition: every player's noise rows are
        // rewritten from it on each call and the slot array is patched to match it.
        Some(Box::new(Self {
            slots: self.slots.clone(),
            baseline: self.baseline.clone(),
            comparison: self.comparison.clone(),
            scratch: self.scratch.clone(),
            players_dense: self.players_dense.clone(),
            group_bit: self.group_bit.clone(),
            baseline_noise: Arc::clone(&self.baseline_noise),
            baseline_units: self.baseline_units,
            comparison_noise: Arc::clone(&self.comparison_noise),
            comparison_units: self.comparison_units,
            baseline_rows: self.baseline_rows.clone(),
            comparison_rows: self.comparison_rows.clone(),
            eval_order: self.eval_order.clone(),
            eval_gathers: self.eval_gathers.clone(),
            outcome: self.outcome,
            n_nodes: self.n_nodes,
            n_samples: self.n_samples,
            measure: self.measure,
            baseline_law: self.baseline_law,
            law_memo: crate::change_common::LawMemo::default(),
            noise_buf: self.noise_buf.clone(),
            values_buf: vec![0.0; self.n_nodes * self.n_samples],
            ws: MechanismWorkspace::default(),
        }))
    }
}
