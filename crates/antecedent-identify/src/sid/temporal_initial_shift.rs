//! Checked source observational ID composed with a root initial-law shift.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{SidLimits, temporal_sequence::TemporalSequenceSpec};
use crate::{IdIdentifier, IdentificationError, IdentificationResult, IdentificationWorkspace};
use antecedent_core::{
    CausalQuery, EvidenceCatalog, ExecutionContext, FactorNeed, Intervention,
    InterventionalDistributionQuery, RegimeId, Value, VariableId,
};
use antecedent_expr::{CausalExprArena, DomainRef, ExprId, ExprNode};
use antecedent_graph::{Admg, DenseNodeId};
use std::sync::Arc;

/// Every nonbaseline DAG mechanism is invariant under this frozen sufficient rule.
pub const INITIAL_SHIFT_SOURCE_RULE: &str = "dag_root_initial_shift_v1";
/// Actual source observational ID and its population-bound executable program.
#[derive(Clone, Debug)]
pub struct CheckedInitialShiftSource {
    graph: Admg,
    result: IdentificationResult,
    arena: CausalExprArena,
    root: ExprId,
}
impl CheckedInitialShiftSource {
    /// Actual source causal graph supplied to native observational ID.
    #[must_use]
    pub const fn graph(&self) -> &Admg {
        &self.graph
    }
    /// Actual native identification, including its source expression and derivation trace.
    #[must_use]
    pub const fn result(&self) -> &IdentificationResult {
        &self.result
    }
    /// Source-population bound executable arena.
    #[must_use]
    pub const fn arena(&self) -> &CausalExprArena {
        &self.arena
    }
    /// Executable joint-distribution root.
    #[must_use]
    pub const fn root(&self) -> ExprId {
        self.root
    }
}
/// Bind the unchanged single-study ID expression to its real observational source.
/// This changes provider labels, never distributions, interventions or algebra.
fn bind_source(
    result: &IdentificationResult,
) -> Result<(CausalExprArena, ExprId), IdentificationError> {
    let mut arena = result.arena.clone();
    let population = arena.intern_population("source");
    let mut mapped = Vec::with_capacity(result.arena.len());
    for index in 0..result.arena.len() {
        let id = ExprId::from_raw(
            u32::try_from(index)
                .map_err(|_| IdentificationError::unsupported("initial shift expression limit"))?,
        );
        let node = match result.arena.node(id).clone() {
            ExprNode::Distribution {
                variables,
                conditioned_on,
                intervention,
                domain,
                population: old,
                regime,
            } => {
                if domain != DomainRef::Observational
                    || !result.arena.intervention_assignments(intervention).is_empty()
                    || !result.arena.population(old).is_empty()
                    || regime.is_some()
                {
                    return Err(IdentificationError::unsupported(
                        "temporal_initial_shift.nonobservational_source_factor",
                    ));
                }
                ExprNode::Distribution {
                    variables,
                    conditioned_on,
                    intervention,
                    domain,
                    population,
                    regime: Some(RegimeId::from_raw(0)),
                }
            }
            ExprNode::Kernel { body, bound, .. } => ExprNode::Kernel {
                body: mapped[body.raw() as usize],
                bound,
                population,
                regime: None,
            },
            ExprNode::Product(list) => {
                let list = arena.intern_list(
                    result.arena.list(list).iter().map(|child| mapped[child.raw() as usize]),
                );
                ExprNode::Product(list)
            }
            ExprNode::SumOut { variables, expr } => {
                ExprNode::SumOut { variables, expr: mapped[expr.raw() as usize] }
            }
            ExprNode::IntegralOut { variables, expr } => {
                ExprNode::IntegralOut { variables, expr: mapped[expr.raw() as usize] }
            }
            ExprNode::Ratio { numerator, denominator } => ExprNode::Ratio {
                numerator: mapped[numerator.raw() as usize],
                denominator: mapped[denominator.raw() as usize],
            },
            _ => {
                return Err(IdentificationError::unsupported(
                    "temporal_initial_shift.non_distribution_program",
                ));
            }
        };
        mapped.push(arena.intern(node));
    }
    Ok((arena, mapped[result.estimands[0].functional.raw() as usize]))
}
/// Identify source joint (S0,Y)|do(A1,...,AH) before initial-law standardization.
///
/// The coordinates are numbered in time order: the root state is `0`, the
/// action of step `k` is `2k - 1`, the covariate before the action of step
/// `k > 1` is `2k - 2` and the outcome is `2H`.
/// Root S0 is unconfounded and every other DAG mechanism is invariant. Therefore
/// conditioning the checked source intervention law on supported S0 and mixing
/// the given target S0 law transports its response, without experimental evidence.
/// # Errors
/// Unsupported selection/latent scope, missing observational source joint, failed
/// source ID, cancellation or bounded source-program exhaustion.
pub fn identify_initial_shift_source(
    spec: &TemporalSequenceSpec,
    catalog: &EvidenceCatalog,
    limits: SidLimits,
    ctx: &ExecutionContext,
) -> Result<CheckedInitialShiftSource, IdentificationError> {
    if ctx.cancellation.is_cancelled() {
        return Err(IdentificationError::Cancelled);
    }
    let graph = spec.diagram().causal_graph();
    let root = VariableId::from_raw(0);
    let steps = spec.horizon();
    let scope = || IdentificationError::unsupported("temporal_initial_shift.source_scope");
    let coordinate =
        |index: usize| u32::try_from(index).map(VariableId::from_raw).map_err(|_| scope());
    let actions = (0..steps).map(|step| coordinate(2 * step + 1)).collect::<Result<Vec<_>, _>>()?;
    let outcome = coordinate(2 * steps)?;
    if graph.node_count() != 2 * steps + 1
        || graph.has_bidirected()
        || spec.diagram().selection_targets() != [root]
        || !graph.parents(DenseNodeId::from_raw(0)).is_empty()
        || spec.slots().baseline != [root]
        || spec.slots().actions != actions
        || spec.slots().outcome != outcome
    {
        return Err(scope());
    }
    let variables = (0..=2 * steps).map(coordinate).collect::<Result<Vec<_>, _>>()?;
    let regime = catalog
        .satisfying_regime(&FactorNeed {
            population: "source",
            variables: &variables,
            conditioned_on: &[],
            interventions: &[],
        })
        .ok_or_else(|| {
            IdentificationError::missing_evidence(
                "temporal_initial_shift.source_joint_missing",
                "actual measured source observational joint is required",
            )
        })?;
    if regime.id != RegimeId::from_raw(0) {
        return Err(IdentificationError::invalid_catalog("temporal_initial_shift.source_regime"));
    }
    let identifier = IdIdentifier::new();
    let prepared = identifier.prepare(graph)?;
    let query = CausalQuery::Distribution(
        InterventionalDistributionQuery::new(
            outcome,
            actions
                .iter()
                .map(|action| Intervention::set(*action, Value::f64(0.)))
                .collect::<Arc<[_]>>(),
        )
        .with_outcomes(Arc::from([root, outcome])),
    );
    let result = identifier.identify(&prepared, &query, &mut IdentificationWorkspace::default())?;
    if ctx.cancellation.is_cancelled() {
        return Err(IdentificationError::Cancelled);
    }
    if result.status != antecedent_core::IdentificationStatus::NonparametricallyIdentified
        || result.estimands.len() != 1
    {
        return Err(IdentificationError::unsupported("temporal_initial_shift.source_id"));
    }
    if result.arena.len() > limits.steps || limits.depth < graph.node_count() {
        return Err(IdentificationError::Budget { budget: crate::IdentificationBudget::Steps });
    }
    let (arena, root) = bind_source(&result)?;
    Ok(CheckedInitialShiftSource { graph: graph.clone(), result, arena, root })
}
