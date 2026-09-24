//! Portable classical transport proofs. Decoding never reruns identification.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
use crate::{
    IoError,
    expr_wire::{ExprArenaWire, expr_arena_from_wire, expr_arena_to_wire},
};
use antecedent_core::ExecutionContext;
use antecedent_core::{DistributionAvailability, EvidenceCatalog, RegimeId, VariableId};
use antecedent_expr::{CausalExprArena, ExprId, ExprNode};
use antecedent_graph::SelectionDiagram;
use antecedent_identify::sid::SidDerivationRecord;
use antecedent_identify::{ClassicalTransportDerivation, ClassicalTransportQuery, SidLimits};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// One locally checked proof step and its dependencies.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransportProofStepView {
    /// Stable ordinal within the checked derivation.
    pub index: usize,
    /// The checked theorem rule.
    pub rule: String,
    /// Earlier steps this rule uses.
    pub children: Vec<usize>,
    /// Required factor leaves reachable from this step's output.
    pub factor_nodes: Vec<u32>,
}

/// A source-specific law required by the checked expression.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransportFactorView {
    /// Expression-node identity within this proof.
    pub expression_node: u32,
    /// Population supplying the factor.
    pub population: String,
    /// Jointly required response variables.
    pub variables: Vec<VariableId>,
    /// Required conditioning coordinates.
    pub conditioned_on: Vec<VariableId>,
    /// Exact intervention set; separate regimes cannot be pooled.
    pub interventions: Vec<VariableId>,
    /// Regime supplying the factor, if available.
    pub supplied_by: Option<RegimeId>,
    /// Bound dataset snapshot, when this regime has one.
    pub snapshot_identity: Option<String>,
    /// Exact reason this factor cannot bind, if any.
    pub binding_failure: Option<String>,
}

/// Inspection of a checked theorem proof against one supplied evidence catalog.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransportProofView {
    /// Root expression node produced by the checked theorem derivation.
    pub root_expression_node: u32,
    /// Reachable checked rule graph.
    pub steps: Vec<TransportProofStepView>,
    /// Required factor leaves, including those that fail to bind.
    pub factors: Vec<TransportFactorView>,
}

fn collect_factors(
    arena: &CausalExprArena,
    id: ExprId,
    catalog: &EvidenceCatalog,
    target: &str,
    seen: &mut BTreeSet<u32>,
    factors: &mut Vec<TransportFactorView>,
) {
    if !seen.insert(id.raw()) {
        return;
    }
    match arena.node(id) {
        ExprNode::Distribution { variables, conditioned_on, intervention, population, .. } => {
            let name = arena.population(*population);
            let vars = arena.var_set(*variables);
            let conditions = arena.var_set(*conditioned_on);
            let interventions = arena.intervention_set(*intervention);
            let regime = catalog
                .regimes
                .iter()
                .filter(|regime| {
                    regime.population.as_ref() == name
                        && regime.evidence_kind.can_satisfy_factor()
                        && regime.intervention_values.is_empty()
                        && regime.conditioned_on.is_empty()
                        && matches!(regime.distribution, DistributionAvailability::Joint)
                        && regime.interventions.len() == interventions.len()
                        && regime.interventions.iter().all(|v| interventions.contains(v))
                        && vars.iter().chain(conditions).all(|v| regime.measured.contains(v))
                })
                .min_by_key(|regime| regime.id.raw());
            let target_sampling_failure = name == target
                && catalog
                    .target_sampling
                    .is_some_and(|sampling| !sampling.represents_target_law());
            let supplied_by = if target_sampling_failure { None } else { regime.map(|r| r.id) };
            let closest = catalog
                .regimes
                .iter()
                .filter(|candidate| {
                    candidate.population.as_ref() == name
                        && candidate.interventions.len() == interventions.len()
                        && candidate.interventions.iter().all(|v| interventions.contains(v))
                })
                .min_by_key(|candidate| candidate.id.raw());
            let binding_failure = if target_sampling_failure {
                Some("target sampling does not represent the target population".into())
            } else if regime.is_none() {
                let detail = match closest {
                    None => "matching population and intervention regime absent",
                    Some(candidate) if !candidate.evidence_kind.can_satisfy_factor() => {
                        "matching regime is only manipulable or proposed"
                    }
                    Some(candidate) if !candidate.intervention_values.is_empty() => {
                        "matching regime is limited to concrete intervention values"
                    }
                    Some(candidate) if !candidate.conditioned_on.is_empty() => {
                        "matching regime was already conditioned"
                    }
                    Some(candidate)
                        if !matches!(candidate.distribution, DistributionAvailability::Joint) =>
                    {
                        "matching regime supplies separate marginals, not the required joint law"
                    }
                    Some(_) => "matching regime does not measure every required coordinate",
                };
                Some(format!(
                    "{detail}: {name} law over {vars:?} given {conditions:?} under do({interventions:?})"
                ))
            } else {
                None
            };
            factors.push(TransportFactorView {
                expression_node: id.raw(),
                population: name.into(),
                variables: vars.to_vec(),
                conditioned_on: conditions.to_vec(),
                interventions,
                supplied_by,
                snapshot_identity: supplied_by.and_then(|id| {
                    catalog
                        .bindings
                        .iter()
                        .find(|binding| binding.regime == id)
                        .map(|binding| binding.snapshot_identity.to_string())
                }),
                binding_failure,
            });
        }
        ExprNode::Kernel { body, .. }
        | ExprNode::SumOut { expr: body, .. }
        | ExprNode::IntegralOut { expr: body, .. } => {
            collect_factors(arena, *body, catalog, target, seen, factors);
        }
        ExprNode::Expectation { distribution, .. } => {
            collect_factors(arena, *distribution, catalog, target, seen, factors);
        }
        ExprNode::Product(list) => {
            for child in arena.list(*list) {
                collect_factors(arena, *child, catalog, target, seen, factors);
            }
        }
        ExprNode::Ratio { numerator, denominator } => {
            collect_factors(arena, *numerator, catalog, target, seen, factors);
            collect_factors(arena, *denominator, catalog, target, seen, factors);
        }
        ExprNode::Contrast { left, right, .. } => {
            collect_factors(arena, *left, catalog, target, seen, factors);
            collect_factors(arena, *right, catalog, target, seen, factors);
        }
    }
}

/// Untrusted expression plus typed local premises. Successful deserialization
/// alone is not verification; consumers must call `check` against their inputs.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransportProofWire {
    /// Versioned theoretical evidence scope, query and rule premises.
    pub proof: SidDerivationRecord,
    /// Original certified functional, never the optimized physical plan.
    pub expression: ExprArenaWire,
}
impl TransportProofWire {
    /// Verify the proof and inspect its rule graph and source-specific factor
    /// obligations against a catalog. Inspection never turns proposed studies
    /// into available evidence.
    ///
    /// # Errors
    /// Invalid proof, diagram, query, catalog, or exhausted checking budget.
    pub fn inspect(
        &self,
        diagram: &SelectionDiagram,
        query: &ClassicalTransportQuery,
        catalog: &EvidenceCatalog,
        limits: SidLimits,
        ctx: &ExecutionContext,
    ) -> Result<TransportProofView, IoError> {
        catalog.validate().map_err(|error| IoError::Convert(error.to_string()))?;
        let checked = self.check(diagram, query, limits, ctx)?;
        let mut reachable = BTreeSet::new();
        let mut pending = vec![self.proof.root_step];
        while let Some(index) = pending.pop() {
            if reachable.insert(index) {
                pending.extend(self.proof.steps[index].children.iter().copied());
            }
        }
        let steps = reachable
            .into_iter()
            .map(|index| {
                let step = &self.proof.steps[index];
                let mut step_seen = BTreeSet::new();
                let mut step_factors = Vec::new();
                collect_factors(
                    checked.arena(),
                    ExprId::from_raw(step.output),
                    catalog,
                    &query.target,
                    &mut step_seen,
                    &mut step_factors,
                );
                let mut factor_nodes = step_factors
                    .into_iter()
                    .map(|factor| factor.expression_node)
                    .collect::<Vec<_>>();
                factor_nodes.sort_unstable();
                TransportProofStepView {
                    index,
                    rule: format!("{:?}", step.rule),
                    children: step.children.clone(),
                    factor_nodes,
                }
            })
            .collect();
        let mut seen = BTreeSet::new();
        let mut factors = Vec::new();
        collect_factors(
            checked.arena(),
            checked.root(),
            catalog,
            &query.target,
            &mut seen,
            &mut factors,
        );
        factors.sort_by_key(|factor| factor.expression_node);
        Ok(TransportProofView { root_expression_node: checked.root().raw(), steps, factors })
    }
    /// Encode a native checked proof.
    ///
    /// # Errors
    /// Expression exceeds wire capacity.
    pub fn from_checked(proof: &ClassicalTransportDerivation) -> Result<Self, IoError> {
        Ok(Self { proof: proof.to_record(), expression: expr_arena_to_wire(proof.arena())? })
    }
    /// Independently verify every local premise, expression and input identity.
    /// Does not fetch data, fit models, or rerun identification.
    ///
    /// # Errors
    /// Malformed expression, substituted query or invalid derivation.
    pub fn check(
        &self,
        diagram: &SelectionDiagram,
        query: &ClassicalTransportQuery,
        limits: SidLimits,
        ctx: &ExecutionContext,
    ) -> Result<ClassicalTransportDerivation, IoError> {
        if self.proof.steps.len() > limits.steps || ctx.cancellation.is_cancelled() {
            return Err(IoError::Convert("transport proof budget/cancellation".into()));
        }
        let bytes = self
            .expression
            .nodes
            .len()
            .saturating_mul(128)
            .saturating_add(
                self.expression
                    .var_sets
                    .iter()
                    .map(|v| v.len().saturating_mul(8))
                    .fold(0usize, usize::saturating_add),
            )
            .saturating_add(
                self.expression
                    .lists
                    .iter()
                    .map(|v| v.len().saturating_mul(8))
                    .fold(0usize, usize::saturating_add),
            )
            .saturating_add(
                self.expression
                    .interventions
                    .iter()
                    .map(|v| v.len().saturating_mul(64))
                    .fold(0usize, usize::saturating_add),
            );
        if ctx
            .memory
            .hard_limit_bytes
            .is_some_and(|limit| u64::try_from(bytes).map_or(true, |bytes| bytes > limit))
        {
            return Err(IoError::Convert("transport proof memory budget".into()));
        }
        ClassicalTransportDerivation::from_record_checked(
            self.proof.clone(),
            expr_arena_from_wire(&self.expression)?,
            diagram,
            query,
            limits,
            ctx,
        )
        .map_err(|e| IoError::Convert(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{DistributionAvailability, EvidenceKind, EvidenceRegime, RegimeKind};
    use antecedent_graph::{Admg, DenseNodeId};
    use antecedent_identify::{ClassicalTransportResult, identify_classical_transport};
    use std::sync::Arc;

    #[test]
    fn inspection_names_missing_and_bound_source_laws() {
        let mut graph = Admg::with_variables(2);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let diagram = SelectionDiagram::try_new(graph, []).unwrap();
        let query = ClassicalTransportQuery {
            outcomes: Arc::from([VariableId::from_raw(1)]),
            treatments: Arc::from([VariableId::from_raw(0)]),
            source: Arc::from("source"),
            target: Arc::from("target"),
        };
        let ctx = ExecutionContext::for_tests(2);
        let limits = SidLimits::default();
        let ClassicalTransportResult::Identified(proof) =
            identify_classical_transport(&diagram, &query, limits, &ctx).unwrap()
        else {
            panic!("simple DAG must be identified");
        };
        let wire = TransportProofWire::from_checked(&proof).unwrap();
        let empty = EvidenceCatalog::empty();
        let missing = wire.inspect(&diagram, &query, &empty, limits, &ctx).unwrap();
        assert!(!missing.steps.is_empty());
        assert!(missing.factors.iter().any(|factor| factor.binding_failure.is_some()));

        let regimes = ["source", "target"]
            .into_iter()
            .enumerate()
            .map(|(i, population)| {
                EvidenceRegime::try_new(
                    RegimeId::from_raw(i as u32),
                    RegimeKind::Observational,
                    EvidenceKind::Available,
                    [],
                    [],
                    [VariableId::from_raw(0), VariableId::from_raw(1)],
                    population,
                    DistributionAvailability::Joint,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let catalog = EvidenceCatalog::try_new([], regimes, [], None).unwrap();
        let bound = wire.inspect(&diagram, &query, &catalog, limits, &ctx).unwrap();
        assert!(bound.factors.iter().all(|factor| factor.binding_failure.is_none()));
        assert!(bound.factors.iter().all(|factor| factor.supplied_by.is_some()));
        let mut margins = catalog.regimes.to_vec();
        for regime in &mut margins {
            regime.distribution = DistributionAvailability::SeparateMarginals {
                variables: Arc::from([VariableId::from_raw(0), VariableId::from_raw(1)]),
            };
        }
        let margins = EvidenceCatalog::try_new([], margins, [], None).unwrap();
        let refused = wire.inspect(&diagram, &query, &margins, limits, &ctx).unwrap();
        assert!(refused.factors.iter().any(|factor| {
            factor
                .binding_failure
                .as_deref()
                .is_some_and(|reason| reason.contains("separate marginals"))
        }));
        assert!(empty.regimes.is_empty());
    }
}
