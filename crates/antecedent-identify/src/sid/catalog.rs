//! Catalog-aware alternative search for classical transport.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, BitSet, BoundTransportFunctional, CausalExprArena, ClassicalTransportDerivation,
    ClassicalTransportQuery, DenseNodeId, Engine, ExecutionContext, ExprId, ExprNode,
    GraphWorkspace, IdentificationError, MutilatedSelection, ProofStep, RefCell, Rule,
    SelectionDiagram, SidLimits, SidMeter, State, SubsetSearchEnd, VariableId, difference,
    for_each_admissible_subset, graph_signature, standardization_binds, stop_error,
};

/// Result of explicitly bounded catalog-aware alternative search.
#[derive(Clone, Debug)]
pub enum CatalogTransportResult {
    /// A checked derivation with available supplied-catalog leaves.
    Identified(Box<BoundTransportFunctional>),
    /// The searched strategies did not bind. This is not an impossibility proof.
    MissingEvidence {
        /// Strategies attempted, in deterministic order.
        searched: Arc<[Arc<str>]>,
        /// Per-strategy unmet obligations; no claim to exhaustive formula search.
        obligations: Arc<[Arc<str>]>,
    },
    /// Neither bounded strategy produced a derivation; no catalog impossibility claim.
    NotCertified {
        /// Strategies attempted, in deterministic order.
        searched: Arc<[Arc<str>]>,
        /// Scope notes for each attempted strategy.
        obligations: Arc<[Arc<str>]>,
    },
}

impl CatalogTransportResult {
    /// Identification status in the shared transport vocabulary
    /// ([`TransportOutcomeKind::IDENTIFICATION`](antecedent_core::TransportOutcomeKind::IDENTIFICATION)).
    #[must_use]
    pub const fn identification_status(&self) -> antecedent_core::TransportOutcomeKind {
        match self {
            Self::Identified(_) => antecedent_core::TransportOutcomeKind::Identified,
            Self::MissingEvidence { .. } => antecedent_core::TransportOutcomeKind::MissingEvidence,
            Self::NotCertified { .. } => antecedent_core::TransportOutcomeKind::NotCertified,
        }
    }
}

/// Search stages of catalog-aware transport identification, in the order they are tried.
const SEARCH_STAGES: [&str; 3] =
    ["target_first_sid", "pretreatment_standardization", "catalog_available_source_first_sid"];

/// The first `count` search stages, as recorded on a result.
fn searched_stages(count: usize) -> Arc<[Arc<str>]> {
    SEARCH_STAGES.iter().take(count).map(|&stage| Arc::from(stage)).collect()
}

/// Pretreatment candidates: non-descendants of the treatments among the remaining nodes.
fn pretreatment_candidates(
    diagram: &SelectionDiagram,
    state: &State,
    x_nodes: &[DenseNodeId],
) -> Vec<DenseNodeId> {
    let mut downstream = BitSet::with_len(diagram.causal_graph().node_count());
    diagram.causal_graph().descendants_of(x_nodes, &mut downstream, &mut GraphWorkspace::default());
    difference(&difference(&state.v, &state.x), &state.y)
        .iter()
        .filter(|z| !downstream.contains(*z))
        .collect()
}

/// The obligation recorded when the pretreatment-standardization search finds no derivation.
fn pretreatment_obligation(end: SubsetSearchEnd, steps: usize, n_candidates: usize) -> String {
    if end == SubsetSearchEnd::Capped {
        format!(
            "pretreatment standardization search stopped after {steps} candidate subsets of {n_candidates} pretreatment covariates; inconclusive"
        )
    } else {
        "searched pretreatment standardizations: required supplied source conditional or target marginal is missing".to_owned()
    }
}

/// One-step derivation that standardizes over `over`, or transports directly when it is empty.
fn standardization_derivation(
    diagram: &SelectionDiagram,
    query: &ClassicalTransportQuery,
    arena: CausalExprArena,
    state: &State,
    output: ExprId,
    over: Vec<VariableId>,
) -> ClassicalTransportDerivation {
    ClassicalTransportDerivation {
        query: query.clone(),
        graph_signature: graph_signature(diagram),
        selection_targets: diagram.selection_targets().into(),
        arena,
        root: output,
        root_step: 0,
        sources: Vec::new(),
        proof: vec![ProofStep {
            state: state.clone(),
            rule: if over.is_empty() { Rule::DirectTransport } else { Rule::Standardize },
            children: Vec::new(),
            output,
            parameters: over,
        }],
    }
}

/// Try target-first sID, then catalog-aware source/target recursive choices.
/// This bounded two-strategy search does not inherit classical completeness.
///
/// # Errors
/// Invalid input, exhausted shared search budget, or cancellation. Never a negative theorem claim.
pub fn identify_catalog_transport(
    diagram: &SelectionDiagram,
    query: &ClassicalTransportQuery,
    catalog: &antecedent_core::EvidenceCatalog,
    limits: SidLimits,
    ctx: &ExecutionContext,
) -> Result<CatalogTransportResult, IdentificationError> {
    identify_catalog_transport_metered(diagram, query, catalog, SidMeter::Limits(limits), ctx)
}

/// [`identify_catalog_transport`] charging `meter`: its own [`SidLimits`], or a
/// shared budget that the recursive searches and every verification replay
/// charge. The pretreatment-standardization subset search keeps its own count
/// of `meter.limits().operations` separation tests (its exhaustion is an
/// obligation); under a shared budget each of those tests is also charged to
/// it, and each derivation it finds is verified under `meter`.
///
/// # Errors
/// As [`identify_catalog_transport`]; under a shared budget, its stop.
#[allow(
    clippy::too_many_lines,
    reason = "one linear pipeline: source pass, then enlargement, then catalog search"
)]
pub(crate) fn identify_catalog_transport_metered(
    diagram: &SelectionDiagram,
    query: &ClassicalTransportQuery,
    catalog: &antecedent_core::EvidenceCatalog,
    meter: SidMeter<'_>,
    ctx: &ExecutionContext,
) -> Result<CatalogTransportResult, IdentificationError> {
    catalog.validate().map_err(|e| IdentificationError::invalid_catalog(e.to_string()))?;
    let limits = meter.limits();
    let mut engine = Engine::new_metered(diagram, query, meter, ctx)?;
    let state = engine.initial()?;
    let mut obligations = Vec::<Arc<str>>::new();
    let mut had_derivation = false;
    let mut result = engine.solve(state.clone(), false, 0)?;
    if result.is_none() {
        result = engine.solve(state.clone(), true, 0)?;
    }
    for strategy in 0..2 {
        if let Some(root_step) = result {
            had_derivation = true;
            let derivation = engine.solved_derivation(root_step, Vec::new())?;
            match derivation.bind_catalog_validated(catalog) {
                Ok(mut bound) => {
                    bound.searched = searched_stages(if strategy == 0 { 1 } else { 3 });
                    return Ok(CatalogTransportResult::Identified(Box::new(bound)));
                }
                Err(error) => obligations.push(Arc::from(error.to_string())),
            }
        } else {
            obligations.push(Arc::from("strategy produced no bindable derivation"));
        }
        if strategy == 0 {
            // Independent sufficient-rule alternatives use only target marginals
            // and source conditionals; they need not bind the target joint law.
            // Pretreatment candidates are the non-descendants of the treatments;
            // subsets are tried smallest first under the search's own budget, so
            // its exhaustion is an obligation, never a failure of the whole call
            // and never a reason to skip the cheaper source-first strategy.
            let x_nodes = state.x.to_dense_ids();
            let candidates = pretreatment_candidates(diagram, &state, &x_nodes);
            let state_outcomes = engine.vars(&state.y)?;
            let state_treatments = engine.vars(&state.x)?;
            let targets = engine.dense_all(diagram.selection_targets())?;
            let selection =
                MutilatedSelection::build(diagram.causal_graph(), &state.v, &state.x, &targets)?;
            let mut found = None;
            // A shared budget is charged by every separation test (`tick`) and by
            // the engine while it standardizes and verifies a subset (`visit`);
            // the two closures hand it back and forth through one cell.
            let (tick_depth, tick_bytes) =
                (engine.depth_offset.saturating_add(1), engine.live_bytes());
            let shared = RefCell::new(engine.shared.take());
            let end = for_each_admissible_subset(
                &selection,
                &state.y.to_dense_ids(),
                &x_nodes,
                &candidates,
                limits.steps,
                || match shared.borrow_mut().as_deref_mut() {
                    Some(budget) => budget.charge(tick_depth, tick_bytes).map_err(stop_error),
                    None if ctx.cancellation.is_cancelled() => Err(IdentificationError::Cancelled),
                    None => Ok(()),
                },
                |subset| {
                    engine.shared = shared.borrow_mut().take();
                    let visited = (|| {
                        let over = subset
                            .iter()
                            .map(|z| engine.prepared.dense_to_var(*z))
                            .collect::<Result<Vec<_>, _>>()?;
                        // The subset passed the separation test that produced it,
                        // and its two leaves are probed against the catalog before
                        // anything is interned or cloned.
                        if !standardization_binds(
                            catalog,
                            query,
                            &state_outcomes,
                            &state_treatments,
                            &over,
                        ) {
                            return Ok(false);
                        }
                        let Some(output) = engine.standardized(&state, &over, true)? else {
                            return Ok(false);
                        };
                        had_derivation = true;
                        let derivation = standardization_derivation(
                            diagram,
                            query,
                            engine.arena.clone(),
                            &state,
                            output,
                            over,
                        );
                        let mut bound = derivation.bind_catalog_validated(catalog)?;
                        engine.verify(&derivation)?;
                        bound.searched = searched_stages(2);
                        found = Some(bound);
                        Ok(true)
                    })();
                    *shared.borrow_mut() = engine.shared.take();
                    visited
                },
            );
            engine.shared = shared.into_inner();
            let end = end?;
            if let Some(bound) = found {
                return Ok(CatalogTransportResult::Identified(Box::new(bound)));
            }
            obligations.push(Arc::from(pretreatment_obligation(
                end,
                limits.steps,
                candidates.len(),
            )));
            // This cache is scoped to immutable evidence and strategy. Changing
            // the evidence-selection policy invalidates it, not just its root.
            engine.memo.clear();
            engine.source_catalog = Some(catalog);
            result = engine.solve(state.clone(), true, 0)?;
        }
    }
    let searched = searched_stages(3);
    let obligations = obligations.into();
    Ok(if had_derivation {
        CatalogTransportResult::MissingEvidence { searched, obligations }
    } else {
        CatalogTransportResult::NotCertified { searched, obligations }
    })
}

pub(super) fn bind_distribution(
    id: ExprId,
    source: &CausalExprArena,
    catalog: &antecedent_core::EvidenceCatalog,
    target: &str,
) -> Result<ExprNode, IdentificationError> {
    let ExprNode::Distribution {
        variables, conditioned_on, intervention, domain, population, ..
    } = source.node(id).clone()
    else {
        return Err(IdentificationError::InvariantViolated {
            message: "expected a distribution leaf",
        });
    };
    let name = source.population(population);
    let vars = source.var_set(variables);
    let conditions = source.var_set(conditioned_on);
    let interventions = source.intervention_set(intervention);
    if name == target && catalog.target_sampling.is_some_and(|s| !s.represents_target_law()) {
        return Err(IdentificationError::missing_evidence(
            "transport.missing_evidence",
            "target sampling does not represent target law",
        ));
    }
    // One predicate decides regime satisfaction for every transport route:
    // `EvidenceRegime::satisfies`, which the structural identifier also uses.
    let need = antecedent_core::FactorNeed {
        population: name,
        variables: vars,
        conditioned_on: conditions,
        interventions: &interventions,
    };
    let regime = catalog.satisfying_regime(&need).ok_or_else(|| {
        IdentificationError::missing_evidence(
            "transport.missing_evidence",
            format!(
                "this derivation needs joint {name} law over {vars:?} given {conditions:?} under do({interventions:?}); alternative search not exhausted"
            ),
        )
    })?;
    Ok(ExprNode::Distribution {
        variables,
        conditioned_on,
        intervention,
        domain,
        population,
        regime: Some(regime.id),
    })
}
