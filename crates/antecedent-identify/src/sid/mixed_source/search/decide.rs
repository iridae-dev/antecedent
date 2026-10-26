//! Decide items of the parent module.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::super::{Admg, Arc, CatalogTransportResult, ClassicalTransportQuery, ExecutionContext, IdentificationError, MzTransportDecision, SearchBudget, SearchLimits, SearchReceipt, SearchStop, SelectionDiagram, SharedSearch, VariableId, identify_catalog_transport_metered};
use super::super::{EvidenceCatalog, FRONTIER_SHOWN, INVALID_QUERY, MIXED_SOURCE_DEFAULT_LIMITS, MIXED_SOURCE_MAX_DO, MIXED_SOURCE_MAX_MOVE, MIXED_SOURCE_RULE_SET, MetaSource, MetaTransportQuery, MixedInput, MixedQuantity, MixedSearchSummary, MixedSourceDerivation, MixedSourceQuery, MixedStageRecord, RegimeId, ValidatedMixedSourceQuery, ZTransportDecision, ZTransportLimitsReceipt, ZTransportQuery, bounds_exceeded, decide_z_transport_reporting, dense_variables, identify_meta_catalog_metered, invalid_derivation, invalid_query, validate_mixed_source_query};

use super::*;

const STAGE_SID: &str = "target_first_sid";

const STAGE_Z: &str = "z_transport";

const STAGE_MZ: &str = "mz_transport";

const STAGE_META: &str = "meta_transport";

const STAGE_RULES: &str = "rule_search";

/// Whether the classical meta-transport contract applies to `query`: two or more
/// declared sources, each able to experiment on every observed variable (the
/// unrestricted source-experiment family that theorem assumes). A source with a
/// restricted controllable set is the mz route's business, and one source is the
/// z route's; meta is then not tried and never approximated.
pub(crate) fn meta_applies(query: &MixedSourceQuery, variables: &[VariableId]) -> bool {
    query.sources.len() >= 2
        && query
            .sources
            .iter()
            .all(|source| variables.iter().all(|v| source.controllable.contains(v)))
}

pub(crate) fn stage_names(query: &MixedSourceQuery, meta: bool) -> Vec<&'static str> {
    let named = match query.sources.len() {
        0 => None,
        1 => Some(STAGE_Z),
        _ => Some(STAGE_MZ),
    };
    std::iter::once(STAGE_SID)
        .chain(named)
        .chain(meta.then_some(STAGE_META))
        .chain(std::iter::once(STAGE_RULES))
        .collect()
}

/// Decide a bounded mixed-source query.
///
/// Stages run in a fixed order, each only when the earlier ones do not identify:
/// target-first sID over the catalog's target evidence, the declared z (one
/// source) or mz (two or more) route, classical meta-transport when two or more
/// declared sources are unrestricted, then the generic rule search over the
/// target population's distributions and, when it fails, the relaxed pass that
/// names the exact missing joint. One [`SearchBudget`] under `limits` is charged
/// by every step of every stage, so the limits bound the whole decision. A stop
/// anywhere is [`MixedSourceDecision::Exhausted`], never a verdict.
///
/// # Errors
///
/// Limits above [`MIXED_SOURCE_DEFAULT_LIMITS`] or a query outside its bounds
/// (`mixed_search.bounds_exceeded`), a malformed query or catalog
/// (`mixed_search.invalid_query`), a model artifact offered as an experimental
/// law (`mixed_search.posterior_as_law`), or a derivation that fails its own check.
pub fn decide_mixed_source(
    graph: &Admg,
    query: &MixedSourceQuery,
    catalog: &EvidenceCatalog,
    limits: SearchLimits,
    ctx: &ExecutionContext,
) -> Result<MixedSourceDecision, IdentificationError> {
    if limits.operations > MIXED_SOURCE_DEFAULT_LIMITS.operations
        || limits.depth > MIXED_SOURCE_DEFAULT_LIMITS.depth
    {
        return Err(bounds_exceeded());
    }
    let validated = validate_mixed_source_query(graph, query, catalog)?;
    catalog.validate().map_err(|error| {
        IdentificationError::invalid_catalog(format!("{INVALID_QUERY}: {error}"))
    })?;
    match SearchBudget::new(limits, ctx) {
        Ok(budget) => {
            decide_charged(graph, &validated, catalog, &mut SharedSearch::new(budget), ctx)
        }
        Err(receipt) => Ok(MixedSourceDecision::Exhausted(SearchReceipt {
            unevaluated: validated.stage_plan().iter().map(|s| format!("stage:{s}")).collect(),
            ..receipt
        })),
    }
}

/// [`decide_mixed_source`] charging a budget an enclosing plan already shares
/// (the study planner re-decides each preview catalog on it). Validation,
/// stages and outcomes are exactly those of the public route; the caller owns
/// the limits.
///
/// # Errors
///
/// As [`decide_mixed_source`], except the limits.
pub(crate) fn decide_mixed_source_shared(
    graph: &Admg,
    query: &MixedSourceQuery,
    catalog: &EvidenceCatalog,
    search: &mut SharedSearch<'_>,
    ctx: &ExecutionContext,
) -> Result<MixedSourceDecision, IdentificationError> {
    let validated = validate_mixed_source_query(graph, query, catalog)?;
    catalog.validate().map_err(|error| {
        IdentificationError::invalid_catalog(format!("{INVALID_QUERY}: {error}"))
    })?;
    decide_charged(graph, &validated, catalog, search, ctx)
}

/// The receipt of a stop: finished stages and completed generations are explored;
/// the stopped and later stages, then `pending`, are unevaluated.
fn stopped_receipt(
    search: &SharedSearch<'_>,
    stop: SearchStop,
    names: &[&'static str],
    stages: &[MixedStageRecord],
    explored: Vec<String>,
    pending: Vec<String>,
) -> SearchReceipt {
    let explored =
        stages.iter().map(|s| format!("stage:{}", s.stage)).chain(explored).collect::<Vec<_>>();
    let unevaluated = names[stages.len().min(names.len())..]
        .iter()
        .map(|s| format!("stage:{s}"))
        .chain(pending)
        .collect::<Vec<_>>();
    search.receipt(stop, explored, unevaluated)
}

fn named_route_error(error: &IdentificationError) -> IdentificationError {
    invalid_query(&format_args!("a theorem-scoped route refused the query: {error}"))
}

/// Run the theorem-scoped routes. `Some` is a final decision (a route identifies,
/// or the budget stopped); `None` continues to the rule search.
#[allow(
    clippy::too_many_lines,
    reason = "One linear pass over the named stages and their outcomes"
)]
fn named_routes(
    graph: &Admg,
    query: &MixedSourceQuery,
    plan: &[&'static str],
    catalog: &EvidenceCatalog,
    search: &mut SharedSearch<'_>,
    ctx: &ExecutionContext,
    stages: &mut Vec<MixedStageRecord>,
) -> Result<Option<MixedSourceDecision>, IdentificationError> {
    let solved = |stages: &mut Vec<MixedStageRecord>, route: &'static str| {
        stages.push(MixedStageRecord { stage: route.into(), outcome: "identified" });
        Ok(Some(MixedSourceDecision::NamedRoute { route, stages: stages.clone() }))
    };
    let shared = SelectionDiagram::try_new(graph.clone(), Arc::<[VariableId]>::from([]))
        .map_err(|error| invalid_query(&error))?;
    let classical = ClassicalTransportQuery {
        outcomes: Arc::clone(&query.outcomes),
        treatments: Arc::clone(&query.treatments),
        source: Arc::from("<no source>"),
        target: Arc::clone(&query.target),
    };
    match identify_catalog_transport_metered(&shared, &classical, catalog, search.meter(), ctx) {
        Ok(CatalogTransportResult::Identified(_)) => return solved(stages, STAGE_SID),
        Ok(CatalogTransportResult::MissingEvidence { .. }) => {
            stages.push(MixedStageRecord { stage: STAGE_SID.into(), outcome: "missing_evidence" });
        }
        Ok(CatalogTransportResult::NotCertified { .. }) => {
            stages.push(MixedStageRecord { stage: STAGE_SID.into(), outcome: "not_certified" });
        }
        Err(error) if error.is_budget_or_cancel() => {
            let stop = search.stop_of(&error);
            let receipt = stopped_receipt(search, stop, plan, stages, Vec::new(), Vec::new());
            return Ok(Some(MixedSourceDecision::Exhausted(receipt)));
        }
        Err(error) => return Err(named_route_error(&error)),
    }
    match query.sources.len() {
        0 => {}
        1 => {
            let source = &query.sources[0];
            let diagram =
                SelectionDiagram::try_new(graph.clone(), Arc::clone(&source.selection_targets))
                    .map_err(|error| invalid_query(&error))?;
            let single = ZTransportQuery {
                outcomes: Arc::clone(&query.outcomes),
                treatments: Arc::clone(&query.treatments),
                controllable: Arc::clone(&source.controllable),
                experiment_assignment: Arc::clone(&source.experiment_assignment),
                source: Arc::clone(&source.population),
                target: Arc::clone(&query.target),
            };
            let mut stopped: Option<ZTransportLimitsReceipt> = None;
            let outcome = match decide_z_transport_reporting(
                &diagram,
                &single,
                catalog,
                search.meter(),
                ctx,
                &mut stopped,
            ) {
                Ok(ZTransportDecision::Identified(_)) => return solved(stages, STAGE_Z),
                Ok(ZTransportDecision::MissingEvidence { .. }) => "missing_evidence",
                Ok(ZTransportDecision::ProvenNonTransportable(_)) => "obstruction",
                Ok(ZTransportDecision::NotCertified { .. }) => "not_certified",
                Err(error) if error.is_budget_or_cancel() => {
                    let stop = search.stop_of(&error);
                    let explored = stopped
                        .into_iter()
                        .flat_map(|r| r.explored_rules)
                        .map(|r| format!("rule:{r}"))
                        .collect();
                    let receipt = stopped_receipt(search, stop, plan, stages, explored, Vec::new());
                    return Ok(Some(MixedSourceDecision::Exhausted(receipt)));
                }
                Err(error) => return Err(named_route_error(&error)),
            };
            stages.push(MixedStageRecord { stage: STAGE_Z.into(), outcome });
        }
        _ => {
            let outcome = match super::super::super::mz_transport::decide_mz_transport_shared(
                graph,
                &query.mz_query(),
                catalog,
                search,
                ctx,
            ) {
                Ok(MzTransportDecision::Identified { .. }) => return solved(stages, STAGE_MZ),
                Ok(MzTransportDecision::MissingEvidence { .. }) => "missing_evidence",
                Ok(MzTransportDecision::ProvenNonTransportable(_)) => "obstruction",
                Ok(MzTransportDecision::NotCertified(_)) => "not_certified",
                Ok(MzTransportDecision::Exhausted(inner)) => {
                    let receipt = stopped_receipt(
                        search,
                        inner.stop,
                        plan,
                        stages,
                        inner.explored,
                        inner.unevaluated,
                    );
                    return Ok(Some(MixedSourceDecision::Exhausted(receipt)));
                }
                Err(error) if error.is_budget_or_cancel() => {
                    let stop = search.stop_of(&error);
                    let receipt =
                        stopped_receipt(search, stop, plan, stages, Vec::new(), Vec::new());
                    return Ok(Some(MixedSourceDecision::Exhausted(receipt)));
                }
                Err(error) => return Err(named_route_error(&error)),
            };
            stages.push(MixedStageRecord { stage: STAGE_MZ.into(), outcome });
        }
    }
    if plan.contains(&STAGE_META) {
        let meta = MetaTransportQuery {
            outcomes: Arc::clone(&query.outcomes),
            treatments: Arc::clone(&query.treatments),
            target: Arc::clone(&query.target),
            sources: query
                .sources
                .iter()
                .map(|source| MetaSource {
                    population: source.population.to_string(),
                    selections: source.selection_targets.iter().map(|v| v.raw()).collect(),
                })
                .collect(),
        };
        let outcome =
            match identify_meta_catalog_metered(graph, &meta, catalog, search.meter(), ctx) {
                Ok(CatalogTransportResult::Identified(_)) => return solved(stages, STAGE_META),
                Ok(CatalogTransportResult::MissingEvidence { .. }) => "missing_evidence",
                Ok(CatalogTransportResult::NotCertified { .. }) => "not_certified",
                Err(error) if error.is_budget_or_cancel() => {
                    let stop = search.stop_of(&error);
                    let receipt =
                        stopped_receipt(search, stop, plan, stages, Vec::new(), Vec::new());
                    return Ok(Some(MixedSourceDecision::Exhausted(receipt)));
                }
                Err(error) => return Err(named_route_error(&error)),
            };
        stages.push(MixedStageRecord { stage: STAGE_META.into(), outcome });
    }
    Ok(None)
}

fn decide_charged(
    graph: &Admg,
    validated: &ValidatedMixedSourceQuery,
    catalog: &EvidenceCatalog,
    search: &mut SharedSearch<'_>,
    ctx: &ExecutionContext,
) -> Result<MixedSourceDecision, IdentificationError> {
    let mut stages = Vec::new();
    if let Some(decision) = named_routes(
        graph,
        &validated.query,
        &validated.stage_plan(),
        catalog,
        search,
        ctx,
        &mut stages,
    )? {
        return Ok(decision);
    }
    rule_search(graph, validated, catalog, search, stages)
}

/// The receipt of a stop inside the rule search: completed generations are
/// explored; the interrupted generation's unexpanded quantities and the relaxed
/// pass are unevaluated.
fn rule_stop_receipt(
    search: &SharedSearch<'_>,
    stop: SearchStop,
    validated: &ValidatedMixedSourceQuery,
    stages: &[MixedStageRecord],
    engine: &Engine<'_>,
    relaxed_pending: bool,
) -> SearchReceipt {
    let explored = (1..=engine.completed).map(|g| format!("rule_search:generation:{g}")).collect();
    let mut pending = Vec::new();
    if engine.unexpanded > 0 {
        pending.push(format!(
            "rule_search:generation:{}:unexpanded={}",
            engine.generation, engine.unexpanded
        ));
    }
    if engine.max_move < engine.n || engine.max_do < engine.n {
        pending.push(format!(
            "rule_search:deeper_bounds:move>{}:do>{}",
            engine.max_move, engine.max_do
        ));
    }
    if relaxed_pending {
        pending.push("rule_search:relaxed_missing_joint".to_owned());
    }
    stopped_receipt(search, stop, &validated.stage_plan(), stages, explored, pending)
}

/// Search, check and package the proof of `found` and its alternatives.
fn finish_found(
    graph: &Admg,
    validated: &ValidatedMixedSourceQuery,
    catalog: &EvidenceCatalog,
    search: &mut SharedSearch<'_>,
    stages: &[MixedStageRecord],
    engine: &mut Engine<'_>,
) -> Result<MixedSourceDecision, IdentificationError> {
    let shared = SelectionDiagram::try_new(graph.clone(), Arc::<[VariableId]>::from([]))
        .map_err(|error| invalid_query(&error))?;
    let signature = super::super::super::graph_signature(&shared);
    let found = engine.found.ok_or_else(invalid_derivation)?;
    let summary = MixedSearchSummary {
        operations_to_proof: engine.operations_to_proof,
        generation: engine.generation_of(found),
    };
    let mut last_steps = vec![engine.steps[found].clone()];
    last_steps.extend(engine.alternatives.clone());
    let mut derivations = Vec::new();
    for last in &last_steps {
        let (steps, arena) = engine.extract(last, catalog, &validated.query.target)?;
        // Every candidate proof is checked, one charged operation per step.
        for _ in &steps {
            let generation = engine.generation;
            if let Err(stop) = engine.charge(search, generation) {
                let receipt = rule_stop_receipt(search, stop, validated, stages, engine, false);
                return Ok(MixedSourceDecision::Exhausted(receipt));
            }
        }
        check_steps(graph, &validated.query, catalog, &steps, Some(&arena), false)?;
        derivations.push(MixedSourceDerivation {
            query: validated.query.clone(),
            graph_signature: signature.clone(),
            root_step: steps.len() - 1,
            steps,
            arena,
            stages: stages.to_vec(),
            summary,
            rule_set: MIXED_SOURCE_RULE_SET,
        });
    }
    let derivation = derivations.remove(0);
    let cited = Arc::<[RegimeId]>::from(derivation.cited_regimes());
    Ok(MixedSourceDecision::Identified {
        derivation: Box::new(derivation),
        cited,
        alternatives: derivations,
    })
}

/// Run the closure to its fixpoint. The first pass moves at most
/// [`MIXED_SOURCE_MAX_MOVE`] variables per application and carries at most
/// [`MIXED_SOURCE_MAX_DO`] actions; when that pass ends at a fixpoint having skipped an
/// application only because of a bound, both bounds are raised by one and the closure is
/// rebuilt, up to the number of observed variables (where the bounds no longer cut anything
/// off). Every pass charges the same shared budget, so a search that cannot finish deeper
/// ends as a budget stop, never as a fixpoint it did not reach.
fn run_deepening<'g>(
    graph: &'g Admg,
    variables: &[VariableId],
    inputs: Vec<MixedInput>,
    query: &MixedSourceQuery,
    search: &mut SharedSearch<'_>,
) -> Result<(Engine<'g>, RunEnd), IdentificationError> {
    let n = variables.len();
    let mut bounds = (MIXED_SOURCE_MAX_MOVE, MIXED_SOURCE_MAX_DO);
    let mut engine = Engine::with_bounds(graph, variables.to_vec(), inputs.clone(), query, bounds)?;
    loop {
        let end = engine.run(search);
        if !matches!(end, RunEnd::Fixpoint)
            || !engine.pruned
            || (bounds.0 >= n && bounds.1 >= n)
        {
            return Ok((engine, end));
        }
        bounds = ((bounds.0 + 1).min(n).max(bounds.0), (bounds.1 + 1).min(n).max(bounds.1));
        let operations = engine.operations;
        engine = Engine::with_bounds(graph, variables.to_vec(), inputs.clone(), query, bounds)?;
        engine.operations = operations;
    }
}

pub(super) fn rule_search(
    graph: &Admg,
    validated: &ValidatedMixedSourceQuery,
    catalog: &EvidenceCatalog,
    search: &mut SharedSearch<'_>,
    mut stages: Vec<MixedStageRecord>,
) -> Result<MixedSourceDecision, IdentificationError> {
    let variables = dense_variables(graph)?;
    let relaxed_pending = !validated.marginal_inputs.is_empty();
    let (mut engine, end) =
        run_deepening(graph, &variables, validated.inputs.clone(), &validated.query, search)?;
    match end {
        RunEnd::Found => {
            return finish_found(graph, validated, catalog, search, &stages, &mut engine);
        }
        RunEnd::Stopped(stop) => {
            let receipt =
                rule_stop_receipt(search, stop, validated, &stages, &engine, relaxed_pending);
            return Ok(MixedSourceDecision::Exhausted(receipt));
        }
        RunEnd::Fixpoint => {}
    }
    stages.push(MixedStageRecord { stage: STAGE_RULES.into(), outcome: "not_certified" });
    if relaxed_pending {
        let mut inputs = validated.inputs.clone();
        inputs.extend(validated.marginal_inputs.iter().cloned());
        inputs.sort_by_key(|input| input.regime);
        let (relaxed, relaxed_end) =
            run_deepening(graph, &variables, inputs, &validated.query, search)?;
        stages.pop();
        match relaxed_end {
            RunEnd::Found => {
                let found = relaxed.found.ok_or_else(invalid_derivation)?;
                let last = relaxed.steps[found].clone();
                let (proof, _) = relaxed.extract(&last, catalog, &validated.query.target)?;
                check_steps(graph, &validated.query, catalog, &proof, None, true)?;
                let leaves = proof
                    .iter()
                    .enumerate()
                    .filter(|(_, step)| step.relaxed)
                    .filter_map(|(index, step)| {
                        let leaf = step.source.as_ref()?;
                        let input = validated
                            .marginal_inputs
                            .iter()
                            .find(|input| input.regime == leaf.regime)?;
                        Some(MixedMissingLeaf {
                            step: index,
                            regime: leaf.regime,
                            study: Arc::clone(&leaf.study),
                            variables: step.y.clone(),
                            intervened: step.intervened.clone(),
                            marginals: input.marginals.clone(),
                        })
                    })
                    .collect::<Vec<_>>();
                if leaves.is_empty() {
                    return Err(invalid_derivation());
                }
                stages.push(MixedStageRecord {
                    stage: STAGE_RULES.into(),
                    outcome: "missing_evidence",
                });
                return Ok(MixedSourceDecision::MissingEvidence(Box::new(MixedMissingEvidence {
                    leaves,
                    proof,
                    stages,
                })));
            }
            RunEnd::Stopped(stop) => {
                let receipt = rule_stop_receipt(search, stop, validated, &stages, &relaxed, false);
                return Ok(MixedSourceDecision::Exhausted(receipt));
            }
            RunEnd::Fixpoint => {
                stages
                    .push(MixedStageRecord { stage: STAGE_RULES.into(), outcome: "not_certified" });
            }
        }
    }
    let goal = MixedQuantity {
        y: validated.query.outcomes.to_vec(),
        intervened: validated.query.treatments.to_vec(),
        conditioned: Vec::new(),
    };
    let frontier = engine
        .steps
        .iter()
        .rev()
        .take(FRONTIER_SHOWN)
        .map(|raw| engine.quantity(raw))
        .collect::<Vec<_>>();
    Ok(MixedSourceDecision::NotCertified(Box::new(MixedSearchInspection {
        stages,
        goal,
        frontier,
        quantities: engine.steps.len(),
        generations: engine.completed,
        operations: engine.operations,
        rule_counts: engine.rule_counts.iter().map(|(rule, count)| (*rule, *count)).collect(),
        exclusions: validated.exclusions.clone(),
    })))
}
