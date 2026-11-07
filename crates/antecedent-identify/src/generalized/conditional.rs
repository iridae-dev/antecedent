//! Conditional adjustment identification and bounded candidate enumeration.

use super::{
    Admg, AverageEffectQuery, BitSet, CAPPED_COMPLETION_DIAGNOSTIC_CODE,
    CONDITIONAL_RESULTS_TRUNCATED_DIAGNOSTIC_CODE, CONDITIONAL_SEARCH_BOUNDED_DIAGNOSTIC_CODE,
    CausalQuery, DSeparationWorkspace, DenseNodeId, Diagnostic, DiagnosticKind, DiagnosticSeverity,
    IdentificationError, IdentificationResult, IdentificationStatus, IdentifiedEstimand, Value,
    VariableId, dense_u32, directed_closure, mutilate_outgoing, not_identified,
};

pub(crate) fn validate_dag_conditional_adjustment(
    dag: &antecedent_graph::Dag,
    query: &AverageEffectQuery,
    result: &mut IdentificationResult,
    config: &crate::backdoor::AdjustmentSearchConfig,
) -> Result<(), IdentificationError> {
    if query.effect_modifiers.is_empty() {
        return Ok(());
    }
    let admg = crate::prepared::dag_to_admg(dag)?;
    let t = dag
        .nodes()
        .iter()
        .position(|node| *node == antecedent_graph::NodeRef::Static(query.treatment))
        .ok_or(IdentificationError::UnknownVariable { id: query.treatment })?;
    validate_conditional_adjustment(
        &admg,
        &mutilate_outgoing(&admg, DenseNodeId::from_raw(dense_u32(t))),
        dag.nodes(),
        query,
        result,
        config,
    )
}

/// The regression conditions on Z union W. A marginal adjustment certificate
/// for Z alone does not support this conditional regression: W may be a
/// mediator, or may open a collider path. This is a sufficient backdoor check,
/// not a complete conditional-ID algorithm.
#[allow(
    clippy::too_many_lines,
    reason = "identification steps and their shared state form one bounded procedure"
)]
pub(super) fn validate_conditional_adjustment(
    graph: &Admg,
    mutilated: &Admg,
    nodes: &[antecedent_graph::NodeRef],
    query: &AverageEffectQuery,
    result: &mut IdentificationResult,
    config: &crate::backdoor::AdjustmentSearchConfig,
) -> Result<(), IdentificationError> {
    if query.effect_modifiers.is_empty() {
        return Ok(());
    }
    let dense = |v: VariableId| crate::prepared::dense_of_static(nodes, v);
    let t = dense(query.treatment)?;
    let y = dense(query.outcome)?;
    let modifiers =
        query.effect_modifiers.iter().copied().map(dense).collect::<Result<Vec<_>, _>>()?;
    let descendants = directed_closure(graph, &[t], false);
    let pretreatment = modifiers.iter().all(|w| *w != y && !descendants.contains(*w));
    let mut ws = DSeparationWorkspace::default();
    let mut valid = Vec::new();
    if pretreatment {
        for estimand in &result.estimands {
            let mut conditioning = estimand
                .adjustment_set
                .iter()
                .copied()
                .map(dense)
                .collect::<Result<Vec<_>, _>>()?;
            conditioning.extend_from_slice(&modifiers);
            conditioning.sort_unstable();
            conditioning.dedup();
            if mutilated.is_m_separated(t, y, &conditioning, &mut ws)? {
                valid.push(estimand.clone());
            }
        }
    }
    if valid.is_empty() && pretreatment {
        let found =
            constrained_conditional_set(graph, mutilated, nodes, query, &modifiers, config)?;
        match found {
            ConditionalSearch::Found(adjustments, examined, budget_exhausted, truncated) => {
                for adjustment in adjustments {
                    let (active, control) = conditional_levels(query)?;
                    let functional = result.arena.backdoor_ate(
                        query.treatment,
                        query.outcome,
                        &adjustment,
                        active,
                        control,
                    );
                    valid.push(IdentifiedEstimand::backdoor(
                        "backdoor.adjustment",
                        adjustment.into(),
                        functional,
                    ));
                }
                result.status = IdentificationStatus::NonparametricallyIdentified;
                result.query = CausalQuery::AverageEffect(query.clone());
                result.performance.candidates_examined += examined;
                result.performance.sets_returned = valid.len() as u64;
                result.diagnostics.retain(|d| d.code.as_ref() != CAPPED_COMPLETION_DIAGNOSTIC_CODE);
                result.derivation.push("conditional.adjustment.search", "searched with the fixed pre-treatment modifier in every separation test; alternative joint conditioning set certified");
                if budget_exhausted {
                    result.diagnostics.push(crate::result::search_bounded_diagnostic(
                        CONDITIONAL_SEARCH_BOUNDED_DIAGNOSTIC_CODE,
                        format!(
                            "conditional adjustment search stopped (max_examinations={}, declared \
                             candidate ceiling or leasable storage) before it finished; the \
                             returned set separates treatment from outcome but a smaller one may \
                             exist",
                            config.max_examinations
                        ),
                    ));
                }
                if truncated {
                    result.diagnostics.push(Diagnostic::new(
                        CONDITIONAL_RESULTS_TRUNCATED_DIAGNOSTIC_CODE,
                        DiagnosticKind::Execution,
                        DiagnosticSeverity::Info,
                        format!(
                            "conditional adjustment search stopped at max_results={}; every \
                             returned set is certified but further sets may exist",
                            config.max_results
                        ),
                    ));
                }
                if result.required_assumptions.entries.is_empty() {
                    result
                        .required_assumptions
                        .push(crate::assumptions::causal_markov("conditional.adjustment"));
                }
            }
            ConditionalSearch::Absent => {}
        }
    }
    if valid.is_empty() {
        *result = not_identified(
            result.query.clone(),
            "conditional adjustment not certified: modifiers must be pre-treatment and \
             Z union modifiers must block backdoor paths; marginal ATE identification \
             alone is insufficient; constrained adjustment search found no set (general conditional ID not attempted)",
        );
        result.diagnostics.push(Diagnostic::new(
            "identify.conditional.adjustment_unverified",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            "conditional adjustment is not certified on this completion; its mass is retained as unidentified",
        ));
    } else {
        result.estimands = valid;
    }
    Ok(())
}

fn conditional_levels(query: &AverageEffectQuery) -> Result<(Value, Value), IdentificationError> {
    match (&query.active, &query.control) {
        (
            antecedent_core::Intervention::Set { value: a, .. },
            antecedent_core::Intervention::Set { value: c, .. },
        ) => Ok((a.clone(), c.clone())),
        _ => Err(IdentificationError::unsupported(
            "conditional adjustment requires Set interventions",
        )),
    }
}

/// A declared candidate cap is an undecided search outcome; inability to fund a search
/// inside that cap is a physical failure and must not contribute unidentified class mass.
pub(super) fn fund_adjustment_enumeration(
    search: &mut crate::exact_search::ExactSearch<'_>,
    candidates: usize,
    ceiling: usize,
    max_results: usize,
) -> Result<bool, IdentificationError> {
    if candidates > ceiling {
        return Ok(false);
    }
    if !search.fund_enumeration(candidates, ceiling, max_results)? {
        return Err(IdentificationError::budget(crate::IdentificationBudget::Memory));
    }
    Ok(true)
}

pub(super) enum ConditionalSearch {
    /// Certified sets, separation tests spent, and whether the examination budget cut the
    /// search for smaller sets short, and whether `max_results` ended the enumeration.
    Found(Vec<Vec<VariableId>>, u64, bool, bool),
    /// Proved: no admissible set exists.
    Absent,
}

/// How a conditional search ended.
#[derive(Clone, Copy)]
struct Outcome {
    budget_exhausted: bool,
    truncated: bool,
}

#[allow(
    clippy::too_many_lines,
    reason = "identification steps and their shared state form one bounded procedure"
)]
pub(super) fn constrained_conditional_set(
    graph: &Admg,
    cut: &Admg,
    nodes: &[antecedent_graph::NodeRef],
    query: &AverageEffectQuery,
    modifiers: &[DenseNodeId],
    config: &crate::backdoor::AdjustmentSearchConfig,
) -> Result<ConditionalSearch, IdentificationError> {
    config.validate()?;
    let dense = |v| crate::prepared::dense_of_static(nodes, v);
    let t = dense(query.treatment)?;
    let y = dense(query.outcome)?;
    let candidates = gac_conditional_candidates(graph, nodes, t, y, modifiers, config);
    // The candidates are An({T,Y} + modifiers) minus the forbidden variables, so together
    // with the modifiers they are the ancestral conditioning set: it separates T from Y in
    // the cut graph whenever any admissible set does (van der Zander, Liskiewicz & Textor
    // 2014). One test decides existence; the subset search only chooses among valid sets.
    let mut ws = DSeparationWorkspace::default();
    let mut examined = 1u64;
    let mut all_conditioned = candidates.clone();
    all_conditioned.extend_from_slice(modifiers);
    if !cut.is_m_separated(t, y, &all_conditioned, &mut ws)? {
        return Ok(ConditionalSearch::Absent);
    }
    let context = crate::exact_search::context_or_default(config.context.as_ref());
    let mut search = crate::exact_search::ExactSearch::open(&context)?;
    if !fund_adjustment_enumeration(
        &mut search,
        candidates.len(),
        config.max_candidates,
        config.max_results,
    )? {
        // Over the declared ceiling the subset search is skipped, not the decision: the
        // candidates were just certified as a conditioning set, so they are the answer.
        return Ok(finish_conditional_search(
            &[],
            &candidates,
            nodes,
            examined,
            Outcome { budget_exhausted: true, truncated: false },
            config,
        ));
    }
    let mut truncated = false;
    let mut found: Vec<Vec<DenseNodeId>> = Vec::new();
    // Found sets as bit masks, so each superset/subset test is word-wise.
    let mut found_masks: Vec<BitSet> = Vec::new();
    let mut z_mask = BitSet::with_len(graph.node_count());
    let mut budget_exhausted = false;
    let m = candidates.len();
    let descending = config.maximal_only && !config.minimal_only;
    for step in 0..=m {
        let size = if descending { m - step } else { step };
        let mut error = None;
        crate::enum_masks::for_each_mask_of_size(&candidates, size, |z| {
            if let Err(e) = search.observe() {
                error = Some(e);
                return true;
            }
            if !found_masks.is_empty() {
                z_mask.clear();
                for &d in z {
                    z_mask.insert(d);
                }
                let skipped = (config.minimal_only
                    && found_masks.iter().any(|old| old.is_subset_of(&z_mask)))
                    || (config.maximal_only
                        && found_masks.iter().any(|old| z_mask.is_subset_of(old)));
                if skipped {
                    if examined >= config.max_examinations {
                        budget_exhausted = true;
                        return true;
                    }
                    examined += 1;
                    return false;
                }
            }
            // The ancestral set was tested above; it is not tested twice.
            let known_valid = z == candidates.as_slice();
            if !known_valid && examined >= config.max_examinations {
                budget_exhausted = true;
                return true;
            }
            let mut conditioned = z.to_vec();
            conditioned.extend_from_slice(modifiers);
            let separated = if known_valid {
                Ok(true)
            } else {
                examined += 1;
                cut.is_m_separated(t, y, &conditioned, &mut ws)
            };
            match separated {
                Ok(true) => {
                    found.push(z.to_vec());
                    let mut mask = BitSet::with_len(graph.node_count());
                    for &d in z {
                        mask.insert(d);
                    }
                    found_masks.push(mask);
                    let complete = config.minimal_only && z.is_empty();
                    let full = found.len() >= config.max_results;
                    truncated |= full && !complete;
                    complete || full
                }
                Ok(false) => false,
                Err(e) => {
                    error = Some(IdentificationError::from(e));
                    true
                }
            }
        });
        if let Some(e) = error {
            return Err(e);
        }
        // The budget is noticed by the first subset that is refused a test, so that a
        // search which ends exactly on its budget is complete rather than bounded.
        if found.len() >= config.max_results
            || (config.minimal_only && found.iter().any(Vec::is_empty))
            || budget_exhausted
        {
            break;
        }
    }
    Ok(finish_conditional_search(
        &found,
        &candidates,
        nodes,
        examined,
        Outcome { budget_exhausted, truncated },
        config,
    ))
}

/// Translate the completed dense search and rank its admissible sets. An exhausted
/// selection search still carries the ancestral set whose validity was established.
fn finish_conditional_search(
    found: &[Vec<DenseNodeId>],
    candidates: &[DenseNodeId],
    nodes: &[antecedent_graph::NodeRef],
    examined: u64,
    Outcome { budget_exhausted, truncated }: Outcome,
    config: &crate::backdoor::AdjustmentSearchConfig,
) -> ConditionalSearch {
    let mut sets: Vec<Vec<VariableId>> = found
        .iter()
        .map(|set| {
            set.iter()
                .map(|id| match nodes[id.as_usize()] {
                    antecedent_graph::NodeRef::Static(v) => v,
                    _ => unreachable!("static graph"),
                })
                .collect()
        })
        .collect();
    if sets.is_empty() && budget_exhausted {
        // The budget stopped the search for smaller sets, but the ancestral set is already
        // known valid: existence is decided, only the choice of set is unfinished.
        sets.push(
            candidates
                .iter()
                .map(|id| match nodes[id.as_usize()] {
                    antecedent_graph::NodeRef::Static(v) => v,
                    _ => unreachable!("static graph"),
                })
                .collect(),
        );
    }
    rank_conditional_sets(&mut sets, config);
    if sets.is_empty() {
        ConditionalSearch::Absent
    } else {
        ConditionalSearch::Found(sets, examined, budget_exhausted, truncated)
    }
}

fn gac_conditional_candidates(
    graph: &Admg,
    nodes: &[antecedent_graph::NodeRef],
    t: DenseNodeId,
    y: DenseNodeId,
    modifiers: &[DenseNodeId],
    config: &crate::backdoor::AdjustmentSearchConfig,
) -> Vec<DenseNodeId> {
    let mut seeds = vec![t, y];
    seeds.extend_from_slice(modifiers);
    let ancestors = directed_closure(graph, &seeds, true);
    let descendants = directed_closure(graph, &[t], false);
    nodes
        .iter()
        .enumerate()
        .filter_map(|(i, node)| {
            let antecedent_graph::NodeRef::Static(variable) = node else {
                return None;
            };
            let id = DenseNodeId::from_raw(dense_u32(i));
            let outside_history = config.max_history_lag.is_some_and(|cap| {
                config.history_lags.iter().any(|(v, lag)| v == variable && *lag > cap)
            });
            (id != y
                && !descendants.contains(id)
                && ancestors.contains(id)
                && !modifiers.contains(&id)
                && !config.forbidden.contains(variable)
                && !outside_history)
                .then_some(id)
        })
        .collect()
}

fn rank_conditional_sets(
    sets: &mut [Vec<VariableId>],
    config: &crate::backdoor::AdjustmentSearchConfig,
) {
    if !config.measurement_costs.is_empty() {
        let cost = |set: &[VariableId]| {
            set.iter()
                .map(|v| {
                    config
                        .measurement_costs
                        .iter()
                        .find(|(id, _)| id == v)
                        .map_or(1.0, |(_, cost)| *cost)
                })
                .sum::<f64>()
        };
        sets.sort_by(|a, b| cost(a).total_cmp(&cost(b)));
    }
}

// Perkovic et al. (2018), Definition 6 and Theorem 7. A MAG arrow need not
// exclude hidden confounding. Delete only visible first edges of causal paths;
// keep side-effect edges so conditioning cannot conceal collider bias.
