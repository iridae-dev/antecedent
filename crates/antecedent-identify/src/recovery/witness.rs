//! Self-censoring witness and stored-recovery verification.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Admg, Arc, At, BTreeMap, BTreeSet, CausalExprArena, ClassicalTransportQuery,
    ClassicalTransportResult, DomainRef, EvidenceCatalog, ExecutionContext, ExprId, ExprNode,
    MGraph, NodeRef, ObservationRecoveryQuery, RecoveredEffect, RecoveredEffectQuery,
    RecoveredEffectRecord, RecoveryDecision, RecoveryDerivation, RecoveryDerivationRecord,
    RecoveryDetail, RecoveryError, RecoveryLimits, RecoveryWitness, Role, SearchLimits,
    SelectionDiagram, SharedSearch, VariableId, WITNESS_SCALE, WitnessCheck, WitnessMechanism,
    budget_error, check_formula, RECOVERY_DEFAULT_LIMITS, decide_observation_recovery, identify_classical_transport_metered,
    ids, observed_distribution, refuse, resolve_roles, var,
};

/// The closed-form self-censoring witness for `x -> r`: every mechanism a fair
/// coin except `x` and `r`, which differ between the two models while
/// `P(r = 1, x)` is equal.
pub(super) fn self_censoring_witness(
    m: &MGraph<'_>,
    x: VariableId,
    r: VariableId,
) -> Result<RecoveryWitness, RecoveryError> {
    let half = WITNESS_SCALE / 2;
    let mut first = Vec::new();
    let mut second = Vec::new();
    for (v, role) in &m.roles {
        if matches!(role, Role::Proxy(_)) {
            continue;
        }
        let parents =
            m.graph.parents(m.dense[v]).iter().map(|p| var(m.graph, *p)).collect::<Vec<_>>();
        let configs = crate::pow2::checked_pow2(parents.len()).ok_or_else(|| {
            refuse(
                RecoveryDetail::BoundsExceeded,
                "a witness mechanism has too many parents to enumerate",
            )
        })?;
        let at = parents.iter().position(|p| *p == x);
        let table = |x1: u32, x0: u32| -> Vec<u32> {
            (0..configs)
                .map(|c| {
                    let bit = at.map(|i| (c >> (parents.len() - 1 - i)) & 1);
                    if bit == Some(1) { x1 } else { x0 }
                })
                .collect()
        };
        let (a, b) = if *v == x {
            (vec![half; configs], vec![24; configs])
        } else if *v == r {
            (table(36, 24), table(45, 20))
        } else {
            (vec![half; configs], vec![half; configs])
        };
        let raw = parents.iter().map(|p| p.raw()).collect::<Vec<_>>();
        first.push(WitnessMechanism { node: v.raw(), parents: raw.clone(), numerators: a });
        second.push(WitnessMechanism { node: v.raw(), parents: raw, numerators: b });
    }
    Ok(RecoveryWitness { edge: (x.raw(), r.raw()), first, second })
}

/// Verify a witness against an m-graph by exact enumeration: both models are
/// positive and Markov to the graph, agree on every observed-law cell, and
/// differ on the target `P(X, O)`. No theorem is trusted. A verified witness
/// shows non-recoverability over every model Markov to the m-graph, not under
/// faithful or generic parameters.
///
/// Run under the default operation limit (one operation per enumerated
/// configuration); a witness needing more is refused with a budget detail, never
/// accepted or rejected as a theorem.
///
/// # Errors
///
/// [`RecoveryDetail::InvalidQuery`] for a malformed role declaration, or
/// [`RecoveryDetail::InvalidDerivation`] when the witness does not verify.
pub fn verify_recovery_witness(
    graph: &Admg,
    query: &ObservationRecoveryQuery,
    witness: &RecoveryWitness,
) -> Result<WitnessCheck, RecoveryError> {
    let query = query.canonical();
    let m = resolve_roles(graph, &query)?;
    if graph.has_bidirected() {
        return Err(refuse(RecoveryDetail::InvalidDerivation, "a witness is checked on a DAG"));
    }
    let mut remaining = RECOVERY_DEFAULT_LIMITS.search.operations;
    verify_witness_charged(&m, witness, &mut |_| {
        if remaining == 0 {
            return Err(refuse(
                RecoveryDetail::Budget,
                "the witness enumeration exceeds the default operation limit",
            ));
        }
        remaining -= 1;
        Ok(())
    })
}

#[allow(
    clippy::too_many_lines,
    reason = "One linear pass per stage; splitting it would scatter the checks"
)]
pub(super) fn verify_witness_charged(
    m: &MGraph<'_>,
    witness: &RecoveryWitness,
    charge_step: &mut dyn FnMut(u64) -> Result<(), RecoveryError>,
) -> Result<WitnessCheck, RecoveryError> {
    let bad = |why: &str| refuse(RecoveryDetail::InvalidDerivation, format!("witness: {why}"));
    // Nodes with a mechanism: every non-proxy node, in variable order.
    let nodes: Vec<VariableId> =
        m.roles.iter().filter(|(_, r)| !matches!(r, Role::Proxy(_))).map(|(v, _)| *v).collect();
    let index: BTreeMap<VariableId, usize> =
        nodes.iter().enumerate().map(|(i, v)| (*v, i)).collect();
    let n = nodes.len();
    // The configuration index is a `u64`; the budget stops the enumeration long
    // before this addressing bound is reachable.
    if n >= 63 {
        return Err(refuse(
            RecoveryDetail::Budget,
            "the witness enumeration is not addressable within a 64-bit configuration index",
        ));
    }
    let check_model = |model: &[WitnessMechanism]| -> Result<(), RecoveryError> {
        if model.len() != n {
            return Err(bad("a model must give one mechanism per non-proxy node"));
        }
        for mechanism in model {
            let v = VariableId::from_raw(mechanism.node);
            if !index.contains_key(&v) {
                return Err(bad("a mechanism names a proxy or an unknown node"));
            }
            let graph_parents = m
                .graph
                .parents(m.dense[&v])
                .iter()
                .map(|p| var(m.graph, *p).raw())
                .collect::<Vec<_>>();
            if mechanism.parents != graph_parents {
                return Err(bad(
                    "a mechanism's parents are not the node's graph parents (not Markov to the m-graph)",
                ));
            }
            if crate::pow2::checked_pow2(graph_parents.len())
                .is_none_or(|configs| mechanism.numerators.len() != configs)
            {
                return Err(bad("a mechanism must give one probability per parent configuration"));
            }
            if mechanism.numerators.iter().any(|k| *k == 0 || *k >= WITNESS_SCALE) {
                return Err(bad(
                    "a mechanism probability is not strictly between 0 and 1 (positivity)",
                ));
            }
        }
        let distinct: BTreeSet<u32> = model.iter().map(|m| m.node).collect();
        if distinct.len() != n {
            return Err(bad("a node has two mechanisms"));
        }
        Ok(())
    };
    check_model(&witness.first)?;
    check_model(&witness.second)?;
    let substantive: Vec<VariableId> = m
        .roles
        .iter()
        .filter(|(_, r)| matches!(r, Role::Partial(_) | Role::Full))
        .map(|(v, _)| *v)
        .collect();
    // Observed coordinates: each response and fully observed value, and each
    // proxy's value in {0, 1, ?} (encoded 0, 1, 2).
    let pairs = witness_roles(m);
    let enumerate = |model: &[WitnessMechanism],
                     charge_step: &mut dyn FnMut(u64) -> Result<(), RecoveryError>|
     -> Result<(Masses, Masses), RecoveryError> {
        let by_node: BTreeMap<u32, &WitnessMechanism> = model.iter().map(|m| (m.node, m)).collect();
        let mut observed = BTreeMap::new();
        let mut target = BTreeMap::new();
        for config in 0u64..(1u64 << n) {
            charge_step(
                u64::try_from(observed.len() + target.len()).unwrap_or(u64::MAX).saturating_mul(48),
            )?;
            let value = |v: VariableId| u8::from((config >> index[&v]) & 1 == 1);
            let mut mass: u128 = 1;
            for v in &nodes {
                let mechanism = by_node[&v.raw()];
                let parent_config = mechanism.parents.iter().fold(0usize, |acc, p| {
                    (acc << 1) | usize::from(value(VariableId::from_raw(*p)))
                });
                let k = u128::from(mechanism.numerators[parent_config]);
                mass *= if value(*v) == 1 { k } else { u128::from(WITNESS_SCALE) - k };
            }
            let mut cell = Vec::new();
            for (v, role) in &m.roles {
                cell.push(match role {
                    Role::Partial(_) => continue,
                    Role::Proxy(i) => {
                        let p = &pairs[*i];
                        if value(p.1) == 1 { value(p.0) } else { 2 }
                    }
                    Role::Response(_) | Role::Full => value(*v),
                });
            }
            *observed.entry(cell).or_insert(0) += mass;
            let target_cell: Vec<u8> = substantive.iter().map(|v| value(*v)).collect();
            *target.entry(target_cell).or_insert(0) += mass;
        }
        Ok((observed, target))
    };
    let (observed_a, target_a) = enumerate(&witness.first, charge_step)?;
    let (observed_b, target_b) = enumerate(&witness.second, charge_step)?;
    if observed_a != observed_b {
        return Err(bad("the two models disagree on the observed law"));
    }
    let Some((cell, mass_a)) =
        target_a.iter().find(|(cell, mass)| target_b.get(*cell) != Some(mass))
    else {
        return Err(bad("the two models agree on the target"));
    };
    let denominator = u128::from(WITNESS_SCALE).pow(u32::try_from(n).unwrap_or(u32::MAX));
    Ok(WitnessCheck {
        observed_cells: observed_a.len(),
        differing_cell: substantive.iter().copied().zip(cell.iter().copied()).collect(),
        masses: (*mass_a, target_b.get(cell).copied().unwrap_or(0)),
        denominator,
    })
}

/// Exact integer masses per cell (observed or target levels).
type Masses = BTreeMap<Vec<u8>, u128>;

/// `(variable, response)` per partially observed index.
fn witness_roles(m: &MGraph<'_>) -> Vec<(VariableId, VariableId)> {
    let mut out = BTreeMap::new();
    for (v, role) in &m.roles {
        match role {
            Role::Partial(i) => out.entry(*i).or_insert((*v, *v)).0 = *v,
            Role::Response(i) => out.entry(*i).or_insert((*v, *v)).1 = *v,
            _ => {}
        }
    }
    out.into_values().collect()
}

/// Identify the downstream effect with the ordinary target ID stage of sID on
/// the causal restriction of the m-graph, charged to the shared budget.
#[allow(
    clippy::too_many_lines,
    reason = "One linear pass per stage; splitting it would scatter the checks"
)]
pub(super) fn identify_effect(
    m: &MGraph<'_>,
    query: &ObservationRecoveryQuery,
    effect: &RecoveredEffectQuery,
    search: &mut SharedSearch<'_>,
    at: At,
    ctx: &ExecutionContext,
) -> Result<(RecoveredEffect, RecoveredEffectRecord), RecoveryError> {
    use RecoveryDetail::{HandoffMismatch, InvalidDerivation, InvalidQuery};
    let substantive: BTreeSet<VariableId> = query.substantive().into_iter().collect();
    let mut nodes = BTreeSet::new();
    for node in effect.graph.nodes() {
        let NodeRef::Static(v) = node else {
            return Err(refuse(HandoffMismatch, "the effect graph must have static nodes only"));
        };
        nodes.insert(*v);
    }
    if nodes != substantive {
        return Err(refuse(
            HandoffMismatch,
            "the effect graph's variables are not exactly the recovered law's variables X and O",
        ));
    }
    if effect.graph.has_bidirected() {
        return Err(refuse(
            HandoffMismatch,
            "the recovered law factorizes by a DAG; a bidirected edge changes the factorization",
        ));
    }
    let Some(edges) = effect.graph.static_directed_edges_sorted() else {
        return Err(refuse(HandoffMismatch, "the effect graph must have static nodes only"));
    };
    let edges: Vec<(u32, u32)> =
        edges.into_iter().map(|(from, to)| (from.raw(), to.raw())).collect();
    let mut restriction = Vec::new();
    for v in &substantive {
        for child in m.children(*v) {
            if substantive.contains(&child) {
                restriction.push((v.raw(), child.raw()));
            }
        }
    }
    restriction.sort_unstable();
    if edges != restriction {
        return Err(refuse(
            HandoffMismatch,
            "the effect graph is not the m-graph's causal restriction (factorization mismatch)",
        ));
    }
    let mut outcomes = effect.outcomes.to_vec();
    outcomes.sort_unstable();
    let mut treatments = effect.treatments.to_vec();
    treatments.sort_unstable();
    if outcomes.is_empty()
        || outcomes.windows(2).any(|w| w[0] == w[1])
        || treatments.windows(2).any(|w| w[0] == w[1])
        || outcomes.iter().any(|o| treatments.contains(o))
    {
        return Err(refuse(
            InvalidQuery,
            "effect outcomes must be non-empty, distinct and disjoint from treatments",
        ));
    }
    if outcomes.iter().chain(&treatments).any(|v| !substantive.contains(v)) {
        return Err(refuse(
            HandoffMismatch,
            "effect outcomes and treatments must be variables of the recovered law",
        ));
    }
    let diagram = SelectionDiagram::try_new(effect.graph.clone(), Arc::<[VariableId]>::from([]))
        .map_err(|e| refuse(HandoffMismatch, e.to_string()))?;
    let target: Arc<str> = Arc::clone(&query.population);
    let sid_query = ClassicalTransportQuery {
        outcomes: outcomes.clone().into(),
        treatments: treatments.clone().into(),
        // The source is a label only: a derivation citing any leaf other than the
        // target's observational law is refused below, so no source law is used.
        source: Arc::from(format!("{target}#recovery-no-source")),
        target: Arc::clone(&target),
    };
    search.begin_stage();
    let result = identify_classical_transport_metered(&diagram, &sid_query, search.meter(), ctx)
        .map_err(|error| {
            if error.is_budget_or_cancel() {
                budget_error(search, at, search.stop_of(&error))
            } else {
                refuse(InvalidDerivation, format!("ordinary target ID failed: {error}"))
            }
        })?;
    search.end_stage();
    // Every effect on a DAG whose variables the recovered law all covers is
    // identified by ordinary target ID (the g-formula); anything else is a defect,
    // reported as a failed derivation, never as a claim about the effect.
    let ClassicalTransportResult::Identified(derivation) = result else {
        return Err(refuse(
            InvalidDerivation,
            "ordinary target ID did not identify an effect on a fully recovered DAG",
        ));
    };
    let arena = derivation.arena().clone();
    let root = derivation.root();
    // Defense in depth, unreachable through the public API: sID on a selection
    // diagram without selection nodes identifies every DAG effect by target ID,
    // whose leaves are all target observational laws (audit mutant A7 survives
    // every public-route test for that reason). The predicate is unit tested
    // directly (`the_effect_leaf_predicate_refuses_foreign_or_interventional_leaves`).
    if !cites_only_target_observational(&arena, root, &target) {
        return Err(refuse(
            InvalidDerivation,
            "the effect derivation cites a law other than the recovered target law",
        ));
    }
    let record = RecoveredEffectRecord {
        outcomes: ids(&outcomes),
        treatments: ids(&treatments),
        edges,
        root: root.raw(),
        rules: derivation.rules().into_iter().map(str::to_owned).collect(),
    };
    Ok((
        RecoveredEffect { arena, root, outcomes: outcomes.into(), treatments: treatments.into() },
        record,
    ))
}

/// Whether every distribution leaf under `root` is an observational,
/// non-interventional law of `target`: the only law a recovered law supplies.
pub(super) fn cites_only_target_observational(
    arena: &CausalExprArena,
    root: ExprId,
    target: &str,
) -> bool {
    arena.distribution_leaves(root).into_iter().all(|leaf| {
        let ExprNode::Distribution { intervention, domain, population, .. } = arena.node(leaf)
        else {
            return true;
        };
        arena.population(*population) == target
            && *domain == DomainRef::Observational
            && arena.intervention_assignments(*intervention).is_empty()
    })
}

/// The stored artifact parts [`verify_observation_recovery`] re-checks: the derivation
/// record, its expression arena and, when an effect was derived, the effect arena.
#[derive(Clone, Copy, Debug)]
pub struct StoredRecovery<'a> {
    /// Stored derivation record.
    pub record: &'a RecoveryDerivationRecord,
    /// Stored recovered-law arena.
    pub arena: &'a CausalExprArena,
    /// Stored downstream-effect arena.
    pub effect_arena: Option<&'a CausalExprArena>,
}

/// Re-check a stored derivation: re-decide from the stored inputs under the
/// stored limits (refused when above `max`), accept only an identical record and
/// arena, and run the independent formula checker on the stored record.
///
/// # Errors
///
/// [`RecoveryDetail::BoundsExceeded`] for stored limits above `max`;
/// [`RecoveryDetail::InvalidDerivation`] when anything differs; the decision's
/// own refusal when the inputs no longer recover.
pub fn verify_observation_recovery(
    graph: &Admg,
    query: &ObservationRecoveryQuery,
    catalog: &EvidenceCatalog,
    effect: Option<&RecoveredEffectQuery>,
    stored: &StoredRecovery<'_>,
    max: RecoveryLimits,
    ctx: &ExecutionContext,
) -> Result<RecoveryDerivation, RecoveryError> {
    let StoredRecovery { record, arena, effect_arena } = *stored;
    let stored = RecoveryLimits {
        search: SearchLimits {
            operations: record.receipt.operations_limit,
            depth: record.receipt.depth_limit,
        },
        memory_bytes: record.receipt.memory_bytes,
    };
    if stored.search.operations > max.search.operations
        || stored.search.depth > max.search.depth
        || stored.memory_bytes > max.memory_bytes
    {
        return Err(refuse(
            RecoveryDetail::BoundsExceeded,
            "the stored search limits exceed the consumer's maxima",
        ));
    }
    let invalid = |why: &str| refuse(RecoveryDetail::InvalidDerivation, why.to_owned());
    let canonical = query.canonical();
    let observed = observed_distribution(&canonical, catalog)?;
    // The independent checker runs on the stored record and arena first, so a
    // tampered formula is refused for what it is, not only for differing.
    check_formula(graph, &canonical, &observed, record, arena, &mut |_| Ok(()))?;
    let decision = decide_observation_recovery(graph, query, catalog, effect, stored, ctx)?;
    let RecoveryDecision::Recovered(fresh) = decision else {
        return Err(invalid("the stored inputs are nonrecoverable"));
    };
    if fresh.record != *record {
        return Err(invalid("the stored derivation record is not reproduced"));
    }
    if fresh.arena != *arena {
        return Err(invalid("the stored formula expression is not reproduced"));
    }
    match (&fresh.effect, effect_arena) {
        (None, None) => {}
        (Some(e), Some(stored)) if e.arena == *stored => {}
        _ => return Err(invalid("the stored effect expression is not reproduced")),
    }
    Ok(*fresh)
}
