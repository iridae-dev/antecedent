use super::{
    Abort, BTreeMap, BTreeSet, Built, Closure, Conflict, CounterfactualFactor,
    CounterfactualFunctional, CounterfactualLevel, CounterfactualLevelIndex,
    CounterfactualMovedAtom, CounterfactualSum, CounterfactualSymbol, CounterfactualSymbolId,
    CounterfactualTransportProblem, CounterfactualVariableIndex, Factored,
    INITIAL_COMPLEMENT_DEPTH, IdentificationError, Level, Search, Validated, WorldIdx, atoms_map,
    build_network, factorize, invalid_query, separated,
};

pub(super) struct DeriveCtx<'a> {
    problem: &'a CounterfactualTransportProblem,
    closure: &'a Closure,
    /// Nesting of the binary consistency complement allowed in the current attempt.
    complement_depth: std::cell::Cell<usize>,
    /// Some complement was left untried because of `complement_depth`.
    pub(super) complement_pruned: std::cell::Cell<bool>,
}

impl<'a> DeriveCtx<'a> {
    pub(super) fn new(problem: &'a CounterfactualTransportProblem, closure: &'a Closure) -> Self {
        Self {
            problem,
            closure,
            complement_depth: std::cell::Cell::new(INITIAL_COMPLEMENT_DEPTH),
            complement_pruned: std::cell::Cell::new(false),
        }
    }
}

fn mask_text(problem: &CounterfactualTransportProblem, mask: u32) -> String {
    let names: Vec<String> =
        problem.mask_variables(mask).iter().map(|v| v.raw().to_string()).collect();
    format!("{{{}}}", names.join(","))
}

fn lift_level(level: Level) -> CounterfactualLevel {
    match level {
        Level::Fixed(l) => CounterfactualLevel::Fixed(CounterfactualLevelIndex::new(l)),
        Level::Symbol(s) => CounterfactualLevel::Symbol(CounterfactualSymbolId::new(s)),
    }
}

/// Derive the probability of one conjunction from the closure, trying the direct
/// factorization first and then the binary consistency complement.
#[allow(
    clippy::too_many_lines,
    reason = "identification steps and their shared state form one bounded procedure"
)]
fn derive_conjunction(
    ctx: &DeriveCtx<'_>,
    search: &mut Search<'_>,
    worlds: &[WorldIdx],
    atoms: &BTreeMap<(usize, usize), usize>,
    depth: usize,
    used: &BTreeSet<(usize, usize)>,
) -> Result<Option<CounterfactualFunctional>, Abort> {
    let problem = ctx.problem;
    search.charge(depth, (atoms.len() as u64).saturating_mul(256))?;
    let net = match build_network(problem, worlds, atoms)? {
        Built::Zero => return Ok(Some(CounterfactualFunctional::Zero)),
        Built::Net(net) => net,
    };
    match factorize(problem, &net) {
        Factored::Terms { symbols, terms } => {
            let mut factors: Vec<CounterfactualFactor> = Vec::with_capacity(terms.len());
            let mut missing: Option<u32> = None;
            for term in &terms {
                if let Some(item) = ctx.closure.target_item(term.mask) {
                    factors.push(CounterfactualFactor {
                        item,
                        outputs: term
                            .outputs
                            .iter()
                            .map(|&(v, level)| {
                                (CounterfactualVariableIndex::new(v), lift_level(level))
                            })
                            .collect(),
                        parents: term
                            .parents
                            .iter()
                            .map(|&(v, level)| {
                                (CounterfactualVariableIndex::new(v), lift_level(level))
                            })
                            .collect(),
                    });
                } else {
                    missing = Some(term.mask);
                    break;
                }
            }
            match missing {
                None => {
                    search.admit_summation(
                        symbols.iter().map(|&(_, v)| problem.levels[v].len() as u64),
                    )?;
                    return Ok(Some(CounterfactualFunctional::Sum(CounterfactualSum {
                        symbols: symbols
                            .iter()
                            .map(|&(id, variable)| CounterfactualSymbol {
                                id: CounterfactualSymbolId::new(id),
                                variable: CounterfactualVariableIndex::new(variable),
                            })
                            .collect(),
                        factors,
                    })));
                }
                Some(mask) => search.note(format!(
                    "the single-world factor over variables {} is not derivable from the \
                     evidence",
                    mask_text(problem, mask)
                )),
            }
        }
        Factored::Conflict(Conflict::Copies { variable }) => search.note(format!(
            "variable {} appears with two different readings in one district: the joint of \
             its worlds is a cross-world law",
            problem.variables[variable].raw()
        )),
        Factored::Conflict(Conflict::Inputs { variable }) => search.note(format!(
            "variable {} is read at two values by one district: the factor is not a \
             single-world interventional law",
            problem.variables[variable].raw()
        )),
    }
    {
        for (&(world, variable), &level) in atoms {
            if problem.levels[variable].len() != 2
                || used.contains(&(world, variable))
                || worlds[world].fixed.contains_key(&variable)
            {
                continue;
            }
            if depth >= ctx.complement_depth.get() {
                ctx.complement_pruned.set(true);
                continue;
            }
            let mut whole = atoms.clone();
            whole.remove(&(world, variable));
            let mut flipped = atoms.clone();
            flipped.insert((world, variable), 1 - level);
            let mut used_next = used.clone();
            used_next.insert((world, variable));
            let Some(a) = derive_conjunction(ctx, search, worlds, &whole, depth + 1, &used_next)?
            else {
                continue;
            };
            let Some(b) = derive_conjunction(ctx, search, worlds, &flipped, depth + 1, &used_next)?
            else {
                continue;
            };
            return Ok(Some(CounterfactualFunctional::Complement {
                variable: CounterfactualVariableIndex::new(variable),
                whole: Box::new(a),
                flipped: Box::new(b),
            }));
        }
    }
    Ok(None)
}

/// The derived plan of a whole query.
pub(super) struct QueryPlan {
    pub(super) numerator: CounterfactualFunctional,
    pub(super) denominator: CounterfactualFunctional,
    pub(super) guard: Option<CounterfactualFunctional>,
    pub(super) checked: bool,
    pub(super) moved: Vec<CounterfactualMovedAtom>,
    pub(super) rules: Vec<String>,
}

fn one() -> CounterfactualFunctional {
    CounterfactualFunctional::Sum(CounterfactualSum { symbols: Vec::new(), factors: Vec::new() })
}

pub(super) fn merged(
    event: &BTreeMap<(usize, usize), usize>,
    given: &BTreeMap<(usize, usize), usize>,
) -> Option<BTreeMap<(usize, usize), usize>> {
    let mut out = event.clone();
    for (&key, &level) in given {
        match out.insert(key, level) {
            Some(existing) if existing != level => return None,
            _ => {}
        }
    }
    Some(out)
}

/// A reduced conditional query: worlds, event, and the atoms moved into interventions.
type ReducedConditioning =
    (Vec<WorldIdx>, BTreeMap<(usize, usize), usize>, Vec<CounterfactualMovedAtom>);

/// The conditional reduction: repeatedly move a conditioning atom into an
/// intervention (in every world its node appears in) when the event's nodes are
/// m-separated from it given the other conditioning nodes.
fn reduce_conditioning(
    problem: &CounterfactualTransportProblem,
    worlds: &[WorldIdx],
    event: &BTreeMap<(usize, usize), usize>,
    given: &BTreeMap<(usize, usize), usize>,
) -> Result<ReducedConditioning, IdentificationError> {
    let mut worlds: Vec<WorldIdx> = worlds.to_vec();
    let mut given = given.clone();
    let mut moved: Vec<CounterfactualMovedAtom> = Vec::new();
    loop {
        if given.is_empty() {
            break;
        }
        let Some(all) = merged(event, &given) else { break };
        let Built::Net(net) = build_network(problem, &worlds, &all)? else {
            break;
        };
        let event_nodes: Vec<usize> = event
            .keys()
            .filter_map(|&(w, v)| net.pair_node.get(w).and_then(|row| row[v]))
            .collect();
        if event_nodes.is_empty() {
            break;
        }
        let mut chosen: Option<((usize, usize), usize)> = None;
        for (&(world, variable), &level) in &given {
            let Some(node) = net.pair_node.get(world).and_then(|row| row[variable]) else {
                continue;
            };
            if event_nodes.contains(&node) {
                continue;
            }
            let others: Vec<usize> = given
                .keys()
                .filter(|&&key| key != (world, variable))
                .filter_map(|&(w, v)| net.pair_node.get(w).and_then(|row| row[v]))
                .filter(|&n| n != node)
                .collect();
            if others.iter().any(|n| event_nodes.contains(n)) {
                continue;
            }
            if separated(problem, &net, &event_nodes, &[node], &others) {
                chosen = Some(((world, variable), level));
                break;
            }
        }
        let Some(((world, variable), level)) = chosen else { break };
        let node = net.pair_node[world][variable];
        for (w, row) in net.pair_node.iter().enumerate() {
            for (v, id) in row.iter().enumerate() {
                if *id == node && node.is_some() {
                    worlds[w].fixed.insert(v, level);
                    moved.push(CounterfactualMovedAtom {
                        world: w,
                        variable: CounterfactualVariableIndex::new(v),
                        level: CounterfactualLevelIndex::new(level),
                    });
                }
            }
        }
        given.remove(&(world, variable));
        // Remaining atoms that now sit on an intervened pair are consumed by the
        // network build (they are true or contradictory).
        given.retain(|&(w, v), l| worlds[w].fixed.get(&v).is_none_or(|fixed| *fixed != *l));
    }
    moved.sort_by_key(|m| (m.world, m.variable));
    moved.dedup();
    Ok((worlds, given, moved))
}

/// [`derive_query_within`] with the complement nesting deepened on demand: a failure that cut
/// off a complement is retried one level deeper, so the nesting is bounded by the search
/// budget (an `Exhausted` stop) and never by a constant.
pub(super) fn derive_query(
    ctx: &DeriveCtx<'_>,
    search: &mut Search<'_>,
    validated: &Validated,
) -> Result<Option<QueryPlan>, Abort> {
    ctx.complement_depth.set(INITIAL_COMPLEMENT_DEPTH);
    loop {
        ctx.complement_pruned.set(false);
        let plan = derive_query_within(ctx, search, validated)?;
        if plan.is_some() || !ctx.complement_pruned.get() {
            return Ok(plan);
        }
        ctx.complement_depth.set(ctx.complement_depth.get() + 1);
    }
}

fn derive_query_within(
    ctx: &DeriveCtx<'_>,
    search: &mut Search<'_>,
    validated: &Validated,
) -> Result<Option<QueryPlan>, Abort> {
    let problem = ctx.problem;
    let event = atoms_map(&validated.event);
    let given = atoms_map(&validated.given)
        .ok_or_else(|| Abort::Error(invalid_query("the conditioning event is contradictory")))?;
    let Some(event) = event else {
        // The event contradicts itself: probability zero given any possible conditioning.
        return Ok(Some(QueryPlan {
            numerator: CounterfactualFunctional::Zero,
            denominator: one(),
            guard: None,
            checked: given.is_empty(),
            moved: Vec::new(),
            rules: vec!["event.contradiction".to_owned()],
        }));
    };
    let used = BTreeSet::new();
    // Variant A: reduced conditioning (only when something moves).
    if !given.is_empty() {
        let (worlds, reduced_given, moved) =
            reduce_conditioning(problem, &validated.worlds, &event, &given)?;
        if !moved.is_empty() {
            let numerator_atoms = merged(&event, &reduced_given);
            let numerator = match numerator_atoms {
                Some(atoms) => derive_conjunction(ctx, search, &worlds, &atoms, 0, &used)?,
                None => Some(CounterfactualFunctional::Zero),
            };
            let denominator = if reduced_given.is_empty() {
                Some(one())
            } else {
                derive_conjunction(ctx, search, &worlds, &reduced_given, 0, &used)?
            };
            if let (Some(numerator), Some(denominator)) = (numerator, denominator) {
                let guard = derive_conjunction(ctx, search, &validated.worlds, &given, 0, &used)?;
                let checked = guard.is_some();
                return Ok(Some(QueryPlan {
                    numerator,
                    denominator,
                    guard,
                    checked,
                    moved,
                    rules: vec!["conditioning.reduction".to_owned()],
                }));
            }
        }
    }
    // Variant B: the ratio of the two original conjunctions.
    let numerator = match merged(&event, &given) {
        Some(atoms) => derive_conjunction(ctx, search, &validated.worlds, &atoms, 0, &used)?,
        None => Some(CounterfactualFunctional::Zero),
    };
    let Some(numerator) = numerator else {
        return Ok(None);
    };
    let denominator = if given.is_empty() {
        one()
    } else {
        let Some(den) = derive_conjunction(ctx, search, &validated.worlds, &given, 0, &used)?
        else {
            return Ok(None);
        };
        den
    };
    Ok(Some(QueryPlan {
        numerator,
        denominator,
        guard: None,
        checked: true,
        moved: Vec::new(),
        rules: vec!["conditioning.ratio".to_owned()],
    }))
}

pub(super) fn collect_items(functional: &CounterfactualFunctional, out: &mut Vec<usize>) {
    match functional {
        CounterfactualFunctional::Zero => {}
        CounterfactualFunctional::Sum(sum) => out.extend(sum.factors.iter().map(|f| f.item)),
        CounterfactualFunctional::Complement { whole, flipped, .. } => {
            collect_items(whole, out);
            collect_items(flipped, out);
        }
    }
}

pub(super) fn remap_items(functional: &mut CounterfactualFunctional, map: &BTreeMap<usize, usize>) {
    match functional {
        CounterfactualFunctional::Zero => {}
        CounterfactualFunctional::Sum(sum) => {
            for factor in &mut sum.factors {
                if let Some(&new) = map.get(&factor.item) {
                    factor.item = new;
                }
            }
        }
        CounterfactualFunctional::Complement { whole, flipped, .. } => {
            remap_items(whole, map);
            remap_items(flipped, map);
        }
    }
}
