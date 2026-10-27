//! Checker items of the parent module.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::super::{
    Admg, Arc, CausalExprArena, ExprId, IdentificationError, SelectionDiagram, VariableId,
};
use super::super::{
    BTreeSet, EvidenceCatalog, LeafBinding, MixedRule, MixedSourceDerivation, MixedSourceQuery,
    MixedStep, build_expression, invalid_derivation, invalid_query, validate_mixed_source_query,
};

use super::*;

/// Whether `a` and `b` are connected in the augmented graph of the ancestors of
/// `{a, b} ∪ cond` once the conditioned nodes are removed: adjacent nodes, and
/// every pair inside a bidirected district together with that district's parents.
pub(super) fn connected(
    parents: &[Vec<usize>],
    siblings: &[Vec<usize>],
    cond: &Set,
    a: usize,
    b: usize,
) -> bool {
    let n = parents.len();
    let mut ancestral = vec![false; n];
    let mut pending = cond.iter().copied().chain([a, b]).collect::<Vec<_>>();
    for &i in &pending {
        ancestral[i] = true;
    }
    while let Some(node) = pending.pop() {
        for &parent in &parents[node] {
            if !ancestral[parent] {
                ancestral[parent] = true;
                pending.push(parent);
            }
        }
    }
    let mut adjacent = vec![vec![false; n]; n];
    let mut link = |p: usize, q: usize| {
        if p != q {
            adjacent[p][q] = true;
            adjacent[q][p] = true;
        }
    };
    for node in (0..n).filter(|&i| ancestral[i]) {
        for &parent in &parents[node] {
            link(parent, node);
        }
        for &sibling in &siblings[node] {
            if ancestral[sibling] {
                link(node, sibling);
            }
        }
    }
    let mut assigned = vec![false; n];
    for seed in (0..n).filter(|&i| ancestral[i]) {
        if assigned[seed] {
            continue;
        }
        let mut district = vec![seed];
        assigned[seed] = true;
        let mut cursor = 0;
        while cursor < district.len() {
            let node = district[cursor];
            cursor += 1;
            for &sibling in &siblings[node] {
                if ancestral[sibling] && !assigned[sibling] {
                    assigned[sibling] = true;
                    district.push(sibling);
                }
            }
        }
        let mut clique = district.clone();
        for &member in &district {
            clique.extend(parents[member].iter().copied());
        }
        clique.sort_unstable();
        clique.dedup();
        for (i, &p) in clique.iter().enumerate() {
            for &q in &clique[i + 1..] {
                link(p, q);
            }
        }
    }
    let mut seen = vec![false; n];
    let mut stack = vec![a];
    seen[a] = true;
    while let Some(node) = stack.pop() {
        if node == b {
            return true;
        }
        for (next, visited) in seen.iter_mut().enumerate() {
            if adjacent[node][next] && !*visited && (!cond.contains(&next) || next == b) {
                *visited = true;
                stack.push(next);
            }
        }
    }
    false
}

/// Whether the do-calculus rule `rule` moves the premise quantity
/// `P(y | do(d), z)` to `conclusion` through the variables `w`, deciding every
/// side condition with [`Checker`]. `free` are the variables the premise's
/// expression still mentions: a deletion may not drop one. Rules that are not
/// one-premise do-calculus rules never hold here.
pub(super) fn unary_rule_holds(
    checker: &Checker,
    universe: &Set,
    rule: MixedRule,
    (py, pd, pz): (&Set, &Set, &Set),
    (y, d, z): (&Set, &Set, &Set),
    w: &Set,
    free: &Set,
) -> bool {
    let union = |sets: &[&Set]| sets.iter().flat_map(|s| s.iter().copied()).collect::<Set>();
    let minus = |a: &Set, b: &Set| a.difference(b).copied().collect::<Set>();
    if w.is_empty() || py != y {
        return false;
    }
    match rule {
        MixedRule::Rule1Insert => {
            w.is_subset(&outside_of(universe, py, pd, pz))
                && pd == d
                && *z == union(&[pz, w])
                && checker.separated(pd, &Set::new(), py, w, &union(&[pd, pz]))
        }
        MixedRule::Rule1Delete => {
            w.is_subset(pz)
                && w.is_disjoint(free)
                && pd == d
                && *z == minus(pz, w)
                && checker.separated(pd, &Set::new(), py, w, &union(&[pd, z]))
        }
        MixedRule::Rule2ToDo => {
            w.is_subset(pz)
                && *d == union(&[pd, w])
                && *z == minus(pz, w)
                && checker.separated(pd, w, py, w, &union(&[pd, z]))
        }
        MixedRule::Rule2ToObservation => {
            w.is_subset(pd)
                && *d == minus(pd, w)
                && *z == union(&[pz, w])
                && checker.separated(d, w, py, w, &union(&[d, pz]))
        }
        MixedRule::Rule3Insert => {
            // The inserted actions that are ancestors of the observed set in the
            // graph without the edges into the existing actions keep their edges
            // (Pearl's `W(Z)` exception): only the other ones are cut.
            let past = checker.ancestors(pz, pd);
            let exposed = union(&[pd, &minus(w, &past)]);
            w.is_subset(&outside_of(universe, py, pd, pz))
                && *d == union(&[pd, w])
                && z == pz
                && checker.separated(&exposed, &Set::new(), py, w, &union(&[pd, pz]))
        }
        MixedRule::Rule3Delete => {
            let past = checker.ancestors(pz, d);
            let exposed = union(&[d, &minus(w, &past)]);
            w.is_subset(pd)
                && w.is_disjoint(free)
                && *d == minus(pd, w)
                && z == pz
                && checker.separated(&exposed, &Set::new(), py, w, &union(&[d, pz]))
        }
        _ => false,
    }
}

/// Check every step of a proof, independently of the search that produced it.
///
/// Each step's quantity is recomputed from its rule and premises, every
/// do-calculus side condition is decided by [`Checker`], a deletion may not drop a
/// variable the premise's expression still mentions, each expression is rebuilt in
/// a fresh arena and must equal the stored one, each expression's free variables
/// stay inside the quantity, and every input step must name an eligible catalog
/// distribution with its recorded source identity. The last step must answer the
/// query. A relaxed (missing-joint) input is accepted only when `allow_relaxed`.
#[allow(clippy::too_many_lines, reason = "One pass over the ten rules of the frozen set")]
pub(super) fn check_steps(
    graph: &Admg,
    query: &MixedSourceQuery,
    catalog: &EvidenceCatalog,
    steps: &[MixedStep],
    arena: Option<&CausalExprArena>,
    allow_relaxed: bool,
) -> Result<(), IdentificationError> {
    let checker = Checker::new(graph)?;
    let validated = validate_mixed_source_query(graph, query, catalog)?;
    let target = query.target.as_ref();
    let universe = (0..checker.variables.len()).collect::<Set>();
    let mut fresh = CausalExprArena::new();
    let mut quantities: Vec<(Set, Set, Set)> = Vec::with_capacity(steps.len());
    let mut expressions: Vec<ExprId> = Vec::with_capacity(steps.len());
    for (index, step) in steps.iter().enumerate() {
        let (y, d, z) = (
            checker.set(&step.y)?,
            checker.set(&step.intervened)?,
            checker.set(&step.conditioned)?,
        );
        let w = checker.set(&step.params)?;
        let disjoint = y.is_disjoint(&d) && y.is_disjoint(&z) && d.is_disjoint(&z);
        if !disjoint
            || y.is_empty()
            || step.premises.iter().any(|p| *p >= index)
            || (step.rule != MixedRule::Input && (step.source.is_some() || step.relaxed))
            || (step.rule == MixedRule::Product && !w.is_empty())
        {
            return Err(invalid_derivation());
        }
        let premise = |k: usize| step.premises.get(k).map(|p| &quantities[*p]);
        let free_of = |k: usize| -> Result<Set, IdentificationError> {
            let expression = expressions[*step.premises.get(k).ok_or_else(invalid_derivation)?];
            Ok(fresh
                .free_variables(expression)
                .iter()
                .filter_map(|v| checker.variables.iter().position(|x| x == v))
                .collect())
        };
        let union = |sets: &[&Set]| sets.iter().flat_map(|s| s.iter().copied()).collect::<Set>();
        let expected: Option<(Set, Set, Set)> = match (step.rule, step.premises.len()) {
            (MixedRule::Input, 0) => {
                let leaf = step.source.as_ref().ok_or_else(invalid_derivation)?;
                let input = validated
                    .inputs
                    .iter()
                    .chain(validated.marginal_inputs.iter())
                    .find(|input| input.regime == leaf.regime)
                    .ok_or_else(invalid_derivation)?;
                let relaxed = !input.marginals.is_empty();
                let same = checker.set(&input.measured)? == y
                    && checker.set(&input.intervened)? == d
                    && z.is_empty()
                    && w.is_empty()
                    && leaf.study == input.study
                    && leaf.identity == input.identity
                    && leaf.snapshot == input.snapshot
                    && leaf.population.as_ref() == target
                    && step.relaxed == relaxed
                    && (allow_relaxed || !relaxed);
                same.then(|| (y.clone(), d.clone(), z.clone()))
            }
            (MixedRule::Marginalize, 1) => {
                let (py, pd, pz) = premise(0).ok_or_else(invalid_derivation)?;
                (!w.is_empty()
                    && w.is_subset(py)
                    && &union(&[&y, &w]) == py
                    && pd == &d
                    && pz == &z)
                    .then(|| (y.clone(), d.clone(), z.clone()))
            }
            (MixedRule::Condition, 1) => {
                let (py, pd, pz) = premise(0).ok_or_else(invalid_derivation)?;
                (!w.is_empty()
                    && w.is_subset(py)
                    && &union(&[&y, &w]) == py
                    && pd == &d
                    && z == union(&[pz, &w]))
                .then(|| (y.clone(), d.clone(), z.clone()))
            }
            (MixedRule::Product, 2) => {
                let (ay, ad, az) = premise(0).ok_or_else(invalid_derivation)?;
                let (by, bd, bz) = premise(1).ok_or_else(invalid_derivation)?;
                (ad == bd
                    && ad == &d
                    && az == &union(&[bz, by])
                    && ay.is_disjoint(by)
                    && bz == &z
                    && y == union(&[ay, by]))
                .then(|| (y.clone(), d.clone(), z.clone()))
            }
            (rule, 1) => {
                let (py, pd, pz) = premise(0).ok_or_else(invalid_derivation)?;
                if w.is_empty() || py != &y {
                    return Err(invalid_derivation());
                }
                let free = free_of(0)?;
                unary_rule_holds(&checker, &universe, rule, (py, pd, pz), (&y, &d, &z), &w, &free)
                    .then(|| (y.clone(), d.clone(), z.clone()))
            }
            _ => None,
        };
        if expected.is_none() {
            return Err(invalid_derivation());
        }
        let premise_expressions = step.premises.iter().map(|p| expressions[*p]).collect::<Vec<_>>();
        let expression = build_expression(&mut fresh, step, &premise_expressions, target)?;
        if arena.is_some() && expression != step.expression {
            return Err(invalid_derivation());
        }
        let inside = union(&[&y, &d, &z]);
        let free = fresh
            .free_variables(expression)
            .iter()
            .map(|v| checker.variables.iter().position(|x| x == v).ok_or_else(invalid_derivation))
            .collect::<Result<Set, _>>()?;
        if !free.is_subset(&inside) {
            return Err(invalid_derivation());
        }
        quantities.push((y, d, z));
        expressions.push(expression);
    }
    let last = quantities.last().ok_or_else(invalid_derivation)?;
    let goal = (checker.set(&query.outcomes)?, checker.set(&query.treatments)?, Set::new());
    if *last != goal || arena.is_some_and(|stored| *stored != fresh) {
        return Err(invalid_derivation());
    }
    check_with_shared_expression_layer(&fresh, steps, &expressions)
}

/// The shared expression layer's own checks of the rebuilt root: the shared
/// compiler must accept it (well-formed nodes, resolvable references), and its
/// distribution leaves, read by the shared scope analysis, must be exactly the
/// `(population, regime)` pairs of the input steps the proof reaches from its
/// last step. A leaf the proof does not cite, or a cited leaf the expression
/// dropped, fails here even though the step-by-step replay passed.
pub(super) fn check_with_shared_expression_layer(
    fresh: &CausalExprArena,
    steps: &[MixedStep],
    expressions: &[ExprId],
) -> Result<(), IdentificationError> {
    let root = *expressions.last().ok_or_else(invalid_derivation)?;
    fresh.compile(root).map_err(|_| invalid_derivation())?;
    let mut reached = BTreeSet::new();
    let mut pending = vec![steps.len() - 1];
    while let Some(index) = pending.pop() {
        if reached.insert(index) {
            pending.extend(steps[index].premises.iter().copied());
        }
    }
    let expected = reached
        .into_iter()
        .filter_map(|index| steps[index].source.as_ref())
        .map(|leaf| LeafBinding {
            population: Arc::clone(&leaf.population),
            regime: Some(leaf.regime),
        })
        .collect::<Vec<_>>();
    fresh.bind_certificate(root, &expected).map_err(|_| invalid_derivation())
}

fn outside_of(universe: &Set, y: &Set, d: &Set, z: &Set) -> Set {
    universe
        .iter()
        .copied()
        .filter(|v| !y.contains(v) && !d.contains(v) && !z.contains(v))
        .collect()
}

/// Independently check a derivation against a graph and catalog.
///
/// # Errors
/// `mixed_search.invalid_derivation` when any step, expression or source binding
/// fails its check; `mixed_search.invalid_query` for a malformed query or catalog.
pub fn verify_mixed_source_derivation(
    graph: &Admg,
    derivation: &MixedSourceDerivation,
    catalog: &EvidenceCatalog,
) -> Result<(), IdentificationError> {
    let shared = SelectionDiagram::try_new(graph.clone(), Arc::<[VariableId]>::from([]))
        .map_err(|error| invalid_query(&error))?;
    if super::super::super::graph_signature(&shared) != derivation.graph_signature
        || derivation.query != derivation.query.canonical()
        || derivation.root_step + 1 != derivation.steps.len()
    {
        return Err(invalid_derivation());
    }
    check_steps(
        graph,
        &derivation.query,
        catalog,
        &derivation.steps,
        Some(&derivation.arena),
        false,
    )
}
