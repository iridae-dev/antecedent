//! The potential-outcome network of one counterfactual conjunction.
//!
//! A conjunction of atoms `(world, variable, level)` over worlds that carry hard
//! interventions and edge routes (the nested counterfactual `Y_{x, M_{x'}}` is a
//! route) is unnested into a network of *nodes*. A node is one variable together
//! with the values or nodes its parents are read from; two `(world, variable)`
//! pairs with the same variable and the same parent readings are one node, because
//! one mechanism applied to one input under one exogenous state is one value
//! (the merge of Shpitser and Pearl's counterfactual graph, made by hash-consing).
//!
//! Value consistency. On the event, a node whose level the conjunction fixes is a
//! constant for every reader, so readers of it read that level instead of the
//! node. This is the consistency axiom (`Y_{X=x} = Y` when `X = x` is part of the
//! same conjunction) and is what lets two pairs with different upstream worlds
//! merge. The substitution changes only which node a reader points at, never the
//! event, so it is applied to a fixed point. Two pairs that merge but are fixed
//! to different levels make the conjunction impossible (probability zero).
//!
//! The kept network is the ancestral closure of the fixed nodes. Nodes that no
//! fixed node reads are summed out of the joint and drop out.
//!
//! Soundness of what this module returns: every node is a deterministic function
//! of its inputs and the exogenous state of its variable's district, with the same
//! function in every world, so the joint of all nodes is that of a semi-Markovian
//! model whose graph is [`Network`]; the global Markov property of that graph is
//! used by [`separated`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use super::CounterfactualTransportProblem as Problem;
use super::COUNTERFACTUAL_TRANSPORT_MAX_NODES;
use crate::error::IdentificationError;

/// One world of a query in internal indices: hard interventions and the edges
/// whose child reads its parent from another world.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct WorldIdx {
    /// `variable -> level index` set by `do(.)`.
    pub(super) fixed: BTreeMap<usize, usize>,
    /// `(parent, child) -> world` the child reads the parent from.
    pub(super) routes: BTreeMap<(usize, usize), usize>,
}

/// What a node reads for one parent: a constant level or another node.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum Input {
    /// A level index, fixed by an intervention or by the conjunction.
    Const(usize),
    /// A node of the network that is not fixed.
    Node(usize),
}

/// One node of the kept network.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct NetNode {
    /// The node's variable (internal index).
    pub(super) variable: usize,
    /// Parent readings in ascending parent index.
    pub(super) inputs: Vec<(usize, Input)>,
    /// The level the conjunction fixes the node at, if any.
    pub(super) value: Option<usize>,
}

/// The kept network, canonically ordered: variables in topological order, nodes
/// of one variable by their (remapped) readings.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct Network {
    /// Nodes in canonical order.
    pub(super) nodes: Vec<NetNode>,
    /// `[world][variable] ->` kept node, `None` when the pair is a constant of its
    /// world or its node was pruned.
    pub(super) pair_node: Vec<Vec<Option<usize>>>,
}

/// Outcome of building a conjunction's network.
pub(super) enum Built {
    /// The conjunction contradicts itself or an intervention.
    Zero,
    /// The kept network (possibly empty: probability one).
    Net(Network),
}

/// Raw, unpruned nodes of one construction pass.
struct Raw {
    nodes: Vec<(usize, Vec<(usize, Input)>)>,
    pair_node: Vec<Vec<Option<usize>>>,
}

fn invariant(message: &'static str) -> IdentificationError {
    IdentificationError::InvariantViolated { message }
}

/// Build the network of the conjunction `atoms` (`(world, variable) -> level`)
/// over `worlds`.
///
/// # Errors
///
/// `counterfactual_transport.bounds_exceeded` when the kept network is larger
/// than the contract's node bound, or an internal invariant violation.
pub(super) fn build_network(
    problem: &Problem,
    worlds: &[WorldIdx],
    atoms: &BTreeMap<(usize, usize), usize>,
) -> Result<Built, IdentificationError> {
    let n = problem.variables.len();
    let w = worlds.len();
    let mut known: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    for (&(world, variable), &level) in atoms {
        let Some(row) = worlds.get(world) else {
            return Err(invariant("an atom names a world that does not exist"));
        };
        if let Some(&fixed) = row.fixed.get(&variable) {
            if fixed != level {
                return Ok(Built::Zero);
            }
            continue;
        }
        known.insert((world, variable), level);
    }
    loop {
        let raw = construct(problem, worlds, &known)?;
        let mut node_value: Vec<Option<usize>> = vec![None; raw.nodes.len()];
        for (&(world, variable), &level) in &known {
            if let Some(id) = raw.pair_node[world][variable] {
                match node_value[id] {
                    None => node_value[id] = Some(level),
                    Some(existing) if existing != level => return Ok(Built::Zero),
                    Some(_) => {}
                }
            }
        }
        let mut additions: Vec<((usize, usize), usize)> = Vec::new();
        for world in 0..w {
            for variable in 0..n {
                if let Some(id) = raw.pair_node[world][variable] {
                    if let Some(level) = node_value[id] {
                        if !known.contains_key(&(world, variable)) {
                            additions.push(((world, variable), level));
                        }
                    }
                }
            }
        }
        if additions.is_empty() {
            return finalize(problem, &raw, &node_value).map(Built::Net);
        }
        known.extend(additions);
    }
}

/// One construction pass: hash-cons the nodes of every `(world, variable)` pair,
/// readers of a pair in `known` reading its level.
fn construct(
    problem: &Problem,
    worlds: &[WorldIdx],
    known: &BTreeMap<(usize, usize), usize>,
) -> Result<Raw, IdentificationError> {
    let n = problem.variables.len();
    let w = worlds.len();
    let mut pair_node: Vec<Vec<Option<usize>>> = vec![vec![None; n]; w];
    let mut reading: Vec<Vec<Option<Input>>> = vec![vec![None; n]; w];
    let mut nodes: Vec<(usize, Vec<(usize, Input)>)> = Vec::new();
    let mut index: BTreeMap<(usize, Vec<(usize, Input)>), usize> = BTreeMap::new();
    for &variable in &problem.topo {
        for world in 0..w {
            if let Some(&level) = worlds[world].fixed.get(&variable) {
                reading[world][variable] = Some(Input::Const(level));
                continue;
            }
            let mut inputs: Vec<(usize, Input)> = Vec::with_capacity(problem.parents[variable].len());
            for &parent in &problem.parents[variable] {
                let source = worlds[world].routes.get(&(parent, variable)).copied().unwrap_or(world);
                let input = reading
                    .get(source)
                    .and_then(|row| row[parent])
                    .ok_or_else(|| invariant("a parent is read before it is built"))?;
                inputs.push((parent, input));
            }
            let key = (variable, inputs);
            let id = if let Some(&id) = index.get(&key) {
                id
            } else {
                let id = nodes.len();
                nodes.push(key.clone());
                index.insert(key, id);
                id
            };
            pair_node[world][variable] = Some(id);
            reading[world][variable] = Some(match known.get(&(world, variable)) {
                Some(&level) => Input::Const(level),
                None => Input::Node(id),
            });
        }
    }
    Ok(Raw { nodes, pair_node })
}

/// Prune to the ancestral closure of the fixed nodes and order canonically.
fn finalize(
    problem: &Problem,
    raw: &Raw,
    node_value: &[Option<usize>],
) -> Result<Network, IdentificationError> {
    let m = raw.nodes.len();
    let mut keep = vec![false; m];
    let mut stack: Vec<usize> = (0..m).filter(|&i| node_value[i].is_some()).collect();
    for &i in &stack {
        keep[i] = true;
    }
    while let Some(i) = stack.pop() {
        for &(_, input) in &raw.nodes[i].1 {
            if let Input::Node(j) = input {
                if !keep[j] {
                    keep[j] = true;
                    stack.push(j);
                }
            }
        }
    }
    if keep.iter().filter(|k| **k).count() > COUNTERFACTUAL_TRANSPORT_MAX_NODES {
        return Err(IdentificationError::UnsupportedInput {
            code: super::BOUNDS_EXCEEDED,
        });
    }
    let mut rank: Vec<Option<usize>> = vec![None; m];
    let mut ordered: Vec<NetNode> = Vec::new();
    for &variable in &problem.topo {
        let mut group: Vec<(Vec<(usize, Input)>, usize)> = Vec::new();
        for i in 0..m {
            if keep[i] && raw.nodes[i].0 == variable {
                let mut mapped = Vec::with_capacity(raw.nodes[i].1.len());
                for &(parent, input) in &raw.nodes[i].1 {
                    let input = match input {
                        Input::Const(c) => Input::Const(c),
                        Input::Node(j) => Input::Node(
                            rank[j].ok_or_else(|| invariant("a kept node reads an unranked node"))?,
                        ),
                    };
                    mapped.push((parent, input));
                }
                group.push((mapped, i));
            }
        }
        group.sort();
        for (mapped, i) in group {
            rank[i] = Some(ordered.len());
            ordered.push(NetNode { variable, inputs: mapped, value: node_value[i] });
        }
    }
    let pair_node: Vec<Vec<Option<usize>>> = raw
        .pair_node
        .iter()
        .map(|row| row.iter().map(|id| id.and_then(|id| rank[id])).collect::<Vec<_>>())
        .collect();
    Ok(Network { nodes: ordered, pair_node })
}

// ---------------------------------------------------------------------------
// Districts and the single-world factors of the network's joint law.
// ---------------------------------------------------------------------------

/// A level argument of a factor: a declared level index or a summation symbol
/// (the node whose value is summed).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum Level {
    /// A declared level index.
    Fixed(usize),
    /// The free node with this index in the network.
    Symbol(usize),
}

/// One district's factor `Q[S](outputs | parents)`: a single-world interventional
/// law of the variable set `S`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Term {
    /// Bit set of the district's variables.
    pub(super) mask: u32,
    /// `(variable, level)` of every member, ascending variable.
    pub(super) outputs: Vec<(usize, Level)>,
    /// `(variable, level)` of every parent outside the district that a member
    /// reads, ascending variable.
    pub(super) parents: Vec<(usize, Level)>,
}

/// Why a network's joint law is not a product of single-world factors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Conflict {
    /// Two nodes of one variable with different readings share a district: their
    /// joint is a cross-world law no single experiment contains.
    Copies {
        /// The variable (internal index).
        variable: usize,
    },
    /// Members of one district read one parent at two values, or read a member of
    /// their own district at a value other than the one the member is fixed to.
    Inputs {
        /// The variable (internal index).
        variable: usize,
    },
}

/// The factorization of a network.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Factored {
    /// A product of single-world factors summed over the free nodes.
    Terms {
        /// `(node index, variable)` of every free node.
        symbols: Vec<(usize, usize)>,
        /// One term per district, ascending least node.
        terms: Vec<Term>,
    },
    /// The joint law needs a cross-world factor.
    Conflict(Conflict),
}

/// Split the network into districts and read each as a single-world factor.
///
/// Two nodes share a district when they are copies of one variable (one
/// exogenous term) or their variables are joined by a bidirected edge. The joint
/// of the nodes is the product over districts of the district's law given the
/// parents outside it, because districts read disjoint exogenous terms. The
/// district's law is a single-world interventional law `P(S | do(parents))`
/// exactly when it has one node per variable and every member reads each outside
/// parent at one value; anything else is a [`Conflict`].
pub(super) fn factorize(problem: &Problem, net: &Network) -> Factored {
    let m = net.nodes.len();
    let mut parent: Vec<usize> = (0..m).collect();
    fn find(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    for a in 0..m {
        for b in (a + 1)..m {
            let (va, vb) = (net.nodes[a].variable, net.nodes[b].variable);
            if va == vb || problem.bidirected_adjacent(va, vb) {
                let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
                if ra != rb {
                    parent[ra.max(rb)] = ra.min(rb);
                }
            }
        }
    }
    let mut components: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for i in 0..m {
        let root = find(&mut parent, i);
        components.entry(root).or_default().push(i);
    }
    let symbols: Vec<(usize, usize)> = net
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| node.value.is_none())
        .map(|(i, node)| (i, node.variable))
        .collect();
    let mut terms = Vec::with_capacity(components.len());
    for members in components.values() {
        let mut member_of: BTreeMap<usize, usize> = BTreeMap::new();
        for &i in members {
            if member_of.insert(net.nodes[i].variable, i).is_some() {
                return Factored::Conflict(Conflict::Copies { variable: net.nodes[i].variable });
            }
        }
        let mut mask = 0u32;
        let mut outputs: Vec<(usize, Level)> = Vec::with_capacity(members.len());
        let mut parents: BTreeMap<usize, Input> = BTreeMap::new();
        for &i in members {
            let node = &net.nodes[i];
            mask |= 1 << node.variable;
            outputs.push((
                node.variable,
                match node.value {
                    Some(level) => Level::Fixed(level),
                    None => Level::Symbol(i),
                },
            ));
            for &(p, input) in &node.inputs {
                if let Some(&member) = member_of.get(&p) {
                    let consistent = match input {
                        Input::Node(j) => j == member,
                        Input::Const(c) => net.nodes[member].value == Some(c),
                    };
                    if !consistent {
                        return Factored::Conflict(Conflict::Inputs { variable: p });
                    }
                } else {
                    match parents.get(&p) {
                        Some(&existing) if existing != input => {
                            return Factored::Conflict(Conflict::Inputs { variable: p });
                        }
                        Some(_) => {}
                        None => {
                            parents.insert(p, input);
                        }
                    }
                }
            }
        }
        outputs.sort_by_key(|(variable, _)| *variable);
        let parents = parents
            .into_iter()
            .map(|(p, input)| {
                (
                    p,
                    match input {
                        Input::Const(c) => Level::Fixed(c),
                        Input::Node(j) => Level::Symbol(j),
                    },
                )
            })
            .collect();
        terms.push(Term { mask, outputs, parents });
    }
    Factored::Terms { symbols, terms }
}

// ---------------------------------------------------------------------------
// Separation in the network graph.
// ---------------------------------------------------------------------------

/// Whether every node of `from` is m-separated from every node of `to` given
/// `given` in the network's graph: a directed edge `input -> node` for every
/// node reading another node, and a bidirected edge between nodes of one
/// variable or of variables joined by a bidirected edge of the problem.
///
/// The three node sets must be disjoint. The graph is acyclic, so the global
/// Markov property of the model whose graph this is makes m-separation imply
/// conditional independence of the nodes' joint law.
pub(super) fn separated(
    problem: &Problem,
    net: &Network,
    from: &[usize],
    to: &[usize],
    given: &[usize],
) -> bool {
    let m = net.nodes.len();
    let mut parents_of: Vec<Vec<usize>> = vec![Vec::new(); m];
    let mut children_of: Vec<Vec<usize>> = vec![Vec::new(); m];
    for (i, node) in net.nodes.iter().enumerate() {
        for &(_, input) in &node.inputs {
            if let Input::Node(j) = input {
                parents_of[i].push(j);
                children_of[j].push(i);
            }
        }
    }
    let mut spouses: Vec<Vec<usize>> = vec![Vec::new(); m];
    for a in 0..m {
        for b in (a + 1)..m {
            let (va, vb) = (net.nodes[a].variable, net.nodes[b].variable);
            if va == vb || problem.bidirected_adjacent(va, vb) {
                spouses[a].push(b);
                spouses[b].push(a);
            }
        }
    }
    let mut conditioned = vec![false; m];
    for &g in given {
        conditioned[g] = true;
    }
    // Ancestors of the conditioning set (inclusive).
    let mut ancestor = conditioned.clone();
    let mut stack: Vec<usize> = given.to_vec();
    while let Some(i) = stack.pop() {
        for &p in &parents_of[i] {
            if !ancestor[p] {
                ancestor[p] = true;
                stack.push(p);
            }
        }
    }
    let mut is_target = vec![false; m];
    for &t in to {
        is_target[t] = true;
    }
    // State: (node, arrived through an arrowhead at the node). Start nodes are
    // entered unconditioned with no arrowhead and impose no passing condition.
    let mut visited = vec![[false; 2]; m];
    let mut work: Vec<(usize, bool, bool)> = Vec::new();
    for &s in from {
        work.push((s, false, true));
    }
    while let Some((u, arrow_in, start)) = work.pop() {
        // Edges out of u: (neighbour, arrowhead at u, arrowhead at neighbour).
        let mut edges: Vec<(usize, bool, bool)> = Vec::new();
        for &c in &children_of[u] {
            edges.push((c, false, true));
        }
        for &p in &parents_of[u] {
            edges.push((p, true, false));
        }
        for &b in &spouses[u] {
            edges.push((b, true, true));
        }
        for (v, arrow_at_u, arrow_at_v) in edges {
            if !start {
                let collider = arrow_in && arrow_at_u;
                if collider {
                    if !ancestor[u] {
                        continue;
                    }
                } else if conditioned[u] {
                    continue;
                }
            }
            if is_target[v] {
                return false;
            }
            let slot = usize::from(arrow_at_v);
            if !visited[v][slot] {
                visited[v][slot] = true;
                work.push((v, arrow_at_v, false));
            }
        }
    }
    true
}
