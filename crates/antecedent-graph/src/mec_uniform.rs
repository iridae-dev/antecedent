//! Exact size of a Markov equivalence class, and exactly uniform draws of its members.
//!
//! The members of the class a [`Cpdag`] represents are the DAGs obtained by orienting each of
//! its undirected (chain) components acyclically without a new unshielded collider, the
//! components independently. Each component of an essential graph is chordal, so the class
//! size is the product over components of their numbers of acyclic moral orientations.
//!
//! One component is counted by clique-picking (Wienöbst, Bannach and Liśkiewicz, "Polynomial-
//! Time Algorithms for Counting and Sampling Markov Equivalent DAGs", AAAI 2021). Root a clique
//! tree of the component. Every acyclic moral orientation has a topological order that starts
//! with the vertices of a maximal clique `K` in some order `π`, and exactly one pair `(K, π)`
//! has the property that `π` does not begin with the vertex set of any separator on the path
//! from the root of the clique tree to `K`. Fixing `K` first in order `π` (every edge inside
//! `K` oriented by `π`, every edge leaving `K` oriented away from it) and closing under Meek's
//! orientation rules leaves undirected components that are again chordal, whose orientations
//! extend the fixed part independently and whose set does not depend on `π`. So
//!
//! `count(G) = Σ_K φ(K) · Π_H count(H)`,
//!
//! where `φ(K)` is the number of orders of `K` that avoid every forbidden prefix (computed by
//! inclusion–exclusion over the nested separators) and `H` ranges over the undirected
//! components left once `K` is fixed. Subproblems are memoized by their edge sets.
//!
//! A uniform member is drawn by the same recursion: a clique with probability proportional to
//! its term, an order of it uniformly among the orders that avoid the forbidden prefixes (by
//! rejection from uniform orders, which accepts at least one order in `|K|`), and each
//! remaining component independently. Draws read a `SplitMix64` stream and are reproduced
//! exactly by its seed. Every drawn DAG is checked to be a member of the class.
//!
//! The count is exact in `u128`; a class with more members is refused rather than rounded.
//! Counting is charged to a [`SearchBudget`]: one operation per clique of every subproblem,
//! and one per undirected edge Meek's rules examine while closing the orientation the clique
//! fixes, so the charge grows with the closure work. A clique or a star charges one operation
//! per clique; a path of `n` vertices, whose closure orients one edge per pass, about `n^3 / 3`
//! (1,327,280 for 160 vertices), so the default completion budget of 4,000,000 operations holds
//! paths of a little over 200 vertices and a class sample's four-fold budget paths of about 360.
//!
//! The per-component product counts the members of an essential graph: a chain graph (no
//! directed cycle through its undirected components, no directed edge inside one) whose
//! components are chordal and with no directed edge `a -> b` next to an undirected `b - c`
//! whose ends are not adjacent. A graph that breaks any of these (a partially directed graph
//! closed under background knowledge can) constrains orientations across components, so it
//! is refused rather than counted.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, VecDeque};
use std::fmt;

use antecedent_core::{
    CanonicalWriter, ExecutionContext, SearchBudget, SearchLimits, SearchStop, derive_stream,
    splitmix64,
};

use crate::cpdag::Cpdag;
use crate::cpdag_completion::is_mec_member;
use crate::dag::Dag;
use crate::types::DenseNodeId;

/// A chain component: its nodes and its undirected edges.
type Component = (Vec<u32>, Vec<(u32, u32)>);

const DAG_MEMBER_TAG: &str = "antecedent.graph.mec_member.v1";

/// Most uniform orders drawn for one clique before the draw is abandoned. An order is accepted
/// with probability at least `1 / |K|`, so this bound is never reached in practice.
const MAX_ORDER_ATTEMPTS: usize = 1 << 20;

/// Why a Markov equivalence class could not be counted or sampled.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MecUniformError {
    /// The CPDAG holds an orientation-conflict (`x-x`) edge.
    ConflictEdge,
    /// An undirected component is not chordal, so the graph represents no Markov
    /// equivalence class.
    NotChordal,
    /// The graph is not the essential graph of a class: a directed edge `a -> b` meets an
    /// undirected edge `b - c` with `a` and `c` not adjacent, a directed edge joins two
    /// vertices of one undirected component, or the directed edges between undirected
    /// components form a cycle.
    NotEssential,
    /// The class has more members than `u128` holds.
    CountOverflow,
    /// The counting search stopped at a bound.
    Stopped(SearchStop),
    /// A drawn orientation was not a member of the class.
    MemberCheckFailed,
    /// No order of a clique avoiding its forbidden prefixes was drawn within the attempt bound.
    OrderNotDrawn,
}

impl fmt::Display for MecUniformError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConflictEdge => formatter.write_str("the CPDAG holds an orientation conflict"),
            Self::NotChordal => {
                formatter.write_str("an undirected component of the CPDAG is not chordal")
            }
            Self::NotEssential => formatter.write_str(
                "the graph is not an essential graph: a directed edge points into an undirected \
                 edge whose far end it does not touch, joins two vertices of one undirected \
                 component, or closes a directed cycle through the undirected components",
            ),
            Self::CountOverflow => formatter.write_str("the class has more members than u128"),
            Self::Stopped(stop) => write!(formatter, "the class count stopped: {}", stop.code()),
            Self::MemberCheckFailed => {
                formatter.write_str("a drawn orientation is not a member of the class")
            }
            Self::OrderNotDrawn => {
                formatter.write_str("no admissible clique order was drawn within the bound")
            }
        }
    }
}

impl std::error::Error for MecUniformError {}

/// The 64-bit content fingerprint of a DAG member: its node count and sorted directed edges.
/// Equal for equal graphs whatever order an enumerator or sampler produced them in.
#[must_use]
pub fn dag_fingerprint(dag: &Dag) -> u64 {
    let pairs = dag.directed_edge_pairs_sorted();
    let mut writer = CanonicalWriter::new(DAG_MEMBER_TAG);
    writer.word(dag.node_count() as u64).word(pairs.len() as u64);
    for (from, to) in pairs {
        writer.word(u64::from(from)).word(u64::from(to));
    }
    first_word(&writer.finish())
}

pub(crate) fn first_word(bytes: &[u8; 32]) -> u64 {
    let mut word = [0_u8; 8];
    word.copy_from_slice(&bytes[..8]);
    u64::from_le_bytes(word)
}

/// One maximal clique of a subproblem with what drawing it needs.
#[derive(Clone, Debug)]
struct CliqueChoice {
    /// Global vertex ids, ascending.
    clique: Vec<u32>,
    /// Separators on the path from the clique-tree root that are subsets of the clique, as
    /// global vertex ids, distinct, ordered by size.
    forbidden: Vec<Vec<u32>>,
    /// `φ(K) · Π count(H)`.
    weight: u128,
}

/// One connected chordal undirected subgraph and its count.
#[derive(Clone, Debug)]
struct Subproblem {
    vertices: Vec<u32>,
    edges: Vec<(u32, u32)>,
    count: u128,
    choices: Vec<CliqueChoice>,
}

/// The exact count of a Markov equivalence class and a uniform sampler of its members.
#[derive(Clone, Debug)]
pub struct MecUniform {
    cpdag: Cpdag,
    /// Arena index of every undirected component of the CPDAG with at least one edge.
    components: Vec<usize>,
    arena: Vec<Subproblem>,
    index: BTreeMap<Vec<(u32, u32)>, usize>,
    size: u128,
}

impl MecUniform {
    /// Count the class of `cpdag` exactly, charging the count to a budget under `limits` and
    /// `ctx`.
    ///
    /// # Errors
    ///
    /// A conflict edge, a non-chordal undirected component, a graph that is not an essential
    /// graph ([`MecUniformError::NotEssential`]), a count above `u128`, or a budget stop.
    pub fn new(
        cpdag: &Cpdag,
        limits: SearchLimits,
        ctx: &ExecutionContext,
    ) -> Result<Self, MecUniformError> {
        if cpdag.conflict_edge_count() > 0 {
            return Err(MecUniformError::ConflictEdge);
        }
        check_essential(cpdag)?;
        let budget = SearchBudget::new(limits, ctx)
            .map_err(|receipt| MecUniformError::Stopped(receipt.stop))?;
        let mut counter = Counter { arena: Vec::new(), index: BTreeMap::new(), budget };
        let mut components = Vec::new();
        let mut size: u128 = 1;
        for (vertices, edges) in undirected_components(cpdag) {
            let (slot, count) = counter.count(vertices, edges, 1)?;
            components.push(slot);
            size = size.checked_mul(count).ok_or(MecUniformError::CountOverflow)?;
        }
        Ok(Self {
            cpdag: cpdag.clone(),
            components,
            arena: counter.arena,
            index: counter.index,
            size,
        })
    }

    /// Number of DAG members of the class.
    #[must_use]
    pub const fn class_size(&self) -> u128 {
        self.size
    }

    /// One member drawn uniformly, advancing the `SplitMix64` state `state`.
    ///
    /// # Errors
    ///
    /// A drawn orientation that is not a member of the class (a CPDAG whose components do not
    /// orient independently), or a clique order not drawn within the attempt bound.
    pub fn sample(&self, state: &mut u64) -> Result<Dag, MecUniformError> {
        let mut oriented = Vec::new();
        for &slot in &self.components {
            self.sample_into(slot, state, &mut oriented)?;
        }
        let mut graph = self.cpdag.clone();
        for (from, to) in oriented {
            graph
                .orient_undirected(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to))
                .map_err(|_| MecUniformError::MemberCheckFailed)?;
        }
        let dag = graph.try_into_dag().map_err(|_| MecUniformError::MemberCheckFailed)?;
        if !is_mec_member(&self.cpdag, &dag) {
            return Err(MecUniformError::MemberCheckFailed);
        }
        Ok(dag)
    }

    /// `draws` members drawn uniformly and independently (with replacement) from the stream
    /// `(seed, stream)`. The same arguments give the same members in the same order.
    ///
    /// # Errors
    ///
    /// As [`Self::sample`].
    pub fn sample_members(
        &self,
        draws: usize,
        seed: u64,
        stream: u64,
    ) -> Result<Vec<Dag>, MecUniformError> {
        let mut state = derive_stream(seed, stream);
        (0..draws).map(|_| self.sample(&mut state)).collect()
    }

    fn sample_into(
        &self,
        slot: usize,
        state: &mut u64,
        oriented: &mut Vec<(u32, u32)>,
    ) -> Result<(), MecUniformError> {
        let Some(sub) = self.arena.get(slot) else {
            return Err(MecUniformError::MemberCheckFailed);
        };
        let mut pick = below(state, sub.count);
        let mut chosen = None;
        for choice in &sub.choices {
            if pick < choice.weight {
                chosen = Some(choice);
                break;
            }
            pick -= choice.weight;
        }
        let choice = chosen.ok_or(MecUniformError::MemberCheckFailed)?;
        let order = admissible_order(choice, state)?;
        let local = Local::new(&sub.vertices, &sub.edges);
        let order: Vec<usize> = order.iter().filter_map(|v| local.position(*v)).collect();
        let (orientation, _) = closure(&local, &order);
        for a in 0..local.n {
            for b in 0..local.n {
                if orientation.directed(a, b) {
                    oriented.push((local.vertices[a], local.vertices[b]));
                }
            }
        }
        for (_, edges) in orientation.undirected_components(&local) {
            let next = *self.index.get(&edges).ok_or(MecUniformError::MemberCheckFailed)?;
            self.sample_into(next, state, oriented)?;
        }
        Ok(())
    }
}

/// The exact number of DAG members of the class `cpdag` represents, under the default
/// completion search budget.
///
/// # Errors
///
/// As [`MecUniform::new`].
pub fn mec_size(cpdag: &Cpdag) -> Result<u128, MecUniformError> {
    let (limits, ctx) = crate::completion::default_completion_search();
    MecUniform::new(cpdag, limits, &ctx).map(|class| class.class_size())
}

/// Refuse a graph whose members the per-component product does not count: a directed edge
/// `a -> b` next to an undirected `b - c` with `a`, `c` not adjacent; a directed edge whose
/// two ends lie in one undirected component; or a cycle of directed edges between undirected
/// components (so the graph is not a chain graph).
fn check_essential(cpdag: &Cpdag) -> Result<(), MecUniformError> {
    for edge in cpdag.edges() {
        let Some((from, to)) = edge.parent_child() else {
            continue;
        };
        for far in cpdag.undirected_neighbors(to) {
            if far != from && !cpdag.has_edge(from, far) {
                return Err(MecUniformError::NotEssential);
            }
        }
    }
    let component = component_of_every_vertex(cpdag);
    let count = component.iter().copied().max().map_or(0, |last| last + 1);
    let mut successors: Vec<Vec<usize>> = vec![Vec::new(); count];
    let mut indegree = vec![0_usize; count];
    for edge in cpdag.edges() {
        let Some((from, to)) = edge.parent_child() else {
            continue;
        };
        let (Some(&a), Some(&b)) = (component.get(from.as_usize()), component.get(to.as_usize()))
        else {
            return Err(MecUniformError::NotEssential);
        };
        if a == b {
            return Err(MecUniformError::NotEssential);
        }
        if !successors[a].contains(&b) {
            successors[a].push(b);
            indegree[b] += 1;
        }
    }
    // Kahn's order over the components: a component left unordered lies on a cycle.
    let mut ready: Vec<usize> = (0..count).filter(|&c| indegree[c] == 0).collect();
    let mut ordered = 0;
    while let Some(c) = ready.pop() {
        ordered += 1;
        for &next in &successors[c] {
            indegree[next] -= 1;
            if indegree[next] == 0 {
                ready.push(next);
            }
        }
    }
    if ordered == count { Ok(()) } else { Err(MecUniformError::NotEssential) }
}

/// The undirected component of every vertex, numbered from zero in order of each component's
/// smallest vertex; a vertex without undirected edges is its own component.
fn component_of_every_vertex(cpdag: &Cpdag) -> Vec<usize> {
    let n = cpdag.node_count();
    let mut component = vec![usize::MAX; n];
    let mut next = 0;
    for start in 0..n {
        if component[start] != usize::MAX {
            continue;
        }
        component[start] = next;
        let mut queue = VecDeque::from([DenseNodeId::from_index(start)]);
        while let Some(v) = queue.pop_front() {
            for w in cpdag.undirected_neighbors(v) {
                if component[w.as_usize()] == usize::MAX {
                    component[w.as_usize()] = next;
                    queue.push_back(w);
                }
            }
        }
        next += 1;
    }
    component
}

/// The undirected components of `cpdag` with at least one edge, as global vertex ids and
/// edges `(a, b)` with `a < b`, both ascending.
fn undirected_components(cpdag: &Cpdag) -> Vec<Component> {
    let n = cpdag.node_count();
    let mut seen = vec![false; n];
    let mut out = Vec::new();
    for start in 0..n {
        let start_id = DenseNodeId::from_index(start);
        if seen[start] || cpdag.undirected_neighbors(start_id).is_empty() {
            continue;
        }
        let mut vertices = Vec::new();
        let mut queue = VecDeque::from([start_id]);
        seen[start] = true;
        while let Some(v) = queue.pop_front() {
            vertices.push(v.raw());
            for w in cpdag.undirected_neighbors(v) {
                if !seen[w.as_usize()] {
                    seen[w.as_usize()] = true;
                    queue.push_back(w);
                }
            }
        }
        vertices.sort_unstable();
        let mut edges = Vec::new();
        for &v in &vertices {
            for w in cpdag.undirected_neighbors(DenseNodeId::from_raw(v)) {
                if v < w.raw() {
                    edges.push((v, w.raw()));
                }
            }
        }
        edges.sort_unstable();
        out.push((vertices, edges));
    }
    out
}

/// A subproblem on local indices `0..n`.
struct Local {
    n: usize,
    vertices: Vec<u32>,
    adjacency: Vec<bool>,
}

impl Local {
    fn new(vertices: &[u32], edges: &[(u32, u32)]) -> Self {
        let n = vertices.len();
        let mut local = Self { n, vertices: vertices.to_vec(), adjacency: vec![false; n * n] };
        for &(a, b) in edges {
            if let (Some(i), Some(j)) = (local.position(a), local.position(b)) {
                local.adjacency[i * n + j] = true;
                local.adjacency[j * n + i] = true;
            }
        }
        local
    }

    fn position(&self, vertex: u32) -> Option<usize> {
        self.vertices.binary_search(&vertex).ok()
    }

    fn adjacent(&self, a: usize, b: usize) -> bool {
        self.adjacency[a * self.n + b]
    }
}

/// A partial orientation of a subproblem's edges.
struct Orientation {
    n: usize,
    directed: Vec<bool>,
    undirected: Vec<bool>,
}

impl Orientation {
    fn directed(&self, a: usize, b: usize) -> bool {
        self.directed[a * self.n + b]
    }

    fn undirected(&self, a: usize, b: usize) -> bool {
        self.undirected[a * self.n + b]
    }

    fn orient(&mut self, from: usize, to: usize) {
        self.undirected[from * self.n + to] = false;
        self.undirected[to * self.n + from] = false;
        self.directed[from * self.n + to] = true;
    }

    /// Meek's rules: whether the undirected edge `x - y` is forced to `x -> y`.
    fn forced(&self, local: &Local, x: usize, y: usize) -> bool {
        let n = self.n;
        // R1: z -> x - y with z, y not adjacent.
        if (0..n).any(|z| z != y && self.directed(z, x) && !local.adjacent(z, y)) {
            return true;
        }
        // R2: x -> z -> y.
        if (0..n).any(|z| self.directed(x, z) && self.directed(z, y)) {
            return true;
        }
        // R3: x - z -> y and x - w -> y with z, w not adjacent.
        let into: Vec<usize> =
            (0..n).filter(|&z| self.undirected(x, z) && self.directed(z, y)).collect();
        for (i, &z) in into.iter().enumerate() {
            if into[i + 1..].iter().any(|&w| !local.adjacent(z, w)) {
                return true;
            }
        }
        // R4: w -> z -> y with x adjacent to w and z, and w, y not adjacent.
        (0..n).any(|w| {
            w != y
                && local.adjacent(x, w)
                && !local.adjacent(w, y)
                && (0..n)
                    .any(|z| self.directed(w, z) && self.directed(z, y) && local.adjacent(x, z))
        })
    }

    /// The undirected components left, with at least one edge, as global ids.
    fn undirected_components(&self, local: &Local) -> Vec<Component> {
        let n = self.n;
        let mut seen = vec![false; n];
        let mut out = Vec::new();
        for start in 0..n {
            if seen[start] || !(0..n).any(|w| self.undirected(start, w)) {
                continue;
            }
            let mut members = Vec::new();
            let mut queue = VecDeque::from([start]);
            seen[start] = true;
            while let Some(v) = queue.pop_front() {
                members.push(v);
                for w in 0..n {
                    if self.undirected(v, w) && !seen[w] {
                        seen[w] = true;
                        queue.push_back(w);
                    }
                }
            }
            let mut vertices: Vec<u32> = members.iter().map(|&v| local.vertices[v]).collect();
            vertices.sort_unstable();
            let mut edges = Vec::new();
            for &a in &members {
                for &b in &members {
                    if a < b && self.undirected(a, b) {
                        edges.push((local.vertices[a], local.vertices[b]));
                    }
                }
            }
            edges.sort_unstable();
            out.push((vertices, edges));
        }
        out
    }
}

/// Orient `order` (local indices) first: inside it by its order, every edge leaving it away
/// from it, then close under Meek's rules. Also returns the closure's work: the number of
/// undirected edges Meek's rules examined over every pass.
fn closure(local: &Local, order: &[usize]) -> (Orientation, u64) {
    let n = local.n;
    let mut orientation =
        Orientation { n, directed: vec![false; n * n], undirected: vec![false; n * n] };
    let mut first = vec![false; n];
    for &v in order {
        first[v] = true;
    }
    for a in 0..n {
        for b in 0..n {
            if a == b || !local.adjacent(a, b) {
                continue;
            }
            match (first[a], first[b]) {
                (_, true) => {}
                (true, false) => orientation.directed[a * n + b] = true,
                (false, false) => orientation.undirected[a * n + b] = true,
            }
        }
    }
    for (i, &a) in order.iter().enumerate() {
        for &b in &order[i + 1..] {
            orientation.directed[a * n + b] = true;
        }
    }
    let mut work: u64 = 0;
    loop {
        let mut changed = false;
        for x in 0..n {
            for y in 0..n {
                if !orientation.undirected(x, y) {
                    continue;
                }
                work = work.saturating_add(1);
                if orientation.forced(local, x, y) {
                    orientation.orient(x, y);
                    changed = true;
                }
            }
        }
        if !changed {
            return (orientation, work);
        }
    }
}

/// The maximal cliques of a connected chordal subproblem (local indices, ascending), from a
/// maximum-cardinality search; refuses a non-chordal subproblem.
fn maximal_cliques(local: &Local) -> Result<Vec<Vec<usize>>, MecUniformError> {
    let n = local.n;
    let mut weight = vec![0_usize; n];
    let mut visited = vec![false; n];
    let mut position = vec![0_usize; n];
    let mut order = Vec::with_capacity(n);
    for step in 0..n {
        let mut best: Option<usize> = None;
        for v in 0..n {
            if !visited[v] && best.is_none_or(|b| weight[v] > weight[b]) {
                best = Some(v);
            }
        }
        let Some(v) = best else {
            break;
        };
        visited[v] = true;
        position[v] = step;
        order.push(v);
        for w in 0..n {
            if !visited[w] && local.adjacent(v, w) {
                weight[w] += 1;
            }
        }
    }
    // The reverse search order is a perfect elimination order exactly when the graph is
    // chordal: the neighbours searched before each vertex are pairwise adjacent.
    let mut candidates: Vec<Vec<usize>> = Vec::with_capacity(n);
    for &v in &order {
        let mut earlier: Vec<usize> =
            (0..n).filter(|&w| local.adjacent(v, w) && position[w] < position[v]).collect();
        for (i, &a) in earlier.iter().enumerate() {
            if earlier[i + 1..].iter().any(|&b| !local.adjacent(a, b)) {
                return Err(MecUniformError::NotChordal);
            }
        }
        earlier.push(v);
        earlier.sort_unstable();
        candidates.push(earlier);
    }
    let mut cliques = Vec::new();
    for (i, candidate) in candidates.iter().enumerate() {
        let dominated = candidates.iter().enumerate().any(|(j, other)| {
            j != i && is_subset(candidate, other) && (other.len() > candidate.len() || j < i)
        });
        if !dominated {
            cliques.push(candidate.clone());
        }
    }
    Ok(cliques)
}

/// Whether ascending `small` is a subset of ascending `large`.
fn is_subset<T: Ord>(small: &[T], large: &[T]) -> bool {
    small.iter().all(|x| large.binary_search(x).is_ok())
}

fn intersection<T: Ord + Copy>(a: &[T], b: &[T]) -> Vec<T> {
    a.iter().copied().filter(|x| b.binary_search(x).is_ok()).collect()
}

/// The parent of every clique in a maximum-weight spanning tree of the clique intersection
/// graph, grown from clique 0 (a clique tree of a connected chordal graph).
fn clique_tree(cliques: &[Vec<usize>]) -> Vec<Option<usize>> {
    let c = cliques.len();
    let mut parent = vec![None; c];
    if c == 0 {
        return parent;
    }
    let mut in_tree = vec![false; c];
    in_tree[0] = true;
    for _ in 1..c {
        let mut best: Option<(usize, usize, usize)> = None;
        for i in (0..c).filter(|&i| in_tree[i]) {
            for j in (0..c).filter(|&j| !in_tree[j]) {
                let shared = intersection(&cliques[i], &cliques[j]).len();
                if best.is_none_or(|(w, _, _)| shared > w) {
                    best = Some((shared, i, j));
                }
            }
        }
        let Some((_, i, j)) = best else {
            break;
        };
        in_tree[j] = true;
        parent[j] = Some(i);
    }
    parent
}

fn factorial(n: usize) -> Option<u128> {
    (2..=n).try_fold(1_u128, |acc, k| acc.checked_mul(u128::try_from(k).ok()?))
}

/// The number of orders of a clique of `size` vertices that do not begin with any of
/// `forbidden` (proper subsets of the clique, distinct, ordered by size), by inclusion–exclusion
/// on the shortest forbidden prefix an order begins with.
fn orders_avoiding<T: Ord>(size: usize, forbidden: &[Vec<T>]) -> Option<u128> {
    let mut first_forbidden: Vec<u128> = Vec::with_capacity(forbidden.len());
    for (j, set) in forbidden.iter().enumerate() {
        let mut value = factorial(set.len())?;
        for (i, smaller) in forbidden[..j].iter().enumerate() {
            if smaller.len() < set.len() && is_subset(smaller, set) {
                let tail = factorial(set.len() - smaller.len())?;
                value = value.checked_sub(first_forbidden[i].checked_mul(tail)?)?;
            }
        }
        first_forbidden.push(value);
    }
    let mut total = factorial(size)?;
    for (j, set) in forbidden.iter().enumerate() {
        let tail = factorial(size.checked_sub(set.len())?)?;
        total = total.checked_sub(first_forbidden[j].checked_mul(tail)?)?;
    }
    Some(total)
}

struct Counter<'a> {
    arena: Vec<Subproblem>,
    index: BTreeMap<Vec<(u32, u32)>, usize>,
    budget: SearchBudget<'a>,
}

impl Counter<'_> {
    /// The arena slot and member count of the connected chordal subproblem `edges`.
    fn count(
        &mut self,
        vertices: Vec<u32>,
        edges: Vec<(u32, u32)>,
        depth: usize,
    ) -> Result<(usize, u128), MecUniformError> {
        if let Some(&slot) = self.index.get(&edges) {
            return Ok((slot, self.arena[slot].count));
        }
        let local = Local::new(&vertices, &edges);
        let cliques = maximal_cliques(&local)?;
        let parent = clique_tree(&cliques);
        let mut choices = Vec::with_capacity(cliques.len());
        let mut count: u128 = 0;
        for (k, clique) in cliques.iter().enumerate() {
            let bytes = u64::try_from(self.arena.len().saturating_add(1).saturating_mul(256))
                .unwrap_or(u64::MAX);
            self.budget.charge(depth, bytes).map_err(MecUniformError::Stopped)?;
            let mut forbidden: Vec<Vec<usize>> = Vec::new();
            let (mut child, mut up) = (k, parent[k]);
            while let Some(p) = up {
                let separator = intersection(&cliques[p], &cliques[child]);
                if !separator.is_empty()
                    && separator.len() < clique.len()
                    && is_subset(&separator, clique)
                    && !forbidden.contains(&separator)
                {
                    forbidden.push(separator);
                }
                child = p;
                up = parent[p];
            }
            forbidden.sort_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));
            let phi =
                orders_avoiding(clique.len(), &forbidden).ok_or(MecUniformError::CountOverflow)?;
            let (orientation, work) = closure(&local, clique);
            for _ in 0..work {
                self.budget.charge(depth, bytes).map_err(MecUniformError::Stopped)?;
            }
            let mut weight = phi;
            for (sub_vertices, sub_edges) in orientation.undirected_components(&local) {
                let (_, sub_count) = self.count(sub_vertices, sub_edges, depth + 1)?;
                weight = weight.checked_mul(sub_count).ok_or(MecUniformError::CountOverflow)?;
            }
            count = count.checked_add(weight).ok_or(MecUniformError::CountOverflow)?;
            choices.push(CliqueChoice {
                clique: clique.iter().map(|&v| local.vertices[v]).collect(),
                forbidden: forbidden
                    .iter()
                    .map(|set| set.iter().map(|&v| local.vertices[v]).collect())
                    .collect(),
                weight,
            });
        }
        let slot = self.arena.len();
        self.arena.push(Subproblem { vertices, edges: edges.clone(), count, choices });
        self.index.insert(edges, slot);
        Ok((slot, count))
    }
}

/// A uniform `u128` from two `SplitMix64` outputs.
fn next_u128(state: &mut u64) -> u128 {
    (u128::from(splitmix64(state)) << 64) | u128::from(splitmix64(state))
}

/// A uniform integer in `[0, bound)`, `bound > 0`, without modulo bias: outputs below
/// `2^128 mod bound` are rejected so the accepted range is a whole number of periods.
fn below(state: &mut u64, bound: u128) -> u128 {
    if bound <= 1 {
        return 0;
    }
    let threshold = bound.wrapping_neg() % bound;
    loop {
        let draw = next_u128(state);
        if draw >= threshold {
            return draw % bound;
        }
    }
}

fn index_below(state: &mut u64, bound: usize) -> usize {
    let value = below(state, u128::try_from(bound).unwrap_or(u128::MAX));
    usize::try_from(value).unwrap_or(0)
}

/// An order of the clique drawn uniformly among those that avoid its forbidden prefixes.
fn admissible_order(choice: &CliqueChoice, state: &mut u64) -> Result<Vec<u32>, MecUniformError> {
    let mut order = choice.clique.clone();
    for _ in 0..MAX_ORDER_ATTEMPTS {
        for i in (1..order.len()).rev() {
            let j = index_below(state, i + 1);
            order.swap(i, j);
        }
        let begins_forbidden = choice.forbidden.iter().any(|set| {
            let mut prefix = order[..set.len()].to_vec();
            prefix.sort_unstable();
            prefix == *set
        });
        if !begins_forbidden {
            return Ok(order);
        }
    }
    Err(MecUniformError::OrderNotDrawn)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn undirected(n: u32, edges: &[(u32, u32)]) -> Cpdag {
        let mut g = Cpdag::with_variables(n);
        for &(a, b) in edges {
            g.insert_undirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        g
    }

    #[test]
    fn orders_avoiding_counts_nested_prefixes() {
        // Orders of {0,1,2} avoiding prefixes {0} and {0,1}: start with 1 or 2 (4 orders),
        // less (1,0,2): 3.
        assert_eq!(orders_avoiding(3, &[vec![0], vec![0, 1]]), Some(3));
        assert_eq!(orders_avoiding(3, &[] as &[Vec<u32>]), Some(6));
        assert_eq!(orders_avoiding(2, &[vec![1]]), Some(1));
    }

    #[test]
    fn closed_forms() {
        // A path of n nodes has n members, a complete graph n!, k disjoint edges 2^k.
        let path = undirected(6, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]);
        assert_eq!(mec_size(&path), Ok(6));
        let complete: Vec<(u32, u32)> =
            (0..5).flat_map(|a| ((a + 1)..5).map(move |b| (a, b))).collect();
        assert_eq!(mec_size(&undirected(5, &complete)), Ok(120));
        let disjoint = undirected(8, &[(0, 1), (2, 3), (4, 5), (6, 7)]);
        assert_eq!(mec_size(&disjoint), Ok(16));
        assert_eq!(mec_size(&Cpdag::with_variables(3)), Ok(1));
    }

    #[test]
    fn a_four_cycle_is_not_chordal() {
        let cycle = undirected(4, &[(0, 1), (1, 2), (2, 3), (0, 3)]);
        assert_eq!(mec_size(&cycle), Err(MecUniformError::NotChordal));
    }

    #[test]
    fn a_flag_is_not_an_essential_graph() {
        let mut g = undirected(3, &[(1, 2)]);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        assert_eq!(mec_size(&g), Err(MecUniformError::NotEssential));
    }

    #[test]
    fn a_directed_edge_inside_an_undirected_component_is_refused() {
        // `a -> b` with `a - c - b`: the path component counts three orientations, one of
        // which closes a cycle through `a -> b`, and misses `a -> c <- b`.
        let mut inside = undirected(3, &[(0, 2), (2, 1)]);
        inside.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        assert_eq!(mec_size(&inside), Err(MecUniformError::NotEssential));
        // `a -> b -> c` with `c - a`: the directed edges between the components `{a, c}` and
        // `{b}` form a cycle.
        let mut cycle = undirected(3, &[(0, 2)]);
        cycle.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        cycle.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
        assert_eq!(mec_size(&cycle), Err(MecUniformError::NotEssential));
        // A compelled edge into a clique it fully touches is an essential graph's.
        let mut essential = undirected(3, &[(1, 2)]);
        essential.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        essential.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
        assert_eq!(mec_size(&essential), Ok(2));
    }

    #[test]
    fn the_count_is_charged_for_its_closure_work() {
        // A path on thirty nodes has 29 cliques, and fixing each one leaves Meek's rules to
        // orient the rest of the path edge by edge over many passes: thousands of examined
        // edges, far more than one operation per clique.
        let path: Vec<(u32, u32)> = (0..29).map(|a| (a, a + 1)).collect();
        let graph = undirected(30, &path);
        let ctx = ExecutionContext::for_tests(0);
        let starved = SearchLimits { operations: 100, depth: usize::MAX };
        assert!(matches!(
            MecUniform::new(&graph, starved, &ctx),
            Err(MecUniformError::Stopped(SearchStop::Operations))
        ));
        let ample = SearchLimits { operations: 1_000_000, depth: usize::MAX };
        assert_eq!(MecUniform::new(&graph, ample, &ctx).map(|class| class.class_size()), Ok(30));
    }

    #[test]
    fn components_well_beyond_the_tested_sizes_fit_the_completion_budget() {
        // Measured charges (cliques plus examined closure edges): a path of n nodes about
        // n^3 / 3 (30: 7,715; 60: 66,730; 120: 554,660; 160: 1,327,280), a binary tree far less
        // (160: 155,908), a star or a clique one operation per clique (a star of 160: 159; a
        // clique of 20: 1). Every component the sampling tests draw (paths of 30, random chordal
        // graphs of at most 12 nodes) charges below 10,000, and these fit the completion budget
        // of 4,000,000 operations, a quarter of what a class sample may charge.
        let (limits, ctx) = crate::completion::default_completion_search();
        let path: Vec<(u32, u32)> = (0..119).map(|a| (a, a + 1)).collect();
        let binary: Vec<(u32, u32)> = (1..160).map(|a| ((a - 1) / 2, a)).collect();
        let star: Vec<(u32, u32)> = (1..160).map(|a| (0, a)).collect();
        let clique: Vec<(u32, u32)> =
            (0..12).flat_map(|a| ((a + 1)..12).map(move |b| (a, b))).collect();
        // A tree's members are its rootings (n of them); a clique's are its orders.
        for (graph, size) in [
            (undirected(120, &path), 120),
            (undirected(160, &binary), 160),
            (undirected(160, &star), 160),
            (undirected(12, &clique), 479_001_600),
        ] {
            assert_eq!(MecUniform::new(&graph, limits, &ctx).map(|c| c.class_size()), Ok(size));
        }
    }

    #[test]
    fn below_stays_in_range() {
        let mut state = 7;
        for bound in [1_u128, 2, 3, 7, 1 << 70, u128::MAX] {
            for _ in 0..64 {
                assert!(below(&mut state, bound) < bound);
            }
        }
    }
}
