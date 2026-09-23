//! S-admissibility over treatment-mutilated selection diagrams, shared by the
//! sID engine and the structural transport identifier.
//!
//! A selection node `S_t` is a parentless cause of its target `t`. Because it is
//! only ever a path endpoint, one node per target is equivalent to the paper's
//! full set of S nodes. `D_{X̄}` removes directed edges into `X` and bidirected
//! edges at `X` (do(X) also severs latent confounding of `X`), and is induced on
//! the current vertex set. A set `Z` is S-admissible for `Y` when `Y` is
//! m-separated from every selection node given `X ∪ Z` in that graph.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_graph::{Admg, BitSet, DSeparationWorkspace, DenseNodeId};

use crate::IdentificationError;
use crate::enum_masks::for_each_mask_of_size;

/// `D_{X̄}` over a vertex set, with one selection node per applicable target.
pub(crate) struct MutilatedSelection {
    graph: Admg,
    selection_nodes: Vec<DenseNodeId>,
}

impl MutilatedSelection {
    /// Build over the subgraph induced by `v`. Targets outside `v` or intervened
    /// are skipped: an intervention removes the mechanism's dependence on `S`.
    pub(crate) fn build(
        original: &Admg,
        v: &BitSet,
        x: &BitSet,
        targets: &[DenseNodeId],
    ) -> Result<Self, IdentificationError> {
        let n = original.node_count();
        let capacity = || IdentificationError::msg("selection graph capacity");
        let count = u32::try_from(n + targets.len()).map_err(|_| capacity())?;
        let mut graph = Admg::with_variables(count);
        for from in v.to_dense_ids() {
            for &to in original.children(from) {
                if v.contains(to) && !x.contains(to) {
                    graph.insert_directed(from, to)?;
                }
            }
            for &to in original.bidirected_neighbors(from) {
                if from.raw() < to.raw() && v.contains(to) && !x.contains(from) && !x.contains(to) {
                    graph.insert_bidirected(from, to)?;
                }
            }
        }
        let mut selection_nodes = Vec::with_capacity(targets.len());
        for (i, &target) in targets.iter().enumerate() {
            if !v.contains(target) || x.contains(target) {
                continue;
            }
            let selection = DenseNodeId::from_raw(u32::try_from(n + i).map_err(|_| capacity())?);
            graph.insert_directed(selection, target)?;
            selection_nodes.push(selection);
        }
        Ok(Self { graph, selection_nodes })
    }

    /// Every outcome is m-separated from every selection node given `conditions`.
    pub(crate) fn separates(
        &self,
        outcomes: &[DenseNodeId],
        conditions: &[DenseNodeId],
        ws: &mut DSeparationWorkspace,
    ) -> Result<bool, IdentificationError> {
        for &selection in &self.selection_nodes {
            for &outcome in outcomes {
                if !self.graph.is_m_separated(selection, outcome, conditions, ws)? {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }
}

/// How an ordered standardizer-subset search ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SubsetSearchEnd {
    /// The visitor accepted a subset.
    Accepted,
    /// Every subset was examined; none was accepted.
    Exhausted,
    /// The evaluation budget ran out before every subset was examined. The
    /// search is inconclusive, not a proof that no admissible set exists.
    Capped,
}

/// Visit the S-admissible subsets of `candidates` in size-ascending order (then
/// lexicographic), stopping when `visit` returns `true`.
///
/// `base` is always conditioned on (the treatments). At most `budget` subsets are
/// separation-tested; this budget is the search's own and never charges the
/// caller's recursion budget. `tick` runs once per tested subset (cancellation).
pub(crate) fn for_each_admissible_subset(
    selection: &MutilatedSelection,
    outcomes: &[DenseNodeId],
    base: &[DenseNodeId],
    candidates: &[DenseNodeId],
    budget: usize,
    mut tick: impl FnMut() -> Result<(), IdentificationError>,
    mut visit: impl FnMut(&[DenseNodeId]) -> Result<bool, IdentificationError>,
) -> Result<SubsetSearchEnd, IdentificationError> {
    let mut ws = DSeparationWorkspace::default();
    let mut conditions = Vec::with_capacity(base.len() + candidates.len());
    let mut evaluated = 0usize;
    let mut step = |subset: &[DenseNodeId]| -> Result<bool, IdentificationError> {
        tick()?;
        conditions.clear();
        conditions.extend_from_slice(base);
        conditions.extend_from_slice(subset);
        if !selection.separates(outcomes, &conditions, &mut ws)? {
            return Ok(false);
        }
        visit(subset)
    };
    for size in 0..=candidates.len() {
        let mut capped = false;
        let mut failure = None;
        let accepted = for_each_mask_of_size(candidates, size, |subset| {
            if evaluated >= budget {
                capped = true;
                return true;
            }
            evaluated += 1;
            match step(subset) {
                Ok(stop) => stop,
                Err(error) => {
                    failure = Some(error);
                    true
                }
            }
        });
        if let Some(error) = failure {
            return Err(error);
        }
        if capped {
            return Ok(SubsetSearchEnd::Capped);
        }
        if accepted {
            return Ok(SubsetSearchEnd::Accepted);
        }
    }
    Ok(SubsetSearchEnd::Exhausted)
}

/// Whether `outcomes` are separated from every selection target given `x ∪ given`,
/// decided by Richardson's augmented-graph criterion instead of path search.
///
/// Reads the original graph directly and shares no code with
/// [`MutilatedSelection`], so it is a second implementation of the same
/// premise for use by proof checkers. Criterion (Richardson 2003): restrict the
/// mutilated graph to the ancestors of `{S, Y} ∪ Z`, join every pair of nodes
/// that are adjacent or collider-connected, and test ordinary undirected
/// separation by `Z`. Collider-connection through a bidirected district `D` links
/// exactly the members of `D` and the parents of `D` to each other.
pub(crate) fn independently_separated(
    original: &Admg,
    v: &BitSet,
    x: &BitSet,
    targets: &[DenseNodeId],
    outcomes: &[DenseNodeId],
    given: &[DenseNodeId],
) -> bool {
    let n = original.node_count();
    let total = n + 1; // one selection node, re-aimed at each target in turn
    let selection = n;
    let mut conditioned = vec![false; total];
    for node in x.to_dense_ids() {
        conditioned[node.as_usize()] = true;
    }
    for node in given {
        conditioned[node.as_usize()] = true;
    }
    let live = |i: usize| v.contains(DenseNodeId::from_raw(u32::try_from(i).expect("fit")));
    let mut parents = vec![Vec::new(); total];
    let mut siblings = vec![Vec::new(); total];
    for (from, from_siblings) in siblings.iter_mut().enumerate().take(n) {
        if !live(from) {
            continue;
        }
        let id = DenseNodeId::from_raw(u32::try_from(from).expect("fit"));
        for child in original.children(id) {
            let to = child.as_usize();
            if live(to) && !x.contains(*child) {
                parents[to].push(from);
            }
        }
        for sibling in original.bidirected_neighbors(id) {
            let to = sibling.as_usize();
            if live(to) && !x.contains(id) && !x.contains(*sibling) {
                from_siblings.push(to);
            }
        }
    }
    for &target in targets {
        let t = target.as_usize();
        if !live(t) || x.contains(target) {
            continue;
        }
        parents[t].push(selection);
        let clear = outcomes.iter().all(|outcome| {
            let y = outcome.as_usize();
            !conditioned[selection]
                && !conditioned[y]
                && !connected(&parents, &siblings, &conditioned, selection, y)
        });
        parents[t].pop();
        if !clear {
            return false;
        }
    }
    true
}

/// Is `a` connected to `b` in the augmented graph of the ancestors of `{a, b} ∪ Z`
/// once the conditioned nodes are removed?
fn connected(
    parents: &[Vec<usize>],
    siblings: &[Vec<usize>],
    conditioned: &[bool],
    a: usize,
    b: usize,
) -> bool {
    let total = parents.len();
    let mut ancestral = vec![false; total];
    let mut pending: Vec<usize> =
        (0..total).filter(|&i| conditioned[i] || i == a || i == b).collect();
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
    let mut adjacent = vec![vec![false; total]; total];
    let mut link = |p: usize, q: usize| {
        if p != q {
            adjacent[p][q] = true;
            adjacent[q][p] = true;
        }
    };
    for node in (0..total).filter(|&i| ancestral[i]) {
        for &parent in &parents[node] {
            link(parent, node);
        }
        for &sibling in &siblings[node] {
            if ancestral[sibling] {
                link(node, sibling);
            }
        }
    }
    // Collider paths run only through bidirected edges, so each district of the
    // ancestral subgraph contributes one clique over itself and its parents.
    let mut assigned = vec![false; total];
    for seed in (0..total).filter(|&i| ancestral[i]) {
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
    let mut seen = vec![false; total];
    let mut stack = vec![a];
    seen[a] = true;
    while let Some(node) = stack.pop() {
        if node == b {
            return true;
        }
        for next in 0..total {
            if adjacent[node][next] && !seen[next] && (!conditioned[next] || next == b) {
                seen[next] = true;
                stack.push(next);
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(i: u32) -> DenseNodeId {
        DenseNodeId::from_raw(i)
    }

    fn all(n: u32) -> BitSet {
        let mut set = BitSet::with_len(n as usize);
        for i in 0..n {
            set.insert(d(i));
        }
        set
    }

    /// Both implementations must agree on every ADMG over four nodes with the
    /// selection node attached to node 1, for every choice of treatment set,
    /// outcome and conditioning set.
    #[test]
    fn augmented_graph_criterion_agrees_with_m_separation_on_all_four_node_admgs() {
        let n = 4u32;
        let pairs: Vec<(u32, u32)> = (0..n).flat_map(|a| (a + 1..n).map(move |b| (a, b))).collect();
        // Each unordered pair: none, a->b, or bidirected (acyclic by index order).
        let mut checked = 0usize;
        for code in 0..3usize.pow(u32::try_from(pairs.len()).unwrap()) {
            let mut graph = Admg::with_variables(n);
            let mut rest = code;
            for &(a, b) in &pairs {
                match rest % 3 {
                    1 => graph.insert_directed(d(a), d(b)).unwrap(),
                    2 => graph.insert_bidirected(d(a), d(b)).unwrap(),
                    _ => {}
                }
                rest /= 3;
            }
            let v = all(n);
            for x_mask in 0..(1u32 << n) {
                let mut x = BitSet::with_len(n as usize);
                for i in (0..n).filter(|i| (x_mask >> i) & 1 == 1) {
                    x.insert(d(i));
                }
                if x.contains(d(1)) {
                    continue;
                }
                let selection = MutilatedSelection::build(&graph, &v, &x, &[d(1)]).unwrap();
                for outcome in (0..n).filter(|i| !x.contains(d(*i)) && *i != 1) {
                    for z_mask in 0..(1u32 << n) {
                        let given: Vec<_> = (0..n)
                            .filter(|i| (z_mask >> i) & 1 == 1)
                            .filter(|i| !x.contains(d(*i)) && *i != outcome)
                            .map(d)
                            .collect();
                        let mut conditions = x.to_dense_ids();
                        conditions.extend(given.iter().copied());
                        let by_search = selection
                            .separates(
                                &[d(outcome)],
                                &conditions,
                                &mut DSeparationWorkspace::default(),
                            )
                            .unwrap();
                        let by_criterion =
                            independently_separated(&graph, &v, &x, &[d(1)], &[d(outcome)], &given);
                        assert_eq!(
                            by_search, by_criterion,
                            "graph code {code}, x {x_mask:#b}, outcome {outcome}, given {given:?}"
                        );
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 10_000);
    }

    #[test]
    fn intervention_removes_latent_confounding_of_the_treatment() {
        // X <-> Y, X <-> T, selection on T, conditioning on T. Unmutilated, S -> T <-> X <-> Y
        // is open (T and X are both conditioned colliders). do(X) severs both bidirected
        // edges at X, so S and Y are separated.
        let mut graph = Admg::with_variables(3);
        graph.insert_bidirected(d(0), d(1)).unwrap();
        graph.insert_bidirected(d(0), d(2)).unwrap();
        let v = all(3);
        let mut x = BitSet::with_len(3);
        x.insert(d(0));
        let selection = MutilatedSelection::build(&graph, &v, &x, &[d(2)]).unwrap();
        let conditions = [d(0), d(2)];
        assert!(
            selection
                .separates(&[d(1)], &conditions, &mut DSeparationWorkspace::default())
                .unwrap()
        );
        assert!(independently_separated(&graph, &v, &x, &[d(2)], &[d(1)], &[d(2)]));
        // Without the intervention the same path is open.
        let none = BitSet::with_len(3);
        let open = MutilatedSelection::build(&graph, &v, &none, &[d(2)]).unwrap();
        assert!(
            !open.separates(&[d(1)], &conditions, &mut DSeparationWorkspace::default()).unwrap()
        );
        assert!(!independently_separated(&graph, &v, &none, &[d(2)], &[d(1)], &[d(0), d(2)]));
    }

    #[test]
    fn subset_search_is_size_ascending_and_reports_a_cap_instead_of_failing() {
        // Selection on 0; 0 -> 3, 1 -> 3, 2 -> 3. Only conditioning on 0 blocks S -> 0 -> 3.
        let mut graph = Admg::with_variables(4);
        graph.insert_directed(d(0), d(3)).unwrap();
        graph.insert_directed(d(1), d(3)).unwrap();
        graph.insert_directed(d(2), d(3)).unwrap();
        let v = all(4);
        let x = BitSet::with_len(4);
        let selection = MutilatedSelection::build(&graph, &v, &x, &[d(0)]).unwrap();
        let mut seen = Vec::new();
        let end = for_each_admissible_subset(
            &selection,
            &[d(3)],
            &[],
            &[d(1), d(2), d(0)],
            usize::MAX,
            || Ok(()),
            |subset| {
                seen.push(subset.to_vec());
                Ok(true)
            },
        )
        .unwrap();
        assert_eq!(end, SubsetSearchEnd::Accepted);
        assert_eq!(seen, vec![vec![d(0)]]);

        let end = for_each_admissible_subset(
            &selection,
            &[d(3)],
            &[],
            &[d(1), d(2), d(0)],
            2,
            || Ok(()),
            |_| Ok(false),
        )
        .unwrap();
        assert_eq!(end, SubsetSearchEnd::Capped);
    }
}
