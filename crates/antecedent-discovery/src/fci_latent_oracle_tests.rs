//! Static FCI under a population d-separation oracle on random DAGs with hidden variables.
//!
//! The oracle answers every conditional-independence query exactly, from the generating DAG,
//! so any disagreement is the algorithm's, not sampling error.  The reference quantities are
//! computed here from definitions, not from FCI's own routines:
//!
//! * adjacency: observed `x`, `y` are adjacent in the maximal ancestral graph exactly when no
//!   subset of the other observed variables d-separates them (found by exhausting subsets);
//! * unshielded colliders: for non-adjacent `x`, `y` with common neighbour `z`, the PAG has
//!   arrowheads into `z` on both edges exactly when `z` lies in no separating set;
//! * soundness of every mark: an arrowhead at `B` on an edge with `A` means `B` is not an
//!   ancestor of `A` in the generating DAG (hidden nodes included), and a tail at `B` means it
//!   is.
//!
//! Orientation completeness (that no mark Zhang's rules can determine is left a circle) is not
//! asserted; that is a separate claim with its own references.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_graph::{DenseNodeId, Endpoint};
use antecedent_stats::{
    CiBatchRequest, CiBatchResult, CiResult, CiWorkspace, ConditionalIndependenceTest,
    PreparedCiTest, StatsError,
};

use crate::engine::DiscoveryWorkspace;
use crate::fci::Fci;

struct Dag {
    n: usize,
    /// `parents[v]`: parents of `v`.
    parents: Vec<Vec<usize>>,
}

impl Dag {
    /// `ancestors[v][u]`: `u` is an ancestor of `v` or `u == v`.
    fn ancestor_matrix(&self) -> Vec<Vec<bool>> {
        let mut anc = vec![vec![false; self.n]; self.n];
        // Nodes are topologically ordered by index (parents have smaller indices).
        for v in 0..self.n {
            anc[v][v] = true;
            for &p in &self.parents[v] {
                for u in 0..self.n {
                    if anc[p][u] {
                        anc[v][u] = true;
                    }
                }
            }
        }
        anc
    }

    /// d-separation by the moral graph of the ancestral set of `{x, y} ∪ z`.
    fn d_separated(&self, x: usize, y: usize, z: &[usize]) -> bool {
        let anc = self.ancestor_matrix();
        let in_set = |u: usize| u == x || u == y || z.contains(&u);
        let relevant: Vec<bool> =
            (0..self.n).map(|u| (0..self.n).any(|w| in_set(w) && anc[w][u])).collect();
        let mut adj = vec![vec![false; self.n]; self.n];
        for v in 0..self.n {
            if !relevant[v] {
                continue;
            }
            for &p in &self.parents[v] {
                adj[v][p] = true;
                adj[p][v] = true;
                for &q in &self.parents[v] {
                    if p != q {
                        adj[p][q] = true;
                    }
                }
            }
        }
        let mut seen = vec![false; self.n];
        let mut stack = vec![x];
        seen[x] = true;
        while let Some(v) = stack.pop() {
            if v == y {
                return false;
            }
            for u in 0..self.n {
                if adj[v][u] && relevant[u] && !seen[u] && !z.contains(&u) {
                    seen[u] = true;
                    stack.push(u);
                }
            }
        }
        true
    }
}

struct DsepOracle {
    dag: Arc<Dag>,
    /// DAG node of each observed column.
    observed: Vec<usize>,
}

impl ConditionalIndependenceTest for DsepOracle {
    fn test_batch(
        &self,
        prepared: &PreparedCiTest,
        request: &CiBatchRequest<'_>,
        _workspace: &mut CiWorkspace,
        _ctx: &ExecutionContext,
    ) -> Result<CiBatchResult, StatsError> {
        prepared.ensure_compatible(request)?;
        let results = request
            .queries
            .iter()
            .map(|q| {
                let z: Vec<usize> = request.z_flat[q.z_start..q.z_start + q.z_len]
                    .iter()
                    .map(|&c| self.observed[c])
                    .collect();
                let separated = self.dag.d_separated(self.observed[q.x], self.observed[q.y], &z);
                CiResult {
                    statistic: if separated { 0.0 } else { 1.0 },
                    p_value: if separated { 1.0 } else { 0.0 },
                    df: 0.0,
                    ci: None,
                }
            })
            .collect();
        Ok(CiBatchResult { results })
    }
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 =
            self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }
}

/// A random DAG on `n` nodes (edges only from lower to higher index, density about 0.4) with
/// a random subset of `hidden` nodes unobserved. Returns the DAG and the observed node list.
fn random_case(rng: &mut Lcg, n: usize, hidden: usize) -> (Dag, Vec<usize>) {
    let mut parents = vec![Vec::new(); n];
    for v in 0..n {
        for u in 0..v {
            if rng.next() % 5 < 2 {
                parents[v].push(u);
            }
        }
    }
    let mut is_hidden = vec![false; n];
    let mut placed = 0;
    while placed < hidden {
        let v = (rng.next() as usize) % n;
        if !is_hidden[v] {
            is_hidden[v] = true;
            placed += 1;
        }
    }
    let observed = (0..n).filter(|&v| !is_hidden[v]).collect();
    (Dag { n, parents }, observed)
}

fn subsets_of(items: &[usize]) -> impl Iterator<Item = Vec<usize>> + '_ {
    (0..(1u32 << items.len())).map(move |mask| {
        items.iter().enumerate().filter(|(i, _)| mask >> i & 1 == 1).map(|(_, &v)| v).collect()
    })
}

fn mark_at(pag: &antecedent_graph::Pag, a: usize, b: usize, at: usize) -> Endpoint {
    let edge = pag
        .edge_between(DenseNodeId::from_raw(a as u32), DenseNodeId::from_raw(b as u32))
        .expect("edge");
    if edge.a.as_usize() == at { edge.at_a } else { edge.at_b }
}

#[test]
fn fci_under_a_dsep_oracle_recovers_the_mag_skeleton_and_colliders_with_sound_marks() {
    let mut rng = Lcg(0xF1C0_0001);
    let mut checked_edges = 0;
    let mut checked_colliders = 0;
    let mut checked_marks = 0;
    for case in 0..260 {
        let n = 5 + case % 3;
        let hidden = 1 + (case / 3) % 2;
        let (dag, observed) = random_case(&mut rng, n, hidden);
        let dag = Arc::new(dag);
        let k = observed.len();
        let anc = dag.ancestor_matrix();

        // Reference: separating sets by exhaustion over the observed others.
        let mut sep: Vec<Vec<Option<Vec<Vec<usize>>>>> = vec![vec![None; k]; k];
        for i in 0..k {
            for j in (i + 1)..k {
                let others: Vec<usize> = (0..k).filter(|&c| c != i && c != j).collect();
                let separating: Vec<Vec<usize>> = subsets_of(&others)
                    .filter(|s| {
                        let z: Vec<usize> = s.iter().map(|&c| observed[c]).collect();
                        dag.d_separated(observed[i], observed[j], &z)
                    })
                    .collect();
                sep[i][j] = Some(separating);
            }
        }
        let separable = |i: usize, j: usize| {
            let (a, b) = if i < j { (i, j) } else { (j, i) };
            sep[a][b].as_ref().is_some_and(|s| !s.is_empty())
        };

        let names: Vec<String> = (0..k).map(|i| format!("v{i}")).collect();
        let column: Vec<f64> = (0..40).map(|r| f64::from(r) * 0.1).collect();
        let data =
            TabularData::from_f64_columns(names.iter().map(|s| (s.as_str(), column.as_slice())))
                .unwrap();
        let variables: Vec<VariableId> = (0..k).map(|i| VariableId::from_raw(i as u32)).collect();
        let mut fci = Fci::new()
            .with_fdr(false)
            .with_ci(Arc::new(DsepOracle { dag: Arc::clone(&dag), observed: observed.clone() }));
        fci.constraints.max_cond_size = k;
        fci.constraints.alpha = 0.5;
        let mut ws = DiscoveryWorkspace::default();
        let result = fci
            .run(&data, &variables, &mut ws, &ExecutionContext::for_tests(11))
            .unwrap_or_else(|e| panic!("case {case}: {e}"));
        let pag = &result.evidence.graph;
        let adjacent = |i: usize, j: usize| {
            pag.has_edge(DenseNodeId::from_raw(i as u32), DenseNodeId::from_raw(j as u32))
        };
        let describe = || format!("case {case}: parents {:?} observed {observed:?}", dag.parents);

        for i in 0..k {
            for j in (i + 1)..k {
                assert_eq!(adjacent(i, j), !separable(i, j), "adjacency {i}-{j}; {}", describe());
                checked_edges += 1;
            }
        }
        // Unshielded colliders from the exhaustive separating sets.
        for z in 0..k {
            for x in 0..k {
                for y in (x + 1)..k {
                    if x == z || y == z || !adjacent(x, z) || !adjacent(y, z) || adjacent(x, y) {
                        continue;
                    }
                    let sets = sep[x][y].as_ref().unwrap();
                    let containing = sets.iter().filter(|s| s.contains(&z)).count();
                    assert!(
                        containing == 0 || containing == sets.len(),
                        "separating sets disagree on {z} for ({x},{y}); {}",
                        describe()
                    );
                    let collider = containing == 0;
                    let into_z = |a: usize| mark_at(pag, a, z, z) == Endpoint::Arrow;
                    assert_eq!(
                        into_z(x) && into_z(y),
                        collider,
                        "collider {x} *-> {z} <-* {y}; {}",
                        describe()
                    );
                    checked_colliders += 1;
                }
            }
        }
        // Every definite mark must agree with the true ancestry among observed variables.
        for i in 0..k {
            for j in 0..k {
                if i == j || !adjacent(i, j) {
                    continue;
                }
                // Mark at `j` on the edge between `i` and `j`.
                let at_j = mark_at(pag, i, j, j);
                let j_ancestor_of_i = anc[observed[i]][observed[j]];
                match at_j {
                    Endpoint::Arrow => assert!(
                        !j_ancestor_of_i,
                        "arrowhead at {j} on {i}-{j} but {j} is an ancestor of {i}; {}",
                        describe()
                    ),
                    Endpoint::Tail => assert!(
                        j_ancestor_of_i,
                        "tail at {j} on {i}-{j} but {j} is not an ancestor of {i}; {}",
                        describe()
                    ),
                    Endpoint::Circle => {}
                    Endpoint::Conflict => panic!("conflict mark; {}", describe()),
                }
                checked_marks += 1;
            }
        }
    }
    assert!(checked_edges > 1500 && checked_colliders > 40 && checked_marks > 400);
    eprintln!(
        "fci oracle: {checked_edges} pairs, {checked_colliders} colliders, {checked_marks} marks"
    );
}
