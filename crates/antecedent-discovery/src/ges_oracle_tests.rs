//! GES against the equivalence class a linear-Gaussian model implies, computed from its
//! definition.
//!
//! The data columns are built so that their sample covariance is exactly the population
//! covariance of a random linear structural model: the noise columns are made exactly
//! orthonormal (mean zero, unit variance, uncorrelated) by Gram-Schmidt and each variable is
//! the exact linear combination of its parents and its noise. The Gaussian BIC of every graph
//! is then the population BIC at `n` rows, which is maximised exactly by the members of the true
//! Markov equivalence class when the model is faithful, so the search must return that class:
//! the same adjacencies and the same compelled edges. The expected class comes from enumerating
//! acyclic orientations of the skeleton that keep the unshielded colliders; neither the
//! Insert/Reverse/Delete operators nor the Meek rules are used to produce it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeSet, HashSet};

use antecedent_core::{ExecutionContext, VariableId};
use antecedent_graph::DenseNodeId;

use super::Ges;
use crate::engine::DiscoveryWorkspace;
use crate::test_tables::continuous_table;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 11
    }

    fn unit(&mut self) -> f64 {
        self.next() as f64 / (1u64 << 53) as f64
    }
}

/// `count` columns of `n` rows: exactly zero mean, exactly uncorrelated, sum of squares `n`.
fn orthonormal_noise(rng: &mut Lcg, n: usize, count: usize) -> Vec<Vec<f64>> {
    let mut basis: Vec<Vec<f64>> = Vec::new();
    for _ in 0..count {
        let mut v: Vec<f64> = (0..n).map(|_| rng.unit() - 0.5).collect();
        for _ in 0..2 {
            let mean = v.iter().sum::<f64>() / n as f64;
            v.iter_mut().for_each(|x| *x -= mean);
            for b in &basis {
                let dot: f64 = v.iter().zip(b).map(|(x, y)| x * y).sum::<f64>() / n as f64;
                v.iter_mut().zip(b).for_each(|(x, y)| *x -= dot * y);
            }
        }
        let scale = (v.iter().map(|x| x * x).sum::<f64>() / n as f64).sqrt();
        v.iter_mut().for_each(|x| *x /= scale);
        basis.push(v);
    }
    basis
}

#[derive(Clone)]
struct Model {
    n: usize,
    /// `parents[c]` lists `(parent, coefficient)`.
    parents: Vec<Vec<(usize, f64)>>,
    order: Vec<usize>,
}

impl Model {
    fn edges(&self) -> Vec<(usize, usize)> {
        (0..self.n).flat_map(|c| self.parents[c].iter().map(move |&(p, _)| (p, c))).collect()
    }

    fn columns(&self, rows: usize, rng: &mut Lcg) -> Vec<Vec<f64>> {
        let noise = orthonormal_noise(rng, rows, self.n);
        let mut cols: Vec<Vec<f64>> = vec![Vec::new(); self.n];
        for &v in &self.order {
            let mut x = noise[v].clone();
            for &(p, b) in &self.parents[v] {
                x.iter_mut().zip(&cols[p]).for_each(|(xi, pi)| *xi += b * pi);
            }
            cols[v] = x;
        }
        cols
    }

    fn ancestors(&self, seeds: &[usize]) -> HashSet<usize> {
        let mut seen: HashSet<usize> = seeds.iter().copied().collect();
        let mut stack = seeds.to_vec();
        while let Some(v) = stack.pop() {
            for &(p, _) in &self.parents[v] {
                if seen.insert(p) {
                    stack.push(p);
                }
            }
        }
        seen
    }

    /// Lauritzen's moral-ancestral-graph criterion.
    fn d_separated(&self, x: usize, y: usize, z: &[usize]) -> bool {
        let mut seeds = vec![x, y];
        seeds.extend_from_slice(z);
        let keep = self.ancestors(&seeds);
        let mut adj: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); self.n];
        for &v in &keep {
            let ps: Vec<usize> =
                self.parents[v].iter().map(|&(p, _)| p).filter(|p| keep.contains(p)).collect();
            for (i, &a) in ps.iter().enumerate() {
                adj[a].insert(v);
                adj[v].insert(a);
                for &b in &ps[i + 1..] {
                    adj[a].insert(b);
                    adj[b].insert(a);
                }
            }
        }
        let blocked: HashSet<usize> = z.iter().copied().collect();
        let mut seen = HashSet::from([x]);
        let mut stack = vec![x];
        while let Some(v) = stack.pop() {
            for &u in &adj[v] {
                if !blocked.contains(&u) && seen.insert(u) {
                    stack.push(u);
                }
            }
        }
        !seen.contains(&y)
    }
}

fn covariance(cols: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = cols[0].len() as f64;
    cols.iter()
        .map(|a| {
            cols.iter().map(|b| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>() / n).collect()
        })
        .collect()
}

/// Partial correlation of `i` and `j` given `z` from the inverse of the joint covariance block.
fn partial_correlation(cov: &[Vec<f64>], i: usize, j: usize, z: &[usize]) -> f64 {
    let idx: Vec<usize> = [i, j].iter().chain(z).copied().collect();
    let m = idx.len();
    let mut a: Vec<Vec<f64>> =
        (0..m).map(|r| (0..m).map(|c| cov[idx[r]][idx[c]]).collect()).collect();
    let mut inv: Vec<Vec<f64>> =
        (0..m).map(|r| (0..m).map(|c| f64::from(u8::from(r == c))).collect()).collect();
    for c in 0..m {
        let piv = (c..m).max_by(|&p, &q| a[p][c].abs().total_cmp(&a[q][c].abs())).unwrap();
        a.swap(c, piv);
        inv.swap(c, piv);
        let d = a[c][c];
        for k in 0..m {
            a[c][k] /= d;
            inv[c][k] /= d;
        }
        for r in 0..m {
            if r != c {
                let f = a[r][c];
                for k in 0..m {
                    a[r][k] -= f * a[c][k];
                    inv[r][k] -= f * inv[c][k];
                }
            }
        }
    }
    -inv[0][1] / (inv[0][0] * inv[1][1]).sqrt()
}

fn adjacent(edges: &[(usize, usize)], a: usize, b: usize) -> bool {
    edges.iter().any(|&(u, v)| (u == a && v == b) || (u == b && v == a))
}

fn unshielded_colliders(n: usize, edges: &[(usize, usize)]) -> BTreeSet<(usize, usize, usize)> {
    let mut out = BTreeSet::new();
    for c in 0..n {
        let ps: Vec<usize> = edges.iter().filter(|e| e.1 == c).map(|e| e.0).collect();
        for (i, &a) in ps.iter().enumerate() {
            for &b in &ps[i + 1..] {
                if !adjacent(edges, a, b) {
                    out.insert((a.min(b), c, a.max(b)));
                }
            }
        }
    }
    out
}

fn acyclic(n: usize, edges: &[(usize, usize)]) -> bool {
    let mut indeg = vec![0usize; n];
    edges.iter().for_each(|&(_, v)| indeg[v] += 1);
    let mut ready: Vec<usize> = (0..n).filter(|&v| indeg[v] == 0).collect();
    let mut seen = 0;
    while let Some(v) = ready.pop() {
        seen += 1;
        for &(u, w) in edges {
            if u == v {
                indeg[w] -= 1;
                if indeg[w] == 0 {
                    ready.push(w);
                }
            }
        }
    }
    seen == n
}

/// `((a, b), Some((parent, child)))` for a compelled edge, `None` for a reversible one.
type ClassEdge = ((usize, usize), Option<(usize, usize)>);

fn class_edges(n: usize, edges: &[(usize, usize)]) -> Vec<ClassEdge> {
    let target = unshielded_colliders(n, edges);
    let mut members: Vec<Vec<(usize, usize)>> = Vec::new();
    for mask in 0u32..(1 << edges.len()) {
        let oriented: Vec<(usize, usize)> = edges
            .iter()
            .enumerate()
            .map(|(i, &(a, b))| if mask >> i & 1 == 0 { (a, b) } else { (b, a) })
            .collect();
        if acyclic(n, &oriented) && unshielded_colliders(n, &oriented) == target {
            members.push(oriented);
        }
    }
    edges
        .iter()
        .enumerate()
        .map(|(i, &(a, b))| {
            let forward = members.iter().all(|m| m[i] == (a, b));
            let backward = members.iter().all(|m| m[i] == (b, a));
            ((a.min(b), a.max(b)), if forward { Some((a, b)) } else if backward { Some((b, a)) } else { None })
        })
        .collect()
}

fn random_model(rng: &mut Lcg, n: usize, density: u64) -> Model {
    let mut order: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        order.swap(i, (rng.next() % (i as u64 + 1)) as usize);
    }
    let mut parents = vec![Vec::new(); n];
    for i in 0..n {
        for j in (i + 1)..n {
            if rng.next() % 100 < density {
                let sign = if rng.next() % 2 == 0 { 1.0 } else { -1.0 };
                parents[order[j]].push((order[i], sign * (0.6 + 0.4 * rng.unit())));
            }
        }
    }
    Model { n, parents, order }
}

/// A model is usable when every d-connection leaves a partial correlation the score can see
/// and every d-separation leaves none (faithfulness with a visible margin).
fn clearly_faithful(model: &Model, cov: &[Vec<f64>]) -> bool {
    let n = model.n;
    for i in 0..n {
        for j in (i + 1)..n {
            let rest: Vec<usize> = (0..n).filter(|&k| k != i && k != j).collect();
            for mask in 0u32..(1 << rest.len()) {
                let z: Vec<usize> =
                    rest.iter().enumerate().filter(|(b, _)| mask >> b & 1 == 1).map(|(_, &k)| k).collect();
                let r = partial_correlation(cov, i, j, &z).abs();
                if model.d_separated(i, j, &z) {
                    if r > 1e-9 {
                        return false;
                    }
                } else if r < 0.12 {
                    return false;
                }
            }
        }
    }
    true
}

fn run_ges(cols: &[Vec<f64>]) -> crate::pc::StaticCpdagDiscoveryResult {
    let n = cols.len();
    let names: Vec<String> = (0..n).map(|i| format!("v{i}")).collect();
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    let data = continuous_table(&refs, cols.to_vec());
    let vars: Vec<VariableId> = (0..n as u32).map(VariableId::from_raw).collect();
    Ges::new()
        .run(&data, &vars, &mut DiscoveryWorkspace::default(), &ExecutionContext::for_tests(1))
        .unwrap()
}

fn assert_class(model: &Model, label: &str) {
    let n = model.n;
    let mut rng = Lcg(0xC0FFEE ^ (label.len() as u64) << 20 ^ model.edges().len() as u64);
    let cols = model.columns(4000, &mut rng);
    let result = run_ges(&cols);
    let graph = &result.evidence.graph;
    let expected = class_edges(n, &model.edges());
    let adjacency: HashSet<(usize, usize)> = expected.iter().map(|e| e.0).collect();
    for a in 0..n {
        for b in (a + 1)..n {
            let found =
                graph.edge_between(DenseNodeId::from_raw(a as u32), DenseNodeId::from_raw(b as u32));
            assert_eq!(found.is_some(), adjacency.contains(&(a, b)), "{label}: adjacency {a}-{b}; dag {:?}", model.edges());
        }
    }
    for ((a, b), compelled) in expected {
        let edge = graph
            .edge_between(DenseNodeId::from_raw(a as u32), DenseNodeId::from_raw(b as u32))
            .unwrap();
        let got = edge.parent_child().map(|(p, c)| (p.raw() as usize, c.raw() as usize));
        assert_eq!(got, compelled, "{label}: edge {a}-{b}; dag {:?}", model.edges());
        if compelled.is_none() {
            assert!(edge.is_undirected(), "{label}: edge {a}-{b} must stay undirected");
        }
    }
}

fn model(n: usize, edges: &[(usize, usize)]) -> Model {
    let mut parents = vec![Vec::new(); n];
    for (k, &(p, c)) in edges.iter().enumerate() {
        parents[c].push((p, if k % 2 == 0 { 0.8 } else { -0.7 }));
    }
    // Hand-built edge lists below are listed parent-before-child, so id order is topological
    // only when the lists say so; use a real topological order.
    let mut order = Vec::new();
    let mut placed = vec![false; n];
    while order.len() < n {
        for v in 0..n {
            if !placed[v] && parents[v].iter().all(|&(p, _)| placed[p]) {
                placed[v] = true;
                order.push(v);
            }
        }
    }
    Model { n, parents, order }
}

#[test]
fn exact_covariance_columns_reproduce_the_population_covariance() {
    let m = model(3, &[(0, 1), (1, 2)]);
    let cols = m.columns(500, &mut Lcg(7));
    let cov = covariance(&cols);
    // Var(x0) = 1, Var(x1) = 0.8^2 + 1, Var(x2) = 0.49 Var(x1) + 1 and the cross terms follow.
    assert!((cov[0][0] - 1.0).abs() < 1e-12);
    assert!((cov[1][1] - 1.64).abs() < 1e-12);
    assert!((cov[0][1] - 0.8).abs() < 1e-12);
    assert!((cov[2][2] - (0.49 * 1.64 + 1.0)).abs() < 1e-12);
    assert!((cov[1][2] + 0.7 * 1.64).abs() < 1e-12);
    assert!((cov[0][2] + 0.56).abs() < 1e-12);
    // Partial correlation of the chain ends given the middle vanishes exactly.
    assert!(partial_correlation(&cov, 0, 2, &[1]).abs() < 1e-12);
    assert!(m.d_separated(0, 2, &[1]) && !m.d_separated(0, 2, &[]));
}

#[test]
fn known_classes_are_recovered_from_population_scores() {
    assert_class(&model(3, &[(0, 1), (1, 2)]), "chain");
    assert_class(&model(3, &[(0, 2), (1, 2)]), "collider");
    assert_class(&model(4, &[(0, 2), (1, 2), (2, 3)]), "collider then compelled tail");
    assert_class(&model(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]), "diamond");
    assert_class(&model(4, &[(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]), "complete graph");
    assert_class(&model(5, &[(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)]), "diamond with tail");
}

#[test]
fn random_models_recover_their_markov_equivalence_class() {
    let mut rng = Lcg(0x9E37_79B9_7F4A_7C15);
    let mut checked = 0;
    for trial in 0..120u32 {
        let n = 4 + (trial as usize % 3);
        let m = random_model(&mut rng, n, 40);
        let edges = m.edges();
        if edges.is_empty() || edges.len() > 10 {
            continue;
        }
        let cols = m.columns(4000, &mut Lcg(u64::from(trial) + 1));
        if !clearly_faithful(&m, &covariance(&cols)) {
            continue;
        }
        assert_class(&m, &format!("trial {trial}"));
        checked += 1;
    }
    assert!(checked >= 40, "only {checked} usable random models");
}
