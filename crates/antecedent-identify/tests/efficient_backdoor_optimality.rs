//! The efficient back-door identifier against the optimal-adjustment-set definition and against
//! exact linear-Gaussian asymptotic variances.
//!
//! The references share no code with the identifier:
//!
//! 1. The optimal set is restated from its definition (Henckel, Perkovic and Maathuis 2022):
//!    `cn` is the set of non-treatment descendants of the treatment that are ancestors of the
//!    outcome, `forb` is the descendants of `cn` together with the treatment, and `O = pa(cn) \ forb`.
//!    For every labelled four-node DAG and a stride of the five-node DAGs, every ordered
//!    treatment/outcome pair must report exactly this set.
//! 2. Semantics: validity of every subset is decided by the back-door definition with a
//!    moral-graph d-separation, and for seeded linear-Gaussian parameterisations the population
//!    regression coefficient of the treatment equals the total effect for every valid set and the
//!    asymptotic variance `sigma2_{Y|T,Z} / sigma2_{T|Z}` of that coefficient is never smaller
//!    than the reported set's. The reported set must also be valid.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{AverageEffectQuery, CausalQuery, VariableId};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_identify::{EfficientBackdoorIdentifier, IdentificationStatus, IdentificationWorkspace};

type Arcs = Vec<(usize, usize)>;

fn acyclic(n: usize, arcs: &[(usize, usize)]) -> bool {
    let mut indegree = vec![0usize; n];
    for &(_, to) in arcs {
        indegree[to] += 1;
    }
    let mut ready: Vec<usize> = (0..n).filter(|&v| indegree[v] == 0).collect();
    let mut seen = 0;
    while let Some(v) = ready.pop() {
        seen += 1;
        for &(from, to) in arcs {
            if from == v {
                indegree[to] -= 1;
                if indegree[to] == 0 {
                    ready.push(to);
                }
            }
        }
    }
    seen == n
}

fn all_dags(n: usize) -> Vec<Arcs> {
    let pairs: Vec<(usize, usize)> = (0..n).flat_map(|i| ((i + 1)..n).map(move |j| (i, j))).collect();
    let mut out = Vec::new();
    for code in 0..3usize.pow(u32::try_from(pairs.len()).unwrap()) {
        let (mut arcs, mut rest) = (Vec::new(), code);
        for &(i, j) in &pairs {
            match rest % 3 {
                1 => arcs.push((i, j)),
                2 => arcs.push((j, i)),
                _ => {}
            }
            rest /= 3;
        }
        if acyclic(n, &arcs) {
            out.push(arcs);
        }
    }
    out
}

fn dag_of(n: usize, arcs: &[(usize, usize)]) -> Dag {
    let mut g = Dag::with_variables(u32::try_from(n).unwrap());
    for &(from, to) in arcs {
        g.insert_directed(
            DenseNodeId::from_raw(u32::try_from(from).unwrap()),
            DenseNodeId::from_raw(u32::try_from(to).unwrap()),
        )
        .unwrap();
    }
    g
}

fn descendants(n: usize, arcs: &[(usize, usize)], of: &[usize]) -> Vec<bool> {
    let mut reach = vec![false; n];
    for &v in of {
        reach[v] = true;
    }
    loop {
        let mut grew = false;
        for &(from, to) in arcs {
            if reach[from] && !reach[to] {
                reach[to] = true;
                grew = true;
            }
        }
        if !grew {
            return reach;
        }
    }
}

fn ancestors(n: usize, arcs: &[(usize, usize)], of: &[usize]) -> Vec<bool> {
    let reversed: Arcs = arcs.iter().map(|&(a, b)| (b, a)).collect();
    descendants(n, &reversed, of)
}

/// Moral-graph d-separation of `x` and `y` given `z`.
fn d_separated(n: usize, arcs: &[(usize, usize)], x: &[usize], y: &[usize], z: &[usize]) -> bool {
    let mut seeds: Vec<usize> = x.iter().chain(y).chain(z).copied().collect();
    seeds.sort_unstable();
    let keep = ancestors(n, arcs, &seeds);
    let mut adj = vec![vec![false; n]; n];
    for &(from, to) in arcs {
        if keep[from] && keep[to] {
            adj[from][to] = true;
            adj[to][from] = true;
        }
    }
    for child in 0..n {
        let parents: Vec<usize> =
            arcs.iter().filter(|&&(f, t)| t == child && keep[f] && keep[t]).map(|a| a.0).collect();
        for &a in &parents {
            for &b in &parents {
                if a != b {
                    adj[a][b] = true;
                }
            }
        }
    }
    let mut seen = vec![false; n];
    let mut stack: Vec<usize> = x.iter().copied().filter(|v| !z.contains(v)).collect();
    for &v in &stack {
        seen[v] = true;
    }
    while let Some(v) = stack.pop() {
        if y.contains(&v) {
            return false;
        }
        for w in 0..n {
            if keep[w] && adj[v][w] && !seen[w] && !z.contains(&w) {
                seen[w] = true;
                stack.push(w);
            }
        }
    }
    true
}

/// Pearl's back-door criterion: no member descends from `t` and `z` separates `t` from `y`
/// once the arrows leaving `t` are removed.
fn backdoor_valid(n: usize, arcs: &[(usize, usize)], t: usize, y: usize, z: &[usize]) -> bool {
    let de = descendants(n, arcs, &[t]);
    if z.iter().any(|&v| de[v]) {
        return false;
    }
    let cut: Arcs = arcs.iter().copied().filter(|&(from, _)| from != t).collect();
    d_separated(n, &cut, &[t], &[y], z)
}

/// The optimal adjustment set from its definition.
fn optimal_set(n: usize, arcs: &[(usize, usize)], t: usize, y: usize) -> Vec<usize> {
    let de_t = descendants(n, arcs, &[t]);
    let an_y = ancestors(n, arcs, &[y]);
    let cn: Vec<usize> = (0..n).filter(|&v| v != t && de_t[v] && an_y[v]).collect();
    let mut forb = descendants(n, arcs, &cn);
    forb[t] = true;
    (0..n).filter(|&v| !forb[v] && arcs.iter().any(|&(from, to)| from == v && cn.contains(&to))).collect()
}

fn subsets(pool: &[usize]) -> Vec<Vec<usize>> {
    (0..(1usize << pool.len()))
        .map(|mask| pool.iter().enumerate().filter(|(i, _)| mask >> i & 1 == 1).map(|(_, &v)| v).collect())
        .collect()
}

fn reported(n: usize, arcs: &[(usize, usize)], t: usize, y: usize) -> (IdentificationStatus, Vec<usize>) {
    let g = dag_of(n, arcs);
    let id = EfficientBackdoorIdentifier::new();
    let prepared = id.prepare(&g).unwrap();
    let query = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
        VariableId::from_raw(u32::try_from(t).unwrap()),
        VariableId::from_raw(u32::try_from(y).unwrap()),
    ));
    let result = id.identify(&prepared, &query, &mut IdentificationWorkspace::default()).unwrap();
    assert!(result.estimands.len() <= 1);
    let mut set: Vec<usize> = result
        .estimands
        .first()
        .map(|e| e.adjustment_set.iter().map(|v| v.raw() as usize).collect())
        .unwrap_or_default();
    set.sort_unstable();
    (result.status, set)
}

/// Linear-Gaussian SCM `x = B x + e` with the arcs as non-zero entries of `B`.
struct Gaussian {
    n: usize,
    cov: Vec<Vec<f64>>,
    total: Vec<Vec<f64>>,
}

impl Gaussian {
    fn new(n: usize, arcs: &[(usize, usize)], seed: u64) -> Self {
        let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        let mut uniform = move || {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            f64::from(u32::try_from(state >> 33).unwrap()) / f64::from(1u32 << 31)
        };
        let mut b = vec![vec![0.0; n]; n];
        for &(from, to) in arcs {
            let magnitude = 0.5 + uniform();
            b[to][from] = if uniform() < 0.5 { -magnitude } else { magnitude };
        }
        let noise: Vec<f64> = (0..n).map(|_| 0.5 + 1.5 * uniform()).collect();
        // (I - B)^{-1} by forward substitution over a topological order (B is nilpotent).
        let mut total = vec![vec![0.0; n]; n];
        for i in 0..n {
            total[i][i] = 1.0;
        }
        for _ in 0..n {
            let prev = total.clone();
            for i in 0..n {
                for j in 0..n {
                    let mut s = if i == j { 1.0 } else { 0.0 };
                    for (k, row) in prev.iter().enumerate() {
                        s += b[i][k] * row[j];
                    }
                    total[i][j] = s;
                }
            }
        }
        let mut cov = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                cov[i][j] = (0..n).map(|k| total[i][k] * noise[k] * total[j][k]).sum();
            }
        }
        Self { n, cov, total }
    }

    /// Conditional covariance block of `a` given `s` (Schur complement).
    fn conditional(&self, a: &[usize], s: &[usize]) -> Vec<Vec<f64>> {
        let mut out: Vec<Vec<f64>> = a.iter().map(|&i| a.iter().map(|&j| self.cov[i][j]).collect()).collect();
        if s.is_empty() {
            return out;
        }
        let m = s.len();
        let mut aug: Vec<Vec<f64>> = (0..m)
            .map(|r| {
                let mut row: Vec<f64> = s.iter().map(|&c| self.cov[s[r]][c]).collect();
                row.extend(a.iter().map(|&c| self.cov[s[r]][c]));
                row
            })
            .collect();
        for col in 0..m {
            let pivot = (col..m).max_by(|&p, &q| aug[p][col].abs().total_cmp(&aug[q][col].abs())).unwrap();
            aug.swap(col, pivot);
            let p = aug[col][col];
            for v in &mut aug[col] {
                *v /= p;
            }
            for r in 0..m {
                if r != col {
                    let f = aug[r][col];
                    for c in 0..aug[r].len() {
                        let delta = f * aug[col][c];
                        aug[r][c] -= delta;
                    }
                }
            }
        }
        for (i, &ai) in a.iter().enumerate() {
            for (j, _) in a.iter().enumerate() {
                let mut s_sum = 0.0;
                for (r, &sr) in s.iter().enumerate() {
                    s_sum += self.cov[ai][sr] * aug[r][m + j];
                }
                out[i][j] -= s_sum;
            }
        }
        debug_assert_eq!(self.n, self.cov.len());
        out
    }

    /// Population coefficient of `t` in the regression of `y` on `t` and `z`, and its asymptotic
    /// variance factor `sigma2_{y|t,z} / sigma2_{t|z}`.
    fn regression(&self, t: usize, y: usize, z: &[usize]) -> (f64, f64) {
        let cond = self.conditional(&[t, y], z);
        let beta = cond[0][1] / cond[0][0];
        let mut with_t = vec![t];
        with_t.extend_from_slice(z);
        let resid = self.conditional(&[y], &with_t)[0][0];
        (beta, resid / cond[0][0])
    }
}

fn sweep(n: usize, dags: &[Arcs], seeds: &[u64]) -> (usize, usize, usize, usize) {
    let (mut pairs, mut strictly_better, mut larger_than_minimal, mut parent_cases) = (0, 0, 0, 0);
    for arcs in dags {
        for t in 0..n {
            for y in 0..n {
                if t == y {
                    continue;
                }
                let pool: Vec<usize> = (0..n).filter(|&v| v != t && v != y).collect();
                let valid: Vec<Vec<usize>> =
                    subsets(&pool).into_iter().filter(|z| backdoor_valid(n, arcs, t, y, z)).collect();
                let (status, found) = reported(n, arcs, t, y);
                if valid.is_empty() {
                    // Only the outcome being a direct parent of the treatment leaves no valid
                    // set: the arrow into the treatment is a back-door path nothing can block.
                    assert!(arcs.contains(&(y, t)), "arcs={arcs:?} t={t} y={y}");
                    assert_eq!(status, IdentificationStatus::NotIdentified, "arcs={arcs:?}");
                    assert!(found.is_empty());
                    parent_cases += 1;
                    continue;
                }
                // The optimal set and its variance guarantee concern a treatment with a causal
                // path to the outcome; otherwise the effect is zero, any valid set recovers it
                // and the identifier only has to return one.
                let causal = descendants(n, arcs, &[t])[y];
                if causal {
                    assert_eq!(found, optimal_set(n, arcs, t, y), "arcs={arcs:?} t={t} y={y}");
                }
                assert_eq!(
                    status,
                    IdentificationStatus::NonparametricallyIdentified,
                    "arcs={arcs:?} t={t} y={y}"
                );
                assert!(backdoor_valid(n, arcs, t, y, &found), "arcs={arcs:?} t={t} y={y} set={found:?}");
                pairs += 1;
                assert!(valid.contains(&found));
                let smallest = valid.iter().map(Vec::len).min().unwrap();
                if found.len() > smallest {
                    larger_than_minimal += 1;
                }
                for &seed in seeds {
                    let scm = Gaussian::new(n, arcs, seed);
                    let truth = scm.total[y][t];
                    let (_, v_found) = scm.regression(t, y, &found);
                    let mut any_worse = false;
                    if !causal {
                        assert!(truth.abs() < 1e-12);
                    }
                    for z in &valid {
                        let (beta, variance) = scm.regression(t, y, z);
                        assert!(
                            (beta - truth).abs() < 1e-8 * (1.0 + truth.abs()),
                            "arcs={arcs:?} t={t} y={y} z={z:?}: {beta} vs {truth}"
                        );
                        assert!(
                            !causal || v_found <= variance * (1.0 + 1e-9),
                            "arcs={arcs:?} t={t} y={y} z={z:?} seed={seed}: reported {found:?} has variance {v_found} > {variance}"
                        );
                        any_worse |= variance > v_found * (1.0 + 1e-6);
                    }
                    strictly_better += usize::from(any_worse);
                }
            }
        }
    }
    (pairs, strictly_better, larger_than_minimal, parent_cases)
}

#[test]
fn reported_set_is_the_defined_optimal_set_on_all_four_node_dags() {
    let dags = all_dags(4);
    assert_eq!(dags.len(), 543);
    let (pairs, strictly_better, larger, parents) = sweep(4, &dags, &[1, 2, 3]);
    assert_eq!(pairs + parents, 543 * 12);
    assert!(parents > 100);
    // The comparison is not vacuous: valid sets with strictly larger variance and optimal sets
    // larger than the smallest valid set both occur.
    assert!(strictly_better > 300, "{strictly_better}");
    assert!(larger > 10, "{larger}");
}

#[test]
fn reported_set_is_the_defined_optimal_set_on_a_stride_of_five_node_dags() {
    let dags: Vec<Arcs> = all_dags(5).into_iter().step_by(37).collect();
    assert!(dags.len() > 200);
    let (pairs, strictly_better, _, parents) = sweep(5, &dags, &[7]);
    assert_eq!(pairs + parents, dags.len() * 20);
    assert!(strictly_better > 300, "{strictly_better}");
}

#[test]
fn a_forbidden_optimal_member_falls_back_to_a_valid_set_that_may_cost_variance() {
    // z -> t, z -> m, m -> y, t -> y: O = {m}. Forbidding m leaves {z}, valid but larger variance.
    let arcs: Arcs = vec![(0, 1), (0, 2), (2, 3), (1, 3)];
    let g = dag_of(4, &arcs);
    let mut id = EfficientBackdoorIdentifier::new();
    id.config.forbidden = vec![VariableId::from_raw(2)].into();
    let prepared = id.prepare(&g).unwrap();
    let query = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
        VariableId::from_raw(1),
        VariableId::from_raw(3),
    ));
    let result = id.identify(&prepared, &query, &mut IdentificationWorkspace::default()).unwrap();
    let set: Vec<usize> = result.estimands[0].adjustment_set.iter().map(|v| v.raw() as usize).collect();
    assert_eq!(set, vec![0]);
    assert!(backdoor_valid(4, &arcs, 1, 3, &set));
    let scm = Gaussian::new(4, &arcs, 11);
    let (_, fallback) = scm.regression(1, 3, &set);
    let (_, optimal) = scm.regression(1, 3, &optimal_set(4, &arcs, 1, 3));
    assert!(optimal <= fallback);
}
