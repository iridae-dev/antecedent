//! Exhaustive checks of the back-door and front-door criteria and of their adjustment formulas.
//!
//! Two independent references, neither sharing code with the identifiers:
//!
//! 1. Each criterion restated from its definition (Pearl, Causality, Definitions 3.3.1 and
//!    3.3.3). For every labelled DAG and every ordered treatment/outcome pair, every subset of
//!    the other nodes is tested with a d-separation implemented by moralising the ancestral graph
//!    (the identifiers use reachability-based d-separation). The sets an identifier returns, and
//!    its status, must equal the sets the definition accepts (all of them, only the
//!    inclusion-minimal ones, or only the inclusion-maximal ones, as configured).
//! 2. Semantics: for each accepted set, a seeded binary SCM Markov to the DAG gives the exact
//!    `P(y | do(t))` by truncated factorisation, and the adjustment or front-door formula
//!    evaluated on the observational joint alone must reproduce it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{AverageEffectQuery, CausalQuery, VariableId};
use antecedent_graph::{Dag, DenseNodeId};

use super::*;
use crate::frontdoor::{FrontDoorIdentifier, FrontDoorSearchConfig};
use crate::identifier::IdentificationWorkspace;
use crate::result::IdentificationStatus;

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

fn topological(n: usize, arcs: &[(usize, usize)]) -> Vec<usize> {
    let mut order = Vec::new();
    let mut placed = vec![false; n];
    while order.len() < n {
        let next = (0..n)
            .find(|&v| {
                !placed[v] && arcs.iter().filter(|&&(_, to)| to == v).all(|&(from, _)| placed[from])
            })
            .expect("acyclic");
        placed[next] = true;
        order.push(next);
    }
    order
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

/// d-separation of node sets by the moral-graph criterion: restrict to the ancestors of
/// `x ∪ y ∪ z`, marry co-parents, drop undirected the nodes of `z`, test connectivity.
fn d_separated(n: usize, arcs: &[(usize, usize)], x: &[usize], y: &[usize], z: &[usize]) -> bool {
    let mut keep = vec![false; n];
    for &v in x.iter().chain(y).chain(z) {
        keep[v] = true;
    }
    loop {
        let mut grew = false;
        for &(from, to) in arcs {
            if keep[to] && !keep[from] {
                keep[from] = true;
                grew = true;
            }
        }
        if !grew {
            break;
        }
    }
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
    let blocked = |v: usize| z.contains(&v);
    let mut seen = vec![false; n];
    let mut stack: Vec<usize> = x.iter().copied().filter(|&v| !blocked(v)).collect();
    for &v in &stack {
        seen[v] = true;
    }
    while let Some(v) = stack.pop() {
        if y.contains(&v) {
            return false;
        }
        for w in 0..n {
            if keep[w] && adj[v][w] && !seen[w] && !blocked(w) {
                seen[w] = true;
                stack.push(w);
            }
        }
    }
    true
}

/// Pearl's front-door conditions for a non-empty `m`.
fn satisfies_definition(n: usize, arcs: &[(usize, usize)], t: usize, y: usize, m: &[usize]) -> bool {
    // (i) every directed path from t to y contains a member of m.
    let mut reach = vec![false; n];
    let mut stack = vec![t];
    reach[t] = true;
    while let Some(v) = stack.pop() {
        for &(from, to) in arcs {
            if from == v && !reach[to] && !m.contains(&to) {
                reach[to] = true;
                stack.push(to);
            }
        }
    }
    if reach[y] {
        return false;
    }
    // (ii) no back-door path from t to m: t independent of m in the graph without t's out-arrows.
    let no_t_out: Arcs = arcs.iter().copied().filter(|&(from, _)| from != t).collect();
    if !d_separated(n, &no_t_out, &[t], m, &[]) {
        return false;
    }
    // (iii) t blocks every back-door path from m to y: m independent of y given t without m's out-arrows.
    let no_m_out: Arcs = arcs.iter().copied().filter(|&(from, _)| !m.contains(&from)).collect();
    d_separated(n, &no_m_out, m, &[y], &[t])
}

fn subsets_of(pool: &[usize]) -> Vec<Vec<usize>> {
    (1..(1usize << pool.len()))
        .map(|mask| pool.iter().enumerate().filter(|(i, _)| mask >> i & 1 == 1).map(|(_, &v)| v).collect())
        .collect()
}

fn reported(
    n: usize,
    g: &Dag,
    t: usize,
    y: usize,
) -> (IdentificationStatus, Vec<Vec<usize>>) {
    let id = FrontDoorIdentifier {
        config: FrontDoorSearchConfig { max_results: usize::MAX, max_mediator_set_size: n },
    };
    let prepared = id.prepare(g).unwrap();
    let query = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
        VariableId::from_raw(u32::try_from(t).unwrap()),
        VariableId::from_raw(u32::try_from(y).unwrap()),
    ));
    let result = id.identify(&prepared, &query, &mut IdentificationWorkspace::default()).unwrap();
    let mut sets: Vec<Vec<usize>> = result
        .estimands
        .iter()
        .map(|e| {
            let mut s: Vec<usize> = e.mediators.iter().map(|v| v.raw() as usize).collect();
            s.sort_unstable();
            s
        })
        .collect();
    sets.sort();
    (result.status, sets)
}

/// Seeded binary SCM Markov to `arcs`: `table[v][k]` is `P(v = 1 | parent bits k)`.
struct Scm {
    n: usize,
    parents: Vec<Vec<usize>>,
    table: Vec<Vec<f64>>,
}

impl Scm {
    fn new(n: usize, arcs: &[(usize, usize)], seed: u64) -> Self {
        let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        let mut uniform = move || {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            0.1 + 0.8 * (f64::from(u32::try_from(state >> 33).unwrap()) / f64::from(1u32 << 31))
        };
        let parents: Vec<Vec<usize>> =
            (0..n).map(|v| arcs.iter().filter(|a| a.1 == v).map(|a| a.0).collect()).collect();
        let table = parents
            .iter()
            .map(|p| (0..(1usize << p.len())).map(|_| uniform()).collect())
            .collect();
        Self { n, parents, table }
    }

    fn weight(&self, state: usize, skip: Option<usize>) -> f64 {
        (0..self.n)
            .filter(|&v| Some(v) != skip)
            .map(|v| {
                let k = self.parents[v]
                    .iter()
                    .enumerate()
                    .fold(0usize, |acc, (i, &p)| acc | ((state >> p & 1) << i));
                let p1 = self.table[v][k];
                if state >> v & 1 == 1 { p1 } else { 1.0 - p1 }
            })
            .product()
    }

    fn observed(&self, fixed: &[(usize, usize)]) -> f64 {
        (0..(1usize << self.n))
            .filter(|s| fixed.iter().all(|&(v, b)| s >> v & 1 == b))
            .map(|s| self.weight(s, None))
            .sum()
    }

    fn do_effect(&self, t: usize, y: usize, value: usize) -> f64 {
        (0..(1usize << self.n))
            .filter(|s| s >> t & 1 == value && s >> y & 1 == 1)
            .map(|s| self.weight(s, Some(t)))
            .sum()
    }

    /// The adjustment formula for `z`, from observational probabilities only.
    fn adjustment(&self, t: usize, y: usize, z: &[usize], value: usize) -> f64 {
        let mut total = 0.0;
        for assignment in 0..(1usize << z.len()) {
            let bits: Vec<(usize, usize)> =
                z.iter().enumerate().map(|(i, &v)| (v, assignment >> i & 1)).collect();
            let mut with_t = bits.clone();
            with_t.push((t, value));
            let mut with_y = with_t.clone();
            with_y.push((y, 1));
            total += self.observed(&with_y) / self.observed(&with_t) * self.observed(&bits);
        }
        total
    }

    /// The front-door formula, from observational probabilities only.
    fn frontdoor(&self, t: usize, y: usize, m: &[usize], value: usize) -> f64 {
        let mut total = 0.0;
        for assignment in 0..(1usize << m.len()) {
            let bits: Vec<(usize, usize)> =
                m.iter().enumerate().map(|(i, &v)| (v, assignment >> i & 1)).collect();
            let mut given_t = bits.clone();
            given_t.push((t, value));
            let p_m_given_t = self.observed(&given_t) / self.observed(&[(t, value)]);
            let mut inner = 0.0;
            for t_prime in 0..2 {
                let mut with_t = bits.clone();
                with_t.push((t, t_prime));
                let mut with_y = with_t.clone();
                with_y.push((y, 1));
                inner += self.observed(&with_y) / self.observed(&with_t) * self.observed(&[(t, t_prime)]);
            }
            total += p_m_given_t * inner;
        }
        total
    }
}

fn sweep(n: usize, dags: &[Arcs], seeds: &[u64]) -> (usize, usize, usize) {
    let (mut identified, mut refused, mut numeric) = (0, 0, 0);
    for arcs in dags {
        let g = dag_of(n, arcs);
        for t in 0..n {
            for y in 0..n {
                if t == y {
                    continue;
                }
                let pool: Vec<usize> = (0..n).filter(|&v| v != t && v != y).collect();
                let mut expected: Vec<Vec<usize>> = subsets_of(&pool)
                    .into_iter()
                    .filter(|m| satisfies_definition(n, arcs, t, y, m))
                    .collect();
                expected.sort();
                let (status, found) = reported(n, &g, t, y);
                assert_eq!(found, expected, "arcs={arcs:?} t={t} y={y}");
                if expected.is_empty() {
                    refused += 1;
                    assert_eq!(status, IdentificationStatus::NotIdentified, "arcs={arcs:?}");
                    continue;
                }
                identified += 1;
                assert_eq!(
                    status,
                    IdentificationStatus::NonparametricallyIdentified,
                    "arcs={arcs:?}"
                );
                for &seed in seeds {
                    let scm = Scm::new(n, arcs, seed);
                    for m in &expected {
                        for value in 0..2 {
                            let truth = scm.do_effect(t, y, value);
                            let formula = scm.frontdoor(t, y, m, value);
                            assert!(
                                (truth - formula).abs() < 1e-10,
                                "arcs={arcs:?} t={t} y={y} m={m:?} do(t={value}): {truth} vs {formula}"
                            );
                            numeric += 1;
                        }
                    }
                }
            }
        }
    }
    (identified, refused, numeric)
}

#[test]
fn frontdoor_search_matches_the_definition_and_the_formula_recovers_truth_on_every_four_node_dag() {
    let dags = all_dags(4);
    assert_eq!(dags.len(), 543);
    let (identified, refused, numeric) = sweep(4, &dags, &[3, 11]);
    // Both verdicts and the numerical check occur often enough that agreement is not vacuous.
    assert!(identified > 100, "{identified}");
    assert!(refused > 1000, "{refused}");
    assert!(numeric > 200, "{numeric}");
}

#[test]
fn frontdoor_five_node_sample_includes_multi_member_mediator_sets() {
    let dags: Vec<Arcs> = all_dags(5).into_iter().step_by(23).collect();
    let (identified, _, numeric) = sweep(5, &dags, &[5]);
    assert!(identified > 100 && numeric > 200, "{identified} {numeric}");
    // The sample reaches mediator sets with two members.
    let has_pair = dags.iter().any(|arcs| {
        let g = dag_of(5, arcs);
        (0..5).any(|t| (0..5).any(|y| t != y && reported(5, &g, t, y).1.iter().any(|m| m.len() > 1)))
    });
    assert!(has_pair);
}

#[test]
fn the_references_reject_the_standard_counterexamples() {
    // Direct edge: nothing intercepts t -> y.
    assert!(!satisfies_definition(3, &[(0, 1), (0, 2)], 0, 1, &[2]));
    // Confounded mediator: u2 -> m and u2 -> y opens a back-door path from m to y that t does not block.
    assert!(!satisfies_definition(4, &[(0, 1), (1, 2), (3, 1), (3, 2)], 0, 2, &[1]));
    // Classic front door with a latent-style common cause node of t and y.
    assert!(satisfies_definition(4, &[(0, 1), (1, 2), (3, 0), (3, 2)], 0, 2, &[1]));
    assert_eq!(topological(4, &[(0, 1), (1, 2), (3, 0), (3, 2)]).len(), 4);
    // Back-door: a descendant of t is never admissible; the confounder is required.
    assert!(!is_backdoor_adjustment(3, &[(0, 1), (0, 2), (1, 2)], 0, 2, &[1]));
    assert!(!is_backdoor_adjustment(3, &[(1, 0), (1, 2), (0, 2)], 0, 2, &[]));
    assert!(is_backdoor_adjustment(3, &[(1, 0), (1, 2), (0, 2)], 0, 2, &[1]));
    // M-bias: conditioning on the collider opens a path.
    let m_bias = [(3, 0), (3, 1), (4, 1), (4, 2), (0, 2)];
    assert!(is_backdoor_adjustment(5, &m_bias, 0, 2, &[]));
    assert!(!is_backdoor_adjustment(5, &m_bias, 0, 2, &[1]));
}


/// Pearl's back-door conditions: `z` has no descendant of `t` and `t` is d-separated from `y`
/// given `z` once the arrows out of `t` are removed.
fn is_backdoor_adjustment(n: usize, arcs: &[(usize, usize)], t: usize, y: usize, z: &[usize]) -> bool {
    let mut descendant = vec![false; n];
    let mut stack = vec![t];
    while let Some(v) = stack.pop() {
        for &(from, to) in arcs {
            if from == v && !descendant[to] {
                descendant[to] = true;
                stack.push(to);
            }
        }
    }
    if z.iter().any(|&v| descendant[v]) {
        return false;
    }
    let no_t_out: Arcs = arcs.iter().copied().filter(|&(from, _)| from != t).collect();
    d_separated(n, &no_t_out, &[t], &[y], z)
}

fn proper_subsets_of(pool: &[usize]) -> Vec<Vec<usize>> {
    (0..(1usize << pool.len()))
        .map(|mask| pool.iter().enumerate().filter(|(i, _)| mask >> i & 1 == 1).map(|(_, &v)| v).collect())
        .collect()
}

fn is_subset(a: &[usize], b: &[usize]) -> bool {
    a.iter().all(|v| b.contains(v))
}

fn backdoor_report(
    g: &Dag,
    t: usize,
    y: usize,
    configure: impl FnOnce(&mut AdjustmentSearchConfig),
) -> (IdentificationStatus, Vec<Vec<usize>>) {
    let mut id = BackdoorIdentifier::new();
    id.config.max_results = usize::MAX;
    configure(&mut id.config);
    let prepared = id.prepare(g).unwrap();
    let query = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
        VariableId::from_raw(u32::try_from(t).unwrap()),
        VariableId::from_raw(u32::try_from(y).unwrap()),
    ));
    let result = id.identify(&prepared, &query, &mut IdentificationWorkspace::default()).unwrap();
    let mut sets: Vec<Vec<usize>> = result
        .estimands
        .iter()
        .map(|e| {
            let mut s: Vec<usize> = e.adjustment_set.iter().map(|v| v.raw() as usize).collect();
            s.sort_unstable();
            s
        })
        .collect();
    sets.sort();
    (result.status, sets)
}

fn backdoor_sweep(n: usize, dags: &[Arcs], seeds: &[u64]) -> (usize, usize, usize) {
    let (mut identified, mut refused, mut numeric) = (0, 0, 0);
    for arcs in dags {
        let g = dag_of(n, arcs);
        for t in 0..n {
            for y in 0..n {
                if t == y {
                    continue;
                }
                let pool: Vec<usize> = (0..n).filter(|&v| v != t && v != y).collect();
                let mut valid: Vec<Vec<usize>> = proper_subsets_of(&pool)
                    .into_iter()
                    .filter(|z| is_backdoor_adjustment(n, arcs, t, y, z))
                    .collect();
                valid.sort();
                let mut minimal: Vec<Vec<usize>> = valid
                    .iter()
                    .filter(|z| !valid.iter().any(|w| w != *z && is_subset(w, z)))
                    .cloned()
                    .collect();
                minimal.sort();
                let mut maximal: Vec<Vec<usize>> = valid
                    .iter()
                    .filter(|z| !valid.iter().any(|w| w != *z && is_subset(z, w)))
                    .cloned()
                    .collect();
                maximal.sort();
                let context = format!("arcs={arcs:?} t={t} y={y}");

                let (status, all) = backdoor_report(&g, t, y, |c| c.minimal_only = false);
                assert_eq!(all, valid, "all valid sets: {context}");
                let (_, least) = backdoor_report(&g, t, y, |_| {});
                assert_eq!(least, minimal, "inclusion-minimal sets: {context}");
                let (_, most) = backdoor_report(&g, t, y, |c| {
                    c.minimal_only = false;
                    c.maximal_only = true;
                });
                assert_eq!(most, maximal, "inclusion-maximal sets: {context}");
                // An undersized candidate ceiling forces the certified fallback, which must still
                // decide existence exactly and return an admissible set.
                let (fallback_status, fallback) =
                    backdoor_report(&g, t, y, |c| c.max_candidates = 0);
                assert_eq!(fallback_status, status, "fallback status: {context}");
                assert!(fallback.iter().all(|z| valid.contains(z)), "fallback {fallback:?}: {context}");

                if valid.is_empty() {
                    refused += 1;
                    assert_eq!(status, IdentificationStatus::NotIdentified, "{context}");
                    continue;
                }
                identified += 1;
                assert_eq!(status, IdentificationStatus::NonparametricallyIdentified, "{context}");
                for &seed in seeds {
                    let scm = Scm::new(n, arcs, seed);
                    for z in &valid {
                        for value in 0..2 {
                            let truth = scm.do_effect(t, y, value);
                            let formula = scm.adjustment(t, y, z, value);
                            assert!(
                                (truth - formula).abs() < 1e-10,
                                "{context} z={z:?} do(t={value}): {truth} vs {formula}"
                            );
                            numeric += 1;
                        }
                    }
                }
            }
        }
    }
    (identified, refused, numeric)
}

#[test]
fn backdoor_search_matches_the_definition_and_adjustment_recovers_truth_on_every_four_node_dag() {
    let (identified, refused, numeric) = backdoor_sweep(4, &all_dags(4), &[7]);
    assert!(identified > 300 && refused > 100 && numeric > 1000, "{identified} {refused} {numeric}");
}

#[test]
fn backdoor_search_matches_the_definition_on_a_stride_of_the_five_node_dags() {
    let all = all_dags(5);
    assert_eq!(all.len(), 29_281);
    let dags: Vec<Arcs> = all.into_iter().step_by(11).collect();
    let (identified, refused, _) = backdoor_sweep(5, &dags, &[]);
    assert!(identified > 10_000 && refused > 500, "{identified} {refused}");
}

#[test]
fn forbidden_variables_are_never_adjusted_for_and_the_reference_agrees() {
    // Forbid each single candidate in turn: the identifier must return exactly the valid sets
    // that avoid it, and the decision must follow.
    for arcs in all_dags(4) {
        let g = dag_of(4, &arcs);
        for t in 0..4 {
            for y in 0..4 {
                if t == y {
                    continue;
                }
                let pool: Vec<usize> = (0..4).filter(|&v| v != t && v != y).collect();
                for &banned in &pool {
                    let mut expected: Vec<Vec<usize>> = proper_subsets_of(&pool)
                        .into_iter()
                        .filter(|z| !z.contains(&banned) && is_backdoor_adjustment(4, &arcs, t, y, z))
                        .collect();
                    expected.sort();
                    let (status, got) = backdoor_report(&g, t, y, |c| {
                        c.minimal_only = false;
                        c.forbidden = Arc::from([VariableId::from_raw(u32::try_from(banned).unwrap())]);
                    });
                    assert_eq!(got, expected, "arcs={arcs:?} t={t} y={y} banned={banned}");
                    assert_eq!(
                        status == IdentificationStatus::NotIdentified,
                        expected.is_empty(),
                        "arcs={arcs:?} t={t} y={y} banned={banned}"
                    );
                }
            }
        }
    }
}
