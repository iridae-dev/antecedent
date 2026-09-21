//! Brute-force evidence for visibility-aware response identification on MAGs.
//!
//! Every DAG over a few observed and latent binary variables is enumerated.
//! Its MAG, its latent projection and its interventional law are computed here
//! from first principles (bitmask ancestry, moral-graph d-separation, exact
//! enumeration of a randomly parameterised SCM) — none of it goes through the
//! identification code under test. Two claims are then checked:
//!
//! 1. the latent projection is an edge-subgraph of the confounded ADMG the
//!    identifier runs Shpitser–Pearl ID on, so ID is sound there;
//! 2. whenever the PAG response route reports a MAG identified, the functional
//!    it returns, evaluated on the exact observational law, equals the exact
//!    `E[Y | do(T = 1)]` of the generating DAG.
//!
//! The same sweep shows the check can fail: ID on the MAG read as an
//! unconfounded ADMG returns functionals that miss the truth.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::many_single_char_names,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::precedence
)]

use std::collections::HashMap;
use std::sync::Arc;

use antecedent_core::{
    CausalRng, IdentificationStatus, Intervention, ResponseFunctional, ResponseQuery, Value,
    VariableId,
};
use antecedent_expr::{Assignment, DistributionProvider, EvalContext, EvalError, FactorSpec};
use antecedent_graph::{Dag, DenseNodeId, Pag, is_mag_completion, latent_project};

use crate::generalized::{mag_to_admg, mag_to_confounded_admg};
use crate::id::IdIdentifier;
use crate::identifier::IdentificationWorkspace;
use crate::response_id::identify_pag_response_general;
use crate::result::IdentificationResult;

const MAX_NODES: usize = 8;

/// DAG whose nodes are numbered in a topological order; `parents[j]` is a mask
/// of positions `< j`. Observed nodes keep their relative order, so observed
/// id `i` is the `i`-th non-latent position.
#[derive(Clone, Copy)]
struct SmallDag {
    n: usize,
    latent: u8,
    parents: [u8; MAX_NODES],
}

impl SmallDag {
    fn from_index(n: usize, latent: u8, mut edges: u32) -> Self {
        let mut parents = [0u8; MAX_NODES];
        for (j, row) in parents.iter_mut().enumerate().take(n).skip(1) {
            *row = (edges & ((1 << j) - 1)) as u8;
            edges >>= j;
        }
        Self { n, latent, parents }
    }

    fn observed(&self) -> Vec<usize> {
        (0..self.n).filter(|&j| self.latent >> j & 1 == 0).collect()
    }

    fn children(&self, v: usize) -> u8 {
        (0..self.n).filter(|&c| self.parents[c] >> v & 1 == 1).fold(0, |m, c| m | 1 << c)
    }

    /// Reflexive ancestor masks.
    fn ancestors(&self) -> [u8; MAX_NODES] {
        let mut anc = [0u8; MAX_NODES];
        for j in 0..self.n {
            anc[j] = 1 << j;
            for p in 0..j {
                if self.parents[j] >> p & 1 == 1 {
                    anc[j] |= anc[p];
                }
            }
        }
        anc
    }

    /// d-separation by the moralised ancestral graph (Lauritzen et al. 1990).
    fn d_separated(&self, anc: [u8; MAX_NODES], x: usize, y: usize, z: u8) -> bool {
        let mut keep = anc[x] | anc[y];
        for (j, a) in anc.iter().enumerate().take(self.n) {
            if z >> j & 1 == 1 {
                keep |= a;
            }
        }
        let mut adj = [0u8; MAX_NODES];
        for c in 0..self.n {
            if keep >> c & 1 == 0 {
                continue;
            }
            let pa = self.parents[c];
            for p in 0..self.n {
                if pa >> p & 1 == 1 {
                    adj[p] |= (1 << c) | (pa & !(1 << p));
                    adj[c] |= 1 << p;
                }
            }
        }
        let mut seen = 1u8 << x;
        let mut frontier = seen;
        while frontier != 0 {
            let v = frontier.trailing_zeros() as usize;
            frontier &= frontier - 1;
            let next = adj[v] & !z & !seen;
            seen |= next;
            frontier |= next;
        }
        seen >> y & 1 == 0
    }

    /// Observed nodes reachable from each node along directed paths whose
    /// interior is latent.
    fn reach_through_latents(&self) -> [u8; MAX_NODES] {
        let mut reach = [0u8; MAX_NODES];
        for v in (0..self.n).rev() {
            let ch = self.children(v);
            for c in 0..self.n {
                if ch >> c & 1 == 1 {
                    reach[v] |= if self.latent >> c & 1 == 1 { reach[c] } else { 1 << c };
                }
            }
        }
        reach
    }
}

/// Pair code over observed ids `a < b`: 1 `a -> b`, 2 `b -> a`, 3 `a <-> b`.
type PairMarks = Vec<(usize, usize, u8)>;

/// The MAG of `dag` over its observed nodes (Richardson & Spirtes 2002): `a`
/// and `b` are adjacent iff no observed set d-separates them, and the edge
/// marks follow ancestry.
fn mag_of(dag: &SmallDag) -> PairMarks {
    let obs = dag.observed();
    let anc = dag.ancestors();
    let mut out = Vec::new();
    for (i, &a) in obs.iter().enumerate() {
        for (j, &b) in obs.iter().enumerate().skip(i + 1) {
            let others: Vec<usize> = obs.iter().copied().filter(|&o| o != a && o != b).collect();
            let inseparable = (0..1u32 << others.len()).all(|mask| {
                let z = others
                    .iter()
                    .enumerate()
                    .filter(|(bit, _)| mask >> bit & 1 == 1)
                    .fold(0u8, |m, (_, &o)| m | 1 << o);
                !dag.d_separated(anc, a, b, z)
            });
            if inseparable {
                let code = if anc[b] >> a & 1 == 1 {
                    1
                } else if anc[a] >> b & 1 == 1 {
                    2
                } else {
                    3
                };
                out.push((i, j, code));
            }
        }
    }
    out
}

fn node(i: usize) -> DenseNodeId {
    DenseNodeId::from_raw(i as u32)
}

fn pag_from_marks(k: usize, marks: &PairMarks) -> Pag {
    let mut pag = Pag::with_variables(k as u32);
    for &(a, b, code) in marks {
        match code {
            1 => pag.insert_directed(node(a), node(b)).unwrap(),
            2 => pag.insert_directed(node(b), node(a)).unwrap(),
            _ => pag.insert_bidirected(node(a), node(b)).unwrap(),
        }
    }
    pag
}

type Pairs = Vec<(usize, usize)>;

/// Latent projection over observed ids: `(directed, bidirected)` pair lists.
fn projection_of(dag: &SmallDag) -> (Pairs, Pairs) {
    let obs = dag.observed();
    let id_of = |pos: usize| obs.iter().position(|&o| o == pos).unwrap();
    let reach = dag.reach_through_latents();
    let mut directed = Vec::new();
    let mut bidirected = Vec::new();
    for (v, reached) in reach.iter().enumerate().take(dag.n) {
        let targets: Vec<usize> = (0..dag.n).filter(|&c| reached >> c & 1 == 1).collect();
        if dag.latent >> v & 1 == 0 {
            directed.extend(targets.iter().map(|&c| (id_of(v), id_of(c))));
        } else {
            for (i, &a) in targets.iter().enumerate() {
                for &b in &targets[i + 1..] {
                    let pair = (id_of(a).min(id_of(b)), id_of(a).max(id_of(b)));
                    if !bidirected.contains(&pair) {
                        bidirected.push(pair);
                    }
                }
            }
        }
    }
    (directed, bidirected)
}

/// Exact observational law over the observed binary variables.
struct ObservedLaw {
    k: usize,
    pmf: Vec<f64>,
}

impl ObservedLaw {
    fn mass(&self, mask: usize, value: usize) -> f64 {
        self.pmf.iter().enumerate().filter(|(i, _)| i & mask == value).map(|(_, p)| p).sum()
    }
}

fn bit_of(assignment: &Assignment, var: VariableId) -> Result<usize, EvalError> {
    let value =
        assignment.get(var).and_then(Value::as_f64).ok_or(EvalError::MissingBinding(var))?;
    Ok(usize::from(value == 1.0))
}

impl DistributionProvider for ObservedLaw {
    // Every leaf is read as the observational conditional of `variables` given
    // `conditioned_on`, with do-targets held at their assigned level: a target
    // outside `variables` is conditioned on (`P(y | z; do(t))` is `P(y | t, z)`).
    fn probability(
        &self,
        spec: &FactorSpec<'_>,
        assignment: &Assignment,
        _ctx: &EvalContext,
    ) -> Result<f64, EvalError> {
        let (mut num_mask, mut num_val, mut den_mask, mut den_val) = (0, 0, 0, 0);
        for &v in spec.variables {
            num_mask |= 1 << v.raw();
            num_val |= bit_of(assignment, v)? << v.raw();
        }
        let given = spec
            .conditioned_on
            .iter()
            .copied()
            .chain(spec.intervention.iter().map(|a| a.variable))
            .filter(|v| !spec.variables.contains(v));
        for v in given {
            den_mask |= 1 << v.raw();
            den_val |= bit_of(assignment, v)? << v.raw();
        }
        let den = self.mass(den_mask, den_val);
        if den <= 0.0 {
            return Err(EvalError::DivisionByZero);
        }
        Ok(self.mass(num_mask | den_mask, num_val | den_val) / den)
    }

    fn support(
        &self,
        vars: &[VariableId],
        _ctx: &EvalContext,
    ) -> Result<Arc<[Arc<[Value]>]>, EvalError> {
        assert!(vars.iter().all(|v| (v.raw() as usize) < self.k));
        Ok((0..1usize << vars.len())
            .map(|row| {
                (0..vars.len()).map(|i| Value::f64((row >> i & 1) as f64)).collect::<Arc<[_]>>()
            })
            .collect())
    }

    fn outcome(
        &self,
        var: VariableId,
        assignment: &Assignment,
        _ctx: &EvalContext,
    ) -> Result<f64, EvalError> {
        assignment.get(var).and_then(Value::as_f64).ok_or(EvalError::MissingBinding(var))
    }

    fn n_draws(&self) -> Option<usize> {
        None
    }
}

/// A randomly parameterised binary SCM on `dag`, enumerated exactly.
struct ExactScm {
    dag: SmallDag,
    /// `cpt[j][parent configuration]` is `P(V_j = 1 | pa)`.
    cpt: Vec<Vec<f64>>,
}

impl ExactScm {
    fn random(dag: SmallDag, seed: u64) -> Self {
        let mut rng = CausalRng::from_seed(seed);
        let cpt = (0..dag.n)
            .map(|j| {
                (0..1usize << dag.parents[j].count_ones())
                    .map(|_| 0.15 + 0.7 * rng.next_f64())
                    .collect()
            })
            .collect();
        Self { dag, cpt }
    }

    fn factor(&self, j: usize, world: usize) -> f64 {
        let mut config = 0;
        let mut slot = 0;
        for p in 0..self.dag.n {
            if self.dag.parents[j] >> p & 1 == 1 {
                config |= (world >> p & 1) << slot;
                slot += 1;
            }
        }
        let p1 = self.cpt[j][config];
        if world >> j & 1 == 1 { p1 } else { 1.0 - p1 }
    }

    fn observed_law(&self) -> ObservedLaw {
        let obs = self.dag.observed();
        let mut pmf = vec![0.0; 1 << obs.len()];
        for world in 0..1usize << self.dag.n {
            let p: f64 = (0..self.dag.n).map(|j| self.factor(j, world)).product();
            let cell = obs.iter().enumerate().fold(0, |m, (i, &o)| m | (world >> o & 1) << i);
            pmf[cell] += p;
        }
        ObservedLaw { k: obs.len(), pmf }
    }

    /// `E[Y | do(T = 1)]` by the truncated factorisation.
    fn mean_under_do_one(&self, t: usize, y: usize) -> f64 {
        (0..1usize << self.dag.n)
            .filter(|world| world >> t & 1 == 1 && world >> y & 1 == 1)
            .map(|world| {
                (0..self.dag.n).filter(|&j| j != t).map(|j| self.factor(j, world)).product::<f64>()
            })
            .sum()
    }
}

fn do_one(t: usize, y: usize) -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(y as u32),
        interventions: Arc::from([Intervention::set(
            VariableId::from_raw(t as u32),
            Value::f64(1.0),
        )]),
    })
}

fn identified(result: IdentificationResult) -> Option<IdentificationResult> {
    (result.status == IdentificationStatus::NonparametricallyIdentified
        && !result.estimands.is_empty())
    .then_some(result)
}

fn evaluate(result: &IdentificationResult, law: &ObservedLaw) -> f64 {
    result
        .arena
        .compile(result.estimands[0].functional)
        .unwrap()
        .evaluate(&result.arena, law, &EvalContext::default())
        .unwrap()
}

#[derive(Default)]
struct PerMag {
    /// Route under test, per ordered `(t, y)`.
    route: HashMap<(usize, usize), Option<IdentificationResult>>,
    /// ID on the MAG read as an unconfounded ADMG (the unsound reading).
    naive: HashMap<(usize, usize), Option<IdentificationResult>>,
}

#[derive(Debug, Default)]
struct Tally {
    dags: u64,
    mags: usize,
    numeric_checks: u64,
    identified_by_adjustment: u64,
    identified_beyond_adjustment: u64,
    /// Of those, the ones where `T` is an ancestor of `Y` (a non-null effect).
    causal_beyond_adjustment: u64,
    naive_wrong: u64,
    worst_error: f64,
}

/// Both claims on one DAG; the numeric claim only when `numeric` is set.
fn check(
    dag: &SmallDag,
    numeric: bool,
    seed: u64,
    cache: &mut HashMap<PairMarks, PerMag>,
    tally: &mut Tally,
) {
    let obs = dag.observed();
    let k = obs.len();
    tally.dags += 1;
    let marks = mag_of(dag);
    let mag = pag_from_marks(k, &marks);
    assert!(is_mag_completion(&mag), "the MAG of a DAG is maximal and ancestral: {marks:?}");

    let (directed, bidirected) = projection_of(dag);
    let worst = mag_to_confounded_admg(&mag).expect("directed/bidirected MAG");
    for &(a, b) in &directed {
        assert!(worst.children(node(a)).contains(&node(b)), "{marks:?} lacks {a}->{b}");
    }
    for &(a, b) in &bidirected {
        assert!(
            worst.bidirected_neighbors(node(a)).contains(&node(b)),
            "{marks:?} lacks {a}<->{b}: a visible edge hid a latent common cause"
        );
    }
    cross_check_library_projection(dag, &directed, &bidirected);
    if !numeric {
        return;
    }

    let scm = ExactScm::random(*dag, seed);
    let law = scm.observed_law();
    let anc = dag.ancestors();
    let entry = cache.entry(marks).or_default();
    for t in 0..k {
        for y in (0..k).filter(|&y| y != t) {
            let truth = scm.mean_under_do_one(obs[t], obs[y]);
            let route = entry.route.entry((t, y)).or_insert_with(|| {
                let env = identify_pag_response_general(&mag, &do_one(t, y)).unwrap();
                assert_eq!(env.cases.len(), 1);
                identified(env.cases[0].result.clone())
            });
            if let Some(result) = route {
                let error = (evaluate(result, &law) - truth).abs();
                assert!(error < 1e-10, "t={t} y={y} error={error} on {mag:?}");
                tally.worst_error = tally.worst_error.max(error);
                tally.numeric_checks += 1;
                if result.estimands[0].method.as_ref() == "general.id" {
                    tally.identified_beyond_adjustment += 1;
                    if anc[obs[y]] >> obs[t] & 1 == 1 {
                        tally.causal_beyond_adjustment += 1;
                    }
                } else {
                    tally.identified_by_adjustment += 1;
                }
            }
            let naive = entry.naive.entry((t, y)).or_insert_with(|| {
                let admg = mag_to_admg(&mag).unwrap();
                let id = IdIdentifier::new();
                let prepared = id.prepare(&admg).unwrap();
                id.identify_response(
                    &prepared,
                    &do_one(t, y),
                    &mut IdentificationWorkspace::default(),
                )
                .ok()
                .and_then(identified)
            });
            if let Some(result) = naive {
                if (evaluate(result, &law) - truth).abs() > 1e-6 {
                    tally.naive_wrong += 1;
                }
            }
        }
    }
}

fn useless_latent(dag: &SmallDag) -> bool {
    (0..dag.n).any(|v| dag.latent >> v & 1 == 1 && dag.children(v).count_ones() < 2)
}

/// Sweep every DAG with `k` observed and `l` latent nodes (latents with fewer
/// than two children add nothing to the projection and are skipped). The
/// numeric check runs on every `stride`-th DAG.
fn sweep(k: usize, l: usize, stride: u64, tally: &mut Tally) {
    let n = k + l;
    let mut cache = HashMap::new();
    let mut counter = 0u64;
    for latent in 0..1u8 << n {
        if latent.count_ones() as usize != l {
            continue;
        }
        for edges in 0..1u32 << (n * (n - 1) / 2) {
            let dag = SmallDag::from_index(n, latent, edges);
            if useless_latent(&dag) {
                continue;
            }
            counter += 1;
            let seed = 0x5EED ^ (u64::from(edges) << 8) ^ u64::from(latent);
            check(&dag, counter % stride == 0, seed, &mut cache, tally);
        }
    }
    tally.mags += cache.len();
}

/// Uniformly drawn edge sets and latent positions, for sizes too large to enumerate.
fn random_sweep(k: usize, l: usize, samples: usize, seed: u64, tally: &mut Tally) {
    let n = k + l;
    let mut rng = CausalRng::from_seed(seed);
    let mut cache = HashMap::new();
    for _ in 0..samples {
        let mut latent = 0u8;
        while latent.count_ones() as usize != l {
            latent |= 1 << (rng.next_u64() % n as u64);
        }
        let edges = (rng.next_u64() & rng.next_u64()) as u32 & ((1u32 << (n * (n - 1) / 2)) - 1);
        let dag = SmallDag::from_index(n, latent, edges);
        check(&dag, true, rng.next_u64(), &mut cache, tally);
    }
    tally.mags += cache.len();
}

fn cross_check_library_projection(
    dag: &SmallDag,
    directed: &[(usize, usize)],
    bidirected: &[(usize, usize)],
) {
    let mut g = Dag::with_variables(dag.n as u32);
    for c in 0..dag.n {
        for p in 0..dag.n {
            if dag.parents[c] >> p & 1 == 1 {
                g.insert_directed(node(p), node(c)).unwrap();
            }
        }
    }
    let observed: Vec<_> = dag.observed().into_iter().map(node).collect();
    let admg = latent_project(&g, &observed).unwrap();
    let k = observed.len();
    for a in 0..k {
        for b in 0..k {
            assert_eq!(
                admg.children(node(a)).contains(&node(b)),
                directed.contains(&(a, b)),
                "directed projection {a}->{b}"
            );
            if a < b {
                assert_eq!(
                    admg.bidirected_neighbors(node(a)).contains(&node(b)),
                    bidirected.contains(&(a, b)),
                    "bidirected projection {a}<->{b}"
                );
            }
        }
    }
}

#[test]
fn confounded_admg_contains_every_projection_and_identified_means_are_exact() {
    let mut tally = Tally::default();
    for (k, l, stride) in [(2, 1, 1), (2, 2, 1), (3, 1, 1), (3, 2, 1), (4, 1, 1), (4, 2, 97)] {
        sweep(k, l, stride, &mut tally);
    }
    random_sweep(5, 2, 1_500, 0xA11CE, &mut tally);
    random_sweep(5, 3, 1_500, 0xB0B, &mut tally);
    eprintln!("{tally:?}");
    assert!(tally.numeric_checks > 10_000, "{tally:?}");
    assert!(
        tally.identified_beyond_adjustment > 0,
        "the sweep must exercise identification that adjustment cannot reach: {tally:?}"
    );
    assert!(
        tally.naive_wrong > 0,
        "reading a MAG as an unconfounded ADMG must be caught by this oracle: {tally:?}"
    );
}

fn dag_with(n: usize, latent: u8, rows: &[(usize, u8)]) -> SmallDag {
    let mut parents = [0u8; MAX_NODES];
    for &(child, mask) in rows {
        parents[child] = mask;
    }
    SmallDag { n, latent, parents }
}

/// Identify `E[obs y | do(obs t = 1)]` on the MAG of `dag`; when identified,
/// the worst error against the exact truth over several parameterisations.
fn route_error(dag: &SmallDag, t: usize, y: usize) -> Option<(String, f64)> {
    let obs = dag.observed();
    let mag = pag_from_marks(obs.len(), &mag_of(dag));
    let env = identify_pag_response_general(&mag, &do_one(t, y)).unwrap();
    assert_eq!(env.cases.len(), 1);
    let result = identified(env.cases[0].result.clone())?;
    let worst = (0..32)
        .map(|seed| {
            let scm = ExactScm::random(*dag, seed);
            (evaluate(&result, &scm.observed_law()) - scm.mean_under_do_one(obs[t], obs[y])).abs()
        })
        .fold(0.0, f64::max);
    Some((result.estimands[0].method.to_string(), worst))
}

/// `T <- L -> Y`, `T -> Y`: the MAG is the single invisible edge `T -> Y`. The
/// route refuses; `P(y | t)`, which the unconfounded reading returns, is wrong.
#[test]
fn invisible_edge_is_refused_and_the_unconfounded_reading_is_numerically_wrong() {
    let dag = dag_with(3, 0b1, &[(1, 0b1), (2, 0b11)]);
    assert_eq!(mag_of(&dag), vec![(0, 1, 1)]);
    assert!(route_error(&dag, 0, 1).is_none());

    let mag = pag_from_marks(2, &mag_of(&dag));
    let id = IdIdentifier::new();
    let prepared = id.prepare(&mag_to_admg(&mag).unwrap()).unwrap();
    let naive = id
        .identify_response(&prepared, &do_one(0, 1), &mut IdentificationWorkspace::default())
        .unwrap();
    let scm = ExactScm::random(dag, 3);
    let gap = (evaluate(&naive, &scm.observed_law()) - scm.mean_under_do_one(1, 2)).abs();
    assert!(gap > 1e-3, "confounding bias must be visible: {gap}");
}

/// `R -> T -> Y` with `R` not adjacent to `Y`: `R` witnesses that `T -> Y` is visible.
#[test]
fn direct_witness_makes_the_edge_visible_and_the_mean_is_exact() {
    let dag = dag_with(3, 0, &[(1, 0b1), (2, 0b10)]);
    let mag = pag_from_marks(3, &mag_of(&dag));
    assert!(crate::joint_response::visible(&mag, node(1), node(2)));
    let (_, error) = route_error(&dag, 1, 2).expect("visible edge identifies");
    assert!(error < 1e-12, "{error}");
}

/// `D -> C <-> T`, `C -> Y`, `T -> Y`: `D` is not adjacent to `Y` and reaches `T`
/// along a collider path whose interior `C` is a parent of `Y`, so `T -> Y` is
/// visible without any direct witness.
#[test]
fn collider_path_witness_makes_the_edge_visible_and_the_mean_is_exact() {
    // Topological positions: L D C T Y; observed ids D=0 C=1 T=2 Y=3.
    let dag = dag_with(5, 0b1, &[(2, 0b11), (3, 0b1), (4, 0b1100)]);
    assert_eq!(mag_of(&dag), vec![(0, 1, 1), (1, 2, 3), (1, 3, 1), (2, 3, 1)]);
    let mag = pag_from_marks(4, &mag_of(&dag));
    assert!(!mag.neighbors(node(2)).any(|(v, at_t, _)| {
        v != node(3) && at_t == antecedent_graph::Endpoint::Arrow && !mag.has_edge(v, node(3))
    }));
    assert!(crate::joint_response::visible(&mag, node(2), node(3)));
    let (_, error) = route_error(&dag, 2, 3).expect("visible edge identifies");
    assert!(error < 1e-12, "{error}");
}

/// `T -> M -> Y`, `M -> B -> Y`, `A -> Y` with latent `T <- L1 -> A <- L2 -> B`.
/// The back-door path `T <-> A -> Y` needs `A`, which opens `T <-> A <-> B -> Y`,
/// and `B` is a descendant of the mediator: no adjustment set exists. Every
/// directed edge is visible, so ID on the MAG separates `T` from its child.
#[test]
fn visible_edges_identify_a_causal_effect_that_no_adjustment_set_reaches() {
    // Topological positions: L1 L2 T A M B Y.
    let mut parents = [0u8; MAX_NODES];
    parents[2] = 0b1; // T <- L1
    parents[3] = 0b11; // A <- L1, L2
    parents[4] = 0b100; // M <- T
    parents[5] = 0b1_0010; // B <- M, L2
    parents[6] = 0b11_1000; // Y <- A, M, B
    let dag = SmallDag { n: 7, latent: 0b11, parents };
    let (t, a, m, b, y) = (0, 1, 2, 3, 4);
    assert_eq!(
        mag_of(&dag),
        vec![(t, a, 3), (t, m, 1), (a, b, 3), (a, y, 1), (m, b, 1), (m, y, 1), (b, y, 1)]
    );
    let mag = pag_from_marks(5, &mag_of(&dag));
    assert!(crate::generalized::invisible_directed_edges(&mag).is_empty());

    let env = identify_pag_response_general(&mag, &do_one(t, y)).unwrap();
    assert_eq!(env.status, IdentificationStatus::NonparametricallyIdentified);
    let result = &env.cases[0].result;
    assert_eq!(result.estimands[0].method.as_ref(), "general.id");
    for seed in 0..32 {
        let scm = ExactScm::random(dag, seed);
        let truth = scm.mean_under_do_one(2, 6);
        let error = (evaluate(result, &scm.observed_law()) - truth).abs();
        assert!(error < 1e-12, "seed={seed} error={error}");
    }

    let mut tally = Tally::default();
    check(&dag, true, 7, &mut HashMap::new(), &mut tally);
    assert!(tally.causal_beyond_adjustment >= 1, "{tally:?}");
}

/// Every DAG over four observed and two latent nodes, numerically checked.
#[test]
#[ignore = "exhaustive sweep; minutes in a debug build"]
fn exhaustive_four_observed_two_latent() {
    let mut tally = Tally::default();
    sweep(4, 2, 1, &mut tally);
    eprintln!("{tally:?}");
}
