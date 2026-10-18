//! Sparse connected structural causal models of ten to twenty variables with a known total
//! effect, for the discovery-then-estimation studies.
//!
//! Three topologies, with maximum degree at most four, connected, and with every edge incident to
//! the treatment oriented by its equivalence class while the nuisance chains hanging off
//! exogenous roots stay undirected (so the class is a partially directed graph, never a
//! single DAG):
//!
//! * `hub_confounded_multi_path`: a hub `h` confounds the treatment `x` and the outcome `y`;
//!   `x` has two parents (`h`, `p`) and reaches `y` through the mediator `m`, while the hub opens
//!   the second, confounding path `x <- h -> y`; nuisance chains hang off the hub (twice, so the
//!   hub has degree four), the treatment's other parent and the mediator.
//! * `layered_multi_mediator_nuisance`: two exogenous parents `a1`, `a2` of `x`; `x` reaches
//!   `y` through a chain of two mediators `m1`, `m2`; a mediator-outcome confounder `k` acts on
//!   `m2` and `y`; nuisance chains hang off `a1`, `a2`, `k` and `m2`.
//! * `confounded_direct_and_mediated`: the hub `h` confounds `x` and `y`, and `x` reaches `y`
//!   both directly and through the mediator `m` (several causal paths in one confounded
//!   graph). Its coefficients are a fixed base vector with a small jitter, because the
//!   strong-faithfulness margin of this denser core is met only near that base.
//!
//! The model is linear Gaussian and standardized: every column has unit variance, each node's
//! noise variance is what its parents leave unexplained, so a structural coefficient is also
//! the standardized one and the correlation along a path is the product of its coefficients. A
//! log-linear study exponentiates every column, so the logarithm of each column follows the
//! same law. The truth is computed from the structural coefficients alone, by summing the
//! products of coefficients over the directed paths from the treatment to the outcome, never
//! from simulated data.
//!
//! A redraw rule enforces the strong-faithfulness margin the family publishes: for every
//! adjacent pair, and every pair with a common neighbour (the pairs whose separating set PC's
//! orientation reads), the population partial correlation given each conditioning set PC can
//! form from the true neighbourhoods (at most [`MAX_DEGREE`] members of one endpoint's
//! neighbours) is at least [`MIN_MARGIN`] in magnitude, whenever the set does not d-separate the
//! pair and is no larger than the smallest set that does. Conditioning sets larger than the
//! smallest separating set never decide a pair, because PC removes it there. The additional
//! tests formed from still-standing false adjacencies can be inspected with
//! [`Scm::population_pc_margin`]; they do not change admission to this measured family.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::similar_names,
    dead_code,
    reason = "bounded simulation sizes and the single-letter names of the laws themselves"
)]

use std::collections::BTreeSet;

/// Smallest population partial correlation the published family admits for a true edge or the
/// endpoints of an unshielded collider.
pub const MIN_MARGIN: f64 = 0.15;
/// Smallest magnitude of the total effect of the treatment on the outcome.
pub const MIN_TOTAL_EFFECT: f64 = 0.1;
/// Largest degree of any node.
pub const MAX_DEGREE: usize = 4;
/// Magnitude range of a structural coefficient among the core nodes.
const CORE_WEIGHT: (f64, f64) = (0.4, 0.7);
/// Magnitude range of a structural coefficient along a nuisance chain.
const CHAIN_WEIGHT: (f64, f64) = (0.5, 0.7);
/// Half-width of the jitter around the base coefficients of the denser core.
const BASE_JITTER: f64 = 0.03;
/// Base coefficients of `confounded_direct_and_mediated` in node order (`h`, `p`, `x` on
/// `(h, p)`, `m` on `x`, `y` on `(x, m, h)`, then the nuisance chains), chosen so the margin is
/// 0.18 for every prefix up to twenty variables.
const BASE_DENSE: [f64; 21] = [
    -0.39, 0.59, -0.73, -0.61, 0.34, -0.64, -0.7, -0.62, 0.78, 0.7, -0.53, 0.56, 0.6, 0.43, 0.69,
    -0.71, 0.65, -0.61, 0.46, 0.62, -0.66,
];
/// Magnitude range of every coefficient of the weak-signal boundary family.
const WEAK_WEIGHT: (f64, f64) = (0.1, 0.25);
/// Smallest variance a node's own noise keeps.
const MIN_NOISE_VARIANCE: f64 = 0.2;
/// A partial correlation below this is an exact conditional independence of the population.
const SEPARATED: f64 = 1e-9;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Topology {
    HubConfoundedMultiPath,
    LayeredMultiMediatorNuisance,
    ConfoundedDirectAndMediated,
}

impl Topology {
    pub const fn name(self) -> &'static str {
        match self {
            Self::HubConfoundedMultiPath => "hub_confounded_multi_path",
            Self::LayeredMultiMediatorNuisance => "layered_multi_mediator_nuisance",
            Self::ConfoundedDirectAndMediated => "confounded_direct_and_mediated",
        }
    }

    /// The topologies a family declared with maximum degree `max_degree` contains: the denser
    /// third core has degree four, so it belongs to degree four only.
    pub fn for_degree(max_degree: usize) -> Vec<Self> {
        let mut all = vec![Self::HubConfoundedMultiPath, Self::LayeredMultiMediatorNuisance];
        if max_degree >= MAX_DEGREE {
            all.push(Self::ConfoundedDirectAndMediated);
        }
        all
    }
}

/// How a family member departs from the published conditions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    /// The published family: coefficients redrawn until the margin holds.
    Licensed,
    /// Coefficients of magnitude 0.1 to 0.25, with no margin rule: signal below the published
    /// strength.
    WeakSignal,
    /// The confounder's direct coefficient on the outcome is set to cancel the path through the
    /// mediator exactly, so the confounder and the outcome are marginally independent: an
    /// unfaithful law.
    CancellingPath,
}

/// `SplitMix64` stream: every draw is a finalized hash of a counter.
#[derive(Clone)]
pub struct Stream(u64);

impl Stream {
    pub const fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (f64::from(u32::try_from(z >> 40).expect("24 bits")) + 0.5) / f64::from(1u32 << 24)
    }

    pub fn normal(&mut self) -> f64 {
        let (u, v) = (self.unit(), self.unit());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }

    /// A centred, unit-variance, right-skewed draw: `(z^2 - 1) / sqrt(2)`.
    pub fn skewed(&mut self) -> f64 {
        let z = self.normal();
        (z * z - 1.0) / std::f64::consts::SQRT_2
    }

    fn signed(&mut self, low: f64, high: f64) -> f64 {
        let magnitude = low + (high - low) * self.unit();
        if self.unit() < 0.5 { -magnitude } else { magnitude }
    }
}

/// A structural causal model: nodes in topological order, each with its weighted parents and
/// the standard deviation of its own noise.
#[derive(Clone, Debug)]
pub struct Scm {
    pub topology: Topology,
    pub names: Vec<String>,
    /// `parents[v]` lists `(parent, weight)` with `parent < v`.
    pub parents: Vec<Vec<(usize, f64)>>,
    /// Noise standard deviation of each node: the square root of what its parents leave of a
    /// unit variance.
    pub noise_sd: Vec<f64>,
    pub treatment: usize,
    pub outcome: usize,
}

/// The unweighted shape: parent lists in topological order and the roles.
struct Shape {
    names: Vec<String>,
    parents: Vec<Vec<usize>>,
    treatment: usize,
    outcome: usize,
    /// Nodes before the nuisance chains.
    core: usize,
    /// Base coefficients in node order, when the topology draws around a base.
    base: Option<&'static [f64]>,
}

struct Builder {
    names: Vec<String>,
    parents: Vec<Vec<usize>>,
}

impl Builder {
    fn node(&mut self, name: &str, parents: &[usize]) -> usize {
        self.names.push(name.to_owned());
        self.parents.push(parents.to_vec());
        self.names.len() - 1
    }
}

/// Hang `count` nuisance nodes in round-robin chains off `anchors`.
fn chains(builder: &mut Builder, label: &str, anchors: &[usize], count: usize) {
    let mut tails = anchors.to_vec();
    for j in 0..count {
        let slot = j % anchors.len();
        let node = builder.node(&format!("{label}{j}"), &[tails[slot]]);
        tails[slot] = node;
    }
}

fn shape(topology: Topology, variables: usize, degree_cap: usize) -> Shape {
    let mut b = Builder { names: Vec::new(), parents: Vec::new() };
    match topology {
        Topology::HubConfoundedMultiPath => {
            let h = b.node("h", &[]);
            let p = b.node("p", &[]);
            let x = b.node("x", &[h, p]);
            let m = b.node("m", &[x]);
            let y = b.node("y", &[m, h]);
            let core = b.names.len();
            let anchors: &[usize] =
                if degree_cap >= MAX_DEGREE { &[h, p, m, h] } else { &[h, p, m] };
            chains(&mut b, "n", anchors, variables - core);
            Shape { names: b.names, parents: b.parents, treatment: x, outcome: y, core, base: None }
        }
        Topology::LayeredMultiMediatorNuisance => {
            let a1 = b.node("a1", &[]);
            let a2 = b.node("a2", &[]);
            let k = b.node("k", &[]);
            let x = b.node("x", &[a1, a2]);
            let m1 = b.node("m1", &[x]);
            let m2 = b.node("m2", &[m1, k]);
            let y = b.node("y", &[m2, k]);
            let core = b.names.len();
            chains(&mut b, "n", &[a1, a2, k, m2], variables - core);
            Shape { names: b.names, parents: b.parents, treatment: x, outcome: y, core, base: None }
        }
        Topology::ConfoundedDirectAndMediated => {
            assert!(degree_cap >= MAX_DEGREE, "the denser core has degree four");
            let h = b.node("h", &[]);
            let p = b.node("p", &[]);
            let x = b.node("x", &[h, p]);
            let m = b.node("m", &[x]);
            let y = b.node("y", &[x, m, h]);
            let core = b.names.len();
            chains(&mut b, "n", &[h, p, m, h], variables - core);
            Shape {
                names: b.names,
                parents: b.parents,
                treatment: x,
                outcome: y,
                core,
                base: Some(&BASE_DENSE),
            }
        }
    }
}

impl Scm {
    /// A model of `variables` nodes whose weights are drawn from the stream of `seed`, redrawn
    /// until the noise variances, the faithfulness margin and the total effect are admitted.
    ///
    /// # Panics
    /// Fewer than ten or more than twenty variables, or no admitted draw in 100,000 attempts.
    pub fn new(topology: Topology, variables: usize, seed: u64) -> Self {
        Self::with_variant(topology, variables, seed, Variant::Licensed)
    }

    /// As [`Self::new`], for a family member that departs from the published conditions.
    pub fn with_variant(topology: Topology, variables: usize, seed: u64, variant: Variant) -> Self {
        Self::with_degree(topology, variables, seed, variant, MAX_DEGREE)
    }

    /// As [`Self::with_variant`] for a family whose graphs have degree at most `degree_cap`
    /// (three or four; the denser core needs four).
    pub fn with_degree(
        topology: Topology,
        variables: usize,
        seed: u64,
        variant: Variant,
        degree_cap: usize,
    ) -> Self {
        assert!((10..=20).contains(&variables), "ten to twenty variables");
        let shape = shape(topology, variables, degree_cap);
        assert_eq!(shape.names.len(), variables);
        let mut stream = Stream::new(seed);
        for _ in 0..100_000 {
            let admitted = Self::draw(&shape, topology, variant, &mut stream).filter(|scm| {
                variant != Variant::Licensed
                    || (scm.total_effect().abs() >= MIN_TOTAL_EFFECT
                        && scm.margin_with_cutoff(MIN_MARGIN, false) >= MIN_MARGIN)
            });
            if let Some(scm) = admitted {
                return scm;
            }
        }
        panic!("no admitted draw for {} with {variables} variables", topology.name());
    }

    /// One draw of the weights, or `None` when some node's parents explain too much of it.
    fn draw(
        shape: &Shape,
        topology: Topology,
        variant: Variant,
        stream: &mut Stream,
    ) -> Option<Self> {
        let p = shape.names.len();
        let mut cov = vec![vec![0.0; p]; p];
        let mut parents = Vec::with_capacity(p);
        let mut noise_sd = Vec::with_capacity(p);
        let mut admitted = true;
        let mut drawn = 0usize;
        for (child, list) in shape.parents.iter().enumerate() {
            let mut weighted: Vec<(usize, f64)> = list
                .iter()
                .map(|&parent| {
                    let weight = match (variant, shape.base) {
                        (Variant::WeakSignal, _) | (_, None) => {
                            let (low, high) = match variant {
                                Variant::WeakSignal => WEAK_WEIGHT,
                                _ if child < shape.core => CORE_WEIGHT,
                                _ => CHAIN_WEIGHT,
                            };
                            stream.signed(low, high)
                        }
                        (_, Some(base)) => base[drawn] + BASE_JITTER * (2.0 * stream.unit() - 1.0),
                    };
                    drawn += 1;
                    (parent, weight)
                })
                .collect();
            if variant == Variant::CancellingPath && child == shape.outcome {
                // The confounder is the outcome's last parent: choose its coefficient so that
                // its covariance with the outcome is zero.
                let (&(confounder, _), rest) = weighted.split_last().expect("outcome parents");
                let cancelling: f64 =
                    -rest.iter().map(|&(a, w)| w * cov[confounder][a]).sum::<f64>();
                if let Some(last) = weighted.last_mut() {
                    last.1 = cancelling;
                }
            }
            let explained: f64 = weighted
                .iter()
                .flat_map(|&(a, wa)| weighted.iter().map(move |&(b, wb)| (a, wa, b, wb)))
                .map(|(a, wa, b, wb)| wa * wb * cov[a][b])
                .sum();
            let residual = 1.0 - explained;
            if residual < MIN_NOISE_VARIANCE {
                admitted = false;
            }
            for earlier in 0..child {
                let c: f64 = weighted.iter().map(|&(parent, w)| w * cov[parent][earlier]).sum();
                cov[child][earlier] = c;
                cov[earlier][child] = c;
            }
            cov[child][child] = 1.0;
            parents.push(weighted);
            noise_sd.push(residual.max(0.0).sqrt());
        }
        admitted.then(|| Self {
            topology,
            names: shape.names.clone(),
            parents,
            noise_sd,
            treatment: shape.treatment,
            outcome: shape.outcome,
        })
    }

    pub fn variables(&self) -> usize {
        self.names.len()
    }

    /// Column position of the node called `name`.
    pub fn index(&self, name: &str) -> usize {
        self.names.iter().position(|n| n == name).expect("a node of the model")
    }

    /// True directed edges as `(parent, child)`.
    pub fn edges(&self) -> BTreeSet<(usize, usize)> {
        self.parents
            .iter()
            .enumerate()
            .flat_map(|(child, list)| list.iter().map(move |&(parent, _)| (parent, child)))
            .collect()
    }

    pub fn degrees(&self) -> Vec<usize> {
        let mut degree = vec![0; self.variables()];
        for (a, b) in self.edges() {
            degree[a] += 1;
            degree[b] += 1;
        }
        degree
    }

    /// The treatment's parents.
    pub fn treatment_parents(&self) -> Vec<usize> {
        self.parents[self.treatment].iter().map(|&(p, _)| p).collect()
    }

    fn weight(&self, parent: usize, child: usize) -> f64 {
        self.parents[child].iter().find(|&&(p, _)| p == parent).map_or(0.0, |&(_, w)| w)
    }

    /// The total effect of the treatment on the outcome: the sum over directed paths of the
    /// product of the coefficients along the path. It is the change in the outcome's mean per
    /// unit set of the treatment, and on the log scale of a log-linear study the elasticity.
    pub fn total_effect(&self) -> f64 {
        self.path_sum(self.treatment)
    }

    fn path_sum(&self, from: usize) -> f64 {
        if from == self.outcome {
            return 1.0;
        }
        (from + 1..self.variables())
            .map(|child| {
                let w = self.weight(from, child);
                if w == 0.0 { 0.0 } else { w * self.path_sum(child) }
            })
            .sum()
    }

    /// The truth of a log-linear study: the ratio of the outcome's mean under the intervention
    /// that multiplies the treatment by `exp(log_shift)` to its mean under the baseline
    /// intervention. Given the intervention the log of the outcome is Gaussian with a mean that
    /// moves by `log_shift * total_effect` and a variance that does not depend on the
    /// intervention, so the ratio of means is the exponential of the shift in the log mean.
    pub fn mean_ratio(&self, log_shift: f64) -> f64 {
        (log_shift * self.total_effect()).exp()
    }

    /// Population covariance of the (linear-scale) columns: unit variances and, between
    /// `earlier` and `child`, the sum over the child's parents of weight times covariance.
    fn covariance(&self) -> Vec<Vec<f64>> {
        let p = self.variables();
        let mut cov = vec![vec![0.0; p]; p];
        for child in 0..p {
            for earlier in 0..child {
                let c: f64 = self.parents[child].iter().map(|&(a, w)| w * cov[a][earlier]).sum();
                cov[child][earlier] = c;
                cov[earlier][child] = c;
            }
            cov[child][child] = 1.0;
        }
        cov
    }

    /// The strong-faithfulness margin the family publishes (see the module documentation):
    /// the smallest absolute population partial correlation over the true-neighbourhood sets.
    pub fn margin(&self) -> f64 {
        self.margin_with_cutoff(0.0, false)
    }

    /// Diagnostic margin including every relevant test PC makes on the population covariance,
    /// including conditioning sets formed from false adjacencies it has not removed yet.
    /// This diagnostic never rejects or redraws a member of the measured family.
    pub fn population_pc_margin(&self) -> f64 {
        self.margin_with_cutoff(0.0, true)
    }

    /// Admission only needs to know whether the margin clears `cutoff`: stop at the first
    /// counterexample. This preserves every accepted draw and its stream position while
    /// avoiding the rest of the margin search on rejected draws. A zero cutoff computes
    /// the full minimum for the public diagnostic.
    fn margin_with_cutoff(&self, cutoff: f64, population_tests: bool) -> f64 {
        let p = self.variables();
        let cov = self.covariance();
        let edges = self.edges();
        let adjacent = |a: usize, b: usize| edges.contains(&(a.min(b), a.max(b)));
        let neighbours: Vec<BTreeSet<usize>> =
            (0..p).map(|v| (0..p).filter(|&u| u != v && adjacent(u, v)).collect()).collect();
        let partial = |u: usize, v: usize, given: &[usize]| -> f64 {
            let mut index = vec![u, v];
            index.extend_from_slice(given);
            let sub: Vec<Vec<f64>> =
                index.iter().map(|&i| index.iter().map(|&j| cov[i][j]).collect()).collect();
            let precision = invert(&sub);
            -precision[0][1] / (precision[0][0] * precision[1][1]).sqrt()
        };
        let mut margin = f64::INFINITY;
        for u in 0..p {
            for v in u + 1..p {
                if !adjacent(u, v) && neighbours[u].is_disjoint(&neighbours[v]) {
                    continue;
                }
                let mut sets: Vec<(usize, f64)> = Vec::new();
                for (end, other) in [(u, v), (v, u)] {
                    let pool: Vec<usize> =
                        neighbours[end].iter().copied().filter(|&w| w != other).collect();
                    for mask in 0u32..(1 << pool.len()) {
                        let size = mask.count_ones() as usize;
                        if size > MAX_DEGREE {
                            continue;
                        }
                        let given: Vec<usize> = pool
                            .iter()
                            .enumerate()
                            .filter(|&(k, _)| (mask >> k) & 1 == 1)
                            .map(|(_, &w)| w)
                            .collect();
                        sets.push((size, partial(u, v, &given).abs()));
                    }
                }
                let separating =
                    sets.iter().filter(|&&(_, r)| r < SEPARATED).map(|&(size, _)| size).min();
                for &(size, r) in &sets {
                    if r >= SEPARATED && separating.is_none_or(|k| size <= k) {
                        margin = margin.min(r);
                        if margin < cutoff {
                            return margin;
                        }
                    }
                }
            }
        }
        if !population_tests {
            return margin;
        }
        // PC forms its conditioning sets from the adjacencies still standing at each depth, not
        // from the true neighbours: a pair that is only separated at a later depth is still
        // adjacent while a true edge is tested, and the sets it contributes (for example the
        // outcome in the set that tests the treatment's mediator edge) can leave the true
        // edge's partial correlation near zero. Run PC on the population: every test that does
        // not separate a true edge, or a pair with a common neighbour, must also clear the
        // margin.
        let mut standing: Vec<Vec<bool>> =
            (0..p).map(|u| (0..p).map(|v| u != v).collect()).collect();
        for depth in 0..=MAX_DEGREE {
            for u in 0..p {
                for v in u + 1..p {
                    if !standing[u][v] {
                        continue;
                    }
                    let near = |w: usize| -> Vec<usize> {
                        (0..p).filter(|&q| q != w && q != u + v - w && standing[w][q]).collect()
                    };
                    let mut candidates = combinations(&near(u), depth);
                    candidates.extend(combinations(&near(v), depth));
                    candidates.sort();
                    candidates.dedup();
                    let relevant = adjacent(u, v) || !neighbours[u].is_disjoint(&neighbours[v]);
                    for given in candidates {
                        let r = partial(u, v, &given).abs();
                        if r < SEPARATED {
                            standing[u][v] = false;
                            standing[v][u] = false;
                            break;
                        }
                        if relevant {
                            margin = margin.min(r);
                            if margin < cutoff {
                                return margin;
                            }
                        }
                    }
                }
            }
            let widest = (0..p)
                .map(|v| standing[v].iter().filter(|&&s| s).count().saturating_sub(1))
                .max()
                .unwrap_or(0);
            if widest < depth + 1 {
                break;
            }
        }
        margin
    }

    /// Mean outcome under the intervention that sets the treatment to `value`, by simulation:
    /// the structural equations run with the treatment's own equation replaced by the
    /// constant, on `rows` rows of the same noise draws whatever `value` is (common random
    /// numbers). With `exponentiate` the outcome is exponentiated first, the mean of a
    /// log-linear study's outcome.
    pub fn intervention_mean(&self, value: f64, rows: usize, seed: u64, exponentiate: bool) -> f64 {
        let p = self.variables();
        let mut stream = Stream::new(seed);
        let mut row = vec![0.0; p];
        let mut total = 0.0;
        for _ in 0..rows {
            for v in 0..p {
                let draw = self.noise_sd[v] * stream.normal();
                row[v] = if v == self.treatment {
                    value
                } else {
                    self.parents[v].iter().map(|&(parent, w)| w * row[parent]).sum::<f64>() + draw
                };
            }
            let y = row[self.outcome];
            total += if exponentiate { y.exp() } else { y };
        }
        total / rows as f64
    }

    /// The truth of a linear study by intervention simulation: the change in the outcome's mean
    /// when the treatment is set to one instead of zero.
    pub fn simulated_total_effect(&self, rows: usize, seed: u64) -> f64 {
        self.intervention_mean(1.0, rows, seed, false)
            - self.intervention_mean(0.0, rows, seed, false)
    }

    /// The truth of a log-linear study by intervention simulation: the ratio of the outcome's
    /// mean when the logarithm of the treatment is set to `log_shift` to its mean when it is set
    /// to zero (the treatment multiplied by `exp(log_shift)`).
    pub fn simulated_mean_ratio(&self, log_shift: f64, rows: usize, seed: u64) -> f64 {
        self.intervention_mean(log_shift, rows, seed, true)
            / self.intervention_mean(0.0, rows, seed, true)
    }

    /// Background knowledge that is true of this model: every edge incident to the treatment
    /// as `(cause, effect)`.
    pub fn required_links(&self) -> Vec<(usize, usize)> {
        self.edges()
            .into_iter()
            .filter(|&(a, b)| a == self.treatment || b == self.treatment)
            .collect()
    }

    /// Background knowledge that is true of this model: no nuisance node causes a core node.
    pub fn forbidden_links(&self) -> Vec<(usize, usize)> {
        let nuisance = |v: usize| self.names[v].starts_with('n');
        (0..self.variables())
            .filter(|&n| nuisance(n))
            .flat_map(|n| (0..self.variables()).filter(|&c| !nuisance(c)).map(move |c| (n, c)))
            .collect()
    }

    /// Background knowledge that is true of this model: tiers by causal depth, so a later tier
    /// never causes an earlier one.
    pub fn tiers(&self) -> Vec<Vec<usize>> {
        let mut depth = vec![0usize; self.variables()];
        for v in 0..self.variables() {
            depth[v] = self.parents[v].iter().map(|&(a, _)| depth[a] + 1).max().unwrap_or(0);
        }
        let top = depth.iter().copied().max().unwrap_or(0);
        (0..=top).map(|d| (0..self.variables()).filter(|&v| depth[v] == d).collect()).collect()
    }

    /// `rows` observations of the linear-scale columns, in node order.
    pub fn simulate(&self, rows: usize, seed: u64) -> Vec<Vec<f64>> {
        self.simulate_with(rows, seed, Stream::normal)
    }

    /// As [`Self::simulate`] with right-skewed unit-variance noise: the partial-correlation test
    /// is then misspecified for the conditional independences of colliders.
    pub fn simulate_skewed(&self, rows: usize, seed: u64) -> Vec<Vec<f64>> {
        self.simulate_with(rows, seed, Stream::skewed)
    }

    fn simulate_with(
        &self,
        rows: usize,
        seed: u64,
        mut noise: impl FnMut(&mut Stream) -> f64,
    ) -> Vec<Vec<f64>> {
        let p = self.variables();
        let mut stream = Stream::new(seed);
        let mut columns = vec![Vec::with_capacity(rows); p];
        let mut row = vec![0.0; p];
        for _ in 0..rows {
            for v in 0..p {
                let mean: f64 = self.parents[v].iter().map(|&(parent, w)| w * row[parent]).sum();
                row[v] = mean + self.noise_sd[v] * noise(&mut stream);
            }
            for (column, value) in columns.iter_mut().zip(&row) {
                column.push(*value);
            }
        }
        columns
    }

    /// `rows` observations of the log-linear columns: the exponential of [`Self::simulate`].
    pub fn simulate_log_linear(&self, rows: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut columns = self.simulate(rows, seed);
        for column in &mut columns {
            for value in column.iter_mut() {
                *value = value.exp();
            }
        }
        columns
    }
}

/// Every subset of `pool` with `size` members, in lexicographic order.
fn combinations(pool: &[usize], size: usize) -> Vec<Vec<usize>> {
    if size == 0 {
        return vec![Vec::new()];
    }
    let mut out = Vec::new();
    for (k, &first) in pool.iter().enumerate() {
        for mut rest in combinations(&pool[k + 1..], size - 1) {
            rest.insert(0, first);
            out.push(rest);
        }
    }
    out
}

/// Inverse by Gauss-Jordan elimination with partial pivoting.
pub fn invert(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = matrix.len();
    let mut a: Vec<Vec<f64>> = matrix.to_vec();
    let mut inverse: Vec<Vec<f64>> =
        (0..n).map(|i| (0..n).map(|j| f64::from(u8::from(i == j))).collect()).collect();
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&p, &q| a[p][col].abs().total_cmp(&a[q][col].abs()))
            .expect("a non-empty range");
        a.swap(col, pivot);
        inverse.swap(col, pivot);
        let d = a[col][col];
        for j in 0..n {
            a[col][j] /= d;
            inverse[col][j] /= d;
        }
        for row in 0..n {
            if row != col {
                let f = a[row][col];
                for j in 0..n {
                    let (sa, si) = (f * a[col][j], f * inverse[col][j]);
                    a[row][j] -= sa;
                    inverse[row][j] -= si;
                }
            }
        }
    }
    inverse
}
