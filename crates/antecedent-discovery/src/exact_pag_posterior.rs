//! Exact posterior over PAG classes of Gaussian maximal ancestral graphs, for very small
//! graphs.
//!
//! Every maximal ancestral graph (MAG) on `n ≤ `[`PAG_ENUM_MAX_NODES`] labelled nodes is
//! enumerated: each pair of nodes is non-adjacent, `i -> j`, `j -> i` or `i <-> j`, and a
//! candidate is kept when it is ancestral (no directed cycle, no bidirected edge between a node
//! and its ancestor) and maximal (every non-adjacent pair is m-separated by some set). Each MAG
//! is fitted by residual iterative conditional fitting, the Gaussian maximum likelihood of the
//! model `x = Bx + e`, `Cov(e) = Omega` with `Omega` zero off the bidirected edges, and scored
//! by BIC. MAGs with the same set of m-separation statements are Markov equivalent; their BIC
//! weights are summed and the class is reported as its PAG: an endpoint keeps a tail or an
//! arrowhead only when every MAG of the class agrees, and is a circle otherwise.
//!
//! The weights are `exp(BIC)` under a uniform prior over MAGs, the BIC approximation to the
//! marginal likelihood, so the posterior inherits that approximation. Selection variables are
//! not modelled (no undirected edges). Above four variables the candidate space is no longer
//! enumerated; a sampler over MAGs is the next attempt.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;

use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_prob::kish_ess;

use crate::adjacency_mask::{AdjacencyMask, set_edge};
use crate::error::DiscoveryError;
use crate::graph_posterior::{
    GraphPosterior, GraphPosteriorAtomKind, SamplingSummary, accumulate_marginals,
    analytic_graph_diagnostics, normalize_log_weights,
};
use crate::pc::collect_float_columns;

/// Most variables whose MAGs are enumerated.
pub const PAG_ENUM_MAX_NODES: usize = 4;

/// Sweeps of residual iterative conditional fitting before a fit is accepted as it stands.
const RICF_SWEEPS: usize = 500;
/// Relative log-likelihood change at which a fit has converged.
const RICF_TOLERANCE: f64 = 1e-10;

/// Edge between an unordered pair `(i, j)`, `i < j`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Link {
    None,
    /// `i -> j`.
    Forward,
    /// `j -> i`.
    Backward,
    /// `i <-> j`.
    Both,
}

const LINKS: [Link; 4] = [Link::None, Link::Forward, Link::Backward, Link::Both];

/// A candidate mixed graph on `p` nodes.
#[derive(Clone, Debug)]
struct Mixed {
    p: usize,
    /// `links[pair_index(i, j)]` for `i < j`.
    links: Vec<Link>,
}

fn pair_index(p: usize, i: usize, j: usize) -> usize {
    debug_assert!(i < j);
    // Pairs in lexicographic order.
    i * p - i * (i + 1) / 2 + (j - i - 1)
}

impl Mixed {
    fn link(&self, a: usize, b: usize) -> Link {
        if a < b {
            self.links[pair_index(self.p, a, b)]
        } else {
            self.links[pair_index(self.p, b, a)]
        }
    }

    fn adjacent(&self, a: usize, b: usize) -> bool {
        a != b && self.link(a, b) != Link::None
    }

    /// `a -> b`.
    fn directed(&self, a: usize, b: usize) -> bool {
        if a < b { self.link(a, b) == Link::Forward } else { self.link(b, a) == Link::Backward }
    }

    fn bidirected(&self, a: usize, b: usize) -> bool {
        a != b && self.link(a, b) == Link::Both
    }

    /// Arrowhead at `b` on the edge between `a` and `b`.
    fn arrow_at(&self, a: usize, b: usize) -> bool {
        self.directed(a, b) || self.bidirected(a, b)
    }

    /// `reach[a][b]`: a directed path from `a` to `b`.
    fn reach(&self) -> Vec<bool> {
        let p = self.p;
        let mut reach = vec![false; p * p];
        for a in 0..p {
            for b in 0..p {
                reach[a * p + b] = a != b && self.directed(a, b);
            }
        }
        for k in 0..p {
            for a in 0..p {
                for b in 0..p {
                    if reach[a * p + k] && reach[k * p + b] {
                        reach[a * p + b] = true;
                    }
                }
            }
        }
        reach
    }

    fn ancestral(&self) -> bool {
        let p = self.p;
        let reach = self.reach();
        (0..p).all(|a| !reach[a * p + a])
            && (0..p).all(|a| {
                (a + 1..p)
                    .all(|b| !self.bidirected(a, b) || (!reach[a * p + b] && !reach[b * p + a]))
            })
    }

    /// Whether `x` and `y` are m-separated given the set `z` (a bitmask of nodes).
    fn m_separated(&self, reach: &[bool], x: usize, y: usize, z: u32) -> bool {
        let mut path = vec![x];
        !self.connects(reach, y, z, &mut path)
    }

    /// Depth-first search for an m-connecting path from the end of `path` to `y`.
    fn connects(&self, reach: &[bool], y: usize, z: u32, path: &mut Vec<usize>) -> bool {
        let p = self.p;
        let Some(&last) = path.last() else { return false };
        for next in 0..p {
            if next == last || path.contains(&next) || !self.adjacent(last, next) {
                continue;
            }
            // `last` becomes an inner node when the path continues through it.
            if path.len() >= 2 {
                let prev = path[path.len() - 2];
                let collider = self.arrow_at(prev, last) && self.arrow_at(next, last);
                let in_z = (z >> last) & 1 == 1;
                let open = if collider {
                    in_z || (0..p).any(|d| (z >> d) & 1 == 1 && reach[last * p + d])
                } else {
                    !in_z
                };
                if !open {
                    continue;
                }
            }
            if next == y {
                return true;
            }
            path.push(next);
            let found = self.connects(reach, y, z, path);
            path.pop();
            if found {
                return true;
            }
        }
        false
    }

    /// Every non-adjacent pair is m-separated by some set.
    fn maximal(&self, reach: &[bool]) -> bool {
        let p = self.p;
        for a in 0..p {
            for b in a + 1..p {
                if self.adjacent(a, b) {
                    continue;
                }
                let others: Vec<usize> = (0..p).filter(|&c| c != a && c != b).collect();
                let separated = (0..1u32 << others.len()).any(|subset| {
                    let z = others
                        .iter()
                        .enumerate()
                        .filter(|(bit, _)| (subset >> bit) & 1 == 1)
                        .fold(0u32, |acc, (_, &node)| acc | (1 << node));
                    self.m_separated(reach, a, b, z)
                });
                if !separated {
                    return false;
                }
            }
        }
        true
    }

    /// The set of m-separation statements, as a 128-bit commutative hash of the statements.
    ///
    /// Two graphs have the same signature exactly when they have the same statements, up to a
    /// hash collision of probability about 2^-128 per pair of graphs.
    fn signature(&self, reach: &[bool]) -> u128 {
        let p = self.p;
        let (mut high, mut low) = (0u64, 0u64);
        for a in 0..p {
            for b in a + 1..p {
                for z in 0..1u32 << p {
                    if (z >> a) & 1 == 1 || (z >> b) & 1 == 1 {
                        continue;
                    }
                    if self.m_separated(reach, a, b, z) {
                        let id = ((pair_index(p, a, b) as u64) << 32) | u64::from(z);
                        high ^= mix(id);
                        low ^= mix(id ^ 0xA076_1D64_78BD_642F);
                    }
                }
            }
        }
        (u128::from(high) << 64) | u128::from(low)
    }

    /// The links as one code per pair, a key for caches.
    fn code(&self) -> Vec<u8> {
        self.links.iter().map(|&link| link as u8).collect()
    }
}

/// The `SplitMix64` finalizer, a bijective avalanche of one word.
const fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Enumerate the maximal ancestral graphs on `p` nodes.
fn enumerate_mags(p: usize) -> Vec<Mixed> {
    let pairs = p * (p - 1) / 2;
    let mut out = Vec::new();
    for code in 0..4usize.pow(u32::try_from(pairs).unwrap_or(0)) {
        let links: Vec<Link> = (0..pairs).map(|pair| LINKS[(code >> (2 * pair)) & 3]).collect();
        let graph = Mixed { p, links };
        if !graph.ancestral() {
            continue;
        }
        let reach = graph.reach();
        if graph.maximal(&reach) {
            out.push(graph);
        }
    }
    out
}

/// Cholesky factor (lower, row-major) of a symmetric positive definite matrix.
#[allow(
    clippy::neg_cmp_op_on_partial_ord,
    reason = "negated comparison is deliberate: a NaN pivot must take the rejecting branch"
)]
fn cholesky(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if !(sum > 1e-300) {
                    return None;
                }
                l[i * n + i] = sum.sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }
    Some(l)
}

/// Inverse and log-determinant of a symmetric positive definite matrix.
fn invert_spd(a: &[f64], n: usize) -> Option<(Vec<f64>, f64)> {
    let l = cholesky(a, n)?;
    let log_det = 2.0 * (0..n).map(|i| l[i * n + i].ln()).sum::<f64>();
    let mut inverse = vec![0.0; n * n];
    for col in 0..n {
        // Solve L y = e_col, then L' x = y.
        let mut y = vec![0.0; n];
        for i in 0..n {
            let mut sum = f64::from(u8::from(i == col));
            for k in 0..i {
                sum -= l[i * n + k] * y[k];
            }
            y[i] = sum / l[i * n + i];
        }
        for i in (0..n).rev() {
            let mut sum = y[i];
            for k in i + 1..n {
                sum -= l[k * n + i] * inverse[k * n + col];
            }
            inverse[i * n + col] = sum / l[i * n + i];
        }
    }
    Some((inverse, log_det))
}

/// Solve `a x = b` (`k × k`, row-major) by elimination with partial pivoting.
fn solve_linear(mut a: Vec<f64>, mut b: Vec<f64>, k: usize) -> Option<Vec<f64>> {
    for col in 0..k {
        let pivot =
            (col..k).max_by(|&i, &j| a[i * k + col].abs().total_cmp(&a[j * k + col].abs()))?;
        if a[pivot * k + col].abs() < 1e-13 {
            return None;
        }
        if pivot != col {
            for c in 0..k {
                a.swap(col * k + c, pivot * k + c);
            }
            b.swap(col, pivot);
        }
        for row in col + 1..k {
            let factor = a[row * k + col] / a[col * k + col];
            for c in col..k {
                a[row * k + c] -= factor * a[col * k + c];
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = vec![0.0; k];
    for row in (0..k).rev() {
        let tail: f64 = (row + 1..k).map(|c| a[row * k + c] * x[c]).sum();
        x[row] = (b[row] - tail) / a[row * k + row];
    }
    Some(x)
}

/// `(I - B) S (I - B)'` for row-major `p × p` matrices.
fn residual_covariance(b: &[f64], s: &[f64], p: usize) -> Vec<f64> {
    let mut m = vec![0.0; p * p];
    for i in 0..p {
        for j in 0..p {
            m[i * p + j] = f64::from(u8::from(i == j)) - b[i * p + j];
        }
    }
    let mut ms = vec![0.0; p * p];
    for i in 0..p {
        for j in 0..p {
            ms[i * p + j] = (0..p).map(|k| m[i * p + k] * s[k * p + j]).sum();
        }
    }
    let mut r = vec![0.0; p * p];
    for i in 0..p {
        for j in 0..p {
            r[i * p + j] = (0..p).map(|k| ms[i * p + k] * m[j * p + k]).sum();
        }
    }
    r
}

/// BIC of the Gaussian MAG model fitted to the covariance `s` of `n` rows, or `None` when the
/// fit is degenerate.
///
/// Residual iterative conditional fitting alternates the exact generalised-least-squares update
/// of the free coefficients `B` given `Omega` with a sweep of iterative conditional fitting of
/// `Omega` given `B`: node `i`'s residual is regressed on the transformed residuals
/// `Z = Lambda e_{-i}` of its spouses, `Lambda = (Omega_{-i,-i})^{-1}`, which yields the free
/// covariances `omega_{i,sp(i)}` and the variance `omega_ii`.
#[allow(
    clippy::needless_range_loop,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::neg_cmp_op_on_partial_ord,
    reason = "small dense matrices indexed by node pairs; one fitting loop; a NaN variance must reject"
)]
fn mag_bic(graph: &Mixed, s: &[f64], n: usize) -> Option<f64> {
    let p = graph.p;
    let free_b: Vec<(usize, usize)> = (0..p)
        .flat_map(|i| (0..p).filter(move |&k| k != i).map(move |k| (i, k)))
        .filter(|&(i, k)| graph.directed(k, i))
        .collect();
    let spouses: Vec<Vec<usize>> =
        (0..p).map(|i| (0..p).filter(|&j| graph.bidirected(i, j)).collect()).collect();
    let mut b = vec![0.0; p * p];
    let mut omega = vec![0.0; p * p];
    for i in 0..p {
        omega[i * p + i] = s[i * p + i];
        if !(omega[i * p + i] > 0.0) {
            return None;
        }
    }
    let mut previous = f64::NEG_INFINITY;
    let mut log_likelihood = f64::NEG_INFINITY;
    for _ in 0..RICF_SWEEPS {
        let (lambda, _) = invert_spd(&omega, p)?;
        let m = free_b.len();
        if m > 0 {
            let mut system = vec![0.0; m * m];
            let mut rhs = vec![0.0; m];
            for (row, &(i, k)) in free_b.iter().enumerate() {
                rhs[row] = (0..p).map(|j| lambda[i * p + j] * s[k * p + j]).sum();
                for (col, &(j, l)) in free_b.iter().enumerate() {
                    system[row * m + col] = lambda[i * p + j] * s[k * p + l];
                }
            }
            let solved = solve_linear(system, rhs, m)?;
            for (&(i, k), value) in free_b.iter().zip(solved) {
                b[i * p + k] = value;
            }
        }
        let r = residual_covariance(&b, s, p);
        for i in 0..p {
            let spouse = &spouses[i];
            if spouse.is_empty() {
                omega[i * p + i] = r[i * p + i];
                continue;
            }
            let others: Vec<usize> = (0..p).filter(|&c| c != i).collect();
            let sub: Vec<f64> = others
                .iter()
                .flat_map(|&a| others.iter().map(move |&c| (a, c)))
                .map(|(a, c)| omega[a * p + c])
                .collect();
            let q = others.len();
            let (inner, _) = invert_spd(&sub, q)?;
            let position = |node: usize| others.iter().position(|&c| c == node).unwrap_or(0);
            // c_j = E[e_i Z_j] and G_jl = E[Z_j Z_l] with Z = inner e_{-i}.
            let mut cross = vec![0.0; spouse.len()];
            let mut gram = vec![0.0; spouse.len() * spouse.len()];
            for (a, &j) in spouse.iter().enumerate() {
                let row = position(j);
                cross[a] = (0..q).map(|k| inner[row * q + k] * r[i * p + others[k]]).sum();
                for (c, &l) in spouse.iter().enumerate() {
                    let col = position(l);
                    let mut value = 0.0;
                    for k in 0..q {
                        for m2 in 0..q {
                            value += inner[row * q + k]
                                * r[others[k] * p + others[m2]]
                                * inner[col * q + m2];
                        }
                    }
                    gram[a * spouse.len() + c] = value;
                }
            }
            let theta = solve_linear(gram, cross.clone(), spouse.len())?;
            let variance = r[i * p + i] - cross.iter().zip(&theta).map(|(c, t)| c * t).sum::<f64>();
            let mut shift = 0.0;
            for (a, &j) in spouse.iter().enumerate() {
                for (c, &l) in spouse.iter().enumerate() {
                    shift += theta[a] * inner[position(j) * q + position(l)] * theta[c];
                }
            }
            for (a, &j) in spouse.iter().enumerate() {
                omega[i * p + j] = theta[a];
                omega[j * p + i] = theta[a];
            }
            omega[i * p + i] = variance + shift;
        }
        let (inverse, log_det) = invert_spd(&omega, p)?;
        let r = residual_covariance(&b, s, p);
        let trace: f64 =
            (0..p).map(|i| (0..p).map(|j| inverse[i * p + j] * r[j * p + i]).sum::<f64>()).sum();
        log_likelihood =
            -0.5 * n as f64 * (p as f64 * (2.0 * std::f64::consts::PI).ln() + log_det + trace);
        if !log_likelihood.is_finite() {
            return None;
        }
        if (log_likelihood - previous).abs() <= RICF_TOLERANCE * (1.0 + log_likelihood.abs()) {
            break;
        }
        previous = log_likelihood;
    }
    let bidirected = (0..p).map(|i| spouses[i].len()).sum::<usize>() / 2;
    let parameters = free_b.len() + p + bidirected;
    Some(log_likelihood - 0.5 * parameters as f64 * (n as f64).ln())
}

/// Endpoint mark of a MAG edge at one node, or a circle when MAGs of a class disagree.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mark {
    Tail,
    Arrow,
    Circle,
}

/// The exact posterior over PAG classes of the Gaussian MAGs on `variables`.
///
/// # Errors
///
/// More than [`PAG_ENUM_MAX_NODES`] variables, fewer than two rows, a non-finite value, or no
/// MAG with a finite score.
#[allow(
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    reason = "one pipeline: moments, enumeration, fits, class merge, packing"
)]
pub fn exact_pag_posterior(
    data: &TabularData,
    variables: &[VariableId],
    ctx: &ExecutionContext,
) -> Result<GraphPosterior, DiscoveryError> {
    let p = variables.len();
    if p < 2 {
        return Err(DiscoveryError::unsupported("a PAG posterior needs at least two variables"));
    }
    if p > PAG_ENUM_MAX_NODES {
        return Err(DiscoveryError::unsupported(
            "the exact PAG posterior enumerates the maximal ancestral graphs of at most four \
             variables; use the PAG sampler for up to eight",
        ));
    }
    let (s, n) = covariance_of(data, variables)?;
    let mags = enumerate_mags(p);
    let mut kept: Vec<(Mixed, u128, f64)> = Vec::new();
    let mut rejected = 0u64;
    for graph in mags {
        if ctx.cancellation.is_cancelled() {
            return Err(DiscoveryError::Cancelled);
        }
        match mag_bic(&graph, &s, n) {
            Some(score) => {
                let signature = graph.signature(&graph.reach());
                kept.push((graph, signature, score));
            }
            None => rejected += 1,
        }
    }
    if kept.is_empty() {
        return Err(DiscoveryError::unsupported("no maximal ancestral graph has a finite BIC"));
    }
    let log_w: Vec<f64> = kept.iter().map(|(_, _, score)| *score).collect();
    let mag_weights = normalize_log_weights(&log_w)?;
    // Merge Markov-equivalent MAGs into classes, in first-seen order.
    let mut slot: HashMap<u128, usize> = HashMap::new();
    let mut members: Vec<Vec<usize>> = Vec::new();
    let mut mass: Vec<f64> = Vec::new();
    for (index, (_, signature, _)) in kept.iter().enumerate() {
        let at = *slot.entry(*signature).or_insert_with(|| {
            members.push(Vec::new());
            mass.push(0.0);
            members.len() - 1
        });
        members[at].push(index);
        mass[at] += mag_weights[index];
    }
    let mut atoms = Vec::with_capacity(members.len());
    let mut circles = Vec::with_capacity(members.len());
    for class in &members {
        let graphs: Vec<&Mixed> = class.iter().map(|&m| &kept[m].0).collect();
        let (adjacency, marks) = class_pag(p, &graphs)?;
        atoms.push(adjacency);
        circles.push(marks);
    }
    let note = format!(
        "exact_pag_posterior mags={} classes={} unscored={rejected} weights=exp(gaussian_mag_bic) \
         uniform_prior_over_mags",
        kept.len(),
        atoms.len()
    );
    let total: f64 = mass.iter().sum();
    let weights: Vec<f64> = mass.iter().map(|m| m / total).collect();
    let diagnostics = analytic_graph_diagnostics(atoms.len(), kish_ess(&weights));
    finish_posterior(p, weights, atoms, circles, diagnostics, note, rejected, "exact_pag_posterior")
}

/// The PAG of a Markov equivalence class: adjacency and circle-endpoint masks.
///
/// An endpoint keeps an arrowhead or a tail only where every graph of the class agrees and is
/// a circle otherwise.
fn class_pag(p: usize, class: &[&Mixed]) -> Result<(AdjacencyMask, AdjacencyMask), DiscoveryError> {
    let first = class[0];
    let mut adjacency = AdjacencyMask::EMPTY;
    let mut marks = AdjacencyMask::EMPTY;
    for i in 0..p {
        for j in i + 1..p {
            if !first.adjacent(i, j) {
                continue;
            }
            let endpoint = |at: usize, from: usize| {
                if class.iter().all(|m| m.arrow_at(from, at)) {
                    Mark::Arrow
                } else if class.iter().all(|m| !m.arrow_at(from, at)) {
                    Mark::Tail
                } else {
                    Mark::Circle
                }
            };
            let (at_i, at_j) = (endpoint(i, j), endpoint(j, i));
            match (at_i, at_j) {
                (Mark::Tail, Mark::Arrow) => adjacency = set_edge(adjacency, p, i, j, true),
                (Mark::Arrow, Mark::Tail) => adjacency = set_edge(adjacency, p, j, i, true),
                (Mark::Arrow, Mark::Arrow) => {
                    adjacency = set_edge(adjacency, p, i, j, true);
                    adjacency = set_edge(adjacency, p, j, i, true);
                }
                (Mark::Circle, Mark::Arrow) => {
                    adjacency = set_edge(adjacency, p, i, j, true);
                    marks = set_edge(marks, p, i, j, true);
                }
                (Mark::Arrow, Mark::Circle) => {
                    adjacency = set_edge(adjacency, p, j, i, true);
                    marks = set_edge(marks, p, j, i, true);
                }
                (Mark::Circle, Mark::Circle) => {
                    adjacency = set_edge(adjacency, p, i, j, true);
                    adjacency = set_edge(adjacency, p, j, i, true);
                    marks = set_edge(marks, p, i, j, true);
                    marks = set_edge(marks, p, j, i, true);
                }
                _ => {
                    return Err(DiscoveryError::unsupported(
                        "a class of maximal ancestral graphs produced a tail-circle edge, which \
                         the PAG packing cannot represent",
                    ));
                }
            }
        }
    }
    Ok((adjacency, marks))
}

#[allow(
    clippy::too_many_arguments,
    reason = "the packed classes, their weights and the provenance of the run"
)]
fn finish_posterior(
    p: usize,
    weights: Vec<f64>,
    atoms: Vec<AdjacencyMask>,
    circles: Vec<AdjacencyMask>,
    mut diagnostics: antecedent_prob::InferenceDiagnostics,
    note: String,
    rejected: u64,
    algorithm: &str,
) -> Result<GraphPosterior, DiscoveryError> {
    let ess = kish_ess(&weights);
    let (edge, orientation) = accumulate_marginals(p, &weights, &atoms);
    diagnostics.notes.push(std::sync::Arc::from(note));
    Ok(GraphPosterior::from_atoms(
        p,
        weights,
        atoms,
        edge,
        orientation,
        SamplingSummary { ess, diagnostics, rejected_invalid: rejected },
    )?
    .with_atom_kind(GraphPosteriorAtomKind::Pag)
    .with_mark_atoms(circles)?
    .with_algorithm(algorithm))
}

/// Most variables the MAG sampler searches.
pub const PAG_MCMC_MAX_NODES: usize = 8;
/// Largest equivalence class completed by exploration before its marks are flagged as possibly
/// over-claimed.
const CLASS_COMPLETION_CAP: usize = 4000;

/// Every maximal ancestral graph reachable from `seed` by single-pair moves through graphs of
/// signature `signature`; `complete` is false when the exploration hit the cap.
fn complete_class(seed: &Mixed, signature: u128) -> (Vec<Mixed>, bool) {
    let mut seen: std::collections::HashSet<Vec<u8>> = std::collections::HashSet::new();
    seen.insert(seed.code());
    let mut members = vec![seed.clone()];
    let mut frontier = 0;
    let pairs = seed.links.len();
    while frontier < members.len() {
        if members.len() >= CLASS_COMPLETION_CAP {
            return (members, false);
        }
        let current = members[frontier].clone();
        frontier += 1;
        for pair in 0..pairs {
            for &link in &LINKS {
                if link == current.links[pair] {
                    continue;
                }
                let mut candidate = current.clone();
                candidate.links[pair] = link;
                if seen.contains(&candidate.code()) || !candidate.ancestral() {
                    continue;
                }
                let reach = candidate.reach();
                if !candidate.maximal(&reach) || candidate.signature(&reach) != signature {
                    continue;
                }
                seen.insert(candidate.code());
                members.push(candidate);
            }
        }
    }
    (members, true)
}

/// Sampling schedule of [`pag_posterior_mcmc`].
#[derive(Clone, Copy, Debug)]
pub struct PagMcmcSchedule {
    /// Independent chains (at least two, for the agreement diagnostic).
    pub chains: usize,
    /// Warmup moves discarded per chain.
    pub warmup: usize,
    /// Post-warmup moves per chain, before thinning.
    pub draws: usize,
    /// Keep every `thin`-th post-warmup move.
    pub thin: usize,
}

/// Maps a uniform draw on `[0, 1)` to an index below `bound`.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "a uniform draw in [0, 1) scaled by a small bound floors to a non-negative index"
)]
fn uniform_index(draw: f64, bound: usize) -> usize {
    ((draw * bound as f64) as usize).min(bound.saturating_sub(1))
}

/// Posterior over PAG classes of the Gaussian maximal ancestral graphs, by Metropolis sampling.
///
/// A chain starts at the empty graph and moves by changing the link of one pair, chosen
/// uniformly, to one of the three other link types, chosen uniformly: the proposal is
/// symmetric, so a move is accepted with probability `min(1, exp(BIC' - BIC))` under the
/// uniform prior over graphs, and a proposal that is not ancestral, not maximal or not
/// scorable is rejected (the chain stays). Scores and m-separation signatures are cached per
/// graph. The pooled visit frequency of each Markov equivalence class is its posterior
/// weight; each visited class is completed to all its graphs reachable through single-pair
/// moves within the class, and its PAG is read from them.
///
/// The chains agree when the total-variation distance between each chain's class frequencies
/// and the pooled ones is at most 0.15; the result's `converged` flag records that, and no
/// finer convergence statistic is computed. The sampler only reaches the graphs connected to
/// the empty graph by single-pair moves through maximal ancestral graphs.
///
/// # Errors
///
/// Fewer than two or more than [`PAG_MCMC_MAX_NODES`] variables, fewer than two chains, an
/// empty schedule, missing or non-finite values, or no scorable graph.
#[allow(
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    reason = "one pipeline: moments, chains, class merge and completion, packing"
)]
pub fn pag_posterior_mcmc(
    data: &TabularData,
    variables: &[VariableId],
    schedule: PagMcmcSchedule,
    ctx: &ExecutionContext,
) -> Result<GraphPosterior, DiscoveryError> {
    let p = variables.len();
    if !(2..=PAG_MCMC_MAX_NODES).contains(&p) {
        return Err(DiscoveryError::unsupported(
            "the PAG sampler searches between two and eight variables",
        ));
    }
    if schedule.chains < 2 || schedule.draws == 0 || schedule.thin == 0 {
        return Err(DiscoveryError::unsupported(
            "the PAG sampler needs at least two chains, one draw and a positive thinning",
        ));
    }
    let (s, n) = covariance_of(data, variables)?;
    let pairs = p * (p - 1) / 2;
    let empty = Mixed { p, links: vec![Link::None; pairs] };
    let mut cache: HashMap<Vec<u8>, Option<(f64, u128)>> = HashMap::new();
    let evaluate = |graph: &Mixed, cache: &mut HashMap<Vec<u8>, Option<(f64, u128)>>| {
        let key = graph.code();
        if let Some(found) = cache.get(&key) {
            return *found;
        }
        let value = if graph.ancestral() {
            let reach = graph.reach();
            if graph.maximal(&reach) {
                mag_bic(graph, &s, n).map(|score| (score, graph.signature(&reach)))
            } else {
                None
            }
        } else {
            None
        };
        cache.insert(key, value);
        value
    };
    let (start_score, _) = evaluate(&empty, &mut cache)
        .ok_or_else(|| DiscoveryError::unsupported("the empty graph has no finite BIC"))?;
    let mut representative: HashMap<u128, Mixed> = HashMap::new();
    let mut counts: Vec<HashMap<u128, u64>> = Vec::with_capacity(schedule.chains);
    let mut rejected = 0u64;
    for chain in 0..schedule.chains {
        let mut rng =
            ctx.rng.stream_for(antecedent_core::StreamDomain::McmcStructure, chain as u64);
        let mut state = empty.clone();
        let mut score = start_score;
        let mut visits: HashMap<u128, u64> = HashMap::new();
        for step in 0..schedule.warmup + schedule.draws {
            if ctx.cancellation.is_cancelled() {
                return Err(DiscoveryError::Cancelled);
            }
            let pair = uniform_index(rng.next_f64(), pairs);
            let offset = 1 + uniform_index(rng.next_f64(), 3);
            let current = LINKS.iter().position(|&l| l == state.links[pair]).unwrap_or(0);
            let mut proposal = state.clone();
            proposal.links[pair] = LINKS[(current + offset) % 4];
            match evaluate(&proposal, &mut cache) {
                Some((candidate, _)) => {
                    if rng.next_f64().ln() < candidate - score {
                        state = proposal;
                        score = candidate;
                    }
                }
                None => rejected += 1,
            }
            if step >= schedule.warmup && (step - schedule.warmup) % schedule.thin == 0 {
                if let Some((_, signature)) = evaluate(&state, &mut cache) {
                    *visits.entry(signature).or_insert(0) += 1;
                    representative.entry(signature).or_insert_with(|| state.clone());
                }
            }
        }
        counts.push(visits);
    }
    // Pooled class frequencies, in a stable order.
    let mut signatures: Vec<u128> = representative.keys().copied().collect();
    signatures.sort_unstable();
    let totals: Vec<f64> = counts.iter().map(|c| c.values().sum::<u64>() as f64).collect();
    let pooled_total: f64 = totals.iter().sum();
    if pooled_total == 0.0 {
        return Err(DiscoveryError::unsupported("the PAG sampler recorded no draw"));
    }
    let pooled: Vec<f64> = signatures
        .iter()
        .map(|sig| {
            counts.iter().map(|c| c.get(sig).copied().unwrap_or(0) as f64).sum::<f64>()
                / pooled_total
        })
        .collect();
    let agreement = counts
        .iter()
        .zip(&totals)
        .map(|(chain, &total)| {
            0.5 * signatures
                .iter()
                .zip(&pooled)
                .map(|(sig, pooled)| {
                    (chain.get(sig).copied().unwrap_or(0) as f64 / total.max(1.0) - pooled).abs()
                })
                .sum::<f64>()
        })
        .fold(0.0, f64::max);
    let mut atoms = Vec::with_capacity(signatures.len());
    let mut circles = Vec::with_capacity(signatures.len());
    let mut incomplete = 0usize;
    for signature in &signatures {
        let (members, complete) = complete_class(&representative[signature], *signature);
        if !complete {
            incomplete += 1;
        }
        let graphs: Vec<&Mixed> = members.iter().collect();
        let (adjacency, marks) = class_pag(p, &graphs)?;
        atoms.push(adjacency);
        circles.push(marks);
    }
    let mut diagnostics = analytic_graph_diagnostics(atoms.len(), kish_ess(&pooled));
    diagnostics.converged = agreement <= 0.15;
    diagnostics.n_chains = u32::try_from(schedule.chains).ok();
    diagnostics.n_warmup = u32::try_from(schedule.warmup).ok();
    diagnostics.backend_id = std::sync::Arc::from("pag_mcmc");
    let note = format!(
        "pag_mcmc classes={} graphs_scored={} chains={} draws_per_chain={} \
         max_chain_total_variation={agreement:.4} incomplete_classes={incomplete} \
         weights=visit_frequency_of_exp(gaussian_mag_bic) uniform_prior_over_mags",
        atoms.len(),
        cache.len(),
        schedule.chains,
        schedule.draws / schedule.thin
    );
    finish_posterior(p, pooled, atoms, circles, diagnostics, note, rejected, "pag_mcmc")
}

/// Centered covariance of the columns and the row count.
fn covariance_of(
    data: &TabularData,
    variables: &[VariableId],
) -> Result<(Vec<f64>, usize), DiscoveryError> {
    let p = variables.len();
    let columns = collect_float_columns(data, variables)?;
    let n = columns[0].len();
    if n < 2 || columns.iter().any(|c| c.len() != n) {
        return Err(DiscoveryError::stats_msg("insufficient rows for a MAG fit"));
    }
    if columns.iter().any(|c| c.iter().any(|v| !v.is_finite())) {
        return Err(DiscoveryError::data_msg("the MAG fit needs complete finite columns"));
    }
    let means: Vec<f64> = columns.iter().map(|c| c.iter().sum::<f64>() / n as f64).collect();
    let mut s = vec![0.0; p * p];
    for i in 0..p {
        for j in 0..p {
            s[i * p + j] = (0..n)
                .map(|row| (columns[i][row] - means[i]) * (columns[j][row] - means[j]))
                .sum::<f64>()
                / n as f64;
        }
    }
    Ok((s, n))
}

#[cfg(test)]
mod tests {
    use antecedent_core::{CausalRng, ExecutionContext};
    use antecedent_data::{TableView, TabularData};

    use super::*;
    use crate::adjacency_mask::has_edge;

    fn normal(rng: &mut CausalRng) -> f64 {
        let u1 = rng.next_f64().max(1e-12);
        let u2 = rng.next_f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    fn table(columns: &[(&str, Vec<f64>)]) -> (TabularData, Vec<VariableId>) {
        let data =
            TabularData::from_f64_columns(columns.iter().map(|(n, v)| (*n, v.as_slice()))).unwrap();
        let ids = data.schema().variables().iter().map(|v| v.id).collect();
        (data, ids)
    }

    #[test]
    fn two_node_mags_are_the_four_edge_types_and_three_are_equivalent() {
        let mags = enumerate_mags(2);
        assert_eq!(mags.len(), 4);
        let signatures: std::collections::HashSet<u128> =
            mags.iter().map(|m| m.signature(&m.reach())).collect();
        // The empty graph separates the pair; the three adjacent graphs share no statement.
        assert_eq!(signatures.len(), 2);
    }

    #[test]
    fn every_enumerated_graph_is_ancestral_and_maximal() {
        for p in 2..=3 {
            for graph in enumerate_mags(p) {
                assert!(graph.ancestral());
                assert!(graph.maximal(&graph.reach()));
            }
        }
        // Three nodes have the 25 DAGs among their maximal ancestral graphs.
        let dags = enumerate_mags(3)
            .iter()
            .filter(|g| (0..3).all(|a| (a + 1..3).all(|b| !g.bidirected(a, b))))
            .count();
        assert_eq!(dags, 25);
    }

    #[test]
    fn the_empty_graph_fits_to_the_marginal_variances() {
        let mut rng = CausalRng::from_seed(3);
        let x: Vec<f64> = (0..500).map(|_| normal(&mut rng)).collect();
        let y: Vec<f64> = (0..500).map(|_| 2.0 * normal(&mut rng)).collect();
        let n = 500usize;
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let var = |v: &[f64]| v.iter().map(|a| (a - mean(v)).powi(2)).sum::<f64>() / v.len() as f64;
        let cov =
            x.iter().zip(&y).map(|(a, b)| (a - mean(&x)) * (b - mean(&y))).sum::<f64>() / n as f64;
        let s = [var(&x), cov, cov, var(&y)];
        let empty = Mixed { p: 2, links: vec![Link::None] };
        let expected = -0.5
            * n as f64
            * (2.0 * (2.0 * std::f64::consts::PI).ln() + s[0].ln() + s[3].ln() + 2.0)
            - 0.5 * 2.0 * (n as f64).ln();
        assert!((mag_bic(&empty, &s, n).unwrap() - expected).abs() < 1e-6);
        // The three adjacent graphs are saturated and tie.
        let scores: Vec<f64> = [Link::Forward, Link::Backward, Link::Both]
            .iter()
            .map(|&link| mag_bic(&Mixed { p: 2, links: vec![link] }, &s, n).unwrap())
            .collect();
        assert!((scores[0] - scores[1]).abs() < 1e-6);
        assert!((scores[0] - scores[2]).abs() < 1e-5, "{scores:?}");
    }

    #[test]
    fn a_chain_is_one_circle_class_and_a_collider_is_directed() {
        let mut rng = CausalRng::from_seed(21);
        let n = 2500;
        let a: Vec<f64> = (0..n).map(|_| normal(&mut rng)).collect();
        let b: Vec<f64> = a.iter().map(|a| 0.9 * a + 0.5 * normal(&mut rng)).collect();
        let c: Vec<f64> = b.iter().map(|b| 0.9 * b + 0.5 * normal(&mut rng)).collect();
        let (data, ids) = table(&[("a", a.clone()), ("b", b.clone()), ("c", c)]);
        let ctx = ExecutionContext::for_tests(1);
        let chain = exact_pag_posterior(&data, &ids, &ctx).unwrap();
        assert_eq!(chain.atom_kind, GraphPosteriorAtomKind::Pag);
        let top = (0..chain.n_graphs)
            .max_by(|&i, &j| chain.weights[i].total_cmp(&chain.weights[j]))
            .unwrap();
        assert!(chain.weights[top] > 0.8, "{:?}", chain.weights);
        let atom = chain.atoms[top];
        assert!(has_edge(atom, 3, 0, 1) && has_edge(atom, 3, 1, 0));
        assert!(has_edge(atom, 3, 1, 2) && has_edge(atom, 3, 2, 1));
        assert!(!has_edge(atom, 3, 0, 2) && !has_edge(atom, 3, 2, 0));
        let circles = chain.mark_atoms.as_ref().unwrap()[top];
        assert!(has_edge(circles, 3, 0, 1) && has_edge(circles, 3, 1, 0));

        let x: Vec<f64> = (0..n).map(|_| normal(&mut rng)).collect();
        let z: Vec<f64> = (0..n).map(|_| normal(&mut rng)).collect();
        let y: Vec<f64> = x.iter().zip(&z).map(|(x, z)| x + z + 0.5 * normal(&mut rng)).collect();
        let (data, ids) = table(&[("x", x), ("y", y), ("z", z)]);
        let collider = exact_pag_posterior(&data, &ids, &ctx).unwrap();
        let top = (0..collider.n_graphs)
            .max_by(|&i, &j| collider.weights[i].total_cmp(&collider.weights[j]))
            .unwrap();
        let atom = collider.atoms[top];
        // x o-> y <-o z: the bidirected variants of the two edges are Markov equivalent to the
        // directed ones, so their tails are circles.
        assert!(has_edge(atom, 3, 0, 1) && !has_edge(atom, 3, 1, 0));
        assert!(has_edge(atom, 3, 2, 1) && !has_edge(atom, 3, 1, 2));
        let circles = collider.mark_atoms.as_ref().unwrap()[top];
        assert!(has_edge(circles, 3, 0, 1) && has_edge(circles, 3, 2, 1));
        assert!(!has_edge(circles, 3, 1, 0) && !has_edge(circles, 3, 1, 2));
    }

    #[test]
    fn the_sampler_finds_the_exact_posterior_of_a_small_graph_and_scales_past_four_variables() {
        let mut rng = CausalRng::from_seed(5);
        let n = 1500;
        let a: Vec<f64> = (0..n).map(|_| normal(&mut rng)).collect();
        let b: Vec<f64> = a.iter().map(|a| 0.9 * a + 0.5 * normal(&mut rng)).collect();
        let c: Vec<f64> = b.iter().map(|b| 0.9 * b + 0.5 * normal(&mut rng)).collect();
        let (data, ids) = table(&[("a", a.clone()), ("b", b.clone()), ("c", c.clone())]);
        let ctx = ExecutionContext::for_tests(2);
        let schedule = PagMcmcSchedule { chains: 2, warmup: 300, draws: 2000, thin: 1 };
        let sampled = pag_posterior_mcmc(&data, &ids, schedule, &ctx).unwrap();
        let exact = exact_pag_posterior(&data, &ids, &ctx).unwrap();
        let top = |g: &GraphPosterior| {
            let at =
                (0..g.n_graphs).max_by(|&i, &j| g.weights[i].total_cmp(&g.weights[j])).unwrap();
            (g.atoms[at], g.mark_atoms.as_ref().unwrap()[at])
        };
        assert_eq!(top(&sampled), top(&exact));
        // Five variables: a chain with an independent fifth and fourth variable.
        let d: Vec<f64> = (0..n).map(|_| normal(&mut rng)).collect();
        let e: Vec<f64> = (0..n).map(|_| normal(&mut rng)).collect();
        let (wide, ids) = table(&[("a", a), ("b", b), ("c", c), ("d", d), ("e", e)]);
        let small = PagMcmcSchedule { chains: 2, warmup: 200, draws: 800, thin: 1 };
        let posterior = pag_posterior_mcmc(&wide, &ids, small, &ctx).unwrap();
        assert_eq!(posterior.atom_kind, GraphPosteriorAtomKind::Pag);
        let mass: f64 = posterior.weights.iter().sum();
        assert!((mass - 1.0).abs() < 1e-9);
        assert!(
            pag_posterior_mcmc(&wide, &ids, PagMcmcSchedule { chains: 1, ..small }, &ctx).is_err()
        );
    }

    #[test]
    fn more_than_four_variables_are_refused() {
        let cols: Vec<(&str, Vec<f64>)> = ["a", "b", "c", "d", "e"]
            .iter()
            .map(|n| (*n, (0..10).map(f64::from).collect()))
            .collect();
        let (data, ids) = table(&cols);
        assert!(exact_pag_posterior(&data, &ids, &ExecutionContext::for_tests(1)).is_err());
    }
}
