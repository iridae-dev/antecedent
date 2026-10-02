//! Self-certifying two-model witnesses for ADMG conditional transport (2.2B B1).
//!
//! A [`ConditionalWitnessRecord`] is two finite latent models over the
//! selection diagram's observed variables (binary) and one discrete latent per
//! bidirected edge. Each model is a pair of populations: the source and the
//! target share every latent law and every mechanism except those of the
//! selection targets, whose target mechanisms are listed separately (the
//! population-difference semantics of a selection node, used conservatively:
//! only mechanisms differ, never a latent law). Every parameter is an exact
//! rational strictly between 0 and 1, so every law either model induces is
//! strictly positive.
//!
//! [`verify_conditional_witness`] trusts no theorem. By exact enumeration it
//! checks that the two models are compatible with the diagram (one mechanism
//! per node over exactly its graph parents and incident latents; one target
//! mechanism per selection target), that they agree on EVERY source
//! experimental law `P(v \ z | do(z))` (every subset `Z`, every level `z`) and
//! on the target observational law `P*(v)`, and that they give different values
//! of the query `P*(y | do(x), w)` at the recorded level, where the conditioning
//! event has positive mass. Such a pair shows that no function of the
//! complete source experimental family and the target observational law
//! computes the query on every model compatible with the diagram: the
//! conditional query is not transportable (Lemma 2 of Lee, Correa and
//! Bareinboim, AAAI 2020, which is the definition read off; no completeness
//! theorem is used). A witness over binary observed variables is a witness for
//! the diagram, not for the user's variable domains.
//!
//! [`search_conditional_witness`] looks for such a pair. It is a bounded,
//! deterministic heuristic, never a proof by itself: starting from a seeded
//! strictly positive base model, it perturbs ONE parameter block at a time (one
//! latent law, or one node's mechanisms in both populations). Every supplied law
//! is linear in a single block, so a perturbation in the null space of the
//! block's Jacobian leaves every law exactly unchanged. The null space is found
//! modulo a prime and lifted by rational reconstruction; the result counts only
//! after the exact verifier accepts it. When no block yields a verified pair the
//! search returns nothing and the decision stays `not_certified`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::conditional::{
    ConditionalTransportQuery, dense_variables, invalid_derivation, validate,
};
use crate::IdentificationError;
use antecedent_core::ExecutionContext;
use antecedent_graph::{DenseNodeId, SelectionDiagram};

/// Most enumeration work (law cells times latent configurations times nodes)
/// a witness may need; larger witnesses are refused before any arithmetic.
pub const CONDITIONAL_WITNESS_MAX_WORK: u64 = 1 << 22;
/// Most levels of one latent variable in a witness.
pub const CONDITIONAL_WITNESS_MAX_LATENT_LEVELS: usize = 8;
/// Enumeration work above which the search does not start (the verifier's cap
/// is larger, so every searched witness fits it).
const SEARCH_MAX_WORK: u64 = 1 << 20;
/// Most free parameters of one perturbed block in the search.
const SEARCH_MAX_BLOCK_PARAMETERS: usize = 256;
/// Latent cardinalities the search tries, in order.
const SEARCH_LATENT_LEVELS: [usize; 3] = [3, 5, 2];
/// Base models per latent cardinality.
const SEARCH_SEEDS: u64 = 2;

/// Stable failure text of a witness that does not verify.
const NOT_VERIFIED: &str = "admg_transport.invalid_derivation";

/// One latent variable: the bidirected edge it realizes and its law.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WitnessLatentRecord {
    /// The bidirected edge's endpoints (variable ids, lower dense node first).
    pub edge: [u32; 2],
    /// `P(latent = l)` for every level `l`, exact rationals `"p/q"`.
    pub probabilities: Vec<String>,
}

/// One mechanism `P(node = 1 | observed parents, incident latents)`.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WitnessKernelRecord {
    /// The node (variable id).
    pub node: u32,
    /// Its observed parents in the causal graph (variable ids, dense order).
    pub parents: Vec<u32>,
    /// Its incident bidirected edges, in the model's latent order.
    pub latents: Vec<[u32; 2]>,
    /// `P(node = 1 | row)` per row, exact rationals `"p/q"`. The row index is the
    /// parents' bits (first parent most significant) followed by the latents'
    /// levels in mixed radix (first latent most significant).
    pub ones: Vec<String>,
}

/// One model: shared latent laws, the source mechanism of every node and the
/// target mechanism of every selection target (every other node's target
/// mechanism is its source mechanism).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WitnessModelRecord {
    /// One latent per bidirected edge, in dense edge order.
    pub latents: Vec<WitnessLatentRecord>,
    /// Source mechanisms, one per node in dense order.
    pub source: Vec<WitnessKernelRecord>,
    /// Target mechanisms, one per selection target in dense order.
    pub target: Vec<WitnessKernelRecord>,
}

/// Untrusted portable two-model witness for a conditional query.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConditionalWitnessRecord {
    /// The first model.
    pub first: WitnessModelRecord,
    /// The second model.
    pub second: WitnessModelRecord,
    /// Treatment levels, in query order.
    pub treatment_level: Vec<u8>,
    /// Conditioned levels, in query order.
    pub conditioned_level: Vec<u8>,
    /// Outcome levels, in query order.
    pub outcome_level: Vec<u8>,
    /// `P*(y | do(x), w)` at the recorded level in the first model, `"p/q"`.
    pub first_value: String,
    /// The same in the second model.
    pub second_value: String,
}

/// What a verified witness established.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConditionalWitnessCheck {
    /// Law cells compared (every source experiment's cells plus the target's).
    pub cells_compared: usize,
    /// The query value in the first model, `"p/q"`.
    pub first_value: String,
    /// The query value in the second model, `"p/q"`.
    pub second_value: String,
}

fn rejected(_why: &str) -> IdentificationError {
    debug_assert_eq!(NOT_VERIFIED, "admg_transport.invalid_derivation");
    invalid_derivation()
}

// ---------------------------------------------------------------------------
// Exact rationals (checked i128; any overflow refuses).

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Q {
    n: i128,
    d: i128,
}

const fn gcd(mut a: i128, mut b: i128) -> i128 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

impl Q {
    const ZERO: Self = Self { n: 0, d: 1 };
    const ONE: Self = Self { n: 1, d: 1 };

    fn new(n: i128, d: i128) -> Option<Self> {
        if d == 0 {
            return None;
        }
        let g = gcd(n, d);
        let (mut n, mut d) = (n / g, d / g);
        if d < 0 {
            n = n.checked_neg()?;
            d = d.checked_neg()?;
        }
        Some(Self { n, d })
    }
    fn add(self, o: Self) -> Option<Self> {
        let g = gcd(self.d, o.d);
        let l = (self.d / g).checked_mul(o.d)?;
        let a = self.n.checked_mul(l / self.d)?;
        let b = o.n.checked_mul(l / o.d)?;
        Self::new(a.checked_add(b)?, l)
    }
    fn mul(self, o: Self) -> Option<Self> {
        let g1 = gcd(self.n, o.d).max(1);
        let g2 = gcd(o.n, self.d).max(1);
        Self::new((self.n / g1).checked_mul(o.n / g2)?, (self.d / g2).checked_mul(o.d / g1)?)
    }
    fn div(self, o: Self) -> Option<Self> {
        if o.n == 0 {
            return None;
        }
        self.mul(Self::new(o.d, o.n)?)
    }
    fn parse(text: &str) -> Option<Self> {
        let (n, d) = text.split_once('/')?;
        if n.is_empty() || d.is_empty() || d.starts_with(['-', '+']) || n.starts_with('+') {
            return None;
        }
        let q = Self::new(n.parse().ok()?, d.parse().ok()?)?;
        // Canonical spelling only: a record is data, and one value has one text.
        (q.to_text() == text).then_some(q)
    }
    fn to_text(self) -> String {
        format!("{}/{}", self.n, self.d)
    }
    const fn is_strict_probability(self) -> bool {
        self.n > 0 && self.n < self.d
    }
}

// ---------------------------------------------------------------------------
// The diagram's structure, shared by the verifier and the search.

/// Dense structure of a selection diagram for witness enumeration.
struct Shape {
    n: usize,
    /// Variable id of every dense node.
    ids: Vec<u32>,
    /// Observed parents of every node (dense, ascending).
    parents: Vec<Vec<usize>>,
    /// Bidirected edges (dense, `a < b`), lexicographic.
    edges: Vec<(usize, usize)>,
    /// Incident edge indices of every node, ascending.
    incident: Vec<Vec<usize>>,
    /// Whether each node is a selection target.
    selected: Vec<bool>,
}

impl Shape {
    fn new(diagram: &SelectionDiagram) -> Result<Self, IdentificationError> {
        let graph = diagram.causal_graph();
        let variables = dense_variables(graph)?;
        let n = variables.len();
        let dense = |d: DenseNodeId| d.as_usize();
        let mut parents = Vec::with_capacity(n);
        let mut edges = Vec::new();
        for i in 0..n {
            let node = DenseNodeId::from_raw(u32::try_from(i).map_err(|_| invalid_derivation())?);
            let mut ps: Vec<usize> = graph.parents(node).iter().copied().map(dense).collect();
            ps.sort_unstable();
            parents.push(ps);
            let mut bs: Vec<usize> = graph
                .bidirected_neighbors(node)
                .iter()
                .copied()
                .map(dense)
                .filter(|b| *b > i)
                .collect();
            bs.sort_unstable();
            edges.extend(bs.into_iter().map(|b| (i, b)));
        }
        let incident = (0..n)
            .map(|i| (0..edges.len()).filter(|e| edges[*e].0 == i || edges[*e].1 == i).collect())
            .collect();
        let selected = variables.iter().map(|v| diagram.selection_targets().contains(v)).collect();
        Ok(Self {
            n,
            ids: variables.iter().map(|v| v.raw()).collect(),
            parents,
            edges,
            incident,
            selected,
        })
    }

    fn dense_of(&self, id: u32) -> Result<usize, IdentificationError> {
        self.ids.iter().position(|v| *v == id).ok_or_else(invalid_derivation)
    }

    /// Rows of node `j`'s mechanism under latent cardinalities `cards`.
    fn rows(&self, j: usize, cards: &[usize]) -> usize {
        (1usize << self.parents[j].len())
            * self.incident[j].iter().map(|e| cards[*e]).product::<usize>()
    }

    /// Row of node `j` in `world` (observed bits) and latent configuration `u`.
    fn row(&self, j: usize, world: usize, u: &[usize], cards: &[usize]) -> usize {
        let mut row = 0usize;
        for &p in &self.parents[j] {
            row = (row << 1) | ((world >> p) & 1);
        }
        for &e in &self.incident[j] {
            row = row * cards[e] + u[e];
        }
        row
    }

    /// Enumeration work of one model with these latent cardinalities.
    fn work(&self, cards: &[usize]) -> u64 {
        let cells = (1u64 << (2 * self.n)) + (1u64 << self.n);
        let configs = cards.iter().map(|c| *c as u64).fold(1u64, u64::saturating_mul);
        cells.saturating_mul(configs).saturating_mul(self.n.max(1) as u64)
    }
}

/// Advance a mixed-radix latent configuration; false after the last one.
fn next_config(u: &mut [usize], cards: &[usize]) -> bool {
    for i in (0..u.len()).rev() {
        u[i] += 1;
        if u[i] < cards[i] {
            return true;
        }
        u[i] = 0;
    }
    false
}

// ---------------------------------------------------------------------------
// Verification.

/// A parsed, shape-checked model.
struct Model {
    cards: Vec<usize>,
    latents: Vec<Vec<Q>>,
    source: Vec<Vec<Q>>,
    /// Target mechanism of every node (the source one unless selected).
    target: Vec<Vec<Q>>,
}

fn parse_model(shape: &Shape, record: &WitnessModelRecord) -> Result<Model, IdentificationError> {
    if record.latents.len() != shape.edges.len() {
        return Err(rejected("one latent per bidirected edge"));
    }
    let mut cards = Vec::with_capacity(shape.edges.len());
    let mut latents = Vec::with_capacity(shape.edges.len());
    for (latent, &(a, b)) in record.latents.iter().zip(&shape.edges) {
        if latent.edge != [shape.ids[a], shape.ids[b]] {
            return Err(rejected("a latent does not realize its bidirected edge"));
        }
        let k = latent.probabilities.len();
        if !(2..=CONDITIONAL_WITNESS_MAX_LATENT_LEVELS).contains(&k) {
            return Err(rejected("latent cardinality"));
        }
        let law = latent
            .probabilities
            .iter()
            .map(|p| Q::parse(p).filter(|q| q.is_strict_probability()))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| rejected("a latent probability is not in (0, 1)"))?;
        let total = law
            .iter()
            .try_fold(Q::ZERO, |acc, p| acc.add(*p))
            .ok_or_else(|| rejected("overflow"))?;
        if total != Q::ONE {
            return Err(rejected("a latent law does not sum to one"));
        }
        cards.push(k);
        latents.push(law);
    }
    if shape.work(&cards) > CONDITIONAL_WITNESS_MAX_WORK {
        return Err(rejected("witness exceeds the enumeration bound"));
    }
    let kernel = |j: usize, record: &WitnessKernelRecord| -> Result<Vec<Q>, IdentificationError> {
        let parents: Vec<u32> = shape.parents[j].iter().map(|p| shape.ids[*p]).collect();
        let incident: Vec<[u32; 2]> = shape.incident[j]
            .iter()
            .map(|e| [shape.ids[shape.edges[*e].0], shape.ids[shape.edges[*e].1]])
            .collect();
        if record.node != shape.ids[j] || record.parents != parents || record.latents != incident {
            return Err(rejected("a mechanism is not over its node's graph parents and latents"));
        }
        if record.ones.len() != shape.rows(j, &cards) {
            return Err(rejected("a mechanism has the wrong number of rows"));
        }
        record
            .ones
            .iter()
            .map(|p| Q::parse(p).filter(|q| q.is_strict_probability()))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| rejected("a mechanism probability is not in (0, 1)"))
    };
    if record.source.len() != shape.n {
        return Err(rejected("one source mechanism per node"));
    }
    let source =
        (0..shape.n).map(|j| kernel(j, &record.source[j])).collect::<Result<Vec<_>, _>>()?;
    let selected: Vec<usize> = (0..shape.n).filter(|j| shape.selected[*j]).collect();
    if record.target.len() != selected.len() {
        return Err(rejected("one target mechanism per selection target"));
    }
    let mut target = source.clone();
    for (&j, record) in selected.iter().zip(&record.target) {
        target[j] = kernel(j, record)?;
    }
    Ok(Model { cards, latents, source, target })
}

/// A model's parameters as integers over one common denominator per latent law
/// and per mechanism, tabulated per node, world and latent configuration, so a
/// cell is an integer sum over a fixed denominator.
struct Integral {
    configs: usize,
    /// Product of the latent laws' numerators per configuration, and of their
    /// denominators.
    latent_mass: Vec<i128>,
    latent_den: i128,
    /// Per population and node: the mechanism's denominator and the numerator
    /// of `P(v_j = world_j | row)` at `[world * configs + code]`.
    source: Vec<(i128, Vec<i128>)>,
    target: Vec<(i128, Vec<i128>)>,
}

impl Integral {
    fn new(shape: &Shape, model: &Model) -> Option<Self> {
        let common = |row: &[Q]| -> Option<(i128, Vec<i128>)> {
            let mut lcm = 1i128;
            for q in row {
                lcm = (lcm / gcd(lcm, q.d)).checked_mul(q.d)?;
            }
            Some((lcm, row.iter().map(|q| q.n.checked_mul(lcm / q.d)).collect::<Option<_>>()?))
        };
        let cards = &model.cards;
        let configs: usize = cards.iter().product();
        let laws = model.latents.iter().map(|law| common(law)).collect::<Option<Vec<_>>>()?;
        let mut latent_den = 1i128;
        for (d, _) in &laws {
            latent_den = latent_den.checked_mul(*d)?;
        }
        let mut latent_mass = Vec::with_capacity(configs);
        let mut u = vec![0usize; cards.len()];
        let mut codes = Vec::with_capacity(configs);
        for _ in 0..configs {
            let mut mass = 1i128;
            for (e, (_, law)) in laws.iter().enumerate() {
                mass = mass.checked_mul(law[u[e]])?;
            }
            latent_mass.push(mass);
            codes.push(u.clone());
            next_config(&mut u, cards);
        }
        let worlds = 1usize << shape.n;
        let table = |kernels: &[Vec<Q>]| -> Option<Vec<(i128, Vec<i128>)>> {
            (0..shape.n)
                .map(|j| {
                    let (d, ones) = common(&kernels[j])?;
                    let mut factors = vec![0i128; worlds * configs];
                    for world in 0..worlds {
                        for (code, u) in codes.iter().enumerate() {
                            let one = ones[shape.row(j, world, u, cards)];
                            factors[world * configs + code] =
                                if (world >> j) & 1 == 1 { one } else { d - one };
                        }
                    }
                    Some((d, factors))
                })
                .collect()
        };
        Some(Self {
            configs,
            latent_mass,
            latent_den,
            source: table(&model.source)?,
            target: table(&model.target)?,
        })
    }
}

/// `P(world restricted to the non-intervened nodes | do(mask = world & mask))`
/// in one population of `model`, exactly (`None` on overflow).
fn cell(shape: &Shape, model: &Integral, target: bool, mask: usize, world: usize) -> Option<Q> {
    let kernels = if target { &model.target } else { &model.source };
    let free: Vec<&(i128, Vec<i128>)> =
        (0..shape.n).filter(|j| mask & (1 << j) == 0).map(|j| &kernels[j]).collect();
    let mut den = model.latent_den;
    for (d, _) in &free {
        den = den.checked_mul(*d)?;
    }
    let base = world * model.configs;
    let mut total = 0i128;
    for (code, latent) in model.latent_mass.iter().enumerate() {
        let mut mass = *latent;
        for (_, factors) in &free {
            mass = mass.checked_mul(factors[base + code])?;
        }
        total = total.checked_add(mass)?;
    }
    Q::new(total, den)
}

/// Dense coordinates and levels of the query at the recorded level.
struct Level {
    x_mask: usize,
    x_bits: usize,
    w_mask: usize,
    w_bits: usize,
    y_mask: usize,
    y_bits: usize,
}

fn level(
    shape: &Shape,
    query: &ConditionalTransportQuery,
    record: &ConditionalWitnessRecord,
) -> Result<Level, IdentificationError> {
    let bits = |vars: &[antecedent_core::VariableId], levels: &[u8]| {
        if vars.len() != levels.len() || levels.iter().any(|l| *l > 1) {
            return Err(rejected("a level does not bind its coordinates"));
        }
        let (mut mask, mut set) = (0usize, 0usize);
        for (v, l) in vars.iter().zip(levels) {
            let j = shape.dense_of(v.raw())?;
            mask |= 1 << j;
            set |= usize::from(*l) << j;
        }
        Ok((mask, set))
    };
    let (x_mask, x_bits) = bits(&query.base.treatments, &record.treatment_level)?;
    let (w_mask, w_bits) = bits(&query.conditioned_on, &record.conditioned_level)?;
    let (y_mask, y_bits) = bits(&query.base.outcomes, &record.outcome_level)?;
    Ok(Level { x_mask, x_bits, w_mask, w_bits, y_mask, y_bits })
}

/// `P*(y | do(x), w)` at `level` in `model`; `None` on overflow or when the
/// conditioning event has no mass (impossible for a strictly positive model,
/// still refused rather than divided).
fn query_value(shape: &Shape, model: &Integral, at: &Level) -> Option<Q> {
    let (mut num, mut den) = (Q::ZERO, Q::ZERO);
    for world in 0..(1usize << shape.n) {
        if world & at.x_mask != at.x_bits || world & at.w_mask != at.w_bits {
            continue;
        }
        let p = cell(shape, model, true, at.x_mask, world)?;
        den = den.add(p)?;
        if world & at.y_mask == at.y_bits {
            num = num.add(p)?;
        }
    }
    if den.n == 0 { None } else { num.div(den) }
}

/// The two query values a shape-checked witness gives, after checking that the
/// models agree on every supplied law.
fn evaluate(
    shape: &Shape,
    query: &ConditionalTransportQuery,
    record: &ConditionalWitnessRecord,
    ctx: &ExecutionContext,
) -> Result<(Q, Q, usize), IdentificationError> {
    let overflow = || rejected("arithmetic overflow");
    let first = Integral::new(shape, &parse_model(shape, &record.first)?).ok_or_else(overflow)?;
    let second = Integral::new(shape, &parse_model(shape, &record.second)?).ok_or_else(overflow)?;
    let mut compared = 0usize;
    let full = 1usize << shape.n;
    for mask in 0..=full {
        if ctx.cancellation.is_cancelled() {
            return Err(IdentificationError::Cancelled);
        }
        // mask == full encodes the target observational law.
        let (target, intervened) = if mask == full { (true, 0) } else { (false, mask) };
        for world in 0..full {
            let a = cell(shape, &first, target, intervened, world).ok_or_else(overflow)?;
            let b = cell(shape, &second, target, intervened, world).ok_or_else(overflow)?;
            if a != b {
                return Err(rejected("the models disagree on a supplied law"));
            }
            compared += 1;
        }
    }
    let at = level(shape, query, record)?;
    let no_value = || rejected("arithmetic overflow or a zero-mass conditioning event");
    let a = query_value(shape, &first, &at).ok_or_else(no_value)?;
    let b = query_value(shape, &second, &at).ok_or_else(no_value)?;
    Ok((a, b, compared))
}

/// Verify a two-model witness by exact enumeration: both models are compatible
/// with `diagram` and strictly positive, agree on every source experimental law
/// and on the target observational law, and give different values of `query`
/// at the recorded level (equal to the recorded values). No theorem is trusted.
///
/// Not charged to a search budget: the enumeration is refused above
/// [`CONDITIONAL_WITNESS_MAX_WORK`] before any arithmetic, and cancellation is
/// polled per law.
///
/// # Errors
/// `admg_transport.invalid_derivation` when any check fails (including an
/// arithmetic overflow), an invalid query, or cancellation.
pub fn verify_conditional_witness(
    diagram: &SelectionDiagram,
    query: &ConditionalTransportQuery,
    record: &ConditionalWitnessRecord,
    ctx: &ExecutionContext,
) -> Result<ConditionalWitnessCheck, IdentificationError> {
    validate(diagram, query)?;
    let shape = Shape::new(diagram)?;
    let (a, b, cells_compared) = evaluate(&shape, query, record, ctx)?;
    if a == b {
        return Err(rejected("the models agree on the query"));
    }
    if a.to_text() != record.first_value || b.to_text() != record.second_value {
        return Err(rejected("a recorded query value is not the model's"));
    }
    Ok(ConditionalWitnessCheck {
        cells_compared,
        first_value: record.first_value.clone(),
        second_value: record.second_value.clone(),
    })
}

// ---------------------------------------------------------------------------
// Search (modular linear algebra; only the exact verifier certifies).

const P: u64 = (1 << 61) - 1;

/// `a * b mod P` by Mersenne folding (`2^61 = 1 mod P`); inputs below `P`.
#[allow(
    clippy::cast_possible_truncation,
    reason = "the low 61 bits are masked and the product is below 2^122, so the high part fits"
)]
const fn fp_mul(a: u64, b: u64) -> u64 {
    let x = a as u128 * b as u128;
    let folded = (x as u64 & P) + (x >> 61) as u64;
    let once = if folded >= P { folded - P } else { folded };
    if once >= P { once - P } else { once }
}
const fn fp_add(a: u64, b: u64) -> u64 {
    let s = a + b;
    if s >= P { s - P } else { s }
}
const fn fp_sub(a: u64, b: u64) -> u64 {
    if a >= b { a - b } else { a + P - b }
}
fn fp_pow(mut a: u64, mut e: u64) -> u64 {
    let mut r = 1;
    while e > 0 {
        if e & 1 == 1 {
            r = fp_mul(r, a);
        }
        a = fp_mul(a, a);
        e >>= 1;
    }
    r
}
fn fp_inv(a: u64) -> u64 {
    fp_pow(a, P - 2)
}
fn fp_of(q: Q) -> Option<u64> {
    let p = i128::from(P);
    let n = u64::try_from(q.n.rem_euclid(p)).ok()?;
    let d = u64::try_from(q.d.rem_euclid(p)).ok()?;
    (d != 0).then(|| fp_mul(n, fp_inv(d)))
}

/// Rational reconstruction of `a mod P` with numerator and denominator below
/// `2^30`; `None` when no such fraction exists.
fn reconstruct(a: u64) -> Option<Q> {
    let bound: i128 = 1 << 30;
    let (mut r0, mut r1) = (i128::from(P), i128::from(a));
    let (mut t0, mut t1) = (0i128, 1i128);
    while r1 >= bound {
        let q = r0 / r1;
        (r0, r1) = (r1, r0 - q * r1);
        (t0, t1) = (t1, t0 - q * t1);
    }
    if t1 == 0 || t1.abs() >= bound {
        return None;
    }
    let q = Q::new(r1, t1)?;
    (fp_of(q)? == a).then_some(q)
}

/// Deterministic generator (splitmix64).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn small(&mut self, n: u64) -> i128 {
        i128::from(1 + self.next() % n)
    }
}

/// A seeded strictly positive base model: latent weights in `1..=4`
/// normalized, mechanism probabilities in `{1/9, ..., 8/9}`.
fn base_model(shape: &Shape, levels: usize, seed: u64) -> Option<Model> {
    let mut rng = Rng(0xB1C0_2D17 ^ seed.wrapping_mul(0x1000_0000_01B3) ^ levels as u64);
    let cards = vec![levels; shape.edges.len()];
    let mut latents = Vec::with_capacity(cards.len());
    for _ in &shape.edges {
        let weights: Vec<i128> = (0..levels).map(|_| rng.small(4)).collect();
        let total: i128 = weights.iter().sum();
        latents.push(weights.iter().map(|w| Q::new(*w, total)).collect::<Option<Vec<_>>>()?);
    }
    let mut kernel = |j: usize| {
        (0..shape.rows(j, &cards)).map(|_| Q::new(rng.small(8), 9)).collect::<Option<Vec<_>>>()
    };
    let source = (0..shape.n).map(&mut kernel).collect::<Option<Vec<_>>>()?;
    let mut target = source.clone();
    for (j, row) in target.iter_mut().enumerate() {
        if shape.selected[j] {
            *row = kernel(j)?;
        }
    }
    Some(Model { cards, latents, source, target })
}

/// A block of parameters every supplied law is linear in. (Latent laws are
/// also such blocks; on the three- and four-node classes they never produced a
/// witness a mechanism block did not, so the search does not try them.)
#[derive(Clone, Copy)]
enum Block {
    /// Node `j`'s source mechanism, and its target one when selected.
    Kernel(usize),
    /// Selection target `j`'s target mechanism alone: only the target law
    /// depends on it, so its Jacobian has the target's cells only.
    Target(usize),
}

/// Columns of a block: `(population is target, mechanism row)`.
fn block_columns(shape: &Shape, model: &Model, block: Block) -> Vec<(bool, usize)> {
    match block {
        Block::Kernel(j) => {
            let rows = shape.rows(j, &model.cards);
            let mut columns: Vec<(bool, usize)> = (0..rows).map(|r| (false, r)).collect();
            if shape.selected[j] {
                columns.extend((0..rows).map(|r| (true, r)));
            }
            columns
        }
        Block::Target(j) => (0..shape.rows(j, &model.cards)).map(|r| (true, r)).collect(),
    }
}

/// The model's parameters modulo `P`.
struct ModModel {
    latents: Vec<Vec<u64>>,
    source: Vec<Vec<u64>>,
    target: Vec<Vec<u64>>,
}

fn mod_model(model: &Model) -> Option<ModModel> {
    let all = |v: &Vec<Vec<Q>>| {
        v.iter()
            .map(|row| row.iter().map(|q| fp_of(*q)).collect::<Option<Vec<_>>>())
            .collect::<Option<Vec<_>>>()
    };
    Some(ModModel {
        latents: all(&model.latents)?,
        source: all(&model.source)?,
        target: all(&model.target)?,
    })
}

/// Every supplied law cell as `(target population, intervened mask, world)`.
fn law_cells(n: usize) -> Vec<(bool, usize, usize)> {
    let full = 1usize << n;
    let mut cells: Vec<(bool, usize, usize)> =
        (0..full).flat_map(|mask| (0..full).map(move |world| (false, mask, world))).collect();
    cells.extend((0..full).map(|world| (true, 0, world)));
    cells
}

/// A base model's factors modulo `P`, tabulated once per base model: for every
/// node, world and latent configuration, the mechanism row and the factor
/// `P(v_j = world_j | row)` in each population, and the latent mass.
struct Tables {
    configs: usize,
    /// Level of every latent in every configuration, `[e][code]`.
    level: Vec<Vec<usize>>,
    /// `[j][world * configs + code]`.
    row: Vec<Vec<usize>>,
    source: Vec<Vec<u64>>,
    target: Vec<Vec<u64>>,
    /// Latent laws modulo `P`.
    latents: Vec<Vec<u64>>,
}

impl Tables {
    fn new(shape: &Shape, cards: &[usize], m: &ModModel) -> Self {
        let configs: usize = cards.iter().product();
        let mut level = vec![vec![0usize; configs]; cards.len()];
        let mut u = vec![0usize; cards.len()];
        for code in 0..configs {
            for (e, l) in u.iter().enumerate() {
                level[e][code] = *l;
            }
            next_config(&mut u, cards);
        }
        let worlds = 1usize << shape.n;
        let mut row = vec![vec![0usize; worlds * configs]; shape.n];
        let mut source = vec![vec![0u64; worlds * configs]; shape.n];
        let mut target = vec![vec![0u64; worlds * configs]; shape.n];
        let mut u = vec![0usize; cards.len()];
        for j in 0..shape.n {
            for world in 0..worlds {
                for code in 0..configs {
                    for (e, l) in u.iter_mut().enumerate() {
                        *l = level[e][code];
                    }
                    let r = shape.row(j, world, &u, cards);
                    let one = (world >> j) & 1 == 1;
                    let factor = |p: u64| if one { p } else { fp_sub(1, p) };
                    let at = world * configs + code;
                    row[j][at] = r;
                    source[j][at] = factor(m.source[j][r]);
                    target[j][at] = factor(m.target[j][r]);
                }
            }
        }
        Self { configs, level, row, source, target, latents: m.latents.clone() }
    }

    /// Latent mass of every configuration.
    fn latent_mass(&self) -> Vec<u64> {
        (0..self.configs)
            .map(|code| {
                self.latents
                    .iter()
                    .enumerate()
                    .fold(1, |mass, (e, law)| fp_mul(mass, law[self.level[e][code]]))
            })
            .collect()
    }

    /// One cell `(target population, intervened mask, world)` modulo `P`.
    fn cell(&self, n: usize, target: bool, mask: usize, world: usize, latent: &[u64]) -> u64 {
        let factors = if target { &self.target } else { &self.source };
        let base = world * self.configs;
        let mut total = 0u64;
        for (code, mass) in latent.iter().enumerate() {
            let mut m = *mass;
            for (j, table) in factors.iter().enumerate().take(n) {
                if mask & (1 << j) == 0 {
                    m = fp_mul(m, table[base + code]);
                }
            }
            total = fp_add(total, m);
        }
        total
    }
}

/// The Jacobian of `cells` (`(target population, intervened mask, world)`)
/// with respect to `block`, modulo `P` (exact, since every cell is linear in
/// the block).
fn jacobian(
    shape: &Shape,
    cards: &[usize],
    t: &Tables,
    block: Block,
    columns: &[(bool, usize)],
    cells: &[(bool, usize, usize)],
) -> Vec<Vec<u64>> {
    let mut out = vec![vec![0u64; columns.len()]; cells.len()];
    let (block_node, rows) = match block {
        Block::Kernel(j) => (j, shape.rows(j, cards)),
        Block::Target(j) => (j, 0),
    };
    let latent = t.latent_mass();
    for (r, &(target, mask, world)) in cells.iter().enumerate() {
        if mask & (1 << block_node) != 0 {
            // An intervened node's mechanism does not enter the cell.
            continue;
        }
        let factors = if target { &t.target } else { &t.source };
        let base = world * t.configs;
        let out_row = &mut out[r];
        for (code, latent_mass) in latent.iter().enumerate() {
            // Product of every factor outside the block.
            let mut mass = *latent_mass;
            for (j, table) in factors.iter().enumerate() {
                if mask & (1 << j) == 0 && j != block_node {
                    mass = fp_mul(mass, table[base + code]);
                }
            }
            let j = block_node;
            let row = t.row[j][base + code];
            // Columns are the source rows, then (selected) the target rows; a
            // target block has the target rows only.
            let c = if target && shape.selected[j] { rows + row } else { row };
            out_row[c] = if (world >> j) & 1 == 1 {
                fp_add(out_row[c], mass)
            } else {
                fp_sub(out_row[c], mass)
            };
        }
    }
    out
}

/// A null-space basis of `matrix` modulo `P`, from its reduced row echelon form.
fn null_space(mut matrix: Vec<Vec<u64>>, columns: usize) -> Vec<Vec<u64>> {
    let mut pivots = Vec::new();
    let mut r = 0;
    for c in 0..columns {
        let Some(p) = (r..matrix.len()).find(|i| matrix[*i][c] != 0) else { continue };
        matrix.swap(r, p);
        let inv = fp_inv(matrix[r][c]);
        for x in &mut matrix[r] {
            *x = fp_mul(*x, inv);
        }
        let pivot_row = matrix[r].clone();
        for (i, row) in matrix.iter_mut().enumerate() {
            if i != r && row[c] != 0 {
                let f = row[c];
                for (x, y) in row.iter_mut().zip(&pivot_row) {
                    *x = fp_sub(*x, fp_mul(f, *y));
                }
            }
        }
        pivots.push(c);
        r += 1;
        if r == matrix.len() {
            break;
        }
    }
    (0..columns)
        .filter(|c| !pivots.contains(c))
        .map(|free| {
            let mut v = vec![0u64; columns];
            v[free] = 1;
            for (i, &pc) in pivots.iter().enumerate() {
                v[pc] = fp_sub(0, matrix[i][free]);
            }
            v
        })
        .collect()
}

/// `base + eps * direction` on `block`, with `eps` small enough that every
/// parameter stays strictly inside (0, 1).
fn perturb(
    shape: &Shape,
    model: &Model,
    block: Block,
    columns: &[(bool, usize)],
    direction: &[Q],
) -> Option<Model> {
    // Integer direction: clear denominators, divide out the common factor.
    let mut lcm = 1i128;
    for q in direction {
        lcm = (lcm / gcd(lcm, q.d)).checked_mul(q.d)?;
    }
    let ints: Vec<i128> =
        direction.iter().map(|q| q.n.checked_mul(lcm / q.d)).collect::<Option<_>>()?;
    let g = ints.iter().fold(0i128, |acc, x| gcd(acc, *x));
    if g == 0 {
        return None;
    }
    let ints: Vec<i128> = ints.iter().map(|x| x / g).collect();
    let largest = ints.iter().map(|x| x.abs()).max()?;
    // Base mechanism probabilities are at least 1/9 away from 0 and 1, and no
    // entry moves by more than 1/20.
    let eps = Q::new(1, largest.checked_mul(20)?)?;
    let mut out = Model {
        cards: model.cards.clone(),
        latents: model.latents.clone(),
        source: model.source.clone(),
        target: model.target.clone(),
    };
    for (&(population, index), &d) in columns.iter().zip(&ints) {
        if d == 0 {
            continue;
        }
        let step = eps.mul(Q::new(d, 1)?)?;
        let (Block::Kernel(j) | Block::Target(j)) = block;
        if population {
            out.target[j][index] = out.target[j][index].add(step)?;
        } else {
            out.source[j][index] = out.source[j][index].add(step)?;
            if !shape.selected[j] {
                // A non-selected node's target mechanism is its source one.
                out.target[j][index] = out.source[j][index];
            }
        }
    }
    Some(out)
}

fn model_record(shape: &Shape, model: &Model) -> WitnessModelRecord {
    let text = |row: &[Q]| row.iter().map(|q| q.to_text()).collect::<Vec<_>>();
    let kernel = |j: usize, ones: &[Q]| WitnessKernelRecord {
        node: shape.ids[j],
        parents: shape.parents[j].iter().map(|p| shape.ids[*p]).collect(),
        latents: shape.incident[j]
            .iter()
            .map(|e| [shape.ids[shape.edges[*e].0], shape.ids[shape.edges[*e].1]])
            .collect(),
        ones: text(ones),
    };
    WitnessModelRecord {
        latents: shape
            .edges
            .iter()
            .zip(&model.latents)
            .map(|(&(a, b), law)| WitnessLatentRecord {
                edge: [shape.ids[a], shape.ids[b]],
                probabilities: text(law),
            })
            .collect(),
        source: (0..shape.n).map(|j| kernel(j, &model.source[j])).collect(),
        target: (0..shape.n)
            .filter(|j| shape.selected[*j])
            .map(|j| kernel(j, &model.target[j]))
            .collect(),
    }
}

/// Live-state estimate (bytes) of one search attempt on a block of `columns`
/// parameters: the base model's tables (row, source and target factors per
/// node, world and latent configuration) and the two Jacobians (every supplied
/// law cell, and every query cell, by every column).
fn attempt_bytes(shape: &Shape, configs: usize, law_cells: usize, columns: usize) -> u64 {
    let word = std::mem::size_of::<u64>() as u64;
    let worlds = 1u64 << shape.n;
    let tables =
        3u64.saturating_mul(shape.n as u64).saturating_mul(worlds).saturating_mul(configs as u64);
    let jacobians = (law_cells as u64).saturating_add(worlds).saturating_mul(columns as u64);
    tables.saturating_add(jacobians).saturating_mul(word)
}

/// Search for a verified two-model witness of `query`'s non-transportability.
///
/// Deterministic and bounded: at most three latent cardinalities times two
/// seeds times one parameter block at a time, with one `charge` (the caller's
/// budget, carrying the live-state estimate in bytes) for each block's linear
/// algebra and one more for every null-space direction tried, so every unit
/// of work is metered; nothing is attempted when one model's enumeration
/// exceeds an internal bound. Returns `NotFound` when no attempt produced a pair the exact
/// verifier accepts and `OutOfScope` when the diagram is too large to attempt;
/// a found record has been accepted by [`verify_conditional_witness`], whose
/// check travels with it.
///
/// # Errors
/// Whatever `charge` returns (budget or cancellation), or an invalid query.
pub fn search_conditional_witness(
    diagram: &SelectionDiagram,
    query: &ConditionalTransportQuery,
    charge: &mut dyn FnMut(u64) -> Result<(), IdentificationError>,
    ctx: &ExecutionContext,
) -> Result<WitnessSearch, IdentificationError> {
    validate(diagram, query)?;
    let shape = Shape::new(diagram)?;
    let levels = QueryLevels::new(&shape, query)?;
    let cells = law_cells(shape.n);
    let target_cells: Vec<(bool, usize, usize)> =
        cells.iter().copied().filter(|(target, _, _)| *target).collect();
    // The target law under do(x): every world, from which each level's
    // numerator and denominator are sums.
    let query_cells: Vec<(bool, usize, usize)> =
        (0..1usize << shape.n).map(|world| (true, levels.x_mask, world)).collect();
    let mut attempted = false;
    // Order measured on the three- and four-node classes: three latent levels
    // succeed more often than five or two, and target-only mechanisms are the
    // cheapest systems.
    for seed in 0..SEARCH_SEEDS {
        for cardinality in SEARCH_LATENT_LEVELS {
            if shape.work(&vec![cardinality; shape.edges.len()]) > SEARCH_MAX_WORK {
                continue;
            }
            attempted = true;
            let blocks: Vec<Block> = (0..shape.n)
                .filter(|j| shape.selected[*j])
                .map(Block::Target)
                .chain((0..shape.n).map(Block::Kernel))
                .collect();
            let Some(base) = base_model(&shape, cardinality, seed) else { continue };
            let Some(base_mod) = mod_model(&base) else { continue };
            let tables = Tables::new(&shape, &base.cards, &base_mod);
            let latent = tables.latent_mass();
            let q0: Vec<u64> = query_cells
                .iter()
                .map(|&(t, mask, world)| tables.cell(shape.n, t, mask, world, &latent))
                .collect();
            for block in blocks {
                let columns = block_columns(&shape, &base, block);
                let bytes = attempt_bytes(&shape, tables.configs, cells.len(), columns.len());
                // One charge for the block's linear algebra (its Jacobians and
                // null space) ...
                charge(bytes)?;
                if columns.is_empty() || columns.len() > SEARCH_MAX_BLOCK_PARAMETERS {
                    continue;
                }
                let gradient =
                    jacobian(&shape, &base.cards, &tables, block, &columns, &query_cells);
                // Skip a block the query does not depend on.
                if gradient.iter().all(|row| row.iter().all(|c| *c == 0)) {
                    continue;
                }
                let jac = if matches!(block, Block::Target(_)) {
                    jacobian(&shape, &base.cards, &tables, block, &columns, &target_cells)
                } else {
                    jacobian(&shape, &base.cards, &tables, block, &columns, &cells)
                };
                for direction in null_space(jac, columns.len()) {
                    // ... and one for every null-space direction tried (its
                    // screening, lifting, perturbation and exact check).
                    charge(bytes)?;
                    // The query's change along the direction, modulo P.
                    let q1: Vec<u64> = gradient
                        .iter()
                        .map(|row| {
                            row.iter()
                                .zip(&direction)
                                .fold(0, |acc, (a, b)| fp_add(acc, fp_mul(*a, *b)))
                        })
                        .collect();
                    let Some(at) = levels.first_moved(&q0, &q1) else { continue };
                    let Some(direction) =
                        direction.iter().map(|a| reconstruct(*a)).collect::<Option<Vec<_>>>()
                    else {
                        continue;
                    };
                    let Some(second) = perturb(&shape, &base, block, &columns, &direction) else {
                        continue;
                    };
                    let mut record = ConditionalWitnessRecord {
                        first: model_record(&shape, &base),
                        second: model_record(&shape, &second),
                        treatment_level: at.0,
                        conditioned_level: at.1,
                        outcome_level: at.2,
                        first_value: String::new(),
                        second_value: String::new(),
                    };
                    // The exact verifier is the only certificate: the pair's
                    // values are read off once, and the record then passes
                    // through the public verifier exactly as a consumer runs it.
                    match evaluate(&shape, query, &record, ctx) {
                        Ok((a, b, _)) if a != b => {
                            record.first_value = a.to_text();
                            record.second_value = b.to_text();
                            let check = verify_conditional_witness(diagram, query, &record, ctx)?;
                            return Ok(WitnessSearch::Found(Box::new(record), check));
                        }
                        Err(IdentificationError::Cancelled) => {
                            return Err(IdentificationError::Cancelled);
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    Ok(if attempted { WitnessSearch::NotFound } else { WitnessSearch::OutOfScope })
}

/// Query coordinates as dense masks, for the search's modular screening.
struct QueryLevels {
    x: Vec<usize>,
    w: Vec<usize>,
    y: Vec<usize>,
    x_mask: usize,
    w_mask: usize,
    y_mask: usize,
}

impl QueryLevels {
    fn new(shape: &Shape, query: &ConditionalTransportQuery) -> Result<Self, IdentificationError> {
        let dense =
            |vs: &[antecedent_core::VariableId]| -> Result<Vec<usize>, IdentificationError> {
                vs.iter().map(|v| shape.dense_of(v.raw())).collect()
            };
        let (x, w, y) = (
            dense(&query.base.treatments)?,
            dense(&query.conditioned_on)?,
            dense(&query.base.outcomes)?,
        );
        let mask = |vs: &[usize]| vs.iter().fold(0usize, |m, j| m | (1 << j));
        Ok(Self { x_mask: mask(&x), w_mask: mask(&w), y_mask: mask(&y), x, w, y })
    }

    /// Bits and levels of `vs` at row-major code `code` (first coordinate most
    /// significant).
    fn bits(vs: &[usize], code: usize) -> (usize, Vec<u8>) {
        let mut set = 0usize;
        let mut levels = Vec::with_capacity(vs.len());
        for (i, j) in vs.iter().enumerate() {
            let bit = (code >> (vs.len() - 1 - i)) & 1;
            set |= bit << j;
            levels.push(u8::from(bit == 1));
        }
        (set, levels)
    }

    /// The first level (treatments, conditioned, outcomes; row-major) at which
    /// `q0 + t * q1` (the target law under do(x) per world, modulo P) changes
    /// `P*(y | do(x), w)` to first order: `N1 D0 != N0 D1`. Every world of
    /// `q0` is under the same intervened mask, so each treatment level selects
    /// its own worlds.
    fn first_moved(&self, q0: &[u64], q1: &[u64]) -> Option<(Vec<u8>, Vec<u8>, Vec<u8>)> {
        for xc in 0..(1usize << self.x.len()) {
            let (x_bits, x_levels) = Self::bits(&self.x, xc);
            for wc in 0..(1usize << self.w.len()) {
                let (w_bits, w_levels) = Self::bits(&self.w, wc);
                for yc in 0..(1usize << self.y.len()) {
                    let (y_bits, y_levels) = Self::bits(&self.y, yc);
                    let (mut n0, mut d0, mut n1, mut d1) = (0u64, 0u64, 0u64, 0u64);
                    for (world, (a, b)) in q0.iter().zip(q1).enumerate() {
                        if world & self.x_mask != x_bits || world & self.w_mask != w_bits {
                            continue;
                        }
                        d0 = fp_add(d0, *a);
                        d1 = fp_add(d1, *b);
                        if world & self.y_mask == y_bits {
                            n0 = fp_add(n0, *a);
                            n1 = fp_add(n1, *b);
                        }
                    }
                    if fp_mul(n1, d0) != fp_mul(n0, d1) {
                        return Some((x_levels, w_levels, y_levels));
                    }
                }
            }
        }
        None
    }
}

/// What a bounded witness search found. Neither `NotFound` nor `OutOfScope`
/// is a claim about the query.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WitnessSearch {
    /// A pair the exact verifier accepted, with what it established.
    Found(Box<ConditionalWitnessRecord>, ConditionalWitnessCheck),
    /// Every attempt within the bounds failed to produce a verified pair.
    NotFound,
    /// The diagram is too large for any attempt within the search bounds.
    OutOfScope,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rationals_are_canonical_and_checked() {
        let a = Q::parse("1/3").unwrap();
        let b = Q::parse("1/6").unwrap();
        assert_eq!(a.add(b).unwrap().to_text(), "1/2");
        assert!(Q::parse("2/6").is_none(), "non-canonical spelling");
        assert!(Q::parse("1/-3").is_none());
        assert!(Q::parse("+1/3").is_none());
        assert!(Q::new(i128::MAX, 1).unwrap().add(Q::ONE).is_none(), "overflow refuses");
    }

    #[test]
    fn reconstruction_recovers_small_fractions() {
        for (n, d) in [(3, 7), (-5, 12), (0, 1), (1_000_003, 999_983)] {
            let q = Q::new(n, d).unwrap();
            assert_eq!(reconstruct(fp_of(q).unwrap()), Some(q));
        }
    }
}
