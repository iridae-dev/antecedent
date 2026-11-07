//! Self-certifying two-model witnesses for ADMG conditional transport.
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

mod search;
#[cfg(test)]
use search::{WitnessCertification, base_model, fp_of, reconstruct};
pub use search::{search_conditional_witness, search_conditional_witness_within};

/// Enumeration work (law cells times latent configurations times nodes) one budget operation
/// covers. A witness enumeration is quoted in this unit against the operations the caller's
/// budget still allows, and its tables are leased from the context's admission pool before they
/// are allocated; there is no fixed work ceiling.
pub const WITNESS_WORK_PER_OPERATION: u64 = 1 << 8;
/// Operations [`verify_conditional_witness`] declares for itself when the caller supplies no
/// budget (see [`verify_conditional_witness_within`]).
pub const WITNESS_VERIFY_DEFAULT_OPERATIONS: u64 = 1 << 14;
/// Operations a bare charge function is taken to allow a witness search (see
/// [`search_conditional_witness`]).
pub const WITNESS_SEARCH_DEFAULT_OPERATIONS: u64 = 1 << 12;
/// Most levels of one latent variable in a *stored artifact* of a witness (a parse
/// bound only). Verification applies no level cap: its enumeration is quoted against
/// the caller's operations and the admission pool before any table is built.
pub const CONDITIONAL_WITNESS_MAX_LATENT_LEVELS: usize = 8;
/// Most free parameters of one perturbed block in the search.
const SEARCH_MAX_BLOCK_PARAMETERS: usize = 256;
/// Latent cardinalities the search tries, in order.
const SEARCH_LATENT_LEVELS: [usize; 3] = [3, 5, 2];
/// Base models per latent cardinality.
const SEARCH_SEEDS: u64 = 2;

/// Stable failure text of a witness that does not verify.
const NOT_VERIFIED: &str = antecedent_core::reason_details::admg_transport::INVALID_DERIVATION;

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
    debug_assert_eq!(
        NOT_VERIFIED,
        antecedent_core::reason_details::admg_transport::INVALID_DERIVATION
    );
    invalid_derivation()
}

// ---------------------------------------------------------------------------
// Exact rationals (checked i128; any overflow refuses).

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Q {
    n: i128,
    d: i128,
}

fn gcd(a: i128, b: i128) -> Option<i128> {
    let mut a = a.checked_abs()?;
    let mut b = b.checked_abs()?;
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    Some(a)
}

impl Q {
    const ZERO: Self = Self { n: 0, d: 1 };
    const ONE: Self = Self { n: 1, d: 1 };

    fn new(n: i128, d: i128) -> Option<Self> {
        if d == 0 {
            return None;
        }
        let g = gcd(n, d)?;
        let (mut n, mut d) = (n / g, d / g);
        if d < 0 {
            n = n.checked_neg()?;
            d = d.checked_neg()?;
        }
        Some(Self { n, d })
    }
    fn add(self, o: Self) -> Option<Self> {
        let g = gcd(self.d, o.d)?;
        let l = (self.d / g).checked_mul(o.d)?;
        let a = self.n.checked_mul(l / self.d)?;
        let b = o.n.checked_mul(l / o.d)?;
        Self::new(a.checked_add(b)?, l)
    }
    fn mul(self, o: Self) -> Option<Self> {
        let g1 = gcd(self.n, o.d)?.max(1);
        let g2 = gcd(o.n, self.d)?.max(1);
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
            let node = DenseNodeId::try_from_usize(i).map_err(|_| invalid_derivation())?;
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
    ///
    /// `None` when the parent-set power of two does not fit in `usize`. Under the
    /// observed-node cap that exponent is at most five, and the value is the same
    /// shift the length check used to make.
    fn rows(&self, j: usize, cards: &[usize]) -> Option<usize> {
        let parents = crate::pow2::checked_pow2(self.parents[j].len())?;
        Some(parents * self.incident[j].iter().map(|e| cards[*e]).product::<usize>())
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
        // `1 << (2n)` overflows once `n >= 32`; such a diagram is over every work bound,
        // so it quotes `u64::MAX` and the budget refuses it before any table exists.
        let Some(cells) = crate::pow2::checked_pow2_u64(self.n.saturating_mul(2))
            .zip(crate::pow2::checked_pow2_u64(self.n))
            .map(|(wide, narrow)| wide.saturating_add(narrow))
        else {
            return u64::MAX;
        };
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
// Work and storage quotes, derived from the caller's budget.

/// What one enumeration needs: its work in the unit of
/// [`WITNESS_WORK_PER_OPERATION`] and the bytes of the tables it allocates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Quote {
    work: u64,
    bytes: u64,
}

impl Quote {
    /// The quote of one model with latent cardinalities `cards`, whose tables hold
    /// `entry_bytes` per (node, world, latent configuration) and per latent configuration.
    fn of(shape: &Shape, cards: &[usize], entry_bytes: u64) -> Self {
        let configs = cards.iter().map(|c| *c as u64).fold(1u64, u64::saturating_mul);
        let worlds = crate::pow2::checked_pow2_u64(shape.n).unwrap_or(u64::MAX);
        let tables = (shape.n.max(1) as u64)
            .saturating_mul(worlds)
            .saturating_mul(configs)
            .saturating_mul(entry_bytes);
        let latent = configs
            .saturating_mul(cards.len().max(1) as u64)
            .saturating_mul(std::mem::size_of::<usize>() as u64)
            .saturating_add(configs.saturating_mul(entry_bytes));
        Self { work: shape.work(cards), bytes: tables.saturating_add(latent) }
    }

    const fn plus(self, other: Self) -> Self {
        Self {
            work: self.work.saturating_add(other.work),
            bytes: self.bytes.saturating_add(other.bytes),
        }
    }

    /// Check the work against `operations` still allowed and lease the bytes from the
    /// context's pool. Nothing is allocated here; a refusal happens before any table exists.
    ///
    /// # Errors
    /// `Budget(Steps)` when the work exceeds what `operations` cover, `Budget(Memory)` when the
    /// pool cannot fund the tables, and `Cancelled`.
    fn admit(
        self,
        operations: u64,
        ctx: &ExecutionContext,
    ) -> Result<antecedent_core::execution::AdmissionLease, IdentificationError> {
        if ctx.cancellation.is_cancelled() {
            return Err(IdentificationError::Cancelled);
        }
        if self.work > operations.saturating_mul(WITNESS_WORK_PER_OPERATION) {
            return Err(IdentificationError::budget(crate::IdentificationBudget::Steps));
        }
        ctx.acquire_admission(0, self.bytes).map_err(crate::error::map_admission_error)
    }
}

/// Bytes per tabulated entry of the verifier's exact integer tables (`i128`, two populations).
const VERIFY_ENTRY_BYTES: u64 = 2 * std::mem::size_of::<i128>() as u64;
/// Bytes per tabulated entry of the search's modular tables (row, source and target words).
const SEARCH_ENTRY_BYTES: u64 = 3 * std::mem::size_of::<u64>() as u64;

/// The meter a witness search charges: one charge per block and per direction, with the
/// live-state estimate of the attempt, and the operations the caller's budget still allows.
pub trait WitnessMeter {
    /// Charge one operation whose live state needs `bytes`.
    ///
    /// # Errors
    /// The stop of the underlying budget, as an identification error.
    fn charge(&mut self, bytes: u64) -> Result<(), IdentificationError>;
    /// Operations the budget still allows.
    fn operations_remaining(&self) -> u64;
}

/// A bare charge function: declares no operation limit of its own, so the search is quoted
/// against [`WITNESS_SEARCH_DEFAULT_OPERATIONS`].
struct ChargeFn<'a>(&'a mut dyn FnMut(u64) -> Result<(), IdentificationError>);

impl WitnessMeter for ChargeFn<'_> {
    fn charge(&mut self, bytes: u64) -> Result<(), IdentificationError> {
        (self.0)(bytes)
    }
    fn operations_remaining(&self) -> u64 {
        WITNESS_SEARCH_DEFAULT_OPERATIONS
    }
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
        if k < 2 {
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
    let kernel = |j: usize, record: &WitnessKernelRecord| -> Result<Vec<Q>, IdentificationError> {
        let parents: Vec<u32> = shape.parents[j].iter().map(|p| shape.ids[*p]).collect();
        let incident: Vec<[u32; 2]> = shape.incident[j]
            .iter()
            .map(|e| [shape.ids[shape.edges[*e].0], shape.ids[shape.edges[*e].1]])
            .collect();
        if record.node != shape.ids[j] || record.parents != parents || record.latents != incident {
            return Err(rejected("a mechanism is not over its node's graph parents and latents"));
        }
        if Some(record.ones.len()) != shape.rows(j, &cards) {
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
                lcm = (lcm / gcd(lcm, q.d)?).checked_mul(q.d)?;
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
        let worlds = crate::pow2::checked_pow2(shape.n)?;
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
    for world in 0..crate::pow2::checked_pow2(shape.n)? {
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
    operations: u64,
    ctx: &ExecutionContext,
) -> Result<(Q, Q, usize), IdentificationError> {
    let overflow = || rejected("arithmetic overflow");
    let (first_model, second_model) =
        (parse_model(shape, &record.first)?, parse_model(shape, &record.second)?);
    // Both models' exact tables are quoted against the budget and leased before either is built.
    let _tables = Quote::of(shape, &first_model.cards, VERIFY_ENTRY_BYTES)
        .plus(Quote::of(shape, &second_model.cards, VERIFY_ENTRY_BYTES))
        .admit(operations, ctx)?;
    let first = Integral::new(shape, &first_model).ok_or_else(overflow)?;
    let second = Integral::new(shape, &second_model).ok_or_else(overflow)?;
    let mut compared = 0usize;
    let full = crate::pow2::checked_pow2(shape.n).ok_or_else(overflow)?;
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
/// Declares [`WITNESS_VERIFY_DEFAULT_OPERATIONS`] as its budget; see
/// [`verify_conditional_witness_within`] to verify under the caller's own.
///
/// # Errors
/// `admg_transport.invalid_derivation` when any check fails (including an
/// arithmetic overflow), an invalid query, or cancellation; a budget error when the
/// enumeration's quote exceeds the declared operations or the pool cannot fund its tables.
pub fn verify_conditional_witness(
    diagram: &SelectionDiagram,
    query: &ConditionalTransportQuery,
    record: &ConditionalWitnessRecord,
    ctx: &ExecutionContext,
) -> Result<ConditionalWitnessCheck, IdentificationError> {
    verify_conditional_witness_within(
        diagram,
        query,
        record,
        WITNESS_VERIFY_DEFAULT_OPERATIONS,
        ctx,
    )
}

/// [`verify_conditional_witness`] under `operations` of the caller's budget.
///
/// The enumeration is quoted before any arithmetic: its work (in units of
/// [`WITNESS_WORK_PER_OPERATION`]) must fit `operations`, and the bytes of the exact tables
/// of both models are leased from the context's admission pool before either is built.
/// Cancellation is polled per law.
///
/// # Errors
/// As [`verify_conditional_witness`].
pub fn verify_conditional_witness_within(
    diagram: &SelectionDiagram,
    query: &ConditionalTransportQuery,
    record: &ConditionalWitnessRecord,
    operations: u64,
    ctx: &ExecutionContext,
) -> Result<ConditionalWitnessCheck, IdentificationError> {
    validate(diagram, query)?;
    let shape = Shape::new(diagram)?;
    let (a, b, cells_compared) = evaluate(&shape, query, record, operations, ctx)?;
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
        assert!(Q::parse(&format!("{}/3", i128::MIN)).is_none());
        assert!(Q::new(i128::MIN, 3).is_none());
        assert!(Q::new(i128::MAX, 1).unwrap().add(Q::ONE).is_none(), "overflow refuses");
    }

    #[test]
    fn certification_propagates_table_memory_exhaustion() {
        use antecedent_core::VariableId;
        use antecedent_graph::Admg;
        use std::sync::Arc;

        let diagram = SelectionDiagram::try_new(Admg::with_variables(2), []).unwrap();
        let shape = Shape::new(&diagram).unwrap();
        let model = base_model(&shape, 3, 1).unwrap();
        let query = ConditionalTransportQuery {
            base: crate::sid::ClassicalTransportQuery {
                outcomes: vec![VariableId::from_raw(1)].into(),
                treatments: vec![VariableId::from_raw(0)].into(),
                source: Arc::from("source"),
                target: Arc::from("target"),
            },
            conditioned_on: Vec::new().into(),
        };
        let ctx = ExecutionContext::for_tests(1).with_memory_limit(Some(0));
        let certification = WitnessCertification {
            diagram: &diagram,
            query: &query,
            shape: &shape,
            quoted_operations: 1000,
            ctx: &ctx,
        };
        let result = certification.try_pair(&model, &model, (vec![1], vec![], vec![1]));
        assert!(matches!(result, Err(IdentificationError::Budget { .. })));
    }

    #[test]
    fn reconstruction_recovers_small_fractions() {
        for (n, d) in [(3, 7), (-5, 12), (0, 1), (1_000_003, 999_983)] {
            let q = Q::new(n, d).unwrap();
            assert_eq!(reconstruct(fp_of(q).unwrap()), Some(q));
        }
    }
}
