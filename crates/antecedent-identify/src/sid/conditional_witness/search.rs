//! Bounded modular search for conditional transport witnesses.

use super::{
    ChargeFn, ConditionalTransportQuery, ConditionalWitnessRecord, ExecutionContext,
    IdentificationError, Model, Q, Quote, SEARCH_ENTRY_BYTES, SEARCH_LATENT_LEVELS,
    SEARCH_MAX_BLOCK_PARAMETERS, SEARCH_SEEDS, SelectionDiagram, Shape,
    WITNESS_SEARCH_DEFAULT_OPERATIONS, WITNESS_WORK_PER_OPERATION, WitnessKernelRecord,
    WitnessLatentRecord, WitnessMeter, WitnessModelRecord, WitnessSearch, evaluate, gcd,
    next_config, rejected, validate, verify_conditional_witness_within,
};

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
pub(super) fn fp_of(q: Q) -> Option<u64> {
    let p = i128::from(P);
    let n = u64::try_from(q.n.rem_euclid(p)).ok()?;
    let d = u64::try_from(q.d.rem_euclid(p)).ok()?;
    (d != 0).then(|| fp_mul(n, fp_inv(d)))
}

/// Rational reconstruction of `a mod P` with numerator and denominator below
/// `2^30`; `None` when no such fraction exists.
pub(super) fn reconstruct(a: u64) -> Option<Q> {
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

/// `SplitMix64` starting at `state`, the same step as [`antecedent_core::CausalRng`].
fn witness_rng(state: u64) -> antecedent_core::CausalRng {
    antecedent_core::CausalRng::from_state(state)
}

fn small(rng: &mut antecedent_core::CausalRng, n: u64) -> i128 {
    i128::from(1 + rng.next_u64() % n)
}

/// A seeded strictly positive base model: latent weights in `1..=4`
/// normalized, mechanism probabilities in `{1/9, ..., 8/9}`.
pub(super) fn base_model(shape: &Shape, levels: usize, seed: u64) -> Option<Model> {
    let mut rng = witness_rng(0xB1C0_2D17 ^ seed.wrapping_mul(0x1000_0000_01B3) ^ levels as u64);
    let cards = vec![levels; shape.edges.len()];
    let mut latents = Vec::with_capacity(cards.len());
    for _ in &shape.edges {
        let weights: Vec<i128> = (0..levels).map(|_| small(&mut rng, 4)).collect();
        let total: i128 = weights.iter().sum();
        latents.push(weights.iter().map(|w| Q::new(*w, total)).collect::<Option<Vec<_>>>()?);
    }
    let mut kernel = |j: usize| {
        let rows = shape.rows(j, &cards)?;
        (0..rows).map(|_| Q::new(small(&mut rng, 8), 9)).collect::<Option<Vec<_>>>()
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
fn block_columns(shape: &Shape, model: &Model, block: Block) -> Option<Vec<(bool, usize)>> {
    match block {
        Block::Kernel(j) => {
            let rows = shape.rows(j, &model.cards)?;
            let mut columns: Vec<(bool, usize)> = (0..rows).map(|r| (false, r)).collect();
            if shape.selected[j] {
                columns.extend((0..rows).map(|r| (true, r)));
            }
            Some(columns)
        }
        Block::Target(j) => {
            let rows = shape.rows(j, &model.cards)?;
            Some((0..rows).map(|r| (true, r)).collect())
        }
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
fn law_cells(n: usize) -> Option<Vec<(bool, usize, usize)>> {
    let full = crate::pow2::checked_pow2(n)?;
    let mut cells: Vec<(bool, usize, usize)> =
        (0..full).flat_map(|mask| (0..full).map(move |world| (false, mask, world))).collect();
    cells.extend((0..full).map(|world| (true, 0, world)));
    Some(cells)
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
    fn new(shape: &Shape, cards: &[usize], m: &ModModel) -> Option<Self> {
        let configs: usize = cards.iter().product();
        let mut level = vec![vec![0usize; configs]; cards.len()];
        let mut u = vec![0usize; cards.len()];
        for code in 0..configs {
            for (e, l) in u.iter().enumerate() {
                level[e][code] = *l;
            }
            next_config(&mut u, cards);
        }
        let worlds = crate::pow2::checked_pow2(shape.n)?;
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
        Some(Self { configs, level, row, source, target, latents: m.latents.clone() })
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
        Block::Kernel(j) => match shape.rows(j, cards) {
            Some(rows) => (j, rows),
            None => return Vec::new(),
        },
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
        lcm = (lcm / gcd(lcm, q.d)?).checked_mul(q.d)?;
    }
    let ints: Vec<i128> =
        direction.iter().map(|q| q.n.checked_mul(lcm / q.d)).collect::<Option<_>>()?;
    let g = ints.iter().try_fold(0i128, |acc, x| gcd(acc, *x))?;
    if g == 0 {
        return None;
    }
    let ints: Vec<i128> = ints.iter().map(|x| x / g).collect();
    let largest = ints.iter().try_fold(0i128, |largest, x| Some(largest.max(x.checked_abs()?)))?;
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
    let worlds = crate::pow2::checked_pow2_u64(shape.n).unwrap_or(u64::MAX);
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
/// of work is metered. A cardinality is attempted only when its enumeration is
/// quoted within the operations the meter still allows and the context's pool
/// funds its tables (leased before they are built); a bare charge function allows
/// [`WITNESS_SEARCH_DEFAULT_OPERATIONS`]. Returns `NotFound` when no attempt produced
/// a pair the exact verifier accepts and `OutOfScope` when no cardinality's quote fits; a
/// found record has been accepted by [`verify_conditional_witness`], whose check travels
/// with it.
///
/// # Errors
/// Whatever `charge` returns (budget or cancellation), or an invalid query.
pub fn search_conditional_witness(
    diagram: &SelectionDiagram,
    query: &ConditionalTransportQuery,
    charge: &mut dyn FnMut(u64) -> Result<(), IdentificationError>,
    ctx: &ExecutionContext,
) -> Result<WitnessSearch, IdentificationError> {
    search_conditional_witness_within(diagram, query, &mut ChargeFn(charge), ctx)
}

/// [`search_conditional_witness`] against a [`WitnessMeter`] that also reports the operations
/// the caller's budget still allows, from which every quote is derived.
///
/// # Errors
/// As [`search_conditional_witness`].
pub fn search_conditional_witness_within(
    diagram: &SelectionDiagram,
    query: &ConditionalTransportQuery,
    meter: &mut dyn WitnessMeter,
    ctx: &ExecutionContext,
) -> Result<WitnessSearch, IdentificationError> {
    validate(diagram, query)?;
    let shape = Shape::new(diagram)?;
    // The law cells (`4^n + 2^n` of them) and query cells are built only once an attempt's
    // quote, whose work counts every law cell, has been admitted: a diagram too wide for any
    // attempt is refused without materialising them.
    let mut prepared: Option<SearchCells> = None;
    let mut attempted = false;
    let mut budget_refused = false;
    // Order measured on the three- and four-node classes: three latent levels
    // succeed more often than five or two, and target-only mechanisms are the
    // cheapest systems.
    for seed in 0..SEARCH_SEEDS {
        for cardinality in SEARCH_LATENT_LEVELS {
            // The base model's modular tables, and the exact check of the pair a perturbation
            // would produce (two models of this cardinality), are quoted against what the
            // budget allows now and the tables are leased before they are built; a refusal
            // skips the cardinality. The same allowance covers every check in the attempt.
            let quoted_operations = meter.operations_remaining();
            let cards = vec![cardinality; shape.edges.len()];
            let table_quote = Quote::of(&shape, &cards, SEARCH_ENTRY_BYTES);
            let pair_quote = Quote { work: table_quote.work.saturating_mul(2), ..table_quote };
            let _tables = match pair_quote.admit(quoted_operations, ctx) {
                Ok(lease) => lease,
                Err(IdentificationError::Cancelled) => {
                    return Err(IdentificationError::Cancelled);
                }
                Err(IdentificationError::Budget { .. })
                    if pair_quote.work
                        <= WITNESS_SEARCH_DEFAULT_OPERATIONS
                            .saturating_mul(WITNESS_WORK_PER_OPERATION) =>
                {
                    // The standard allowance would cover this attempt, so the caller's
                    // remaining budget, not the shape, is what refused it: the search did
                    // not run its course, which is not a verdict of "no witness".
                    budget_refused = true;
                    continue;
                }
                Err(_) => continue,
            };
            attempted = true;
            if prepared.is_none() {
                prepared = Some(SearchCells::new(&shape, query)?);
            }
            let Some(SearchCells { levels, cells, target_cells, query_cells }) = prepared.as_ref()
            else {
                continue;
            };
            let blocks: Vec<Block> = (0..shape.n)
                .filter(|j| shape.selected[*j])
                .map(Block::Target)
                .chain((0..shape.n).map(Block::Kernel))
                .collect();
            let Some(base) = base_model(&shape, cardinality, seed) else { continue };
            let Some(base_mod) = mod_model(&base) else { continue };
            let Some(tables) = Tables::new(&shape, &base.cards, &base_mod) else { continue };
            let latent = tables.latent_mass();
            let q0: Vec<u64> = query_cells
                .iter()
                .map(|&(t, mask, world)| tables.cell(shape.n, t, mask, world, &latent))
                .collect();
            for block in blocks {
                let Some(columns) = block_columns(&shape, &base, block) else { continue };
                let bytes = attempt_bytes(&shape, tables.configs, cells.len(), columns.len());
                // One charge for the block's linear algebra (its Jacobians and
                // null space) ...
                meter.charge(bytes)?;
                if columns.is_empty() || columns.len() > SEARCH_MAX_BLOCK_PARAMETERS {
                    continue;
                }
                let gradient = jacobian(&shape, &base.cards, &tables, block, &columns, query_cells);
                // Skip a block the query does not depend on.
                if gradient.iter().all(|row| row.iter().all(|c| *c == 0)) {
                    continue;
                }
                let jac = if matches!(block, Block::Target(_)) {
                    jacobian(&shape, &base.cards, &tables, block, &columns, target_cells)
                } else {
                    jacobian(&shape, &base.cards, &tables, block, &columns, cells)
                };
                for direction in null_space(jac, columns.len()) {
                    // ... and one for every null-space direction tried (its
                    // screening, lifting, perturbation and exact check).
                    meter.charge(bytes)?;
                    // The query's change along the direction, modulo P.
                    let q1 = modular_direction_effect(&gradient, &direction);
                    let Some(at) = levels.first_moved(&q0, &q1) else { continue };
                    let Some(direction) =
                        direction.iter().map(|a| reconstruct(*a)).collect::<Option<Vec<_>>>()
                    else {
                        continue;
                    };
                    let Some(second) = perturb(&shape, &base, block, &columns, &direction) else {
                        continue;
                    };
                    let certification = WitnessCertification {
                        diagram,
                        query,
                        shape: &shape,
                        quoted_operations,
                        ctx,
                    };
                    if let Some(witness) = certification.try_pair(&base, &second, at)? {
                        return Ok(witness);
                    }
                }
            }
        }
    }
    if budget_refused {
        return Err(IdentificationError::budget(crate::IdentificationBudget::Steps));
    }
    Ok(if attempted { WitnessSearch::NotFound } else { WitnessSearch::OutOfScope })
}

/// Screen the query's change along a null-space direction modulo the search
/// prime. Row and coefficient order agree with the Jacobian construction.
fn modular_direction_effect(gradient: &[Vec<u64>], direction: &[u64]) -> Vec<u64> {
    gradient
        .iter()
        .map(|row| row.iter().zip(direction).fold(0, |acc, (a, b)| fp_add(acc, fp_mul(*a, *b))))
        .collect()
}

/// Exact certification boundary for a candidate pair found by modular search.
/// Screening alone never publishes a witness: evaluate and then run the same
/// public verifier that independent consumers use.
pub(super) struct WitnessCertification<'a> {
    pub(super) diagram: &'a SelectionDiagram,
    pub(super) query: &'a ConditionalTransportQuery,
    pub(super) shape: &'a Shape,
    pub(super) quoted_operations: u64,
    pub(super) ctx: &'a ExecutionContext,
}

impl WitnessCertification<'_> {
    pub(super) fn try_pair(
        &self,
        first: &Model,
        second: &Model,
        at: (Vec<u8>, Vec<u8>, Vec<u8>),
    ) -> Result<Option<WitnessSearch>, IdentificationError> {
        let mut record = ConditionalWitnessRecord {
            first: model_record(self.shape, first),
            second: model_record(self.shape, second),
            treatment_level: at.0,
            conditioned_level: at.1,
            outcome_level: at.2,
            first_value: String::new(),
            second_value: String::new(),
        };
        match evaluate(self.shape, self.query, &record, self.quoted_operations, self.ctx) {
            Ok((a, b, _)) if a != b => {
                record.first_value = a.to_text();
                record.second_value = b.to_text();
                let check = verify_conditional_witness_within(
                    self.diagram,
                    self.query,
                    &record,
                    self.quoted_operations,
                    self.ctx,
                )?;
                Ok(Some(WitnessSearch::Found(Box::new(record), check)))
            }
            Err(error) if error.is_budget_or_cancel() => Err(error),
            _ => Ok(None),
        }
    }
}

/// The cells a witness search screens: every supplied law cell, its target-population
/// part, and the target law under do(x) at every world.
struct SearchCells {
    levels: QueryLevels,
    cells: Vec<(bool, usize, usize)>,
    target_cells: Vec<(bool, usize, usize)>,
    query_cells: Vec<(bool, usize, usize)>,
}

impl SearchCells {
    fn new(shape: &Shape, query: &ConditionalTransportQuery) -> Result<Self, IdentificationError> {
        let bound = || rejected("witness exceeds the enumeration bound");
        if shape.n >= usize::BITS as usize {
            return Err(bound());
        }
        let levels = QueryLevels::new(shape, query)?;
        let cells = law_cells(shape.n).ok_or_else(bound)?;
        let target_cells = cells.iter().copied().filter(|(target, _, _)| *target).collect();
        // The target law under do(x): every world, from which each level's
        // numerator and denominator are sums.
        let query_worlds = crate::pow2::checked_pow2(shape.n).ok_or_else(bound)?;
        let query_cells = (0..query_worlds).map(|world| (true, levels.x_mask, world)).collect();
        Ok(Self { levels, cells, target_cells, query_cells })
    }
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
        let (x_n, w_n, y_n) = (
            crate::pow2::checked_pow2(self.x.len())?,
            crate::pow2::checked_pow2(self.w.len())?,
            crate::pow2::checked_pow2(self.y.len())?,
        );
        for xc in 0..x_n {
            let (x_bits, x_levels) = Self::bits(&self.x, xc);
            for wc in 0..w_n {
                let (w_bits, w_levels) = Self::bits(&self.w, wc);
                for yc in 0..y_n {
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
