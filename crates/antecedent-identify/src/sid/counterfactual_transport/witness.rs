//! Replayable two-model witnesses of non-transportability.
//!
//! A witness is two finite structural models of the declared model class that
//! give *identical* laws for every regime of the declared evidence family (every
//! population's observational law and every experiment on a declared
//! controllable set) and *different* values of the counterfactual query. Such a
//! pair shows that no function of the declared evidence computes the query on
//! every model of the class: the query is not transportable. The proof is the
//! pair itself; the verifier trusts no theorem and re-derives every number by
//! exact rational enumeration.
//!
//! Model class of the witness. Every variable `V` is a deterministic *response
//! function* of its parents' values (a lookup table over the supports below),
//! drawn from a distribution over response functions. The distribution is one
//! joint law over the response functions of a bidirected clique `S` of the graph
//! (the latent terms the clique's edges name are exactly this joint) and an
//! independent baseline for every other variable. Populations share every
//! response function and every law except that a variable a source selects is a
//! constant (level of position 0) in that source, which is one admissible
//! different mechanism. Variables take at most two declared levels (the levels
//! the query mentions, then the first declared ones); a declared level outside
//! the support is read as the support's first level, so the models are
//! admissible for the declared domains. Every probability is an exact rational
//! strictly between 0 and 1 on the clique's states.
//!
//! Search. For a fixed baseline every regime-law cell and both the numerator and
//! denominator of the query are linear in the clique's law. A perturbation `d`
//! with `E d = 0` (`E` the cell matrix with a row of ones) leaves every law
//! unchanged, and the query ratio changes exactly when `qN.d * SD - qD.d * SN`
//! is nonzero. The null space is found by exact rational elimination; the
//! perturbed law is accepted only after [`verify_witness`] recomputes both
//! models. The search is a bounded heuristic, never a proof by itself: when no
//! witness is found the decision stays `not_certified`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use antecedent_core::SearchStop;
use serde::{Deserialize, Serialize};

use super::CounterfactualTransportProblem as Problem;
use super::network::{Built, Input, Network, WorldIdx, build_network};

/// Most response-function tuples of the varied clique.
const MAX_CLIQUE_STATES: usize = 256;
/// Most response-function tuples of the independent baseline (uniform kind).
const MAX_BASELINE_STATES: usize = 4096;
/// Most parent-configuration rows of one variable's response function.
const MAX_DOMAIN: usize = 16;
/// Most regime-law cells (rows of the evidence matrix).
const MAX_ROWS: usize = 4096;
/// Most simulations one witness may need (states times regimes times levels).
pub(super) const MAX_SIMULATIONS: u64 = 1 << 22;

/// What a verified witness established.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CounterfactualTransportWitnessCheck {
    /// Regime-law cells compared (both models, every family regime and level).
    pub cells_compared: usize,
    /// The query value in the first model, `"n/d"`.
    pub first_value: String,
    /// The query value in the second model, `"n/d"`.
    pub second_value: String,
}

/// Untrusted portable two-model witness.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CounterfactualTransportWitness {
    /// Variable ids of the varied bidirected clique, ascending.
    pub clique: Vec<u32>,
    /// Baseline of every other variable: `uniform`, `parity` or `conjunction`.
    pub baseline: String,
    /// Per variable (ascending id) the declared level indices of its support.
    pub support: Vec<Vec<u32>>,
    /// First model's law over the clique's response-function tuples, in
    /// canonical order (first variable most significant), exact `"n/d"`.
    pub first: Vec<String>,
    /// Second model's law over the same tuples.
    pub second: Vec<String>,
    /// The query value in the first model, `"n/d"`.
    pub first_value: String,
    /// The query value in the second model, `"n/d"`.
    pub second_value: String,
}

/// Inputs of a witness search or verification: the declared model class and the
/// query's conjunctions, all in internal indices.
pub(super) struct WitnessProblem<'a> {
    pub(super) problem: &'a Problem,
    /// Selected variable set of each population (index 0 is the target).
    pub(super) selected: &'a [u32],
    /// `(population, intervened set)` of every regime of the declared family.
    pub(super) family: &'a [(usize, u32)],
    pub(super) worlds: &'a [WorldIdx],
    /// Numerator conjunction (event and conditioning atoms).
    pub(super) numerator: &'a BTreeMap<(usize, usize), usize>,
    /// Conditioning atoms (empty for an unconditional query).
    pub(super) given: &'a BTreeMap<(usize, usize), usize>,
}

/// Result of a witness search.
pub(super) enum WitnessOutcome {
    /// A verified witness.
    Found(CounterfactualTransportWitness),
    /// No witness within the bounded family; the reason names why.
    NotFound(String),
}

// ---------------------------------------------------------------------------
// Exact rationals (checked i128; overflow is a failure, never a wrap).
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Rat {
    n: i128,
    d: i128,
}

fn gcd(mut a: i128, mut b: i128) -> i128 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

impl Rat {
    const ZERO: Self = Self { n: 0, d: 1 };

    fn new(n: i128, d: i128) -> Option<Self> {
        if d == 0 {
            return None;
        }
        let g = gcd(n, d).max(1);
        let (mut n, mut d) = (n / g, d / g);
        if d < 0 {
            n = n.checked_neg()?;
            d = d.checked_neg()?;
        }
        Some(Self { n, d })
    }
    const fn int(n: i128) -> Self {
        Self { n, d: 1 }
    }
    fn add(self, o: Self) -> Option<Self> {
        let g = gcd(self.d, o.d).max(1);
        let l = (self.d / g).checked_mul(o.d)?;
        let a = self.n.checked_mul(l / self.d)?;
        let b = o.n.checked_mul(l / o.d)?;
        Self::new(a.checked_add(b)?, l)
    }
    fn neg(self) -> Option<Self> {
        Some(Self { n: self.n.checked_neg()?, d: self.d })
    }
    fn sub(self, o: Self) -> Option<Self> {
        self.add(o.neg()?)
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
    const fn is_zero(self) -> bool {
        self.n == 0
    }
    fn text(self) -> String {
        format!("{}/{}", self.n, self.d)
    }
    fn parse(text: &str) -> Option<Self> {
        let (n, d) = text.split_once('/')?;
        if n.is_empty() || d.is_empty() || d.starts_with(['-', '+']) || n.starts_with('+') {
            return None;
        }
        let q = Self::new(n.parse().ok()?, d.parse().ok()?)?;
        // One value, one spelling.
        (q.text() == text).then_some(q)
    }
    /// Strictly inside (0, 1).
    const fn is_strict_probability(self) -> bool {
        self.n > 0 && self.n < self.d
    }
}

// ---------------------------------------------------------------------------
// The finite model: supports, response functions, simulation.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Baseline {
    Uniform,
    Parity,
    Conjunction,
}

impl Baseline {
    const ALL: [Self; 3] = [Self::Uniform, Self::Parity, Self::Conjunction];
    const fn tag(self) -> &'static str {
        match self {
            Self::Uniform => "uniform",
            Self::Parity => "parity",
            Self::Conjunction => "conjunction",
        }
    }
    fn from_tag(tag: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|b| b.tag() == tag)
    }
}

/// The prepared finite model class of one problem.
struct Model<'a> {
    wp: &'a WitnessProblem<'a>,
    n: usize,
    /// `positions[v]`: declared level index at each support position.
    positions: Vec<Vec<usize>>,
    /// `canon[v][declared level]`: support position (position 0 outside).
    canon: Vec<Vec<usize>>,
    /// Rows of each variable's response function (product of parent supports).
    dom: Vec<usize>,
    /// Number of response functions of each variable.
    nfun: Vec<usize>,
    numerator: Network,
    denominator: Option<Network>,
}

fn pow_usize(base: usize, exp: usize) -> Option<usize> {
    let mut acc: usize = 1;
    for _ in 0..exp {
        acc = acc.checked_mul(base)?;
    }
    Some(acc)
}

impl<'a> Model<'a> {
    fn new(wp: &'a WitnessProblem<'a>) -> Result<Self, String> {
        let problem = wp.problem;
        let n = problem.variables.len();
        let mut mentioned: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
        for world in wp.worlds {
            for (&v, &l) in &world.fixed {
                mentioned[v].insert(l);
            }
        }
        for (&(_, v), &l) in wp.numerator.iter().chain(wp.given.iter()) {
            mentioned[v].insert(l);
        }
        let mut positions: Vec<Vec<usize>> = Vec::with_capacity(n);
        let mut canon: Vec<Vec<usize>> = Vec::with_capacity(n);
        for (v, mentioned_v) in mentioned.iter().enumerate().take(n) {
            let declared = problem.levels[v].len();
            if mentioned_v.len() > 2 {
                return Err(format!(
                    "variable {} is read at more than two levels; the witness model class is \
                     two-level",
                    problem.variables[v].raw()
                ));
            }
            let target = declared.min(2);
            let mut support: BTreeSet<usize> = mentioned_v.clone();
            let mut next = 0usize;
            while support.len() < target {
                support.insert(next);
                next += 1;
            }
            let pos: Vec<usize> = support.into_iter().collect();
            let mut map = vec![0usize; declared];
            for (p, &level) in pos.iter().enumerate() {
                map[level] = p;
            }
            positions.push(pos);
            canon.push(map);
        }
        let mut dom = Vec::with_capacity(n);
        let mut nfun = Vec::with_capacity(n);
        for v in 0..n {
            let rows = problem.parents[v]
                .iter()
                .try_fold(1usize, |acc, &p| acc.checked_mul(positions[p].len()))
                .ok_or_else(|| "a response function has too many rows".to_owned())?;
            if rows > MAX_DOMAIN {
                return Err(format!(
                    "variable {} has {rows} parent configurations; the witness model class \
                     is bounded at {MAX_DOMAIN}",
                    problem.variables[v].raw()
                ));
            }
            let count = pow_usize(positions[v].len(), rows)
                .ok_or_else(|| "a response-function space overflows".to_owned())?;
            dom.push(rows);
            nfun.push(count);
        }
        let numerator =
            match build_network(problem, wp.worlds, wp.numerator).map_err(|e| e.to_string())? {
                Built::Zero => {
                    return Err("the query's numerator is structurally impossible".to_owned());
                }
                Built::Net(net) => net,
            };
        let denominator = if wp.given.is_empty() {
            None
        } else {
            match build_network(problem, wp.worlds, wp.given).map_err(|e| e.to_string())? {
                Built::Zero => return Err("the conditioning event is impossible".to_owned()),
                Built::Net(net) => Some(net),
            }
        };
        Ok(Self { wp, n, positions, canon, dom, nfun, numerator, denominator })
    }

    fn support_size(&self, v: usize) -> usize {
        self.positions[v].len()
    }

    /// Output of response function `f` of `v` at parent-configuration row `row`.
    fn digit(&self, v: usize, f: usize, row: usize) -> usize {
        let c = self.support_size(v);
        if c == 1 {
            return 0;
        }
        (f / c.pow(u32::try_from(row).unwrap_or(0))) % c
    }

    /// The deterministic baseline function of `v`.
    fn rule_function(&self, v: usize, baseline: Baseline) -> usize {
        let c = self.support_size(v);
        let parents = &self.wp.problem.parents[v];
        let mut f = 0usize;
        let mut place = 1usize;
        for row in 0..self.dom[v] {
            let mut rem = row;
            let mut sum = 0usize;
            let mut all_one = !parents.is_empty();
            for &p in parents.iter().rev() {
                let s = self.support_size(p);
                let pos = rem % s;
                rem /= s;
                sum += pos;
                if pos == 0 {
                    all_one = false;
                }
            }
            let out = match baseline {
                Baseline::Parity => sum % c,
                Baseline::Conjunction => usize::from(all_one).min(c - 1),
                Baseline::Uniform => 0,
            };
            f += out * place;
            place = place.saturating_mul(c);
        }
        f
    }

    /// Solve every variable for one response-function assignment in one
    /// population under `fixed` (support positions of intervened variables).
    fn run(&self, funcs: &[usize], population: usize, fixed: &[Option<usize>], out: &mut [usize]) {
        let selected = self.wp.selected[population];
        for &v in &self.wp.problem.topo {
            out[v] = if let Some(pos) = fixed[v] {
                pos
            } else if (selected >> v) & 1 == 1 {
                0
            } else {
                let mut row = 0usize;
                for &p in &self.wp.problem.parents[v] {
                    row = row * self.support_size(p) + out[p];
                }
                self.digit(v, funcs[v], row)
            };
        }
    }

    /// Whether the conjunction of `net` holds under `funcs` in the target.
    fn holds(&self, net: &Network, funcs: &[usize], values: &mut Vec<usize>) -> bool {
        values.clear();
        for node in &net.nodes {
            let mut row = 0usize;
            for &(parent, input) in &node.inputs {
                let pos = match input {
                    Input::Const(level) => self.canon[parent][level],
                    Input::Node(j) => values[j],
                };
                row = row * self.support_size(parent) + pos;
            }
            let value = self.digit(node.variable, funcs[node.variable], row);
            values.push(value);
        }
        net.nodes.iter().zip(values.iter()).all(|(node, &value)| match node.value {
            Some(level) => self.positions[node.variable].get(value) == Some(&level),
            None => true,
        })
    }

    /// Family layout: per regime its free variables, intervened variables and
    /// the row offset of its first cell.
    fn layout(&self) -> Result<Layout, String> {
        let full = self.wp.problem.full_mask();
        let mut entries = Vec::with_capacity(self.wp.family.len());
        let mut offset = 0usize;
        let mut simulations: u64 = 0;
        for &(population, intervened) in self.wp.family {
            let zvars: Vec<usize> = (0..self.n).filter(|v| (intervened >> *v) & 1 == 1).collect();
            let fvars: Vec<usize> =
                (0..self.n).filter(|v| ((full & !intervened) >> *v) & 1 == 1).collect();
            let zcount: usize = zvars.iter().map(|&v| self.support_size(v)).product();
            let ccount: usize = fvars.iter().map(|&v| self.support_size(v)).product();
            let cells = zcount
                .checked_mul(ccount)
                .ok_or_else(|| "the regime-law cell count overflows".to_owned())?;
            entries.push(LayoutEntry { population, zvars, fvars, zcount, ccount, offset });
            offset = offset
                .checked_add(cells)
                .filter(|&o| o <= MAX_ROWS)
                .ok_or_else(|| "the declared family has too many regime-law cells".to_owned())?;
            simulations = simulations.saturating_add(zcount as u64);
        }
        Ok(Layout { entries, rows: offset, simulations })
    }

    /// Row index of every `(regime, level combination, cell)` reached under
    /// `funcs`, calling `visit(row)`.
    fn for_each_cell(
        &self,
        layout: &Layout,
        funcs: &[usize],
        scratch: &mut Scratch,
        visit: &mut dyn FnMut(usize),
    ) {
        for entry in &layout.entries {
            for z in 0..entry.zcount {
                scratch.fixed.iter_mut().for_each(|f| *f = None);
                let mut rem = z;
                for &v in entry.zvars.iter().rev() {
                    let s = self.support_size(v);
                    scratch.fixed[v] = Some(rem % s);
                    rem /= s;
                }
                self.run(funcs, entry.population, &scratch.fixed, &mut scratch.out);
                let mut cell = 0usize;
                for &v in &entry.fvars {
                    cell = cell * self.support_size(v) + scratch.out[v];
                }
                visit(entry.offset + z * entry.ccount + cell);
            }
        }
    }
}

struct LayoutEntry {
    population: usize,
    zvars: Vec<usize>,
    fvars: Vec<usize>,
    zcount: usize,
    ccount: usize,
    offset: usize,
}

struct Layout {
    entries: Vec<LayoutEntry>,
    rows: usize,
    simulations: u64,
}

struct Scratch {
    fixed: Vec<Option<usize>>,
    out: Vec<usize>,
}

impl Scratch {
    fn new(n: usize) -> Self {
        Self { fixed: vec![None; n], out: vec![0; n] }
    }
}

/// Advance a mixed-radix counter; `false` after the last configuration.
fn advance(digits: &mut [usize], radices: &[usize]) -> bool {
    for i in (0..digits.len()).rev() {
        digits[i] += 1;
        if digits[i] < radices[i] {
            return true;
        }
        digits[i] = 0;
    }
    false
}

fn is_clique(problem: &Problem, vars: &[usize]) -> bool {
    vars.iter()
        .enumerate()
        .all(|(i, &a)| vars[i + 1..].iter().all(|&b| problem.bidirected_adjacent(a, b)))
}

// ---------------------------------------------------------------------------
// Search.
// ---------------------------------------------------------------------------

/// One attempt's integer matrices.
struct Counts {
    /// `rows x states`, entry = number of baseline states producing the cell.
    cells: Vec<Vec<i128>>,
    numerator: Vec<i128>,
    denominator: Vec<i128>,
}

fn collect_counts(
    model: &Model<'_>,
    layout: &Layout,
    clique: &[usize],
    baseline: Baseline,
) -> Option<(Counts, usize, usize)> {
    let n = model.n;
    let clique_radices: Vec<usize> = clique.iter().map(|&v| model.nfun[v]).collect();
    let states: usize = clique_radices.iter().product();
    if !(2..=MAX_CLIQUE_STATES).contains(&states) {
        return None;
    }
    let rest: Vec<usize> = (0..n).filter(|v| !clique.contains(v)).collect();
    let rest_radices: Vec<usize> = match baseline {
        Baseline::Uniform => rest.iter().map(|&v| model.nfun[v]).collect(),
        _ => Vec::new(),
    };
    let rest_states: usize = rest_radices.iter().product();
    if rest_states > MAX_BASELINE_STATES {
        return None;
    }
    let work = (states as u64)
        .saturating_mul(rest_states.max(1) as u64)
        .saturating_mul(layout.simulations.max(1));
    if work > MAX_SIMULATIONS {
        return None;
    }
    let mut counts = Counts {
        cells: vec![vec![0i128; states]; layout.rows],
        numerator: vec![0; states],
        denominator: vec![0; states],
    };
    let mut funcs = vec![0usize; n];
    let mut scratch = Scratch::new(n);
    let mut values: Vec<usize> = Vec::new();
    let mut rest_digits = vec![0usize; rest_radices.len()];
    for &v in &rest {
        funcs[v] = model.rule_function(v, baseline);
    }
    loop {
        if baseline == Baseline::Uniform {
            for (k, &v) in rest.iter().enumerate() {
                funcs[v] = rest_digits[k];
            }
        }
        let mut clique_digits = vec![0usize; clique.len()];
        for u in 0..states {
            for (k, &v) in clique.iter().enumerate() {
                funcs[v] = clique_digits[k];
            }
            {
                let cells = &mut counts.cells;
                model.for_each_cell(layout, &funcs, &mut scratch, &mut |row| {
                    cells[row][u] += 1;
                });
            }
            let holds_numerator = model.holds(&model.numerator, &funcs, &mut values);
            let holds_denominator = match &model.denominator {
                Some(net) => model.holds(net, &funcs, &mut values),
                None => true,
            };
            if holds_numerator {
                counts.numerator[u] += 1;
            }
            if holds_denominator {
                counts.denominator[u] += 1;
            }
            advance(&mut clique_digits, &clique_radices);
        }
        if rest_radices.is_empty() || !advance(&mut rest_digits, &rest_radices) {
            break;
        }
    }
    Some((counts, states, rest_states.max(1)))
}

/// Integer null space of `rows` (each of length `cols`), by exact rational
/// reduced row echelon form. `None` on overflow.
fn nullspace(rows: &[Vec<i128>], cols: usize) -> Option<Vec<Vec<i128>>> {
    let mut m: Vec<Vec<Rat>> =
        rows.iter().map(|r| r.iter().map(|&x| Rat::int(x)).collect()).collect();
    let mut pivots: Vec<usize> = Vec::new();
    let mut row = 0usize;
    for col in 0..cols {
        let Some(p) = (row..m.len()).find(|&r| !m[r][col].is_zero()) else {
            continue;
        };
        m.swap(row, p);
        let pivot = m[row][col];
        for c in 0..cols {
            m[row][c] = m[row][c].div(pivot)?;
        }
        for r in 0..m.len() {
            if r != row && !m[r][col].is_zero() {
                let factor = m[r][col];
                for c in 0..cols {
                    let delta = factor.mul(m[row][c])?;
                    m[r][c] = m[r][c].sub(delta)?;
                }
            }
        }
        pivots.push(col);
        row += 1;
        if row == m.len() {
            break;
        }
    }
    let mut basis = Vec::new();
    for free in 0..cols {
        if pivots.contains(&free) {
            continue;
        }
        let mut vector = vec![Rat::ZERO; cols];
        vector[free] = Rat::int(1);
        for (r, &pc) in pivots.iter().enumerate() {
            vector[pc] = m[r][free].neg()?;
        }
        let mut lcm = 1i128;
        for q in &vector {
            let g = gcd(lcm, q.d).max(1);
            lcm = (lcm / g).checked_mul(q.d)?;
        }
        let mut integers = Vec::with_capacity(cols);
        for q in &vector {
            integers.push(q.n.checked_mul(lcm / q.d)?);
        }
        basis.push(integers);
    }
    Some(basis)
}

/// Search for a verified witness; see the module documentation.
#[allow(
    clippy::too_many_lines,
    reason = "one search over the cliques and baselines whose charging, caching and outcome bookkeeping stay together"
)]
pub(super) fn search_witness(
    wp: &WitnessProblem<'_>,
    charge: &mut dyn FnMut(usize, u64) -> Result<(), SearchStop>,
) -> Result<WitnessOutcome, SearchStop> {
    let model = match Model::new(wp) {
        Ok(model) => model,
        Err(reason) => return Ok(WitnessOutcome::NotFound(reason)),
    };
    let layout = match model.layout() {
        Ok(layout) => layout,
        Err(reason) => return Ok(WitnessOutcome::NotFound(reason)),
    };
    let n = model.n;
    // Cliques of the bidirected graph, largest first, then ascending mask.
    let mut cliques: Vec<Vec<usize>> = Vec::new();
    for mask in 1u32..(1u32 << n) {
        let vars: Vec<usize> = (0..n).filter(|v| (mask >> *v) & 1 == 1).collect();
        if is_clique(wp.problem, &vars) {
            cliques.push(vars);
        }
    }
    cliques.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
    let mut reasons: Vec<String> = Vec::new();
    for clique in &cliques {
        let states: usize = clique.iter().map(|&v| model.nfun[v]).product();
        if !(2..=MAX_CLIQUE_STATES).contains(&states) {
            continue;
        }
        for baseline in Baseline::ALL {
            charge(0, (states as u64).saturating_mul(layout.rows as u64).saturating_mul(16))?;
            let Some((counts, m, _)) = collect_counts(&model, &layout, clique, baseline) else {
                continue;
            };
            let sum_n: i128 = counts.numerator.iter().sum();
            let sum_d: i128 = counts.denominator.iter().sum();
            if sum_d == 0 {
                continue;
            }
            // Distinct non-zero rows plus the normalization row.
            let mut distinct: BTreeSet<Vec<i128>> = BTreeSet::new();
            for row in &counts.cells {
                if row.iter().any(|&x| x != 0) {
                    distinct.insert(row.clone());
                }
            }
            distinct.insert(vec![1i128; m]);
            let rows: Vec<Vec<i128>> = distinct.into_iter().collect();
            let Some(basis) = nullspace(&rows, m) else {
                reasons.push("exact elimination overflowed".to_owned());
                continue;
            };
            for delta in basis {
                let dot = |q: &[i128]| -> Option<i128> {
                    q.iter()
                        .zip(&delta)
                        .try_fold(0i128, |acc, (a, b)| acc.checked_add(a.checked_mul(*b)?))
                };
                let (Some(dn), Some(dd)) = (dot(&counts.numerator), dot(&counts.denominator))
                else {
                    continue;
                };
                let (Some(left), Some(right)) = (sum_d.checked_mul(dn), sum_n.checked_mul(dd))
                else {
                    continue;
                };
                if left == right {
                    continue;
                }
                let max_abs = delta.iter().map(|x| x.abs()).max().unwrap_or(0);
                if max_abs == 0 {
                    continue;
                }
                let m_i = i128::try_from(m).unwrap_or(i128::MAX);
                let Some(denominator) = max_abs.checked_mul(2).and_then(|x| x.checked_mul(m_i))
                else {
                    continue;
                };
                let first: Vec<String> = (0..m).map(|_| Rat { n: 1, d: m_i }.text()).collect();
                let second: Option<Vec<String>> = delta
                    .iter()
                    .map(|&x| {
                        let numerator = max_abs.checked_mul(2)?.checked_add(x)?;
                        Rat::new(numerator, denominator).map(Rat::text)
                    })
                    .collect();
                let Some(second) = second else {
                    continue;
                };
                let record = CounterfactualTransportWitness {
                    clique: clique.iter().map(|&v| wp.problem.variables[v].raw()).collect(),
                    baseline: baseline.tag().to_owned(),
                    support: model
                        .positions
                        .iter()
                        .map(|p| p.iter().map(|&l| u32::try_from(l).unwrap_or(u32::MAX)).collect())
                        .collect(),
                    first,
                    second,
                    first_value: String::new(),
                    second_value: String::new(),
                };
                // Fill the recorded values from the verifier itself.
                match evaluate_pair(&model, &layout, &record) {
                    Ok((a, b, _)) if a != b => {
                        let mut record = record;
                        record.first_value = a.text();
                        record.second_value = b.text();
                        if verify_witness(wp, &record).is_ok() {
                            return Ok(WitnessOutcome::Found(record));
                        }
                        reasons.push("a candidate failed exact verification".to_owned());
                    }
                    Ok(_) => reasons.push("a candidate did not separate the query".to_owned()),
                    Err(reason) => reasons.push(reason),
                }
            }
        }
    }
    let reason = if reasons.is_empty() {
        "no clique of the bidirected graph with a bounded response-function space leaves the \
         query undetermined by the declared evidence in the searched model family"
            .to_owned()
    } else {
        reasons.join("; ")
    };
    Ok(WitnessOutcome::NotFound(reason))
}

// ---------------------------------------------------------------------------
// Verification.
// ---------------------------------------------------------------------------

type CellMap = BTreeMap<usize, Rat>;

/// Both models' regime-law cells and query values, by exact enumeration.
#[allow(
    clippy::too_many_lines,
    reason = "a flat sequence of exact enumeration steps over one pair of models, each feeding the next"
)]
fn evaluate_pair(
    model: &Model<'_>,
    layout: &Layout,
    record: &CounterfactualTransportWitness,
) -> Result<(Rat, Rat, usize), String> {
    let wp = model.wp;
    let problem = wp.problem;
    let clique: Vec<usize> = record
        .clique
        .iter()
        .map(|&id| {
            problem
                .variables
                .iter()
                .position(|v| v.raw() == id)
                .ok_or_else(|| "the witness names an unknown variable".to_owned())
        })
        .collect::<Result<_, _>>()?;
    if clique.is_empty() || clique.windows(2).any(|w| w[0] >= w[1]) || !is_clique(problem, &clique)
    {
        return Err("the varied set is not a bidirected clique".to_owned());
    }
    let baseline =
        Baseline::from_tag(&record.baseline).ok_or_else(|| "unknown baseline".to_owned())?;
    let expected_support: Vec<Vec<u32>> = model
        .positions
        .iter()
        .map(|p| p.iter().map(|&l| u32::try_from(l).unwrap_or(u32::MAX)).collect())
        .collect();
    if record.support != expected_support {
        return Err("the recorded supports are not the query's supports".to_owned());
    }
    let radices: Vec<usize> = clique.iter().map(|&v| model.nfun[v]).collect();
    let states: usize = radices.iter().product();
    if !(2..=MAX_CLIQUE_STATES).contains(&states) {
        return Err("the varied clique's state space is outside the bound".to_owned());
    }
    let law = |texts: &[String]| -> Result<Vec<Rat>, String> {
        if texts.len() != states {
            return Err("a law does not cover the clique's states".to_owned());
        }
        let law: Vec<Rat> = texts
            .iter()
            .map(|t| Rat::parse(t).filter(|q| q.is_strict_probability()))
            .collect::<Option<_>>()
            .ok_or_else(|| "a probability is not an exact rational in (0, 1)".to_owned())?;
        let total = law
            .iter()
            .try_fold(Rat::ZERO, |acc, q| acc.add(*q))
            .ok_or_else(|| "overflow".to_owned())?;
        if total != Rat::int(1) {
            return Err("a law does not sum to one".to_owned());
        }
        Ok(law)
    };
    let first = law(&record.first)?;
    let second = law(&record.second)?;
    let n = model.n;
    let rest: Vec<usize> = (0..n).filter(|v| !clique.contains(v)).collect();
    let rest_radices: Vec<usize> = match baseline {
        Baseline::Uniform => rest.iter().map(|&v| model.nfun[v]).collect(),
        _ => Vec::new(),
    };
    let rest_states: usize = rest_radices.iter().product();
    if rest_states > MAX_BASELINE_STATES {
        return Err("the baseline state space is outside the bound".to_owned());
    }
    let work = (states as u64)
        .saturating_mul(rest_states.max(1) as u64)
        .saturating_mul(layout.simulations.max(1));
    if work > MAX_SIMULATIONS {
        return Err("the witness exceeds the simulation bound".to_owned());
    }
    let weight = Rat::new(1, i128::try_from(rest_states.max(1)).map_err(|_| "overflow")?)
        .ok_or_else(|| "overflow".to_owned())?;
    let overflow = || "arithmetic overflow".to_owned();
    let mut cells: [CellMap; 2] = [CellMap::new(), CellMap::new()];
    let mut numerator = [Rat::ZERO; 2];
    let mut denominator = [Rat::ZERO; 2];
    let mut funcs = vec![0usize; n];
    let mut scratch = Scratch::new(n);
    let mut values: Vec<usize> = Vec::new();
    for &v in &rest {
        funcs[v] = model.rule_function(v, baseline);
    }
    let mut rest_digits = vec![0usize; rest_radices.len()];
    loop {
        if baseline == Baseline::Uniform {
            for (k, &v) in rest.iter().enumerate() {
                funcs[v] = rest_digits[k];
            }
        }
        let mut digits = vec![0usize; clique.len()];
        for u in 0..states {
            for (k, &v) in clique.iter().enumerate() {
                funcs[v] = digits[k];
            }
            let mut touched: Vec<usize> = Vec::new();
            model.for_each_cell(layout, &funcs, &mut scratch, &mut |row| touched.push(row));
            let holds_numerator = model.holds(&model.numerator, &funcs, &mut values);
            let holds_denominator = match &model.denominator {
                Some(net) => model.holds(net, &funcs, &mut values),
                None => true,
            };
            for (which, law) in [&first, &second].into_iter().enumerate() {
                let w = law[u].mul(weight).ok_or_else(overflow)?;
                for &row in &touched {
                    let entry = cells[which].entry(row).or_insert(Rat::ZERO);
                    *entry = entry.add(w).ok_or_else(overflow)?;
                }
                if holds_numerator {
                    numerator[which] = numerator[which].add(w).ok_or_else(overflow)?;
                }
                if holds_denominator {
                    denominator[which] = denominator[which].add(w).ok_or_else(overflow)?;
                }
            }
            advance(&mut digits, &radices);
        }
        if rest_radices.is_empty() || !advance(&mut rest_digits, &rest_radices) {
            break;
        }
    }
    if cells[0] != cells[1] {
        return Err("the models disagree on a regime law".to_owned());
    }
    if denominator[0].is_zero() || denominator[1].is_zero() {
        return Err("the conditioning event has no mass in a model".to_owned());
    }
    let a = numerator[0].div(denominator[0]).ok_or_else(overflow)?;
    let b = numerator[1].div(denominator[1]).ok_or_else(overflow)?;
    Ok((a, b, layout.rows))
}

/// Verify a witness by exact enumeration: both laws are strictly positive
/// probability vectors on the clique's response-function tuples, the two models
/// give identical laws for every regime of the declared family, and the query
/// values differ and equal the recorded ones.
pub(super) fn verify_witness(
    wp: &WitnessProblem<'_>,
    record: &CounterfactualTransportWitness,
) -> Result<CounterfactualTransportWitnessCheck, String> {
    let model = Model::new(wp)?;
    let layout = model.layout()?;
    let (a, b, rows) = evaluate_pair(&model, &layout, record)?;
    if a == b {
        return Err("the models agree on the query".to_owned());
    }
    if a.text() != record.first_value || b.text() != record.second_value {
        return Err("a recorded query value is not the model's".to_owned());
    }
    Ok(CounterfactualTransportWitnessCheck {
        cells_compared: rows,
        first_value: record.first_value.clone(),
        second_value: record.second_value.clone(),
    })
}
