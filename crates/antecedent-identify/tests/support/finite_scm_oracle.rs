//! Independent exact oracle for finite structural causal models.
//!
//! This module is a self-contained truth oracle for counterfactual transport
//! tests. It depends on nothing in the workspace (no expression, graph or
//! identification types), so a value asserted against it can only come from
//! enumerating a finite SCM, never from the algorithm under test.
//!
//! Model. Endogenous variables take levels `0..levels`. Exogenous variables are
//! discrete with an explicit probability table. Each endogenous variable is a
//! deterministic [`Mechanism`] of the already-solved endogenous values and the
//! exogenous values. Endogenous variables are registered in a causal order and
//! a mechanism may only read earlier variables. Latent confounding is an
//! exogenous variable read by two or more mechanisms. A population is a
//! [`FiniteScm`]; populations that differ at selection nodes are produced by
//! swapping a mechanism or an exogenous law (`with_mechanism`, `with_exo_law`).
//!
//! Counterfactuals. All worlds of a query share one exogenous state. A
//! [`CfVar`] is an endogenous variable evaluated in the world where a set of
//! interventions holds; the level assigned to an intervened variable may itself
//! be a nested counterfactual (`Y_{x, M_{x'}}`). Probabilities are exact sums
//! over exogenous states. Fixtures use dyadic probabilities, so every `f64`
//! sum and product is exact and equality may be asserted bit-for-bit.
//!
//! Conversion note: [`Table`] and [`EvidenceTable`] are plain variable names,
//! level counts and row-major probabilities (last variable fastest). A test
//! converts them to whatever the code under test consumes with a thin local
//! adapter; nothing here knows those types.
//!
//! Inclusion: declare `mod finite_scm_oracle;` at the test crate root with
//! `#[path = "support/finite_scm_oracle.rs"]`. A fixture module that uses this
//! one reaches it as `crate::finite_scm_oracle`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::needless_range_loop,
    clippy::float_cmp,
    clippy::type_complexity,
    dead_code,
    missing_docs,
    reason = "test oracle: small exact indices and dyadic probabilities; exact bit comparisons are the assertion"
)]

use std::fmt;
use std::rc::Rc;

// ---------------------------------------------------------------------------
// Mechanisms
// ---------------------------------------------------------------------------

/// Where a lookup-table mechanism reads one input from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Input {
    /// An earlier endogenous variable, by index.
    Endo(usize),
    /// An exogenous variable, by index.
    Exo(usize),
}

/// A deterministic structural equation. It receives the endogenous values
/// solved so far (indexed by variable; entries of later variables are
/// meaningless) and the full exogenous state, and returns this variable's level.
#[derive(Clone)]
pub struct Mechanism {
    f: Rc<dyn Fn(&[usize], &[usize]) -> usize>,
}

impl fmt::Debug for Mechanism {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Mechanism(..)")
    }
}

impl Mechanism {
    /// A closure mechanism.
    pub fn new(f: impl Fn(&[usize], &[usize]) -> usize + 'static) -> Self {
        Self { f: Rc::new(f) }
    }

    /// A lookup table over `inputs`. `radices[k]` is the number of levels of
    /// `inputs[k]`; `table` is row-major with the last input fastest.
    pub fn lookup(inputs: Vec<Input>, radices: Vec<usize>, table: Vec<usize>) -> Self {
        assert_eq!(inputs.len(), radices.len(), "one radix per input");
        let expected: usize = radices.iter().product();
        assert_eq!(table.len(), expected, "lookup table size");
        Self::new(move |endo, exo| {
            let mut idx = 0usize;
            for (input, radix) in inputs.iter().zip(&radices) {
                let value = match *input {
                    Input::Endo(i) => endo[i],
                    Input::Exo(k) => exo[k],
                };
                assert!(value < *radix, "lookup input out of range");
                idx = idx * radix + value;
            }
            table[idx]
        })
    }

    pub fn eval(&self, endo: &[usize], exo: &[usize]) -> usize {
        (self.f)(endo, exo)
    }
}

// ---------------------------------------------------------------------------
// Tables
// ---------------------------------------------------------------------------

/// A finite joint law over named discrete variables. `probs` is row-major with
/// the last variable fastest.
#[derive(Clone, Debug, PartialEq)]
pub struct Table {
    pub variables: Vec<String>,
    pub levels: Vec<usize>,
    pub probs: Vec<f64>,
}

impl Table {
    fn empty(variables: Vec<String>, levels: Vec<usize>) -> Self {
        let size: usize = levels.iter().product();
        Self {
            variables,
            levels,
            probs: vec![0.0; size],
        }
    }

    fn position(&self, name: &str) -> usize {
        self.variables
            .iter()
            .position(|v| v == name)
            .unwrap_or_else(|| panic!("table has no variable {name}"))
    }

    /// Mixed-radix decode of a cell index into one level per variable.
    pub fn decode(&self, mut cell: usize) -> Vec<usize> {
        let mut out = vec![0; self.levels.len()];
        for k in (0..self.levels.len()).rev() {
            out[k] = cell % self.levels[k];
            cell /= self.levels[k];
        }
        out
    }

    /// Sum of all cells (1 for a normalized law).
    pub fn total(&self) -> f64 {
        self.probs.iter().sum()
    }

    /// Probability that every `(variable, level)` in `assignment` holds.
    pub fn probability(&self, assignment: &[(&str, usize)]) -> f64 {
        let pos: Vec<(usize, usize)> = assignment
            .iter()
            .map(|(n, l)| (self.position(n), *l))
            .collect();
        (0..self.probs.len())
            .filter(|&c| {
                let d = self.decode(c);
                pos.iter().all(|&(p, l)| d[p] == l)
            })
            .map(|c| self.probs[c])
            .sum()
    }

    /// Marginal law over `keep`, in the order given.
    pub fn marginal(&self, keep: &[&str]) -> Table {
        let pos: Vec<usize> = keep.iter().map(|n| self.position(n)).collect();
        let mut out = Table::empty(
            keep.iter().map(|s| (*s).to_owned()).collect(),
            pos.iter().map(|&p| self.levels[p]).collect(),
        );
        for c in 0..self.probs.len() {
            let d = self.decode(c);
            let mut idx = 0;
            for (k, &p) in pos.iter().enumerate() {
                idx = idx * out.levels[k] + d[p];
            }
            out.probs[idx] += self.probs[c];
        }
        out
    }

    /// `P(target | given)`, or `None` when `given` has probability zero.
    pub fn conditional(&self, target: &[(&str, usize)], given: &[(&str, usize)]) -> Option<f64> {
        let mut joint: Vec<(&str, usize)> = target.to_vec();
        joint.extend_from_slice(given);
        let denominator = self.probability(given);
        if denominator == 0.0 {
            None
        } else {
            Some(self.probability(&joint) / denominator)
        }
    }

    /// Largest absolute cellwise difference. Both tables must have the same
    /// variable set and levels; variable order may differ.
    pub fn max_abs_diff(&self, other: &Table) -> f64 {
        assert_eq!(self.variables.len(), other.variables.len(), "variable sets");
        let map: Vec<usize> = other.variables.iter().map(|n| self.position(n)).collect();
        for (k, &p) in map.iter().enumerate() {
            assert_eq!(self.levels[p], other.levels[k], "level counts differ");
        }
        let mut worst: f64 = 0.0;
        for c in 0..self.probs.len() {
            let d = self.decode(c);
            let mut idx = 0;
            for (k, &p) in map.iter().enumerate() {
                idx = idx * other.levels[k] + d[p];
            }
            worst = worst.max((self.probs[c] - other.probs[idx]).abs());
        }
        worst
    }
}

// ---------------------------------------------------------------------------
// Counterfactual terms
// ---------------------------------------------------------------------------

/// The level an intervention assigns: a constant or a nested counterfactual.
#[derive(Clone, Debug, PartialEq)]
pub enum Setting {
    Level(usize),
    Nested(Box<CfVar>),
}

impl Setting {
    pub fn cf(var: CfVar) -> Self {
        Setting::Nested(Box::new(var))
    }
}

/// A counterfactual variable `V_{interventions}` evaluated in a world that
/// shares the exogenous state of every other term of the query.
#[derive(Clone, Debug, PartialEq)]
pub struct CfVar {
    pub var: String,
    pub interventions: Vec<(String, Setting)>,
}

impl CfVar {
    /// `V` with no intervention (the factual world).
    pub fn factual(var: &str) -> Self {
        Self {
            var: var.to_owned(),
            interventions: Vec::new(),
        }
    }

    /// `V_{x1=l1, x2=l2, ...}` with constant levels.
    pub fn under(var: &str, interventions: &[(&str, usize)]) -> Self {
        Self {
            var: var.to_owned(),
            interventions: interventions
                .iter()
                .map(|(n, l)| ((*n).to_owned(), Setting::Level(*l)))
                .collect(),
        }
    }

    /// `V_{x1=s1, ...}` where a setting may be a nested counterfactual.
    pub fn nested(var: &str, interventions: Vec<(&str, Setting)>) -> Self {
        Self {
            var: var.to_owned(),
            interventions: interventions
                .into_iter()
                .map(|(n, s)| (n.to_owned(), s))
                .collect(),
        }
    }

    /// A readable label, e.g. `Y_{X=1,M_{X=0}}`.
    pub fn label(&self) -> String {
        if self.interventions.is_empty() {
            return self.var.clone();
        }
        let parts: Vec<String> = self
            .interventions
            .iter()
            .map(|(n, s)| match s {
                Setting::Level(l) => format!("{n}={l}"),
                Setting::Nested(c) => format!("{n}={}", c.label()),
            })
            .collect();
        format!("{}_{{{}}}", self.var, parts.join(","))
    }
}

/// One conjunct: `var` takes `level`.
#[derive(Clone, Debug, PartialEq)]
pub struct Atom {
    pub var: CfVar,
    pub level: usize,
}

impl Atom {
    pub fn new(var: CfVar, level: usize) -> Self {
        Self { var, level }
    }
    /// Factual atom `V = level`.
    pub fn factual(var: &str, level: usize) -> Self {
        Self::new(CfVar::factual(var), level)
    }
}

/// Joint mass of an event and of its conditioning evidence.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConditionalMass {
    /// `P(event, evidence)`.
    pub joint: f64,
    /// `P(evidence)`.
    pub evidence: f64,
}

impl ConditionalMass {
    /// `P(event | evidence)`, or `None` when the evidence has probability zero.
    pub fn value(&self) -> Option<f64> {
        if self.evidence == 0.0 {
            None
        } else {
            Some(self.joint / self.evidence)
        }
    }
}

// ---------------------------------------------------------------------------
// The SCM
// ---------------------------------------------------------------------------

/// A finite structural causal model.
#[derive(Clone, Debug, Default)]
pub struct FiniteScm {
    pub endo_names: Vec<String>,
    pub levels: Vec<usize>,
    pub mechanisms: Vec<Mechanism>,
    pub exo_names: Vec<String>,
    pub exo_laws: Vec<Vec<f64>>,
}

/// Structure read off the mechanisms by exhaustive functional dependence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InducedStructure {
    /// `(parent, child)` names, sorted.
    pub directed: Vec<(String, String)>,
    /// Pairs that read a common exogenous variable (`a < b` by name), sorted.
    pub bidirected: Vec<(String, String)>,
}

/// Where two populations of one model differ.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectionDiscrepancy {
    /// Endogenous variables whose mechanism functions differ.
    pub mechanism_vars: Vec<String>,
    /// Exogenous variables whose law differs.
    pub exo_laws: Vec<String>,
    /// Union of `mechanism_vars` and every endogenous variable that reads a
    /// differing exogenous variable: the nodes a selection diagram must flag.
    pub flagged: Vec<String>,
}

fn all_assignments(levels: &[usize]) -> Vec<Vec<usize>> {
    let mut out = vec![Vec::new()];
    for &l in levels {
        let mut next = Vec::with_capacity(out.len() * l);
        for prefix in &out {
            for v in 0..l {
                let mut p = prefix.clone();
                p.push(v);
                next.push(p);
            }
        }
        out = next;
    }
    out
}

fn check_unique(regime: &[(usize, usize)]) {
    for (i, a) in regime.iter().enumerate() {
        for b in &regime[i + 1..] {
            assert_ne!(a.0, b.0, "a variable is intervened on twice in one world");
        }
    }
}

impl FiniteScm {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an exogenous variable with law `law` (levels = `law.len()`).
    #[must_use]
    pub fn exo(mut self, name: &str, law: Vec<f64>) -> Self {
        assert!(!self.exo_names.iter().any(|n| n == name), "duplicate exo");
        assert!(law.iter().all(|&p| p >= 0.0), "negative probability");
        let total: f64 = law.iter().sum();
        assert!((total - 1.0).abs() <= 1e-15, "exogenous law must sum to 1");
        self.exo_names.push(name.to_owned());
        self.exo_laws.push(law);
        self
    }

    /// Append an endogenous variable. Variables are registered in causal order.
    #[must_use]
    pub fn endo(mut self, name: &str, levels: usize, mechanism: Mechanism) -> Self {
        assert!(levels >= 1);
        assert!(!self.endo_names.iter().any(|n| n == name), "duplicate endo");
        self.endo_names.push(name.to_owned());
        self.levels.push(levels);
        self.mechanisms.push(mechanism);
        self
    }

    pub fn n_endo(&self) -> usize {
        self.endo_names.len()
    }

    pub fn endo_index(&self, name: &str) -> usize {
        self.endo_names
            .iter()
            .position(|n| n == name)
            .unwrap_or_else(|| panic!("no endogenous variable {name}"))
    }

    pub fn exo_index(&self, name: &str) -> usize {
        self.exo_names
            .iter()
            .position(|n| n == name)
            .unwrap_or_else(|| panic!("no exogenous variable {name}"))
    }

    /// A copy with the mechanism of `name` replaced (a selection node on `name`).
    #[must_use]
    pub fn with_mechanism(&self, name: &str, mechanism: Mechanism) -> Self {
        let mut out = self.clone();
        let i = out.endo_index(name);
        out.mechanisms[i] = mechanism;
        out
    }

    /// A copy with the law of exogenous `name` replaced.
    #[must_use]
    pub fn with_exo_law(&self, name: &str, law: Vec<f64>) -> Self {
        let mut out = self.clone();
        let k = out.exo_index(name);
        assert_eq!(law.len(), out.exo_laws[k].len(), "level count must match");
        let total: f64 = law.iter().sum();
        assert!((total - 1.0).abs() <= 1e-15, "exogenous law must sum to 1");
        out.exo_laws[k] = law;
        out
    }

    /// A copy in which `outcome` reads the *natural* value of `treatment`
    /// (recomputed from `treatment`'s own mechanism) instead of the possibly
    /// intervened one. Observational and every intervention law that does not
    /// intervene on `outcome` are unchanged except that `do(treatment)` no
    /// longer reaches `outcome` through the direct equation; it is the standard
    /// construction of an observationally equivalent model with a different
    /// interventional effect. `treatment` must precede `outcome`.
    #[must_use]
    pub fn blind_to_natural(&self, outcome: &str, treatment: &str) -> Self {
        let y = self.endo_index(outcome);
        let t = self.endo_index(treatment);
        assert!(t < y, "treatment must precede outcome in the causal order");
        let mt = self.mechanisms[t].clone();
        let my = self.mechanisms[y].clone();
        self.with_mechanism(
            outcome,
            Mechanism::new(move |v, u| {
                let mut w = v.to_vec();
                w[t] = mt.eval(v, u);
                my.eval(&w, u)
            }),
        )
    }

    fn resolve_do(&self, regime: &[(&str, usize)]) -> Vec<(usize, usize)> {
        let out: Vec<(usize, usize)> = regime
            .iter()
            .map(|(n, l)| {
                let i = self.endo_index(n);
                assert!(*l < self.levels[i], "intervention level out of range");
                (i, *l)
            })
            .collect();
        check_unique(&out);
        out
    }

    /// Solve every endogenous variable in causal order for one exogenous state
    /// under a set of constant interventions.
    pub fn solve(&self, exo: &[usize], regime: &[(usize, usize)]) -> Vec<usize> {
        let mut v = vec![0usize; self.n_endo()];
        for i in 0..self.n_endo() {
            v[i] = match regime.iter().find(|(j, _)| *j == i) {
                Some(&(_, l)) => l,
                None => self.mechanisms[i].eval(&v, exo),
            };
            assert!(v[i] < self.levels[i], "mechanism returned an out-of-range level");
        }
        v
    }

    /// Visit every exogenous state of positive probability with its probability.
    pub fn for_each_exo_state(&self, f: &mut dyn FnMut(&[usize], f64)) {
        let k = self.exo_laws.len();
        let mut state = vec![0usize; k];
        loop {
            let p: f64 = state
                .iter()
                .enumerate()
                .map(|(i, &l)| self.exo_laws[i][l])
                .product();
            if p > 0.0 {
                f(&state, p);
            }
            let mut i = k;
            loop {
                if i == 0 {
                    return;
                }
                i -= 1;
                state[i] += 1;
                if state[i] < self.exo_laws[i].len() {
                    break;
                }
                state[i] = 0;
            }
        }
    }

    // ----- interventional laws ------------------------------------------------

    /// Joint law `P(observed | do(regime))`. An intervened variable that is
    /// listed in `observed` appears as a point mass at its set level. An empty
    /// regime is the observational law.
    pub fn law(&self, regime: &[(&str, usize)], observed: &[&str]) -> Table {
        let doi = self.resolve_do(regime);
        let oi: Vec<usize> = observed.iter().map(|n| self.endo_index(n)).collect();
        let mut table = Table::empty(
            observed.iter().map(|s| (*s).to_owned()).collect(),
            oi.iter().map(|&i| self.levels[i]).collect(),
        );
        let levels = table.levels.clone();
        self.for_each_exo_state(&mut |u, p| {
            let v = self.solve(u, &doi);
            let mut idx = 0;
            for (k, &i) in oi.iter().enumerate() {
                idx = idx * levels[k] + v[i];
            }
            table.probs[idx] += p;
        });
        table
    }

    fn all_names(&self) -> Vec<&str> {
        self.endo_names.iter().map(String::as_str).collect()
    }

    /// Observational joint law of every endogenous variable.
    pub fn observational(&self) -> Table {
        self.law(&[], &self.all_names())
    }

    /// Joint law of every endogenous variable under `do(regime)`.
    pub fn interventional(&self, regime: &[(&str, usize)]) -> Table {
        self.law(regime, &self.all_names())
    }

    // ----- counterfactuals ----------------------------------------------------

    fn eval_cf(&self, c: &CfVar, exo: &[usize]) -> usize {
        let doi: Vec<(usize, usize)> = c
            .interventions
            .iter()
            .map(|(n, s)| {
                let i = self.endo_index(n);
                let l = match s {
                    Setting::Level(l) => *l,
                    Setting::Nested(inner) => self.eval_cf(inner, exo),
                };
                assert!(l < self.levels[i], "counterfactual level out of range");
                (i, l)
            })
            .collect();
        check_unique(&doi);
        self.solve(exo, &doi)[self.endo_index(&c.var)]
    }

    /// `P(all atoms hold)` over a single shared exogenous state.
    pub fn probability(&self, atoms: &[Atom]) -> f64 {
        let mut mass = 0.0;
        self.for_each_exo_state(&mut |u, p| {
            if atoms.iter().all(|a| self.eval_cf(&a.var, u) == a.level) {
                mass += p;
            }
        });
        mass
    }

    /// `P(event | evidence)` as exact masses; evidence may itself be
    /// counterfactual. `value()` is `None` for probability-zero evidence.
    pub fn conditional(&self, event: &[Atom], evidence: &[Atom]) -> ConditionalMass {
        let mut joint = 0.0;
        let mut ev = 0.0;
        self.for_each_exo_state(&mut |u, p| {
            if evidence.iter().all(|a| self.eval_cf(&a.var, u) == a.level) {
                ev += p;
                if event.iter().all(|a| self.eval_cf(&a.var, u) == a.level) {
                    joint += p;
                }
            }
        });
        ConditionalMass { joint, evidence: ev }
    }

    /// Joint law of a list of counterfactual variables (cross-world joint).
    pub fn counterfactual_table(&self, vars: &[CfVar]) -> Table {
        let idx: Vec<usize> = vars.iter().map(|v| self.endo_index(&v.var)).collect();
        let mut table = Table::empty(
            vars.iter().map(CfVar::label).collect(),
            idx.iter().map(|&i| self.levels[i]).collect(),
        );
        let levels = table.levels.clone();
        self.for_each_exo_state(&mut |u, p| {
            let mut cell = 0;
            for (k, v) in vars.iter().enumerate() {
                cell = cell * levels[k] + self.eval_cf(v, u);
            }
            table.probs[cell] += p;
        });
        table
    }

    // ----- structure ------------------------------------------------------------

    fn depends_on_endo(&self, i: usize, j: usize) -> bool {
        debug_assert!(j < i);
        let exo_assign = all_assignments(&self.exo_laws.iter().map(Vec::len).collect::<Vec<_>>());
        for prefix in all_assignments(&self.levels[..i]) {
            let mut v = prefix.clone();
            v.resize(self.n_endo(), 0);
            for e in &exo_assign {
                let base = self.mechanisms[i].eval(&v, e);
                for l in 0..self.levels[j] {
                    if l == prefix[j] {
                        continue;
                    }
                    let mut w = v.clone();
                    w[j] = l;
                    if self.mechanisms[i].eval(&w, e) != base {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn depends_on_exo(&self, i: usize, k: usize) -> bool {
        let others: Vec<usize> = self.exo_laws.iter().map(Vec::len).collect();
        let exo_assign = all_assignments(&others);
        for prefix in all_assignments(&self.levels[..i]) {
            let mut v = prefix.clone();
            v.resize(self.n_endo(), 0);
            for e in &exo_assign {
                let base = self.mechanisms[i].eval(&v, e);
                for l in 0..others[k] {
                    if l == e[k] {
                        continue;
                    }
                    let mut f = e.clone();
                    f[k] = l;
                    if self.mechanisms[i].eval(&v, &f) != base {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Exogenous variables a given endogenous mechanism actually reads.
    pub fn exo_readers(&self, exo: &str) -> Vec<String> {
        let k = self.exo_index(exo);
        (0..self.n_endo())
            .filter(|&i| self.depends_on_exo(i, k))
            .map(|i| self.endo_names[i].clone())
            .collect()
    }

    /// Directed and bidirected structure by exhaustive functional dependence.
    pub fn induced_structure(&self) -> InducedStructure {
        let mut directed = Vec::new();
        for i in 0..self.n_endo() {
            for j in 0..i {
                if self.depends_on_endo(i, j) {
                    directed.push((self.endo_names[j].clone(), self.endo_names[i].clone()));
                }
            }
        }
        let mut bidirected = Vec::new();
        for k in 0..self.exo_laws.len() {
            let readers: Vec<usize> = (0..self.n_endo())
                .filter(|&i| self.depends_on_exo(i, k))
                .collect();
            for (a, &i) in readers.iter().enumerate() {
                for &j in &readers[a + 1..] {
                    let (x, y) = (self.endo_names[i].clone(), self.endo_names[j].clone());
                    let pair = if x <= y { (x, y) } else { (y, x) };
                    bidirected.push(pair);
                }
            }
        }
        directed.sort();
        directed.dedup();
        bidirected.sort();
        bidirected.dedup();
        InducedStructure {
            directed,
            bidirected,
        }
    }
}

/// Where populations `a` and `b` (same variables and levels) differ. A valid
/// selection diagram flags a superset of `flagged`.
pub fn selection_discrepancy(a: &FiniteScm, b: &FiniteScm) -> SelectionDiscrepancy {
    assert_eq!(a.endo_names, b.endo_names, "populations must share endogenous variables");
    assert_eq!(a.levels, b.levels, "populations must share levels");
    assert_eq!(a.exo_names, b.exo_names, "populations must share exogenous variables");
    let exo_dims: Vec<usize> = a.exo_laws.iter().map(Vec::len).collect();
    for (la, lb) in a.exo_laws.iter().zip(&b.exo_laws) {
        assert_eq!(la.len(), lb.len(), "exogenous level counts must match");
    }
    let exo_assign = all_assignments(&exo_dims);
    let mut mechanism_vars = Vec::new();
    for i in 0..a.n_endo() {
        let mut differs = false;
        'search: for prefix in all_assignments(&a.levels[..i]) {
            let mut v = prefix;
            v.resize(a.n_endo(), 0);
            for e in &exo_assign {
                if a.mechanisms[i].eval(&v, e) != b.mechanisms[i].eval(&v, e) {
                    differs = true;
                    break 'search;
                }
            }
        }
        if differs {
            mechanism_vars.push(a.endo_names[i].clone());
        }
    }
    let mut exo_laws = Vec::new();
    let mut flagged = mechanism_vars.clone();
    for k in 0..a.exo_laws.len() {
        let same = a.exo_laws[k]
            .iter()
            .zip(&b.exo_laws[k])
            .all(|(x, y)| (x - y).abs() <= 1e-15);
        if !same {
            exo_laws.push(a.exo_names[k].clone());
            for reader in a.exo_readers(&a.exo_names[k]) {
                if !flagged.contains(&reader) {
                    flagged.push(reader);
                }
            }
        }
    }
    flagged.sort();
    SelectionDiscrepancy {
        mechanism_vars,
        exo_laws,
        flagged,
    }
}

/// Every intervention regime over the endogenous variables of `scm`: each
/// subset of variables crossed with each level combination (the empty regime is
/// the observational law).
pub fn all_regimes(scm: &FiniteScm) -> Vec<Vec<(String, usize)>> {
    let n = scm.n_endo();
    let mut out = Vec::new();
    for mask in 0..(1usize << n) {
        let members: Vec<usize> = (0..n).filter(|i| mask >> i & 1 == 1).collect();
        let dims: Vec<usize> = members.iter().map(|&i| scm.levels[i]).collect();
        for vals in all_assignments(&dims) {
            out.push(
                members
                    .iter()
                    .zip(vals)
                    .map(|(&i, l)| (scm.endo_names[i].clone(), l))
                    .collect(),
            );
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Populations, evidence and witnesses
// ---------------------------------------------------------------------------

/// A named population (source or target) with its SCM.
#[derive(Clone, Debug)]
pub struct Population {
    pub name: String,
    pub scm: FiniteScm,
}

impl Population {
    pub fn new(name: &str, scm: FiniteScm) -> Self {
        Self {
            name: name.to_owned(),
            scm,
        }
    }
}

pub fn population<'a>(populations: &'a [Population], name: &str) -> &'a FiniteScm {
    &populations
        .iter()
        .find(|p| p.name == name)
        .unwrap_or_else(|| panic!("no population {name}"))
        .scm
}

/// One supplied evidence table: a population, an intervention regime (empty =
/// observational) and the observed variables the table is the joint law of.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvidenceSpec {
    pub population: String,
    pub regime: Vec<(String, usize)>,
    pub variables: Vec<String>,
}

impl EvidenceSpec {
    pub fn new(population: &str, regime: &[(&str, usize)], variables: &[&str]) -> Self {
        Self {
            population: population.to_owned(),
            regime: regime.iter().map(|(n, l)| ((*n).to_owned(), *l)).collect(),
            variables: variables.iter().map(|s| (*s).to_owned()).collect(),
        }
    }
}

/// The exact finite law a spec denotes in a given set of populations.
#[derive(Clone, Debug, PartialEq)]
pub struct EvidenceTable {
    pub population: String,
    pub regime: Vec<(String, usize)>,
    pub table: Table,
}

/// Exact evidence tables for `specs` (population-labelled observational and
/// interventional joints, including marginals over observed subsets).
pub fn build_evidence(populations: &[Population], specs: &[EvidenceSpec]) -> Vec<EvidenceTable> {
    specs
        .iter()
        .map(|s| {
            let scm = population(populations, &s.population);
            let regime: Vec<(&str, usize)> =
                s.regime.iter().map(|(n, l)| (n.as_str(), *l)).collect();
            let vars: Vec<&str> = s.variables.iter().map(String::as_str).collect();
            EvidenceTable {
                population: s.population.clone(),
                regime: s.regime.clone(),
                table: scm.law(&regime, &vars),
            }
        })
        .collect()
}

/// A counterfactual target query in a named population.
#[derive(Clone, Debug, PartialEq)]
pub struct TargetQuery {
    pub population: String,
    pub event: Vec<Atom>,
    /// Factual (or counterfactual) conditioning evidence; empty = unconditional.
    pub evidence: Vec<Atom>,
}

impl TargetQuery {
    pub fn new(population: &str, event: Vec<Atom>, evidence: Vec<Atom>) -> Self {
        Self {
            population: population.to_owned(),
            event,
            evidence,
        }
    }

    /// Exact value, or `None` when the conditioning evidence has probability 0.
    pub fn value(&self, populations: &[Population]) -> Option<f64> {
        population(populations, &self.population)
            .conditional(&self.event, &self.evidence)
            .value()
    }
}

/// Outcome of a verified non-identifiability witness.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WitnessReport {
    /// Largest cellwise disagreement of the two models over all supplied tables.
    pub max_evidence_discrepancy: f64,
    pub value_a: f64,
    pub value_b: f64,
    pub gap: f64,
}

/// Largest cellwise disagreement of two model families on `specs`.
pub fn evidence_discrepancy(a: &[Population], b: &[Population], specs: &[EvidenceSpec]) -> f64 {
    let ta = build_evidence(a, specs);
    let tb = build_evidence(b, specs);
    ta.iter()
        .zip(&tb)
        .map(|(x, y)| x.table.max_abs_diff(&y.table))
        .fold(0.0, f64::max)
}

/// Verify that two model families agree on every supplied evidence table (to
/// `evidence_tol`, typically `1e-15`) yet differ on the target query by more
/// than `gap_tol`. Both target values must be defined.
pub fn verify_non_identifiability_witness(
    a: &[Population],
    b: &[Population],
    specs: &[EvidenceSpec],
    target: &TargetQuery,
    evidence_tol: f64,
    gap_tol: f64,
) -> Result<WitnessReport, String> {
    let discrepancy = evidence_discrepancy(a, b, specs);
    if discrepancy > evidence_tol {
        return Err(format!(
            "models disagree on supplied evidence by {discrepancy:e} > {evidence_tol:e}"
        ));
    }
    let value_a = target
        .value(a)
        .ok_or_else(|| "target query undefined in model A".to_owned())?;
    let value_b = target
        .value(b)
        .ok_or_else(|| "target query undefined in model B".to_owned())?;
    let gap = (value_a - value_b).abs();
    if gap <= gap_tol {
        return Err(format!(
            "models agree on the target query: {value_a} vs {value_b} (gap {gap:e})"
        ));
    }
    Ok(WitnessReport {
        max_evidence_discrepancy: discrepancy,
        value_a,
        value_b,
        gap,
    })
}

/// A Bernoulli law `[1 - p1, p1]`.
pub fn bern(p1: f64) -> Vec<f64> {
    vec![1.0 - p1, p1]
}
