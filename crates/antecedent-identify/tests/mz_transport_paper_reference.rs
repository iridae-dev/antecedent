//! Paper-versus-code departures of the multi-source limited-experiment search
//! (`TR^mz`, R-443 Fig. 3), pinned against an independent literal reference.
//!
//! The reference below re-implements the paper's recursion on bitmask graphs with
//! its own path-enumerating m-separation; it shares no code with the search it
//! checks and decides success or failure only (never a formula). It covers three
//! of the departures documented on the X1 record:
//!
//! * repeated exchange in one branch (Fig. 3 line 10 fires only with `I = ∅`;
//!   the search may also exchange the active source's remaining controllables),
//! * a paper FAIL reached after an exchange that the search does not certify
//!   (`search_incomplete`, never an obstruction),
//! * the line-10 separation test in the graph with edges into `X` removed.
//!
//! Finding pinned by the differential test: the repeated-exchange extension is
//! structurally unreachable. At any line-10 state every non-treatment vertex is
//! an ancestor of `Y` in `D_Xbar` (line 3 is at its fixed point), so `V \ X` is one
//! c-component `C0` whose directed paths to `Y` stay inside `C0`. An exchange
//! removes only treatment vertices, so afterwards `V \ X` is still `C0`, lines 3
//! and 4 cannot fire again (line 8 only narrows to a c-component containing `C0`),
//! and a controllable of the active source that was not in `X` at the exchange can
//! never enter `X` later. The search therefore agrees with the literal
//! one-exchange reference on every fixture, and the cumulative joint regime
//! `do(Z_cum)` is always the single joint `do(Z_i ∩ X)` of one exchange.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::cell::Cell;
use std::sync::Arc;

use antecedent_core::{
    DependenceGroup, DistributionAvailability, EvidenceCatalog, EvidenceKind, EvidenceRegime,
    ExecutionContext, InterventionAssignment, RegimeBinding, RegimeId, RegimeKind, SamplingDesign,
    Value, VariableId,
};
use antecedent_graph::{Admg, DenseNodeId};
use antecedent_identify::{
    MZ_TRANSPORT_DEFAULT_LIMITS, MzTransportDecision, MzTransportQuery, MzTransportRoute,
    SidLimits, ZTransportSourceSpec, decide_mz_transport, verify_mz_transport_obstruction,
};

fn v(i: usize) -> VariableId {
    VariableId::from_raw(u32::try_from(i).unwrap())
}

// ---------------------------------------------------------------------------
// The literal reference.
// ---------------------------------------------------------------------------

/// A selection-diagram family over `n <= 8` nodes; node sets are bitmasks.
#[derive(Clone, Debug)]
struct Model {
    n: usize,
    directed: Vec<(usize, usize)>,
    bidirected: Vec<(usize, usize)>,
}

#[derive(Clone, Debug)]
struct Source {
    name: &'static str,
    /// Controllable set `Z_i` as a bitmask.
    controllable: u32,
    /// Nodes the selection node `S_i` points into, as a bitmask.
    targets: u32,
}

fn has(mask: u32, i: usize) -> bool {
    (mask >> i) & 1 == 1
}

fn bits(mask: u32) -> impl Iterator<Item = usize> {
    (0..32).filter(move |&i| has(mask, i))
}

impl Model {
    fn all(&self) -> u32 {
        (1 << self.n) - 1
    }

    /// Ancestors of `of` (itself included) through directed edges inside `within`,
    /// ignoring edges into `cut`.
    fn ancestors(&self, within: u32, of: u32, cut: u32) -> u32 {
        let mut set = of & within;
        loop {
            let mut grown = set;
            for &(a, b) in &self.directed {
                let inside = has(within, a) && has(within, b);
                if inside && !has(cut, b) && has(set, b) {
                    grown |= 1 << a;
                }
            }
            if grown == set {
                return set;
            }
            set = grown;
        }
    }

    /// Bidirected-connected components of the subgraph induced on `set`.
    fn districts(&self, set: u32) -> Vec<u32> {
        let mut left = set;
        let mut out = Vec::new();
        while left != 0 {
            let mut comp = left & left.wrapping_neg();
            loop {
                let mut grown = comp;
                for &(a, b) in &self.bidirected {
                    if has(set, a) && has(set, b) {
                        if has(comp, a) {
                            grown |= 1 << b;
                        }
                        if has(comp, b) {
                            grown |= 1 << a;
                        }
                    }
                }
                if grown == comp {
                    break;
                }
                comp = grown;
            }
            out.push(comp);
            left &= !comp;
        }
        out
    }

    /// `(S ⊥ Y | X)` on `v` by explicit path enumeration, for a selection node
    /// pointing into `targets` (each target alone: the paper has one selection
    /// node per mismatched mechanism). With `mutilate` the graph is `D_Xbar`:
    /// no arrowhead enters `X`, so edges into `X`, bidirected edges at `X` and
    /// selection edges into `X` are dropped.
    fn selection_separated(&self, v: u32, x: u32, y: u32, targets: u32, mutilate: bool) -> bool {
        let cut = if mutilate { x } else { 0 };
        let directed: Vec<(usize, usize)> = self
            .directed
            .iter()
            .copied()
            .filter(|&(a, b)| has(v, a) && has(v, b) && !has(cut, b))
            .collect();
        let bidirected: Vec<(usize, usize)> = self
            .bidirected
            .iter()
            .copied()
            .filter(|&(a, b)| has(v, a) && has(v, b) && !has(cut, a) && !has(cut, b))
            .collect();
        // Colliders open only inside the ancestors of the conditioning set.
        let mut an_x = x;
        loop {
            let mut grown = an_x;
            for &(a, b) in &directed {
                if has(an_x, b) {
                    grown |= 1 << a;
                }
            }
            if grown == an_x {
                break;
            }
            an_x = grown;
        }
        let selection = self.n;
        for t in bits(targets & v & !cut) {
            // (from, to, arrowhead at from, arrowhead at to)
            let mut edges = directed.iter().map(|&(a, b)| (a, b, false, true)).collect::<Vec<_>>();
            edges.extend(bidirected.iter().map(|&(a, b)| (a, b, true, true)));
            edges.push((selection, t, false, true));
            for goal in bits(y) {
                let mut visited = 1u32 << selection;
                let walk = Walk { edges: &edges, goal, cond: x, an_cond: an_x };
                if walk.connects(selection, None, &mut visited) {
                    return false;
                }
            }
        }
        true
    }
}

/// An m-connecting path search: every non-collider outside the conditioning set,
/// every collider inside its ancestors.
struct Walk<'a> {
    edges: &'a [(usize, usize, bool, bool)],
    goal: usize,
    cond: u32,
    an_cond: u32,
}

impl Walk<'_> {
    fn connects(&self, at: usize, head_into_at: Option<bool>, visited: &mut u32) -> bool {
        if at == self.goal {
            return true;
        }
        for &(a, b, head_a, head_b) in self.edges {
            // Orient the edge instance away from `at`.
            let (next, head_at_self, head_at_next) = if a == at {
                (b, head_a, head_b)
            } else if b == at {
                (a, head_b, head_a)
            } else {
                continue;
            };
            if has(*visited, next) {
                continue;
            }
            if let Some(into) = head_into_at {
                let open =
                    if into && head_at_self { has(self.an_cond, at) } else { !has(self.cond, at) };
                if !open {
                    continue;
                }
            }
            *visited |= 1 << next;
            let found = self.connects(next, Some(head_at_next), visited);
            *visited &= !(1 << next);
            if found {
                return true;
            }
        }
        false
    }
}

/// How the reference treats an exchange once one has happened in the branch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Exchanges {
    /// Fig. 3 line 10 exactly: it fires only when `I = ∅`, so one exchange per branch.
    OnePerBranch,
    /// The search's extension: the active source may exchange its remaining
    /// controllables again; no other source may.
    ActiveSourceAgain,
}

/// The source of the branch's last exchange and its controllables not yet exchanged.
#[derive(Clone, Copy, Debug)]
struct Active {
    source: usize,
    remaining: u32,
}

struct Reference<'a> {
    model: &'a Model,
    sources: &'a [Source],
    exchanges: Exchanges,
    /// Times a second exchange by an active source was eligible.
    repeat_eligible: Cell<usize>,
}

impl<'a> Reference<'a> {
    fn new(model: &'a Model, sources: &'a [Source], exchanges: Exchanges) -> Self {
        Self { model, sources, exchanges, repeat_eligible: Cell::new(0) }
    }

    /// `TR^mz(y, x, V)` succeeds? `active` is `I ≠ ∅` when present.
    fn succeeds(&self, y: u32, x: u32, v: u32, active: Option<Active>) -> bool {
        let m = self.model;
        // Line 1.
        if x == 0 {
            return true;
        }
        // Line 2.
        let an = m.ancestors(v, y, 0);
        if an != v {
            return self.succeeds(y, x & an, an, active);
        }
        // Line 3.
        let w = v & !x & !m.ancestors(v, y, x);
        if w != 0 {
            return self.succeeds(y, x | w, v, active);
        }
        // Line 4.
        let districts = m.districts(v & !x);
        if districts.len() > 1 {
            return districts.iter().all(|&c| self.succeeds(c, v & !c, v, active));
        }
        let c0 = districts[0];
        let containing = m.districts(v);
        // Lines 5-8.
        if containing.len() > 1 {
            if containing.contains(&c0) {
                return true;
            }
            let larger = *containing.iter().find(|&&d| d & c0 == c0).unwrap();
            return self.succeeds(y, x & larger, larger, active);
        }
        // Line 10: the candidate exchanges.
        let candidates: Vec<(usize, u32)> = match (active, self.exchanges) {
            (None, _) => {
                self.sources.iter().enumerate().map(|(i, s)| (i, s.controllable)).collect()
            }
            (Some(_), Exchanges::OnePerBranch) => Vec::new(),
            (Some(a), Exchanges::ActiveSourceAgain) => vec![(a.source, a.remaining)],
        };
        for (i, controllable) in candidates {
            let activated = controllable & x;
            if activated != 0 && m.selection_separated(v, x, y, self.sources[i].targets, true) {
                if active.is_some() {
                    self.repeat_eligible.set(self.repeat_eligible.get() + 1);
                }
                let next = Active { source: i, remaining: controllable & !activated };
                if self.succeeds(y, x & !activated, v & !activated, Some(next)) {
                    return true;
                }
            }
        }
        // Line 11: FAIL.
        false
    }

    fn identifies(&self, y: u32, x: u32) -> bool {
        self.succeeds(y, x, self.model.all(), None)
    }
}

// ---------------------------------------------------------------------------
// Bridging a model to the search under test.
// ---------------------------------------------------------------------------

fn admg(model: &Model) -> Admg {
    let mut g = Admg::with_variables(u32::try_from(model.n).unwrap());
    let d = |i: usize| DenseNodeId::from_raw(u32::try_from(i).unwrap());
    for &(a, b) in &model.directed {
        g.insert_directed(d(a), d(b)).unwrap();
    }
    for &(a, b) in &model.bidirected {
        g.insert_bidirected(d(a), d(b)).unwrap();
    }
    g
}

fn spec(source: &Source) -> ZTransportSourceSpec {
    let list = |mask: u32| bits(mask).map(v).collect::<Vec<_>>();
    ZTransportSourceSpec {
        population: Arc::from(source.name),
        controllable: list(source.controllable).into(),
        experiment_assignment: list(source.controllable)
            .into_iter()
            .map(|variable| InterventionAssignment { variable, value: Value::f64(0.0) })
            .collect::<Vec<_>>()
            .into(),
        selection_targets: list(source.targets).into(),
    }
}

fn mz_query(y: u32, x: u32, sources: &[Source]) -> MzTransportQuery {
    MzTransportQuery {
        outcomes: bits(y).map(v).collect::<Vec<_>>().into(),
        treatments: bits(x).map(v).collect::<Vec<_>>().into(),
        target: Arc::from("target"),
        sources: sources.iter().map(spec).collect::<Vec<_>>().into(),
    }
}

fn regime(id: u32, n: usize, population: &str, on: u32) -> EvidenceRegime {
    EvidenceRegime::try_new(
        RegimeId::from_raw(id),
        if on == 0 { RegimeKind::Observational } else { RegimeKind::Experimental },
        EvidenceKind::Available,
        bits(on).map(v).collect::<Vec<_>>(),
        [],
        (0..n).filter(|&i| !has(on, i)).map(v).collect::<Vec<_>>(),
        population,
        DistributionAvailability::Joint,
    )
    .unwrap()
}

/// The paper's information family (Def. 2): the target's observational law and,
/// per source, every non-empty subset of its controllable set as a joint regime.
/// Returns the catalog and, per regime id, the `(population, intervened mask)`.
fn family_catalog(n: usize, sources: &[Source]) -> (EvidenceCatalog, Vec<(String, u32)>) {
    let mut regimes = vec![regime(0, n, "target", 0)];
    let mut index = vec![("target".to_owned(), 0)];
    for source in sources {
        for subset in 1..=source.controllable {
            if subset & !source.controllable == 0 {
                let id = u32::try_from(regimes.len()).unwrap();
                regimes.push(regime(id, n, source.name, subset));
                index.push((source.name.to_owned(), subset));
            }
        }
    }
    let bindings = regimes
        .iter()
        .map(|r| RegimeBinding {
            dataset_identity: None,
            regime: r.id,
            snapshot_identity: Arc::from(format!("snapshot-{}", r.id.raw())),
            schema_names: Arc::from([]),
            sampling: SamplingDesign::Independent,
            weights: None,
            dependence: DependenceGroup::IndependentStudies,
        })
        .collect::<Vec<_>>();
    (EvidenceCatalog::try_new([], regimes, bindings, None).unwrap(), index)
}

/// Decide over the full information family; `None` when the query is refused.
fn decide(model: &Model, sources: &[Source], y: u32, x: u32) -> Option<MzTransportDecision> {
    let (catalog, _) = family_catalog(model.n, sources);
    decide_mz_transport(
        &admg(model),
        &mz_query(y, x, sources),
        &catalog,
        MZ_TRANSPORT_DEFAULT_LIMITS,
        &ExecutionContext::for_tests(1),
    )
    .ok()
}

/// A decision that certified a formula (identified, or bound short of supplied
/// evidence, which a complete family never leaves).
fn certified(decision: &MzTransportDecision) -> bool {
    matches!(
        decision,
        MzTransportDecision::Identified { .. }
            | MzTransportDecision::MissingEvidence { derivation: Some(_), .. }
    )
}

fn model(n: usize, directed: &[(usize, usize)], bidirected: &[(usize, usize)]) -> Model {
    Model { n, directed: directed.to_vec(), bidirected: bidirected.to_vec() }
}

fn source(name: &'static str, controllable: &[usize], targets: &[usize]) -> Source {
    let mask = |nodes: &[usize]| nodes.iter().fold(0u32, |acc, i| acc | (1 << i));
    Source { name, controllable: mask(controllable), targets: mask(targets) }
}

// ---------------------------------------------------------------------------
// Differential test: the literal one-exchange reference versus the search.
// ---------------------------------------------------------------------------

struct Lcg(u64);
impl Lcg {
    fn below(&mut self, bound: u64) -> u64 {
        self.0 =
            self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) % bound
    }
    fn chance(&mut self, percent: u64) -> bool {
        self.below(100) < percent
    }
}

struct Case {
    model: Model,
    sources: Vec<Source>,
    y: u32,
    x: u32,
}

/// Seeded two-source selection ADMGs on four to six nodes: acyclic directed and
/// arbitrary bidirected edges, one outcome (the last node), one or two treatments,
/// sparse controllable sets and selection targets.
fn generate(seed: u64, count: usize) -> Vec<Case> {
    let mut rng = Lcg(seed);
    let mut out = Vec::new();
    while out.len() < count {
        let n = 4 + usize::try_from(rng.below(3)).unwrap();
        let mut m = Model { n, directed: Vec::new(), bidirected: Vec::new() };
        for a in 0..n {
            for b in a + 1..n {
                if rng.chance(35) {
                    m.directed.push((a, b));
                }
                if rng.chance(35) {
                    m.bidirected.push((a, b));
                }
            }
        }
        let y = 1u32 << (n - 1);
        let x = loop {
            let x = u32::try_from(rng.below(u64::from(m.all()))).unwrap() + 1;
            if x & y == 0 && x.count_ones() <= 2 {
                break x;
            }
        };
        let sources = ["a", "b"]
            .into_iter()
            .map(|name| {
                let mut controllable = 0u32;
                for i in 0..n - 1 {
                    if rng.chance(25) {
                        controllable |= 1 << i;
                    }
                }
                if controllable == 0 {
                    controllable = 1 << rng.below((n - 1) as u64);
                }
                let mut targets = 0u32;
                for i in 0..n {
                    if rng.chance(30) {
                        targets |= 1 << i;
                    }
                }
                Source { name, controllable, targets }
            })
            .collect();
        out.push(Case { model: m, sources, y, x });
    }
    out
}

/// Departure 3 (repeated exchange). The literal reference (line 10 only with
/// `I = ∅`) and the search agree on every fixture: the search never identifies
/// where the paper's algorithm fails, and a second exchange by the active source
/// is never even eligible, so the extension cannot change a verdict. Fixtures:
/// R-443 Figure 1(c,d) and (e,f) variants, then 6000 seeded two-source ADMGs.
#[test]
fn the_search_never_identifies_where_the_one_exchange_reference_fails() {
    let eligible = Cell::new(0usize);
    let check = |case: &Case| -> Option<(bool, bool)> {
        let decision = decide(&case.model, &case.sources, case.y, case.x)?;
        let literal = Reference::new(&case.model, &case.sources, Exchanges::OnePerBranch)
            .identifies(case.y, case.x);
        let extended = Reference::new(&case.model, &case.sources, Exchanges::ActiveSourceAgain);
        let extended_verdict = extended.identifies(case.y, case.x);
        eligible.set(eligible.get() + extended.repeat_eligible.get());
        // The extension adds nothing to the reference either.
        assert_eq!(literal, extended_verdict, "{:?} {:?}", case.model, case.sources);
        let ours = certified(&decision);
        // Soundness of the extension: never identified where the paper FAILs.
        assert!(!ours || literal, "identified where the paper fails: {:?}", case.model);
        // Parity on these fixtures: nothing the paper identifies is lost.
        assert_eq!(ours, literal, "verdict differs: {:?} {:?}", case.model, case.sources);
        Some((ours, literal))
    };

    // R-443 Figure 1 variants: Z1 = 0, X = 1, Z2 = 2, Y = 3.
    let fig_1cd = model(4, &[(0, 1), (1, 2), (2, 3)], &[(0, 1), (0, 2), (0, 3)]);
    let fig_1ef = model(4, &[(0, 1), (2, 1), (1, 3)], &[(0, 1), (0, 3), (2, 1), (2, 3)]);
    let unhelpful = source("u", &[1], &[0, 1, 2, 3]);
    let figures = [
        // (c,d): complementary sources identify; each alone is an obstruction.
        (&fig_1cd, vec![source("a", &[2], &[0, 2]), source("b", &[0], &[0, 3])], true),
        (&fig_1cd, vec![source("a", &[2], &[0, 2]), unhelpful.clone()], false),
        (&fig_1cd, vec![source("b", &[0], &[0, 3]), unhelpful.clone()], false),
        // (e,f): the experiment split across two sources is the paper's FAIL.
        (&fig_1ef, vec![source("a", &[2], &[0]), source("b", &[0], &[2])], false),
        // (e): both experiments in one source identify.
        (&fig_1ef, vec![source("a", &[0, 2], &[0]), unhelpful.clone()], true),
    ];
    for (m, sources, expected) in figures {
        let case = Case { model: m.clone(), sources, y: 1 << 3, x: 1 << 1 };
        assert_eq!(check(&case), Some((expected, expected)), "{:?}", case.sources);
    }

    let (mut identified, mut not_identified) = (0, 0);
    for case in generate(11, 6000) {
        if let Some((ours, _)) = check(&case) {
            identified += usize::from(ours);
            not_identified += usize::from(!ours);
        }
    }
    // The corpus exercises both verdicts.
    assert!(identified > 300 && not_identified > 500, "{identified} / {not_identified}");
    // A second exchange by the active source is never eligible on any fixture.
    assert_eq!(eligible.get(), 0);
}

/// The cumulative joint regime. When one exchange activates two controllables of
/// a source at once (Figure 1(e): `Z1` and `Z2` both in `X` after line 3), the
/// formula cites the single joint regime `do(Z1, Z2)` of that source, even with
/// `do(Z1)` and `do(Z2)` also supplied: `do(Z_cum)` is the joint of the exchange.
#[test]
fn an_exchange_of_two_controllables_cites_the_one_cumulative_joint_regime() {
    let fig_1ef = model(4, &[(0, 1), (2, 1), (1, 3)], &[(0, 1), (0, 3), (2, 1), (2, 3)]);
    let sources = [source("a", &[0, 2], &[0]), source("u", &[1], &[0, 1, 2, 3])];
    let (catalog, index) = family_catalog(4, &sources);
    let decision = decide_mz_transport(
        &admg(&fig_1ef),
        &mz_query(1 << 3, 1 << 1, &sources),
        &catalog,
        MZ_TRANSPORT_DEFAULT_LIMITS,
        &ExecutionContext::for_tests(1),
    )
    .unwrap();
    let MzTransportDecision::Identified { derivation, cited } = decision else {
        panic!("Figure 1(e) identifies, got {decision:?}");
    };
    assert_eq!(derivation.route(), &MzTransportRoute::SingleSource { population: Arc::from("a") });
    let cited: Vec<(String, u32)> =
        cited.iter().map(|regime| index[usize::try_from(regime.raw()).unwrap()].clone()).collect();
    assert_eq!(cited, [("a".to_owned(), 0b101)], "the joint do(Z1, Z2), not do(Z1) or do(Z2)");
    // The catalog did supply the separate experiments: they were not substituted.
    assert!(index.contains(&("a".to_owned(), 0b001)) && index.contains(&("a".to_owned(), 0b100)));
}

// ---------------------------------------------------------------------------
// Departure 4: a paper FAIL after an exchange is not certified.
// ---------------------------------------------------------------------------

/// `P_{A,B}(y)` on `A -> Y <- B` with `A <-> Y <-> B` (nodes A = 0, B = 1, Y = 2).
/// Both sources can exchange into `do(A)` at the first line-10 call (`I = ∅`), but
/// `B` is still confounded afterwards and, with `I ≠ ∅`, the paper's line 10 cannot
/// fire again: a FAIL reached after an exchange. The search explores both
/// candidate exchanges, finds both branches failing, and reports `search_incomplete`:
/// no obstruction is certified, because the only forced terminal it may certify
/// has no active experiment. The same graph with experiments that can never
/// exchange (they control an unrelated node) is a forced terminal and a checked
/// obstruction: the verdict differs by how the failure was reached, not by truth.
#[test]
fn a_paper_fail_after_exchanges_is_search_incomplete_never_an_obstruction() {
    let (a, b, y) = (0, 1, 2);
    let confounded = model(3, &[(a, y), (b, y)], &[(a, y), (b, y)]);
    let (y_mask, x_mask) = (1u32 << y, (1u32 << a) | (1u32 << b));
    let sources = [source("a", &[a], &[]), source("b", &[a], &[])];

    // The paper's FAIL: literal and extended reference alike.
    for exchanges in [Exchanges::OnePerBranch, Exchanges::ActiveSourceAgain] {
        assert!(!Reference::new(&confounded, &sources, exchanges).identifies(y_mask, x_mask));
    }
    let decision = decide(&confounded, &sources, y_mask, x_mask).unwrap();
    assert!(!matches!(decision, MzTransportDecision::ProvenNonTransportable(_)), "{decision:?}");
    let MzTransportDecision::NotCertified(inspection) = &decision else {
        panic!("expected not_certified, got {decision:?}");
    };
    assert_eq!(inspection.detail, "mz_transport.search_incomplete");
    assert_eq!(decision.reason_code(), Some("transport_not_certified"));
    assert_eq!(decision.detail_code(), Some("mz_transport.search_incomplete"));
    let last = inspection.stages.last().map(|s| (s.stage.as_str(), s.outcome));
    assert_eq!(last, Some(("multi_source", "not_certified")));
    // Both candidate exchanges of the I = ∅ call were tried and ended in a failure.
    let rules = &inspection.explored_rules;
    for population in ["a", "b"] {
        let exchange = format!("ztr.line10.source_exchange:{population}:");
        assert!(rules.iter().any(|r| r.starts_with(&exchange)), "{rules:?}");
    }
    let first_exchange = rules.iter().position(|r| r.starts_with("ztr.line10.source_exchange"));
    let first_fail = rules.iter().position(|r| r == "ztr.line11.fail");
    assert!(first_exchange.is_some() && first_fail > first_exchange, "{rules:?}");

    // Declaration order does not matter.
    let reversed = [sources[1].clone(), sources[0].clone()];
    let again = decide(&confounded, &reversed, y_mask, x_mask).unwrap();
    assert_eq!(again.detail_code(), Some("mz_transport.search_incomplete"));

    // Same graph, but no source can ever exchange: a forced line-11 terminal with
    // no active experiment, replayed as a checked obstruction.
    let with_spare = model(4, &[(a, y), (b, y)], &[(a, y), (b, y)]);
    let spare = [source("a", &[3], &[]), source("b", &[3], &[])];
    let paper = Reference::new(&with_spare, &spare, Exchanges::OnePerBranch);
    assert!(!paper.identifies(y_mask, x_mask));
    let forced = decide(&with_spare, &spare, y_mask, x_mask).unwrap();
    let MzTransportDecision::ProvenNonTransportable(obstruction) = &forced else {
        panic!("expected a forced obstruction, got {forced:?}");
    };
    assert!(obstruction.sources().iter().all(|(_, active, _)| active.is_empty()));
    verify_mz_transport_obstruction(
        &admg(&with_spare),
        obstruction,
        SidLimits::default(),
        &ExecutionContext::for_tests(1),
    )
    .unwrap();
}

// ---------------------------------------------------------------------------
// Departure 6: the line-10 separation condition.
// ---------------------------------------------------------------------------

/// What the search reports for a query decided by source `a` with source `b`
/// unable to exchange.
#[derive(Debug, PartialEq, Eq)]
enum Line10 {
    /// A source exchanges and identifies the effect.
    Exchanges,
    /// No source can: a replayed obstruction with each source's premises
    /// `(population, candidate active controllables, selection separated)`.
    Obstruction(Vec<(String, Vec<u32>, bool)>),
}

fn outcome_of(decision: &MzTransportDecision) -> Line10 {
    match decision {
        MzTransportDecision::Identified { .. } => Line10::Exchanges,
        MzTransportDecision::ProvenNonTransportable(o) => Line10::Obstruction(
            o.sources()
                .into_iter()
                .map(|(population, active, separated)| {
                    (population.to_owned(), active.to_vec(), separated)
                })
                .collect(),
        ),
        other => panic!("unexpected outcome {other:?}"),
    }
}

fn obstruction(premises: &[(&str, &[u32], bool)]) -> Line10 {
    Line10::Obstruction(
        premises.iter().map(|(p, active, sep)| ((*p).to_owned(), active.to_vec(), *sep)).collect(),
    )
}

#[test]
fn the_line_10_separation_is_tested_in_the_graph_with_edges_into_x_removed() {
    // Nodes X = 0, Y = 1, T = 2, W = 3. Bow `X -> Y`, `X <-> Y`; `T` is a child
    // of X outside An(Y); `W` is a parent of Y. Source `b` can never exchange
    // (its selection node points into Y), so `a` decides each verdict.
    let bow = |extra_directed: &[(usize, usize)]| {
        let mut directed = vec![(0, 1)];
        directed.extend_from_slice(extra_directed);
        model(4, &directed, &[(0, 1)])
    };
    let (x_mask, y_mask) = (1u32, 2u32);
    let blocker = source("b", &[0], &[1]);
    let on_a = |controllable: &[usize], targets: &[usize]| {
        vec![source("a", controllable, targets), blocker.clone()]
    };
    let paper = |m: &Model, sources: &[Source]| {
        Reference::new(m, sources, Exchanges::OnePerBranch).identifies(y_mask, x_mask)
    };

    // (a) Separated only after the mutilation. `S_a -> X` with `X <-> Y`: in the
    // unmutilated graph the conditioned X is a collider (S -> X <-> Y) and opens the
    // path; with edges into X removed, S -> X is cut. The exchange is allowed.
    let m = bow(&[(0, 2)]);
    let state = 0b0011; // the state {X, Y} the c-factor recursion reaches
    assert!(!m.selection_separated(state, x_mask, y_mask, 1, false), "open without mutilation");
    assert!(m.selection_separated(state, x_mask, y_mask, 1, true), "separated in D_Xbar");
    let sources = on_a(&[0], &[0]);
    assert!(paper(&m, &sources));
    let decision = decide(&m, &sources, y_mask, x_mask).unwrap();
    assert_eq!(outcome_of(&decision), Line10::Exchanges);
    let MzTransportDecision::Identified { derivation, .. } = &decision else { unreachable!() };
    // Only `a` is cited: b could not exchange, and the target alone fails first.
    let cited_sources = match derivation.route() {
        MzTransportRoute::SingleSource { population } => vec![population.to_string()],
        MzTransportRoute::Combined { populations } => {
            populations.iter().map(ToString::to_string).collect()
        }
        MzTransportRoute::TargetOnly => vec![],
    };
    assert_eq!(cited_sources, ["a"]);
    assert!(derivation.rules().iter().any(|r| r.starts_with("ztr.line10.source_exchange")));
    assert_eq!(
        derivation.stages().first().map(|s| (s.stage.as_str(), s.outcome)),
        Some(("target_only", "not_certified"))
    );

    // (b) Not separated even after the mutilation: S_a -> Y. No exchange.
    let sources = on_a(&[0], &[1]);
    assert!(!m.selection_separated(state, x_mask, y_mask, 2, true));
    assert!(!paper(&m, &sources));
    assert_eq!(
        outcome_of(&decide(&m, &sources, y_mask, x_mask).unwrap()),
        obstruction(&[("a", &[0], false), ("b", &[0], false)])
    );

    // (c) S_a points into a non-ancestor of Y (T, a child of X): the line-2
    // restriction to An(Y) drops T before line 10, so the selection is separated
    // and the exchange is allowed; the experiment is needed (the bow is not
    // identifiable from the target) and used.
    let sources = on_a(&[0], &[2]);
    assert!(m.selection_separated(0b0111, x_mask, y_mask, 4, true));
    assert!(paper(&m, &sources));
    assert_eq!(outcome_of(&decide(&m, &sources, y_mask, x_mask).unwrap()), Line10::Exchanges);

    // (d) S_a points into an ancestor of Y outside X (W -> Y): S -> W -> Y is open
    // whatever X is, so the node is not separated while it stays in the state.
    let with_parent = bow(&[(3, 1)]);
    assert!(!with_parent.selection_separated(0b1011, x_mask, y_mask, 8, true));

    // (e) The second conjunct: S_a separated (no targets) but its only controllable
    // is not a treatment, so there is nothing to exchange; a separated selection is
    // not enough.
    let sources = on_a(&[2], &[]);
    assert!(!paper(&m, &sources));
    assert_eq!(
        outcome_of(&decide(&m, &sources, y_mask, x_mask).unwrap()),
        obstruction(&[("a", &[], true), ("b", &[0], false)])
    );
}
