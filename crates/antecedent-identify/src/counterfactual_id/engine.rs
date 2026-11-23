//! Parallel-world graph search for counterfactual identification.

use crate::sid::SearchCharge;

use super::{
    BTreeMap, BTreeSet, BitSet, CfSymbol, CfTerm, CfValue, CounterfactualFunctional,
    CounterfactualGraphRecord, CounterfactualIdProblem, CounterfactualNodeRecord,
    CounterfactualObstruction, DenseNodeId, GraphWorkspace, HashMap, IdentificationError,
    MeteredId, PreparedAdmg, SearchStop, SharedSearch, components_by, find, hedge_obstruction,
    identify_interventional_metered, implicit_closure, stop_error,
};

/// `var_{sub} = value`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct Atom {
    pub(super) var: u32,
    pub(super) value: CfValue,
    pub(super) sub: BTreeMap<u32, CfValue>,
}

/// Why a derivation stopped.
pub(super) enum Halt {
    Stop(SearchStop),
    Obstruction(Box<CounterfactualObstruction>),
    Failure(String),
}

/// A parent of a parallel-worlds node: fixed by the world, or a node.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ParentRef {
    Fixed(CfValue),
    Node(usize),
}

/// A node of the counterfactual graph (a merged class of parallel-world nodes).
#[derive(Clone, Debug)]
struct CgNode {
    var: u32,
    value: Option<CfValue>,
    eff_sub: BTreeMap<u32, CfValue>,
    parents: Vec<usize>,
}

struct Cg {
    nodes: Vec<CgNode>,
    bidirected: Vec<Vec<usize>>,
    /// Bidirected components, computed once. [`Self::record`] clones this into the artifact.
    districts: Vec<Vec<usize>>,
}

impl Cg {
    /// Bidirected-connected components, each ascending, ordered by smallest member.
    fn districts(&self) -> &[Vec<usize>] {
        &self.districts
    }

    /// Strict ancestors of `node` (the graph is acyclic).
    fn ancestors(&self, node: usize) -> BTreeSet<usize> {
        implicit_closure(self.nodes[node].parents.iter().copied(), |&p| {
            self.nodes[p].parents.iter().copied()
        })
    }

    fn record(&self) -> CounterfactualGraphRecord {
        let mut bidirected = Vec::new();
        for (i, adj) in self.bidirected.iter().enumerate() {
            bidirected.extend(adj.iter().filter(|&&j| j > i).map(|&j| (i, j)));
        }
        bidirected.sort_unstable();
        CounterfactualGraphRecord {
            nodes: self
                .nodes
                .iter()
                .map(|n| CounterfactualNodeRecord {
                    variable: n.var,
                    value: n.value,
                    subscript: n.eff_sub.iter().map(|(v, x)| (*v, *x)).collect(),
                    parents: n.parents.clone(),
                })
                .collect(),
            bidirected,
            districts: self.districts.clone(),
        }
    }
}

pub(super) struct Engine<'p, 'c> {
    pub(super) problem: &'p CounterfactualIdProblem,
    pub(super) prepared: PreparedAdmg,
    pub(super) parents: Vec<Vec<u32>>,
    pub(super) search: SharedSearch<'c>,
    pub(super) next_symbol: u32,
    /// Live bytes of the finished terms (their arenas), charged with every step.
    pub(super) retained_bytes: u64,
    pub(super) first_graph: Option<CounterfactualGraphRecord>,
    pub(super) workspace: GraphWorkspace,
}

fn event_bytes(event: &[Atom]) -> u64 {
    event.iter().map(|a| 48 + 24 * a.sub.len() as u64).sum()
}

impl Engine<'_, '_> {
    fn charge(&mut self, depth: usize, bytes: u64) -> Result<(), Halt> {
        self.search.charge(depth, self.retained_bytes.saturating_add(bytes)).map_err(Halt::Stop)
    }

    pub(super) fn id_star(
        &mut self,
        event: Vec<Atom>,
        depth: usize,
    ) -> Result<CounterfactualFunctional, Halt> {
        self.charge(depth, event_bytes(&event))?;
        // Line 1.
        if event.is_empty() {
            return Ok(CounterfactualFunctional::One);
        }
        // Line 2: an event its own subscript sets away from its level.
        for atom in &event {
            if let Some(set) = atom.sub.get(&atom.var) {
                match (set, atom.value) {
                    (CfValue::Level(a), CfValue::Level(b)) if *a != b => {
                        return Ok(CounterfactualFunctional::Zero);
                    }
                    (a, b) if *a == b => {}
                    _ => {
                        return Err(Halt::Failure("a self-set event with an unknown level".into()));
                    }
                }
            }
        }
        // Line 3: drop events their own subscript sets to their level.
        let event: Vec<Atom> =
            event.into_iter().filter(|a| a.sub.get(&a.var) != Some(&a.value)).collect();
        if event.is_empty() {
            return Ok(CounterfactualFunctional::One);
        }
        // Line 4.
        let Some(cg) = self.make_cg(&event, depth)? else {
            return Ok(CounterfactualFunctional::Zero);
        };
        if depth == 0 && self.first_graph.is_none() {
            self.first_graph = Some(cg.record());
        }
        if cg.districts().len() > 1 {
            return self.line_five(&cg, cg.districts(), depth);
        }
        self.single_district(&cg, depth)
    }

    /// Line 5: sum over free nodes of the product of district terms.
    fn line_five(
        &mut self,
        cg: &Cg,
        districts: &[Vec<usize>],
        depth: usize,
    ) -> Result<CounterfactualFunctional, Halt> {
        let mut symbols = Vec::new();
        let values: Vec<CfValue> = cg
            .nodes
            .iter()
            .map(|node| {
                node.value.unwrap_or_else(|| {
                    let id = self.next_symbol;
                    self.next_symbol += 1;
                    symbols.push(CfSymbol { id, variable: node.var });
                    CfValue::Symbol(id)
                })
            })
            .collect();
        let mut factors = Vec::with_capacity(districts.len());
        for district in districts {
            let inside: BTreeSet<usize> = district.iter().copied().collect();
            let outside_parents: BTreeSet<usize> = district
                .iter()
                .flat_map(|&c| cg.nodes[c].parents.iter().copied())
                .filter(|p| !inside.contains(p))
                .collect();
            let mut atoms = Vec::with_capacity(district.len());
            for &c in district {
                let mut sub = cg.nodes[c].eff_sub.clone();
                // ID* sets every node of the district to the values of all nodes
                // outside it; restricting that to the node's ancestors is an exact
                // simplification (a node does not depend on a non-ancestor), so
                // dropping the restriction gives an equal functional: a mutant
                // that removes it is equivalent, not an unchecked guard.
                let ancestors = cg.ancestors(c);
                for &p in outside_parents.intersection(&ancestors) {
                    let var = cg.nodes[p].var;
                    if let Some(previous) = sub.insert(var, values[p])
                        && previous != values[p]
                    {
                        return Err(Halt::Failure(
                            "two copies of one variable among a node's ancestors".into(),
                        ));
                    }
                }
                atoms.push(Atom { var: cg.nodes[c].var, value: values[c], sub });
            }
            factors.push(self.id_star(atoms, depth + 1)?);
        }
        Ok(CounterfactualFunctional::Sum { symbols, factors })
    }

    /// Lines 6-9: one district.
    fn single_district(&mut self, cg: &Cg, depth: usize) -> Result<CounterfactualFunctional, Halt> {
        let mut sub: BTreeMap<u32, CfValue> = BTreeMap::new();
        for node in &cg.nodes {
            for (&var, &value) in &node.eff_sub {
                if let Some(previous) = sub.insert(var, value)
                    && previous != value
                {
                    return Err(Halt::Obstruction(Box::new(
                        CounterfactualObstruction::ConflictingSubscripts {
                            graph: cg.record(),
                            variable: var,
                            subscript: previous,
                            event: Some(value),
                        },
                    )));
                }
            }
        }
        let mut event: BTreeMap<u32, CfValue> = BTreeMap::new();
        for node in &cg.nodes {
            if let Some(&set) = sub.get(&node.var) {
                // A node read at another level, or left free (`None`: its level is
                // summed, so it may differ), conflicts. The free case cannot occur
                // on the effect-on-the-treated shape; a synthetic conjunction in the
                // unit tests reaches it.
                if node.value != Some(set) {
                    return Err(Halt::Obstruction(Box::new(
                        CounterfactualObstruction::ConflictingSubscripts {
                            graph: cg.record(),
                            variable: node.var,
                            subscript: set,
                            event: node.value,
                        },
                    )));
                }
                continue;
            }
            let Some(value) = node.value else {
                // A free node of the district is marginalized.
                continue;
            };
            match event.insert(node.var, value) {
                Some(previous) if previous != value => {
                    // Two unmerged copies of one variable read at two levels with
                    // no subscript conflict: every parent pair that kept them apart
                    // is, at its root, a node read at exactly the level the other
                    // world fixes (anything else conflicted above), so by
                    // consistency the copies are equal and the conjunction has
                    // probability zero. Unreachable on the effect-on-the-treated
                    // shape (no variable has two copies); reached by a synthetic
                    // conjunction in the unit tests.
                    return match (previous, value) {
                        (CfValue::Level(_), CfValue::Level(_)) => {
                            Ok(CounterfactualFunctional::Zero)
                        }
                        _ => Err(Halt::Obstruction(Box::new(
                            CounterfactualObstruction::ConflictingSubscripts {
                                graph: cg.record(),
                                variable: node.var,
                                subscript: previous,
                                event: Some(value),
                            },
                        ))),
                    };
                }
                _ => {}
            }
        }
        if event.is_empty() {
            return Ok(CounterfactualFunctional::One);
        }
        let n = self.problem.node_count();
        let mut y = BitSet::with_len(n);
        for &v in event.keys() {
            y.insert(DenseNodeId::from_raw(v));
        }
        let mut x = BitSet::with_len(n);
        for &v in sub.keys() {
            x.insert(DenseNodeId::from_raw(v));
        }
        let retained = self.retained_bytes;
        let search = &mut self.search;
        let mut charge = |id_depth: usize, bytes: u64| -> Result<(), IdentificationError> {
            search.charge(depth + id_depth, retained.saturating_add(bytes)).map_err(stop_error)
        };
        let outcome = identify_interventional_metered(
            &self.prepared,
            &y,
            &x,
            &mut self.workspace,
            &mut charge,
        );
        match outcome {
            Ok(MeteredId::Identified { arena, expression }) => {
                self.retained_bytes =
                    self.retained_bytes.saturating_add((arena.len() as u64).saturating_mul(96));
                Ok(CounterfactualFunctional::Term(Box::new(CfTerm {
                    intervention: sub.into_iter().collect(),
                    event: event.into_iter().collect(),
                    arena,
                    expression,
                })))
            }
            Ok(MeteredId::Hedge(hedge)) => {
                Err(Halt::Obstruction(Box::new(hedge_obstruction(self.problem, &x, &y, &hedge))))
            }
            Err(error) => Err(match error {
                IdentificationError::Cancelled | IdentificationError::Budget { .. } => {
                    Halt::Stop(self.search.stop_of(&error))
                }
                other => Halt::Failure(other.to_string()),
            }),
        }
    }

    /// The functional of `P(Y_{X = active} = level, X = observed)`: ID* on the
    /// conjunction, and when ID* stops, the binary consistency complement (its
    /// refusal keeps the ID* obstruction).
    pub(super) fn numerator(
        &mut self,
        treatment: u32,
        active: f64,
        observed: f64,
        outcome: u32,
        level: u64,
    ) -> Result<CounterfactualFunctional, Halt> {
        let event = vec![
            Atom {
                var: outcome,
                value: CfValue::Level(level),
                sub: BTreeMap::from([(treatment, CfValue::Level(active.to_bits()))]),
            },
            Atom {
                var: treatment,
                value: CfValue::Level(observed.to_bits()),
                sub: BTreeMap::new(),
            },
        ];
        match self.id_star(event, 0) {
            Err(Halt::Obstruction(obstruction)) => {
                match self.consistency_complement(treatment, active, outcome, level)? {
                    Some(functional) => Ok(functional),
                    None => Err(Halt::Obstruction(obstruction)),
                }
            }
            other => other,
        }
    }

    /// The binary-treatment consistency complement of
    /// `P(Y_{X = active} = level, X = x')`: `P_x(Y = level) - P(Y = level, X = x)`
    /// when `X` has exactly two levels (so `X = x'` is `X != x`) and complete ID
    /// identifies `P_x(Y)` from `P(V)`; `None` otherwise (the caller keeps the
    /// ID* refusal). Charged to the shared budget like every ID call.
    fn consistency_complement(
        &mut self,
        treatment: u32,
        active: f64,
        outcome: u32,
        level: u64,
    ) -> Result<Option<CounterfactualFunctional>, Halt> {
        if self.problem.levels[treatment as usize].len() != 2 {
            return Ok(None);
        }
        self.charge(0, 0)?;
        let n = self.problem.node_count();
        let mut y = BitSet::with_len(n);
        y.insert(DenseNodeId::from_raw(outcome));
        let mut x = BitSet::with_len(n);
        x.insert(DenseNodeId::from_raw(treatment));
        let retained = self.retained_bytes;
        let search = &mut self.search;
        let mut charge = |id_depth: usize, bytes: u64| -> Result<(), IdentificationError> {
            search.charge(id_depth, retained.saturating_add(bytes)).map_err(stop_error)
        };
        let outcome_id = identify_interventional_metered(
            &self.prepared,
            &y,
            &x,
            &mut self.workspace,
            &mut charge,
        );
        match outcome_id {
            Ok(MeteredId::Identified { arena, expression }) => {
                self.retained_bytes =
                    self.retained_bytes.saturating_add((arena.len() as u64).saturating_mul(96));
                let active = CfValue::Level(active.to_bits());
                let mut observed = vec![(treatment, active), (outcome, CfValue::Level(level))];
                observed.sort_unstable();
                Ok(Some(CounterfactualFunctional::ConsistencyComplement {
                    interventional: Box::new(CfTerm {
                        intervention: vec![(treatment, active)],
                        event: vec![(outcome, CfValue::Level(level))],
                        arena,
                        expression,
                    }),
                    observed,
                }))
            }
            Ok(MeteredId::Hedge(_)) => Ok(None),
            Err(error) => Err(match error {
                IdentificationError::Cancelled | IdentificationError::Budget { .. } => {
                    Halt::Stop(self.search.stop_of(&error))
                }
                other => Halt::Failure(other.to_string()),
            }),
        }
    }

    /// make-cg: `None` when two events on one merged node disagree.
    #[allow(
        clippy::too_many_lines,
        reason = "the parallel-worlds construction, merge and restriction in order"
    )]
    fn make_cg(&mut self, event: &[Atom], depth: usize) -> Result<Option<Cg>, Halt> {
        // Disjoint borrows: the budget is charged while the graph's parents and
        // order are read, so neither is copied per call.
        let Engine { problem, parents: graph_parents, search, retained_bytes, .. } = self;
        let topo = problem.topological_order();
        let problem: &CounterfactualIdProblem = problem;
        let n = problem.node_count();
        let worlds: Vec<BTreeMap<u32, CfValue>> =
            event.iter().map(|a| a.sub.clone()).collect::<BTreeSet<_>>().into_iter().collect();
        // Parallel-world nodes: (world, variable) for every variable a world does not set.
        let mut index: HashMap<(usize, u32), usize> = HashMap::new();
        let mut pw: Vec<(usize, u32)> = Vec::new();
        for (w, sub) in worlds.iter().enumerate() {
            for v in 0..n {
                let Ok(v) = u32::try_from(v) else {
                    return Err(Halt::Failure("variable index does not fit u32".into()));
                };
                if !sub.contains_key(&v) {
                    index.insert((w, v), pw.len());
                    pw.push((w, v));
                }
            }
        }
        let parent_refs = |w: usize, v: u32| -> Vec<ParentRef> {
            graph_parents[v as usize]
                .iter()
                .map(|&p| match worlds[w].get(&p) {
                    Some(&value) => ParentRef::Fixed(value),
                    None => ParentRef::Node(index[&(w, p)]),
                })
                .collect()
        };
        // Union-find, merged in topological order.
        let mut root: Vec<usize> = (0..pw.len()).collect();
        for &v in topo {
            let bytes = (pw.len() as u64).saturating_mul(64).saturating_add(event_bytes(event));
            search.charge(depth, retained_bytes.saturating_add(bytes)).map_err(Halt::Stop)?;
            let copies: Vec<usize> =
                (0..worlds.len()).filter_map(|w| index.get(&(w, v)).copied()).collect();
            for (i, &a) in copies.iter().enumerate() {
                for &b in &copies[i + 1..] {
                    let (ra, rb) = (find(&mut root, a), find(&mut root, b));
                    if ra == rb {
                        continue;
                    }
                    let pa = parent_refs(pw[a].0, v);
                    let pb = parent_refs(pw[b].0, v);
                    // Lemma 24. Two fixed parents at different levels never meet
                    // on the effect-on-the-treated shape (one intervened world);
                    // a synthetic two-world conjunction in the unit tests does.
                    let same = pa.iter().zip(&pb).all(|(x, y)| match (x, y) {
                        (ParentRef::Node(i), ParentRef::Node(j)) => {
                            find(&mut root, *i) == find(&mut root, *j)
                        }
                        (ParentRef::Fixed(i), ParentRef::Fixed(j)) => i == j,
                        _ => false,
                    });
                    if same {
                        root[ra.max(rb)] = ra.min(rb);
                    }
                }
            }
        }
        // Classes, the event on them, and an inconsistent merge.
        let class_of: Vec<usize> = (0..pw.len()).map(|i| find(&mut root, i)).collect();
        let mut value_of: BTreeMap<usize, CfValue> = BTreeMap::new();
        let mut event_classes = BTreeSet::new();
        for atom in event {
            let Some(w) = worlds.iter().position(|s| *s == atom.sub) else {
                return Err(Halt::Failure(
                    "an event world is missing from the submodel list".into(),
                ));
            };
            let Some(&node) = index.get(&(w, atom.var)) else {
                return Err(Halt::Failure("an event on a variable its world sets".into()));
            };
            let class = class_of[node];
            event_classes.insert(class);
            match value_of.insert(class, atom.value) {
                Some(previous) if previous != atom.value => {
                    return match (previous, atom.value) {
                        (CfValue::Level(_), CfValue::Level(_)) => Ok(None),
                        _ => Err(Halt::Failure(
                            "one node read at a symbolic and another level".into(),
                        )),
                    };
                }
                _ => {}
            }
        }
        // Ancestors of the event over merged classes.
        let class_parents = |class: usize| -> Vec<usize> {
            let (w, v) = pw[class];
            let mut out: Vec<usize> = parent_refs(w, v)
                .into_iter()
                .filter_map(|p| match p {
                    ParentRef::Node(i) => Some(class_of[i]),
                    ParentRef::Fixed(_) => None,
                })
                .collect();
            out.sort_unstable();
            out.dedup();
            out
        };
        let keep = implicit_closure(event_classes.iter().copied(), |&c| class_parents(c));
        // Canonical order: by variable, then representative world.
        let mut kept: Vec<usize> = keep.into_iter().collect();
        kept.sort_by_key(|&c| (pw[c].1, pw[c].0));
        let position: HashMap<usize, usize> =
            kept.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        let mut nodes: Vec<CgNode> = Vec::with_capacity(kept.len());
        for &c in &kept {
            let (w, v) = pw[c];
            let parents: Vec<usize> = class_parents(c).into_iter().map(|p| position[&p]).collect();
            let fixed: BTreeMap<u32, CfValue> = graph_parents[v as usize]
                .iter()
                .filter_map(|&p| worlds[w].get(&p).map(|&x| (p, x)))
                .collect();
            nodes.push(CgNode {
                var: v,
                value: value_of.get(&c).copied(),
                eff_sub: fixed,
                parents,
            });
        }
        propagate_effective_subscripts(&mut nodes, topo)?;
        // Bidirected: two copies of one variable share its exogenous term; a
        // bidirected edge of the graph joins every pair of copies. (No variable
        // has two copies on the effect-on-the-treated shape; the unit tests
        // reach the copy edge with a synthetic two-world conjunction.)
        let mut bidirected = vec![Vec::new(); nodes.len()];
        for i in 0..nodes.len() {
            for j in i + 1..nodes.len() {
                let (a, b) = (nodes[i].var, nodes[j].var);
                if a == b || problem.confounded(a, b) {
                    bidirected[i].push(j);
                    bidirected[j].push(i);
                }
            }
        }
        let districts = components_by(nodes.len(), |i| bidirected[i].iter().copied());
        Ok(Some(Cg { nodes, bidirected, districts }))
    }
}

fn propagate_effective_subscripts(nodes: &mut [CgNode], topo: &[u32]) -> Result<(), Halt> {
    // Effective subscripts: fixed parents of a node and of its ancestors, in
    // topological order of the variables (a parent's variable precedes).
    let order: Vec<usize> = {
        let rank: HashMap<u32, usize> = topo.iter().enumerate().map(|(i, &v)| (v, i)).collect();
        let mut o: Vec<usize> = (0..nodes.len()).collect();
        o.sort_by_key(|&i| rank[&nodes[i].var]);
        o
    };
    for &i in &order {
        let mut sub = nodes[i].eff_sub.clone();
        for &p in &nodes[i].parents.clone() {
            for (&var, &value) in &nodes[p].eff_sub {
                if let Some(previous) = sub.insert(var, value)
                    && previous != value
                {
                    return Err(Halt::Failure(
                        "a node's ancestors are set to two levels of one variable".into(),
                    ));
                }
            }
        }
        nodes[i].eff_sub = sub;
    }
    Ok(())
}
