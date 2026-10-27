//! Mixed-source rule engine and the decision that runs it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    Admg, Arc, CausalExprArena, DSeparationWorkspace, DenseNodeId, ExprId, HashMap,
    IdentificationError, SearchCharge, SearchReceipt, SearchStop, SharedSearch, VariableId,
};
#[cfg(test)]
use super::super::{ExecutionContext, SearchBudget, SearchLimits};

use super::{
    BTreeMap, BTreeSet, BUDGET, EvidenceCatalog, INVALID_QUERY, MAX_ALTERNATIVES, MISSING_JOINT,
    Mask, MixedExclusion, MixedInput, MixedQuantity, MixedRule, MixedSourceDerivation,
    MixedSourceLeaf, MixedSourceQuery, MixedStageRecord, MixedStep, NAMED_ROUTE, NOT_CERTIFIED,
    RegimeId, bit, bounds_exceeded, build_expression, dense_variables, invalid_derivation,
    invalid_query, members, render_steps, size, subsets_within,
};
#[cfg(test)]
use super::{
    MIXED_SOURCE_DEFAULT_LIMITS, MIXED_SOURCE_MAX_DO, MIXED_SOURCE_MAX_MOVE,
    validate_mixed_source_query,
};

mod checker;
mod decide;
pub use checker::verify_mixed_source_derivation;
use checker::{check_steps, connected};
#[cfg(test)]
use checker::{check_with_shared_expression_layer, unary_rule_holds};
pub use decide::decide_mixed_source;
pub(crate) use decide::decide_mixed_source_shared;
#[cfg(test)]
use decide::rule_search;
pub(super) use decide::{meta_applies, stage_names};

// ------------------------------------------------------------------ engine

/// A step of the working search, in masks over dense graph indices.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Raw {
    rule: MixedRule,
    premises: Vec<usize>,
    params: Mask,
    y: Mask,
    d: Mask,
    z: Mask,
    /// Variables the step's expression mentions freely.
    free: Mask,
    /// Input steps: index into the engine's input list.
    input: Option<usize>,
    relaxed: bool,
}

type Key = (Mask, Mask, Mask);

impl Raw {
    const fn key(&self) -> Key {
        (self.y, self.d, self.z)
    }
}

enum RunEnd {
    Found,
    Fixpoint,
    Stopped(SearchStop),
}

struct Engine<'g> {
    graph: &'g Admg,
    n: usize,
    universe: Mask,
    /// Parent mask per node, for the ancestor closures.
    parent_masks: Vec<Mask>,
    variables: Vec<VariableId>,
    inputs: Vec<MixedInput>,
    target: Key,
    ws: DSeparationWorkspace,
    /// Mutilated graphs by `(over, under)`: a rule application reuses the one its
    /// side condition needs instead of rebuilding it.
    graphs: HashMap<(Mask, Mask), Admg>,
    /// Decided side conditions by `(over, under, a, b, cond)`.
    separations: HashMap<(Mask, Mask, Mask, Mask, Mask), bool>,
    steps: Vec<Raw>,
    known: HashMap<Key, usize>,
    by_dz: HashMap<(Mask, Mask), Vec<usize>>,
    /// Steps by their hard-intervention set, for the product with a known marginal.
    by_d: HashMap<Mask, Vec<usize>>,
    /// Most variables one rule application moves in this pass.
    max_move: usize,
    /// Largest hard-intervention set a derived quantity may carry in this pass.
    max_do: usize,
    /// Some application was skipped only because of `max_move` or `max_do`.
    pruned: bool,
    alternatives: Vec<Raw>,
    found: Option<usize>,
    operations: usize,
    operations_to_proof: usize,
    generation: usize,
    completed: usize,
    unexpanded: usize,
    rule_counts: BTreeMap<&'static str, usize>,
}

/// `graph` (over `n` dense nodes) with every edge into `over` and every directed
/// edge out of `under` removed (bidirected edges at an `over` node are edges into it).
fn mutilated(graph: &Admg, n: usize, over: Mask, under: Mask) -> Admg {
    debug_assert_eq!(n, graph.node_count());
    graph
        .copy_filtered(
            antecedent_graph::CopyLabels::StaticIndex { extra: 0 },
            |from, to| over & bit(to.as_usize()) == 0 && under & bit(from.as_usize()) == 0,
            // A bidirected edge at an intervened node is an edge into it.
            |a, b| over & (bit(a.as_usize()) | bit(b.as_usize())) == 0,
        )
        .expect("mutilated ADMG dropped an edge")
}

impl<'g> Engine<'g> {
    #[cfg(test)]
    fn new(
        graph: &'g Admg,
        variables: Vec<VariableId>,
        inputs: Vec<MixedInput>,
        query: &MixedSourceQuery,
    ) -> Result<Self, IdentificationError> {
        Self::with_bounds(
            graph,
            variables,
            inputs,
            query,
            (MIXED_SOURCE_MAX_MOVE, MIXED_SOURCE_MAX_DO),
        )
    }

    /// An engine whose rule applications move at most `bounds.0` variables and whose
    /// quantities carry at most `bounds.1` actions.
    fn with_bounds(
        graph: &'g Admg,
        variables: Vec<VariableId>,
        inputs: Vec<MixedInput>,
        query: &MixedSourceQuery,
        bounds: (usize, usize),
    ) -> Result<Self, IdentificationError> {
        let n = variables.len();
        if n >= usize::try_from(Mask::BITS).unwrap_or(usize::MAX) {
            return Err(bounds_exceeded());
        }
        let mask_of = |vs: &[VariableId]| {
            vs.iter().try_fold(0, |acc: Mask, v| {
                variables
                    .iter()
                    .position(|w| w == v)
                    .map(|i| acc | bit(i))
                    .ok_or_else(|| invalid_query(&"variable outside the graph"))
            })
        };
        let target = (mask_of(&query.outcomes)?, mask_of(&query.treatments)?, 0);
        let masks = inputs
            .iter()
            .map(|input| Ok((mask_of(&input.measured)?, mask_of(&input.intervened)?)))
            .collect::<Result<Vec<_>, IdentificationError>>()?;
        let mut engine = Self {
            graph,
            n,
            universe: (1 << n) - 1,
            parent_masks: graph
                .node_ids()
                .map(|v| graph.parents(v).iter().fold(0, |acc, p| acc | bit(p.as_usize())))
                .collect(),
            variables,
            inputs: Vec::new(),
            target,
            ws: DSeparationWorkspace::default(),
            graphs: HashMap::new(),
            separations: HashMap::new(),
            steps: Vec::new(),
            known: HashMap::new(),
            by_dz: HashMap::new(),
            by_d: HashMap::new(),
            max_move: bounds.0,
            max_do: bounds.1,
            pruned: false,
            alternatives: Vec::new(),
            found: None,
            operations: 0,
            operations_to_proof: 0,
            generation: 0,
            completed: 0,
            unexpanded: 0,
            rule_counts: BTreeMap::new(),
        };
        for (index, (input, (y, d))) in inputs.iter().zip(masks).enumerate() {
            engine.add(Raw {
                rule: MixedRule::Input,
                premises: Vec::new(),
                params: 0,
                y,
                d,
                z: 0,
                free: y | d,
                input: Some(index),
                relaxed: !input.marginals.is_empty(),
            });
        }
        engine.inputs = inputs;
        Ok(engine)
    }

    fn variables_of(&self, mask: Mask) -> Vec<VariableId> {
        let mut out = members(mask).map(|i| self.variables[i]).collect::<Vec<_>>();
        out.sort_unstable();
        out
    }

    /// Live state: every step and alternative, plus the cached mutilated graphs
    /// and decided side conditions (an estimate of each entry's footprint).
    fn live_bytes(&self) -> u64 {
        let per_step = std::mem::size_of::<Raw>() + 96;
        let per_graph = 64 + 16 * self.n * self.n;
        let per_separation = std::mem::size_of::<(Mask, Mask, Mask, Mask, Mask)>() + 8;
        u64::try_from(
            (self.steps.len() + self.alternatives.len()) * per_step
                + self.graphs.len() * per_graph
                + self.separations.len() * per_separation,
        )
        .unwrap_or(u64::MAX)
    }

    fn charge(
        &mut self,
        search: &mut SharedSearch<'_>,
        generation: usize,
    ) -> Result<(), SearchStop> {
        search.charge(generation, self.live_bytes())?;
        self.operations += 1;
        Ok(())
    }

    /// Add a candidate; only the first derivation of a quantity is kept, and
    /// another last step of the target is remembered as an alternative.
    fn add(&mut self, raw: Raw) {
        let key = raw.key();
        if self.known.contains_key(&key) {
            let duplicate = |other: &Raw| {
                other.rule == raw.rule
                    && other.premises == raw.premises
                    && other.params == raw.params
                    && other.input == raw.input
            };
            if key == self.target
                && self.alternatives.len() < MAX_ALTERNATIVES
                && !self.found.is_some_and(|f| duplicate(&self.steps[f]))
                && !self.alternatives.iter().any(duplicate)
            {
                self.alternatives.push(raw);
            }
            return;
        }
        let index = self.steps.len();
        self.known.insert(key, index);
        self.by_dz.entry((raw.d, raw.z)).or_default().push(index);
        self.by_d.entry(raw.d).or_default().push(index);
        *self.rule_counts.entry(raw.rule.as_str()).or_default() += 1;
        if key == self.target && self.found.is_none() {
            self.found = Some(index);
            self.operations_to_proof = self.operations;
        }
        self.steps.push(raw);
    }

    /// Ancestors of `seeds` (inclusive) within `G` without the edges into `over`.
    fn ancestors_without_into(&self, seeds: Mask, over: Mask) -> Mask {
        antecedent_graph::algo::mask_closure(seeds, &self.parent_masks, over)
    }

    /// `a` is m-separated from `b` given `cond` in `G` without the edges into
    /// `over` and out of `under`. Each mutilated graph is built once and each
    /// decided side condition is remembered.
    fn separated(&mut self, over: Mask, under: Mask, a: Mask, b: Mask, cond: Mask) -> bool {
        if let Some(&decided) = self.separations.get(&(over, under, a, b, cond)) {
            return decided;
        }
        let graph = match self.graphs.entry((over, under)) {
            std::collections::hash_map::Entry::Occupied(cached) => cached.into_mut(),
            std::collections::hash_map::Entry::Vacant(slot) => {
                let graph = mutilated(self.graph, self.n, over, under);
                slot.insert(graph)
            }
        };
        let dense = DenseNodeId::from_index;
        let conditions = members(cond).map(dense).collect::<Vec<_>>();
        let decided = members(a).all(|x| {
            members(b).all(|y| {
                graph.is_m_separated(dense(x), dense(y), &conditions, &mut self.ws).unwrap_or(false)
            })
        });
        self.separations.insert((over, under, a, b, cond), decided);
        decided
    }

    /// Apply every searched rule to step `q`: marginalize, condition, rules 1 and
    /// 3 insertion, rule 2 both ways, and product. The two deletion rules are
    /// checker-only (see the module docs). Each attempted application charges
    /// the budget.
    fn expand(
        &mut self,
        search: &mut SharedSearch<'_>,
        q: usize,
        generation: usize,
    ) -> Result<(), SearchStop> {
        self.charge(search, generation)?;
        let (y, d, z, free) = {
            let base = &self.steps[q];
            (base.y, base.d, base.z, base.free)
        };
        let unary = |rule, params, y, d, z, free| Raw {
            rule,
            premises: vec![q],
            params,
            y,
            d,
            z,
            free,
            input: None,
            relaxed: false,
        };
        // Each list is cut to what the budget can still pay for (one operation per
        // application, plus one so the stop is observed), so a wide pool is never
        // materialized ahead of the charge.
        let room =
            usize::try_from(search.operations_remaining()).unwrap_or(usize::MAX).saturating_add(1);
        let (max_move, max_do) = (self.max_move, self.max_do);
        if size(y) > 1 {
            self.pruned |= size(y) - 1 > max_move;
            let moves = subsets_within(y, max_move, room, |w| w != y);
            for &w in &moves {
                self.charge(search, generation)?;
                self.add(unary(MixedRule::Marginalize, w, y & !w, d, z, free & !w));
            }
            for &w in &moves {
                self.charge(search, generation)?;
                self.add(unary(MixedRule::Condition, w, y & !w, d, z | w, free));
            }
        }
        let pool = self.universe & !(y | d | z);
        let outside = subsets_within(pool, max_move, room, |_| true);
        self.pruned |= size(pool) > max_move;
        for &w in &outside {
            self.charge(search, generation)?;
            if self.separated(d, 0, y, w, d | z) {
                self.add(unary(MixedRule::Rule1Insert, w, y, d, z | w, free));
            }
        }
        let allowed = max_do.saturating_sub(size(d));
        self.pruned |= size(z).min(max_move) > allowed;
        for w in subsets_within(z, max_move, room, |w| size(w) <= allowed) {
            self.charge(search, generation)?;
            if self.separated(d, w, y, w, d | (z & !w)) {
                self.add(unary(MixedRule::Rule2ToDo, w, y, d | w, z & !w, free));
            }
        }
        self.pruned |= size(d) > max_move;
        for w in subsets_within(d, max_move, room, |_| true) {
            self.charge(search, generation)?;
            if self.separated(d & !w, w, y, w, (d & !w) | z) {
                self.add(unary(MixedRule::Rule2ToObservation, w, y, d & !w, z | w, free));
            }
        }
        // Pearl's `W(Z)`: the inserted actions that are ancestors of the observed
        // set in `G` without the edges into the existing actions keep their edges.
        let past = self.ancestors_without_into(z, d);
        self.pruned |= size(pool).min(max_move) > allowed;
        for &w in &outside {
            if size(w) > allowed {
                continue;
            }
            self.charge(search, generation)?;
            if self.separated(d | (w & !past), 0, y, w, d | z) {
                self.add(unary(MixedRule::Rule3Insert, w, y, d | w, z, free));
            }
        }
        // Product: `q` as the conditional `P(A | B, Z')` of a known `P(B | Z')`. The known
        // marginals are found through the steps sharing `d` (never by walking the submasks
        // of `z`), in the order of the submasks they stand for.
        let mut marginals = self
            .by_d
            .get(&d)
            .map(|steps| {
                steps
                    .iter()
                    .copied()
                    .filter(|&j| {
                        let step = &self.steps[j];
                        step.y & !z == 0 && step.z == z & !step.y
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        marginals.sort_by_key(|&j| (size(self.steps[j].y), self.steps[j].y));
        for j in marginals {
            self.charge(search, generation)?;
            let (b, marginal_free) = (self.steps[j].y, self.steps[j].free);
            self.add(Raw {
                rule: MixedRule::Product,
                premises: vec![q, j],
                params: 0,
                y: y | b,
                d,
                z: z & !b,
                free: free | marginal_free,
                input: None,
                relaxed: false,
            });
        }
        // Product: `q` as the marginal `P(B | Z)` of a known conditional `P(A | B, Z)`.
        let partners = self.by_dz.get(&(d, z | y)).cloned().unwrap_or_default();
        for j in partners {
            self.charge(search, generation)?;
            let (conditional_y, conditional_free) = (self.steps[j].y, self.steps[j].free);
            self.add(Raw {
                rule: MixedRule::Product,
                premises: vec![j, q],
                params: 0,
                y: conditional_y | y,
                d,
                z,
                free: conditional_free | free,
                input: None,
                relaxed: false,
            });
        }
        Ok(())
    }

    /// Chain generation by generation until the target is found, no new quantity
    /// appears, or the shared budget stops. A found target keeps its generation
    /// running so alternative last steps can be collected; a stop there is a
    /// stop like any other, because the proof's check still has to be charged to
    /// the same budget and can no longer be.
    fn run(&mut self, search: &mut SharedSearch<'_>) -> RunEnd {
        if self.found.is_some() {
            return RunEnd::Found;
        }
        let mut start = 0;
        let mut generation = 0;
        loop {
            let end = self.steps.len();
            if start == end {
                return RunEnd::Fixpoint;
            }
            generation += 1;
            self.generation = generation;
            for q in start..end {
                if let Err(stop) = self.expand(search, q, generation) {
                    self.unexpanded = end - q;
                    return RunEnd::Stopped(stop);
                }
            }
            self.completed = generation;
            if self.found.is_some() {
                return RunEnd::Found;
            }
            start = end;
        }
    }

    fn generation_of(&self, index: usize) -> usize {
        let mut memo = vec![0usize; index + 1];
        for i in 0..=index {
            memo[i] = self.steps[i].premises.iter().map(|p| memo[*p] + 1).max().unwrap_or(0);
        }
        memo[index]
    }

    fn quantity(&self, raw: &Raw) -> MixedQuantity {
        MixedQuantity {
            y: self.variables_of(raw.y),
            intervened: self.variables_of(raw.d),
            conditioned: self.variables_of(raw.z),
        }
    }

    /// The proof of step `last` (an existing step, or `replace` standing for it):
    /// its premise closure renumbered in order, as portable [`MixedStep`]s whose
    /// expressions are built in a fresh arena.
    fn extract(
        &self,
        last: &Raw,
        catalog: &EvidenceCatalog,
        target: &str,
    ) -> Result<(Vec<MixedStep>, CausalExprArena), IdentificationError> {
        let mut needed = BTreeSet::new();
        let mut pending = last.premises.clone();
        while let Some(index) = pending.pop() {
            if needed.insert(index) {
                pending.extend(self.steps[index].premises.iter().copied());
            }
        }
        let order = needed.into_iter().collect::<Vec<_>>();
        let renumbered =
            order.iter().enumerate().map(|(new, old)| (*old, new)).collect::<BTreeMap<_, _>>();
        let position = |old: usize| renumbered.get(&old).copied();
        let mut arena = CausalExprArena::new();
        let mut steps: Vec<MixedStep> = Vec::with_capacity(order.len() + 1);
        for raw in order.iter().map(|i| &self.steps[*i]).chain([last]) {
            let premises = raw
                .premises
                .iter()
                .map(|p| position(*p).ok_or_else(invalid_derivation))
                .collect::<Result<Vec<_>, _>>()?;
            let source = raw
                .input
                .map(|index| {
                    let input = &self.inputs[index];
                    let distribution =
                        catalog.distribution(input.regime).ok_or_else(invalid_derivation)?;
                    Ok::<_, IdentificationError>(MixedSourceLeaf {
                        regime: input.regime,
                        study: Arc::clone(&input.study),
                        population: Arc::clone(&distribution.population),
                        snapshot: input.snapshot.clone(),
                        identity: input.identity.clone(),
                    })
                })
                .transpose()?;
            let mut step = MixedStep {
                rule: raw.rule,
                premises,
                params: self.variables_of(raw.params),
                y: self.variables_of(raw.y),
                intervened: self.variables_of(raw.d),
                conditioned: self.variables_of(raw.z),
                source,
                relaxed: raw.relaxed,
                expression: ExprId::from_raw(0),
            };
            let expressions =
                step.premises.iter().map(|p| steps[*p].expression).collect::<Vec<_>>();
            step.expression = build_expression(&mut arena, &step, &expressions, target)?;
            steps.push(step);
        }
        Ok((steps, arena))
    }
}

// ----------------------------------------------------------- independent checker

/// The proof checker. It reads the graph's adjacency directly and decides every
/// do-calculus side condition with Richardson's augmented-graph criterion, sharing
/// no code with the search's m-separation oracle, and rebuilds every expression in
/// a fresh arena; the shared expression layer then compiles the root and binds
/// its leaves ([`check_with_shared_expression_layer`]).
///
/// The conditional transport checker (`sid::conditional`) reuses its separation
/// test for rule-2 premises, so it is visible to the parent module.
///
/// Its ancestor closure and m-separation search are an independent implementation by
/// design (with `selection_separation::independently_separated` and the cfg(test)
/// `mag_id_bruteforce` bitmask oracle): they must not be rewritten through the graph
/// crate's closure owners or the search engine's `Admg::is_m_separated`, or the three
/// encodings stop being able to check one another.
pub(crate) struct Checker {
    variables: Vec<VariableId>,
    parents: Vec<Vec<usize>>,
    siblings: Vec<Vec<usize>>,
}

pub(crate) type Set = BTreeSet<usize>;

impl Checker {
    pub(crate) fn new(graph: &Admg) -> Result<Self, IdentificationError> {
        let variables = dense_variables(graph)?;
        let n = variables.len();
        let dense = DenseNodeId::from_index;
        let mut parents = vec![Vec::new(); n];
        let mut siblings = vec![Vec::new(); n];
        for (from, from_siblings) in siblings.iter_mut().enumerate() {
            for to in graph.children(dense(from)) {
                parents[to.as_usize()].push(from);
            }
            for to in graph.bidirected_neighbors(dense(from)) {
                from_siblings.push(to.as_usize());
            }
        }
        Ok(Self { variables, parents, siblings })
    }

    fn set(&self, variables: &[VariableId]) -> Result<Set, IdentificationError> {
        let set = variables
            .iter()
            .map(|v| self.variables.iter().position(|w| w == v).ok_or_else(invalid_derivation))
            .collect::<Result<Set, _>>()?;
        if set.len() == variables.len() { Ok(set) } else { Err(invalid_derivation()) }
    }

    /// Ancestors of `seeds` (inclusive) along directed edges not entering `over`.
    fn ancestors(&self, seeds: &Set, over: &Set) -> Set {
        let mut out = seeds.clone();
        let mut pending = seeds.iter().copied().collect::<Vec<_>>();
        while let Some(node) = pending.pop() {
            if over.contains(&node) {
                continue;
            }
            for parent in &self.parents[node] {
                if out.insert(*parent) {
                    pending.push(*parent);
                }
            }
        }
        out
    }

    /// `a` and `b` are m-separated given `cond` after every edge into `over` and
    /// every directed edge out of `under` is removed.
    pub(crate) fn separated(&self, over: &Set, under: &Set, a: &Set, b: &Set, cond: &Set) -> bool {
        let n = self.variables.len();
        let mut parents = vec![Vec::new(); n];
        let mut siblings = vec![Vec::new(); n];
        for node in 0..n {
            if over.contains(&node) {
                continue;
            }
            parents[node] =
                self.parents[node].iter().copied().filter(|p| !under.contains(p)).collect();
            siblings[node] =
                self.siblings[node].iter().copied().filter(|s| !over.contains(s)).collect();
        }
        a.iter().all(|x| {
            b.iter().all(|y| {
                !cond.contains(x) && !cond.contains(y) && x != y && {
                    !connected(&parents, &siblings, cond, *x, *y)
                }
            })
        })
    }
}

// ---------------------------------------------------------------- decision

/// The exact leaf a missing-joint refusal needs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MixedMissingLeaf {
    /// The step of the relaxed proof that cites it.
    pub step: usize,
    /// The regime that holds the variables only as separate marginals.
    pub regime: RegimeId,
    /// Its study.
    pub study: Arc<str>,
    /// The jointly measured variables the proof needs as one joint law.
    pub variables: Vec<VariableId>,
    /// The regime's hard interventions.
    pub intervened: Vec<VariableId>,
    /// The variables the regime supplies as separate marginals.
    pub marginals: Vec<VariableId>,
}

/// A derivation that would identify the query if the catalog held a joint it holds
/// only as separate marginals. The proof is checked, with its relaxed leaves marked.
#[derive(Clone, Debug)]
pub struct MixedMissingEvidence {
    /// Every joint the proof cites that the catalog lacks.
    pub leaves: Vec<MixedMissingLeaf>,
    /// The relaxed proof, in step order; its relaxed leaves are marked.
    pub proof: Vec<MixedStep>,
    /// Stages evaluated before the rule search, in order.
    pub stages: Vec<MixedStageRecord>,
}

impl MixedMissingEvidence {
    /// The relaxed proof rendered as a compact graph.
    #[must_use]
    pub fn proof_graph(&self, names: &[String]) -> Vec<String> {
        render_steps(&self.proof, names, self.proof.len().saturating_sub(1))
    }
}

/// What a search that stopped short of a formula explored.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MixedSearchInspection {
    /// Every stage evaluated, in order.
    pub stages: Vec<MixedStageRecord>,
    /// The target quantity nothing derived.
    pub goal: MixedQuantity,
    /// The most recently derived quantities.
    pub frontier: Vec<MixedQuantity>,
    /// Quantities derived.
    pub quantities: usize,
    /// Generations completed.
    pub generations: usize,
    /// Rule applications charged by the rule search.
    pub operations: usize,
    /// Quantities derived per rule.
    pub rule_counts: Vec<(&'static str, usize)>,
    /// Every regime that was not an input, with its reason.
    pub exclusions: Vec<MixedExclusion>,
}

/// Outcome of a bounded mixed-source decision. None of them is a
/// non-identification claim.
#[derive(Clone, Debug)]
pub enum MixedSourceDecision {
    /// A theorem-scoped route identifies the query; use it.
    NamedRoute {
        /// `target_first_sid`, `z_transport`, `mz_transport` or `meta_transport`.
        route: &'static str,
        /// Stages evaluated before it and including it.
        stages: Vec<MixedStageRecord>,
    },
    /// A checked derivation from the supplied distributions.
    Identified {
        /// The checked derivation.
        derivation: Box<MixedSourceDerivation>,
        /// Catalog regimes the proof cites.
        cited: Arc<[RegimeId]>,
        /// Other derivations of the target found in the same generation, each
        /// independently checked. Empty when none was found.
        alternatives: Vec<MixedSourceDerivation>,
    },
    /// The query is identified only if the catalog held a joint it holds as
    /// separate marginals.
    MissingEvidence(Box<MixedMissingEvidence>),
    /// The frozen rule set reached its fixpoint without the target.
    NotCertified(Box<MixedSearchInspection>),
    /// A limit or cancellation stopped the search; nothing is claimed.
    Exhausted(SearchReceipt),
}

impl MixedSourceDecision {
    /// Identification status in the shared transport vocabulary
    /// ([`TransportOutcomeKind::IDENTIFICATION`](antecedent_core::TransportOutcomeKind::IDENTIFICATION)).
    ///
    /// A named route is identified by that theorem-scoped route (prepare it there); its
    /// legacy Python `outcome` spellings `named_route` and `exhausted` read as
    /// `identified` and `budget_cancel`.
    #[must_use]
    pub const fn identification_status(&self) -> antecedent_core::TransportOutcomeKind {
        match self {
            Self::Identified { .. } | Self::NamedRoute { .. } => {
                antecedent_core::TransportOutcomeKind::Identified
            }
            Self::NotCertified(_) => antecedent_core::TransportOutcomeKind::NotCertified,
            Self::MissingEvidence(_) => antecedent_core::TransportOutcomeKind::MissingEvidence,
            Self::Exhausted(_) => antecedent_core::TransportOutcomeKind::BudgetCancel,
        }
    }
}

impl MixedSourceDecision {
    /// Top-level reason code of a non-identified outcome; `None` when identified.
    #[must_use]
    pub const fn reason_code(&self) -> Option<&'static str> {
        match self {
            Self::Identified { .. } => None,
            Self::NamedRoute { .. } => Some("route_not_supported"),
            Self::MissingEvidence(_) => Some("transport_missing_evidence"),
            Self::NotCertified(_) => Some("transport_not_certified"),
            Self::Exhausted(_) => Some("transport_budget_cancel"),
        }
    }

    /// Stable `mixed_search.*` detail of a non-identified outcome.
    #[must_use]
    pub const fn detail_code(&self) -> Option<&'static str> {
        match self {
            Self::Identified { .. } => None,
            Self::NamedRoute { .. } => Some(NAMED_ROUTE),
            Self::MissingEvidence(_) => Some(MISSING_JOINT),
            Self::NotCertified(_) => Some(NOT_CERTIFIED),
            Self::Exhausted(_) => Some(BUDGET),
        }
    }
}

// ----------------------------------------------------------------- binding

/// A checked mixed-source derivation bound to the catalog it cites. Every leaf
/// names its regime, so binding re-checks the derivation against the catalog and
/// records the cited regimes; one provider then evaluates the formula.
#[derive(Clone, Debug)]
pub struct BoundMixedSourceFunctional {
    derivation: MixedSourceDerivation,
    catalog: EvidenceCatalog,
    cited: Arc<[RegimeId]>,
}

impl BoundMixedSourceFunctional {
    /// Checked symbolic derivation.
    #[must_use]
    pub const fn derivation(&self) -> &MixedSourceDerivation {
        &self.derivation
    }
    /// Formula arena.
    #[must_use]
    pub const fn arena(&self) -> &CausalExprArena {
        &self.derivation.arena
    }
    /// Formula root.
    #[must_use]
    pub fn root(&self) -> ExprId {
        self.derivation.root()
    }
    /// Frozen catalog the leaves are bound to.
    #[must_use]
    pub const fn catalog(&self) -> &EvidenceCatalog {
        &self.catalog
    }
    /// Every catalog regime a leaf cites, sorted and without repeats.
    #[must_use]
    pub fn cited_regimes(&self) -> &[RegimeId] {
        &self.cited
    }
    /// The study and population of each cited regime.
    #[must_use]
    pub fn cited_sources(&self) -> Vec<(RegimeId, Arc<str>, Arc<str>)> {
        self.derivation
            .leaves()
            .into_iter()
            .map(|(_, leaf)| (leaf.regime, Arc::clone(&leaf.study), Arc::clone(&leaf.population)))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }
}

/// Bind a checked derivation to a catalog: the derivation is re-checked against
/// the catalog, so a leaf always reads the regime whose study supplied it.
///
/// # Errors
/// `mixed_search.invalid_derivation` when the derivation does not check against
/// `graph` and `catalog`; `mixed_search.invalid_query` for a malformed catalog.
pub fn bind_mixed_source_catalog(
    graph: &Admg,
    derivation: &MixedSourceDerivation,
    catalog: &EvidenceCatalog,
) -> Result<BoundMixedSourceFunctional, IdentificationError> {
    catalog.validate().map_err(|error| {
        IdentificationError::invalid_catalog(format!("{INVALID_QUERY}: {error}"))
    })?;
    verify_mixed_source_derivation(graph, derivation, catalog)?;
    Ok(BoundMixedSourceFunctional {
        derivation: derivation.clone(),
        catalog: catalog.clone(),
        cited: Arc::from(derivation.cited_regimes()),
    })
}

#[cfg(test)]
mod tests;
