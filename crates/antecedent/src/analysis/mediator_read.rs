// SPDX-License-Identifier: MIT OR Apache-2.0
//! Whether an identified functional reads a variable's column at all.
//!
//! Equivalent topological orders emit equivalent but differently conditioned Tian factors, so
//! the answer is read from the functional's own algebra and the graph, never from a refit: the
//! test normalises the product of distribution leaves the functional sums over, then removes
//! the variable by the two rewrites the chain rule supports.
//!
//! * `sum_m P(m | C) = 1` when `m` appears in no other factor.
//! * `sum_m P(m | C) P(Y | C, D, m) = P(Y | C, D)` when the two factors are read at the same
//!   intervention, domain, population and regime and `m` is m-separated from `D` given `C` in
//!   the supplied graph (so `P(m | C) = P(m | C, D)`). The graph's Markov property is the
//!   premise the identification itself rests on.
//!
//! The test is sound, not complete: it answers `true` only when the variable is removed, and
//! `false` for every shape it does not recognise (a ratio, a kernel, an integral, several
//! factors that mention the variable, a leaf with several targets for the variable).
//!
//! The test is per formula and per graph. A caller holding one formula per equivalence-class
//! completion would answer "unread" only when every completion's formula is unread against its
//! own completion graph; an undecided completion reads the variable. Today's only caller, the
//! mediator placebo, holds one supplied Dag or Admg and its single static formula.

use std::collections::{BTreeSet, HashSet};

use antecedent_core::{RegimeId, VariableId};
use antecedent_expr::{
    CausalExprArena, DomainRef, ExprId, ExprNode, InterventionSetId, PopulationKeyId,
};

/// Most expression nodes one test visits; a larger expression is not decided (`false`).
const NODE_BUDGET: usize = 100_000;

/// Whether `m` is independent of `others` given `given` in the supplied graph.
pub(super) type Independence<'a> = dyn Fn(VariableId, &[VariableId], &[VariableId]) -> bool + 'a;

/// `P(targets | cond)` read at one intervention, domain, population and regime.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Leaf {
    targets: BTreeSet<VariableId>,
    cond: BTreeSet<VariableId>,
    intervened: BTreeSet<VariableId>,
    intervention: InterventionSetId,
    domain: DomainRef,
    population: PopulationKeyId,
    regime: Option<RegimeId>,
}

impl Leaf {
    fn mentions(&self, variable: VariableId) -> bool {
        self.targets.contains(&variable)
            || self.cond.contains(&variable)
            || self.intervened.contains(&variable)
    }

    fn read_like(&self, other: &Self) -> bool {
        self.intervention == other.intervention
            && self.domain == other.domain
            && self.population == other.population
            && self.regime == other.regime
    }
}

/// `sum_{sums} prod leaves`.
#[derive(Debug, Default)]
struct Flat {
    sums: BTreeSet<VariableId>,
    leaves: Vec<Leaf>,
}

impl Flat {
    fn mentioned(&self) -> BTreeSet<VariableId> {
        let mut all = self.sums.clone();
        for leaf in &self.leaves {
            all.extend(&leaf.targets);
            all.extend(&leaf.cond);
            all.extend(&leaf.intervened);
        }
        all
    }
}

/// Whether the functional rooted at `root` provably does not read the column of `variable`.
///
/// `outcome` must be the outcome of every expectation in the functional: a functional over
/// another variable space is not decided. `independent` answers m-separation in the graph the
/// functional was identified on.
pub(super) fn functional_never_reads(
    arena: &CausalExprArena,
    root: ExprId,
    variable: VariableId,
    outcome: VariableId,
    independent: &Independence<'_>,
) -> bool {
    let mut budget = NODE_BUDGET;
    unread(arena, root, variable, outcome, independent, &mut budget)
}

fn unread(
    arena: &CausalExprArena,
    id: ExprId,
    variable: VariableId,
    outcome: VariableId,
    independent: &Independence<'_>,
    budget: &mut usize,
) -> bool {
    match arena.node(id) {
        ExprNode::Contrast { left, right, .. } => {
            unread(arena, *left, variable, outcome, independent, budget)
                && unread(arena, *right, variable, outcome, independent, budget)
        }
        ExprNode::Expectation { function, distribution } => {
            function.variable() == outcome
                && function.variable() != variable
                && distribution_unread(arena, *distribution, variable, independent, budget)
        }
        _ => false,
    }
}

fn distribution_unread(
    arena: &CausalExprArena,
    id: ExprId,
    variable: VariableId,
    independent: &Independence<'_>,
    budget: &mut usize,
) -> bool {
    if !mentions(arena, id, variable, &mut HashSet::new(), budget) {
        return *budget > 0;
    }
    flatten(arena, id, budget).is_some_and(|flat| removable(&flat, variable, independent))
}

/// Whether `variable` occurs anywhere under `id`. An exhausted budget reports a mention.
fn mentions(
    arena: &CausalExprArena,
    id: ExprId,
    variable: VariableId,
    seen: &mut HashSet<ExprId>,
    budget: &mut usize,
) -> bool {
    if !seen.insert(id) {
        return false;
    }
    if *budget == 0 {
        return true;
    }
    *budget -= 1;
    match arena.node(id) {
        ExprNode::Distribution { variables, conditioned_on, intervention, .. } => {
            arena.var_set(*variables).contains(&variable)
                || arena.var_set(*conditioned_on).contains(&variable)
                || arena.intervention_set(*intervention).contains(&variable)
        }
        ExprNode::Kernel { body, bound, .. } => {
            arena.var_set(*bound).contains(&variable)
                || mentions(arena, *body, variable, seen, budget)
        }
        ExprNode::Product(list) => {
            arena.list(*list).iter().any(|child| mentions(arena, *child, variable, seen, budget))
        }
        ExprNode::SumOut { variables, expr } | ExprNode::IntegralOut { variables, expr } => {
            arena.var_set(*variables).contains(&variable)
                || mentions(arena, *expr, variable, seen, budget)
        }
        ExprNode::Ratio { numerator, denominator } => {
            mentions(arena, *numerator, variable, seen, budget)
                || mentions(arena, *denominator, variable, seen, budget)
        }
        ExprNode::Expectation { function, distribution } => {
            function.variable() == variable
                || mentions(arena, *distribution, variable, seen, budget)
        }
        ExprNode::Contrast { left, right, .. } => {
            mentions(arena, *left, variable, seen, budget)
                || mentions(arena, *right, variable, seen, budget)
        }
    }
}

/// `id` as one sum over a product of leaves, or `None` for any other shape (including a sum
/// whose bound variable another factor also names, which a flat sum would capture).
fn flatten(arena: &CausalExprArena, id: ExprId, budget: &mut usize) -> Option<Flat> {
    if *budget == 0 {
        return None;
    }
    *budget -= 1;
    match arena.node(id) {
        ExprNode::Distribution {
            variables,
            conditioned_on,
            intervention,
            domain,
            population,
            regime,
        } => Some(Flat {
            sums: BTreeSet::new(),
            leaves: vec![Leaf {
                targets: arena.var_set(*variables).iter().copied().collect(),
                cond: arena.var_set(*conditioned_on).iter().copied().collect(),
                intervened: arena.intervention_set(*intervention).into_iter().collect(),
                intervention: *intervention,
                domain: *domain,
                population: *population,
                regime: *regime,
            }],
        }),
        ExprNode::Product(list) => {
            let parts = arena
                .list(*list)
                .iter()
                .map(|child| flatten(arena, *child, budget))
                .collect::<Option<Vec<_>>>()?;
            let mentioned: Vec<BTreeSet<VariableId>> = parts.iter().map(Flat::mentioned).collect();
            for (at, part) in parts.iter().enumerate() {
                let captured = mentioned
                    .iter()
                    .enumerate()
                    .any(|(other, seen)| other != at && !part.sums.is_disjoint(seen));
                if captured {
                    return None;
                }
            }
            let mut merged = Flat::default();
            for part in parts {
                merged.sums.extend(part.sums);
                merged.leaves.extend(part.leaves);
            }
            Some(merged)
        }
        ExprNode::SumOut { variables, expr } => {
            let mut flat = flatten(arena, *expr, budget)?;
            for variable in arena.var_set(*variables) {
                if !flat.sums.insert(*variable) {
                    return None;
                }
            }
            Some(flat)
        }
        _ => None,
    }
}

/// Whether summing `variable` out of the product removes it, by the two rewrites of the module
/// documentation.
fn removable(flat: &Flat, variable: VariableId, independent: &Independence<'_>) -> bool {
    if !flat.leaves.iter().any(|leaf| leaf.mentions(variable)) {
        return true;
    }
    if !flat.sums.contains(&variable)
        || flat.leaves.iter().any(|leaf| leaf.intervened.contains(&variable))
    {
        return false;
    }
    let targeting: Vec<&Leaf> =
        flat.leaves.iter().filter(|leaf| leaf.targets.contains(&variable)).collect();
    let conditioning: Vec<&Leaf> = flat
        .leaves
        .iter()
        .filter(|leaf| !leaf.targets.contains(&variable) && leaf.cond.contains(&variable))
        .collect();
    let [law] = targeting.as_slice() else {
        return false;
    };
    if law.targets.len() != 1 {
        return false;
    }
    match conditioning.as_slice() {
        [] => true,
        [reader] => {
            let rest: BTreeSet<VariableId> =
                reader.cond.iter().copied().filter(|v| *v != variable).collect();
            if !law.read_like(reader) || !law.cond.is_subset(&rest) {
                return false;
            }
            let extra: Vec<VariableId> = rest.difference(&law.cond).copied().collect();
            let given: Vec<VariableId> = law.cond.iter().copied().collect();
            extra.is_empty() || independent(variable, &extra, &given)
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use antecedent_core::{CausalQuery, PathSpecificEffectQuery, VariableId};
    use antecedent_graph::{Admg, DSeparationWorkspace, Dag, DenseNodeId};
    use antecedent_identify::{IdentificationWorkspace, PathSpecificIdentifier};

    use super::functional_never_reads;

    fn v(raw: u32) -> VariableId {
        VariableId::from_raw(raw)
    }

    /// `t -> a, t -> b, a -> y, b -> y, a <-> y` with the four variables at the given dense
    /// ids, so the label-canonical topological order is `t, a, b, y` or `t, b, a, y`.
    fn district_graph(t: u32, a: u32, b: u32, y: u32) -> Admg {
        let node = DenseNodeId::from_raw;
        let mut graph = Admg::with_variables(4);
        for (from, to) in [(t, a), (t, b), (a, y), (b, y)] {
            graph.insert_directed(node(from), node(to)).unwrap();
        }
        graph.insert_bidirected(node(a), node(y)).unwrap();
        graph
    }

    fn m_separated(graph: &Admg) -> impl Fn(VariableId, &[VariableId], &[VariableId]) -> bool + '_ {
        move |m, others, given| {
            let node = |variable: VariableId| DenseNodeId::from_raw(variable.raw());
            let others: Vec<_> = others.iter().copied().map(node).collect();
            let given: Vec<_> = given.iter().copied().map(node).collect();
            graph
                .is_m_separated_sets(
                    &[node(m)],
                    &others,
                    &given,
                    &mut DSeparationWorkspace::default(),
                )
                .unwrap_or(false)
        }
    }

    fn identified(
        graph: &Admg,
        t: u32,
        y: u32,
        mediator: u32,
    ) -> (antecedent_identify::IdentificationResult, VariableId) {
        let identifier = PathSpecificIdentifier::new();
        let prepared = identifier.prepare(graph).unwrap();
        let query = CausalQuery::PathSpecific(
            PathSpecificEffectQuery::binary(v(t), v(y)).with_path_nodes(vec![v(mediator)]),
        );
        let result = identifier
            .identify(&prepared, &query, &mut IdentificationWorkspace::default())
            .unwrap();
        (result, v(y))
    }

    #[test]
    fn the_district_functional_does_not_read_its_mediator_in_either_topological_order() {
        // (t, a, b, y) orders the mediator before the other parent of y; (t, b, a, y) after it.
        for (t, a, b, y) in [(0, 1, 2, 3), (0, 2, 1, 3)] {
            let graph = district_graph(t, a, b, y);
            assert_eq!(
                graph.topological_order().unwrap().iter().map(|id| id.raw()).collect::<Vec<_>>(),
                if a < b { [t, a, b, y] } else { [t, b, a, y] },
            );
            let (result, outcome) = identified(&graph, t, y, a);
            let functional = result.estimands[0].functional;
            assert!(
                functional_never_reads(
                    &result.arena,
                    functional,
                    v(a),
                    outcome,
                    &m_separated(&graph)
                ),
                "order with a = {a}, b = {b}"
            );
            // The other parent of the outcome is read.
            assert!(
                !functional_never_reads(
                    &result.arena,
                    functional,
                    v(b),
                    outcome,
                    &m_separated(&graph)
                ),
                "order with a = {a}, b = {b}"
            );
        }
    }

    #[test]
    fn a_functional_that_needs_an_independence_the_graph_does_not_give_is_read() {
        let graph = district_graph(0, 1, 2, 3);
        let (result, outcome) = identified(&graph, 0, 3, 1);
        let functional = result.estimands[0].functional;
        // With every independence withheld the leaf conditioned on the other parent of the
        // outcome cannot be matched to the mediator's own law.
        assert!(!functional_never_reads(&result.arena, functional, v(1), outcome, &|_, _, _| {
            false
        }));
    }

    #[test]
    fn the_edge_g_formula_reads_its_mediator() {
        // w -> t, t -> a, t -> b, a -> y, b -> y: the mediator's law is read at the active
        // level and the outcome's at both, so no chain-rule marginalisation removes it.
        let mut dag = Dag::with_variables(5);
        for (from, to) in [(0, 1), (1, 2), (1, 3), (2, 4), (3, 4)] {
            dag.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
        }
        let graph = Admg::from_dag(&dag).unwrap();
        let (result, outcome) = identified(&graph, 1, 4, 2);
        let functional = result.estimands[0].functional;
        assert!(!functional_never_reads(
            &result.arena,
            functional,
            v(2),
            outcome,
            &m_separated(&graph)
        ));
    }

    #[test]
    fn a_variable_the_functional_never_names_is_unread_and_another_outcome_is_not_decided() {
        let graph = district_graph(0, 1, 2, 3);
        let (result, outcome) = identified(&graph, 0, 3, 1);
        let functional = result.estimands[0].functional;
        assert!(functional_never_reads(&result.arena, functional, v(9), outcome, &|_, _, _| false));
        assert!(!functional_never_reads(&result.arena, functional, v(9), v(2), &|_, _, _| false));
    }
}
