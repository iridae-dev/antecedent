// SPDX-License-Identifier: MIT OR Apache-2.0
//! Theorem-scoped ioSCM identification, Forre and Mooij (2020), Algorithm 1.
//! Consolidated districts merge bidirected components and SCCs. Successful
//! formulas are retained in antecedent-expr. FAIL means only that this procedure
//! did not identify the target: the paper makes no completeness claim.

use antecedent_core::{ExecutionContext, VariableId};
use antecedent_expr::{
    CausalExprArena, DomainRef, ExprId, ExprNode, FunctionalProgram, ProgramLimits, ProgramSchema,
    ProgramVariable,
};
use antecedent_graph::{CyclicMixedGraph, DenseNodeId, SigmaSeparationWorkspace, sigma_separates};
use std::collections::BTreeSet;
use thiserror::Error;

type Set = BTreeSet<usize>;

/// Scientific hypotheses of the generalized cyclic ID result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CyclicIdPremises {
    /// The supplied graph is the induced DMG of a compatible ioSCM.
    pub io_scm: bool,
    /// Joint observational law has a density relative to a product of SCC measures.
    pub scc_product_density: bool,
}

/// Outcome of a bounded identification attempt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CyclicIdStatus {
    /// A theorem-scoped functional was derived.
    Identified,
    /// Algorithm 1 reached FAIL; no impossibility claim is made.
    NotIdentifiedByProcedure,
    /// Work limit stopped the search before a result.
    BudgetExhausted,
}

/// Retained derivation and executable expression, with its scope.
pub struct CyclicIdentification {
    /// Distinct success/procedure-failure/work-limit status.
    pub status: CyclicIdStatus,
    /// Structurally checked functional only on successful identification.
    pub program: Option<FunctionalProgram>,
    /// Derivation steps, including failed attempts.
    pub derivation: Vec<String>,
    /// Scientific assumptions; not inferred from graph syntax or numeric solutions.
    pub premises: CyclicIdPremises,
}

/// Invalid query, missing scientific premises, cancellation, or IR construction error.
#[derive(Debug, Error)]
pub enum CyclicIdError {
    /// Malformed input or unchecked assumptions.
    #[error("cyclic identification input: {0}")]
    Invalid(String),
    /// Cancellation or resource admission failed.
    #[error("cyclic identification resource refusal: {0}")]
    Resource(String),
}
fn invalid(s: impl std::fmt::Display) -> CyclicIdError {
    CyclicIdError::Invalid(s.to_string())
}
fn id(i: usize) -> DenseNodeId {
    DenseNodeId::from_raw(u32::try_from(i).expect("bounded graph order"))
}
fn var(i: usize) -> VariableId {
    VariableId::from_raw(u32::try_from(i).expect("bounded graph order"))
}

fn induced(graph: &CyclicMixedGraph, active: &Set) -> CyclicMixedGraph {
    let mut sub = CyclicMixedGraph::with_variables(
        u32::try_from(graph.node_count()).expect("bounded graph order"),
    );
    for (a, b) in graph.directed_edge_pairs_sorted() {
        if active.contains(&(a as usize)) && active.contains(&(b as usize)) {
            sub.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b))
                .expect("original graph edge");
        }
    }
    for (a, b) in graph.bidirected_edge_pairs_sorted() {
        if active.contains(&(a as usize)) && active.contains(&(b as usize)) {
            sub.insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b))
                .expect("original graph edge");
        }
    }
    sub
}
fn components(graph: &CyclicMixedGraph, active: &Set) -> Vec<Set> {
    induced(graph, active)
        .condensation()
        .components()
        .iter()
        .map(|c| c.iter().map(|i| i.as_usize()).filter(|i| active.contains(i)).collect::<Set>())
        .filter(|s| !s.is_empty())
        .collect()
}
fn ancestors(graph: &CyclicMixedGraph, active: &Set, seeds: &Set) -> Set {
    let mut result = seeds.clone();
    loop {
        let previous = result.len();
        for (a, b) in graph.directed_edge_pairs_sorted() {
            if active.contains(&(a as usize)) && result.contains(&(b as usize)) {
                result.insert(a as usize);
            }
        }
        if result.len() == previous {
            return result;
        }
    }
}
fn districts(graph: &CyclicMixedGraph, active: &Set) -> Vec<Set> {
    let mut groups = components(graph, active);
    for (a, b) in graph.bidirected_edge_pairs_sorted() {
        let (a, b) = (a as usize, b as usize);
        if !active.contains(&a) || !active.contains(&b) {
            continue;
        }
        let i = groups.iter().position(|s| s.contains(&a)).expect("active node");
        let j = groups.iter().position(|s| s.contains(&b)).expect("active node");
        if i != j {
            let group = groups.remove(i.max(j));
            groups[i.min(j)].extend(group);
        }
    }
    groups.sort_by_key(|s| s.first().copied());
    groups
}

struct Search<'a> {
    graph: &'a CyclicMixedGraph,
    arena: CausalExprArena,
    continuous: bool,
    steps: usize,
    max_steps: usize,
    trace: Vec<String>,
    ctx: &'a ExecutionContext,
    exhausted: bool,
}
impl Search<'_> {
    fn marginal(&mut self, body: ExprId, remove: &Set) -> ExprId {
        if remove.is_empty() {
            return body;
        }
        let variables = self.arena.intern_var_set(remove.iter().copied().map(var));
        self.arena.intern(if self.continuous {
            ExprNode::IntegralOut { variables, expr: body }
        } else {
            ExprNode::SumOut { variables, expr: body }
        })
    }
    fn product(&mut self, parts: Vec<ExprId>) -> ExprId {
        if parts.len() == 1 {
            return parts[0];
        }
        let list = self.arena.intern_list(parts);
        self.arena.intern(ExprNode::Product(list))
    }
    fn extract(&mut self, active: &Set, kernel: ExprId, district: &Set) -> ExprId {
        let mut previous = Set::new();
        let mut factors = Vec::new();
        for block in components(self.graph, active) {
            let mut prefix = previous.clone();
            prefix.extend(&block);
            if block.is_subset(district) {
                let numerator =
                    self.marginal(kernel, &active.difference(&prefix).copied().collect());
                let factor = if previous.is_empty() {
                    numerator
                } else {
                    let denominator =
                        self.marginal(kernel, &active.difference(&previous).copied().collect());
                    self.arena.intern(ExprNode::Ratio { numerator, denominator })
                };
                factors.push(factor);
            }
            previous = prefix;
        }
        self.product(factors)
    }
    fn identify_district(
        &mut self,
        target: &Set,
        current: &Set,
        kernel: ExprId,
    ) -> Result<Option<ExprId>, CyclicIdError> {
        if self.ctx.cancellation.is_cancelled() {
            return Err(CyclicIdError::Resource("cancelled".into()));
        }
        if self.steps >= self.max_steps {
            self.exhausted = true;
            return Ok(None);
        }
        self.steps += 1;
        let ancestral = ancestors(self.graph, current, target);
        self.trace.push(format!("IDCD: C={target:?}, D={current:?}, Anc(C)={ancestral:?}"));
        let marginal = self.marginal(kernel, &current.difference(&ancestral).copied().collect());
        if &ancestral == target {
            return Ok(Some(marginal));
        }
        if &ancestral == current {
            self.trace.push(
                "FAIL: ancestral/consolidated-district shrinking stopped; no completeness claim"
                    .into(),
            );
            return Ok(None);
        }
        let district = districts(self.graph, &ancestral)
            .into_iter()
            .find(|d| target.is_subset(d))
            .ok_or_else(|| invalid("target not in consolidated district"))?;
        let next = self.extract(&ancestral, marginal, &district);
        self.identify_district(target, &district, next)
    }
}

fn query_sets(
    graph: &CyclicMixedGraph,
    outcomes: &[usize],
    treatments: &[usize],
    premises: CyclicIdPremises,
) -> Result<(Set, Set, Set), CyclicIdError> {
    let n = graph.node_count();
    if n == 0 || n > 32 || !premises.io_scm {
        return Err(invalid("1..32 nodes and the ioSCM premise required"));
    }
    let y: Set = outcomes.iter().copied().collect();
    let x: Set = treatments.iter().copied().collect();
    if y.is_empty()
        || y.len() != outcomes.len()
        || x.len() != treatments.len()
        || y.iter().chain(&x).any(|&i| i >= n)
        || !x.is_disjoint(&y)
    {
        return Err(invalid("disjoint unique in-range outcomes/treatments required"));
    }
    Ok(((0..n).collect(), y, x))
}

/// Generalized cyclic ID (Algorithm 1), with SCC chain-rule kernels in the
/// existing expression IR. `continuous=false` uses sums instead of integrals.
/// No input/selection nodes are admitted by this first observation-only entry point.
pub fn identify_cyclic(
    graph: &CyclicMixedGraph,
    outcomes: &[usize],
    treatments: &[usize],
    premises: CyclicIdPremises,
    continuous: bool,
    max_steps: usize,
    ctx: &ExecutionContext,
) -> Result<CyclicIdentification, CyclicIdError> {
    let (all, y, x) = query_sets(graph, outcomes, treatments, premises)?;
    if !premises.scc_product_density { return Err(invalid("generalized ID requires the SCC-product-density premise")); }
    if max_steps > 10_000 {
        return Err(invalid("identification step cap"));
    }
    let _lease = ctx
        .acquire_admission(
            1,
            (graph.node_count() * graph.node_count() * 128 + max_steps * 4096) as u64,
        )
        .map_err(|e| CyclicIdError::Resource(e.to_string()))?;
    let mut arena = CausalExprArena::new();
    let variables = arena.intern_var_set(all.iter().copied().map(var));
    let empty = arena.empty_var_set();
    let intervention = arena.empty_intervention_set();
    let joint = arena.intern_distribution(variables, empty, intervention, DomainRef::Observational);
    let mut search = Search {
        graph,
        arena,
        continuous,
        steps: 0,
        max_steps,
        trace: vec![
            "Forre-Mooij 2020 Theorem 9.10, Algorithm 1; ioSCM and SCC-product density assumed"
                .into(),
        ],
        ctx,
        exhausted: false,
    };
    let free = all.difference(&x).copied().collect();
    let h = ancestors(graph, &free, &y);
    let mut factors = Vec::new();
    for c in districts(graph, &h) {
        let d = districts(graph, &all)
            .into_iter()
            .find(|d| c.is_subset(d))
            .ok_or_else(|| invalid("consolidated district"))?;
        let q = search.extract(&all, joint, &d);
        match search.identify_district(&c, &d, q)? {
            Some(expr) => factors.push(expr),
            None => {
                return Ok(CyclicIdentification {
                    status: if search.exhausted {
                        CyclicIdStatus::BudgetExhausted
                    } else {
                        CyclicIdStatus::NotIdentifiedByProcedure
                    },
                    program: None,
                    derivation: search.trace,
                    premises,
                });
            }
        }
    }
    let product = search.product(factors);
    let root = search.marginal(product, &h.difference(&y).copied().collect());
    let schema = ProgramSchema::new(
        all.iter().map(|&i| (var(i), ProgramVariable { name: format!("v{i}").into() })),
    );
    let program =
        FunctionalProgram::new(search.arena, schema, root, root, ProgramLimits::default())
            .map_err(invalid)?;
    Ok(CyclicIdentification {
        status: CyclicIdStatus::Identified,
        program: Some(program),
        derivation: search.trace,
        premises,
    })
}

/// Check the cyclic backdoor specialization (Corollary 8.4(4)) directly on the
/// extended graph: Z sigma-independent of intervention indicators, and Y
/// sigma-independent of indicators given X,Z. This is a sufficient criterion.
pub fn cyclic_backdoor_valid(
    graph: &CyclicMixedGraph,
    outcomes: &[usize],
    treatments: &[usize],
    adjustment: &[usize],
    premises: CyclicIdPremises,
) -> Result<bool, CyclicIdError> {
    let (all, y, x) = query_sets(graph, outcomes, treatments, premises)?;
    let z: Set = adjustment.iter().copied().collect();
    if x.is_empty()
        || z.len() != adjustment.len()
        || !z.is_subset(&all)
        || !z.is_disjoint(&x)
        || !z.is_disjoint(&y)
    {
        return Err(invalid("adjustment set"));
    }
    let n = graph.node_count();
    let mut extended =
        CyclicMixedGraph::with_variables(u32::try_from(n + x.len()).map_err(invalid)?);
    for (a, b) in graph.directed_edge_pairs_sorted() {
        extended
            .insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b))
            .map_err(invalid)?;
    }
    for (a, b) in graph.bidirected_edge_pairs_sorted() {
        extended
            .insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b))
            .map_err(invalid)?;
    }
    let mut ws = SigmaSeparationWorkspace::default();
    let conditioned: Vec<_> = x.union(&z).copied().map(id).collect();
    for (k, &treatment) in x.iter().enumerate() {
        let indicator = id(n + k);
        extended.insert_directed(indicator, id(treatment)).map_err(invalid)?;
    }
    for k in 0..x.len() {
        for &a in &z {
            if !sigma_separates(&extended, id(a), id(n + k), &[], &mut ws)
                .map_err(invalid)?
                .status
                .is_separated()
            {
                return Ok(false);
            }
        }
        for &a in &y {
            if !sigma_separates(&extended, id(a), id(n + k), &conditioned, &mut ws)
                .map_err(invalid)?
                .status
                .is_separated()
            {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn premises() -> CyclicIdPremises {
        CyclicIdPremises { io_scm: true, scc_product_density: true }
    }
    #[test]
    fn cyclic_id_retains_functionals_and_distinguishes_fail_from_budget() {
        let mut g = CyclicMixedGraph::with_variables(3);
        g.insert_directed(id(0), id(1)).unwrap();
        g.insert_directed(id(1), id(2)).unwrap();
        g.insert_directed(id(2), id(1)).unwrap();
        let ctx = ExecutionContext::for_tests(1);
        let identified = identify_cyclic(&g, &[2], &[0], premises(), false, 100, &ctx).unwrap();
        assert_eq!(identified.status, CyclicIdStatus::Identified);
        assert!(identified.program.is_some());
        let failed = identify_cyclic(&g, &[2], &[1], premises(), false, 100, &ctx).unwrap();
        assert_eq!(failed.status, CyclicIdStatus::NotIdentifiedByProcedure);
        let bounded = identify_cyclic(&g, &[2], &[0], premises(), false, 0, &ctx).unwrap();
        assert_eq!(bounded.status, CyclicIdStatus::BudgetExhausted);
        assert!(cyclic_backdoor_valid(&g, &[2], &[0], &[], premises()).unwrap());
        assert!(!cyclic_backdoor_valid(&g, &[2], &[1], &[], premises()).unwrap());
    }
    #[test]
    fn cyclic_id_latent_bow_is_not_licensed_and_acyclic_chain_is() {
        let mut g = CyclicMixedGraph::with_variables(2);
        g.insert_directed(id(0), id(1)).unwrap();
        let ctx = ExecutionContext::for_tests(2);
        assert_eq!(
            identify_cyclic(&g, &[1], &[0], premises(), true, 100, &ctx).unwrap().status,
            CyclicIdStatus::Identified
        );
        g.insert_bidirected(id(0), id(1)).unwrap();
        assert_eq!(
            identify_cyclic(&g, &[1], &[0], premises(), true, 100, &ctx).unwrap().status,
            CyclicIdStatus::NotIdentifiedByProcedure
        );
    }
}

#[cfg(test)]
mod numerical_tests {
    use super::*;
    use antecedent_core::Value;
    use antecedent_expr::{Assignment, DistributionProvider, EvalContext, EvalError, FactorSpec};
    use std::sync::Arc;
    struct Joint([f64; 8]);
    impl DistributionProvider for Joint {
        fn probability(
            &self,
            spec: &FactorSpec<'_>,
            assignment: &Assignment,
            _: &EvalContext,
        ) -> Result<f64, EvalError> {
            let mass = |variables: &[VariableId]| -> Result<f64, EvalError> {
                let constraints = variables
                    .iter()
                    .map(|&v| {
                        Ok((
                            v.raw() as usize,
                            assignment
                                .get(v)
                                .and_then(Value::as_f64)
                                .ok_or(EvalError::MissingBinding(v))?,
                        ))
                    })
                    .collect::<Result<Vec<_>, EvalError>>()?;
                Ok(self
                    .0
                    .iter()
                    .enumerate()
                    .filter(|(cell, _)| {
                        constraints.iter().all(|(i, v)| f64::from(((cell >> i) & 1) as u8) == *v)
                    })
                    .map(|(_, p)| p)
                    .sum())
            };
            let denominator = mass(spec.conditioned_on)?;
            let mut variables = spec.variables.to_vec();
            variables.extend_from_slice(spec.conditioned_on);
            Ok(mass(&variables)? / denominator)
        }
        fn support(
            &self,
            vars: &[VariableId],
            _: &EvalContext,
        ) -> Result<Arc<[Arc<[Value]>]>, EvalError> {
            Ok((0..(1usize << vars.len()))
                .map(|cell| {
                    (0..vars.len())
                        .map(|i| Value::f64(f64::from(((cell >> i) & 1) as u8)))
                        .collect()
                })
                .collect())
        }
        fn outcome(
            &self,
            v: VariableId,
            assignment: &Assignment,
            _: &EvalContext,
        ) -> Result<f64, EvalError> {
            assignment.get(v).and_then(Value::as_f64).ok_or(EvalError::MissingBinding(v))
        }
        fn n_draws(&self) -> Option<usize> {
            None
        }
    }
    #[test]
    fn cyclic_id_scc_chain_rule_functional_evaluates_against_independent_joint_table() {
        // Independent probability table. Its conditional block is treated jointly;
        // the formula must not replace it by two local equilibrium conditionals.
        let mut joint = [0.0; 8];
        for (cell, probability) in joint.iter_mut().enumerate() {
            let t = (cell & 1) != 0;
            let a = (cell & 2) != 0;
            let b = (cell & 4) != 0;
            let pt = if t { 0.6 } else { 0.4 };
            let pa = if t { 0.8 } else { 0.2 };
            let pb = 0.1 + if a { 0.5 } else { 0.0 } + if t { 0.2 } else { 0.0 };
            *probability = pt * (if a { pa } else { 1.0 - pa }) * (if b { pb } else { 1.0 - pb });
        }
        let mut g = CyclicMixedGraph::with_variables(3);
        for (a, b) in [(0, 1), (0, 2), (1, 2), (2, 1)] {
            g.insert_directed(id(a), id(b)).unwrap();
        }
        let result = identify_cyclic(
            &g,
            &[2],
            &[0],
            CyclicIdPremises { io_scm: true, scc_product_density: true },
            false,
            100,
            &ExecutionContext::for_tests(0),
        )
        .unwrap();
        let program = result.program.unwrap().compile().unwrap();
        for (t, expected) in [(0.0, 0.2), (1.0, 0.7)] {
            let mut env = Assignment::new();
            env.set(var(0), Value::f64(t));
            env.set(var(2), Value::f64(1.0));
            let evaluated =
                program.evaluate_with(&Joint(joint), &EvalContext::default(), &env).unwrap();
            assert!((evaluated - expected).abs() < 1e-12, "{evaluated} vs {expected}");
        }
    }
    #[test]
    fn cyclic_id_acyclic_frontdoor_limit_matches_latent_generator_interventions() {
        let mut joint = [0.0; 8];
        for (cell, probability) in joint.iter_mut().enumerate() {
            let x = cell & 1 != 0;
            let m = cell & 2 != 0;
            let y = cell & 4 != 0;
            for u in [false, true] {
                let px = if u { 0.8 } else { 0.1 };
                let pm = if x { 0.9 } else { 0.2 };
                let py = 0.1 + if m { 0.5 } else { 0.0 } + if u { 0.25 } else { 0.0 };
                *probability += 0.5
                    * (if x { px } else { 1.0 - px })
                    * (if m { pm } else { 1.0 - pm })
                    * (if y { py } else { 1.0 - py });
            }
        }
        let mut graph = CyclicMixedGraph::with_variables(3);
        graph.insert_directed(id(0), id(1)).unwrap();
        graph.insert_directed(id(1), id(2)).unwrap();
        graph.insert_bidirected(id(0), id(2)).unwrap();
        let result = identify_cyclic(
            &graph,
            &[2],
            &[0],
            CyclicIdPremises { io_scm: true, scc_product_density: true },
            false,
            100,
            &ExecutionContext::for_tests(0),
        )
        .unwrap();
        assert_eq!(result.status, CyclicIdStatus::Identified);
        let program = result.program.unwrap().compile().unwrap();
        for (x, expected) in [(0.0, 0.325), (1.0, 0.675)] {
            let mut assignment = Assignment::new();
            assignment.set(var(0), Value::f64(x));
            assignment.set(var(2), Value::f64(1.0));
            let found =
                program.evaluate_with(&Joint(joint), &EvalContext::default(), &assignment).unwrap();
            assert!((found - expected).abs() < 1e-12, "frontdoor {found} vs {expected}");
        }
    }
}
