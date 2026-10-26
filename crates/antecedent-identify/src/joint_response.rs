//! Joint response adjustment using the generalized back-door criterion.
//!
//! Uses Definition 20 of Perkovic et al. (JMLR 18, 2018): no descendants
//! of any target in Z, and each target's back-door paths blocked by Z plus
//! the other targets. This is sufficient, not complete general response ID.
//!
//! Existence is decided by one set, as for a single treatment. Let `R` be the candidates: the
//! ancestors of the targets and the outcome that descend from no target. Treatment `x` must be
//! separated from the outcome in its own proper back-door graph by `Z` together with the
//! other targets `I = X \ {x}`. In that graph the ancestral hull of `{x, y} ∪ I` contains
//! `R ∪ I` (a directed path from an ancestor to its first target or outcome never leaves `x`
//! by an edge the cut removed), so by the constructive separator theorem (Tian, Paz and
//! Pearl 1998; van der Zander et al. 2014) `R` separates for `x` whenever any admissible `Z`
//! does. `R` is therefore a common set for every treatment exactly when a joint set exists:
//! one separation test per treatment on `R` decides existence, and a failure is a proof that
//! no joint set exists.
//!
//! When `R` passes, the search for a smaller set is exhaustive by size and bounded by
//! [`crate::GeneralizedAdjustmentConfig::max_examinations`], the declared candidate ceiling
//! and the storage the context can lease. A search any of them stops returns `R` itself,
//! already certified (valid, not minimal), with `identify.joint.search_bounded` as an
//! Execution diagnostic.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    CausalQuery, Diagnostic, DiagnosticKind, DiagnosticSeverity, NodeRef, ResponseQuery,
};
use antecedent_expr::{CausalExprArena, DomainRef, ExprNode, OutcomeExprId};
use antecedent_graph::{Admg, Cpdag, DSeparationWorkspace, Dag, DenseNodeId, Endpoint, Pag};

use crate::generalized::{directed_closure, mag_to_admg, not_identified};
use crate::{
    DerivationTrace, GeneralizedAdjustmentIdentifier, IdentificationEnvelope, IdentificationError,
    IdentificationPerformanceRecord, IdentificationResult, IdentifiedEstimand,
};

impl GeneralizedAdjustmentIdentifier {
    /// Identify a joint response with a common adjustment set per MEC completion.
    ///
    /// # Errors
    /// Invalid query, unsupported observation/temporal policy, or graph error.
    pub fn identify_cpdag_response_envelope(
        &self,
        graph: &Cpdag,
        query: &ResponseQuery,
    ) -> Result<IdentificationEnvelope<Dag>, IdentificationError> {
        self.cpdag_envelope_with(graph, |dag| self.identify_joint_dag_response(dag, query))
    }

    /// Identify a joint response per MAG completion, including edge visibility.
    ///
    /// # Errors
    /// Invalid query, unsupported observation/temporal policy, or graph error.
    pub fn identify_pag_response_envelope(
        &self,
        graph: &Pag,
        query: &ResponseQuery,
    ) -> Result<IdentificationEnvelope<Pag>, IdentificationError> {
        self.pag_envelope_with(graph, |mag| {
            let Some(admg) = mag_to_admg(mag) else {
                return Ok(not_identified(
                    CausalQuery::Response(query.clone()),
                    "completion is not a directed/bidirected MAG",
                ));
            };
            identify_joint_within(
                &admg,
                mag.nodes(),
                query,
                (self.config.max_candidates, self.config.max_examinations),
                self.config.context.as_ref(),
                |from, to| visible(mag, from, to),
            )
        })
    }

    /// Certify one common joint back-door adjustment set on a DAG.
    ///
    /// # Errors
    /// Invalid query, unsupported observation/temporal policy, or graph error.
    pub fn identify_joint_dag_response(
        &self,
        dag: &Dag,
        query: &ResponseQuery,
    ) -> Result<IdentificationResult, IdentificationError> {
        let admg = crate::prepared::dag_to_admg(dag)?;
        self.identify_joint_admg_response(&admg, query)
    }

    /// Certify one common joint back-door set on a **known** ADMG.
    ///
    /// This is ordinary generalized adjustment after mutilating outgoing
    /// directed edges from each treatment (Pearl / Perkovic Def. 20 on a
    /// single graph). It is not MAG/PAG identification: there is no Markov
    /// equivalence class and no visibility witness. A drawn treatment↔outcome
    /// edge, or any other open back-door, refuses that pair.
    ///
    /// # Errors
    /// Invalid query, unsupported observation/temporal policy, or graph error.
    pub fn identify_joint_admg_response(
        &self,
        graph: &Admg,
        query: &ResponseQuery,
    ) -> Result<IdentificationResult, IdentificationError> {
        identify_joint_within(
            graph,
            graph.nodes(),
            query,
            (self.config.max_candidates, self.config.max_examinations),
            self.config.context.as_ref(),
            |_, _| true,
        )
    }
}

// A -> B is visible if a node nonadjacent to B has an arrowhead into A,
// or reaches A along a collider path into A whose internal nodes parent B.
// This is a marked-path search over a PAG's endpoints (collider paths whose internal nodes
// parent `B`), not a directed closure, so it is not one of the graph crate's closure owners.
pub(crate) fn visible(mag: &Pag, from: DenseNodeId, to: DenseNodeId) -> bool {
    let adjacent = |a, b| mag.neighbors(a).any(|(v, _, _)| v == b);
    let parent = |a, b| {
        mag.neighbors(a)
            .any(|(v, at_a, at_b)| v == b && at_a == Endpoint::Tail && at_b == Endpoint::Arrow)
    };
    let mut seen = vec![false; mag.node_count()];
    let mut stack = vec![from];
    seen[from.as_usize()] = true;
    while let Some(current) = stack.pop() {
        for (next, at_current, at_next) in mag.neighbors(current) {
            if at_current != Endpoint::Arrow || next == to {
                continue;
            }
            if !adjacent(next, to) {
                return true;
            }
            if at_next == Endpoint::Arrow && parent(next, to) && !seen[next.as_usize()] {
                seen[next.as_usize()] = true;
                stack.push(next);
            }
        }
    }
    false
}

pub(crate) struct PreparedJointResponse {
    pub query: CausalQuery,
    pub treatments: Vec<antecedent_core::VariableId>,
    pub outcome: antecedent_core::VariableId,
    pub targets: Vec<DenseNodeId>,
    pub y: DenseNodeId,
}

pub(crate) fn prepare_joint_response(
    nodes: &[NodeRef],
    response: &ResponseQuery,
) -> Result<PreparedJointResponse, IdentificationError> {
    response
        .validate()
        .map_err(|_| IdentificationError::unsupported("invalid joint response query"))?;
    if response.temporal.is_some()
        || response.observation != antecedent_core::ObservationSpec::Complete
    {
        return Err(IdentificationError::unsupported(
            "joint adjustment requires complete-observation static data",
        ));
    }
    if !matches!(
        response.functional,
        antecedent_core::ResponseFunctional::InterventionResponse { .. }
    ) {
        return Err(IdentificationError::unsupported(
            "joint adjustment requires InterventionResponse",
        ));
    }
    let treatments = response.functional.treatment_ids();
    if treatments.len() < 2
        || treatments.iter().enumerate().any(|(i, t)| treatments[..i].contains(t))
    {
        return Err(IdentificationError::unsupported(
            "joint response requires at least two distinct treatment targets",
        ));
    }
    let (_, outcome) = response
        .functional
        .primary_pair()
        .ok_or_else(|| IdentificationError::unsupported("joint response has no outcome"))?;
    let dense = |variable| crate::prepared::dense_of_static(nodes, variable);
    let targets: Vec<_> = treatments.iter().copied().map(dense).collect::<Result<_, _>>()?;
    let y = dense(outcome)?;
    Ok(PreparedJointResponse {
        query: CausalQuery::Response(response.clone()),
        treatments,
        outcome,
        targets,
        y,
    })
}

#[cfg(test)]
fn identify_joint(
    graph: &Admg,
    nodes: &[NodeRef],
    response: &ResponseQuery,
    max_candidates: usize,
    max_examinations: u64,
    visible_edge: impl Fn(DenseNodeId, DenseNodeId) -> bool,
) -> Result<IdentificationResult, IdentificationError> {
    identify_joint_within(
        graph,
        nodes,
        response,
        (max_candidates, max_examinations),
        None,
        visible_edge,
    )
}

/// The joint search under the caller's `(candidate ceiling, examination budget)` and context:
/// the enumeration's storage is leased from the context's pool before it is allocated and the
/// context's cancellation is observed at every test.
fn identify_joint_within(
    graph: &Admg,
    nodes: &[NodeRef],
    response: &ResponseQuery,
    (max_candidates, max_examinations): (usize, u64),
    context: Option<&antecedent_core::ExecutionContext>,
    visible_edge: impl Fn(DenseNodeId, DenseNodeId) -> bool,
) -> Result<IdentificationResult, IdentificationError> {
    let PreparedJointResponse { query, treatments, outcome, targets, y } =
        prepare_joint_response(nodes, response)?;
    let descendants = directed_closure(graph, &targets, false);
    let mut seeds = targets.clone();
    seeds.push(y);
    let ancestors = directed_closure(graph, &seeds, true);
    let candidates: Vec<_> = (0..graph.node_count())
        .map(DenseNodeId::from_index)
        .filter(|&v| v != y && !descendants.contains(v) && ancestors.contains(v))
        .collect();
    let context = crate::exact_search::context_or_default(context);
    let mut exact = crate::exact_search::ExactSearch::open(&context)?;
    let backdoor_graphs = targets
        .iter()
        .map(|&target| backdoor_graph(graph, target, &visible_edge))
        .collect::<Result<Vec<_>, _>>()?;
    let mut workspace = DSeparationWorkspace::default();
    // Each treatment is separated from the outcome in its own proper back-door graph while the
    // other treatments are conditioned on.
    let mut separates = |z: &[DenseNodeId]| -> Result<bool, IdentificationError> {
        for (index, &target) in targets.iter().enumerate() {
            let mut conditioned = z.to_vec();
            conditioned.extend(targets.iter().copied().filter(|&v| v != target));
            if !backdoor_graphs[index]
                .is_m_separated(target, y, &conditioned, &mut workspace)
                .map_err(IdentificationError::from)?
            {
                return Ok(false);
            }
        }
        Ok(true)
    };
    // One test on the maximal candidate set decides existence (see the module documentation).
    exact.observe()?;
    if !separates(&candidates)? {
        return Ok(joint_scientifically_unidentified(query, 1));
    }
    // The joint search retains one set. Above the declared ceiling, or when the context cannot
    // lease the enumeration's storage, the subset search is skipped, not the decision: the
    // maximal set just tested is a valid joint adjustment set.
    if !exact.fund_enumeration(candidates.len(), max_candidates, 1)? {
        return Ok(joint_result_bounded(query, nodes, &candidates, &treatments, outcome, 1));
    }
    let search = crate::enum_masks::first_set_by_size(&candidates, 1, max_examinations, |z| {
        exact.observe()?;
        if z == candidates.as_slice() {
            return Ok(true);
        }
        separates(z)
    })?;
    // The budget ended the search for a smaller set; the maximal set is already certified.
    Ok(match search.found {
        Some(z) => joint_result(query, nodes, &z, &treatments, outcome, search.examined),
        None => joint_result_bounded(
            query,
            nodes,
            &candidates,
            &treatments,
            outcome,
            search.examined,
        ),
    })
}

/// Diagnostic code of a joint search for a smaller set that stopped at its budget: the
/// identified result carries the certified maximal set, valid but not minimal.
pub const JOINT_SEARCH_BOUNDED_DIAGNOSTIC_CODE: &str = "identify.joint.search_bounded";

/// The joint search for a smaller set stopped (examination budget, declared candidate ceiling
/// or leasable storage): the certified maximal set is returned, valid but not minimal, with the
/// Execution diagnostic of a bounded search.
fn joint_result_bounded(
    query: CausalQuery,
    nodes: &[NodeRef],
    z: &[DenseNodeId],
    treatments: &[antecedent_core::VariableId],
    outcome: antecedent_core::VariableId,
    examined: u64,
) -> IdentificationResult {
    let mut result = joint_result(query, nodes, z, treatments, outcome, examined);
    result.diagnostics.push(crate::result::search_bounded_diagnostic(
        JOINT_SEARCH_BOUNDED_DIAGNOSTIC_CODE,
        format!(
            "joint adjustment search ran {examined} separation tests before its examination \
             budget, declared candidate ceiling or leasable storage stopped it; the returned \
             set (all {} admissible candidates) separates every treatment from the outcome \
             but a smaller set may exist",
            z.len()
        ),
    ));
    result
}

/// Proven non-ID: the maximal candidate set does not separate, so no joint set exists (not a
/// budget miss).
pub(crate) fn joint_scientifically_unidentified(
    query: CausalQuery,
    examined: u64,
) -> IdentificationResult {
    let mut result = not_identified(
        query,
        "no joint generalized back-door set exists; general response ID not attempted",
    );
    result.performance.candidates_examined = examined;
    result.diagnostics.push(Diagnostic::new(
        "identify.joint.adjustment",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Warning,
        "no joint generalized back-door set found; open back-door or a drawn treatment↔outcome edge",
    ));
    result
}

/// Perkovic Def. 20 on a proposed common set: no descendants of any target in `z`,
/// and one m-separation check per treatment in that treatment's back-door graph.
pub(crate) fn joint_adjustment_holds(
    graph: &Admg,
    targets: &[DenseNodeId],
    y: DenseNodeId,
    z: &[DenseNodeId],
    visible_edge: impl Fn(DenseNodeId, DenseNodeId) -> bool,
) -> Result<bool, IdentificationError> {
    if z.iter().any(|&v| v == y || targets.contains(&v)) {
        return Ok(false);
    }
    let descendants = directed_closure(graph, targets, false);
    if z.iter().any(|&v| descendants.contains(v)) {
        return Ok(false);
    }
    let backdoor_graphs = targets
        .iter()
        .map(|&target| backdoor_graph(graph, target, &visible_edge))
        .collect::<Result<Vec<_>, _>>()?;
    let mut workspace = DSeparationWorkspace::default();
    for (index, &target) in targets.iter().enumerate() {
        let mut conditioned = z.to_vec();
        conditioned.extend(targets.iter().copied().filter(|&v| v != target));
        match backdoor_graphs[index].is_m_separated(target, y, &conditioned, &mut workspace) {
            Ok(true) => {}
            Ok(false) => return Ok(false),
            Err(e) => return Err(IdentificationError::from(e)),
        }
    }
    Ok(true)
}

fn backdoor_graph(
    graph: &Admg,
    target: DenseNodeId,
    visible_edge: &impl Fn(DenseNodeId, DenseNodeId) -> bool,
) -> Result<Admg, IdentificationError> {
    graph
        .copy_filtered(
            antecedent_graph::CopyLabels::StaticIndex { extra: 0 },
            |from, to| from != target || !visible_edge(from, to),
            // The cut removes visible directed edges. Bidirected edges stay.
            |_a, _b| true,
        )
        .map_err(IdentificationError::from)
}

pub(crate) fn joint_result(
    query: CausalQuery,
    nodes: &[NodeRef],
    z: &[DenseNodeId],
    treatments: &[antecedent_core::VariableId],
    outcome: antecedent_core::VariableId,
    examined: u64,
) -> IdentificationResult {
    let adjustment: Arc<[_]> = z
        .iter()
        .map(|v| match nodes[v.as_usize()] {
            NodeRef::Static(variable) => variable,
            _ => unreachable!("static graph"),
        })
        .collect();
    let mut arena = CausalExprArena::new();
    let y_set = arena.intern_var_set([outcome]);
    let z_set = arena.intern_var_set(adjustment.iter().copied());
    let xz_set = arena.intern_var_set(treatments.iter().copied().chain(adjustment.iter().copied()));
    let empty = arena.empty_var_set();
    let observational = arena.empty_intervention_set();
    let conditional =
        arena.intern_distribution(y_set, xz_set, observational, DomainRef::Observational);
    let marginal = arena.intern_distribution(z_set, empty, observational, DomainRef::Observational);
    let factors = arena.intern_list([conditional, marginal]);
    let product = arena.intern(ExprNode::Product(factors));
    let distribution = arena.intern(ExprNode::SumOut { variables: z_set, expr: product });
    let functional = arena
        .intern(ExprNode::Expectation { function: OutcomeExprId::identity(outcome), distribution });
    let estimand = IdentifiedEstimand::backdoor("backdoor.adjustment", adjustment, functional);
    let mut derivation = DerivationTrace::default();
    derivation.push("response.joint_adjustment", format!("certified E[Y|do(X=x)] = sum_z E[Y|X=x,Z=z] P(z) for all {} targets; generalized back-door criterion (Perkovic et al., 2018, Definition 20)", treatments.len()));
    let mut assumptions = antecedent_core::AssumptionSet::default();
    assumptions.push(crate::assumptions::causal_markov("response.joint_adjustment"));
    IdentificationResult::identified(
        query,
        vec![estimand],
        arena,
        derivation,
        assumptions,
        IdentificationPerformanceRecord { candidates_examined: examined, sets_returned: 1 },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generalized::mag_to_admg;
    use antecedent_core::{
        CausalSchemaBuilder, IdentificationStatus, Intervention, ResponseFunctional, Value,
        VariableId,
    };
    use antecedent_graph::{BitSet, GraphWorkspace, MarkedEdge, TieredBackground, WithinTier};

    fn n(i: u32) -> DenseNodeId {
        DenseNodeId::from_raw(i)
    }

    fn joint(t1: VariableId, t2: VariableId, y: VariableId) -> ResponseQuery {
        ResponseQuery::new(ResponseFunctional::InterventionResponse {
            outcome: y,
            interventions: Arc::from([
                Intervention::set(t1, Value::f64(1.0)),
                Intervention::set(t2, Value::f64(1.0)),
            ]),
        })
    }

    #[test]
    fn joint_and_response_delegation_share_context_without_changing_the_claim() {
        let mut dag = Dag::with_variables(4);
        for child in [1, 2, 3] {
            dag.insert_directed(n(0), n(child)).unwrap();
        }
        for parent in [1, 2] {
            dag.insert_directed(n(parent), n(3)).unwrap();
        }
        let query =
            joint(VariableId::from_raw(1), VariableId::from_raw(2), VariableId::from_raw(3));
        for through_response in [false, true] {
            let identify = |context: Option<antecedent_core::ExecutionContext>| {
                if through_response {
                    let mut identifier = crate::ResponseIdentifier::new();
                    identifier.backdoor.config.context = context;
                    let prepared = identifier
                        .prepare_with_assumptions(&dag, antecedent_core::AssumptionSet::new())?;
                    identifier.identify(
                        &prepared,
                        &CausalQuery::Response(query.clone()),
                        &mut crate::IdentificationWorkspace::default(),
                    )
                } else {
                    let mut identifier = GeneralizedAdjustmentIdentifier::new();
                    identifier.config.context = context;
                    identifier.identify_joint_dag_response(&dag, &query)
                }
            };
            let baseline = identify(None).unwrap();
            assert_eq!(baseline.status, IdentificationStatus::NonparametricallyIdentified);
            assert_eq!(baseline.estimands[0].adjustment_set.as_ref(), &[VariableId::from_raw(0)]);
            let funded = identify(Some(antecedent_core::ExecutionContext::for_tests(86))).unwrap();
            assert_eq!(funded.query, baseline.query);
            assert_eq!(funded.status, baseline.status);
            assert_eq!(funded.required_assumptions, baseline.required_assumptions);
            assert_eq!(funded.derivation, baseline.derivation);
            assert_eq!(funded.performance, baseline.performance);
            assert_eq!(funded.estimands, baseline.estimands);
            assert_eq!(funded.estimand_claims, baseline.estimand_claims);
            assert_eq!(funded.diagnostics, baseline.diagnostics);
            let cancelled = antecedent_core::ExecutionContext::for_tests(86);
            cancelled.cancellation.cancel();
            assert_eq!(identify(Some(cancelled)).unwrap_err(), IdentificationError::Cancelled);
            // A context that cannot lease the enumeration's storage skips the search for a
            // smaller set, not the decision: the certified maximal set is returned, flagged.
            let limited =
                antecedent_core::ExecutionContext::for_tests(86).with_memory_limit(Some(1));
            let bounded = identify(Some(limited)).unwrap();
            assert_eq!(bounded.status, baseline.status);
            assert_eq!(bounded.estimands[0].adjustment_set.as_ref(), &[VariableId::from_raw(0)]);
            assert!(crate::envelope::search_truncated(&bounded));
        }
    }

    #[test]
    fn mag_visibility_needs_a_witness_and_accepts_a_collider_path() {
        let mut mag = Pag::with_variables(4);
        mag.insert_marked(MarkedEdge::directed(n(0), n(1))).unwrap();
        assert!(!visible(&mag, n(0), n(1)));
        mag.insert_marked(MarkedEdge::bidirected(n(2), n(0))).unwrap();
        mag.insert_marked(MarkedEdge::directed(n(2), n(1))).unwrap();
        assert!(!visible(&mag, n(0), n(1)));
        mag.insert_marked(MarkedEdge::directed(n(3), n(2))).unwrap();
        assert!(visible(&mag, n(0), n(1)));
    }

    #[test]
    fn mag_visibility_is_the_wrong_object_for_a_known_closure_admg() {
        let schema = CausalSchemaBuilder::new()
            .continuous("z")
            .finish()
            .continuous("t1")
            .finish()
            .continuous("t2")
            .finish()
            .continuous("y")
            .finish()
            .build()
            .unwrap();
        let background = TieredBackground::from_named(
            &schema,
            &[vec!["z"], vec!["t1", "t2"], vec!["y"]],
            WithinTier::CoDetermined,
        )
        .unwrap();
        let pag = background.to_pag(&schema).unwrap();
        let admg = mag_to_admg(&pag).expect("CoDetermined closure is an ancestral ADMG");
        let t1 = n(schema.id_of("t1").unwrap().raw());
        let t2 = n(schema.id_of("t2").unwrap().raw());
        let y = n(schema.id_of("y").unwrap().raw());
        let z = schema.id_of("z").unwrap();
        assert!(admg.bidirected_neighbors(t1).contains(&t2));
        let mut descendants = BitSet::default();
        let mut ws = GraphWorkspace::default();
        admg.descendants_of(&[t1], &mut descendants, &mut ws);
        assert!(!descendants.contains(t2), "walking ↔ as a directed path would put t2 in De(t1)");
        assert!(
            !visible(&pag, t1, y) && !visible(&pag, t2, y),
            "complete earlier→later MAG has no visibility witness — that is MAG-as-MEC, not CoDetermined"
        );

        let query = joint(
            schema.id_of("t1").unwrap(),
            schema.id_of("t2").unwrap(),
            schema.id_of("y").unwrap(),
        );
        let mag_path = identify_joint(
            &admg,
            pag.nodes(),
            &query,
            16,
            crate::GeneralizedAdjustmentConfig::default().max_examinations,
            |from, to| visible(&pag, from, to),
        )
        .unwrap();
        assert_eq!(
            mag_path.status,
            IdentificationStatus::NotIdentified,
            "MAG visibility must not be the CoDetermined license: {:?}",
            mag_path.derivation
        );
        let known = GeneralizedAdjustmentIdentifier::new()
            .identify_joint_admg_response(&admg, &query)
            .unwrap();
        assert_eq!(
            known.status,
            IdentificationStatus::NonparametricallyIdentified,
            "known-ADMG joint adjustment must accept Z={{z}}: {:?}",
            known.derivation
        );
        assert_eq!(known.estimands[0].adjustment_set.as_ref(), &[z]);
    }

    #[test]
    fn drawn_treatment_outcome_bidirected_refuses_that_pair() {
        let mut admg = Admg::with_variables(4);
        admg.insert_directed(n(0), n(1)).unwrap();
        admg.insert_directed(n(0), n(2)).unwrap();
        admg.insert_directed(n(0), n(3)).unwrap();
        admg.insert_directed(n(1), n(3)).unwrap();
        admg.insert_directed(n(2), n(3)).unwrap();
        admg.insert_bidirected(n(1), n(2)).unwrap();
        admg.insert_bidirected(n(1), n(3)).unwrap();
        let query =
            joint(VariableId::from_raw(1), VariableId::from_raw(2), VariableId::from_raw(3));
        let id = GeneralizedAdjustmentIdentifier::new()
            .identify_joint_admg_response(&admg, &query)
            .unwrap();
        assert_eq!(
            id.status,
            IdentificationStatus::NotIdentified,
            "drawn T↔Y is an open back-door: {:?}",
            id.derivation
        );
        assert!(
            id.diagnostics.iter().any(|d| d.kind == antecedent_core::DiagnosticKind::Scientific
                && d.code.as_ref() != crate::generalized::CAPPED_COMPLETION_DIAGNOSTIC_CODE),
            "drawn T↔Y must be a scientific refuse, not a budget miss: {:?}",
            id.diagnostics
        );
        assert!(
            !id.diagnostics.iter().any(|d| {
                d.code.as_ref() == crate::generalized::CAPPED_COMPLETION_DIAGNOSTIC_CODE
            }),
            "drawn T↔Y must not carry the cap diagnostic"
        );
    }

    /// Seventeen common causes of both treatments and the outcome: the only joint set is all
    /// of them, reached last by a search that grows the set size. Stopping after three
    /// separation tests leaves the maximal set, already certified by one test, as the answer:
    /// valid, not minimal, and flagged as the result of a bounded search.
    #[test]
    fn joint_examination_budget_exit_returns_the_certified_maximal_set() {
        let mut admg = Admg::with_variables(20);
        admg.insert_directed(n(0), n(2)).unwrap();
        admg.insert_directed(n(1), n(2)).unwrap();
        for i in 3..20 {
            for child in [n(0), n(1), n(2)] {
                admg.insert_directed(n(i), child).unwrap();
            }
        }
        let query =
            joint(VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2));
        let bounded = identify_joint(&admg, admg.nodes(), &query, 40, 3, |_, _| true).unwrap();
        assert_eq!(bounded.status, IdentificationStatus::NonparametricallyIdentified);
        assert_eq!(bounded.performance.candidates_examined, 3);
        assert_eq!(bounded.estimands[0].adjustment_set.len(), 17);
        let kinds: Vec<_> =
            bounded.diagnostics.iter().map(|d| (d.code.as_ref().to_owned(), d.kind)).collect();
        assert_eq!(
            kinds,
            vec![(
                JOINT_SEARCH_BOUNDED_DIAGNOSTIC_CODE.to_owned(),
                antecedent_core::DiagnosticKind::Execution
            )],
            "an unfinished search for a smaller set carries no scientific negative"
        );
        assert!(crate::envelope::search_truncated(&bounded));

        let full = identify_joint(
            &admg,
            admg.nodes(),
            &query,
            40,
            crate::GeneralizedAdjustmentConfig::default().max_examinations,
            |_, _| true,
        )
        .unwrap();
        assert_eq!(full.status, IdentificationStatus::NonparametricallyIdentified);
        assert_eq!(full.estimands[0].adjustment_set.len(), 17);
        assert!(full.diagnostics.is_empty());
    }

    /// `T1 <-> Y` with thirty common causes: the bidirected edge is an open path no joint set
    /// can block, and one separation test per treatment on the maximal set proves it, where an
    /// exhaustive search would visit `2^30` subsets. The proof is scientific, not a budget miss.
    #[test]
    fn joint_non_identification_is_decided_by_one_test_per_treatment() {
        let causes = 30u32;
        let mut admg = Admg::with_variables(3 + causes);
        admg.insert_directed(n(0), n(2)).unwrap();
        admg.insert_directed(n(1), n(2)).unwrap();
        admg.insert_bidirected(n(0), n(2)).unwrap();
        for i in 0..causes {
            for child in [n(0), n(1), n(2)] {
                admg.insert_directed(n(3 + i), child).unwrap();
            }
        }
        let query =
            joint(VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2));
        let id = identify_joint(&admg, admg.nodes(), &query, usize::MAX, 4, |_, _| true).unwrap();
        assert_eq!(id.status, IdentificationStatus::NotIdentified);
        assert_eq!(id.performance.candidates_examined, 1);
        assert!(id.diagnostics.iter().any(|d| d.kind == antecedent_core::DiagnosticKind::Scientific));
        assert!(!crate::envelope::search_truncated(&id));
    }

    #[test]
    fn joint_candidate_ceiling_keeps_the_certified_set_and_flags_the_unfinished_search() {
        let mut admg = Admg::with_variables(20);
        let t1 = n(0);
        let t2 = n(1);
        let y = n(2);
        admg.insert_directed(t1, y).unwrap();
        admg.insert_directed(t2, y).unwrap();
        for i in 3..20 {
            let z = n(i);
            admg.insert_directed(z, t1).unwrap();
            admg.insert_directed(z, t2).unwrap();
            admg.insert_directed(z, y).unwrap();
        }
        let query =
            joint(VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2));
        let id = GeneralizedAdjustmentIdentifier {
            config: crate::GeneralizedAdjustmentConfig { max_candidates: 16, ..Default::default() },
        }
        .identify_joint_admg_response(&admg, &query)
        .unwrap();
        assert_eq!(
            id.status,
            IdentificationStatus::NonparametricallyIdentified,
            "the ceiling skips the search for a smaller set, not the decision: {:?}",
            id.derivation
        );
        assert_eq!(id.estimands[0].adjustment_set.len(), 17);
        assert!(
            id.diagnostics.iter().any(|d| {
                d.code.as_ref() == JOINT_SEARCH_BOUNDED_DIAGNOSTIC_CODE
                    && d.kind == antecedent_core::DiagnosticKind::Execution
            }),
            "the unfinished search is an execution diagnostic: {:?}",
            id.diagnostics
        );
        assert!(
            !id.diagnostics.iter().any(|d| d.kind == antecedent_core::DiagnosticKind::Scientific),
            "the ceiling must not be stamped scientific: {:?}",
            id.diagnostics
        );
    }
}
