//! Joint response adjustment using the generalized back-door criterion.
//!
//! Uses Definition 20 of Perkovic et al. (JMLR 18, 2018): no descendants
//! of any target in Z, and each target's back-door paths blocked by Z plus
//! the other targets. This is sufficient, not complete general response ID.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    CausalQuery, Diagnostic, DiagnosticKind, DiagnosticSeverity, NodeRef, ResponseQuery,
};
use antecedent_expr::{CausalExprArena, DomainRef, ExprNode, OutcomeExprId};
use antecedent_graph::{Admg, Cpdag, DSeparationWorkspace, Dag, DenseNodeId, Endpoint, Pag};

use crate::generalized::{capped_completion_result, directed_closure, mag_to_admg, not_identified};
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
            identify_joint(&admg, mag.nodes(), query, self.config.max_candidates, |from, to| {
                visible(mag, from, to)
            })
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
        let mut admg =
            Admg::with_variables(u32::try_from(dag.node_count()).expect("node count fits"));
        for edge in dag.edges() {
            if let Some((from, to)) = edge.parent_child() {
                admg.insert_directed(from, to)?;
            }
        }
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
        identify_joint(graph, graph.nodes(), query, self.config.max_candidates, |_, _| true)
    }
}

// A -> B is visible if a node nonadjacent to B has an arrowhead into A,
// or reaches A along a collider path into A whose internal nodes parent B.
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
    let dense = |variable| {
        nodes
            .iter()
            .position(|n| *n == NodeRef::Static(variable))
            .map(|i| DenseNodeId::from_raw(u32::try_from(i).expect("node index fits")))
            .ok_or(IdentificationError::UnknownVariable { id: variable })
    };
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

fn identify_joint(
    graph: &Admg,
    nodes: &[NodeRef],
    response: &ResponseQuery,
    max_candidates: usize,
    visible_edge: impl Fn(DenseNodeId, DenseNodeId) -> bool,
) -> Result<IdentificationResult, IdentificationError> {
    let PreparedJointResponse { query, treatments, outcome, targets, y } =
        prepare_joint_response(nodes, response)?;
    let descendants = directed_closure(graph, &targets, false);
    let mut seeds = targets.clone();
    seeds.push(y);
    let ancestors = directed_closure(graph, &seeds, true);
    let candidates: Vec<_> = (0..graph.node_count())
        .map(|i| DenseNodeId::from_raw(u32::try_from(i).expect("node index fits")))
        .filter(|&v| v != y && !descendants.contains(v) && ancestors.contains(v))
        .collect();
    if candidates.len() > max_candidates {
        return Ok(capped_completion_result(query, candidates.len(), max_candidates));
    }
    let backdoor_graphs = targets
        .iter()
        .map(|&target| backdoor_graph(graph, target, &visible_edge))
        .collect::<Result<Vec<_>, _>>()?;
    let mut examined = 0u64;
    let mut found = None;
    let mut workspace = DSeparationWorkspace::default();
    for size in 0..=candidates.len() {
        let mut error = None;
        crate::enum_masks::for_each_mask_of_size(&candidates, size, |z| {
            examined += 1;
            for (index, &target) in targets.iter().enumerate() {
                let mut conditioned = z.to_vec();
                conditioned.extend(targets.iter().copied().filter(|&v| v != target));
                match backdoor_graphs[index].is_m_separated(target, y, &conditioned, &mut workspace)
                {
                    Ok(true) => {}
                    Ok(false) => return false,
                    Err(e) => {
                        error = Some(IdentificationError::from(e));
                        return true;
                    }
                }
            }
            found = Some(z.to_vec());
            true
        });
        if let Some(e) = error {
            return Err(e);
        }
        if found.is_some() {
            break;
        }
    }
    let Some(z) = found else {
        return Ok(joint_scientifically_unidentified(query, examined));
    };
    Ok(joint_result(query, nodes, &z, &treatments, outcome, examined))
}

/// Proven non-ID after an exhaustive joint search (not a budget miss).
pub(crate) fn joint_scientifically_unidentified(
    query: CausalQuery,
    examined: u64,
) -> IdentificationResult {
    let mut result = not_identified(
        query,
        "no joint generalized back-door set found; general response ID not attempted",
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
    let mut cut = Admg::with_variables(u32::try_from(graph.node_count()).expect("node count fits"));
    for i in 0..graph.node_count() {
        let u = DenseNodeId::from_raw(u32::try_from(i).expect("node index fits"));
        for &v in graph.children(u) {
            if u != target || !visible_edge(u, v) {
                cut.insert_directed(u, v)?;
            }
        }
        for &v in graph.bidirected_neighbors(u) {
            if u < v {
                cut.insert_bidirected(u, v)?;
            }
        }
    }
    Ok(cut)
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
    let conditional = arena.intern(ExprNode::Distribution {
        variables: y_set,
        conditioned_on: xz_set,
        intervention: observational,
        domain: DomainRef::Observational,
    });
    let marginal = arena.intern(ExprNode::Distribution {
        variables: z_set,
        conditioned_on: empty,
        intervention: observational,
        domain: DomainRef::Observational,
    });
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
        let mag_path =
            identify_joint(&admg, pag.nodes(), &query, 16, |from, to| visible(&pag, from, to))
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
        assert_ne!(id.status, IdentificationStatus::Undetermined);
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

    #[test]
    fn joint_candidate_cap_is_undetermined_not_scientific() {
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
        let id = GeneralizedAdjustmentIdentifier::new()
            .identify_joint_admg_response(&admg, &query)
            .unwrap();
        assert_eq!(
            id.status,
            IdentificationStatus::Undetermined,
            "budget cap must not be NotIdentified: {:?}",
            id.derivation
        );
        assert!(
            id.diagnostics.iter().any(|d| {
                d.code.as_ref() == crate::generalized::CAPPED_COMPLETION_DIAGNOSTIC_CODE
                    && d.kind == antecedent_core::DiagnosticKind::Execution
            }),
            "cap must be an execution diagnostic: {:?}",
            id.diagnostics
        );
        assert!(
            !id.diagnostics.iter().any(|d| d.kind == antecedent_core::DiagnosticKind::Scientific),
            "cap must not be stamped scientific: {:?}",
            id.diagnostics
        );
    }
}
