//! Path-restricted natural-effect identification (Avin, Shpitser & Pearl 2005).
//!
//! Decide which treatment out-edges start a selected path and which start a
//! complementary path by directed reachability, and refuse when a recanting
//! witness exists. The edge assignment does not need the paths themselves.
//! When every directed path is selected the path-specific effect is the total
//! effect: surgically delete treatment out-edges that start no selected path
//! and run general ID for the active/control contrast. When a complementary
//! path exists, emit the edge g-formula (Shpitser 2013): the treatment enters
//! each child's factor at the active level if that edge starts a selected path
//! and at the control level otherwise, minus the all-control g-formula.
//!
//! # Recanting witness
//!
//! Avin, Shpitser & Pearl (2005) define a recanting witness for π as a node
//! `W ≠ T` with a directed path `q` from `T` to `W` and two directed paths
//! `s₁`, `s₂` from `W` to `Y` such that `q·s₁ ∈ π` and `q·s₂ ∉ π`: `W`'s
//! response to the shared prefix `q` would be needed at the active level along
//! one continuation and at the control level along the other, a cross-world
//! quantity no observational law pins down. The effect is identified in a DAG
//! iff no witness exists. Taking `W` to be the first node of `q` after `T`
//! shows the witness exists iff some child `C` of `T` starts both a path in π
//! and a path outside it (conversely, a child that starts both is itself a
//! witness with `q = T → C`). The check is therefore exact: it compares the
//! first hops of the selected and the complementary paths. Without a witness,
//! π is exactly the set of paths leaving `T` through an edge that starts a
//! selected path, which is the edge assignment the edge g-formula binds.
//!
//! A node reached from `T` along both kinds of path but through *different*
//! children (`T → A → W → Y` selected, `T → B → W → Y` not) is not a witness:
//! `W` takes `A` from the active world and `B` from the control world through
//! one ordinary factor `P(w | a, b)`.
//!
//! On an ADMG a directed witness still refuses (latent variables cannot remove
//! one). A bidirected edge among the outcome's ancestors replaces the node-wise
//! edge g-formula by Shpitser's (2013) district formula: with `V* = An(Y)` in the
//! graph without the treatment, each side is `Σ_{V* \ Y} Π_D Q[D]`, where `D` runs
//! over the districts of the subgraph on `V*` and the kernel `Q[D] = P(D | do(V \ D))`
//! is identified by the complete ID algorithm, read at the treatment level of the
//! edges that leave the treatment into `D`. A district that receives a selected
//! and a complementary treatment edge is a recanting district: its kernel would be
//! needed at both treatment levels at once, which no observational law pins down,
//! and the effect is refused. A district whose kernel ID cannot identify (a hedge)
//! refuses the effect for the same reason the total effect would.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashSet;
use std::sync::Arc;

use antecedent_core::{
    AverageEffectQuery, CausalQuery, PathSpecificEffectQuery, Value, VariableId,
};
use antecedent_expr::{
    CausalExprArena, ContrastOp, DerivationMeta, DomainRef, EstimandMethod, ExprId, ExprNode,
    IdentifiedEstimand, InterventionAssignment, OutcomeExprId,
};
use antecedent_graph::{Admg, BitSet, Dag, DenseNodeId, GraphWorkspace};

use crate::error::IdentificationError;

/// Diagnostic code of a path-specific effect whose selected path set is empty: the effect is
/// identified, and zero by structure.
pub const PATH_SPECIFIC_STRUCTURAL_ZERO_DIAGNOSTIC_CODE: &str =
    "identify.path_specific.structurally_zero";
use crate::id::{AssignedIds, AssignedProblem, IdIdentifier, identify_assigned_problems};
use crate::identifier::IdentificationWorkspace;
use crate::prepared::PreparedAdmg;
use crate::result::{
    DerivationTrace, IdentificationPerformanceRecord, IdentificationResult, IdentificationStatus,
};

/// Diagnostic code: path enumeration stopped at its budget before the recanting-witness
/// check could run over the complete path set. The `NotIdentified` it accompanies is a
/// search bound, not a statement about the graph.
#[cfg(test)]
pub(crate) const SEARCH_BOUNDED_DIAGNOSTIC_CODE: &str = "identify.path_specific.search_bounded";

/// Path-restricted natural-effect identifier.
#[derive(Clone, Debug, Default)]
pub struct PathSpecificIdentifier {
    /// Underlying general ID engine.
    pub inner: IdIdentifier,
}

impl PathSpecificIdentifier {
    /// Create the identifier.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Prepare a DAG as an ADMG.
    ///
    /// # Errors
    ///
    /// Graph construction failure.
    pub fn prepare_dag(&self, graph: &Dag) -> Result<PreparedAdmg, IdentificationError> {
        self.inner.prepare_dag(graph)
    }

    /// Prepare an ADMG.
    ///
    /// # Errors
    ///
    /// Graph validation failure.
    pub fn prepare(&self, graph: &Admg) -> Result<PreparedAdmg, IdentificationError> {
        self.inner.prepare(graph)
    }

    /// Identify a path-specific natural effect.
    ///
    /// # Errors
    ///
    /// Unsupported query shape, unknown variables, or path enumeration failure.
    pub fn identify(
        &self,
        prepared: &PreparedAdmg,
        query: &CausalQuery,
        workspace: &mut IdentificationWorkspace,
    ) -> Result<IdentificationResult, IdentificationError> {
        let mediation_path;
        let q = match query {
            CausalQuery::PathSpecific(q) => q,
            CausalQuery::Mediation(m) => {
                m.validate()
                    .map_err(|_| IdentificationError::unsupported("invalid mediation query"))?;
                mediation_path = PathSpecificEffectQuery {
                    treatment: m.treatment,
                    outcome: m.outcome,
                    path_nodes: Arc::clone(&m.mediators),
                    control: m.control.clone(),
                    active: m.active.clone(),
                    target_population: m.target_population.clone(),
                    ..PathSpecificEffectQuery::binary(m.treatment, m.outcome)
                };
                &mediation_path
            }
            _ => {
                return Err(IdentificationError::unsupported(
                    "PathSpecificIdentifier requires path-specific or static mediation query",
                ));
            }
        };
        q.validate()
            .map_err(|_| IdentificationError::unsupported("invalid path-specific query"))?;
        self.identify_path_specific(prepared, q, query.clone(), workspace)
    }

    #[allow(
        clippy::too_many_lines,
        reason = "one linear identification pipeline whose stages share the prepared graph and workspace"
    )]
    fn identify_path_specific(
        &self,
        prepared: &PreparedAdmg,
        q: &PathSpecificEffectQuery,
        query: CausalQuery,
        workspace: &mut IdentificationWorkspace,
    ) -> Result<IdentificationResult, IdentificationError> {
        let mut derivation = DerivationTrace::default();
        derivation.push("path_specific", "Avin–Shpitser–Pearl path-restricted natural effect");
        let mut perf = IdentificationPerformanceRecord::default();

        let t = prepared.var_to_dense(q.treatment)?;
        let y = prepared.var_to_dense(q.outcome)?;
        let path_filter: HashSet<DenseNodeId> =
            q.path_nodes.iter().map(|&v| prepared.var_to_dense(v)).collect::<Result<_, _>>()?;

        let admg = prepared.admg();
        // Directed skeleton. Selected versus complementary paths are a reachability
        // question on the treatment's children; the estimand only needs those first hops.
        let dag = admg.directed_part().map_err(IdentificationError::from)?;
        let hops = classify_first_hops(
            &dag,
            t,
            y,
            path_question(&query, &path_filter),
            &mut workspace.graph,
        )?;
        perf.candidates_examined =
            perf.candidates_examined.saturating_add(hops.children_reaching_outcome);

        if hops.selected.is_empty() {
            // No selected path carries the treatment's change to the outcome, so the active and
            // control arms agree on every path and the effect is zero whatever the data say.
            derivation.push(
                "path_specific.empty",
                "no directed path matches the path filter, so the path-specific effect is \
                 structurally zero",
            );
            let mut arena = CausalExprArena::new();
            let outcome = arena.intern_var_set([q.outcome]);
            let empty = arena.empty_var_set();
            let empty_i = arena.empty_intervention_set();
            let law = arena.intern_distribution(outcome, empty, empty_i, DomainRef::Observational);
            let mean = arena.intern(ExprNode::Expectation {
                function: OutcomeExprId::identity(q.outcome),
                distribution: law,
            });
            let functional = arena.intern(ExprNode::Contrast {
                left: mean,
                right: mean,
                op: ContrastOp::Difference,
            });
            arena.set_derivation(
                functional,
                DerivationMeta::rule(
                    "path_specific.empty",
                    Some(Arc::from("a contrast of identical arms: the selected path set is empty")),
                ),
            );
            let estimand = IdentifiedEstimand::new(
                Arc::from(EstimandMethod::PathSpecificNatural.as_str()),
                Arc::from([]),
                Arc::from([]),
                Arc::clone(&q.path_nodes),
                functional,
                None,
            );
            let mut result = IdentificationResult::identified(
                query,
                vec![estimand],
                arena,
                derivation,
                prepared.declared_assumptions().clone(),
                perf,
            );
            result.diagnostics.push(antecedent_core::Diagnostic::new(
                PATH_SPECIFIC_STRUCTURAL_ZERO_DIAGNOSTIC_CODE,
                antecedent_core::DiagnosticKind::Scientific,
                antecedent_core::DiagnosticSeverity::Info,
                "no directed path from the treatment to the outcome matches the path filter, so \
                 the path-specific effect is identified and equals zero",
            ));
            return Ok(result);
        }
        derivation.push(
            "path_specific.paths",
            format!("{} treatment out-edge(s) start a selected path", hops.selected.len()),
        );

        if let Some(w) = recanting_witness(&hops.selected, &hops.complement) {
            derivation.push(
                "path_specific.recanting",
                format!(
                    "recanting witness dense={} (a treatment child starting both a selected and a \
                     complementary path) blocks nonparametric path-specific ID",
                    w.raw()
                ),
            );
            return Ok(IdentificationResult::not_identified(
                query,
                derivation,
                prepared.declared_assumptions().clone(),
                perf,
            ));
        }
        derivation.push("path_specific.recanting", "no recanting witness");

        let keep_out: HashSet<DenseNodeId> = hops.selected.iter().copied().collect();

        // A complementary path means the contrast is not the total effect: the
        // treatment must reach the outcome at the active level along π and at the
        // control level along every other path. General ID on the surgical graph
        // cannot express that (its factors condition on the treatment as a
        // predecessor, so one level would reach every factor), so use the edge
        // g-formula instead.
        if !hops.complement.is_empty() {
            let v_star = outcome_ancestors_without_treatment(admg, t, y);
            if v_star.iter().any(|v| !admg.bidirected_neighbors(v).is_empty()) {
                return district_g_formula(
                    DistrictRequest { prepared, q, query, t, y, keep_out: &keep_out, v_star },
                    derivation,
                    perf,
                    &mut workspace.graph,
                );
            }
            let (arena, functional) = edge_g_formula(prepared, q, t, y, &keep_out)?;
            derivation.push(
                "path_specific.edge_gformula",
                format!(
                    "complementary paths leave through {} treatment out-edge(s): edge g-formula \
                     with the active level on {} treatment out-edge(s) on π and the control \
                     level on the rest",
                    hops.complement.len(),
                    keep_out.len()
                ),
            );
            let estimand = IdentifiedEstimand::new(
                Arc::from(EstimandMethod::PathSpecificNatural.as_str()),
                Arc::from([]),
                Arc::from([]),
                Arc::clone(&q.path_nodes),
                functional,
                None,
            );
            return Ok(IdentificationResult::identified(
                query,
                vec![estimand],
                arena,
                derivation,
                prepared.declared_assumptions().clone(),
                perf,
            ));
        }

        let surgical = surgical_admg(admg, t, &keep_out)?;
        let surgical_prep =
            PreparedAdmg::with_assumptions(surgical, prepared.declared_assumptions().clone())?;
        derivation.push(
            "path_specific.surgery",
            format!(
                "kept {} treatment out-edge(s) on π; deleted complementary out-edges",
                keep_out.len()
            ),
        );

        let ate = AverageEffectQuery::new(
            q.treatment,
            q.outcome,
            Arc::from([]),
            q.control.clone(),
            q.active.clone(),
            q.target_population.clone(),
        );
        let mut id_res = self.inner.identify_ate(&surgical_prep, &ate, workspace)?;
        id_res.derivation.steps.splice(0..0, derivation.steps);
        id_res.performance.candidates_examined =
            id_res.performance.candidates_examined.saturating_add(perf.candidates_examined);
        id_res.query = query;
        if id_res.status == IdentificationStatus::NonparametricallyIdentified {
            for est in &mut id_res.estimands {
                est.method = Arc::from(EstimandMethod::PathSpecificNatural.as_str());
                // Surface path intermediates as mediators metadata.
                est.mediators = Arc::clone(&q.path_nodes);
            }
        }
        Ok(id_res)
    }
}

/// Which directed paths from a treatment child to the outcome are selected.
enum PathQuestion {
    /// Every directed path.
    All,
    /// Paths whose intermediate nodes include every member of the set.
    AllRequired(Vec<DenseNodeId>),
    /// Paths that pass through none of the mediators.
    Avoiding(Vec<DenseNodeId>),
    /// Paths that pass through some mediator.
    Through(Vec<DenseNodeId>),
}

/// Treatment children that start a selected path, and those that start a complementary path.
struct FirstHops {
    selected: Vec<DenseNodeId>,
    complement: Vec<DenseNodeId>,
    /// Treatment children that reach the outcome, the nodes the criterion inspects.
    children_reaching_outcome: u64,
}

fn path_question(query: &CausalQuery, filter: &HashSet<DenseNodeId>) -> PathQuestion {
    if let CausalQuery::Mediation(m) = query {
        use antecedent_core::MediationContrast;
        let mediators: Vec<DenseNodeId> = filter.iter().copied().collect();
        return match m.contrast {
            MediationContrast::Total => PathQuestion::All,
            MediationContrast::Direct | MediationContrast::NaturalDirect => {
                PathQuestion::Avoiding(mediators)
            }
            MediationContrast::Mediated | MediationContrast::NaturalIndirect => {
                PathQuestion::Through(mediators)
            }
        };
    }
    if filter.is_empty() {
        PathQuestion::All
    } else {
        let mut required: Vec<DenseNodeId> = filter.iter().copied().collect();
        required.sort_unstable_by_key(|n| n.raw());
        PathQuestion::AllRequired(required)
    }
}

/// Classify each treatment child by whether a selected path and a complementary path leave
/// through it.
///
/// In a DAG a set of nodes lies on one directed path exactly when reachability totally orders
/// them, and a node lies on some `C → Y` path exactly when `C` reaches it and it reaches `Y`.
/// Both are reachability questions. The edge g-formula and the surgical total effect bind one
/// level per treatment out-edge, so they need this classification and not the path list.
fn classify_first_hops(
    dag: &Dag,
    t: DenseNodeId,
    y: DenseNodeId,
    question: PathQuestion,
    ws: &mut GraphWorkspace,
) -> Result<FirstHops, IdentificationError> {
    let mut an_y = BitSet::with_len(dag.node_count());
    dag.ancestors_of(&[y], &mut an_y, ws);
    let children: Vec<DenseNodeId> = dag.children(t).to_vec();
    let mut selected = Vec::new();
    let mut complement = Vec::new();
    let mut children_reaching_outcome = 0u64;

    match question {
        PathQuestion::All => {
            for &child in &children {
                if an_y.contains(child) {
                    children_reaching_outcome += 1;
                    selected.push(child);
                }
            }
        }
        PathQuestion::AllRequired(required) => {
            let mut open_without = Vec::with_capacity(required.len());
            for &node in &required {
                open_without.push(ancestors_avoiding(dag, y, &[node], ws)?);
            }
            for &child in &children {
                if !an_y.contains(child) {
                    continue;
                }
                children_reaching_outcome += 1;
                if required_on_one_path(dag, child, &required, &an_y) {
                    selected.push(child);
                    let misses_one = required
                        .iter()
                        .enumerate()
                        .any(|(i, node)| *node != child && open_without[i].contains(child));
                    if misses_one {
                        complement.push(child);
                    }
                } else {
                    complement.push(child);
                }
            }
        }
        PathQuestion::Avoiding(mediators) => {
            return classify_mediation(dag, y, &children, &mediators, &an_y, false, ws);
        }
        PathQuestion::Through(mediators) => {
            return classify_mediation(dag, y, &children, &mediators, &an_y, true, ws);
        }
    }

    Ok(FirstHops { selected, complement, children_reaching_outcome })
}

fn classify_mediation(
    dag: &Dag,
    y: DenseNodeId,
    children: &[DenseNodeId],
    mediators: &[DenseNodeId],
    an_y: &BitSet,
    want_mediated: bool,
    ws: &mut GraphWorkspace,
) -> Result<FirstHops, IdentificationError> {
    let mut selected = Vec::new();
    let mut complement = Vec::new();
    let mut children_reaching_outcome = 0u64;
    let open = ancestors_avoiding(dag, y, mediators, ws)?;
    let mediator_set: HashSet<DenseNodeId> = mediators.iter().copied().collect();
    for &child in children {
        if !an_y.contains(child) {
            continue;
        }
        children_reaching_outcome += 1;
        let unmediated = !mediator_set.contains(&child) && open.contains(child);
        let mediated = mediator_set.contains(&child)
            || mediators.iter().any(|m| an_y.contains(*m) && dag.reaches(child, *m));
        let (is_selected, is_complement) =
            if want_mediated { (mediated, unmediated) } else { (unmediated, mediated) };
        if is_selected {
            selected.push(child);
        }
        if is_complement {
            complement.push(child);
        }
    }
    Ok(FirstHops { selected, complement, children_reaching_outcome })
}

/// Ancestors of `y`, including `y`, along directed paths that touch no node in `avoid`.
fn ancestors_avoiding(
    dag: &Dag,
    y: DenseNodeId,
    avoid: &[DenseNodeId],
    ws: &mut GraphWorkspace,
) -> Result<BitSet, IdentificationError> {
    let filtered = dag.copy_filtered(|from, to| !avoid.contains(&from) && !avoid.contains(&to))?;
    let mut an = BitSet::with_len(filtered.node_count());
    filtered.ancestors_of(&[y], &mut an, ws);
    Ok(an)
}

/// Whether some directed path from `start` to `Y` visits every node of `required`.
///
/// `an_y` is the ancestor set of `Y` in `dag` (so membership means "reaches `Y`").
fn required_on_one_path(
    dag: &Dag,
    start: DenseNodeId,
    required: &[DenseNodeId],
    an_y: &BitSet,
) -> bool {
    if required.is_empty() {
        return an_y.contains(start);
    }
    for &node in required {
        if !an_y.contains(node) || !dag.reaches(start, node) {
            return false;
        }
    }
    for i in 0..required.len() {
        for j in (i + 1)..required.len() {
            let (a, b) = (required[i], required[j]);
            if a != b && !dag.reaches(a, b) && !dag.reaches(b, a) {
                return false;
            }
        }
    }
    true
}

/// A recanting witness for π, if one exists: a child of the treatment that
/// starts both a selected and a complementary directed path to the outcome.
///
/// By the argument in the module docs this is exactly the Avin–Shpitser–Pearl
/// criterion: the witness exists iff some child of `T` starts both kinds of path.
fn recanting_witness(selected: &[DenseNodeId], complement: &[DenseNodeId]) -> Option<DenseNodeId> {
    let complement: HashSet<DenseNodeId> = complement.iter().copied().collect();
    selected
        .iter()
        .copied()
        .filter(|child| complement.contains(child))
        .min_by_key(|child| child.raw())
}

/// Ancestors of the outcome (the outcome included) reached without passing through the
/// treatment: the nodes whose factors the path-specific functional multiplies.
fn outcome_ancestors_without_treatment(admg: &Admg, t: DenseNodeId, y: DenseNodeId) -> BitSet {
    let n = admg.node_count();
    let (mut seeds, mut active) = (BitSet::with_len(n), BitSet::full(n));
    seeds.insert(y);
    active.remove(t);
    admg.ancestors_within(&seeds, &active, None, &mut GraphWorkspace::default())
}

/// What the district formula reads.
struct DistrictRequest<'a> {
    prepared: &'a PreparedAdmg,
    q: &'a PathSpecificEffectQuery,
    query: CausalQuery,
    t: DenseNodeId,
    y: DenseNodeId,
    /// Treatment children whose edge starts a selected path.
    keep_out: &'a HashSet<DenseNodeId>,
    /// `An(Y)` without the treatment.
    v_star: BitSet,
}

/// District formula for a path-specific natural effect on an ADMG whose outcome ancestors
/// carry latent confounding (Shpitser 2013).
///
/// Each district of the subgraph on `V*` contributes its kernel `Q[D] = P(D | do(V \ D))`,
/// identified by ID with the treatment bound to the level of the edges that enter `D`: the
/// active level when a selected treatment edge enters `D`, the control level otherwise. The
/// reference side is the ordinary interventional mean at the control level. A district
/// entered by a selected and a complementary treatment edge is a recanting district, and a
/// district whose kernel ID cannot identify leaves the effect unidentified.
fn district_g_formula(
    request: DistrictRequest<'_>,
    mut derivation: DerivationTrace,
    mut perf: IdentificationPerformanceRecord,
    ws: &mut GraphWorkspace,
) -> Result<IdentificationResult, IdentificationError> {
    let DistrictRequest { prepared, q, query, t, y, keep_out, v_star } = request;
    let admg = prepared.admg();
    let n = admg.node_count();
    let active = crate::intervention_support::require_set_value(&q.active, "path-specific")?;
    let control = crate::intervention_support::require_set_value(&q.control, "path-specific")?;
    let children = admg.children(t).to_vec();
    let districts = prepared.c_components(&v_star);

    let mut problems = Vec::with_capacity(districts.len() + 1);
    for district in &districts {
        let selected = children.iter().any(|c| district.contains(*c) && keep_out.contains(c));
        let complementary = children.iter().any(|c| district.contains(*c) && !keep_out.contains(c));
        if selected && complementary {
            derivation.push(
                "path_specific.recanting_district",
                format!(
                    "district {{{}}} receives a treatment edge on a selected path and one off it: \
                     its kernel is needed at both treatment levels, so a recanting district \
                     blocks nonparametric path-specific ID",
                    district.iter().map(|d| d.raw().to_string()).collect::<Vec<_>>().join(",")
                ),
            );
            return Ok(IdentificationResult::not_identified(
                query,
                derivation,
                prepared.declared_assumptions().clone(),
                perf,
            ));
        }
        let level = if selected { active.clone() } else { control.clone() };
        let mut intervened = BitSet::full(n);
        intervened.difference_with(district);
        problems.push(AssignedProblem {
            y: district.clone(),
            x: intervened,
            assign: Arc::from([(t, level)]),
        });
    }
    derivation.push(
        "path_specific.districts",
        format!(
            "{} district(s) of the outcome ancestors; kernels identified by general ID at the \
             treatment level of the edges that enter each district",
            districts.len()
        ),
    );

    problems.push(control_reference_problem(n, t, y, control.clone()));

    let mut arena = CausalExprArena::new();
    let functionals = match identify_assigned_problems(
        prepared,
        &mut arena,
        &mut derivation,
        &mut perf,
        ws,
        &problems,
    )? {
        AssignedIds::Identified(functionals) => functionals,
        AssignedIds::Hedge(hedge) => {
            derivation.push(
                "path_specific.district_hedge",
                format!(
                    "a district kernel has a hedge (F of {} nodes, F' of {} nodes) and is not \
                     identified",
                    hedge.f.len(),
                    hedge.f_prime.len()
                ),
            );
            return Ok(IdentificationResult::not_identified(
                query,
                derivation,
                prepared.declared_assumptions().clone(),
                perf,
            ));
        }
    };
    let functional = district_contrast(prepared, q, y, &v_star, &mut arena, &functionals)?;
    derivation.push(
        "path_specific.district_gformula",
        "district formula with the active level on districts entered by a selected treatment edge",
    );
    let estimand = IdentifiedEstimand::new(
        Arc::from(EstimandMethod::PathSpecificNatural.as_str()),
        Arc::from([]),
        Arc::from([]),
        Arc::clone(&q.path_nodes),
        functional,
        None,
    );
    Ok(IdentificationResult::identified(
        query,
        vec![estimand],
        arena,
        derivation,
        crate::id::with_causal_markov(prepared, "path_specific.district_gformula"),
        perf,
    ))
}

/// The reference side is the ordinary outcome law under control treatment,
/// rather than a district kernel intervening on every node outside a district.
fn control_reference_problem(
    nodes: usize,
    t: DenseNodeId,
    y: DenseNodeId,
    control: Value,
) -> AssignedProblem {
    let (mut outcome, mut treatment) = (BitSet::with_len(nodes), BitSet::with_len(nodes));
    outcome.insert(y);
    treatment.insert(t);
    AssignedProblem { y: outcome, x: treatment, assign: Arc::from([(t, control)]) }
}

/// Assemble the contrast after all district kernels and the control reference
/// have been identified. Expression interning order is the district order.
fn district_contrast(
    prepared: &PreparedAdmg,
    q: &PathSpecificEffectQuery,
    y: DenseNodeId,
    v_star: &BitSet,
    arena: &mut CausalExprArena,
    functionals: &[ExprId],
) -> Result<ExprId, IdentificationError> {
    let Some((&reference, kernels)) = functionals.split_last() else {
        return Err(IdentificationError::unsupported("path-specific district formula is empty"));
    };

    let body = if kernels.len() == 1 {
        kernels[0]
    } else {
        let list = arena.intern_list(kernels.to_vec());
        arena.intern(ExprNode::Product(list))
    };
    let summed: Vec<VariableId> = v_star
        .iter()
        .filter(|&v| v != y)
        .map(|v| prepared.dense_to_var(v))
        .collect::<Result<_, _>>()?;
    let distribution = if summed.is_empty() {
        body
    } else {
        let variables = arena.intern_var_set(summed);
        arena.intern(ExprNode::SumOut { variables, expr: body })
    };
    let left = arena.intern(ExprNode::Expectation {
        function: OutcomeExprId::identity(q.outcome),
        distribution,
    });
    let right = arena.intern(ExprNode::Expectation {
        function: OutcomeExprId::identity(q.outcome),
        distribution: reference,
    });
    let contrast = arena.intern(ExprNode::Contrast { left, right, op: ContrastOp::Difference });
    arena.set_derivation(
        contrast,
        DerivationMeta::rule(
            "path_specific.district_gformula",
            Some(Arc::from(
                "product of district kernels read at the treatment level of the edges entering \
                 each district, minus the all-control interventional mean",
            )),
        ),
    );
    arena.simplify(contrast).map_err(IdentificationError::expression)
}

/// Edge g-formula for a path-specific natural effect (Avin, Shpitser & Pearl
/// 2005; Shpitser 2013) on a graph without latent confounding among the
/// outcome's ancestors.
///
/// With `V* = An(Y)` in the graph with the treatment's incoming edges removed,
/// minus the treatment itself, each side of the contrast is
///
/// `E[Y] = Σ_{v* \ y} Π_{V ∈ V*} P(v | pa(v))`,
///
/// where a factor whose parents include the treatment `T` binds `T` to the
/// active level if `T → V` starts a path in π (`keep_out`) and to the control
/// level otherwise. The reference side binds every treatment edge to the
/// control level (the ordinary g-formula for `do(T = control)`). The per-factor
/// binding is carried as that factor's intervention set, so any evaluator of the
/// expression IR (the discrete plug-in and its shared Dirichlet row law
/// included) evaluates exactly this functional.
///
/// Callers must have ruled out a recanting witness: then every treatment child
/// that is an ancestor of `Y` lies on π-paths only or on complementary paths
/// only, so the edge assignment is well defined and equals the path-specific
/// effect.
fn edge_g_formula(
    prepared: &PreparedAdmg,
    q: &PathSpecificEffectQuery,
    t: DenseNodeId,
    y: DenseNodeId,
    keep_out: &HashSet<DenseNodeId>,
) -> Result<(CausalExprArena, ExprId), IdentificationError> {
    let admg = prepared.admg();
    // Ancestors of Y (Y included) reached without passing through T.
    let n = admg.node_count();
    let (mut seeds, mut active) = (BitSet::with_len(n), BitSet::full(n));
    seeds.insert(y);
    active.remove(t);
    let v_star = admg.ancestors_within(
        &seeds,
        &active,
        None,
        &mut antecedent_graph::GraphWorkspace::default(),
    );
    // `P(v | pa(v))` is the district factor only when `v` is a singleton
    // district; latent confounding would need Shpitser's district formula.
    if v_star.iter().any(|v| !admg.bidirected_neighbors(v).is_empty()) {
        return Err(IdentificationError::unsupported(
            "path-specific edge g-formula requires no bidirected edge on an ancestor of the \
             outcome other than the treatment (the district-level formula for latent confounding \
             is not implemented)",
        ));
    }

    let t_var = prepared.dense_to_var(t)?;
    let active = crate::intervention_support::require_set_value(&q.active, "path-specific")?;
    let control = crate::intervention_support::require_set_value(&q.control, "path-specific")?;

    let mut arena = CausalExprArena::new();
    let empty_i = arena.empty_intervention_set();
    let order: Vec<DenseNodeId> =
        prepared.topo().iter().copied().filter(|&v| v_star.contains(v)).collect();
    let side = |edge_level: &dyn Fn(DenseNodeId) -> Value,
                arena: &mut CausalExprArena|
     -> Result<ExprId, IdentificationError> {
        let mut factors = Vec::with_capacity(order.len());
        for &vi in &order {
            let parents = admg.parents(vi);
            let cond: Vec<VariableId> =
                parents.iter().map(|&p| prepared.dense_to_var(p)).collect::<Result<_, _>>()?;
            let (intervention, domain) = if parents.contains(&t) {
                let assignment = InterventionAssignment { variable: t_var, value: edge_level(vi) };
                (
                    crate::intervention_support::intern_assignments(arena, [assignment])?,
                    DomainRef::Interventional,
                )
            } else {
                (empty_i, DomainRef::Observational)
            };
            let variables = arena.intern_var_set([prepared.dense_to_var(vi)?]);
            let conditioned_on = arena.intern_var_set(cond);
            factors.push(arena.intern_distribution(
                variables,
                conditioned_on,
                intervention,
                domain,
            ));
        }
        let body = if factors.len() == 1 {
            factors[0]
        } else {
            let list = arena.intern_list(factors);
            arena.intern(ExprNode::Product(list))
        };
        let summed: Vec<VariableId> = order
            .iter()
            .filter(|&&v| v != y)
            .map(|&v| prepared.dense_to_var(v))
            .collect::<Result<_, _>>()?;
        let distribution = if summed.is_empty() {
            body
        } else {
            let variables = arena.intern_var_set(summed);
            arena.intern(ExprNode::SumOut { variables, expr: body })
        };
        Ok(arena.intern(ExprNode::Expectation {
            function: OutcomeExprId::identity(q.outcome),
            distribution,
        }))
    };
    let path_level = |vi: DenseNodeId| {
        if keep_out.contains(&vi) { active.clone() } else { control.clone() }
    };
    let left = side(&path_level, &mut arena)?;
    let right = side(&|_| control.clone(), &mut arena)?;
    let contrast = arena.intern(ExprNode::Contrast { left, right, op: ContrastOp::Difference });
    arena.set_derivation(
        contrast,
        DerivationMeta::rule(
            "path_specific.edge_gformula",
            Some(Arc::from(
                "active level on treatment edges that start a selected path; control elsewhere",
            )),
        ),
    );
    let functional = arena.simplify(contrast).map_err(IdentificationError::expression)?;
    Ok((arena, functional))
}

fn surgical_admg(
    admg: &Admg,
    treatment: DenseNodeId,
    keep_out: &HashSet<DenseNodeId>,
) -> Result<Admg, IdentificationError> {
    admg.copy_filtered(
        antecedent_graph::CopyLabels::Preserve,
        |from, to| from != treatment || keep_out.contains(&to),
        // Path surgery deletes directed treatment edges only. Bidirected edges stay.
        |_a, _b| true,
    )
    .map_err(IdentificationError::from)
}

#[cfg(test)]
mod tests {
    use antecedent_core::{Intervention, MediationContrast, MediationQuery, Value, VariableId};
    use antecedent_graph::DenseNodeId;

    use super::*;
    use crate::identifier::IdentificationWorkspace;
    use crate::result::IdentificationStatus;

    fn chain_with_direct() -> Dag {
        // T → M → Y and T → Y
        let mut dag = Dag::with_variables(3);
        let t = DenseNodeId::from_raw(0);
        let m = DenseNodeId::from_raw(1);
        let y = DenseNodeId::from_raw(2);
        dag.insert_directed(t, m).unwrap();
        dag.insert_directed(m, y).unwrap();
        dag.insert_directed(t, y).unwrap();
        dag
    }

    #[test]
    fn mediated_only_path_identifies() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../conformance/identify/path_specific/expected.json"
        ))
        .unwrap();
        assert_eq!(fixture["cases"][0]["method"].as_str(), Some("path_specific.natural"));
        let dag = chain_with_direct();
        let id = PathSpecificIdentifier::new();
        let prep = id.prepare_dag(&dag).unwrap();
        let q = PathSpecificEffectQuery::binary(VariableId::from_raw(0), VariableId::from_raw(2))
            .with_path_nodes([VariableId::from_raw(1)]);
        let cq = CausalQuery::PathSpecific(q);
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &cq, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        assert_eq!(res.estimands[0].method.as_ref(), EstimandMethod::PathSpecificNatural.as_str());
    }

    /// A truncated path enumeration must never yield an "identified" verdict.
    ///
    /// The recanting-witness rule concludes identifiability from the *absence* of a witness.
    /// On the graph below `w` is a genuine recanting witness (`w→a` on π, `w→b` off it), so
    /// the correct answer is `NotIdentified`. Before the fix, `max_paths = 1` enumerated only
    /// the π path, left the complement empty, found no witness, and returned
    /// `NonparametricallyIdentified` — licensing an estimand Avin–Shpitser–Pearl forbids.
    #[test]
    fn truncated_path_enumeration_never_claims_identification() {
        // 0=t 1=w 2=a 3=b 4=c 5=y;  t→c→y, t→w→b→y, t→w→a→y.  π = paths through `a`.
        let mut dag = Dag::with_variables(6);
        for (u, v) in [(0, 4), (4, 5), (0, 1), (1, 3), (3, 5), (1, 2), (2, 5)] {
            dag.insert_directed(DenseNodeId::from_raw(u), DenseNodeId::from_raw(v)).unwrap();
        }
        let id = PathSpecificIdentifier::new();
        let prep = id.prepare_dag(&dag).unwrap();

        for max_paths in [1usize, 2, 3, 64] {
            let q =
                PathSpecificEffectQuery::binary(VariableId::from_raw(0), VariableId::from_raw(5))
                    .with_path_nodes([VariableId::from_raw(2)])
                    .with_max_paths(max_paths);
            let cq = CausalQuery::PathSpecific(q);
            let mut ws = IdentificationWorkspace::default();
            let res = id.identify(&prep, &cq, &mut ws).unwrap();
            assert_eq!(
                res.status,
                IdentificationStatus::NotIdentified,
                "max_paths={max_paths} must not identify past a recanting witness"
            );
            let detail = witness(&res).unwrap_or_default();
            assert!(
                detail.contains("dense=1"),
                "max_paths={max_paths} must name the witness, not stop on a path budget: {detail}"
            );
        }

        // The untruncated run must reach the recanting rule itself, not merely fail closed.
        let q = PathSpecificEffectQuery::binary(VariableId::from_raw(0), VariableId::from_raw(5))
            .with_path_nodes([VariableId::from_raw(2)]);
        let cq = CausalQuery::PathSpecific(q);
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &cq, &mut ws).unwrap();
        assert!(
            res.derivation
                .steps
                .iter()
                .any(|s| s.rule.contains("recanting") && s.detail.contains("blocks")),
            "full enumeration should cite the recanting witness, got {:?}",
            res.derivation.steps
        );
    }

    /// Four layers of three fully connected nodes give 3^4 = 81 directed t→y paths.
    /// The effect along the first-layer node is identified: every selected path leaves
    /// through that one child, and every other first-layer child starts only a
    /// complementary path. Reachability answers that without enumerating the 81 paths,
    /// including when `max_paths` / `max_len` could not hold them.
    #[test]
    fn path_explosion_is_decided_by_reachability() {
        // 0=t, 1..=12 four layers of three, 13=y.
        let mut dag = Dag::with_variables(14);
        let layer = |k: u32| (1 + 3 * k)..(4 + 3 * k);
        for v in layer(0) {
            dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(v)).unwrap();
        }
        for k in 0..3 {
            for u in layer(k) {
                for v in layer(k + 1) {
                    dag.insert_directed(DenseNodeId::from_raw(u), DenseNodeId::from_raw(v))
                        .unwrap();
                }
            }
        }
        for u in layer(3) {
            dag.insert_directed(DenseNodeId::from_raw(u), DenseNodeId::from_raw(13)).unwrap();
        }
        // Default max_paths is 64 and max_len 3 cannot hold a 6-node path. Both used to
        // refuse. The edge assignment is active on node 1 and control on nodes 2 and 3.
        let q = PathSpecificEffectQuery::binary(VariableId::from_raw(0), VariableId::from_raw(13))
            .with_path_nodes([VariableId::from_raw(1)])
            .with_max_len(3);
        let res = identify_on(&dag, &CausalQuery::PathSpecific(q));
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        assert!(
            res.diagnostics.iter().all(|d| d.code.as_ref() != SEARCH_BOUNDED_DIAGNOSTIC_CODE),
            "{:?}",
            res.diagnostics
        );
        assert!(has_rule(&res, "path_specific.edge_gformula"));
        let (left, right) = contrast_levels(&res);
        assert_eq!(left, vec![(1, 1.0), (2, 0.0), (3, 0.0)]);
        assert_eq!(right, vec![(1, 0.0), (2, 0.0), (3, 0.0)]);
    }

    fn has_rule(res: &IdentificationResult, rule: &str) -> bool {
        res.derivation.steps.iter().any(|s| s.rule.as_ref() == rule)
    }

    /// With a complementary direct path the functional is the edge g-formula:
    /// the mediator factor binds the treatment to the active level and the
    /// outcome factor binds it to the control level on the left side.
    #[test]
    fn complementary_path_uses_edge_g_formula_levels() {
        let dag = chain_with_direct();
        let id = PathSpecificIdentifier::new();
        let prep = id.prepare_dag(&dag).unwrap();
        let q = PathSpecificEffectQuery::binary(VariableId::from_raw(0), VariableId::from_raw(2))
            .with_path_nodes([VariableId::from_raw(1)]);
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &CausalQuery::PathSpecific(q), &mut ws).unwrap();
        assert!(has_rule(&res, "path_specific.edge_gformula"), "{:?}", res.derivation.steps);

        let (left, right) = contrast_levels(&res);
        assert_eq!(left, vec![(1, 1.0), (2, 0.0)], "M at active, Y at control");
        assert_eq!(right, vec![(1, 0.0), (2, 0.0)], "reference side all control");
    }

    /// `(factor variable, bound treatment level)` pairs, sorted by variable.
    type FactorLevels = Vec<(u32, f64)>;

    /// Factor levels on each side of the edge g-formula contrast.
    fn contrast_levels(res: &IdentificationResult) -> (FactorLevels, FactorLevels) {
        let ExprNode::Contrast { left, right, .. } = res.arena.node(res.estimands[0].functional)
        else {
            panic!("expected a contrast");
        };
        let levels = |root: ExprId| {
            let mut out = Vec::new();
            let mut stack = vec![root];
            while let Some(id) = stack.pop() {
                match res.arena.node(id) {
                    ExprNode::Distribution { variables, intervention, .. } => {
                        for a in res.arena.intervention_assignments(*intervention) {
                            out.push((
                                res.arena.var_set(*variables)[0].raw(),
                                a.value.as_f64().unwrap(),
                            ));
                        }
                    }
                    ExprNode::Product(list) => stack.extend(res.arena.list(*list)),
                    ExprNode::SumOut { expr, .. } | ExprNode::IntegralOut { expr, .. } => {
                        stack.push(*expr);
                    }
                    ExprNode::Expectation { distribution, .. } => stack.push(*distribution),
                    other => panic!("unexpected node {other:?}"),
                }
            }
            out.sort_by(|a, b| a.0.cmp(&b.0));
            out
        };
        (levels(*left), levels(*right))
    }

    fn dag_from(n: u32, edges: &[(u32, u32)]) -> Dag {
        let mut dag = Dag::with_variables(n);
        for &(u, v) in edges {
            dag.insert_directed(DenseNodeId::from_raw(u), DenseNodeId::from_raw(v)).unwrap();
        }
        dag
    }

    fn identify_on(dag: &Dag, query: &CausalQuery) -> IdentificationResult {
        let id = PathSpecificIdentifier::new();
        let prep = id.prepare_dag(dag).unwrap();
        let mut ws = IdentificationWorkspace::default();
        id.identify(&prep, query, &mut ws).unwrap()
    }

    fn path_query(t: u32, y: u32, path_nodes: &[u32]) -> CausalQuery {
        CausalQuery::PathSpecific(
            PathSpecificEffectQuery::binary(VariableId::from_raw(t), VariableId::from_raw(y))
                .with_path_nodes(
                    path_nodes.iter().map(|&v| VariableId::from_raw(v)).collect::<Vec<_>>(),
                ),
        )
    }

    fn mediation_query(
        t: u32,
        y: u32,
        mediators: &[u32],
        contrast: MediationContrast,
    ) -> CausalQuery {
        CausalQuery::Mediation(MediationQuery::binary(
            VariableId::from_raw(t),
            VariableId::from_raw(y),
            mediators.iter().map(|&v| VariableId::from_raw(v)).collect::<Vec<_>>(),
            contrast,
        ))
    }

    fn witness(res: &IdentificationResult) -> Option<String> {
        res.derivation
            .steps
            .iter()
            .find(|s| s.rule.as_ref() == "path_specific.recanting" && s.detail.contains("blocks"))
            .map(|s| s.detail.to_string())
    }

    /// `T → A → W → Y` selected, `T → B → W → Y` not. `W` is on both paths but
    /// they leave `T` through different children, so `W` is not a recanting
    /// witness: `Σ P(a|t₁) P(b|t₀) P(w|a,b) P(y|w)` identifies the effect.
    #[test]
    fn shared_descendant_through_distinct_children_identifies() {
        // 0=T 1=A 2=B 3=W 4=Y
        let dag = dag_from(5, &[(0, 1), (1, 3), (0, 2), (2, 3), (3, 4)]);
        let res = identify_on(&dag, &path_query(0, 4, &[1]));
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        assert!(witness(&res).is_none(), "{:?}", res.derivation.steps);
        assert!(has_rule(&res, "path_specific.edge_gformula"));
        let (left, right) = contrast_levels(&res);
        assert_eq!(left, vec![(1, 1.0), (2, 0.0)], "A at active, B at control");
        assert_eq!(right, vec![(1, 0.0), (2, 0.0)]);
    }

    /// Natural direct effect with a post-mediator node: `T → M1 → W → Y`,
    /// `T → W → Y`, `T → Y`, mediator `M1`. `W` is on a mediated and an
    /// unmediated path, but `T`'s children `M1`, `W`, `Y` each start one kind
    /// only, so both natural effects are identified.
    #[test]
    fn mediation_with_post_mediator_node_identifies_both_natural_effects() {
        // 0=T 1=M1 2=W 3=Y
        let dag = dag_from(4, &[(0, 1), (1, 2), (0, 2), (2, 3), (0, 3)]);
        let direct =
            identify_on(&dag, &mediation_query(0, 3, &[1], MediationContrast::NaturalDirect));
        assert_eq!(direct.status, IdentificationStatus::NonparametricallyIdentified);
        assert!(has_rule(&direct, "path_specific.edge_gformula"));
        let (left, _) = contrast_levels(&direct);
        assert_eq!(left, vec![(1, 0.0), (2, 1.0), (3, 1.0)], "M1 control; W and Y active");

        let indirect =
            identify_on(&dag, &mediation_query(0, 3, &[1], MediationContrast::NaturalIndirect));
        assert_eq!(indirect.status, IdentificationStatus::NonparametricallyIdentified);
        let (left, _) = contrast_levels(&indirect);
        assert_eq!(left, vec![(1, 1.0), (2, 0.0), (3, 0.0)], "M1 active; W and Y control");
    }

    /// `T → L → M → Y`, `T → L → Y`, `T → M → Y`: `L` is a child of `T` that
    /// starts a mediated and an unmediated path, a genuine recanting witness for
    /// both natural effects and for the path set through `M`.
    #[test]
    fn treatment_child_starting_both_kinds_of_path_refuses() {
        // 0=T 1=L 2=M 3=Y
        let dag = dag_from(4, &[(0, 1), (1, 2), (1, 3), (0, 2), (2, 3)]);
        for query in [
            path_query(0, 3, &[2]),
            mediation_query(0, 3, &[2], MediationContrast::NaturalIndirect),
            mediation_query(0, 3, &[2], MediationContrast::NaturalDirect),
        ] {
            let res = identify_on(&dag, &query);
            assert_eq!(res.status, IdentificationStatus::NotIdentified);
            let detail = witness(&res).expect("recanting witness step");
            assert!(detail.contains("dense=1"), "L is the witness: {detail}");
        }
    }

    /// A selection that is not closed under the path edges: only
    /// `T → A → R → Z → Y` is selected, so `A` starts it and the unselected
    /// `T → A → R → Y`.
    #[test]
    fn selection_splitting_below_a_treatment_child_refuses() {
        // 0=T 1=A 2=B 3=R 4=Z 5=Y
        let dag = dag_from(6, &[(0, 1), (0, 2), (1, 3), (2, 3), (3, 5), (3, 4), (4, 5)]);
        let res = identify_on(&dag, &path_query(0, 5, &[1, 4]));
        assert_eq!(res.status, IdentificationStatus::NotIdentified);
        assert!(witness(&res).expect("witness").contains("dense=1"));
    }

    /// Every case frozen in `conformance/identify/path_specific`.
    #[test]
    fn frozen_identification_cases() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../conformance/identify/path_specific/expected.json"
        ))
        .unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let id = case["id"].as_str().unwrap();
            let mut names: Vec<String> = Vec::new();
            let mut index = |name: &str| -> u32 {
                let at = names.iter().position(|n| n == name).unwrap_or_else(|| {
                    names.push(name.to_owned());
                    names.len() - 1
                });
                u32::try_from(at).unwrap()
            };
            let edges: Vec<(u32, u32)> = case["dag"]
                .as_array()
                .unwrap()
                .iter()
                .map(|e| {
                    let (u, v) = e.as_str().unwrap().split_once("->").unwrap();
                    (index(u.trim()), index(v.trim()))
                })
                .collect();
            let path_nodes: Vec<u32> = case["included_path_nodes"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| index(v.as_str().unwrap()))
                .collect();
            let (t, y) = (index("T"), index("Y"));
            let witness_node = case["witness"].as_str().map(&mut index);
            let dag = dag_from(u32::try_from(names.len()).unwrap(), &edges);
            let res = identify_on(&dag, &path_query(t, y, &path_nodes));
            match case["status"].as_str().unwrap() {
                "identified" => {
                    assert_eq!(
                        res.status,
                        IdentificationStatus::NonparametricallyIdentified,
                        "{id}"
                    );
                    if let Some(method) = case["method"].as_str() {
                        assert_eq!(res.estimands[0].method.as_ref(), method, "{id}");
                    }
                }
                "not_identified" => {
                    assert_eq!(res.status, IdentificationStatus::NotIdentified, "{id}");
                    let w = witness_node.expect("not_identified case names its witness");
                    let detail = witness(&res).unwrap_or_default();
                    assert!(detail.contains(&format!("dense={w} ")), "{id}: {detail}");
                }
                other => panic!("{id}: unknown status {other}"),
            }
        }
    }

    /// A mediator-outcome bidirected edge puts the mediator (off the natural direct effect)
    /// and the outcome (on it) in one district entered by both kinds of treatment edge:
    /// a recanting district, refused rather than answered with a node-wise product.
    #[test]
    fn mediator_outcome_confounding_is_a_recanting_district() {
        let mut admg = Admg::with_variables(3);
        let (t, m, y) =
            (DenseNodeId::from_raw(0), DenseNodeId::from_raw(1), DenseNodeId::from_raw(2));
        admg.insert_directed(t, m).unwrap();
        admg.insert_directed(m, y).unwrap();
        admg.insert_directed(t, y).unwrap();
        admg.insert_bidirected(m, y).unwrap();
        let id = PathSpecificIdentifier::new();
        let prep = id.prepare(&admg).unwrap();
        let q = PathSpecificEffectQuery::binary(VariableId::from_raw(0), VariableId::from_raw(2))
            .with_path_nodes([VariableId::from_raw(1)]);
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &CausalQuery::PathSpecific(q), &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NotIdentified);
        assert!(has_rule(&res, "path_specific.recanting_district"), "{:?}", res.derivation.steps);
    }

    #[test]
    fn all_paths_identifies_total() {
        let dag = chain_with_direct();
        let id = PathSpecificIdentifier::new();
        let prep = id.prepare_dag(&dag).unwrap();
        let q = PathSpecificEffectQuery::binary(VariableId::from_raw(0), VariableId::from_raw(2));
        let cq = CausalQuery::PathSpecific(q);
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &cq, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        assert!(!has_rule(&res, "path_specific.edge_gformula"), "no complement: total effect");
    }

    #[test]
    fn set_interventions_required() {
        let dag = chain_with_direct();
        let id = PathSpecificIdentifier::new();
        let prep = id.prepare_dag(&dag).unwrap();
        let mut q =
            PathSpecificEffectQuery::binary(VariableId::from_raw(0), VariableId::from_raw(2));
        q.control = Intervention::set(VariableId::from_raw(0), Value::f64(0.0));
        let cq = CausalQuery::PathSpecific(q);
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &cq, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
    }
}
