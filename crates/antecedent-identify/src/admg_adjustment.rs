//! Adjustment identification of a response functional on a latent-confounded ADMG.
//!
//! A bidirected edge records a latent common cause, so an observed covariate set is a valid
//! adjustment for a treatment/outcome pair only when it m-separates the pair in the proper
//! back-door graph (Shpitser, VanderWeele and Robins 2010, adjustment criterion; Perkovic et
//! al. 2018 for the constructive set) and contains no forbidden node. The proper back-door
//! graph deletes the first edge of every proper causal path from the treatment, and the
//! forbidden set is the descendants of the treatment's causal-path children. A valid set
//! exists exactly when the ancestral candidate set `An({T, Y})` restricted to the allowed
//! nodes separates the pair there (van der Zander, Liskiewicz and Textor 2014), so one
//! m-separation test per pair decides existence.
//!
//! A response that names several pairs (a Jacobian or a directional derivative) needs one
//! adjustment set that is valid for every pair, because the estimator regresses the
//! outcomes once on the treatments and that set. The candidates are each pair's own
//! constructive set and the union of them restricted to the nodes every pair allows; the
//! first that is valid for all pairs is returned. When none is, the response is not
//! identified by adjustment on this ADMG. That is not a statement about identifiability
//! in general: the general ID algorithm may identify it through a functional that is not
//! an adjustment (front-door), and the caller decides what to do with that case.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, CausalQuery, DerivativeScale, ResponseFunctional, ResponseQuery,
    TemporalEffectQuery, Value, VariableId,
};
use antecedent_expr::{CausalExprArena, IdentifiedEstimand};
use antecedent_graph::{
    Admg, CopyLabels, Cpdag, DSeparationWorkspace, DenseNodeId, Pag, TemporalAdmg,
    UnrolledTemporalAdmg,
};

use antecedent_expr::EstimandMethod;

use crate::envelope::IdentificationEnvelope;
use crate::error::IdentificationError;
use crate::generalized::{
    GeneralizedAdjustmentIdentifier, directed_closure, mag_dense_to_var, mag_to_admg, mag_to_confounded_admg,
    not_identified, pag_var_to_dense,
};
use crate::response::{append_observation_assumptions, response_pairs};
use crate::result::{DerivationTrace, IdentificationPerformanceRecord, IdentificationResult};
use crate::temporal_admg::{TemporalAdmgWindowPlan, plan_temporal_admg_window};

/// One treatment/outcome pair's graph facts.
struct PairGeometry {
    treatment: VariableId,
    outcome: VariableId,
    t: DenseNodeId,
    y: DenseNodeId,
    /// The ADMG with the first edge of every proper causal path from `t` removed.
    proper_backdoor: Admg,
    /// Nodes no adjustment set may contain: descendants of the causal-path children of `t`.
    forbidden: Vec<bool>,
    /// The ancestral candidate set: `An({t, y})` minus the forbidden nodes, `t` and `y`.
    constructive: Vec<DenseNodeId>,
}

/// How an ADMG's dense nodes map to the query's variables.
struct NodeMap<'a> {
    dense: &'a dyn Fn(VariableId) -> Result<DenseNodeId, IdentificationError>,
    variable: &'a dyn Fn(DenseNodeId) -> Result<VariableId, IdentificationError>,
}

fn geometry(
    admg: &Admg,
    nodes: &NodeMap<'_>,
    treatment: VariableId,
    outcome: VariableId,
) -> Result<PairGeometry, IdentificationError> {
    let t = (nodes.dense)(treatment)?;
    let y = (nodes.dense)(outcome)?;
    let n = admg.node_count();
    let descendants_of_t = directed_closure(admg, &[t], false);
    let ancestors_of_y = directed_closure(admg, &[y], true);
    let ancestors_of_pair = directed_closure(admg, &[t, y], true);
    // Children of the treatment on a proper causal path to the outcome, and everything
    // downstream of them: conditioning there would block or bias the causal effect.
    let causal_children: Vec<DenseNodeId> =
        admg.children(t).iter().copied().filter(|child| ancestors_of_y.contains(*child)).collect();
    let mediator_roots: Vec<DenseNodeId> = DenseNodeId::all(n)
        .filter(|&node| {
            node != t && descendants_of_t.contains(node) && ancestors_of_y.contains(node)
        })
        .collect();
    let forbidden_closure = directed_closure(admg, &mediator_roots, false);
    let forbidden: Vec<bool> =
        DenseNodeId::all(n).map(|node| forbidden_closure.contains(node)).collect();

    let mut proper_backdoor = Admg::with_variables(u32::try_from(n).map_err(|_| {
        IdentificationError::InvariantViolated { message: "graph node count exceeds u32" }
    })?);
    for source in DenseNodeId::all(n) {
        for &child in admg.children(source) {
            if source == t && causal_children.contains(&child) {
                continue;
            }
            proper_backdoor.insert_directed(source, child)?;
        }
        for &other in admg.bidirected_neighbors(source) {
            if other.raw() > source.raw() {
                proper_backdoor.insert_bidirected(source, other)?;
            }
        }
    }
    let constructive = DenseNodeId::all(n)
        .filter(|&node| {
            node != t
                && node != y
                && ancestors_of_pair.contains(node)
                && !forbidden[node.as_usize()]
        })
        .collect();
    Ok(PairGeometry { treatment, outcome, t, y, proper_backdoor, forbidden, constructive })
}

/// Whether `z` is a valid adjustment set for the pair.
fn valid_for(
    pair: &PairGeometry,
    z: &[DenseNodeId],
    workspace: &mut DSeparationWorkspace,
) -> Result<bool, IdentificationError> {
    if z.iter().any(|&node| node == pair.t || node == pair.y || pair.forbidden[node.as_usize()]) {
        return Ok(false);
    }
    Ok(pair.proper_backdoor.is_m_separated(pair.t, pair.y, z, workspace)?)
}

/// Identify a response functional on `admg` by one common adjustment set.
///
/// The returned result carries one back-door estimand per treatment/outcome pair, all with
/// the same adjustment set, so a regression on the treatments and that set estimates every
/// pair. A result with no estimands and status `NotIdentified` means no adjustment set
/// exists on this ADMG for at least one pair; general ID may still identify the response.
///
/// # Errors
///
/// A variable missing from the graph, an invalid response query, or a graph error.
pub fn identify_admg_response_adjustment(
    admg: &Admg,
    query: &ResponseQuery,
) -> Result<IdentificationResult, IdentificationError> {
    let dense = |variable: VariableId| {
        admg.dense_of_static(variable).ok_or(IdentificationError::UnknownVariable { id: variable })
    };
    let variable = |node: DenseNodeId| {
        admg.static_variable(node)
            .ok_or(IdentificationError::UnknownVariable { id: VariableId::from_raw(node.raw()) })
    };
    adjustment_on(admg, query, &NodeMap { dense: &dense, variable: &variable })
}

/// Identify a derivative or Jacobian over every DAG completion of a CPDAG.
///
/// Each completion is identified by one adjustment set valid for every treatment/outcome
/// pair the functional names, so a completion whose pairs need different sets is reported
/// as not identified by adjustment rather than estimated with a set that is invalid for one
/// of them.
///
/// # Errors
///
/// Invalid query, unknown variable or graph errors.
pub fn identify_cpdag_derivative_pairs(
    cpdag: &Cpdag,
    query: &ResponseQuery,
) -> Result<IdentificationEnvelope<antecedent_graph::Dag>, IdentificationError> {
    GeneralizedAdjustmentIdentifier::new().cpdag_envelope_with(cpdag, |dag| {
        let admg = crate::prepared::dag_to_admg(dag)?;
        identify_admg_response_adjustment(&admg, query)
    })
}

/// Identify a derivative or Jacobian over every MAG completion of a PAG.
///
/// Each completion is read through the generalized adjustment criterion of Perkovic et al.
/// (2018): the MAG is amenable to adjustment for a pair only when every causal path from the
/// treatment starts with a visible edge, and a set is then valid when it m-separates the pair
/// in the proper back-door graph of the MAG and holds no forbidden node. One set valid for
/// every pair the functional names is searched exactly as on an ADMG. A completion with an
/// invisible first edge of a causal path, or with no common set, is reported as not
/// identified by adjustment; that is a refusal of this route, not a proof that general ID
/// cannot identify it.
///
/// # Errors
///
/// Invalid query, unknown variable or graph errors.
pub fn identify_pag_derivative_pairs(
    pag: &Pag,
    query: &ResponseQuery,
) -> Result<IdentificationEnvelope<Pag>, IdentificationError> {
    GeneralizedAdjustmentIdentifier::new().pag_envelope_with(pag, |mag| {
        let Some(admg) = mag_to_admg(mag) else {
            return Ok(not_identified(
                CausalQuery::Response(query.clone()),
                "completion is not a directed/bidirected MAG; adjustment was not attempted",
            ));
        };
        let dense = |variable: VariableId| pag_var_to_dense(mag, variable);
        let variable = |node: DenseNodeId| mag_dense_to_var(mag, node);
        for (treatment, outcome) in response_pairs(&query.functional)? {
            let t = dense(treatment)?;
            let ancestors_of_y = directed_closure(&admg, &[dense(outcome)?], true);
            if admg.children(t).iter().any(|&child| {
                ancestors_of_y.contains(child) && !crate::joint_response::visible(mag, t, child)
            }) {
                return Ok(not_identified(
                    CausalQuery::Response(query.clone()),
                    "MAG is not adjustment amenable: a causal path starts with an invisible \
                     edge; no adjustment set identifies this effect",
                ));
            }
        }
        adjustment_on(&admg, query, &NodeMap { dense: &dense, variable: &variable })
    })
}

/// A derivative response identified by adjustment on the unrolled window of a temporal ADMG.
#[derive(Clone, Debug)]
pub struct TemporalAdmgAdjustment {
    /// Identification over the window's coordinates; its adjustment set holds window
    /// coordinates (`period * variable_count + variable`).
    pub result: IdentificationResult,
    /// Where the policy, the outcome and the history sit in the window.
    pub plan: TemporalAdmgWindowPlan,
    /// The unrolled window the result is stated on.
    pub window: UnrolledTemporalAdmg,
}

/// Identify a derivative or Jacobian at one horizon of a temporal ADMG by adjustment on its
/// unrolled window.
///
/// `probe` is the unit pulse contrast at that horizon under the response's temporal policy
/// (the query the temporal routes identify a horizon with); it fixes the window, the treated
/// period and the outcome period. Every treatment of `query` is read at the treated period and
/// every outcome at the outcome period, and the window ADMG, in which a cause that precedes the
/// window is a latent common cause of its in-window effects, is searched for one adjustment
/// set valid for every pair. A result with status `NotIdentified` means no such set exists on
/// the window; the general ID algorithm may still identify the contrast.
///
/// # Errors
///
/// A treatment, outcome or probe the template does not hold, a window the graph crate cannot
/// number, or an invalid query.
pub fn identify_temporal_admg_response_adjustment(
    template: &TemporalAdmg,
    probe: &TemporalEffectQuery,
    query: &ResponseQuery,
    variable_count: u32,
) -> Result<TemporalAdmgAdjustment, IdentificationError> {
    let plan = plan_temporal_admg_window(template, probe)?;
    let window = template
        .unroll_window_over(plan.periods, variable_count)
        .map_err(|error| IdentificationError::msg(error.to_string()))?;
    let treated_period = plan.treated.first().copied().ok_or(IdentificationError::unsupported(
        "a temporal derivative needs a treated period in its policy",
    ))?;
    let at = |variable: VariableId, period: u32| {
        window
            .coordinate(variable, period)
            .ok_or(IdentificationError::UnknownVariable { id: variable })
    };
    let mut treatments: Vec<VariableId> = Vec::new();
    for variable in query.functional.treatment_ids() {
        let coordinate = at(variable, treated_period)?;
        if !treatments.contains(&coordinate) {
            treatments.push(coordinate);
        }
    }
    let mut outcomes: Vec<VariableId> = Vec::new();
    for variable in query.functional.outcome_ids() {
        let coordinate = at(variable, plan.outcome_period)?;
        if !outcomes.contains(&coordinate) {
            outcomes.push(coordinate);
        }
    }
    let window_query = ResponseQuery::new(ResponseFunctional::Jacobian {
        at: vec![0.0; treatments.len()].into(),
        treatments: treatments.into(),
        outcomes: outcomes.into(),
        scale: DerivativeScale::Identity,
    });
    let mut result = identify_admg_response_adjustment(window.admg(), &window_query)?;
    result.query = CausalQuery::Response(query.clone());
    Ok(TemporalAdmgAdjustment { result, plan, window })
}

/// Largest front-door mediator set the search tries.
const FRONT_DOOR_MAX_MEDIATORS: usize = 3;

/// Whether `m` satisfies the front-door criterion on `admg`: it intercepts every directed path
/// from `t` to `y`, no back-door path leaves `t` into `m`, and every back-door path from `m` to
/// `y` is blocked by `t`. Bidirected edges (latent common causes) stay in every mutilated graph.
fn is_front_door_set(
    admg: &Admg,
    t: DenseNodeId,
    y: DenseNodeId,
    m: &[DenseNodeId],
    workspace: &mut DSeparationWorkspace,
) -> Result<bool, IdentificationError> {
    let without_m = admg.copy_filtered(
        CopyLabels::StaticIndex { extra: 0 },
        |from, to| !m.contains(&from) && !m.contains(&to),
        |_, _| true,
    )?;
    if without_m.reaches(t, y) {
        return Ok(false);
    }
    let t_mutilated =
        admg.copy_filtered(CopyLabels::StaticIndex { extra: 0 }, |from, _| from != t, |_, _| true)?;
    for &mediator in m {
        if !t_mutilated.is_m_separated(t, mediator, &[], workspace)? {
            return Ok(false);
        }
    }
    let m_mutilated = admg.copy_filtered(
        CopyLabels::StaticIndex { extra: 0 },
        |from, _| !m.contains(&from),
        |_, _| true,
    )?;
    for &mediator in m {
        if !m_mutilated.is_m_separated(mediator, y, &[t], workspace)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// The smallest front-door mediator set of `treatment -> outcome` on the ADMG, if any.
fn front_door_set_on(
    admg: &Admg,
    nodes: &NodeMap<'_>,
    treatment: VariableId,
    outcome: VariableId,
) -> Result<Option<Vec<VariableId>>, IdentificationError> {
    let t = (nodes.dense)(treatment)?;
    let y = (nodes.dense)(outcome)?;
    let descendants = directed_closure(admg, &[t], false);
    let ancestors = directed_closure(admg, &[y], true);
    let mut pool: Vec<DenseNodeId> = DenseNodeId::all(admg.node_count())
        .filter(|&node| {
            node != t && node != y && descendants.contains(node) && ancestors.contains(node)
        })
        .collect();
    if pool.len() > 12 {
        pool.retain(|node| admg.children(t).contains(node));
    }
    let mut workspace = DSeparationWorkspace::default();
    for size in 1..=FRONT_DOOR_MAX_MEDIATORS.min(pool.len()) {
        let mut chosen: Vec<usize> = (0..size).collect();
        loop {
            let candidate: Vec<DenseNodeId> = chosen.iter().map(|&i| pool[i]).collect();
            if is_front_door_set(admg, t, y, &candidate, &mut workspace)? {
                return candidate
                    .iter()
                    .map(|&node| (nodes.variable)(node))
                    .collect::<Result<Vec<_>, _>>()
                    .map(Some);
            }
            // Next combination in lexicographic order.
            let mut position = size;
            while position > 0 && chosen[position - 1] == pool.len() - size + position - 1 {
                position -= 1;
            }
            if position == 0 {
                break;
            }
            chosen[position - 1] += 1;
            for later in position..size {
                chosen[later] = chosen[later - 1] + 1;
            }
        }
    }
    Ok(None)
}

/// The identified result of a front-door mediator set for the pair of a single-pair
/// derivative: a `frontdoor` estimand over the mediators.
fn front_door_result(
    query: &ResponseQuery,
    treatment: VariableId,
    outcome: VariableId,
    mediators: Vec<VariableId>,
) -> IdentificationResult {
    let mut assumptions = AssumptionSet::default();
    assumptions.push(crate::assumptions::causal_markov("frontdoor.admg"));
    append_observation_assumptions(query, &mut assumptions);
    let vars: Arc<[VariableId]> = mediators.into();
    let mut arena = CausalExprArena::new();
    let functional =
        arena.frontdoor_ate(treatment, outcome, &vars, Value::f64(1.0), Value::f64(0.0));
    let mut derivation = DerivationTrace::default();
    derivation.push(
        "frontdoor.criterion",
        format!(
            "the {} mediator(s) intercept every directed path, no back-door path leaves the \
             treatment into them, and the treatment blocks every back-door path from them to the \
             outcome (latent common causes kept as bidirected edges)",
            vars.len()
        ),
    );
    IdentificationResult::identified(
        CausalQuery::Response(query.clone()),
        vec![IdentifiedEstimand::frontdoor("frontdoor", vars, functional)],
        arena,
        derivation,
        assumptions,
        IdentificationPerformanceRecord { candidates_examined: 1, sets_returned: 1 },
    )
}

/// Front-door identification of a single-pair derivative on an ADMG, for the case that no
/// adjustment set exists: `Ok(None)` when no front-door set of at most three mediators exists.
///
/// # Errors
///
/// A query that is not a single-pair derivative, a missing variable or a graph error.
pub fn identify_admg_front_door_derivative(
    admg: &Admg,
    query: &ResponseQuery,
) -> Result<Option<IdentificationResult>, IdentificationError> {
    let pairs = response_pairs(&query.functional)?;
    let &[(treatment, outcome)] = pairs.as_slice() else {
        return Ok(None);
    };
    let dense = |variable: VariableId| {
        admg.dense_of_static(variable).ok_or(IdentificationError::UnknownVariable { id: variable })
    };
    let variable = |node: DenseNodeId| {
        admg.static_variable(node)
            .ok_or(IdentificationError::UnknownVariable { id: VariableId::from_raw(node.raw()) })
    };
    let nodes = NodeMap { dense: &dense, variable: &variable };
    Ok(front_door_set_on(admg, &nodes, treatment, outcome)?
        .map(|mediators| front_door_result(query, treatment, outcome, mediators)))
}

/// Replace every completion of a PAG response envelope that general ID identifies but no
/// adjustment reaches, by its front-door identification when one exists. A completion with no
/// front-door set keeps its general-ID result.
///
/// # Errors
///
/// A graph error while testing a completion.
pub fn upgrade_id_only_derivative_cases(
    envelope: &mut IdentificationEnvelope<Pag>,
    query: &ResponseQuery,
) -> Result<(), IdentificationError> {
    let pairs = response_pairs(&query.functional)?;
    let &[(treatment, outcome)] = pairs.as_slice() else {
        return Ok(());
    };
    for case in &mut envelope.cases {
        let id_only =
            case.result.estimands.first().is_some_and(|estimand| {
                estimand.method_kind().ok() == Some(EstimandMethod::GeneralId)
            });
        if !id_only {
            continue;
        }
        let Some(admg) = mag_to_confounded_admg(&case.graph) else { continue };
        let mag = &case.graph;
        let dense = |variable: VariableId| pag_var_to_dense(mag, variable);
        let variable = |node: DenseNodeId| mag_dense_to_var(mag, node);
        let nodes = NodeMap { dense: &dense, variable: &variable };
        if let Some(mediators) = front_door_set_on(&admg, &nodes, treatment, outcome)? {
            case.result = front_door_result(query, treatment, outcome, mediators);
        }
    }
    Ok(())
}

/// Front-door identification of a single-pair derivative at one horizon of a temporal ADMG, on
/// its unrolled window, for the case that no adjustment set exists: `Ok(None)` when no
/// front-door set of at most three mediators exists on the window. The mediators are window
/// coordinates.
///
/// # Errors
///
/// As [`identify_temporal_admg_response_adjustment`].
pub fn identify_temporal_admg_front_door_derivative(
    template: &TemporalAdmg,
    probe: &TemporalEffectQuery,
    query: &ResponseQuery,
    variable_count: u32,
) -> Result<Option<TemporalAdmgAdjustment>, IdentificationError> {
    let pairs = response_pairs(&query.functional)?;
    let &[(treatment, outcome)] = pairs.as_slice() else {
        return Ok(None);
    };
    let plan = plan_temporal_admg_window(template, probe)?;
    let window = template
        .unroll_window_over(plan.periods, variable_count)
        .map_err(|error| IdentificationError::msg(error.to_string()))?;
    let treated_period = plan.treated.first().copied().ok_or(IdentificationError::unsupported(
        "a temporal derivative needs a treated period in its policy",
    ))?;
    let t = window
        .coordinate(treatment, treated_period)
        .ok_or(IdentificationError::UnknownVariable { id: treatment })?;
    let y = window
        .coordinate(outcome, plan.outcome_period)
        .ok_or(IdentificationError::UnknownVariable { id: outcome })?;
    let window_query = ResponseQuery::new(ResponseFunctional::PointDerivative {
        outcome: y,
        treatment: t,
        at: 0.0,
        order: 1,
        scale: DerivativeScale::Identity,
    });
    let Some(mut result) = identify_admg_front_door_derivative(window.admg(), &window_query)?
    else {
        return Ok(None);
    };
    result.query = CausalQuery::Response(query.clone());
    Ok(Some(TemporalAdmgAdjustment { result, plan, window }))
}

fn adjustment_on(
    admg: &Admg,
    query: &ResponseQuery,
    nodes: &NodeMap<'_>,
) -> Result<IdentificationResult, IdentificationError> {
    query
        .validate()
        .map_err(|error| IdentificationError::InvalidQuery { message: error.to_string() })?;
    let causal_query = CausalQuery::Response(query.clone());
    let pairs = response_pairs(&query.functional)?;
    let geometries = pairs
        .iter()
        .map(|&(treatment, outcome)| geometry(admg, nodes, treatment, outcome))
        .collect::<Result<Vec<_>, _>>()?;

    let mut assumptions = AssumptionSet::default();
    assumptions.push(crate::assumptions::causal_markov("generalized.adjustment.admg"));
    append_observation_assumptions(query, &mut assumptions);

    // Candidate sets: each pair's own constructive set, then the union of those restricted
    // to the nodes every pair allows.
    let mut candidates: Vec<Vec<DenseNodeId>> =
        geometries.iter().map(|pair| pair.constructive.clone()).collect();
    let mut union: Vec<DenseNodeId> = geometries
        .iter()
        .flat_map(|pair| pair.constructive.iter().copied())
        .filter(|node| {
            geometries
                .iter()
                .all(|pair| *node != pair.t && *node != pair.y && !pair.forbidden[node.as_usize()])
        })
        .collect();
    union.sort_unstable();
    union.dedup();
    candidates.push(union);

    let mut workspace = DSeparationWorkspace::default();
    let mut examined = 0_u64;
    let mut chosen: Option<Vec<DenseNodeId>> = None;
    for mut candidate in candidates {
        candidate.sort_unstable();
        candidate.dedup();
        examined += 1;
        let mut all_valid = true;
        for pair in &geometries {
            if !valid_for(pair, &candidate, &mut workspace)? {
                all_valid = false;
                break;
            }
        }
        if all_valid {
            chosen = Some(candidate);
            break;
        }
    }

    let performance = IdentificationPerformanceRecord {
        candidates_examined: examined,
        sets_returned: u64::from(chosen.is_some()),
    };
    let Some(chosen) = chosen else {
        let mut derivation = DerivationTrace::default();
        derivation.push(
            "generalized.adjustment.admg",
            "no common adjustment set m-separates every treatment/outcome pair in its proper \
             back-door graph: the ancestral candidate set leaves an open non-causal path through \
             a latent common cause",
        );
        return Ok(IdentificationResult::not_identified(
            causal_query,
            derivation,
            assumptions,
            performance,
        ));
    };

    let z_vars: Arc<[VariableId]> =
        chosen.iter().map(|&node| (nodes.variable)(node)).collect::<Result<Vec<_>, _>>()?.into();
    let label =
        if z_vars.is_empty() { "generalized.adjustment.empty" } else { "generalized.adjustment" };
    let mut arena = CausalExprArena::new();
    let estimands = geometries
        .iter()
        .map(|pair| {
            let functional = arena.backdoor_ate(
                pair.treatment,
                pair.outcome,
                &z_vars,
                Value::f64(1.0),
                Value::f64(0.0),
            );
            IdentifiedEstimand::backdoor(label, Arc::clone(&z_vars), functional)
        })
        .collect();
    let mut derivation = DerivationTrace::default();
    derivation.push(
        "generalized.adjustment.admg",
        format!(
            "Z (size {}) m-separates every treatment/outcome pair in its proper back-door graph \
             and contains no node downstream of a causal-path child of the treatment",
            z_vars.len()
        ),
    );
    Ok(IdentificationResult::identified(
        causal_query,
        estimands,
        arena,
        derivation,
        assumptions,
        performance,
    ))
}

#[cfg(test)]
mod tests {
    use antecedent_core::{
        DerivativeScale, IdentificationStatus, ResponseFunctional, ResponseQuery,
    };

    use super::*;

    fn point_derivative(treatment: u32, outcome: u32) -> ResponseQuery {
        ResponseQuery::new(ResponseFunctional::PointDerivative {
            outcome: VariableId::from_raw(outcome),
            treatment: VariableId::from_raw(treatment),
            at: 0.0,
            order: 1,
            scale: DerivativeScale::Identity,
        })
    }

    fn node(i: u32) -> DenseNodeId {
        DenseNodeId::from_raw(i)
    }

    /// X=0, Y=1, W=2, V=3. W -> X, W -> Y, X -> Y and a latent cause between V and W.
    fn observed_confounder_with_latent_elsewhere() -> Admg {
        let mut g = Admg::with_variables(4);
        g.insert_directed(node(2), node(0)).unwrap();
        g.insert_directed(node(2), node(1)).unwrap();
        g.insert_directed(node(0), node(1)).unwrap();
        g.insert_bidirected(node(3), node(2)).unwrap();
        g
    }

    #[test]
    fn observed_confounder_is_the_adjustment_set_when_a_latent_cause_sits_elsewhere() {
        let g = observed_confounder_with_latent_elsewhere();
        let result = identify_admg_response_adjustment(&g, &point_derivative(0, 1)).unwrap();
        assert_eq!(result.status, IdentificationStatus::NonparametricallyIdentified);
        let set: Vec<_> = result.estimands[0].adjustment_set.iter().map(|v| v.raw()).collect();
        assert!(set.contains(&2), "the observed confounder must be adjusted: {set:?}");
    }

    #[test]
    fn direct_latent_confounding_of_treatment_and_outcome_has_no_adjustment_set() {
        let mut g = Admg::with_variables(3);
        g.insert_directed(node(0), node(1)).unwrap();
        g.insert_bidirected(node(0), node(1)).unwrap();
        g.insert_directed(node(2), node(0)).unwrap();
        let result = identify_admg_response_adjustment(&g, &point_derivative(0, 1)).unwrap();
        assert_eq!(result.status, IdentificationStatus::NotIdentified);
        assert!(result.estimands.is_empty());
    }

    #[test]
    fn a_mediator_is_never_adjusted() {
        // X=0 -> M=1 -> Y=2, with W=3 -> X and W -> Y.
        let mut g = Admg::with_variables(4);
        g.insert_directed(node(0), node(1)).unwrap();
        g.insert_directed(node(1), node(2)).unwrap();
        g.insert_directed(node(3), node(0)).unwrap();
        g.insert_directed(node(3), node(2)).unwrap();
        g.insert_bidirected(node(1), node(3)).unwrap();
        let result = identify_admg_response_adjustment(&g, &point_derivative(0, 2)).unwrap();
        assert_eq!(result.status, IdentificationStatus::NonparametricallyIdentified);
        let set: Vec<_> = result.estimands[0].adjustment_set.iter().map(|v| v.raw()).collect();
        assert!(!set.contains(&1), "the mediator is forbidden: {set:?}");
        assert!(set.contains(&3), "the confounder is required: {set:?}");
    }

    #[test]
    fn a_jacobian_over_two_outcomes_shares_one_set() {
        // X=0 -> Y1=1, X -> Y2=2, W=3 -> X, W -> Y1, W -> Y2, latent cause between W and Y1 is
        // absent; the common set is {W}.
        let mut g = Admg::with_variables(4);
        for (a, b) in [(0, 1), (0, 2), (3, 0), (3, 1), (3, 2)] {
            g.insert_directed(node(a), node(b)).unwrap();
        }
        g.insert_bidirected(node(1), node(2)).unwrap();
        let query = ResponseQuery::new(ResponseFunctional::Jacobian {
            outcomes: Arc::from([VariableId::from_raw(1), VariableId::from_raw(2)]),
            treatments: Arc::from([VariableId::from_raw(0)]),
            at: Arc::from([0.0]),
            scale: DerivativeScale::Identity,
        });
        let result = identify_admg_response_adjustment(&g, &query).unwrap();
        assert_eq!(result.status, IdentificationStatus::NonparametricallyIdentified);
        assert_eq!(result.estimands.len(), 2);
        assert_eq!(result.estimands[0].adjustment_set, result.estimands[1].adjustment_set);
        assert_eq!(result.estimands[0].adjustment_set.as_ref(), &[VariableId::from_raw(3)]);
    }

    #[test]
    fn the_classic_front_door_graph_has_no_adjustment_set_but_a_front_door_set() {
        // X=0 -> M=1 -> Y=2 with a latent common cause of X and Y.
        let mut g = Admg::with_variables(3);
        g.insert_directed(node(0), node(1)).unwrap();
        g.insert_directed(node(1), node(2)).unwrap();
        g.insert_bidirected(node(0), node(2)).unwrap();
        let query = point_derivative(0, 2);
        let adjusted = identify_admg_response_adjustment(&g, &query).unwrap();
        assert_eq!(adjusted.status, IdentificationStatus::NotIdentified);
        let front_door = identify_admg_front_door_derivative(&g, &query).unwrap().unwrap();
        assert_eq!(front_door.estimands[0].mediators.as_ref(), &[VariableId::from_raw(1)]);
        assert_eq!(front_door.estimands[0].method.as_ref(), "frontdoor");
    }

    #[test]
    fn a_mediator_confounded_with_the_outcome_is_not_a_front_door_set() {
        // As above, plus a latent cause of M and Y: the back-door path M <-> Y is open given X.
        let mut g = Admg::with_variables(3);
        g.insert_directed(node(0), node(1)).unwrap();
        g.insert_directed(node(1), node(2)).unwrap();
        g.insert_bidirected(node(0), node(2)).unwrap();
        g.insert_bidirected(node(1), node(2)).unwrap();
        assert!(
            identify_admg_front_door_derivative(&g, &point_derivative(0, 2)).unwrap().is_none()
        );
    }
}
