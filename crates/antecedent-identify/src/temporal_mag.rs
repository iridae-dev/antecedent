//! Query-window certificates for stationary temporal MAGs.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
use crate::generalized::{identify_on_mag_completion, not_identified};
use crate::{IdentificationError, IdentificationResult};

pub(crate) const HISTORY_CAPPED: &str = "identify.temporal.history_capped";
fn boundary_failure(query: &TemporalEffectQuery, detail: &str) -> IdentificationResult {
    let mut result = not_identified(CausalQuery::TemporalEffect(query.clone()), detail);
    result.diagnostics.push(antecedent_core::Diagnostic::new(
        HISTORY_CAPPED,
        antecedent_core::DiagnosticKind::Execution,
        antecedent_core::DiagnosticSeverity::Warning,
        detail,
    ));
    result
}
use antecedent_core::{
    CausalQuery, Intervention, TemporalEffectQuery, TemporalIndexer, TemporalNodeKey, Value,
    VariableId,
};
use antecedent_graph::{DenseNodeId, Endpoint, NodeRef, Pag, TemporalPag};
use std::collections::HashSet;

pub(crate) fn window(
    graph: &TemporalPag,
    query: &TemporalEffectQuery,
) -> Result<TemporalIndexer, IdentificationError> {
    query.validate().map_err(|_| IdentificationError::unsupported("invalid temporal query"))?;
    let seeds = query_keys(query)?;
    let mut variables = query
        .treatment
        .raw()
        .max(query.outcome.raw())
        .checked_add(1)
        .ok_or_else(|| IdentificationError::msg("temporal schema overflow"))?;
    let mut max_lag = 0;
    for node in graph.nodes() {
        if let NodeRef::Lagged { variable, lag } = node {
            variables = variables.max(
                variable
                    .raw()
                    .checked_add(1)
                    .ok_or_else(|| IdentificationError::msg("temporal schema overflow"))?,
            );
            max_lag = max_lag.max(lag.raw());
        }
    }
    let earliest = seeds.iter().map(|k| k.offset).min().unwrap_or(0).min(0).unsigned_abs();
    let latest = seeds.iter().map(|k| k.offset).max().unwrap_or(0).max(0).unsigned_abs();
    let cap = query
        .max_history_lag
        .unwrap_or_else(|| variables.saturating_mul(max_lag).saturating_add(earliest));
    if earliest > cap {
        return Err(IdentificationError::NotCertified {
            message: "requested treatment window exceeds the temporal history cap",
        });
    }
    let (depth, _) = closure(graph, &seeds, cap, true);
    let history = depth.saturating_add(max_lag).min(cap).max(earliest);
    TemporalIndexer::new(variables, history, latest.saturating_add(1))
        .map_err(|e| IdentificationError::msg(e.to_string()))
}

pub(crate) fn query_keys(
    query: &TemporalEffectQuery,
) -> Result<Vec<TemporalNodeKey>, IdentificationError> {
    let offset = query
        .try_treatment_offset()
        .map_err(|_| IdentificationError::unsupported("temporal query has no treatment offset"))?;
    Ok(vec![
        TemporalNodeKey { variable: query.treatment, offset },
        TemporalNodeKey { variable: query.outcome, offset: query.outcome_offset() },
    ])
}

// Symbolic parent expansion checks every boundary-crossing template edge, not
// merely parents of nodes on the deepest included slice. Possible parents give
// a shared window; definite parents certify each completion's relevant ancestry.
fn closure(
    graph: &TemporalPag,
    seeds: &[TemporalNodeKey],
    cap: u32,
    possible: bool,
) -> (u32, bool) {
    let mut pending: Vec<_> =
        seeds.iter().map(|k| (k.variable.raw(), i64::from(k.offset))).collect();
    let mut seen = HashSet::new();
    let mut depth = 0;
    let mut closed = true;
    let edges = graph.edges();
    while let Some((variable, offset)) = pending.pop() {
        if !seen.insert((variable, offset)) {
            continue;
        }
        depth = depth.max(u32::try_from((-offset).max(0)).unwrap_or(u32::MAX));
        for edge in &edges {
            for (parent, child, at_parent, at_child) in
                [(edge.a, edge.b, edge.at_a, edge.at_b), (edge.b, edge.a, edge.at_b, edge.at_a)]
            {
                if !(at_parent == Endpoint::Tail || possible && at_parent == Endpoint::Circle)
                    || !(at_child == Endpoint::Arrow || possible && at_child == Endpoint::Circle)
                {
                    continue;
                }
                let (
                    NodeRef::Lagged { variable: pv, lag: pl },
                    NodeRef::Lagged { variable: cv, lag: cl },
                ) = (graph.nodes()[parent.as_usize()], graph.nodes()[child.as_usize()])
                else {
                    continue;
                };
                if cv.raw() != variable || pl.raw() < cl.raw() {
                    continue;
                }
                let parent_offset = offset + i64::from(cl.raw()) - i64::from(pl.raw());
                if parent_offset < -i64::from(cap) {
                    closed = false;
                    continue;
                }
                pending.push((pv.raw(), parent_offset));
            }
        }
    }
    (depth, closed)
}

pub(crate) fn identify(
    graph: &TemporalPag,
    query: &TemporalEffectQuery,
    indexer: &TemporalIndexer,
    max_candidates: usize,
) -> Result<(IdentificationResult, Pag), IdentificationError> {
    let unfolded = graph.unfold(indexer.clone())?;
    let keys = query_keys(query)?;
    let id = |key| indexer.dense_id(key).map_err(|e| IdentificationError::msg(e.to_string()));
    let treatment = id(keys[0])?;
    let outcome = id(keys[1])?;
    let failed = |detail: &str| not_identified(CausalQuery::TemporalEffect(query.clone()), detail);
    let mut result = if !closure(graph, &keys, indexer.history(), false).1 {
        boundary_failure(
            query,
            "temporal MAG ancestry crosses the history boundary; finite-window adjustment is not certified",
        )
    } else if !antecedent_graph::completion::is_mag_completion(&unfolded.pag) {
        failed(
            "stationary temporal completion is not a maximal ancestral graph in the query window",
        )
    } else {
        let level = |intervention: &Intervention| -> Result<Value, IdentificationError> {
            match intervention {
                Intervention::Set { value, .. } => Ok(value.clone()),
                _ => Err(IdentificationError::unsupported(
                    "temporal MAG adjustment requires Set interventions",
                )),
            }
        };
        identify_on_mag_completion(
            &unfolded.pag,
            VariableId::from_raw(treatment),
            VariableId::from_raw(outcome),
            DenseNodeId::from_raw(treatment),
            DenseNodeId::from_raw(outcome),
            level(&query.active)?,
            level(&query.control)?,
            max_candidates,
        )?
    };
    let mut conditioning = keys.clone();
    for estimand in &result.estimands {
        for variable in estimand.adjustment_set.iter() {
            conditioning.push(
                indexer
                    .key_of(variable.raw())
                    .map_err(|e| IdentificationError::msg(e.to_string()))?,
            );
        }
    }
    if !closure(graph, &conditioning, indexer.history(), false).1 {
        result = boundary_failure(
            query,
            "temporal MAG adjustment covariates have ancestry outside the certified history window",
        );
    }
    crate::temporal_backdoor::annotate_temporal(
        &mut result,
        query,
        keys[0],
        keys[1],
        indexer.history(),
        indexer.horizon(),
    );
    Ok((result, unfolded.pag))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GeneralizedAdjustmentIdentifier, IdentificationStatus};
    use antecedent_core::{Lag, TemporalPolicy};
    use antecedent_graph::{MarkedEdge, MiddleMark};
    fn node(graph: &mut TemporalPag, variable: u32, lag: u32) -> DenseNodeId {
        graph.add_lagged(VariableId::from_raw(variable), Lag::from_raw(lag)).unwrap()
    }
    fn query() -> TemporalEffectQuery {
        let mut query =
            TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0);
        query.policy = TemporalPolicy::pulse(-1);
        query
    }
    #[test]
    fn latent_backdoor_is_adjusted_with_its_actual_lag() {
        let mut graph = TemporalPag::empty();
        let treatment = node(&mut graph, 0, 1);
        let outcome = node(&mut graph, 1, 0);
        let confounder = node(&mut graph, 2, 1);
        let witness = node(&mut graph, 3, 1);
        graph.insert_directed(witness, treatment).unwrap();
        graph.insert_directed(treatment, outcome).unwrap();
        graph.insert_directed(confounder, outcome).unwrap();
        graph
            .insert_marked(MarkedEdge {
                a: treatment,
                b: confounder,
                at_a: Endpoint::Arrow,
                at_b: Endpoint::Arrow,
                middle: MiddleMark::Empty,
            })
            .unwrap();
        let bundle = GeneralizedAdjustmentIdentifier::new()
            .identify_temporal_pag_envelope(&graph, &query())
            .unwrap();
        assert_eq!(bundle.envelope.status, IdentificationStatus::NonparametricallyIdentified);
        assert_eq!(bundle.envelope.cases.len(), 1);
        let adjustment = &bundle.envelope.cases[0].result.estimands[0].adjustment_set;
        assert_eq!(adjustment.len(), 1);
        assert_eq!(
            bundle.indexers[0].key_of(adjustment[0].raw()).unwrap(),
            TemporalNodeKey { variable: VariableId::from_raw(2), offset: -1 }
        );
    }
    #[test]
    fn stationary_witness_replicas_constrain_circle_completions() {
        let mut graph = TemporalPag::empty();
        let treatment_now = node(&mut graph, 0, 0);
        let treatment_past = node(&mut graph, 0, 1);
        let outcome = node(&mut graph, 1, 0);
        let witness = node(&mut graph, 2, 0);
        graph.insert_directed(witness, treatment_now).unwrap();
        graph.insert_circle_arrow(treatment_past, outcome).unwrap();
        let bundle = GeneralizedAdjustmentIdentifier::new()
            .identify_temporal_pag_envelope(&graph, &query())
            .unwrap();
        // R[-1] -> T[-1] exists by stationarity; a bidirected T[-1] <-> Y[0]
        // would add a collider contradicted by the unfolded source PAG.
        assert_eq!(bundle.envelope.cases.len(), 1);
        assert_eq!(bundle.envelope.status, IdentificationStatus::NonparametricallyIdentified);
    }
    #[test]
    fn history_caps_retain_uncertified_mass() {
        let mut graph = TemporalPag::empty();
        let treatment = node(&mut graph, 0, 1);
        let outcome = node(&mut graph, 1, 0);
        let witness = node(&mut graph, 2, 2);
        graph.insert_directed(witness, treatment).unwrap();
        graph.insert_directed(treatment, outcome).unwrap();
        let mut query = query();
        query.max_history_lag = Some(1);
        let bundle = GeneralizedAdjustmentIdentifier::new()
            .identify_temporal_pag_envelope(&graph, &query)
            .unwrap();
        assert_eq!(bundle.envelope.cases.len(), 1);
        assert_eq!(bundle.envelope.status, IdentificationStatus::NotIdentified);
        assert_eq!(bundle.envelope.truncated_completions, 1);
        query.max_history_lag = Some(2);
        let bundle = GeneralizedAdjustmentIdentifier::new()
            .identify_temporal_pag_envelope(&graph, &query)
            .unwrap();
        assert_eq!(bundle.envelope.status, IdentificationStatus::NonparametricallyIdentified);
    }
}
