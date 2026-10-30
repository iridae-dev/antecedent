//! Front-door criterion identification for DAGs.
//!
//! # Search completeness
//!
//! Candidate mediator sets include:
//!
//! 1. every singleton `{v}` with `v ∉ {T, Y}`,
//! 2. every non-empty subset of `children(T) \ {Y}` up to
//!    [`FrontDoorSearchConfig::max_mediator_set_size`],
//! 3. the full set `children(T) \ {Y}` when it exceeds that cardinality bound,
//! 4. every non-empty subset of **all** intermediates `V \ {T,Y}` up to
//!    `max_mediator_set_size` (covers non-child intermediates and mixed sets).
//!
//! Enumeration is still bounded by `max_mediator_set_size` and `max_results`.
//! A `NotIdentified` result means no candidate within those bounds qualifies.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{AssumptionSet, AverageEffectQuery, CausalQuery, VariableId};
use antecedent_expr::CausalExprArena;
use antecedent_graph::{DSeparationWorkspace, Dag, DenseNodeId};

use crate::backdoor::{
    PreparedIdentificationGraph, remove_nodes, remove_outgoing, remove_outgoing_set,
};
use crate::error::IdentificationError;
use crate::identification_run::IdentificationRun;
use crate::identifier::IdentificationWorkspace;
use crate::result::{
    DerivationTrace, IdentificationPerformanceRecord, IdentificationResult, IdentifiedEstimand,
};

/// Configuration for front-door mediator-set search.
#[derive(Clone, Debug)]
pub struct FrontDoorSearchConfig {
    /// Maximum number of mediator sets to return.
    pub max_results: usize,
    /// Maximum cardinality of subsets of all intermediates enumerated as candidates.
    ///
    /// All singletons are tested; all-intermediate subsets are searched only for
    /// pools of at most 12 nodes. Child subsets obey this cardinality bound, with
    /// the full child set also tested once when oversized.
    pub max_mediator_set_size: usize,
}

impl Default for FrontDoorSearchConfig {
    fn default() -> Self {
        Self { max_results: 64, max_mediator_set_size: 4 }
    }
}

/// Diagnostic code of a front-door mediator search that was cut by `max_mediator_set_size`,
/// the intermediate-pool bound, or `max_results`: a `NotIdentified` carrying it is a search
/// bound, not a structural refusal.
pub(crate) const FRONTDOOR_SEARCH_BOUNDED_DIAGNOSTIC_CODE: &str =
    "identify.frontdoor.search_bounded";

/// Front-door criterion identifier for static DAGs.
#[derive(Clone, Debug, Default)]
pub struct FrontDoorIdentifier {
    /// Search configuration.
    pub config: FrontDoorSearchConfig,
}

impl FrontDoorIdentifier {
    /// Create with default config.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Prepare a graph for front-door identification.
    ///
    /// # Errors
    ///
    /// Currently infallible; reserved for validation.
    pub fn prepare(&self, graph: &Dag) -> Result<PreparedIdentificationGraph, IdentificationError> {
        self.prepare_with_assumptions(graph, AssumptionSet::new())
    }

    /// Prepare a graph, retaining caller-declared assumptions for the result.
    ///
    /// # Errors
    ///
    /// Currently infallible; reserved for validation.
    pub fn prepare_with_assumptions(
        &self,
        graph: &Dag,
        assumptions: AssumptionSet,
    ) -> Result<PreparedIdentificationGraph, IdentificationError> {
        Ok(PreparedIdentificationGraph::wrap(graph, assumptions))
    }

    /// Identify an average-effect query via the front-door criterion.
    ///
    /// Searches singletons and bounded subsets of `children(T) \ {Y}` (see module
    /// docs); a `NotIdentified` status is relative to those candidate families.
    ///
    /// # Errors
    ///
    /// Unsupported query or unknown variables.
    pub fn identify(
        &self,
        prepared: &PreparedIdentificationGraph,
        query: &CausalQuery,
        workspace: &mut IdentificationWorkspace,
    ) -> Result<IdentificationResult, IdentificationError> {
        antecedent_core::execution_attempt::run_operation(
            antecedent_core::execution_attempt::Operation::Identification,
            || {
                let CausalQuery::AverageEffect(ate) = query else {
                    return Err(IdentificationError::UnsupportedQuery {
                        message: "front-door identification only supports AverageEffect",
                    });
                };
                ate.validate().map_err(|_| IdentificationError::UnsupportedQuery {
                    message: "invalid average-effect query",
                })?;
                self.identify_ate(prepared, ate, query.clone(), workspace)
            },
        )
    }

    #[allow(
        clippy::too_many_lines,
        reason = "one linear identification pipeline whose stages share the prepared graph and workspace"
    )]
    fn identify_ate(
        &self,
        prepared: &PreparedIdentificationGraph,
        ate: &AverageEffectQuery,
        query: CausalQuery,
        workspace: &mut IdentificationWorkspace,
    ) -> Result<IdentificationResult, IdentificationError> {
        crate::identification_run::require_marginal_target(ate)?;
        let run = IdentificationRun::open(prepared, ate, "frontdoor")?;
        let dag = run.dag();
        let (t, y) = (run.t, run.y);

        let children_of_t: Vec<DenseNodeId> =
            dag.children(t).iter().copied().filter(|&c| c != y).collect();
        let intermediates: Vec<DenseNodeId> = (0..dag.node_count())
            .map(DenseNodeId::from_index)
            .filter(|&v| v != t && v != y)
            .collect();
        let max_k = self.config.max_mediator_set_size.min(children_of_t.len());
        let child_overflow = children_of_t.len() > max_k;
        let mut search_bounded = child_overflow
            || intermediates.len() > 12
            || intermediates.len() > self.config.max_mediator_set_size.max(1);
        let mut valid: Vec<Vec<DenseNodeId>> = Vec::new();
        let mut examined = 0u64;
        let mut error = None;
        let stopped = for_each_frontdoor_candidate(
            &intermediates,
            &children_of_t,
            self.config.max_mediator_set_size,
            |m| {
                examined += 1;
                match is_frontdoor_set(dag, t, y, m, &mut workspace.dsep) {
                    Ok(true) => {
                        valid.push(m.to_vec());
                        valid.len() >= self.config.max_results
                    }
                    Ok(false) => false,
                    Err(err) => {
                        error = Some(err);
                        true
                    }
                }
            },
        );
        if let Some(err) = error {
            return Err(err);
        }
        search_bounded |= stopped;

        let assumptions = run.base_assumptions();

        let mut derivation = DerivationTrace::default();
        derivation.push(
            "frontdoor.criterion",
            "M intercepts all directed T->Y paths; T-M and M-Y backdoors blocked",
        );

        if valid.is_empty() {
            let mut result = IdentificationResult::not_identified(
                query,
                derivation,
                assumptions,
                IdentificationPerformanceRecord { candidates_examined: examined, sets_returned: 0 },
            );
            if search_bounded {
                result.diagnostics.push(crate::result::search_bounded_diagnostic(
                    FRONTDOOR_SEARCH_BOUNDED_DIAGNOSTIC_CODE,
                    "front-door mediator search was truncated by max_mediator_set_size / \
                     max_results; NotIdentified may be a search bound, not structural non-ID",
                ));
            }
            return Ok(result);
        }

        let mut arena = CausalExprArena::new();
        let mut estimands = Vec::with_capacity(valid.len());
        for m in &valid {
            let vars: Vec<VariableId> = run.variables(m)?;
            let (active, control) =
                run.hard_set_levels("front-door ATE requires Set interventions")?;
            let functional =
                arena.frontdoor_ate(ate.treatment, ate.outcome, &vars, active, control);
            estimands.push(IdentifiedEstimand::frontdoor("frontdoor", Arc::from(vars), functional));
            derivation.push("frontdoor.mediator_set", format!("|M|={}", m.len()));
        }

        let mut result = IdentificationResult::identified(
            query,
            estimands,
            arena,
            derivation,
            assumptions,
            IdentificationPerformanceRecord {
                candidates_examined: examined,
                sets_returned: u64::try_from(valid.len()).unwrap_or(u64::MAX),
            },
        );
        if search_bounded {
            result.diagnostics.push(crate::result::search_bounded_diagnostic(
                FRONTDOOR_SEARCH_BOUNDED_DIAGNOSTIC_CODE,
                "front-door mediator search was truncated by max_mediator_set_size / max_results; \
                 NotIdentified (or a missing set) may be a search bound, not structural non-ID",
            ));
        }
        Ok(result)
    }
}

/// Whether `m` satisfies the front-door criterion relative to `t` and `y`.
fn is_frontdoor_set(
    dag: &Dag,
    t: DenseNodeId,
    y: DenseNodeId,
    m: &[DenseNodeId],
    ws: &mut DSeparationWorkspace,
) -> Result<bool, IdentificationError> {
    if m.is_empty() {
        return Ok(false);
    }

    // 1. M intercepts all directed paths from T to Y: removing M entirely
    // must disconnect T from Y via directed edges.
    let without_m = remove_nodes(dag, m)?;
    if without_m.reaches(t, y) {
        return Ok(false);
    }

    // 2. No unblocked backdoor path from T to any m in M: in G with T's
    // outgoing edges removed, T must be d-separated from every m given ∅.
    let t_mutilated = remove_outgoing(dag, t)?;
    for &mi in m {
        let sep = t_mutilated.is_d_separated(t, mi, &[], ws).map_err(IdentificationError::from)?;
        if !sep {
            return Ok(false);
        }
    }

    // 3. All backdoor paths from M to Y are blocked by T: in G with every
    // m's outgoing edges removed, each m must be d-separated from Y given {T}.
    let m_mutilated = remove_outgoing_set(dag, m)?;
    for &mi in m {
        let sep = m_mutilated.is_d_separated(mi, y, &[t], ws).map_err(IdentificationError::from)?;
        if !sep {
            return Ok(false);
        }
    }

    Ok(true)
}

/// Visit the historical candidate order without materializing its combinatorial family.
fn for_each_frontdoor_candidate(
    intermediates: &[DenseNodeId],
    children: &[DenseNodeId],
    max_size: usize,
    mut visit: impl FnMut(&[DenseNodeId]) -> bool,
) -> bool {
    for node in intermediates {
        if visit(std::slice::from_ref(node)) {
            return true;
        }
    }
    let max_children = max_size.min(children.len());
    for k in 2..=max_children {
        if crate::enum_masks::for_each_mask_of_size(children, k, &mut visit) {
            return true;
        }
    }
    if children.len() > max_children && children.len() > 1 && visit(children) {
        return true;
    }
    if intermediates.len() <= 12 {
        let max_all = max_size.min(intermediates.len());
        for k in 2..=max_all {
            if crate::enum_masks::for_each_mask_of_size(intermediates, k, |set| {
                if set.iter().all(|node| children.contains(node)) { false } else { visit(set) }
            }) {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::result::IdentificationStatus;
    use antecedent_core::{AverageEffectQuery, DiagnosticKind};

    #[test]
    fn streamed_candidates_match_eager_order_and_stop_early() {
        use std::collections::BTreeSet;
        for n in 0..=7 {
            let nodes: Vec<_> = (0..n).map(DenseNodeId::from_index).collect();
            for mask in 0..(1usize << n) {
                let children: Vec<_> = nodes
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &v)| (mask & (1 << i) != 0).then_some(v))
                    .rev()
                    .collect();
                for max_size in 0..=4 {
                    let mut eager = Vec::new();
                    let mut seen = BTreeSet::new();
                    let mut push = |set: &[DenseNodeId]| {
                        let mut key: Vec<_> = set.iter().map(|d| d.raw()).collect();
                        key.sort_unstable();
                        if seen.insert(key) {
                            eager.push(set.to_vec());
                        }
                    };
                    for node in &nodes {
                        push(std::slice::from_ref(node));
                    }
                    let child_max = max_size.min(children.len());
                    for size in 1..=child_max {
                        crate::enum_masks::for_each_mask_of_size(&children, size, |set| {
                            push(set);
                            false
                        });
                    }
                    if children.len() > child_max {
                        push(&children);
                    }
                    for size in 1..=max_size.min(nodes.len()) {
                        crate::enum_masks::for_each_mask_of_size(&nodes, size, |set| {
                            push(set);
                            false
                        });
                    }
                    let mut actual = Vec::new();
                    assert!(!for_each_frontdoor_candidate(&nodes, &children, max_size, |set| {
                        actual.push(set.to_vec());
                        false
                    }));
                    assert_eq!(actual, eager, "n={n}, mask={mask}, max_size={max_size}");
                }
            }
        }
        let nodes: Vec<_> = (0..200).map(DenseNodeId::from_index).collect();
        let mut visited = 0;
        assert!(for_each_frontdoor_candidate(&nodes, &nodes, 4, |_| {
            visited += 1;
            true
        }));
        assert_eq!(visited, 1);
    }

    #[test]
    fn classic_frontdoor_with_unmeasured_confounder() {
        // U -> T, U -> Y, T -> M -> Y (U unmeasured; here modeled but never
        // adjustable so the direct T<-U->Y backdoor is unblockable without M).
        let mut g = Dag::with_variables(4);
        let t = DenseNodeId::from_raw(0);
        let m = DenseNodeId::from_raw(1);
        let y = DenseNodeId::from_raw(2);
        let u = DenseNodeId::from_raw(3);
        g.insert_directed(t, m).unwrap();
        g.insert_directed(m, y).unwrap();
        g.insert_directed(u, t).unwrap();
        g.insert_directed(u, y).unwrap();

        let id = FrontDoorIdentifier::new();
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(2),
        ));
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &q, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        assert!(res.estimands.iter().any(|e| e.mediators.as_ref() == [VariableId::from_raw(1)]));
    }

    #[test]
    fn chain_without_confounder_still_identifies() {
        // T -> M -> Y only.
        let mut g = Dag::with_variables(3);
        let t = DenseNodeId::from_raw(0);
        let m = DenseNodeId::from_raw(1);
        let y = DenseNodeId::from_raw(2);
        g.insert_directed(t, m).unwrap();
        g.insert_directed(m, y).unwrap();

        let id = FrontDoorIdentifier::new();
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(2),
        ));
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &q, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        assert!(res.estimands.iter().any(|e| e.mediators.as_ref() == [VariableId::from_raw(1)]));
    }

    #[test]
    fn direct_edge_with_no_mediator_is_not_identified() {
        // T -> Y directly, no mediator can block the direct path.
        let mut g = Dag::with_variables(3);
        let t = DenseNodeId::from_raw(0);
        let y = DenseNodeId::from_raw(1);
        let w = DenseNodeId::from_raw(2);
        g.insert_directed(t, y).unwrap();
        g.insert_directed(t, w).unwrap();

        let id = FrontDoorIdentifier::new();
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ));
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &q, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NotIdentified);
        assert!(res.estimands.is_empty());
    }

    #[test]
    fn proper_subset_of_children_identifies_when_full_set_fails_backdoor() {
        // T→M1→Y, T→M2→Y, T→M3 with W confounding M3–Y and U confounding T–Y.
        // No singleton intercepts both directed paths; {M1,M2} is FD; {M1,M2,M3} fails
        // criterion 3 for M3. Pre-fix search (singletons + full only) missed {M1,M2}.
        let mut g = Dag::with_variables(6);
        let t = DenseNodeId::from_raw(0);
        let m1 = DenseNodeId::from_raw(1);
        let m2 = DenseNodeId::from_raw(2);
        let m3 = DenseNodeId::from_raw(3);
        let y = DenseNodeId::from_raw(4);
        let w = DenseNodeId::from_raw(5);
        g.insert_directed(t, m1).unwrap();
        g.insert_directed(m1, y).unwrap();
        g.insert_directed(t, m2).unwrap();
        g.insert_directed(m2, y).unwrap();
        g.insert_directed(t, m3).unwrap();
        g.insert_directed(w, m3).unwrap();
        g.insert_directed(w, y).unwrap();
        g.insert_directed(w, t).unwrap();

        let id = FrontDoorIdentifier::new();
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(4),
        ));
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &q, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        let want = [VariableId::from_raw(1), VariableId::from_raw(2)];
        assert!(
            res.estimands.iter().any(|e| {
                let mut m = e.mediators.to_vec();
                m.sort_by_key(|v| v.raw());
                m == want
            }),
            "expected mediators {{M1,M2}}; got {:?}",
            res.estimands.iter().map(|e| e.mediators.clone()).collect::<Vec<_>>()
        );
    }
    #[test]
    fn nonchild_cardinality_limit_reports_incomplete_search() {
        // Only one treatment child: the skipped {M,W} intermediate set used
        // to go unreported because the warning only checked child subsets.
        let mut graph = Dag::with_variables(4);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        graph.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
        let identifier = FrontDoorIdentifier {
            config: FrontDoorSearchConfig { max_results: 64, max_mediator_set_size: 1 },
        };
        let prepared = identifier.prepare(&graph).unwrap();
        let query = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(2),
        ));
        let result = identifier
            .identify(&prepared, &query, &mut IdentificationWorkspace::default())
            .unwrap();
        assert_eq!(result.status, IdentificationStatus::NonparametricallyIdentified);
        let bounded = result
            .diagnostics
            .iter()
            .find(|d| d.code.as_ref() == "identify.frontdoor.search_bounded")
            .expect("bounded search is reported");
        // How far the search ran is a fact about the run, not about the graph.
        assert_eq!(bounded.kind, DiagnosticKind::Execution);
    }
}
