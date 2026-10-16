//! Efficient backdoor adjustment-set selection.
//!
//! Selects the *optimal* adjustment set `O` (Henckel, Perković & Maathuis 2022),
//! the O-set construction this efficient backdoor implementation follows: with
//! `cn = de(T) ∩ an(Y) \ {T}` (the nodes on proper causal paths from `T` to
//! `Y`, including `Y`), `forb = de(cn) ∪ {T}`, the O-set is
//! `O = pa(cn) \ forb`. Among all valid backdoor sets in a fully observed
//! DAG, `O` yields the smallest asymptotic variance for the adjusted effect
//! estimate. The candidate `O` is still validated against the backdoor
//! criterion (d-separation in the treatment-outgoing-mutilated graph, no
//! descendants of `T`, no configured forbidden variables) before being
//! returned.
//!
//! When `O` fails validation (e.g. a member is configured as forbidden), the
//! identifier falls back to a minimum-cardinality search: subsets are
//! enumerated by increasing size, stopping at the first size class with a
//! valid set, with ties broken by maximizing `|Z ∩ Pa(T)|`, then lower
//! measurement cost, then lexicographic variable ids. Equal costs leave the
//! previous order unchanged. The fallback keeps at most `max_results` valid sets; hitting
//! that limit truncates collection rather than erroring. Whether any valid set
//! exists is settled first by one separation test on `An({T,Y}) \ forbidden`;
//! that ancestral set is also the answer when the pool exceeds `max_candidates`
//! or the examination budget runs out, valid but not minimum-cardinality.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{AssumptionSet, AverageEffectQuery, CausalQuery, VariableId};
use antecedent_expr::CausalExprArena;
use antecedent_graph::{BitSet, Dag, DenseNodeId, GraphWorkspace};

use crate::backdoor::{
    AdjustmentProblem, AdjustmentSearchConfig, PreparedIdentificationGraph, dense_to_var,
    is_backdoor_adjustment, screen_ancestor_candidates,
};
use crate::error::IdentificationError;
use crate::identifier::IdentificationWorkspace;
use crate::result::{
    DerivationTrace, IdentificationPerformanceRecord, IdentificationResult, IdentifiedEstimand,
};

/// Identifier that returns one efficient backdoor adjustment set.
#[derive(Clone, Debug, Default)]
pub struct EfficientBackdoorIdentifier {
    /// Shared search limits / forbidden variables.
    pub config: AdjustmentSearchConfig,
}

/// The adjustment set a search settled on and how it got there.
#[derive(Clone, Copy)]
struct EfficientChoice<'a> {
    /// Candidate sets; the first is reported.
    chosen: &'a [Vec<DenseNodeId>],
    /// Separation tests run.
    examined: u64,
    /// Selection rule, recorded in the derivation.
    rule: &'a str,
    /// Whether enumeration hit `max_results`.
    truncated: bool,
}

impl EfficientBackdoorIdentifier {
    /// Create with default config.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Cap exact candidate-subset enumeration on the min-cardinality fallback (default 40).
    #[must_use]
    pub fn with_max_candidates(mut self, max_candidates: usize) -> Self {
        self.config.max_candidates = max_candidates;
        self
    }

    /// Prepare a graph (same wrap as [`crate::backdoor::BackdoorIdentifier`]).
    ///
    /// # Errors
    ///
    /// Propagates prepare failures.
    pub fn prepare(
        &self,
        graph: &antecedent_graph::Dag,
    ) -> Result<PreparedIdentificationGraph, IdentificationError> {
        self.prepare_with_assumptions(graph, AssumptionSet::new())
    }

    /// Prepare a graph, retaining caller-declared assumptions for the result.
    ///
    /// # Errors
    ///
    /// Propagates prepare failures.
    pub fn prepare_with_assumptions(
        &self,
        graph: &antecedent_graph::Dag,
        assumptions: AssumptionSet,
    ) -> Result<PreparedIdentificationGraph, IdentificationError> {
        Ok(PreparedIdentificationGraph::wrap(graph, assumptions))
    }

    /// Identify via efficient backdoor; at most one estimand is returned,
    /// preferring the O-set (optimal adjustment set) and falling back to a
    /// minimum-cardinality search when the O-set fails validation.
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
        let CausalQuery::AverageEffect(ate) = query else {
            return Err(IdentificationError::UnsupportedQuery {
                message: "efficient backdoor only supports AverageEffect",
            });
        };
        ate.validate().map_err(|_| IdentificationError::UnsupportedQuery {
            message: "invalid average-effect query",
        })?;
        self.identify_ate(prepared, ate, query.clone(), workspace)
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
        let dag = prepared.dag();
        let problem = AdjustmentProblem::new(
            dag,
            ate.treatment,
            ate.outcome,
            &self.config,
            &mut workspace.graph,
        )?;
        let t = problem.t;
        let y = problem.y;
        let forbidden = &problem.forbidden;
        let mutilated = &problem.mutilated;
        let mut candidates = problem.candidates.clone();
        candidates.retain(|id| !problem.exceeds_lag(*id));

        let mut examined = 0u64;

        // Primary rule: the O-set (optimal adjustment set, Henckel et al.).
        let o_set = optimal_adjustment_set(dag, t, y, &mut workspace.graph);
        examined += 1;
        if o_set.iter().all(|v| !forbidden.contains(*v) && !problem.exceeds_lag(*v))
            && is_backdoor_adjustment(mutilated, t, y, &o_set, &mut workspace.dsep)?
        {
            return Self::finish(
                ate,
                query,
                prepared,
                dag,
                &EfficientChoice {
                    chosen: &[o_set],
                    examined,
                    rule: "optimal (O-set)",
                    truncated: false,
                },
            );
        }

        // Fallback: minimum-cardinality search, sizes ascending, stopping at
        // the first size class that yields a valid set. Collection is capped
        // at `max_results`; hitting the cap truncates rather than errors.
        let mut candidates: Vec<DenseNodeId> = candidates;
        let mut m = candidates.len();
        let cap = self.config.max_candidates;

        // Existence is decided by one separation test: An({T,Y}) \ forbidden separates T from
        // Y in G underbar T whenever any admissible set does (van der Zander, Liskiewicz &
        // Textor 2014). If it fails no set exists and the size-ordered search is not needed;
        // if it passes it is the certified fallback wherever the search cannot finish.
        let constructive: Vec<DenseNodeId> =
            screen_ancestor_candidates(dag, t, y, forbidden, &problem, true, &mut workspace.graph);
        examined += 1;
        if !is_backdoor_adjustment(mutilated, t, y, &constructive, &mut workspace.dsep)? {
            let mut assumptions = default_assumptions();
            for record in &prepared.declared_assumptions().entries {
                assumptions.push(record.clone());
            }
            let mut derivation = DerivationTrace::default();
            derivation.push(
                "backdoor.efficient",
                "no valid adjustment set: the ancestral set An({T,Y}) minus forbidden does not \
                 separate T from Y",
            );
            return Ok(IdentificationResult::not_identified(
                query,
                derivation,
                assumptions,
                IdentificationPerformanceRecord { candidates_examined: examined, sets_returned: 0 },
            ));
        }
        if m > cap {
            candidates.clone_from(&constructive);
            m = candidates.len();
        }
        if m > cap {
            return Self::finish(
                ate,
                query,
                prepared,
                dag,
                &EfficientChoice {
                    chosen: &[constructive],
                    examined,
                    rule: "ancestral set (candidate pool over max_candidates; valid, not \
                           minimum-cardinality)",
                    truncated: false,
                },
            );
        }

        let mut valid: Vec<Vec<DenseNodeId>> = Vec::new();
        let mut truncated = false;
        let mut budget_exhausted = false;
        // `minimal_only` (the default) keeps the smallest size class. `maximal_only`
        // without it walks sizes downward and keeps the first class that separates,
        // which is what the shared search configuration asks for.
        let sizes: Vec<usize> = if self.config.maximal_only && !self.config.minimal_only {
            (0..=m).rev().collect()
        } else {
            (0..=m).collect()
        };
        'sizes: for size in sizes {
            let mut early_stop = false;
            let mut enum_err: Option<IdentificationError> = None;
            crate::enum_masks::for_each_mask_of_size(&candidates, size, |z| {
                if enum_err.is_some() {
                    return true;
                }
                if examined >= self.config.max_examinations {
                    budget_exhausted = true;
                    early_stop = true;
                    return true;
                }
                examined += 1;
                match is_backdoor_adjustment(mutilated, t, y, z, &mut workspace.dsep) {
                    Ok(true) => {
                        valid.push(z.to_vec());
                        if valid.len() >= self.config.max_results {
                            early_stop = true;
                            return true;
                        }
                    }
                    Ok(false) => {}
                    Err(e) => {
                        enum_err = Some(e);
                        return true;
                    }
                }
                false
            });
            if let Some(e) = enum_err {
                return Err(e);
            }
            if early_stop {
                truncated = true;
                break 'sizes;
            }
            if !valid.is_empty() {
                break;
            }
        }

        if valid.is_empty() && budget_exhausted {
            // The budget stopped the search for a minimum-cardinality set, but the ancestral
            // set is already known valid: identifiability is decided, the choice is not.
            let mut result = Self::finish(
                ate,
                query,
                prepared,
                dag,
                &EfficientChoice {
                    chosen: &[constructive],
                    examined,
                    rule: "ancestral set (examination budget exhausted; valid, not \
                           minimum-cardinality)",
                    truncated: false,
                },
            )?;
            result.diagnostics.push(crate::result::search_bounded_diagnostic(
                crate::backdoor::BACKDOOR_SEARCH_BOUNDED_DIAGNOSTIC_CODE,
                "efficient backdoor search exhausted max_examinations before it found a                  minimum-cardinality set; the returned set is valid but not minimal",
            ));
            return Ok(result);
        }

        let parent_set: BitSet = {
            let mut b = BitSet::with_len(dag.node_count());
            for p in dag.parents(t) {
                b.insert(*p);
            }
            b
        };
        valid.sort_by(|a, b| {
            let pa = a.iter().filter(|n| parent_set.contains(**n)).count();
            let pb = b.iter().filter(|n| parent_set.contains(**n)).count();
            a.len()
                .cmp(&b.len())
                .then_with(|| pb.cmp(&pa))
                .then_with(|| {
                    problem
                        .total_cost(a)
                        .partial_cmp(&problem.total_cost(b))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| {
                    let mut aa: Vec<_> = a.iter().map(|n| n.as_usize()).collect();
                    let mut bb: Vec<_> = b.iter().map(|n| n.as_usize()).collect();
                    aa.sort_unstable();
                    bb.sort_unstable();
                    aa.cmp(&bb)
                })
        });
        let best = vec![valid.into_iter().next().expect("non-empty")];
        let choice =
            EfficientChoice { chosen: &best, examined, rule: "min_cardinality", truncated };
        Self::finish(ate, query, prepared, dag, &choice)
    }

    fn finish(
        ate: &AverageEffectQuery,
        query: CausalQuery,
        prepared: &PreparedIdentificationGraph,
        dag: &antecedent_graph::Dag,
        choice: &EfficientChoice<'_>,
    ) -> Result<IdentificationResult, IdentificationError> {
        let EfficientChoice { chosen, examined, rule, truncated } = *choice;
        let z = &chosen[0];
        let vars: Vec<VariableId> =
            z.iter().map(|d| dense_to_var(*d, dag)).collect::<Result<_, _>>()?;
        let (active, control) = match (&ate.active, &ate.control) {
            (
                antecedent_core::Intervention::Set { value: active, .. },
                antecedent_core::Intervention::Set { value: control, .. },
            ) => (active.clone(), control.clone()),
            _ => {
                return Err(IdentificationError::UnsupportedQuery {
                    message: "efficient backdoor ATE requires Set interventions",
                });
            }
        };
        let mut arena = CausalExprArena::new();
        let functional = arena.backdoor_ate(ate.treatment, ate.outcome, &vars, active, control);
        let mut derivation = DerivationTrace::default();
        derivation.push("backdoor.efficient", format!("selected via {rule}; |Z|={}", vars.len()));
        if truncated {
            // Mirror `BackdoorIdentifier`: hitting `max_results` must leave a
            // visible mark, not silently narrow the enumeration.
            derivation.push(
                "backdoor.efficient",
                "enumeration truncated at max_results; smaller-cost sets may exist beyond the cap",
            );
        }
        let mut assumptions = default_assumptions();
        for record in &prepared.declared_assumptions().entries {
            assumptions.push(record.clone());
        }
        Ok(IdentificationResult::identified(
            query,
            vec![IdentifiedEstimand::backdoor("backdoor.efficient", Arc::from(vars), functional)],
            arena,
            derivation,
            assumptions,
            IdentificationPerformanceRecord { candidates_examined: examined, sets_returned: 1 },
        ))
    }
}

/// The O-set of Henckel, Perković & Maathuis (2022) (public for backdoor over-cap fallback).
pub(crate) fn optimal_adjustment_set_pub(
    dag: &Dag,
    t: DenseNodeId,
    y: DenseNodeId,
    gws: &mut GraphWorkspace,
) -> Vec<DenseNodeId> {
    optimal_adjustment_set(dag, t, y, gws)
}

/// The O-set of Henckel, Perković & Maathuis (2022): `pa(cn) \ (de(cn) ∪ {T})`,
/// where `cn = de(T) ∩ an(Y) \ {T}` are the nodes on proper causal paths from
/// `T` to `Y` (including `Y` itself). In a fully observed DAG this is the
/// valid backdoor set with the smallest asymptotic variance.
fn optimal_adjustment_set(
    dag: &Dag,
    t: DenseNodeId,
    y: DenseNodeId,
    gws: &mut GraphWorkspace,
) -> Vec<DenseNodeId> {
    let n = dag.node_count();
    let mut de_t = BitSet::with_len(n);
    dag.descendants_of(&[t], &mut de_t, gws);
    let mut an_y = BitSet::with_len(n);
    dag.ancestors_of(&[y], &mut an_y, gws);

    let cn: Vec<DenseNodeId> = (0..n)
        .map(DenseNodeId::from_index)
        .filter(|&v| v != t && de_t.contains(v) && an_y.contains(v))
        .collect();

    let mut forb = BitSet::with_len(n);
    dag.descendants_of(&cn, &mut forb, gws);
    forb.insert(t);

    let mut seen = BitSet::with_len(n);
    let mut o_set = Vec::new();
    for &c in &cn {
        for &p in dag.parents(c) {
            if !forb.contains(p) && !seen.contains(p) {
                seen.insert(p);
                o_set.push(p);
            }
        }
    }
    o_set.sort_unstable_by_key(|d| d.as_usize());
    o_set
}

fn default_assumptions() -> AssumptionSet {
    let mut assumptions = AssumptionSet::new();
    assumptions.push(crate::assumptions::causal_markov("backdoor.efficient"));
    assumptions
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::result::IdentificationStatus;
    use antecedent_graph::Dag;

    #[test]
    fn o_set_selects_confounder() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../conformance/identify/efficient_adjustment/expected.json"
        ))
        .unwrap();
        assert_eq!(fixture["cases"][0]["optimal_set"][0].as_str(), Some("Z"));
        // T <- Z -> Y, T -> Y (O = pa(Y) \ {T, Y} = {Z})
        let mut g = Dag::with_variables(3);
        let t = DenseNodeId::from_raw(0);
        let y = DenseNodeId::from_raw(1);
        let z = DenseNodeId::from_raw(2);
        g.insert_directed(z, t).unwrap();
        g.insert_directed(z, y).unwrap();
        g.insert_directed(t, y).unwrap();

        let id = EfficientBackdoorIdentifier::new();
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ));
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &q, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        assert_eq!(res.estimands.len(), 1);
        assert_eq!(&*res.estimands[0].method, "backdoor.efficient");
        assert_eq!(res.estimands[0].adjustment_set.as_ref(), &[VariableId::from_raw(2)]);
        assert!(res.derivation.steps.iter().any(|s| s.detail.contains("optimal")));
    }

    #[test]
    fn o_set_prefers_outcome_parent_over_treatment_parent() {
        // W -> T -> Y, Q -> Y: Pa(T) = {W} is valid but anti-efficient; the
        // O-set is {Q} (the outcome's other parent).
        let mut g = Dag::with_variables(4);
        let w = DenseNodeId::from_raw(0);
        let t = DenseNodeId::from_raw(1);
        let y = DenseNodeId::from_raw(2);
        let q_node = DenseNodeId::from_raw(3);
        g.insert_directed(w, t).unwrap();
        g.insert_directed(t, y).unwrap();
        g.insert_directed(q_node, y).unwrap();

        let id = EfficientBackdoorIdentifier::new();
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(1),
            VariableId::from_raw(2),
        ));
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &q, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        assert_eq!(res.estimands[0].adjustment_set.as_ref(), &[VariableId::from_raw(3)]);
        assert!(res.derivation.steps.iter().any(|s| s.detail.contains("optimal")));
    }

    #[test]
    fn many_disconnected_covariates_no_longer_hit_result_limit() {
        // T -> Y plus 8 disconnected covariates: every one of the 2^8 = 256
        // covariate subsets is a valid backdoor set, which previously tripped
        // the 64-set ResultLimitExceeded error during full enumeration.
        let mut g = Dag::with_variables(10);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();

        let id = EfficientBackdoorIdentifier::new();
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ));
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &q, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        assert!(res.estimands[0].adjustment_set.is_empty());
    }

    #[test]
    fn fallback_search_when_o_set_forbidden() {
        // A -> T -> Y, A -> B -> Y plus disconnected covariates. O = {B}, but
        // B is forbidden, so the fallback must find the minimal valid set {A}
        // without erroring on the many valid supersets.
        let mut g = Dag::with_variables(10);
        let a = DenseNodeId::from_raw(0);
        let t = DenseNodeId::from_raw(1);
        let y = DenseNodeId::from_raw(2);
        let b = DenseNodeId::from_raw(3);
        g.insert_directed(a, t).unwrap();
        g.insert_directed(t, y).unwrap();
        g.insert_directed(a, b).unwrap();
        g.insert_directed(b, y).unwrap();

        let mut id = EfficientBackdoorIdentifier::new();
        id.config.forbidden = Arc::from([VariableId::from_raw(3)]);
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(1),
            VariableId::from_raw(2),
        ));
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &q, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        assert_eq!(res.estimands[0].adjustment_set.as_ref(), &[VariableId::from_raw(0)]);
        assert!(res.derivation.steps.iter().any(|s| s.detail.contains("min_cardinality")));
    }

    #[test]
    fn no_admissible_set_is_a_proof_not_a_bounded_search() {
        // T <- U -> Y with U forbidden, plus thirty measured confounders. The path through U
        // cannot be blocked, so no set exists; two separation tests (the O-set, then the
        // ancestral set) prove it without touching the 2^30 subsets.
        let n_conf = 30u32;
        let mut g = Dag::with_variables(3 + n_conf);
        let t = DenseNodeId::from_raw(0);
        let y = DenseNodeId::from_raw(1);
        let u = DenseNodeId::from_raw(2);
        g.insert_directed(t, y).unwrap();
        g.insert_directed(u, t).unwrap();
        g.insert_directed(u, y).unwrap();
        for i in 0..n_conf {
            let c = DenseNodeId::from_raw(3 + i);
            g.insert_directed(c, t).unwrap();
            g.insert_directed(c, y).unwrap();
        }
        let mut id = EfficientBackdoorIdentifier::new();
        id.config.forbidden = Arc::from([VariableId::from_raw(2)]);
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ));
        let res = id.identify(&prep, &q, &mut IdentificationWorkspace::default()).unwrap();
        assert_eq!(res.status, IdentificationStatus::NotIdentified);
        assert!(!crate::envelope::search_truncated(&res), "{:?}", res.diagnostics);
        assert_eq!(res.performance.candidates_examined, 2);
    }

    #[test]
    fn over_cap_pool_returns_the_ancestral_set_instead_of_erroring() {
        // C2..C7 confound T and Y; C8 only causes Y and is forbidden, so the O-set is refused
        // and the six-variable pool exceeds max_candidates = 2. The ancestral set is exactly
        // the confounders.
        let mut g = Dag::with_variables(9);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        for i in 2..=7 {
            let c = DenseNodeId::from_raw(i);
            g.insert_directed(c, DenseNodeId::from_raw(0)).unwrap();
            g.insert_directed(c, DenseNodeId::from_raw(1)).unwrap();
        }
        g.insert_directed(DenseNodeId::from_raw(8), DenseNodeId::from_raw(1)).unwrap();
        let mut id = EfficientBackdoorIdentifier::new().with_max_candidates(2);
        id.config.forbidden = Arc::from([VariableId::from_raw(8)]);
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ));
        let res = id.identify(&prep, &q, &mut IdentificationWorkspace::default()).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        let expected: Vec<VariableId> = (2..=7).map(VariableId::from_raw).collect();
        assert_eq!(res.estimands[0].adjustment_set.as_ref(), expected.as_slice());
    }

    #[test]
    fn history_lag_rejects_an_o_set_that_needs_the_lagged_confounder() {
        // Both Z and W confound T and Y, so the O-set is {Z, W}. W's lag is past the cap,
        // and {Z} alone does not block W's back-door path.
        let mut g = Dag::with_variables(4);
        let (t, y, z, w) = (
            DenseNodeId::from_raw(0),
            DenseNodeId::from_raw(1),
            DenseNodeId::from_raw(2),
            DenseNodeId::from_raw(3),
        );
        for parent in [z, w] {
            g.insert_directed(parent, t).unwrap();
            g.insert_directed(parent, y).unwrap();
        }
        g.insert_directed(t, y).unwrap();
        let mut id = EfficientBackdoorIdentifier::new();
        id.config.max_history_lag = Some(0);
        id.config.history_lags =
            Arc::from([(VariableId::from_raw(2), 0u32), (VariableId::from_raw(3), 4)]);
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ));
        let res = id.identify(&prep, &q, &mut IdentificationWorkspace::default()).unwrap();
        assert_eq!(res.status, IdentificationStatus::NotIdentified);

        let open = EfficientBackdoorIdentifier::new();
        let open_res = open.identify(&prep, &q, &mut IdentificationWorkspace::default()).unwrap();
        assert_eq!(open_res.status, IdentificationStatus::NonparametricallyIdentified);
        assert_eq!(
            open_res.estimands[0].adjustment_set.as_ref(),
            &[VariableId::from_raw(2), VariableId::from_raw(3)]
        );
    }

    #[test]
    fn measurement_cost_breaks_ties_after_parent_overlap() {
        // T <- U <- Z2 <- Z4 -> Y, T <- Z3 -> Y, and a precision parent P -> Y.
        // Forbidding U and P rejects the O-set (it contains P) and leaves two
        // minimum-cardinality sets, {Z2, Z3} and {Z4, Z3}, with the same number
        // of parents of T. Lexicographic order prefers Z2; a lower cost on Z4
        // must win.
        let mut g = Dag::with_variables(7);
        let (t, y, z3, z2, u, z4, p) = (
            DenseNodeId::from_raw(0),
            DenseNodeId::from_raw(1),
            DenseNodeId::from_raw(2),
            DenseNodeId::from_raw(3),
            DenseNodeId::from_raw(4),
            DenseNodeId::from_raw(5),
            DenseNodeId::from_raw(6),
        );
        g.insert_directed(z3, t).unwrap();
        g.insert_directed(z3, y).unwrap();
        g.insert_directed(u, t).unwrap();
        g.insert_directed(z2, u).unwrap();
        g.insert_directed(z4, z2).unwrap();
        g.insert_directed(z4, y).unwrap();
        g.insert_directed(p, y).unwrap();
        g.insert_directed(t, y).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ));
        let mut plain = EfficientBackdoorIdentifier::new();
        plain.config.forbidden =
            Arc::from([VariableId::from_raw(u.raw()), VariableId::from_raw(p.raw())]);
        let prep = plain.prepare(&g).unwrap();
        let plain_res = plain.identify(&prep, &q, &mut IdentificationWorkspace::default()).unwrap();
        assert_eq!(
            plain_res.estimands[0].adjustment_set.as_ref(),
            &[VariableId::from_raw(z3.raw()), VariableId::from_raw(z2.raw())]
        );

        let mut priced = plain.clone();
        priced.config.measurement_costs = Arc::from([
            (VariableId::from_raw(z2.raw()), 10.0),
            (VariableId::from_raw(z4.raw()), 0.1),
        ]);
        let priced_res =
            priced.identify(&prep, &q, &mut IdentificationWorkspace::default()).unwrap();
        assert_eq!(
            priced_res.estimands[0].adjustment_set.as_ref(),
            &[VariableId::from_raw(z3.raw()), VariableId::from_raw(z4.raw())]
        );
    }

    #[test]
    fn empty_when_no_backdoor_paths() {
        let mut g = Dag::with_variables(2);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let id = EfficientBackdoorIdentifier::new();
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ));
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &q, &mut ws).unwrap();
        assert!(res.estimands[0].adjustment_set.is_empty());
        assert_eq!(&*res.estimands[0].method, "backdoor.efficient");
    }
}
