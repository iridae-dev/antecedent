//! Backdoor adjustment identification for DAGs.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::many_single_char_names)]

use std::sync::Arc;

use causal_core::{AssumptionSet, AverageEffectQuery, CausalQuery, VariableId};
use causal_expr::CausalExprArena;
use causal_graph::{BitSet, DSeparationWorkspace, Dag, DenseNodeId};

use crate::error::IdentificationError;
use crate::identifier::IdentificationWorkspace;
use crate::result::{
    DerivationTrace, IdentificationPerformanceRecord, IdentificationResult, IdentifiedEstimand,
};

/// Configuration for adjustment-set enumeration.
#[derive(Clone, Debug)]
pub struct AdjustmentSearchConfig {
    /// Maximum number of adjustment sets to return.
    pub max_results: usize,
    /// Variables forbidden from appearing in Z.
    pub forbidden: Arc<[VariableId]>,
    /// If true, only return inclusion-minimal sets.
    pub minimal_only: bool,
}

impl Default for AdjustmentSearchConfig {
    fn default() -> Self {
        Self { max_results: 64, forbidden: Arc::from([]), minimal_only: true }
    }
}

/// Prepared DAG for identification (cached ancestry helpers via workspaces).
#[derive(Clone, Debug)]
pub struct PreparedIdentificationGraph {
    dag: Dag,
    /// Caller-declared assumptions captured at [`BackdoorIdentifier::prepare_with_assumptions`].
    declared_assumptions: AssumptionSet,
}

impl PreparedIdentificationGraph {
    /// Wrap a DAG with no extra declared assumptions.
    #[must_use]
    pub fn new(dag: Dag) -> Self {
        Self { dag, declared_assumptions: AssumptionSet::new() }
    }

    /// Wrap a DAG together with caller-declared assumptions.
    #[must_use]
    pub fn with_assumptions(dag: Dag, declared_assumptions: AssumptionSet) -> Self {
        Self { dag, declared_assumptions }
    }

    /// Borrow the DAG.
    #[must_use]
    pub fn dag(&self) -> &Dag {
        &self.dag
    }

    /// Assumptions declared at prepare time (merged into identification results).
    #[must_use]
    pub fn declared_assumptions(&self) -> &AssumptionSet {
        &self.declared_assumptions
    }
}

/// Identifier for static DAGs .
#[derive(Clone, Debug, Default)]
pub struct BackdoorIdentifier {
    /// Search configuration.
    pub config: AdjustmentSearchConfig,
}

impl BackdoorIdentifier {
    /// Create with default config.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Prepare a graph with no extra declared assumptions.
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
        Ok(PreparedIdentificationGraph::with_assumptions(graph.clone(), assumptions))
    }

    /// Identify an average-effect query via backdoor adjustment.
    ///
    /// At most `config.max_results` adjustment sets are returned; when more
    /// qualifying sets exist the result is truncated (noted in the derivation
    /// trace) rather than treated as an error.
    ///
    /// # Errors
    ///
    /// Unsupported query, unknown variables, or a candidate pool too large
    /// for exact enumeration.
    pub fn identify(
        &self,
        prepared: &PreparedIdentificationGraph,
        query: &CausalQuery,
        workspace: &mut IdentificationWorkspace,
    ) -> Result<IdentificationResult, IdentificationError> {
        let CausalQuery::AverageEffect(ate) = query else {
            return Err(IdentificationError::UnsupportedQuery {
                message: match query {
                    CausalQuery::Distribution(_) => {
                        "Distribution identification deferred (requires IDC; coordinate with deep identification)"
                    }
                    CausalQuery::PathSpecific(_) => {
                        "PathSpecific identification deferred (path-restricted ID / natural effects)"
                    }
                    _ => "backdoor only supports AverageEffect",
                },
            });
        };
        ate.validate().map_err(|_| IdentificationError::UnsupportedQuery {
            message: "invalid average-effect query",
        })?;
        self.identify_ate(prepared, ate, query.clone(), workspace)
    }

    #[allow(clippy::too_many_lines)]
    fn identify_ate(
        &self,
        prepared: &PreparedIdentificationGraph,
        ate: &AverageEffectQuery,
        query: CausalQuery,
        workspace: &mut IdentificationWorkspace,
    ) -> Result<IdentificationResult, IdentificationError> {
        let dag = prepared.dag();
        let t = var_to_dense(ate.treatment, dag)?;
        let y = var_to_dense(ate.outcome, dag)?;
        let mut forbidden = BitSet::with_len(dag.node_count());
        for &v in self.config.forbidden.iter() {
            forbidden.insert(var_to_dense(v, dag)?);
        }
        forbidden.insert(t);
        forbidden.insert(y);

        // Descendants of T cannot be in Z.
        let mut desc = BitSet::with_len(dag.node_count());
        dag.descendants_of(&[t], &mut desc, &mut workspace.graph);
        for i in 0..dag.node_count() {
            let id = DenseNodeId::from_raw(u32::try_from(i).expect("fit"));
            if desc.contains(id) {
                forbidden.insert(id);
            }
        }

        // G underbar T: remove outgoing edges from T.
        let mutilated = remove_outgoing(dag, t)?;

        let candidates: Vec<DenseNodeId> = (0..dag.node_count())
            .map(|i| DenseNodeId::from_raw(u32::try_from(i).expect("fit")))
            .filter(|id| !forbidden.contains(*id))
            .collect();

        let mut valid: Vec<Vec<DenseNodeId>> = Vec::new();
        let mut examined = 0u64;
        let mut truncated = false;

        // Enumerate subsets by increasing size for minimal-first. When more
        // than `max_results` qualifying sets exist, the first `max_results`
        // (in size-then-mask order) are returned and the truncation is noted
        // in the derivation trace.
        let m = candidates.len();
        if m > 20 {
            return Err(IdentificationError::NotIdentified {
                message: "candidate set too large for exact enumeration (>20)",
            });
        }
        'sizes: for size in 0..=m {
            let mut early_stop = false;
            let mut enum_err: Option<IdentificationError> = None;
            crate::enum_masks::for_each_mask_of_size(&candidates, size, |z| {
                if enum_err.is_some() {
                    return true;
                }
                examined += 1;
                match is_backdoor_adjustment(&mutilated, t, y, z, &mut workspace.dsep) {
                    Ok(false) => return false,
                    Err(e) => {
                        enum_err = Some(e);
                        return true;
                    }
                    Ok(true) => {}
                }
                // Inclusion-minimal: skip any set that has a previously accepted
                // valid subset (filter is live across size classes).
                if self.config.minimal_only && valid.iter().any(|prev| is_subset(prev, z)) {
                    return false;
                }
                valid.push(z.to_vec());
                if valid.len() >= self.config.max_results {
                    truncated = true;
                    early_stop = true;
                    return true;
                }
                false
            });
            if let Some(e) = enum_err {
                return Err(e);
            }
            if early_stop {
                break 'sizes;
            }
            // Continue larger sizes when `minimal_only`: distinct inclusion-minimal
            // sets need not share a cardinality (e.g. {A} and {B,C}).
        }

        let mut assumptions = AssumptionSet::new();
        assumptions.push(crate::assumptions::causal_markov("backdoor"));
        for record in &prepared.declared_assumptions().entries {
            assumptions.push(record.clone());
        }

        let mut derivation = DerivationTrace::default();
        derivation.push(
            "backdoor.criterion",
            "Z blocks all backdoor paths and contains no descendants of T",
        );
        if truncated {
            derivation.push(
                "backdoor.enumeration",
                format!(
                    "result limit reached; returning first {} adjustment sets (more exist)",
                    valid.len()
                ),
            );
        }

        if valid.is_empty() {
            return Ok(IdentificationResult::not_identified(
                query,
                derivation,
                assumptions,
                IdentificationPerformanceRecord { candidates_examined: examined, sets_returned: 0 },
            ));
        }

        let mut arena = CausalExprArena::new();
        let mut estimands = Vec::with_capacity(valid.len());
        for z in &valid {
            let vars: Vec<VariableId> =
                z.iter().map(|d| dense_to_var(*d, dag)).collect::<Result<_, _>>()?;
            let (active, control) = match (&ate.active, &ate.control) {
                (
                    causal_core::Intervention::Set { value: active, .. },
                    causal_core::Intervention::Set { value: control, .. },
                ) => (active.clone(), control.clone()),
                _ => {
                    return Err(IdentificationError::UnsupportedQuery {
                        message: " backdoor ATE requires Set interventions",
                    });
                }
            };
            let functional = arena.backdoor_ate(ate.treatment, ate.outcome, &vars, active, control);
            estimands.push(IdentifiedEstimand::backdoor(
                "backdoor.adjustment",
                Arc::from(vars),
                functional,
            ));
            derivation.push("backdoor.adjustment_set", format!("|Z|={}", z.len()));
        }

        Ok(IdentificationResult::identified(
            query,
            estimands,
            arena,
            derivation,
            assumptions,
            IdentificationPerformanceRecord {
                candidates_examined: examined,
                sets_returned: u64::try_from(valid.len()).unwrap_or(u64::MAX),
            },
        ))
    }
}

fn is_subset(small: &[DenseNodeId], big: &[DenseNodeId]) -> bool {
    small.iter().all(|s| big.contains(s))
}

pub(crate) fn is_backdoor_adjustment(
    mutilated: &Dag,
    t: DenseNodeId,
    y: DenseNodeId,
    z: &[DenseNodeId],
    ws: &mut DSeparationWorkspace,
) -> Result<bool, IdentificationError> {
    mutilated.is_d_separated(t, y, z, ws).map_err(IdentificationError::from)
}

pub(crate) fn remove_outgoing(dag: &Dag, t: DenseNodeId) -> Result<Dag, IdentificationError> {
    remove_outgoing_set(dag, &[t])
}

/// `G` with outgoing edges from every node in `nodes` removed (node ids preserved).
pub(crate) fn remove_outgoing_set(
    dag: &Dag,
    nodes: &[DenseNodeId],
) -> Result<Dag, IdentificationError> {
    let n = u32::try_from(dag.node_count()).map_err(|_| IdentificationError::msg("n"))?;
    let mut out = Dag::with_variables(n);
    for e in dag.edges() {
        let (from, to) = e.parent_child().expect("dag");
        if nodes.contains(&from) {
            continue;
        }
        out.insert_directed(from, to).map_err(IdentificationError::from)?;
    }
    Ok(out)
}

/// `G` with every node in `nodes` fully removed (both incoming and outgoing
/// edges dropped; node ids and count preserved).
pub(crate) fn remove_nodes(dag: &Dag, nodes: &[DenseNodeId]) -> Result<Dag, IdentificationError> {
    let n = u32::try_from(dag.node_count()).map_err(|_| IdentificationError::msg("n"))?;
    let mut out = Dag::with_variables(n);
    for e in dag.edges() {
        let (from, to) = e.parent_child().expect("dag");
        if nodes.contains(&from) || nodes.contains(&to) {
            continue;
        }
        out.insert_directed(from, to).map_err(IdentificationError::from)?;
    }
    Ok(out)
}

pub(crate) fn var_to_dense(id: VariableId, dag: &Dag) -> Result<DenseNodeId, IdentificationError> {
    for (i, node) in dag.nodes().iter().enumerate() {
        if let causal_graph::NodeRef::Static(v) = node {
            if *v == id {
                return Ok(DenseNodeId::from_raw(u32::try_from(i).expect("fit")));
            }
        }
    }
    Err(IdentificationError::UnknownVariable { id })
}

pub(crate) fn dense_to_var(id: DenseNodeId, dag: &Dag) -> Result<VariableId, IdentificationError> {
    match dag.nodes().get(id.as_usize()) {
        Some(causal_graph::NodeRef::Static(v)) => Ok(*v),
        _ => Err(IdentificationError::msg("expected static node")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::result::IdentificationStatus;
    use causal_core::AverageEffectQuery;

    #[test]
    fn var_to_dense_respects_node_labels_not_raw_ids() {
        // Nodes labeled with non-dense VariableIds (raw 10, 20, 30 at dense 0,1,2).
        let mut g = Dag::empty();
        let t = g.add_node(causal_graph::NodeRef::Static(VariableId::from_raw(10))).unwrap();
        let y = g.add_node(causal_graph::NodeRef::Static(VariableId::from_raw(20))).unwrap();
        let z = g.add_node(causal_graph::NodeRef::Static(VariableId::from_raw(30))).unwrap();
        g.insert_directed(z, t).unwrap();
        g.insert_directed(z, y).unwrap();
        g.insert_directed(t, y).unwrap();

        assert_eq!(var_to_dense(VariableId::from_raw(10), &g).unwrap(), t);
        assert_eq!(var_to_dense(VariableId::from_raw(20), &g).unwrap(), y);
        assert_eq!(var_to_dense(VariableId::from_raw(30), &g).unwrap(), z);
        assert!(var_to_dense(VariableId::from_raw(0), &g).is_err());

        let id = BackdoorIdentifier::new();
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(10),
            VariableId::from_raw(20),
        ));
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &q, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        assert_eq!(res.estimands[0].adjustment_set.as_ref(), &[VariableId::from_raw(30)]);
    }

    #[test]
    fn confounding_requires_z() {
        // T <- Z -> Y, T -> Y
        let mut g = Dag::with_variables(3);
        let t = DenseNodeId::from_raw(0);
        let y = DenseNodeId::from_raw(1);
        let z = DenseNodeId::from_raw(2);
        g.insert_directed(z, t).unwrap();
        g.insert_directed(z, y).unwrap();
        g.insert_directed(t, y).unwrap();

        let id = BackdoorIdentifier::new();
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ));
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &q, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        assert_eq!(res.estimands.len(), 1);
        assert_eq!(res.estimands[0].adjustment_set.as_ref(), &[VariableId::from_raw(2)]);
    }

    #[test]
    fn result_limit_truncates_instead_of_erroring() {
        // T <- A -> B -> C -> Y, T -> Y: minimal singletons {A}, {B}, {C}.
        // With max_results = 2 the first two are returned, not an error.
        let mut g = Dag::with_variables(5);
        let t = DenseNodeId::from_raw(0);
        let y = DenseNodeId::from_raw(1);
        let a = DenseNodeId::from_raw(2);
        let b = DenseNodeId::from_raw(3);
        let c = DenseNodeId::from_raw(4);
        g.insert_directed(a, t).unwrap();
        g.insert_directed(a, b).unwrap();
        g.insert_directed(b, c).unwrap();
        g.insert_directed(c, y).unwrap();
        g.insert_directed(t, y).unwrap();

        let mut id = BackdoorIdentifier::new();
        id.config.max_results = 2;
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ));
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &q, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        assert_eq!(res.estimands.len(), 2);
        assert!(res.derivation.steps.iter().any(|s| s.rule.as_ref() == "backdoor.enumeration"));
    }

    #[test]
    fn inclusion_minimal_keeps_distinct_cardinalities() {
        // Two disjoint backdoor paths: T <- A -> Y and T <- B <- C -> Y.
        // {A,B}, {A,C} are min-cardinality; with minimal_only we still want only
        // inclusion-minimal sets. Here singletons fail, so {A,B} and {A,C} both
        // qualify at size 2 — and continuing sizes must not add their supersets.
        let mut g = Dag::with_variables(5);
        let t = DenseNodeId::from_raw(0);
        let y = DenseNodeId::from_raw(1);
        let a = DenseNodeId::from_raw(2);
        let b = DenseNodeId::from_raw(3);
        let c = DenseNodeId::from_raw(4);
        g.insert_directed(a, t).unwrap();
        g.insert_directed(a, y).unwrap();
        g.insert_directed(b, t).unwrap();
        g.insert_directed(c, b).unwrap();
        g.insert_directed(c, y).unwrap();
        g.insert_directed(t, y).unwrap();

        let id = BackdoorIdentifier::new();
        let prep = id.prepare(&g).unwrap();
        let q = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ));
        let mut ws = IdentificationWorkspace::default();
        let res = id.identify(&prep, &q, &mut ws).unwrap();
        assert_eq!(res.status, IdentificationStatus::NonparametricallyIdentified);
        let sets: Vec<Vec<VariableId>> = res
            .estimands
            .iter()
            .map(|e| e.adjustment_set.iter().copied().collect())
            .collect();
        // Inclusion-minimal: {A,B} and {A,C} (size 2). Supersets like {A,B,C} excluded.
        assert!(sets.iter().all(|s| s.len() == 2), "sets={sets:?}");
        assert_eq!(sets.len(), 2, "sets={sets:?}");
    }

    #[test]
    fn empty_adjustment_when_no_backdoor() {
        // T -> Y only
        let mut g = Dag::with_variables(2);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let id = BackdoorIdentifier::new();
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
    fn distribution_and_path_specific_are_unsupported() {
        use causal_core::{
            Intervention, InterventionalDistributionQuery, PathSpecificEffectQuery, Value,
        };

        let mut g = Dag::with_variables(2);
        let t = DenseNodeId::from_raw(0);
        let y = DenseNodeId::from_raw(1);
        g.insert_directed(t, y).unwrap();
        let id = BackdoorIdentifier::new();
        let prep = id.prepare(&g).unwrap();
        let mut ws = IdentificationWorkspace::default();

        let dist = CausalQuery::distribution(InterventionalDistributionQuery::new(
            VariableId::from_raw(1),
            [Intervention::set(VariableId::from_raw(0), Value::f64(1.0))],
        ));
        let err = id.identify(&prep, &dist, &mut ws).unwrap_err();
        assert!(matches!(
            err,
            IdentificationError::UnsupportedQuery { message }
            if message.contains("IDC")
        ));

        let path = CausalQuery::path_specific(PathSpecificEffectQuery::binary(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ));
        let err = id.identify(&prep, &path, &mut ws).unwrap_err();
        assert!(matches!(
            err,
            IdentificationError::UnsupportedQuery { message }
            if message.contains("PathSpecific")
        ));
    }

    fn random_dag(rng: &mut causal_core::CausalRng, n: u32) -> Dag {
        let mut g = Dag::with_variables(n);
        let mut order: Vec<u32> = (0..n).collect();
        for i in (1..n as usize).rev() {
            let j = (rng.next_u64() as usize) % (i + 1);
            order.swap(i, j);
        }
        for i in 0..n as usize {
            for j in (i + 1)..n as usize {
                if rng.next_u64() % 3 == 0 {
                    let a = DenseNodeId::from_raw(order[i]);
                    let b = DenseNodeId::from_raw(order[j]);
                    let _ = g.insert_directed(a, b);
                }
            }
        }
        g
    }

    fn descendants_of(g: &Dag, t: DenseNodeId) -> std::collections::HashSet<DenseNodeId> {
        let mut s = std::collections::HashSet::new();
        let mut stack = vec![t];
        while let Some(u) = stack.pop() {
            for &c in g.children(u) {
                if s.insert(c) {
                    stack.push(c);
                }
            }
        }
        s
    }

    fn assert_valid_adjustment_sets(
        g: &Dag,
        t: DenseNodeId,
        y: DenseNodeId,
        estimands: &[causal_expr::IdentifiedEstimand],
        dsep: &mut causal_graph::DSeparationWorkspace,
    ) {
        let mutilated = remove_outgoing(g, t).unwrap();
        let descendants = descendants_of(g, t);
        for est in estimands {
            let z: Vec<DenseNodeId> = est
                .adjustment_set
                .iter()
                .map(|v| var_to_dense(*v, g).unwrap())
                .collect();
            for &zi in &z {
                assert!(
                    !descendants.contains(&zi),
                    "adjustment set contains descendant of T: {zi:?}"
                );
                assert_ne!(zi, t);
                assert_ne!(zi, y);
            }
            assert!(
                is_backdoor_adjustment(&mutilated, t, y, &z, dsep).unwrap(),
                "Z={z:?} does not block backdoors T={} Y={}",
                t.raw(),
                y.raw()
            );
        }
    }

    #[test]
    fn property_backdoor_sets_block_and_exclude_descendants() {
        let mut rng = causal_core::CausalRng::from_seed(99);
        let id = BackdoorIdentifier::new();
        let eff = crate::EfficientBackdoorIdentifier::new();
        let mut ws = IdentificationWorkspace::default();
        let mut dsep = causal_graph::DSeparationWorkspace::default();
        for _ in 0..40 {
            let n = 4 + (rng.next_u64() % 2) as u32; // 4..=5
            let g = random_dag(&mut rng, n);
            let t = DenseNodeId::from_raw((rng.next_u64() % u64::from(n)) as u32);
            let mut y = DenseNodeId::from_raw((rng.next_u64() % u64::from(n)) as u32);
            if y == t {
                y = DenseNodeId::from_raw((t.raw() + 1) % n);
            }
            let prep = id.prepare(&g).unwrap();
            let q = CausalQuery::average_effect(AverageEffectQuery::with_levels(
                VariableId::from_raw(t.raw()),
                VariableId::from_raw(y.raw()),
                // Non-binary levels: identification is graph-theoretic.
                -1.5,
                2.25,
            ));
            if let Ok(res) = id.identify(&prep, &q, &mut ws) {
                if res.status == IdentificationStatus::NonparametricallyIdentified {
                    assert_valid_adjustment_sets(&g, t, y, &res.estimands, &mut dsep);
                }
            }
            if let Ok(res) = eff.identify(&prep, &q, &mut ws) {
                if res.status == IdentificationStatus::NonparametricallyIdentified {
                    assert_valid_adjustment_sets(&g, t, y, &res.estimands, &mut dsep);
                }
            }
        }
    }
}
