//! Reference covariate and negative-control selection.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    CausalQuery, DSeparationWorkspace, DenseNodeId, PreparedRefit, TargetRefitError, VariableId,
    lagged_nodes, static_node, unsupported,
};

impl PreparedRefit<'_> {
    pub(super) fn adjustment(&self) -> Vec<VariableId> {
        self.result.estimand.adjustment_set.iter().copied().collect()
    }

    /// Covariates of the reference outcome model besides the treatment: the estimand's
    /// adjustment set, or, for a structural-model query (a counterfactual or a nested natural
    /// effect on a DAG), which carries no adjustment set, the other float parents of the
    /// outcome, so the reference model does not fold a confounder into the treatment slope.
    pub(super) fn reference_covariates(&self) -> Vec<VariableId> {
        let adjustment = self.adjustment();
        if !adjustment.is_empty()
            || !matches!(
                self.study().query,
                CausalQuery::Counterfactual(_)
                    | CausalQuery::NestedCounterfactual(_)
                    | CausalQuery::PathSpecific(_)
                    | CausalQuery::Distribution(_)
            )
        {
            return adjustment;
        }
        let mut parents: Vec<VariableId> = Vec::new();
        if let Some(posterior) = &self.study().graph_posterior {
            // A posterior carries no single structure (its placeholder graph has no edges): the
            // reference model takes the outcome's parents over every DAG atom, so it does not
            // fold an atom's confounder into the treatment slope.
            if posterior.atom_kind() == antecedent_discovery::GraphPosteriorAtomKind::Dag {
                for atom in posterior.atoms().iter() {
                    let Ok(dag) = antecedent_discovery::adjacency_mask::dag_from_adjacency_mask(
                        atom.clone(),
                        posterior.n_vars(),
                    ) else {
                        continue;
                    };
                    if let Some(outcome) = static_node(&dag, self.outcome) {
                        parents.extend(dag.parents(outcome).iter().map(|parent| {
                            VariableId::from_raw(
                                u32::try_from(parent.as_usize()).unwrap_or(u32::MAX),
                            )
                        }));
                    }
                }
            }
        } else if let Some(dag) = self.study().graph.as_dag() {
            if let Some(outcome) = static_node(dag, self.outcome) {
                parents.extend(dag.parents(outcome).iter().filter_map(|parent| {
                    match dag.nodes().get(parent.as_usize())? {
                        antecedent_core::NodeRef::Static(variable) => Some(*variable),
                        _ => None,
                    }
                }));
            }
        } else if let Some(tdag) = self.study().graph.as_temporal_dag() {
            for (outcome, lag) in lagged_nodes(tdag, self.outcome) {
                if lag != 0 {
                    continue;
                }
                parents.extend(tdag.parents(outcome).iter().filter_map(|parent| {
                    match tdag.nodes().get(parent.as_usize())? {
                        antecedent_core::NodeRef::Lagged { variable, .. } => Some(*variable),
                        _ => None,
                    }
                }));
            }
        }
        parents.sort_unstable();
        parents.dedup();
        parents.retain(|variable| {
            *variable != self.treatment && *variable != self.outcome && self.is_float(*variable)
        });
        if parents.is_empty() { adjustment } else { parents }
    }

    /// Negative-control exposure: declared, or the graph's first candidate.
    pub(super) fn control_exposure(&self) -> Result<VariableId, TargetRefitError> {
        if let Some(id) = self.options.negative_control_exposure {
            return Ok(id);
        }
        if self.study().graph_posterior.is_some() {
            return Err(unsupported(
                "a graph posterior has no single structure to read a negative-control exposure \
                 from (its placeholder graph has no edges, so every column would read as \
                 unrelated to the treatment and the outcome)",
                "declare a negative-control exposure in TargetRefitOptions",
            ));
        }
        let adjustment = self.adjustment();
        let n = self.storage().columns().len();
        let candidates = (0..n)
            .filter_map(|i| u32::try_from(i).ok().map(VariableId::from_raw))
            .filter(|v| *v != self.treatment && *v != self.outcome && !adjustment.contains(v))
            .filter(|v| self.is_float(*v));
        let graph = &self.study().graph;
        for v in candidates {
            let ok = if let Some(dag) = graph.as_dag() {
                let (Some(t), Some(y), Some(x)) = (
                    static_node(dag, self.treatment),
                    static_node(dag, self.outcome),
                    static_node(dag, v),
                ) else {
                    continue;
                };
                let z: Vec<DenseNodeId> =
                    adjustment.iter().filter_map(|a| static_node(dag, *a)).collect();
                let mut ws = DSeparationWorkspace::default();
                !dag.reaches(t, x)
                    && !dag.reaches(x, y)
                    && dag.is_d_separated(x, y, &z, &mut ws).unwrap_or(false)
            } else if let Some(tdag) = graph.as_temporal_dag() {
                let ys = lagged_nodes(tdag, self.outcome);
                let ts = lagged_nodes(tdag, self.treatment);
                let xs = lagged_nodes(tdag, v);
                !xs.is_empty()
                    && xs.iter().all(|(x, _)| ys.iter().all(|(y, _)| !tdag.reaches(*x, *y)))
                    && ts.iter().all(|(t, _)| xs.iter().all(|(x, _)| !tdag.reaches(*t, *x)))
            } else {
                false
            };
            if ok {
                return Ok(v);
            }
        }
        Err(unsupported(
            "the graph supplies no negative-control exposure: no float variable outside the \
             adjustment set is separated from the outcome and unaffected by the treatment",
            "declare a negative-control exposure in TargetRefitOptions",
        ))
    }

    /// Negative-control outcome on a static graph: declared, or the graph's first candidate.
    pub(super) fn control_outcome(&self) -> Result<VariableId, TargetRefitError> {
        if let Some(id) = self.options.negative_control_outcome {
            return Ok(id);
        }
        if self.study().graph_posterior.is_some() {
            return Err(unsupported(
                "a graph posterior has no single structure to read a negative-control outcome \
                 from (its placeholder graph has no edges)",
                "declare a negative-control outcome in TargetRefitOptions",
            ));
        }
        let adjustment = self.adjustment();
        let Some(dag) = self.study().graph.as_dag() else {
            return Err(unsupported(
                "the graph is not a DAG, so no negative-control outcome can be read from it",
                "declare a negative-control outcome in TargetRefitOptions",
            ));
        };
        let n = self.storage().columns().len();
        let (Some(t), Some(_)) = (static_node(dag, self.treatment), static_node(dag, self.outcome))
        else {
            return Err(unsupported(
                "the treatment or outcome is not a node of the graph",
                "declare a negative-control outcome in TargetRefitOptions",
            ));
        };
        let z: Vec<DenseNodeId> = adjustment.iter().filter_map(|a| static_node(dag, *a)).collect();
        for i in 0..n {
            let Ok(raw) = u32::try_from(i) else { continue };
            let v = VariableId::from_raw(raw);
            if v == self.treatment
                || v == self.outcome
                || adjustment.contains(&v)
                || !self.is_float(v)
            {
                continue;
            }
            let Some(w) = static_node(dag, v) else {
                continue;
            };
            let mut ws = DSeparationWorkspace::default();
            if !dag.reaches(t, w) && dag.is_d_separated(t, w, &z, &mut ws).unwrap_or(false) {
                return Ok(v);
            }
        }
        Err(unsupported(
            "the graph supplies no negative-control outcome: no float variable outside the \
             adjustment set is unaffected by and separated from the treatment",
            "declare a negative-control outcome in TargetRefitOptions",
        ))
    }
}
