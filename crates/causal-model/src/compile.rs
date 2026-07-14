//! Compiled topological execution plans (DESIGN.md §15.1).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation)]

use std::sync::Arc;

use causal_core::VariableId;
use causal_graph::{Dag, DenseNodeId, NodeRef};

use crate::error::ModelError;

/// Plan for gathering parent values into an aligned buffer for one child node.
#[derive(Clone, Debug)]
pub struct ParentGatherPlan {
    /// Child dense id.
    pub child: DenseNodeId,
    /// Parent dense ids in gather order.
    pub parents: Arc<[DenseNodeId]>,
}

impl ParentGatherPlan {
    /// Number of parents.
    #[must_use]
    pub fn n_parents(&self) -> usize {
        self.parents.len()
    }

    /// Gather parent columns from a column-major value buffer into `out`
    /// (`parent * n_rows + row`).
    pub fn gather(&self, values: &[f64], n_rows: usize, out: &mut [f64]) {
        debug_assert!(out.len() >= self.parents.len().saturating_mul(n_rows));
        for (pi, &p) in self.parents.iter().enumerate() {
            let src = p.as_usize() * n_rows;
            let dst = pi * n_rows;
            out[dst..dst + n_rows].copy_from_slice(&values[src..src + n_rows]);
        }
    }
}

/// Layout of sampled outputs.
#[derive(Clone, Debug)]
pub struct ModelOutputLayout {
    /// Dense node order (same as compile topo order).
    pub node_order: Arc<[DenseNodeId]>,
    /// Variable id per dense node (static graphs).
    pub variables: Arc<[VariableId]>,
}

/// Mechanism slot filled by fitting / registry.
#[derive(Clone, Debug, Default)]
pub enum MechanismSlot {
    /// Unassigned.
    #[default]
    Vacant,
    /// Assigned family id pending fit.
    Pending {
        /// Family registry id.
        family_id: Arc<str>,
    },
    /// Fitted linear Gaussian: intercept + parent coeffs + residual σ.
    LinearGaussian {
        /// Intercept.
        intercept: f64,
        /// Coefficients aligned with [`ParentGatherPlan::parents`].
        coeffs: Arc<[f64]>,
        /// Residual standard deviation.
        sigma: f64,
    },
    /// Discrete categorical over finite support.
    ///
    /// Unconditional when `logit_coeffs` is `None` (use `probs`).
    /// Parent-conditional when `logit_coeffs` is `Some`: softmax over
    /// `support.len()` rows of length `1 + n_parents` (intercept + parent coeffs).
    Discrete {
        /// Support values.
        support: Arc<[f64]>,
        /// Unconditional probabilities (same length as support); ignored when
        /// `logit_coeffs` is set.
        probs: Arc<[f64]>,
        /// Optional softmax logit coefficients, row-major `[k * (1 + p) + j]`.
        logit_coeffs: Option<Arc<[f64]>>,
    },
    /// Constant mechanism.
    Constant {
        /// Fixed value.
        value: f64,
    },
}

/// Per-node mechanism storage for a compiled model.
#[derive(Clone, Debug)]
pub struct CompiledMechanismStore {
    /// Slot per dense node id (index = dense raw).
    pub slots: Arc<[MechanismSlot]>,
}

impl CompiledMechanismStore {
    /// Vacant slots for `n` nodes.
    #[must_use]
    pub fn vacant(n: usize) -> Self {
        Self { slots: Arc::from(vec![MechanismSlot::Vacant; n]) }
    }

    /// Slot for dense id.
    #[must_use]
    pub fn get(&self, id: DenseNodeId) -> &MechanismSlot {
        &self.slots[id.as_usize()]
    }
}

/// Immutable compiled causal model plan (DESIGN.md §15.1).
#[derive(Clone, Debug)]
pub struct CompiledCausalModel {
    /// Topological dense node order.
    pub node_order: Arc<[DenseNodeId]>,
    /// Parent gather plans aligned with `node_order`.
    pub parent_gathers: Arc<[ParentGatherPlan]>,
    /// Mechanisms per dense node.
    pub mechanisms: CompiledMechanismStore,
    /// Output layout.
    pub output_layout: ModelOutputLayout,
    /// Source DAG (shared; never cloned per intervention).
    pub graph: Arc<Dag>,
}

impl CompiledCausalModel {
    /// Compile a static DAG into a topological execution plan.
    ///
    /// Mechanisms start vacant; assignment/fit fills them .
    ///
    /// # Errors
    ///
    /// Cyclic graph or non-static nodes.
    pub fn compile(graph: Dag) -> Result<Self, ModelError> {
        let order = graph.topological_order().ok_or_else(|| ModelError::NotDag {
            message: "graph has no topological order".into(),
        })?;
        let n = graph.node_count();
        let mut variables = Vec::with_capacity(n);
        for i in 0..n {
            let id = DenseNodeId::from_raw(i as u32);
            match graph.nodes().get(i) {
                Some(NodeRef::Static(v)) => variables.push(*v),
                Some(other) => {
                    return Err(ModelError::Unsupported {
                        message: format!(
                            "CompiledCausalModel requires Static nodes, got {other:?}"
                        ),
                    });
                }
                None => {
                    return Err(ModelError::Shape { message: "node missing".into() });
                }
            }
            let _ = id;
        }
        let mut gathers = Vec::with_capacity(order.len());
        for &child in &order {
            let parents = graph.parents(child).to_vec();
            gathers.push(ParentGatherPlan { child, parents: Arc::from(parents) });
        }
        let node_order = Arc::from(order);
        Ok(Self {
            output_layout: ModelOutputLayout {
                node_order: Arc::clone(&node_order),
                variables: Arc::from(variables),
            },
            node_order,
            parent_gathers: Arc::from(gathers),
            mechanisms: CompiledMechanismStore::vacant(n),
            graph: Arc::new(graph),
        })
    }

    /// Number of nodes.
    #[must_use]
    pub fn n_nodes(&self) -> usize {
        self.graph.node_count()
    }

    /// Dense id for a variable, if present.
    #[must_use]
    pub fn dense_of(&self, var: VariableId) -> Option<DenseNodeId> {
        self.output_layout
            .variables
            .iter()
            .position(|v| *v == var)
            .map(|i| DenseNodeId::from_raw(i as u32))
    }

    /// Replace mechanism store (fit / assignment).
    #[must_use]
    pub fn with_mechanisms(mut self, mechanisms: CompiledMechanismStore) -> Self {
        self.mechanisms = mechanisms;
        self
    }

    /// Gather plan for a child dense id.
    #[must_use]
    pub fn gather_for(&self, child: DenseNodeId) -> Option<&ParentGatherPlan> {
        self.parent_gathers.iter().find(|g| g.child == child)
    }
}

/// Probabilistic causal model (PCM): observational mechanisms without required invertibility.
#[derive(Clone, Debug)]
pub struct ProbabilisticCausalModel {
    /// Compiled plan.
    pub compiled: CompiledCausalModel,
}

impl ProbabilisticCausalModel {
    /// Wrap a compiled plan.
    #[must_use]
    pub fn new(compiled: CompiledCausalModel) -> Self {
        Self { compiled }
    }
}

/// Structural causal model (SCM): additive / structural assignments with noise.
#[derive(Clone, Debug)]
pub struct StructuralCausalModel {
    /// Compiled plan.
    pub compiled: CompiledCausalModel,
}

impl StructuralCausalModel {
    /// Wrap.
    #[must_use]
    pub fn new(compiled: CompiledCausalModel) -> Self {
        Self { compiled }
    }
}

/// Invertible SCM supporting abduction (noise inference).
#[derive(Clone, Debug)]
pub struct InvertibleStructuralCausalModel {
    /// Compiled plan.
    pub compiled: CompiledCausalModel,
}

impl InvertibleStructuralCausalModel {
    /// Wrap; caller ensures mechanisms are invertible families.
    #[must_use]
    pub fn new(compiled: CompiledCausalModel) -> Self {
        Self { compiled }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use causal_core::VariableId;
    use causal_graph::Dag;

    #[test]
    fn compile_chain_topo_order() {
        let mut g = Dag::with_variables(3);
        let a = DenseNodeId::from_raw(0);
        let b = DenseNodeId::from_raw(1);
        let c = DenseNodeId::from_raw(2);
        g.insert_directed(a, b).unwrap();
        g.insert_directed(b, c).unwrap();
        let plan = CompiledCausalModel::compile(g).unwrap();
        assert_eq!(plan.n_nodes(), 3);
        assert_eq!(plan.node_order.as_ref(), &[a, b, c]);
        assert_eq!(plan.gather_for(c).unwrap().n_parents(), 1);
        assert_eq!(plan.dense_of(VariableId::from_raw(1)), Some(b));
    }
}
