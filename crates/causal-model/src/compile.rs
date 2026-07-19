//! Compiled topological execution plans.
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

/// Slow-path dynamic mechanism.
///
/// Built-ins stay on concrete [`MechanismSlot`] variants; user/Python wrappers
/// implement this trait and live in [`MechanismSlot::Dynamic`].
pub trait DynamicMechanism: Send + Sync {
    /// Sample structural noise into `output` (length ≥ `n_rows`).
    ///
    /// # Errors
    ///
    /// Shape / unsupported.
    fn sample_noise_column(
        &self,
        n_rows: usize,
        rng: &mut causal_core::CausalRng,
        output: &mut [f64],
    ) -> Result<(), ModelError>;

    /// Evaluate `x = f(parents, noise)` into `output`.
    ///
    /// # Errors
    ///
    /// Shape / unsupported.
    fn evaluate_column(
        &self,
        parents: crate::batch::ParentBatch<'_>,
        noise: &[f64],
        output: &mut [f64],
        workspace: &mut crate::batch::MechanismWorkspace,
    ) -> Result<(), ModelError>;

    /// Infer exogenous noise (optional; default unsupported).
    ///
    /// # Errors
    ///
    /// Shape / unsupported.
    fn infer_noise_column(
        &self,
        _value: &[f64],
        _parents: crate::batch::ParentBatch<'_>,
        _output: &mut [f64],
    ) -> Result<(), ModelError> {
        Err(ModelError::Unsupported {
            message: "dynamic mechanism does not support noise inference".into(),
        })
    }

    /// Log-density of observed values (optional; default unsupported).
    ///
    /// # Errors
    ///
    /// Shape / unsupported.
    fn log_prob_column(
        &self,
        _values: &[f64],
        _parents: crate::batch::ParentBatch<'_>,
        _output: &mut [f64],
    ) -> Result<(), ModelError> {
        Err(ModelError::Unsupported {
            message: "dynamic mechanism does not support log_prob".into(),
        })
    }
}

/// Mechanism slot filled by fitting / registry.
#[derive(Clone, Default)]
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
    /// Coefficients are baseline-category multinomial-logit MLEs (reference category
    /// index 0 is pinned to zero).
    Discrete {
        /// Support values.
        support: Arc<[f64]>,
        /// Unconditional probabilities (same length as support); ignored when
        /// `logit_coeffs` is set.
        probs: Arc<[f64]>,
        /// Optional softmax logit coefficients, row-major `[k * (1 + p)]`.
        logit_coeffs: Option<Arc<[f64]>>,
    },
    /// Constant mechanism.
    Constant {
        /// Fixed value.
        value: f64,
    },
    /// Hierarchical linear Gaussian (partial-pooling / ridge toward prior mean 0).
    HierarchicalLinear {
        /// Intercept.
        intercept: f64,
        /// Parent coefficients (shrunk).
        coeffs: Arc<[f64]>,
        /// Residual standard deviation.
        sigma: f64,
        /// Shrinkage strength used at fit (`λ` on diagonal of XtX).
        shrinkage: f64,
    },
    /// Bayesian VAR-style linear Gaussian on parent lags (single-equation).
    Bvar {
        /// Intercept.
        intercept: f64,
        /// Parent / lag coefficients.
        coeffs: Arc<[f64]>,
        /// Residual standard deviation.
        sigma: f64,
    },
    /// Linear Gaussian state-space observation mechanism (1-D LGSSM).
    ///
    /// Latent: `x_t = a x_{t-1} + σ_proc ε`; observation: `y_t = x_t + σ_obs η`.
    /// Parents unused at evaluate time (state evolves from shared noise).
    LinearGaussianStateSpace {
        /// AR coefficient.
        a: f64,
        /// Process noise std.
        process_std: f64,
        /// Observation noise std.
        obs_std: f64,
        /// Initial latent mean.
        initial_mean: f64,
    },
    /// Gaussian-process mechanism (RBF dual form); requires `gaussian-process` feature to fit.
    GaussianProcess {
        /// Length scale.
        length_scale: f64,
        /// Signal variance.
        variance: f64,
        /// Observation noise std.
        noise_std: f64,
        /// Training parent rows, row-major `[n_train * n_parents]`.
        x_train: Arc<[f64]>,
        /// Training rows.
        n_train: usize,
        /// Parent arity.
        n_parents: usize,
        /// Dual coefficients `α = (K + σ²I)^{-1} y`.
        alpha: Arc<[f64]>,
    },
    /// Explicit slow-path dynamic / user mechanism (not serializable).
    Dynamic {
        /// Stable label for diagnostics (e.g. variable name).
        id: Arc<str>,
        /// Object-safe mechanism implementation.
        mechanism: Arc<dyn DynamicMechanism>,
    },
}

impl std::fmt::Debug for MechanismSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Vacant => write!(f, "Vacant"),
            Self::Pending { family_id } => f.debug_struct("Pending").field("family_id", family_id).finish(),
            Self::LinearGaussian { intercept, coeffs, sigma } => f
                .debug_struct("LinearGaussian")
                .field("intercept", intercept)
                .field("coeffs", coeffs)
                .field("sigma", sigma)
                .finish(),
            Self::Discrete { support, probs, logit_coeffs } => f
                .debug_struct("Discrete")
                .field("support", support)
                .field("probs", probs)
                .field("logit_coeffs", logit_coeffs)
                .finish(),
            Self::Constant { value } => f.debug_struct("Constant").field("value", value).finish(),
            Self::HierarchicalLinear { intercept, coeffs, sigma, shrinkage } => f
                .debug_struct("HierarchicalLinear")
                .field("intercept", intercept)
                .field("coeffs", coeffs)
                .field("sigma", sigma)
                .field("shrinkage", shrinkage)
                .finish(),
            Self::Bvar { intercept, coeffs, sigma } => f
                .debug_struct("Bvar")
                .field("intercept", intercept)
                .field("coeffs", coeffs)
                .field("sigma", sigma)
                .finish(),
            Self::LinearGaussianStateSpace { a, process_std, obs_std, initial_mean } => f
                .debug_struct("LinearGaussianStateSpace")
                .field("a", a)
                .field("process_std", process_std)
                .field("obs_std", obs_std)
                .field("initial_mean", initial_mean)
                .finish(),
            Self::GaussianProcess {
                length_scale,
                variance,
                noise_std,
                n_train,
                n_parents,
                ..
            } => f
                .debug_struct("GaussianProcess")
                .field("length_scale", length_scale)
                .field("variance", variance)
                .field("noise_std", noise_std)
                .field("n_train", n_train)
                .field("n_parents", n_parents)
                .finish(),
            Self::Dynamic { id, .. } => f
                .debug_struct("Dynamic")
                .field("id", id)
                .field("mechanism", &"<dyn DynamicMechanism>")
                .finish(),
        }
    }
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

    /// Replace the slot at `id`, returning a new store (copy-on-write).
    ///
    /// # Errors
    ///
    /// Out-of-range dense id.
    pub fn with_replaced(
        &self,
        id: DenseNodeId,
        slot: MechanismSlot,
    ) -> Result<Self, ModelError> {
        let idx = id.as_usize();
        if idx >= self.slots.len() {
            return Err(ModelError::Shape {
                message: "mechanism slot index out of range".into(),
            });
        }
        let mut slots = self.slots.as_ref().to_vec();
        slots[idx] = slot;
        Ok(Self { slots: Arc::from(slots) })
    }
}

/// Immutable compiled causal model plan.
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
