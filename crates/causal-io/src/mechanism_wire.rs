//! Mechanism store wire types.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_model::{CompiledMechanismStore, MechanismSlot};
use serde::{Deserialize, Serialize};

use crate::error::IoError;

/// Model kind tag for bundles.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelKindWire {
    /// Probabilistic causal model.
    Pcm,
    /// Structural causal model.
    Scm,
    /// Invertible SCM.
    InvertibleScm,
}

/// One mechanism slot on the wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum MechanismSlotWire {
    /// Vacant.
    Vacant,
    /// Pending fit.
    Pending {
        /// Family id.
        family_id: String,
    },
    /// Linear Gaussian.
    LinearGaussian {
        /// Intercept.
        intercept: f64,
        /// Parent coeffs.
        coeffs: Vec<f64>,
        /// Residual σ.
        sigma: f64,
    },
    /// Discrete.
    Discrete {
        /// Support.
        support: Vec<f64>,
        /// Unconditional probs.
        probs: Vec<f64>,
        /// Optional logit coeffs.
        logit_coeffs: Option<Vec<f64>>,
    },
    /// Constant.
    Constant {
        /// Value.
        value: f64,
    },
    /// Hierarchical linear Gaussian.
    HierarchicalLinear {
        /// Intercept.
        intercept: f64,
        /// Coeffs.
        coeffs: Vec<f64>,
        /// Sigma.
        sigma: f64,
        /// Shrinkage.
        shrinkage: f64,
    },
    /// BVAR-style linear.
    Bvar {
        /// Intercept.
        intercept: f64,
        /// Coeffs.
        coeffs: Vec<f64>,
        /// Sigma.
        sigma: f64,
    },
    /// LGSSM.
    LinearGaussianStateSpace {
        /// AR.
        a: f64,
        /// Process std.
        process_std: f64,
        /// Obs std.
        obs_std: f64,
        /// Initial mean.
        initial_mean: f64,
    },
    /// GP dual form.
    GaussianProcess {
        /// Length scale.
        length_scale: f64,
        /// Variance.
        variance: f64,
        /// Noise std.
        noise_std: f64,
        /// Training X row-major.
        x_train: Vec<f64>,
        /// n_train.
        n_train: usize,
        /// n_parents.
        n_parents: usize,
        /// Dual coefficients.
        alpha: Vec<f64>,
    },
}

/// Mechanism store wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MechanismStoreWire {
    /// Slots in dense node order.
    pub slots: Vec<MechanismSlotWire>,
}

/// Encode mechanism store.
///
/// # Errors
///
/// [`MechanismSlot::Dynamic`] cannot be serialized (DESIGN §24.4 / §25.4).
pub fn mechanisms_to_wire(store: &CompiledMechanismStore) -> Result<MechanismStoreWire, IoError> {
    let mut slots = Vec::with_capacity(store.slots.len());
    for s in store.slots.iter() {
        slots.push(slot_to_wire(s)?);
    }
    Ok(MechanismStoreWire { slots })
}

/// Decode mechanism store.
///
/// # Errors
///
/// Never today; reserved for validation.
pub fn mechanisms_from_wire(w: &MechanismStoreWire) -> Result<CompiledMechanismStore, IoError> {
    Ok(CompiledMechanismStore {
        slots: w.slots.iter().map(slot_from_wire).collect::<Vec<_>>().into(),
    })
}

fn slot_to_wire(s: &MechanismSlot) -> Result<MechanismSlotWire, IoError> {
    match s {
        MechanismSlot::Vacant => Ok(MechanismSlotWire::Vacant),
        MechanismSlot::Pending { family_id } => {
            Ok(MechanismSlotWire::Pending { family_id: family_id.to_string() })
        }
        MechanismSlot::LinearGaussian { intercept, coeffs, sigma } => {
            Ok(MechanismSlotWire::LinearGaussian {
                intercept: *intercept,
                coeffs: coeffs.to_vec(),
                sigma: *sigma,
            })
        }
        MechanismSlot::Discrete { support, probs, logit_coeffs } => {
            Ok(MechanismSlotWire::Discrete {
                support: support.to_vec(),
                probs: probs.to_vec(),
                logit_coeffs: logit_coeffs.as_ref().map(|c| c.to_vec()),
            })
        }
        MechanismSlot::Constant { value } => Ok(MechanismSlotWire::Constant { value: *value }),
        MechanismSlot::HierarchicalLinear { intercept, coeffs, sigma, shrinkage } => {
            Ok(MechanismSlotWire::HierarchicalLinear {
                intercept: *intercept,
                coeffs: coeffs.to_vec(),
                sigma: *sigma,
                shrinkage: *shrinkage,
            })
        }
        MechanismSlot::Bvar { intercept, coeffs, sigma } => Ok(MechanismSlotWire::Bvar {
            intercept: *intercept,
            coeffs: coeffs.to_vec(),
            sigma: *sigma,
        }),
        MechanismSlot::LinearGaussianStateSpace { a, process_std, obs_std, initial_mean } => {
            Ok(MechanismSlotWire::LinearGaussianStateSpace {
                a: *a,
                process_std: *process_std,
                obs_std: *obs_std,
                initial_mean: *initial_mean,
            })
        }
        MechanismSlot::GaussianProcess {
            length_scale,
            variance,
            noise_std,
            x_train,
            n_train,
            n_parents,
            alpha,
        } => Ok(MechanismSlotWire::GaussianProcess {
            length_scale: *length_scale,
            variance: *variance,
            noise_std: *noise_std,
            x_train: x_train.to_vec(),
            n_train: *n_train,
            n_parents: *n_parents,
            alpha: alpha.to_vec(),
        }),
        MechanismSlot::Dynamic { id, .. } => Err(IoError::Convert(format!(
            "cannot serialize Dynamic mechanism slot `{id}` (Python/user callbacks are not artifact-safe)"
        ))),
    }
}

fn slot_from_wire(s: &MechanismSlotWire) -> MechanismSlot {
    match s {
        MechanismSlotWire::Vacant => MechanismSlot::Vacant,
        MechanismSlotWire::Pending { family_id } => {
            MechanismSlot::Pending { family_id: Arc::from(family_id.as_str()) }
        }
        MechanismSlotWire::LinearGaussian { intercept, coeffs, sigma } => {
            MechanismSlot::LinearGaussian {
                intercept: *intercept,
                coeffs: Arc::from(coeffs.as_slice()),
                sigma: *sigma,
            }
        }
        MechanismSlotWire::Discrete { support, probs, logit_coeffs } => MechanismSlot::Discrete {
            support: Arc::from(support.as_slice()),
            probs: Arc::from(probs.as_slice()),
            logit_coeffs: logit_coeffs.as_ref().map(|c| Arc::from(c.as_slice())),
        },
        MechanismSlotWire::Constant { value } => MechanismSlot::Constant { value: *value },
        MechanismSlotWire::HierarchicalLinear { intercept, coeffs, sigma, shrinkage } => {
            MechanismSlot::HierarchicalLinear {
                intercept: *intercept,
                coeffs: Arc::from(coeffs.as_slice()),
                sigma: *sigma,
                shrinkage: *shrinkage,
            }
        }
        MechanismSlotWire::Bvar { intercept, coeffs, sigma } => MechanismSlot::Bvar {
            intercept: *intercept,
            coeffs: Arc::from(coeffs.as_slice()),
            sigma: *sigma,
        },
        MechanismSlotWire::LinearGaussianStateSpace { a, process_std, obs_std, initial_mean } => {
            MechanismSlot::LinearGaussianStateSpace {
                a: *a,
                process_std: *process_std,
                obs_std: *obs_std,
                initial_mean: *initial_mean,
            }
        }
        MechanismSlotWire::GaussianProcess {
            length_scale,
            variance,
            noise_std,
            x_train,
            n_train,
            n_parents,
            alpha,
        } => MechanismSlot::GaussianProcess {
            length_scale: *length_scale,
            variance: *variance,
            noise_std: *noise_std,
            x_train: Arc::from(x_train.as_slice()),
            n_train: *n_train,
            n_parents: *n_parents,
            alpha: Arc::from(alpha.as_slice()),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use causal_model::DynamicMechanism;
    use std::sync::Arc;

    struct StubMech;
    impl DynamicMechanism for StubMech {
        fn sample_noise_column(
            &self,
            n_rows: usize,
            _rng: &mut causal_core::CausalRng,
            output: &mut [f64],
        ) -> Result<(), causal_model::ModelError> {
            output[..n_rows].fill(0.0);
            Ok(())
        }
        fn evaluate_column(
            &self,
            parents: causal_model::ParentBatch<'_>,
            _noise: &[f64],
            output: &mut [f64],
            _ws: &mut causal_model::MechanismWorkspace,
        ) -> Result<(), causal_model::ModelError> {
            output[..parents.n_rows].fill(0.0);
            Ok(())
        }
    }

    #[test]
    fn dynamic_slot_refuses_serialization() {
        let store = CompiledMechanismStore {
            slots: Arc::from([MechanismSlot::Dynamic {
                id: Arc::from("y"),
                mechanism: Arc::new(StubMech),
            }]),
        };
        let err = mechanisms_to_wire(&store).unwrap_err();
        assert!(matches!(err, IoError::Convert(_)));
    }
}
