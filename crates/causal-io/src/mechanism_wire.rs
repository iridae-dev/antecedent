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
}

/// Mechanism store wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MechanismStoreWire {
    /// Slots in dense node order.
    pub slots: Vec<MechanismSlotWire>,
}

/// Encode mechanism store.
#[must_use]
pub fn mechanisms_to_wire(store: &CompiledMechanismStore) -> MechanismStoreWire {
    MechanismStoreWire {
        slots: store.slots.iter().map(slot_to_wire).collect(),
    }
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

fn slot_to_wire(s: &MechanismSlot) -> MechanismSlotWire {
    match s {
        MechanismSlot::Vacant => MechanismSlotWire::Vacant,
        MechanismSlot::Pending { family_id } => {
            MechanismSlotWire::Pending { family_id: family_id.to_string() }
        }
        MechanismSlot::LinearGaussian { intercept, coeffs, sigma } => {
            MechanismSlotWire::LinearGaussian {
                intercept: *intercept,
                coeffs: coeffs.to_vec(),
                sigma: *sigma,
            }
        }
        MechanismSlot::Discrete { support, probs, logit_coeffs } => MechanismSlotWire::Discrete {
            support: support.to_vec(),
            probs: probs.to_vec(),
            logit_coeffs: logit_coeffs.as_ref().map(|c| c.to_vec()),
        },
        MechanismSlot::Constant { value } => MechanismSlotWire::Constant { value: *value },
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
    }
}
