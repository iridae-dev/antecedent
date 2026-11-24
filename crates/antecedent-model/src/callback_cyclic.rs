//! A declared batch callback as a local differentiable feedback mechanism.
//!
//! [`CallbackCyclicMechanism`] lets a [`LocalCyclicModel`](crate::local_cyclic::LocalCyclicModel)
//! run a [`BatchCallback`] that provides its derivatives
//! ([`BatchCallback::value_and_derivatives`]). The callback's declaration (name, input labels,
//! output, pinned dependencies, implementation identity) is what a compiled program binds; its
//! code stays process-local. The Newton solve and the additive abduction are those of
//! `LocalCyclicModel` unchanged: the adapter only forwards each evaluation.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{BatchCallback, CallbackError, CallbackFailure};

use crate::local_cyclic::LocalCyclicMechanism;
use crate::nonlinear_equilibrium::EquilibriumError;

/// One feedback mechanism whose value and derivatives come from a declared callback.
#[derive(Clone, Debug)]
pub struct CallbackCyclicMechanism {
    id: String,
    reads: Vec<usize>,
    callback: Arc<dyn BatchCallback>,
}

impl CallbackCyclicMechanism {
    /// Bind `callback` as the mechanism `id` reading the endogenous coordinates `reads`, in the
    /// callback's declared input order.
    ///
    /// # Errors
    ///
    /// An empty id, a malformed declaration, or a declaration whose input count differs from
    /// `reads`.
    pub fn new(
        id: impl Into<String>,
        reads: Vec<usize>,
        callback: Arc<dyn BatchCallback>,
    ) -> Result<Self, EquilibriumError> {
        let id = id.into();
        let spec = callback.spec();
        if id.is_empty() || !spec.is_well_formed() || spec.inputs.len() != reads.len() {
            return Err(EquilibriumError::Invalid(format!(
                "callback mechanism {id}: the declaration must be well formed and declare one input \
                 per read coordinate"
            )));
        }
        Ok(Self { id, reads, callback })
    }

    /// The bound callback.
    #[must_use]
    pub fn callback(&self) -> &Arc<dyn BatchCallback> {
        &self.callback
    }
}

impl LocalCyclicMechanism for CallbackCyclicMechanism {
    fn id(&self) -> &str {
        &self.id
    }

    fn reads(&self) -> &[usize] {
        &self.reads
    }

    fn evaluate(&self, inputs: &[f64]) -> Result<(f64, Vec<f64>), EquilibriumError> {
        self.callback.value_and_derivatives(inputs).map_err(|error| match error {
            CallbackError::Failed(failure) => {
                EquilibriumError::CallbackFailed { mechanism: self.id.clone(), failure }
            }
            CallbackError::Raised { message } => EquilibriumError::CallbackFailed {
                mechanism: self.id.clone(),
                failure: CallbackFailure::new(None, message, None),
            },
            other => EquilibriumError::Invalid(format!("local callback {}: {other}", self.id)),
        })
    }
}
