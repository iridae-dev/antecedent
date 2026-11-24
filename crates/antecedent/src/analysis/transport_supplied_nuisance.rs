//! Caller-supplied nuisance columns of a trial-to-target transport.
//!
//! The compiled trial-to-target route ([`crate::CausalQuery::Transport`] with
//! [`crate::StudyBuilder::transport_trial`]) reads known selection and treatment probabilities
//! from the table. A caller who also holds trial outcome regressions, fitted wherever they were
//! fitted, names their columns here and the route evaluates the same equations the kernel
//! evaluates on those values. Nothing here fits a nuisance or chooses a bandwidth.
//!
//! * [`TransportOutcomeRegressions`] are the two trial outcome regressions
//!   `E[Y | X, S = 1, A = a]` evaluated on every row. The route then also forms the augmented
//!   (AIPW) contrast of Dahabreh et al. (2019) beside the IPW contrast it publishes; see
//!   [`crate::StudyBuilder::transport_outcome_regressions`].
//! * [`TransportResponseGridColumns`] are the observed-treatment regression and, per point of
//!   the response grid, the regression at that point and the caller's local response weight:
//!   the inputs of the augmented transported response grid, evaluated by
//!   [`crate::analysis::transport_response_grid`].
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::VariableId;

/// Trial outcome regressions under control and under treatment, one column each, evaluated on
/// every row of the table (trial and target rows alike).
///
/// The columns are values the caller supplies; the route reads them, it does not fit them.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub struct TransportOutcomeRegressions {
    /// `E[Y | X, S = 1, A = 0]` on every row.
    pub control: VariableId,
    /// `E[Y | X, S = 1, A = 1]` on every row.
    pub treated: VariableId,
}

impl TransportOutcomeRegressions {
    /// The regressions under control and under treatment.
    #[must_use]
    pub const fn new(control: VariableId, treated: VariableId) -> Self {
        Self { control, treated }
    }
}

/// The caller-supplied regressions and local response weights of an augmented transported
/// response grid, one column per grid point, in grid order.
///
/// `grid_regressions[g]` holds the source outcome regression at grid point `g` evaluated on
/// every row; `response_weights[g]` holds the caller's treatment-density or local-smoothing
/// weight of every row at that point (signed weights are accepted, as local-linear smoothers
/// need near a boundary). `observed_regression` holds the regression at the observed source
/// treatment of every row.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub struct TransportResponseGridColumns {
    /// The regression at each row's observed treatment.
    pub observed_regression: VariableId,
    /// One regression column per grid point.
    pub grid_regressions: Arc<[VariableId]>,
    /// One response-weight column per grid point.
    pub response_weights: Arc<[VariableId]>,
}

impl TransportResponseGridColumns {
    /// The observed-treatment regression, and the per-point regressions and weights in grid
    /// order.
    #[must_use]
    pub fn new(
        observed_regression: VariableId,
        grid_regressions: impl Into<Arc<[VariableId]>>,
        response_weights: impl Into<Arc<[VariableId]>>,
    ) -> Self {
        Self {
            observed_regression,
            grid_regressions: grid_regressions.into(),
            response_weights: response_weights.into(),
        }
    }
}
