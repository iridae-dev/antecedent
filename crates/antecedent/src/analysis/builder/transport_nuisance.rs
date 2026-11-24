//! Caller-supplied outcome regressions of a trial-to-target transport study.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::StudyBuilder;
use crate::analysis::transport_supplied_nuisance::TransportOutcomeRegressions;

impl StudyBuilder {
    /// Caller-supplied trial outcome regressions for a [`CausalQuery::Transport`] study with
    /// [`Self::transport_trial`] columns.
    ///
    /// The route reads both columns on every row and evaluates the augmented trial-to-target
    /// contrast of `antecedent_estimate::trial_to_target_effect` with them, beside the IPW
    /// contrast. The published estimate, its standard error and its interval stay the IPW
    /// ones (the cell's calibration is measured for them); the augmented contrast is carried
    /// in the transport payload ([`crate::StudyResult::transport`]) and stated in the
    /// `estimate.transport.trial_aipw_supplied` diagnostic, with its delta-method standard
    /// error when a weighted target is declared.
    ///
    /// Frequentist inference on unweighted rows of one selection diagram only: a Bayesian
    /// study, observation weights, a graph class or a graph posterior refuse
    /// (`option_not_applicable`), and so does a study whose query is not a transport.
    ///
    /// [`CausalQuery::Transport`]: antecedent_core::CausalQuery::Transport
    #[must_use]
    pub fn transport_outcome_regressions(
        mut self,
        regressions: TransportOutcomeRegressions,
    ) -> Self {
        self.declared.transport_outcome_regressions = Some(regressions);
        self
    }
}
