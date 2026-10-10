//! Uncertain pre-action state of the finite two-step sequence (2.3A, X5).
//!
//! The target two-step mean marginalized over the initial state is
//! `sum_{s0} P_target(s0) * R(s0)`, where `R(s0)` is the whole sequence's
//! response given `s0` read from the histories (the history-conditioned
//! functional): `R(s0) = sum_l P(l | s0, a1) E[y | s0, a1, l, a2]`. The initial
//! state is supplied as a finite target law with its own snapshot id; a source
//! law or a single state cannot answer a target-marginal query. Fixing the
//! state is a different estimand with its own result type, and neither type
//! converts into the other. Point-only: no interval claim.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use antecedent_core::reason_code;

use crate::error::EstimationError;
use crate::temporal_dependent_interval::{
    ResponseGap, SequenceTallies, TemporalEstimator, TemporalUnitPanel, UnitHistories, fold,
    fold_text,
};

/// Most support points of an initial-state law.
pub const INITIAL_STATE_MAX_STATES: usize = 64;
/// Largest deviation of a law's total mass from one.
pub const INITIAL_STATE_MASS_TOLERANCE: f64 = 1e-9;
/// The only inference claim of this route.
pub const INITIAL_STATE_INFERENCE_CLAIM: &str = "point_only";

/// Detail: the initial-state route is closed.
pub const INITIAL_STATE_ROUTE_FROZEN: &str = "initial_state.route_frozen";
/// Detail: only a source law or a point state was supplied for a target-marginal query.
pub const INITIAL_STATE_TARGET_LAW_MISSING: &str = "initial_state.target_law_missing";
/// Detail: more support points than [`INITIAL_STATE_MAX_STATES`].
pub const INITIAL_STATE_TOO_MANY_STATES: &str = "initial_state.too_many_states";
/// Detail: a state with positive target mass has no history support.
pub const INITIAL_STATE_SUPPORT_GAP: &str = "initial_state.support_gap";

/// The refusal of the public initial-state route until its evidence passes.
#[must_use]
pub fn route_frozen_refusal() -> EstimationError {
    EstimationError::refused(
        reason_code!("cell_not_licensed"),
        format!(
            "{INITIAL_STATE_ROUTE_FROZEN}: the target initial-state distribution route has not \
             passed truth and artifact evidence"
        ),
    )
}

fn target_law_missing(message: &str) -> EstimationError {
    EstimationError::refused(
        reason_code!("transport_missing_evidence"),
        format!("{INITIAL_STATE_TARGET_LAW_MISSING}: {message}"),
    )
}

pub(crate) fn support_gap(s0: u32, gap: ResponseGap) -> EstimationError {
    let what = match gap {
        ResponseGap::NoState => "no history starts in it under the first action".to_owned(),
        ResponseGap::NoCell { l2 } => {
            format!("no history with step-2 covariate {l2} takes the second action")
        }
    };
    EstimationError::refused(
        reason_code!("transport_support_failure"),
        format!("{INITIAL_STATE_SUPPORT_GAP}: initial state {s0} has target mass but {what}"),
    )
}

/// Whose law an initial-state distribution is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InitialStatePopulation {
    /// The source population's law; it cannot stand in for the target's.
    Source,
    /// The target population's law.
    Target,
}

/// A finite law of the observed pre-action state, with its own snapshot.
#[derive(Clone, Debug, PartialEq)]
pub struct InitialStateLaw {
    population: InitialStatePopulation,
    snapshot_id: String,
    states: Vec<(u32, f64)>,
    digest: u64,
}

impl InitialStateLaw {
    /// Declare a law over at most [`INITIAL_STATE_MAX_STATES`] state levels.
    ///
    /// # Errors
    /// `route_not_supported` / `initial_state.too_many_states` above the cap; a data
    /// error for an empty, duplicated, negative, non-finite or non-normalized law
    /// or an empty snapshot id.
    pub fn new(
        population: InitialStatePopulation,
        snapshot_id: impl Into<String>,
        states: Vec<(u32, f64)>,
    ) -> Result<Self, EstimationError> {
        let snapshot_id = snapshot_id.into();
        if states.len() > INITIAL_STATE_MAX_STATES {
            return Err(EstimationError::refused(
                reason_code!("route_not_supported"),
                format!(
                    "{INITIAL_STATE_TOO_MANY_STATES}: {} support points exceed the cap of \
                     {INITIAL_STATE_MAX_STATES}",
                    states.len()
                ),
            ));
        }
        if snapshot_id.is_empty() || states.is_empty() {
            return Err(EstimationError::data_msg(
                "an initial-state law needs a snapshot id and at least one state",
            ));
        }
        let levels = states.iter().map(|s| s.0).collect::<BTreeSet<_>>();
        let total: f64 = states.iter().map(|s| s.1).sum();
        if levels.len() != states.len()
            || states.iter().any(|s| !s.1.is_finite() || s.1 < 0.0)
            || (total - 1.0).abs() > INITIAL_STATE_MASS_TOLERANCE
        {
            return Err(EstimationError::data_msg(
                "an initial-state law needs distinct states and finite non-negative masses \
                 summing to one",
            ));
        }
        let mut digest = fold_text(0x1417, &snapshot_id);
        digest = fold(digest, u64::from(population == InitialStatePopulation::Target));
        for (level, mass) in &states {
            digest = fold(fold(digest, u64::from(*level)), mass.to_bits());
        }
        Ok(Self { population, snapshot_id, states, digest })
    }

    /// Whose law this is.
    #[must_use]
    pub const fn population(&self) -> InitialStatePopulation {
        self.population
    }

    /// The law's own snapshot id.
    #[must_use]
    pub fn snapshot_id(&self) -> &str {
        &self.snapshot_id
    }

    /// Support points and masses.
    #[must_use]
    pub fn states(&self) -> &[(u32, f64)] {
        &self.states
    }

    /// Identity of the population, snapshot, states and masses.
    #[must_use]
    pub const fn digest(&self) -> u64 {
        self.digest
    }
}

/// What a caller supplies about the initial state of a marginal query.
#[derive(Clone, Debug, PartialEq)]
pub enum InitialStateSpec {
    /// One state: a fixed-state evidence, never a law.
    Point(u32),
    /// A finite law of either population.
    Law(InitialStateLaw),
}

/// One state's contribution to the marginalized response.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StateContribution {
    /// State level.
    pub s0: u32,
    /// Target mass of the state.
    pub mass: f64,
    /// Response of the whole sequence given the state, from the histories.
    pub response: f64,
}

/// The sequence response with the initial state held fixed at `s0`.
#[derive(Clone, Debug, PartialEq)]
pub struct FixedStateEffect {
    /// The response `R(s0)`.
    pub value: f64,
    /// Actions of the two steps.
    pub sequence: [u32; 2],
    /// The fixed state.
    pub s0: u32,
    /// Snapshot of the panel the histories came from.
    pub panel_snapshot_id: String,
}

impl FixedStateEffect {
    /// Estimand label.
    pub const LABEL: &'static str = "fixed_initial_state";

    /// The label of this result; never `marginalized_initial_state`.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        Self::LABEL
    }
}

/// The sequence response marginalized over a target initial-state law.
#[derive(Clone, Debug, PartialEq)]
pub struct MarginalizedEffect {
    /// `sum_s0 P_target(s0) R(s0)`.
    pub value: f64,
    /// Actions of the two steps.
    pub sequence: [u32; 2],
    /// Every positive-mass state with its mass and conditional response.
    pub contributions: Vec<StateContribution>,
    /// Snapshot id of the target initial-state law.
    pub state_snapshot_id: String,
    /// Identity of the target initial-state law.
    pub state_law_digest: u64,
    /// Snapshot of the panel the histories came from.
    pub panel_snapshot_id: String,
    /// Always [`INITIAL_STATE_INFERENCE_CLAIM`].
    pub inference_claim: &'static str,
}

impl MarginalizedEffect {
    /// Estimand label.
    pub const LABEL: &'static str = "marginalized_initial_state";

    /// The label of this result; never `fixed_initial_state`.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        Self::LABEL
    }
}

/// The fixed-state estimand of a sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedStateQuery {
    /// Actions of the two steps.
    pub sequence: [u32; 2],
    /// The state held fixed.
    pub s0: u32,
}

impl FixedStateQuery {
    fn response(&self, units: &[&UnitHistories]) -> Result<f64, EstimationError> {
        SequenceTallies::of(units, self.sequence)
            .response(self.s0)
            .map_err(|gap| support_gap(self.s0, gap))
    }

    /// Evaluate on a panel.
    ///
    /// # Errors
    /// `initial_state.support_gap` when the state has no history support.
    pub fn effect(&self, panel: &TemporalUnitPanel) -> Result<FixedStateEffect, EstimationError> {
        let units = panel.units().iter().collect::<Vec<_>>();
        Ok(FixedStateEffect {
            value: self.response(&units)?,
            sequence: self.sequence,
            s0: self.s0,
            panel_snapshot_id: panel.snapshot_id().to_owned(),
        })
    }
}

impl TemporalEstimator for FixedStateQuery {
    fn label(&self) -> &'static str {
        FixedStateEffect::LABEL
    }

    fn estimate(&self, units: &[&UnitHistories]) -> Result<f64, EstimationError> {
        self.response(units)
    }
}

/// The target-marginal estimand of a sequence: only a target law can form it.
#[derive(Clone, Debug, PartialEq)]
pub struct MarginalizedQuery {
    sequence: [u32; 2],
    law: InitialStateLaw,
}

impl MarginalizedQuery {
    /// Bind a sequence to the target initial-state law.
    ///
    /// # Errors
    /// `transport_missing_evidence` / `initial_state.target_law_missing` for a point
    /// state or a source-population law.
    pub fn new(sequence: [u32; 2], spec: InitialStateSpec) -> Result<Self, EstimationError> {
        match spec {
            InitialStateSpec::Point(s0) => Err(target_law_missing(&format!(
                "a single state ({s0}) is not a target initial-state law; fixing it is a \
                 different estimand"
            ))),
            InitialStateSpec::Law(law) if law.population() != InitialStatePopulation::Target => {
                Err(target_law_missing(
                    "only a source initial-state law was supplied; source and target laws \
                     cannot be interchanged",
                ))
            }
            InitialStateSpec::Law(law) => Ok(Self { sequence, law }),
        }
    }

    /// The bound target law.
    #[must_use]
    pub const fn law(&self) -> &InitialStateLaw {
        &self.law
    }

    fn contributions(
        &self,
        units: &[&UnitHistories],
    ) -> Result<Vec<StateContribution>, EstimationError> {
        crate::temporal_history_fit::FittedSequenceHistory::fit(units, self.sequence)
            .contributions(&self.law)
    }

    /// Evaluate the sum over the target law on a panel.
    ///
    /// # Errors
    /// `initial_state.support_gap` when a state with positive target mass has no
    /// history support.
    pub fn effect(&self, panel: &TemporalUnitPanel) -> Result<MarginalizedEffect, EstimationError> {
        let units = panel.units().iter().collect::<Vec<_>>();
        let contributions = self.contributions(&units)?;
        Ok(MarginalizedEffect {
            value: contributions.iter().map(|c| c.mass * c.response).sum(),
            sequence: self.sequence,
            contributions,
            state_snapshot_id: self.law.snapshot_id.clone(),
            state_law_digest: self.law.digest,
            panel_snapshot_id: panel.snapshot_id().to_owned(),
            inference_claim: INITIAL_STATE_INFERENCE_CLAIM,
        })
    }
}

impl TemporalEstimator for MarginalizedQuery {
    fn label(&self) -> &'static str {
        MarginalizedEffect::LABEL
    }

    fn estimate(&self, units: &[&UnitHistories]) -> Result<f64, EstimationError> {
        Ok(self.contributions(units)?.iter().map(|c| c.mass * c.response).sum())
    }
}
