//! The answer of a counterfactual that contrasts several outcomes or simultaneous interventions.
//!
//! One outcome under one hard intervention answers with an [`IteResult`]. A counterfactual with
//! several outcomes, a joint baseline of simultaneous interventions or nested interventions
//! answers with one contrast per outcome, all read from the same two replays of one abduction.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::VariableId;

use super::{ExogenousPosterior, IteResult, NoiseInferenceKind, UnitEffectIntervals};

/// The unit-level contrast of one outcome, with the uncertainty and support the execution
/// reported for it.
#[derive(Clone, Debug)]
pub struct ContrastedOutcome {
    /// The outcome the contrast reads.
    pub outcome: VariableId,
    /// Per-unit effects, aligned with the factual rows.
    pub unit_effects: Arc<[f64]>,
    /// Mean of the unit effects (the posterior mean under Bayesian inference).
    pub mean_ite: f64,
    /// Standard error of [`Self::mean_ite`]: the posterior standard deviation or the bootstrap
    /// standard error. `None` for a point result, which has no sampling uncertainty.
    pub standard_error: Option<f64>,
    /// Per-unit posterior intervals, formed under Bayesian inference only.
    pub unit_effect_intervals: Option<UnitEffectIntervals>,
    /// Per-unit flags: the unit's prediction leaves the observed support of the worlds.
    pub unit_extrapolative: Option<Arc<[bool]>>,
    /// Whether every unit carries the same effect by construction of the fitted mechanisms on
    /// the paths from the intervened variables to this outcome.
    pub unit_effects_homogeneous: bool,
}

/// Contrasts of several outcomes between two worlds that share one abduced noise state.
#[derive(Clone, Debug)]
pub struct CounterfactualContrasts {
    /// The variables both worlds intervene on, ordered.
    pub treatments: Arc<[VariableId]>,
    /// One contrast per query outcome, in the query's order. Never empty.
    pub contrasts: Vec<ContrastedOutcome>,
    /// How noise was obtained.
    pub noise_inference: NoiseInferenceKind,
    /// Shared exogenous state.
    pub exogenous: ExogenousPosterior,
}

impl CounterfactualContrasts {
    /// The first outcome's contrast: the one the result's scalar estimate describes.
    ///
    /// # Panics
    ///
    /// Never for a value an execution produced: it holds at least one contrast.
    #[must_use]
    pub fn primary(&self) -> &ContrastedOutcome {
        &self.contrasts[0]
    }

    /// The contrast of `outcome`, when the query listed it.
    #[must_use]
    pub fn contrast(&self, outcome: VariableId) -> Option<&ContrastedOutcome> {
        self.contrasts.iter().find(|contrast| contrast.outcome == outcome)
    }

    /// The contrast of `outcome` as a single-contrast [`IteResult`], when the query listed it.
    #[must_use]
    pub fn ite_result(&self, outcome: VariableId) -> Option<IteResult> {
        self.contrast(outcome).map(|contrast| IteResult {
            unit_effects: Arc::clone(&contrast.unit_effects),
            mean_ite: contrast.mean_ite,
            noise_inference: self.noise_inference,
            exogenous: self.exogenous.clone(),
            unit_effect_intervals: contrast.unit_effect_intervals.clone(),
            unit_extrapolative: contrast.unit_extrapolative.clone(),
        })
    }
}
