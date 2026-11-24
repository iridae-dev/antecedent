//! Vocabulary of the operations a prepared study answers on its own fitted structural model: the
//! model-implied strength of every edge, the population do-contrast of one parent on one child, a
//! named do-sampler, the model's held-out falsification evaluation and a nested counterfactual.
//!
//! [`ModelOperation`] is the declaration a prepared study runs (`PreparedStudy::model_operation`
//! in the facade). It is not a [`crate::CausalQuery`]: the operation reads the mechanisms the
//! study fits on its own graph and table, so the study's query, contract and identity are not
//! changed by it. Each target names the operation and every control of its engine; it carries no
//! result. The operations are statements about the fitted model, not identified estimands of the
//! population, and run with calibration status `not_measured`.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use crate::canonical_encode::{CanonicalEncode, count, encode_list};
use crate::identity::CanonicalWriter;
use crate::ids::VariableId;
use crate::intervention::Intervention;

/// Why a model-operation declaration is refused. The display text of a variant is the stable
/// statement of the contract.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ModelOperationFault {
    /// Refused: the contrast width must be finite and nonzero
    #[error("the contrast width must be finite and nonzero")]
    ContrastWidth,
    /// Refused: the parent and the child of a do-contrast must differ
    #[error("the parent and the child of a do-contrast must differ")]
    ContrastSameVariable,
    /// Refused: a do-sampler needs at least one draw
    #[error("a do-sampler needs at least one draw")]
    SamplerDraws,
    /// Refused: the weighting do-sampler targets exactly one intervention
    #[error("the weighting do-sampler targets exactly one hard intervention")]
    WeightingInterventions,
    /// Refused: a sampler control must be finite, and positive where it is a scale
    #[error("a do-sampler control must be finite, and positive where it is a scale")]
    SamplerControl,
    /// Refused: the evaluation level must be in (0, 1)
    #[error("the evaluation level must be in (0, 1)")]
    EvaluationLevel,
    /// Refused: the evaluation needs at least one permutation
    #[error("the evaluation needs at least one permutation")]
    EvaluationPermutations,
    /// Refused: a nested counterfactual's frozen set may not hold the outcome or an inner target
    #[error(
        "a nested counterfactual's frozen set may not hold the outcome or an inner intervention \
         target"
    )]
    NestedFrozenConflict,
    /// Refused: an intervention of a model operation is invalid
    #[error("an intervention of a model operation is invalid")]
    InvalidIntervention,
}

/// Model-implied strength of every edge of the fitted structure: the cut-edge divergence and the
/// variance contribution of each, defined for linear-family mechanisms.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct ArrowStrengthsTarget;

impl ArrowStrengthsTarget {
    /// Validate the target (a declaration with no controls is always valid).
    ///
    /// # Errors
    ///
    /// Never; the signature matches the other targets.
    pub const fn validate(&self) -> Result<(), ModelOperationFault> {
        Ok(())
    }
}

impl CanonicalEncode for ArrowStrengthsTarget {
    fn encode(&self, writer: &mut CanonicalWriter) {
        writer.word(1);
    }
}

/// Population do-contrast of `parent` on `child`:
/// `E[child | do(parent = mean + delta/2)] - E[child | do(parent = mean - delta/2)]`, with common
/// random numbers across the two arms.
///
/// The two arms draw from the attribution stream of `seed`'s own generator, so the draws are a
/// function of the declaration and not of the execution context the study runs under; a direct
/// call under a context seeded with `seed` draws the same numbers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PopulationDoContrastTarget {
    /// The intervened variable.
    pub parent: VariableId,
    /// The variable read.
    pub child: VariableId,
    /// Width of the contrast, finite and nonzero.
    pub delta: f64,
    /// Most observed rows accepted (`None` accepts every row); a larger table is refused.
    pub max_units: Option<usize>,
    /// Seed of the generator whose attribution stream both arms draw from.
    pub seed: u64,
}

impl PopulationDoContrastTarget {
    /// Validate the target.
    ///
    /// # Errors
    ///
    /// [`ModelOperationFault::ContrastWidth`] for a zero or non-finite width and
    /// [`ModelOperationFault::ContrastSameVariable`] when parent and child coincide.
    pub fn validate(&self) -> Result<(), ModelOperationFault> {
        if !self.delta.is_finite() || self.delta == 0.0 {
            return Err(ModelOperationFault::ContrastWidth);
        }
        if self.parent == self.child {
            return Err(ModelOperationFault::ContrastSameVariable);
        }
        Ok(())
    }
}

impl CanonicalEncode for PopulationDoContrastTarget {
    fn encode(&self, writer: &mut CanonicalWriter) {
        writer.word(2);
        self.parent.encode(writer);
        self.child.encode(writer);
        writer.real(self.delta);
        encode_optional_count(writer, self.max_units);
        writer.word(self.seed);
    }
}

/// The named do-sampler and its controls.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DoSamplerMethod {
    /// Reweights the observed rows toward `do(T = t)`; one hard intervention, `draws` is ignored.
    Weighting {
        /// Half-width of the window of observed rows matched to a root treatment's do-value
        /// (`None` is the engine's default).
        root_match_tolerance: Option<f64>,
    },
    /// Ancestral draws smoothed by a Gaussian kernel.
    Kde {
        /// Bandwidth (`None` is Silverman's rule).
        bandwidth: Option<f64>,
    },
    /// Random-walk Metropolis-Hastings against the kernel density of pilot draws.
    Mcmc {
        /// Proposal standard deviation (`None` is the engine's default).
        proposal_sd: Option<f64>,
        /// Burn-in iterations (`None` is the engine's default).
        burn_in: Option<usize>,
        /// Thinning (`None` is the engine's default).
        thin: Option<usize>,
    },
}

/// A do-sampler run for one outcome under hard interventions.
#[derive(Clone, Debug, PartialEq)]
pub struct DoSamplerTarget {
    /// Sampler and its controls.
    pub method: DoSamplerMethod,
    /// Outcome variable.
    pub outcome: VariableId,
    /// Hard interventions, in declared order.
    pub interventions: Arc<[Intervention]>,
    /// Draws requested (at least 1; ignored by the weighting sampler).
    pub draws: usize,
    /// Seed of the sampling stream.
    pub seed: u64,
}

impl DoSamplerTarget {
    /// Validate the target.
    ///
    /// # Errors
    ///
    /// A fault naming the violated sampler contract.
    pub fn validate(&self) -> Result<(), ModelOperationFault> {
        match self.method {
            DoSamplerMethod::Weighting { root_match_tolerance } => {
                if self.interventions.len() != 1 {
                    return Err(ModelOperationFault::WeightingInterventions);
                }
                if root_match_tolerance.is_some_and(|t| !t.is_finite() || t < 0.0) {
                    return Err(ModelOperationFault::SamplerControl);
                }
            }
            DoSamplerMethod::Kde { bandwidth } => {
                if self.draws == 0 {
                    return Err(ModelOperationFault::SamplerDraws);
                }
                if bandwidth.is_some_and(|b| !b.is_finite() || b <= 0.0) {
                    return Err(ModelOperationFault::SamplerControl);
                }
            }
            DoSamplerMethod::Mcmc { proposal_sd, .. } => {
                if self.draws == 0 {
                    return Err(ModelOperationFault::SamplerDraws);
                }
                if proposal_sd.is_some_and(|s| !s.is_finite() || s <= 0.0) {
                    return Err(ModelOperationFault::SamplerControl);
                }
            }
        }
        Ok(())
    }
}

impl CanonicalEncode for DoSamplerTarget {
    fn encode(&self, writer: &mut CanonicalWriter) {
        writer.word(3);
        match self.method {
            DoSamplerMethod::Weighting { root_match_tolerance } => {
                writer.word(1);
                encode_optional_real(writer, root_match_tolerance);
            }
            DoSamplerMethod::Kde { bandwidth } => {
                writer.word(2);
                encode_optional_real(writer, bandwidth);
            }
            DoSamplerMethod::Mcmc { proposal_sd, burn_in, thin } => {
                writer.word(3);
                encode_optional_real(writer, proposal_sd);
                encode_optional_count(writer, burn_in);
                encode_optional_count(writer, thin);
            }
        }
        self.outcome.encode(writer);
        encode_list(writer, &self.interventions);
        writer.word(count(self.draws)).word(self.seed);
    }
}

/// The fitted model's falsification evaluation: in-sample log-likelihood, residual independence,
/// local Markov checks and a permutation baseline.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModelEvaluationTarget {
    /// Significance level of the independence tests, in `(0, 1)`.
    pub alpha: f64,
    /// Permutation replicates of the baseline (at least 1).
    pub n_permutations: usize,
    /// Seed of the permutations.
    pub seed: u64,
}

impl ModelEvaluationTarget {
    /// Validate the target.
    ///
    /// # Errors
    ///
    /// [`ModelOperationFault::EvaluationLevel`] or [`ModelOperationFault::EvaluationPermutations`].
    pub fn validate(&self) -> Result<(), ModelOperationFault> {
        if !(self.alpha.is_finite() && self.alpha > 0.0 && self.alpha < 1.0) {
            return Err(ModelOperationFault::EvaluationLevel);
        }
        if self.n_permutations == 0 {
            return Err(ModelOperationFault::EvaluationPermutations);
        }
        Ok(())
    }
}

impl CanonicalEncode for ModelEvaluationTarget {
    fn encode(&self, writer: &mut CanonicalWriter) {
        writer.word(4).real(self.alpha).word(count(self.n_permutations)).word(self.seed);
    }
}

/// The nested counterfactual `Y_{inner, F_{outer}}`: the mean over the study's rows of `outcome`
/// under the `inner` interventions while every variable in `frozen` keeps the value it takes in
/// the `outer` world, on the noise abduced from each row. For the mediation contrast
/// `Y_{x, M_{x'}}` take `outer = do(X = x')`, `inner = do(X = x)` and `frozen = {M}`.
#[derive(Clone, Debug, PartialEq)]
pub struct NestedCounterfactualTarget {
    /// Interventions of the outer world, in declared order.
    pub outer: Arc<[Intervention]>,
    /// Interventions of the inner world, in declared order.
    pub inner: Arc<[Intervention]>,
    /// Variables held at their outer-world values in the inner world.
    pub frozen: Arc<[VariableId]>,
    /// The variable whose mean is read.
    pub outcome: VariableId,
}

impl NestedCounterfactualTarget {
    /// Validate the target.
    ///
    /// # Errors
    ///
    /// [`ModelOperationFault::InvalidIntervention`] for an invalid intervention and
    /// [`ModelOperationFault::NestedFrozenConflict`] when the frozen set holds the outcome or an
    /// inner intervention target.
    pub fn validate(&self) -> Result<(), ModelOperationFault> {
        if self.outer.iter().chain(self.inner.iter()).any(|iv| iv.validate().is_err()) {
            return Err(ModelOperationFault::InvalidIntervention);
        }
        let inner_targets: Vec<VariableId> =
            self.inner.iter().filter_map(Intervention::primary_variable).collect();
        if self.frozen.iter().any(|v| *v == self.outcome || inner_targets.contains(v)) {
            return Err(ModelOperationFault::NestedFrozenConflict);
        }
        Ok(())
    }
}

impl CanonicalEncode for NestedCounterfactualTarget {
    fn encode(&self, writer: &mut CanonicalWriter) {
        writer.word(5);
        encode_list(writer, &self.outer);
        encode_list(writer, &self.inner);
        encode_list(writer, &self.frozen);
        self.outcome.encode(writer);
    }
}

/// An operation a prepared study runs on the mechanisms it fits on its own graph and table.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum ModelOperation {
    /// Model-implied strength of every edge (linear-family mechanisms).
    ArrowStrengths(ArrowStrengthsTarget),
    /// Population do-contrast of one parent on one child.
    PopulationDoContrast(PopulationDoContrastTarget),
    /// A named do-sampler for one outcome.
    DoSample(DoSamplerTarget),
    /// The fitted model's falsification evaluation on the study's rows.
    ModelEvaluation(ModelEvaluationTarget),
    /// A nested counterfactual mean over the study's rows.
    NestedCounterfactual(NestedCounterfactualTarget),
}

impl ModelOperation {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::ArrowStrengths(_) => "arrow_strengths",
            Self::PopulationDoContrast(_) => "population_do_contrast",
            Self::DoSample(_) => "do_sample",
            Self::ModelEvaluation(_) => "model_evaluation",
            Self::NestedCounterfactual(_) => "nested_counterfactual",
        }
    }

    /// Whether the operation is an attribution of the fitted model (edge strengths, do-contrast)
    /// rather than a simulation of it (sampler, evaluation, nested counterfactual).
    #[must_use]
    pub const fn is_attribution(&self) -> bool {
        matches!(self, Self::ArrowStrengths(_) | Self::PopulationDoContrast(_))
    }

    /// The variables the operation names.
    #[must_use]
    pub fn variables(&self) -> Vec<VariableId> {
        let mut out = Vec::new();
        match self {
            Self::ArrowStrengths(_) | Self::ModelEvaluation(_) => {}
            Self::PopulationDoContrast(target) => out.extend([target.parent, target.child]),
            Self::DoSample(target) => {
                out.extend(target.interventions.iter().filter_map(Intervention::primary_variable));
                out.push(target.outcome);
            }
            Self::NestedCounterfactual(target) => {
                out.extend(
                    target
                        .outer
                        .iter()
                        .chain(target.inner.iter())
                        .filter_map(Intervention::primary_variable),
                );
                out.extend(target.frozen.iter().copied());
                out.push(target.outcome);
            }
        }
        out.sort_unstable_by_key(|v| v.raw());
        out.dedup();
        out
    }

    /// Validate the operation.
    ///
    /// # Errors
    ///
    /// The fault of the operation's target.
    pub fn validate(&self) -> Result<(), ModelOperationFault> {
        match self {
            Self::ArrowStrengths(target) => target.validate(),
            Self::PopulationDoContrast(target) => target.validate(),
            Self::DoSample(target) => {
                if target.interventions.iter().any(|iv| iv.validate().is_err()) {
                    return Err(ModelOperationFault::InvalidIntervention);
                }
                target.validate()
            }
            Self::ModelEvaluation(target) => target.validate(),
            Self::NestedCounterfactual(target) => target.validate(),
        }
    }
}

impl CanonicalEncode for ModelOperation {
    fn encode(&self, writer: &mut CanonicalWriter) {
        // Each target writes its own leading tag (1 to 5), so the operation needs no other.
        match self {
            Self::ArrowStrengths(target) => target.encode(writer),
            Self::PopulationDoContrast(target) => target.encode(writer),
            Self::DoSample(target) => target.encode(writer),
            Self::ModelEvaluation(target) => target.encode(writer),
            Self::NestedCounterfactual(target) => target.encode(writer),
        }
    }
}

fn encode_optional_real(writer: &mut CanonicalWriter, value: Option<f64>) {
    match value {
        None => {
            writer.word(0);
        }
        Some(value) => {
            writer.word(1).real(value);
        }
    }
}

fn encode_optional_count(writer: &mut CanonicalWriter, value: Option<usize>) {
    match value {
        None => {
            writer.word(0);
        }
        Some(value) => {
            writer.word(1).word(count(value));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn variable(raw: u32) -> VariableId {
        VariableId::from_raw(raw)
    }

    #[test]
    fn contrast_refuses_a_null_width_and_a_self_contrast() {
        let ok = PopulationDoContrastTarget {
            parent: variable(0),
            child: variable(1),
            delta: 1.0,
            max_units: None,
            seed: 0,
        };
        assert!(ok.validate().is_ok());
        let null = PopulationDoContrastTarget { delta: 0.0, ..ok };
        assert_eq!(null.validate(), Err(ModelOperationFault::ContrastWidth));
        let same = PopulationDoContrastTarget { child: variable(0), ..ok };
        assert_eq!(same.validate(), Err(ModelOperationFault::ContrastSameVariable));
    }

    #[test]
    fn weighting_sampler_needs_exactly_one_intervention() {
        let target = DoSamplerTarget {
            method: DoSamplerMethod::Weighting { root_match_tolerance: None },
            outcome: variable(1),
            interventions: Arc::from([]),
            draws: 0,
            seed: 1,
        };
        assert_eq!(target.validate(), Err(ModelOperationFault::WeightingInterventions));
    }

    #[test]
    fn a_nested_counterfactual_refuses_a_frozen_outcome_or_inner_target() {
        use crate::value::Value;
        let target = NestedCounterfactualTarget {
            outer: Arc::from([Intervention::set(variable(0), Value::f64(0.0))]),
            inner: Arc::from([Intervention::set(variable(0), Value::f64(1.0))]),
            frozen: Arc::from([variable(1)]),
            outcome: variable(2),
        };
        assert!(target.validate().is_ok());
        let frozen_outcome =
            NestedCounterfactualTarget { frozen: Arc::from([variable(2)]), ..target.clone() };
        assert_eq!(frozen_outcome.validate(), Err(ModelOperationFault::NestedFrozenConflict));
        let frozen_inner =
            NestedCounterfactualTarget { frozen: Arc::from([variable(0)]), ..target };
        assert_eq!(frozen_inner.validate(), Err(ModelOperationFault::NestedFrozenConflict));
    }

    #[test]
    fn evaluation_refuses_a_level_outside_the_open_interval() {
        let target = ModelEvaluationTarget { alpha: 1.0, n_permutations: 5, seed: 0 };
        assert_eq!(target.validate(), Err(ModelOperationFault::EvaluationLevel));
    }
}
