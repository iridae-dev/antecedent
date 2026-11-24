//! Wire form of a fitted-model operation ([`antecedent_core::ModelOperation`]): the declaration a
//! prepared study runs on its own fitted model. A new, additive type: no query wire changes, and
//! the bytes of every existing query are untouched.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    ArrowStrengthsTarget, DoSamplerMethod, DoSamplerTarget, Intervention, ModelEvaluationTarget,
    ModelOperation, NestedCounterfactualTarget, PopulationDoContrastTarget, VariableId,
};
use serde::{Deserialize, Serialize};

use crate::convert::{vars_from_raw, vars_to_raw};
use crate::error::{IoError, fit_u64};
use crate::query_wire::InterventionWire;

fn fit_usize(value: u64) -> Result<usize, IoError> {
    usize::try_from(value).map_err(|_| IoError::TooLarge)
}

fn interventions_to_wire(interventions: &[Intervention]) -> Result<Vec<InterventionWire>, IoError> {
    interventions.iter().map(InterventionWire::from_domain_recoverable).collect()
}

fn interventions_from_wire(interventions: &[InterventionWire]) -> Arc<[Intervention]> {
    interventions.iter().map(InterventionWire::to_domain).collect::<Vec<_>>().into()
}

/// Wire form of [`ModelOperation`].
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ModelOperationWire {
    /// Model-implied strength of every edge.
    ArrowStrengths {},
    /// Population do-contrast of one parent on one child.
    PopulationDoContrast {
        /// Intervened variable.
        parent: u32,
        /// Variable read.
        child: u32,
        /// Width of the contrast.
        delta: f64,
        /// Most observed rows accepted; absent accepts every row.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max_units: Option<u64>,
        /// Seed of the generator both arms draw from.
        seed: u64,
    },
    /// A named do-sampler for one outcome.
    DoSample {
        /// `weighting`, `kde` or `mcmc`.
        method: String,
        /// Weighting: half-width of the root-treatment match window.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        root_match_tolerance: Option<f64>,
        /// Kernel density: bandwidth.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        bandwidth: Option<f64>,
        /// Metropolis-Hastings: proposal standard deviation.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        proposal_sd: Option<f64>,
        /// Metropolis-Hastings: burn-in iterations.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        burn_in: Option<u64>,
        /// Metropolis-Hastings: thinning.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        thin: Option<u64>,
        /// Outcome.
        outcome: u32,
        /// Hard interventions.
        interventions: Vec<InterventionWire>,
        /// Draws requested.
        draws: u64,
        /// Seed.
        seed: u64,
    },
    /// The fitted model's falsification evaluation.
    ModelEvaluation {
        /// Significance level.
        alpha: f64,
        /// Permutation replicates.
        n_permutations: u64,
        /// Seed.
        seed: u64,
    },
    /// A nested counterfactual mean.
    NestedCounterfactual {
        /// Outer-world interventions.
        outer: Vec<InterventionWire>,
        /// Inner-world interventions.
        inner: Vec<InterventionWire>,
        /// Variables held at their outer-world values.
        frozen: Vec<u32>,
        /// Outcome.
        outcome: u32,
    },
}

impl ModelOperationWire {
    /// Encode.
    ///
    /// # Errors
    ///
    /// An intervention holding a runtime-bound callback, a count that does not fit, or an
    /// operation this version does not know.
    pub fn from_domain(operation: &ModelOperation) -> Result<Self, IoError> {
        Ok(match operation {
            ModelOperation::ArrowStrengths(_) => Self::ArrowStrengths {},
            ModelOperation::PopulationDoContrast(target) => Self::PopulationDoContrast {
                parent: target.parent.raw(),
                child: target.child.raw(),
                delta: target.delta,
                max_units: target.max_units.map(fit_u64).transpose()?,
                seed: target.seed,
            },
            ModelOperation::DoSample(target) => {
                let (method, root_match_tolerance, bandwidth, proposal_sd, burn_in, thin) =
                    match target.method {
                        DoSamplerMethod::Weighting { root_match_tolerance } => {
                            ("weighting", root_match_tolerance, None, None, None, None)
                        }
                        DoSamplerMethod::Kde { bandwidth } => {
                            ("kde", None, bandwidth, None, None, None)
                        }
                        DoSamplerMethod::Mcmc { proposal_sd, burn_in, thin } => (
                            "mcmc",
                            None,
                            None,
                            proposal_sd,
                            burn_in.map(fit_u64).transpose()?,
                            thin.map(fit_u64).transpose()?,
                        ),
                    };
                Self::DoSample {
                    method: method.to_owned(),
                    root_match_tolerance,
                    bandwidth,
                    proposal_sd,
                    burn_in,
                    thin,
                    outcome: target.outcome.raw(),
                    interventions: interventions_to_wire(&target.interventions)?,
                    draws: fit_u64(target.draws)?,
                    seed: target.seed,
                }
            }
            ModelOperation::ModelEvaluation(target) => Self::ModelEvaluation {
                alpha: target.alpha,
                n_permutations: fit_u64(target.n_permutations)?,
                seed: target.seed,
            },
            ModelOperation::NestedCounterfactual(target) => Self::NestedCounterfactual {
                outer: interventions_to_wire(&target.outer)?,
                inner: interventions_to_wire(&target.inner)?,
                frozen: vars_to_raw(&target.frozen),
                outcome: target.outcome.raw(),
            },
            other => {
                return Err(IoError::Convert(format!(
                    "model operation `{}` has no wire form in this version",
                    other.as_str()
                )));
            }
        })
    }

    /// Decode and validate.
    ///
    /// # Errors
    ///
    /// An unknown sampler, a control the named sampler does not read, a count that does not
    /// fit, or an operation that fails validation.
    pub fn to_domain(&self) -> Result<ModelOperation, IoError> {
        let operation = match self {
            Self::ArrowStrengths {} => ModelOperation::ArrowStrengths(ArrowStrengthsTarget),
            Self::PopulationDoContrast { parent, child, delta, max_units, seed } => {
                ModelOperation::PopulationDoContrast(PopulationDoContrastTarget {
                    parent: VariableId::from_raw(*parent),
                    child: VariableId::from_raw(*child),
                    delta: *delta,
                    max_units: max_units.map(fit_usize).transpose()?,
                    seed: *seed,
                })
            }
            Self::DoSample {
                method,
                root_match_tolerance,
                bandwidth,
                proposal_sd,
                burn_in,
                thin,
                outcome,
                interventions,
                draws,
                seed,
            } => {
                // A control the named sampler does not read is refused, never dropped.
                let foreign = |given: bool, name: &str| {
                    if given {
                        Err(IoError::Convert(format!(
                            "{name} does not apply to the {method} do-sampler"
                        )))
                    } else {
                        Ok(())
                    }
                };
                let mcmc_controls = proposal_sd.is_some() || burn_in.is_some() || thin.is_some();
                let sampler = match method.as_str() {
                    "weighting" => {
                        foreign(bandwidth.is_some(), "bandwidth")?;
                        foreign(mcmc_controls, "a Metropolis-Hastings control")?;
                        DoSamplerMethod::Weighting { root_match_tolerance: *root_match_tolerance }
                    }
                    "kde" => {
                        foreign(root_match_tolerance.is_some(), "root_match_tolerance")?;
                        foreign(mcmc_controls, "a Metropolis-Hastings control")?;
                        DoSamplerMethod::Kde { bandwidth: *bandwidth }
                    }
                    "mcmc" => {
                        foreign(root_match_tolerance.is_some(), "root_match_tolerance")?;
                        foreign(bandwidth.is_some(), "bandwidth")?;
                        DoSamplerMethod::Mcmc {
                            proposal_sd: *proposal_sd,
                            burn_in: burn_in.map(fit_usize).transpose()?,
                            thin: thin.map(fit_usize).transpose()?,
                        }
                    }
                    other => {
                        return Err(IoError::Convert(format!("unknown do-sampler `{other}`")));
                    }
                };
                ModelOperation::DoSample(DoSamplerTarget {
                    method: sampler,
                    outcome: VariableId::from_raw(*outcome),
                    interventions: interventions_from_wire(interventions),
                    draws: fit_usize(*draws)?,
                    seed: *seed,
                })
            }
            Self::ModelEvaluation { alpha, n_permutations, seed } => {
                ModelOperation::ModelEvaluation(ModelEvaluationTarget {
                    alpha: *alpha,
                    n_permutations: fit_usize(*n_permutations)?,
                    seed: *seed,
                })
            }
            Self::NestedCounterfactual { outer, inner, frozen, outcome } => {
                ModelOperation::NestedCounterfactual(NestedCounterfactualTarget {
                    outer: interventions_from_wire(outer),
                    inner: interventions_from_wire(inner),
                    frozen: vars_from_raw(frozen),
                    outcome: VariableId::from_raw(*outcome),
                })
            }
        };
        operation.validate().map_err(|fault| IoError::Convert(fault.to_string()))?;
        Ok(operation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::Value;

    fn round_trip(operation: &ModelOperation) {
        let wire = ModelOperationWire::from_domain(operation).unwrap();
        assert_eq!(&wire.to_domain().unwrap(), operation);
    }

    #[test]
    fn every_model_operation_round_trips() {
        let v = VariableId::from_raw;
        round_trip(&ModelOperation::ArrowStrengths(ArrowStrengthsTarget));
        round_trip(&ModelOperation::PopulationDoContrast(PopulationDoContrastTarget {
            parent: v(0),
            child: v(1),
            delta: 0.5,
            max_units: Some(10),
            seed: 3,
        }));
        round_trip(&ModelOperation::DoSample(DoSamplerTarget {
            method: DoSamplerMethod::Mcmc { proposal_sd: Some(0.2), burn_in: Some(5), thin: None },
            outcome: v(1),
            interventions: Arc::from([Intervention::set(v(0), Value::f64(1.0))]),
            draws: 7,
            seed: 9,
        }));
        round_trip(&ModelOperation::ModelEvaluation(ModelEvaluationTarget {
            alpha: 0.05,
            n_permutations: 3,
            seed: 1,
        }));
        round_trip(&ModelOperation::NestedCounterfactual(NestedCounterfactualTarget {
            outer: Arc::from([Intervention::set(v(0), Value::f64(0.0))]),
            inner: Arc::from([Intervention::set(v(0), Value::f64(1.0))]),
            frozen: Arc::from([v(1)]),
            outcome: v(2),
        }));
    }
}
