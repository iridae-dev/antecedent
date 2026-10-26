//! Compiled native-design targets for means, local cohort effects and donor contrasts.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::{JointPanelError, check_cancel, failure};
use antecedent_core::{CanonicalWriter, ExecutionContext, SemanticDigest};
use antecedent_estimate::joint_design::compiled::{
    JointDesignInference, JointDesignRecipe, JointNativeDesignLaw,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Typed source proof and exact native target semantics, shared across inference modes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeJointIdentification {
    /// Proof representation version.
    pub version: u16,
    /// Method owning the identifying functional and estimator.
    pub method: String,
    /// Executable observed-law functional retained by the native recipe.
    pub identifying_functional: String,
    /// Actual target class; structural/local effects are never renamed population means.
    pub target_kind: String,
    /// Ordered program/history/cohort/structural-feature target keys.
    pub targets: Vec<String>,
    /// Original population and averaging domain.
    pub population: String,
    /// Independent assignment/sampling unit, distinct from per-period observations.
    pub sampling_unit: String,
    /// Scientific design restrictions, retained separately from statistical likelihood priors.
    pub required_assumptions: Vec<String>,
}
/// Prepared native target retaining actual source data and the design-specific functional.
#[derive(Clone, Debug)]
pub struct PreparedNativeJointDesign {
    recipe: Arc<JointDesignRecipe>,
    identification: Arc<NativeJointIdentification>,
    dependency: SemanticDigest,
    point: Arc<JointNativeDesignLaw>,
    _lease: Arc<antecedent_core::execution::AdmissionLease>,
}
/// Executed native target with its exact statistical procedure and original design proof.
#[derive(Clone, Debug)]
pub struct NativeJointDesignResult {
    prepared: PreparedNativeJointDesign,
    inference: Arc<JointDesignInference>,
    law: Arc<JointNativeDesignLaw>,
    dependency: SemanticDigest,
    _lease: Arc<antecedent_core::execution::AdmissionLease>,
}
pub(super) fn admitted(
    ctx: &ExecutionContext,
    bytes: u64,
) -> Result<Arc<antecedent_core::execution::AdmissionLease>, JointPanelError> {
    ctx.acquire_admission(0, bytes).map(Arc::new).map_err(|error| match error {
        antecedent_core::execution::AdmissionError::Cancelled => {
            JointPanelError::Engine(super::super::engine::PanelError::Cancelled)
        }
        _ => JointPanelError::Engine(super::super::engine::PanelError::Resource(error.to_string())),
    })
}
/// Encoded size of an in-memory input; the caller's admission, quoted from this size, is the bound.
fn encoded_size<T: Serialize>(value: &T) -> Result<usize, JointPanelError> {
    Ok(antecedent_io::convert::to_cbor(value).map_err(failure)?.len())
}
pub(super) fn summary(recipe: &JointDesignRecipe) -> NativeJointIdentification {
    use JointDesignRecipe::{
        DifferenceInDifferences, GaussianLongitudinal, Instrumented, InstrumentedLocal,
        Longitudinal, Randomized, Switchback, SyntheticControl,
    };
    let (targets, population, sampling_unit, required_assumptions) = match recipe {
        Randomized { design, .. } => (
            design.worlds.clone(),
            design.population.clone(),
            design.sampling_unit.clone(),
            vec![
                design.consistency.clone(),
                design.between_unit_isolation.clone(),
                "known positive-probability assignment of every complete program".into(),
            ],
        ),
        Switchback { design, .. } => (
            design.worlds.iter().map(|(key, _)| key.clone()).collect(),
            design.population.clone(),
            "independently randomized complete sequence".into(),
            vec![
                design.no_anticipation.clone(),
                design.finite_carryover.clone(),
                design.sequence_isolation.clone(),
                format!(
                    "finite carryover {}; full schedule inclusion probabilities and first-history exclusion retained",
                    design.carryover
                ),
            ],
        ),
        Longitudinal { design, .. } => (
            design.worlds.iter().map(|(key, _)| key.clone()).collect(),
            design.sampling_population.clone(),
            "independent longitudinal episode".into(),
            vec![
                design.sequential_exchangeability.clone(),
                design.consistency.clone(),
                "positivity for every retained complete treatment/covariate history".into(),
            ],
        ),
        GaussianLongitudinal{design,..}=>(design.worlds.iter().map(|w|w.key.clone()).collect(),design.population.clone(),"independent complete continuous longitudinal episode".into(),vec![design.sequential_exchangeability.clone(),design.consistency.clone(),design.conditional_model.clone(),"fitted full-rank transformed joint assignment density; exact integration of all treatment-affected covariate histories".into()]),
        InstrumentedLocal { design, .. } => (vec![design.key.clone()], design.population.clone(), "independent Bernoulli encouragement unit".into(), vec![design.exclusion.clone(),design.monotonicity.clone(),design.consistency_and_isolation.clone(),"known positive encouragement probabilities; nonzero receipt first stage; target remains encouragement-specific compliers".into()]),
        Instrumented { design, .. } => (
            design.contrasts.iter().map(|(key, _)| key.clone()).collect(),
            design.population.clone(),
            "independent IV source unit".into(),
            vec![
                design.structural_model.clone(),
                design.moment_restrictions.clone(),
                "full instrument design and joint projected structural-feature rank".into(),
            ],
        ),
        DifferenceInDifferences { design, .. } => (
            design.cells.iter().map(|cell| cell.key.clone()).collect(),
            design.population.clone(),
            "independent complete rollout unit".into(),
            vec![
                design.no_anticipation.clone(),
                design.parallel_trends.clone(),
                design.consistency_and_isolation.clone(),
                "supported never/not-yet-treated comparison cohorts at both retained periods"
                    .into(),
            ],
        ),
        SyntheticControl { design, .. } => (
            design.worlds.iter().map(|world| world.key.clone()).collect(),
            design.population.clone(),
            "original treated unit and unaffected donor reference panel".into(),
            vec![
                design.counterfactual_model.clone(),
                design.donor_isolation.clone(),
                "simplex donor fit, preperiod error and explicit untreated donor set retained"
                    .into(),
            ],
        ),
    };
    NativeJointIdentification {
        version: 1,
        method: recipe.method().into(),
        identifying_functional: recipe.functional().into(),
        target_kind: recipe.target_kind().into(),
        targets,
        population,
        sampling_unit,
        required_assumptions,
    }
}
impl PreparedNativeJointDesign {
    /// Compile every named joint design through its actual method owner. The native
    /// target kind survives compilation; an IV local/structural effect, cohort ATT or
    /// donor-model treated effect cannot silently become an intervention-world ATE.
    /// # Errors
    /// Missing premises/support/rank, malformed native design, cancellation or admission refusal.
    pub fn compile(
        recipe: JointDesignRecipe,
        ctx: &ExecutionContext,
    ) -> Result<Self, JointPanelError> {
        check_cancel(ctx)?;
        let size = encoded_size(&recipe)?;
        let quote = u64::try_from(size)
            .ok()
            .and_then(|n| n.checked_mul(128))
            .ok_or_else(|| failure("native joint-design admission overflow"))?;
        let native_width = summary(&recipe).targets.len();
        let quote = quote
            .checked_add(
                (native_width as u64)
                    .checked_mul(native_width as u64)
                    .and_then(|v| v.checked_mul(64))
                    .ok_or_else(|| failure("native joint covariance admission overflow"))?,
            )
            .ok_or_else(|| failure("native joint admission overflow"))?;
        let lease = admitted(ctx, quote)?;
        let point = recipe
            .execute_native(&JointDesignInference::Frequentist, ctx)
            .map_err(|error| super::execution_failure(ctx, error))?;
        let identification = summary(&recipe);
        let mut writer = CanonicalWriter::new("antecedent.compiled_native_joint_design.v1");
        writer.bytes(&antecedent_io::convert::to_cbor(&recipe).map_err(failure)?);
        let dependency = SemanticDigest::from_bytes(writer.finish());
        Ok(Self {
            recipe: Arc::new(recipe),
            identification: Arc::new(identification),
            dependency,
            point: Arc::new(point),
            _lease: lease,
        })
    }
    /// Exact native target, identifying functional and retained causal restrictions.
    #[must_use]
    pub fn identification(&self) -> &NativeJointIdentification {
        &self.identification
    }
    /// Versioned proof usable by the existing compiled-panel design-attempt contract.
    #[must_use]
    pub fn panel_design_proof(&self) -> super::super::JointPanelDesignProof {
        super::super::JointPanelDesignProof::V1(super::super::JointPanelDesignTheorem::Native {
            identification: self.identification.as_ref().clone(),
            source_dependency: self.dependency,
        })
    }
    /// Original source, design, requested targets and estimator recipe.
    #[must_use]
    pub fn recipe(&self) -> &JointDesignRecipe {
        &self.recipe
    }
    /// Full source/design identity required by independent current consumption.
    #[must_use]
    pub const fn dependency(&self) -> SemanticDigest {
        self.dependency
    }
    /// Execute a compatible native statistical procedure using the same design proof.
    /// # Errors
    /// Invalid posterior/conformal assumptions, incomplete uncertainty support or resource refusal.
    pub fn execute(
        &self,
        inference: JointDesignInference,
        ctx: &ExecutionContext,
    ) -> Result<NativeJointDesignResult, JointPanelError> {
        check_cancel(ctx)?;
        let size = encoded_size(&inference)?;
        let draws = match &inference {
            JointDesignInference::Frequentist
            | JointDesignInference::SyntheticConformal { .. }
            | JointDesignInference::SwitchbackExactRandomization { .. } => 1,
            JointDesignInference::LocalComplierBayesian { draws, .. } => *draws,
            JointDesignInference::GaussianLongitudinalBootstrap { replicates, .. } => *replicates,
            JointDesignInference::GaussianLongitudinalBayesian { prior } => prior.draws,
            JointDesignInference::RandomizedGaussian { prior } => prior.draws,
            JointDesignInference::RandomizedFinitePopulationGaussian { prior } => prior.arm.draws,
            JointDesignInference::LongitudinalDirichlet { prior } => prior.draws,
            JointDesignInference::SwitchbackGaussian { prior } => prior.draws,
            JointDesignInference::InstrumentedGaussian { prior } => prior.draws,
            JointDesignInference::RolloutGaussian { prior } => prior.draws,
            JointDesignInference::SyntheticBayesian { prior } => prior.draws,
        };
        let dense = match (self.recipe.as_ref(), &inference) {
            (_, JointDesignInference::InstrumentedGaussian { prior }) => prior.parameter_mean.len(),
            (
                JointDesignRecipe::DifferenceInDifferences { design, units },
                JointDesignInference::RolloutGaussian { .. },
            ) => {
                let times = units
                    .iter()
                    .flat_map(|u| u.outcomes.iter().map(|(t, _)| t))
                    .collect::<std::collections::BTreeSet<_>>();
                units.len() + times.len() + design.cells.len()
            }
            _ => 0,
        };
        let quote = size
            .checked_mul(128)
            .and_then(|n| {
                draws
                    .checked_mul(self.identification.targets.len())
                    .and_then(|d| d.checked_mul(64))
                    .and_then(|d| n.checked_add(d))
            })
            .and_then(|n| {
                dense
                    .checked_mul(dense)
                    .and_then(|d| d.checked_mul(32))
                    .and_then(|d| n.checked_add(d))
            })
            .and_then(|n| u64::try_from(n).ok())
            .ok_or_else(|| failure("native joint-design statistical admission overflow"))?;
        let lease = admitted(ctx, quote)?;
        let law = if matches!(inference, JointDesignInference::Frequentist) {
            self.point.clone()
        } else {
            Arc::new(
                self.recipe
                    .execute_native(&inference, ctx)
                    .map_err(|error| super::execution_failure(ctx, error))?,
            )
        };
        let dependency = super::retained_law_identity(self.dependency, &inference, law.as_ref())?;
        check_cancel(ctx)?;
        Ok(NativeJointDesignResult {
            prepared: self.clone(),
            inference: Arc::new(inference),
            law,
            dependency,
            _lease: lease,
        })
    }
}
impl NativeJointDesignResult {
    /// Retained native point, joint uncertainty region or genuine posterior law.
    #[must_use]
    pub fn law(&self) -> &JointNativeDesignLaw {
        &self.law
    }
    /// Exact fitted-law slots, priors, RNG stream and numerical recipe identity.
    #[must_use]
    pub const fn dependency_identity(&self) -> SemanticDigest {
        self.dependency
    }
    /// Same exact identification result consumed by either inference mode.
    #[must_use]
    pub fn prepared(&self) -> &PreparedNativeJointDesign {
        &self.prepared
    }
    /// Freeze the actual source, proof, native target and full statistical recipe.
    /// # Errors
    /// Encoding failure or portable size limit.
    pub fn export_artifact(&self) -> Result<Vec<u8>, JointPanelError> {
        self.export_artifact_with_limits(super::DesignArtifactLimits::default())
    }
    /// [`Self::export_artifact`] under a caller-declared byte limit.
    /// # Errors
    /// As [`Self::export_artifact`], with `limits.max_bytes` as the size bound.
    pub fn export_artifact_with_limits(
        &self,
        limits: super::DesignArtifactLimits,
    ) -> Result<Vec<u8>, JointPanelError> {
        let wire = NativeReplay {
            version: 1,
            recipe: self.prepared.recipe.as_ref().clone(),
            identification: self.prepared.identification.as_ref().clone(),
            inference: self.inference.as_ref().clone(),
            law: self.law.as_ref().clone(),
        };
        let bytes = antecedent_io::convert::to_cbor(&wire).map_err(failure)?;
        if bytes.len() > limits.max_bytes {
            return Err(failure("native joint-design artifact exceeds the declared portable size"));
        }
        Ok(bytes)
    }
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeReplay {
    version: u16,
    recipe: JointDesignRecipe,
    identification: NativeJointIdentification,
    inference: JointDesignInference,
    law: JointNativeDesignLaw,
}
fn decode(
    bytes: &[u8],
    ctx: &ExecutionContext,
    limits: super::DesignArtifactLimits,
) -> Result<(NativeReplay, Arc<antecedent_core::execution::AdmissionLease>), JointPanelError> {
    if bytes.len() > limits.max_bytes {
        return Err(failure("native joint-design artifact exceeds the declared decode size"));
    }
    let lease = admitted(
        ctx,
        (bytes.len() as u64)
            .checked_mul(128)
            .ok_or_else(|| failure("native joint-design artifact admission overflow"))?,
    )?;
    let wire: NativeReplay = antecedent_io::convert::from_cbor(bytes).map_err(failure)?;
    if wire.version != 1 {
        return Err(failure("unsupported native joint-design artifact version"));
    }
    Ok((wire, lease))
}
/// Independently replay historical native design evidence without granting current-source authority.
/// # Errors
/// Malformed or changed native proof, target, statistical recipe or retained numerical evidence.
pub fn replay_native_joint_design(
    bytes: &[u8],
    ctx: &ExecutionContext,
) -> Result<JointNativeDesignLaw, JointPanelError> {
    replay_native_joint_design_with_limits(bytes, ctx, super::DesignArtifactLimits::default())
}

/// [`replay_native_joint_design`] under a caller-declared byte limit.
/// # Errors
/// As [`replay_native_joint_design`], with `limits.max_bytes` as the size bound.
pub fn replay_native_joint_design_with_limits(
    bytes: &[u8],
    ctx: &ExecutionContext,
    limits: super::DesignArtifactLimits,
) -> Result<JointNativeDesignLaw, JointPanelError> {
    replay_native_joint_design_execution(bytes, ctx, limits)
        .map(|result| result.law.as_ref().clone())
}

pub(crate) fn replay_native_joint_design_execution(
    bytes: &[u8],
    ctx: &ExecutionContext,
    limits: super::DesignArtifactLimits,
) -> Result<NativeJointDesignResult, JointPanelError> {
    let (wire, _lease) = decode(bytes, ctx, limits)?;
    let prepared = PreparedNativeJointDesign::compile(wire.recipe, ctx)?;
    if prepared.identification.as_ref() != &wire.identification {
        return Err(failure("native joint-design artifact changed the scientific target or proof"));
    }
    let result = prepared.execute(wire.inference, ctx)?;
    if result.law.as_ref() != &wire.law {
        return Err(failure("native joint-design artifact does not reproduce retained evidence"));
    }
    Ok(result)
}
impl PreparedNativeJointDesign {
    /// Consume an artifact only after matching independently retained current source and design.
    /// # Errors
    /// Different source/design/target, invalid proof or nonreproducible law.
    pub fn consume_current(
        &self,
        bytes: &[u8],
        ctx: &ExecutionContext,
    ) -> Result<NativeJointDesignResult, JointPanelError> {
        self.consume_current_with_limits(bytes, ctx, super::DesignArtifactLimits::default())
    }
    /// [`Self::consume_current`] under a caller-declared byte limit.
    /// # Errors
    /// As [`Self::consume_current`], with `limits.max_bytes` as the size bound.
    pub fn consume_current_with_limits(
        &self,
        bytes: &[u8],
        ctx: &ExecutionContext,
        limits: super::DesignArtifactLimits,
    ) -> Result<NativeJointDesignResult, JointPanelError> {
        let (wire, _lease) = decode(bytes, ctx, limits)?;
        if &wire.recipe != self.recipe.as_ref()
            || &wire.identification != self.identification.as_ref()
        {
            return Err(failure(
                "native joint-design artifact differs from caller-current source, target or identifying premises",
            ));
        }
        let result = self.execute(wire.inference, ctx)?;
        if result.law.as_ref() != &wire.law {
            return Err(failure("native joint-design artifact contains nonreproducible evidence"));
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
pub(crate) use tests::{gaussian_longitudinal_chain_fixture, native_experiment_fixtures};
