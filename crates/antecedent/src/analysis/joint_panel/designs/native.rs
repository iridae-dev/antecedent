//! Compiled native-design targets for means, local cohort effects and donor contrasts.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::{JointPanelError, check_cancel, failure};
use crate::analysis::joint_panel::compiled::{Entry, require_release_support};
use crate::analysis::lowering::joint_panel::design::{self as design_lowering, DesignInput};
use crate::planner::lowered::{JointPanelRoute, LoweredProgram};
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
    /// The program every execution of this design enters.
    program: Arc<LoweredProgram>,
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
fn encoded_size<T: Serialize>(value: &T, ctx: &ExecutionContext) -> Result<usize, JointPanelError> {
    struct Counter<'a> {
        bytes: usize,
        ctx: &'a ExecutionContext,
    }
    impl std::io::Write for Counter<'_> {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if self.ctx.cancellation.is_cancelled() {
                return Err(std::io::Error::other("cancelled"));
            }
            self.bytes = self
                .bytes
                .checked_add(bytes.len())
                .ok_or_else(|| std::io::Error::other("encoded size overflow"))?;
            if self.ctx.memory_limit_bytes().is_some_and(|limit| self.bytes as u64 > limit) {
                return Err(std::io::Error::other("encoded source exceeds memory limit"));
            }
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut counter = Counter { bytes: 0, ctx };
    ciborium::ser::into_writer(value, &mut counter).map_err(|error| {
        if ctx.cancellation.is_cancelled() {
            JointPanelError::Engine(super::super::engine::PanelError::Cancelled)
        } else {
            JointPanelError::Engine(super::super::engine::PanelError::Resource(error.to_string()))
        }
    })?;
    Ok(counter.bytes)
}
#[allow(clippy::too_many_lines, reason = "the summary covers each native design recipe")]
pub(super) fn summary(recipe: &JointDesignRecipe) -> NativeJointIdentification {
    use JointDesignRecipe::{
        AdaptiveRegime, DecisionCluster, DecisionExperiment, DifferenceInDifferences, Excursion,
        GaussianLongitudinal, Instrumented, InstrumentedLocal, Longitudinal, Randomized,
        StaticRegime, Switchback, SyntheticControl,
    };
    let (targets, population, sampling_unit, required_assumptions) = match recipe {
        AdaptiveRegime { recipe } => (
            recipe.outcomes.iter().map(|(_, variable, units)| format!("{variable} [{units}]")).collect(),
            recipe.population.clone(), "independent complete longitudinal subject".into(),
            vec![recipe.policy_prespecification.clone(), recipe.positivity.clone(), recipe.exchangeability.clone(), recipe.consistency.clone(), recipe.followup.clone(), recipe.isolation.clone(), "observed support for every reachable policy history".into()],
        ),
        DecisionCluster { recipe } => {
            use antecedent_estimate::joint_design::experiment::ExperimentExposure;
            let exposure = match &recipe.exposure {
                ExperimentExposure::NoInterference { assumption } | ExperimentExposure::ContrastBounds { assumption, .. }
                | ExperimentExposure::ArmBounds { assumption, .. } => assumption.clone(),
            };
            let mut assumptions = vec![recipe.assignment_law.clone(), recipe.consistency.clone(), exposure];
            assumptions.extend(recipe.adjustments.iter().map(|a| a.preassignment.clone()));
            (recipe.outcomes.iter().map(|o| format!("{} [{}]", o.variable_id, o.units)).collect(),
                recipe.population.clone(), "independent randomized cluster".into(), assumptions)
        },
        DecisionExperiment { recipe } => {
            use antecedent_estimate::joint_design::experiment::{ExperimentalDesign, ExperimentExposure};
            let (unit, premise) = match &recipe.design {
                ExperimentalDesign::MatchedPairs { preassignment, independent_pairs, .. } =>
                    ("independent randomized matched pair", vec![preassignment.clone(), independent_pairs.clone()]),
                ExperimentalDesign::Switchback { no_carryover, washout_rule, independent_sequences, .. } =>
                    ("independent randomized switchback sequence", vec![no_carryover.clone(), washout_rule.clone(), independent_sequences.clone()]),
            };
            let exposure = match &recipe.exposure {
                ExperimentExposure::NoInterference { assumption } | ExperimentExposure::ContrastBounds { assumption, .. }
                | ExperimentExposure::ArmBounds { assumption, .. } => assumption.clone(),
            };
            let mut assumptions = premise; assumptions.push(recipe.consistency.clone()); assumptions.push(exposure);
            (recipe.outcomes.iter().map(|o| format!("{} [{}]", o.variable_id, o.units)).collect(), recipe.population.clone(), unit.into(), assumptions)
        },
        StaticRegime { recipe } => (
            recipe.outcomes.iter().map(|o| format!("{} [{}]", o.variable_id, o.units)).collect(),
            recipe.population.clone(), "independent complete longitudinal subject".into(),
            vec![recipe.positivity.clone(), recipe.exchangeability.clone(), recipe.consistency.clone(),
                recipe.followup.clone(), recipe.model_assumption.clone(), recipe.isolation.clone()],
        ),
        Excursion { recipe } => (
            recipe.outcomes.iter().map(|o| format!("{} [{}]", o.variable_id, o.units)).collect(),
            recipe.population.clone(), "independent randomized longitudinal unit".into(),
            vec![recipe.assignment_law.clone(), recipe.independence.clone(), recipe.followup.clone(),
                recipe.consistency.clone(), recipe.isolation.clone(), format!("lag {} with all other assignments following the recorded law", recipe.lag)],
        ),
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
        Entry::Public.admit(ctx)?;
        check_cancel(ctx)?;
        let size = encoded_size(&recipe, ctx)?;
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
        let program = Arc::new(design_lowering::lower(
            DesignInput::Native { identification: &identification, recipe: &recipe },
            dependency,
            quote,
        )?);
        Ok(Self {
            recipe: Arc::new(recipe),
            identification: Arc::new(identification),
            dependency,
            point: Arc::new(point),
            program,
            _lease: lease,
        })
    }
    /// The lowered program every execution of this design enters.
    #[must_use]
    pub const fn lowered(&self) -> &Arc<LoweredProgram> {
        &self.program
    }
    /// One view over every layer of the lowered program this design executes. Reading it
    /// executes nothing.
    #[must_use]
    pub fn plan_view(&self) -> crate::planner::PlanView {
        crate::planner::PlanView::of_program(
            Arc::clone(&self.program),
            crate::planner::PlanStage::Prepared,
        )
    }
    /// Identity of the lowered program this design executes.
    #[must_use]
    pub fn program_identity(&self) -> SemanticDigest {
        self.program.identity()
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
        self.execute_entry(Entry::Public, inference, ctx)
    }
    /// [`Self::execute`] through `entry`: the public path enters the design program and passes
    /// the release support gate; a child of an enclosing program's dispatch enters without it.
    pub(crate) fn execute_entry(
        &self,
        entry: Entry,
        inference: JointDesignInference,
        ctx: &ExecutionContext,
    ) -> Result<NativeJointDesignResult, JointPanelError> {
        let scoped = entry.enter(ctx, &self.program)?;
        if entry == Entry::Public {
            require_release_support(&scoped, JointPanelRoute::Design)?;
        }
        let ctx = &scoped;
        check_cancel(ctx)?;
        let size = encoded_size(&inference, ctx)?;
        let draws = match &inference {
            JointDesignInference::Frequentist
            | JointDesignInference::SyntheticConformal { .. }
            | JointDesignInference::SwitchbackExactRandomization { .. } => 1,
            JointDesignInference::LocalComplierBayesian { draws, .. } => *draws,
            JointDesignInference::StaticRegimeBootstrap { replicates, .. }
            | JointDesignInference::GaussianLongitudinalBootstrap { replicates, .. } => *replicates,
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
    /// No native design composition inherits a release support from its inputs.
    #[must_use]
    pub const fn support(&self) -> &'static str {
        "unsupported"
    }
    /// Calibration of this composed native law is not established by execution alone.
    #[must_use]
    pub const fn calibration(&self) -> &'static str {
        "not_measured"
    }
    /// Exact retained statistical inference recipe, including resampling controls.
    #[must_use]
    pub fn inference(&self) -> &JointDesignInference {
        &self.inference
    }
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
    /// Identity of the lowered program that produced this law.
    #[must_use]
    pub fn program_identity(&self) -> SemanticDigest {
        self.prepared.program.identity()
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
        self.export_artifact_with_context(&ExecutionContext::production_default(0), limits)
    }
    /// Export under the caller's cancellation, admission and artifact-size controls.
    /// # Errors
    /// Invalid encoding, cancellation or resource refusal.
    pub fn export_artifact_with_context(
        &self,
        ctx: &ExecutionContext,
        limits: super::DesignArtifactLimits,
    ) -> Result<Vec<u8>, JointPanelError> {
        let wire = NativeReplayRef {
            version: 1,
            recipe: self.prepared.recipe.as_ref(),
            identification: self.prepared.identification.as_ref(),
            inference: self.inference.as_ref(),
            law: self.law.as_ref(),
        };
        let size = encoded_size(&wire, ctx)?;
        if size > limits.max_bytes {
            return Err(JointPanelError::Engine(super::super::engine::PanelError::Resource(
                "native joint-design artifact exceeds the declared portable size".into(),
            )));
        }
        let quote = u64::try_from(size).ok().and_then(|n| n.checked_mul(2)).ok_or_else(|| {
            JointPanelError::Engine(super::super::engine::PanelError::Resource(
                "artifact size overflow".into(),
            ))
        })?;
        let _lease = admitted(ctx, quote)?;
        check_cancel(ctx)?;
        let bytes = antecedent_io::convert::to_cbor(&wire).map_err(failure)?;
        check_cancel(ctx)?;
        Ok(bytes)
    }
    /// Two-sided interval at `level` for a prespecified contrast of the retained targets.
    ///
    /// A posterior law gives the equal-tailed interval of its aligned contrast draws. A
    /// whole-program randomized design gives a Student-t interval whose variance is unbiased
    /// under equal unit-outcome variance and whose degrees of freedom are Satterthwaite's
    /// (bias-reduced linearization), so few or unequally weighted units widen the interval; with
    /// equal weights it is the retained sandwich variance on `units - 1` degrees of freedom.
    /// Other frequentist moment laws give the large-sample normal interval of their joint
    /// covariance. A conformal region is projected at its own joint level, and a local
    /// complier ratio returns its posterior interval or its retained 95% Wald interval.
    /// # Errors
    /// Invalid level or coefficients, a level the retained law cannot resolve, a withheld or
    /// unbounded component, or a law that carries no joint uncertainty.
    pub fn contrast_interval(
        &self,
        coefficients: &[f64],
        level: f64,
    ) -> Result<[f64; 2], JointPanelError> {
        use antecedent_estimate::joint_design::compiled::JointDesignLaw;
        if !level.is_finite()
            || level <= 0.0
            || level >= 1.0
            || coefficients.iter().any(|c| !c.is_finite())
        {
            return Err(failure(
                "a contrast interval needs a level strictly inside (0, 1) and finite coefficients",
            ));
        }
        match (self.prepared.recipe.as_ref(), self.law.as_ref()) {
            (_, JointNativeDesignLaw::Moments(JointDesignLaw::Bayesian(posterior))) => {
                if posterior.draws.iter().any(|draw| draw.len() != coefficients.len()) {
                    return Err(failure("contrast coefficients do not match the retained targets"));
                }
                let values = posterior
                    .draws
                    .iter()
                    .map(|draw| draw.iter().zip(coefficients).map(|(v, c)| v * c).sum::<f64>())
                    .collect();
                equal_tailed(values, level)
            }
            (
                JointDesignRecipe::Randomized { units, .. },
                JointNativeDesignLaw::Moments(JointDesignLaw::Frequentist(_)),
            ) => randomized_contrast_interval(units, coefficients, level),
            (_, JointNativeDesignLaw::Moments(law)) => {
                let (mean, variance) = law.estimate().contrast(coefficients).map_err(failure)?;
                let radius = antecedent_stats::normal_ppf(0.5 + 0.5 * level) * variance.sqrt();
                Ok([mean - radius, mean + radius])
            }
            (_, JointNativeDesignLaw::SyntheticConformal(region)) => {
                if (region.joint_level - level).abs() > 1e-12 {
                    return Err(failure(
                        "a conformal region is projected only at its own retained joint level",
                    ));
                }
                region
                    .contrast_interval(coefficients)
                    .map(|(_, interval)| interval)
                    .map_err(failure)
            }
            (_, JointNativeDesignLaw::LocalComplier(local)) => {
                let &[scale] = coefficients else {
                    return Err(failure("a local complier effect has exactly one coordinate"));
                };
                let [lower, upper] = match &local.posterior {
                    Some(draws) => equal_tailed(draws.iter().map(|draw| draw[0]).collect(), level)?,
                    None if (level - 0.95).abs() <= 1e-12 => {
                        local.interval_95.ok_or_else(|| {
                            failure("the Wald interval is withheld at a weak stage or thin support")
                        })?
                    }
                    None => {
                        return Err(failure("the retained Wald interval is at the 95% level only"));
                    }
                };
                let (a, b) = (scale * lower, scale * upper);
                Ok([a.min(b), a.max(b)])
            }
            _ => Err(failure(
                "the retained native law carries no joint uncertainty for a contrast interval",
            )),
        }
    }
}
/// Equal-tailed interval of retained draws; the tail must hold at least one draw on each side.
fn equal_tailed(mut values: Vec<f64>, level: f64) -> Result<[f64; 2], JointPanelError> {
    let n = values.len();
    let tail = 0.5 * (1.0 - level) * n as f64;
    // Largest whole number of draws not exceeding the tail mass.
    let cut = (1..=n).take_while(|k| *k as f64 <= tail).count();
    if cut == 0 || values.iter().any(|v| !v.is_finite()) {
        return Err(failure("too few finite retained draws to resolve the requested level"));
    }
    values.sort_by(f64::total_cmp);
    Ok([values[cut], values[n - 1 - cut]])
}
/// Student-t interval for a contrast of population-weighted Horvitz--Thompson program means.
///
/// The contrast is `sum z_i / W` with `z_i = c[arm_i] w_i y_i / p_i` and residual
/// `d_i = z_i - w_i * estimate`. Under the working model `Var(z_i) = s^2 w_i^2` (a common
/// outcome variance) `E d_i^2 = s^2 w_i^2 (1 - 2 w_i / W + sum w^2 / W^2)`, so dividing each
/// squared residual by that factor makes the variance unbiased. The degrees of freedom are
/// Satterthwaite's `tr(A O)^2 / tr(A O A O)` for the variance's quadratic form `A` and the
/// working covariance `O = diag(w_i^2)`, evaluated without forming a unit-by-unit matrix.
fn randomized_contrast_interval(
    units: &[antecedent_estimate::joint_design::randomized::JointRandomizedUnit],
    coefficients: &[f64],
    level: f64,
) -> Result<[f64; 2], JointPanelError> {
    if units.iter().any(|u| u.arm >= coefficients.len() || u.probabilities.len() <= u.arm) {
        return Err(failure("contrast coefficients do not match the retained programs"));
    }
    let total = units.iter().map(|u| u.population_weight).sum::<f64>();
    let scores = units
        .iter()
        .map(|u| coefficients[u.arm] * u.population_weight * u.outcome / u.probabilities[u.arm])
        .collect::<Vec<_>>();
    let estimate = scores.iter().sum::<f64>() / total;
    let weight_square = units.iter().map(|u| u.population_weight.powi(2)).sum::<f64>();
    let mut variance = 0.0_f64;
    // g_i, the reciprocal of each residual's expected share of its own working variance.
    let mut correction = Vec::with_capacity(units.len());
    for (unit, score) in units.iter().zip(&scores) {
        let w = unit.population_weight;
        let share = 1.0 - 2.0 * w / total + weight_square / total.powi(2);
        if share <= 1e-12 {
            return Err(failure(
                "one unit carries too much of the population weight for a small-sample interval",
            ));
        }
        correction.push(1.0 / share);
        variance += (score - w * estimate).powi(2) / share / total.powi(2);
    }
    // A = diag(g) - (1 u' + u 1') / W + q 1 1' with u_i = g_i w_i and q = sum g_i w_i^2 / W^2;
    // its off-diagonal part is r_i + r_j with r_i = q / 2 - u_i / W.
    let q =
        units.iter().zip(&correction).map(|(u, g)| g * u.population_weight.powi(2)).sum::<f64>()
            / total.powi(2);
    let (mut trace, mut diagonal, mut omega_sum, mut omega_r, mut omega_rr) =
        (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    for (unit, g) in units.iter().zip(&correction) {
        let w = unit.population_weight;
        let omega = w * w;
        let r = 0.5 * q - g * w / total;
        trace += omega * (g + 2.0 * r);
        diagonal += omega * omega * (g * g + 4.0 * g * r);
        omega_sum += omega;
        omega_r += omega * r;
        omega_rr += omega * r * r;
    }
    let trace_square = diagonal + 2.0 * omega_rr * omega_sum + 2.0 * omega_r * omega_r;
    let degrees = trace * trace / trace_square;
    let radius = antecedent_stats::student_t_ppf(0.5 + 0.5 * level, degrees) * variance.sqrt();
    if !(estimate.is_finite() && radius.is_finite() && degrees > 0.0) {
        return Err(failure("randomized contrast interval is not finite"));
    }
    Ok([estimate - radius, estimate + radius])
}
/// Fewest sequences per requested history at which the switchback intervals were measured.
const SWITCHBACK_SEQUENCES_PER_HISTORY: usize = 32;
/// Smallest estimated receipt first stage at which the complier intervals were measured.
const COMPLIER_FIRST_STAGE: f64 = 0.4;
/// Largest ratio, either way, between the declared prior mean of the residual variance and the
/// pre-period mean squared fit error at which the synthetic posterior interval was measured.
const SYNTHETIC_PRIOR_RATIO: f64 = 2.0;

impl NativeJointDesignResult {
    /// Calibration key of [`Self::contrast_interval`] at `level`.
    ///
    /// The key names the design's estimator and target, the interval construction, the
    /// inference mode and prior model, the independent units behind the fit (pre-periods for
    /// a synthetic rollout), the resampling or posterior size, and the support condition the
    /// interval was measured under: sequences per requested history for a switchback, units
    /// per program and weight equality for a randomized design, the estimated first stage
    /// for a complier effect, and for a synthetic posterior whether the declared prior mean
    /// of the residual variance is within a factor of two of the pre-period fit error.
    /// # Errors
    /// A law that carries no joint uncertainty, or a design with no measured interval.
    #[allow(
        clippy::too_many_lines,
        reason = "one arm per native design keeps each measured condition beside its design"
    )]
    pub fn contrast_interval_basis(
        &self,
        level: f64,
    ) -> Result<antecedent_core::CalibrationBasis, JointPanelError> {
        use antecedent_estimate::joint_design::compiled::JointDesignLaw;
        let recipe = self.prepared.recipe.as_ref();
        let (inference, interval_method) = match (recipe, self.law.as_ref()) {
            (_, JointNativeDesignLaw::Moments(JointDesignLaw::Bayesian(_))) => {
                ("Bayesian", "equal_tailed_posterior")
            }
            (
                JointDesignRecipe::Randomized { .. },
                JointNativeDesignLaw::Moments(JointDesignLaw::Frequentist(_)),
            ) => ("Frequentist", "satterthwaite_t"),
            (_, JointNativeDesignLaw::Moments(JointDesignLaw::FrequentistBootstrap { .. })) => {
                ("Frequentist", "bootstrap_covariance_normal")
            }
            (_, JointNativeDesignLaw::Moments(JointDesignLaw::Frequentist(_))) => {
                ("Frequentist", "normal_approximation")
            }
            (_, JointNativeDesignLaw::SyntheticConformal(_)) => {
                ("Frequentist", "conformal_projection")
            }
            (_, JointNativeDesignLaw::LocalComplier(local)) if local.posterior.is_some() => {
                ("Bayesian", "equal_tailed_posterior")
            }
            (_, JointNativeDesignLaw::LocalComplier(_)) => ("Frequentist", "wald_ratio"),
            _ => {
                return Err(failure(
                    "the retained native law carries no joint uncertainty for a contrast interval",
                ));
            }
        };
        let (posterior, replicates, draws) = match self.inference.as_ref() {
            JointDesignInference::Frequentist
            | JointDesignInference::SyntheticConformal { .. }
            | JointDesignInference::SwitchbackExactRandomization { .. } => ("", None, None),
            JointDesignInference::StaticRegimeBootstrap { replicates, .. }
            | JointDesignInference::GaussianLongitudinalBootstrap { replicates, .. } => {
                ("", Some(*replicates), None)
            }
            JointDesignInference::LocalComplierBayesian { draws, .. } => {
                ("bayesian_bootstrap_ratio", None, Some(*draws))
            }
            JointDesignInference::GaussianLongitudinalBayesian { prior } => {
                ("gaussian_longitudinal", None, Some(prior.draws))
            }
            JointDesignInference::RandomizedGaussian { prior } => {
                ("randomized_gaussian", None, Some(prior.draws))
            }
            JointDesignInference::RandomizedFinitePopulationGaussian { prior } => {
                ("randomized_finite_population_gaussian", None, Some(prior.arm.draws))
            }
            JointDesignInference::LongitudinalDirichlet { prior } => {
                ("longitudinal_dirichlet", None, Some(prior.draws))
            }
            JointDesignInference::SwitchbackGaussian { prior } => {
                ("switchback_gaussian", None, Some(prior.draws))
            }
            JointDesignInference::InstrumentedGaussian { prior } => {
                ("instrumented_gaussian", None, Some(prior.draws))
            }
            JointDesignInference::RolloutGaussian { prior } => {
                ("rollout_gaussian", None, Some(prior.draws))
            }
            JointDesignInference::SyntheticBayesian { prior } => {
                ("synthetic_dirichlet_inverse_gamma", None, Some(prior.draws))
            }
        };
        let (rows, support) = match recipe {
            JointDesignRecipe::Switchback { design, sequences } => (
                sequences.len(),
                if sequences.len() >= SWITCHBACK_SEQUENCES_PER_HISTORY * design.worlds.len() {
                    "sequences_per_history_at_least_32"
                } else {
                    "sequences_per_history_below_32"
                },
            ),
            JointDesignRecipe::Randomized { design, units } => {
                let equal = units.windows(2).all(|pair| {
                    pair[0].population_weight.to_bits() == pair[1].population_weight.to_bits()
                });
                let per_program = if equal {
                    super::super::measured_intervals::EQUAL_WEIGHT_UNITS_PER_PROGRAM
                } else {
                    super::super::measured_intervals::UNEQUAL_WEIGHT_UNITS_PER_PROGRAM
                };
                let dense = units.len() >= per_program * design.worlds.len();
                (
                    units.len(),
                    match (equal, dense) {
                        (true, true) => "equal_weights_32_units_per_program",
                        (true, false) => "equal_weights_sparse_programs",
                        (false, true) => "unequal_weights_4_units_per_program",
                        (false, false) => "unequal_weights_sparse_programs",
                    },
                )
            }
            JointDesignRecipe::Instrumented { units, .. } => (units.len(), "independent_units"),
            JointDesignRecipe::DifferenceInDifferences { units, .. } => {
                (units.len(), "independent_units")
            }
            JointDesignRecipe::GaussianLongitudinal { episodes, .. } => {
                (episodes.len(), "independent_episodes")
            }
            JointDesignRecipe::Longitudinal { episodes, .. } => {
                (episodes.len(), "independent_episodes")
            }
            JointDesignRecipe::InstrumentedLocal { units, .. } => (
                units.len(),
                match self.law.as_ref() {
                    JointNativeDesignLaw::LocalComplier(local)
                        if local.first_stage.abs() >= COMPLIER_FIRST_STAGE =>
                    {
                        "first_stage_at_least_0_4"
                    }
                    _ => "first_stage_below_0_4",
                },
            ),
            JointDesignRecipe::SyntheticControl { design, observations } => {
                let pre_periods = design
                    .worlds
                    .iter()
                    .map(|world| {
                        observations
                            .iter()
                            .filter(|row| {
                                row.unit == world.treated_unit && row.time < world.intervention
                            })
                            .count()
                    })
                    .min()
                    .unwrap_or(0);
                let support = match (self.inference.as_ref(), self.prepared.point.as_ref()) {
                    (
                        JointDesignInference::SyntheticBayesian { prior },
                        JointNativeDesignLaw::SyntheticPoint(points),
                    ) => {
                        let fit_error =
                            points.iter().map(|p| p.preperiod_rmse.powi(2)).sum::<f64>()
                                / points.len().max(1) as f64;
                        let declared = prior.variance_scale / (prior.variance_shape - 1.0);
                        if fit_error > 0.0
                            && declared.is_finite()
                            && declared >= fit_error / SYNTHETIC_PRIOR_RATIO
                            && declared <= fit_error * SYNTHETIC_PRIOR_RATIO
                        {
                            "variance_prior_within_2x_preperiod_error"
                        } else {
                            "variance_prior_outside_2x_preperiod_error"
                        }
                    }
                    _ => "exchangeable_residuals",
                };
                (pre_periods, support)
            }
            _ => return Err(failure("no contrast interval of this native design was measured")),
        };
        Ok(super::super::measured_intervals::joint_response_basis(
            &super::super::measured_intervals::JointResponseAxes {
                procedure: "native_design_contrast",
                inference,
                estimator: recipe.method(),
                interval_method,
                support,
                posterior,
                functional: recipe.target_kind(),
            },
            level,
            rows,
            replicates,
            draws,
        ))
    }
    /// [`Self::contrast_interval`] with its calibration key and that key's standing against
    /// the shipped coverage records: calibrated only at a measured level, construction,
    /// support condition, sample size and resampling or posterior size.
    /// # Errors
    /// As [`Self::contrast_interval`] and [`Self::contrast_interval_basis`].
    pub fn contrast_interval_reading(
        &self,
        coefficients: &[f64],
        level: f64,
    ) -> Result<super::super::JointResponseReading<[f64; 2]>, JointPanelError> {
        let bounds = self.contrast_interval(coefficients, level)?;
        let basis = self.contrast_interval_basis(level)?;
        let calibration = crate::analysis::joint_law_composed_basis::slot(&basis);
        Ok(super::super::JointResponseReading { bounds, basis, calibration })
    }
}
#[derive(Serialize)]
struct NativeReplayRef<'a> {
    version: u16,
    recipe: &'a JointDesignRecipe,
    identification: &'a NativeJointIdentification,
    inference: &'a JointDesignInference,
    law: &'a JointNativeDesignLaw,
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

/// Replay a native design and retain its compiler-owned execution and admission.
///
/// This is historical replay; current-source authority requires independently matching a
/// current prepared source with `consume_current`.
///
/// # Errors
/// Malformed artifact, nonreproducible law, cancellation or resource admission refusal.
pub fn replay_native_joint_design_execution(
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

impl crate::analysis::program_handle::CompiledProgramHandle for PreparedNativeJointDesign {
    fn program(&self) -> &std::sync::Arc<crate::planner::lowered::LoweredProgram> {
        &self.program
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
pub(crate) use tests::{gaussian_longitudinal_chain_fixture, native_experiment_fixtures};
