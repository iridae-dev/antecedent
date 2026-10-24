//! Execute the next coarser response attempt after unique change attribution fails.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::{
    Arc, ExecutionContext, JointPanelError, JointPanelFunctional, JointPanelResponseFunctional,
    JointPanelResult, JointPanelWorld, PanelResponse, ScientificTarget, ScientificWorldId,
    SemanticDigest, contracts, engine, joint_panel_target, source,
};
use antecedent_design::scientific_source::{ScientificDecisionSource, ScientificSourcePayload};
use engine::time_varying::{CoarserPanelContrast, PanelExplanation};

pub use engine::decomposition::{
    ChangeComponent, DecompositionEndpoint, DecompositionJointUncertainty, MechanismPeriodMap,
    PanelDecomposition, PanelDecompositionRequest, ReferenceDecomposition,
    ReferenceDecompositionCovariance, StreamedDecomposition,
};

impl JointPanelResult {
    /// Execute all sixteen reference-world hybrids on this exact jointly fitted law.
    ///
    /// This is a model-conditional reference decomposition, not a claim of uniquely
    /// identified causal attribution. Endpoints explicitly bind exposure/history,
    /// saturation, population and remaining mechanism blocks. Temporal mechanism
    /// indices refer to this one fitted path, preserving selection and draw coupling.
    /// A stationary fit cannot manufacture a distinct after mechanism.
    ///
    /// # Errors
    /// Invalid endpoints, paths, resources, cancellation or numerical failure.
    pub fn decompose_reference(
        &self,
        request: &PanelDecompositionRequest,
        periods: &MechanismPeriodMap,
        ctx: &ExecutionContext,
    ) -> Result<PanelDecomposition, JointPanelError> {
        source::check_cancelled(ctx)?;
        let data = self.prepared().model_data();
        Ok(match &self.fit {
            super::JointPanelFit::Invariant(fit) => engine::decomposition::decompose_invariant_law(
                data, request, periods, &fit.point, &fit.draws, ctx,
            )?,
            super::JointPanelFit::Varying(fit) => engine::decomposition::decompose_temporal_law(
                data, request, periods, &fit.point, &fit.draws, ctx,
            )?,
        })
    }
}

impl JointPanelResult {
    /// [`Self::decompose_reference`] reduced to the point decomposition and the law's joint
    /// moments, without holding any per-draw table or paired path of the whole law.
    ///
    /// The values equal those of the retained route; use it when the per-draw tables are not
    /// needed. Failed law slots stay counted as failed.
    ///
    /// # Errors
    /// Invalid endpoints, paths, resources, cancellation or numerical failure.
    pub fn decompose_reference_streamed(
        &self,
        request: &PanelDecompositionRequest,
        periods: &MechanismPeriodMap,
        ctx: &ExecutionContext,
    ) -> Result<StreamedDecomposition, JointPanelError> {
        source::check_cancelled(ctx)?;
        let data = self.prepared().model_data();
        Ok(match &self.fit {
            super::JointPanelFit::Invariant(fit) => {
                engine::decomposition::decompose_invariant_law_streamed(
                    data, request, periods, &fit.point, &fit.draws, ctx,
                )?
            }
            super::JointPanelFit::Varying(fit) => {
                engine::decomposition::decompose_temporal_law_streamed(
                    data, request, periods, &fit.point, &fit.draws, ctx,
                )?
            }
        })
    }
}

/// Explicit comparison periods/programs; a fitted change curve supplies no attribution proof.
#[derive(Clone, Debug)]
pub struct JointPanelExplanationRequest {
    /// Finer explanation whose unique causal attribution is not established.
    pub explanation: PanelExplanation,
    /// Program whose later-minus-earlier response is compared.
    pub world: ScientificWorldId,
    /// Explicit baseline program whose same response change is subtracted.
    pub reference_world: ScientificWorldId,
    /// Inclusive earlier reference periods.
    pub before: (usize, usize),
    /// Inclusive later comparison periods, strictly after the earlier window.
    pub after: (usize, usize),
}
/// Actual coarser compiled execution and its original unique-attribution refusal.
#[derive(Clone, Debug)]
pub struct JointPanelExplanationAttempt {
    /// Requested explanation, world identities and actual period windows.
    pub request: JointPanelExplanationRequest,
    /// Retained refusal; a fitted curve does not distinguish the finer explanations.
    pub unique_refusal: CoarserPanelContrast,
    /// Identity of the parent scientific question, distinct from the derived target.
    pub parent_target: SemanticDigest,
    /// Actual scalar change-of-program-differences mean.
    pub point: f64,
    /// Same aligned fitted draws, with failed slots retained; not a calibrated interval.
    pub draws: Vec<Option<f64>>,
    /// Source bound to the actual derived per-period Linear target and conditional proof.
    pub source: ScientificDecisionSource,
    compiled: JointPanelResult,
    retention: Arc<antecedent_core::execution::AdmissionLease>,
}
impl JointPanelExplanationAttempt {
    /// Refuse promotion from a coarser response change to unique causal attribution.
    /// # Errors
    /// Always refuses without an independently earned finer attribution proof.
    pub fn require_unique_attribution(&self) -> Result<(), JointPanelError> {
        Err(JointPanelError::Unsupported(format!(
            "unique {:?} attribution remains unsupported; executed the coarser total response contrast instead",
            self.request.explanation
        )))
    }
    /// Exact derived compiled response, usable by the existing independent artifact path.
    #[must_use]
    pub fn compiled_response(&self) -> &JointPanelResult {
        self.retention.observe_current();
        &self.compiled
    }
}
impl JointPanelResult {
    /// Refuse unique attribution, then execute the next coarser total response contrast.
    ///
    /// The actual target is (later−earlier) for `world` minus the same change for
    /// `reference_world`. Each selected period is its own prefix world in the
    /// common Linear contrast. This reuses the already fitted mechanism and draws.
    /// # Errors
    /// Invalid worlds/windows, target compilation, numerical, resource or cancellation errors.
    pub fn attempt_explanation(
        &self,
        request: JointPanelExplanationRequest,
        ctx: &ExecutionContext,
    ) -> Result<JointPanelExplanationAttempt, JointPanelError> {
        source::check_cancelled(ctx)?;
        let selected = selected_worlds(self, &request)?;
        let quote = expansion_quote(&selected, &request)?
            .checked_add(
                u64::try_from(self.draws.len())
                    .map_err(JointPanelError::compile)?
                    .checked_mul(std::mem::size_of::<Option<f64>>() as u64)
                    .ok_or_else(|| JointPanelError::compile("coarser draw storage overflow"))?,
            )
            .ok_or_else(|| JointPanelError::compile("coarser storage overflow"))?;
        source::check_memory(ctx, quote)?;
        let retention = Arc::new(
            ctx.acquire_admission(0, quote)
                .map_err(|error| engine::PanelError::Resource(error.to_string()))?,
        );
        let (worlds, terms) = prefix_worlds(&selected, &request)?;
        let target = coarse_target(self, &selected, &worlds, &terms)?;
        let prepared = self.prepared().retarget(worlds, target, ctx)?;
        // Reuse this result's immutable fitted law even if the caller disabled future caching.
        let compiled = prepared.evaluate_fit(self.fit.clone(), self.dependency_identity(), ctx)?;
        let weights = compiled
            .request()
            .worlds
            .iter()
            .zip(&terms)
            .map(|(world, (_, coefficient))| {
                let mut weights = vec![0.0; world.horizon()];
                if let Some(last) = weights.last_mut() {
                    *last = *coefficient;
                }
                weights
            })
            .collect::<Vec<_>>();
        let reduce = |rows: &[PanelResponse]| -> Result<f64, JointPanelError> {
            let value = engine::project_panel_responses(rows, &weights)?.iter().sum::<f64>();
            if value.is_finite() {
                Ok(value)
            } else {
                Err(engine::PanelError::Numerical("coarser response contrast overflow".into())
                    .into())
            }
        };
        let point = reduce(&compiled.point)?;
        let draws = compiled
            .draws
            .iter()
            .map(|row| {
                source::check_cancelled(ctx)?;
                row.as_deref().map(reduce).transpose()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut means =
            compiled.numerical_project_mean_source(&JointPanelResponseFunctional::TerminalMean)?;
        let mut quantity = means.coordinates[0].clone();
        quantity.regime_id = format!("compiled_contrast:{}", compiled.request().target.identity());
        quantity.functional_id = "mean_linear_window_change".into();
        quantity.horizon = u32::try_from(request.after.1 + 1).map_err(JointPanelError::compile)?;
        means.coordinates = vec![quantity];
        means.means = vec![point];
        let source = ScientificDecisionSource::new(
            compiled.request().target.clone(),
            contracts::conditional_projection_product(compiled.request())?,
            compiled.dependency_identity(),
            ScientificSourcePayload::Means(means),
        )
        .map_err(JointPanelError::compile)?;
        Ok(JointPanelExplanationAttempt {
            unique_refusal: engine::time_varying::require_coarser_contrast(request.explanation),
            request,
            parent_target: self.request().target.identity(),
            point,
            draws,
            source,
            compiled,
            retention,
        })
    }
}
fn selected_worlds<'a>(
    result: &'a JointPanelResult,
    request: &JointPanelExplanationRequest,
) -> Result<[&'a JointPanelWorld; 2], JointPanelError> {
    let world = |id| {
        result
            .request()
            .worlds
            .iter()
            .find(|world| world.world == id)
            .ok_or_else(|| JointPanelError::compile("explanation names an absent world"))
    };
    let reference = world(request.reference_world)?;
    let active = world(request.world)?;
    if request.world == request.reference_world
        || !reference.reference
        || request.before.0 > request.before.1
        || request.after.0 > request.after.1
        || request.before.1 >= request.after.0
        || [reference, active].iter().any(|world| request.after.1 >= world.horizon())
    {
        return Err(JointPanelError::compile(
            "explanation requires distinct worlds, an explicit reference and ordered in-range disjoint windows",
        ));
    }
    Ok([reference, active])
}
type PrefixWorlds = (Vec<JointPanelWorld>, Vec<(ScientificWorldId, f64)>);

fn prefix_worlds(
    selected: &[&JointPanelWorld; 2],
    request: &JointPanelExplanationRequest,
) -> Result<PrefixWorlds, JointPanelError> {
    let mut worlds = Vec::new();
    let mut terms = Vec::new();
    for (world_index, world) in selected.iter().enumerate() {
        for (window, sign) in [(request.before, -1.0), (request.after, 1.0)] {
            let coefficient =
                sign / (window.1 - window.0 + 1) as f64 * if world_index == 0 { -1.0 } else { 1.0 };
            for period in window.0..=window.1 {
                let id = ScientificWorldId::from_raw(
                    u32::try_from(worlds.len()).map_err(JointPanelError::compile)?,
                );
                let mut prefix = (**world).clone();
                prefix.world = id;
                prefix.reference = worlds.is_empty();
                prefix.program.id = format!("{}@period={period}", prefix.program.id);
                for channel in &mut prefix.program.channels {
                    channel.values.truncate(period + 1);
                }
                for scenario in &mut prefix.program.scenarios {
                    scenario.confounder.truncate(period + 1);
                    scenario.season.truncate(period + 1);
                }
                worlds.push(prefix);
                terms.push((id, coefficient));
            }
        }
    }
    Ok((worlds, terms))
}
fn expansion_quote(
    selected: &[&JointPanelWorld; 2],
    request: &JointPanelExplanationRequest,
) -> Result<u64, JointPanelError> {
    let mut bytes = 0_usize;
    for world in selected {
        let headers = std::mem::size_of::<JointPanelWorld>()
            + world.program.id.len()
            + 32
            + world
                .program
                .channels
                .iter()
                .map(|channel| {
                    std::mem::size_of::<engine::PanelChannelProgram>() + channel.id.len()
                })
                .sum::<usize>()
            + world
                .program
                .scenarios
                .iter()
                .map(|scenario| {
                    std::mem::size_of::<engine::PanelScenario>()
                        + scenario.group.len()
                        + scenario
                            .history
                            .iter()
                            .map(|channel| {
                                std::mem::size_of::<engine::PanelChannelHistory>()
                                    + channel.id.len()
                            })
                            .sum::<usize>()
                })
                .sum::<usize>();
        let width = world.program.channels.len() + 2 * world.program.scenarios.len();
        for period in (request.before.0..=request.before.1).chain(request.after.0..=request.after.1)
        {
            bytes = period
                .checked_add(1)
                .and_then(|n| n.checked_mul(width))
                .and_then(|n| n.checked_mul(8))
                .and_then(|n| n.checked_add(headers))
                .and_then(|n| bytes.checked_add(n))
                .ok_or_else(|| JointPanelError::compile("coarser prefix expansion overflow"))?;
        }
    }
    u64::try_from(
        bytes
            .checked_add(4096)
            .ok_or_else(|| JointPanelError::compile("coarser storage overflow"))?,
    )
    .map_err(JointPanelError::compile)
}

fn coarse_target(
    result: &JointPanelResult,
    selected: &[&JointPanelWorld; 2],
    worlds: &[JointPanelWorld],
    terms: &[(ScientificWorldId, f64)],
) -> Result<ScientificTarget, JointPanelError> {
    let generated = joint_panel_target(
        result.prepared().source_data().schema(),
        &result.request().columns,
        result.request().lag_order,
        worlds,
        JointPanelFunctional::Worlds,
        result.request().target.inference(),
    )?;
    let original = contracts::target_worlds(&result.request().target)?;
    let generated_worlds = contracts::target_worlds(&generated)?;
    let per_program = worlds.len() / 2;
    let restored = generated_worlds
        .iter()
        .enumerate()
        .map(|(index, prefix)| {
            let source = original
                .iter()
                .find(|world| world.id() == selected[index / per_program].world)
                .ok_or_else(|| JointPanelError::compile("missing original coarser source world"))?;
            super::projection::restore_prefix_clock(
                prefix,
                source,
                u32::try_from(worlds[index].horizon()).map_err(JointPanelError::compile)?,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let collection =
        antecedent_core::WorldCollection::new(restored).map_err(JointPanelError::compile)?;
    let contrast = antecedent_core::InterventionContrast::linear(
        collection,
        terms.iter().copied(),
        worlds[0].world,
    )
    .map_err(JointPanelError::compile)?;
    let mut assumptions = generated.assumptions().records().clone();
    assumptions.extend_unique(result.request().target.assumptions().records().entries.iter().filter(|record| !matches!(&record.assumption,antecedent_core::Assumption::Custom{id,..} if id.starts_with("panel.context.world."))));
    Ok(ScientificTarget::new(
        antecedent_core::TargetForm::Contrast(contrast),
        antecedent_core::ModelAssumptions::new(
            result.request().target.assumptions().graph(),
            assumptions,
        ),
        generated.inference(),
    ))
}

/// A checked identifying premise for one of the sixteen reference worlds.
#[derive(Clone, Debug, PartialEq)]
pub struct HybridReferenceSupport {
    /// Exposure/saturation/population/mechanism bit mask.
    pub mask: u8,
    /// Original identified intervention that exactly supplies this hybrid.
    pub source_world: Option<ScientificWorldId>,
    /// Concrete missing transport or identification obligation, if any.
    pub unresolved: Vec<String>,
    /// The same missing premises as typed obligations, for choosing the next design.
    pub obligations: Vec<ReferenceObligation>,
}
/// A premise the retained design has not earned for an assembled reference world.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ReferenceObligation {
    /// The source target carries no causal identification proof at all.
    MissingIdentificationProof,
    /// The assembled hybrid is not an intervention/context world of the retained proof.
    UnregisteredHybridWorld,
    /// Saturation/mechanism taken from another fitted period has no identified intervention.
    CrossPeriodMechanismTransport,
    /// Changed population weights have no identified conditional responses or transport law.
    ChangedPopulationTransport,
    /// The retained final-horizon target does not identify the time-weighted functional.
    NonTerminalFunctional,
}
/// Reference-world decomposition together with separately checked causal support.
/// A model-conditional decomposition remains useful when a causal premise is missing.
#[derive(Clone, Debug)]
pub struct CheckedReferenceDecomposition {
    /// Complete numerical execution and its original aligned law slots.
    pub decomposition: PanelDecomposition,
    /// Exact compiler identification product whose world permissions were consumed.
    pub identification_identity: SemanticDigest,
    /// Every hybrid is checked; endpoint support is not substituted for hybrid support.
    pub hybrids: Vec<HybridReferenceSupport>,
}
impl CheckedReferenceDecomposition {
    /// Whether all reference responses follow from retained identifying premises.
    /// This does not establish uniquely determined attribution or interval coverage.
    #[must_use]
    pub fn is_identified_reference(&self) -> bool {
        self.hybrids.len() == 16 && self.hybrids.iter().all(|world| world.unresolved.is_empty())
    }
    /// Require the checked reference-world claim, preserving individual missing premises.
    /// # Errors
    /// At least one hybrid lacks an identifying or transport premise.
    pub fn require_identified_reference(&self) -> Result<(), JointPanelError> {
        if self.is_identified_reference() {
            return Ok(());
        }
        Err(JointPanelError::Unsupported(format!(
            "reference-world identification obligations: {:?}",
            self.hybrids.iter().filter(|world| !world.unresolved.is_empty()).collect::<Vec<_>>()
        )))
    }
}
/// Executed finer decomposition or an actually evaluated coarser response contrast.
#[derive(Clone, Debug)]
pub enum IdentifiedReferenceAttempt {
    /// Every hybrid inherits an authentic exact-world identification result.
    Supported(CheckedReferenceDecomposition),
    /// Finer identification failed; retain its numerical result and execute the fallback.
    Coarser {
        /// Concrete per-hybrid missing premises, with the conditional numbers.
        requested: CheckedReferenceDecomposition,
        /// Actual next coarser response execution from the same fitted law.
        executed: JointPanelExplanationAttempt,
    },
}
impl JointPanelResult {
    /// Evaluate the reference worlds and check their identification and transport.
    ///
    /// Exact already identified interventions with unchanged fitted mechanism and
    /// empirical population inherit their original world proof. Changed parameter
    /// periods, population weights or context require independent transport evidence;
    /// a fitted curve or a matching dose range cannot supply that evidence. These
    /// obligations do not prevent the model-conditional reference calculation.
    ///
    /// # Errors
    /// Numerical request validation, cancellation, admission or response failure.
    pub fn decompose_reference_checked(
        &self,
        request: &PanelDecompositionRequest,
        periods: &MechanismPeriodMap,
        ctx: &ExecutionContext,
    ) -> Result<CheckedReferenceDecomposition, JointPanelError> {
        let decomposition = self.decompose_reference(request, periods, ctx)?;
        let groups = self.prepared().model_data().groups.len();
        let empirical = 1.0 / groups as f64;
        let terminal = request.time_weights.iter().enumerate().all(|(index, weight)| {
            weight.to_bits()
                == if index + 1 == request.time_weights.len() { 1.0_f64 } else { 0.0_f64 }.to_bits()
        });
        let mut hybrids = Vec::with_capacity(16);
        for mask in 0_u8..16 {
            source::check_cancelled(ctx)?;
            let exposure = if mask & 1 == 0 { &request.before } else { &request.after };
            let population = if mask & 4 == 0 { &request.before } else { &request.after };
            let mut program = exposure.program.clone();
            for scenario in &mut program.scenarios {
                let context = population
                    .program
                    .scenarios
                    .iter()
                    .find(|context| context.group == scenario.group)
                    .ok_or_else(|| {
                        JointPanelError::compile("hybrid population omits a source group")
                    })?;
                scenario.confounder.clone_from(&context.confounder);
                scenario.season.clone_from(&context.season);
            }
            let source_world = self
                .request()
                .worlds
                .iter()
                .find(|world| {
                    world.program.channels == program.channels
                        && world.program.scenarios == program.scenarios
                })
                .map(|world| world.world);
            let mut unresolved = Vec::new();
            let mut obligations = Vec::new();
            if !self.identification().is_identified_claim() {
                unresolved
                    .push("the source target has no retained causal identification proof".into());
                obligations.push(ReferenceObligation::MissingIdentificationProof);
            }
            if source_world.is_none() {
                unresolved.push("the assembled hybrid is not an intervention/context world in the retained identification product".into());
                obligations.push(ReferenceObligation::UnregisteredHybridWorld);
            }
            let natural = |axis: &[usize]| {
                axis.iter().enumerate().all(|(index, period)| {
                    *period
                        == match &self.fit {
                            super::JointPanelFit::Invariant(_) => 0,
                            super::JointPanelFit::Varying(_) => index,
                        }
                })
            };
            // The before path is used by every hybrid; the after path only by hybrids that
            // take saturation or mechanism from the after endpoint.
            let uses_after_mechanism = mask & 0b1010 != 0;
            if !natural(&periods.before)
                || (uses_after_mechanism
                    && (periods.before != periods.after || !natural(&periods.after)))
            {
                unresolved.push("cross-period saturation/mechanism transport requires a separately identified parameter intervention; time variation alone supplies no such proof".into());
                obligations.push(ReferenceObligation::CrossPeriodMechanismTransport);
            }
            if population
                .population_weights
                .iter()
                .any(|weight| (*weight - empirical).abs() > 1e-12)
            {
                unresolved.push("changed population weights require identified conditional responses and an explicit population transport law".into());
                obligations.push(ReferenceObligation::ChangedPopulationTransport);
            }
            if !terminal {
                unresolved.push("the retained final-horizon target does not identify the requested intermediate/time-weighted functional".into());
                obligations.push(ReferenceObligation::NonTerminalFunctional);
            }
            hybrids.push(HybridReferenceSupport { mask, source_world, unresolved, obligations });
        }
        Ok(CheckedReferenceDecomposition {
            decomposition,
            identification_identity: self.identification().identity(),
            hybrids,
        })
    }
    /// Execute an identified reference decomposition, or the supplied next coarser target.
    /// Resource/cancellation/numerical errors propagate and are never called scientific refusal.
    /// # Errors
    /// Invalid finer request, invalid fallback target, cancellation or resource failure.
    pub fn attempt_identified_reference_decomposition(
        &self,
        request: &PanelDecompositionRequest,
        periods: &MechanismPeriodMap,
        fallback: JointPanelExplanationRequest,
        ctx: &ExecutionContext,
    ) -> Result<IdentifiedReferenceAttempt, JointPanelError> {
        let requested = self.decompose_reference_checked(request, periods, ctx)?;
        if requested.is_identified_reference() {
            return Ok(IdentifiedReferenceAttempt::Supported(requested));
        }
        let executed = self.attempt_explanation(fallback, ctx)?;
        Ok(IdentifiedReferenceAttempt::Coarser { requested, executed })
    }
}

mod design_adaptive;
pub use design_adaptive::{DesignAdaptiveReference, IdentifiedComponentStep, NextDesignStep};
mod scientific;
pub use scientific::{
    HistoricalPopulationReference, HistoricalReferenceDecomposition, PopulationReferenceAttempt,
    PopulationReferenceBindings, PopulationReferenceDecomposition,
    PopulationReferenceTransportProof, ReferenceComponentBindings,
    ScientificReferenceDecomposition, decompose_population_reference,
    decompose_scientific_reference, replay_population_reference, replay_reference_decomposition,
};
