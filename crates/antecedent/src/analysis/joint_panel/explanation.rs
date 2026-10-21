//! Execute the next coarser response attempt after unique change attribution fails.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::{
    Arc, ExecutionContext, JointPanelError, JointPanelFunctional, JointPanelResponseFunctional,
    JointPanelResult, JointPanelWorld, PanelResponse, ScientificTarget, ScientificWorldId,
    SemanticDigest, contracts, engine, joint_panel_target, source,
};
use antecedent_design::scientific_source::{ScientificDecisionSource, ScientificSourcePayload};
use engine::time_varying::{CoarserPanelContrast, PanelExplanation};

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
