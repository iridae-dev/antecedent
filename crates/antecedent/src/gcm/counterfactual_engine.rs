use super::{
    AbductionMissingPolicy, Arc, CausalError, CausalRng, CompiledCausalModel, CounterfactualEngine,
    CounterfactualWorld, ExecutionContext, ExogenousPosterior, Intervention, MechanismWorkspace,
    NoiseInferenceKind, TabularData, Value, VariableId,
};

internal_engine! {
    /// Abduction once, then unit-level ITE for binary hard interventions on `treatment`.
    ///
    /// # Errors
    ///
    /// Abduction / prediction failures.
    #[doc(hidden)]
    pub fn counterfactual_ite(
        model: CompiledCausalModel,
        data: &TabularData,
        treatment: VariableId,
        outcome: VariableId,
        active: f64,
        control: f64,
        ctx: &ExecutionContext,
    ) -> Result<IteResult, CausalError> {
        let engine = CounterfactualEngine::new(model);
        let exo = engine.abduct(data, AbductionMissingPolicy::Error, ctx)?;
        ite_from_abduction(&engine, exo, treatment, outcome, (active, control), ctx)
    }
}

/// [`counterfactual_ite`] with the noise of many-to-one (categorical) nodes drawn from `rng`.
///
/// One posterior draw of the mechanisms owns one abduction: a categorical node's disturbance is
/// uniform within the factual category's CDF bin given that draw's parameters, so a Bayesian
/// refit loop passes one stream through every draw and the noise posterior varies with the
/// parameters instead of repeating one seeded draw. Invertible mechanisms consume no randomness
/// and give the same unit effects as [`counterfactual_ite`].
///
/// # Errors
///
/// Abduction / prediction failures.
pub(crate) fn counterfactual_ite_with_noise(
    model: CompiledCausalModel,
    data: &TabularData,
    contrast: (VariableId, VariableId, f64, f64),
    ctx: &ExecutionContext,
    rng: &mut CausalRng,
) -> Result<IteResult, CausalError> {
    let (treatment, outcome, active, control) = contrast;
    let engine = CounterfactualEngine::new(model);
    let exo = engine.abduct_with_rng(data, AbductionMissingPolicy::Error, rng)?;
    ite_from_abduction(&engine, exo, treatment, outcome, (active, control), ctx)
}

/// Replay one abduced noise state under both interventions and summarize the unit effects.
pub(super) fn ite_from_abduction(
    engine: &CounterfactualEngine,
    exo: ExogenousPosterior,
    treatment: VariableId,
    outcome: VariableId,
    levels: (f64, f64),
    ctx: &ExecutionContext,
) -> Result<IteResult, CausalError> {
    let (active, control) = levels;
    let mut ws = MechanismWorkspace::default();
    let ite = engine.individual_treatment_effect(
        &exo,
        outcome,
        Intervention::set(treatment, Value::f64(active)),
        Intervention::set(treatment, Value::f64(control)),
        &mut ws,
        ctx,
    )?;
    let n = ite.len().max(1) as f64;
    let mean = ite.iter().sum::<f64>() / n;
    Ok(IteResult {
        unit_effects: ite,
        mean_ite: mean,
        noise_inference: exo.kind,
        exogenous: exo,
        unit_effect_intervals: None,
        unit_extrapolative: None,
    })
}

/// Abduction once, then the unit-level contrast of every listed outcome between two worlds,
/// each world a *set* of simultaneous interventions applied in one mutilated model.
///
/// The single-treatment [`counterfactual_ite`] is the case of one outcome and one `Set` per
/// world. Here `active` and `control` may each hold several interventions (a joint treatment
/// and its joint baseline), and every outcome is read from the same two replays, so the
/// contrasts share one abduced noise state.
///
/// # Errors
///
/// An empty outcome or world, an outcome that is not in the model, or abduction / prediction
/// failures.
pub fn counterfactual_contrasts(
    model: CompiledCausalModel,
    data: &TabularData,
    outcomes: &[VariableId],
    worlds: (&[Intervention], &[Intervention]),
    ctx: &ExecutionContext,
) -> Result<ContrastSetResult, CausalError> {
    let (active, control) = worlds;
    let request = ContrastRequest { outcomes, active, control, allow_nested: false };
    request.require_complete()?;
    let nodes = contrast_outcome_nodes(&model, outcomes)?;
    let engine = CounterfactualEngine::new(model);
    let exo = engine.abduct(data, AbductionMissingPolicy::Error, ctx)?;
    contrasts_from_abduction(&engine, exo, &request, &nodes, ctx)
}

/// What a contrast set replays: the outcomes read and the two worlds, each a set of
/// simultaneous interventions.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ContrastRequest<'a> {
    /// Outcomes read from both worlds.
    pub outcomes: &'a [VariableId],
    /// Interventions of the active world.
    pub active: &'a [Intervention],
    /// Interventions of the control world.
    pub control: &'a [Intervention],
    /// Whether nested interventions are admitted in either world.
    pub allow_nested: bool,
}

impl ContrastRequest<'_> {
    fn require_complete(&self) -> Result<(), CausalError> {
        if self.outcomes.is_empty() || self.active.is_empty() || self.control.is_empty() {
            return Err(CausalError::Compile {
                message: "a contrast set needs at least one outcome and a non-empty active and \
                          control world"
                    .into(),
            });
        }
        Ok(())
    }
}

/// The dense node of each contrast outcome.
pub(crate) fn contrast_outcome_nodes(
    model: &CompiledCausalModel,
    outcomes: &[VariableId],
) -> Result<Vec<antecedent_graph::DenseNodeId>, CausalError> {
    outcomes
        .iter()
        .map(|&outcome| {
            model.dense_of(outcome).ok_or_else(|| CausalError::Compile {
                message: format!("contrast outcome {outcome} is not in the model"),
            })
        })
        .collect()
}

/// Replay one abduced noise state under both worlds of `request` and contrast every outcome.
///
/// `nodes` are the outcomes' dense nodes in `model` ([`contrast_outcome_nodes`]); the engine's
/// model must be the one they were read from or a refit of it.
pub(crate) fn contrasts_from_abduction(
    engine: &CounterfactualEngine,
    exo: ExogenousPosterior,
    request: &ContrastRequest<'_>,
    nodes: &[antecedent_graph::DenseNodeId],
    ctx: &ExecutionContext,
) -> Result<ContrastSetResult, CausalError> {
    request.require_complete()?;
    let worlds = [
        CounterfactualWorld { unit_rows: None, interventions: Arc::from(request.active) },
        CounterfactualWorld { unit_rows: None, interventions: Arc::from(request.control) },
    ];
    let mut ws = MechanismWorkspace::default();
    let replay = engine.predict_retaining_outcomes(
        &exo,
        &worlds,
        request.outcomes,
        request.allow_nested,
        &mut ws,
        ctx,
    )?;
    let mut contrasts = Vec::with_capacity(request.outcomes.len());
    for (&outcome, &node) in request.outcomes.iter().zip(nodes) {
        let treated = replay.outcome_column(0, node)?;
        let baseline = replay.outcome_column(1, node)?;
        let effects: Vec<f64> = treated.iter().zip(baseline).map(|(a, c)| a - c).collect();
        let mean_ite = effects.iter().sum::<f64>() / effects.len().max(1) as f64;
        contrasts.push(OutcomeContrast { outcome, unit_effects: Arc::from(effects), mean_ite });
    }
    Ok(ContrastSetResult { contrasts, noise_inference: exo.kind, exogenous: exo })
}

/// [`counterfactual_contrasts`] over a fitted model with the noise of many-to-one (categorical)
/// nodes drawn from `rng`: one posterior draw of the mechanisms owns one abduction shared by every
/// outcome, as in [`counterfactual_ite_with_noise`].
pub(crate) fn counterfactual_contrasts_with_noise(
    model: CompiledCausalModel,
    data: &TabularData,
    request: &ContrastRequest<'_>,
    nodes: &[antecedent_graph::DenseNodeId],
    ctx: &ExecutionContext,
    rng: &mut CausalRng,
) -> Result<ContrastSetResult, CausalError> {
    let engine = CounterfactualEngine::new(model);
    let exo = engine.abduct_with_rng(data, AbductionMissingPolicy::Error, rng)?;
    contrasts_from_abduction(&engine, exo, request, nodes, ctx)
}

/// The unit-level contrast of one outcome in a [`ContrastSetResult`].
#[derive(Clone, Debug)]
pub struct OutcomeContrast {
    /// The outcome the contrast reads.
    pub outcome: VariableId,
    /// Per-unit effects, aligned with the factual rows.
    pub unit_effects: Arc<[f64]>,
    /// Mean of the unit effects.
    pub mean_ite: f64,
}

/// Contrasts of several outcomes between two worlds that share one abduced noise state.
#[derive(Clone, Debug)]
pub struct ContrastSetResult {
    /// One contrast per requested outcome, in the order requested.
    pub contrasts: Vec<OutcomeContrast>,
    /// How noise was obtained.
    pub noise_inference: NoiseInferenceKind,
    /// Shared exogenous state.
    pub exogenous: ExogenousPosterior,
}

impl ContrastSetResult {
    /// The contrast of `outcome`, when it was requested.
    #[must_use]
    pub fn contrast(&self, outcome: VariableId) -> Option<&OutcomeContrast> {
        self.contrasts.iter().find(|contrast| contrast.outcome == outcome)
    }

    /// The contrast of `outcome` as a single-contrast [`IteResult`] (no unit intervals and no
    /// extrapolation flags), when it was requested.
    #[must_use]
    pub fn ite_result(&self, outcome: VariableId) -> Option<IteResult> {
        self.contrast(outcome).map(|contrast| IteResult {
            unit_effects: Arc::clone(&contrast.unit_effects),
            mean_ite: contrast.mean_ite,
            noise_inference: self.noise_inference,
            exogenous: self.exogenous.clone(),
            unit_effect_intervals: None,
            unit_extrapolative: None,
        })
    }
}

/// ITE summary with visible noise-inference kind.
#[derive(Clone, Debug)]
pub struct IteResult {
    /// Per-unit effects.
    pub unit_effects: Arc<[f64]>,
    /// Mean ITE.
    pub mean_ite: f64,
    /// How noise was obtained.
    pub noise_inference: NoiseInferenceKind,
    /// Shared exogenous state.
    pub exogenous: ExogenousPosterior,
    /// Per-unit posterior intervals, when the inference mode produced per-unit
    /// draws (Bayesian: `unit_posterior_quantile`, a credible interval for the
    /// observed unit's contrast under the fitted mechanism, mechanism-refit
    /// uncertainty only, not a predictive interval for a new unit).
    ///
    /// `None` for Frequentist results, deliberately. The unit effect
    /// `Y_i(a) - Y_i(a0)` is not identified from data; it is computed under the
    /// fitted SCM. Its value depends jointly on every path mechanism's
    /// parameters and on disturbances abducted with those same parameters (or,
    /// downstream of a discrete mechanism, on one seeded noise draw under a
    /// rank-preserving coupling, which no sampling interval describes), and
    /// each mechanism family is selected on the same data, so a delta-method
    /// interval conditional on the selected family would ignore selection. The
    /// refit construction that carries this uncertainty is the Bayesian
    /// Dirichlet mechanism-refit posterior, already published and calibrated
    /// under `InferenceMode::Bayesian`; a frequentist copy of it would target
    /// the same object. A pointwise interval for the conditional average effect
    /// `tau(x)` (not a unit ITE) is `DrLearner::fit_pointwise_profiles`
    /// (uncalibrated). The diagnostic `gcm.counterfactual.uncertainty_unavailable`
    /// carries this reason on the result.
    pub unit_effect_intervals: Option<UnitEffectIntervals>,
    /// Per-unit extrapolation flags, aligned with [`Self::unit_effects`]: `true`
    /// when the unit's prediction into the arm it did not receive leaves that
    /// arm's observed support (see `gcm.counterfactual.support`).
    pub unit_extrapolative: Option<Arc<[bool]>>,
}

/// Level-tagged per-unit intervals for [`IteResult::unit_effects`].
#[derive(Clone, Debug, PartialEq)]
pub struct UnitEffectIntervals {
    /// Lower bound per unit, aligned with the unit effects.
    pub lower: Arc<[f64]>,
    /// Upper bound per unit.
    pub upper: Arc<[f64]>,
    /// Nominal level the bounds were read at.
    pub level: f64,
    /// Construction id (`unit_posterior_quantile`, `IntervalMethod::as_str`).
    pub method: &'static str,
}
