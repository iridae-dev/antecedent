use super::{
    Arc, CausalError, CompiledCausalModel, Dag, ExecutionContext, Intervention,
    InterventionOverlay, ModelView, NoiseReplayer, PolicySupport, VariableId, policy_support,
    replay_noise,
};

/// Model-based value of a policy on one outcome, with the unintervened outcome on the same
/// noise, and the per-route support the policy was run under.
///
/// The means are properties of the fitted model, not of the population: they are as good as the
/// structure and the mechanisms are. Only the simulation error is quantified here
/// ([`Self::policy_simulation_stderr`], [`Self::difference_simulation_stderr`]); the model-fit
/// uncertainty is not, and no interval is claimed.
#[derive(Clone, Debug, PartialEq)]
pub struct PolicyValue {
    /// The outcome variable.
    pub outcome: VariableId,
    /// Draws of the policy run and of the baseline replayed on its noise.
    pub n_draws: usize,
    /// Mean outcome under the policy.
    pub policy_mean: f64,
    /// Mean outcome with no intervention, on the same noise.
    pub baseline_mean: f64,
    /// `policy_mean - baseline_mean`, from the paired draws.
    pub mean_difference: f64,
    /// Monte Carlo standard error of `policy_mean`.
    pub policy_simulation_stderr: f64,
    /// Monte Carlo standard error of `mean_difference` (paired, so common noise cancels).
    pub difference_simulation_stderr: f64,
    /// The support of each route for this policy.
    pub support: PolicySupport,
}

internal_engine! {
    /// Value of `interventions` on `outcome` under the fitted model: draw `n` rows with their noise
    /// under the policy, replay the same noise with no intervention, and report both means and
    /// their paired difference with the simulation error.
    ///
    /// This is [`sample_with_noise`](super::sample_with_noise) and [`replay_noise`], the executor every policy route shares;
    /// a policy the simulation route refuses is refused here with the same registered reason code.
    ///
    /// # Errors
    ///
    /// A refused policy, an outcome absent from the model, fewer than two draws, a table over the
    /// context's memory limit, or cancellation.
    #[doc(hidden)]
    pub fn policy_value(
        model: &CompiledCausalModel,
        interventions: &[Intervention],
        outcome: VariableId,
        n: usize,
        ctx: &ExecutionContext,
    ) -> Result<PolicyValue, CausalError> {
        let node = policy_value_request(model, outcome, n)?;
        let support = policy_support(model, interventions);
        let overlay = InterventionOverlay::from_interventions(model, interventions)?;
        policy_value_on(model, &overlay, support, (outcome, node), n, ctx)
    }
}

/// The outcome's dense node, after checking the request can report a simulation error.
fn policy_value_request(
    model: &CompiledCausalModel,
    outcome: VariableId,
    n: usize,
) -> Result<antecedent_graph::DenseNodeId, CausalError> {
    if n < 2 {
        return Err(CausalError::Compile {
            message: "a policy value needs at least two draws to report its simulation error"
                .into(),
        });
    }
    model.dense_of(outcome).ok_or_else(|| CausalError::Compile {
        message: format!("outcome {outcome} is absent from the fitted model"),
    })
}

/// The executor of [`policy_value`] on an already compiled overlay and resolved support, shared
/// by the one-shot call and [`PreparedPolicy`].
fn policy_value_on(
    model: &CompiledCausalModel,
    overlay: &InterventionOverlay,
    support: PolicySupport,
    (outcome, node): (VariableId, antecedent_graph::DenseNodeId),
    n: usize,
    ctx: &ExecutionContext,
) -> Result<PolicyValue, CausalError> {
    let (policy, noise) =
        NoiseReplayer::new().sample(&ModelView::with_overlay(model, overlay.clone()), n, ctx)?;
    let baseline = replay_noise(model, &[], &noise, 0, ctx)?;
    let treated = policy.column(node.as_usize()).map_err(CausalError::from)?;
    let untreated = baseline.column(node.as_usize()).map_err(CausalError::from)?;
    let count = n as f64;
    let mean = |column: &[f64]| column.iter().sum::<f64>() / count;
    let stderr = |values: &mut dyn Iterator<Item = f64>, centre: f64| {
        let ss: f64 = values.map(|v| (v - centre) * (v - centre)).sum();
        (ss / (count - 1.0) / count).sqrt()
    };
    let (policy_mean, baseline_mean) = (mean(treated), mean(untreated));
    let mean_difference = policy_mean - baseline_mean;
    Ok(PolicyValue {
        outcome,
        n_draws: n,
        policy_mean,
        baseline_mean,
        mean_difference,
        policy_simulation_stderr: stderr(&mut treated.iter().copied(), policy_mean),
        difference_simulation_stderr: stderr(
            &mut treated.iter().zip(untreated).map(|(t, u)| t - u),
            mean_difference,
        ),
        support,
    })
}

/// A policy program compiled and supported once against one fitted model, rerun without being
/// compiled again: the retained form of [`policy_value`].
///
/// [`Self::prepare`] validates the program against the model (a refused program is refused
/// there, with the registered reason code), compiles its overlay and resolves the support of
/// every route. [`Self::value`] then runs the executor [`policy_value`] runs, on the retained
/// overlay, so it equals the one-shot value bit for bit at the same context.
/// [`Self::rebind`] carries the program to a refitted model: the overlay and support are
/// resolved against that model, and the program itself is not compiled again.
#[derive(Clone, Debug)]
pub struct PreparedPolicy {
    interventions: Arc<[Intervention]>,
    graph: Arc<Dag>,
    overlay: InterventionOverlay,
    support: PolicySupport,
    program_compilations: u64,
    model_bindings: u64,
}

impl PreparedPolicy {
    /// Compile `interventions` against `model` and resolve the support of each route.
    ///
    /// # Errors
    ///
    /// A program the model refuses (an unknown variable, a mistyped or circular expression, a
    /// node both set and shifted), with its registered reason code.
    pub fn prepare(
        model: &CompiledCausalModel,
        interventions: Vec<Intervention>,
    ) -> Result<Self, CausalError> {
        let overlay = InterventionOverlay::from_interventions(model, &interventions)?;
        let support = policy_support(model, &interventions);
        Ok(Self {
            interventions: Arc::from(interventions),
            graph: Arc::clone(&model.graph),
            overlay,
            support,
            program_compilations: 1,
            model_bindings: 1,
        })
    }

    /// The same program bound to `model` (for example the model refitted on refreshed data):
    /// the overlay and support are resolved against it; [`Self::program_compilations`] is
    /// unchanged and [`Self::model_bindings`] grows by one.
    ///
    /// # Errors
    ///
    /// A model that refuses the program.
    pub fn rebind(&self, model: &CompiledCausalModel) -> Result<Self, CausalError> {
        let overlay = if model.graph.as_ref() == self.graph.as_ref() {
            self.overlay.clone()
        } else {
            InterventionOverlay::from_interventions(model, &self.interventions)?
        };
        Ok(Self {
            interventions: Arc::clone(&self.interventions),
            graph: Arc::clone(&model.graph),
            overlay,
            support: policy_support(model, &self.interventions),
            program_compilations: self.program_compilations,
            model_bindings: self.model_bindings.saturating_add(1),
        })
    }

    /// The program.
    #[must_use]
    pub fn interventions(&self) -> &[Intervention] {
        &self.interventions
    }

    /// The support of each route, fixed at preparation.
    #[must_use]
    pub fn support(&self) -> &PolicySupport {
        &self.support
    }

    /// How many times the program was compiled: one, however often it is rerun or rebound.
    #[must_use]
    pub fn program_compilations(&self) -> u64 {
        self.program_compilations
    }

    /// How many models the program's overlay and support were resolved against.
    #[must_use]
    pub fn model_bindings(&self) -> u64 {
        self.model_bindings
    }

    /// Value of the retained program on `outcome`, run on the retained compiled overlay without
    /// compiling the program again; it equals the compiled policy-value route on the same model,
    /// program and context. `model` must be the model the policy was prepared or rebound on.
    ///
    /// # Errors
    ///
    /// An outcome absent from the model, fewer than two draws, a table over the context's
    /// memory limit, or cancellation.
    pub fn value(
        &self,
        model: &CompiledCausalModel,
        outcome: VariableId,
        n: usize,
        ctx: &ExecutionContext,
    ) -> Result<PolicyValue, CausalError> {
        let node = policy_value_request(model, outcome, n)?;
        policy_value_on(model, &self.overlay, self.support.clone(), (outcome, node), n, ctx)
    }
}
