//! A fitted causal model as the response source of an adaptive-policy search.
//!
//! Each candidate policy is lowered to an assigned value program on the action variable and
//! rolled out with [`antecedent_model::rollout_temporal`], so the policy sees exactly the
//! lagged and current values the model realises. The source holds one compiled model per
//! parameter or posterior draw and a fixed noise seed per split: every policy meets the same
//! parameter draws and the same exogenous noise (common random numbers), which makes policy
//! values and their paired contrasts share draws. The optimisation and held-out splits use
//! independent noise streams.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    derive_stream, CausalRng, ExecutionContext, Intervention, InterventionSequence,
    SequencedIntervention, TemporalPolicy, ValueExpr, VariableId,
};
use antecedent_design::adaptive_policy::AdaptivePolicyFamily;
use antecedent_design::adaptive_policy_search::{
    AdaptivePolicySearchError, DrawSplit, HeldOutDesign, PolicyPath, PolicyRolloutSource,
    ResponseStanding,
};
use antecedent_model::{
    rollout_temporal, CompiledCausalModel, MechanismWorkspace, ModelError, TemporalNoise,
    TemporalPlan, ValueBatch,
};

/// Stage utility of a path: asymmetric linear plus quadratic loss of the outcome around a
/// target, and linear plus quadratic action cost, discounted over steps.
///
/// `stage = -(over * max(y - target, 0) + under * max(target - y, 0) + quadratic * (y - target)^2
///           + action_linear * |a| + action_quadratic * a^2)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StageUtility {
    /// Variable whose realised value is scored each step.
    pub outcome: VariableId,
    /// Outcome target.
    pub target: f64,
    /// Loss per unit above the target.
    pub over: f64,
    /// Loss per unit below the target.
    pub under: f64,
    /// Quadratic loss coefficient around the target.
    pub quadratic: f64,
    /// Loss per unit of absolute action.
    pub action_linear: f64,
    /// Quadratic action loss coefficient.
    pub action_quadratic: f64,
    /// Per-step discount in `(0, 1]`.
    pub discount: f64,
}

impl StageUtility {
    fn value(&self, y: f64, a: f64) -> f64 {
        let d = y - self.target;
        -(self.over * d.max(0.0)
            + self.under * (-d).max(0.0)
            + self.quadratic * d * d
            + self.action_linear * a.abs()
            + self.action_quadratic * a * a)
    }
}

/// How the rollout rows are laid out and seeded.
#[derive(Clone, Debug)]
pub struct ModelRolloutDesign {
    /// Steps per path.
    pub horizon: usize,
    /// Rows (independent units) simulated per model on the optimisation split.
    pub rows: usize,
    /// Rows per model on the held-out split; zero when the design has no held-out noise.
    pub held_out_rows: usize,
    /// Master seed of the exogenous noise; the two splits use independent derived streams.
    pub noise_seed: u64,
    /// Realised state before time zero, applied to every row, as `(variable, value)`;
    /// variables not listed start at zero. Needed when a policy reads a lagged value.
    pub initial_state: Vec<(VariableId, f64)>,
    /// Cost per unit of absolute action.
    pub unit_cost: f64,
    /// Exposure per unit of positive action.
    pub exposure_weight: f64,
    /// The policy decides every `decision_interval` steps (1 = every step). At the steps in
    /// between the action holds its last realised value, so the action variable is still
    /// assigned explicitly (never left to its natural mechanism). Step 0 is always a decision.
    pub decision_interval: usize,
}

/// A fitted causal model (or one model per parameter/posterior draw) as response source.
pub struct ModelPolicySource {
    models: Vec<Arc<CompiledCausalModel>>,
    action: VariableId,
    utility: StageUtility,
    design: ModelRolloutDesign,
    observed: Vec<VariableId>,
    standing: ResponseStanding,
}

impl ModelPolicySource {
    /// Bind compiled models (one per parameter draw), the action variable the policy assigns,
    /// the stage utility and the rollout design.
    ///
    /// `observed` lists the variables a policy may read when it acts; `standing` states
    /// whether the fitted model's identification supports a causal policy-effect reading.
    ///
    /// # Errors
    /// No models, a variable absent from a model, an invalid design or utility.
    pub fn new(
        models: Vec<Arc<CompiledCausalModel>>,
        action: VariableId,
        utility: StageUtility,
        design: ModelRolloutDesign,
        observed: Vec<VariableId>,
        standing: ResponseStanding,
    ) -> Result<Self, AdaptivePolicySearchError> {
        let bad = |m: &str| AdaptivePolicySearchError::Invalid(m.into());
        if models.is_empty()
            || design.horizon == 0
            || design.rows == 0
            || design.decision_interval == 0
        {
            return Err(bad("a model source needs models, a horizon and rows"));
        }
        let finite = [
            utility.target,
            utility.over,
            utility.under,
            utility.quadratic,
            utility.action_linear,
            utility.action_quadratic,
            utility.discount,
            design.unit_cost,
            design.exposure_weight,
        ];
        if finite.iter().any(|v| !v.is_finite())
            || !(utility.discount > 0.0 && utility.discount <= 1.0)
        {
            return Err(bad("utility and design coefficients must be finite; discount in (0, 1]"));
        }
        for model in &models {
            let missing = [action, utility.outcome]
                .into_iter()
                .chain(design.initial_state.iter().map(|(v, _)| *v))
                .any(|v| model.dense_of(v).is_none());
            if missing {
                return Err(bad(
                    "action, outcome and initial-state variables must be in the model",
                ));
            }
        }
        let n_nodes = models[0].n_nodes();
        if models.iter().any(|m| m.n_nodes() != n_nodes) {
            return Err(bad("every parameter draw must share one graph"));
        }
        Ok(Self { models, action, utility, design, observed, standing })
    }

    fn rows_for(&self, split: DrawSplit) -> usize {
        match split {
            DrawSplit::Optimization => self.design.rows,
            DrawSplit::HeldOut => self.design.held_out_rows,
        }
    }

    fn sequence(
        &self,
        family: &AdaptivePolicyFamily,
        theta: &[f64],
    ) -> Result<InterventionSequence, AdaptivePolicySearchError> {
        let lower = |e: antecedent_design::adaptive_policy::AdaptivePolicyError| {
            AdaptivePolicySearchError::Invalid(e.to_string())
        };
        let every = self.design.decision_interval;
        let steps = if every == 1 && !family.is_open_loop() {
            let last = i32::try_from(self.design.horizon - 1)
                .map_err(|_| AdaptivePolicySearchError::Invalid("horizon exceeds i32".into()))?;
            let expr = family.to_value_expr(theta, 0).map_err(lower)?;
            vec![SequencedIntervention::new(
                Intervention::assign(self.action, expr),
                TemporalPolicy::sustained(0, last),
            )]
        } else {
            let mut steps = Vec::with_capacity(self.design.horizon);
            for t in 0..self.design.horizon {
                let at = i32::try_from(t).map_err(|_| {
                    AdaptivePolicySearchError::Invalid("horizon exceeds i32".into())
                })?;
                // A decision step runs the policy; any other step holds the last action.
                let expr = if t % every == 0 {
                    family.to_value_expr(theta, t / every).map_err(lower)?
                } else {
                    ValueExpr::lag(self.action, 1)
                };
                steps.push(SequencedIntervention::new(
                    Intervention::assign(self.action, expr),
                    TemporalPolicy::pulse(at),
                ));
            }
            steps
        };
        Ok(InterventionSequence::new(steps))
    }

    fn initial_history(
        &self,
        model: &CompiledCausalModel,
        rows: usize,
        depth: u32,
    ) -> Vec<ValueBatch> {
        if depth == 0 || self.design.initial_state.is_empty() {
            return Vec::new();
        }
        let n_nodes = model.n_nodes();
        let mut values = vec![0.0; rows * n_nodes];
        for (variable, value) in &self.design.initial_state {
            if let Some(node) = model.dense_of(*variable) {
                values[node.as_usize() * rows..(node.as_usize() + 1) * rows].fill(*value);
            }
        }
        let batch = ValueBatch { n_rows: rows, n_nodes, values: Arc::from(values) };
        vec![batch; depth as usize]
    }
}

fn map_model_error(error: ModelError) -> AdaptivePolicySearchError {
    match error {
        ModelError::Cancelled => AdaptivePolicySearchError::Cancelled,
        other => AdaptivePolicySearchError::Source(other.to_string()),
    }
}

impl PolicyRolloutSource for ModelPolicySource {
    fn identity(&self) -> String {
        format!(
            "fitted_model_rollout:models={}:rows={}:held_out_rows={}:horizon={}:seed={}",
            self.models.len(),
            self.design.rows,
            self.design.held_out_rows,
            self.design.horizon,
            self.design.noise_seed
        )
    }

    fn n_draws(&self, split: DrawSplit) -> usize {
        self.models.len() * self.rows_for(split)
    }

    fn horizon(&self) -> usize {
        self.design.horizon
    }

    fn standing(&self) -> ResponseStanding {
        self.standing.clone()
    }

    fn heldout_design(&self) -> HeldOutDesign {
        if self.design.held_out_rows > 0 {
            HeldOutDesign::IndependentDraws
        } else {
            HeldOutDesign::Unavailable(
                "the rollout design declares no independent held-out noise".into(),
            )
        }
    }

    fn observable_variables(&self) -> Option<Vec<VariableId>> {
        Some(self.observed.clone())
    }

    fn rollout(
        &self,
        family: &AdaptivePolicyFamily,
        theta: &[f64],
        split: DrawSplit,
        ctx: &ExecutionContext,
    ) -> Result<Vec<PolicyPath>, AdaptivePolicySearchError> {
        let rows = self.rows_for(split);
        let sequence = self.sequence(family, theta)?;
        let depth = family.max_lag().max(u32::from(self.design.decision_interval > 1));
        let split_tag = u64::from(split == DrawSplit::HeldOut);
        let mut out = Vec::with_capacity(self.models.len() * rows);
        for (m, model) in self.models.iter().enumerate() {
            let history = self.initial_history(model, rows, depth);
            let plan = TemporalPlan {
                sequence: &sequence,
                n_steps: self.design.horizon,
                n_rows: rows,
                window: depth,
                initial_history: &history,
            };
            let mut rng = CausalRng::from_seed(derive_stream(
                derive_stream(self.design.noise_seed, split_tag),
                m as u64,
            ));
            let mut ws = MechanismWorkspace::default();
            let path = rollout_temporal(model, &plan, TemporalNoise::Draw(&mut rng), &mut ws, ctx)
                .map_err(map_model_error)?;
            let action_node = model.dense_of(self.action).map(|n| n.as_usize()).unwrap_or(0);
            let outcome_node =
                model.dense_of(self.utility.outcome).map(|n| n.as_usize()).unwrap_or(0);
            for row in 0..rows {
                let mut utility = 0.0;
                let mut discount = 1.0;
                let (mut actions, mut costs, mut exposures) = (Vec::new(), Vec::new(), Vec::new());
                for t in 0..self.design.horizon {
                    let a = path.column(t, action_node).map_err(map_model_error)?[row];
                    let y = path.column(t, outcome_node).map_err(map_model_error)?[row];
                    utility += discount * self.utility.value(y, a);
                    discount *= self.utility.discount;
                    actions.push(a);
                    costs.push(self.design.unit_cost * a.abs());
                    exposures.push(self.design.exposure_weight * a.max(0.0));
                }
                out.push(PolicyPath {
                    utility,
                    actions,
                    step_costs: costs,
                    step_exposures: exposures,
                });
            }
        }
        Ok(out)
    }
}
