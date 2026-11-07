//! Intervention and world planning for nonlinear SDEs.

use serde::{Deserialize, Serialize};

use crate::mechanism_expr::MechanismExpression;

use super::{MAX_SDE_KNOTS, SdeError, SdeInput, SdeSystem, bad, unsupported};

/// A sustained `do(X_i = value)` that follows a piecewise-constant schedule.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SdeAction {
    /// State coordinate held.
    pub variable: usize,
    /// Strictly increasing action times, on the simulation grid.
    pub times: Vec<f64>,
    /// From `times[j]` until the next time the coordinate is held at `values[j]`.
    pub values: Vec<f64>,
}

impl SdeAction {
    fn value_at(&self, t: f64) -> Option<f64> {
        if !t.is_finite() || t < self.times[0] {
            return None;
        }
        let index = self.times.partition_point(|&x| x <= t) - 1;
        Some(self.values[index])
    }
}

/// A named intervention regime: coordinate holds and replaced input schedules.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SdeWorld {
    /// Stable world identity.
    pub id: String,
    /// Component holds (`do` on state coordinates).
    #[serde(default)]
    pub actions: Vec<SdeAction>,
    /// Replacement schedules, by input name; must lie inside the declared envelope.
    #[serde(default)]
    pub inputs: Vec<SdeInput>,
}

/// A paired contrast of two worlds under shared random numbers.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SdeContrast {
    /// Stable contrast identity.
    pub id: String,
    /// World id of the treated arm.
    pub treated: String,
    /// World id of the control arm.
    pub control: String,
}

/// Reserved identity of the world with no intervention.
pub const BASELINE_WORLD: &str = "baseline";

/// A world with its input schedules fully resolved against the system.
#[derive(Clone, Debug)]
pub struct WorldPlan {
    /// World identity.
    pub id: String,
    /// Component holds.
    pub actions: Vec<SdeAction>,
    /// One schedule per system input, in system order.
    pub inputs: Vec<SdeInput>,
}

// Map a scheduled action to a declared grid cell without comparing absolute
// times in the path loop. Tolerance is capped below half a cell, so it cannot
// silently select a neighboring action step.
pub(super) fn action_cell(time: f64, start: f64, step: f64) -> Result<f64, SdeError> {
    if time <= start {
        return Ok(0.0);
    }
    let cells = (time - start) / step;
    let rounded = cells.round();
    let tolerance =
        1e-9 * step + 8.0 * f64::EPSILON * (time.abs() + start.abs() + (rounded * step).abs());
    if !cells.is_finite()
        || rounded > (1_u64 << 53) as f64
        || !tolerance.is_finite()
        || tolerance >= 0.5 * step
        || (time - (start + rounded * step)).abs() > tolerance
    {
        return Err(unsupported(
            antecedent_core::reason_details::sde::ACTION_TIME_OFF_GRID,
            format!("action time {time} is not representable on the declared grid"),
        ));
    }
    Ok(rounded)
}

fn expression_nodes(expression: &MechanismExpression) -> usize {
    match expression {
        MechanismExpression::Constant { .. }
        | MechanismExpression::Variable { .. }
        | MechanismExpression::Parameter { .. } => 1,
        MechanismExpression::Add { terms } | MechanismExpression::Multiply { factors: terms } => {
            terms.iter().map(expression_nodes).fold(1, usize::saturating_add)
        }
        MechanismExpression::Tanh { argument }
        | MechanismExpression::Sin { argument }
        | MechanismExpression::Softplus { argument } => {
            expression_nodes(argument).saturating_add(1)
        }
    }
}

fn input_bytes(input: &SdeInput) -> usize {
    input
        .times
        .len()
        .saturating_add(input.values.len())
        .saturating_mul(std::mem::size_of::<f64>())
        .saturating_add(input.name.len())
        .saturating_add(std::mem::size_of::<SdeInput>())
}

impl SdeSystem {
    /// Bytes [`Self::plan_world`] allocates for `world`: its identity and action schedules, and
    /// one schedule per system input, a world replacement standing in for the system's own.
    /// The plan is held for the whole run, so a caller admits this before building it.
    #[must_use]
    pub fn world_plan_bytes(&self, world: &SdeWorld) -> u64 {
        let actions = world.actions.iter().fold(0usize, |total, action| {
            let knots = action.times.len().saturating_add(action.values.len());
            total
                .saturating_add(knots.saturating_mul(std::mem::size_of::<f64>()))
                .saturating_add(std::mem::size_of::<SdeAction>())
        });
        let inputs = self.inputs.iter().fold(0usize, |total, own| {
            let held = world.inputs.iter().rfind(|r| r.name == own.name).unwrap_or(own);
            total.saturating_add(input_bytes(held))
        });
        let bytes = std::mem::size_of::<WorldPlan>()
            .saturating_add(world.id.len())
            .saturating_add(actions)
            .saturating_add(inputs);
        u64::try_from(bytes).unwrap_or(u64::MAX)
    }

    /// Bytes recompiling this system at other parameters holds while the run lasts: the cloned
    /// declaration, one tape node per expression node, and the interval enclosure's value and
    /// derivative rows over every state and input.
    pub(super) fn recertification_bytes(&self) -> Result<u64, SdeError> {
        let width = self.variables.len().saturating_add(self.inputs.len());
        let nodes = self
            .drift
            .iter()
            .chain(self.diffusion.iter().map(|g| &g.expression))
            .map(expression_nodes)
            .fold(0usize, usize::saturating_add);
        let per_node = 88usize.saturating_add(width.saturating_mul(32));
        let inputs = self.inputs.iter().map(input_bytes).fold(0usize, usize::saturating_add);
        let names = self.variables.iter().map(String::len).fold(0usize, usize::saturating_add);
        let bytes = nodes
            .saturating_mul(per_node)
            .saturating_add(inputs)
            .saturating_add(names)
            .saturating_add(std::mem::size_of::<SdeSystem>());
        u64::try_from(bytes).map_err(|_| bad("SDE recertification size overflow"))
    }

    /// Resolve a world against this system and a simulation grid.
    ///
    /// # Errors
    /// Invalid actions, unknown or out-of-envelope replacement inputs, or action times off
    /// the grid (`sde.action_time_off_grid`).
    #[allow(
        clippy::float_cmp,
        reason = "exact equality is required for discrete support or floating point progress"
    )]
    pub fn plan_world(
        &self,
        world: &SdeWorld,
        start: f64,
        step: f64,
    ) -> Result<WorldPlan, SdeError> {
        if !start.is_finite() || !step.is_finite() || step <= 0.0 || start + step == start {
            return Err(bad("SDE action grid is not representable"));
        }
        self.plan_world_checked(world, &world.id, Some((start, step)))
    }

    /// Resolve a world for a continuous-time observation schedule. Action times
    /// remain as declared; this does not promise event-exact numerical propagation.
    ///
    /// # Errors
    /// A non-finite start, invalid action, or replacement input outside its envelope.
    pub fn plan_world_continuous(
        &self,
        world: &SdeWorld,
        start: f64,
    ) -> Result<WorldPlan, SdeError> {
        if !start.is_finite() {
            return Err(bad("SDE observation start is not finite"));
        }
        self.plan_world_checked(world, &world.id, None)
    }

    /// [`Self::plan_world_continuous`] under the identity `id` instead of `world.id`, without
    /// copying the world to rename it.
    ///
    /// # Errors
    /// As [`Self::plan_world_continuous`]; an empty `id` is an invalid identity.
    pub fn plan_world_continuous_as(
        &self,
        world: &SdeWorld,
        id: &str,
        start: f64,
    ) -> Result<WorldPlan, SdeError> {
        if !start.is_finite() {
            return Err(bad("SDE observation start is not finite"));
        }
        self.plan_world_checked(world, id, None)
    }

    fn plan_world_checked(
        &self,
        world: &SdeWorld,
        id: &str,
        grid: Option<(f64, f64)>,
    ) -> Result<WorldPlan, SdeError> {
        let d = self.variables.len();
        if id.trim().is_empty() {
            return Err(bad("SDE world identity"));
        }
        let mut held = std::collections::BTreeSet::new();
        for action in &world.actions {
            let n = action.times.len();
            if action.variable >= d
                || n == 0
                || n > MAX_SDE_KNOTS
                || action.values.len() != n
                || action.times.iter().chain(&action.values).any(|v| !v.is_finite())
                || action.times.windows(2).any(|w| w[1] <= w[0])
                || !held.insert(action.variable)
            {
                return Err(bad("SDE action"));
            }
            if !self.boundary(action.variable).contains_all(&action.values) {
                return Err(bad("SDE action value outside the declared state interval"));
            }
            if let Some((start, step)) = grid {
                for &time in &action.times {
                    action_cell(time, start, step)?;
                }
            }
        }
        let mut inputs = self.inputs.clone();
        for replacement in &world.inputs {
            replacement.validate()?;
            let slot = inputs
                .iter()
                .position(|i| i.name == replacement.name)
                .ok_or_else(|| bad("SDE replacement input is not declared by the system"))?;
            let (lo, hi) = self.inputs[slot].range();
            let (rlo, rhi) = replacement.range();
            if rlo < lo || rhi > hi {
                return Err(unsupported(
                    antecedent_core::reason_details::sde::INPUT_OUTSIDE_CERTIFIED_ENVELOPE,
                    format!(
                        "replacement schedule of '{}' has range [{rlo}, {rhi}] outside the \
                         certified envelope [{lo}, {hi}]; widen the declared envelope so the \
                         regularity certificate covers it",
                        replacement.name
                    ),
                ));
            }
            inputs[slot] = replacement.clone();
            inputs[slot].envelope = self.inputs[slot].envelope;
        }
        Ok(WorldPlan { id: id.to_owned(), actions: world.actions.clone(), inputs })
    }
}

impl WorldPlan {
    /// The value coordinate `variable` is held at under this world's actions at time `t`,
    /// if an action is in force. Actions take effect at their declared time, inclusive;
    /// the preceding value is held strictly before the next action time. Non-finite
    /// query times return `None`.
    #[must_use]
    pub fn held_value(&self, variable: usize, t: f64) -> Option<f64> {
        self.actions.iter().find(|a| a.variable == variable).and_then(|a| a.value_at(t))
    }
}
