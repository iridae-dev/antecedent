//! Per-path external schedules for the compiled event response route.

use antecedent_core::ExecutionContext;

use super::simulation::run_path;
use super::{CANCEL_STRIDE, action_cell, check_cancel, counter_normal};
use super::{CompiledSde, SdeError, SdeScheme, SdeSimulationPolicy, WorldPlan, bad};

/// Execute one path on the coarse and coupled fine grid using checked world schedules.
/// The replacement schedules must have been resolved by the system's world planner.
/// This adapter leaves the original ensemble and sampling entry points untouched.
///
/// # Errors
/// Invalid initial state or unsupported scheme, solver failure or cancellation.
pub fn solve_event_input_path(
    compiled: &CompiledSde,
    plan: &WorldPlan,
    initial: &[f64],
    policy: &SdeSimulationPolicy,
    path: u64,
    ctx: &ExecutionContext,
) -> Result<(Vec<f64>, Vec<f64>), SdeError> {
    policy.validate()?;
    let dimension = compiled.dimension();
    if path >= policy.paths as u64
        || policy.output_every != 1
        || !policy.check_step
        || initial.len() != dimension
        || initial.iter().zip(&compiled.certificate.state_bounds).any(|(&x, &[lo, hi])| {
            !x.is_finite() || lo.is_some_and(|v| x < v) || hi.is_some_and(|v| x > v)
        })
        || plan.inputs.len() != compiled.system.inputs.len()
        || plan
            .inputs
            .iter()
            .zip(&compiled.system.inputs)
            .any(|(input, declared)| input.name != declared.name)
        || (matches!(policy.scheme, SdeScheme::Milstein)
            && !compiled.certificate.diagonal_milstein_admissible)
    {
        return Err(bad("event path grid, world, initial state or scheme"));
    }
    // Recheck holds and schedules against this exact compiled system. Callers
    // may construct a WorldPlan directly, including with arbitrary indices.
    let world = super::SdeWorld {
        id: plan.id.clone(),
        actions: plan.actions.clone(),
        inputs: plan.inputs.clone(),
    };
    let plan = compiled.system.plan_world(&world, policy.start, policy.step)?;
    let theta = compiled.system.parameter_values();
    let mut coarse = vec![0.0; (policy.steps + 1) * dimension];
    run_path(compiled, &theta, &plan, policy, initial, path, false, &mut coarse, ctx)?;
    // Keep the Brownian primitive at the original half-step precision. A fine
    // runner normally records coarse timestamps, so use a separate recorder
    // that advances on the same half-step grid.
    let fine = run_event_fine_grid(compiled, &theta, &plan, policy, initial, path, ctx)?;
    Ok((coarse, fine))
}

// A separate fine recorder leaves the previously calibrated solver's item
// and output contract intact. Its even steps are checked against run_path.
fn run_event_fine_grid(
    compiled: &CompiledSde,
    theta: &[f64],
    world: &WorldPlan,
    policy: &SdeSimulationPolicy,
    initial: &[f64],
    path: u64,
    ctx: &ExecutionContext,
) -> Result<Vec<f64>, SdeError> {
    let d = compiled.dimension();
    let half = policy.step / 2.0;
    let cells = world
        .actions
        .iter()
        .map(|action| {
            action
                .times
                .iter()
                .map(|&time| action_cell(time, policy.start, policy.step))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut state = initial.to_vec();
    let mut arguments = vec![0.0; d + world.inputs.len()];
    let mut next = vec![0.0; d];
    let mut rows = Vec::with_capacity((2 * policy.steps + 1) * d);
    for step in 0..=2 * policy.steps {
        if step % CANCEL_STRIDE == 0 {
            check_cancel(ctx)?;
        }
        for (action, times) in world.actions.iter().zip(&cells) {
            let count = times.partition_point(|&cell| 2.0 * cell <= step as f64);
            if count > 0 {
                state[action.variable] = action.values[count - 1];
            }
        }
        rows.extend_from_slice(&state);
        if step == 2 * policy.steps {
            break;
        }
        arguments[..d].copy_from_slice(&state);
        let time = policy.start + step as f64 * half;
        for (slot, input) in world.inputs.iter().enumerate() {
            arguments[d + slot] = input.value_at(time);
        }
        for i in 0..d {
            let held = world
                .actions
                .iter()
                .zip(&cells)
                .any(|(action, times)| action.variable == i && 2.0 * times[0] <= step as f64);
            if held {
                next[i] = state[i];
                continue;
            }
            let drift = compiled.drift[i].evaluate(&arguments, theta)?.value();
            let evaluation = compiled.diffusion[i].evaluate(&arguments, theta)?;
            let h = evaluation.value();
            let square_root = compiled.system.diffusion[i].square_root;
            let diffusion = if square_root { h.max(0.0).sqrt() } else { h };
            let dw = half.sqrt() * counter_normal(policy.seed, path, step as u64, i as u64);
            let mut value = state[i] + drift * half + diffusion * dw;
            if matches!(policy.scheme, SdeScheme::Milstein) {
                let derivative = evaluation.gradient()[i];
                let correction = if square_root {
                    if h > 0.0 { 0.25 * derivative } else { 0.0 }
                } else {
                    0.5 * diffusion * derivative
                };
                value += correction * (dw * dw - half);
            }
            next[i] = value;
        }
        for i in 0..d {
            state[i] = compiled.system.boundary(i).apply(next[i]);
        }
        if state.iter().any(|value| !value.is_finite()) {
            return Err(SdeError::Numerical(format!(
                "nonfinite refined state at step {step} of path {path}"
            )));
        }
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refined_recorder_agrees_at_all_shared_brownian_grid_states() {
        let system = super::super::SdeSystem {
            variables: vec!["x".into()],
            inputs: vec![],
            parameters: vec![],
            drift: vec![crate::mechanism_expr::MechanismExpression::Constant { value: 1.0 }],
            diffusion: vec![super::super::SdeDiffusion {
                expression: crate::mechanism_expr::MechanismExpression::Constant { value: 0.4 },
                square_root: false,
            }],
            boundaries: vec![],
        };
        let policy = SdeSimulationPolicy {
            start: 0.0,
            step: 0.25,
            steps: 4,
            output_every: 1,
            scheme: SdeScheme::EulerMaruyama,
            paths: 2,
            seed: 54,
            weak_tolerance: 0.01,
            check_step: true,
            retain_final_samples: false,
        };
        let compiled = system.compile().unwrap();
        let plan = system
            .plan_world(
                &super::super::SdeWorld { id: "baseline".into(), ..Default::default() },
                policy.start,
                policy.step,
            )
            .unwrap();
        let ctx = ExecutionContext::for_tests(0);
        let (_, fine) = solve_event_input_path(&compiled, &plan, &[0.0], &policy, 1, &ctx).unwrap();
        let mut reference = vec![0.0; 5];
        run_path(&compiled, &[], &plan, &policy, &[0.0], 1, true, &mut reference, &ctx).unwrap();
        assert_eq!(fine.into_iter().step_by(2).collect::<Vec<_>>(), reference);
    }
}
