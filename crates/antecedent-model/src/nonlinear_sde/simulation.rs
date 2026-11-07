//! Admitted simulation, path sampling and ensemble diagnostics for nonlinear SDEs.

use std::sync::Arc;

use antecedent_core::ExecutionContext;
use antecedent_core::execution::AdmissionLease;

use super::{
    BASELINE_WORLD, CANCEL_STRIDE, CompiledSde, MAX_SDE_RETAINED_CELLS, MAX_SDE_STEP_EVALUATIONS,
    MAX_WORLDS, SdeContrast, SdeContrastEnsemble, SdeEnsembleReport, SdeError, SdeNumericalError,
    SdePaths, SdeRetention, SdeScheme, SdeSimulationPolicy, SdeSystem, SdeWorld, SdeWorldEnsemble,
    WorldPlan, action_cell, bad, check_cancel, counter_normal, unsupported,
};

struct Levels {
    coarse: Vec<f64>,
    fine: Option<Vec<f64>>,
}

/// One path at one step level, writing the recorded states into `out`.
#[allow(clippy::too_many_arguments, clippy::too_many_lines, reason = "one scheme step loop")]
pub(super) fn run_path(
    compiled: &CompiledSde,
    theta: &[f64],
    world: &WorldPlan,
    policy: &SdeSimulationPolicy,
    initial: &[f64],
    path: u64,
    fine: bool,
    out: &mut [f64],
    ctx: &ExecutionContext,
) -> Result<(), SdeError> {
    let d = compiled.dimension();
    let k = world.inputs.len();
    let m = if fine { 2usize } else { 1usize };
    let dt = policy.step / m as f64;
    let cells: Vec<Vec<f64>> = world
        .actions
        .iter()
        .map(|action| {
            action.times.iter().map(|&time| action_cell(time, policy.start, policy.step)).collect()
        })
        .collect::<Result<_, _>>()?;
    let record_every = policy.output_every * m;
    let total = policy.steps * m;
    let mut x = initial.to_vec();
    let mut vars = vec![0.0; d + k];
    let mut next = vec![0.0; d];
    let apply_actions = |x: &mut [f64], step_index: usize| {
        for (action, cells) in world.actions.iter().zip(&cells) {
            let count = cells.partition_point(|&cell| cell * m as f64 <= step_index as f64);
            if count > 0 {
                x[action.variable] = action.values[count - 1];
            }
        }
    };
    apply_actions(&mut x, 0);
    out[..d].copy_from_slice(&x);
    let mut slot = 1usize;
    let sqrt_half = (policy.step / 2.0).sqrt();
    for n in 0..total {
        if n % CANCEL_STRIDE == 0 {
            check_cancel(ctx)?;
        }
        let t = policy.start + n as f64 * dt;
        vars[..d].copy_from_slice(&x);
        for (j, input) in world.inputs.iter().enumerate() {
            vars[d + j] = input.value_at(t);
        }
        for i in 0..d {
            let held = world
                .actions
                .iter()
                .zip(&cells)
                .any(|(action, cells)| action.variable == i && cells[0] * m as f64 <= n as f64);
            if held {
                next[i] = x[i];
                continue;
            }
            let f = compiled.drift[i].evaluate(&vars, theta)?.value();
            let ge = compiled.diffusion[i].evaluate(&vars, theta)?;
            let sqrt = compiled.system.diffusion[i].square_root;
            let h = ge.value();
            let g = if sqrt { h.max(0.0).sqrt() } else { h };
            let dw = if fine {
                sqrt_half * counter_normal(policy.seed, path, n as u64, i as u64)
            } else {
                sqrt_half
                    * (counter_normal(policy.seed, path, 2 * n as u64, i as u64)
                        + counter_normal(policy.seed, path, 2 * n as u64 + 1, i as u64))
            };
            let mut value = x[i] + f * dt + g * dw;
            if matches!(policy.scheme, SdeScheme::Milstein) {
                let dh = ge.gradient()[i];
                let correction =
                    if sqrt { if h > 0.0 { 0.25 * dh } else { 0.0 } } else { 0.5 * g * dh };
                value += correction * (dw * dw - dt);
            }
            next[i] = value;
        }
        for i in 0..d {
            x[i] = compiled.system.boundary(i).apply(next[i]);
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(SdeError::Numerical(format!(
                "state became non-finite at step {n} of path {path}"
            )));
        }
        apply_actions(&mut x, n + 1);
        if (n + 1) % record_every == 0 {
            out[slot * d..(slot + 1) * d].copy_from_slice(&x);
            slot += 1;
        }
    }
    Ok(())
}

fn run_world(
    compiled: &CompiledSde,
    theta: &[f64],
    world: &WorldPlan,
    policy: &SdeSimulationPolicy,
    initial: &[f64],
    ctx: &ExecutionContext,
) -> Result<Levels, SdeError> {
    let d = compiled.dimension();
    let n_out = policy.steps / policy.output_every + 1;
    let cells = policy.paths * n_out * d;
    let mut coarse = vec![0.0; cells];
    let mut fine = policy.check_step.then(|| vec![0.0; cells]);
    for p in 0..policy.paths {
        check_cancel(ctx)?;
        let range = p * n_out * d..(p + 1) * n_out * d;
        run_path(
            compiled,
            theta,
            world,
            policy,
            initial,
            p as u64,
            false,
            &mut coarse[range.clone()],
            ctx,
        )?;
        if let Some(fine) = fine.as_mut() {
            run_path(
                compiled,
                theta,
                world,
                policy,
                initial,
                p as u64,
                true,
                &mut fine[range],
                ctx,
            )?;
        }
    }
    Ok(Levels { coarse, fine })
}

fn validate_initial(compiled: &CompiledSde, initial: &[f64]) -> Result<(), SdeError> {
    let d = compiled.dimension();
    if initial.len() != d
        || initial.iter().any(|v| !v.is_finite())
        || !(0..d).all(|i| compiled.system.boundary(i).contains(initial[i]))
    {
        return Err(bad("SDE initial state (dimension, finiteness, declared intervals)"));
    }
    Ok(())
}

fn resolve_theta(compiled: &CompiledSde, theta: Option<&[f64]>) -> Result<Vec<f64>, SdeError> {
    let theta = theta.map_or_else(|| compiled.system.parameter_values(), <[f64]>::to_vec);
    if theta.len() != compiled.system.parameters.len() || theta.iter().any(|v| !v.is_finite()) {
        return Err(bad("SDE parameter vector"));
    }
    Ok(theta)
}

// The regularity certificate is parameter-specific. Recompile only when an
// override differs from the certified defaults, before running either route.
#[allow(
    clippy::float_cmp,
    reason = "exact equality is required for discrete support or floating point progress"
)]
fn effective_compiled<'a>(
    compiled: &'a CompiledSde,
    theta: &[f64],
    ctx: &ExecutionContext,
) -> Result<(std::borrow::Cow<'a, CompiledSde>, Option<AdmissionLease>), SdeError> {
    if compiled.system.parameters.iter().zip(theta).all(|(p, value)| p.value == *value) {
        return Ok((std::borrow::Cow::Borrowed(compiled), None));
    }
    check_cancel(ctx)?;
    // The recertified system lives as long as the run, so its bytes are held that long.
    let lease = ctx
        .acquire_admission(0, compiled.system.recertification_bytes()?)
        .map_err(|e| SdeError::Resource(e.to_string()))?;
    let mut system = compiled.system.clone();
    for (parameter, value) in system.parameters.iter_mut().zip(theta) {
        parameter.value = *value;
    }
    Ok((std::borrow::Cow::Owned(system.compile()?), Some(lease)))
}

// Admission covers retained path arrays and, when requested, the simultaneous
// report arrays, their row headers, final samples, and report metadata. The
// existing cell and work limits apply to the combined execution and reporting. The
// bytes the returned output keeps are leased apart from the run's own, so the output
// can carry its lease after the run returns: the report arrays of a report run, the
// coarse path array of a path run. Returns the run's own lease, then the output's.
#[allow(clippy::too_many_arguments, reason = "one checked admission for both output routes")]
pub(super) fn admit(
    compiled: &CompiledSde,
    policy: &SdeSimulationPolicy,
    worlds: usize,
    contrasts: usize,
    reports: bool,
    name_bytes: usize,
    ctx: &ExecutionContext,
) -> Result<(AdmissionLease, AdmissionLease), SdeError> {
    let d = compiled.dimension();
    let n_out = policy.steps / policy.output_every + 1;
    let levels = if policy.check_step { 2usize } else { 1usize };
    let overflow = || bad("SDE admitted size overflow");
    let mul = |a: usize, b: usize| a.checked_mul(b).ok_or_else(overflow);
    let add = |a: usize, b: usize| a.checked_add(b).ok_or_else(overflow);
    let path_cells = mul(mul(mul(mul(worlds, policy.paths)?, n_out)?, d)?, levels)?;
    let mut bytes = add(add(mul(path_cells, 8)?, mul(n_out, 8)?)?, 4096)?;
    let path_bytes = bytes;
    let mut report_work = 0usize;
    if reports {
        let world_rows = 4 + if policy.check_step { 4 } else { 0 };
        let contrast_rows = 2 + if policy.check_step { 4 } else { 0 };
        let world_cells = mul(mul(worlds, n_out)?, add(mul(d, d)?, mul(world_rows - 1, d)?)?)?;
        let contrast_cells = mul(mul(mul(contrasts, n_out)?, contrast_rows)?, d)?;
        let final_cells =
            if policy.retain_final_samples { mul(mul(worlds, policy.paths)?, d)? } else { 0 };
        let headers = add(
            mul(mul(worlds, n_out)?, world_rows)?,
            mul(mul(contrasts, n_out)?, contrast_rows)?,
        )?;
        let final_headers =
            if policy.retain_final_samples { mul(worlds, policy.paths)? } else { 0 };
        bytes = add(bytes, mul(add(add(world_cells, contrast_cells)?, final_cells)?, 8)?)?;
        bytes = add(bytes, mul(add(headers, final_headers)?, std::mem::size_of::<Vec<f64>>())?)?;
        bytes = add(bytes, mul(worlds, std::mem::size_of::<SdeWorldEnsemble>())?)?;
        bytes = add(bytes, mul(contrasts, std::mem::size_of::<SdeContrastEnsemble>())?)?;
        bytes = add(bytes, name_bytes)?;
        // Conservatively bound reductions over paths for covariance and each
        // of the ordinary and step-halving report coordinates.
        let world_ops = add(mul(d, d)?, mul(if policy.check_step { 16 } else { 3 }, d)?)?;
        let contrast_ops = mul(if policy.check_step { 16 } else { 2 }, d)?;
        report_work = mul(
            mul(n_out, policy.paths)?,
            add(mul(worlds, world_ops)?, mul(contrasts, contrast_ops)?)?,
        )?;
    }
    if bytes > MAX_SDE_RETAINED_CELLS.saturating_mul(8) {
        return Err(SdeError::Resource(format!(
            "SDE paths and reports need {bytes} bytes, above the retained-cell limit"
        )));
    }
    let path_work =
        mul(mul(mul(worlds, policy.paths)?, policy.steps)?, if policy.check_step { 3 } else { 1 })?;
    let work = add(path_work, report_work)?;
    if work as u128 > u128::from(MAX_SDE_STEP_EVALUATIONS) {
        return Err(SdeError::Resource(format!(
            "SDE simulation and report need {work} work units, above the limit {MAX_SDE_STEP_EVALUATIONS}"
        )));
    }
    let lease = |bytes: usize| {
        ctx.acquire_admission(0, u64::try_from(bytes).map_err(|_| overflow())?)
            .map_err(|e| SdeError::Resource(e.to_string()))
    };
    let (transient, retained) = if reports {
        (path_bytes, bytes - path_bytes)
    } else {
        let coarse = mul(mul(mul(worlds, policy.paths)?, n_out)?, d)?;
        let kept = add(add(mul(coarse, 8)?, mul(n_out, 8)?)?, 4096)?;
        (bytes - kept, kept)
    };
    Ok((lease(transient)?, lease(retained)?))
}

/// Admit the plans of `worlds` before any is built, and hold the lease while they live.
fn admit_world_plans<'a>(
    system: &SdeSystem,
    worlds: impl Iterator<Item = &'a SdeWorld>,
    ctx: &ExecutionContext,
) -> Result<AdmissionLease, SdeError> {
    let bytes =
        worlds.fold(0u64, |total, world| total.saturating_add(system.world_plan_bytes(world)));
    ctx.acquire_admission(0, bytes).map_err(|e| SdeError::Resource(e.to_string()))
}

pub(super) fn mean_and_se(
    values: impl Iterator<Item = f64>,
    n: usize,
    ctx: &ExecutionContext,
) -> Result<(f64, f64, f64), SdeError> {
    // Two-pass would need storage; Welford keeps the single pass stable.
    let (mut mean, mut m2, mut count) = (0.0, 0.0, 0.0);
    for (index, v) in values.enumerate() {
        if index % CANCEL_STRIDE == 0 {
            check_cancel(ctx)?;
        }
        count += 1.0;
        let delta = v - mean;
        mean += delta / count;
        m2 += delta * (v - mean);
    }
    let variance = if n > 1 { m2 / (count - 1.0) } else { 0.0 };
    Ok((mean, variance, (variance / count).sqrt()))
}

/// Sum in iteration order, as [`Iterator::sum`] does, polling cancellation along the way.
pub(super) fn sum_polled(
    values: impl Iterator<Item = f64>,
    ctx: &ExecutionContext,
) -> Result<f64, SdeError> {
    let mut total = -0.0;
    for (index, v) in values.enumerate() {
        if index % CANCEL_STRIDE == 0 {
            check_cancel(ctx)?;
        }
        total += v;
    }
    Ok(total)
}

#[allow(clippy::too_many_arguments, reason = "the inputs describe one scientific operation")]
pub(super) fn numerical_error(
    coarse: impl Fn(usize, usize, usize) -> f64,
    fine: impl Fn(usize, usize, usize) -> f64,
    paths: usize,
    n_out: usize,
    d: usize,
    step: f64,
    tolerance: f64,
    ctx: &ExecutionContext,
) -> Result<SdeNumericalError, SdeError> {
    let mut weak = vec![vec![0.0; d]; n_out];
    let mut weak_se = vec![vec![0.0; d]; n_out];
    let mut strong = vec![vec![0.0; d]; n_out];
    let mut richardson = vec![vec![0.0; d]; n_out];
    let mut converged = true;
    for t in 0..n_out {
        check_cancel(ctx)?;
        for i in 0..d {
            check_cancel(ctx)?;
            let (mean_diff, _, se) =
                mean_and_se((0..paths).map(|p| coarse(p, t, i) - fine(p, t, i)), paths, ctx)?;
            let (mean_fine, var_fine, _) =
                mean_and_se((0..paths).map(|p| fine(p, t, i)), paths, ctx)?;
            let rms = (sum_polled(
                (0..paths).map(|p| {
                    let e = coarse(p, t, i) - fine(p, t, i);
                    e * e
                }),
                ctx,
            )? / paths as f64)
                .sqrt();
            let scale = mean_fine.abs() + var_fine.sqrt() + 1e-12;
            if mean_diff.abs() - 2.0 * se > tolerance * scale {
                converged = false;
            }
            weak[t][i] = mean_diff;
            weak_se[t][i] = se;
            strong[t][i] = rms;
            richardson[t][i] = 2.0 * mean_fine - (mean_fine + mean_diff);
        }
    }
    Ok(SdeNumericalError {
        refined_step: step / 2.0,
        weak_difference: weak,
        weak_difference_standard_error: weak_se,
        strong_rms_difference: strong,
        richardson_mean: richardson,
        step_converged: converged,
    })
}

/// Simulate every world at the declared step, with paired contrasts and step halving.
///
/// `worlds` need not contain the baseline: it is always simulated first, as
/// [`BASELINE_WORLD`], and a user world may not reuse that identity.
///
/// # Errors
/// `Invalid`, `Unsupported` (Milstein with state-coupled diffusion, off-grid actions,
/// out-of-envelope inputs), `Numerical` for non-finite states and `Resource` for budget or
/// cancellation.
#[allow(clippy::too_many_lines, reason = "one ensemble: plan, run, then three kinds of moments")]
pub fn simulate_sde(
    compiled: &CompiledSde,
    theta: Option<&[f64]>,
    initial: &[f64],
    policy: &SdeSimulationPolicy,
    worlds: &[SdeWorld],
    contrasts: &[SdeContrast],
    ctx: &ExecutionContext,
) -> Result<SdeEnsembleReport, SdeError> {
    policy.validate()?;
    validate_initial(compiled, initial)?;
    let theta = resolve_theta(compiled, theta)?;
    let (effective, _recertified) = effective_compiled(compiled, &theta, ctx)?;
    let compiled = effective.as_ref();
    if matches!(policy.scheme, SdeScheme::Milstein)
        && !compiled.certificate.diagonal_milstein_admissible
    {
        return Err(unsupported(
            antecedent_core::reason_details::sde::MILSTEIN_NONCOMMUTATIVE,
            "a diffusion depends on another state, so the diagonal Milstein correction omits \
             Levy-area terms; use Euler-Maruyama or make each diffusion depend only on its own state",
        ));
    }
    if worlds.len() + 1 > MAX_WORLDS || contrasts.len() > MAX_WORLDS * MAX_WORLDS {
        return Err(bad("SDE world/contrast count"));
    }
    let baseline = SdeWorld { id: BASELINE_WORLD.into(), ..SdeWorld::default() };
    let _plan_lease =
        admit_world_plans(&compiled.system, std::iter::once(&baseline).chain(worlds), ctx)?;
    let mut plans = vec![compiled.system.plan_world(&baseline, policy.start, policy.step)?];
    let mut ids = std::collections::BTreeSet::new();
    ids.insert(BASELINE_WORLD.to_owned());
    for world in worlds {
        if !ids.insert(world.id.clone()) {
            return Err(bad("SDE world identities must be distinct and not 'baseline'"));
        }
        plans.push(compiled.system.plan_world(world, policy.start, policy.step)?);
    }
    let mut contrast_slots = Vec::with_capacity(contrasts.len());
    let mut contrast_ids = std::collections::BTreeSet::new();
    for c in contrasts {
        let find = |id: &str| plans.iter().position(|p| p.id == id);
        let (Some(t), Some(r)) = (find(&c.treated), find(&c.control)) else {
            return Err(bad("SDE contrast references an undeclared world"));
        };
        if c.id.trim().is_empty() || !contrast_ids.insert(c.id.clone()) {
            return Err(bad("SDE contrast identity"));
        }
        contrast_slots.push((t, r));
    }
    let name_bytes = plans
        .iter()
        .try_fold(0usize, |total, p| total.checked_add(p.id.len()))
        .and_then(|n| {
            contrasts.iter().try_fold(n, |total, c| {
                total
                    .checked_add(c.id.len())?
                    .checked_add(c.treated.len())?
                    .checked_add(c.control.len())
            })
        })
        .ok_or_else(|| bad("SDE report name size overflow"))?;
    let (_run, retained) =
        admit(compiled, policy, plans.len(), contrasts.len(), true, name_bytes, ctx)?;
    let d = compiled.dimension();
    let n_out = policy.steps / policy.output_every + 1;
    let paths = policy.paths;
    let times = policy.output_times();
    let mut runs = Vec::with_capacity(plans.len());
    for plan in &plans {
        runs.push(run_world(compiled, &theta, plan, policy, initial, ctx)?);
    }
    let at = |levels: &[f64], p: usize, t: usize, i: usize| levels[(p * n_out + t) * d + i];
    let mut world_reports = Vec::with_capacity(plans.len());
    for (plan, run) in plans.iter().zip(&runs) {
        let mut mean = vec![vec![0.0; d]; n_out];
        let mut variance = vec![vec![0.0; d]; n_out];
        let mut mc = vec![vec![0.0; d]; n_out];
        let mut covariance = vec![vec![0.0; d * d]; n_out];
        for t in 0..n_out {
            check_cancel(ctx)?;
            for i in 0..d {
                check_cancel(ctx)?;
                let (m, v, se) =
                    mean_and_se((0..paths).map(|p| at(&run.coarse, p, t, i)), paths, ctx)?;
                mean[t][i] = m;
                variance[t][i] = v;
                mc[t][i] = se;
            }
            for i in 0..d {
                for j in 0..=i {
                    check_cancel(ctx)?;
                    let c = sum_polled(
                        (0..paths).map(|p| {
                            (at(&run.coarse, p, t, i) - mean[t][i])
                                * (at(&run.coarse, p, t, j) - mean[t][j])
                        }),
                        ctx,
                    )? / (paths as f64 - 1.0);
                    covariance[t][i * d + j] = c;
                    covariance[t][j * d + i] = c;
                }
            }
        }
        let numerical = run
            .fine
            .as_ref()
            .map(|fine| {
                numerical_error(
                    |p, t, i| at(&run.coarse, p, t, i),
                    |p, t, i| at(fine, p, t, i),
                    paths,
                    n_out,
                    d,
                    policy.step,
                    policy.weak_tolerance,
                    ctx,
                )
            })
            .transpose()?;
        let final_samples = if policy.retain_final_samples {
            let mut samples = Vec::with_capacity(paths);
            for p in 0..paths {
                if p % 1024 == 0 {
                    check_cancel(ctx)?;
                }
                samples.push((0..d).map(|i| at(&run.coarse, p, n_out - 1, i)).collect());
            }
            Some(samples)
        } else {
            None
        };
        world_reports.push(SdeWorldEnsemble {
            id: plan.id.clone(),
            mean,
            variance,
            covariance,
            monte_carlo_standard_error: mc,
            numerical,
            final_samples,
        });
    }
    let mut contrast_reports = Vec::with_capacity(contrasts.len());
    for (c, &(ti, ci)) in contrasts.iter().zip(&contrast_slots) {
        let (treated, control) = (&runs[ti], &runs[ci]);
        let mut mean_difference = vec![vec![0.0; d]; n_out];
        let mut se = vec![vec![0.0; d]; n_out];
        for t in 0..n_out {
            check_cancel(ctx)?;
            for i in 0..d {
                check_cancel(ctx)?;
                let (m, _, s) = mean_and_se(
                    (0..paths).map(|p| at(&treated.coarse, p, t, i) - at(&control.coarse, p, t, i)),
                    paths,
                    ctx,
                )?;
                mean_difference[t][i] = m;
                se[t][i] = s;
            }
        }
        let numerical = match (&treated.fine, &control.fine) {
            (Some(tf), Some(cf)) => Some(numerical_error(
                |p, t, i| at(&treated.coarse, p, t, i) - at(&control.coarse, p, t, i),
                |p, t, i| at(tf, p, t, i) - at(cf, p, t, i),
                paths,
                n_out,
                d,
                policy.step,
                policy.weak_tolerance,
                ctx,
            )?),
            _ => None,
        };
        contrast_reports.push(SdeContrastEnsemble {
            id: c.id.clone(),
            treated: c.treated.clone(),
            control: c.control.clone(),
            mean_difference,
            monte_carlo_standard_error: se,
            numerical,
        });
    }
    Ok(SdeEnsembleReport {
        scheme: policy.scheme,
        step: policy.step,
        seed: policy.seed,
        paths,
        times,
        parameters: theta,
        worlds: world_reports,
        contrasts: contrast_reports,
        certificate: compiled.certificate.clone(),
        error_semantics: vec![
            "monte_carlo_standard_error is sampling error of a finite ensemble at the stated parameters; it shrinks with paths".into(),
            "numerical.weak_difference is the observed effect of halving the step on the mean, paired by path; it estimates discretization bias and shrinks with the step".into(),
            "neither field is statistical uncertainty of fitted parameters, which the estimation layer reports separately".into(),
            "paired contrasts share every Brownian increment across worlds, so their Monte Carlo standard error is that of the paired difference".into(),
        ],
        retention: SdeRetention(Some(Arc::new(retained))),
    })
}

/// Raw declared-step paths of one world, for data generation and downstream functionals.
///
/// # Errors
/// As [`simulate_sde`].
pub fn sample_sde_paths(
    compiled: &CompiledSde,
    theta: Option<&[f64]>,
    initial: &[f64],
    policy: &SdeSimulationPolicy,
    world: &SdeWorld,
    ctx: &ExecutionContext,
) -> Result<SdePaths, SdeError> {
    policy.validate()?;
    validate_initial(compiled, initial)?;
    let theta = resolve_theta(compiled, theta)?;
    let (effective, _recertified) = effective_compiled(compiled, &theta, ctx)?;
    let compiled = effective.as_ref();
    if matches!(policy.scheme, SdeScheme::Milstein)
        && !compiled.certificate.diagonal_milstein_admissible
    {
        return Err(unsupported(
            antecedent_core::reason_details::sde::MILSTEIN_NONCOMMUTATIVE,
            "a diffusion depends on another state; the diagonal Milstein correction is not exact",
        ));
    }
    let _plan_lease = admit_world_plans(&compiled.system, std::iter::once(world), ctx)?;
    let plan = compiled.system.plan_world(world, policy.start, policy.step)?;
    let mut single = policy.clone();
    single.check_step = false;
    let (_run, retained) = admit(compiled, &single, 1, 0, false, 0, ctx)?;
    let run = run_world(compiled, &theta, &plan, &single, initial, ctx)?;
    Ok(SdePaths {
        times: policy.output_times(),
        paths: policy.paths,
        dimension: compiled.dimension(),
        states: run.coarse,
        retention: SdeRetention(Some(Arc::new(retained))),
    })
}
