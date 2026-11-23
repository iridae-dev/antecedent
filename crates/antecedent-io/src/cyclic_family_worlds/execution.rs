use super::{
    A, Arc, CertificateDemandWire, Clamp, CompiledSystemPlan, ContinuousTimeSpecification,
    CyclicAction, CyclicArtifactError, CyclicComponentWire, CyclicFamily, CyclicVariables,
    CyclicWorldRequest, CyclicWorldResult, CyclicWorldWire, EquilibriumModel, ExecutionContext,
    GaussianDisturbance, HardClamp, LinearCyclicSpecification, LinearCyclicWire, LinearRouteSet,
    LinearStructuralMap, MAX_DRAWS, MAX_REQUEST_WORLD_WORK, NonlinearCyclicSpecification,
    OdePolicy, OdeSystem, SolvePolicy, admit_worlds, budget, compile_equilibria, compile_ode,
    evaluate_continuous_time_world, evaluate_linear_worlds, evaluate_linear_worlds_routed,
    evaluate_nonlinear_world, gaussian_equilibrium_peak_bytes, invalid, refused,
    resolve_cyclic_world, variables_of, world_answer, world_requests,
};

/// Hard-clamp positions and levels of every world, after admission on `family`.
fn admitted_clamps(
    family: CyclicFamily,
    variables: &CyclicVariables,
    initial_laws: &[Arc<str>],
    requests: &[CyclicWorldRequest],
) -> Result<Vec<Vec<(usize, f64)>>, CyclicArtifactError> {
    requests
        .iter()
        .map(|request| {
            let resolved =
                resolve_cyclic_world(family, variables, initial_laws, request).map_err(refused)?;
            Ok(resolved
                .actions()
                .iter()
                .filter_map(|action| match action {
                    CyclicAction::HardClamp { node, level, .. } => Some((*node, *level)),
                    _ => None,
                })
                .collect())
        })
        .collect()
}

/// The execution plan of every linear world, and the support routes they were built from.
///
/// The routes are the one computation of this request's routing: the program keeps them, and
/// the executor compiles each world from the held route when it matches the world's map and
/// clamp set exactly, so routing is charged here and nowhere else.
pub(in crate::cyclic_family_artifact) fn linear_world_plans(
    variables: &[u32],
    model: &LinearCyclicWire,
    worlds: &[CyclicWorldWire],
) -> Result<(Vec<CompiledSystemPlan>, LinearRouteSet), CyclicArtifactError> {
    let bound = variables_of(variables)?;
    let requests = world_requests(worlds)?;
    let map = LinearStructuralMap::try_new(model.coefficients.clone(), model.intercept.clone())
        .map_err(invalid)?;
    let clamp_sets = admitted_clamps(CyclicFamily::LinearEquilibrium, &bound, &[], &requests)?;
    // Routing reads the whole coefficient matrix and depends only on which nodes are clamped,
    // so worlds that clamp the same nodes share one route, and the routes of all worlds are
    // charged together against the request's work limit.
    let route_work = antecedent_model::scc_compile::gaussian_equilibrium_route_work(bound.len())
        .map_err(invalid)?;
    let mut routes = LinearRouteSet::new(&map);
    let mut charged = 0_u64;
    let mut plans = Vec::with_capacity(clamp_sets.len());
    for clamps in clamp_sets {
        let nodes: Vec<usize> = clamps.iter().map(|(node, _)| *node).collect();
        let clamp = HardClamp::try_new(bound.len(), &nodes).map_err(invalid)?;
        if let Some(route) = routes.get(&map, &clamp) {
            plans.push(super::super::compiled::plan_from_route(route));
            continue;
        }
        charged = charged.saturating_add(route_work);
        if charged > MAX_REQUEST_WORLD_WORK {
            return Err(invalid(format!(
                "{}{}: cyclic_worlds.work_limit: routing {} distinct clamp sets of a \
                 {}-variable model exceeds the request work limit of {MAX_REQUEST_WORLD_WORK}. \
                 Next attempt: fewer distinct clamp sets",
                antecedent_core::reason_code::PREFIX,
                antecedent_core::reason_code!("execution_resource_limit"),
                routes.len() + 1,
                bound.len(),
            )));
        }
        let (route, _) = routes.route(&map, &clamp).map_err(invalid)?;
        plans.push(super::super::compiled::plan_from_route(&route));
    }
    Ok((plans, routes))
}

pub(in crate::cyclic_family_artifact) fn equilibrium_world_plans(
    variables: &[u32],
    model: &EquilibriumModel,
    worlds: &[CyclicWorldWire],
) -> Result<Vec<CompiledSystemPlan>, CyclicArtifactError> {
    model.validate()?;
    let bound = variables_of(variables)?;
    let requests = world_requests(worlds)?;
    let clamp_sets: Vec<Vec<Clamp>> =
        admitted_clamps(CyclicFamily::NonlinearEquilibrium, &bound, &[], &requests)?
            .into_iter()
            .map(|clamps| {
                clamps.into_iter().map(|(variable, value)| Clamp { variable, value }).collect()
            })
            .collect();
    let borrowed: Vec<&[Clamp]> = clamp_sets.iter().map(Vec::as_slice).collect();
    compile_equilibria(model, &borrowed).map_err(CyclicArtifactError::from)
}

pub(in crate::cyclic_family_artifact) fn ode_world_plans(
    variables: &[u32],
    model: &OdeSystem,
    initial_laws: &[String],
    worlds: &[CyclicWorldWire],
) -> Result<Vec<CompiledSystemPlan>, CyclicArtifactError> {
    let bound = variables_of(variables)?;
    let requests = world_requests(worlds)?;
    let laws: Vec<Arc<str>> = initial_laws.iter().map(|law| Arc::from(law.as_str())).collect();
    admitted_clamps(CyclicFamily::ContinuousTime, &bound, &laws, &requests)?;
    Ok(vec![compile_ode(model)?])
}

pub(in crate::cyclic_family_artifact) fn execute_linear_worlds(
    variables: &[u32],
    model: &LinearCyclicWire,
    component: &CyclicComponentWire,
    worlds: &[CyclicWorldWire],
    certificate: CertificateDemandWire,
    routes: Option<&LinearRouteSet>,
    ctx: &ExecutionContext,
) -> Result<A, CyclicArtifactError> {
    let bound = variables_of(variables)?;
    let map = LinearStructuralMap::try_new(model.coefficients.clone(), model.intercept.clone())
        .map_err(invalid)?;
    let disturbance =
        GaussianDisturbance::try_new(model.noise_mean.clone(), model.noise_covariance.clone())
            .map_err(invalid)?;
    let evidence: Vec<&str> = component.certifying_evidence.iter().map(String::as_str).collect();
    let spec = LinearCyclicSpecification::try_new(
        bound,
        map,
        disturbance,
        &component.component_id,
        &component.source_id,
        &evidence,
    )
    .map_err(refused)?;
    let requests = world_requests(worlds)?;
    if ctx.cancellation.is_cancelled() {
        return Err(invalid("cyclic world evaluation cancelled"));
    }
    let order = variables.len();
    let _lease = admit_worlds(
        ctx,
        gaussian_equilibrium_peak_bytes(order).map_err(invalid)?,
        CyclicWorldResult::retained_gaussian_estimate(order),
        requests.len(),
    )?;
    let (demand, budget) = (certificate.engine(), budget(ctx));
    let results = match routes {
        Some(routes) => evaluate_linear_worlds_routed(&spec, &requests, demand, budget, routes),
        None => evaluate_linear_worlds(&spec, &requests, demand, budget),
    }
    .map_err(refused)?;
    let answers = requests
        .iter()
        .map(|request| {
            let result = results
                .get(request.world())
                .ok_or_else(|| invalid("an evaluated world is missing"))?;
            world_answer(result)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(A::Worlds(answers))
}

pub(in crate::cyclic_family_artifact) fn execute_equilibrium_worlds(
    variables: &[u32],
    model: &EquilibriumModel,
    worlds: &[CyclicWorldWire],
    noise: &[Vec<f64>],
    initial: &[f64],
    policy: SolvePolicy,
    ctx: &ExecutionContext,
) -> Result<A, CyclicArtifactError> {
    let spec = NonlinearCyclicSpecification::try_new(variables_of(variables)?, model.clone())
        .map_err(refused)?;
    let n = model.order();
    if noise.is_empty()
        || noise.len() > MAX_DRAWS
        || noise.iter().any(|row| row.len() != n || row.iter().any(|value| !value.is_finite()))
        || initial.len() != n
        || noise.len().saturating_mul(worlds.len()).saturating_mul(n) > 1_000_000
    {
        return Err(invalid(
            "equilibrium worlds need finite disturbance rows of the model's order and an \
             initial guess of that order",
        ));
    }
    let flat = noise.concat();
    let requests = world_requests(worlds)?;
    let _lease = admit_worlds(
        ctx,
        0,
        CyclicWorldResult::retained_draws_estimate(n, noise.len()),
        requests.len(),
    )?;
    let mut answers = Vec::with_capacity(requests.len());
    for request in &requests {
        if ctx.cancellation.is_cancelled() {
            return Err(invalid("cyclic world evaluation cancelled"));
        }
        let result = evaluate_nonlinear_world(&spec, request, &flat, initial, policy, ctx)
            .map_err(refused)?;
        answers.push(world_answer(&result)?);
    }
    Ok(A::Worlds(answers))
}

#[allow(clippy::too_many_arguments, reason = "each argument is one field of the typed request")]
pub(in crate::cyclic_family_artifact) fn execute_ode_worlds(
    variables: &[u32],
    model: &OdeSystem,
    initial: &[f64],
    initial_laws: &[String],
    times: &[f64],
    worlds: &[CyclicWorldWire],
    policy: OdePolicy,
    ctx: &ExecutionContext,
) -> Result<A, CyclicArtifactError> {
    let laws: Vec<Arc<str>> = initial_laws.iter().map(|law| Arc::from(law.as_str())).collect();
    let spec = ContinuousTimeSpecification::try_new(
        variables_of(variables)?,
        model.clone(),
        initial.to_vec(),
        laws,
        times.to_vec(),
    )
    .map_err(refused)?;
    let requests = world_requests(worlds)?;
    let _lease = admit_worlds(
        ctx,
        0,
        CyclicWorldResult::retained_trajectory_estimate(variables.len(), times.len()),
        requests.len(),
    )?;
    let mut answers = Vec::with_capacity(requests.len());
    for request in &requests {
        if ctx.cancellation.is_cancelled() {
            return Err(invalid("cyclic world evaluation cancelled"));
        }
        let result =
            evaluate_continuous_time_world(&spec, request, policy, ctx).map_err(refused)?;
        answers.push(world_answer(&result)?);
    }
    Ok(A::Worlds(answers))
}
