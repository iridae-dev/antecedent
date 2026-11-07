use super::{
    CERTIFYING_EVIDENCE_ATTEMPT, CertificateBasis, CertificateGap, CompileBudget, CyclicAction,
    CyclicComponent, CyclicFamily, CyclicInterventionRefusal, CyclicRefusalKind, CyclicVariables,
    CyclicWorldOutcome, CyclicWorldRequest, CyclicWorldResult, CyclicWorldResults,
    GaussianDisturbance, HardClamp, InterventionRequest, InterventionRequestError, LinearRouteSet,
    LinearStructuralMap, LinearSupportRoute, MechanismReplacement,
    QUERY_SPECIFIC_CERTIFICATE_ATTEMPT, ResolvedCyclicWorld, SccCompileError, ScientificWorldId,
    SolvabilityRefusal, WorldCertificate, WorldFailure, WorldMass, WorldParts, WorldSolvedBy,
    assess, compile_routed_with, duplicate_world, execution_refusal, free_indices,
    gauss_jordan_solve, gaussian_equilibrium_admission, gaussian_equilibrium_route_work,
    hard_clamp_of, refusal, resolve_cyclic_world, revalidate_solvability, specification_refusal,
};

/// Whether an uncertified linear world is answered.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum CertificateDemand {
    /// Withhold a world whose certificate was not shown.
    Required,
    /// Answer a uniquely solvable world and mark it uncertified.
    ReportUncertified,
}

/// Specified linear Gaussian equilibrium with bound variables and the
/// evidence an admitted component cites.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearCyclicSpecification {
    variables: CyclicVariables,
    model: LinearStructuralMap,
    disturbance: GaussianDisturbance,
    component_id: String,
    source_id: String,
    certifying_evidence: Vec<String>,
}

impl LinearCyclicSpecification {
    /// Bind `model` and `disturbance` to `variables`.
    ///
    /// # Errors
    ///
    /// Dimensions that disagree, or a component id, source id, or evidence list
    /// the component gate would refuse.
    pub fn try_new(
        variables: CyclicVariables,
        model: LinearStructuralMap,
        disturbance: GaussianDisturbance,
        component_id: &str,
        source_id: &str,
        certifying_evidence: &[&str],
    ) -> Result<Self, CyclicInterventionRefusal> {
        let family = CyclicFamily::LinearEquilibrium;
        if model.dimension() != variables.len() || disturbance.dimension() != variables.len() {
            return Err(specification_refusal(
                family,
                None,
                "the map, the disturbance, and the variables have different orders",
            ));
        }
        let evidence_refused = component_id.is_empty()
            || source_id.is_empty()
            || certifying_evidence.is_empty()
            || certifying_evidence.iter().any(|cited| cited.is_empty() || *cited == source_id);
        if evidence_refused {
            return Err(refusal(
                None,
                Some(family),
                "specification",
                CyclicRefusalKind::Evidence {
                    message: "the component needs an id, a source, and certifying evidence other \
                              than the source"
                        .to_owned(),
                },
                CERTIFYING_EVIDENCE_ATTEMPT,
            ));
        }
        Ok(Self {
            variables,
            model,
            disturbance,
            component_id: component_id.to_owned(),
            source_id: source_id.to_owned(),
            certifying_evidence: certifying_evidence
                .iter()
                .map(|cited| (*cited).to_owned())
                .collect(),
        })
    }

    /// Bound variables.
    #[must_use]
    pub const fn variables(&self) -> &CyclicVariables {
        &self.variables
    }

    /// Structural map.
    #[must_use]
    pub const fn model(&self) -> &LinearStructuralMap {
        &self.model
    }

    /// Disturbance law.
    #[must_use]
    pub const fn disturbance(&self) -> &GaussianDisturbance {
        &self.disturbance
    }
}

/// Intervened map, disturbance and clamp of one linear world.
struct LinearWorldSystem {
    model: LinearStructuralMap,
    disturbance: GaussianDisturbance,
    clamp: InterventionRequest,
}

fn linear_world_system(
    spec: &LinearCyclicSpecification,
    resolved: &ResolvedCyclicWorld,
) -> Result<LinearWorldSystem, CyclicInterventionRefusal> {
    let family = CyclicFamily::LinearEquilibrium;
    let world = Some(resolved.world);
    let n = spec.model.dimension();
    let mut coefficients = spec.model.coefficients().to_vec();
    let mut intercept = spec.model.intercept().to_vec();
    let mut mean = spec.disturbance.mean().to_vec();
    let (mut nodes, mut levels) = (Vec::new(), Vec::new());
    for action in &resolved.actions {
        match action {
            CyclicAction::HardClamp { node, level, .. } => {
                nodes.push(*node);
                levels.push(*level);
            }
            CyclicAction::ReplaceMechanism { node, replacement, .. } => {
                let row = &mut coefficients[node * n..(node + 1) * n];
                match replacement {
                    MechanismReplacement::Constant { value } => {
                        row.fill(0.0);
                        intercept[*node] = *value;
                    }
                    MechanismReplacement::Linear { intercept: value, weights } => {
                        row.copy_from_slice(weights);
                        intercept[*node] = *value;
                    }
                }
            }
            CyclicAction::ChangeMechanism { node, intercept_shift, .. } => {
                intercept[*node] += intercept_shift;
            }
            CyclicAction::AdditiveShock { node, delta, .. } => mean[*node] += delta,
            CyclicAction::InitialState { .. } => {
                return Err(refusal(
                    world,
                    Some(family),
                    "initial_state_change",
                    CyclicRefusalKind::Capability {
                        reason: "an equilibrium has no initial-state law",
                    },
                    family.next_attempt("initial_state_change"),
                ));
            }
        }
    }
    let model = LinearStructuralMap::try_new(coefficients, intercept)
        .map_err(|error| specification_refusal(family, world, error))?;
    let disturbance = spec
        .disturbance
        .with_mean(mean)
        .map_err(|error| specification_refusal(family, world, error))?;
    let clamp = InterventionRequest::hard_clamp(n, &nodes, &levels)
        .map_err(|error| specification_refusal(family, world, error))?;
    Ok(LinearWorldSystem { model, disturbance, clamp })
}

/// Evaluate one world of a specified linear Gaussian equilibrium.
///
/// The intervened free block is rechecked first; a singular, numerically
/// singular, or non-finite block is a result with failed mass one. When
/// `ρ(|K|) < 1` holds for the intervened map, the world is compiled through an
/// admitted component and certified. Otherwise the world is withheld under
/// [`CertificateDemand::Required`], or solved directly and reported
/// uncertified under [`CertificateDemand::ReportUncertified`].
///
/// # Errors
///
/// A refusal of the request ([`resolve_cyclic_world`]), a malformed intervened
/// map or disturbance, a component-gate refusal, or a budget refusal.
pub fn evaluate_linear_world(
    spec: &LinearCyclicSpecification,
    request: &CyclicWorldRequest,
    demand: CertificateDemand,
    budget: CompileBudget,
) -> Result<CyclicWorldResult, CyclicInterventionRefusal> {
    evaluate_charged(spec, request, demand, &mut LinearLedger::new(budget), None)
}

/// Work and bytes one linear evaluation may spend across every world it evaluates.
///
/// The disturbance covariance belongs to the specification, and its positive-semidefinite
/// verdict is shared by every world's disturbance, so the check is charged and run by the
/// first world that finds it unchecked.
struct LinearLedger {
    remaining_work: u64,
    max_bytes: u64,
}

impl LinearLedger {
    const fn new(budget: CompileBudget) -> Self {
        Self { remaining_work: budget.max_work, max_bytes: budget.max_bytes }
    }

    const fn budget(&self) -> CompileBudget {
        CompileBudget { max_work: self.remaining_work, max_bytes: self.max_bytes }
    }

    fn charge(&mut self, work: u64, bytes: u64) -> Result<(), SccCompileError> {
        if bytes > self.max_bytes {
            return Err(SccCompileError::Budget {
                resource: "bytes",
                requested: bytes,
                limit: self.max_bytes,
            });
        }
        if work > self.remaining_work {
            return Err(SccCompileError::Budget {
                resource: "work",
                requested: work,
                limit: self.remaining_work,
            });
        }
        self.remaining_work -= work;
        Ok(())
    }
}

/// Admission work of the positive-semidefinite check of an `n × n` covariance, and its bytes.
fn covariance_check_cost(n: usize) -> (u64, u64) {
    let n = n as u64;
    let square = n.saturating_mul(n);
    (square.saturating_mul(n).saturating_add(square), square.saturating_mul(16))
}

/// Admission work of the solvability recheck and the family assessment of a free block of
/// `free` nodes: one pivoted factorization and a few passes over the coefficients.
fn solvability_check_cost(free: usize) -> u64 {
    let free = free as u64;
    free.saturating_mul(free)
        .saturating_mul(free)
        .saturating_mul(2)
        .saturating_add(free.saturating_mul(free).saturating_mul(2))
}

/// Admission work and bytes of [`direct_free_block_moments`] over `free` of `n` nodes: two
/// dense Gauss-Jordan solves, and the factor, right-hand sides and their copies.
fn direct_solve_cost(n: usize, free: usize) -> (u64, u64) {
    let (n, m) = (n as u64, free as u64);
    let work = m
        .saturating_mul(m)
        .saturating_mul(m)
        .saturating_mul(4)
        .saturating_add(m.saturating_mul(m).saturating_mul(8));
    let floats = m
        .saturating_mul(m)
        .saturating_mul(4)
        .saturating_add(m)
        .saturating_add(n.saturating_mul(n))
        .saturating_add(n);
    (work, floats.saturating_mul(8))
}

fn evaluate_charged(
    spec: &LinearCyclicSpecification,
    request: &CyclicWorldRequest,
    demand: CertificateDemand,
    ledger: &mut LinearLedger,
    routes: Option<&LinearRouteSet>,
) -> Result<CyclicWorldResult, CyclicInterventionRefusal> {
    let family = CyclicFamily::LinearEquilibrium;
    let resolved = resolve_cyclic_world(family, &spec.variables, &[], request)?;
    let world = resolved.world;
    let system = linear_world_system(spec, &resolved)?;
    let n = system.model.dimension();
    if !system.disturbance.psd_verified() {
        let (work, bytes) = covariance_check_cost(n);
        ledger.charge(work, bytes).map_err(|error| execution_refusal(family, world, error))?;
        system
            .disturbance
            .require_psd_once()
            .map_err(|error| specification_refusal(family, Some(world), error))?;
    }
    let (clamp, values) = hard_clamp_of(&system.clamp)
        .map_err(|error| specification_refusal(family, Some(world), error))?;
    ledger
        .charge(solvability_check_cost(n - clamp.indices().len().min(n)), 0)
        .map_err(|error| execution_refusal(family, world, error))?;
    let assessment = assess(&system.model, clamp)
        .map_err(|error| specification_refusal(family, Some(world), error))?;
    let parts = match revalidate_solvability(&system.model, &system.clamp) {
        Err(InterventionRequestError::NotSolvable { refusal: refused }) => {
            failed_parts(WorldFailure::solvability(refused))
        }
        Err(error) => return Err(specification_refusal(family, Some(world), error)),
        Ok(_) if assessment.family.is_admitted() => {
            let evidence: Vec<&str> = spec.certifying_evidence.iter().map(String::as_str).collect();
            let component = CyclicComponent::try_new(
                &spec.component_id,
                &spec.source_id,
                &evidence,
                &system.model,
                &assessment.family,
            )
            .map_err(|error| {
                refusal(
                    Some(world),
                    Some(family),
                    "specification",
                    CyclicRefusalKind::Evidence { message: error.to_string() },
                    CERTIFYING_EVIDENCE_ATTEMPT,
                )
            })?;
            certified_linear_parts(world, &component, &system, (clamp, &values), (ledger, routes))?
        }
        Ok(_) => uncertified_linear_parts(
            world,
            &system,
            (clamp.indices(), &values),
            assessment.family.rho_abs_upper(),
            (demand, ledger),
        )?,
    };
    Ok(CyclicWorldResult::from_parts(resolved, &spec.variables, request, parts))
}

pub(super) fn failed_parts(failure: WorldFailure) -> WorldParts {
    let next_attempt = failure.next_attempt;
    (
        CyclicWorldOutcome::NoSolution { failure },
        WorldCertificate::NotCertified { gap: CertificateGap::NoSolution, next_attempt },
        WorldSolvedBy::NotSolved,
        WorldMass::FAILED,
    )
}

/// The world under an admitted `ρ(|K|) < 1` certificate, compiled through `component`.
fn certified_linear_parts(
    world: ScientificWorldId,
    component: &CyclicComponent,
    system: &LinearWorldSystem,
    (clamp, values): (&HardClamp, &[f64]),
    (ledger, routes): (&mut LinearLedger, Option<&LinearRouteSet>),
) -> Result<WorldParts, CyclicInterventionRefusal> {
    let family = CyclicFamily::LinearEquilibrium;
    let budget = ledger.budget();
    // A held route was charged when it was computed; the solve spends only what follows it.
    let held = routes.is_some_and(|set| set.get(&system.model, clamp).is_some());
    // A solve that fails after it was admitted spent the work it was admitted for.
    let spent = || {
        let routing = if held {
            gaussian_equilibrium_route_work(system.model.dimension()).unwrap_or(0)
        } else {
            0
        };
        gaussian_equilibrium_admission(&system.model, clamp)
            .map_or(budget.max_work, |cost| cost.work().saturating_sub(routing))
    };
    let routed = match compile_routed_with(
        component,
        &system.model,
        &system.disturbance,
        (clamp, values),
        budget,
        routes,
    ) {
        Ok(routed) => routed,
        Err(SccCompileError::NumericallySingularBlock { members }) => {
            ledger.remaining_work = ledger.remaining_work.saturating_sub(spent());
            return Ok(failed_parts(WorldFailure::solvability(
                SolvabilityRefusal::NumericallySingular { members },
            )));
        }
        Err(SccCompileError::NonFiniteEquilibrium) => {
            ledger.remaining_work = ledger.remaining_work.saturating_sub(spent());
            return Ok(failed_parts(WorldFailure::solvability(SolvabilityRefusal::NonFinite)));
        }
        Err(error @ SccCompileError::Budget { .. }) => {
            return Err(execution_refusal(family, world, error));
        }
        Err(error) => return Err(specification_refusal(family, Some(world), error)),
    };
    let solved_by = match routed.route() {
        LinearSupportRoute::Acyclic { order } => {
            WorldSolvedBy::AcyclicOrder { order: order.clone() }
        }
        LinearSupportRoute::Cyclic { next_attempt, blocks } => {
            WorldSolvedBy::StronglyConnectedBlocks {
                dag_refusal: next_attempt.clone(),
                blocks: blocks.clone(),
            }
        }
    };
    let equilibrium = routed.equilibrium();
    ledger.remaining_work = ledger.remaining_work.saturating_sub(equilibrium.admission().work());
    Ok((
        CyclicWorldOutcome::GaussianLaw {
            mean: equilibrium.mean().to_vec(),
            covariance: equilibrium.covariance().to_vec(),
            residual_inf: Some(equilibrium.residual_inf()),
            block_diagnostics: Some(equilibrium.blocks().to_vec()),
        },
        WorldCertificate::Certified {
            basis: CertificateBasis::LinearFamilyContraction {
                rho_abs_upper: component.family_rho_abs_upper(),
            },
        },
        solved_by,
        WorldMass::SOLVED,
    ))
}

/// The world when `ρ(|K|) < 1` was not shown: withheld, or solved directly and
/// marked uncertified.
fn uncertified_linear_parts(
    world: ScientificWorldId,
    system: &LinearWorldSystem,
    (clamp_nodes, values): (&[usize], &[f64]),
    rho_abs_upper: f64,
    (demand, ledger): (CertificateDemand, &mut LinearLedger),
) -> Result<WorldParts, CyclicInterventionRefusal> {
    let certificate = WorldCertificate::NotCertified {
        gap: CertificateGap::LinearFamilyBoundInsufficient { rho_abs_upper },
        next_attempt: QUERY_SPECIFIC_CERTIFICATE_ATTEMPT,
    };
    if demand == CertificateDemand::Required {
        return Ok((
            CyclicWorldOutcome::Withheld,
            certificate,
            WorldSolvedBy::NotSolved,
            WorldMass::WITHHELD,
        ));
    }
    let n = system.model.dimension();
    let (work, bytes) = direct_solve_cost(n, n - clamp_nodes.len().min(n));
    ledger
        .charge(work, bytes)
        .map_err(|error| execution_refusal(CyclicFamily::LinearEquilibrium, world, error))?;
    Ok(match direct_free_block_moments(&system.model, clamp_nodes, values, &system.disturbance) {
        Ok((mean, covariance, free)) => (
            CyclicWorldOutcome::GaussianLaw {
                mean,
                covariance,
                residual_inf: None,
                block_diagnostics: None,
            },
            certificate,
            WorldSolvedBy::DirectFreeBlockSolve { free },
            WorldMass::SOLVED,
        ),
        Err(refused) => failed_parts(WorldFailure::solvability(refused)),
    })
}

/// Evaluate every request against one linear specification.
///
/// `budget` bounds the worlds together: the work of each is charged against what the earlier
/// worlds left, and the disturbance covariance is checked once.
///
/// # Errors
///
/// Two requests with one world id, or the first refusal of
/// [`evaluate_linear_world`].
pub fn evaluate_linear_worlds(
    spec: &LinearCyclicSpecification,
    requests: &[CyclicWorldRequest],
    demand: CertificateDemand,
    budget: CompileBudget,
) -> Result<CyclicWorldResults, CyclicInterventionRefusal> {
    evaluate_worlds(spec, requests, demand, budget, None)
}

/// [`evaluate_linear_worlds`] that reuses the support routes already computed for `spec`'s map.
///
/// A world whose intervened map and clamp set match a route in `routes` is compiled from it,
/// and the routing work, charged when the route was computed, is not charged again. Every
/// other world routes itself and is charged for it. A set built for another map is ignored,
/// so the result never depends on where `routes` came from.
///
/// # Errors
///
/// The errors of [`evaluate_linear_worlds`].
pub fn evaluate_linear_worlds_routed(
    spec: &LinearCyclicSpecification,
    requests: &[CyclicWorldRequest],
    demand: CertificateDemand,
    budget: CompileBudget,
    routes: &LinearRouteSet,
) -> Result<CyclicWorldResults, CyclicInterventionRefusal> {
    evaluate_worlds(spec, requests, demand, budget, Some(routes))
}

fn evaluate_worlds(
    spec: &LinearCyclicSpecification,
    requests: &[CyclicWorldRequest],
    demand: CertificateDemand,
    budget: CompileBudget,
    routes: Option<&LinearRouteSet>,
) -> Result<CyclicWorldResults, CyclicInterventionRefusal> {
    let mut seen = std::collections::BTreeSet::new();
    for request in requests {
        if !seen.insert(request.world) {
            return Err(duplicate_world(request.world, CyclicFamily::LinearEquilibrium));
        }
    }
    let mut ledger = LinearLedger::new(budget);
    let results = requests
        .iter()
        .map(|request| evaluate_charged(spec, request, demand, &mut ledger, routes))
        .collect::<Result<Vec<_>, _>>()?;
    CyclicWorldResults::try_from_results(results)
}

/// The free block's mean, its covariance and the free node indices.
type FreeBlockMoments = (Vec<f64>, Vec<f64>, Vec<usize>);

/// Gaussian moments of the free block without a family certificate.
///
/// `(I - K_FF) μ_F = c_F + K_FH a + μ_ε,F` and
/// `Σ_X,FF = (I - K_FF)^{-1} Σ_ε,FF (I - K_FF)^{-T}`, both by solves. The
/// inverse is not formed. `clamp_nodes` is ascending and `levels` follows it.
fn direct_free_block_moments(
    model: &LinearStructuralMap,
    clamp_nodes: &[usize],
    levels: &[f64],
    disturbance: &GaussianDisturbance,
) -> Result<FreeBlockMoments, SolvabilityRefusal> {
    let n = model.dimension();
    let (k, c) = (model.coefficients(), model.intercept());
    let (mu, sigma) = (disturbance.mean(), disturbance.covariance());
    let free = free_indices(n, clamp_nodes);
    let m = free.len();
    let mut mean = vec![0.0; n];
    for (&node, &level) in clamp_nodes.iter().zip(levels) {
        mean[node] = level;
    }
    let mut covariance = vec![0.0; n * n];
    if m == 0 {
        return Ok((mean, covariance, free));
    }
    let mut factor = vec![0.0; m * m];
    let mut rhs = vec![0.0; m * (m + 1)];
    for (p, &row) in free.iter().enumerate() {
        let mut value = c[row] + mu[row];
        for (&node, &level) in clamp_nodes.iter().zip(levels) {
            value += k[row * n + node] * level;
        }
        rhs[p] = value;
        for (q, &col) in free.iter().enumerate() {
            let identity = if p == q { 1.0 } else { 0.0 };
            factor[p * m + q] = identity - k[row * n + col];
            // Right-hand side q + 1 is column q of Σ_FF.
            rhs[(q + 1) * m + p] = sigma[row * n + col];
        }
    }
    let mut first = factor.clone();
    if !gauss_jordan_solve(&mut first, &mut rhs, m, m + 1) {
        return Err(SolvabilityRefusal::NumericallySingular { members: free });
    }
    // Y = A^{-1} Σ_FF has Y[p][q] at rhs[(q + 1) * m + p]. Column q of Yᵀ is row q of Y.
    let mut transposed = vec![0.0; m * m];
    for q in 0..m {
        for p in 0..m {
            transposed[q * m + p] = rhs[(p + 1) * m + q];
        }
    }
    if !gauss_jordan_solve(&mut factor, &mut transposed, m, m) {
        return Err(SolvabilityRefusal::NumericallySingular { members: free });
    }
    // Z = A^{-1} Yᵀ = A^{-1} Σ_FF A^{-T}, with Z[p][q] at transposed[q * m + p].
    for (p, &row) in free.iter().enumerate() {
        mean[row] = rhs[p];
        for (q, &col) in free.iter().enumerate() {
            covariance[row * n + col] = 0.5 * (transposed[q * m + p] + transposed[p * m + q]);
        }
    }
    if mean.iter().chain(&covariance).any(|value| !value.is_finite()) {
        return Err(SolvabilityRefusal::NonFinite);
    }
    Ok((mean, covariance, free))
}
