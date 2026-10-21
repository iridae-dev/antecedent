// SPDX-License-Identifier: MIT OR Apache-2.0
//! Retained typed cyclic programs. Numerical work stays in model/estimate engines;
//! encoding and independent consumers use this same execution owner.
use super::{
    CyclicArtifactError, CyclicFamilyAnswer as A, CyclicFamilyReport, CyclicFamilyRequest as R,
    bounded_json, invalid,
};
use antecedent_core::{
    CanonicalWriter, ExecutionContext, KernelPolicy,
    execution::AdmissionLease,
    execution_attempt::{PhysicalEvent, record_physical},
};
use antecedent_model::compile::{
    CompiledSystemPlan, ExecutionBlock, SystemSemantics, compile_constraints, compile_equilibrium,
    compile_ode, compile_ou,
};
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug)]
pub(super) enum FrozenFamilyFitValue {
    Gaussian(FrozenGaussianDensity),
    Structure {
        fit: antecedent_estimate::cyclic_likelihood::CyclicStructureFit,
        bootstrap: Vec<Option<antecedent_estimate::cyclic_likelihood::CyclicStructureFit>>,
    },
    Bootstrap(Vec<Option<Vec<f64>>>),
    Parameter(antecedent_estimate::cyclic_likelihood::CyclicPosterior),
    Joint(antecedent_estimate::cyclic_likelihood::CyclicJointAbduction),
    Partial(antecedent_estimate::cyclic_likelihood::CyclicAbductionPosterior),
    Ou(Option<antecedent_estimate::ou_mechanisms::OuMechanismFit>),
    Adam(antecedent_estimate::cyclic_adam::CyclicAdamFit),
    Gradient(antecedent_estimate::cyclic_gradient::CyclicGradientFit),
    Polynomial(antecedent_estimate::constraint_likelihood::PolynomialFit),
    Ode(antecedent_estimate::cyclic_dynamics::OdeFit),
    OuFit(antecedent_estimate::cyclic_dynamics::OuFit),
}
pub(crate) use antecedent_estimate::cyclic_gaussian_functional::FrozenGaussianDensity;
#[derive(Debug)]
struct FrozenFit {
    kernel: KernelPolicy,
    value: FrozenFamilyFitValue,
    retention: AdmissionLease,
}
impl FrozenFit {
    fn capture(
        answer: &A,
        kernel: KernelPolicy,
        ctx: &ExecutionContext,
    ) -> Result<Option<Self>, CyclicArtifactError> {
        fn retain<T: serde::Serialize + Clone>(
            value: &T,
            ctx: &ExecutionContext,
            slots: u64,
            wrap: impl FnOnce(T) -> FrozenFamilyFitValue,
        ) -> Result<(FrozenFamilyFitValue, AdmissionLease), CyclicArtifactError> {
            let bytes = crate::retained_allocation::allocation_bytes(value)
                .map_err(invalid)?
                .checked_add(std::mem::size_of::<FrozenFamilyFitValue>() as u64)
                .and_then(|bytes| bytes.checked_add(slots))
                .ok_or_else(|| invalid("fit retention size overflow"))?;
            let retention = ctx.acquire_admission(0, bytes).map_err(invalid)?;
            Ok((wrap(value.clone()), retention))
        }
        let (value, retention) = match answer {
            A::GaussianFunctional(report) if report.fitted_rows.is_some() => retain(
                &(&report.mean, &report.covariance, report.fitted_rows),
                ctx,
                0,
                |(mean, covariance, fitted_rows)| {
                    FrozenFamilyFitValue::Gaussian(FrozenGaussianDensity {
                        mean: mean.clone(),
                        covariance: covariance.clone(),
                        fitted_rows,
                    })
                },
            )?,
            A::StructureFit { fit, bootstrap } => retain(
                &(fit, bootstrap),
                ctx,
                crate::retained_allocation::typed_slots(bootstrap).map_err(invalid)?,
                |(fit, bootstrap)| FrozenFamilyFitValue::Structure {
                    fit: fit.clone(),
                    bootstrap: bootstrap.clone(),
                },
            )?,
            A::FrequentistBootstrap { parameters, .. } => retain(
                parameters,
                ctx,
                crate::retained_allocation::typed_slots(parameters).map_err(invalid)?,
                FrozenFamilyFitValue::Bootstrap,
            )?,
            A::ParameterPosterior { posterior, .. } => {
                retain(posterior, ctx, 0, FrozenFamilyFitValue::Parameter)?
            }
            A::JointCounterfactual { posterior, .. } => {
                retain(posterior, ctx, 0, FrozenFamilyFitValue::Joint)?
            }
            A::PartialCounterfactual { abduction, .. } => {
                retain(abduction, ctx, 0, FrozenFamilyFitValue::Partial)?
            }
            A::OuMechanisms { fit, .. } => retain(fit, ctx, 0, FrozenFamilyFitValue::Ou)?,
            A::AdamFit(fit) => retain(fit, ctx, 0, FrozenFamilyFitValue::Adam)?,
            A::GradientFit(fit) => retain(fit, ctx, 0, FrozenFamilyFitValue::Gradient)?,
            A::PolynomialFit(fit) => retain(fit, ctx, 0, FrozenFamilyFitValue::Polynomial)?,
            A::OdeFit(fit) => retain(fit, ctx, 0, FrozenFamilyFitValue::Ode)?,
            A::OuFit(fit) => retain(fit, ctx, 0, FrozenFamilyFitValue::OuFit)?,
            _ => return Ok(None),
        };
        Ok(Some(Self { kernel, value, retention }))
    }
}
/// One immutable scientific request, compiled family schedules and synchronized statistical products.
/// Specified models remain model conditional; compilation grants no scientific license.
#[derive(Clone, Debug)]
pub struct CompiledCyclicFamilyProgram {
    request: Arc<R>,
    identity: String,
    fit_identity: String,
    plans: Arc<[CompiledSystemPlan]>,
    fit: Arc<Mutex<Option<FrozenFit>>>,
    source: Option<Arc<crate::cyclic_functional::CheckedCyclicFunctionalSource>>,
    retention: Arc<AdmissionLease>,
}
impl CompiledCyclicFamilyProgram {
    /// Compile without fitting, sampling, solving a trajectory or encoding an artifact.
    pub fn compile(request: &R, ctx: &ExecutionContext) -> Result<Self, CyclicArtifactError> {
        Self::compile_bound(request, ctx, None)
    }
    fn compile_bound(
        request: &R,
        ctx: &ExecutionContext,
        previous: Option<&Self>,
    ) -> Result<Self, CyclicArtifactError> {
        if ctx.cancellation.is_cancelled() {
            return Err(invalid("cyclic compilation cancelled"));
        }
        let plans = compile_plans(request)?;
        let retained_source = previous
            .filter(|previous| same_source(request, &previous.request))
            .and_then(|previous| previous.source.clone());
        let source = if retained_source.is_some() {
            retained_source
        } else {
            match request {
                R::Functional { source, source_identity, program, .. }
                | R::GaussianFunctional { source, source_identity, program, .. } => {
                    Some(Arc::new(crate::cyclic_functional::checked_functional_source(
                        source,
                        source_identity,
                        program,
                        ctx,
                    )?))
                }
                _ => None,
            }
        };
        let (bytes, retention) = bounded_json(request, ctx)?;
        let identity =
            CanonicalWriter::digest_bytes_hex("antecedent.cyclic_family.request.v1", &bytes);
        let mut dependencies: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(invalid)?;
        let excluded: &[&str] = match request {
            R::FrequentistBootstrap { .. } | R::ParameterPosterior { .. } => {
                &["worlds", "noise", "initial", "solve_policy"]
            }
            R::RecoveredLinear { .. } | R::OuMechanisms { .. } => &["worlds"],
            R::JointCounterfactual { .. } | R::PartialCounterfactual { .. } => {
                &["worlds", "solve_policy"]
            }
            R::GaussianFunctional { .. } => &["assignments"],
            _ => &[],
        };
        if let Some(object) = dependencies.as_object_mut() {
            for field in excluded {
                object.remove(*field);
            }
        }
        let fit_bytes = serde_json::to_vec(&dependencies).map_err(invalid)?;
        let fit_identity =
            CanonicalWriter::digest_bytes_hex("antecedent.cyclic_family.fit.v1", &fit_bytes);
        Ok(Self {
            request: Arc::new(request.clone()),
            identity,
            fit_identity,
            plans: plans.into(),
            fit: Arc::default(),
            source,
            retention: Arc::new(retention),
        })
    }
    /// Original typed request, including actual statistical and initial-law dependencies.
    #[must_use]
    pub fn request(&self) -> &R {
        &self.request
    }
    /// Complete target/source/numerical-policy binding, inspectable before execution.
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }
    /// Existing equilibrium/dynamic/equation compiler products; never coerced to DAG semantics.
    #[must_use]
    pub fn execution_plans(&self) -> &[CompiledSystemPlan] {
        &self.plans
    }
    /// Actual request reservation retained by this owner.
    #[must_use]
    pub fn retained_bytes(&self) -> u64 {
        self.retention.remainder().bytes()
    }
    /// Compile changed targets and share only statistical products whose actual inputs match.
    pub fn rebind(&self, request: &R, ctx: &ExecutionContext) -> Result<Self, CyclicArtifactError> {
        let mut next = Self::compile_bound(request, ctx, Some(self))?;
        if next.fit_identity == self.fit_identity {
            next.fit = Arc::clone(&self.fit);
        }
        Ok(next)
    }
    /// Execute retained typed operations; numerical algorithms remain in their original engines.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<CyclicFamilyReport, CyclicArtifactError> {
        if ctx.cancellation.is_cancelled() {
            return Err(invalid("cyclic execution cancelled"));
        }
        self.retention.observe_current();
        antecedent_core::numerical_admission::with_context(ctx, || {
            let mut cache = self.fit.lock().map_err(|_| invalid("cyclic fit lock poisoned"))?;
            let retained = cache.as_ref().filter(|fit| fit.kernel == ctx.kernel_policy);
            if let Some(fit) = retained {
                fit.retention.observe_current();
                record_physical(PhysicalEvent::CacheHit, 1);
                record_physical(PhysicalEvent::ModelFit, 0);
            }
            let report = self.request.execute_bound(
                &self.identity,
                retained.map(|fit| &fit.value),
                self.source.as_deref(),
                ctx,
            )?;
            if retained.is_none() {
                *cache = FrozenFit::capture(&report.answer, ctx.kernel_policy, ctx)?;
            }
            Ok(report)
        })
    }
    /// Execute and then encode separately. No artifact is produced during compilation/inspection.
    pub fn export(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<(CyclicFamilyReport, Vec<u8>), CyclicArtifactError> {
        let report = self.execute(ctx)?;
        let bytes = self.request.encode_report(&report, ctx)?;
        Ok((report, bytes))
    }
}
/// Pre-admit output multiplicities before numerical execution. The typed post-execution
/// traversal checks the retained shape and never confuses short JSON numbers or nulls with RAM.
#[allow(
    clippy::too_many_lines,
    reason = "each cyclic family retains distinct typed output axes and bounded numerical policies"
)]
pub(super) fn output_bytes(request: &R) -> Result<u64, CyclicArtifactError> {
    fn overflow() -> CyclicArtifactError {
        invalid("output size overflow")
    }
    fn product(values: &[usize]) -> Result<u64, CyclicArtifactError> {
        values.iter().try_fold(1u64, |n, &v| {
            n.checked_mul(u64::try_from(v).map_err(invalid)?).ok_or_else(overflow)
        })
    }
    fn width(terms: &[(usize, usize)], extra: usize) -> Result<usize, CyclicArtifactError> {
        terms.iter().try_fold(extra, |sum, &(n, scale)| {
            sum.checked_add(n.checked_mul(scale).ok_or_else(overflow)?).ok_or_else(overflow)
        })
    }
    fn square(n: usize) -> Result<usize, CyclicArtifactError> {
        n.checked_mul(n).ok_or_else(overflow)
    }
    fn one_more(n: usize) -> Result<usize, CyclicArtifactError> {
        n.checked_add(1).ok_or_else(overflow)
    }
    let base = crate::retained_allocation::allocation_bytes(request).map_err(invalid)?;
    let extra = match request {
        R::ParameterPosterior { model, parameters, sampling, worlds, noise, .. } => product(&[
            sampling.chains,
            sampling.draws,
            worlds.len().max(1),
            noise.len().max(1),
            width(&[(model.order(), 128), (parameters.len(), 16)], 512)?,
        ])?,
        R::JointCounterfactual { model, parameters, sampling, worlds, .. } => product(&[
            sampling.chains,
            sampling.draws,
            worlds.len().max(1),
            width(&[(model.order(), 256), (parameters.len(), 16)], 512)?,
        ])?,
        R::PartialCounterfactual { model, sampling, worlds, .. } => product(&[
            sampling.chains,
            sampling.draws,
            worlds.len().max(1),
            width(&[(model.order(), 256)], 512)?,
        ])?,
        R::FrequentistBootstrap { model, parameters, replicates, worlds, noise, .. } => {
            product(&[
                *replicates,
                worlds.len().max(1),
                noise.len().max(1),
                width(&[(model.order(), 128), (parameters.len(), 16)], 512)?,
            ])?
        }
        R::EquilibriumActions { model, worlds, noise, factual, .. } => product(&[
            worlds.len(),
            noise.len().checked_add(usize::from(factual.is_some())).ok_or_else(overflow)?,
            width(&[(model.order(), 128)], 512)?,
        ])?,
        R::StructureFit { model, bootstrap_replicates, .. } => product(&[
            one_more(*bootstrap_replicates)?,
            width(
                &[(square(model.order())?, 64)],
                4096usize
                    .checked_add(std::mem::size_of::<
                        Option<antecedent_estimate::cyclic_likelihood::CyclicStructureFit>,
                    >())
                    .ok_or_else(overflow)?,
            )?,
        ])?,
        R::AdamFit { policy, .. } => product(&[
            policy.max_epochs,
            std::mem::size_of::<antecedent_estimate::cyclic_adam::CyclicAdamEpoch>(),
        ])?,
        R::Ode { initial, times, .. } => {
            product(&[times.len(), width(&[(initial.len(), 16)], 64)?])?
        }
        R::OuMechanisms { model, worlds, .. } => product(&[
            worlds.len(),
            width(&[(square(model.equations.len())?, 16), (model.equations.len(), 16)], 512)?,
        ])?,
        R::OuTransient { initial_mean, times, .. } => product(&[
            times.len(),
            width(&[(square(initial_mean.len())?, 16), (initial_mean.len(), 16)], 128)?,
        ])?,
        R::Polynomial { model, policy } => {
            product(&[one_more(policy.max_boxes)?, width(&[(model.variables.len(), 32)], 64)?])?
        }
        R::EquilibriumRoots { model, policy, .. } => {
            product(&[one_more(policy.max_boxes)?, width(&[(model.order(), 32)], 64)?])?
        }
        R::PolynomialFit { candidates, policy, .. } => product(&[
            candidates.len(),
            one_more(policy.max_boxes)?,
            width(
                &[(candidates.iter().map(|c| c.model.variables.len()).max().unwrap_or(0), 32)],
                64,
            )?,
        ])?,
        R::PolynomialCausal { program } => {
            let atoms = program.candidates.iter().try_fold(0usize, |n, candidate| {
                n.checked_add(candidate.atoms.len()).ok_or_else(overflow)
            })?;
            product(&[
                atoms,
                program.worlds.len().max(1),
                width(
                    &[(
                        program
                            .candidates
                            .iter()
                            .map(|c| c.model.variables.len())
                            .max()
                            .unwrap_or(0),
                        64,
                    )],
                    1024,
                )?,
            ])?
        }
        R::Functional { assignments, .. } | R::GaussianFunctional { assignments, .. } => {
            product(&[assignments.len(), 512])?
        }
        R::RecoveredLinear { nodes, draws, worlds, .. } => product(&[
            draws.len().max(1),
            worlds.len().max(1),
            width(&[(square(nodes.len())?, 32), (nodes.len(), 16)], 256)?,
        ])?,
        _ => 0,
    };
    base.checked_add(extra).and_then(|n| n.checked_add(4096)).ok_or_else(overflow)
}

fn compile_plans(request: &R) -> Result<Vec<CompiledSystemPlan>, CyclicArtifactError> {
    let plans = match request {
        R::AdamFit { model, .. }
        | R::EquilibriumRoots { model, .. }
        | R::GradientFit { model, .. }
        | R::EquilibriumActions { model, .. }
        | R::FrequentistBootstrap { model, .. }
        | R::ParameterPosterior { model, .. }
        | R::JointCounterfactual { model, .. }
        | R::PartialCounterfactual { model, .. }
        | R::StructureFit { model, .. } => {
            model.validate()?;
            let worlds = match request {
                R::FrequentistBootstrap { worlds, .. }
                | R::ParameterPosterior { worlds, .. }
                | R::JointCounterfactual { worlds, .. }
                | R::PartialCounterfactual { worlds, .. } => Some(worlds),
                _ => None,
            };
            if let Some(worlds) = worlds {
                worlds
                    .iter()
                    .map(|world| compile_equilibrium(model, world))
                    .collect::<Result<Vec<_>, _>>()?
            } else {
                vec![compile_equilibrium(model, &[])?]
            }
        }
        R::Ode { model, .. } | R::OdeFit { model, .. } => vec![compile_ode(model)?],
        R::OuStationary { model } | R::OuTransient { model, .. } | R::OuFit { model, .. } => {
            vec![compile_ou(model)?]
        }
        R::Polynomial { model, .. } => vec![compile_constraints(model)?],
        R::PolynomialCausal { program } => program
            .candidates
            .iter()
            .map(|candidate| compile_constraints(&candidate.model))
            .collect::<Result<Vec<_>, _>>()?,
        R::PolynomialFit { candidates, .. } => candidates
            .iter()
            .map(|candidate| compile_constraints(&candidate.model))
            .collect::<Result<Vec<_>, _>>()?,
        R::OuMechanisms { model, .. } => {
            model.validate()?;
            vec![]
        }
        R::Functional { .. } | R::GaussianFunctional { .. } | R::Graph { .. } => vec![],
        R::RecoveredLinear { nodes, draws, worlds, outcome, source_id, certifying_evidence } => {
            recovered_plans(nodes, draws, worlds, *outcome, source_id, certifying_evidence)?
        }
    };
    Ok(plans)
}

fn recovered_plans(
    nodes: &[u32],
    draws: &[super::RecoveredDrawWire],
    worlds: &[super::RecoveredInterventionWire],
    outcome: usize,
    source_id: &str,
    certifying_evidence: &[String],
) -> Result<Vec<CompiledSystemPlan>, CyclicArtifactError> {
    // Retention checks the supplied law. It does not fit, sample, or estimate.
    validate_recovered_law(nodes, draws, worlds, outcome, source_id, certifying_evidence)?;
    let support = support_map(nodes, draws)?;
    worlds
        .iter()
        .map(|world| {
            if world.clamp_values.len() != world.clamp_nodes.len()
                || world.clamp_values.iter().any(|value| !value.is_finite())
            {
                return Err(invalid("recovered clamp"));
            }
            let clamp = antecedent_model::HardClamp::try_new(nodes.len(), &world.clamp_nodes)
                .map_err(invalid)?;
            let route = antecedent_model::scc_compile::route_linear_support(&support, &clamp)
                .map_err(invalid)?;
            Ok(plan_from_route(&route))
        })
        .collect()
}

/// Execute interventions from the retained recovered law.
///
/// Preparation has already retained that law. This does not fit. Specified
/// models remain model-conditional; compilation grants no scientific license.
pub(super) fn execute_recovered_linear(
    nodes: &[u32],
    draws: &[super::RecoveredDrawWire],
    worlds: &[super::RecoveredInterventionWire],
    outcome: usize,
    source_id: &str,
    certifying_evidence: &[String],
) -> Result<A, CyclicArtifactError> {
    validate_recovered_law(nodes, draws, worlds, outcome, source_id, certifying_evidence)?;
    let law = retain_recovered(nodes, draws)?;
    let mut answers = Vec::with_capacity(worlds.len());
    for world in worlds {
        let plan = antecedent_estimate::recovery::RetainedRecoveryPlan::try_new(
            antecedent_estimate::recovery::RecoveryPlanParts {
                uncertainty: law.clone(),
                clamp_nodes: world.clamp_nodes.clone(),
                clamp_values: world.clamp_values.clone(),
                outcome,
                source_id: source_id.to_owned(),
                certifying_evidence: certifying_evidence.to_vec(),
            },
        )
        .map_err(invalid)?;
        let evaluation = antecedent_estimate::recovery::consume_retained_recovery_plan(&plan);
        answers.push(world_answer(&evaluation));
    }
    Ok(A::RecoveredLinear(super::RecoveredLinearReport { worlds: answers }))
}

fn validate_recovered_law(
    nodes: &[u32],
    draws: &[super::RecoveredDrawWire],
    worlds: &[super::RecoveredInterventionWire],
    outcome: usize,
    source_id: &str,
    certifying_evidence: &[String],
) -> Result<(), CyclicArtifactError> {
    if nodes.is_empty()
        || nodes.len() > 64
        || draws.is_empty()
        || draws.len() > 10_000
        || worlds.is_empty()
        || worlds.len() > 256
        || outcome >= nodes.len()
        || source_id.is_empty()
        || certifying_evidence.is_empty()
        || certifying_evidence.iter().any(|item| item.is_empty() || item == source_id)
    {
        return Err(invalid("recovered linear law"));
    }
    let law = retain_recovered(nodes, draws)?;
    let first = &worlds[0];
    let plan = antecedent_estimate::recovery::RetainedRecoveryPlan::try_new(
        antecedent_estimate::recovery::RecoveryPlanParts {
            uncertainty: law,
            clamp_nodes: first.clamp_nodes.clone(),
            clamp_values: first.clamp_values.clone(),
            outcome,
            source_id: source_id.to_owned(),
            certifying_evidence: certifying_evidence.to_vec(),
        },
    )
    .map_err(invalid)?;
    drop(plan);
    Ok(())
}

fn retain_recovered(
    nodes: &[u32],
    draws: &[super::RecoveredDrawWire],
) -> Result<
    antecedent_identify::experimental_recovery::RecoveredCoefficientUncertainty,
    CyclicArtifactError,
> {
    let ids: Vec<antecedent_core::VariableId> =
        nodes.iter().copied().map(antecedent_core::VariableId::from_raw).collect();
    let inputs = draws.iter().map(draw_input).collect();
    let retained =
        antecedent_identify::experimental_recovery::RecoveredCoefficientUncertainty::retain(
            &ids, inputs,
        );
    retained.map_err(invalid)
}

fn draw_input(
    draw: &super::RecoveredDrawWire,
) -> antecedent_identify::experimental_recovery::RecoveryDrawInput {
    use antecedent_core::VariableId;
    use antecedent_identify::experimental_recovery::{
        DirectCoefficient, RecoveryDrawInput, UnresolvedDirection, UnresolvedReason,
    };
    RecoveryDrawInput {
        coefficients: draw
            .coefficients
            .iter()
            .map(|coef| {
                (
                    DirectCoefficient {
                        into: VariableId::from_raw(coef.into),
                        from: VariableId::from_raw(coef.from),
                    },
                    coef.value,
                )
            })
            .collect(),
        intercept: draw.intercept.clone(),
        noise_mean: draw.noise_mean.clone(),
        noise_covariance: draw.noise_covariance.clone(),
        unresolved: draw
            .unresolved
            .iter()
            .map(|wire| UnresolvedDirection {
                support: wire
                    .support
                    .iter()
                    .map(|item| DirectCoefficient {
                        into: VariableId::from_raw(item.into),
                        from: VariableId::from_raw(item.from),
                    })
                    .collect(),
                reason: match &wire.reason {
                    super::RecoveredUnresolvedWire::IncompletePairCoverage {
                        target,
                        missing_clamps,
                    } => UnresolvedReason::IncompletePairCoverage {
                        target: VariableId::from_raw(*target),
                        missing_clamps: missing_clamps
                            .iter()
                            .copied()
                            .map(VariableId::from_raw)
                            .collect(),
                    },
                    super::RecoveredUnresolvedWire::DeficientConstraintRank {
                        target,
                        column_rank,
                        width,
                    } => UnresolvedReason::DeficientConstraintRank {
                        target: VariableId::from_raw(*target),
                        column_rank: *column_rank,
                        width: *width,
                    },
                },
            })
            .collect(),
        weight: draw.weight,
        marked_inadmissible: draw.marked_inadmissible,
    }
}

fn support_map(
    nodes: &[u32],
    draws: &[super::RecoveredDrawWire],
) -> Result<antecedent_model::LinearStructuralMap, CyclicArtifactError> {
    let n = nodes.len();
    let len = n.checked_mul(n).ok_or_else(|| invalid("recovered support overflow"))?;
    let mut coefficients = vec![0.0; len];
    for draw in draws {
        for coef in &draw.coefficients {
            if !coef.value.is_finite() || coef.into == coef.from {
                continue;
            }
            #[allow(clippy::float_cmp, reason = "a structural zero is not a support edge")]
            if coef.value == 0.0 {
                continue;
            }
            let Some(into) = nodes.iter().position(|node| *node == coef.into) else {
                continue;
            };
            let Some(from) = nodes.iter().position(|node| *node == coef.from) else {
                continue;
            };
            coefficients[into * n + from] = coef.value;
        }
    }
    antecedent_model::LinearStructuralMap::try_new(coefficients, vec![0.0; n]).map_err(invalid)
}

fn plan_from_route(
    route: &antecedent_model::scc_compile::LinearSupportRoute,
) -> CompiledSystemPlan {
    use antecedent_model::scc_compile::LinearSupportRoute;
    match route {
        LinearSupportRoute::Acyclic { order } => CompiledSystemPlan {
            semantics: SystemSemantics::AcyclicScm,
            blocks: order
                .iter()
                .map(|&variable| ExecutionBlock::Mechanism {
                    variable,
                    mechanism: format!("node:{variable}"),
                })
                .collect(),
        },
        LinearSupportRoute::Cyclic { blocks, .. } => CompiledSystemPlan {
            semantics: SystemSemantics::EquilibriumScm,
            blocks: blocks
                .iter()
                .map(|variables| {
                    if variables.len() == 1 {
                        ExecutionBlock::Mechanism {
                            variable: variables[0],
                            mechanism: format!("node:{}", variables[0]),
                        }
                    } else {
                        ExecutionBlock::SimultaneousEquations {
                            variables: variables.clone(),
                            equations: variables
                                .iter()
                                .map(|variable| format!("node:{variable}"))
                                .collect(),
                        }
                    }
                })
                .collect(),
        },
    }
}

fn world_answer(
    evaluation: &antecedent_estimate::recovery::CertifiedInterventionEvaluation,
) -> super::RecoveredWorldAnswer {
    use antecedent_estimate::recovery::{EvaluatedOutcome, SolvedBy};
    use antecedent_identify::experimental_recovery::InadmissibleCoefficientDraw;
    let mixture = evaluation.outcome_mixture();
    let mixture_mean = mixture.as_ref().map(|item| item.mean);
    let mixture_variance =
        mixture.as_ref().map(antecedent_estimate::recovery::OutcomeMixture::variance);
    let between_draw_variance = mixture.as_ref().map(|item| item.between_draw_variance);
    super::RecoveredWorldAnswer {
        draws: evaluation
            .draws()
            .iter()
            .map(|draw| super::RecoveredEvaluatedDraw {
                weight: draw.weight(),
                outcome: match draw.outcome() {
                    EvaluatedOutcome::Equilibrium {
                        mean,
                        covariance,
                        disturbance_mean,
                        disturbance_covariance,
                        solved_by,
                    } => super::RecoveredDrawAnswer::Equilibrium {
                        mean: mean.clone(),
                        covariance: covariance.clone(),
                        disturbance_mean: disturbance_mean.clone(),
                        disturbance_covariance: disturbance_covariance.clone(),
                        solved_by: match solved_by {
                            SolvedBy::Acyclic => super::RecoveredSolvedBy::Acyclic,
                            SolvedBy::Cyclic { next_attempt } => super::RecoveredSolvedBy::Cyclic {
                                next_attempt: next_attempt.clone(),
                            },
                        },
                    },
                    EvaluatedOutcome::Inadmissible(reason) => {
                        super::RecoveredDrawAnswer::Inadmissible {
                            reason: match reason {
                                InadmissibleCoefficientDraw::Marked => "marked",
                                InadmissibleCoefficientDraw::WeightNotMass => "weight_not_mass",
                                InadmissibleCoefficientDraw::UnresolvedDirections => {
                                    "unresolved_directions"
                                }
                                InadmissibleCoefficientDraw::StructuralDiagonal => {
                                    "structural_diagonal"
                                }
                                InadmissibleCoefficientDraw::NonFiniteParameter => {
                                    "non_finite_parameter"
                                }
                                InadmissibleCoefficientDraw::NoiseNotJoint => "noise_not_joint",
                                InadmissibleCoefficientDraw::IncompleteMechanism => {
                                    "incomplete_mechanism"
                                }
                            }
                            .to_owned(),
                        }
                    }
                    EvaluatedOutcome::NotCertified(fault) => {
                        super::RecoveredDrawAnswer::NotCertified { fault: fault.to_string() }
                    }
                },
            })
            .collect(),
        outcome: evaluation.outcome(),
        total_mass: evaluation.total_mass(),
        withheld_mass: evaluation.withheld_mass(),
        mixture_mean,
        mixture_variance,
        between_draw_variance,
    }
}

fn same_source(left: &R, right: &R) -> bool {
    match (left, right) {
        (
            R::Functional { source: a, source_identity: ai, program: ap, .. },
            R::Functional { source: b, source_identity: bi, program: bp, .. },
        )
        | (
            R::GaussianFunctional { source: a, source_identity: ai, program: ap, .. },
            R::GaussianFunctional { source: b, source_identity: bi, program: bp, .. },
        ) => a == b && ai == bi && ap == bp,
        _ => false,
    }
}

#[cfg(test)]
mod resource_tests {
    use super::*;
    #[test]
    fn cyclic_family_report_keeps_output_admitted_after_program_drop() {
        let pool = antecedent_core::execution::AdmissionPool::fixed(1, 1_000_000);
        let ctx = ExecutionContext::for_tests(3).with_admission(pool.clone());
        let request = R::OuStationary {
            model: antecedent_model::continuous_cyclic::OrnsteinUhlenbeck {
                drift: vec![-1.],
                center: vec![0.],
                diffusion: vec![2.],
            },
        };
        let program = CompiledCyclicFamilyProgram::compile(&request, &ctx).unwrap();
        let first = program.execute(&ctx).unwrap();
        let second = program.execute(&ctx).unwrap();
        assert_eq!(first, second);
        assert!(first.retained_admitted_bytes() > 0);
        drop(program);
        assert_eq!(
            pool.byte_budget() - pool.available().bytes(),
            first.retained_admitted_bytes() + second.retained_admitted_bytes()
        );
        drop(first);
        drop(second);
        assert_eq!(pool.available().bytes(), pool.byte_budget());
    }
    #[test]
    fn frozen_structure_cache_charges_full_failed_fit_slots() {
        use antecedent_estimate::cyclic_likelihood::{CyclicFit, CyclicStructureFit};
        let fit = CyclicStructureFit {
            fit: CyclicFit {
                model: antecedent_model::nonlinear_equilibrium::EquilibriumModel {
                    equations: vec![],
                    noise_covariance: vec![],
                },
                log_likelihood: 0.,
                objective: 0.,
                sweeps: 0,
                converged: false,
                evaluations: 0,
            },
            directed_edges: vec![],
            threshold: 0.1,
            l1_penalty: 0.,
            heldout_log_likelihood: None,
            selection_converged: false,
        };
        let bootstrap = (0..100).map(|_| None).collect::<Vec<Option<CyclicStructureFit>>>();
        let expected = crate::retained_allocation::typed_slots(&bootstrap).unwrap();
        let answer = A::StructureFit { fit, bootstrap };
        let ctx = ExecutionContext::for_tests(3);
        let cached = FrozenFit::capture(&answer, ctx.kernel_policy, &ctx).unwrap().unwrap();
        assert!(cached.retention.remainder().bytes() >= expected);
    }
}
