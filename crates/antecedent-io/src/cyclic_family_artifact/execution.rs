//! Family-specific dispatch for admitted cyclic requests.

use super::{
    CyclicActionWorldReport, CyclicFamilyAnswer, CyclicFamilyRequest, FrozenFamilyFitValue,
    compiled, invalid, ou_trajectory_error, sde_error, worlds,
};
use crate::cyclic_system_artifact::CyclicArtifactError;
use antecedent_core::ExecutionContext;

impl CyclicFamilyRequest {
    #[allow(
        clippy::too_many_lines,
        reason = "typed family dispatch preserves each route's inputs and error order"
    )]
    pub(super) fn execute_answer(
        &self,
        frozen: Option<&FrozenFamilyFitValue>,
        source_proof: Option<&crate::cyclic_functional::CheckedCyclicFunctionalSource>,
        linear_routes: Option<&antecedent_model::scc_compile::LinearRouteSet>,
        ctx: &ExecutionContext,
    ) -> Result<CyclicFamilyAnswer, CyclicArtifactError> {
        Ok(match self {
            Self::GaussianFunctional { law, assignments, .. } => {
                CyclicFamilyAnswer::GaussianFunctional(
                    crate::cyclic_gaussian_functional::evaluate_gaussian_cyclic_functional_checked(
                        source_proof
                            .ok_or_else(|| invalid("compiled functional source missing"))?,
                        law,
                        assignments,
                        match frozen {
                            Some(FrozenFamilyFitValue::Gaussian(density)) => Some(density),
                            _ => None,
                        },
                        ctx,
                    )?,
                )
            }
            Self::AdamFit { model, parameters, training, validation, policy } => {
                CyclicFamilyAnswer::AdamFit(
                    if let Some(FrozenFamilyFitValue::Adam(retained)) = frozen {
                        retained.clone()
                    } else {
                        antecedent_estimate::cyclic_adam::fit_cyclic_adam(
                            model, parameters, training, validation, *policy, ctx,
                        )?
                    },
                )
            }

            Self::EquilibriumRoots { model, domain, noise, clamps, policy } => {
                let system =
                    antecedent_model::equilibrium_polynomial::compile_polynomial_equilibrium(
                        model, domain, noise, clamps, ctx,
                    )?;
                CyclicFamilyAnswer::Polynomial(system.isolate(*policy, ctx)?)
            }

            Self::OuMechanisms { model, regimes, parameters, policy, worlds } => {
                if worlds.is_empty()
                    || worlds.len() > 128
                    || (parameters.is_empty() != regimes.is_empty())
                {
                    return Err(invalid("OU mechanism world/fit design"));
                }
                let fit = if parameters.is_empty() {
                    None
                } else if let Some(FrozenFamilyFitValue::Ou(retained)) = frozen {
                    retained.clone()
                } else {
                    Some(antecedent_estimate::ou_mechanisms::fit_ou_mechanisms(
                        model, parameters, regimes, *policy, ctx,
                    )?)
                };
                let fitted = fit.as_ref().map_or(model, |f| &f.model);
                let laws = worlds
                    .iter()
                    .map(|replacements| fitted.stationary_outcome(replacements, ctx))
                    .collect::<Result<Vec<_>, _>>()?;
                CyclicFamilyAnswer::OuMechanisms { fit, laws }
            }

            Self::Functional { law, assignments, .. } => CyclicFamilyAnswer::Functional(
                crate::cyclic_functional::evaluate_cyclic_functional_checked(
                    source_proof.ok_or_else(|| invalid("compiled functional source missing"))?,
                    law,
                    assignments,
                    ctx,
                )?,
            ),
            Self::GradientFit { model, parameters, rows, policy } => {
                CyclicFamilyAnswer::GradientFit(
                    if let Some(FrozenFamilyFitValue::Gradient(retained)) = frozen {
                        retained.clone()
                    } else {
                        antecedent_estimate::cyclic_gradient::fit_cyclic_gradient(
                            model, parameters, rows, *policy, ctx,
                        )?
                    },
                )
            }
            Self::PolynomialCausal { program } => {
                CyclicFamilyAnswer::PolynomialCausal(program.execute(ctx)?)
            }
            Self::FrequentistBootstrap {
                design,
                model,
                parameters,
                rows,
                policy,
                replicates,
                seed,
                worlds,
                noise,
                initial,
                solve_policy,
            } => {
                let count = replicates
                    .saturating_mul(worlds.len())
                    .saturating_mul(noise.len())
                    .saturating_mul(model.order());
                if *replicates == 0
                    || *replicates > 1000
                    || worlds.is_empty()
                    || noise.is_empty()
                    || count > 100_000
                    || rows.len() > 10_000
                {
                    return Err(invalid("frequentist bootstrap response/data limits"));
                }
                let mut context = ctx.clone();
                context.rng = antecedent_core::RngFactory::from_seed(*seed);
                let draws = if let Some(FrozenFamilyFitValue::Bootstrap(retained)) = frozen {
                    retained.clone()
                } else {
                    antecedent_estimate::cyclic_likelihood::bootstrap_cyclic_fit_with_design(
                        model,
                        parameters,
                        rows,
                        *policy,
                        *replicates,
                        *design,
                        &context,
                    )?
                };
                let mut output = Vec::with_capacity(worlds.len());
                for clamps in worlds {
                    let mut attempts = Vec::with_capacity(draws.len());
                    for draw in &draws {
                        let answer = draw
                            .as_ref()
                            .map(|values| {
                                let fitted =
                                    antecedent_estimate::cyclic_likelihood::with_parameters(
                                        model, parameters, values,
                                    )?;
                                noise
                                    .iter()
                                    .map(|u| {
                                        fitted.solve_outcome(u, clamps, initial, *solve_policy, ctx)
                                    })
                                    .collect::<Result<
                                        Vec<_>,
                                        antecedent_model::nonlinear_equilibrium::EquilibriumError,
                                    >>()
                            })
                            .transpose()?;
                        attempts.push(answer);
                    }
                    output.push(attempts);
                }
                CyclicFamilyAnswer::FrequentistBootstrap { parameters: draws, worlds: output }
            }
            Self::EquilibriumActions { model, worlds, noise, initial, factual, policy } => {
                model.validate()?;
                let count = worlds
                    .len()
                    .saturating_mul(noise.len() + usize::from(factual.is_some()))
                    .saturating_mul(model.order());
                if worlds.is_empty()
                    || worlds.len() > 256
                    || count > 100_000
                    || (noise.is_empty() && factual.is_none())
                {
                    return Err(invalid("action world/response limits"));
                }
                let mut ids = std::collections::BTreeSet::new();
                let mut plans = Vec::with_capacity(worlds.len());
                for world in worlds {
                    if world.id.trim().is_empty() || world.id.len() > 256 || !ids.insert(&world.id)
                    {
                        return Err(invalid("unique nonempty action world identities required"));
                    }
                    plans.push(model.compile_actions(&world.actions)?);
                }
                let factual_noise =
                    factual.as_ref().map(|state| model.abduce(state)).transpose()?;
                let mut output = Vec::with_capacity(worlds.len());
                for (world, plan) in worlds.iter().zip(plans) {
                    let rows = noise
                        .iter()
                        .map(|u| plan.solve(u, initial, *policy, ctx))
                        .collect::<Result<Vec<_>, _>>()?;
                    let counterfactual = factual_noise
                        .as_ref()
                        .map(|u| plan.solve(u, initial, *policy, ctx))
                        .transpose()?;
                    output.push(CyclicActionWorldReport {
                        id: world.id.clone(),
                        blocks: plan.blocks()?,
                        rows,
                        counterfactual,
                    });
                }
                CyclicFamilyAnswer::EquilibriumActions(output)
            }
            Self::PolynomialFit { candidates, measurements, policy } => {
                CyclicFamilyAnswer::PolynomialFit(
                    if let Some(FrozenFamilyFitValue::Polynomial(retained)) = frozen {
                        retained.clone()
                    } else {
                        antecedent_estimate::constraint_likelihood::fit_polynomial_candidates(
                            candidates,
                            measurements,
                            *policy,
                            ctx,
                        )?
                    },
                )
            }
            Self::ParameterPosterior {
                model,
                parameters,
                rows,
                prior_sd,
                sampling,
                worlds,
                noise,
                initial,
                solve_policy,
            } => {
                let count = sampling
                    .chains
                    .checked_mul(sampling.draws)
                    .and_then(|n| n.checked_mul(worlds.len()))
                    .and_then(|n| n.checked_mul(noise.len()))
                    .ok_or_else(|| invalid("posterior response size overflow"))?;
                if count > 100_000 || worlds.is_empty() || noise.is_empty() || rows.len() > 10_000 {
                    return Err(invalid("posterior response/data limits"));
                }
                let context = sampling.context(ctx);
                let posterior = if let Some(FrozenFamilyFitValue::Parameter(retained)) = frozen {
                    retained.clone()
                } else {
                    antecedent_estimate::cyclic_likelihood::sample_cyclic_posterior(
                        model,
                        parameters,
                        rows,
                        *prior_sd,
                        sampling.options(),
                        &context,
                    )?
                };
                let mut output = Vec::with_capacity(worlds.len());
                for clamps in worlds {
                    let mut by_parameter = Vec::new();
                    for draw in &posterior.samples {
                        let fitted = antecedent_estimate::cyclic_likelihood::with_parameters(
                            model, parameters, draw,
                        )?;
                        by_parameter.push(
                            noise
                                .iter()
                                .map(|noise| {
                                    fitted.solve_outcome(noise, clamps, initial, *solve_policy, ctx)
                                })
                                .collect::<Result<Vec<_>, _>>()?,
                        );
                    }
                    output.push(by_parameter);
                }
                CyclicFamilyAnswer::ParameterPosterior { posterior, worlds: output }
            }
            Self::JointCounterfactual {
                model,
                parameters,
                rows,
                prior_sd,
                observations,
                measurement_variances,
                initial,
                sampling,
                worlds,
                solve_policy,
            } => {
                let count = sampling
                    .chains
                    .checked_mul(sampling.draws)
                    .and_then(|draws| draws.checked_mul(worlds.len()))
                    .ok_or_else(|| invalid("joint counterfactual size overflow"))?;
                if count > 100_000 || worlds.is_empty() || rows.len() > 10_000 {
                    return Err(invalid("joint counterfactual retention/data limits"));
                }
                let posterior = if let Some(FrozenFamilyFitValue::Joint(retained)) = frozen {
                    retained.clone()
                } else {
                    antecedent_estimate::cyclic_likelihood::abduce_cyclic_joint(
                        model,
                        parameters,
                        rows,
                        *prior_sd,
                        observations,
                        measurement_variances,
                        initial,
                        sampling.options(),
                        &sampling.context(ctx),
                    )?
                };
                let mut output = Vec::with_capacity(worlds.len());
                for clamps in worlds {
                    let mut draws = Vec::with_capacity(posterior.parameter_draws.len());
                    for ((values, factual), noise) in posterior
                        .parameter_draws
                        .iter()
                        .zip(&posterior.factual.states)
                        .zip(&posterior.factual.disturbances)
                    {
                        let draw_model = antecedent_estimate::cyclic_likelihood::with_parameters(
                            model, parameters, values,
                        )?;
                        draws.push(draw_model.solve_outcome(
                            noise,
                            clamps,
                            factual,
                            *solve_policy,
                            ctx,
                        )?);
                    }
                    output.push(draws);
                }
                CyclicFamilyAnswer::JointCounterfactual { posterior, worlds: output }
            }
            Self::PartialCounterfactual {
                model,
                observations,
                measurement_variances,
                initial,
                sampling,
                worlds,
                solve_policy,
            } => {
                let count = sampling
                    .chains
                    .checked_mul(sampling.draws)
                    .and_then(|n| n.checked_mul(worlds.len()))
                    .ok_or_else(|| invalid("counterfactual size overflow"))?;
                if count > 100_000 || worlds.is_empty() {
                    return Err(invalid("counterfactual retention limit"));
                }
                let context = sampling.context(ctx);
                let abduction = if let Some(FrozenFamilyFitValue::Partial(retained)) = frozen {
                    retained.clone()
                } else {
                    antecedent_estimate::cyclic_likelihood::abduce_cyclic_partial(
                        model,
                        observations,
                        measurement_variances,
                        initial,
                        sampling.options(),
                        &context,
                    )?
                };
                let mut output = Vec::with_capacity(worlds.len());
                for clamps in worlds {
                    output.push(
                        abduction
                            .disturbances
                            .iter()
                            .zip(&abduction.states)
                            .map(|(noise, factual)| {
                                model.solve_outcome(noise, clamps, factual, *solve_policy, ctx)
                            })
                            .collect::<Result<Vec<_>, _>>()?,
                    );
                }
                CyclicFamilyAnswer::PartialCounterfactual { abduction, worlds: output }
            }
            Self::Graph { request } => CyclicFamilyAnswer::Graph(request.execute(ctx)?),
            Self::Ode { model, initial, times, schedule, policy } => {
                CyclicFamilyAnswer::Ode(model.trajectory(initial, times, schedule, *policy, ctx)?)
            }
            Self::OuStationary { model } => {
                CyclicFamilyAnswer::OuStationary(model.stationary_covariance(ctx)?)
            }
            Self::OuTransient { model, initial_mean, initial_covariance, times, policy } => {
                CyclicFamilyAnswer::OuTransient(model.moments(
                    initial_mean,
                    initial_covariance,
                    times,
                    *policy,
                    ctx,
                )?)
            }
            Self::StructureFit {
                seed,
                design,
                model,
                training,
                heldout,
                policy,
                threshold,
                bootstrap_replicates,
            } => {
                if let Some(FrozenFamilyFitValue::Structure { fit, bootstrap }) = frozen {
                    CyclicFamilyAnswer::StructureFit {
                        fit: fit.clone(),
                        bootstrap: bootstrap.clone(),
                    }
                } else {
                    let fit = antecedent_estimate::cyclic_likelihood::learn_cyclic_structure(
                        model, training, heldout, *policy, *threshold, ctx,
                    )?;
                    let mut context = ctx.clone();
                    context.rng = antecedent_core::RngFactory::from_seed(*seed);
                    let bootstrap =
                    antecedent_estimate::cyclic_likelihood::bootstrap_cyclic_structure_with_design(
                        model,
                        training,
                        heldout,
                        *policy,
                        *threshold,
                        *bootstrap_replicates,
                        *design,
                        &context,
                    )?;
                    CyclicFamilyAnswer::StructureFit { fit, bootstrap }
                }
            }
            Self::OdeFit { model, parameters, observations, integration, policy } => {
                CyclicFamilyAnswer::OdeFit(
                    if let Some(FrozenFamilyFitValue::Ode(retained)) = frozen {
                        retained.clone()
                    } else {
                        antecedent_estimate::cyclic_dynamics::fit_ode_likelihood(
                            model,
                            parameters,
                            observations,
                            *integration,
                            *policy,
                            ctx,
                        )?
                    },
                )
            }
            Self::OuFit { model, parameters, rows, policy } => CyclicFamilyAnswer::OuFit(
                if let Some(FrozenFamilyFitValue::OuFit(retained)) = frozen {
                    retained.clone()
                } else {
                    antecedent_estimate::cyclic_dynamics::fit_ou_likelihood(
                        model, parameters, rows, *policy, ctx,
                    )?
                },
            ),
            Self::OuTrajectoryFit {
                model,
                observation,
                parameters,
                trajectories,
                policy,
                predictions,
                stationary_worlds,
            } => {
                if predictions.len() > 128 || stationary_worlds.len() > 128 {
                    return Err(invalid("OU trajectory prediction design count"));
                }
                let fit = if let Some(FrozenFamilyFitValue::OuTrajectory(retained)) = frozen {
                    retained.clone()
                } else {
                    antecedent_estimate::ou_trajectory::fit_ou_trajectories(
                        model,
                        observation,
                        parameters,
                        trajectories,
                        *policy,
                        ctx,
                    )
                    .map_err(ou_trajectory_error)?
                };
                CyclicFamilyAnswer::OuTrajectory(
                    antecedent_estimate::ou_trajectory::ou_trajectory_report(
                        fit,
                        observation,
                        predictions,
                        stationary_worlds,
                        ctx,
                    )
                    .map_err(ou_trajectory_error)?,
                )
            }
            Self::SdeAnalysis {
                model,
                initial,
                simulation,
                interventions,
                contrasts,
                parameters,
                observation,
                filter_prior,
                trajectories,
                fit_policy,
                parameter_draws,
                parameter_seed,
            } => {
                let compiled = model.compile().map_err(|e| sde_error(e.into()))?;
                let fit = if let Some(FrozenFamilyFitValue::Sde(retained)) = frozen {
                    Some(retained.clone())
                } else if trajectories.is_empty() {
                    None
                } else {
                    Some(
                        antecedent_estimate::nonlinear_sde_fit::fit_sde_quasi_likelihood(
                            &compiled,
                            parameters,
                            observation.as_ref(),
                            filter_prior.as_ref(),
                            trajectories,
                            fit_policy.unwrap_or_default(),
                            ctx,
                        )
                        .map_err(sde_error)?,
                    )
                };
                CyclicFamilyAnswer::SdeAnalysis(
                    antecedent_estimate::nonlinear_sde_fit::sde_report(
                        &compiled,
                        fit,
                        initial,
                        simulation,
                        interventions,
                        contrasts,
                        *parameter_draws,
                        *parameter_seed,
                        ctx,
                    )
                    .map_err(sde_error)?,
                )
            }
            Self::Polynomial { model, policy } => {
                CyclicFamilyAnswer::Polynomial(model.isolate(*policy, ctx)?)
            }
            Self::RecoveredLinear {
                nodes,
                draws,
                worlds,
                outcome,
                source_id,
                certifying_evidence,
                uncertainty,
            } => compiled::execute_recovered_linear(
                nodes,
                draws,
                uncertainty,
                worlds,
                *outcome,
                source_id,
                certifying_evidence,
            )?,
            Self::RetainedExperimentalRecovery { inputs, worlds, outcome, .. } => {
                worlds::execute_retained_recovery(
                    inputs.law(),
                    worlds,
                    *outcome,
                    inputs.source_id(),
                    inputs.certifying_evidence(),
                    ctx,
                )?
            }
            Self::ExperimentalRecovery {
                regimes,
                dropped,
                samples,
                options,
                inference,
                worlds,
                outcome,
                source_id,
                certifying_evidence,
            } => worlds::execute_experimental_recovery(
                regimes,
                dropped,
                samples,
                *options,
                inference,
                worlds,
                *outcome,
                source_id,
                certifying_evidence,
                match frozen {
                    Some(FrozenFamilyFitValue::Recovered(law)) => Some(law),
                    _ => None,
                },
                ctx,
            )?,
            Self::LinearWorlds { variables, model, component, worlds, certificate } => {
                worlds::execute_linear_worlds(
                    variables,
                    model,
                    component,
                    worlds,
                    *certificate,
                    linear_routes,
                    ctx,
                )?
            }
            Self::EquilibriumWorlds { variables, model, worlds, noise, initial, policy } => {
                worlds::execute_equilibrium_worlds(
                    variables, model, worlds, noise, initial, *policy, ctx,
                )?
            }
            Self::OdeWorlds { variables, model, initial, initial_laws, times, worlds, policy } => {
                worlds::execute_ode_worlds(
                    variables,
                    model,
                    initial,
                    initial_laws,
                    times,
                    worlds,
                    *policy,
                    ctx,
                )?
            }
            Self::DynamicInference { problem, method, quantities } => {
                worlds::execute_dynamic_inference(
                    problem,
                    method,
                    quantities,
                    match frozen {
                        Some(FrozenFamilyFitValue::Dynamic(fit)) => Some(fit),
                        _ => None,
                    },
                    ctx,
                )?
            }
            Self::System { request }
                if matches!(frozen, Some(FrozenFamilyFitValue::SystemOwner(_))) =>
            {
                let Some(FrozenFamilyFitValue::SystemOwner(owner)) = frozen else { unreachable!() };
                CyclicFamilyAnswer::System(std::sync::Arc::new(if owner.request() == request {
                    owner.estimate(ctx)?
                } else {
                    owner.rebind(request.clone(), ctx)?.estimate(ctx)?
                }))
            }
            Self::System { request } => {
                let fit = match (&request.fit, frozen) {
                    (None, _) => None,
                    (Some(_), Some(FrozenFamilyFitValue::System(retained))) => {
                        Some(retained.clone())
                    }
                    (Some(recipe), _) => {
                        Some(antecedent_estimate::cyclic_likelihood::fit_cyclic_likelihood(
                            &request.model,
                            &recipe.parameters,
                            &recipe.rows,
                            recipe.options,
                            ctx,
                        )?)
                    }
                };
                CyclicFamilyAnswer::System(std::sync::Arc::new(
                    request.execute_bound(fit.as_ref(), ctx)?,
                ))
            }
        })
    }
}
