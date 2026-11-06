//! Projection of selectors from native cyclic answers.

use super::{
    A, Axis, CyclicArtifactError, CyclicEvidenceSelector, EquilibriumOutcome, EvidencePort,
    EvidenceSource, Functional, GaussianWorld, InformationDetail, Interval, PolynomialWorldDetail,
    PosteriorDetail, Projection, R, RecoveredWithOrigin, RootDetail, SamplingLawDetail,
    SolutionSetDetail, Time, UnsolvedWorld, WithinDraw, World, WorldContext, detail, invalid,
    point, selector_mismatch, state, supported_state,
};

/// Draw-aligned projection of one recovered world.
///
/// Supplied draws declare no meaning and stay on the interventional port. Draws
/// from a declared sampling law or posterior project the per-draw interventional
/// mean onto that port: parameter uncertainty is not relabelled as outcome
/// variation, and a sampling law is not read as a posterior.
fn recovered_response<'a>(
    found: &'a antecedent_io::cyclic_family_artifact::RecoveredWorldAnswer,
    origin: &'a antecedent_io::cyclic_family_artifact::RecoveredDrawOriginWire,
    world: usize,
    variable: usize,
) -> Result<Projection<'a>, CyclicArtifactError> {
    let mut samples = Vec::with_capacity(found.draws.len());
    let mut masses = Vec::with_capacity(found.draws.len());
    for draw in &found.draws {
        masses.push(antecedent_estimate::polynomial_scm::MassBounds {
            lower: draw.weight,
            upper: draw.weight,
        });
        samples.push(match &draw.outcome {
            antecedent_io::cyclic_family_artifact::RecoveredDrawAnswer::Equilibrium {
                mean,
                ..
            } => Some(*mean.get(variable).ok_or_else(selector_mismatch)?),
            antecedent_io::cyclic_family_artifact::RecoveredDrawAnswer::Inadmissible { .. }
            | antecedent_io::cyclic_family_artifact::RecoveredDrawAnswer::NotCertified { .. } => {
                None
            }
        });
    }
    let (port, axis, functional) = match origin.law_meaning() {
        None => {
            (EvidencePort::InterventionalLaw, Axis::Predictive, Functional::RecoveredIntervention)
        }
        Some("sampling_law") => {
            (EvidencePort::SamplingLaw, Axis::Sampling, Functional::RecoveredInterventionalMean)
        }
        Some(_) => (
            EvidencePort::ParameterPosterior,
            Axis::Parameter,
            Functional::RecoveredInterventionalMean,
        ),
    };
    let mut p = point(None, variable, World::indexed(world), Time::Equilibrium, functional, port);
    p.samples = samples;
    p.masses = Some(masses);
    p.axes = vec![axis];
    p.diagnostics = if origin.is_supplied() {
        detail(found)
    } else {
        detail(RecoveredWithOrigin { world: found, origin })
    };
    Ok(p)
}

/// The interventional law within one recovered draw.
fn recovered_within(
    found: &antecedent_io::cyclic_family_artifact::RecoveredWorldAnswer,
    world: usize,
    draw: usize,
    variable: usize,
) -> Result<Projection<'_>, CyclicArtifactError> {
    let item = found.draws.get(draw).ok_or_else(selector_mismatch)?;
    let (value, variance) = match &item.outcome {
        antecedent_io::cyclic_family_artifact::RecoveredDrawAnswer::Equilibrium {
            mean,
            covariance,
            ..
        } => {
            let n = mean.len();
            let value = *mean.get(variable).ok_or_else(selector_mismatch)?;
            (Some(value), covariance.get(variable * n + variable).copied())
        }
        _ => (None, None),
    };
    let mut p = point(
        value,
        variable,
        World::Draw { world, draw },
        Time::Equilibrium,
        Functional::InterventionalOutcome,
        EvidencePort::InterventionalLaw,
    );
    p.axes = vec![Axis::Predictive];
    p.masses = Some(vec![antecedent_estimate::polynomial_scm::MassBounds {
        lower: item.weight,
        upper: item.weight,
    }]);
    p.diagnostics = detail(WithinDraw { variance, draw: item });
    Ok(p)
}

/// Projection of one typed world of a specified model.
fn typed_world(
    found: &antecedent_io::cyclic_family_artifact::CyclicWorldAnswer,
    variable: usize,
    time: Option<usize>,
) -> Result<Projection<'_>, CyclicArtifactError> {
    use antecedent_io::cyclic_family_artifact::CyclicWorldOutcomeAnswer as O;
    if variable >= found.variables.len() {
        return Err(selector_mismatch());
    }
    let label = World::Indexed(u64::from(found.world));
    let mut p = match (&found.outcome, time) {
        (O::GaussianLaw { mean, covariance, .. }, None) => {
            let n = mean.len();
            let mut p = point(
                Some(*mean.get(variable).ok_or_else(selector_mismatch)?),
                variable,
                label,
                Time::Equilibrium,
                Functional::InterventionalMean,
                EvidencePort::InterventionalLaw,
            );
            p.axes = vec![Axis::Predictive];
            p.diagnostics = detail(GaussianWorld {
                variance: covariance.get(variable * n + variable),
                world: WorldContext(found),
            });
            p
        }
        (O::EquilibriumDraws { outcomes }, None) => {
            let mut p = point(
                None,
                variable,
                label,
                Time::Equilibrium,
                Functional::SpecifiedNoiseResponse,
                EvidencePort::Scenario,
            );
            p.samples = outcomes.iter().map(|outcome| state(outcome, variable)).collect();
            p.diagnostics = detail(WorldContext(found));
            p
        }
        (O::Trajectory { trajectory }, Some(index)) => {
            let value = trajectory
                .states
                .get(index)
                .and_then(|row| row.get(variable))
                .copied()
                .ok_or_else(selector_mismatch)?;
            let mut p = point(
                Some(value),
                variable,
                label,
                Time::Step(index),
                Functional::TrajectoryState,
                EvidencePort::Scenario,
            );
            p.diagnostics = detail(WorldContext(found));
            p
        }
        (O::NoSolution { .. } | O::Withheld, _) => {
            let mut p = point(
                None,
                variable,
                label,
                Time::Equilibrium,
                Functional::NoSolution,
                EvidencePort::Scenario,
            );
            p.diagnostics =
                detail(UnsolvedWorld { outcome: &found.outcome, world: WorldContext(found) });
            p
        }
        _ => return Err(selector_mismatch()),
    };
    p.bounds.clear();
    Ok(p)
}

/// Endogenous coordinates of one dynamic quantity.
const fn quantity_coordinates(
    quantity: &antecedent_estimate::cyclic_dynamic_inference::DynamicQuantity,
) -> (usize, Option<usize>) {
    use antecedent_estimate::cyclic_dynamic_inference::DynamicQuantity as Q;
    match quantity {
        Q::OdeState { variable, .. }
        | Q::OuStationaryMean { variable, .. }
        | Q::OuMechanismMean { variable, .. } => (*variable, None),
        Q::OuStationaryCovariance { variable, other, .. }
        | Q::OuMechanismCovariance { variable, other, .. } => (*variable, Some(*other)),
    }
}

/// Projection of one declared quantity of a fitted continuous-time family.
fn dynamic_quantity<'a>(
    quantities: &[antecedent_estimate::cyclic_dynamic_inference::DynamicQuantity],
    inference: &'a antecedent_io::cyclic_family_artifact::DynamicInferenceAnswer,
    quantity: usize,
) -> Result<Projection<'a>, CyclicArtifactError> {
    use antecedent_estimate::cyclic_dynamic_inference::DynamicQuantity as Q;
    use antecedent_io::cyclic_family_artifact::DynamicInferenceAnswer as D;
    let spec = quantities.get(quantity).ok_or_else(selector_mismatch)?;
    let (variable, other) = quantity_coordinates(spec);
    let time = match spec {
        Q::OdeState { time, .. } => Time::Instant(*time),
        _ => Time::Stationary,
    };
    let label = World::Quantity(quantity);
    let functional =
        if other.is_some() { Functional::DynamicCovariance } else { Functional::DynamicQuantity };
    let mut p = match inference {
        D::SamplingLaw(law) => {
            let mut p = point(None, variable, label, time, functional, EvidencePort::SamplingLaw);
            p.samples = law
                .replicates
                .iter()
                .map(|replicate| replicate.quantities.get(quantity).copied().flatten())
                .collect();
            p.masses = Some(
                law.weights
                    .iter()
                    .map(|weight| antecedent_estimate::polynomial_scm::MassBounds {
                        lower: *weight,
                        upper: *weight,
                    })
                    .collect(),
            );
            p.axes = vec![Axis::Sampling];
            p.diagnostics = detail(SamplingLawDetail {
                estimate: law.quantity_estimates.get(quantity),
                standard_error: law.quantity_standard_errors.get(quantity),
                failed_mass: law.quantity_failed_mass.get(quantity),
                replicate_failed_mass: &law.failed_mass,
                method: &law.method,
            });
            p
        }
        D::Information(information) => {
            let mut p = point(
                information.quantity_estimates.get(quantity).copied().flatten(),
                variable,
                label,
                time,
                functional,
                EvidencePort::CovarianceInfluence,
            );
            p.axes = vec![Axis::Sampling];
            p.diagnostics = detail(InformationDetail {
                standard_error: information.quantity_standard_errors.get(quantity),
                regularity: &information.regularity,
            });
            p
        }
        D::Posterior(posterior) => {
            let mut p =
                point(None, variable, label, time, functional, EvidencePort::ParameterPosterior);
            p.samples = posterior
                .quantity_draws
                .iter()
                .map(|draw| draw.get(quantity).copied().flatten())
                .collect();
            p.axes = vec![Axis::Parameter];
            p.diagnostics = detail(PosteriorDetail {
                failed_mass: posterior.quantity_failed_mass.get(quantity),
                max_rhat: &posterior.max_rhat,
                min_bulk_ess: &posterior.min_bulk_ess,
                min_tail_ess: &posterior.min_tail_ess,
                diagnostics_pass: &posterior.diagnostics_pass,
            });
            p
        }
    };
    p.other = other;
    Ok(p)
}

/// The fitted log likelihood, the only coordinate a point fit offers.
fn log_likelihood<'a>(
    value: f64,
    time: Time,
    selector: &CyclicEvidenceSelector,
) -> Result<Projection<'a>, CyclicArtifactError> {
    let CyclicEvidenceSelector::Likelihood { candidate: None } = selector else {
        return Err(selector_mismatch());
    };
    Ok(point(
        Some(value),
        0,
        World::Fixed("training"),
        time,
        Functional::LogLikelihood,
        EvidencePort::Likelihood,
    ))
}

/// Named structural-action worlds: the supplied-noise rows or the abducted factual response.
fn actions<'a>(
    worlds: &'a [antecedent_io::cyclic_family_artifact::CyclicActionWorldReport],
    selector: &CyclicEvidenceSelector,
) -> Result<Projection<'a>, CyclicArtifactError> {
    let CyclicEvidenceSelector::Action { world, noise_row, variable, counterfactual } = selector
    else {
        return Err(selector_mismatch());
    };
    let w = worlds.get(*world).ok_or_else(selector_mismatch)?;
    let value = if *counterfactual { w.counterfactual.as_ref() } else { w.rows.get(*noise_row) }
        .ok_or_else(selector_mismatch)?;
    let mut p = point(
        state(value, *variable),
        *variable,
        World::Declared(w.id.clone()),
        Time::Equilibrium,
        if *counterfactual {
            Functional::FactualCounterfactual
        } else {
            Functional::SpecifiedNoiseResponse
        },
        EvidencePort::Scenario,
    );
    // A residual-checked numeric state is not an outer enclosure of all solutions.
    p.bounds.clear();
    p.diagnostics = detail(value);
    Ok(p)
}

/// Parameter posterior responses, `(world, parameter_draw, noise_row)`.
fn parameter_posterior<'a>(
    worlds: &[Vec<Vec<EquilibriumOutcome>>],
    selector: &CyclicEvidenceSelector,
) -> Result<Projection<'a>, CyclicArtifactError> {
    let CyclicEvidenceSelector::Response { world, noise_row, variable } = selector else {
        return Err(selector_mismatch());
    };
    let draws = worlds.get(*world).ok_or_else(selector_mismatch)?;
    let values = draws
        .iter()
        .map(|draw| {
            draw.get(*noise_row)
                .map(|value| supported_state(value, *variable))
                .ok_or_else(selector_mismatch)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut p = point(
        None,
        *variable,
        World::Noise { world: *world, noise_row: *noise_row },
        Time::Equilibrium,
        Functional::ConditionalStructuralResponse,
        EvidencePort::ParameterPosterior,
    );
    p.samples = values;
    p.axes = vec![Axis::Parameter];
    Ok(p)
}

/// Counterfactual predictions sharing each factual disturbance across worlds.
/// `joint` marks the joint parameter/factual posterior, which adds the parameter axis.
fn counterfactual_responses<'a>(
    worlds: &[Vec<EquilibriumOutcome>],
    joint: bool,
    selector: &CyclicEvidenceSelector,
) -> Result<Projection<'a>, CyclicArtifactError> {
    let CyclicEvidenceSelector::Response { world, noise_row, variable } = selector else {
        return Err(selector_mismatch());
    };
    if *noise_row != 0 {
        return Err(selector_mismatch());
    }
    let draws = worlds.get(*world).ok_or_else(selector_mismatch)?;
    let mut p = point(
        None,
        *variable,
        World::indexed(*world),
        Time::Equilibrium,
        Functional::Outcome,
        EvidencePort::InterventionalLaw,
    );
    p.samples = draws.iter().map(|v| supported_state(v, *variable)).collect();
    p.axes = vec![Axis::Predictive];
    if joint {
        p.axes.push(Axis::Parameter);
    }
    Ok(p)
}

/// Conditional frequentist bootstrap responses; a failed fit keeps its slot.
fn bootstrap<'a>(
    worlds: &[Vec<Option<Vec<EquilibriumOutcome>>>],
    selector: &CyclicEvidenceSelector,
) -> Result<Projection<'a>, CyclicArtifactError> {
    let CyclicEvidenceSelector::Response { world, noise_row, variable } = selector else {
        return Err(selector_mismatch());
    };
    let draws = worlds.get(*world).ok_or_else(selector_mismatch)?;
    let mut p = point(
        None,
        *variable,
        World::Noise { world: *world, noise_row: *noise_row },
        Time::Equilibrium,
        Functional::ConditionalStructuralResponse,
        EvidencePort::SamplingLaw,
    );
    p.samples = draws
        .iter()
        .map(|draw| match draw {
            Some(rows) => rows
                .get(*noise_row)
                .map(|v| supported_state(v, *variable))
                .ok_or_else(selector_mismatch),
            None => Ok(None),
        })
        .collect::<Result<Vec<_>, _>>()?;
    p.axes = vec![Axis::Sampling];
    Ok(p)
}

/// Stationary laws of the shared-mechanism family in the requested equation worlds.
/// Trajectory OU fit: its likelihood, held-out trajectory predictions and stationary worlds.
#[allow(clippy::too_many_lines, reason = "the projection keeps the trajectory variants together")]
fn ou_trajectory<'a>(
    report: &'a antecedent_estimate::ou_trajectory::OuTrajectoryReport,
    selector: &CyclicEvidenceSelector,
) -> Result<Projection<'a>, CyclicArtifactError> {
    let uncertainty_note = "process variability is the value's covariance; sampling uncertainty of the fitted parameters is the parameter variance, a distinct meaning that is never added to it";
    match selector {
        CyclicEvidenceSelector::Likelihood { candidate: None } => {
            let mut projection =
                log_likelihood(report.fit.log_likelihood, Time::Trajectory, selector)?;
            // Fitting, identification and numerical stability stay three separate
            // records; none implies another.
            projection.diagnostics = detail(serde_json::json!({
                "fitting": report.fit.fitting,
                "identification": report.fit.identification,
                "numerics": report.fit.numerics,
            }));
            Ok(projection)
        }
        CyclicEvidenceSelector::OuTrajectoryState { design, time, variable, other } => {
            let prediction = report.predictions.get(*design).ok_or_else(selector_mismatch)?;
            let at = *prediction.times.get(*time).ok_or_else(selector_mismatch)?;
            let mean = prediction.state_mean.get(*time).ok_or_else(selector_mismatch)?;
            let n = mean.len();
            if *variable >= n || other.is_some_and(|j| j >= n) {
                return Err(selector_mismatch());
            }
            let covariance_mode = other.is_some();
            let j = other.unwrap_or(*variable);
            let value = if covariance_mode {
                prediction.state_covariance[*time][*variable * n + j]
            } else {
                mean[*variable]
            };
            let sampling = if covariance_mode {
                prediction
                    .state_covariance_parameter_variance
                    .as_ref()
                    .map(|v| v[*time][*variable * n + j])
            } else {
                prediction
                    .state_mean_parameter_covariance
                    .as_ref()
                    .map(|v| v[*time][*variable * n + j])
            };
            let mut p = point(
                Some(value),
                *variable,
                World::Declared(prediction.id.clone()),
                Time::At(at),
                if covariance_mode { Functional::PredictiveCovariance } else { Functional::Mean },
                if covariance_mode { EvidencePort::Likelihood } else { EvidencePort::Mean },
            );
            p.other = *other;
            p.diagnostics = detail(serde_json::json!({
                "parameter_variance": sampling,
                "meaning": uncertainty_note,
            }));
            Ok(p)
        }
        CyclicEvidenceSelector::OuTrajectoryStationary { world, variable, other } => {
            let law = report.stationary.get(*world).ok_or_else(selector_mismatch)?;
            let functional =
                if other.is_some() { Functional::PredictiveCovariance } else { Functional::Mean };
            let port = if other.is_some() { EvidencePort::Likelihood } else { EvidencePort::Mean };
            let n = law.mean.len();
            if law.failure.is_some() {
                let mut p = point(
                    None,
                    *variable,
                    World::Declared(law.id.clone()),
                    Time::Stationary,
                    functional,
                    port,
                );
                p.diagnostics = detail(serde_json::json!({ "failure": law.failure }));
                return Ok(p);
            }
            if *variable >= n || other.is_some_and(|j| j >= n) {
                return Err(selector_mismatch());
            }
            let j = other.unwrap_or(*variable);
            let (value, sampling) = if other.is_some() {
                (
                    law.covariance[*variable * n + j],
                    law.covariance_parameter_variance.as_ref().map(|v| v[*variable * n + j]),
                )
            } else {
                (
                    law.mean[*variable],
                    law.mean_parameter_covariance.as_ref().map(|v| v[*variable * n + j]),
                )
            };
            let mut p = point(
                Some(value),
                *variable,
                World::Declared(law.id.clone()),
                Time::Stationary,
                functional,
                port,
            );
            p.other = *other;
            p.diagnostics = detail(serde_json::json!({
                "parameter_variance": sampling,
                "meaning": uncertainty_note,
            }));
            Ok(p)
        }
        _ => Err(selector_mismatch()),
    }
}
fn ou_mechanisms<'a>(
    laws: &'a [antecedent_estimate::ou_mechanisms::OuMechanismOutcome],
    fit_log_likelihood: Option<f64>,
    selector: &CyclicEvidenceSelector,
) -> Result<Projection<'a>, CyclicArtifactError> {
    match selector {
        CyclicEvidenceSelector::OuRegime { world, variable, other } => {
            let outcome = laws.get(*world).ok_or_else(selector_mismatch)?;
            let functional =
                if other.is_some() { Functional::PredictiveCovariance } else { Functional::Mean };
            let port = if other.is_some() { EvidencePort::Likelihood } else { EvidencePort::Mean };
            let antecedent_estimate::ou_mechanisms::OuMechanismOutcome::Stationary(law) = outcome
            else {
                let mut p = point(
                    None,
                    *variable,
                    World::Equation(*world),
                    Time::Stationary,
                    functional,
                    port,
                );
                p.diagnostics = detail(outcome);
                return Ok(p);
            };
            let n = law.mean.len();
            if *variable >= n || other.is_some_and(|j| j >= n) {
                return Err(selector_mismatch());
            }
            Ok(point(
                Some(other.map_or(law.mean[*variable], |j| law.covariance[*variable * n + j])),
                *variable,
                World::Equation(*world),
                Time::Stationary,
                functional,
                port,
            ))
        }
        CyclicEvidenceSelector::Likelihood { candidate: None } => {
            let value = fit_log_likelihood.ok_or_else(selector_mismatch)?;
            log_likelihood(value, Time::Stationary, selector)
        }
        _ => Err(selector_mismatch()),
    }
}

/// Deterministic ODE path.
fn ode_path<'a>(
    path: &antecedent_model::continuous_cyclic::OdeTrajectory,
    selector: &CyclicEvidenceSelector,
) -> Result<Projection<'a>, CyclicArtifactError> {
    let CyclicEvidenceSelector::Ode { time, variable } = selector else {
        return Err(selector_mismatch());
    };
    Ok(point(
        Some(*path.states.get(*time).and_then(|s| s.get(*variable)).ok_or_else(selector_mismatch)?),
        *variable,
        World::Fixed("initial_value_world"),
        Time::At(*path.times.get(*time).ok_or_else(selector_mismatch)?),
        Functional::ConditionalTrajectory,
        EvidencePort::Mean,
    ))
}

/// Stationary OU moments, read from the declared model and the stored covariance.
fn ou_stationary<'a>(
    request: &R,
    covariance: &[f64],
    selector: &CyclicEvidenceSelector,
) -> Result<Projection<'a>, CyclicArtifactError> {
    let R::OuStationary { model } = request else {
        return Err(selector_mismatch());
    };
    match selector {
        CyclicEvidenceSelector::OuMean { time: None, variable } => Ok(point(
            Some(*model.center.get(*variable).ok_or_else(selector_mismatch)?),
            *variable,
            World::Fixed("stationary"),
            Time::Stationary,
            Functional::Mean,
            EvidencePort::Mean,
        )),
        CyclicEvidenceSelector::OuCovariance { time: None, variable, other } => {
            let n = model.center.len();
            if *variable >= n || *other >= n {
                return Err(selector_mismatch());
            }
            Ok(point(
                Some(covariance[*variable * n + other]),
                *variable,
                World::Fixed("stationary"),
                Time::Stationary,
                Functional::PredictiveCovariance,
                EvidencePort::Likelihood,
            ))
        }
        _ => Err(selector_mismatch()),
    }
}

/// Finite-horizon OU mean and covariance.
fn ou_transient<'a>(
    path: &antecedent_model::continuous_cyclic::OuMoments,
    selector: &CyclicEvidenceSelector,
) -> Result<Projection<'a>, CyclicArtifactError> {
    match selector {
        CyclicEvidenceSelector::OuMean { time: Some(time), variable } => Ok(point(
            Some(
                *path
                    .means
                    .get(*time)
                    .and_then(|s| s.get(*variable))
                    .ok_or_else(selector_mismatch)?,
            ),
            *variable,
            World::Fixed("initial_law_world"),
            Time::At(*path.times.get(*time).ok_or_else(selector_mismatch)?),
            Functional::Mean,
            EvidencePort::Mean,
        )),
        CyclicEvidenceSelector::OuCovariance { time: Some(time), variable, other } => {
            let row = path.means.get(*time).ok_or_else(selector_mismatch)?;
            let n = row.len();
            if *variable >= n || *other >= n {
                return Err(selector_mismatch());
            }
            let covariance = path.covariances.get(*time).ok_or_else(selector_mismatch)?;
            Ok(point(
                Some(*covariance.get(*variable * n + other).ok_or_else(selector_mismatch)?),
                *variable,
                World::Fixed("initial_law_world"),
                Time::At(*path.times.get(*time).ok_or_else(selector_mismatch)?),
                Functional::PredictiveCovariance,
                EvidencePort::Likelihood,
            ))
        }
        _ => Err(selector_mismatch()),
    }
}

/// Nonlinear SDE ensembles and the quasi-likelihood fit behind them.
///
/// The world mean carries its Monte Carlo standard error, the step-halving evidence and the
/// parameter-draw spread as diagnostics, three separate quantities that are never summed;
/// none of them is a confidence or credible interval of the mean.
fn sde_analysis<'a>(
    report: &antecedent_estimate::nonlinear_sde_fit::SdeReport,
    selector: &CyclicEvidenceSelector,
) -> Result<Projection<'a>, CyclicArtifactError> {
    match selector {
        CyclicEvidenceSelector::Likelihood { candidate: None } => {
            let fit = report.fit.as_ref().ok_or_else(selector_mismatch)?;
            let mut projection =
                log_likelihood(fit.log_quasi_likelihood, Time::Trajectory, selector)?;
            projection.diagnostics = detail(serde_json::json!({
                "meaning": "Euler quasi log likelihood, a pseudo-likelihood and not the exact transition likelihood",
                "fitting": fit.fitting,
                "identification": fit.identification,
                "discretization": fit.discretization,
            }));
            Ok(projection)
        }
        CyclicEvidenceSelector::World { world, variable, time } => {
            let simulation = &report.simulation;
            let ensemble = simulation.worlds.get(*world as usize).ok_or_else(selector_mismatch)?;
            let index = time.unwrap_or(simulation.times.len().saturating_sub(1));
            let row =
                |grid: &Vec<Vec<f64>>| grid.get(index).and_then(|r| r.get(*variable)).copied();
            let value = row(&ensemble.mean).ok_or_else(selector_mismatch)?;
            let at = *simulation.times.get(index).ok_or_else(selector_mismatch)?;
            let numerical = ensemble.numerical.as_ref().map(|n| {
                serde_json::json!({
                    "refined_step": n.refined_step,
                    "weak_difference": n.weak_difference.get(index).and_then(|r| r.get(*variable)),
                    "weak_difference_standard_error": n.weak_difference_standard_error.get(index).and_then(|r| r.get(*variable)),
                    "step_converged": n.step_converged,
                })
            });
            let spread = report.parameter_uncertainty.as_ref().and_then(|u| {
                u.worlds.iter().find(|w| w.id == ensemble.id).and_then(|w| {
                    w.mean_between_draw_variance.get(index).and_then(|r| r.get(*variable)).copied()
                })
            });
            let mut p = point(
                Some(value),
                *variable,
                World::Indexed(u64::from(*world)),
                Time::At(at),
                Functional::Mean,
                EvidencePort::Mean,
            );
            p.diagnostics = detail(serde_json::json!({
                "world": ensemble.id,
                "monte_carlo_standard_error": row(&ensemble.monte_carlo_standard_error),
                "variance": row(&ensemble.variance),
                "numerical_approximation": numerical,
                "parameter_between_draw_variance": spread,
                "scheme": simulation.scheme,
                "step": simulation.step,
                "seed": simulation.seed,
                "paths": simulation.paths,
            }));
            Ok(p)
        }
        _ => Err(selector_mismatch()),
    }
}

/// Certified roots and unresolved regions of a bounded polynomial system.
fn polynomial_solutions<'a>(
    solutions: &antecedent_model::equation_system::IsolatedSolutions,
    selector: &CyclicEvidenceSelector,
) -> Result<Projection<'a>, CyclicArtifactError> {
    match selector {
        CyclicEvidenceSelector::Root { root, variable } => {
            let enclosure = *solutions
                .roots
                .get(*root)
                .and_then(|s| s.get(*variable))
                .ok_or_else(selector_mismatch)?;
            let mut p = point(
                None,
                *variable,
                World::Root(*root),
                Time::Algebraic,
                Functional::SolutionCoordinate,
                EvidencePort::Scenario,
            );
            p.bounds = vec![Some(enclosure)];
            p.diagnostics = detail(RootDetail {
                unresolved_regions: solutions.unresolved.len(),
                selection_law: false,
            });
            Ok(p)
        }
        CyclicEvidenceSelector::SolutionSet { variable } => {
            let mut p = point(
                None,
                *variable,
                World::Fixed("all_model_solutions"),
                Time::Algebraic,
                Functional::ModelSolutionSet,
                EvidencePort::Scenario,
            );
            p.samples.clear();
            p.bounds = solutions
                .roots
                .iter()
                .chain(&solutions.unresolved)
                .map(|region| {
                    region.get(*variable).copied().map(Some).ok_or_else(selector_mismatch)
                })
                .collect::<Result<Vec<_>, _>>()?;
            p.diagnostics = detail(SolutionSetDetail {
                certified_roots: solutions.roots.len(),
                unresolved_regions: solutions.unresolved.len(),
                selection_law: false,
                outer_enclosure: true,
            });
            Ok(p)
        }
        _ => Err(selector_mismatch()),
    }
}

/// The outcomes of the event a functional projection reads: one for a scalar coordinate,
/// several for a joint event.
fn event_outcomes(
    source: &antecedent_io::cyclic_graph::CyclicGraphRequest,
    message: &'static str,
) -> Result<Vec<usize>, CyclicArtifactError> {
    use antecedent_io::cyclic_graph::CyclicGraphRequest as G;
    let outcomes = match source {
        G::Identify { outcomes, .. } | G::Adjust { outcomes, .. } => outcomes,
        G::Calculus { y, .. } => y,
        _ => return Err(selector_mismatch()),
    };
    if outcomes.is_empty() {
        return Err(invalid(message));
    }
    Ok(outcomes.clone())
}

/// A projection of `value` over `outcomes`: a scalar coordinate for one outcome, a joint
/// event coordinate for several.
pub(super) fn event_point<'a>(
    value: Option<f64>,
    outcomes: &[usize],
    world: World,
    functional: Functional,
    port: EvidencePort,
) -> Projection<'a> {
    let mut p = point(value, outcomes[0], world, Time::Static, functional, port);
    if outcomes.len() > 1 {
        p.joint = Some(outcomes.to_vec());
    }
    p
}

/// One typed world of a specified model, by scientific world id.
fn world_of<'a>(
    worlds: &'a [antecedent_io::cyclic_family_artifact::CyclicWorldAnswer],
    selector: &CyclicEvidenceSelector,
) -> Result<Projection<'a>, CyclicArtifactError> {
    let CyclicEvidenceSelector::World { world, variable, time } = selector else {
        return Err(selector_mismatch());
    };
    let found = worlds.iter().find(|item| item.world == *world).ok_or_else(selector_mismatch)?;
    typed_world(found, *variable, *time)
}

impl EvidenceSource for A {
    #[allow(
        clippy::too_many_lines,
        reason = "one dispatch per answer kind keeps each kind's selectors and uncertainty port explicit"
    )]
    fn project<'a>(
        &'a self,
        request: &'a R,
        selector: &CyclicEvidenceSelector,
    ) -> Result<Projection<'a>, CyclicArtifactError> {
        use CyclicEvidenceSelector as S;
        match self {
            A::EquilibriumActions(worlds) => actions(worlds, selector),
            A::ParameterPosterior { worlds, .. } => parameter_posterior(worlds, selector),
            A::JointCounterfactual { worlds, .. } => {
                counterfactual_responses(worlds, true, selector)
            }
            A::PartialCounterfactual { worlds, .. } => {
                counterfactual_responses(worlds, false, selector)
            }
            A::FrequentistBootstrap { worlds, .. } => bootstrap(worlds, selector),
            A::OuMechanisms { laws, fit } => {
                ou_mechanisms(laws, fit.as_ref().map(|fit| fit.log_likelihood), selector)
            }
            A::Ode(path) => ode_path(path, selector),
            A::OuStationary(covariance) => ou_stationary(request, covariance, selector),
            A::OuTransient(path) => ou_transient(path, selector),
            A::Polynomial(solutions) => polynomial_solutions(solutions, selector),
            A::Functional(report) => {
                let (R::Functional { source, .. }, S::Functional { assignment }) =
                    (request, selector)
                else {
                    return Err(selector_mismatch());
                };
                let outcomes =
                    event_outcomes(source, "functional evidence projection requires an outcome")?;
                let value = report.values.get(*assignment).ok_or_else(selector_mismatch)?;
                let mut p = event_point(
                    value.value,
                    &outcomes,
                    World::Assignment(*assignment),
                    Functional::IdentifiedProbability,
                    EvidencePort::Mean,
                );
                p.diagnostics = detail(value);
                Ok(p)
            }
            A::GaussianFunctional(report) => {
                let (R::GaussianFunctional { source, .. }, S::Functional { assignment }) =
                    (request, selector)
                else {
                    return Err(selector_mismatch());
                };
                let outcomes =
                    event_outcomes(source, "Gaussian density evidence requires an outcome")?;
                let value = report.values.get(*assignment).ok_or_else(selector_mismatch)?;
                let mut p = event_point(
                    value.value,
                    &outcomes,
                    World::Assignment(*assignment),
                    Functional::IdentifiedDensity,
                    EvidencePort::Likelihood,
                );
                p.diagnostics = detail(value);
                Ok(p)
            }
            A::AdamFit(fit) => {
                log_likelihood(fit.training_log_likelihood, Time::Equilibrium, selector)
            }
            A::GradientFit(fit) => log_likelihood(fit.log_likelihood, Time::Static, selector),
            A::OdeFit(fit) => log_likelihood(fit.log_likelihood, Time::Trajectory, selector),
            A::OuFit(fit) => log_likelihood(fit.log_likelihood, Time::Stationary, selector),
            A::OuTrajectory(report) => ou_trajectory(report, selector),
            A::SdeAnalysis(report) => sde_analysis(report, selector),
            A::StructureFit { fit, .. } => {
                log_likelihood(fit.fit.log_likelihood, Time::Equilibrium, selector)
            }
            A::PolynomialFit(fit) => {
                let S::Likelihood { candidate: Some(candidate) } = selector else {
                    return Err(selector_mismatch());
                };
                let c = fit.candidates.get(*candidate).ok_or_else(selector_mismatch)?;
                let mut p = point(
                    None,
                    0,
                    World::Declared(c.candidate.clone()),
                    Time::Algebraic,
                    Functional::LogLikelihood,
                    EvidencePort::Likelihood,
                );
                p.bounds =
                    vec![c.lower.zip(c.upper).map(|(lower, upper)| Interval { lower, upper })];
                p.diagnostics = detail(c);
                Ok(p)
            }
            A::RecoveredLinear(report) => match selector {
                S::Response { world, noise_row, variable } => {
                    let R::RecoveredLinear { uncertainty, .. } = request else {
                        return Err(selector_mismatch());
                    };
                    if *noise_row != 0 {
                        return Err(selector_mismatch());
                    }
                    let found = report.worlds.get(*world).ok_or_else(selector_mismatch)?;
                    recovered_response(found, uncertainty, *world, *variable)
                }
                S::RecoveredDraw { world, draw, variable } => recovered_within(
                    report.worlds.get(*world).ok_or_else(selector_mismatch)?,
                    *world,
                    *draw,
                    *variable,
                ),
                _ => Err(selector_mismatch()),
            },
            A::ExperimentalRecovery(report) => match selector {
                S::Response { world, noise_row, variable } => {
                    if *noise_row != 0 {
                        return Err(selector_mismatch());
                    }
                    let found = &report.worlds.get(*world).ok_or_else(selector_mismatch)?.answer;
                    recovered_response(found, &report.law.origin, *world, *variable)
                }
                S::RecoveredDraw { world, draw, variable } => recovered_within(
                    &report.worlds.get(*world).ok_or_else(selector_mismatch)?.answer,
                    *world,
                    *draw,
                    *variable,
                ),
                _ => Err(selector_mismatch()),
            },
            A::Worlds(worlds) => world_of(worlds, selector),
            A::DynamicInference { fit, inference } => match selector {
                S::DynamicQuantity { quantity } => {
                    let R::DynamicInference { quantities, .. } = request else {
                        return Err(selector_mismatch());
                    };
                    dynamic_quantity(quantities, inference, *quantity)
                }
                S::Likelihood { candidate: None } => {
                    log_likelihood(fit.log_likelihood(), Time::Dynamic, selector)
                }
                _ => Err(selector_mismatch()),
            },
            A::System(report) => {
                let S::Action { world, noise_row, variable, counterfactual } = selector else {
                    return Err(selector_mismatch());
                };
                let value = if *counterfactual {
                    report.counterfactuals.as_ref().and_then(|rows| rows.get(*world))
                } else {
                    report.worlds.get(*world).and_then(|rows| rows.get(*noise_row))
                }
                .ok_or_else(selector_mismatch)?;
                let mut p = point(
                    Some(*value.values.get(*variable).ok_or_else(selector_mismatch)?),
                    *variable,
                    World::indexed(*world),
                    Time::Equilibrium,
                    if *counterfactual {
                        Functional::FactualCounterfactual
                    } else {
                        Functional::SpecifiedNoiseResponse
                    },
                    EvidencePort::Scenario,
                );
                p.diagnostics = detail(value);
                Ok(p)
            }
            A::Graph(graph) => {
                let S::Graph = selector else {
                    return Err(selector_mismatch());
                };
                let mut p = point(
                    None,
                    0,
                    World::Fixed("graphical"),
                    Time::Static,
                    Functional::GraphicalEvidence,
                    EvidencePort::Scenario,
                );
                p.diagnostics = detail(graph);
                Ok(p)
            }
            A::PolynomialCausal(report) => {
                let S::PolynomialWorld { world, variable } = selector else {
                    return Err(selector_mismatch());
                };
                let mut p = point(
                    None,
                    *variable,
                    World::Declared(world.clone()),
                    Time::Algebraic,
                    Functional::CounterfactualOutcomeSet,
                    EvidencePort::Scenario,
                );
                p.samples.clear();
                p.bounds.clear();
                let mut masses = Vec::new();
                for atom in &report.atoms {
                    let bound = match atom.states.get(world).ok_or_else(selector_mismatch)? {
                        antecedent_estimate::polynomial_scm::PolynomialSelectedState::Selected {
                            enclosure,
                        } => Some(*enclosure.get(*variable).ok_or_else(selector_mismatch)?),
                        _ => None,
                    };
                    p.samples.push(None);
                    p.bounds.push(bound);
                    if let Some(mass) = atom.posterior_mass {
                        masses.push(mass);
                    }
                }
                p.diagnostics = detail(PolynomialWorldDetail {
                    world_mass: &report.world_mass,
                    positive_evidence_certified: masses.len() == report.atoms.len()
                        && masses.iter().any(|m| m.lower > 0.0),
                });
                p.masses = Some(masses);
                p.axes = vec![Axis::Predictive, Axis::Parameter];
                Ok(p)
            }
        }
    }
}
