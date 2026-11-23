use super::{
    A, CyclicArtifactError, CyclicBootstrapDesign, Deserialize, DynamicBootstrapMethod,
    DynamicBootstrapRequest, DynamicFitRef, DynamicInferenceError, DynamicObservedInformation,
    DynamicPosterior, DynamicPosteriorOptions, DynamicPrior, DynamicQuantity, DynamicSamplingLaw,
    ExecutionContext, OdeFit, OdeInferenceProblem, OuFit, OuInferenceProblem, OuMechanismFit,
    OuMechanismInferenceProblem, ResamplingRuleWire, RngFactory, SamplingDesignWire, Serialize,
    bootstrap_dynamic_fit, dynamic_observed_information, invalid, sample_dynamic_posterior,
};

/// The declared fitting procedure of a continuous-time family.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "family", content = "problem", rename_all = "snake_case")]
pub enum DynamicProblemWire {
    /// ODE drift fit under Gaussian measurement error.
    Ode(OdeInferenceProblem),
    /// Stationary OU fit.
    Ou(OuInferenceProblem),
    /// Shared-mechanism OU fit.
    OuMechanisms(OuMechanismInferenceProblem),
}

impl DynamicProblemWire {
    /// State order of the declared system.
    #[must_use]
    pub fn order(&self) -> usize {
        match self {
            Self::Ode(problem) => problem.template.equations.len(),
            Self::Ou(problem) => problem.template.center.len(),
            Self::OuMechanisms(problem) => problem.template.equations.len(),
        }
    }
}

/// How the uncertainty of the fitted dynamic family is produced.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "method", rename_all = "snake_case", deny_unknown_fields)]
pub enum DynamicMethodWire {
    /// Whole-procedure bootstrap. A sampling law.
    Bootstrap {
        /// Parametric or nonparametric replicate data.
        kind: DynamicBootstrapMethod,
        /// Replicates, failed ones included.
        replicates: usize,
        /// Declared sampling design of the units.
        design: SamplingDesignWire,
        /// Attempted resampling rule; the design must allow it.
        resampling: ResamplingRuleWire,
        /// Strata of a nonparametric draw.
        #[serde(default)]
        stratification: CyclicBootstrapDesign,
        /// Seed of the replicate streams.
        seed: u64,
    },
    /// Observed information and its delta-method covariance, when the likelihood is regular.
    Information {
        /// Declared sampling design of the units.
        design: SamplingDesignWire,
    },
    /// Posterior under an explicit independent normal prior.
    Posterior {
        /// Prior over the estimated coordinates.
        prior: DynamicPrior,
        /// Sampler policy.
        #[serde(default)]
        options: DynamicPosteriorOptions,
        /// Declared sampling design of the units.
        design: SamplingDesignWire,
        /// Seed of the chains.
        seed: u64,
    },
}

/// Point fit of a continuous-time family.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "family", content = "fit", rename_all = "snake_case")]
pub enum DynamicFitAnswer {
    /// ODE fit.
    Ode(OdeFit),
    /// Stationary OU fit.
    Ou(OuFit),
    /// Shared-mechanism OU fit.
    OuMechanisms(OuMechanismFit),
}

impl DynamicFitAnswer {
    /// Log likelihood at the estimate.
    #[must_use]
    pub const fn log_likelihood(&self) -> f64 {
        match self {
            Self::Ode(fit) => fit.log_likelihood,
            Self::Ou(fit) => fit.log_likelihood,
            Self::OuMechanisms(fit) => fit.log_likelihood,
        }
    }
}

/// Uncertainty of a fitted dynamic family.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "method", content = "result", rename_all = "snake_case")]
pub enum DynamicInferenceAnswer {
    /// Bootstrap sampling law.
    SamplingLaw(DynamicSamplingLaw),
    /// Observed information, or the reason no covariance exists.
    Information(DynamicObservedInformation),
    /// Parameter posterior with quantity draws.
    Posterior(DynamicPosterior),
}

fn dynamic_error(error: DynamicInferenceError) -> CyclicArtifactError {
    match error {
        DynamicInferenceError::Model(model) => CyclicArtifactError::Execution(model),
        other => invalid(format!("dynamic inference refused: {other}")),
    }
}

fn fit_dynamic(
    problem: &DynamicProblemWire,
    ctx: &ExecutionContext,
) -> Result<DynamicFitAnswer, CyclicArtifactError> {
    Ok(match problem {
        DynamicProblemWire::Ode(p) => {
            DynamicFitAnswer::Ode(antecedent_estimate::cyclic_dynamics::fit_ode_likelihood(
                &p.template,
                &p.parameters,
                &p.data,
                p.integration,
                p.policy,
                ctx,
            )?)
        }
        DynamicProblemWire::Ou(p) => {
            DynamicFitAnswer::Ou(antecedent_estimate::cyclic_dynamics::fit_ou_likelihood(
                &p.template,
                &p.parameters,
                &p.rows,
                p.options,
                ctx,
            )?)
        }
        DynamicProblemWire::OuMechanisms(p) => {
            DynamicFitAnswer::OuMechanisms(antecedent_estimate::ou_mechanisms::fit_ou_mechanisms(
                &p.template,
                &p.parameters,
                &p.regimes,
                p.policy,
                ctx,
            )?)
        }
    })
}

fn seeded(ctx: &ExecutionContext, seed: u64) -> ExecutionContext {
    let mut context = ctx.clone();
    context.rng = RngFactory::from_seed(seed);
    context
}

pub(in crate::cyclic_family_artifact) fn execute_dynamic_inference(
    problem: &DynamicProblemWire,
    method: &DynamicMethodWire,
    quantities: &[DynamicQuantity],
    frozen: Option<&DynamicFitAnswer>,
    ctx: &ExecutionContext,
) -> Result<A, CyclicArtifactError> {
    let fit = match frozen {
        Some(retained) => retained.clone(),
        None => fit_dynamic(problem, ctx)?,
    };
    let target = match (problem, &fit) {
        (DynamicProblemWire::Ode(problem), DynamicFitAnswer::Ode(fit)) => {
            DynamicFitRef::Ode { problem, fit }
        }
        (DynamicProblemWire::Ou(problem), DynamicFitAnswer::Ou(fit)) => {
            DynamicFitRef::Ou { problem, fit }
        }
        (DynamicProblemWire::OuMechanisms(problem), DynamicFitAnswer::OuMechanisms(fit)) => {
            DynamicFitRef::OuMechanisms { problem, fit }
        }
        _ => return Err(invalid("retained dynamic fit belongs to a different family")),
    };
    let inference = match method {
        DynamicMethodWire::Bootstrap {
            kind,
            replicates,
            design,
            resampling,
            stratification,
            seed,
        } => DynamicInferenceAnswer::SamplingLaw(
            bootstrap_dynamic_fit(
                target,
                DynamicBootstrapRequest {
                    method: *kind,
                    replicates: *replicates,
                    design: design.engine(),
                    resampling: resampling.engine(),
                    stratification: *stratification,
                },
                quantities,
                &seeded(ctx, *seed),
            )
            .map_err(dynamic_error)?,
        ),
        DynamicMethodWire::Information { design } => DynamicInferenceAnswer::Information(
            dynamic_observed_information(target, design.engine(), quantities, ctx)
                .map_err(dynamic_error)?,
        ),
        DynamicMethodWire::Posterior { prior, options, design, seed } => {
            DynamicInferenceAnswer::Posterior(
                sample_dynamic_posterior(
                    target,
                    prior,
                    *options,
                    design.engine(),
                    quantities,
                    &seeded(ctx, *seed),
                )
                .map_err(dynamic_error)?,
            )
        }
    };
    Ok(A::DynamicInference { fit, inference: Box::new(inference) })
}
