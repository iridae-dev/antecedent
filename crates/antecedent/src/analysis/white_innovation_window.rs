//! Untempered Bayesian fit of one lag-aligned temporal window under a declared premise.
//!
//! The tempered posterior of a single-step temporal linear effect is a generalized posterior.
//! When the caller declares that the outcome innovation is serially independent, the same lag
//! window has a full-likelihood posterior instead, and a Ljung-Box test of the residuals can
//! refute the declaration. The declared premise and the test result travel in the posterior's
//! assumptions and diagnostics, so the result cannot be read as the tempered posterior.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{ExecutionContext, IdentificationStatus};
use antecedent_estimate::white_innovation_posterior::{
    WhiteInnovationDeclaration, fit_white_innovation_posterior,
};
use antecedent_estimate::{
    BayesianGComputationAte, BayesianTemporalGcomp, CausalPosterior, EffectEstimate, OverlapPolicy,
    PreparedBayesianProblem, PreparedEstimationProblem,
};

use crate::BayesianConfig;
use crate::error::CausalError;

/// The untempered fit of one lag-aligned window.
#[derive(Clone, Debug)]
pub(crate) struct WhiteInnovationWindowFit {
    pub(crate) estimate: EffectEstimate,
    pub(crate) posterior: CausalPosterior,
    pub(crate) prepared: PreparedBayesianProblem,
    pub(crate) estimator: BayesianTemporalGcomp,
}

/// Fit the untempered posterior of `prep` under the white-innovation premise `cfg` declares.
///
/// # Errors
/// A missing declaration, a transferred prior, a horizon above one step, a non-Gaussian
/// likelihood, a residual test that rejects whiteness, or a failed fit.
pub(crate) fn fit_white_innovation_window(
    cfg: &BayesianConfig,
    prep: &PreparedEstimationProblem,
    horizon_steps: u32,
    identification: IdentificationStatus,
    ctx: &ExecutionContext,
) -> Result<WhiteInnovationWindowFit, CausalError> {
    let Some(gate) = cfg.white_innovation else {
        return Err(CausalError::Compile {
            message: "an untempered temporal fit needs a declared white-innovation premise".into(),
        });
    };
    if cfg.has_transfer() {
        return Err(CausalError::Unsupported {
            message: "the untempered temporal posterior uses the isotropic Gaussian prior and \
                      does not accept coefficient or transferred priors",
        });
    }
    let inner = BayesianGComputationAte {
        backend: cfg.backend,
        likelihood: cfg.likelihood,
        n_draws: cfg.n_draws,
        seed: ctx.rng.master_seed(),
        overlap: OverlapPolicy::ExplicitOverride,
        prior_scale: cfg.prior_scale,
        prior: None,
    };
    let declaration = WhiteInnovationDeclaration { horizon_steps, gate };
    let fit = fit_white_innovation_posterior(&inner, prep, &declaration, identification, ctx)
        .map_err(CausalError::from)?;
    let estimate = super::helpers::effect_from_posterior(&fit.posterior)?
        .with_n_obs(u64::try_from(prep.design.nrows).unwrap_or(u64::MAX));
    Ok(WhiteInnovationWindowFit {
        estimate,
        posterior: fit.posterior,
        prepared: BayesianGComputationAte::from_prepared_estimation(prep),
        estimator: BayesianTemporalGcomp { inner },
    })
}
