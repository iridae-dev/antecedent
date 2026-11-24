//! Execution of the fitted-model operations a model route answers beyond attribution and plain
//! simulation: edge strengths, the population do-contrast, a named do-sampler and the held-out
//! falsification evaluation.
//!
//! Each runner calls the same kernel function the direct `antecedent::gcm` entry calls, with the
//! inputs the target declares, so the compiled answer and the direct answer are the same numbers.
//! Every target carries its own seed. The population do-contrast draws from the attribution
//! stream of a generator seeded with the target's seed, which is the stream the direct entry
//! draws from under a context seeded the same way; the do-samplers start their generator at the
//! target's seed, as a direct caller passing `CausalRng::from_seed(seed)` does.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{
    CausalRng, DoSamplerMethod, DoSamplerTarget, ExecutionContext, Intervention,
    ModelEvaluationTarget, PopulationDoContrastTarget, RngFactory,
};
use antecedent_data::{TableView, TabularData};
use antecedent_model::{
    DoSampleResult, KdeDoSampler, McmcDoSampler, MechanismWorkspace, ModelEvaluationReport,
    ModelEvaluator, WeightingDoSampler,
};

use crate::error::CausalError;
use crate::gcm::{ArrowStrength, FittedGcm, PopulationDoContrast};

/// Model-implied strength of every edge of the fitted model.
///
/// # Errors
///
/// A non-linear mechanism anywhere the strengths read, or a degenerate noise variance.
pub(crate) fn run_arrow_strengths(fitted: &FittedGcm) -> Result<Vec<ArrowStrength>, CausalError> {
    crate::gcm::arrow_strengths(&fitted.model).map_err(CausalError::from)
}

/// Population do-contrast of the target's parent on its child over the study's rows.
///
/// The kernel draws both arms from the context's attribution stream; the context it is given
/// keeps every setting of `ctx` (threads, cancellation, admission) and replaces only the
/// generator, by one seeded with the target's seed.
///
/// # Errors
///
/// A table larger than the target's row cap, or a sampling failure.
pub(crate) fn run_population_do_contrast(
    fitted: &FittedGcm,
    data: &TabularData,
    target: &PopulationDoContrastTarget,
    ctx: &ExecutionContext,
) -> Result<PopulationDoContrast, CausalError> {
    let cap = target.max_units.unwrap_or_else(|| data.row_count());
    let mut seeded = ctx.clone();
    seeded.rng = RngFactory::from_seed(target.seed);
    crate::gcm::population_do_contrast(
        &fitted.model,
        data,
        target.parent,
        target.child,
        target.delta,
        cap,
        &seeded,
    )
    .map_err(CausalError::from)
}

/// The named do-sampler for the target's outcome under its hard interventions.
///
/// # Errors
///
/// A weighting target that does not hold exactly one hard `set` intervention, or a sampling
/// failure.
pub(crate) fn run_do_sampler(
    fitted: &FittedGcm,
    data: &TabularData,
    target: &DoSamplerTarget,
    ctx: &ExecutionContext,
) -> Result<DoSampleResult, CausalError> {
    let mut rng = CausalRng::from_seed(target.seed);
    let mut ws = MechanismWorkspace::default();
    match target.method {
        DoSamplerMethod::Weighting { root_match_tolerance } => {
            let [Intervention::Set { variable, value }] = &target.interventions[..] else {
                return Err(CausalError::Compile {
                    message: "the weighting do-sampler targets exactly one hard `set` intervention"
                        .into(),
                });
            };
            let level = value.as_f64().ok_or_else(|| CausalError::Compile {
                message: "the weighting do-sampler targets a numeric level".into(),
            })?;
            let mut sampler = WeightingDoSampler::new(*variable, target.outcome);
            if let Some(tolerance) = root_match_tolerance {
                sampler = sampler.with_root_match_tolerance(tolerance);
            }
            sampler.estimate(&fitted.model, data, level, ctx).map_err(CausalError::from)
        }
        DoSamplerMethod::Kde { bandwidth } => {
            let mut sampler = KdeDoSampler::new(target.outcome);
            sampler.bandwidth = bandwidth;
            sampler
                .sample(&fitted.model, &target.interventions, target.draws, &mut rng, &mut ws, ctx)
                .map_err(CausalError::from)
        }
        DoSamplerMethod::Mcmc { proposal_sd, burn_in, thin } => {
            let mut sampler = McmcDoSampler::new(target.outcome);
            if let Some(proposal_sd) = proposal_sd {
                sampler.proposal_sd = proposal_sd;
            }
            if let Some(burn_in) = burn_in {
                sampler.burn_in = burn_in;
            }
            if let Some(thin) = thin {
                sampler.thin = thin;
            }
            sampler
                .sample(&fitted.model, &target.interventions, target.draws, &mut rng, &mut ws, ctx)
                .map_err(CausalError::from)
        }
    }
}

/// The fitted model's falsification evaluation on the study's rows.
///
/// # Errors
///
/// Empty data or a mechanism failure.
pub(crate) fn run_model_evaluation(
    fitted: &FittedGcm,
    data: &TabularData,
    target: &ModelEvaluationTarget,
    ctx: &ExecutionContext,
) -> Result<ModelEvaluationReport, CausalError> {
    ModelEvaluator { alpha: target.alpha, n_permutations: target.n_permutations, seed: target.seed }
        .evaluate(&fitted.model, data, ctx)
        .map_err(CausalError::from)
}
