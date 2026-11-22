//! Probability distributions, columnar posteriors, priors, and inference backends.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![cfg_attr(
    test,
    allow(
        clippy::float_cmp,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

pub(crate) mod admission;
pub mod backend;
pub mod conjugate;
pub mod conjugate_counts;
pub mod conjugate_moment_match;
pub mod diagnostics;
pub mod error;
pub mod external_prior;
/// Aligned finite Bayesian conditioning and explicit experimental likelihoods.
pub mod finite_belief;
pub mod fit_counts;
pub mod gaussian_target;
pub mod gradient_hmc;
pub mod graph_samples;
pub mod hmc;
pub mod laplace;
pub(crate) mod likelihood_terms;
pub(crate) mod linalg;
pub mod mcmc_stats;
pub mod metropolis;
pub mod posterior;
pub mod power_prior_alpha;
pub mod prior;
pub(crate) mod streams;
pub mod transport;
pub mod transport_target;

pub use backend::{
    BayesDesignRef, BayesFitOptions, BayesFitResult, BayesLikelihood, InferenceBackend,
    LaplaceWorkspace,
};
pub use conjugate::{
    ConjugateGaussianBackend, fit_conjugate_gaussian, sample_gamma, sample_inv_gamma,
};
pub use conjugate_counts::{BetaBinomialPosterior, GammaPoissonPosterior};
pub use conjugate_moment_match::{BetaHyperparameters, GammaHyperparameters};
pub use diagnostics::{
    ConflictSummary, HessianFactorization, InferenceDiagnostics, PriorSensitivityFamily,
    PriorSensitivitySummary,
};
pub use error::ProbError;
pub use external_prior::{
    ComposedPrior, CoordinateMapping, ExternalPriorSource, ExternalPriorWeight,
    NamedCoordinateSelection, compose_external_priors, compose_external_priors_with_alphas,
    kish_ess, rename_coordinates, select_coordinates_by_name,
};
pub use graph_samples::{GraphEnvelopeSubsample, GraphIdentFlag, WeightedGraphSamples};
pub use hmc::{HmcGlmBackend, HmcOptions, fit_hmc_glm, fit_hmc_glm_with_context};
pub use laplace::{LaplaceGlmBackend, fit_laplace_glm, sample_gaussian_mvn};
pub use mcmc_stats::{
    McmcSummary, ParameterMcmcDiagnostics, all_chains_moved, mcmc_summary, mcmc_summary_parallel,
    min_bulk_ess, min_tail_ess, parameter_mcmc_diagnostics,
};
pub use posterior::{
    EffectBatch, PosteriorBatch, PosteriorDraws, PosteriorEvalWorkspace, PosteriorQuantityKind,
    PosteriorSchema, PosteriorSummary,
};
pub use power_prior_alpha::{
    AlphaCriterion, AlphaSelectionOptions, AlphaSensitivityRow, BetaSource, GammaSource,
    LinearTarget, SelectedAlphaReport, compose_beta_power_priors, compose_gamma_power_priors,
    default_alpha_grid, select_beta_power_prior, select_beta_prior, select_gamma_power_prior,
    select_gamma_prior, select_linear_power_prior, select_linear_prior,
};
pub use prior::{
    CoefficientCorrelation, CoefficientPrecision, ContrastCoding, EffectPrior,
    GaussianCoefficientPrior, GaussianVarianceModel, InvGammaPrior, PriorSet, PriorSpec,
};
pub use transport::{
    POPULATION_TAG_KEY, TRANSPORT_MIN_KISH_ESS, TransportAdjustment, TransportContext,
    TransportError, TransportOutcome, TransportPolicy, apply_transport, compose_with_transport,
};
pub use transport_target::{
    BandwidthSensitivityRow, KernelFamily, TARGET_MIN_ESS_FRACTION, TargetOverlap, TransportKernel,
    WeightedTarget, apply_weighted_target_transport, check_weights_overlap,
    compose_with_weighted_target_transport, kernel_bandwidth_sensitivity,
};
