//! Probability distributions, columnar posteriors, priors, and inference backends.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::neg_cmp_op_on_partial_ord,
    clippy::unnecessary_wraps,
    clippy::manual_memcpy,
    clippy::single_match_else,
    clippy::match_same_arms,
    clippy::manual_range_contains,
    clippy::doc_markdown,
    clippy::needless_range_loop,
    clippy::too_many_lines,
    clippy::too_many_arguments
)]

pub mod backend;
pub mod conjugate;
pub mod diagnostics;
pub mod error;
pub mod external_prior;
pub mod graph_samples;
pub mod hmc;
pub mod laplace;
pub(crate) mod linalg;
pub mod mcmc_stats;
pub mod posterior;
pub mod prior;
pub mod transport;

pub use backend::{
    BayesDesignRef, BayesFitOptions, BayesFitResult, BayesLikelihood, InferenceBackend,
    LaplaceWorkspace, coefficient_schema,
};
pub use conjugate::{ConjugateGaussianBackend, fit_conjugate_gaussian};
pub use diagnostics::{
    ConflictSummary, HessianFactorization, InferenceDiagnostics, PriorSensitivitySummary,
};
pub use error::ProbError;
pub use external_prior::{
    ComposedPrior, ExternalPriorSource, ExternalPriorWeight, compose_external_priors,
    compose_external_priors_with_alphas,
};
pub use graph_samples::{GraphEnvelopeSubsample, GraphIdentFlag, WeightedGraphSamples};
pub use hmc::{HmcGlmBackend, HmcOptions, fit_hmc_glm};
pub use laplace::{LaplaceGlmBackend, fit_laplace_glm, sample_gaussian_mvn};
pub use mcmc_stats::{max_split_rhat, min_bulk_ess};
pub use posterior::{
    EffectBatch, PosteriorBatch, PosteriorDraws, PosteriorEvalWorkspace, PosteriorQuantityKind,
    PosteriorSchema, PosteriorSummary,
};
pub use prior::{
    ContrastCoding, EffectPrior, GaussianCoefficientPrior, InvGammaPrior, PriorSet, PriorSpec,
};
pub use transport::{
    POPULATION_TAG_KEY, TRANSPORT_ASSUMPTION_ID, TransportAdjustment, TransportContext,
    TransportError, TransportOutcome, TransportPolicy, apply_transport, compose_with_transport,
    populations_require_transport,
};
