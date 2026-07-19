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
pub mod graph_samples;
pub mod hmc;
pub mod laplace;
pub(crate) mod linalg;
pub(crate) mod mcmc_stats;
pub mod posterior;
pub mod prior;

pub use backend::{
    BayesDesignRef, BayesFitOptions, BayesFitResult, BayesLikelihood, InferenceBackend,
    LaplaceWorkspace, coefficient_schema,
};
pub use conjugate::{ConjugateGaussianBackend, fit_conjugate_gaussian};
pub use diagnostics::{HessianFactorization, InferenceDiagnostics, PriorSensitivitySummary};
pub use error::ProbError;
pub use graph_samples::{GraphIdentFlag, WeightedGraphSamples};
pub use hmc::{HmcGlmBackend, HmcOptions, fit_hmc_glm};
pub use laplace::{LaplaceGlmBackend, fit_laplace_glm};
pub use posterior::{
    EffectBatch, PosteriorBatch, PosteriorDraws, PosteriorEvalWorkspace, PosteriorQuantityKind,
    PosteriorSchema, PosteriorSummary,
};
pub use prior::{ContrastCoding, GaussianCoefficientPrior, InvGammaPrior, PriorSet, PriorSpec};
