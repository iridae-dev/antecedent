//! Probability distributions, columnar posteriors, priors, and inference backends.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod backend;
pub mod conjugate;
pub mod diagnostics;
pub mod error;
pub mod graph_samples;
pub mod posterior;
pub mod prior;

pub use backend::{
    BayesDesignRef, BayesFitOptions, BayesFitResult, BayesLikelihood, InferenceBackend,
    LaplaceWorkspace, coefficient_schema,
};
pub use conjugate::{ConjugateGaussianBackend, fit_conjugate_gaussian};
pub use diagnostics::{
    HessianFactorization, InferenceDiagnostics, PriorSensitivitySummary,
};
pub use error::ProbError;
pub use graph_samples::{GraphIdentFlag, WeightedGraphSamples};
pub use posterior::{
    EffectBatch, PosteriorBatch, PosteriorDraws, PosteriorEvalWorkspace, PosteriorQuantityKind,
    PosteriorSchema, PosteriorSummary,
};
pub use prior::{
    ContrastCoding, GaussianCoefficientPrior, InvGammaPrior, PriorSet, PriorSpec,
};
