//! Effect refuters and validation diagnostics.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod bayesian_checks;
pub mod bootstrap_refute;
pub mod common;
pub mod data_subset;
pub mod dummy_outcome;
pub mod error;
pub mod evalue;
pub mod graph_refute;
pub mod overlap;
pub mod overlap_rule;
pub mod placebo;
pub mod rcc;
pub mod reisz;
pub mod sensitivity;
pub mod stability;
pub mod suite;
pub mod unobserved_common_cause;
pub mod validator;

pub use bayesian_checks::{
    DEFAULT_MAX_RELATIVE_PRIOR_RANGE, PosteriorPredictiveCheck, PredictiveCheckKind,
    PredictiveCheckReport, PriorPredictiveCheck, PriorSensitivity, with_prior_sensitivity,
};
pub use bootstrap_refute::BootstrapRefute;
pub use common::{RefutationProblem, RefutationReport};
pub use data_subset::DataSubsetRefuter;
pub use dummy_outcome::DummyOutcome;
pub use error::ValidationError;
pub use evalue::{DEFAULT_EVALUE_THRESHOLD, EValue};
pub use graph_refute::GraphRefuter;
pub use overlap::OverlapRefuter;
pub use overlap_rule::OverlapRuleRefuter;
pub use placebo::{PlaceboMode, PlaceboTreatment};
pub use rcc::RandomCommonCause;
pub use reisz::ReiszSensitivity;
pub use sensitivity::{LinearSensitivity, NonparametricSensitivity, PartialLinearSensitivity};
pub use stability::{BlockBootstrapStability, DiscoveryStabilityReport, LinkStability};
pub use suite::{BayesianSuiteContext, ValidationOutcome, ValidationSuite, ValidatorId};
pub use unobserved_common_cause::UnobservedCommonCause;
pub use validator::{PreparedRefutation, Validator, run_validator};

#[cfg(test)]
#[allow(clippy::cast_precision_loss, clippy::many_single_char_names)]
mod tests;
