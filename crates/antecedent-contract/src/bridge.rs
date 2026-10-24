//! Typed bridges between scientific quantities.
//!
//! A source claim answers a question about one scientific quantity. A
//! downstream consumer often asks about another: the latent construct a
//! measurement stands for, the behavior a stated preference predicts, the
//! terminal outcome a surrogate endpoint tracks, the causal quantity a
//! predictive model approximates, the intervention response a model parameter
//! encodes, the decision utility a scientific endpoint is scored against. The
//! source claim does not answer those questions by being renamed. A bridge is
//! the explicit, separately identified derivation that states the extra
//! assumptions under which the source answers the downstream question.
//!
//! * A [`BridgeContract`] names the source and target quantities with their
//!   [`QuantityClass`], one of six [`BridgeKind`]s, the declared
//!   [`BridgeAssumption`]s (each with an identity), the population and regime
//!   the bridge's own evidence was established for, the
//!   [`BridgeMapping`] that carries the value, and, when the evidence was
//!   established elsewhere than the claim's population, the
//!   [`TransportabilityPremise`] that licenses carrying it.
//! * [`derive_bridged_claim`] checks the contract against a source claim and
//!   produces a [`BridgedClaim`]: a new [`crate::ClaimEnvelope`] with its own
//!   identity (binding the contract, its assumptions and the source claim
//!   identity), its own provenance chain and a handoff receipt. The source
//!   claim is not modified and stays separately inspectable; the derived claim
//!   never reports more than the source held.
//! * Every route a bridge cannot license is a typed, coded refusal under the
//!   `scientific_bridge` detail namespace that names the declaration or
//!   evidence that would license it: an association offered as an
//!   intervention, a proxy offered as a construct, a stated preference offered
//!   as realized behavior, a predictive importance offered as a causal effect,
//!   a model score offered as a probability, a scientific endpoint offered as
//!   a utility, a missing assumption, a population or regime mismatch, a
//!   transported bridge whose premise is not declared, and an unlicensed
//!   substitution between a mean, a predictive law, a posterior, an
//!   identified set and a joint outcome law.
//!
//! The bridges carry one mean value. This layer produces no interval,
//! posterior or joint law and a request for any of them refuses; the interval
//! of the bridged value, from the source claim's declared standard error and the
//! mapping's own estimation uncertainty, is built where the coverage records
//! are held (`antecedent_io::bridge_interval`).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::result_large_err,
    reason = "a refusal is the cold path of a once-per-call check; boxing the error would not pay"
)]

mod claim;
mod contract;
mod kind;
mod mapping;
mod refusal;

pub use claim::{
    BRIDGE_CLAIM_IDENTITY_VERSION, BridgeSource, BridgedClaim, bridged_claim_id,
    derive_bridged_claim,
};
pub use contract::{
    AffineDrawLaw, AffineInnovation, AnalyticLaw, BRIDGE_IDENTITY_VERSION, BRIDGE_SOURCE_PORT,
    BRIDGE_TARGET_PORT, BridgeAssumption, BridgeContract, BridgeDrawsSource, BridgeEndpoint,
    BridgeInterfaceBasis, BridgeScope, BridgedDistributionClaim, CorrectionDrawLaw, DrawMapping,
    DrawsPlan, EstimandBridge, ExperimentalPrior, ExposureSetting, ImportedExperimentalPrior,
    Licensing, MAX_BRIDGE_ASSUMPTIONS, MAX_BRIDGE_TEXT_BYTES, MappedDrawsSummary,
    MappingUncertainty, NonlinearMap, NonlinearMoment, PreservedSupport, PropagatedUncertainty,
    SourceMappingDependence, SupportMap, TransportabilityPremise, ValueInterval,
    bridged_distribution_claim_id, nonlinear_expectation, preserved_support, support_image,
};
pub use kind::{AssumptionKind, BridgeKind, QuantityClass, ValueKind};
pub use mapping::{
    AppliedMapping, BridgeMapping, MAX_VALIDATION_TRIALS, MIN_VALIDATION_TRIALS, MappingKind,
    MeasurementDirection, TrialEffectPair, TrialLevelFit, TrialLevelValidation,
    fit_trial_level_linear,
};
pub use refusal::{
    BridgeRefusal, chain_stage_not_affine, draws_outside_domain, draws_unusable,
    interval_not_licensed, wrong_value_kind,
};
