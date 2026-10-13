//! Scientific interface contracts: what a component produces and consumes,
//! and the decision whether two components may be connected, made from their
//! declarations before any number is computed.
//!
//! A syntactically compatible interface is not scientific compatibility. Two
//! components can agree on array shape, dtype and a variable name and still
//! disagree on units, population, intervention regime, support, the
//! measurement model the value was corrected under, whether the producer
//! offers a mean where the consumer needs a law, or what its interval means.
//! This module is the one declaration that states those things, in the
//! vocabulary the contract crate already owns, extending rather than
//! duplicating the semantic and provider contracts:
//!
//! * [`ProducedPort`] and [`ConsumedPort`] carry a
//!   [`crate::ScientificQuantity`] (variable, role, units, population,
//!   regime, horizon, functional, conditioning, transform), a semantic
//!   identity, a [`SupportDomain`], the [`ContractOperation`]s offered or
//!   required, the [`UncertaintyMeaning`] of any covariance, law, draws,
//!   interval or posterior, the [`CalibrationStatus`] held or the
//!   [`CalibrationRequirement`] relied on, the measurement-model identities
//!   the value is established under, the [`EvidenceBasis`] (evidence-ledger
//!   source identities) or the [`DependenceRequirement`], and the provenance
//!   stages supplied or required;
//! * an [`InterfaceContract`] bundles the ports of one component and has one
//!   canonical, versioned identity ([`InterfaceContract::canonical_identity`]);
//! * [`check_connection`] decides one connection and [`check_composition`] a
//!   whole set, refusing a cycle with its path named. Every refusal is a
//!   registered reason code with an `interface_contract.*` detail and the
//!   declaration that would license the connection.
//!
//! Algorithm inputs stay in their owning lower crates: the contract names a
//! measurement model by identity, a calibration record by identity and an
//! evidence source by its ledger identity; estimators and identifiers do not
//! depend on this module.
//!
//! # What is and is not decided
//!
//! The decision is about declarations. It does not execute either component,
//! verify that a declared calibration record exists or covers, or establish
//! that a declared population is the one the data came from; those are the
//! claims of the artifacts the declarations describe. A connection never
//! raises a label: the calibration label a connection carries is the
//! producer's, a point-only producer stays point-only, and nothing here
//! licenses an interval or a posterior.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::result_large_err,
    reason = "a refusal is the cold path of a once-per-call check; boxing the error would not pay"
)]

mod composition;
mod connection;
mod declaration;
mod refusal;

pub use composition::{
    ComponentConnection, ComposedInterfaces, MAX_COMPOSITION_COMPONENTS,
    MAX_COMPOSITION_CONNECTIONS, check_composition,
};
pub use connection::{Connection, check_connection};
pub use declaration::{
    CalibrationRequirement, CalibrationStatus, ConsumedPort, ContractOperation,
    DependenceRequirement, EvidenceBasis, INTERFACE_CONTRACT_VERSION, InterfaceContract,
    MAX_INTERFACE_PORTS, MAX_INTERFACE_TEXT_BYTES, MAX_PORT_SET_ITEMS, ProducedPort, SupportDomain,
    UncertaintyMeaning, UncertaintyParadigm, operations_of_capabilities,
};
pub use refusal::InterfaceRefusal;
