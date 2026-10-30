//! Durable wire forms for transportability and randomized-interference results.
//!
//! The wire types and the structural identification conversions live in
//! `antecedent-io-core`; the estimate conversions need the estimators. Both are
//! public here under the one path.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_io_core::transport_interference_wire::{
    ExposureProbabilityMethodWire, InterferenceEstimateWire, InterferenceTransportCertificateWire,
    NotCertifiedCertificateWire, PopulationFactorWire, RandomizationContrastWire,
    TransportCertificateWire, TransportEffectEstimateWire, TransportFormulaWire,
    TransportIdentificationWire, TransportOverlapDiagnosticWire,
    transport_identification_from_wire, transport_identification_to_wire,
};

pub use crate::transport_effect_wire::{
    interference_estimate_from_wire, interference_estimate_to_wire, transport_effect_from_wire,
    transport_effect_to_wire, try_interference_estimate_from_wire,
};
