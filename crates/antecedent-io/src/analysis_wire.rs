//! Analysis / estimate / identification / refutation / diagnostic wire types.
//!
//! The identification and diagnostic half lives in `antecedent-io-core`; the
//! effect-estimate and refutation half needs the estimators. Both are public
//! here under the one path.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_io_core::analysis_wire::{
    DiagnosticWire, HedgeCertificateWire, HedgeProblemWire, IdentificationResultWire,
    IdentifiedEstimandWire, RdDesignWire, RefutationReportWire, diagnostic_from_wire,
    diagnostic_to_wire, identification_from_wire, identification_status_pascal,
    identification_to_wire, identification_to_wire_with_registry,
};

pub use crate::estimate_wire::{
    CandidateSelectionWire, ClipSensitivityWire, EffectEstimateWire, EstimateOverlapPolicyWire,
    FirstStageDiagnosticsWire, JointCovarianceWire, OverlapReportWire, PreparationWire,
    PropensityIntervalWire, ScoreColumnWire, ScoreInferenceWire, ScoreTableWire,
    UncertaintyWithheldWire, effect_estimate_from_owned_wire, effect_estimate_from_wire,
    effect_estimate_to_wire, refutation_from_wire, refutation_to_wire,
};
