//! Scientific declaration types that public routes take and return.
//!
//! A consumer of the facade crate builds quantities, readings and model declarations
//! with these types without importing component crates.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use crate::analysis::mechanism_exports::model_crate::compile::MechanismSlot;
pub use crate::analysis::mechanism_exports::model_crate::compose::{
    ComposedFinite, ComposedLinearGaussian, ComposedModelSpec, CrossCovarianceBlock, Domain,
    FiniteLaw, FunctionalUncertainty, JointGaussian, MeanContrast, MechanismFamily, MechanismForm,
    MechanismUncertainty, ParameterId, PosteriorDependence, SampleTable, TabularRow,
    UncertaintyMeaning,
};
pub use antecedent_contract::{QuantityRole, RequestedReading, ScientificQuantity};
pub use antecedent_core::SemanticDigest;
pub use antecedent_core::measurement_model::{
    ArmBinding, BinaryArmRates, DifferentialAxis, ErrorStructure, MappingSemantics,
    MeasurementBinding, MeasurementKind, MeasurementModel, MeasurementParameters, MeasurementRoute,
    MeasurementVariable, ParameterEvidence, PosteriorReference, RepresentationMapping,
    RepresentationVersion,
};

// --- Further items the language bindings use, re-exported from their defining crates ---

/// Contrasts of intervention worlds and the dose × time surface.
pub use antecedent_core::InterventionContrast;
/// Worlds, bindings, decision time, and assumptions of a scientific target.
pub use antecedent_core::{
    Conditioning, GraphFamily, InferenceSelection, InitialLaw, InterventionWorld, ModelAssumptions,
    ObservationUnit, OutcomeQuantity, QuantityUnits, ScientificWorldId, TimeFrame, WorldAction,
    WorldBinding, WorldCollection,
};
/// The target form, the packaged `ScientificTarget`, and its semantic identity.
pub use antecedent_core::{ScientificTarget, TargetForm};
