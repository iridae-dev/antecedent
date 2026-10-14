//! Scientific declaration types that public routes take and return.
//!
//! A consumer of the facade crate builds quantities, readings and model declarations
//! with these types without importing component crates.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_contract::{QuantityRole, RequestedReading, ScientificQuantity};
pub use antecedent_core::SemanticDigest;
pub use antecedent_core::measurement_model::{
    ArmBinding, BinaryArmRates, DifferentialAxis, ErrorStructure, MappingSemantics,
    MeasurementBinding, MeasurementKind, MeasurementModel, MeasurementParameters, MeasurementRoute,
    MeasurementVariable, ParameterEvidence, PosteriorReference, RepresentationMapping,
    RepresentationVersion,
};
pub use antecedent_model::compile::MechanismSlot;
pub use antecedent_model::compose::{
    ComposedFinite, ComposedLinearGaussian, ComposedModelSpec, CrossCovarianceBlock, Domain,
    FiniteLaw, FunctionalUncertainty, JointGaussian, MeanContrast, MechanismFamily, MechanismForm,
    MechanismUncertainty, ParameterId, PosteriorDependence, SampleTable, TabularRow,
    UncertaintyMeaning,
};
