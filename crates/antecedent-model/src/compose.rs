//! Composition of independently fitted causal mechanisms.
//!
//! Individually valid conditional mechanisms do not imply a valid joint model.
//! This module compiles *declared* mechanisms into one executable joint law,
//! and refuses, with a structured finding, whenever the declarations do not
//! establish a coherent one.
//!
//! # Boundary
//!
//! - **Structure**: explicitly mapped acyclic compositions only. A cycle is
//!   refused before anything executes.
//! - **Families** (one family per composition, never mixed):
//!   - additive-noise linear-Gaussian mechanisms with a declared parent set,
//!     executed in closed form (mean, covariance, intervention law by path
//!     algebra), sampled structurally, and propagated through parameter
//!     uncertainty by the first-order delta method for functionals of
//!     interventional means;
//!   - finite-discrete conditional tables, executed by exact enumeration and
//!     truncated factorization; parameter uncertainty is not propagated (the
//!     route is point-only).
//! - **Declared, not discovered**: a mechanism enters a composition only
//!   through [`MechanismDescriptor`]. Nothing is inferred about units,
//!   populations, shared causes or dependence from names or fits.
//! - **Uncertainty**: separately fitted mechanisms are not independent by
//!   default. Unknown dependence stays unknown; a claim over more than one
//!   mechanism's parameters needs an aligned joint law, an explicit
//!   cross-mechanism covariance for every pair, or a justified declaration of
//!   independence, and otherwise the result is a refusal. Execution without an
//!   uncertainty claim is a point-only result; no interval is produced here.
//!
//! The semantic mapping from scientific quantities to these descriptors lives
//! in `antecedent-contract`; this crate carries no claim vocabulary.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub mod coherence;
pub mod descriptor;
pub mod gaussian;
pub mod tabular;
pub mod uncertainty;

use antecedent_core::{StreamDomain, mix64};

pub use coherence::{
    COVARIANCE_TOLERANCE, CoherenceFinding, CoherenceKind, check_composition,
    is_positive_semidefinite, parameter_layout, parameter_terms, uncertainty_findings,
};
pub use descriptor::{
    ComposedModelSpec, Containment, ContextCoordinate, CrossCovarianceBlock, Domain,
    InvarianceDeclaration, LatentDependence, MAX_COMPOSED_LATENTS, MAX_COMPOSED_MECHANISMS,
    MAX_FINITE_STATES, MAX_LEVELS_PER_VARIABLE, MAX_PARENTS_PER_MECHANISM,
    MECHANISM_DESCRIPTOR_VERSION, MechanismContext, MechanismDescriptor, MechanismFamily,
    MechanismForm, MechanismOrigin, MechanismProvenance, MechanismUncertainty, ParameterId,
    ParameterTerm, ParentBinding, PosteriorDependence, SelectionDiagramDeclaration,
    SelectionNodeDeclaration, TabularRow, UncertaintyMeaning,
};
pub use gaussian::{
    ComposedLinearGaussian, ContrastArm, FunctionalUncertainty, JointGaussian, MeanContrast,
    SampleTable,
};
pub use tabular::{ComposedFinite, FiniteLaw};

/// Largest number of cells (rows times noise sources) one sampling call may
/// allocate.
pub const MAX_SAMPLE_CELLS: usize = 1 << 26;

/// Stream domain of composition sampling. Streams are keyed by the *name* of the
/// noise source, so a source's draws do not depend on how many other variables
/// the composition has or on their order.
pub const COMPOSITION_STREAM_DOMAIN: StreamDomain =
    StreamDomain::external(StreamDomain::EXTERNAL_BASE + 0x4D43);

/// Stream index of one named noise source of one role.
///
/// `role` separates idiosyncratic noise (0), latent causes (1) and finite
/// categorical uniforms (2) that share a name.
#[must_use]
pub fn noise_stream_index(role: u64, name: &str) -> u64 {
    let mut h = mix64(role.wrapping_add(0x6D65_6368));
    h = mix64(h ^ (name.len() as u64));
    for b in name.bytes() {
        h = mix64(h ^ u64::from(b));
    }
    h
}

/// Why a composition could not be compiled or executed.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum CompositionError {
    /// The declarations do not establish a coherent joint law (or, for an
    /// uncertainty claim, an aligned joint parameter law).
    #[error("mechanisms do not compose: {} finding(s)", .findings.len())]
    Incoherent {
        /// Every defect found, in fixed order.
        findings: Vec<CoherenceFinding>,
    },
    /// The request is outside the bounded family.
    #[error("unsupported: {message}")]
    Unsupported {
        /// What is outside the boundary.
        message: String,
    },
    /// Cancellation was observed at a work boundary.
    #[error("composition execution cancelled")]
    Cancelled,
    /// A numerical failure.
    #[error("numerical error: {message}")]
    Numerical {
        /// Context.
        message: String,
    },
}

impl CompositionError {
    /// The findings of an [`Self::Incoherent`] error, empty otherwise.
    #[must_use]
    pub fn findings(&self) -> &[CoherenceFinding] {
        match self {
            Self::Incoherent { findings } => findings,
            _ => &[],
        }
    }
}

pub(crate) fn incoherent(finding: CoherenceFinding) -> CompositionError {
    CompositionError::Incoherent { findings: vec![finding] }
}
