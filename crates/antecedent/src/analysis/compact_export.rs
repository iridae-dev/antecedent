//! B4 compact runtime export: a coefficient vector, its full covariance, the finite basis, the
//! declared support of every input and a refusal mask, sealed under one identity and consumable
//! by an independent verifier that rejects out-of-support and masked queries and tampered
//! identities.
//!
//! Scope. The export evaluates a point prediction `f(x) = b' phi(x)` inside the declared support
//! and outside the refusal mask, and nothing else. The only uncertainty it reports is the
//! model-based standard error `sqrt(phi' V phi)` from the stored covariance, labelled
//! `model_based_sqrt_phi_v_phi` with calibration **unmeasured**: it is not an interval, not a
//! coverage claim, and carries no extrapolation or misspecification uncertainty.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_io::compact_export::{
    CALIBRATION_STATUS, COMPACT_EXPORT_ARTIFACT_KIND, COMPACT_EXPORT_VERSION, CompactExport,
    ExportInput, ExportLimits, ExportQuery, ExportRefusal, ExportScope, ExportSpec, InputSupport,
    MAX_POLYNOMIAL_DEGREE, MaskCondition, MaskRegion, POINT_SCOPE, PointWithSe, QueryValue,
    SE_BASIS, TermSpec, UNCERTAINTY_SCOPE,
};

use crate::error::CausalError;

impl From<ExportRefusal> for CausalError {
    fn from(error: ExportRefusal) -> Self {
        CausalError::Compile { message: error.to_string() }
    }
}
