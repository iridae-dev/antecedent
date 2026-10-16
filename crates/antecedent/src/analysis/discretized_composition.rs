//! A composed linear-Gaussian model discretized into an executable finite model,
//! with a computed error bound: prepare, evaluate, refresh, export and consume.
//!
//! A [`PreparedMechanismComposition`] in the linear-Gaussian family has closed-form
//! interventional means. The finite counterfactual routes work on variables with
//! finitely many levels. [`PreparedMechanismComposition::discretize`] is the
//! bounded bridge between them: every variable is replaced by a declared number of
//! equal-probability cells (open outer cells, never truncated) and every mechanism
//! by the conditional table its discretized parents induce. The result carries,
//! beside the discretized value of a contrast, the closed-form Gaussian value at
//! the same intervention values and a *computed* bound on the error of the mean
//! (a guarantee for the mean of every variable and linear contrasts of means,
//! nothing else).
//!
//! # What a value means
//!
//! A discretized value is `point_only`. It carries no interval, posterior or
//! calibration ([`PreparedDiscretizedComposition::uncertainty`] refuses with
//! `cell_not_licensed`); parameter uncertainty declared on the source mechanisms is
//! not propagated through the discretization and the result says so. Exact
//! finite-law arithmetic supplies no calibration license. Levels are declared per
//! variable, never defaulted.
//!
//! # Lifecycle
//!
//! * **Prepare** ([`PreparedMechanismComposition::discretize`]) refuses a finite
//!   composition, a mechanism with a bounded support, a shared latent cause, an
//!   undeclared or out-of-range level count and a joint space above the
//!   enumeration bound, each with its registered code and detail.
//! * **Refresh.** [`PreparedDiscretizedComposition::refresh`] takes the refit
//!   composition (the result of
//!   [`PreparedMechanismComposition::refresh_mechanisms`]) and discretizes it
//!   again under the **same declared plan**: the declared level counts and method
//!   are **kept**; the cell edges, cell values, conditional tables, the error
//!   bound and the identity are **recomputed**; a refit whose variables no longer
//!   match the plan is **refused**, never patched. Level edges are never reused
//!   across a refit and refreshing the composition does not rebuild a bridge
//!   silently. The call takes `&self` and returns a new value, so a refused
//!   refresh leaves the prepared bridge unchanged.
//! * **Export / consume.** [`PreparedDiscretizedComposition::export`] writes the
//!   source premises, the plan and the claimed numbers;
//!   [`consume_discretized_composition`] rebuilds the discretization from the
//!   carried premises and requires every claimed number to equal the re-derived
//!   one bit for bit; a consumer that retained the identity independently refuses
//!   a self-consistent artifact of another composition.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use super::mechanism_exports::model_crate::discretized_scm::{
    ArmDiagnostic, BRIDGE_CALIBRATION, BRIDGE_ERROR_SCOPE, BRIDGE_INFERENCE, BridgeRefusal,
    DISCRETIZED_SCM_VERSION, DiscreteArm, DiscreteContrast, DiscretizationMethod,
    DiscretizationPlan, DiscretizedContrast, DiscretizedLinearGaussian, InferenceParadigm,
    MAX_BRIDGE_LEVELS, MIN_BRIDGE_LEVELS, PARITY_TABLE, ParadigmCell, ParityRoute, ParityRow,
    parity_row, require_paradigm,
};
pub use super::mechanism_exports::model_crate::discretized_scm_artifact::{
    ArtifactRefusal as DiscretizedCompositionArtifactRefusal, DISCRETIZED_SCM_ARTIFACT_VERSION,
};
use antecedent_core::ExecutionContext;

use antecedent_model::discretized_scm_artifact::{consume_discretized_scm, export_discretized_scm};

use super::mechanism_composition::{MechanismCompositionRefusal, PreparedMechanismComposition};

/// Why a composition was not discretized, refreshed or exported.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum DiscretizationRefusal {
    /// The composition has no closed-form Gaussian executable (a finite-table
    /// composition) or was cancelled.
    Composition(MechanismCompositionRefusal),
    /// The bridge refused the composition or the plan.
    Bridge(BridgeRefusal),
    /// Export or consumption refused.
    Artifact(DiscretizedCompositionArtifactRefusal),
}

impl DiscretizationRefusal {
    /// Registered reason code.
    #[must_use]
    pub fn code(&self) -> &'static str {
        match self {
            Self::Composition(e) => e.code(),
            Self::Bridge(e) => e.reason_code,
            Self::Artifact(e) => e.reason_code,
        }
    }

    /// Namespaced detail.
    #[must_use]
    pub fn detail(&self) -> String {
        match self {
            Self::Composition(e) => e.details().first().copied().unwrap_or_default().to_owned(),
            Self::Bridge(e) => e.detail.to_owned(),
            Self::Artifact(e) => e.detail.to_owned(),
        }
    }

    /// What would license the request, when the refusal names it.
    #[must_use]
    pub fn remedy(&self) -> Option<&'static str> {
        match self {
            Self::Composition(_) => None,
            Self::Bridge(e) => Some(e.remedy),
            Self::Artifact(e) => Some(e.remedy),
        }
    }
}

impl std::fmt::Display for DiscretizationRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}: {}", antecedent_core::reason_code::PREFIX, self.code(), self.detail())
    }
}

impl std::error::Error for DiscretizationRefusal {}

impl From<BridgeRefusal> for DiscretizationRefusal {
    fn from(value: BridgeRefusal) -> Self {
        Self::Bridge(value)
    }
}

/// What an inspection of a discretized composition shows.
#[derive(Clone, Debug, PartialEq)]
pub struct DiscretizedCompositionInspection {
    /// Always `point_only`.
    pub inference: &'static str,
    /// Always `no_interval_or_calibration_claim`.
    pub calibration: &'static str,
    /// What the error bound covers.
    pub error_scope: &'static str,
    /// The declared method.
    pub method: &'static str,
    /// `(variable, levels, largest mechanism mean error)` in canonical order.
    pub variables: Vec<(String, usize, f64)>,
    /// Identity of the discretization (hex).
    pub identity: String,
    /// Identity of the source composition's plan (hex) at discretization.
    pub source_plan_identity: String,
}

/// A composition discretized into an executable finite model.
#[derive(Clone, Debug)]
pub struct PreparedDiscretizedComposition {
    plan: DiscretizationPlan,
    bridge: DiscretizedLinearGaussian,
    source_plan_identity: String,
}

impl PreparedMechanismComposition {
    /// Discretize a linear-Gaussian composition under a declared plan.
    ///
    /// # Errors
    /// [`DiscretizationRefusal::Composition`] for a finite-table composition, and
    /// [`DiscretizationRefusal::Bridge`] for a level count missing, out of range or
    /// naming an unknown variable, a joint space above the enumeration bound, a
    /// declared support other than the real line, a mechanism loading on a shared
    /// latent cause, a degenerate marginal, or cancellation.
    pub fn discretize(
        &self,
        plan: &DiscretizationPlan,
        ctx: &ExecutionContext,
    ) -> Result<PreparedDiscretizedComposition, DiscretizationRefusal> {
        let model = self.gaussian().map_err(DiscretizationRefusal::Composition)?;
        let bridge = DiscretizedLinearGaussian::new(model, plan, ctx)?;
        Ok(PreparedDiscretizedComposition {
            plan: plan.clone(),
            bridge,
            source_plan_identity: self.plan_identity_hex(),
        })
    }
}

impl PreparedDiscretizedComposition {
    /// The declared plan.
    #[must_use]
    pub const fn plan(&self) -> &DiscretizationPlan {
        &self.plan
    }

    /// The discretized model, for the finite routes and per-variable grids.
    #[must_use]
    pub const fn bridge(&self) -> &DiscretizedLinearGaussian {
        &self.bridge
    }

    /// Identity of the discretization: source mechanisms, method and every level
    /// count (hex).
    #[must_use]
    pub fn identity_hex(&self) -> String {
        self.bridge.identity_hex()
    }

    /// Identity of the source composition's plan (hex) the bridge was built from.
    #[must_use]
    pub fn source_plan_identity(&self) -> &str {
        &self.source_plan_identity
    }

    /// Evaluate a contrast on the discretized model beside the closed-form
    /// Gaussian value at the same intervention values and the computed bound.
    ///
    /// # Errors
    /// An unknown outcome, variable or level, a repeated intervention or
    /// cancellation.
    pub fn mean_contrast(
        &self,
        contrast: &DiscreteContrast,
        ctx: &ExecutionContext,
    ) -> Result<DiscretizedContrast, DiscretizationRefusal> {
        Ok(self.bridge.mean_contrast(contrast, ctx)?)
    }

    /// An interval, band, posterior or law-valued reading of a discretized value.
    ///
    /// # Errors
    /// Always `cell_not_licensed` (`discretized_scm.point_only`): a discretized
    /// value is a point with a computed mean-error bound.
    pub fn uncertainty(
        &self,
        paradigm: InferenceParadigm,
    ) -> Result<std::convert::Infallible, DiscretizationRefusal> {
        Err(DiscretizationRefusal::Bridge(self.bridge.uncertainty_refusal(paradigm)))
    }

    /// Discretize the refit composition again under the declared plan; see the
    /// module documentation for what is kept, recomputed and refused.
    ///
    /// # Errors
    /// As [`PreparedMechanismComposition::discretize`].
    pub fn refresh(
        &self,
        refit: &PreparedMechanismComposition,
        ctx: &ExecutionContext,
    ) -> Result<Self, DiscretizationRefusal> {
        refit.discretize(&self.plan, ctx)
    }

    /// Serialize the premises, the plan and the claimed numbers.
    ///
    /// # Errors
    /// A source mechanism the format cannot carry.
    pub fn export(&self) -> Result<Vec<u8>, DiscretizationRefusal> {
        export_discretized_scm(&self.bridge).map_err(DiscretizationRefusal::Artifact)
    }

    /// Plain-data summary for inspection.
    #[must_use]
    pub fn inspect(&self) -> DiscretizedCompositionInspection {
        let variables = self
            .bridge
            .variables()
            .into_iter()
            .map(|name| {
                let levels = self.bridge.level_names(name).map_or(0, <[String]>::len);
                let error = self.bridge.mechanism_mean_error(name).unwrap_or(f64::NAN);
                (name.to_owned(), levels, error)
            })
            .collect();
        DiscretizedCompositionInspection {
            inference: BRIDGE_INFERENCE,
            calibration: BRIDGE_CALIBRATION,
            error_scope: BRIDGE_ERROR_SCOPE,
            method: self.bridge.method().as_str(),
            variables,
            identity: self.bridge.identity_hex(),
            source_plan_identity: self.source_plan_identity.clone(),
        }
    }
}

/// Consume an exported discretization: the model is rebuilt from the carried
/// premises with the current code and every claimed number must equal the
/// re-derived one.
///
/// `expected_identity` is the identity the consumer holds for the composition it
/// expects.
///
/// # Errors
/// Malformed bytes, another version, premises that no longer discretize, an
/// identity that differs from the claimed or expected one, or a claimed number
/// that differs from the re-derived one.
pub fn consume_discretized_composition(
    bytes: &[u8],
    expected_identity: Option<[u8; 32]>,
    ctx: &ExecutionContext,
) -> Result<DiscretizedLinearGaussian, DiscretizationRefusal> {
    consume_discretized_scm(bytes, expected_identity, ctx).map_err(DiscretizationRefusal::Artifact)
}
