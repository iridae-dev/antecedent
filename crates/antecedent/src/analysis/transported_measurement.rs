//! A measurement model carried to another population under a declared invariance
//! premise, applied through the existing corrected-functional entry points.
//!
//! A [`MeasurementModel`] is established for one population: its sensitivity and
//! specificity, confusion matrix or error variance were learned on a validation sample
//! there, and the prepare checks of [`StudyBuilder::measurement_model`] refuse data from
//! any other population. Carrying the parameters is a transport claim. This module makes
//! it only when the caller declares the premise, and returns the corrected functional
//! together with the typed receipt that says how the model reached the target:
//!
//! 1. [`transport_measurement`] runs the library's transport decision
//!    ([`transport_measurement_model`]) for the population, regime and arm of the analysis
//!    binding. A model established for the target is used as is; otherwise a
//!    [`PopulationErrorStructure::Invariant`] declaration that names every error
//!    parameter the route uses re-binds the source model; a differential declaration
//!    never borrows another population's rates; no declaration refuses naming the premise.
//! 2. [`transport_measurement_estimate`] then prepares and estimates the corrected
//!    functional on the re-bound model through the existing facade entry
//!    ([`StudyBuilder::measurement_model`] and `PreparedMeasurementModel::estimate`), so
//!    every prepare check (construct, regime, arm, representation, assumed structure,
//!    route) still runs against the re-bound model.
//! 3. The result is a [`TransportedMeasurementEstimate`]: the estimate, the
//!    [`TransportReceipt`] and its identity. The corrected value is never presented as
//!    native to the target when a parameter was carried ([`TransportReceipt::carried`]).
//!
//! # What is and is not claimed
//!
//! The invariance premise is declared, never tested. Only the parameter estimation
//! uncertainty the correction already propagates is reported; the uncertainty of the
//! premise itself is not propagated, the standard error is conditional on the premise
//! and uncalibrated, and the interval route stays closed
//! ([`TransportedMeasurementEstimate::interval`]).
//!
//! # Export and consumption
//!
//! [`TransportedMeasurementEstimate::export`] writes the existing measurement-model
//! artifact of the re-bound model (the same bytes [`MeasurementEstimate::export`] writes,
//! consumed by `consume_measurement_model_artifact`). That artifact does not contain the
//! transport receipt. A consumer that holds the source models and the declaration
//! re-derives the receipt with [`verify_measurement_transport`], which recomputes the
//! transport and refuses when the identity it retained is not the recomputed one
//! (`measurement_model.transport_receipt_mismatch`); a receipt is never trusted from a
//! stored copy.
//!
//! # Refresh with outcome-only data
//!
//! A transported model is derived, never stored back. New validation data in the source
//! population is a new [`MeasurementModel`] with a new identity: call
//! [`transport_measurement_estimate`] again and the receipt and estimate are recomputed.
//! Outcome-only data from the target population carries no validation information; pass it
//! as a new [`MeasurementInput`] and it is corrected under the same receipt.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::measurement::LatentPosteriorTrust;
use antecedent_contract::{ExternalScientificObject, ExternalTrustState};
use antecedent_core::ExecutionContext;
use antecedent_core::measurement_model::{
    CorrectedFunctionalRecord, MeasurementInference, MeasurementModel, MeasurementRoute,
    ParameterUncertainty,
};
use antecedent_estimate::measurement::{DataBinding, MeasurementError};
pub use antecedent_estimate::measurement_transport::{
    ErrorParameter, InvarianceDeclaration, MEASUREMENT_INVARIANCE_ASSUMPTION,
    PopulationErrorStructure, TransportBasis, TransportReceipt, TransportedCorrection,
    TransportedMeasurementModel, transport_measurement_model,
};

// Differential error pooled across validated populations and carried to the target with the
// between-population heterogeneity propagated (the invariance premise's uncertainty).
pub use antecedent_estimate::measurement_transport_pooled::{
    PooledDifferentialCorrection, PooledDifferentialRequest, PooledRate, PopulationValidation,
    transport_pooled_differential_prevalence,
};

use super::StudyBuilder;
use super::composition_verification::{
    ClaimStanding, CompositionNode, NodeKind, TrustStanding, library_transported_node,
    library_transported_standing,
};
use super::measurement_model::{
    MeasurementAnalysisBinding, MeasurementEstimate, MeasurementInput, MeasurementModelRefusal,
};

/// Detail of a retained receipt identity that is not the recomputed one.
pub const DETAIL_TRANSPORT_RECEIPT_MISMATCH: &str = "measurement_model.transport_receipt_mismatch";

fn remedy_of(detail: &str) -> Option<&'static str> {
    Some(match detail {
        "measurement_model.invariance_premise_required" => {
            "declare that the error parameters are invariant from a named source population, \
             declare the error differential by population, or establish a validation sample in \
             the target population"
        }
        "measurement_model.invariance_not_declared" => {
            "name every error parameter the correction uses as invariant; a premise about one \
             parameter does not license a correction that uses another"
        }
        "measurement_model.invariance_target_mismatch" => {
            "declare the invariance between the source population and the analysis population"
        }
        "measurement_model.invariance_source_unknown" => {
            "name a source population that has an established measurement model"
        }
        "measurement_model.differential_population_unestablished" => {
            "establish a validation sample in the target population; another population's \
             rates are not borrowed under a differential structure"
        }
        "measurement_model.differential_axis_unsupported" => {
            "an error declared to depend on population or time is not carried by an \
             invariance premise; establish a model for the target"
        }
        "measurement_model.regime_mismatch" => {
            "the premise covers the population only; supply a model established for the \
             analysis regime"
        }
        "measurement_model.arm_mismatch" => {
            "the premise covers the population only; supply a model established for the \
             analysis arm"
        }
        "measurement_model.nothing_to_transport" => {
            "correct the target data with its own replicates instead of carrying parameters"
        }
        "measurement_model.posterior_bound_to_snapshot" => {
            "an external posterior is bound to its source snapshot; request a posterior for \
             the target population from the provider"
        }
        "measurement_model.no_model" => {
            "supply the measurement model established for a source population"
        }
        _ => return None,
    })
}

fn lift(error: &MeasurementError) -> MeasurementModelRefusal {
    MeasurementModelRefusal {
        code: error.code,
        detail: error.detail.to_owned(),
        message: error.message.clone(),
        remedy: remedy_of(error.detail).map(str::to_owned),
    }
}

fn data_binding(binding: &MeasurementAnalysisBinding) -> DataBinding {
    DataBinding {
        population_id: binding.population_id.clone(),
        regime_id: binding.regime_id.clone(),
        arm_id: binding.arm_id.clone(),
        representation: binding.representation.clone(),
    }
}

/// Make a measurement model usable on the population of `binding`.
///
/// `models` are the models established so far, each for its own population. See
/// [`transport_measurement_model`] for the order of refusal.
///
/// # Errors
/// A refusal naming the unmet premise (`measurement_model.invariance_premise_required`,
/// `.invariance_not_declared`, `.invariance_target_mismatch`, `.invariance_source_unknown`,
/// `.differential_population_unestablished`, `.regime_mismatch`, `.nothing_to_transport`,
/// ...); no model is issued.
pub fn transport_measurement(
    models: &[MeasurementModel],
    structure: Option<&PopulationErrorStructure>,
    binding: &MeasurementAnalysisBinding,
) -> Result<TransportedMeasurementModel, MeasurementModelRefusal> {
    transport_measurement_model(models, structure, &data_binding(binding))
        .map_err(|error| lift(&error))
}

/// Recompute the transport and require its identity to be the retained one.
///
/// The fresh-consumer path: the receipt is re-derived from the source models, the
/// declaration and the analysis binding, never read from a stored copy.
///
/// # Errors
/// Whatever [`transport_measurement`] refuses, and
/// `measurement_model.transport_receipt_mismatch` (`measurement_binding_mismatch`) when the
/// recomputed identity is not `expected_identity_hex`.
pub fn verify_measurement_transport(
    models: &[MeasurementModel],
    structure: Option<&PopulationErrorStructure>,
    binding: &MeasurementAnalysisBinding,
    expected_identity_hex: &str,
) -> Result<TransportReceipt, MeasurementModelRefusal> {
    let transported = transport_measurement(models, structure, binding)?;
    let receipt = transported.receipt().clone();
    if receipt.identity.to_hex() != expected_identity_hex {
        return Err(MeasurementModelRefusal {
            code: antecedent_core::reason_code!("measurement_binding_mismatch"),
            detail: DETAIL_TRANSPORT_RECEIPT_MISMATCH.to_owned(),
            message: format!(
                "the recomputed transport identity is {}, the retained one is {expected_identity_hex}",
                receipt.identity.to_hex()
            ),
            remedy: Some(
                "re-derive the transport from the source models and the declaration the \
                 consumer retained"
                    .to_owned(),
            ),
        });
    }
    Ok(receipt)
}

/// A corrected functional of a transported measurement model, with its receipt.
#[derive(Clone, Debug)]
pub struct TransportedMeasurementEstimate {
    estimate: MeasurementEstimate,
    receipt: TransportReceipt,
}

impl TransportedMeasurementEstimate {
    /// The corrected functional on the re-bound model.
    #[must_use]
    pub const fn estimate(&self) -> &MeasurementEstimate {
        &self.estimate
    }

    /// The transport receipt.
    #[must_use]
    pub const fn receipt(&self) -> &TransportReceipt {
        &self.receipt
    }

    /// Identity of the transport as lowercase hex: source model, re-bound model, basis.
    #[must_use]
    pub fn identity_hex(&self) -> String {
        self.receipt.identity.to_hex()
    }

    /// Whether any parameter was carried from another population.
    #[must_use]
    pub fn carried(&self) -> bool {
        self.receipt.carried()
    }

    /// Assumptions that travel with the corrected functional: the receipt's, then the
    /// route's.
    #[must_use]
    pub fn assumptions(&self) -> Vec<String> {
        self.receipt.assumptions()
    }

    /// The standing this corrected functional states as a claim in a composition
    /// graph.
    ///
    /// A corrected value is a point with an uncalibrated standard error at most:
    /// the interval route is closed ([`Self::interval`]), so no calibration
    /// record is claimed. Trust follows the provenance of the route: a
    /// model-based route rests on no provider (not applicable); an external
    /// latent posterior rests on its provider at the trust it was admitted at
    /// (an exact-request verification, or the supplier's attestation), and one
    /// with no stated trust is undeclared. Support is `extrapolative` when a
    /// parameter was carried from another population ([`Self::carried`]) and
    /// supported otherwise.
    #[must_use]
    pub fn claim_standing(&self) -> ClaimStanding {
        let mut standing = if self.carried() {
            library_transported_standing("extrapolative", &[])
        } else {
            ClaimStanding::native_point()
        };
        standing.trust = match (self.route(), self.provider_trust()) {
            (MeasurementRoute::ExternalLatentPosterior, Some(_)) => TrustStanding::Held,
            (MeasurementRoute::ExternalLatentPosterior, None) => TrustStanding::Undeclared,
            _ => TrustStanding::NotApplicable,
        };
        standing
    }

    /// The composition-graph node of this corrected functional under `key`,
    /// carrying [`Self::claim_standing`]. Its identity binds the transport
    /// receipt, the route and the corrected point, so a changed source model,
    /// declaration, binding or value recomputes it.
    #[must_use]
    pub fn composition_node(&self, key: &str) -> CompositionNode {
        let route = self.route().as_str();
        let identity = format!("{}:{}", self.identity_hex(), self.point().to_bits());
        let node = library_transported_node(
            key,
            route,
            &identity,
            &self.receipt.target_population,
            if self.carried() { "extrapolative" } else { "supported" },
            self.assumptions(),
        );
        let kind =
            if self.carried() { NodeKind::TransportedResult } else { NodeKind::MeasurementModel };
        CompositionNode { kind, ..node }.with_standing(self.claim_standing())
    }

    /// The corrected functional record.
    #[must_use]
    pub const fn record(&self) -> &CorrectedFunctionalRecord {
        self.estimate.record()
    }

    /// The corrected point value.
    #[must_use]
    pub const fn point(&self) -> f64 {
        self.estimate.point()
    }

    /// The delta-method standard error, conditional on the premise and uncalibrated.
    #[must_use]
    pub const fn standard_error(&self) -> Option<f64> {
        self.estimate.standard_error()
    }

    /// Always point-only or an uncalibrated standard error; never an interval.
    #[must_use]
    pub const fn inference(&self) -> MeasurementInference {
        self.estimate.inference()
    }

    /// Whether the measurement parameters' estimation uncertainty is in the variance. The
    /// invariance premise is never in it.
    #[must_use]
    pub const fn parameter_uncertainty(&self) -> ParameterUncertainty {
        self.estimate.parameter_uncertainty()
    }

    /// The route that produced the result.
    #[must_use]
    pub const fn route(&self) -> MeasurementRoute {
        self.estimate.route()
    }

    /// How far a provider posterior was trusted, for the posterior route.
    #[must_use]
    pub const fn provider_trust(&self) -> Option<LatentPosteriorTrust> {
        self.estimate.provider_trust()
    }

    /// An interval for the corrected functional.
    ///
    /// # Errors
    /// Always `cell_not_licensed`: no coverage record measures the whole
    /// measurement-plus-transport-plus-estimation method.
    pub fn interval(&self) -> Result<std::convert::Infallible, MeasurementModelRefusal> {
        self.estimate.interval()
    }

    /// Write the re-bound model and the result as the measurement-model artifact. The
    /// receipt is re-derived by the consumer ([`verify_measurement_transport`]).
    ///
    /// # Errors
    /// As [`MeasurementEstimate::export`].
    pub fn export(&self, artifact_id: &str) -> Result<Vec<u8>, MeasurementModelRefusal> {
        self.estimate.export(artifact_id)
    }
}

/// Transport a measurement model to the analysis population and form the corrected
/// functional on the re-bound model.
///
/// Order of refusal: cancellation, the transport decision ([`transport_measurement`]),
/// then every prepare check and estimator refusal of
/// [`StudyBuilder::measurement_model`] and `PreparedMeasurementModel::estimate` on the
/// re-bound model. Nothing is returned partially.
///
/// # Errors
/// A [`MeasurementModelRefusal`] naming the first unmet premise.
pub fn transport_measurement_estimate(
    models: &[MeasurementModel],
    structure: Option<&PopulationErrorStructure>,
    binding: MeasurementAnalysisBinding,
    input: MeasurementInput,
    provider: Option<(&ExternalScientificObject, &ExternalTrustState)>,
    ctx: &ExecutionContext,
) -> Result<TransportedMeasurementEstimate, MeasurementModelRefusal> {
    if ctx.cancellation.is_cancelled() {
        return Err(MeasurementModelRefusal {
            code: antecedent_core::reason_code!("cancelled_no_claim"),
            detail: "measurement_model.cancelled".to_owned(),
            message: "cancelled before a corrected functional was formed; this is not a \
                      verdict on the data"
                .to_owned(),
            remedy: None,
        });
    }
    let transported = transport_measurement(models, structure, &binding)?;
    let receipt = transported.receipt().clone();
    let prepared = StudyBuilder::measurement_model(
        transported.model().clone(),
        binding,
        input,
        provider,
        ctx,
    )?;
    let estimate = prepared.estimate(ctx)?;
    Ok(TransportedMeasurementEstimate { estimate, receipt })
}
